"""Level 3: Reconstruction autoencoder for anomaly detection.

CNN-LSTM architecture on curated telemetry features. Trains on fleet-wide
healthy windows; anomaly scores are per-window reconstruction MSE. Designed
to run on GPU (SLURM cluster) but falls back to CPU when CUDA is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch required for autoencoder. Install with: pip install torch")


# ── CNN-LSTM Autoencoder ────────────────────────────────────────────────

class CNNLSTMEncoder(nn.Module):
    """Convolutional + LSTM encoder for the CNN-LSTM autoencoder.

    Applies two 1-D convolutional layers to capture local temporal patterns,
    then passes the result through an LSTM to produce a sequence of latent
    representations.

    Attributes:
        conv1: First ``Conv1d`` layer mapping ``n_features`` to ``hidden_dim``.
        conv2: Second ``Conv1d`` layer maintaining ``hidden_dim`` channels.
        lstm: LSTM mapping ``hidden_dim`` to ``latent_dim`` over the sequence.
    """

    def __init__(self, n_features: int, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, latent_dim, batch_first=True)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (batch, seq_len, hidden)
        out, (h, c) = self.lstm(x)
        return out, h


class CNNLSTMDecoder(nn.Module):
    """LSTM + transposed-convolutional decoder for the CNN-LSTM autoencoder.

    Inverts the encoder by running an LSTM over the latent sequence, then
    applying two transposed 1-D convolutional layers to reconstruct the
    original feature sequence.

    Attributes:
        lstm: LSTM mapping ``latent_dim`` to ``hidden_dim`` over the sequence.
        conv1: First ``ConvTranspose1d`` layer maintaining ``hidden_dim`` channels.
        conv2: Final ``ConvTranspose1d`` layer mapping back to ``n_features``.
    """

    def __init__(self, n_features: int, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.conv1 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.ConvTranspose1d(hidden_dim, n_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x


class CNNLSTMAutoencoder(nn.Module):
    """Full CNN-LSTM sequence-to-sequence autoencoder.

    Composes `CNNLSTMEncoder` and `CNNLSTMDecoder` into a single
    ``nn.Module``. The output shape matches the input shape, and the
    reconstruction error is used as the anomaly score.

    Attributes:
        encoder: The `CNNLSTMEncoder` submodule.
        decoder: The `CNNLSTMDecoder` submodule.
    """

    def __init__(self, n_features: int, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.encoder = CNNLSTMEncoder(n_features, hidden_dim, latent_dim)
        self.decoder = CNNLSTMDecoder(n_features, hidden_dim, latent_dim)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ── Training & Scoring ──────────────────────────────────────────────────

class AutoencoderModel:
    """CNN-LSTM autoencoder trained to reconstruct healthy telemetry windows.

    Anomaly scores are per-window mean squared reconstruction error. The model
    is fitted on fleet-wide healthy windows and scored per device per day.
    Requires PyTorch; raises `ImportError` on construction if PyTorch is not
    installed.

    Attributes:
        model: The underlying `CNNLSTMAutoencoder` ``nn.Module``.
        device: The ``torch.device`` used for training and inference.
        n_features: Number of input feature channels.
        batch_size: Mini-batch size used during training and scoring.
        epochs: Number of training epochs.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 50,
        device: str | None = None,
    ):
        _check_torch()
        self.n_features = n_features
        self.batch_size = batch_size
        self.epochs = epochs

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = CNNLSTMAutoencoder(n_features, hidden_dim, latent_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction="none")
        self._fitted = False

    def fit(
        self,
        windows: np.ndarray,
        val_windows: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Train the autoencoder on healthy telemetry windows.

        NaN values are replaced with 0.0 before training. Logs per-epoch
        loss to the module logger.

        Args:
            windows: Healthy training windows, shape (n_windows, seq_len,
                n_features).
            val_windows: Optional validation windows of the same shape. When
                provided, validation loss is computed each epoch.

        Returns:
            Dictionary with keys ``"train"`` and ``"val"``, each mapping to a
            list of per-epoch mean MSE losses. ``"val"`` is an empty list when
            no validation windows are supplied.
        """
        X = np.nan_to_num(windows, nan=0.0).astype(np.float32)
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if val_windows is not None:
            X_val = np.nan_to_num(val_windows, nan=0.0).astype(np.float32)
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_val)),
                batch_size=self.batch_size, shuffle=False,
            )

        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            n_batches = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                recon = self.model(batch)
                loss = self.criterion(recon, batch).mean()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_train = total_loss / max(n_batches, 1)
            train_losses.append(avg_train)

            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    val_total, val_n = 0.0, 0
                    for (vbatch,) in val_loader:
                        vbatch = vbatch.to(self.device)
                        vrecon = self.model(vbatch)
                        val_total += self.criterion(vrecon, vbatch).mean().item()
                        val_n += 1
                avg_val = val_total / max(val_n, 1)
                val_losses.append(avg_val)
                logger.info(
                    "Epoch %d/%d — train: %.4f  val: %.4f",
                    epoch + 1, self.epochs, avg_train, avg_val,
                )
            else:
                logger.info("Epoch %d/%d — train: %.4f", epoch + 1, self.epochs, avg_train)

        self._fitted = True
        return {"train": train_losses, "val": val_losses}

    def score(self, windows: np.ndarray) -> np.ndarray:
        """Compute per-window reconstruction MSE anomaly scores.

        Args:
            windows: Windows to score, shape (n_windows, seq_len, n_features).
                NaN values are replaced with 0.0.

        Returns:
            1-D float array of length n_windows containing mean squared
            reconstruction error per window. Higher values indicate greater
            anomaly likelihood.

        Raises:
            RuntimeError: If called before `fit`.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        X = np.nan_to_num(windows, nan=0.0).astype(np.float32)
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        scores = []

        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                recon = self.model(batch)
                mse = self.criterion(recon, batch).mean(dim=(1, 2))
                scores.append(mse.cpu().numpy())

        return np.concatenate(scores)

    def score_per_feature(self, windows: np.ndarray) -> np.ndarray:
        """Compute reconstruction MSE broken down by feature channel.

        Useful for explainability: identifies which feature channels contributed
        most to a high anomaly score.

        Args:
            windows: Windows to score, shape (n_windows, seq_len, n_features).
                NaN values are replaced with 0.0. The full tensor is moved to
                ``self.device`` in one shot, so memory limits apply.

        Returns:
            Float array of shape (n_windows, n_features) containing mean
            squared reconstruction error per window per feature channel.

        Raises:
            RuntimeError: If called before `fit`.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        X = np.nan_to_num(windows, nan=0.0).astype(np.float32)
        tensor = torch.from_numpy(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            recon = self.model(tensor)
            per_feature = self.criterion(recon, tensor).mean(dim=1)

        return per_feature.cpu().numpy()

    def save(self, path: Path) -> None:
        """Serialize the model weights and architecture parameters to disk.

        Args:
            path: Destination file path. Parent directories are created if
                they do not exist. Serialization uses ``torch.save``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "n_features": self.n_features,
        }, path)
        logger.info("Saved autoencoder to %s", path)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "AutoencoderModel":
        """Deserialize a previously saved model from disk.

        Args:
            path: Path to a file produced by `save`.
            device: Device string (e.g. ``"cuda"``, ``"cpu"``) to load the
                model onto. Defaults to ``None``, which uses the same
                auto-detect logic as ``__init__``.

        Returns:
            A fully restored `AutoencoderModel` instance ready for scoring.
        """
        _check_torch()
        data = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(n_features=data["n_features"], device=device)
        obj.model.load_state_dict(data["model_state"])
        obj._fitted = True
        logger.info("Loaded autoencoder from %s", path)
        return obj
