"""Cypresses-styled CNN-LSTM autoencoder architecture. Figure 4.3.

Brief (derived from thesis/main.tex § Level 3: CNN-LSTM Autoencoder,
line 375 ff., and from the model definition in
``pdm/models/autoencoder.py`` — every layer's dims and hyperparameters
are sourced from that code, not from the matplotlib PNG (which had a
wrong batch size of 64 instead of the prose's correct 2048):

    Figure number:   4.3 (\\ref{fig:ae_arch})
    Chapter:         4 — Methods
    Title:           CNN-LSTM autoencoder architecture
    Output format:   SVG, single figure float (no sub-figures)
    Composition:     9-block horizontal linear flow with grouped eyebrow
                     labels (ENCODER / BOTTLENECK / DECODER). All blocks
                     same width (88 u); the Latent block is a non-compute
                     visual marker between LSTM enc and LSTM dec, making
                     the bottleneck legible. Tensor shapes go BELOW each
                     block in mono — modern time-series-AE convention
                     (Tran et al. IEEE Access 2024; Homayouni et al.
                     IEEE BigData; Maciag et al. Sensors 25(5):1610 2025).
                     U-shape is a U-Net image-segmentation convention,
                     not used here.
    Stages (left to right):
      01 Input        input    (30, 46)   46 feat · T = 30
      02 Conv1D #1    process  (30, 64)   64 ch · k=3 p=1 · +ReLU
      03 Conv1D #2    process  (30, 64)   64 ch · k=3 p=1 · +ReLU
      04 LSTM (enc)   process  (30, 32)   h = 32
      05 Latent       process  (30, 32)   (no body — pure bottleneck marker)
      06 LSTM (dec)   process  (30, 64)   h = 64
      07 ConvT1D #1   process  (30, 64)   64 ch · k=3 p=1 · +ReLU
      08 ConvT1D #2   process  (30, 46)   46 ch · k=3 p=1 · no act
      09 Output       input    (30, 46)   46 feat · T = 30
    Edges: bare forest arrows between adjacent blocks.

Drops vs the original matplotlib version (intentional, all justified by
the 2026-05-12 research pass):
  - Mirror-row layout         → flatten to one horizontal flow
  - MSE loss + score formula  → numbered equation in prose
  - Training Configuration    → already in prose
  - Dashed latent-bottleneck arrow → Latent block IS the bottleneck

Source canvas: 940 x 280 user units. At \\textwidth scale ~0.483:
  - label font 22 → 10.6 pt (above §3 10 pt label floor)
  - body  font 18 →  8.7 pt (at §3 9 pt body floor, accepted)
  - mono  font 14 →  6.8 pt (below §3 8.5 pt floor — documented
                              deviation, same as Fig 3.2 specimen).

Text-fit pre-flight (per /figure-illustration skill step 7) — every
string verified against its container before render:
  - longest body line "k=3 p=1" (7 char × 9.5 = 67 u) in 76 u usable → ok
  - longest mono line "(30, 46)" (8 char × 8.4 = 67 u) in 88 u block → ok
  - longest label "ConvT1D" would be 84 u at 22 pt → split to ("ConvT","1D")
  - eyebrow "BOTTLENECK" (10 char × 10 u = 100 u) over a 88 u block →
    placed at the centre of the latent block; renders slightly wider
    than the block, which is fine for a group label.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from thesis.tokens.cypresses import (
    COLOR, FILL_MIX, ROLE, STROKE_WIDTH,
    svg_edge, svg_document,
)


# ── Canvas ─────────────────────────────────────────────────────────────
W, H = 940, 280

# Layer positions (rectilinear): all 9 blocks same height/width, with
# slightly larger gutters either side of the bottleneck for visual
# breathing room.
BLOCK_W = 88
BLOCK_H = 130
BLOCK_TOP = 78
GUTTER = 12               # between adjacent encoder / decoder layers
LATENT_GUTTER = 18        # either side of the Latent bottleneck

# Compute x positions (centres = x + BLOCK_W / 2).
def _compute_xs():
    xs = []
    x = 20                # left margin
    xs.append(x)
    # 01 → 02 → 03 → 04 (encoder)
    for _ in range(3):
        x += BLOCK_W + GUTTER
        xs.append(x)
    # 04 → 05 (LSTM enc → Latent) wider gutter
    x += BLOCK_W + LATENT_GUTTER
    xs.append(x)
    # 05 → 06 (Latent → LSTM dec) wider gutter
    x += BLOCK_W + LATENT_GUTTER
    xs.append(x)
    # 06 → 07 → 08 → 09 (decoder)
    for _ in range(3):
        x += BLOCK_W + GUTTER
        xs.append(x)
    return xs


BLOCK_XS = _compute_xs()
# Sanity: should be 9 entries; right edge of block 9 should be ≤ W - 20.
assert len(BLOCK_XS) == 9
assert BLOCK_XS[-1] + BLOCK_W <= W - 18, (
    f"Block 9 right edge {BLOCK_XS[-1] + BLOCK_W} exceeds canvas margin"
)


def _card(*, x, y, w, h, role):
    fill = FILL_MIX.get(role, COLOR["canvas_cream"])
    stroke = ROLE[role]
    sw = STROKE_WIDTH[role]
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'rx="4" ry="4" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def _label_set(*, x, y, w, label, body_lines=()):
    """Bold node label (optionally multi-line) + Inter body lines.
    All font sizes ≥ 23 px → rendered ≥ 11 pt at W=940, textwidth=6.3 in.
    """
    out = []
    label_lines = (label,) if isinstance(label, str) else tuple(label)
    label_y = y + 28
    for line in label_lines:
        out.append(
            f'<text x="{x + w / 2}" y="{label_y}" text-anchor="middle" '
            f'font-family="Inter, sans-serif" font-size="23" font-weight="600" '
            f'fill="{COLOR["cypress_deep"]}">{line}</text>'
        )
        label_y += 26
    cur_y = label_y + 4
    for line in body_lines:
        out.append(
            f'<text x="{x + w / 2}" y="{cur_y}" text-anchor="middle" '
            f'font-family="Inter, sans-serif" font-size="23" font-weight="400" '
            f'fill="{COLOR["forest"]}">{line}</text>'
        )
        cur_y += 26
    return "".join(out)


def _node(idx, role, label, body=()):
    x = BLOCK_XS[idx]
    return _card(x=x, y=BLOCK_TOP, w=BLOCK_W, h=BLOCK_H, role=role) + \
           _label_set(x=x, y=BLOCK_TOP, w=BLOCK_W, label=label, body_lines=body)


def _shape_annotation(idx, shape_text):
    """Mono tensor-shape annotation centred below block ``idx``.
    font-size=23 → rendered ≥ 11 pt at W=940, textwidth=6.3 in.
    """
    x = BLOCK_XS[idx] + BLOCK_W / 2
    y = BLOCK_TOP + BLOCK_H + 28
    return (
        f'<text x="{x}" y="{y}" text-anchor="middle" '
        f'font-family="JetBrains Mono, monospace" font-size="23" '
        f'fill="{COLOR["olive_shadow"]}">{shape_text}</text>'
    )


def _group_eyebrow(*, cx, text):
    """Small-caps eyebrow group label centred above a span of blocks.
    font-size=23 → rendered ≥ 11 pt.
    """
    return (
        f'<text x="{cx}" y="50" text-anchor="middle" '
        f'font-family="JetBrains Mono, monospace" font-size="23" '
        f'letter-spacing="2.0" font-weight="600" '
        f'fill="{COLOR["olive_shadow"]}">{text.upper()}</text>'
    )


def build_figure():
    parts = []

    # ── Group eyebrows ────────────────────────────────────────────────
    # Encoder spans blocks 0..3 (Input through LSTM enc).
    encoder_cx = (BLOCK_XS[0] + BLOCK_XS[3] + BLOCK_W) / 2
    bottleneck_cx = BLOCK_XS[4] + BLOCK_W / 2
    decoder_cx = (BLOCK_XS[5] + BLOCK_XS[8] + BLOCK_W) / 2
    parts.append(_group_eyebrow(cx=encoder_cx, text="Encoder"))
    parts.append(_group_eyebrow(cx=bottleneck_cx, text="Bottleneck"))
    parts.append(_group_eyebrow(cx=decoder_cx, text="Decoder"))

    # ── Blocks ────────────────────────────────────────────────────────
    parts.append(_node(0, "input",   "Input",
                       body=("46 feat", "T = 30")))
    parts.append(_node(1, "process", "Conv1D",
                       body=("64 ch", "k=3 p=1", "+ ReLU")))
    parts.append(_node(2, "process", "Conv1D",
                       body=("64 ch", "k=3 p=1", "+ ReLU")))
    parts.append(_node(3, "process", "LSTM",
                       body=("h = 32",)))
    parts.append(_node(4, "process", "Latent",
                       body=()))
    parts.append(_node(5, "process", "LSTM",
                       body=("h = 64",)))
    parts.append(_node(6, "process", ("ConvT", "1D"),
                       body=("64 ch", "k=3 p=1", "+ ReLU")))
    parts.append(_node(7, "process", ("ConvT", "1D"),
                       body=("46 ch", "k=3 p=1", "no act")))
    parts.append(_node(8, "input",   "Output",
                       body=("46 feat", "T = 30")))

    # ── Shape annotations ────────────────────────────────────────────
    shapes = [
        "(30, 46)",   # Input
        "(30, 64)",   # Conv1D #1
        "(30, 64)",   # Conv1D #2
        "(30, 32)",   # LSTM enc
        "(30, 32)",   # Latent
        "(30, 64)",   # LSTM dec
        "(30, 64)",   # ConvT1D #1
        "(30, 46)",   # ConvT1D #2
        "(30, 46)",   # Output
    ]
    for i, s in enumerate(shapes):
        parts.append(_shape_annotation(i, s))

    # ── Edges (bare arrows) ──────────────────────────────────────────
    edge_y = BLOCK_TOP + BLOCK_H / 2
    for i in range(8):
        x1 = BLOCK_XS[i] + BLOCK_W + 2
        x2 = BLOCK_XS[i + 1] - 2
        parts.append(svg_edge(x1=x1, y1=edge_y, x2=x2, y2=edge_y))

    return "".join(parts)


def main():
    body = build_figure()
    svg = svg_document(width=W, height=H, body=body)
    out = Path(__file__).with_suffix(".svg")
    out.write_text(svg, encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
