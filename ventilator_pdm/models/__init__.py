"""Anomaly detector implementations for Elisa 800 telemetry.

Each submodule provides a self-contained detector family: statistical
baselines, direct-signal thresholds, Isolation Forest, One-Class SVM, and
a CNN-LSTM autoencoder. Detectors share a common fit/score interface so they
can be swapped via `pdm.registry` without caller changes.
"""
