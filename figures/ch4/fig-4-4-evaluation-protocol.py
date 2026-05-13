"""Cypresses-styled evaluation-protocol figure (Figure 4.4b).

Brief (derived from thesis/main.tex § Evaluation Framework, informed by
the 2026-05-12 research report on PdM methodology-figure conventions):

    Figure number:   4.4b
    Chapter:         4 — Methods / Evaluation
    Title:           Evaluation protocol
    Output format:   SVG
    Composition:     4 numbered cards in a horizontal row, no arrows
                     between them. Per the research synthesis, protocol
                     steps are a rhetorical sequence (numbered checklist)
                     rather than a data-flow pipeline — using arrows here
                     would mislead the reader into thinking each step
                     produces input for the next.
    Cards (left → right):
      1  Ground truth — 3 known O2 sensor failures, CMMS-logged,
                        observation window [-14 d, +2 d]
      2  Detection lead time — days before failure of first sustained alert
      3  Per-feature attribution — SHAP / IForest feature importance
      4  Model comparison — IForest vs CNN-LSTM autoencoder

Source canvas: 940 × 240 user units. At width=\\textwidth the scale is
~0.483; source font 22 → 10.6 pt rendered, 18 → 8.7 pt rendered, mono
badge 30 → 14.5 pt rendered.

All four cards use the `evaluation` role (turquoise stroke + 20 %
turquoise-over-cream fill) per DESIGN.md §7.2.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from thesis.tokens.cypresses import (
    COLOR, FILL_MIX, ROLE, STROKE_WIDTH, svg_document,
)

W, H = 940, 240
CARD_W = 200
CARD_H = 180
CARD_Y = 30
CARD_XS = [40, 260, 480, 700]


def _evaluation_card(*, x, y, w, h, number, label, body_lines):
    """One numbered evaluation-protocol card."""
    role = "evaluation"
    fill = FILL_MIX[role]
    stroke = ROLE[role]
    sw = STROKE_WIDTH[role]

    out = []
    # Card.
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'rx="6" ry="6" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )
    # Number badge in mono at top-left, in olive-shadow.
    # 30 px → rendered 14.5 pt (PASS). Label and body bumped to 23 px → 11.1 pt.
    out.append(
        f'<text x="{x + 16}" y="{y + 40}" '
        f'font-family="JetBrains Mono, monospace" font-size="30" '
        f'font-weight="500" fill="{COLOR["olive_shadow"]}">{number:02d}</text>'
    )
    # Bold label below the number.
    out.append(
        f'<text x="{x + w / 2}" y="{y + 80}" text-anchor="middle" '
        f'font-family="Inter, sans-serif" font-size="23" font-weight="600" '
        f'fill="{COLOR["cypress_deep"]}">{label}</text>'
    )
    # Body lines (centred block inside the card).
    body_y0 = y + 110
    for i, line in enumerate(body_lines):
        out.append(
            f'<text x="{x + w / 2}" y="{body_y0 + i * 26}" text-anchor="middle" '
            f'font-family="Inter, sans-serif" font-size="23" font-weight="400" '
            f'fill="{COLOR["forest"]}">{line}</text>'
        )
    return "".join(out)


def build_protocol():
    parts = []
    cards = [
        (1, "Ground truth", ["3 O2 sensor failures", "CMMS-logged",
                              "window [-14 d, +2 d]"]),
        (2, "Lead time", ["Days before failure", "of first sustained alert"]),
        (3, "Attribution", ["Which features drive", "the anomaly score?",
                             "(SHAP / per-feature)"]),
        (4, "Model comparison", ["IForest vs",
                                  "CNN-LSTM AE",
                                  "Detection + lead time"]),
    ]
    for x, (number, label, body) in zip(CARD_XS, cards):
        parts.append(_evaluation_card(
            x=x, y=CARD_Y, w=CARD_W, h=CARD_H,
            number=number, label=label, body_lines=body,
        ))
    return "".join(parts)


def main():
    body = build_protocol()
    svg = svg_document(width=W, height=H, body=body)
    out = Path(__file__).with_suffix(".svg")
    out.write_text(svg, encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
