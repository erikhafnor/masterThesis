"""Cypresses-styled Isolation Forest feature pipeline. Figure 4.1a.

Brief (derived from thesis/main.tex § Level 2: Isolation Forest, line 348ff,
informed by the 2026-05-12 research report on iForest / PdM architecture
figure conventions):

    Figure number:   4.1a (\\ref{fig:iforest_pipeline})
    Chapter:         4 — Methods
    Title:           Isolation Forest feature pipeline
    Output format:   SVG
    Composition:     5-node linear horizontal pipeline, identical layout
                     pattern to Fig 4.4. Sage bookend at the start (raw
                     window), olive-yellow ranking node at the end (the
                     iForest model, which produces a *ranked* anomaly
                     score across the fleet).
    Stages:
      1  30-min window (input)        46 features × 30 rows raw telemetry
      2  Summary statistics (process) mean · std · min · max · delta
      3  Feature vector (process)     230 dims  (5 stats × 46 features)
      4  StandardScaler (process)     zero mean · unit variance
      5  Isolation Forest (ranking)   200 trees · contamination 0.01
    Edges: all bare arrows — node names carry the flow.

Source canvas: 940 × 240 user units. At \\textwidth = 6.3 in the scale is
~0.483; node-label 22 → 10.6 pt, body 18 → 8.7 pt, mono 16 → 7.7 pt
(matches Fig 4.4 / Fig 3.1 precedent).

The score formula s(x) = 2^(-E[h(x)]/c(n)) is intentionally NOT in this
figure — it lives as a numbered equation in the surrounding prose
(research recommendation: thesis methods chapters separate the diagram
from the formula). The hyperparameter footer of the original PNG is
likewise dropped (already in prose).
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
W, H = 940, 240
NODE_CY = 120
NODE_W = 160
NODE_H = 180
NODE_XS = [40, 220, 400, 580, 760]


def _card(*, x, y, w, h, role):
    fill = FILL_MIX.get(role, COLOR["canvas_cream"])
    stroke = ROLE[role]
    sw = STROKE_WIDTH[role]
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'rx="4" ry="4" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def _label_set(*, x, y, w, label, body_lines=(), mono_lines=()):
    # All sizes ≥ 23 px → rendered ≥ 11 pt at W=940, textwidth=6.3 in.
    out = []
    label_lines = (label,) if isinstance(label, str) else tuple(label)
    label_y = y + 36
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
    for line in mono_lines:
        out.append(
            f'<text x="{x + w / 2}" y="{cur_y}" text-anchor="middle" '
            f'font-family="JetBrains Mono, monospace" font-size="23" '
            f'fill="{COLOR["olive_shadow"]}">{line}</text>'
        )
        cur_y += 26
    return "".join(out)


def _node(x, role, label, body=(), mono=()):
    y = NODE_CY - NODE_H / 2
    return _card(x=x, y=y, w=NODE_W, h=NODE_H, role=role) + _label_set(
        x=x, y=y, w=NODE_W, label=label, body_lines=body, mono_lines=mono
    )


def build_figure():
    parts = []
    parts.append(_node(NODE_XS[0], "input",
                       ("30-min", "window"),
                       body=("46 features", "× 30 rows"),
                       mono=("raw telemetry",)))
    parts.append(_node(NODE_XS[1], "process",
                       "Summary stats",
                       body=("mean · std · min", "max · delta"),
                       mono=("per feature",)))
    parts.append(_node(NODE_XS[2], "process",
                       "Feature vector",
                       body=("230 dims",),
                       mono=("5 stats", "× 46 vars")))
    parts.append(_node(NODE_XS[3], "process",
                       ("Standard", "Scaler"),
                       body=("zero mean,", "unit variance")))
    parts.append(_node(NODE_XS[4], "ranking",
                       ("Isolation", "Forest"),
                       body=("200 trees", "contamination", "= 0.01")))

    def right_x(i):
        return NODE_XS[i] + NODE_W

    def left_x(i):
        return NODE_XS[i]

    # All bare arrows; final edge is "strong" cypress-deep to mark the
    # transition into the model that produces the scored output.
    for i in range(4):
        style = "strong" if i == 3 else "default"
        parts.append(svg_edge(
            x1=right_x(i) + 4, y1=NODE_CY,
            x2=left_x(i + 1) - 4, y2=NODE_CY,
            style=style,
        ))

    return "".join(parts)


def main():
    body = build_figure()
    svg = svg_document(width=W, height=H, body=body)
    out = Path(__file__).with_suffix(".svg")
    out.write_text(svg, encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
