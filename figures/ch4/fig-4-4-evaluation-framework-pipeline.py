"""Cypresses-styled scoring pipeline for Figure 4.4a.

Brief (derived from thesis/main.tex § Evaluation Framework, line 397 ff.,
informed by the 2026-05-12 research report on PdM evaluation-figure
conventions):

    Figure number:   4.4a
    Chapter:         4 — Methods / Evaluation
    Title:           Fleet-Percentile Scoring Pipeline
    Output format:   SVG
    Composition:     linear, 5-node horizontal pipeline (no swim-lane).
                     Linear pipelines are the dominant convention in 2024-2026
                     PdM / anomaly-detection methodology figures (Sensors,
                     EAAI, MDPI Systems, T&F IJCIM). Swim-lanes read as
                     BPMN engineering documentation and are uncommon in
                     ML methodology figures (DESIGN.md v0.2 §15c).
    Stages:
      1  Fleet (input)        — small cluster grid with D7 anomalous
      2  Model score (process) — IForest / CNN-LSTM AE per device·window
      3  Fleet percentile (ranking) — rank vs the 19 active fleet devices at time t
      4  Sustained check       — inline mini-waveform showing 3 consecutive
                                   crossings; replaces the BPMN decision diamond
      5  Alert (alert)         — NORMAL / WARNING (>95th) / CRITICAL (>99th)
    Edges:
      fleet → score          : default
      score → percentile     : default, label "raw scores"
      percentile → check     : default, label "percentile"
      check → alert          : strong, label "yes"
    Highlights: D7 in the fleet cluster (anomalous device).

Source canvas: 900 × 460 user units. At \\textwidth = 6.3 in the scale is
~0.50; source font sizes are picked so rendered text hits the §3 print
floors (node label 22 → 11 pt, node body 18 → 9 pt, eyebrow 17 → 8.5 pt).

DEVIATION FROM DESIGN.md (recorded in v0.2 amendment):
- Stroke widths bumped from 1.25 / 1.75 baseline to 1.6 / 2.0+ per
  STROKE_WIDTH in cypresses.py, so role hue carries through stroke
  weight when fills stay at 18-24 % over canvas-cream.
- The "decision diamond" of DESIGN.md §7.5 is replaced by a sustained-
  exceedance mini-waveform — this is the convention in PdM / sliding-
  window detection figures, and is more pedagogically informative.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from thesis.tokens.cypresses import (
    COLOR, FILL_MIX, ROLE, STROKE_WIDTH,
    svg_node, svg_edge, svg_document, svg_sustained_waveform,
)

# ── Canvas geometry ────────────────────────────────────────────────────
W, H = 940, 240

# Nodes share a baseline y-centre.
NODE_CY = 120
NODE_W = 150
NODE_H = 180

# 5 nodes, evenly spaced. Stage-4 (sustained check) is slightly wider
# (160 px) to host the mini-waveform; subsequent x-offset corrects.
NODE_XS = [40, 210, 380, 545, 735]


def _node_label(x, y, w, label, body_lines=()):
    """Render a card-style node label set: bold title + 1–2 body lines.
    All sizes ≥ 23 px → rendered ≥ 11 pt at W=940, textwidth=6.3 in.
    """
    out = []
    label_y = y + 28
    out.append(
        f'<text x="{x + w / 2}" y="{label_y}" text-anchor="middle" '
        f'font-family="Inter, sans-serif" font-size="23" font-weight="600" '
        f'fill="{COLOR["cypress_deep"]}">{label}</text>'
    )
    for i, line in enumerate(body_lines):
        out.append(
            f'<text x="{x + w / 2}" y="{label_y + 30 + i * 26}" '
            f'text-anchor="middle" '
            f'font-family="Inter, sans-serif" font-size="23" font-weight="400" '
            f'fill="{COLOR["forest"]}">{line}</text>'
        )
    return "".join(out)


def _node_card(x, y, w, h, role):
    """Render the rect for a card-style node (no internal text)."""
    fill = FILL_MIX.get(role, COLOR["canvas_cream"])
    stroke = ROLE[role]
    sw = STROKE_WIDTH[role]
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'rx="6" ry="6" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def _fleet_cluster(*, x, y, n_show=8, anomaly_idx=6, cols=4,
                   cell=22, gap=4):
    """Compact cluster grid inside the Fleet node."""
    out = []
    sage = ROLE["input"]
    fill = FILL_MIX["input"]
    for i in range(n_show):
        col = i % cols
        row = i // cols
        cx = x + col * (cell + gap)
        cy = y + row * (cell + gap)
        is_anomaly = (i == anomaly_idx)
        stroke = COLOR["cypress_deep"] if is_anomaly else sage
        sw = 2.0 if is_anomaly else 1.2
        out.append(
            f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" '
            f'rx="2" ry="2" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )
        if is_anomaly:
            out.append(
                f'<rect x="{cx - 3}" y="{cy - 3}" width="{cell + 6}" height="{cell + 6}" '
                f'rx="4" ry="4" fill="none" stroke="{COLOR["cypress_deep"]}" '
                f'stroke-width="0.9" stroke-dasharray="2,2"/>'
            )
    return "".join(out)


def build_pipeline():
    parts = []

    # Stage 1: Fleet input.
    fx, fy = NODE_XS[0], NODE_CY - NODE_H / 2
    parts.append(_node_card(fx, fy, NODE_W, NODE_H, "input"))
    parts.append(
        f'<text x="{fx + NODE_W / 2}" y="{fy + 28}" text-anchor="middle" '
        f'font-family="Inter, sans-serif" font-size="23" font-weight="600" '
        f'fill="{COLOR["cypress_deep"]}">Fleet</text>'
    )
    # Cluster sits below the label, centred.
    cluster_w = 4 * 22 + 3 * 4  # cell=22, gap=4, cols=4 → 100 wide
    parts.append(_fleet_cluster(x=fx + (NODE_W - cluster_w) / 2, y=fy + 50))
    parts.append(
        f'<text x="{fx + NODE_W / 2}" y="{fy + NODE_H - 12}" text-anchor="middle" '
        f'font-family="JetBrains Mono, monospace" font-size="23" '
        f'fill="{COLOR["olive_shadow"]}">19 active devices</text>'
    )

    # Stage 2: Model score.
    sx, sy = NODE_XS[1], NODE_CY - NODE_H / 2
    parts.append(_node_card(sx, sy, NODE_W, NODE_H, "process"))
    parts.append(_node_label(
        sx, sy, NODE_W, "Model score",
        body_lines=("IForest /", "CNN-LSTM AE", "per window"),
    ))

    # Stage 3: Fleet percentile.
    px, py = NODE_XS[2], NODE_CY - NODE_H / 2
    parts.append(_node_card(px, py, NODE_W, NODE_H, "ranking"))
    parts.append(_node_label(
        px, py, NODE_W, "Percentile",
        body_lines=("rank(s_i)", "across fleet", "at time t"),
    ))

    # Stage 4: Sustained check (with inline mini-waveform — replaces the
    # decision diamond from the v0.1 attempt).
    cx, cy = NODE_XS[3], NODE_CY - NODE_H / 2
    # Slightly wider node to host the waveform comfortably.
    check_w = 160
    parts.append(_node_card(cx, cy, check_w, NODE_H, "evaluation"))
    parts.append(
        f'<text x="{cx + check_w / 2}" y="{cy + 28}" text-anchor="middle" '
        f'font-family="Inter, sans-serif" font-size="23" font-weight="600" '
        f'fill="{COLOR["cypress_deep"]}">Sustained?</text>'
    )
    parts.append(svg_sustained_waveform(
        cx=cx + check_w / 2, cy=cy + 72, w=126, h=46,
        n_windows=5, n_above=3,
    ))
    parts.append(
        f'<text x="{cx + check_w / 2}" y="{cy + NODE_H - 22}" text-anchor="middle" '
        f'font-family="Inter, sans-serif" font-size="23" '
        f'fill="{COLOR["forest"]}">&#8805;95th, &#8805;3 win</text>'
    )

    # Stage 5: Alert.
    ax, ay = NODE_XS[4], NODE_CY - NODE_H / 2
    parts.append(_node_card(ax, ay, NODE_W, NODE_H, "alert"))
    parts.append(_node_label(
        ax, ay, NODE_W, "Alert",
        body_lines=("NORMAL", "WARNING", "CRITICAL"),
    ))

    # ── Edges ────────────────────────────────────────────────────────
    edge_y = NODE_CY
    # Helper: arrow from right edge of node i to left edge of node i+1.
    def right_x(i):  # right edge of node i
        return NODE_XS[i] + (160 if i == 3 else NODE_W)
    def left_x(i):  # left edge of node i
        return NODE_XS[i]

    # Edge labels intentionally omitted — the node names already carry the
    # semantics ("Model score" -> "Percentile" -> "Sustained?" -> "Alert"
    # is self-evident; the caption holds any extra detail). The final arrow
    # remains "strong" (cypress-deep, thicker) to mark the alert path.
    parts.append(svg_edge(x1=right_x(0), y1=edge_y, x2=left_x(1) - 3, y2=edge_y))
    parts.append(svg_edge(x1=right_x(1), y1=edge_y, x2=left_x(2) - 3, y2=edge_y))
    parts.append(svg_edge(x1=right_x(2), y1=edge_y, x2=left_x(3) - 3, y2=edge_y))
    parts.append(svg_edge(x1=right_x(3), y1=edge_y, x2=left_x(4) - 3, y2=edge_y,
                          style="strong"))

    return "".join(parts)


def main():
    body = build_pipeline()
    svg = svg_document(width=W, height=H, body=body)
    out = Path(__file__).with_suffix(".svg")
    out.write_text(svg, encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
