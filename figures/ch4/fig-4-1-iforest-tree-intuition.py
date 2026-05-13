"""Cypresses-styled isolation-tree splitting intuition. Figure 4.1b.

Brief (derived from the 2026-05-12 research report on iForest figure
conventions, and Liu et al. 2008 / 2012 binary-tree intuition):

    Figure number:   4.1b (\\ref{fig:iforest_tree})
    Chapter:         4 — Methods
    Title:           Isolation-tree splitting intuition
    Output format:   SVG
    Composition:     3-level binary tree on the left, "× 200 trees ·
                     average path length" arrow to a score-interpretation
                     strip on the right. The anomalous leaf carries the
                     DESIGN.md §4.1 anomaly treatment (cypress-deep
                     stroke + dashed halo); every other tree node uses
                     the ranking role (olive-yellow) since the tree
                     produces a path-length, which downstream becomes
                     a rank-based score.
    Tree:
       Root (k < val)
         ├─ Left   (m < val)
         │   ├─ Leaf  ANOMALY  path = 2     ← anomaly-treatment
         │   └─ Leaf  ...deeper...
         └─ Right  (n < val)
             ├─ Leaf  ...deeper...
             └─ Leaf  ...deeper...           ← caption "long path → normal"
    Score strip:
       score → 1.0   anomalous   (alert role)
       score → 0.5   normal      (input role)
       (higher = more anomalous; negated decision_function)
    Connector:  tree → score strip, labelled "× 200 trees · avg path length"

Source canvas: 940 × 460 user units. The score formula s(x) =
2^(-E[h(x)]/c(n)) is in the prose as a numbered equation, not here.

Tree edges are drawn as plain forest 1.25 px lines without arrowheads:
these are structural parent-child relationships, not flow arrows.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from thesis.tokens.cypresses import (
    COLOR, FILL_MIX, ROLE, STROKE_WIDTH,
    svg_document,
)


# ── Canvas ─────────────────────────────────────────────────────────────
W, H = 940, 460


# ── Geometry ──────────────────────────────────────────────────────────
# Left half hosts the tree, right half the score strip.
TREE_X0, TREE_X1 = 20, 560
SCORE_X0, SCORE_X1 = 620, 920


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))


def _internal_node(*, cx, cy, w=180, h=56, text=""):
    """Rounded-rect internal tree node (ranking role).
    font-size=23 → rendered ≥ 11 pt at W=940, textwidth=6.3 in.
    """
    x = cx - w / 2
    y = cy - h / 2
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" ry="4" '
        f'fill="{FILL_MIX["ranking"]}" stroke="{ROLE["ranking"]}" '
        f'stroke-width="{STROKE_WIDTH["ranking"]}"/>',
        f'<text x="{cx}" y="{cy + 8}" text-anchor="middle" '
        f'font-family="JetBrains Mono, monospace" font-size="23" font-weight="500" '
        f'fill="{COLOR["cypress_deep"]}">{_escape(text)}</text>',
    ]
    return "".join(parts)


def _leaf(*, cx, cy, w=120, h=52, label="", sub=None, anomaly=False):
    """Leaf node. If anomaly=True, applies DESIGN.md §4.1 treatment:
    cypress-deep stroke at 2.0 px + 1 px dashed halo offset 4 px."""
    x = cx - w / 2
    y = cy - h / 2
    if anomaly:
        fill = FILL_MIX["alert"]
        stroke = COLOR["cypress_deep"]
        sw = 2.0
    else:
        fill = FILL_MIX["ranking"]
        stroke = ROLE["ranking"]
        sw = STROKE_WIDTH["ranking"]
    parts = []
    if anomaly:
        parts.append(
            f'<rect x="{x - 4}" y="{y - 4}" width="{w + 8}" height="{h + 8}" '
            f'rx="6" ry="6" fill="none" stroke="{COLOR["cypress_deep"]}" '
            f'stroke-width="1" stroke-dasharray="3,3"/>'
        )
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" ry="4" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )
    # Two-line content: bold label + optional sub-line.
    # font-size=23 → rendered ≥ 11 pt at W=940, textwidth=6.3 in.
    if sub:
        parts.append(
            f'<text x="{cx}" y="{cy - 4}" text-anchor="middle" '
            f'font-family="Inter, sans-serif" font-size="23" font-weight="600" '
            f'fill="{COLOR["cypress_deep"]}">{_escape(label)}</text>'
        )
        parts.append(
            f'<text x="{cx}" y="{cy + 20}" text-anchor="middle" '
            f'font-family="JetBrains Mono, monospace" font-size="23" '
            f'fill="{COLOR["forest"]}">{_escape(sub)}</text>'
        )
    else:
        parts.append(
            f'<text x="{cx}" y="{cy + 8}" text-anchor="middle" '
            f'font-family="JetBrains Mono, monospace" font-size="23" '
            f'fill="{COLOR["forest"]}">{_escape(label)}</text>'
        )
    return "".join(parts)


def _tree_edge(*, x1, y1, x2, y2):
    """Plain forest line, no arrowhead — these are structural tree edges,
    not flow arrows. Drawn with stroke 1.25 px."""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{COLOR["forest"]}" stroke-width="1.25"/>'
    )


def _leaf_annotation(*, cx, y, text):
    """Italic olive-shadow caption under a leaf.
    font-size=23 → rendered ≥ 11 pt at W=940, textwidth=6.3 in.
    """
    return (
        f'<text x="{cx}" y="{y}" text-anchor="middle" '
        f'font-family="Inter, sans-serif" font-size="23" font-style="italic" '
        f'fill="{COLOR["olive_shadow"]}">{_escape(text)}</text>'
    )


def _score_row(*, cx, cy, w, h, marker, label, role):
    """One row in the score interpretation strip: marker chip + label.
    font-size=23 → rendered ≥ 11 pt at W=940, textwidth=6.3 in.
    """
    chip_w = 90
    x = cx - w / 2
    y = cy - h / 2
    parts = [
        # Background row
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" ry="4" '
        f'fill="{FILL_MIX[role]}" stroke="{ROLE[role]}" '
        f'stroke-width="{STROKE_WIDTH[role]}"/>',
        # Marker chip on left (mono)
        f'<text x="{x + 14}" y="{cy + 8}" '
        f'font-family="JetBrains Mono, monospace" font-size="23" font-weight="500" '
        f'fill="{COLOR["cypress_deep"]}">{_escape(marker)}</text>',
        # Label on right
        f'<text x="{x + chip_w + 18}" y="{cy + 8}" '
        f'font-family="Inter, sans-serif" font-size="23" font-weight="500" '
        f'fill="{COLOR["cypress_deep"]}">{_escape(label)}</text>',
    ]
    return "".join(parts)


def _section_title(*, x, y, text, anchor="start"):
    # 23 px → rendered ≥ 11 pt at W=940, textwidth=6.3 in.
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="JetBrains Mono, monospace" font-size="23" '
        f'letter-spacing="1.8" font-weight="600" '
        f'fill="{COLOR["olive_shadow"]}">{text.upper()}</text>'
    )


def build_figure():
    parts = []

    # ── Tree (left half) ──────────────────────────────────────────────
    parts.append(_section_title(x=TREE_X0 + 10, y=30,
                                text="One isolation tree (conceptual)"))

    # Tree node centres
    root_cx = (TREE_X0 + TREE_X1) / 2          # 290
    root_cy = 88

    int_cy = 220
    int_left_cx = TREE_X0 + 140                # 160
    int_right_cx = TREE_X1 - 140               # 420

    leaf_cy = 348
    leaf_xs = [TREE_X0 + 60, TREE_X0 + 220, TREE_X0 + 360, TREE_X0 + 500]
    #            80           240             380            560 — but TREE_X1=560
    # Let me recompute with bounds. With TREE_X1=560, leaf 4 at center 560 puts
    # its right edge at 560+60=620 — overlaps the score strip starting at 620.
    # Reduce leaf 4 cx:
    leaf_xs = [TREE_X0 + 60, TREE_X0 + 200, TREE_X0 + 340, TREE_X0 + 480]
    #             80           220             360            500

    # Root (internal)
    parts.append(_internal_node(cx=root_cx, cy=root_cy,
                                text="feature_j  <  split"))

    # Two internal nodes (level 1)
    parts.append(_internal_node(cx=int_left_cx, cy=int_cy, w=170,
                                text="feature_k  <  val"))
    parts.append(_internal_node(cx=int_right_cx, cy=int_cy, w=170,
                                text="feature_m  <  val"))

    # Four leaves (level 2)
    parts.append(_leaf(cx=leaf_xs[0], cy=leaf_cy, w=120, h=56,
                       label="ANOMALY", sub="path = 2", anomaly=True))
    parts.append(_leaf(cx=leaf_xs[1], cy=leaf_cy, w=110, h=56,
                       label="…deeper…"))
    parts.append(_leaf(cx=leaf_xs[2], cy=leaf_cy, w=110, h=56,
                       label="…deeper…"))
    parts.append(_leaf(cx=leaf_xs[3], cy=leaf_cy, w=110, h=56,
                       label="…deeper…"))

    # Tree edges (parent bottom → child top)
    root_bottom = root_cy + 28          # h/2 = 28
    int_top = int_cy - 28
    int_bottom = int_cy + 28
    leaf_top = leaf_cy - 28

    parts.append(_tree_edge(x1=root_cx, y1=root_bottom,
                            x2=int_left_cx, y2=int_top))
    parts.append(_tree_edge(x1=root_cx, y1=root_bottom,
                            x2=int_right_cx, y2=int_top))
    parts.append(_tree_edge(x1=int_left_cx, y1=int_bottom,
                            x2=leaf_xs[0], y2=leaf_top))
    parts.append(_tree_edge(x1=int_left_cx, y1=int_bottom,
                            x2=leaf_xs[1], y2=leaf_top))
    parts.append(_tree_edge(x1=int_right_cx, y1=int_bottom,
                            x2=leaf_xs[2], y2=leaf_top))
    parts.append(_tree_edge(x1=int_right_cx, y1=int_bottom,
                            x2=leaf_xs[3], y2=leaf_top))

    # Leaf annotations
    parts.append(_leaf_annotation(cx=leaf_xs[0], y=leaf_cy + 50,
                                  text="short path → anomalous"))
    parts.append(_leaf_annotation(cx=leaf_xs[3], y=leaf_cy + 50,
                                  text="long path → normal"))

    # ── Connector: tree → score strip ────────────────────────────────
    # Exits the tree zone at its right boundary, sitting BELOW the
    # internal-node row with clear vertical gap (avoids the halo
    # overlapping the bottom-right corner of "feature_m < val");
    # arrives at the score strip's left edge at the same y. Solid forest
    # line with stealth arrowhead, italic label above on per-line
    # canvas-cream halos sized to the text width (not a fixed wide
    # rectangle that bleeds into the tree zone).
    conn_y = int_cy + 80                # 80 u below internal-node centre
                                        #   → 52 u below their bottoms
    conn_x1 = TREE_X1 + 6
    conn_x2 = SCORE_X0 - 6
    parts.append(
        f'<line x1="{conn_x1}" y1="{conn_y}" x2="{conn_x2}" y2="{conn_y}" '
        f'stroke="{COLOR["forest"]}" stroke-width="1.25" '
        f'marker-end="url(#arrow)"/>'
    )
    # Two stacked, per-line halos sized to actual text width.
    # font-size=23 → rendered ≥ 11 pt at W=940, textwidth=6.3 in.
    conn_mx = (conn_x1 + conn_x2) / 2
    line1 = "× 200 trees"
    line2 = "avg path length"
    line1_w = len(line1) * 14.0 + 14
    line2_w = len(line2) * 14.0 + 14
    line1_y = conn_y - 36
    line2_y = conn_y - 10
    parts.append(
        f'<rect x="{conn_mx - line1_w / 2}" y="{line1_y - 18}" '
        f'width="{line1_w}" height="22" '
        f'fill="{COLOR["canvas_cream"]}" stroke="none"/>'
    )
    parts.append(
        f'<rect x="{conn_mx - line2_w / 2}" y="{line2_y - 18}" '
        f'width="{line2_w}" height="22" '
        f'fill="{COLOR["canvas_cream"]}" stroke="none"/>'
    )
    parts.append(
        f'<text x="{conn_mx}" y="{line1_y}" text-anchor="middle" '
        f'font-family="Inter, sans-serif" font-size="23" font-style="italic" '
        f'fill="{COLOR["olive_shadow"]}">{line1}</text>'
    )
    parts.append(
        f'<text x="{conn_mx}" y="{line2_y}" text-anchor="middle" '
        f'font-family="Inter, sans-serif" font-size="23" font-style="italic" '
        f'fill="{COLOR["olive_shadow"]}">{line2}</text>'
    )

    # ── Score strip (right half) ─────────────────────────────────────
    parts.append(_section_title(x=SCORE_X0 + 6, y=30, text="Anomaly score"))

    strip_cx = (SCORE_X0 + SCORE_X1) / 2
    row_w = SCORE_X1 - SCORE_X0 - 12
    row_h = 52
    row_gap = 18

    parts.append(_score_row(
        cx=strip_cx, cy=80 + row_h / 2, w=row_w, h=row_h,
        marker="→ 1.0", label="anomalous", role="alert",
    ))
    parts.append(_score_row(
        cx=strip_cx, cy=80 + row_h + row_gap + row_h / 2, w=row_w, h=row_h,
        marker="→ 0.5", label="normal", role="input",
    ))

    # Caveat below the two rows
    # font-size=23 → rendered ≥ 11 pt at W=940, textwidth=6.3 in.
    caveat_y = 80 + 2 * row_h + row_gap + 46
    parts.append(
        f'<text x="{strip_cx}" y="{caveat_y}" text-anchor="middle" '
        f'font-family="Inter, sans-serif" font-size="23" font-style="italic" '
        f'fill="{COLOR["olive_shadow"]}">higher = more anomalous;</text>'
    )
    parts.append(
        f'<text x="{strip_cx}" y="{caveat_y + 28}" text-anchor="middle" '
        f'font-family="Inter, sans-serif" font-size="23" font-style="italic" '
        f'fill="{COLOR["olive_shadow"]}">negated <tspan font-family="JetBrains Mono, monospace" '
        f'font-style="normal">decision_function</tspan></text>'
    )

    return "".join(parts)


def main():
    body = build_figure()
    svg = svg_document(width=W, height=H, body=body)
    out = Path(__file__).with_suffix(".svg")
    out.write_text(svg, encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
