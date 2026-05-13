"""Cypresses-styled CMMS feedback loop / continuous learning. Figure 4.8b.

Brief (derived from thesis/main.tex § Production Service, lines 502-532;
and from the actual code in pdm/service/label_matcher.py and
pdm/cli/feedback.py, informed by the 2026-05-12 research report on
closed-loop ML / active-learning / clinical MLOps conventions AND the
post-render feedback that flat-text boxes need inline visual primitives
to read didactically):

    Figure number:   4.8b (\\ref{fig:deployment_feedback})
    Chapter:         4 — Methods
    Title:           CMMS feedback loop and continuous learning
    Output format:   SVG
    Composition:     5-node main row + branch-in from below. Each node
                     carries a small inline glyph illustrating its role:
                       - CMMS work order → spreadsheet-row glyph
                       - Label matcher → 4 colour-coded outcome chips
                                          (confirmed_fault / scheduled_pv /
                                          unrelated, plus a dimmed
                                          false_alarm slot the auto path
                                          never fills)
                       - Feedback adapter → labels-accumulating glyph
                                              (3 tags stacking up)
                       - Retrain IForest → tiny 3-node tree glyph
                       - Active model → version tag "v_i → v_{i+1}"
                       - CLI → terminal-prompt glyph ($ pdm-feedback)

The DUAL-INPUT structure of the figure reflects the actual code: 3 of 4
labels are auto-classified by pdm/service/label_matcher.py; the 4th
(`false_alarm`) is set MANUALLY via `pdm-feedback record --type
false_alarm`. The manual edge uses dashed olive-shadow and enters the
feedback adapter from below, with the "manual" label rendered as a
free-floating annotation halfway up the vertical line (not via the
svg_edge built-in label, which placed the halo too close to the
adapter node's bottom edge in the v1 render).

Companion: Figure 4.8a shows the forward inference pipeline.
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
W, H = 940, 420

# Main row (5 nodes).
MAIN_CY = 130
NODE_W = 160
NODE_H = 220
COL_GUTTER = 14
LEFT_MARGIN = 42
NODE_XS = [LEFT_MARGIN + i * (NODE_W + COL_GUTTER) for i in range(5)]

# CLI branch-in (below main row, vertically aligned with feedback adapter).
CLI_W = 220
CLI_H = 120
CLI_CY = 340
FEEDBACK_CX = NODE_XS[2] + NODE_W / 2
CLI_X = FEEDBACK_CX - CLI_W / 2
CLI_Y = CLI_CY - CLI_H / 2


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
    label_y = y + 30
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


# ── Inline visual primitives ──────────────────────────────────────────

def _work_order_glyph(*, cx, cy, w=90, h=44, role="process"):
    """Document-with-table-rows glyph suggesting a work-order spreadsheet."""
    stroke = ROLE[role]
    fill = FILL_MIX[role]
    out = [
        # outer document
        f'<rect x="{cx - w / 2}" y="{cy - h / 2}" width="{w}" height="{h}" '
        f'rx="2" ry="2" fill="{fill}" stroke="{stroke}" stroke-width="1.0"/>',
    ]
    # 3 mock-row pairs (left tile + right bar)
    row_h = (h - 8) / 3
    for i in range(3):
        ry = cy - h / 2 + 4 + i * (row_h + 1)
        out.append(
            f'<rect x="{cx - w / 2 + 4}" y="{ry}" width="{12}" '
            f'height="{row_h - 2}" rx="1" ry="1" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="0.5"/>'
        )
        bar_w = (w - 28) * (0.65 if i < 2 else 0.85)
        out.append(
            f'<rect x="{cx - w / 2 + 20}" y="{ry + (row_h - 6) / 2}" '
            f'width="{bar_w}" height="{4}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="0.4"/>'
        )
    return "".join(out)


def _outcome_chips_glyph(*, cx, cy, w=120, h=46):
    """Four small chips representing the 4 outcome labels — auto trio in
    their role colours, manual false_alarm slot dimmed to canvas-cream
    with a dashed outline showing it's set elsewhere.

    Text labels removed: at chip_h ≈ 21 px a 23 px font would overflow.
    Colour identity (alert=conf_fault, ranking=sched_pv, evaluation=unrelated,
    canvas-cream-dashed=false_alarm) carries the meaning.
    """
    chip_w = (w - 6) / 2
    chip_h = (h - 4) / 2
    chips = [
        # (fill, stroke, dashed)
        (FILL_MIX["alert"],      ROLE["alert"],         False),
        (FILL_MIX["ranking"],    ROLE["ranking"],       False),
        (FILL_MIX["evaluation"], ROLE["evaluation"],    False),
        (COLOR["canvas_cream"],  COLOR["olive_shadow"], True),
    ]
    out = []
    for i, (fill, stroke, dashed) in enumerate(chips):
        col = i % 2
        row = i // 2
        x = cx - w / 2 + col * (chip_w + 4)
        y = cy - h / 2 + row * (chip_h + 4)
        dash_attr = 'stroke-dasharray="3,2" ' if dashed else ''
        out.append(
            f'<rect x="{x}" y="{y}" width="{chip_w}" height="{chip_h}" '
            f'rx="3" ry="3" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{1.2 if not dashed else 0.9}" {dash_attr}/>'
        )
    return "".join(out)


def _label_stack_glyph(*, cx, cy, w=80, h=46, role="evaluation"):
    """Stack of 5 small label tags accumulating (one per CMMS outcome)."""
    stroke = ROLE[role]
    fill = FILL_MIX[role]
    n = 5
    tag_w = w * 0.66
    tag_h = 7
    out = []
    for i in range(n):
        x = cx - tag_w / 2 + i * 1.6
        y = cy - h / 2 + 4 + i * (tag_h + 2)
        if i >= n:
            continue
        # rounded-tag shape (rectangle with a notched left edge)
        notch = 6
        out.append(
            f'<path d="M {x + notch} {y} L {x + tag_w} {y} '
            f'L {x + tag_w} {y + tag_h} L {x + notch} {y + tag_h} '
            f'L {x} {y + tag_h / 2} Z" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="0.8"/>'
        )
    return "".join(out)


def _mini_tree_glyph(*, cx, cy, w=70, h=46, role="process"):
    """Tiny isolation-tree icon: root + 2 children + 4 leaves."""
    stroke = ROLE[role]
    fill = FILL_MIX[role]
    # Root
    root_x = cx
    root_y = cy - h / 2 + 6
    # Children
    lvl1_y = cy - 4
    lvl1_xs = [cx - w * 0.28, cx + w * 0.28]
    # Leaves
    lvl2_y = cy + h / 2 - 4
    lvl2_xs = [cx - w * 0.38, cx - w * 0.16, cx + w * 0.16, cx + w * 0.38]
    out = []
    # Edges
    for c in lvl1_xs:
        out.append(
            f'<line x1="{root_x}" y1="{root_y + 3}" '
            f'x2="{c}" y2="{lvl1_y - 3}" '
            f'stroke="{stroke}" stroke-width="0.7"/>'
        )
    for p, leaves in zip(lvl1_xs, [lvl2_xs[:2], lvl2_xs[2:]]):
        for l in leaves:
            out.append(
                f'<line x1="{p}" y1="{lvl1_y + 3}" '
                f'x2="{l}" y2="{lvl2_y - 3}" '
                f'stroke="{stroke}" stroke-width="0.7"/>'
            )
    # Nodes
    out.append(
        f'<circle cx="{root_x}" cy="{root_y}" r="3.5" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="0.9"/>'
    )
    for c in lvl1_xs:
        out.append(
            f'<circle cx="{c}" cy="{lvl1_y}" r="3" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="0.8"/>'
        )
    for l in lvl2_xs:
        out.append(
            f'<circle cx="{l}" cy="{lvl2_y}" r="2.5" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="0.7"/>'
        )
    return "".join(out)


def _version_tag_glyph(*, cx, cy, w=110, h=40, role="ranking"):
    """Two version-pill tags with an arrow: v_i → v_{i+1}.

    Tags use font-size=23 (rendered ≥ 11 pt), so pill height is bumped to
    32 px to contain the ascenders/descenders comfortably.
    """
    stroke = ROLE[role]
    fill = FILL_MIX[role]
    tag_h = 32   # tall enough for 23 px text
    tag_r = tag_h / 2
    tag_w = (w - 22) / 2
    out = []
    # left tag (previous version, dimmer)
    x0 = cx - w / 2
    out.append(
        f'<rect x="{x0}" y="{cy - tag_r}" width="{tag_w}" height="{tag_h}" '
        f'rx="{tag_r}" ry="{tag_r}" fill="{fill}" stroke="{stroke}" stroke-width="0.9"/>'
        f'<text x="{x0 + tag_w / 2}" y="{cy + 8}" text-anchor="middle" '
        f'font-family="JetBrains Mono, monospace" font-size="23" '
        f'fill="{COLOR["forest"]}">v3</text>'
    )
    # arrow
    arrow_x1 = x0 + tag_w + 2
    arrow_x2 = x0 + tag_w + 20
    out.append(
        f'<line x1="{arrow_x1}" y1="{cy}" x2="{arrow_x2 - 3}" y2="{cy}" '
        f'stroke="{stroke}" stroke-width="1.0"/>'
        f'<path d="M {arrow_x2 - 5} {cy - 3} L {arrow_x2} {cy} '
        f'L {arrow_x2 - 5} {cy + 3} Z" fill="{stroke}"/>'
    )
    # right tag (active version — bold, cypress-deep stroke)
    x1 = x0 + tag_w + 22
    out.append(
        f'<rect x="{x1}" y="{cy - tag_r}" width="{tag_w}" height="{tag_h}" '
        f'rx="{tag_r}" ry="{tag_r}" fill="{fill}" stroke="{COLOR["cypress_deep"]}" '
        f'stroke-width="1.8"/>'
        f'<text x="{x1 + tag_w / 2}" y="{cy + 8}" text-anchor="middle" '
        f'font-family="JetBrains Mono, monospace" font-size="23" '
        f'font-weight="600" fill="{COLOR["cypress_deep"]}">v4</text>'
    )
    return "".join(out)


def _cli_terminal_glyph(*, cx, cy, w=200, h=80):
    """Terminal-prompt visual: shortened commands at font-size=23.

    Width bumped from 170 to 200 and height from 58 to 80 to accommodate
    23 px text (rendered ≥ 11 pt at W=940, textwidth=6.3 in).
    Commands abbreviated to fit within 200 px at ~14 px/char:
      '$ pdm feedback' → 14 chars × 14 = 196 px (just fits).
    """
    stroke = COLOR["cypress_deep"]
    fill = COLOR["cypress_deep"]
    out = [
        f'<rect x="{cx - w / 2}" y="{cy - h / 2}" width="{w}" height="{h}" '
        f'rx="4" ry="4" fill="{fill}" stroke="{stroke}" stroke-width="0.8"/>'
    ]
    # title bar
    out.append(
        f'<rect x="{cx - w / 2}" y="{cy - h / 2}" width="{w}" height="12" '
        f'fill="{COLOR["olive_shadow"]}" stroke="none"/>'
    )
    # 3 traffic lights
    for i, c in enumerate([ROLE["alert"], ROLE["ranking"], ROLE["input"]]):
        out.append(
            f'<circle cx="{cx - w / 2 + 8 + i * 8}" cy="{cy - h / 2 + 6}" '
            f'r="2.5" fill="{c}" stroke="none"/>'
        )
    # prompt text — abbreviated to fit at 23 px (~14 px/char)
    out.append(
        f'<text x="{cx - w / 2 + 10}" y="{cy - 6}" '
        f'font-family="JetBrains Mono, monospace" font-size="23" '
        f'fill="{COLOR["canvas_cream"]}">$ pdm feedback</text>'
    )
    out.append(
        f'<text x="{cx - w / 2 + 10}" y="{cy + 22}" '
        f'font-family="JetBrains Mono, monospace" font-size="23" '
        f'fill="{COLOR["canvas_cream"]}">--type fa</text>'
    )
    return "".join(out)


# ── Build ─────────────────────────────────────────────────────────────

def build_figure():
    parts = []
    top = MAIN_CY - NODE_H / 2

    # ── 01 CMMS work order (spreadsheet-row glyph) ───────────────────
    parts.append(_card(x=NODE_XS[0], y=top, w=NODE_W, h=NODE_H, role="process"))
    parts.append(_label_set(
        x=NODE_XS[0], y=top, w=NODE_W,
        label=("CMMS", "work order"),
        body_lines=("Medusa export",),
    ))
    parts.append(_work_order_glyph(
        cx=NODE_XS[0] + NODE_W / 2, cy=top + NODE_H - 36,
        w=110, h=48, role="process",
    ))

    # ── 02 Label matcher (4 outcome chips) ───────────────────────────
    parts.append(_card(x=NODE_XS[1], y=top, w=NODE_W, h=NODE_H, role="process"))
    parts.append(_label_set(
        x=NODE_XS[1], y=top, w=NODE_W,
        label="Label matcher",
        body_lines=("auto-classify",),
    ))
    parts.append(_outcome_chips_glyph(
        cx=NODE_XS[1] + NODE_W / 2, cy=top + NODE_H - 36,
        w=132, h=50,
    ))

    # ── 03 Feedback adapter (labels-accumulating tag stack) ──────────
    parts.append(_card(x=NODE_XS[2], y=top, w=NODE_W, h=NODE_H, role="evaluation"))
    parts.append(_label_set(
        x=NODE_XS[2], y=top, w=NODE_W,
        label=("Feedback", "adapter"),
        body_lines=("feedback.py",),
    ))
    parts.append(_label_stack_glyph(
        cx=NODE_XS[2] + NODE_W / 2, cy=top + NODE_H - 32,
        w=92, h=52, role="evaluation",
    ))

    # ── 04 Retrain IForest (mini-tree glyph) ─────────────────────────
    parts.append(_card(x=NODE_XS[3], y=top, w=NODE_W, h=NODE_H, role="process"))
    parts.append(_label_set(
        x=NODE_XS[3], y=top, w=NODE_W,
        label=("Retrain", "IForest"),
        body_lines=("≥10 labels,", "≥1 confirmed"),
    ))
    parts.append(_mini_tree_glyph(
        cx=NODE_XS[3] + NODE_W / 2, cy=top + NODE_H - 32,
        w=84, h=46, role="process",
    ))

    # ── 05 Active model (version-tag glyph) ──────────────────────────
    parts.append(_card(x=NODE_XS[4], y=top, w=NODE_W, h=NODE_H, role="ranking"))
    parts.append(_label_set(
        x=NODE_XS[4], y=top, w=NODE_W,
        label="Active model",
        body_lines=("versioned",),
    ))
    parts.append(_version_tag_glyph(
        cx=NODE_XS[4] + NODE_W / 2, cy=top + NODE_H - 30,
        w=132, h=40, role="ranking",
    ))

    # ── Manual CLI branch-in (terminal glyph) ────────────────────────
    parts.append(_cli_terminal_glyph(
        cx=FEEDBACK_CX, cy=CLI_CY, w=210, h=72,
    ))

    # ── Main-row edges (dashed olive-shadow feedback style) ──────────
    for i in range(4):
        x1 = NODE_XS[i] + NODE_W + 2
        x2 = NODE_XS[i + 1] - 2
        parts.append(svg_edge(
            x1=x1, y1=MAIN_CY, x2=x2, y2=MAIN_CY,
            style="feedback",
        ))

    # ── Manual CLI → feedback adapter (vertical dashed; label rendered
    #    separately so its halo doesn't bleed into either node) ──────
    fb_bottom = MAIN_CY + NODE_H / 2 + 2
    cli_top = CLI_CY - 36 - 2
    parts.append(svg_edge(
        x1=FEEDBACK_CX, y1=cli_top,
        x2=FEEDBACK_CX, y2=fb_bottom,
        style="feedback",
    ))
    # Position the "manual" label at the true midpoint of the vertical
    # segment, with a halo sized to the actual text. Clear gap from
    # both node boundaries.
    mid_y = (cli_top + fb_bottom) / 2
    label_text = "manual"
    halo_w = len(label_text) * 7.2 + 12
    parts.append(
        f'<rect x="{FEEDBACK_CX - halo_w / 2}" y="{mid_y - 9}" '
        f'width="{halo_w}" height="18" '
        f'fill="{COLOR["canvas_cream"]}" stroke="none"/>'
        f'<text x="{FEEDBACK_CX}" y="{mid_y + 4}" text-anchor="middle" '
        f'font-family="Inter, sans-serif" font-size="14" font-style="italic" '
        f'fill="{COLOR["olive_shadow"]}">{label_text}</text>'
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
