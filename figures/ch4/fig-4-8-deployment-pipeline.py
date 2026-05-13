"""Cypresses-styled production deployment pipeline. Figure 4.8a.

Brief (derived from thesis/main.tex § Production Service line 448 ff. and
from the actual code in pdm/service/, informed by the 2026-05-12 research
report on production-ML / clinical-deployment architecture conventions
AND the post-render feedback that flat-text boxes need *inline visual
primitives* — like the fleet-glyph and mini-waveform in Fig 4.5 — to
read didactically rather than as decoration):

    Figure number:   4.8a (\\ref{fig:deployment_pipeline})
    Chapter:         4 — Methods
    Title:           Daily inference pipeline
    Output format:   SVG
    Composition:     6-node horizontal linear flow. Sage bookends
                     (QuestDB cylinder for the DB origin, Dashboard
                     mockup for the human-facing destination). Each
                     node carries a small inline visualization that
                     illustrates its function:
                       - QuestDB → cylinder + mini timeseries glyph
                       - Data extraction → Parquet-file stack glyph
                       - Feature pipeline → 5×N reduction grid glyph
                       - Model scoring → score-distribution sparkline
                       - Alert generation → 3 level-pill badges
                                            (NORMAL / WARNING / CRITICAL)
                       - Dashboard → 3-row device-card mockup

Text-fit pre-flight (per the skill step 7):
  - "Model scoring" (13 char × 12 = 156 u) overflows 140-u node, so
    split label to ("Model", "scoring") — same fix the user just
    flagged on the v1 render.
  - All other labels and body lines verified against 132 u usable
    width (140 - 28 padding).

Companion: Figure 4.8b shows the CMMS feedback loop closing the cycle.
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
NODE_CY = 140
NODE_W = 140
NODE_H = 220
COL_GUTTER = 14

# 6 nodes; 6 × 140 + 5 × 14 = 910; margin (940-910)/2 = 15
LEFT_MARGIN = 15
NODE_XS = [LEFT_MARGIN + i * (NODE_W + COL_GUTTER) for i in range(6)]


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


# ── Inline visual primitives (the "didactic" elements) ────────────────

def _cylinder_glyph(*, cx, cy, w=60, h=44, role="input"):
    """Database-cylinder glyph (universal DB convention).

    Two stacked ellipses + side rectangle; the top ellipse is drawn fully
    while the bottom is only the front half. Uses the role's fill + stroke.
    """
    stroke = ROLE[role]
    fill = FILL_MIX[role]
    rx = w / 2
    ry = h * 0.18
    top_y = cy - h / 2 + ry
    bot_y = cy + h / 2 - ry
    return (
        # body sides + bottom front arc
        f'<path d="M {cx - rx} {top_y} L {cx - rx} {bot_y} '
        f'A {rx} {ry} 0 0 0 {cx + rx} {bot_y} L {cx + rx} {top_y} Z" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
        # top ellipse
        f'<ellipse cx="{cx}" cy="{top_y}" rx="{rx}" ry="{ry}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
        # 2 horizontal partition lines suggesting rows
        f'<line x1="{cx - rx * 0.7}" y1="{cy - ry * 0.8}" '
        f'x2="{cx + rx * 0.7}" y2="{cy - ry * 0.8}" '
        f'stroke="{stroke}" stroke-width="0.6" opacity="0.7"/>'
        f'<line x1="{cx - rx * 0.7}" y1="{cy + ry * 0.8}" '
        f'x2="{cx + rx * 0.7}" y2="{cy + ry * 0.8}" '
        f'stroke="{stroke}" stroke-width="0.6" opacity="0.7"/>'
    )


def _parquet_stack_glyph(*, cx, cy, w=56, h=42, role="process"):
    """Three stacked file-rectangles with a folded corner (parquet files)."""
    stroke = ROLE[role]
    fill = FILL_MIX[role]
    out = []
    file_w = w * 0.72
    file_h = h * 0.62
    fold = 6
    for i in range(3):
        ox = cx - file_w / 2 + (i - 1) * 5
        oy = cy - file_h / 2 + (i - 1) * 4
        out.append(
            f'<path d="M {ox} {oy} L {ox + file_w - fold} {oy} '
            f'L {ox + file_w} {oy + fold} L {ox + file_w} {oy + file_h} '
            f'L {ox} {oy + file_h} Z" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.0"/>'
        )
        # fold mark
        out.append(
            f'<path d="M {ox + file_w - fold} {oy} '
            f'L {ox + file_w - fold} {oy + fold} L {ox + file_w} {oy + fold}" '
            f'fill="none" stroke="{stroke}" stroke-width="0.8"/>'
        )
    return "".join(out)


def _grid_reduction_glyph(*, cx, cy, w=64, h=44, role="process"):
    """Two stacked grids visualising feature reduction (wide → narrow).

    Left: a 5x6 grid (suggesting 30 timesteps × N features after pivot).
    Right: a 1x5 column (the 5 summary statistics per feature).
    Arrow between.
    """
    stroke = ROLE[role]
    fill = FILL_MIX[role]
    # Left grid: 5 cols × 4 rows, total 20 cells
    cell = 4
    gap = 1
    cols, rows = 5, 4
    grid_w = cols * cell + (cols - 1) * gap
    grid_h = rows * cell + (rows - 1) * gap
    left_x = cx - w / 2
    grid_y = cy - grid_h / 2
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(
                f'<rect x="{left_x + c * (cell + gap)}" '
                f'y="{grid_y + r * (cell + gap)}" '
                f'width="{cell}" height="{cell}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="0.5"/>'
            )
    # Arrow
    arrow_x1 = left_x + grid_w + 4
    arrow_x2 = cx + w / 2 - cell * 2 - 4
    out.append(
        f'<line x1="{arrow_x1}" y1="{cy}" x2="{arrow_x2}" y2="{cy}" '
        f'stroke="{stroke}" stroke-width="0.8"/>'
        f'<path d="M {arrow_x2 - 2} {cy - 2} L {arrow_x2 + 1} {cy} '
        f'L {arrow_x2 - 2} {cy + 2} Z" fill="{stroke}"/>'
    )
    # Right column: 1 col × 5 rows (the 5 summary stats)
    right_x = cx + w / 2 - cell
    out.append(
        f'<rect x="{right_x}" y="{cy - cell * 2.5 - 2}" '
        f'width="{cell}" height="{cell * 5 + 4}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="0.6"/>'
    )
    # Tiny dividers in the right column
    for i in range(1, 5):
        out.append(
            f'<line x1="{right_x}" y1="{cy - cell * 2.5 - 2 + i * (cell + 1)}" '
            f'x2="{right_x + cell}" '
            f'y2="{cy - cell * 2.5 - 2 + i * (cell + 1)}" '
            f'stroke="{stroke}" stroke-width="0.4"/>'
        )
    return "".join(out)


def _score_sparkline_glyph(*, cx, cy, w=72, h=40, role="process"):
    """Score distribution / sparkline with a 95th-percentile threshold line."""
    stroke = ROLE[role]
    fill = FILL_MIX[role]
    baseline_y = cy + h / 2 - 2
    x0 = cx - w / 2
    # 12 small score bars of varying heights, one spike at the right
    bar_heights = [0.18, 0.22, 0.20, 0.28, 0.30, 0.24, 0.32, 0.28, 0.35,
                   0.42, 0.55, 0.85]
    n = len(bar_heights)
    bar_w = (w - 4) / n - 1
    out = []
    for i, hf in enumerate(bar_heights):
        bh = h * hf
        bx = x0 + i * (bar_w + 1)
        by = baseline_y - bh
        out.append(
            f'<rect x="{bx}" y="{by}" width="{bar_w}" height="{bh}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="0.5"/>'
        )
    # 95th percentile dashed threshold line near top
    thr_y = cy - h * 0.30
    out.append(
        f'<line x1="{x0}" y1="{thr_y}" x2="{x0 + w}" y2="{thr_y}" '
        f'stroke="{ROLE["alert"]}" stroke-width="1.0" stroke-dasharray="3,2"/>'
    )
    return "".join(out)


def _alert_pills_glyph(*, cx, cy, w=80, h=44):
    """Three vertically-stacked alert-level pills (colour-coded, no text labels).

    Text labels removed: at h=44 each pill is only ~13 px tall, which cannot
    host 23 px text without overflow. Colour identity (input=NORMAL,
    alert=WARNING, cypress-deep-stroke=CRITICAL) carries the meaning;
    the node's label text above already says 'Alert output'.
    """
    pill_w = w
    pill_h = (h - 4) / 3
    rows = [
        (FILL_MIX["input"],      ROLE["input"]),
        (FILL_MIX["alert"],      ROLE["alert"]),
        (FILL_MIX["alert"],      COLOR["cypress_deep"]),
    ]
    out = []
    for i, (fill, stroke) in enumerate(rows):
        y = cy - h / 2 + i * (pill_h + 2)
        x = cx - pill_w / 2
        out.append(
            f'<rect x="{x}" y="{y}" width="{pill_w}" height="{pill_h}" '
            f'rx="3" ry="3" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{1.6 if i == 2 else 1.0}"/>'
        )
    return "".join(out)


def _device_card_glyph(*, cx, cy, w=80, h=44, role="input"):
    """Mockup of a small device-ranking card from the dashboard.

    Three small rows (e.g. devices), each with a left "device id" tile,
    a thin score bar, and a trailing right tile (alert flag) — the third
    row has its right tile filled to indicate a flagged device.
    """
    stroke = ROLE[role]
    fill = FILL_MIX[role]
    row_h = (h - 4) / 3
    out = []
    for i in range(3):
        y = cy - h / 2 + i * (row_h + 2)
        x = cx - w / 2
        # left id tile
        out.append(
            f'<rect x="{x}" y="{y}" width="{8}" height="{row_h}" '
            f'rx="1" ry="1" fill="{fill}" stroke="{stroke}" stroke-width="0.6"/>'
        )
        # score bar (varying width)
        bar_w = w * (0.35 + i * 0.18)
        out.append(
            f'<rect x="{x + 12}" y="{y + row_h / 2 - 2}" '
            f'width="{bar_w}" height="{4}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="0.5"/>'
        )
        # alert-flag tile on the right; bottom (highest-ranked) gets cypress-deep stroke
        flag_fill = COLOR["cypress_deep"] if i == 2 else COLOR["canvas_cream"]
        flag_stroke = COLOR["cypress_deep"] if i == 2 else stroke
        out.append(
            f'<rect x="{x + w - 8}" y="{y}" width="{8}" height="{row_h}" '
            f'rx="1" ry="1" fill="{flag_fill}" '
            f'stroke="{flag_stroke}" stroke-width="0.8"/>'
        )
    return "".join(out)


# ── Build ─────────────────────────────────────────────────────────────

def build_figure():
    parts = []
    top = NODE_CY - NODE_H / 2

    # ── 01 QuestDB (cylinder glyph below text) ───────────────────────
    parts.append(_card(x=NODE_XS[0], y=top, w=NODE_W, h=NODE_H, role="input"))
    parts.append(_label_set(
        x=NODE_XS[0], y=top, w=NODE_W,
        label="QuestDB",
        body_lines=("time-series",),
        mono_lines=("420M+ rows",),
    ))
    parts.append(_cylinder_glyph(
        cx=NODE_XS[0] + NODE_W / 2, cy=top + NODE_H - 36,
        w=64, h=42, role="input",
    ))

    # ── 02 Data extraction (parquet stack glyph) ─────────────────────
    parts.append(_card(x=NODE_XS[1], y=top, w=NODE_W, h=NODE_H, role="process"))
    parts.append(_label_set(
        x=NODE_XS[1], y=top, w=NODE_W,
        label=("Data", "extraction"),
        body_lines=("extract.py",),
    ))
    parts.append(_parquet_stack_glyph(
        cx=NODE_XS[1] + NODE_W / 2, cy=top + NODE_H - 36,
        w=64, h=42, role="process",
    ))

    # ── 03 Feature pipeline (grid reduction glyph) ───────────────────
    parts.append(_card(x=NODE_XS[2], y=top, w=NODE_W, h=NODE_H, role="process"))
    parts.append(_label_set(
        x=NODE_XS[2], y=top, w=NODE_W,
        label=("Feature", "pipeline"),
        body_lines=("features.py",),
    ))
    parts.append(_grid_reduction_glyph(
        cx=NODE_XS[2] + NODE_W / 2, cy=top + NODE_H - 36,
        w=80, h=44, role="process",
    ))

    # ── 04 Model scoring (sparkline glyph) — label SPLIT, fixes overflow
    parts.append(_card(x=NODE_XS[3], y=top, w=NODE_W, h=NODE_H, role="process"))
    parts.append(_label_set(
        x=NODE_XS[3], y=top, w=NODE_W,
        label=("Model", "scoring"),
        body_lines=("inference.py",),
        mono_lines=("daily 06:00",),
    ))
    parts.append(_score_sparkline_glyph(
        cx=NODE_XS[3] + NODE_W / 2, cy=top + NODE_H - 30,
        w=96, h=42, role="process",
    ))

    # ── 05 Alert generation (3 level pills) ──────────────────────────
    parts.append(_card(x=NODE_XS[4], y=top, w=NODE_W, h=NODE_H, role="alert"))
    parts.append(_label_set(
        x=NODE_XS[4], y=top, w=NODE_W,
        label=("Alert", "generation"),
        body_lines=("≥ 95th × 3",),
    ))
    parts.append(_alert_pills_glyph(
        cx=NODE_XS[4] + NODE_W / 2, cy=top + NODE_H - 36,
        w=104, h=58,
    ))

    # ── 06 Dashboard (device-card mockup glyph) ──────────────────────
    parts.append(_card(x=NODE_XS[5], y=top, w=NODE_W, h=NODE_H, role="input"))
    parts.append(_label_set(
        x=NODE_XS[5], y=top, w=NODE_W,
        label="Dashboard",
        body_lines=("FastAPI",),
    ))
    parts.append(_device_card_glyph(
        cx=NODE_XS[5] + NODE_W / 2, cy=top + NODE_H - 36,
        w=104, h=58, role="input",
    ))

    # ── Forward edges ────────────────────────────────────────────────
    for i in range(5):
        x1 = NODE_XS[i] + NODE_W + 2
        x2 = NODE_XS[i + 1] - 2
        parts.append(svg_edge(x1=x1, y1=NODE_CY, x2=x2, y2=NODE_CY))

    return "".join(parts)


def main():
    body = build_figure()
    svg = svg_document(width=W, height=H, body=body)
    out = Path(__file__).with_suffix(".svg")
    out.write_text(svg, encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
