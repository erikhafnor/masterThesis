"""Cypresses-styled data ingestion pipeline for Figure 3.1 (\\ref{fig:ingestion}).

Brief (derived from thesis/main.tex § Data Ingestion Infrastructure
lines 193-211, informed by the 2026-05-12 research report on 2024-2026
healthcare-telemetry and streaming-ETL pipeline figure conventions):

    Figure number:   3.1
    Chapter:         3 — Data and System Description
    Title:           Data Ingestion Pipeline
    Subtitle:        From ventilator HL7 telemetry to QuestDB storage.
    Output format:   SVG
    Composition:     Three stacked rows (Physical / Parsing / Storage)
                     matching the three layers described in the thesis
                     prose; left-margin labels rather than swim-lane
                     boxes. Linear left-to-right flow within each row,
                     U-turn polyline connectors between rows.
                     Protocols stay on edges, components in nodes
                     (Hassan et al. 2023, Sensors 25(9):2945 2025).
    Stages:
      Physical row (4 nodes):
        01 Elisa 800 ventilators  (input)    30-device fleet, ICU/anaesthesia
        02 Capsule MDIP gateway   (process)  SalviaA DDI device-to-network bridge
        03 MLLP receiver          (process)  TCP 2554, HL7 ACK
        04 HL7 parser             (process)  MSH / PID / OBR / OBX
      Parsing row (4 nodes):
        05 OBX extraction         (process)  variable_id, value, unit, timestamp
        06 Serial-number tracking (process)  device_id tagging, Elisa 800 only
        07 Bitfield decoder       (process)  vars 801-804 -> 17 boolean flags
        08 Value mapping          (process)  ~190 vars -> labels + units
      Storage row (2 nodes):
        09 ILP writer             (process)  Influx Line Protocol, TCP 9009
        10 QuestDB                (input)    pdm_medical_device, 420M+ rows;
                                              terminal store, becomes the
                                              input for every downstream
                                              figure in the thesis.
    Edges (only protocol-boundary labels survive):
      02 -> 03 : "MLLP"   (HL7 v2.x over MLLP-framed TCP)
      09 -> 10 : "ILP"    (Influx Line Protocol over TCP)
      all other within-row edges : bare arrows
      end-of-row -> start-of-next-row : U-turn polyline, bare arrow into
        the top-centre of the next row's first node.
    Highlights:  none (infrastructure figure — no anomaly to mark).
    Caption:     unchanged from main.tex line 207; existing prose around
                 the figure carries the quantitative context (190 vars,
                 17 booleans, 420M rows, ~58 / 22 s sample rates).

Source canvas: 940 x 540 user units. At \\textwidth = 6.3 in the scale
is ~0.483; source font 22 -> 10.6 pt (node label), 16 -> 7.7 pt (body
short), 14 -> 6.8 pt (mono). Body sizes are deliberately tighter than
the Fig 4.4 pipeline (which has 5 nodes and breathing room); a 10-node
3-row layout needs more compact node copy.
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

# ── Canvas geometry ────────────────────────────────────────────────────
W, H = 940, 580
LEFT_MARGIN = 110
TOP_MARGIN = 24

ROW_H = 140
ROW_GAP = 48              # DESIGN.md §5: lane gutters ≥ 48 u, never less
NODE_W = 160
NODE_H = ROW_H
COL_GUTTER = 60

NODE_XS = [LEFT_MARGIN + i * (NODE_W + COL_GUTTER) for i in range(4)]
ROW_Y = [
    TOP_MARGIN + ROW_H / 2,
    TOP_MARGIN + ROW_H + ROW_GAP + ROW_H / 2,
    TOP_MARGIN + 2 * (ROW_H + ROW_GAP) + ROW_H / 2,
]


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


def _node(x, y, role, label, body=(), mono=()):
    return _card(x=x, y=y, w=NODE_W, h=NODE_H, role=role) + _label_set(
        x=x, y=y, w=NODE_W, label=label, body_lines=body, mono_lines=mono
    )


def _layer_label(*, y_centre, text):
    # 23 px → rendered ≥ 11 pt at W=940, textwidth=6.3 in.
    return (
        f'<text x="55" y="{y_centre}" text-anchor="middle" '
        f'dominant-baseline="middle" '
        f'font-family="JetBrains Mono, monospace" font-size="23" '
        f'letter-spacing="2.4" font-weight="600" '
        f'fill="{COLOR["olive_shadow"]}">{text}</text>'
    )


def _fleet_glyph(*, x, y, cols=5, rows=2, cell=10, gap=3):
    """Compact 5x2 grid suggesting a fleet of 30 devices (15 cells shown).

    Sits inside the Elisa 800 input node, beneath the body text. Pure
    decoration but cheap and on-brand.
    """
    out = []
    sage = ROLE["input"]
    fill = FILL_MIX["input"]
    for r in range(rows):
        for c in range(cols):
            cx = x + c * (cell + gap)
            cy = y + r * (cell + gap)
            out.append(
                f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" '
                f'rx="1" ry="1" fill="{fill}" '
                f'stroke="{sage}" stroke-width="0.9"/>'
            )
    return "".join(out)


def build_pipeline():
    parts = []

    # ── Layer labels (left margin) ────────────────────────────────────
    parts.append(_layer_label(y_centre=ROW_Y[0], text="PHYSICAL"))
    parts.append(_layer_label(y_centre=ROW_Y[1], text="PARSING"))
    parts.append(_layer_label(y_centre=ROW_Y[2], text="STORAGE"))

    # ── Row 1: Physical ───────────────────────────────────────────────
    r1_top = ROW_Y[0] - NODE_H / 2

    # Node 1: Elisa 800 fleet (input role) — with a small cluster glyph
    # underneath the body text to make the "fleet" feel concrete.
    parts.append(_card(x=NODE_XS[0], y=r1_top, w=NODE_W, h=NODE_H, role="input"))
    parts.append(_label_set(
        x=NODE_XS[0], y=r1_top, w=NODE_W,
        label="Elisa 800",
        body_lines=("ventilator fleet",),
    ))
    glyph_w = 5 * 10 + 4 * 3  # cols=5, cell=10, gap=3 -> 62
    parts.append(_fleet_glyph(
        x=NODE_XS[0] + (NODE_W - glyph_w) / 2,
        y=r1_top + 88,
    ))
    parts.append(
        f'<text x="{NODE_XS[0] + NODE_W / 2}" y="{r1_top + NODE_H - 10}" '
        f'text-anchor="middle" '
        f'font-family="JetBrains Mono, monospace" font-size="23" '
        f'fill="{COLOR["olive_shadow"]}">30 devices</text>'
    )

    parts.append(_node(NODE_XS[1], r1_top, "process",
                       "Capsule MDIP",
                       body=("SalviaA DDI", "gateway")))
    parts.append(_node(NODE_XS[2], r1_top, "process",
                       "MLLP receiver",
                       body=("TCP 2554", "HL7 ACK")))
    parts.append(_node(NODE_XS[3], r1_top, "process",
                       "HL7 parser",
                       body=("MSH · PID", "OBR · OBX")))

    # ── Row 2: Parsing ────────────────────────────────────────────────
    r2_top = ROW_Y[1] - NODE_H / 2
    parts.append(_node(NODE_XS[0], r2_top, "process",
                       "OBX extraction",
                       body=("variable_id, value,", "unit, timestamp")))
    parts.append(_node(NODE_XS[1], r2_top, "process",
                       "Serial tracking",
                       body=("device_id tagging,", "Elisa 800 only")))
    parts.append(_node(NODE_XS[2], r2_top, "process",
                       ("Bitfield", "decoder"),
                       body=("vars 801–804", "→ 17 booleans")))
    parts.append(_node(NODE_XS[3], r2_top, "process",
                       "Value mapping",
                       body=("~190 vars", "→ labels + units")))

    # ── Row 3: Storage ────────────────────────────────────────────────
    r3_top = ROW_Y[2] - NODE_H / 2
    parts.append(_node(NODE_XS[0], r3_top, "process",
                       "ILP writer",
                       body=("Influx Line", "Protocol")))

    # Terminal QuestDB node — spans columns 2-4 (wider) and uses the
    # `input` role to mark it as "the data source for every downstream
    # figure in the thesis". One sage bookend at each end of the diagram.
    qx = NODE_XS[1]
    qw = NODE_XS[3] + NODE_W - NODE_XS[1]   # spans cols 2..4
    parts.append(_card(x=qx, y=r3_top, w=qw, h=NODE_H, role="input"))
    parts.append(_label_set(
        x=qx, y=r3_top, w=qw,
        label="QuestDB",
        body_lines=("pdm_medical_device  ·  420M+ rows",),
        mono_lines=("time-series store",),
    ))

    # ── Within-row edges ──────────────────────────────────────────────
    def right_x(col):
        return NODE_XS[col] + NODE_W

    def left_x(col):
        return NODE_XS[col]

    # Row 1: 0->1, 1->2 ("MLLP"), 2->3
    parts.append(svg_edge(x1=right_x(0) + 4, y1=ROW_Y[0],
                          x2=left_x(1) - 4, y2=ROW_Y[0]))
    parts.append(svg_edge(x1=right_x(1) + 4, y1=ROW_Y[0],
                          x2=left_x(2) - 4, y2=ROW_Y[0],
                          label="MLLP"))
    parts.append(svg_edge(x1=right_x(2) + 4, y1=ROW_Y[0],
                          x2=left_x(3) - 4, y2=ROW_Y[0]))

    # Row 2: 0->1, 1->2, 2->3 (no labels — node names carry the flow).
    for c in range(3):
        parts.append(svg_edge(
            x1=right_x(c) + 4, y1=ROW_Y[1],
            x2=left_x(c + 1) - 4, y2=ROW_Y[1],
        ))

    # Row 3: ILP -> QuestDB (labeled).
    parts.append(svg_edge(x1=right_x(0) + 4, y1=ROW_Y[2],
                          x2=qx - 4, y2=ROW_Y[2],
                          label="ILP"))

    # ── Row-to-row U-turn connectors ──────────────────────────────────
    # Exit right edge of last node in row N, run a short elbow right,
    # drop to the midline of the row-gap, sweep all the way back to the
    # centreline of the next row's first node, drop into its top.
    def u_turn(row_from, row_to):
        x_exit = right_x(3) + 4
        x_elbow_right = x_exit + 4              # stays inside W=940 canvas
        y_top = ROW_Y[row_from]
        y_mid_gap = (TOP_MARGIN + (row_from + 1) * ROW_H
                     + row_from * ROW_GAP + ROW_GAP / 2)
        x_target = NODE_XS[0] + NODE_W / 2
        y_node_top = TOP_MARGIN + row_to * (ROW_H + ROW_GAP)
        d = (
            f"M {x_exit} {y_top} "
            f"L {x_elbow_right} {y_top} "
            f"L {x_elbow_right} {y_mid_gap} "
            f"L {x_target} {y_mid_gap} "
            f"L {x_target} {y_node_top - 4}"
        )
        return (
            f'<path d="{d}" fill="none" '
            f'stroke="{COLOR["forest"]}" stroke-width="1.25" '
            f'marker-end="url(#arrow)"/>'
        )

    parts.append(u_turn(0, 1))
    parts.append(u_turn(1, 2))

    return "".join(parts)


def main():
    body = build_pipeline()
    svg = svg_document(width=W, height=H, body=body)
    out = Path(__file__).with_suffix(".svg")
    out.write_text(svg, encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
