"""Cypresses-styled HL7 v2.x message anatomy. Figure 3.2 (\\ref{fig:hl7_anatomy}).

Brief (derived from thesis/main.tex lines 211-218 + the 2026-05-12 research
report on packet/message-anatomy figure conventions):

    Figure number:   3.2
    Chapter:         3 — Data and System Description
    Title:           Anatomy of an HL7 v2.x Telemetry Message
    Output format:   SVG, single figure float (no sub-figures)
    Composition:     Annotated specimen + side-mapping + byte ruler
                     in three vertically stacked rhetorical zones.
                     Single figure (not split) — all three zones answer
                     ONE question: "what does this message look like,
                     decoded?" Convention: Kuo et al. JAMIA Open 2019,
                     HL7 v2.2 Ch.2, Kurose & Ross *Computer Networking*.
    Zones:
      A (left+right top): Specimen (left) + Database Column Mapping (right)
        - Specimen is 7 stacked segment rows (MSH / PID / OBR + 4 OBX),
          each with a cypress-deep tag chip on the left and the literal
          pipe-delimited mono content to its right. ALL rows share the
          `input` semantic role; tone ladder (30 % / 22 % / 14 % mix
          over canvas-cream) differentiates segment families without
          burning extra role-hues on "tell segments apart".
        - Mapping is 6 process-role chips: HL7 field -> DB column.
        - 3 dashed olive-shadow leader lines tie OBX#1 → device_serial
          chip, OBX#2 → the four generic OBX-field chips (single fan),
          and OBX#4 → bitfield_* chip.
      B (full-width bottom): Bitfield byte ruler.
        - 8 equal cells, MSB-on-left, bit indices above, abbreviated
          flag names below. Evaluation-role (turquoise) cells —
          the decoded booleans are "the verification of the encoding".
        - "vars 802-804 encode 9 more booleans in the same way" caveat.

Source canvas: 940 x 700 user units (2026-05-13: height bumped from 600 to
700 to accommodate larger row heights needed by the 23 px font floor).
All text ≥ 23 px so rendered pt ≥ 11 at W=940, textwidth=6.3 in.
HL7 segment strings are abbreviated with "…" to fit in the available 422 px
content zone at 23 px / ~13.8 px per char — the key structural information
(tag, variable id, numeric value, unit) is preserved.

DEVIATION FROM DESIGN.md (record on top of v0.2):
  - Sage tone ladder at 30 % / 22 % / 14 % over canvas-cream (DESIGN.md
    §2.2 specifies a single 18 % mix per role). Justification: single-
    role tone variation is the standards-doc convention for
    sibling-parts-of-one-artifact (HL7 v2.2 Ch.2, IETF RFC header
    diagrams). Will write a v0.3 amendment if this pattern recurs.
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
# Height increased from 600 to 700 to accommodate 23 px font-size floor
# (larger row heights and wider byte-ruler cells).
W, H = 940, 700


# ── Sage tone ladder (one-off — see deviation note in docstring) ──────
def _mix(role_hex: str, pct: float, bg_hex: str = COLOR["canvas_cream"]) -> str:
    def hex_to_rgb(h):
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r1, g1, b1 = hex_to_rgb(role_hex)
    r2, g2, b2 = hex_to_rgb(bg_hex)
    r = round(r1 * pct + r2 * (1 - pct))
    g = round(g1 * pct + g2 * (1 - pct))
    b = round(b1 * pct + b2 * (1 - pct))
    return f"#{r:02X}{g:02X}{b:02X}"


SAGE_TONES = {
    "deep":  _mix(ROLE["input"], 0.30),
    "mid":   _mix(ROLE["input"], 0.22),
    "light": _mix(ROLE["input"], 0.14),
}


# ── Data (every literal string here is sourced from the existing
#    matplotlib generator at generate_thesis_figures.py:671 or directly
#    quoted from the thesis prose — nothing fabricated) ─────────────────
SEGMENT_DATA = [
    # (tag, tone_key, segment_text, descriptive_label)
    # Strings abbreviated with "…" to fit 422 px at font-size=23 (~13.8 px/char, 30-char max).
    ("MSH", "deep",
     "MSH|CAPSULE|SalviaA|ORU^R01|…",
     "Message Header"),
    ("PID", "mid",
     "PID|1||PATIENT_001|…",
     "Patient Identification"),
    ("OBR", "mid",
     "OBR|1|||Elisa800^MDIP|…",
     "Observation Request"),
    ("OBX", "light",
     "OBX|1|NM|1913||8204514|…",
     "Serial number (var 1913)"),
    ("OBX", "light",
     "OBX|2|NM|635||21.0|%|…",
     "FiO₂ measured (var 635)"),
    ("OBX", "light",
     "OBX|3|NM|2098||410.0|kPa|…",
     "O₂ supply pressure (var 2098)"),
    ("OBX", "light",
     "OBX|4|NM|801||255|…",
     "Bitfield (var 801)"),
]

MAPPING_CHIPS = [
    # (hl7_field, db_column)
    # "var 801…804" abbreviated to "v801…804" so it fits in the left chip zone at 23 px.
    ("OBX.3",       "variable_id"),
    ("OBX.5",       "value"),
    ("OBX.6",       "unit"),
    ("OBX.14",      "timestamp"),
    ("var 1913",    "device_serial"),
    ("v801…804",    "bitfield_*"),
]

# Bit-position → abbreviated flag name. The "_ok" suffix is universal
# (every bit means "sensor OK when set") and is called out in the caveat
# below the ruler, so it's dropped here to keep cell footers from
# colliding with their neighbours.
BITFIELD_FLAGS = {
    # Abbreviated to ≤ 7 chars so they fit in cell_w=100 at font-size=23 px
    # (~13.8 px/char × 7 = 96.6 px < 100 px cell width).
    0: "o2_flow",
    1: "o2_conc",
    2: "air_flw",
    3: "insp_fl",
    4: "exp_flw",
    5: "airwy_p",
    6: "baro_p",
    7: "o2_cell",
}


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))


def _section_title(*, x, y, text):
    # 23 px → rendered ≥ 11 pt at W=940, textwidth=6.3 in.
    return (
        f'<text x="{x}" y="{y}" '
        f'font-family="JetBrains Mono, monospace" font-size="23" '
        f'letter-spacing="1.8" font-weight="600" '
        f'fill="{COLOR["olive_shadow"]}">{text.upper()}</text>'
    )


def _specimen_row(*, x, y, w, h, tag, tone_key, mono_text, label):
    """One segment row: descriptive italic label above + tag chip + mono.

    All font sizes ≥ 23 px (rendered ≥ 11 pt at W=940, textwidth=6.3 in).
    Row height h must be ≥ 36 px to contain 23 px text with margins.
    """
    parts = []
    # Label above row (23 px, italic)
    parts.append(
        f'<text x="{x + 4}" y="{y - 6}" '
        f'font-family="Inter, sans-serif" font-size="23" font-style="italic" '
        f'fill="{COLOR["olive_shadow"]}">'
        f'<tspan font-weight="600">{tag}</tspan> — {_escape(label)}</text>'
    )
    # Row background (sage tone)
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" ry="3" '
        f'fill="{SAGE_TONES[tone_key]}" stroke="{ROLE["input"]}" '
        f'stroke-width="1.0"/>'
    )
    # Tag chip (cypress-deep with canvas-cream tag)
    chip_w = 52  # widened from 42 to 52 to fit 3-char tags at 23 px
    parts.append(
        f'<rect x="{x + 4}" y="{y + 4}" width="{chip_w}" height="{h - 8}" '
        f'rx="2" ry="2" fill="{COLOR["cypress_deep"]}" stroke="none"/>'
    )
    parts.append(
        f'<text x="{x + 4 + chip_w / 2}" y="{y + h / 2 + 8}" text-anchor="middle" '
        f'font-family="JetBrains Mono, monospace" font-size="23" font-weight="600" '
        f'fill="{COLOR["canvas_cream"]}">{tag}</text>'
    )
    # Mono message content
    parts.append(
        f'<text x="{x + chip_w + 18}" y="{y + h / 2 + 8}" '
        f'font-family="JetBrains Mono, monospace" font-size="23" '
        f'fill="{COLOR["cypress_deep"]}">{_escape(mono_text)}</text>'
    )
    return "".join(parts)


def _mapping_chip(*, x, y, w, h, hl7_field, db_column):
    """Two-element chip: HL7 field on left, → arrow, db column on right.

    All font sizes ≥ 23 px (rendered ≥ 11 pt). Chip height h should be ≥ 40.
    """
    parts = []
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" ry="4" '
        f'fill="{FILL_MIX["process"]}" stroke="{ROLE["process"]}" '
        f'stroke-width="{STROKE_WIDTH["process"]}"/>'
    )
    parts.append(
        f'<text x="{x + 14}" y="{y + h / 2 + 8}" '
        f'font-family="JetBrains Mono, monospace" font-size="23" font-weight="500" '
        f'fill="{COLOR["cypress_deep"]}">{_escape(hl7_field)}</text>'
    )
    arrow_x = x + w * 0.42
    parts.append(
        f'<text x="{arrow_x}" y="{y + h / 2 + 8}" text-anchor="middle" '
        f'font-family="Inter, sans-serif" font-size="23" '
        f'fill="{COLOR["forest"]}">→</text>'
    )
    parts.append(
        f'<text x="{arrow_x + 18}" y="{y + h / 2 + 8}" '
        f'font-family="JetBrains Mono, monospace" font-size="23" '
        f'fill="{COLOR["forest"]}">{_escape(db_column)}</text>'
    )
    return parts and "".join(parts)


def _leader(*, x1, y1, x2, y2):
    """Single dashed olive-shadow Bezier leader (no arrowhead — these are
    annotation leaders, not flow arrows)."""
    cx1 = x1 + 30
    cx2 = x2 - 30
    return (
        f'<path d="M {x1} {y1} C {cx1} {y1}, {cx2} {y2}, {x2} {y2}" '
        f'fill="none" stroke="{COLOR["olive_shadow"]}" stroke-width="1.0" '
        f'stroke-dasharray="3,3"/>'
    )


def _byte_ruler(*, x, y, w, value=255):
    """Horizontal byte ruler: 8 equal cells, MSB-on-left, bit indices above,
    flag names below. Kurose & Ross convention.

    All font sizes ≥ 23 px (rendered ≥ 11 pt at W=940, textwidth=6.3 in).
    cell_w=100 (up from 70) so 7-char flag names fit at 23 px (~13.8 px/char).
    """
    parts = []
    n = 8
    cell_w = 100   # increased from 70 to fit 23 px flag names (≤ 7 chars)
    cell_h = 48
    cell_gap = 4
    total_w = n * cell_w + (n - 1) * cell_gap
    start_x = x + (w - total_w) / 2
    binary_str = format(value, "08b")

    # Header line: "var 801 = 255 → binary 11111111"
    header_text = (
        f'var 801 = {value}  →  binary '
        f'{binary_str[:4]} {binary_str[4:]}'
    )
    parts.append(
        f'<text x="{x + w / 2}" y="{y + 18}" text-anchor="middle" '
        f'font-family="JetBrains Mono, monospace" font-size="23" font-weight="500" '
        f'fill="{COLOR["cypress_deep"]}">{header_text}</text>'
    )

    # 8 cells, bit 7 on the left.
    cells_top = y + 40
    for i in range(n):
        bit_index = n - 1 - i
        bit_value = binary_str[i]
        cx = start_x + i * (cell_w + cell_gap)
        # Bit index above (23 px)
        parts.append(
            f'<text x="{cx + cell_w / 2}" y="{cells_top - 8}" text-anchor="middle" '
            f'font-family="JetBrains Mono, monospace" font-size="23" '
            f'fill="{COLOR["olive_shadow"]}">bit {bit_index}</text>'
        )
        # Cell box (evaluation role — decoded booleans)
        parts.append(
            f'<rect x="{cx}" y="{cells_top}" width="{cell_w}" height="{cell_h}" '
            f'rx="3" ry="3" fill="{FILL_MIX["evaluation"]}" '
            f'stroke="{ROLE["evaluation"]}" stroke-width="{STROKE_WIDTH["evaluation"]}"/>'
        )
        # Bit value inside the cell (23 px bold)
        parts.append(
            f'<text x="{cx + cell_w / 2}" y="{cells_top + cell_h / 2 + 8}" '
            f'text-anchor="middle" '
            f'font-family="JetBrains Mono, monospace" font-size="23" font-weight="600" '
            f'fill="{COLOR["cypress_deep"]}">{bit_value}</text>'
        )
        # Flag name below (23 px) — abbreviated to ≤ 7 chars in BITFIELD_FLAGS
        parts.append(
            f'<text x="{cx + cell_w / 2}" y="{cells_top + cell_h + 26}" '
            f'text-anchor="middle" '
            f'font-family="JetBrains Mono, monospace" font-size="23" '
            f'fill="{COLOR["forest"]}">{BITFIELD_FLAGS[bit_index]}</text>'
        )

    caveat_y = cells_top + cell_h + 56
    parts.append(
        f'<text x="{x + w / 2}" y="{caveat_y}" text-anchor="middle" '
        f'font-family="Inter, sans-serif" font-size="23" font-style="italic" '
        f'fill="{COLOR["olive_shadow"]}">'
        f'each bit = 1 indicates the sensor is OK; '
        f'vars 802–804 encode 9 more booleans the same way</text>'
    )
    return "".join(parts)


def build_figure():
    parts = []

    # ── Specimen panel (left half) ────────────────────────────────────
    spec_x = 26
    spec_y = 66
    spec_w = 488
    row_h = 38           # increased from 32 to accommodate 23 px text
    row_v_gap = 32       # increased from 18 to give room for 23 px label above

    parts.append(_section_title(x=spec_x, y=spec_y - 24, text="Specimen — HL7 v2.x"))

    row_y_centres = []
    cur_y = spec_y
    for tag, tone, mono, label in SEGMENT_DATA:
        parts.append(_specimen_row(
            x=spec_x, y=cur_y, w=spec_w, h=row_h,
            tag=tag, tone_key=tone, mono_text=mono, label=label,
        ))
        row_y_centres.append(cur_y + row_h / 2)
        cur_y += row_h + row_v_gap
    spec_bottom = cur_y - row_v_gap

    # ── Mapping chips (right half) ────────────────────────────────────
    map_x = 548
    map_y = 66
    map_w = 366
    chip_h = 42          # increased from 36 to accommodate 23 px text
    chip_gap = 10

    parts.append(_section_title(x=map_x, y=map_y - 24, text="Database column mapping"))

    chip_y_centres = []
    cur_y = map_y
    for hl7, db in MAPPING_CHIPS:
        parts.append(_mapping_chip(
            x=map_x, y=cur_y, w=map_w, h=chip_h,
            hl7_field=hl7, db_column=db,
        ))
        chip_y_centres.append(cur_y + chip_h / 2)
        cur_y += chip_h + chip_gap

    # ── Leaders ───────────────────────────────────────────────────────
    # Three leader bundles (research recommendation: dashed leaders from
    # specimen to chips, not a 3-column table):
    #  (a) OBX#2 (var 635, the canonical numeric measurement OBX) fans to
    #      the four generic OBX-field chips (OBX.3 / .5 / .6 / .14).
    #  (b) OBX#1 (serial) → chip 5 (device_serial).
    #  (c) OBX#4 (bitfield 801) → chip 6 (bitfield_*).
    obx2_y = row_y_centres[4]
    obx1_y = row_y_centres[3]
    obx4_y = row_y_centres[6]
    leader_src_x = spec_x + spec_w + 4
    leader_dst_x = map_x - 4

    for chip_idx in range(4):
        parts.append(_leader(
            x1=leader_src_x, y1=obx2_y,
            x2=leader_dst_x, y2=chip_y_centres[chip_idx],
        ))
    parts.append(_leader(
        x1=leader_src_x, y1=obx1_y,
        x2=leader_dst_x, y2=chip_y_centres[4],
    ))
    parts.append(_leader(
        x1=leader_src_x, y1=obx4_y,
        x2=leader_dst_x, y2=chip_y_centres[5],
    ))

    # ── Byte ruler (full-width bottom) ────────────────────────────────
    br_x = 26
    br_y = max(spec_bottom, chip_y_centres[-1] + chip_h / 2) + 40
    br_w = W - 52
    parts.append(_section_title(x=br_x, y=br_y - 12, text="Bitfield decoding — variables 801…804"))
    parts.append(_byte_ruler(x=br_x, y=br_y + 6, w=br_w))

    return "".join(parts)


def main():
    body = build_figure()
    svg = svg_document(width=W, height=H, body=body)
    out = Path(__file__).with_suffix(".svg")
    out.write_text(svg, encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
