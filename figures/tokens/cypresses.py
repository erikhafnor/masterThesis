"""Cypresses design-system tokens for thesis illustrations.

This module is the single source of truth for the colour palette, role map,
and small SVG helpers used by every flowchart / explanatory illustration
in the thesis. The canonical spec is ``thesis/design/DESIGN.md``; if a
token disagrees with that file, the file wins and this module is wrong.

Use this module from any figure-generator script that emits SVG. For
matplotlib data plots, the legacy V5 baseline still applies (see
``llm-wiki/wiki/thesis/figure-style-guide.md``); the two systems are
intentionally separate.

Example:
    from thesis.tokens.cypresses import COLOR, ROLE, FILL_MIX, svg_node

    out = svg_node(
        x=40, y=40, w=200, h=80,
        role="process",
        label="Model Scoring",
        body="per-window, per-device",
    )
"""
from __future__ import annotations

# ── Palette (sampled from Van Gogh, *Cypresses*, 1889) ─────────────────
# These twelve hexes are the only colours permitted anywhere in a
# Cypresses-styled figure. No pure black, no pure white, no off-palette
# grey. See DESIGN.md §2.
COLOR = {
    "cypress_deep":  "#1B2713",  # darkest — borders, primary text, anomaly outline
    "forest":        "#2E3B21",  # secondary text, strong edges
    "cypress_mid":   "#414D2E",  # heading accents
    "olive_shadow":  "#5A613B",  # tertiary text, decision role
    "moss":          "#7A7949",  # alert role
    "olive_yellow":  "#989662",  # ranking role
    "slate_teal":    "#51776F",  # process role
    "sage":          "#699188",  # input role
    "turquoise":     "#7FAB9B",  # evaluation role
    "sky_mint":      "#9CC4AE",  # subtle fills
    "stone_cream":   "#B8B69E",  # neutral fill
    "canvas_cream":  "#CCCDB3",  # page background — THE canvas
}

# ── Role map (DESIGN.md §4) ────────────────────────────────────────────
# Every node carries exactly one role. The role chooses the stroke colour
# and the fill mix percentage.
ROLE = {
    "input":      COLOR["sage"],
    "process":    COLOR["slate_teal"],
    "ranking":    COLOR["olive_yellow"],
    "decision":   COLOR["olive_shadow"],
    "alert":      COLOR["moss"],
    "evaluation": COLOR["turquoise"],
    "anomaly":    COLOR["cypress_deep"],
}

# ── Precomputed fill mixes ─────────────────────────────────────────────
# DESIGN.md §2.2: a role's fill is the role colour mixed at 18 % (mid
# roles) or 22-24 % (alert / ranking) over canvas-cream. Never use the
# role colour at 100 % as a fill. Mixed once here so callers don't.
def _mix(role_hex: str, pct: float, bg_hex: str = COLOR["canvas_cream"]) -> str:
    """Linear sRGB mix of ``role_hex`` at ``pct`` (0-1) over ``bg_hex``."""
    def hex_to_rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r1, g1, b1 = hex_to_rgb(role_hex)
    r2, g2, b2 = hex_to_rgb(bg_hex)
    r = round(r1 * pct + r2 * (1 - pct))
    g = round(g1 * pct + g2 * (1 - pct))
    b = round(b1 * pct + b2 * (1 - pct))
    return f"#{r:02X}{g:02X}{b:02X}"


FILL_MIX = {
    "input":      _mix(ROLE["input"],      0.18),
    "process":    _mix(ROLE["process"],    0.18),
    "ranking":    _mix(ROLE["ranking"],    0.24),
    "decision":   COLOR["canvas_cream"],   # decisions have no fill mix
    "alert":      _mix(ROLE["alert"],      0.22),
    "evaluation": _mix(ROLE["evaluation"], 0.20),
}

# Stroke widths per role. Amended 2026-05-12 (DESIGN.md v0.2) — bumped from
# the original 1.25/1.5/1.75 set so role identity carries through stroke
# weight even when fills stay light (Fig 4.4 redesign feedback).
STROKE_WIDTH = {
    "input":      1.6,
    "process":    1.6,
    "ranking":    1.6,
    "decision":   1.8,
    "alert":      2.2,
    "evaluation": 1.6,
    "anomaly":    2.0,
}

# ── Typography (DESIGN.md §3) ──────────────────────────────────────────
# Sizes given in points; convert to pixels at 1 pt = 1.333 px (96 dpi)
# only for screen mock-ups.
#
# Print-floor note (2026-05-13): all SVG source sizes are chosen so that
# every text element renders at ≥ 11 pt in the compiled PDF.  At W=940 px
# and \textwidth = 6.3 in the scale is 6.3/9.792 ≈ 0.643, giving
# rendered_pt = source_px × (6.3 × 72) / 940 = source_px × 0.4826.
# Minimum source px = ceil(11 / 0.4826) = 23.  All TYPE entries below
# and all hardcoded font-size= strings in the svg_* helpers reflect this.
TYPE = {
    "figtitle":   {"family": "EB Garamond", "size_pt": 14,  "weight": 500},
    "subtitle":   {"family": "EB Garamond", "size_pt": 11,  "weight": 400, "italic": True},
    "nodelabel":  {"family": "Inter",       "size_pt": 10,  "weight": 600},
    "nodebody":   {"family": "Inter",       "size_pt": 9,   "weight": 400},
    "edgelabel":  {"family": "Inter",       "size_pt": 8.5, "weight": 400, "italic": True},
    "annotation": {"family": "JetBrains Mono", "size_pt": 8.5, "weight": 400},
    "caption":    {"family": "EB Garamond", "size_pt": 9.5, "weight": 400, "italic": True},
}

# ── Spacing (DESIGN.md §5) ─────────────────────────────────────────────
SPACE = {
    "1": 4, "2": 8, "3": 12, "4": 16, "5": 24,
    "6": 32, "7": 48, "8": 64,
}

NODE = {
    "radius":          4,
    "stroke_default":  1.25,
    "stroke_strong":   1.75,
    "padding_x":       SPACE["4"],
    "padding_y":       SPACE["3"],
    "min_w":           160,
    "min_h":           64,
}

ARROW = {"len": 9, "width": 7}


# ── SVG helpers ────────────────────────────────────────────────────────
def svg_arrow_marker_defs() -> str:
    """Reusable <marker> blocks for default / strong / feedback arrowheads.

    Geometry: a "stealth" arrowhead — longer than wide (10 × 5, ratio 2:1)
    with a concave back at x=2.5 — gives the refined academic-figure look
    matching TikZ's `Stealth` style. Replaces the chunky 9 × 7 filled
    triangle from v0.1 (Erik review 2026-05-12: "currently they look like
    AI-slop drawing").
    """
    return f"""<defs>
  <marker id="arrow" viewBox="0 0 10 5" refX="10" refY="2.5"
          markerWidth="10" markerHeight="5" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L10,2.5 L0,5 L2.5,2.5 Z" fill="{COLOR['forest']}"/>
  </marker>
  <marker id="arrow-strong" viewBox="0 0 10 5" refX="10" refY="2.5"
          markerWidth="10" markerHeight="5" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L10,2.5 L0,5 L2.5,2.5 Z" fill="{COLOR['cypress_deep']}"/>
  </marker>
  <marker id="arrow-feedback" viewBox="0 0 10 5" refX="10" refY="2.5"
          markerWidth="10" markerHeight="5" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L10,2.5 L0,5 L2.5,2.5 Z" fill="{COLOR['olive_shadow']}"/>
  </marker>
</defs>"""


def svg_node(*, x: float, y: float, w: float, h: float, role: str,
             label: str, body: str | None = None,
             anomaly: bool = False) -> str:
    """A Cypresses-styled rectangular node (DESIGN.md §7).

    Args:
        x, y, w, h: top-left corner and size in SVG user units (≈ px).
        role: one of the keys in :data:`ROLE`.
        label: bold label, rendered in Inter 10 pt / 600.
        body: optional second line, Inter 9 pt / 400.
        anomaly: if True, applies the anomaly treatment from DESIGN.md §4.1.
    """
    stroke = COLOR["cypress_deep"] if anomaly else ROLE[role]
    stroke_w = NODE["stroke_strong"] if anomaly else STROKE_WIDTH[role]
    fill = FILL_MIX[role]
    rect = (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'rx="{NODE["radius"]}" ry="{NODE["radius"]}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}"/>'
    )
    halo = ""
    if anomaly:
        # 1 px dashed halo at +4 px offset, in cypress-deep.
        halo = (
            f'<rect x="{x-4}" y="{y-4}" width="{w+8}" height="{h+8}" '
            f'rx="{NODE["radius"]+2}" ry="{NODE["radius"]+2}" '
            f'fill="none" stroke="{COLOR["cypress_deep"]}" '
            f'stroke-width="1" stroke-dasharray="3,3"/>'
        )
    # Text positioning: label on top, body below if present.
    # font-size=23 -> rendered ≥ 11 pt at W=940, textwidth=6.3 in.
    label_y = y + NODE["padding_y"] + 18  # baseline for 23 px label
    label_svg = (
        f'<text x="{x + NODE["padding_x"]}" y="{label_y}" '
        f'font-family="Inter, sans-serif" font-size="23" font-weight="600" '
        f'fill="{COLOR["cypress_deep"]}">{_escape(label)}</text>'
    )
    body_svg = ""
    if body:
        body_y = label_y + 26
        body_svg = (
            f'<text x="{x + NODE["padding_x"]}" y="{body_y}" '
            f'font-family="Inter, sans-serif" font-size="23" '
            f'fill="{COLOR["forest"]}">{_escape(body)}</text>'
        )
    return halo + rect + label_svg + body_svg


def svg_decision(*, cx: float, cy: float, w: float = 170, h: float = 130,
                 text: str) -> str:
    """Decision diamond (DESIGN.md §7.5). ``cx``/``cy`` is the centre."""
    hw, hh = w / 2, h / 2
    points = f"{cx},{cy - hh} {cx + hw},{cy} {cx},{cy + hh} {cx - hw},{cy}"
    poly = (
        f'<polygon points="{points}" '
        f'fill="{COLOR["canvas_cream"]}" '
        f'stroke="{ROLE["decision"]}" stroke-width="{STROKE_WIDTH["decision"]}"/>'
    )
    # Text inside, centred. Inter 23 px / 600 / cypress-deep (≥ 11 pt rendered).
    txt = (
        f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="middle" '
        f'font-family="Inter, sans-serif" font-size="23" font-weight="600" '
        f'fill="{COLOR["cypress_deep"]}">{_escape(text)}</text>'
    )
    return poly + txt


def svg_edge(*, x1: float, y1: float, x2: float, y2: float,
             style: str = "default", label: str | None = None) -> str:
    """Edge between two points (DESIGN.md §7.3).

    Args:
        style: one of ``"default"``, ``"strong"``, ``"feedback"``.
        label: optional italic label, rendered on a canvas-cream halo
            at the midpoint.
    """
    if style == "strong":
        stroke = COLOR["cypress_deep"]
        sw = 1.75
        marker = "arrow-strong"
        dash = ""
    elif style == "feedback":
        stroke = COLOR["olive_shadow"]
        sw = 1.25
        marker = "arrow-feedback"
        dash = 'stroke-dasharray="4,3"'
    else:
        stroke = COLOR["forest"]
        sw = 1.25
        marker = "arrow"
        dash = ""

    line = (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{stroke}" stroke-width="{sw}" {dash} '
        f'marker-end="url(#{marker})"/>'
    )

    label_svg = ""
    if label:
        # Position labels ABOVE the arrow line (not on it) so they don't
        # bleed into the node borders when gutters are narrow. The arrow
        # remains visible underneath; the label sits in clear canvas-cream.
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - 14
        approx_w = 6.8 * len(label)
        approx_w = 14.0 * len(label)  # ~14 px/char at font-size=23
        label_svg = (
            f'<rect x="{mx - approx_w / 2 - 2}" y="{my - 18}" '
            f'width="{approx_w + 4}" height="22" '
            f'fill="{COLOR["canvas_cream"]}" stroke="none"/>'
            f'<text x="{mx}" y="{my}" text-anchor="middle" '
            f'font-family="Inter, sans-serif" font-size="23" '
            f'font-style="italic" fill="{COLOR["olive_shadow"]}">'
            f'{_escape(label)}</text>'
        )
    return line + label_svg


def svg_title_block(*, x: float, y: float, width: float,
                    eyebrow: str, title: str,
                    subtitle: str | None = None) -> str:
    """Title block (DESIGN.md §8.2): eyebrow + serif title + optional
    italic subtitle, with a 1 px olive-shadow rule below."""
    # All font sizes ≥ 23 px so rendered pt ≥ 11 at W=940, textwidth=6.3 in.
    eyebrow_y = y + 18
    title_y   = eyebrow_y + 34
    rule_y    = title_y + 22
    subtitle_y = title_y + 28
    eyebrow_svg = (
        f'<text x="{x}" y="{eyebrow_y}" '
        f'font-family="JetBrains Mono, monospace" font-size="23" '
        f'letter-spacing="1.8" fill="{COLOR["olive_shadow"]}" '
        f'font-weight="500">{_escape(eyebrow.upper())}</text>'
    )
    title_svg = (
        f'<text x="{x}" y="{title_y}" '
        f'font-family="EB Garamond, serif" font-size="28" font-weight="600" '
        f'fill="{COLOR["cypress_deep"]}">{_escape(title)}</text>'
    )
    sub_svg = ""
    final_rule_y = rule_y
    if subtitle:
        sub_svg = (
            f'<text x="{x}" y="{subtitle_y}" '
            f'font-family="EB Garamond, serif" font-size="23" font-style="italic" '
            f'fill="{COLOR["forest"]}">{_escape(subtitle)}</text>'
        )
        final_rule_y = subtitle_y + 18
    rule_svg = (
        f'<line x1="{x}" y1="{final_rule_y}" x2="{x + width}" y2="{final_rule_y}" '
        f'stroke="{COLOR["olive_shadow"]}" stroke-width="1"/>'
    )
    return eyebrow_svg + title_svg + sub_svg + rule_svg


def svg_page_background(width: float, height: float) -> str:
    """Canvas-cream page background, always emitted as the first child."""
    return (
        f'<rect x="0" y="0" width="{width}" height="{height}" '
        f'fill="{COLOR["canvas_cream"]}"/>'
    )


def svg_document(*, width: float, height: float, body: str) -> str:
    """Wrap ``body`` in a full SVG with arrow defs + canvas-cream background.

    Width / height are in SVG user units (≈ px at 1:1 zoom). For print,
    LaTeX scales ``\\includegraphics[width=\\textwidth]``.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
     viewBox="0 0 {width} {height}">
{svg_arrow_marker_defs()}
{svg_page_background(width, height)}
{body}
</svg>"""


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


def svg_sustained_waveform(*, cx: float, cy: float, w: float = 150,
                            h: float = 60, n_windows: int = 5,
                            n_above: int = 3) -> str:
    """Inline mini-waveform illustrating the sustained-exceedance criterion.

    Draws ``n_windows`` evenly spaced vertical bars with a horizontal dashed
    threshold line at mid-height. The last ``n_above`` bars cross the
    threshold (taller); the earlier bars stay below it (shorter). The
    crossing bars are stroked in the focal role colour and the threshold
    line is in olive-yellow (matching the ranking role used upstream).

    Args:
        cx, cy: centre of the waveform area.
        w, h: width and height of the waveform area in user units.
        n_windows: total number of 5-min windows shown.
        n_above: trailing windows that cross the threshold (>=3 satisfies
            the sustained criterion).
    """
    x0, y0 = cx - w / 2, cy - h / 2
    threshold_y = y0 + h * 0.40  # threshold sits in the upper third
    baseline_y = y0 + h
    bar_w = (w * 0.6) / n_windows
    gap = (w - bar_w * n_windows) / (n_windows + 1)
    bars = []
    for i in range(n_windows):
        x = x0 + gap + i * (bar_w + gap)
        if i >= n_windows - n_above:
            # Crossing bar: extends from baseline above the threshold.
            top_y = y0 + h * 0.18
            stroke = COLOR["cypress_deep"]
            fill = FILL_MIX["alert"]
        else:
            # Below-threshold bar.
            top_y = y0 + h * 0.62
            stroke = ROLE["evaluation"]
            fill = FILL_MIX["evaluation"]
        bars.append(
            f'<rect x="{x}" y="{top_y}" width="{bar_w}" height="{baseline_y - top_y}" '
            f'rx="1" ry="1" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
        )
    # Threshold line (dashed, olive-yellow ranking colour).
    threshold = (
        f'<line x1="{x0}" y1="{threshold_y}" x2="{x0 + w}" y2="{threshold_y}" '
        f'stroke="{ROLE["ranking"]}" stroke-width="1.2" stroke-dasharray="4,3"/>'
    )
    return threshold + "".join(bars)


__all__ = [
    "COLOR", "ROLE", "FILL_MIX", "STROKE_WIDTH",
    "TYPE", "SPACE", "NODE", "ARROW",
    "svg_arrow_marker_defs", "svg_node", "svg_decision",
    "svg_edge", "svg_title_block", "svg_page_background", "svg_document",
    "svg_sustained_waveform",
]
