#!/usr/bin/env python3
"""
Generate a crisp vector SVG of the Periodic Table of ML Systems
from the canonical YAML source of truth (periodic-table/table.yml).

The SVG follows the paper's typographic conventions:
  * Helvetica neue / Arial sans family
  * Block colors taken directly from the YAML (matches the website brand)
  * White background suitable for print and screen
  * Mathematical layout: 15 columns x 8 rows of square-ish element cells

Output: paper/figures/periodic_table_hero.svg

The SVG is then rasterized to PDF by the existing Makefile rule via
`rsvg-convert -f pdf`.

Run:
    python3 paper/scripts/generate_periodic_svg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")

# ── Paths ────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PAPER_DIR = HERE.parent
REPO_DIR = PAPER_DIR.parent
YAML_PATH = REPO_DIR / "table.yml"
OUT_PATH = PAPER_DIR / "figures" / "periodic_table_hero.svg"

# ── Layout constants (in SVG user units = px) ────────────────────────
# Sized so that the figure embeds cleanly at \textwidth (~504pt) on a
# two-column letter page. At that scale, font 22 prints as ~11pt.
CELL_W = 60
CELL_H = 60
LEFT_MARGIN = 78        # row labels
RIGHT_MARGIN = 22
TOP_MARGIN = 96         # title + block headers
BOTTOM_MARGIN = 72      # legend + footnote

GRID_COLS = 15
GRID_ROWS = 8

CANVAS_W = LEFT_MARGIN + GRID_COLS * CELL_W + RIGHT_MARGIN
CANVAS_H = TOP_MARGIN + GRID_ROWS * CELL_H + BOTTOM_MARGIN

# ── Colors ───────────────────────────────────────────────────────────
TEXT_PRIMARY = "#1a1a1a"
TEXT_SECONDARY = "#555"
TEXT_MUTED = "#888"
GRID_LINE = "#e8e8e8"

FONT_FAMILY = '"Helvetica Neue", Helvetica, Arial, sans-serif'

# ── Helpers ──────────────────────────────────────────────────────────
def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def tint(hex_color: str, alpha: float) -> str:
    """Blend the color toward white by (1 - alpha). Returns #rrggbb."""
    r, g, b = hex_to_rgb(hex_color)
    nr = round(r + (255 - r) * (1 - alpha))
    ng = round(g + (255 - g) * (1 - alpha))
    nb = round(b + (255 - b) * (1 - alpha))
    return f"#{nr:02x}{ng:02x}{nb:02x}"


def darken(hex_color: str, factor: float) -> str:
    r, g, b = hex_to_rgb(hex_color)
    return f"#{round(r * factor):02x}{round(g * factor):02x}{round(b * factor):02x}"


def escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ── Load YAML ────────────────────────────────────────────────────────
data = yaml.safe_load(YAML_PATH.read_text())

blocks = {b["key"]: b for b in data["blocks"]}
rows = {r["index"]: r for r in data["rows"]}
elements = data["elements"]

# Index elements by (row, col) for fast lookup
grid: dict[tuple[int, int], dict] = {}
for el in elements:
    grid[(el["row"], el["col"])] = el

# ── Begin SVG ────────────────────────────────────────────────────────
parts: list[str] = []
parts.append('<?xml version="1.0" encoding="UTF-8"?>')
parts.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" '
    f'font-family=\'{FONT_FAMILY}\'>'
)

# Background
parts.append(f'<rect width="{CANVAS_W}" height="{CANVAS_H}" fill="#ffffff"/>')

# ── Title ────────────────────────────────────────────────────────────
title_y = 28
parts.append(
    f'<text x="{CANVAS_W/2}" y="{title_y}" text-anchor="middle" '
    f'font-size="18" font-weight="700" fill="{TEXT_PRIMARY}">'
    f'The Periodic Table of Machine Learning Systems'
    f'</text>'
)
parts.append(
    f'<text x="{CANVAS_W/2}" y="{title_y + 16}" text-anchor="middle" '
    f'font-size="9" font-weight="400" fill="{TEXT_SECONDARY}">'
    f'{len(elements)} primitives \u2014 abstraction layers (rows) '
    f'\u00d7 information-processing roles (columns)'
    f'</text>'
)

# ── Block (column-group) headers ─────────────────────────────────────
header_y = TOP_MARGIN - 26
header_h = 18

for blk in data["blocks"]:
    col_indices = blk["cols"]
    first_col = col_indices[0]
    last_col = col_indices[-1]
    n_cols = last_col - first_col + 1
    x = LEFT_MARGIN + (first_col - 1) * CELL_W
    w = n_cols * CELL_W
    color = blk["color"]

    # Header bar
    parts.append(
        f'<rect x="{x + 2}" y="{header_y}" width="{w - 4}" height="{header_h}" '
        f'rx="3" fill="{color}"/>'
    )
    # Block name (white on color)
    parts.append(
        f'<text x="{x + w/2}" y="{header_y + 12}" text-anchor="middle" '
        f'font-size="10" font-weight="700" fill="#ffffff" letter-spacing="0.5">'
        f'{escape_xml(blk["name"].upper())}'
        f'</text>'
    )

# ── Row labels (left side) ───────────────────────────────────────────
for row_idx, row in rows.items():
    cy = TOP_MARGIN + (row_idx - 1) * CELL_H + CELL_H / 2
    parts.append(
        f'<text x="{LEFT_MARGIN - 12}" y="{cy + 4}" text-anchor="end" '
        f'font-size="11" font-weight="700" fill="{TEXT_PRIMARY}">'
        f'{escape_xml(row["name"])}'
        f'</text>'
    )

# ── Element cells ────────────────────────────────────────────────────
# Mendeleev-style: prominent two-letter symbol, small ID in the corner.
# Full element names are enumerated in Section 2.1 of the paper, so the
# figure stays uncluttered and crisp at \textwidth.
for el in elements:
    block_key = el["block"]
    color = blocks[block_key]["color"]
    fill = tint(color, 0.20)
    stroke = darken(color, 0.85)

    col = el["col"]
    row = el["row"]
    x = LEFT_MARGIN + (col - 1) * CELL_W + 2
    y = TOP_MARGIN + (row - 1) * CELL_H + 2
    w = CELL_W - 4
    h = CELL_H - 4

    # Cell background
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'rx="4" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>'
    )

    # ID number, top-right corner
    parts.append(
        f'<text x="{x + w - 4}" y="{y + 11}" text-anchor="end" '
        f'font-size="7" font-weight="400" fill="{TEXT_MUTED}">'
        f'{el["id"]}'
        f'</text>'
    )

    # Symbol (large, bold, centered)
    sym_y = y + h / 2 + 8
    parts.append(
        f'<text x="{x + w/2}" y="{sym_y}" text-anchor="middle" '
        f'font-size="22" font-weight="700" fill="{TEXT_PRIMARY}">'
        f'{escape_xml(el["sym"])}'
        f'</text>'
    )

# ── Bottom legend ────────────────────────────────────────────────────
legend_y = TOP_MARGIN + GRID_ROWS * CELL_H + 22
swatch = 14
legend_gap = 18

# Compute total width: per-block (swatch + 4 + name + " (sub)" )
def estimate_w(blk: dict) -> float:
    name_chars = len(blk["name"])
    sub_chars = len(blk["sub"])
    return swatch + 5 + name_chars * 5.4 + 4 + sub_chars * 4.0

total_legend_w = sum(estimate_w(b) for b in data["blocks"]) + legend_gap * (len(data["blocks"]) - 1)
legend_x = (CANVAS_W - total_legend_w) / 2

for blk in data["blocks"]:
    color = blk["color"]
    # Color swatch
    parts.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="{swatch}" height="{swatch}" '
        f'rx="2" fill="{color}"/>'
    )
    text_x = legend_x + swatch + 5
    parts.append(
        f'<text x="{text_x}" y="{legend_y + swatch - 3}" '
        f'font-size="9" font-weight="700" fill="{TEXT_PRIMARY}">'
        f'{escape_xml(blk["name"])}'
        f' <tspan font-weight="400" fill="{TEXT_SECONDARY}" font-size="8.5">'
        f'\u2014 {escape_xml(blk["sub"].lower())}'
        f'</tspan>'
        f'</text>'
    )
    legend_x += estimate_w(blk) + legend_gap

# ── Footnote ─────────────────────────────────────────────────────────
foot_y = CANVAS_H - 12
parts.append(
    f'<text x="{CANVAS_W/2}" y="{foot_y}" text-anchor="middle" '
    f'font-size="8" font-style="italic" fill="{TEXT_MUTED}">'
    f'Each element satisfies the formal irreducibility criterion (Section 2.3); '
    f'the table is the input to the Constraint-Driven Lowering Heuristic.'
    f'</text>'
)

parts.append("</svg>")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text("\n".join(parts))
print(f"Wrote {OUT_PATH}  ({len(elements)} elements, {CANVAS_W}x{CANVAS_H})")
