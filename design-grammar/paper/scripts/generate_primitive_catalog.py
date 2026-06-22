#!/usr/bin/env python3
"""
Generate a crisp vector SVG of the ML Systems Design Grammar primitive catalog
from the canonical YAML source of truth (design-grammar/grammar.yml).

The SVG follows the paper's typographic conventions:
  * Helvetica neue / Arial sans family
  * Role colors taken directly from the YAML (matches the website brand)
  * White background suitable for print and screen
  * Mathematical layout: 18 columns x 8 layers of square-ish primitive cells

Output: paper/figures/primitive_catalog.svg
The SVG is then rasterized to PDF by the existing Makefile rule via
`rsvg-convert -f pdf`.

Run:
  python3 paper/scripts/generate_primitive_catalog.py
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
YAML_PATH = REPO_DIR / "grammar.yml"
OUT_PATH = PAPER_DIR / "figures" / "primitive_catalog.svg"

# ── Layout constants (in SVG user units = px) ────────────────────────
# Sized so that the figure embeds cleanly at \textwidth (~504pt) on a
# two-column letter page. At that scale, font 22 prints as ~11pt.
CELL_W = 60
CELL_H = 60
LEFT_MARGIN = 78        # layer labels
RIGHT_MARGIN = 22
TOP_MARGIN = 96         # title + role headers
BOTTOM_MARGIN = 72      # legend + footnote

GRID_COLS = 18
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

roles = {r["key"]: r for r in data["roles"]}
layers = {r["index"]: r for r in data["layers"]}
primitives = data["primitives"]

# Index primitives by (layer, col) for fast lookup
grid: dict[tuple[int, int], dict] = {}
for primitive in primitives:
    grid[(primitive["layer"], primitive["col"])] = primitive

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
    f'ML Systems Design Grammar: Primitive Catalog'
    f'</text>'
)
parts.append(
    f'<text x="{CANVAS_W/2}" y="{title_y + 16}" text-anchor="middle" '
    f'font-size="9" font-weight="400" fill="{TEXT_SECONDARY}">'
    f'{len(primitives)} primitives \u2014 abstraction layers '
    f'\u00d7 information-processing roles (columns)'
    f'</text>'
)

# ── Role (column-group) headers ──────────────────────────────────────
header_y = TOP_MARGIN - 26
header_h = 18

for role in data["roles"]:
    col_indices = role["cols"]
    first_col = col_indices[0]
    last_col = col_indices[-1]
    n_cols = last_col - first_col + 1
    x = LEFT_MARGIN + (first_col - 1) * CELL_W
    w = n_cols * CELL_W
    color = role["color"]

    # Header bar
    parts.append(
        f'<rect x="{x + 2}" y="{header_y}" width="{w - 4}" height="{header_h}" '
        f'rx="3" fill="{color}"/>'
    )
    # Role name (white on color)
    parts.append(
        f'<text x="{x + w/2}" y="{header_y + 12}" text-anchor="middle" '
        f'font-size="10" font-weight="700" fill="#ffffff" letter-spacing="0.5">'
        f'{escape_xml(role["name"].upper())}'
        f'</text>'
    )

# ── Row labels (left side) ───────────────────────────────────────────
for layer_idx, layer in layers.items():
    cy = TOP_MARGIN + (layer_idx - 1) * CELL_H + CELL_H / 2
    parts.append(
        f'<text x="{LEFT_MARGIN - 12}" y="{cy + 4}" text-anchor="end" '
        f'font-size="11" font-weight="700" fill="{TEXT_PRIMARY}">'
        f'{escape_xml(layer["name"])}'
        f'</text>'
    )

# ── Primitive cells ──────────────────────────────────────────────────
# Prominent two-letter symbol, small ID in the corner.
# Full primitive names are enumerated in Section 2.1 of the paper, so the
# figure stays uncluttered and crisp at \textwidth.
for primitive in primitives:
    role_key = primitive["role"]
    color = roles[role_key]["color"]
    fill = tint(color, 0.20)
    stroke = darken(color, 0.85)

    col = primitive["col"]
    layer = primitive["layer"]
    x = LEFT_MARGIN + (col - 1) * CELL_W + 2
    y = TOP_MARGIN + (layer - 1) * CELL_H + 2
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
        f'{primitive["id"]}'
        f'</text>'
    )

    # Symbol (large, bold, centered)
    sym_y = y + h / 2 + 8
    parts.append(
        f'<text x="{x + w/2}" y="{sym_y}" text-anchor="middle" '
        f'font-size="22" font-weight="700" fill="{TEXT_PRIMARY}">'
        f'{escape_xml(primitive["sym"])}'
        f'</text>'
    )

# ── Bottom legend ────────────────────────────────────────────────────
legend_y = TOP_MARGIN + GRID_ROWS * CELL_H + 22
swatch = 14
legend_gap = 18

# Compute total width: per-role (swatch + 4 + name + " (sub)" )
def estimate_w(role: dict) -> float:
    name_chars = len(role["name"])
    sub_chars = len(role["sub"])
    return swatch + 5 + name_chars * 5.4 + 4 + sub_chars * 4.0

total_legend_w = sum(estimate_w(r) for r in data["roles"]) + legend_gap * (len(data["roles"]) - 1)
legend_x = (CANVAS_W - total_legend_w) / 2

for role in data["roles"]:
    color = role["color"]
    # Color swatch
    parts.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="{swatch}" height="{swatch}" '
        f'rx="2" fill="{color}"/>'
    )
    text_x = legend_x + swatch + 5
    parts.append(
        f'<text x="{text_x}" y="{legend_y + swatch - 3}" '
        f'font-size="9" font-weight="700" fill="{TEXT_PRIMARY}">'
        f'{escape_xml(role["name"])}'
        f' <tspan font-weight="400" fill="{TEXT_SECONDARY}" font-size="8.5">'
        f'\u2014 {escape_xml(role["sub"].lower())}'
        f'</tspan>'
        f'</text>'
    )
    legend_x += estimate_w(role) + legend_gap

# ── Footnote ─────────────────────────────────────────────────────────
foot_y = CANVAS_H - 12
parts.append(
    f'<text x="{CANVAS_W/2}" y="{foot_y}" text-anchor="middle" '
    f'font-size="8" font-style="italic" fill="{TEXT_MUTED}">'
    f'Each primitive satisfies the formal irreducibility criterion (Section 2.3); '
    f'the catalog supplies the primitive vocabulary for constraint-driven rewrite search.'
    f'</text>'
)

parts.append("</svg>")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text("\n".join(parts))
print(f"Wrote {OUT_PATH}  ({len(primitives)} primitives, {CANVAS_W}x{CANVAS_H})")
