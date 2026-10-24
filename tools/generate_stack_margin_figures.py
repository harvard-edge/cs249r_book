#!/usr/bin/env python3
"""
generate_stack_margin_figures.py

Writes the HTML margin stack figure (images/svg/margin_stack.svg) for every
chapter in all four volumes.

The PDF draws the stack with a TikZ macro inside the chapter's Purpose
marginfigure (\\mlsysstack, \\mlfleetstack, \\mlagentstack,
\\mlphysicalstackfull). That macro call stays the single source of truth: this
script reads its weights from the .qmd and draws a compact native SVG sized for
the HTML margin column, with real text and a transparent background. Dark mode
inverts it in CSS (`_dark-mode-base.scss`), so only the light palette is drawn
here.

The chapter .qmd files are read, never written. The HTML block that displays
the SVG already sits next to each Purpose marginfigure.

Usage:
    python3 tools/generate_stack_margin_figures.py            # all volumes
    python3 tools/generate_stack_margin_figures.py vol1 vol3  # some volumes
"""

import glob
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOKS_DIR = REPO_ROOT / "books"

FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
WIDTH = 220
ROW_H = 22
GAP = 3
INK_MUTED = "#6B7280"

# Layer order is top to bottom, as drawn. `args` maps each layer to its
# position in the PDF macro's argument list (0-based).
VOLUMES = {
    "vol1": {
        "macro": "mlsysstack",
        "accent": "#A51C30",
        "alt": "Machine Learning Systems Architecture Stack",
        "layers": ["Applications", "Operations", "Serving", "Training",
                   "Models", "Frameworks", "Hardware"],
        "args": [6, 5, 4, 3, 2, 1, 0],
        "rail": {"label": "DATA", "arg": 7},
    },
    "vol2": {
        "macro": "mlfleetstack",
        "accent": "#1F407A",
        "alt": "Fleet Machine Learning Systems Stack",
        "layers": ["Governance", "Assurance", "Operations", "Serving",
                   "Control", "Execution", "Fabric", "Facility"],
        "args": [7, 6, 5, 4, 3, 2, 1, 0],
        "parts": ["IV", "IV", "III", "III", "II", "II", "I", "I"],
    },
    "vol3": {
        "macro": "mlagentstack",
        "accent": "#581C87",
        "alt": "Agentic Machine Learning Systems Stack",
        "layers": ["Distributed Fleet", "Trajectory Learning", "Agent Runtime",
                   "Tool Actuation", "Context Memory", "Foundation Model"],
        "tags": ["Network", "Compiler", "OS", "I/O", "Memory", "Processor"],
        "args": [0, 1, 2, 3, 4, 5],
        "rail": {"label": "TRAJECTORY", "arg": None},
    },
    "vol4": {
        "macro": "mlphysicalstackfull",
        "accent": "#065F46",
        "alt": "Physical AI Systems Stack",
        "layers": ["Brain", "Nervous System", "Body", "Boundary",
                   "Physical World"],
        "tags": ["1–50 Hz", "1 kHz", "10–25 kHz", "event", "continuous"],
        "args": [0, 1, 2, 3, 4],
        # Dashed rule after the first layer: above proposes, below commands.
        "divider_after": 0,
    },
}


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def tint(w):
    """Fill opacity for weight w in [0, 100].

    Tiles are the accent at partial opacity over a transparent background, so
    the dark-mode invert turns them into light-on-dark tints instead of tiles
    darker than the page.
    """
    return 0.10 + 0.90 * (max(0, min(100, w)) / 100.0)


def text(x, y, s, size, fill, weight=600, anchor="start", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" dominant-baseline="central"{extra}>'
            f'{escape(s)}</text>')


def render(vol, weights):
    spec = VOLUMES[vol]
    accent = spec["accent"]
    layers = spec["layers"]
    n = len(layers)
    vals = [weights[i] for i in spec["args"]]
    # A chapter that weights every layer equally has no focus; show it flat.
    flat = len(set(vals)) == 1
    rail_arg = spec.get("rail", {}).get("arg")
    top = max(vals + ([weights[rail_arg]] if rail_arg is not None else []))

    left = 26 if "parts" in spec else 0
    rail_w = 18 if "rail" in spec else 0
    right = WIDTH - (rail_w + 6 if rail_w else 0)
    height = n * ROW_H + (n - 1) * GAP
    if "divider_after" in spec:
        height += 8

    out = []
    y = 0.0
    rows = []
    for i, (name, w) in enumerate(zip(layers, vals)):
        focus = (not flat) and w == top and w >= 50
        level = 22 if flat else w
        alpha = 1.0 if focus else tint(level * 0.55)
        ink = "#FFFFFF" if focus else accent
        out.append(f'<rect x="{left}" y="{y:.1f}" width="{right - left}" '
                   f'height="{ROW_H}" rx="4" fill="{accent}" '
                   f'fill-opacity="{alpha:.2f}"/>')
        out.append(text(left + 9, y + ROW_H / 2, name, 11.5, ink,
                        700 if focus else 600))
        if "tags" in spec:
            out.append(text(right - 8, y + ROW_H / 2, spec["tags"][i], 9.5,
                            "#FFFFFF" if focus else INK_MUTED, 500, "end",
                            ' opacity="0.9"' if focus else ""))
        rows.append((y, y + ROW_H))
        y += ROW_H + GAP
        if spec.get("divider_after") == i:
            out.append(f'<line x1="{left}" x2="{right}" y1="{y + 2:.1f}" '
                       f'y2="{y + 2:.1f}" stroke="{INK_MUTED}" '
                       f'stroke-width="1" stroke-dasharray="3 2.5"/>')
            y += 8

    if "parts" in spec:
        parts = spec["parts"]
        i = 0
        while i < n:
            j = i
            while j + 1 < n and parts[j + 1] == parts[i]:
                j += 1
            y0, y1 = rows[i][0], rows[j][1]
            out.append(f'<line x1="{left - 7}" x2="{left - 7}" y1="{y0 + 2:.1f}" '
                       f'y2="{y1 - 2:.1f}" stroke="{accent}" stroke-width="1.5" '
                       f'stroke-linecap="round" opacity="0.55"/>')
            cy = (y0 + y1) / 2
            out.append(text(left - 12, cy, parts[i], 8, accent, 700, "end",
                            ' opacity="0.8"'))
            i = j + 1

    if "rail" in spec:
        rail = spec["rail"]
        x0 = WIDTH - rail_w
        w = weights[rail["arg"]] if rail["arg"] is not None else 100
        focus = (not flat) and rail["arg"] is not None and w == top and w >= 50
        if rail["arg"] is None:
            alpha, ink = tint(30), accent
        else:
            alpha = 1.0 if focus else tint((22 if flat else w) * 0.55)
            ink = "#FFFFFF" if focus else accent
        out.append(f'<rect x="{x0}" y="0" width="{rail_w}" height="{height:.1f}" '
                   f'rx="4" fill="{accent}" fill-opacity="{alpha:.2f}"/>')
        cx, cy = x0 + rail_w / 2, height / 2
        out.append(text(cx, cy, rail["label"], 9, ink, 700, "middle",
                        f' letter-spacing="1.2" transform="rotate(-90 {cx:.1f} {cy:.1f})"'))
        if rail["arg"] is None:
            # Trajectory runs upward through the stack.
            out.append(f'<path d="M{cx:.1f} 3 l-4 6 h8 z" fill="{accent}" opacity="0.7"/>')

    body = "\n".join(out)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{height:.0f}" viewBox="0 0 {WIDTH} {height:.1f}" '
            f'role="img" aria-label="{escape(spec["alt"])}">\n'
            f'<title>{escape(spec["alt"])}</title>\n{body}\n</svg>\n')


MACRO_RE = re.compile(r"\\(ml[a-z]+)((?:\{[^}]*\})+)")


def chapter_weights(qmd_text, macro):
    m = re.search(r"## Purpose.*?\\begin\{marginfigure\}(.*?)\\end\{marginfigure\}",
                  qmd_text, re.DOTALL)
    if not m:
        return None
    for name, args in MACRO_RE.findall(m.group(1)):
        if name == macro:
            return [int(float(a)) for a in re.findall(r"\{([^}]*)\}", args)]
    return None


def main(vols):
    written, skipped = 0, []
    for vol in vols:
        spec = VOLUMES[vol]
        for f in sorted(glob.glob(str(BOOKS_DIR / vol / "*" / "*.qmd"))):
            if not re.search(r"/\d\d_", f):
                continue
            svg_path = Path(f).parent / "images" / "svg" / "margin_stack.svg"
            content = Path(f).read_text(encoding="utf-8")
            if "images/svg/margin_stack.svg" not in content:
                continue
            weights = chapter_weights(content, spec["macro"])
            if weights is None:
                skipped.append(f)
                continue
            svg_path.parent.mkdir(parents=True, exist_ok=True)
            svg_path.write_text(render(vol, weights), encoding="utf-8")
            written += 1
    print(f"Wrote {written} margin stack SVGs.")
    for f in skipped:
        print(f"⚠️  No {VOLUMES[Path(f).parts[-3]]['macro']} call in Purpose: {f}")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or list(VOLUMES)))
