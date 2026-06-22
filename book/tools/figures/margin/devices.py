#!/usr/bin/env python3
"""
devices.py - the canonical margin-figure renderers, in the locked visual
language (.claude/rules/figure-visual-language.md + margin-figures.md).

This is the single source of truth for HOW a margin device is drawn. Insertion
work imports these; do not re-implement device shapes elsewhere.

Editor boundary:
  * Change this module for shared device behavior: geometry, label placement,
    semantic colors, font bands, native size, or SVG export.
  * Change generate_margin_figures.py for one figure's labels, values, device
    choice, chapter path, or asset name.
  * Change the QMD .column-margin block for prose placement, caption, and alt text.

Contract enforced here:
  • figure_style.set_book_style() first  → book font (Helvetica) + semantic palette
  • margin overrides: font.size ~5.5pt, axes.grid OFF, native ~1.25in width
  • font bands: ordinary labels 5.0–5.5pt; in-bar numbers 4.7–5.2pt bold;
    tiny scale cues 3.9–4.6pt; emphasis labels max 6.0pt
  • labels stay inside the visual frame and never sit directly on strokes,
    markers, arrows, or bar edges; move labels to whitespace before shrinking
  • SACRED COLOR: red (RedLine/RedFill) ONLY on danger/limit/fault — never a series/category
      memory=BlueLine · compute=OrangeLine · data=GreenLine · network/coupling=VioletLine
      neutral=grid/primary · selection accent (non-resource)=crimson
  • numbers go INSIDE bars (white, bold); short bars label just outside in dark
  • one canonical shape per concept (the "lite" margin form of the body figure)

Usage:
    from book.tools.figures.margin.devices import new_fig, ladder, knee, sparkline, roofline, \
        ironbar, dam, taxonomy, blast, budget_envelope, sequence_strip, \
        causal_chain, all_to_all_topology, save
    fig, ax = new_fig('hierarchy-ladder')
    ladder(ax, [("HBM",3350),("DRAM",100),("NVMe",7),("SSD",1),("net",0.1)])
    save(fig, "out.png")            # PNG draft; emit PDF/SVG for the real build

Data devices (ladder/knee/sparkline/roofline/ironbar) must be fed by the same
source of truth as the adjacent prose. New production figures should import or
derive from MLSysIM/LEGO outputs where practical; legacy hard-coded constants are
allowed only with an audit note and should be migrated in a later SSOT pass.
"""
import os
import re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from book.tools.figures import style as figure_style

C = figure_style.COLORS
MEM, COMP, DATA, NET = C['BlueLine'], C['OrangeLine'], C['GreenLine'], C['VioletLine']
RED, REDFILL = C['RedLine'], C['RedFill']
GRID, INK, SEL = C['grid'], C['primary'], C['crimson']  # SEL = non-resource selection accent
GREENFILL = C['GreenFill']  # positive-divergence gap fill (keeps RED sacred)
TIME = "#5E6B73"  # slate — time/latency/rate (a neutral backdrop dimension, not a spent resource)

# ── the semantic color contract, keyed by what a ladder MEASURES (unit decides) ──
#   bytes (capacity)     → MEM  blue    : kv-cache, optimizer state, device RAM
#   bytes/sec (bandwidth)→ NET  violet  : interconnect, storage-hierarchy bandwidth
#   joules/watts (energy)→ COMP orange  : pJ/op, deployment power; compute-intensity
#   seconds (time)       → TIME slate   : feedback cadence, MTBF
#   meters (reach)       → NET  violet  : physical interconnect reach / fabric geometry
# RED is never a domain hue — it stays sacred for danger/limit/selection.
DOMAIN_COLOR = {
    'memory': MEM, 'capacity': MEM,
    'bandwidth': NET, 'network': NET,
    'distance': NET, 'reach': NET, 'physical': NET,
    'energy': COMP, 'power': COMP, 'compute': COMP,
    'data': DATA, 'count': DATA,
    'time': TIME, 'latency': TIME, 'rate': TIME,
}

FONT_BASE = 5.5
FONT_ORDINARY = (5.0, 5.5)
FONT_IN_BAR = (4.7, 5.2)
FONT_TINY_CUE = (3.9, 4.6)
FONT_EMPHASIS_MAX = 6.0

# per-device native margin size (~1.25in wide; rendered at width="100%")
FIGSIZE = {
    'hierarchy-ladder': (1.25, 1.75), 'scale-anchor': (1.3, 1.05),
    'sparkline-trend': (1.3, 0.85), 'thumbnail-roofline': (1.3, 1.15),
    'iron-law-bar': (1.4, 0.5), 'dam-locator': (1.05, 1.15),
    'taxonomy-mini': (1.1, 1.1), 'blast-radius': (1.2, 1.0),
    'budget-envelope': (1.22, 0.72), 'sequence-strip': (1.20, 0.70),
    'causal-chain': (1.20, 0.78), 'all-to-all-topology': (1.15, 0.98),
    'other-new': (1.1, 1.0),
}

def new_fig(device):
    """set_book_style() (book Helvetica stack + semantic palette) + margin overrides
    (~5.5pt, no grid) + the device's native figsize. `svg.fonttype='path'` OUTLINES
    the Helvetica labels into vector paths in the SVG, so they render identically in
    the HTML site and after the Linux SVG->PDF conversion — no fontconfig/Helvetica
    dependency on the build/CI machine (matches the book's vector-figure fidelity)."""
    figure_style.set_book_style()
    plt.rcParams.update({
        'font.size': FONT_BASE,
        'axes.grid': False,
        'svg.fonttype': 'path',
        'svg.hashsalt': 'mlsysbook-margin-figures',
    })
    fig, ax = plt.subplots(figsize=FIGSIZE.get(device, (1.2, 1.1)), dpi=300)
    return fig, ax

def _clean(ax, keep=()):
    for s in ("top", "right", "left", "bottom"):
        if s not in keep:
            ax.spines[s].set_visible(False)
        else:
            ax.spines[s].set_color(GRID)
            ax.spines[s].set_linewidth(0.55)
    ax.set_xticks([]); ax.set_yticks([])

def save(fig, path, pad=0.02):
    """Emit PRODUCTION VECTOR SVG into the house images/svg/ dir — the book's
    established vector convention (141 body SVGs live there; Quarto auto-converts
    SVG->PDF for print via rsvg/inkscape, and HTML uses the SVG directly). Callers pass
    the logical .../images/png/<name>.png path; we write the .svg twin so margin figures
    are resolution-independent in BOTH the website and the print PDF. Fonts are outlined
    (new_fig sets svg.fonttype='path') so labels are identical everywhere. Returns the
    svg path written."""
    svg_path = path.replace(os.sep + "images" + os.sep + "png" + os.sep,
                            os.sep + "images" + os.sep + "svg" + os.sep)
    svg_path = os.path.splitext(svg_path)[0] + ".svg"
    os.makedirs(os.path.dirname(svg_path), exist_ok=True)
    fig.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        facecolor="white",
        pad_inches=pad,
        metadata={"Date": None},
    )
    with open(svg_path, "r", encoding="utf-8") as f:
        svg = f.read()
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
    plt.close(fig)
    return svg_path

# ── magnitude-span → hierarchy-ladder ──────────────────────────────────────────
def ladder(ax, tiers, wall=False, color=None, domain=None, style='bars'):
    """tiers=[(label,value),...]; rungs top(biggest)→bottom, label INSIDE the bar
    (white bold) when it fits, else just outside in dark ink; an optional red ceiling
    line marks a limit (sacred red).

    SCALE IS ADAPTIVE so bars read honestly:
      • range ≤ 25× → LINEAR axis: bar lengths are visually PROPORTIONAL (to scale),
        so a 6× difference looks 6× (e.g. optimizer 2100 GB vs weights 350 GB).
      • range > 25× → LOG axis: keeps the smallest rung visible across many orders of
        magnitude (e.g. energy 640→0.5 pJ, power 3 MW→50 mW), where linear would
        collapse small rungs to invisible slivers.

    COLOR ENCODES WHAT THE LADDER MEASURES — the UNIT decides (see DOMAIN_COLOR):
      bytes→MEM blue, bytes/s→NET violet, joules/watts→COMP orange, seconds→TIME slate.
    Pass `domain=` ('memory'|'bandwidth'|'energy'|'time'|…) and the renderer picks the
    book-consistent hue, so a reader learns 'orange = energy' across all chapters. The
    generator declares MEANING; the renderer owns the hue. `color=` overrides outright.
    Default (neither given) is MEM blue."""
    tiers = sorted(tiers, key=lambda r: r[1], reverse=True)
    n = len(tiers); vals = [t[1] for t in tiers]
    c = color or DOMAIN_COLOR.get(domain, MEM)
    log_scale = (max(vals) / min(vals)) > 25
    ax.set_ylim(-0.4, n - 0.4)
    h = 0.98 if style == 'staircase' else 0.66  # staircase = contiguous (a hierarchy of levels)

    def _rung(yy, v, left):
        if style == 'lollipop':                          # value as a POSITION on the scale
            ax.hlines(yy, left, v, color=GRID, lw=0.65); ax.plot(v, yy, "o", color=c, ms=4.2)
        else:                                            # 'bars' (separated) / 'staircase' (contiguous)
            ax.barh(yy, v, left=left, height=h, color=c, alpha=0.92)

    def _clean_label(lab):
        lab = str(lab).strip()
        lab = re.sub(
            r"(?<=\d)(?=(?:TB/s|GB/s|MB/s|GiB|GB|MB|KB|ms|us|ns|yr|h|pJ|nJ|uJ|mW|kW|MW|W|tCO2)\b)",
            " ",
            lab,
        )
        return "\n".join(part[:1].upper() + part[1:] if part[:1].islower() else part for part in lab.split("\n"))

    def _label(lab, v, yy, frac, inside_x, out_x):
        lab = _clean_label(lab)
        # Margin labels need a conservative fit test; a bar can be visibly
        # substantial yet still too short for a right-aligned Helvetica label.
        if style != 'lollipop' and frac > 0.040 * len(lab) + 0.08:   # fits inside the bar
            ax.text(inside_x, yy, lab, fontsize=5.2, va="center", ha="right",
                    color="white", fontweight="bold")
        else:                                            # outside (always for lollipop), dark ink
            ax.text(out_x, yy, lab, fontsize=5.2, va="center", ha="left",
                    color=INK, fontweight="bold")

    if log_scale:
        xmin = min(vals) * 0.4; xmax = max(vals) * 2.2
        ax.set_xscale("log"); ax.set_xlim(xmin, xmax)
        span = np.log10(xmax) - np.log10(xmin)
        for i, (lab, v) in enumerate(tiers):
            yy = n - 1 - i
            _rung(yy, v if style == 'lollipop' else v - xmin, xmin)
            _label(lab, v, yy, (np.log10(v) - np.log10(xmin)) / span, v * 0.92, v * 1.55)
        if wall:
            ax.plot([xmin, xmax], [n - 0.45, n - 0.45], color=RED, lw=0.75)
        ax.text(xmax, -0.32, "log scale", ha="right", va="bottom",
                color="#777777", fontsize=3.9)
    else:
        xmax = max(vals) * 1.12; pad = 0.015 * xmax
        ax.set_xlim(0, xmax)
        for i, (lab, v) in enumerate(tiers):
            yy = n - 1 - i
            _rung(yy, v, 0)
            _label(lab, v, yy, v / xmax, v - pad, v + pad)
        if wall:
            ax.plot([0, xmax], [n - 0.45, n - 0.45], color=RED, lw=0.75)
    _clean(ax, keep=("bottom",))
    ax.tick_params(axis="x", which="both", length=0)  # baseline spine, no tick marks

# ── threshold-knee → scale-anchor ──────────────────────────────────────────────
def knee(ax, knee_frac=0.75, style='shaded', pct_label=None):
    """one bent curve hitting a knee (red owns danger). Three self-evident variants:
      • 'shaded' (default) → red danger WASH right of the knee: a region to avoid.
      • 'dashed'           → a dashed red threshold line + the % label: a PRECISE cutoff
                             (when the exact number, e.g. ρ=70%, is the point). Pass
                             pct_label to override the text (defaults to knee_frac%).
      • 'twotone'          → the curve itself recolors green→red at the knee: a
                             safe→danger REGIME CHANGE (no zone, the line carries it)."""
    r = np.linspace(0, 0.97, 200); lat = 1 / (1 - r)
    kx = knee_frac * 100; ky = 1 / (1 - knee_frac)
    if style == 'twotone':
        m = r < knee_frac
        ax.plot(r[m] * 100, lat[m], color=DATA, lw=1.35)     # safe (green)
        ax.plot(r[~m] * 100, lat[~m], color=RED, lw=1.35)    # danger (red)
    else:
        ax.plot(r * 100, lat, color=INK, lw=1.35)
        if style == 'dashed':
            ax.axvline(kx, color=RED, lw=0.65, ls="--")
            ax.text(max(kx - 5, 4), ky + 5, pct_label or ("%g%%" % kx), fontsize=5.4,
                    color=RED, ha="right", fontweight="bold")
        else:                                                # 'shaded'
            ax.axvspan(kx, 100, color=REDFILL, alpha=0.6)
        ax.plot(kx, ky, "o", color=RED, ms=3.3)
    ax.set_xlim(0, 100); ax.set_ylim(0, 36); _clean(ax, keep=("bottom", "left"))

# ── trend → divergence sparkline ───────────────────────────────────────────────
def sparkline(ax, steep=1.8, threat=True, style='gap', saturating=False, endpoints=None):
    """trend strokes. threat=True -> the accelerating series is RED (a danger/limit:
    the data wall, runaway cost); threat=False -> GREEN (positive progress outpacing a
    baseline) — keeps RED sacred. Three self-evident variants:
      • 'gap' (default)  → two strokes + shaded fill: a DIVERGENCE between two series.
      • 'enddots'        → two strokes + a dot on each endpoint: a TWO-ENDPOINT before/after.
                           `endpoints=[(y0,y1),(y0,y1)]` (each in [0,1]) sets the two
                           strokes' start/end so a series can FALL (e.g. model size dropping
                           after compression); default is two rising strokes.
      • 'inflection'     → one trajectory + baseline + marker at the turning point.
                           saturating=False (default): a CONVEX accelerating curve
                           (compounding returns). saturating=True: a CONCAVE rise-then-
                           PLATEAU (diminishing returns / saturation — the common scaling-
                           law shape), marker at the flattening knee."""
    t = np.linspace(0, 1, 100); b = 0.1 + 0.16 * t
    fast, fill = (RED, REDFILL) if threat else (DATA, GREENFILL)
    if style == 'inflection':
        if saturating:
            a = 0.12 + 0.83 * (1 - np.exp(-3.4 * t)); kx = 0.5   # concave rise → plateau
        else:
            a = 0.1 + 0.85 * t ** 2.2; kx = 0.7                  # convex accelerating
        ax.axhline(0.12, color=GRID, lw=0.6)             # the baseline it pulls away from
        ax.plot(t, a, color=INK, lw=1.35)
        ki = int(kx * (len(t) - 1))
        ax.plot(t[ki], a[ki], "o", color=fast, ms=3.3)   # the turning point
    elif style == 'enddots':
        if endpoints:
            (a0, a1), (b0, b1) = endpoints
            a = a0 + (a1 - a0) * t; b = b0 + (b1 - b0) * t
        else:
            a = 0.1 + 0.85 * t ** steep
        ax.plot(t, a, color=fast, lw=1.35); ax.plot(t, b, color=MEM, lw=1.35)
        ax.plot(1, a[-1], "o", color=fast, ms=3.3); ax.plot(1, b[-1], "o", color=MEM, ms=3.3)
    else:                                                # 'gap'
        a = 0.1 + 0.85 * t ** steep
        ax.plot(t, a, color=fast, lw=1.35); ax.plot(t, b, color=MEM, lw=1.35)
        ax.fill_between(t, b, a, color=fill, alpha=0.4)
    ax.set_xlim(0, 1.05 if style == 'enddots' else 1); ax.set_ylim(0, 1); _clean(ax)

# ── bottleneck-regime → roofline elbow ─────────────────────────────────────────
def roofline(ax, ridge=60.0, dot_ai=6.0):
    """blue memory-bound slope + orange compute-bound ceiling + ridge dropline +
    workload dot. Axis limits are DERIVED from ridge and dot_ai so any real
    (ridge, dot) pair stays on-axis with both regimes legible — e.g. an H100
    ridge≈295 with a decode workload at AI≈1 (deep memory-bound) renders without
    clipping the dot or collapsing the orange ceiling to a sliver."""
    dot_y = min(dot_ai / ridge, 1.0)
    lo, hi = min(dot_ai, ridge), max(dot_ai, ridge)
    xmin, xmax = lo / 5.0, hi * 5.0
    ymin = max(min(dot_y, xmin / ridge) / 3.0, 1e-4)
    x = np.logspace(np.log10(xmin), np.log10(xmax), 200)
    y = np.minimum(x / ridge, 1.0); m = x < ridge
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(x[m], y[m], color=MEM, lw=1.45); ax.plot(x[~m], y[~m], color=COMP, lw=1.45)
    ax.axvline(ridge, color=GRID, ls="--", lw=0.55)
    ax.plot(dot_ai, dot_y, "o", color=INK, ms=3.2)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, 2.0); _clean(ax, keep=("bottom", "left"))
    ax.tick_params(axis="both", which="both", length=0)  # spines only, no log tick marks

# ── term-dominance → iron-law stacked bar ──────────────────────────────────────
def ironbar(ax, segs=(("D", 0.25, MEM), ("C", 0.55, COMP), ("L", 0.2, NET)), dom=1,
            style='stacked'):
    """which iron-law term dominates. Three self-evident variants:
      • 'stacked' (default) → one stacked bar, dominant segment full resource color,
                              rest desaturated: the COMPOSITION of the total.
      • 'trio'              → three separated horizontal bars, dominant one shaded: a
                              side-by-side magnitude comparison of the terms.
      • 'columns'           → three vertical columns, dominant one shaded: the same
                              comparison where vertical bars read more naturally.
    In every variant the DOMINANT term (index `dom`) carries its resource color and the
    rest desaturate to gray — the eye lands on what dominates."""
    if style == 'stacked':
        left = 0
        total = sum(v for _, v, _ in segs)
        for i, (l, v, c) in enumerate(segs):
            ax.barh(0, v, left=left, height=0.5,
                    color=c if i == dom else GRID, alpha=0.95 if i == dom else 0.7)
            frac = v / total if total else 0
            if frac > 0.026 * len(l) + 0.06:
                label_color = "white" if i == dom else INK
                ax.text(left + v / 2, 0, l, fontsize=6, ha="center", va="center",
                        color=label_color, fontweight="bold")
            else:
                ax.text(left + v / 2, 0.34, l, fontsize=4.7, ha="center", va="bottom",
                        color=INK, fontweight="bold")
            left += v
        ax.set_xlim(0, left); ax.set_ylim(-0.5, 0.5)
    elif style == 'columns':
        for i, (l, v, c) in enumerate(segs):
            ax.bar(i, v, width=0.6, color=c if i == dom else GRID,
                   alpha=0.95 if i == dom else 0.55)
            ax.text(i, -0.04 * max(v for _, v, _ in segs), l, fontsize=5.6,
                    ha="center", va="top", color=INK)
        ax.set_xlim(-0.5, len(segs) - 0.5); ax.set_ylim(0, max(v for _, v, _ in segs) * 1.15)
    else:                                                # 'trio' — separated horizontal
        y = list(range(len(segs)))[::-1]
        for yi, (l, v, c) in zip(y, segs):
            i = len(segs) - 1 - yi
            ax.barh(yi, v, height=0.6, color=c if i == dom else GRID,
                    alpha=0.95 if i == dom else 0.35)
            ax.text(0, yi + 0.44, l, fontsize=5.2, color=INK)
        ax.set_xlim(0, max(v for _, v, _ in segs) * 1.05); ax.set_ylim(-0.5, len(segs) - 0.2)
    _clean(ax)

# ── dam-axis → D·A·M triangle ──────────────────────────────────────────────────
def dam(ax, focus=1, vol="vol1", style='triangle'):
    """the D·A·M (vol2: D·A·I) axes. `focus` selects the reading: int (0/1/2) lights one
    axis as a single-axis LOCATOR ("this section is about axis X"); "all" lights all three
    as the COUPLED TRIAD (the framework intro — three coupled axes). Three self-evident
    shape variants carry the same reading:
      • 'triangle' (default) → vertices joined by VIOLET coupling edges: the coupling is
                               visible (best for the coupled-triad reading).
      • 'boxes'              → three stacked labeled boxes, the focus one lit: a compact
                               vertical locator for a tall, narrow margin.
      • 'pills'              → three side-by-side pills, the focus one lit: a compact
                               horizontal locator for a short, wide gap."""
    mlabel = "I" if vol == "vol2" else "M"
    names = {"D": "Data", "A": "Algorithm", mlabel: ("Infrastructure" if vol == "vol2" else "Machine")}
    triad = [("D", DATA), ("A", COMP), (mlabel, MEM)]
    def _lit(i): return (focus == "all") or (i == focus)
    if style == 'boxes':                                 # stacked locator (top=D … bottom=M)
        for row, (i, (g, rc)) in enumerate(zip((0, 1, 2), triad)):
            on = _lit(i); y = 2 - row
            ax.add_patch(plt.Rectangle((0, y), 1.6, 0.82, facecolor=rc if on else "#DDD"))
            ax.text(0.22, y + 0.41, g, fontsize=9, ha="center", va="center",
                    color="white" if on else "#999", fontweight="bold")
            ax.text(0.5, y + 0.41, names[g], fontsize=5, va="center",
                    color="white" if on else "#999")
        ax.set_xlim(0, 1.7); ax.set_ylim(-0.1, 3.0)
    elif style == 'pills':                               # side-by-side locator
        for i, (g, rc) in enumerate(triad):
            on = _lit(i)
            ax.add_patch(plt.Rectangle((i * 1.1, 0), 1.0, 0.62, facecolor=rc if on else "#DDD"))
            ax.text(i * 1.1 + 0.5, 0.31, g, fontsize=8, ha="center", va="center",
                    color="white" if on else "#999", fontweight="bold")
        ax.set_xlim(-0.1, 3.3); ax.set_ylim(-0.2, 0.85)
    else:                                                # 'triangle' — coupling edges visible
        pts = [(0.5, 0.9), (0.08, 0.12), (0.92, 0.12)]
        ax.plot([0.5, 0.08, 0.92, 0.5], [0.9, 0.12, 0.12, 0.9], color=NET, lw=1.35)
        for i, ((g, rc), (x, y)) in enumerate(zip(triad, pts)):
            on = _lit(i)
            ax.plot(x, y, "o", color=rc if on else "#DDD", ms=17)
            ax.text(x, y, g, fontsize=9, ha="center", va="center",
                    color="white" if on else "#999", fontweight="bold")
        ax.set_xlim(-0.18, 1.18); ax.set_ylim(-0.08, 1.12)
    _clean(ax)

# ── classification → taxonomy-mini ─────────────────────────────────────────────
def taxonomy(ax, hot=3, style='quadrant', items=None):
    """classification (NEUTRAL fills — classification owns no resource color; the
    selected cell uses the crimson selection accent). Three self-evident variants:
      • 'quadrant' (default) → solid 2x2, the 'you are here' cell filled crimson: a
                               two-axis taxonomy with one occupied corner.
      • 'dotcells'           → 2x2 outline cells each holding a status dot, the live one
                               crimson: a 2x2 of ON/OFF states (which cells are active).
      • 'listdots'           → a vertical labeled list, each row a status dot: a STAGED
                               or sequential set (e.g. detect→defend→recover→monitor).
                               Pass items=[(label, color), ...]; top item renders first."""
    if style == 'listdots':
        items = items or [("detect", DATA), ("defend", COMP), ("recover", RED), ("monitor", GRID)]
        for i, (lab, c) in enumerate(items[::-1]):       # first item on top
            ax.plot(0, i, "o", color=c, ms=6)
            ax.text(0.16, i, lab, fontsize=5.5, va="center", color=INK)
        ax.set_xlim(-0.1, 1.3); ax.set_ylim(-0.5, len(items) - 0.5); _clean(ax); return
    for i in range(2):
        for j in range(2):
            on = (i * 2 + j) == hot
            if style == 'dotcells':                      # outlined cells + status dots
                ax.add_patch(plt.Rectangle((j, i), 0.92, 0.92,
                             facecolor="none", edgecolor=GRID, lw=0.65))
                ax.plot(j + 0.46, i + 0.46, "o", color=SEL if on else GRID, ms=7)
            else:                                        # 'quadrant' — solid fills
                ax.add_patch(plt.Rectangle((j, i), 0.92, 0.92,
                             facecolor=SEL if on else "#EEE", edgecolor="white", lw=0.75))
    ax.set_xlim(-0.1, 2); ax.set_ylim(-0.1, 2); _clean(ax)

# ── correlated-failure → blast-radius fan ──────────────────────────────────────
def blast(ax, n=5, style='fan'):
    """one RED source (the fault) propagating outward (sacred red). Three self-evident
    variants for the shape of the propagation:
      • 'fan' (default) → source → N arrows to N independent consumers: one fault hits
                          many peers directly (noisy neighbor, a downed shared switch).
      • 'tree'          → source → a few → many: a HIERARCHICAL cascade (1→3→6), where
                          the failure amplifies down levels (a root dependency falling).
      • 'rings'         → concentric severity zones around the source: a blast RADIUS,
                          impact graded by distance (incident/fault-domain reach)."""
    if style == 'tree':
        ax.plot(0.05, 0.5, "s", color=RED, ms=9)
        for m in np.linspace(0.25, 0.75, 3):
            ax.plot([0.1, 0.5], [0.5, m], color="#BBB", lw=0.65)
            ax.plot(0.5, m, "o", color=NET, ms=5)
            for lf in (m - 0.08, m + 0.08):
                ax.plot([0.55, 0.95], [m, lf], color="#DDD", lw=0.5)
                ax.plot(0.95, lf, "o", color=MEM, ms=3)
        ax.set_xlim(0, 1.05); ax.set_ylim(0, 1)
    elif style == 'rings':
        for rad, a in [(0.45, 0.15), (0.30, 0.30), (0.15, 0.6)]:
            ax.add_patch(plt.Circle((0.5, 0.5), rad, color=RED, alpha=a))
        ax.plot(0.5, 0.5, "s", color=RED, ms=8)
        for ang in np.linspace(0, 2 * np.pi, n, endpoint=False):
            ax.plot(0.5 + 0.45 * np.cos(ang), 0.5 + 0.45 * np.sin(ang), "o", color=MEM, ms=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    else:                                                # 'fan'
        ax.plot(0.06, 0.5, "s", color=RED, ms=10)
        for yy in np.linspace(0.06, 0.94, n):
            ax.annotate("", xy=(0.95, yy), xytext=(0.13, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.65))
            ax.plot(0.95, yy, "o", color=MEM, ms=5)
        ax.set_xlim(0, 1.05); ax.set_ylim(0, 1)
    _clean(ax)

# ── bounded operating budget → budget-envelope ────────────────────────────────
def budget_envelope(ax, rows=(("used", 0.7, COMP), ("limit", 1.0, RED)), limit=1.0,
                    style='burn', limit_label=None):
    """finite budget, quota, capacity, or matched-rate envelope.

      • 'burn'    → one or more bars on a shared denominator with a red limit
                    marker; red wash shows over-budget territory.
      • 'matched' → two equal-width resource bands; the job is matching rates or
                    capacities, not ranking magnitudes.
    """
    if style == 'matched':
        rows = rows or (("comm cap", 1.0, NET), ("storage BW", 1.0, MEM))
        ys = np.linspace(0.64, 0.34, len(rows))
        for (label, _, color), y in zip(rows, ys):
            ax.add_patch(plt.Rectangle((0.14, y - 0.07), 0.72, 0.14,
                                       facecolor=color, edgecolor="white", lw=0.45))
            ax.text(0.50, y, label, ha="center", va="center", color="white",
                    fontsize=5.0, fontweight="bold")
        ax.plot([0.14, 0.86], [0.20, 0.20], color=DATA, lw=1.0)
        ax.text(0.50, 0.10, "matched", ha="center", va="center", color=DATA,
                fontsize=5.0, fontweight="bold")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); _clean(ax); return

    max_value = max([limit] + [value for _, value, _ in rows])
    xmax = max_value * 1.08
    x0, w, h = 0.34, 0.58, 0.13
    limit_x = x0 + w * limit / xmax
    ax.axvspan(limit_x, x0 + w, color=REDFILL, alpha=0.34, lw=0)
    ax.axvline(limit_x, color=RED, lw=0.7, ls="--")
    ys = np.linspace(0.62, 0.32, len(rows))
    for (label, value, color), y in zip(rows, ys):
        ax.text(x0 - 0.04, y, label, ha="right", va="center", color=INK, fontsize=4.7)
        bar_w = w * value / xmax
        ax.add_patch(plt.Rectangle((x0, y - h / 2), bar_w, h,
                                   facecolor=color, edgecolor="white", lw=0.35))
        if bar_w > 0.19:
            value_x = x0 + bar_w / 2
            if value > limit:
                over_w = x0 + bar_w - limit_x
                if over_w > 0.17:
                    value_x = limit_x + over_w / 2
                else:
                    value_x = x0 + bar_w - 0.02
            ax.text(value_x, y, f"{value:g}", ha="center", va="center",
                    color="white", fontsize=4.7, fontweight="bold")
        else:
            ax.text(x0 + bar_w + 0.02, y, f"{value:g}", ha="left", va="center",
                    color=INK, fontsize=4.7)
    label_x = limit_x
    label_ha = "center"
    if limit_x > 0.78:
        label_x = limit_x - 0.01
        label_ha = "right"
    elif limit_x < 0.22:
        label_x = limit_x + 0.01
        label_ha = "left"
    ax.text(label_x, 0.82, limit_label or "limit", ha=label_ha, va="center",
            color=RED, fontsize=4.8, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); _clean(ax)

# ── short ordered window/process → sequence-strip ─────────────────────────────
def sequence_strip(ax, steps=(("1", GRID), ("2", GRID), ("3", GRID)), bracket=None,
                   bracket_label=None):
    """compact ordered strip for windows or phases whose order is the concept."""
    n = len(steps)
    x0, w, gap, y, h = 0.10, 0.78, 0.012, 0.42, 0.16
    cell = (w - gap * (n - 1)) / n
    for i, (label, color) in enumerate(steps):
        x = x0 + i * (cell + gap)
        ax.add_patch(plt.Rectangle((x, y), cell, h, facecolor=color,
                                   edgecolor="white", lw=0.35))
        ax.text(x + cell / 2, y + h / 2, label, ha="center", va="center",
                color="white" if color != GRID else "#555555",
                fontsize=4.6, fontweight="bold")
    if bracket:
        a, b = bracket
        left = x0 + a * (cell + gap)
        right = x0 + b * (cell + gap) + cell
        ax.plot([left, right], [0.72, 0.72], color=RED, lw=1.1)
        ax.plot([left, left], [0.68, 0.76], color=RED, lw=0.75)
        ax.plot([right, right], [0.68, 0.76], color=RED, lw=0.75)
        if bracket_label:
            ax.text((left + right) / 2, 0.84, bracket_label, ha="center",
                    va="center", color=RED, fontsize=4.8, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); _clean(ax)

# ── compact dependency or feedback relation → causal-chain ────────────────────
def causal_chain(ax, labels=("cause", "effect"), style='chain', colors=None):
    """tiny causal/dependency chain; use only when the arrows are the concept."""
    labels = list(labels)
    colors = colors or [COMP] * max(len(labels) - 1, 1) + [RED]
    if style == 'loop':
        pts = [(0.20, 0.64), (0.72, 0.64), (0.46, 0.25)]
        for (label, (x, y), color) in zip(labels[:3], pts, colors[:3]):
            ax.add_patch(plt.Rectangle((x - 0.15, y - 0.075), 0.30, 0.15,
                                       facecolor=color, edgecolor="white", lw=0.45))
            ax.text(x, y, label, ha="center", va="center", color="white",
                    fontsize=4.8, fontweight="bold")
        for start, end in [((0.35, 0.64), (0.57, 0.64)),
                           ((0.72, 0.56), (0.52, 0.32)),
                           ((0.36, 0.31), (0.20, 0.55))]:
            ax.annotate("", xy=end, xytext=start,
                        arrowprops=dict(arrowstyle="->", color=NET, lw=0.85))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); _clean(ax); return

    xs = np.linspace(0.14, 0.86, len(labels))
    for i, (x, label) in enumerate(zip(xs, labels)):
        c = colors[min(i, len(colors) - 1)]
        ax.plot(x, 0.58, "o", color=c, ms=8)
        ax.text(x, 0.31, label, ha="center", va="center", color=INK,
                fontsize=4.8)
        if i < len(labels) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.055, 0.58), xytext=(x + 0.055, 0.58),
                        arrowprops=dict(arrowstyle="->", color=GRID, lw=0.75))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); _clean(ax)

# ── many-to-many coupling road sign → all-to-all-topology ─────────────────────
def all_to_all_topology(ax, n=4):
    """tiny mesh for relationships where many endpoints each depend on all peers."""
    if n != 4:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = [(0.5 + 0.33 * np.cos(a), 0.5 + 0.33 * np.sin(a)) for a in angles]
    else:
        pts = [(0.25, 0.72), (0.75, 0.72), (0.25, 0.28), (0.75, 0.28)]
    for i, (x0, y0) in enumerate(pts):
        for j, (x1, y1) in enumerate(pts):
            if i < j:
                ax.plot([x0, x1], [y0, y1], color=NET, lw=0.65, alpha=0.55)
    for i, (x, y) in enumerate(pts, 1):
        ax.plot(x, y, "s", color=MEM, ms=10)
        ax.text(x, y, str(i), ha="center", va="center", color="white",
                fontsize=5.0, fontweight="bold")
    ax.text(0.50, 0.08, "all-to-all", ha="center", va="center",
            color=INK, fontsize=5.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); _clean(ax)

DEVICES = {
    'hierarchy-ladder': ladder, 'scale-anchor': knee, 'sparkline-trend': sparkline,
    'thumbnail-roofline': roofline, 'iron-law-bar': ironbar, 'dam-locator': dam,
    'taxonomy-mini': taxonomy, 'blast-radius': blast,
    'budget-envelope': budget_envelope, 'sequence-strip': sequence_strip,
    'causal-chain': causal_chain, 'all-to-all-topology': all_to_all_topology,
}

if __name__ == "__main__":  # smoke test
    import os
    out = "/tmp/margin_devices_smoke"; os.makedirs(out, exist_ok=True)
    fig, ax = new_fig('hierarchy-ladder')
    ladder(ax, [("HBM", 3350), ("DRAM", 100), ("NVMe", 7), ("SSD", 1), ("net", 0.1)])
    save(fig, f"{out}/ladder.png")
    print("smoke OK ->", out)
