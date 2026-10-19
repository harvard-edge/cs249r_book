#!/usr/bin/env python3
"""Figures for Lecture 16 (Conclusion).

Plots use the shared slideplot helper; diagrams are written as SVG with a
small builder below, one file per build step, every step on the same canvas
so nothing moves. "Chapter" means the rendered Conclusion chapter. Every
value is sourced:

  - Stack numbers (one per layer, each from the lecture that taught it, with
    the value and source recorded in that lecture's lecture_map.yml):
    TPU 30 to 80x performance per watt (L11, Jouppi et al. 2017); a datasheet
    20 against a sustained 1.5 TOPS/W (L12, chapter Sec. 12.7 scenario); 8.6 ms of
    eager host time against 0.9 ms of weight reads for one GPT-2 XL token
    (L7, Frameworks Sec. 7.1); 7B model 14 GB in FP16 against a phone's 8 GB
    (L10, Model Compression Sec. 10.1); 16 bytes per parameter, PaLM about
    8.6 TB, about 100 H100s (L8, derived there); batch 1 $3.92 against
    batch 8 $0.52 per million requests, 7.5x (L13, Example 13.1); Zillow
    $304 million write-down (L14, Zillow Q3 2021 letter); 17.7 against 46.5
    percent (L15, Obermeyer et al. 2019); Tay suspended after 16 hours (L4,
    War Story 4.1); GPT-2 saw each WebText token about 17 times (L9).
  - Fleet failure chain: chapter Sec. 16.1 (composite scenario, margin
    figure). Lecture tags name the deck that teaches each idea: calibration
    set (10b), training-serving skew (4, 13b), drift (4b, 14b), slices
    against one aggregate metric (12b). MobileNetV2 journey: chapter table 2 (8.7x fewer FLOPs for a
    3x3, 256-output-channel layer; INT8 is 1/4 of FP32 storage; p99 < 50 ms).
  - Roofline: H100 3.35 TB/s, FP16 989 TFLOP/s, INT8 1,979 TOPS; ridges
    295 and 591 FLOP/byte; GPT-2 XL decode 1 FLOP/byte; GPT-2 XL QKV
    projection at prefill batch 8, 1,046.7 FLOP/byte; ResNet-50 stage-3 3x3
    conv with INT8 weights and FP16 activations, 569 FLOP/byte (Hardware
    Acceleration, Napkin Math 11.4 to 11.10, as drawn in Lecture 11).
  - Thirteen principles, their parts and kinds: chapter Sec. 16.3, table 3
    and figure 1 (3 bounds, 2 decompositions, 2 fitted models, 2 policies,
    4 heuristics).
  - Amdahl: 90 percent host preprocessing, 10x faster inference, 1.1x
    (chapter Sec. 16.7).
  - Tails: illustrative TTFT mean 50 ms, p99 2,000 ms (chapter Sec. 16.4,
    scenario assumptions). Measured: Dean and Barroso, "The Tail at Scale,"
    CACM 56(2), 2013, Table 1, read from the paper: one random leaf 1 ms
    (50th percentile), 10 ms (99th); all leaf requests 40 ms (50th),
    140 ms (99th).
  - MTBF: MTTF 5.7 years (about 50,000 hours) / 1,024 GPUs = 48.8 hours
    (chapter Sec. 16.5). 16,384 GPUs: 3.05 hours (same model, derived).
    Llama 3 405B: up to 16K H100 GPUs, 419 unexpected interruptions in a
    54-day snapshot (Llama Team, "The Llama 3 Herd of Models," 2024,
    arXiv 2407.21783, Sec. 3.3.4, read); 54 x 24 / 419 = 3.1 hours (derived).
  - Composed request: retrieval 50 ms, tools 110 and 70 ms in parallel,
    generation 180 ms, verifier 30 ms; critical path 370 ms (chapter
    self-check 16.5, question 1).
  - Voice request: KWS 200 KB INT8 in 1 MB flash, 20 MFLOP, milliwatts;
    front end 600 MFLOP, 3.5 MB INT8, 5 W; DLRM 100 GB of embeddings
    against 80 GB; GPT-2 3 GB per token, 0.9 ms floor, 295x the compute
    floor (chapter Sec. 16.8).

  - Llama 3 Table 5 rows (frame 28) are typeset in the .tex, quoted from the
    paper; the hours compared in llama_mtbf come from calcs.py.
  - Power ladder (power_tiers.svg): 3 MW, 200 W, 5 W, 1 mW (System
    Paradigms Sec. 2.1, as in Lecture 0), redrawn here at slot size.

Every derived value drawn is imported from calcs.py (2026-10-19). The script
ends by writing text-free thumb_*.svg versions of the figures used on goal,
takeaway, recall, and part-locator slots (tools/greek_thumbs.py).

Copied from other decks (credited on the slide as in the source deck):
gft_step2.svg and ironlaw_bars_step1.svg (Lecture 1), data_wall_step2.svg
(Lecture 9), robot-arms-bmw-leipzig.jpg (Lecture 2), thai-clinic-nurse.jpg
(Lecture 3); see SOURCES.yml.

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

plt = sp.plt
C = sp.C
out = Path(__file__).parent

CRIMSON = "#A51C30"
FONT = 'font-family="Helvetica, Arial, sans-serif"'


class Svg:
    """A tiny SVG builder. Every element carries the first step it appears in;
    write(stem, n) emits stem_step1..n, hidden elements kept but transparent."""

    def __init__(self, w, h):
        self.w, self.h, self.items = w, h, []

    def add(self, first, s, last=99):
        self.items.append((first, last, s))

    def rect(self, x, y, w, h, fill, stroke, first=1, sw=2, rx=8, dash=None,
             last=99, opacity=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(first, f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
                        f'opacity="{opacity}"{d}/>', last)

    def text(self, x, y, s, size=22, color=C["text"], weight="normal",
             anchor="middle", first=1, last=99, italic=False, opacity=1.0):
        lines = s.split("\n")
        y0 = y - (len(lines) - 1) * size * 0.6
        spans = "".join(
            f'<tspan x="{x}" y="{y0 + i * size * 1.2:.1f}">{ln}</tspan>'
            for i, ln in enumerate(lines))
        st = ' font-style="italic"' if italic else ""
        self.add(first, f'<text {FONT} font-size="{size}" fill="{color}" '
                        f'font-weight="{weight}"{st} opacity="{opacity}" '
                        f'text-anchor="{anchor}" dominant-baseline="middle">'
                        f'{spans}</text>', last)

    def line(self, x1, y1, x2, y2, color, first=1, sw=2.5, arrow=True,
             dash=None, last=99, opacity=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#a{color[1:]})"' if arrow else ""
        self.add(first, f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                        f'stroke="{color}" stroke-width="{sw}" '
                        f'opacity="{opacity}"{d}{m}/>', last)

    def circle(self, x, y, r, fill, stroke, first=1, sw=2, last=99, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(first, f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
                        f'stroke="{stroke}" stroke-width="{sw}"{d}/>', last)

    def write(self, stem, steps):
        colors = sorted(set(C.values()) | {CRIMSON, "#7FC8E8"})
        defs = "".join(
            f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in colors)
        for k in range(1, steps + 1):
            body = []
            for first, last, s in self.items:
                vis = first <= k <= last
                body.append(s if vis else '<g opacity="0">' + s + "</g>")
            svg = (f'<svg xmlns="http://www.w3.org/2000/svg" '
                   f'viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                   f'height="{self.h}"><defs>{defs}</defs>'
                   f'<rect width="100%" height="100%" fill="white"/>'
                   + "".join(body) + "</svg>")
            name = f"{stem}_step{k}.svg" if steps > 1 else f"{stem}.svg"
            (out / name).write_text(svg)


import calcs as K  # noqa: E402  (every derived value drawn below)
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402


def ellipse(g, cx, cy, rx, ry, fill, stroke, first=1, sw=1.5, dash=None, last=99):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    g.add(first, f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}/>', last)


def bars(path, rows, size, log=False, ticks=None, xlabel=None, xlim=None, fs=11,
         lfs=None):
    """Horizontal bars drawn at the size of their slot. rows: (label, value,
    color key, printed value). ticks: list of (x, label) for plain tick labels
    (no 10^n superscripts, which render too small)."""
    fig, ax = plt.subplots(figsize=size)
    n = len(rows)
    vals = [r[1] for r in rows]
    for i, (label, v, ck, txt) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, v, color=C[ck], height=0.62)
        ax.text(-0.02, y, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=lfs or fs, color=C["text"], fontweight="bold")
        ax.text(v * 1.15 if log else v + max(vals) * 0.02, y, txt, va="center",
                fontsize=fs, color=C[ck], fontweight="bold")
    ax.set_yticks([])
    if log:
        ax.set_xscale("log")
    ax.set_xlim(*(xlim or ((min(vals) / 3, max(vals) * 40) if log else (0, max(vals) * 1.35))))
    if ticks:
        ax.xaxis.set_major_locator(FixedLocator([t for t, _ in ticks]))
        ax.set_xticklabels([s for _, s in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", labelsize=fs - 1)
    ax.set_ylim(-0.6, n - 0.4)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fs - 1)
    sp.save(fig, path)


# =============================================================================
# The stack with one number per layer (retrieval reveal, frame 2).
# Slot 0.92 textwidth x 5.6 cm: canvas 900 x 366, so 20 px renders near 8.7 pt.
# =============================================================================
g = Svg(900, 366)
layers = [
    ("Applications", "Black share enrolled: 17.7% by cost, 46.5% by need", "L15", 0.12),
    ("Operations", "$304M write-down, every server up", "L14", 0.17),
    ("Serving", f"Batch 1 costs {K.BATCH_COST_RATIO:.1f}× batch 8", "L13", 0.22),
    ("Training", "16 bytes per parameter: PaLM ≈ 100 H100s", "L8", 0.12),
    ("Models", f"7B model in FP16 = {K.MODEL_7B_GB:.0f} GB; phone has 8 GB", "L10", 0.17),
    ("Frameworks", "Host 8.6 ms against 0.9 ms of weight reads", "L7", 0.22),
    ("Hardware", "TPU 30 to 80× perf/W; 20 TOPS/W rated, 1.5 sustained", "L11, L12", 0.12),
]
for i, (name, num, lec, op) in enumerate(layers):
    y = 6 + i * 46
    g.add(1, f'<rect x="10" y="{y}" width="786" height="40" rx="7" fill="{CRIMSON}" '
             f'fill-opacity="{op}"/>')
    g.text(24, y + 21, name, 22, CRIMSON, "bold", "start")
    g.text(190, y + 21, num, 20, C["text"], "normal", "start")
    g.text(786, y + 21, lec, 18, CRIMSON, "bold", "end")
g.add(1, f'<rect x="806" y="6" width="56" height="316" rx="7" fill="{CRIMSON}" '
         f'fill-opacity="0.18"/>')
g.add(1, f'<text {FONT} x="834" y="164" font-size="22" font-weight="700" fill="{CRIMSON}" '
         f'text-anchor="middle" dominant-baseline="middle" letter-spacing="3" '
         f'transform="rotate(-90 834 164)">DATA</text>')
g.line(834, 322, 834, 334, CRIMSON, arrow=False, sw=2)
g.text(862, 348, "Data: Tay suspended after 16 hours (L4); each WebText token "
       f"seen about {K.WEBTEXT_PASSES:.0f} times (L9)", 19, C["text"], "normal", "end")
g.write("stack_numbers", 1)

# =============================================================================
# The fleet failure chain, tagged by lecture (frame 6; thumbnails on 4, 31, 33).
# Slot 0.95 textwidth x 4.9 cm: canvas 900 x 312, 19 px renders near 8.5 pt.
# =============================================================================
g = Svg(900, 312)
stages = [("INT8\ncalibration", C["compute"], C["compute_l"], "Lecture 10b\ncalibration set"),
          ("Firmware\npreprocessing", C["neutral"], C["neutral_l"], "Lectures 4, 14\ntraining-serving skew"),
          ("Input shift on\none cohort", C["neutral"], C["neutral_l"], "Lectures 4b, 14b\ndrift"),
          ("Accuracy\n−4 points", C["danger"], C["danger_l"], "Lecture 12b\naggregate metric only")]
for i, (lab, stroke, fill, tag) in enumerate(stages):
    x = 10 + i * 225
    g.rect(x, 6, 200, 86, fill, stroke, sw=2.5)
    g.text(x + 100, 49, lab, 22, stroke if i in (0, 3) else C["text"], "bold")
    if i < 3:
        g.line(x + 202, 49, x + 223, 49, C["neutral"], sw=3)
    g.text(x + 100, 124, tag, 19, CRIMSON, "bold", first=2)
g.text(450, 172, "Every team met its own target; no test crossed the interfaces.",
       20, C["neutral"], "normal", italic=True, first=2)
# step 3: the two measurements that would have caught it
g.rect(10, 196, 425, 108, C["data_l"], C["data"], first=3)
g.text(222, 224, "Skew check (principle 11)", 21, C["data"], "bold", first=3)
g.text(222, 268, "same captured images through\nboth preprocessing paths", 19,
       C["text"], first=3)
g.rect(465, 196, 425, 108, C["data_l"], C["data"], first=3)
g.text(677, 224, "Cohort-level monitoring", 21, C["data"], "bold", first=3)
g.text(677, 268, "accuracy by device and firmware,\nfrom day one", 19, C["text"], first=3)
g.write("chain", 3)

# =============================================================================
# One phone model through nine lectures (frame 7; chapter table 2).
# Slot 0.92 textwidth x 5.5 cm: canvas 900 x 362, 18 px renders near 7.8 pt.
# =============================================================================
g = Svg(900, 362)
phases = [
    ("Envelope", "Watt-scale power,\nheat, battery", "L2"),
    ("Architecture", f"Depthwise separable:\n{K.DW_RATIO:.1f}× fewer FLOPs", "L6"),
    ("Training", "Large batches to train;\nserves one image", "L8"),
    ("Data selection", "Milder augmentation\nfor a small model", "L9"),
    ("Compression", f"INT8: {K.INT8_OF_FP32 * 100:.0f}% of\nFP32's bytes", "L10b"),
    ("Acceleration", "Do the phone's NPU\nkernels run INT8?", "L11"),
    ("Benchmarking", "Gain must survive\npreprocessing", "L12"),
    ("Serving", "p99 under 50 ms", "L13"),
    ("Operations", "Accuracy tracked\nper device cohort", "L14b"),
]
for i, (name, num, lec) in enumerate(phases):
    row, col = divmod(i, 3)
    x, y = 10 + col * 300, 4 + row * 112
    first = row + 1
    g.rect(x, y, 264, 94, "white", C["neutral"], first=first, sw=2)
    g.text(x + 12, y + 20, name, 21, CRIMSON, "bold", "start", first=first)
    g.text(x + 254, y + 20, lec, 18, C["neutral"], "bold", "end", first=first)
    g.text(x + 132, y + 63, num, 18, C["text"], first=first)
    if col < 2:
        g.line(x + 266, y + 47, x + 298, y + 47, C["neutral"], first=first, sw=2.5)
for row in (0, 1):
    y0 = 4 + row * 112
    g.line(880, y0 + 96, 880, y0 + 103, C["neutral"], arrow=False, first=row + 2, sw=2)
    g.line(880, y0 + 103, 142, y0 + 103, C["neutral"], arrow=False, first=row + 2, sw=2,
           dash="5 4")
    g.line(142, y0 + 103, 142, y0 + 111, C["neutral"], first=row + 2, sw=2)
# step 3: the propagation the chapter traces
g.text(450, 346, "Depthwise layers go bandwidth-bound, so the next lever is bytes:"
       " INT8, if the NPU runs it.", 18, C["danger"], "bold", first=3)
g.write("journey", 3)

# =============================================================================
# Roofline: one model, two regimes; INT8 raises the ridge (frame 9)
# =============================================================================
import numpy as np  # noqa: E402

BW, P16, P8 = K.H100_BW / 1e12, K.H100_FP16 / 1e12, K.H100_INT8 / 1e12
xs = np.logspace(np.log10(0.4), np.log10(30000), 400)
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    ax.plot(xs, np.minimum(P16, BW * xs), color=C["memory"], lw=3, zorder=2)
    ax.plot(xs[xs >= P16 / BW], np.full((xs >= P16 / BW).sum(), P16),
            color=C["compute"], lw=3, zorder=3)
    ax.axvline(P16 / BW, color=C["neutral"], lw=1, ls="--")
    ax.text(P16 / BW / 1.08, 1.4, f"ridge {K.RIDGE16:.0f}", color=C["neutral"], fontsize=10,
            ha="right")
    on3 = 1.0 if k >= 3 else 0.0
    m = xs >= P8 / BW
    ax.plot(xs[m], np.full(m.sum(), P8), color=C["danger"], lw=2.5, ls="--", alpha=on3)
    ax.plot(xs[(xs >= P16 / BW) & (xs <= P8 / BW)],
            BW * xs[(xs >= P16 / BW) & (xs <= P8 / BW)], color=C["danger"], lw=2.5,
            ls="--", alpha=on3)
    ax.axvline(P8 / BW, color=C["danger"], lw=1, ls=":", alpha=on3)
    ax.text(P8 / BW * 1.08, 1.4, f"INT8 ridge {K.RIDGE8:.0f}", color=C["danger"], fontsize=10,
            alpha=on3)
    ax.text(28000, P8 * 1.1, "INT8 peak 1,979", color=C["danger"], fontsize=10, ha="right",
            va="bottom", fontweight="bold", alpha=on3)
    ax.text(28000, P16 / 1.1, "FP16 peak 989", color=C["compute"], fontsize=10, ha="right",
            va="top", fontweight="bold")
    # step 1: decode
    ax.plot([1], [BW], "o", ms=11, color=C["memory"], mec="white", mew=1.5, zorder=5)
    ax.annotate("GPT-2 XL decode,\nbatch 1: 1", (1, BW), xytext=(14, 0),
                textcoords="offset points", va="center", fontsize=11, fontweight="bold",
                color=C["memory"])
    a2 = 1.0 if k >= 2 else 0.0
    ax.plot([K.PREFILL_I], [P16], "o", ms=11, color=C["compute"], mec="white", mew=1.5,
            zorder=5, alpha=a2)
    ax.annotate(f"Same model,\nbatched prefill: {K.PREFILL_I:,.0f}", (K.PREFILL_I, P16),
                xytext=(6, -36), textcoords="offset points", ha="left", va="top", fontsize=11,
                fontweight="bold", color=C["compute"], alpha=a2)
    ax.plot([K.RESNET_INT8_I], [BW * K.RESNET_INT8_I], "o", ms=11, color=C["danger"],
            mec="white", mew=1.5, zorder=6, alpha=on3)
    ax.annotate(f"ResNet-50 conv, INT8\nweights: {K.RESNET_INT8_I}, below {K.RIDGE8:.0f}",
                (K.RESNET_INT8_I, BW * K.RESNET_INT8_I), xytext=(-30, 14),
                textcoords="offset points", ha="right", va="bottom", fontsize=11,
                fontweight="bold", color=C["danger"], alpha=on3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.4, 30000)
    ax.set_ylim(1, 30000)
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["1", "10", "100", "1,000", "10,000"])
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["1", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("TFLOP/s on one H100")
    sp.save(fig, out / f"roofline_step{k}.svg")

# =============================================================================
# The thirteen principles as a cycle: slide variant of book figure 1.
# Canvas 900 x 410: at 0.92 textwidth x 5.3 cm (frame 12) 22 px renders near
# 8 pt; at 0.88 textwidth x 5.0 cm (frame 18) near 7.6 pt.
# =============================================================================
PARTS = [
    ("Foundations (Data)", 6, 6, 136, [(1, "Data as code", "heuristic"),
                                       (2, "Data gravity", "heuristic")]),
    ("Development (Model)", 474, 6, 136, [(3, "Iron law", "decomposition"),
                                          (4, "Silicon contract", "heuristic")]),
    ("Optimization", 474, 190, 214, [(5, "Pareto frontier", "bound"),
                                     (6, "Arithmetic intensity", "bound"),
                                     (7, "Energy movement", "decomposition"),
                                     (8, "Amdahl's law", "bound")]),
    ("Deployment", 6, 190, 214, [(9, "Verification gap", "policy"),
                                 (10, "Statistical drift", "fitted model"),
                                 (11, "Skew diagnostic", "heuristic"),
                                 (12, "Latency budget", "policy"),
                                 (13, "Bias feedback", "fitted model")]),
]
ARROW = "#7FC8E8"


def cycle(stem, steps=1, kinds_from=None, lit=None, trace=None):
    """kinds_from: step at which kind tags appear (None: never).
    lit: part index to light (others dimmed). trace: list of (principle,
    first step) highlighted in order."""
    g = Svg(900, 410)
    for p, (title, x, y, h, items) in enumerate(PARTS):
        dim = lit is not None and p != lit
        op = 0.28 if dim else 1.0
        on = lit is not None and p == lit
        g.rect(x, y, 420, h, "#FBEFF1" if on else "white",
               CRIMSON if on else "#BBBBBB", sw=3 if on else 2, opacity=op)
        g.text(x + 16, y + 24, title, 23, CRIMSON, "bold", "start", opacity=op)
        for j, (num, name, kind) in enumerate(items):
            ly = y + 62 + j * 34
            for pr, f in (trace or []):
                if pr == num:
                    g.rect(x + 6, ly - 16, 408, 32, C["compute_l"], C["compute"],
                           sw=1.5, rx=5, first=f)
            g.text(x + 42, ly, str(num), 22, C["neutral"], "bold", "end", opacity=op)
            g.text(x + 52, ly, name, 22, C["text"], "normal", "start", opacity=op)
            if kinds_from:
                g.text(x + 410, ly, kind, 20, C["neutral"], "normal", "end",
                       first=kinds_from, italic=True, opacity=op)
    for x1, y1, x2, y2 in ((428, 40, 470, 40), (684, 144, 684, 186),
                           (472, 380, 430, 380), (216, 186, 216, 144)):
        g.line(x1, y1, x2, y2, ARROW, sw=8)
    ellipse(g, 450, 165, 88, 40, "white", C["neutral"], dash="4 3")
    g.text(450, 165, "Conservation\nof complexity", 19, C["text"], "bold")
    g.write(stem, steps)


cycle("cycle", steps=2, kinds_from=2)
for p in range(4):
    cycle(f"cycle_part{p + 1}", lit=p)
cycle("cycle_trace", steps=2, trace=[(5, 1), (4, 1), (6, 1), (7, 1),
                                     (12, 2), (9, 2), (11, 2)])

# =============================================================================
# Amdahl: a 10x faster kernel inside a request that is 90 percent host.
# Slot 0.85 textwidth x 3.0 cm; drawn near that size.
# =============================================================================
host_pct = K.SERIAL * 100
inf_pct = (1 - K.SERIAL) * 100
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.7, 1.05))
    rows = [("Before", host_pct, inf_pct, f"+ inference {inf_pct:.0f}% = 100%"),
            ("After 10× kernel", host_pct, inf_pct / K.SPEEDUP,
             f"+ {inf_pct / K.SPEEDUP:.0f}% = {K.NEW_TIME * 100:.0f}%: {K.AMDAHL:.1f}× faster")]
    for i, (lab, host, inf, tot) in enumerate(rows):
        y = 1 - i
        on = 1.0 if (i == 0 or k == 2) else 0.0
        ax.barh(y, host, color=C["neutral"], height=0.62, alpha=on)
        ax.barh(y, inf, left=host, color=C["compute"], height=0.62, alpha=on)
        ax.text(45, y, f"Host preprocessing {host_pct:.0f}%", ha="center", va="center",
                color="white", fontsize=11, fontweight="bold", alpha=on)
        ax.text(-1, y, lab, ha="right", va="center", fontsize=11, fontweight="bold",
                color=C["text"], alpha=on)
        ax.text(host + inf + 1.5, y, tot, va="center", fontsize=11, fontweight="bold",
                color=C["compute"] if i == 0 else C["danger"], alpha=on)
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=10)
    ax.set_ylim(-0.5, 1.5)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Share of the original request time (%)", fontsize=10)
    sp.save(fig, out / f"amdahl_step{k}.svg")

# =============================================================================
# Tails: illustrative scenario, then a measured Google service
# =============================================================================
sp.hbars(str(out / "tail_illus.svg"), [
    ("Mean TTFT", K.TTFT_MEAN, "neutral", f"{K.TTFT_MEAN} ms"),
    ("p99 TTFT", K.TTFT_P99, "danger", f"{K.TTFT_P99:,} ms ({K.TTFT_P99 / K.TTFT_MEAN:.0f}×)"),
], "ms", log=False, xlabel="Time to first token (ms)", size=(3.4, 1.6))
sp.hbars(str(out / "tail_measured.svg"), [
    ("One leaf, median", 1, "memory", "1 ms"),
    ("One leaf, p99", K.LEAF_P99, "memory", f"{K.LEAF_P99} ms"),
    ("All leaves, median", 40, "danger", "40 ms"),
    ("All leaves, p99", K.ALL_P99, "danger", f"{K.ALL_P99} ms ({K.ALL_P99 / K.LEAF_P99:.0f}×)"),
], "ms", log=False, xlabel="Latency (ms)", size=(3.4, 2.1))

# =============================================================================
# From one GPU to a pool (frame 23). Slot 0.62 textwidth x 3.2 cm.
# =============================================================================
bars(out / "mtbf.svg", [
    ("One GPU (MTTF)", K.MTTF_H, "neutral", f"{K.MTTF_H:,} h ({K.MTTF_YEARS} years)"),
    ("A three-week run", K.RUN_H, "memory", f"{K.RUN_H} h"),
    (f"{K.N_GPU:,} GPUs (MTBF)", K.MTBF_H, "danger",
     f"{K.MTBF_H:.1f} h ({K.MTBF_H / 24:.0f} days)"),
], size=(2.4, 1.05), log=True, xlim=(10, 3e6),
    ticks=[(10, "10"), (1000, "1,000"), (100000, "100,000")],
    xlabel="Hours (log scale)", fs=10)

# Llama 3 against the model (frame 28). Slot about 0.45 textwidth x 1.9 cm.
bars(out / "llama_mtbf.svg", [
    (f"Model, {K.LLAMA_GPUS:,} GPUs", K.MODEL_16K_H, "neutral", f"{K.MODEL_16K_H:.2f} h"),
    ("Llama 3, measured", K.LLAMA_GAP_H, "danger", f"{K.LLAMA_GAP_H:.1f} h"),
], size=(1.5, 0.62), xlim=(0, 4.5), ticks=[(0, "0"), (1, "1"), (2, "2"), (3, "3"), (4, "4")],
    xlabel="Hours between interruptions", fs=10)

# Power ladder by deployment tier (frame 25; System Paradigms Sec. 2.1, as in
# Lecture 0). Slot 0.52 textwidth x 4.0 cm.
bars(out / "power_tiers.svg", [
    ("Cloud", 3e6, "neutral", "3 MW"),
    ("Edge", 200, "neutral", "200 W"),
    ("Mobile", 5, "neutral", "5 W"),
    ("TinyML", 1e-3, "neutral", "1 mW"),
], size=(1.85, 1.45), log=True, xlim=(1e-4, 1e9),
    ticks=[(1e-3, "1 mW"), (1, "1 W"), (1e6, "1 MW")],
    xlabel="Power (log scale)", fs=10)

# =============================================================================
# A composed assistant's critical path (frame 26).
# Slot 0.88 textwidth x 4.2 cm: canvas 900 x 290, 19 px renders near 7.8 pt.
# =============================================================================
g = Svg(900, 290)
nodes = {"ret": (10, 96, "Retrieval", f"{K.RET} ms"), "ta": (245, 6, "Tool A", f"{K.TOOL_A} ms"),
         "tb": (245, 186, "Tool B", f"{K.TOOL_B} ms"), "gen": (480, 96, "Generation", f"{K.GEN} ms"),
         "ver": (715, 96, "Verifier", f"{K.VER} ms")}
crit = {"ret", "ta", "gen", "ver"}
for key, (x, y, name, ms) in nodes.items():
    g.rect(x, y, 175, 92, "white", C["neutral"], sw=2)
    g.text(x + 87, y + 30, name, 23, C["text"], "bold")
    g.text(x + 87, y + 64, ms, 23, C["memory"], "bold")
    if key in crit:
        g.rect(x, y, 175, 92, "none", C["danger"], sw=4, first=2)
edges = [(185, 142, 243, 60, True), (185, 142, 243, 232, False),
         (420, 52, 478, 125, True), (420, 232, 478, 160, False),
         (655, 142, 713, 142, True)]
for x1, y1, x2, y2, c in edges:
    g.line(x1, y1, x2, y2, C["neutral"], sw=2.5)
    if c:
        g.line(x1, y1, x2, y2, C["danger"], sw=3.5, first=2)
g.text(890, 222, "Critical path", 20, C["danger"], "bold", "end", first=2)
g.text(890, 252, f"{K.RET} + {K.TOOL_A} + {K.GEN} + {K.VER} = {K.CRIT} ms", 20, C["danger"],
       "bold", "end", first=2)
# step 3: what can go wrong at each interface
for x, y in ((214, 101), (449, 88), (449, 196)):
    g.circle(x, y, 13, C["danger"], C["danger"], first=3)
    g.text(x, y, "×", 20, "white", "bold", first=3)
g.text(10, 70, "invented tool name", 19, C["danger"], "bold", "start", first=3)
g.text(560, 66, "malformed JSON", 19, C["danger"], "bold", "start", first=3)
g.text(468, 236, "timeout", 19, C["danger"], "bold", "start", first=3)
g.write("dag", 3)

# =============================================================================
# From one node to a fleet (frame 27).
# Slot 0.58 textwidth x 4.8 cm: canvas 900 x 480, 27 px renders near 7.4 pt.
# =============================================================================
g = Svg(900, 480)


def node(x, y, s, first=1, last=99, hl=False):
    g.rect(x, y, 70 * s, 50 * s, "#F6EEE9", CRIMSON if hl else "#B08A75",
           sw=3 if hl else 1.5, rx=5, first=first, last=last)
    for j in range(4):
        g.rect(x + (6 + j * 16) * s, y + 8 * s, 12 * s, 14 * s, C["compute_l"],
               C["compute"], sw=1, rx=2, first=first, last=last)
    g.rect(x + 6 * s, y + 30 * s, 58 * s, 13 * s, C["memory_l"], C["memory"], sw=1,
           rx=2, first=first, last=last)


node(60, 90, 4.4, last=1)
g.text(214, 440, "One node", 27, C["text"], "bold", last=1)
for r in range(4):
    for c in range(4):
        node(10 + c * 108, 10 + r * 100, 1.15, first=2, hl=(r == 1 and c == 1))
        if c < 3:
            g.line(10 + c * 108 + 80, 39 + r * 100, 10 + (c + 1) * 108, 39 + r * 100,
                   C["network"], sw=2.5, arrow=False, first=2)
    if r < 3:
        for c in range(4):
            g.line(50 + c * 108, 68 + r * 100, 50 + c * 108, 110 + r * 100, C["network"],
                   sw=2.5, arrow=False, first=2)
g.text(214, 440, "A fleet: nodes on a network", 27, C["network"], "bold", first=2)
rows = [("Bandwidth", "HBM in the node", "+ network bisection"),
        ("A failure", "an exception", "routine: fault tolerance"),
        ("Progress", "a step", "coordination")]
for i, (head_, one, many) in enumerate(rows):
    y = 40 + i * 150
    g.text(475, y, head_, 29, CRIMSON, "bold", "start")
    g.text(475, y + 44, one, 27, C["text"], "normal", "start", last=1)
    g.text(475, y + 44, many, 27, C["network"], "bold", "start", first=2)
g.write("fleet", 2)

# =============================================================================
# The voice-assistant request through four lighthouses (frame 30).
# Slot 0.95 textwidth x 3.9 cm: canvas 900 x 248, 17 px renders near 7.6 pt.
# The slide places one photo above each column (x = 2 + 228 i, width 215).
# =============================================================================
g = Svg(900, 248)
stages = [("Wake word", "KWS, microcontroller", ["200 KB INT8 in 1 MB", "20 MFLOP",
                                                 "milliwatts"], "Power", C["danger"]),
          ("Front end", "MobileNetV2-size, phone", ["600 MFLOP", "3.5 MB INT8", "5 W"],
           "Compute", C["compute"]),
          ("Ranking", "DLRM, cloud", ["100 GB of tables", "one H100: 80 GB"],
           "Capacity", C["network"]),
          ("Reply", "GPT-2, cloud H100", [f"{K.GPT2_BYTES / 1e9:.0f} GB per token",
                                             f"{K.GPT2_MEM_MS:.1f} ms floor",
                                             f"{K.GPT2_MEM_MS / K.GPT2_COMP_MS:.0f}× compute floor"],
           "Bandwidth", C["memory"])]
for i, (name, sub, nums, bind, col) in enumerate(stages):
    x = 2 + i * 228
    g.rect(x, 2, 215, 180, "white", C["neutral"], sw=2)
    g.text(x + 107, 26, name, 22, CRIMSON, "bold")
    g.text(x + 107, 54, sub, 17, C["neutral"])
    for j, s in enumerate(nums):
        g.text(x + 107, 90 + j * 28, s, 19, C["text"])
    if i < 3:
        g.line(x + 216, 92, x + 227, 92, C["neutral"], sw=2.5)
    g.rect(x + 12, 194, 191, 50, "white", C["neutral"], sw=1.5, dash="5 4", last=i + 1)
    g.text(x + 107, 219, "binding term?", 20, C["danger"], "bold", last=i + 1)
    g.rect(x + 12, 194, 191, 50, col, col, first=i + 2)
    g.text(x + 107, 219, bind, 24, "white", "bold", first=i + 2)
g.write("voice", 5)

# =============================================================================
# Text-free thumbnails for goals, takeaways, recall, and part locators: each
# keeps its figure's shapes with every word replaced by a faint bar.
# =============================================================================
import subprocess  # noqa: E402

THUMBS = ["chain_step2.svg", "chain_step3.svg", "roofline_step3.svg", "cycle_step2.svg",
          "mtbf.svg", "data_wall_step2.svg"] + \
         [f"cycle_part{p}.svg" for p in range(1, 5)]
subprocess.run([sys.executable, str(Path(__file__).resolve().parents[3] / "tools" /
                                    "greek_thumbs.py")] + [str(out / t) for t in THUMBS],
               check=True)

print("figures written to", out)
