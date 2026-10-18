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

Copied from other decks (credited on the slide as in the source deck):
gft_step2.svg and ironlaw_bars_step1.svg (Lecture 1), power_tiers_step4.svg
(Lecture 0), data_wall_step2.svg (Lecture 9), robot-arms-bmw-leipzig.jpg
(Lecture 2, see SOURCES.yml).

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


# =============================================================================
# The stack with one number per layer (retrieval reveal, frame 2)
# =============================================================================
g = Svg(900, 545)
layers = [
    ("Applications", "Black share enrolled: 17.7% by cost, 46.5% by need", "L15", 0.12),
    ("Operations", "$304M write-down, every server up", "L14", 0.17),
    ("Serving", "Batch 1 costs 7.5× batch 8", "L13", 0.22),
    ("Training", "16 bytes per parameter: PaLM ≈ 100 H100s", "L8", 0.12),
    ("Models", "7B model in FP16 = 14 GB; phone has 8 GB", "L10", 0.17),
    ("Frameworks", "Host 8.6 ms against 0.9 ms of weight reads", "L7", 0.22),
    ("Hardware", "TPU 30 to 80× perf/W; 1.5 TOPS/W sustained, not 20", "L11, L12", 0.12),
]
for i, (name, num, lec, op) in enumerate(layers):
    y = 20 + i * 64
    g.add(1, f'<rect x="20" y="{y}" width="780" height="54" rx="8" fill="{CRIMSON}" '
             f'fill-opacity="{op}"/>')
    g.text(38, y + 27, name, 23, CRIMSON, "bold", "start")
    g.text(215, y + 27, num, 20, C["text"], "normal", "start")
    g.text(788, y + 27, lec, 17, CRIMSON, "bold", "end")
g.add(1, f'<rect x="814" y="20" width="56" height="438" rx="8" fill="{CRIMSON}" '
         f'fill-opacity="0.18"/>')
g.add(1, f'<text {FONT} x="842" y="239" font-size="22" font-weight="700" fill="{CRIMSON}" '
         f'text-anchor="middle" dominant-baseline="middle" letter-spacing="3" '
         f'transform="rotate(-90 842 239)">DATA</text>')
g.text(870, 500, "Data: Tay suspended 16 hours after launch (L4);  "
       "GPT-2 saw each WebText token about 17 times (L9)", 18, C["text"], "normal", "end")
g.line(842, 470, 842, 482, CRIMSON, arrow=False, sw=2)
g.write("stack_numbers", 1)

# =============================================================================
# The fleet failure chain, tagged by lecture (frames 3, 6, 29)
# =============================================================================
g = Svg(900, 390)
stages = [("INT8\ncalibration", C["compute"], C["compute_l"], "Lecture 10b\ncalibration set"),
          ("Firmware\npreprocessing", C["neutral"], C["neutral_l"], "Lectures 4, 13b\ntraining-serving skew"),
          ("Input shift on\none cohort", C["neutral"], C["neutral_l"], "Lectures 4b, 14b\ndrift"),
          ("Accuracy\n−4 points", C["danger"], C["danger_l"], "Lecture 12b\naggregate metric only")]
for i, (lab, stroke, fill, tag) in enumerate(stages):
    x = 20 + i * 222
    g.rect(x, 30, 190, 100, fill, stroke, sw=2.5)
    g.text(x + 95, 80, lab, 22, stroke if i in (0, 3) else C["text"], "bold")
    if i < 3:
        g.line(x + 192, 80, x + 218, 80, C["neutral"], sw=3)
    g.text(x + 95, 168, tag, 17, CRIMSON, "bold", first=2)
g.text(450, 225, "Every team met its own target; no test crossed the interfaces.",
       19, C["neutral"], "normal", italic=True, first=2)
# step 3: the two measurements that would have caught it
g.rect(20, 260, 412, 104, C["data_l"], C["data"], first=3)
g.text(226, 292, "Skew check (principle 11)", 20, C["data"], "bold", first=3)
g.text(226, 330, "same captured images through\nboth preprocessing paths", 17,
       C["text"], first=3)
g.rect(464, 260, 412, 104, C["data_l"], C["data"], first=3)
g.text(670, 292, "Cohort-level monitoring", 20, C["data"], "bold", first=3)
g.text(670, 330, "accuracy by device and firmware,\nfrom day one", 17, C["text"], first=3)
g.write("chain", 3)

# =============================================================================
# One phone model through nine lectures (frame 7; chapter table 2)
# =============================================================================
g = Svg(900, 470)
phases = [
    ("Envelope", "Watt-scale power, heat, battery", "L2"),
    ("Architecture", "Depthwise separable:\n8.7× fewer FLOPs per layer", "L6"),
    ("Training", "Large batches to train;\nserves one image at a time", "L8"),
    ("Data selection", "Milder augmentation\nfor a small model", "L9"),
    ("Compression", "INT8: 1/4 the bytes of FP32", "L10b"),
    ("Acceleration", "Do the phone's NPU\nkernels run INT8?", "L11"),
    ("Benchmarking", "Gain must survive\npreprocessing and transfer", "L12"),
    ("Serving", "p99 under 50 ms", "L13"),
    ("Operations", "Accuracy tracked\nper device cohort", "L14b"),
]
for i, (name, num, lec) in enumerate(phases):
    row, col = divmod(i, 3)
    x, y = 20 + col * 300, 20 + row * 150
    first = row + 1
    g.rect(x, y, 260, 118, "white", C["neutral"], first=first, sw=2)
    g.text(x + 14, y + 26, name, 21, CRIMSON, "bold", "start", first=first)
    g.text(x + 248, y + 26, lec, 16, C["neutral"], "bold", "end", first=first)
    g.text(x + 130, y + 78, num, 17, C["text"], first=first)
    if col < 2:
        g.line(x + 262, y + 59, x + 296, y + 59, C["neutral"], first=first, sw=2.5)
for row in (0, 1):
    g.line(880, 20 + row * 150 + 120, 880, 20 + row * 150 + 145, C["neutral"],
           arrow=False, first=row + 2, sw=2)
    g.line(880, 20 + row * 150 + 145, 20 + 130, 20 + row * 150 + 145, C["neutral"],
           arrow=False, first=row + 2, sw=2, dash="5 4")
    g.line(150, 20 + row * 150 + 145, 150, 20 + (row + 1) * 150 - 2, C["neutral"],
           first=row + 2, sw=2)
# step 3: the propagation the chapter traces
g.line(430, 140, 430, 168, C["danger"], first=3, sw=3)
g.text(430, 455, "Depthwise layers go bandwidth-bound, so the next lever is bytes (INT8),"
       " which pays only if the NPU runs it.", 16, C["danger"], "bold", first=3)
g.write("journey", 3)

# =============================================================================
# Roofline: one model, two regimes; INT8 raises the ridge (frame 9)
# =============================================================================
import numpy as np  # noqa: E402

BW, P16, P8 = 3.35, 989.0, 1979.0
xs = np.logspace(np.log10(0.4), np.log10(30000), 400)
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    ax.plot(xs, np.minimum(P16, BW * xs), color=C["memory"], lw=3, zorder=2)
    ax.plot(xs[xs >= P16 / BW], np.full((xs >= P16 / BW).sum(), P16),
            color=C["compute"], lw=3, zorder=3)
    ax.axvline(P16 / BW, color=C["neutral"], lw=1, ls="--")
    ax.text(P16 / BW / 1.08, 1.4, "ridge 295", color=C["neutral"], fontsize=10, ha="right")
    on3 = 1.0 if k >= 3 else 0.0
    m = xs >= P8 / BW
    ax.plot(xs[m], np.full(m.sum(), P8), color=C["danger"], lw=2.5, ls="--", alpha=on3)
    ax.plot(xs[(xs >= P16 / BW) & (xs <= P8 / BW)],
            BW * xs[(xs >= P16 / BW) & (xs <= P8 / BW)], color=C["danger"], lw=2.5,
            ls="--", alpha=on3)
    ax.axvline(P8 / BW, color=C["danger"], lw=1, ls=":", alpha=on3)
    ax.text(P8 / BW * 1.08, 1.4, "INT8 ridge 591", color=C["danger"], fontsize=10,
            alpha=on3)
    ax.text(28000, P8 * 1.1, "INT8 peak 1,979", color=C["danger"], fontsize=10, ha="right", va="bottom",
            fontweight="bold", alpha=on3)
    ax.text(28000, P16 / 1.1, "FP16 peak 989", color=C["compute"], fontsize=10, ha="right", va="top",
            fontweight="bold")
    # step 1: decode
    ax.plot([1], [BW], "o", ms=11, color=C["memory"], mec="white", mew=1.5, zorder=5)
    ax.annotate("GPT-2 XL decode,\nbatch 1: 1", (1, BW), xytext=(14, 0),
                textcoords="offset points", va="center", fontsize=11, fontweight="bold",
                color=C["memory"])
    a2 = 1.0 if k >= 2 else 0.0
    ax.plot([1046.7], [P16], "o", ms=11, color=C["compute"], mec="white", mew=1.5,
            zorder=5, alpha=a2)
    ax.annotate("Same model,\nbatched prefill: 1,047", (1046.7, P16), xytext=(6, -36),
                textcoords="offset points", ha="left", va="top", fontsize=11, fontweight="bold",
                color=C["compute"], alpha=a2)
    ax.plot([569], [BW * 569], "o", ms=11, color=C["danger"], mec="white", mew=1.5,
            zorder=6, alpha=on3)
    ax.annotate("ResNet-50 conv, INT8\nweights: 569, below 591", (569, BW * 569),
                xytext=(-30, 14), textcoords="offset points", ha="right", va="bottom",
                fontsize=11, fontweight="bold", color=C["danger"], alpha=on3)
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
# The thirteen principles as a cycle: slide variant of book figure 1
# =============================================================================
PARTS = [
    ("Foundations (Data)", 20, 20, [(1, "Data as code", "heuristic"),
                                    (2, "Data gravity", "heuristic")]),
    ("Development (Model)", 520, 20, [(3, "Iron law", "decomposition"),
                                      (4, "Silicon contract", "heuristic")]),
    ("Optimization", 520, 270, [(5, "Pareto frontier", "bound"),
                                (6, "Arithmetic intensity", "bound"),
                                (7, "Energy movement", "decomposition"),
                                (8, "Amdahl's law", "bound")]),
    ("Deployment", 20, 270, [(9, "Verification gap", "policy"),
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
    g = Svg(900, 500)
    for p, (title, x, y, items) in enumerate(PARTS):
        dim = lit is not None and p != lit
        op = 0.28 if dim else 1.0
        on = lit is not None and p == lit
        g.rect(x, y, 360, 210, "#FBEFF1" if on else "white",
               CRIMSON if on else "#BBBBBB", sw=3 if on else 2, opacity=op)
        g.text(x + 18, y + 30, title, 24, CRIMSON, "bold", "start", opacity=op)
        for j, (num, name, kind) in enumerate(items):
            ly = y + 70 + j * 29
            for pr, f in (trace or []):
                if pr == num:
                    g.rect(x + 8, ly - 14, 344, 28, C["compute_l"], C["compute"],
                           sw=1.5, rx=5, first=f)
            g.text(x + 46, ly, str(num), 21, C["neutral"], "bold", "end", opacity=op)
            g.text(x + 58, ly, name, 21, C["text"], "normal", "start", opacity=op)
            if kinds_from:
                g.text(x + 346, ly, kind, 16, C["neutral"], "normal", "end",
                       first=kinds_from, italic=True, opacity=op)
    for x1, y1, x2, y2 in ((385, 115, 513, 115), (700, 234, 700, 266),
                           (515, 385, 387, 385), (200, 266, 200, 234)):
        g.line(x1, y1, x2, y2, ARROW, sw=9)
    g.circle(450, 250, 62, "white", C["neutral"], sw=1.5, dash="4 3")
    g.text(450, 250, "Conservation\nof complexity", 16, C["text"], "bold")
    g.write(stem, steps)


cycle("cycle", steps=2, kinds_from=2)
for p in range(4):
    cycle(f"cycle_part{p + 1}", lit=p)
cycle("cycle_trace", steps=2, trace=[(5, 1), (4, 1), (6, 1), (7, 1),
                                     (12, 2), (9, 2), (11, 2)])

# =============================================================================
# Amdahl: a 10x faster kernel inside a request that is 90 percent host
# =============================================================================
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 1.6))
    rows = [("Before", 90, 10, "+ inference 10% = 100%"), ("After 10× kernel", 90, 1, "+ 1% = 91%: 1.1× faster")]
    for i, (lab, host, inf, tot) in enumerate(rows):
        y = 1 - i
        on = 1.0 if (i == 0 or k == 2) else 0.0
        ax.barh(y, host, color=C["neutral"], height=0.6, alpha=on)
        ax.barh(y, inf, left=host, color=C["compute"], height=0.6, alpha=on)
        ax.text(45, y, "Host preprocessing 90%", ha="center", va="center", color="white",
                fontsize=12, fontweight="bold", alpha=on)
        ax.text(-1, y, lab, ha="right", va="center", fontsize=12, fontweight="bold",
                color=C["text"], alpha=on)
        ax.text(host + inf + 1.5, y, tot,
                va="center", fontsize=12, fontweight="bold",
                color=C["compute"] if i == 0 else C["danger"], alpha=on)
    ax.set_yticks([])
    ax.set_xlim(0, 150)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(-0.6, 1.6)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Share of the original request time (%)")
    sp.save(fig, out / f"amdahl_step{k}.svg")

# =============================================================================
# Tails: illustrative scenario, then a measured Google service
# =============================================================================
sp.hbars(str(out / "tail_illus.svg"), [
    ("Mean TTFT", 50, "neutral", "50 ms"),
    ("p99 TTFT", 2000, "danger", "2,000 ms (40×)"),
], "ms", log=False, xlabel="Time to first token (ms)", size=(3.4, 1.6))
sp.hbars(str(out / "tail_measured.svg"), [
    ("One leaf, median", 1, "memory", "1 ms"),
    ("One leaf, p99", 10, "memory", "10 ms"),
    ("All leaves, median", 40, "danger", "40 ms"),
    ("All leaves, p99", 140, "danger", "140 ms (14×)"),
], "ms", log=False, xlabel="Latency (ms)", size=(3.4, 2.1))

# =============================================================================
# From one GPU to a pool, and the measured Llama 3 run
# =============================================================================
sp.hbars(str(out / "mtbf.svg"), [
    ("One GPU (MTTF)", 50000, "neutral", "50,000 h (5.7 years)"),
    ("A three-week run", 504, "memory", "504 h"),
    ("1,024 GPUs (MTBF)", 48.8, "danger", "48.8 h (2 days)"),
], "h", log=True, xlabel="Hours (log scale)", size=(5.8, 2.0))
sp.hbars(str(out / "llama_mtbf.svg"), [
    ("Model, 16,384 GPUs", 3.05, "neutral", "3.05 h"),
    ("Llama 3, measured", 3.1, "danger", "3.1 h"),
], "h", log=False, xlabel="Hours between interruptions (derived)", size=(3.6, 1.3))

# =============================================================================
# A composed assistant's critical path (frame 26)
# =============================================================================
g = Svg(900, 330)
nodes = {"ret": (20, 120, "Retrieval", "50 ms"), "ta": (250, 40, "Tool A", "110 ms"),
         "tb": (250, 200, "Tool B", "70 ms"), "gen": (480, 120, "Generation", "180 ms"),
         "ver": (710, 120, "Verifier", "30 ms")}
crit = {"ret", "ta", "gen", "ver"}
for key, (x, y, name, ms) in nodes.items():
    g.rect(x, y, 170, 90, "white", C["neutral"], sw=2)
    g.text(x + 85, y + 30, name, 21, C["text"], "bold")
    g.text(x + 85, y + 62, ms, 21, C["memory"], "bold")
    if key in crit:
        g.rect(x, y, 170, 90, "none", C["danger"], sw=4, first=2)
edges = [(190, 165, 248, 95, True), (190, 165, 248, 235, False),
         (420, 85, 478, 155, True), (420, 245, 478, 180, False),
         (650, 165, 708, 165, True)]
for x1, y1, x2, y2, c in edges:
    g.line(x1, y1, x2, y2, C["neutral"], sw=2.5)
    if c:
        g.line(x1, y1, x2, y2, C["danger"], sw=3.5, first=2)
g.text(795, 270, "Critical path", 18, C["danger"], "bold", first=2)
g.text(795, 300, "50 + 110 + 180 + 30", 18, C["danger"], "bold", first=2)
# step 3: what can go wrong at each interface
for x, y in ((219, 130), (449, 120), (449, 212)):
    g.circle(x, y, 13, C["danger"], C["danger"], first=3)
    g.text(x, y, "×", 20, "white", "bold", first=3)
g.text(105, 95, "invented tool name", 17, C["danger"], "bold", first=3)
g.text(520, 100, "malformed JSON", 17, C["danger"], "bold", first=3)
g.text(470, 240, "timeout", 17, C["danger"], "bold", "start", first=3)
g.write("dag", 3)

# =============================================================================
# From one node to a fleet (frame 27)
# =============================================================================
g = Svg(900, 360)


def node(x, y, s, first=1, last=99, hl=False):
    g.rect(x, y, 70 * s, 50 * s, "#F6EEE9", CRIMSON if hl else "#B08A75",
           sw=3 if hl else 1.5, rx=5, first=first, last=last)
    for j in range(4):
        g.rect(x + (6 + j * 16) * s, y + 8 * s, 12 * s, 14 * s, C["compute_l"],
               C["compute"], sw=1, rx=2, first=first, last=last)
    g.rect(x + 6 * s, y + 30 * s, 58 * s, 13 * s, C["memory_l"], C["memory"], sw=1,
           rx=2, first=first, last=last)


node(150, 100, 3, last=1)
g.text(255, 300, "One node", 22, C["text"], "bold", last=1)
for r in range(4):
    for c in range(5):
        node(20 + c * 98, 20 + r * 76, 1, first=2, hl=(r == 1 and c == 1))
        if c < 4:
            g.line(20 + c * 98 + 70, 45 + r * 76, 20 + (c + 1) * 98, 45 + r * 76,
                   C["network"], sw=2, arrow=False, first=2)
    if r < 3:
        for c in range(5):
            g.line(55 + c * 98, 70 + r * 76, 55 + c * 98, 96 + r * 76, C["network"],
                   sw=2, arrow=False, first=2)
g.text(255, 340, "A fleet: nodes joined by a network", 21, C["network"], "bold", first=2)
rows = [("Bandwidth", "HBM inside the node", "+ network bisection"),
        ("A failure", "an exception", "routine: fault tolerance"),
        ("Progress", "a step", "coordination")]
for i, (head, one, many) in enumerate(rows):
    y = 60 + i * 100
    g.text(530, y, head, 24, CRIMSON, "bold", "start")
    g.text(530, y + 34, one, 23, C["text"], "normal", "start", last=1)
    g.text(530, y + 34, many, 23, C["network"], "bold", "start", first=2)
g.write("fleet", 2)

# =============================================================================
# The voice-assistant request through four lighthouses (frame 28)
# =============================================================================
g = Svg(900, 400)
stages = [("Wake word", "KWS, microcontroller", ["200 KB INT8", "in 1 MB flash", "20 MFLOP",
                                                 "milliwatts"], "Power", C["danger"]),
          ("Front end", "MobileNetV2-sized, phone", ["600 MFLOP", "3.5 MB INT8", "5 W"],
           "Compute", C["compute"]),
          ("Ranking", "DLRM, cloud", ["100 GB of tables", "one H100: 80 GB"],
           "Capacity", C["network"]),
          ("Reply", "GPT-2, cloud H100", ["3 GB per token", "0.9 ms floor",
                                          "295× compute floor"], "Bandwidth", C["memory"])]
for i, (name, sub, nums, bind, col) in enumerate(stages):
    x = 15 + i * 223
    g.rect(x, 20, 198, 250, "white", C["neutral"], sw=2)
    g.text(x + 99, 50, name, 22, CRIMSON, "bold")
    g.text(x + 99, 80, sub, 15, C["neutral"])
    for j, s in enumerate(nums):
        g.text(x + 99, 125 + j * 34, s, 18, C["text"])
    if i < 3:
        g.line(x + 200, 145, x + 221, 145, C["neutral"], sw=2.5)
    g.rect(x + 14, 295, 170, 60, "white", C["neutral"], sw=1.5, dash="5 4", last=i + 1)
    g.text(x + 99, 325, "???", 24, C["danger"], "bold", last=i + 1)
    g.rect(x + 14, 295, 170, 60, col, col, first=i + 2)
    g.text(x + 99, 325, bind, 24, "white", "bold", first=i + 2)
g.text(450, 383, "binding term", 15, C["neutral"], italic=True)
g.write("voice", 5)

print("figures written to", out)
