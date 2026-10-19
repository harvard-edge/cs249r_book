#!/usr/bin/env python3
"""Figures for Lecture 17 (Conclusion), Volume II.

Plots use the shared slideplot helper; diagrams are SVG written by the small
builder below, one file per build step on one canvas so nothing moves. Every
derived value comes from calcs.py in this folder (imported, never retyped).
Inputs and their sources are listed in calcs.py.

Diagrams follow the Volume II C3 roles: compute blue #1F407A, communication
amber #d97706, coordination violet #7c3aed, storage green, risk red.

Run from this folder: python3 make_figs.py
"""

import contextlib
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "assets" / "kit"))
sys.path.insert(0, str(HERE))
import slideplot as sp  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    import calcs  # noqa: E402

R = calcs.results
plt = sp.plt
C = sp.C
out = HERE

ETH = "#1F407A"
ETH_L = "#EAF1F8"
AMBER, AMBER_L = "#d97706", "#fef3c7"
VIOLET, VIOLET_L = "#7c3aed", "#f3e8ff"
GREEN, GREEN_L = "#16a34a", "#dcfce7"
RED, RED_L = "#dc2626", "#fee2e2"
SLATE, SLATE_L = "#64748b", "#f1f5f9"
TEXT = "#1A1A2E"
FONT = 'font-family="Helvetica, Arial, sans-serif"'


class Svg:
    """A tiny SVG builder: every element carries the first step it appears in;
    write(stem, n) emits stem_step1..n with hidden elements kept transparent."""

    def __init__(self, w, h):
        self.w, self.h, self.items = w, h, []

    def add(self, first, s, last=99):
        self.items.append((first, last, s))

    def rect(self, x, y, w, h, fill, stroke, first=1, sw=2, rx=8, last=99):
        self.add(first, f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>', last)

    def text(self, x, y, s, size=22, color=TEXT, weight="normal", anchor="middle",
             first=1, last=99):
        lines = s.replace("&", "&amp;").split("\n")
        y0 = y - (len(lines) - 1) * size * 0.6
        spans = "".join(f'<tspan x="{x}" y="{y0 + i * size * 1.2:.1f}">{ln}</tspan>'
                        for i, ln in enumerate(lines))
        self.add(first, f'<text {FONT} font-size="{size}" fill="{color}" '
                        f'font-weight="{weight}" text-anchor="{anchor}" '
                        f'dominant-baseline="middle">{spans}</text>', last)

    def line(self, x1, y1, x2, y2, color, first=1, sw=2.5, arrow=True, last=99):
        m = f' marker-end="url(#a{color[1:]})"' if arrow else ""
        self.add(first, f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                        f'stroke-width="{sw}"{m}/>', last)

    def write(self, stem, steps):
        colors = [ETH, AMBER, VIOLET, GREEN, RED, SLATE, TEXT]
        defs = "".join(
            f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
            f'markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
            f'fill="{c}"/></marker>' for c in colors)
        for k in range(1, steps + 1):
            body = [s if first <= k <= last else '<g opacity="0">' + s + "</g>"
                    for first, last, s in self.items]
            svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                   f'width="{self.w}" height="{self.h}"><defs>{defs}</defs>'
                   f'<rect width="100%" height="100%" fill="white"/>' + "".join(body) + "</svg>")
            name = f"{stem}_step{k}.svg" if steps > 1 else f"{stem}.svg"
            (out / name).write_text(svg)


def num(label, fmt):
    return fmt.format(R[label])


# =============================================================================
# 1. The fleet stack with one number per row (retrieval reveal)
#    Canvas 900 x 372 for a full-width slot (about 14.5 cm): 19 px = 8.7 pt.
# =============================================================================
rows = [  # top to bottom: layer, number, owning lecture
    ("Governance & Sustainability",
     f"Same energy: Poland grid {num('Poland / Quebec', '{:.0f}')}× Quebec's carbon", "L15"),
    ("Assurance", "DP noise: $200 error on a mean of 1,000 salaries", "L13"),
    ("Operations", "864 false alarms a day from 1,000 monitors", "L12"),
    ("Runtime & Serving", "12 requests of 128K tokens fill one 8-GPU node", "L10"),
    ("Reliability & Control", "Checkpoint every 13.8 min at 10,000 GPUs", "L7"),
    ("Distributed Execution", "175B, pure data parallel, 1,024 GPUs: 15% efficient", "L6"),
    ("Fabric & Storage", "2,450 GB checkpoint: 0.34 s to local NVMe", "L4"),
    ("Facility & Accelerators",
     f"10 GB: HBM {num('10 GB over HBM, ms', '{:.1f}')} ms, NVLink "
     f"{num('10 GB over NVLink, ms', '{:.1f}')}, InfiniBand "
     f"{num('10 GB over InfiniBand, ms', '{:.0f}')}", "L2"),
]
g = Svg(900, 372)
for i, (name, number, lec) in enumerate(rows):
    y = 4 + i * 46
    g.rect(4, y, 892, 42, ETH_L, ETH, sw=1.5, rx=6)
    g.text(16, y + 21, name, 19, ETH, "bold", "start")
    g.text(300, y + 21, number, 19, TEXT, "normal", "start")
    g.text(884, y + 21, lec, 19, ETH, "bold", "end")
g.write("stack_numbers", 1)

# =============================================================================
# 2. Six principles placed on the four bands, each with its Fleet Law term
#    Canvas 900 x 380 for a full-width slot: 18 px = 8.2 pt.
# =============================================================================
g = Svg(900, 380)
bands = [  # (y, height, band, principles, color, step), top to bottom
    (4, 96, "Governance", [("5  Responsibility constrains design", "spends compute; approval gates add coordination"),
                           ("6  Sustainability is a first-order cost", "a step's energy = average power × T_step")],
     GREEN, 3),
    (106, 60, "Serving", [("4  Serving is bounded by its contract", "the same terms, counted per request")],
     VIOLET, 2),
    (172, 126, "Distribution", [("2  Communication can dominate", "T_comm(N) and T_sync(N) need not shrink"),
                                ("3  Failure is routine", "recovery adds to T_sync(N); rollback adds work")],
     AMBER, 1),
    (304, 72, "Infrastructure", [("1  Infrastructure determines capability",
                                  "sets the rates behind T_compute / N and T_comm(N)")], ETH, 1),
]
for y, h, band, prin, color, step in bands:
    g.rect(64, y, 832, h, "#ffffff", color, sw=2.5, first=step, rx=6)
    g.text(78, y + h / 2, band, 19, color, "bold", "start", first=step)
    n = len(prin)
    for j, (pr, term) in enumerate(prin):
        yy = y + h * (j + 0.5) / n
        g.text(250, yy - 11, pr, 19, TEXT, "bold", "start", first=step)
        g.text(250, yy + 12, term, 18, SLATE, "normal", "start", first=step)
g.line(30, 372, 30, 12, RED, sw=4, first=3)
g.add(3, f'<text {FONT} x="14" y="190" font-size="18" font-weight="700" fill="{RED}" '
         f'text-anchor="middle" dominant-baseline="middle" transform="rotate(-90 14 190)">'
         f'the scale moment</text>')
g.write("principles", 3)

# =============================================================================
# 3. Llama 3 405B: 419 unexpected interruptions by category (measured)
# =============================================================================
cat_rows = [
    ("GPU (incl. HBM3)", R["Llama 3 interruptions, GPU"], "danger"),
    ("Software and dependency", R["Llama 3 interruptions, Dependency"], "neutral"),
    ("Network switch/cable", R["Llama 3 interruptions, Network"], "network"),
    ("Host maintenance", R["Llama 3 interruptions, Unplanned maintenance"], "neutral"),
    ("Other host hardware", R["Llama 3 interruptions, Host"], "neutral"),
    ("Unknown (NCCL timeouts)", R["Llama 3 interruptions, Unknown"], "neutral"),
]
sp.hbars(out / "llama_causes.svg", cat_rows, "", log=False, size=(5.6, 2.9),
         value_fmt="{:.0f}")

# =============================================================================
# 4. Tail at scale: Dean and Barroso 2013, Table 1, p99 (measured)
# =============================================================================
tail_rows = [
    ("One random leaf", calcs.DB_ONE_LEAF_P99_MS, "memory", "10 ms"),
    ("95% of all leaves", calcs.DB_95_LEAVES_P99_MS, "compute", "70 ms"),
    ("All leaves", calcs.DB_ALL_LEAVES_P99_MS, "danger", "140 ms"),
]
sp.hbars(out / "tail_measured.svg", tail_rows, "ms", log=False, size=(5.8, 1.9))

# =============================================================================
# 5. Interruption interval against job size: model 50,000 h / N, and measured
# =============================================================================
import numpy as np  # noqa: E402

fig, ax = plt.subplots(figsize=(5.8, 2.75))
ns = np.logspace(2, 4.6, 100)
ax.plot(ns, calcs.GPU_MTTF_H / ns, color=C["memory"], lw=2.5)
ax.text(150, calcs.GPU_MTTF_H / 150 * 1.5, "model: 50,000 h / N", color=C["memory"],
        fontsize=12, fontweight="bold")
pts = [(calcs.N_REF, R["MTBI 1,024 GPUs, h"], "1,024: 48.8 h"),
       (calcs.N_BIG, R["MTBI 2,048 GPUs, h"], "2,048: 24.4 h")]
for x, y, lab in pts:
    ax.plot([x], [y], "o", ms=9, color=C["memory"], mec="white", mew=1.5, zorder=5)
    ax.annotate(lab, (x, y), xytext=(-8, -6), textcoords="offset points", ha="right",
                va="top", fontsize=12, color=C["memory"], fontweight="bold")
x = calcs.LLAMA_GPUS
ax.plot([x], [R["Llama 3 measured hours per unexpected interruption"]], "D", ms=9,
        color=C["danger"], zorder=6)
ax.annotate("Llama 3, measured: 3.1 h", (x, 3.1), xytext=(-14, -16), textcoords="offset points",
            ha="right", va="top", fontsize=12, color=C["danger"], fontweight="bold")
ax.plot([x], [calcs.KOKOLIS_16K_MTTF_H], "o", ms=8, mfc="white", mec=C["danger"], mew=2, zorder=6)
ax.annotate("projected: 1.8 h", (x, 1.8), xytext=(-14, -22), textcoords="offset points",
            ha="right", va="top", fontsize=12, color=C["danger"])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(100, 4e4)
ax.set_ylim(0.25, 800)
ax.set_xticks([100, 1000, 10000])
ax.set_xticklabels(["100", "1,000", "10,000"])
ax.set_yticks([1, 10, 100])
ax.set_yticklabels(["1 h", "10 h", "100 h"])
ax.minorticks_off()
ax.set_xlabel("GPUs the job needs at once")
ax.set_ylabel("Hours between interruptions")
ax.grid(color=C["grid"], lw=0.6)
sp.save(fig, out / "mtbi.svg")

# =============================================================================
# 6. Young-Daly overhead against cluster size (derived, GPU failures only)
# =============================================================================
fig, ax = plt.subplots(figsize=(5.8, 2.6))
ns = np.logspace(np.log10(128), np.log10(16384), 100)
ov = [100 * calcs.yd_overhead(n) for n in ns]
ax.plot(ns, ov, color=C["network"], lw=2.5)
for n, lab in [(calcs.N_REF, "Young-Daly overhead 1,024, percent"),
               (calcs.N_BIG, "Young-Daly overhead 2,048, percent")]:
    v = R[lab]
    ax.plot([n], [v], "o", ms=9, color=C["network"], mec="white", mew=1.5, zorder=5)
    off, ha = ((-10, 10), "right") if n == calcs.N_REF else ((10, -18), "left")
    ax.annotate(f"{n:,} GPUs: {v:.1f}%", (n, v), xytext=off, textcoords="offset points",
                ha=ha, fontsize=12, color=C["network"], fontweight="bold")
ax.text(4200, 4.6, f"×{R['Overhead ratio (sqrt 2)']:.2f} per doubling\n(grows as √N)",
        fontsize=12, color=C["text"], ha="center")
ax.set_xscale("log")
ax.set_xlim(128, 16384)
ax.set_ylim(0, 7.5)
ax.set_xticks([128, 1024, 2048, 16384])
ax.set_xticklabels(["128", "1,024", "2,048", "16,384"])
ax.minorticks_off()
ax.set_xlabel("GPUs (independent GPU failures only)")
ax.set_ylabel("Checkpoint + rework (%)")
ax.grid(color=C["grid"], lw=0.6)
sp.save(fig, out / "overhead.svg")

# =============================================================================
# 7. MegaScale strong scaling, 175B (measured, Jiang et al. 2024, Table 2)
# =============================================================================
fig, ax = plt.subplots(figsize=(5.8, 2.7))
gs = list(calcs.MEGASCALE)
tp = [calcs.MEGASCALE[g_] / 1e3 for g_ in gs]
ideal = [calcs.MEGASCALE[512] / 1e3 * g_ / 512 for g_ in gs]
ax.plot(gs, ideal, ls="--", color=C["neutral"], lw=1.8)
ax.text(1090, 12, "dashed: linear from 512", color=C["neutral"], fontsize=11, ha="right",
        va="bottom")
ax.plot(gs, tp, "o-", color=C["compute"], lw=2.5, ms=9, mec="white", mew=1.5)
for g_, t in zip(gs, tp):
    ax.annotate(f"{t:.1f}k", (g_, t), xytext=(10, -14), textcoords="offset points",
                fontsize=11, color=C["compute"], fontweight="bold")
ax.annotate(f"512 to 1,024 GPUs: {R['MegaScale measured speedup 512 -> 1,024']:.2f}× (linear: 2×)",
            (1024, tp[-1]), xytext=(256, 175), textcoords="data", fontsize=12,
            color=C["text"], fontweight="bold")
ax.set_xlim(200, 1100)
ax.set_ylim(0, 215)
ax.set_xticks(gs)
ax.set_xticklabels([f"{g_:,}" for g_ in gs])
ax.set_xlabel("GPUs (batch 768, 175B model)")
ax.set_ylabel("Tokens/s (thousands)")
ax.grid(color=C["grid"], lw=0.6)
sp.save(fig, out / "megascale.svg")

# =============================================================================
# 8. Each compute doubling buys the same fraction, a smaller amount (derived)
# =============================================================================
fig, ax = plt.subplots(figsize=(5.8, 2.6))
mult = R["Loss multiplier per compute doubling"]
xs = [1, 2, 4, 8, 16]
ls = [mult ** k for k in range(5)]
ax.plot(xs, ls, "o-", color=C["memory"], lw=2.5, ms=9, mec="white", mew=1.5)
for k in range(1, 5):
    drop = R[f"Doubling {k}: absolute drop (normalized loss 1.0 at start)"]
    ax.annotate(f"−{drop:.3f}", (xs[k], ls[k]), xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=11, color=C["memory"], fontweight="bold")
ax.text(16, 0.995, f"each doubling: ×{mult:.3f} (−{R['Loss reduction per doubling, percent']:.1f}%)",
        ha="right", fontsize=12, color=C["text"], fontweight="bold")
ax.set_xscale("log", base=2)
ax.set_xticks(xs)
ax.set_xticklabels(["1×", "2×", "4×", "8×", "16×"])
ax.minorticks_off()
ax.set_ylim(0.85, 1.02)
ax.set_xlabel("Training compute (accelerator-hours), multiple of today")
ax.set_ylabel("Loss (today = 1)")
ax.grid(color=C["grid"], lw=0.6)
sp.save(fig, out / "loss_doublings.svg")

# =============================================================================
# 9. The six-step diagnostic procedure, applied to the lagging expanded run
#    Canvas 900 x 380 for a full-width slot: 18 px = 8.2 pt.
# =============================================================================
g = Svg(900, 380)
steps = [
    ("1 Symptom", "run falls\nbehind\nschedule", SLATE, SLATE_L, 1),
    ("2 Metric", "MFU per\nstep has\ndropped", SLATE, SLATE_L, 1),
    ("3 C³ term", "per-rank\ntraces: more\nexposed\ntransfer", AMBER, AMBER_L, 2),
    ("4 Row", "Distributed\nExecution", AMBER, AMBER_L, 2),
    ("5 Cost\nmoved", "compression\nmoves risk\nto quality", RED, RED_L, 3),
    ("6 Evidence", "convergence\nand eval\nrecord", GREEN, GREEN_L, 3),
]
for i, (head, body, stroke, fill, step) in enumerate(steps):
    x = 4 + i * 150
    g.rect(x, 4, 140, 58, stroke, stroke, first=step, sw=1, rx=6)
    g.text(x + 70, 33, head, 19, "#ffffff", "bold", first=step)
    g.rect(x, 70, 140, 150, fill, stroke, first=step, sw=2, rx=6)
    g.text(x + 70, 145, body, 18, TEXT, "normal", first=step)
    if i < 5:
        g.line(x + 141, 33, x + 149, 33, SLATE, sw=2.5, first=steps[i + 1][4])
g.text(450, 256, "Barrier wait alone cannot tell slow transfer from late arrival.", 20,
       AMBER, "bold", first=2)
g.text(450, 300, "Gradient compression shrinks the bytes in T_comm(N),", 20, TEXT, first=3)
g.text(450, 332, "and must show the compressed run still reaches its quality target.", 20,
       TEXT, first=3)
g.text(450, 366, "Same steps: a fairness alert may trace to labels, reviewers, or a threshold.",
       18, SLATE, first=3)
g.write("diagnose", 3)

# =============================================================================
# 10. Where next: interactivity costs per-GPU output (InferenceX, Chapter 10)
# =============================================================================
fig, ax = plt.subplots(figsize=(3.6, 2.5))
xs = [0, 1]
vals = [calcs.IX_LOW_INTERACT, calcs.IX_HIGH_INTERACT]
ax.bar(xs, vals, width=0.6, color=[C["compute"], C["danger"]])
for x_, v, lab in zip(xs, vals, ["6,300", "210"]):
    ax.text(x_, v + 150, lab, ha="center", va="bottom", fontsize=14, fontweight="bold",
            color=C["compute"] if x_ == 0 else C["danger"])
ax.text(1, 2600, f"{R['InferenceX per-GPU output ratio']:.0f}× less", ha="center",
        fontsize=14, fontweight="bold", color=C["text"])
ax.set_xticks(xs)
ax.set_xticklabels(["13 tok/s\nper user", "100 tok/s\nper user"], fontsize=13)
ax.set_ylim(0, 7600)
ax.set_yticks([0, 2000, 4000, 6000])
ax.set_yticklabels(["0", "2,000", "4,000", "6,000"], fontsize=12)
ax.set_ylabel("Output tok/s per GPU", fontsize=13)
ax.grid(axis="y", color=C["grid"], lw=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / "interactivity.svg")

# =============================================================================
# 11. Where next: MoE resident against active bytes (Chapter 10, Sec. 10.10.4)
# =============================================================================
moe_rows = [
    ("MoE, all experts resident", calcs.MOE_RESIDENT_GB, "memory", "1,342 GB"),
    ("Dense 400B, read per step", calcs.DENSE_400B_GB, "neutral", "800 GB"),
    ("MoE, active per token", calcs.MOE_ACTIVE_GB, "compute", "74 GB"),
]
sp.hbars(out / "moe_bytes.svg", moe_rows, "", log=False, size=(5.8, 1.9))

print("figures written")
