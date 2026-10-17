#!/usr/bin/env python3
"""Data plots for Lecture 0 (course overview).

Every value is sourced:
  - Energy of one chat answer: Napkin Math 13.5 (Model Serving). 1,000 W node
    at 912 tokens/s gives about 1.1 J/token, about 548 J for 500 tokens; at
    10 percent GPU utilization, about 6.6 J/token. Google's measured median
    Gemini text prompt is 0.24 Wh = 864 J (Elsworth et al. 2025, cited there).
    Charging a phone is about 40,000 J (same box).
  - Daily chat prompts: about 2.5 billion (OpenAI to Axios, July 2025).
  - Datacenter electricity: 415 TWh in 2024, about 945 TWh projected for 2030
    (IEA, Energy and AI, 2025, executive summary).
  - Energy per operation: DRAM access 160 pJ, FP16 multiply 1.1 pJ, INT8
    multiply 0.2 pJ (Introduction, Sec. 1.7, after Horowitz 2014).
  - Deployment power: cloud 3 MW, edge 200 W, mobile 5 W, TinyML 1 mW
    (System Paradigms, Sec. 2.1).

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

out = Path(__file__).parent

# Linear axes print thousands with a comma separator ("10,000", not "10000").
# slideplot does not take a tick formatter, so the deck wraps its save step.
_save = sp.save


def _save_with_commas(fig, path):
    for ax in fig.axes:
        if ax.get_xscale() == "linear":
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _save(fig, path)


sp.save = _save_with_commas

# One chat answer against everyday energy, on a linear axis so the ratio
# reads true (built up in three steps)
sp.hbars_steps(out / "energy_answer", [
    ("One answer (estimate)", 548, "compute"),
    ("One prompt (Google, measured)", 864, "compute"),
    ("Charging a phone", 40000, "neutral", "40,000 J  = 73 answers"),
], "J", log=False, xlabel="Energy (joules)")

# Fleet scale: about 2.5 billion ChatGPT prompts a day (OpenAI to Axios,
# July 2025) at Google's measured 0.24 Wh each is about 600 MWh a day,
# against 1,287 MWh to train GPT-3 (Patterson et al. 2021). Estimate: the
# per-prompt figure is Gemini's median, applied to another service.
sp.hbars_steps(out / "fleet_daily", [
    ("Chat prompts, one day (estimate)", 600, "compute", "about 600 MWh"),
    ("Training GPT-3, once", 1287, "neutral", "1,287 MWh"),
], "MWh", log=False, xlabel="Energy (megawatt-hours)", size=sp.SHORT)

# Training against generating text on one accelerator (schematic roofline)
# H100 ridge point 295 FLOP/byte (Hardware Acceleration, Sec. 11.6).
# Decode at batch 1 with 16-bit weights: about 2 operations per 2 bytes of
# weights, about 1 FLOP/byte. The training point's position is illustrative:
# it only has to sit above the ridge, and the label says so.
# Drawn here rather than with sp.roofline_steps so the two-line training label
# fits inside the compute-bound region without crossing the ridge line.
def roofline_train_decode():
    import numpy as np
    ridge, peak = 295.0, 1.0
    xs = np.logspace(-1.7, 5, 300)
    roof = np.minimum(peak, xs / ridge * peak)
    points = [("Generating\ntext", 1.0, "memory", (-10, 6), "right", "bottom"),
              ("Training,\nlarge batch\n(illustrative)", 5000.0, "compute",
               (0, -12), "center", "top")]
    for k in range(0, len(points) + 1):
        fig, ax = sp.plt.subplots(figsize=sp.SIZE)
        mem, cmp_ = xs <= ridge, xs >= ridge
        ax.plot(xs[mem], roof[mem], color=sp.C["memory"], lw=3)
        ax.plot(xs[cmp_], roof[cmp_], color=sp.C["compute"], lw=3)
        ax.fill_between(xs[mem], 1e-4, roof[mem], color=sp.C["memory_l"], alpha=0.6)
        ax.fill_between(xs[cmp_], 1e-4, roof[cmp_], color=sp.C["compute_l"], alpha=0.6)
        ax.text(ridge / 15, 1.3e-3, "Memory bound", color=sp.C["memory"],
                fontsize=12, fontweight="bold", ha="center")
        ax.text(ridge * 1.6, 1.3e-3, "Compute bound", color=sp.C["compute"],
                fontsize=12, fontweight="bold")
        for label, x, ck, offset, ha, va in points[:k]:
            y = min(peak, x / ridge * peak)
            ax.plot([x], [y], "o", ms=11, color=sp.C[ck], mec="white", mew=1.5, zorder=5)
            ax.annotate(label, (x, y), xytext=offset, textcoords="offset points",
                        ha=ha, va=va, fontsize=12, fontweight="bold",
                        color=sp.C[ck], linespacing=1.1)
        ax.axvline(ridge, color=sp.C["neutral"], lw=1, ls="--")
        ax.text(ridge * 1.1, 0.005, "Ridge: 295 FLOP/byte", color=sp.C["neutral"],
                fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.02, 1e5)
        ax.set_ylim(5e-4, 3)
        ax.minorticks_off()
        ticks = [1, 10, 100, 1000, 10000]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:,g}" for v in ticks])
        ax.set_yticks([])
        ax.set_xlabel("Arithmetic intensity (FLOPs per byte moved, log scale)")
        ax.set_ylabel("Achieved performance")
        ax.text(0.01, 0.97, "Performance axis schematic", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color=sp.C["neutral"])
        sp.save(fig, out / f"roofline_train_decode_step{k}.svg")


roofline_train_decode()

# Utilization is energy: joules per generated token
sp.hbars_steps(out / "energy_utilization", [
    ("Well batched", 1.1, "compute"),
    ("Poorly batched (10% of the rate)", 6.6, "danger", "6.6 J per token: 6× more"),
], "J per token", log=False, value_fmt="{:.1f}", size=sp.SHORT)

# Datacenter electricity (IEA)
sp.hbars_steps(out / "datacenter_electricity", [
    ("2024", 415, "neutral"),
    ("2030 (projected)", 945, "danger"),
], "TWh", log=False, xlabel="Datacenter electricity per year (TWh)",
    size=sp.SHORT)

# Where the energy goes: one access against one operation
sp.hbars_steps(out / "energy_per_op", [
    ("Read one byte from DRAM", 160, "memory"),
    ("One FP16 multiply", 1.1, "compute"),
    ("One INT8 multiply", 0.2, "compute"),
], "pJ", xlabel="Energy (picojoules, log scale; 45 nm, Horowitz 2014)", value_fmt="{:g}")

# One model, four machines: power envelope by deployment tier
sp.hbars_steps(out / "power_tiers", [
    ("Cloud", 3e6, "neutral", "3 MW"),
    ("Edge", 200, "neutral", "200 W"),
    ("Mobile", 5, "neutral", "5 W"),
    ("TinyML", 1e-3, "neutral", "1 mW"),
], "W", xlabel="Power (watts, log scale)", value_fmt="{:g}")

# Silent failure (illustrative, after the Introduction's figure in Sec. 1.6):
# infrastructure health stays flat while recall falls from 95 to 85 percent.
import numpy as np  # noqa: E402
weeks = np.arange(0, 13)
recall = 95 - 10 * weeks / 12
for k, last in ((1, 0), (2, 12)):
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.7))
    ax.plot(weeks, [99.9] * len(weeks), color=sp.C["neutral"], lw=4)
    ax.text(12.2, 99.9, "Health checks", va="center", color=sp.C["neutral"],
            fontsize=12, fontweight="bold")
    if k == 2:
        ax.plot(weeks, recall, color=sp.C["danger"], lw=4)
        ax.fill_between(weeks, recall, 99.9, color=sp.C["danger_l"])
        ax.text(12.2, 85, "Recall  85%", va="center", color=sp.C["danger"],
                fontsize=12, fontweight="bold")
        ax.text(0.3, 91.2, "Recall  95%", ha="left", va="center", color=sp.C["danger"],
                fontsize=12, fontweight="bold")
    ax.set_xlim(0, 12)
    ax.set_ylim(80, 102)
    ax.set_xlabel("Weeks after deployment")
    ax.set_ylabel("Percent")
    sp.save(fig, out / f"silent_failure_step{k}.svg")

# The stack as the map of the course: the kit's final stack (same canvas, so
# nothing moves after the eighth build step) with each lecture placed on its
# layer, following the kit's course_order figure. Lectures 1 to 3 and 16 span
# the whole stack and are named on the slide instead.
kit_vol1 = Path(__file__).resolve().parents[3] / "assets" / "kit" / "vol1"
stack_svg = (kit_vol1 / "stack_build8.svg").read_text()
ACCENT = "#A51C30"
LAYER_Y = {"Hardware": 412, "Frameworks": 348, "Models": 284, "Training": 220,
           "Serving": 156, "Operations": 92, "Applications": 28}
ON_LAYER = {"Hardware": ["L11", "L12"], "Frameworks": ["L7"],
            "Models": ["L5", "L6", "L10"], "Training": ["L8"],
            "Serving": ["L13"], "Operations": ["L14"], "Applications": ["L15"]}


def badge(x, y, w, h, label):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" '
            f'fill="#FFFFFF" stroke="{ACCENT}" stroke-width="1.5"/>\n'
            f'<text x="{x + w / 2}" y="{y + h / 2}" font-family="Helvetica, Arial, '
            f'sans-serif" font-size="24" font-weight="700" fill="{ACCENT}" '
            f'text-anchor="middle" dominant-baseline="central">{label}</text>\n')


badges = ""
for layer, labels in ON_LAYER.items():
    y0 = LAYER_Y[layer]
    right = 688
    for label in reversed(labels):
        badges += badge(right - 66, y0 + 7, 66, 40, label)
        right -= 72
# Data lectures sit inside the data bar, below its label
for label, y in (("L4", 352), ("L9", 404)):
    badges += badge(715, y, 54, 40, label)
(out / "stack_lectures.svg").write_text(stack_svg.replace("</svg>", badges + "</svg>"))

print("figures written")
