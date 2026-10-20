#!/usr/bin/env python3
"""Figures for Volume II, Lecture 0 (course overview).

Every value is sourced. "Derived" marks arithmetic done here from sourced
inputs; the slide note shows the same arithmetic (images/calcs.py).

  - GPT-3: 3.14e23 FLOPs (Brown et al. 2020); one A100 at its 312 TFLOP/s
    peak needs about 32 years; the run used about 10,000 V100s for about
    15 days (Patterson et al. 2021). Introduction Sec. 1.1 and footnote 6.
  - Landmark accelerator counts: MLSysIM Literature.TrainingRuns and the
    model registry, as plotted in the Introduction's cluster-size figure
    (Sec. 1.1, figure 2). GPT-3's point is an estimate (Langston 2020).
  - The stack with each lecture on its row (kit stack_build8 canvas).

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

out = Path(__file__).parent
FONT = "Helvetica, Arial, sans-serif"



# -----------------------------------------------------------------------------
# 1. GPT-3: one accelerator against the fleet (training time, days, log)
# -----------------------------------------------------------------------------
GPT3_FLOPS, A100_PEAK = 3.14e23, 312e12          # Introduction Sec. 1.1, footnote 6
V100_COUNT, V100_DELIVERED = 10_000, 24.6e12     # Patterson et al. 2021, via Sec. 1.1
DAY = 86400
one_a100_days = GPT3_FLOPS / A100_PEAK / DAY                          # derived
fleet_days = GPT3_FLOPS / (V100_COUNT * V100_DELIVERED) / DAY         # derived
sp.hbars_steps(out / "gpt3_days", [
    ("One A100", one_a100_days, "neutral", f"{one_a100_days / 365:.0f} years"),
    ("10,000 V100s", fleet_days, "memory", f"{fleet_days:.0f} days"),
], "days", xlabel="GPT-3 training time (days, log)", size=(3.8, 2.2))


# -----------------------------------------------------------------------------
# 2. Landmark accelerator counts, 2012 to 2024
# -----------------------------------------------------------------------------
def cluster_counts():
    rows = [("AlexNet", 2012, 2, False, (8, -3, "left")),
            ("Megatron-LM", 2019, 512, False, (-8, 4, "right")),
            ("GPT-3 (estimate)", 2020, 10000, True, (-10, 0, "right")),
            ("PaLM", 2022, 6144, False, (-8, -13, "right")),
            ("LLaMA 1", 2023, 2048, False, (-4, -17, "right")),
            ("DeepSeek-V3", 2024, 2048, False, (4, -17, "left")),
            ("Llama 3", 2024, 16384, False, (-9, 6, "right"))]
    fig, ax = sp.plt.subplots(figsize=(5.7, 2.4))
    ver = [r for r in rows if not r[3]]
    coeffs = np.polyfit([r[1] for r in ver], np.log10([r[2] for r in ver]), 1)
    xs = np.linspace(2011.5, 2024.5, 100)
    ax.plot(xs, 10 ** np.polyval(coeffs, xs), "--", color=sp.C["neutral"], lw=1.5)
    ax.text(2021.0, 40,
            f"fit: about {10 ** coeffs[0]:.0f}× per year", color=sp.C["neutral"],
            fontsize=12, fontweight="bold", rotation=0, ha="center")
    for name, yr, n, est, (dx, dy, ha) in rows:
        ax.plot([yr], [n], "o", ms=10, mfc="white" if est else sp.C["memory"],
                mec=sp.C["memory"], mew=2, zorder=5)
        ax.annotate(name, (yr, n), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=12, color=sp.C["memory"],
                    fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(1, 1e5)
    ax.set_xlim(2011.3, 2026.3)
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.set_yticklabels(["1", "10", "100", "1,000", "10,000"])
    ax.set_xticks([2012, 2014, 2016, 2018, 2020, 2022, 2024])
    ax.set_ylabel("Accelerators")
    ax.grid(axis="y", color=sp.C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / "cluster_counts.svg")


cluster_counts()



# -----------------------------------------------------------------------------
# 3. The stack as the map of the course: the kit's final build step (same
#     canvas, so nothing moves) with each lecture on its row, following the
#     kit's roadmap. Lectures 0, 1, and 17 span the stack; the slide says so.
# -----------------------------------------------------------------------------
kit = Path(__file__).resolve().parents[3] / "assets" / "kit" / "vol2"
stack_svg = (kit / "stack_build8.svg").read_text()
ACCENT = "#1F407A"
ROW_Y = {"Facility": 418.25, "Fabric": 362.5, "Execution": 306.75, "Control": 251.0,
         "Serving": 195.25, "Operations": 139.5, "Assurance": 83.75, "Governance": 28.0}
ON_ROW = {"Facility": ["L2"], "Fabric": ["L3", "L4"], "Execution": ["L5", "L6"],
          "Control": ["L7", "L8"], "Serving": ["L9", "L10", "L11"],
          "Operations": ["L12"], "Assurance": ["L13", "L14"],
          "Governance": ["L15", "L16"]}


def badge(x, y, w, h, label, size=22):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" '
            f'fill="#FFFFFF" stroke="{ACCENT}" stroke-width="1.5"/>\n'
            f'<text x="{x + w / 2}" y="{y + h / 2}" font-family="{FONT}" '
            f'font-size="{size}" font-weight="700" fill="{ACCENT}" '
            f'text-anchor="middle" dominant-baseline="central">{label}</text>\n')


badges = ""
for row, labels in ON_ROW.items():
    right = 710
    for label in reversed(labels):
        w = 58 if len(label) <= 2 else 70
        badges += badge(right - w, ROW_Y[row] + 6, w, 36, label)
        right -= w + 6
(out / "stack_lectures.svg").write_text(stack_svg.replace("</svg>", badges + "</svg>"))

print("figures written")

