#!/usr/bin/env python3
"""Figures for Volume II, Lecture 0 (course overview).

Every value comes from calcs.py, which reads MLSysIM; this file types only
years, layout, and label offsets.

  - GPT-3: training work over one A100's peak, against the published run's
    days (Models.Language.GPT3; Patterson et al. 2021). Introduction Sec. 1.1.
  - Landmark accelerator counts: Literature.TrainingRuns and the model
    registry, as plotted in the Introduction's cluster-size figure (Sec. 1.1,
    figure 2); the fit is Scenarios.LandmarkClusterGrowth2012to2024. GPT-3's
    point is an estimate (Langston 2020).
  - The stack with each lecture on its row (kit stack_build8 canvas).

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import slideplot as sp  # noqa: E402
import calcs  # noqa: E402
from deckvals import num, val  # noqa: E402

out = Path(__file__).parent
FONT = "Helvetica, Arial, sans-serif"



# -----------------------------------------------------------------------------
# 1. GPT-3: one accelerator against the fleet (training time, days, log)
# -----------------------------------------------------------------------------
sp.hbars_steps(out / "gpt3_days", [
    ("One A100", num("one_a100_days"), "neutral", f"{val('one_a100_years')} years"),
    (f"{val('gpt3_gpus')} V100s", num("gpt3_days_1d"), "memory", f"{val('gpt3_days')} days"),
], "days", xlabel="GPT-3 training time (days, log)", size=(3.8, 2.2))


# -----------------------------------------------------------------------------
# 2. Landmark accelerator counts, 2012 to 2024
# -----------------------------------------------------------------------------
def cluster_counts():
    rows = [("AlexNet", 2012, num("alexnet_gpus"), False, (8, -3, "left")),
            ("Megatron-LM", 2019, num("megatron_gpus"), False, (-8, 4, "right")),
            ("GPT-3 (estimate)", 2020, num("msft_gpus"), True, (-10, 0, "right")),
            ("PaLM", 2022, num("palm_chips"), False, (0, 13, "center")),
            ("LLaMA 1", 2023, num("llama1_gpus"), False, (-4, -17, "right")),
            ("DeepSeek-V3", 2024, num("deepseek_gpus"), False, (4, -17, "left")),
            ("Llama 3", 2024, num("llama3_gpus"), False, (9, 4, "left"))]
    fig, ax = sp.plt.subplots(figsize=(5.7, 2.4))
    _, slope, intercept = calcs.GROW.fit   # the filled points only
    xs = np.linspace(2011.5, 2024.5, 100)
    ax.plot(xs, 10 ** (intercept + slope * xs), "--", color=sp.C["neutral"], lw=1.5)
    ax.text(2021.0, 40,
            f"fit: about {val('cluster_growth_round')}× per year", color=sp.C["neutral"],
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


def badge(x, y, w, h, label, size=26):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" '
            f'fill="#FFFFFF" stroke="{ACCENT}" stroke-width="1.5"/>\n'
            f'<text x="{x + w / 2}" y="{y + h / 2}" font-family="{FONT}" '
            f'font-size="{size}" font-weight="700" fill="{ACCENT}" '
            f'text-anchor="middle" dominant-baseline="central">{label}</text>\n')


badges = ""
for row, labels in ON_ROW.items():
    right = 710
    for label in reversed(labels):
        w = 66 if len(label) <= 2 else 80
        badges += badge(right - w, ROW_Y[row] + 4, w, 40, label)
        right -= w + 6
(out / "stack_lectures.svg").write_text(stack_svg.replace("</svg>", badges + "</svg>"))

print("figures written")

