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

out = Path(__file__).parent

# One chat answer against everyday energy, on a linear axis so the ratio
# reads true (built up in three steps)
sp.hbars_steps(out / "energy_answer", [
    ("One answer (estimate)", 548, "compute"),
    ("One prompt (Google, measured)", 864, "compute"),
    ("Charging a phone", 40000, "neutral"),
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
sp.roofline_steps(out / "roofline_train_decode", [
    ("Generating text, one user", 1.0, "memory"),
    ("Training, large batch", 1000.0, "compute"),
])

# Utilization is energy: joules per generated token
sp.hbars_steps(out / "energy_utilization", [
    ("Well batched", 1.1, "compute"),
    ("10% GPU utilization", 6.6, "danger"),
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
    ax.text(0.0, -0.36, "Illustrative", transform=ax.transAxes, fontsize=9,
            color=sp.C["neutral"])
    sp.save(fig, out / f"silent_failure_step{k}.svg")

print("figures written")
