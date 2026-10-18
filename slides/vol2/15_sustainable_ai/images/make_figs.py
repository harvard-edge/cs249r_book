#!/usr/bin/env python3
"""Figures for Volume II, Lecture 15 (Sustainable AI).

Every value is sourced. "SA" is the rendered chapter, Sustainable AI.
Values computed here are labeled "derived" on the figure or its slide.
Registry values come from MLSysIM (grid factors, PUE, H100 TDP and
embodied carbon, the DGX H100 rack profile).

Running case: Llama 3.1 405B, trained on up to 16,384 H100 GPUs at 700 W
TDP (Dubey et al. 2024, "The Llama 3 Herd of Models", Sec. 3.3.1); 30.84M
GPU-hours, location-based 8,930 t CO2eq, market-based 0 t (Llama 3.1 model
card, "Training Greenhouse Gas Emissions" table, meta-llama/llama-models).
Derived with the chapter's method (SA Napkin Math 15.6 and Sec. 15.6.2):
  accelerator power 16,384 x 0.7 kW = 11.5 MW; IT power x 30.8/22.4 (DGX
  H100 rack profile, SA table 7) = 15.8 MW; facility x PUE 1.12 = 17.7 MW;
  accelerator energy 30.84M GPU-h x 0.7 kW = 21,588 MWh; IT 29,684 MWh;
  facility 33,246 MWh; US-average 429 g/kWh -> 14,262 t; Quebec 20 g/kWh
  -> 665 t; Poland 820 g/kWh -> 27,261 t. Duration if every GPU ran
  throughout: 30.84M / 16,384 = 1,882 h = 78 days. Embodied: 164 kg per H100
  (SA Sec. 15.5.5, NVIDIA HGX H100 PCF 1,312 kg / 8); time-allocated over a
  3-year life: 30.84M GPU-h x 164 kg / 26,280 h = 192 t.

Outside data, read from the sources' own tables:
  - CSO Ireland, Data Centres Metered Electricity Consumption 2023, Table 1:
    data-center share of metered electricity 5% (2015) to 21% (2023).
  - Elsworth et al. 2025 (arXiv 2508.15734), Table 1: median Gemini Apps
    text prompt, May 2025. Existing approach: active accelerators 0.10 Wh
    counted; CPU and DRAM 0.04, idle machines 0.02, overhead 0.01 not
    counted. Comprehensive: 0.14 + 0.06 + 0.02 + 0.02 = 0.24 Wh.
  - Patterson et al. 2021 (arXiv 2104.10350), Table 4: parameters, net
    tCO2e, grid factor, PUE for T5, Meena, GShard-600B, Switch, GPT-3.
  - Schneider et al. 2025 (arXiv 2502.01671), Table 1: TPU v6e embodied
    692 kg, operational 2,141 kg (market-based) or 5,759 kg (location-based)
    over 6 years per chip including host share.
  - Luccioni et al. 2023 (BLOOM, arXiv 2211.02001), Table 3: embodied 11.2 t
    of 50.5 t (22.2 percent); Table 5: dynamic emissions of the BigScience
    project by process, final 176B run 24.69 of 66.29 t.
  - Energy-Charts (Fraunhofer ISE), EU public net generation, week 47 of
    2023, hourly (SA figure 19 data, copied as europe_generation_2023w47.csv).
    Fossil share = (gas + coal_oil) / all generation, derived.
  - LBNL (SA Sec. 15.6.1): US data centers 58 TWh (2014), 192 TWh (2024),
    2030 reference 649 TWh in a 521 to 843 TWh range.

Chapter scenarios (labeled on the slides): DGX H100 rack budget (SA table
7); PUE notebook (SA Napkin Math 15.3); replacement break-even (SA Sec. 15.8.6); rebound
(SA Checkpoint 15.5); radio and arithmetic energies (SA Systems Perspective
15.3); DRAM operand against FLOP (SA Sec. 15.4.2).

Run from this folder: python3 make_figs.py
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calcs import (CSO, E_ACC, E_FAC, E_IT, EMB_T, EMBODIED_KG, GRID, HOURS,  # noqa: E402
                   P_ACC, P_FAC, P_IT, PUE, tco2)

C = sp.C
plt = sp.plt
out = Path(__file__).parent


def hide_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def note(ax, text, x=0.0, y=-0.02, fs=10, ha="left", va="top"):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fs,
            color=C["neutral"], ha=ha, va=va)


# ================================================================ hook
# CSO Ireland, Table 1: % of metered electricity used by data centers (calcs.CSO).


def hook_fig(path, size=(5.4, 3.6), fs=15, annotate=False):
    fig, ax = plt.subplots(figsize=size)
    years = [y for y, _ in CSO]
    vals = [v for _, v in CSO]
    cols = [C["memory"] if y < 2022 else C["danger"] for y in years]
    ax.bar(years, vals, color=cols, width=0.72)
    for y, v in CSO:
        if y in (2015, 2019, 2022, 2023):
            ax.text(y, v + 0.5, f"{v}%", ha="center", va="bottom", fontsize=fs,
                    fontweight="bold", color=C["danger"] if y >= 2022 else C["memory"])
    ax.text(2014.6, 22.5, "2022: grid statement", fontsize=fs - 1,
            fontweight="bold", color=C["danger"], va="center",
            alpha=1.0 if annotate else 0.0)
    ax.set_xticks([2015, 2017, 2019, 2021, 2023])
    ax.tick_params(labelsize=fs - 2)
    ax.set_ylim(0, 25)
    ax.set_yticks([0, 10, 20])
    ax.set_yticklabels(["0%", "10%", "20%"])
    ax.set_ylabel("Data centers' share of\nmetered electricity", fontsize=fs - 2)
    sp.save(fig, path)


hook_fig(out / "hook_step1.svg")
hook_fig(out / "hook_step2.svg", annotate=True)
hook_fig(out / "hook_thumb.svg", size=(3.6, 2.0), fs=13)


# ===================================================== power and energy
def power_energy(k, size=(5.6, 3.4), fs=16):
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0.06, 0.16, 0.30, 0.66])
    bx = fig.add_axes([0.50, 0.16, 0.47, 0.66])
    segs = [("GPUs at\n700 W", P_ACC, C["compute"]),
            ("host,\nfabric", P_IT - P_ACC, C["memory"]),
            ("PUE", P_FAC - P_IT, C["neutral"])]
    bottom = 0
    for lab, v, col in segs:
        ax.bar(0, v, bottom=bottom, color=col, width=0.6)
        ax.text(0.36, bottom + v / 2, lab, va="center", ha="left",
                fontsize=fs - 3, color=col, fontweight="bold")
        bottom += v
    ax.text(0, P_FAC + 0.5, f"{P_FAC:.1f} MW", ha="center", va="bottom",
            fontsize=fs, fontweight="bold")
    ax.set_xlim(-0.45, 1.25)
    ax.set_ylim(0, 22)
    ax.set_ylabel("MW", fontsize=fs - 2)
    ax.set_xticks([])
    ax.tick_params(labelsize=fs - 3)
    ax.set_title("Power: every instant", fontsize=fs, loc="left",
                 fontweight="bold", color=C["danger"], x=-0.2)
    a = 1.0 if k >= 2 else 0.0
    t = np.linspace(0, HOURS / 24, 400)
    p = P_FAC * np.ones_like(t)
    bx.fill_between(t, 0, p, color=C["compute_l"], alpha=a, lw=0)
    bx.plot(t, p, color=C["compute"], lw=2.5, alpha=a)
    bx.text(HOURS / 48, P_FAC / 2, f"{E_FAC / 1000:.1f} GWh", ha="center",
            va="center", fontsize=fs + 2, fontweight="bold",
            color=C["compute"], alpha=a)
    bx.text(HOURS / 48, P_FAC / 2 - 4.2, f"{P_FAC:.1f} MW × {HOURS:,.0f} h",
            ha="center", va="center", fontsize=fs - 3, color=C["text"], alpha=a)
    bx.set_xlim(0, 90)
    bx.set_ylim(0, 22)
    bx.set_xlabel("Days", fontsize=fs - 2, alpha=max(a, 0.0))
    bx.tick_params(labelsize=fs - 3, colors=C["text"] if a else "white")
    for s in ("bottom", "left"):
        bx.spines[s].set_alpha(a)
    bx.set_title("Energy: an account over time", fontsize=fs, loc="left",
                 fontweight="bold", color=C["compute"], alpha=a)
    sp.save(fig, out / f"power_energy_step{k}.svg")


for k in (1, 2):
    power_energy(k)


# ========================================================= rack budget
RACK_ROWS = [("GPUs (32 × 700 W)", 22.4, "compute"),
             ("Host CPUs, DRAM", 3.2, "memory"),
             ("NVSwitch", 1.6, "network"),
             ("InfiniBand", 0.8, "network"),
             ("Conversion in IT", 2.8, "neutral")]


def rack_fig(k, size=(5.8, 2.3), fs=14):
    fig, ax = plt.subplots(figsize=size)
    left = 0
    for i, (lab, v, ck) in enumerate(RACK_ROWS):
        show = (i == 0) or k >= 2
        ax.barh(0, v, left=left, color=C[ck], height=0.7,
                alpha=1.0 if show else 0.0, edgecolor="white", lw=1)
        left += v
    ax.text(11.2, 0, "GPUs 22.4 kW", ha="center", va="center", fontsize=fs,
            color="white", fontweight="bold")
    a2 = 1.0 if k >= 2 else 0.0
    ax.text(26.6, 0.5, "host, switches, conversion: 8.4 kW", ha="center",
            va="bottom", fontsize=fs - 1, color=C["memory"], fontweight="bold",
            alpha=a2)
    a3 = 1.0 if k >= 3 else 0.0
    ax.barh(0, 2.7, left=30.8, color=C["neutral_l"], edgecolor=C["neutral"],
            hatch="//", height=0.7, lw=1, alpha=a3)
    ax.text(34.0, 0, "+ PUE 2.7 kW\n= 33.5 kW (1.50 ×)", ha="left", va="center",
            fontsize=fs - 1, fontweight="bold", color=C["text"], alpha=a3)
    ax.annotate("", xy=(0, -0.55), xytext=(30.8, -0.55),
                arrowprops=dict(arrowstyle="|-|", color=C["memory"], lw=1.5, alpha=a2))
    ax.text(15.4, -0.85, "IT load 30.8 kW = 1.38 × the GPUs", ha="center",
            va="top", fontsize=fs, color=C["memory"], fontweight="bold", alpha=a2)
    ax.set_xlim(0, 47)
    ax.set_ylim(-1.55, 1.05)
    hide_axis(ax)
    sp.save(fig, out / f"rack_step{k}.svg")


for k in (1, 2, 3):
    rack_fig(k)


# ========================================================= carbon chain
def chain_fig(k, size=(5.9, 2.4), fs=11.5):
    fig, ax = plt.subplots(figsize=size)
    w, gap = 2.3, 0.45
    boxes = [
        ("GPU", f"{E_ACC / 1000:.1f} GWh", "30.84M GPU-h\n× 700 W", "compute"),
        ("IT", f"{E_IT / 1000:.1f} GWh", "× 1.375 host,\nfabric", "memory"),
        ("Facility", f"{E_FAC / 1000:.1f} GWh", f"× PUE {PUE:.2f}", "neutral"),
        ("CO$_2$", f"{tco2(E_FAC, GRID['US average']):,.0f} t", "× 429 g/kWh\n(US average)", "danger"),
    ]
    for i, (title, val, how, ck) in enumerate(boxes):
        x = i * (w + gap)
        a = 1.0 if k > i else 0.0
        ax.add_patch(FancyBboxPatch((x, 0.75), w, 1.05,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=C[ck + "_l"], edgecolor=C[ck],
                                    lw=1.8, alpha=a))
        ax.text(x + w / 2, 1.55, title, ha="center", va="center", fontsize=fs,
                fontweight="bold", color=C[ck], alpha=a)
        ax.text(x + w / 2, 1.08, val, ha="center", va="center", fontsize=fs + 1.5,
                fontweight="bold", color=C["text"], alpha=a)
        ax.text(x + w / 2, 0.45, how, ha="center", va="center", fontsize=fs - 1,
                color=C["text"], alpha=a)
        if i > 0:
            ax.annotate("", xy=(x - 0.04, 1.27), xytext=(x - gap + 0.04, 1.27),
                        arrowprops=dict(arrowstyle="-|>", color=C["text"], lw=1.8,
                                        alpha=a))
    ax.set_xlim(-0.05, 4 * w + 3 * gap + 0.05)
    ax.set_ylim(0.05, 1.9)
    hide_axis(ax)
    sp.save(fig, out / f"chain_step{k}.svg")


for k in (1, 2, 3, 4):
    chain_fig(k)


# ============================================================== PUE bars
def pue_fig(size=(5.8, 2.5), fs=12):
    fig, ax = plt.subplots(figsize=size)
    rows = [("Legacy air", 1.58), ("Survey average, 2022", 1.55),
            ("State of the art", 1.10), ("Google fleet, 2024", 1.09)]
    for i, (lab, pue) in enumerate(rows):
        y = 3 - i
        ax.barh(y, 2.0, color=C["memory"], height=0.6)
        ax.barh(y, 2.0 * (pue - 1), left=2.0, color=C["neutral_l"],
                edgecolor=C["neutral"], hatch="//", height=0.6, lw=1)
        ax.text(-0.05, y, f"{lab}\nPUE {pue:.2f}", ha="right", va="center",
                fontsize=fs, fontweight="bold")
        ax.text(2.0 * pue + 0.04, y, f"{2.0 * pue:.2f} MW", va="center",
                fontsize=fs, fontweight="bold")
    ax.text(1.0, 3, "IT 2 MW", ha="center", va="center", color="white",
            fontsize=fs, fontweight="bold")
    ax.text(2.6, -0.75, "1.58 to 1.10 saves 0.96 MW, 8,410 MWh a year",
            ha="center", va="center", fontsize=fs, color=C["data"],
            fontweight="bold")
    ax.set_xlim(0, 3.6)
    ax.set_ylim(-1.1, 3.5)
    hide_axis(ax)
    sp.save(fig, out / "pue.svg")


pue_fig()


# =================================================== prompt boundary
def prompt_fig(size=(5.8, 2.3), fs=12):
    fig, ax = plt.subplots(figsize=size)
    parts = [("accelerators", "compute"), ("host CPU,\nDRAM", "memory"),
             ("idle\nmachines", "network"), ("overhead", "neutral")]
    rows = [("Accelerators only", [0.10, 0.04, 0.02, 0.01], [True, False, False, False], 0.10),
            ("Full stack", [0.14, 0.06, 0.02, 0.02], [True, True, True, True], 0.24)]
    for i, (lab, vals, counted, total) in enumerate(rows):
        y = 1 - i
        left = 0
        for (pname, ck), v, c in zip(parts, vals, counted):
            if c:
                ax.barh(y, v, left=left, color=C[ck], height=0.55,
                        edgecolor="white", lw=1)
            else:
                ax.barh(y, v, left=left, color="white", edgecolor=C[ck],
                        hatch="//", height=0.55, lw=1.2)
            left += v
        ax.text(-0.004, y, lab, ha="right", va="center", fontsize=fs,
                fontweight="bold")
        ax.text(left + 0.004, y, f"{total:.2f} Wh" + ("" if i else " counted"),
                va="center", fontsize=fs + 1, fontweight="bold",
                color=C["danger"] if i else C["text"])
    x = 0
    label_x = [0.07, 0.17, 0.255, 0.33]
    for (pname, ck), v, lx in zip(parts, [0.14, 0.06, 0.02, 0.02], label_x):
        ax.annotate(pname, xy=(x + v / 2, -0.29), xytext=(lx, -0.72),
                    ha="center", va="center", fontsize=fs - 1, color=C[ck],
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=C[ck], lw=1))
        x += v
    ax.text(0.0, 1.45, "Hatched parts are excluded from the 0.10 Wh",
            ha="left", fontsize=fs - 1, color=C["neutral"])
    ax.set_xlim(0, 0.37)
    ax.set_ylim(-1.05, 1.65)
    hide_axis(ax)
    sp.save(fig, out / "prompt_boundary.svg")


prompt_fig()


# =========================================== Patterson Table 4 scatter
PAT = [("T5", 11, 46.7, 0.545, "TPU v3"), ("Meena", 2.6, 96.4, 0.415, "TPU v3"),
       ("GShard-600B", 619, 4.3, 0.177, "TPU v3"),
       ("Switch", 1500, 59.1, 0.330, "TPU v3"), ("GPT-3", 175, 552.1, 0.429, "V100")]


def patterson_fig(size=(5.4, 3.4), fs=14):
    fig, ax = plt.subplots(figsize=size)
    for name, params, t, g, hw in PAT:
        ck = "danger" if name == "GPT-3" else ("data" if name == "GShard-600B" else "memory")
        ax.plot(params, t, "o", ms=12, color=C[ck], mec="white", mew=1.5)
        off = {"T5": (8, 6), "Meena": (8, 4), "GShard-600B": (-8, 8),
               "Switch": (-8, 8), "GPT-3": (10, -4)}[name]
        ha = "right" if off[0] < 0 else "left"
        ax.annotate(f"{name}\n{t:g} t", (params, t), xytext=off,
                    textcoords="offset points", ha=ha, fontsize=fs - 2,
                    fontweight="bold", color=C[ck])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 8000)
    ax.set_ylim(2, 2000)
    ax.set_xlabel("Parameters (billions)", fontsize=fs - 1)
    ax.set_ylabel("Net t CO$_2$e, final run", fontsize=fs - 1)
    ax.tick_params(labelsize=fs - 3)
    ax.minorticks_off()
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.set_yticks([10, 100, 1000])
    ax.set_yticklabels(["10", "100", "1,000"])
    ax.grid(color=C["grid"], lw=0.6)
    sp.save(fig, out / "patterson.svg")


patterson_fig()


# ================================================ energy per operand
sp.hbars(out / "energy_ladder.svg", [
    ("DRAM operand (FP32)", 640, "memory", "640 pJ"),
    ("GPU FLOP", 10, "compute", "10 pJ"),
], "pJ", log=False, size=(5.0, 1.25), xlabel=None)


# ===================================================== embodied share
def embodied_fig(k, size=(5.8, 2.9), fs=12):
    fig, ax = plt.subplots(figsize=size)
    us = tco2(E_FAC, GRID["US average"])
    qc = tco2(E_FAC, GRID["Quebec"])
    rows = [("US average grid", us, f"{EMB_T / (us + EMB_T) * 100:.1f}%"),
            ("Quebec grid", qc, f"{EMB_T / (qc + EMB_T) * 100:.0f}%")]
    ax.text(0, 4.62, "Llama 3.1 405B run (derived)", fontsize=fs,
            fontweight="bold", color=C["text"], va="center")
    for i, (lab, op, share) in enumerate(rows):
        y = 4 - i
        ax.barh(y, 100 * op / (op + EMB_T), color=C["compute"], height=0.6)
        ax.barh(y, 100 * EMB_T / (op + EMB_T), left=100 * op / (op + EMB_T),
                color="#8B5A2B", height=0.6)
        ax.text(-1, y, lab, ha="right", va="center", fontsize=fs, fontweight="bold")
        ax.text(101, y, f"embodied {share}", va="center", fontsize=fs,
                fontweight="bold", color="#8B5A2B")
    a = 1.0 if k >= 2 else 0.0
    ax.text(0, 2.12, "Measured lifecycle studies", fontsize=fs, fontweight="bold",
            color=C["text"], va="center", alpha=a)
    meas = [("TPU v6e, 6 years", 692 / (692 + 5759) * 100, 692 / (692 + 2141) * 100),
            ("BLOOM final run", 22.2, None)]
    for i, (lab, lb, mb) in enumerate(meas):
        y = 1.5 - i
        ax.barh(y, 100 - lb, color=C["compute"], height=0.6, alpha=a)
        ax.barh(y, lb, left=100 - lb, color="#8B5A2B", height=0.6, alpha=a)
        ax.text(-1, y, lab, ha="right", va="center", fontsize=fs,
                fontweight="bold", alpha=a)
        txt = (f"embodied {lb:.0f}% (location)\nor {mb:.0f}% (market)" if mb
               else "embodied 22%\n(57 g/kWh grid)")
        ax.text(101, y, txt, va="center", fontsize=fs - 1, fontweight="bold",
                color="#8B5A2B", alpha=a)
    ax.text(45, 4, "operational", ha="center", va="center", fontsize=fs,
            color="white", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 4.9)
    hide_axis(ax)
    sp.save(fig, out / f"embodied_step{k}.svg")


for k in (1, 2):
    embodied_fig(k)


# ===================================================== break-even
def breakeven_fig(k, size=(5.4, 3.5), fs=14):
    fig, ax = plt.subplots(figsize=size)
    saved_kwh_day = 0.35 * 24                 # 50% of 700 W, continuous
    days = np.logspace(0, np.log10(1500), 300)
    ax.axhline(EMBODIED_KG, color="#8B5A2B", lw=2.2, ls="--")
    ax.text(1.15, EMBODIED_KG - 12, "164 kg to make", va="top",
            fontsize=fs - 2, color="#8B5A2B", fontweight="bold")
    ax.axvline(3 * 365, color=C["neutral"], lw=1.2, ls=":")
    ax.text(3 * 365 * 0.93, 285, "3-year\nlife", ha="right", va="top",
            fontsize=fs - 3, color=C["neutral"])
    if k >= 2:
        us = saved_kwh_day * GRID["US average"] / 1000 * days
        ax.plot(days, us, color=C["danger"], lw=3)
        d = EMBODIED_KG / (saved_kwh_day * GRID["US average"] / 1000)
        ax.plot(d, EMBODIED_KG, "o", ms=10, color=C["danger"])
        ax.text(1.15, 215, f"US average:\n{d / 7:.1f} weeks",
                ha="left", fontsize=fs - 1, fontweight="bold", color=C["danger"])
    if k >= 3:
        qc = saved_kwh_day * GRID["Quebec"] / 1000 * days
        ax.plot(days, qc, color=C["data"], lw=3)
        d = EMBODIED_KG / (saved_kwh_day * GRID["Quebec"] / 1000)
        ax.plot(d, EMBODIED_KG, "o", ms=10, color=C["data"])
        ax.text(d * 0.6, 215, f"Quebec:\n{d / 365:.1f} years", ha="right",
                fontsize=fs - 1, fontweight="bold", color=C["data"])
    ax.set_xscale("log")
    ax.set_xlim(1, 1500)
    ax.set_ylim(0, 300)
    ax.set_xticks([1, 7, 30, 365])
    ax.set_xticklabels(["1 day", "1 week", "1 month", "1 year"])
    ax.minorticks_off()
    ax.set_ylabel("kg CO$_2$ avoided", fontsize=fs - 1)
    ax.tick_params(labelsize=fs - 3)
    sp.save(fig, out / f"breakeven_step{k}.svg")


for k in (1, 2, 3):
    breakeven_fig(k)


# ===================================================== BLOOM project
BLOOM = [("176B (final run)", 24.69, "danger"), ("104B", 15.19, "memory"),
         ("1B", 9.06, "memory"), ("13B", 4.97, "memory"),
         ("Other models", 3.66, "memory"), ("Misc. processes", 3.3, "neutral"),
         ("6B", 2.95, "memory"), ("Evaluation", 2.46, "neutral")]
sp.hbars(out / "bloom_dev.svg", [(n, v, ck, f"{v:g} t") for n, v, ck in BLOOM],
         "t", log=False, size=(5.8, 2.7),
         xlabel="t CO$_2$eq, dynamic GPU energy, French grid 57 g/kWh")


# ===================================================== crossover
def crossover_fig(size=(5.4, 3.3), fs=14):
    fig, ax = plt.subplots(figsize=size)
    days = np.linspace(0, 200, 300)
    serve = 1e9 * 0.24 / 1e6 * days            # MWh: prompts/day x Wh
    ax.axhline(E_FAC / 1000, color=C["neutral"], lw=2.5)
    ax.text(3, E_FAC / 1000 + 1.2, f"training once: {E_FAC / 1000:.1f} GWh",
            fontsize=fs - 1, color=C["neutral"], fontweight="bold")
    ax.plot(days, serve / 1000, color=C["compute"], lw=3)
    d = E_FAC / (1e9 * 0.24 / 1e6)
    ax.plot(d, E_FAC / 1000, "o", ms=11, color=C["compute"])
    ax.text(d + 6, E_FAC / 1000 - 6, f"day {d:.0f}", fontsize=fs,
            fontweight="bold", color=C["compute"])
    ax.text(110, 10, "serving: 1 billion\nprompts a day\nat 0.24 Wh", fontsize=fs - 2,
            color=C["compute"], fontweight="bold")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 50)
    ax.set_xlabel("Days of service", fontsize=fs - 1)
    ax.set_ylabel("Facility GWh", fontsize=fs - 1)
    ax.tick_params(labelsize=fs - 3)
    sp.save(fig, out / "crossover.svg")


crossover_fig()


# ===================================================== edge ladder
def edge_fig(size=(5.6, 2.0), fs=14):
    fig, ax = plt.subplots(figsize=size)
    rows = [("Radio, 1 bit", 100_000, 500_000, "network", "100,000 to 500,000 pJ"),
            ("FP32 multiply", 3.7, None, "compute", "3.7 pJ"),
            ("INT32 add", 0.1, None, "data", "0.1 pJ")]
    for i, (lab, lo, hi, ck, txt) in enumerate(rows):
        y = 2 - i
        ax.barh(y, lo, color=C[ck], height=0.6)
        if hi:
            ax.barh(y, hi - lo, left=lo, color=C[ck + "_l"], height=0.6)
        ax.text((hi or lo) * 1.5, y, txt, va="center", fontsize=fs,
                fontweight="bold", color=C[ck])
        ax.text(0.045, y, lab, ha="right", va="center", fontsize=fs, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.05, 3e10)
    ax.set_ylim(-0.6, 2.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([1, 1e3, 1e6])
    ax.set_xticklabels(["1 pJ", "1,000 pJ", "10$^6$ pJ"])
    ax.minorticks_off()
    ax.tick_params(labelsize=fs - 2)
    sp.save(fig, out / "edge_ladder.svg")


edge_fig()


# ===================================================== nine regions
def regions_fig(size=(5.8, 2.9), fs=12):
    fig, ax = plt.subplots(figsize=size)
    items = sorted(GRID.items(), key=lambda kv: kv[1])
    n = len(items)
    for i, (name, g) in enumerate(items):
        t = tco2(E_FAC, g)
        y = n - 1 - i
        ck = {"Norway": "data", "Quebec": "data", "Poland": "danger",
              "US average": "compute"}.get(name, "neutral")
        ax.barh(y, t, color=C[ck] if ck != "neutral" else C["neutral_l"],
                edgecolor=C[ck], height=0.7)
        ax.text(-300, y, f"{name} ({g:g} g/kWh)", ha="right", va="center",
                fontsize=fs, fontweight="bold")
        ax.text(t + 300, y, f"{t:,.0f} t", va="center", fontsize=fs,
                fontweight="bold", color=C[ck])
    ax.set_xlim(0, 33000)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 10000, 20000, 30000])
    ax.set_xticklabels(["0", "10,000", "20,000", "30,000"])
    ax.tick_params(labelsize=fs - 1)
    ax.set_xlabel("t CO$_2$ for the same 33.2 GWh (derived)", fontsize=fs - 1)
    sp.save(fig, out / "regions.svg")


regions_fig()


# ===================================================== EU week
def eu_week(k, size=(5.8, 2.5), fs=12):
    rows = list(csv.DictReader(open(out / "europe_generation_2023w47.csv")))
    gen_cols = ["nuclear", "hydro", "biomass_other", "wind", "solar", "gas", "coal_oil"]
    t = np.array([datetime.fromisoformat(r["time_cet"]) for r in rows])
    hrs = np.array([(x - t[0]).total_seconds() / 3600 for x in t])
    fos = np.array([(float(r["gas"]) + float(r["coal_oil"])) /
                    sum(float(r[c]) for c in gen_cols) * 100 for r in rows])
    fig, ax = plt.subplots(figsize=size)
    ax.plot(hrs, fos, color=C["text"], lw=2)
    roll = np.convolve(fos, np.ones(24) / 24, mode="valid")   # window ends at i+23
    lo, hi = int(np.argmin(roll)), int(np.argmax(roll))
    if k >= 2:
        for i0, ck, lab in ((hi, "danger", f"{roll[hi]:.0f}% fossil"),
                            (lo, "data", f"{roll[lo]:.0f}% fossil")):
            ax.axvspan(hrs[i0], hrs[i0 + 23], color=C[ck + "_l"], zorder=0)
            ax.text(hrs[i0] + 11.5, 46.5, "24-h job\n" + lab, ha="center",
                    va="top", fontsize=fs - 1, fontweight="bold", color=C[ck])
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ax.set_xticks([12 + 24 * i for i in range(7)])
    ax.set_xticklabels(days)
    ax.tick_params(labelsize=fs - 1)
    ax.set_xlim(0, 168)
    ax.set_ylim(15, 47)
    ax.set_yticks([20, 30, 40])
    ax.set_yticklabels(["20%", "30%", "40%"])
    ax.set_ylabel("Fossil share\nof generation", fontsize=fs - 1)
    sp.save(fig, out / f"euweek_step{k}.svg")
    return roll[lo], roll[hi]


for k in (1, 2):
    EU_LO, EU_HI = eu_week(k)


# ===================================================== rebound
def jevons_fig(k, size=(5.4, 3.2), fs=14):
    fig, ax = plt.subplots(figsize=size)
    bars = [("Before", 1.0, "neutral"), ("Energy\nper query", 0.5, "data"),
            ("Queries", 4.0, "memory"), ("Fleet\nenergy", 2.0, "danger")]
    for i, (lab, v, ck) in enumerate(bars):
        a = 1.0 if (i < 3 or k >= 2) else 0.0
        ax.bar(i, v, color=C[ck], width=0.62, alpha=a)
        ax.text(i, v + 0.08, f"{v:g}×", ha="center", va="bottom", fontsize=fs,
                fontweight="bold", color=C[ck], alpha=a)
        ax.text(i, -0.15, lab, ha="center", va="top", fontsize=fs - 2,
                fontweight="bold")
    if k < 2:
        ax.text(3, 0.35, "???", ha="center", fontsize=fs + 4, fontweight="bold",
                color=C["danger"])
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, 4.8)
    hide_axis(ax)
    sp.save(fig, out / f"jevons_step{k}.svg")


for k in (1, 2):
    jevons_fig(k)


# ===================================================== US demand
def lbnl_fig(size=(5.4, 3.3), fs=14):
    fig, ax = plt.subplots(figsize=size)
    ax.bar(0, 58, color=C["neutral"], width=0.6)
    ax.bar(1, 192, color=C["memory"], width=0.6)
    ax.bar(2, 521, color=C["danger_l"], edgecolor=C["danger"], hatch="//",
           width=0.6, lw=1)
    ax.add_patch(Rectangle((1.7, 521), 0.6, 843 - 521, fill=False,
                           edgecolor=C["danger"], ls="--", lw=1.5))
    ax.plot([1.7, 2.3], [649, 649], color=C["danger"], lw=3)
    for x, v, ck in ((0, 58, "neutral"), (1, 192, "memory")):
        ax.text(x, v + 15, f"{v} TWh", ha="center", fontsize=fs, fontweight="bold",
                color=C[ck])
    ax.text(2.36, 649, "649 TWh\nreference", va="center", fontsize=fs - 2,
            fontweight="bold", color=C["danger"])
    ax.text(2.36, 843, "843", va="center", fontsize=fs - 3, color=C["danger"])
    ax.text(2.36, 521, "521", va="center", fontsize=fs - 3, color=C["danger"])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["2014", "2024", "2030\nprojected"])
    ax.tick_params(labelsize=fs - 2)
    ax.set_xlim(-0.5, 3.0)
    ax.set_ylim(0, 920)
    ax.set_ylabel("US data-center TWh", fontsize=fs - 2)
    sp.save(fig, out / "lbnl.svg")


lbnl_fig()

if __name__ == "__main__":
    print(f"EU 24-h fossil share {EU_LO:.1f} to {EU_HI:.1f}%")
