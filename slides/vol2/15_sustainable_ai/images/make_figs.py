#!/usr/bin/env python3
"""Figures for Volume II, Lecture 15 (Sustainable AI).

Every value comes from MLSysIM through calcs.py: figures import calcs (which
records every slide value) and read the formatted strings with val() and the
numbers with num(), or the same MLSysIM objects calcs selects (the running
case's footprint, the Literature series behind the outside tables). Nothing
here is typed except layout: positions, sizes, colors, and axis ranges.

Run from this folder: python3 make_figs.py
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcs  # noqa: E402  (records every value; figures read them)
from deckvals import val, num  # noqa: E402

m, R, RUN = calcs.m, calcs.R, calcs.RUN
P_ACC = m(R.power.accelerator_power, "MW")
P_IT = m(R.power.it_power, "MW")
P_FAC = m(R.power.facility_power, "MW")
E_FAC = m(R.facility_energy, "MWh")
HOURS = m(R.duration, "hour")
EMB_T = m(R.embodied_allocated, "metric_ton")
EMBODIED_KG = num("emb_kg")
CARBON = {name: m(t, "metric_ton") for name, t in R.carbon.items()}
GRID = {name: g.carbon_intensity_g_kwh for name, g in RUN.grids.items()}

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
# CSO Ireland, Table 1: % of metered electricity used by data centers (Literature.Series).
CSO = [(int(r["year"]), r["share_pct"]) for r in calcs.CSO]


def hook_fig(path, size=(5.4, 3.6), fs=15, annotate=False):
    fig, ax = plt.subplots(figsize=size)
    years = [y for y, _ in CSO]
    vals = [v for _, v in CSO]
    cols = [C["memory"] if y < 2022 else C["danger"] for y in years]
    ax.bar(years, vals, color=cols, width=0.72)
    for y, v in CSO:
        if y in (2015, 2019, 2022, 2023):
            ax.text(y, v + 0.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=fs,
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


hook_fig(out / "hook_step1.svg", size=(4.4, 2.2))
hook_fig(out / "hook_step2.svg", size=(4.4, 2.2), annotate=True)
hook_fig(out / "hook_thumb.svg", size=(4.0, 2.1), fs=17)


# ===================================================== power and energy
def power_energy(k, size=(5.6, 3.4), fs=16):
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0.06, 0.16, 0.30, 0.66])
    bx = fig.add_axes([0.50, 0.16, 0.47, 0.66])
    segs = [(f"GPUs at\n{val('tdp_w')} W", P_ACC, C["compute"]),
            ("host,\nfabric", P_IT - P_ACC, C["memory"]),
            ("PUE", P_FAC - P_IT, C["neutral"])]
    bottom = 0
    for lab, v, col in segs:
        ax.bar(0, v, bottom=bottom, color=col, width=0.6)
        ax.text(0.36, bottom + v / 2, lab, va="center", ha="left",
                fontsize=fs - 3, color=col, fontweight="bold")
        bottom += v
    ax.text(0, P_FAC + 0.5, f"{val('p_fac_mw1')} MW", ha="center", va="bottom",
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
    bx.text(HOURS / 48, P_FAC / 2, f"{val('e_fac_gwh')} GWh", ha="center",
            va="center", fontsize=fs + 2, fontweight="bold",
            color=C["compute"], alpha=a)
    bx.text(HOURS / 48, P_FAC / 2 - 4.2, f"{val('p_fac_mw1')} MW × {val('run_hours')} h",
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
RACK_ROWS = [(f"GPUs ({val('rack_gpus')} × {val('tdp_w')} W)", num("rack_gpu_kw"), "compute"),
             ("Host CPUs, DRAM", num("rack_host_kw"), "memory"),
             ("NVSwitch", num("rack_nvs_kw"), "network"),
             ("InfiniBand", num("rack_ib_kw"), "network"),
             ("Conversion in IT", num("rack_conv_kw"), "neutral")]


def rack_fig(k, size=(5.8, 2.3), fs=14):
    fig, ax = plt.subplots(figsize=size)
    left = 0
    for i, (lab, v, ck) in enumerate(RACK_ROWS):
        show = (i == 0) or k >= 2
        ax.barh(0, v, left=left, color=C[ck], height=0.7,
                alpha=1.0 if show else 0.0, edgecolor="white", lw=1)
        left += v
    ax.text(11.2, 0, f"GPUs {val('rack_gpu_kw')} kW", ha="center", va="center", fontsize=fs,
            color="white", fontweight="bold")
    a2 = 1.0 if k >= 2 else 0.0
    ax.text(26.6, 0.5, f"host, switches, conversion: {val('rack_support_kw')} kW", ha="center",
            va="bottom", fontsize=fs - 1, color=C["memory"], fontweight="bold",
            alpha=a2)
    a3 = 1.0 if k >= 3 else 0.0
    ax.barh(0, num("rack_cool_kw"), left=num("rack_it_kw"), color=C["neutral_l"], edgecolor=C["neutral"],
            hatch="//", height=0.7, lw=1, alpha=a3)
    ax.text(34.0, 0, f"+ PUE {val('rack_cool_kw')} kW\n= {val('rack_total_kw')} kW ({val('rack_total_ratio')} ×)", ha="left", va="center",
            fontsize=fs - 1, fontweight="bold", color=C["text"], alpha=a3)
    ax.annotate("", xy=(0, -0.55), xytext=(num("rack_it_kw"), -0.55),
                arrowprops=dict(arrowstyle="|-|", color=C["memory"], lw=1.5, alpha=a2))
    ax.text(15.4, -0.85, f"IT load {val('rack_it_kw')} kW = {val('it_ratio2')} × the GPUs", ha="center",
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
        ("GPU", f"{val('e_acc_gwh')} GWh", f"{val('card_gpuh_m')}M GPU-h\n× {val('tdp_w')} W", "compute"),
        ("IT", f"{val('e_it_gwh')} GWh", f"× {val('it_ratio')} host,\nfabric", "memory"),
        ("Facility", f"{val('e_fac_gwh')} GWh", f"× PUE {val('pue_run')}", "neutral"),
        ("CO$_2$", f"{val('co2_us')} t", f"× {val('g_us')} g/kWh\n(US average)", "danger"),
    ]
    for i, (title, value, how, ck) in enumerate(boxes):
        x = i * (w + gap)
        a = 1.0 if k > i else 0.0
        ax.add_patch(FancyBboxPatch((x, 0.75), w, 1.05,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=C[ck + "_l"], edgecolor=C[ck],
                                    lw=1.8, alpha=a))
        ax.text(x + w / 2, 1.55, title, ha="center", va="center", fontsize=fs,
                fontweight="bold", color=C[ck], alpha=a)
        ax.text(x + w / 2, 1.08, value, ha="center", va="center", fontsize=fs + 1.5,
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
    rows = [("Legacy air", "legacy"), ("Survey average, 2022", "survey"),
            ("State of the art", "soa"), ("Google fleet, 2024", "google")]
    it = num("pue_it_mw")
    for i, (lab, key) in enumerate(rows):
        y = 3 - i
        fac = num(f"pue_fac_{key}")
        ax.barh(y, it, color=C["memory"], height=0.6)
        ax.barh(y, fac - it, left=it, color=C["neutral_l"],
                edgecolor=C["neutral"], hatch="//", height=0.6, lw=1)
        ax.text(-0.05, y, f"{lab}\nPUE {val('pue_' + key)}", ha="right", va="center",
                fontsize=fs, fontweight="bold")
        ax.text(fac + 0.04, y, f"{val('pue_fac_' + key)} MW", va="center",
                fontsize=fs, fontweight="bold")
    ax.text(1.0, 3, f"IT {val('pue_it_mw')} MW", ha="center", va="center", color="white",
            fontsize=fs, fontweight="bold")
    ax.text(2.6, -0.75, f"{val('pue_legacy')} to {val('pue_soa')} saves {val('pue_saved_mw')} MW, "
            f"{val('pue_saved_mwh0')} MWh a year",
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
    keys = ("accel", "host", "idle", "ovh")
    narrow = [num(f"prompt_{k}_x_wh") for k in keys]
    full = [num(f"prompt_{k}_wh") for k in keys]
    rows = [("Accelerators only", narrow, [True, False, False, False], "prompt_narrow_wh"),
            ("Full stack", full, [True, True, True, True], "prompt_full_wh")]
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
        ax.text(left + 0.004, y, f"{val(total)} Wh" + ("" if i else " counted"),
                va="center", fontsize=fs + 1, fontweight="bold",
                color=C["danger"] if i else C["text"])
    x = 0
    label_x = [0.07, 0.17, 0.255, 0.33]
    for (pname, ck), v, lx in zip(parts, full, label_x):
        ax.annotate(pname, xy=(x + v / 2, -0.29), xytext=(lx, -0.72),
                    ha="center", va="center", fontsize=fs - 1, color=C[ck],
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=C[ck], lw=1))
        x += v
    ax.text(0.0, 1.45, f"Hatched parts are excluded from the {val('prompt_narrow_wh')} Wh",
            ha="left", fontsize=fs - 1, color=C["neutral"])
    ax.set_xlim(0, 0.37)
    ax.set_ylim(-1.05, 1.65)
    hide_axis(ax)
    sp.save(fig, out / "prompt_boundary.svg")


prompt_fig()


# =========================================== Patterson Table 4 scatter
SHORT = {"Switch Transformer": "Switch"}
PAT = [(SHORT.get(r["model"], r["model"]), r["params_b"], r["net_tco2e"])
       for r in calcs.SERIES.PattersonTrainingFootprints.entries]


def patterson_fig(size=(4.4, 2.6), fs=16):
    fig, ax = plt.subplots(figsize=size)
    for name, params, t in PAT:
        ck = "danger" if name == "GPT-3" else ("data" if name == "GShard-600B" else "memory")
        ax.plot(params, t, "o", ms=12, color=C[ck], mec="white", mew=1.5)
        off = {"T5": (-8, -30), "Meena": (8, 4), "GShard-600B": (-8, 8),
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
    ("DRAM operand (FP32)", num("operand_pj"), "memory", f"{val('operand_pj')} pJ"),
    ("GPU FLOP", num("flop_pj"), "compute", f"{val('flop_pj')} pJ"),
], "pJ", log=False, size=(5.0, 1.25), xlabel=None)


# ===================================================== embodied share
def embodied_fig(k, size=(5.8, 2.9), fs=12):
    fig, ax = plt.subplots(figsize=size)
    us = CARBON["US average"]
    qc = CARBON["Quebec"]
    rows = [("US average grid", us, f"{val('emb_share_us')}%"),
            ("Quebec grid", qc, f"{val('emb_share_qc')}%")]
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
    meas = [(f"TPU v6e, {val('v6e_years')} years", calcs.V6E_LB_SHARE, calcs.V6E_MB_SHARE),
            ("BLOOM final run", num("bloom_emb_pct"), None)]
    for i, (lab, lb, mb) in enumerate(meas):
        y = 1.5 - i
        ax.barh(y, 100 - lb, color=C["compute"], height=0.6, alpha=a)
        ax.barh(y, lb, left=100 - lb, color="#8B5A2B", height=0.6, alpha=a)
        ax.text(-1, y, lab, ha="right", va="center", fontsize=fs,
                fontweight="bold", alpha=a)
        txt = (f"embodied {val('v6e_lb_pct')}% (location)\nor {val('v6e_mb_pct')}% (market)" if mb
               else f"embodied {lb:.0f}%\n({val('bloom_g')} g/kWh grid)")
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
def breakeven_fig(k, size=(4.4, 2.6), fs=16):
    fig, ax = plt.subplots(figsize=size)
    saved_kwh_day = m(calcs.PAY.result.saved_energy_per_day, "kWh")   # half of the TDP, continuous
    life_days = m(RUN.service_life, "day")
    days = np.logspace(0, np.log10(1500), 300)
    ax.axhline(EMBODIED_KG, color="#8B5A2B", lw=2.2, ls="--")
    ax.text(1.15, EMBODIED_KG - 12, f"{val('emb_kg')} kg to make", va="top",
            fontsize=fs - 2, color="#8B5A2B", fontweight="bold")
    ax.axvline(life_days, color=C["neutral"], lw=1.2, ls=":")
    ax.text(life_days * 0.93, 8, f"{val('life_y')}-year\nlife", ha="right", va="bottom",
            fontsize=fs - 3, color=C["neutral"])
    if k >= 2:
        us = saved_kwh_day * GRID["US average"] / 1000 * days
        ax.plot(days, us, color=C["danger"], lw=3)
        d = EMBODIED_KG / (saved_kwh_day * GRID["US average"] / 1000)
        ax.plot(d, EMBODIED_KG, "o", ms=10, color=C["danger"])
        ax.text(1.15, 215, f"US average:\n{val('pay_us_weeks')} weeks",
                ha="left", fontsize=fs - 1, fontweight="bold", color=C["danger"])
    if k >= 3:
        qc = saved_kwh_day * GRID["Quebec"] / 1000 * days
        ax.plot(days, qc, color=C["data"], lw=3)
        d = EMBODIED_KG / (saved_kwh_day * GRID["Quebec"] / 1000)
        ax.plot(d, EMBODIED_KG, "o", ms=10, color=C["data"])
        ax.text(d * 0.6, 215, f"Quebec:\n{val('pay_qc_years')} years", ha="right",
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
BLOOM_NAMES = {"BLOOM-176B (final run)": ("176B (final run)", "danger"),
               "Miscellaneous processes": ("Misc. processes", "neutral"),
               "Evaluation": ("Evaluation", "neutral")}
BLOOM = [(*BLOOM_NAMES.get(r["process"], (r["process"].replace("BLOOM-", ""), "memory")), r["tco2e"])
         for r in calcs.BLOOM]
BLOOM = [(n, v, ck) for n, ck, v in BLOOM]
sp.hbars(out / "bloom_dev.svg", [(n, v, ck, f"{v:g} t") for n, v, ck in BLOOM],
         "t", log=False, size=(5.8, 2.7),
         xlabel=f"t CO$_2$eq, dynamic GPU energy, French grid {val('bloom_g')} g/kWh")


# ===================================================== crossover
def crossover_fig(size=(5.4, 3.3), fs=14):
    fig, ax = plt.subplots(figsize=size)
    days = np.linspace(0, 200, 300)
    daily = m(calcs.SERVE.result.daily_energy, "MWh")
    serve = daily * days                       # MWh: prompts/day x Wh per prompt
    ax.axhline(E_FAC / 1000, color=C["neutral"], lw=2.5)
    ax.text(3, E_FAC / 1000 + 1.2, f"training once: {val('e_fac_gwh')} GWh",
            fontsize=fs - 1, color=C["neutral"], fontweight="bold")
    ax.plot(days, serve / 1000, color=C["compute"], lw=3)
    d = calcs.SERVE.result.crossover_days
    ax.plot(d, E_FAC / 1000, "o", ms=11, color=C["compute"])
    ax.text(d + 6, E_FAC / 1000 - 6, f"day {val('serve_cross_day')}", fontsize=fs,
            fontweight="bold", color=C["compute"])
    ax.text(110, 10, f"serving: {val('prompts_b')} billion\nprompts a day\nat {val('prompt_full_wh')} Wh", fontsize=fs - 0.5,
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
    rows = [("Radio, 1 bit", num("radio_lo_pj"), num("radio_hi_pj"), "network",
             f"{val('radio_lo_pj')} to {val('radio_hi_pj')} pJ"),
            ("FP32 multiply", num("fp32_pj"), None, "compute", f"{val('fp32_pj')} pJ"),
            ("INT32 add", num("int32_add_pj"), None, "data", f"{val('int32_add_pj')} pJ")]
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
        t = CARBON[name]
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
    ax.set_xlabel(f"t CO$_2$ for the same {val('e_fac_gwh')} GWh (derived)", fontsize=fs - 1)
    sp.save(fig, out / "regions.svg")


regions_fig()


# ===================================================== EU week
def eu_week(k, size=(5.8, 2.5), fs=12):
    eu = calcs.EU
    t = np.array([datetime.fromisoformat(x) for x in eu.times])
    hrs = np.array([(x - t[0]).total_seconds() / 3600 for x in t])
    fos = np.array(eu.shares)
    fig, ax = plt.subplots(figsize=size)
    ax.plot(hrs, fos, color=C["text"], lw=2)
    res = eu.result
    roll = np.array(res.means)                 # window i covers hours i to i + window - 1
    lo, hi = res.low_index, res.high_index
    span = eu.window - 1
    if k >= 2:
        for i0, ck, lab in ((hi, "danger", f"{val('eu_hi_pct')}% fossil"),
                            (lo, "data", f"{val('eu_lo_pct')}% fossil")):
            ax.axvspan(hrs[i0], hrs[i0 + span], color=C[ck + "_l"], zorder=0)
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
    bars = [("Before", 1.0, "neutral"), ("Energy\nper query", num("jev_unit"), "data"),
            ("Queries", num("jev_vol"), "memory"), ("Fleet\nenergy", num("jev_net"), "danger")]
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
    y14, y24 = num("us_2014_twh"), num("us_2024_twh")
    lo, ref, hi = num("us_2030_lo"), num("us_2030_twh"), num("us_2030_hi")
    ax.bar(0, y14, color=C["neutral"], width=0.6)
    ax.bar(1, y24, color=C["memory"], width=0.6)
    ax.bar(2, lo, color=C["danger_l"], edgecolor=C["danger"], hatch="//",
           width=0.6, lw=1)
    ax.add_patch(Rectangle((1.7, lo), 0.6, hi - lo, fill=False,
                           edgecolor=C["danger"], ls="--", lw=1.5))
    ax.plot([1.7, 2.3], [ref, ref], color=C["danger"], lw=3)
    for x, key, ck in ((0, "us_2014_twh", "neutral"), (1, "us_2024_twh", "memory")):
        ax.text(x, num(key) + 15, f"{val(key)} TWh", ha="center", fontsize=fs, fontweight="bold",
                color=C[ck])
    ax.text(2.36, ref, f"{val('us_2030_twh')} TWh\nreference", va="center", fontsize=fs - 2,
            fontweight="bold", color=C["danger"])
    ax.text(2.36, hi, val("us_2030_hi"), va="center", fontsize=fs - 3, color=C["danger"])
    ax.text(2.36, lo, val("us_2030_lo"), va="center", fontsize=fs - 3, color=C["danger"])
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
