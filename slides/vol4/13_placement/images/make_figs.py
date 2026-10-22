#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 13 (Workload Placement), Volume IV.

Every value comes from calcs.py, which reads it from MLSysIM (the scenarios
and Literature entries named there): numbers drawn as labels are the strings
the slides print (val), and plotted magnitudes are calcs' plot inputs. Each
figure is drawn at the size of its slot (full width about 5.8 in wide; a
column about 5.4 by 3.6 in) so its type stays legible on the slide. A figure
revealed in steps is written as name_step1.svg ... name_stepN.svg with every
element drawn in every step (hidden ones transparent), so nothing moves.

Colors follow the Volume IV causal-boundary palette (volumes/vol4/palette.md):
navy and blue for cognition and the unprivileged proposer, green for the
deterministic permission side, red only for hazards, limits, and shared
failure paths, slate for the physical board and plant, amber for latency.

Run from this folder: python3 make_figs.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402  (sets the shared rcParams)

import calcs  # noqa: E402
from deckvals import val, num  # noqa: E402

plt = sp.plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).parent

P = {
    "navy": "#1F407A", "navy_l": "#EFF6FF",
    "blue": "#2563EB", "blue_l": "#DBEAFE",
    "green": "#059669", "green_l": "#DCFCE7", "green_d": "#047857",
    "red": "#DC2626", "red_l": "#FEE2E2", "red_d": "#991B1B",
    "slate": "#334155", "slate_l": "#F1F5F9",
    "amber": "#D97706", "amber_l": "#FEF3C7",
    "gray": "#64748B", "gray_l": "#E2E8F0",
    "violet": "#7E317B", "petrol": "#0D9488",
    "text": "#1A1A2E",
}


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", format="svg", facecolor="white",
                bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def a(shown):
    return 1.0 if shown else 0.0


def box(ax, x, y, w, h, text, fc, ec, shown=True, fs=13, tc=None, dashed=False,
        bold=True, rad=0.08, lw=1.8, va="center", sub=None, subfs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rad}",
                                fc=fc, ec=ec, lw=lw, ls="--" if dashed else "-",
                                alpha=a(shown)))
    ty = y + h / 2 + (0.13 if sub else 0)
    ax.text(x + w / 2, ty, text, ha="center", va=va, fontsize=fs,
            color=tc or P["text"], fontweight="bold" if bold else "normal",
            alpha=a(shown))
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center",
                fontsize=subfs, color=P["gray"], alpha=a(shown))


def arrow(ax, x0, y0, x1, y1, color, shown=True, lw=2.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=16, color=color, lw=lw, ls=ls,
                                 alpha=a(shown)))


def canvas(w, h, xlim, ylim):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------------------
# Hook: the thermal hangover on a shared die (Napkin Math 13.2), 3 steps
# ---------------------------------------------------------------------------
def hook():
    t_trip = calcs.T_TO_TRIP
    rec = calcs.RECOVERY
    tq, tl = calcs.T_QUIET, calcs.T_LOADED
    tau = calcs.TAU
    t_end = t_trip + rec + 5.5
    t1 = np.linspace(0, t_trip, 300)
    temp1 = tl - (tl - tq) * np.exp(-t1 / tau)
    t2 = np.linspace(t_trip, t_end, 200)
    temp2 = tq + (calcs.T_TRIP - tq) * np.exp(-(t2 - t_trip) / tau)
    for k in (1, 2, 3):
        fig = plt.figure(figsize=(5.8, 3.0))
        ax = fig.add_axes([0.11, 0.42, 0.86, 0.52])
        bx = fig.add_axes([0.11, 0.10, 0.86, 0.17])
        ax.axvspan(0, t_trip, color=P["blue_l"], lw=0)
        ax.text(t_trip / 2, 57, "proposer inference running", ha="center",
                fontsize=12, color=P["navy"], fontweight="bold")
        ax.plot(t1, temp1, color=P["slate"], lw=2.5)
        ax.plot(t2, temp2, color=P["slate"], lw=2.5, alpha=a(k >= 3))
        ax.axhline(calcs.T_TRIP, color=P["red"], lw=1.5, ls="--")
        ax.text(0.4, calcs.T_TRIP - 4.5, f"trip {val('t_trip_c')} °C", color=P["red"],
                fontsize=11, fontweight="bold")
        ax.axhline(calcs.T_CLEAR, color=P["green"], lw=1.5, ls="--", alpha=a(k >= 3))
        ax.text(0.4, calcs.T_CLEAR - 4.5, f"clear {val('t_clear_c')} °C", color=P["green"],
                fontsize=11, fontweight="bold", alpha=a(k >= 3))
        ax.plot([t_trip], [calcs.T_TRIP], "o", ms=8, color=P["red"], alpha=a(k >= 2))
        ax.annotate(f"{val('t_trip_s')} s: clock halved", (t_trip, calcs.T_TRIP),
                    xytext=(-12, 5), textcoords="offset points", ha="right",
                    fontsize=12, color=P["red_d"], fontweight="bold", alpha=a(k >= 2))
        ax.axvspan(t_trip, t_trip + rec, color=P["red_l"], lw=0, alpha=a(k >= 3) * 0.9)
        ax.text(t_trip + rec + 0.3, 58, f"{val('recover_s')} s\nafter stop",
                fontsize=12, color=P["red_d"], fontweight="bold", alpha=a(k >= 3))
        ax.text(t_trip + 0.4, 66, "?", fontsize=20, color=P["red_d"], fontweight="bold",
                alpha=a(k == 2))
        ax.set_xlim(0, t_end)
        ax.set_ylim(50, 104)
        ax.set_yticks([53, 75, 95])
        ax.set_ylabel("Junction (°C)")
        ax.set_xticks([0, 10, 20, 30])
        ax.tick_params(labelbottom=False)
        # tick strip: each 1 ms tick's response against the 400 us deadline
        bx.set_xlim(0, t_end)
        bx.set_ylim(0, 1)
        bx.add_patch(Rectangle((0, 0.15), t_trip, 0.7, color=P["green_l"], ec=P["green"]))
        bx.text(t_trip / 2, 0.5, f"every tick on time: {val('resp_nom')} µs of {val('deadline')}",
                ha="center", va="center", fontsize=11, color=P["green_d"], fontweight="bold")
        bx.add_patch(Rectangle((t_trip, 0.15), rec, 0.7, color=P["red_l"], ec=P["red"],
                               alpha=a(k >= 2)))
        bx.text(t_trip + rec / 2, 1.12, f"{val('resp_thr')} µs", ha="center", va="bottom",
                fontsize=11, color=P["red_d"], fontweight="bold", alpha=a(k >= 2))
        bx.add_patch(Rectangle((t_trip + rec, 0.15), t_end - t_trip - rec, 0.7,
                               color=P["green_l"], ec=P["green"], alpha=a(k >= 3)))
        bx.set_yticks([])
        bx.set_xticks([0, 10, 20, 30])
        bx.set_xlabel("Time (s)")
        for s in ("left", "top", "right"):
            bx.spines[s].set_visible(False)
        save(fig, f"hook_step{k}")


# ---------------------------------------------------------------------------
# The tick budget: 135, 250, 310, 400, 1,000 us (ch 13.2), 4 steps
# ---------------------------------------------------------------------------
def budget():
    for k in (1, 2, 3, 4):
        fig, ax = plt.subplots(figsize=(5.8, 2.2))
        y = 0.5
        ax.barh(y, calcs.SENSE, left=0, height=0.42, color=P["petrol"])
        ax.barh(y, calcs.CHECK, left=calcs.SENSE, height=0.42, color=P["green"])
        ax.barh(y, calcs.STAGE, left=calcs.SENSE + calcs.CHECK, height=0.42,
                color=P["green_d"])
        ax.text(0, 1.02, f"unloaded tick {val('unloaded_us')}", fontsize=12,
                color=P["green_d"], fontweight="bold")
        # declared WCET
        ax.plot([calcs.WCET] * 2, [0.15, 0.95], color=P["slate"], lw=2, ls="--",
                alpha=a(k >= 2))
        ax.text(calcs.WCET, 0.02, f"WCET {val('wcet')}", ha="center", va="top",
                fontsize=12, color=P["slate"], fontweight="bold", alpha=a(k >= 2))
        # blocking + interference band and deadline
        ax.add_patch(Rectangle((calcs.CEILING, 0.2), calcs.DEADLINE - calcs.CEILING,
                               0.6, fc=P["gray_l"], hatch="//", ec=P["gray"], lw=1,
                               alpha=a(k >= 3)))
        ax.text(calcs.DEADLINE + 12, 0.5, f"{val('b_plus_i')} µs: the MCU's own tasks", fontsize=12,
                va="center", color=P["gray"], alpha=a(k >= 3))
        ax.plot([calcs.DEADLINE] * 2, [0.15, 0.95], color=P["red"], lw=2.2,
                alpha=a(k >= 3))
        ax.text(calcs.DEADLINE + 8, 0.02, f"deadline {val('deadline')}", va="top",
                fontsize=12, color=P["red_d"], fontweight="bold", alpha=a(k >= 3))
        # the 60 us allowance
        ax.add_patch(Rectangle((calcs.WCET, 0.2), calcs.CEILING - calcs.WCET, 0.6,
                               fc=P["amber_l"], ec=P["amber"], lw=1.5, alpha=a(k >= 4)))
        ax.text((calcs.WCET + calcs.CEILING) / 2, y, f"{val('allowance_us')}",
                ha="center", va="center", fontsize=13, color=P["amber"], fontweight="bold",
                alpha=a(k >= 4))
        ax.plot([calcs.PERIOD] * 2, [0.15, 0.95], color=P["slate"], lw=1.5,
                alpha=a(k >= 4))
        ax.text(calcs.PERIOD, 0.02, "next tick", ha="right", va="top", fontsize=12,
                color=P["slate"], alpha=a(k >= 4))
        ax.set_xlim(0, 1010)
        ax.set_ylim(-0.3, 1.3)
        ax.set_yticks([])
        ax.set_xticks([0, 200, 400, 600, 800, 1000])
        ax.set_xticklabels(["0", "200", "400", "600", "800", "1,000"])
        ax.set_xlabel("Time within one 1 kHz tick (µs)")
        ax.spines["left"].set_visible(False)
        save(fig, f"budget_step{k}")


# ---------------------------------------------------------------------------
# The compute board and four candidate placements (ch 13.2.2 figure 1), 4 steps
# ---------------------------------------------------------------------------
def badge(ax, x, y, s, shown, color):
    ax.add_patch(plt.Circle((x, y), 0.22, fc="white", ec=color, lw=2, alpha=a(shown)))
    ax.text(x, y, s, ha="center", va="center", fontsize=13, fontweight="bold",
            color=color, alpha=a(shown))


def board():
    for k in (1, 2, 3, 4):
        fig, ax = canvas(5.9, 3.4, (0, 11.8), (0, 6.8))
        # SoC outline
        box(ax, 0.1, 1.15, 7.3, 5.55, "", "white", P["slate"], rad=0.12, lw=2)
        ax.text(0.3, 6.38, "Application SoC: one die, one package", fontsize=13,
                fontweight="bold", color=P["slate"])
        box(ax, 0.3, 4.75, 1.65, 1.15, "app\ncores", P["navy_l"], P["navy"], fs=12)
        box(ax, 2.1, 4.75, 1.65, 1.15, "neural\naccel.", P["navy_l"], P["navy"], fs=12)
        badge(ax, 1.8, 5.72, "a", True, P["navy"])
        box(ax, 3.9, 4.75, 1.45, 1.15, "RT\ncluster", P["green_l"], P["green"], fs=12,
            shown=k >= 2)
        badge(ax, 5.2, 5.72, "b", k >= 2, P["green"])
        # safety island
        box(ax, 5.55, 3.1, 1.7, 2.8, "", P["green_l"], P["green"], dashed=True, shown=k >= 3)
        ax.text(6.4, 5.45, "safety\nisland", ha="center", va="center", fontsize=12,
                fontweight="bold", color=P["green_d"], alpha=a(k >= 3))
        ax.text(6.4, 4.35, "own crystal\nown rail\nSRAM", ha="center", va="center",
                fontsize=11, color=P["green_d"], alpha=a(k >= 3), linespacing=1.3)
        ax.text(6.2, 3.35, "firewall", ha="center", va="center", fontsize=11,
                color=P["slate"], alpha=a(k >= 3), style="italic")
        badge(ax, 7.0, 3.1, "c", k >= 3, P["green"])
        # shared blocks
        shared = [(0.3, 3.85, 2.55, "last-level cache"), (2.95, 3.85, 2.4, "DRAM ctrl"),
                  (0.3, 3.0, 2.55, "clock governor"), (2.95, 3.0, 2.4, "core rail"),
                  (0.3, 2.15, 5.05, "reset and boot chain"),
                  (0.3, 1.3, 7.0, "fabric, die temperature, package heat path")]
        for x, y0, w, lab in shared:
            box(ax, x, y0, w, 0.7, lab, P["red_l"], P["red"], fs=11, bold=False, lw=1.3)
        # DRAM below
        box(ax, 2.5, 0.35, 2.4, 0.6, "LPDDR5 DRAM", P["red_l"], P["red"], fs=11, bold=False,
            lw=1.3)
        # discrete MCU
        box(ax, 8.15, 2.35, 3.75, 4.35, "", P["green_l"], P["green"], rad=0.12, lw=2,
            shown=k >= 4)
        ax.text(8.3, 6.33, "Safety MCU", fontsize=13, fontweight="bold",
                color=P["green_d"], alpha=a(k >= 4))
        badge(ax, 11.55, 6.38, "d", k >= 4, P["green"])
        for i, lab in enumerate(["lockstep cores", "private memory", "own clock",
                                 "reset supervisor"]):
            box(ax, 8.35, 5.3 - i * 0.72, 3.35, 0.58, lab, "white", P["green"], fs=11,
                bold=False, shown=k >= 4, lw=1.2)
        arrow(ax, 7.4, 1.65, 8.15, 2.55, P["slate"], shown=k >= 4)
        ax.text(7.5, 2.0, "board link", fontsize=11, color=P["slate"], ha="right",
                fontweight="bold", alpha=0)
        box(ax, 8.6, 1.3, 2.7, 0.7, f"isolated rail, {val('rail_holdup_s')} s", P["green_l"], P["green"], fs=11,
            bold=False, shown=k >= 4, lw=1.3)
        # battery bar
        box(ax, 0.1, -0.0 + 0.02, 11.6, 0.0, "", "white", "white", shown=False)
        ax.plot([5.3, 11.0], [0.65, 0.65], color=P["slate"], lw=0)
        box(ax, 5.3, 0.35, 6.4, 0.6, "battery and board supply: shared by all",
            P["slate_l"], P["slate"], fs=11, bold=False, lw=1.3)
        save(fig, f"board_step{k}")


# ---------------------------------------------------------------------------
# Shared-SoC memory path (ch 13.3.1, Napkin Math 13.1): one LPDDR5 controller
# ---------------------------------------------------------------------------
def socmem():
    fig, ax = canvas(5.8, 2.2, (0, 11.6), (0, 4.4))
    box(ax, 0.1, 0.1, 8.25, 4.25, "", "white", P["slate"], rad=0.12, lw=2)
    ax.text(0.3, 4.02, "Application SoC, unified memory", fontsize=13,
            fontweight="bold", color=P["slate"], va="center")
    box(ax, 0.3, 2.95, 2.2, 0.75, "CPU cluster", P["navy_l"], P["navy"], fs=12)
    box(ax, 2.65, 2.95, 2.8, 0.75, "NPU (proposer)", P["navy_l"], P["navy"], fs=12)
    box(ax, 5.6, 2.95, 2.55, 0.75, "RT cores (tick)", P["green_l"], P["green"], fs=12)
    for x in (1.4, 4.05, 6.88):
        arrow(ax, x, 2.95, x, 2.72, P["slate"], lw=1.4)
    box(ax, 0.3, 2.2, 7.85, 0.5, "fabric, arbiter, shared last-level cache",
        P["red_l"], P["red"], fs=12, bold=False, lw=1.3)
    box(ax, 0.3, 0.3, 7.85, 1.7, "", P["red_l"], P["red"], lw=1.3)
    ax.text(0.5, 1.68, "One LPDDR5 controller, first come first served", fontsize=12,
            color=P["red_d"], fontweight="bold", va="center")
    box(ax, 0.5, 0.5, 0.95, 0.85, f"{val('snapshot_kib')} KB\nread", P["green_l"], P["green"], fs=11)
    box(ax, 1.55, 0.5, 1.0, 0.85, "telem.", P["slate_l"], P["slate"], fs=11, bold=False)
    box(ax, 2.65, 0.5, 1.15, 0.85, "camera", P["slate_l"], P["slate"], fs=11, bold=False)
    box(ax, 3.9, 0.5, 4.05, 0.85, f"weight tile, {val('tile_tx')} transactions", P["navy_l"], P["navy"],
        fs=11)
    arrow(ax, 8.15, 0.92, 8.85, 0.92, P["slate"], lw=2.0)
    box(ax, 8.85, 0.3, 2.65, 1.7, "LPDDR5 DRAM", P["slate_l"], P["slate"], fs=12,
        sub=f"{val('bw_gbs')} GB/s peak", subfs=11)
    ax.text(10.18, 3.15, f"{val('backlog_tx')} transactions,\n{val('backlog_bytes')} B\nserved before\nthe read",
            ha="center", va="center", fontsize=12, color=P["red_d"], fontweight="bold",
            linespacing=1.25)
    save(fig, "socmem")


# ---------------------------------------------------------------------------
# Backlog budget on one 1 ms period (Napkin Math 13.1, ch figure 2), 5 steps
# ---------------------------------------------------------------------------
def backlog():
    labels = ["unloaded", f"{val('n_sense')} backlogs", f"{val('n_wcet')} backlogs",
              f"{val('n_dead')} backlogs", f"{val('n1_wcet')}, one channel"]
    rows = [(lab, n, w, r) for lab, (n, w, r) in zip(labels, calcs.BACKLOG_ROWS)]
    for k in range(1, 6):
        fig, ax = plt.subplots(figsize=(5.8, 2.7))
        for i, (lab, n, w, r) in enumerate(rows):
            yy = len(rows) - 1 - i
            show = i < k
            mem = w + r
            if n == 0:
                ax.barh(yy, mem, height=0.6, color=P["petrol"], alpha=a(show))
            else:
                ax.barh(yy, mem, height=0.6, color=P["amber_l"], ec=P["amber"], hatch="///",
                        alpha=a(show), lw=1)
            ax.barh(yy, calcs.COMPUTE_AFTER, left=mem, height=0.6, color=P["green"],
                    alpha=a(show))
            ax.text(-12, yy, lab, ha="right", va="center", fontsize=12,
                    fontweight="bold", color=P["text"], alpha=a(show))
            ax.text(mem + calcs.COMPUTE_AFTER + 8, yy, f"{mem + calcs.COMPUTE_AFTER:.0f}",
                    va="center", fontsize=11, color=P["text"], alpha=a(show))
        for x, lab, c, ha, yo in ((calcs.SENSE, val("sense_us"), P["petrol"], "center", 0.0),
                                  (calcs.WCET, f"WCET {val('wcet')}", P["slate"], "center", 0.0),
                                  (calcs.DEADLINE, f"deadline {val('deadline')}", P["red"], "left", 0.0),
                                  (calcs.PERIOD, "next tick", P["slate"], "right", 0.0)):
            ax.axvline(x, color=c, lw=1.6, ls="--")
            ax.text(x + (6 if ha == "left" else -6 if ha == "right" else 0),
                    len(rows) - 0.35 + yo, lab, ha=ha, fontsize=11, color=c,
                    fontweight="bold")
        ax.axvspan(calcs.CEILING, calcs.DEADLINE, color=P["gray_l"], lw=0)
        ax.set_xlim(0, 1010)
        ax.set_ylim(-0.6, len(rows) + 0.05)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Time within one tick (µs); hatched: memory wait")
        fig.subplots_adjust(left=0.24, right=0.98, bottom=0.2, top=0.95)
        save(fig, f"backlog_step{k}")


# ---------------------------------------------------------------------------
# Published shared-cache interference (Valsan 2016; Bechtel and Yun 2019)
# ---------------------------------------------------------------------------
def measured():
    kinds = ["network", "network", "network", "danger", "danger"]
    rows = [(lab, x, kind, f"{val(key)}×") for (lab, x, key), kind in zip(calcs.MEASURED, kinds)]
    sp.hbars(str(OUT / "measured.svg"), rows, "", log=True,
             xlabel="Victim slowdown on its own core and cache partition", size=(5.8, 2.6))


# ---------------------------------------------------------------------------
# Tolerable miss-stall share against slowdown (eq 2), 2 steps
# ---------------------------------------------------------------------------
def stallshare():
    s = np.logspace(np.log10(1.5), np.log10(1000), 300)
    u = calcs.UNLOADED
    wc = (calcs.WCET / u - 1) / (s - 1) * 100
    dl = (calcs.CEILING / u - 1) / (s - 1) * 100
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.4, 3.3))
        ax.fill_between(s, wc, 100, color=P["red_l"], lw=0)
        ax.plot(s, wc, color=P["red"], lw=2.5)
        ax.plot(s, dl, color=P["gray"], lw=1.8, ls="--")
        ax.text(40, 25, "declared WCET\nrefuted", color=P["red_d"], fontsize=12,
                fontweight="bold")
        ax.text(60, 4.5, "deadline", color=P["gray"], fontsize=11)
        for sl, yv, lab in ((calcs.S_MSHR, calcs.PHI_MSHR, f"{val('s_mshr')}×: {val('phi21')}%"),
                            (calcs.S_WB, calcs.PHI_WB, f"{val('s_wb')}×: {val('phi300')}%")):
            ax.plot([sl], [yv], "o", ms=9, color=P["red"], alpha=a(k >= 2))
            ax.annotate(lab, (sl, yv), xytext=(-10, -22), textcoords="offset points",
                        ha="right", fontsize=12, color=P["red_d"], fontweight="bold",
                        alpha=a(k >= 2))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1.5, 1000)
        ax.set_ylim(0.1, 100)
        ax.set_xticks([2, 10, 100, 1000])
        ax.set_xticklabels(["2", "10", "100", "1,000"])
        ax.set_yticks([0.1, 1, 10, 100])
        ax.set_yticklabels(["0.1", "1", "10", "100"])
        ax.minorticks_off()
        ax.set_xlabel("Interference slowdown of stalls (×)")
        ax.set_ylabel("Tolerable stall share (%)")
        fig.subplots_adjust(left=0.17, right=0.97, bottom=0.18, top=0.96)
        save(fig, f"stallshare_step{k}")


# ---------------------------------------------------------------------------
# Thermal hangover worked example (Napkin Math 13.2, ch figure 3), 2 steps
# ---------------------------------------------------------------------------
def hangover():
    rec = calcs.RECOVERY
    tq = calcs.T_QUIET
    tau = calcs.TAU
    t = np.linspace(0, 6, 300)
    temp = tq + (calcs.T_TRIP - tq) * np.exp(-t / tau)
    for k in (1, 2):
        fig = plt.figure(figsize=(5.8, 3.2))
        ax = fig.add_axes([0.17, 0.56, 0.80, 0.40])
        bx = fig.add_axes([0.17, 0.12, 0.80, 0.30])
        ax.axvspan(0, rec, color=P["red_l"], lw=0)
        ax.text(rec / 2, 59, f"{val('f_thr_mhz')} MHz", ha="center", fontsize=12, color=P["red_d"],
                fontweight="bold")
        ax.plot(t, temp, color=P["slate"], lw=2.5)
        for yv, c, lab in ((calcs.T_TRIP, P["red"], val("t_trip_c")), (calcs.T_CLEAR, P["green"], val("t_clear_c")),
                           (tq, P["gray"], val("t_quiet"))):
            ax.axhline(yv, color=c, lw=1.3, ls="--")
        ax.text(rec + 0.1, 88, f"{val('recover_s')} s = {val('recover_ticks')} ticks",
                fontsize=12, color=P["red_d"], fontweight="bold")
        ax.set_xlim(0, 6)
        ax.set_ylim(50, 98)
        ax.set_yticks([tq, calcs.T_CLEAR, calcs.T_TRIP])
        ax.set_ylabel("°C")
        ax.text(5.95, 62, "seconds after\ninference stops", ha="right", fontsize=11,
                color=P["gray"])
        # tick bars
        bx.set_xlim(0, 1000)
        bx.set_ylim(-0.6, 1.6)
        for yy, ex, lab, c, cl, show in ((1, calcs.EXEC_NOM, f"{val('f_nom_ghz')} GHz", P["green"], P["green_l"], True),
                                       (0, calcs.EXEC_THR, f"{val('f_thr_mhz')} MHz", P["red"], P["red_l"], k >= 2)):
            bx.barh(yy, ex, height=0.6, color=cl, ec=c, lw=1.5, alpha=a(show))
            bx.barh(yy, calcs.B_PLUS_I, left=ex, height=0.6, color=P["gray_l"],
                    ec=P["gray"], hatch="//", lw=1, alpha=a(show))
            bx.text(-15, yy, lab, ha="right", va="center", fontsize=12, fontweight="bold",
                    color=c, alpha=a(show))
            bx.text(ex / 2, yy, f"{ex:.0f} µs", ha="center", va="center", fontsize=11,
                    color=P["text"], fontweight="bold", alpha=a(show))
            bx.text(ex + calcs.B_PLUS_I + 10, yy, f"{ex + calcs.B_PLUS_I:.0f}" if yy == 0 else "",
                    va="center", fontsize=11, color=c, fontweight="bold", alpha=a(show))
        bx.axvline(calcs.DEADLINE, color=P["red"], lw=2)
        bx.text(calcs.DEADLINE + 8, 1.35, f"deadline {val('deadline')}", ha="left", fontsize=11,
                color=P["red_d"], fontweight="bold")
        bx.set_yticks([])
        bx.set_xticks([0, calcs.WCET, calcs.DEADLINE, num("resp_thr"), calcs.PERIOD])
        bx.set_xticklabels(["0", val("wcet"), val("deadline"), val("resp_thr"), f"{calcs.PERIOD:,.0f}"])
        bx.set_xlabel("µs within one tick")
        for s in ("left", "top", "right"):
            bx.spines[s].set_visible(False)
        save(fig, f"hangover_step{k}")


# ---------------------------------------------------------------------------
# Coupling-timescale ladder (ch 13.4)
# ---------------------------------------------------------------------------
def ladder():
    colors = [P["red"], P["slate"], P["amber"], P["amber"], P["violet"]]
    texts = [f"{val('recover_s')} s", f"{val('t_tick_ms')} ms", f"{val('ladder_backlog_us')} µs",
             f"{val('conflict_ns')} ns", f"{val('rise_ns')} ns"]
    rows = [(lab, x, c, t) for (lab, x), c, t in zip(calcs.LADDER, colors, texts)]
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    lo = 3e-10
    for i, (lab, v, c, txt) in enumerate(rows):
        yy = len(rows) - 1 - i
        ax.barh(yy, v - lo, left=lo, height=0.62, color=c)
        ax.text(v * 1.6, yy, txt, va="center", fontsize=13, color=c, fontweight="bold")
        ax.text(lo / 1.5, yy, lab, ha="right", va="center", fontsize=13,
                fontweight="bold", color=P["text"])
    ax.set_xscale("log")
    ax.set_xlim(lo, 300)
    ax.set_xticks([1e-9, 1e-6, 1e-3, 1])
    ax.set_xticklabels(["1 ns", "1 µs", "1 ms", "1 s"])
    ax.minorticks_off()
    ax.tick_params(axis="x", labelsize=12)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=P["gray_l"], lw=0.8)
    ax.set_axisbelow(True)
    save(fig, "ladder")


# ---------------------------------------------------------------------------
# Regulation result (Bechtel and Yun 2019, Sec. VI), activity figure
# ---------------------------------------------------------------------------
def regulation():
    kinds = ["danger", "data", "data"]
    rows = [(lab, x, kind, f"{val(key)}×") for (lab, x, key), kind in zip(calcs.REGULATION, kinds)]
    for k in (1, 2):
        sp.hbars(str(OUT / f"regulation_step{k}.svg"), rows, "", log=True, step=1 if k == 1 else None,
                 xlabel="Victim slowdown, Raspberry Pi 3, three write attackers", size=(5.8, 1.9))


# ---------------------------------------------------------------------------
# Board-link toll (Napkin Math 13.3, ch figure), 2 steps
# ---------------------------------------------------------------------------
def boardlink():
    segs = [(f"inference {val('chunk_infer_ms')}", calcs.INFER, P["navy"], "white"),
            ("", calcs.LINK, P["violet"], "white"),
            (val("valid_ms"), calcs.VALID, P["slate"], "white"),
            ("", calcs.SMALL, P["gray"], "white")]
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.1))
        x = 0
        for lab, w, c, tc in segs:
            ax.barh(0, w, left=x, height=0.5, color=c)
            if lab:
                ax.text(x + w / 2, 0, lab, ha="center", va="center", color=tc,
                        fontsize=12, fontweight="bold")
            x += w
        ax.annotate(f"board link {val('link_ms')}", (calcs.INFER + calcs.LINK / 2, 0.25),
                    xytext=(calcs.INFER - 6, 0.62), fontsize=12, color=P["violet"],
                    fontweight="bold", ha="center",
                    arrowprops=dict(arrowstyle="-", color=P["violet"], lw=1.5))
        ax.text(x, 0.38, f"{val('d2a_ms')} ms", ha="center", fontsize=12,
                fontweight="bold", color=P["text"])
        ax.text(calcs.INFER + calcs.LINK + calcs.VALID / 2, -0.42, "validation",
                ha="center", va="center", fontsize=11, color=P["slate"])
        ax.add_patch(Rectangle((x, -0.25), calcs.CEIL_MS - x, 0.5, fill=False, ls=":",
                               ec=P["green"], lw=2, alpha=a(k >= 2)))
        ax.text((x + calcs.CEIL_MS) / 2, 0, f"reserve\n{val('reserve_ms')} ms",
                ha="center", va="center", fontsize=11, color=P["green_d"],
                fontweight="bold", alpha=a(k >= 2))
        ax.axvline(calcs.CEIL_MS, color=P["red"], lw=2.2, alpha=a(k >= 2))
        ax.text(calcs.CEIL_MS, 0.62, f"ceiling {val('ceil_ms')}", ha="center", fontsize=12,
                color=P["red_d"], fontweight="bold", alpha=a(k >= 2))
        ax.set_xlim(0, 112)
        ax.set_ylim(-0.7, 0.9)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Milliseconds from perception dispatch to admission")
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.27, top=0.97)
        save(fig, f"boardlink_step{k}")


# ---------------------------------------------------------------------------
# Windowed watchdog (ch 13.5.2, eq 5), 3 steps
# ---------------------------------------------------------------------------
def watchdog():
    for k in (1, 2, 3):
        fig, ax = canvas(5.8, 2.3, (-0.3, 3.45), (-0.75, 1.2))
        ax.plot([0, 3.0], [0, 0], color=P["slate"], lw=2)
        ax.text(3.05, 0, "time", va="center", fontsize=11, color=P["slate"])
        ax.plot([0, 0], [-0.12, 0.12], color=P["slate"], lw=2)
        ax.text(0, -0.35, "last feed", ha="center", fontsize=12, color=P["slate"])
        # window regions
        ax.add_patch(Rectangle((0.05, 0.15), 0.75, 0.6, color=P["red_l"], ec=P["red"],
                               alpha=a(k >= 2)))
        ax.text(0.43, 0.45, "too early:\ntrip", ha="center", va="center", fontsize=12,
                color=P["red_d"], fontweight="bold", alpha=a(k >= 2))
        ax.add_patch(Rectangle((0.8, 0.15), 0.8, 0.6, color=P["green_l"], ec=P["green"]))
        ax.text(1.2, 0.45, "valid\nwindow", ha="center", va="center", fontsize=12,
                color=P["green_d"], fontweight="bold")
        ax.add_patch(Rectangle((1.6, 0.15), 1.35, 0.6, color=P["red_l"], ec=P["red"],
                               alpha=a(k >= 1)))
        ax.text(2.27, 0.45, "too late: reset", ha="center", va="center", fontsize=12,
                color=P["red_d"], fontweight="bold")
        ax.text(0.8, 0.95, "earliest", ha="center", fontsize=12, color=P["slate"],
                alpha=a(k >= 2))
        ax.text(1.6, 0.95, "latest", ha="center", fontsize=12, color=P["slate"])
        # healthy-range bracket (eq 5)
        ax.annotate("", (0.88, -0.25), (1.52, -0.25),
                    arrowprops=dict(arrowstyle="<->", color=P["green_d"], lw=1.8),
                    alpha=a(k >= 3))
        ax.text(1.2, -0.62, "healthy feeds: period ± spread ± jitter", ha="center",
                fontsize=12, color=P["green_d"], alpha=a(k >= 3))
        save(fig, f"watchdog_step{k}")


# ---------------------------------------------------------------------------
# Illustrative stress trace (ch 13.7.3)
# ---------------------------------------------------------------------------
def trace():
    labels = ["p50", "p99", "p99.9", "max"]
    cold = [calcs.COLD["p50"], calcs.COLD["p99"], None, calcs.COLD["max"]]
    hot = [calcs.HOT["p50"], calcs.HOT["p99"], calcs.HOT["p99.9"], calcs.HOT["max"]]
    cold_txt = [val("cold_p50"), val("cold_p99"), None, val("cold_max")]
    hot_txt = [val("hot_p50"), val("hot_p99"), val("hot_p999"), val("hot_max")]
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(4.6, 3.1))
        xs = np.arange(len(labels))
        for i in range(len(labels)):
            if cold[i] is not None:
                ax.bar(xs[i] - 0.2, cold[i], 0.38, color=P["blue_l"], ec=P["navy"])
                ax.text(xs[i] - 0.2, cold[i] + 8, cold_txt[i], ha="center", fontsize=12,
                        color=P["navy"])
            ax.bar(xs[i] + 0.2, hot[i], 0.38, color=P["red_l"], ec=P["red"], alpha=a(k >= 2))
            ax.text(xs[i] + 0.2, hot[i] + 8, hot_txt[i], ha="center", fontsize=12,
                    color=P["red_d"], fontweight="bold", alpha=a(k >= 2))
        ax.axhline(calcs.WCET, color=P["slate"], lw=1.8, ls="--")
        ax.text(1.42, calcs.WCET + 8, "WCET", ha="left", fontsize=12,
                color=P["slate"], fontweight="bold")
        ax.axhline(calcs.DEADLINE, color=P["red"], lw=1.8)
        ax.text(-0.45, calcs.DEADLINE + 8, "deadline", fontsize=12,
                color=P["red_d"], fontweight="bold")
        ax.text(0.02, 0.97, "cold, idle", transform=ax.transAxes, fontsize=12,
                color=P["navy"], fontweight="bold", va="top")
        ax.text(0.40, 0.97, "warm, contended", transform=ax.transAxes, fontsize=12,
                color=P["red_d"], fontweight="bold", va="top", alpha=a(k >= 2))
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0, 560)
        ax.set_xlim(-0.5, 3.7)
        ax.set_ylabel("Tick time (µs)")
        fig.subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.97)
        save(fig, f"trace_step{k}")


if __name__ == "__main__":
    hook()
    budget()
    board()
    socmem()
    backlog()
    measured()
    stallshare()
    hangover()
    ladder()
    regulation()
    boardlink()
    watchdog()
    trace()
    print("figures written")
