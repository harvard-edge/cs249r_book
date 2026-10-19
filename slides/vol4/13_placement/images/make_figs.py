#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 13 (Workload Placement), Volume IV.

Every derived value comes from calcs.py (inputs from the rendered chapter,
Workload Placement, and the papers it cites). Values read from papers carry
their figure or section in the comment beside them (see calcs.py). Each
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

plt = sp.plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

calcs.main()
V = calcs.VALUES
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


def f(key):
    """A calcs value as a float (calcs prints strings with commas)."""
    return float(str(V[key]).replace(",", ""))


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
    t_trip = f("t_trip_s")
    rec = f("recover_s")
    tq, tl = f("t_quiet"), f("t_loaded")
    tau = calcs.TAU_S
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
        ax.text(0.4, calcs.T_TRIP - 4.5, f"trip {calcs.T_TRIP:.0f} °C", color=P["red"],
                fontsize=11, fontweight="bold")
        ax.axhline(calcs.T_CLEAR, color=P["green"], lw=1.5, ls="--", alpha=a(k >= 3))
        ax.text(0.4, calcs.T_CLEAR - 4.5, f"clear {calcs.T_CLEAR:.0f} °C", color=P["green"],
                fontsize=11, fontweight="bold", alpha=a(k >= 3))
        ax.plot([t_trip], [calcs.T_TRIP], "o", ms=8, color=P["red"], alpha=a(k >= 2))
        ax.annotate(f"{V['t_trip_s']} s: clock halved", (t_trip, calcs.T_TRIP),
                    xytext=(-12, 5), textcoords="offset points", ha="right",
                    fontsize=12, color=P["red_d"], fontweight="bold", alpha=a(k >= 2))
        ax.axvspan(t_trip, t_trip + rec, color=P["red_l"], lw=0, alpha=a(k >= 3) * 0.9)
        ax.text(t_trip + rec + 0.3, 58, f"{V['recover_s']} s\nafter stop",
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
        bx.text(t_trip / 2, 0.5, f"every tick on time: {V['resp_nom']} µs of 400",
                ha="center", va="center", fontsize=11, color=P["green_d"], fontweight="bold")
        bx.add_patch(Rectangle((t_trip, 0.15), rec, 0.7, color=P["red_l"], ec=P["red"],
                               alpha=a(k >= 2)))
        bx.text(t_trip + rec / 2, 1.12, f"{V['resp_thr']} µs", ha="center", va="bottom",
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
        ax.barh(y, calcs.SENSE_US, left=0, height=0.42, color=P["petrol"])
        ax.barh(y, calcs.CHECK_US, left=calcs.SENSE_US, height=0.42, color=P["green"])
        ax.barh(y, calcs.STAGE_US, left=calcs.SENSE_US + calcs.CHECK_US, height=0.42,
                color=P["green_d"])
        ax.text(0, 1.02, f"unloaded tick {V['unloaded_us']}", fontsize=12,
                color=P["green_d"], fontweight="bold")
        # declared WCET
        ax.plot([calcs.WCET_US] * 2, [0.15, 0.95], color=P["slate"], lw=2, ls="--",
                alpha=a(k >= 2))
        ax.text(calcs.WCET_US, 0.02, f"WCET {calcs.WCET_US:.0f}", ha="center", va="top",
                fontsize=12, color=P["slate"], fontweight="bold", alpha=a(k >= 2))
        # blocking + interference band and deadline
        ax.add_patch(Rectangle((f("ceiling_us"), 0.2), calcs.DEADLINE_US - f("ceiling_us"),
                               0.6, fc=P["gray_l"], hatch="//", ec=P["gray"], lw=1,
                               alpha=a(k >= 3)))
        ax.text(calcs.DEADLINE_US + 12, 0.5, "90 µs: the MCU's own tasks", fontsize=12,
                va="center", color=P["gray"], alpha=a(k >= 3))
        ax.plot([calcs.DEADLINE_US] * 2, [0.15, 0.95], color=P["red"], lw=2.2,
                alpha=a(k >= 3))
        ax.text(calcs.DEADLINE_US + 8, 0.02, f"deadline {calcs.DEADLINE_US:.0f}", va="top",
                fontsize=12, color=P["red_d"], fontweight="bold", alpha=a(k >= 3))
        # the 60 us allowance
        ax.add_patch(Rectangle((calcs.WCET_US, 0.2), f("ceiling_us") - calcs.WCET_US, 0.6,
                               fc=P["amber_l"], ec=P["amber"], lw=1.5, alpha=a(k >= 4)))
        ax.text((calcs.WCET_US + f("ceiling_us")) / 2, y, f"{V['allowance_us']}",
                ha="center", va="center", fontsize=13, color=P["amber"], fontweight="bold",
                alpha=a(k >= 4))
        ax.plot([calcs.PERIOD_US] * 2, [0.15, 0.95], color=P["slate"], lw=1.5,
                alpha=a(k >= 4))
        ax.text(calcs.PERIOD_US, 0.02, "next tick", ha="right", va="top", fontsize=12,
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
        box(ax, 8.6, 1.3, 2.7, 0.7, "isolated rail, 2 s", P["green_l"], P["green"], fs=11,
            bold=False, shown=k >= 4, lw=1.3)
        # battery bar
        box(ax, 0.1, -0.0 + 0.02, 11.6, 0.0, "", "white", "white", shown=False)
        ax.plot([5.3, 11.0], [0.65, 0.65], color=P["slate"], lw=0)
        box(ax, 5.3, 0.35, 6.4, 0.6, "battery and board supply: shared by all",
            P["slate_l"], P["slate"], fs=11, bold=False, lw=1.3)
        save(fig, f"board_step{k}")


# ---------------------------------------------------------------------------
# Backlog budget on one 1 ms period (Napkin Math 13.1, ch figure 2), 5 steps
# ---------------------------------------------------------------------------
def backlog():
    wait = f("backlog_us")
    wait1 = f("backlog1_us")
    snap = f("snap_us")
    rows = [("unloaded", 0, wait), (f"{V['n_sense']} backlogs", V["n_sense"], wait),
            (f"{V['n_wcet']} backlogs", V["n_wcet"], wait),
            (f"{V['n_dead']} backlogs", V["n_dead"], wait),
            (f"{V['n1_wcet']}, one channel", V["n1_wcet"], wait1)]
    for k in range(1, 6):
        fig, ax = plt.subplots(figsize=(5.8, 2.7))
        for i, (lab, n, w) in enumerate(rows):
            yy = len(rows) - 1 - i
            show = i < k
            mem = calcs.SENSE_US if n == 0 else n * w + snap
            if n == 0:
                ax.barh(yy, mem, height=0.6, color=P["petrol"], alpha=a(show))
            else:
                ax.barh(yy, mem, height=0.6, color=P["amber_l"], ec=P["amber"], hatch="///",
                        alpha=a(show), lw=1)
            ax.barh(yy, calcs.LATER_WORK_US, left=mem, height=0.6, color=P["green"],
                    alpha=a(show))
            ax.text(-12, yy, lab, ha="right", va="center", fontsize=12,
                    fontweight="bold", color=P["text"], alpha=a(show))
            ax.text(mem + calcs.LATER_WORK_US + 8, yy, f"{mem + calcs.LATER_WORK_US:.0f}",
                    va="center", fontsize=11, color=P["text"], alpha=a(show))
        for x, lab, c, ha, yo in ((calcs.SENSE_US, "40", P["petrol"], "center", 0.0),
                                  (calcs.WCET_US, "WCET 250", P["slate"], "center", 0.0),
                                  (calcs.DEADLINE_US, "deadline 400", P["red"], "left", 0.0),
                                  (calcs.PERIOD_US, "next tick", P["slate"], "right", 0.0)):
            ax.axvline(x, color=c, lw=1.6, ls="--")
            ax.text(x + (6 if ha == "left" else -6 if ha == "right" else 0),
                    len(rows) - 0.35 + yo, lab, ha=ha, fontsize=11, color=c,
                    fontweight="bold")
        ax.axvspan(f("ceiling_us"), calcs.DEADLINE_US, color=P["gray_l"], lw=0)
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
    rows = [("Cortex-A9, read co-runners", calcs.VALSAN_A9, "network", "2.1×"),
            ("Nehalem, read co-runners", calcs.VALSAN_NEH, "network", "7.9×"),
            ("Cortex-A15, read co-runners", calcs.VALSAN_A15, "network", "10.6×"),
            ("Cortex-A15, write co-runners", calcs.VALSAN_A15_W, "danger", "21.4×"),
            ("Cortex-A53, write attackers", calcs.BECHTEL_RPI3, "danger", "346×")]
    sp.hbars(str(OUT / "measured.svg"), rows, "", log=True,
             xlabel="Victim slowdown on its own core and cache partition", size=(5.8, 2.6))


# ---------------------------------------------------------------------------
# Tolerable miss-stall share against slowdown (eq 2), 2 steps
# ---------------------------------------------------------------------------
def stallshare():
    s = np.logspace(np.log10(1.5), np.log10(1000), 300)
    u = f("unloaded_us")
    wc = (calcs.WCET_US / u - 1) / (s - 1) * 100
    dl = (f("ceiling_us") / u - 1) / (s - 1) * 100
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.4, 3.3))
        ax.fill_between(s, wc, 100, color=P["red_l"], lw=0)
        ax.plot(s, wc, color=P["red"], lw=2.5)
        ax.plot(s, dl, color=P["gray"], lw=1.8, ls="--")
        ax.text(40, 25, "declared WCET\nrefuted", color=P["red_d"], fontsize=12,
                fontweight="bold")
        ax.text(60, 4.5, "deadline", color=P["gray"], fontsize=11)
        for sl, key, lab in ((21, "phi21", "21×: 4.3%"), (300, "phi300", "300×: 0.28%")):
            yv = f(key)
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
    rec = f("recover_s")
    tq = f("t_quiet")
    tau = calcs.TAU_S
    t = np.linspace(0, 6, 300)
    temp = tq + (calcs.T_TRIP - tq) * np.exp(-t / tau)
    for k in (1, 2):
        fig = plt.figure(figsize=(5.8, 3.2))
        ax = fig.add_axes([0.17, 0.56, 0.80, 0.40])
        bx = fig.add_axes([0.17, 0.12, 0.80, 0.30])
        ax.axvspan(0, rec, color=P["red_l"], lw=0)
        ax.text(rec / 2, 59, "600 MHz", ha="center", fontsize=12, color=P["red_d"],
                fontweight="bold")
        ax.plot(t, temp, color=P["slate"], lw=2.5)
        for yv, c, lab in ((calcs.T_TRIP, P["red"], "95"), (calcs.T_CLEAR, P["green"], "85"),
                           (tq, P["gray"], "53")):
            ax.axhline(yv, color=c, lw=1.3, ls="--")
        ax.text(rec + 0.1, 88, f"{V['recover_s']} s = {V['recover_ticks']} ticks",
                fontsize=12, color=P["red_d"], fontweight="bold")
        ax.set_xlim(0, 6)
        ax.set_ylim(50, 98)
        ax.set_yticks([53, 85, 95])
        ax.set_ylabel("°C")
        ax.text(5.95, 62, "seconds after\ninference stops", ha="right", fontsize=11,
                color=P["gray"])
        # tick bars
        bx.set_xlim(0, 1000)
        bx.set_ylim(-0.6, 1.6)
        for yy, ex, lab, c, cl, show in ((1, calcs.WCET_US, "1.2 GHz", P["green"], P["green_l"], True),
                                       (0, f("thr_us"), "600 MHz", P["red"], P["red_l"], k >= 2)):
            bx.barh(yy, ex, height=0.6, color=cl, ec=c, lw=1.5, alpha=a(show))
            bx.barh(yy, f("overhead_us"), left=ex, height=0.6, color=P["gray_l"],
                    ec=P["gray"], hatch="//", lw=1, alpha=a(show))
            bx.text(-15, yy, lab, ha="right", va="center", fontsize=12, fontweight="bold",
                    color=c, alpha=a(show))
            bx.text(ex / 2, yy, f"{ex:.0f} µs", ha="center", va="center", fontsize=11,
                    color=P["text"], fontweight="bold", alpha=a(show))
            bx.text(ex + f("overhead_us") + 10, yy, f"{ex + f('overhead_us'):.0f}" if yy == 0 else "",
                    va="center", fontsize=11, color=c, fontweight="bold", alpha=a(show))
        bx.axvline(calcs.DEADLINE_US, color=P["red"], lw=2)
        bx.text(calcs.DEADLINE_US + 8, 1.35, "deadline 400", ha="left", fontsize=11,
                color=P["red_d"], fontweight="bold")
        bx.set_yticks([])
        bx.set_xticks([0, 250, 400, 590, 1000])
        bx.set_xticklabels(["0", "250", "400", "590", "1,000"])
        bx.set_xlabel("µs within one tick")
        for s in ("left", "top", "right"):
            bx.spines[s].set_visible(False)
        save(fig, f"hangover_step{k}")


# ---------------------------------------------------------------------------
# Coupling-timescale ladder (ch 13.4)
# ---------------------------------------------------------------------------
def ladder():
    rows = [("thermal recovery", f("recover_s"), P["red"], f"{V['recover_s']} s"),
            ("permission tick", 1e-3, P["slate"], "1 ms"),
            ("DRAM backlog", f("ladder_backlog_us") * 1e-6, P["amber"],
             f"{V['ladder_backlog_us']} µs"),
            ("bank conflict", f("conflict_ns") * 1e-9, P["amber"], f"{V['conflict_ns']} ns"),
            ("droop rise", calcs.DT_S, P["violet"], "2 ns")]
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
    rows = [("no regulation", calcs.BECHTEL_FIG13_UNREG, "danger", "345×"),
            ("writes capped 100 MB/s", calcs.BECHTEL_W100, "data", "1.34×"),
            ("writes capped 50 MB/s", calcs.BECHTEL_W50, "data", "1.2×")]
    for k in (1, 2):
        sp.hbars(str(OUT / f"regulation_step{k}.svg"), rows, "", log=True, step=1 if k == 1 else None,
                 xlabel="Victim slowdown, Raspberry Pi 3, three write attackers", size=(5.8, 1.9))


# ---------------------------------------------------------------------------
# Board-link toll (Napkin Math 13.3, ch figure), 2 steps
# ---------------------------------------------------------------------------
def boardlink():
    segs = [("inference 40", calcs.INFER_MS, P["navy"], "white"),
            ("", f("link_ms"), P["violet"], "white"),
            ("18", calcs.VALID_MS, P["slate"], "white"),
            ("", f("small_ms"), P["gray"], "white")]
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.1))
        x = 0
        for lab, w, c, tc in segs:
            ax.barh(0, w, left=x, height=0.5, color=c)
            if lab:
                ax.text(x + w / 2, 0, lab, ha="center", va="center", color=tc,
                        fontsize=12, fontweight="bold")
            x += w
        ax.annotate(f"board link {V['link_ms']}", (calcs.INFER_MS + f("link_ms") / 2, 0.25),
                    xytext=(calcs.INFER_MS - 6, 0.62), fontsize=12, color=P["violet"],
                    fontweight="bold", ha="center",
                    arrowprops=dict(arrowstyle="-", color=P["violet"], lw=1.5))
        ax.text(x, 0.38, f"{V['d2a_ms']} ms", ha="center", fontsize=12,
                fontweight="bold", color=P["text"])
        ax.text(calcs.INFER_MS + f("link_ms") + calcs.VALID_MS / 2, -0.42, "validation",
                ha="center", va="center", fontsize=11, color=P["slate"])
        ax.add_patch(Rectangle((x, -0.25), f("ceil_ms") - x, 0.5, fill=False, ls=":",
                               ec=P["green"], lw=2, alpha=a(k >= 2)))
        ax.text((x + f("ceil_ms")) / 2, 0, f"reserve\n{V['reserve_ms']} ms",
                ha="center", va="center", fontsize=11, color=P["green_d"],
                fontweight="bold", alpha=a(k >= 2))
        ax.axvline(f("ceil_ms"), color=P["red"], lw=2.2, alpha=a(k >= 2))
        ax.text(f("ceil_ms"), 0.62, f"ceiling {V['ceil_ms']}", ha="center", fontsize=12,
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
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(4.6, 3.1))
        xs = np.arange(len(labels))
        for i in range(len(labels)):
            if cold[i] is not None:
                ax.bar(xs[i] - 0.2, cold[i], 0.38, color=P["blue_l"], ec=P["navy"])
                ax.text(xs[i] - 0.2, cold[i] + 8, f"{cold[i]}", ha="center", fontsize=12,
                        color=P["navy"])
            ax.bar(xs[i] + 0.2, hot[i], 0.38, color=P["red_l"], ec=P["red"], alpha=a(k >= 2))
            ax.text(xs[i] + 0.2, hot[i] + 8, f"{hot[i]}", ha="center", fontsize=12,
                    color=P["red_d"], fontweight="bold", alpha=a(k >= 2))
        ax.axhline(calcs.WCET_US, color=P["slate"], lw=1.8, ls="--")
        ax.text(1.42, calcs.WCET_US + 8, "WCET", ha="left", fontsize=12,
                color=P["slate"], fontweight="bold")
        ax.axhline(calcs.DEADLINE_US, color=P["red"], lw=1.8)
        ax.text(-0.45, calcs.DEADLINE_US + 8, "deadline", fontsize=12,
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
