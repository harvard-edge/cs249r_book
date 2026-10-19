"""Figures for Volume IV, Lecture 18 (Engineering Physical AI).

Run:  python3 make_figs.py   (writes SVGs here; `make deck V=4 D=18_frontier`
converts them to PDF). Every derived value comes from calcs.py; data plots read
the MLSysIM registry or a typed table with its source named on the slide.
Figures are drawn at the size of their slot: full width 5.8 in wide with
12 to 13 pt type, a column slot about 5.4 by 3.6 in with 15 to 16 pt type.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "assets" / "kit"))   # slides/assets/kit
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon  # noqa: E402
import numpy as np  # noqa: E402
import slideplot as sp  # noqa: E402
import calcs as c  # noqa: E402

# Volume IV roles (rules/volumes/vol4/palette.md)
NAVY = "#1F407A"      # cognitive deliberation (learned model)
PROPOSE = "#2563EB"   # unprivileged proposal
NERVE = "#059669"     # deterministic nervous system, permission path
NERVE_L = "#DCFCE7"
RED = "#DC2626"       # consequence: clearance, veto, hazard
RED_L = "#FEE2E2"
PLANT = "#334155"     # physical plant
PETROL = "#0F766E"    # sensory transduction
AMBER = "#D97706"     # latency lag, delay travel
GRAY = "#94A3B8"
LIGHT = "#E2E8F0"
PINE = "#1A4D3E"      # volume accent
TEXT = "#1A1A2E"


def out(name):
    return str(HERE / name)


def box(ax, x, y, w, h, fc, ec, lw=1.6, r=0.06, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def arrow(ax, x0, y0, x1, y1, col, lw=2.0, style="-|>", z=3):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=col, lw=lw, shrinkA=0, shrinkB=0,
                                mutation_scale=14), zorder=z)


# ---------------------------------------------------------------------------
# Segment 1: the closed loop at three rates (sec. 18.1, figure 1), column slot
# ---------------------------------------------------------------------------
def closed_loop(step):
    fig = plt.figure(figsize=(5.9, 3.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 5.9)
    ax.set_ylim(0, 3.7)
    ax.axis("off")
    levels = [  # name, y, fill, edge, rate
        ("Brain", 3.05, "#EFF6FF", NAVY, "proposes at 2 Hz"),
        ("Nervous System", 2.05, NERVE_L, NERVE, "admits at 1 kHz"),
        ("Body", 1.45, "#F8FAFC", PLANT, "current loop 20 kHz"),
        ("Boundary", 0.85, "#F8FAFC", PLANT, "event-timed"),
        ("World", 0.25, "#F0FDFA", PETROL, "continuous"),
    ]
    for name, y, fc, ec, rate in levels:
        box(ax, 0.15, y, 3.75, 0.48, fc, ec, r=0.05)
        ax.text(0.3, y + 0.24, name, fontsize=15, fontweight="bold", color=ec, va="center")
        ax.text(3.78, y + 0.24, rate, fontsize=14, color=TEXT, va="center", ha="right")
    ax.plot([0.15, 3.9], [2.79, 2.79], color=GRAY, lw=1.5, ls="--")
    ax.text(2.0, 2.79, "proposal boundary", fontsize=14, color=PLANT, ha="center", va="center", style="italic",
            bbox=dict(fc="white", ec="none", pad=1.5))
    if step >= 2:
        arrow(ax, 4.05, 0.49, 4.6, 0.49, PETROL)
        ax.plot([4.6, 4.6], [0.49, 3.29], color=PETROL, lw=2.0)
        arrow(ax, 4.6, 2.29, 3.95, 2.29, PETROL)
        arrow(ax, 4.6, 3.29, 3.95, 3.29, PETROL)
        ax.text(4.68, 2.29, "fresh\nstate", fontsize=15, color=PETROL, va="center")
        ax.text(4.68, 3.29, "obser-\nvations", fontsize=15, color=PETROL, va="center")
        ax.text(4.68, 0.49, "only what\nsensors\ncapture", fontsize=15, color=RED, va="center",
                fontweight="bold")
    sp.save(fig, out(f"closed_loop_step{step}.svg"))


# ---------------------------------------------------------------------------
# Segment 1: Open X-Embodiment, episodes by robot (real data), full width
# ---------------------------------------------------------------------------
def oxe():
    from collections import Counter
    from mlsysim import Literature
    ep = Counter()
    for r in Literature.OpenXEmbodimentComposition.entries:
        ep[r["robot"]] += r["episodes"] or 0
    top = ep.most_common(8)
    rest = sum(ep.values()) - sum(v for _, v in top)
    n_robots = len(ep)
    rows = [(k, v) for k, v in top] + [(f"{n_robots - 8} other robots", rest)]
    fig, ax = plt.subplots(figsize=(5.8, 2.55))
    for i, (k, v) in enumerate(rows):
        y = len(rows) - 1 - i
        col = NAVY if i < 8 else GRAY
        ax.barh(y, v / 1000, color=col, height=0.66)
        ax.text(v / 1000 + 8, y, f"{v / 1000:,.0f}k", va="center", fontsize=11, color=col, fontweight="bold")
        ax.text(-8, y, k, va="center", ha="right", fontsize=11, color=TEXT, fontweight="bold")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 820)
    ax.set_xlabel("Episodes (thousands)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.text(810, 1.2, f"{n_robots} robots\n{sum(ep.values()) / 1e6:.2f} million episodes", ha="right",
            fontsize=12, color=NAVY, fontweight="bold", va="center")
    ax.grid(axis="x", color=sp.C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out("oxe_robots.svg"))
    return n_robots, sum(ep.values())


# ---------------------------------------------------------------------------
# Segment 2: the packaging cell, three machine classes (sec. 18.2.2 panel a)
# ---------------------------------------------------------------------------
def cell_schematic():
    fig = plt.figure(figsize=(5.6, 3.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 5.6)
    ax.set_ylim(0, 3.0)
    ax.axis("off")
    ax.plot([0.9, 5.2], [1.0, 1.0], color=PLANT, lw=4)
    ax.add_patch(Rectangle((0.15, 0.75), 0.75, 0.62, fc="#F8FAFC", ec=PLANT, lw=1.6))
    ax.text(0.52, 1.06, "drive", fontsize=14, ha="center", va="center", color=PLANT, fontweight="bold")
    ax.add_patch(Rectangle((2.0, 1.03), 1.1, 0.42, fc="#F8FAFC", ec=PLANT, lw=1.6))
    ax.text(2.55, 1.24, "carriage", fontsize=14, ha="center", va="center", color=PLANT, fontweight="bold")
    ax.add_patch(Rectangle((2.2, 1.45), 0.7, 0.32, fc="white", ec=PLANT, lw=1.2))
    ax.text(2.55, 1.61, "carton", fontsize=12, ha="center", va="center", color=PLANT)
    ax.add_patch(Rectangle((2.3, 2.15), 0.5, 0.3, fc="white", ec=PLANT, lw=1.4))
    arrow(ax, 2.55, 2.15, 2.55, 1.8, PLANT)
    arrow(ax, 3.15, 1.24, 3.75, 1.24, PLANT)
    ax.text(3.45, 1.32, "1.50 m/s", fontsize=12, ha="center", color=PLANT)
    ax.add_patch(Rectangle((4.7, 1.05), 0.12, 0.6, fc="white", ec=PLANT, lw=1.4))
    ax.plot([4.6, 4.6], [0.55, 2.0], color=RED, lw=2, ls="--")
    ax.text(4.85, 1.85, "gate", fontsize=14, color=PLANT, fontweight="bold")
    ax.text(4.6, 0.45, "stop line", fontsize=12, color=RED, ha="center", va="top")
    # class labels
    ax.text(0.52, 0.55, "Class 3: heat", fontsize=13, color=AMBER, ha="center", va="top", fontweight="bold")
    ax.text(2.55, 0.85, "Class 1: moving mass", fontsize=13, color=RED, ha="center", va="top", fontweight="bold")
    ax.text(2.95, 2.3, "Class 2: contact force", fontsize=13, color=NAVY, ha="left", va="center",
            fontweight="bold")
    sp.save(fig, out("cell_schematic.svg"))


# ---------------------------------------------------------------------------
# Segment 2: the four-law audit, one question per law (sec. 18.2.1), full width
# ---------------------------------------------------------------------------
AUDIT = [
    ("1 Irreversibility", "What can the task not undo,\nand which bound excludes it?", "limit records", PLANT),
    ("2 Endogenous data", "Does evaluation cover the\nstates inside that envelope?", "provenance,\nevaluation record", sp.C["data"]),
    ("3 Proposal is not\npermission", "Which independent path\nadmits each proposal, in time?", "enforcement record,\ntiming budget", NERVE),
    ("4 Permission must\nbe earned", "Which claims does the\nevidence support? Who signs?", "release manifest,\nregister entries", NAVY),
]


def audit(step):
    fig = plt.figure(figsize=(5.8, 2.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 5.8)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    h, gap = 0.55, 0.08
    for i, (law, q, rec, col) in enumerate(AUDIT):
        y = 2.6 - (i + 1) * (h + gap) + 0.02
        a = 1.0 if i < step else 0.0
        box(ax, 0.05, y, 1.72, h, "white", col, lw=2.0 * a + 0.001)
        ax.text(0.12, y + h / 2, law, fontsize=11, fontweight="bold", color=col, va="center", alpha=a,
                linespacing=1.0)
        ax.text(1.85, y + h / 2, q, fontsize=10.5, color=TEXT, va="center", alpha=a, linespacing=1.05)
        box(ax, 4.3, y + 0.03, 1.45, h - 0.06, "#F1F5F9", col, lw=1.2 * a + 0.001, r=0.04)
        ax.text(5.025, y + h / 2, rec, fontsize=10, color=col, ha="center", va="center", alpha=a,
                fontweight="bold", linespacing=1.0)
        if a:
            arrow(ax, 4.1, y + h / 2, 4.28, y + h / 2, col, lw=1.5)
    sp.save(fig, out(f"audit_step{step}.svg"))


# ---------------------------------------------------------------------------
# Segment 2: the delay wall, stop bars at 1.50 and 0.50 m/s (Napkin Math 18.1)
# ---------------------------------------------------------------------------
def cell_stop(step):
    tau = c.cell_tau()
    rows = [(f"{c.CELL_V:.2f} m/s", *c.cell_stop(c.CELL_V, tau)),
            (f"{c.CELL_V_MODE:.2f} m/s", *c.cell_stop(c.CELL_V_MODE, tau))]
    fig, ax = plt.subplots(figsize=(5.8, 2.0))
    clear = c.CELL_CLEAR * 1000
    res = c.CELL_RESERVE * 1000
    ax.add_patch(Rectangle((clear - res, -0.6), res, 2.15, fc="none", ec=GRAY, hatch="///", lw=0))
    ax.text(clear - res - 2, 1.62, "reserve", fontsize=11, color=PLANT, ha="right", va="bottom")
    ax.axvline(clear, color=RED, lw=2, ls="--")
    ax.text(clear + 2, 1.62, f"stop line {clear:.0f} mm", fontsize=11, color=RED, va="bottom")
    for i, (lab, d1, d2) in enumerate(rows):
        y = 1 - i
        a = 1.0 if i < step else 0.0
        d1, d2 = d1 * 1000, d2 * 1000
        ax.barh(y, d1, color=AMBER, height=0.55, alpha=a)
        ax.barh(y, d2, left=d1, color=PLANT, height=0.55, alpha=a)
        ax.text(-4, y, lab, ha="right", va="center", fontsize=12, fontweight="bold", color=TEXT, alpha=a)
        if i == 0:
            ax.text(d1 * 0.32, y, f"delay {d1:.0f}", color="white", ha="center", va="center", fontsize=11,
                    fontweight="bold", alpha=a)
            ax.text(d1 + d2 / 2, y, f"braking {d2:.0f}", color="white", ha="center", va="center",
                    fontsize=11, fontweight="bold", alpha=a)
            ax.text(d1 + d2 + 3, y, f"{d1 + d2:.0f} mm", color=RED, va="center", fontsize=12,
                    fontweight="bold", alpha=a)
        else:
            ax.text(clear + 3, y, f"{d1:.0f} + {d2:.1f} = {d1 + d2:.1f} mm, {clear - d1 - d2:.1f} mm spare",
                    color=PLANT, va="center", fontsize=11, fontweight="bold", alpha=a)
    ax.set_xlim(0, 200)
    ax.set_ylim(-0.6, 1.95)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Distance from closure, mm", fontsize=11)
    sp.save(fig, out(f"cell_stop_step{step}.svg"))


# ---------------------------------------------------------------------------
# Segment 2: what moves the ceiling (sec. 18.2.2 and the deTec4 datasheet)
# ---------------------------------------------------------------------------
def cell_ceilings():
    tau = c.cell_tau()
    budget = c.CELL_CLEAR - c.CELL_RESERVE
    half = [(n, v / 2 if n.startswith("transfer") else v) for n, v in c.CELL_CHAIN]
    curtain = [("curtain", c.CURTAIN_MS), ("bus", 2.0), ("brake onset", 10.0)]
    rows = [
        ("Certified curtain,\n22 ms chain", c.ceiling(c.cell_tau(curtain), c.CELL_A, budget), NERVE),
        ("Processing halved,\n46 ms chain", c.ceiling(c.cell_tau(half), c.CELL_A, budget), NERVE),
        ("As planned,\n64 ms chain", c.ceiling(tau, c.CELL_A, budget), PLANT),
        ("Brake halved,\n7.5 m/s²", c.ceiling(tau, c.CELL_A / 2, budget), RED),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for i, (lab, v, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, color=col, height=0.6)
        ax.text(0.03, y, f"{v:.2f} m/s", va="center", fontsize=15, fontweight="bold", color="white")
        ax.text(-0.02, y, lab, va="center", ha="right", fontsize=14, color=TEXT, fontweight="bold",
                linespacing=1.0)
    ax.axvline(c.CELL_V_MODE, color=PINE, lw=2, ls="--")
    ax.text(c.CELL_V_MODE, 3.55, "0.50 m/s mode", color=PINE, fontsize=14, ha="center", va="bottom",
            fontweight="bold", bbox=dict(fc="white", ec="none", pad=1))
    ax.set_xlim(0, 1.15)
    ax.set_ylim(-0.5, 3.9)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=14)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Speed ceiling, m/s", fontsize=14)
    sp.save(fig, out("cell_ceilings.svg"))


# ---------------------------------------------------------------------------
# Segment 2: the restriction moves the proposer off its evaluated states
# ---------------------------------------------------------------------------
def derate_gap():
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    v = np.linspace(0, 2.0, 400)
    ev = np.exp(-0.5 * ((v - 1.5) / 0.12) ** 2)
    rt = np.exp(-0.5 * ((v - 0.45) / 0.06) ** 2)
    ax.fill_between(v, 0, ev, color=NAVY, alpha=0.25)
    ax.plot(v, ev, color=NAVY, lw=2.5)
    ax.fill_between(v, 0, rt, color=RED, alpha=0.2)
    ax.plot(v, rt, color=RED, lw=2.5)
    ax.text(1.5, 1.08, "states evaluated\nat 1.50 m/s", color=NAVY, fontsize=15, ha="center",
            fontweight="bold", va="bottom")
    ax.text(0.02, 1.38, "states admitted\nat 0.50 m/s", color=RED, fontsize=15, ha="left",
            fontweight="bold", va="bottom")
    ax.plot([0.55, 0.55], [0, 1.25], color=PINE, lw=2, ls="--")
    ax.text(0.6, 0.45, "ceiling\n0.55 m/s", color=PINE, fontsize=14, va="center")
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 1.95)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=14)
    ax.set_xlabel("Carriage speed, m/s (shapes illustrative)", fontsize=14)
    sp.save(fig, out("derate_gap.svg"))


# ---------------------------------------------------------------------------
# Segment 3: the friction floor, a = min(a_brake, mu g) (Napkin Math 18.2)
# ---------------------------------------------------------------------------
def friction_knee():
    mu_min = c.A_BRAKE / c.G
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    mu = np.linspace(0, 0.7, 300)
    a = np.minimum(c.A_BRAKE, mu * c.G)
    ax.axvspan(0, mu_min, color=RED_L, alpha=0.7, lw=0)
    ax.plot(mu, a, color=AMBER, lw=3)
    ax.axvline(mu_min, color=PLANT, lw=1.5, ls="--")
    ax.text(mu_min / 2, 2.35, "floor binds", color=RED, fontsize=15, ha="center", fontweight="bold")
    ax.text(0.47, 2.35, "brake binds", color=PLANT, fontsize=15, ha="center", fontweight="bold")
    ax.text(mu_min + 0.01, 0.1, f"μ = {mu_min:.3f}", color=PLANT, fontsize=14)
    pts = [("dry 0.60", c.MU_DRY, PLANT, (0.6, 1.62)), ("inspected 0.12", c.MU_INSP, AMBER, (0.25, 1.05)),
           ("film 0.05", c.MU_OIL, RED, (0.25, 0.4))]
    for lab, m, col, xy in pts:
        y = c.a_avail(m)
        ax.plot([m], [y], "o", ms=10, color=col, mec="white", mew=1.5, zorder=5)
        ax.annotate(f"{lab}: {y:.2f} m/s²", (m, y), xytext=xy, textcoords="data",
                    fontsize=13, color=col, fontweight="bold", ha="right" if m > 0.5 else "left",
                    va="center", arrowprops=None if m > 0.5 else dict(arrowstyle="-", color=col, lw=1))
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 2.7)
    ax.tick_params(labelsize=13)
    ax.set_xlabel("Floor friction coefficient μ", fontsize=14)
    ax.set_ylabel("Available decel., m/s²", fontsize=14)
    sp.save(fig, out("friction_knee.svg"))


# ---------------------------------------------------------------------------
# Segment 3: measured locked-wheel friction (TRL PPR073, table 5.1), full width
# ---------------------------------------------------------------------------
def trl_friction():
    mu_min = c.A_BRAKE / c.G
    rows = c.TRL_ROWS
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    for i, (lab, mu) in enumerate(rows):
        y = len(rows) - 1 - i
        col = RED if mu < mu_min else (AMBER if "smooth" in lab else PLANT)
        ax.barh(y, mu, color=col, height=0.62)
        ax.text(mu + 0.012, y, f"{mu:.2f}", va="center", fontsize=12, fontweight="bold", color=col)
        ax.text(-0.012, y, lab, va="center", ha="right", fontsize=11.5, color=TEXT, fontweight="bold")
    ax.axvline(mu_min, color=RED, lw=1.8, ls="--")
    ax.text(mu_min + 0.01, 5.55, f"floor binds below {mu_min:.3f}", color=RED, fontsize=11,
            fontweight="bold", va="bottom")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, 6.1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.set_xlabel("Measured locked-wheel friction coefficient", fontsize=11)
    sp.save(fig, out("trl_friction.svg"))


# ---------------------------------------------------------------------------
# Segment 3 (and the hook): the budget's terms on dry concrete and on the film
# ---------------------------------------------------------------------------
def film_terms(step):
    fig, ax = plt.subplots(figsize=(5.8, 2.1))
    names = [("delay", AMBER), ("braking", PLANT), ("ramp", GRAY), ("overhead", "#64748B")]
    for i, (lab, mu) in enumerate((("dry floor", c.MU_DRY), ("film", c.MU_OIL))):
        y = 1 - i
        a = 1.0 if i < step else 0.0
        t = c.budget_terms(c.V_AISLE, mu)
        x = 0
        for key, col in names:
            v = t[key] * 1000
            ax.barh(y, v, left=x, color=col, height=0.55, alpha=a, edgecolor="white", lw=1)
            if v > 300:
                ax.text(x + min(v / 2, 230), y, f"{v:,.1f}", color="white", ha="center", va="center", fontsize=11,
                        fontweight="bold", alpha=a)
            x += v
        ax.text(-20, y, lab, ha="right", va="center", fontsize=12, fontweight="bold", color=TEXT, alpha=a)
        tot = x / 1000
        ax.text(max(x, c.D_CLEAR * 1000) + 30, y, f"{tot * 1000:,.1f} mm" if tot < 1.5 else f"{tot:.2f} m", va="center",
                fontsize=12, fontweight="bold", color=RED if tot > c.D_CLEAR else PLANT, alpha=a)
    ax.axvline(c.D_CLEAR * 1000, color=RED, lw=2, ls="--")
    ax.text(c.D_CLEAR * 1000 + 15, 1.5, "clear distance 1,100 mm", color=RED, fontsize=11,
            fontweight="bold", va="bottom")
    td = c.budget_terms(c.V_AISLE, c.MU_DRY)
    legend = f"delay {td['delay']*1000:.1f}   ramp {td['ramp']*1000:.1f}   overheads {td['overhead']*1000:.0f} mm"
    ax.text(1150, -0.72, legend, color=PLANT, fontsize=11, fontweight="bold", va="top")
    ax.set_xlim(0, 2600)
    ax.set_ylim(-1.05, 1.85)
    ax.set_yticks([])
    ax.set_xticks([0, 500, 1000, 1500, 2000, 2500])
    ax.set_xticklabels(["0", "500", "1,000", "1,500", "2,000", "2,500 mm"])
    ax.tick_params(labelsize=10)
    ax.spines["left"].set_visible(False)
    sp.save(fig, out(f"film_terms_step{step}.svg"))


# ---------------------------------------------------------------------------
# Segment 3: observational indistinguishability (definition 18.1), full width
# ---------------------------------------------------------------------------
def indist():
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    t = np.linspace(0, 10, 400)
    base = 0.5 + 0.08 * np.sin(1.3 * t)
    td = 6.0
    ax.axvspan(0, td, color="#FEF3C7", alpha=0.7, lw=0)
    ax.plot(t, base + 1.0, color=PROPOSE, lw=2.5)
    haz = np.where(t < td, base, base - 0.25 * (1 - np.exp(-(t - td) / 0.8)))
    ax.plot(t, haz, color=RED, lw=2.5)
    ax.plot(t[t < td], base[t < td] + 1.0, color=PROPOSE, lw=2.5)
    ax.text(0.15, 1.75, "dry floor: sensor history", color=PROPOSE, fontsize=12, fontweight="bold")
    ax.text(0.15, 0.13, "film: the same history", color=RED, fontsize=12, fontweight="bold")
    ax.axvline(td, color=TEXT, lw=1.8, ls="--")
    ax.text(td + 0.1, 1.95, "deadline: last moment a\nstop still ends in time", fontsize=11, color=TEXT,
            va="top")
    ax.text(td / 2, 1.15, "identical until the deadline", fontsize=12, color=AMBER, ha="center",
            fontweight="bold")
    ax.text(8.2, 0.05, "differs too late", fontsize=11, color=RED, ha="center")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.05, 2.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time as the base approaches the film", fontsize=11)
    sp.save(fig, out("indist.svg"))


# ---------------------------------------------------------------------------
# Segment 3: time form against distance form (pitfall, sec. 18.3.4)
# ---------------------------------------------------------------------------
def overreserve():
    from mlsysim.physics import calc_scurve_stop_time, calc_scurve_stop_distance
    from mlsysim.core.units import meter, second, ms, millimeter
    vv = c.V_AISLE * meter / second
    a = c.A_BRAKE * meter / second**2
    j = c.J_BRAKE * meter / second**3
    t_stop = calc_scurve_stop_time(vv, a, j).to(ms).magnitude
    d_stop = calc_scurve_stop_distance(vv, a, j).to(millimeter).magnitude
    t_eq = d_stop / c.V_AISLE
    t = c.budget_terms(c.V_AISLE, c.MU_DRY)
    spare_ms = (c.D_CLEAR - sum(t.values())) / c.V_AISLE * 1000
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    ax.barh(2, t_eq, color=PLANT, height=0.55)
    ax.barh(2, t_stop - t_eq, left=t_eq, color=RED, height=0.55)
    ax.text(t_eq / 2, 2, f"{t_eq:.1f}", color="white", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(t_eq + (t_stop - t_eq) / 2, 2, f"+{t_stop - t_eq:.1f} over", color="white", ha="center",
            va="center", fontsize=12, fontweight="bold")
    ax.barh(1, t_eq, color=PLANT, height=0.55)
    ax.text(t_eq + 10, 1, f"{t_eq:.1f} ms", va="center", fontsize=12, fontweight="bold", color=PLANT)
    ax.barh(0, spare_ms, color=GRAY, height=0.55)
    ax.text(spare_ms + 10, 0, f"{spare_ms:.1f} ms", va="center", fontsize=12, fontweight="bold", color=PLANT)
    for y, lab in ((2, f"stop time {t_stop:.1f} ms"), (1, "distance / speed"), (0, "the whole spare")):
        ax.text(-10, y, lab, ha="right", va="center", fontsize=12, fontweight="bold", color=TEXT)
    ax.set_xlim(0, 1000)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.set_xlabel("ms at 1.3 m/s", fontsize=11)
    sp.save(fig, out("overreserve.svg"))


# ---------------------------------------------------------------------------
# Segment 3: five responses when the state stays hidden (sec. 18.3.4)
# ---------------------------------------------------------------------------
def responses(step):
    fig = plt.figure(figsize=(5.8, 3.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 5.8)
    ax.set_ylim(0, 3.0)
    ax.axis("off")
    box(ax, 0.05, 1.95, 1.9, 0.95, "#FEF3C7", AMBER)
    ax.text(1.0, 2.42, "Can an admissible\nprobe separate the\nstates in time?", fontsize=12.5, ha="center",
            va="center", color=TEXT, fontweight="bold", linespacing=1.1)
    box(ax, 2.35, 2.15, 1.55, 0.55, "#EFF6FF", PROPOSE)
    ax.text(3.125, 2.42, "active probing", fontsize=13, ha="center", va="center", color=PROPOSE, fontweight="bold")
    arrow(ax, 1.95, 2.42, 2.34, 2.42, PLANT)
    ax.text(2.15, 2.5, "yes", fontsize=12, ha="center", va="bottom", color=PLANT)
    ax.text(3.98, 2.42, "torque pulse on\nthe floor; proof-load\ntug on the latch", fontsize=12, va="center",
            color=TEXT, linespacing=1.05)
    if step >= 2:
        arrow(ax, 1.0, 1.95, 1.0, 1.55, PLANT)
        ax.text(1.08, 1.75, "no", fontsize=12, va="center", color=PLANT)
        opts = [("proof test", "tare the sensor;\nload-test latch", NERVE),
                ("conservative\naction", "a speed safe\non both floors", NERVE),
                ("passive\ncontainment", "a spring that\ncaps force", PLANT),
                ("restricted\ndomain", "no routes without\nspill procedure", RED)]
        w = 1.38
        for i, (n, ex, col) in enumerate(opts):
            x = 0.05 + i * (w + 0.06)
            box(ax, x, 0.85, w, 0.68, "white", col)
            ax.text(x + w / 2, 1.19, n, fontsize=12.5, ha="center", va="center", color=col, fontweight="bold",
                    linespacing=1.0)
            ax.text(x + w / 2, 0.42, ex, fontsize=12, ha="center", va="center", color=TEXT, linespacing=1.05)
    sp.save(fig, out(f"responses_step{step}.svg"))


# ---------------------------------------------------------------------------
# Segment 3: peak latch force with torque removed (sec. 18.3.4), column slot
# ---------------------------------------------------------------------------
def latch_bars():
    import math
    k_eq = (c.F_FLOOR - c.F_TRIP) / (c.V_FAST * c.T_RESP + c.V_FAST**2 / (
        2 * c.V_GUARD**2 / (2 * ((c.F_FLOOR - c.F_TRIP) / c.K_PLATE - c.V_GUARD * c.T_RESP))))

    def coast(k, v):
        return math.sqrt(c.F_TRIP**2 + k * c.M_EFF * v**2)
    rows = [("rigid plate,\n0.03 m/s", coast(c.K_PLATE, c.V_GUARD)),
            ("rigid plate,\n0.10 m/s", coast(c.K_PLATE, c.V_FAST)),
            ("series spring,\n0.10 m/s", coast(k_eq, c.V_FAST))]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for i, (lab, f) in enumerate(rows):
        y = 2 - i
        ax.barh(y, min(f, c.F_FLOOR), color=AMBER, height=0.6)
        if f > c.F_FLOOR:
            ax.barh(y, f - c.F_FLOOR, left=c.F_FLOOR, color=RED, height=0.6)
        if f > c.F_FLOOR:
            ax.text(f + 4, y, f"{f:.1f} N", va="center", fontsize=15, fontweight="bold", color=RED)
        else:
            ax.text(f / 2, y, f"{f:.1f} N", va="center", ha="center", fontsize=15, fontweight="bold", color="white")
        ax.text(-4, y, lab, va="center", ha="right", fontsize=14, color=TEXT, fontweight="bold", linespacing=1.0)
    ax.axvline(c.F_FLOOR, color=RED, lw=2, ls="--")
    ax.text(c.F_FLOOR + 5, 2.55, "100 N declared floor", color=RED, fontsize=14, ha="left", va="bottom",
            fontweight="bold")
    ax.set_xlim(0, 290)
    ax.set_ylim(-0.5, 2.95)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=13)
    ax.set_xlabel("Peak latch force, torque removed at 15 N", fontsize=13)
    sp.save(fig, out("latch_bars.svg"))


# ---------------------------------------------------------------------------
# Segment 4: Waymo rider-only injury rates against the human benchmark (real)
# ---------------------------------------------------------------------------
def waymo():
    from mlsysim import Literature
    rows = sorted((r for r in Literature.WaymoSafetyImpact.entries if r["outcome"] == "Any-injury-reported"),
                  key=lambda r: r["data_cut"])
    miles = [r["ro_miles_millions"] for r in rows]
    wy = [r["waymo_ipmm"] for r in rows]
    hb = [r["benchmark_ipmm"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.8, 2.45))
    ax.plot(miles, hb, "o-", color=GRAY, lw=2.5, ms=6)
    ax.plot(miles, wy, "o-", color=NAVY, lw=2.5, ms=6)
    ax.text(miles[-1] + 6, hb[-1], f"human benchmark\n{hb[-1]:.2f}", color=PLANT, fontsize=11,
            fontweight="bold", va="center")
    ax.text(miles[-1] + 6, wy[-1], f"Waymo, rider-only\n{wy[-1]:.2f}", color=NAVY, fontsize=11,
            fontweight="bold", va="center")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 4.8)
    ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.tick_params(labelsize=10)
    ax.set_xlabel("Cumulative rider-only miles (millions)", fontsize=11)
    ax.set_ylabel("Injury crashes per\nmillion miles", fontsize=11)
    ax.grid(axis="y", color=sp.C["grid"], lw=0.6)
    sp.save(fig, out("waymo_rates.svg"))
    return rows[-1]


# ---------------------------------------------------------------------------
# Synthesis: the finished aisle budget at 1.3 m/s, one row per owning lecture
# ---------------------------------------------------------------------------
def budget_final(step):
    t = c.budget_terms(c.V_AISLE, c.MU_DRY)
    tf = c.budget_terms(c.V_AISLE, c.MU_OIL)
    da = c.V_AISLE * c.DELTA_DA * 1000
    segs = [("delay", t["delay"] * 1000, AMBER, "delay L1, L4, L8", 1.62),
            ("braking", t["braking"] * 1000, PLANT, "braking L1, L2", 1.3),
            ("jerk ramp", t["ramp"] * 1000, GRAY, "ramp L11", 1.62),
            ("overheads", t["overhead"] * 1000, "#64748B", "overheads L12", 1.3)]
    fig, ax = plt.subplots(figsize=(5.8, 2.45))
    x = 0
    for lab, v, col, tag, ty in segs:
        ax.barh(1, v, left=x, color=col, height=0.5, edgecolor="white", lw=1)
        if v > 300:
            ax.text(x + v / 2, 1, f"{v:.1f}", color="white", ha="center", va="center", fontsize=10.5,
                    fontweight="bold")
        x += v
    tot = x
    if step >= 2:
        ax.barh(1, da, left=x, color=NERVE, height=0.5, edgecolor="white", lw=1)
        ax.text(c.D_CLEAR * 1000 + 25, 1, f"{c.D_CLEAR * 1000 - tot - da:.1f} mm spare (L16)", color=NERVE,
                va="center", fontsize=11, fontweight="bold")
    else:
        ax.text(c.D_CLEAR * 1000 + 25, 1, f"{tot:.1f} mm known", color=PLANT, va="center", fontsize=11,
                fontweight="bold")
    if step >= 3:
        x = 0
        for (lab, v0, col, tag, ty), key in zip(segs, ("delay", "braking", "ramp", "overhead")):
            v = tf[key] * 1000
            ax.barh(0, v, left=x, color=col, height=0.5, edgecolor="white", lw=1)
            x += v
        ax.text(x + 25, 0, f"on the film: {x / 1000:.2f} m (L18)", color=RED, va="center", fontsize=11,
                fontweight="bold")
    pos = [(0, 1.72), (950, 1.72), (1850, 1.72), (0, 1.42), (950, 1.42)]
    tags = [(tag, col) for _, _, col, tag, _ in segs] + [("dispatch L13", NERVE)]
    for k, ((tag, col), (px, py)) in enumerate(zip(tags, pos)):
        if k == 4 and step < 2:
            break
        ax.text(px, py, tag, color=col, fontsize=10.5, fontweight="bold", va="bottom")
    ax.plot([c.D_CLEAR * 1000] * 2, [-0.4, 1.32], color=RED, lw=2, ls="--")
    ax.text(c.D_CLEAR * 1000, -0.42, "1,100 mm clear", color=RED, fontsize=10.5, ha="center", va="top",
            fontweight="bold", bbox=dict(fc="white", ec="none", pad=1))
    ax.set_xlim(0, 3200)
    ax.set_ylim(-0.75, 2.05)
    ax.set_yticks([])
    ax.set_xticks([])
    for sname in ("left", "bottom"):
        ax.spines[sname].set_visible(False)
    sp.save(fig, out(f"budget_final_step{step}.svg"))


# ---------------------------------------------------------------------------
# Synthesis: the record chain, closed by the residual-claims register
# ---------------------------------------------------------------------------
def record_chain():
    rows = [
        ("Part I", "irreversibility", PLANT, [(2, "limit"), (4, "handoff")]),
        ("Part II", "endogenous data", sp.C["data"], [(5, "provenance"), (6, "policy manifest"),
                                                      (7, "evaluation")]),
        ("Part III", "proposal is not", NERVE, [(8, "observation contract"), (9, "belief"),
                                                (10, "intent lease")]),
        ("", "permission", NERVE, [(11, "trajectory"), (12, "enforcement"), (13, "placement")]),
        ("Part IV", "permission must", NAVY, [(14, "authority"), (15, "fault"), (16, "release manifest")]),
        ("", "be earned", NAVY, [(17, "approval")]),
        ("Close", "", RED, [(18, "residual-claims register")]),
    ]
    fig = plt.figure(figsize=(5.8, 2.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 5.8)
    ax.set_ylim(-2.55, 0.2)
    ax.axis("off")
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def width(tx):
        bb = tx.get_window_extent(renderer=rend)
        (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return x1 - x0

    y = 0
    for part, law, col, recs in rows:
        ax.text(0.05, y, part, color=col, fontsize=11, fontweight="bold", va="center")
        ax.text(0.78, y, law, color=col, fontsize=10.5, va="center")
        x = 2.0
        for lec, name in recs:
            hi = lec == 18
            t1 = ax.text(x + 0.06, y, f"L{lec}", color="white" if hi else col, fontsize=10.5, va="center",
                         fontweight="bold", zorder=4)
            w1 = width(t1)
            t2 = ax.text(x + 0.06 + w1 + 0.06, y, name, color="white" if hi else TEXT, fontsize=10.5,
                         va="center", fontweight="bold" if hi else "normal", zorder=4)
            w = w1 + width(t2) + 0.18
            ax.add_patch(FancyBboxPatch((x, y - 0.14), w, 0.28, boxstyle="round,pad=0,rounding_size=0.04",
                                        fc=RED if hi else "white", ec=col, lw=1.5, clip_on=False, zorder=3))
            x += w + 0.07
        y -= 0.38
    sp.save(fig, out("record_chain.svg"))


# ---------------------------------------------------------------------------
# Text-free thumbnails drawn as shapes (goals and takeaways)
# ---------------------------------------------------------------------------
def thumb_register():
    fig, ax = plt.subplots(figsize=(2.0, 1.0))
    for i in range(3):
        for j, col in enumerate((RED, AMBER, NERVE, NAVY)):
            ax.add_patch(Rectangle((j * 1.05, 2 - i), 0.95, 0.8, fc=col, alpha=0.25 + 0.15 * j, ec="none"))
    ax.set_xlim(-0.05, 4.2)
    ax.set_ylim(-0.1, 2.9)
    ax.axis("off")
    sp.save(fig, out("thumb_register.svg"))


def thumb_gaps():
    fig, ax = plt.subplots(figsize=(2.0, 1.0))
    for (x, y), col in zip(((0, 1), (1.1, 1), (0, 0), (1.1, 0)), (RED, AMBER, NAVY, PLANT)):
        ax.add_patch(FancyBboxPatch((x, y), 1.0, 0.85, boxstyle="round,pad=0,rounding_size=0.08", fc="white",
                                    ec=col, lw=2.5))
    ax.set_xlim(-0.1, 2.2)
    ax.set_ylim(-0.1, 1.95)
    ax.axis("off")
    sp.save(fig, out("thumb_gaps.svg"))


def main():
    for k in (1, 2):
        closed_loop(k)
    n_robots, n_ep = oxe()
    print(f"OXE: {n_robots} robots, {n_ep:,.0f} episodes")
    cell_schematic()
    for k in (1, 2, 3, 4):
        audit(k)
    for k in (1, 2):
        cell_stop(k)
    cell_ceilings()
    derate_gap()
    friction_knee()
    trl_friction()
    for k in (1, 2):
        film_terms(k)
    indist()
    overreserve()
    for k in (1, 2):
        responses(k)
    latch_bars()
    last = waymo()
    print(f"Waymo last cut {last['data_cut']:.0f}: {last['ro_miles_millions']} M miles, "
          f"{last['waymo_ipmm']:.2f} against {last['benchmark_ipmm']:.2f}, {last['waymo_count']:.0f} crashes, "
          f"reduction {-100 * last['pct_reduction']:.0f} percent")
    for k in (1, 2, 3):
        budget_final(k)
    record_chain()
    thumb_register()
    thumb_gaps()


if __name__ == "__main__":
    main()
