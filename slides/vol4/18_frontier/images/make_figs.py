"""Figures for Volume IV, Lecture 18 (Engineering Physical AI).

Run:  python3 make_figs.py   (writes SVGs here; `make deck V=4 D=18_frontier`
converts them to PDF). Every value comes from calcs.py, which reads MLSysIM;
labels print the recorded strings (val) so figure and slide text agree.
Figures are drawn at the size of their slot: full width 5.8 in wide with
12 to 13 pt type, a column slot about 5.4 by 3.6 in with 15 to 16 pt type.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "assets" / "kit"))   # slides/assets/kit
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon  # noqa: E402
import numpy as np  # noqa: E402
import slideplot as sp  # noqa: E402
import calcs as c  # noqa: E402
from deckvals import val  # noqa: E402

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
        ("Brain", 3.05, "#EFF6FF", NAVY, f"proposes at {val('intent_hz')} Hz"),
        ("Nervous System", 2.05, NERVE_L, NERVE, f"admits at {val('permission_khz')} kHz"),
        ("Body", 1.45, "#F8FAFC", PLANT, f"current loop {val('current_khz')} kHz"),
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


# Open X-Embodiment is Lecture 5's dataset; this deck recalls it in one line.


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
    ax.text(3.45, 1.32, f"{val('cell_v')} m/s", fontsize=12, ha="center", color=PLANT)
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
    rows = [(f"{v:.2f} m/s", d1, d2) for v, d1, d2 in c.CELL_ROWS]
    fig, ax = plt.subplots(figsize=(5.8, 2.0))
    clear = c.CELL_CLEAR_MM
    res = c.CELL_RESERVE_MM
    ax.add_patch(Rectangle((clear - res, -0.6), res, 2.15, fc="none", ec=GRAY, hatch="///", lw=0))
    ax.text(clear - res - 2, 1.62, "reserve", fontsize=11, color=PLANT, ha="right", va="bottom")
    ax.axvline(clear, color=RED, lw=2, ls="--")
    ax.text(clear + 2, 1.62, f"stop line {clear:.0f} mm", fontsize=11, color=RED, va="bottom")
    for i, (lab, d1, d2) in enumerate(rows):
        y = 1 - i
        a = 1.0 if i < step else 0.0
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
    labels = [(f"Certified curtain,\n{val('cell_tau_curtain')} ms chain", NERVE),
              (f"Processing halved,\n{val('cell_tau_fastproc')} ms chain", NERVE),
              (f"As planned,\n{val('cell_tau')} ms chain", PLANT),
              (f"Brake halved,\n{val('cell_a_half')} m/s²", RED)]
    rows = [(lab, v, col) for (lab, col), (_, v) in zip(labels, c.CEILINGS)]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for i, (lab, v, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, color=col, height=0.6)
        ax.text(0.03, y, f"{v:.2f} m/s", va="center", fontsize=15, fontweight="bold", color="white")
        ax.text(-0.02, y, lab, va="center", ha="right", fontsize=14, color=TEXT, fontweight="bold",
                linespacing=1.0)
    ax.axvline(c.V_MODE, color=PINE, lw=2, ls="--")
    ax.text(c.V_MODE, 3.55, f"{val('cell_v_mode')} m/s mode", color=PINE, fontsize=14, ha="center", va="bottom",
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
    v_nom, v_ceil = c.CELL_ROWS[0][0], c.CEILINGS[2][1]
    ev = np.exp(-0.5 * ((v - v_nom) / 0.12) ** 2)
    rt = np.exp(-0.5 * ((v - 0.45) / 0.06) ** 2)
    ax.fill_between(v, 0, ev, color=NAVY, alpha=0.25)
    ax.plot(v, ev, color=NAVY, lw=2.5)
    ax.fill_between(v, 0, rt, color=RED, alpha=0.2)
    ax.plot(v, rt, color=RED, lw=2.5)
    ax.text(v_nom, 1.08, f"states evaluated\nat {val('cell_v')} m/s", color=NAVY, fontsize=15, ha="center",
            fontweight="bold", va="bottom")
    ax.text(0.02, 1.38, f"states admitted\nat {val('cell_v_mode')} m/s", color=RED, fontsize=15, ha="left",
            fontweight="bold", va="bottom")
    ax.plot([v_ceil, v_ceil], [0, 1.25], color=PINE, lw=2, ls="--")
    ax.text(v_ceil + 0.05, 0.45, f"ceiling\n{val('cell_ceiling')} m/s", color=PINE, fontsize=14, va="center")
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
    mu_min = c.MU_MIN
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    mu = np.linspace(0, 0.7, 300)
    a = np.minimum(c.A_BRAKE, mu * c.G)
    ax.axvspan(0, mu_min, color=RED_L, alpha=0.7, lw=0)
    ax.plot(mu, a, color=AMBER, lw=3)
    ax.axvline(mu_min, color=PLANT, lw=1.5, ls="--")
    ax.text(mu_min / 2, 2.35, "floor binds", color=RED, fontsize=15, ha="center", fontweight="bold")
    ax.text(0.47, 2.35, "brake binds", color=PLANT, fontsize=15, ha="center", fontweight="bold")
    ax.text(mu_min + 0.01, 0.1, f"μ = {val('mu_min')}", color=PLANT, fontsize=14)
    pts = [(f"dry {val('mu_dry')}", c.MU_DRY, c.A_AVAIL["dry"], PLANT, (0.6, 1.62)),
           (f"inspected {val('mu_insp')}", c.MU_INSP, c.A_AVAIL["insp"], AMBER, (0.25, 1.05)),
           (f"film {val('mu_oil')}", c.MU_OIL, c.A_AVAIL["film"], RED, (0.25, 0.4))]
    for lab, m, y, col, xy in pts:
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
    mu_min = c.MU_MIN
    rows = c.TRL_ROWS
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    for i, (lab, mu) in enumerate(rows):
        y = len(rows) - 1 - i
        col = RED if mu < mu_min else (AMBER if "smooth" in lab else PLANT)
        ax.barh(y, mu, color=col, height=0.62)
        ax.text(mu + 0.012, y, f"{mu:.2f}", va="center", fontsize=12, fontweight="bold", color=col)
        ax.text(-0.012, y, lab, va="center", ha="right", fontsize=11.5, color=TEXT, fontweight="bold")
    ax.axvline(mu_min, color=RED, lw=1.8, ls="--")
    ax.text(mu_min + 0.01, 5.55, f"floor binds below {val('mu_min')}", color=RED, fontsize=11,
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
    for i, (lab, row) in enumerate((("dry floor", "dry"), ("film", "film"))):
        y = 1 - i
        a = 1.0 if i < step else 0.0
        x = 0
        for (key, col), v in zip(names, c.TERMS[row]):
            ax.barh(y, v, left=x, color=col, height=0.55, alpha=a, edgecolor="white", lw=1)
            if v > 300:
                ax.text(x + min(v / 2, 230), y, f"{v:,.1f}", color="white", ha="center", va="center", fontsize=11,
                        fontweight="bold", alpha=a)
            x += v
        ax.text(-20, y, lab, ha="right", va="center", fontsize=12, fontweight="bold", color=TEXT, alpha=a)
        tot = x / 1000
        total = val("budget") + " mm" if row == "dry" else val("film_m") + " m"
        ax.text(max(x, c.D_CLEAR_MM) + 30, y, total, va="center",
                fontsize=12, fontweight="bold", color=RED if x > c.D_CLEAR_MM else PLANT, alpha=a)
    ax.axvline(c.D_CLEAR_MM, color=RED, lw=2, ls="--")
    ax.text(c.D_CLEAR_MM + 15, 1.5, f"clear distance {val('d_clear_mm')} mm", color=RED, fontsize=11,
            fontweight="bold", va="bottom")
    legend = f"delay {val('reaction')}   ramp {val('ramp')}   overheads {val('allowances')} mm"
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
    t_stop, t_eq, spare_ms = c.T_STOP_MS, c.T_EQ_MS, c.SPARE_MS
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    ax.barh(2, t_eq, color=PLANT, height=0.55)
    ax.barh(2, t_stop - t_eq, left=t_eq, color=RED, height=0.55)
    ax.text(t_eq / 2, 2, val("t_eq"), color="white", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(t_eq + (t_stop - t_eq) / 2, 2, f"+{val('t_over')} over", color="white", ha="center",
            va="center", fontsize=12, fontweight="bold")
    ax.barh(1, t_eq, color=PLANT, height=0.55)
    ax.text(t_eq + 10, 1, f"{val('t_eq')} ms", va="center", fontsize=12, fontweight="bold", color=PLANT)
    ax.barh(0, spare_ms, color=GRAY, height=0.55)
    ax.text(spare_ms + 10, 0, f"{val('t_spare')} ms", va="center", fontsize=12, fontweight="bold", color=PLANT)
    for y, lab in ((2, f"stop time {val('t_stop')} ms"), (1, "distance / speed"), (0, "the whole spare")):
        ax.text(-10, y, lab, ha="right", va="center", fontsize=12, fontweight="bold", color=TEXT)
    ax.set_xlim(0, 1000)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.set_xlabel(f"ms at {val('v_aisle')} m/s", fontsize=11)
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
    names = ("rigid plate", "rigid plate", "series spring")
    rows = [(f"{n},\n{v:.2f} m/s", f) for n, (v, f) in zip(names, c.LATCH_BARS)]
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
    ax.text(c.F_FLOOR + 5, 2.55, f"{val('f_floor')} N declared floor", color=RED, fontsize=14, ha="left", va="bottom",
            fontweight="bold")
    ax.set_xlim(0, 290)
    ax.set_ylim(-0.5, 2.95)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=13)
    ax.set_xlabel(f"Peak latch force, torque removed at {val('f_trip')} N", fontsize=13)
    sp.save(fig, out("latch_bars.svg"))


# The Waymo crash rates are Lecture 16's data; this deck recalls them in one line.


# The aisle stopping budget is the course kit's bar (budget_bar_aisle_step7 and
# step8); thumb_budget_bar_aisle_step8.svg is the kit bar run through
# tools/greek_thumbs.py.


# The record chain is the course kit's (assets/kit/vol4/record_chain_l18).


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
    thumb_register()
    thumb_gaps()


if __name__ == "__main__":
    main()
