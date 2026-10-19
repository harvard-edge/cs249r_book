#!/usr/bin/env python3
"""Figures for Lecture 15 (Adversarial Verification), drawn at the size of their slots.

Every derived value comes from calcs.py; inputs carry their source there.
Published measurements (DriveFI, Wang et al., VNN-COMP, O'Kelly et al.) are
typed in calcs.py with the table they were read from. Plots use the kit's
slideplot helper settings (large type, direct labels). Colors follow the
Volume IV palette: plant slate, hazard red, latency amber, sensing teal,
deterministic green, proposal blue; nothing uses the accent.

Run from this folder: python3 make_figs.py   (then `make deck V=4 D=15_verification`)
"""

import contextlib
import io
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import slideplot as sp  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from slideplot import plt  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    import calcs as c  # noqa: E402

SLATE = "#334155"
SLATE_L = "#CBD5E1"
RED = "#DC2626"
RED_L = "#FEE2E2"
AMBER = "#D97706"
AMBER_L = "#FDE7C4"
TEAL = "#0D9488"
GREEN = "#059669"
GREEN_L = "#DCFCE7"
BLUE = "#2563EB"
BLUE_L = "#DBEAFE"
NAVY = "#1F407A"
GRAY = "#888888"
TEXT = "#1A1A2E"


def out(name):
    return str(HERE / f"{name}.svg")


# The recall thumbnail is the kit's stopping-budget bar (budget_bar_aisle_step8)
# run through tools/greek_thumbs.py: thumb_budget_bar_aisle_step8.svg.


# -----------------------------------------------------------------------------
# Segment 1: speed ceiling against floor friction (AV 15.5.1)
# -----------------------------------------------------------------------------
def corner():
    mus = np.linspace(0.08, 0.40, 200)
    ceil = np.array([c.ceiling(m) for m in mus])
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(3.3, 2.2))
        ax.plot(mus, ceil, color=SLATE, lw=3)
        ax.axhline(c.V_AISLE, color=BLUE, lw=2.2, ls="--")
        ax.text(0.395, c.V_AISLE + 0.012, "aisle speed 1.3 m/s", ha="right", va="bottom",
                color=BLUE, fontsize=10.5, fontweight="bold")
        ax.text(0.395, c.ceiling(0.3) + 0.012, "dry-floor ceiling 1.44 m/s", ha="right",
                va="bottom", color=SLATE, fontsize=10)
        ax.axvline(c.mu_knee, color=GRAY, lw=1.2, ls=":")
        ax.text(c.mu_knee + 0.006, 1.595, "traction binds below 0.204", fontsize=9.5,
                color=GRAY, va="top")
        if step >= 2:
            bad = ceil < c.V_AISLE
            ax.fill_between(mus, ceil, c.V_AISLE, where=bad, color=RED_L, alpha=0.9)
            ax.plot([c.MU_INSPECTED], [c.ceiling(c.MU_INSPECTED)], "o", color=RED, ms=9)
            ax.annotate("", xy=(c.MU_INSPECTED + 0.006, c.ceiling(c.MU_INSPECTED)), xytext=(0.212, 1.2),
                        arrowprops=dict(arrowstyle="-", color=RED, lw=1.2))
            ax.text(0.215, 1.22,
                    "inspected floor 0.12:\nceiling 1.23 m/s", color=RED, fontsize=10.5,
                    fontweight="bold", va="top")
        ax.set_xlim(0.08, 0.40)
        ax.set_ylim(1.0, 1.6)
        ax.set_xlabel("Floor friction coefficient")
        ax.set_ylabel("Speed ceiling (m/s)")
        sp.save(fig, out(f"corner_step{step}"))


def drivefi():
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    rows = [("Every fault,\nexhaustive", c.DFI_EXHAUSTIVE_DAYS * 24, SLATE,
             f"{c.DFI_FAULTS:,} faults: 615 days"),
            ("Bayesian\nsearch", c.DFI_BAYES_HOURS, RED,
             f"{c.DFI_FOUND} faults in < 4 h; {c.DFI_HAZARD} hazards")]
    for i, (lab, v, col, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=col, height=0.5)
        ax.text(-0.15, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=12, fontweight="bold", color=TEXT)
        ax.text(v * 1.3 if i == 1 else 1.2, y + (0 if i == 1 else 0.42), txt,
                va="center", fontsize=12, color=col if i == 1 else TEXT, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(1, 3e4)
    ax.set_ylim(-0.6, 1.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Hours of simulated injection (log)")
    sp.save(fig, out("drivefi"))


# -----------------------------------------------------------------------------
# Segment 2: three reference events, three bounds (AV figure 1, as bars in mm)
# -----------------------------------------------------------------------------
def refevents():
    v = c.V_AISLE
    cap_blind = v * c.TAU_PREBRAKE * 1e3
    stop = c.suffix(v) * 1e3
    lease = v * c.onset_bound() * 1e3
    rows = [("Capture-\nreferenced", [(cap_blind, AMBER, f"{cap_blind:.1f}"),
                                       (stop, SLATE, f"stop {stop:.1f}"),
                                       ((c.LOC + c.MARGIN) * 1e3, SLATE_L, "150"),
                                       (c.TRACK * 1e3, GREEN_L, "")],
             f"{c.capture_subtotal(v)*1e3:.1f}"),
            ("Renewal-\nreferenced (F1)", [(lease, AMBER, f"{lease:.1f}"),
                                           (stop, SLATE, f"stop {stop:.1f}"),
                                           (c.TRACK * 1e3, GREEN_L, "")],
             f"{c.f1_bound(v)*1e3:.1f}"),
            ("Appearance-\nreferenced", None, "no value yet")]
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.25))
        for i, (lab, segs, total) in enumerate(rows):
            y = 2 - i
            al = 1.0 if i < step else 0.0
            ax.text(-25, y, lab, ha="right", va="center", fontsize=11.5,
                    fontweight="bold", color=TEXT, alpha=al)
            if segs is None:
                ax.barh(y, 1000, color="white", height=0.55, edgecolor=GRAY, hatch="//",
                        alpha=al, lw=1.2)
                ax.text(500, y, "needs witnessed appearance-to-capture\nand dispatch-to-admission intervals",
                        ha="center", va="center", fontsize=11, color=TEXT, alpha=al,
                        bbox=dict(fc="white", ec="none", alpha=0.9 * al))
                continue
            x = 0
            for w, col, txt in segs:
                ax.barh(y, w, left=x, color=col, height=0.55, alpha=al, edgecolor="white")
                if txt and w > 130:
                    ax.text(x + w / 2, y, txt, ha="center", va="center", fontsize=11,
                            color="white" if col in (SLATE, AMBER) else TEXT, alpha=al,
                            fontweight="bold")
                x += w
            ax.text(1125, y, f"{total} mm", va="center", fontsize=12, fontweight="bold",
                    color=BLUE if i == 1 else TEXT, alpha=al)
        if step >= 2:
            g = (c.capture_subtotal(v) - c.f1_bound(v)) * 1e3
            ax.annotate("", xy=(c.f1_bound(v) * 1e3, 1.5), xytext=(c.capture_subtotal(v) * 1e3, 1.5),
                        arrowprops=dict(arrowstyle="<->", color=RED, lw=1.8))
            ax.text(c.f1_bound(v) * 1e3 - 12, 1.5, f"{g:.1f} mm",
                    ha="right", va="center", color=RED, fontsize=12, fontweight="bold")
        ax.axvline(c.D_CLEAR * 1e3, color=RED, lw=2, ls="--")
        ax.text(c.D_CLEAR * 1e3 + 15, 2.42, "clear 1,100 mm", ha="left", va="bottom",
                color=RED, fontsize=11.5, fontweight="bold")
        ax.set_xlim(0, 1330)
        ax.set_ylim(-0.45, 2.75)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Travel at 1.3 m/s from the reference event (mm)")
        sp.save(fig, out(f"refevents_step{step}"))


# -----------------------------------------------------------------------------
# Segment 2: one F1 trace against its oracle (AV figure 4, redrawn)
# -----------------------------------------------------------------------------
def speed_profile(t_on, t):
    """Speed after an S-curve stop starting at t_on (s): jerk-limited ramp, then constant a."""
    v0, a, j = c.V_AISLE, c.A_BRAKE, c.J_MAX
    tr = a / j
    v = np.full_like(t, v0)
    s = t - t_on
    ramp = (s > 0) & (s <= tr)
    v[ramp] = v0 - 0.5 * j * s[ramp] ** 2
    v1 = v0 - 0.5 * j * tr ** 2
    lin = s > tr
    v[lin] = np.maximum(v1 - a * (s[lin] - tr), 0)
    return v


def f1trace():
    on = c.onset_bound()
    for step in (1, 2):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.8, 2.75),
                                       gridspec_kw=dict(height_ratios=[1.25, 1], hspace=0.55))
        chans = [("lease", 0, 60, TEAL), ("stop cmd", 61, 200, GREEN),
                 ("drive frame", 62, 200, GREEN), ("decel. onset", on * 1e3, 200, GREEN)]
        for i, (lab, t1, t2, col) in enumerate(chans):
            y = 3 - i
            if lab == "lease":
                ax1.plot([0, 60, 60, 130], [y + 0.35, y + 0.35, y, y], color=col, lw=2.5)
            else:
                ax1.plot([0, t1, t1, 130], [y, y, y + 0.35, y + 0.35], color=col, lw=2.5)
            ax1.text(-3, y + 0.15, lab, ha="right", va="center", fontsize=11, color=TEXT)
        ax1.axvspan(62, on * 1e3, color=GREEN_L, alpha=0.8, lw=0)
        ax1.axvline(on * 1e3, color=GREEN, lw=1.2, ls=":")
        ax1.text(on * 1e3 + 2, 4.0, "82 ms bound", ha="left", va="bottom", color=GREEN,
                 fontsize=11, fontweight="bold")
        if step >= 2:
            ax1.plot([0, 112, 112, 130], [-0.05, -0.05, 0.3, 0.3], color=RED, lw=2.5, ls="--")
            ax1.annotate("", xy=(82, -0.5), xytext=(112, -0.5),
                         arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
            ax1.text(114, -0.5, "-30 ms", ha="left", va="center", color=RED, fontsize=11,
                     fontweight="bold")
        ax1.set_xlim(0, 130)
        ax1.set_ylim(-0.85, 4.5)
        ax1.set_yticks([])
        ax1.spines["left"].set_visible(False)
        ax1.set_xlabel("Time after the last valid renewal (ms)", fontsize=11)
        t = np.linspace(0, 1.2, 600)
        ax2.plot(t * 1e3, speed_profile(on, t), color=GREEN, lw=2.5)
        ax2.text(1195, 1.28, f"on time: {(c.f1_bound(c.V_AISLE)-c.TRACK)*1e3:.1f} mm, rest at 1,013 ms",
                 fontsize=11, color=GREEN, fontweight="bold", ha="right", va="center")
        if step >= 2:
            ax2.plot(t * 1e3, speed_profile(c.ON_LATE, t), color=RED, lw=2.5, ls="--")
            ax2.text(1195, 1.0, f"late: {c.late_travel*1e3:.1f} mm, margin +1 mm",
                     fontsize=11, color=RED, fontweight="bold", ha="right", va="center")
        ax2.set_xlim(0, 1200)
        ax2.set_ylim(0, 1.45)
        ax2.set_ylabel("m/s", fontsize=11)
        ax2.set_xlabel("Time after the last valid renewal (ms)", fontsize=11)
        sp.save(fig, out(f"f1trace_step{step}"))


# -----------------------------------------------------------------------------
# Segment 3: a patch in view (Wang et al. 2025, Table 1a)
# -----------------------------------------------------------------------------
def wang():
    fig, ax = plt.subplots(figsize=(3.1, 2.25))
    x = np.arange(4)
    ben = [100 - f for f in c.WANG_BENIGN_FR]
    pat = [100 - f for f in c.WANG_UADA_FR]
    ax.bar(x - 0.2, ben, width=0.38, color=BLUE)
    ax.bar(x + 0.2, [max(p, 1.2) for p in pat], width=0.38, color=RED)
    for xi, b in zip(x, ben):
        ax.text(xi - 0.2, b + 2, f"{b:.0f}", ha="center", fontsize=9.5, color=BLUE, fontweight="bold")
        ax.text(xi + 0.2, 4, "0", ha="center", fontsize=9.5, color=RED, fontweight="bold")
    ax.text(3.55, 118, "no patch", color=BLUE, fontsize=10.5, fontweight="bold", ha="right")
    ax.text(3.55, 106, "patch in view", color=RED, fontsize=10.5, fontweight="bold", ha="right")
    ax.set_xticks(x, c.WANG_SUITES)
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel("Task success (%)")
    ax.set_xlabel("LIBERO task suite (simulated)")
    sp.save(fig, out("wang_patch"))


# -----------------------------------------------------------------------------
# Segment 4: the qualification ladder (AV figure 2, redrawn)
# -----------------------------------------------------------------------------
LADDER_COLS = ["Decision\nlogic", "CPU\ntiming", "Bus\ntiming",
               "Supply\nsag", "Drives,\nbrakes", "Floor\ncontact"]
LADDER_ROWS = [("SIL", "R---MM", "sweeps"),
               ("PIL", "RRM-MM", "target-bound"),
               ("HIL", "RRRRRM", "real time"),
               ("Physical rig", "RRRRRR", "wear-limited")]


def ladder():
    cell_w, cell_h = 1.0, 0.62
    for step in (1, 2, 3, 4):
        fig, ax = plt.subplots(figsize=(5.8, 2.55))
        for j, lab in enumerate(LADDER_COLS):
            ax.text(j * cell_w + 0.5, 4 * cell_h + 0.08, lab, ha="center", va="bottom",
                    fontsize=10, color=TEXT, fontweight="bold")
        ax.text(6 * cell_w + 0.15, 4 * cell_h + 0.08, "Throughput", ha="left", va="bottom",
                fontsize=10.5, color=TEXT, fontweight="bold")
        for i, (name, pattern, thr) in enumerate(LADDER_ROWS):
            y = (3 - i) * cell_h
            shown = i < step
            ax.text(-0.12, y + cell_h / 2, name, ha="right", va="center", fontsize=11,
                    color=TEXT if shown else "#EEEEEE", fontweight="bold")
            for j, ch in enumerate(pattern):
                x = j * cell_w
                if not shown:
                    ax.add_patch(Rectangle((x + 0.05, y + 0.05), cell_w - 0.1, cell_h - 0.1,
                                           fc="white", ec="#EEEEEE", lw=1))
                    continue
                if ch == "R":
                    ax.add_patch(Rectangle((x + 0.05, y + 0.05), cell_w - 0.1, cell_h - 0.1,
                                           fc=GREEN, ec=GREEN, lw=1.5))
                elif ch == "M":
                    ax.add_patch(Rectangle((x + 0.05, y + 0.05), cell_w - 0.1, cell_h - 0.1,
                                           fc="white", ec=SLATE, lw=1.6, ls="--"))
                    ax.text(x + 0.5, y + cell_h / 2, "model", ha="center", va="center",
                            fontsize=10, color=SLATE)
                else:
                    ax.add_patch(Rectangle((x + 0.05, y + 0.05), cell_w - 0.1, cell_h - 0.1,
                                           fc="white", ec="#DDDDDD", lw=1))
                    ax.text(x + 0.5, y + cell_h / 2, "absent", ha="center", va="center",
                            fontsize=10, color=GRAY)
            if shown:
                ax.text(6 * cell_w + 0.15, y + cell_h / 2, thr, ha="left", va="center",
                        fontsize=10.5, color=TEXT)
        ax.set_xlim(-1.2, 7.6)
        ax.set_ylim(-0.05, 4 * cell_h + 0.75)
        ax.axis("off")
        sp.save(fig, out(f"ladder_step{step}"))


# -----------------------------------------------------------------------------
# Segment 4: what a proof can reach (VNN-COMP 2025, Table 3)
# -----------------------------------------------------------------------------
def vnn():
    fig, ax = plt.subplots(figsize=(3.5, 2.15))
    vals = sorted(c.VNN_MAX_PARAMS.values())
    rng = np.random.default_rng(3)
    ys = rng.uniform(-0.25, 0.25, len(vals))
    ax.scatter(vals, ys, s=60, color=SLATE, alpha=0.8, zorder=3)
    big = max(vals)
    ax.annotate("VGGNet16, 138M", xy=(big, ys[vals.index(big)]), xytext=(3e3, 0.72),
                fontsize=9.5, color=SLATE, arrowprops=dict(arrowstyle="-", color=SLATE))
    ax.axvline(c.CHUNK_PARAMS, color=BLUE, lw=2, ls="--")
    ax.text(c.CHUNK_PARAMS / 1.4, -0.75, "chunk policy\n80M", color=BLUE, fontsize=9.5,
            fontweight="bold", ha="right", va="bottom")
    ax.axvline(c.INTENT_PARAMS, color=RED, lw=2.5)
    ax.text(c.INTENT_PARAMS * 1.6, 0.55, "intent\nmodel\n7B", color=RED, fontsize=9.5,
            fontweight="bold", ha="left")
    ax.set_xscale("log")
    ax.set_xlim(100, 1e12)
    ax.set_ylim(-0.85, 1.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Parameters (largest network per scored benchmark)")
    sp.save(fig, out("vnn"))


# -----------------------------------------------------------------------------
# Segment 4: signed robustness of two clearance traces (AV 15.9.1, illustrative)
# -----------------------------------------------------------------------------
def stl():
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(3.2, 2.25))
        t = np.linspace(0, 2.0, 400)
        a = 0.15 + 0.85 * (t - 1.0) ** 2 / 1.0
        b = -0.04 + 0.95 * (t - 1.15) ** 2 / 1.0
        ax.axhline(0, color=RED, lw=2)
        ax.text(0.03, -0.02, "clearance limit", color=RED, fontsize=9.5, ha="left", va="top")
        ax.plot(t, a, color=GREEN, lw=2.5)
        ax.annotate("", xy=(1.0, 0), xytext=(1.0, 0.15),
                    arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.6))
        ax.text(0.0, 1.13, r"solid: $\rho = +0.15$ m, passes", color=GREEN, fontsize=10.5,
                ha="left", va="center", fontweight="bold", transform=ax.transAxes)
        if step >= 2:
            ax.plot(t, b, color=RED, lw=2.5, ls="--")
            ax.annotate("", xy=(1.15, 0), xytext=(1.15, -0.04),
                        arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
            ax.text(0.0, 1.04, r"dashed: $\rho = -0.04$ m, fails", color=RED, fontsize=10.5,
                    ha="left", va="center", fontweight="bold", transform=ax.transAxes)
        ax.set_xlim(0, 2.0)
        ax.set_ylim(-0.15, 0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Clearance margin (m)")
        sp.save(fig, out(f"stl_step{step}"))


# -----------------------------------------------------------------------------
# Segment 5: runs to estimate one rare rate (AV 15.10.2)
# -----------------------------------------------------------------------------
def is_ladder():
    rows = [("Plain Monte Carlo", c.n_mc, SLATE, f"{c.n_mc:.1e}".replace("e+10", r"$\times10^{10}$")),
            ("Zero-failure campaign", 3 / c.p, AMBER, f"{3/c.p:.1e}".replace("e+09", r"$\times10^{9}$")),
            ("Importance sampling", c.n_is, GREEN, f"{c.n_is:,.0f}")]
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(4.8, 1.8))
        for i, (lab, v, col, txt) in enumerate(rows):
            y = 2 - i
            al = 1.0 if i < step else 0.0
            ax.barh(y, v, color=col, height=0.55, alpha=al)
            ax.text(v * 1.6, y, txt, va="center", fontsize=12, fontweight="bold", color=col, alpha=al)
            ax.text(-0.02, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=11.5, color=TEXT, fontweight="bold", alpha=al)
        ax.set_xscale("log")
        ax.set_xlim(1, 5e12)
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Runs for 10 percent relative error (log)")
        sp.save(fig, out(f"is_ladder_step{step}"))


def okelly():
    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    x = np.arange(len(c.OK_GAMMA))
    cols = {"Naive MC, 100K": SLATE_L, "Cross-entropy, 100K": GREEN, "Naive MC, 1.3M runs": SLATE}
    order = ["Naive MC, 100K", "Cross-entropy, 100K", "Naive MC, 1.3M runs"]
    for k, name in enumerate(order):
        vals = c.OK_RSE[name]
        ax.bar(x + (k - 1) * 0.27, vals, width=0.26, color=cols[name], edgecolor=SLATE, lw=0.6)
    ax.text(3.45, 66, "naive, 100K runs", color=SLATE, fontsize=9.5, ha="right")
    ax.text(3.45, 56, "cross-entropy, 100K", color=GREEN, fontsize=9.5, ha="right", fontweight="bold")
    ax.text(3.45, 46, "naive, 1.3M runs", color=SLATE, fontsize=9.5, ha="right", fontweight="bold")
    ax.set_xticks(x, [f"{g:.2f}" for g in c.OK_GAMMA])
    ax.set_xlabel(r"Danger threshold $\gamma$ (rarer to the left)")
    ax.set_ylabel("Relative std. error (%)")
    ax.set_ylim(0, 75)
    sp.save(fig, out("okelly"))


def evt():
    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    y = np.linspace(0, 60, 300)
    sig = 10.0
    for xi, col, lab in [(0.3, RED, r"$\xi>0$: heavy tail"), (0.0, SLATE, r"$\xi=0$: exponential"),
                         (-0.3, GREEN, r"$\xi<0$: endpoint")]:
        if xi == 0:
            s = np.exp(-y / sig)
        else:
            base = 1 + xi * y / sig
            s = np.where(base > 0, np.abs(base) ** (-1 / xi), 0.0)
        ax.semilogy(y, np.maximum(s, 1e-6), color=col, lw=2.5)
    ax.text(42, 0.15, r"$\xi>0$: heavy tail", color=RED, fontsize=10.0, fontweight="bold", va="bottom")
    ax.text(37, 2e-3, r"$\xi=0$: exponential", color=SLATE, fontsize=10.0, fontweight="bold", va="top")
    end = -sig / -0.3
    ax.axvline(end, color=GREEN, ls=":", lw=1.5)
    ax.text(end + 1, 1e-5, "fitted endpoint", color=GREEN, fontsize=10.0, fontweight="bold",
            ha="left", va="bottom")
    ax.set_ylim(1e-6, 1.5)
    ax.set_xlim(0, 60)
    ax.set_xlabel("Overshoot past the threshold (mm, illustrative)")
    ax.set_ylabel("P(exceed)")
    sp.save(fig, out("evt"))


# The record chain is the course kit's (assets/kit/vol4/record_chain_l15 and
# thumb_record_chain).


SMALL = {"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5}


def main():
    with plt.rc_context(SMALL):
        corner()
        wang()
        vnn()
        stl()
        okelly()
        evt()
    drivefi()
    refevents()
    f1trace()
    ladder()
    is_ladder()


if __name__ == "__main__":
    main()
