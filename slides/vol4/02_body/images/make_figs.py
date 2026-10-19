#!/usr/bin/env python3
"""Figures for Lecture 2 (The Physical Body), drawn at the size of their slots.

Every derived value comes from calcs.py; inputs carry their source there.
Plots use the kit's slideplot helper (large type, direct labels). Colors follow
the Volume IV palette: plant slate, hazard red, latency amber, sensing teal,
deterministic green; nothing uses the accent.

Run from this folder: python3 make_figs.py   (then `make deck V=4 D=02_body`)
"""

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import slideplot as sp  # noqa: E402
from slideplot import plt  # noqa: E402

import calcs as c  # noqa: E402

# Volume IV palette (rules/volumes/vol4/palette.md)
SLATE = "#334155"
SLATE_L = "#CBD5E1"
RED = "#DC2626"
RED_L = "#FEE2E2"
AMBER = "#D97706"
AMBER_L = "#FDE7C4"
TEAL = "#0D9488"
GREEN = "#059669"
GREEN_L = "#DCFCE7"
GRAY = "#888888"
TEXT = "#1A1A2E"
COL = (5.0, 3.3)          # column figure beside a photo or text


def out(name):
    return str(HERE / f"{name}.svg")


# -----------------------------------------------------------------------------
# Freshness: where the age of one grasp observation goes (table 1, illustrative)
# -----------------------------------------------------------------------------
def waterfall():
    stages = c.STAGES_P50
    colors = [TEAL, TEAL, AMBER, SLATE, SLATE, SLATE]
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.4))
        x = 0.0
        for i, ((name, v), col) in enumerate(zip(stages, colors)):
            shown = step >= 2 or name.startswith("Backbone")
            ax.barh(1, v, left=x, color=col, height=0.5, edgecolor="white",
                    lw=1.5, alpha=1.0 if shown else 0.0)
            x += v
        # inference only, on its own row in step 1 for the benchmark view
        ax.barh(0, 18.5, left=0, color=AMBER, height=0.5)
        ax.text(19.5, 0, "benchmark: 18.5 ms", va="center",
                fontsize=12, color=AMBER, fontweight="bold")
        ax.text(-1, 1, "Capture to\ntorque onset", ha="right", va="center",
                fontsize=12, color=TEXT, fontweight="bold", alpha=1 if step >= 2 else 0)
        ax.text(-1, 0, "Model only", ha="right", va="center", fontsize=12,
                color=TEXT, fontweight="bold")
        ax.text(c.p50_sum, 1.45, f"median {c.p50_sum:.0f}", ha="center", va="bottom",
                fontsize=11.5, color=SLATE, fontweight="bold", alpha=1 if step >= 2 else 0)
        # labels for the non-inference stages above the bar
        ax.text(11.2 / 2, 1.45, "exposure,\nreadout", ha="center", va="bottom",
                fontsize=11, color=TEAL, alpha=1 if step >= 2 else 0)
        a3 = 1 if step >= 3 else 0
        ax.axvline(c.E_DEADLINE, color=GREEN, lw=2, ls="--", alpha=a3)
        ax.text(c.E_DEADLINE - 0.8, -0.55, f"deadline {c.E_DEADLINE:.0f} ms",
                color=GREEN, fontsize=11, fontweight="bold", ha="right", alpha=a3)
        ax.plot([c.P99_E2E], [1], marker="|", ms=26, mew=2.5, color=SLATE, alpha=a3)
        ax.text(c.P99_E2E, 0.72, f"p99 {c.P99_E2E}", ha="center", va="top",
                fontsize=11, color=SLATE, alpha=a3)
        ax.plot([c.P999_E2E], [1], marker="|", ms=26, mew=2.5, color=RED, alpha=a3)
        ax.text(c.P999_E2E, 0.72, f"p99.9 {c.P999_E2E}", ha="center", va="top",
                fontsize=11, color=RED, fontweight="bold", alpha=a3)
        ax.set_xlim(0, 75)
        ax.set_ylim(-0.75, 2.2)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Age of the observation (ms)")
        sp.save(fig, out(f"waterfall_step{step}"))


def mug_travel():
    rows = [("Median\n35 ms", c.mug_p50, SLATE, f"{c.mug_p50:.0f} mm"),
            ("p99.9\n68.4 ms", c.mug_tail, RED, f"{c.mug_tail:.1f} mm")]
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(3.7, 1.9))
        for i, (lab, v, col, txt) in enumerate(rows):
            y = 1 - i
            al = 1.0 if i < step else 0.0
            ax.barh(y, v, color=col, height=0.6, alpha=al)
            ax.text(v + 0.3, y, txt, va="center", fontsize=12, fontweight="bold",
                    color=col, alpha=al)
            ax.text(-0.4, y, lab, ha="right", va="center", fontsize=12,
                    fontweight="bold", color=TEXT)
        ax.axvline(c.e_max, color=GREEN, lw=2, ls="--")
        ax.text(c.e_max, 1.55, f"allowed {c.e_max:.0f} mm", color=GREEN, fontsize=11.5,
                fontweight="bold", ha="center")
        ax.set_xlim(0, 17)
        ax.set_ylim(-0.5, 1.8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Mug travel (mm)", fontsize=12)
        sp.save(fig, out(f"mug_travel_step{step}"))


# -----------------------------------------------------------------------------
# Stopping: three cases against the clear distance (PB figure 2)
# -----------------------------------------------------------------------------
def stop_bars():
    cases = [("Body terms\n1.5 m/s", c.onset, c.brake, c.overhead),
             ("Elementary delay\n1.5 m/s", c.blind_elem, c.brake, c.overhead),
             ("Same delay\n2 m/s", c.blind_fast, c.brake_fast, c.overhead)]
    clear = c.D_CLEAR * 1000
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.5))
        for i, (name, blind, brk, ovh) in enumerate(cases):
            y = 2 - i
            a = 1.0 if i < step else 0.0
            ax.barh(y, blind, color=AMBER, height=0.58, alpha=a)
            ax.barh(y, brk, left=blind, color=SLATE, height=0.58, alpha=a)
            ax.barh(y, ovh, left=blind + brk, color=SLATE_L, height=0.58, alpha=a)
            total = blind + brk + ovh
            col = RED if total > clear else TEXT
            ax.text(1470, y, f"{total:,.1f} mm", va="center", fontsize=12,
                    fontweight="bold", color=col, alpha=a)
            ax.text(-25, y, name, ha="right", va="center", fontsize=11.5,
                    color=TEXT, fontweight="bold", alpha=a)
        ax.axvline(clear, color=RED, lw=2, ls="--")
        ax.text(clear + 10, 2.62, "person at 1,100 mm", color=RED, fontsize=11,
                fontweight="bold")
        # legend as direct labels on the first bar
        ax.text(c.onset + c.brake / 2, 2, "braking", ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
        ax.text(c.blind_elem / 2, 1, "blind", ha="center", va="center",
                color="white", fontsize=11, fontweight="bold", alpha=1 if step >= 2 else 0)
        ax.text(c.onset + c.brake + c.overhead / 2, 2, "fixed", ha="center",
                va="center", color=TEXT, fontsize=10.5)
        ax.set_xlim(0, 1460)
        ax.set_ylim(-0.5, 2.9)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Distance from detection (mm)")
        sp.save(fig, out(f"stop_bars_step{step}"))


def vceiling():
    d = np.linspace(0.15, 2.0, 300)
    curves = [(c.T_ACT, c.A_BRAKE, SLATE, "body terms only", 2.78),
              (c.TAU_ELEM, c.A_BRAKE, AMBER, "133.6 ms delay", 2.22),
              (c.TAU_ELEM, c.MU_INSPECTED * c.G, RED, "inspected floor", 1.66)]
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.5))
        for k, (tau, a, col, lab, ylab) in enumerate(curves):
            vs = [c.v_ceiling(x, tau, a) for x in d]
            alpha = 1.0 if k < step else 0.0
            ax.plot(d, vs, color=col, lw=2.6, alpha=alpha)
            v11 = c.v_ceiling(c.D_CLEAR, tau, a)
            ax.plot([c.D_CLEAR], [v11], "o", ms=8, color=col, alpha=alpha)
            ax.text(2.04, ylab, f"{lab}:\n{v11:.2f} m/s at 1.1 m", va="center",
                    fontsize=11, color=col, fontweight="bold", alpha=alpha,
                    linespacing=0.95)
        ax.axhline(c.V_DRIVE, color=GRAY, lw=1.2, ls=":")
        ax.text(1.35, c.V_DRIVE - 0.08, "drive limit 1.5 m/s", fontsize=10.5,
                color=GRAY, va="top")
        ax.axvline(c.D_CLEAR, color=GRAY, lw=1, ls="--")
        ax.set_xlim(0, 2.0)
        ax.set_ylim(0, 2.9)
        fig.subplots_adjust(right=0.68)
        ax.set_xlabel("Clear distance (m)")
        ax.set_ylabel("Speed ceiling (m/s)")
        sp.save(fig, out(f"vceiling_step{step}"))


def decel_ceilings():
    rows = [("Dry floor", c.a_dry, AMBER), ("Peak current", c.a_torque, AMBER),
            ("Inspected floor", c.a_inspected, RED), ("Rated current", c.a_rated, RED)]
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    n = len(rows) + 1
    for i, (lab, v, col) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, v, color=col, height=0.6)
        ax.text(v + 0.06, y, f"{v:.2f}", va="center", fontsize=12.5,
                fontweight="bold", color=col)
        ax.text(-0.06, y, lab, ha="right", va="center", fontsize=12.5,
                fontweight="bold", color=TEXT)
    ax.text(-0.06, 0, "Tip-over", ha="right", va="center", fontsize=12.5,
            fontweight="bold", color=TEXT)
    ax.text(0.08, 0, "unmeasured", va="center", fontsize=12.5, color=GRAY,
            style="italic")
    ax.axvline(c.A_BRAKE, color=TEXT, lw=1.8, ls="--")
    ax.text(c.A_BRAKE + 0.05, n - 0.35, "needed 2.0", fontsize=11.5,
            fontweight="bold", color=TEXT)
    ax.set_xlim(0, 5.6)
    ax.set_ylim(-0.5, n)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Deceleration ceiling (m/s²)")
    sp.save(fig, out("decel_ceilings"))


# -----------------------------------------------------------------------------
# Actuators
# -----------------------------------------------------------------------------
def volt_need():
    rows = [("Braking", c.e_bemf - c.I_PEAK * c.R_LINE, GREEN, "10.7 V"),
            ("Driving", c.e_bemf + c.I_PEAK * c.R_LINE, RED, "45.7 V")]
    fig, ax = plt.subplots(figsize=(3.2, 1.6))
    for i, (lab, v, col, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=col, height=0.6)
        ax.text(v + 1, y, txt, va="center", fontsize=12, fontweight="bold", color=col)
        ax.text(-1, y, lab, ha="right", va="center", fontsize=12, fontweight="bold", color=TEXT)
    ax.axvline(c.V_DEPLETED, color=GRAY, lw=1.8, ls="--")
    ax.text(c.V_DEPLETED - 2, 1.5, "42 V pack", ha="right", fontsize=11.5, color=GRAY,
            fontweight="bold")
    ax.set_xlim(0, 58)
    ax.set_ylim(-0.5, 1.85)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Volts at 25 A, 1.5 m/s", fontsize=12)
    sp.save(fig, out("volt_need"))


def accel_ratio():
    n = np.linspace(1, 160, 400)
    share = 2 * (n / c.N_STAR) / (1 + (n / c.N_STAR) ** 2)
    fig, ax = plt.subplots(figsize=COL)
    left = n <= c.N_STAR
    ax.plot(n[left], share[left] * 100, color=SLATE, lw=3)
    ax.plot(n[~left], share[~left] * 100, color=RED, lw=3)
    ax.fill_between(n[~left], 0, share[~left] * 100, color=RED_L, alpha=0.6)
    ax.plot([c.N_STAR], [100], "o", ms=9, color=SLATE)
    ax.text(c.N_STAR + 3, 103, f"N* = {c.N_STAR:.0f}", fontsize=13,
            fontweight="bold", color=SLATE)
    ax.plot([100], [c.share100 * 100], "o", ms=9, color=RED)
    ax.text(100, c.share100 * 100 + 7, f"100:1 keeps {c.share100:.1%}",
            fontsize=13, fontweight="bold", color=RED, ha="left")
    ax.text(10, 25, "torque-\nlimited", fontsize=12.5, color=SLATE)
    ax.text(105, 25, "rotor-\ndominated", fontsize=12.5, color=RED)
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 118)
    ax.set_xlabel("Gear ratio N")
    ax.set_ylabel("Joint acceleration (%)")
    sp.save(fig, out("accel_ratio"))


# -----------------------------------------------------------------------------
# Heat
# -----------------------------------------------------------------------------
def envelope():
    """S1 and S3 regions of the TQ ILM 70x18 (PB figure 6a; datasheet torques,
    voltage-limit line from the chapter: 2,110 rpm at no load, 1,570 at peak)."""
    n0, n_pk = 2110.0, 1570.0
    tp, tc = c.TAU_PEAK_MOTOR, c.TAU_CONT_MOTOR

    def nlim(t):
        return n0 + (n_pk - n0) * t / tp

    fig, ax = plt.subplots(figsize=(3.6, 2.0))
    ax.fill([0, nlim(0), nlim(tc), 0], [0, 0, tc, tc], color=SLATE_L)
    ax.fill([0, nlim(tc), nlim(tp), 0], [tc, tc, tp, tp], color=AMBER_L)
    ax.plot([nlim(0), nlim(tp)], [0, tp], color=RED, lw=2.5)
    ax.text(150, tc / 2, "S1 continuous", fontsize=12, fontweight="bold", color=SLATE, va="center")
    ax.text(150, (tc + tp) / 2, "S3 bursts", fontsize=12, fontweight="bold", color=AMBER, va="center")
    ax.text(nlim(tp * 0.55) + 60, tp * 0.55, "voltage\nlimit", fontsize=11.5, color=RED)
    ax.text(700, tp + 0.15, f"peak {tp}", fontsize=11.5, color=TEXT)
    ax.text(700, tc + 0.15, f"continuous {tc}", fontsize=11.5, color=TEXT)
    ax.set_xlim(0, 2600)
    ax.set_ylim(0, 5.0)
    ax.set_xticks([0, 1000, 2000])
    ax.set_yticks([0, 2, 4])
    ax.tick_params(labelsize=11.5)
    ax.set_xlabel("Motor speed (rpm)", fontsize=12)
    ax.set_ylabel("Torque (N·m)", fontsize=12)
    sp.save(fig, out("envelope"))


def burst():
    t = np.linspace(0, 100, 400)

    def temp(t0):
        return c.T_INF - (c.T_INF - t0) * np.exp(-t / c.TAU_TH)

    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(4.1, 2.1))
        ax.axhline(c.T_LIM, color=RED, lw=2)
        ax.plot(t, temp(c.T_AMB), color=SLATE, lw=2.6)
        ax.plot([c.t_cold], [c.T_LIM], "o", ms=8, color=SLATE)
        ax.text(c.t_cold + 2, c.T_LIM - 4, f"cold: {c.t_cold:.1f} s",
                color=SLATE, fontsize=12, fontweight="bold", va="top")
        a = 1.0 if step >= 2 else 0.0
        ax.plot(t, temp(c.t_eq_rated), color=AMBER, lw=2.6, alpha=a)
        ax.plot([c.t_hot], [c.T_LIM], "o", ms=8, color=AMBER, alpha=a)
        ax.text(1, 182, f"heat-soaked start: {c.t_hot:.1f} s",
                color=AMBER, fontsize=12, fontweight="bold", alpha=a)
        ax.set_xlim(0, 100)
        ax.set_ylim(20, 200)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([50, 100, 150, 200])
        ax.tick_params(labelsize=11.5)
        ax.set_xlabel("Time in the burst (s)", fontsize=12)
        ax.set_ylabel("Winding (°C)", fontsize=12)
        sp.save(fig, out(f"burst_step{step}"))


# -----------------------------------------------------------------------------
# Power
# -----------------------------------------------------------------------------
def _scale(name, lo, hi, marks, band=None, xlabel="Drive bus voltage (V)", size=(4.2, 1.45)):
    fig, ax = plt.subplots(figsize=size)
    ax.axhline(0, color=GRAY, lw=1)
    if band:
        ax.axvspan(band[0], band[1], ymin=0.35, ymax=0.65, color=SLATE_L)
    for v, lab, col, up in marks:
        ax.plot([v, v], [-0.18, 0.18], color=col, lw=2.5)
        yl = up if not isinstance(up, bool) else (0.35 if up else -0.42)
        ax.text(v, yl, lab, ha="center", va="bottom" if yl > 0 else "top",
                fontsize=11.5, color=col, fontweight="bold", linespacing=0.95)
    ax.tick_params(labelsize=11.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-1.3, 1.9)
    ax.set_yticks([])
    for s in ("left",):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(xlabel)
    sp.save(fig, out(name))


def bus():
    _scale("bus_sag", 30, 57, marks=[(c.V_UVLO, "lockout\n33", RED, True),
            (c.V_DEPLETED - c.sag, "37.8", AMBER, False),
            (c.V_DEPLETED, "depleted\n42", SLATE, True),
            (c.V_FULL - c.sag, "49.8", AMBER, False),
            (c.V_FULL, "full\n54", SLATE, True)])
    _scale("bus_surge", 45, 190, size=(3.8, 1.45), marks=
           [(c.V_FULL, "", SLATE, 0.35),
            (c.V_CLAMP, "clamp\n60", GREEN, 0.35),
            (c.V_CONV_MAX, "converter\n67.2", AMBER, -0.42),
            (c.V_SWITCH, "switches\n100", RED, True),
            (c.v_final, "no chopper\n176.5", RED, True)],
           xlabel="Bus voltage (V)")


def phase_short():
    rpm = np.linspace(0, 420, 400)
    y = rpm / c.rpm_peak
    torque = 2 * y / (1 + y ** 2)
    fig, ax = plt.subplots(figsize=(3.6, 1.75))
    ax.axvspan(0, c.rpm, color=SLATE_L, alpha=0.7)
    ax.plot(rpm[rpm <= c.rpm_peak], torque[rpm <= c.rpm_peak], color=SLATE, lw=3)
    ax.plot(rpm[rpm >= c.rpm_peak], torque[rpm >= c.rpm_peak], color=RED, lw=3)
    ax.plot([c.rpm_peak], [1], "o", ms=9, color=TEXT)
    ax.text(c.rpm_peak + 8, 1.06, f"peak {c.rpm_peak:.1f}", fontsize=12,
            fontweight="bold", color=TEXT)
    ax.text(6, 1.12, f"drive to {c.rpm:.1f}", fontsize=11.5, color=SLATE)
    ax.text(270, 0.35, "fades with\nspeed", fontsize=11.5, color=RED)
    ax.set_xlim(0, 420)
    ax.set_ylim(0, 1.4)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=11.5)
    ax.set_xlabel("Wheel speed (rpm)", fontsize=12)
    ax.set_ylabel("Torque / peak", fontsize=12)
    sp.save(fig, out("phase_short"))


# -----------------------------------------------------------------------------
# Recall thumbnail: the elementary stop of Lecture 1 (drawn, then made text-free)
# -----------------------------------------------------------------------------
def recall_bar():
    fig, ax = plt.subplots(figsize=(5.8, 1.4))
    ax.barh(0, c.blind_elem, color=AMBER, height=0.6)
    ax.barh(0, c.brake, left=c.blind_elem, color=SLATE, height=0.6)
    ax.axvline(1100, color=RED, lw=3, ls="--")
    ax.set_xlim(0, 1250)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    sp.save(fig, out("recall_bar"))


if __name__ == "__main__":
    waterfall()
    mug_travel()
    stop_bars()
    vceiling()
    decel_ceilings()
    volt_need()
    accel_ratio()
    envelope()
    burst()
    bus()
    phase_short()
    recall_bar()
    print("figures written to", HERE)
