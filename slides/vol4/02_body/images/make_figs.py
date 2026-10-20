#!/usr/bin/env python3
"""Figures for Lecture 2 (The Physical Body), drawn at the size of their slots.

Every number drawn here comes from calcs.py, which reads MLSysIM (the
Embodied.* machine and site, Actuators.*, and the Scenarios.* the chapter also
reads): plotted geometry uses the scenario values, and labels use the
formatted strings the slides print (deckvals val/num). Axis ranges, tick
positions, and label coordinates are layout. Plots use the kit's slideplot
helper (large type, direct labels). Colors follow the Volume IV palette: plant
slate, hazard red, latency amber, sensing teal, deterministic green; nothing
uses the accent.

Run from this folder: python3 make_figs.py   (then `make deck V=4 D=02_body`)
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import slideplot as sp  # noqa: E402
from slideplot import plt  # noqa: E402

import calcs as c  # noqa: E402  (records every value; figures read them)
from deckvals import val, num  # noqa: E402

mlsysim = c.mlsysim

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
# Freshness: where the age of one grasp observation goes (illustrative profile)
# -----------------------------------------------------------------------------
def waterfall():
    stages = [(name, c.m(p50, "ms")) for name, p50, _, _ in c.GRASP.stages]
    colors = [TEAL, TEAL, AMBER, SLATE, SLATE, SLATE]
    backbone = c.m(c.GRASP.result.focus_stage, "ms")
    median = c.m(c.GRASP.result.stage_sum_p50, "ms")
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.4))
        x = 0.0
        for i, ((name, v), col) in enumerate(zip(stages, colors)):
            shown = step >= 2 or name == c.GRASP.focus_stage
            ax.barh(1, v, left=x, color=col, height=0.5, edgecolor="white",
                    lw=1.5, alpha=1.0 if shown else 0.0)
            x += v
        # inference only, on its own row in step 1 for the benchmark view
        ax.barh(0, backbone, left=0, color=AMBER, height=0.5)
        ax.text(backbone + 1, 0, f"benchmark: {val('st_backbone')} ms", va="center",
                fontsize=12, color=AMBER, fontweight="bold")
        ax.text(-1, 1, "Capture to\ntorque onset", ha="right", va="center",
                fontsize=12, color=TEXT, fontweight="bold", alpha=1 if step >= 2 else 0)
        ax.text(-1, 0, "Model only", ha="right", va="center", fontsize=12,
                color=TEXT, fontweight="bold")
        ax.text(median, 1.45, f"median {val('p50_sum')}", ha="center", va="bottom",
                fontsize=11.5, color=SLATE, fontweight="bold", alpha=1 if step >= 2 else 0)
        # labels for the non-inference stages above the bar
        ax.text(stages[0][1] / 2, 1.45, "exposure,\nreadout", ha="center", va="bottom",
                fontsize=11, color=TEAL, alpha=1 if step >= 2 else 0)
        a3 = 1 if step >= 3 else 0
        ax.axvline(num("deadline"), color=GREEN, lw=2, ls="--", alpha=a3)
        ax.text(num("deadline") - 0.8, -0.55, f"deadline {val('deadline')} ms",
                color=GREEN, fontsize=11, fontweight="bold", ha="right", alpha=a3)
        ax.plot([num("p99_sys")], [1], marker="|", ms=26, mew=2.5, color=SLATE, alpha=a3)
        ax.text(num("p99_sys"), 0.72, f"p99 {val('p99_sys')}", ha="center", va="top",
                fontsize=11, color=SLATE, alpha=a3)
        ax.plot([num("p999_sys")], [1], marker="|", ms=26, mew=2.5, color=RED, alpha=a3)
        ax.text(num("p999_sys"), 0.72, f"p99.9 {val('p999_sys')}", ha="center", va="top",
                fontsize=11, color=RED, fontweight="bold", alpha=a3)
        ax.set_xlim(0, 75)
        ax.set_ylim(-0.75, 2.2)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Age of the observation (ms)")
        sp.save(fig, out(f"waterfall_step{step}"))


def mug_travel():
    g = c.GRASP.result
    rows = [(f"Median\n{val('p50_sum')} ms", c.m(g.travel_p50, "mm"), SLATE, f"{val('mug_p50')} mm"),
            (f"p99.9\n{val('p999_sys')} ms", c.m(g.travel_tail, "mm"), RED, f"{val('mug_tail')} mm")]
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
        ax.axvline(num("emax_mm"), color=GREEN, lw=2, ls="--")
        ax.text(num("emax_mm"), 1.55, f"allowed {val('emax_mm')} mm", color=GREEN, fontsize=11.5,
                fontweight="bold", ha="center")
        ax.set_xlim(0, 17)
        ax.set_ylim(-0.5, 1.8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Mug travel (mm)", fontsize=12)
        sp.save(fig, out(f"mug_travel_step{step}"))


def vceiling():
    d = np.linspace(0.15, 2.0, 300)
    curves = [(c.BODY, SLATE, "body terms only", val("vmax_body"), 2.78),
              (c.FULL, AMBER, f"{val('tau_delay_ms')} ms delay", val("vmax_full"), 2.22),
              (c.INSPECTED, RED, "inspected floor", val("vmax_insp"), 1.66)]
    d_clear = c.m(c.AISLE.d_clear, "m")
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.5))
        for k, (scen, col, lab, vtxt, ylab) in enumerate(curves):
            vs = [c.m(scen.variant(clear_distance=x * c.ureg.meter).result.speed_ceiling, "m/s") for x in d]
            alpha = 1.0 if k < step else 0.0
            ax.plot(d, vs, color=col, lw=2.6, alpha=alpha)
            ax.plot([d_clear], [c.m(scen.result.speed_ceiling, "m/s")], "o", ms=8, color=col, alpha=alpha)
            ax.text(2.04, ylab, f"{lab}:\n{vtxt} m/s at {val('d_clear_m1')} m", va="center",
                    fontsize=11, color=col, fontweight="bold", alpha=alpha,
                    linespacing=0.95)
        ax.axhline(num("v_drive"), color=GRAY, lw=1.2, ls=":")
        ax.text(1.35, num("v_drive") - 0.08, f"drive limit {val('v_drive')} m/s", fontsize=10.5,
                color=GRAY, va="top")
        ax.axvline(d_clear, color=GRAY, lw=1, ls="--")
        ax.set_xlim(0, 2.0)
        ax.set_ylim(0, 2.9)
        fig.subplots_adjust(right=0.68)
        ax.set_xlabel("Clear distance (m)")
        ax.set_ylabel("Speed ceiling (m/s)")
        sp.save(fig, out(f"vceiling_step{step}"))


def decel_ceilings():
    r = c.CEIL.result
    rows = [("Dry floor", r.a_traction_dry, "a_dry", AMBER), ("Peak current", r.a_torque, "a_torque", AMBER),
            ("Inspected floor", r.a_traction_inspected, "a_insp", RED), ("Rated current", r.a_torque_rated, "a_rated", RED)]
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    n = len(rows) + 1
    for i, (lab, q, key, col) in enumerate(rows):
        v = c.m(q, "m/s**2")
        y = n - 1 - i
        ax.barh(y, v, color=col, height=0.6)
        ax.text(v + 0.06, y, val(key), va="center", fontsize=12.5,
                fontweight="bold", color=col)
        ax.text(-0.06, y, lab, ha="right", va="center", fontsize=12.5,
                fontweight="bold", color=TEXT)
    ax.text(-0.06, 0, "Tip-over", ha="right", va="center", fontsize=12.5,
            fontweight="bold", color=TEXT)
    ax.text(0.08, 0, "unmeasured", va="center", fontsize=12.5, color=GRAY,
            style="italic")
    a_brake = c.m(c.AISLE.a_brake, "m/s**2")
    ax.axvline(a_brake, color=TEXT, lw=1.8, ls="--")
    ax.text(a_brake + 0.05, n - 0.35, f"needed {val('a_brake_1')}", fontsize=11.5,
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
    r = c.CEIL.result
    rows = [("Braking", c.m(r.v_brake, "V"), GREEN, f"{val('v_brake')} V"),
            ("Driving", c.m(r.v_drive, "V"), RED, f"{val('v_drive_need')} V")]
    fig, ax = plt.subplots(figsize=(3.2, 1.6))
    for i, (lab, v, col, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=col, height=0.6)
        ax.text(v + 1, y, txt, va="center", fontsize=12, fontweight="bold", color=col)
        ax.text(-1, y, lab, ha="right", va="center", fontsize=12, fontweight="bold", color=TEXT)
    ax.axvline(num("v_low"), color=GRAY, lw=1.8, ls="--")
    ax.text(num("v_low") - 2, 1.5, f"{val('v_low')} V pack", ha="right", fontsize=11.5, color=GRAY,
            fontweight="bold")
    ax.set_xlim(0, 58)
    ax.set_ylim(-0.5, 1.85)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Volts at {val('hub_i_peak')} A, {val('v_drive')} m/s", fontsize=12)
    sp.save(fig, out("volt_need"))


def accel_ratio():
    n_star = c.ROTOR.result.matched_ratio
    n = np.linspace(1, 160, 400)
    share = np.array([mlsysim.calc_peak_normalized_response(x / n_star) for x in n])
    fig, ax = plt.subplots(figsize=COL)
    left = n <= n_star
    ax.plot(n[left], share[left] * 100, color=SLATE, lw=3)
    ax.plot(n[~left], share[~left] * 100, color=RED, lw=3)
    ax.fill_between(n[~left], 0, share[~left] * 100, color=RED_L, alpha=0.6)
    ax.plot([n_star], [100], "o", ms=9, color=SLATE)
    ax.text(n_star + 3, 103, f"N* = {val('n_star')}", fontsize=13,
            fontweight="bold", color=SLATE)
    kept = c.ROTOR.result.acceleration_share * 100
    ax.plot([c.ROTOR.gear_ratio], [kept], "o", ms=9, color=RED)
    ax.text(c.ROTOR.gear_ratio, kept + 7, f"{val('n_gear')}:1 keeps {val('accel_pct')}%",
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
    """S1 and S3 regions of the TQ ILM 70x18 (PB figure 6a): datasheet torques,
    and the voltage limit from calc_voltage_limited_speed (no-load speed from
    the data sheet)."""
    motor = c.MOTOR
    tp, tc = c.m(motor.peak_torque, "N*m"), c.m(motor.rated_torque, "N*m")

    def nlim(t):
        return c.m(mlsysim.calc_voltage_limited_speed(t * c.ureg("N*m"), motor.torque_constant,
                                                      motor.terminal_resistance, motor.rated_voltage,
                                                      motor.no_load_speed), "rpm")

    fig, ax = plt.subplots(figsize=(3.6, 2.0))
    ax.fill([0, nlim(0), nlim(tc), 0], [0, 0, tc, tc], color=SLATE_L)
    ax.fill([0, nlim(tc), nlim(tp), 0], [tc, tc, tp, tp], color=AMBER_L)
    ax.plot([nlim(0), nlim(tp)], [0, tp], color=RED, lw=2.5)
    ax.text(150, tc / 2, "S1 continuous", fontsize=12, fontweight="bold", color=SLATE, va="center")
    ax.text(150, (tc + tp) / 2, "S3 bursts", fontsize=12, fontweight="bold", color=AMBER, va="center")
    ax.text(nlim(tp * 0.55) + 60, tp * 0.55, "voltage\nlimit", fontsize=11.5, color=RED)
    ax.text(700, tp + 0.15, f"peak {val('motor_peak')}", fontsize=11.5, color=TEXT)
    ax.text(700, tc + 0.15, f"continuous {val('motor_rated')}", fontsize=11.5, color=TEXT)
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
    r = c.BURST.result

    def temp(start):
        return np.array([c.m(mlsysim.calc_first_order_temperature(x * c.ureg.second, start, r.asymptote,
                                                                  r.time_constant), "degC") for x in t])

    t_lim = num("t_limit")
    cold, hot = c.m(r.horizon_cold, "s"), c.m(r.horizon_hot, "s")
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(4.1, 2.1))
        ax.axhline(t_lim, color=RED, lw=2)
        ax.plot(t, temp(c.BURST.ambient), color=SLATE, lw=2.6)
        ax.plot([cold], [t_lim], "o", ms=8, color=SLATE)
        ax.text(cold + 2, t_lim - 4, f"cold: {val('t_cold')} s",
                color=SLATE, fontsize=12, fontweight="bold", va="top")
        a = 1.0 if step >= 2 else 0.0
        ax.plot(t, temp(r.rated_equilibrium), color=AMBER, lw=2.6, alpha=a)
        ax.plot([hot], [t_lim], "o", ms=8, color=AMBER, alpha=a)
        ax.text(1, 182, f"heat-soaked start: {val('t_hot')} s",
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
    u = c.BUS.result
    _scale("bus_sag", 30, 57, marks=[(num("v_uvlo"), f"lockout\n{val('v_uvlo')}", RED, True),
            (c.m(u.sustained_depleted, "V"), val("v_sag_low"), AMBER, False),
            (num("v_low"), f"depleted\n{val('v_low')}", SLATE, True),
            (c.m(u.sustained_full, "V"), val("v_sag_full"), AMBER, False),
            (num("v_full"), f"full\n{val('v_full')}", SLATE, True)])
    _scale("bus_surge", 45, 190, size=(3.8, 1.45), marks=
           [(num("v_full"), "", SLATE, 0.35),
            (num("v_clamp"), f"clamp\n{val('v_clamp')}", GREEN, 0.35),
            (num("v_conv_max"), f"converter\n{val('v_conv_max')}", AMBER, -0.42),
            (num("v_switch"), f"switches\n{val('v_switch')}", RED, True),
            (c.m(u.capacitor_only_endpoint, "V"), f"no chopper\n{val('v_final')}", RED, True)],
           xlabel="Bus voltage (V)")


def phase_short():
    peak = c.m(c.ELEC.result.short_peak_wheel_speed, "rpm")
    drive = c.m(c.CEIL.result.wheel_speed, "rpm")
    rpm = np.linspace(0, 420, 400)
    torque = np.array([mlsysim.calc_peak_normalized_response(x / peak) for x in rpm])
    fig, ax = plt.subplots(figsize=(3.6, 1.75))
    ax.axvspan(0, drive, color=SLATE_L, alpha=0.7)
    ax.plot(rpm[rpm <= peak], torque[rpm <= peak], color=SLATE, lw=3)
    ax.plot(rpm[rpm >= peak], torque[rpm >= peak], color=RED, lw=3)
    ax.plot([peak], [1], "o", ms=9, color=TEXT)
    ax.text(peak + 8, 1.06, f"peak {val('rpm_short')}", fontsize=12,
            fontweight="bold", color=TEXT)
    ax.text(6, 1.12, f"drive to {val('rpm_drive')}", fontsize=11.5, color=SLATE)
    ax.text(270, 0.35, "fades with\nspeed", fontsize=11.5, color=RED)
    ax.set_xlim(0, 420)
    ax.set_ylim(0, 1.4)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=11.5)
    ax.set_xlabel("Wheel speed (rpm)", fontsize=12)
    ax.set_ylabel("Torque / peak", fontsize=12)
    sp.save(fig, out("phase_short"))


if __name__ == "__main__":
    waterfall()
    mug_travel()
    vceiling()
    decel_ceilings()
    volt_need()
    accel_ratio()
    envelope()
    burst()
    bus()
    phase_short()
    print("figures written to", HERE)
