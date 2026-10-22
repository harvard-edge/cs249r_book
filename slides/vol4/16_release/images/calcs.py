#!/usr/bin/env python3
"""Every number on the Lecture 16 slides and notes (Deployment Release), read from MLSysIM.

MLSysIM holds the inputs (the Embodied.* running machine and its aisle record,
with the Keyence SZ-V32 scanner; Literature.* for the Challenger O-ring record
of Dalal, Fowlkes, and Hoadley, the Rogers Commission's figures, Waymo's
Safety Impact releases, and three NHTSA recall records) and the calculations:
the finished stopping budgets and the weakest deceleration that still fits
(Scenarios.WarehouseFinishedStop*), the renewal- and flag-referenced onset
bounds (WarehouseRenewalReferencedStop, WarehouseFlagReferencedStop), the
dim-aisle observation age (WarehouseNavCameraAge*), the stipulated argument
(SafetyArgumentConfidence), the exposure wall (HazardExposureWall), and the
latch and contact equations of physics.robotics. The chapter reads the same
scenarios. This file only selects values, converts units, and formats them;
it types no numbers. Each value is recorded with v() under the key the deck
prints as \\val{key}; running this file writes images/numbers.tex.
make_figs.py imports it for the figure inputs.

Run from this folder: python3 calcs.py
"""

import calendar
import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402

from mlsysim.physics.drivetrain import calc_required_friction  # noqa: E402
from mlsysim.physics.robotics import (  # noqa: E402
    calc_kinetic_energy,
    calc_latch_contact_stopping_profile,
    calc_transient_impact_force,
)

Embodied, Scenarios, Literature = mlsysim.Embodied, mlsysim.Scenarios, mlsysim.Literature
Q_ = mlsysim.core.units.Q_

RM = Embodied.MobileManipulator.WarehouseMobileManipulator
AISLE = Embodied.Scenario.WarehouseAisle
SPACE = Literature.SpaceflightAnomalies
REPORTS = Literature.IncidentReports
RECORD = load_series("slides/vol4/16_release/images/data/challenger_oring_distress_record.csv")

STOP = Scenarios.WarehouseFinishedStopAtAisleSpeed
DISPATCH = Scenarios.WarehouseFinishedStopWithDispatch
INSPECTED = Scenarios.WarehouseFinishedStopInspectedFloor
RESTRICTED = Scenarios.WarehouseFinishedStopRestricted
SILENT = Scenarios.WarehouseRenewalReferencedStop
FLAGGED = Scenarios.WarehouseFlagReferencedStop
AGE = Scenarios.WarehouseNavCameraAgeAtAisleSpeed
DIM = Scenarios.WarehouseNavCameraAgeDimAisle
ARG = Scenarios.SafetyArgumentConfidence
WALL = Scenarios.HazardExposureWall


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def sci(key, value, fmt, label):
    """Record a positive value as mantissa (key_mant) and power of ten (key_exp)."""
    exp = math.floor(math.log10(value))
    mant = value / 10 ** exp
    if float(fmt.format(mant)) >= 10:  # the mantissa rounds up to the next decade
        exp, mant = exp + 1, mant / 10
    v(f"{key}_mant", mant, fmt, f"{label}, mantissa")
    v(f"{key}_exp", exp, "{:.0f}", f"{label}, power of ten")


def ymd(n):
    """A YYYYMMDD record field as a date."""
    y, md = divmod(int(n), 10 ** 4)
    return date(y, *divmod(md, 100))


stop, dispatch, insp = STOP.result, DISPATCH.result, INSPECTED.result
silent, flagged = SILENT.result, FLAGGED.result
latch_lo = calc_latch_contact_stopping_profile(AISLE.v_latch_approach, scenario=AISLE)
latch_hi = calc_latch_contact_stopping_profile(AISLE.v_latch_high, scenario=AISLE)
m_eff, m_hand, k_hand = RM.arm_effective_contact_mass, AISLE.m_contact_human, AISLE.k_contact_human

# Inputs for make_figs.py, at full precision.
ORING = [(int(e["flight_order"]), e["joint_temp_f"], int(e["orings_distressed"])) for e in RECORD.entries]
TEMPS = sorted(t for _, t, _ in ORING)
# The cold group: every flight at or below the warmest temperature up to which all flights had distress.
COLD_MAX = max(t for t in TEMPS if all(d > 0 for _, tt, d in ORING if tt <= t))
WARM_MIN = min(t for t in TEMPS if t > COLD_MAX)
COLD = [d for _, t, d in ORING if t <= COLD_MAX]
WARM = [d for _, t, d in ORING if t >= WARM_MIN]
T_CHALLENGER = SPACE.ChallengerLaunchAirF
T_JOINT_51L = SPACE.ChallengerAftJointEstimateF
T_JOINT_51L_ERR = SPACE.ChallengerAftJointEstimateUncertaintyF
T_THIOKOL = SPACE.ThiokolMinimumJointTempF
G = m(Q_(1, "standard_gravity"), "m/s**2")
V = m(AISLE.v_aisle, "m/s")
T_LEASE, T_TICK = m(AISLE.t_lease, "ms"), m(AISLE.t_tick, "ms")
T_BUS, T_ONSET = m(AISLE.t_bus, "ms"), m(AISLE.t_brake_onset, "ms")
T_SILENT, T_FLAGGED = m(SILENT.delay, "ms"), m(FLAGGED.delay, "ms")
A_LATCH_LO = m(latch_lo["a_arm_needed"], "m/s**2")
A_LATCH_HI = m(latch_hi["a_arm_needed"], "m/s**2")
DISTRESSED = sum(1 for _, _, d in ORING if d > 0)


print("== Hook: Challenger (Rogers Commission; Dalal et al. 1989 via UCI)")
v("air_f", T_CHALLENGER, "{:.0f}", "launch air temperature, F")
v("colder_f", SPACE.ChallengerColderThanPriorLaunchF, "{:.0f}", "colder than any previous launch, F")
v("coldest_f", TEMPS[0], "{:.0f}", "coldest joint in the record, F")
v("warmest_f", TEMPS[-1], "{:.0f}", "warmest joint in the record, F")
v("thiokol_f", T_THIOKOL, "{:.0f}", "Thiokol's recommended floor, F")
v("n_flights", len(ORING), "{:.0f}", "inspected flights in the record")
v("n_prior", SPACE.ChallengerPriorJointFlights, "{:.0f}", "previous flights")
v("flight_no", SPACE.ChallengerPriorJointFlights + 1, "{:.0f}", "Challenger's flight number (ordinal)")
v("rings_words", max(e["orings_at_risk"] for e in RECORD.entries), "words", "O-rings per flight")
v("breakup_s", SPACE.ChallengerBreakupS, "{:.0f}", "liftoff to breakup, s")
v("joint_f", T_JOINT_51L, "{:.0f}", "right aft joint estimate, F")
v("joint_err_f", T_JOINT_51L_ERR, "{:.0f}", "its uncertainty, F")
v("cold_max_f", COLD_MAX, "{:.0f}", "warmest flight of the all-distressed cold group, F")
v("warm_min_f", WARM_MIN, "{:.0f}", "coldest flight above it, F")
v("cold_hit", sum(1 for d in COLD if d > 0), "{:.0f}", "cold flights with distress")
v("cold_n", len(COLD), "{:.0f}", "cold flights")
v("warm_hit", sum(1 for d in WARM if d > 0), "{:.0f}", "warm flights with distress")
v("warm_n", len(WARM), "{:.0f}", "warm flights")
v("n_distressed", DISTRESSED, "{:.0f}", "flights with distress")

print("== Recall: fault F1, the stalled proposer (Lecture 15; DR NM 16.2)")
v("t_lease", T_LEASE, "{:.0f}", "chunk lease, ms")
v("t_tick", T_TICK, "{:.0f}", "permission tick, ms")
v("t_bus", T_BUS, "{:.0f}", "fieldbus transfer, ms")
v("t_onset", T_ONSET, "{:.0f}", "brake onset, ms")
v("t_silent", T_SILENT, "{:.0f}", "last renewal to brake onset, ms")

print("== Claims: the three criteria and the envelope (DR equation 1, table 1)")
v("v_aisle", V, "{:.1f}", "aisle speed, m/s")
v("d_clear_m", m(AISLE.d_clear, "m"), "{:.2f}", "clear distance, m")
v("d_clear_mm", m(AISLE.d_clear, "mm"), "{:,.0f}", "clear distance, mm")
v("f_latch", m(AISLE.f_latch_limit, "N"), "{:.0f}", "latch force limit, N")
v("v_latch", m(AISLE.v_latch_approach, "m/s"), "{:.2f}", "guarded latch approach, m/s")
v("f_contact", m(AISLE.f_contact_criterion, "N"), "{:.0f}", "handover force criterion, N")
v("v_handover", m(AISLE.v_handover, "m/s"), "{:.2f}", "handover speed, m/s")
v("wcet", m(AISLE.enforcer_wcet, "us"), "{:.0f}", "enforcer WCET, declared, us")
v("deadline", m(AISLE.enforcer_deadline, "us"), "{:.0f}", "permission deadline, us")
v("payload", m(RM.onboard_payload_capacity, "kg"), "{:.0f}", "tote-rack payload, kg")
v("mu_min", calc_required_friction(AISLE.a_brake), "{:.3f}", "friction floor for a_brake")

print("== Estimate: the aisle spare (DR Napkin Math 16.1)")
v("tau_delay", m(AISLE.tau_delay, "ms"), "{:.1f}", "pre-brake delay, ms")
v("a_brake", m(AISLE.a_brake, "m/s**2"), "{:.0f}", "credible deceleration, m/s^2")
v("j_brake", m(stop.jerk, "m/s**3"), "{:.2f}", "jerk bound, m/s^3")
v("allowances", m(stop.overhead + stop.tracking_inset, "mm"), "{:.0f}", "localization, tracking, margin, mm")
v("d_loc", m(AISLE.delta_loc, "mm"), "{:.0f}", "localization allowance, mm")
v("d_track", m(AISLE.eps_track, "mm"), "{:.0f}", "tracking allowance, mm")
v("d_margin", m(AISLE.delta_margin, "mm"), "{:.0f}", "protective margin, mm")
v("reaction", m(stop.delay_travel, "mm"), "{:.1f}", "reaction travel, mm")
v("scurve", m(stop.scurve_travel, "mm"), "{:.1f}", "jerk-limited stop, mm")
v("budget", m(stop.required_distance, "mm"), "{:.1f}", "budget, mm")
v("spare", m(stop.spare, "mm"), "{:.1f}", "spare, mm")
v("t_spare", m(stop.delay_spare, "ms"), "{:.1f}", "spare as extra delay, ms")
v("a_min", m(stop.min_deceleration, "m/s**2"), "{:.2f}", "weakest deceleration that still fits, m/s^2")
v("shortfall_pct", 100 * stop.deceleration_shortfall, "{:.0f}", "deceleration shortfall that spends the spare, %")
v("t_da", m(AISLE.t_dispatch_to_admission, "ms"), "{:.1f}", "dispatch-to-admission allocation, ms")
v("travel_da", m(stop.travel_over(AISLE.t_dispatch_to_admission), "mm"), "{:.1f}", "its travel, mm")
v("reserve", m(dispatch.spare, "mm"), "{:.1f}", "reserve after admission work, mm")
v("t_reserve", m(dispatch.delay_spare, "ms"), "{:.1f}", "reserve as time, ms")

print("== Safety case: the exposure wall and a defeater (Lecture 7; DR 16.3.1)")
wall_rate = min(WALL.rates, key=lambda r: m(r, "1/hour"))
v("wall_rate_exp", math.log10(m(wall_rate, "1/hour")), "{:.0f}", "hazard-rate exponent, per hour")
v("confidence_pct", 100 * WALL.confidence, "{:.0f}", "confidence, %")
sci("wall_h", m(WALL.exposure(wall_rate), "hour"), "{:.1f}", "failure-free hours")
v("mu_oil", AISLE.mu_oil_film, "{:.2f}", "clear oil film friction")

print("== Premises (DR equations 2 and 3; Scenarios.SafetyArgumentConfidence)")
v("n_premises_words", ARG.premises, "words", "required premises")
v("p_premise", ARG.premise_confidence, "{:.2f}", "confidence per premise")
v("p_conj", ARG.conjunction, "{:.3f}", "conjunction")
v("p_leg", ARG.leg_confidence, "{:.2f}", "confidence per leg")
v("p_leg_miss", 1 - ARG.leg_confidence, "{:.1f}", "one leg's failure probability")
v("p_leg_miss2", 1 - ARG.leg_confidence, "{:.2f}", "one leg's failure probability, two places")
v("p_legs", ARG.multilegged, "{:.2f}", "two independent legs")

print("== Vote: a dim aisle (DR 16.3.2; Scenarios.WarehouseNavCameraAge*)")
v("t_exp", m(AGE.exposure_time, "ms"), "{:.0f}", "nominal exposure, ms")
v("t_exp_dim", m(DIM.exposure_time, "ms"), "{:.0f}", "dim-aisle exposure, ms")
v("dt_age", m(DIM.result.age - AGE.result.age, "ms"), "{:.0f}", "added observation age, ms")
d_dim = DIM.result.age_travel - AGE.result.age_travel
v("dim_travel", m(d_dim, "mm"), "{:.1f}", "its travel at the aisle speed, mm")
v("dim_share_pct", 100 * m(d_dim / stop.spare, ""), "{:.0f}", "share of the spare, %")

print("== Certification: Keyence SZ-V32 (MLSysIM registry)")
sc = RM.safety_scanner
v("scan_fast", m(sc.response_time, "ms"), "{:.0f}", "high-speed-mode response, ms")
v("scan_std", m(sc.response_time_standard_mode, "ms"), "{:.0f}", "standard-mode response, ms")
v("scan_fast_mm", m(stop.travel_over(sc.response_time), "mm"), "{:.0f}", "travel during it, mm")
v("scan_std_mm", m(stop.travel_over(sc.response_time_standard_mode), "mm"), "{:.0f}", "travel during it, mm")

print("== Door predict: the latch branch (DR 16.6; Lecture 12's equations)")
v("f_trip", m(AISLE.f_latch_tripwire, "N"), "{:.0f}", "tripwire, N")
v("f_onset", m(latch_lo["f_at_brake_onset"], "N"), "{:.0f}", "force at brake onset, guarded, N")
v("x_rem", m(latch_lo["x_remaining"], "mm"), "{:.2f}", "latch travel left, mm")
v("a_arm", A_LATCH_LO, "{:.0f}", "arm deceleration needed, guarded, m/s^2")
v("t_arm", m(latch_lo["t_arm_stop"], "ms"), "{:.0f}", "guarded arm stop, ms")
v("v_latch_hi", m(AISLE.v_latch_high, "m/s"), "{:.2f}", "fast approach, m/s")
v("f_onset_hi", m(latch_hi["f_at_brake_onset"], "N"), "{:.0f}", "force at brake onset, fast, N")
v("a_arm_hi", A_LATCH_HI, "{:,.0f}", "arm deceleration needed, fast, m/s^2")
v("a_arm_ratio", A_LATCH_HI / A_LATCH_LO, "{:.0f}", "fast over guarded")
v("a_arm_hi_g", A_LATCH_HI / G, "{:.0f}", "fast demand, g")
v("g_one", G, "{:.1f}", "one g, m/s^2")

print("== Tree: the handover leaf (Lecture 12's transient contact model)")
v("f_handover", m(calc_transient_impact_force(AISLE.v_handover, m_eff, k_hand, human_mass=m_hand), "N"),
  "{:.1f}", "free-hand transient force at the handover speed, N")
v("t_age", m(AISLE.perception_age, "ms"), "{:.1f}", "dispatch age, ms")

print("== COND: restricted mug delivery (DR 16.4.1, 16.6.1)")
v("mu_insp", AISLE.mu_inspected_floor, "{:.2f}", "inspected-floor friction")
v("g", G, "{:.2f}", "standard gravity, m/s^2")
v("a_insp", m(INSPECTED.deceleration, "m/s**2"), "{:.2f}", "traction-limited deceleration, m/s^2")
v("ceiling_insp", m(insp.speed_ceiling, "m/s"), "{:.2f}", "inspected-floor ceiling, m/s")
v("v_restricted", m(RESTRICTED.speed, "m/s"), "{:.0f}", "signed restricted speed, m/s")

print("== Waymo: rider-only crash rates by outcome (vol4/16_release/data/waymo_safety_impact_headline.csv)")
rows = load_series("vol4/16_release/data/waymo_safety_impact_headline.csv").entries
last = max(r["data_cut"] for r in rows)
latest = {r["outcome"]: r for r in rows if r["data_cut"] == last}
inj, ssi, fat = latest["Any-injury-reported"], latest["Suspected Serious Injury+"], latest["Any Fatal"]
cut_y, cut_m = divmod(int(last), 100)
v("waymo_cut", f"{calendar.month_name[cut_m]} {cut_y}", "{}", "latest release")
v("waymo_miles", inj["ro_miles_millions"], "{:.0f}", "rider-only miles, millions")
for key, r in (("inj", inj), ("ssi", ssi), ("fat", fat)):
    v(f"{key}_n", r["waymo_count"], "{:.0f}", f"{r['outcome']}: crashes")
    v(f"{key}_lo", -100 * r["pct_reduction_ci_hi"], "{:.0f}", f"{r['outcome']}: reduction, lower bound, %")
    v(f"{key}_hi", -100 * r["pct_reduction_ci_lo"], "{:.0f}", f"{r['outcome']}: reduction, upper bound, %")
v("ssi_over_inj", (ssi["pct_reduction_ci_hi"] - ssi["pct_reduction_ci_lo"])
  / (inj["pct_reduction_ci_hi"] - inj["pct_reduction_ci_lo"]), "{:.1f}", "interval width, serious injury over injury")

print("== Expiry: the silent proposer (DR Napkin Math 16.2)")
v("mm_per_ms", m(STOP.result.travel_over(Q_(1, "ms")), "mm"), "{:.1f}", "travel per ms at the aisle speed, mm")
v("d_silent", m(silent.delay_travel, "mm"), "{:.1f}", "silent proposer, travel to onset, mm")
v("t_flagged", T_FLAGGED, "{:.0f}", "immediate flag to onset, ms")
v("d_flagged", m(flagged.delay_travel, "mm"), "{:.1f}", "immediate flag, travel to onset, mm")
v("d_saved", m(silent.delay_travel - flagged.delay_travel, "mm"), "{:.0f}", "travel monitoring can save, mm")
v("t_crossover", T_SILENT - T_FLAGGED, "{:.0f}", "flag latency that saves nothing, ms")

print("== Updates: Waymo recalls (NHTSA, Literature.IncidentReports)")
recalls = [(REPORTS.WaymoRecall24E013Vehicles, ymd(REPORTS.WaymoRecall24E013ReportDate)),
           (REPORTS.WaymoRecall24E049Vehicles, ymd(REPORTS.WaymoRecall24E049ReportDate)),
           (REPORTS.WaymoRecall25E034Vehicles, ymd(REPORTS.WaymoRecall25E034ReportDate))]
for key, (n, _) in zip(("rc1", "rc2", "rc3"), recalls):
    v(f"{key}_n", n, "{:,.0f}", "vehicles in the recall")
(n0, d0), (n2, d2) = recalls[0], recalls[-1]
v("rc_words", len(recalls), "words", "recalls")
v("rc_months", (d2.year - d0.year) * (len(calendar.month_name) - 1) + d2.month - d0.month, "{:.0f}", "months from the first report to the third")
v("rc_growth", n2 / n0, "{:.1f}", "fleet growth across the recalls, times")
v("rc_days", (d2 - ymd(REPORTS.WaymoRecall25E034RemedyCompleteDate)).days, "{:.0f}",
  "25E-034: days from every vehicle repaired to the report")

print("== Vote: canary consequence (DR 16.10)")
v("m_loaded", m(RM.loaded_mass, "kg"), "{:.0f}", "loaded mass, kg")
v("ke", m(calc_kinetic_energy(RM.loaded_mass, AISLE.v_aisle), "J"), "{:.0f}", "kinetic energy at the aisle speed, J")

if __name__ == "__main__":
    write(__file__)
