#!/usr/bin/env python3
"""Every number on the Lecture 7 slides and notes (Closed-Loop Evaluation), read from MLSysIM.

MLSysIM holds the inputs (the Embodied.* machine and its site record, the
latch scenario table, Literature.IncidentReports for NHTSA PE 16-007 and its
replication, Literature.RoadVehicleSafety for RAND RR-1478 and Waymo's
rider-only miles, Literature.RobotLearning for Mandlekar et al. 2021, TRI's
large behavior models, and SIMPLER) and the calculations (the Scenarios.*
entries for closed-loop evaluation: the latch campaign and its zero-failure
bounds, the sequential test, the two-checkpoint comparison, the heading drift,
the stiff-contact residual, the exposure wall, the force tail, the road-mileage requirements, and the crash-rate denominator);
the chapter reads the same scenarios. This file only selects values, converts
units, and formats them; it types no numbers. Each value is recorded with v()
under the key the deck prints as \\val{key}; running this file writes
images/numbers.tex. make_figs.py imports it for the scenario objects.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.physics.robotics.assurance import RULE_OF_THREE_NUMERATOR  # noqa: E402

Embodied, Scenarios, Literature = mlsysim.Embodied, mlsysim.Scenarios, mlsysim.Literature

RM = Embodied.MobileManipulator.WarehouseMobileManipulator
AISLE = Embodied.Scenario.WarehouseAisle
LIT = Literature.RobotLearning
ROAD_LIT = Literature.RoadVehicleSafety

HOOK = Scenarios.AutosteerCrashRateDenominator
CAMPAIGN = Scenarios.LatchEvaluationCampaign
TABLE = CAMPAIGN.table
DRIFT = Scenarios.AisleHeadingDrift
CONTACT = Scenarios.StiffSurfaceResidual
SPRT = Scenarios.LatchSequentialTest
COMPARE = Scenarios.LatchCheckpointComparison
WALL = Scenarios.HazardExposureWall
ROAD = Scenarios.RoadMilesToDemonstrate
TAIL = Scenarios.LatchForceTail

SITE_RATE, RATE_E6, RATE_E9 = WALL.rates
NINE_90, NINE_99, NINE_999 = CAMPAIGN.nine_targets


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def power_of_ten(x):
    return math.floor(math.log10(x))


def sci(key, x, mant_fmt, label):
    """Record x as a mantissa (key_m) and a power of ten (key_e)."""
    e = power_of_ten(x)
    v(f"{key}_m", x / 10 ** e, mant_fmt, f"{label}, mantissa")
    v(f"{key}_e", e, "{:.0f}", f"{label}, power of ten")


print("== Hook: NHTSA PE 16-007 and its replication")
v("nhtsa_before", HOOK.reported_before, "{:.1f}", "airbag crashes per million miles, before Autosteer")
v("nhtsa_after", HOOK.reported_after, "{:.1f}", "airbag crashes per million miles, after")
v("nhtsa_drop_pct", -100 * HOOK.reported_change, "{:.1f}", "reported drop, %")
v("nhtsa_claim_pct", -100 * round(HOOK.reported_change, 1), "{:.0f}", "the report's rounding of the drop, %")
v("qcs_before", HOOK.cohort_before, "{:.2f}", "complete-exposure cohort, before")
v("qcs_after", HOOK.cohort_after, "{:.2f}", "complete-exposure cohort, after")
v("qcs_rise_pct", 100 * HOOK.cohort_change, "{:.0f}", "complete-exposure cohort change, %")
v("qcs_share_pct", 100 * HOOK.exact_exposure_share, "{:.0f}", "vehicles with exact exposure, %")

print("== Running case: the latch campaign")
v("standoff_m", m(CAMPAIGN.standoff, "m"), "{:.0f}", "base standoff, m")
v("v_guard", m(AISLE.v_latch_approach, "m/s"), "{:.2f}", "guarded approach, m/s")
v("f_trip", m(AISLE.f_latch_tripwire, "N"), "{:.0f}", "contact tripwire, N")
v("f_lim", m(AISLE.f_latch_limit, "N"), "{:.0f}", "latch structural limit, N")
v("run_s", m(CAMPAIGN.run, "s"), "{:.0f}", "one run, s")
v("reset_s", m(CAMPAIGN.reset, "s"), "{:.0f}", "automated reset, s")
v("cycle_s", m(CAMPAIGN.cycle, "s"), "{:.0f}", "door cycle, s")
v("n_clean", CAMPAIGN.clean_runs, "{:.0f}", "clean runs")
v("n_clean_words", CAMPAIGN.clean_runs, "words", "clean runs, spelled")
v("cells_words", TABLE.cells, "words", "scenario-table cells")
v("supported_words", TABLE.supported, "words", "supported cells")
v("absences_words", TABLE.absences, "words", "cells without evidence")
v("plates_words", len(TABLE.offsets), "words", "plate positions")
v("speeds_words", len(TABLE.speeds), "words", "approach speeds")
v("plate_offset_mm", m(max(TABLE.offsets), "mm"), "{:.0f}", "displaced plate, mm")
v("chunk_hz", m(RM.chunk_rate, "Hz"), "{:.0f}", "chunk policy rate, Hz")
v("lease_ms", m(AISLE.t_lease, "ms"), "{:.0f}", "chunk lease, ms")
v("jam_s", m(CAMPAIGN.jam_reset, "s"), "{:.0f}", "manual reset of a jammed run, s")
v("jam_cycle_s", m(CAMPAIGN.jam_cycle, "s"), "{:.0f}", "jammed run plus its reset, s")
v("jam_pct", 100 * CAMPAIGN.jam_reset_share, "{:.1f}", "jam reset share of cell time, %")

print("== Denominator filtering: one takeover in ten runs")
filter_runs = 10                     # a structural count: runs in the vote example
v("filter_runs", filter_runs, "{:.0f}", "runs in the batch")
v("filter_kept", filter_runs - 1, "{:.0f}", "runs the policy completed")
v("filter_pct", 100 * (filter_runs - 1) / filter_runs, "{:.0f}", "with the takeover kept, %")
v("filter_dropped_pct", 100 * (filter_runs - 1) / (filter_runs - 1), "{:.0f}", "with the takeover dropped, %")

print("== Offline illusion (Mandlekar et al. 2021, Appendix G)")
v("sq_best", float(LIT.RobomimicSquareBestSuccessPct), "{:.1f}", "Square: best checkpoint, %")
v("sq_low", float(LIT.RobomimicSquareLowestValLossSuccessPct), "{:.1f}", "Square: lowest validation loss, %")
v("tr_best", float(LIT.RobomimicTransportBestSuccessPct), "{:.1f}", "Transport: best checkpoint, %")
v("tr_low", float(LIT.RobomimicTransportLowestValLossSuccessPct), "{:.1f}", "Transport: lowest validation loss, %")

print("== Aisle drift (equation 2)")
v("heading_deg", m(DRIFT.heading_offset, "degree"), "{:.0f}", "heading bias, degrees")
v("aisle_m", m(AISLE.aisle_length, "m"), "{:.0f}", "aisle length, m")
v("v_base", m(DRIFT.speed, "m/s"), "{:.1f}", "drive limit, m/s")
v("clear_cm", m(AISLE.side_clearance, "cm"), "{:.0f}", "side clearance, cm")
v("clear_m", m(AISLE.side_clearance, "m"), "{:.2f}", "side clearance, m")
sci("mse", m(DRIFT.angular_mse, "radian**2"), "{:.2f}", "replay angular MSE, rad^2")
v("t_hit", m(DRIFT.time_to_clearance, "s"), "{:.1f}", "time to spend the clearance, s")
v("d_hit", m(DRIFT.distance_to_clearance, "m"), "{:.1f}", "distance along the aisle, m")
v("d_hit_r", m(DRIFT.distance_to_clearance, "m"), "{:.0f}", "the same, whole meters (the answer choice)")

print("== Stiff contact: F = k x")
sci("k_contact", m(CONTACT.stiffness, "N/m"), "{:.0f}", "contact stiffness, N/m")
v("residual_mm", m(CONTACT.residual, "mm"), "{:.0f}", "residual, mm")
v("residual_n", m(CONTACT.residual_force, "N"), "{:,.0f}", "force at the residual, N")
v("trip_pen_mm", m(CONTACT.tripwire_penetration, "mm"), "{:.2f}", "penetration at the tripwire, mm")
v("lim_pen_mm", m(CONTACT.limit_penetration, "mm"), "{:.1f}", "penetration at the latch limit, mm")

print("== Clean-run bound (equation 3)")
v("conf_pct", 100 * CAMPAIGN.confidence, "{:.0f}", "one-sided confidence, %")
v("alpha", CAMPAIGN.alpha, "{:.2f}", "significance level")
v("alpha_pct", 100 * CAMPAIGN.alpha, "{:.0f}", "significance level, %")
v("alpha_inv", 1 / CAMPAIGN.alpha, "{:.0f}", "one over the significance level")
v("alpha_inv_words", 1 / CAMPAIGN.alpha, "words", "the same, spelled")
v("lb", CAMPAIGN.lower_bound, "{:.3f}", "success lower limit after the clean runs")
v("lb_pct", 100 * CAMPAIGN.lower_bound, "{:.1f}", "the same, %")
v("fail_up_pct", 100 * CAMPAIGN.failure_upper, "{:.1f}", "failure upper limit, %")
v("one_in_words", round(1 / CAMPAIGN.failure_upper), "words", "one run in this many, spelled")
v("weak_one_in_words", CAMPAIGN.weak_policy_one_in, "words", "the weak policy fails one run in this many")
v("weak_clean_pct", 100 * CAMPAIGN.weak_policy_clean_probability, "{:.1f}", "weak policy logs a clean campaign, %")
v("ln_inv_alpha", CAMPAIGN.ln_inv_alpha, "{:.3f}", "ln(1/alpha)")
v("rule3", RULE_OF_THREE_NUMERATOR, "{:.0f}", "the rule of three's numerator")
v("rule3_bound", CAMPAIGN.rule_of_three, "{:.2f}", "rule of three after the clean runs")

print("== Nines ladder (equation 5, Napkin Math 7.2)")
v("t90_pct", 100 * NINE_90, "{:.0f}", "first nine, %")
v("t99_pct", 100 * NINE_99, "{:.0f}", "second nine, %")
v("t999_pct", 100 * NINE_999, "{:.1f}", "third nine, %")
v("n90", CAMPAIGN.runs_for(NINE_90), "{:,.0f}", "runs for 90%")
v("n99", CAMPAIGN.runs_for(NINE_99), "{:,.0f}", "runs for 99%")
v("n999", CAMPAIGN.target_runs, "{:,.0f}", "runs for 99.9%")
v("opening_min", m(CAMPAIGN.campaign_time(), "minute"), "{:.1f}", "the clean runs, min")
v("h99", m(CAMPAIGN.campaign_time(CAMPAIGN.runs_for(NINE_99)), "hour"), "{:.1f}", "cell time for 99%, h")
v("h999", m(CAMPAIGN.campaign_time(CAMPAIGN.target_runs), "hour"), "{:.1f}", "cell time for 99.9%, h")
v("op_h999", m(CAMPAIGN.operator_time(CAMPAIGN.target_runs), "hour"), "{:.1f}", "operator-hours for 99.9%")
v("operators_words", CAMPAIGN.operators, "words", "safety operators, spelled")
v("shifts_words", math.floor(CAMPAIGN.shifts(CAMPAIGN.target_runs)), "words", "whole operator shifts, spelled")

print("== Estimate: 99.5 percent")
v("est_target", CAMPAIGN.estimate_target, "{:.3f}", "estimate target")
v("est_target_pct", 100 * CAMPAIGN.estimate_target, "{:.1f}", "estimate target, %")
v("est_rate", CAMPAIGN.estimate_target_rate, "{:.3f}", "estimate failure rate")
v("est_r3", CAMPAIGN.estimate_rule_of_three_runs, "{:.0f}", "rule-of-three runs")
v("est_n", CAMPAIGN.estimate_runs, "{:.0f}", "exact runs")
v("est_h", m(CAMPAIGN.campaign_time(CAMPAIGN.estimate_runs), "hour"), "{:.1f}", "cell time, h")
v("lb_one_fail_pct", 100 * CAMPAIGN.lower_bound_with_failures, "{:.1f}", "lower limit with one failure, %")
v("obs_one_fail_pct", 100 * CAMPAIGN.observed_rate_with_failures, "{:.0f}", "observed rate with one failure, %")

print("== Sequential test (equations 7, 8)")
v("sprt_p0", SPRT.p_unacceptable, "{:.2f}", "unacceptable failure rate")
v("sprt_p1", SPRT.p_acceptable, "{:.2f}", "acceptable failure rate")
v("sprt_p0_pct", 100 * SPRT.p_unacceptable, "{:.0f}", "unacceptable failure rate, %")
v("sprt_p1_pct", 100 * SPRT.p_acceptable, "{:.0f}", "acceptable failure rate, %")
v("sprt_alpha", SPRT.alpha, "{:.2f}", "alpha")
v("sprt_beta", SPRT.beta, "{:.2f}", "beta")
v("sprt_up", SPRT.success_increment, "{:.4f}", "increment per clean run")
v("sprt_down", -SPRT.failure_increment, "{:.3f}", "decrement per failure")
v("sprt_accept", SPRT.accept, "{:.2f}", "accept boundary")
v("sprt_reject", -SPRT.reject, "{:.2f}", "reject boundary, magnitude")
v("sprt_n_accept", SPRT.runs_to_accept, "{:.0f}", "clean runs to accept")
v("sprt_trace_n", SPRT.trace_clean_runs, "{:.0f}", "clean runs before the failure")
v("sprt_trace", SPRT.trace_score, "{:.2f}", "score after the trace")
v("sprt_fixed_n", SPRT.fixed_runs, "{:.0f}", "fixed zero-failure runs for 90%")
v("sprt_fixed_reject_pct", 100 * SPRT.fixed_rejects_acceptable, "{:.1f}", "fixed test rejects an acceptable policy, %")
v("looks_words", CAMPAIGN.looks, "words", "independent looks, spelled")
v("any_fp_pct", 100 * CAMPAIGN.any_false_positive, "{:.1f}", "at least one false positive, %")

print("== Comparing two checkpoints")
v("arm_n", COMPARE.arm_runs, "{:.0f}", "runs per arm")
v("arm_a", COMPARE.successes_a, "{:.0f}", "successes, arm A")
v("arm_b", COMPARE.successes_b, "{:.0f}", "successes, arm B")
v("arm_fail", COMPARE.failures, "{:.0f}", "failures in both arms")
v("arm_total", COMPARE.total_runs, "{:.0f}", "runs in both arms")
v("one_arm_p", COMPARE.all_in_one_arm, "{:.3f}", "all failures in one arm")
v("fisher_p", COMPARE.fisher_p, "{:.3f}", "Fisher two-sided p")
v("p_high_pct", 100 * COMPARE.p_high, "{:.0f}", "better policy, %")
v("p_low_pct", 100 * COMPARE.p_low, "{:.0f}", "worse policy, %")
v("power_pct", 100 * COMPARE.power, "{:.0f}", "power, %")
v("runs_per_arm", COMPARE.runs_per_arm, "{:.0f}", "runs per arm to separate them")
v("tri_trials", float(LIT.TriLbmRealWorldTrials), "{:,.0f}", "TRI real-world trials")
v("simpler_episodes", float(LIT.SimplerEvaluationEpisodes), "{:,.0f}", "SIMPLER evaluation episodes")

print("== Exposure wall (equation 9, Napkin Math 7.3)")
v("site_e", power_of_ten(m(SITE_RATE, "1/hour")), "{:.0f}", "site requirement, power of ten per hour")
v("e6_e", power_of_ten(m(RATE_E6, "1/hour")), "{:.0f}", "teaching target, power of ten per hour")
v("e9_e", power_of_ten(m(RATE_E9, "1/hour")), "{:.0f}", "catastrophic band, power of ten per hour")
v("site_h", m(WALL.exposure(SITE_RATE), "hour"), "{:,.0f}", "hours at the site requirement")
v("site_years", m(WALL.exposure(SITE_RATE), "year"), "{:.2f}", "the same, years of one machine")
v("e6_mh", m(WALL.exposure(RATE_E6), "hour") / 10 ** 6, "{:.1f}", "hours at 1e-6, millions")
v("e6_years", m(WALL.exposure(RATE_E6), "year"), "{:.0f}", "machine-years at 1e-6")
v("e9_bh", m(WALL.exposure(RATE_E9), "hour") / 10 ** 9, "{:.1f}", "hours at 1e-9, billions")
v("rigs", WALL.rigs, "{:.0f}", "parallel rigs")
v("e9_fleet_years", m(WALL.fleet_time(RATE_E9), "year"), "{:,.0f}", "years for the rigs at 1e-9")
v("rig_year_h", m(WALL.rig_time, "hour"), "{:,.0f}", "one rig-year, h")
sci("rig_rate", m(WALL.rig_rate, "1/hour"), "{:.1f}", "rate one rig-year bounds, per hour")
v("rig_short_e6", WALL.shortfall(RATE_E6), "{:.0f}", "one rig-year short of 1e-6 by")

print("== Road vehicles: RAND against Waymo")
v("waymo_m", ROAD.fleet_miles, "{:.1f}", "Waymo rider-only miles, millions")
v("rand_fatal_m", ROAD.zero_failure["fatalities"], "{:.0f}", "fatalities, zero-failure miles, millions")
v("rand_injury_m", ROAD.zero_failure["injuries"], "{:.1f}", "injuries, millions")
v("rand_crash_m", ROAD.zero_failure["crashes"], "{:.1f}", "crashes, millions")
v("rand_fatal_rate", ROAD.rates["fatalities"], "{:.2f}", "fatality rate per 100 million miles")
v("rand_fatal_short", ROAD.shortfall("fatalities"), "{:.1f}", "fatality requirement over Waymo's miles")
v("rand_precision_bm", ROAD.precision["fatalities"] / 10 ** 3, "{:.1f}", "fatality precision row, billions")
v("rand_precision_x", ROAD.precision_over_fleet("fatalities"), "{:.0f}", "precision row over Waymo's miles")
v("waymo_injury_pct", 100 * ROAD_LIT.WaymoInjuryCrashReduction, "{:.0f}", "fewer injury crashes, %")
v("waymo_airbag_pct", 100 * ROAD_LIT.WaymoAirbagCrashReduction, "{:.0f}", "fewer airbag-deployment crashes, %")

print("== Mean against peak")
v("f_mean", m(TAIL.mean_force, "N"), "{:.0f}", "mean latch force, N")
v("f_p99", m(TAIL.p99_force, "N"), "{:.0f}", "99th-percentile peak, N")
v("tail_trials_words", TAIL.trials, "words", "door trials, spelled")

print("== Wrap-up: the perfect-record fallacy")
v("fallacy_n", CAMPAIGN.fallacy_runs, "{:,.0f}", "clean runs in the fallacy")
v("fallacy_n_exp", CAMPAIGN.fallacy_runs, "{:.0f}", "the same, without a separator (inside an exponent)")
v("fallacy_lb_pct", 100 * CAMPAIGN.fallacy_lower_bound, "{:.2f}", "their lower limit, %")
v("fallacy_lb", CAMPAIGN.fallacy_lower_bound, "{:.4f}", "their lower limit")

if __name__ == "__main__":
    write(__file__)
