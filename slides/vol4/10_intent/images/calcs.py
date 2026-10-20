#!/usr/bin/env python3
"""Every number on the Lecture 10 slides and notes (Grounded Intent), read from MLSysIM.

MLSysIM holds the inputs (the Embodied.* running machine and its aisle
record, Literature.* for AutoRT, OK-Robot, KnowNo, the published action
models, and Bohannon's walking speeds) and the calculations (Scenarios.*,
backed by the solvers in engine/solvers/grounding.py); the chapter reads the
same scenarios. This file only selects values, converts units, and formats
them; it types no numbers. Each value is recorded with v() under the key the
deck prints as \\val{key}; running this file writes images/numbers.tex.
make_figs.py imports it for the scenario objects and the published rows.

Run from this folder: python3 calcs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Embodied, Scenarios, Literature = mlsysim.Embodied, mlsysim.Scenarios, mlsysim.Literature
Q_ = mlsysim.core.units.Q_

RM = Embodied.MobileManipulator.WarehouseMobileManipulator
AISLE = Embodied.Scenario.WarehouseAisle
AUTORT = Literature.RobotGrounding
VLA = Literature.VlaSystems
WALK = Literature.HumanMotion

CADENCE = Scenarios.ConveyorMugIntentCadence
SETTLING = Scenarios.ConveyorMugSettlingDrift
SLOW_BELT = Scenarios.ConveyorMugSlowBelt
REACH = Scenarios.ConveyorMugDynamicReach
OCCUPANCY = Scenarios.AisleCoworkerForwardOccupancy
EXPIRY = Scenarios.ConveyorMugIntentLeaseExpiry
GOAL_SET = Scenarios.ConveyorMugGoalSetOffsets
BESIDE = Scenarios.MugJarReferentSet
LOW_JAR = Scenarios.MugJarReferentSetLowJar
ALONE = Scenarios.JarAloneReferentSet
REGION = Scenarios.DistantHandleConfidenceRegion
LATCH = Scenarios.LatchUnbrakedStall


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


def period_ms(rate_hz):
    """Period in ms of a published rate in Hz."""
    return m(1 / Q_(float(rate_hz), "Hz"), "ms")


cad, reach, occ, exp = CADENCE.result, REACH.result, OCCUPANCY.result, EXPIRY.result
beside, low_jar, alone, region = BESIDE.result, LOW_JAR.result, ALONE.result, REGION.result

print("== Hook: AutoRT's task filter (Ahn et al. 2024, Sec. 5.3)")
passed = AUTORT.AutoRtKeptTasks - AUTORT.AutoRtKeptAcceptableTasks
assert passed == AUTORT.AutoRtUnsuitableTasks - AUTORT.AutoRtRejectedUnsuitable
assert AUTORT.AutoRtUnsuitableTasks == AUTORT.AutoRtSampledTasks - AUTORT.AutoRtAcceptableTasks
v("autort_sampled", AUTORT.AutoRtSampledTasks, "{:.0f}", "tasks in the human-judged sample")
v("autort_acceptable", AUTORT.AutoRtAcceptableTasks, "{:.0f}", "acceptable before the filter")
v("autort_kept", AUTORT.AutoRtKeptTasks, "{:.0f}", "tasks the filter kept")
v("autort_kept_ok", AUTORT.AutoRtKeptAcceptableTasks, "{:.0f}", "acceptable among the kept")
v("autort_unsuitable", AUTORT.AutoRtUnsuitableTasks, "{:.0f}", "unsuitable tasks")
v("autort_rejected", AUTORT.AutoRtRejectedUnsuitable, "{:.0f}", "unsuitable tasks the filter rejected")
v("autort_passed", passed, "{:.0f}", "unsuitable tasks the filter passed")
v("autort_pct_before", 100 * AUTORT.AutoRtAcceptableTasks / AUTORT.AutoRtSampledTasks, "{:.0f}",
  "acceptable share before the filter, %")
v("autort_pct_after", 100 * AUTORT.AutoRtKeptAcceptableTasks / AUTORT.AutoRtKeptTasks, "{:.0f}",
  "acceptable share of the kept, %")
v("autort_recall", 100 * AUTORT.AutoRtRejectedUnsuitable / AUTORT.AutoRtUnsuitableTasks, "{:.0f}",
  "filter recall, %")
v("autort_miss_pct", 100 * passed / AUTORT.AutoRtUnsuitableTasks, "{:.0f}", "filter miss rate, %")

print("== The conveyor mug and the intent model (Napkin Math 10.2, Sec. 10.5)")
v("e0", m(AISLE.grasp_initial_error, "mm"), "{:.0f}", "initial handle error, mm")
v("r_tol", m(AISLE.grasp_tolerance, "mm"), "{:.0f}", "grasp tolerance, mm")
v("budget", m(cad.margin, "mm"), "{:.0f}", "error budget, mm")
v("v_belt_m", m(AISLE.v_conveyor, "m/s"), "{:.2f}", "belt speed, m/s")
v("v_belt_mm", m(AISLE.v_conveyor, "mm/s"), "{:.0f}", "belt speed, mm/s")
v("v_belt_mm_per_ms", m(AISLE.v_conveyor, "mm/ms"), "{:.1f}", "belt speed, mm per ms")
v("a_slip", m(AISLE.a_conveyor_slip, "m/s**2"), "{:.1f}", "residual slip, m/s^2")
v("tau_u", m(cad.untracked_horizon, "ms"), "{:.0f}", "untracked evidence horizon, ms")
v("tau_t", m(cad.tracked_horizon, "ms"), "{:.0f}", "tracked evidence horizon, ms")
v("t_infer", m(RM.intent_inference_latency, "ms"), "{:.0f}", "intent inference, ms (computed)")
v("f_intent", m(RM.intent_rate, "Hz"), "{:.0f}", "intent rate, Hz")
v("n_intent_values_words", RM.intent_model.action_dim, "words", "values one intent inference decodes")
v("t_period", m(cad.intent_period, "ms"), "{:.0f}", "intent period, ms")
v("err_period", m(cad.error_at_period, "mm"), "{:.0f}", "untracked error at one period, mm")
v("infer_over_tau", cad.inference_over_horizon, "{:.1f}", "inference over the untracked horizon")
v("overrun", m(cad.inference_overrun, "ms"), "{:.0f}", "inference overruns the horizon by, ms")
v("t_stall", m(cad.stall, "ms"), "{:.0f}", "host stall, ms")
v("err_stall_u", m(cad.stall_error_untracked, "mm"), "{:.0f}", "untracked error after the stall, mm")
v("err_stall_t", m(cad.stall_error_tracked, "mm"), "{:.0f}", "tracked error after the stall, mm")
v("t_onset", m(cad.stop_onset, "ms"), "{:.0f}", "stop onset after expiry, ms")
v("t_brake", m(AISLE.t_brake_onset, "ms"), "{:.0f}", "brake onset, ms")
v("f_tick_khz", m(RM.permission_rate, "kHz"), "{:.0f}", "permission tick rate, kHz")
v("t_chunk_lease", m(AISLE.t_lease, "ms"), "{:.0f}", "chunk lease, ms")

print("== Estimate: model-only renewal (Napkin Math 10.2, Checkpoint 10.1)")
v("renew_age", m(cad.renewal_age, "ms"), "{:.0f}", "oldest evidence behind a model-only target, ms")
v("renew_age_s", m(cad.renewal_age, "s"), "{:.3f}", "the same, s")
v("v_renew", m(cad.renewal_speed_ceiling, "mm/s"), "{:.1f}", "drift a model-only renewal can follow, mm/s")
v("renew_pct", 100 * cad.renewal_share, "{:.1f}", "as a share of the belt speed, %")
v("v_settling", m(SETTLING.target_speed, "mm/s"), "{:.0f}", "settling drift, mm/s")
v("tau_settling", m(SETTLING.result.untracked_horizon, "ms"), "{:.0f}", "its horizon, ms")
v("v_slow_belt", m(SLOW_BELT.target_speed, "mm/s"), "{:.0f}", "slowed belt, mm/s")
v("tau_slow_belt", m(SLOW_BELT.result.untracked_horizon, "ms"), "{:.0f}", "its horizon, ms")

print("== Goal set is a specification (Sec. 10.1, equation 1)")
d_in, d_out = GOAL_SET.offsets
n_in, n_out = GOAL_SET.norms
v("d_goal_in", m(d_in, "mm"), "{:.0f}", "inside offset, mm")
v("d_goal_out", m(d_out, "mm"), "{:.0f}", "outside offset, mm")
v("norm_in", n_in, "{:.2f}", "normalized distance, inside")
v("norm_out", n_out, "{:.2f}", "normalized distance, outside")

print("== Conformal refusal: the jar beside the mug (Sec. 10.3.1)")
v("score_mug", BESIDE.scores[0], "{:.2f}", "raw mug score")
v("score_jar", BESIDE.scores[1], "{:.2f}", "raw jar score")
v("score_jar_low", LOW_JAR.scores[1], "{:.2f}", "raw jar score, contrast case")
v("n_cal", BESIDE.calibration_count, "{:.0f}", "calibration scenes")
v("n_cal_plus", BESIDE.calibration_count + 1, "{:.0f}", "calibration scenes plus one")
v("alpha", BESIDE.alpha, "{:.2f}", "allowed miscoverage")
v("one_minus_alpha", 1 - BESIDE.alpha, "{:.1f}", "coverage level")
v("rank", beside.rank, "{:.0f}", "calibration rank of qhat")
v("qhat", BESIDE.threshold_score, "{:.2f}", "assumed qhat")
v("thresh", beside.threshold, "{:.2f}", "inclusion threshold 1 - qhat")
v("p_mug", beside.normalized[0], "{:.2f}", "normalized mug")
v("p_jar", beside.normalized[1], "{:.2f}", "normalized jar")
v("set_size", beside.set_size, "{:.0f}", "prediction-set size")
v("set_size_words", beside.set_size, "words", "the same, in words")
v("p_mug_low", low_jar.normalized[0], "{:.2f}", "normalized mug, contrast case")
v("p_jar_low", low_jar.normalized[1], "{:.2f}", "normalized jar, contrast case")
v("set_size_low", low_jar.set_size, "{:.0f}", "prediction-set size, contrast case")
v("entropy", beside.entropy, "{:.3f}", "entropy, nats")
v("max_entropy", beside.max_entropy, "{:.3f}", "two-candidate maximum, nats")
v("p_jar_alone", alone.normalized[0], "{:.2f}", "the jar alone, normalized")

print("== KnowNo on hardware (Ren et al. 2023, Table 2 and Fig. 1)")
KN = Literature.RobotGrounding
v("knowno_thresh", KN.KnowNoExampleThreshold, "{:.2f}", "calibrated threshold in the example")
v("knowno_target", KN.KnowNoTargetSuccess, "{:.2f}", "target success level")
v("knowno_succ", KN.KnowNoSuccess, "{:.2f}", "KnowNo task success")
v("knowno_help_pct", 100 * KN.KnowNoHelpRate, "{:.0f}", "KnowNo help rate, %")
v("simple_succ", KN.SimpleSetSuccess, "{:.2f}", "Simple Set task success")
v("simple_help_pct", 100 * KN.SimpleSetHelpRate, "{:.0f}", "Simple Set help rate, %")
v("nohelp_succ", KN.NoHelpSuccess, "{:.2f}", "No Help task success")
v("help_gap_pts", 100 * (KN.SimpleSetHelpRate - KN.KnowNoHelpRate), "{:.0f}",
  "help-rate reduction against the simple set, points")

print("== Imprecise estimate: the distant view (Sec. 10.3.3)")
sx, sy, sz = REGION.sigmas
ax_x, ax_y, ax_z = region.semi_axes
v("sigma_x", m(sx, "mm"), "{:.0f}", "lateral std. dev., mm")
v("sigma_y", m(sy, "mm"), "{:.0f}", "lateral std. dev., mm")
v("sigma_z", m(sz, "mm"), "{:.0f}", "depth std. dev., mm")
v("chi2", region.quantile, "{:.3f}", "chi-square quantile, 3 dof, 0.95")
v("coverage_pct", 100 * REGION.coverage, "{:.0f}", "region coverage, %")
v("semi_x", m(ax_x, "mm"), "{:.1f}", "95% semi-axis, mm")
v("semi_y", m(ax_y, "mm"), "{:.1f}", "95% semi-axis, mm")
v("semi_z", m(ax_z, "mm"), "{:.1f}", "95% depth semi-axis, mm")
v("depth_over_tol", region.worst_over_tolerance, "{:.1f}", "depth semi-axis over the tolerance")
v("axes_within_words", region.axes_within, "words", "axes inside the tolerance")

print("== Action tokens (RT-1 Sec. 5.1; OpenVLA Sec. 3.2)")
assert VLA.Rt1ActionBins == VLA.OpenVlaActionBins
v("n_bins", VLA.OpenVlaActionBins, "{:.0f}", "bin edges per action dimension")
v("n_bin_intervals", VLA.OpenVlaActionBins - 1, "{:.0f}", "bins between the edges")

print("== Latch ceilings (Napkin Math 10.1)")
v("f_trip", m(AISLE.f_latch_tripwire, "N"), "{:.0f}", "tripwire, N")
v("v_latch", m(LATCH.guarded_speed, "m/s"), "{:.2f}", "guarded approach, m/s")
v("k_latch", m(AISLE.k_latch, "N/mm"), "{:.0f}", "plate stiffness, N/mm")
v("t_contact", m(AISLE.t_contact_response, "ms"), "{:.0f}", "contact loop, ms")
v("f_limit", m(AISLE.f_latch_limit, "N"), "{:.0f}", "latch limit, N")
v("force_rate", m(LATCH.force_rate(LATCH.guarded_speed), "N/ms"), "{:.0f}", "force rise, N per ms")
v("f_rise", m(LATCH.force_rate(LATCH.guarded_speed) * AISLE.t_contact_response, "N"), "{:.0f}",
  "rise during the contact loop, N")
v("f_brake", m(LATCH.force_at_brake_onset_guarded, "N"), "{:.0f}", "force at the brake command, N")
v("t_stall_arm", m(LATCH.stall, "ms"), "{:.0f}", "unbraked stall, ms")
v("x_stall", m(LATCH.stall_travel, "mm"), "{:.0f}", "stall travel, mm")
v("f_stall", m(LATCH.stall_force, "N"), "{:,.0f}", "stall force, N")
v("stall_over_limit", LATCH.stall_over_limit, "{:.0f}", "stall force over the limit")

print("== Dynamic reachability (Sec. 10.6, Definition 10.2)")
v("d_target", m(REACH.target_distance, "mm"), "{:.0f}", "gripper-to-handle distance, mm")
v("t_transit", m(reach.transit_floor, "ms"), "{:.0f}", "transit floor, ms")
v("v_tcp", m(REACH.speed_ceiling, "m/s"), "{:.0f}", "TCP speed ceiling, m/s")
v("reach_u", m(reach.reach_untracked, "mm"), "{:.0f}", "reach inside the untracked window, mm")
v("shortfall", m(reach.shortfall, "ms"), "{:.0f}", "shortfall against the tracked window, ms")
v("n_captures_words", reach.fresh_captures, "words", "fresh captures a staged approach needs")

print("== Forward occupancy and the freezing robot (Napkin Math 10.3)")
v("v_human", m(AISLE.v_human_approach, "m/s"), "{:.1f}", "ISO 13855 approach speed, m/s")
v("v_human_mm", m(AISLE.v_human_approach, "mm/s"), "{:,.0f}", "the same, mm/s")
v("d_occ_u", m(occ.growth_untracked, "mm"), "{:.0f}", "growth over the untracked horizon, mm")
v("d_occ_t", m(occ.growth_tracked, "mm"), "{:.0f}", "growth over the tracked horizon, mm")
v("d_occ_long", m(occ.growth_forecast, "m"), "{:.1f}", "growth over the forecast, m")
v("forecast_words", m(OCCUPANCY.forecast, "s"), "words", "forecast horizon, s")
v("forecast_s", m(OCCUPANCY.forecast, "s"), "{:.0f}", "the same, figure label")
v("occ_over_aisle", occ.forecast_over_aisle, "{:.1f}", "forecast set over the aisle width")
v("aisle_w", m(AISLE.aisle_width, "m"), "{:.2f}", "aisle width, m")
v("t_block", m(occ.aisle_span_horizon, "s"), "{:.2f}", "horizon at which one set spans the aisle, s")

print("== Measured walking speeds (Bohannon 1997, reference table)")
WALK_ROWS = [  # (group, comfortable m/s, maximum m/s)
    ("men 20s", WALK.WalkComfortableMen20sCmS, WALK.WalkMaximumMen20sCmS),
    ("men 50s", WALK.WalkComfortableMen50sCmS, WALK.WalkMaximumMen50sCmS),
    ("women 20s", WALK.WalkComfortableWomen20sCmS, WALK.WalkMaximumWomen20sCmS),
    ("women 50s", WALK.WalkComfortableWomen50sCmS, WALK.WalkMaximumWomen50sCmS)]
WALK_ROWS = [(g, m(Q_(float(c), "cm/s"), "m/s"), m(Q_(float(x), "cm/s"), "m/s")) for g, c, x in WALK_ROWS]
comfortable = [c for _, c, _ in WALK_ROWS]
fastest = [x for _, _, x in WALK_ROWS]
v("walk_comf", sum(comfortable) / len(comfortable), "{:.1f}", "comfortable walking, mean of the groups, m/s")
v("walk_max_lo", min(fastest), "{:.1f}", "maximum walking, slowest group, m/s")
v("walk_max_hi", max(fastest), "{:.1f}", "maximum walking, fastest group, m/s")
v("walk_max_over_iso", max(fastest) / m(AISLE.v_human_approach, "m/s"), "{:.2f}",
  "fastest maximum over the ISO design speed")

print("== Published action-model rates as periods")
VLA_ROWS = [  # (label, low period ms, high period ms)
    ("RT-2 55B", period_ms(VLA.Rt2LargeRateHighHz), period_ms(VLA.Rt2LargeRateLowHz)),
    ("RT-1", period_ms(VLA.Rt1RateHz), period_ms(VLA.Rt1RateHz)),
    ("RT-2 5B", period_ms(VLA.Rt2SmallRateHz), period_ms(VLA.Rt2SmallRateHz)),
    ("OpenVLA 7B", period_ms(VLA.OpenVlaRtx4090RateHz), period_ms(VLA.OpenVlaRtx4090RateHz)),
    ("pi0 chunk", float(VLA.Pi0OnboardLatencyMs), float(VLA.Pi0OnboardLatencyMs)),
]
v("rt1_hz", VLA.Rt1RateHz, "{:.0f}", "RT-1 rate, Hz")
v("rt2l_lo_hz", VLA.Rt2LargeRateLowHz, "{:.0f}", "RT-2 55B rate, low end, Hz")
v("rt2l_hi_hz", VLA.Rt2LargeRateHighHz, "{:.0f}", "RT-2 55B rate, high end, Hz")
v("rt2s_hz", VLA.Rt2SmallRateHz, "{:.0f}", "RT-2 5B rate, Hz")
v("openvla_hz", VLA.OpenVlaRtx4090RateHz, "{:.0f}", "OpenVLA rate, Hz")
v("pi0_chunk", VLA.Pi0ChunkHorizon, "{:.0f}", "pi0 chunk, steps")
v("pi0_ms", VLA.Pi0OnboardLatencyMs, "{:.0f}", "pi0 chunk inference, ms")
v("pi0_rate", VLA.Pi0ControlRateHz, "{:.0f}", "pi0 streaming rate, Hz")
v("rt2l_period_lo", VLA_ROWS[0][1], "{:,.0f}", "RT-2 55B period, low, ms")
v("rt2l_period_hi", VLA_ROWS[0][2], "{:,.0f}", "RT-2 55B period, high, ms")
v("rt1_period", VLA_ROWS[1][1], "{:,.0f}", "RT-1 period, ms")
v("rt2s_period", VLA_ROWS[2][1], "{:,.0f}", "RT-2 5B period, ms")
v("openvla_period", VLA_ROWS[3][1], "{:,.0f}", "OpenVLA period, ms")
v("pi0_over_tau", ratio(Q_(float(VLA.Pi0OnboardLatencyMs), "ms"), cad.untracked_horizon), "{:.1f}",
  "pi0 chunk inference over the untracked horizon")

print("== OK-Robot failure analysis (Liu et al. 2024, Figs. 1 and 4)")
OKR = Literature.RobotGrounding
OKR_ROWS = dict(trials=OKR.OkRobotTrials, success=OKR.OkRobotSuccesses,
                nav_wrong_object=OKR.OkRobotWrongOrNoObject, nav_other=OKR.OkRobotOtherNavigation,
                grasp_model=OKR.OkRobotGraspModel, hardware=OKR.OkRobotHardware, placing=OKR.OkRobotPlacing)
assert sum(x for k, x in OKR_ROWS.items() if k != "trials") == OKR_ROWS["trials"]
v("okr_trials", OKR.OkRobotTrials, "{:.0f}", "trials")
v("okr_success", OKR.OkRobotSuccesses, "{:.0f}", "successes")
v("okr_success_pct", 100 * OKR.OkRobotSuccesses / OKR.OkRobotTrials, "{:.1f}", "success share, %")
v("okr_wrong", OKR.OkRobotWrongOrNoObject, "{:.0f}", "wrong or no object")
v("okr_wrong_pct", 100 * OKR.OkRobotWrongOrNoObject / OKR.OkRobotTrials, "{:.1f}", "its share, %")
v("okr_grasp", OKR.OkRobotGraspModel, "{:.0f}", "grasp-model failures")
v("okr_grasp_seg", OKR.OkRobotGraspWrongSegment, "{:.0f}", "of which the segmenter picked the wrong object")
v("okr_homes_pct", OKR.OkRobotHomesSuccessPct, "{:.1f}", "success in homes as found, %")
v("okr_declutter_pct", OKR.OkRobotDeclutteredSuccessPct, "{:.1f}", "success in decluttered homes, %")

print("== The red-mug intent lease (Sec. 10.8, equation 9)")
v("t_admin", m(EXPIRY.administrative_lease, "ms"), "{:.0f}", "administrative lease, ms")
v("d_clear_arm", m(EXPIRY.clearance, "mm"), "{:.0f}", "clearance ahead of the gripper, mm")
v("a_arm", m(exp.credible_deceleration, "m/s**2"), "{:.1f}", "credible arm deceleration, m/s^2")
v("d_stop_arm", m(exp.stop_distance, "mm"), "{:.0f}", "stop from the TCP ceiling, mm")
v("t_stop_deadline", m(exp.stopping_deadline, "ms"), "{:.0f}", "stopping deadline after admission, ms")
v("t_expiry_t", m(exp.expiry_tracked, "ms"), "{:.0f}", "tracked expiry, ms")
v("t_expiry_u", m(exp.expiry_untracked, "ms"), "{:.0f}", "untracked expiry, ms")
print(f"   binding term: untracked {exp.binding_untracked}, tracked {exp.binding_tracked}")


if __name__ == "__main__":
    write(__file__)
