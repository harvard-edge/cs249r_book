#!/usr/bin/env python3
"""Every number on the Lecture 15 slides and notes (Adversarial Verification), read from MLSysIM.

MLSysIM holds the inputs (the Embodied.* running machine and its aisle record;
Literature.* for Bookout v. Toyota and NASA's tin whisker, DriveFI, the
adversarial patch, RoboPAIR, Wang et al.'s OpenVLA table, VNN-COMP 2025's
scored benchmarks, and O'Kelly et al.'s rare-event table) and the
calculations: the finished stopping budgets (Scenarios.WarehouseFinishedStop*),
the renewal- and flag-referenced oracles and the onset trials scored against
them (Scenarios.WarehouseRenewal*, WarehouseFlagReferencedStop), the shared
drive bus (WarehouseDriveBusStop), the permission-rail hold-up
(WarehousePermissionRailHoldUp), the diagnostic-coverage campaign
(WarehouseDiagnosticCoverage*), the queue-delay rare event
(WarehouseQueueDelaySpareExhaustion), and the exposure wall
(HazardExposureWall). The chapter reads the same scenarios. This file only
selects values, converts units, and formats them; it types no numbers. Each
value is recorded with v() under the key the deck prints as \\val{key};
running this file writes images/numbers.tex. make_figs.py and make_record.py
import it for the scenario objects and plot inputs.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

from mlsysim.physics.drivetrain import calc_required_friction, calc_traction_deceleration_limit  # noqa: E402

Embodied, Scenarios, Literature, Models = mlsysim.Embodied, mlsysim.Scenarios, mlsysim.Literature, mlsysim.Models
Q_ = mlsysim.core.units.Q_

RM = Embodied.MobileManipulator.WarehouseMobileManipulator
AISLE = Embodied.Scenario.WarehouseAisle
REPORTS = Literature.IncidentReports
STUDIES = Literature.VerificationStudies
ROBUST = Literature.RobustnessStudies

STOP = Scenarios.WarehouseFinishedStopAtAisleSpeed
DISPATCH = Scenarios.WarehouseFinishedStopWithDispatch
INSPECTED = Scenarios.WarehouseFinishedStopInspectedFloor
RENEWAL = Scenarios.WarehouseRenewalReferencedStop
RESTRICTED = Scenarios.WarehouseRenewalReferencedStopRestricted
FLAG = Scenarios.WarehouseFlagReferencedStop
LATE = Scenarios.WarehouseRenewalStopLateOnsetTrial
CHK = Scenarios.WarehouseRenewalStopNinetyMsOnsetTrial
HOLDUP = Scenarios.WarehousePermissionRailHoldUp
BUS = Scenarios.WarehouseDriveBusStop
COV = Scenarios.WarehouseDiagnosticCoverageHundredFit
COV_TIGHT = Scenarios.WarehouseDiagnosticCoverageTenFit
RARE = Scenarios.WarehouseQueueDelaySpareExhaustion
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


stop, dispatch, insp = STOP.result, DISPATCH.result, INSPECTED.result
renewal, restricted, late, chk = RENEWAL.result, RESTRICTED.result, LATE.result, CHK.result
holdup, bus = HOLDUP.result, BUS.result

# Inputs for make_figs.py and make_record.py, SI magnitudes at full precision.
V_AISLE = m(AISLE.v_aisle, "m/s")
A_BRAKE = m(AISLE.a_brake, "m/s**2")
J_MAX = m(AISLE.j_brake, "m/s**3")
D_CLEAR = m(AISLE.d_clear, "m")
MU_INSPECTED = AISLE.mu_inspected_floor
MU_KNEE = calc_required_friction(AISLE.a_brake)
V_RESTRICTED = m(RESTRICTED.speed, "m/s")
PAYLOAD = m(RM.onboard_payload_capacity, "kg")
BUS_NOMINAL = m(RM.drive_bus_nominal, "V")
ONSET = m(RENEWAL.delay, "s")
ONSET_LATE = m(LATE.onset, "s")
T_LEASE = m(AISLE.t_lease, "s")
T_TICK = m(AISLE.t_tick, "s")
T_BUS = m(AISLE.t_bus, "s")
T_PROBE = m(LATE.probe_at, "s")
LOC_MARGIN = m(stop.overhead, "m")
TRACK = m(AISLE.eps_track, "m")
CAPTURE_DELAY_TRAVEL = m(stop.delay_travel, "m")
RESIDENT_STOP = m(stop.braking_travel + stop.jerk_ramp_travel, "m")
RENEWAL_DELAY_TRAVEL = m(renewal.delay_travel, "m")
CAPTURE_BOUND = m(stop.required_distance, "m")
RENEWAL_BOUND = m(renewal.required_distance, "m")
RESTRICTED_BOUND = m(restricted.required_distance, "m")
LATE_TRAVEL = m(late.travel, "m")
INTENT_PARAMS = m(RM.intent_model.parameters, "param")
CHUNK_PARAMS = m(RM.chunk_model.parameters, "param")
VNN_BENCHMARKS = Literature.VnnComp2025ScoredBenchmarks.entries
VNN_MAX_PARAMS = [e["params_high"] for e in VNN_BENCHMARKS]
WANG = Literature.Wang2025OpenVlaLiberoUntargeted.entries
OKELLY = Literature.OKelly2018RareEventTable1.entries
DFI_EXHAUSTIVE_H = m(Q_(STUDIES.DriveFiExhaustiveDays, "day"), "hour")
DFI_BAYES_H = STUDIES.DriveFiBayesianHours
N_MC, N_ZERO, N_IS = RARE.result.monte_carlo_runs, RARE.result.zero_failure_runs, RARE.result.importance_runs


def ceiling_at(mu):
    """Pre-dispatch speed ceiling in m/s on a floor of friction mu (optimistic all-weight screen)."""
    return m(INSPECTED.variant(deceleration=min(AISLE.a_brake, calc_traction_deceleration_limit(mu))).result
             .speed_ceiling, "m/s")


print("== Hook: Bookout v. Toyota (War Story 15.1)")
v("verdict_musd", REPORTS.ToyotaBookoutVerdictUsd / 10 ** 6, "{:.0f}", "jury verdict, million USD")
v("whisker_ohm", REPORTS.ToyotaTinWhiskerShortOhm, "{:.0f}", "tin-whisker short, ohm")

print("== Recall: the spare at aisle speed (AV 15.2; Lectures 12 to 14)")
v("v_aisle", V_AISLE, "{:.1f}", "aisle speed, m/s")
v("d_clear_mm", m(AISLE.d_clear, "mm"), "{:,.0f}", "clear distance, mm")
v("d_clear_m", D_CLEAR, "{:.2f}", "clear distance, m")
v("budget", m(stop.required_distance, "mm"), "{:.1f}", "capture-referenced subtotal, mm")
v("spare", m(stop.spare, "mm"), "{:.1f}", "spare, mm")
v("spare_mm0", m(stop.spare, "mm"), "{:.0f}", "spare, whole mm")
v("t_spare", m(stop.delay_spare, "ms"), "{:.1f}", "spare as waiting time, ms")
v("t_da", m(AISLE.t_dispatch_to_admission, "ms"), "{:.1f}", "dispatch-to-admission allocation, ms")
v("travel_da", m(stop.travel_over(AISLE.t_dispatch_to_admission), "mm"), "{:.1f}", "its travel, mm")
v("spare_left", m(dispatch.spare, "mm"), "{:.1f}", "spare after the allocation, mm")
v("t_left", m(dispatch.delay_spare, "ms"), "{:.1f}", "waiting time after the allocation, ms")
travel_takeover = stop.travel_over(AISLE.t_takeover_in_loop)
v("t_takeover", m(AISLE.t_takeover_in_loop, "ms"), "{:.0f}", "in-loop takeover allowance, ms")
v("travel_takeover", m(travel_takeover, "mm"), "{:.1f}", "travel during it, mm")
v("takeover_over", m(travel_takeover - stop.spare, "mm"), "{:.1f}", "overrun past the spare, mm")

print("== Faults F1 to F4 (AV 15.1)")
v("chunk_period", m(1 / RM.chunk_rate, "ms"), "{:.0f}", "chunk renewal period, ms")
v("bus_nominal", BUS_NOMINAL, "{:.1f}", "drive bus, nominal, V")
v("bus_uvlo", m(RM.drive_bus_uvlo, "V"), "{:.0f}", "converter lockout, V")

print("== DriveFI (Jha et al., DSN 2019, sec. I and VI)")
v("dfi_faults", STUDIES.DriveFiFaultListSize, "{:,.0f}", "faults in the list")
v("dfi_days", STUDIES.DriveFiExhaustiveDays, "{:.0f}", "exhaustive evaluation, days")
v("dfi_hours", DFI_BAYES_H, "{:.0f}", "Bayesian search, hours (less than)")
v("dfi_found", STUDIES.DriveFiCriticalFaults, "{:.0f}", "safety-critical faults found")
v("dfi_hazard", STUDIES.DriveFiHazardFaults, "{:.0f}", "faults that became hazards")
v("dfi_scenes_haz", STUDIES.DriveFiHazardScenes, "{:.0f}", "hazardous scenes")
v("dfi_scenes", STUDIES.DriveFiScenes, "{:,.0f}", "scenes")
v("dfi_scene_pct", 100 * STUDIES.DriveFiHazardScenes / STUDIES.DriveFiScenes, "{:.2f}", "hazardous share of scenes, %")
v("dfi_random", STUDIES.DriveFiRandomExperiments, "{:,.0f}", "random FI experiments")
v("dfi_gpu", STUDIES.DriveFiGpuInjections, "{:,.0f}", "GPU injections into DriveAV")
v("dfi_sdc", STUDIES.DriveFiGpuSdcPct, "{:.1f}", "silent data corruption, %")
v("dfi_due", STUDIES.DriveFiGpuDuePct, "{:.2f}", "kernel panics and hangs, %")

print("== Envelope: the coupled corner (AV 15.2, 15.5.1; optimistic all-weight screen)")
v("a_brake", A_BRAKE, "{:.0f}", "credible deceleration, m/s^2")
v("mu_knee", MU_KNEE, "{:.3f}", "friction where traction stops binding")
v("mu_insp", MU_INSPECTED, "{:.2f}", "inspected-floor friction")
v("a_insp", m(INSPECTED.deceleration, "m/s**2"), "{:.2f}", "traction-limited deceleration, m/s^2")
v("ceiling_dry", m(stop.speed_ceiling, "m/s"), "{:.2f}", "dry-floor ceiling, m/s")
v("ceiling_insp", m(insp.speed_ceiling, "m/s"), "{:.2f}", "inspected-floor ceiling, m/s")
v("budget_insp_m", m(insp.required_distance, "m"), "{:.2f}", "1.3 m/s on the inspected floor, m")
v("bus_low", m(RM.drive_bus_battery_low, "V"), "{:.0f}", "depleted drive bus, V")
v("stall_ms", m(LATE.onset - RENEWAL.delay, "ms"), "{:.0f}", "memory stall (the late onset), ms")
v("stall_mm", m(late.extra_travel, "mm"), "{:.0f}", "its travel at the aisle speed, mm")
v("v_restricted", V_RESTRICTED, "{:.0f}", "candidate restricted speed, m/s")

print("== Three reference events (AV 15.2, figure 1)")
v("tau_delay", m(AISLE.tau_delay, "ms"), "{:.1f}", "pre-brake delay from capture, ms")
v("capture_travel", m(stop.delay_travel, "mm"), "{:.1f}", "travel before braking from capture, mm")
v("stop_mm", RESIDENT_STOP * 10 ** 3, "{:.1f}", "resident stop at 1.3 m/s, mm")
v("loc_margin", m(stop.overhead, "mm"), "{:.0f}", "localization plus protective margin, mm")
v("eps_track", m(AISLE.eps_track, "mm"), "{:.0f}", "tracking inset, mm")
v("onset", m(RENEWAL.delay, "ms"), "{:.0f}", "onset bound after the last valid renewal, ms")
v("renewal_travel", m(renewal.delay_travel, "mm"), "{:.1f}", "travel before onset, mm")
v("f1_bound", m(renewal.required_distance, "mm"), "{:.1f}", "F1 travel bound at 1.3 m/s, mm")
gap = stop.required_distance - renewal.required_distance
v("gap", m(gap, "mm"), "{:.1f}", "capture subtotal less the F1 bound, mm")

print("== Worked example: the F1 oracle (AV Napkin Math 15.1)")
v("t_lease", m(AISLE.t_lease, "ms"), "{:.0f}", "chunk lease, ms")
v("t_tick", m(AISLE.t_tick, "ms"), "{:.0f}", "permission tick, ms")
v("t_bus", m(AISLE.t_bus, "ms"), "{:.0f}", "fieldbus cycle, ms")
v("t_act", m(AISLE.t_brake_onset, "ms"), "{:.0f}", "stop command to drive onset, ms")
v("j_brake", J_MAX, "{:.2f}", "jerk bound, m/s^3")
v("sfx_v2", m(stop.braking_travel, "mm"), "{:.1f}", "v^2/2a, mm")
v("sfx_jerk", m(stop.jerk_ramp_travel, "mm"), "{:.1f}", "v a/2j, mm")
v("t_age", m(AISLE.perception_age, "ms"), "{:.1f}", "evidence age at dispatch, ms")
v("age_travel", m(stop.travel_over(AISLE.perception_age), "mm"), "{:.1f}", "its travel, mm")
v("late_equiv", m(gap / AISLE.v_aisle, "ms"), "{:.0f}", "the gap as late onset, ms")
v("f1_restricted", m(restricted.required_distance, "mm"), "{:.1f}", "F1 bound at 1.0 m/s on 0.12, mm")
v("oracle_cut_pct", 100 * (1 - m(restricted.required_distance / renewal.required_distance, "")), "{:.1f}",
  "oracle shortening, %")
v("speed_cut_pct", 100 * (1 - m(RESTRICTED.speed / RENEWAL.speed, "")), "{:.1f}", "speed reduction, %")

print("== The late F1 trace (AV 15.9.1, figure 4)")
v("late_ms", m(-late.onset_margin, "ms"), "{:.0f}", "onset lateness, ms")
v("late_rho_mm", m(late.distance_margin, "mm"), "{:.0f}", "distance margin, mm")
v("t_cmd", m(AISLE.t_lease + AISLE.t_tick, "ms"), "{:.0f}", "stop command after the renewal, ms")
v("t_frame", m(AISLE.t_lease + AISLE.t_tick + AISLE.t_bus, "ms"), "{:.0f}", "drive frame after the renewal, ms")
v("rest_on_time", m(late.rest_time_bound, "ms"), "{:,.0f}", "rest, on time, ms")
v("travel_on_time", m(renewal.stop_distance, "mm"), "{:.1f}", "on-time travel to rest, mm")
v("t_probe", m(LATE.probe_at, "ms"), "{:.0f}", "replayed renewal probe, ms")
v("onset_late", m(LATE.onset, "ms"), "{:.0f}", "late onset, ms")
v("travel_late", m(late.travel, "mm"), "{:.1f}", "late travel, mm")
v("rho_onset_n", -late.onset_margin_normalized, "{:.3f}", "normalized onset margin (negative)")
v("rho_dist_n", late.distance_margin_normalized, "{:.4f}", "normalized distance margin")

print("== Diagnose: onset at 90 ms (Checkpoint 15.1)")
v("chk_onset", m(CHK.onset, "ms"), "{:.0f}", "measured onset, ms")
v("chk_rho", m(-chk.onset_margin, "ms"), "{:.0f}", "onset miss, ms")
v("chk_extra", m(chk.extra_travel, "mm"), "{:.1f}", "extra travel before braking, mm")

print("== Supply headroom (AV Napkin Math 15.2)")
v("i_rail", m(AISLE.i_rail_transient, "A"), "{:.0f}", "drive transient, A")
v("bus_r_mohm", m(RM.drive_bus_resistance, "mohm"), "{:.0f}", "source resistance, mOhm")
v("bus_r_ohm", m(RM.drive_bus_resistance, "ohm"), "{:.3f}", "source resistance, ohm")
v("bus_min", m(bus.sustained_depleted, "V"), "{:.1f}", "sustained sag from the depleted bus, V")
v("bus_margin", m(bus.margin_depleted, "V"), "{:.1f}", "headroom above lockout, V")
v("i_cross", m(bus.crossing_current, "A"), "{:.0f}", "step that reaches lockout, A")
v("r_cross", m(bus.crossing_resistance, "mohm"), "{:.0f}", "resistance where the headroom vanishes, mOhm")
v("holdup_s", m(holdup.holdup, "s"), "{:.0f}", "permission-rail hold-up, s")
v("flag_to_set", m(holdup.flag_to_set, "ms"), "{:,.1f}", "F4b flag to brakes set, ms")
v("flag_path", m(holdup.flag_path, "ms"), "{:.0f}", "flag to onset, ms")
v("holdup_stop", m(holdup.stop_time, "ms"), "{:,.1f}", "jerk-limited stop from the inspected ceiling, ms")
v("engage", m(holdup.engage, "ms"), "{:.0f}", "brake engagement, ms")
v("holdup_margin", m(holdup.margin, "ms"), "{:.1f}", "hold-up left, ms")

print("== Published attacks (Brown 2017; Wang 2025, Table 1a; Robey 2025, Tables 2 and 3)")
v("patch_banana", ROBUST.PatchBananaConfidencePct, "{:.0f}", "VGG16 banana confidence, %")
v("patch_toaster", ROBUST.PatchToasterConfidencePct, "{:.0f}", "VGG16 toaster confidence, %")
v("openvla_b", m(Models.Embodied.OpenVLA_7B.parameters, "param") / 10 ** 9, "{:.0f}", "OpenVLA parameters, billions")
wang_success = [100 - e["benign_failure_pct"] for e in WANG]
v("wang_low", min(wang_success), "{:.0f}", "lowest benign success, %")
v("wang_high", max(wang_success), "{:.0f}", "highest benign success, %")
v("wang_patched", min(100 - e["uada_failure_pct"] for e in WANG), "words", "success with the patch, %")
v("robopair_n", ROBUST.RoboPairHarmfulTasks, "{:.0f}", "harmful tasks per robot")
v("go2_direct", ROBUST.RoboPairGo2DirectCompliant, "{:.0f}", "Go2, direct prompt")
v("go2_pair", ROBUST.RoboPairGo2Compliant, "{:.0f}", "Go2, RoboPAIR")
v("jackal_direct", ROBUST.RoboPairJackalDirectCompliant, "{:.0f}", "Jackal, direct prompt")
v("jackal_pair", ROBUST.RoboPairJackalCompliant, "{:.0f}", "Jackal, RoboPAIR")
v("pair_pct", 100 * ROBUST.RoboPairGo2Compliant / ROBUST.RoboPairHarmfulTasks, "{:.0f}", "RoboPAIR success, %")
v("enforcer_deadline", m(AISLE.enforcer_deadline, "us"), "{:.0f}", "permission-check deadline, us")
v("side_clearance", m(AISLE.side_clearance, "m"), "{:.2f}", "side clearance, m")

print("== Proof scale (VNN-COMP 2025, Table 3)")
v("vnn_n", len(VNN_BENCHMARKS), "{:.0f}", "scored benchmarks")
v("vnn_max_m", max(VNN_MAX_PARAMS) / 10 ** 6, "{:.0f}", "largest network, millions of parameters")
v("intent_b", INTENT_PARAMS / 10 ** 9, "{:.0f}", "intent model, billions of parameters")
v("chunk_m", CHUNK_PARAMS / 10 ** 6, "{:.0f}", "chunk policy, millions of parameters")
v("intent_over_vnn", INTENT_PARAMS / max(VNN_MAX_PARAMS), "{:.1f}", "intent model over the largest, times")

print("== Transfer discount (AV 15.9.4, L = 1 m per unit)")
v("eps_gap_max", m(stop.spare, "m"), "{:.3f}", "largest discrepancy that keeps the spare positive")

print("== Coverage count (AV Napkin Math 15.3)")
cov, cov_tight = COV.result, COV_TIGHT.result
v("fit_sensors", COV.strata["sensors"], "{:,.0f}", "sensors, FIT")
v("fit_compute", COV.strata["compute"], "{:,.0f}", "compute, FIT")
v("fit_power", COV.strata["transceivers and power"], "{:,.0f}", "transceivers and power, FIT")
v("fit_gate", COV.strata["gate drivers"], "{:,.0f}", "gate drivers, FIT")
v("fit_target", COV.target, "{:.0f}", "unmitigated target, FIT")
v("fit_target_tight", COV_TIGHT.target, "{:.0f}", "tighter target, FIT")
v("confidence_pct", 100 * COV.confidence, "{:.0f}", "confidence, %")
v("miss_confidence", 1 - COV.confidence, "{:.2f}", "1 - C")
v("fit_raw", cov.raw_rate, "{:,.0f}", "raw rate, FIT")
v("cov_pct", 100 * cov.coverage, "{:.2f}", "coverage, %")
v("cov_frac", cov.coverage, "{:.4f}", "coverage as a fraction")
v("n_exact", cov.injections_exact, "{:.1f}", "zero-miss injections, exact")
v("n_cov", cov.injections, "{:,.0f}", "zero-miss injections")
v("rule3", cov.rule_of_three, "{:,.0f}", "rule of three")
v("cov_tight_pct", 100 * cov_tight.coverage, "{:.2f}", "coverage at the tighter target, %")
v("n_cov_tight", cov_tight.injections, "{:,.0f}", "injections at the tighter target")
v("n_gate", cov.allocation["gate drivers"], "{:.0f}", "injections to gate drivers")
v("n_power", cov.allocation["transceivers and power"], "{:.0f}", "injections to transceivers and power")
v("n_sensors", cov.allocation["sensors"], "{:.0f}", "injections to sensors")
v("n_compute", cov.allocation["compute"], "{:.0f}", "injections to compute")
wall_rate = WALL.rates[-1]
v("wall_rate_exp", math.log10(m(wall_rate, "1/hour")), "{:.0f}", "hazard-rate exponent, per hour")
v("wall_bh", m(WALL.exposure(wall_rate), "hour") / 10 ** 9, "{:.0f}", "failure-free hours, billions")

print("== Importance sampling (AV 15.10.2)")
rare = RARE.result
v("is_mean", m(RARE.mean, "ms"), "{:.0f}", "mean queue delay, ms")
v("rel_err_pct", 100 * RARE.relative_error, "{:.0f}", "target relative error, %")
sci("p_fail", rare.failure_probability, "{:.1f}", "failure probability per stop")
sci("n_mc", N_MC, "{:.1f}", "plain Monte Carlo runs")
sci("n_zero", N_ZERO, "{:.1f}", "zero-failure runs")
sci("is_speedup", rare.speedup, "{:.0f}", "plain over weighted runs")
v("n_is", math.ceil(N_IS), "{:,.0f}", "importance-sampling runs")
v("n_levels_words", rare.n_levels, "words", "cross-entropy levels")
for i, level in enumerate(rare.levels, 1):
    v(f"ce_level{i}", m(level, "ms"), "{:.1f}", f"cross-entropy level {i}, ms")
v("ce_mean", m(rare.proposal_mean, "ms"), "{:.1f}", "final proposal mean, ms")
v("q_fail_pct", 100 * rare.proposal_failure_share, "{:.1f}", "proposal runs that fail, %")
v("ess", rare.effective_sample_size, "{:.0f}", "effective sample size of the weighted runs")

print("== O'Kelly et al. 2018 (Table 1; sec. 4)")
OK_100K = [e for e in OKELLY if e["runs"] == min(r["runs"] for r in OKELLY)]
OK_RSE = {(e["method"], e["runs"], e["gamma"]): 100 * e["std_error"] / e["estimate"] for e in OKELLY}
rarest = min(e["gamma"] for e in OKELLY)
naive_rarest = [e for e in OK_100K if e["method"] == "Naive MC" and e["gamma"] == rarest][0]
ce_rarest = [e for e in OK_100K if e["method"] == "Cross-entropy" and e["gamma"] == rarest][0]
v("ok_runs", naive_rarest["runs"], "{:,.0f}", "runs per arm")
v("ok_big_runs_m", max(e["runs"] for e in OKELLY) / 10 ** 6, "{:.1f}", "large naive baseline, millions of runs")
v("ok_event_exp", math.floor(math.log10(naive_rarest["estimate"])), "{:.0f}", "order of the rarest event")
v("ok_rse_naive", 100 * naive_rarest["std_error"] / naive_rarest["estimate"], "{:.0f}", "naive relative error, %")
v("ok_rse_ce", 100 * ce_rarest["std_error"] / ce_rarest["estimate"], "{:.0f}", "cross-entropy relative error, %")
v("ok_vr_low", STUDIES.OKellyVarianceReductionLowX, "{:.0f}", "variance reduction, low end, times")
v("ok_vr_high", STUDIES.OKellyVarianceReductionHighX, "{:.0f}", "variance reduction, high end, times")

print("== Record (AV 15.4.2)")
v("payload", PAYLOAD, "{:.0f}", "full rack payload, kg")

if __name__ == "__main__":
    write(__file__)
