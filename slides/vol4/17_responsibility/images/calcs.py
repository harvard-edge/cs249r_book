#!/usr/bin/env python3
"""Every number on the Lecture 17 slides and notes (Responsible Physical AI), read from MLSysIM.

MLSysIM holds the inputs (the Embodied.* running machine and its aisle record;
Literature.IncidentReports for the Cruise pullover and the three penalties;
Literature.Fairness, PedestrianAp75BySkinType, and PedestrianMissRateByAge for
the published per-group detection tables of Wilson et al. 2019 and Li et al.
2025) and the calculations: the decision chain (AdmittedCommandDecisionChain),
the rack-end oversight time (WarehouseRackEndOversightTime over the out-of-loop
takeover budget), the assistant's queue and the labor it creates
(WarehouseRemoteAssistantQueue*, WarehouseRemoteAssistantLabor), the stage exit
(WarehouseSupervisedStageExit), the detection shortfall
(WarehouseGroupDetectionShortfall, DetectionShortfallModel), the group sample
(PedestrianGroupEvaluationSample), the crossing exposure, event-triggered
retention, and the Cruise penalties (CruiseDisclosurePenalties). The chapter
reads the same scenarios. This file only selects values, converts units, and
formats them; it types no numbers. Each value is recorded with v() under the
key the deck prints as \\val{key}; running this file writes images/numbers.tex.
make_figs.py imports it for the scenario objects and plot inputs.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402

Embodied, Scenarios, Literature = mlsysim.Embodied, mlsysim.Scenarios, mlsysim.Literature
Q = mlsysim.core.units.Q_
from mlsysim.physics import calc_kinetic_energy  # noqa: E402

RM = Embodied.MobileManipulator.WarehouseMobileManipulator
AISLE = Embodied.Scenario.WarehouseAisle
IR, FAIR = Literature.IncidentReports, Literature.Fairness

PENALTIES = Scenarios.CruiseDisclosurePenalties
CHAIN = Scenarios.AdmittedCommandDecisionChain
OVERSIGHT = Scenarios.WarehouseRackEndOversightTime
IN_AISLE = Scenarios.WarehouseInLoopTakeoverAtAisleSpeed
QUEUE = Scenarios.WarehouseRemoteAssistantQueue
QUEUES = {q.machines: q for q in (Scenarios.WarehouseRemoteAssistantQueueTenMachines, QUEUE,
                                   Scenarios.WarehouseRemoteAssistantQueueThirtyTwoMachines)}
CHECK = Scenarios.WarehouseRemoteAssistantQueueTwentyFourMachines
STAGE = Scenarios.WarehouseSupervisedStageExit
SHORTFALL = Scenarios.WarehouseGroupDetectionShortfall
SAMPLE = Scenarios.PedestrianGroupEvaluationSample
CROSSINGS = Scenarios.WarehouseCrossingExposure
RETENTION = Scenarios.WarehouseEventTriggeredRetention
LABOR = Scenarios.WarehouseRemoteAssistantLabor
LI = load_series("slides/vol4/17_responsibility/images/data/li2025_pedestrian_miss_rate_by_age.csv")
WILSON = load_series("slides/vol4/17_responsibility/images/data/wilson2019_bdd100k_person_ap75.csv")


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


print("== Hook and timeline: the Cruise pullover and the record (ch 17.1, 17.4.2)")
v("drag_m", m(Q(IR.CruisePulloverDistanceFt, "ft"), "m"), "{:.1f}", "pullover distance, m")
v("cpuc_days", PENALTIES.days_withheld, "{:.0f}", "days without a full account")
v("cpuc_daily", PENALTIES.daily_maximum, "{:,.0f}", "CPUC daily maximum, USD")
v("cpuc_fine", PENALTIES.per_day_penalty, "{:,.0f}", "CPUC penalty, USD")
v("nhtsa_m", m(Q(PENALTIES.others["NHTSA"], "USD"), "MUSD"), "{:.1f}", "NHTSA penalty, million USD")
v("doj_fine", PENALTIES.others["DOJ"], "{:,.0f}", "DOJ criminal fine, USD")
v("pen_total", PENALTIES.total, "{:,.0f}", "penalties for the record, total, USD")
v("pen_total_m", m(Q(PENALTIES.total, "USD"), "MUSD"), "{:.1f}", "the same, million USD")
CRUISE = dict(drag_m=m(Q(IR.CruisePulloverDistanceFt, "ft"), "m"), cpuc=PENALTIES.per_day_penalty,
              nhtsa=PENALTIES.others["NHTSA"], doj=PENALTIES.others["DOJ"])

print("== Running machine: one admitted aisle command (ch 17.2)")
V_AISLE, A_BRAKE = m(AISLE.v_aisle, "m/s"), m(AISLE.a_brake, "m/s**2")
v("v_aisle", V_AISLE, "{:.1f}", "aisle speed, m/s")
v("d_clear", m(AISLE.d_clear, "m"), "{:.2f}", "rack-end clear distance, m")
v("mass", m(RM.loaded_mass, "kg"), "{:.0f}", "loaded mass, kg")
v("ke_aisle", m(calc_kinetic_energy(RM.loaded_mass, AISLE.v_aisle), "J"), "{:.0f}", "kinetic energy at the aisle speed, J")
v("a_brake", A_BRAKE, "{:.0f}", "credible deceleration, m/s^2")

print("== Estimate: a six-link chain of decisions (ch 17.2.1)")
CHAIN_C = CHAIN.completeness_by_link
p_good, p_mid, p_poor = CHAIN.link_probabilities
v("links", CHAIN.links, "{:.0f}", "decisions behind the command")
for key, p in (("good", p_good), ("mid", p_mid), ("poor", p_poor)):
    v(f"p_{key}", p, "{:.2f}", f"per-link retrieval, {key}")
    v(f"c_{key}", CHAIN_C[p], "{:.2f}", f"chain resolves, {key}")
v("c_poor_pct", 100 * CHAIN_C[p_poor], "{:.0f}", "the same, percent")

print("== Time to act at the rack end (ch 17.3.1)")
T_WINDOW, T_OOL = m(OVERSIGHT.hazard_window, "s"), m(OVERSIGHT.budget.takeover, "s")
TRAVEL, RATIO = m(OVERSIGHT.takeover_travel, "m"), OVERSIGHT.takeover_over_window
v("t_window", T_WINDOW, "{:.2f}", "time the hazard leaves, s")
v("t_ool", T_OOL, "{:.3f}", "out-of-loop takeover, s")
v("t_ool1", T_OOL, "{:.1f}", "the same, one decimal")
v("travel_ool", TRAVEL, "{:.1f}", "travel during the takeover, m")
v("ool_ratio", RATIO, "{:.1f}", "takeover over window")
v("ool_ratio_words", round(RATIO), "words", "about, in words")
v("ool_ratio_int", round(RATIO), "{:.0f}", "about, x")
v("t_in", m(IN_AISLE.takeover, "s"), "{:.3f}", "in-loop takeover, s")

print("== One assistant, many machines (ch Napkin Math 17.1, Checkpoint 17.1)")
REQ_H, SERVICE_S = m(QUEUE.request_rate, "1/hour"), m(QUEUE.service_time, "s")
v("req_rate", REQ_H, "{:.1f}", "requests per machine-hour")
v("service_min", m(QUEUE.service_time, "minute"), "{:.0f}", "service time, min")
v("service_s", SERVICE_S, "{:.0f}", "service time, s")
QROWS = {}
for n, q in QUEUES.items():
    r = q.result
    QROWS[n] = (r.utilization, m(r.mean_wait, "s"), m(r.p99_wait, "s"))
    v(f"n{n}", n, "{:.0f}", "machines")
    v(f"rho{n}", r.utilization, "{:.2f}", "utilization")
    v(f"wq{n}", m(r.mean_wait, "s"), "{:.0f}", "mean wait, s")
    v(f"wq{n}_min", m(r.mean_wait, "minute"), "{:.0f}", "mean wait, min")
r20 = QUEUE.result
v("n20_words", QUEUE.machines, "words", "machines, in words")
v("n10_req_h", m(QUEUES[10].result.arrival_rate, "1/hour"), "{:.0f}", "fleet requests per hour, 10 machines")
v("rho20_pct", 100 * r20.utilization, "{:.0f}", "utilization at 20, percent")
v("p99_20_s", m(r20.p99_wait, "s"), "{:,.0f}", "99th-percentile wait at 20, s")
v("p99_20_min", m(r20.p99_wait, "minute"), "{:.1f}", "the same, min")
v("p99_20_min0", m(r20.p99_wait, "minute"), "{:.0f}", "the same, whole min")
v("rho_cap", QUEUE.utilization_cap, "{:.1f}", "utilization cap")
v("n_max", QUEUE.max_machines, "{:.0f}", "machines at the cap")
v("chk_n", CHECK.machines, "{:.0f}", "Checkpoint 17.1 machines")
v("chk_rate", m(CHECK.request_rate, "1/hour"), "{:.1f}", "requests per machine-hour")
v("chk_s", m(CHECK.service_time, "s"), "{:.0f}", "service time, s")
v("chk_rho", CHECK.result.utilization, "{:.2f}", "utilization")
v("chk_wq", m(CHECK.result.mean_wait, "s"), "{:.0f}", "mean wait, s")

print("== Leaving a stage on evidence (ch 17.4.1)")
RATE = STAGE.rates[0]
T_REQ, DAYS_REQ = m(STAGE.exposure(RATE), "hour"), m(STAGE.calendar_days(RATE), "day")
FLEET_H_DAY = STAGE.rigs * m(STAGE.hours_per_day, "hour/day")
v("stage_per_h", 1 / m(RATE, "1/hour"), "{:,.0f}", "one intervention per this many hours")
v("stage_conf_pct", 100 * STAGE.confidence, "{:.0f}", "confidence, percent")
v("stage_alpha", 1 - STAGE.confidence, "{:.2f}", "alpha")
v("stage_inv_alpha", 1 / (1 - STAGE.confidence), "{:.0f}", "1 / alpha")
v("stage_rate_exp_mag", -math.log10(m(RATE, "1/hour")), "{:.0f}", "rate exponent magnitude, per hour (spoken: 10 to the minus)")
v("t_req", T_REQ, "{:,.0f}", "failure-free hours")
v("stage_n", STAGE.rigs, "{:.0f}", "machines")
v("stage_n_words", STAGE.rigs, "words", "machines, in words")
v("stage_h_day", m(STAGE.hours_per_day, "hour/day"), "{:.0f}", "hours per machine-day")
v("fleet_h_day", FLEET_H_DAY, "{:.0f}", "fleet hours per day")
v("days_req", DAYS_REQ, "{:.1f}", "days to reach them")
v("days_req_ceil", math.ceil(DAYS_REQ), "{:.0f}", "whole days")

print("== Published per-group detection (ch 17.5; Li Table 7, Wilson Table 4)")
LI_AVG = (FAIR.LiMissRateAverageAdultPct, FAIR.LiMissRateAverageChildPct)
WILSON_AVG = (FAIR.WilsonAp75AverageLighterPct, FAIR.WilsonAp75AverageDarkerPct)
v("li_child", LI_AVG[1], "{:.2f}", "children missed, average, %")
v("li_adult", LI_AVG[0], "{:.2f}", "adults missed, average, %")
v("li_gap", LI_AVG[1] - LI_AVG[0], "{:.2f}", "child minus adult, points")
v("w_ds", WILSON_AVG[1], "{:.1f}", "AP75 darker, average, %")
v("w_ls", WILSON_AVG[0], "{:.1f}", "AP75 lighter, average, %")
v("w_hold_ls", FAIR.WilsonHoldoutLighter, "{:.0f}", "holdout lighter")
v("w_hold_ds", FAIR.WilsonHoldoutDarker, "{:.0f}", "holdout darker")
v("w_need_ls", FAIR.WilsonNeededLighter, "{:,.0f}", "needed lighter")
v("w_need_ds", FAIR.WilsonNeededDarker, "{:,.0f}", "needed darker")
v("w_gap", FAIR.WilsonApGapToVerify, "{:.2f}", "true AP gap those counts verify")
v("n_detectors_words", len(LI), "words", "detectors in Li Table 7")
v("w_models_words", len(WILSON), "words", "models in Wilson Table 4")

print("== Two frames late (ch Napkin Math 17.2, Table 1, figure 2)")
DS = SHORTFALL.result
ROWS = DS.rows
T_FRAME_MS = m(DS.frame_period, "ms")
v("cam_hz", m(RM.nav_camera.frame_rate, "Hz"), "{:.0f}", "navigation camera, Hz")
v("t_frame", T_FRAME_MS, "{:.1f}", "frame period, ms")
v("p_miss_exp", math.log10(SHORTFALL.all_miss_probability), "{:.0f}", "all-frames miss exponent")
v("p_miss_inv", 1 / SHORTFALL.all_miss_probability, "{:,.0f}", "1 in this many")
v("p_ref", SHORTFALL.reference, "{:.2f}", "reference per-frame detection")
v("p_ref_pct", 100 * SHORTFALL.reference, "{:.0f}", "the same, percent")
v("n_ref", DS.reference_frames, "{:.0f}", "reference frames")
v("d_brake", m(DS.braking_travel, "m"), "{:.2f}", "braking distance at the aisle speed, m")
v("tau", m(AISLE.tau_delay, "ms"), "{:.1f}", "pre-brake delay, ms")
for i, r in enumerate(ROWS):
    v(f"g{i}_p", r.per_frame_detection, "{:.2f}", "per-frame detection")
    v(f"g{i}_pct", 100 * r.per_frame_detection, "{:.0f}", "the same, percent")
    v(f"g{i}_n", r.frames, "{:.0f}", "frames")
    v(f"g{i}_dn", r.extra_frames, "{:.0f}", "extra frames")
    v(f"g{i}_dt", m(r.extra_time, "ms"), "{:.1f}", "extra time, ms")
    v(f"g{i}_dr", m(r.distance_lost, "mm"), "{:.0f}", "distance lost, mm")
    v(f"g{i}_dr_m", m(r.distance_lost, "m"), "{:.3f}", "distance lost, m")
    v(f"g{i}_vi", m(r.impact_speed, "m/s"), "{:.2f}", "impact speed, m/s")
    v(f"g{i}_e", m(r.impact_energy, "J"), "{:.0f}", "impact energy, J")
v("gap_pts", 100 * (SHORTFALL.reference - ROWS[1].per_frame_detection), "{:.0f}", "per-frame gap, points")
v("g1_dn_words", ROWS[1].extra_frames, "words", "extra frames, in words")

print("== Published miss rates in meters and joules (derived from Li Table 7)")
# Each detector's per-image miss rate is treated as an independent per-frame
# miss probability at the running machine's speed, camera, mass, and deceleration.
LI_CONV = {}
for e in LI.entries:
    r = SHORTFALL.variant(reference=1 - e["miss_rate_adult_pct"] / 100,
                          groups=(1 - e["miss_rate_child_pct"] / 100,)).result
    LI_CONV[e["detector"]] = (r.reference_frames, r.rows[0])
avg = SHORTFALL.variant(reference=1 - LI_AVG[0] / 100, groups=(1 - LI_AVG[1] / 100,)).result.rows[0]
li_rows = [row for _, row in LI_CONV.values()]
for key, fn in (("dr", lambda r: m(r.distance_lost, "mm")), ("vi", lambda r: m(r.impact_speed, "m/s")),
                ("e", lambda r: m(r.impact_energy, "J"))):
    fmt = "{:.2f}" if key == "vi" else "{:.0f}"
    v(f"li_{key}_lo", min(fn(r) for r in li_rows), fmt, f"Li conversion, {key}, smallest")
    v(f"li_{key}_hi", max(fn(r) for r in li_rows), fmt, f"Li conversion, {key}, largest")
v("li_avg_dn", avg.extra_frames, "{:.0f}", "Li average, extra frames")
v("li_avg_dr", m(avg.distance_lost, "mm"), "{:.0f}", "Li average, mm")
v("li_avg_vi", m(avg.impact_speed, "m/s"), "{:.2f}", "Li average, m/s")
v("li_avg_e", m(avg.impact_energy, "J"), "{:.0f}", "Li average, J")

print("== Sample sizes for each group (ch 17.5.2)")
CI_S, CI_L = SAMPLE.small_interval, SAMPLE.large_interval
v("s_total", SAMPLE.total, "{:,.0f}", "test set")
v("s_share_pct", 100 * SAMPLE.small_share, "{:.0f}", "small group share, percent")
v("s_small", SAMPLE.small, "{:.0f}", "small group")
v("s_large", SAMPLE.large, "{:,.0f}", "large group")
v("s_recall", SAMPLE.recall, "{:.2f}", "detection rate in both")
v("s_pa", SAMPLE.rate_a, "{:.2f}", "rate a")
v("s_pb", SAMPLE.rate_b, "{:.2f}", "rate b")
v("s_power_pct", 100 * SAMPLE.power, "{:.0f}", "power, percent")
v("s_conf_pct", 100 * SAMPLE.confidence, "{:.0f}", "interval confidence, percent")
v("ci_l_lo", CI_L[0], "{:.3f}", "large group interval, low")
v("ci_l_hi", CI_L[1], "{:.3f}", "large group interval, high")
v("ci_s_lo", CI_S[0], "{:.3f}", "small group interval, low")
v("ci_s_hi", CI_S[1], "{:.3f}", "small group interval, high")
v("w_ratio", round(SAMPLE.width_ratio), "{:.0f}", "width ratio, about")
v("n_needed", SAMPLE.per_group_needed, "{:,.0f}", "people per group")
v("share_needed_pct", 100 * SAMPLE.share_needed, "{:.0f}", "share of the set per group, percent")

print("== Who bears the risk (ch 17.6)")
v("enc_pct", 100 * CROSSINGS.encounter_share, "{:.0f}", "encounters at three crossings, percent")
v("users_pct", 100 * CROSSINGS.user_share, "{:.0f}", "workers who use them, percent")
v("enc_share", CROSSINGS.encounter_share, "{:.2f}", "encounter share")
v("users_share", CROSSINGS.user_share, "{:.2f}", "user share")
v("rest_enc", 1 - CROSSINGS.encounter_share, "{:.2f}", "everyone else's encounter share")
v("rest_users", 1 - CROSSINGS.user_share, "{:.2f}", "everyone else's share of workers")
v("exp_ratio", CROSSINGS.concentration, "{:.0f}", "crossing users' exposure over the average")
v("rest_ratio_pct", 100 * CROSSINGS.rest_ratio, "{:.0f}", "everyone else, percent of the average")

print("== Keeping less (ch 17.7.2)")
v("ret_pre", m(RETENTION.pre_trigger, "s"), "{:.0f}", "pre-trigger window, s")
v("ret_post", m(RETENTION.post_trigger, "s"), "{:.0f}", "after the trigger, s")
v("ret_window", m(RETENTION.window, "s"), "{:.0f}", "window per event, s")
v("ret_events", RETENTION.events_per_day, "{:.0f}", "events per machine-day")
v("ret_day_h", m(RETENTION.operating_day, "hour"), "{:.0f}", "operating day, h")
v("ret_kept_min", m(RETENTION.kept_per_day, "minute"), "{:.0f}", "kept per machine-day, min")
v("ret_day_min", m(RETENTION.operating_day, "minute"), "{:.0f}", "operating day, min")
v("ret_kept_pct", 100 * RETENTION.kept_share, "{:.0f}", "share kept, percent")
v("ret_reduction", RETENTION.reduction, "{:.0f}", "reduction, x")

print("== Where the labor went (ch Napkin Math 17.3)")
v("lab_req_h", m(LABOR.request_time, "hour"), "{:.0f}", "request hours")
v("lab_watch_h", m(LABOR.watch_time, "hour"), "{:.0f}", "watching and waiting hours")
v("lab_day_h", m(LABOR.operating_day, "hour"), "{:.0f}", "assistant staffed, h")
v("lab_sweeps_words", LABOR.sweeps, "words", "sweeps, in words")
v("lab_sweep_min", m(LABOR.sweep_time, "minute"), "{:.0f}", "one sweep, min")
v("lab_sweep_h", m(LABOR.sweep_total, "hour"), "{:.1f}", "sweep hours")
v("lab_total_h", m(LABOR.total, "hour"), "{:.1f}", "person-hours per day")
v("lab_shifts", LABOR.shifts, "{:.1f}", "full-time shifts")

if __name__ == "__main__":
    write(__file__)
