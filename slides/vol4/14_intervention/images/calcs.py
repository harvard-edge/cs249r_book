#!/usr/bin/env python3
"""Every number on the Lecture 14 slides and notes (Supervisory Intervention), read from MLSysIM.

MLSysIM holds the inputs (the Embodied.* running machine and its aisle record
with the sourced human-response terms; Literature.* for the California DMV
filings, the Zhang et al. take-over series, the 49 CFR 563 recorder windows,
UN R157, Sirius, Merat et al., ISO 10218-1, the Cortex-M4 cryptography cycle
counts, and the Lufthansa 2904 report) and the calculations: the takeover
budgets at each request speed (Scenarios.Warehouse*Takeover*,
TakeoverBudgetModel over the finished S-curve stopping budget), the ceiling
ladder (WarehouseTakeoverCeilingLadder), the bumpless transfer
(WarehouseTakeoverBrakingBlend), authentication and the verifier flood
(WarehouseRemoteCommandAuthentication), and the disengagement record
(WaymoCaliforniaDisengagementRecord). The chapter reads the same scenarios.
This file only selects values, converts units, and formats them; it types no
numbers. Each value is recorded with v() under the key the deck prints as
\\val{key}; running this file writes images/numbers.tex. make_figs.py imports
it for the scenario objects and plot inputs.

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
TAKEOVER, CRYPTO, EDR = Literature.HumanTakeover, Literature.Crypto, Literature.EventDataRecorders
REPORTS = Literature.IncidentReports

RECORD = Scenarios.WaymoCaliforniaDisengagementRecord
IN_AISLE = Scenarios.WarehouseInLoopTakeoverAtAisleSpeed
OOL_AISLE = Scenarios.WarehouseOutOfLoopTakeoverAtAisleSpeed
IN_SLOW = Scenarios.WarehouseInLoopTakeoverAtRequestSpeed
IN_RETIRED = Scenarios.WarehouseInLoopTakeoverAtRetiredSlowdown
OOL_CRAWL = Scenarios.WarehouseOutOfLoopTakeoverAtCrawl
OOL_CRAWL_P99 = Scenarios.WarehouseOutOfLoopTakeoverAtCrawlP99
WINDOW_AISLE = Scenarios.WarehouseFixedWindowAtAisleSpeed
WINDOW_CRAWL = Scenarios.WarehouseFixedWindowAtCrawl
LADDER = Scenarios.WarehouseTakeoverCeilingLadder
BLEND = Scenarios.WarehouseTakeoverBrakingBlend
AUTH = Scenarios.WarehouseRemoteCommandAuthentication


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


aisle, ool, slow = IN_AISLE.result, OOL_AISLE.result, IN_SLOW.result
crawl, rows, blend, auth = OOL_CRAWL.result, LADDER.result.rows, BLEND.result, AUTH.result

print("== Hook: Waymo's California DMV filings (SI 14.1)")
v("dmv_first", RECORD.first["miles_per_disengagement"], "{:,.0f}", "2015 miles per disengagement")
v("dmv_2020", RECORD.row(RECORD.swing_from)["miles_per_disengagement"], "{:,.0f}", "2020 miles per disengagement")
v("dmv_drop", RECORD.rate_drop, "{:.1f}", "2020 to 2021 rate fell, x")
v("dmv_growth", RECORD.miles_growth, "{:.1f}", "2020 to 2021 miles grew, x")
v("dmv_latest_n", RECORD.latest["disengagements"], "{:.0f}", "2025 safety-driver takeovers")
v("dmv_latest", RECORD.latest["miles_per_disengagement"], "{:,.0f}", "2025 miles per disengagement")
v("dmv_dl", RECORD.latest_driverless["miles_per_disengagement"], "{:,.0f}", "2025 driverless miles per disengagement")
v("csv_rows", TAKEOVER.WaymoDisengagementRows2024, "{:.0f}", "2024 CSV rows, Waymo")
v("csv_driver", TAKEOVER.WaymoDisengagementsByTestDriver2024, "{:.0f}", "begun by the test driver")
v("csv_av", TAKEOVER.WaymoDisengagementsByAvSystem2024, "{:.0f}", "begun by the AV system")
v("csv_driver_pct", 100 * TAKEOVER.WaymoDisengagementsByTestDriver2024 / TAKEOVER.WaymoDisengagementRows2024,
  "{:.1f}", "share begun by the driver, %")

print("== Recall: the finished budget at the aisle speed (SE Napkin Math 12.1)")
machine = aisle.machine
v("v_aisle", m(IN_AISLE.speed, "m/s"), "{:.1f}", "aisle speed, m/s")
v("d_clear_m", m(AISLE.d_clear, "m"), "{:.2f}", "rack-end clear distance, m")
v("d_clear_mm", m(AISLE.d_clear, "mm"), "{:,.0f}", "the same, mm")
v("spare", m(machine.spare, "mm"), "{:.1f}", "spare at the aisle speed, mm")
v("spare0", m(machine.spare, "mm"), "{:.0f}", "the same, whole mm")
v("deadline", m(aisle.deadline, "ms"), "{:.1f}", "spare as delay (T_latest), ms")
v("deadline2", m(aisle.deadline, "ms"), "{:.2f}", "the same, two decimals")
v("d2a", m(AISLE.t_dispatch_to_admission, "ms"), "{:.1f}", "dispatch-to-admission allocation, ms")
v("spare_da", m(aisle.machine_with_dispatch.spare, "mm"), "{:.1f}", "spare after the allocation, mm")
v("deadline_da", m(aisle.deadline_after_admission, "ms"), "{:.1f}", "deadline after the allocation, ms")
v("machine_stop1", m(machine.required_distance, "mm"), "{:.1f}", "finished budget at the aisle speed, mm")
v("machine_stop", m(machine.required_distance, "mm"), "{:.2f}", "the same, two decimals")
v("delay_travel", m(machine.delay_travel, "mm"), "{:.2f}", "travel during the pre-brake delay, mm")
v("tau", m(AISLE.tau_delay, "ms"), "{:.1f}", "pre-brake delay, ms")
v("scurve", m(machine.scurve_travel, "mm"), "{:.2f}", "resident S-curve stop at the aisle speed, mm")
v("fixed", m(AISLE.fixed_overhead, "mm"), "{:.0f}", "fixed allowances, mm")
v("over_spare_words", aisle.takeover_over_spare, "words", "in-loop travel over the spare, about")

print("== The in-loop timing strip (SI eq. 1, figure 2)")
v("t_in", m(IN_AISLE.takeover, "ms"), "{:.0f}", "in-loop takeover, ms")
v("t_video", m(AISLE.t_video_glass_to_glass, "ms"), "{:.0f}", "video glass to glass, ms")
v("t_perc", m(AISLE.t_human_perceptual_slow, "ms"), "{:.0f}", "perceive, slow bound, ms")
v("t_cog", m(AISLE.t_human_cognitive_slow, "ms"), "{:.0f}", "decide, slow bound, ms")
v("t_mot", m(AISLE.t_human_motor_slow, "ms"), "{:.0f}", "act, slow bound, ms")
v("t_hs", m(AISLE.t_handshake, "ms"), "{:.0f}", "four-phase handshake, ms")
v("t_console", m(AISLE.t_takeover_transport, "ms"), "{:.0f}", "console term, ms")
v("travel_in", m(aisle.takeover_travel, "mm"), "{:.1f}", "travel during the in-loop takeover, mm")
v("over_spare", aisle.takeover_over_spare, "{:.1f}", "that travel over the spare, x")

print("== The takeover tax, partial budget (the stopping-budget bar plus the wait)")
v("stop_in", m(aisle.with_takeover.required_distance, "mm"), "{:,.2f}", "stop with the in-loop takeover, mm")
v("over_in", m(-aisle.with_takeover.spare, "mm"), "{:.1f}", "past the rack end, mm")
v("t_ool_s", m(OOL_AISLE.takeover, "s"), "{:.1f}", "out-of-loop takeover, s")
v("travel_ool", m(ool.takeover_travel, "mm"), "{:,.1f}", "travel during it, mm")
v("stop_ool", m(ool.with_takeover.required_distance, "mm"), "{:,.2f}", "stop with the out-of-loop takeover, mm")
v("over_ool_m", m(-ool.with_takeover.spare, "m"), "{:.2f}", "past the rack end, m")
v("stop_in_da", m(aisle.with_takeover_and_dispatch.required_distance, "mm"), "{:,.2f}", "Napkin Math 14.1 in-loop stop, with the allocation, mm")
v("stop_ool_da", m(ool.with_takeover_and_dispatch.required_distance, "mm"), "{:,.2f}", "Napkin Math 14.1 out-of-loop stop, with the allocation, mm")

print("== Take-over times (Zhang et al. 2019) and the ceilings they allow (SI table 2)")
MTOT = [r["mean_takeover_time_s"] for r in Literature.TakeoverTimeMeta.entries]
v("n_cond", len(MTOT), "{:.0f}", "study conditions")
v("n_studies", TAKEOVER.Zhang2019TakeoverStudies, "{:.0f}", "studies")
v("tot_mean", m(AISLE.t_takeover_driving_mean, "s"), "{:.2f}", "mean condition, s")
v("tot_p50", m(AISLE.t_takeover_driving_p50, "s"), "{:.2f}", "median condition, s")
v("tot_p90", m(AISLE.t_takeover_driving_p90, "s"), "{:.2f}", "90th percentile, s")
v("tot_p99", m(AISLE.t_takeover_driving_p99, "s"), "{:.1f}", "99th percentile, s")
v("tot_p99_0", m(AISLE.t_takeover_driving_p99, "s"), "{:.0f}", "the same, whole seconds")
v("tot_max", max(MTOT), "{:.1f}", "slowest condition, s")
v("t_console_s", m(AISLE.t_takeover_transport, "s"), "{:.3f}", "console term, s")
v("t_ool3", m(OOL_AISLE.takeover, "s"), "{:.3f}", "out-of-loop takeover, s")
for key in ("p50", "mean", "p90", "p99"):
    v(f"ceil_{key}", m(rows[key].with_takeover.speed_ceiling, "m/s"), "{:.2f}", f"ceiling, {key}, m/s")
v("n_off_scale", sum(1 for x in MTOT if x > 10), "{:.0f}", "conditions above 10 s")
CEIL = {k: m(r.with_takeover.speed_ceiling, "m/s") for k, r in rows.items()}
TOT = {k: m(LADDER.responses[k], "s") for k in ("p50", "mean", "p90", "p99")}

print("== The speed a human buys (SI Napkin Math 14.2, eq. 3)")
part, comp = aisle.with_takeover, aisle.with_takeover_and_dispatch
v("room", m(AISLE.d_clear - AISLE.fixed_overhead, "m"), "{:.2f}", "room to brake, m")
v("a_brake", m(AISLE.a_brake, "m/s**2"), "{:.0f}", "credible deceleration, m/s^2")
v("j_brake", m(AISLE.j_brake, "m/s**3"), "{:.2f}", "jerk bound, m/s^3")
v("tau_s", m(AISLE.tau_delay, "s"), "{:.4f}", "pre-brake delay, s")
v("t_in_s", m(IN_AISLE.takeover, "s"), "{:.3f}", "in-loop takeover, s")
v("half_ramp", m(part.ramp_delay, "ms"), "{:.1f}", "half a jerk ramp, a/2j, ms")
v("b_in", m(part.ceiling_lead_time, "s"), "{:.4f}", "time at full speed b, s")
v("root_in", m(part.ceiling_root, "s"), "{:.4f}", "square-root term, s")
v("ceil_in", m(part.speed_ceiling, "m/s"), "{:.3f}", "in-loop ceiling, m/s")
v("ceil_in2", m(part.speed_ceiling, "m/s"), "{:.2f}", "the same, two decimals")
v("knee", m(part.jerk_plateau_speed, "m/s"), "{:.3f}", "a^2/j, m/s")
v("rigid", m(part.rigid_speed_ceiling, "m/s"), "{:.2f}", "perfect-brake ceiling, m/s")
v("ceil_in_da", m(comp.speed_ceiling, "m/s"), "{:.3f}", "in-loop ceiling with the allocation, m/s")
v("v_retired", m(IN_RETIRED.speed, "m/s"), "{:.1f}", "retired slow-down, m/s")
v("short_retired", m(-IN_RETIRED.result.time_left_after_admission, "ms"), "{:.1f}", "it misses by, ms")
v("ceil_ool", m(ool.with_takeover.speed_ceiling, "m/s"), "{:.2f}", "out-of-loop ceiling, m/s")

print("== The fallback deadline (SI eq. 4, Napkin Math 14.3)")
v("v_slow", m(IN_SLOW.speed, "m/s"), "{:.2f}", "in-loop request speed, m/s")
v("dl_slow", m(slow.deadline, "ms"), "{:.1f}", "deadline at the request speed, ms")
v("left_slow", m(slow.time_left, "ms"), "{:.1f}", "left after the in-loop takeover, ms")
v("left_slow_da", m(slow.time_left_after_admission, "ms"), "{:.1f}", "and after the allocation, ms")
v("v_step", m(AISLE.v_takeover_resolution, "m/s"), "{:.2f}", "speed resolution, m/s")
v("dl_ratio", m(slow.deadline / aisle.deadline, ""), "{:.1f}", "deadline gain from slowing, x")
v("v_crawl", m(OOL_CRAWL.speed, "m/s"), "{:.2f}", "crawl, m/s")
v("dl_crawl_s", m(crawl.deadline, "s"), "{:.2f}", "deadline at the crawl, s")
v("left_crawl_s", m(crawl.time_left, "s"), "{:.2f}", "left after the out-of-loop takeover, s")
v("t_ool_p99", m(OOL_CRAWL_P99.takeover, "s"), "{:.1f}", "99th-percentile takeover with the console, s")
r157 = Q_(TAKEOVER.R157MinimumRiskManoeuvreDelayS, "s")
v("r157", m(r157, "s"), "{:.0f}", "UN R157 transition time, s")
v("r157_travel", m(machine.travel_over(r157), "m"), "{:.0f}", "travel at the aisle speed, m")

print("== Estimate: an expected brake signal (Checkpoint 14.1)")
exp = rows["expected"]
v("t_exp", m(LADDER.responses["expected"], "ms"), "{:.0f}", "expected brake signal, ms")
v("w_exp", m(exp.takeover, "ms"), "{:.0f}", "with the console term, ms")
v("b_exp", m(exp.with_takeover.ceiling_lead_time, "s"), "{:.4f}", "b, s")
v("ceil_exp", m(exp.with_takeover.speed_ceiling, "m/s"), "{:.2f}", "ceiling, m/s")
v("ceil_exp3", m(exp.with_takeover.speed_ceiling, "m/s"), "{:.3f}", "the same, three decimals")
room = AISLE.d_clear - AISLE.fixed_overhead
v("room2a", m((room + room) / AISLE.a_brake, "s**2"), "{:.2f}", "2 room / a, s^2")
v("t_surp_s", m(LADDER.responses["surprise"], "s"), "{:.1f}", "surprise, s")
v("ceil_surp", m(rows["surprise"].with_takeover.speed_ceiling, "m/s"), "{:.2f}", "surprise ceiling, m/s")

print("== A person beside the machine (SI 14.3.1, footnote 5)")
v_manual = Q_(TAKEOVER.Iso10218ManualReducedSpeedMmS, "mm/s")
t_mhp = AISLE.t_human_simple_reaction_slow
v("v_manual", m(v_manual, "mm/s"), "{:.0f}", "manual reduced speed, mm/s")
v("t_mhp_slow", m(t_mhp, "ms"), "{:.0f}", "slow-bound human response, ms")
v("manual_travel", m(v_manual * t_mhp, "mm"), "{:.1f}", "travel during it, mm")

print("== Leases and the handshake (SI 14.3)")
v("chunk_period", m(RM.chunk_period, "ms"), "{:.0f}", "chunk renewal period, ms")
v("t_lease", m(AISLE.t_lease, "ms"), "{:.0f}", "chunk lease, ms")
v("t_manual", m(IN_AISLE.manual_lease, "ms"), "{:.0f}", "manual lease, ms")
v("lease_ext", m(slow.lease_extension, "ms"), "{:.0f}", "manual lease beyond the chunk lease, ms")
v("lease_ext_travel", m(slow.lease_extension_travel, "mm"), "{:.0f}", "its travel at the request speed, mm")
v("hs_req", m(AISLE.t_handshake_request, "ms"), "{:.0f}", "request, ms")
v("hs_ack", m(AISLE.t_handshake_acknowledge, "ms"), "{:.0f}", "acknowledge, ms")
v("hs_commit", m(AISLE.t_tick, "ms"), "{:.0f}", "commit (one tick), ms")
v("hs_conf", m(AISLE.t_handshake_confirm, "ms"), "{:.0f}", "confirm, ms")
v("hs_acked", m(AISLE.t_handshake_request + AISLE.t_handshake_acknowledge, "ms"), "{:.0f}", "commit begins, ms")
v("tick", m(AISLE.t_tick, "ms"), "{:.0f}", "permission tick, ms")
v("tick_khz", m(1 / AISLE.t_tick, "kHz"), "{:.0f}", "permission rate, kHz")

print("== Vote: the handover timeout (SI eq. 4, Checkpoint 14.2)")
v("timeout_cfg", m(IN_AISLE.configured_timeout, "ms"), "{:.0f}", "configured ceiling, ms")
v("ticks_da", aisle.timeout_ticks_after_admission, "{:.0f}", "whole ticks after the allocation")
v("timeout_tighter", m(IN_AISLE.configured_timeout - aisle.timeout_after_admission, "ms"), "{:.1f}",
  "deadline tighter than the ceiling by, ms")
v("fixed_stop", m(WINDOW_AISLE.result.with_takeover.required_distance, "m"), "{:.2f}", "500 ms window waited at 1.3 m/s, m")
v("fixed_over", m(-WINDOW_AISLE.result.with_takeover.spare, "m"), "{:.2f}", "past the rack end, m")
v("fixed_crawl", m(WINDOW_CRAWL.result.with_takeover.required_distance, "m"), "{:.2f}", "the same window at the crawl, m")

print("== Bumpless transfer (SI 14.4, eq. 6 and 7)")
v("step_jerk", m(blend.step_jerk, "m/s**3"), "{:,.0f}", "one-tick step as jerk, m/s^3")
v("window", m(blend.window, "ms"), "{:.2f}", "base blend window, ms")
v("window_ticks", blend.window_ticks, "{:,.0f}", "base window, whole ticks")
v("arm_torque", m(BLEND.torque_change, "N*m"), "{:.0f}", "arm torque change, N m")
v("arm_rate", m(BLEND.torque_rate_limit, "N*m/s"), "{:.0f}", "torque-rate ceiling, N m/s")
v("arm_window", m(blend.arm_window, "ms"), "{:.1f}", "arm blend window, ms")
v("arm_ticks", blend.arm_window_ticks, "{:.0f}", "arm window, whole ticks")
v("blended", m(blend.blended_stop, "mm"), "{:.1f}", "blended stop from the request speed, mm")
v("scurve_slow", m(blend.scurve_travel, "mm"), "{:.1f}", "resident stop from it, mm")
v("step_slow", m(blend.step_stop, "mm"), "{:.1f}", "ideal step stop, mm")

print("== The remote-assistant session at the request speed (SI table 7)")
v("s_spare", m(slow.machine.spare, "mm"), "{:.1f}", "spare at the request speed, mm")
v("s_travel", m(slow.takeover_travel, "mm"), "{:.1f}", "takeover travel, mm")
v("s_after", m(slow.spare_after_takeover, "mm"), "{:.1f}", "left, mm")
v("s_da_travel", m(slow.admission_travel, "mm"), "{:.1f}", "allocation travel, mm")
v("s_after_da", m(slow.spare_after_admission, "mm"), "{:.1f}", "left, mm")
v("s_after_lease", m(slow.spare_after_lease, "mm"), "{:.1f}", "left after the lease extension, mm")
v("blend_reserve", m(blend.reserve_use, "mm"), "{:.0f}", "a braking blend would use up to, mm")
v("crawl_left", m(crawl.spare_after_takeover, "mm"), "{:.1f}", "crawl's spare after the out-of-loop takeover, mm")

print("== Authentication and the verifier flood (SI 14.5, 14.9)")
v("t_ed", m(auth.verify_time, "ms"), "{:.1f}", "Ed25519 verification, ms")
v("ed_travel", m(auth.verify_travel, "mm"), "{:.2f}", "its travel at the aisle speed, mm")
v("n_miss", auth.forged_to_miss, "{:.0f}", "forged packets that push the veto past the deadline")
v("veto_at", m(auth.veto_verified_at, "ms"), "{:.1f}", "veto verified at, ms")
v("n_miss_da", auth.forged_to_miss_after_admission, "{:.0f}", "the same after the allocation")
v("flood_rate", auth.forged_rate, "{:.0f}", "packets per second")
v("fresh", m(auth.freshness_window, "ms"), "{:.0f}", "freshness window, ms")
v("fresh_travel", m(auth.freshness_travel, "mm"), "{:.0f}", "its travel, mm")
v("fresh_left", m(auth.spare_after_freshness, "mm"), "{:.1f}", "spare left, mm")

print("== Flight 2904 (SI War Story 14.1; Warsaw accident report)")
lh_gear = REPORTS.Dlh2904LeftGearContactAfterRightS
v("lh_gear", lh_gear, "{:.0f}", "left gear touched after the right, s")
v("lh_brake", lh_gear + REPORTS.Dlh2904WheelBrakesAfterLeftGearS, "{:.0f}", "wheel brakes acted after first touchdown, s")
v("lh_kt", REPORTS.Dlh2904RunwayExitGroundSpeedKt, "{:.0f}", "ground speed at the runway end, kt")
v("lh_dead_words", REPORTS.Dlh2904Fatalities, "words", "fatalities")

print("== The record and the masks (SI 14.7, 14.8)")
v("edr_old_s", EDR.LegacyPreCrashWindowS, "{:.0f}", "recorder window today, s")
v("edr_old_hz", EDR.LegacyPreCrashRateHz, "{:.0f}", "recorder rate today, Hz")
v("edr_old_n", EDR.LegacyPreCrashWindowS * EDR.LegacyPreCrashRateHz, "{:.0f}", "samples in that window")
v("edr_new_s", EDR.PreCrashWindowS, "{:.0f}", "recorder window after the phase-in, s")
v("edr_new_hz", EDR.PreCrashRateHz, "{:.0f}", "recorder rate after the phase-in, Hz")
edr_period = 1 / Q_(EDR.PreCrashRateHz, "Hz")
v("edr_period", m(edr_period, "ms"), "{:.0f}", "one sample per, ms")
v("edr_ticks", m(edr_period / AISLE.t_tick, ""), "{:.0f}", "permission ticks per sample")
v("pre_s", m(AISLE.pre_trigger_window, "s"), "{:.0f}", "pre-trigger window, s")
v("tuples", m(AISLE.pre_trigger_window / AISLE.t_tick, ""), "{:,.0f}", "tuples in the pre-trigger window")
v("drift_s", m(OOL_AISLE.takeover, "s"), "{:.1f}", "drift interval (the out-of-loop takeover), s")
v("drift_ticks", m(OOL_AISLE.takeover / AISLE.t_tick, ""), "{:,.0f}", "drift interval, ticks")
sirius = Q_(TAKEOVER.SiriusPreInterventionWindowS, "s")
v("sirius_s", m(sirius, "s"), "{:.0f}", "Sirius pre-intervention window, s")
v("sirius_left_s", m(OOL_AISLE.takeover - sirius, "s"), "{:.1f}", "drift left outside it, s")
v("sirius_left_ticks", m((OOL_AISLE.takeover - sirius) / AISLE.t_tick, ""), "{:,.0f}", "the same in ticks")
v("merat_lo", TAKEOVER.MeratLateralStabilizationLowS, "{:.0f}", "lateral control settles, low, s")
v("merat_hi", TAKEOVER.MeratLateralStabilizationHighS, "{:.0f}", "high, s")

# Plot inputs for make_figs.py, SI magnitudes at full precision.
DMV = RECORD.rows
DRV, DL = RECORD.drivered_permit, RECORD.driverless_permit
DMV_LAST = RECORD.latest
V_AISLE = m(IN_AISLE.speed, "m/s")
SPARE = m(machine.spare, "m")
T_VIDEO, T_PERC = m(AISLE.t_video_glass_to_glass, "s"), m(AISLE.t_human_perceptual_slow, "s")
T_COG, T_MOT = m(AISLE.t_human_cognitive_slow, "s"), m(AISLE.t_human_motor_slow, "s")
T_IN, T_OOL, T_HS = m(IN_AISLE.takeover, "s"), m(OOL_AISLE.takeover, "s"), m(AISLE.t_handshake, "s")
T_DA, TICK = m(AISLE.t_dispatch_to_admission, "s"), m(AISLE.t_tick, "s")
HS = {"Request": m(AISLE.t_handshake_request, "s"), "Acknowledge": m(AISLE.t_handshake_acknowledge, "s"),
      "Commit": m(AISLE.t_tick, "s"), "Confirm": m(AISLE.t_handshake_confirm, "s")}
V_SLOW, V_CRAWL = m(IN_SLOW.speed, "m/s"), m(OOL_CRAWL.speed, "m/s")
DL_AISLE, DL_SLOW, DL_CRAWL = m(aisle.deadline, "s"), m(slow.deadline, "s"), m(crawl.deadline, "s")
DA_BASE, W_BASE = m(AISLE.a_brake, "m/s**2"), m(blend.window, "s")
S0, S1 = m(slow.machine.spare, "m"), m(slow.spare_after_takeover, "m")
S2, S3 = m(slow.spare_after_admission, "m"), m(slow.spare_after_lease, "m")
BLEND_RESERVE = m(blend.reserve_use, "m")
T_ED, N_MISS, N_MISS_DA = m(auth.verify_time, "s"), auth.forged_to_miss, auth.forged_to_miss_after_admission


def deadline(speed):
    """Fallback deadline T_latest in s at a request speed in m/s (TakeoverBudgetModel)."""
    return m(IN_AISLE.variant(speed=Q_(speed, "m/s")).result.deadline, "s")


if __name__ == "__main__":
    write(__file__)
