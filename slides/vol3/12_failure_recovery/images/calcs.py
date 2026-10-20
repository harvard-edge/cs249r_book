#!/usr/bin/env python3
"""Every number on the Volume III Lecture 12 slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations, and the chapter reads the same
entries: one repair against unwinding and rebuilding
(Scenarios.TestConfigRepairVsRebuild), the remediation agent's pivot loss
(Scenarios.RemediationCanaryPivotLoss, from Agents.Remediation.OnCall), the
undetected loop (Scenarios.RepairLoopToTurnCeiling), layered retries
(Scenarios.ThreeLayerRetryAmplification), the bulkhead and the jittered
retry cohort (Scenarios.BlockingCallBulkhead, RetryCohortJitter), the
cleanup graph and the breaker window of the chapter's Exercises 1 and 4
(Scenarios.TestEnvironmentCleanupGraph, TestServiceBreakerWindow), all backed
by engine/solvers/recovery.py. Cited figures are Literature entries: the SEC
order on Knight Capital and the CrowdStrike review (IncidentReports), the
MAST traces (MultiAgentFailures), and Table 1 of Metastable Failures in the
Wild (MetastableFailures, counted by Scenarios.RetrySustainedMetastableOutages).
This file only selects values, converts units, and formats them; it types no
numbers. Each value is recorded with v() under the key the deck prints as
\\val{key}; running this file writes images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Scenarios = mlsysim.Scenarios
Literature = mlsysim.Literature

REPAIR_S = Scenarios.TestConfigRepairVsRebuild
PIVOT_S = Scenarios.RemediationCanaryPivotLoss
LOOP_S = Scenarios.RepairLoopToTurnCeiling
RETRY_S = Scenarios.ThreeLayerRetryAmplification
POOL_S = Scenarios.BlockingCallBulkhead
COHORT_S = Scenarios.RetryCohortJitter
CLEAN_S = Scenarios.TestEnvironmentCleanupGraph
BREAKER_S = Scenarios.TestServiceBreakerWindow
OUTAGES_S = Scenarios.RetrySustainedMetastableOutages

REPAIR = REPAIR_S.result
PIVOT = PIVOT_S.result
LOOP = LOOP_S.result
RETRY = RETRY_S.result
POOL = POOL_S.result
COHORT = COHORT_S.result
CLEAN = CLEAN_S.result
BREAKER = BREAKER_S.result
OUTAGES = OUTAGES_S.result
INC = Literature.IncidentReports
MAST = Literature.MultiAgentFailures


def mag(q, unit):
    return q.to(unit).magnitude


def pct(x):
    return x * 100


def usd(q):
    return mag(q, "USD")


print("== Knight Capital, 1 August 2012 (Literature.IncidentReports, SEC order)")
_servers, _missed = INC.KnightSmarsServers, INC.KnightServersMissed
v("kn_servers", _servers, "{:.0f}", "SMARS servers")
v("kn_servers_words", _servers, "words", "the same, in words")
v("kn_missed", _missed, "{:.0f}", "servers the deployment missed")
v("kn_correct", _servers - _missed, "{:.0f}", "servers that received the new code")
v("kn_correct_words", _servers - _missed, "words", "the same, in words")
v("kn_parents", INC.KnightParentOrders, "{:.0f}", "incoming parent orders")
v("kn_exec_m", INC.KnightExecutions / 10 ** 6, "{:.0f}", "executions, millions")
v("kn_stocks", INC.KnightStocks, "{:.0f}", "stocks")
v("kn_shares_m", INC.KnightShares / 10 ** 6, "{:.0f}", "shares, millions")
v("kn_minutes", INC.KnightTradingMinutes, "{:.0f}", "minutes of erroneous trading")
v("kn_loss_m", INC.KnightRealizedLossUsd / 10 ** 6, "{:.0f}", "realized loss, $ millions")
v("kn_emails", INC.KnightAlertEmails, "{:.0f}", "'Power Peg disabled' e-mails")

print("== Cleanup graph (Scenarios.TestEnvironmentCleanupGraph; Exercise 1)")
_d = {k: mag(t, "s") for k, t in CLEAN_S.durations.items()}
v("cl_ops_words", len(_d), "words", "registered cleanup operations")
for _k in _d:
    v(f"cl_{_k}", _d[_k], "{:.0f}", f"{_k}, s")
v("cl_stack", mag(CLEAN.stack_time, "s"), "{:.0f}", "reverse stack, s")
v("cl_graph", mag(CLEAN.graph_time, "s"), "{:.0f}", "cleanup graph, s")

print("== Repair against unwind (Scenarios.TestConfigRepairVsRebuild)")
v("rp_unwind", mag(REPAIR_S.unwind_time, "s"), "{:.1f}", "compensation, s")
v("rp_rebuild", mag(REPAIR_S.rebuild_time, "s"), "{:.0f}", "rebuild, s")
v("rp_rebuild_in", REPAIR_S.rebuild_input, "{:,.0f}", "rebuild input tokens")
v("rp_rebuild_out", REPAIR_S.rebuild_output, "{:,.0f}", "rebuild output tokens")
v("rp_startup", mag(REPAIR_S.startup_time, "s"), "{:.2f}", "pre-output startup, s")
v("rp_out", REPAIR_S.repair_output, "{:.0f}", "repair output tokens")
v("rp_rate", mag(REPAIR_S.output_rate, "token/s"), "{:.0f}", "decode rate, tokens/s")
v("rp_validate", mag(REPAIR_S.validation_time, "s"), "{:.1f}", "validation, s")
v("rp_price_in", mag(REPAIR_S.input_price, "USD/Mtoken"), "{:.0f}", "input price, $ per million")
v("rp_price_out", mag(REPAIR_S.output_price, "USD/Mtoken"), "{:.0f}", "output price, $ per million")
v("rp_back_s", mag(REPAIR.fallback_time, "s"), "{:.1f}", "C_fallback, s")
v("rp_back_usd", usd(REPAIR.fallback_cost), "{:.3f}", "C_fallback, $")
v("rp_rep_s", mag(REPAIR.repair_time, "s"), "{:.2f}", "C_repair, s")
v("rp_rep_usd", usd(REPAIR.repair_cost), "{:.5f}", "C_repair, $")
v("rp_pstar_s", pct(REPAIR.break_even_time), "{:.1f}", "p*, seconds, %")
v("rp_pstar_usd", pct(REPAIR.break_even_cost), "{:.1f}", "p*, dollars, %")
v("rp_p", REPAIR_S.assumed_success, "{:.1f}", "assumed repair success")
v("rp_exp_s", mag(REPAIR.expected_time, "s"), "{:.2f}", "expected time at the assumed p, s")
v("rp_save", pct(REPAIR.time_saved_fraction), "{:.1f}", "time saved against unwinding now, %")
v("rp_one_in", REPAIR.one_in, "{:.0f}", "one success in this many clears both axes")
v("rp_one_in_pct", pct(1 / REPAIR.one_in), "{:.2f}", "the same, %")

print("== CrowdStrike, 19 July 2024 (Literature.IncidentReports)")
_t0 = mlsysim.Q_(INC.CrowdStrikeReleaseUtcHour, "hour") + mlsysim.Q_(INC.CrowdStrikeReleaseUtcMinute, "minute")
_t1 = mlsysim.Q_(INC.CrowdStrikeRevertUtcHour, "hour") + mlsysim.Q_(INC.CrowdStrikeRevertUtcMinute, "minute")
v("cs_t0_h", INC.CrowdStrikeReleaseUtcHour, "{:02.0f}", "released, hour UTC")
v("cs_t0_m", INC.CrowdStrikeReleaseUtcMinute, "{:02.0f}", "released, minute")
v("cs_t1_h", INC.CrowdStrikeRevertUtcHour, "{:02.0f}", "reverted, hour UTC")
v("cs_t1_m", INC.CrowdStrikeRevertUtcMinute, "{:02.0f}", "reverted, minute")
v("cs_window", mag(_t1 - _t0, "min"), "{:.0f}", "exposure window, min")
v("cs_devices_m", INC.CrowdStrikeAffectedWindowsDevices / 10 ** 6, "{:.1f}", "Windows devices, millions")

print("== Pivot loss (Scenarios.RemediationCanaryPivotLoss, Agents.Remediation.OnCall)")
_w = PIVOT_S.workload
v("pl_pbad_pct", pct(_w.wrong_change_probability), "{:.0f}", "change wrong, %")
v("pl_pbad", _w.wrong_change_probability, "{:.2f}", "the same, probability")
v("pl_impact", usd(_w.wrong_change_impact), "{:,.0f}", "consequence, $")
v("pl_catch_pct", pct(_w.canary_catch_rate), "{:.0f}", "canary catch rate, %")
v("pl_catch", _w.canary_catch_rate, "{:.1f}", "the same, fraction")
v("pl_miss", 1 - _w.canary_catch_rate, "{:.1f}", "missed fraction")
v("pl_revert_pct", pct(_w.canary_recovery_fraction), "{:.0f}", "revert avoids, %")
v("pl_residue", 1 - _w.canary_recovery_fraction, "{:.1f}", "share a revert cannot avoid")
v("pl_residue_pct", pct(1 - _w.canary_recovery_fraction), "{:.0f}", "the same, %")
v("pl_first", usd(PIVOT.loss_without_check), "{:.0f}", "rollout first, $")
v("pl_caught", usd(PIVOT.loss_caught), "{:.2f}", "caught changes, $")
v("pl_missed", usd(PIVOT.loss_missed), "{:.0f}", "missed changes, $")
v("pl_total", usd(PIVOT.loss_with_check), "{:.2f}", "canary first, $")
v("pl_ratio", PIVOT.reduction, "{:.1f}", "reduction, x")
v("pl_missed_share", pct(PIVOT.missed_share), "{:.1f}", "remaining loss from missed changes, %")
v("pl_step_pts", pct(PIVOT_S.catch_step), "{:.0f}", "catch-rate step, points")
v("pl_raised_pct", pct(PIVOT.raised_catch_rate), "{:.0f}", "raised catch rate, %")
v("pl_per_step", usd(PIVOT.loss_per_catch_step), "{:.2f}", "loss removed by the step, $")
v("pl_raised_total", usd(PIVOT.loss_with_check_raised), "{:.2f}", "loss at the raised catch rate, $")
v("pl_attempt", usd(PIVOT.attempt_cost), "{:.0f}", "attempt cost, $")
v("pl_step_vs_attempt", PIVOT.step_vs_attempt, "{:.1f}", "step / attempt cost, x")
v("pl_loss_vs_attempt", PIVOT.loss_vs_attempt, "{:.1f}", "remaining loss / attempt cost, x")

print("== Loop cost (Scenarios.RepairLoopToTurnCeiling)")
v("lp_first", LOOP_S.first_input, "{:,.0f}", "first input, tokens")
v("lp_grow", LOOP.appended_per_turn, "{:,.0f}", "appended per turn, tokens")
v("lp_out", LOOP_S.output_per_turn, "{:.0f}", "output per turn, tokens")
v("lp_cap", LOOP_S.ceiling, "{:.0f}", "turn ceiling")
v("lp_stop", LOOP_S.stop_after, "{:.0f}", "stop after turn")
v("lp_turn_s", mag(LOOP_S.turn_time, "s"), "{:.0f}", "seconds per turn")
v("lp_cap_in", LOOP.ceiling_input, "{:,.0f}", "input at the ceiling, tokens")
v("lp_cap_usd", usd(LOOP.ceiling_cost), "{:.2f}", "charge at the ceiling, $")
v("lp_cap_min", mag(LOOP.ceiling_time, "min"), "{:.2f}", "time at the ceiling, min")
v("lp_stop_in", LOOP.stop_input, "{:,.0f}", "input stopped early, tokens")
v("lp_stop_usd", usd(LOOP.stop_cost), "{:.3f}", "charge stopped early, $")
v("lp_stop_s", mag(LOOP.stop_time, "s"), "{:.0f}", "time stopped early, s")
v("lp_ratio", LOOP.cost_ratio, "{:.1f}", "ceiling / early stop, x")
v("lp_ratio_round", LOOP.cost_ratio, "{:.0f}", "the same, whole")
v("lp_turn_ratio", LOOP_S.ceiling / LOOP_S.stop_after, "{:.1f}", "turns ratio (the linear guess), x")
v("lp_half", LOOP.half_turns, "{:.0f}", "half the ceiling, turns")
v("lp_half_ratio", LOOP.half_ratio, "{:.1f}", "input cut by stopping at half, x")

print("== Multi-agent traces (Literature.MultiAgentFailures, MAST)")
v("ma_traces", MAST.MastTraces, "{:,.0f}", "annotated traces")
v("ma_steprep", MAST.MastStepRepetitionPct, "{:.1f}", "step repetition, %")

print("== Layered retries (Scenarios.ThreeLayerRetryAmplification)")
v("rt_attempts", RETRY_S.attempts_per_layer, "{:.0f}", "attempts per layer")
v("rt_layered", RETRY.layered, "{:.0f}", "requests at the dependency")
v("rt_coord", RETRY.coordinated, "{:.0f}", "one owner, same allowance")

print("== Bulkhead and jitter (Scenarios.BlockingCallBulkhead, RetryCohortJitter)")
v("bh_arrivals", POOL_S.arrivals, "{:.0f}", "blocking calls")
v("bh_workers", POOL_S.workers, "{:.0f}", "workers")
v("bh_cap", POOL_S.worker_cap, "{:.0f}", "dependency cap, workers")
v("bh_queue", POOL_S.queue_cap, "{:.0f}", "dependency queue")
v("bh_free_uncapped", POOL.uncapped_free, "{:.0f}", "free workers uncapped")
v("bh_rejected", POOL.rejected, "{:.0f}", "rejected at admission")
v("bh_free", POOL.protected_workers, "{:.0f}", "workers kept free")
v("jt_retries", COHORT_S.retries, "{:.0f}", "retries in the cohort")
v("jt_burst_ms", mag(COHORT_S.burst_window, "ms"), "{:.0f}", "synchronized window, ms")
v("jt_span_s", mag(COHORT_S.jitter_span, "s"), "{:.0f}", "jitter span, s")
v("jt_sync", mag(COHORT.synchronized_rate, "1/s"), "{:,.0f}", "synchronized rate, per s")
v("jt_mean", mag(COHORT.jittered_mean_rate, "1/s"), "{:.0f}", "jittered mean rate, per s")
v("jt_spread", COHORT.spread_ratio, "{:,.0f}", "spread ratio")
v("jt_expected", COHORT.expected_in_window, "{:.3f}", "expected retries in one window")

print("== Retry-sustained outages (Scenarios.RetrySustainedMetastableOutages)")
_ms = dict(OUTAGES.with_duration)
v("ms_incidents", OUTAGES.incidents, "{:.0f}", "incidents in Table 1")
v("ms_retry", OUTAGES.retry_sustained, "{:.0f}", "sustained by retries")
v("ms_share", pct(OUTAGES.share), "{:.0f}", "share, %")
v("ms_over_words", OUTAGES.over_hours, "words", "retry-sustained outages over the threshold")
v("ms_threshold", OUTAGES_S.threshold_hours, "{:.0f}", "threshold, hours")
v("ms_ibm1", _ms["IBM1"], "{:.2f}", "IBM1, hours")
v("ms_aws1", _ms["AWS1"], "{:.1f}", "AWS1, hours")

print("== Breaker window (Scenarios.TestServiceBreakerWindow; Exercise 4)")
_o = BREAKER_S.outcomes
v("br_n", BREAKER.outcomes, "{:.0f}", "outcomes in the window")
for _k in _o:
    v(f"br_{_k}", _o[_k], "{:.0f}", f"{_k} outcomes")
v("br_threshold", BREAKER_S.threshold, "{:.2f}", "trip fraction")
v("br_min", BREAKER_S.min_outcomes, "{:.0f}", "minimum outcomes")
v("br_classified", BREAKER.classified_failures, "{:.0f}", "failures, classified")
v("br_classified_f", BREAKER.classified_fraction, "{:.2f}", "failure fraction, classified")
v("br_naive", BREAKER.naive_failures, "{:.0f}", "failures, every non-success")
v("br_naive_f", BREAKER.naive_fraction, "{:.2f}", "failure fraction, every non-success")

if __name__ == "__main__":
    write(__file__)
