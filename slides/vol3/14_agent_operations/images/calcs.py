#!/usr/bin/env python3
"""Every number on the Volume III Lecture 14 slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations, and the chapter reads the same
entries: the timing trial's spans and the cut tests
(Scenarios.RepairTrialSpanTimeline), head sampling
(Scenarios.RareFailureUnderHeadSampling and its variants), verdict-aware
retention (Scenarios.VerdictAwareTraceRetention), the controlled branches and
their Wilson bounds (Scenarios.InterfaceRepairBranches), tool-call drift
(Scenarios.TestLintToolDrift), the SPRT canary monitor and its allocations
(Scenarios.AcceptanceCanaryMonitor, ...Doubled, ...Full), Argo's ladder on
agent tasks (Scenarios.WebCanaryLadderOnAgentTasks), the release gate's
family-wise error (Scenarios.ReleaseGateFamilywiseError), and the paired
comparison owned by Agent Evaluation (Scenarios.RepairCandidatePairedComparison),
backed by engine/solvers/operations.py and physics/statistics.py. Cited
figures are Literature entries: OpenAI's sycophancy timeline
(IncidentReports), Dapper's Table 2 and first production sampling rate
(DapperSamplingOverhead, DistributedTracing), and SWE-agent's Table 3 viewer
ablation (AgentRetrieval). This file only selects values, converts units, and
formats them; it types no numbers. Each value is recorded with v() under the
key the deck prints as \\val{key}; running this file writes images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402

Scenarios = mlsysim.Scenarios
Literature = mlsysim.Literature

TRIAL_S = Scenarios.RepairTrialSpanTimeline
HEAD_S = Scenarios.RareFailureUnderHeadSampling
HEAD50_S = Scenarios.RecurringFailureUnderHeadSampling
HEAD5_S = Scenarios.FailureUnderFivePercentHeadSample
HEAD1_S = Scenarios.FailureUnderOnePercentHeadSample
KEEP_S = Scenarios.VerdictAwareTraceRetention
BRANCH_S = Scenarios.InterfaceRepairBranches
PAIR_S = Scenarios.RepairCandidatePairedComparison
PAIR = PAIR_S.result
DRIFT_S = Scenarios.TestLintToolDrift
MON_S = Scenarios.AcceptanceCanaryMonitor
MON2_S = Scenarios.AcceptanceCanaryMonitorDoubled
MONF_S = Scenarios.AcceptanceCanaryMonitorFull
LADDER_S = Scenarios.WebCanaryLadderOnAgentTasks
FAMILY_S = Scenarios.ReleaseGateFamilywiseError

TRIAL = TRIAL_S.result
KEEP = KEEP_S.result
MON = MON_S.result
MON2 = MON2_S.result
MONF = MONF_S.result
INC = Literature.IncidentReports
DAPPER = {r["sampling_frequency"]: r for r in load_series("slides/vol3/14_agent_operations/images/data/dapper_sampling_overhead.csv").entries}
TRACING = Literature.DistributedTracing
SWE = Literature.AgentRetrieval


def mag(q, unit):
    return q.to(unit).magnitude


def pct(x):
    return x * 100


print("== GPT-4o sycophancy timeline (Literature.IncidentReports, OpenAI 2025)")
v("hk_start", INC.OpenAiSycophancyRolloutStartDay, "{:.0f}", "rollout started, day of April")
v("hk_done", INC.OpenAiSycophancyRolloutCompleteDay, "{:.0f}", "rollout completed, day of April")
v("hk_monitor_words", INC.OpenAiSycophancyMonitoringDays, "words", "days monitored")
v("hk_rollback", INC.OpenAiSycophancyRollbackStartDay, "{:.0f}", "rollback initiated, day of April")
v("hk_rollback_h", INC.OpenAiSycophancyRollbackHours, "{:.0f}", "rollback duration, hours")

print("== Paired comparison (Scenarios.RepairCandidatePairedComparison; Agent Evaluation)")
_d, _lo, _hi = PAIR.difference, PAIR.lower, PAIR.upper
v("pr_pairs", PAIR.pairs, "{:.0f}", "task pairs")
v("pr_cand_only", PAIR_S.candidate_only, "{:.0f}", "candidate-only acceptances")
v("pr_base_only", PAIR_S.baseline_only, "{:.0f}", "baseline-only acceptances")
v("pr_gain", pct(_d), "{:.0f}", "observed gain, points")
v("pr_lo", pct(_lo), "{:.2f}", "interval lower limit, points")
v("pr_hi", pct(_hi), "{:.2f}", "interval upper limit, points")
v("pr_margin", pct(PAIR_S.tolerance), "{:.0f}", "noninferiority margin, points")
v("pr_need", PAIR.pairs_if_gain_holds, "{:.0f}", "pairs that center the lower limit on the margin")

print("== Timing trial (Scenarios.RepairTrialSpanTimeline)")
_tests, _lint = TRIAL_S.duration("tests"), TRIAL_S.duration("lint")
_cut = TRIAL_S.intervention["tests"]
_synth = TRIAL_S.duration("synthesize")
v("tr_leaf", mag(TRIAL.leaf_sum, "s"), "{:.1f}", "leaf spans summed, s")
v("tr_path", mag(TRIAL.critical_path, "s"), "{:.1f}", "critical path, s")
v("tr_wall", mag(TRIAL.elapsed, "s"), "{:.1f}", "segment, s")
v("tr_gap", mag(TRIAL.unexplained, "s"), "{:.1f}", "unexplained gap, s")
v("tr_tests", mag(_tests, "s"), "{:.1f}", "tests, s")
v("tr_lint", mag(_lint, "s"), "{:.1f}", "lint, s")
v("tr_lint_int", mag(_lint, "s"), "{:.0f}", "the same, whole seconds")
v("tr_cut", mag(_cut, "s"), "{:.1f}", "tests after the cache, s")
v("tr_cut_int", mag(_cut, "s"), "{:.0f}", "the same, whole seconds")
v("tr_changed", mag(TRIAL.changed_elapsed, "s"), "{:.1f}", "segment after the cut, s")
v("tr_report_start", mag(TRIAL.changed_elapsed - _synth, "s"), "{:.1f}", "report starts after the cut, s")
v("tr_saving", mag(TRIAL.saving, "s"), "{:.1f}", "saving, s")
v("tr_saving_int", mag(TRIAL.saving, "s"), "{:.0f}", "the same, whole seconds")
v("tr_reduction", mag(_tests - _cut, "s"), "{:.1f}", "cut to the tests, s")
v("tr_reduction_int", mag(_tests - _cut, "s"), "{:.0f}", "the same, whole seconds")

print("== Dapper Table 2 and Section 4.4 (slides/vol3/14_agent_operations/images/data/dapper_sampling_overhead.csv, DistributedTracing)")
_all = DAPPER["1/1"]
v("dp_full", _all["latency_change_pct"], "{:.1f}", "tracing every request, latency %")
v("dp_full_int", _all["latency_change_pct"], "{:.0f}", "the same, whole percent")
v("dp_first", TRACING.DapperFirstProductionSamplingDenominator, "{:,.0f}", "candidates per sampled trace")
v("dp_err", TRACING.DapperLatencyErrorPct, "{:.1f}", "latency measurement error, %")
v("dp_tput_max", -min(r["throughput_change_pct"] for r in DAPPER.values()), "{:.2f}",
  "largest throughput change, %")
_rows = sorted(DAPPER.values(), key=lambda r: r["sampled_per"])
_last_visible = _rows[:-1][-1]                     # the last row before one in 1,024
v("dp_mid_den", _last_visible["sampled_per"], "{:.0f}", "one in this many")
v("dp_mid", _last_visible["latency_change_pct"], "{:.2f}", "its latency change, %")
v("dp_low", DAPPER["1/1024"]["latency_change_pct"], "{:.2f}", "one in 1,024, latency %")

print("== Head sampling (Scenarios.RareFailureUnderHeadSampling and variants)")
v("hs1_p", pct(HEAD1_S.retention_probability), "{:.0f}", "Purpose: sample, %")
v("hs1_n", HEAD1_S.occurrences, "{:.0f}", "Purpose: recurrences")
v("hs1_miss", pct(HEAD1_S.miss), "{:.0f}", "Purpose: no detailed record, %")
v("hs10_p", pct(HEAD_S.retention_probability), "{:.0f}", "sample, %")
v("hs10_n", HEAD_S.occurrences, "{:.0f}", "occurrences")
v("hs10_n_words", HEAD_S.occurrences, "words", "the same, in words")
v("hs10_miss", pct(HEAD_S.miss), "{:.0f}", "no trace kept, %")
v("hs50_n", HEAD50_S.occurrences, "{:.0f}", "occurrences")
v("hs50_miss", pct(HEAD50_S.miss), "{:.1f}", "no trace kept, %")
v("hs5_p", pct(HEAD5_S.retention_probability), "{:.0f}", "checkpoint: sample, %")
v("hs5_keep", 1 - HEAD5_S.retention_probability, "{:.2f}", "checkpoint: chance one trial is not kept")
v("hs5_n", HEAD5_S.occurrences, "{:.0f}", "checkpoint: occurrences")
v("hs5_miss", pct(HEAD5_S.miss), "{:.0f}", "checkpoint: no trace kept, %")
v("hs5_miss_dec", HEAD5_S.miss, "{:.2f}", "the same, probability")

print("== Retention (Scenarios.VerdictAwareTraceRetention)")
v("rt_trials", KEEP_S.trials, "{:,.0f}", "trials per hour")
v("rt_mb", mag(KEEP_S.trace_size, "MB"), "{:.1f}", "trace body, MB")
v("rt_failed", KEEP_S.failed, "{:.0f}", "failed per hour")
v("rt_pending", KEEP_S.unresolved, "{:.0f}", "pending per hour")
v("rt_ordinary", KEEP_S.ordinary, "{:,.0f}", "ordinary per hour")
v("rt_hours", mag(KEEP_S.window, "hour"), "{:.0f}", "window, hours")
v("rt_full", mag(KEEP.full_volume, "GB"), "{:.0f}", "keep everything, GB a day")
v("rt_head_p", pct(KEEP_S.head_probability), "{:.0f}", "head sample, %")
v("rt_head_n", KEEP.head_count, "{:.0f}", "head-sampled traces per hour")
v("rt_head_gb", mag(KEEP.head_volume, "GB"), "{:.1f}", "head sampling, GB a day")
v("rt_head_failed", KEEP.head_failed, "{:.0f}", "expected failures kept per hour")
v("rt_head_pending", KEEP.head_unresolved, "{:.0f}", "expected pending kept per hour")
v("rt_control", pct(KEEP_S.control_probability), "{:.0f}", "control sample, %")
v("rt_tail_control", KEEP.tail_control, "{:.0f}", "control traces per hour")
v("rt_tail_n", KEEP.tail_count, "{:.0f}", "tail-selected traces per hour")
v("rt_tail_gb", mag(KEEP.tail_volume, "GB"), "{:.3f}", "tail selection, GB a day")
v("rt_tail_gb1", mag(KEEP.tail_volume, "GB"), "{:.1f}", "the same, one decimal")
v("rt_mult", KEEP.full_over_tail, "{:.1f}", "keep everything / tail selection, x")
v("rt_floor", mag(KEEP.floor_volume, "GB"), "{:.1f}", "floor, GB a day")
v("rt_point", mag(KEEP.per_point_volume, "GB"), "{:.1f}", "per point of control sampling, GB a day")

print("== Controlled branches (Scenarios.InterfaceRepairBranches)")
v("br_runs", BRANCH_S.runs("paginated"), "{:.0f}", "continuations per branch")
v("br_orig_acc", BRANCH_S.accepted("original"), "{:.0f}", "original: accepted")
v("br_orig_forb", BRANCH_S.outcomes["original"]["forbidden"], "{:.0f}", "original: forbidden rewrite")
v("br_warn_acc", BRANCH_S.accepted("warning"), "{:.0f}", "warning: accepted")
v("br_warn_budget", BRANCH_S.outcomes["warning"]["budget_stop"], "{:.0f}", "warning: budget stop")
v("br_pag_acc", BRANCH_S.accepted("paginated"), "{:.0f}", "paginated: accepted")
v("br_conf", pct(BRANCH_S.confidence), "{:.0f}", "Wilson confidence, %")
v("br_upper", pct(BRANCH_S.failure_upper("paginated")), "{:.0f}", "Wilson upper failure limit, %")
v("br_target", pct(BRANCH_S.target_bound), "{:.0f}", "target upper limit, %")
v("br_need", BRANCH_S.clean_runs_for_target, "{:.0f}", "clean runs for the target")
for _k, _key in (("original", "orig"), ("warning", "warn"), ("paginated", "pag")):
    _a, _b = BRANCH_S.acceptance_interval(_k)
    v(f"br_{_key}_lo", _a, "{:.3f}", f"{_k}: Wilson acceptance, low")
    v(f"br_{_key}_hi", _b, "{:.3f}", f"{_k}: Wilson acceptance, high")

print("== SWE-agent Table 3 (Literature.AgentRetrieval)")
v("sw_100", SWE.SweBenchLiteSweAgentResolved, "{:.1f}", "100-line window, % resolved")
v("sw_full", SWE.SweAgentFullFileViewerResolved, "{:.1f}", "whole files, % resolved")
v("sw_n", SWE.SweBenchLiteInstances, "{:.0f}", "SWE-bench Lite instances")

print("== Tool drift (Scenarios.TestLintToolDrift)")
v("js_base_tests", pct(DRIFT_S.baseline["tests"]), "{:.0f}", "baseline tests share, %")
v("js_base_lint", pct(DRIFT_S.baseline["lint"]), "{:.0f}", "baseline lint share, %")
v("js_val", DRIFT_S.divergence, "{:.3f}", "JS divergence, nats")
v("js_max", DRIFT_S.maximum, "{:.3f}", "maximum, ln 2")
v("js_share", pct(DRIFT_S.share_of_maximum), "{:.0f}", "share of the maximum, %")

print("== SPRT canary (Scenarios.AcceptanceCanaryMonitor)")
v("sp_p0", MON_S.p_base, "{:.2f}", "baseline acceptance")
v("sp_p1", MON_S.p_degraded, "{:.2f}", "degraded acceptance")
v("sp_alpha", pct(MON_S.alpha), "{:.0f}", "false hold, %")
v("sp_beta", pct(MON_S.beta), "{:.0f}", "missed degradation, %")
v("sp_fail", MON.failure_step, "{:.2f}", "step per failure")
v("sp_succ", -MON.success_step, "{:.2f}", "step per success (down)")
v("sp_upper", MON.upper, "{:.2f}", "upper boundary")
v("sp_lower", MON.lower, "{:.2f}", "lower boundary")
v("sp_n", MON.expected_samples, "{:.0f}", "expected labeled tasks, whole")
v("sp_n2", MON.expected_samples, "{:.2f}", "expected labeled tasks")
v("sp_hold", MON.failures_to_hold, "{:.0f}", "failures in a row to hold")
v("sp_hold_words", MON.failures_to_hold, "words", "the same, in words")
v("sp_hold_less", MON.failures_to_hold - 1, "{:.0f}", "one fewer")
v("sp_k_less", (MON.failures_to_hold - 1) * MON.failure_step, "{:.2f}", "score after one fewer")
v("sp_k_hold", MON.failures_to_hold * MON.failure_step, "{:.2f}", "score at the hold")
v("cn_rate", mag(MON_S.arrival_rate, "1/hour"), "{:.0f}", "eligible tasks an hour")
v("cn_frac", pct(MON_S.fraction), "{:.0f}", "canary share, %")
v("cn_frac_dec", MON_S.fraction, "{:.2f}", "canary share, fraction")
v("cn_delay", mag(MON_S.label_delay, "hour"), "{:.1f}", "label delay, h")
v("cn_delay_min", mag(MON_S.label_delay, "minute"), "{:.0f}", "label delay, min")
v("cn_cand_rate", mag(MON.candidate_rate, "1/hour"), "{:.0f}", "candidate tasks an hour")
v("cn_admit", mag(MON.admission_time, "hour"), "{:.2f}", "admission time, h")
v("cn_decide", mag(MON.decision_time, "hour"), "{:.2f}", "time to a decision, h")
v("cn_frac2", pct(MON2_S.fraction), "{:.0f}", "doubled share, %")
v("cn_decide2", mag(MON2.decision_time, "hour"), "{:.1f}", "time to a decision at the doubled share, h")
v("cn_decide_full", mag(MONF.decision_time, "hour"), "{:.2f}", "time to a decision at full allocation, h")
v("vt_canary", MON.excess_failures, "{:.0f}", "excess failures at the canary share")
v("vt_full", MON.full_excess_failures, "{:.0f}", "excess failures at full allocation")
v("vt_pending", MON.pending, "{:.1f}", "tasks admitted while labels pend, canary")
v("vt_full_pending", MON.full_pending, "{:.0f}", "the same, full allocation")
v("vt_gap", MON_S.p_base - MON_S.p_degraded, "{:.2f}", "extra failure probability per task")

print("== Argo ladder on agent tasks (Scenarios.WebCanaryLadderOnAgentTasks)")
_first_share, _pause = LADDER_S.stages[0]
v("ar_first", pct(_first_share), "{:.0f}", "first weight, %")
for _i, (_share, _) in enumerate(LADDER_S.stages[1:]):
    v(f"ar_w{_i + 1 + 1}", pct(_share), "{:.0f}", f"weight of step {_i + 1 + 1}, %")
v("ar_pause", mag(_pause, "minute"), "{:.0f}", "pause per step, min")
v("ar_total", mag(LADDER_S.total, "minute"), "{:.0f}", "fully promoted after, min")
v("ar_labeled", LADDER_S.labeled_by_completion, "{:.0f}", "labeled tasks at promotion")

print("== Release gate (Scenarios.ReleaseGateFamilywiseError)")
v("fw_k", FAMILY_S.family_size, "{:.0f}", "gate terms")
v("fw_k_words", FAMILY_S.family_size, "words", "the same, in words")
v("fw_alpha", pct(FAMILY_S.significance), "{:.0f}", "error per term, %")
v("fw_union", pct(FAMILY_S.union_bound), "{:.0f}", "union bound, %")
v("fw_ind", pct(FAMILY_S.independent), "{:.1f}", "independent errors, %")
v("fw_per", pct(FAMILY_S.per_test), "{:.2f}", "Bonferroni level per term, %")

if __name__ == "__main__":
    write(__file__)
