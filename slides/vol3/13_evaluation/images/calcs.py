#!/usr/bin/env python3
"""Every number on the Volume III Lecture 13 slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations, and the chapter reads the same
entries: repeated-run consistency (Scenarios.RecurringTaskConsistency and its
variants, horizon compounding over recurrences), the four-verdict denominator
(UnresolvedTrialsOfHundred), the judge's error rates (ResearchReportJudgeCalibration),
chance agreement (IndependentJudgesNinetyPercent), tasks against repeats
(RepairSuiteTaskRepeatBudget), Wald and Wilson at 50 tasks (FiftyTaskIntervals),
the paired comparison of candidate_v2 and baseline_v1
(RepairCandidatePairedComparison), one task against new tasks
(TwoTaskTypeConsistency), and batch against sequential selection
(CappedCandidateSelection), all backed by engine/solvers/acceptance.py. Public
data are Literature entries: Terminal-Bench scaffolds and SWE-bench Verified
runs (read through TerminalBenchScaffoldSpread and SweBenchVerifiedRankTies),
the tau-bench and METR figures (AgentEvaluation), Zheng et al.'s judges
(JudgePositionBias), and Kapoor et al.'s cost table (HumanEvalAgentCost). The
one other source is the deck's own data (data/): a small real evaluation run
for this lecture (eval_run.json, written by eval_run.py), summarized by
mlsysim.solvers.PairedRunEvaluationModel. This file only selects values,
converts units, and formats them; it types no numbers. Each value is recorded
with v() under the key the deck prints as \\val{key}; running this file writes
images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402
from mlsysim.physics import calc_independent_success  # noqa: E402
from mlsysim.solvers import PairedRunEvaluationModel  # noqa: E402

Scenarios = mlsysim.Scenarios
Literature = mlsysim.Literature
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

HOOK_S = Scenarios.RecurringTaskConsistency
HOOK_LO_S = Scenarios.RecurringTaskConsistencyLowerRate
VERD_S = Scenarios.UnresolvedTrialsOfHundred
JUDGE_S = Scenarios.ResearchReportJudgeCalibration
KAPPA_S = Scenarios.IndependentJudgesNinetyPercent
KAPPA_CHK_S = Scenarios.IndependentJudgesEightyPercent
TR_S = Scenarios.RepairSuiteTaskRepeatBudget
WL_S = Scenarios.FiftyTaskIntervals
PC_S = Scenarios.RepairCandidatePairedComparison
MIX_S = Scenarios.TwoTaskTypeConsistency
SEL_S = Scenarios.CappedCandidateSelection
SC_S = Scenarios.TerminalBenchScaffoldSpread
RK_S = Scenarios.SweBenchVerifiedRankTies

HOOK = HOOK_S.result
HOOK_LO = HOOK_LO_S.result
VERD = VERD_S.result
JUDGE = JUDGE_S.result
TR = TR_S.result
WL = WL_S.result
PC = PC_S.result
MIX = MIX_S.result
SEL = SEL_S.result
SC = SC_S.result
RK = RK_S.result
AE = Literature.AgentEvaluation
ZHENG = {r["judge"]: r for r in load_series("slides/vol3/13_evaluation/images/data/judge_position_bias.csv").entries}
KAPOOR = {r["agent"]: r for r in load_series("slides/vol3/13_evaluation/images/data/humaneval_agent_cost_accuracy.csv").entries}


def pct(x):
    return x * 100


def usd(q):
    return q.to("USD").magnitude


print("== Hook: pass^k over recurrences (Scenarios.RecurringTaskConsistency)")
_k = HOOK_S.long_horizon
_p = HOOK_S.step_success
v("hook_p", _p, "{:.1f}", "single-run acceptance")
v("hook_p_pct", pct(_p), "{:.0f}", "the same, %")
v("hook_fail_pct", pct(1 - _p), "{:.0f}", "single-run failure, %")
v("hook_k", _k, "{:.0f}", "recurrences")
v("hook_pow", HOOK.success[_k], "{:.2f}", "pass^k")
v("hook_pow_pct", pct(HOOK.success[_k]), "{:.0f}", "the same, %")
v("hook_p_lo_pct", pct(HOOK_LO_S.step_success), "{:.0f}", "lower single-run rate, %")
v("hook_pow_lo_pct", pct(HOOK_LO.success[_k]), "{:.0f}", "pass^k at the lower rate, %")
v("hook_at_k_pct", pct(calc_independent_success(_p, _k)), "{:.0f}", "pass@k, %")
v("hook_target_pct", pct(HOOK_S.target_success), "{:.0f}", "consistency to hold, %")
v("hook_need_pct", pct(1 - HOOK.required_step_error), "{:.1f}", "single-run rate that holds it, %")
v("hook_fail_need_pct", pct(HOOK.required_step_error), "{:.1f}", "single-run failure that holds it, %")
v("hook_cut", HOOK.error_ratio, "{:.1f}", "failure-rate cut, x")

print("== tau-bench abstract (Literature.AgentEvaluation)")
v("tau_success_pct", AE.TauBenchSuccessUpperBoundPct, "{:.0f}", "task success, upper bound, %")
v("tau_pow_pct", AE.TauBenchRetailPassPowUpperBoundPct, "{:.0f}", "retail pass^k, upper bound, %")
v("tau_k", AE.TauBenchPassPowK, "{:.0f}", "k")

print("== Same model, many scaffolds (Scenarios.TerminalBenchScaffoldSpread)")
v("sc_lo", pct(SC.widest_low), "{:.0f}", "widest model, lowest score, %")
v("sc_hi", pct(SC.widest_high), "{:.0f}", "widest model, highest score, %")
v("sc_n", SC.widest_count, "{:.0f}", "its scaffolds")
v("sc_min", SC_S.min_scaffolds, "{:.0f}", "minimum scaffolds per model")
v("sc_models", SC.models, "{:.0f}", "models kept")
v("sc_median", pct(SC.median_spread), "{:.0f}", "median spread, points")
v("sc_se", pct(SC.median_se), "{:.0f}", "median reported standard error, points")

print("== Unresolved trials (Scenarios.UnresolvedTrialsOfHundred)")
v("vd_n", VERD.trials, "{:.0f}", "trials")
v("vd_pass", VERD_S.passed, "{:.0f}", "accepted")
v("vd_fail", VERD_S.failed, "{:.0f}", "failed")
v("vd_unres", VERD_S.unresolved, "{:.0f}", "unresolved")
v("vd_rate", pct(VERD.default_rate), "{:.0f}", "default rate, %")
v("vd_upper", pct(VERD.upper_bound), "{:.0f}", "upper bound, %")
v("vd_drop", pct(VERD.dropped_rate), "{:.1f}", "unresolved dropped, %")

print("== Our run (data/eval_run.json, PairedRunEvaluationModel)")
RUN_DATA = json.loads((DATA / "eval_run.json").read_text())
TASKS = [t["id"] for t in RUN_DATA["tasks"]]
BASE, CAND = "baseline_v1", "candidate_v2"
RUN = PairedRunEvaluationModel().solve(RUN_DATA["rows"], TASKS, RUN_DATA["runs_done"], BASE, CAND)
_spec = importlib.util.spec_from_file_location("eval_run", DATA / "eval_run.py")
_er = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_er)
_orig_visible = sum(_er.run_checks(t["code"], [t["visible"]])[0] == "pass" for t in _er.TASKS)
_hidden = [len(t["hidden"]) for t in RUN_DATA["tasks"]]
v("run_tasks", RUN.tasks, "{:.0f}", "tasks")
v("run_tasks_words", RUN.tasks, "words", "the same, in words")
v("run_runs", RUN.runs, "{:.0f}", "runs per task and configuration")
v("run_trials", RUN.trials[BASE], "{:.0f}", "trials per configuration")
v("run_temp", RUN_DATA["temperature"], "{:.1f}", "sampling temperature")
v("run_cutoff", RUN_DATA["cutoff_s"], "{:.0f}", "grader cutoff, s")
v("run_hidden_min_words", min(_hidden), "words", "fewest hidden tests per task")
v("run_hidden_max_words", max(_hidden), "words", "most hidden tests per task")
v("run_orig_vis", _orig_visible, "{:.0f}", "buggy originals passing their visible test")
for _cfg, _tag in ((BASE, "b"), (CAND, "c")):
    v(f"run_{_tag}_vis", RUN.visible_pass[_cfg], "{:.0f}", f"{_cfg} visible passes")
    v(f"run_{_tag}_vis_pct", pct(RUN.visible_pass[_cfg] / RUN.trials[_cfg]), "{:.1f}", f"{_cfg} visible pass, %")
    v(f"run_{_tag}_vis_rej", RUN.visible_pass_sealed_reject[_cfg], "{:.0f}", f"{_cfg} visible passes the sealed tests reject")
    v(f"run_{_tag}_vis_rej_pct", pct(RUN.visible_pass_sealed_reject[_cfg] / RUN.visible_pass[_cfg]), "{:.1f}",
      f"{_cfg} share of visible passes rejected, %")
    v(f"run_{_tag}_pct", pct(RUN.rate[_cfg]), "{:.1f}", f"{_cfg} sealed acceptance, %")
    v(f"run_{_tag}_calls", RUN.calls[_cfg], "{:.0f}", f"{_cfg} model calls")
    v(f"run_{_tag}_acc", RUN.accepted[_cfg], "{:.0f}", f"{_cfg} accepted")
    v(f"run_{_tag}_tokens", RUN.tokens[_cfg], "{:,.0f}", f"{_cfg} tokens, all trials")
    v(f"run_{_tag}_tpa", RUN.tokens_per_accepted[_cfg], "{:.0f}", f"{_cfg} tokens per accepted")
v("run_extra_calls", RUN.calls[CAND] - RUN.calls[BASE], "{:.0f}", "calls the retry added")
v("run_tpa_more_pct", pct(RUN.tokens_per_accepted_ratio - 1), "{:.0f}", "more tokens per accepted, %")
v("run_both", RUN.both, "{:.0f}", "run pairs both accepted")
v("run_cand_only", RUN.candidate_only, "{:.0f}", "candidate only")
v("run_base_only", RUN.baseline_only, "{:.0f}", "baseline only")
v("run_neither", RUN.neither, "{:.0f}", "neither")
v("run_gain", pct(RUN.difference), "{:.1f}", "gain, points")
v("run_lower_abs", -pct(RUN.lower), "{:.1f}", "paired interval lower limit, minus points")
v("run_upper", pct(RUN.upper), "{:.1f}", "paired interval upper limit, points")
v("run_p", RUN.exact_p, "{:.3f}", "McNemar exact p")
v("run_c_at_max", pct(RUN.pass_at[CAND][RUN.runs]), "{:.1f}", "candidate pass@8, %")
v("run_b_at_max", pct(RUN.pass_at[BASE][RUN.runs]), "{:.1f}", "baseline pass@8, %")
v("run_pow_max", pct(RUN.pass_pow[CAND][RUN.runs]), "{:.1f}", "candidate pass^8, %")
v("run_b_pow_max", pct(RUN.pass_pow[BASE][RUN.runs]), "{:.1f}", "baseline pass^8, %")

print("== METR, o3 (Literature.AgentEvaluation)")
v("metr_hacks", AE.MetrO3ReBenchHackRuns, "{:.0f}", "RE-Bench runs judged reward hacking")
v("metr_runs", AE.MetrO3ReBenchRuns, "{:.0f}", "RE-Bench runs")

print("== Judge B against a reference (Scenarios.ResearchReportJudgeCalibration)")
v("gr_reports", JUDGE_S.reference_failures + JUDGE_S.reference_passes, "{:.0f}", "reports")
v("gr_ref_fail", JUDGE_S.reference_failures, "{:.0f}", "reference failures")
v("gr_ref_pass", JUDGE_S.reference_passes, "{:.0f}", "reference passes")
v("gr_acc_fail", JUDGE_S.accepted_failures, "{:.0f}", "failures accepted")
v("gr_rej_fail", JUDGE_S.reference_failures - JUDGE_S.accepted_failures, "{:.0f}", "failures rejected")
v("gr_acc_pass", JUDGE_S.accepted_passes, "{:.0f}", "passes accepted")
v("gr_rej_pass", JUDGE_S.reference_passes - JUDGE_S.accepted_passes, "{:.0f}", "passes rejected")
v("gr_accepted", JUDGE.accepted, "{:.0f}", "all accepted")
v("gr_far", pct(JUDGE.false_accept_rate), "{:.0f}", "false-acceptance rate, %")
v("gr_far_lo", pct(JUDGE.false_accept_low), "{:.1f}", "Wilson low, %")
v("gr_far_hi", pct(JUDGE.false_accept_high), "{:.1f}", "Wilson high, %")
v("gr_fam", pct(JUDGE.failure_among_accepted), "{:.1f}", "failure among accepted, %")

print("== Zheng et al. 2023, Table 2 (slides/vol3/13_evaluation/images/data/judge_position_bias.csv)")
for _judge, _tag in (("GPT-4", "g4"), ("GPT-3.5", "g35"), ("Claude-v1", "cl")):
    v(f"zh_{_tag}_cons", ZHENG[_judge]["consistency_pct"], "{:.1f}", f"{_judge} consistent, %")
    v(f"zh_{_tag}_first", ZHENG[_judge]["biased_first_pct"], "{:.1f}", f"{_judge} favors first, %")

print("== Chance agreement (Scenarios.IndependentJudgesNinetyPercent)")
_pass, _fail = KAPPA_S.class_shares
v("kc_p_pct", pct(_pass), "{:.0f}", "each judge passes, %")
v("kc_p", _pass, "{:.1f}", "the same, fraction")
v("kc_q", _fail, "{:.1f}", "each judge fails, fraction")
v("kc_both_pass", _pass * _pass, "{:.2f}", "both pass")
v("kc_both_fail", _fail * _fail, "{:.2f}", "both fail")
v("kc_both_pass_pct", pct(_pass * _pass), "{:.0f}", "both pass, %")
v("kc_both_fail_pct", pct(_fail * _fail), "{:.0f}", "both fail, %")
v("kc_split_pct", pct(_pass * _fail), "{:.0f}", "one passes, the other fails, %")
v("kc_po_pct", pct(KAPPA_S.agreement), "{:.0f}", "observed agreement, %")
v("kc_pe", KAPPA_S.chance_agreement, "{:.2f}", "chance agreement")
v("kc_pe_pct", pct(KAPPA_S.chance_agreement), "{:.0f}", "the same, %")
v("kc_kappa", KAPPA_S.kappa, "{:.0f}", "kappa")
v("kc_chk_pct", pct(KAPPA_CHK_S.class_shares[0]), "{:.0f}", "checkpoint pass rate, %")
v("kc_chk_po_pct", pct(KAPPA_CHK_S.agreement), "{:.0f}", "checkpoint agreement, %")
v("kc_chk_kappa", KAPPA_CHK_S.kappa, "{:.0f}", "checkpoint kappa")

print("== Tasks or repeats (Scenarios.RepairSuiteTaskRepeatBudget)")
v("tr_mean", TR_S.mean_rate, "{:.1f}", "pilot mean acceptance")
v("tr_mean_c", 1 - TR_S.mean_rate, "{:.1f}", "its complement")
v("tr_vb", TR_S.between_task_variance, "{:.2f}", "between-task variance")
v("tr_vw", TR.within_variance, "{:.2f}", "within-task variance")
v("tr_ratio", TR.variance_ratio, "{:.0f}", "variance ratio, runs")
v("tr_budget", TR_S.budget_runs, "{:.0f}", "graded runs")
v("tr_fixture", TR_S.fixture_cost_runs, "{:.0f}", "fixture cost, runs")
for _r in TR_S.free_runs:
    v(f"tr_free_n{_r}", TR.free_tasks[_r], "{:.0f}", f"tasks at {_r} runs, new tasks free")
    v(f"tr_free_r{_r}", _r, "{:.0f}", "runs per task")
    v(f"tr_free_se{_r}", pct(TR.free_se[_r]), "{:.2f}", "standard error, points")
_free_best = min(TR.free_se, key=TR.free_se.get)
v("tr_free_best_words", _free_best, "words", "best runs per task, new tasks free")
v("tr_best_r", TR.best_runs, "{:.0f}", "best runs per task with fixture cost")
v("tr_best_n", TR.best_tasks, "{:.0f}", "its tasks")
v("tr_best_se", pct(TR.best_se), "{:.2f}", "its standard error, points")
v("tr_one_n", TR.paid_tasks[1], "{:.0f}", "tasks run once with fixture cost")
v("tr_one_se", pct(TR.paid_se[1]), "{:.2f}", "their standard error, points")

print("== Leaderboard ties (Scenarios.SweBenchVerifiedRankTies)")
v("rk_models", RK.models, "{:.0f}", "models")
v("rk_top", RK.top, "{:.0f}", "shown")
v("rk_pairs", RK.pairs, "{:.0f}", "neighboring pairs")
v("rk_tied", RK.tied, "{:.0f}", "tied pairs")
v("rk_top_score", pct(RK.top_score), "{:.1f}", "top score, %")
v("rk_last_score", pct(RK.last_score), "{:.1f}", "last shown score, %")
v("rk_se", pct(RK.typical_se), "{:.2f}", "median standard error, points")
v("rk_se_round", pct(RK.typical_se), "{:.0f}", "the same, whole points")
v("rk_gap", pct(RK.needed_gap), "{:.2f}", "gap needed, points")
v("rk_gap_round", pct(RK.needed_gap), "{:.1f}", "the same, one decimal")
v("conf_pct", pct(RK_S.confidence), "{:.0f}", "confidence, %")
v("z95", WL.z_critical, "{:.2f}", "two-sided 95 percent quantile")

print("== Wald and Wilson (Scenarios.FiftyTaskIntervals)")
_x0, _x2, _x25 = WL_S.successes
v("wl_n", WL_S.trials, "{:.0f}", "tasks")
v("wl_cases_words", len(WL_S.successes), "words", "counts compared")
v("wl_x0_words", _x0, "words", "first count, in words")
v("wl_x0", _x0, "{:.0f}", "first count")
v("wl_x2", _x2, "{:.0f}", "second count")
v("wl_x25", _x25, "{:.0f}", "third count")
v("wl_x0_hi_round", pct(WL.wilson[_x0][1]), "{:.0f}", "Wilson upper at the first count, whole %")
v("wl_x0_hi", pct(WL.wilson[_x0][1]), "{:.2f}", "Wilson upper at the first count, %")
v("wl_x2_wald_lo_abs", -pct(WL.wald[_x2][0]), "{:.2f}", "Wald lower at the second count, minus %")
v("wl_x2_lo", pct(WL.wilson[_x2][0]), "{:.2f}", "Wilson lower at the second count, %")
v("wl_x2_hi", pct(WL.wilson[_x2][1]), "{:.2f}", "Wilson upper at the second count, %")

print("== Paired comparison (Scenarios.RepairCandidatePairedComparison)")
v("pc_n", PC.pairs, "{:.0f}", "task pairs")
v("pc_both", PC_S.both, "{:.0f}", "both accepted")
v("pc_cand", PC_S.candidate_only, "{:.0f}", "candidate only")
v("pc_base", PC_S.baseline_only, "{:.0f}", "baseline only")
v("pc_neither", PC_S.neither, "{:.0f}", "neither")
v("pc_disc_diff", PC_S.candidate_only - PC_S.baseline_only, "{:.0f}", "discordant difference")
v("pc_tol", pct(PC_S.tolerance), "{:.0f}", "tolerance, points")
v("pc_gain", pct(PC.difference), "{:.0f}", "gain, points")
v("pc_lower_abs", -pct(PC.lower), "{:.2f}", "lower limit, minus points")
v("pc_upper", pct(PC.upper), "{:.2f}", "upper limit, points")
v("pc_lower_width", pct(PC.lower_width), "{:.2f}", "lower half-width, points")
v("pc_clear", pct(PC.clear_width), "{:.0f}", "half-width the gate needs, points")
v("pc_gain_holds", PC.pairs_if_gain_holds, "{:,.0f}", "pairs if the gain holds")
v("pc_no_gain", PC.pairs_if_no_gain, "{:,.0f}", "pairs if the true gain is zero")
v("pc_no_gain_round", round(PC.pairs_if_no_gain, -2), "{:,.0f}", "the same, as the chapter rounds it")
v("pc_p", PC.exact_p, "{:.3f}", "McNemar exact p")
v("pc_share_pct", pct(PC.paired_share), "{:.0f}", "paired variance share, %")
v("pc_mult", PC.independent_multiple, "{:.1f}", "tasks independent arms need, x")

print("== One task against new tasks (Scenarios.TwoTaskTypeConsistency)")
v("mix_same_pct", pct(MIX.same_task_all), "{:.0f}", "one task repeated, %")
v("mix_new_pct", pct(MIX.new_tasks_all), "{:.0f}", "new tasks, %")

print("== Kapoor et al. 2024, Table A1 (slides/vol3/13_evaluation/images/data/humaneval_agent_cost_accuracy.csv)")
_lats, _warm = KAPOOR["LATS (GPT-4)"], KAPOOR["Warming (GPT-4)"]
_retry, _gpt4 = KAPOOR["Retry (GPT-4)"], KAPOOR["GPT-4 (baseline)"]
v("ha_problems", AE.HumanEvalProblems, "{:.0f}", "HumanEval problems")
v("ha_runs", AE.KapoorRunsPerAgent, "{:.0f}", "runs per agent")
v("ha_runs_words", AE.KapoorRunsPerAgent, "words", "the same, in words")
v("ha_lats_acc", _lats["accuracy_pct"], "{:.1f}", "LATS (GPT-4) accuracy, %")
v("ha_lats_cost", _lats["total_cost_usd"], "{:.2f}", "LATS (GPT-4) cost, $")
v("ha_warm_acc", _warm["accuracy_pct"], "{:.1f}", "Warming accuracy, %")
v("ha_warm_cost", _warm["total_cost_usd"], "{:.2f}", "Warming cost, $")
v("ha_retry_acc", _retry["accuracy_pct"], "{:.1f}", "Retry accuracy, %")
v("ha_retry_cost", _retry["total_cost_usd"], "{:.2f}", "Retry cost, $")
v("ha_gpt4_acc", _gpt4["accuracy_pct"], "{:.1f}", "GPT-4 one call accuracy, %")
v("ha_gpt4_cost", _gpt4["total_cost_usd"], "{:.2f}", "GPT-4 one call cost, $")
v("ha_ratio", _lats["total_cost_usd"] / _warm["total_cost_usd"], "{:.0f}", "LATS over Warming cost, x")

print("== Batch or sequential (Scenarios.CappedCandidateSelection)")
_gen, _chk = usd(SEL_S.generation_cost), usd(SEL_S.check_cost)
v("sel_p", SEL_S.pass_rate, "{:.1f}", "pass probability")
v("sel_p_c", 1 - SEL_S.pass_rate, "{:.1f}", "its complement")
v("sel_cap", SEL_S.cap, "{:.0f}", "candidate cap")
v("sel_cap_words", SEL_S.cap, "words", "the same, in words")
v("sel_gen", _gen, "{:.2f}", "generation, $")
v("sel_chk", _chk, "{:.2f}", "checking, $")
v("sel_unit", _gen + _chk, "{:.2f}", "one candidate, $")
v("sel_q", SEL.availability, "{:.3f}", "availability")
v("sel_q_pct", pct(SEL.availability), "{:.1f}", "the same, %")
v("sel_en", SEL.expected_candidates, "{:.2f}", "sequential candidates")
v("sel_batch", usd(SEL.batch_trial_cost), "{:.2f}", "batch per trial, $")
v("sel_seq", usd(SEL.sequential_trial_cost), "{:.4f}", "sequential per trial, $")
v("sel_batch_acc", usd(SEL.batch_per_accepted), "{:.3f}", "batch per accepted, $")
v("sel_seq_acc", usd(SEL.sequential_per_accepted), "{:.3f}", "sequential per accepted, $")
v("sel_doubled", usd(SEL.doubled_rate_per_accepted), "{:.2f}", "sequential with p doubled, $")
v("sel_batch_cents", usd(SEL.batch_trial_cost) * 100, "{:.0f}", "batch per trial, cents")
v("sel_seq_cents", usd(SEL.sequential_trial_cost) * 100, "{:.2f}", "sequential per trial, cents")
v("sel_batch_acc_cents", usd(SEL.batch_per_accepted) * 100, "{:.1f}", "batch per accepted, cents")
v("sel_seq_acc_cents", usd(SEL.sequential_per_accepted) * 100, "{:.1f}", "sequential per accepted, cents")
v("sel_premium_pct", pct(SEL.batch_premium), "{:.0f}", "batch premium, %")

if __name__ == "__main__":
    write(__file__)
