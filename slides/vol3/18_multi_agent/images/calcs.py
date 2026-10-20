#!/usr/bin/env python3
"""Every number on the Volume III Lecture 18 slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations, and the chapter reads the same
entries: the work-partition screen (RepairPartitionScreen, its halved-overhead
variant, and the 90 percent checkpoint partition with its two variants), the
universal scalability law for the repair partition and the research agent
(RepairPartitionScalability, ResearchFanoutScalability), the token break-even
(ResearchSystemTokenBreakEven), one round of messages (EightAgentMessageRound),
the dependency schedule (InterfaceMigrationSchedule and its shorter-node
variants), reference passing (SixChildReferencePassing), shared writes
(ThousandFileSharedWrites, TwoThousandFileSharedWrites), correlated reviewers
(FiveCorrelatedReviewers), the fan-out deadline (EightWorkerDeadlineFanout),
the credit window (DiffReviewCreditWindow), split allowances
(RemediationChildAllowances), and the matched cohort (MatchedDelegationCohort),
backed by physics/coordination.py. The recall slide reads the reinforcement
learning lecture's verifier batch (VerifierFalseAcceptsAtTenPercentPass).
Published data are Literature entries: the research system's report
(MultiAgentSystems; its token multiples are on Agents.Research.OrchestratorWorker),
correlated model errors on Helm (CorrelatedModelErrors), fan-out tail latency
and hedging (FanoutTailLatency), and the MAST failure taxonomy (MultiAgentFailures).
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
Agents = mlsysim.Agents

RESEARCH = Agents.Research.OrchestratorWorker
REPORT = Literature.MultiAgentSystems
HELM = Literature.CorrelatedModelErrors
TAIL = Literature.FanoutTailLatency
MAST = Literature.MultiAgentFailures

BE = Scenarios.ResearchSystemTokenBreakEven
RC_S = Scenarios.VerifierFalseAcceptsAtTenPercentPass
RC = RC_S.result
SCR = Scenarios.RepairPartitionScreen
SCR_H = Scenarios.RepairPartitionScreenHalvedOverhead
EST = Scenarios.NinetyPercentPartitionScreen
EST_HC = Scenarios.NinetyPercentPartitionScreenHalvedOverhead
EST_HS = Scenarios.NinetyPercentPartitionScreenHalvedSerial
USL = Scenarios.RepairPartitionScalability
USL_R = Scenarios.ResearchFanoutScalability
MSG = Scenarios.EightAgentMessageRound
DAG = Scenarios.InterfaceMigrationSchedule
DAG_V5 = Scenarios.InterfaceMigrationScheduleShorterV5
DAG_V6 = Scenarios.InterfaceMigrationScheduleShorterV6
REF = Scenarios.SixChildReferencePassing
WR = Scenarios.ThousandFileSharedWrites
WR2 = Scenarios.TwoThousandFileSharedWrites
RV = Scenarios.FiveCorrelatedReviewers
FO = Scenarios.EightWorkerDeadlineFanout
CW = Scenarios.DiffReviewCreditWindow
AL = Scenarios.RemediationChildAllowances
CO = Scenarios.MatchedDelegationCohort

# The seven most frequent MAST failure modes, as the retrieval slide draws them.
MAST_MODES = [  # (label, share in percent, category)
    ("Step repetition", MAST.MastStepRepetitionPct, "design"),
    ("Reasoning-action mismatch", MAST.MastReasoningActionMismatchPct, "align"),
    ("Unaware of termination", MAST.MastUnawareOfTerminationPct, "design"),
    ("Disobey task specification", MAST.MastDisobeyTaskSpecificationPct, "design"),
    ("Incorrect verification", MAST.MastIncorrectVerificationPct, "verify"),
    ("No or incomplete verification", MAST.MastIncompleteVerificationPct, "verify"),
    ("Task derailment", MAST.MastTaskDerailmentPct, "align"),
]


def pct(x):
    return x * 100


def usd(q):
    return q.to("USD").magnitude


def s(q):
    return q.to("s").magnitude


print("== Hook and NM 18.6 (Scenarios.ResearchSystemTokenBreakEven; Literature.MultiAgentSystems)")
v("hook_agent", BE.agent_tokens_vs_chat, "{:.0f}", "agent tokens over chat, x")
v("hook_multi", BE.multi_agent_tokens_vs_chat, "{:.0f}", "multi-agent tokens over chat, x")
v("hook_gain_pct", REPORT.ResearchSystemEvalGainPct, "{:.1f}", "reported evaluation gain, %")
v("hook_m", BE.token_multiple, "{:.2f}", "token multiple per attempt, x")
v("hook_breakeven_pct", pct(BE.max_single_acceptance), "{:.1f}", "break-even single-agent acceptance, %")
v("hook_acc_mult", BE.acceptance_multiplier, "{:.3f}", "the gain as an acceptance multiple, x")
v("hook_cost_ratio", BE.cost_ratio_if_gain_is_acceptance, "{:.2f}", "cost per accepted task then, x")
v("be_single_pct", pct(BE.single_acceptance), "{:.0f}", "a single agent that accepts, %")
v("be_need_pct", pct(BE.required_multi_acceptance), "{:.0f}", "delegated acceptance it would need, %")
v("rs_early", REPORT.ResearchSystemEarlySubagents, "{:.0f}", "subagents early versions spawned")
v("rs_var", REPORT.ResearchSystemTokenVariancePct, "{:.0f}", "variance explained by tokens, %")

print("== Recall (Scenarios.VerifierFalseAcceptsAtTenPercentPass)")
v("rc_rollouts", RC.rollouts, "{:,.0f}", "rollouts")
v("rc_pass_pct", pct(RC_S.pass_rate), "{:.0f}", "true pass rate, %")
v("rc_pass", RC_S.pass_rate, "{:.2f}", "the same, as a fraction")
v("rc_fail", 1 - RC_S.pass_rate, "{:.2f}", "failing share")
v("rc_fa_pct", pct(RC_S.false_accept_rate), "{:.1f}", "false-accept rate, %")
v("rc_fa", RC_S.false_accept_rate, "{:.3f}", "the same, as a fraction")
v("rc_true", RC.true_accepts, "{:.2f}", "true accepts")
v("rc_true1", RC.true_accepts, "{:.1f}", "true accepts, one decimal")
v("rc_false", RC.false_accepts, "{:.2f}", "false accepts")
v("rc_accepts", RC.accepts, "{:,.2f}", "all accepts")
v("rc_false_pct", pct(RC.false_share), "{:.1f}", "false share of rewards, %")

print("== NM 18.1 screen (Scenarios.RepairPartitionScreen and its halved-overhead variant)")
v("scr_p", SCR.parallel_fraction, "{:.2f}", "parallel fraction")
v("scr_p1", SCR.parallel_fraction, "{:.1f}", "the same, one decimal")
v("scr_c", SCR.overhead_per_extra_worker, "{:.2f}", "overhead per added worker")
v("scr_t4", SCR.elapsed[4], "{:.3f}", "elapsed at four workers")
v("scr_t8", SCR.elapsed[8], "{:.3f}", "elapsed at eight workers")
v("scr_best", SCR.best_workers, "{:.0f}", "best worker count")
v("scr_best_words", SCR.best_workers, "words", "the same, in words")
v("scr_tbest", SCR.best_time, "{:.3f}", "elapsed at the best count")
v("scr_speedup", SCR.best_speedup, "{:.1f}", "speedup at the best count, x")
v("scr_cap", SCR.speedup_cap, "{:.0f}", "ceiling 1/(1-p), x")
v("scr_sqrt", SCR.continuous_best, "{:.1f}", "sqrt(p/c)")
v("scr_c_half", SCR_H.overhead_per_extra_worker, "{:.2f}", "halved overhead")
v("scr_half_best", SCR_H.best_workers, "{:.0f}", "best count, halved overhead")
v("scr_half_speedup", SCR_H.best_speedup, "{:.1f}", "speedup, halved overhead, x")

print("== Checkpoint 18.1 estimate (Scenarios.NinetyPercentPartitionScreen and variants)")
v("est_p", EST.parallel_fraction, "{:.1f}", "parallel fraction")
v("est_ratio", EST.parallel_fraction / EST.overhead_per_extra_worker, "{:.0f}", "p/c")
v("est_sqrt", EST.continuous_best, "{:.2f}", "sqrt(p/c)")
v("est_best", EST.best_workers, "{:.0f}", "best count")
v("est_speedup", EST.best_speedup, "{:.2f}", "speedup, x")
v("est_cap", EST.speedup_cap, "{:.0f}", "ceiling, x")
v("est_hc_best", EST_HC.best_workers, "{:.0f}", "halved overhead: best count")
v("est_hc_speedup", EST_HC.best_speedup, "{:.2f}", "halved overhead: speedup, x")
v("est_hs_p", EST_HS.parallel_fraction, "{:.2f}", "halved serial: parallel fraction")
v("est_hs_best", EST_HS.best_workers, "{:.0f}", "halved serial: best count")
v("est_hs_best_words", EST_HS.best_workers, "words", "the same, in words")
v("est_hs_speedup", EST_HS.best_speedup, "{:.2f}", "halved serial: speedup, x")
v("est_hs_cap", EST_HS.speedup_cap, "{:.0f}", "halved serial: ceiling, x")
v("est_extra", EST_HC.best_workers - EST.best_workers, "{:.0f}", "added workers the overhead cut needs")
v("est_extra_words", EST_HC.best_workers - EST.best_workers, "words", "the same, in words")

print("== USL (Scenarios.RepairPartitionScalability, ResearchFanoutScalability)")
v("usl_s", USL.contention, "{:.2f}", "repair sigma")
v("usl_k", USL.coherency, "{:.2f}", "repair kappa")
v("usl_rs", USL_R.contention, "{:.2f}", "research sigma")
v("usl_rk", USL_R.coherency, "{:.3f}", "research kappa")
v("usl_best", USL.best_workers, "{:.0f}", "repair peak count")
v("usl_peak", USL.peak_capacity, "{:.1f}", "repair peak capacity, x")
v("usl_slower", USL.first_slower, "{:.0f}", "first count slower than one worker")
v("usl_cap_slower", USL.capacity[USL.first_slower], "{:.3f}", "capacity there")
v("usl_rbest", USL_R.best_workers, "{:.0f}", "research peak count")
v("usl_rpeak", USL_R.peak_capacity, "{:.1f}", "research peak capacity, x")
v("usl_fan", USL_R.best_workers / USL.best_workers, "{:.1f}", "fan-out ratio, x")

print("== One message round (Scenarios.EightAgentMessageRound)")
v("msg_n_words", MSG.workers, "words", "agents")
v("msg_mesh", MSG.mesh_deliveries, "{:.0f}", "full-mesh deliveries")
v("msg_sup", MSG.supervisor_deliveries, "{:.0f}", "supervisor deliveries")

print("== NM 18.2 schedule (Scenarios.InterfaceMigrationSchedule and variants)")
v("dag_work", s(DAG.work), "{:.0f}", "aggregate slot work, s")
v("dag_slots", DAG.slots, "{:.0f}", "slots")
v("dag_bound", s(DAG.work_bound), "{:.2f}", "work bound, s")
v("dag_span", s(DAG.makespan), "{:.1f}", "critical path, s")
v("dag_peak", DAG.peak_slots, "{:.0f}", "most slots busy")
v("dag_speedup", DAG.speedup, "{:.2f}", "speedup over sequence, x")
v("dag_slack", s(DAG.free_float[6]), "{:.1f}", "slack on the v4-v6 branch, s")
v("dag_cut", s(DAG.durations[5] - DAG_V5.durations[5]), "{:.0f}", "proposed cut, s")
v("dag_span_v5", s(DAG_V5.makespan), "{:.1f}", "critical path with v5 cut, s")
v("dag_span_v6", s(DAG_V6.makespan), "{:.1f}", "critical path with v6 cut, s")

print("== NM 18.3 reference passing (Scenarios.SixChildReferencePassing)")
v("ref_n_words", REF.workers, "words", "children")
v("ref_slice", REF.slice_tokens, "{:,.0f}", "pasted slice, tokens")
v("ref_env", REF.reference_tokens, "{:.0f}", "reference, tokens")
v("ref_read", REF.read_tokens, "{:,.0f}", "reads, tokens")
v("ref_tok", REF.reference_total_tokens, "{:,.0f}", "reference total, tokens")
v("ref_paste", REF.pasted_tokens, "{:,.0f}", "pasted total, tokens")
v("ref_red", REF.reduction, "{:.1f}", "reduction, x")
v("ref_be", REF.break_even_read_tokens, "{:,.0f}", "reads at which the saving vanishes")

print("== Equation 4, shared writes (Scenarios.ThousandFileSharedWrites, TwoThousandFileSharedWrites)")
v("bd_files", WR.files, "{:,.0f}", "files")
v("bd_w", WR.writes_per_worker, "{:.0f}", "writes per worker")
v("bd_small_words", WR.small_fleet, "words", "small fleet")
v("bd_large_words", WR.large_fleet, "words", "large fleet")
v("bd_even_words", WR.even_odds_fleet, "words", "first fleet above even odds")
v("bd_p2", pct(WR.collision[WR.small_fleet]), "{:.1f}", "shared write, small fleet, %")
v("bd_p5", pct(WR.collision[WR.even_odds_fleet]), "{:.1f}", "shared write at even odds, %")
v("bd_p8", pct(WR.collision[WR.large_fleet]), "{:.1f}", "shared write, large fleet, %")
v("bd_d2", pct(WR.disjoint[WR.small_fleet]), "{:.1f}", "disjoint, small fleet, %")
v("bd_files2", WR2.files, "{:,.0f}", "larger repository, files")
v("bd_even2", WR2.even_odds_fleet, "{:.0f}", "its first fleet above even odds")

print("== Correlated model errors (Literature.CorrelatedModelErrors)")
v("kim_models", HELM.HelmModels, "{:.0f}", "models")
v("kim_q", HELM.HelmQuestions, "{:,.0f}", "questions")
v("kim_pct", HELM.HelmSameWrongAnswerPct, "{:.0f}", "same wrong answer, %")
v("kim_wrong", HELM.HelmIncorrectChoices, "{:.0f}", "incorrect choices per question")
v("kim_wrong_words", HELM.HelmIncorrectChoices, "words", "the same, in words")
v("kim_random_pct", pct(1 / HELM.HelmIncorrectChoices), "{:.0f}", "uniform random agreement, %")

print("== NM 18.4 reviewers (Scenarios.FiveCorrelatedReviewers)")
v("rv_n_words", RV.reviewers, "words", "reviewers")
v("rv_q", pct(RV.false_approval), "{:.0f}", "false approval, %")
v("rv_q_dec", RV.false_approval, "{:.2f}", "the same, as a fraction")
v("rv_rho", RV.correlation, "{:.2f}", "correlation")
v("rv_maj_ind", pct(RV.independent_majority(RV.reviewers)), "{:.2f}", "majority, independent, %")
v("rv_maj_mix", pct(RV.correlated_majority(RV.reviewers)), "{:.2f}", "majority, correlated, %")
v("rv_large", RV.large_ensemble, "{:.0f}", "larger panel")
v("rv_maj15_ind", pct(RV.independent_majority(RV.large_ensemble)), "{:.2f}", "larger panel, independent, %")
v("rv_maj15_mix", pct(RV.correlated_majority(RV.large_ensemble)), "{:.1f}", "larger panel, correlated, %")
v("rv_floor", pct(RV.floor), "{:.1f}", "floor, %")
v("rv_unan_ind", pct(RV.independent_unanimous), "{:.3f}", "unanimous, independent, %")
v("rv_unan_mix", pct(RV.correlated_unanimous), "{:.2f}", "unanimous, correlated, %")
v("rv_ratio", RV.unanimous_ratio, "{:.0f}", "unanimous ratio, x")
v("rv_more_words", RV.large_ensemble - RV.reviewers, "words", "reviewers added")
v("rv_spend_words", RV.large_ensemble / RV.reviewers, "words", "review spend multiple")
v("rv_drop_words", round(pct(RV.correlated_majority(RV.reviewers) - RV.correlated_majority(RV.large_ensemble))),
  "words", "points the larger panel buys, about")

print("== NM 18.5 fan-out deadline (Scenarios.EightWorkerDeadlineFanout)")
v("fo_n", FO.workers, "{:.0f}", "workers")
v("fo_n_words", FO.workers, "words", "the same, in words")
v("fo_k", FO.required, "{:.0f}", "quorum")
v("fo_p", FO.on_time_probability, "{:.1f}", "on-time probability")
v("fo_p_pct", pct(FO.on_time_probability), "{:.0f}", "the same, %")
v("fo_all", pct(FO.all_on_time), "{:.2f}", "all on time, %")
v("fo_quorum", pct(FO.quorum_on_time), "{:.2f}", "quorum on time, %")
v("fo_even", FO.even_odds_workers, "{:.0f}", "first fleet below even odds")

print("== Fan-out tail latency (Literature.FanoutTailLatency)")
v("dn_one_p50", TAIL.FanoutOneLeafP50Ms, "{:.0f}", "one leaf, median, ms")
v("dn_one_p99", TAIL.FanoutOneLeafP99Ms, "{:.0f}", "one leaf, 99th, ms")
v("dn_all_p50", TAIL.FanoutAllLeavesP50Ms, "{:.0f}", "all leaves, median, ms")
v("dn_all_p99", TAIL.FanoutAllLeavesP99Ms, "{:.0f}", "all leaves, 99th, ms")
v("dn_ratio", TAIL.FanoutAllLeavesP99Ms / TAIL.FanoutOneLeafP99Ms, "{:.0f}", "99th, all over one, x")
v("dn_delay", TAIL.HedgeDelayMs, "{:.0f}", "hedge delay, ms")
v("dn_keys", TAIL.HedgeKeys, "{:,.0f}", "keys read")
v("dn_before", TAIL.HedgeBeforeP999Ms, "{:,.0f}", "99.9th without hedging, ms")
v("dn_after", TAIL.HedgeAfterP999Ms, "{:.0f}", "99.9th with hedging, ms")
v("dn_extra", TAIL.HedgeExtraRequestsPct, "{:.0f}", "added requests, %")

print("== Credit window (Scenarios.DiffReviewCreditWindow)")
v("cw_prod", CW.producer_rate.to("1/s").magnitude, "{:.0f}", "producer, items/s")
v("cw_tok", CW.tokens_per_item.to("token").magnitude, "{:.0f}", "tokens per critique")
v("cw_rate", CW.decode_rate.to("token/s").magnitude, "{:.0f}", "reviewer decode, tokens/s")
v("cw_mu", CW.service_rate.to("1/s").magnitude, "{:.2f}", "service rate, items/s")
v("cw_min", CW.horizon.to("minute").magnitude, "{:.0f}", "horizon, minutes")
v("cw_backlog", CW.backlog, "{:,.0f}", "backlog")
v("cw_wait", CW.backlog_wait.to("hour").magnitude, "{:.1f}", "newest wait, h")
v("cw_w", CW.credits, "{:.0f}", "window")
v("cw_res", s(CW.window_residence), "{:.0f}", "residence, s")
v("cw_wt", CW.window_for_target, "{:.0f}", "largest window for the target")

print("== Exercise 7(a) allowances (Scenarios.RemediationChildAllowances)")
_admitted, _left = AL.admission
v("al_rem", usd(AL.remaining), "{:.0f}", "remaining, $")
v("al_res", usd(AL.finalization_reserve), "{:.0f}", "finalization reserve, $")
v("al_own", usd(AL.own_reserved), "{:.0f}", "root's own work, $")
v("al_n_words", len(AL.requests), "words", "children")
v("al_ask", usd(AL.requests[0]), "{:.0f}", "each child asks, $")
v("al_avail", usd(AL.available), "{:.0f}", "available to children, $")
v("al_copied", usd(AL.copied_liability), "{:.0f}", "copied liability, $")
v("al_left", usd(_left), "{:.0f}", "left after the first admission, $")
v("al_equal", usd(AL.equal_allowance), "{:.0f}", "largest equal allowance, $")
v("al_naive", usd(AL.naive_allowance), "{:.2f}", "split ignoring reserve and own work, $")
v("al_nested", usd(AL.nested_request), "{:.0f}", "inspection grandchild, $")

print("== NM 18.7 matched cohort (Scenarios.MatchedDelegationCohort)")
_s, _d = CO.single, CO.delegated
v("co_n", CO.attempted, "{:.0f}", "migrations")
v("co_budget", usd(CO.budget), "{:.0f}", "ceiling, $")
v("co_cutoff", s(CO.cutoff), "{:.0f}", "final cutoff, s")
v("co_deadline", s(CO.deadline), "{:.0f}", "deadline, s")
v("co_floor", CO.on_time_floor, "{:.0f}", "on-time floor")
v("co_s_acc", _s.accepted, "{:.0f}", "single accepted")
v("co_m_acc", _d.accepted, "{:.0f}", "delegated accepted")
v("co_s_on", _s.on_time, "{:.0f}", "single on time")
v("co_m_on", _d.on_time, "{:.0f}", "delegated on time")
v("co_s_spend", usd(CO.spend(_s)), "{:.0f}", "single spend, $")
v("co_m_spend", usd(CO.spend(_d)), "{:.0f}", "delegated spend, $")
v("co_s_disc", usd(CO.discarded(_s)), "{:.0f}", "single failed and canceled, $")
v("co_m_disc", usd(CO.discarded(_d)), "{:.0f}", "delegated failed and canceled, $")
v("co_s_pa", usd(CO.cost_per_accepted(_s)), "{:.3f}", "single cost per accepted, $")
v("co_m_pa", usd(CO.cost_per_accepted(_d)), "{:.3f}", "delegated cost per accepted, $")
v("co_ratio", CO.cost_ratio, "{:.1f}", "cost per accepted ratio, x")
v("co_extra", usd(CO.extra_spend), "{:.0f}", "added spend, $")
v("co_extra_on", CO.extra_on_time, "{:.0f}", "added on-time acceptances")
v("co_marg", usd(CO.marginal_on_time_cost), "{:.2f}", "per added on-time acceptance, $")

print("== MAST (Literature.MultiAgentFailures)")
v("mast_traces", MAST.MastTraces, "{:,.0f}", "annotated traces")
v("mast_fw", MAST.MastFrameworks, "{:.0f}", "frameworks")
v("mast_fw_words", MAST.MastFrameworks, "words", "the same, in words")
v("mast_modes", MAST.MastFailureModes, "{:.0f}", "failure modes")
v("mast_shown", len(MAST_MODES), "{:.0f}", "modes drawn")
v("mast_shown_words", len(MAST_MODES), "words", "the same, in words")
v("mast_design", MAST.MastSystemDesignPct, "{:.1f}", "system design, %")
v("mast_align", MAST.MastInterAgentMisalignmentPct, "{:.1f}", "inter-agent misalignment, %")
v("mast_verify", MAST.MastTaskVerificationPct, "{:.1f}", "task verification, %")
v("mast_ver_gain", MAST.MastVerificationStepGainPct, "{:.1f}", "verification-step gain, %")

if __name__ == "__main__":
    write(__file__)
