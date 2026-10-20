#!/usr/bin/env python3
"""Every number on the Volume III Lecture 19 slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations, and the chapter reads the same
entries (solvers in engine/solvers/cost_accounting.py): the token-price hook
(RepairCohortTokenPriceComparison over FrontierModelRepairCohort and
LowTokenPriceRepairCohort), the price of a fixed capability
(PriceAtFixedCapabilityDecline), the cached 30-turn bill (RepairResendCacheBill
on CodingAgentResend30Turns), compaction payback (RepairCompactionPayback and
its half-cached-rate variant), retries (CheapRetryAgainstStrongCall), the
serial episode and speculative decoding (SerialRepairEpisode,
SerialRepairEpisodeSpeculation), long-horizon cost against score
(FrontierSweCostVsScore), the cascade and its false-accept harm
(CodeEditThreeTierCascade), admission against a quota
(RepairTrajectoryQuotaAdmission), Kingman's queue price
(BurstyModelServerQueuePrice) and tail exposure (ThirtyCallTailExposure),
owning capacity (DedicatedCapacityBreakEven), the conserved budget
(RepairBudgetLateBill), architectures with fallback
(DependencyUpgradeArchitectures), adaptation payback
(CodeReviewAdaptationPayback), and the four lighthouse workloads
(LighthouseWorkloadLevers). The recall slide reads Lecture 18's
ResearchSystemTokenBreakEven. Real data are Literature series read through
scenarios: SWE-bench Verified recorded run costs (SweAgentClaude35SonnetRunCost
and its variants, the $4 cap from Kapoor et al. 2024), Brown et al. (2024)
coverage (SweBenchLiteCoverageIndependence), and SemiAnalysis InferenceX
(InferenceXH200SglangConcurrency). This file only selects values, converts
units, and formats them; it types no numbers. Each value is recorded with v()
under the key the deck prints as \\val{key}; running this file writes
images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Scenarios = mlsysim.Scenarios

HOOK = Scenarios.RepairCohortTokenPriceComparison
FR, LP = HOOK.baseline, HOOK.alternative
PRICE = Scenarios.PriceAtFixedCapabilityDecline
BE = Scenarios.ResearchSystemTokenBreakEven
BILL = Scenarios.RepairResendCacheBill
TRAJ = BILL.trajectory
COMP = Scenarios.RepairCompactionPayback
COMP_H = Scenarios.RepairCompactionPaybackHalfCachedRate
RETRY = Scenarios.CheapRetryAgainstStrongCall
COVER = Scenarios.SweBenchLiteCoverageIndependence
EPI = Scenarios.SerialRepairEpisode
SPEC = Scenarios.SerialRepairEpisodeSpeculation
FSWE = Scenarios.FrontierSweCostVsScore
CAS = Scenarios.CodeEditThreeTierCascade
ADM = Scenarios.RepairTrajectoryQuotaAdmission
SERVE = Scenarios.InferenceXH200SglangConcurrency
QUEUE = Scenarios.BurstyModelServerQueuePrice
TAIL = Scenarios.ThirtyCallTailExposure
OWN = Scenarios.DedicatedCapacityBreakEven
CAP = Scenarios.SweAgentClaude35SonnetCostCap
BUD = Scenarios.RepairBudgetLateBill
ARCH = Scenarios.DependencyUpgradeArchitectures
ADAPT = Scenarios.CodeReviewAdaptationPayback
LH = Scenarios.LighthouseWorkloadLevers

# The SWE-bench Verified runs the cohort slide compares, with their slide labels.
SWE_RUNS = [
    (Scenarios.SweAgentClaude35SonnetRunCost, "2024 run A"),
    (Scenarios.SweAgentClaude4SonnetRunCost, "2025 run B"),
    (Scenarios.MiniSweAgentOpus46RunCost, "2026 run C"),
    (Scenarios.MiniSweAgentGpt52CodexRunCost, "2026 run D"),
]
# The three price series the trend slide draws: (benchmark, threshold model).
PRICE_SERIES = [("MMLU", "GPT-3"), ("MMLU", "GPT-4-0314"), ("GPQA Diamond", "GPT-4-0314")]
# The four lighthouse workloads in table order.
WORKLOADS = ["repair", "research", "remediation", "support"]


def pct(x):
    return x * 100


def usd(q):
    return q.to("USD").magnitude


def per_m(q):
    return q.to("USD / Mtoken").magnitude


def s(q):
    return q.to("s").magnitude


def pct_fmt(x):
    """Whole percent when the value is whole, else one decimal (the chapter's rule)."""
    return "{:.0f}" if abs(pct(x) - round(pct(x))) < 10 ** -9 else "{:.1f}"


print("== Hook and NM 19.1 (Scenarios.RepairCohortTokenPriceComparison)")
H = HOOK.result
HF, HL = H.baseline, H.alternative
v("hk_tasks", FR.tasks, "{:.0f}", "tickets")
v("hk_fr_in", per_m(FR.input_price), "{:.0f}", "frontier input $/M")
v("hk_fr_out", per_m(FR.output_price), "{:.0f}", "frontier output $/M")
v("hk_lp_in", per_m(LP.input_price), "{:.2f}", "low-price input $/M")
v("hk_lp_out", per_m(LP.output_price), "{:.2f}", "low-price output $/M")
v("hk_fr_turns", FR.turns, "{:.0f}", "frontier turns")
v("hk_lp_turns", LP.turns, "{:.0f}", "low-price turns")
v("hk_fr_tin", FR.input_tokens_per_turn, "{:,.0f}", "frontier input tokens per turn")
v("hk_fr_tout", FR.output_tokens_per_turn, "{:,.0f}", "frontier output tokens per turn")
v("hk_lp_tin", LP.input_tokens_per_turn, "{:,.0f}", "low-price input tokens per turn")
v("hk_lp_tout", LP.output_tokens_per_turn, "{:,.0f}", "low-price output tokens per turn")
v("hk_env", usd(FR.environment_per_turn), "{:.2f}", "environment per turn, $")
v("hk_assess", usd(FR.assessment_per_attempt), "{:.2f}", "assessment per attempt, $")
v("hk_fr_acc", FR.accepted, "{:.0f}", "frontier accepted")
v("hk_lp_acc", LP.accepted, "{:.0f}", "low-price accepted")
v("hk_fr_ceff", HF.cost_per_accepted, "{:.2f}", "frontier $ per accepted")
v("hk_lp_ceff", HL.cost_per_accepted, "{:.2f}", "low-price $ per accepted")
v("hk_ratio", H.cost_ratio, "{:.1f}", "cost ratio, x")
v("hk_token_cut", pct(H.model_cut), "{:.1f}", "token-charge cut, %")
v("hk_attempt_cut", pct(H.attempt_cut), "{:.1f}", "attempt-cost cut, %")
v("hk_acc_cut", pct(H.acceptance_cut), "{:.1f}", "acceptance drop, %")
v("hk_triage", usd(FR.triage_per_rejected), "{:.0f}", "triage per rejected attempt, $")
v("hk_fr_ceff_tri", HF.cost_per_accepted_with_triage, "{:.2f}", "frontier with triage, $")
v("hk_lp_ceff_tri", HL.cost_per_accepted_with_triage, "{:.2f}", "low-price with triage, $")
v("hk_ratio_tri", H.cost_ratio_with_triage, "{:.1f}", "ratio with triage, x")
v("hk_fr_model", HF.model_cost, "{:.3f}", "frontier model tokens per attempt, $")
v("hk_lp_model", HL.model_cost, "{:.3f}", "low-price model tokens per attempt, $")
v("hk_fr_envc", HF.environment_cost, "{:.3f}", "frontier environment per attempt, $")
v("hk_lp_envc", HL.environment_cost, "{:.3f}", "low-price environment per attempt, $")
v("hk_fr_attempt", HF.attempt_cost, "{:.3f}", "frontier attempt, $")
v("hk_lp_attempt", HL.attempt_cost, "{:.3f}", "low-price attempt, $")
v("hk_lp_rej_per_acc", HL.rejected_per_accepted, "{:.0f}", "low price: rejected per accepted")
v("hk_fr_rej_per_acc", HF.rejected_per_accepted, "{:.2f}", "frontier: rejected per accepted")

print("== Price of a fixed capability (Scenarios.PriceAtFixedCapabilityDecline)")
P = PRICE.result
v("pr_n", P.series_count, "{:.0f}", "threshold series")
v("pr_lo", round(P.rate_low), "{:.0f}", "slowest decline, x/yr")
v("pr_med", round(P.rate_median), "{:.0f}", "median decline, x/yr")
v("pr_hi", round(P.rate_high, -1), "{:.0f}", "fastest decline, x/yr")
for key, (bench, thr) in zip(("pr_gpt3", "pr_gpt4", "pr_gpqa"), PRICE_SERIES):
    v(key, P.series_rates[f"{bench} / {thr}"], "{:.0f}", f"{bench} at {thr}, x/yr")

print("== Recall (Scenarios.ResearchSystemTokenBreakEven)")
v("rc_multi", BE.multi_agent_tokens_vs_chat, "{:.0f}", "multi-agent tokens over chat, x")
v("rc_agent", BE.agent_tokens_vs_chat, "{:.0f}", "agent tokens over chat, x")
v("rc_m", BE.token_multiple, "{:.2f}", "token multiple, x")
v("rc_be_pct", pct(BE.max_single_acceptance), "{:.1f}", "break-even single acceptance, %")

print("== SWE-bench Verified recorded costs (RecordedRunCost scenarios)")
for i, (sc, label) in enumerate(SWE_RUNS):
    r = sc.result
    print(f"   {label}: n={r.tasks} resolved={r.resolved} per resolved ${r.cost_per_resolved:.2f} "
          f"mean resolved ${r.mean_resolved_cost:.2f} understatement {r.survivor_understatement:.1f}x")
A = SWE_RUNS[0][0].result
D = SWE_RUNS[-1][0].result
v("sw_a_tasks", A.tasks, "{:.0f}", "run A tasks")
v("sw_a_total", A.cost_per_attempt * A.tasks, "{:,.2f}", "run A total recorded cost, $")
v("sw_a_resolved", A.resolved, "{:.0f}", "run A resolved")
v("sw_a_per_res", A.cost_per_resolved, "{:.2f}", "run A cost per resolved task, $")
v("sw_a_mean_res", A.mean_resolved_cost, "{:.2f}", "run A mean resolved-run cost, $")
v("sw_a_mean_unres", A.mean_unresolved_cost, "{:.2f}", "run A mean unresolved-run cost, $")
v("sw_a_under", A.survivor_understatement, "{:.1f}", "run A understatement, x")
v("sw_d_under", D.survivor_understatement, "{:.1f}", "run D understatement, x")
v("sw_a_unres", A.tasks - A.resolved, "{:.0f}", "run A unresolved")
v("sw_a_rej_per_res", (A.tasks - A.resolved) / A.resolved, "{:.0f}", "run A unresolved per resolved, rounded")

print("== Multiplier (the hook's two acceptances)")
v("mu_lp_rej", HL.rejected_per_accepted, "{:.0f}", "failures per success at the low acceptance")
v("mu_fr_rej", HF.rejected_per_accepted, "{:.2f}", "failures per success at the high acceptance")

print("== Tariff and NM 19.2 (Scenarios.RepairResendCacheBill)")
B = BILL.result
v("tf_in", per_m(BILL.input_price), "{:.0f}", "fresh input $/M")
v("tf_cached", per_m(BILL.input_price) * BILL.cached_fraction, "{:.2f}", "cached input $/M")
v("tf_out", per_m(BILL.output_price), "{:.0f}", "output $/M")
v("tf_cached_pct", pct(BILL.cached_fraction), "{:.0f}", "cached share of the input price, %")
v("tf_half_pct", pct(COMP_H.bill.cached_fraction), "{:.0f}", "newer cached share, %")
v("rs_turns", TRAJ.turns, "{:.0f}", "turns")
v("rs_ratio", B.cache_ratio, "{:.1f}", "uncached over cached bill, x")
v("rs_reread_pct", pct(B.reread_share), "{:.1f}", "cached re-reads in the cached bill, %")
v("rs_miss_frac_pct", pct(BILL.miss_fraction), "{:.0f}", "prefix that misses, %")
v("rs_miss_pct", pct(B.miss_share), "{:.1f}", "bill rise from the miss, %")
v("rs_first", TRAJ.first_prompt, "{:,.0f}", "first prompt, tokens")
v("rs_append", TRAJ.appended, "{:,.0f}", "appended per turn, tokens")
v("rs_out", TRAJ.output_per_turn, "{:.0f}", "output per call, tokens")
v("rs_sent", B.sent_tokens, "{:,.0f}", "logical input, tokens")
v("rs_no", usd(B.total_uncached), "{:.3f}", "bill resent fresh, $")
v("rs_fresh", B.fresh_tokens, "{:,.0f}", "fresh tokens under reuse")
v("rs_re", usd(B.total_cached), "{:.3f}", "cached bill, $")
v("rs_miss", usd(B.miss_cost), "{:.3f}", "the miss adds, $")

print("== NM 19.3 compaction (Scenarios.RepairCompactionPayback and variant)")
C, CH = COMP.result, COMP_H.result
v("cp_summary", COMP.summary_tokens, "{:,.0f}", "summary output, tokens")
v("cp_to", COMP.compact_to, "{:,.0f}", "compacted context, tokens")
v("cp_upfront_c", pct(usd(C.upfront)), "{:.2f}", "upfront, cents")
v("cp_fewer", C.context_at_compaction - COMP.compact_to, "{:,.0f}", "fewer cached tokens per later call")
v("cp_save_c", pct(usd(C.per_call_saving)), "{:.3f}", "saving per later call, cents")
v("cp_n", C.breakeven_calls, "{:.0f}", "payback, calls")
v("cp_rest", C.remaining_calls, "{:.0f}", "calls that remain")
v("cp_net_c", -pct(C.net_by_calls[C.remaining_calls]), "{:.2f}", "saving with the calls that remain, cents")
v("cp_n_half", CH.breakeven_calls, "{:.0f}", "payback at the half rate, calls")
v("cp_net_half_c", pct(CH.net_by_calls[CH.remaining_calls]), "{:.2f}", "loss at the half rate, cents")

print("== NM 19.4 retries (Scenarios.CheapRetryAgainstStrongCall)")
R = RETRY.result
v("rt_p", RETRY.pass_probability, "{:.2f}", "cheap pass probability")
v("rt_c", usd(RETRY.attempt_cost), "{:.2f}", "cheap attempt, $")
v("rt_k", RETRY.max_attempts, "{:.0f}", "cap")
v("rt_k_words", RETRY.max_attempts, "words", "the same, in words")
v("rt_ps", RETRY.strong_pass_probability, "{:.2f}", "strong pass probability")
v("rt_cs", usd(RETRY.strong_attempt_cost), "{:.2f}", "strong attempt, $")
v("rt_fail", 1 - RETRY.pass_probability, "{:.1f}", "cheap failure probability")
v("rt_cov_pct", pct(R.coverage), "{:.1f}", "coverage, %")
v("rt_cov", R.coverage, "{:.3f}", "coverage")
v("rt_p1", RETRY.pass_probability, "{:.1f}", "cheap pass, one decimal")
v("rt_att", R.expected_attempts, "{:.3f}", "mean attempts")
v("rt_seq", usd(R.sequential_per_accepted), "{:.3f}", "sequential per accepted, $")
v("rt_seq_task", usd(R.sequential_cost), "{:.3f}", "sequential per submitted task, $")
v("rt_par", usd(R.batch_per_accepted), "{:.3f}", "batch per accepted, $")
v("rt_prem", R.batch_premium, "{:.1f}", "batch premium, x")
v("rt_strong", usd(R.strong_per_accepted), "{:.3f}", "strong per accepted, $")
v("rt_ps_pct", pct(RETRY.strong_pass_probability), "{:.0f}", "strong coverage, %")
CV = COVER.result
KS = sorted(CV.measured)
K_MID = KS[4]  # the fifth pool size the study reports
v("cv_tasks", COVER.rows[0]["n_problems"], "{:.0f}", "repair tasks")
v("cv_p1_pct", pct(CV.first), "{:.0f}", "one sample solves, %")
v("cv_k", K_MID, "{:.0f}", "pool size")
v("cv_meas_pct", pct(CV.measured[K_MID]), "{:.0f}", "measured coverage there, %")
v("cv_ind_pct", pct(CV.independent[K_MID]), "{:.0f}", "independence predicts, %")

print("== NM 19.5 critical path (Scenarios.SerialRepairEpisode, SerialRepairEpisodeSpeculation)")
E, SP = EPI.result, SPEC.result
v("ep_speedup", EPI.model_speedup, "{:.0f}", "model speedup, x")
v("ep_model", s(E.model_time), "{:.2f}", "model calls, s")
v("ep_total", s(E.total_time), "{:.2f}", "episode, s")
v("ep_cap", E.limiting_speedup, "{:.2f}", "ceiling 1/(1-f), x")
v("ep_f_pct", pct(E.model_fraction), "{:.1f}", "model share, %")
v("ep_tests", s(E.tests_time), "{:.2f}", "tests, s")
v("ep_life", s(E.lifecycle_time), "{:.2f}", "sandbox creation and export, s")
v("ep_runtime", s(E.runtime_time), "{:.2f}", "runtime setup, s")
v("ep_changed", s(E.changed_time), "{:.2f}", "with the faster model, s")
v("ep_saving", s(E.saving), "{:.2f}", "saving, s")
v("ep_sp", E.speedup, "{:.2f}", "episode speedup, x")
v("ep_inf", s(E.other_time), "{:.2f}", "instant model calls, s")
v("ep_half_tests", s(E.tests_half_saving), "{:.1f}", "halving the tests saves, s")
v("sp_gamma", SPEC.draft_length, "{:.0f}", "drafted tokens per round")
v("sp_r", SPEC.draft_cost_ratio, "{:.2f}", "draft cost ratio")
v("sp_be", SP.break_even_accepted, "{:.1f}", "break-even accepted per round")
v("sp_m_hi", SPEC.accepted_high, "{:.1f}", "accepted per round, aligned")
v("sp_m_lo", SPEC.accepted_low, "{:.1f}", "accepted per round, hard task")
v("sp_s_hi", SP.speedup_high, "{:.2f}", "decode speedup, aligned, x")
v("sp_s_lo", SP.speedup_low, "{:.2f}", "decode speedup, hard task, x")
v("sp_t_hi", s(SP.episode_time_high), "{:.2f}", "episode, aligned, s")
v("sp_t_lo", s(SP.episode_time_low), "{:.2f}", "episode, hard task, s")
v("sp_ep_hi", SP.episode_speedup_high, "{:.2f}", "episode speedup, aligned, x")

print("== FrontierSWE (Scenarios.FrontierSweCostVsScore)")
F = FSWE.result
v("fs_n", F.configurations, "{:.0f}", "configurations")
v("fs_lo", F.cost_low, "{:,.0f}", "cheapest trial, $")
v("fs_hi", F.cost_high, "{:,.0f}", "costliest trial, $")
v("fs_ratio", F.cost_ratio, "{:.0f}", "top over runner-up cost, x")
v("fs_gap", F.score_gap, "{:.0f}", "score gap, points")
v("fs_spread", pct(F.median_spread), "{:.0f}", "median best-to-worst spread, points")
v("fs_trials", FSWE.trials, "{:.0f}", "trials per task")
v("fs_trials_words", FSWE.trials, "words", "the same, in words")
v("fs_budget", FSWE.budget_hours, "{:.0f}", "hours per trial")
v("fs_h_lo", F.hours_low, "{:.0f}", "shortest trial, h")
v("fs_h_hi", F.hours_high, "{:.0f}", "longest trial, h")

print("== NM 19.6 cascade (Scenarios.CodeEditThreeTierCascade)")
Q = CAS.result
v("cs_tasks", CAS.tasks, "{:,.0f}", "tasks")
for j, name in enumerate(("one", "two", "three")):
    v(f"cs_acc_{name}", CAS.accepted_by_tier[j], "{:,.0f}", f"tier {j + 1} accepted")
    v(f"cs_reach_{name}", Q.reached[j], "{:,.0f}", f"tier {j + 1} reached")
    v(f"cs_cond_{name}", pct(Q.conditional_acceptance[j]), "{:.0f}", f"tier {j + 1} conditional acceptance, %")
    v(f"cs_call_{name}", usd(CAS.call_costs[j]), "{:.4f}", f"tier {j + 1} call, $")
v("cs_gate", usd(CAS.gate_cost), "{:.4f}", "gate, $")
v("cs_unres", Q.unresolved, "{:,.0f}", "left unresolved")
v("cs_spend", Q.spend, "{:,.0f}", "cascade spend, $")
v("cs_acc", Q.accepted, "{:,.0f}", "accepted")
v("cs_per", Q.per_accepted, "{:.5f}", "cascade per accepted, $")
v("cs_direct", Q.direct_spend, "{:,.0f}", "direct spend, $")
v("cs_direct_acc", CAS.direct_accepted, "{:,.0f}", "direct accepted")
v("cs_direct_per", Q.direct_per_accepted, "{:.5f}", "direct per accepted, $")
v("cs_ratio", Q.direct_ratio, "{:.1f}", "direct over cascade, x")
v("cs_all_t", Q.all_tiers_time, "{:.2f}", "every tier, s")
v("cs_direct_t", Q.direct_time, "{:.2f}", "direct, s")
v("cs_mean_t", Q.mean_time, "{:.2f}", "mean, s")
v("dg_fa_pct", pct(CAS.false_accept_rate), "{:.1f}", "false-accept rate, %")
v("dg_fa", CAS.false_accept_rate, "{:.3f}", "the same, as a fraction")
v("dg_harm_each", usd(CAS.harm_per_false_accept), "{:.0f}", "harm per false accept, $")
v("dg_harm", Q.harm_cost, "{:,.0f}", "harm, $")
v("dg_false", CAS.accepted_by_tier[0] * CAS.false_accept_rate, "{:.0f}", "false accepts")
v("dg_per", Q.per_accepted_with_harm, "{:.5f}", "cascade with harm per accepted, $")
v("dg_saving", Q.saving_per_accepted, "{:.3f}", "saving per accepted, $")

print("== NM 19.7 capacity (Scenarios.RepairTrajectoryQuotaAdmission)")
AD = ADM.result
v("ad_arrival", ADM.arrival.to("1/minute").magnitude, "{:.0f}", "admissions per minute")
v("ad_arrival_words", ADM.arrival.to("1/minute").magnitude, "words", "the same, in words")
v("ad_res", ADM.residence.to("minute").magnitude, "{:.0f}", "minutes present")
v("ad_conc", AD.concurrent, "{:.0f}", "trajectories present")
v("ad_calls", ADM.calls, "{:.0f}", "calls per trajectory")
v("ad_tok", ADM.tokens_per_call, "{:,.0f}", "tokens per call")
v("ad_share_pct", pct(AD.quota_share), "{:.0f}", "quota used, %")
v("ad_quota_m", ADM.quota.to("1/minute").magnitude / 10 ** 6, "{:.0f}", "quota, million per minute")
v("ad_drop_pct", pct(ADM.acceptance_drop), "{:.0f}", "acceptance drop, %")
v("ad_after_pct", pct(AD.quota_share_after_drop), "{:.0f}", "quota used after the drop, %")
v("ad_call_rate", AD.call_rate.to("1/minute").magnitude, "{:.0f}", "calls per minute")
v("ad_demand_m", AD.demand.to("1/minute").magnitude / 10 ** 6, "{:.1f}", "demand, million per minute")
v("ad_ceiling", AD.arrival_ceiling.to("1/minute").magnitude, "{:.2f}", "admission ceiling per minute")

print("== Serving measurement (Scenarios.InferenceXH200SglangConcurrency)")
X = SERVE.result
v("ix_lo", X.points[0][0], "{:.0f}", "lowest concurrency")
v("ix_hi", X.points[-1][0], "{:.0f}", "highest concurrency")
v("ix_mult", X.concurrency_multiple, "{:.0f}", "concurrency multiple, x")
v("ix_mult_words", X.concurrency_multiple, "words", "the same, in words")
v("ix_thr", X.throughput_multiple, "{:.1f}", "throughput multiple, x")
v("ix_tpot_lo", X.token_time_low.to("ms").magnitude, "{:.1f}", "time per token at low concurrency, ms")
v("ix_tpot_hi", X.token_time_high.to("ms").magnitude, "{:.1f}", "time per token at high concurrency, ms")

print("== NM 19.8 queue (Scenarios.BurstyModelServerQueuePrice, ThirtyCallTailExposure)")
W = QUEUE.result
v("q_lo_pct", pct(QUEUE.low_utilization), "{:.0f}", "low utilization, %")
v("q_hi_pct", pct(QUEUE.high_utilization), "{:.0f}", "high utilization, %")
v("q_service", s(QUEUE.service), "{:.0f}", "service time, s")
v("q_gain_pct", pct(W.throughput_gain), "{:.1f}", "throughput gain, %")
v("q_ratio", W.wait_ratio, "{:.1f}", "wait ratio, x")
v("q_ca2", QUEUE.arrival_scv, "{:.0f}", "c_a^2")
v("q_cs2", QUEUE.service_scv, "{:.0f}", "c_s^2")
v("q_v", W.variability, "{:.0f}", "burstiness term")
v("q_w_lo", s(W.wait_low), "{:.0f}", "wait at low utilization, s")
v("q_w_hi", s(W.wait_high), "{:.0f}", "wait at high utilization, s")
v("q_hold", QUEUE.holding_rate.to("USD/s").magnitude, "{:.3f}", "holding rate, $/s")
v("q_h_lo", W.holding_low, "{:.3f}", "holding per call, low, $")
v("q_h_hi", W.holding_high, "{:.3f}", "holding per call, high, $")
v("tl_calls", TAIL.calls, "{:.0f}", "serial calls")
v("tl_p", TAIL.tail_probability, "{:.2f}", "per-call tail probability")
v("tl_hit", TAIL.any_tail, "{:.2f}", "chance of at least one")

print("== NM 19.9 owning capacity (Scenarios.DedicatedCapacityBreakEven)")
O = OWN.result
v("ow_fixed", OWN.fixed_per_hour.to("USD/hour").magnitude, "{:.0f}", "fixed, $/h")
v("ow_rate", OWN.reference_rate.to("1/s").magnitude, "{:,.0f}", "reference output tokens/s")
v("ow_var", per_m(OWN.variable_price), "{:.0f}", "marginal $/M output")
v("ow_cached_pct", pct(OWN.cached_fraction), "{:.0f}", "input cached, %")
v("ow_api", O.api_price, "{:.1f}", "API $/M output with input")
v("ow_cross_pct", pct(O.crossing), "{:.1f}", "break-even load, %")
v("ow_load_pct", pct(OWN.load), "{:.0f}", "operating load, %")
v("ow_save", O.saving_hour, "{:.2f}", "owning saves, $/h")
v("ow_cross0_pct", pct(O.crossing_no_cache), "{:.1f}", "break-even with no caching, %")
v("ow_ratio", OWN.input_per_output, "{:.0f}", "input per output token")
v("ow_tok", O.delivered_hour_tokens, "{:,.0f}", "output tokens per hour at load")
v("ow_api_h", O.api_hour, "{:.2f}", "API, $/h")
v("ow_ded_h", O.dedicated_hour, "{:.2f}", "owned, $/h")

print("== Cost cap (Scenarios.SweAgentClaude35SonnetCostCap)")
K = CAP.result
v("cap_limit", K.cost_limit, "{:.0f}", "cap per task, $")
v("cap_tasks", K.tasks, "{:.0f}", "tasks")
v("cap_over", K.over_limit_runs, "{:.0f}", "runs above the cap")
v("cap_max", K.max_cost, "{:.2f}", "costliest run, $")

print("== NM 19.10 budget (Scenarios.RepairBudgetLateBill)")
G = BUD.result
U = G.units_per_dollar
v("bg_budget", G.budget / U, "{:.0f}", "root budget, $")
v("bg_plan", G.planning / U, "{:.3f}", "planning, $")
v("bg_trace", G.allocations[0] / U, "{:.2f}", "trace child grant, $")
v("bg_fuzz", G.allocations[1] / U, "{:.2f}", "fuzzing child grant, $")
v("bg_trace_set", G.settled_children[0] / U, "{:.2f}", "trace child settled, $")
v("bg_refund", G.refund_first / U, "{:.2f}", "trace child returns, $")
v("bg_fuzz_set", G.settled_children[1] / U, "{:.2f}", "fuzzing child settled at stop, $")
v("bg_pending", G.pending / U, "{:.0f}", "reserved for the remote job, $")
v("bg_apparent", G.child_apparent_free / U, "{:.2f}", "a release at local stop returns, $")
v("bg_free_stop", G.child_free_at_stop / U, "{:.2f}", "truly free at stop, $")
v("bg_late", G.late_charge / U, "{:.2f}", "late bill, $")
v("bg_fuzz_final", (G.settled_children[1] + G.late_charge) / U, "{:.2f}", "fuzzing child settled at close, $")
v("bg_free_settled", G.child_free_settled / U, "{:.2f}", "returned at settlement, $")

print("== NM 19.11 architectures (Scenarios.DependencyUpgradeArchitectures)")
AR = ARCH.result
RULES, CALL, LOOP = "Rule-based rewriter", "Single model call", ARCH.loop_name
v("ar_tickets", ARCH.tickets, "{:,.0f}", "upgrades a month")
v("ar_human", usd(ARCH.human_cost), "{:.0f}", "engineer fallback, $")
v("ar_hres_pct", pct(ARCH.human_resolution), "{:.0f}", "engineer resolves, %")
v("ar_loop_pct", pct(AR.loop_acceptance), "{:.2f}", "loop acceptance, %")
v("ar_turns", AR.loop_turns, "{:.3f}", "mean loop turns")
v("ar_cap_words", len(ARCH.turn_acceptance), "words", "loop turn cap, in words")
v("ar_rules_pct", pct(ARCH.single_acceptance[RULES]), "{:.0f}", "rewriter passes, %")
v("ar_call_pct", pct(ARCH.single_acceptance[CALL]), "{:.0f}", "single call passes, %")
for j, name in enumerate(("one", "two", "three")):
    v(f"ar_r_{name}_pct", pct(ARCH.turn_acceptance[j]), "{:.0f}", f"loop turn {j + 1} passes, %")
v("ar_rules_auto", usd(AR.automated_per_accepted[RULES]), "{:.5f}", "rewriter automated per acceptance, $")
v("ar_rules_res", usd(AR.per_resolved[RULES]), "{:.2f}", "rewriter per resolved, $")
v("ar_floor_pct", pct(ARCH.resolution_floor), "{:.0f}", "service floor, %")
v("ar_call_res", usd(AR.per_resolved[CALL]), "{:.2f}", "single call per resolved, $")
v("ar_loop_res", usd(AR.per_resolved[LOOP]), "{:.2f}", "loop per resolved, $")
v("ar_point", usd(AR.point_value), "{:.2f}", "fallback saved per point, $")
N = ADAPT.result
v("ad_invest", usd(N.investment), "{:,.0f}", "adaptation investment, $")
v("ad_tasks", N.breakeven_whole, "{:,.0f}", "tasks to repay it")

print("== Lighthouse 19.1 (Scenarios.LighthouseWorkloadLevers)")
L = LH.result.rows
for w in WORKLOADS:
    x = L[w]
    v(f"lh_{w}_attempt", x.attempt, "{:.2f}", f"{w}: attempt, $")
    v(f"lh_{w}_p", x.acceptance, "{:.2f}", f"{w}: acceptance")
    v(f"lh_{w}_ceff", x.cost_per_accepted, "{:.2f}", f"{w}: C_eff, $")
    v(f"lh_{w}_tok", pct(x.halve_tokens), pct_fmt(x.halve_tokens), f"{w}: halve token price, %")
    v(f"lh_{w}_big", pct(x.halve_largest), pct_fmt(x.halve_largest), f"{w}: halve largest term, %")
    v(f"lh_{w}_acc", pct(x.acceptance_gain), pct_fmt(x.acceptance_gain), f"{w}: ten more points, %")
v("lh_repair_share", pct(L["repair"].model_share), "{:.1f}", "repair: model share, %")
v("lh_research_share", pct(L["research"].model_share), "{:.1f}", "research: model share, %")
v("lh_repair_turns", LH.workloads["repair"]["turns"], "{:.0f}", "repair turns")
v("lh_handoff", LH.workloads["support"]["handoff"], "{:.0f}", "support handoff, $")
v("lh_gain_pts", pct(LH.acceptance_gain), "{:.0f}", "acceptance gain, points")
v("lh_gain_words", pct(LH.acceptance_gain), "words", "the same, in words")
v("lh_threshold_pct", pct(L["repair"].token_threshold), "{:.0f}", "repair threshold, %")

if __name__ == "__main__":
    write(__file__)
