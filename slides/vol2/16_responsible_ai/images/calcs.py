#!/usr/bin/env python3
"""Every number on the Lecture 16 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (Literature, ReferenceStats, the hardware and model
registries) and the calculations (Scenarios.*, each a scenario type); the
chapter, Responsible AI, reads the same scenarios. This file only selects
values, converts units, and formats them; it types no numbers. Each value is
recorded with v() under the key the deck prints as \\val{key}; running this
file writes images/numbers.tex.

Sources by slide:
  - Hook, flows, close: Scenarios.OfqualStandardization2020 over
    Literature.Series.OfqualGradeMatrix2020 and OfqualCandidatesLowered2020;
    report-text shares and dates in Literature.Fairness (Ofqual 2021).
  - Loan model and z-test: Scenarios.LoanScreeningTwoGroups and
    Scenarios.LoanApprovalGapTest (Napkin Math 16.1, 16.2; chapter 16.5.2).
  - COMPAS: Scenarios.CompasTwoYearRecidivism over
    Literature.Series.CompasTwoYearConfusion (ProPublica data, recomputed).
  - Dual thresholds: Scenarios.GroupThresholdSketch; fairness cost:
    Scenarios.CreditParityAccuracyCost (Napkin Math 16.4).
  - Obermeyer, Apple Card, four-fifths rule: Literature.Fairness.
  - Gender Shades: Literature.Series.GenderShadesAudit.
  - Monitoring: Scenarios.FairnessMonitor10K (chapter 16.5.5).
  - Overheads: Literature.ResponsibleAIOverhead (chapter table 5).
  - SHAP: Scenarios.ExactShapEnumeration, Scenarios.InlineShapRetrofit,
    Scenarios.ImageExplanationLog.
  - DP-SGD: Literature.Series.AbadiDpSgdAccuracy and
    Literature.DifferentialPrivacy; the cost scenario Scenarios.Gpt3PrivacyPlan.
  - Unlearning: Scenarios.Gpt3ShardedUnlearning (Napkin Math 16.6).
  - RLHF labels: Scenarios.RewardModelLabelBudget.
  - Automation bias: Scenarios.RadiologyAutomationBias (Napkin Math 16.7).
  - Waymo: Literature.Series.WaymoRiderOnlyMiles and WaymoVsHumanBenchmark;
    fleet failures: Scenarios.SafetyComplianceAtBillionRequests.
  - Tempe recall: Literature.IncidentReports.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Literature, Scenarios = mlsysim.Literature, mlsysim.Scenarios
FAIR = Literature.Fairness
SERIES = Literature.Series

OFQ = Scenarios.OfqualStandardization2020
LOAN = Scenarios.LoanScreeningTwoGroups
ZT = Scenarios.LoanApprovalGapTest
COMPAS = Scenarios.CompasTwoYearRecidivism
DUAL = Scenarios.GroupThresholdSketch
TAX = Scenarios.CreditParityAccuracyCost
MON = Scenarios.FairnessMonitor10K
SHAP = Scenarios.ExactShapEnumeration
RETRO = Scenarios.InlineShapRetrofit
XLOG = Scenarios.ImageExplanationLog
DPPLAN = Scenarios.Gpt3PrivacyPlan
SISA = Scenarios.Gpt3ShardedUnlearning
RLHF = Scenarios.RewardModelLabelBudget
AB = Scenarios.RadiologyAutomationBias
FLEET = Scenarios.SafetyComplianceAtBillionRequests
RAI = Literature.ResponsibleAIOverhead
DP = Literature.DifferentialPrivacy

# Figure inputs (make_figs.py imports these)
OFQ_GRADES = OFQ.grades
OFQ_SPLIT = [OFQ.split(g) for g in OFQ_GRADES]
OFQ_T2 = [OFQ.candidates_with(k) for k in range(len(OFQ.candidates.entries))]
BLACK, WHITE = COMPAS.first, COMPAS.second
GS = SERIES.GenderShadesAudit.entries
DPSGD = SERIES.AbadiDpSgdAccuracy.entries
MILES = SERIES.WaymoRiderOnlyMiles.entries
RATES = SERIES.WaymoVsHumanBenchmark.entries


def pct(x):
    return 100 * x


def f(x):
    """A Literature scalar as a float."""
    return float(x)


def gs_err(company, subgroup, when):
    return next(r[f"error_{when}_pct"] for r in GS if r["company"] == company and r["subgroup"] == subgroup)


def bench(name):
    return next(r for r in RATES if r["benchmark"] == name)


def z_for(n):
    """z of the loan gap at n applicants per group (figure curve)."""
    return ZT.z_at(n)


print("== Hook and close: Ofqual 2020 (Grading gaps in summer 2020)")
v("ofq_lower_pct", f(FAIR.OfqualEntriesLowerPct), "{:.0f}", "entries lowered, report text, %")
v("ofq_same_pct", f(FAIR.OfqualEntriesSamePct), "{:.0f}", "entries unchanged, report text, %")
v("ofq_higher_pct", f(FAIR.OfqualEntriesHigherPct), "{:.0f}", "entries raised, report text (just over), %")
v("ofq_interim_lower_pct", f(FAIR.OfqualInterimEntriesLowerPct), "{:.1f}", "interim report, entries lowered, %")
v("ofq_entries", OFQ.stated_entries, "{:,.0f}", "A level entries (stated n)")
v("ofq_cands", OFQ.total_candidates, "{:,.0f}", "candidates (stated n)")
v("ofq_zero", OFQ.candidates_with(0), "{:,.0f}", "candidates with no grade lowered")
v("ofq_zero_pct", pct(OFQ.candidate_share(0)), "{:.0f}", "the same, %")
v("ofq_any_pct", pct(OFQ.at_least_one_lowered_share), "{:.0f}", "candidates with at least one lowered, %")
v("ofq_any_pct_1", pct(OFQ.at_least_one_lowered_share), "{:.1f}", "the same, one decimal")
v("ofq_table_lower_pct", OFQ.entries_lowered_pct, "{:.1f}", "entries lowered, weighted from Table 1, %")
one_lower = [OFQ.split(g)[2] for g in OFQ_GRADES[:-1]]   # A* to E: U cannot go lower
v("ofq_one_lower_min", min(one_lower), "{:.0f}", "smallest one-grade-lower share, CAG A* to E, %")
v("ofq_one_lower_max", max(one_lower), "{:.0f}", "largest one-grade-lower share, CAG A* to E, %")
v("ofq_results_day", f(FAIR.OfqualResultsDayAugust2020), "{:.0f}", "results day, August")
v("ofq_reversal_day", f(FAIR.OfqualReversalDayAugust2020), "{:.0f}", "reversal day, August")
v("ofq_days_word", f(FAIR.OfqualReversalDayAugust2020) - f(FAIR.OfqualResultsDayAugust2020), "words",
  "days from results to reversal")

print("== Loan model (Napkin Math 16.1, 16.2)")
for g in ("A", "B"):
    k = g.lower()
    c = LOAN.groups[g]
    v(f"loan_{k}_approved", LOAN.positives(g), "{:.0f}", f"group {g} approved")
    v(f"loan_{k}_repaid", c["tp"], "{:.0f}", f"group {g} approved and repaid")
    v(f"loan_{k}_fn", c["fn"], "{:.0f}", f"group {g} rejected who would repay")
    v(f"loan_{k}_fp", c["fp"], "{:.0f}", f"group {g} approved who default")
    v(f"loan_{k}_repayers", c["tp"] + c["fn"], "{:.0f}", f"group {g} who would repay")
    v(f"loan_{k}_defaulters", c["fp"] + c["tn"], "{:.0f}", f"group {g} who would default")
    v(f"loan_{k}_approval", LOAN.positive_rate(g), "{:.2f}", f"group {g} approval rate")
    v(f"loan_{k}_tpr", LOAN.tpr(g), "{:.3f}", f"group {g} TPR")
    v(f"loan_{k}_fpr", LOAN.fpr(g), "{:.3f}", f"group {g} FPR")
    v(f"loan_{k}_ppv", LOAN.ppv(g), "{:.3f}", f"group {g} PPV")
v("loan_n", LOAN.size("A"), "{:.0f}", "applicants per group")
v("loan_parity_gap", LOAN.gap("positive_rate"), "{:.2f}", "parity gap")
v("loan_tpr_gap", LOAN.gap("tpr"), "{:.3f}", "TPR gap")
v("loan_di", LOAN.disparate_impact, "{:.2f}", "disparate impact ratio")
v("four_fifths", f(FAIR.FourFifthsRuleRatio), "{:.1f}", "four-fifths screen")

print("== COMPAS, recomputed from ProPublica's data")
for g, k in ((BLACK, "black"), (WHITE, "white")):
    c = COMPAS.groups[g]
    v(f"compas_{k}_fpr", pct(COMPAS.fpr(g)), "{:.1f}", f"{g} FPR, %")
    v(f"compas_{k}_ppv", pct(COMPAS.ppv(g)), "{:.1f}", f"{g} PPV, %")
    v(f"compas_{k}_base", pct(COMPAS.base_rate(g)), "{:.1f}", f"{g} base rate, %")
    for q in ("tp", "fp", "fn", "tn"):
        v(f"compas_{k}_{q}", c[q], "{:.0f}", f"{g} {q.upper()}")

print("== Dual thresholds (illustrative) and the fairness cost (Napkin Math 16.4)")
t_one, t_b = DUAL.uniform_threshold, DUAL.group_thresholds["B"]
v("dual_t", t_one, "{:.2f}", "single cutoff")
v("dual_tb", t_b, "{:.2f}", "Group B cutoff")
v("dual_tpr_a", pct(DUAL.tpr("A", t_one)), "{:.0f}", "TPR A at the single cutoff, %")
v("dual_tpr_b", pct(DUAL.tpr("B", t_one)), "{:.0f}", "TPR B at the single cutoff, %")
v("dual_tpr_b2", pct(DUAL.tpr("B", t_b)), "{:.0f}", "TPR B at its own cutoff, %")
v("dual_fpr_b", pct(DUAL.fpr("B", t_one)), "{:.0f}", "FPR B at the single cutoff, %")
v("dual_fpr_b2", pct(DUAL.fpr("B", t_b)), "{:.0f}", "FPR B at its own cutoff, %")
v("dual_gap_word", DUAL.tpr("A", t_one) / DUAL.tpr("B", t_one), "words", "opportunity gap, times")
same = sorted(set(DUAL.scores["A"]["qualified"]) & set(DUAL.scores["B"]["qualified"]))[0]
v("dual_same_score", same, "{:.2f}", "a score both groups hold")
v("tax_base", TAX.baseline_accuracy, "{:.0f}", "accuracy before, %")
v("tax_after", TAX.constrained_accuracy, "{:.0f}", "accuracy under parity, %")
v("tax_pts", TAX.accuracy_cost, "{:.0f}", "accuracy points")
v("tax_rel", pct(TAX.relative_cost), "{:.1f}", "relative drop, %")
v("tax_a_appr", TAX.context["group_a_approval"], "{:.0f}", "Group A approval, %")
v("tax_b_appr", TAX.context["group_b_approval_unconstrained"], "{:.0f}", "Group B approval before, %")

print("== Obermeyer et al. 2019")
ob0, ob1 = f(FAIR.ObermeyerBlackShareBeforePct), f(FAIR.ObermeyerBlackShareAfterPct)
v("ober_before", ob0, "{:.1f}", "Black share auto-identified, cost label, %")
v("ober_after", ob1, "{:.1f}", "after the simulated fix, %")
v("ober_ratio", ob1 / ob0, "{:.1f}", "ratio")

print("== Gender Shades (Buolamwini and Gebru 2018; Raji and Buolamwini 2019)")
IBM_ACC = {}
for sg in ("lighter male", "lighter female", "darker male", "darker female"):
    IBM_ACC[sg] = 100 - gs_err("IBM", sg, "may2017")
v("gs_ibm_lm_acc", IBM_ACC["lighter male"], "{:.1f}", "IBM accuracy, lighter men, %")
v("gs_ibm_df_acc", IBM_ACC["darker female"], "{:.1f}", "IBM accuracy, darker women, %")
for co, k in (("IBM", "ibm"), ("Face++", "facepp"), ("Microsoft", "msft")):
    v(f"gs_{k}_17", gs_err(co, "darker female", "may2017"), "{:.1f}", f"{co} darker-female error 2017, %")
    v(f"gs_{k}_18", gs_err(co, "darker female", "aug2018"), "{:.1f}", f"{co} darker-female error 2018, %")
v("gs_amazon_18", gs_err("Amazon", "darker female", "aug2018"), "{:.1f}", "Amazon 2018, %")
v("gs_kairos_18", gs_err("Kairos", "darker female", "aug2018"), "{:.1f}", "Kairos 2018, %")

print("== Two-proportion z-test (chapter 16.5.2)")
lo, hi = ZT.interval
slo, shi = ZT.interval_at(ZT.small_n_per_group)
v("z_rate_a", ZT.rate_a, "{:.2f}", "approval rate A")
v("z_rate_b", ZT.rate_b, "{:.2f}", "approval rate B")
v("z_pooled", ZT.pooled_rate, "{:.2f}", "pooled rate")
v("z_n", ZT.n_per_group, "{:.0f}", "applicants per group")
v("z_n_total", ZT.n_per_group + ZT.n_per_group, "{:.0f}", "applicants in all")
v("z_small_n", ZT.small_n_per_group, "{:.0f}", "small sample per group")
v("z_small_total", ZT.small_n_per_group + ZT.small_n_per_group, "{:.0f}", "small sample in all")
v("z_big", ZT.z, "{:.2f}", "z at 100 per group")
v("z_small", ZT.z_at(ZT.small_n_per_group), "{:.2f}", "z at 10 per group")
v("z_crit", ZT.z_critical, "{:.2f}", "two-sided 5 percent critical value")
v("z_nstar", ZT.n_at_critical, "{:.0f}", "per-group n where z reaches the critical value")
v("z_gap_pts", pct(ZT.gap), "{:.0f}", "gap, points")
v("z_ci_lo", lo, "{:.2f}", "95% interval low, 100 per group")
v("z_ci_hi", hi, "{:.2f}", "95% interval high, 100 per group")
v("z_ci_lo_small", -slo, "{:.2f}", "95% interval low, 10 per group (magnitude; printed with a minus)")
v("z_ci_hi_small", shi, "{:.2f}", "95% interval high, 10 per group")

print("== Monitoring (chapter 16.5.5)")
v("mon_qps", MON.request_rate.to("1/s").magnitude, "{:,.0f}", "decisions per second")
v("mon_sample_pct", pct(MON.sample_share), "{:.0f}", "sampled, %")
v("mon_rec_b", MON.record_size.to("byte").magnitude, "{:.0f}", "bytes per record")
v("mon_groups", MON.groups, "{:.0f}", "groups")
v("mon_mpg", MON.metrics_per_group, "{:.0f}", "metrics per group")
v("mon_far_pct", pct(MON.false_alarm_rate), "{:.0f}", "false-alarm rate, %")
v("mon_far", MON.false_alarm_rate, "{:.2f}", "false-alarm rate")
v("mon_day_m", MON.decisions_per_day / 10 ** 6, "{:.0f}", "decisions per day, millions")
v("mon_rec_m", MON.records_per_day / 10 ** 6, "{:.2f}", "records per day, millions")
v("mon_gb", MON.log_per_day.to("GB").magnitude, "{:.2f}", "log per day, GB")
v("mon_metrics", MON.metrics, "{:.0f}", "metrics")
v("mon_alerts", MON.false_alerts_per_interval, "{:.1f}", "expected false alerts per interval")

print("== Safeguard overheads (chapter table 5)")
OVERHEAD = [
    ("Fairness-aware training", f(RAI.FairnessTrainingLowPct), f(RAI.FairnessTrainingHighPct), "train"),
    ("DP-SGD", f(RAI.DpTrainingLowPct), f(RAI.DpTrainingHighPct), "train"),
    ("Federated learning", f(RAI.FederatedTrainingLowPct), f(RAI.FederatedTrainingHighPct), "train"),
    ("Adversarial training", f(RAI.AdversarialTrainingLowPct), f(RAI.AdversarialTrainingHighPct), "train"),
    ("Sampled KernelSHAP", f(RAI.ApproxShapInferenceLowPct), f(RAI.ApproxShapInferenceHighPct), "serve"),
    ("Exhaustive SHAP", pct(f(RAI.ExactShapInferenceLowX)), pct(f(RAI.ExactShapInferenceHighX)), "serve"),
]
v("oh_fair_lo", f(RAI.FairnessTrainingLowPct), "{:.0f}", "fairness-aware training low, %")
v("oh_fair_hi", f(RAI.FairnessTrainingHighPct), "{:.0f}", "high, %")
v("oh_dp_lo", f(RAI.DpTrainingLowPct), "{:.0f}", "DP-SGD low, %")
v("oh_dp_hi", f(RAI.DpTrainingHighPct), "{:.0f}", "high, %")
v("oh_fed_lo", f(RAI.FederatedTrainingLowPct), "{:.0f}", "federated low, %")
v("oh_fed_hi", f(RAI.FederatedTrainingHighPct), "{:.0f}", "high, %")
v("oh_adv_lo", 1 + f(RAI.AdversarialTrainingLowPct) / 100, "{:.0f}", "adversarial training low, times")
v("oh_adv_hi", 1 + f(RAI.AdversarialTrainingHighPct) / 100, "{:.0f}", "high, times")
v("oh_shap_lo", f(RAI.ApproxShapInferenceLowPct), "{:.0f}", "sampled KernelSHAP low, %")
v("oh_shap_hi", f(RAI.ApproxShapInferenceHighPct), "{:.0f}", "high, %")
v("oh_exact_lo", f(RAI.ExactShapInferenceLowX), "{:.0f}", "exhaustive SHAP low, times")
v("oh_exact_hi", f(RAI.ExactShapInferenceHighX), "{:,.0f}", "high, times")
v("oh_exact_lo_pct", OVERHEAD[-1][1], "{:,.0f}", "exhaustive SHAP low as drawn, %")
v("oh_exact_hi_pct", OVERHEAD[-1][2], "{:,.0f}", "high as drawn, %")

print("== Exact SHAP and the inline retrofit (chapter 16.7.2, 16.11)")
n_small, n_large = SHAP.feature_counts[0], SHAP.feature_counts[-1]
v("shap_n_small", n_small, "{:.0f}", "features, small")
v("shap_n_large", n_large, "{:.0f}", "features, large")
v("shap_sub_small", SHAP.subsets(n_small), "{:,.0f}", "subsets, small")
v("shap_sub_large", SHAP.subsets(n_large), "{:,.0f}", "subsets, large")
v("shap_t_large_s", SHAP.time(n_large).to("s").magnitude, "{:.2f}", "time, large, s")
v("shap_us", SHAP.time_per_subset.to("us").magnitude, "{:.0f}", "time per subset, us")
v("shap_us_word", SHAP.time_per_subset.to("us").magnitude, "words", "the same, in words")
v("shap_budget_ms", SHAP.request_budget.to("ms").magnitude, "{:.0f}", "request budget, ms")
v("retro_base", RETRO.base_latency.to("ms").magnitude, "{:.0f}", "base latency, ms")
v("retro_add_lo", RETRO.added_low.to("ms").magnitude, "{:.0f}", "added low, ms")
v("retro_add_hi", RETRO.added_high.to("ms").magnitude, "{:.0f}", "added high, ms")
v("retro_lo", RETRO.latency_low.to("ms").magnitude, "{:.0f}", "latency low, ms")
v("retro_hi", RETRO.latency_high.to("ms").magnitude, "{:.0f}", "latency high, ms")
v("retro_cap_lo", RETRO.capacity_low, "{:.1f}", "capacity low, times")
v("retro_cap_hi", RETRO.capacity_high, "{:.0f}", "capacity high, times")
v("xlog_mb_lo", XLOG.map_size_low.to("MB").magnitude, "{:.0f}", "map size low, MB")
v("xlog_mb_hi", XLOG.map_size_high.to("MB").magnitude, "{:.0f}", "map size high, MB")
v("xlog_tb_lo", XLOG.monthly_low.to("TB").magnitude, "{:.0f}", "TB per month low")
v("xlog_tb_hi", XLOG.monthly_high.to("TB").magnitude, "{:.0f}", "TB per month high")

print("== DP-SGD accuracy (Abadi et al. 2016)")
MNIST = [r for r in DPSGD if r["dataset"] == "MNIST"]
CIFAR = [r for r in DPSGD if r["dataset"] == "CIFAR-10"]
mnist0, cifar0 = f(DP.MnistNonprivateAccuracyPct), f(DP.AbadiCifar10NonprivateAccuracyPct)
v("dp_mnist0", mnist0, "{:.1f}", "MNIST nonprivate, %")
v("dp_cifar0", cifar0, "{:.0f}", "CIFAR-10 nonprivate, about, %")
for i, r in enumerate(MNIST):
    v(f"dp_mnist_eps{i}", r["epsilon"], "{:g}", "MNIST epsilon")
    v(f"dp_mnist_acc{i}", r["accuracy_pct"], "{:.0f}", "MNIST accuracy, %")
    v(f"dp_mnist_loss{i}", mnist0 - r["accuracy_pct"], "{:.1f}", "MNIST loss, points")
for i, r in enumerate(CIFAR):
    v(f"dp_cifar_eps{i}", r["epsilon"], "{:g}", "CIFAR-10 epsilon")
    v(f"dp_cifar_acc{i}", r["accuracy_pct"], "{:.0f}", "CIFAR-10 accuracy, %")
v("dp_delta_exp", -round(math.log10(MNIST[0]["delta"])), "{:.0f}", "delta, 10 to the minus this")
v("dpplan_eps", DPPLAN.epsilon_strong, "{:.0f}", "strong budget")
v("dpplan_base_m", DPPLAN.base_cost.to("USD").magnitude / 10 ** 6, "{:.1f}", "base cost, $M")
v("dpplan_strong_m", DPPLAN.cost_strong.to("USD").magnitude / 10 ** 6, "{:.1f}", "cost at the strong budget, $M")

print("== Unlearning (Napkin Math 16.6)")
v("sisa_params_b", SISA.workload.parameters.to("count").magnitude / 10 ** 9, "{:.0f}", "model parameters, billions")
v("sisa_gpus", SISA.n_accelerators, "{:,.0f}", "accelerators")
v("sisa_mfu_pct", pct(SISA.mfu), "{:.0f}", "MFU, %")
v("sisa_days", SISA.planning_days, "{:.0f}", "full retrain, days")
v("sisa_full_m", SISA.full_cost.to("USD").magnitude / 10 ** 6, "{:.1f}", "full retrain cost, $M")
v("sisa_shards", SISA.shards, "{:.0f}", "shards")
v("sisa_req", SISA.requests_per_day, "{:,.0f}", "requests per day")
v("sisa_shard_cost", SISA.shard_cost.to("USD").magnitude, "{:,.0f}", "one-shard cost, $")
v("sisa_shard_h", SISA.shard_time.to("hour").magnitude, "{:.2f}", "one-shard time, h")
v("sisa_shard_h1", SISA.shard_time.to("hour").magnitude, "{:.1f}", "the same, one decimal")
v("sisa_keep", 1 - SISA.shard_share, "{:.2f}", "per-request chance a shard is untouched")
p_un = SISA.p_shard_untouched
exp10 = math.floor(math.log10(p_un))
v("sisa_p_mant", p_un / 10 ** exp10, "{:.1f}", "P(untouched) mantissa")
v("sisa_p_exp", -exp10, "{:.0f}", "P(untouched), 10 to the minus this")
v("sisa_touched", SISA.shards_touched_per_day, "{:.0f}", "shards touched per day (about)")
v("sisa_batch_m", SISA.daily_batch_cost.to("USD").magnitude / 10 ** 6, "{:.1f}", "daily batch cost, $M")
v("sisa_drop_lo", pct(SISA.accuracy_drop_low), "{:.0f}", "accuracy drop low, %")
v("sisa_drop_hi", pct(SISA.accuracy_drop_high), "{:.0f}", "accuracy drop high, %")

print("== RLHF labels (chapter 16.8.2)")
v("rlhf_n_lo", RLHF.comparisons_low, "{:,.0f}", "comparisons low")
v("rlhf_n_hi", RLHF.comparisons_high, "{:,.0f}", "comparisons high")
v("rlhf_p_lo", RLHF.price_low.to("USD").magnitude, "{:.2f}", "price low, $")
v("rlhf_p_hi", RLHF.price_high.to("USD").magnitude, "{:.0f}", "price high, $")
v("rlhf_b_lo", RLHF.budget_low.to("USD").magnitude, "{:,.0f}", "budget low, $")
v("rlhf_b_hi_m", RLHF.budget_high.to("USD").magnitude / 10 ** 6, "{:.1f}", "budget high, $M")
v("rlhf_spread", RLHF.spread, "{:.0f}", "spread, times")

print("== Automation bias (Napkin Math 16.7)")
v("ab_human", pct(AB.human_sensitivity), "{:.0f}", "radiologist alone, %")
v("ab_ai", pct(AB.ai_sensitivity), "{:.0f}", "AI alone, %")
v("ab_accept_pct", pct(AB.miss_acceptance), "{:.0f}", "acceptance of AI misses, %")
v("ab_accept", AB.miss_acceptance, "{:.1f}", "acceptance, as a fraction")
v("ab_miss", AB.ai_miss_rate, "{:.2f}", "AI miss rate")
v("ab_miss_pct", pct(AB.ai_miss_rate), "{:.0f}", "AI miss rate, %")
v("ab_accepted", AB.accepted_misses(), "{:.3f}", "accepted misses")
v("ab_accepted_pct", pct(AB.accepted_misses()), "{:.1f}", "accepted misses, %")
v("ab_team", pct(AB.team_sensitivity_bound), "{:.1f}", "team sensitivity at most, %")
v("ab_gain", pct(AB.gain_over_ai), "{:.1f}", "gain over the AI alone, points")
v("ab_lo_pct", pct(AB.acceptance_low), "{:.0f}", "acceptance range low, %")
v("ab_hi_pct", pct(AB.acceptance_high), "{:.0f}", "acceptance range high, %")
v("ab_miss_lo", pct(AB.accepted_misses(AB.acceptance_low)), "{:.1f}", "accepted misses at the low, %")
v("ab_miss_hi", pct(AB.accepted_misses(AB.acceptance_high)), "{:.1f}", "accepted misses at the high, %")

print("== Waymo (Safety Impact Data Hub through June 2026)")
m0, m1 = MILES[0]["cumulative_ro_miles_millions"], MILES[-1]["cumulative_ro_miles_millions"]
inj, fatal = bench("Any-injury-reported"), bench("Any Fatal")
v("waymo_m0", m0, "{:.0f}", "rider-only miles, first release, millions")
v("waymo_m1", m1, "{:.0f}", "rider-only miles, last release, millions")
v("waymo_growth", m1 / m0, "{:.1f}", "growth, times")
v("waymo_growth0", m1 / m0, "{:.0f}", "growth, times, whole")
v("waymo_inj", inj["waymo_ipmm"], "{:.2f}", "injury crashes per million miles")
v("waymo_bench", inj["bench_ipmm"], "{:.2f}", "human benchmark, per million miles")
v("waymo_inj_n", inj["waymo_ipmm"] * m1, "{:.0f}", "rate times miles, injury crashes")
v("waymo_fatal_n", fatal["waymo_count"], "{:.0f}", "fatal crashes")
v("fleet_fail_pct", pct(FLEET.failure_probability), "{:.1f}", "unsafe responses, %")
v("fleet_fail_day", FLEET.daily_failures, "{:,.0f}", "unsafe responses per day")

print("== Apple Card and the Tempe recall")
v("apple_ratio", f(FAIR.AppleCardReportedLimitRatio), "{:.0f}", "reported credit-limit ratio")
v("tempe_detect_s", f(Literature.IncidentReports.TempeFirstDetectionS), "{:.1f}", "first detection before impact, s")

if __name__ == "__main__":
    write(__file__)
