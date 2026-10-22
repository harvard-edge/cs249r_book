#!/usr/bin/env python3
"""Every number on the Lecture 15 (Responsible Engineering, Part 1) slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations: the cited results in
Literature.Fairness (Obermeyer et al. 2019, Gender Shades, ProPublica's COMPAS
analysis, Lum and Isaac 2016), the Gender Shades re-audit rows in
Literature.Series.GenderShadesAudit, Strubell et al. 2019 in
Literature.TrainingRuns, and the fairness Scenarios.* the Responsible
Engineering chapter reads: the credit service's confusion matrices, the price
and the feedback stability of closing its TPR gap, the evaluation budget of a
1 percent group, the COMPAS rates against Chouldechova's identity, the shared
threshold on sixteen applicants, the hypothetical frontier, the proxy against
its goal, and the ranking service's exploration slice. This file only selects
values, converts units, and formats them; it types no numbers. Each value is
recorded with v() under the key the deck prints as \\val{key}; running this
file writes images/numbers.tex. make_figs.py imports this module, so figures
and slides share one computation.

The Fairlearn MetricFrame output on the matrices slide is quoted as the tool
printed it (fairlearn_gate.py, fairlearn_gate_output.txt).

Run from this folder: python3 calcs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, val, num, write, mlsysim  # noqa: E402,F401

Scenarios, Literature = mlsysim.Scenarios, mlsysim.Literature
F, RUNS = Literature.Fairness, Literature.TrainingRuns
AUDIT = Literature.Series.GenderShadesAudit

CREDIT = Scenarios.CreditServiceTwoGroups
PRICE = Scenarios.CreditServiceEqualOpportunityPrice
LOOP = Scenarios.CreditServiceBiasFeedback
BUDGET = Scenarios.OnePercentGroupEvaluationBudget
COMPAS = Scenarios.CompasPublishedRateIdentity
THRESH = Scenarios.SharedThresholdSixteenApplicants
FRONTIER = Scenarios.FairnessAccuracyFrontierSketch
PROXY = Scenarios.ClicksAgainstSatisfaction
EXPOSURE = Scenarios.RecommenderExplorationExposure


def lit(entry):
    """A Literature value as a plain number."""
    return float(entry)


def pct(x):
    """A fraction as percent."""
    return 100 * x


def audit(company, subgroup, column):
    """One Gender Shades audit value (percent)."""
    rows = [r for r in AUDIT.entries if r["company"] == company and r["subgroup"] == subgroup]
    assert len(rows) == 1, (company, subgroup)
    return rows[0][column]


print("== Hook and care chain (Literature.Fairness, Obermeyer et al. 2019)")
HOOK_COST, HOOK_HEALTH = lit(F.ObermeyerBlackShareBeforePct), lit(F.ObermeyerBlackShareAfterPct)
v("hook_cost", HOOK_COST, "{:.1f}", "Black share of auto-enrolled, ranked by cost, %")
v("hook_cost_0", HOOK_COST, "{:.0f}", "the same, whole percent (vote choice)")
v("hook_health", HOOK_HEALTH, "{:.1f}", "Black share ranked by health, %")
v("hook_health_0", HOOK_HEALTH, "{:.0f}", "the same, whole percent (vote answer)")
v("hook_mult", HOOK_HEALTH / HOOK_COST, "{:.1f}", "ranked by health / ranked by cost")
v("enroll_pct", lit(F.ObermeyerEnrollmentPercentile), "{:.0f}", "auto-enrollment percentile")
v("top_pct", 100 - lit(F.ObermeyerEnrollmentPercentile), "{:.0f}", "share enrolled, top %")
v("patients_black", lit(F.ObermeyerBlackPatients), "{:,.0f}", "Black patients in the sample")
v("patients_white", lit(F.ObermeyerWhitePatients), "{:,.0f}", "White patients in the sample")
ILL_B, ILL_W = lit(F.ObermeyerChronicConditionsBlackAt97th), lit(F.ObermeyerChronicConditionsWhiteAt97th)
v("ill_black", ILL_B, "{:.1f}", "chronic conditions, Black, at the 97th percentile")
v("ill_white", ILL_W, "{:.1f}", "chronic conditions, White")
v("ill_more_pct", pct(ILL_B / ILL_W - 1), "{:.1f}", "more chronic illnesses, %")
v("ill_more_pct_0", pct(ILL_B / ILL_W - 1), "{:.0f}", "the same, whole percent (slide)")
v("spend_gap", lit(F.ObermeyerSpendingGapUsd), "{:,.0f}", "less spent per year at equal illness, $")
v("replication_n", lit(F.ObermeyerManufacturerReplicationPatients), "{:,.0f}", "patients in the manufacturer's replication")
v("bias_cut_pct", lit(F.ObermeyerBiasReductionPct), "{:.0f}", "bias reduction with a combined label, %")
v("amazon_years", lit(F.AmazonRecruitingHistoryYears), "words", "years of hiring history (Lecture 3)")

print("== Label choice (Obermeyer et al. 2019, Table 2)")
LABEL_COST = pct(lit(F.ObermeyerTop3BlackFractionTotalCost))
LABEL_AVOID = pct(lit(F.ObermeyerTop3BlackFractionAvoidableCost))
LABEL_CHRONIC = pct(lit(F.ObermeyerTop3BlackFractionActiveChronic))
v("label_cost", LABEL_COST, "{:.1f}", "Black share of top 3%, total-cost label, %")
v("label_avoid", LABEL_AVOID, "{:.1f}", "avoidable-cost label, %")
v("label_chronic", LABEL_CHRONIC, "{:.1f}", "active-chronic-conditions label, %")
v("label_mult", LABEL_CHRONIC / LABEL_COST, "{:.1f}", "chronic label / cost label")

print("== Proxy and goal (Scenarios.ClicksAgainstSatisfaction)")
v("clicks_end", PROXY.proxy_end, "{:.2f}", "clicks, normalized")
v("sat_end", PROXY.goal_end, "{:.2f}", "satisfaction, normalized")
v("proxy_gap", PROXY.gap, "{:.2f}", "gap clicks - satisfaction")
v("click_win_pct", pct(PROXY.proxy_change), "{:.0f}", "the dashboard's click win, %")

print("== Gender Shades (Literature.Fairness, Table 4, Face++)")
GS_ALL, GS_LM = lit(F.GenderShadesErrAll), lit(F.GenderShadesErrLightMale)
GS_DM, GS_LF = lit(F.GenderShadesErrDarkMale), lit(F.GenderShadesErrLightFemale)
GS_DF = lit(F.GenderShadesErrDarkFemale)
v("gs_all", GS_ALL, "{:.1f}", "Face++ error, all faces, %")
v("gs_all_0", GS_ALL, "{:.0f}", "the same, spoken")
v("gs_lm", GS_LM, "{:.1f}", "lighter men, %")
v("gs_dm", GS_DM, "{:.1f}", "darker men, %")
v("gs_lf", GS_LF, "{:.1f}", "lighter women, %")
v("gs_df", GS_DF, "{:.1f}", "darker women, %")
v("gs_mult", GS_DF / GS_LM, "{:.0f}", "darker women / lighter men")
v("ppb_n", lit(F.GenderShadesPpbSubjects), "{:,.0f}", "Pilot Parliaments Benchmark subjects")

print("== Estimate: one group within one point (Scenarios.OnePercentGroupEvaluationBudget)")
v("group_share_pct", pct(BUDGET.group_share), "{:.0f}", "the group's share of users, %")
v("group_share_frac", BUDGET.group_share, "{:.2f}", "the same as a fraction")
v("margin_pts", pct(BUDGET.half_width), "{:.0f}", "target margin, points")
v("margin_frac", BUDGET.half_width, "{:.2f}", "target margin as a fraction")
v("conf_pct", pct(BUDGET.confidence), "{:.0f}", "confidence, %")
v("worst_rate", BUDGET.rate, "{:.1f}", "worst-case error rate")
v("worst_var", BUDGET.variance, "{:.2f}", "p(1-p) at the worst case")
v("z95", BUDGET.z, "{:.2f}", "two-sided z at 95%")
v("n_one", BUDGET.samples_needed, "{:,.0f}", "examples from the group for +/-1 point")
v("target_n", BUDGET.target_per_group, "{:,.0f}", "the chapter's rounded per-group target")
v("random_total", BUDGET.random_collection, "{:,.0f}", "random collection to expect the target")
v("random_mult", BUDGET.random_multiple, "{:.0f}", "random / stratified")

print("== Slice C: the gap between two groups")
v("slice_c_margin", pct(BUDGET.gap_half_width), "{:.1f}", "slice C gap and interval, points")
v("slice_c_frac", BUDGET.gap_half_width, "{:.3f}", "the same as a fraction")
v("gap_var", BUDGET.variance + BUDGET.variance, "{:.1f}", "summed variance of two worst-case rates")
v("n_gap_c", BUDGET.gap_samples_per_group(), "{:,.0f}", "per group for the slice C interval")
v("n_gap_1pt", BUDGET.gap_samples_per_group(BUDGET.half_width), "{:,.0f}", "per group for +/-1 point")
v("gap_margin_at_n_one", pct(BUDGET.gap_half_width_at(BUDGET.samples_needed)), "{:.2f}",
  "gap margin with the one-group count per group, points")
v("gap_margin_ratio", BUDGET.gap_half_width_at(BUDGET.samples_needed) / BUDGET.half_width, "{:.1f}",
  "gap margin over the one-group margin (the square root of 2)")
v("gap_data_ratio", BUDGET.gap_samples_per_group(BUDGET.half_width) / BUDGET.samples_needed, "{:.0f}",
  "data the gap needs over one group's")

print("== Credit service (Scenarios.CreditServiceTwoGroups)")
v("gate_pts", LOOP.gate, "{:.0f}", "release gate on the TPR gap, points")
v("credit_a_n", CREDIT.size("A"), "{:,.0f}", "Group A applicants")
v("credit_b_n", CREDIT.size("B"), "{:,.0f}", "Group B applicants")
v("credit_all_n", CREDIT.size("A") + CREDIT.size("B"), "{:,.0f}", "all applicants")
v("credit_base_pct", pct(CREDIT.base_rate("A")), "{:.0f}", "share who would repay, both groups, %")
v("credit_a_appr", CREDIT.positives("A"), "{:,.0f}", "Group A approved")
v("credit_b_appr", CREDIT.positives("B"), "{:,.0f}", "Group B approved")
for g in ("A", "B"):
    k = g.lower()
    v(f"appr_{k}", pct(CREDIT.positive_rate(g)), "{:.0f}", f"Group {g} approval, %")
    v(f"tpr_{k}", pct(CREDIT.tpr(g)), "{:.0f}", f"Group {g} TPR, %")
    v(f"fpr_{k}", pct(CREDIT.fpr(g)), "{:.0f}", f"Group {g} FPR, %")
    v(f"fnr_{k}", pct(CREDIT.fnr(g)), "{:.0f}", f"Group {g} FNR, %")
v("appr_gap", pct(CREDIT.gap("positive_rate")), "{:.0f}", "approval gap, points")
v("tpr_gap", pct(CREDIT.gap("tpr")), "{:.0f}", "TPR gap, points")
v("fpr_gap", pct(CREDIT.gap("fpr")), "{:.0f}", "FPR gap, points")
v("fnr_mult", CREDIT.fnr("B") / CREDIT.fnr("A"), "words", "Group B FNR / Group A FNR")

print("== Shared threshold (Scenarios.SharedThresholdSixteenApplicants)")
T1, T2 = THRESH.uniform_threshold, THRESH.alternative_threshold
v("thr_n", sum(THRESH.size(g) for g in THRESH.scores), "words", "applicants")
v("acc_line1", pct(THRESH.accuracy(T1)), "{:.0f}", "overall accuracy, first line, %")
v("acc_line2", pct(THRESH.accuracy(T2)), "{:.2f}", "overall accuracy, second line, %")
v("err_a1", THRESH.errors("A", T1), "{:.0f}", "Subgroup A errors, first line")
v("err_b1", THRESH.errors("B", T1), "{:.0f}", "Subgroup B errors, first line")
v("err_a2", THRESH.errors("A", T2), "words", "Subgroup A errors, second line")
v("err_b2", THRESH.errors("B", T2), "{:.0f}", "Subgroup B errors, second line")
v("err_a1_w", THRESH.errors("A", T1), "words", "Subgroup A errors, first line, in words")
v("err_b2_w", THRESH.errors("B", T2), "words", "Subgroup B errors, second line, in words")

print("== COMPAS (Literature.Fairness; Scenarios.CompasPublishedRateIdentity)")
for g, k in (("Black", "blk"), ("White", "wht")):
    d = COMPAS.groups[g]
    v(f"c_{k}_reoff", d["positives"], "{:,.0f}", f"{g} defendants who reoffended")
    v(f"c_{k}_n", d["total"], "{:,.0f}", f"{g} defendants")
    v(f"c_{k}_p", COMPAS.base_rate(g), "{:.3f}", f"{g} base rate")
    v(f"c_{k}_1mp", 1 - COMPAS.base_rate(g), "{:.3f}", f"{g} 1 - p")
    v(f"c_{k}_odds", COMPAS.odds(g), "{:.3f}", f"{g} p/(1-p)")
    v(f"c_{k}_ppv", d["ppv"], "{:.2f}", f"{g} PPV")
    v(f"c_{k}_ppv_pct", pct(d["ppv"]), "{:.0f}", f"{g} PPV, %")
    v(f"c_{k}_1mppv", 1 - d["ppv"], "{:.2f}", f"{g} 1 - PPV")
    v(f"c_{k}_prec", COMPAS.precision_term(g), "{:.3f}", f"{g} (1-PPV)/PPV")
    v(f"c_{k}_fnr", d["fnr"], "{:.3f}", f"{g} FNR")
    v(f"c_{k}_fnr_pct", pct(d["fnr"]), "{:.2f}", f"{g} FNR, % as published")
    v(f"c_{k}_fnr_pct0", pct(d["fnr"]), "{:.0f}", f"{g} FNR, whole percent")
    v(f"c_{k}_1mfnr", 1 - d["fnr"], "{:.3f}", f"{g} 1 - FNR")
    v(f"c_{k}_fpr", COMPAS.fpr(g), "{:.3f}", f"{g} FPR from the identity")
    v(f"c_{k}_fpr_pub", pct(d["fpr"]), "{:.2f}", f"{g} FPR as published, %")
    v(f"c_{k}_fpr_pub0", pct(d["fpr"]), "{:.0f}", f"{g} FPR, whole percent")
    v(f"c_{k}_base_pct", pct(COMPAS.base_rate(g)), "{:.0f}", f"{g} base rate, %")
v("c_defendants", lit(F.CompasDefendantsTotal), "{:,.0f}", "defendants in ProPublica's data")

print("== Price of fairness, Group B (Scenarios.CreditServiceEqualOpportunityPrice)")
usd = lambda q: q.to("dollar").magnitude  # noqa: E731
v("pr_base", PRICE.base_rate, "{:.1f}", "Group B would repay")
v("pr_1mbase", 1 - PRICE.base_rate, "{:.1f}", "Group B would default")
v("pr_gain", usd(PRICE.repay_margin), "{:,.0f}", "margin on a repaid loan, $")
v("pr_loss", usd(PRICE.default_loss), "{:,.0f}", "loss on a default, $")
v("pr_tpr0", PRICE.tpr_before, "{:.1f}", "Group B TPR before")
v("pr_tpr0_pct", pct(PRICE.tpr_before), "{:.0f}", "the same, %")
v("pr_tpr1_pct", pct(PRICE.tpr_after), "{:.0f}", "Group B TPR after, %")
v("pr_fpr0", PRICE.fpr_before, "{:.1f}", "Group B FPR before")
v("pr_fpr_rise", PRICE.fpr_increase, "{:.1f}", "FPR rise")
v("pr_fpr_rise_pts", pct(PRICE.fpr_increase), "{:.0f}", "FPR rise, points")
v("pr_tpr_gap", PRICE.tpr_gap, "{:.1f}", "TPR rise")
v("pr_tpr_gap_pts", pct(PRICE.tpr_gap), "{:.0f}", "TPR rise, points")
v("pr_before", usd(PRICE.baseline_utility), "{:,.0f}", "utility per applicant before, $")
v("pr_added_repaid", usd(PRICE.added_repaid_value), "{:,.0f}", "added repaid value, $")
v("pr_added_defaults", usd(PRICE.added_default_cost), "{:,.0f}", "added default cost, $")
v("pr_net", -usd(PRICE.net_change), "{:,.0f}", "net loss per Group B applicant, $")
v("pr_after", usd(PRICE.utility_after), "{:,.0f}", "utility after, $")
v("pr_all", usd(PRICE.population_average_loss), "{:,.2f}", "loss spread over every applicant, $")

print("== Frontier (Scenarios.FairnessAccuracyFrontierSketch)")
DISP_A, ACC_B = FRONTIER.point("A")[0], FRONTIER.point("B")[1]
v("disp_a", DISP_A, "{:.2f}", "A's disparity")
v("disp_b", FRONTIER.candidates["B"], "{:.2f}", "B's disparity")
v("acc_b_pct", pct(ACC_B), "{:.0f}", "B's accuracy, %")
v("frontier_mult", FRONTIER.disparity_ratio("A", "B"), "{:.1f}", "A disparity / B disparity")

print("== PredPol (Literature.Fairness, Lum and Isaac 2016)")
v("predpol_added_pct", lit(F.LumIsaacAddedCrimePct), "{:.0f}", "added recorded crime in flagged cells, %")

print("== Feedback (Scenarios.CreditServiceBiasFeedback)")
v("fb_gap0", LOOP.launch_gap, "{:.1f}", "gap at launch, points")
v("fb_gap0_0", LOOP.launch_gap, "{:.0f}", "the same, whole points")
v("fb_gap1", LOOP.next_audit_gap, "{:.1f}", "gap at the next audit, points")
v("fb_alpha", LOOP.factor, "{:.2f}", "fitted feedback factor")
v("fb_alpha_1", LOOP.factor, "{:.1f}", "the same, one decimal (spoken)")
v("fb_cross", LOOP.crossing, "{:.1f}", "ln(gate/gap0)/ln(factor)")
v("fb_fail_q", LOOP.breach_cycle, "{:.0f}", "quarter the gate fails")
v("fb_fail_q_w", LOOP.breach_cycle, "words", "the same, in words")
v("fb_gap_fail", LOOP.gap_at_breach, "{:.1f}", "gap in that quarter, points")
v("fb_orig", LOOP.unmitigated_gap, "{:.0f}", "original gap, points")
v("fb_orig_year", LOOP.unmitigated_at_horizon, "{:.1f}", "original gap after four quarters, points")
v("fb_alpha_expl", LOOP.sampled_factor, "{:.2f}", "refit factor with randomized approvals")
v("fb_decay", LOOP.sampled_gap_at_breach, "{:.1f}", "gap with the refit factor in that quarter")

print("== Ranking service (Scenarios.RecommenderExplorationExposure)")
v("rk_users_m", EXPOSURE.daily_users / 10 ** 6, "{:.0f}", "daily users, millions")
v("rk_items", EXPOSURE.slots_per_user, "{:.0f}", "items per user a day")
v("rk_impr_m", EXPOSURE.impressions / 10 ** 6, "{:.0f}", "impressions a day, millions")
v("rk_catalog_m", EXPOSURE.catalog_items / 10 ** 6, "{:.0f}", "catalog, millions of items")
v("rk_seen_pct", pct(EXPOSURE.catalog_coverage), "{:.3f}", "share of catalog seen per user a day, %")
v("rk_explore_pct", pct(EXPOSURE.explore_share), "{:.0f}", "exploration share, %")
v("rk_explore_m", EXPOSURE.explore_impressions / 10 ** 6, "{:.0f}", "exploration impressions a day, millions")

print("== Raji and Buolamwini 2019 (Literature.Series.GenderShadesAudit, darker female)")
for company, k in (("Face++", "fpp"), ("IBM", "ibm"), ("Microsoft", "msft"), ("Amazon", "amzn"), ("Kairos", "kai")):
    before, after = audit(company, "darker female", "error_may2017_pct"), audit(company, "darker female", "error_aug2018_pct")
    if before is not None and before == before:
        v(f"raji_{k}_may", before, "{:.1f}", f"{company}, May 2017, %")
    v(f"raji_{k}_aug", after, "{:.2f}" if company != "Face++" else "{:.1f}", f"{company}, August 2018, %")

v("raji_days_min", lit(F.RajiReleaseDaysMin), "{:.0f}", "fewest days to a new release")
v("raji_days_max", lit(F.RajiReleaseDaysMax), "{:.0f}", "most days to a new release")

print("== Chen et al. 2019 (Literature.RecommendationSystems)")
RS = Literature.RecommendationSystems
v("chen_users_pct", lit(RS.ChenExplorationUserSharePct), "{:.0f}", "users whose exploration data was used, %")
v("chen_gain_pct", lit(RS.ChenWatchTimeGainPct), "{:.2f}", "watch-time gain, %")

print("== Next lecture (Literature.TrainingRuns, Strubell et al. 2019)")
NAS_T = (lit(RUNS.StrubellNasTransformerCo2eLb) * mlsysim.core.units.ureg.lb).to("t").magnitude
CAR_T = (lit(RUNS.StrubellCarLifetimeCo2eLb) * mlsysim.core.units.ureg.lb).to("t").magnitude
v("nas_t", NAS_T, "{:.0f}", "architecture search estimate, tonnes CO2e")
v("car_t", CAR_T, "{:.0f}", "one car's lifetime, tonnes CO2e")
v("nas_cars", NAS_T / CAR_T, "words", "search / car lifetimes")


if __name__ == "__main__":
    write(__file__)
