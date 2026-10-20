#!/usr/bin/env python3
"""Every number on the Lecture 14 (ML Operations, Part 1) slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations: the ML-operations scenarios
(the ranking service's accuracy drift, the DLRM and ResNet-50 release
footprints on H100s, the keyword spotter's dual-bank update, the retraining
pipeline break-even, the skew cost, the A/B sizing, and the Argo Rollouts
gate), the cited results in Literature.IncidentReports (Zillow Offers, Knight
Capital) and Literature.MlOperations (Papasian and Underwood, Google SRE), and
the Wild-Time and outage series in Literature.Series. The ML Operations
chapter reads the same scenarios. This file only selects values, converts
units, and formats them; it types no numbers. Each value is recorded with v()
under the key the deck prints as \\val{key}; running this file writes
images/numbers.tex. make_figs.py imports this module, so figures and slides
share one computation.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, val, num, write, mlsysim  # noqa: E402

from mlsysim.core.units import Q_, DAYS_PER_YEAR  # noqa: E402

Scenarios, Literature, Hardware = mlsysim.Scenarios, mlsysim.Literature, mlsysim.Hardware
IR, OPS, SERIES = Literature.IncidentReports, Literature.MlOperations, Literature.Series

DRIFT = Scenarios.RankingServiceAccuracyDrift
RANK = Scenarios.DlrmReleaseFootprintH100
RESNET = Scenarios.ResNet50Fp16ReleaseFootprintH100
KWS = Scenarios.KeywordSpotterDualBankEsp32S3
AUTO, AUTO_UP = Scenarios.RetrainingPipelineAutomation, Scenarios.RetrainingPipelineAutomationWithUpkeep
CHURN, CHURN_UP = Scenarios.ChurnPipelineAutomation, Scenarios.ChurnPipelineAutomationWithUpkeep
SKEW = Scenarios.RankingSkewOnePercent
AB = Scenarios.ConversionRelativeLiftExperiment
ARGO_LADDER = Scenarios.WebCanaryLadderOnAgentTasks  # the Argo Rollouts docs ladder
ARGO_GATE = Scenarios.ArgoRolloutsSuccessRateGate

# Wild-Time rows in the order the figure draws them, with the slide's labels.
WILD_LABELS = {
    "Yearbook (portraits)": "Yearbook portraits",
    "MIMIC mortality": "MIMIC mortality (AUC)",
    "MIMIC readmission": "MIMIC readmission",
    "HuffPost (news)": "HuffPost news",
    "arXiv (papers)": "arXiv papers",
    "FMoW-Time (satellite)": "FMoW satellite",
}
_rows = {r["dataset"]: r for r in SERIES.TemporalShiftWildTime.entries}
WILD = [(label, _rows[d]["id_mean"], _rows[d]["ood_mean"]) for d, label in WILD_LABELS.items()]
WILD_DROPS = [(n, a - b) for n, a, b in WILD]
OUTAGES = [r["outages"] for r in SERIES.OutagesByMlness.entries]


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def lit(entry):
    """A Literature value as a plain number."""
    return float(entry)


def pct(x):
    """A fraction as percent."""
    return 100 * x


def clock(hour, minute, key, label):
    """Record a clock time as two keys, hour and two-digit minute."""
    v(f"{key}_h", hour, "{:.0f}", f"{label}, hour")
    v(f"{key}_min", minute, "{:02.0f}", f"{label}, minute")


print("== Hook: Zillow Offers (Literature.IncidentReports, Q2 and Q3 2021 letters)")
v("zq2_bought", lit(IR.ZillowQ2HomesBought), "{:,.0f}", "Q2 homes bought")
v("zq2_sold", lit(IR.ZillowQ2HomesSold), "{:,.0f}", "Q2 homes sold")
v("zq2_inv", lit(IR.ZillowQ2Inventory), "{:,.0f}", "Q2 inventory")
v("zq3_bought", lit(IR.ZillowQ3HomesBought), "{:,.0f}", "Q3 homes bought")
v("zq3_sold", lit(IR.ZillowQ3HomesSold), "{:,.0f}", "Q3 homes sold")
v("zq3_inv", lit(IR.ZillowQ3Inventory), "{:,.0f}", "Q3 inventory")
v("zq3_contract", lit(IR.ZillowQ3UnderContract), "{:,.0f}", "Q3 under contract")
v("z_bought_per_sold", round(lit(IR.ZillowQ3HomesBought) / lit(IR.ZillowQ3HomesSold)), "words",
  "Q3 bought per home sold, to the whole number (title)")
v("z_writedown_m", lit(IR.ZillowQ3WriteDownUsd) / 10 ** 6, "{:.0f}", "Q3 write-down, $ million")
v("z_q4_low_m", lit(IR.ZillowQ4WriteDownLowUsd) / 10 ** 6, "{:.0f}", "Q4 write-down expected, low, $ million")
v("z_q4_high_m", lit(IR.ZillowQ4WriteDownHighUsd) / 10 ** 6, "{:.0f}", "Q4 write-down expected, high, $ million")
v("z_staff_pct", pct(lit(IR.ZillowWorkforceReductionFraction)), "{:.0f}", "workforce cut, %")

print("== Silent failure: Wild-Time (Literature.Series.TemporalShiftWildTime)")
drops = [d for _, d in WILD_DROPS]
v("wt_min_drop", min(drops), "{:.0f}", "smallest drop, points")
v("wt_max_drop", max(drops), "{:.1f}", "largest drop, points")

print("== Ranking service (Scenarios.DlrmReleaseFootprintH100, RankingServiceAccuracyDrift)")
v("dlrm_gb", m(RANK.copy_bytes, "GB"), "{:.0f}", "DLRM tables, GB")
v("h100_gb", m(RANK.device.memory.capacity, "GiB"), "{:.0f}", "H100 memory (labeled GB)")
v("per_copy", RANK.accelerators_per_copy, "{:.0f}", "H100s per copy of the tables")
v("per_copy_words", RANK.accelerators_per_copy, "words", "the same, in words")
v("acc0_pct", pct(DRIFT.initial_accuracy), "{:.0f}", "accuracy at launch, %")
v("acc4_pct", pct(DRIFT.later_accuracy), "{:.0f}", "accuracy four weeks later, %")
v("drift_week", m(DRIFT.elapsed, "week"), "{:.0f}", "week of the second reading")
v("drift_week_words", m(DRIFT.elapsed, "week"), "words", "the same, in words")
v("noticed_week", m(DRIFT.noticed, "week"), "{:.0f}", "week a product manager noticed")
v("noticed_week_words", m(DRIFT.noticed, "week"), "words", "the same, in words")

print("== Hidden debt: outages (Literature.Series.OutagesByMlness, Papasian and Underwood 2020)")
v("outages_total", lit(OPS.PapasianPostmortems), "{:.0f}", "postmortems")
v("outages_not_ml", OUTAGES[0], "{:.0f}", "outages rated 0 (not ML)")
v("outages_all_ml", OUTAGES[-1], "{:.0f}", "outages rated 4 (entirely ML)")

print("== Automation break-even (Scenarios.RetrainingPipelineAutomation, ChurnPipelineAutomation)")
v("auto_build_h", m(AUTO.build_time, "hour"), "{:.0f}", "pipeline build, h")
v("auto_manual_h", m(AUTO.manual_rate, "hour/week"), "{:.0f}", "manual retraining, h a week")
v("auto_be", m(AUTO.breakeven, "week"), "{:.0f}", "break-even, weeks")
v("auto_upkeep_h", m(AUTO_UP.upkeep_rate, "hour/week"), "{:.0f}", "upkeep, h a week")
v("auto_net_h", m(AUTO_UP.manual_rate - AUTO_UP.upkeep_rate, "hour/week"), "{:.0f}",
  "hours recovered a week with upkeep")
v("auto_be_up", m(AUTO_UP.breakeven, "week"), "{:.0f}", "break-even with upkeep, weeks")
v("auto_be_up_1", m(AUTO_UP.breakeven, "week"), "{:.1f}", "the same, one decimal (figure)")
v("churn_build_h", m(CHURN.build_time, "hour"), "{:.0f}", "self-check pipeline build, h")
v("churn_manual_h", m(CHURN.manual_rate, "hour/week"), "{:.0f}", "self-check manual work, h a week")
v("churn_upkeep_h", m(CHURN_UP.upkeep_rate, "hour/week"), "{:.0f}", "self-check upkeep, h a week")
v("churn_be", m(CHURN.breakeven, "week"), "{:.0f}", "self-check break-even, weeks")
v("churn_be_up", m(CHURN_UP.breakeven, "week"), "{:.0f}", "self-check break-even with upkeep, weeks")

print("== Parity: skew cost (Scenarios.RankingSkewOnePercent, equation 2)")
v("skew_pct", pct(SKEW.skew_rate), "{:.0f}", "skew errors, %")
v("skew_qpd_m", m(SKEW.query_rate, "1/day") / 10 ** 6, "{:.0f}", "queries a day, millions")
v("skew_cost_err", m(SKEW.cost_per_error, "USD"), "{:.2f}", "cost per error, $")
v("skew_cost_yr", m(SKEW.cost_over(Q_(DAYS_PER_YEAR, "day")), "USD"),
  "{:,.0f}", "cost a year, $")

print("== Safe release: Knight Capital (Literature.IncidentReports, SEC Release 34-70694)")
v("sre_change_pct", pct(lit(OPS.SreOutagesFromLiveChangesFraction)), "{:.0f}", "outages from changes, %")
servers, missed = lit(IR.KnightSmarsServers), lit(IR.KnightServersMissed)
v("knight_servers", servers, "{:.0f}", "SMARS servers")
v("knight_servers_words", servers, "words", "the same, in words")
v("knight_updated", servers - missed, "{:.0f}", "servers with the new code")
v("knight_updated_words", servers - missed, "words", "the same, in words")
v("knight_loss_m", lit(IR.KnightRealizedLossUsd) / 10 ** 6, "{:.0f}", "loss, $ million")
v("knight_min", lit(IR.KnightTradingMinutes), "{:.0f}", "minutes of erroneous trading")
v("knight_orders", lit(IR.KnightParentOrders), "{:.0f}", "parent orders")
v("knight_exec_m", lit(IR.KnightExecutions) / 10 ** 6, "{:.0f}", "executions, millions")
v("knight_stocks", lit(IR.KnightStocks), "{:.0f}", "stocks")
v("knight_shares_m", lit(IR.KnightShares) / 10 ** 6, "{:.0f}", "shares, millions")
v("knight_emails", lit(IR.KnightAlertEmails), "{:.0f}", "'Power Peg disabled' e-mails")
v("knight_fine_m", lit(IR.KnightCivilPenaltyUsd) / 10 ** 6, "{:.0f}", "civil penalty, $ million")
opening = Q_(lit(IR.KnightMarketOpenHour), "hour") + Q_(lit(IR.KnightMarketOpenMinute), "minute")
end = opening + Q_(lit(IR.KnightTradingMinutes), "minute")
end_hour = math.floor(m(end, "hour"))
clock(lit(IR.KnightFirstAlertEmailHour), lit(IR.KnightFirstAlertEmailMinute), "knight_alert", "first e-mail")
clock(lit(IR.KnightMarketOpenHour), lit(IR.KnightMarketOpenMinute), "knight_open", "market open")
clock(end_hour, m(end - Q_(end_hour, "hour"), "minute"), "knight_end", "open plus the trading minutes")

print("== Staged exposure (Scenarios.DlrmReleaseFootprintH100, WebCanaryLadderOnAgentTasks)")
v("canary_pct", pct(RANK.canary_share), "{:.0f}", "canary share, %")
v("canary_rest_pct", pct(1 - RANK.canary_share), "{:.0f}", "traffic left on v1, %")
for i, share in enumerate(RANK.ramp):
    v(f"ramp_{i}", pct(share), "{:.0f}", f"ramp stage {i}, %")
for i, (share, pause) in enumerate(ARGO_LADDER.stages):
    v(f"argo_w{i}", pct(share), "{:.0f}", f"Argo setWeight stage {i}, %")
pauses = {m(p, "minute") for _, p in ARGO_LADDER.stages}
assert len(pauses) == 1, pauses
v("argo_pause_min", pauses.pop(), "{:.0f}", "pause after each step, min")
v("argo_threshold", ARGO_GATE.success_threshold, "{:.2f}", "successCondition threshold")
v("argo_interval_min", m(ARGO_GATE.interval, "minute"), "{:.0f}", "analysis interval, min")
v("argo_limit", ARGO_GATE.failure_limit, "{:.0f}", "failureLimit")
v("argo_limit_words", ARGO_GATE.failure_limit, "words", "failureLimit, in words")
v("argo_abort", ARGO_GATE.failures_to_abort, "{:.0f}", "failed measurement that aborts")

print("== A/B sizing (Scenarios.ConversionRelativeLiftExperiment, equation 4)")
v("ab_base_pct", pct(AB.baseline_rate), "{:.0f}", "baseline conversion, %")
v("ab_base_pct_1", pct(AB.baseline_rate), "{:.1f}", "the same, one decimal")
v("ab_rel_pct", pct(AB.relative_lift), "{:.0f}", "relative lift, %")
v("ab_target_pct", pct(AB.target_rate), "{:.1f}", "target conversion, %")
v("ab_conf_pct", pct(AB.confidence), "{:.0f}", "confidence, %")
v("ab_power_pct", pct(AB.power), "{:.0f}", "power, %")
v("ab_avail", AB.available_per_variant, "{:,.0f}", "users per variant available")
v("ab_needed", math.ceil(AB.per_variant), "{:,.0f}", "users per variant needed")
v("ab_mde_pp", pct(AB.detectable_lift()), "{:.2f}", "detectable difference, percentage points")
v("ab_mde_rate", pct(AB.baseline_rate + AB.detectable_lift()), "{:.2f}", "detectable rate, %")
v("ab_mde_rel", pct(AB.detectable_lift() / AB.baseline_rate), "{:.0f}", "detectable relative lift, %")
v("ab_mde_pp_book", pct(AB.detectable_lift()), "{:.1f}", "the detectable difference as the chapter rounds it")

print("== Node-day memory map (Scenarios.DlrmReleaseFootprintH100, ResNet50Fp16ReleaseFootprintH100)")
serving_standby = len(("serving", "standby"))
with_canary = len(("serving", "standby", "canary"))
v("node_gpus_words", RANK.node_accelerators, "words", "accelerators per node, in words")
v("node_gpus", RANK.node_accelerators, "{:.0f}", "accelerators per node")
v("two_copies", RANK.accelerators_for(serving_standby), "{:.0f}", "H100s holding two copies")
v("two_copies_words", RANK.accelerators_for(serving_standby), "words", "the same, in words")
v("retrain_gpus", RANK.retrain_accelerators, "{:.0f}", "accelerators a retraining run holds")
v("retrain_gpus_words", RANK.retrain_accelerators, "words", "the same, in words")
v("retrain_h", m(RANK.retrain_time, "hour"), "{:.0f}", "retraining run, h")
v("retrain_h_words", m(RANK.retrain_time, "hour"), "words", "the same, in words")
v("retrain_gpu_h", m(RANK.retrain_accelerator_hours, "hour"), "{:.0f}", "accelerator-hours of the run")
v("free_copies_words", RANK.free_after(with_canary) // RANK.accelerators_per_copy, "words",
  "copies that still fit beside serving, standby, and canary")
v("resnet_mb", m(RESNET.copy_bytes, "MB"), "{:.1f}", "ResNet-50 FP16 weights, MB")

print("== Dual-bank update (Scenarios.KeywordSpotterDualBankEsp32S3)")
v("kws_kb", m(KWS.bank_bytes, "KB"), "{:.0f}", "keyword-spotter model, KB")
v("kws_banks_kb", m(KWS.banks_bytes, "KB"), "{:.0f}", "two banks, KB")
v("flash_mb", m(KWS.flash, "MiB"), "{:.0f}", "flash (labeled MB)")

if __name__ == "__main__":
    write(__file__)
