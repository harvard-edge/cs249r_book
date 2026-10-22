#!/usr/bin/env python3
"""Every number on the Lecture 12 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (registry, Literature, Ops) and the calculations
(Scenarios.*, each a scenario type over a physics equation); the chapter, ML
Operations at Scale, reads the same scenarios. This file only selects values,
converts units, and formats them; it types no numbers. Each value is recorded
with v() under the key the deck prints as \\val{key}; running this file writes
images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Literature, Scenarios, Ops = mlsysim.Literature, mlsysim.Scenarios, mlsysim.Ops
Q_ = mlsysim.core.units.Q_
FLEET = Literature.FleetOperations
MLOPS = Literature.MlOperations
OUTAGE = Literature.IncidentReports

HOOK = Scenarios.NullFeatureRankingIncident          # hook and synthesis (Example 12.2)
REDEF = Scenarios.EngagementScoreRedefinitionIncident  # Example 12.3
LOAD = Scenarios.OperationalLoadSketch
LOOKUPS = Scenarios.DailyBillionRecommendationLookups
PIT = Scenarios.ClicksBeforeNoonJoin
HISTORY = Scenarios.RecommendationFeatureHistory
INCIDENT = Scenarios.AggregationWindowRevenueIncident
EXPOSURE = Scenarios.CanaryVersusCutoverExposure
CLICKS = Scenarios.DelayedAlertClickLoss
CANARY = Scenarios.CanaryGuardrailSchedule
TEST, SMALL = Scenarios.CtrLiftExperiment, Scenarios.SmallCtrLiftExperiment
SIGMA3, SIGMA2 = Scenarios.ThreeSigmaModelFleetAlerts, Scenarios.TwoSigmaModelFleetAlerts
SLO = Scenarios.RecommendationSloBudgetVector
SRE = Scenarios.SreMultiwindowBurnAlerts
NODE = Scenarios.OwnVsRentDgxH100Node
OWN1K, HALF1K = Scenarios.OwnVsRentH100Cluster1K, Scenarios.OwnVsRentH100Cluster1KHalfBusy
OWN10K = Scenarios.OwnVsRentH100Cluster10K
CAMPAIGN = Scenarios.BurstyTrainingOn1KCluster
STARTUP, COMPANY = Scenarios.StartupOneModelTco, Scenarios.FiftyModelCompanyTco
INT8, TRAINOPT = Scenarios.Int8InferencePayback, Scenarios.TrainingOptimizationPayback
PLATFORM, PLATFORM_LOW = Scenarios.SharedPlatformBenefit50, Scenarios.SharedPlatformBenefit20


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def pct(x):
    return 100 * x


def mil(x):
    return x / 10 ** 6


def clock(key, minutes_after_midnight, label):
    """Record a UTC clock time as hour and minute keys (printed hh:mm)."""
    hours = m(Q_(minutes_after_midnight, "minute"), "hour")
    whole = math.floor(hours)
    v(f"{key}_h", whole, "{:.0f}", f"{label}, hour")
    v(f"{key}_m", m(Q_(hours - whole, "hour"), "minute"), "{:02.0f}", f"{label}, minute")


print("== Hook and synthesis: a null feature for 48 hours (Example 12.2)")
v("hook_qps", m(HOOK.request_rate, "1/s"), "{:,.0f}", "ranking model, queries/s")
v("hook_gate_ms", m(HOOK.latency_gate, "ms"), "{:.0f}", "latency gate passed, ms")
v("hook_null_pct", pct(HOOK.affected_share), "{:.0f}", "queries with the null feature, %")
v("hook_conv_pct", pct(HOOK.metric_drop), "{:.1f}", "conversion fall, %")
v("hook_hours", m(HOOK.undetected_time, "hour"), "{:.0f}", "hours unalerted")

print("== Operational load sketch (chapter figure 1)")
v("load_cross_round", round(LOAD.crossing_models, -1), "{:.0f}", "total crosses one team's capacity near, models")
v("load_conflicts_100", LOAD.conflicts(100), "{:,.0f}", "conflict pairs at 100 models")
v("load_alerts_100", LOAD.alerts(100), "{:,.0f}", "alert units at 100 models")

print("== FBLearner Flow (Meta, 2016)")
v("fbl_models", float(FLEET.FbLearnerModelsTrained), "{:,.0f}", "models trained, lower bound")
v("fbl_runs", float(FLEET.FbLearnerWorkflowRunsPerMonth), "{:,.0f}", "workflow runs in one month, lower bound")
v("fbl_authors", float(FLEET.FbLearnerWorkflowAuthors), "{:.0f}", "workflow authors, upper bound")
v("fbl_eng_pct", pct(float(FLEET.FbLearnerEngineerShare)), "{:.0f}", "engineers using it, %, lower bound")
v("fbl_pred_m", mil(float(FLEET.FbLearnerPredictionsPerSecond)), "{:.0f}", "predictions/s, millions, lower bound")

print("== Example 12.3: a shared feature redefined in place")
v("redef_models", REDEF.consumer_models, "{:.0f}", "downstream models")
v("redef_models_word", REDEF.consumer_models, "words", "the same, in words")
v("redef_teams", REDEF.owning_teams, "{:.0f}", "teams")
v("redef_teams_word", REDEF.owning_teams, "words", "the same, in words")
v("redef_acc_pct", pct(REDEF.accuracy_drop), "{:.0f}", "ranking accuracy fall, %")
v("redef_debug_h", m(REDEF.debug_time, "hour"), "{:.0f}", "hours of debugging")

print("== Monitor fleet (Napkin Math 12.7 and the 2-sigma variant)")
v("al_models", SIGMA3.models, "{:.0f}", "models monitored")
v("al_metrics", SIGMA3.metrics_per_model, "{:.0f}", "metrics per model")
v("al_interval_min", m(SIGMA3.check_interval, "minute"), "{:.0f}", "check interval, min")
v("al_monitors", SIGMA3.monitors, "{:,.0f}", "monitors")
v("al_checks_day", SIGMA3.checks_per_day, "{:.0f}", "checks per monitor per day")
v("al_p3", SIGMA3.false_positive_rate, "{:.3f}", "3-sigma false-positive rate per check")
v("al_daily3", SIGMA3.daily_false_alerts, "{:,.0f}", "3-sigma false crossings per day")
v("al_p2", SIGMA2.false_positive_rate, "{:.2f}", "2-sigma false-positive rate per check")
v("al_p2_pct", pct(SIGMA2.false_positive_rate), "{:.0f}", "the same, %")
v("al_daily2", SIGMA2.daily_false_alerts, "{:,.0f}", "2-sigma false crossings per day")
v("al_dedup_pct", pct(SIGMA2.dedup_share), "{:.0f}", "crossings deduplicated, %")
v("al_left", SIGMA2.remaining_after_dedup, "{:.0f}", "pages left per day")
v("al_any3", SIGMA3.any_false_alert(SIGMA3.monitors), "{:.2f}", "P(any false), 3-sigma, all monitors")

print("== Feature store scale (Sec. 12.4.6)")
v("fs_recs_b", m(LOOKUPS.request_rate, "1/day") / 10 ** 9, "{:.0f}", "recommendations a day, billions")
v("fs_features", LOOKUPS.reads_per_request, "{:.0f}", "features per recommendation")
v("fs_lookups_b", LOOKUPS.reads_per_day / 10 ** 9, "{:.0f}", "lookups a day, billions")
v("fs_lookups_m", mil(LOOKUPS.reads_per_second), "{:.2f}", "average lookups/s, millions")
v("fs_budget_ms", m(LOOKUPS.request_budget, "ms"), "{:.0f}", "recommendation budget, ms")
v("fs_store_lo_ms", m(LOOKUPS.lookup_budget_low, "ms"), "{:.0f}", "store's share, low, ms")
v("fs_store_hi_ms", m(LOOKUPS.lookup_budget_high, "ms"), "{:.0f}", "store's share, high, ms")

print("== Point-in-time join (figure 6) and feature history (Sec. 12.4.3)")
v("pit_before", PIT.events_before_prediction, "{:.0f}", "clicks before the noon prediction")
v("pit_before_word", PIT.events_before_prediction, "words", "the same, in words")
v("pit_day", PIT.events_by_day_end, "{:.0f}", "clicks by midnight")
v("pit_day_word", PIT.events_by_day_end, "words", "the same, in words")
v("pit_ratio", PIT.leakage_inflation, "{:.1f}", "naive join over the true value")
v("fh_users_m", mil(HISTORY.entities), "{:.0f}", "users, millions")
v("fh_features", HISTORY.features, "{:,.0f}", "features")
v("fh_values_t", HISTORY.values / 10 ** 12, "{:.0f}", "feature values in history, trillions")
v("fh_row_b", m(HISTORY.row_bytes, "byte"), "{:.0f}", "bytes per row")
v("fh_pb", m(HISTORY.total_bytes, "PB"), "{:.0f}", "history before compression, PB")

print("== Data validation (Breck et al. 2019) and TFX (Baylor et al. 2017)")
v("breck_pipelines", float(MLOPS.BreckPipelines), "{:.0f}", "pipelines, lower bound")
v("breck_days", float(MLOPS.BreckStudyDays), "{:.0f}", "days")
v("breck_new_pct", pct(float(MLOPS.BreckNewFeatureColumnFraction)), "{:.0f}", "new feature column, %")
v("breck_missing_pct", pct(float(MLOPS.BreckMissingFeatureColumnFraction)), "{:.0f}", "missing feature column, %")
v("breck_category_pct", pct(float(MLOPS.BreckUnexpectedCategoryFraction)), "{:.0f}", "unexpected category, %")
v("breck_few_pct", pct(float(MLOPS.BreckTooFewExamplesFraction)), "{:.0f}", "too few examples, %")
v("breck_gone_pct", pct(float(MLOPS.BreckDataMissingFraction)), "{:.0f}", "data completely missing, %")
v("tfx_play_pct", pct(float(MLOPS.TfxPlaySkewFixInstallLift)), "{:.0f}", "Play install-rate gain, %")
load_lo, load_hi = float(MLOPS.TfxModelLoadP999LowMs), float(MLOPS.TfxModelLoadP999HighMs)
own_lo, own_hi = float(MLOPS.TfxDedicatedLoadP999LowMs), float(MLOPS.TfxDedicatedLoadP999HighMs)
v("tfx_load_lo", load_lo, "{:,.0f}", "p99.9 during loads, low, ms")
v("tfx_load_hi", load_hi, "{:,.0f}", "p99.9 during loads, high, ms")
v("tfx_own_lo", own_lo, "{:.0f}", "p99.9 with a loading threadpool, low, ms")
v("tfx_own_hi", own_hi, "{:.0f}", "p99.9 with a loading threadpool, high, ms")
v("tfx_ratio_lo", load_lo / own_lo, "{:.0f}", "low ends, ratio")
v("tfx_ratio_hi", load_hi / own_hi, "{:.0f}", "high ends, ratio")

print("== Broken-feature cost (Sec. 12.4.7, constructed)")
v("fc_ctr_before", pct(INCIDENT.metric_before), "{:.1f}", "CTR before, %")
v("fc_ctr_after", pct(INCIDENT.metric_after), "{:.2f}", "CTR after, %")
v("fc_drop_pct", pct(INCIDENT.relative_drop), "{:.0f}", "CTR and revenue fall, %")
v("fc_drop", INCIDENT.relative_drop, "{:.2f}", "the same, fraction")
v("fc_revenue_m", mil(m(INCIDENT.weekly_revenue, "USD")), "{:.0f}", "revenue per week, $M")
v("fc_days", m(INCIDENT.undetected, "day"), "{:.0f}", "days undetected")
v("fc_days_word", m(INCIDENT.undetected, "day"), "words", "the same, in words")
v("fc_week_days", m(Q_(1, "week"), "day"), "{:.0f}", "days in a week")
v("fc_loss_k", m(INCIDENT.undetected_loss, "USD") / 10 ** 3, "{:.0f}", "five-day loss, $K")
v("fc_alert_h", m(INCIDENT.alerted, "hour"), "{:.0f}", "alert and containment, h")
v("fc_undetected_h", m(INCIDENT.undetected, "hour"), "{:.0f}", "five days, h")
v("fc_alert_loss_k", m(INCIDENT.alerted_loss, "USD") / 10 ** 3, "{:.1f}", "four-hour loss, $K")
v("fc_window_ratio", INCIDENT.window_ratio, "{:.0f}", "exposure window ratio")
v("fc_reduction_pct", pct(INCIDENT.impact_reduction), "{:.0f}", "loss reduction, %")
v("fc_slo_h", m(INCIDENT.freshness_slo, "hour"), "{:.0f}", "freshness SLO, h")
v("fc_mean_before", INCIDENT.feature_mean_before, "{:.2f}", "user_engagement_score mean before")
v("fc_mean_after", INCIDENT.feature_mean_after, "{:.2f}", "user_engagement_score mean after")
v("fc_band_pct", pct(INCIDENT.shift_threshold), "{:.0f}", "distribution-gate band, %")

print("== Exposure (Napkin Math 12.5) and the delayed alert (Napkin Math 12.6)")
v("exp_full_pct", pct(EXPOSURE.full_share), "{:.0f}", "all-at-once share, %")
v("exp_canary_pct", pct(EXPOSURE.canary_share), "{:.0f}", "canary share, %")
v("exp_ratio", EXPOSURE.exposure_ratio, "{:.0f}", "exposure ratio")
v("click_ctr_before", pct(CLICKS.ctr_before), "{:.0f}", "CTR before, %")
v("click_ctr_after", pct(CLICKS.ctr_after), "{:.1f}", "CTR after, %")
v("click_drop_pt", pct(CLICKS.ctr_drop), "{:.1f}", "CTR loss, points")
v("click_drop", CLICKS.ctr_drop, "{:.3f}", "CTR loss, fraction")
v("click_qps", m(CLICKS.request_rate, "1/s"), "{:,.0f}", "requests/s")
v("click_requests_m", mil(CLICKS.requests), "{:.0f}", "requests a day, millions")
v("click_lost_m", mil(CLICKS.lost_clicks), "{:.2f}", "lost clicks a day, millions")
v("click_loss_m", mil(m(CLICKS.loss, "USD")), "{:.2f}", "loss a day, $M")
v("click_value", m(CLICKS.value_per_click, "USD"), "{:.2f}", "value per click, $")
v("click_loss_hour_k", m(CLICKS.loss_rate, "USD/hour") / 10 ** 3, "{:.0f}", "loss per hour, $K")

print("== Canary stages (equation 7, Sec. 12.5.5)")
share1, dur1 = CANARY.stages[0]
share2, dur2 = CANARY.stages[1]
share3, dur3 = CANARY.stages[2]
share4, dur4 = CANARY.stages[3]
v("canary_samples", CANARY.samples_per_variant, "{:,.0f}", "guardrail samples per stage")
v("canary_rate_exp", math.log10(m(CANARY.request_rate, "1/hour")), "{:.0f}", "request rate, power of ten per hour")
v("canary_rate_m", mil(m(CANARY.request_rate, "1/hour")), "{:.0f}", "request rate, millions per hour")
v("canary_p1", share1, "{:.2f}", "first stage share")
v("canary_p1_pct", pct(share1), "{:.0f}", "first stage share, %")
v("canary_t1_h", m(CANARY.stage_minimum(share1), "hour"), "{:.0f}", "minimum first stage, h")
v("canary_p2_pct", pct(share2), "{:.0f}", "second stage share, %")
v("canary_t2_min", m(CANARY.stage_minimum(share2), "minute"), "{:.0f}", "minimum second stage, min")
v("canary_d1_h", m(dur1, "hour"), "{:.0f}", "configured first stage, h")
v("canary_d2_min", m(dur2, "minute"), "{:.0f}", "configured second and third stages, min")
v("canary_p3_pct", pct(share3), "{:.0f}", "third stage share, %")
v("canary_p4_pct", pct(share4), "{:.0f}", "fourth stage share, %")
v("canary_total_h", m(CANARY.total, "hour"), "{:.0f}", "configured schedule, h")

print("== Experiment sizing (equation 4, Sec. 12.5.3)")
v("mde_p", TEST.baseline_rate, "{:.2f}", "baseline CTR")
v("mde_d", TEST.lift, "{:.3f}", "lift to detect")
v("mde_za", TEST.z_alpha, "{:.2f}", "z for 95% confidence")
v("mde_zb", TEST.z_beta, "{:.2f}", "z for 80% power")
v("mde_z2", TEST.z_sum_squared, "{:.2f}", "(z_a + z_b)^2")
v("mde_var", TEST.two_variance, "{:.3f}", "2 p (1 - p)")
v("mde_d2", TEST.lift_squared, "{:.6f}", "lift squared")
v("mde_n", TEST.per_variant, "{:,.0f}", "users per variant")
v("mde_d_pt", pct(TEST.lift), "{:.1f}", "lift, points")
v("mde_total_round", round(TEST.total, -4), "{:,.0f}", "users in all, rounded")
v("mde_req_m", mil(m(TEST.daily_requests, "1/day")), "{:.0f}", "requests a day, millions")
v("mde_hours_bound", math.ceil(m(TEST.collection_time, "hour")), "{:.0f}", "hours to collect, upper bound")
v("mde2_p_pct", pct(SMALL.baseline_rate), "{:.0f}", "small-lift baseline, %")
v("mde2_d_pt", pct(SMALL.lift), "{:.2f}", "small lift, points")
v("mde2_n", SMALL.per_variant, "{:,.0f}", "users per variant, small lift")
v("mde2_hours", m(SMALL.collection_time, "hour"), "{:.0f}", "hours to collect, small lift")
v("fw_tests", TEST.family_size, "{:.0f}", "uncorrected tests")
v("fw_tests_word", TEST.family_size, "words", "the same, in words")
v("fw_conf", 1 - TEST.significance, "{:.2f}", "1 - alpha")
v("fw_pct", pct(TEST.familywise_error), "{:.0f}", "P(at least one false win), %")
v("netflix_mult", float(MLOPS.NetflixInterleavingSampleReduction), "{:.0f}", "interleaving: fewer users, times")

print("== Monitoring hierarchy, SLO vector (figure 12), burn rates (SRE Workbook)")
v("psi_review", float(Ops.Monitoring.PsiCriticalThreshold), "{:.2f}", "PSI that opens a review")
v("slo_fresh", SLO.consumption["freshness"], "{:.2f}", "freshness budget consumed")
v("slo_mean", SLO.mean, "{:.2f}", "mean of the four dimensions")
v("sre_slo_pct", pct(SRE.slo_target), "{:.1f}", "SLO, %")
v("sre_days", m(SRE.budget_period, "day"), "{:.0f}", "budget period, days")
v("sre_hours", m(SRE.budget_period, "hour"), "{:.0f}", "budget period, hours")
for tier, key in (("page_fast", "fast"), ("page_slow", "slow"), ("ticket", "ticket")):
    burn, window = SRE.tiers[tier]
    v(f"sre_{key}_burn", burn, "{:g}", f"{tier} burn rate")
    v(f"sre_{key}_spent_pct", pct(SRE.budget_spent(tier)), "{:.0f}", f"{tier} budget spent, %")
v("sre_fast_window_word", m(SRE.tiers["page_fast"][1], "hour"), "words", "fast window, hours, in words")
v("sre_fast_window_h", m(SRE.tiers["page_fast"][1], "hour"), "{:.0f}", "fast window, h")
v("sre_slow_window_h", m(SRE.tiers["page_slow"][1], "hour"), "{:.0f}", "slow window, h")
v("sre_ticket_window_d", m(SRE.tiers["ticket"][1], "day"), "{:.0f}", "ticket window, days")

print("== Meta outage, 4 October 2021 (Cloudflare)")
start = float(OUTAGE.FacebookOutage2021UnreachableStartMinuteUtc)
end = float(OUTAGE.FacebookOutage2021UnreachableEndMinuteUtc)
clock("meta_peak", float(OUTAGE.FacebookOutage2021RoutingPeakMinuteUtc), "routing changes peak")
clock("meta_start", start, "facebook.com stops resolving")
clock("meta_end", end, "facebook.com resolves again")
clock("meta_return", float(OUTAGE.FacebookOutage2021RoutesReturnMinuteUtc), "routes return")
v("meta_hours", m(Q_(end - start, "minute"), "hour"), "{:.1f}", "hours unreachable")
v("meta_dns", float(OUTAGE.FacebookOutage2021DnsQueryMultiple), "{:.0f}", "DNS queries over normal")

print("== Philly (Jeon et al. 2019, Tables 3 and 6)")
killed_t, failed_t = float(FLEET.PhillyGpuTimeKilledFraction), float(FLEET.PhillyGpuTimeUnsuccessfulFraction)
v("philly_jobs", float(FLEET.PhillyJobCount), "{:,.0f}", "jobs in the trace")
v("philly_months_word", float(FLEET.PhillyTraceMonths), "words", "trace length, months, in words")
v("philly_pass_pct", pct(float(FLEET.PhillyJobsPassedFraction)), "{:.0f}", "jobs passed, %")
v("philly_killed_pct", pct(float(FLEET.PhillyJobsKilledFraction)), "{:.0f}", "jobs killed, %")
v("philly_failed_pct", pct(float(FLEET.PhillyJobsUnsuccessfulFraction)), "{:.0f}", "jobs failed, %")
v("philly_killed_time", pct(killed_t), "{:.1f}", "GPU time in killed jobs, %")
v("philly_failed_time", pct(failed_t), "{:.1f}", "GPU time in failed jobs, %")
v("philly_lost_pct", pct(killed_t + failed_t), "{:.0f}", "GPU time in killed and failed jobs, %")
v("philly_util_pct", pct(float(FLEET.PhillyMeanGpuUtilization)), "{:.0f}", "mean utilization of in-use GPUs, %")

print("== Own against rent: the 1,024-H100 cluster (figure 13)")
v("own_gpus", OWN1K.gpus, "{:,.0f}", "GPUs")
v("own_nodes", OWN1K.nodes, "{:.0f}", "nodes")
v("gpus_per_node", OWN1K.gpus_per_node, "{:.0f}", "GPUs per node")
v("own_hw_m", mil(m(OWN1K.hardware_capex, "USD")), "{:.1f}", "servers, $M")
v("node_price_k", m(OWN1K.node_price, "USD") / 10 ** 3, "{:.0f}", "node price, $K")
v("node_price", m(OWN1K.node_price, "USD"), "{:,.0f}", "node price, $")
v("own_net_m", mil(m(OWN1K.network_capex, "USD")), "{:.0f}", "InfiniBand, $M")
v("own_fac_m", mil(m(OWN1K.facility_capex, "USD")), "{:.0f}", "facility, $M")
v("own_upfront_m", mil(m(OWN1K.upfront, "USD")), "{:.1f}", "up front, $M")
v("own_opex_m", mil(m(OWN1K.owned_cost_per_year, "USD")), "{:.1f}", "power and staff, $M a year")
v("rent_rate", m(OWN1K.rent_rate, "USD/hour"), "{:.0f}", "rent, $ per GPU-hour")
v("own_util_pct", pct(OWN1K.utilization), "{:.0f}", "sustained utilization, %")
v("own_rent_month_m", mil(m(OWN1K.rent_per_month, "USD")), "{:.2f}", "rent a month, $M")
v("own_cross_months", OWN1K.crossover_months, "{:.1f}", "rent overtakes owning, months")
v("own_years_word", OWN1K.years, "words", "service life, years, in words")
v("own_life_months", m(Q_(OWN1K.years, "year"), "month"), "{:.0f}", "service life, months")
v("half_util_pct", pct(HALF1K.utilization), "{:.0f}", "lower utilization, %")
v("half_rent_month_m", mil(m(HALF1K.rent_per_month, "USD")), "{:.2f}", "rent a month at that utilization, $M")
v("be_node", pct(NODE.breakeven_utilization), "{:.1f}", "break-even, one node, %")
v("be_1k", pct(OWN1K.breakeven_utilization), "{:.1f}", "break-even, this cluster, %")
v("be_10k", pct(OWN10K.breakeven_utilization), "{:.1f}", "break-even, 10,000 GPUs, %")
v("run_owned_m", mil(m(CAMPAIGN.owned_cost_per_run, "USD")), "{:.1f}", "owned cost per run, $M")
v("runs_per_year_word", CAMPAIGN.runs_per_year, "words", "runs a year, in words")
v("run_rent_lo_m", mil(m(CAMPAIGN.rented_cost_per_run(CAMPAIGN.run_low), "USD")), "{:.1f}", "rented run, short, $M")
v("run_rent_hi_m", mil(m(CAMPAIGN.rented_cost_per_run(CAMPAIGN.run_high), "USD")), "{:.1f}", "rented run, long, $M")

print("== Own against rent: 10,000 H100s for three years (Napkin Math 12.9)")
v("tenk_gpus", OWN10K.gpus, "{:,.0f}", "GPUs")
v("tenk_exp", math.log10(OWN10K.gpus), "{:.0f}", "GPUs, power of ten")
v("tenk_nodes", OWN10K.nodes, "{:,.0f}", "nodes")
v("tenk_hw_m", mil(m(OWN10K.hardware_capex, "USD")), "{:.1f}", "hardware, $M")
v("tenk_net_m", mil(m(OWN10K.network_capex, "USD")), "{:.0f}", "fabric, $M")
v("tenk_fac_m", mil(m(OWN10K.facility_capex, "USD")), "{:.0f}", "liquid-cooled hall, $M")
v("tenk_upfront_m", mil(m(OWN10K.upfront, "USD")), "{:.1f}", "up front, $M")
v("tenk_power_m", mil(m(OWN10K.electricity_per_year_full, "USD")), "{:.1f}", "GPU-only electricity, $M a year")
v("tenk_staff_m", mil(m(OWN10K.staffing_per_year, "USD")), "{:.0f}", "staff, $M a year")
v("tenk_opex_m", mil(m(OWN10K.owned_cost_per_year, "USD")), "{:.1f}", "running cost, $M a year")
v("tenk_years", OWN10K.years, "{:.0f}", "years")
v("tenk_years_word", OWN10K.years, "words", "the same, in words")
v("tenk_owned_m", mil(m(OWN10K.owned_total, "USD")), "{:.1f}", "owned three-year total, $M")
v("tenk_tdp_w", m(OWN10K.power / OWN10K.gpus, "W"), "{:.0f}", "GPU power, W")
v("tenk_pue", OWN10K.pue, "{:.1f}", "PUE")
v("hours_per_year", m(OWN10K.hours_per_year, "hour"), "{:,.0f}", "hours a year")
v("kwh_price", m(OWN10K.electricity_price, "USD/kWh"), "{:.2f}", "electricity, $ per kWh")
v("tenk_util", OWN10K.utilization, "{:.1f}", "utilization, fraction")
v("tenk_rent_year_m", mil(m(OWN10K.rent_per_year, "USD")), "{:.1f}", "rent a year, $M")
v("tenk_rent_total_m", mil(m(OWN10K.rented_total, "USD")), "{:.1f}", "rent over three years, $M")
v("tenk_saving_m", mil(m(OWN10K.savings, "USD")), "{:.1f}", "saving, $M")
v("tenk_full_m", mil(m(OWN10K.rent_lifetime_full, "USD")), "{:,.1f}", "rent for every hour of three years, $M")
v("tenk_hours", m(OWN10K.lifetime_hours, "hour"), "{:,.0f}", "hours in three years")

print("== TCO (table 31), payback, platform benefit")
v("tco_startup", m(STARTUP.total, "USD/month"), "{:,.0f}", "startup, $ a month")
v("tco_startup_dom_pct", pct(STARTUP.shares[STARTUP.dominant]), "{:.0f}", "startup's largest term (iteration), %")
v("tco_company", m(COMPANY.total, "USD/month"), "{:,.0f}", "50-model company, $ a month")
v("tco_company_dom_pct", pct(COMPANY.shares[COMPANY.dominant]), "{:.0f}", "company's largest term (inference), %")
v("int8_cost_k", m(INT8.implementation_cost, "USD") / 10 ** 3, "{:.0f}", "INT8 implementation cost, $K")
v("int8_bill_k", m(INT8.monthly_bill, "USD/month") / 10 ** 3, "{:.0f}", "inference bill, $K a month")
v("int8_cut_pct", pct(INT8.reduction), "{:.0f}", "inference cut, %")
v("int8_weeks_word", round(m(INT8.payback, "week")), "words", "INT8 payback, weeks, in words")
v("train_payback_months", m(TRAINOPT.payback, "month"), "{:.1f}", "training optimization payback, months")
v("platform_models", PLATFORM.models, "{:.0f}", "models")
v("platform_hours", m(PLATFORM.hours_saved, "hour/month"), "{:.0f}", "hours saved per model a month")
v("platform_rate", m(PLATFORM.engineer_rate, "USD/hour"), "{:.0f}", "loaded engineer rate, $ an hour")
v("platform_cost", m(PLATFORM.platform_cost, "USD/month"), "{:,.0f}", "platform cost, $ a month")
v("platform_bcr", PLATFORM.benefit_cost_ratio, "{:.2f}", "benefit-cost ratio")
v("platform_low_models", PLATFORM_LOW.models, "{:.0f}", "smaller portfolio, models")

if __name__ == "__main__":
    write(__file__)
