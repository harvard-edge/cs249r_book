#!/usr/bin/env python3
"""Every number on the Lecture 14b (ML Operations, Part 2) slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations: the ML-operations scenarios
(confirming a drop from labeled outcomes, the cost of a late detection, the
ranking service's fitted decay and retraining interval, the user-age PSI, the
K-S test at a million samples, the daily metric tests of one model, accuracy
by device, the ML Test Score audit, and the severity policy), the cited
results in Literature.IncidentReports (Unity, Zillow) and
Literature.MlOperations (Breck et al., Visa, Drew et al., the ML Test Score,
He et al.), and the label-free detection series of Rabanser et al. in
local CSV series. The ML Operations chapter reads the same scenarios. This
file only selects values, converts units, and formats them; it types no
numbers. Each value is recorded with v() under the key the deck prints as
\\val{key}; running this file writes images/numbers.tex. make_figs.py imports
this module, so figures and slides share one computation.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, val, num, write, mlsysim  # noqa: E402,F401
from mlsysim.core.series_loader import load_series  # noqa: E402

from mlsysim.core.units import Q_, DAYS_PER_MONTH, DAYS_PER_YEAR  # noqa: E402

Scenarios, Literature, Ops = mlsysim.Scenarios, mlsysim.Literature, mlsysim.Ops
IR, OPS, MON = Literature.IncidentReports, Literature.MlOperations, Ops.Monitoring

DRIFT = Scenarios.RankingServiceAccuracyDrift
RANK = Scenarios.DlrmReleaseFootprintH100
CONFIRM = Scenarios.LabeledOutcomeDriftConfirmation
DELAY = Scenarios.RecommenderDetectionDelayLoss
KS_1K, KS_100K, KS_1M = Scenarios.LoudnessDriftKs1000, Scenarios.LoudnessDriftKs100k, Scenarios.ServingDayKsAgainstTraining
ALERTS = [Scenarios.OneModelDailyMetricTests, Scenarios.OneModelDailyMetricTestsAtOnePercent,
          Scenarios.OneModelDailyMetricTestsAtOnePerThousand]
AGE = Scenarios.RankingUserAgePsi
RETRAIN = Scenarios.RankingRetrainingInterval
RETRAIN_CHEAP = Scenarios.RankingRetrainingIntervalQuarterRunCost
SLICES = Scenarios.RecommendationAccuracyByDevice
MLTS = Scenarios.MlTestScoreMonitoringGapAudit
SEVERITY = Scenarios.RecommendationIncidentSeverityPolicy
RABANSER = load_series("slides/vol1/14b_ml_ops/images/data/rabanser2019_label_free_detection.csv").entries


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def lit(entry):
    """A Literature value as a plain number."""
    return float(entry)


def pct(x):
    """A fraction as percent."""
    return 100 * x


def dur_s(sec):
    """A collection time at one label a second, as the figure's tick prints it."""
    q = Q_(sec, "s")
    if m(q, "minute") < 1:
        return f"{m(q, 's'):.0f} s"
    if m(q, "hour") < 1:
        return f"{m(q, 'minute'):.1f} min"
    return f"{m(q, 'hour'):.1f} h"


def dur_d(days):
    """A collection time at 100 labels a day, as the figure's tick prints it."""
    if days < 1:
        return f"{m(Q_(days, 'day'), 'hour'):.1f} h"
    return f"{days:,.0f} day" + ("s" if days != 1 else "")


print("== Hook: Unity Audience Pinpointer (Literature.IncidentReports, Q1 2022 call)")
q2, q3 = lit(IR.UnityHeadwindQ2Fraction), lit(IR.UnityHeadwindQ3Fraction)
UNITY_SHARES = [q2, q3, 1 - q2 - q3]  # the fourth quarter is the remainder
v("unity_m", lit(IR.UnityPinpointerHeadwindUsd) / 10 ** 6, "{:.0f}", "2022 headwind, $ million")
for q, share in zip(("q2", "q3", "q4"), UNITY_SHARES):
    v(f"unity_{q}_pct", pct(share), "{:.0f}", f"share in {q.upper()}, %")

print("== Running example (Scenarios.RankingServiceAccuracyDrift, DlrmReleaseFootprintH100, RankingRetrainingInterval)")
v("dlrm_gb", m(RANK.copy_bytes, "GB"), "{:.0f}", "DLRM tables, GB")
v("acc0_pct", pct(DRIFT.initial_accuracy), "{:.0f}", "accuracy at launch, %")
v("acc4_pct", pct(DRIFT.later_accuracy), "{:.0f}", "accuracy four weeks later, %")
v("q_day", m(RETRAIN.query_rate, "1/day"), "{:,.0f}", "requests a day")
v("run_cost", m(RETRAIN.run_cost, "USD"), "{:,.0f}", "retraining run, $")

print("== Input checks (Literature.MlOperations, Breck et al. 2019; Baylor et al. 2017)")
v("breck_pipelines", lit(OPS.BreckPipelines), "{:.0f}", "pipelines, more than")
v("breck_days", lit(OPS.BreckStudyDays), "{:.0f}", "study window, days")
v("breck_new_pct", pct(lit(OPS.BreckNewFeatureColumnFraction)), "{:.0f}", "new feature column fired, %")
v("breck_ood_pct", pct(lit(OPS.BreckUnexpectedCategoryFraction)), "{:.0f}", "out-of-domain category fired, %")
v("breck_missing_pct", pct(lit(OPS.BreckMissingFeatureColumnFraction)), "{:.0f}", "missing column fired, %")
v("play_pct", pct(lit(OPS.TfxPlaySkewFixInstallLift)), "{:.0f}", "Google Play install gain, %")
BRECK_ROWS = [  # (figure label, fired, fixed)
    ("New feature column", OPS.BreckNewFeatureColumnFraction, OPS.BreckNewFeatureColumnFixedFraction),
    ("Out-of-domain category", OPS.BreckUnexpectedCategoryFraction, OPS.BreckUnexpectedCategoryFixedFraction),
    ("Missing feature column", OPS.BreckMissingFeatureColumnFraction, OPS.BreckMissingFeatureColumnFixedFraction),
    ("Data completely missing", OPS.BreckDataMissingFraction, OPS.BreckDataMissingFixedFraction),
    ("Feature in too few examples", OPS.BreckTooFewExamplesFraction, OPS.BreckTooFewExamplesFixedFraction),
]
BRECK = [(label, pct(lit(fired)), f"{pct(lit(fixed)):.0f}%") for label, fired, fixed in BRECK_ROWS]

print("== Half the drop needs about four times the evidence (Scenarios.LabeledOutcomeDriftConfirmation)")
DROPS = [CONFIRM.drop] + CONFIRM.compared_drops
NEED = [math.ceil(CONFIRM.outcomes_needed(d)) for d in DROPS]
halve = (CONFIRM.compared_drops[1] / CONFIRM.compared_drops[-1]) ** 2  # halving the drop
v("base_pct", pct(CONFIRM.baseline_accuracy), "{:.0f}", "baseline accuracy, %")
v("conf_pct", pct(CONFIRM.confidence), "{:.0f}", "confidence, %")
v("power_pct", pct(CONFIRM.power), "{:.0f}", "power, %")
for i, (d, n) in enumerate(zip(DROPS, NEED)):
    v(f"drop{i}_pts", pct(d), "{:g}", f"drop {i}, points")
    v(f"need{i}", n, "{:,.0f}", f"outcomes to confirm a {pct(d):g}-point drop")
v("halve_words", halve, "words", "halve the drop: times the outcomes, in words")
v("need_quick", halve * NEED[2], "{:,.0f}", "the quick estimate: that factor times the 1-point need")
v("budget", CONFIRM.evidence_budget, "{:,.0f}", "evidence budget, outcomes")

print("== How long until a drop is confirmed (the same scenario; Visa rules; Zillow letter)")
t_fast, t_slow = CONFIRM.collection_time("high"), CONFIRM.collection_time("low")
v("rate_hi", m(CONFIRM.label_rates["high"], "1/s"), "{:.0f}", "labels a second")
v("rate_lo", m(CONFIRM.label_rates["low"], "1/day"), "{:.0f}", "labels a day")
v("t_fast_s", m(t_fast, "s"), "{:,.0f}", "collection at 1/s, seconds")
v("t_fast_min", m(t_fast, "minute"), "{:.1f}", "collection at 1/s, minutes")
v("t_fast_min_int", m(t_fast, "minute"), "{:.0f}", "the same, whole minutes")
v("t_slow_d", m(t_slow, "day"), "{:.0f}", "collection at 100/day, days")
v("t_ratio", (t_slow / t_fast).to("").magnitude, "{:,.0f}", "slow / fast collection")
fraud_days = lit(OPS.VisaCardAbsentFraudDisputeDays)
v("fraud_days", fraud_days, "{:.0f}", "card-absent fraud dispute window, days")
v("fraud_months_words", fraud_days / DAYS_PER_MONTH, "words", "the same in 30-day months, in words")
v("zillow_lo_mo", lit(IR.ZillowResaleHorizonLowMonths), "{:.0f}", "resale horizon, low, months")
v("zillow_hi_mo", lit(IR.ZillowResaleHorizonHighMonths), "{:.0f}", "resale horizon, high, months")
v("zillow_lo_words", lit(IR.ZillowResaleHorizonLowMonths), "words", "the low end, in words")
v("zillow_hi_words", lit(IR.ZillowResaleHorizonHighMonths), "words", "the high end, in words")

print("== Label-free tests (slides/vol1/14b_ml_ops/images/data/rabanser2019_label_free_detection.csv, Rabanser et al. 2019 Table 2a)")
COIN = 1 / len(("shift", "no shift"))  # chance level of a two-way decision
RAB_N = [round(r["samples"]) for r in RABANSER]
RAB = {k: [r[k] for r in RABANSER] for k in ("large", "medium", "small")}
RAB_TICK_N = [n for n in RAB_N if math.log10(n).is_integer()]  # the decade ticks
RAB_TICKS = [(n, dur_s(m(CONFIRM.collection_time_for(n, "high"), "s")),
              dur_d(m(CONFIRM.collection_time_for(n, "low"), "day"))) for n in RAB_TICK_N]


def first_above(shift):
    return next(n for n, a in zip(RAB_N, RAB[shift]) if a > COIN)


v("rab_large_n", first_above("large"), "{:,.0f}", "large shift passes 50% at, samples")
v("rab_medium_n", first_above("medium"), "{:,.0f}", "medium shift passes 50% at, samples")
v("rab_small_n", first_above("small"), "{:,.0f}", "small shift passes 50% at, samples")
v("rab_small_days", m(CONFIRM.collection_time_for(first_above("small"), "low"), "day"), "{:.0f}",
  "the same at 100 a day, days")
i_budget = RAB_N.index(CONFIRM.evidence_budget)
v("rab_large_at_budget", RAB["large"][i_budget], "{:.2f}", "large shift at the budget")
v("rab_small_at_budget", RAB["small"][i_budget], "{:.2f}", "small shift at the budget")
v("rab_small_at_budget_pct", pct(RAB["small"][i_budget]), "{:.0f}", "the same, %")
v("rab_small_last_pct", pct(RAB["small"][-1]), "{:.0f}", "small shift at the largest sample, %")
v("rab_last_n", RAB_N[-1], "{:,.0f}", "largest sample")
v("coin_pct", pct(COIN), "{:.0f}", "coin flip, %")

print("== A month of silent failure (Scenarios.RecommenderDetectionDelayLoss)")
v("rev_m", m(DELAY.annual_value, "USD") / 10 ** 6, "{:.0f}", "revenue a year, $ million")
v("drop_pct", pct(DELAY.quality_drop), "{:.0f}", "conversion drop, %")
v("drop_frac", DELAY.quality_drop, "{:.2f}", "the same as a fraction")
v("slow_d", m(DELAY.slow_detection, "day"), "{:.0f}", "monthly review, days")
v("fast_d", m(DELAY.fast_detection, "day"), "{:.0f}", "daily check, days")
v("fast_d_words", m(DELAY.fast_detection, "day"), "words", "the same, in words")
v("days_year", DAYS_PER_YEAR, "{:.0f}", "days a year")
v("loss_slow", m(DELAY.slow_loss, "USD"), "{:,.0f}", "loss at the monthly review, $")
v("loss_fast", m(DELAY.fast_loss, "USD"), "{:,.0f}", "loss at the daily check, $")
v("loss_diff", m(DELAY.savings_per_incident, "USD"), "{:,.0f}", "difference per incident, $")
v("loss_year", m(DELAY.annual_savings, "USD"), "{:,.0f}", "a year of incidents, $")
v("incidents", DELAY.incidents_per_year, "{:.0f}", "incidents a year")
v("incidents_words", DELAY.incidents_per_year, "words", "the same, in words")
v("loss_whole_m", m(DELAY.full_year_loss, "USD") / 10 ** 6, "{:.1f}", "a whole year's loss, $ million")
v("slow_over_fast", DELAY.slow_over_fast, "{:.0f}", "monthly / daily")
LOSS_BARS = [("Monthly review", m(DELAY.slow_loss, "USD")), ("Daily check", m(DELAY.fast_loss, "USD"))]

print("== K-S at a million samples (Scenarios.LoudnessDriftKs1000, LoudnessDriftKs100k, ServingDayKsAgainstTraining)")
v("ks_c", KS_1M.coefficient, "{:.2f}", "K-S coefficient at 0.05")
for i, S in enumerate((KS_1K, KS_100K, KS_1M)):
    v(f"ks_n{i}", S.n_reference, "{:,.0f}", f"samples per side {i}")
v("ks_1k", KS_1K.critical_value, "{:.3f}", "critical value at 1,000")
v("ks_100k", KS_100K.critical_value, "{:.3f}", "critical value at 100,000")
v("ks_1m", KS_1M.critical_value, "{:.4f}", "critical value at 1,000,000")
v("ks_1m_3", KS_1M.critical_value, "{:.3f}", "the same, three decimals (the shift students judge)")
v("ks_gate", KS_1M.alert_gate, "{:.1f}", "promotion gate")
v("ks_ratio", KS_1M.gate_over_critical, "{:.0f}", "gate / critical value at 1,000,000")
KS = {S.n_reference: S.critical_value for S in (KS_1K, KS_100K, KS_1M)}

print("== False alerts (Scenarios.OneModelDailyMetricTests*); ICU alarms (Drew et al. 2014)")
A0_ = ALERTS[0]
v("variants", A0_.models, "{:.0f}", "deployment variants")
v("variants_words", A0_.models, "words", "the same, in words")
v("metrics", A0_.metrics_per_model, "{:.0f}", "metrics each")
v("series", A0_.monitors, "{:.0f}", "series")
FALSE_PER_DAY = {S.false_positive_rate: S.daily_false_alerts for S in ALERTS}
for i, S in enumerate(ALERTS):
    v(f"fpr{i}", S.false_positive_rate, "{:g}", f"false-positive rate {i}")
    v(f"false{i}", S.daily_false_alerts, "{:g}", f"false alerts a day at rate {i}")
v("false2_week", ALERTS[-1].daily_false_alerts * m(Q_(1, "week"), "day"), "{:.2f}", "a week at the lowest rate")
v("drew_alarms", lit(OPS.DrewUniqueAlarms), "{:,.0f}", "unique alarms")
v("drew_patients", lit(OPS.DrewIcuPatients), "{:.0f}", "ICU patients")
v("drew_days", lit(OPS.DrewStudyDays), "{:.0f}", "days")
v("drew_annotated", lit(OPS.DrewAnnotatedArrhythmiaAlarms), "{:,.0f}", "annotated arrhythmia alarms")
ICU_FALSE_PCT = pct(lit(OPS.DrewFalseArrhythmiaAlarmFraction))
ICU_TRUE_PCT = 100 - ICU_FALSE_PCT
v("drew_false_pct", ICU_FALSE_PCT, "{:.1f}", "false, %")

print("== The age feature's PSI (Scenarios.RankingUserAgePsi; Ops.Monitoring bands)")
AGE_BINS = [b.replace("-", "–") for b in AGE.bins]
AGE_TRAIN = [pct(x) for x in AGE.baseline_shares]
AGE_SERVE = [pct(x) for x in AGE.current_shares]
moves = [round(pct(c - b)) for b, c in zip(AGE.baseline_shares, AGE.current_shares)]
big, small = max(abs(x) for x in moves), min(abs(x) for x in moves)
old_bin = AGE.bins[-1]
b_old, c_old = AGE.shares(old_bin)
v("psi", AGE.psi, "{:.3f}", "PSI total")
v("psi_bins_words", len(AGE.bins), "words", "bins, in words")
v("psi_big_n_words", sum(abs(x) == big for x in moves), "words", "bins that moved most, in words")
v("psi_big_pts", big, "{:.0f}", "their move, points")
v("psi_small_n_words", sum(abs(x) == small for x in moves), "words", "bins that moved least, in words")
v("psi_small_pts", small, "{:.0f}", "their move, points")
v("psi_old_diff", c_old - b_old, "{:.2f}", "66+ share change")
v("psi_old_ratio", c_old / b_old, "{:.1f}", "66+ share ratio")
v("psi_old_term", AGE.term(old_bin), "{:.4f}", "66+ term")
v("psi_warn", MON.PsiWarnThreshold, "{:.1f}", "PSI warning threshold")
v("psi_crit", MON.PsiCriticalThreshold, "{:.2f}", "PSI critical threshold")
SEV = [(lvl, q) for lvl, q in SEVERITY.response_deadlines.items()]
for lvl, q in SEV:
    k = lvl.lower()
    if m(q, "hour") < 1:
        v(f"sev_{k}", m(q, "minute"), "{:.0f}", f"{lvl} deadline, minutes")
    else:
        v(f"sev_{k}", m(q, "hour"), "{:.0f}", f"{lvl} deadline, hours")

print("== Retraining cadence (Literature.MlOperations, He et al. 2014)")
v("he_ne_pct", pct(lit(OPS.HeDailyTrainingNeReductionFraction)), "{:.0f}", "NE reduction, daily against weekly, %")

print("== Retraining interval (Scenarios.RankingRetrainingInterval, equations 8 to 12)")
SLOPE = m(RETRAIN.staleness_slope, "USD/day**2")
TSTAR = m(RETRAIN.optimal_interval, "day")
CR = m(RETRAIN.run_cost, "USD")
two_c = RETRAIN.staleness_slope * RETRAIN.optimal_interval ** 2  # 2C: one cycle's triangle is C
v("v_query", m(RETRAIN.value_per_query, "USD"), "{:.2f}", "value per query, $")
v("a0", DRIFT.initial_accuracy, "{:.2f}", "A0")
v("gamma", m(DRIFT.decay_rate, "1/day"), "{:.5f}", "fitted decay rate, per day")
v("slope", SLOPE, "{:,.0f}", "QVA0 gamma, $ per day per day")
v("two_c", m(two_c, "USD"), "{:,.0f}", "2C")
v("tstar_sq", m(RETRAIN.optimal_interval ** 2, "day**2"), "{:.2f}", "2C / QVA0 gamma, day^2")
v("tstar", TSTAR, "{:.2f}", "T*, days")
v("tstar_int", TSTAR, "{:.0f}", "T*, whole days (title)")
v("c_cheap", m(RETRAIN_CHEAP.run_cost, "USD"), "{:,.0f}", "cheaper run, $")
v("tstar_cheap", m(RETRAIN_CHEAP.optimal_interval, "day"), "{:.1f}", "T* at the cheaper run, days")
v("c_dear", m(RETRAIN.run_cost * (RETRAIN.run_cost / RETRAIN_CHEAP.run_cost), "USD"), "{:,.0f}",
  "a run as many times dearer as the cheap one is cheaper, $")
CADENCE_COST = []  # (days, cost per day)
for T in RETRAIN.compared_intervals:
    d = round(m(T, "day"))
    c = m(RETRAIN.cost_rate(T), "USD/day")
    CADENCE_COST.append((d, c))
    v(f"cost_{d}d", c, "{:,.0f}", f"cost per day, every {d} days, $")
v("retrain_gpus", RANK.retrain_accelerators, "{:.0f}", "accelerators a run holds")
v("retrain_gpus_words", RANK.retrain_accelerators, "words", "the same, in words")
v("retrain_h", m(RANK.retrain_time, "hour"), "{:.0f}", "run length, h")
v("retrain_h_words", m(RANK.retrain_time, "hour"), "words", "the same, in words")
v("retrain_gpu_h", m(RANK.retrain_accelerator_hours, "hour"), "{:.0f}", "accelerator-hours")
v("acc_cost", m(RETRAIN.compute_cost, "USD"), "{:.0f}", "accelerator cost of a run, $")
v("acc_share_pct", pct(RETRAIN.compute_share), "{:.1f}", "share of the run cost, %")

print("== Scheduled against triggered retraining (Scenarios.RankingServiceAccuracyDrift)")
FLOOR = DRIFT.later_accuracy
SCHED = RETRAIN.scheduled_interval
T_HIT = m(DRIFT.time_to(FLOOR), "day")
DELAYS = [DELAY.fast_detection, CONFIRM.collection_time("low")]  # a daily check; 1,000 labels at 100 a day
A_AFTER_SCHED = DRIFT.accuracy_at(SCHED)
RESETS = [DRIFT.accuracy_at(DRIFT.time_to(FLOOR) + d) for d in DELAYS]
v("sched_d", m(SCHED, "day"), "{:.0f}", "scheduled interval, days")
v("a_sched", A_AFTER_SCHED, "{:.3f}", "accuracy before each scheduled reset")
v("floor", FLOOR, "{:.2f}", "floor")
v("t_hit", T_HIT, "{:.1f}", "days to the floor")
v("delay_short_d", m(DELAYS[0], "day"), "{:.0f}", "short detection delay, days")
v("delay_long_d", m(DELAYS[1], "day"), "{:.0f}", "long detection delay, days")
v("delay_long_words", m(DELAYS[1], "day"), "words", "the same, in words")
v("reset_short", RESETS[0], "{:.3f}", "reset after the short delay")
v("reset_long", RESETS[1], "{:.3f}", "reset after the long delay")
v("trigger_long", DRIFT.trigger_for_floor(FLOOR, DELAYS[1]), "{:.3f}", "trigger that holds the floor")

print("== Slices (Scenarios.RecommendationAccuracyByDevice)")
SLICE_ROWS = [(k, pct(SLICES.traffic_shares[k]), pct(SLICES.accuracies[k])) for k in SLICES.traffic_shares]
worst = SLICES.worst_segment
v("overall_pct", pct(SLICES.overall_accuracy), "{:.0f}", "overall accuracy, %")
v("tablet_acc_pct", pct(SLICES.accuracies[worst]), "{:.0f}", "tablet accuracy, %")
v("tablet_share_pct", pct(SLICES.traffic_shares[worst]), "{:.0f}", "tablet traffic, %")

print("== ML Test Score (Literature.MlOperations; Scenarios.MlTestScoreMonitoringGapAudit)")
sections, per = lit(OPS.MlTestScoreSections), lit(OPS.MlTestScoreTestsPerSection)
v("mlts_tests", sections * per, "{:.0f}", "tests")
v("mlts_per", per, "{:.0f}", "tests per section")
v("mlts_sections", sections, "{:.0f}", "sections")
v("mlts_sections_words", sections, "words", "sections, in words")
v("mlts_manual", lit(OPS.MlTestScoreManualTestPoints), "{:.1f}", "points, documented manual test")
v("mlts_auto", lit(OPS.MlTestScoreAutomatedTestPoints), "{:.0f}", "points, automated test")
MLTS_SCORES = MLTS.section_scores
MLTS_KEYS = {"Data": "data", "Model": "model", "Infrastructure": "infra", "Monitoring": "monit"}
for k, s in MLTS_SCORES.items():
    v(f"mlts_{MLTS_KEYS[k]}", s, "{:.0f}", f"{k} section score")
v("mlts_added_words", sum(MLTS.proposal.values()), "words", "model tests proposed, in words")
v("mlts_score", MLTS.score, "{:.0f}", "composite score")
v("mlts_after", MLTS.score_after_proposal, "{:.0f}", "score after the proposal")
v("postmortems", lit(OPS.PapasianPostmortems), "{:.0f}", "Part 1's postmortems")

if __name__ == "__main__":
    write(__file__)
