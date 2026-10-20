#!/usr/bin/env python3
"""Every number on the Lecture 4 (Data Engineering, Part 1) slides and notes, read from MLSysIM.

MLSysIM holds the inputs (the keyword-spotter Scenarios.*, Literature, Datasets,
Infrastructure pricing) and the calculations; the chapter reads the same
scenarios. This file only selects values, converts units, and formats them; it
types no numbers. Each value is recorded with v() under the key the deck prints
as \\val{key}; running this file writes images/numbers.tex. make_figs.py imports
this module, so figures and slides share one computation.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Literature, Datasets, Scenarios = mlsysim.Literature, mlsysim.Datasets, mlsysim.Scenarios
ReferenceStats = mlsysim.ReferenceStats
PV = Literature.WakeWordBenchmarks
TIMES = Literature.AnnotationTimes

TARGETS = Scenarios.KeywordSpotterTargets
WAKES = Scenarios.KeywordSpotterFalseWakes
EVIDENCE = Scenarios.KeywordSpotterFalseWakeEvidence
BUDGET = Scenarios.KeywordSpotterDataBudget
MFCC = Scenarios.KeywordSpotterMfccChoice
EXAMPLES = Scenarios.KeywordSpotterExampleCountChoice
GRANULARITY = Scenarios.FullHdMaskVersusBoxes
MASKS = Scenarios.CityscapesMasksForMillionFrames
RUN = Scenarios.ImageNetLabelingVersusResNetRun
CHANCE = Scenarios.BinaryLabelChanceAgreement
DRIFT = Scenarios.WakeWordLabelQualityDrift
ACTIVE = Scenarios.ActiveLearningTenMillionPool
MSWC, BY_HAND = Datasets.MSWC, Scenarios.MswcByHand
SPLIT = Scenarios.SpeechCommandsHashSplit
INGEST = Scenarios.MillionEventBatchVersusStreaming
CORPUS = Scenarios.PretrainingCorpusGrowth

# Picovoice engines in the order the slide plots them: (label, miss rate, percent)
PV_ENGINES = [("PocketSphinx", PV.PicovoiceMissPctPocketSphinx), ("Snowboy", PV.PicovoiceMissPctSnowboy),
              ("Porcupine", PV.PicovoiceMissPctPorcupine)]

# Northcutt et al. (2021) rows the slide shows, in its order: (series name, slide label)
LABEL_ERROR_ROWS = [("QuickDraw", "QuickDraw (images)"), ("CIFAR-100", "CIFAR-100 (images)"),
                    ("ImageNet (val)", "ImageNet val (images)"), ("IMDB", "IMDB (text)"),
                    ("AudioSet", "AudioSet (audio)"), ("MNIST", "MNIST (images)")]


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def pct(x):
    """A fraction as percent."""
    return 100 * x


def label_errors():
    """{series dataset name: row} of the Northcutt label-error series."""
    return {r["dataset"]: r for r in Literature.Series.TestSetLabelErrors.entries}


print("== Hook (Tay, Literature.IncidentReports)")
v("tay_hours", Literature.IncidentReports.TayHoursToSuspension, "{:.0f}", "Tay launch to suspension, hours")

print("== Running example (keyword-spotter targets scenario)")
v("kws_detect_pct", pct(TARGETS.detection_target), "{:.0f}", "wake words caught, %")
v("kws_false_wakes", TARGETS.false_wake_budget, "{:.0f}", "false wakes a month allowed")
v("kws_model_kb", m(TARGETS.model_memory, "KB"), "{:.0f}", "always-on model budget, KB")
v("kws_prep_kb", m(TARGETS.preprocessing_memory, "KB"), "{:.0f}", "preprocessing-state budget, KB")
v("kws_budget_k", m(TARGETS.data_budget, "USD") / 10 ** 3, "{:.0f}", "data budget, thousands of dollars")
v("kws_budget_usd", m(TARGETS.data_budget, "USD"), "{:,.0f}", "data budget, dollars")
v("kws_months", m(TARGETS.timeline, "month"), "{:.0f}", "time to production, months")
v("kws_base_pct", pct(TARGETS.base_detection_rate), "{:.0f}", "illustrative base detection rate, %")

print("== Predict: false wakes a month (false-alarm budget scenario)")
rej = WAKES.rejection_levels
v("rej_a", pct(rej[0]), "{:.1f}", "first per-window rejection, %")
v("rej_b", pct(rej[1]), "{:.2f}", "second per-window rejection, %")
v("rej_c", pct(rej[2]), "{:.3f}", "third per-window rejection, %")
for key, r in zip(("a", "b", "c"), rej):
    v(f"fw_{key}", WAKES.false_alarms(r), "{:,.0f}", "false wakes a month at that rejection")
v("rej_a_miss", pct(1 - rej[0]), "{:.1f}", "per-window false-positive rate at the first level, %")
v("fw_a_round", round(WAKES.false_alarms(rej[0]), -2), "{:,.0f}", "the same count, to the hundred (answer choice)")
v("fw_a_day", WAKES.false_alarms(rej[0]) / m(WAKES.horizon, "day"), "{:.0f}", "false wakes a day at the first level")
v("fw_c_over", WAKES.false_alarms(rej[2]) / WAKES.budget, "{:.0f}", "times over budget at the third level")
v("windows_month", WAKES.windows, "{:,.0f}", "one-second windows a month")
v("listen_h_day", m(WAKES.listening_per_day, "hour"), "{:.0f}", "listening hours a day")
v("days_month", m(WAKES.horizon, "day"), "{:.0f}", "days in the month")
v("rej_needed", pct(WAKES.rejection_needed), "{:.5f}", "rejection that meets the budget, % (chapter precision)")
fpr = WAKES.target_false_positive_rate
fpr_exp = math.floor(math.log10(fpr))
v("fpr_mant", fpr / 10 ** fpr_exp, "{:.1f}", "per-window false-positive rate, mantissa")
v("fpr_exp", fpr_exp, "{:.0f}", "per-window false-positive rate, power of ten")

print("== Picovoice benchmark (Literature.WakeWordBenchmarks)")
for name, miss in PV_ENGINES:
    v(f"pv_{name.lower()}", miss, "{:.1f}", f"{name} miss rate, %")
pv_rate = mlsysim.core.units.Q_(PV.PicovoiceFalseAlarmsPerHour, "1/hour")
v("pv_hours", 1 / PV.PicovoiceFalseAlarmsPerHour, "{:.0f}", "hours per false alarm at the operating point")
v("pv_month", WAKES.false_alarms_at_rate(pv_rate), "{:.0f}", "false alarms in a month at that rate")
v("pv_over", WAKES.false_alarms_at_rate(pv_rate) / WAKES.budget, "{:.0f}", "times the false-wake budget")
v("month_hours", m(WAKES.listening_time, "hour"), "{:.0f}", "listening hours in a month")
v("pv_recordings", PV.PicovoiceRecordingsPerWord, "{:.0f}", "recordings per word (more than)")
v("pv_speakers", PV.PicovoiceSpeakers, "{:.0f}", "speakers (more than)")
v("pv_snr", PV.PicovoiceMixSnrDb, "{:.0f}", "mixing SNR, dB")
v("pv_envs", PV.PicovoiceNoiseEnvironments, "{:.0f}", "noise environments")
v("pv_words_word", PV.PicovoiceWakeWords, "words", "wake words averaged, in prose")

print("== Data budget (data-budget scenario)")
v("share_label", pct(BUDGET.shares["labeling"]), "{:.0f}", "labeling share, %")
v("share_label_frac", BUDGET.shares["labeling"], "{:.2f}", "labeling share, fraction")
v("label_price", m(BUDGET.label_price, "USD"), "{:.2f}", "price per label, $")
v("review_pct", pct(BUDGET.review_overhead), "{:.0f}", "review overhead, %")
v("review_mult", 1 + BUDGET.review_overhead, "{:.1f}", "price multiplier with review")
v("labeling_k", m(BUDGET.allocation("labeling"), "USD") / 10 ** 3, "{:.0f}", "labeling allocation, $K")
v("labeling_usd", m(BUDGET.allocation("labeling"), "USD"), "{:,.0f}", "labeling allocation, $")
v("label_cost", m(BUDGET.effective_label_cost, "USD"), "{:.2f}", "cost per label with review, $")
v("labels_k", BUDGET.labels / 10 ** 3, "{:.0f}", "labels the budget buys, thousands")
v("labels", BUDGET.labels, "{:,.0f}", "labels the budget buys")
v("storage_k", m(BUDGET.allocation("storage and processing"), "USD") / 10 ** 3, "{:.1f}", "storage and processing, $K")
v("governance_k", m(BUDGET.allocation("governance"), "USD") / 10 ** 3, "{:.1f}", "governance, $K")
v("mfcc_lo", MFCC.baseline, "{:.0f}", "MFCC coefficients chosen")
v("mfcc_hi", MFCC.alternative, "{:.0f}", "MFCC coefficients rejected")

print("== Rule of three (zero-event exposure scenario)")
v("conf_pct", pct(EVIDENCE.confidence), "{:.0f}", "one-sided confidence, %")
v("rule3", ReferenceStats.PhysicalAISafety.RuleOfThreeMultiplier, "{:.0f}", "rule-of-three factor")
v("rule3_word", ReferenceStats.PhysicalAISafety.RuleOfThreeMultiplier, "words", "the same, in prose")
v("eval_hours", round(m(EVIDENCE.exposure_needed, "hour"), -1), "{:,.0f}",
  "listening needed, hours, to the ten (as the chapter rounds)")
v("eval_gb", round(m(EVIDENCE.recording_size, "GB")), "{:.0f}", "that audio at 16 kHz, 16-bit, GB")
v("eval_khz", m(EVIDENCE.sample_rate, "kHz"), "{:.0f}", "recording rate, kHz")
v("eval_bits", m(EVIDENCE.sample_width, "bit"), "{:.0f}", "bits per sample")
v("clip_s", m(WAKES.window, "second"), "{:.0f}", "clip length, s")

print("== Label granularity (mask-versus-boxes scenario; Literature.AnnotationTimes)")
v("mask_w", GRANULARITY.image_width, "{:.0f}", "mask width, pixels")
v("mask_h", GRANULARITY.image_height, "{:.0f}", "mask height, pixels")
v("mask_m", GRANULARITY.mask_entries / 10 ** 6, "{:.1f}", "mask entries, millions")
v("boxes_word", GRANULARITY.boxes, "words", "boxes, in prose")
v("coords_word", GRANULARITY.coords_per_box, "words", "coordinates per box, in prose")
v("box_entries", GRANULARITY.box_entries, "{:.0f}", "entries for the boxes")
v("mask_ratio", GRANULARITY.mask_to_box_ratio, "{:,.0f}", "mask entries over box entries")
v("box_s", TIMES.ImageNetBoxDrawSeconds, "{:.0f}", "seconds to draw one box")
v("mask_hours", TIMES.CityscapesFineMaskHours, "{:.1f}", "hours to annotate one Cityscapes image")
v("mask_frames_word", MASKS.items / 10 ** 6, "words", "frames to mask, millions, in prose")
v("mask_total_mh", m(MASKS.total_time, "hour") / 10 ** 6, "{:.1f}", "total annotation, millions of hours")
v("mask_person_years", MASKS.person_years, "{:,.0f}", "person-years")
v("work_year_h", m(MASKS.work_year, "hour"), "{:,.0f}", "working hours in a person-year")

print("== Labeling against training (labeling-versus-training scenario)")
v("lab_budget_k", m(RUN.labeling_budget, "USD") / 10 ** 3, "{:.0f}", "labeling budget, $K")
v("lab_budget_usd", m(RUN.labeling_budget, "USD"), "{:,.0f}", "labeling budget, $")
v("lab_weeks_lo_word", m(RUN.labeling_time_low, "week"), "words", "labeling time, low, weeks, in prose")
v("lab_weeks_hi_word", m(RUN.labeling_time_high, "week"), "words", "labeling time, high, weeks, in prose")
v("lab_weeks_lo", m(RUN.labeling_time_low, "week"), "{:.0f}", "labeling time, low, weeks")
v("lab_weeks_hi", m(RUN.labeling_time_high, "week"), "{:.0f}", "labeling time, high, weeks")
v("gpus", RUN.accelerators, "{:.0f}", "A100s in the run")
v("gpus_word", RUN.accelerators, "words", "the same, in prose")
v("epochs", RUN.epochs, "{:.0f}", "epochs")
v("train_h", m(RUN.train_time, "hour"), "{:.1f}", "ResNet-50 run, hours")
v("train_usd_lo", m(RUN.train_cost_low, "USD"), "{:.0f}", "run cost, low, $")
v("train_usd_hi", m(RUN.train_cost_high, "USD"), "{:.0f}", "run cost, high, $")
v("lab_ratio_lo", round(RUN.ratio_low, -2), "{:,.0f}", "labeling / run, low, to the hundred")
v("lab_ratio_hi", round(RUN.ratio_high, -2), "{:,.0f}", "labeling / run, high, to the hundred")
v("img_per_s", m(RUN.throughput_per_accelerator, "count/second"), "{:,.0f}", "ResNet-50 images a second per A100")

print("== Label errors (Northcutt et al. 2021 via Literature.Series)")
errs = label_errors()
for name, _ in LABEL_ERROR_ROWS:
    key = "".join(c for c in name.lower() if c.isalnum())
    v(f"err_{key}", errs[name]["pct_error"], "{:.2f}", f"{name} test labels wrong, %")
v("err_max", max(r["pct_error"] for r in errs.values()), "{:.0f}", "largest share, whole percent")
v("err_rows_word", len(LABEL_ERROR_ROWS), "words", "rows the slide shows, in prose")
v("err_all_word", len(errs), "words", "rows in the book figure, in prose")

print("== Agreement (chance-agreement and drift scenarios)")
share = CHANCE.class_shares[0]
v("not_spam_pct", pct(share), "{:.0f}", "each annotator's 'not spam' share, %")
v("not_spam", share, "{:.1f}", "the same, fraction")
v("spam", CHANCE.class_shares[1], "{:.1f}", "the other class, fraction")
v("p_e", CHANCE.chance_agreement, "{:.2f}", "chance agreement")
v("p_o_pct", pct(CHANCE.observed_agreement), "{:.0f}", "raw agreement, %")
v("kappa_base", DRIFT.baseline_kappa, "{:.2f}", "campaign kappa, baseline")
v("kappa_later", DRIFT.later_kappa, "{:.2f}", "campaign kappa, later")
v("drift_months_word", m(DRIFT.elapsed, "month"), "words", "months between, in prose")

print("== Active learning (active-learning budget scenario)")
v("al_pool_m", ACTIVE.pool / 10 ** 6, "{:.0f}", "unlabeled pool, millions")
v("al_budget_k", m(ACTIVE.label_budget, "USD") / 10 ** 3, "{:.0f}", "budget, $K")
v("al_budget_usd", m(ACTIVE.label_budget, "USD"), "{:,.0f}", "budget, $")
v("al_label", m(ACTIVE.label_price, "USD"), "{:.2f}", "price per label, $")
v("al_score", m(ACTIVE.scoring_price, "USD"), "{:.2f}", "scoring price per image, $")
v("al_base_pct", pct(ACTIVE.baseline_accuracy), "{:.0f}", "accuracy now, %")
v("al_target_pct", pct(ACTIVE.target_accuracy), "{:.0f}", "accuracy target, %")
v("al_random_m", ACTIVE.random_labels_needed / 10 ** 6, "{:.0f}", "random-sampling labels, millions")
v("al_random_k", m(ACTIVE.random_label_cost, "USD") / 10 ** 3, "{:.0f}", "random sampling, $K")
v("al_random_usd", m(ACTIVE.random_label_cost, "USD"), "{:,.0f}", "random sampling, $")
v("al_active_lo", ACTIVE.active_labels_low, "{:,.0f}", "hard examples, low")
v("al_active_hi", ACTIVE.active_labels_high, "{:,.0f}", "hard examples, high")
v("al_active_usd_lo", m(ACTIVE.active_label_cost_low, "USD"), "{:,.0f}", "their labels, low, $")
v("al_active_usd_hi", m(ACTIVE.active_label_cost_high, "USD"), "{:,.0f}", "their labels, high, $")
v("al_scoreall_k", m(ACTIVE.full_pool_scoring_cost, "USD") / 10 ** 3, "{:.0f}", "scoring the whole pool, $K")
v("al_scoreall_usd", m(ACTIVE.full_pool_scoring_cost, "USD"), "{:,.0f}", "scoring the whole pool, $")
v("al_cand_k", ACTIVE.candidate_pool / 10 ** 3, "{:.0f}", "candidates scored, thousands")
v("al_cand", ACTIVE.candidate_pool, "{:,.0f}", "candidates scored")
v("al_compute_k", m(ACTIVE.compute_budget, "USD") / 10 ** 3, "{:.0f}", "scoring spend, $K")
v("al_compute_usd", m(ACTIVE.compute_budget, "USD"), "{:,.0f}", "scoring spend, $")
v("al_rest_usd", m(ACTIVE.label_budget - ACTIVE.compute_budget, "USD"), "{:,.0f}", "left for labels, $")
v("al_labels_k", ACTIVE.affordable_labels / 10 ** 3, "{:.0f}", "labels bought, thousands")
v("al_labels", ACTIVE.affordable_labels, "{:,.0f}", "labels bought")

print("== Forced alignment (Datasets.MSWC; by-hand labor scenario)")
v("mswc_m", m(MSWC.training_examples, "count") / 10 ** 6, "{:.1f}", "MSWC clips, millions")
v("mswc_kw", MSWC.keywords, "{:,.0f}", "keywords")
v("mswc_kw_k", MSWC.keywords / 10 ** 3, "{:.0f}", "keywords, thousands")
v("mswc_lang", MSWC.languages, "{:.0f}", "languages")
v("mswc_s", m(BY_HAND.time_per_item, "second"), "{:.0f}", "seconds per manual label")
v("mswc_hours", m(BY_HAND.total_time, "hour"), "{:,.0f}", "hours by hand")
v("mswc_years", BY_HAND.person_years, "{:.1f}", "person-years by hand")
v("mswc_clip_s", m(MSWC.sample_duration, "second"), "{:.0f}", "clip length, s")

print("== Speech Commands (Datasets.SpeechCommandsV2; hash-split scenario)")
v("sc_val_pct", SPLIT.validation_percent, "{:.0f}", "default validation share, %")
v("sc_test_pct", SPLIT.test_percent, "{:.0f}", "default test share, %")
v("sc_train_pct", SPLIT.train_percent, "{:.0f}", "training share, %")
sc = SPLIT.dataset
v("sc_train", m(sc.training_examples, "count"), "{:,.0f}", "v0.02 training clips")
v("sc_valid", m(sc.validation_examples, "count"), "{:,.0f}", "v0.02 validation clips")
v("sc_test", m(sc.test_examples, "count"), "{:,.0f}", "v0.02 test clips")
v("sc_test_share", pct(SPLIT.released_test_share), "{:.1f}", "v0.02 test share of the released splits, %")

print("== Fallacies and synthesis (design-space, ingest, and corpus-growth scenarios)")
v("ex_mult", EXAMPLES.ratio, "{:.0f}", "examples multiplied")
v("ex_mult_word", EXAMPLES.ratio, "words", "the same, in prose")
v("ex_gain_lo", pct(EXAMPLES.gain_low), "{:.0f}", "detection gain, low, points")
v("ex_gain_hi", pct(EXAMPLES.gain_high), "{:.0f}", "detection gain, high, points")
v("stream_ratio_word", round(INGEST.cost_ratio), "words", "streaming over batch compute cost, in prose")
v("corpus_growth", CORPUS.growth, "{:.1f}", "pre-training corpus growth per year")
stock = {r["bound"]: r["tokens"] for r in Literature.Series.PublicTextStock.entries}
v("text_stock_t", stock["median"] / 10 ** 12, "{:.0f}", "public text stock, median, trillions of tokens")
v("stock_year", round(CORPUS.crossing(stock["median"])), "{:.0f}", "year the fit reaches the median stock")


if __name__ == "__main__":
    write(__file__)
