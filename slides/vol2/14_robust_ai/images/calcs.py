#!/usr/bin/env python3
"""Every number on the Lecture 14 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (Literature, the dataset and number-format registries)
and the calculations (Scenarios.*, each a scenario type over a physics
equation); the chapter, Robust AI, reads the same scenarios. This file only
selects values, converts units, and formats them; it types no numbers. Each
value is recorded with v() under the key the deck prints as \\val{key};
running this file writes images/numbers.tex.

Sources by slide:
  - Hook and synthesis: Literature.RobustnessStudies (Eykholt et al. 2018).
  - Tempe: Literature.IncidentReports (NTSB HAR-19/03; preliminary report).
  - Fleet silent errors: Scenarios.FleetSilentErrorHour1K and 10K.
  - ImageNet-V2: Scenarios.ImageNetTestbedV2Shift.
  - Mean test: Scenarios.FeatureMeanShiftTest (Napkin Math 14.1).
  - Fraud service: Scenarios.FraudCountryPsi, FraudAmountDrift, FraudAmountKs.
  - Calibration, jailbreaks, FGSM, BadNets: Literature.RobustnessStudies.
  - Selective answering and conformal: Scenarios.SelectiveAnswering90,
    Scenarios.ConformalThreshold1K.
  - PGD-7 and FGSM work: Scenarios.PgdSevenTraining and
    FgsmCleanPlusAdversarialTraining (Napkin Math 14.2).
  - Four cells: Scenarios.ResNet50CleanVersusRobustAttacked.
  - RobustBench: Scenarios.RobustBenchCifar10LinfProgress.
  - Certified radius: Scenarios.SmoothingRadiusCifar10.
  - Bit flip: Scenarios.Fp32WeightExponentFlip; Llama 3 interruptions from
    Literature.TrainingRuns and Literature.Series.Llama3Interruptions; GPT-3
    parameters from Models.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Literature, Scenarios, Models = mlsysim.Literature, mlsysim.Scenarios, mlsysim.Models
STUDY = Literature.RobustnessStudies
TEMPE = Literature.IncidentReports

SDC1K, SDC10K = Scenarios.FleetSilentErrorHour1K, Scenarios.FleetSilentErrorHour10K
IV2 = Scenarios.ImageNetTestbedV2Shift
MEAN = Scenarios.FeatureMeanShiftTest
PSI, KL, KS = Scenarios.FraudCountryPsi, Scenarios.FraudAmountDrift, Scenarios.FraudAmountKs
SEL = Scenarios.SelectiveAnswering90
CONF = Scenarios.ConformalThreshold1K
PGD, FGSM = Scenarios.PgdSevenTraining, Scenarios.FgsmCleanPlusAdversarialTraining
CELLS = Scenarios.ResNet50CleanVersusRobustAttacked
RB = Scenarios.RobustBenchCifar10LinfProgress
CERT = Scenarios.SmoothingRadiusCifar10
FLIP = Scenarios.Fp32WeightExponentFlip
LLAMA3 = Literature.Series.Llama3Interruptions.entries


def pct(x):
    return 100 * x


def f(x):
    """A Literature scalar as a float."""
    return float(x)


def row(rows, key, name):
    return next(r for r in rows if r[key] == name)


print("== Hook and synthesis: stickers on a stop sign (Eykholt et al. 2018)")
v("lisa_clean_pct", f(STUDY.LisaCnnCleanAccuracyPct), "{:.0f}", "LISA-CNN clean test accuracy, %")
v("sticker_lab_pct", f(STUDY.StickerLabTargetedSuccessPct), "{:.0f}", "targeted success, lab images, %")
v("sticker_drive_pct", f(STUDY.StickerDriveByTargetedSuccessPct), "{:.1f}", "targeted success, drive-by frames, %")

print("== Tempe, 2018 (NTSB HAR-19/03)")
detect, recog = f(TEMPE.TempeFirstDetectionS), f(TEMPE.TempeEmergencyRecognitionS)
hold, alert = f(TEMPE.TempeActionSuppressionS), f(TEMPE.TempeSuppressionEndS)
prelim = f(TEMPE.TempePreliminaryRecognitionS)
assert round(recog - hold, 1) == alert       # the data table's alert time is recognition less the hold
v("tempe_detect_s", detect, "{:.1f}", "first detection before impact, s")
v("tempe_recog_s", recog, "{:.1f}", "collision recognized before impact, s")
v("tempe_gap_s", detect - recog, "{:.1f}", "detection to recognition, s (derived)")
v("tempe_hold_s", hold, "{:.0f}", "braking held, s")
v("tempe_hold_word", hold, "words", "the same, in words")
v("tempe_alert_s", alert, "{:.1f}", "alert before impact, s")
v("tempe_prelim_s", prelim, "{:.1f}", "preliminary report's recognition time, s")

print("== Fleet silent errors (self-check; rate assumed)")
v("sdc_p_exp", -math.log10(SDC1K.event_rate), "{:.0f}", "silent-error rate per device-hour, 10 to the minus this")
v("sdc_n_lo", SDC1K.n_devices, "{:,.0f}", "accelerators, small job")
v("sdc_n_hi", SDC10K.n_devices, "{:,.0f}", "accelerators, large job")
v("sdc_q", 1 - SDC10K.event_rate, "{:.4f}", "per-device chance of no error in an hour")
v("sdc_p_lo", SDC1K.p_step, "{:.3f}", "P(at least one) per hour, small job")
v("sdc_p_hi", SDC10K.p_step, "{:.2f}", "P(at least one) per hour, large job")
v("sdc_np", SDC10K.exposure * SDC10K.event_rate, "{:.0f}", "expected errors per hour, large job (the e exponent)")
v("sdc_limit", 1 - math.exp(-SDC10K.exposure * SDC10K.event_rate), "{:.2f}", "1 - e^-Np")

print("== ImageNet-V2 (Taori et al. testbed)")
r = IV2.result
alex, r50 = row(IV2.rows, "model", "alexnet"), row(IV2.rows, "model", "resnet50")
v("iv2_models", r.model_count, "{:.0f}", "models")
v("iv2_floor", IV2.original_floor, "{:.0f}", "ImageNet floor, %")
v("iv2_gap_min", r.gap_min, "{:.0f}", "smallest loss, points")
v("iv2_gap_max", r.gap_max, "{:.0f}", "largest loss, points")
v("iv2_gap_median", r.gap_median, "{:.0f}", "median loss, points")
v("iv2_alex_val", alex["val"], "{:.0f}", "AlexNet ImageNet, %")
v("iv2_alex_v2", alex["imagenet_v2"], "{:.0f}", "AlexNet ImageNet-V2, %")
v("iv2_r50_val", r50["val"], "{:.0f}", "ResNet-50 ImageNet, %")
v("iv2_r50_v2", r50["imagenet_v2"], "{:.0f}", "ResNet-50 ImageNet-V2, %")

print("== Mean-shift test (Napkin Math 14.1)")
v("ms_base", MEAN.reference_mean, "{:.2f}", "baseline mean")
v("ms_obs", MEAN.observed_mean, "{:.2f}", "observed mean")
v("ms_shift", MEAN.mean_shift, "{:.2f}", "move in the mean")
v("ms_rel_pct", pct(MEAN.mean_shift / MEAN.reference_mean), "{:.0f}", "move as a share of the mean, %")
v("ms_sd", MEAN.feature_sd, "{:.1f}", "feature spread")
v("ms_n", MEAN.n_samples, "{:,.0f}", "requests")
v("ms_n_code", MEAN.n_samples, "{:.0f}", "requests, as code")
v("ms_se", MEAN.standard_error, "{:.4f}", "standard error")
v("ms_se_ch", MEAN.standard_error, "{:.3f}", "standard error as the chapter rounds it")
v("ms_z", MEAN.z_score, "{:.1f}", "standard errors")
assert MEAN.p_value < MEAN.significance_bound
v("ms_p_bound", MEAN.significance_bound, "{:.3f}", "p-value bound")
v("ms_p_bound_pct", pct(MEAN.significance_bound), "{:.1f}", "the same, %")

print("== Fraud service: country PSI, amount KL and KS (tables 3 and 4)")
v("psi_total", PSI.psi, "{:.3f}", "PSI, user country")
for name in PSI.bins:
    v(f"psi_{name.lower()}", PSI.term(name), "{:.3f}", f"PSI contribution, {name}")
v("psi_other_pct", pct(PSI.term("Other") / PSI.psi), "{:.0f}", "Other's share of the PSI, %")
other_base, other_now = PSI.shares("Other")
v("psi_other_base_pct", pct(other_base), "{:.0f}", "Other's baseline traffic share, %")
v("psi_other_now_pct", pct(other_now), "{:.0f}", "Other's current traffic share, %")
v("kl_amount", KL.kl, "{:.3f}", "KL divergence, amount")
v("kl_trigger", KL.kl_trigger, "{:.2f}", "KL investigate threshold")
v("kl_mu0", KL.reference_mean, "{:.1f}", "reference mean, log-dollars")
v("kl_sd0", KL.reference_sd, "{:.1f}", "reference spread")
v("kl_mu1", KL.current_mean, "{:.1f}", "current mean")
v("kl_sd1", KL.current_sd, "{:.1f}", "current spread")
v("ks_stat", KS.observed_statistic, "{:.3f}", "KS statistic, amount")
v("ks_crit", KS.critical_value, "{:.3f}", "KS critical value")
v("ks_n", KS.n_reference, "{:,.0f}", "samples per side")
v("ks_alpha", KS.significance, "{:.2f}", "Bonferroni-corrected level")

print("== Calibration (Guo et al. 2017, Table 1, ECE in points)")
v("ece_bins", f(STUDY.CalibrationEceBins), "{:.0f}", "bins")
v("ece_lenet", f(STUDY.Cifar100LeNet5EcePct), "{:.2f}", "CIFAR-100 LeNet 5")
v("ece_lenet_ts", f(STUDY.Cifar100LeNet5EceTempScaledPct), "{:.2f}", "the same, temperature scaled")
v("ece_r110", f(STUDY.Cifar100ResNet110EcePct), "{:.2f}", "CIFAR-100 ResNet 110")
v("ece_r110_title", f(STUDY.Cifar100ResNet110EcePct), "{:.1f}", "the same, title")
v("ece_r110_ts", f(STUDY.Cifar100ResNet110EceTempScaledPct), "{:.2f}", "the same, temperature scaled")
v("ece_r152", f(STUDY.ImageNetResNet152EcePct), "{:.2f}", "ImageNet ResNet 152")
v("ece_r152_ts", f(STUDY.ImageNetResNet152EceTempScaledPct), "{:.2f}", "the same, temperature scaled")

print("== Selective answering (illustrative curve and load)")
v("sel_rate", SEL.request_rate.to("1/s").magnitude, "{:,.0f}", "requests/s")
v("sel_cov_pct", pct(SEL.coverage), "{:.0f}", "coverage, %")
v("sel_rest_pct", pct(1 - SEL.coverage), "{:.0f}", "routed share, %")
v("sel_fallback", SEL.fallback_rate.to("1/s").magnitude, "{:,.0f}", "requests/s to fallback")
v("sel_risk_pct", pct(SEL.risk), "{:.1f}", "error at the operating coverage, %")
v("sel_full_pct", pct(SEL.full_risk), "{:.1f}", "error answering all, %")

print("== Conformal rank")
v("conf_n", CONF.calibration_count, "{:,.0f}", "calibration examples")
v("conf_n1", CONF.calibration_count + 1, "{:,.0f}", "ranks a test point can take")
v("conf_alpha", CONF.alpha, "{:.1f}", "miscoverage")
v("conf_target", CONF.target_coverage, "{:.1f}", "target coverage")
v("conf_target_pct", pct(CONF.target_coverage), "{:.0f}", "the same, %")
v("conf_bound", CONF.rank_bound, "{:.1f}", "(n + 1)(1 - alpha)")
v("conf_rank", CONF.rank, "{:.0f}", "cutoff rank")
v("conf_rank_ord", CONF.rank, "{:.0f}st", "the same, ordinal")

print("== Jailbreak transfer (Zou et al. 2023, Table 2)")
v("gcg_behaviors", f(STUDY.GcgHarmfulBehaviors), "{:.0f}", "harmful behaviors")
for key, name in (("only", "BehaviorOnly"), ("sure", "SureHeres"), ("gcg", "Gcg"), ("ens", "GcgEnsemble")):
    v(f"gpt35_{key}", f(getattr(STUDY, f"Gpt35{name}AsrPct")), "{:.1f}", f"GPT-3.5, {name}, %")
    v(f"gpt4_{key}", f(getattr(STUDY, f"Gpt4{name}AsrPct")), "{:.1f}", f"GPT-4, {name}, %")
v("gpt35_ens_title", f(STUDY.Gpt35GcgEnsembleAsrPct), "{:.0f}", "GPT-3.5 ensemble, title, %")

print("== FGSM panda (Goodfellow et al. 2015)")
v("fgsm_eps", f(STUDY.FgsmPandaEpsilon), "{:.3f}", "step size")
v("fgsm_panda_pct", f(STUDY.FgsmPandaConfidencePct), "{:.1f}", "panda confidence, %")
v("fgsm_gibbon_pct", f(STUDY.FgsmGibbonConfidencePct), "{:.1f}", "gibbon confidence, %")

print("== Adversarial training work (Napkin Math 14.2)")
v("pgd_attack", PGD.attack_passes, "{:.0f}", "PGD attack steps")
v("pgd_attack_word", PGD.attack_passes, "words", "the same, in words")
v("pgd_update", PGD.update_passes, "{:.0f}", "update passes")
v("pgd_passes", PGD.passes, "{:.0f}", "passes per batch")
v("pgd_passes_word", PGD.passes, "words", "the same, in words")
v("fgsm_passes_word", FGSM.passes, "words", "FGSM listing passes, in words")

print("== Four cells (Systems Perspective 14.1)")
v("cells_clean", CELLS.standard_clean_accuracy, "{:.0f}", "standard ResNet-50 clean, %")
v("cells_robust", CELLS.robust_attacked_accuracy, "{:.0f}", "robust model under attack, %")
v("cells_gap", round(CELLS.standard_clean_accuracy) - round(CELLS.robust_attacked_accuracy), "{:.0f}",
  "the cross-condition gap, points (of the printed values, as the chapter does)")

print("== RobustBench CIFAR-10, AutoAttack")
first, last = RB.best_first, RB.best_last
chan = row(RB.rows, "model_id", "Chan2020Jacobian")
v("rb_first_year", RB.first_year, "{:.0f}", "first year")
v("rb_last_year", RB.last_year, "{:.0f}", "last year")
v("rb_years_word", RB.last_year - RB.first_year, "words", "years between, in words")
v("rb_first_pct", first["autoattack_acc"], "{:.0f}", "best under attack, first year, %")
clean = sorted(r["clean_acc"] for r in RB.rows)
v("rb_clean_lo", clean[1], "{:.0f}", "clean accuracy, lowest but one, %")
v("rb_clean_hi", clean[-1], "{:.0f}", "clean accuracy, highest, %")
v("rb_clean_min", clean[0], "{:.0f}", "clean accuracy, the one low entry, %")
v("rb_last_pct", last["autoattack_acc"], "{:.0f}", "best under attack, last year, %")
v("rb_last_clean_pct", last["clean_acc"], "{:.0f}", "that model's clean accuracy, %")
v("rb_fail_pct", 100 - last["autoattack_acc"], "{:.0f}", "best still fails on, %")
v("rb_floor_pct", RB.broken_floor, "{:.0f}", "broken floor, %")
v("rb_broken", len(RB.broken), "{:.0f}", "defenses below the floor")
v("rb_broken_word", len(RB.broken), "words", "the same, in words")
v("rb_chan_clean", chan["clean_acc"], "{:.2f}", "Chan2020Jacobian clean, %")
v("rb_chan_aa", chan["autoattack_acc"], "{:.2f}", "Chan2020Jacobian under AutoAttack, %")

print("== Certified radius (Sec. 14.6.1.1)")
v("cert_p", CERT.top_class_probability, "{:.3f}", "top-class probability")
v("cert_sigma", CERT.noise_sd, "{:.1f}", "noise standard deviation")
v("cert_q", CERT.quantile, "{:.2f}", "Phi^-1(p_A)")
v("cert_r", CERT.radius, "{:.1f}", "certified radius")
v("eps_levels", CERT.linf_levels, "{:.0f}", "l-infinity budget, pixel levels")
v("eps_scale", CERT.pixel_levels - 1, "{:.0f}", "largest pixel value")
v("cert_d", CERT.coordinates, "{:,.0f}", "values per image")
v("cert_corner", CERT.box_corner_l2, "{:.2f}", "box corner, l2 length")
cifar = mlsysim.Datasets.CIFAR10
v("cifar_w", cifar.image_width, "{:.0f}", "CIFAR-10 width")
v("cifar_h", cifar.image_height, "{:.0f}", "CIFAR-10 height")
v("cifar_c", cifar.image_channels, "{:.0f}", "CIFAR-10 channels")

print("== BadNets (Gu et al. 2017, Table 4)")
v("bn_clean", f(STUDY.BadNetsBaselineCleanAccuracyPct), "{:.1f}", "honest model, clean, %")
v("bn_back_clean", f(STUDY.BadNetsBackdoorCleanAccuracyPct), "{:.1f}", "backdoored, clean, %")
v("bn_drop", f(STUDY.BadNetsBaselineCleanAccuracyPct) - f(STUDY.BadNetsBackdoorCleanAccuracyPct), "{:.1f}",
  "clean accuracy drop, points")
v("bn_attack", f(STUDY.BadNetsStopAsSpeedLimitPct), "{:.1f}", "stop read as speed limit, %")
v("bn_attack_title", f(STUDY.BadNetsStopAsSpeedLimitPct), "{:.0f}", "the same, title")
v("bn_postit", f(STUDY.BadNetsPostItConfidencePct), "{:.0f}", "Post-it confidence, %")

print("== Bit flip in an FP32 weight (figure 14) and Llama 3 (Lecture 7)")
mant_exp = math.floor(math.log10(FLIP.corrupted))
v("flip_w", FLIP.value, "{:.3f}", "stored weight")
v("flip_bit", FLIP.flipped_bit, "{:.0f}", "flipped bit")
v("flip_scale", FLIP.scale_log2, "{:.0f}", "multiplied by 2 to this")
v("flip_mant", FLIP.corrupted / 10 ** mant_exp, "{:.1f}", "corrupted value, mantissa")
v("flip_exp", mant_exp, "{:.0f}", "corrupted value, power of ten")
v("flip_word", FLIP.word_bits, "{:.0f}", "bits in the word")
v("flip_exp_bits", FLIP.exponent_bits, "{:.0f}", "exponent bits")
v("flip_exp_bits_word", FLIP.exponent_bits, "words", "the same, in words")
v("flip_frac_bits", FLIP.fraction_bits, "{:.0f}", "mantissa bits")
sdc = row(LLAMA3, "component", "Silent Data Corruption")
v("llama_sdc", sdc["interruption_count"], "{:.0f}", "Llama 3 silent data corruptions caught")
v("llama_sdc_word", sdc["interruption_count"], "words", "the same, in words")
v("llama_interrupts", f(Literature.TrainingRuns.Llama3_405B_UnexpectedInterruptions), "{:.0f}",
  "Llama 3 unexpected interruptions")
v("gpt3_params_b", Models.Language.GPT3.parameters.to("count").magnitude / 10 ** 9, "{:.0f}",
  "GPT-3 parameters, billions")

if __name__ == "__main__":
    write(__file__)
