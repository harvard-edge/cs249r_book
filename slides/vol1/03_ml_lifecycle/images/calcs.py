#!/usr/bin/env python3
"""Every number on the Lecture 3 (ML Lifecycle) slides and notes, read from MLSysIM.

MLSysIM holds the inputs (Literature.ScreeningStudies, Literature.Surveys,
Literature.PredictionQuality, ReferenceStats, the registry) and the
calculations (the retinopathy-screener Scenarios.*); the chapter reads the same
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

Models, Literature, Scenarios = mlsysim.Models, mlsysim.Literature, mlsysim.Scenarios
ReferenceStats = mlsysim.ReferenceStats
STUDY = Literature.ScreeningStudies
SURVEYS = Literature.Surveys
TORCHVISION = ReferenceStats.TorchvisionParameterCounts

REQ = Scenarios.RetinopathyScreenerRequirements
LATE = Scenarios.RetinopathyScreenerLateMemoryLimit
ITER = Scenarios.SlowVersusFastIteration
TRAP = Scenarios.AlwaysNormalScreener
YIELD = Scenarios.ThaiClinicRetinalPhotoYield
FDA = Scenarios.IdxDrPivotalTrial
FDA_IMAGEABILITY = Scenarios.IdxDrPivotalImageability
PHOTO = Scenarios.FundusPhotoAt224
UPLOAD = Scenarios.RuralClinicRetinalUpload
COST = Scenarios.RetinopathyScreeningCloudVersusEdge
CADENCE = Scenarios.RetinopathyMonitoringCadences

# CrowdFlower 2016 time sinks, in the order the chapter's figure draws them.
SURVEY = [("Cleaning and organizing data", SURVEYS.DataScientistCleaning),
          ("Collecting datasets", SURVEYS.DataScientistCollecting),
          ("Mining data for patterns", SURVEYS.DataScientistMining),
          ("Other", SURVEYS.DataScientistOther),
          ("Refining algorithms", SURVEYS.DataScientistRefiningAlgorithms),
          ("Building training sets", SURVEYS.DataScientistBuildingTrainingSets)]
DATA_WORK = (SURVEYS.DataScientistCleaning, SURVEYS.DataScientistCollecting)

# torchvision run shown on the memory slide: (label, parameter count)
TORCHVISION_RUN = [("mobilenet_v2", TORCHVISION.MobileNetV2), ("resnet50", TORCHVISION.ResNet50),
                   ("vgg16", TORCHVISION.VGG16)]


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def pct(x):
    """A fraction as percent."""
    return 100 * x


print("== Hook (Gulshan et al. 2016; Beede et al. 2020, via Literature)")
v("gul_dev_images", round(STUDY.GulshanDevelopmentImages, -3), "{:,.0f}",
  "development-set images, to the thousand")
v("gul_sens", STUDY.GulshanHighSensitivityPointSensitivity, "{:.1f}", "high-sensitivity point: sensitivity, %")
v("gul_spec", STUDY.GulshanHighSensitivityPointSpecificity, "{:.1f}", "high-sensitivity point: specificity, %")
v("gul_hispec_sens", STUDY.GulshanHighSpecificityPointSensitivity, "{:.1f}",
  "high-specificity point: sensitivity, %")
v("gul_hispec_spec", STUDY.GulshanHighSpecificityPointSpecificity, "{:.1f}",
  "high-specificity point: specificity, %")
v("gul_eyepacs", STUDY.GulshanEyePacs1Images, "{:,.0f}", "EyePACS-1 images behind the curve")
v("gul_graders", STUDY.GulshanEyePacs1Ophthalmologists, "{:.0f}", "ophthalmologists plotted")
v("gul_graders_word", STUDY.GulshanEyePacs1Ophthalmologists, "words", "the same, in prose")
v("gul_auc", STUDY.GulshanEyePacs1Auc, "{:.3f}", "EyePACS-1 AUC")
v("beede_clinics", STUDY.BeedeClinics, "{:.0f}", "clinics observed")
v("beede_months_word", STUDY.BeedeStudyMonths, "words", "months of observation, in prose")
v("beede_submitted", YIELD.attempted, "{:,.0f}", "photos submitted")
v("beede_ungradable", YIELD.rejected, "{:,.0f}", "photos rejected as ungradable")
v("ungradable_pct", pct(YIELD.rejected_share), "{:.0f}", "ungradable share, %")

print("== Day-153 failure (late-constraint scenario)")
v("model_gb", m(LATE.model_memory, "GB"), "{:.0f}", "memory the model needs, GB")
v("budget_mb", m(REQ.memory_budget, "MB"), "{:.0f}", "reservation for the model, MB")
v("device_mem_lo", m(LATE.device_memory_low, "GB"), "{:.0f}", "clinic device memory, low, GB")
v("device_mem_hi", m(LATE.device_memory_high, "GB"), "{:.0f}", "clinic device memory, high, GB")
v("day_request", LATE.request_day, "{:.0f}", "day of the request")
(day_a, sens_a), (day_b, sens_b) = sorted(LATE.validation_sensitivity.items())
v("day_val_a", day_a, "{:.0f}", "first validation milestone, day")
v("day_val_b", day_b, "{:.0f}", "second validation milestone, day")
v("val_a_pct", pct(sens_a), "{:.0f}", "sensitivity at the first milestone, %")
v("val_b_pct", pct(sens_b), "{:.0f}", "sensitivity at the second milestone, %")
v("day_handoff", LATE.handoff_day, "{:.0f}", "handed to deployment, day")
v("day_found", LATE.discovery_day, "{:.0f}", "memory need reported, day")
v("day_check", LATE.check_day, "{:.0f}", "device checked, day")
v("day_discard", LATE.discard_day, "{:.0f}", "work discarded, day")
v("months_lost_word", round(m(LATE.work_discarded, "month")), "words", "months of work discarded, in prose")
v("overrun_x", LATE.overrun_ratio, "{:.0f}", "model memory / reservation")
v("overrun_word", LATE.overrun_ratio, "words", "the same, in prose")

print("== Memory: torchvision run (ReferenceStats) at FP32")
v("fp32_bytes", m(mlsysim.BYTES_FP32 * mlsysim.core.units.param, "byte"), "{:.0f}", "bytes per FP32 weight")
v("fp32_bytes_word", m(mlsysim.BYTES_FP32 * mlsysim.core.units.param, "byte"), "words", "the same, in prose")
for label, n in TORCHVISION_RUN:
    v(f"tv_{label}", m(n, "param"), "{:,.0f}", f"torchvision {label} parameters")
    v(f"mb_{label}", m(n * mlsysim.BYTES_FP32, "MB"), "{:.0f}", f"{label} FP32 weights, MB")
v("mnv2_mops", m(Models.Vision.MobileNetV2.inference_flops, "MFLOP"), "{:.0f}",
  "MobileNetV2 operations per inference, millions")

print("== Repair schedule (late-constraint scenario)")
REPAIR = [(name, m(d, "day")) for name, d in LATE.repair_tasks.items()]
for key, (_, days) in zip(("rep_prep", "rep_train", "rep_val", "rep_pack"), REPAIR):
    v(key, days, "{:.0f}", f"repair task, days")
v("rep_total", m(LATE.repair_elapsed, "day"), "{:.0f}", "days in sequence")
v("rep_total_word", m(LATE.repair_elapsed, "day"), "words", "the same, in prose")
v("rep_tasks_word", len(REPAIR), "words", "dependent artifacts redone, in prose")

print("== Time sinks (CrowdFlower 2016 via Literature)")
v("sv_cleaning", SURVEYS.DataScientistCleaning, "{:.0f}", "cleaning and organizing, %")
v("sv_collecting", SURVEYS.DataScientistCollecting, "{:.0f}", "collecting datasets, %")
v("sv_data_work", sum(DATA_WORK), "{:.0f}", "data work together, %")
v("sv_algorithms", SURVEYS.DataScientistRefiningAlgorithms, "{:.0f}", "refining algorithms, %")

print("== Iteration (slow-versus-fast scenario)")
v("it_weeks", m(ITER.window, "week"), "{:.0f}", "development window, weeks")
v("it_months_word", round(m(ITER.window, "month")), "words", "the same window, months, in prose")
v("it_slow_cycle", m(ITER.slow_cycle, "week"), "{:.0f}", "large run, weeks")
v("it_slow_cycle_word", m(ITER.slow_cycle, "week"), "words", "the same, in prose")
v("it_fast_cycle", m(ITER.fast_cycle, "hour"), "{:.0f}", "small run, hours")
v("it_fast_cycle_word", m(ITER.fast_cycle, "hour"), "words", "the same, in prose")
v("it_slow_start", ITER.slow_start, "{:.0f}", "large start, %")
v("it_slow_gain", ITER.slow_gain, "{:.2f}", "large gain per run, points")
v("it_fast_start", ITER.fast_start, "{:.0f}", "small start, %")
v("it_fast_gain", ITER.fast_gain, "{:.1f}", "small gain per run, points")
v("it_ceiling", ITER.ceiling, "{:.0f}", "ceiling, %")
v("it_slow_runs", ITER.slow_runs, "{:.0f}", "large runs")
v("it_slow_final", ITER.slow_final, "{:.1f}", "large final accuracy, %")
v("it_hours_week", m(1 * mlsysim.core.units.week, "hour"), "{:.0f}", "hours in a week")
v("it_fast_possible", ITER.fast_possible_runs, "{:,.0f}", "small possible runs")
v("it_fast_runs", ITER.fast_effective_runs, "{:.0f}", "small runs a team learns from")
v("it_fast_uncapped", ITER.fast_uncapped, "{:.0f}", "small uncapped accuracy, %")
v("it_fast_final", ITER.fast_final, "{:.0f}", "small capped accuracy, %")
v("it_ideas_x", ITER.ideas_ratio, "{:.1f}", "ideas tested, fast / slow")
v("it_gap", ITER.final_gap, "{:.1f}", "gap between the final accuracies, points")

print("== Requirements and the release gate (requirements scenario)")
v("floor_sens", pct(REQ.sensitivity_floor), "{:.0f}", "sensitivity floor, %")
v("floor_spec", pct(REQ.specificity_floor), "{:.0f}", "specificity floor, %")
v("power_lo", m(REQ.power_low, "W"), "{:.0f}", "clinic device power, low, W")
v("power_hi", m(REQ.power_high, "W"), "{:.0f}", "clinic device power, high, W")
v("gate_all_pct", pct(REQ.slice_sensitivity["all clinic photos"]), "{:.0f}", "sensitivity, all clinic photos, %")
v("gate_cataract_pct", pct(REQ.slice_sensitivity["cataract patients"]), "{:.0f}",
  "sensitivity, cataract patients, %")

print("== Accuracy trap (always-normal scenario)")
v("trap_pop", TRAP.population, "{:.0f}", "patients")
v("trap_sick", TRAP.sick, "{:.0f}", "patients with disease")
v("trap_healthy", TRAP.healthy, "{:.0f}", "healthy patients")
v("trap_right", TRAP.true_positives + TRAP.true_negatives, "{:.0f}", "patients the model is right about")
v("trap_acc", pct(TRAP.accuracy), "{:.0f}", "accuracy, %")
v("trap_sens", pct(TRAP.sensitivity), "{:.0f}", "sensitivity, %")
v("trap_missed", TRAP.missed, "{:.0f}", "cases missed")

print("== Amazon recruiting tool (Reuters 2018 via Literature)")
AMAZON = Literature.Fairness
v("amazon_years", AMAZON.AmazonRecruitingHistoryYears, "{:.0f}", "years of past hiring decisions")
v("amazon_years_word", AMAZON.AmazonRecruitingHistoryYears, "words", "the same, in prose")
v("amazon_colleges_word", AMAZON.AmazonRecruitingWomensCollegesDowngraded, "words",
  "all-women's colleges downgraded, in prose")

print("== Stage 3: 224 pixels (resize scenario)")
v("px_src_h", PHOTO.source_height, "{:.0f}", "photo height, px")
v("px_src_w", round(PHOTO.source_width, -3), "{:,.0f}", "photo width, px, to the thousand")
v("px_target", PHOTO.target_size, "{:.0f}", "ImageNet input side, px")
v("px_scale", PHOTO.scale, "{:.2f}", "scale factor")
v("lesion_lo", min(PHOTO.feature_sizes), "{:.0f}", "exudate width as captured, low, px")
v("lesion_hi", max(PHOTO.feature_sizes), "{:.0f}", "exudate width as captured, high, px")
v("lesion_mid", PHOTO.feature_mid, "{:.0f}", "midpoint exudate width, px")
v("lesion_lo_224", min(PHOTO.resized_features), "{:.0f}", "exudate width at 224, low, px")
v("lesion_hi_224", max(PHOTO.resized_features), "{:.0f}", "exudate width at 224, high, px")
v("lesion_mid_224", PHOTO.resized(PHOTO.feature_mid), "{:.0f}", "midpoint exudate width at 224, px")

print("== FDA De Novo DEN180001, IDx-DR (Literature; screening-counts scenarios)")
v("fda_enrolled", STUDY.IdxDrPivotalEnrolled, "{:.0f}", "patients enrolled")
v("fda_sites", STUDY.IdxDrPivotalSites, "{:.0f}", "primary care sites")
v("fda_floor_sens", STUDY.IdxDrSensitivityEndpoint, "{:.1f}", "prespecified sensitivity, %")
v("fda_floor_spec", STUDY.IdxDrSpecificityEndpoint, "{:.1f}", "prespecified specificity, %")
v("fda_tp", FDA.true_positives, "{:.0f}", "sick patients detected")
v("fda_sick", FDA.sick, "{:.0f}", "sick patients")
v("fda_tn", FDA.true_negatives, "{:.0f}", "healthy patients cleared")
v("fda_healthy", FDA.healthy, "{:.0f}", "healthy patients")
v("fda_sens", pct(FDA.sensitivity), "{:.1f}", "observed sensitivity, %")
v("fda_spec", pct(FDA.specificity), "{:.1f}", "observed specificity, %")
v("fda_sens_r", pct(FDA.sensitivity), "{:.0f}", "the same, as the summary rounds it")
v("fda_spec_r", pct(FDA.specificity), "{:.0f}", "the same, as the summary rounds it")
v("fda_image_r", pct(FDA_IMAGEABILITY.usable_share), "{:.0f}", "imageability, %")

print("== WILDS Camelyon17 (Koh et al. 2021 via Literature)")
wid, wood = Literature.PredictionQuality.WildsCamelyon17IdAccuracy, Literature.PredictionQuality.WildsCamelyon17OodAccuracy
v("wilds_id", wid, "{:.1f}", "training hospitals, %")
v("wilds_ood", wood, "{:.1f}", "hospital 5, %")
v("wilds_drop", wid - wood, "{:.1f}", "drop, points")

print("== Clinic upload (daily-upload scenario)")
v("up_patients", UPLOAD.cases_per_day, "{:.0f}", "patients a day")
v("up_photos", UPLOAD.files_per_case, "{:.0f}", "photos per patient")
v("up_photo_mb", m(UPLOAD.file_size, "MB"), "{:.0f}", "MB per photo")
v("up_uplink_mbit", m(UPLOAD.uplink, "Mbit/s"), "{:.0f}", "uplink, Mb/s")
v("up_uplink_mbs", m(UPLOAD.uplink, "MB/s"), "{:.2f}", "uplink, MB/s")
v("up_shift_h", m(UPLOAD.window, "hour"), "{:.0f}", "shift, h")
v("up_daily_mb", m(UPLOAD.daily_volume, "MB"), "{:,.0f}", "daily volume, MB")
v("up_daily_gb", m(UPLOAD.daily_volume, "GB"), "{:.0f}", "daily volume, GB")
v("up_secs", m(UPLOAD.upload_time, "s"), "{:,.0f}", "upload time, s")
v("up_hours", m(UPLOAD.upload_time, "hour"), "{:.1f}", "upload time, h")
v("up_share", pct(UPLOAD.window_share), "{:.0f}", "share of the shift, %")
v("up_share_1", pct(UPLOAD.window_share), "{:.1f}", "the same, one decimal")
v("up_photo_s", m(UPLOAD.file_upload_time, "s"), "{:.0f}", "one photo, s")
v("up_summary_kb", m(UPLOAD.summary_size, "KB"), "{:.0f}", "summary per patient, KB")
v("up_reduction", UPLOAD.reduction, "{:,.0f}", "traffic reduction")

print("== Cloud against edge (cost scenario)")
v("cost_clinics", COST.sites, "{:.0f}", "clinics")
v("cost_patients", COST.cases_per_site_day, "{:.0f}", "patients per clinic per day")
v("cost_images_word", COST.images_per_case, "words", "images per patient, in prose")
v("cost_cloud_img", m(COST.cloud_cost_per_image, "dollar"), "{:.2f}", "cloud per image, $")
v("cost_cloud_inf", m(COST.cloud_inference_annual, "dollar/year"), "{:,.0f}", "cloud inference, $/year")
v("cost_cloud_net", m(COST.cloud_fixed_annual, "dollar/year"), "{:,.0f}", "connectivity, $/year")
v("cost_cloud_year", m(COST.cloud_annual, "dollar/year"), "{:,.0f}", "cloud per year, $")
v("cost_device", m(COST.edge_device_cost, "dollar"), "{:.0f}", "device per clinic, $")
v("cost_capex", m(COST.edge_capex, "dollar"), "{:,.0f}", "edge up front, $")
v("cost_maint", m(COST.edge_maintenance_annual, "dollar/year"), "{:,.0f}", "edge maintenance, $/year")
v("cost_edge_inf", m(COST.edge_inference_annual, "dollar/year"), "{:,.0f}", "edge inference, $/year")
v("cost_savings", m(COST.annual_savings, "dollar/year"), "{:,.0f}", "savings, $/year")
v("cost_payback", m(COST.payback, "year"), "{:.1f}", "payback, years")
v("cost_cloud_k", m(COST.cloud_annual, "kilodollar/year"), "{:.0f}", "cloud per year, $k (figure)")
v("cost_capex_k", m(COST.edge_capex, "kilodollar"), "{:.0f}", "edge up front, $k (figure)")
v("cost_edge_k", m(COST.edge_annual, "kilodollar/year"), "{:.0f}", "edge per year, $k (figure)")

print("== Monitoring cadences (cadence scenario)")
v("cad_orders_word", round(CADENCE.span_orders), "words", "orders of magnitude, minute to quarter, in prose")

if __name__ == "__main__":
    write(__file__)
