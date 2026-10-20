#!/usr/bin/env python3
"""Every number on the Lecture 11 (Edge Intelligence) slides and notes, read from MLSysIM.

MLSysIM holds the inputs (registry, Literature anchors and series) and the
calculations (Scenarios.*); the chapter reads the same scenarios. This file
only selects values, converts units, and formats them; it types no numbers.
Each value is recorded with v() under the key the deck prints as \\val{key};
running this file writes images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Hardware, Literature, Platforms = mlsysim.Hardware, mlsysim.Literature, mlsysim.Platforms
ReferenceStats, Scenarios, Datasets = mlsysim.ReferenceStats, mlsysim.Scenarios, mlsysim.Datasets
FL = Literature.OnDeviceLearning
SPECTRUM, TIERS = ReferenceStats.EdgeDeviceSpectrum, ReferenceStats.EdgeAdaptationTierProfile
POWER, BATTERY = ReferenceStats.MobilePower, ReferenceStats.PhoneBattery.EnergyWh
STACK = Scenarios.PhoneFullUpdateWorkingSet
JOB = Scenarios.PhoneBackgroundFineTuneBattery
LORA, ADAPTERS = Scenarios.TransformerLayerRank4Update, Scenarios.CompactEncoderResidualAdapters
WALL = Scenarios.CameraModelTenContextPersonalization
UPLINK, UPLOAD = Scenarios.PhoneImagesVersusQuantizedUpdates, Scenarios.FederatedAdapterUploadMobileUplink
TRAIN_SEC = Scenarios.TrainingSecondVersusWakeWordChecks
DEADLINES = Scenarios.ForegroundDeadlinesVersusRoundTrip
COVER, MASK = Scenarios.KeyboardFleetParticipationCoverage, Scenarios.TwoClientPairwiseMask
GATE, SPREAD = Scenarios.AdaptedModelShadowGate, Scenarios.PhoneSocAiScoreSpread
MILLION = 10 ** 6


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


print("== Hook: Gboard (Hard et al. 2018; Bonawitz et al. 2019)")
v("gb_ngram", FL.GboardNgramRecallClientPct, "{:.1f}", "n-gram recall on client caches, %")
v("gb_server", FL.GboardServerRecallClientPct, "{:.1f}", "server CIFG recall on client caches, %")
v("gb_fed", FL.GboardFederatedRecallClientPct, "{:.1f}", "federated CIFG recall on client caches, %")
v("gb_ngram_ci", FL.GboardNgramRecallClientCiPct, "{:.1f}", "n-gram 95% CI, %")
v("gb_server_ci", FL.GboardServerRecallClientCiPct, "{:.1f}", "server 95% CI, %")
v("gb_fed_ci", FL.GboardFederatedRecallClientCiPct, "{:.1f}", "federated 95% CI, %")
v("gb_rel_gain", (FL.GboardFederatedRecallClientPct / FL.GboardServerRecallClientPct - 1) * 100,
  "{:.0f}", "relative gain over the server model, %")
v("gb_fed_logs", FL.GboardFederatedRecallLogsPct, "{:.1f}", "federated recall on server logs, %")
v("gb_server_logs", FL.GboardServerRecallLogsPct, "{:.1f}", "server recall on server logs, %")
v("gb_fed_ctr", FL.GboardFederatedCtrPct, "{:.2f}", "federated CTR, %")
v("gb_server_ctr", FL.GboardServerCtrPct, "{:.2f}", "server CTR, %")
v("gb_params_m", FL.GboardModelParameters / MILLION, "{:.1f}", "model parameters, millions")
v("gb_clients_m", FL.GboardClients / MILLION, "{:.1f}", "phones, millions")
v("gb_sentences_m", FL.GboardSentences / MILLION, "{:.0f}", "sentences, millions")
v("gb_mem_gb", FL.GboardMinAvailableMemoryGb, "{:.0f}", "available memory to join, GB")
v("gb_rounds", FL.GboardRounds, "{:,.0f}", "rounds")
v("gb_days_lo", FL.GboardTrainingDaysLow, "{:.0f}", "training days, low")
v("gb_days_hi", FL.GboardTrainingDaysHigh, "{:.0f}", "training days, high")
v("gb_updates_lo", FL.GboardUpdatesPerRoundLow, "{:.0f}", "updates to close a round, low")
v("gb_updates_hi", FL.GboardUpdatesPerRoundHigh, "{:.0f}", "updates to close a round, high")
v("fl_round_min_lo", FL.FlRoundMinutesLow, "{:.0f}", "minutes per round, low")
v("fl_round_min_hi", FL.FlRoundMinutesHigh, "{:.0f}", "minutes per round, high")
v("fl_slowdown", FL.FlSlowdownVsDatacenter, "{:.0f}", "slower than data-center training, x")
v("fl_daynight", FL.FlDayNightParticipationRatio, "{:.0f}", "day-night participation swing, x")

print("== Phone power, battery, and uplink")
v("power_lo", m(POWER.MobileMlSustainedLow, "W"), "{:.0f}", "sustained power, low, W")
v("power_sus_hi", m(POWER.MobileMlSustainedHigh, "W"), "{:.0f}", "sustained power, high, W")
v("power_burst", m(POWER.MobileMlBurstHigh, "W"), "{:.0f}", "burst ceiling, W")
v("battery_wh", m(BATTERY, "Wh"), "{:.0f}", "phone battery, Wh")
v("burst_hours", ratio(BATTERY, POWER.MobileMlBurstHigh * mlsysim.core.units.hour), "{:.1f}",
  "hours a battery lasts at the burst ceiling")
v("uplink_lo", m(UPLOAD.links["low"], "megabit/second"), "{:.0f}", "mobile uplink, low, Mb/s")
v("uplink_hi", m(UPLOAD.links["high"], "megabit/second"), "{:.0f}", "mobile uplink, high, Mb/s")

print("== Device classes")
v("phone_gb", m(SPECTRUM.FlagshipSmartphoneRamHigh, "GiB"), "{:.0f}", "flagship phone memory, GB")
v("iot_gb", m(SPECTRUM.IotDeviceRamHigh, "GiB"), "{:.0f}", "IoT device memory, up to, GB")
v("mcu_kb", m(SPECTRUM.MicrocontrollerSram, "KiB"), "{:.0f}", "microcontroller SRAM, KB")
v("mem_ratio", ratio(SPECTRUM.FlagshipSmartphoneRamHigh, SPECTRUM.MicrocontrollerSram), "{:,.0f}",
  "phone over microcontroller memory")
img = Datasets.ImageNet
img_bytes = img.image_width * img.image_height * img.image_channels * mlsysim.core.units.BYTES_INT8
v("img_px", img.image_width, "{:.0f}", "image side, pixels")
v("img_kb", m(img_bytes, "kB"), "{:.0f}", "one 224 x 224 x 3 8-bit image, KB")
# The chapter rounds this share to the nearest ten percent (footnote, "roughly").
v("img_sram_pct", round(ratio(img_bytes, SPECTRUM.MicrocontrollerSram) * 10) * 10, "{:.0f}",
  "image share of microcontroller SRAM, % (chapter's rounding)")

print("== Deadlines against the round trip")
for key, name in (("dl_control", "safety control"), ("dl_arvr", "AR/VR motion to photon"),
                  ("dl_camera", "camera frame"), ("dl_voice", "voice response")):
    v(key, m(DEADLINES.deadlines[name], "ms"), "{:.0f}", f"{name} deadline, ms")
v("rtt_lo", m(DEADLINES.round_trip_low, "ms"), "{:.0f}", "network round trip, low, ms")
v("rtt_hi", m(DEADLINES.round_trip_high, "ms"), "{:.0f}", "network round trip, high, ms")

print("== Training working set (10M parameters, FP32, Adam)")
v("stack_params_m", STACK.parameters / MILLION, "{:.0f}", "parameters, millions")
v("stack_weights_mb", m(STACK.weights, "MB"), "{:.0f}", "weights, MB")
v("stack_grads_mb", m(STACK.gradients, "MB"), "{:.0f}", "gradients, MB")
v("stack_adam_mb", m(STACK.optimizer_state, "MB"), "{:.0f}", "two Adam moments, MB")
v("stack_persist_mb", m(STACK.persistent, "MB"), "{:.0f}", "weights + gradients + Adam, MB")
v("stack_acts_mb", m(STACK.activations, "MB"), "{:.0f}", "saved activations, MB")
v("stack_total_mb", m(STACK.total, "MB"), "{:.0f}", "training stack, MB")
v("stack_amp", STACK.amplification, "{:.0f}", "stack over weights, x")
v("stack_amp_words", STACK.amplification, "words", "stack over weights, spelled")
v("stack_floor", STACK.floor, "{:.0f}", "activation-free floor, x")
v("stack_floor_words", STACK.floor, "words", "weight-sized arrays, spelled")
v("stack_ceiling", STACK.ceiling, "{:.0f}", "top of the quoted range, x")
v("stack_act_factor", STACK.activation_factor, "{:.0f}", "activations over weights, x (assumed)")
v("phone_installed_gb", m(Platforms.Mobile.ram, "GiB"), "{:.0f}", "phone memory installed, GB")
v("allow_lo_mb", m(TIERS.KeyboardBackgroundBudgetLow, "MB"), "{:.0f}", "background allowance, low, MB")
v("allow_hi_mb", m(TIERS.KeyboardBackgroundBudgetHigh, "MB"), "{:.0f}", "background allowance, high, MB")
v("kb_update_lo_mb", m(TIERS.KeyboardGradientMemoryLow, "MB"), "{:.0f}", "keyboard update, low, MB")
v("kb_update_hi_mb", m(TIERS.KeyboardGradientMemoryHigh, "MB"), "{:.0f}", "keyboard update, high, MB")
v("kb_update_mb", m(TIERS.KeyboardGradientExample, "MB"), "{:.0f}", "keyboard update, selected point, MB")
v("kb_allow_mb", m(TIERS.KeyboardBackgroundBudgetExample, "MB"), "{:.0f}", "keyboard allowance, selected point, MB")

print("== Bytes per update step")
A17, H100, SD8 = Hardware.Mobile.iPhone15Pro, Hardware.Cloud.H100, Hardware.Mobile.Snapdragon8Gen3
v("step_mb", m(STACK.step_traffic, "MB"), "{:.0f}", "DRAM traffic of one step, MB")
v("step_ratio", STACK.traffic_ratio, "{:.0f}", "step traffic over weight bytes, x")
v("a17_bw", m(A17.memory.bandwidth, "GB/s"), "{:.0f}", "A17 Pro DRAM bandwidth, GB/s")
v("a17_bw1", m(A17.memory.bandwidth, "GB/s"), "{:.1f}", "A17 Pro DRAM bandwidth, GB/s, one decimal")
v("step_ms", m(STACK.step_floor, "ms"), "{:.0f}", "traffic time per step, ms")
v("step_ms1", m(STACK.step_floor, "ms"), "{:.1f}", "traffic time per step, ms, one decimal")
v("h100_bw", m(H100.memory.bandwidth, "GB/s"), "{:,.0f}", "H100 HBM3 bandwidth, GB/s")
v("hbm_over_sd8", ratio(H100.memory.bandwidth, SD8.memory.bandwidth), "{:.0f}", "HBM3 over Snapdragon 8 Gen 3")
v("hbm_over_a17", ratio(H100.memory.bandwidth, A17.memory.bandwidth), "{:.0f}", "HBM3 over A17 Pro")
v("hbm_over_sd8_1", ratio(H100.memory.bandwidth, SD8.memory.bandwidth), "{:.1f}", "same, one decimal")
v("hbm_over_a17_1", ratio(H100.memory.bandwidth, A17.memory.bandwidth), "{:.1f}", "same, one decimal")
v("weight_read_pct", STACK.weight_read_share * 100, "{:.0f}", "three weight reads, share of step traffic, %")
ane = {r["chip"]: r for r in Literature.Series.AppleNeuralEngine.entries}
v("ane_a11", ane["A11 Bionic"]["tops"], "{:.1f}", "A11 stated rating, TOPS")
v("ane_a11_year", ane["A11 Bionic"]["year"], "{:.0f}", "A11 year")
v("ane_a17", ane["A17 Pro"]["tops"], "{:.0f}", "A17 Pro stated rating, TOPS")
v("ane_a17_year", ane["A17 Pro"]["year"], "{:.0f}", "A17 Pro year")

print("== Budget terms and personalization")
v("lora_dim", LORA.in_dim, "{:.0f}", "layer width")
v("lora_rank", LORA.rank, "{:.0f}", "LoRA rank")
v("lora_full", LORA.full_params, "{:,.0f}", "full layer parameters")
v("lora_params", LORA.update_params, "{:,.0f}", "rank-4 update parameters")
v("lora_cut_pct", round(LORA.reduction * 100), "{:.0f}", "parameters removed, % (chapter's rounding)")
v("tinytl_saving", FL.TinyTlMemorySavingVsFull, "{:.1f}", "TinyTL memory saving, up to, x")
v("wall_params_m", m(WALL.model_params, "Mparam"), "{:.0f}", "camera model parameters, millions")
v("wall_model_mb", m(WALL.model_size, "MB"), "{:.0f}", "one model, MB")
v("wall_contexts", WALL.contexts, "{:.0f}", "contexts")
v("wall_copies_mb", m(WALL.total_copies, "MB"), "{:.0f}", "ten copies, MB")
v("wall_adapter_kb", m(WALL.adapter_size, "kB"), "{:.0f}", "one adapter, KB")
v("wall_shared_mb", m(WALL.total_shared, "MB"), "{:.0f}", "backbone + adapters, MB")
v("wall_total_saving", WALL.total_savings, "{:.1f}", "total saving, x")
v("wall_context_saving", WALL.per_context_savings, "{:.0f}", "per-context saving, x")
v("adapter_width", ADAPTERS.width, "{:.0f}", "adapter width d")
v("adapter_bottleneck", ADAPTERS.bottleneck, "{:.0f}", "adapter bottleneck r")
v("adapter_params", ADAPTERS.params_per_adapter, "{:,.0f}", "parameters per adapter")
v("adapter_kb", m(ADAPTERS.adapter_bytes, "kB"), "{:.1f}", "one adapter in FP16, KB")

print("== Battery (one background update)")
v("job_w", m(JOB.power_draw, "W"), "{:.1f}", "fine-tune draw, W")
v("job_min", m(JOB.required_runtime, "minute"), "{:.0f}", "runtime, minutes")
v("job_h", m(JOB.required_runtime, "hour"), "{:.1f}", "runtime, hours")
v("job_wh", m(JOB.required_energy, "Wh"), "{:.2f}", "energy, Wh")
v("job_pct", JOB.budget_ratio * 100, "{:.0f}", "battery share, %")

print("== A training second against wake-word checks")
kws = {d["name"].split("\n")[0] + " " + d["name"].split("\n")[1]: d["energy_uj"]
       for d in Literature.EdgeInferenceBenchmarks.list}
v("kws_lo_uj", m(TRAIN_SEC.inference_energy_low, "microjoule"), "{:.0f}", "Syntiant Core 2 KWS, uJ")
v("kws_lo_uj1", m(TRAIN_SEC.inference_energy_low, "microjoule"), "{:.1f}", "same, one decimal")
v("kws_hi_uj", m(TRAIN_SEC.inference_energy_high, "microjoule"), "{:.0f}", "STM32N6 KWS, uJ")
v("kws_hi_uj1", m(TRAIN_SEC.inference_energy_high, "microjoule"), "{:.1f}", "same, one decimal")
v("stm_ic_uj", kws["STM32N6 NPU (IC, high-perf)"], "{:.0f}", "STM32N6 image classification, uJ")
v("coral_mj", m(mlsysim.core.units.Q_(kws["Google Coral Edge TPU"], "microjoule"), "mJ"), "{:.1f}", "Google Coral, mJ")
v("orin_mj", m(mlsysim.core.units.Q_(kws["Jetson AGX Orin (ResNet-50)"], "microjoule"), "mJ"), "{:.0f}", "Jetson AGX Orin ResNet-50, mJ")
v("train_j_lo", m(TRAIN_SEC.training_energy_low, "J"), "{:.0f}", "one training second, low, J")
v("train_j_hi", m(TRAIN_SEC.training_energy_high, "J"), "{:.0f}", "one training second, high, J")
v("checks_lo", TRAIN_SEC.checks_low, "{:,.0f}", "checks one training second buys, low")
v("check_rate", m(TRAIN_SEC.check_rate, "1/s"), "{:.0f}", "detector checks per second (assumed)")
v("detector_mw", m(TRAIN_SEC.detector_power, "mW"), "{:.2f}", "detector power, mW")
v("detector_min_lo", m(TRAIN_SEC.time_to_training_second_low, "minute"), "{:.0f}", "minutes to one training second, low")
v("detector_min_hi", m(TRAIN_SEC.time_to_training_second_high, "minute"), "{:.0f}", "minutes to one training second, high")

print("== The round")
v("overselect", FL.FlOverSelectionPct / 100, "{:.1f}", "selected over target")
v("overselect_pct", FL.FlOverSelectionPct, "{:.0f}", "selected over target, %")
v("dropout_lo", FL.FlDropoutPctLow, "{:.0f}", "dropout, low, %")
v("dropout_hi", FL.FlDropoutPctHigh, "{:.0f}", "dropout, high, %")
v("secagg_rounds_words", FL.FlSecAggProtocolRounds, "words", "Secure Aggregation message rounds, spelled")

print("== FedAvg rounds (McMahan et al. 2017, Table 2)")
rows = Literature.Series.FedAvgMnistCnnRounds.entries
sgd, best = rows[0], max(rows, key=lambda r: r["updates_per_round"])
e5 = max(rows[1:], key=lambda r: r["non_iid_rounds"])  # the full-batch E = 5 row, worst non-IID
helps = [sgd["non_iid_rounds"] / r["non_iid_rounds"] for r in rows[1:] if r["non_iid_rounds"] < sgd["non_iid_rounds"]]
v("fedsgd_iid", sgd["iid_rounds"], "{:.0f}", "FedSGD rounds, IID")
v("fedsgd_noniid", sgd["non_iid_rounds"], "{:.0f}", "FedSGD rounds, non-IID")
v("fedavg_best_u", best["updates_per_round"], "{:,.0f}", "most local updates per round")
v("fedavg_best_iid", best["iid_rounds"], "{:.0f}", "rounds at that u, IID")
v("fedavg_best_noniid", best["non_iid_rounds"], "{:.0f}", "rounds at that u, non-IID")
v("fedavg_iid_gain", sgd["iid_rounds"] / best["iid_rounds"], "{:.1f}", "IID reduction, x")
v("fedavg_iid_gain0", sgd["iid_rounds"] / best["iid_rounds"], "{:.0f}", "IID reduction, x, whole")
v("fedavg_e5_epochs", e5["local_epochs"], "{:.0f}", "local epochs of the full-batch row")
v("fedavg_e5_noniid", e5["non_iid_rounds"], "{:,.0f}", "non-IID rounds, E = 5, full batch")
v("fedavg_help_lo", min(helps), "{:.1f}", "smallest non-IID reduction among rows that help, x")
v("fedavg_help_hi", max(helps), "{:.1f}", "largest non-IID reduction, x")

print("== Uplink (raw images against updates)")
v("raw_mb", m(UPLINK.raw_volume, "MB"), "{:.0f}", "raw images a week, MB")
v("upd_params_m", UPLINK.update_params / MILLION, "{:.0f}", "update parameters, millions")
v("upd_bits", UPLINK.update_bits_per_param, "{:.0f}", "bits per update parameter")
v("upd_mb", m(UPLINK.update_size, "MB"), "{:.1f}", "one update, MB")
v("upd_ratio", UPLINK.reduction, "{:.0f}", "raw week over one update, x")
v("break_even", UPLINK.break_even_rounds, "{:.0f}", "break-even rounds")
v("daily_rounds", UPLINK.rounds_per_week["daily"], "{:.0f}", "daily cadence, rounds a week")
v("daily_mb", m(UPLINK.weekly_upload("daily"), "MB"), "{:.1f}", "daily cadence upload, MB a week")
v("hourly_rounds", UPLINK.rounds_per_week["hourly"], "{:.0f}", "hourly cadence, rounds a week")
v("hourly_mb", m(UPLINK.weekly_upload("hourly"), "MB"), "{:.0f}", "hourly cadence upload, MB a week")
v("adapter_upd_mb", m(UPLOAD.payload, "MiB"), "{:.0f}", "reference adapter update, MB")
# The chapter rounds the slow upload to whole minutes (EdgeRoundUploadTime).
v("adapter_upload_min", round(m(UPLOAD.transfer_time("low"), "minute")), "{:.0f}",
  "adapter upload at the low uplink, minutes (chapter's rounding)")

print("== Secure aggregation cells (illustrative)")
v("mask_a0", MASK.upload_a[0], "{:+.1f}", "A's first uploaded cell")
for i, x in enumerate(MASK.total):
    v(f"mask_sum{i}", x, "{:+.1f}", f"sum cell {i}")

print("== Sessions (Bonawitz et al. 2019, Table 1)")
v("sess_ok", FL.FlSessionsAccepted, "{:,.0f}", "accepted sessions")
v("sess_late", FL.FlSessionsRejectedLate, "{:,.0f}", "rejected-late sessions")
v("sess_intr", FL.FlSessionsInterrupted, "{:,.0f}", "interrupted sessions")
v("sess_ok_pct", FL.FlSessionsAcceptedPct, "{:.0f}", "accepted, % (stated)")
v("sess_late_pct", FL.FlSessionsRejectedLatePct, "{:.0f}", "rejected late, % (stated)")
v("sess_intr_pct", FL.FlSessionsInterruptedPct, "{:.0f}", "interrupted, % (stated)")

print("== Phone processors (AI Benchmark)")
lo, hi = SPREAD.lowest("fleet"), SPREAD.highest("fleet")
glo, ghi = SPREAD.lowest("generation"), SPREAD.highest("generation")
v("ai_first_year", SPREAD.windows["fleet"][0], "{:.0f}", "first release year a fleet still holds")
v("ai_spread", SPREAD.spread("fleet"), "{:.0f}", "AI score spread, 2020 on, x")
v("ai_spread_round", round(SPREAD.spread("fleet"), -2), "{:.0f}", "same, chapter's rounding")
v("ai_gen_spread", SPREAD.spread("generation"), "{:.0f}", "spread within one generation, x")
v("ai_lo_name", lo["processor"], "{}", "lowest-scoring processor, 2020 on")
v("ai_hi_name", hi["processor"], "{}", "highest-scoring processor")
v("ai_gen_lo_name", glo["processor"].replace("Google ", ""), "{}", "lowest in one generation")
v("ai_gen_hi_name", ghi["processor"], "{}", "highest in one generation")

print("== Coverage (illustrative counts per 100 deployed)")
for cohort, key in (("flagship", "flag"), ("mid-tier", "mid"), ("budget", "budget")):
    v(f"cov_{key}_deployed", COVER.counts[cohort][0], "{:.0f}", f"{cohort} deployed")
    v(f"cov_{key}_ratio", COVER.coverage_ratio(cohort), "{:.2f}", f"{cohort} coverage ratio")
v("cov_admitted", COVER.total("admitted"), "{:.0f}", "admitted phones")
v("cov_completed", COVER.total("completed"), "{:.0f}", "completed updates")
v("cov_budget_done", COVER.counts["budget"][-1], "{:.0f}", "budget completed updates")
v("cov_budget_fleet_pct", COVER.share("budget", "deployed") * 100, "{:.0f}", "budget share of the fleet, %")
v("cov_budget_done_pct", COVER.share("budget", "completed") * 100, "{:.0f}", "budget share of completed, %")

print("== Release gate (illustrative letters)")
v("gate_dis", GATE.disagreements, "{:.0f}", "disagreements")
v("gate_n", GATE.inputs, "{:.0f}", "inputs")
v("gate_pct", GATE.disagreement_rate * 100, "{:.0f}", "disagreement, %")
v("gate_tau_pct", GATE.tolerance * 100, "{:.0f}", "tolerance, %")
v("rank_flag", TIERS.FlagshipLoraRank, "{:.0f}", "flagship LoRA rank")
v("rank_mid", TIERS.MidTierLoraRank, "{:.0f}", "mid-tier LoRA rank")
v("fleet_devices_m", TIERS.VoiceAssistantFleetDevices / MILLION, "{:.0f}", "voice-assistant fleet, millions")

if __name__ == "__main__":
    write(__file__)
