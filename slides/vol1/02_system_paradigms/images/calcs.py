#!/usr/bin/env python3
"""Every number on the Lecture 2 (System Paradigms) slides and notes, read from MLSysIM.

MLSysIM holds the inputs (registry, Literature) and the calculations
(Scenarios.*); the chapter reads the same scenarios. This file only selects
values, converts units, and formats them; it types no numbers. Each value is
recorded with v() under the key the deck prints as \\val{key}; running this
file writes images/numbers.tex. make_figs.py imports this module, so figures
and slides share one computation.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402

Hardware, Models, Systems = mlsysim.Hardware, mlsysim.Models, mlsysim.Systems
Platforms, Literature, Scenarios = mlsysim.Platforms, mlsysim.Literature, mlsysim.Scenarios
ReferenceStats = mlsysim.ReferenceStats
LIT = Literature.DeviceMeasurements
POD = Systems.Pods.TPUv4_4096
TPU4, SPARK, ESP = Hardware.Cloud.TPUv4, Hardware.Workstation.DGX_Spark, Hardware.Tiny.ESP32_S3
A100, TPU1 = Hardware.Cloud.A100, Hardware.Cloud.TPUv1
PHONES = (Hardware.Mobile.Pixel8, Hardware.Mobile.iPhone15Pro, Hardware.Mobile.Snapdragon8Gen3)

ARM = Scenarios.RoboticArmRegionalCloud
COAST = Scenarios.CrossCountryFiberPath
ROUTES = Scenarios.AwsUsEast1Routes.routes          # label -> propagation scenario
CAMS = Scenarios.FactoryCameras1080p
DRONE = Scenarios.Drone4KObstacleOffload
RESNET, RESNET_NPU = Scenarios.ResNet50A100Batch1, Scenarios.ResNet50PhoneNpuBatch1
SELF_CHECK = Scenarios.A100AtTenFlopPerByte
DECODE = Scenarios.Llama3_70B_DecodeH100.variant(hardware=A100)
TPU_ROOF = Scenarios.TpuV1ProductionRoofline
BATTERY = Scenarios.AlwaysOnObjectDetectorPhone
THERMAL = Scenarios.UnoptimizedVisionModelPhone
ENERGY = Scenarios.EnergyPerInferenceTiers
UPLOAD = Scenarios.RadioUploadVsMobileNetV2
DOORBELL, CHIME = Scenarios.SmartDoorbell, Scenarios.SmartDoorbellResponsiveChime
FRAME = Scenarios.SmartDoorbellFrame
BRAKING = Scenarios.EmergencyBraking100kmh
VOICE = Scenarios.VoiceAssistantBillionDevices
CAMERA = Scenarios.PhoneCameraPipeline
TRENDS = Scenarios.MicroprocessorTrendsAroundDennard
WALL_FITS = {"flops": Scenarios.MemoryWallFlopsTrend, "dram": Scenarios.MemoryWallDramTrend,
             "interconnect": Scenarios.MemoryWallInterconnectTrend}


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


def mobile_tdp_w():
    """The mobile tier's sustained-power range (low, high) in watts."""
    lo, hi = Platforms.Mobile.tdp_range_w.split("-")
    return float(lo), float(hi)


def mobile_ram_gb():
    lo, hi = Platforms.Mobile.ram_range.split("-")
    return float(lo), float(hi)


def tiny_kws():
    """MLPerf Tiny v1.2 keyword-spotting energy rows."""
    return load_series("slides/vol1/02_system_paradigms/images/data/mlperf_tiny_v1_2_kws_energy.csv").entries


def wall_points():
    """{series: [(year, normalized)]} from the memory-wall Literature series."""
    return {k: s.points for k, s in WALL_FITS.items()}


print("== Tiers: power ladder and memory")
v("pod_mw", m(POD.power, "MW"), "{:.0f}", "TPU v4 pod power, MW")
v("pod_chips", POD.chips, "{:,.0f}", "chips in the pod")
v("pod_mem_tb", m(POD.memory, "TiB"), "{:.0f}", "pod HBM, labeled TB (128 TiB)")
v("edge_w", m(SPARK.tdp, "W"), "{:.0f}", "DGX Spark power, W")
v("edge_mem_gb", m(SPARK.memory.capacity, "GiB"), "{:.0f}", "DGX Spark memory, labeled GB")
v("mobile_w", mobile_tdp_w()[1], "{:.0f}", "phone power, top of the mobile range, W")
v("mobile_w_lo", mobile_tdp_w()[0], "{:.0f}", "bottom of the mobile range, W")
v("mobile_ram_lo", mobile_ram_gb()[0], "{:.0f}", "phone RAM, low, GB")
v("mobile_ram_hi", mobile_ram_gb()[1], "{:.0f}", "phone RAM, high, GB")
v("tiny_mw", m(Platforms.Tiny.power_threshold, "mW"), "{:.0f}", "TinyML average-power target, mW")
v("tiny_sram_kb", m(Platforms.Tiny.ram, "KiB"), "{:.0f}", "ESP32-S3 SRAM, labeled KB")
v("tiny_price_word", m(ReferenceStats.EdgeDeviceSpectrum.MicrocontrollerBoardCost, "USD"), "words",
  "microcontroller board price, dollars, in prose")
v("tiny_price", m(ReferenceStats.EdgeDeviceSpectrum.MicrocontrollerBoardCost, "USD"), "{:.0f}",
  "microcontroller board price, dollars")
span_orders = math.log10(ratio(POD.power, Platforms.Tiny.power_threshold))
v("power_span_word", math.floor(span_orders), "words", "whole orders of magnitude, megawatts to milliwatts")
v("mem_drop_m", ratio(POD.memory, Platforms.Tiny.ram) / 10 ** 6, "{:.0f}",
  "pod HBM over TinyML SRAM, millions (128 TiB / 512 KiB)")

print("== Silicon contract")
v("tpu4_tflops", m(TPU4.compute.peak_flops, "TFLOP/s"), "{:.0f}", "TPU v4 BF16, TFLOP/s")
v("tpu4_bw_gbs", m(TPU4.memory.bandwidth, "GB/s"), "{:,.0f}", "TPU v4 HBM bandwidth, GB/s")
v("tpu4_hbm_gib", m(TPU4.memory.capacity, "GiB"), "{:.0f}", "TPU v4 HBM, GiB")
v("pod_eflops", m(Scenarios.TPUv4Pod4096.peak_flops, "EFLOP/s"), "{:.3f}", "pod BF16 peak, EFLOP/s")
v("pod_mem_tib", m(Scenarios.TPUv4Pod4096.chip_memory_total, "TiB"), "{:.0f}", "pod HBM, TiB")
v("edge_pflops", m(SPARK.compute.precision_flops["fp4_sparse"], "PFLOP/s"), "{:.0f}",
  "DGX Spark sparse FP4, PFLOP/s")
v("edge_bw_gbs", m(Platforms.Edge.bandwidth_threshold, "GB/s"), "{:.0f}", "edge tier bandwidth floor, GB/s")
v("phone_tops_lo", min(m(p.compute.peak_flops, "TOPS") for p in PHONES), "{:.0f}", "phone NPU, low, TOPS")
v("phone_tops_hi", max(m(p.compute.peak_flops, "TOPS") for p in PHONES), "{:.0f}", "phone NPU, high, TOPS")
v("phone_bw_lo", min(m(p.memory.bandwidth, "GB/s") for p in PHONES), "{:.1f}", "phone bandwidth, low, GB/s")
v("phone_bw_lo_int", min(m(p.memory.bandwidth, "GB/s") for p in PHONES), "{:.0f}",
  "the same, whole GB/s (notes)")
v("phone_bw_hi", max(m(p.memory.bandwidth, "GB/s") for p in PHONES), "{:.0f}", "phone bandwidth, high, GB/s")
v("tiny_tops", m(Platforms.Tiny.compute_threshold, "TOPS"), "{:.0f}", "TinyML compute ceiling, TOPS")

print("== Distance (propagation scenarios)")
v("arm_km", m(ARM.distance, "km"), "{:,.0f}", "arm to cloud region, km")
v("arm_budget_ms", m(ARM.budget, "ms"), "{:.0f}", "arm safety budget, ms")
v("arm_rtt_ms", m(ARM.round_trip, "ms"), "{:.0f}", "arm fiber floor, ms")
v("arm_overrun_ms", m(ARM.overrun, "ms"), "{:.0f}", "floor past the budget, ms")
fiber = mlsysim.physics.SPEED_OF_LIGHT_FIBER_KM_S
v("fiber_km_s", m(fiber, "km/s"), "{:,.0f}", "light in fiber, km/s")
v("fiber_km_ms", m(fiber, "km/ms"), "{:.0f}", "light in fiber, km per ms")
v("coast_km", m(COAST.distance, "km"), "{:,.0f}", "California to Virginia, km")
v("coast_rtt_ms", m(COAST.round_trip, "ms"), "{:.0f}", "coast-to-coast floor, ms")
for label, r in ROUTES.items():
    k = label.lower()
    v(f"rtt_{k}_km", m(r.distance, "km"), "{:,.0f}", f"N. Virginia to {label}, great circle, km")
    v(f"rtt_{k}_floor", m(r.round_trip, "ms"), "{:.1f}", f"{label} fiber floor, ms")
    v(f"rtt_{k}_meas", m(r.measured_round_trip, "ms"), "{:.1f}", f"{label} measured median, ms")
    v(f"rtt_{k}_ratio", r.measured_over_floor, "{:.1f}", f"{label} measured / floor")
v("rtt_ohio_km_under", math.ceil(m(ROUTES["Ohio"].distance, "km") / 100) * 100, "{:,.0f}",
  "Ohio distance, next hundred km")

print("== Factory cameras (sensor uplink scenario)")
v("cam_count", CAMS.streams, "{:.0f}", "cameras")
v("cam_w", CAMS.frame_width, "{:.0f}", "frame width, pixels")
v("cam_h", CAMS.frame_height, "{:.0f}", "frame height, pixels")
v("cam_bpp", m(CAMS.bytes_per_pixel, "byte"), "{:.0f}", "bytes per pixel")
v("cam_fps", m(CAMS.frame_rate, "Hz"), "{:.0f}", "frames per second")
v("cam_mbs", m(CAMS.stream_rate, "MB/s"), "{:.1f}", "one camera, MB/s")
v("cam_total_gbs", m(CAMS.total_rate, "GB/s"), "{:.1f}", "100 cameras, GB/s")
v("link_gbit", m(CAMS.link_bandwidth, "Gbit/s"), "{:.0f}", "factory link, Gb/s")
v("link_gbs", m(CAMS.link_bandwidth, "GB/s"), "{:.2f}", "factory link, GB/s")
v("cam_short", CAMS.link_shortfall, "{:.0f}", "raw / link")
v("cam_short_1", CAMS.link_shortfall, "{:.1f}", "raw / link, one decimal (figure)")
v("event_kb", m(CAMS.event_size, "KB"), "{:.0f}", "defect record, KB")
v("event_rate", m(CAMS.event_rate, "1/s"), "{:.0f}", "defect records per second")
v("event_kbs", m(CAMS.event_bandwidth, "KB/s"), "{:.0f}", "defect records, KB/s")
v("event_reduction", round(CAMS.event_reduction, -3), "{:,.0f}", "raw / records, to the thousand")

print("== Drone (offload-deadline scenario)")
v("drone_frame_mb", m(DRONE.payload, "MB"), "{:.1f}", "one 4K frame, MB")
v("drone_uplink_mbit", m(DRONE.uplink_bandwidth, "Mbit/s"), "{:.0f}", "uplink, Mb/s")
v("drone_remote_ms", m(DRONE.remote_response, "ms"), "{:.0f}", "remote round trip + compute, ms")
v("drone_deadline_ms", m(DRONE.deadline, "ms"), "{:.1f}", "frame deadline, ms")
v("drone_fps", m(DRONE.frame_rate, "Hz"), "{:.0f}", "frame rate, FPS")
v("drone_send_ms", m(DRONE.transmit_time, "ms"), "{:,.1f}", "send one frame, ms")
v("drone_send_s_word", m(DRONE.transmit_time, "s"), "words", "the same, whole seconds, in prose")
v("drone_bound_ms", m(DRONE.remote_lower_bound, "ms"), "{:,.1f}", "remote-path lower bound, ms")
v("drone_ratio", DRONE.deadline_ratio, "{:.0f}", "lower bound / deadline")
v("drone_compressed_ms", m(Scenarios.Drone4KObstacleOffloadCompressed.transmit_time, "ms"), "{:.0f}",
  "send after compression, ms")

print("== Memory wall (Gholami et al. 2024 via Literature; illustrative chapter rates)")
v("wall_flops_20y", LIT.MemoryWallFlopsGrowth20Years, "{:,.0f}", "peak FLOP/s over 20 years")
v("wall_dram_20y", LIT.MemoryWallDramGrowth20Years, "{:,.0f}", "DRAM bandwidth over 20 years")
v("wall_ic_20y", LIT.MemoryWallInterconnectGrowth20Years, "{:,.0f}", "interconnect over 20 years")
v("wall_flops_2y", LIT.MemoryWallFlopsGrowthPer2Years, "{:.1f}", "peak FLOP/s every 2 years")
v("wall_dram_2y", LIT.MemoryWallDramGrowthPer2Years, "{:.1f}", "DRAM bandwidth every 2 years")
v("wall_ic_2y", LIT.MemoryWallInterconnectGrowthPer2Years, "{:.1f}", "interconnect every 2 years")
for k, s in WALL_FITS.items():
    v(f"wall_fit_{k}", s.growth, "{:.1f}", f"{k}: refit growth every 2 years (figure)")
illus = Scenarios.IllustrativeMemoryWall
v("illus_compute_x", illus.compute_annual_factor, "{:.1f}", "chapter's compute growth per year")
v("illus_bw_x", illus.bandwidth_annual_factor, "{:.1f}", "chapter's bandwidth growth per year")

print("== A100 roofline (ResNet-50 scenario; self-check scenario; decode scenario on the A100)")
p = RESNET.result
v("a100_tflops", m(p.peak_flops_actual, "TFLOP/s"), "{:.0f}", "A100 FP16, TFLOP/s")
v("a100_tbs", m(p.peak_bw_actual, "TB/s"), "{:.2f}", "A100 HBM, TB/s")
v("a100_ridge", m(RESNET.ridge, "flop/byte"), "{:.0f}", "ridge, FLOP/byte")
v("check_intensity", m(SELF_CHECK.intensity, "flop/byte"), "{:.0f}", "self-check intensity, FLOP/byte")
v("check_tflops", m(SELF_CHECK.attainable, "TFLOP/s"), "{:.1f}", "self-check attainable, TFLOP/s")
v("check_pct", 100 * SELF_CHECK.peak_fraction, "{:.1f}", "self-check share of peak, percent")
v("resnet_gflop", m(RESNET.workload.inference_flops, "GFLOP"), "{:.1f}", "ResNet-50 per image, GFLOP")
v("resnet_mb", m(RESNET.traffic, "MB"), "{:.1f}", "bytes moved at FP16, MB")
v("resnet_compute_ms", m(p.latency_compute, "ms"), "{:.3f}", "A100 compute time, ms")
v("resnet_memory_ms", m(p.latency_memory, "ms"), "{:.3f}", "A100 memory time, ms")
v("resnet_intensity", m(RESNET.intensity, "flop/byte"), "{:.0f}", "intensity, FLOP/byte")
q = RESNET_NPU.result
v("npu_tops", m(q.peak_flops_actual, "TOPS"), "{:.0f}", "phone NPU INT8, TOPS")
v("npu_gbs", m(q.peak_bw_actual, "GB/s"), "{:.1f}", "phone bandwidth, GB/s")
v("npu_mb", m(RESNET_NPU.traffic, "MB"), "{:.1f}", "bytes moved at INT8, MB")
v("npu_compute_ms", m(RESNET_NPU.workload.inference_flops / q.peak_flops_actual, "ms"), "{:.2f}",
  "NPU compute time, ms")
v("npu_memory_ms", m(q.latency_memory, "ms"), "{:.2f}", "NPU memory time, ms")
d = DECODE.result
v("decode_intensity", m(d.intensity, "flop/byte"), "{:.0f}", "decode intensity, FLOP/byte")
v("decode_tflops", m(d.attainable_flops, "TFLOP/s"), "{:.0f}", "decode attainable, TFLOP/s")
v("decode_pct_under", math.ceil(100 * d.peak_fraction), "{:.0f}", "decode share of peak, under this percent")

print("== TPU v1 roofline (Jouppi et al. 2017 via Literature)")
apps = TPU_ROOF.apps
mix = {r["family"]: r["deployed_share_pct"] for r in load_series("slides/vol1/02_system_paradigms/images/data/tpu_v1_deployment_mix.csv").entries}
bound_share = sum(mix[f] for f in {a["family"] for a in apps if a["bound"] == "memory"})
v("tpu1_tops", m(TPU_ROOF.peak, "TOPS"), "{:.0f}", "TPU v1 INT8 peak, TOPS")
v("tpu1_ridge", TPU_ROOF.ridge, "{:,.0f}", "ridge, ops per weight byte")
v("tpu1_apps_word", len(apps), "words", "production applications")
v("tpu1_slope_share", bound_share, "{:.0f}", "deployed share of slope-bound families, percent")
v("tpu1_cnn_share", sum(mix[f] for f in {a["family"] for a in apps if a["bound"] == "compute"}), "{:.0f}",
  "deployed share of ceiling-bound families, percent")
v("tpu1_all_share", sum(mix.values()), "{:.0f}", "the six applications' share of the workload, percent")
v("tpu1_cnn0_tops", next(a["measured_tops"] for a in apps if a["app"] == "CNN0"), "{:.0f}",
  "CNN0 measured, TOPS")

print("== Lighthouse archetypes")
v("gpt2_b", m(Models.Language.GPT2.parameters, "Gparam"), "{:.1f}", "GPT-2 parameters, billions")
v("llama70_b", m(Models.Language.Llama2_70B.parameters, "Gparam"), "{:.0f}", "Llama 2 70B, billions")
v("dlrm_gb", m(Models.Recommendation.DLRM.size_in_bytes(), "GB"), "{:.0f}", "DLRM embeddings, GB")
v("mobilenet_x", ratio(Models.Vision.ResNet50.inference_flops, Models.Vision.MobileNetV2.inference_flops),
  "{:.1f}", "ResNet-50 / MobileNetV2 FLOPs")
v("kws_kb", m(Models.Tiny.DS_CNN.size_in_bytes(mlsysim.BYTES_INT8), "KB"), "{:.0f}", "KWS at INT8, KB")

print("== Power wall (clock-scaling scenario; trends scenario)")
clock = Scenarios.DoubledClockPower
v("clock_x", clock.frequency_ratio, "{:.0f}", "clock multiplied by")
v("clock_power_x", clock.power_ratio, "{:.0f}", "dynamic power multiplied by")
v("clock_power_word", clock.power_ratio, "words", "the same, in prose")
v("trend_ghz", m(TRENDS.peak("frequency", "after_2005") * mlsysim.core.units.MHz, "GHz"), "{:.1f}",
  "fastest clock after 2005, GHz")
v("trend_transistors_x", round(TRENDS.growth("transistors", "mid_2000s", "early_2020s"), -1), "{:.0f}",
  "transistor growth, mid-2000s to early 2020s, to the ten")

print("== Throttling (3DMark via Literature; thermal scenario)")
best, low = LIT.WildLifeStressBestLoopIphone15ProMax, LIT.WildLifeStressLowestLoopIphone15ProMax
v("wl_loops", LIT.WildLifeStressLoopsRun, "{:.0f}", "loops in the run")
v("wl_loops_word", LIT.WildLifeStressLoopsRun, "words", "the same, in prose")
v("wl_best", best, "{:,.0f}", "best loop score")
v("wl_low", low, "{:,.0f}", "lowest loop score")
v("wl_stability", 100 * low / best, "{:.1f}", "stability, percent")
v("phone_ceiling_w", m(THERMAL.sustained_limit, "W"), "{:.0f}", "passive-cooling ceiling, W")
v("hot_model_w", m(THERMAL.power, "W"), "{:.0f}", "unoptimized model, W")
v("heat_rate", m(THERMAL.heating_rate, "delta_degC/s"), "{:.0f}", "heating rate, degrees C per s")
v("trip_c", m(THERMAL.trip_temperature, "degC"), "{:.0f}", "thermal trip point, degrees C")
v("weight_shrink_x", ratio(mlsysim.BYTES_FP32, mlsysim.BYTES_INT8), "{:.0f}", "FP32 / INT8 bytes")

print("== Battery tax (battery scenario)")
v("batt_w", m(BATTERY.power_draw, "W"), "{:.0f}", "detector power, W")
v("batt_wh", m(BATTERY.battery_energy, "Wh"), "{:.0f}", "battery, Wh")
v("batt_day_h", m(BATTERY.required_runtime, "hour"), "{:.0f}", "required runtime, h")
v("batt_runtime_h", m(BATTERY.runtime, "hour"), "{:.1f}", "runtime, h")
v("batt_day_wh", m(BATTERY.required_energy, "Wh"), "{:.0f}", "a day at the draw, Wh")
v("batt_pct", 100 * BATTERY.budget_ratio, "{:.0f}", "share of the battery, percent")

print("== Energy per inference (energy scenario; upload scenario; MLPerf Tiny via Literature)")
e = ENERGY.energies
v("e_kws_uj", m(e["kws"], "uJ"), "{:.0f}", "TinyML KWS anchor, uJ")
v("e_mobile_mj", m(e["mobilenet_phone"], "mJ"), "{:.1f}", "MobileNetV2 on a phone, mJ")
v("e_edge_mj", m(e["resnet_edge"], "mJ"), "{:.1f}", "ResNet-50 on a Jetson, mJ")
v("e_cloud_mj", m(e["resnet_cloud"], "mJ"), "{:.1f}", "ResNet-50 on an A100, mJ")
v("e_llm_kj", m(e["llm_query"], "kJ"), "{:.0f}", "cloud LLM query anchor, kJ")
span = ENERGY.span("kws", "llm_query")
v("e_span", span, "{:,.0f}", "LLM query / KWS")
v("e_span_m", span / 10 ** 6, "{:.0f}", "the same, millions")
v("e_span_word", math.log10(span), "words", "the same, orders of magnitude, in prose")
v("upload_mb", m(UPLOAD.payload, "MB"), "{:.0f}", "payload, MB")
v("upload_mj", m(UPLOAD.transmit_energy, "mJ"), "{:.0f}", "send it by radio, mJ")
v("npu_mj", m(UPLOAD.accelerator_energy, "mJ"), "{:.1f}", "NPU alone, mJ")
v("upload_npu_x", UPLOAD.ratio_accelerator, "{:,.0f}", "send / NPU")
v("upload_phone_x", UPLOAD.ratio_device, "{:.0f}", "send / whole phone")
kws = tiny_kws()
best_kws = min(kws, key=lambda r: r["energy_uj"])
m4 = next(r for r in kws if r["board"] == "NUCLEO-L4R5ZI")
v("tiny_best_uj", best_kws["energy_uj"], "{:.1f}", "MLPerf Tiny KWS best (NDP9120), uJ")
v("tiny_m4_uj", m4["energy_uj"], "{:,.0f}", "Cortex-M4 board, uJ")
v("tiny_m4_x", m4["energy_uj"] / best_kws["energy_uj"], "{:.0f}", "Cortex-M4 / NDP9120")

print("== Doorbell (doorbell scenarios)")
ev, ev_strict = DOORBELL.evaluate(), CHIME.evaluate()
fp = FRAME.result
weights, flash = ev_strict.feasibility.metrics["weight_size"], ev_strict.feasibility.metrics["capacity"]
lat = ev_strict.performance.metrics["latency"]
v("db_weights_mb", m(weights, "MB"), "{:.1f}", "16-bit weights, MB")
v("db_flash_mb", m(flash, "MB"), "{:.1f}", "flash, MB")
v("db_flash_pct", 100 * ratio(weights, flash), "{:.1f}", "weights / flash, percent")
v("db_mflop", m(DOORBELL.workload.inference_flops, "MFLOP"), "{:.0f}", "operations per frame, millions")
v("db_rate", m(fp.peak_flops_actual, "GFLOP/s"), "{:.2f}", "sustained rate, GFLOP/s")
v("db_compute_ms", m(fp.latency_compute, "ms"), "{:.0f}", "compute time, ms")
v("db_overhead_ms", m(fp.latency_overhead, "ms"), "{:.0f}", "fixed overhead, ms")
v("db_latency_ms", m(lat, "ms"), "{:.0f}", "latency per frame, ms")
v("db_sla_ms", m(DOORBELL.sla_latency, "ms"), "{:.0f}", "chime budget, ms")
v("db_strict_ms", m(CHIME.sla_latency, "ms"), "{:.0f}", "responsive chime budget, ms")
v("db_sla_pct", 100 * ratio(lat, DOORBELL.sla_latency), "{:.1f}", "latency / 200 ms budget, percent")
v("db_strict_pct", 100 * round(ratio(lat, CHIME.sla_latency), 2), "{:.0f}", "latency / 50 ms budget, percent")

print("== Screening gates (braking scenario)")
v("brake_kmh", m(BRAKING.speed, "km/hour"), "{:.0f}", "speed, km/h")
v("brake_ms", m(BRAKING.latency, "ms"), "{:.0f}", "budget, ms")
v("brake_m", m(BRAKING.distance, "m"), "{:.1f}", "travel during the budget, m")
v("brake_cloud_lo", m(BRAKING.remote_delay_low, "ms"), "{:.0f}", "cloud round trip, low, ms")
v("brake_cloud_hi", m(BRAKING.remote_delay_high, "ms"), "{:.0f}", "cloud round trip, high, ms")

print("== Voice assistant (wake-word fleet scenario)")
v("voice_devices_b", VOICE.devices / 10 ** 9, "{:.0f}", "devices, billions")
v("voice_khz", m(VOICE.audio_sample_rate, "kHz"), "{:.0f}", "audio sample rate, kHz")
v("voice_bits", m(VOICE.audio_sample_bits, "bit"), "{:.0f}", "bits per sample")
v("voice_kbs", m(VOICE.stream_rate, "KB/s"), "{:.0f}", "one stream, KB/s")
v("voice_tbs", m(VOICE.fleet_stream_rate, "TB/s"), "{:.0f}", "every microphone, TB/s")
v("voice_queries", VOICE.queries_per_day, "{:.0f}", "queries per device per day")
v("voice_query_s_word", m(VOICE.query_audio, "s"), "words", "audio per query, s, in prose")
v("voice_cut", round(VOICE.upload_reduction, -1), "{:.0f}", "audio cut, to the ten")
v("voice_after_gbs", m(VOICE.fleet_query_rate, "GB/s"), "{:.0f}", "audio after the cut, GB/s")
v("voice_cost_usd", m(VOICE.cloud_detection_cost, "USD/year"), "{:.2f}", "cloud detection per device, $/year")
v("voice_cost_m", m(VOICE.cloud_detection_cost_per_year, "USD/year") / 10 ** 6, "{:.0f}",
  "cloud detection, $M a year")
v("voice_local_mw_lo", m(VOICE.local_power_low, "mW"), "{:.1f}", "on-device detector, low, mW")
v("voice_local_mw_hi", m(VOICE.local_power_high, "mW"), "{:.0f}", "on-device detector, high, mW")
v("voice_queries_b", VOICE.queries_per_fleet_day / 10 ** 9, "{:.0f}", "queries a day, billions")
v("voice_gpu_ms", m(VOICE.accelerator_time_per_query, "ms"), "{:.0f}", "GPU time per query, ms")
v("voice_gpuh_m", VOICE.accelerator_hours_per_day / 10 ** 6, "{:.1f}", "GPU-hours a day, millions")
v("voice_gpus_dc", VOICE.accelerators_per_datacenter, "{:,.0f}", "GPUs per datacenter")
v("voice_dc_avg", VOICE.datacenters_average, "{:.1f}", "datacenters, average")
v("voice_peak_x", VOICE.peak_ratio, "{:.1f}", "peak / average")
v("voice_dc_peak", VOICE.datacenters_peak, "{:.1f}", "datacenters at peak")

print("== Hey Siri (Apple 2017 via Literature)")
v("siri_layers_word", LIT.HeySiriFirstPassLayers, "words", "first-pass layers, in prose")
v("siri_layers", LIT.HeySiriFirstPassLayers, "{:.0f}", "first-pass layers")
v("siri_units", LIT.HeySiriFirstPassUnits, "{:.0f}", "first-pass units per layer")
v("siri_units2", LIT.HeySiriSecondPassUnits, "{:.0f}", "second-pass units per layer")
v("siri_wider", LIT.HeySiriSecondPassUnits / LIT.HeySiriFirstPassUnits, "{:.0f}", "second / first width")

print("== Camera pipeline (pipeline scenario)")
v("cam_pipe_ms", m(CAMERA.total, "ms"), "{:.0f}", "pipeline, ms")
v("cam_speedup", CAMERA.speedup, "{:.0f}", "faster ML stage")
v("cam_isp_ms", m(CAMERA.stages["isp"], "ms"), "{:.0f}", "image processing, ms")
v("cam_ml_ms", m(CAMERA.stages["ml"], "ms"), "{:.0f}", "ML stage, ms")
v("cam_post_ms", m(CAMERA.stages["post"], "ms"), "{:.0f}", "tone mapping, ms")
v("cam_ml_fast_ms", m(CAMERA.accelerated_stage, "ms"), "{:.0f}", "ML stage after, ms")
v("cam_pipe_fast_ms", m(CAMERA.accelerated_total, "ms"), "{:.0f}", "pipeline after, ms")
v("cam_overall_x", CAMERA.overall_speedup, "{:.2f}", "end-to-end speedup")
v("cam_limit_x", CAMERA.limit, "{:.2f}", "limit with an infinitely fast model")

if __name__ == "__main__":
    write(__file__)
