#!/usr/bin/env python3
"""Data plots for Lecture 2 (System Paradigms).

Every value comes from calcs.py, which reads MLSysIM (registry, Literature,
and Scenarios.*); figure labels print the same strings the slides print
(deckvals.val). Sources, by figure:
  - Power tiers: TPU v4 pod, DGX Spark, the mobile envelope, and the TinyML
    envelope (Systems.Pods, Hardware, Platforms).
  - Round trips: Scenarios.RoboticArmRegionalCloud and AwsUsEast1Routes
    (cloudping.co medians via slides/vol1/02_system_paradigms/images/data/aws_interregion_rtt_p50.csv).
  - Camera bandwidth: Scenarios.FactoryCameras1080p.
  - Memory wall: Gholami et al., IEEE Micro 2024, Fig. 1 markers
    (slides/vol1/02_system_paradigms/images/data/memory_wall_hardware_scaling.csv) and their log-linear fits
    (Scenarios.MemoryWall*Trend).
  - A100 roofline: Scenarios.ResNet50A100Batch1; decode intensity from the
    Llama-3 70B decode scenario on the A100.
  - TPU roofline: Scenarios.TpuV1ProductionRoofline (Jouppi et al. 2017).
  - Voice assistant: Scenarios.VoiceAssistantBillionDevices.
  - Microprocessor trends: vol1/02_system_paradigms/data/microprocessor_trends.csv (Karl Rupp).
  - Energy per inference: Scenarios.EnergyPerInferenceTiers and
    RadioUploadVsMobileNetV2; measured row from MLPerf Tiny v1.2
    (slides/vol1/02_system_paradigms/images/data/mlperf_tiny_v1_2_kws_energy.csv).
  - Doorbell: Scenarios.SmartDoorbell and SmartDoorbellResponsiveChime.

Run from this folder: python3 make_figs.py
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
from mlsysim.core.series_loader import load_series
import mlsysim
_VALS = {'pod_mw': '3', 'pod_chips': '4,096', 'pod_mem_tb': '128', 'edge_w': '200', 'edge_mem_gb': '128', 'mobile_w': '5', 'mobile_w_lo': '3', 'mobile_ram_lo': '8', 'mobile_ram_hi': '16', 'tiny_mw': '1', 'tiny_sram_kb': '512', 'tiny_price_word': 'ten', 'tiny_price': '10', 'power_span_word': 'nine', 'mem_drop_m': '268', 'tpu4_tflops': '275', 'tpu4_bw_gbs': '1,200', 'tpu4_hbm_gib': '32', 'pod_eflops': '1.126', 'pod_mem_tib': '128', 'edge_pflops': '1', 'edge_bw_gbs': '270', 'phone_tops_lo': '15', 'phone_tops_hi': '45', 'phone_bw_lo': '51.2', 'phone_bw_lo_int': '51', 'phone_bw_hi': '77', 'tiny_tops': '1', 'arm_km': '1,500', 'arm_budget_ms': '10', 'arm_rtt_ms': '15', 'arm_overrun_ms': '5', 'fiber_km_s': '200,000', 'fiber_km_ms': '200', 'coast_km': '3,600', 'coast_rtt_ms': '36', 'rtt_ohio_km': '484', 'rtt_ohio_floor': '4.8', 'rtt_ohio_meas': '13.4', 'rtt_ohio_ratio': '2.8', 'rtt_oregon_km': '3,503', 'rtt_oregon_floor': '35.0', 'rtt_oregon_meas': '68.1', 'rtt_oregon_ratio': '1.9', 'rtt_ireland_km': '5,462', 'rtt_ireland_floor': '54.6', 'rtt_ireland_meas': '68.1', 'rtt_ireland_ratio': '1.2', 'rtt_tokyo_km': '10,872', 'rtt_tokyo_floor': '108.7', 'rtt_tokyo_meas': '148.3', 'rtt_tokyo_ratio': '1.4', 'rtt_ohio_km_under': '500', 'cam_count': '100', 'cam_w': '1920', 'cam_h': '1080', 'cam_bpp': '3', 'cam_fps': '30', 'cam_mbs': '186.6', 'cam_total_gbs': '18.7', 'link_gbit': '10', 'link_gbs': '1.25', 'cam_short': '15', 'cam_short_1': '14.9', 'event_kb': '1', 'event_rate': '20', 'event_kbs': '20', 'event_reduction': '933,000', 'drone_frame_mb': '24.9', 'drone_uplink_mbit': '100', 'drone_remote_ms': '110', 'drone_deadline_ms': '16.7', 'drone_fps': '60', 'drone_send_ms': '1,990.7', 'drone_send_s_word': 'two', 'drone_bound_ms': '2,100.7', 'drone_ratio': '126', 'drone_compressed_ms': '20', 'wall_flops_20y': '60,000', 'wall_dram_20y': '100', 'wall_ic_20y': '30', 'wall_flops_2y': '3.0', 'wall_dram_2y': '1.6', 'wall_ic_2y': '1.4', 'wall_fit_flops': '3.0', 'wall_fit_dram': '1.6', 'wall_fit_interconnect': '1.4', 'illus_compute_x': '1.6', 'illus_bw_x': '1.2', 'a100_tflops': '312', 'a100_tbs': '2.04', 'a100_ridge': '153', 'check_intensity': '10', 'check_tflops': '20.4', 'check_pct': '6.5', 'resnet_gflop': '8.2', 'resnet_mb': '56.2', 'resnet_compute_ms': '0.026', 'resnet_memory_ms': '0.028', 'resnet_intensity': '146', 'npu_tops': '35', 'npu_gbs': '51.2', 'npu_mb': '28.1', 'npu_compute_ms': '0.23', 'npu_memory_ms': '0.55', 'decode_intensity': '1', 'decode_tflops': '2', 'decode_pct_under': '1', 'tpu1_tops': '92', 'tpu1_ridge': '1,350', 'tpu1_apps_word': 'six', 'tpu1_slope_share': '90', 'tpu1_cnn_share': '5', 'tpu1_all_share': '95', 'tpu1_cnn0_tops': '86', 'gpt2_b': '1.5', 'llama70_b': '70', 'dlrm_gb': '100', 'mobilenet_x': '13.7', 'kws_kb': '200', 'clock_x': '2', 'clock_power_x': '8', 'clock_power_word': 'eight', 'trend_ghz': '4.7', 'trend_transistors_x': '190', 'wl_loops': '20', 'wl_loops_word': 'twenty', 'wl_best': '10,012', 'wl_low': '7,045', 'wl_stability': '70.4', 'phone_ceiling_w': '3', 'hot_model_w': '12', 'heat_rate': '1', 'trip_c': '80', 'weight_shrink_x': '4', 'batt_w': '2', 'batt_wh': '15', 'batt_day_h': '24', 'batt_runtime_h': '7.5', 'batt_day_wh': '48', 'batt_pct': '320', 'e_kws_uj': '10', 'e_mobile_mj': '2.6', 'e_edge_mj': '15.9', 'e_cloud_mj': '75.5', 'e_llm_kj': '1', 'e_span': '100,000,000', 'e_span_m': '100', 'e_span_word': 'eight', 'upload_mb': '1', 'upload_mj': '100', 'npu_mj': '0.1', 'upload_npu_x': '1,000', 'upload_phone_x': '39', 'tiny_best_uj': '31.5', 'tiny_m4_uj': '1,820', 'tiny_m4_x': '58', 'db_weights_mb': '0.5', 'db_flash_mb': '4.2', 'db_flash_pct': '11.9', 'db_mflop': '25', 'db_rate': '0.25', 'db_compute_ms': '100', 'db_overhead_ms': '1', 'db_latency_ms': '101', 'db_sla_ms': '200', 'db_strict_ms': '50', 'db_sla_pct': '50.5', 'db_strict_pct': '202', 'brake_kmh': '100', 'brake_ms': '100', 'brake_m': '2.8', 'brake_cloud_lo': '50', 'brake_cloud_hi': '150', 'voice_devices_b': '1', 'voice_khz': '16', 'voice_bits': '16', 'voice_kbs': '32', 'voice_tbs': '32', 'voice_queries': '20', 'voice_query_s_word': 'five', 'voice_cut': '860', 'voice_after_gbs': '37', 'voice_cost_usd': '0.50', 'voice_cost_m': '500', 'voice_local_mw_lo': '0.1', 'voice_local_mw_hi': '1', 'voice_queries_b': '20', 'voice_gpu_ms': '200', 'voice_gpuh_m': '1.1', 'voice_gpus_dc': '10,000', 'voice_dc_avg': '4.6', 'voice_peak_x': '4.5', 'voice_dc_peak': '20.8', 'siri_layers_word': 'five', 'siri_layers': '5', 'siri_units': '32', 'siri_units2': '192', 'siri_wider': '6', 'cam_pipe_ms': '200', 'cam_speedup': '10', 'cam_isp_ms': '100', 'cam_ml_ms': '60', 'cam_post_ms': '40', 'cam_ml_fast_ms': '6', 'cam_pipe_fast_ms': '146', 'cam_overall_x': '1.37', 'cam_limit_x': '1.43'}
_NUMS = {'pod_mw': 3.0, 'pod_chips': 4096.0, 'pod_mem_tb': 128.0, 'edge_w': 200.0, 'edge_mem_gb': 128.0, 'mobile_w': 5.0, 'mobile_w_lo': 3.0, 'mobile_ram_lo': 8.0, 'mobile_ram_hi': 16.0, 'tiny_mw': 1.0, 'tiny_sram_kb': 512.0, 'tiny_price': 10.0, 'mem_drop_m': 268.0, 'tpu4_tflops': 275.0, 'tpu4_bw_gbs': 1200.0, 'tpu4_hbm_gib': 32.0, 'pod_eflops': 1.126, 'pod_mem_tib': 128.0, 'edge_pflops': 1.0, 'edge_bw_gbs': 270.0, 'phone_tops_lo': 15.0, 'phone_tops_hi': 45.0, 'phone_bw_lo': 51.2, 'phone_bw_lo_int': 51.0, 'phone_bw_hi': 77.0, 'tiny_tops': 1.0, 'arm_km': 1500.0, 'arm_budget_ms': 10.0, 'arm_rtt_ms': 15.0, 'arm_overrun_ms': 5.0, 'fiber_km_s': 200000.0, 'fiber_km_ms': 200.0, 'coast_km': 3600.0, 'coast_rtt_ms': 36.0, 'rtt_ohio_km': 484.0, 'rtt_ohio_floor': 4.8, 'rtt_ohio_meas': 13.4, 'rtt_ohio_ratio': 2.8, 'rtt_oregon_km': 3503.0, 'rtt_oregon_floor': 35.0, 'rtt_oregon_meas': 68.1, 'rtt_oregon_ratio': 1.9, 'rtt_ireland_km': 5462.0, 'rtt_ireland_floor': 54.6, 'rtt_ireland_meas': 68.1, 'rtt_ireland_ratio': 1.2, 'rtt_tokyo_km': 10872.0, 'rtt_tokyo_floor': 108.7, 'rtt_tokyo_meas': 148.3, 'rtt_tokyo_ratio': 1.4, 'rtt_ohio_km_under': 500.0, 'cam_count': 100.0, 'cam_w': 1920.0, 'cam_h': 1080.0, 'cam_bpp': 3.0, 'cam_fps': 30.0, 'cam_mbs': 186.6, 'cam_total_gbs': 18.7, 'link_gbit': 10.0, 'link_gbs': 1.25, 'cam_short': 15.0, 'cam_short_1': 14.9, 'event_kb': 1.0, 'event_rate': 20.0, 'event_kbs': 20.0, 'event_reduction': 933000.0, 'drone_frame_mb': 24.9, 'drone_uplink_mbit': 100.0, 'drone_remote_ms': 110.0, 'drone_deadline_ms': 16.7, 'drone_fps': 60.0, 'drone_send_ms': 1990.7, 'drone_bound_ms': 2100.7, 'drone_ratio': 126.0, 'drone_compressed_ms': 20.0, 'wall_flops_20y': 60000.0, 'wall_dram_20y': 100.0, 'wall_ic_20y': 30.0, 'wall_flops_2y': 3.0, 'wall_dram_2y': 1.6, 'wall_ic_2y': 1.4, 'wall_fit_flops': 3.0, 'wall_fit_dram': 1.6, 'wall_fit_interconnect': 1.4, 'illus_compute_x': 1.6, 'illus_bw_x': 1.2, 'a100_tflops': 312.0, 'a100_tbs': 2.04, 'a100_ridge': 153.0, 'check_intensity': 10.0, 'check_tflops': 20.4, 'check_pct': 6.5, 'resnet_gflop': 8.2, 'resnet_mb': 56.2, 'resnet_compute_ms': 0.026, 'resnet_memory_ms': 0.028, 'resnet_intensity': 146.0, 'npu_tops': 35.0, 'npu_gbs': 51.2, 'npu_mb': 28.1, 'npu_compute_ms': 0.23, 'npu_memory_ms': 0.55, 'decode_intensity': 1.0, 'decode_tflops': 2.0, 'decode_pct_under': 1.0, 'tpu1_tops': 92.0, 'tpu1_ridge': 1350.0, 'tpu1_slope_share': 90.0, 'tpu1_cnn_share': 5.0, 'tpu1_all_share': 95.0, 'tpu1_cnn0_tops': 86.0, 'gpt2_b': 1.5, 'llama70_b': 70.0, 'dlrm_gb': 100.0, 'mobilenet_x': 13.7, 'kws_kb': 200.0, 'clock_x': 2.0, 'clock_power_x': 8.0, 'trend_ghz': 4.7, 'trend_transistors_x': 190.0, 'wl_loops': 20.0, 'wl_best': 10012.0, 'wl_low': 7045.0, 'wl_stability': 70.4, 'phone_ceiling_w': 3.0, 'hot_model_w': 12.0, 'heat_rate': 1.0, 'trip_c': 80.0, 'weight_shrink_x': 4.0, 'batt_w': 2.0, 'batt_wh': 15.0, 'batt_day_h': 24.0, 'batt_runtime_h': 7.5, 'batt_day_wh': 48.0, 'batt_pct': 320.0, 'e_kws_uj': 10.0, 'e_mobile_mj': 2.6, 'e_edge_mj': 15.9, 'e_cloud_mj': 75.5, 'e_llm_kj': 1.0, 'e_span': 100000000.0, 'e_span_m': 100.0, 'upload_mb': 1.0, 'upload_mj': 100.0, 'npu_mj': 0.1, 'upload_npu_x': 1000.0, 'upload_phone_x': 39.0, 'tiny_best_uj': 31.5, 'tiny_m4_uj': 1820.0, 'tiny_m4_x': 58.0, 'db_weights_mb': 0.5, 'db_flash_mb': 4.2, 'db_flash_pct': 11.9, 'db_mflop': 25.0, 'db_rate': 0.25, 'db_compute_ms': 100.0, 'db_overhead_ms': 1.0, 'db_latency_ms': 101.0, 'db_sla_ms': 200.0, 'db_strict_ms': 50.0, 'db_sla_pct': 50.5, 'db_strict_pct': 202.0, 'brake_kmh': 100.0, 'brake_ms': 100.0, 'brake_m': 2.8, 'brake_cloud_lo': 50.0, 'brake_cloud_hi': 150.0, 'voice_devices_b': 1.0, 'voice_khz': 16.0, 'voice_bits': 16.0, 'voice_kbs': 32.0, 'voice_tbs': 32.0, 'voice_queries': 20.0, 'voice_cut': 860.0, 'voice_after_gbs': 37.0, 'voice_cost_usd': 0.5, 'voice_cost_m': 500.0, 'voice_local_mw_lo': 0.1, 'voice_local_mw_hi': 1.0, 'voice_queries_b': 20.0, 'voice_gpu_ms': 200.0, 'voice_gpuh_m': 1.1, 'voice_gpus_dc': 10000.0, 'voice_dc_avg': 4.6, 'voice_peak_x': 4.5, 'voice_dc_peak': 20.8, 'siri_layers': 5.0, 'siri_units': 32.0, 'siri_units2': 192.0, 'siri_wider': 6.0, 'cam_pipe_ms': 200.0, 'cam_speedup': 10.0, 'cam_isp_ms': 100.0, 'cam_ml_ms': 60.0, 'cam_post_ms': 40.0, 'cam_ml_fast_ms': 6.0, 'cam_pipe_fast_ms': 146.0, 'cam_overall_x': 1.37, 'cam_limit_x': 1.43}
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
from mlsysim.core.series_loader import load_series
out = Path(__file__).parent
C = sp.C
Q_ = mlsysim.Q_

def mag(q, unit):
    return q.to(unit).magnitude
sp.hbars_steps(out / 'power_tiers', [('Cloud', mag(Q_(3, 'MW'), 'W'), 'neutral', '3 MW   ·  128 TB'), ('Edge', mag(Q_(200, 'W'), 'W'), 'neutral', '200 W  ·  128 GB'), ('Mobile', 5.0, 'neutral', '5 W  ·  8–16 GB'), ('TinyML', mag(Q_(1, 'mW'), 'W'), 'neutral', '1 mW  ·  512 KB')], 'W', xlabel='Power (watts, log scale)   ·   memory of the example platform')

def rtt(step):
    rows = [('Arm to cloud, 1,500 km', None, mag(Q_(15.0, 'ms'), 'ms'), None)]
    rows += [('Virginia to Ohio, 484 km', None, 4.836163932, 13.446), ('Virginia to Oregon, 3,503 km', None, 35.02514629, 68.147), ('Virginia to Ireland, 5,462 km', None, 54.61523595, 68.126), ('Virginia to Tokyo, 10,872 km', None, 108.7218142, 148.291)]
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.75))
    n = len(rows)
    for i, (label, km, floor, meas) in enumerate(rows):
        y = n - 1 - i
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
        ax.barh(y + 0.17, floor, color=C['danger_l'], edgecolor=C['danger'], lw=1.2, height=0.32)
        ax.text(max(floor, 10) + 1.5, y + 0.17, f'{floor:.1f}' if floor < 10 else f'{floor:.0f}', va='center', fontsize=11, color=C['danger'], fontweight='bold')
        if meas is not None:
            a2 = 1.0 if step == 2 else 0.0
            ax.barh(y - 0.17, meas, color=C['danger'], height=0.32, alpha=a2)
            ax.text(meas + 1.5, y - 0.17, f'{meas:.0f} measured', va='center', fontsize=11, color=C['danger'], fontweight='bold', alpha=a2)
    budget = 10.0
    ax.axvline(budget, color=C['data'], lw=2.5)
    ax.text(budget + 1, n - 0.25, 'Arm budget 10 ms', color=C['data'], fontsize=11, fontweight='bold', va='center')
    ax.text(0.99, 0.99, 'light: fiber floor\ndark: AWS median', transform=ax.transAxes, ha='right', va='top', fontsize=10, color=C['neutral'])
    ax.set_xlim(0, 185)
    ax.set_ylim(-0.6, n + 0.05)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('Round trip (ms)')
    sp.save(fig, out / f'rtt_measured_step{step}.svg')
for k in (1, 2):
    rtt(k)
CAM_RAW = mag(Q_(18662400000.0, 'B / s'), 'B/s')
CAM_LINK = mag(Q_(10, 'Gbps'), 'B/s')
CAM_META = mag(Q_(20000.0, 'B / s'), 'B/s')
with sp.plt.rc_context({'xtick.labelsize': 14}):
    sp.hbars_steps(out / 'camera_bandwidth', [('100 raw camera feeds', CAM_RAW, 'danger', '18.7 GB/s'), ('10 Gb/s factory link', CAM_LINK, 'neutral', '1.25 GB/s: 15× too small'), ('Defect metadata only', CAM_META, 'data', '20 KB/s')], 'B/s', xlabel='Data rate (bytes per second, log scale)')

def memory_wall():
    pts = {'flops': [(1997.079403, 1.000037468), (1998.002669, 0.6873408712), (2002.410782, 38.34979599), (2006.550409, 132.7496217), (2008.007908, 294.9996246), (2009.010624, 324.4996449), (2010.0106, 471.9998417), (2011.010576, 501.4998904), (2012.010552, 1180.001746), (2013.013268, 1711.003785), (2014.013244, 4130.016323), (2016.013196, 5015.021692), (2013.013268, 7375.037687), (2016.013196, 21240.15291), (2007.336691, 1528.103034), (2007.298336, 273.7596118), (2008.007908, 619.5001189), (2009.010624, 1770.004006), (2010.0106, 2065.00532), (2010.865374, 4663.969504), (2012.010552, 8850.048275), (2015.01322, 18124.92488), (2014.013244, 16225.10813), (2016.013196, 29500.23125), (2017.015911, 62540.58324), (2017.484393, 330404.1712), (2020.382954, 920413.4814), (2022.736322, 2919074.458), (2008.007908, 3097.51048), (2009.010624, 8850.048275), (2012.010552, 14750.09533), (2013.013268, 23600.17464), (2016.013196, 47200.41423), (2017.015911, 40503.84349), (2017.388505, 132751.4376), (2018.363824, 362854.6468), (2022.388385, 811261.6819)], 'dram': [(2003.753216, 1.000037468), (2006.799718, 4.21069608), (2008.490088, 7.579261674), (2015.533755, 33.68570692), (2016.481677, 48.17322707), (2016.752904, 21.07328501), (2017.484393, 59.015845), (2018.807649, 29.474986), (2020.489801, 102.3074015), (2020.684316, 50.02199862), (2021.506215, 127.3086222), (2022.237704, 134.151063)], 'interconnect': [(2003.005288, 1.000037468), (2007.046287, 2.000077688), (2010.843457, 3.200127225), (2016.273463, 8.000332585), (2016.873449, 6.400263258), (2017.484393, 10.00042036), (2019.424073, 12.80054413), (2020.489801, 20.00086777), (2022.018531, 25.60112349), (2022.736322, 40.00179058)]}
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.75))
    spec = [('flops', C['compute'], 'Peak FLOP/s'), ('dram', C['memory'], 'DRAM bandwidth'), ('interconnect', C['network'], 'Interconnect bandwidth')]
    for key, col, name in spec:
        xs = [p[0] for p in pts[key]]
        ys = [p[1] for p in pts[key]]
        g, b, a0 = {'Peak FLOP/s': (3.023523601, 0.2402566815, -479.7416946), 'DRAM bandwidth': (1.591025414, 0.1008385584, -201.8305106), 'Interconnect bandwidth': (1.414744219, 0.075338964, -150.9612684)}[name]
        ax.scatter(xs, ys, s=22, color=col, edgecolors='white', linewidths=0.6, zorder=3)
        fx = np.array([min(xs), max(xs)])
        ax.plot(fx, 10 ** (a0 + b * fx), color=col, lw=2.5, alpha=0.55, zorder=2)
        ly = {'flops': 4000000.0, 'dram': 500, 'interconnect': 12}[key]
        ax.text(2023.3, ly, f"{name}\n{ {'Peak FLOP/s': '3.0', 'DRAM bandwidth': '1.6', 'Interconnect bandwidth': '1.4'}[name]}× every 2 years", color=col, fontsize=11, fontweight='bold', va='center', ha='left')
    for label, (x, y), off in (('R10000', (1997.07, 1.0), (4, 8)), ('H100', (2022.7, 2920000.0), (-6, 4))):
        ax.annotate(label, (x, y), xytext=off, textcoords='offset points', ha='left' if off[0] > 0 else 'right', va='bottom', fontsize=10, color=C['text'])
    ax.set_yscale('log')
    ax.set_xlim(1996, 2023.2)
    ax.set_ylim(0.3, 30000000.0)
    ax.set_xticks([1996, 2002, 2008, 2014, 2020])
    ax.set_yticks([1, 100.0, 10000.0, 1000000.0])
    ax.set_yticklabels(['1×', '100×', '10,000×', '1,000,000×'])
    ax.minorticks_off()
    ax.grid(axis='y', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel("Growth from each\nseries' first point (log)")
    sp.save(fig, out / 'memory_wall_measured.svg')
memory_wall()
PEAK, BW = (312.0, 2.04)
RIDGE = PEAK / BW

def roofline(name, parts):
    xs = np.logspace(-1, 4, 400)
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.75))
    slope = xs * BW
    if 'slope' in parts:
        m = xs <= (RIDGE if 'ceiling' in parts else 10000.0)
        ax.plot(xs[m], slope[m], color=C['memory'], lw=3.5)
        ax.text(0.13, 60, 'Memory slope:\nintensity × 2.04 TB/s', color=C['memory'], fontsize=11, fontweight='bold', ha='left', va='top')
    if 'ceiling' in parts:
        m = xs >= RIDGE
        ax.plot(xs[m], [PEAK] * m.sum(), color=C['compute'], lw=3.5)
        ax.text(1300, PEAK * 0.7, 'Compute ceiling:\n312 TFLOP/s', color=C['compute'], fontsize=11, fontweight='bold', ha='center', va='top')
    if 'ridge' in parts:
        ax.axvline(RIDGE, color=C['neutral'], lw=1.2, ls='--')
        ax.text(RIDGE * 1.1, 0.25, 'Ridge\n153 FLOP/byte', color=C['neutral'], fontsize=10, va='bottom')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1, 10000.0)
    ax.set_ylim(0.15, 3000)
    ax.minorticks_off()
    ax.set_xticks([0.1, 1, 10, 100, 1000, 10000])
    ax.set_xticklabels(['0.1', '1', '10', '100', '1,000', '10,000'])
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(['1', '10', '100', '1,000'])
    ax.set_xlabel('Arithmetic intensity (FLOP per byte moved, log scale)')
    ax.set_ylabel('TFLOP/s (log)')
    sp.save(fig, out / f'{name}.svg')
roofline('roofline_step1', ['slope'])
roofline('roofline_step2', ['slope', 'ceiling'])
roofline('roofline_step3', ['slope', 'ceiling', 'ridge'])

def roofline_col(name, point=None, peak2=False, arrow=False, qmark=False):
    """The same A100 roofline, drawn at the size of a half-slide column
    (about 3.3 x 2.05 in) so its 12 pt labels are not shrunk. The slope label
    is left off: the full-width build on the previous slide teaches it."""
    xs = np.logspace(-1, 4, 400)
    fig, ax = sp.plt.subplots(figsize=(3.3, 2.05))
    m = xs <= RIDGE
    ax.plot(xs[m], xs[m] * BW, color=C['memory'], lw=3.5)
    ax.plot(xs[~m], [PEAK] * (~m).sum(), color=C['compute'], lw=3.5)
    ax.text(1500, PEAK * 0.72, '312 TFLOP/s', color=C['compute'], fontsize=12, fontweight='bold', ha='center', va='top')
    ax.axvline(RIDGE, color=C['neutral'], lw=1.2, ls='--')
    ax.text(RIDGE * 1.15, 0.22, 'Ridge\n153', color=C['neutral'], fontsize=12, va='bottom')
    if peak2:
        r2 = 2 * PEAK / BW
        ax.plot(xs[xs <= r2], xs[xs <= r2] * BW, color=C['memory'], lw=1.5, ls=':')
        ax.plot(xs[xs >= r2], [2 * PEAK] * (xs >= r2).sum(), color=C['compute'], lw=2, ls=':')
        ax.text(2200, 2 * PEAK * 1.2, '2× FLOP/s', color=C['compute'], fontsize=12, ha='center', va='bottom')
    if point:
        label, x, ck, dx, dy, ha, va = point
        y = min(PEAK, x * BW)
        ax.plot([x], [y], 'o', ms=11, color=C[ck], mec='white', mew=1.5, zorder=5)
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords='offset points', ha=ha, va=va, fontsize=12, fontweight='bold', color=C[ck])
    if arrow:
        ax.annotate('', xy=(256, PEAK), xytext=(1.3, 1.3 * BW), arrowprops=dict(arrowstyle='-|>', color=C['neutral'], lw=2))
        ax.text(4, 6, 'batch B', color=C['neutral'], fontsize=12, fontweight='bold', ha='left', va='top')
    if qmark:
        ax.text(1.0, 6, '???', color=C['danger'], fontsize=18, fontweight='bold', ha='center')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1, 10000.0)
    ax.set_ylim(0.15, 3000)
    ax.minorticks_off()
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(['1', '10', '100', '1,000'], fontsize=12)
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(['1', '10', '100', '1,000'], fontsize=12)
    ax.set_xlabel('FLOP per byte (log scale)', fontsize=12)
    ax.set_ylabel('TFLOP/s (log scale)', fontsize=12)
    sp.save(fig, out / f'{name}.svg')
roofline_col('roofline_resnet', point=('ResNet-50,\nbatch 1: 146', 146.0, 'memory', -10, 4, 'right', 'bottom'))
roofline_col('roofline_decode_q', peak2=True, qmark=True)
roofline_col('roofline_decode_a', point=('Decode,\nbatch 1: ≈1', 1.0, 'danger', 6, -8, 'left', 'top'), peak2=True, arrow=True)

def tpu_roofline():
    peak, ridge = (mag(Q_(92, 'TOPS'), 'TOPS'), 1350.0)
    xs = np.logspace(1, 4, 400)
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.75))
    m = xs <= ridge
    ax.plot(xs[m], xs[m] * peak / ridge, color=C['memory'], lw=3.5)
    ax.plot(xs[~m], [peak] * (~m).sum(), color=C['compute'], lw=3.5)
    ax.axvline(ridge, color=C['neutral'], lw=1.2, ls='--')
    ax.text(ridge * 0.92, 1.25, 'Ridge\n1,350', color=C['neutral'], fontsize=11, ha='right', va='bottom')
    ax.text(5200, peak * 1.15, '92 TOPS peak', color=C['compute'], fontsize=11, fontweight='bold', ha='center', va='bottom')
    offs = {'MLP0': (9, -3, 'left', 'top'), 'MLP1': (-10, 4, 'right', 'bottom'), 'LSTM0': (-11, 2, 'right', 'bottom'), 'LSTM1': (9, -3, 'left', 'top'), 'CNN0': (4, -9, 'left', 'top'), 'CNN1': (10, 0, 'left', 'center')}
    for a in [{'app': 'MLP0', 'family': 'MLP', 'intensity': 200.0, 'measured_tops': 12.3, 'roof_tops': 13.62962963, 'roof_fraction': 0.9024456522, 'bound': 'memory'}, {'app': 'MLP1', 'family': 'MLP', 'intensity': 168.0, 'measured_tops': 9.7, 'roof_tops': 11.44888889, 'roof_fraction': 0.8472437888, 'bound': 'memory'}, {'app': 'LSTM0', 'family': 'LSTM', 'intensity': 64.0, 'measured_tops': 3.7, 'roof_tops': 4.361481481, 'roof_fraction': 0.8483355978, 'bound': 'memory'}, {'app': 'LSTM1', 'family': 'LSTM', 'intensity': 96.0, 'measured_tops': 2.8, 'roof_tops': 6.542222222, 'roof_fraction': 0.4279891304, 'bound': 'memory'}, {'app': 'CNN0', 'family': 'CNN', 'intensity': 2888.0, 'measured_tops': 86.0, 'roof_tops': 92, 'roof_fraction': 0.9347826087, 'bound': 'compute'}, {'app': 'CNN1', 'family': 'CNN', 'intensity': 1750.0, 'measured_tops': 14.1, 'roof_tops': 92, 'roof_fraction': 0.1532608696, 'bound': 'compute'}]:
        name, x, y = (a['app'], a['intensity'], a['measured_tops'])
        ck = 'memory' if x < ridge else 'compute'
        ax.plot([x], [y], 'o', ms=10, color=C[ck], mec='white', mew=1.5, zorder=5)
        dx, dy, ha, va = offs[name]
        ax.annotate(f'{name} {y:g}', (x, y), xytext=(dx, dy), textcoords='offset points', ha=ha, va=va, fontsize=11, fontweight='bold', color=C[ck])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(10, 10000.0)
    ax.set_ylim(1, 300)
    ax.minorticks_off()
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(['10', '100', '1,000', '10,000'])
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(['1', '10', '100'])
    ax.set_xlabel('Operations per byte of weights read (log scale)')
    ax.set_ylabel('Measured TOPS (log)')
    sp.save(fig, out / 'tpu_roofline.svg')
tpu_roofline()
sp.hbars_steps(out / 'voice_upload', [('Every microphone streams', mag(Q_(32000000000000.0, 'B / s'), 'B/s'), 'danger', '32 TB/s'), ('Wake word on the device', mag(Q_(37037037040.0, 'B / s'), 'B/s'), 'data', '≈37 GB/s: 860× less')], 'B/s', xlabel='Audio sent to the cloud (bytes per second, log scale)', size=sp.SHORT)

def trends():
    rows = load_series('vol1/02_system_paradigms/data/microprocessor_trends.csv').entries
    series = [('transistors', 'Transistors (K)', C['danger'], 'o', 30000000.0), ('specint', 'Single-thread\nperf. (SPECint ×10³)', C['memory'], 's', 120000.0), ('frequency', 'Clock (MHz)', C['data'], '^', 3500.0), ('watts', 'Power (W)', C['compute'], 'D', 250.0), ('cores', 'Cores', C['neutral'], 'v', 16.0)]
    fig, ax = sp.plt.subplots(figsize=(3.3, 2.45))
    ax.axvspan(2004.5, 2006.5, color=C['grid'], alpha=0.6, zorder=0)
    ax.text(2005.5, 250000000.0, 'End of Dennard\nscaling', ha='center', va='bottom', fontsize=10, color=C['text'])
    for key, label, col, mk, ly in series:
        pts = [(r['year'], r['value']) for r in rows if r['series'] == key]
        ax.scatter([x for x, _ in pts], [y for _, y in pts], s=9, color=col, marker=mk, alpha=0.8, edgecolors='none')
        ax.text(2024, ly, label, color=col, fontsize=10, fontweight='bold', va='center', ha='left')
    ax.set_yscale('log')
    ax.set_xlim(1970, 2023)
    ax.set_ylim(0.5, 200000000.0)
    ax.set_xticks([1970, 1990, 2010])
    ax.set_yticks([1, 100.0, 10000.0, 1000000.0, 100000000.0])
    ax.minorticks_off()
    ax.set_ylabel('Value (log scale)')
    ax.grid(axis='y', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / 'microprocessor_trends.svg')
trends()

def energy(step):
    J = lambda key, unit: mag(Q_(_NUMS[key], unit), 'J')
    rows = [('TinyML: keyword spotting', J('e_kws_uj', 'uJ'), 'compute', '≈10 μJ'), ('Measured: KWS, Syntiant NDP9120', J('tiny_best_uj', 'uJ'), 'compute', '31.5 μJ (MLPerf Tiny)'), ('Mobile: MobileNetV2, phone', J('e_mobile_mj', 'mJ'), 'memory', '≈2.6 mJ'), ('Mobile: send 1 MB by radio', J('upload_mj', 'mJ'), 'danger', '≈100 mJ: 39× the phone'), ('Edge: ResNet-50, Jetson', J('e_edge_mj', 'mJ'), 'data', '≈15.9 mJ'), ('Cloud: ResNet-50, A100', J('e_cloud_mj', 'mJ'), 'danger', '≈75.5 mJ'), ('Cloud: LLM query (anchor)', J('e_llm_kj', 'kJ'), 'danger', '≈1 kJ')]
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.95))
    n = len(rows)
    for i, (label, v, ck, txt) in enumerate(rows):
        y = n - 1 - i
        send = label.startswith('Mobile: send')
        meas = label.startswith('Measured')
        shown = step == 2 or not send
        alpha = 1.0 if shown else 0.0
        if send:
            ax.barh(y, v, left=1e-06, height=0.62, facecolor='white', edgecolor=C[ck], lw=2.2, ls='--', alpha=alpha)
        elif meas:
            ax.barh(y, v, left=1e-06, facecolor=C['compute_l'], edgecolor=C[ck], lw=1.5, hatch='///', height=0.62)
        else:
            ax.barh(y, v, left=1e-06, color=C[ck], height=0.62)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12, fontweight='bold', color=C['danger'] if send else C['text'], alpha=alpha)
        ax.text(v * 2.2, y, txt, va='center', fontsize=12, fontweight='bold', color=C[ck], alpha=alpha)
    ynpu = n - 3
    npu = J('npu_mj', 'mJ')
    ax.plot([npu, npu], [ynpu - 0.31, ynpu + 0.31], color='white', lw=2.5, alpha=1.0 if step == 2 else 0.0)
    ax.text(0.5, ynpu, '(tick: NPU alone ≈0.1 mJ)', color=C['memory'], fontsize=12, fontweight='bold', ha='left', va='center', alpha=1.0 if step == 2 else 0.0)
    ax.annotate('', xy=(400000.0, n - 1), xytext=(400000.0, 0), arrowprops=dict(arrowstyle='<->', color=C['danger'], lw=2))
    ax.text(600000.0, (n - 1) / 2, '100,000,000×', color=C['danger'], fontsize=12, fontweight='bold', rotation=90, va='center', ha='left')
    ax.set_xscale('log')
    ax.set_xlim(1e-06, 1000000.0)
    ax.set_xticks([1e-06, 0.0001, 0.01, 1, 100.0, 10000.0, 1000000.0])
    ax.tick_params(axis='x', labelsize=13)
    ax.minorticks_off()
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('Energy per inference (joules, log scale)')
    sp.save(fig, out / f'energy_per_inference_step{step}.svg')
for k in (1, 2):
    energy(k)

def doorbell(step):
    budget, pct = ('db_sla_ms', 'db_sla_pct') if step == 1 else ('db_strict_ms', 'db_strict_pct')
    lat = _NUMS[pct]
    lat_txt = f'101 ms of {_VALS[budget]} ms: {_VALS[pct]}%'
    flash = 11.9
    rows = [('Latency\nper frame', lat, 'danger' if lat > 100 else 'data', lat_txt), ('Weights in\n4.2 MB flash', flash, 'data', '11.9%')]
    fig, ax = sp.plt.subplots(figsize=(4.0, 2.3))
    for i, (label, v, ck, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=C[ck], height=0.55)
        ax.text(-0.02, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
        if i == 0:
            ax.text(2, y + 0.34, txt, va='bottom', ha='left', fontsize=12, fontweight='bold', color=C[ck], zorder=4, bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
        else:
            ax.text(v + 4, y, txt, va='center', ha='left', fontsize=12, fontweight='bold', color=C[ck])
    ax.axvline(100, color=C['neutral'], lw=2, ls='--')
    ax.set_xlim(0, 230)
    ax.set_ylim(-0.5, 1.9)
    ax.set_yticks([])
    ax.set_xticks([0, 100, 200])
    ax.set_xticklabels(['0', '100%\nbudget', '200%'], fontsize=12)
    ax.spines['left'].set_visible(False)
    sp.save(fig, out / f'doorbell_step{step}.svg')
for k in (1, 2):
    doorbell(k)
print('figures written')
