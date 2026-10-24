#!/usr/bin/env python3
"""Data plots for Lecture 1 (Introduction).

Every plotted value and every printed label comes from MLSysIM through
calcs.py (val() and num() give the formatted string and the float the slides
print), or from the measured series MLSysIM holds:
  - Google Flu Trends against the CDC count: slides/vol1/01_introduction/images/data/google_flu_trends_cdc_ilinet.csv
    (Delphi Epidata API), seasons and peaks from Scenarios.GoogleFluTrendsDrift.
  - GPU-days: the registry's AlexNet, GPT-3, and GPT-4-class training runs.
  - Iron-law bars: Scenarios.ResNet50Batch64StepH100 and Gpt2XlDecodeStepH100
    (MLSysIM's roofline engine on one H100); the second bar of each pair
    applies the scenario's bandwidth multiple to the memory term only.
  - Degradation sketch: the recall points of Scenarios.PedestrianRecallWinterShift;
    the "better model" intercept, the flatter slope, and the winter position
    are illustrative layout, labeled "Illustrative" on the plot.
  - Amdahl pipeline: Scenarios.ModelSpeedupRequestPath.
  - ImageNetV2 scatter and fit: slides/vol1/01_introduction/images/data/recht_imagenetv2_shift.csv (Recht et al. 2019).
  - MLPerf: slides/vol1/01_introduction/images/data/mlperf_inference_v4_llama2_70b_offline.csv and the H100 and
    H200 registry entries.

Run from this folder with the MLSysBook venv (it imports mlsysim):
  python3 make_figs.py
"""
from types import SimpleNamespace
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
_VALS = {'gpt3_ops_mant': '3.14', 'gpt3_ops_exp': '23', 'n_gpu': '1,024', 'a100_tflops': '312', 'eta_base': '0.45', 'eta_base_pct': '45', 'eta_opt_pct': '60', 'days_base': '25', 'days_opt': '19', 'gft_peak': '10.6', 'cdc_peak': '6.1', 'gft_peak_ratio': '1.7', 'gft_week_ratio': '2.3', 'gft_miss_weeks': '100', 'gft_window_weeks': '108', 'gft_stable_pct': '13', 'gft_ratio_first_drift': '1.27', 'gft_ratio_peak': '1.66', 'img_w': '224', 'img_h': '224', 'img_channels_word': 'three', 'pixel_levels': '256', 'input_elements': '150,528', 'input_digits': '362,508', 'imagenet_val': '50,000', 'alexnet_params_m': '60', 'fp32_bytes': '4', 'alexnet_weights_mb': '240', 'gtx580_gb': '3', 'alexnet_top5': '15.3', 'runner_up_top5': '26.2', 'imagenet_train_m': '1.3', 'deep_blue_chips': '480', 'deep_blue_low_m': '100', 'deep_blue_high_m': '200', 'alphago_days_word': 'three', 'alphago_wins': '100', 'alphago_losses': '0', 'alexnet_gpus_word': 'two', 'alexnet_days_word': 'six', 'alexnet_gpu_days': '12', 'gpt3_gpu_years': '405', 'gpt3_gpu_days': '148,000', 'gpt4_gpu_days_m': '2.5', 'gpt4_gpus': '25,000', 'gpt4_days': '100', 'chin_k': '6', 'gpt3_params_b': '175', 'gpt3_tokens_b': '300', 'gpt3_6nd_mant': '3.15', 'gpt3_6nd_exp': '23', 'recall_before': '95', 'recall_after': '85', 'recht_models': '67', 'recht_networks': '64', 'recht_baselines_word': 'three', 'recht_net_gap_min': '10.3', 'recht_net_gap_max': '13.6', 'recht_gap_min_round': '6', 'recht_gap_max_round': '14', 'recht_slope': '0.97', 'recht_icpt_neg': '9.5', 'skew_days': '412', 'skew_seconds': '35,596,800', 'skew_seconds_m': '35.6', 'skew_ratio': '86,400', 'nap_gb': '2', 'nap_bw': '100', 'nap_gflop': '1', 'nap_rate': '100', 'nap_over': '5', 'nap_bw_gain': '2', 'nap_mem': '20', 'nap_comp': '10', 'nap_total': '35', 'nap_mem2': '10', 'nap_total2': '25', 'nap_naive': '17.5', 'nap_cut_pct': '29', 'nap_naive_cut_pct': '50', 'nap_overlap': '25', 'nap_overlap2': '15', 'bud_mem': '2', 'bud_comp': '8', 'bud_over': '1', 'bud_total2': '7', 'bud_naive': '5.5', 'rn_mem': '0.11', 'rn_comp': '1.06', 'rn_over': '0.51', 'rn_mem2': '0.06', 'rn_total': '1.68', 'rn_total2': '1.63', 'rn_cut_pct': '3', 'rn_ai': '1,387', 'g2_mem': '0.99', 'g2_comp': '0.006', 'g2_over': '0.49', 'g2_mem2': '0.49', 'g2_total': '1.48', 'g2_total2': '0.99', 'g2_cut_pct': '33', 'g2_ai': '0.9', 'rn_traffic_mb': '378', 'rn_batch': '64', 'rn_overlapped': '1.57', 'g2_params_b': '1.5', 'g2_weights_gb': '3', 'sn_eta': '0.5', 'bw_gain': '2', 'launch_us': '10', 'mlperf_h100_tps': '21805.8', 'mlperf_h200_tps': '27738.3', 'mlperf_h100_tps_c': '21,806', 'mlperf_h200_tps_c': '27,738', 'mlperf_cts_tps_c': '31,712', 'mlperf_h200_speedup': '1.27', 'mlperf_cts_speedup': '1.45', 'mlperf_gpus_word': 'eight', 'h_tflops': '989', 'compute_ratio': '1.00', 'h200_bw': '4.8', 'h100_bw': '3.35', 'bw_ratio': '1.43', 'h200_gb': '141', 'h100_gb': '80', 'cap_ratio': '1.76', 'h_tdp': '700', 'gpt2_params_m': '124', 'gpt2_layers': '12', 'addmm_calls': '48', 'mm_calls': '1', 'block_mb': '340', 'head_mb': '154', 'addmm_ms': '4.47', 'mm_ms': '1.96', 'addmm_gbs': '76', 'mm_gbs': '79', 'matmul_share': '57', 'step_self_share': '26', 'am_pre': '60', 'am_model': '45', 'am_post': '25', 'am_total': '130', 'am_model2': '15', 'am_total2': '100', 'am_untouched': '85', 'am_speedup': '3', 'am_speedup_word': 'three', 'am_cut_pct': '23', 'am_model_cut_pct': '67', 'am_share_pct': '35', 'am_overall': '1.3', 'am_bound': '1.5', 'algo_mult': '44', 'growth_exp': '8', 'growth_exp_word': 'eight', 'growth_doubling_mo': '5'}
_NUMS = {'gpt3_ops_mant': 3.14, 'gpt3_ops_exp': 23.0, 'n_gpu': 1024.0, 'a100_tflops': 312.0, 'eta_base': 0.45, 'eta_base_pct': 45.0, 'eta_opt_pct': 60.0, 'days_base': 25.0, 'days_opt': 19.0, 'gft_peak': 10.6, 'cdc_peak': 6.1, 'gft_peak_ratio': 1.7, 'gft_week_ratio': 2.3, 'gft_miss_weeks': 100.0, 'gft_window_weeks': 108.0, 'gft_stable_pct': 13.0, 'gft_ratio_first_drift': 1.27, 'gft_ratio_peak': 1.66, 'img_w': 224.0, 'img_h': 224.0, 'pixel_levels': 256.0, 'input_elements': 150528.0, 'input_digits': 362508.0, 'imagenet_val': 50000.0, 'alexnet_params_m': 60.0, 'fp32_bytes': 4.0, 'alexnet_weights_mb': 240.0, 'gtx580_gb': 3.0, 'alexnet_top5': 15.3, 'runner_up_top5': 26.2, 'imagenet_train_m': 1.3, 'deep_blue_chips': 480.0, 'deep_blue_low_m': 100.0, 'deep_blue_high_m': 200.0, 'alphago_wins': 100.0, 'alphago_losses': 0.0, 'alexnet_gpu_days': 12.0, 'gpt3_gpu_years': 405.0, 'gpt3_gpu_days': 148000.0, 'gpt4_gpu_days_m': 2.5, 'gpt4_gpus': 25000.0, 'gpt4_days': 100.0, 'chin_k': 6.0, 'gpt3_params_b': 175.0, 'gpt3_tokens_b': 300.0, 'gpt3_6nd_mant': 3.15, 'gpt3_6nd_exp': 23.0, 'recall_before': 95.0, 'recall_after': 85.0, 'recht_models': 67.0, 'recht_networks': 64.0, 'recht_net_gap_min': 10.3, 'recht_net_gap_max': 13.6, 'recht_gap_min_round': 6.0, 'recht_gap_max_round': 14.0, 'recht_slope': 0.97, 'recht_icpt_neg': 9.5, 'skew_days': 412.0, 'skew_seconds': 35596800.0, 'skew_seconds_m': 35.6, 'skew_ratio': 86400.0, 'nap_gb': 2.0, 'nap_bw': 100.0, 'nap_gflop': 1.0, 'nap_rate': 100.0, 'nap_over': 5.0, 'nap_bw_gain': 2.0, 'nap_mem': 20.0, 'nap_comp': 10.0, 'nap_total': 35.0, 'nap_mem2': 10.0, 'nap_total2': 25.0, 'nap_naive': 17.5, 'nap_cut_pct': 29.0, 'nap_naive_cut_pct': 50.0, 'nap_overlap': 25.0, 'nap_overlap2': 15.0, 'bud_mem': 2.0, 'bud_comp': 8.0, 'bud_over': 1.0, 'bud_total2': 7.0, 'bud_naive': 5.5, 'rn_mem': 0.11, 'rn_comp': 1.06, 'rn_over': 0.51, 'rn_mem2': 0.06, 'rn_total': 1.68, 'rn_total2': 1.63, 'rn_cut_pct': 3.0, 'rn_ai': 1387.0, 'g2_mem': 0.99, 'g2_comp': 0.006, 'g2_over': 0.49, 'g2_mem2': 0.49, 'g2_total': 1.48, 'g2_total2': 0.99, 'g2_cut_pct': 33.0, 'g2_ai': 0.9, 'rn_traffic_mb': 378.0, 'rn_batch': 64.0, 'rn_overlapped': 1.57, 'g2_params_b': 1.5, 'g2_weights_gb': 3.0, 'sn_eta': 0.5, 'bw_gain': 2.0, 'launch_us': 10.0, 'mlperf_h100_tps': 21805.8, 'mlperf_h200_tps': 27738.3, 'mlperf_h100_tps_c': 21806.0, 'mlperf_h200_tps_c': 27738.0, 'mlperf_cts_tps_c': 31712.0, 'mlperf_h200_speedup': 1.27, 'mlperf_cts_speedup': 1.45, 'h_tflops': 989.0, 'compute_ratio': 1.0, 'h200_bw': 4.8, 'h100_bw': 3.35, 'bw_ratio': 1.43, 'h200_gb': 141.0, 'h100_gb': 80.0, 'cap_ratio': 1.76, 'h_tdp': 700.0, 'gpt2_params_m': 124.0, 'gpt2_layers': 12.0, 'addmm_calls': 48.0, 'mm_calls': 1.0, 'block_mb': 340.0, 'head_mb': 154.0, 'addmm_ms': 4.47, 'mm_ms': 1.96, 'addmm_gbs': 76.0, 'mm_gbs': 79.0, 'matmul_share': 57.0, 'step_self_share': 26.0, 'am_pre': 60.0, 'am_model': 45.0, 'am_post': 25.0, 'am_total': 130.0, 'am_model2': 15.0, 'am_total2': 100.0, 'am_untouched': 85.0, 'am_speedup': 3.0, 'am_cut_pct': 23.0, 'am_model_cut_pct': 67.0, 'am_share_pct': 35.0, 'am_overall': 1.3, 'am_bound': 1.5, 'algo_mult': 44.0, 'growth_exp': 8.0, 'growth_doubling_mo': 5.0}
import sys
from datetime import date, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
out = Path(__file__).parent
C = sp.C

def epiweek_to_date(ew):
    """Approximate start date of an MMWR epiweek (good to a few days)."""
    y, w = divmod(ew, 100)
    jan4 = date(y, 1, 4)
    start = jan4 - timedelta(days=(jan4.weekday() + 1) % 7)
    return start + timedelta(weeks=w - 1)
gft, cdc = ({200340: 0.902, 200341: 0.952, 200342: 1.092, 200343: 1.209, 200344: 1.249, 200345: 1.374, 200346: 1.702, 200347: 2.148, 200348: 2.968, 200349: 3.524, 200350: 5.389, 200351: 7.276, 200352: 8.196, 200353: 5.25, 200401: 3.154, 200402: 2.057, 200403: 1.616, 200404: 1.396, 200405: 1.227, 200406: 1.186, 200407: 1.162, 200408: 1.118, 200409: 0.989, 200410: 0.919, 200411: 0.846, 200412: 0.883, 200413: 0.808, 200414: 0.831, 200415: 0.796, 200416: 0.719, 200417: 0.722, 200418: 0.705, 200419: 0.68, 200420: 0.663, 200421: 0.649, 200422: 0.65, 200423: 0.653, 200424: 0.631, 200425: 0.623, 200426: 0.602, 200427: 0.629, 200428: 0.597, 200429: 0.576, 200430: 0.587, 200431: 0.593, 200432: 0.599, 200433: 0.618, 200434: 0.676, 200435: 0.701, 200436: 0.81, 200437: 0.826, 200438: 0.898, 200439: 0.922, 200440: 1.139, 200441: 1.408, 200442: 1.63, 200443: 1.767, 200444: 1.804, 200445: 1.819, 200446: 1.761, 200447: 1.595, 200448: 1.8, 200449: 1.782, 200450: 1.85, 200451: 2.206, 200452: 2.715, 200501: 2.673, 200502: 2.459, 200503: 2.562, 200504: 2.886, 200505: 3.429, 200506: 4.037, 200507: 4.065, 200508: 3.54, 200509: 2.875, 200510: 2.554, 200511: 2.23, 200512: 2.024, 200513: 1.749, 200514: 1.421, 200515: 1.186, 200516: 1.083, 200517: 1.017, 200518: 0.959, 200519: 0.902, 200520: 0.85, 200521: 0.85, 200522: 0.826, 200523: 0.751, 200524: 0.761, 200525: 0.74, 200526: 0.684, 200527: 0.711, 200528: 0.712, 200529: 0.697, 200530: 0.695, 200531: 0.693, 200532: 0.71, 200533: 0.739, 200534: 0.787, 200535: 0.772, 200536: 0.881, 200537: 1.053, 200538: 1.068, 200539: 1.177, 200540: 1.364, 200541: 1.504, 200542: 1.532, 200543: 1.591, 200544: 1.858, 200545: 1.778, 200546: 1.811, 200547: 1.64, 200548: 1.808, 200549: 1.806, 200550: 2.05, 200551: 2.494, 200552: 2.915, 200601: 2.522, 200602: 2.098, 200603: 1.998, 200604: 2.137, 200605: 2.155, 200606: 2.221, 200607: 2.437, 200608: 2.567, 200609: 2.564, 200610: 2.457, 200611: 2.17, 200612: 2.057, 200613: 1.896, 200614: 1.574, 200615: 1.363, 200616: 1.191, 200617: 1.09, 200618: 1.06, 200619: 1.003, 200620: 0.939, 200621: 0.927, 200622: 0.884, 200623: 0.818, 200624: 0.775, 200625: 0.729, 200626: 0.706, 200627: 0.735, 200628: 0.717, 200629: 0.695, 200630: 0.695, 200631: 0.677, 200632: 0.681, 200633: 0.717, 200634: 0.741, 200635: 0.76, 200636: 0.861, 200637: 0.965, 200638: 1.101, 200639: 1.133, 200640: 1.178, 200641: 1.23, 200642: 1.297, 200643: 1.445, 200644: 1.577, 200645: 1.636, 200646: 1.758, 200647: 1.595, 200648: 1.889, 200649: 1.886, 200650: 2.079, 200651: 2.305, 200652: 2.547, 200701: 2.419, 200702: 2.082, 200703: 1.968, 200704: 2.182, 200705: 2.409, 200706: 2.806, 200707: 2.97, 200708: 2.941, 200709: 2.641, 200710: 2.424, 200711: 2.086, 200712: 1.893, 200713: 1.604, 200714: 1.375, 200715: 1.372, 200716: 1.243, 200717: 1.08, 200718: 0.98, 200719: 0.931, 200720: 0.878, 200721: 0.85, 200722: 0.84, 200723: 0.774, 200724: 0.764, 200725: 0.748, 200726: 0.72, 200727: 0.708, 200728: 0.701, 200729: 0.705, 200730: 0.699, 200731: 0.705, 200732: 0.723, 200733: 0.744, 200734: 0.763, 200735: 0.809, 200736: 0.903, 200737: 0.957, 200738: 1.041, 200739: 1.082, 200740: 1.18, 200741: 1.21, 200742: 1.326, 200743: 1.377, 200744: 1.512, 200745: 1.594, 200746: 1.644, 200747: 1.465, 200748: 1.7, 200749: 1.632, 200750: 1.618, 200751: 1.709, 200752: 1.954, 200801: 2.278, 200802: 2.199, 200803: 2.23, 200804: 2.921, 200805: 3.88, 200806: 4.928, 200807: 5.352, 200808: 5.805, 200809: 5.05, 200810: 4.059, 200811: 3.229, 200812: 2.592, 200813: 2.172, 200814: 1.776, 200815: 1.508, 200816: 1.331, 200817: 1.18, 200818: 1.046, 200819: 0.947, 200820: 0.922, 200821: 0.873, 200822: 0.828, 200823: 0.77, 200824: 0.727, 200825: 0.713, 200826: 0.662, 200827: 0.646, 200828: 0.665, 200829: 0.638, 200830: 0.627, 200831: 0.634, 200832: 0.648, 200833: 0.684, 200834: 0.71, 200835: 0.771, 200836: 0.903, 200837: 1.009, 200838: 1.114, 200839: 1.221, 200840: 1.275, 200841: 1.334, 200842: 1.404, 200843: 1.543, 200844: 1.659, 200845: 1.747, 200846: 1.953, 200847: 2.194, 200848: 1.969, 200849: 2.045, 200850: 1.938, 200851: 1.87, 200852: 1.942, 200853: 2.1, 200901: 2.133, 200902: 1.971, 200903: 2.191, 200904: 2.555, 200905: 3.116, 200906: 3.484, 200907: 3.499, 200908: 3.447, 200909: 3.22, 200910: 2.711, 200911: 2.348, 200912: 2.021, 200913: 1.699, 200914: 1.462, 200915: 1.378, 200916: 1.381, 200917: 2.129, 200918: 1.885, 200919: 1.818, 200920: 1.892, 200921: 1.76, 200922: 1.743, 200923: 1.907, 200924: 1.895, 200925: 1.686, 200926: 1.49, 200927: 1.337, 200928: 1.278, 200929: 1.257, 200930: 1.324, 200931: 1.276, 200932: 1.302, 200933: 1.481, 200934: 1.915, 200935: 2.498, 200936: 3.361, 200937: 3.79, 200938: 4.254, 200939: 4.926, 200940: 5.424, 200941: 6.037, 200942: 7.112, 200943: 6.533, 200944: 5.722, 200945: 4.468, 200946: 3.587, 200947: 2.655, 200948: 2.56, 200949: 2.19, 200950: 2.032, 200951: 2.055, 200952: 2.199, 201001: 2.169, 201002: 1.916, 201003: 1.86, 201004: 1.861, 201005: 1.91, 201006: 1.944, 201007: 2.0, 201008: 1.968, 201009: 1.888, 201010: 1.793, 201011: 1.619, 201012: 1.511, 201013: 1.35, 201014: 1.257, 201015: 1.123, 201016: 1.11, 201017: 1.149, 201018: 1.088, 201019: 1.06, 201020: 1.042, 201021: 0.989, 201022: 0.965, 201023: 0.858, 201024: 0.808, 201025: 0.786, 201026: 0.765, 201027: 0.76, 201028: 0.752, 201029: 0.737, 201030: 0.744, 201031: 0.757, 201032: 0.754, 201033: 0.804, 201034: 0.846, 201035: 0.924, 201036: 1.03, 201037: 1.187, 201038: 1.315, 201039: 1.335, 201040: 1.43, 201041: 1.421, 201042: 1.566, 201043: 1.596, 201044: 1.685, 201045: 1.758, 201046: 1.864, 201047: 1.661, 201048: 1.96, 201049: 1.943, 201050: 2.113, 201051: 2.512, 201052: 2.917, 201101: 2.978, 201102: 2.726, 201103: 3.089, 201104: 3.554, 201105: 3.601, 201106: 3.784, 201107: 4.077, 201108: 3.565, 201109: 2.98, 201110: 2.482, 201111: 2.076, 201112: 1.776, 201113: 1.589, 201114: 1.405, 201115: 1.228, 201116: 1.126, 201117: 1.055, 201118: 1.013, 201119: 0.965, 201120: 0.937, 201121: 0.882, 201122: 0.859, 201123: 0.811, 201124: 0.744, 201125: 0.714, 201126: 0.651, 201127: 0.639, 201128: 0.626, 201129: 0.61, 201130: 0.604, 201131: 0.597, 201132: 0.616, 201133: 0.634, 201134: 0.717, 201135: 0.821, 201136: 0.951, 201137: 1.127, 201138: 1.343, 201139: 1.521, 201140: 1.587, 201141: 1.64, 201142: 1.608, 201143: 1.638, 201144: 1.788, 201145: 1.797, 201146: 1.88, 201147: 1.785, 201148: 2.033, 201149: 1.981, 201150: 2.094, 201151: 2.37, 201152: 2.741, 201201: 2.86, 201202: 2.5, 201203: 2.411, 201204: 2.475, 201205: 2.507, 201206: 2.348, 201207: 2.501, 201208: 2.71, 201209: 2.519, 201210: 2.586, 201211: 2.556, 201212: 2.162, 201213: 1.936, 201214: 1.801, 201215: 1.799, 201216: 1.785, 201217: 1.729, 201218: 1.625, 201219: 1.651, 201220: 1.624, 201221: 1.537, 201222: 1.428, 201223: 1.329, 201224: 1.265, 201225: 1.136, 201226: 0.966, 201227: 0.911, 201228: 0.939, 201229: 0.897, 201230: 0.869, 201231: 0.869, 201232: 0.889, 201233: 0.977, 201234: 1.095, 201235: 1.199, 201236: 1.374, 201237: 1.543, 201238: 1.737, 201239: 1.895, 201240: 2.018, 201241: 2.325, 201242: 2.391, 201243: 2.369, 201244: 2.349, 201245: 2.492, 201246: 2.913, 201247: 3.04, 201248: 3.641, 201249: 4.427, 201250: 5.132, 201251: 6.359, 201252: 7.904, 201301: 8.618, 201302: 10.112, 201303: 10.555, 201304: 9.408, 201305: 7.209, 201306: 5.541, 201307: 4.555, 201308: 4.24, 201309: 3.784, 201310: 3.504, 201311: 3.242, 201312: 2.894, 201313: 2.673, 201314: 2.482, 201315: 2.121, 201316: 1.747, 201317: 1.681, 201318: 1.534, 201319: 1.438, 201320: 1.286, 201321: 1.165, 201322: 1.149, 201323: 1.063, 201324: 1.024, 201325: 0.997, 201326: 0.966, 201327: 0.925, 201328: 0.941, 201329: 0.907, 201330: 0.956, 201331: 0.85, 201332: 0.869, 201333: 0.907, 201334: 0.952, 201335: 0.977, 201336: 1.085, 201337: 1.154, 201338: 1.259, 201339: 1.35, 201340: 1.363, 201341: 1.405, 201342: 1.354, 201343: 1.445, 201344: 1.46, 201345: 1.56, 201346: 1.628, 201347: 1.661, 201348: 1.708, 201349: 2.043, 201350: 2.238, 201351: 3.196, 201352: 4.625, 201401: 5.008, 201402: 4.691, 201403: 4.486, 201404: 4.022, 201405: 3.472, 201406: 2.91, 201407: 2.496, 201408: 2.228, 201409: 1.986, 201410: 1.873, 201411: 1.773, 201412: 1.738, 201413: 1.75, 201414: 1.778, 201415: 1.771, 201416: 1.607, 201417: 1.53, 201418: 1.467, 201419: 1.415, 201420: 1.339, 201421: 1.289, 201422: 1.318, 201423: 1.155, 201424: 1.071, 201425: 1.054, 201426: 0.995, 201427: 0.99, 201428: 0.964, 201429: 1.028, 201430: 1.008, 201431: 0.793, 201432: 0.825, 201433: 0.762, 201434: 0.808, 201435: 0.827, 201436: 0.948, 201437: 1.096, 201438: 1.229, 201439: 1.164, 201440: 1.192, 201441: 1.291, 201442: 1.352, 201443: 1.349, 201444: 1.224, 201445: 1.374, 201446: 1.379, 201447: 1.549, 201448: 1.784, 201449: 2.287, 201450: 3.046, 201451: 5.234, 201452: 6.911, 201453: 6.22, 201501: 4.647, 201502: 4.303, 201503: 4.58, 201504: 3.998, 201505: 3.498, 201506: 3.445, 201507: 2.956, 201508: 2.671, 201509: 2.42, 201510: 2.467, 201511: 2.123, 201512: 2.015, 201513: 1.901, 201514: 1.651, 201515: 1.56, 201516: 1.195, 201517: 1.353, 201518: 1.388, 201519: 1.144, 201520: 1.164, 201521: 1.211, 201522: 1.143, 201523: 1.15, 201524: 0.786, 201525: 0.82, 201526: 1.073, 201527: 1.024, 201528: 0.77, 201529: 0.767, 201530: 0.666, 201531: 0.735, 201532: 0.671}, {200340: 0.884, 200341: 1.0277, 200342: 1.283, 200343: 1.3266, 200344: 1.773, 200345: 2.4381, 200346: 2.726, 200347: 3.7319, 200348: 4.5489, 200349: 5.3991, 200350: 7.3874, 200351: 7.481, 200352: 7.628, 200353: 5.1405, 200401: 2.8913, 200402: 2.1043, 200403: 1.887, 200404: 1.7766, 200405: 1.46, 200406: 1.3779, 200407: 1.3053, 200408: 1.0869, 200409: 0.9919, 200410: 1.0277, 200411: 1.0579, 200412: 0.8494, 200413: 0.7471, 200414: 0.6617, 200415: 0.6795, 200416: 0.6331, 200417: 0.5699, 200418: 0.5914, 200419: 0.589, 200420: 0.6786, 200421: 0.6376, 200422: 0.8133, 200423: 0.8346, 200424: 0.6162, 200425: 0.5133, 200426: 0.3519, 200427: 0.5426, 200428: 0.581, 200429: 0.4955, 200430: 0.6121, 200431: 0.5315, 200432: 0.52, 200433: 0.5745, 200434: 0.6632, 200435: 0.6147, 200436: 0.6568, 200437: 0.723, 200438: 0.6919, 200439: 0.7828, 200440: 0.9823, 200441: 1.0182, 200442: 1.1768, 200443: 1.3012, 200444: 1.2775, 200445: 1.4039, 200446: 1.403, 200447: 1.7397, 200448: 1.5689, 200449: 1.6593, 200450: 1.8134, 200451: 2.3763, 200452: 2.7705, 200501: 2.4006, 200502: 2.5415, 200503: 2.8587, 200504: 3.5501, 200505: 4.5859, 200506: 5.0581, 200507: 5.4415, 200508: 4.4729, 200509: 3.6752, 200510: 3.3353, 200511: 2.7622, 200512: 2.4684, 200513: 2.0008, 200514: 1.5432, 200515: 1.502, 200516: 1.2292, 200517: 1.2185, 200518: 1.1583, 200519: 1.249, 200520: 1.1689, 200521: 1.0789, 200522: 1.2306, 200523: 1.1475, 200524: 1.0025, 200525: 0.9576, 200526: 0.8948, 200527: 0.7982, 200528: 0.7578, 200529: 0.826, 200530: 0.7449, 200531: 0.787, 200532: 0.6847, 200533: 0.7474, 200534: 0.674, 200535: 0.8697, 200536: 0.8995, 200537: 0.8546, 200538: 1.0112, 200539: 0.9778, 200540: 1.1778, 200541: 1.2195, 200542: 1.3069, 200543: 1.3393, 200544: 1.5837, 200545: 1.4707, 200546: 1.6112, 200547: 1.827, 200548: 1.7748, 200549: 1.9445, 200550: 2.3551, 200551: 3.1088, 200552: 3.2824, 200601: 2.6291, 200602: 2.2515, 200603: 2.3586, 200604: 2.4188, 200605: 2.5275, 200606: 2.6495, 200607: 3.1294, 200608: 3.1244, 200609: 3.131, 200610: 3.0826, 200611: 2.6443, 200612: 2.4229, 200613: 2.3114, 200614: 1.8638, 200615: 1.4494, 200616: 1.3023, 200617: 1.1346, 200618: 1.0626, 200619: 1.0343, 200620: 1.0179, 200621: 0.9182, 200622: 0.931, 200623: 0.8573, 200624: 0.7863, 200625: 0.7194, 200626: 0.7189, 200627: 0.7748, 200628: 0.766, 200629: 0.6428, 200630: 0.622, 200631: 0.6163, 200632: 0.6857, 200633: 0.6227, 200634: 0.6118, 200635: 0.6388, 200636: 0.8066, 200637: 0.7624, 200638: 0.8134, 200639: 0.8079, 200640: 1.1538, 200641: 1.1482, 200642: 1.2274, 200643: 1.2174, 200644: 1.3281, 200645: 1.4775, 200646: 1.5339, 200647: 1.9127, 200648: 1.838, 200649: 1.9538, 200650: 2.4041, 200651: 2.8745, 200652: 3.026, 200701: 2.3782, 200702: 2.0923, 200703: 2.294, 200704: 2.745, 200705: 3.0443, 200706: 3.5, 200707: 3.5785, 200708: 3.2898, 200709: 2.8915, 200710: 2.6207, 200711: 2.5339, 200712: 2.0725, 200713: 1.8708, 200714: 1.4004, 200715: 1.4689, 200716: 1.1551, 200717: 1.079, 200718: 0.9849, 200719: 1.0332, 200720: 0.9266, 200721: 0.9201, 200722: 0.9541, 200723: 0.7633, 200724: 0.8172, 200725: 0.7646, 200726: 0.7931, 200727: 0.6691, 200728: 0.6212, 200729: 0.6037, 200730: 0.6491, 200731: 0.5683, 200732: 0.6515, 200733: 0.6522, 200734: 0.8038, 200735: 0.7875, 200736: 0.9433, 200737: 0.9469, 200738: 0.9871, 200739: 1.0521, 200740: 0.9952, 200741: 1.22, 200742: 1.2855, 200743: 1.3775, 200744: 1.4672, 200745: 1.5584, 200746: 1.6196, 200747: 1.8112, 200748: 1.6186, 200749: 1.6212, 200750: 1.6735, 200751: 1.951, 200752: 2.5002, 200801: 2.4394, 200802: 2.2932, 200803: 2.6291, 200804: 3.9297, 200805: 4.992, 200806: 5.7776, 200807: 5.9832, 200808: 5.6168, 200809: 4.5323, 200810: 3.8325, 200811: 3.2152, 200812: 2.6924, 200813: 2.0337, 200814: 1.6528, 200815: 1.3113, 200816: 1.1652, 200817: 1.0203, 200818: 0.8776, 200819: 0.8251, 200820: 0.7873, 200821: 0.8054, 200822: 0.8861, 200823: 0.7437, 200824: 0.7068, 200825: 0.6559, 200826: 0.6335, 200827: 0.5718, 200828: 0.5583, 200829: 0.5979, 200830: 0.5772, 200831: 0.597, 200832: 0.5647, 200833: 0.5857, 200834: 0.504, 200835: 0.4681, 200836: 0.5765, 200837: 0.6814, 200838: 0.7948, 200839: 0.8681, 200840: 1.0683, 200841: 1.0076, 200842: 0.9794, 200843: 1.0494, 200844: 1.1322, 200845: 1.1316, 200846: 1.1547, 200847: 1.2045, 200848: 1.3608, 200849: 1.3032, 200850: 1.302, 200851: 1.4511, 200852: 1.7993, 200853: 2.1213, 200901: 1.4714, 200902: 1.6155, 200903: 1.9533, 200904: 2.4291, 200905: 2.9669, 200906: 3.5735, 200907: 3.3665, 200908: 3.2962, 200909: 3.1723, 200910: 2.5804, 200911: 2.4292, 200912: 1.9903, 200913: 1.7242, 200914: 1.6035, 200915: 1.2842, 200916: 1.2932, 200917: 2.7401, 200918: 2.2239, 200919: 1.7666, 200920: 1.9217, 200921: 1.8611, 200922: 1.6842, 200923: 1.7393, 200924: 1.83, 200925: 1.5813, 200926: 1.4917, 200927: 1.389, 200928: 1.347, 200929: 1.1606, 200930: 1.2085, 200931: 1.1554, 200932: 1.1452, 200933: 1.4664, 200934: 2.3802, 200935: 3.4468, 200936: 4.0531, 200937: 4.2662, 200938: 4.2158, 200939: 4.8861, 200940: 6.0692, 200941: 7.0947, 200942: 7.7151, 200943: 7.5513, 200944: 6.6712, 200945: 5.2063, 200946: 4.0448, 200947: 3.4972, 200948: 2.6902, 200949: 2.4712, 200950: 2.4477, 200951: 2.6459, 200952: 2.6157, 201001: 1.9071, 201002: 1.8674, 201003: 1.8807, 201004: 1.9691, 201005: 2.1139, 201006: 2.0969, 201007: 1.9029, 201008: 1.9695, 201009: 1.9233, 201010: 1.8966, 201011: 1.7518, 201012: 1.5263, 201013: 1.4957, 201014: 1.2231, 201015: 1.1775, 201016: 1.0911, 201017: 1.1196, 201018: 1.0817, 201019: 1.125, 201020: 1.062, 201021: 1.0891, 201022: 1.1776, 201023: 0.9749, 201024: 0.9871, 201025: 0.996, 201026: 0.9047, 201027: 0.9419, 201028: 0.8804, 201029: 0.8426, 201030: 0.8584, 201031: 0.8338, 201032: 0.8474, 201033: 0.8883, 201034: 0.7654, 201035: 0.9186, 201036: 1.0143, 201037: 1.0123, 201038: 1.118, 201039: 1.1181, 201040: 1.1094, 201041: 1.2434, 201042: 1.2573, 201043: 1.2573, 201044: 1.4341, 201045: 1.5184, 201046: 1.6162, 201047: 1.8438, 201048: 1.7253, 201049: 1.8792, 201050: 2.3371, 201051: 3.0353, 201052: 3.136, 201101: 2.5334, 201102: 2.9129, 201103: 3.445, 201104: 4.1619, 201105: 4.5516, 201106: 4.465, 201107: 4.5512, 201108: 4.0822, 201109: 3.3881, 201110: 2.9914, 201111: 2.6154, 201112: 2.0907, 201113: 1.8615, 201114: 1.6138, 201115: 1.422, 201116: 1.3382, 201117: 1.2, 201118: 1.1489, 201119: 1.1219, 201120: 1.0671, 201121: 1.0506, 201122: 1.152, 201123: 0.971, 201124: 0.8947, 201125: 0.8436, 201126: 0.8154, 201127: 0.8231, 201128: 0.6746, 201129: 0.6319, 201130: 0.6415, 201131: 0.5298, 201132: 0.6441, 201133: 0.6677, 201134: 0.7057, 201135: 0.8042, 201136: 0.9078, 201137: 1.0055, 201138: 1.036, 201139: 1.0758, 201140: 1.1609, 201141: 1.1979, 201142: 1.2793, 201143: 1.2919, 201144: 1.4282, 201145: 1.398, 201146: 1.4583, 201147: 1.5655, 201148: 1.3431, 201149: 1.4814, 201150: 1.634, 201151: 1.8313, 201152: 2.1045, 201201: 1.7363, 201202: 1.5507, 201203: 1.6289, 201204: 1.7604, 201205: 1.9273, 201206: 1.9199, 201207: 2.0995, 201208: 2.2397, 201209: 2.2192, 201210: 2.1669, 201211: 2.3891, 201212: 1.9647, 201213: 1.8423, 201214: 1.6867, 201215: 1.5314, 201216: 1.3791, 201217: 1.2952, 201218: 1.3952, 201219: 1.3252, 201220: 1.2306, 201221: 1.2303, 201222: 1.2688, 201223: 1.1078, 201224: 1.0956, 201225: 1.1103, 201226: 1.1062, 201227: 1.1082, 201228: 0.9994, 201229: 0.9242, 201230: 0.9493, 201231: 0.9789, 201232: 0.925, 201233: 0.8485, 201234: 1.0445, 201235: 1.0986, 201236: 1.2207, 201237: 1.2498, 201238: 1.2289, 201239: 1.2939, 201240: 1.2303, 201241: 1.3278, 201242: 1.3399, 201243: 1.3784, 201244: 1.5009, 201245: 1.6516, 201246: 1.7675, 201247: 2.3061, 201248: 2.1632, 201249: 2.8304, 201250: 3.4376, 201251: 4.3055, 201252: 6.0608, 201301: 4.6493, 201302: 4.3258, 201303: 4.5111, 201304: 4.2171, 201305: 3.7896, 201306: 3.2921, 201307: 3.0197, 201308: 2.715, 201309: 2.4845, 201310: 2.5781, 201311: 2.3924, 201312: 2.0678, 201313: 1.9028, 201314: 1.659, 201315: 1.4301, 201316: 1.2776, 201317: 1.1901, 201318: 1.0956, 201319: 1.0859, 201320: 1.0734, 201321: 1.0168, 201322: 1.0485, 201323: 0.9203, 201324: 0.9208, 201325: 0.9043, 201326: 0.8306, 201327: 0.8628, 201328: 0.7322, 201329: 0.7246, 201330: 0.7408, 201331: 0.726, 201332: 0.7556, 201333: 0.7618, 201334: 0.7799, 201335: 0.8991, 201336: 0.9995, 201337: 1.0558, 201338: 1.0967, 201339: 1.2471, 201340: 1.1575, 201341: 1.2718, 201342: 1.3162, 201343: 1.3744, 201344: 1.4678, 201345: 1.5834, 201346: 1.6025, 201347: 1.7234, 201348: 2.1108, 201349: 2.1708, 201350: 2.3697, 201351: 3.16, 201352: 4.5905, 201401: 4.282, 201402: 3.5629, 201403: 3.3638, 201404: 3.3966, 201405: 3.1192, 201406: 2.8357, 201407: 2.4848, 201408: 2.2741, 201409: 2.0724, 201410: 1.9821, 201411: 1.8244, 201412: 1.8204, 201413: 1.7539, 201414: 1.7454, 201415: 1.6724, 201416: 1.5166, 201417: 1.4478, 201418: 1.2903, 201419: 1.3739, 201420: 1.3421, 201421: 1.3456, 201422: 1.2129, 201423: 1.1929, 201424: 1.1415, 201425: 1.065, 201426: 1.1267, 201427: 0.9474, 201428: 0.9442, 201429: 0.9348, 201430: 0.8174, 201431: 0.8528, 201432: 0.747, 201433: 0.7611, 201434: 0.7676, 201435: 0.8622, 201436: 0.9849, 201437: 1.1629, 201438: 1.2273, 201439: 1.2071, 201440: 1.1619, 201441: 1.3091, 201442: 1.3707, 201443: 1.4337, 201444: 1.4279, 201445: 1.616, 201446: 1.6598, 201447: 2.0546, 201448: 2.5354, 201449: 2.5425, 201450: 3.6056, 201451: 4.9098, 201452: 5.9822, 201453: 5.4742, 201501: 4.2137, 201502: 4.2102, 201503: 4.2597, 201504: 3.9776, 201505: 3.6344, 201506: 3.2653, 201507: 2.9308, 201508: 2.5447, 201509: 2.4723, 201510: 2.3567, 201511: 2.1701, 201512: 2.043, 201513: 1.954, 201514: 1.7229, 201515: 1.5369, 201516: 1.3828, 201517: 1.4627, 201518: 1.3893, 201519: 1.3468, 201520: 1.2545, 201521: 1.2164, 201522: 1.1673, 201523: 1.0465, 201524: 0.9437, 201525: 0.935, 201526: 0.9347, 201527: 0.8641, 201528: 0.8065, 201529: 0.7427, 201530: 0.7379, 201531: 0.7432, 201532: 0.8053})
weeks = sorted((w for w in gft if w in cdc and 201027 <= w <= 201426))
peak_lo = 2012 * 100 + 40
peak_hi = (2012 + 1) * 100 + 20
xs = [epiweek_to_date(w) for w in weeks]

def gft_plot(path, cut=None, annotate=False, size=(5.8, 2.6), gft_label=(201027, 9.6), font=11):
    """gft_label: (epiweek, y) of the top-left corner of the red series label."""
    fig, ax = sp.plt.subplots(figsize=size)
    ax.plot(xs, [cdc[w] for w in weeks], color=C['neutral'], lw=3)
    ax.text(epiweek_to_date(201432), cdc[weeks[-1]] + 0.2, 'CDC\n(doctor visits)', color=C['neutral'], fontsize=font, fontweight='bold', va='bottom')
    shown = [w for w in weeks if cut is None or w <= cut]
    ax.plot([epiweek_to_date(w) for w in shown], [gft[w] for w in shown], color=C['danger'], lw=3)
    ax.text(epiweek_to_date(gft_label[0]), gft_label[1], 'Google Flu Trends\n(from searches)', color=C['danger'], fontsize=font, fontweight='bold', va='top')
    if annotate:
        pk = max((w for w in weeks if peak_lo <= w <= peak_hi), key=lambda w: gft[w])
        ck = max((w for w in weeks if peak_lo <= w <= peak_hi), key=lambda w: cdc[w])
        ax.annotate('10.6%', (epiweek_to_date(pk), gft[pk]), xytext=(8, -2), textcoords='offset points', color=C['danger'], fontsize=12, fontweight='bold', va='center')
        ax.annotate('6.1%', (epiweek_to_date(ck), cdc[ck]), xytext=(-6, 4), textcoords='offset points', ha='right', color=C['neutral'], fontsize=12, fontweight='bold')
    ax.set_xlim(xs[0], epiweek_to_date(201452))
    ax.set_xticks([date(y, 1, 1) for y in range(2011, 2015)])
    ax.set_xticklabels([str(y) for y in range(2011, 2015)])
    ax.set_ylim(0, 11.5)
    ax.set_ylabel('Flu-like illness\n(% of doctor visits)')
    ax.grid(axis='y', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)
gft_plot(out / 'gft_step1.svg', cut=201131)
gft_plot(out / 'gft_step2.svg', annotate=True)
gft_plot(out / 'gft_column.svg', annotate=True, size=(3.3, 2.05), gft_label=(201036, 10.9), font=10)
seasons = list({'03-04': 0.867527399, '04-05': 0.9071853923, '05-06': 0.9126897181, '06-07': 0.9190220271, '07-08': 0.9619466246, '08-09': 1.123616219, '09-10': 0.9062482172, '10-11': 0.927285893, '11-12': 1.267420211, '12-13': 1.657189515}.items())
drift_labels = {'11-12': '1.27', '12-13': '1.66'}
fig, ax = sp.plt.subplots(figsize=(5.8, 2.4))
for i, (lab, r) in enumerate(seasons):
    ck = 'danger' if lab in drift_labels else 'neutral'
    ax.bar(i, r, color=C[ck], width=0.66)
    if lab in drift_labels:
        ax.text(i, r + 0.04, drift_labels[lab], ha='center', va='bottom', color=C[ck], fontsize=12, fontweight='bold')
ax.axhline(1.0, color=C['text'], lw=1.2, ls='--')
ax.text(-0.45, 1.04, 'estimate = CDC count', fontsize=10, color=C['text'], va='bottom')
ax.set_xticks(range(len(seasons)))
ax.set_xticklabels([s[0] for s in seasons], fontsize=10)
ax.set_ylim(0, 1.95)
ax.set_xlabel('Flu season')
ax.set_ylabel('Estimate / CDC')
sp.save(fig, out / 'gft_ratio.svg')
sp.hbars_steps(out / 'gpu_days', [('AlexNet (2012)', 12.0, 'neutral', '12 GPU-days'), ('GPT-3 (2020)', 148000.0, 'compute', 'about 148,000'), ('GPT-4-class (2023)', 2.5 * 10 ** 6, 'danger', 'about 2.5 million (est.)')], 'GPU-days', xlabel='Accelerator-days to train (log scale)', value_fmt='{:,.0f}')
TERMS = {'rn': (0.1129086787, 1.061274014, 0.51), 'g2': (0.9850746269, 0.006066734075, 0.49)}
print('iron-law terms (ms):', TERMS)

def iron_bars(path, show_double):
    """Stacked memory, compute, and overhead bars with each segment's value
    in ms, so students can halve the memory term and add. Values sit inside
    the segments; the term names sit above the first bar."""
    fig, ax = sp.plt.subplots(figsize=(4.4, 1.75))
    ypos = [3.6, 2.75, 1.55, 0.7]
    names = ['ResNet-50\nbatch 64', 'GPT-2 XL decode\nbatch 1']
    rows = []
    for name, (key, sc) in zip(names, (('rn', SimpleNamespace(batch_size=64, efficiency=0.5, hardware=SimpleNamespace(accelerator_count=None, battery_capacity=None, compute=SimpleNamespace(clock_rate=None, core_count=None, fp32_core_count=16896, fp32_cores_per_sm=128, max_registers_per_thread=255, max_threads_per_sm=2048, peak_flops=Q_(989.0, 'TFLOPs / s'), precision_flops={'tf32': Q_(494.5, 'TFLOPs / s'), 'fp16_dense': Q_(989.0, 'TFLOPs / s'), 'fp16_sparse': Q_(1979.0, 'TFLOPs / s'), 'fp8': Q_(1979.0, 'TFLOPs / s'), 'int8': Q_(1979, 'TOPS'), 'tf32_sparse': Q_(989.0, 'TFLOPs / s'), 'int8_sparse': Q_(3958, 'TOPS'), 'fp32': Q_(67.0, 'TFLOPs / s')}, sm_count=132, systolic_array_dim=None, systolic_macs_per_cycle=None, tensor_core_count=528, tensor_cores_per_sm=4, warp_size=32), die_area=Q_(814, 'mm ** 2'), die_to_die_bandwidth=None, dispatch_tax=Q_(0.01, 'ms'), embodied_carbon=Q_(164.0, 'kg'), embodied_carbon_kg=164.0, interconnect=SimpleNamespace(bandwidth=Q_(64.0, 'GB / s'), bandwidth_per_direction=Q_(64.0, 'GB / s'), direction='per_direction', latency=None, name='PCIe Gen5 x16'), memory=SimpleNamespace(bandwidth=Q_(3.35, 'TB / s'), bus_width=None, capacity=Q_(80, 'GiB'), flash_bandwidth=None, flash_capacity=None, hbm_stacks=5, l2_cache=Q_(50, 'MiB'), pin_data_rate=None, register_file_per_sm=Q_(256, 'KiB'), shared_memory_per_sm=Q_(228, 'KiB'), sram_bandwidth=None, sram_capacity=None), metadata=SimpleNamespace(last_verified=None, source_label='NVIDIA H100 datasheet; NVIDIA Hopper Architecture In-Depth; Choquette (2023), IEEE Micro; NVIDIA HGX H100 product carbon footprint', version=None), mig_profiles=None, name='NVIDIA H100', nvlink=SimpleNamespace(bandwidth=Q_(900.0, 'GB / s'), bandwidth_per_direction=Q_(450.0, 'GB / s'), direction='bidirectional_total', latency=Q_(500, 'ns'), name='NVLink 4.0'), release_year=2022, storage=SimpleNamespace(bandwidth=Q_(7.0, 'GB / s'), capacity=Q_(2, 'TiB'), latency=None), tdp=Q_(700, 'W'), tdp_max=None, tdp_min=None, transistor_count=80000000000.0, unit_cost=Q_(25000, 'dollar'), unit_cost_max=Q_(30000, 'dollar')), iron_law=SimpleNamespace(compute_time=Q_(1.061274014, 'ms'), memory_time=Q_(0.1129086787, 'ms'), naive_upgraded_time=Q_(0.8420913464, 'ms'), overhead=Q_(0.51, 'ms'), overlapped_time=Q_(1.571274014, 'ms'), sequential_reduction=0.03352031795, sequential_time=Q_(1.684182693, 'ms'), upgraded_compute_time=Q_(1.061274014, 'ms'), upgraded_memory_time=Q_(0.05645433934, 'ms'), upgraded_overlapped_time=Q_(1.571274014, 'ms'), upgraded_sequential_time=Q_(1.627728353, 'ms')), memory_speedup=2.0, memory_traffic=Q_(0.3782440736, 'GB'), name='ResNet-50 inference at batch 64 on one H100', precision='fp16', result=SimpleNamespace(arithmetic_intensity=Q_(1387.463907, 'flop / B'), bottleneck='Compute', energy=Q_(0.5899796764, 'J'), energy_per_inference=Q_(0.5899796764, 'J'), feasible=True, hfu=0.3377113109, latency=Q_(1.571274014, 'ms'), latency_compute=Q_(1.061274014, 'ms'), latency_memory=Q_(0.1129086787, 'ms'), latency_overhead=Q_(0.51, 'ms'), memory_footprint=Q_(51114064.0, 'B'), mfu=0.3377113109, offload_effective_bw=None, offload_spill_bytes=None, overhead_dominated=False, peak_bw_actual=Q_(3.35, 'TB / s'), peak_flops_actual=Q_(494.5, 'TFLOPs / s'), throughput=Q_(40731.27884, '1 / s')), workload=SimpleNamespace(architecture='CNN', data_rate=None, embedding_entries=None, inference_energy=None, inference_flops=Q_(8200000000.0, 'flop'), layers=50, metadata=SimpleNamespace(last_verified=None, source_label='He et al. (2016), Deep Residual Learning for Image Recognition', version=None), model_size=None, name='ResNet-50', parameter_range_max=None, parameter_range_min=None, parameters=Q_(25557032.0, 'param'), release_year=2015, training_dataset_size=None, width_multiplier=None))), ('g2', SimpleNamespace(batch_size=1, efficiency=0.5, hardware=SimpleNamespace(accelerator_count=None, battery_capacity=None, compute=SimpleNamespace(clock_rate=None, core_count=None, fp32_core_count=16896, fp32_cores_per_sm=128, max_registers_per_thread=255, max_threads_per_sm=2048, peak_flops=Q_(989.0, 'TFLOPs / s'), precision_flops={'tf32': Q_(494.5, 'TFLOPs / s'), 'fp16_dense': Q_(989.0, 'TFLOPs / s'), 'fp16_sparse': Q_(1979.0, 'TFLOPs / s'), 'fp8': Q_(1979.0, 'TFLOPs / s'), 'int8': Q_(1979, 'TOPS'), 'tf32_sparse': Q_(989.0, 'TFLOPs / s'), 'int8_sparse': Q_(3958, 'TOPS'), 'fp32': Q_(67.0, 'TFLOPs / s')}, sm_count=132, systolic_array_dim=None, systolic_macs_per_cycle=None, tensor_core_count=528, tensor_cores_per_sm=4, warp_size=32), die_area=Q_(814, 'mm ** 2'), die_to_die_bandwidth=None, dispatch_tax=Q_(0.01, 'ms'), embodied_carbon=Q_(164.0, 'kg'), embodied_carbon_kg=164.0, interconnect=SimpleNamespace(bandwidth=Q_(64.0, 'GB / s'), bandwidth_per_direction=Q_(64.0, 'GB / s'), direction='per_direction', latency=None, name='PCIe Gen5 x16'), memory=SimpleNamespace(bandwidth=Q_(3.35, 'TB / s'), bus_width=None, capacity=Q_(80, 'GiB'), flash_bandwidth=None, flash_capacity=None, hbm_stacks=5, l2_cache=Q_(50, 'MiB'), pin_data_rate=None, register_file_per_sm=Q_(256, 'KiB'), shared_memory_per_sm=Q_(228, 'KiB'), sram_bandwidth=None, sram_capacity=None), metadata=SimpleNamespace(last_verified=None, source_label='NVIDIA H100 datasheet; NVIDIA Hopper Architecture In-Depth; Choquette (2023), IEEE Micro; NVIDIA HGX H100 product carbon footprint', version=None), mig_profiles=None, name='NVIDIA H100', nvlink=SimpleNamespace(bandwidth=Q_(900.0, 'GB / s'), bandwidth_per_direction=Q_(450.0, 'GB / s'), direction='bidirectional_total', latency=Q_(500, 'ns'), name='NVLink 4.0'), release_year=2022, storage=SimpleNamespace(bandwidth=Q_(7.0, 'GB / s'), capacity=Q_(2, 'TiB'), latency=None), tdp=Q_(700, 'W'), tdp_max=None, tdp_min=None, transistor_count=80000000000.0, unit_cost=Q_(25000, 'dollar'), unit_cost_max=Q_(30000, 'dollar')), iron_law=SimpleNamespace(compute_time=Q_(0.006066734075, 'ms'), memory_time=Q_(0.9850746269, 'ms'), naive_upgraded_time=Q_(0.7405706805, 'ms'), overhead=Q_(0.49, 'ms'), overlapped_time=Q_(1.475074627, 'ms'), sequential_reduction=0.3325390313, sequential_time=Q_(1.481141361, 'ms'), upgraded_compute_time=Q_(0.006066734075, 'ms'), upgraded_memory_time=Q_(0.4925373134, 'ms'), upgraded_overlapped_time=Q_(0.9825373134, 'ms'), upgraded_sequential_time=Q_(0.9886040475, 'ms')), memory_speedup=2.0, memory_traffic=Q_(3.3, 'GB'), name='GPT-2 XL decoding one token for one user on one H100', precision='fp16', result=SimpleNamespace(arithmetic_intensity=Q_(0.9090909091, 'flop / B'), bottleneck='Memory', energy=Q_(0.3112520215, 'J'), energy_per_inference=Q_(0.3112520215, 'J'), feasible=True, hfu=0.002056415982, latency=Q_(1.475074627, 'ms'), latency_compute=Q_(0.006066734075, 'ms'), latency_memory=Q_(0.9850746269, 'ms'), latency_overhead=Q_(0.49, 'ms'), memory_footprint=Q_(3000000000.0, 'B'), mfu=0.002056415982, offload_effective_bw=None, offload_spill_bytes=None, overhead_dominated=False, peak_bw_actual=Q_(3.35, 'TB / s'), peak_flops_actual=Q_(494.5, 'TFLOPs / s'), throughput=Q_(677.9318021, '1 / s')), workload=SimpleNamespace(architecture='Transformer', context_length=1024, data_rate=None, embedding_entries=None, head_dim=None, head_width=64, heads=25, hidden_dim=1600, inference_energy=None, inference_flops=Q_(3000000000.0, 'flop'), kv_heads=None, kv_lora_rank=None, kv_width=1600, layers=48, metadata=SimpleNamespace(last_verified=None, source_label='Radford et al. (2019), Language Models are Unsupervised Multitask Learners', version=None), model_size=None, name='GPT-2 (1.5B)', parameter_range_max=None, parameter_range_min=None, parameters=Q_(1500000000.0, 'param'), qk_rope_head_dim=None, queries_per_kv_head=1, release_year=None, training_accelerators_ref=None, training_co2e=None, training_dataset_size=Q_(40, 'GB'), training_days_ref=None, training_energy_mwh=None, training_gpu_days=None, training_hardware_label=None, training_ops=Q_(1.5e+21, 'flop'), training_tokens=None, vocab_size=50257))))):
        m, c, o = TERMS[key]
        rows.append((name, key, m, c, o, False))
        rows.append(('2× bandwidth', key, m / {'ResNet-50\nbatch 64': 2.0, 'GPT-2 XL decode\nbatch 1': 2.0}[name], c, o, True))
    wide = 0.2
    for i, (name, key, m, c, o, dbl) in enumerate(rows):
        y = ypos[i]
        vis = not dbl or show_double
        a = 1.0 if vis else 0.0
        left = 0
        terms = ((m, 'memory', f'{key}_mem2' if dbl else f'{key}_mem'), (c, 'compute', f'{key}_comp'), (o, 'neutral', f'{key}_over'))
        for v, ck, vkey in terms:
            col = C[ck + '_l'] if dbl else C[ck]
            ax.barh(y, v, left=left, color=col, edgecolor='white', height=0.62, alpha=a)
            if vis and v >= wide:
                ax.text(left + v / 2, y, _VALS[vkey], ha='center', va='center', color=C['text'] if dbl else 'white', fontsize=11, fontweight='bold')
            left += v
        ax.text(-0.03, y, name, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=11, color=C['text'] if vis else 'white', fontweight='bold' if not dbl else 'normal', linespacing=1.0)
        lab = f"{_VALS[key + '_total2']} ms  (-{_VALS[key + '_cut_pct']}%)" if dbl else f"{_VALS[key + '_total']} ms"
        ax.text(left + 0.03, y, lab, va='center', fontsize=12, color=C['text'], fontweight='bold', alpha=1 if vis else 0)
    rm, rc, ro = TERMS['rn']
    top = ypos[0] + 0.36
    ax.text(0, top, 'memory 0.11', ha='left', va='bottom', color=C['memory'], fontsize=10, fontweight='bold')
    ax.text(rm + rc * 0.7, top, 'compute', ha='center', va='bottom', color=C['compute'], fontsize=10, fontweight='bold')
    ax.text(rm + rc + ro / 2, top, 'overhead', ha='center', va='bottom', color=C['neutral'], fontsize=10, fontweight='bold')
    gm, gc, go = TERMS['g2']
    top = ypos[2] + 0.36
    ax.text(gm / 2, top, 'memory', ha='center', va='bottom', color=C['memory'], fontsize=10, fontweight='bold')
    ax.text(gm + gc / 2, top, 'compute 0.006', ha='left', va='bottom', color=C['compute'], fontsize=10, fontweight='bold')
    ax.plot([gm + gc / 2, gm + gc / 2], [ypos[2] + 0.31, top + 0.05], color=C['compute'], lw=1.2)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 2.25)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_ylim(0.3, 4.45)
    ax.set_xlabel('Time per step on one H100 (ms)')
    ax.grid(axis='x', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)
iron_bars(out / 'ironlaw_bars_step1.svg', False)
iron_bars(out / 'ironlaw_bars_step2.svg', True)

def degradation(path, step):
    fig, ax = sp.plt.subplots(figsize=(3.9, 1.2))
    winter = 0.6
    xs_line = [0, 1]
    r0, r1 = (95.0, 85.0)
    base = (r0, (r0 - r1) / winter)
    lines = [(base, C['danger'], "today's model", 1), ((99, base[1]), C['compute'], 'better model, same data: starts higher, falls as fast', 2), ((r0, 5.0), C['data'], 'winter images in training: falls more slowly', 3)]
    for (a0, lam), col, lab, k in lines:
        ys = [a0 - lam * x for x in xs_line]
        ax.plot(xs_line, ys, color=col, lw=3, solid_capstyle='round', alpha=1 if k <= step else 0)
    if step >= 1:
        ax.plot([0, winter], [r0, r1], 'o', color=C['danger'], ms=7, zorder=4)
        ax.text(0.025, 91.8, '95%', color=C['danger'], fontsize=11, fontweight='bold', va='top')
        ax.text(winter - 0.015, 83.5, '85%', color=C['danger'], fontsize=11, fontweight='bold', va='top', ha='right')
        ax.text(1.01, r0 - base[1], "today's model", color=C['danger'], fontsize=11, fontweight='bold', va='center')
    ax.text(1.01, 99 - base[1], 'better model, same data', color=C['compute'], fontsize=11, fontweight='bold', va='center', alpha=1 if step >= 2 else 0)
    ax.text(1.01, 90, 'winter images in training', color=C['data'], fontsize=11, fontweight='bold', va='center', alpha=1 if step >= 3 else 0)
    ax.axvline(winter, color=C['neutral'], lw=1, ls='--', zorder=0)
    ax.text(winter + 0.01, 100.5, 'winter', color=C['neutral'], fontsize=10, va='top')
    ax.text(0.99, 100.5, 'Illustrative', color=C['neutral'], fontsize=10, va='top', ha='right', style='italic')
    ax.set_xlim(0, 1)
    ax.set_ylim(75, 101)
    ax.set_xticks([0])
    ax.set_xticklabels(['training conditions'])
    ax.set_yticks([80, 90, 100])
    ax.set_xlabel('Distance from the training data (drift)', labelpad=2)
    ax.set_ylabel('Recall (%)')
    ax.grid(axis='y', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)
for k in (1, 2, 3):
    degradation(out / f'degradation_step{k}.svg', k)

def amdahl(path, show_after):
    fig, ax = sp.plt.subplots(figsize=(5.8, 1.9))
    rows = [('Before', ('am_pre', 'am_model', 'am_post'), 'am_total'), ('After', ('am_pre', 'am_model2', 'am_post'), 'am_total2')]
    for i, (name, keys, total_key) in enumerate(rows):
        y = 1 - i
        vis = i == 0 or show_after
        left = 0
        for k, ck, lab in zip(keys, ('neutral', 'compute', 'neutral'), ('preprocess 60', 'model', 'post 25')):
            v = _NUMS[k]
            ax.barh(y, v, left=left, color=C[ck] if ck == 'compute' else C['neutral_l'], edgecolor='white', linewidth=2, height=0.6, alpha=1 if vis else 0)
            if vis:
                txt = (f'model {_VALS[k]}' if v > 20 else _VALS[k]) if ck == 'compute' else lab
                ax.text(left + v / 2, y, txt, ha='center', va='center', fontsize=11, fontweight='bold', color='white' if ck == 'compute' else C['text'])
            left += v
        ax.text(-2, y, name, ha='right', va='center', fontsize=13, fontweight='bold', color=C['text'] if vis else 'white')
        if vis:
            ax.text(left + 2, y, f"{ {'Before': '130', 'After': '100'}[name]} ms" + ('   (-23%)' if i == 1 else ''), va='center', fontsize=13, fontweight='bold', color=C['danger'] if i == 1 else C['text'])
    ax.set_xlim(0, 165)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('End-to-end latency (ms)')
    sp.save(fig, path)
amdahl(out / 'amdahl_step1.svg', False)
amdahl(out / 'amdahl_step2.svg', True)
print('figures written')

def literature():
    algo = [('AlexNet', 2012.5, 1.0), ('VGG-11', 2014.7, 0.723687789), ('GoogLeNet', 2014.75, 4.333876221), ('ResNet-18', 2015.9, 2.718079673), ('DenseNet-121', 2016.7, 3.209891435), ('MobileNet-v1', 2017.3, 11.0875), ('ShuffleNet-v1', 2017.5, 20.62790698), ('MobileNet-v2', 2018.1, 13.17326733), ('ShuffleNet-v2', 2018.5, 24.63888889), ('EfficientNet-B0', 2019.4, 44.35)]
    comp = [('AlexNet', 2012, 4.7e+17), ('VGG-16', 2014, 1.23e+19), ('GoogLeNet', 2014, 1.51e+18), ('ResNet-152', 2015, 1.04e+19), ('Transformer', 2017, 7.42e+18), ('AlphaGoZero', 2017, 6.49e+20), ('BERT-Large', 2018, 2.85e+20), ('GPT-2-XL', 2019, 1.92e+21), ('T5-11 billion', 2019, 3.3e+22), ('GPT-3', 2020, 3.14e+23), ('Gopher', 2021, 6.31e+23), ('PaLM', 2022, 2.53e+24), ('GPT-4-class', 2023, 2.1e+25), ('Gemini-Ultra', 2023, 5e+25), ('Llama-3-70 billion', 2024, 7.86e+24), ('Llama 3.1 405B', 2024, 3.8e+25), ('Grok-3', 2025, 3.5e+26)]
    return (algo, comp)
ALGO, COMP = literature()
HALF = (3.5, 2.8)
fig, ax = sp.plt.subplots(figsize=HALF)
ax.scatter([a[1] for a in ALGO], [a[2] for a in ALGO], s=60, color=C['compute'], edgecolors='white', zorder=3)
for name, yr, f in ALGO:
    if name in ('AlexNet', 'MobileNet-v2', 'EfficientNet-B0'):
        lab = 'EfficientNet: 44×' if name == 'EfficientNet-B0' else name
        off = {'AlexNet': (0, 10), 'MobileNet-v2': (8, -6), 'EfficientNet-B0': (-6, 0)}[name]
        ax.annotate(lab, (yr, f), xytext=off, textcoords='offset points', ha={'EfficientNet-B0': 'right', 'MobileNet-v2': 'left'}.get(name, 'center'), va='bottom' if name == 'AlexNet' else 'center', fontsize=11, fontweight='bold', color=C['compute'] if name == 'EfficientNet-B0' else C['text'])
ax.set_xlim(2012, 2020)
ax.set_ylim(0, 50)
ax.set_xticks([2012, 2014, 2016, 2018, 2020])
ax.set_ylabel('Less compute than AlexNet (×)')
ax.set_xlabel('Year')
ax.grid(axis='y', color=C['grid'], linewidth=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / 'algo_efficiency.svg')
fig, ax = sp.plt.subplots(figsize=HALF)
ax.scatter([c[1] for c in COMP], [c[2] for c in COMP], s=60, color=C['danger'], edgecolors='white', zorder=3)
for name, yr, fl in COMP:
    if name in ('AlexNet', 'GPT-3', 'GPT-4-class'):
        off = {'AlexNet': (4, 8), 'GPT-3': (-6, 6), 'GPT-4-class': (-6, 6)}[name]
        ax.annotate(name, (yr, fl), xytext=off, textcoords='offset points', ha='left' if name == 'AlexNet' else 'right', va='bottom', fontsize=11, fontweight='bold', color=C['text'])
ax.set_yscale('log')
ax.set_xlim(2011, 2026)
ax.set_xticks([2012, 2016, 2020, 2024])
ax.set_ylabel('Training compute (FLOPs)')
ax.set_xlabel('Year')
ax.minorticks_off()
ax.grid(axis='y', color=C['grid'], linewidth=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / 'compute_growth.svg')
print('literature figures written')
REC = [('pnasnet_large_tf', 82.9, 72.2), ('pnasnet5large', 82.7, 72.1), ('nasnet_large_tf', 82.7, 72.2), ('nasnetalarge', 82.5, 72.2), ('senet154', 81.3, 70.5), ('polynet', 80.9, 70.3), ('inception_resnet_v2_tf', 80.4, 69.7), ('inceptionresnetv2', 80.3, 69.6), ('se_resnext101_32x4d', 80.2, 69.3), ('inception_v4_tf', 80.2, 68.8), ('inceptionv4', 80.1, 69.1), ('dpn107', 79.7, 68.1), ('dpn131', 79.4, 67.9), ('dpn92', 79.4, 67.3), ('dpn98', 79.2, 67.8), ('se_resnext50_32x4d', 79.1, 67.9), ('resnext101_64x4d', 79.0, 67.1), ('xception', 78.8, 67.2), ('se_resnet152', 78.7, 67.5), ('se_resnet101', 78.4, 67.2), ('resnet152', 78.3, 67.0), ('resnext101_32x4d', 78.2, 66.2), ('inception_v3_tf', 78.0, 66.1), ('resnet_v2_152_tf', 77.8, 66.1), ('se_resnet50', 77.6, 66.2), ('fbresnet152', 77.4, 65.8), ('resnet101', 77.4, 65.7), ('inceptionv3', 77.3, 65.7), ('inception_v3', 77.2, 65.4), ('densenet161', 77.1, 65.3), ('dpn68b', 77.0, 64.7), ('resnet_v2_101_tf', 77.0, 64.6), ('densenet201', 76.9, 64.7), ('resnet_v1_152_tf', 76.8, 64.6), ('resnet_v1_101_tf', 76.4, 64.5), ('cafferesnet101', 76.2, 64.3), ('resnet50', 76.1, 63.3), ('dpn68', 75.9, 63.4), ('densenet169', 75.6, 63.9), ('resnet_v2_50_tf', 75.6, 62.7), ('resnet_v1_50_tf', 75.2, 62.6), ('densenet121', 74.4, 62.2), ('vgg19_bn', 74.2, 61.9), ('pnasnet_mobile_tf', 74.1, 60.9), ('nasnetamobile', 74.1, 61.6), ('inception_v2_tf', 74.0, 61.2), ('nasnet_mobile_tf', 74.0, 60.8), ('bninception', 73.5, 62.1), ('vgg16_bn', 73.4, 60.8), ('resnet34', 73.3, 61.2), ('vgg19', 72.4, 59.7), ('vgg16', 71.6, 58.8), ('vgg13_bn', 71.6, 59.0), ('mobilenet_v1_tf', 71.0, 57.4), ('vgg_19_tf', 71.0, 58.6), ('vgg_16_tf', 70.9, 58.4), ('vgg11_bn', 70.4, 57.4), ('vgg13', 69.9, 57.1), ('inception_v1_tf', 69.8, 56.6), ('resnet18', 69.8, 57.2), ('vgg11', 69.0, 55.8), ('squeezenet1_1', 58.2, 45.3), ('squeezenet1_0', 58.1, 45.0), ('alexnet', 56.5, 44.0), ('fv_64k', 35.1, 24.1), ('fv_16k', 28.3, 19.2), ('fv_4k', 21.2, 15.0)]
fig, ax = sp.plt.subplots(figsize=(5.4, 1.6))
ox = [r[1] for r in REC]
ny = [r[2] for r in REC]
slope, icpt = (0.9676271793, -9.452284493)
ax.plot([15, 90], [15, 90], color=C['neutral'], lw=1.5, ls='--', zorder=1)
ax.text(54, 57, 'no loss (new = original)', color=C['neutral'], fontsize=11, rotation=0, ha='right', va='bottom')
ax.plot([18, 86], [icpt + slope * 18, icpt + slope * 86], color=C['danger'], lw=2.2, zorder=2)
ax.scatter(ox, ny, s=34, color=C['compute'], edgecolors='white', linewidths=0.6, zorder=3)
ax.text(55, 30, 'fit: new = 0.97 × original − 9.5', color=C['danger'], fontsize=12, fontweight='bold', ha='left')
ax.text(19, 84, '67 models', color=C['compute'], fontsize=12, fontweight='bold', ha='left', va='top')
ax.set_xlim(15, 90)
ax.set_ylim(10, 90)
ax.set_xticks([20, 40, 60, 80])
ax.set_yticks([20, 40, 60, 80])
ax.set_xlabel('Top-1 on original ImageNet test set (%)')
ax.set_ylabel('ImageNetV2\ntop-1 (%)')
ax.grid(color=C['grid'], linewidth=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / 'recht_scatter.svg')
H100_RUN, H200_RUN, CTS_RUN = ({'system': 'DGX-H100_H100-SXM-80GBx8_TRT', 'accelerator': 'H100', 'accelerators': 8.0, 'tokens_per_s': 21805.8, 'result': 'VALID'}, {'system': 'H200-SXM-141GBx8_TRT', 'accelerator': 'H200', 'accelerators': 8.0, 'tokens_per_s': 27738.3, 'result': 'VALID'}, {'system': 'H200-SXM-141GB-CTSx8_TRT', 'accelerator': 'H200', 'accelerators': 8.0, 'tokens_per_s': 31712.0, 'result': 'VALID'})
bw_ratio = 1.432835821
rows = [('H100, 700 W', H100_RUN, 'neutral', '1.00', 'mlperf_h100_tps_c'), ('H200, 700 W', H200_RUN, 'memory', '1.27', 'mlperf_h200_tps_c'), ('H200, 1000 W', CTS_RUN, 'memory', '1.45', 'mlperf_cts_tps_c')]
fig, ax = sp.plt.subplots(figsize=(5.0, 1.3))
for i, (lab, run, ck, mult, tkey) in enumerate(rows):
    y = len(rows) - 1 - i
    r = run['tokens_per_s'] / H100_RUN['tokens_per_s']
    ax.barh(y, r, color=C[ck], height=0.6)
    ax.text(-0.02, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
    ax.text(r - 0.03, y, f'{mult}×', ha='right', va='center', fontsize=12, fontweight='bold', color='white')
    ax.text(1.62, y, f"{ {'mlperf_h100_tps_c': '21,806', 'mlperf_h200_tps_c': '27,738', 'mlperf_cts_tps_c': '31,712'}[tkey]} tokens/s", va='center', fontsize=11, fontweight='bold', color=C[ck])
ax.axvline(1.0, color=C['compute'], lw=1.8, ls='--', zorder=0)
ax.text(0.98, 2.45, 'compute ratio 1.00', color=C['compute'], fontsize=10, ha='right', va='bottom', fontweight='bold')
ax.axvline(bw_ratio, color=C['memory'], lw=1.8, ls='--', zorder=0)
ax.text(bw_ratio + 0.02, 2.45, 'bandwidth ratio 1.43', color=C['memory'], fontsize=10, ha='left', va='bottom', fontweight='bold')
ax.set_xlim(0, 2.2)
ax.set_ylim(-0.5, 2.95)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
ax.set_xlabel('Throughput relative to H100 (8 GPUs each)')
sp.save(fig, out / 'mlperf_h200.svg')
print('measured-evidence figures written')
