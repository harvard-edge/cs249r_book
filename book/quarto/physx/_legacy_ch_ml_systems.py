"""
Chapter Calculations: ML Systems (ml_systems.qmd)
=================================================
Every derived number in the ML Systems chapter is computed here.
The .qmd file imports this module and uses inline code to insert values.

To verify: python3 -c "from ch_ml_systems import C; print(vars(C))"
"""

from constants import *
import math

class C:
    """All computed values for ml_systems.qmd, organized by section."""

    # ══════════════════════════════════════════════════════════════════
    # ResNet-50 Cloud vs Mobile Worked Example (lines ~507-537)
    # ══════════════════════════════════════════════════════════════════

    # -- Given values (formatted for display) --
    resnet_gflops = f"{RESNET50_FLOPs / 1e9:.1f}"            # "4.1"
    resnet_params_m = f"{RESNET50_PARAMS / 1e6:.1f}"          # "25.6"

    # Model sizes at different precisions
    _resnet_bytes_fp32 = RESNET50_PARAMS * 4
    _resnet_bytes_fp16 = RESNET50_PARAMS * 2
    _resnet_bytes_int8 = RESNET50_PARAMS * 1
    resnet_mb_fp32 = f"{_resnet_bytes_fp32 / MB:.0f}"         # "102"
    resnet_mb_fp16 = f"{_resnet_bytes_fp16 / MB:.0f}"         # "51"
    resnet_mb_int8 = f"{_resnet_bytes_int8 / MB:.0f}"         # "26" (actually 25.6)

    # (a) Cloud: A100 FP16
    a100_tflops_fp16 = f"{A100_FLOPS_FP16_TENSOR / TB:.0f}"  # "312"
    a100_bw_tbs = f"{A100_MEM_BW / TB:.0f}"                   # "2"

    _cloud_compute_time = RESNET50_FLOPs / A100_FLOPS_FP16_TENSOR  # seconds
    _cloud_memory_time = _resnet_bytes_fp16 / A100_MEM_BW          # seconds
    cloud_compute_ms = f"{_cloud_compute_time / MS:.3f}"       # "0.013"
    cloud_memory_ms = f"{_cloud_memory_time / MS:.3f}"         # "0.026" (actually 0.025)

    _cloud_bottleneck_ratio = _cloud_memory_time / _cloud_compute_time
    cloud_bottleneck_ratio = f"{_cloud_bottleneck_ratio:.0f}"  # "2" (memory is 2x slower)
    cloud_bottleneck = "Memory" if _cloud_memory_time > _cloud_compute_time else "Compute"

    _cloud_arith_intensity = RESNET50_FLOPs / _resnet_bytes_fp16
    cloud_arith_intensity = f"{_cloud_arith_intensity:.0f}"    # "80"

    # (b) Mobile: Flagship NPU INT8
    mobile_tops_int8 = f"{MOBILE_NPU_TOPS_INT8 / TB:.0f}"     # "35"
    mobile_bw_gbs = f"{MOBILE_NPU_MEM_BW / GB:.0f}"           # "100"
    mobile_model_mb = f"{_resnet_bytes_int8 / MB:.0f}"         # "26" (INT8 quantized)

    _mobile_compute_time = RESNET50_FLOPs / MOBILE_NPU_TOPS_INT8   # seconds
    _mobile_memory_time = _resnet_bytes_int8 / MOBILE_NPU_MEM_BW   # seconds
    mobile_compute_ms = f"{_mobile_compute_time / MS:.2f}"     # "0.12"
    mobile_memory_ms = f"{_mobile_memory_time / MS:.2f}"       # "0.26"

    _mobile_bottleneck_ratio = _mobile_memory_time / _mobile_compute_time
    mobile_bottleneck_ratio = f"{_mobile_bottleneck_ratio:.0f}"
    mobile_bottleneck = "Memory" if _mobile_memory_time > _mobile_compute_time else "Compute"

    # Key insight: bandwidth ratio vs inference ratio
    _bw_ratio_cloud_mobile = A100_MEM_BW / MOBILE_NPU_MEM_BW
    _compute_ratio_cloud_mobile = A100_FLOPS_FP16_TENSOR / MOBILE_NPU_TOPS_INT8
    bw_ratio_cloud_mobile = f"{_bw_ratio_cloud_mobile:.0f}"    # "20"

    # Actual inference speedup (limited by memory, not compute)
    _inference_ratio = _mobile_memory_time / _cloud_memory_time
    inference_ratio_approx = f"{_inference_ratio:.0f}"          # ~10

    # ══════════════════════════════════════════════════════════════════
    # Factory Camera Bandwidth Bottleneck (lines ~869-880)
    # ══════════════════════════════════════════════════════════════════

    _num_cameras = 100
    _raw_bytes_per_frame = VIDEO_1080P_WIDTH * VIDEO_1080P_HEIGHT * VIDEO_BYTES_PER_PIXEL_RGB
    _raw_bytes_per_sec = _raw_bytes_per_frame * VIDEO_FPS_STANDARD

    raw_rate_per_camera_mbs = f"{_raw_bytes_per_sec / MB:.0f}"     # "187"
    total_rate_gbs = f"{_num_cameras * _raw_bytes_per_sec / GB:.1f}"  # "18.7"

    # Cloud upload cost (24/7 streaming)
    _monthly_bytes = _num_cameras * _raw_bytes_per_sec * 3600 * 24 * 30
    _monthly_cost = (_monthly_bytes / GB) * CLOUD_EGRESS_PER_GB
    monthly_cloud_cost = f"{_monthly_cost / 1e6:.1f}"           # millions $/month
    # NOTE: At raw 18.7 GB/s × $0.09/GB, this is ~$4.4M/month.
    # The original text said "$145,000/month" which appears to be an error
    # (possibly calculated for compressed video or a single camera).

    # Network reality
    _network_capacity = NETWORK_10G_BW  # 1.25 GB/s
    _total_rate = _num_cameras * _raw_bytes_per_sec
    _bw_shortage = _total_rate / _network_capacity
    network_capacity_gbs = f"{_network_capacity / GB:.2f}"      # "1.25"
    bw_shortage_ratio = f"{_bw_shortage:.0f}"                   # "15"

    # Edge vs cloud data reduction
    _edge_metadata_bytes = 1 * KB  # ~1 KB per detection
    _reduction_factor = _raw_bytes_per_sec / _edge_metadata_bytes
    edge_data_reduction = f"{_reduction_factor:,.0f}"

    # ══════════════════════════════════════════════════════════════════
    # 1000x Energy Gap (lines ~884-894)
    # ══════════════════════════════════════════════════════════════════

    tx_energy_mj = f"{NETWORK_5G_ENERGY_PER_MB_MJ}"           # "100"
    compute_energy_mj = f"{ENERGY_MOBILENET_INF_MJ}"           # "0.1"
    _energy_gap = NETWORK_5G_ENERGY_PER_MB_MJ / ENERGY_MOBILENET_INF_MJ
    energy_gap = f"{_energy_gap:.0f}"                          # "1000"

    # ══════════════════════════════════════════════════════════════════
    # Speed of Light Latency (line ~260)
    # ══════════════════════════════════════════════════════════════════

    # California to Virginia
    _ca_va_km = 3600  # straight-line distance
    _ca_va_rtt_s = (_ca_va_km * 2) / SPEED_OF_LIGHT_FIBER_KM_S
    ca_va_latency_ms = f"{_ca_va_rtt_s / MS:.0f}"             # "36"

    # Robot surgery example (line ~665)
    _surgery_km = 1500
    _surgery_rtt_s = (_surgery_km * 2) / SPEED_OF_LIGHT_FIBER_KM_S
    surgery_latency_ms = f"{_surgery_rtt_s / MS:.0f}"          # "15"

    # ══════════════════════════════════════════════════════════════════
    # YOLOv8-nano Factory Edge Compute (lines ~926-943)
    # ══════════════════════════════════════════════════════════════════

    _num_edge_cameras = 20
    _edge_fps = 15
    _inferences_per_sec = _num_edge_cameras * _edge_fps
    _yolo_compute_per_sec = YOLOV8_NANO_FLOPs * _inferences_per_sec

    yolo_gflops = f"{YOLOV8_NANO_FLOPs / 1e9:.1f}"            # "3.2"
    yolo_total_gflops_sec = f"{_yolo_compute_per_sec / 1e9:.0f}"  # "960"
    yolo_with_headroom_tflops = f"{_yolo_compute_per_sec * 2 / TB:.0f}"  # "~2"
