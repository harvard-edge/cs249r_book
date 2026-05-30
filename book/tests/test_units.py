"""
Unit conversion validation for the textbook's physics engine.

Run:  python3 book/quarto/mlsys/test_units.py
From: repository root (mlsysbook-vols/)

Catches regressions where pint .to() returns raw base-unit magnitudes
instead of human-readable values (e.g., 312000000000000 instead of 312).

Also validates that every unit alias used in QMD files is properly
registered in pint's registry, not just a Python variable.
"""

import sys
import os

# Make `mlsysim` importable without requiring a pip install.
#
# Layout reminder:
#   <repo_root>/mlsysim/                ← project directory (NOT a Python package)
#   <repo_root>/mlsysim/mlsysim/        ← actual Python package (has __init__.py)
#   <repo_root>/mlsysim/mlsysim/core/   ← submodule
#
# So the directory we must put on sys.path is `<repo_root>/mlsysim/`, NOT
# `<repo_root>/`. The previous `_repo_root` insert "worked" only in dev
# environments where `mlsysim` was pip-installed (e.g. `pip install -e mlsysim/`),
# masking the bug. CI has no editable install, so the import failed there.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_book_dir = os.path.dirname(_script_dir)                  # book/
_repo_root = os.path.dirname(_book_dir)                   # repo root
_mlsysim_project = os.path.join(_repo_root, "mlsysim")    # contains the package
sys.path.insert(0, _mlsysim_project)
sys.path.insert(0, _book_dir)

from mlsysim.core.constants import *
from mlsysim.physics import *
from mlsysim.fmt import fmt, fmt_sci, fmt_unit
from mlsysim import Hardware, Models, Systems, Literature, Scenarios

# Legacy aliases for unit tests (hardware/model specs live in registries).
_a100 = Hardware.Cloud.A100
_v100 = Hardware.Cloud.V100
_h100 = Hardware.Cloud.H100
_t4 = Hardware.Cloud.T4
_resnet50 = Models.Vision.ResNet50
_gpt3 = Models.Language.GPT3
_gpt2 = Models.Language.GPT2
_yolov8_nano = Models.Vision.YOLOv8_Nano
_mobilenetv2 = Models.Vision.MobileNetV2
_bert_base = Models.Language.BERT_Base
_llama3_8b = Models.Language.Llama3_8B
_b200 = Hardware.Cloud.B200
_mi300x = Hardware.Cloud.MI300X
_tpuv4 = Hardware.Cloud.TPUv4
_tpuv6 = Hardware.Cloud.TPUv6
_mobile = Hardware.Mobile.iPhone15Pro

A100_FLOPS_FP16_TENSOR = _a100.compute.peak_flops
A100_FLOPS_FP32 = _a100.compute.precision_flops["fp32"]
A100_FLOPS_TF32 = _a100.compute.precision_flops["tf32"]
A100_TOPS_INT8 = _a100.compute.precision_flops["int8"]
A100_MEM_BW = _a100.memory.bandwidth
A100_MEM_CAPACITY = _a100.memory.capacity
A100_TDP = _a100.tdp

V100_FLOPS_FP16_TENSOR = _v100.compute.peak_flops
V100_MEM_BW = _v100.memory.bandwidth
V100_MEM_CAPACITY = _v100.memory.capacity
V100_TDP = _v100.tdp

H100_FLOPS_FP16_TENSOR = _h100.compute.peak_flops
H100_FLOPS_TF32 = _h100.compute.precision_flops["tf32"]
H100_TOPS_INT8 = _h100.compute.precision_flops["int8"]
H100_MEM_BW = _h100.memory.bandwidth
H100_MEM_CAPACITY = _h100.memory.capacity
H100_TDP = _h100.tdp

T4_FLOPS_FP16_TENSOR = _t4.compute.peak_flops
T4_TDP = _t4.tdp

RESNET50_PARAMS = _resnet50.parameters
RESNET50_FLOPs = _resnet50.inference_flops
GPT3_PARAMS = _gpt3.parameters
GPT2_PARAMS = _gpt2.parameters
YOLOV8_NANO_FLOPs = _yolov8_nano.inference_flops
BERT_BASE_PARAMS = _bert_base.parameters
BERT_BASE_FLOPs = _bert_base.inference_flops
MOBILENETV2_PARAMS = _mobilenetv2.parameters
MOBILENETV2_FLOPs = _mobilenetv2.inference_flops
LLAMA3_8B_PARAMS = _llama3_8b.parameters

B200_FLOPS_FP16_TENSOR = _b200.compute.peak_flops
B200_MEM_BW = _b200.memory.bandwidth
B200_MEM_CAPACITY = _b200.memory.capacity

MI300X_FLOPS_FP16_TENSOR = _mi300x.compute.peak_flops
MI300X_MEM_BW = _mi300x.memory.bandwidth

TPUV4_FLOPS_BF16 = _tpuv4.compute.precision_flops["bf16"]
TPUV4_MEM_BW = _tpuv4.memory.bandwidth

TPUV6_FLOPS_BF16 = _tpuv6.compute.peak_flops
TPUV6_MEM_BW = _tpuv6.memory.bandwidth

MOBILE_NPU_TOPS_INT8 = _mobile.compute.peak_flops
MOBILE_NPU_MEM_BW = _mobile.memory.bandwidth

NVLINK_V100_BW = _v100.nvlink.bandwidth
NVLINK_A100_BW = _a100.nvlink.bandwidth
NVLINK_H100_BW = _h100.nvlink.bandwidth
PCIE_GEN4_BW = _a100.interconnect.bandwidth
PCIE_GEN5_BW = _h100.interconnect.bandwidth
INFINIBAND_HDR_BW = Systems.Fabrics.InfiniBand_HDR.bandwidth
INFINIBAND_NDR_BW = Systems.Fabrics.InfiniBand_NDR.bandwidth

GPT3_TRAINING_OPS = (
    Literature.Chinchilla.ComputeConstant * _gpt3.parameters * _gpt3.training_tokens
)

FAILURES = []


def check(label, actual, expected, tol=0.01):
    """Assert a value is within tolerance of expected."""
    rel_err = abs(actual - expected) / expected if expected != 0 else abs(actual)
    if rel_err > tol:
        FAILURES.append(f"  ✗ {label}: got {actual}, expected {expected} (err={rel_err:.2%})")
        return False
    return True


# ── 1. Unit Registration ─────────────────────────────────────────────

def test_units_are_registered():
    """Every scale alias must be a pint Unit, not a raw Quantity.

    If a unit is just a Quantity (e.g., GB = 1e9 * byte), then:
      - .to(GB) won't work correctly
      - value / GB won't scale the magnitude
    Registered Units fix both issues.
    """
    ok = True
    units_to_check = {
        'KB': KB, 'MB': MB, 'GB': GB, 'TB': TB, 'PB': PB,
        'KiB': KiB, 'MiB': MiB, 'GiB': GiB, 'TiB': TiB,
        'GFLOPs': GFLOPs, 'TFLOPs': TFLOPs, 'ZFLOPs': ZFLOPs,
        'Mparam': Mparam,
        'Gbps': Gbps,
        'MS': MS, 'US': US, 'NS': NS,
    }
    for name, unit in units_to_check.items():
        if not isinstance(unit, type(ureg.byte)):  # pint.Unit
            FAILURES.append(f"  ✗ {name} is {type(unit).__name__}, not a registered pint Unit")
            ok = False
    return ok


# ── 2. Data Unit Conversions ─────────────────────────────────────────

def test_data_units():
    """Verify byte-scale .to() conversions return human-readable magnitudes."""
    ok = True
    ok &= check("1 KB -> KB", (1 * KB).to(KB).magnitude, 1.0)
    ok &= check("1 MB -> MB", (1 * MB).to(MB).magnitude, 1.0)
    ok &= check("1 GB -> GB", (1 * GB).to(GB).magnitude, 1.0)
    ok &= check("1 TB -> TB", (1 * TB).to(TB).magnitude, 1.0)
    ok &= check("1 PB -> PB", (1 * PB).to(PB).magnitude, 1.0)
    ok &= check("1 KiB -> KiB", (1 * KiB).to(KiB).magnitude, 1.0)
    ok &= check("1 GiB -> GiB", (1 * GiB).to(GiB).magnitude, 1.0)
    ok &= check("1 TiB -> TiB", (1 * TiB).to(TiB).magnitude, 1.0)

    # Cross-scale
    ok &= check("1 TB -> GB", (1 * TB).to(GB).magnitude, 1000.0)
    ok &= check("1 GB -> MB", (1 * GB).to(MB).magnitude, 1000.0)
    ok &= check("1 GiB -> MiB", (1 * GiB).to(MiB).magnitude, 1024.0)

    # Rate conversions
    ok &= check("900 GB/s -> GB/s", (900 * GB / second).to(GB / second).magnitude, 900.0)
    ok &= check("2039 GB/s -> GB/s", (2039 * GB / second).to(GB / second).magnitude, 2039.0)
    ok &= check("1 TB/hr -> TB/hr", (1 * TB / hour).to(TB / hour).magnitude, 1.0)
    ok &= check("3.35 TB/s -> GB/s", (3.35 * TB / second).to(GB / second).magnitude, 3350.0)
    return ok


# ── 3. FLOP Unit Conversions ─────────────────────────────────────────

def test_flop_units():
    """Verify FLOP-scale .to() conversions."""
    ok = True
    ok &= check("1 GFLOPs -> GFLOPs", (1 * GFLOPs).to(GFLOPs).magnitude, 1.0)
    ok &= check("1 TFLOPs -> TFLOPs", (1 * TFLOPs).to(TFLOPs).magnitude, 1.0)
    ok &= check("1 TFLOPs -> GFLOPs", (1 * TFLOPs).to(GFLOPs).magnitude, 1000.0)
    ok &= check("1 TOPS -> TOPS", (1 * TOPS).to(TOPS).magnitude, 1.0)

    # Hardware specs
    ok &= check("A100 312 TFLOPs/s", A100_FLOPS_FP16_TENSOR.to(TFLOPs / second).magnitude, 312.0)
    ok &= check("V100 125 TFLOPs/s", V100_FLOPS_FP16_TENSOR.to(TFLOPs / second).magnitude, 125.0)
    ok &= check("H100 989 TFLOPs/s", H100_FLOPS_FP16_TENSOR.to(TFLOPs / second).magnitude, 989.0)
    ok &= check("T4 65 TFLOPs/s", T4_FLOPS_FP16_TENSOR.to(TFLOPs / second).magnitude, 65.0)
    ok &= check("Mobile 35 TOPS", MOBILE_NPU_TOPS_INT8.to(TOPS).magnitude, 35.0)

    # Model FLOPs
    ok &= check("ResNet 4.1 GFLOPs", RESNET50_FLOPs.to(GFLOPs).magnitude, 4.1)
    ok &= check("YOLOv8-nano 8.7 GFLOPs", YOLOV8_NANO_FLOPs.to(GFLOPs).magnitude, 8.7)
    return ok


# ── 4. Parameter Unit Conversions ────────────────────────────────────

def test_param_units():
    """Verify parameter-scale .to() conversions."""
    ok = True
    ok &= check("ResNet 25.6 Mparam", RESNET50_PARAMS.to(Mparam).magnitude, 25.6)
    ok &= check("GPT-3 175000 Mparam", GPT3_PARAMS.to(Mparam).magnitude, 175000.0)
    return ok


# ── 5. Network Unit Conversions ──────────────────────────────────────

def test_network_units():
    """Verify network-scale conversions."""
    ok = True
    ok &= check("10 Gbps -> Gbps", Systems.Fabrics.Ethernet_10G.bandwidth.to(Gbps).magnitude, 10.0)
    ok &= check("10 Gbps -> GB/s", Systems.Fabrics.Ethernet_10G.bandwidth.to(GB / second).magnitude, 1.25)
    return ok


# ── 6. Memory Bandwidth Conversions ─────────────────────────────────

def test_memory_bandwidth():
    """Verify hardware memory bandwidth conversions."""
    ok = True
    ok &= check("A100 2039 GB/s", A100_MEM_BW.to(GB / second).magnitude, 2039.0)
    ok &= check("A100 ~2 TB/s", A100_MEM_BW.to(TB / second).magnitude, 2.039)
    ok &= check("V100 900 GB/s", V100_MEM_BW.to(GB / second).magnitude, 900.0)
    ok &= check("H100 3350 GB/s", H100_MEM_BW.to(GB / second).magnitude, 3350.0)
    ok &= check("Mobile 100 GB/s", MOBILE_NPU_MEM_BW.to(GB / second).magnitude, 100.0)
    return ok


# ── 7. Memory Capacity Conversions ──────────────────────────────────

def test_memory_capacity():
    """Verify hardware memory capacity conversions."""
    ok = True
    ok &= check("A100 80 GiB", A100_MEM_CAPACITY.to(GiB).magnitude, 80.0)
    ok &= check("V100 32 GiB", V100_MEM_CAPACITY.to(GiB).magnitude, 32.0)
    ok &= check("H100 80 GiB", H100_MEM_CAPACITY.to(GiB).magnitude, 80.0)
    return ok


# ── 8. Derived Calculation Sanity ────────────────────────────────────

def test_derived_values():
    """Verify key derived calculations produce sane values."""
    ok = True

    # GPT-3 training time
    days = calc_training_time_days(GPT3_TRAINING_OPS, 1024, A100_FLOPS_FP16_TENSOR, 0.45)
    ok &= check("GPT-3 training ~25 days", days, 25.0, tol=0.1)

    # Waymo data rates
    ok &= check("Waymo low 1 TB/hr", Scenarios.Workloads.WaymoDataPerHourLow.to(TB / hour).magnitude, 1.0)
    ok &= check("Waymo high 19 TB/hr", Scenarios.Workloads.WaymoDataPerHourHigh.to(TB / hour).magnitude, 19.0)

    # ResNet model sizes (params * bytes_per_param)
    fp32_bytes = RESNET50_PARAMS.magnitude * 4 * byte
    fp16_bytes = RESNET50_PARAMS.magnitude * 2 * byte
    int8_bytes = RESNET50_PARAMS.magnitude * 1 * byte
    ok &= check("ResNet FP32 ~102 MB", fp32_bytes.to(MB).magnitude, 102.4)
    ok &= check("ResNet FP16 ~51 MB", fp16_bytes.to(MB).magnitude, 51.2)
    ok &= check("ResNet INT8 ~26 MB", int8_bytes.to(MB).magnitude, 25.6)

    # Camera bandwidth (ml_systems.qmd worked example)
    raw_bps = (1920 * 1080 * 3 * byte * 30 * ureg.Hz).to('byte/second')
    ok &= check("1080p camera ~187 MB/s", raw_bps.to(MB / second).magnitude, 186.624, tol=0.01)

    # Roofline analysis
    result = calc_bottleneck(RESNET50_FLOPs, fp16_bytes, A100_FLOPS_FP16_TENSOR, A100_MEM_BW)
    ok &= check("A100 ResNet compute_ms small", result["compute_ms"], 0.013, tol=0.1)
    ok &= check("A100 ResNet memory_ms small", result["memory_ms"], 0.025, tol=0.1)

    return ok


# ── 9. Sentinel: No Suspiciously Large Magnitudes ────────────────────

def test_no_large_raw_magnitudes():
    """
    Catch any .to() that returns suspiciously large numbers,
    indicating the conversion fell through to base units.
    """
    ok = True
    conversions = [
        ("A100 TFLOPs/s", A100_FLOPS_FP16_TENSOR.to(TFLOPs / second).magnitude, 1e4),
        ("A100 GB/s", A100_MEM_BW.to(GB / second).magnitude, 1e4),
        ("H100 GB/s", H100_MEM_BW.to(GB / second).magnitude, 1e4),
        ("H100 TB/s", H100_MEM_BW.to(TB / second).magnitude, 100),
        ("Waymo TB/hr", Scenarios.Workloads.WaymoDataPerHourHigh.to(TB / hour).magnitude, 100),
        ("ResNet GFLOPs", RESNET50_FLOPs.to(GFLOPs).magnitude, 1e4),
        ("ResNet Mparam", RESNET50_PARAMS.to(Mparam).magnitude, 1e4),
        ("A100 GiB", A100_MEM_CAPACITY.to(GiB).magnitude, 1e4),
        ("10G Gbps", Systems.Fabrics.Ethernet_10G.bandwidth.to(Gbps).magnitude, 100),
    ]
    for label, value, threshold in conversions:
        if abs(value) > threshold:
            FAILURES.append(
                f"  ✗ {label}: magnitude {value} exceeds {threshold} — likely raw base units!"
            )
            ok = False
    return ok


# ── 10. Time Unit Conversions ──────────────────────────────────────────

def test_time_units():
    """Verify registered time units convert correctly."""
    ok = True
    ok &= check("1 s -> MS", (1 * second).to(MS).magnitude, 1000.0)
    ok &= check("1 s -> US", (1 * second).to(US).magnitude, 1e6)
    ok &= check("1 s -> NS", (1 * second).to(NS).magnitude, 1e9)
    ok &= check("1 MS -> MS", (1 * MS).to(MS).magnitude, 1.0)
    ok &= check("1000 US -> MS", (1000 * US).to(MS).magnitude, 1.0)
    ok &= check("1000 NS -> US", (1000 * NS).to(US).magnitude, 1.0)
    return ok


# ── 11. Extended GPU Specs ───────────────────────────────────────────

def test_extended_gpu_specs():
    """Verify all GPU spec constants convert correctly."""
    ok = True

    # A100 full spec sheet
    ok &= check("A100 FP32", A100_FLOPS_FP32.to(TFLOPs / second).magnitude, 19.5)
    ok &= check("A100 TF32", A100_FLOPS_TF32.to(TFLOPs / second).magnitude, 156.0)
    ok &= check("A100 INT8", A100_TOPS_INT8.to(TOPS).magnitude, 624.0)
    ok &= check("A100 TDP", A100_TDP.to(watt).magnitude, 400.0)

    # H100 full spec sheet (Dense values; Sparse is 2x)
    ok &= check("H100 TF32", H100_FLOPS_TF32.to(TFLOPs / second).magnitude, 494.0)
    ok &= check("H100 INT8", H100_TOPS_INT8.to(TOPS).magnitude, 1979.0)
    ok &= check("H100 TDP", H100_TDP.to(watt).magnitude, 700.0)

    # V100
    ok &= check("V100 TDP", V100_TDP.to(watt).magnitude, 300.0)

    # B200
    ok &= check("B200 FP16", B200_FLOPS_FP16_TENSOR.to(TFLOPs / second).magnitude, 2250.0)
    ok &= check("B200 BW", B200_MEM_BW.to(TB / second).magnitude, 8.0)
    ok &= check("B200 Mem", B200_MEM_CAPACITY.to(GiB).magnitude, 192.0)

    # MI300X
    ok &= check("MI300X FP16", MI300X_FLOPS_FP16_TENSOR.to(TFLOPs / second).magnitude, 1307.0, tol=0.01)
    ok &= check("MI300X BW", MI300X_MEM_BW.to(TB / second).magnitude, 5.3)

    # TPUv4
    ok &= check("TPUv4 BF16", TPUV4_FLOPS_BF16.to(TFLOPs / second).magnitude, 275.0)
    ok &= check("TPUv4 BW", TPUV4_MEM_BW.to(GB / second).magnitude, 1200.0)

    # TPUv6
    ok &= check("TPUv6 BF16", TPUV6_FLOPS_BF16.to(TFLOPs / second).magnitude, 918.0)
    ok &= check("TPUv6 BW", TPUV6_MEM_BW.to(GB / second).magnitude, 1600.0)

    # T4
    ok &= check("T4 TDP", T4_TDP.to(watt).magnitude, 70.0)

    return ok


# ── 12. Interconnect Conversions ─────────────────────────────────────

def test_interconnect_specs():
    """Verify interconnect bandwidth conversions."""
    ok = True
    ok &= check("NVLink V100", NVLINK_V100_BW.to(GB / second).magnitude, 300.0)
    ok &= check("NVLink A100", NVLINK_A100_BW.to(GB / second).magnitude, 600.0)
    ok &= check("NVLink H100", NVLINK_H100_BW.to(GB / second).magnitude, 900.0)
    ok &= check("PCIe Gen4", PCIE_GEN4_BW.to(GB / second).magnitude, 32.0)
    ok &= check("PCIe Gen5", PCIE_GEN5_BW.to(GB / second).magnitude, 64.0)
    ok &= check("IB HDR", INFINIBAND_HDR_BW.to(Gbps).magnitude, 200.0)
    ok &= check("IB NDR", INFINIBAND_NDR_BW.to(Gbps).magnitude, 400.0)
    ok &= check("100G net", Systems.Fabrics.Ethernet_100G.bandwidth.to(Gbps).magnitude, 100.0)
    ok &= check("100G -> GB/s", Systems.Fabrics.Ethernet_100G.bandwidth.to(GB / second).magnitude, 12.5)
    return ok


# ── 13. Energy Conversions ───────────────────────────────────────────

def test_energy_specs():
    """Verify energy constants are consistent."""
    ok = True
    # Op/memory energy is a tech-class fact (Hardware.Tech), not a flat constant.
    op = Hardware.Tech.Op
    mem = Hardware.Tech.Memory
    # FP32 > FP16 > INT8 (energy ordering)
    fp32 = op.FlopFp32.energy.magnitude
    fp16 = op.FlopFp16.energy.magnitude
    int8 = op.OpInt8.energy.m_as(ureg.picojoule / count)
    if not (fp32 > fp16 > int8):
        FAILURES.append(f"  ✗ Energy ordering: FP32={fp32} > FP16={fp16} > INT8={int8}")
        ok = False
    ok &= check("DRAM >> compute", mem.HBM3.energy_per_access.magnitude / op.FlopFp32.energy.magnitude, 173.0, tol=0.05)
    ok &= check("L2 > L1 > reg",
                 mem.L2.energy_per_access.magnitude / mem.L1.energy_per_access.magnitude, 4.0)
    return ok


# ── 14. Model Spec Conversions ───────────────────────────────────────

def test_model_specs():
    """Verify model constants convert correctly."""
    ok = True
    ok &= check("GPT-2 1500 Mparam", GPT2_PARAMS.to(Mparam).magnitude, 1500.0)
    ok &= check("BERT 110 Mparam", BERT_BASE_PARAMS.to(Mparam).magnitude, 110.0)
    ok &= check("MobileNetV2 3.5 Mparam", MOBILENETV2_PARAMS.to(Mparam).magnitude, 3.5)
    ok &= check("Llama3-8B 8030 Mparam", LLAMA3_8B_PARAMS.to(Mparam).magnitude, 8030.0)
    ok &= check("BERT 22 GFLOPs", BERT_BASE_FLOPs.to(GFLOPs).magnitude, 22.0)
    ok &= check("MobileNetV2 0.3 GFLOPs", MOBILENETV2_FLOPs.to(GFLOPs).magnitude, 0.3)
    return ok


# ── 15. Ridge Point Derivations ──────────────────────────────────────

def test_ridge_points():
    """Verify ridge point (arithmetic intensity threshold) calculations.

    Ridge point = peak FLOPs / peak bandwidth (FLOP/byte).
    Below this intensity, a workload is memory-bound; above, compute-bound.
    """
    ok = True

    def ridge(flops, bw):
        return (flops / bw).to(flop / byte).magnitude

    ok &= check("V100 ridge ~139", ridge(V100_FLOPS_FP16_TENSOR, V100_MEM_BW), 139.0, tol=0.02)
    ok &= check("A100 ridge ~153", ridge(A100_FLOPS_FP16_TENSOR, A100_MEM_BW), 153.0, tol=0.02)
    ok &= check("H100 ridge ~295", ridge(H100_FLOPS_FP16_TENSOR, H100_MEM_BW), 295.0, tol=0.02)
    return ok


# ── 16. Formula Helper Functions ─────────────────────────────────────

def test_formula_helpers():
    """Verify fmt(), fmt_unit(), and fmt_sci() produce correct formatted strings."""
    ok = True

    # fmt()
    ok &= check("fmt GB/s", float(fmt(A100_MEM_BW, "GB/s", precision=0, commas=False)), 2039.0)
    ok &= check("fmt ms", float(fmt(RESNET50_FLOPs / A100_FLOPS_FP16_TENSOR, "ms", 3, commas=False)), 0.013, tol=0.1)

    # fmt_unit()
    if fmt_unit(A100_FLOPS_FP16_TENSOR) != "TFLOP/s":
        FAILURES.append(
            f"  ✗ fmt_unit() FLOP rate notation: got '{fmt_unit(A100_FLOPS_FP16_TENSOR)}', expected 'TFLOP/s'"
        )
        ok = False
    if fmt_unit(1 * GFLOPs) != "GFLOPs":
        FAILURES.append(
            f"  ✗ fmt_unit() FLOP count notation: got '{fmt_unit(1 * GFLOPs)}', expected 'GFLOPs'"
        )
        ok = False

    # fmt_sci() - check format, not value
    result = fmt_sci(RESNET50_FLOPs)
    if "× 10" not in result or not any(c in result for c in '⁰¹²³⁴⁵⁶⁷⁸⁹'):
        FAILURES.append(f"  ✗ fmt_sci() format: got '{result}', expected Unicode scientific notation (e.g., 4.10 × 10⁹)")
        ok = False

    return ok


# ── 17. Robustness: Wrong-Unit HardwareNode ──────────────────────────

def test_hardware_wrong_unit_raises():
    """HardwareNode stores Quantities; validate that construction works with correct types."""
    from mlsysim.hardware.types import HardwareNode, ComputeCore, MemoryHierarchy
    ok = True
    # Verify that a correctly-constructed HardwareNode works
    try:
        node = HardwareNode(
            name="Good", release_year=2024,
            compute=ComputeCore(peak_flops=312 * TFLOPs / second),
            memory=MemoryHierarchy(
                capacity=80 * GiB,
                bandwidth=2039 * GB / second,
            ))
        # Verify canonical nested access to compute/memory specs
        assert node.compute.peak_flops == 312 * TFLOPs / second
        assert node.memory.capacity == 80 * GiB
        assert node.memory.bandwidth == 2039 * GB / second
    except Exception as e:
        FAILURES.append(f"  ✗ HardwareNode construction failed unexpectedly: {e}")
        ok = False
    return ok


# ── 18. Robustness: model_memory() Wrong Units ───────────────────────

def test_model_memory_wrong_units():
    """Passing watts as params to model_memory() must raise DimensionalityError."""
    import pint
    ok = True
    try:
        model_memory(100 * watt, BYTES_FP32)
        FAILURES.append("  ✗ model_memory() accepted wrong params unit (watt) silently")
        ok = False
    except (pint.DimensionalityError, ValueError):
        pass  # Expected
    return ok


# ── 19. Fleet Formulas: Quantity Inputs + Quantity Returns ────────────

def test_fleet_formulas_accept_quantities():
    """Fleet formulas must accept Pint Quantities and return typed Quantities."""
    ok = True

    # calc_ring_allreduce_time: Quantity inputs → Quantity[second] return
    result = calc_ring_allreduce_time(1 * GB, 256, 50 * GB / second, 5e-6 * second)
    if not isinstance(result, ureg.Quantity):
        FAILURES.append(f"  ✗ calc_ring_allreduce_time must return Quantity, got {type(result).__name__}")
        ok = False
    elif not result.is_compatible_with(ureg.second):
        FAILURES.append(f"  ✗ calc_ring_allreduce_time must return Quantity[second], got {result.units}")
        ok = False

    # calc_failure_probability: both Quantities → plain float (dimensionless)
    result2 = calc_failure_probability(1000 * ureg.hour, 24 * ureg.hour)
    if isinstance(result2, ureg.Quantity):
        FAILURES.append("  ✗ calc_failure_probability must return float, not Quantity")
        ok = False
    elif not (0 < result2 < 1):
        FAILURES.append(f"  ✗ calc_failure_probability result {result2} not in (0, 1)")
        ok = False

    # calc_checkpoint_size: returns Quantity[byte]
    ckpt = calc_checkpoint_size(7e9, 2)
    if not isinstance(ckpt, ureg.Quantity):
        FAILURES.append(f"  ✗ calc_checkpoint_size must return Quantity, got {type(ckpt).__name__}")
        ok = False
    elif not ckpt.is_compatible_with(ureg.byte):
        FAILURES.append(f"  ✗ calc_checkpoint_size must return Quantity[byte], got {ckpt.units}")
        ok = False

    # calc_mtbf_cluster: returns Quantity[hour]
    mtbf = calc_mtbf_cluster(100000 * ureg.hour, 1024)
    if not isinstance(mtbf, ureg.Quantity):
        FAILURES.append(f"  ✗ calc_mtbf_cluster must return Quantity, got {type(mtbf).__name__}")
        ok = False
    elif not mtbf.is_compatible_with(ureg.hour):
        FAILURES.append(f"  ✗ calc_mtbf_cluster must return Quantity[hour], got {mtbf.units}")
        ok = False

    return ok


# ── 20. Failure Probability: Mixed-Type Guard ─────────────────────────

def test_failure_probability_mixed_types():
    """Mixed Quantity + raw number in calc_failure_probability must raise TypeError."""
    ok = True
    try:
        calc_failure_probability(1000 * ureg.second, 24)   # Quantity + raw
        FAILURES.append("  ✗ calc_failure_probability accepted (Qty, raw) silently")
        ok = False
    except TypeError:
        pass  # Expected — mixed-type guard fired
    try:
        calc_failure_probability(1000, 24 * ureg.second)   # raw + Quantity
        FAILURES.append("  ✗ calc_failure_probability accepted (raw, Qty) silently")
        ok = False
    except TypeError:
        pass  # Expected
    return ok


# NOTE: tests for fmt_full() and fmt_split() were retired alongside the
# helpers themselves (2026-05). The two-pattern unification (fmt() with
# prefix=/suffix= params + fmt_val()/fmt_unit() for Pint-magnitude/unit
# split) replaced both. If you need to format "value unit" combined, use
# fmt(qty, suffix=" unit"); for separate columns, use fmt_val + fmt_unit.


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("Unit registration (all aliases are pint Units)", test_units_are_registered),
        ("Data unit conversions (KB → PB, rates)", test_data_units),
        ("FLOP unit conversions (GFLOPs, TFLOPs)", test_flop_units),
        ("Parameter unit conversions (Mparam)", test_param_units),
        ("Network unit conversions (Gbps)", test_network_units),
        ("Memory bandwidth conversions", test_memory_bandwidth),
        ("Memory capacity conversions", test_memory_capacity),
        ("Derived calculations (training, roofline)", test_derived_values),
        ("No raw base-unit magnitudes (sentinel)", test_no_large_raw_magnitudes),
        ("Time unit conversions (MS, US, NS)", test_time_units),
        ("Extended GPU specs (A100/H100/B200/TPU)", test_extended_gpu_specs),
        ("Interconnect specs (NVLink, PCIe, IB)", test_interconnect_specs),
        ("Energy specs (FP32 > FP16 > INT8)", test_energy_specs),
        ("Model specs (GPT-2, BERT, MobileNet)", test_model_specs),
        ("Ridge point derivations (V100/A100/H100)", test_ridge_points),
        ("Formula helpers (fmt, fmt_sci)", test_formula_helpers),
        # ── Robustness tests (Changes 1-6 verification) ──────────────────
        ("Robustness: wrong-unit HardwareSpec raises DimensionalityError", test_hardware_wrong_unit_raises),
        ("Robustness: model_memory() rejects wrong-unit params", test_model_memory_wrong_units),
        ("Fleet formulas: accept Quantities, return typed Quantities", test_fleet_formulas_accept_quantities),
        ("Fleet formulas: failure probability rejects mixed types", test_failure_probability_mixed_types),
    ]

    all_ok = True
    for name, fn in tests:
        FAILURES.clear()
        result = fn()
        status = "PASS" if result else "FAIL"
        print(f"[{status}] {name}")
        for f in FAILURES:
            print(f)
        all_ok &= result

    print()
    if all_ok:
        print("All unit tests passed ✓")
    else:
        print("FAILURES detected — fix issues above")
        sys.exit(1)
