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

# Add book/quarto to path (works from any working directory)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_quarto_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _quarto_dir)

from mlsys.constants import *
from mlsys.formulas import *
from mlsys.formatting import fmt, sci, fmt_full, fmt_split

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

    # Hardware specs
    ok &= check("A100 312 TFLOPs/s", A100_FLOPS_FP16_TENSOR.to(TFLOPs / second).magnitude, 312.0)
    ok &= check("V100 125 TFLOPs/s", V100_FLOPS_FP16_TENSOR.to(TFLOPs / second).magnitude, 125.0)
    ok &= check("H100 989 TFLOPs/s", H100_FLOPS_FP16_TENSOR.to(TFLOPs / second).magnitude, 989.0)
    ok &= check("T4 65 TFLOPs/s", T4_FLOPS_FP16_TENSOR.to(TFLOPs / second).magnitude, 65.0)
    ok &= check("Mobile 50 TFLOPs/s", MOBILE_NPU_TOPS_INT8.to(TFLOPs / second).magnitude, 50.0)

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
    ok &= check("10 Gbps -> Gbps", NETWORK_10G_BW.to(Gbps).magnitude, 10.0)
    ok &= check("10 Gbps -> GB/s", NETWORK_10G_BW.to(GB / second).magnitude, 1.25)
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
    ok &= check("Waymo low 1 TB/hr", WAYMO_DATA_PER_HOUR_LOW.to(TB / hour).magnitude, 1.0)
    ok &= check("Waymo high 19 TB/hr", WAYMO_DATA_PER_HOUR_HIGH.to(TB / hour).magnitude, 19.0)

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
        ("Waymo TB/hr", WAYMO_DATA_PER_HOUR_HIGH.to(TB / hour).magnitude, 100),
        ("ResNet GFLOPs", RESNET50_FLOPs.to(GFLOPs).magnitude, 1e4),
        ("ResNet Mparam", RESNET50_PARAMS.to(Mparam).magnitude, 1e4),
        ("A100 GiB", A100_MEM_CAPACITY.to(GiB).magnitude, 1e4),
        ("10G Gbps", NETWORK_10G_BW.to(Gbps).magnitude, 100),
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
    ok &= check("A100 INT8", A100_FLOPS_INT8.to(TFLOPs / second).magnitude, 624.0)
    ok &= check("A100 TDP", A100_TDP.to(watt).magnitude, 400.0)

    # H100 full spec sheet (Dense values; Sparse is 2x)
    ok &= check("H100 TF32", H100_FLOPS_TF32.to(TFLOPs / second).magnitude, 494.0)
    ok &= check("H100 INT8", H100_FLOPS_INT8.to(TFLOPs / second).magnitude, 1979.0)
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
    ok &= check("TPUv6 BF16", TPUV6_FLOPS_BF16.to(TFLOPs / second).magnitude, 2150.0)
    ok &= check("TPUv6 BW", TPUV6_MEM_BW.to(TB / second).magnitude, 4.5)

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
    ok &= check("100G net", NETWORK_100G_BW.to(Gbps).magnitude, 100.0)
    ok &= check("100G -> GB/s", NETWORK_100G_BW.to(GB / second).magnitude, 12.5)
    return ok


# ── 13. Energy Conversions ───────────────────────────────────────────

def test_energy_specs():
    """Verify energy constants are consistent."""
    ok = True
    # FP32 > FP16 > INT8 (energy ordering)
    fp32 = ENERGY_FLOP_FP32_PJ.magnitude
    fp16 = ENERGY_FLOP_FP16_PJ.magnitude
    int8 = ENERGY_FLOP_INT8_PJ.magnitude
    if not (fp32 > fp16 > int8):
        FAILURES.append(f"  ✗ Energy ordering: FP32={fp32} > FP16={fp16} > INT8={int8}")
        ok = False
    ok &= check("DRAM >> compute", ENERGY_DRAM_ACCESS_PJ.magnitude / ENERGY_FLOP_FP32_PJ.magnitude, 173.0, tol=0.05)
    ok &= check("L2 > L1 > reg",
                 ENERGY_SRAM_L2_PJ.magnitude / ENERGY_SRAM_L1_PJ.magnitude, 4.0)
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
    """Verify fmt() and sci() produce correct formatted strings."""
    ok = True

    # fmt()
    ok &= check("fmt GB/s", float(fmt(A100_MEM_BW, "GB/s", precision=0, commas=False)), 2039.0)
    ok &= check("fmt ms", float(fmt(RESNET50_FLOPs / A100_FLOPS_FP16_TENSOR, "ms", 3, commas=False)), 0.013, tol=0.1)

    # sci() - check format, not value
    result = sci(RESNET50_FLOPs)
    if "× 10" not in result or not any(c in result for c in '⁰¹²³⁴⁵⁶⁷⁸⁹'):
        FAILURES.append(f"  ✗ sci() format: got '{result}', expected Unicode scientific notation (e.g., 4.10 × 10⁹)")
        ok = False

    return ok


# ── 17. Robustness: Wrong-Unit HardwareSpec ──────────────────────────

def test_hardware_wrong_unit_raises():
    """Wrong-unit HardwareSpec must raise DimensionalityError, not pass silently."""
    from mlsys.hardware import HardwareSpec
    import pint
    ok = True
    # Passing watts as memory_bw (wrong dimension) must fail at construction
    try:
        HardwareSpec("Bad", 2024,
            memory_bw=100 * watt,           # WRONG — should be data/time
            peak_flops=312 * TFLOPs / second,
            memory_capacity=80 * GiB)
        FAILURES.append("  ✗ HardwareSpec accepted wrong memory_bw unit (watt) silently")
        ok = False
    except (pint.DimensionalityError, ValueError):
        pass  # Expected — dimension-first validation fired
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


# ── 21. fmt_full(): Compact String Display ────────────────────────────

def test_fmt_full_returns_string():
    """fmt_full must return a non-empty 'value unit' string for a valid Quantity."""
    ok = True
    result = fmt_full(A100_MEM_BW)
    if not isinstance(result, str):
        FAILURES.append(f"  ✗ fmt_full must return str, got {type(result).__name__}")
        ok = False
    elif " " not in result:
        FAILURES.append(f"  ✗ fmt_full result '{result}' must contain 'value unit' with space")
        ok = False
    return ok


def test_fmt_full_rejects_raw_number():
    """fmt_full() must reject raw numbers (int/float) with TypeError."""
    ok = True
    try:
        fmt_full(2039)
        FAILURES.append("  ✗ fmt_full() accepted raw int without raising TypeError")
        ok = False
    except TypeError:
        pass  # Expected
    try:
        fmt_full(2039.0)
        FAILURES.append("  ✗ fmt_full() accepted raw float without raising TypeError")
        ok = False
    except TypeError:
        pass  # Expected
    return ok


# ── 22. fmt_split(): Table Column Tuple ───────────────────────────────

def test_fmt_split_returns_tuple():
    """fmt_split must return a (value_str, unit_str) 2-tuple of non-empty strings."""
    ok = True
    result = fmt_split(A100_MEM_BW)
    if not isinstance(result, tuple) or len(result) != 2:
        FAILURES.append(f"  ✗ fmt_split must return 2-tuple, got {result!r}")
        ok = False
    else:
        val, unit = result
        if not isinstance(val, str) or not isinstance(unit, str):
            FAILURES.append(
                f"  ✗ fmt_split must return (str, str), got "
                f"({type(val).__name__}, {type(unit).__name__})"
            )
            ok = False
        elif not val or not unit:
            FAILURES.append(f"  ✗ fmt_split must return non-empty strings, got ({val!r}, {unit!r})")
            ok = False
    return ok


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
        ("Formula helpers (fmt, sci)", test_formula_helpers),
        # ── Robustness tests (Changes 1-6 verification) ──────────────────
        ("Robustness: wrong-unit HardwareSpec raises DimensionalityError", test_hardware_wrong_unit_raises),
        ("Robustness: model_memory() rejects wrong-unit params", test_model_memory_wrong_units),
        ("Fleet formulas: accept Quantities, return typed Quantities", test_fleet_formulas_accept_quantities),
        ("Fleet formulas: failure probability rejects mixed types", test_failure_probability_mixed_types),
        ("fmt_full(): returns compact 'value unit' string", test_fmt_full_returns_string),
        ("fmt_full(): rejects raw numbers with TypeError", test_fmt_full_rejects_raw_number),
        ("fmt_split(): returns (value_str, unit_str) tuple", test_fmt_split_returns_tuple),
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
