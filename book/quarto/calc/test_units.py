"""
Unit conversion validation for the textbook's physics engine.

Run:  python3 book/quarto/calc/test_units.py
From: repository root (mlsysbook-vols/)

Catches regressions where pint .to() returns raw base-unit magnitudes
instead of human-readable values (e.g., 312000000000000 instead of 312).

Also validates that every unit alias used in QMD files is properly
registered in pint's registry, not just a Python variable.
"""

import sys
sys.path.insert(0, "book/quarto")

from calc.constants import *
from calc.formulas import *

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
    ok &= check("Mobile 35 TFLOPs/s", MOBILE_NPU_TOPS_INT8.to(TFLOPs / second).magnitude, 35.0)

    # Model FLOPs
    ok &= check("ResNet 4.1 GFLOPs", RESNET50_FLOPs.to(GFLOPs).magnitude, 4.1)
    ok &= check("YOLO 3.2 GFLOPs", YOLOV8_NANO_FLOPs.to(GFLOPs).magnitude, 3.2)
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


# ── 10. Formula Helper Functions ─────────────────────────────────────

def test_formula_helpers():
    """Verify fmt() and sci() produce correct formatted strings."""
    ok = True

    # fmt()
    ok &= check("fmt GB/s", float(fmt(A100_MEM_BW, "GB/s", precision=0, commas=False)), 2039.0)
    ok &= check("fmt ms", float(fmt(RESNET50_FLOPs / A100_FLOPS_FP16_TENSOR, "ms", 3, commas=False)), 0.013, tol=0.1)

    # sci() - check format, not value
    result = sci(RESNET50_FLOPs)
    if "\\times 10^{" not in result:
        FAILURES.append(f"  ✗ sci() format: got '{result}', expected LaTeX scientific notation")
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
        ("Formula helpers (fmt, sci)", test_formula_helpers),
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
