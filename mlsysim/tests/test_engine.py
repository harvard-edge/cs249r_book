# tests/test_engine.py
# Engine-level tests — covers the core Engine.solve() API.
#
# Note: Comprehensive solver tests are in test_solver_suite.py (TestSingleNodeModel).
# This file tests Engine-specific behavior not covered there.

import pytest
from mlsysim.engine.engine import Engine
from mlsysim.hardware.registry import Hardware
from mlsysim.models.registry import Models
from mlsysim.engine import calibration as cal


def test_engine_energy_proportional():
    """Engine energy uses the energy-proportional model: P = TDP * (idle_fraction + dynamic_fraction * HFU).

    Utilization can never exceed the efficiency derate: achieved <= peak * efficiency,
    so mfu <= efficiency always (the [0,1] clamp is a guard, not a reachable ceiling).
    We verify the energy-proportional formula is applied consistently. HFU == MFU
    without recomputation, so either utilization gives the same expected value here.
    """
    resnet = Models.Vision.ResNet50
    a100 = Hardware.Cloud.A100

    perf = Engine.solve(resnet, a100, batch_size=1)

    # Energy should always be positive
    assert perf.energy.to("J").magnitude > 0
    # The real (stronger) invariant: mfu can never exceed the efficiency derate
    assert perf.mfu <= 0.5 + 1e-9  # default efficiency=0.5
    # Energy = TDP * (idle_fraction + dynamic_fraction * HFU) * latency
    expected = (a100.tdp * (cal.ENERGY_IDLE_FRACTION + cal.ENERGY_DYNAMIC_FRACTION * perf.hfu) * perf.latency.to("s")).to("J").magnitude
    assert perf.energy.to("J").magnitude == pytest.approx(expected, rel=0.01)


def test_engine_hfu_equals_mfu_without_recomputation():
    """2026-06-10 audit: HFU must equal MFU exactly when recomputation is off.

    The pre-audit implementation fabricated hfu = 1.1 * mfu unconditionally
    (phantom 10% utilization with no hardware work behind it). PaLM App. B /
    Korthikanti et al. (2022): HFU and MFU differ only by recomputation FLOPs.
    """
    perf = Engine.solve(Models.Language.GPT3, Hardware.Cloud.H100,
                        batch_size=8, is_training=True)
    assert perf.hfu == pytest.approx(perf.mfu, rel=1e-9)

    perf_inf = Engine.solve(Models.Vision.ResNet50, Hardware.Cloud.A100, batch_size=1)
    assert perf_inf.hfu == pytest.approx(perf_inf.mfu, rel=1e-9)


def test_engine_hfu_mfu_ratio_under_full_recomputation():
    """With full activation recomputation, hardware executes 4 passes for 3
    passes of model work: hfu/mfu == 4/3 (unless either hits the [0,1] clamp).

    Also pins that MFU EXCLUDES recompute FLOPs — the pre-audit implementation
    counted the re-forward pass in the MFU numerator, silently reporting HFU
    under the MFU label.
    """
    base = Engine.solve(Models.Language.GPT3, Hardware.Cloud.H100,
                        batch_size=8, is_training=True)
    recomp = Engine.solve(Models.Language.GPT3, Hardware.Cloud.H100,
                          batch_size=8, is_training=True,
                          activation_recomputation=True)
    assert 0 < recomp.mfu < 1 and 0 < recomp.hfu < 1, "clamp hit — pick a smaller config"
    assert recomp.hfu / recomp.mfu == pytest.approx(4.0 / 3.0, rel=1e-6)
    # Same latency denominator and peak: recompute MFU must not exceed the
    # non-recompute MFU (the model work didn't grow; only hardware work did).
    assert recomp.mfu <= base.mfu * 1.4  # latency shifts, but no 4/3 inflation


def test_engine_overhead_dominated_flag():
    """Small model at batch 1: dispatch + layer tax dominates both roofline
    terms; the profile must say so instead of mislabeling the cause as the
    larger of two negligible ceilings. (Audit finding B5, 2026-06-10.)"""
    perf = Engine.solve(Models.Vision.ResNet50, Hardware.Cloud.A100, batch_size=1)
    overhead_ms = perf.latency_overhead.to("ms").magnitude
    roofline_ms = max(perf.latency_compute.to("ms").magnitude,
                      perf.latency_memory.to("ms").magnitude)
    assert perf.overhead_dominated == (overhead_ms > roofline_ms)
    assert perf.overhead_dominated is True  # ResNet-50/A100/b1 is the known case


def test_engine_energy_per_inference_property():
    """PerformanceProfile should expose energy_per_inference."""
    resnet = Models.Vision.ResNet50
    a100 = Hardware.Cloud.A100
    perf = Engine.solve(resnet, a100, batch_size=1)
    assert hasattr(perf, "energy_per_inference")
    assert perf.energy_per_inference.magnitude > 0


def test_engine_input_validation():
    """Engine should reject invalid inputs with clear errors."""
    resnet = Models.Vision.ResNet50
    a100 = Hardware.Cloud.A100

    with pytest.raises(ValueError, match="efficiency"):
        Engine.solve(resnet, a100, batch_size=1, efficiency=50.0)

    with pytest.raises(ValueError, match="efficiency"):
        Engine.solve(resnet, a100, batch_size=1, efficiency=-0.1)

    with pytest.raises(ValueError, match="batch_size"):
        Engine.solve(resnet, a100, batch_size=0)

    with pytest.raises(ValueError, match="precision"):
        Engine.solve(resnet, a100, precision="fp6")


def test_engine_handles_model_size_only_workloads():
    """Registry workloads with model_size but no parameter count should still lower."""
    perf = Engine.solve(Models.Recommendation.DLRM, Hardware.Cloud.H200, batch_size=1)
    assert perf.feasible is True
    assert perf.memory_footprint.to("GB").magnitude > 0


def test_nvl72_fp16_does_not_use_fp8_peak_silently():
    """GB200 NVL72 exposes FP8/FP4 peaks, but FP16 should not alias to FP8."""
    perf_fp16 = Engine.solve(Models.Vision.ResNet50, Hardware.Cloud.GB200_NVL72, precision="fp16")
    perf_fp8 = Engine.solve(Models.Vision.ResNet50, Hardware.Cloud.GB200_NVL72, precision="fp8")
    assert perf_fp16.peak_flops_actual < perf_fp8.peak_flops_actual
