# tests/test_engine.py
# Engine-level tests — covers the core Engine.solve() API.
#
# Note: Comprehensive solver tests are in test_solver_suite.py (TestSingleNodeModel).
# This file tests Engine-specific behavior not covered there.

import pytest
from mlsysim.core.engine import Engine
from mlsysim.hardware.registry import Hardware
from mlsysim.models.registry import Models


def test_engine_energy_proportional():
    """Engine energy uses the energy-proportional model: P = TDP * (0.3 + 0.7 * MFU).

    For memory-bound workloads, MFU can reach 1.0 (the clamped ceiling) because
    achieved_flops/peak_flops exceeds 1 when latency is dominated by memory time.
    We verify the model applies correctly by checking energy > 0 and that the
    energy-proportional formula is consistent.
    """
    resnet = Models.ResNet50
    a100 = Hardware.A100

    perf = Engine.solve(resnet, a100, batch_size=1)

    # Energy should always be positive
    assert perf.energy.to("J").magnitude > 0
    # Energy = TDP * (0.3 + 0.7 * MFU) * latency
    expected = (a100.tdp * (0.3 + 0.7 * perf.mfu) * perf.latency.to("s")).to("J").magnitude
    assert perf.energy.to("J").magnitude == pytest.approx(expected, rel=0.01)


def test_engine_energy_per_inference_property():
    """PerformanceProfile should expose energy_per_inference."""
    resnet = Models.ResNet50
    a100 = Hardware.A100
    perf = Engine.solve(resnet, a100, batch_size=1)
    assert hasattr(perf, "energy_per_inference")
    assert perf.energy_per_inference.magnitude > 0


def test_engine_input_validation():
    """Engine should reject invalid inputs with clear errors."""
    resnet = Models.ResNet50
    a100 = Hardware.A100

    with pytest.raises(ValueError, match="efficiency"):
        Engine.solve(resnet, a100, batch_size=1, efficiency=50.0)

    with pytest.raises(ValueError, match="efficiency"):
        Engine.solve(resnet, a100, batch_size=1, efficiency=-0.1)

    with pytest.raises(ValueError, match="batch_size"):
        Engine.solve(resnet, a100, batch_size=0)


def test_engine_handles_model_size_only_workloads():
    """Registry workloads with model_size but no parameter count should still lower."""
    perf = Engine.solve(Models.DLRM, Hardware.H200, batch_size=1)
    assert perf.feasible is True
    assert perf.memory_footprint.to("GB").magnitude > 0


def test_nvl72_fp16_does_not_use_fp8_peak_silently():
    """GB200 NVL72 exposes FP8/FP4 peaks, but FP16 should not alias to FP8."""
    perf_fp16 = Engine.solve(Models.ResNet50, Hardware.NVL72, precision="fp16")
    perf_fp8 = Engine.solve(Models.ResNet50, Hardware.NVL72, precision="fp8")
    assert perf_fp16.peak_flops_actual < perf_fp8.peak_flops_actual
