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
