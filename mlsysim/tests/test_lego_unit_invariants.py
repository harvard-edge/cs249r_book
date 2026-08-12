"""Golden invariants for LEGO unit discipline."""

from __future__ import annotations

import pytest

from mlsysim.core.units import (
    Bparam,
    GB,
    GiB,
    Q_,
    TB,
    TFLOP,
    byte,
    hour,
    kWh,
    metric_ton,
    MWh,
    second,
    watt,
)
from mlsysim.hardware.registry import Hardware
from mlsysim.physics.quantities import (
    carbon_from_energy,
    compute_time,
    energy_from_power,
    memory_from_params,
    transfer_time,
)


def test_bandwidth_to_time_invariant():
    t = transfer_time(Q_("16 GB"), Q_("3.35 TB/s"))
    assert t.to("ms").magnitude == pytest.approx(4.8, rel=0.02)


def test_compute_to_time_invariant():
    t = compute_time(Q_("989 TFLOP"), Q_("989 TFLOP/s"))
    assert t.to("second").magnitude == pytest.approx(1.0)


def test_params_to_memory_invariant():
    mem = memory_from_params(Q_("7 Bparam"), Q_("2 byte / param"))
    assert mem.to(GB).magnitude == pytest.approx(14.0)


def test_energy_from_power_invariant():
    e = energy_from_power(Q_("700 W"), Q_("1 hour"))
    assert e.to(kWh).magnitude == pytest.approx(0.7)


def test_carbon_from_energy_invariant():
    carbon = carbon_from_energy(Q_("1287 MWh"), Q_("429 gram / kWh"))
    assert carbon.to(metric_ton).magnitude == pytest.approx(552, rel=0.01)


def test_h100_registry_invariants():
    h100 = Hardware.Cloud.H100
    assert h100.memory.capacity.to(GiB).magnitude == pytest.approx(80)
    assert h100.memory.bandwidth.to(TB / second).magnitude == pytest.approx(
        3.35, rel=0.01
    )
    assert h100.compute.peak_flops.to(TFLOP / second).magnitude == pytest.approx(
        989, rel=0.01
    )
