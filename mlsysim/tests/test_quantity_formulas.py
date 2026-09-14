"""Tests for quantity-first physics helpers."""

from __future__ import annotations

import pytest
import pint

from mlsysim.core.units import (
    Bparam,
    GB,
    Q_,
    TB,
    TFLOP,
    byte,
    count,
    gram,
    hour,
    kWh,
    metric_ton,
    MWh,
    param,
    second,
    ureg,
    watt,
)
from mlsysim.physics.quantities import (
    carbon_from_energy,
    compute_time,
    energy_from_power,
    memory_from_params,
    token_throughput,
    transfer_time,
)


def test_transfer_time_bandwidth_golden():
    t = transfer_time(Q_("16 GB"), Q_("3.35 TB/s"))
    assert t.to("ms").magnitude == pytest.approx(4.78, rel=0.01)


def test_compute_time_golden():
    t = compute_time(Q_("989 TFLOP"), Q_("989 TFLOP/s"))
    assert t.to("s").magnitude == pytest.approx(1.0)


def test_energy_from_power_golden():
    e = energy_from_power(Q_("700 W"), Q_("1 hour"))
    assert e.to("kWh").magnitude == pytest.approx(0.7)


def test_carbon_from_energy_golden():
    carbon = carbon_from_energy(Q_("1287 MWh"), Q_("429 gram / kWh"))
    assert carbon.to(metric_ton).magnitude == pytest.approx(552, rel=0.01)


def test_memory_from_params_golden():
    mem = memory_from_params(Q_("7 Bparam"), Q_("2 byte / param"))
    assert mem.to(GB).magnitude == pytest.approx(14.0)


def test_token_throughput():
    rate = token_throughput(Q_("1000 count"), Q_("2 second"))
    assert rate.to("count / second").magnitude == pytest.approx(500.0)


def test_wrong_dimensions_raise():
    with pytest.raises(pint.DimensionalityError):
        transfer_time(Q_("1 second"), Q_("1 GB/s"))
    with pytest.raises(pint.DimensionalityError):
        compute_time(Q_("1 second"), Q_("1 TFLOP/s"))
    with pytest.raises(pint.DimensionalityError):
        energy_from_power(Q_("1 GB"), Q_("1 hour"))
