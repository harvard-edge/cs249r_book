"""Characterization tests for MLSysIM Pint registry and book-facing unit aliases."""

from __future__ import annotations

import pytest

from mlsysim.core.units import (
    Bparam,
    GB,
    GiB,
    Gbps,
    Kparam,
    Mparam,
    MS,
    NS,
    Q_,
    TB,
    TFLOP,
    TOPS,
    Tparam,
    US,
    byte,
    count,
    hour,
    joule,
    kilogram,
    kilowatt,
    km,
    kWh,
    megawatt,
    metric_ton,
    milliwatt,
    minute,
    mJ,
    MW,
    MWh,
    param,
    pJ,
    second,
    ureg,
    watt,
    Wh,
)


def test_decimal_data_scales():
    assert Q_("1 GB").to("byte").magnitude == pytest.approx(1e9)
    assert Q_("1 TB").to("byte").magnitude == pytest.approx(1e12)


def test_binary_data_scales():
    assert Q_("1 GiB").to("byte").magnitude == 1073741824


def test_flop_rate_dimensions():
    assert Q_("1 TFLOP/s").to(TFLOP / second).magnitude == pytest.approx(1)
    assert Q_("1 TOPS").to(TOPS).magnitude == pytest.approx(1)


def test_gbps_to_gigabytes_per_second():
    assert Q_("1 Gbps").to("GB/s").magnitude == pytest.approx(0.125)


def test_energy_aliases():
    assert Q_("1 kWh").to("J").magnitude == pytest.approx(3.6e6)
    assert Q_("1 MWh").to("kWh").magnitude == pytest.approx(1000)
    assert Wh == ureg.watt_hour


def test_param_scales():
    assert Q_("1 Bparam").to("param").magnitude == pytest.approx(1e9)


def test_legacy_time_aliases_match_si():
    assert Q_(1, MS).to("second").magnitude == pytest.approx(1e-3)
    assert Q_(1, US).to("second").magnitude == pytest.approx(1e-6)
    assert Q_(1, NS).to("second").magnitude == pytest.approx(1e-9)


def test_exported_aliases_match_registry():
    assert mJ == ureg.millijoule
    assert MW == ureg.megawatt
    assert kilowatt == ureg.kilowatt
    assert milliwatt == ureg.milliwatt
    assert kWh == ureg.kilowatt_hour
    assert kilogram == ureg.kilogram
    assert metric_ton == ureg.metric_ton
    assert km == ureg.kilometer
    assert pJ == ureg.picojoule
    assert minute == ureg.minute


def test_mlsysbook_units_file_loaded():
    """Custom units from mlsysbook_units.txt must parse like inline defines."""
    assert Q_("2.5 TFLOP").to(TFLOP).magnitude == pytest.approx(2.5)
    assert Q_("3 Bparam").to(Bparam).magnitude == pytest.approx(3)


def test_registry_yaml_strings_still_parse():
    assert Q_("80 GiB").to(GiB).magnitude == pytest.approx(80)
    assert Q_("3.35 TB/s").to(TB / second).magnitude == pytest.approx(3.35)


def test_gpt3_training_energy_is_quantity():
    from mlsysim import Models

    energy = Models.Language.GPT3.training_energy_mwh
    assert energy.to(MWh).magnitude == pytest.approx(1287, rel=1e-3)


def test_literature_transatlantic_flight_co2_anchor():
    from mlsysim import Literature
    from mlsysim.core.provenance import scalar_value

    anchor = Literature.Sustainability.TransatlanticRoundTripCo2Kg
    assert scalar_value(anchor) == pytest.approx(1000)
    assert anchor.provenance.ref
