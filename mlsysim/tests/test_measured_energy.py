import importlib.util
from pathlib import Path

import pytest

from mlsysim import Q_
from mlsysim.core.provenance import Provenance, ProvenanceKind
from mlsysim.engine.measured_energy import (
    coldstart_energy,
    load_measured,
    measured_power,
    standby_power,
)
from mlsysim.engine.solvers import TemporalShiftModel
from mlsysim.hardware.registry import Hardware

BUILDER = Path(__file__).resolve().parents[1] / "paper" / "scripts" / "build_measured_gpu_energy.py"


def _builder():
    spec = importlib.util.spec_from_file_location("build_measured_gpu_energy", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_registry_entries_load():
    assert Hardware.Cloud.L4.tdp.m_as("W") == 72
    assert Hardware.Cloud.A10.tdp.m_as("W") == 150
    assert Hardware.Cloud.L4.compute.peak_flops.m_as("TFLOPs/s") == 121.0
    assert Hardware.Cloud.A10.memory.bandwidth.m_as("GB/s") == 600.0
    for gpu in (Hardware.Cloud.L4, Hardware.Cloud.A10):
        assert gpu.metadata.provenance.url.startswith("https://www.nvidia.com/")


def test_measured_table_matches_its_source_data():
    builder = _builder()
    assert builder.TABLE.read_text() == builder.build()


def test_manifest_covers_every_run():
    import csv

    data = _builder().DATA
    manifest = {r["run_id"]: r for r in csv.DictReader(open(data / "runs_manifest.csv"))}
    for name in ("idle_runs", "coldstart_runs", "pause_steps", "power_runs"):
        for row in csv.DictReader(open(data / f"{name}.csv")):
            entry = manifest[row["run_id"]]
            assert entry["meta_json_sha256"] == row["meta_sha256"], row["run_id"]
            assert entry["gpu_name"] == row["gpu_name"], row["run_id"]
            assert len(entry["samples_nvml_parquet_sha256"]) == 64, row["run_id"]


def test_measured_table_is_complete():
    data = load_measured()
    provenance = Provenance(**data["provenance"])
    assert provenance.kind == ProvenanceKind.DERIVED
    for gpu, g in data["gpus"].items():
        assert 0 < g["loaded_idle_power_w"] < g["tdp_w"], gpu
        assert g["coldstart_energy_j"] > 0 and g["reserved_idle_power_w"] > 0, gpu
        assert g["coldstart_repeats"] >= 3, gpu
        assert getattr(Hardware.Cloud, gpu).tdp.m_as("W") == g["tdp_w"], gpu


def test_measured_power_is_capped_and_monotone():
    for gpu, g in load_measured()["gpus"].items():
        lo, hi = measured_power(gpu, 0.0, 0.0).m_as("W"), measured_power(gpu, 1.0, 1.0).m_as("W")
        assert abs(lo - g["loaded_idle_power_w"]) < 1e-9
        assert lo <= measured_power(gpu, 0.1, 0.1).m_as("W") <= hi <= g["tdp_w"] + 1e-9


def test_measured_power_rejects_bad_inputs():
    with pytest.raises(ValueError):
        measured_power("H100", 1.5, 0.0)
    with pytest.raises(KeyError):
        measured_power("TPUv4", 0.1, 0.1)


def test_waiting_quantities_come_from_the_table():
    g = load_measured()["gpus"]["H100"]
    assert standby_power("H100").m_as("W") == g["reserved_idle_power_w"]
    assert coldstart_energy("H100").m_as("J") == g["coldstart_energy_j"]
    with pytest.raises(KeyError):
        standby_power("TPUv4")


def test_standby_power_is_charged_at_each_waiting_step():
    # A dirty release hour and clean later hours: the standby charge must use the
    # intensity of each step actually spent waiting, and a start now pays nothing.
    ci_g = [900.0, 50.0, 60.0, 900.0, 900.0]
    power = measured_power("H100", 0.3, 0.0)
    standby = standby_power("H100")
    result = TemporalShiftModel().solve(
        [Q_(c, "g/kWh") for c in ci_g], power, Q_(1, "hour"), Q_(4, "hour"), standby_power=standby
    )
    run_kwh, standby_kwh = power.m_as("kW"), standby.m_as("kW")
    for k, net in enumerate(result.net_emissions_by_start):
        expected_g = run_kwh * ci_g[k] + standby_kwh * sum(ci_g[:k])
        assert net.m_as("g") == pytest.approx(expected_g, rel=1e-12)
    assert result.net_emissions_by_start[0].m_as("g") == pytest.approx(run_kwh * ci_g[0], rel=1e-12)
    assert result.best_start.m_as("hour") == 1.0


def test_coldstart_is_not_a_release_time_deferral():
    # TemporalShiftModel charges deferral_energy at the release step's intensity, but a
    # released GPU pays its cold start when the job restarts. With a dirty release hour
    # and a clean restart hour the two charges differ, which is why coldstart_energy()
    # is not offered as a deferral_energy input.
    ci_g = [900.0, 50.0, 50.0]
    energy = coldstart_energy("H100")
    result = TemporalShiftModel().solve(
        [Q_(c, "g/kWh") for c in ci_g], Q_(0, "kW"), Q_(1, "hour"), Q_(2, "hour"), deferral_energy=energy
    )
    charged_at_release_g = result.net_emissions_by_start[1].m_as("g")
    restart_charge_g = energy.m_as("kWh") * ci_g[1]
    assert charged_at_release_g == pytest.approx(energy.m_as("kWh") * ci_g[0], rel=1e-12)
    assert restart_charge_g < charged_at_release_g / 10
