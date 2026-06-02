from __future__ import annotations

import pickle

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

import mlsysim
from mlsysim import Hardware, Models
from mlsysim.core.loader import load_collection, load_registry
from mlsysim.hardware.types import HardwareNode


class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str


def test_load_collection_rejects_duplicate_yaml_keys(tmp_path):
    yaml_file = tmp_path / "entries.yaml"
    yaml_file.write_text(
        """
Duplicate:
  name: first
Duplicate:
  name: second
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_collection(yaml_file, _Entry, name="Entries")


def test_load_registry_rejects_duplicate_registry_keys(tmp_path):
    (tmp_path / "a.yaml").write_text("__key__: Duplicate\nname: first\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("__key__: Duplicate\nname: second\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate registry key"):
        load_registry(tmp_path, _Entry, name="Entries")


def test_load_collection_rejects_unknown_schema_fields(tmp_path):
    yaml_file = tmp_path / "entries.yaml"
    yaml_file.write_text(
        """
Entry:
  name: example
  stray_field: should-fail
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="Extra inputs|extra_forbidden"):
        load_collection(yaml_file, _Entry, name="Entries")


def test_hardware_list_excludes_technology_reference_tiers():
    items = Hardware.list()

    assert items
    assert all(isinstance(item, HardwareNode) for item in items)
    assert "NVLink" not in {item.name for item in items}


def test_hardware_sorting_normalizes_pint_units():
    names = [item.name for item in Hardware.list(sort_by="compute.peak_flops", reverse=True)]

    assert names[0] == "NVIDIA GB200 NVL72"
    assert names.index("Cerebras CS-3 (WSE-3)") < names.index("NVIDIA B200")


def test_generated_registry_classes_are_picklable_by_public_module():
    assert Models.Language.__module__ == "mlsysim.models.registry"
    assert pickle.loads(pickle.dumps(Models.Language)) is Models.Language


def test_package_scenarios_export_names_executable_bundles():
    assert hasattr(mlsysim.Scenarios, "ChatbotServing")
    assert hasattr(mlsysim.Scenarios, "SmartDoorbell")
    assert hasattr(mlsysim.ReferenceStats, "MobilePower")
    assert not hasattr(mlsysim, "ScenarioBundles")
    assert not hasattr(mlsysim, "Applications")
    assert not hasattr(mlsysim.Scenarios, "Models")
    assert not hasattr(mlsysim.Scenarios, "Hardware")
    assert not hasattr(mlsysim.Scenarios, "Clusters")
    assert not hasattr(mlsysim.Scenarios, "Nodes")


def test_scenario_reuses_reference_stats_for_waymo_data_rate():
    assert (
        mlsysim.Scenarios.AutonomousVehicle_Waymo.workload.data_rate
        is mlsysim.ReferenceStats.Workloads.WaymoDataPerHourHigh
    )


def test_mobile_power_scenario_ranges_remain_available():
    mobile_power = mlsysim.ReferenceStats.MobilePower

    assert mobile_power.MobileMlSustainedLow.m_as("watt") == 2
    assert mobile_power.MobileMlSustainedHigh.m_as("watt") == 3
    assert mobile_power.MobileMlBurstLow.m_as("watt") == 5
    assert mobile_power.MobileMlBurstHigh.m_as("watt") == 10
    assert mobile_power.BackgroundAdaptationLow.m_as("milliwatt") == 500
    assert mobile_power.BackgroundAdaptationHigh.m_as("milliwatt") == 1000


def test_models_have_no_flat_leaf_aliases():
    assert not hasattr(mlsysim.Models, "GPT3")
    assert not hasattr(mlsysim.Models, "ResNet50")
