from __future__ import annotations

import pytest

from mlsysbook_labs import resolve_mlsysim_ref


def test_resolve_mlsysim_model_and_hardware_refs():
    model = resolve_mlsysim_ref("Models.Tiny.DS_CNN")
    hardware = resolve_mlsysim_ref("Hardware.Tiny.OuraRing")

    assert model.name == "DS-CNN (KWS)"
    assert hardware.name == "Oura Ring 4 (wearable reference profile)"


def test_resolve_mlsysim_system_ref():
    system = resolve_mlsysim_ref("Systems.Clusters.Lab_64_H100")

    assert system.name == "Lab Cluster (64 H100 GPUs)"


def test_rejects_non_registry_roots():
    with pytest.raises(ValueError, match="Unsupported MLSysIM registry root"):
        resolve_mlsysim_ref("NotARegistry.Root")


def test_raises_key_error_for_missing_registry_path():
    with pytest.raises(KeyError, match="Could not resolve"):
        resolve_mlsysim_ref("Models.Tiny.NotARealModel")
