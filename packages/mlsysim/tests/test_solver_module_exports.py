import importlib

import pytest

import mlsysim
import mlsysim.ops as ops
import mlsysim.solvers as public_solvers
import mlsysim.engine.solvers as engine_solvers


def test_engine_solver_shim_stays_deleted():
    # 2026-06-06 no-backward-compat policy: the middle re-export module
    # (mlsysim.engine.solver) was removed; mlsysim.engine.solvers is the
    # canonical implementation package and mlsysim.solvers the public path.
    # This pin keeps the shim from quietly coming back.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("mlsysim.engine.solver")


def test_public_solvers_mirror_engine_solvers_exactly():
    # mlsysim.solvers derives mechanically from engine.solvers.__all__, so the
    # two surfaces must expose identical names bound to identical objects.
    assert sorted(public_solvers.__all__) == sorted(engine_solvers.__all__)
    for name in engine_solvers.__all__:
        assert getattr(public_solvers, name) is getattr(engine_solvers, name)


def test_solver_implementations_live_in_domain_modules():
    assert engine_solvers.SingleNodeModel.__module__.endswith(".solvers.performance")
    assert engine_solvers.DistributedModel.__module__.endswith(".solvers.distributed")
    assert engine_solvers.TrainingMemoryModel.__module__.endswith(".solvers.training")
    assert engine_solvers.ServingModel.__module__.endswith(".solvers.serving")
    assert engine_solvers.EconomicsModel.__module__.endswith(".solvers.economics")
    assert engine_solvers.DataModel.__module__.endswith(".solvers.data")
    assert engine_solvers.CompressionModel.__module__.endswith(".solvers.compression")


def test_package_root_does_not_reexport_solver_aliases():
    root_only = (
        "SingleNodeModel",
        "DistributedModel",
        "ReliabilityModel",
        "SustainabilityModel",
        "EconomicsModel",
        "ServingModel",
        "TrainingMemoryModel",
        "ServingCapacityModel",
        "DataModel",
        "PlacementOptimizer",
        "Monitoring",
    )

    assert mlsysim.Ops.Monitoring.__name__ == "Monitoring"
    assert not hasattr(ops, "Monitoring")
    for name in root_only:
        assert not hasattr(mlsysim, name)
