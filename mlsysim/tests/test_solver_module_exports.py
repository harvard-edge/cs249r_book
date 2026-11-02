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


def test_package_root_exposes_solvers_subpackage():
    assert hasattr(mlsysim, "solvers")
    assert mlsysim.solvers is public_solvers
    assert mlsysim.solvers.SingleNodeModel is engine_solvers.SingleNodeModel


def test_plot_roofline_efficiency_and_batch_size():
    plt = pytest.importorskip("matplotlib.pyplot")
    from mlsysim.viz.plots import plot_roofline

    fig1, ax1 = plot_roofline(
        mlsysim.Hardware.Cloud.A100,
        workloads=[mlsysim.Models.Vision.ResNet50],
        efficiency=1.0,
        batch_size=1,
        precision="fp16",
    )
    fig2, ax2 = plot_roofline(
        mlsysim.Hardware.Cloud.A100,
        workloads=[mlsysim.Models.Vision.ResNet50],
        efficiency=0.5,
        batch_size=1,
        precision="fp16",
    )
    assert fig1 is not None and ax1 is not None
    assert fig2 is not None and ax2 is not None
    plt.close(fig1)
    plt.close(fig2)


def test_quantity_properties_on_results_registries_and_sourced():
    from mlsysim.core.provenance import Provenance, ProvenanceKind, Sourced
    from mlsysim.core.units import Q_

    # Sourced .magnitude and .to()
    prov = Provenance(kind=ProvenanceKind.DATASHEET, ref="Test", url="https://example.com", verified="2026-03-06")
    s_float = Sourced(1.12, prov)
    assert s_float.magnitude == pytest.approx(1.12)
    assert s_float.to().magnitude == pytest.approx(1.12)
    assert s_float.to("dimensionless").magnitude == pytest.approx(1.12)

    # GridProfile, CoolingProfile, RackProfile Quantity properties
    grid = mlsysim.Infrastructure.Grids.Iowa
    assert grid.carbon_intensity.to("g / kWh").magnitude == pytest.approx(grid.carbon_intensity_g_kwh)
    assert grid.water_intensity.to("L / kWh").magnitude == pytest.approx(grid.wue)

    cooling = mlsysim.Infrastructure.FacilityCooling.LiquidCooled
    assert cooling.water_intensity.to("L / kWh").magnitude == pytest.approx(cooling.wue)

    rack = mlsysim.Infrastructure.Racks.AI_Standard
    assert rack.power.to("kW").magnitude == pytest.approx(rack.power_kw)

    # HardwareNode embodied_carbon Quantity property
    a100 = mlsysim.Hardware.Cloud.A100
    assert a100.embodied_carbon is not None
    assert a100.embodied_carbon.to("kg").magnitude == pytest.approx(a100.embodied_carbon_kg)

    # SustainabilityResult Quantity properties
    fleet = mlsysim.Systems.Clusters.Lab_64_A100
    sust = mlsysim.solvers.SustainabilityModel().solve(
        fleet=fleet,
        duration_days=30.0,
        datacenter=mlsysim.Infrastructure.Datacenters.Iowa_Reference,
    )
    assert sust.it_energy.to("kWh").magnitude == pytest.approx(sust.it_energy_kwh)
    assert sust.total_energy.to("kWh").magnitude == pytest.approx(sust.total_energy_kwh)
    assert sust.carbon_footprint.to("kg").magnitude == pytest.approx(sust.carbon_footprint_kg)
    assert sust.water_usage.to("L").magnitude == pytest.approx(sust.water_usage_liters)
    assert sust.embodied_carbon.to("kg").magnitude == pytest.approx(sust.embodied_carbon_kg)

    # EconomicsResult Quantity properties
    econ = mlsysim.solvers.EconomicsModel().solve(
        fleet=fleet,
        duration_days=365.0,
        kwh_price=0.08,
    )
    assert econ.tco.to("USD").magnitude == pytest.approx(econ.tco_usd)
    assert econ.capex.to("USD").magnitude == pytest.approx(econ.capex_usd)
    assert econ.total_opex.to("USD").magnitude == pytest.approx(econ.total_opex_usd)
    assert econ.carbon_footprint.to("kg").magnitude == pytest.approx(econ.carbon_footprint_kg)

