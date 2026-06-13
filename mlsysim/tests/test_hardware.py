import pytest
from pydantic import ValidationError
from mlsysim.hardware import Hardware, HardwareNode
from mlsysim.core.units import Q_, ureg

def test_hardware_registry():
    a100 = Hardware.Cloud.A100
    assert a100.name == "NVIDIA A100"
    assert a100.release_year == 2020
    assert a100.compute.peak_flops.magnitude == 312.0  # Dense FP16 Tensor Core

    # Check ridge point calculation
    ridge = a100.ridge_point()
    assert "flop/B" in str(ridge.units) or "flop / byte" in str(ridge.units)
    assert 100 < ridge.magnitude < 200  # ~153 flop/byte (312 TFLOPS / 2.039 TB/s)

def test_hardware_validation():
    # Should raise error on invalid quantity string
    with pytest.raises(ValidationError):
        HardwareNode(
            name="Broken",
            release_year=2025,
            compute={"peak_flops": "not a number"},
            memory={"capacity": "10 GiB", "bandwidth": "100 GB/s"}
        )

def test_json_serialization():
    a100 = Hardware.Cloud.A100
    json_data = a100.model_dump_json()
    assert "NVIDIA A100" in json_data
    assert "312" in json_data  # FP16 Tensor Core peak


def test_tpuv4_is_not_alias_to_tpuv5p():
    assert Hardware.Cloud.TPUv4.name == "Google TPU v4"
    assert Hardware.Cloud.TPUv4 is not Hardware.Cloud.TPUv5p
    assert Hardware.Cloud.TPUv4.compute.peak_flops == (275 * ureg.TFLOPs / ureg.second)


def test_nvlink_on_cloud_gpus():
    """NVLink bandwidth lives on HardwareNode, not loose constants."""
    from mlsysim.core.units import GB, second

    assert Hardware.Cloud.V100.nvlink.name == "NVLink 2.0"
    assert Hardware.Cloud.V100.nvlink.bandwidth.m_as(GB / second) == 300.0
    assert Hardware.Cloud.A100.nvlink.bandwidth.m_as(GB / second) == 600.0
    assert Hardware.Cloud.H100.nvlink.bandwidth.m_as(GB / second) == 900.0
    assert Hardware.Cloud.B200.nvlink.bandwidth.m_as(GB / second) == 1800.0


def test_lab_track_hardware_profiles():
    """Canonical lab tracks have MLSysIM-owned hardware profiles."""
    oura = Hardware.Tiny.OuraRing
    robotaxi = Hardware.Edge.RoboTaxi

    assert oura.name == "Oura Ring 4 (wearable reference profile)"
    assert oura.memory.sram_capacity.to("KiB").magnitude == pytest.approx(512)
    assert oura.memory.flash_capacity.to("MB").magnitude == pytest.approx(2)
    assert oura.battery_capacity.to("Wh").magnitude == pytest.approx(0.06)
    assert oura.metadata.provenance.kind.value == "estimate"

    assert robotaxi.name == "RoboTaxi Reference Compute (NVIDIA DRIVE AGX Orin class)"
    assert robotaxi.compute.precision_flops["int8"].to("TOPS").magnitude == pytest.approx(254)
    assert robotaxi.memory.capacity.to("GB").magnitude == pytest.approx(32)
    assert robotaxi.tdp.to("W").magnitude == pytest.approx(60)
    assert robotaxi.metadata.provenance.kind.value == "estimate"


def test_h100_die_area_is_registry_backed():
    assert Hardware.Cloud.H100.die_area is not None
    assert Hardware.Cloud.H100.die_area.m_as("mm^2") == pytest.approx(814.0)


def test_h100_unit_cost_range_is_registry_backed():
    assert Hardware.Cloud.H100.unit_cost is not None
    assert Hardware.Cloud.H100.unit_cost_max is not None
    assert Hardware.Cloud.H100.unit_cost.m_as("USD") == pytest.approx(25_000)
    assert Hardware.Cloud.H100.unit_cost_max.m_as("USD") == pytest.approx(30_000)


def test_memory_tech_bandwidth_tiers():
    """Memory-interface bandwidth tiers live in Hardware.Tech.Memory."""
    from mlsysim.core.units import GB, second

    assert Hardware.Tech.Memory.DDR4_3200.bandwidth.m_as(GB / second) == pytest.approx(51.2)
    assert Hardware.Tech.Memory.HBM2.bandwidth.m_as(GB / second) == pytest.approx(900)
    assert Hardware.Tech.Memory.HBM3.bandwidth.m_as(GB / second) == pytest.approx(1600)
    assert Hardware.Tech.Memory.GDDR6X.bandwidth.m_as(GB / second) == pytest.approx(760)


def test_interconnect_direction_convention():
    """2026-06-10 audit (findings_provenance.md M1/M2): NVLink datasheet
    figures are bidirectional totals; PCIe figures are per-direction. The
    schema must declare the convention and the per-direction accessor must
    halve only the bidirectional entries."""
    from mlsysim.hardware.registry import Hardware
    from mlsysim.core.units import ureg

    h100 = Hardware.Cloud.H100
    assert h100.nvlink.direction == "bidirectional_total"
    assert h100.nvlink.bandwidth.m_as("GB/s") == 900.0
    assert h100.nvlink.bandwidth_per_direction.m_as("GB/s") == 450.0
    # PCIe stays per-direction: accessor is the identity
    assert h100.interconnect.direction == "per_direction"
    assert h100.interconnect.bandwidth_per_direction == h100.interconnect.bandwidth
    # All NVLink-family entries are tagged
    for dev in ("V100", "A100", "H200", "B200", "TPUv5p"):
        nv = getattr(Hardware.Cloud, dev).nvlink
        assert nv.direction == "bidirectional_total", dev
