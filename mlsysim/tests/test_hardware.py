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


def test_tpuv5p_vmem_capacity():
    assert Hardware.Cloud.TPUv5p.memory.sram_capacity == (128 * ureg.MiB)


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
    assert oura.memory.flash_capacity.to("MiB").magnitude == pytest.approx(2)
    assert oura.battery_capacity.to("Wh").magnitude == pytest.approx(0.06)
    assert oura.metadata.provenance.kind.value == "estimate"

    assert robotaxi.name == "RoboTaxi Reference Compute (NVIDIA DRIVE AGX Orin class)"
    assert robotaxi.compute.precision_flops["int8"].to("TOPS").magnitude == pytest.approx(254)
    assert robotaxi.memory.capacity.to("GiB").magnitude == pytest.approx(32)
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


def _all_hardware_nodes():
    """Every HardwareNode across the accelerator categories (Tech holds
    tech-classes, not nodes, so it is excluded)."""
    nodes = []
    for cat in ("Cloud", "Edge", "Mobile", "Tiny", "Workstation"):
        reg = getattr(Hardware, cat, None)
        if reg is None:
            continue
        for attr in dir(reg):
            if attr.startswith("_"):
                continue
            obj = getattr(reg, attr)
            if isinstance(obj, HardwareNode):
                nodes.append(obj)
    return nodes


def test_precision_keys_use_canonical_vocabulary():
    """Guard against the 'canonical precision aliased away' trap (F1, 2026-06).

    Engine.solve maps a user precision through resolve_precision() to a
    PRECISION_MAP key, looks that key up in compute.precision_flops, and falls
    through to peak_flops (the FP16 dense rate) when the key is absent. That
    fall-through is correct ONLY for fp16/bf16 (peak == FP16 dense); for any
    other precision it silently substitutes the FP16 rate.

    H100/H200 once stored FP32 under 'fp32_cuda' (not 'fp32'), so
    Engine.solve(..., precision='fp32') used 989 TFLOP/s instead of the stored
    67 — a ~15x overestimate, reachable by no precision string. This fails if a
    chip stores an off-vocabulary alias of a canonical precision (e.g.
    'fp32_cuda') without also exposing the canonical key. Genuinely-extra labels
    that are NOT a PRECISION_MAP precision (fp16_sparse, fp4, fp4_sparse) are
    unreachable-but-harmless and intentionally not flagged."""
    from mlsysim.core.units import PRECISION_MAP
    safe_fallthrough = {"fp16", "bf16"}  # correctly fall through to FP16 peak
    offenders = []
    for node in _all_hardware_nodes():
        pf = getattr(node.compute, "precision_flops", None) or {}
        keys = set(pf)
        for p in PRECISION_MAP:
            if p in safe_fallthrough:
                continue
            aliases = sorted(k for k in keys if k != p and k.startswith(p))
            if aliases and p not in keys:
                offenders.append(f"{node.name}: stores {aliases} but no reachable '{p}' key")
    assert not offenders, (
        "Off-vocabulary precision aliases shadow a canonical PRECISION_MAP key, "
        "so Engine.solve silently falls through to the FP16 peak:\n  "
        + "\n  ".join(offenders)
    )


def test_h100_h200_fp32_resolves_to_cuda_core_rate_not_fp16_peak():
    """Regression for F1: precision='fp32' on H100/H200 must reach the stored
    non-tensor FP32 rate (67 TFLOP/s), not fall through to peak_flops (989)."""
    from mlsysim.core.units import resolve_precision
    key, _ = resolve_precision("fp32")
    for dev in ("H100", "H200"):
        node = getattr(Hardware.Cloud, dev)
        pf = node.compute.precision_flops
        assert key in pf, f"{dev}: '{key}' is not a precision_flops key"
        assert pf[key].m_as("TFLOP/s") == pytest.approx(67.0), dev
        assert pf[key].m_as("TFLOP/s") != node.compute.peak_flops.m_as("TFLOP/s"), dev
