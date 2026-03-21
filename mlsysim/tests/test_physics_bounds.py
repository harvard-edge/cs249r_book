"""
Automated Physics Verification Suite
------------------------------------
This test suite "bulletproofs" the mlsysim Silicon Zoo.
It iterates over every registered hardware node and ensures that its
specifications obey known laws of physics and sensible bounds. 
This prevents contributors from accidentally adding a chip with 
"80 TB/s" of bandwidth when they meant "80 GB/s".
"""
import pytest
from mlsysim.hardware.registry import Hardware
from mlsysim.core.constants import ureg

def get_all_hardware():
    """Extracts all instantiated HardwareNode objects from the registry."""
    nodes = []
    # Collect from all sub-registries
    for registry in [Hardware.Cloud, Hardware.Workstation, Hardware.Mobile, Hardware.Edge, Hardware.Tiny]:
        for attr_name in dir(registry):
            if not attr_name.startswith('_'):
                attr = getattr(registry, attr_name)
                # Check if it's a HardwareNode
                if hasattr(attr, 'compute') and hasattr(attr, 'memory'):
                    nodes.append(attr)
    return nodes

@pytest.mark.parametrize("node", get_all_hardware(), ids=lambda n: n.name)
def test_physics_arithmetic_intensity(node):
    """
    Ridge point (Arithmetic Intensity) must be positive and within
    historical/physical bounds (typically between 1 and 2000 FLOP/byte).
    """
    ridge = node.ridge_point()
    
    # Must be strictly positive
    assert ridge.magnitude > 0, f"{node.name} has zero or negative ridge point: {ridge}"
    
    # Tiny edge devices might have very low ridge points (e.g. 0.05), but cloud GPUs
    # are usually 100-500. We set a safe global upper bound of 5000 FLOP/byte.
    # Anything higher implies a typo in FLOPS (too high) or Bandwidth (too low).
    assert ridge.m_as("flop/byte") < 5000, f"{node.name} has physically improbable ridge point: {ridge}"

@pytest.mark.parametrize("node", get_all_hardware(), ids=lambda n: n.name)
def test_physics_power_density(node):
    """
    TDP must be within safe operating limits.
    A single chip rarely exceeds 1500W. 
    A rack-scale system (like NVL72) might reach 150,000W.
    """
    if node.tdp is not None:
        tdp_w = node.tdp.m_as("watt")
        assert tdp_w > 0, f"{node.name} has zero or negative TDP: {tdp_w}W"
        assert tdp_w <= 150_000, f"{node.name} exceeds max rack-scale power density: {tdp_w}W"

@pytest.mark.parametrize("node", get_all_hardware(), ids=lambda n: n.name)
def test_physics_memory_bandwidth(node):
    """
    Memory bandwidth must be physically achievable.
    Sub-GB/s is possible for TinyML.
    Wafer-scale (Cerebras) can hit ~25 PB/s.
    """
    bw_gbs = node.memory.bandwidth.m_as("GB/s")
    assert bw_gbs > 0, f"{node.name} has zero memory bandwidth."
    
    # Check for accidental TB/s vs GB/s typos
    if "Cloud" in node.__class__.__module__ or "Cloud" in str(node):
        # A cloud GPU should generally have > 100 GB/s bandwidth
        assert bw_gbs > 100 or node.name == "Google TPU v1", f"{node.name} has suspiciously low bandwidth for cloud: {bw_gbs} GB/s"

@pytest.mark.parametrize("node", get_all_hardware(), ids=lambda n: n.name)
def test_physics_peak_flops(node):
    """
    Peak FLOPS must be positive.
    """
    flops_tflops = node.compute.peak_flops.m_as("TFLOPs/s")
    assert flops_tflops > 0, f"{node.name} has zero peak FLOPS."
