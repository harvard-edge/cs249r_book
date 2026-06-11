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
from mlsysim.core.units import ureg
from mlsysim.models.registry import Models
from mlsysim.models.types import TransformerWorkload

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


# --- Transformer model spec sanity bounds -----------------------------------
# 2026-06-10 (audit campaign, task #15): the hardware tests above catch a chip
# typed as "80 TB/s" when "80 GB/s" was meant. Registered model specs need the
# same guard. A transposed parameter count (124M typed as 421M) or a decimal
# slip (124 typed as 12.4) in a registry YAML is same-order-of-magnitude, so a
# loose bound misses it. Two layers protect the registry:
#   1. a universal attention-geometry invariant over EVERY transformer, and
#   2. a curated web-verified spec table for the models the book audit pinned.
# Extend EXPECTED_TRANSFORMER_SPECS whenever a new model spec is registered
# (the value you web-verified it against is exactly what belongs here).

def get_all_transformer_models():
    """Every registered TransformerWorkload in Models.Language."""
    models = []
    lang = Models.Language
    for attr_name in dir(lang):
        if attr_name.startswith("_"):
            continue
        attr = getattr(lang, attr_name)
        if isinstance(attr, TransformerWorkload):
            models.append(attr)
    return models

# Web-verified canonical specs for models the audit added/pinned. A registry
# value that diverges from these by more than the tolerance fails CI. Keyed by
# the model's `name`. Add an entry (params in M, flops in M, integer geometry)
# when you register a new model, citing the source you verified against.
EXPECTED_TRANSFORMER_SPECS = {
    # GPT-2 Small — Radford et al. (2019), "Language Models are Unsupervised
    # Multitask Learners". 124M params, 12 layers, 768 hidden, 12 heads; the
    # per-token forward pass is ~2N = 248 MFLOP.
    "GPT-2 Small (124M)": dict(
        parameters_m=124.0, inference_flops_m=248.0,
        layers=12, hidden_dim=768, heads=12,
    ),
}

@pytest.mark.parametrize(
    "model", get_all_transformer_models(), ids=lambda m: m.name
)
def test_transformer_attention_geometry(model):
    """
    Universal attention invariant: hidden_dim must split evenly into heads, the
    resulting head dimension must sit in the historically sane 32-256 range, and
    hidden_dim must be a multiple of 64. A transposed hidden_dim/heads (768/12
    typed as 12/768) or a malformed dimension breaks this immediately.
    """
    if model.hidden_dim is None or model.heads is None:
        pytest.skip(f"{model.name} has no attention geometry")
    h, nh = model.hidden_dim, model.heads
    assert h > 0 and nh > 0, f"{model.name}: non-positive hidden_dim/heads ({h}/{nh})"
    assert h % nh == 0, f"{model.name}: hidden_dim {h} not divisible by heads {nh}"
    head_dim = h // nh
    assert 32 <= head_dim <= 256, (
        f"{model.name}: head_dim {head_dim} outside the sane 32-256 range "
        f"(hidden_dim {h} / heads {nh})"
    )
    assert h % 64 == 0, f"{model.name}: hidden_dim {h} is not a multiple of 64"
    assert 1 <= model.layers <= 200, (
        f"{model.name}: layers {model.layers} outside the sane 1-200 range"
    )

@pytest.mark.parametrize(
    "model", get_all_transformer_models(), ids=lambda m: m.name
)
def test_transformer_curated_specs(model):
    """
    Curated guard: a model the book audit pinned must match its web-verified
    spec within 1%. Catches a transposed or decimal-slipped registry value
    (124M typed as 421M, or 124 typed as 12.4) that the geometry invariant alone
    would miss because the wrong value is the same order of magnitude.
    """
    expected = EXPECTED_TRANSFORMER_SPECS.get(model.name)
    if expected is None:
        pytest.skip(f"{model.name} not in the curated spec table")
    params_m = model.parameters.m_as("param") / 1e6
    assert abs(params_m - expected["parameters_m"]) / expected["parameters_m"] <= 0.01, (
        f"{model.name}: parameters {params_m:.1f}M diverges from web-verified "
        f"{expected['parameters_m']}M (registry typo?)"
    )
    if model.inference_flops is not None and "inference_flops_m" in expected:
        flops_m = model.inference_flops.m_as("flop") / 1e6
        assert abs(flops_m - expected["inference_flops_m"]) / expected["inference_flops_m"] <= 0.01, (
            f"{model.name}: inference_flops {flops_m:.1f}M diverges from "
            f"web-verified {expected['inference_flops_m']}M (registry typo?)"
        )
    for field in ("layers", "hidden_dim", "heads"):
        if field in expected:
            actual = getattr(model, field)
            assert actual == expected[field], (
                f"{model.name}: {field} {actual} diverges from web-verified "
                f"{expected[field]} (registry typo?)"
            )
