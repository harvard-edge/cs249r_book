# tests/test_empirical.py
# Empirical Validation Suite for mlsysim
# Validates first-principles analytical results against real-world benchmarks.
#
# Philosophy: These tests compare analytical model OUTPUT against EXTERNAL
# benchmark data (MLPerf, NVIDIA published numbers). The targets are NOT
# derived from the model itself — they come from measured hardware performance.
# Wide tolerances (30-50%) are expected because the analytical model deliberately
# omits framework overhead, kernel scheduling, and data pipeline effects.

import pytest
from mlsysim.core.constants import Q_, ureg
from mlsysim.core.solver import SingleNodeModel, DistributedModel, ServingModel
from mlsysim.hardware.registry import Hardware
from mlsysim.models.registry import Models
from mlsysim.systems.types import Fleet, NetworkFabric, Node

# ─── 1. RESNET-50 TRAINING (SINGLE NODE) ───────────────────────────────────
@pytest.mark.empirical
def test_resnet50_h100_throughput():
    """
    Validate ResNet-50 throughput on H100 against MLPerf v3.1.

    External target: ~4,800 samples/s (MLPerf Training v3.1, NVIDIA H100 SXM,
    single GPU, closed division). Source: mlcommons.org/en/training-normal-31/

    The analytical model over-predicts because it does not account for
    data pipeline overhead (Wall 9) or framework kernel launch tax (Wall 3).
    We accept a 50% tolerance to validate the model is in the right ballpark.
    """
    model = Models.ResNet50
    hardware = Hardware.H100

    # efficiency=0.08 calibrated to the "accelerator overkill" regime where
    # ResNet-50 kernels are too small to saturate H100 tensor cores.
    profile = SingleNodeModel().solve(model, hardware, batch_size=256, efficiency=0.08, is_training=True)

    throughput = profile.throughput.m_as("1/s")
    mlperf_target = 4800.0  # MLPerf v3.1 single-GPU H100

    # Analytical model yields higher than MLPerf due to missing overhead.
    # Accept 50% tolerance — we're validating order of magnitude, not exact match.
    assert throughput == pytest.approx(mlperf_target, rel=0.50), \
        f"ResNet-50 H100 throughput {throughput:.0f} vs MLPerf target {mlperf_target:.0f}"

# ─── 2. LLAMA-3-8B INFERENCE (SINGLE NODE) ──────────────────────────────────
@pytest.mark.empirical
def test_llama3_8b_h100_itl():
    """
    Validate Llama-3-8B Inter-Token Latency (ITL) on H100.

    External target: ~5-10ms at batch 1 (NVIDIA TensorRT-LLM benchmarks).
    Decode is memory-bandwidth-bound: ITL ≈ model_weights / HBM_bandwidth.
    The analytical model gives a lower bound (no framework overhead).
    """
    model = Models.Llama3_8B
    hardware = Hardware.H100

    res = ServingModel().solve(model, hardware, seq_len=1024, batch_size=1, efficiency=0.60)

    itl_ms = res.itl.m_as("ms")
    # Analytical lower bound should be 3-8ms; real-world is 5-12ms
    assert 1.0 < itl_ms < 20.0, f"Llama-3-8B ITL {itl_ms:.1f}ms outside plausible range [1, 20]ms"

# ─── 3. DISTRIBUTED EFFICIENCY (8x H100 CLUSTER) ───────────────────────────
@pytest.mark.empirical
def test_distributed_resnet_efficiency():
    """
    Validate Scaling Efficiency for 8-GPU H100 within a single NVLink node.

    External target: >90% scaling efficiency for small DP over NVLink.
    Source: NVIDIA DGX H100 data sheet claims near-linear scaling within node.
    """
    model = Models.ResNet50
    h100 = Hardware.H100

    node = Node(
        name="H100-Node",
        accelerator=h100,
        accelerators_per_node=8,
        intra_node_bw=Q_("900 GB/s")
    )
    fabric = NetworkFabric(name="NVLink 4.0", bandwidth=Q_("900 GB/s"))
    fleet = Fleet(name="H100-Fleet", node=node, count=1, fabric=fabric)

    res = DistributedModel().solve(model, fleet, batch_size=2048, efficiency=0.45)

    assert res.scaling_efficiency > 0.90, \
        f"Scaling efficiency {res.scaling_efficiency:.2f} too low for 8-GPU NVLink"

# ─── 4. DIMENSIONAL INTEGRITY ───────────────────────────────────────────────
def test_dimensional_integrity():
    """Verify that results preserve Pint units and can be converted."""
    model = Models.ResNet50
    hardware = Hardware.H100
    profile = SingleNodeModel().solve(model, hardware)

    assert profile.latency.check('[time]')
    assert profile.throughput.check('1/[time]')
