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
from mlsysim.core.units import Q_, ureg
from mlsysim.engine.empirical import EMPIRICAL_ANCHORS, anchor_by_id
from mlsysim.engine.solvers import SingleNodeModel, DistributedModel, ServingModel
from mlsysim.hardware.registry import Hardware
from mlsysim.models.registry import Models
from mlsysim.physics import calc_transformer_training_flops
from mlsysim.systems.types import Fleet, NetworkFabric, Node

# ─── 1. RESNET-50 TRAINING (SINGLE NODE) ───────────────────────────────────
@pytest.mark.empirical
def test_resnet50_h100_throughput():
    """
    Validate ResNet-50 throughput on H100 against the benchmark registry.

    External target: Literature.Benchmarks.ResNet50H100TrainThroughput.

    The analytical model over-predicts because it does not account for
    data pipeline overhead (Wall 9) or framework kernel launch tax (Wall 3).
    We accept a 50% tolerance to validate the model is in the right ballpark.
    """
    h100_anchor = anchor_by_id("resnet50_h100_train_bs256")
    model = h100_anchor.workload
    hardware = h100_anchor.hardware

    # efficiency=0.08 calibrated to the "accelerator overkill" regime where
    # ResNet-50 kernels are too small to saturate H100 tensor cores.
    profile = SingleNodeModel().solve(model, hardware, batch_size=256, efficiency=0.08, is_training=True)

    throughput = profile.throughput.m_as("1/s")
    mlperf_target = float(h100_anchor.target)

    # Analytical model yields higher than MLPerf due to missing overhead.
    # Accept 50% tolerance — we're validating order of magnitude, not exact match.
    assert throughput == pytest.approx(mlperf_target, rel=0.50), \
        f"ResNet-50 H100 throughput {throughput:.0f} vs MLPerf target {mlperf_target:.0f}"

# ─── 2. LLAMA-3-8B INFERENCE (SINGLE NODE) ──────────────────────────────────
@pytest.mark.empirical
def test_llama3_8b_h100_itl():
    """
    Validate Llama-3-8B Inter-Token Latency (ITL) on H100.

    External target: Literature.Benchmarks.Llama3_8B_H100_ITLLower/Upper.
    Decode is memory-bandwidth-bound: ITL ≈ model_weights / HBM_bandwidth.
    The analytical model gives a lower bound (no framework overhead).
    """
    llama_anchor = anchor_by_id("llama3_8b_h100_bs1_itl")
    model = llama_anchor.workload
    hardware = llama_anchor.hardware

    res = ServingModel().solve(model, hardware, seq_len=1024, batch_size=1, efficiency=0.60)

    itl_ms = res.itl.m_as("ms")
    lower = float(llama_anchor.lower) / 3
    upper = float(llama_anchor.upper) * 2
    assert lower < itl_ms < upper, f"Llama-3-8B ITL {itl_ms:.1f}ms outside plausible range"

# ─── 3. DISTRIBUTED EFFICIENCY (8x H100 CLUSTER) ───────────────────────────
@pytest.mark.empirical
def test_distributed_resnet_efficiency():
    """
    Validate Scaling Efficiency for 8-GPU H100 within a single NVLink node.

    External target: >90% scaling efficiency for small DP over NVLink.
    Source: NVIDIA DGX H100 data sheet claims near-linear scaling within node.
    """
    model = Models.Vision.ResNet50
    h100 = Hardware.Cloud.H100

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
    model = Models.Vision.ResNet50
    hardware = Hardware.Cloud.H100
    profile = SingleNodeModel().solve(model, hardware)

    assert profile.latency.check('[time]')
    assert profile.throughput.check('1/[time]')


@pytest.mark.empirical
def test_documented_calibration_table_stays_in_range():
    """Check the sourced, domain-reviewed calibration anchors.

    These are broad sanity bands, not exact benchmarks. They catch registry or
    formula drift that would move MLSysIM outside the published calibration
    envelope for common teaching examples.
    """
    a100_anchor = anchor_by_id("resnet50_a100_train_bs256")
    resnet_a100 = SingleNodeModel().solve(
        a100_anchor.workload,
        a100_anchor.hardware,
        **a100_anchor.solver_kwargs,
    )
    assert a100_anchor.accepts(resnet_a100.throughput.m_as(a100_anchor.units))

    h100_anchor = anchor_by_id("resnet50_h100_train_bs256")
    resnet_h100 = SingleNodeModel().solve(
        h100_anchor.workload,
        h100_anchor.hardware,
        **h100_anchor.solver_kwargs,
    )
    assert h100_anchor.accepts(resnet_h100.throughput.m_as(h100_anchor.units))

    llama_anchor = anchor_by_id("llama3_8b_h100_bs1_itl")
    llama_itl = ServingModel().solve(
        llama_anchor.workload,
        llama_anchor.hardware,
        **llama_anchor.solver_kwargs,
    ).itl.m_as(llama_anchor.units)
    assert llama_anchor.accepts(llama_itl)

    gpt3_anchor = anchor_by_id("gpt3_175b_training_flops")
    gpt3 = gpt3_anchor.workload
    gpt3_flops = calc_transformer_training_flops(gpt3.parameters, gpt3.training_tokens)
    assert gpt3_anchor.accepts(gpt3_flops.m_as(gpt3_anchor.units))


def test_empirical_anchors_have_provenance_and_review_notes():
    for anchor in EMPIRICAL_ANCHORS:
        assert anchor.registry_paths
        assert anchor.provenance_ids
        assert anchor.review_notes
        values = [anchor.target, anchor.lower, anchor.upper]
        for value in values:
            if value is not None:
                assert value.provenance.ref
                assert value.provenance.verified
