import pytest

import mlsysim
from mlsysim.core.units import Q_
from mlsysim.engine.solvers import (
    DistributedModel,
    EconomicsModel,
    ServingModel,
    SingleNodeModel,
    SustainabilityModel,
    TrainingMemoryModel,
)
from mlsysim.physics import calc_bottleneck


def test_golden_roofline_resnet50_a100_batch1():
    """Golden single-node example for canonical roofline calculations."""
    result = SingleNodeModel().solve(
        mlsysim.Models.Vision.ResNet50,
        mlsysim.Hardware.Cloud.A100,
        batch_size=1,
        precision="fp16",
        efficiency=0.5,
    )

    assert result.feasible is True
    assert result.bottleneck == "Compute"
    assert result.latency.m_as("ms") == pytest.approx(0.5675641025641026)
    assert result.throughput.m_as("1/s") == pytest.approx(1761.915518409758)
    assert result.memory_footprint.m_as("GB") == pytest.approx(0.0512)
    assert result.arithmetic_intensity.m_as("flop/byte") == pytest.approx(145.5965909090909)


def test_golden_bottleneck_intensity_scaled_units():
    result = calc_bottleneck(
        Q_("1 TFLOP"),
        Q_("1 GB"),
        Q_("1 TFLOP/s"),
        Q_("1 GB/s"),
    )

    assert result["intensity"] == pytest.approx(1000.0)


def test_golden_training_memory_llama3_8b_h100():
    """Golden training-memory attribution for a common LLM teaching example."""
    result = TrainingMemoryModel().solve(
        mlsysim.Models.Language.Llama3_8B,
        mlsysim.Hardware.Cloud.H100,
        batch_size=8,
        seq_len=1024,
        precision="fp16",
        activation_checkpointing="selective",
    )

    assert result.feasible is False
    assert result.weights.m_as("GB") == pytest.approx(16.060000000000002)
    assert result.gradients.m_as("GB") == pytest.approx(16.060000000000002)
    assert result.optimizer_state.m_as("GB") == pytest.approx(96.36)
    # 2026-06-06 audit: re-pinned to the Korthikanti-exact selective bound
    # (34*s*b*h FP16 bytes per layer). The previous 21.47 GB came from a
    # 10-coefficient model with doubled FP16 bytes that matched no source.
    assert result.activations.m_as("GB") == pytest.approx(36.507222016)
    assert result.communication_buffers.m_as("GB") == pytest.approx(0.803)
    assert result.total_memory.m_as("GB") == pytest.approx(165.790222016)
    assert result.memory_utilization == pytest.approx(1.9300522051751614)


def test_golden_serving_llama3_8b_h100():
    """Golden prefill/decode split for Llama serving."""
    result = ServingModel().solve(
        mlsysim.Models.Language.Llama3_8B,
        mlsysim.Hardware.Cloud.H100,
        seq_len=2048,
        batch_size=1,
        precision="fp16",
        efficiency=0.5,
    )

    assert result.feasible is True
    assert result.ttft.m_as("ms") == pytest.approx(70.97037058756726)
    assert result.itl.m_as("ms") == pytest.approx(5.194159837611941)
    assert result.kv_cache_size.m_as("GB") == pytest.approx(0.268435456)
    assert result.model_weights_size.m_as("GB") == pytest.approx(16.060000000000002)
    assert result.total_memory_required.m_as("GB") == pytest.approx(16.328435456)
    assert result.memory_utilization == pytest.approx(0.1900880068540573)


def test_golden_distributed_llama3_8b_research_cluster():
    """Golden distributed-training decomposition for 3D parallelism."""
    result = DistributedModel().solve(
        mlsysim.Models.Language.Llama3_8B,
        mlsysim.Systems.Clusters.Research_256,
        batch_size=1024,
        seq_len=2048,
        precision="fp16",
        efficiency=0.45,
        tp_size=8,
        pp_size=4,
        microbatch_count=16,
        zero_stage=1,
    )

    assert result.parallelism == {"dp": 8, "tp": 8, "pp": 4, "ep": 1}
    # 2026-06-10 audit: re-pinned after the interconnect direction-convention
    # fix. Nodes now feed NVLink's PER-DIRECTION rate (450 GB/s on H100) into
    # the collective beta terms instead of the 900 GB/s bidirectional total,
    # so intra-node-bound communication latencies roughly doubled (they were
    # ~2x optimistic before). findings_provenance.md M1.
    assert result.step_latency_total.m_as("ms") == pytest.approx(4553.363026660118)
    assert result.communication_latency.m_as("ms") == pytest.approx(542.5227635022222)
    assert result.dp_communication_latency.m_as("ms") == pytest.approx(7.813944444444444)
    assert result.tp_communication_latency.m_as("ms") == pytest.approx(534.7088190577778)
    assert result.pipeline_bubble_latency.m_as("ms") == pytest.approx(546.9327631578948)
    assert result.effective_throughput.m_as("1/s") == pytest.approx(7196.439161152336)
    assert result.scaling_efficiency == pytest.approx(0.760736071277139)
    assert result.bubble_fraction == pytest.approx(0.15789473684210525)


def test_golden_sustainability_and_economics_research_cluster():
    fleet = mlsysim.Systems.Clusters.Research_256

    sustainability = SustainabilityModel().solve(fleet, duration_days=1, mfu=0.45)
    assert sustainability.it_energy_kwh.magnitude == pytest.approx(2644.992)
    assert sustainability.total_energy_kwh.magnitude == pytest.approx(2962.3910400000004)
    assert sustainability.carbon_footprint_kg == pytest.approx(1270.8657561600003)
    assert sustainability.water_usage_liters == pytest.approx(5332.303872000001)
    assert sustainability.pue == pytest.approx(1.12)
    assert sustainability.region_name == "US Average"

    economics = EconomicsModel().solve(
        fleet,
        duration_days=1,
        mfu=0.45,
        infrastructure_multiplier=1.0,
    )
    assert economics.capex_usd == pytest.approx(5844.748858447489)
    assert economics.opex_energy_usd == pytest.approx(355.48692480000005)
    assert economics.opex_maintenance_usd == pytest.approx(876.7123287671233)
    assert economics.total_opex_usd == pytest.approx(1232.1992535671234)
    assert economics.tco_usd == pytest.approx(7076.948112014612)
    assert economics.carbon_footprint_kg == pytest.approx(1270.8657561600003)


def test_golden_engine_utilization_and_energy_resnet50_a100():
    """2026-06-10 audit (B6): absolute pins for mfu/hfu/energy — previously
    only bounds/orderings were asserted, so a recompute-inflated MFU, a
    phantom HFU ratio, or a double-counted energy term passed the suite."""
    from mlsysim import Models, Hardware
    from mlsysim.engine.engine import Engine

    p = Engine.solve(Models.Vision.ResNet50, Hardware.Cloud.A100, batch_size=1)
    assert p.mfu == pytest.approx(0.046306754009487236, rel=1e-9)
    assert p.hfu == pytest.approx(p.mfu, rel=1e-12)  # no recompute -> identical
    assert p.energy.m_as("J") == pytest.approx(0.07546666666666667, rel=1e-9)
    assert p.overhead_dominated is True

    # Batch-traffic heuristic (weights x (1 + 0.1 x B)) pinned at batch 32:
    # intensity was previously pinned only implicitly at batch=1.
    p32 = Engine.solve(Models.Vision.ResNet50, Hardware.Cloud.A100, batch_size=32)
    assert p32.arithmetic_intensity.magnitude == pytest.approx(1220.2380952380952, rel=1e-9)


def test_golden_engine_non_transformer_training_memory():
    """2026-06-10 audit (B3): non-Transformer training fallback = weights +
    gradients + Adam state (12 B/param), not the old 3x-weights heuristic
    that understated mixed-precision Adam ~2.7x. ResNet-50 fp16:
    25.6e6 params x (2 + 2 + 12) B = 0.4096 GB."""
    from mlsysim import Models, Hardware
    from mlsysim.engine.engine import Engine

    p = Engine.solve(Models.Vision.ResNet50, Hardware.Cloud.A100,
                     batch_size=8, is_training=True)
    assert p.memory_footprint.m_as("GB") == pytest.approx(0.4096, rel=1e-6)


def test_reference_hardware_tflops_matches_registry():
    """2026-06-10 audit (discipline): cal.REFERENCE_HARDWARE_TFLOPS is a
    convenience scalar duplicating the H100 SXM FP16 dense peak. Cross-pin it
    to the registry so the two sources cannot drift apart silently."""
    from mlsysim import Hardware
    from mlsysim.engine import calibration as cal

    # H100's default peak_flops IS the FP16 dense figure (989 TFLOP/s);
    # fp16 has no explicit precision_flops entry on this device.
    h100_fp16_tflops = Hardware.Cloud.H100.compute.peak_flops.m_as("TFLOP/s")
    assert cal.REFERENCE_HARDWARE_TFLOPS == pytest.approx(h100_fp16_tflops, rel=1e-9)
