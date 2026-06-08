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
    assert result.bottleneck == "Memory"
    assert result.latency.m_as("ms") == pytest.approx(0.5426213830308975)
    assert result.throughput.m_as("1/s") == pytest.approx(1842.9056267822364)
    assert result.memory_footprint.m_as("GB") == pytest.approx(0.0512)
    assert result.arithmetic_intensity.m_as("flop/byte") == pytest.approx(72.79829545454545)


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
    assert result.step_latency_total.m_as("ms") == pytest.approx(4282.2171449090065)
    assert result.communication_latency.m_as("ms") == pytest.approx(271.3768817511111)
    assert result.dp_communication_latency.m_as("ms") == pytest.approx(3.9104722222222223)
    assert result.tp_communication_latency.m_as("ms") == pytest.approx(267.4664095288889)
    assert result.pipeline_bubble_latency.m_as("ms") == pytest.approx(546.9327631578948)
    assert result.effective_throughput.m_as("1/s") == pytest.approx(7652.110785403969)
    assert result.scaling_efficiency == pytest.approx(0.8089051495480867)
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
    assert economics.capex_usd == pytest.approx(7013.698630136986)
    assert economics.opex_energy_usd == pytest.approx(355.48692480000005)
    assert economics.opex_maintenance_usd == pytest.approx(1052.054794520548)
    assert economics.total_opex_usd == pytest.approx(1407.541719320548)
    assert economics.tco_usd == pytest.approx(8421.240349457534)
    assert economics.carbon_footprint_kg == pytest.approx(1270.8657561600003)
