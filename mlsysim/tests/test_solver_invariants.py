import pytest

import mlsysim
from mlsysim.core.units import Q_
from mlsysim.engine.solvers import (
    DistributedModel,
    EconomicsModel,
    ServingModel,
    SustainabilityModel,
    TrainingMemoryModel,
)


def test_precision_memory_is_monotonic_for_model_weights():
    model = mlsysim.Models.Language.Llama3_8B

    fp32 = model.size_in_bytes(Q_("4 byte")).m_as("GB")
    fp16 = model.size_in_bytes(Q_("2 byte")).m_as("GB")
    fp8 = model.size_in_bytes(Q_("1 byte")).m_as("GB")
    int4 = model.size_in_bytes(Q_("0.5 byte")).m_as("GB")

    assert fp32 > fp16 > fp8 > int4
    assert fp16 == pytest.approx(2 * fp8)
    assert fp8 == pytest.approx(2 * int4)


def test_training_memory_components_sum_to_total():
    result = TrainingMemoryModel().solve(
        mlsysim.Models.Language.Llama3_8B,
        mlsysim.Hardware.Cloud.H100,
        batch_size=16,
        seq_len=1024,
        precision="fp16",
        dp_size=4,
        zero_stage=2,
    )

    component_sum = (
        result.weights
        + result.gradients
        + result.optimizer_state
        + result.activations
        + result.communication_buffers
    ).to("GB")
    assert result.total_memory.m_as("GB") == pytest.approx(component_sum.m_as("GB"))
    assert result.memory_utilization == pytest.approx(
        (result.total_memory / result.available_memory).to_base_units().magnitude
    )


def test_zero_stage_does_not_increase_model_state_memory():
    model = mlsysim.Models.Language.Llama3_8B
    hardware = mlsysim.Hardware.Cloud.H100
    solver = TrainingMemoryModel()

    totals = [
        solver.solve(
            model,
            hardware,
            batch_size=16,
            seq_len=1024,
            precision="fp16",
            dp_size=8,
            zero_stage=stage,
        ).total_memory.m_as("GB")
        for stage in range(4)
    ]

    assert totals == sorted(totals, reverse=True)


def test_serving_memory_grows_with_sequence_length():
    model = mlsysim.Models.Language.Llama3_8B
    hardware = mlsysim.Hardware.Cloud.H100
    solver = ServingModel()

    short = solver.solve(model, hardware, seq_len=512, batch_size=1)
    long = solver.solve(model, hardware, seq_len=4096, batch_size=1)

    assert long.kv_cache_size > short.kv_cache_size
    assert long.total_memory_required > short.total_memory_required
    assert long.itl > short.itl


def test_distributed_latency_components_are_non_negative():
    result = DistributedModel().solve(
        mlsysim.Models.Language.Llama3_8B,
        mlsysim.Systems.Clusters.Research_256,
        batch_size=1024,
        seq_len=2048,
        tp_size=8,
        pp_size=4,
        microbatch_count=16,
    )

    quantities = [
        result.dp_communication_latency,
        result.tp_communication_latency,
        result.ep_communication_latency,
        result.communication_latency,
        result.pipeline_bubble_latency,
        result.step_latency_total,
        result.effective_throughput,
    ]
    assert all(q.magnitude >= 0 for q in quantities)
    assert 0.0 <= result.scaling_efficiency <= 1.0
    assert 0.0 <= result.bubble_fraction <= 1.0


def test_sustainability_energy_and_economics_identities_hold():
    fleet = mlsysim.Systems.Clusters.Research_256

    sustainability = SustainabilityModel().solve(fleet, duration_days=2, mfu=0.4)
    assert sustainability.total_energy_kwh >= sustainability.it_energy_kwh
    assert sustainability.carbon_footprint_kg >= 0.0
    assert sustainability.water_usage_liters >= 0.0

    economics = EconomicsModel().solve(
        fleet,
        duration_days=2,
        mfu=0.4,
        infrastructure_multiplier=1.0,
    )
    assert economics.total_opex_usd == pytest.approx(
        economics.opex_energy_usd + economics.opex_maintenance_usd
    )
    assert economics.tco_usd == pytest.approx(economics.capex_usd + economics.total_opex_usd)


def test_si_and_binary_capacity_units_convert_consistently():
    h100_capacity = mlsysim.Hardware.Cloud.H100.memory.capacity

    assert h100_capacity.m_as("GiB") == pytest.approx(80.0)
    assert h100_capacity.m_as("GB") == pytest.approx(85.89934592)
