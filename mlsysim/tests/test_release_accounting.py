"""Boundary and cross-API invariants for first-order accounting."""

import math

import pytest

from mlsysim import Engine, Hardware, Models, Q_
from mlsysim.core._validation import (
    validate_at_least, validate_nonnegative, validate_positive, validate_range,
)
from mlsysim.solvers import ContinuousBatchingModel, ServingModel, TrainingMemoryModel
from mlsysim.physics.memory import optimizer_state_bytes


def with_capacity(hardware, capacity):
    return hardware.model_copy(update={
        "memory": hardware.memory.model_copy(update={"capacity": capacity}),
    })


def test_training_work_scales_with_sequence_length():
    model, hw = Models.Language.GPT2_Small, Hardware.Cloud.H100
    short = Engine.solve(model, hw, is_training=True, seq_len=128)
    long = Engine.solve(model, hw, is_training=True, seq_len=2048)
    assert short.feasible and long.feasible
    assert long.latency_compute / short.latency_compute == pytest.approx(2048 / 128)
    assert long.memory_footprint > short.memory_footprint
    # MFU's numerator must describe the same work as latency, excluding recompute.
    peak = hw.compute.peak_flops
    required_ops = model.lower().total_ops * 3 * 2048
    assert long.mfu == pytest.approx((required_ops / long.latency / peak).to("").magnitude)


def test_sequence_length_does_not_reinterpret_cnn_or_decode():
    for model, training in [(Models.Vision.ResNet50, True), (Models.Language.GPT2_Small, False)]:
        short = Engine.solve(model, Hardware.Cloud.H100, seq_len=128, is_training=training)
        long = Engine.solve(model, Hardware.Cloud.H100, seq_len=2048, is_training=training)
        assert short.latency_compute == long.latency_compute


@pytest.mark.parametrize("optimizer", ["adam", "adamw", "ADAMW", "sgd", "none"])
def test_optimizer_accounting_agrees_across_public_apis(optimizer):
    model, hw = Models.Language.Llama3_8B, Hardware.Cloud.H100
    helper = model.training_memory(1, 128, optimizer=optimizer, strategy="selective")
    solver = TrainingMemoryModel().solve(
        model, hw, 1, seq_len=128, optimizer=optimizer, communication_buffer_fraction=0,
    )
    assert helper.m_as("GB") == pytest.approx(solver.total_memory.m_as("GB"))


def test_unknown_optimizer_is_rejected_by_both_apis():
    model, hw = Models.Language.GPT2_Small, Hardware.Cloud.H100
    with pytest.raises(ValueError, match="Unknown optimizer"):
        model.training_memory(1, 128, optimizer="adma")
    with pytest.raises(ValueError, match="Unknown optimizer"):
        TrainingMemoryModel().solve(model, hw, 1, optimizer="adma")


@pytest.mark.parametrize("optimizer,fp32_copies", [
    ("adam", 3), ("adamw", 3), ("sgd", 1), ("none", 0),
])
def test_optimizer_bytes_follow_declared_storage_contract(optimizer, fp32_copies):
    # Count storage independently of either training API: a master weight plus
    # two moments for Adam; master weight only for momentum-free SGD.
    expected_bytes = fp32_copies * 32 / 8
    assert optimizer_state_bytes(optimizer).m_as("byte") == expected_bytes


def test_decode_weight_traffic_sets_a_memory_floor():
    model, hw = Models.Language.Llama3_8B, Hardware.Cloud.H100
    # Dense FP16 weights alone require P * 16 bits to cross the memory bus.
    # This bound deliberately omits KV reads and layer overhead.
    weight_bytes = model.parameters.m_as("count") * 16 / 8 * Q_(1, "byte")
    minimum = weight_bytes / hw.memory.bandwidth
    result = ServingModel().solve(model, hw, 1, batch_size=1, efficiency=1)
    assert result.feasible
    assert result.itl >= minimum


def test_speculation_requires_both_models_and_both_caches():
    target, draft = Models.Language.Llama3_8B, Models.Language.GPT2_Small
    seq = 2048
    weights = target.size_in_bytes() + draft.size_in_bytes()
    target_kv = target.get_kv_cache_size(seq, 1)
    draft_kv = draft.get_kv_cache_size(seq, 1)
    capacity = weights + target_kv + draft_kv / 2
    hw = with_capacity(Hardware.Cloud.H100, capacity)
    result = ServingModel().solve(target, hw, seq, draft_model=draft)
    assert result.feasible is False
    assert result.total_memory_required > capacity
    assert result.total_memory_required.m_as("byte") == pytest.approx(
        (weights + target_kv + draft_kv).m_as("byte")
    )


def test_disaggregated_prefill_must_fit_its_own_device():
    target = Models.Language.Llama3_8B
    prefill = with_capacity(Hardware.Cloud.H100, Q_("1 GB"))
    result = ServingModel().solve(target, prefill, 128, decode_hardware=Hardware.Cloud.H100)
    assert result.feasible is False
    assert any("Prefill Memory Wall: FAILED" in line for line in result.constraint_trace)


def test_large_batch_decode_cannot_exceed_compute_capacity():
    model, hw, batch = Models.Language.Llama3_8B, Hardware.Cloud.H100, 512
    # Independent minimum: even linear layers alone must execute two FLOPs per
    # parameter per token. Attention and framework work cannot reduce this time.
    minimum = 2 * model.parameters.m_as("count") * batch * Q_(1, "flop") / hw.compute.peak_flops
    serving = ServingModel().solve(model, hw, 1, batch_size=batch, efficiency=1)
    batching = ContinuousBatchingModel().solve(model, hw, max_seq_len=1, max_batch_size=batch, efficiency=1)
    assert serving.feasible and batching.feasible
    assert serving.itl >= minimum
    assert batching.itl >= minimum
    assert batching.itl.m_as("ms") == pytest.approx(serving.itl.m_as("ms"))
    slower = ServingModel().solve(model, hw, 1, batch_size=batch, efficiency=0.5)
    assert slower.itl > serving.itl


@pytest.mark.parametrize("acceptance", [0.0, 0.5, 1.0])
def test_speculation_cannot_evade_target_linear_compute_floor(acceptance):
    model, hw, batch = Models.Language.Llama3_8B, Hardware.Cloud.H100, 512
    # Every emitted token still needs its target-model distribution, including
    # the bonus token after a fully accepted draft. Draft work is additional.
    minimum = 2 * model.parameters.m_as("count") * batch * Q_(1, "flop") / hw.compute.peak_flops
    result = ServingModel().solve(
        model, hw, 1, batch_size=batch, efficiency=1,
        draft_model=Models.Language.GPT2_Small, draft_acceptance_rate=acceptance,
    )
    assert result.feasible
    assert result.itl >= minimum


def test_decode_attention_work_grows_with_context():
    model, base, batch = Models.Language.Llama3_8B, Hardware.Cloud.H100, 8
    # Construct a compute-bound device to isolate context-dependent work;
    # these are test inputs, not specifications for an actual accelerator.
    hw = base.model_copy(update={
        "memory": base.memory.model_copy(update={
            "bandwidth": Q_(1e30, "byte/s"), "capacity": Q_(1e30, "byte"),
        }),
        "compute": base.compute.model_copy(update={"precision_flops": {}}),
    })
    short_context, long_context = 128, 256
    short = ServingModel().solve(model, hw, short_context, batch_size=batch, efficiency=1)
    long = ServingModel().solve(model, hw, long_context, batch_size=batch, efficiency=1)
    # QK^T and attention-weighted V each need one multiply and one add per
    # head element and context token. Linear work and layer overhead cancel.
    extra_flops = (2 * 2 * model.layers * model.heads * model.head_width
                   * (long_context - short_context) * batch * Q_(1, "flop"))
    expected_delta = extra_flops / hw.compute.peak_flops
    assert short.feasible and long.feasible
    assert (long.itl - short.itl).m_as("ms") == pytest.approx(expected_delta.m_as("ms"))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("validate", [
    lambda value: validate_positive(value, "input"),
    lambda value: validate_nonnegative(value, "input"),
    lambda value: validate_at_least(value, 1, "input"),
    lambda value: validate_range(value, 0, 1, "input"),
])
def test_nonfinite_input_fails_at_validation_boundary(value, validate):
    with pytest.raises(ValueError, match="finite"):
        validate(value)


def test_serving_rejects_nonfinite_efficiency():
    with pytest.raises(ValueError, match="finite"):
        ServingModel().solve(Models.Language.GPT2_Small, Hardware.Cloud.H100, 128, efficiency=math.nan)


@pytest.mark.parametrize("fraction", [0.0, 0.5, 0.9, 1.0])
def test_amdahl_infinite_speedup_limit(fraction):
    from mlsysim.physics.performance import calc_amdahls_speedup

    expected = float("inf") if fraction == 1.0 else 1 / (1 - fraction)
    assert calc_amdahls_speedup(fraction, float("inf")) == expected


@pytest.mark.parametrize("speedup", [float("nan"), float("-inf")])
def test_amdahl_rejects_invalid_speedup(speedup):
    from mlsysim.physics.performance import calc_amdahls_speedup

    with pytest.raises(ValueError):
        calc_amdahls_speedup(0.5, speedup)
