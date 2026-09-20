"""Causal and all-track tests for the Chapter 10 compression experiments."""

from __future__ import annotations

import json

import pytest

from mlsysim.engine.v1_10_experiments import (
    CANONICAL_TRACK_IDS,
    MODEL_KEYS,
    TRACK_IDS,
    capture_experiment,
    compression_model_baseline_size,
    evaluate_distillation,
    evaluate_precision,
    evaluate_recipe,
    evaluate_recipe_id,
    evaluate_sparsity,
    evaluate_sparsity_case,
    get_scenario,
    replay_experiment,
    restore_snapshot,
)


@pytest.mark.parametrize(
    ("legacy_id", "canonical_id"),
    tuple(zip(TRACK_IDS, CANONICAL_TRACK_IDS)),
)
def test_all_tracks_resolve_registry_models_and_fp32_baseline(legacy_id, canonical_id):
    scenario = get_scenario(legacy_id)
    alias_scenario = get_scenario(canonical_id)
    parameters = scenario.model.parameters

    assert alias_scenario is scenario
    assert scenario.ledger_track_id == canonical_id
    assert parameters is not None
    assert parameters.to("param").magnitude > 0
    assert scenario.hardware.memory.capacity.to("byte").magnitude > 0
    expected_fp32 = parameters.to("param").magnitude * 4
    assert compression_model_baseline_size(legacy_id).to("byte").magnitude == pytest.approx(
        expected_fp32
    )


@pytest.mark.parametrize(
    ("track_id", "passing_calibration", "failing_bits", "failing_calibration"),
    (
        ("iphone", "ptq", 4, "ptq"),
        ("oura_ring", "ptq", 4, "ptq"),
        ("robotaxi", "qat", 8, "ptq"),
        ("cloud_fleet", "ptq", 4, "ptq"),
    ),
)
def test_precision_has_valid_and_failed_quality_cases(
    track_id, passing_calibration, failing_bits, failing_calibration
):
    passing = evaluate_precision(
        track_id,
        storage_bits=8,
        compute_bits=8,
        calibration=passing_calibration,
    )
    failing = evaluate_precision(
        track_id,
        storage_bits=failing_bits,
        compute_bits=8,
        calibration=failing_calibration,
    )

    passing_quality = passing.task_quality
    assert passing.feasible
    assert passing_quality is not None
    assert passing_quality >= get_scenario(track_id).quality_floor
    assert "quality" in failing.violations
    assert not failing.feasible


def test_quantized_storage_includes_metadata_and_precision_roles_are_separate():
    scenario = get_scenario("iphone")
    parameters = scenario.model.parameters
    assert parameters is not None
    parameter_count = parameters.to("param").magnitude
    storage_only = evaluate_precision(
        "iphone", storage_bits=8, compute_bits=16, calibration="ptq"
    )
    accelerated = evaluate_precision(
        "iphone", storage_bits=8, compute_bits=8, calibration="ptq"
    )

    raw_int8_bytes = parameter_count
    assert storage_only.artifact_size.to("byte").magnitude > raw_int8_bytes
    assert storage_only.artifact_size == accelerated.artifact_size
    assert storage_only.task_quality == accelerated.task_quality
    assert storage_only.model_execution_time > accelerated.model_execution_time


def test_calibration_cost_and_quality_come_from_exact_supplied_rows():
    ptq = evaluate_precision(
        "robotaxi", storage_bits=8, compute_bits=8, calibration="ptq"
    )
    qat = evaluate_precision(
        "robotaxi", storage_bits=8, compute_bits=8, calibration="qat"
    )
    unknown = evaluate_precision(
        "robotaxi", storage_bits=7, compute_bits=8, calibration="ptq"
    )

    qat_quality = qat.task_quality
    ptq_quality = ptq.task_quality
    assert qat.preparation_time > ptq.preparation_time
    assert qat_quality is not None
    assert ptq_quality is not None
    assert qat_quality > ptq_quality
    assert unknown.task_quality is None
    assert "quality_outcome_unavailable" in unknown.violations


def test_runtime_state_has_a_floor_and_does_not_change_task_quality():
    state_16 = evaluate_precision(
        "oura_ring", storage_bits=8, compute_bits=8,
        runtime_state_bits=16, calibration="ptq",
    )
    state_8 = evaluate_precision(
        "oura_ring", storage_bits=8, compute_bits=8,
        runtime_state_bits=8, calibration="ptq",
    )

    assert state_8.runtime_state_size == get_scenario("oura_ring").minimum_runtime_state
    assert state_16.runtime_state_size > state_8.runtime_state_size
    assert state_16.task_quality == state_8.task_quality
    assert state_16.artifact_size == state_8.artifact_size


@pytest.mark.parametrize(
    ("track_id", "passing_sparsity"),
    (
        ("iphone", 0.25),
        ("oura_ring", 0.50),
        ("robotaxi", 0.25),
        ("cloud_fleet", 0.25),
    ),
)
def test_sparsity_has_supported_and_unsupported_paths(track_id, passing_sparsity):
    supported = evaluate_sparsity(
        track_id, sparsity=passing_sparsity, representation="structured"
    )
    masked = evaluate_sparsity(
        track_id, sparsity=0.50, representation="dense_mask"
    )

    assert supported.execution_supported
    assert supported.feasible
    assert not masked.execution_supported
    assert "execution_support" in masked.violations
    assert masked.model_execution_time == get_scenario(track_id).baseline_model_time


def test_sparse_metadata_can_erase_storage_savings_without_changing_quality():
    dense = evaluate_precision(
        "cloud_fleet", storage_bits=16, compute_bits=16, calibration="none"
    )
    index_16 = evaluate_sparsity(
        "cloud_fleet", sparsity=0.25, representation="indexed", index_bits=16
    )
    index_32 = evaluate_sparsity(
        "cloud_fleet", sparsity=0.25, representation="indexed", index_bits=32
    )

    assert index_32.artifact_size > index_16.artifact_size
    assert index_32.artifact_size > dense.artifact_size
    assert index_32.task_quality == index_16.task_quality
    assert index_32.model_execution_time == index_16.model_execution_time


def test_named_sparsity_case_matches_direct_evaluation():
    named = evaluate_sparsity_case("mobile", "structured_25", index_bits=16)
    direct = evaluate_sparsity(
        "mobile", sparsity=0.25, representation="structured", index_bits=16
    )

    assert named == direct


@pytest.mark.parametrize("track_id", TRACK_IDS)
def test_dense_student_trades_one_time_training_for_deployment_savings(track_id):
    small = evaluate_distillation(
        track_id, candidate_id="small_dense", deployment_inferences=10_000_000
    )
    tiny = evaluate_distillation(
        track_id, candidate_id="tiny_dense", deployment_inferences=10_000_000
    )

    assert small.deployment.feasible
    assert small.training_time.to("hour").magnitude > 0
    assert small.latency_saved_per_inference.to("ms").magnitude > 0
    assert small.break_even_inferences is not None
    assert small.amortized_time_per_inference > small.deployment.end_to_end_latency
    assert "quality" in tiny.deployment.violations


def test_deployment_count_changes_amortization_but_not_deployed_outcome():
    few = evaluate_distillation(
        "iphone", candidate_id="small_dense", deployment_inferences=100
    )
    many = evaluate_distillation(
        "iphone", candidate_id="small_dense", deployment_inferences=100_000_000
    )

    assert few.deployment.end_to_end_latency == many.deployment.end_to_end_latency
    assert few.deployment.task_quality == many.deployment.task_quality
    assert few.amortized_time_per_inference > many.amortized_time_per_inference
    assert not few.training_amortized
    assert many.training_amortized


@pytest.mark.parametrize("track_id", TRACK_IDS)
def test_supported_ordered_recipe_recomputes_common_path(track_id):
    sequence = ("distill_small", "prune_structured_25", "quant_int8_ptq")
    recipe = evaluate_recipe(track_id, sequence)

    assert recipe.outcome_available
    assert recipe.deployment.feasible
    assert tuple(stage.transformation for stage in recipe.stages) == sequence
    assert recipe.stages[1].logical_parameters < recipe.stages[0].logical_parameters
    assert recipe.stages[2].storage_bits == 8
    assert recipe.deployment.end_to_end_latency != recipe.naive_end_to_end_latency


def test_recipe_order_changes_representation_and_unknown_outcome_stays_unknown():
    quant_then_distill = evaluate_recipe(
        "iphone", ("quant_int8_ptq", "distill_small")
    )
    distill_then_quant = evaluate_recipe(
        "iphone", ("distill_small", "quant_int8_ptq")
    )

    assert quant_then_distill.stages[-1].storage_bits == 16
    assert distill_then_quant.stages[-1].storage_bits == 8
    assert quant_then_distill.deployment.artifact_size > distill_then_quant.deployment.artifact_size
    assert not quant_then_distill.outcome_available
    assert quant_then_distill.deployment.task_quality is None
    assert "quality_outcome_unavailable" in quant_then_distill.deployment.violations


def test_named_recipe_matches_direct_ordered_evaluation():
    named = evaluate_recipe_id("edge", "distill_prune_quant")
    direct = evaluate_recipe(
        "edge", ("distill_small", "prune_structured_25", "quant_int8_ptq")
    )

    assert named == direct


def test_inputs_are_complete_canonical_and_stable():
    precision = evaluate_precision(
        "mobile", storage_bits=8, compute_bits=8,
        runtime_state_bits=16, calibration="ptq",
    )
    sparse = evaluate_sparsity(
        "edge", sparsity=0.25, representation="structured",
        storage_bits=16, index_bits=16, runtime_state_bits=16,
    )
    distilled = evaluate_distillation(
        "cloud", candidate_id="small_dense",
        deployment_inferences=12_345, runtime_state_bits=16,
    )
    recipe = evaluate_recipe(
        "tinyml", ("prune_structured_25", "quant_int8_ptq"),
        runtime_state_bits=16,
    )

    assert precision.inputs == {
        "track_id": "mobile", "storage_bits": 8, "compute_bits": 8,
        "runtime_state_bits": 16, "calibration": "ptq",
    }
    assert set(sparse.inputs) == {
        "track_id", "sparsity", "representation", "storage_bits",
        "index_bits", "runtime_state_bits",
    }
    assert distilled.deployment.inputs["deployment_inferences"] == 12_345
    assert recipe.deployment.inputs["transformations"] == (
        "prune_structured_25", "quant_int8_ptq"
    )


@pytest.mark.parametrize(
    ("model_key", "inputs"),
    (
        (
            MODEL_KEYS["resource"],
            dict(
                track_id="tinyml", storage_bits=8, compute_bits=8,
                runtime_state_bits=16, calibration="ptq",
            ),
        ),
        (
            MODEL_KEYS["precision"],
            dict(
                track_id="mobile", storage_bits=8, compute_bits=8,
                runtime_state_bits=16, calibration="qat",
            ),
        ),
        (
            MODEL_KEYS["sparsity"],
            dict(
                track_id="edge", sparsity=0.25, representation="structured",
                storage_bits=16, index_bits=16, runtime_state_bits=16,
            ),
        ),
        (
            MODEL_KEYS["distillation"],
            dict(
                track_id="cloud", candidate_id="small_dense",
                deployment_inferences=123_456, runtime_state_bits=16,
            ),
        ),
        (
            MODEL_KEYS["recipe"],
            dict(
                track_id="tinyml",
                transformations=(
                    "distill_small", "prune_structured_25", "quant_int8_ptq"
                ),
                runtime_state_bits=16,
            ),
        ),
    ),
)
def test_json_snapshot_restore_and_replay_are_exact(model_key, inputs):
    captured = capture_experiment(model_key, **inputs)
    payload = json.loads(json.dumps(captured.to_dict()))

    assert payload["model_key"] == model_key
    assert payload["track_id"] == inputs["track_id"]
    assert isinstance(payload["inputs"], dict)
    assert payload["result"]["inputs"] == payload["inputs"]
    restored = restore_snapshot(payload)
    assert restored.to_dict() == payload
    assert replay_experiment(payload).to_dict() == payload


def test_invalid_inputs_fail_loudly():
    with pytest.raises(KeyError):
        get_scenario("unknown")
    with pytest.raises(ValueError):
        evaluate_precision(
            "iphone", storage_bits=0, compute_bits=8, calibration="ptq"
        )
    with pytest.raises(ValueError):
        evaluate_sparsity(
            "iphone", sparsity=1.0, representation="structured"
        )
    with pytest.raises(KeyError):
        evaluate_distillation("iphone", candidate_id="missing")
    with pytest.raises(ValueError):
        evaluate_recipe("iphone", ("magic",))


@pytest.mark.parametrize(
    ("track_id", "expected_max_small_training_hours", "test_low_volume", "test_high_volume"),
    (
        ("oura_ring", 0.5, 10_000, 100_000),
        ("iphone", 5.0, 100_000, 10_000_000),
        ("robotaxi", 7.0, 1_000_000, 10_000_000),
        ("cloud_fleet", 15.0, 1_000_000, 100_000_000),
    ),
)
def test_track_distillation_amortization_reflects_deployment_profile(
    track_id, expected_max_small_training_hours, test_low_volume, test_high_volume
):
    scenario = get_scenario(track_id)
    assert scenario.default_inferences in scenario.lifetime_inferences.values()
    assert scenario.default_inferences_key in scenario.lifetime_inferences

    small = evaluate_distillation(
        track_id, candidate_id="small_dense", deployment_inferences=scenario.default_inferences
    )
    assert small.training_time.to("hour").magnitude <= expected_max_small_training_hours
    assert small.break_even_inferences is not None

    unamortized = evaluate_distillation(
        track_id, candidate_id="small_dense", deployment_inferences=test_low_volume
    )
    amortized = evaluate_distillation(
        track_id, candidate_id="small_dense", deployment_inferences=test_high_volume
    )
    assert not unamortized.training_amortized
    assert amortized.training_amortized
