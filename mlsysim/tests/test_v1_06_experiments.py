"""Causal tests for the Chapter 6 architecture experiments."""

import json
from dataclasses import replace

import pytest

from mlsysim.core.units import ureg
from mlsysim.engine.v1_06_experiments import (
    ArchitectureCandidate,
    choose_architecture,
    default_scale_candidate,
    evaluate_candidate,
    evaluation_replay_snapshot,
    evaluation_to_dict,
    get_candidate,
    get_scenario,
    highest_quality_candidate,
    physical_signature,
    quality_at,
    replay_evaluation,
    replay_snapshot,
    scaling_interpretation,
    sweep_scale,
    with_execution_resources,
)


TRACKS = ("mobile", "tinyml", "edge", "cloud")


@pytest.mark.parametrize("track_id", TRACKS)
def test_all_tracks_have_matched_candidates_and_explicit_quality_fixtures(track_id):
    scenario = get_scenario(track_id)
    assert len(scenario.candidates) >= 3
    assert len({candidate.compatible_axis for candidate in scenario.candidates}) == 1
    assert all(candidate.compatible_axis == scenario.scale_axis for candidate in scenario.candidates)
    for candidate in scenario.candidates:
        assert candidate.quality_curve is not None
        assert candidate.quality_curve.evidence_kind == "illustrative"
        assert "not a benchmark measurement" in candidate.quality_curve.source
        assert len(candidate.quality_curve.observations) >= 3


def test_track_aliases_resolve_to_the_same_scenarios():
    assert get_scenario("iphone") is get_scenario("mobile")
    assert get_scenario("oura_ring") is get_scenario("tinyml")
    assert get_scenario("robotaxi") is get_scenario("edge")
    assert get_scenario("cloud_fleet") is get_scenario("cloud")


def test_dense_counts_are_exact_for_mobile_image_shape():
    scenario = get_scenario("mobile")
    candidate = get_candidate(scenario, "mobile_dense")
    signature = physical_signature(scenario, candidate.candidate_id, scale_value=112)
    input_width = 112 * 112 * candidate.input_channels
    expected_params = (
        input_width * candidate.hidden_width
        + (candidate.layers - 1) * candidate.hidden_width**2
        + candidate.hidden_width * candidate.output_width
    )
    assert signature.parameters == expected_params
    assert signature.macs == scenario.batch_size * expected_params
    assert signature.flops.m_as(ureg.flop) == 2 * signature.macs


def test_convolution_weight_sharing_keeps_parameters_constant_as_resolution_changes():
    scenario = get_scenario("mobile")
    small = physical_signature(scenario, "mobile_conv", scale_value=112)
    large = physical_signature(scenario, "mobile_conv", scale_value=224)
    assert large.parameters == small.parameters
    assert (
        large.macs
        == 4 * small.macs
        - 3
        * scenario.batch_size
        * get_candidate(scenario, "mobile_conv").hidden_width
        * get_candidate(scenario, "mobile_conv").output_width
    )
    assert large.activation_memory > small.activation_memory


def test_recurrent_work_and_serial_floor_grow_with_sequence_length():
    scenario = get_scenario("tinyml")
    short = physical_signature(scenario, "tiny_recurrent", scale_value=4)
    long = physical_signature(scenario, "tiny_recurrent", scale_value=8)
    assert long.logical_steps == 2 * short.logical_steps
    assert long.serial_floor_time == 2 * short.serial_floor_time
    assert long.macs > short.macs
    assert long.resident_state_memory == short.resident_state_memory


def test_attention_quadratic_work_is_separate_from_materialized_score_memory():
    scenario = get_scenario("cloud")
    candidate = get_candidate(scenario, "cloud_attention")
    short = physical_signature(scenario, candidate.candidate_id, scale_value=256)
    long = physical_signature(scenario, candidate.candidate_id, scale_value=512)
    assert long.attention_score_memory == 4 * short.attention_score_memory
    assert long.activation_memory.m_as(ureg.byte) == pytest.approx(
        2 * short.activation_memory.m_as(ureg.byte), rel=0.001
    )
    assert long.macs > 2 * short.macs

    no_scores = replace(candidate, materialize_attention_scores=False)
    modified = replace(
        scenario,
        candidates=tuple(
            no_scores if item.candidate_id == candidate.candidate_id else item for item in scenario.candidates
        ),
    )
    without_scores = physical_signature(modified, candidate.candidate_id, scale_value=512)
    assert without_scores.macs == long.macs
    assert without_scores.attention_score_memory.m_as(ureg.byte) == 0
    assert without_scores.total_runtime_memory < long.total_runtime_memory


def test_compute_and_bandwidth_change_only_their_execution_bounds():
    scenario = get_scenario("edge")
    candidate_id = "edge_attention"
    baseline = evaluate_candidate(scenario, candidate_id)
    more_compute = evaluate_candidate(scenario, candidate_id, compute_multiplier=2.0)
    more_bandwidth = evaluate_candidate(scenario, candidate_id, bandwidth_multiplier=2.0)
    for changed in (more_compute, more_bandwidth):
        assert changed.signature.macs == baseline.signature.macs
        assert changed.signature.total_runtime_memory == baseline.signature.total_runtime_memory
        assert changed.quality_percent == baseline.quality_percent
        assert changed.quality_evidence_kind == baseline.quality_evidence_kind
    assert more_compute.signature.parallel_time <= baseline.signature.parallel_time
    assert more_bandwidth.signature.parallel_time <= baseline.signature.parallel_time


def test_evaluation_records_complete_resolved_replay_inputs():
    scenario = get_scenario("cloud")
    evaluation = evaluate_candidate(
        scenario,
        "cloud_attention",
        scale_value=1_024,
        training_examples=50_000,
        compute_multiplier=1.5,
        bandwidth_multiplier=0.75,
        memory_budget=8 * ureg.GB,
        latency_budget=60 * ureg.millisecond,
        quality_floor_percent=87.0,
    )
    assert evaluation.inputs.track_id == "cloud"
    assert evaluation.inputs.candidate_id == "cloud_attention"
    assert evaluation.inputs.scale_value == 1_024
    assert evaluation.inputs.training_examples == 50_000
    assert evaluation.inputs.compute_multiplier == 1.5
    assert evaluation.inputs.bandwidth_multiplier == 0.75
    assert evaluation.inputs.memory_budget == 8 * ureg.GB
    assert evaluation.inputs.latency_budget == 60 * ureg.millisecond
    assert evaluation.inputs.quality_floor_percent == 87.0
    assert replay_evaluation(evaluation.inputs) == evaluation

    encoded = json.dumps(evaluation_replay_snapshot(evaluation))
    decoded = json.loads(encoded)
    replayed = replay_snapshot(decoded)
    assert replayed == evaluation
    assert decoded["model_key"] == "v1_06.evaluate_candidate.v1"
    assert decoded["inputs"]["memory_budget"] == {"value": 8.0, "unit": "GB"}
    result_record = json.loads(json.dumps(evaluation_to_dict(evaluation)))
    assert result_record["inputs"] == decoded["inputs"]
    assert result_record["latency"]["unit"] == "s"


def test_replay_rejects_wrong_method_and_incomplete_inputs():
    with pytest.raises(ValueError, match="unsupported model_key"):
        replay_snapshot({"model_key": "v1_06.unknown.v1", "inputs": {}})
    with pytest.raises(ValueError, match="evaluation inputs missing"):
        replay_snapshot({"model_key": "v1_06.evaluate_candidate.v1", "inputs": {}})


def test_recurrent_latency_uses_max_critical_path_instead_of_adding_serial_tax():
    scenario = get_scenario("cloud")
    signature = physical_signature(scenario, "cloud_recurrent")
    assert signature.latency == max(signature.parallel_time, signature.serial_floor_time)
    assert signature.latency != signature.parallel_time + signature.serial_floor_time


def test_quality_interpolation_is_independent_of_family_and_hardware():
    scenario = get_scenario("mobile")
    candidate = get_candidate(scenario, "mobile_conv")
    midpoint = quality_at(candidate, 15_000)
    assert midpoint == pytest.approx(81.5)
    slower_system = with_execution_resources(
        scenario,
        compute_rate=scenario.compute_rate / 10,
        memory_bandwidth=scenario.memory_bandwidth / 10,
    )
    assert evaluate_candidate(slower_system, candidate.candidate_id).quality_percent == quality_at(
        candidate, scenario.training_examples
    )
    assert highest_quality_candidate(scenario, training_examples=20_000) == "mobile_attention"


@pytest.mark.parametrize("track_id", TRACKS)
def test_each_track_has_a_valid_and_failed_requirement_case(track_id):
    scenario = get_scenario(track_id)
    candidate_id = scenario.candidates[0].candidate_id
    valid = evaluate_candidate(
        scenario,
        candidate_id,
        memory_budget=100 * ureg.TB,
        latency_budget=1_000 * ureg.second,
        quality_floor_percent=0.0,
    )
    failed = evaluate_candidate(
        scenario,
        candidate_id,
        memory_budget=1 * ureg.byte,
        latency_budget=1 * ureg.nanosecond,
        quality_floor_percent=100.0,
    )
    assert valid.feasible
    assert not failed.feasible
    assert failed.violations == ("memory budget", "latency budget", "quality floor")


@pytest.mark.parametrize("track_id", TRACKS)
def test_scale_sweep_changes_physics_but_not_supplied_quality(track_id):
    scenario = get_scenario(track_id)
    candidate_id = scenario.candidates[0].candidate_id
    points = sweep_scale(scenario, candidate_id)
    assert tuple(point.scale_value for point in points) == scenario.sweep_values
    assert points[-1].evaluation.signature.macs > points[0].evaluation.signature.macs
    assert points[-1].evaluation.quality_percent == points[0].evaluation.quality_percent


def test_incompatible_sweep_axis_and_candidate_cannot_enter_comparison():
    scenario = get_scenario("mobile")
    with pytest.raises(ValueError, match="supports a resolution sweep"):
        sweep_scale(scenario, "mobile_conv", axis="context")

    incompatible = replace(get_candidate(scenario, "mobile_conv"), compatible_axis="context")
    invalid_scenario = replace(scenario, candidates=(incompatible,) + scenario.candidates[1:])
    with pytest.raises(ValueError, match="supports context, not resolution"):
        physical_signature(invalid_scenario, incompatible.candidate_id)


def test_requirement_can_reverse_ranking_without_composite_score():
    scenario = get_scenario("mobile")
    permissive = choose_architecture(
        scenario,
        priority="quality",
        memory_budget=10 * ureg.GB,
        latency_budget=10 * ureg.second,
        quality_floor_percent=0.0,
    )
    memory_limited = choose_architecture(
        scenario,
        priority="quality",
        memory_budget=3.6 * ureg.MB,
        latency_budget=10 * ureg.second,
        quality_floor_percent=0.0,
    )
    assert permissive.selected_id == "mobile_attention"
    assert memory_limited.selected_id == "mobile_conv"


def test_no_feasible_design_is_explicit():
    scenario = get_scenario("tinyml")
    decision = choose_architecture(
        scenario,
        memory_budget=1 * ureg.byte,
        latency_budget=1 * ureg.nanosecond,
        quality_floor_percent=100.0,
    )
    assert decision.selected_id is None
    assert not decision.feasible_ids
    assert len(decision.rejected) == len(scenario.candidates)


def test_bad_track_candidate_and_resource_inputs_fail_loudly():
    with pytest.raises(ValueError, match="unknown track"):
        get_scenario("desktop")
    scenario = get_scenario("mobile")
    with pytest.raises(ValueError, match="not part"):
        physical_signature(scenario, "cloud_attention")
    with pytest.raises(ValueError, match="must be positive"):
        physical_signature(scenario, "mobile_conv", compute_multiplier=0)
    unsupported = replace(get_candidate(scenario, "mobile_conv"), family="graph")
    malformed = replace(scenario, candidates=(unsupported,) + scenario.candidates[1:])
    with pytest.raises(ValueError, match="unsupported architecture family"):
        physical_signature(malformed, unsupported.candidate_id)
    zero_width = replace(get_candidate(scenario, "mobile_conv"), hidden_width=0)
    malformed = replace(scenario, candidates=(zero_width,) + scenario.candidates[1:])
    with pytest.raises(ValueError, match="nonpositive topology fields"):
        physical_signature(malformed, zero_width.candidate_id)


def test_default_scale_candidate_matches_intended_architecture_across_tracks():
    for track_id in TRACKS:
        scenario = get_scenario(track_id)
        candidate_id = default_scale_candidate(scenario)
        candidate = get_candidate(scenario, candidate_id)
        assert candidate.family == "attention"
    assert default_scale_candidate(get_scenario("cloud")) == "cloud_attention"


def test_scaling_interpretation_distinguishes_kernel_and_shape_boundaries():
    tinyml = get_scenario("tinyml")
    tiny_conv_interp = scaling_interpretation(tinyml, "tiny_conv")
    assert "1D temporal" in tiny_conv_interp
    assert "temporal boundary" in tiny_conv_interp

    mobile = get_scenario("mobile")
    mobile_conv_interp = scaling_interpretation(mobile, "mobile_conv")
    assert "2D spatial" in mobile_conv_interp
    assert "strided kernel boundary" in mobile_conv_interp

    tiny_rec_interp = scaling_interpretation(tinyml, "tiny_recurrent")
    assert "Recurrent hidden state" in tiny_rec_interp

    cloud = get_scenario("cloud")
    cloud_dense_interp = scaling_interpretation(cloud, "cloud_dense")
    assert "Dense projection" in cloud_dense_interp

    cloud_attn_interp = scaling_interpretation(cloud, "cloud_attention")
    assert "pairwise score memory" in cloud_attn_interp

    record = evaluation_to_dict(evaluate_candidate(cloud, "cloud_attention"))
    assert record["interpretation"] == cloud_attn_interp
