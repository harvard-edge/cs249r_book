"""Invariant tests for native recommendation and distinct language references."""
import json

import pytest

from mlsysim.core.units import Q_
from mlsysim.labs.experiments.v1_13_experiments import (
    arrival_trace_for_track, get_language_serving_reference,
    get_recommendation_serving_scenario, get_serving_scenario,
    language_scheduling_reference, language_scheduling_to_snapshot, policy_for_track,
    recommendation_state_breakdown, replay_inputs_to_snapshot,
    replay_serving_snapshot, simulate_serving, state_capacity,
)


@pytest.mark.parametrize('pressure', ('baseline', 'elevated', 'overflow'))
def test_native_components_sum_and_no_language_state(pressure):
    parts = recommendation_state_breakdown(pressure)
    scenario = get_recommendation_serving_scenario(pressure)
    assert scenario.resident_model_memory == parts.model_weights + parts.resident_embedding_table
    assert scenario.request_working_state == parts.feature_buffer + parts.gathered_embeddings + parts.ranking_buffer
    assert scenario.request_kv_state == Q_(0, 'byte')
    assert state_capacity(scenario).kv_state_per_request == Q_(0, 'byte')
    assert scenario.reference_preprocessing_signature.startswith('candidate-feature')


def test_native_pressure_changes_fit_without_changing_timing_or_residency():
    base = get_recommendation_serving_scenario()
    elevated = get_recommendation_serving_scenario('elevated')
    overflow = get_recommendation_serving_scenario('overflow')
    assert elevated.execution_latency == base.execution_latency
    assert elevated.resident_model_memory == base.resident_model_memory
    assert elevated.request_working_state == base.request_working_state * 2
    assert state_capacity(elevated).max_requests_per_replica < state_capacity(base).max_requests_per_replica
    assert not state_capacity(overflow).feasible
    assert base.deadline == get_serving_scenario('cloud').deadline


def test_native_schema_mismatch_cannot_be_scored_as_useful_speedup():
    scenario = get_recommendation_serving_scenario()
    arrivals = arrival_trace_for_track('cloud', 'steady')
    policy = policy_for_track('cloud', 'baseline')
    matched = simulate_serving(scenario, arrivals, policy)
    mismatch = simulate_serving(scenario, arrivals, policy, scenario.preprocessing_paths[1])
    assert matched.preprocessing_parity
    assert matched.useful_completion_count > 0
    assert not mismatch.preprocessing_parity
    assert mismatch.useful_completion_count == 0
    assert mismatch.offered_count == matched.offered_count
    snapshot = replay_inputs_to_snapshot(matched.inputs)
    replayed = replay_serving_snapshot(snapshot)
    assert replayed == matched


def test_native_overflow_rejects_without_losing_offered_denominator():
    scenario = get_recommendation_serving_scenario('overflow')
    arrivals = arrival_trace_for_track('cloud', 'steady')
    result = simulate_serving(scenario, arrivals, policy_for_track('cloud', 'baseline'))
    assert result.rejected_count == result.offered_count == len(arrivals)
    assert result.useful_completion_fraction == 0


def test_language_reference_remains_distinct_and_retains_legacy_allocation():
    language = get_language_serving_reference()
    assert language.request_kv_state > Q_(0, 'byte')
    assert language.request_kv_state == get_serving_scenario('cloud').request_kv_state
    assert language.request_kv_state != get_recommendation_serving_scenario().request_kv_state
    assert get_language_serving_reference('elevated').request_kv_state == language.request_kv_state * 2
    assert not state_capacity(get_language_serving_reference('overflow')).feasible


def test_chunking_conserves_work_and_trades_active_stall_for_new_prompt_latency():
    whole = language_scheduling_reference(1)
    chunked = language_scheduling_reference(4)
    assert chunked.total_busy_time == whole.total_busy_time
    assert chunked.prefill_work == whole.prefill_work
    assert chunked.max_active_token_gap < whole.max_active_token_gap
    assert chunked.prompt_first_token > whole.prompt_first_token
    for result in (whole, chunked):
        assert len(result.active_decode_completions) == result.decode_count
        assert sum((s.end - s.start for s in result.timeline if s.phase == 'prefill'), Q_(0, 'ms')) == result.prefill_work
        assert sum(s.stream == 'new' and s.phase == 'decode' for s in result.timeline) == 1
        assert all(left.end == right.start for left, right in zip(result.timeline, result.timeline[1:]))
        assert all(s.end > s.start for s in result.timeline)


@pytest.mark.parametrize('bad', ('missing', None))
def test_bad_pressure_rejected(bad):
    with pytest.raises(ValueError):
        get_recommendation_serving_scenario(bad)


@pytest.mark.parametrize('bad', (0, 3, 1.0, True))
def test_unsupported_chunk_counts_rejected(bad):
    with pytest.raises(ValueError):
        language_scheduling_reference(bad)


@pytest.mark.parametrize("chunks", (1, 2, 4))
def test_language_snapshot_is_json_safe_and_replays_supported_exact_inputs(chunks):
    result = language_scheduling_reference(chunks)
    snapshot = json.loads(json.dumps(language_scheduling_to_snapshot(result)))
    replayed = language_scheduling_to_snapshot(language_scheduling_reference(**snapshot["inputs"]))
    assert snapshot == replayed
    assert snapshot["timeline"][0]["start_ms"] == 0
    assert snapshot["timeline"][-1]["end_ms"] == snapshot["total_busy_time_ms"]
