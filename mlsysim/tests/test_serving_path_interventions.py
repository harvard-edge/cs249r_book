"""Whole-path optimization must preserve task contracts and price resident state."""
from dataclasses import replace
import json

import pytest
from mlsysim.core.units import Q_
from mlsysim.labs.experiments.v1_13_experiments import (
    arrival_trace_for_track, get_recommendation_serving_scenario, get_serving_scenario,
    policy_for_track, replay_serving_snapshot, serving_path_cases, simulate_serving,
    simulation_result_to_snapshot, state_capacity,
)


@pytest.mark.parametrize('track', ('tinyml', 'mobile', 'edge', 'cloud'))
def test_path_changes_preserve_offered_work_and_exact_replay(track):
    base = get_recommendation_serving_scenario() if track == 'cloud' else get_serving_scenario(track)
    arrivals = arrival_trace_for_track(track, 'steady')
    baseline = simulate_serving(base, arrivals, policy_for_track(track, 'baseline'))
    for key, case in serving_path_cases(base).items():
        trial = simulate_serving(case.scenario, arrivals, policy_for_track(track, 'baseline'), case.preprocessing)
        assert trial.offered_count == baseline.offered_count
        assert trial.inputs.arrivals == baseline.inputs.arrivals
        assert trial.inputs.policy == baseline.inputs.policy
        assert trial.inputs.scenario.deadline == base.deadline
        assert trial.preprocessing_parity == (key != 'incompatible')
        assert replay_serving_snapshot(json.loads(json.dumps(simulation_result_to_snapshot(trial)["inputs"]))) == trial
        if key == 'incompatible':
            assert trial.useful_completion_count == 0
        if key == 'host':
            assert case.scenario.execution_latency == base.execution_latency
            assert case.scenario.resident_model_memory == base.resident_model_memory
            assert trial.mean_latency < baseline.mean_latency
            assert state_capacity(case.scenario).fixed_memory > state_capacity(base).fixed_memory
        if key == 'model':
            assert case.preprocessing == base.preprocessing_paths[0]
            assert case.scenario.resident_model_memory > base.resident_model_memory


def test_faster_model_can_lose_all_service_when_second_artifact_does_not_fit():
    base = get_serving_scenario('tinyml')
    case = serving_path_cases(base)['model']
    assert case.scenario.execution_latency < base.execution_latency
    assert state_capacity(base).feasible
    assert not state_capacity(case.scenario).feasible
    trial = simulate_serving(case.scenario, arrival_trace_for_track('tinyml', 'steady'),
                             policy_for_track('tinyml', 'baseline'), case.preprocessing)
    assert trial.rejected_count == trial.offered_count
    assert trial.mean_latency is None
    assert trial.useful_completion_count == 0


def test_host_choice_reverses_at_a_tight_memory_budget():
    base = get_serving_scenario('mobile')
    base = replace(base, device_memory=base.resident_model_memory + base.batch_workspace + base.request_working_state)
    case = serving_path_cases(base)['host']
    assert state_capacity(base).feasible
    assert not state_capacity(case.scenario).feasible
    assert case.preprocessing.output_signature == base.reference_preprocessing_signature
    assert case.preprocessing.latency < base.preprocessing_paths[0].latency


def test_local_speedup_is_not_a_whole_path_speedup():
    base = get_serving_scenario('mobile')
    # One request removes queueing: unchanged ingress/postprocessing remain paid.
    case = serving_path_cases(base)['host']
    policy = policy_for_track('mobile', 'baseline')
    baseline = simulate_serving(base, (Q_(0, 'ms'),), policy)
    trial = simulate_serving(case.scenario, (Q_(0, 'ms'),), policy, case.preprocessing)
    saved = baseline.mean_latency - trial.mean_latency
    assert saved == base.preprocessing_paths[0].latency - case.preprocessing.latency
    assert baseline.mean_latency / trial.mean_latency < base.preprocessing_paths[0].latency / case.preprocessing.latency


@pytest.mark.parametrize('scale', (1, 2, 4, 8, 16))
def test_retained_history_spends_kv_memory_without_claiming_a_timing_change(scale):
    from mlsysim.labs.experiments.v1_13_experiments import get_language_serving_reference, state_capacity_to_snapshot
    from mlsysim.labs.experiments.v1_16_experiments import replay_evidence_arm
    base = get_language_serving_reference()
    trial = get_language_serving_reference(context_scale=scale)
    assert trial.request_kv_state == base.request_kv_state * scale
    assert trial.execution_latency == base.execution_latency
    assert trial.resident_model_memory == base.resident_model_memory
    assert trial.request_working_state == base.request_working_state
    assert state_capacity(trial).max_requests_per_replica <= state_capacity(base).max_requests_per_replica
    result = state_capacity_to_snapshot(state_capacity(trial), pressure='baseline')
    assert replay_evidence_arm('v1_13_experiments', 'v1_13.language_state_capacity',
                               {'inputs': {'pressure': 'baseline', 'context_scale': scale}}) == result


def test_long_history_can_fail_while_weights_still_fit():
    from mlsysim.labs.experiments.v1_13_experiments import get_language_serving_reference
    trial = get_language_serving_reference(context_scale=16)
    assert trial.resident_model_memory < trial.device_memory
    assert not state_capacity(trial).feasible


@pytest.mark.parametrize('scale', (True, 0, -1, 3, 2.0))
def test_history_reference_rejects_unsupported_controls(scale):
    from mlsysim.labs.experiments.v1_13_experiments import get_language_serving_reference
    with pytest.raises(ValueError, match='context_scale'):
        get_language_serving_reference(context_scale=scale)
