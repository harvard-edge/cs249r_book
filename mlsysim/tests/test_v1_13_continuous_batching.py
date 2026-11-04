"""Causal request accounting for a bounded, uncalibrated decode trace."""
import json
import pytest
from mlsysim.labs.experiments.v1_13_experiments import language_batching_reference


def test_refill_reduces_heterogeneous_cohort_waste_without_changing_work():
    static = language_batching_reference('static')
    refill = language_batching_reference('continuous')
    assert refill['total_output_tokens'] == static['total_output_tokens']
    assert refill['completed_count'] == static['offered_count']
    assert refill['idle_slot_iterations'] < static['idle_slot_iterations']
    assert refill['elapsed_ms'] < static['elapsed_ms']
    assert refill['tokens_per_second'] > static['tokens_per_second']
    assert refill['requests'][4]['first_decode_ms'] < static['requests'][4]['first_decode_ms']
    assert refill['peak_memory_bytes'] == static['peak_memory_bytes']


@pytest.mark.parametrize('kwargs', ({'length_mix': 'equal'}, {'arrival_spacing_ms': 160}))
def test_equal_cohorts_or_sparse_arrivals_remove_refill_advantage(kwargs):
    static = language_batching_reference('static', **kwargs)
    refill = language_batching_reference('continuous', **kwargs)
    assert static['elapsed_ms'] == refill['elapsed_ms']
    assert static['requests'] == refill['requests']


@pytest.mark.parametrize('policy', ('static', 'continuous'))
@pytest.mark.parametrize('spacing', (0, 16, 160))
@pytest.mark.parametrize('scale', (1, 2, 4, 8, 16))
def test_trace_conservation_memory_and_json_replay(policy, spacing, scale):
    result = language_batching_reference(policy, arrival_spacing_ms=spacing, context_scale=scale)
    replay = language_batching_reference(**json.loads(json.dumps(result))['inputs'])
    assert result == replay
    assert result['completed_count'] + result['rejected_count'] == result['offered_count']
    if not result['feasible']:
        assert result['tokens_per_second'] is None
        assert result['peak_memory_bytes'] is None
        assert result['completed_count'] == 0
        assert result['timeline'] == []
        return
    assert result['peak_memory_bytes'] <= result['device_memory_bytes']
    emitted = {i: [] for i in range(result['offered_count'])}
    for step in result['timeline']:
        assert len(step['request_ids']) <= step['reserved_slots'] <= result['slot_limit']
        assert step['end_ms'] - step['start_ms'] == result['iteration_ms']
        for request_id in step['request_ids']:
            emitted[request_id].append(step['end_ms'])
    for row in result['requests']:
        times = emitted[row['request_id']]
        assert len(times) == row['output_tokens']
        assert times[0] == row['first_decode_ms']
        assert times[-1] == row['completion_ms']
        assert row['arrival_ms'] <= times[0] - result['iteration_ms']
    assert sum(len(step['request_ids']) for step in result['timeline']) == result['total_output_tokens']


def test_history_only_changes_memory_admission_not_supplied_iteration_cost():
    small = language_batching_reference(context_scale=1)
    large = language_batching_reference(context_scale=8)
    assert large['slot_limit'] < small['slot_limit']
    assert large['iteration_ms'] == small['iteration_ms']
    assert large['total_output_tokens'] == small['total_output_tokens']
    assert large['elapsed_ms'] > small['elapsed_ms']


@pytest.mark.parametrize('kwargs', ({'policy':'random'}, {'length_mix':'arbitrary'},
                                  {'arrival_spacing_ms': True}, {'arrival_spacing_ms': 3},
                                  {'context_scale': 3}))
def test_unsupported_controls_rejected(kwargs):
    with pytest.raises(ValueError):
        language_batching_reference(**kwargs)
