"""Small exact instruments must expose missing evidence and invariant arithmetic."""
import pytest

from mlsysim.labs.experiments import v1_05_experiments as computation
from mlsysim.labs.experiments import v1_06_experiments as architecture


def test_dependency_trace_matches_finite_difference_for_both_weights():
    fixture = computation.DEPENDENCY_FIXTURE
    result = computation.run_dependency_trace()
    epsilon = 1e-5
    def loss(w1, w2):
        prediction = w2 * max(0, w1 * fixture['x'] + fixture['b1']) + fixture['b2']
        return (prediction - fixture['target']) ** 2 / 2
    for key in ('w1', 'w2'):
        plus, minus = dict(fixture), dict(fixture)
        plus[key] += epsilon
        minus[key] -= epsilon
        expected = (loss(plus['w1'], plus['w2']) - loss(minus['w1'], minus['w2'])) / (2 * epsilon)
        assert result['gradients']['d_' + key] == pytest.approx(expected)


def test_missing_forward_value_blocks_only_its_dependent_gradient():
    intact = computation.run_dependency_trace()
    lost = computation.run_dependency_trace(discarded=('x',))
    assert lost['gradients']['d_w1'] is None
    assert lost['gradients']['d_w2'] == intact['gradients']['d_w2']
    assert 'x' not in lost['forward_values']
    assert lost['backward_trace'][-1]['missing_operands'] == ['x']
    lost_h = computation.run_dependency_trace(discarded=('h',))
    assert lost_h['gradients']['d_w2'] is None
    assert lost_h['gradients']['d_w1'] == intact['gradients']['d_w1']
    lost_prediction = computation.run_dependency_trace(discarded=('prediction',))
    assert all(value is None for value in lost_prediction['gradients'].values())


def test_dead_relu_zero_is_known_and_distinct_from_missing():
    dead = computation.run_dependency_trace(x=-1)
    assert dead['gradients']['d_w1'] == 0
    assert dead['backward_trace'][-1]['status'] == 'available'
    missing = computation.run_dependency_trace(x=-1, discarded=('x',))
    assert missing['gradients']['d_w1'] is None
    linear = computation.run_dependency_trace(discarded=('z1',), activation='linear')
    assert linear['gradients']['d_w1'] is not None
    assert computation.run_dependency_trace(discarded=('z1',))['gradients']['d_w1'] is None


def test_nonlinearity_constructs_xor_but_affine_identity_precludes_linear_xor():
    nonlinear = computation.run_nonlinearity_case('relu')
    linear = computation.run_nonlinearity_case('linear')
    assert nonlinear['all_match']
    assert not linear['all_match']
    assert linear['layer_dims'] == nonlinear['layer_dims']
    outputs = [row['output'] for row in linear['rows']]
    targets = [row['target'] for row in linear['rows']]
    assert outputs[0] + outputs[3] == outputs[1] + outputs[2]
    assert targets[0] + targets[3] != targets[1] + targets[2]


@pytest.mark.parametrize('runner', [computation.run_dependency_trace, computation.run_nonlinearity_case])
def test_exact_fixture_replay(runner):
    result = runner()
    assert computation.replay(result['model_key'], result['inputs']) == result


@pytest.mark.parametrize('track', ['tinyml', 'mobile', 'edge', 'cloud'])
def test_score_materialization_changes_state_not_dense_arithmetic(track):
    result = architecture.attention_state_comparison(track)
    materialized, tiled = result['baseline'], result['result']
    assert result['dense_attention_arithmetic_unchanged']
    for invariant in ('macs', 'flops', 'parameters', 'tokens_or_positions', 'resident_state_memory', 'activation_memory'):
        assert materialized[invariant] == tiled[invariant]
    assert materialized['attention_score_memory'].magnitude > 0
    assert tiled['attention_score_memory'].magnitude == 0
    assert materialized['total_runtime_memory'] - tiled['total_runtime_memory'] == materialized['attention_score_memory']


def test_language_reference_kv_linear_growth_and_prompt_weight_reuse():
    short = architecture.language_phase_reference(history_tokens=64)
    long = architecture.language_phase_reference(history_tokens=128)
    assert long['prefill']['kv_state'] == 2 * short['prefill']['kv_state']
    assert long['decode']['kv_state'] - long['decode']['kv_read'] == long['decode']['kv_write']
    assert long['prefill']['weight_read_lower_bound'] == long['decode']['weight_read_lower_bound']
    assert long['prefill']['weight_bytes_per_processed_token'] * long['inputs']['history_tokens'] == long['decode']['weight_bytes_per_processed_token']
    assert long['decode']['macs'] > short['decode']['macs']
    assert long['prefill']['macs'] > 2 * short['prefill']['macs']


def test_embedding_vocabulary_capacity_and_width_have_distinct_effects():
    base = architecture.embedding_lookup_reference()
    more_ids = architecture.embedding_lookup_reference(vocabulary_size=2048)
    wider = architecture.embedding_lookup_reference(vector_width=64)
    assert more_ids['table_memory'] == 2 * base['table_memory']
    assert more_ids['request_read_bytes'] == base['request_read_bytes']
    assert wider['table_memory'] == 2 * base['table_memory']
    assert wider['request_read_bytes'] == 2 * base['request_read_bytes']
    assert base['lookup_macs'] == 0
    half_width = architecture.embedding_lookup_reference(precision='fp16')
    assert half_width['table_memory'] * 2 == base['table_memory']


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_reference_dimensions_rejected(value):
    with pytest.raises(ValueError):
        architecture.embedding_lookup_reference(vocabulary_size=value)
    with pytest.raises(ValueError):
        architecture.language_phase_reference(history_tokens=value)
