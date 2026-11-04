"""Exact counterexamples check structural information loss without quality claims."""
import pytest

from mlsysim.labs.experiments.v1_06_experiments import structure_probe


@pytest.mark.parametrize('transform', ['identity', 'mean', 'reversal_invariant'])
def test_presence_task_remains_compatible_with_all_supplied_representations(transform):
    result = structure_probe(task='presence', transform=transform)
    assert result['required_distinctions_preserved']
    assert result['conflicts'] == []
    for pair in result['collisions']:
        assert pair['labels'][0] == pair['labels'][1]


@pytest.mark.parametrize('transform', ['mean', 'reversal_invariant'])
def test_position_sensitive_task_has_an_exact_incompatible_collision(transform):
    result = structure_probe(task='direction', transform=transform)
    assert not result['required_distinctions_preserved']
    witness = result['conflicts'][0]
    rows = {row['case']: row for row in result['rows']}
    left, right = [rows[key] for key in witness['cases']]
    assert left['input'] != right['input']
    assert left['representation'] == right['representation']
    assert left['required_label'] != right['required_label']
    # A function of the collapsed representation cannot satisfy both labels.
    for proposed_answer in (0, 1):
        assert not (proposed_answer == left['required_label'] == right['required_label'])


def test_ordered_representation_restores_position_distinctions_with_same_inputs():
    baseline = structure_probe(task='direction', transform='mean')
    restored = structure_probe(task='direction', transform='identity')
    assert restored['required_distinctions_preserved']
    assert [row['input'] for row in baseline['rows']] == [row['input'] for row in restored['rows']]
    assert [row['required_label'] for row in baseline['rows']] == [row['required_label'] for row in restored['rows']]
    assert restored['representation_elements_per_case'] > baseline['representation_elements_per_case']


def test_task_change_changes_labels_not_input_or_representation():
    direction = structure_probe(task='direction', transform='mean')
    presence = structure_probe(task='presence', transform='mean')
    assert [(r['input'], r['representation']) for r in direction['rows']] == [(r['input'], r['representation']) for r in presence['rows']]
    assert not direction['required_distinctions_preserved']
    assert presence['required_distinctions_preserved']


@pytest.mark.parametrize('kwargs', [{'task': 'accuracy'}, {'transform': 'trained_cnn'}])
def test_unknown_semantics_fail_instead_of_silently_changing_the_study(kwargs):
    with pytest.raises(ValueError):
        structure_probe(**kwargs)


def test_reference_exact_replay_through_capstone_dispatch():
    from mlsysim.labs.experiments.v1_16_experiments import _validate_reference_node
    reference = structure_probe(task='direction', transform='reversal_invariant')
    _validate_reference_node(reference, track_id='tinyml')
    reference['rows'][0]['required_label'] = 0
    with pytest.raises(ValueError, match='differs from exact replay'):
        _validate_reference_node(reference, track_id='tinyml')
