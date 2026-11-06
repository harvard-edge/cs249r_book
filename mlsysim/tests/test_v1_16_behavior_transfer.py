"""Cause-specific recovery must not silently renew unrelated evidence."""
from copy import deepcopy
import json

import pytest

from mlsysim.labs.experiments.v1_16_experiments import (
    behavioral_transfer_trial, replay, replay_capture,
)


def trial(**kwargs):
    return behavioral_transfer_trial('tinyml', 'mobile', **kwargs)


def test_rollback_repairs_release_but_not_population():
    release = trial(cause='release_regression', response='rollback', phase='retest')
    population = trial(cause='population_shift', response='rollback', phase='retest')
    assert release['before']['correct'] < release['source']['correct']
    assert release['after']['correct'] == release['source']['correct']
    assert release['state'] == 'compatible_retest_pass'
    assert population['after'] == population['before']
    assert population['state'] == 'observed_failure'
    assert release['physical_reference'] == population['physical_reference']


def test_rule_repair_requires_retest_and_new_execution_evidence():
    proposal = trial(response='signal_rule', phase='propose')
    retest = trial(response='signal_rule', phase='retest')
    assert proposal['after'] is None
    assert proposal['state'] == 'proposed_unverified'
    assert retest['after']['correct'] == retest['after']['cases']
    assert retest['state'] == 'compatible_retest_pass'
    assert not retest['candidate_execution_evidence_compatible']
    assert not retest['native_target_quality_available']
    assert not retest['joint_release_supported']
    assert retest['physical_reference']['quality_pct'] is None


def test_source_retest_cannot_certify_new_population_or_rule():
    stale = trial(response='signal_rule', phase='retest', evidence_bundle='source')
    assert stale['after'] is None
    assert stale['state'] == 'incompatible_evidence'
    assert stale['candidate_bundle'] != stale['supplied_bundle']
    restored = trial(cause='release_regression', response='rollback', phase='retest', evidence_bundle='source')
    assert restored['retest_evidence_compatible']
    assert restored['state'] == 'compatible_retest_pass'


def test_containment_counts_work_without_inventing_perfect_quality():
    held = trial(response='contain', phase='retest')
    assert held['state'] == 'contained_for_review'
    assert held['after']['review_cases'] == held['after']['cases']
    assert held['after']['automated_cases'] == 0
    assert held['after']['correctness_pct'] is None
    assert held['after']['gate_passes'] is None
    assert not held['joint_release_supported']


@pytest.mark.parametrize('track,target', [('tinyml','mobile'),('mobile','edge'),('edge','cloud'),('cloud','tinyml')])
@pytest.mark.parametrize('cause', ['release_regression','population_shift'])
def test_exact_json_replay_and_tampering(track, target, cause):
    base = behavioral_transfer_trial(track, target, cause=cause, response='keep', phase='retest')
    result = behavioral_transfer_trial(track, target, cause=cause, response='signal_rule', phase='retest')
    capture = json.loads(json.dumps({'model_key':'v1_16.behavior_transfer.v1','track':track,
        'baseline':base,'result':result,'alternatives':[]}))
    expected = replay_capture('v1_16_experiments', capture)
    assert json.loads(json.dumps(expected['result'])) == capture['result']
    assert replay('v1_16.behavior_transfer.v1', result['inputs']) == result
    bad = deepcopy(capture)
    bad['result']['after']['correct'] -= 1
    with pytest.raises(ValueError, match='differs from exact replay'):
        replay_capture('v1_16_experiments', bad)


@pytest.mark.parametrize('field,value', [('cause','unknown'),('response','magic'),('phase','pass'),('evidence_bundle','guess')])
def test_unknown_conditions_are_rejected(field,value):
    with pytest.raises(ValueError):
        trial(**{field:value})
