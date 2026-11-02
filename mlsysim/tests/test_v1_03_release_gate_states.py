"""Release decisions require repair plus compatible, scoped evidence."""
import json

import pytest

from mlsysim.labs.experiments.v1_03_experiments import (
    TRACKS,
    evaluate_release_checks,
    release_gate_trial,
    replay,
    track_summary,
)


@pytest.mark.parametrize('track_id', TRACKS)
def test_inspection_and_repair_alone_never_pass(track_id):
    detected = evaluate_release_checks(track_id, 'combined')['detected_defects']
    inspected = release_gate_trial(track_id, 'combined')
    repaired = release_gate_trial(track_id, 'combined', detected, 'repair')
    assert not inspected['release_gate_passed']
    assert {gate['status'] for gate in inspected['gates']} == {'detected_failure'}
    assert not repaired['release_gate_passed']
    assert {gate['status'] for gate in repaired['gates']} == {'repaired_unverified'}
    assert inspected['artifact_version'] != repaired['artifact_version']
    assert repaired['passing_defects'] == ()


@pytest.mark.parametrize('track_id', TRACKS)
def test_stale_evidence_cannot_validate_changed_package(track_id):
    repairs = track_summary(track_id)['release_defects']
    stale = release_gate_trial(track_id, 'combined', repairs, 'retest', 1)
    assert stale['evidence_rejected']
    assert not stale['evidence_compatible']
    assert not stale['release_gate_passed']
    assert stale['passing_defects'] == ()
    fresh = release_gate_trial(track_id, 'combined', repairs, 'retest', 2)
    assert fresh['release_gate_passed']
    assert fresh['passing_defects'] == repairs
    assert 'does not establish target-system' in fresh['scope_notice']


@pytest.mark.parametrize('track_id', TRACKS)
def test_retest_clears_only_repaired_failures_within_scope(track_id):
    all_repairs = track_summary(track_id)['release_defects']
    offline_scope = set(evaluate_release_checks(track_id, 'offline')['detected_defects'])
    narrow = release_gate_trial(track_id, 'combined', all_repairs, 'retest', retest_plan='offline')
    assert set(narrow['passing_defects']) == offline_scope
    assert not narrow['release_gate_passed']
    assert all(gate['status'] == 'repaired_unverified' for gate in narrow['gates'] if gate['defect_id'] not in offline_scope)
    one = all_repairs[:1]
    partial = release_gate_trial(track_id, 'combined', one, 'retest')
    assert partial['passing_defects'] == one
    assert all(gate['status'] == 'detected_failure' for gate in partial['gates'] if gate['defect_id'] not in one)
    assert not partial['release_gate_passed']


@pytest.mark.parametrize('track_id', TRACKS)
def test_coverage_gaps_remain_and_retest_without_repair_does_not_pass(track_id):
    inspected = evaluate_release_checks(track_id, 'offline')
    repaired = release_gate_trial(track_id, 'offline', inspected['detected_defects'], 'retest')
    assert repaired['missing_evidence'] == inspected['escaped_defects']
    assert not repaired['release_gate_passed']
    unrepaired = release_gate_trial(track_id, 'combined', phase='retest')
    assert not unrepaired['release_gate_passed']
    assert unrepaired['passing_defects'] == ()


@pytest.mark.parametrize('track_id', TRACKS)
@pytest.mark.parametrize('phase', ['inspect', 'repair', 'retest'])
def test_every_state_replays_exactly_after_json_roundtrip(track_id, phase):
    repairs = () if phase == 'inspect' else track_summary(track_id)['release_defects']
    result = release_gate_trial(track_id, 'combined', repairs, phase)
    snapshot = json.loads(json.dumps(result, allow_nan=False))
    rerun = replay('v1_03_experiments.release_gate_trial', snapshot['inputs'])
    assert json.loads(json.dumps(rerun, allow_nan=False)) == snapshot


@pytest.mark.parametrize('kwargs', [
    {'phase': 'unknown'}, {'repair_defects': 'unknown'},
    {'repair_defects': ['unknown'], 'phase': 'repair'},
    {'evidence_version': True}, {'evidence_version': 1.5},
    {'evidence_version': 0}, {'evidence_version': 2},
    {'retest_plan': 'unknown'},
])
def test_unsupported_actions_and_invalid_versions_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        release_gate_trial('tinyml', 'offline', **kwargs)


def test_inspect_cannot_silently_repair_or_repair_an_unobserved_finding():
    detected = evaluate_release_checks('tinyml', 'offline')['detected_defects']
    missed = evaluate_release_checks('tinyml', 'offline')['escaped_defects']
    with pytest.raises(ValueError):
        release_gate_trial('tinyml', 'offline', detected)
    with pytest.raises(ValueError):
        release_gate_trial('tinyml', 'offline', missed, 'repair')
