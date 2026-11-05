"""Sensing and estimation tradeoffs must follow the supplied arithmetic."""
import json
import math

import pytest

from mlsysim.core.units import ureg
from mlsysim.embodied.registry import Embodied
from mlsysim.labs.experiments import v4_estimation as studies


def values(chapter, part, choice):
    return {r['quantity']: r['value'] for r in studies.study(chapter, part, choice)['rows']}


def quantity(value, unit):
    return ureg.Quantity(value).to(unit).magnitude


@pytest.mark.parametrize('chapter,part', [(8, 'B'), (8, 'C'), (9, 'A')])
def test_all_evidence_is_serializable_explicitly_constructed_and_bounded(chapter, part):
    for choice in studies.choices(chapter, part):
        result = studies.study(chapter, part, choice['id'])
        json.dumps(result)
        assert result['provenance']['case'] == 'constructed teaching case; not measured'
        assert 'does not command hardware' in result['limitation']
        assert result['source'].startswith('books/vol4/')
        assert result['conditions'] and result['trace'] and result['feedback']
        assert len(result['instrument']['x']) == len(result['instrument']['y'])
        assert len(result['provenance']['catalog_sha256']) == 64


def test_age_baseline_is_canonical_and_only_named_stage_changes():
    baseline = studies.study(8, 'C', 'reference')
    faster = studies.study(8, 'C', 'faster_path')
    delayed = studies.study(8, 'C', 'queue_delay')
    site = Embodied.Scenario.WarehouseAisle
    assert sum(baseline['instrument']['y']) == pytest.approx(site.perception_age.to('ms').magnitude)
    assert sum(delayed['instrument']['y']) == pytest.approx(2 * sum(baseline['instrument']['y']))
    for x, a, b in zip(baseline['instrument']['x'], baseline['instrument']['y'], faster['instrument']['y']):
        assert b == pytest.approx(a / 2 if x == 'backbone' else a)
    a = values(8, 'C', 'reference'); b = values(8, 'C', 'faster_path')
    assert a['empirical absolute residual quantile'] == b['empirical absolute residual quantile']
    assert quantity(b['unmodeled displacement'], 'm') < quantity(a['unmodeled displacement'], 'm')


def test_residual_order_statistic_is_not_a_population_bound_or_transferable_condition():
    near = values(8, 'C', 'reference')
    mismatch = values(8, 'C', 'far_unmatched')
    matched = values(8, 'C', 'far_matched')
    missing = values(8, 'C', 'missing_residuals')
    assert studies.empirical_absolute_quantile([-4, 1, -2, 3], .75) == 3
    assert near['residual provenance matches condition'] is True
    assert mismatch['residual provenance matches condition'] is False
    assert 'unknown' in mismatch['sensing clearance allocation']
    assert 'unknown' in missing['empirical absolute residual quantile']
    assert quantity(matched['usable classroom sensing inset'], 'm') > quantity(near['usable classroom sensing inset'], 'm')
    assert near['observation age'] == matched['observation age']
    assert 'not established' in matched['population error guarantee']
    allocation = quantity(near['sensing clearance allocation'], 'm')
    assert allocation == pytest.approx(quantity(near['unmodeled displacement'], 'm') + quantity(near['usable classroom sensing inset'], 'm'))


@pytest.mark.parametrize('residuals,q', [([], .9), ([1], 0), ([1], 2), ([math.nan], .9)])
def test_invalid_residual_claim_refuses(residuals, q):
    with pytest.raises(ValueError):
        studies.empirical_absolute_quantile(residuals, q)


def test_capture_replay_corrects_epoch_but_arrival_update_can_report_smaller_covariance():
    replay = values(9, 'A', 'correct_replay')
    arrival = values(9, 'A', 'arrival_update')
    assert quantity(arrival['posterior present variance'], 'm^2') < quantity(replay['posterior present variance'], 'm^2')
    assert abs(quantity(arrival['present estimation error'], 'm')) > abs(quantity(replay['present estimation error'], 'm'))
    result = studies.study(9, 'A', 'correct_replay')
    p = result['inputs']
    updated = studies.scalar_update(p['capture_prior_range_m'], p['capture_prior_variance_m2'], p['measurement_range_m'], p['measurement_variance_m2'])
    assert quantity(replay['posterior present range'], 'm') == pytest.approx(updated['mean'] + p['known_velocity_m_per_s'] * p['delay_s'])
    assert quantity(replay['posterior present variance'], 'm^2') == pytest.approx(updated['variance'] + p['process_variance_rate_m2_per_s'] * p['delay_s'])


def test_understating_noise_shrinks_covariance_without_changing_estimate():
    normal = values(9, 'A', 'correct_replay'); small = values(9, 'A', 'shrink_covariance')
    assert normal['posterior present range'] == small['posterior present range']
    assert quantity(small['posterior present variance'], 'm^2') < quantity(normal['posterior present variance'], 'm^2')
    assert small['NEES (single scalar sample)'] > normal['NEES (single scalar sample)']
    assert small['NIS (single scalar innovation)'] > normal['NIS (single scalar innovation)']


def test_shared_bias_hides_from_innovations_and_missing_truth_prevents_nees():
    normal = values(9, 'A', 'correct_replay'); biased = values(9, 'A', 'common_bias')
    assert biased['NIS (single scalar innovation)'] == pytest.approx(normal['NIS (single scalar innovation)'])
    assert biased['NEES (single scalar sample)'] > normal['NEES (single scalar sample)']
    missing = values(9, 'A', 'no_truth')
    assert 'unknown' in missing['NEES (single scalar sample)']
    assert studies.study(9, 'A', 'no_truth')['instrument']['x'] == ['posterior standard deviation']
    assert all('truth' not in item or item['truth'] is None for item in studies.study(9, 'A', 'no_truth')['trace'])


def test_expired_history_refuses_update_and_preserves_predicted_covariance():
    result = values(9, 'A', 'history_expired')
    assert 'refused' in result['update epoch']
    assert result['posterior present variance'] == result['predicted present variance']
    assert result['posterior present range'] == result['predicted present range']
    assert 'not computed' in result['NIS (single scalar innovation)']


def test_geometry_uses_capture_transform_and_error_grows_with_range():
    capture = values(8, 'B', 'capture_chain'); arrival = values(8, 'B', 'arrival_transform')
    assert quantity(capture['nominal geometry residual'], 'm') == 0
    assert quantity(arrival['nominal geometry residual'], 'm') > 0
    near = values(8, 'B', 'stale_calibration'); far = values(8, 'B', 'far_calibration')
    assert quantity(far['nominal geometry residual'], 'm') == pytest.approx(2 * quantity(near['nominal geometry residual'], 'm'))
    clock = values(8, 'B', 'foreign_clock')
    assert quantity(clock['nominal geometry residual'], 'm') == 0
    assert 'unknown' in clock['clock-induced spatial bound']


@pytest.mark.parametrize('args', [(0, 0, 1, 1), (0, 1, 1, -1), (math.inf, 1, 1, 1)])
def test_invalid_covariance_refuses(args):
    with pytest.raises(ValueError): studies.scalar_update(*args)


def test_invalid_identity_and_mutable_outputs_do_not_leak():
    with pytest.raises(ValueError): studies.study(9, 'A', 'bogus')
    with pytest.raises(ValueError): studies.choices(8, 'B', 'cloud')
    with pytest.raises(ValueError): studies.choices(9, 'D')
    first = studies.study(9, 'A', 'correct_replay')
    first['inputs']['capture_prior_range_m'] = -999
    assert studies.study(9, 'A', 'correct_replay')['inputs']['capture_prior_range_m'] != -999
