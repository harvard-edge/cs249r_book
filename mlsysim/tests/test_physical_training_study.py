"""Action representation and sequential decoder work are distinct constraints."""
import json

import pytest

from mlsysim.core.units import ureg
from mlsysim.labs.experiments import v4_training as training


def values(choice):
    return {row['quantity']: row['value'] for row in training.study(6, 'A', choice)['rows']}


def test_squared_error_prefers_mean_that_intersects_constructed_obstacle():
    mean = values('mean_path')
    mode = values('bounded_multimodal')
    assert not mean['matches a supplied action mode']
    assert not mean['cross section outside obstacle']
    assert mode['matches a supplied action mode']
    assert mode['cross section outside obstacle']
    assert ureg.Quantity(mean['expected squared action error']) < ureg.Quantity(mode['expected squared action error'])
    assert mean['strictly before renewal']


def test_dependent_work_and_delay_can_exhaust_renewal_without_changing_action():
    quick = values('bounded_multimodal')
    nominal = values('sixteen_nominal')
    late = values('rich_too_slow')
    more = values('more_evaluations')
    assert quick['selected lateral action'] == nominal['selected lateral action'] == late['selected lateral action'] == more['selected lateral action']
    assert quick['encoder time once'] == nominal['encoder time once']
    assert quick['time per dependent evaluation'] == nominal['time per dependent evaluation']
    assert quick['strictly before renewal'] and nominal['strictly before renewal']
    assert not late['strictly before renewal'] and not more['strictly before renewal']
    assert ureg.Quantity(nominal['complete proposal latency']).to('ms').magnitude == pytest.approx(48)
    assert ureg.Quantity(quick['complete proposal latency']).to('ms').magnitude == pytest.approx(18)
    assert ureg.Quantity(late['proposal renewal slack']).to('ms').magnitude == pytest.approx(-3)
    assert 'unknown' in more['quality gain from more evaluations']


def test_opposite_supplied_modes_have_same_cost_without_mean_averaging():
    left, right = values('bounded_multimodal'), values('right_mode')
    assert ureg.Quantity(left['selected lateral action']) == -ureg.Quantity(right['selected lateral action'])
    assert left['expected squared action error'] == right['expected squared action error']
    assert left['complete proposal latency'] == right['complete proposal latency']
    assert not right['physical trajectory qualified']


def test_supported_identities_are_bounded_json_safe_and_preserve_legacy_ids():
    ids = {row['id'] for row in training.choices(6, 'A')}
    assert {'mean_path', 'rich_too_slow', 'bounded_multimodal'} <= ids
    for choice in ids:
        result = training.study(6, 'A', choice)
        json.dumps(result, allow_nan=False)
        assert 'does not command hardware' in result['limitation']
        assert result['trace'] and result['conditions'] and result['instrument']
    with pytest.raises(ValueError):
        training.study(6, 'A', 'missing')
    with pytest.raises(ValueError):
        training.choices(6, 'B')
    with pytest.raises(ValueError):
        training.choices(6, 'A', 'cloud')
