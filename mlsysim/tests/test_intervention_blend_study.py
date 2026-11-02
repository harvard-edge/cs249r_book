"""Causal and unit contracts for the finite intervention comparison."""
import json

import pytest

from mlsysim.core.units import ureg
from mlsysim.labs.experiments import v4_intervention as study


def quantities(choice):
    return {r['quantity']: r['value'] for r in study.study(14, 'C', choice)['rows']}


def q(value, unit):
    return ureg.Quantity(value).to(unit).magnitude


def test_longer_smooth_blend_reverses_distance_verdict_without_rate_failure():
    short = quantities('matched_short_blend')
    long = quantities('long_blend')
    assert short['within command rate limit'] and long['within command rate limit']
    assert q(long['peak base command jerk'], 'm/s^3') < q(short['peak base command jerk'], 'm/s^3')
    assert q(long['blend travel'], 'm') > q(short['blend travel'], 'm')
    assert short['all sampled resident stops fit'] is True
    assert long['all sampled resident stops fit'] is False
    assert long['remaining motion clearance at commit'] == short['remaining motion clearance at commit']


def test_shorter_window_is_not_made_admissible_by_shorter_distance():
    short = quantities('too_short')
    reference = quantities('matched_short_blend')
    assert short['within command rate limit'] is False
    assert q(short['blend travel'], 'm') < q(reference['blend travel'], 'm')
    abrupt = quantities('switch_now')
    assert abrupt['peak base command jerk'] == 'unbounded ideal step'
    assert abrupt['within command rate limit'] is False


def test_exact_trace_matches_boundary_and_integrals_not_euler_steps():
    result = study.study(14, 'C', 'matched_short_blend')
    rows = quantities('matched_short_blend')
    trace = result['tick_trace']
    first, last = trace[0], trace[-1]
    assert q(first['modeled travel'], 'm') == 0
    assert q(first['acceleration command'], 'm/s^2') == 0
    assert q(first['commanded jerk'], 'm/s^3') == 0
    assert q(last['commanded jerk'], 'm/s^3') == 0
    assert last['modeled travel'] == rows['blend travel']
    assert last['modeled speed'] == rows['speed at blend end']
    assert len(result['trace']) <= 22
    assert result['trace'][0] == trace[0] and result['trace'][-1] == trace[-1]
    assert [r['tick'] for r in trace] == list(range(len(trace)))
    assert q(rows['minimum tick stopping reserve'], 'm') == pytest.approx(min(q(r['remaining resident-stop reserve'], 'm') for r in trace), abs=2e-6)
    assert 'partial lower bound' in result['limitation']


def test_controller_matching_is_independent_of_output_interpolation():
    matched = study.study(14, 'C', 'matched_short_blend')
    unmatched = study.study(14, 'C', 'unmatched_controller')
    assert matched['trace'] == unmatched['trace']
    assert quantities('unmatched_controller')['controller state matched'] is False
    assert 'actual post-blend command is unknown' in unmatched['feedback']


def test_less_clearance_changes_reserve_not_command_trace():
    matched = study.study(14, 'C', 'matched_short_blend')
    tight = study.study(14, 'C', 'tight_clearance')
    assert quantities('tight_clearance')['all sampled resident stops fit'] is False
    for a, b in zip(matched['trace'], tight['trace']):
        assert a['acceleration command'] == b['acceleration command']
        assert a['modeled travel'] == b['modeled travel']
        assert a['remaining resident-stop reserve'] != b['remaining resident-stop reserve']


def test_arm_torque_api_keeps_its_own_units_and_rounds_to_ticks():
    rows = quantities('matched_short_blend')
    assert q(rows['arm minimum torque blend window'], 'ms') == pytest.approx(37.5)
    assert q(rows['arm tick-rounded window'], 'ms') >= q(rows['arm minimum torque blend window'], 'ms')
    assert rows['arm tick-rounded window'] != rows['chosen base blend window']


def test_export_contract_and_identity_validation():
    assert study.handles(14, 'C')
    assert not study.handles(14, 'A')
    for case in study.choices(14, 'C'):
        result = study.study(14, 'C', case['id'])
        json.dumps(result)
        assert result['rows'] and result['trace'] and result['conditions']
        assert 'does not command hardware' in result['limitation']
        assert len(result['instrument']['x']) == len(result['instrument']['y'])
    with pytest.raises(ValueError): study.choices(14, 'C', 'cloud')
    with pytest.raises(ValueError): study.study(14, 'B', 'switch_now')
    with pytest.raises(ValueError): study.study(14, 'C', 'unknown')
    result = study.study(14, 'C', 'matched_short_blend')
    result['inputs']['window_factor'] = 9
    assert study.study(14, 'C', 'matched_short_blend')['inputs']['window_factor'] == 1
