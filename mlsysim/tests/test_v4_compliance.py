"""Work conservation, independent force limits, and compliance tradeoffs."""
import json

import pytest

from mlsysim.core.units import ureg
from mlsysim.labs.experiments.v4_compliance import choices, study, energy_tank_step


def step(**overrides):
    inputs = dict(initial=2*ureg.J, reserve=1*ureg.J,
                  realized_damping_power=0*ureg.W, proposed_force=20*ureg.N,
                  speed=1*ureg.m/ureg.s, duration=1*ureg.s, force_cap=10*ureg.N)
    return energy_tank_step(**(inputs | overrides))


def test_spendable_work_clips_force_and_conserves_energy():
    result = step()
    assert result['admitted_force'].to('N').magnitude == pytest.approx(1)
    assert result['final_tank'].to('J').magnitude == pytest.approx(1)
    assert result['final_tank'] == result['initial_tank'] + result['damping_credit'] - result['active_work']


def test_at_reserve_only_realized_damping_can_fund_active_work():
    empty = step(initial=1*ureg.J)
    credited = step(initial=1*ureg.J, realized_damping_power=2*ureg.W)
    assert empty['admitted_force'].magnitude == 0
    assert credited['admitted_force'].to('N').magnitude == pytest.approx(2)
    assert credited['final_tank'].to('J').magnitude == pytest.approx(1)


def test_independent_force_cap_binds_with_abundant_energy():
    result = step(initial=100*ureg.J)
    assert result['admitted_force'].to('N').magnitude == pytest.approx(10)
    assert result['final_tank'].to('J').magnitude == pytest.approx(90)


def test_zero_speed_keeps_force_cap_and_does_not_divide_by_zero():
    result = step(initial=1*ureg.J, speed=0*ureg.m/ureg.s)
    assert result['energy_limited_force'] is None
    assert result['admitted_force'].to('N').magnitude == pytest.approx(10)
    assert result['active_work'].magnitude == 0


@pytest.mark.parametrize('overrides', [dict(initial=0*ureg.J), dict(reserve=0*ureg.J),
    dict(duration=0*ureg.s), dict(speed=-1*ureg.m/ureg.s),
    dict(realized_damping_power=-1*ureg.W), dict(initial=float('nan')*ureg.J)])
def test_invalid_or_unmodeled_inputs_are_refused(overrides):
    with pytest.raises(ValueError):
        step(**overrides)


def result_values(case):
    return {row['quantity']: row['value'] for row in study(case)['rows']}


def test_finite_trace_exhausts_spare_and_bounded_damping_preserves_reduced_contact():
    spare = study('tank_spare')['trace']
    reserve = study('tank_reserve')['trace']
    damping = study('tank_damping')['trace']
    assert ureg.Quantity(spare[0]['admitted_force']).magnitude > 0
    assert ureg.Quantity(spare[-1]['admitted_force']).magnitude == 0
    assert all(ureg.Quantity(row['admitted_force']).magnitude == 0 for row in reserve)
    assert ureg.Quantity(damping[-1]['admitted_force']).magnitude > 0
    for case in (spare, reserve, damping):
        for row in case:
            assert ureg.Quantity(row['final_tank']) >= ureg.Quantity(row['reserve'])


def test_series_spring_lowers_force_but_requires_more_compression():
    rigid = result_values('series_rigid')
    soft = result_values('series_soft')
    for quantity in ('effective stiffness', 'df dt', 'f at brake onset'):
        assert ureg.Quantity(soft[quantity]) < ureg.Quantity(rigid[quantity])
    assert ureg.Quantity(soft['compression at tripwire']) > ureg.Quantity(rigid['compression at tripwire'])
    assert soft['installed spring travel and task completion'] == 'unknown; not qualified'


def test_all_cases_are_exportable_and_preserve_qualification_boundary():
    for case in choices():
        result = study(case['id'])
        json.dumps(result)
        assert result['rows'] and result['instrument']
        assert result['provenance']['case'] == 'constructed teaching case; not measured'
        assert result['provenance']['catalog_sha256']
        assert 'does not command hardware' in result['limitation']
        assert result['source'].startswith('books/vol4/')
