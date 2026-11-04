"""Evidence type and actual closure dominate campaign economic attractiveness."""
import json

import pytest

from mlsysim.core.units import ureg
from mlsysim.labs.experiments import v4_frontier as frontier


def values(choice):
    return {row['quantity']: row['value'] for row in frontier.study(18, 'D', choice)['rows']}


@pytest.mark.parametrize('wrong,right,gap', [
    ('more_nominal', 'matched_campaign', 'unmeasured_observable'),
    ('evaluation_wrong', 'evaluation_matched', 'evaluation_deficit'),
    ('theory_wrong', 'theory_matched', 'theoretical_deficit'),
    ('architecture_wrong', 'architecture_matched', 'architecture_mismatch'),
])
def test_campaign_reversal_holds_gap_and_economics_fixed(wrong, right, gap):
    a, b = values(wrong), values(right)
    assert a['gap class'] == b['gap class'] == gap
    assert a['missing evidence'] == b['missing evidence']
    assert a['restriction opportunity cost'] == b['restriction opportunity cost']
    assert not a['campaign targets missing evidence']
    assert b['campaign targets missing evidence']
    assert not a['plan supports next investigation']
    assert b['plan supports next investigation']
    for case in (a, b):
        assert not case['restriction may be retired now']
        assert not case['physical permission granted']
        assert 'unknown' in case['actual closure result']


def test_cost_accounting_and_payback_condition_use_demand_limited_loss():
    data = frontier._catalog()
    raw = frontier.calculate(next(c for c in data['choices'] if c['id'] == 'matched_campaign'), data)
    a = values('matched_campaign')
    assert a['lost units per operating hour'] == 2
    assert raw['total_campaign_cost'].to('USD').magnitude == pytest.approx(8*50+120)
    assert raw['restriction_opportunity_cost'].to('USD').magnitude == pytest.approx(2*5*100)
    assert ureg.Quantity(a['conditional successful closure break even']).to('hour').magnitude == pytest.approx((8*50+120)/(2*5))
    assert not a['restriction may be retired now']
    low = values('matched_low_demand')
    assert low['lost units per operating hour'] == 0
    assert frontier.calculate(next(c for c in data['choices'] if c['id'] == 'matched_low_demand'), data)['restriction_opportunity_cost'].magnitude == 0
    assert 'no finite payback' in low['conditional successful closure break even']
    assert low['campaign targets missing evidence']


@pytest.mark.parametrize('choice', ['matched_unknown_demand', 'matched_unknown_value'])
def test_missing_economic_inputs_remain_unknown_without_changing_evidence_obligation(choice):
    a = values(choice)
    assert 'unknown' in a['restriction opportunity cost']
    assert 'unknown' in a['conditional successful closure break even']
    assert a['campaign targets missing evidence']
    assert frontier.study(18, 'D', choice)['instrument'] is None
    assert not a['restriction may be retired now']


def test_funding_cannot_supply_a_result_or_grant_authority():
    good, premature = values('matched_campaign'), values('roadmap_release')
    assert good['total campaign cost'] == premature['total campaign cost']
    assert good['conditional successful closure break even'] == premature['conditional successful closure break even']
    assert premature['campaign targets missing evidence']
    assert not premature['plan supports next investigation']
    assert not premature['restriction retained pending actual results']
    assert not premature['restriction may be retired now']
    assert not premature['physical permission granted']


def test_all_declared_cases_are_json_safe_and_legacy_ids_survive():
    ids = {row['id'] for row in frontier.choices(18, 'D')}
    assert {'more_nominal', 'matched_campaign', 'roadmap_release'} <= ids
    for choice in ids:
        result = frontier.study(18, 'D', choice)
        json.dumps(result, allow_nan=False)
        assert 'does not command hardware' in result['limitation']
    with pytest.raises(ValueError):
        frontier.choices(18, 'C')
    with pytest.raises(ValueError):
        frontier.study(18, 'D', 'absent')


def test_plain_html_currency_does_not_expose_markdown_escapes():
    result = values('matched_campaign')
    assert result['total campaign cost'].startswith('$')
    assert result['lost contribution per operating hour'].startswith('$')
    assert result['lost contribution per operating hour'].endswith('/hour')
    assert all(r'\$' not in str(value) for value in result.values())
