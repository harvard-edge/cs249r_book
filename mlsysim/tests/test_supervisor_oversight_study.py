"""Causal staffing comparisons and explicit overload refusal."""
import json
import math

import pytest

from mlsysim.labs.experiments.v4_oversight import calculate, instrument


def test_source_case_reproduces_chapter_mean_waits():
    small = calculate({'queue_case': 'small'})
    reference = calculate({'queue_case': 'reference'})
    large = calculate({'queue_case': 'large'})
    assert small['offered_utilization'] == pytest.approx(.25)
    assert reference['offered_utilization'] == pytest.approx(.5)
    assert large['offered_utilization'] == pytest.approx(.8)
    assert small['mean_wait_before_assistance'].to('second').magnitude == pytest.approx(60)
    assert reference['mean_wait_before_assistance'].to('second').magnitude == pytest.approx(180)
    assert large['mean_wait_before_assistance'].to('second').magnitude == pytest.approx(720)
    for case in (small, reference, large):
        assert not case['p99_within_assistance_service_target']
        assert case['local_fallback_required']
        assert not case['physical_response_deadline_proven']


def test_ratio_changes_queue_without_changing_request_or_service_behavior():
    a, b = (calculate({'queue_case': name}) for name in ('small', 'large'))
    for key in ('help_requests_per_machine', 'mean_assistance_service_time', 'assistance_start_service_target'):
        assert a[key] == b[key]
    assert b['p99_wait_before_assistance'] > a['p99_wait_before_assistance']
    assert b['probability_queue_wait_exceeds_service_target'] > a['probability_queue_wait_exceeds_service_target']
    # Percentile inversion is an independent check of the displayed tail.
    mu = (1/a['mean_assistance_service_time']).to('Hz').magnitude
    lam = a['fleet_help_arrival_rate'].to('Hz').magnitude
    q99 = a['p99_wait_before_assistance'].to('second').magnitude
    assert a['offered_utilization'] * math.exp(-(mu-lam)*q99) == pytest.approx(.01)


def test_capacity_limit_refuses_finite_stationary_wait_and_plot():
    a = calculate({'queue_case': 'saturated'})
    assert not a['stationary_queue_exists']
    assert a['offered_utilization'] == pytest.approx(1)
    assert 'unbounded' in a['p99_wait_before_assistance']
    assert instrument(a) is None
    # Presentation layer can serialize explicit unboundedness without NaN.
    assert json.dumps(a, default=str, allow_nan=False)


def test_instrument_uses_only_finite_same_unit_quantities():
    plot = instrument(calculate({'queue_case': 'reference'}))
    assert plot['unit'] == 'second'
    assert all(math.isfinite(value) for value in plot['y'])
    assert len(plot['x']) == len(plot['y'])
    with pytest.raises(ValueError, match='Unknown supervisor queue'):
        calculate({'queue_case': 'unknown'})
