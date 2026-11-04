"""Sampling and timing are separate uncertainty/exposure mechanisms."""
import pytest
from mlsysim.labs.experiments.v4_subgroups import study, wilson_interval


def test_equal_rates_do_not_imply_equal_precision():
    thin = study(17, 'C', 'equal_thin')['trace']
    more = study(17, 'C', 'equal_supported')['trace']
    assert thin[0]['deadline_detection_rate'] == thin[1]['deadline_detection_rate']
    assert thin[1]['wilson_width'] > thin[0]['wilson_width']
    assert more[1]['wilson_width'] == more[0]['wilson_width']
    assert more[1]['additional_blind_travel_m'] == thin[1]['additional_blind_travel_m']


def test_slowing_changes_distance_without_manufacturing_detection_quality():
    normal = study(17, 'C', 'best_group')['trace']
    slow = study(17, 'C', 'bounded_scope')['trace']
    for a, b in zip(normal, slow):
        assert a['deadline_detection_rate'] == b['deadline_detection_rate']
        assert a['wilson_lower'] == b['wilson_lower']
        assert b['reference_clearance_reserve_m'] > a['reference_clearance_reserve_m']
    assert normal[1]['additional_blind_travel_m'] > slow[1]['additional_blind_travel_m']


def test_zero_and_all_success_samples_retain_uncertainty():
    assert wilson_interval(0, 10)[1] > 0
    assert wilson_interval(10, 10)[0] < 1
    with pytest.raises(ValueError): wilson_interval(1, 0)
    with pytest.raises(ValueError): wilson_interval(11, 10)
    with pytest.raises(ValueError): wilson_interval(True, 10)


def test_later_detection_can_spend_a_reference_stop_budget():
    normal = study(17, 'C', 'best_group')['trace']
    slow = study(17, 'C', 'bounded_scope')['trace']
    assert normal[0]['conditional_reference_budget_met']
    assert not normal[1]['conditional_reference_budget_met']
    assert normal[1]['conditional_speed_at_allocated_boundary_m_per_s'] > 0
    assert normal[1]['conditional_rigid_body_energy_at_allocated_boundary_J'] > 0
    assert slow[1]['conditional_reference_budget_met']
    assert slow[1]['conditional_speed_at_allocated_boundary_m_per_s'] == 0
