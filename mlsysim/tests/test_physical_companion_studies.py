"""Causal contracts and evidence limits of the physical-AI browser companion."""
import json

import pytest

from mlsysim.core.units import ureg
from mlsysim.labs.experiments import v4_studies as studies


def values(result):
    return {row['quantity']: row['value'] for row in result['rows']}


def magnitude(value, unit):
    return ureg.Quantity(value).to(unit).magnitude


@pytest.mark.parametrize('chapter', range(19))
@pytest.mark.parametrize('part', 'ABCD')
def test_every_supported_choice_has_bounded_exportable_evidence(chapter, part):
    available = studies.choices(chapter, part)
    assert len(available) >= 2
    for choice in available:
        result = studies.study(chapter, part, choice['id'])
        assert result['rows'] and result['feedback'] and result['brief'] and result['held_fixed']
        assert 'does not command hardware' in result['limitation']
        assert result['source'].startswith('books/vol4/')
        assert result['provenance']['case'] == 'constructed teaching case; not measured'
        json.dumps(result)  # browser/discussion exports must not leak Pint objects
        instrument = result['instrument']
        if instrument:
            assert len(instrument['x']) == len(instrument['y'])
            assert instrument['unit'] and instrument['scope']


def test_shorter_delay_does_not_remove_braking_distance():
    reference = values(studies.study(1, 'B', 'reference'))
    faster = values(studies.study(1, 'B', 'shorter_delay'))
    assert reference['braking distance'] == faster['braking distance']
    assert magnitude(faster['reaction distance'], 'm') < magnitude(reference['reaction distance'], 'm')


def test_more_gearing_multiplies_inertia_faster_than_ideal_torque():
    a = values(studies.study(2, 'B', 'reference'))
    b = values(studies.study(2, 'B', 'more_reduction'))
    inertia_ratio = magnitude(b['reflected rotor inertia'], 'kg*m^2') / magnitude(a['reflected rotor inertia'], 'kg*m^2')
    torque_ratio = magnitude(b['ideal output torque'], 'N*m') / magnitude(a['ideal output torque'], 'N*m')
    assert inertia_ratio == pytest.approx(torque_ratio**2)


def test_chunk_supply_does_not_renew_information():
    a = values(studies.study(3, 'B', 'reference'))
    b = values(studies.study(3, 'B', 'longer_chunk'))
    assert magnitude(b['chunk span'], 'ms') > magnitude(a['chunk span'], 'ms')
    for name in ('renewal period', 'open loop span', 'untracked evidence horizon', 'tracked evidence horizon'):
        assert a[name] == b[name]
    refused = studies.study(3, 'B', 'too_short')
    assert 'model refusal' in values(refused)
    assert refused['instrument'] is None


def test_clean_trials_and_hazard_exposure_remain_separate():
    a = values(studies.study(7, 'D', 'reference'))
    b = values(studies.study(7, 'D', 'more_trials'))
    assert b['success probability lower bound'] > a['success probability lower bound']
    assert b['separate poisson zero event exposure'] == a['separate poisson zero event exposure']


def test_information_age_is_only_capture_to_dispatch():
    from mlsysim.embodied.registry import Embodied
    a = values(studies.study(8, 'C', 'reference'))
    site = Embodied.Scenario.WarehouseAisle
    assert magnitude(a['observation age'], 'ms') == pytest.approx(site.perception_age.to('ms').magnitude)
    assert magnitude(a['observation age'], 'ms') < site.tau_delay.to('ms').magnitude


def test_gentler_jerk_costs_more_distance_and_time():
    a = values(studies.study(11, 'C', 'c2'))
    b = values(studies.study(11, 'C', 'gentler_stop'))
    assert magnitude(b['braking distance'], 'm') > magnitude(a['braking distance'], 'm')
    assert magnitude(b['braking time'], 's') > magnitude(a['braking time'], 's')


def test_fifo_backlog_does_not_change_declared_priority_bypass():
    a = values(studies.study(13, 'A', 'reference'))
    b = values(studies.study(13, 'A', 'larger_backlog'))
    assert magnitude(b['total fifo'], 'us') > magnitude(a['total fifo'], 'us')
    assert b['total priority bypass'] == a['total priority bypass']


@pytest.mark.parametrize('chapter,part,choice', [(16,'C','other_parent'), (14,'B','two_writers'), (17,'D','keep_everything'), (18,'C','paper_limit')])
def test_filled_conflicting_record_is_not_accepted(chapter, part, choice):
    conflict = studies.study(chapter, part, choice)
    assert all(row['supplied_evidence'] != 'not supplied' for row in conflict['rows'])
    assert any(row['policy_status'] == 'does not support stated claim' for row in conflict['rows'])
    assert 'does not establish the stated claim' in conflict['feedback']


def test_incomplete_release_refuses_without_granting_permission():
    a = studies.study(16, 'B', 'actual_machine')
    assert 'refused' in a['feedback'].lower()
    b = studies.study(16, 'B', 'stipulated_cond')
    assert 'classroom' in str(b['rows'])
    assert 'physical truth remains unverified' in b['feedback']


def test_projection_retains_an_unchanged_feasible_command():
    a = values(studies.study(12, 'B', 'feasible_proposal'))
    assert a['was modified'] is False
    assert a['correction norm'] == 0
    assert 'not a proof' in studies.study(12, 'B', 'reference')['limitation']


def test_callers_cannot_mutate_catalog_and_invalid_identity_fails():
    first = studies.study(16, 'C', 'matched')
    first['rows'][0]['supplied_evidence'] = 'mutated'
    assert studies.study(16, 'C', 'matched')['rows'][0]['supplied_evidence'] != 'mutated'
    with pytest.raises(ValueError): studies.study(1, 'A', 'does-not-exist')
    with pytest.raises(ValueError): studies.choices(99, 'A')
    with pytest.raises(ValueError): studies.choices(1, 'A', 'cloud')


def test_lease_budget_reserves_every_nonlease_prebrake_stage():
    from mlsysim.embodied.registry import Embodied
    site = Embodied.Scenario.WarehouseAisle
    actual = studies._calculate('lease', {'lease_fraction':1})
    reserve = site.perception_age + site.t_tick + site.t_bus + site.t_brake_onset
    assert actual['nonlease_delay_reserved'] == reserve
    assert actual['remaining_lease_budget'] == actual['t_total_delay_max'] - reserve
    assert 't_lease_max' not in actual
    near = ((actual['remaining_lease_budget'] + site.perception_age/2)/site.t_lease).to('').magnitude
    assert studies._calculate('lease', {'lease_fraction':near})['within_ideal_budget'] is False


def test_collection_ledger_keeps_failed_attempt_costs_and_coverage_gaps():
    nominal=studies.study(5,'C','nominal')
    recovery=studies.study(5,'C','recovery')
    a=values(nominal); b=values(recovery)
    assert a['attempted episodes']==b['attempted episodes']
    assert a['retained episodes']>b['retained episodes']
    assert len(a['supported cells'])<len(b['supported cells'])
    assert magnitude(b['cell time'],'minute')>magnitude(a['cell time'],'minute')
    assert b['mechanical cycles']>a['mechanical cycles']
    assert any(row['retained'] is False for row in recovery['trace'])
    assert 'partial_occlusion' in b['missing declared cells']


def test_tentative_occupancy_reaches_consumer_before_identity():
    bad=values(studies.study(9,'B','confirmed_only'))
    immediate=values(studies.study(9,'B','tentative_occupancy'))
    hold=values(studies.study(9,'B','unknown_refusal'))
    assert bad['unsupported clear events']>0
    assert immediate['unsupported clear events']==0
    assert hold['unsupported clear events']==0
    assert hold['events with unreported obstacle']>immediate['events with unreported obstacle']


def test_republication_and_prediction_do_not_refresh_evidence():
    original=values(studies.study(9,'C','old_capture'))
    for choice in ('republish','predict'):
        newer=values(studies.study(9,'C',choice))
        assert newer['original evidence age']==original['original evidence age']
        assert newer['remaining validity']==original['remaining validity']
    captured=values(studies.study(9,'C','new_capture'))
    assert magnitude(captured['remaining validity'],'ms')>magnitude(original['remaining validity'],'ms')


def test_c2_seam_removes_only_the_reflected_rotor_jump():
    step=values(studies.study(11,'C','position_only'))
    smooth=values(studies.study(11,'C','c2'))
    assert magnitude(step['reflected rotor torque jump'],'N*m')>0
    assert magnitude(smooth['reflected rotor torque jump'],'N*m')==0
    assert step['braking distance']==smooth['braking distance']


def test_latch_and_human_contact_cases_do_not_cross_qualify():
    a=values(studies.study(12,'C','guarded'))
    faster_latch=values(studies.study(12,'C','fast_latch'))
    faster_hand=values(studies.study(12,'C','fast_handover'))
    assert magnitude(faster_latch['f at brake onset'],'N')>magnitude(a['f at brake onset'],'N')
    assert faster_latch['transient hand force']==a['transient hand force']
    assert faster_hand['f at brake onset']==a['f at brake onset']
    assert magnitude(faster_hand['transient hand force'],'N')>magnitude(a['transient hand force'],'N')


def test_body_freshness_and_stopping_are_independent_requirements():
    a=values(studies.study(2,'A','reference'))
    b=values(studies.study(2,'A','fresh_path'))
    assert a['required distance']==b['required distance']
    assert magnitude(b['age travel'],'m')<magnitude(a['age travel'],'m')
    assert 'unknown' in b['torque thermal installed qualification']


def test_full_envelope_charges_dispatch_to_admission_exactly_once():
    from mlsysim.embodied.registry import Embodied
    site=Embodied.Scenario.WarehouseAisle
    result=studies._calculate('envelope',{'speed_fraction':1,'jerk_fraction':1})
    assert result['prebrake_travel'].to('m').magnitude == pytest.approx((site.v_aisle*(site.tau_delay+site.t_dispatch_to_admission)).to('m').magnitude)
    assert result['dispatch_to_admission'] == site.t_dispatch_to_admission
    assert result['required_distance'] == result['jerk_limited_braking']+result['prebrake_travel']+site.fixed_overhead


def test_target_tracking_keeps_replacement_and_lease_as_separate_clocks():
    a=studies._calculate('target',{'drift_fraction':1})
    b=studies._calculate('target',{'drift_fraction':1,'tracked':True})
    assert a['target_evidence_validity']==a['untracked_target_horizon']
    assert b['target_evidence_validity']==b['tracked_target_horizon']
    assert a['capture_to_admission_allocation']==b['capture_to_admission_allocation']
    assert a['requested_execution_lease']==b['requested_execution_lease']
    assert b['target_validity_remaining_at_admission']<b['target_evidence_validity']


def test_quantity_display_preserves_small_causal_differences():
    a=values(studies.study(8,'C','reference'))
    b=values(studies.study(8,'C','queue_delay'))
    assert a['unmodeled displacement'] != b['unmodeled displacement']
    body=values(studies.study(2,'A','reference'))
    assert body['freshness spatial allowance'] != body['age travel']
    assert magnitude(body['freshness spatial allowance'],'m') < magnitude(body['age travel'],'m')


def test_vla_precision_and_input_length_affect_different_costs():
    a=studies._calculate('vla',{'precision':'reference','input_fraction':1})
    wider=studies._calculate('vla',{'precision':'fp16','input_fraction':1})
    longer=studies._calculate('vla',{'precision':'reference','input_fraction':2})
    assert wider['stored_weights']>a['stored_weights']
    assert wider['t_weight_sweep']>a['t_weight_sweep']
    assert wider['prefill_flops']==a['prefill_flops']
    assert longer['prefill_flops']>a['prefill_flops']
    assert longer['t_decode']==a['t_decode']
    assert longer['stored_weights']==a['stored_weights']


def test_duty_history_and_rail_transient_remain_separate():
    base=studies._calculate('body_duty',{'current_fraction':1,'hot_start':False,'off_time':'30 s'})
    hot=studies._calculate('body_duty',{'current_fraction':1,'hot_start':True,'off_time':'30 s'})
    short=studies._calculate('body_duty',{'current_fraction':1,'hot_start':False,'off_time':'3 s'})
    stagger=studies._calculate('body_duty',{'current_fraction':.5,'hot_start':False,'off_time':'30 s'})
    assert hot['peak_winding_temperature']>base['peak_winding_temperature']
    assert short['peak_winding_temperature']>base['peak_winding_temperature']
    assert stagger['peak_winding_temperature']==base['peak_winding_temperature']
    assert hot['resistive_droop']==base['resistive_droop']
    assert stagger['resistive_droop']<base['resistive_droop']
    assert len(base['modeled_thermal_trace'])>0


@pytest.mark.parametrize('case,verdict',[('undelivered','INVALID'),('late_stop','FAIL'),('valid_pass','PASS'),('lost_witness','UNOBSERVABLE')])
def test_fault_trial_oracle_keeps_all_four_verdicts(case,verdict):
    result=studies.study(15,'A',case)
    row=next(row for row in result['rows'] if row['requirement']=='computed four-state oracle')
    assert row['supplied_evidence']==verdict
    assert 'no physical permit' in row['policy_status']


def test_thermal_ambient_uses_offset_safe_quantity_construction(monkeypatch):
    original=studies._catalog
    def offset_fixture():
        data=original()
        data['thermal_installation']['ambient']='25 degree_Celsius'
        return data
    monkeypatch.setattr(studies,'_catalog',offset_fixture)
    result=studies._calculate('body_duty',{'current_fraction':1,'hot_start':False,'off_time':'30 s'})
    expected=ureg.Quantity(25,'degree_Celsius').to('K')
    assert result['initial_winding_temperature']==expected
    exported=studies.study(2,'C','reference')
    assert exported['provenance']['catalog'].startswith('mlsysim:')
    assert '/Users/' not in str(exported['provenance'])


def test_zero_margin_boundary_matches_nonnegative_fault_oracle():
    result = studies.study(15, 'A', 'zero_margin')
    assert all(row['policy_status'] == 'supports stated claim' for row in result['rows'] if row['requirement'] != 'computed four-state oracle')
    oracle = next(row for row in result['rows'] if row['requirement'] == 'computed four-state oracle')
    assert oracle['supplied_evidence'] == 'PASS'


def test_staffing_changes_queue_without_changing_role_authority():
    small = studies.study(17, 'B', 'small_fleet')
    large = studies.study(17, 'B', 'large_fleet')
    a = {row['requirement']: row['supplied_evidence'] for row in small['rows']}
    b = {row['requirement']: row['supplied_evidence'] for row in large['rows']}
    assert magnitude(b['p99 wait before assistance'], 's') > magnitude(a['p99 wait before assistance'], 's')
    assert [row for row in small['rows'] if row.get('requirement') == 'authority'] == [row for row in large['rows'] if row.get('requirement') == 'authority']


def test_replacement_rate_is_independent_of_chunk_supply_and_evidence():
    baseline = values(studies.study(3, 'B', 'reference'))
    slower = values(studies.study(3, 'B', 'slower_renewal'))
    assert baseline['chunk span'] == slower['chunk span']
    assert baseline['tracked evidence horizon'] == slower['tracked evidence horizon']
    assert magnitude(slower['renewal period'], 's') > magnitude(baseline['renewal period'], 's')
    assert magnitude(slower['open loop span'], 's') > magnitude(baseline['open loop span'], 's')
    assert 'model refusal' in values(studies.study(3, 'B', 'faster_renewal'))
