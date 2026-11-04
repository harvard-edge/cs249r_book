"""Behavioral falsifiers for finite physical-AI timing instruments."""
import pytest
from mlsysim.labs.experiments import v4_timing as timing


def case(key, choice):
    return timing._catalog()['parts'][key][choice]


def test_gearing_retains_linear_torque_quadratic_inertia():
    base,_=timing.current_response(case('02_B','reference'))
    more,_=timing.current_response(case('02_B','more_reduction'))
    assert more['ideal_output_torque']==2*base['ideal_output_torque']
    assert more['reflected_rotor_inertia']==4*base['reflected_rotor_inertia']


def test_acknowledgment_survives_clamping_but_torque_does_not():
    base, ramp=timing.current_response(case('02_B','reference'))
    clamped, trace=timing.current_response(case('02_B','current_clamp'))
    assert all(r['command_acknowledged'] for r in trace)
    assert clamped['current_clamped']
    assert clamped['delivered_output_torque'] < base['delivered_output_torque']
    assert trace[0]['delivered_A']==0
    assert ramp[1]['delivered_A'] < ramp[-1]['delivered_A']


def test_low_voltage_binds_even_with_current_limit_unchanged():
    result,_=timing.current_response(case('02_B','voltage_limited'))
    assert result['voltage_limited'] and not result['current_clamped']
    assert result['delivered_current_at_last_sample'] < result['requested_current']


@pytest.mark.parametrize('choice',['writer_stall','torn_copy'])
def test_reader_rejects_without_waiting_or_retry(choice):
    result,_=timing.mailbox_lease(case('04_B',choice))
    assert not result['snapshot_accepted']
    assert not result['reader_blocked']
    assert result['reader_attempts']==1
    assert result['retained_capture_epoch'].magnitude==0


def test_queue_spends_freshness_and_longer_lease_cannot_renew_evidence():
    fresh,_=timing.mailbox_lease(case('04_B','reference'))
    queued,_=timing.mailbox_lease(case('04_B','queued_state'))
    longer,_=timing.mailbox_lease(case('04_B','long_lease'))
    assert queued['evidence_age']>fresh['evidence_age']
    assert longer['chosen_lease']>fresh['chosen_lease']
    assert longer['authority_ends_at']==fresh['authority_ends_at']
    assert longer['first_expiry']=='evidence freshness'


def test_independent_reflex_can_end_authority_first():
    result,_=timing.mailbox_lease(case('04_B','protective_reflex'))
    assert result['first_expiry']=='protective reflex'


def test_serial_link_cost_is_separate_from_constant_local_publication_cost():
    base,_=timing.placement(case('13_C','bounded_snapshot'))
    slow,_=timing.placement(case('13_C','slower_link'))
    assert slow['board_link_serialization']==2*base['board_link_serialization']
    assert slow['local_read_cost']==base['local_read_cost']
    assert (slow['dispatch_to_admission_allocation']-base['dispatch_to_admission_allocation']).to('ms').magnitude==pytest.approx(base['board_link_serialization'].to('ms').magnitude)


@pytest.mark.parametrize('choice,verdict',[('early_feed','early trip'),('late_feed','late trip')])
def test_window_rejects_both_early_and_late_completed_feeds(choice,verdict):
    result,_=timing.placement(case('13_C',choice))
    assert result['watchdog_result']==verdict
    assert not result['liveness_claim_supported']


def test_unrelated_timer_does_not_prove_completion():
    result,_=timing.placement(case('13_C','lock_and_timer'))
    assert result['reader_blocked']
    assert result['local_read_cost'] is None
    assert not result['feed_follows_completed_checks']
    assert not result['liveness_claim_supported']


def test_all_cases_return_exportable_evidence_and_honest_boundary():
    for chapter,part in [(2,'B'),(4,'B'),(13,'C')]:
        for item in timing.choices(chapter,part):
            evidence=timing.study(chapter,part,item['id'])
            assert evidence['trace'] and evidence['rows']
            assert 'not' in evidence['limitation']
            assert 'not measured' in evidence['provenance']['case']
    with pytest.raises(ValueError):timing.study(2,'B','missing')
    with pytest.raises(ValueError):timing.choices(2,'B','cloud')
