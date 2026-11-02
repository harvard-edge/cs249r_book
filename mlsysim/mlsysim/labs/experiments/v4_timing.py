"""Finite current/publication/watchdog instruments; no physical qualification.

Case facts live in package data. The current ramp is a zero-speed ideal R-L
voltage-step bound with an ideal command/current limiter, not a servo model.
Publication traces execute a single reader attempt, never an unbounded retry.
"""
from __future__ import annotations

import hashlib
import json
import math
from importlib.resources import files

from mlsysim.core.units import ureg
from mlsysim.embodied.registry import Embodied
from mlsysim.physics.robotics import calc_reflected_inertia, calc_watchdog_lease_bound

_DATA = files('mlsysim.engine').joinpath('data/physical_studies/v4_timing.json')
_BASE = files('mlsysim.engine').joinpath('data/physical_studies/warehouse_investigations.json')
VERSION = 'physical-timing.v1'


def _catalog():
    return json.loads(_DATA.read_text())


def handles(chapter, part):
    return f'{int(chapter):02d}_{str(part).upper()}' in _catalog()['parts']


def _key(chapter, part, track):
    if track not in ('warehouse', 'warehouse_mobile_manipulator'):
        raise ValueError('Timing studies support only the warehouse reference.')
    key = f'{int(chapter):02d}_{str(part).upper()}'
    if key not in _catalog()['parts']:
        raise ValueError(f'Unknown timing investigation {key}')
    return key


def choices(chapter, part, track='warehouse'):
    return [{'id': k, 'label': v['label']} for k, v in _catalog()['parts'][_key(chapter, part, track)].items()]


def current_response(case):
    """Exact zero-speed R-L step with ideal limiting; returns units and trace."""
    p = _catalog()['motor']
    q = ureg.Quantity
    resistance, inductance = q(p['resistance']), q(p['inductance'])
    supply = q(case.get('supply', p['supply']))
    requested = q(p['requested_current'])
    limit = q(p['current_limit']) if case.get('clamp') else requested
    kt = q(p['torque_constant'])
    tau = (inductance/resistance).to('ms')
    ratio = case['gear_ratio']
    # Legacy gear comparison is the declared hypothetical rotor, not a Panda part.
    legacy = json.loads(_BASE.read_text())['parts']['02_B']['choices'][0]['inputs']
    efficiency = legacy['efficiency']
    trace = []
    delivered = 0*ureg.A
    for time in p['sample_times_ms']:
        available = (supply/resistance)*(1-math.exp(-time/tau.magnitude))
        delivered = min(requested, limit, available).to('A')
        trace.append({'time_ms':time, 'command_acknowledged':True,
                      'requested_A':requested.to('A').magnitude, 'delivered_A':delivered.magnitude,
                      'delivered_output_Nm':(kt*delivered*ratio*efficiency).to('N*m').magnitude})
    result = {'requested_current':requested, 'current_limit':limit, 'supply':supply,
              'electrical_time_constant':tau, 'delivered_current_at_last_sample':delivered,
              'command_acknowledged':True, 'requested_output_torque':kt*requested*ratio*efficiency,
              'delivered_output_torque':kt*delivered*ratio*efficiency,
              'ideal_output_torque':q(legacy['motor_torque'])*ratio*efficiency,
              'reflected_rotor_inertia':calc_reflected_inertia(ratio,q(legacy['rotor_inertia'])),
              'current_clamped':bool(limit < min(requested,supply/resistance)),
              'voltage_limited':bool(supply/resistance < min(requested,limit))}
    return result, trace


def publication(case):
    """Execute one seqlock read. Sequence change/odd state rejects without retry."""
    p = _catalog()['mailbox']
    mode = case['mode']
    before = p['committed_sequence']
    after = before
    copied = True
    blocked = mode == 'blocking'
    if mode in ('odd','blocking'):
        before += 1
        after = before
        copied = False
    elif mode == 'torn':
        after += 2
    accepted = not blocked and before % 2 == 0 and before == after
    capture = p['new_capture_ms'] if accepted and mode not in ('queue','stale') else p['previous_capture_ms']
    duration = p['sequence_read_us'] + (p['copy_us']+p['sequence_read_us'] if copied else 0)
    return {'reader_blocked':blocked, 'snapshot_accepted':accepted,
            'sequence_before':before,'sequence_after':after, 'reader_attempts':1,
            'local_read_cost':None if blocked else duration*ureg.us,
            'retained_capture_epoch':capture*ureg.ms,
            'evidence_age':(p['now_ms']-capture)*ureg.ms,
            'remaining_evidence_validity':max(0,capture+p['evidence_validity_ms']-p['now_ms'])*ureg.ms,
            'issue_time_renews_capture':False}, [
        {'event':'sequence before','sequence':before},
        {'event':'copy','result':'bounded copy' if copied else ('waits on stalled writer' if blocked else 'skipped: writer active')},
        {'event':'sequence after','sequence':after},
        {'event':'reader decision','result':'wait (unbounded)' if blocked else ('accept complete snapshot' if accepted else 'retain previous snapshot, no retry'),
         'capture_epoch_ms':capture}]


def mailbox_lease(case):
    result, trace = publication(case)
    p = _catalog()['mailbox']
    site = Embodied.Scenario.WarehouseAisle
    raw = calc_watchdog_lease_bound(site.d_clear-site.fixed_overhead,site.v_aisle,site.a_brake)
    raw.pop('t_lease_max')
    reserve = site.perception_age+site.t_tick+site.t_brake_onset+site.t_bus
    chosen = site.t_lease*case['lease_fraction']
    result.update(raw)
    result.update(nonlease_delay_reserved=reserve, remaining_lease_budget=raw['t_total_delay_max']-reserve,
                  chosen_lease=chosen, within_ideal_budget=bool(chosen<=raw['t_total_delay_max']-reserve))
    # Authority begins at constructed admission time now; original capture persists.
    deadlines = {'evidence freshness':p['now_ms']+result['remaining_evidence_validity'].to('ms').magnitude,
                 'execution lease':p['now_ms']+chosen.to('ms').magnitude,
                 'host heartbeat':p['heartbeat_deadline_ms']}
    if 'protective_reflex_ms' in case:
        deadlines['protective reflex'] = case['protective_reflex_ms']
    earliest = min(deadlines.values())
    result['authority_ends_at'] = earliest*ureg.ms
    result['first_expiry'] = ', '.join(k for k,v in deadlines.items() if v==earliest)
    trace += [{'event':name,'deadline_ms':value,'ends_authority':value==earliest} for name,value in deadlines.items()]
    return result, trace


def placement(case):
    result, trace = publication(case)
    site = Embodied.Scenario.WarehouseAisle
    p = _catalog()['watchdog']
    link = site.t_board_link/case.get('link_fraction',1)
    allocation = site.t_dispatch_to_admission-site.t_board_link+link
    result.update(board_link_serialization=link, dispatch_to_admission_allocation=allocation,
                  capture_to_dispatch_age=site.perception_age,
                  local_read_budget=p['local_read_budget_us']*ureg.us,
                  local_read_within_declared_budget=not result['reader_blocked'] and result['local_read_cost']<=p['local_read_budget_us']*ureg.us)
    feed = case['feed']
    times = p['early_feed_ms'] if feed=='early' else p['healthy_feed_ms']
    if feed=='late':
        times = times[:1]+[times[0]+site.mcu_self_watchdog.to('ms').magnitude+1]
    intervals = [b-a for a,b in zip(times,times[1:])]
    upper = site.mcu_self_watchdog.to('ms').magnitude
    statuses = [('early trip' if dt<p['earliest_ms'] else 'late trip' if dt>upper else 'in window') for dt in intervals]
    completed = feed!='timer' and not result['reader_blocked']
    result.update(watchdog_earliest=p['earliest_ms']*ureg.ms,watchdog_latest=site.mcu_self_watchdog,
                  feed_follows_completed_checks=completed,
                  watchdog_result='; '.join(statuses),
                  liveness_claim_supported=completed and all(s=='in window' for s in statuses))
    trace += [{'event':'watchdog feed interval','interval_ms':dt,'window_result':status,'completed_checks':completed}
              for dt,status in zip(intervals,statuses)]
    return result, trace


def study(chapter, part, choice, track='warehouse'):
    from .v4_studies import _flat, BOUNDARY
    key = _key(chapter,part,track)
    cases = _catalog()['parts'][key]
    if choice not in cases:
        raise ValueError(f'Unsupported timing choice {choice!r}')
    case = cases[choice]
    base = json.loads(_BASE.read_text())['parts'][key]
    result, trace = (current_response(case) if key=='02_B' else mailbox_lease(case) if key=='04_B' else placement(case))
    limit = ('Ideal zero-speed R-L voltage step, fixed resistance and torque constant, ideal current limiter, and hypothetical gearing. No closed-loop regulator, back EMF, thermal saturation, compliance or installed-drive measurement.' if key=='02_B' else
             'Finite single-attempt publication and timer trace, not concurrent firmware. Coherence, memory barriers, watchdog wiring, actual bus behavior, WCET and braking remain unqualified.')
    if key=='02_B':
        instrument={'kind':'bar','label':'Requested and delivered current at final sample','unit':'A',
                    'x':['requested current','delivered current'],
                    'y':[result['requested_current'].to('A').magnitude,result['delivered_current_at_last_sample'].to('A').magnitude],
                    'scope':'Ideal zero-speed current ramp; inspect the dated trace for its transient. Not a measured servo response.'}
    else:
        instrument=None
    return {'rows':[{'quantity':k.replace('_',' '),'value':_flat(v) if v is not None else 'unbounded: waiting on writer'} for k,v in result.items()],
            'trace':_flat(trace), 'inputs':case, 'conditions':[{'quantity':k.replace('_',' '),'value':_flat(v)} for k,v in case.items()],
            'feedback':('Inspect delivered current, not packet acknowledgment; gearing multiplies torque and reflected inertia at different rates.' if key=='02_B' else
                        'Inspect sequence acceptance, original evidence age and each timer separately. A bounded refusal preserves the reader tick; a live timer does not prove checks completed.'),
            'limitation':limit+' '+BOUNDARY, 'brief':base['brief'],'held_fixed':base['held_fixed'],
            'source':base['source'],'source_kind':'Analytical reference and constructed finite trace',
            'prediction_prompt':base['prediction_prompt'],'prediction_options':base['prediction_options'],
            'instrument':instrument,'choice':choice,'version':VERSION,
            'provenance':{'case':'constructed teaching case; not measured','catalog':'mlsysim:engine/data/physical_studies/v4_timing.json',
                          'catalog_sha256':hashlib.sha256(_DATA.read_bytes()).hexdigest(),'version':VERSION,'model':key}}
