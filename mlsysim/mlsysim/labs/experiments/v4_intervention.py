"""Finite fixed-endpoint blend comparison, not an installed-controller model.

Matches the chapter's smoothstep integrals and independently budgets a resident
stop from each tick. Torque-rate and translational jerk are separate channels.
The ramp-plus-constant-brake tail lacks a terminal smoothing transition and is
only a lower-bound comparison, never a qualified stopping trajectory.
"""
from __future__ import annotations

import hashlib
import json
import math
from importlib.resources import files

from mlsysim.core.units import ureg
from mlsysim.embodied.registry import Embodied
from mlsysim.fmt import fmt_qty, _is_integer_like
from mlsysim.physics.robotics import (
    calc_c2_stop_suffix, calc_quintic_blend_duration, calc_scurve_stop_distance,
)

_DATA = files('mlsysim.engine').joinpath('data/physical_studies/v4_intervention.json')


def handles(chapter, part):
    return int(chapter) == 14 and str(part).upper() == 'C'


def _catalog():
    return json.loads(_DATA.read_text(encoding='utf-8'))


def _validate(chapter, part, track):
    if not handles(chapter, part):
        raise ValueError('This study supports intervention Part C only.')
    if track not in ('warehouse', 'warehouse_mobile_manipulator'):
        raise ValueError('Physical studies require the warehouse reference track.')


def choices(chapter, part, track='warehouse'):
    _validate(chapter, part, track)
    return [{'id': c['id'], 'label': c['label']} for c in _catalog()['choices']]


def _display(value):
    if hasattr(value, 'units'):
        magnitude = float(value.magnitude)
        precision = 0 if _is_integer_like(magnitude) else max(6, 2-math.floor(math.log10(abs(magnitude))))
        return str(fmt_qty(value, value.units, precision=precision, trim_trailing_zeros=True))
    return value


def _calculate(case, assumptions):
    site = Embodied.Scenario.WarehouseAisle
    v0, a, tick = site.v_takeover_in_loop, site.a_brake, site.t_tick
    jerk = calc_c2_stop_suffix(site.machine.base.max_velocity, a)['peak_jerk']
    minimum = (1.875*a/jerk).to('s')
    count = math.ceil((minimum*case['window_factor']/tick).to('').magnitude - 1e-9)
    window = count*tick
    clearance = ureg.Quantity(assumptions['tight_remaining_motion_clearance' if case['tight_clearance'] else 'remaining_motion_clearance'])
    arm_minimum = calc_quintic_blend_duration(ureg.Quantity(assumptions['arm_torque_change']), ureg.Quantity(assumptions['arm_torque_rate_limit']))
    arm_window = math.ceil((arm_minimum/tick).to('').magnitude - 1e-9)*tick
    resident = calc_scurve_stop_distance(v0, a, jerk)
    trace = []
    worst = resident
    # Exact polynomial integrals at each permission tick, not Euler dynamics.
    for index in range(count+1):
        u = index/count if count else 1.0
        alpha = 10*u**3-15*u**4+6*u**5
        integral = u**6-3*u**5+2.5*u**4
        double_integral = u**7/7-u**6/2+u**5/2
        speed = v0-a*window*integral
        travel = v0*window*u-a*window**2*double_integral
        acceleration = -a*alpha
        rate = -a/window*(30*u*u*(1-u)**2) if count else None
        fallback = calc_scurve_stop_distance(speed, a, jerk)
        required = travel+fallback
        worst = max(worst, required)
        trace.append({'tick': index, 'elapsed': _display((index*tick).to('ms')),
                      'acceleration command': _display(acceleration.to('m/s^2')),
                      'commanded jerk': _display(rate.to('m/s^3')) if count else 'unbounded ideal step',
                      'modeled speed': _display(speed.to('m/s')),
                      'modeled travel': _display(travel.to('m')),
                      'resident stop from current speed': _display(fallback.to('m')),
                      'remaining resident-stop reserve': _display((clearance-required).to('m'))})
    end_speed = v0-a*window/2
    blend_travel = v0*window-a*window**2/7
    lower_bound_stop = blend_travel+end_speed**2/(2*a)
    peak = 1.875*a/window if count else None
    matched = case['controller_matched']
    # A mismatch is disclosed, not silently added to this fixed-endpoint trace.
    bias = 0*a if matched else ureg.Quantity(assumptions['unmatched_controller_bias'])
    return {'minimum base blend window': minimum.to('ms'), 'chosen base blend window': window.to('ms'),
            'base command jerk limit': jerk.to('m/s^3'),
            'peak base command jerk': peak.to('m/s^3') if peak is not None else 'unbounded ideal step',
            'within command rate limit': bool(count and peak <= jerk),
            'remaining motion clearance at commit': clearance,
            'resident stop at commit': resident.to('m'),
            'worst tick travel plus resident stop': worst.to('m'),
            'minimum tick stopping reserve': (clearance-worst).to('m'),
            'all sampled resident stops fit': bool(worst <= clearance),
            'blend travel': blend_travel.to('m'), 'speed at blend end': end_speed.to('m/s'),
            'partial ramp plus ideal brake lower bound': lower_bound_stop.to('m'),
            'controller state matched': matched, 'disclosed unmodeled controller bias': bias.to('m/s^2'),
            'arm minimum torque blend window': arm_minimum.to('ms'), 'arm tick-rounded window': arm_window.to('ms')}, trace


def study(chapter, part, choice, track='warehouse'):
    _validate(chapter, part, track)
    data = _catalog()
    case = next((c for c in data['choices'] if c['id'] == choice), None)
    if case is None:
        raise ValueError(f'Unknown intervention blend case {choice!r}')
    result, tick_trace = _calculate(case, data['assumptions'])
    # The visible trace stays readable. The complete permission-tick evidence is
    # retained separately in every saved/exported result, including the minimum.
    worst_index = min(range(len(tick_trace)), key=lambda i: ureg.Quantity(tick_trace[i]['remaining resident-stop reserve']).to('m').magnitude)
    indices = {round(i*(len(tick_trace)-1)/20) for i in range(21)} | {worst_index}
    trace = [tick_trace[i] for i in sorted(indices)]
    feedback = []
    if not result['within command rate limit']:
        feedback.append('The requested transition exceeds the base command-jerk bound; smoothness alone does not meet that bound.')
    if not result['all sampled resident stops fit']:
        feedback.append('At least one blend tick consumes the remaining resident-stop reserve; retain the resident fallback instead of admitting this blend.')
    if not result['controller state matched']:
        feedback.append('The fixed-endpoint trace is unchanged, but the incoming controller is not pre-biased. Its actual post-blend command is unknown; the displayed trajectory cannot establish a bumpless transfer.')
    if not feedback:
        feedback.append('The tick-rounded blend meets the command-rate and sampled reference reserve checks in this constructed clearance case. This is a comparison result, not permission to execute.')
    return {'rows': [{'quantity': k, 'value': _display(v)} for k, v in result.items()],
            'trace': trace, 'tick_trace': tick_trace,
            'trace_sampling': 'Display samples include exact endpoints and the worst reserve tick; tick_trace retains every permission tick.',
            'feedback': ' '.join(feedback),
            'limitation': 'Exact fixed-endpoint kinematic samples under ideal command tracking. Resident stops are restarted from zero acceleration at each sampled speed, conservatively charging that reference. Between-tick behavior and terminal acceleration smoothing are not qualified. The ramp-plus-ideal-brake tail is a partial lower bound. Controller mismatch is disclosed, not simulated. This browser study does not command hardware, validate braking, or grant deployment permission.',
            'inputs': dict(case) | data['assumptions'],
            'conditions': [{'quantity': 'comparison assumptions', 'value': data['assumption_scope']},
                           {'quantity': 'initial speed', 'value': _display(Embodied.Scenario.WarehouseAisle.v_takeover_in_loop)},
                           {'quantity': 'target deceleration', 'value': _display(Embodied.Scenario.WarehouseAisle.a_brake)}],
            'version': data['version'], 'brief': 'At commit, compare the same latched cruise-to-braking command through different blend windows. Separate the rate bound, finite-tick stopping reserve, and controller state.',
            'held_fixed': 'Entry speed, zero initial acceleration, target deceleration, command-jerk limit and permission tick remain fixed. The tight-clearance case changes only clearance; the unmatched case changes only controller-state evidence.',
            'source_kind': 'Analytical reference with constructed clearance cases',
            'prediction_prompt': 'Which check can fail when a smooth blend window grows?',
            'prediction_options': ['A longer window can pass the rate limit but exhaust stopping reserve.', 'Smoothness removes the travel cost.', 'Matching controller state makes clearance irrelevant.'],
            'instrument': {'kind': 'bar', 'label': 'Worst tick travel plus resident stop against clearance',
                           'x': ['Worst tick requirement', 'Remaining clearance', 'Minimum reserve'],
                           'y': [float(result[k].to('m').magnitude) for k in ['worst tick travel plus resident stop', 'remaining motion clearance at commit', 'minimum tick stopping reserve']],
                           'unit': 'm', 'scope': 'Separate reference quantities, not stacked. Full command/jerk/travel samples are in the tick trace. A negative reserve refuses this blend.'},
            'source': data['source'], 'provenance': {'case': 'constructed teaching case; not measured',
                'catalog': 'mlsysim:engine/data/physical_studies/v4_intervention.json',
                'catalog_sha256': hashlib.sha256(_DATA.read_bytes()).hexdigest(),
                'version': data['version'], 'model': 'fixed_endpoint_quintic_blend',
                'machine': 'Embodied.MobileManipulator.WarehouseMobileManipulator', 'site': 'Embodied.Scenario.WarehouseAisle'}, 'choice': choice}
