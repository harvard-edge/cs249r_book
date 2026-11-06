"""Bounded intentional-contact work ledger and series-spring comparisons.

Reference equations: enforcement chapter's scalar constant-speed interval and
springs-in-series relation. These cases stipulate realized damping and an
independent force cap; neither is an installed-machine qualification.
"""
from __future__ import annotations

import math
import json
import hashlib
from importlib.resources import files

from mlsysim.core.units import ureg
from mlsysim.embodied.registry import Embodied
from mlsysim.physics.robotics import calc_latch_contact_stopping_profile

VERSION = 'intentional-contact-work-ledger.v1'
_DATA = files('mlsysim.engine').joinpath('data/physical_studies/v4_compliance.json')


def _catalog() -> dict:
    return json.loads(_DATA.read_text(encoding='utf-8'))


def choices() -> list[dict]:
    return _catalog()['choices']


def energy_tank_step(initial, reserve, realized_damping_power, proposed_force,
                     speed, duration, force_cap) -> dict:
    """Exact work balance for a stipulated nonnegative constant-speed interval.

    Active force excludes damping: crediting realized damping and debiting total
    force would count dissipation twice. At zero speed, force remains limited by
    the independent force channel although active work is zero. Negative speed,
    force and damping credit are outside this one-direction teaching model.
    """
    fields = {'initial': (initial, 'J'), 'reserve': (reserve, 'J'),
              'realized_damping_power': (realized_damping_power, 'W'),
              'proposed_force': (proposed_force, 'N'), 'speed': (speed, 'm/s'),
              'duration': (duration, 's'), 'force_cap': (force_cap, 'N')}
    values = {}
    for key, (quantity, unit) in fields.items():
        magnitude = float(quantity.to(unit).magnitude)
        if not math.isfinite(magnitude) or magnitude < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
        values[key] = magnitude
    if values['duration'] == 0 or values['reserve'] == 0:
        raise ValueError('duration and reserve must be positive')
    if values['initial'] < values['reserve']:
        raise ValueError('initial tank level is below its reserve')
    credit = (realized_damping_power * duration).to('J')
    allowance = (initial - reserve + credit).to('J')
    travel = (speed * duration).to('m')
    force = min(proposed_force.to('N'), force_cap.to('N'))
    energy_cap = None
    if values['speed'] > 0:
        energy_cap = (allowance / travel).to('N')
        force = min(force, energy_cap)
    debit = (force * travel).to('J')
    final = (initial + credit - debit).to('J')
    # The algebra guarantees the floor; trim only floating-point cancellation.
    if final < reserve and math.isclose(float(final.to('J').magnitude), values['reserve'], rel_tol=1e-12):
        final = reserve.to('J')
    return {'initial_tank': initial.to('J'), 'damping_credit': credit,
            'active_work': debit, 'final_tank': final, 'reserve': reserve.to('J'),
            'admitted_force': force, 'energy_limited_force': energy_cap,
            'independent_force_cap': force_cap.to('N')}


def study(choice: str) -> dict:
    """Return serializable evidence in the shared physical-companion schema."""
    if choice not in {case['id'] for case in choices()}:
        raise ValueError(f'Unknown intentional-contact case {choice!r}')
    # Import presentation conversion only at evaluation time so dispatcher imports
    # do not create a module-initialization cycle.
    from .v4_studies import _flat, BOUNDARY
    site = Embodied.Scenario.WarehouseAisle
    inputs = {'latch_stiffness': site.k_latch, 'latch_approach_speed': site.v_latch_approach,
              'contact_response': site.t_contact_response, 'latch_force_limit': site.f_latch_limit}
    trace = []
    if choice.startswith('tank_'):
        # Declared hypothetical scalar controller settings, not measured machine
        # properties. All work, power and tank scale follows canonical contact inputs.
        settings = _catalog()['tank']
        proposal = site.f_latch_limit * settings['proposal_force_fraction']
        cap = site.f_latch_limit * settings['force_cap_fraction']  # stipulated after all independent allowances
        duration = site.t_contact_response
        speed = site.v_latch_approach
        work = (proposal * speed * duration).to('J')
        reserve = work
        spare = 0 * work if choice == 'tank_reserve' else settings['initial_spare_work_intervals'] * work
        damping = (proposal * speed * (settings['realized_damping_power_fraction'] if choice == 'tank_damping' else 0)).to('W')
        tank = reserve + spare
        inputs.update({'proposed_active_force': proposal, 'stipulated_force_cap': cap,
                       'constant_speed': speed, 'interval': duration, 'initial_tank': tank,
                       'reserve': reserve, 'realized_damping_power_lower_bound': damping,
                       'intervals': settings['intervals']})
        for interval in range(inputs['intervals']):
            step = energy_tank_step(tank, reserve, damping, proposal, speed, duration, cap)
            trace.append({'interval': interval + 1, **step})
            tank = step['final_tank']
        result = {'initial_tank': inputs['initial_tank'], 'final_tank': tank, 'reserve': reserve,
                  'total_active_work': sum((step['active_work'] for step in trace), 0*ureg.J),
                  'total_damping_credit': sum((step['damping_credit'] for step in trace), 0*ureg.J),
                  'last_admitted_active_force': trace[-1]['admitted_force'],
                  'independent_force_cap': cap,
                  'installed_force_channel_qualification': 'unknown; no installed measurements supplied'}
        instrument = {'kind': 'bar', 'label': 'Active contact force admitted in each interval',
                      'x': [str(step['interval']) for step in trace],
                      'y': [float(step['admitted_force'].to('N').magnitude) for step in trace], 'unit': 'N',
                      'scope': 'Constant speed and realized damping are stipulated. Force and work bounds are distinct.'}
        feedback = ('The active command is clipped by both spendable work and the independent force cap. '
                    'At the reserve without damping credit no positive active work remains. '
                    'Credited damping is realized work, not the requested damping command; '
                    'an energy reserve alone cannot bound contact force or certify stability.')
    else:
        stiffness = site.k_latch
        spring = site.k_latch if choice == 'series_soft' else None
        effective = stiffness if spring is None else 1 / (1/stiffness + 1/spring)
        inputs.update({'added_spring_stiffness': spring if spring is not None else 'none',
                       'effective_stiffness': effective})
        result = calc_latch_contact_stopping_profile(latch_stiffness=effective, scenario=site)
        result['effective_stiffness'] = effective
        # The displacement needed for the same spring force grows as stiffness
        # decreases; it is not free protection or evidence the latch can close.
        result['compression_at_tripwire'] = (site.f_latch_tripwire/effective).to('mm')
        result['installed_spring_travel_and_task_completion'] = 'unknown; not qualified'
        instrument = {'kind': 'bar', 'label': 'Latch contact force at brake onset',
                      'x': ['modeled force', 'latch force limit'],
                      'y': [float(result['f_at_brake_onset'].to('N').magnitude), float(site.f_latch_limit.to('N').magnitude)],
                      'unit': 'N', 'scope': 'Series springs lower force-rise rate but require additional compression travel.'}
        feedback = ('Series compliances add, so the effective stiffness is lower than either equal-stiffness element. '
                    'The same speed and delay then produce less force growth, while reaching the same force requires '
                    'more compression. This does not establish available spring travel, latch closure, or human-contact safety.')
    return {'rows': [{'quantity': k.replace('_', ' '), 'value': _flat(v)} for k, v in result.items()],
            'feedback': feedback, 'limitation': ('Stipulated scalar work intervals and linear series-spring reference only; '
                'no closed-loop dynamics, sensor-error qualification, actuator realization, stability proof or physical contact permission. ' + BOUNDARY),
            'inputs': _flat(inputs), 'conditions': [{'quantity': k.replace('_', ' '), 'value': _flat(v)} for k,v in inputs.items()],
            'trace': _flat(trace), 'version': VERSION,
            'brief': 'The arm must push a latch deliberately. Compare finite active-work allowance with passive mechanical compliance; neither replaces an independently qualified force channel.',
            'held_fixed': 'Hold canonical latch approach, response time and latch force limit fixed. Tank cases change initial allowance or realized damping credit; spring cases change only series compliance.',
            'source_kind': 'Analytical contact reference with stipulated controller inputs',
            'prediction_prompt': 'What can permit useful contact without treating an energy bound as a force guarantee?',
            'prediction_options': ['Clip active work at the reserve, retain a separate force cap, and account for the travel cost of compliance.',
                                   'A positive energy reserve guarantees small contact force on any surface.',
                                   'Requested damping replenishes the reserve even if the actuator never realizes it.'],
            'instrument': instrument, 'source': 'books/vol4/12_enforcement/12_enforcement.qmd',
            'provenance': {'case': 'constructed teaching case; not measured', 'model': VERSION, 'catalog_sha256': hashlib.sha256(_DATA.read_bytes()).hexdigest(),
                           'catalog': 'mlsysim:engine/data/physical_studies/v4_compliance.json',
                           'machine': 'Embodied.MobileManipulator.WarehouseMobileManipulator',
                           'site': 'Embodied.Scenario.WarehouseAisle',
                           'assumptions': 'constant nonnegative speed, exact scalar work intervals, realized damping lower bound; linear series springs'},
            'choice': choice}
