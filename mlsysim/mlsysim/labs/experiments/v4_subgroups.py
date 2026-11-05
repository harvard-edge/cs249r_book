"""Subgroup sampling uncertainty and a separate conditional stopping model.

Rates do not determine latency. Each fictional encounter sample estimates only
probability of detection by its stated deadline; delayed stopping uses an
independently supplied delay assumption, never a delay inferred from the rate.
"""
from __future__ import annotations
import hashlib
import json
import math
from importlib.resources import files
from statistics import NormalDist

from mlsysim.core.units import ureg
from mlsysim.embodied.registry import Embodied

_DATA = files('mlsysim.engine').joinpath('data/physical_studies/v4_subgroups.json')


def wilson_interval(successes: int, trials: int, confidence: float = .95) -> tuple[float, float]:
    """Two-sided Wilson score interval for independent Bernoulli encounters."""
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError('A positive integer independent-encounter count is required')
    if isinstance(successes, bool) or not isinstance(successes, int) or not 0 <= successes <= trials:
        raise ValueError('Detected encounters must be an integer within the trial count')
    if not 0 < confidence < 1:
        raise ValueError('Confidence must lie strictly between zero and one')
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    rate = successes / trials
    denominator = 1 + z*z/trials
    center = (rate + z*z/(2*trials))/denominator
    radius = z*math.sqrt(rate*(1-rate)/trials + z*z/(4*trials*trials))/denominator
    return max(0., center-radius), min(1., center+radius)


def choices(chapter=17, part='C', track='warehouse'):
    _identity(chapter, part, track)
    return [{'id': key, 'label': case['label']} for key, case in _catalog()['cases'].items()]


def _identity(chapter, part, track):
    if int(chapter) != 17 or str(part).upper() != 'C' or track not in ('warehouse', 'warehouse_mobile_manipulator'):
        raise ValueError('Subgroup study supports warehouse17/C only')


def _catalog():
    return json.loads(_DATA.read_text())


def study(chapter, part, choice, track='warehouse'):
    _identity(chapter, part, track)
    data = _catalog()
    if choice not in data['cases']:
        raise ValueError(f'Unsupported subgroup case {choice!r}')
    case = data['cases'][choice]
    site = Embodied.Scenario.WarehouseAisle
    speed = site.v_aisle * case['speed_fraction']
    trace = []
    for group in case['groups']:
        lower, upper = wilson_interval(group['detected'], group['encounters'], data['confidence'])
        extra_delay = group['extra_delay_s'] * ureg.s
        reaction = (speed*(site.tau_delay + extra_delay)).to('m')
        braking = (speed**2/(2*site.a_brake)).to('m')
        residual = (site.d_clear-site.fixed_overhead-reaction-braking).to('m')
        remaining_braking_distance = max(site.d_clear-site.fixed_overhead-reaction, 0*ureg.m)
        residual_v2 = max(speed**2-2*site.a_brake*remaining_braking_distance, 0*(ureg.m/ureg.s)**2)
        boundary_speed = math.sqrt(residual_v2.to('m^2/s^2').magnitude)*ureg.m/ureg.s
        boundary_energy = (site.machine.loaded_mass*boundary_speed**2/2).to('J')
        trace.append({**group, 'deadline_detection_rate': group['detected']/group['encounters'],
            'wilson_lower': lower, 'wilson_upper': upper, 'wilson_width': upper-lower,
            'separately_assumed_extra_delay_s': group['extra_delay_s'],
            'additional_blind_travel_m': (speed*extra_delay).to('m').magnitude,
            'ideal_constant_deceleration_distance_m': braking.magnitude,
            'reference_clearance_reserve_m': residual.magnitude,
            'conditional_speed_at_allocated_boundary_m_per_s': boundary_speed.to('m/s').magnitude,
            'conditional_rigid_body_energy_at_allocated_boundary_J': boundary_energy.magnitude,
            'conditional_reference_budget_met': residual.magnitude >= 0})
    total = sum(group['encounters'] for group in case['groups'])
    detected = sum(group['detected'] for group in case['groups'])
    pooled = wilson_interval(detected, total, data['confidence'])
    rows = [{'quantity': 'aggregate deadline detection rate', 'value': detected/total},
        {'quantity': 'aggregate Wilson interval', 'value': list(pooled)},
        {'quantity': 'confidence', 'value': data['confidence']},
        {'quantity': 'chosen speed', 'value': str(speed.to('m/s'))},
        {'quantity': 'independent encounter count', 'value': total},
        {'quantity': 'minimum group lower bound', 'value': min(row['wilson_lower'] for row in trace)},
        {'quantity': 'worst conditional clearance reserve', 'value': str(min(row['reference_clearance_reserve_m'] for row in trace))+' m'}]
    limit = ('Fictional independent Bernoulli encounter samples with two-sided Wilson score intervals; '
        'correlated frames must not be counted as independent encounters. Intervals are not simultaneous '
        'group guarantees, interval overlap is not parity, and pooled results cannot qualify a group. '
        'Separate assumed delay and ideal constant-deceleration stopping omit jerk, plant variation and sensor qualification. Boundary speed and rigid-body energy describe the allocated clearance boundary, not human contact force or injury risk. '
        'This browser study does not command hardware, validate a detector or braking, or grant deployment permission.')
    return {'rows': rows, 'trace': trace, 'inputs': case, 'conditions': [
        {'quantity': 'source of detection delay', 'value': 'separately stipulated teaching assumption, not derived from detection rate'},
        {'quantity': 'fixed reference clearance', 'value': str(site.d_clear)},
        {'quantity': 'reference deceleration', 'value': str(site.a_brake)}],
        'brief': 'A pooled detector summary hides which exposed group has weak evidence. Compare sample uncertainty separately from the distance lost under a delayed-detection assumption.',
        'held_fixed': 'The physical reference and group deadline definition stay fixed. The equal-rate pair changes only independent sample size; the lower-speed case changes the operating envelope.',
        'feedback': 'Inspect each group interval and separately computed clearance reserve. More encounters can narrow uncertainty without removing a timing gap; lower speed changes travel without improving the detector. Neither establishes a physical permit.',
        'limitation': limit, 'source_kind': 'Constructed independent encounter samples and analytical reference',
        'source': data['source'], 'version': data['version'], 'choice': choice,
        'prediction_prompt': 'Which change resolves sampling uncertainty, and which changes physical exposure?',
        'prediction_options': ['A pooled rate establishes every group', 'More independent encounters narrow uncertainty; lower speed changes travel', 'Equal point estimates prove equal support'],
        'instrument': {'kind': 'bar', 'label': 'Group-specific Wilson interval width', 'unit': 'probability',
            'x': [row['group'] for row in trace], 'y': [row['wilson_width'] for row in trace],
            'scope': 'Sampling uncertainty in fictional independent encounters, separate from the assumed latency/stopping model.'},
        'provenance': {'case': 'constructed teaching case; not measured', 'catalog': 'mlsysim:engine/data/physical_studies/v4_subgroups.json',
            'catalog_sha256': hashlib.sha256(_DATA.read_bytes()).hexdigest(), 'version': data['version'],
            'model': 'Wilson score interval plus separately assumed delay and constant deceleration'}}
