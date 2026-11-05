"""Bounded sensing and scalar-estimation investigations over constructed traces.

These fixtures are classroom assumptions, not measurements of the warehouse
machine. Empirical order statistics are not population error guarantees; single
NEES/NIS samples are diagnostics, not covariance-calibration campaigns.
"""
from __future__ import annotations

import copy
import math
import json
import hashlib
from importlib.resources import files

from mlsysim.embodied.registry import Embodied

VERSION = 'warehouse-sensing-estimation.v1'
BOUNDARY = ('Constructed analytical trace only. This browser study does not command hardware, '
            'qualify a physical sensor, establish WCET, validate braking, or grant deployment permission.')
_DATA = files('mlsysim.engine').joinpath('data/physical_studies/v4_estimation.json')


def _catalog():
    return json.loads(_DATA.read_text(encoding='utf-8'))


def _key(chapter, part):
    return f'{int(chapter):02d}_{str(part).upper()}'


def handles(chapter, part):
    return _key(chapter, part) in _catalog()['choices']


def choices(chapter, part, track='warehouse'):
    key = _key(chapter, part)
    if track not in ('warehouse', 'warehouse_mobile_manipulator') or key not in _catalog()['choices']:
        raise ValueError('Unsupported sensing/estimation investigation identity')
    return [{'id': key, 'label': label} for key, label in _catalog()['choices'][key]]


def empirical_absolute_quantile(residuals, quantile):
    """Nearest-rank order statistic of finite absolute residuals, not a bound."""
    if not residuals or not 0 < quantile <= 1:
        raise ValueError('A nonempty residual set and quantile in (0,1] are required')
    if any(not math.isfinite(float(x)) for x in residuals):
        raise ValueError('Residuals must be finite')
    ordered = sorted(abs(float(x)) for x in residuals)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def scalar_update(mean, variance, observation, observation_variance):
    """Independent scalar Gaussian measurement update, with identity observation."""
    if not all(math.isfinite(v) for v in (mean, variance, observation, observation_variance)):
        raise ValueError('Scalar update inputs must be finite')
    if variance <= 0 or observation_variance <= 0:
        raise ValueError('Variances must be strictly positive')
    innovation = observation - mean
    innovation_variance = variance + observation_variance
    gain = variance / innovation_variance
    return {'mean': mean + gain * innovation, 'variance': (1 - gain) * variance,
            'gain': gain, 'innovation': innovation, 'innovation_variance': innovation_variance,
            'NIS': innovation ** 2 / innovation_variance}


def _perception(choice):
    site = Embodied.Scenario.WarehouseAisle
    age_ms = float(site.perception_age.to('ms').magnitude)
    speed = float(site.v_aisle.to('m/s').magnitude)
    catalog = _catalog()
    stages = [{'stage': name, 'duration_ms': age_ms * fraction} for name, fraction in catalog['stage_fractions']]
    if choice == 'queue_delay':
        stages[2]['duration_ms'] += age_ms
    elif choice == 'faster_path':
        stages[5]['duration_ms'] /= 2
    age = sum(row['duration_ms'] for row in stages)
    travel = speed * age / 1000
    condition = 'far rack' if choice.startswith('far_') else 'near rack'
    residual_condition = 'far rack' if choice == 'far_matched' else 'near rack'
    residuals = [x * (catalog['far_residual_scale'] if residual_condition == 'far rack' else 1) for x in catalog['residuals_m']]
    supplied = choice != 'missing_residuals'
    matched = supplied and condition == residual_condition
    statistic = empirical_absolute_quantile(residuals, catalog['residual_quantile']) if supplied else None
    inset = statistic if matched else None
    rows = {'observation age': f'{age:.9g} ms', 'unmodeled displacement': f'{travel:.9g} m',
            'operating condition': condition, 'residual evidence condition': residual_condition if supplied else 'not supplied',
            'empirical absolute residual quantile': f'{statistic:.9g} m' if statistic is not None else 'unknown',
            'residual provenance matches condition': matched,
            'usable classroom sensing inset': f'{inset:.9g} m' if inset is not None else 'unknown: matching residual evidence absent',
            'sensing clearance allocation': f'{travel + inset:.9g} m' if inset is not None else 'unknown',
            'population error guarantee': 'not established by this finite constructed sample'}
    trace = [{'event': row['stage'], 'duration_ms': row['duration_ms']} for row in stages]
    if supplied:
        trace += [{'event': 'constructed sensor-minus-truth residual', 'sample': i + 1,
                   'residual_m': value, 'condition': residual_condition} for i, value in enumerate(residuals)]
    return rows, trace, {'stage_durations_ms': stages, 'speed_m_per_s': speed,
            'quantile': catalog['residual_quantile'], 'quantile_method': 'nearest rank of absolute residuals',
            'residuals_m': residuals if supplied else [], 'operating_condition': condition,
            'residual_condition': residual_condition if supplied else None}, {
            'kind': 'bar', 'label': 'Acquisition-to-dispatch waterfall components',
            'x': [r['stage'] for r in stages], 'y': [r['duration_ms'] for r in stages], 'unit': 'ms',
            'scope': 'Constructed stage split of the canonical total age; components sum to age. Admission and braking are excluded.'}, (
            'Only the selected pipeline stage changes. Age spends blind travel; a matching finite residual set adds a separate spatial allocation. '
            'A near-range statistic cannot qualify a far-range condition. The far fixture scales residuals with range squared as a hypothetical stereo-error case, not a measured sensor property.')


def _memory(choice):
    # Hypothetical scalar range trace. Motion is known exactly here; independent
    # process variance accumulates during the delay. There are no intervening
    # updates, so one capture update plus propagation is the full replay.
    fixture = _catalog()['scalar_trace']
    delay, velocity = fixture['delay_s'], fixture['velocity_m_per_s']
    prior, prior_var = fixture['capture_prior_m'], fixture['capture_prior_variance_m2']
    obs, obs_var = fixture['capture_observation_m'], fixture['observation_variance_m2']
    process_rate = fixture['process_variance_rate_m2_per_s']
    truth = fixture['truth_present_m'] if choice != 'no_truth' else None
    if choice == 'common_bias':
        prior += fixture['shared_bias_m']
        obs += fixture['shared_bias_m']
    noise_scale = fixture['understated_noise_scale'] if choice == 'shrink_covariance' else 1.
    prior_var *= noise_scale
    obs_var *= noise_scale
    process_rate *= noise_scale
    predicted = prior + velocity * delay
    predicted_var = prior_var + process_rate * delay
    update_epoch = 'capture'
    updated = None
    if choice == 'history_expired':
        mean, variance = predicted, predicted_var
        update_epoch = 'refused: capture history unavailable'
    elif choice == 'arrival_update':
        updated = scalar_update(predicted, predicted_var, obs, obs_var)
        mean, variance = updated['mean'], updated['variance']
        update_epoch = 'arrival (incorrect observation epoch)'
    else:
        updated = scalar_update(prior, prior_var, obs, obs_var)
        mean = updated['mean'] + velocity * delay
        variance = updated['variance'] + process_rate * delay
    error = mean - truth if truth is not None else None
    nees = error ** 2 / variance if error is not None else None
    rows = {'update epoch': update_epoch, 'capture-to-arrival delay': f'{delay} s',
            'predicted present range': f'{predicted:.9g} m', 'predicted present variance': f'{predicted_var:.9g} m^2',
            'posterior present range': f'{mean:.9g} m', 'posterior present variance': f'{variance:.9g} m^2',
            'posterior standard deviation': f'{math.sqrt(variance):.9g} m',
            'independent present truth': f'{truth:.9g} m' if truth is not None else 'not supplied',
            'present estimation error': f'{error:.9g} m' if error is not None else 'unknown without independent truth',
            'NEES (single scalar sample)': nees if nees is not None else 'unknown without independent truth',
            'NIS (single scalar innovation)': updated['NIS'] if updated else 'not computed: no update',
            'statistical covariance qualification': 'unknown: no independent repeated-trial campaign supplied'}
    inputs = {'capture_prior_range_m': prior, 'capture_prior_variance_m2': prior_var,
              'measurement_range_m': obs, 'measurement_variance_m2': obs_var,
              'known_velocity_m_per_s': velocity, 'delay_s': delay,
              'process_variance_rate_m2_per_s': process_rate, 'independent_truth_m': truth,
              'replay_history_available': choice != 'history_expired', 'declared_noise_scale': noise_scale}
    trace = [{'event': 'capture', 'time_s': 0., 'prior_range_m': prior, 'observation_range_m': obs},
             {'event': 'arrival', 'time_s': delay, 'predicted_range_m': predicted},
             {'event': update_epoch, 'present_estimate_m': mean, 'present_variance_m2': variance}]
    if updated:
        trace.append({'event': 'update arithmetic', **copy.deepcopy(updated)})
    return rows, trace, inputs, {'kind': 'bar', 'label': 'Reported present uncertainty and independently observed error',
            'x': ['posterior standard deviation'] + (['absolute present error'] if error is not None else []),
            'y': [math.sqrt(variance)] + ([abs(error)] if error is not None else []), 'unit': 'm',
            'scope': 'One constructed scalar trace, not a covariance-calibration campaign. No ground truth means no plotted error.'}, (
            'Correct replay applies the measurement at capture and then propagates its mean and process variance to arrival. '
            'Arrival dating can tighten covariance around the wrong state. Understating all noise leaves the gain unchanged while shrinking reported covariance. '
            'A shared bias can leave innovations unchanged; NIS alone cannot establish absolute state accuracy. '
            'A single NEES or NIS sample cannot establish population consistency; expired history refuses the late update.')


def _geometry(choice):
    fixture = _catalog()['geometry_trace']
    distance = fixture['range_m'] * (fixture['far_range_multiplier'] if choice == 'far_calibration' else 1)
    age_angle = fixture['turn_rate_rad_per_s'] * fixture['delay_s'] if choice == 'arrival_transform' else 0.
    mounting_angle = fixture['calibration_error_rad'] if choice in ('stale_calibration', 'far_calibration') else 0.
    angle = age_angle + mounting_angle
    x, y = distance * math.cos(angle), distance * math.sin(angle)
    residual = math.hypot(x-distance, y)
    bound = fixture['clock_bound_s'] if choice != 'foreign_clock' else None
    # Exact rotation chord for an error angle. An unknown cross-clock epoch
    # leaves this bound unknown even when nominal transform arithmetic is zero.
    clock_residual = 2 * distance * math.sin(abs(fixture['turn_rate_rad_per_s'] * bound) / 2) if bound is not None else None
    rows = {'transform epoch': 'arrival' if choice == 'arrival_transform' else 'nominal capture',
            'landmark range': f'{distance:.9g} m', 'epoch angular error': f'{age_angle:.9g} rad',
            'mounting angular error': f'{mounting_angle:.9g} rad',
            'reference landmark x': f'{distance:.9g} m', 'reference landmark y': '0 m',
            'transformed landmark x': f'{x:.9g} m', 'transformed landmark y': f'{y:.9g} m',
            'nominal geometry residual': f'{residual:.9g} m',
            'clock conversion bound': f'{bound:.9g} s' if bound is not None else 'unknown',
            'clock-induced spatial bound': f'{clock_residual:.9g} m' if clock_residual is not None else 'unknown: clock conversion unbounded',
            'physical calibration qualification': 'not established by constructed geometry'}
    inputs = {**fixture, 'selected_range_m': distance, 'selected_epoch_angle_rad': age_angle,
              'selected_mounting_angle_rad': mounting_angle, 'selected_clock_bound_s': bound}
    trace = [{'event': 'sensor-to-base frame edge', 'rotation_rad': mounting_angle, 'translation_m': [0., 0.]},
             {'event': 'base-to-world capture edge', 'rotation_rad': 0., 'translation_m': [0., 0.]},
             {'event': 'base-to-world arrival edge', 'rotation_rad': fixture['turn_rate_rad_per_s'] * fixture['delay_s'], 'translation_m': [0., 0.]},
             {'event': 'landmark comparison', 'reference_xy_m': [distance, 0.], 'transformed_xy_m': [x, y]}]
    return rows, trace, inputs, {'kind': 'bar', 'label': 'Nominal geometry and bounded clock contribution',
            'x': ['nominal geometry residual'] + (['clock-induced spatial bound'] if clock_residual is not None else []),
            'y': [residual] + ([clock_residual] if clock_residual is not None else []), 'unit': 'm',
            'scope': 'Constructed planar point and frame chain with a turn in place. Separate contributions are not a measured total-error bound.'}, (
            'The same point rotated at arrival moves to a different world location. Capture transforms remove that dating error under the declared model; '
            'they do not remove an angular mounting error, whose spatial consequence grows with range. '
            'Without a clock-conversion bound, zero nominal residual cannot establish an epoch-error bound. All poses and calibration errors are constructed.')


def study(chapter, part, choice, track='warehouse'):
    available = choices(chapter, part, track)
    if choice not in {item['id'] for item in available}:
        raise ValueError(f'Unsupported sensing/estimation choice {choice!r}')
    perception = int(chapter) == 8
    geometry = perception and str(part).upper() == 'B'
    rows, trace, inputs, instrument, feedback = (_geometry if geometry else _perception if perception else _memory)(choice)
    title = 'Date and place the measurement' if geometry else 'Age and error spend the same clearance' if perception else 'Update where the measurement happened'
    source = 'books/vol4/08_perception/08_perception.qmd' if perception else 'books/vol4/09_memory/09_memory.qmd'
    limitation = ('Constructed planar turn-in-place frame chain only; no rolling shutter, per-point sweep compensation, installed calibration qualification or physical clearance guarantee.' if geometry else 'Finite constructed residual order statistic and constant-speed blind travel only; no population bound, real sensor qualification, detector accuracy, or stopping permission.' if perception else
                  'Scalar independent-noise Gaussian update with known constant velocity and no intervening updates. Constructed truth is not measured; no multivariate filter, real calibration campaign, or physical permission.')
    return {'rows': [{'quantity': key, 'value': value} for key, value in rows.items()],
            'trace': trace, 'inputs': inputs, 'conditions': [{'quantity': key.replace('_', ' '), 'value': value} for key, value in inputs.items()],
            'instrument': instrument, 'feedback': feedback, 'limitation': limitation + ' ' + BOUNDARY,
            'brief': ('The robot turns in place while a rack-end observation is in flight. Compare the capture and arrival frame chain, then change the mounting error or landmark range.' if geometry else 'The warehouse team must spend clearance on both stale evidence and spatial error. Compare the same capture pipeline, then challenge whether the residual sample matches the rack condition.' if perception else
                      'The warehouse robot moves while a range observation is in flight. Compare the current estimate, reported uncertainty, and independent truth after arrival dating, capture replay, or refusal.'),
            'held_fixed': ('Same planar point, capture, turn rate and delay; only the named transform epoch, mounting error, range or clock-bound premise changes.' if geometry else 'Canonical warehouse speed and baseline total age; the constructed stage split and residual samples are explicit. Each intervention changes only its named stage or residual-condition premise.' if perception else
                           'Same delay, known motion, original independent truth and capture observation except the explicitly named noise-scale, shared-bias, missing-truth or history intervention.'),
            'source_kind': 'Constructed analytical trace; not measured hardware', 'source': source,
            'prediction_prompt': ('Which temporal geometry or calibration premise changes in this case?' if geometry else 'Which named age allocation or residual-evidence premise changes in this case?' if perception else 'Can this case report smaller covariance without a more accurate current state?'),
            'prediction_options': (['The arrival transform rotates the point at a different physical epoch.',
                                    'Capture dating does not remove mounting error or an unknown clock conversion.',
                                    'Doubling range leaves an angular-error displacement unchanged.'] if geometry else
                                   ['Shortening backbone time also removes earlier exposure and transport time.',
                                    'Age and residual error spend separate portions of the sensing allocation.',
                                    'A residual statistic transfers unchanged to a different operating condition.'] if perception else
                                   ['A tighter covariance necessarily proves a more accurate state.',
                                    'Incorrect dating or understated noise can make covariance look tighter.',
                                    'Innovations always reveal shared absolute-state bias.']),
            'version': VERSION, 'choice': choice,
            'provenance': {'case': 'constructed teaching case; not measured', 'model': title,
                           'fixture': 'engine/data/physical_studies/v4_estimation.json',
                           'catalog_sha256': hashlib.sha256(_DATA.read_bytes()).hexdigest(), 'version': VERSION,
                           'machine': 'Embodied.MobileManipulator.WarehouseMobileManipulator',
                           'site': 'Embodied.Scenario.WarehouseAisle'}}
