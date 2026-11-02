"""Mode averaging and dependent-decoder budgets over a finite constructed case.

A cross-section screen makes averaging visible without claiming a learned
policy, full trajectory, collision guarantee or quality benefit from more steps.
"""
from __future__ import annotations

import hashlib
import json
from importlib.resources import files

from mlsysim.core.units import ureg
from mlsysim.embodied.registry import Embodied
from mlsysim.fmt import fmt_qty, _is_integer_like

_DATA = files('mlsysim.engine').joinpath('data/physical_studies/v4_training.json')
BOUNDARY = ('This browser study does not command hardware, establish WCET, validate braking, '
            'qualify the SO-101 station, or grant deployment permission.')


def _catalog():
    return json.loads(_DATA.read_text(encoding='utf-8'))


def handles(chapter, part):
    return int(chapter) == 6 and str(part).upper() == 'A'


def choices(chapter, part, track='warehouse'):
    if not handles(chapter, part) or track not in ('warehouse', 'warehouse_mobile_manipulator'):
        raise ValueError('Unsupported training investigation identity')
    return [{'id': item['id'], 'label': item['label']} for item in _catalog()['choices']]


def _display(value):
    if hasattr(value, 'units'):
        return str(fmt_qty(value, value.units, precision=0 if _is_integer_like(value.magnitude) else 6, trim_trailing_zeros=True))
    return value


def calculate(case, assumptions):
    machine = Embodied.MobileManipulator.WarehouseMobileManipulator
    modes = [ureg.Quantity(value) for value in assumptions['action_modes']]
    probabilities = assumptions['mode_probabilities']
    mean = sum((p * value for p, value in zip(probabilities, modes)), 0*ureg.m)
    action = mean if case['representation'] == 'mean' else modes[case['mode_index']]
    half_width = ureg.Quantity(assumptions['obstacle_half_width'])
    loss = sum((p*(action-mode)**2 for p, mode in zip(probabilities, modes)), 0*ureg.m**2)
    encoding = ureg.Quantity(assumptions['encoder_time'])
    per_evaluation = (ureg.Quantity(assumptions['reference_total_iterative_time'])-encoding) / assumptions['reference_dependent_evaluations']
    decode = case['evaluations'] * per_evaluation
    stretch = ureg.Quantity(case['stretch'])
    total = encoding + decode + stretch
    deadline = (1/machine.chunk_rate).to('ms')
    return {
        'selected_lateral_action': action.to('m'),
        'conditional_mean_action': mean.to('m'),
        'expected_squared_action_error': loss.to('m^2'),
        'matches_a_supplied_action_mode': any(action == value for value in modes),
        'cross_section_outside_obstacle': bool(abs(action) > half_width),
        'cross_section_obstacle_half_width': half_width,
        'encoder_time_once': encoding,
        'dependent_evaluations': case['evaluations'],
        'time_per_dependent_evaluation': per_evaluation.to('ms'),
        'dependent_decode_time': decode.to('ms'),
        'added_delay': stretch,
        'complete_proposal_latency': total.to('ms'),
        'proposal_renewal_deadline': deadline,
        'proposal_renewal_slack': (deadline-total).to('ms'),
        'strictly_before_renewal': bool(total < deadline),
        'quality_gain_from_more_evaluations': 'unknown; candidates are stipulated, not generated or evaluated by a learned model',
        'physical_trajectory_qualified': False,
    }


def study(chapter, part, choice, track='warehouse'):
    choices(chapter, part, track)
    data = _catalog()
    case = next((item for item in data['choices'] if item['id'] == choice), None)
    if case is None:
        raise ValueError(f'Unsupported training choice {choice!r}')
    assumptions = data['assumptions']
    result = calculate(case, assumptions)
    trace = [{'event': 'supplied action mode', 'mode': index+1,
              'lateral_action': value, 'conditional_probability': probability}
             for index, (value, probability) in enumerate(zip(assumptions['action_modes'], assumptions['mode_probabilities']))]
    trace += [{'event': 'selected candidate', 'lateral_action': _display(result['selected_lateral_action']),
               'screen': 'outside the single obstacle cross-section' if result['cross_section_outside_obstacle'] else 'intersects the single obstacle cross-section'},
              {'event': 'encoding once', 'duration': _display(result['encoder_time_once'])},
              {'event': 'dependent decoder evaluations', 'count': case['evaluations'],
               'per_evaluation': _display(result['time_per_dependent_evaluation']),
               'total': _display(result['dependent_decode_time'])}]
    if not result['cross_section_outside_obstacle']:
        feedback = 'The conditional mean minimizes this squared-error objective but lies between the two supplied modes, inside the constructed obstacle cross-section. Timeliness does not repair this action representation.'
    elif not result['strictly_before_renewal']:
        feedback = 'The selected coordinate matches a supplied mode but its declared dependent evaluations and delay miss proposal renewal. More evaluations provide no measured quality credit in this case.'
    else:
        feedback = 'This supplied mode passes the single cross-section screen and declared renewal budget. Neither result establishes a complete feasible trajectory or learned policy quality.'
    timing_keys = ['encoder_time_once', 'dependent_decode_time', 'added_delay', 'proposal_renewal_slack']
    inputs = dict(case) | assumptions
    return {'rows': [{'quantity': key.replace('_', ' '), 'value': _display(value)} for key, value in result.items()],
            'trace': trace, 'inputs': inputs,
            'conditions': [{'quantity': 'candidate and timing assumptions', 'value': assumptions['kind']},
                           {'quantity': 'geometry scope', 'value': assumptions['geometry_scope']},
                           {'quantity': 'timing source', 'value': assumptions['timing_source']},
                           {'quantity': 'renewal deadline', 'value': _display(result['proposal_renewal_deadline'])}],
            'feedback': feedback,
            'limitation': 'Finite symmetric action coordinates at one cross-section and declared linear dependent-decoder cost. No learned inference, measured hardware timing, full path, closed-loop success, or accuracy-versus-evaluation curve is modeled. ' + BOUNDARY,
            'brief': 'At a rack obstruction, two supplied action modes pass on opposite sides. Compare their conditional mean with either supplied mode, then spend the same proposal renewal budget on dependent decoder evaluations.',
            'held_fixed': 'Same two action coordinates, probabilities, cross-section, encoder cost, per-evaluation cost and warehouse renewal rate. Only the named representation, selected mode, evaluation count or added delay changes.',
            'source_kind': 'Constructed analytical trace; not measured hardware',
            'source': data['source'], 'version': data['version'], 'choice': choice,
            'prediction_prompt': 'Can a lower squared-error action be unusable, or a supplied valid mode arrive too late?',
            'prediction_options': ['The conditional mean can minimize error while intersecting the obstruction.',
                                   'Every additional dependent evaluation is free once encoding finishes.',
                                   'Selecting a demonstrated mode guarantees timing and full trajectory feasibility.'],
            'instrument': {'kind': 'bar', 'label': 'Declared proposal budget and remaining slack',
                           'x': [key.replace('_', ' ') for key in timing_keys],
                           'y': [float(result[key].to('ms').magnitude) for key in timing_keys],
                           'unit': 'ms', 'scope': 'Distinct timing allocations, not stacked. Negative slack misses renewal; action coordinates appear in the trace.'},
            'provenance': {'case': 'constructed teaching case; not measured', 'model': 'finite_mode_and_decoder_budget',
                           'catalog': 'mlsysim:engine/data/physical_studies/v4_training.json',
                           'catalog_sha256': hashlib.sha256(_DATA.read_bytes()).hexdigest(), 'version': data['version'],
                           'machine': 'Embodied.MobileManipulator.WarehouseMobileManipulator',
                           'site': 'Embodied.Scenario.WarehouseAisle'}}
