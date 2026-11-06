"""Match an evidence campaign to a gap, then price its conditional opportunity.

The economic calculation never closes a physical premise. A matching plan can
still fail, expose a counterexample, require redesign, or remain unevaluated.
"""
from __future__ import annotations

import hashlib
import json
from importlib.resources import files

from mlsysim.core.units import ureg
from mlsysim.fmt import fmt_qty, fmt_usd, _is_integer_like

_DATA = files('mlsysim.engine').joinpath('data/physical_studies/v4_frontier.json')
BOUNDARY = ('This browser study does not command hardware, establish WCET, validate braking, '
            'qualify the SO-101 station, or grant deployment permission.')


def _catalog():
    return json.loads(_DATA.read_text(encoding='utf-8'))


def handles(chapter, part):
    return int(chapter) == 18 and str(part).upper() == 'D'


def choices(chapter, part, track='warehouse'):
    if not handles(chapter, part) or track not in ('warehouse', 'warehouse_mobile_manipulator'):
        raise ValueError('Unsupported frontier investigation identity')
    return [{'id': case['id'], 'label': case['label']} for case in _catalog()['choices']]


def _display(value):
    if hasattr(value, 'units'):
        if value.units == ureg.USD:
            # fmt_usd owns currency formatting for Markdown; this evidence field
            # is rendered in a plain HTML cell, where the dollar escape is literal.
            return str(fmt_usd(value.to('USD').magnitude, precision=0)).replace(r'\$', '$')
        if value.units == ureg.USD / ureg.hour:
            return str(fmt_usd(value.to('USD/hour').magnitude, precision=0)).replace(r'\$', '$') + '/hour'
        return str(fmt_qty(value, value.units, precision=0 if _is_integer_like(value.magnitude) else 6,
                           trim_trailing_zeros=True))
    return value


def calculate(case, data):
    gap = data['gap_classes'][case['gap_class']]
    campaign = data['campaigns'][case['campaign']]
    economics = data['economics'][case['economics']]
    labor = campaign['labor_hours'] * campaign['labor_usd_per_hour'] * ureg.USD
    equipment = campaign['equipment_usd'] * ureg.USD
    campaign_cost = labor + equipment
    matched = case['campaign'] == gap['matching_campaign']
    retained = case['restriction'] == 'retained_until_closure'
    result = {
        'gap_class': case['gap_class'],
        'missing_evidence': gap['needed_evidence'],
        'selected_campaign': campaign['label'],
        'campaign_targets_missing_evidence': matched,
        'campaign_labor_cost': labor,
        'campaign_equipment_cost': equipment,
        'total_campaign_cost': campaign_cost,
        'restriction_retained_pending_actual_results': retained,
        'plan_supports_next_investigation': bool(matched and retained),
        'actual_closure_result': 'unknown; campaign is proposed, not executed',
        'restriction_may_be_retired_now': False,
        'physical_permission_granted': False,
    }
    demand = economics['demand_units_per_hour']
    value = economics['contribution_usd_per_unit']
    horizon = economics['horizon_hours'] * ureg.hour
    result['comparison_operating_horizon'] = horizon
    if demand is None:
        result.update({'lost_units_per_operating_hour': 'unknown: demand not supplied',
                       'restriction_opportunity_cost': 'unknown: demand not supplied',
                       'conditional_successful_closure_break_even': 'unknown: demand not supplied'})
    else:
        unrestricted_output = min(demand, economics['unrestricted_capacity_units_per_hour'])
        restricted_output = min(demand, economics['restricted_capacity_units_per_hour'])
        lost_rate = unrestricted_output-restricted_output
        result.update({'unrestricted_demand_limited_units_per_hour': unrestricted_output,
                       'restricted_demand_limited_units_per_hour': restricted_output,
                       'lost_units_per_operating_hour': lost_rate})
        if value is None:
            result.update({'restriction_opportunity_cost': 'unknown: per-unit contribution not supplied',
                           'conditional_successful_closure_break_even': 'unknown: per-unit contribution not supplied'})
        else:
            opportunity_rate = lost_rate * value * ureg.USD / ureg.hour
            result['restriction_opportunity_cost'] = (opportunity_rate*horizon).to('USD')
            result['lost_contribution_per_operating_hour'] = opportunity_rate
            result['conditional_successful_closure_break_even'] = ((campaign_cost/opportunity_rate).to('hour')
                if lost_rate > 0 and value > 0 else 'no finite payback from recovered throughput under these assumptions')
    # A wrong campaign does not buy the capability needed to recover the loss.
    if not matched:
        result['conditional_successful_closure_break_even'] = 'not applicable: this campaign does not target the missing evidence'
    return result


def study(chapter, part, choice, track='warehouse'):
    choices(chapter, part, track)
    data = _catalog()
    case = next((item for item in data['choices'] if item['id'] == choice), None)
    if case is None:
        raise ValueError(f'Unsupported frontier choice {choice!r}')
    gap = data['gap_classes'][case['gap_class']]
    economics = data['economics'][case['economics']]
    result = calculate(case, data)
    if not result['restriction_retained_pending_actual_results']:
        feedback = 'Funding a well-targeted campaign supplies no actual closure result. Removing the restriction before its closure evidence exists is unsupported regardless of the apparent economic return.'
    elif not result['campaign_targets_missing_evidence']:
        feedback = 'This campaign does not supply the missing kind of evidence. A lower price or additional successes cannot repair that mismatch; keep the restriction and choose a campaign that could change this premise.'
    else:
        feedback = 'The campaign targets the missing evidence while retaining the restriction. The price comparison is hypothetical and conditional on actual successful closure and separate approval. A counterexample, failed qualification or unknown result keeps the restriction.'
    trace = [{'event': 'gap', 'class': case['gap_class'], 'record': gap['gap']},
             {'event': 'evidence obligation', 'record': gap['needed_evidence']},
             {'event': 'campaign', 'record': data['campaigns'][case['campaign']]['label'],
              'matches_gap': result['campaign_targets_missing_evidence']},
             {'event': 'closure result', 'record': result['actual_closure_result']},
             {'event': 'current authority', 'restriction_retained': result['restriction_retained_pending_actual_results'],
              'retirement_supported': False}]
    cost = result.get('restriction_opportunity_cost')
    instrument = None
    if hasattr(cost, 'units'):
        instrument = {'kind': 'bar', 'label': 'Hypothetical campaign and restricted-operation opportunity',
                      'x': ['Campaign cost', 'Restricted contribution forgone over horizon'],
                      'y': [float(result['total_campaign_cost'].to('USD').magnitude), float(cost.to('USD').magnitude)],
                      'unit': 'USD',
                      'scope': 'Different cost categories under supplied demand and capacity, not stacked; no safety override or expected success value.'}
    inputs = dict(case) | {'economics_assumptions': economics,
                           'campaign_assumptions': data['campaigns'][case['campaign']]}
    return {'rows': [{'quantity': key.replace('_', ' '), 'value': _display(value)} for key, value in result.items()],
            'trace': trace, 'inputs': inputs,
            'conditions': [{'quantity': key.replace('_', ' '), 'value': value} for key, value in economics.items()] +
                          [{'quantity': 'cost scope', 'value': data['cost_scope']}],
            'feedback': feedback,
            'limitation': 'Four stipulated gap classes and declared hypothetical campaign costs. No campaign is performed, success probability estimated, proof checked, demand forecast validated or restriction closed. Conditional payback is measured after successful and separately approved closure; its attractive value never grants permission. ' + BOUNDARY,
            'brief': 'The warehouse restriction still stands. Decide which missing evidence the next campaign could supply, then compare its hypothetical resource cost with demand-limited contribution lost while the restriction remains. A plan and an actual closure result are different evidence.',
            'held_fixed': 'Within each wrong/right campaign pair, gap class, missing premise and economics stay fixed. The low-demand and missing-input cases separately change only the named economic premise; no case supplies a completed campaign result.',
            'source_kind': 'Constructed decision trace and hypothetical cost accounting; not measured hardware',
            'source': data['source'], 'version': data['version'], 'choice': choice,
            'prediction_prompt': 'What evidence could change this gap, and does an attractive payback close it?',
            'prediction_options': ['The campaign must match the missing evidence, and actual closure remains unknown.',
                                   'Enough nominal successes can create an absent authority path.',
                                   'A short conditional payback authorizes removal before results.'],
            'instrument': instrument,
            'provenance': {'case': 'constructed teaching case; not measured', 'model': 'gap_campaign_and_conditional_opportunity',
                           'catalog': 'mlsysim:engine/data/physical_studies/v4_frontier.json',
                           'catalog_sha256': hashlib.sha256(_DATA.read_bytes()).hexdigest(), 'version': data['version'],
                           'machine': 'Embodied.MobileManipulator.WarehouseMobileManipulator',
                           'site': 'Embodied.Scenario.WarehouseAisle'}}
