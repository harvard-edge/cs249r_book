"""Reproducible physical-AI companion investigations.

A choice selects a declared hypothetical case or supplied record, never a hidden
quality score. Numerical cases use the canonical warehouse machine and robotics
models. Record reviews evaluate a supplied policy, not physical qualification.
"""
from __future__ import annotations

import copy
import json
import hashlib
import math
from importlib.resources import files
from typing import Any

from mlsysim.core.units import ureg, BYTES_FP16
from mlsysim.physics.transformer import calc_vla_inference_latency
from mlsysim.embodied.registry import Embodied
from mlsysim.fmt import fmt_qty, _is_integer_like
from mlsysim.physics.robotics import (
    calc_safe_stopping_distance, calc_max_permitted_velocity, calc_c2_stop_suffix, calc_scurve_stop_distance,
    calc_scurve_stop_time, calc_reflected_inertia, calc_voltage_droop,
    calc_seam_acceleration_torque_jump, calc_latch_contact_stopping_profile,
    calc_transient_impact_force, calc_transient_contact_speed_limit,
    calc_watchdog_lease_bound,
    calc_phase_lag_total,
    calc_chunk_renewal_schedule, calc_tracked_evidence_horizon, calc_empirical_testing_exposure,
    calc_clopper_pearson_zero_failure_bound,
    calc_spatial_information_age_displacement,
    calc_free_space_expiry, calc_target_evidence_horizon,
    calc_cbf_qp_orthogonal_projection, calc_axi_fifo_contention,
)

_DATA = files('mlsysim.engine').joinpath('data/physical_studies/warehouse_investigations.json')
VERSION = 'warehouse-investigations.v2'
BOUNDARY = ('Analytical reference or explicitly constructed evidence only. '
            'This browser study does not command hardware, establish WCET, validate braking, '
            'qualify the SO-101 station, or grant deployment permission.')


def _catalog() -> dict:
    return json.loads(_DATA.read_text(encoding='utf-8'))


def _supporting_data(chapter: int, part: str) -> list:
    names = []
    if int(chapter) == 9 and str(part).upper() == 'B':
        names.append('warehouse_occupancy.json')
    if int(chapter) == 17 and str(part).upper() == 'B':
        names.append('warehouse_oversight.json')
    return [files('mlsysim.engine').joinpath('data/physical_studies/' + name) for name in names]


def _part(chapter: int, part: str, track: str) -> dict:
    if track not in ('warehouse', 'warehouse_mobile_manipulator'):
        raise ValueError('Physical studies support the warehouse reference, not a substituted deployment track.')
    key = f'{int(chapter):02d}_{str(part).upper()}'
    try:
        return _catalog()['parts'][key]
    except KeyError as exc:
        raise ValueError(f'Unknown physical investigation {key}') from exc


def choices(chapter: int, part: str, track: str = 'warehouse') -> list[dict[str, str]]:
    """Available interventions, in display order; IDs are stable export keys."""
    spec = _part(chapter, part, track)
    from . import v4_timing, v4_estimation, v4_intervention, v4_training, v4_frontier
    for extension in (v4_timing, v4_estimation, v4_intervention, v4_training, v4_frontier):
        if extension.handles(chapter, part):
            return extension.choices(chapter, part, track)
    if int(chapter) == 17 and str(part).upper() == 'C':
        from . import v4_subgroups
        return v4_subgroups.choices(chapter, part, track)
    result = [{'id': c['id'], 'label': c['label']} for c in spec['choices']]
    if int(chapter) == 12 and str(part).upper() == 'C':
        from . import v4_compliance
        result += v4_compliance.choices()
    return result


def _flat(value: Any) -> Any:
    if hasattr(value, 'units'):
        magnitude = float(value.magnitude)
        # Explicit display precision preserves small causal differences; the
        # canonical formatter still owns unit labels and trailing-zero rules.
        precision = 0 if _is_integer_like(magnitude) else max(3, 2-math.floor(math.log10(abs(magnitude))))
        return str(fmt_qty(value, value.units, precision=precision, trim_trailing_zeros=True))
    if isinstance(value, dict):
        return {k: _flat(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_flat(v) for v in value]
    return value


def _calculate(model: str, p: dict) -> dict:
    machine = Embodied.MobileManipulator.WarehouseMobileManipulator
    site = Embodied.Scenario.WarehouseAisle
    def q(key):
        return ureg.Quantity(p[key])
    if model == 'stopping':
        speed = site.v_aisle * p['speed_fraction']
        delay = site.tau_delay * p['delay_fraction']
        a = site.a_brake * p['deceleration_fraction']
        distance = calc_safe_stopping_distance(speed, delay, a, site.delta_loc, site.delta_margin + site.eps_track)
        return {'required_distance': distance, 'available_clearance': site.d_clear,
                'reaction_distance': (speed * delay).to('m'), 'braking_distance': (speed**2/(2*a)).to('m'),
                'remaining_clearance': site.d_clear - distance, 'within_reference_clearance': bool(distance <= site.d_clear)}
    if model == 'body':
        result = _calculate('stopping',p)
        speed=site.v_aisle*p['speed_fraction']
        delay=site.tau_delay*p['delay_fraction']
        result['elementary_speed_ceiling']=calc_max_permitted_velocity(site.d_clear,delay,site.a_brake,site.delta_loc,site.delta_margin+site.eps_track)
        result['freshness_spatial_allowance']=site.delta_loc
        result['age_travel']=calc_spatial_information_age_displacement(speed,site.perception_age*p['age_fraction'])
        result['freshness_within_allowance']=bool(result['age_travel']<=site.delta_loc)
        result['torque_thermal_installed_qualification']='unknown; no installed measurement supplied'
        return result
    if model == 'gearing':
        return {'reflected_rotor_inertia': calc_reflected_inertia(p['gear_ratio'], q('rotor_inertia')),
                'ideal_output_torque': q('motor_torque') * p['gear_ratio'] * p['efficiency']}
    if model == 'body_duty':
        # First-order, fixed-resistance thermal reference: C*dT/dt = I²R -
        # (T-Ta)/theta. The closed-form update is exact for each constant-current
        # segment. Installed cooling/capacitance are hypothetical case inputs,
        # not properties inferred for the actual Panda motor or sealed housing.
        from mlsysim.actuators.registry import Actuators
        from mlsysim.physics.constants import INSULATION_CLASS_F_LIMIT
        motor=Actuators.FramelessMotor.TQ_ILM70x18
        joint=Actuators.PandaClassJoint.CSG_25_50_ILM70x18
        installation=_catalog()['thermal_installation']
        ambient_magnitude, ambient_unit = installation['ambient'].split(maxsplit=1)
        # Offset units must be constructed, not parsed as magnitude*unit:
        # newer Pint versions reject multiplication by degree_Celsius.
        ambient=ureg.Quantity(float(ambient_magnitude), ambient_unit).to('K')
        theta=ureg.Quantity(installation['thermal_resistance'])
        capacity=ureg.Quantity(installation['thermal_capacitance'])
        tau=(theta*capacity).to('s')
        rated_loss=(motor.rated_current**2*joint.internal_resistance).to('W')
        loss=rated_loss*installation['burst_current_factor']**2
        initial=ambient+rated_loss*theta if p['hot_start'] else ambient
        temp=initial
        on=ureg.Quantity(installation['on_time'])
        off=ureg.Quantity(p['off_time'])
        trace=[]
        elapsed=0*ureg.s
        peak=temp
        for cycle in range(installation['cycles']):
            equilibrium=ambient+loss*theta
            temp=equilibrium+(temp-equilibrium)*math.exp(-(on/tau).to('').magnitude)
            elapsed+=on
            peak=max(peak,temp)
            trace.append({'cycle':cycle+1,'event':'end of burst','elapsed':elapsed,'winding_temperature':temp})
            temp=ambient+(temp-ambient)*math.exp(-(off/tau).to('').magnitude)
            elapsed+=off
            trace.append({'cycle':cycle+1,'event':'end of recovery','elapsed':elapsed,'winding_temperature':temp})
        return {**_calculate('voltage',p),'initial_winding_temperature':initial,
            'peak_winding_temperature':peak,'insulation_temperature_limit':INSULATION_CLASS_F_LIMIT.to('K'),
            'peak_temperature_reserve':INSULATION_CLASS_F_LIMIT.to('K')-peak,
            'fixed_resistance_peak_copper_loss':loss, 'thermal_time_constant':tau,
            'modeled_thermal_trace':trace}
    if model == 'voltage':
        droop = calc_voltage_droop(site.i_rail_transient * p['current_fraction'], machine.drive_bus_resistance)
        remaining = machine.drive_bus_battery_low - droop
        return {'resistive_droop': droop, 'depleted_bus_after_droop': remaining,
                'converter_uvlo': machine.drive_bus_uvlo, 'above_uvlo': bool(remaining > machine.drive_bus_uvlo)}
    if model == 'vla':
        weights = machine.intent_weight_memory if p['precision']=='reference' else (machine.intent_model.parameters*BYTES_FP16/ureg.param).to('GB')
        soc=machine.brain_soc
        result=calc_vla_inference_latency(machine.intent_model.parameters,weights,
            round(machine.intent_input_tokens*p['input_fraction']),machine.intent_model.action_dim,
            soc.compute.precision_flops['fp16'],soc.memory.bandwidth,
            bus_efficiency=machine.dram_efficiency,compute_utilization=machine.intent_compute_utilization,
            parallel_decode=False,n_layers=machine.intent_model.layers,hidden_dim=machine.intent_model.hidden_dim)
        result['stored_weights']=weights
        return result
    if model == 'chunk_supply':
        result = calc_chunk_renewal_schedule(machine.chunk_rate*p.get('renewal_fraction',1), machine.chunk_step,
            p.get('horizon_override', round(machine.chunk_horizon*p.get('horizon_fraction',1))), machine.chunk_inference_latency)
        result['untracked_evidence_horizon'] = calc_target_evidence_horizon(site.grasp_tolerance, site.grasp_initial_error, site.v_conveyor)
        result['tracked_evidence_horizon'] = calc_tracked_evidence_horizon(site.grasp_tolerance, site.grasp_initial_error, site.a_conveyor_slip)
        return result
    if model == 'lease':
        raw = calc_watchdog_lease_bound(site.d_clear - site.fixed_overhead, site.v_aisle, site.a_brake)
        reserve = site.perception_age + site.t_tick + site.t_brake_onset + site.t_bus
        raw.pop('t_lease_max')  # Legacy alias is a total-delay ceiling, never a lease.
        raw['perception_age_reserved'] = site.perception_age
        raw['nonlease_delay_reserved'] = reserve
        raw['remaining_lease_budget'] = raw['t_total_delay_max'] - reserve
        raw['chosen_lease'] = site.t_lease * p['lease_fraction']
        raw['within_ideal_budget'] = bool(raw['chosen_lease'] <= raw['remaining_lease_budget'])
        return raw
    if model == 'phase':
        return calc_phase_lag_total(q('crossover'), q('sample_period'), q('compute_delay'), q('transport_delay'), q('phase_allowance'))
    if model == 'zero_failure':
        return {'independent_zero_failure_trials': p['trials'], 'confidence': p['confidence'],
                'success_probability_lower_bound': calc_clopper_pearson_zero_failure_bound(p['trials'], p['confidence']),
                'separate_poisson_zero_event_exposure': calc_empirical_testing_exposure(p['hazard_target_per_hour'], p['confidence'])}
    if model == 'age':
        age = site.perception_age * p['age_fraction']
        return {'observation_age': age, 'unmodeled_displacement': calc_spatial_information_age_displacement(site.v_aisle, age)}
    if model == 'expiry':
        horizon = calc_free_space_expiry(site.d_clear, site.delta_loc, site.delta_margin, site.v_human_approach * p['approach_fraction'])
        age = ureg.Quantity(p.get('evidence_age','0 ms'))
        remaining = max(horizon-age,0*ureg.ms)
        return {'free_space_validity': horizon, 'original_evidence_age': age,
            'remaining_validity':remaining,'publication_action':p.get('publication_action','unchanged'),
            'publication_does_not_renew_evidence': True}
    if model == 'target':
        untracked=calc_target_evidence_horizon(site.grasp_tolerance, site.grasp_initial_error, site.v_conveyor*p['drift_fraction'])
        tracked=calc_tracked_evidence_horizon(site.grasp_tolerance, site.grasp_initial_error, site.a_conveyor_slip)
        active=tracked if p.get('tracked',False) else untracked
        remaining=max(active-site.t_capture_to_admission,0*ureg.ms)
        return {'untracked_target_horizon':untracked, 'tracked_target_horizon':tracked,
            'target_evidence_validity':active, 'capture_to_admission_allocation':site.t_capture_to_admission,
            'target_validity_remaining_at_admission':remaining, 'requested_execution_lease':site.t_lease,
            'lease_within_remaining_target_validity':bool(site.t_lease<=remaining)}
    if model == 'collection':
        ledger = _catalog()['collection_ledger']
        selected = [a for a in ledger if a['id'] in p['attempt_ids']]
        return {'attempted_episodes': len(selected),
            'retained_episodes': sum(a['retained'] for a in selected),
            'supported_cells': sorted({a['cell'] for a in selected if a['retained']}),
            'missing_declared_cells': sorted(set(_catalog()['declared_collection_cells']) - {a['cell'] for a in selected if a['retained']}),
            'cell_time': sum(a['cell_minutes'] for a in selected)*ureg.minute,
            'reset_time': sum(a['reset_minutes'] for a in selected)*ureg.minute,
            'mechanical_cycles': sum(a['wear_cycles'] for a in selected),
            'attempt_ledger': selected}
    if model == 'occupancy':
        from .v4_occupancy import calculate
        return calculate(p)
    if model == 'seam':
        from mlsysim.actuators.registry import Actuators
        joint = Actuators.PandaClassJoint.CSG_25_50_ILM70x18
        jump = calc_seam_acceleration_torque_jump(joint.gear_ratio, joint.rotor_inertia, q('acceleration_jump'))
        stop = _calculate('scurve',p)
        return {'reflected_rotor_torque_jump':jump, 'seam_class':p['seam_class'], **stop}
    if model == 'latch_contact':
        speed = site.v_latch_approach * p['speed_fraction']
        result = calc_latch_contact_stopping_profile(speed, scenario=site)
        hand_speed = site.v_handover * p['handover_fraction']
        result['transient_hand_force'] = calc_transient_impact_force(hand_speed, machine.arm_effective_contact_mass,
            site.k_contact_human, human_mass=site.m_contact_human)
        result['hand_force_criterion'] = site.f_contact_criterion
        result['hand_speed_ceiling'] = calc_transient_contact_speed_limit(site.f_contact_criterion,
            machine.arm_effective_contact_mass, site.k_contact_human, human_mass=site.m_contact_human)
        return result
    if model == 'scurve':
        speed = site.v_aisle * p['speed_fraction']
        jerk = calc_c2_stop_suffix(machine.base.max_velocity, site.a_brake)['peak_jerk'] * p['jerk_fraction']
        return {'braking_distance': calc_scurve_stop_distance(speed, site.a_brake, jerk),
                'braking_time': calc_scurve_stop_time(speed, site.a_brake, jerk), 'jerk_bound': jerk}
    if model == 'envelope':
        speed = site.v_aisle * p['speed_fraction']
        jerk = calc_c2_stop_suffix(machine.base.max_velocity, site.a_brake)['peak_jerk'] * p['jerk_fraction']
        braking = calc_scurve_stop_distance(speed, site.a_brake, jerk)
        drift = speed * site.t_capture_to_brake_onset
        needed = braking + drift + site.fixed_overhead
        return {'jerk_limited_braking': braking, 'prebrake_travel': drift.to('m'),
            'capture_to_dispatch':site.perception_age, 'dispatch_to_admission':site.t_dispatch_to_admission,
            'admission_to_brake_onset':site.t_admission_to_brake_onset,
            'fixed_overhead': site.fixed_overhead, 'required_distance': needed.to('m'),
            'clearance': site.d_clear, 'remaining_clearance': (site.d_clear-needed).to('m'),
            'within_reference_clearance': bool(needed <= site.d_clear)}
    if model == 'projection':
        return calc_cbf_qp_orthogonal_projection(p['proposal'], p['normal'], p['bound'])
    if model == 'axi':
        return calc_axi_fifo_contention(tile_bytes=p['tile'], camera_bytes=p['camera'], telemetry_bytes=p['telemetry'],
            safety_read_bytes=p['read'], transaction_bytes=p['transaction'], bandwidth=q('bandwidth'),
            bank_switches=p['switches'], bank_switch_time=q('switch_time'), compute_time=q('compute'))
    if model == 'takeover':
        human = site.t_takeover_in_loop if p['engaged'] else site.t_takeover_out_of_loop
        speed = site.v_takeover_in_loop if p['slow'] else site.v_aisle
        jerk = calc_c2_stop_suffix(machine.base.max_velocity, site.a_brake)['peak_jerk']
        needed = speed * (site.tau_delay + human + site.t_dispatch_to_admission) + calc_scurve_stop_distance(speed, site.a_brake, jerk) + site.fixed_overhead
        return {'human_and_transport_time': human, 'chosen_speed': speed, 'required_distance': needed.to('m'),
                'available_clearance': site.d_clear, 'within_reference_clearance': bool(needed <= site.d_clear)}
    raise ValueError(f'Unknown physical model {model}')


def study(chapter: int, part: str, choice: str, track: str = 'warehouse') -> dict:
    """Evaluate one declared case, returning exportable evidence with its limits."""
    spec = _part(chapter, part, track)
    from . import v4_timing, v4_estimation, v4_intervention, v4_training, v4_frontier
    for extension in (v4_timing, v4_estimation, v4_intervention, v4_training, v4_frontier):
        if extension.handles(chapter, part):
            return extension.study(chapter, part, choice, track)
    if int(chapter) == 17 and str(part).upper() == 'C':
        from . import v4_subgroups
        return v4_subgroups.study(chapter, part, choice, track)
    if int(chapter) == 12 and str(part).upper() == 'C':
        from . import v4_compliance
        if choice in {case['id'] for case in v4_compliance.choices()}:
            return v4_compliance.study(choice)
    selected = next((c for c in spec['choices'] if c['id'] == choice), None)
    if selected is None:
        raise ValueError(f'Unsupported choice {choice!r} for {chapter}/{part}')
    p = copy.deepcopy(selected.get('inputs', {}))
    trace = copy.deepcopy(selected.get('trace', []))
    if spec['model'] == 'record_review':
        # A record contains named policy requirements and observed evidence, no
        # manufactured physical results. Missing evidence stays explicitly unknown.
        requirements = selected['requirements']
        rows = []
        for rule in spec['policy']:
            observed = requirements.get(rule['field'])
            matched = _check_rule(observed, rule)
            rows.append({'requirement': rule['field'], 'supplied_evidence': _flat(observed) if observed is not None and observed != '' else 'not supplied',
                'acceptance_condition': _flat(rule.get('equals', {'operator':rule.get('operator'), 'expected':rule.get('expected')})),
                'policy_status': 'supports stated claim' if matched else ('unknown: required evidence absent' if observed is None or observed == '' else 'does not support stated claim')})
        if int(chapter) == 17 and str(part).upper() == 'B':
            from . import v4_oversight
            result = v4_oversight.calculate(p)
            rows += [{'requirement': key.replace('_', ' '), 'supplied_evidence': _flat(value),
                      'acceptance_condition': 'Declared analytical queue assumption or consequence',
                      'policy_status': 'Constructed queue calculation; not a physical response guarantee'} for key, value in result.items()]
        complete = all(row['policy_status'] == 'supports stated claim' for row in rows[:len(spec['policy'])])
        if spec.get('oracle_type') == 'fault_trial':
            rows.append({'requirement':'computed four-state oracle','supplied_evidence':_fault_verdict(requirements),
                'acceptance_condition':'Precommitted oracle with independent injection and margin witnesses',
                'policy_status':'Classifies this constructed trial only; no physical permit'})
        feedback = selected['feedback'] + (' This packet matches the displayed documentary policy; its physical truth remains unverified.' if complete else ' The displayed policy does not establish the stated claim from this packet; absence or contradiction is not a passing result.')

    else:
        try:
            result = _calculate(spec['model'], p)
        except ValueError as exc:
            # Invalid schedules remain inspectable student evidence. API identity
            # errors were already rejected before entering the model.
            result = {'model_refusal': str(exc)}
        rows = [{'quantity': k.replace('_', ' '), 'value': _flat(v)} for k, v in result.items() if k not in ('attempt_ledger','occupancy_events','modeled_thermal_trace')]
        trace += _flat(result.get('attempt_ledger', result.get('occupancy_events', result.get('modeled_thermal_trace', []))))
        feedback = selected['feedback'] + ' Inspect the reported values and any failed reference budget before deciding what to retain.'
    return {'rows': rows, 'feedback': feedback, 'limitation': spec['limitation'] + ' ' + BOUNDARY,
            'inputs': p, 'conditions':_conditions(spec['model'],p), 'trace': trace, 'version': VERSION,
            'brief': spec['brief'], 'held_fixed': spec['held_fixed'], 'source_kind': spec['source_kind'],
            'prediction_prompt': spec.get('prediction_prompt', 'Which consequence would distinguish these cases for the stated chapter decision?'),
            'prediction_options': spec.get('prediction_options', ['The selected case supports the same claim under the same conditions.', 'The selected case changes a specific premise or cost; the claim needs to change.', 'The supplied evidence cannot yet establish that claim.']),
            'instrument': (v4_oversight.instrument(result) if int(chapter) == 17 and str(part).upper() == 'B' else _instrument(spec['model'], locals().get('result'))),
            'source': spec['source'], 'provenance': {'case': 'constructed teaching case; not measured',
                'catalog': 'mlsysim:engine/data/physical_studies/warehouse_investigations.json',
                'catalog_sha256':hashlib.sha256(_DATA.read_bytes()+b''.join(path.read_bytes() for path in _supporting_data(chapter, part))).hexdigest(),
                'supporting_catalogs': ['mlsysim:engine/data/physical_studies/' + path.name for path in _supporting_data(chapter, part)],
                'version': VERSION, 'model': spec['model'],
                'machine': 'Embodied.MobileManipulator.WarehouseMobileManipulator',
                'site': 'Embodied.Scenario.WarehouseAisle'}, 'choice': choice}


def _instrument(model: str, result: dict | None) -> dict | None:
    """Plot only comparable model-owned quantities with one declared unit."""
    if not result:
        return None
    selections = {
        'stopping': ('Stopping clearance components', 'm', ['reaction_distance', 'braking_distance', 'remaining_clearance']),
        'envelope': ('Full reference stopping allocation', 'm', ['jerk_limited_braking', 'prebrake_travel', 'fixed_overhead', 'remaining_clearance']),
        'age': ('Travel while evidence ages', 'm', ['unmodeled_displacement']),
        'vla': ('Fresh intent proposal cost decomposition', 'ms', ['t_weight_sweep','t_prefill','t_decode','t_total']),
        'body': ('Freshness displacement and allowance', 'm', ['age_travel','freshness_spatial_allowance']),
        'voltage': ('Depleted bus under a resistive transient', 'V', ['resistive_droop','depleted_bus_after_droop','converter_uvlo']),
        'body_duty': ('Repeated-burst thermal reserve', 'K', ['initial_winding_temperature','peak_winding_temperature','insulation_temperature_limit','peak_temperature_reserve']),
        'chunk_supply': ('Supply, renewal and evidence are different clocks', 'ms', ['chunk_span', 'renewal_period', 'open_loop_span', 'untracked_evidence_horizon', 'tracked_evidence_horizon']),
        'scurve': ('Jerk-limited stop distance', 'm', ['braking_distance']),
        'seam': ('Rotor contribution to seam torque jump', 'N*m', ['reflected_rotor_torque_jump']),
        'latch_contact': ('Contact force comparison', 'N', ['f_at_brake_onset','transient_hand_force','hand_force_criterion']),
        'lease': ('Total delay ceiling and lease allocation', 'ms', ['t_total_delay_max', 'nonlease_delay_reserved', 'remaining_lease_budget', 'chosen_lease']),
        'axi': ('Constructed FIFO and priority bypass', 'us', ['total_fifo', 'total_priority_bypass']),
        'takeover': ('Takeover clearance requirement', 'm', ['required_distance', 'available_clearance']),
        'phase': ('Loop phase consumed', 'degree', ['phase_zoh', 'phase_compute', 'phase_transport', 'phase_total']),
        'target': ('Target evidence, replacement and lease clocks', 'ms', ['untracked_target_horizon','tracked_target_horizon','capture_to_admission_allocation','target_validity_remaining_at_admission','requested_execution_lease']),
        'expiry': ('Original evidence age and remaining validity', 'ms', ['free_space_validity','original_evidence_age','remaining_validity']),
    }
    if model == 'occupancy':
        from .v4_occupancy import instrument
        return instrument(result)
    if 'model_refusal' in result or model not in selections:
        return None
    title, unit, keys = selections[model]
    return {'kind': 'bar', 'label': title, 'x': [key.replace('_', ' ') for key in keys],
        'y': [float(result[key].to(unit).magnitude) for key in keys], 'unit': unit,
        'scope': 'Analytical reference; not physical validation. Distinct allocations are shown separately, not stacked.'}


def _check_rule(observed: Any, rule: dict) -> bool:
    """Small explicit policy algebra for supplied classroom facts, not certification."""
    if observed is None:
        return False
    if 'equals' in rule:
        return observed == rule['equals']
    expected = rule.get('expected')
    operator = rule.get('operator')
    if operator == 'eq':
        return observed == expected
    if operator in ('lte','gte'):
        if not isinstance(observed,(int,float)) or isinstance(observed,bool):
            return False
        return observed <= expected if operator == 'lte' else observed >= expected
    if operator == 'unique':
        return isinstance(observed,list) and len(observed) == len(set(observed)) and (expected is None or len(observed)==expected)
    if operator == 'in':
        return observed in expected
    if operator == 'subset':
        return isinstance(observed,list) and set(observed).issubset(set(expected))
    raise ValueError(f'Unknown supplied policy operator {operator!r}')


def _conditions(model: str, p: dict) -> list[dict]:
    """Display starting physical inputs separately from computed evidence."""
    site=Embodied.Scenario.WarehouseAisle
    machine=site.machine
    conditions={}
    if model in ('stopping','body','envelope','takeover','lease'):
        conditions={'reference_clearance':site.d_clear, 'reference_aisle_speed':site.v_aisle,
                    'reference_deceleration':site.a_brake, 'reference_total_prebrake_delay':site.tau_delay,
                    'fixed_spatial_allowance':site.fixed_overhead}
    elif model in ('age','expiry','target','chunk_supply'):
        conditions={'capture_to_dispatch_age':site.perception_age, 'reference_aisle_speed':site.v_aisle,
                    'target_tolerance':site.grasp_tolerance,'initial_target_error':site.grasp_initial_error,
                    'reference_chunk_horizon':machine.chunk_horizon,'reference_chunk_step':machine.chunk_step,
                    'reference_chunk_renewal_rate':machine.chunk_rate}
    elif model == 'latch_contact':
        conditions={'latch_approach_speed':site.v_latch_approach,'latch_stiffness':site.k_latch,
                    'contact_response':site.t_contact_response,'latch_force_limit':site.f_latch_limit,
                    'handover_speed':site.v_handover,'hand_force_criterion':site.f_contact_criterion}
    conditions.update(p)
    return [{'quantity':key.replace('_',' '),'value':_flat(value)} for key,value in conditions.items()]


def _fault_verdict(facts: dict) -> str:
    """Classify a supplied fault-trial record before any aggregate statistics."""
    if (facts.get('injection_witness') != 'achieved' or facts.get('preconditions') != 'matched'
            or facts.get('oracle_epoch') != 'before_trial'):
        return 'INVALID'
    margins=[facts.get('onset_margin'),facts.get('distance_margin')]
    if any(margin not in ('positive','zero','negative') for margin in margins):
        return 'UNOBSERVABLE'
    return 'FAIL' if 'negative' in margins else 'PASS'
