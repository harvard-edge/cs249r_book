"""Supervisor staffing queue under declared, illustrative M/M/1 assumptions.

The queue can refute a staffing promise. It cannot establish that a person will
respond before a physical deadline, or substitute for independent local fallback.
"""
from __future__ import annotations

import json
import math
from importlib.resources import files

from mlsysim.core.units import ureg
from mlsysim.physics import calc_queue_latency_mmc

_DATA = files('mlsysim.engine').joinpath('data/physical_studies/warehouse_oversight.json')


def catalog() -> dict:
    return json.loads(_DATA.read_text(encoding='utf-8'))


def calculate(p: dict) -> dict:
    """Return model-owned quantities for ``queue_case`` (default: reference).

    Unknown cases refuse rather than silently substituting a workload. The
    assistance window is a service target, explicitly separate from robot safety.
    """
    data = catalog()
    key = p.get('queue_case', 'reference')
    if key not in data['cases']:
        raise ValueError(f'Unknown supervisor queue case {key!r}')
    machines = data['cases'][key]['machines']
    arrival = machines * ureg.Quantity(data['help_rate_per_machine'])
    service = ureg.Quantity(data['mean_service_time'])
    window = ureg.Quantity(data['assistance_start_window'])
    offered = float((arrival * service).to('').magnitude)
    rho, p50, p99 = calc_queue_latency_mmc(arrival.to('Hz'), (1/service).to('Hz'), 1)
    stable = offered < 1
    result = {
        'machines_per_assistant': machines,
        'help_requests_per_machine': ureg.Quantity(data['help_rate_per_machine']),
        'fleet_help_arrival_rate': arrival.to('1/hour'),
        'mean_assistance_service_time': service,
        'offered_utilization': offered,
        'stationary_queue_exists': stable,
        'assistance_start_service_target': window,
        'window_basis': data['window_basis'],
        'queue_assumptions': data['provenance']['assumptions'],
        'local_fallback_required': True,
        'physical_response_deadline_proven': False,
    }
    if stable:
        mean_wait = rho * service / (1-rho)
        # M/M/1 unconditional queue tail includes the atom at zero wait.
        miss_probability = rho * math.exp(-float(((1/service-arrival)*window).to('').magnitude))
        result.update({
            'mean_wait_before_assistance': mean_wait.to('second'),
            'p50_wait_before_assistance': p50.to('second'),
            'p99_wait_before_assistance': p99.to('second'),
            'probability_queue_wait_exceeds_service_target': miss_probability,
            'p99_within_assistance_service_target': bool(p99 <= window),
        })
    else:
        # No finite stationary percentile: serialize the refusal explicitly,
        # without JSON Infinity or a misleading zero-valued plotted substitute.
        result.update({
            'mean_wait_before_assistance': 'unbounded; stationary queue does not exist',
            'p50_wait_before_assistance': 'unbounded; stationary queue does not exist',
            'p99_wait_before_assistance': 'unbounded; stationary queue does not exist',
            'probability_queue_wait_exceeds_service_target': 'not defined for a stationary queue',
            'p99_within_assistance_service_target': False,
        })
    return result


def instrument(result: dict) -> dict | None:
    """Finite wait/target comparison only; overloaded queues remain refusals."""
    if not result.get('stationary_queue_exists'):
        return None
    keys = ['mean_wait_before_assistance', 'p99_wait_before_assistance', 'assistance_start_service_target']
    return {'kind': 'bar', 'label': 'Staffing load changes waiting time',
            'x': [key.replace('_', ' ') for key in keys],
            'y': [float(result[key].to('second').magnitude) for key in keys],
            'unit': 'second',
            'scope': 'Illustrative stationary M/M/1 queue; waits exclude service and takeover. No physical response guarantee.'}
