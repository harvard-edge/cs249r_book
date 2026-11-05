"""Four-state spatial memory and query-dependent representation accounting.

Constructed evidence epochs and explicit record-size/read-count assumptions
support a bounded comparison; no sensor, tracking or WCET claim is inferred.
"""
from __future__ import annotations

import copy
import json
from importlib.resources import files

from mlsysim.core.units import ureg

_DATA = files('mlsysim.engine').joinpath('data/physical_studies/warehouse_occupancy.json')


def catalog() -> dict:
    return json.loads(_DATA.read_text(encoding='utf-8'))


def calculate(p: dict) -> dict:
    data = catalog()
    representation = p.get('representation', 'dense')
    query = p.get('query', 'clearance')
    if representation not in data['representations'] or query not in data['queries']:
        raise ValueError('Unknown representation or consumer query')
    # Original confirmation comparison retains exactly the same event counts.
    trace = copy.deepcopy(data['confirmation_trace'])
    tentative = p['tentative_visible']
    unknown_refused = p['unknown_refused']
    for event in trace:
        visible = event['range_hit'] and (tentative or event['track_confirmed'])
        event['visible_to_clearance'] = visible
        event['permission_result'] = 'REFUSE occupied' if visible else ('REFUSE unknown' if unknown_refused else 'UNSUPPORTED clear')
    counts = {
        'events_with_unreported_obstacle': sum(e['range_hit'] and not e['visible_to_clearance'] for e in trace),
        'unsupported_clear_events': sum(e['permission_result'] == 'UNSUPPORTED clear' for e in trace),
    }
    now = ureg.Quantity(data['evaluation_epoch'])
    states = []
    for cell in data['cells']:
        if cell['capture_epoch'] is None:
            state = 'unknown'
            age = 'no observation'
        else:
            age = now - ureg.Quantity(cell['capture_epoch'])
            expired = age >= ureg.Quantity(cell['valid_for'])
            state = 'expired' if expired else ('occupied' if cell['range_hit'] else ('observed free' if cell['observed_free'] else 'unknown'))
        states.append({'event': 'map state', 'cell': cell['cell'], 'state': state,
                       'evidence_age': str(age),
                       'permission_result': 'ELIGIBLE for further clearance checks' if state == 'observed free' else f'REFUSE {state}'})
    # Explicit constructed formats. Neither bytes nor operation counts imply
    # runtime timing; query compatibility is independent of fitting in memory.
    dense = data['representations']['dense']
    cell_count = int(round((ureg.Quantity(dense['volume']) / ureg.Quantity(dense['cell_side'])**3).to('').magnitude))
    dense_storage = cell_count * dense['bytes_per_cell'] * ureg.byte
    graph = data['representations']['semantic']
    graph_storage = (graph['nodes']*graph['bytes_per_node'] + graph['edges']*graph['bytes_per_edge']) * ureg.byte
    chosen = data['representations'][representation]
    reads = chosen['query_reads'].get(query)
    supported = reads is not None
    read_budget = data['consumer_read_budget']
    storage = dense_storage if representation == 'dense' else graph_storage
    return {**counts,
        'dense_cell_count': cell_count,
        'dense_declared_storage': dense_storage,
        'semantic_declared_storage': graph_storage,
        'selected_representation': chosen['label'],
        'selected_storage': storage,
        'stored_fields': chosen['stored_fields'],
        'consumer_query': data['queries'][query],
        'query_supported_by_stored_fields': supported,
        'constructed_records_read': reads if supported else 'unsupported query; no valid read count',
        'declared_consumer_record_read_budget': read_budget,
        'within_declared_read_budget': bool(supported and reads <= read_budget),
        'runtime_query_deadline_established': False,
        'occupancy_events': trace + states,
        'scope': data['provenance']['scope'],
    }


def instrument(result: dict) -> dict:
    return {'kind': 'bar', 'label': 'Storage for the declared record formats',
            'x': ['Local occupancy grid', 'Object-relation graph'],
            'y': [float(result[key].to('byte').magnitude) for key in ('dense_declared_storage', 'semantic_declared_storage')],
            'unit': 'byte',
            'scope': 'Constructed record formats answer different queries; smaller storage does not establish clearance or timing.'}
