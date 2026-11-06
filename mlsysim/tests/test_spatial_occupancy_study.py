"""No semantic confirmation or storage saving can manufacture free space."""
import json

import pytest

from mlsysim.labs.experiments.v4_occupancy import calculate, instrument


def case(**overrides):
    return calculate({'tentative_visible': True, 'unknown_refused': True, **overrides})


def test_original_confirmed_only_counts_remain_and_tentative_occupancy_repairs_them():
    a = case(tentative_visible=False, unknown_refused=False)
    b = case()
    c = case(tentative_visible=False)
    assert a['events_with_unreported_obstacle'] == 2
    assert a['unsupported_clear_events'] == 2
    assert b['events_with_unreported_obstacle'] == 0
    assert b['unsupported_clear_events'] == 0
    assert c['events_with_unreported_obstacle'] == 2
    assert c['unsupported_clear_events'] == 0


def test_four_states_distinguish_free_from_unknown_and_expired():
    a = case()
    states = {row['cell']: row for row in a['occupancy_events'] if row['event'] == 'map state'}
    assert {row['state'] for row in states.values()} == {'occupied', 'observed free', 'unknown', 'expired'}
    assert states['observed-corridor']['permission_result'] == 'ELIGIBLE for further clearance checks'
    assert states['old-corridor']['permission_result'] == 'REFUSE expired'
    assert states['occluded-corner']['permission_result'] == 'REFUSE unknown'


def test_grid_storage_uses_volume_and_cubic_resolution_not_arbitrary_score():
    a = case()
    assert a['dense_cell_count'] == 32000
    assert a['dense_declared_storage'].to('byte').magnitude == 32000 * 8
    assert a['semantic_declared_storage'].to('byte').magnitude == 40 * 64 + 80 * 32
    assert a['constructed_records_read'] == 12
    assert a['within_declared_read_budget']
    assert not a['runtime_query_deadline_established']


def test_query_reversal_follows_fields_not_a_universal_best_representation():
    grid_clear = case()
    graph_clear = case(representation='semantic')
    grid_relation = case(query='relation')
    graph_relation = case(representation='semantic', query='relation')
    assert grid_clear['query_supported_by_stored_fields']
    assert not graph_clear['query_supported_by_stored_fields']
    assert not grid_relation['query_supported_by_stored_fields']
    assert graph_relation['query_supported_by_stored_fields']
    assert graph_clear['selected_storage'] < grid_clear['selected_storage']
    assert not graph_clear['within_declared_read_budget']
    assert graph_clear['occupancy_events'] == grid_clear['occupancy_events']
    assert graph_relation['constructed_records_read'] == 5
    assert len(instrument(grid_clear)['y']) == 2
    json.dumps(grid_clear, default=str, allow_nan=False)
    with pytest.raises(ValueError, match='Unknown representation'):
        case(representation='mystery')
