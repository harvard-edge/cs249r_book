"""Evidence attachments follow both sides of a Physical AI comparison."""
from copy import deepcopy

from mlsysbook_labs.physical_companion import _signature
from mlsysim.labs.experiments.v4_studies import choices, study


def _case(index):
    selected = choices(1, "B")[index]["id"]
    return {"case": selected, "result": study(1, "B", selected)}


def test_repin_invalidates_saved_attachment_without_changing_trial():
    baseline, replacement = _case(0), _case(1)
    value = {"choice": baseline["case"], "prediction": "reaction distance changes", "interpretation": "The unchanged braking term remains."}
    saved = _signature(1, "B", value, pinned=baseline)
    assert saved != _signature(1, "B", value, pinned=replacement)
    assert saved == _signature(1, "B", value, pinned=deepcopy(baseline))


def test_saved_signature_retains_immutable_source_conditions():
    baseline = _case(0)
    value = {"choice": baseline["case"], "prediction": "reaction distance changes", "interpretation": "An explicit comparison."}
    saved = _signature(1, "B", value, pinned=baseline)
    baseline["result"]["inputs"]["late_mutation"] = True
    assert "late_mutation" not in saved["pinned_case"]["result"]["inputs"]
    edited = dict(value, interpretation="A revised explanation.")
    assert saved != _signature(1, "B", edited, pinned=baseline)


def test_changed_study_provenance_invalidates_attachment_even_at_same_choice():
    baseline = _case(0)
    value = {"choice": baseline["case"], "prediction": "reaction distance changes", "interpretation": "An explicit comparison."}
    result = deepcopy(baseline["result"])
    saved = _signature(1, "B", value, result=result, pinned=baseline)
    result["provenance"]["version"] = "a different study revision"
    assert saved != _signature(1, "B", value, result=result, pinned=baseline)


def test_report_repin_preserves_history_but_excludes_stale_evidence():
    from mlsysbook_labs.learning_paths import get_learning_path
    from mlsysbook_labs.physical_companion import _discussion_report
    contract = get_learning_path(4, 1)
    values = [{"choice": choices(1, part)[1]["id"], "prediction": "A bounded claim", "interpretation": "A causal explanation."} for part in "ABCD"]
    first_choice = choices(1, "A")[0]["id"]
    first = {"case": first_choice, "result": study(1, "A", first_choice)}
    pinned = {"A": first}
    saved = {"A": {"signature": _signature(1, "A", values[0], pinned=first), "pinned": deepcopy(first), "current": study(1, "A", values[0]["choice"])}}
    current = _discussion_report(contract, values, saved, pinned, "Hold for further evidence.")
    assert "A" in current["current_evidence"]
    selected = choices(1, "A")[1]["id"]
    pinned["A"] = {"case": selected, "result": study(1, "A", selected)}
    stale = _discussion_report(contract, values, saved, pinned, "Hold for further evidence.")
    assert stale["stale_part_ids"] == ["A"]
    assert "A" not in stale["current_evidence"]
    assert "A" in stale["historical_evidence"]


def test_identical_baseline_does_not_test_an_intervention():
    from mlsysbook_labs.physical_companion import _has_intervention
    baseline = _case(0)
    assert not _has_intervention(1, baseline, deepcopy(baseline['result']))
    assert _has_intervention(0, baseline, deepcopy(baseline['result']))
    assert _has_intervention(1, baseline, _case(1)['result'])


def test_changed_conditions_with_same_outcome_are_valid_counterevidence():
    from mlsysbook_labs.physical_companion import _has_intervention
    baseline = _case(0)
    live = deepcopy(baseline['result'])
    live['inputs']['increased_nonbinding_budget'] = True
    assert live['rows'] == baseline['result']['rows']
    assert _has_intervention(1, baseline, live)
    cosmetics = deepcopy(baseline['result'])
    cosmetics['feedback'] = 'A new label cannot create a comparison.'
    assert not _has_intervention(1, baseline, cosmetics)


def test_changed_model_evidence_invalidates_same_identity_attachment():
    baseline = _case(0)
    value = {"choice": baseline["case"], "prediction": "A causal prediction", "interpretation": "The stated scope matters."}
    result = deepcopy(baseline['result'])
    before = _signature(1, 'B', value, result, baseline)
    result['rows'][0]['value'] = 'Changed canonical model output'
    assert before != _signature(1, 'B', value, result, baseline)


def test_display_rounding_keeps_exact_evidence_and_units():
    from mlsysbook_labs.physical_companion import _display_rows
    original = [{"raw_key": 0.24799999999999998, "distance": "0.6005350000000001 m", "small": 0.0000123456}]
    snapshot = deepcopy(original)
    display = _display_rows(original)
    assert display[0]['raw key'] == '0.248'
    assert display[0]['distance'] == '0.601 m'
    assert display[0]['small'] != '0'
    assert original == snapshot


def test_group_evidence_has_compact_tables_and_exact_trace_disclosure():
    from mlsysbook_labs.physical_companion import _trace_evidence
    from mlsysim.labs.experiments.v4_subgroups import study
    evidence = study(17, 'C', 'equal_thin')
    snapshot = deepcopy(evidence)
    html = _trace_evidence(evidence).text
    assert 'Detected / encounters' in html and 'Wilson interval' in html
    assert 'Boundary speed' in html and 'Clearance reserve' in html
    assert 'Exact trace' in html
    assert evidence == snapshot


def test_queue_summary_separates_quantities_from_role_policy():
    from mlsysbook_labs.physical_companion import _live_evidence
    result = study(17, 'B', 'overloaded_fleet')
    snapshot = deepcopy(result)
    html = _live_evidence(result).text
    assert 'Staffing consequence' in html and 'Human role and stage evidence' in html
    assert 'Full queue accounting' in html
    assert 'unbounded; stationary queue does not exist' in html
    assert result == snapshot


def test_compact_navigation_keeps_selected_investigation_visible():
    import marimo as mo
    from mlsysbook_labs.learning_paths import get_learning_path
    from mlsysbook_labs.physical_companion import physical_investigations, physical_navigation
    contract = get_learning_path(4, 1)
    navigation = physical_navigation(contract)
    assert navigation.value == 'Part A'
    controls = [{'choice': mo.md('Case control'), 'prediction': mo.md('Prediction control')} for _ in contract['parts']]
    values = [{'choice': choices(1, part['id'])[0]['id'], 'prediction': None} for part in contract['parts']]
    view = physical_investigations(contract, controls, values, lambda: {}, lambda _: None,
                                   lambda: {}, lambda _: None, 'Part C', {}, {}, [])
    assert 'Part C ·' in view.text
    assert 'Part A ·' not in view.text
    assert 'Start with a prediction' in view.text
