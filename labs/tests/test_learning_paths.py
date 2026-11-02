"""Student-route contracts and opt-in notebook presentation boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from mlsysbook_labs import learning_paths
from mlsysbook_labs.learning_paths import get_learning_path, normalize_volume, validate_learning_path
from mlsysbook_labs.notebook import notebook_cards, notebook_learning_map, notebook_learning_part, notebook_part_reflection, notebook_shell


@pytest.fixture
def contract():
    return {
        "volume": 3, "chapter": 1, "title": "A development decision",
        "starting_point": "A supplied case; no prior record has been loaded.",
        "prerequisites": ["Read the linked chapter."],
        "learning_objectives": ["Compare competing requirements.", "Defend a reversal condition."],
        "parts": [dict(id=part, label=f"Investigation {part}", concept_id=f"v3_01_{part}",
                       question=f"Which condition matters in {part}?", goal="Explain the changed constraint.",
                       artifact="One saved comparison.", scope="A teaching fixture, not deployment evidence.")
                  for part in "ABCD"],
        "deliverable": "One discussion supported by the saved comparisons.",
        "next_part": "Inspect the requirement before changing it.",
        "next_lab": {"title": "The next lab", "question": "Does this choice survive a new condition?"},
        "narrative": {"project": "Continue the shared project.", "scene": "A requirement has changed.",
                      "carry_forward": "Retain the assumptions; verify compatible evidence before reuse.",
                      "scope": "This route does not establish completion."},
    }


def test_visible_route_keeps_every_part_and_no_completion_claim(contract):
    rendered = notebook_learning_map(contract, current_part="B").text
    for heading in ("What you will learn", "Your route through this lab", "What you will hand in", "What comes next"):
        assert heading in rendered
    for part in contract["parts"]:
        assert f'Part {part["id"]} ·' in rendered
        assert part["question"] in rendered
    assert rendered.count('<li aria-current="step"') == 1
    assert "A supplied case; no prior record has been loaded." in rendered
    assert "<details" not in rendered
    assert "completed" not in rendered


def test_all_authored_text_is_escaped(contract):
    payload = '<img src=x onerror="alert(1)">'
    contract["learning_objectives"] = [payload]
    contract["parts"][0]["question"] = payload
    contract["deliverable"] = payload
    contract["next_lab"]["question"] = payload
    contract["narrative"]["scene"] = payload
    rendered = notebook_learning_map(contract).text
    assert payload not in rendered
    assert rendered.count('&lt;img src=x onerror=&quot;alert(1)&quot;&gt;') == 4


@pytest.mark.parametrize("field", ["starting_point", "learning_objectives", "parts", "deliverable", "next_part", "next_lab"])
def test_missing_visible_contract_rejected(contract, field):
    del contract[field]
    with pytest.raises(ValueError):
        validate_learning_path(contract)


def test_route_identity_and_location_rejected(contract):
    contract["parts"][1]["id"] = "A"
    with pytest.raises(ValueError, match="unique"):
        validate_learning_path(contract)
    contract["parts"][1]["id"] = "B"
    with pytest.raises(ValueError, match="Unknown current part"):
        notebook_learning_map(contract, current_part="Z")


def test_packaged_loader_identity_mutation_and_missing_chapter(tmp_path, monkeypatch, contract):
    directory = tmp_path / "data" / "learning_paths"
    directory.mkdir(parents=True)
    path = directory / "vol3.json"
    path.write_text(json.dumps({"schema_version": 1, "volume": 3, "chapters": [contract]}))
    monkeypatch.setattr(learning_paths.resources, "files", lambda package: tmp_path)
    loaded = get_learning_path("vol3", 1)
    assert loaded == contract
    loaded["learning_objectives"].clear()
    assert get_learning_path("III", 1)["learning_objectives"]
    with pytest.raises(KeyError):
        get_learning_path(3, 2)
    duplicate = deepcopy(contract)
    path.write_text(json.dumps({"schema_version": 1, "volume": 3, "chapters": [contract, duplicate]}))
    with pytest.raises(ValueError, match="conflicting"):
        get_learning_path(3, 1)


def test_theme_map_matches_brand_and_resource_states_remain_semantic(contract):
    brand = (Path(__file__).resolve().parents[2] / "shared/styles/_brand.scss").read_text()
    names = {1: "crimson", 2: "eth-blue", 3: "amethyst", 4: "malachite"}
    for volume, name in names.items():
        match = re.search(rf'\$brand-{name}:\s*(#[\da-fA-F]+)', brand)
        assert learning_paths.VOLUME_ACCENTS[volume] == match.group(1)
        contract["volume"] = volume
        assert f'--nb-accent:{match.group(1)}' in notebook_learning_map(contract).text
        notebook_shell([], volume=volume)
    rendered = notebook_cards([("Fit", "yes", "pass"), ("Deadline", "no", "fail")]).text
    assert 'border-left:3px solid #247a4d' in rendered
    assert 'border-left:3px solid #b42318' in rendered
    assert 'data-state="pass"' in rendered and 'data-state="fail"' in rendered


@pytest.mark.parametrize("value,number", [(1, 1), ("Volume II", 2), ("vol3", 3), ("IV", 4)])
def test_volume_aliases(value, number):
    assert normalize_volume(value) == number


def test_map_is_route_not_a_second_project_or_study_preamble(contract):
    for part in contract["parts"]:
        part["goal"] = contract["learning_objectives"][0]
        part["coverage"] = "Author implementation status."
        part["limitation"] = "No deployment validation."
    rendered = notebook_learning_map(contract).text
    assert rendered.count(contract["learning_objectives"][0]) == 1
    assert "Author implementation status." not in rendered
    assert "No deployment validation." not in rendered
    assert contract["narrative"]["scene"] not in rendered
    assert contract["narrative"]["scene"] in notebook_learning_map(contract, show_narrative=True).text
    assert "<details" not in rendered


def test_part_prompts_and_scope_are_visible_at_instrument(contract):
    part = contract["parts"][0]
    part["explanation_prompt"] = "Which term changed and why?"
    part["reversal_probe"] = '<script>unsafe</script>'
    part["limitation"] = "Does not model shared state."
    reflection = notebook_part_reflection(contract, "A").text
    assert "Which term changed and why?" in reflection
    assert "<script>unsafe</script>" not in reflection
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in reflection
    component = notebook_learning_part(contract, "A", "instrument")
    assert component is not None
    with pytest.raises(ValueError, match="Unknown learning part"):
        notebook_part_reflection(contract, "Z")


def test_optional_coverage_requires_real_text(contract):
    contract["parts"][0]["limitation"] = {"unrendered": "not a textual limitation"}
    with pytest.raises(ValueError, match="part.limitation"):
        validate_learning_path(contract)


def test_prediction_labels_preserve_stable_ids_and_support_saved_values():
    from types import SimpleNamespace
    from mlsysbook_labs.notebook import notebook_prediction_label
    radio = SimpleNamespace(options={"Keep the baseline": "hold", "Change placement": "remote"}, value="hold")
    form = SimpleNamespace(element=radio, value="remote")
    assert notebook_prediction_label(radio) == "Keep the baseline"
    assert notebook_prediction_label(form) == "Change placement"
    assert notebook_prediction_label(form, value="hold") == "Keep the baseline"
    assert notebook_prediction_label(form, value=None) == "Not submitted"
    assert notebook_prediction_label(form, value="old_identifier") == "old_identifier"
    assert radio.value == "hold" and form.value == "remote"


def test_wide_evidence_table_uses_readable_scrollable_measure():
    from mlsysbook_labs.notebook import notebook_table
    rendered = notebook_table([{f"Column {index}": "Population reference" for index in range(8)}]).text
    assert 'min-width:896px' in rendered
    assert 'overflow-x:auto' in rendered


def test_record_table_retains_readable_first_column_inside_html_boundary():
    from mlsysbook_labs.notebook import notebook_table
    rendered = notebook_table([{"Field": "Condition", "Supplied record": "A declared reference"}]).text
    assert ".nb-table th:first-child,.nb-table td:first-child{min-width:95px}" in rendered


def test_heterogeneous_trace_preserves_later_evidence_fields():
    from mlsysbook_labs.notebook import notebook_table
    rendered = notebook_table([
        {"Policy": "steady", "Outcome": "allowed"},
        {"Event": "queue wait", "Wait (ms)": 12, "Policy": "steady"},
        {"Event": "current clamp", "Limit": "<declared>"},
    ]).text
    headers = re.findall(r'<th scope="col">(.*?)</th>', rendered)
    assert headers == ["Policy", "Outcome", "Event", "Wait (ms)", "Limit"]
    assert '<td>queue wait</td>' in rendered and '<td>12</td>' in rendered
    assert '&lt;declared&gt;' in rendered
    # Missing fields retain empty cells so each event stays aligned to its column.
    bodies = re.findall(r'<tr>(.*?)</tr>', rendered)[1:]
    assert all(row.count('<td>') == len(headers) for row in bodies)


def test_wide_evidence_table_is_a_named_keyboard_scroll_region():
    from mlsysbook_labs.notebook import notebook_table
    rendered = notebook_table([{str(i): i for i in range(5)}], caption='Compare "wait" costs').text
    assert 'tabindex="0" role="region" aria-label="Compare &quot;wait&quot; costs"' in rendered


def test_compact_navigation_label_preserves_current_route_marker(contract):
    html = notebook_learning_map(contract, current_part='Part B').text
    assert html == notebook_learning_map(contract, current_part='B').text
