"""Evidence currentness regressions for the shared agentic notebook workflow."""
from copy import deepcopy
from mlsysbook_labs.agentic_companion import current_evidence


def case():
    study = {"inputs": {"policy": "checked", "choice": "checked", "baseline": "open"}, "events": [{"event": "effect blocked"}]}
    record = {"track": "repair", "part": "A", "chapter": 7,
              "study": deepcopy(study), "prediction": "uncertain", "interpretation": "The version changed."}
    return {("repair", "A"): record}, {"A": study}, {"A": "The version changed."}, {"A": "uncertain"}


def test_exact_capture_remains_current():
    saved, live, notes, predictions = case()
    assert current_evidence(saved, "repair", live, notes, predictions)["A"] == saved[("repair", "A")]


def test_change_of_prediction_interpretation_or_policy_invalidates_current_export():
    saved, live, notes, predictions = case()
    assert not current_evidence(saved, "repair", live, notes, {"A": "improves"})
    assert not current_evidence(saved, "repair", live, {"A": "A revised explanation."}, predictions)
    changed = deepcopy(live)
    changed["A"]["inputs"]["policy"] = "unchecked"
    assert not current_evidence(saved, "repair", changed, notes, predictions)
    assert saved[("repair", "A")]["study"]["inputs"]["policy"] == "checked"


def test_track_switch_cannot_relabel_saved_comparison():
    saved, live, notes, predictions = case()
    assert not current_evidence(saved, "support", live, notes, predictions)
    assert current_evidence(saved, "repair", live, notes, predictions)


def test_draft_reports_required_missing_parts_and_prediction_history():
    from mlsysbook_labs.agentic_companion import discussion_record
    saved, live, notes, predictions = case()
    contract = {"chapter": 7, "parts": [{"id": x} for x in "ABCD"]}
    report = discussion_record(contract, "repair", live, notes, predictions, saved, "Hold pending target evidence.")
    assert report["completed_parts"] == ["A"]
    assert report["missing_parts"] == ["B", "C", "D"]
    assert report["status"].startswith("draft")
    assert report["evidence"]["A"]["prediction"] == "uncertain"
    assert not discussion_record(contract, "support", live, notes, predictions, saved, "Inherited text.")["evidence"]


def test_volume_three_learning_maps_preserve_chapter_ownership():
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[2]
    maps = sorted((root / "labs/vol3/concepts").glob("*_concepts.yml"))
    assert len(maps) == 21
    for path in maps:
        m = yaml.safe_load(path.read_text())["lab_concept_map"]
        assert [p["part"] for p in m["runtime_parts"]] == list("ABCD")
        if not m["chapter"]:
            assert not m["core_concepts"]
            continue
        owner = yaml.safe_load((root / m["source_chapter_concept_map"]).read_text())["concept_map"]
        mapped = []
        for concept in m["core_concepts"]:
            assert concept["source_anchor"] in (root / m["source_chapter"]).read_text()
            for source in concept["source_concepts"]:
                chapter = yaml.safe_load((root / source["concept_map"]).read_text())["concept_map"]
                assert source["concept"] in chapter["claims"]
                if m["chapter"] != 20:
                    assert source["chapter"] == m["chapter"]
                    mapped.append(source["concept"])
                else:
                    assert source["chapter"] < 20
        assert sorted(mapped) == sorted(owner["primary_concepts"])


def test_all_entrypoints_follow_the_shared_template_and_bootstrap_import_dependencies():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("generate_agentic_notebooks", root / "labs/tools/generate_agentic_notebooks.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    assert generator.generate(check=True) == []
    source = generator.TEMPLATE.read_text()
    assert 'await micropip.install(["pydantic", "pint", "plotly", "pandas"]' in source


def test_browser_metadata_identity_is_explicit_and_independent_of_mount_filename():
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[2]
    for path in (root / "labs/vol3/concepts").glob("*_concepts.yml"):
        contract = yaml.safe_load(path.read_text())["lab_concept_map"]
        source = (root / contract["source_notebook"]).read_text()
        identity = contract["source_notebook"].removeprefix("labs/")
        assert f'get_lab_metadata("{identity}")' in source
        assert 'get_lab_metadata(__file__)' not in source


def test_starting_policy_inspection_is_not_completed_comparison():
    from mlsysbook_labs.agentic_companion import has_policy_contrast
    saved, live, notes, predictions = case()
    live["A"]["inputs"]["choice"] = live["A"]["inputs"]["baseline"]
    saved[("repair", "A")]["study"] = deepcopy(live["A"])
    assert not has_policy_contrast(live["A"])
    assert not current_evidence(saved, "repair", live, notes, predictions)
    assert not has_policy_contrast({})


def test_capture_is_an_immutable_snapshot_of_live_evidence():
    from mlsysbook_labs.agentic_companion import evidence_actions
    class UI:
        @staticmethod
        def button(**kwargs):
            return kwargs
    class MO:
        ui = UI()
    saved, live, notes, predictions = case()
    captured = {}
    def set_saved(update):
        captured.update(update({}))
    buttons = evidence_actions(MO(), 7, {"parts": [{"id": "A"}]}, "repair", live,
                               predictions, notes, set_saved)
    assert not buttons["A"]["disabled"]
    buttons["A"]["on_click"](None)
    live["A"]["events"][0]["event"] = "changed after capture"
    assert captured[("repair", "A")]["study"]["events"][0]["event"] == "effect blocked"
    live["A"]["inputs"]["choice"] = "open"
    assert evidence_actions(MO(), 7, {"parts": [{"id": "A"}]}, "repair", live,
                            predictions, notes, set_saved)["A"]["disabled"]


def test_part_specific_predictions_are_used_without_grading_their_choice():
    from mlsysbook_labs.agentic_companion import investigation_controls, evaluate_trials
    class UI:
        @staticmethod
        def dictionary(items): return items
        @staticmethod
        def radio(options, **kwargs): return {"options": options, **kwargs}
        @staticmethod
        def dropdown(**kwargs): return kwargs
        @staticmethod
        def text_area(**kwargs): return kwargs
    class MO:
        ui = UI()
    contract = {"parts": [{"id": "A", "question": "Can the retained record close?"}]}
    options = ["The open duty remains", "A reply alone settles it", "Uncertain"]
    predictions, _, _ = investigation_controls(MO(), 10, contract, "repair",
        lambda *a, **k: {"start": "Start", "changed": "Change"},
        lambda *a, **k: {"prediction_options": options})
    assert predictions["A"]["options"] == options
    for selected in options:
        live = evaluate_trials(10, contract, "repair", {"A": selected}, {"A": "changed"},
                               lambda *a, **k: {"evidence": "same admitted investigation"})
        assert live["A"]["evidence"] == "same admitted investigation"


def test_primary_comparison_promotes_declared_fields_without_raw_record_bloat():
    from mlsysbook_labs.agentic_companion import primary_comparison_rows
    result = {"rows": [{"work": "all segments", "internal": "hidden"},
                        {"work": "new segment", "internal": "hidden"}],
              "primary_fields": {"Fresh prefix work": "work"}}
    assert primary_comparison_rows(result) == [{"Measure": "Fresh prefix work", "Start": "all segments", "Trial": "new segment"}]
    result["primary_fields"] = {}
    assert primary_comparison_rows(result) == []


def test_readable_discussion_uses_captured_comparison_and_preserves_limits():
    from mlsysbook_labs.agentic_companion import discussion_markdown
    contract = {"title": "Policy choice", "parts": [{"id": "A", "label": "Changed policy", "question": "What changes?"}],
                "next_lab": {"title": "Next investigation", "question": "What remains unknown?"}}
    study = {"rows": [{"policy": "open | unchecked", "outcome": "effect occurred", "cost": 12},
                       {"policy": "checked", "outcome": "effect blocked", "cost": 18}],
             "primary_fields": {"Declared cost": "cost"}, "feedback": "The check blocked the effect.",
             "limits": ["Supplied trace; not a live experiment."]}
    report = {"track": "repair", "decision": "Choose the check, unless delay becomes binding.", "missing_parts": [],
              "evidence": {"A": {"prediction": "I expect a delay.", "interpretation": "More work buys a blocked effect.", "study": study}}}
    text = discussion_markdown(contract, report)
    assert report["decision"] in text
    assert "| Declared cost | 12 | 18 |" in text
    assert "open \\| unchecked" in text
    assert report["evidence"]["A"]["prediction"] in text
    assert report["evidence"]["A"]["interpretation"] in text
    assert study["limits"][0] in text
    assert "not live-agent validation" in text


def test_readable_draft_does_not_invent_missing_comparisons():
    from mlsysbook_labs.agentic_companion import discussion_markdown
    contract = {"title": "Draft", "parts": [{"id": "B", "label": "Missing part", "question": "Which policy?"}],
                "next_lab": {"title": "Next", "question": "Why?"}}
    text = discussion_markdown(contract, {"track": "repair", "decision": "", "missing_parts": ["B"], "evidence": {}})
    assert "No current saved comparison for this part." in text
    assert "No recommendation written yet." in text
    assert "Parts still needing evidence: B" in text
    assert "Draft; some comparisons are still missing" in text
