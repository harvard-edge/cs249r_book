"""Exercise the label-design control, capture callback, replay, and dependency gate."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import marimo as mo
import pytest

from mlsysbook_labs.experiment_evidence import audit_evidence
from mlsysim.labs.experiments.v1_04_experiments import replay_experiment


@pytest.fixture(scope="module")
def label_lab():
    path = Path(__file__).resolve().parents[1] / "vol1" / "lab_04_data_engr.py"
    spec = spec_from_file_location("data_label_lab", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    _, defaults = module.app.run()
    return module.app, defaults


def _run_label_design(label_lab, rubric, task):
    app, defaults = label_lab
    # Marimo replaces the whole control-defining cell, so preserve companions.
    controls = ("a_policy", "a_rubric", "a_review", "a_task", "b_scope", "b_strategy", "c_compression",
                "c_lanes", "c_storage", "d_policy", "d_trace", "e_level", "final_choice", "final_rejected",
                "final_risk", "final_trigger", "rationale")
    overrides = {key: defaults[key] for key in controls}
    overrides.update(
        a_policy=mo.ui.dropdown({"Newest": "newest"}, value="Newest"),
        a_rubric=mo.ui.dropdown({rubric: rubric}, value=rubric),
        a_task=mo.ui.dropdown({task: task}, value=task),
        a_prediction=SimpleNamespace(value="Finer labels resolve necessary actions"),
    )
    return app.run(defs=overrides)[1]


@pytest.mark.parametrize("task", ("route", "presence"))
def test_label_control_recomputes_exact_evidence_and_capture_replays(label_lab, task):
    coarse = _run_label_design(label_lab, "coarse", task)
    specific = _run_label_design(label_lab, "specific", task)
    before, after = coarse["a_chosen"], specific["a_chosen"]
    assert before.inputs["records"] == after.inputs["records"]
    assert before.inputs["rubric"]["actions_by_cohort"] == after.inputs["rubric"]["actions_by_cohort"]
    assert len(after.result["retained_ids"]) < len(before.result["retained_ids"])
    if task == "route":
        assert before.result["ambiguous_label_count"] > after.result["ambiguous_label_count"]
    else:
        assert before.result["ambiguous_label_count"] == after.result["ambiguous_label_count"] == 0
    specific["a_capture"]._on_click(None)
    capture = specific["get_evidence"]()["A"].to_dict()
    assert capture["inputs"] == {"policy": "newest", "rubric": "specific", "task": task, "review": "single"}
    assert replay_experiment({"method": "retention", "inputs": capture["chosen_result"]["inputs"]}).result == capture["chosen_result"]["outputs"]


def test_saved_label_comparison_stales_when_rubric_or_task_changes(label_lab):
    definitions = _run_label_design(label_lab, "specific", "route")
    definitions["a_capture"]._on_click(None)
    capture = definitions["get_evidence"]()["A"]
    original = capture.to_dict()
    settings = original["inputs"]
    for changed in (settings, {**settings, "rubric": "coarse"}, {**settings, "task": "presence"}):
        audit = audit_evidence(
            {"A": capture}, track=original["track"], required_parts=("A",),
            contrast_required_parts=("A",), per_part_upstream_inputs={"A": changed},
        )
        assert audit.complete == (changed == settings)
        assert audit.stale == (() if changed == settings else ("A",))
    assert capture.to_dict() == original


def test_review_control_changes_exact_audit_and_capture_stales(label_lab):
    app, defaults = label_lab
    controls = ("a_policy", "a_rubric", "a_review", "a_task", "b_scope", "b_strategy", "c_compression",
                "c_lanes", "c_storage", "d_policy", "d_trace", "e_level", "final_choice", "final_rejected",
                "final_risk", "final_trigger", "rationale")
    overrides = {key: defaults[key] for key in controls}
    overrides.update(a_review=mo.ui.dropdown({"Gold": "gold"}, value="Gold"),
                     a_prediction=SimpleNamespace(value="Finer labels resolve necessary actions"))
    definitions = app.run(defs=overrides)[1]
    assert definitions["a_chosen"].result["annotation_audit"]["reference_mismatches"] == 0
    assert definitions["a_base"].result["annotation_audit"]["reference_mismatches"] > 0
    definitions["a_capture"]._on_click(None)
    capture = definitions["get_evidence"]()["A"]
    payload = capture.to_dict()
    assert payload["inputs"]["review"] == "gold"
    assert replay_experiment({"method": "retention", "inputs": payload["chosen_result"]["inputs"]}).result == payload["chosen_result"]["outputs"]
    audit = audit_evidence({"A": capture}, track=payload["track"], required_parts=("A",),
                           contrast_required_parts=("A",),
                           per_part_upstream_inputs={"A": {**payload["inputs"], "review": "single"}})
    assert audit.stale == ("A",)


def test_freshness_trace_control_updates_live_results_and_stales_capture(label_lab):
    app, defaults = label_lab
    controls = ("a_policy", "a_rubric", "a_review", "a_task", "b_scope", "b_strategy", "c_compression",
                "c_lanes", "c_storage", "d_policy", "d_trace", "e_level", "final_choice", "final_rejected",
                "final_risk", "final_trigger", "rationale")
    runs = {}
    for trace in ("changing", "stable"):
        overrides = {key: defaults[key] for key in controls}
        overrides.update({key: defaults[key] for key in ("b_prediction", "c_prediction", "e_prediction")})
        overrides.update(d_trace=mo.ui.dropdown({trace: trace}, value=trace),
                         d_prediction=SimpleNamespace(value="Streaming earns its traffic"))
        runs[trace] = app.run(defs=overrides)[1]
    assert runs["changing"]["d_base"].result["stale_decisions"] > 0
    assert runs["stable"]["d_base"].result["stale_decisions"] == 0
    runs["changing"]["d_capture"]._on_click(None)
    capture = runs["changing"]["get_evidence"]()["D"]
    payload = capture.to_dict()
    assert payload["inputs"]["trace"] == "changing"
    assert replay_experiment({"method": "freshness", "inputs": payload["chosen_result"]["inputs"]}).result == payload["chosen_result"]["outputs"]
    audit = audit_evidence({"D": capture}, track=payload["track"], required_parts=("D",),
                           contrast_required_parts=("D",),
                           per_part_upstream_inputs={"D": {**payload["inputs"], "trace": "stable"}})
    assert audit.stale == ("D",)
