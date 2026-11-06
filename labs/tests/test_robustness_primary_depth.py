"""Real Lab14 controls preserve replay and freshness for the new chapter mechanisms."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
from mlsysbook_labs.experiment_evidence import audit_evidence
from mlsysim.labs.experiments.v2_14_experiments import robustness_review_payload, competence_review_payload


@pytest.fixture(scope="module")
def robustness_lab():
    path=Path(__file__).resolve().parents[1]/"vol2"/"lab_14_robust_ai.py"
    spec=spec_from_file_location("robustness_primary_depth",path)
    module=module_from_spec(spec);spec.loader.exec_module(module)
    return module.app,module.app.run()[1]


def _run(lab, *, evidence=None, **changes):
    app,defaults=lab
    names=("a_scenario","a_delay","b_threshold","b_capacity","c_threat","c_test","c_decision",
           "c_layers","c_lineage","d_count","d_alpha","d_exchange","d_routing","d_source")
    overrides={name:defaults[name] for name in names}
    overrides.update({name:SimpleNamespace(value="locked") for name in
                      ("a_prediction","b_prediction","c_prediction","d_prediction","c_pipeline_prediction","d_generation_prediction")})
    overrides.update({name:SimpleNamespace(value=value) for name,value in changes.items()})
    if any(name.startswith("final_") or name == "rationale" for name in changes):
        overrides["choices"] = defaults["choices"]
    if evidence is not None:
        overrides["get_evidence"] = lambda: evidence
        overrides["set_evidence"] = defaults["set_evidence"]
    return app.run(defs=overrides)[1]


def test_pipeline_capture_contains_replay_and_layer_lineage_staleness(robustness_lab):
    definitions=_run(robustness_lab,c_decision="baseline",c_layers=4,c_lineage=True)
    definitions["c_capture"]._on_click(None)
    capture=definitions["get_evidence"]()["C"];saved=capture.to_dict()
    assert robustness_review_payload(**saved["result"]["inputs"]) == saved["result"]
    assert saved["chosen_result"]["inputs"]["defense_id"] == "baseline"
    assert saved["chosen_result"]["result"]["poisoning"]["descendant_scope_known"]
    assert saved["prediction"]["pipeline"] == "locked"
    for key,value in (("layer_count",1),("lineage",False)):
        altered={**definitions["c_upstream"],key:value}
        assert audit_evidence({"C":capture},track=definitions["track_id"],required_parts=("C",),
                              per_part_upstream_inputs={"C":altered}).stale == ("C",)
    assert capture.to_dict() == saved


def test_generation_capture_replays_reference_and_routing_without_invented_truth(robustness_lab):
    definitions=_run(robustness_lab,d_routing="evidence",d_source="archived")
    definitions["d_capture"]._on_click(None)
    capture=definitions["get_evidence"]()["D"];saved=capture.to_dict()
    assert competence_review_payload(**saved["result"]["inputs"]) == saved["result"]
    assert saved["result"]["result"]["generation"]["current_reference_errors_accepted"] == 1
    assert saved["prediction"]["generation"] == "locked"
    for key,value in (("routing","consistency"),("evidence_source","current")):
        altered={**definitions["d_upstream"],key:value}
        assert audit_evidence({"D":capture},track=definitions["track_id"],required_parts=("D",),
                              per_part_upstream_inputs={"D":altered}).stale == ("D",)


def test_complete_discussion_retains_both_nested_predictions_and_choices(robustness_lab):
    definitions=_run(robustness_lab,c_decision="baseline",d_source="archived",d_routing="evidence")
    for part in "abcd":
        definitions[f"{part}_capture"]._on_click(None)
    captures=definitions["get_evidence"]()
    complete=_run(robustness_lab,evidence=captures,c_decision="baseline",d_source="archived",d_routing="evidence",
                  final_choice="baseline",final_rejected="input_filter",final_trigger="evidence",final_risk="source",
                  rationale="Retain baseline within threat scope; trace historical descendants without claiming repair. Archived support leaves obsolete advice accepted, so verify the current handbook before deployment.")
    assert complete["fleet_discussion_ready"]
    report=complete["report"]
    assert report.snapshot["predictions"]["C"]["pipeline"] == "locked"
    assert report.snapshot["predictions"]["D"]["generation"] == "locked"
    assert report.snapshot["final_decision"]["evidence_source"] == "archived"
    assert report.snapshot["final_decision"]["generation_routing"] == "evidence"
