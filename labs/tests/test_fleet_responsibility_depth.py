"""Real notebook controls preserve bounded mechanisms and invalidate captures."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import marimo as mo
import pytest

from mlsysbook_labs.experiment_evidence import audit_evidence
from mlsysim.labs.experiments.v2_16_experiments import audit_evidence as simulate_audit


@pytest.fixture(scope="module")
def responsible_lab():
    path = Path(__file__).resolve().parents[1] / "vol2" / "lab_16_responsible_ai.py"
    spec = spec_from_file_location("fleet_responsibility_depth", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    _, defaults = module.app.run()
    return module.app, defaults


def _run(responsible_lab, **changed):
    app, defaults = responsible_lab
    names = ("a_threshold", "a_criterion", "a_allowed_gap", "b_explanation_share", "b_review_share",
             "b_exploration", "b_review_order", "c_sampling", "c_label_delay", "c_dossier",
             "c_intersections", "c_confidence", "d_profile", "d_case_review", "d_removal")
    overrides = {name: defaults[name] for name in names}
    overrides.update({f"{part}_prediction": SimpleNamespace(value="test prediction") for part in "abcde"})
    overrides.update({name: SimpleNamespace(value=value) for name, value in changed.items()})
    return app.run(defs=overrides)[1]


def test_real_uncertainty_controls_capture_replay_and_stale(responsible_lab):
    definitions = _run(responsible_lab, c_dossier="expanded", c_intersections=True, c_confidence=0.99)
    definitions["c_capture"]._on_click(None)
    captured = definitions["get_evidence"]()["C"]
    saved = captured.to_dict()
    inputs = definitions["c_result_inputs"]
    replay = simulate_audit(**inputs)
    assert replay.uncertainty == saved["result"]["uncertainty"]
    assert replay.uncertainty["gaps"][1]["lower"] > 0
    upstream = definitions["c_upstream"]
    assert audit_evidence({"C": captured}, track=definitions["track_id"], required_parts=("C",),
                         per_part_upstream_inputs={"C": upstream}, contrast_required_parts=("C",)).complete
    for key, value in (("confidence", 0.8), ("audit_dossier", "pilot"), ("audit_intersections", False)):
        edited = {**upstream, "result": {**upstream["result"], key: value}}
        assert audit_evidence({"C": captured}, track=definitions["track_id"], required_parts=("C",),
                             per_part_upstream_inputs={"C": edited}).stale == ("C",)
    assert captured.to_dict() == saved


def test_feedback_and_unlearning_controls_feed_serialized_evidence(responsible_lab):
    definitions = _run(responsible_lab, b_exploration=5, b_review_order="independent_signal", d_removal="shard_retrain")
    for part in "BD":
        definitions[f"{part.lower()}_capture"]._on_click(None)
    b = definitions["get_evidence"]()["B"].to_dict()
    d = definitions["get_evidence"]()["D"].to_dict()
    assert b["result"]["feedback"]["exposure_trace"][-1]["new_labels_B"] > 0
    assert b["result"]["feedback"]["detected_issues"] > b["baseline"]["feedback"]["detected_issues"]
    assert d["result"]["unlearning"]["matches_retrain_reference"]
    assert not d["baseline"]["unlearning"]["matches_retrain_reference"]
    for part, key, value in (("B", "review_order", "fifo"), ("D", "removal_method", "logs_only")):
        upstream = definitions[f"{part.lower()}_upstream"]
        edited = {**upstream, "result": {**upstream["result"], key: value}}
        assert audit_evidence({part: definitions["get_evidence"]()[part]}, track=definitions["track_id"], required_parts=(part,),
                             per_part_upstream_inputs={part: edited}).stale == (part,)
