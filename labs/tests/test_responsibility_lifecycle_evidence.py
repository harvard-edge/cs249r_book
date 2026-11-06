"""Snapshots preserve refused comparisons and invalidate after a policy edit."""
from mlsysim.labs.experiments.v1_15_experiments import lifecycle_policy
from mlsysbook_labs.experiment_evidence import audit_evidence, capture_evidence


def test_lifecycle_capture_is_replayable_immutable_and_policy_sensitive():
    baseline = lifecycle_policy("cloud")
    candidate = lifecycle_policy("cloud", execution="replayed_candidate", explanation_share_pct=0)
    captured = capture_evidence(
        track="cloud", part="D", prediction="inference", inputs=candidate["inputs"],
        baseline=baseline, result=candidate, upstream_inputs=candidate["inputs"],
        model_key="v1_15_experiments.lifecycle_policy",
    )
    snapshot = captured.to_dict()
    assert snapshot["result"]["comparison_gate"] == "ineligible"
    assert lifecycle_policy(**snapshot["result"]["inputs"]) == snapshot["result"]
    assert lifecycle_policy(**snapshot["baseline"]["inputs"]) == snapshot["baseline"]
    fresh = audit_evidence(
        {"D": captured}, track="cloud", required_parts=("D",),
        per_part_upstream_inputs={"D": candidate["inputs"]}, contrast_required_parts=("D",),
    )
    assert fresh.complete
    candidate["comparison_gate"] = "fake_pass"
    assert captured.to_dict()["result"]["comparison_gate"] == "ineligible"
    restored_obligation = lifecycle_policy("cloud", execution="replayed_candidate")
    stale = audit_evidence(
        {"D": captured}, track="cloud", required_parts=("D",),
        per_part_upstream_inputs={"D": restored_obligation["inputs"]}, contrast_required_parts=("D",),
    )
    assert stale.stale == ("D",)


def test_locked_lifecycle_part_renders_candidate_accounting_and_gate(isolated_native_ledger_directory):
    import importlib.util
    from pathlib import Path
    import marimo as mo

    path = Path(__file__).resolve().parents[1] / "vol1" / "lab_15_responsible_engr.py"
    spec = importlib.util.spec_from_file_location("responsibility_notebook", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predictions = {f"{part}_prediction": mo.ui.radio(["inference"], value="inference" if part == "d" else None)
                   for part in "abcde"}
    outputs, definitions = module.app.run(defs=predictions)
    assert outputs
    assert definitions["d_result"]["execution"] == "replayed_candidate"
    assert definitions["d_result"]["comparison_gate"] == "comparable_on_supplied_evidence"
    assert not definitions["d_result"]["supported_energy_saving"]
