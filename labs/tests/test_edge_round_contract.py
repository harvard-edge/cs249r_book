"""Saved protocol evidence must replay and expire on every causal choice."""
import importlib
import json

import pytest

from mlsysbook_labs.experiment_evidence import capture_evidence, audit_evidence
from mlsysim.labs.experiments.v2_11_experiments import evaluate_fleet_round


def test_round_snapshot_replays_without_aliasing_and_retains_failed_gates():
    common = dict(track_id="mobile", max_clients=4, survivor_floor=4,
                  dropout_case="missing_shares", validation_evidence="shadow")
    baseline = evaluate_fleet_round(**common, selection_policy="freshest")
    result = evaluate_fleet_round(**common, selection_policy="coverage")
    upstream = result["inputs"].copy()
    capture = capture_evidence(track="mobile", part="E", prediction="coverage",
                               inputs=upstream, baseline=baseline, result=result,
                               upstream_inputs=upstream, model_key=result["model_key"])
    snapshot = capture.to_dict()
    module, fn = snapshot["model_key"].split(".")
    replay = getattr(importlib.import_module(f"mlsysim.labs.experiments.{module}"), fn)
    for row in (snapshot["baseline"], snapshot["result"]):
        assert json.loads(json.dumps(replay(**row["inputs"]))) == row
        assert not row["aggregate_available"]
        assert not row["promotion_allowed"]
    result["release_blockers"].clear()
    assert snapshot["result"]["release_blockers"]
    assert audit_evidence({"E": capture}, track="mobile", required_parts=("E",),
                          per_part_upstream_inputs={"E": upstream}, contrast_required_parts=("E",)).complete


@pytest.mark.parametrize("key,value", [
    ("survivor_floor", 4), ("dropout_case", "recoverable"),
    ("release_state", "base_only"), ("validation_evidence", "validated"),
    ("min_uplink_mbps", 2), ("selection_policy", "freshest"),
])
def test_protocol_release_control_changes_stale_saved_evidence(key, value):
    result = evaluate_fleet_round("mobile")
    upstream = result["inputs"].copy()
    capture = capture_evidence(track="mobile", part="E", prediction="coverage", inputs=upstream,
                               baseline=result, result=result, upstream_inputs=upstream)
    changed = {**upstream, key: value}
    audit = audit_evidence({"E": capture}, track="mobile", required_parts=("E",),
                           per_part_upstream_inputs={"E": changed})
    assert audit.stale == ("E",)
    assert capture.to_dict()["result"] == json.loads(json.dumps(result))
