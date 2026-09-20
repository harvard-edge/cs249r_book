from dataclasses import FrozenInstanceError

import pytest

from mlsysbook_labs.experiment_evidence import (
    audit_evidence, capture_evidence, input_fingerprint,
)


def capture(part="A", **overrides):
    args = dict(track="tiny", part=part, prediction="smaller will fit",
                inputs={"model": "small"}, baseline={"inputs": {"model": "large"}},
                result={"inputs": {"model": "small"}}, decision="keep small")
    args.update(overrides)
    return capture_evidence(**args)


def test_capture_is_immutable_and_detached_from_inputs_and_exports():
    original = {"inputs": {"model": ["small"]}, "quality": 90}
    record = capture(result=original)
    original["inputs"]["model"].append("changed")
    exported = record.to_dict()
    exported["result"]["inputs"]["model"].append("also changed")
    assert record.to_dict()["result"]["inputs"]["model"] == ["small"]
    with pytest.raises(FrozenInstanceError):
        record.track = "cloud"


def test_fingerprint_stable_across_mapping_order_and_changes_with_track():
    assert input_fingerprint(track="tiny", inputs={"a": 1, "b": 2}) == input_fingerprint(track="tiny", inputs={"b": 2, "a": 1})
    assert input_fingerprint(track="tiny", inputs={}) != input_fingerprint(track="cloud", inputs={})


def test_per_part_upstream_invalidation_and_track_invalidation():
    records = {"A": capture(), "B": capture("B", upstream_inputs={"chosen_model": "small"})}
    args = dict(track="tiny", required_parts=("A", "B"), per_part_upstream_inputs={"B": {"chosen_model": "small"}})
    assert audit_evidence(records, **args).complete
    args["per_part_upstream_inputs"]["B"]["chosen_model"] = "large"
    assert audit_evidence(records, **args).stale == ("B",)
    args["track"] = "cloud"
    assert audit_evidence(records, **args).stale == ("A", "B")


def test_contrast_ignores_result_values_and_cosmetic_labels():
    record = capture(baseline={"inputs": {"model": "small"}, "title": "before", "quality": 80},
                     result={"inputs": {"model": "small"}, "title": "after", "quality": 90})
    audit = audit_evidence({"A": record}, track="tiny", required_parts=("A",), contrast_required_parts=("A",))
    assert audit.identical_pairs == (("A", "A"),)
    assert not audit.complete
    assert record.to_dict()["prediction"] == "smaller will fit"
    assert record.to_dict()["decision"] == "keep small"


def test_contrast_requires_actual_settings_and_reports_missing():
    audit = audit_evidence({"A": capture(result={"quality": 90})}, track="tiny",
                           required_parts=("A", "B"), contrast_required_parts=("A",))
    assert audit.missing == ("B",)
    assert audit.missing_contrasts == ("A",)
    assert not audit.complete


def test_recapture_replaces_only_selected_record():
    first = capture(result={"inputs": {"model": "large"}})
    records = {"A": first, "B": capture("B")}
    previous_b = records["B"]
    assert not audit_evidence(records, track="tiny", required_parts=("A", "B"), contrast_required_parts=("A",)).complete
    records["A"] = capture()
    assert audit_evidence(records, track="tiny", required_parts=("A", "B"), contrast_required_parts=("A",)).complete
    assert records["B"] is previous_b
    assert first.to_dict()["result"]["inputs"]["model"] == "large"


def test_between_capture_contrast_and_audit_export_are_detached():
    records = {"A": capture(), "B": capture("B")}
    audit = audit_evidence(records, track="tiny", required_parts=("A", "B"), contrast_pairs=(("A", "B"),))
    assert audit.identical_pairs == (("A", "B"),)
    exported = audit.to_dict()
    exported["identical_pairs"].clear()
    assert audit.identical_pairs == (("A", "B"),)


def test_non_json_or_nonfinite_values_rejected():
    with pytest.raises(ValueError):
        capture(result={"latency": float("nan")})
    with pytest.raises(TypeError):
        capture(result={"latency": object()})


def test_numeric_representation_does_not_create_contrast():
    record = capture(baseline={"inputs": {"batch_size": 1}}, result={"inputs": {"batch_size": 1.0}})
    audit = audit_evidence({"A": record}, track="tiny", required_parts=("A",), contrast_required_parts=("A",))
    assert audit.identical_pairs == (("A", "A"),)


def test_no_action_preserves_tested_alternative_and_chosen_baseline():
    baseline = {"inputs": {"action": "none"}, "violations": []}
    rejected = {"inputs": {"action": "larger"}, "violations": ["memory"]}
    record = capture(baseline=baseline, result=rejected, chosen_result=baseline,
                     decision="none", result_role="rejected alternative")
    assert audit_evidence({"A": record}, track="tiny", required_parts=("A",),
                          contrast_required_parts=("A",)).complete
    snapshot = record.to_dict()
    assert snapshot["result"]["violations"] == ["memory"]
    assert snapshot["chosen_result"]["violations"] == []
    assert snapshot["result_role"] == "rejected alternative"
