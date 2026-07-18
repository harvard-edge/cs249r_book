from pathlib import Path

import pytest

from mlperf.assignment import evaluate_assignment_contract, load_assignment_contract


def test_assignment_contract_fixes_cardinality_quality_and_config(tmp_path: Path):
    path = tmp_path / "assignment.yaml"
    path.write_text(
        """\
schema: mlperf-edu-assignment/0.1
id: image-classification-lab
title: Image classification systems lab
allow_extra_results: false
requirements:
  - workload: image-classification
    profile: max
    mode: inference
    count: 1
    quality:
      required: true
      target_met: true
    config:
      batch_size: 8
"""
    )
    contract = load_assignment_contract(path)
    rows = [
        {
            "manifest": "submission.provd.json",
            "workload": "image-classification",
            "canonical_workload": "",
            "variant": "",
            "profile": "max",
            "mode": "inference",
            "phase": None,
            "verified": True,
            "passed": True,
            "quality_required": True,
            "target_met": True,
            "config": {"batch_size": 8, "device": "cpu"},
        }
    ]

    result = evaluate_assignment_contract(contract, rows)

    assert result["passed"] is True
    assert result["assignment_source_sha256"].startswith("sha256:")
    assert result["requirements"][0]["matched_count"] == 1
    assert result["extra_results"] == []


def test_assignment_contract_reports_selector_and_config_mismatches(tmp_path: Path):
    path = tmp_path / "assignment.yaml"
    path.write_text(
        """\
schema: mlperf-edu-assignment/0.1
id: retrieval-lab
requirements:
  - workload: information-retrieval
    profile: max
    count: 1
    quality:
      target_met: true
    config:
      evaluator:
        k: 10
"""
    )
    contract = load_assignment_contract(path)
    rows = [
        {
            "manifest": "submission.provd.json",
            "workload": "information-retrieval",
            "profile": "max",
            "verified": True,
            "passed": True,
            "quality_required": True,
            "target_met": True,
            "config": {"evaluator": {"k": 5}},
        },
        {
            "manifest": "extra.provd.json",
            "workload": "text-classification",
            "profile": "max",
            "verified": True,
            "passed": True,
            "quality_required": True,
            "target_met": True,
            "config": {},
        },
    ]

    result = evaluate_assignment_contract(contract, rows)

    assert result["passed"] is False
    assert any("config.evaluator.k" in error for error in result["errors"])
    assert any("unexpected result" in error for error in result["errors"])


def test_assignment_contract_rejects_duplicate_selectors(tmp_path: Path):
    path = tmp_path / "assignment.yaml"
    path.write_text(
        """\
schema: mlperf-edu-assignment/0.1
id: duplicate
requirements:
  - workload: image-classification
  - workload: image-classification
"""
    )

    with pytest.raises(ValueError, match="duplicates"):
        load_assignment_contract(path)


@pytest.mark.parametrize(
    "body",
    [
        """\
schema: mlperf-edu-assignment/0.1
id: typo
allow_extra_result: false
requirements:
  - workload: image-classification
""",
        """\
schema: mlperf-edu-assignment/0.1
id: typo
requirements:
  - workload: image-classification
    cout: 2
""",
    ],
)
def test_assignment_contract_rejects_unknown_fields(tmp_path: Path, body: str):
    path = tmp_path / "assignment.yaml"
    path.write_text(body)

    with pytest.raises(ValueError, match="unknown fields"):
        load_assignment_contract(path)
