from __future__ import annotations

import ast
from pathlib import Path

from mlsysbook_labs import (
    LAB_CATALOG,
    build_lab_report,
    canonical_track_ids,
    get_lab_track_variant,
    get_track_profile,
    variant_source_trace,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LABS_ROOT = REPO_ROOT / "labs"

REQUIRED_REPORT_HEADERS = (
    "## Lab",
    "## Track And Scenario",
    "## Learning Objectives",
    "## Predictions",
    "## Evidence Summary",
    "## Final Decision",
    "## Big Takeaways",
    "## Reflections",
    "## Residual Risk",
    "## Source Trace",
)

REQUIRED_SNAPSHOT_FIELDS = (
    "metadata",
    "track",
    "scenario",
    "learning_objectives",
    "predictions",
    "evidence_summary",
    "final_decision",
    "big_takeaways",
    "reflections",
    "residual_risk",
    "source_trace",
    "result_snapshot",
    "incomplete_fields",
)

REQUIRED_BUILD_REPORT_KEYWORDS = {
    "track",
    "scenario",
    "learning_objectives",
    "predictions",
    "evidence_summary",
    "final_decision",
    "big_takeaways",
    "reflections",
    "residual_risk",
    "source_trace",
    "result_snapshot",
}


def _build_complete_report(path: str, track_id: str):
    metadata = LAB_CATALOG[path]
    variant = get_lab_track_variant(metadata.lab_id, track_id)
    profile = get_track_profile(track_id)
    source_trace = variant_source_trace(variant, profile)
    return build_lab_report(
        metadata,
        student_id="contract-test",
        track=profile.label,
        scenario=variant.scenario_id,
        learning_objectives=(
            f"Evaluate {metadata.title} for the selected track.",
            f"Compare {variant.primary_metric} against {variant.guardrail_metric}.",
        ),
        predictions={"binding_constraint": variant.primary_metric},
        evidence_summary={
            "hardware_ref": variant.hardware_ref,
            "model_ref": variant.model_ref,
            "primary_metric": variant.primary_metric,
            "guardrail_metric": variant.guardrail_metric,
        },
        final_decision={
            "artifact": variant.assumptions["report_artifact"],
            "decision": variant.objective,
        },
        big_takeaways=(
            f"{profile.label} changes the dominant constraint.",
            f"{variant.guardrail_metric} remains a guardrail.",
            "Source-traced assumptions belong in MLSysIM or mlsysbook_labs.",
        ),
        reflections={
            "diagnosis": variant.objective,
            "tradeoff": f"Optimize {variant.primary_metric} without violating {variant.guardrail_metric}.",
            "residual_risk": "Scenario defaults may not cover all production cases.",
        },
        residual_risk="Scenario defaults may not cover all production cases.",
        source_trace=source_trace,
        result_snapshot={
            "lab_id": metadata.lab_id,
            "track_id": profile.track_id,
            "scenario_id": variant.scenario_id,
            "hardware_ref": variant.hardware_ref,
            "model_ref": variant.model_ref,
            "system_ref": variant.system_ref,
        },
    )


def _build_incomplete_report(path: str, track_id: str):
    metadata = LAB_CATALOG[path]
    variant = get_lab_track_variant(metadata.lab_id, track_id)
    profile = get_track_profile(track_id)
    return build_lab_report(
        metadata,
        track=profile.label,
        scenario=variant.scenario_id,
        source_trace=variant_source_trace(variant, profile),
    )


def _report_call_keyword_sets(source_path: Path) -> list[tuple[int, set[str]]]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    calls: list[tuple[int, set[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            name = ""
        if name != "build_lab_report":
            continue
        calls.append((node.lineno, {kw.arg for kw in node.keywords if kw.arg}))
    return calls


def test_every_catalog_lab_can_generate_complete_report_for_every_track():
    for path in LAB_CATALOG:
        for track_id in canonical_track_ids():
            report = _build_complete_report(path, track_id)
            for header in REQUIRED_REPORT_HEADERS:
                assert header in report.markdown, f"{path} {track_id} missing {header}"
            for field in REQUIRED_SNAPSHOT_FIELDS:
                assert field in report.snapshot, f"{path} {track_id} missing {field}"
            assert report.snapshot["track"]
            assert report.snapshot["scenario"]
            assert report.snapshot["predictions"]
            assert report.snapshot["evidence_summary"]
            assert report.snapshot["final_decision"]
            assert report.snapshot["big_takeaways"]
            assert report.snapshot["reflections"]
            assert report.snapshot["residual_risk"]
            assert report.snapshot["source_trace"]
            assert report.snapshot["result_snapshot"]
            assert report.snapshot["incomplete_fields"] == []
            assert "## Incomplete Fields" not in report.markdown


def test_every_catalog_lab_marks_missing_report_fields_for_every_track():
    expected_missing = {
        "Predictions",
        "Evidence Summary",
        "Final Decision",
        "Big Takeaways",
        "Reflections",
        "Residual Risk",
    }
    for path in LAB_CATALOG:
        for track_id in canonical_track_ids():
            report = _build_incomplete_report(path, track_id)
            missing = set(report.snapshot["incomplete_fields"])
            assert expected_missing <= missing, (
                f"{path} {track_id} missing markers {expected_missing - missing}"
            )
            assert "## Incomplete Fields" in report.markdown


def test_notebook_report_calls_pass_required_schema_keywords():
    sources = sorted((LABS_ROOT / "vol1").glob("lab_*.py"))
    sources += sorted((LABS_ROOT / "vol2").glob("lab_*.py"))
    sources += [
        LABS_ROOT / "mlsysbook_labs" / "migration.py",
        LABS_ROOT / "mlsysbook_labs" / "system_design.py",
    ]

    failures = []
    for source_path in sources:
        source = source_path.read_text(encoding="utf-8")
        # Shared-renderer notebook shells delegate report construction to
        # mlsysbook_labs.system_design.render_system_design_lab.
        if "render_system_design_lab" in source and "def render_system_design_lab" not in source:
            continue
        for lineno, keywords in _report_call_keyword_sets(source_path):
            missing = REQUIRED_BUILD_REPORT_KEYWORDS - keywords
            if missing:
                rel = source_path.relative_to(REPO_ROOT)
                failures.append(f"{rel}:{lineno} missing {sorted(missing)}")

    assert not failures, "build_lab_report calls missing schema keywords:\n" + "\n".join(failures)
