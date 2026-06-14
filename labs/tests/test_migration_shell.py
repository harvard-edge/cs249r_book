from __future__ import annotations

from mlsysbook_labs import (
    baseline_big_takeaways,
    baseline_learning_objectives,
    build_migration_report,
    get_lab_metadata,
    get_lab_track_variant,
    get_track_profile,
    variant_source_trace,
)


def test_baseline_copy_uses_metadata_and_variant_metrics():
    metadata = get_lab_metadata("vol1/lab_01_ml_intro.py")
    variant = get_lab_track_variant(metadata.lab_id, "oura_ring")

    objectives = baseline_learning_objectives(metadata, variant)
    takeaways = baseline_big_takeaways(metadata, variant)

    assert metadata.title in objectives[0]
    assert variant.primary_metric in objectives[1]
    assert variant.guardrail_metric in takeaways[1]


def test_variant_source_trace_is_serializable_and_registry_backed():
    metadata = get_lab_metadata("vol2/lab_03_communication.py")
    variant = get_lab_track_variant(metadata.lab_id, "robotaxi")
    profile = get_track_profile("robotaxi")

    trace = variant_source_trace(variant, profile)

    assert trace["track_id"] == "robotaxi"
    assert trace["hardware_ref"] == "Hardware.Edge.RoboTaxi"
    assert trace["model_ref"] == "Models.Vision.YOLOv8_Nano"
    assert trace["assumptions"]["system_design_variant"] is True
    assert trace["assumptions"].get("fallback_variant") is not True


def test_build_migration_report_marks_baseline_gaps():
    metadata = get_lab_metadata("vol1/lab_11_hw_accel.py")
    variant = get_lab_track_variant(metadata.lab_id, "cloud_fleet")
    profile = get_track_profile("cloud_fleet")

    report = build_migration_report(
        metadata,
        profile,
        variant,
        predictions={"legacy_status": "legacy shell report"},
        final_decision="Use the hand-authored track variant while preserving the legacy shell report path.",
        reflections={"diagnosis": "The shell records track context before deeper migration."},
        residual_risk="The legacy shell still needs notebook-specific part checkpoints.",
    )

    assert "## Source Trace" in report.markdown
    assert "Hardware.Cloud.H100" in report.markdown
    assert "## Incomplete Fields" in report.markdown
    assert "Hand-authored track variant" not in report.snapshot["incomplete_fields"]
    assert "Deep notebook part checkpoints" in report.snapshot["incomplete_fields"]
