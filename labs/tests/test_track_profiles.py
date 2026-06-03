from __future__ import annotations

from mlsysbook_labs import (
    CANONICAL_TRACKS,
    LabMetadata,
    build_lab_report,
    get_track_profile,
    normalize_track_id,
    report_text_fallback,
    track_options,
    track_profile_map,
)


def test_canonical_track_profiles_are_specific_devices_or_systems():
    profiles = track_profile_map()

    assert tuple(profile.track_id for profile in CANONICAL_TRACKS) == (
        "iphone",
        "oura_ring",
        "robotaxi",
        "cloud_fleet",
    )
    assert profiles["iphone"].hardware_ref == "Hardware.Mobile.iPhone15Pro"
    assert profiles["oura_ring"].hardware_ref == "Hardware.Tiny.OuraRing"
    assert profiles["robotaxi"].hardware_ref == "Hardware.Edge.RoboTaxi"
    assert profiles["cloud_fleet"].hardware_ref == "Hardware.Cloud.H100"
    assert profiles["cloud_fleet"].system_ref == "Systems.Clusters.Lab_64_H100"


def test_track_aliases_preserve_legacy_category_inputs():
    assert normalize_track_id("mobile") == "iphone"
    assert normalize_track_id("TinyML") == "oura_ring"
    assert normalize_track_id("edge") == "robotaxi"
    assert normalize_track_id("cloud/fleet") == "cloud_fleet"
    assert get_track_profile("robo taxi").track_id == "robotaxi"
    assert track_options() == {
        "iPhone": "iphone",
        "Oura Ring": "oura_ring",
        "RoboTaxi": "robotaxi",
        "Cloud Fleet": "cloud_fleet",
    }


def test_lab_report_uses_contract_headers_and_incomplete_fields():
    metadata = LabMetadata(
        lab_id="test_lab",
        title="Test Lab",
        volume="Volume I",
        chapter="Test",
        book_anchor="Test anchor",
    )
    report = build_lab_report(
        metadata,
        track="iphone",
        scenario="compression",
        learning_objectives=("Compare candidate configurations.",),
        predictions={"part_a": "SRAM binds first"},
        evidence_summary={"latency_ms": 12.4, "binding_constraint": "thermal"},
        final_decision="Use INT8 with thermal validation.",
        big_takeaways=("Track constraints change the deployable answer.",),
        reflections={"diagnosis": "The same model is not feasible everywhere."},
        residual_risk="Thermal throttling could invalidate the latency estimate.",
        source_trace={"hardware_ref": "Hardware.Mobile.iPhone15Pro"},
    )

    required_headers = (
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
    for header in required_headers:
        assert header in report.markdown

    assert "## Incomplete Fields" not in report.markdown
    assert report.snapshot["source_trace"]["hardware_ref"] == "Hardware.Mobile.iPhone15Pro"
    assert report_text_fallback(report) == report.markdown


def test_lab_report_marks_missing_required_fields():
    metadata = LabMetadata(
        lab_id="test_lab",
        title="Test Lab",
        volume="Volume I",
        chapter="Test",
        book_anchor="Test anchor",
    )
    report = build_lab_report(metadata)

    assert "## Incomplete Fields" in report.markdown
    assert "Predictions" in report.snapshot["incomplete_fields"]
    assert "Evidence Summary" in report.snapshot["incomplete_fields"]
    assert "Final Decision" in report.snapshot["incomplete_fields"]
