from __future__ import annotations

from mlsysbook_labs import (
    ALL_LAB_IDS,
    CANONICAL_TRACKS,
    get_lab_arc_step,
    get_lab_track_variant,
    list_lab_arc_steps,
    list_lab_variants,
    list_track_arcs,
    track_arc_context_summary,
    validate_track_arcs,
)


def test_track_arc_registry_matches_catalog_and_tracks():
    assert validate_track_arcs() == ()
    assert tuple(arc.track_id for arc in list_track_arcs()) == tuple(
        profile.track_id for profile in CANONICAL_TRACKS
    )
    assert tuple(step.lab_id for step in list_lab_arc_steps()) == ALL_LAB_IDS
    assert len(list_lab_arc_steps()) == 35


def test_arc_steps_cover_both_volumes_in_order():
    steps = list_lab_arc_steps()
    volume_1 = tuple(step for step in steps if step.volume == "Volume I")
    volume_2 = tuple(step for step in steps if step.volume == "Volume II")

    assert len(volume_1) == 17
    assert len(volume_2) == 18
    assert tuple(step.sequence for step in volume_1) == tuple(range(1, 18))
    assert tuple(step.sequence for step in volume_2) == tuple(range(1, 19))
    assert get_lab_arc_step("v1_00_architects_portal").concept == "Track selection"
    assert get_lab_arc_step("v2_17_fleet_synthesis").concept == "Volume II synthesis"


def test_track_variants_stay_inside_allowed_arc_families():
    arcs = {arc.track_id: arc for arc in list_track_arcs()}
    for lab_id in ALL_LAB_IDS:
        for variant in list_lab_variants(lab_id):
            arc = arcs[variant.track_id]
            assert variant.hardware_ref in arc.allowed_hardware_refs
            assert variant.model_ref in arc.allowed_model_refs


def test_expected_device_model_pairings_are_enforced():
    assert get_lab_track_variant("v1_10_compression_paradox", "iphone").model_ref == "Models.Vision.MobileNetV2"
    assert get_lab_track_variant("v1_10_compression_paradox", "oura_ring").model_ref == "Models.Tiny.DS_CNN"
    assert get_lab_track_variant("v2_13_price_of_privacy", "oura_ring").model_ref == "Models.Tiny.DS_CNN"
    assert (
        get_lab_track_variant("v1_10_compression_paradox", "robotaxi").hardware_ref
        == "Hardware.Edge.RoboTaxi"
    )
    assert get_lab_track_variant("v2_17_fleet_synthesis", "cloud_fleet").model_ref in {
        "Models.Language.BERT_Base",
        "Models.Language.GPT2",
        "Models.Language.Llama2_70B",
    }


def test_track_arc_context_is_student_facing_not_provenance_copy():
    forbidden = ("Source Trace", "MLSysIM", "Hardware.", "Models.", "Systems.")
    for lab_id in ALL_LAB_IDS:
        for track_id in (profile.track_id for profile in CANONICAL_TRACKS):
            summary = track_arc_context_summary(track_id, lab_id)
            assert set(summary) == {
                "Track mission",
                "System goal",
                "This lab's role",
                "Carry forward",
                "Volume arc",
            }
            assert all(summary.values())
            rendered_text = "\n".join(summary.values())
            for term in forbidden:
                assert term not in rendered_text


def test_fleet_orientation_precedes_technical_arc_without_claiming_engineering_evidence():
    orientation = get_lab_arc_step("v2_00")
    technical_start = get_lab_arc_step("v2_01_scale_illusion")
    assert orientation.sequence == 1
    assert orientation.chapter == "Orientation"
    assert orientation.concept == "Interface rehearsal"
    assert "No engineering conclusion is established" in orientation.arc_role
    assert "no earlier volume or deployment evidence is required" in orientation.carry_forward
    assert technical_start.sequence == orientation.sequence + 1
    assert technical_start.chapter == "Chapter 1"
    for profile in CANONICAL_TRACKS:
        summary = track_arc_context_summary(profile.track_id, "v2_00")
        assert "rehearse" in summary["This lab's role"]
        assert "Lab 01" in summary["Carry forward"]
