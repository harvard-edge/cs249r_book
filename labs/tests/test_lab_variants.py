from __future__ import annotations

from mlsysbook_labs import (
    PILOT_LAB_IDS,
    canonical_track_ids,
    get_lab_track_variant,
    get_track_profile,
    list_lab_variants,
    resolve_mlsysim_ref,
    variant_coverage,
)


def _resolve_ref(ref: str):
    return resolve_mlsysim_ref(ref)


def test_pilot_labs_have_all_canonical_track_variants():
    expected = canonical_track_ids()
    coverage = variant_coverage()

    assert tuple(coverage) == PILOT_LAB_IDS
    for lab_id in PILOT_LAB_IDS:
        assert coverage[lab_id] == expected
        assert len(list_lab_variants(lab_id)) == len(expected)


def test_variant_track_refs_match_canonical_profiles():
    for lab_id in PILOT_LAB_IDS:
        for variant in list_lab_variants(lab_id):
            profile = get_track_profile(variant.track_id)
            assert variant.hardware_ref == profile.hardware_ref
            assert variant.system_ref == profile.system_ref
            assert variant.stakeholder
            assert variant.workload_summary
            assert variant.objective
            assert variant.primary_metric
            assert variant.guardrail_metric
            assert variant.defaults
            assert variant.assumptions


def test_variant_alias_lookup():
    assert get_lab_track_variant("v1_10_compression_paradox", "mobile").track_id == "iphone"
    assert get_lab_track_variant("v1_10_compression_paradox", "tinyml").track_id == "oura_ring"
    assert get_lab_track_variant("v2_11_edge_thermodynamics", "edge").track_id == "robotaxi"
    assert get_lab_track_variant("v2_11_edge_thermodynamics", "cloud").track_id == "cloud_fleet"


def test_variant_registry_paths_resolve():
    scenario_ids = set()
    for lab_id in PILOT_LAB_IDS:
        for variant in list_lab_variants(lab_id):
            scenario_ids.add(variant.scenario_id)
            assert _resolve_ref(variant.hardware_ref) is not None
            assert _resolve_ref(variant.model_ref) is not None
            if variant.system_ref is not None:
                assert _resolve_ref(variant.system_ref) is not None

    variant_count = sum(len(list_lab_variants(lab_id)) for lab_id in PILOT_LAB_IDS)
    assert len(scenario_ids) == variant_count


def test_v1_10_variants_define_compression_guardrails():
    for variant in list_lab_variants("v1_10_compression_paradox"):
        assert "candidate_methods" in variant.defaults
        assert "bit_widths" in variant.defaults
        assert "size_limit_ref" in variant.defaults
        assert "max_accuracy_drop" in variant.defaults
        assert "min_speedup" in variant.defaults
        assert "require_hardware_support" in variant.defaults
        assert variant.assumptions["report_artifact"] == "compression deployment recipe"
