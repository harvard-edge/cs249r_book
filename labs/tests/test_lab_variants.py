from __future__ import annotations

from mlsysbook_labs import (
    ALL_LAB_IDS,
    LAB_CATALOG,
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

    assert tuple(coverage) == ALL_LAB_IDS
    assert len(ALL_LAB_IDS) == len(LAB_CATALOG) == 34
    for lab_id in ALL_LAB_IDS:
        assert coverage[lab_id] == expected
        assert len(list_lab_variants(lab_id)) == len(expected)


def test_variant_track_refs_match_canonical_profiles():
    for lab_id in ALL_LAB_IDS:
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
    for lab_id in ALL_LAB_IDS:
        for variant in list_lab_variants(lab_id):
            scenario_ids.add(variant.scenario_id)
            assert _resolve_ref(variant.hardware_ref) is not None
            assert _resolve_ref(variant.model_ref) is not None
            if variant.system_ref is not None:
                assert _resolve_ref(variant.system_ref) is not None

    variant_count = sum(len(list_lab_variants(lab_id)) for lab_id in ALL_LAB_IDS)
    assert len(scenario_ids) == variant_count


def test_non_pilot_variants_are_marked_as_baseline():
    for lab_id in ALL_LAB_IDS:
        for variant in list_lab_variants(lab_id):
            if lab_id in PILOT_LAB_IDS:
                continue
            assert variant.defaults["implementation_status"] == "baseline_variant_pending_notebook_migration"
            assert variant.assumptions["fallback_variant"] is True


def test_v1_10_variants_define_compression_guardrails():
    for variant in list_lab_variants("v1_10_compression_paradox"):
        assert "candidate_methods" in variant.defaults
        assert "bit_widths" in variant.defaults
        assert "size_limit_ref" in variant.defaults
        assert "max_accuracy_drop" in variant.defaults
        assert "min_speedup" in variant.defaults
        assert "require_hardware_support" in variant.defaults
        assert "validation_tests" in variant.defaults
        assert variant.assumptions["report_artifact"] == "compression deployment recipe"


def test_v1_11_variants_define_roofline_defaults():
    for variant in list_lab_variants("v1_11_hardware_roofline"):
        assert "matrix_dim" in variant.defaults
        assert "precision" in variant.defaults
        assert "compare_tracks" in variant.defaults
        assert "move_the_point" in variant.defaults
        assert "validation_tests" in variant.defaults
        assert variant.assumptions["report_artifact"] == "hardware acceleration diagnosis"


def test_v2_10_variants_define_inference_economy_defaults():
    for variant in list_lab_variants("v2_10_inference_economy"):
        assert "setup_cost" in variant.defaults
        assert "cost_per_event" in variant.defaults
        assert "cost_unit" in variant.defaults
        assert "cost_label" in variant.defaults
        assert "demand_qps" in variant.defaults
        assert "context_tokens" in variant.defaults
        assert "state_kind" in variant.defaults
        assert "precision_bytes" in variant.defaults
        assert "slo_ms" in variant.defaults
        assert "qps_per_slot" in variant.defaults
        assert "validation_tests" in variant.defaults
        assert variant.assumptions["report_artifact"] == "inference serving plan"


def test_v1_12_variants_define_benchmarking_defaults():
    for variant in list_lab_variants("v1_12_benchmarking_trap"):
        assert "benchmark_claim" in variant.defaults
        assert "hidden_failure_metric" in variant.defaults
        assert "component_label" in variant.defaults
        assert "metric_unit" in variant.defaults
        assert "burst_value" in variant.defaults
        assert "component_speedup" in variant.defaults
        assert "serial_pct" in variant.defaults
        assert "duration_s" in variant.defaults
        assert "ambient_c" in variant.defaults
        assert "cooling" in variant.defaults
        assert "accuracy_min_pct" in variant.defaults
        assert "p99_max_ms" in variant.defaults
        assert "power_max_w" in variant.defaults
        assert "throughput_min" in variant.defaults
        assert "tail_base_ms" in variant.defaults
        assert "tail_sigma" in variant.defaults
        assert "tail_slo_ms" in variant.defaults
        assert "validation_tests" in variant.defaults
        assert variant.assumptions["report_artifact"] == "benchmark protocol memo"


def test_v1_13_variants_define_serving_defaults():
    for variant in list_lab_variants("v1_13_tail_latency_trap"):
        assert "arrival_qps" in variant.defaults
        assert "service_ms" in variant.defaults
        assert "replicas" in variant.defaults
        assert "slo_ms" in variant.defaults
        assert "service_cv" in variant.defaults
        assert "batch_size" in variant.defaults
        assert "batch_efficiency_gain" in variant.defaults
        assert "context_tokens" in variant.defaults
        assert "state_kind" in variant.defaults
        assert "precision_bytes" in variant.defaults
        assert "devices_per_replica" in variant.defaults
        assert "deserialize_gbs" in variant.defaults
        assert "runtime_init_ms" in variant.defaults
        assert "warmup_ms" in variant.defaults
        assert "warm_pool_replicas" in variant.defaults
        assert "scale_out_replicas" in variant.defaults
        assert "validation_tests" in variant.defaults
        assert variant.assumptions["report_artifact"] == "serving SLA plan"


def test_v1_14_variants_define_ops_defaults():
    for variant in list_lab_variants("v1_14_silent_degradation"):
        assert "drift_source" in variant.defaults
        assert "monitoring_signal" in variant.defaults
        assert "rollback_policy" in variant.defaults
        assert "escalation_policy" in variant.defaults
        assert "baseline_quality_pct" in variant.defaults
        assert "quality_floor_pct" in variant.defaults
        assert "drift_rate_psi_per_day" in variant.defaults
        assert "quality_loss_per_psi" in variant.defaults
        assert "alert_threshold_psi" in variant.defaults
        assert "label_delay_days" in variant.defaults
        assert "retrain_cost" in variant.defaults
        assert "drift_cost_per_day" in variant.defaults
        assert "current_cadence_days" in variant.defaults
        assert "monitoring_cost_per_day" in variant.defaults
        assert "downstream_models" in variant.defaults
        assert "base_loss_pp" in variant.defaults
        assert "validation_tests" in variant.defaults
        assert variant.assumptions["report_artifact"] == "operations policy memo"


def test_v1_15_variants_define_responsibility_defaults():
    for variant in list_lab_variants("v1_15_no_free_fairness"):
        assert "harmed_party" in variant.defaults
        assert "obligation" in variant.defaults
        assert "audit_signal" in variant.defaults
        assert "subgroups" in variant.defaults
        assert "baseline_quality_pct" in variant.defaults
        assert "baseline_gap_pp" in variant.defaults
        assert "target_gap_pp" in variant.defaults
        assert "fairness_sensitivity" in variant.defaults
        assert "explanation_features" in variant.defaults
        assert "explanation_method" in variant.defaults
        assert "explanation_coverage_pct" in variant.defaults
        assert "base_latency_ms" in variant.defaults
        assert "latency_slo_ms" in variant.defaults
        assert "inference_events_per_day" in variant.defaults
        assert "retrain_frequency_per_year" in variant.defaults
        assert "train_energy_kwh" in variant.defaults
        assert "grid_ci_g_per_kwh" in variant.defaults
        assert "max_energy_factor" in variant.defaults
        assert "max_cost_factor" in variant.defaults
        assert "governance_delay_days" in variant.defaults
        assert "residual_harm" in variant.defaults
        assert "validation_tests" in variant.defaults
        assert variant.assumptions["report_artifact"] == "responsible engineering decision memo"


def test_v2_06_variants_define_collective_defaults():
    for variant in list_lab_variants("v2_06_collective_communication"):
        assert "operation" in variant.defaults
        assert "message_gb" in variant.defaults
        assert "participants" in variant.defaults
        assert "fabric" in variant.defaults
        assert "gpus_per_node" in variant.defaults
        assert "overlap_pct" in variant.defaults
        assert "compression_ratio" in variant.defaults
        assert "topology" in variant.defaults
        assert "optimization_choices" in variant.defaults
        assert "residual_risk" in variant.defaults
        assert "validation_tests" in variant.defaults
        assert variant.assumptions["report_artifact"] == "communication design review"
