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


def test_all_catalog_labs_are_now_pilot_variants():
    assert set(PILOT_LAB_IDS) == set(ALL_LAB_IDS)
    for lab_id in ALL_LAB_IDS:
        for variant in list_lab_variants(lab_id):
            assert variant.assumptions.get("fallback_variant") is not True


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


def test_v1_01_variants_define_triad_defaults():
    for variant in list_lab_variants("v1_01_ai_triad"):
        assert "failure_story" in variant.defaults
        assert "data_axis" in variant.defaults
        assert "algorithm_axis" in variant.defaults
        assert "machine_axis" in variant.defaults
        assert "data_threshold_pct" in variant.defaults
        assert "algorithm_threshold_pct" in variant.defaults
        assert "machine_threshold_pct" in variant.defaults
        assert "default_data_pct" in variant.defaults
        assert "default_algorithm_pct" in variant.defaults
        assert "default_machine_pct" in variant.defaults
        assert "intervention_options" in variant.defaults
        assert "validation_tests" in variant.defaults
        assert variant.assumptions["report_artifact"] == "triad diagnosis memo"


def test_v1_02_variants_define_deployment_envelopes():
    for variant in list_lab_variants("v1_02_physics_of_deployment"):
        assert "envelope_story" in variant.defaults
        assert "workload_knob" in variant.defaults
        assert "workload_unit" in variant.defaults
        assert "knob_min" in variant.defaults
        assert "knob_max" in variant.defaults
        assert "knob_step" in variant.defaults
        assert "default_knob" in variant.defaults
        assert "memory_budget_mb" in variant.defaults
        assert "flash_budget_mb" in variant.defaults
        assert "latency_budget_ms" in variant.defaults
        assert "energy_budget_mj" in variant.defaults
        assert "power_budget_w" in variant.defaults
        assert "bandwidth_budget_gbs" in variant.defaults
        assert "cost_budget_per_1k" in variant.defaults
        assert "placement_options" in variant.defaults
        assert "mitigation_options" in variant.defaults
        assert variant.assumptions["report_artifact"] == "physics-of-deployment memo"


def test_v1_03_variants_define_workflow_defaults():
    for variant in list_lab_variants("v1_03_constraint_tax"):
        assert "constraint_name" in variant.defaults
        assert "failure_story" in variant.defaults
        assert "stage_names" in variant.defaults
        assert "default_discovery_stage" in variant.defaults
        assert "recommended_gate_stage" in variant.defaults
        assert "base_rework_days" in variant.defaults
        assert "base_cycle_days" in variant.defaults
        assert "base_residual_risk_pct" in variant.defaults
        assert "min_residual_risk_pct" in variant.defaults
        assert "default_validation_depth_pct" in variant.defaults
        assert "default_automation_pct" in variant.defaults
        assert "default_hardware_realism_pct" in variant.defaults
        assert "default_data_scale_pct" in variant.defaults
        assert "gate_options" in variant.defaults
        assert "release_policies" in variant.defaults
        assert "rollback_rules" in variant.defaults
        assert variant.assumptions["report_artifact"] == "workflow policy memo"


def test_v1_04_variants_define_data_pipeline_defaults():
    for variant in list_lab_variants("v1_04_data_gravity"):
        assert "data_source" in variant.defaults
        assert "data_rate_mb_s" in variant.defaults
        assert "burst_multiplier" in variant.defaults
        assert "ingest_capacity_mb_s" in variant.defaults
        assert "preprocess_capacity_mb_s" in variant.defaults
        assert "storage_capacity_mb_s" in variant.defaults
        assert "upload_capacity_mb_s" in variant.defaults
        assert "retention_days" in variant.defaults
        assert "local_storage_mb" in variant.defaults
        assert "privacy_stance" in variant.defaults
        assert "movement_strategies" in variant.defaults
        assert "retention_options" in variant.defaults
        assert variant.assumptions["report_artifact"] == "data pipeline architecture memo"


def test_v1_05_variants_define_neural_compute_defaults():
    for variant in list_lab_variants("v1_05_neural_computation"):
        assert "operator_story" in variant.defaults
        assert "tensor_label" in variant.defaults
        assert "batch" in variant.defaults
        assert "channels" in variant.defaults
        assert "height" in variant.defaults
        assert "width" in variant.defaults
        assert "sequence" in variant.defaults
        assert "hidden" in variant.defaults
        assert "precision_bytes" in variant.defaults
        assert "activation_budget_mb" in variant.defaults
        assert "bandwidth_budget_gbs" in variant.defaults
        assert "latency_budget_ms" in variant.defaults
        assert "power_budget_w" in variant.defaults
        assert "default_shape_multiplier" in variant.defaults
        assert "activation_multiplier" in variant.defaults
        assert "ops_gmac_at_default" in variant.defaults
        assert "design_options" in variant.defaults
        assert variant.assumptions["report_artifact"] == "operator budget note"


def test_v1_06_variants_define_architecture_defaults():
    for variant in list_lab_variants("v1_06_architecture_tax"):
        assert "architecture_story" in variant.defaults
        assert "workload_label" in variant.defaults
        assert "scaling_variable" in variant.defaults
        assert "scaling_unit" in variant.defaults
        assert "default_scale" in variant.defaults
        assert "scale_min" in variant.defaults
        assert "scale_max" in variant.defaults
        assert "memory_budget_mb" in variant.defaults
        assert "latency_budget_ms" in variant.defaults
        assert "power_budget_w" in variant.defaults
        assert "quality_floor_pct" in variant.defaults
        assert "kernel_support_floor_pct" in variant.defaults
        assert "candidate_architectures" in variant.defaults
        assert "validation_tests" in variant.defaults
        assert variant.assumptions["report_artifact"] == "architecture recommendation memo"


def test_v1_07_variants_define_framework_defaults():
    for variant in list_lab_variants("v1_07_framework_tax"):
        assert "runtime_story" in variant.defaults
        assert "workload_label" in variant.defaults
        assert "op_count" in variant.defaults
        assert "compute_us_per_op" in variant.defaults
        assert "transfer_us" in variant.defaults
        assert "sync_us" in variant.defaults
        assert "memory_us" in variant.defaults
        assert "shape_dynamism_pct" in variant.defaults
        assert "default_reuse_count" in variant.defaults
        assert "latency_budget_ms" in variant.defaults
        assert "memory_budget_mb" in variant.defaults
        assert "kernel_support_floor_pct" in variant.defaults
        assert "runtime_options" in variant.defaults
        assert "validation_tests" in variant.defaults
        assert variant.assumptions["report_artifact"] == "runtime deployment recommendation"


def test_v1_08_variants_define_training_defaults():
    for variant in list_lab_variants("v1_08_training_gauntlet"):
        assert "training_story" in variant.defaults
        assert "workload_label" in variant.defaults
        assert "default_batch_size" in variant.defaults
        assert "batch_min" in variant.defaults
        assert "batch_max" in variant.defaults
        assert "sample_mb" in variant.defaults
        assert "activation_mb_per_mparam" in variant.defaults
        assert "training_memory_budget_mb" in variant.defaults
        assert "throughput_budget_samples_s" in variant.defaults
        assert "quality_floor_pct" in variant.defaults
        assert "deployment_handoff" in variant.defaults
        assert "strategy_options" in variant.defaults
        assert "validation_tests" in variant.defaults
        assert variant.assumptions["report_artifact"] == "training feasibility plan"


def test_v1_09_variants_define_selection_defaults():
    for variant in list_lab_variants("v1_09_selection_paradox"):
        assert "selection_story" in variant.defaults
        assert "dataset_unit" in variant.defaults
        assert "dataset_size_k" in variant.defaults
        assert "cost_per_k" in variant.defaults
        assert "compute_cost_per_k" in variant.defaults
        assert "storage_mb_per_k" in variant.defaults
        assert "cost_budget" in variant.defaults
        assert "storage_budget_mb" in variant.defaults
        assert "quality_floor_pct" in variant.defaults
        assert "coverage_floor_pct" in variant.defaults
        assert "rare_event_floor_pct" in variant.defaults
        assert "subgroups" in variant.defaults
        assert "policy_options" in variant.defaults
        assert "validation_tests" in variant.defaults
        assert variant.assumptions["report_artifact"] == "data selection policy memo"


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


def test_v1_16_variants_define_capstone_defaults():
    for variant in list_lab_variants("v1_16_architects_audit"):
        assert "architecture_goal" in variant.defaults
        assert "architecture_components" in variant.defaults
        assert "prior_decisions" in variant.defaults
        assert "sensitivity_defaults" in variant.defaults
        assert "revision_options" in variant.defaults
        assert "top_risks" in variant.defaults
        assert "durable_principle" in variant.defaults
        assert "validation_tests" in variant.defaults
        assert variant.assumptions["report_artifact"] == "Volume I architecture memo"


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


def test_remaining_v2_variants_define_system_design_defaults():
    lab_ids = (
        "v2_01_scale_illusion",
        "v2_02_compute_wall",
        "v2_03_network_fabric_design",
        "v2_04_data_pipeline_wall",
        "v2_05_parallelism_design",
        "v2_07_failure_budget_engineering",
        "v2_08_fleet_orchestration",
        "v2_09_optimization_trap",
        "v2_12_silent_fleet",
        "v2_13_price_of_privacy",
        "v2_14_robustness_budget",
        "v2_15_carbon_budget",
        "v2_16_fairness_budget",
        "v2_17_fleet_synthesis",
    )
    for lab_id in lab_ids:
        for variant in list_lab_variants(lab_id):
            assert "concept_label" in variant.defaults
            assert "decision_story" in variant.defaults
            assert "knob_label" in variant.defaults
            assert "knob_unit" in variant.defaults
            assert "default_knob" in variant.defaults
            assert "knob_min" in variant.defaults
            assert "knob_max" in variant.defaults
            assert "knob_step" in variant.defaults
            assert "capacity_budget" in variant.defaults
            assert "latency_budget_ms" in variant.defaults
            assert "cost_budget" in variant.defaults
            assert "quality_floor_pct" in variant.defaults
            assert "guardrail_floor_pct" in variant.defaults
            assert "decision_options" in variant.defaults
            assert "validation_tests" in variant.defaults
            assert variant.assumptions["system_design_variant"] is True
            assert variant.assumptions.get("fallback_variant") is not True
            assert variant.assumptions["report_artifact"]
