"""
Canonical physics and accounting formulas for ML systems.

Domain modules:
  networking, performance, economics, memory, communication,
  reliability, transformer, serving, statistics, agents, context, retrieval, tool_calls, drivetrain, robotics
"""

from ._units import _ensure_unit
from .constants import SPEED_OF_LIGHT_FIBER_KM_S, INSULATION_CLASS_F_LIMIT
from .networking import calc_network_latency_ms
from .performance import (
    dTime,
    calc_training_time,
    calc_amdahls_speedup,
    calc_partial_slowdown_stretch,
    calc_stall_share_breakeven,
    calc_strong_scaling_speedup,
    calc_universal_scalability_law,
    calc_usl_optimal_concurrency,
    calc_bottleneck,
    calc_pipeline_bubble,
    calc_effective_flops,
)
from .economics import calc_monthly_egress_cost, calc_fleet_tco, calc_payback_period
from .memory import (
    model_memory,
    calc_activation_memory,
    calc_checkpoint_size,
    calc_kv_cache_size,
    calc_mla_cache_size,
    calc_paged_kv_cache_size,
    calc_capped_exponential_scale,
    calc_expected_paged_kv_tokens,
)
from .memory import calc_decoder_matmul_weights  # added 2026-10-20
from .communication import (
    calc_alpha_beta_crossover,
    calc_ring_attention_min_chunk,
    calc_ring_collective_data_factor,
    calc_bisection_bandwidth,
    calc_ring_allreduce_data_factor,
    ring_allreduce_data_factor_latex,
    calc_ring_allreduce_latency_steps,
    calc_ring_allreduce_latency_time,
    calc_hop_latency,
    calc_fat_tree_hosts,
    calc_fat_tree_switches,
    calc_fat_tree_path_switches,
    calc_torus_bisection_links,
    calc_torus_diameter,
    calc_oversubscription_effect,
    calc_point_to_point_time,
    calc_double_binary_tree_allreduce_time,
    calc_ring_allreduce_time,
    calc_ring_tree_crossover_size,
    calc_tree_allreduce_time,
    calc_all_to_all_time,
    calc_hierarchical_allreduce_time,
)
from .reliability import (
    calc_young_daly_interval,
    calc_mtbf_cluster,
    calc_mtbf_node,
    calc_availability_stacked,
    calc_failure_probability,
)
from .transformer import (
    calc_transformer_training_flops,
    calc_transformer_decode_flops,
    calc_transformer_prefill_flops,
    calc_transformer_block_weights,
    calc_attention_score_macs,
    calc_attention_crossover,
    calc_vla_inference_latency,
)
from .serving import calc_queue_latency_mmc, calc_mg1_wait_ratio
from .statistics import (
    calc_population_stability_index,
    calc_two_proportion_sample_size,
    calc_two_proportion_z,
    calc_proportion_gap_interval,
    calc_equal_n_for_z,
    calc_monte_carlo_sample_size,
    calc_importance_sampling_exponential,
    calc_optimal_exponential_proposal_mean,
    calc_cross_entropy_exponential_proposal,
    calc_conjunctive_confidence,
    calc_multilegged_confidence,
    calc_chance_agreement,
    calc_cohens_kappa,
    calc_ks_critical_value,
    calc_ks_normal_location_shift,
)
from .agents import (
    calc_kv_cache_bytes_per_token,
    calc_unchecked_success,
    calc_step_error_for_target,
    calc_first_horizon_below,
    calc_cost_per_accepted,
    calc_break_even_acceptance,
    calc_independent_success,
    calc_append_only_context,
    calc_tempered_softmax,
    calc_nucleus,
    calc_pass_posterior,
    calc_reused_prefix_prompt_tokens,
    calc_pass_at_k_unbiased,
    calc_expected_max_std_normal,
    calc_pruned_tree_nodes,
)
from .drivetrain import (
    calc_traction_deceleration_limit,
    calc_required_friction,
    calc_tipover_deceleration_limit,
    calc_max_cg_height_ratio,
    calc_drive_torque_deceleration_limit,
    calc_drive_voltage_headroom,
    calc_braking_limits,
)
from .robotics import (
    calc_safe_stopping_distance,
    calc_max_permitted_velocity,
    calc_c2_stop_suffix,
    calc_scurve_stop_distance,
    calc_scurve_stop_time,
    calc_scurve_max_permitted_velocity,
    calc_latch_contact_stopping_profile,
    calc_stopping_crossover_speed,
    calc_braking_time,
    calc_travel_between_speeds,
    calc_kinetic_energy,
    calc_seam_acceleration_torque_jump,
    calc_inverted_pendulum_fall_time,
    calc_reflected_inertia,
    calc_spatial_information_age_displacement,
    calc_empirical_testing_exposure,
    calc_3dgs_memory_footprint,
    calc_voltage_droop,
    calc_action_chunk_streaming_amortization,
    calc_chunk_renewal_schedule,
    calc_temporal_ensemble_weights,
    calc_watchdog_lease_bound,
    calc_phase_lag_total,
    calc_canfd_frame_bits,
    calc_canfd_bus_utilization,
    calc_ethercat_cycle_time,
    calc_response_time_bound,
    calc_contact_force,
    calc_contact_yield_deadline,
    calc_inductive_voltage_droop,
    calc_alpha_power_gate_delay_stretch,
    calc_clopper_pearson_zero_failure_bound,
    calc_zero_failure_sample_size,
    calc_reduced_mass,
    calc_transient_impact_force,
    calc_transient_contact_speed_limit,
    calc_tripwire_contact_force_accumulation,
    calc_shielded_system_hazard_rate,
    calc_cbf_qp_orthogonal_projection,
    calc_quintic_blend_duration,
    calc_teleop_ingestion_budget,
    calc_tsdf_voxel_grid_budget,
    calc_target_evidence_horizon,
    calc_linear_invalidation_horizon,
    calc_tracked_evidence_horizon,
    calc_free_space_expiry,
    calc_scurve_rest_to_rest_time,
    calc_quasistatic_spill_acceleration,
    calc_axi_fifo_contention,
)

from .power import calc_rail_current_step, calc_inductive_droop
from .scaling import calc_halving_multiple, calc_compute_optimal_allocation
from .reliability import calc_all_up_availability, calc_downtime
from .reliability import (  # added 2026-10-20
    calc_checkpoint_waste,
    calc_step_event_probability,
    calc_probe_detection_window,
)
from .thermal import calc_coolant_mass_flow
from .quantities import (
    transfer_time,
    compute_time,
    energy_from_power,
    carbon_from_energy,
    memory_from_params,
    token_throughput,
)

__all__ = [
    # Physical constants (book LEGO cells use `from mlsysim import *`, which
    # only sees names listed here — audit fix 2026-06-06).
    "SPEED_OF_LIGHT_FIBER_KM_S",
    "INSULATION_CLASS_F_LIMIT",
    "calc_network_latency_ms",
    "dTime",
    "calc_training_time",
    "calc_amdahls_speedup",
    "calc_partial_slowdown_stretch",
    "calc_stall_share_breakeven",
    "calc_strong_scaling_speedup",
    "calc_universal_scalability_law",
    "calc_usl_optimal_concurrency",
    "calc_bottleneck",
    "calc_pipeline_bubble",
    "calc_effective_flops",
    "calc_monthly_egress_cost",
    "calc_fleet_tco",
    "calc_payback_period",
    "model_memory",
    "calc_activation_memory",
    "calc_checkpoint_size",
    "calc_kv_cache_size",
    "calc_mla_cache_size",
    "calc_paged_kv_cache_size",
    "calc_capped_exponential_scale",
    "calc_expected_paged_kv_tokens",
    "calc_ring_allreduce_time",
    "calc_point_to_point_time",
    "calc_ring_allreduce_data_factor",
    "ring_allreduce_data_factor_latex",
    "calc_ring_collective_data_factor",
    "calc_ring_allreduce_latency_steps",
    "calc_ring_allreduce_latency_time",
    "calc_alpha_beta_crossover",
    "calc_ring_attention_min_chunk",
    "calc_oversubscription_effect",
    "calc_bisection_bandwidth",
    "calc_hop_latency",
    "calc_fat_tree_hosts",
    "calc_fat_tree_switches",
    "calc_fat_tree_path_switches",
    "calc_torus_bisection_links",
    "calc_torus_diameter",
    "calc_tree_allreduce_time",
    "calc_double_binary_tree_allreduce_time",
    "calc_ring_tree_crossover_size",
    "calc_all_to_all_time",
    "calc_hierarchical_allreduce_time",
    "calc_young_daly_interval",
    "calc_mtbf_cluster",
    "calc_mtbf_node",
    "calc_availability_stacked",
    "calc_failure_probability",
    "calc_transformer_training_flops",
    "calc_transformer_decode_flops",
    "calc_transformer_prefill_flops",
    "calc_transformer_block_weights",
    "calc_attention_score_macs",
    "calc_attention_crossover",
    "calc_vla_inference_latency",
    "calc_queue_latency_mmc",
    "calc_mg1_wait_ratio",
    "calc_population_stability_index",
    "calc_monte_carlo_sample_size",
    "calc_importance_sampling_exponential",
    "calc_optimal_exponential_proposal_mean",
    "calc_cross_entropy_exponential_proposal",
    "calc_conjunctive_confidence",
    "calc_multilegged_confidence",
    "calc_chance_agreement",
    "calc_cohens_kappa",
    "calc_ks_critical_value",
    "calc_ks_normal_location_shift",
    "calc_two_proportion_sample_size",
    "calc_two_proportion_z",
    "calc_proportion_gap_interval",
    "calc_equal_n_for_z",
    "calc_kv_cache_bytes_per_token",
    "calc_unchecked_success",
    "calc_step_error_for_target",
    "calc_first_horizon_below",
    "calc_cost_per_accepted",
    "calc_break_even_acceptance",
    "calc_independent_success",
    "calc_append_only_context",
    "calc_tempered_softmax",
    "calc_nucleus",
    "calc_pass_posterior",
    "calc_reused_prefix_prompt_tokens",
    "calc_pass_at_k_unbiased",
    "calc_expected_max_std_normal",
    "calc_pruned_tree_nodes",
    "calc_safe_stopping_distance",
    "calc_max_permitted_velocity",
    "calc_c2_stop_suffix",
    "calc_scurve_stop_distance",
    "calc_scurve_stop_time",
    "calc_scurve_max_permitted_velocity",
    "calc_latch_contact_stopping_profile",
    "calc_stopping_crossover_speed",
    "calc_braking_time",
    "calc_travel_between_speeds",
    "calc_kinetic_energy",
    "calc_seam_acceleration_torque_jump",
    "calc_inverted_pendulum_fall_time",
    "calc_reflected_inertia",
    "calc_spatial_information_age_displacement",
    "calc_empirical_testing_exposure",
    "calc_3dgs_memory_footprint",
    "calc_voltage_droop",
    "calc_traction_deceleration_limit",
    "calc_required_friction",
    "calc_tipover_deceleration_limit",
    "calc_max_cg_height_ratio",
    "calc_drive_torque_deceleration_limit",
    "calc_drive_voltage_headroom",
    "calc_braking_limits",
    "calc_action_chunk_streaming_amortization",
    "calc_chunk_renewal_schedule",
    "calc_temporal_ensemble_weights",
    "calc_watchdog_lease_bound",
    "calc_phase_lag_total",
    "calc_canfd_frame_bits",
    "calc_canfd_bus_utilization",
    "calc_ethercat_cycle_time",
    "calc_response_time_bound",
    "calc_contact_force",
    "calc_contact_yield_deadline",
    "calc_inductive_voltage_droop",
    "calc_alpha_power_gate_delay_stretch",
    "calc_clopper_pearson_zero_failure_bound",
    "calc_zero_failure_sample_size",
    "calc_reduced_mass",
    "calc_transient_impact_force",
    "calc_transient_contact_speed_limit",
    "calc_tripwire_contact_force_accumulation",
    "calc_shielded_system_hazard_rate",
    "calc_cbf_qp_orthogonal_projection",
    "calc_quintic_blend_duration",
    "calc_teleop_ingestion_budget",
    "calc_tsdf_voxel_grid_budget",
    "calc_target_evidence_horizon",
    "calc_linear_invalidation_horizon",
    "calc_tracked_evidence_horizon",
    "calc_free_space_expiry",
    "calc_scurve_rest_to_rest_time",
    "calc_quasistatic_spill_acceleration",
    "calc_axi_fifo_contention",
    "calc_halving_multiple",
    "calc_compute_optimal_allocation",
    "calc_all_up_availability",
    "calc_downtime",
    "calc_checkpoint_waste",
    "calc_step_event_probability",
    "calc_probe_detection_window",
    "calc_rail_current_step",
    "calc_inductive_droop",
    "calc_coolant_mass_flow",
    "transfer_time",
    "compute_time",
    "energy_from_power",
    "carbon_from_energy",
    "memory_from_params",
    "token_throughput",
    "calc_decoder_matmul_weights",  # added 2026-10-20
]


# --- Motors, winding heat, and the zero-failure rate bound (added 2026-10-20) ---
from .constants import COPPER_TEMPERATURE_COEFFICIENT, COPPER_COEFFICIENT_REFERENCE_TEMPERATURE
from .motors import (
    calc_copper_resistance_factor,
    calc_first_order_temperature,
    calc_first_order_time_to_temperature,
    calc_periodic_burst_peak_rise,
    calc_voltage_limited_speed,
    calc_inertia_matched_ratio,
    calc_peak_normalized_response,
    calc_contact_force_rate,
)
from .robotics.assurance import calc_zero_failure_rate_upper_bound

__all__ += [
    "COPPER_TEMPERATURE_COEFFICIENT",
    "COPPER_COEFFICIENT_REFERENCE_TEMPERATURE",
    "calc_copper_resistance_factor",
    "calc_first_order_temperature",
    "calc_first_order_time_to_temperature",
    "calc_periodic_burst_peak_rise",
    "calc_voltage_limited_speed",
    "calc_inertia_matched_ratio",
    "calc_peak_normalized_response",
    "calc_contact_force_rate",
    "calc_zero_failure_rate_upper_bound",
]


# --- Context engineering (added 2026-10-20) ---
from .context import (
    calc_observation_budget,
    calc_turns_that_fit,
    calc_softmax_target_weight,
    calc_equivalent_key_count,
    calc_zoned_prefill_tokens,
    calc_compaction_cycle,
)

__all__ += [
    "calc_observation_budget",
    "calc_turns_that_fit",
    "calc_softmax_target_weight",
    "calc_equivalent_key_count",
    "calc_zoned_prefill_tokens",
    "calc_compaction_cycle",
]


# --- Serving at scale (added 2026-10-20) ---
from .serving import (
    calc_littles_law_occupancy,
    calc_batch_knee,
    calc_batch_throughput_share,
    calc_padding_waste_fraction,
    calc_speculative_expected_tokens,
)

__all__ += [
    "calc_littles_law_occupancy",
    "calc_batch_knee",
    "calc_batch_throughput_share",
    "calc_padding_waste_fraction",
    "calc_speculative_expected_tokens",
]

# --- Retrieval from long-term memory (added 2026-10-20) ---
from .retrieval import (
    calc_records_that_fit,
    calc_bm25_idf,
    calc_bm25_length_normalizer,
    calc_bm25_term_score,
    calc_full_tree_nodes,
    calc_vectors_in_budget,
    calc_bandwidth_fit_through_origin,
    calc_interface_bandwidth,
    calc_word_overlap,
)

__all__ += [
    "calc_records_that_fit",
    "calc_bm25_idf",
    "calc_bm25_length_normalizer",
    "calc_bm25_term_score",
    "calc_full_tree_nodes",
    "calc_vectors_in_budget",
    "calc_bandwidth_fit_through_origin",
    "calc_interface_bandwidth",
    "calc_word_overlap",
]

# --- Tool calling (added 2026-10-20) ---
from .tool_calls import (
    calc_timeout_probability,
    calc_blind_retry_duplicate_probability,
    calc_pending_polls,
    calc_selection_break_even,
    calc_integration_adapters,
    calc_layered_retry_attempts,
    calc_jitter_expected_arrivals,
    calc_bulkhead_admission,
)

__all__ += [
    "calc_layered_retry_attempts",
    "calc_jitter_expected_arrivals",
    "calc_bulkhead_admission",
    "calc_timeout_probability",
    "calc_blind_retry_duplicate_probability",
    "calc_pending_polls",
    "calc_selection_break_even",
    "calc_integration_adapters",
]

# --- Correlated samples and imitation-learning bounds (added 2026-10-20) ---
from .statistics import (
    calc_effective_sample_size,
    calc_behavior_cloning_cost_bound,
    calc_dagger_cost_bound,
)
from .constants import AGE_OF_UNIVERSE

__all__ += [
    "calc_effective_sample_size",
    "calc_behavior_cloning_cost_bound",
    "calc_dagger_cost_bound",
    "AGE_OF_UNIVERSE",
]

# --- KV cache residency and turn rate (added 2026-10-20) ---
from .memory import (
    calc_kv_attention_design,
    calc_kv_resident_fit,
    calc_kv_block_count,
    calc_shared_prefix_fit,
    calc_sharing_gain_limit,
)
from .serving import calc_littles_law_rate, calc_residency_factor
from .transformer import calc_weight_only_step_time

__all__ += [
    "calc_kv_attention_design",
    "calc_kv_resident_fit",
    "calc_kv_block_count",
    "calc_shared_prefix_fit",
    "calc_sharing_gain_limit",
    "calc_littles_law_rate",
    "calc_residency_factor",
    "calc_weight_only_step_time",
]

# --- Agent sandboxes: setup amortization and Poisson reserve (added 2026-10-20) ---
from .agents import calc_setup_share, calc_work_for_setup_cap
from .statistics import calc_poisson_tail, calc_poisson_reserve, calc_poisson_mean_for_reserve
from .statistics import calc_cdf_quantile

__all__ += [
    "calc_setup_share",
    "calc_work_for_setup_cap",
    "calc_poisson_tail",
    "calc_poisson_reserve",
    "calc_poisson_mean_for_reserve",
    "calc_cdf_quantile",
]

# --- On-device training (added 2026-10-20) ---
from .memory import (
    calc_update_step_traffic,
    calc_low_rank_update_params,
    calc_residual_adapter_params,
)

__all__ += [
    "calc_update_step_traffic",
    "calc_low_rank_update_params",
    "calc_residual_adapter_params",
]

# Network-architecture layer laws (added 2026-10-20)
from .layers import (
    calc_receptive_field,
    calc_layers_for_receptive_field,
    calc_separable_cost_fraction,
    calc_gradient_product,
    calc_resnet_bottleneck_macs,
)

__all__ += [
    "calc_receptive_field",
    "calc_layers_for_receptive_field",
    "calc_separable_cost_fraction",
    "calc_gradient_product",
    "calc_resnet_bottleneck_macs",
]

# --- Neural computation and number formats (added 2026-10-20) ---
from .neural import (
    calc_dense_weights,
    calc_dense_biases,
    calc_dense_params,
    calc_dense_macs,
    calc_dense_forward_flops,
    calc_dense_backward_flops,
    calc_dense_traffic_values,
    calc_dense_saved_activations,
    calc_dense_inference_activations,
    calc_training_step_ops,
    calc_training_state,
    calc_sigmoid,
    calc_sigmoid_slope,
    calc_cross_entropy,
    calc_relu,
)
from .numerics import calc_float_range, calc_float_round, calc_int_max

__all__ += [
    "calc_dense_weights",
    "calc_dense_biases",
    "calc_dense_params",
    "calc_dense_macs",
    "calc_dense_forward_flops",
    "calc_dense_backward_flops",
    "calc_dense_traffic_values",
    "calc_dense_saved_activations",
    "calc_dense_inference_activations",
    "calc_training_step_ops",
    "calc_training_state",
    "calc_sigmoid",
    "calc_sigmoid_slope",
    "calc_cross_entropy",
    "calc_relu",
    "calc_float_range",
    "calc_float_round",
    "calc_int_max",
]

# --- ML operations at scale: canary stages, error budgets, own against rent (added 2026-10-20) ---
from .statistics import calc_sample_collection_time
from .reliability import calc_error_budget_spent
from .economics import calc_rent_crossover, calc_ownership_breakeven_utilization

__all__ += [
    "calc_sample_collection_time",
    "calc_error_budget_spent",
    "calc_rent_crossover",
    "calc_ownership_breakeven_utilization",
]

# --- Security and privacy: Laplace scale, composition, membership ceiling (added 2026-10-20) ---
from .privacy import (
    calc_laplace_scale,
    calc_basic_composition,
    calc_membership_tpr_ceiling,
    calc_level_bits,
)

__all__ += [
    "calc_laplace_scale",
    "calc_basic_composition",
    "calc_membership_tpr_ceiling",
    "calc_level_bits",
]

# --- Robustness: drift tests, conformal ranks, certified radii, bit flips (added 2026-10-20) ---
from .statistics import (
    calc_ks_coefficient,
    calc_standard_error,
    calc_gaussian_kl_terms,
    calc_gaussian_kl,
    calc_split_conformal_rank,
    calc_smoothing_radius,
    calc_linf_corner_l2,
)
from .numerics import calc_float32_bit_flip

__all__ += [
    "calc_ks_coefficient",
    "calc_standard_error",
    "calc_gaussian_kl_terms",
    "calc_gaussian_kl",
    "calc_split_conformal_rank",
    "calc_smoothing_radius",
    "calc_linf_corner_l2",
    "calc_float32_bit_flip",
]
# --- Injection exposure (added 2026-10-20) ---
from .agents import calc_injection_exposure, calc_injection_read_budget

__all__ += ["calc_injection_exposure", "calc_injection_read_budget"]
# --- The agent harness: admission, fair share, and the budget meter (added 2026-10-20) ---
from .agents import calc_bounded_liability, calc_drr_calls_per_round, calc_used_fraction

# --- Failure recovery (added 2026-10-20) ---
from .agents import (
    calc_repair_break_even,
    calc_repair_first_expected_cost,
    calc_expected_irreversible_loss,
    calc_cleanup_critical_path,
)

__all__ += [
    "calc_repair_break_even",
    "calc_repair_first_expected_cost",
    "calc_expected_irreversible_loss",
    "calc_cleanup_critical_path",
]

__all__ += [
    "calc_bounded_liability",
    "calc_drr_calls_per_round",
    "calc_used_fraction",
]

# --- Closed-loop evaluation statistics (added 2026-10-20) ---
from .robotics.assurance import (
    calc_clopper_pearson_lower_bound,
    calc_rule_of_three_bound,
    calc_rule_of_three_trials,
    calc_zero_failure_mission_count,
)
from .statistics import (
    calc_sprt_increments,
    calc_sprt_boundaries,
    calc_fisher_exact_two_sided,
    calc_cohens_h,
    calc_arcsine_two_arm_sample_size,
    calc_poisson_exact_interval,
)

__all__ += [
    "calc_clopper_pearson_lower_bound",
    "calc_rule_of_three_bound",
    "calc_rule_of_three_trials",
    "calc_zero_failure_mission_count",
    "calc_sprt_increments",
    "calc_sprt_boundaries",
    "calc_fisher_exact_two_sided",
    "calc_cohens_h",
    "calc_arcsine_two_arm_sample_size",
    "calc_poisson_exact_interval",
]

# --- Durable execution (added 2026-10-20) ---
from .reliability import (
    calc_group_commit_waits,
    calc_snapshot_write_volume,
    calc_checkpoint_stride,
    calc_stride_overhead,
)

__all__ += [
    "calc_group_commit_waits",
    "calc_snapshot_write_volume",
    "calc_checkpoint_stride",
    "calc_stride_overhead",
]

# --- Spatial memory: estimation, belief expiry, clearance queries (added 2026-10-20) ---
from .robotics import (
    calc_sphere_scan_cells,
    calc_octree_depth,
    calc_voxel_quantization_error,
    calc_scalar_kalman_update,
    calc_inertial_coast_errors,
    calc_belief_error_bound,
    calc_quadratic_invalidation_horizon,
)
from .statistics import calc_chi2_gate, calc_chi2_mean_band, calc_noncentral_gate_detection

__all__ += [
    "calc_sphere_scan_cells",
    "calc_octree_depth",
    "calc_voxel_quantization_error",
    "calc_scalar_kalman_update",
    "calc_inertial_coast_errors",
    "calc_belief_error_bound",
    "calc_quadratic_invalidation_horizon",
    "calc_chi2_gate",
    "calc_chi2_mean_band",
    "calc_noncentral_gate_detection",
]

# --- Data selection (added 2026-10-20) ---
from .retrieval import calc_lsh_candidate_probability, calc_lsh_threshold
from .scaling import calc_allocation_growth

__all__ += [
    "calc_lsh_candidate_probability",
    "calc_lsh_threshold",
    "calc_allocation_growth",
]

from .scaling import calc_loss_reduction_compute_multiple  # added 2026-10-20

__all__ += [
    "calc_loss_reduction_compute_multiple",
]

from .scaling import calc_power_law_factor  # added 2026-10-20

__all__ += [
    "calc_power_law_factor",
]

# --- Acceptance rates: intervals, paired comparison, pass^k (added 2026-10-20) ---
from .statistics import (
    calc_wald_interval,
    calc_wilson_interval,
    calc_proportion_half_width,
    calc_samples_for_half_width,
    calc_within_task_variance,
    calc_task_averaged_variance,
    calc_newcombe_paired_difference,
    calc_mcnemar_exact_two_sided,
    calc_paired_difference_variance,
)
from .agents import calc_pass_pow_k_unbiased

__all__ += [
    "calc_wald_interval",
    "calc_wilson_interval",
    "calc_proportion_half_width",
    "calc_samples_for_half_width",
    "calc_within_task_variance",
    "calc_task_averaged_variance",
    "calc_newcombe_paired_difference",
    "calc_mcnemar_exact_two_sided",
    "calc_paired_difference_variance",
    "calc_pass_pow_k_unbiased",
]

# --- Agent operations: score intervals, paired differences, drift, sequential samples (added 2026-10-20) ---
from .statistics import (
    calc_wilson_interval,
    calc_wilson_clean_runs,
    calc_newcombe_paired_difference,
    calc_mcnemar_exact_two_sided,
    calc_paired_difference_variance,
    calc_jensen_shannon_divergence,
    calc_sprt_expected_samples,
)

__all__ += [
    "calc_wilson_interval",
    "calc_wilson_clean_runs",
    "calc_newcombe_paired_difference",
    "calc_mcnemar_exact_two_sided",
    "calc_paired_difference_variance",
    "calc_jensen_shannon_divergence",
    "calc_sprt_expected_samples",
]
# Fitting a training step on one accelerator (added 2026-10-20)
from .memory import calc_checkpoint_memory_units
from .numerics import calc_float_spacing, calc_repeated_add
from .transformer import calc_online_softmax

__all__ += [
    "calc_checkpoint_memory_units",
    "calc_float_spacing",
    "calc_repeated_add",
    "calc_online_softmax",
]
from .numerics import calc_float_decimal_digits
from .transformer import calc_score_matrix_intensity

__all__ += ["calc_float_decimal_digits", "calc_score_matrix_intensity"]
# --- Model compression (added 2026-10-20) ---
from .memory import calc_sparse_bytes_per_nonzero, calc_sparse_breakeven_zero_share, calc_csr_bytes
from .performance import calc_pareto_frontier

__all__ += [
    "calc_sparse_bytes_per_nonzero",
    "calc_sparse_breakeven_zero_share",
    "calc_csr_bytes",
    "calc_pareto_frontier",
]

# --- Trajectory curation: split leakage, attempt-group contrast, ordered checks (added 2026-10-20) ---
from .agents import (
    calc_split_straddle,
    calc_mixed_outcome_fraction,
    calc_cost_per_rejection,
    calc_short_circuit_cost,
)
from .retrieval import calc_minhash_standard_error

__all__ += [
    "calc_split_straddle",
    "calc_mixed_outcome_fraction",
    "calc_cost_per_rejection",
    "calc_short_circuit_cost",
    "calc_minhash_standard_error",
]

# --- Fine-tuning on trajectories: weighting, packing, preference pairs (added 2026-10-20) ---
from .fine_tuning import (
    calc_token_average_shares,
    calc_example_average_share,
    calc_first_fit_decreasing,
    calc_preference_pair,
)
from .statistics import calc_proportion_gap_se

__all__ += [
    "calc_token_average_shares",
    "calc_example_average_share",
    "calc_first_fit_decreasing",
    "calc_preference_pair",
    "calc_proportion_gap_se",
]

# --- Quantization grids and block formats (added 2026-10-20) ---
from .numerics import (calc_float_values, calc_int_levels, calc_affine_scale, calc_symmetric_scale,
                       calc_zero_point, calc_quantize, calc_dequantize, calc_block_bits_per_weight,
                       calc_accumulator_headroom)

__all__ += [
    "calc_float_values",
    "calc_int_levels",
    "calc_affine_scale",
    "calc_symmetric_scale",
    "calc_zero_point",
    "calc_quantize",
    "calc_dequantize",
    "calc_block_bits_per_weight",
    "calc_accumulator_headroom",
]


# --- One-server queueing and batch formation (added 2026-10-20) ---
from .serving import (
    calc_mm1_time_in_system,
    calc_exponential_quantile_factor,
    calc_mm1_max_utilization,
    calc_batch_formation_wait,
)

__all__ += [
    "calc_mm1_time_in_system",
    "calc_exponential_quantile_factor",
    "calc_mm1_max_utilization",
    "calc_batch_formation_wait",
]

# --- Coordinating several agents: partitions, schedules, shared writes, correlated votes (added 2026-10-20) ---
from .coordination import (
    calc_partition_elapsed,
    calc_full_mesh_deliveries,
    calc_supervisor_deliveries,
    calc_earliest_schedule,
    calc_disjoint_writes,
    calc_disjoint_writes_approx,
    calc_binomial_tail,
    calc_beta_mixture_shape,
    calc_beta_binomial_pmf,
    calc_beta_tail,
)

__all__ += [
    "calc_partition_elapsed",
    "calc_full_mesh_deliveries",
    "calc_supervisor_deliveries",
    "calc_earliest_schedule",
    "calc_disjoint_writes",
    "calc_disjoint_writes_approx",
    "calc_binomial_tail",
    "calc_beta_mixture_shape",
    "calc_beta_binomial_pmf",
    "calc_beta_tail",
]
# --- Per-group detection: frames to a reliable detection, exact intervals (added 2026-10-20) ---
from .statistics import calc_consecutive_detection_frames, calc_clopper_pearson_interval

__all__ += ["calc_consecutive_detection_frames", "calc_clopper_pearson_interval"]

# --- ML operations: skew cost, detectable lift, decay rate (added 2026-10-20) ---
from .economics import calc_skew_cost
from .statistics import calc_two_proportion_detectable_lift, calc_exponential_decay_rate

__all__ += [
    "calc_skew_cost",
    "calc_two_proportion_detectable_lift",
    "calc_exponential_decay_rate",
]

# --- Benchmark run statistics (added 2026-10-20) ---
from .statistics import (
    calc_t_critical,
    calc_welch_interval,
    calc_mean_half_width,
    calc_samples_per_arm_for_difference,
    calc_mcnemar_max_discordant,
    calc_expected_calibration_error,
    calc_spearman_rank_correlation,
)

__all__ += [
    "calc_t_critical",
    "calc_welch_interval",
    "calc_mean_half_width",
    "calc_samples_per_arm_for_difference",
    "calc_mcnemar_max_discordant",
    "calc_expected_calibration_error",
    "calc_spearman_rank_correlation",
]

# --- Fairness: error-rate identity, bias feedback, gap sample sizes (added 2026-10-20) ---
from .fairness import (
    calc_fpr_from_ppv,
    calc_feedback_disparity,
    calc_cycles_to_gate,
    calc_gap_samples_per_group,
)

__all__ += [
    "calc_fpr_from_ppv",
    "calc_feedback_disparity",
    "calc_cycles_to_gate",
    "calc_gap_samples_per_group",
]
# --- Agent serving economics: queue price and speculative rounds (added 2026-10-20) ---
from .serving import calc_kingman_wait_ratio, calc_speculative_round_speedup

__all__ += ["calc_kingman_wait_ratio", "calc_speculative_round_speedup"]
