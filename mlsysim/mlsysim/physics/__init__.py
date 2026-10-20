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
    calc_vla_inference_latency,
)
from .serving import calc_queue_latency_mmc, calc_mg1_wait_ratio
from .statistics import (
    calc_population_stability_index,
    calc_two_proportion_sample_size,
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
)

__all__ += [
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
)

__all__ += [
    "calc_receptive_field",
    "calc_layers_for_receptive_field",
    "calc_separable_cost_fraction",
    "calc_gradient_product",
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
