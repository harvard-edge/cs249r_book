"""Robotics and Physical AI physics models organized by engineering domain."""

from .stopping import (
    calc_safe_stopping_distance,
    calc_max_permitted_velocity,
    calc_c2_stop_suffix,
    _scurve_stop_si,
    _scurve_si_inputs,
    calc_scurve_stop_distance,
    calc_scurve_stop_time,
    calc_scurve_max_permitted_velocity,
    calc_scurve_rest_to_rest_time,
    calc_latch_contact_stopping_profile,
    calc_stopping_crossover_speed,
    calc_braking_time,
    calc_travel_between_speeds,
    calc_smoothstep_braking_blend,  # added 2026-10-20
    calc_blend_reserve_use,  # added 2026-10-20
)
from .actuation import (
    calc_kinetic_energy,
    calc_reflected_inertia,
    calc_seam_acceleration_torque_jump,
    calc_inverted_pendulum_fall_time,
    calc_voltage_droop,
    calc_contact_force,
    calc_contact_yield_deadline,
    calc_inductive_voltage_droop,
    calc_alpha_power_gate_delay_stretch,
    calc_reduced_mass,
    calc_transient_impact_force,
    calc_transient_contact_speed_limit,
    calc_tripwire_contact_force_accumulation,
    calc_quintic_blend_duration,
    calc_smoothstep_blend_window,  # added 2026-10-20
    calc_quasistatic_spill_acceleration,
)
from .control import (
    CANFD_DATA_FIELD_SIZES,
    calc_action_chunk_streaming_amortization,
    calc_chunk_renewal_schedule,
    calc_temporal_ensemble_weights,
    calc_watchdog_lease_bound,
    calc_phase_lag_total,
    calc_canfd_frame_bits,
    calc_canfd_bus_utilization,
    calc_ethercat_cycle_time,
    calc_response_time_bound,
    calc_cbf_qp_orthogonal_projection,
    calc_axi_fifo_contention,
)
from .perception import (
    calc_spatial_information_age_displacement,
    calc_3dgs_memory_footprint,
    calc_tsdf_voxel_grid_budget,
    calc_target_evidence_horizon,
    calc_tracked_evidence_horizon,
    calc_linear_invalidation_horizon,
    calc_free_space_expiry,
    calc_teleop_ingestion_budget,
    calc_sphere_scan_cells,
    calc_octree_depth,
    calc_voxel_quantization_error,
)
# State estimation and belief expiry (added 2026-10-20)
from .estimation import (
    calc_scalar_kalman_update,
    calc_inertial_coast_errors,
    calc_belief_error_bound,
    calc_quadratic_invalidation_horizon,
)
from .assurance import (
    calc_empirical_testing_exposure,
    calc_clopper_pearson_zero_failure_bound,
    calc_zero_failure_sample_size,
    calc_shielded_system_hazard_rate,
)

# Policy training (added 2026-10-20)
from .learning import (
    calc_discount_effective_horizon,
    calc_discounted_weight,
    calc_discount_for_retained_weight,
    calc_trial_campaign_wall_time,
    calc_iterative_decoder_latency,
    calc_max_decoder_evaluations,
    calc_series_stiffness,
    calc_complementary_series_stiffness,
    calc_impact_stiffness_for_force,
    calc_penalty_stiffness_from_penetration,
    calc_contact_timescale,
)

__all__ = [
    "CANFD_DATA_FIELD_SIZES",
    "_scurve_si_inputs",
    "_scurve_stop_si",
    "calc_safe_stopping_distance",
    "calc_max_permitted_velocity",
    "calc_c2_stop_suffix",
    "calc_scurve_stop_distance",
    "calc_scurve_stop_time",
    "calc_scurve_max_permitted_velocity",
    "calc_scurve_rest_to_rest_time",
    "calc_latch_contact_stopping_profile",
    "calc_stopping_crossover_speed",
    "calc_braking_time",
    "calc_travel_between_speeds",
    "calc_kinetic_energy",
    "calc_reflected_inertia",
    "calc_seam_acceleration_torque_jump",
    "calc_inverted_pendulum_fall_time",
    "calc_voltage_droop",
    "calc_contact_force",
    "calc_contact_yield_deadline",
    "calc_inductive_voltage_droop",
    "calc_alpha_power_gate_delay_stretch",
    "calc_reduced_mass",
    "calc_transient_impact_force",
    "calc_transient_contact_speed_limit",
    "calc_tripwire_contact_force_accumulation",
    "calc_quintic_blend_duration",
    "calc_quasistatic_spill_acceleration",
    "calc_action_chunk_streaming_amortization",
    "calc_chunk_renewal_schedule",
    "calc_temporal_ensemble_weights",
    "calc_watchdog_lease_bound",
    "calc_phase_lag_total",
    "calc_canfd_frame_bits",
    "calc_canfd_bus_utilization",
    "calc_ethercat_cycle_time",
    "calc_response_time_bound",
    "calc_cbf_qp_orthogonal_projection",
    "calc_axi_fifo_contention",
    "calc_spatial_information_age_displacement",
    "calc_3dgs_memory_footprint",
    "calc_tsdf_voxel_grid_budget",
    "calc_target_evidence_horizon",
    "calc_tracked_evidence_horizon",
    "calc_linear_invalidation_horizon",
    "calc_free_space_expiry",
    "calc_teleop_ingestion_budget",
    "calc_empirical_testing_exposure",
    "calc_clopper_pearson_zero_failure_bound",
    "calc_zero_failure_sample_size",
    "calc_shielded_system_hazard_rate",
    "calc_discount_effective_horizon",
    "calc_discounted_weight",
    "calc_discount_for_retained_weight",
    "calc_trial_campaign_wall_time",
    "calc_iterative_decoder_latency",
    "calc_max_decoder_evaluations",
    "calc_series_stiffness",
    "calc_complementary_series_stiffness",
    "calc_impact_stiffness_for_force",
    "calc_penalty_stiffness_from_penetration",
    "calc_contact_timescale",
]


# Sensing error scales (added 2026-10-20)
from .control import calc_delay_phase_margin_ceiling
from .sensing import (
    calc_stereo_depth_sigma,
    calc_stereo_range_for_sigma,
    calc_stereo_baseline_bias,
    calc_lever_arm_error,
    calc_scan_sector_angle,
    calc_contrast_visibility_range,
    calc_thermal_expansion,
)

__all__ += [
    "calc_delay_phase_margin_ceiling",
    "calc_stereo_depth_sigma",
    "calc_stereo_range_for_sigma",
    "calc_stereo_baseline_bias",
    "calc_lever_arm_error",
    "calc_scan_sector_angle",
    "calc_contrast_visibility_range",
    "calc_thermal_expansion",
]

# Spatial memory: estimation, belief expiry, and clearance queries (added 2026-10-20)
__all__ += [
    "calc_sphere_scan_cells",
    "calc_octree_depth",
    "calc_voxel_quantization_error",
    "calc_scalar_kalman_update",
    "calc_inertial_coast_errors",
    "calc_belief_error_bound",
    "calc_quadratic_invalidation_horizon",
]

# Manipulator kinematics: Jacobian conditioning and damped inverse kinematics (added 2026-10-20)
from .kinematics import calc_two_link_singular_values, calc_dls_joint_rate

__all__ += ["calc_two_link_singular_values", "calc_dls_joint_rate"]

# Bumpless transfer into braking (added 2026-10-20)
__all__ += ["calc_smoothstep_braking_blend", "calc_blend_reserve_use", "calc_smoothstep_blend_window"]
