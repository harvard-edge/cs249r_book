"""
Canonical physics and accounting formulas for ML systems.

Domain modules:
  networking, performance, economics, memory, communication,
  reliability, transformer, serving, statistics
"""

from ._units import _ensure_unit
from .constants import SPEED_OF_LIGHT_FIBER_KM_S
from .networking import calc_network_latency_ms
from .performance import (
    dTime,
    calc_training_time,
    calc_training_time_days,
    calc_amdahls_speedup,
    calc_strong_scaling_speedup,
    calc_bottleneck,
    calc_pipeline_bubble,
    calc_effective_flops,
)
from .economics import calc_monthly_egress_cost, calc_fleet_tco
from .memory import (
    model_memory,
    calc_activation_memory,
    calc_checkpoint_size,
    calc_kv_cache_size,
    calc_paged_kv_cache_size,
)
from .communication import (
    calc_alpha_beta_crossover,
    calc_ring_collective_data_factor,
    calc_bisection_bandwidth,
    calc_ring_allreduce_data_factor,
    ring_allreduce_data_factor_latex,
    calc_ring_allreduce_latency_steps,
    calc_ring_allreduce_latency_time,
    calc_hop_latency,
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
from .transformer import calc_transformer_training_flops, calc_transformer_decode_flops
from .serving import calc_queue_latency_mmc
from .statistics import (
    calc_population_stability_index,
    calc_two_proportion_sample_size,
    calc_constraint_propagation_factor,
)
from .agents import (
    calc_trajectory_step_time,
    calc_trajectory_reliability,
    calc_radix_cache_effective_latency,
    calc_test_time_compute_cost,
    calc_multi_agent_coordination_overhead,
)
from .robotics import (
    calc_sensor_to_actuator_latency,
    calc_safe_stopping_distance,
    calc_actuator_thermal_power,
    calc_action_chunk_cadence,
    calc_reflected_inertia,
)

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
    "calc_network_latency_ms",
    "dTime",
    "calc_training_time",
    "calc_training_time_days",
    "calc_amdahls_speedup",
    "calc_strong_scaling_speedup",
    "calc_bottleneck",
    "calc_pipeline_bubble",
    "calc_effective_flops",
    "calc_monthly_egress_cost",
    "calc_fleet_tco",
    "model_memory",
    "calc_activation_memory",
    "calc_checkpoint_size",
    "calc_kv_cache_size",
    "calc_paged_kv_cache_size",
    "calc_ring_allreduce_time",
    "calc_point_to_point_time",
    "calc_ring_allreduce_data_factor",
    "ring_allreduce_data_factor_latex",
    "calc_ring_collective_data_factor",
    "calc_ring_allreduce_latency_steps",
    "calc_ring_allreduce_latency_time",
    "calc_alpha_beta_crossover",
    "calc_oversubscription_effect",
    "calc_bisection_bandwidth",
    "calc_hop_latency",
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
    "calc_queue_latency_mmc",
    "calc_population_stability_index",
    "calc_two_proportion_sample_size",
    "calc_constraint_propagation_factor",
    "calc_trajectory_step_time",
    "calc_trajectory_reliability",
    "calc_radix_cache_effective_latency",
    "calc_test_time_compute_cost",
    "calc_multi_agent_coordination_overhead",
    "calc_sensor_to_actuator_latency",
    "calc_safe_stopping_distance",
    "calc_actuator_thermal_power",
    "calc_action_chunk_cadence",
    "calc_reflected_inertia",
    "transfer_time",
    "compute_time",
    "energy_from_power",
    "carbon_from_energy",
    "memory_from_params",
    "token_throughput",
]
