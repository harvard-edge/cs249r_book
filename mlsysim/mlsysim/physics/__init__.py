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
    calc_ring_allreduce_time,
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

__all__ = [
    "calc_network_latency_ms",
    "dTime",
    "calc_training_time",
    "calc_training_time_days",
    "calc_amdahls_speedup",
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
    "calc_tree_allreduce_time",
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
]
