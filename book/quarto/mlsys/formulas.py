# formulas.py
# Canonical equations for Machine Learning Systems
# centralizing the logic for TCO, Physics, and Performance math.

import math
from .constants import ureg, SPEED_OF_LIGHT_FIBER_KM_S, MS, MB, GB, hour

def _ensure_unit(val, unit):
    """Helper to attach unit if value is a raw number."""
    if isinstance(val, (int, float)):
        return val * unit
    return val

def calc_network_latency_ms(distance_km):
    """Calculates round-trip time in milliseconds."""
    d = _ensure_unit(distance_km, ureg.kilometer)
    round_trip_s = (d * 2) / SPEED_OF_LIGHT_FIBER_KM_S
    return round_trip_s.to(ureg.millisecond).magnitude

def dTime(total_ops, num_devices, peak_flops_per_device, efficiency_eta):
    """
    Core training time calculation (physics-first).
    Returns a Pint Quantity in seconds.
    """
    # ops / (n * p * eta)
    effective_throughput = num_devices * peak_flops_per_device * efficiency_eta
    duration = total_ops / effective_throughput
    return duration.to(ureg.second)


def calc_training_time_days(total_ops, num_devices, peak_flops_per_device, efficiency_eta):
    """Calculates training duration in days."""
    duration = dTime(total_ops, num_devices, peak_flops_per_device, efficiency_eta)
    return duration.to(ureg.day).magnitude

def calc_amdahls_speedup(p, s):
    """
    Calculates overall system speedup given:
    p: fraction of work that can be improved (0.0 to 1.0)
    s: speedup of that fraction
    """
    overall = 1 / ((1 - p) + (p / s))
    return overall

def calc_monthly_egress_cost(bytes_per_sec, cost_per_gb):
    """Calculates monthly cloud egress cost."""
    b_s = _ensure_unit(bytes_per_sec, ureg.byte / ureg.second)
    monthly_bytes = b_s * (30 * ureg.day)
    cost = monthly_bytes * cost_per_gb
    return cost.to(ureg.dollar).magnitude

def calc_fleet_tco(unit_cost, power_w, quantity, years, kwh_price):
    """Calculates Total Cost of Ownership (TCO)."""
    u_cost = _ensure_unit(unit_cost, ureg.dollar)
    p_w = _ensure_unit(power_w, ureg.watt)
    price = _ensure_unit(kwh_price, ureg.dollar / ureg.kilowatt_hour)
    time = _ensure_unit(years, ureg.year)
    fleet_capex = u_cost * quantity
    total_energy = p_w * quantity * time
    power_opex = total_energy * price
    total = fleet_capex + power_opex
    return total.to(ureg.dollar).magnitude

def calc_bottleneck(ops, model_bytes, device_flops, device_bw):
    """Roofline bottleneck analysis."""
    compute_time = ops / device_flops
    memory_time = model_bytes / device_bw
    t_comp_ms = compute_time.to(ureg.millisecond).magnitude
    t_mem_ms = memory_time.to(ureg.millisecond).magnitude
    is_memory_bound = t_mem_ms > t_comp_ms
    ratio = t_mem_ms / t_comp_ms if is_memory_bound else t_comp_ms / t_mem_ms
    intensity = ops / model_bytes
    return {
        "compute_ms": t_comp_ms,
        "memory_ms": t_mem_ms,
        "bottleneck": "Memory" if is_memory_bound else "Compute",
        "ratio": ratio,
        "intensity": intensity.magnitude
    }

def model_memory(params, bytes_per_param, unit=MB):
    """
    Calculate model memory footprint.

    Args:
        params: Number of parameters (pint Quantity with param unit, or raw number)
        bytes_per_param: Bytes per parameter (pint Quantity with byte unit, or raw number)
        unit: Target unit for output (default: MB)

    Returns:
        Memory footprint as float in the specified unit

    Example:
        >>> model_memory(RESNET50_PARAMS, BYTES_FP32)  # Returns ~102 (MB)
        >>> model_memory(GPT3_PARAMS, BYTES_FP16, GB)  # Returns ~350 (GB)
    """
    # Extract magnitudes to avoid param*byte dimension issues
    if hasattr(params, 'magnitude'):
        param_count = params.magnitude
    else:
        param_count = params

    if hasattr(bytes_per_param, 'magnitude'):
        bpp = bytes_per_param.magnitude
    else:
        bpp = bytes_per_param

    total_bytes = param_count * bpp * ureg.byte
    return total_bytes.to(unit).magnitude


# =============================================================================
# Fleet-Scale Formulas (Volume II)
# =============================================================================
# Canonical equations for distributed ML systems: communication costs,
# reliability, pipeline parallelism, and capacity planning.

def calc_ring_allreduce_time(message_bytes, n_gpus, bandwidth_bytes_s, latency_s):
    """
    Ring AllReduce time estimate (bandwidth-optimal algorithm).

    T = 2(N-1)/N × M/β + 2(N-1) × α

    Args:
        message_bytes: Total message size in bytes (M)
        n_gpus: Number of GPUs in the ring (N)
        bandwidth_bytes_s: Per-link bandwidth in bytes/second (β)
        latency_s: Per-message startup latency in seconds (α)

    Returns:
        Estimated AllReduce time in seconds
    """
    n = n_gpus
    bw_term = 2 * (n - 1) / n * message_bytes / bandwidth_bytes_s
    lat_term = 2 * (n - 1) * latency_s
    return bw_term + lat_term


def calc_tree_allreduce_time(message_bytes, n_gpus, bandwidth_bytes_s, latency_s):
    """
    Tree AllReduce time estimate (latency-optimal algorithm).

    T = 2 log₂(N) × M/β + 2 log₂(N) × α

    Args:
        message_bytes: Total message size in bytes (M)
        n_gpus: Number of GPUs (N, must be power of 2 for exact result)
        bandwidth_bytes_s: Per-link bandwidth in bytes/second (β)
        latency_s: Per-message startup latency in seconds (α)

    Returns:
        Estimated AllReduce time in seconds
    """
    log_n = math.log2(n_gpus)
    bw_term = 2 * log_n * message_bytes / bandwidth_bytes_s
    lat_term = 2 * log_n * latency_s
    return bw_term + lat_term


def calc_young_daly_interval(checkpoint_cost_s, mtbf_s):
    """
    Optimal checkpoint interval (Young-Daly model).

    τ_opt = √(2 × δ × M)

    Args:
        checkpoint_cost_s: Time to write one checkpoint in seconds (δ)
        mtbf_s: Mean Time Between Failures in seconds (M)

    Returns:
        Optimal checkpoint interval in seconds
    """
    return math.sqrt(2 * checkpoint_cost_s * mtbf_s)


def calc_mtbf_cluster(component_mtbf_hours, n_components):
    """
    Cluster MTBF from identical independent components.

    MTBF_cluster = MTBF_component / N

    Args:
        component_mtbf_hours: Single component MTBF in hours
        n_components: Number of identical components

    Returns:
        Cluster MTBF in hours
    """
    return component_mtbf_hours / n_components


def calc_mtbf_node(gpu_mtbf_h, n_gpus, nic_mtbf_h, n_nics,
                   psu_mtbf_h, n_psus, other_mtbf_h=None, n_other=0):
    """
    Compound node MTBF from heterogeneous components.

    1/MTBF_node = n_gpu/MTBF_gpu + n_nic/MTBF_nic + n_psu/MTBF_psu + ...

    Args:
        gpu_mtbf_h: GPU MTTF in hours
        n_gpus: GPUs per node
        nic_mtbf_h: NIC MTTF in hours
        n_nics: NICs per node
        psu_mtbf_h: PSU MTTF in hours
        n_psus: PSUs per node
        other_mtbf_h: Optional other component MTTF
        n_other: Count of other components

    Returns:
        Node MTBF in hours
    """
    failure_rate = (n_gpus / gpu_mtbf_h +
                    n_nics / nic_mtbf_h +
                    n_psus / psu_mtbf_h)
    if other_mtbf_h and n_other > 0:
        failure_rate += n_other / other_mtbf_h
    return 1.0 / failure_rate


def calc_pipeline_bubble(n_stages, n_microbatches):
    """
    Pipeline bubble fraction (GPipe / 1F1B).

    bubble = (P - 1) / (P - 1 + M)

    Args:
        n_stages: Number of pipeline stages (P)
        n_microbatches: Number of microbatches (M)

    Returns:
        Bubble fraction (0.0 to 1.0)
    """
    return (n_stages - 1) / (n_stages - 1 + n_microbatches)


def calc_checkpoint_size(n_params, bytes_per_param=16):
    """
    Checkpoint size for mixed-precision Adam training.

    Size = N × bytes_per_param

    Default 16 bytes/param: 2 (BF16 weights) + 2 (gradients) +
    12 (FP32 master weights + momentum + variance).

    Args:
        n_params: Number of model parameters
        bytes_per_param: Bytes per parameter (default 16 for mixed-precision Adam)

    Returns:
        Checkpoint size in bytes
    """
    return n_params * bytes_per_param


def calc_kv_cache_size(n_layers, n_heads, head_dim, seq_len, batch_size,
                       bytes_per_elem=2):
    """
    KV cache memory for autoregressive inference.

    Size = 2 × L × H × D × S × B × bytes

    The factor of 2 accounts for both K and V tensors.

    Args:
        n_layers: Number of transformer layers (L)
        n_heads: Number of attention heads (H)
        head_dim: Dimension per head (D)
        seq_len: Sequence length / context window (S)
        batch_size: Batch size (B)
        bytes_per_elem: Bytes per element (default 2 for FP16/BF16)

    Returns:
        KV cache size in bytes
    """
    return 2 * n_layers * n_heads * head_dim * seq_len * batch_size * bytes_per_elem


def calc_availability_stacked(single_availability, n_replicas):
    """
    System availability with k independent replicas.

    A_system = 1 - (1 - A)^k

    Args:
        single_availability: Per-replica availability (0.0 to 1.0)
        n_replicas: Number of independent replicas (k)

    Returns:
        System availability (0.0 to 1.0)
    """
    return 1.0 - (1.0 - single_availability) ** n_replicas


def calc_failure_probability(mtbf_s, job_duration_s):
    """
    Probability of at least one failure during a job.

    P(≥1 failure) = 1 - e^(-T / MTBF)

    Args:
        mtbf_s: Mean Time Between Failures (seconds or hours — be consistent)
        job_duration_s: Job duration (same units as mtbf_s)

    Returns:
        Probability of at least one failure (0.0 to 1.0)
    """
    return 1.0 - math.exp(-job_duration_s / mtbf_s)


def calc_effective_flops(peak_flops, mfu, scaling_eff, goodput_ratio):
    """
    Effective FLOPS delivered by a fleet after all overheads.

    Effective = Peak × MFU × η_scaling × Goodput/Rawput

    Args:
        peak_flops: Aggregate peak FLOPS of the cluster
        mfu: Model FLOPS Utilization (0.0 to 1.0)
        scaling_eff: Scaling efficiency η (0.0 to 1.0)
        goodput_ratio: Goodput / Rawput ratio (0.0 to 1.0)

    Returns:
        Effective FLOPS (same units as peak_flops)
    """
    return peak_flops * mfu * scaling_eff * goodput_ratio

