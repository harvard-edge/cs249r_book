# formulas.py
# Canonical equations for Machine Learning Systems
# centralizing the logic for TCO, Roofline, and Performance math.

import math
import pint
from .constants import ureg, Q_, SPEED_OF_LIGHT_FIBER_KM_S, MS, MB, GB, hour, second, byte

def _ensure_unit(val, unit):
    """Helper to attach unit if value is a raw number."""
    if isinstance(val, (int, float)):
        return val * unit
    return val

def calc_network_latency_ms(distance_km):
    """
    Calculates round-trip time in milliseconds based on speed of light in fiber.

    Source: Standard networking physics (c/1.5 refractive index).

    Args:
        distance_km: One-way distance in kilometers.

    Returns:
        float: Round-trip latency in milliseconds.
    """
    d = _ensure_unit(distance_km, ureg.kilometer)
    round_trip_s = (d * 2) / SPEED_OF_LIGHT_FIBER_KM_S
    return round_trip_s.m_as(ureg.millisecond)

def dTime(total_ops, num_devices, peak_flops_per_device, efficiency_eta):
    """
    Core training time calculation (first-principles).

    Source: Standard Performance Modeling for Distributed Systems.

    Args:
        total_ops: Total floating point operations for training.
        num_devices: Number of accelerators.
        peak_flops_per_device: Peak FLOP/s per accelerator.
        efficiency_eta: Model FLOPs utilization (0.0 to 1.0).

    Returns:
        Quantity[second]: Estimated training duration.
    """
    ops = _ensure_unit(total_ops, ureg.flop)
    flops = _ensure_unit(peak_flops_per_device, ureg.flop / ureg.second)
    effective_throughput = num_devices * flops * efficiency_eta
    if getattr(effective_throughput, 'magnitude', effective_throughput) == 0:
        return float('inf') * ureg.second
    duration = ops / effective_throughput
    return duration.to(ureg.second)


def calc_training_time_days(total_ops, num_devices, peak_flops_per_device, efficiency_eta):
    """Calculates training duration in days."""
    duration = dTime(total_ops, num_devices, peak_flops_per_device, efficiency_eta)
    return duration.m_as(ureg.day)

def calc_amdahls_speedup(p, s):
    """
    Calculates overall system speedup (Amdahl's Law).
    
    Source: Amdahl (1967), "Validity of the Single Processor Approach to 
    Achieving Large Scale Computing Capabilities."
    
    Args:
        p: fraction of work that can be improved (0.0 to 1.0)
        s: speedup of that fraction
        
    Returns:
        float: The overall system speedup.
    """
    if s == 0:
        return 0.0
    denom = (1 - p) + (p / s)
    if denom == 0:
        return float('inf')
    overall = 1 / denom
    return overall

def calc_monthly_egress_cost(bytes_per_sec, cost_per_gb):
    """
    Calculates monthly cloud egress cost based on standard cloud egress rates.
    
    Args:
        bytes_per_sec: Egress bandwidth in bytes per second.
        cost_per_gb: Cost per gigabyte.
        
    Returns:
        float: Monthly egress cost in dollars.
    """
    b_s = _ensure_unit(bytes_per_sec, ureg.byte / ureg.second)
    monthly_bytes = b_s * (30 * ureg.day)
    cost = monthly_bytes * cost_per_gb
    return cost.m_as(ureg.dollar)

def calc_fleet_tco(unit_cost, power_w, quantity, years, kwh_price):
    """
    Calculates Total Cost of Ownership (TCO).
    
    Source: Barroso et al. (2018), "The Datacenter as a Computer."
    
    Args:
        unit_cost: Cost per unit.
        power_w: Power consumption per unit in Watts.
        quantity: Number of units.
        years: Operational lifespan in years.
        kwh_price: Electricity price per kWh.
        
    Returns:
        float: Total TCO in dollars.
    """
    u_cost = _ensure_unit(unit_cost, ureg.dollar)
    p_w = _ensure_unit(power_w, ureg.watt)
    price = _ensure_unit(kwh_price, ureg.dollar / ureg.kilowatt_hour)
    time = _ensure_unit(years, ureg.year)
    fleet_capex = u_cost * quantity
    total_energy = p_w * quantity * time
    power_opex = total_energy * price
    total = fleet_capex + power_opex
    return total.m_as(ureg.dollar)

def calc_bottleneck(ops, model_bytes, device_flops, device_bw):
    """
    Roofline bottleneck analysis.
    
    Source: Williams et al. (2009), "Roofline Model."
    
    Args:
        ops: Total operations.
        model_bytes: Total memory bytes accessed.
        device_flops: Peak operations per second.
        device_bw: Peak memory bandwidth in bytes per second.
        
    Returns:
        dict: Bottleneck analysis profile.
    """
    o = _ensure_unit(ops, ureg.flop)
    mb = _ensure_unit(model_bytes, ureg.byte)
    df = _ensure_unit(device_flops, ureg.flop / ureg.second)
    dbw = _ensure_unit(device_bw, ureg.byte / ureg.second)

    compute_time = o / df
    memory_time = mb / dbw
    t_comp_ms = compute_time.m_as(ureg.millisecond)
    t_mem_ms = memory_time.m_as(ureg.millisecond)
    
    if t_comp_ms == 0:
        return {
            "compute_ms": 0.0,
            "memory_ms": t_mem_ms,
            "bottleneck": "Memory",
            "ratio": float('inf'),
            "intensity": 0.0
        }

    if t_mem_ms == 0:
        return {
            "compute_ms": t_comp_ms,
            "memory_ms": 0.0,
            "bottleneck": "Compute",
            "ratio": float('inf'),
            "intensity": float('inf') if mb.magnitude == 0 else (o / mb).magnitude
        }

    is_memory_bound = t_mem_ms > t_comp_ms
    ratio = t_mem_ms / t_comp_ms if is_memory_bound else t_comp_ms / t_mem_ms
    intensity = o / mb
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
    if isinstance(params, ureg.Quantity):
        try:
            param_count = params.to(ureg.count).magnitude
        except pint.DimensionalityError:
            raise pint.DimensionalityError(
                params.units, ureg.count,
                extra_msg=f" in model_memory() — params must be in param/count units, got {params.units}"
            )
    else:
        param_count = params

    if isinstance(bytes_per_param, ureg.Quantity):
        try:
            bpp = bytes_per_param.to(ureg.byte).magnitude
        except pint.DimensionalityError:
            raise pint.DimensionalityError(
                bytes_per_param.units, ureg.byte,
                extra_msg=f" in model_memory() — bytes_per_param must be byte units, got {bytes_per_param.units}"
            )
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
        Quantity[second]: Estimated AllReduce time
    """
    msg = _ensure_unit(message_bytes, ureg.byte)
    bw  = _ensure_unit(bandwidth_bytes_s, ureg.byte / ureg.second)
    lat = _ensure_unit(latency_s, ureg.second)
    n = n_gpus
    bw_term = 2 * (n - 1) / n * msg / bw
    lat_term = 2 * (n - 1) * lat
    return (bw_term + lat_term).to(ureg.second)


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
        Quantity[second]: Estimated AllReduce time
    """
    msg = _ensure_unit(message_bytes, ureg.byte)
    bw  = _ensure_unit(bandwidth_bytes_s, ureg.byte / ureg.second)
    lat = _ensure_unit(latency_s, ureg.second)
    log_n = math.log2(n_gpus)
    bw_term = 2 * log_n * msg / bw
    lat_term = 2 * log_n * lat
    return (bw_term + lat_term).to(ureg.second)


def calc_all_to_all_time(message_bytes, n_gpus, bandwidth_bytes_s, latency_s):
    """
    All-to-All communication time estimate (typical for MoE token routing).

    T = (N-1)/N × M/β + (N-1) × α

    Args:
        message_bytes: Total message size in bytes (M) per node
        n_gpus: Number of GPUs (N)
        bandwidth_bytes_s: Per-link bandwidth in bytes/second (β)
        latency_s: Per-message startup latency in seconds (α)

    Returns:
        Quantity[second]: Estimated All-to-All time
    """
    msg = _ensure_unit(message_bytes, ureg.byte)
    bw  = _ensure_unit(bandwidth_bytes_s, ureg.byte / ureg.second)
    lat = _ensure_unit(latency_s, ureg.second)
    n = n_gpus
    bw_term = (n - 1) / n * msg / bw
    lat_term = (n - 1) * lat
    return (bw_term + lat_term).to(ureg.second)


def calc_transformer_training_flops(n_params, n_tokens):
    """
    Estimate total training FLOPs for a Transformer model (6PD rule).
    
    T ≈ 6 × P × D
    
    Source: Kaplan et al. (2020), "Scaling Laws for Neural Language Models"
    
    Args:
        n_params: Number of parameters (P)
        n_tokens: Number of training tokens (D)
        
    Returns:
        Quantity[flop]: Total training FLOPs
    """
    p = _ensure_unit(n_params, ureg.param).to(ureg.count).magnitude
    d = _ensure_unit(n_tokens, ureg.count).magnitude
    return (6 * p * d) * ureg.flop


def calc_activation_memory(n_layers, seq_len, batch_size, hidden_dim, n_heads=None, 
                           precision_bytes=2, strategy="selective"):
    """
    Estimate activation memory for a Transformer layer.
    
    Source: Korthikanti et al. (2023), "Reducing Activation Memory in Transformer Training"
    
    Args:
        n_layers: Number of layers (L)
        seq_len: Sequence length (S)
        batch_size: Batch size (B)
        hidden_dim: Hidden dimension (H)
        n_heads: Number of attention heads (A)
        precision_bytes: Bytes per element (default 2 for FP16)
        strategy: Recompute strategy ('none', 'selective', 'full')
        
    Returns:
        Quantity[byte]: Total activation memory
    """
    s, b, h = seq_len, batch_size, hidden_dim
    # Basic activation per layer: 34 * s * b * h (without recompute)
    # With selective recompute, it's significantly lower.
    if strategy == "full":
        # Only store inputs to the block (1 element per activation)
        elements_per_layer = 1 * s * b * h
    elif strategy == "selective":
        # Store some intermediate activations to avoid full recompute
        # Reference estimate: ~5 elements per activation
        elements_per_layer = 5 * s * b * h
    else:
        # No recompute: store everything (17 elements per activation)
        elements_per_layer = 17 * s * b * h
        
    return (n_layers * elements_per_layer * precision_bytes) * ureg.byte


def calc_hierarchical_allreduce_time(message_bytes, n_nodes, gpus_per_node, 
                                     intra_node_bw, inter_node_bw, 
                                     intra_node_lat=Q_("500 ns"), inter_node_lat=Q_("5 us")):
    """
    Hierarchical AllReduce time estimate (Intra-node NVLink + Inter-node IB).
    
    T = T_intra + T_inter + T_intra
    
    Source: Standard implementation in NCCL / Horovod.
    
    Args:
        message_bytes: Message size (M)
        n_nodes: Number of nodes
        gpus_per_node: GPUs per node (usually 8)
        intra_node_bw: Intra-node bandwidth (NVLink)
        inter_node_bw: Inter-node bandwidth (InfiniBand)
        intra_node_lat: Intra-node latency
        inter_node_lat: Inter-node latency
        
    Returns:
        Quantity[second]: Estimated communication time
    """
    # 1. Intra-node Reduce (to one GPU per node)
    # A Reduce takes exactly half the time of a full Ring AllReduce
    t_reduce = calc_ring_allreduce_time(message_bytes, gpus_per_node, intra_node_bw, intra_node_lat) / 2
    
    # 2. Inter-node AllReduce (between lead GPUs of each node)
    t_allreduce_inter = calc_ring_allreduce_time(message_bytes, n_nodes, inter_node_bw, inter_node_lat)
    
    # 3. Intra-node Broadcast (back to all GPUs)
    # A Broadcast takes exactly half the time of a full Ring AllReduce
    t_broadcast = t_reduce # Symmetry assumption
    
    return t_reduce + t_allreduce_inter + t_broadcast


def calc_young_daly_interval(checkpoint_cost_s, mtbf_s):
    """
    Optimal checkpoint interval (Young-Daly model).

    τ_opt = √(2 × δ × M)

    Args:
        checkpoint_cost_s: Time to write one checkpoint in seconds (δ)
        mtbf_s: Mean Time Between Failures in seconds (M)

    Returns:
        Quantity[second]: Optimal checkpoint interval
    """
    delta = _ensure_unit(checkpoint_cost_s, ureg.second)
    mtbf  = _ensure_unit(mtbf_s, ureg.second)
    seconds = math.sqrt(2 * delta.m_as(ureg.second) * mtbf.m_as(ureg.second))
    return seconds * ureg.second


def calc_mtbf_cluster(component_mtbf_hours, n_components):
    """
    Cluster MTBF from identical independent components.

    MTBF_cluster = MTBF_component / N

    Args:
        component_mtbf_hours: Single component MTBF in hours (or Quantity[hour])
        n_components: Number of identical components

    Returns:
        Quantity[hour]: Cluster MTBF
    """
    if n_components <= 0:
        return float('inf') * ureg.hour
    mtbf = _ensure_unit(component_mtbf_hours, ureg.hour)
    return (mtbf / n_components).to(ureg.hour)


def calc_mtbf_node(gpu_mtbf_h, n_gpus, nic_mtbf_h, n_nics,
                   psu_mtbf_h, n_psus, other_mtbf_h=None, n_other=0):
    """
    Compound node MTBF from heterogeneous components.

    1/MTBF_node = n_gpu/MTBF_gpu + n_nic/MTBF_nic + n_psu/MTBF_psu + ...

    Args:
        gpu_mtbf_h: GPU MTTF in hours (or Quantity[hour])
        n_gpus: GPUs per node
        nic_mtbf_h: NIC MTTF in hours (or Quantity[hour])
        n_nics: NICs per node
        psu_mtbf_h: PSU MTTF in hours (or Quantity[hour])
        n_psus: PSUs per node
        other_mtbf_h: Optional other component MTTF
        n_other: Count of other components

    Returns:
        Quantity[hour]: Node MTBF
    """
    gpu = _ensure_unit(gpu_mtbf_h, ureg.hour)
    nic = _ensure_unit(nic_mtbf_h, ureg.hour)
    psu = _ensure_unit(psu_mtbf_h, ureg.hour)
    rate = 0.0 / ureg.hour
    if n_gpus > 0: rate += n_gpus / gpu
    if n_nics > 0: rate += n_nics / nic
    if n_psus > 0: rate += n_psus / psu
    
    if other_mtbf_h is not None and n_other > 0:
        rate += n_other / _ensure_unit(other_mtbf_h, ureg.hour)
    
    if getattr(rate, 'magnitude', rate) == 0:
        return float('inf') * ureg.hour
    return (1.0 / rate).to(ureg.hour)


def calc_pipeline_bubble(n_stages, n_microbatches, v_stages=1):
    """
    Pipeline bubble fraction (GPipe / 1F1B / Interleaved 1F1B).

    bubble = (P - 1) / (V * M + P - 1)

    Args:
        n_stages: Number of pipeline stages (P)
        n_microbatches: Number of microbatches (M)
        v_stages: Number of virtual stages per GPU (V, default 1)

    Returns:
        Bubble fraction (0.0 to 1.0)
    """
    return (n_stages - 1) / (v_stages * n_microbatches + n_stages - 1)


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
        Quantity[byte]: Checkpoint size
    """
    bpp = _ensure_unit(bytes_per_param, ureg.byte)
    return (n_params * bpp).to(ureg.byte)


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
        Quantity[byte]: KV cache size
    """
    bpe = _ensure_unit(bytes_per_elem, ureg.byte)
    return (2 * n_layers * n_heads * head_dim * seq_len * batch_size * bpe).to(ureg.byte)


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


def calc_failure_probability(mtbf, job_duration):
    """
    Probability of at least one failure during a job.

    P(≥1 failure) = 1 - e^(-T / MTBF)

    If both are Quantities: units auto-reconciled (pass hours or seconds freely).
    If both are raw numbers: caller must use consistent units.
    Mixed types (one Quantity, one raw) raise TypeError — ambiguous unit intent.

    Args:
        mtbf: Mean Time Between Failures (Quantity or raw number)
        job_duration: Job duration (Quantity or raw number; same units if raw)

    Returns:
        Probability of at least one failure (0.0 to 1.0)
    """
    both_qty = isinstance(mtbf, ureg.Quantity) and isinstance(job_duration, ureg.Quantity)
    either_qty = isinstance(mtbf, ureg.Quantity) or isinstance(job_duration, ureg.Quantity)
    if either_qty and not both_qty:
        raise TypeError(
            "calc_failure_probability: both arguments must be Quantities or both raw numbers. "
            "Mixed types are ambiguous — attach units to the raw number first."
        )
    mtbf_mag = getattr(mtbf, 'magnitude', mtbf)
    if mtbf_mag == 0:
        return 1.0

    if both_qty:
        ratio = job_duration.to(ureg.second).magnitude / mtbf.to(ureg.second).magnitude
    else:
        ratio = job_duration / mtbf   # raw: caller responsible for consistent units
    return 1.0 - math.exp(-ratio)


def calc_effective_flops(peak_flops, mfu, scaling_eff, goodput_ratio):
    """
    Effective FLOPS delivered by a fleet after all overheads.

    Effective = Peak × MFU × η_scaling × Goodput/Rawput

    Args:
        peak_flops: Aggregate peak FLOPS of the cluster (Quantity or raw)
        mfu: Model FLOPS Utilization (0.0 to 1.0)
        scaling_eff: Scaling efficiency η (0.0 to 1.0)
        goodput_ratio: Goodput / Rawput ratio (0.0 to 1.0)

    Returns:
        Effective FLOPS in same units as peak_flops
    """
    pf = _ensure_unit(peak_flops, ureg.flop / ureg.second)
    return (pf * mfu * scaling_eff * goodput_ratio).to(ureg.flop / ureg.second)
