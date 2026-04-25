# formulas.py
# Canonical equations for Machine Learning Systems
# centralizing the logic for TCO, Roofline, and Performance math.

import math
import pint
from .constants import ureg, Q_, SPEED_OF_LIGHT_FIBER_KM_S, MB
from ._validation import validate_positive, validate_at_least, validate_range

def _ensure_unit(val, expected_unit, param_name="Value"):
    """
    Helper to attach a unit if a value is a raw number, and verify 
    dimensional correctness if it is already a Pint Quantity.
    
    This function acts as a guardrail for students using the framework,
    ensuring they do not accidentally mix up units (e.g., passing Bytes 
    when Bandwidth is expected).
    """
    if isinstance(val, (int, float)):
        return val * expected_unit
    elif isinstance(val, ureg.Quantity):
        if not val.check(expected_unit):
            raise pint.DimensionalityError(
                val.units, expected_unit,
                extra_msg=f"\n[Pedagogical Error] '{param_name}' requires units of {expected_unit.dimensionality}. You provided '{val.units}'."
            )
        return val
    else:
        raise TypeError(f"Expected a number or Pint Quantity for {param_name}, got {type(val).__name__}")

def calc_network_latency_ms(distance_km):
    """
    Calculates round-trip time in milliseconds based on speed of light in fiber.
    
    Source: Standard networking physics (c/1.5 refractive index).
    """
    d = _ensure_unit(distance_km, ureg.kilometer, "distance_km")
    round_trip_s = (d * 2) / SPEED_OF_LIGHT_FIBER_KM_S
    return round_trip_s.m_as(ureg.millisecond)

def dTime(total_ops, num_devices, peak_flops_per_device, efficiency_eta):
    """
    Core training time calculation (first-principles).

    Source: Standard Performance Modeling for Distributed Systems.
    Returns a Pint Quantity in seconds.
    """
    validate_at_least(num_devices, 1, "num_devices")
    validate_positive(efficiency_eta, "efficiency_eta")
    effective_throughput = num_devices * peak_flops_per_device * efficiency_eta
    duration = total_ops / effective_throughput
    return duration.to(ureg.second)


# Preferred name (consistent with calc_* convention); dTime kept as alias
calc_training_time = dTime

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
    """
    validate_range(p, 0.0, 1.0, "p (parallel fraction)")
    validate_positive(s, "s (speedup factor)")
    overall = 1 / ((1 - p) + (p / s))
    return overall

def calc_monthly_egress_cost(bytes_per_sec, cost_per_gb):
    """Calculates monthly cloud egress cost based on standard cloud egress rates."""
    b_s = _ensure_unit(bytes_per_sec, ureg.byte / ureg.second, "bytes_per_sec")
    monthly_bytes = b_s * (30 * ureg.day)
    cost = monthly_bytes * cost_per_gb
    return cost.m_as(ureg.dollar)

def calc_fleet_tco(unit_cost, power_w, quantity, years, kwh_price):
    """
    Calculates Total Cost of Ownership (TCO).
    
    Source: Barroso et al. (2018), "The Datacenter as a Computer."
    """
    u_cost = _ensure_unit(unit_cost, ureg.dollar, "unit_cost")
    p_w = _ensure_unit(power_w, ureg.watt, "power_w")
    price = _ensure_unit(kwh_price, ureg.dollar / ureg.kilowatt_hour, "kwh_price")
    time = _ensure_unit(years, ureg.year, "years")
    fleet_capex = u_cost * quantity
    total_energy = p_w * quantity * time
    power_opex = total_energy * price
    total = fleet_capex + power_opex
    return total.m_as(ureg.dollar)

def calc_bottleneck(ops, model_bytes, device_flops, device_bw):
    """
    Roofline bottleneck analysis.

    Source: Williams et al. (2009), "Roofline Model."

    Worked Example::

        Llama-3 8B FP16 on H100 (batch=1):
        ops = 2 * 8e9 = 16 GFLOP, model_bytes = 16 GB
        compute = 16e9 / 989e12 = 0.016 ms
        memory  = 16e9 / 3.35e12 = 4.78 ms
        → Memory Bound (memory >> compute)
    """
    compute_time = ops / device_flops
    memory_time = model_bytes / device_bw
    t_comp_ms = compute_time.m_as(ureg.millisecond)
    t_mem_ms = memory_time.m_as(ureg.millisecond)
    
    if t_comp_ms < 1e-15:
        return {
            "compute_ms": 0.0,
            "memory_ms": t_mem_ms,
            "bottleneck": "Memory",
            "ratio": float('inf'),
            "intensity": 0.0
        }

    if t_mem_ms < 1e-15:
        return {
            "compute_ms": t_comp_ms,
            "memory_ms": 0.0,
            "bottleneck": "Compute",
            "ratio": float('inf'),
            "intensity": float('inf')
        }

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

    Worked Example::

        1 GB gradient on 8 GPUs with 50 GB/s NVLink and 500 ns latency:
        T = 2*(8-1)/8 * 1e9/50e9 + 2*(8-1) * 500e-9
          = 1.75 * 0.02 + 14 * 500e-9
          = 35 ms + 7 μs ≈ 35 ms  (bandwidth-dominated)

    Args:
        message_bytes: Total message size in bytes (M)
        n_gpus: Number of GPUs in the ring (N)
        bandwidth_bytes_s: Per-link bandwidth in bytes/second (β)
        latency_s: Per-message startup latency in seconds (α)

    Returns:
        Quantity[second]: Estimated AllReduce time
    """
    validate_at_least(n_gpus, 1, "n_gpus")
    if n_gpus == 1:
        return Q_("0 second")
    msg = _ensure_unit(message_bytes, ureg.byte, "message_bytes")
    bw  = _ensure_unit(bandwidth_bytes_s, ureg.byte / ureg.second, "bandwidth_bytes_s")
    validate_positive(bw, "bandwidth_bytes_s")
    lat = _ensure_unit(latency_s, ureg.second, "latency_s")
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
    validate_at_least(n_gpus, 1, "n_gpus")
    if n_gpus == 1:
        return Q_("0 second")
    msg = _ensure_unit(message_bytes, ureg.byte, "message_bytes")
    bw  = _ensure_unit(bandwidth_bytes_s, ureg.byte / ureg.second, "bandwidth_bytes_s")
    validate_positive(bw, "bandwidth_bytes_s")
    lat = _ensure_unit(latency_s, ureg.second, "latency_s")
    log_n = math.ceil(math.log2(max(n_gpus, 2)))  # ceil handles non-power-of-2
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
    validate_at_least(n_gpus, 1, "n_gpus")
    msg = _ensure_unit(message_bytes, ureg.byte, "message_bytes")
    bw  = _ensure_unit(bandwidth_bytes_s, ureg.byte / ureg.second, "bandwidth_bytes_s")
    lat = _ensure_unit(latency_s, ureg.second, "latency_s")
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
    p = _ensure_unit(n_params, ureg.param, "n_params").to(ureg.count).magnitude
    d = _ensure_unit(n_tokens, ureg.count, "n_tokens").magnitude
    return (6 * p * d) * ureg.flop


def calc_activation_memory(n_layers, seq_len, batch_size, hidden_dim, n_heads=None,
                           precision_bytes=1, strategy="selective"):
    """
    Estimate activation memory for a Transformer layer.

    Source: Korthikanti et al. (2023), "Reducing Activation Memory in Transformer Training"

    The coefficients (34, 10, 2) already incorporate mixed-precision byte widths
    from the original paper (Table 1). The precision_bytes parameter defaults to 1
    to avoid double-counting; set it only if using a non-standard precision layout.

    Args:
        n_layers: Number of layers (L)
        seq_len: Sequence length (S)
        batch_size: Batch size (B)
        hidden_dim: Hidden dimension (H)
        n_heads: Number of attention heads (A)
        precision_bytes: Multiplier (default 1; Korthikanti coefficients already include byte widths)
        strategy: Recompute strategy ('none', 'selective', 'full')

    Returns:
        Quantity[byte]: Total activation memory
    """
    s, b, h = seq_len, batch_size, hidden_dim
    # Coefficients from Korthikanti et al. (2023) Table 1 already include byte widths
    # for mixed-precision training (FP16 activations + FP32 where needed).
    if strategy == "full":
        # Only store inputs to the block
        bytes_per_layer = 2 * s * b * h * precision_bytes
    elif strategy == "selective":
        # Store some intermediate activations to avoid full recompute
        bytes_per_layer = 10 * s * b * h * precision_bytes
    else:
        # No recompute: store everything
        bytes_per_layer = 34 * s * b * h * precision_bytes
        
    return (n_layers * bytes_per_layer) * ureg.byte


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
    # 1. Intra-node Reduce-Scatter (each GPU gets M/G of the reduced result)
    # We model the NCCL-style bucket-fused approach where reduce-scatter
    # partitions the gradient buffer so each GPU holds 1/G of the result.
    t_reduce = calc_ring_allreduce_time(message_bytes, gpus_per_node, intra_node_bw, intra_node_lat)

    # 2. Inter-node AllReduce (between corresponding shards across nodes)
    # Each lead GPU holds M/G after reduce-scatter — this is the key bandwidth
    # optimization of hierarchical collectives (NCCL, Horovod).
    reduced_message = _ensure_unit(message_bytes, ureg.byte, "message_bytes") / gpus_per_node
    t_allreduce_inter = calc_ring_allreduce_time(reduced_message, n_nodes, inter_node_bw, inter_node_lat)
    
    # 3. Intra-node Broadcast (back to all GPUs)
    # Broadcast only carries M/G bytes (the reduced shard), not the full M
    t_broadcast = calc_ring_allreduce_time(reduced_message, gpus_per_node, intra_node_bw, intra_node_lat)
    
    return t_reduce + t_allreduce_inter + t_broadcast


def calc_young_daly_interval(checkpoint_cost_s, mtbf_s):
    """
    Optimal checkpoint interval (Young's first-order approximation).

    τ_opt = √(2 × δ × M)

    This implements Young (1974), not Daly's (2006) higher-order correction
    which adds the checkpoint cost: τ_opt = √(2δM) + δ. The Young form is
    the standard simplification used in practice.

    Args:
        checkpoint_cost_s: Time to write one checkpoint in seconds (δ)
        mtbf_s: Mean Time Between Failures in seconds (M)

    Returns:
        Quantity[second]: Optimal checkpoint interval
    """
    delta = _ensure_unit(checkpoint_cost_s, ureg.second, "checkpoint_cost_s")
    mtbf  = _ensure_unit(mtbf_s, ureg.second, "mtbf_s")
    seconds = math.sqrt(2 * delta.m_as(ureg.second) * mtbf.m_as(ureg.second))
    return seconds * ureg.second


def calc_mtbf_cluster(component_mtbf_hours, n_components, correlation_factor=1.0):
    """
    Cluster MTBF from identical independent components.

    MTBF_cluster = (MTBF_component / N) × correlation_factor

    Args:
        component_mtbf_hours: Single component MTBF in hours (or Quantity[hour])
        n_components: Number of identical components
        correlation_factor: Multiplier for correlated failures (default 1.0).
            Values < 1.0 model correlated failures (e.g., shared power rail,
            same firmware bug) which reduce effective MTBF below the independent
            assumption. Values > 1.0 could model redundancy benefits.

    Returns:
        Quantity[hour]: Cluster MTBF
    """
    mtbf = _ensure_unit(component_mtbf_hours, ureg.hour, "component_mtbf_hours")
    return (mtbf / n_components * correlation_factor).to(ureg.hour)


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
    gpu = _ensure_unit(gpu_mtbf_h, ureg.hour, "gpu_mtbf_h")
    nic = _ensure_unit(nic_mtbf_h, ureg.hour, "nic_mtbf_h")
    psu = _ensure_unit(psu_mtbf_h, ureg.hour, "psu_mtbf_h")
    rate = n_gpus / gpu + n_nics / nic + n_psus / psu
    if other_mtbf_h is not None and n_other > 0:
        rate += n_other / _ensure_unit(other_mtbf_h, ureg.hour, "other_mtbf_h")
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


def calc_checkpoint_size(n_params, bytes_per_param=14):
    """
    Checkpoint size for mixed-precision Adam training.

    Size = N × bytes_per_param

    Default 14 bytes/param: 2 (FP16 weights) + 4 (FP32 master weights) +
    4 (FP32 momentum) + 4 (FP32 variance). Gradients are ephemeral
    and not checkpointed.

    Args:
        n_params: Number of model parameters
        bytes_per_param: Bytes per parameter (default 14 for mixed-precision Adam)

    Returns:
        Quantity[byte]: Checkpoint size
    """
    bpp = _ensure_unit(bytes_per_param, ureg.byte, "bytes_per_param")
    return (n_params * bpp).to(ureg.byte)


def calc_kv_cache_size(n_layers, n_heads, head_dim, seq_len, batch_size,
                       bytes_per_elem=2, kv_precision_bytes=None):
    """
    KV cache memory for autoregressive inference.

    Size = 2 × L × H × D × S × B × bytes

    The factor of 2 accounts for both K and V tensors.

    Worked Example::

        Llama-3 8B (32 layers, 8 KV heads, 128 head_dim) at seq_len=4096, batch=1, FP16:
        Size = 2 × 32 × 8 × 128 × 4096 × 1 × 2 = 536 MB

    Args:
        n_layers: Number of transformer layers (L)
        n_heads: Number of attention heads (H)
        head_dim: Dimension per head (D)
        seq_len: Sequence length / context window (S)
        batch_size: Batch size (B)
        bytes_per_elem: Bytes per element (default 2 for FP16/BF16)
        kv_precision_bytes: Optional override for KV cache precision (e.g., 1 for
            INT8 KV cache quantization). When provided, this overrides bytes_per_elem
            for the KV cache size calculation.

    Returns:
        Quantity[byte]: KV cache size
    """
    effective_bpe = kv_precision_bytes if kv_precision_bytes is not None else bytes_per_elem
    bpe = _ensure_unit(effective_bpe, ureg.byte, "kv_precision_bytes" if kv_precision_bytes is not None else "bytes_per_elem")
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

    Worked Example::

        1000-GPU cluster (MTBF = 50 hours), 14-day training job:
        P = 1 - e^(-(14*24)/50) = 1 - e^(-6.72) = 0.9988
        → 99.9% chance of at least one failure

    If both are Quantities: units auto-reconciled (pass hours or seconds freely).
    If both are raw numbers: caller must use consistent units.
    Mixed types (one Quantity, one raw) raise TypeError — ambiguous unit intent.

    Args:
        mtbf: Mean Time Between Failures (Quantity or raw number)
        job_duration: Job duration (Quantity or raw number; same units if raw)

    Returns:
        Probability of at least one failure (0.0 to 1.0)
    """
    validate_positive(mtbf, "mtbf")
    both_qty = isinstance(mtbf, ureg.Quantity) and isinstance(job_duration, ureg.Quantity)
    either_qty = isinstance(mtbf, ureg.Quantity) or isinstance(job_duration, ureg.Quantity)
    if either_qty and not both_qty:
        raise TypeError(
            "calc_failure_probability: both arguments must be Quantities or both raw numbers. "
            "Mixed types are ambiguous — attach units to the raw number first."
        )
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
    pf = _ensure_unit(peak_flops, ureg.flop / ureg.second, "peak_flops")
    return (pf * mfu * scaling_eff * goodput_ratio).to(ureg.flop / ureg.second)


# =============================================================================
# Inference & Serving Formulas
# =============================================================================

def calc_paged_kv_cache_size(n_layers, n_heads, head_dim, seq_len, batch_size,
                             page_size_tokens=16, bytes_per_elem=2):
    """
    KV cache memory for autoregressive inference using PagedAttention (vLLM).

    Size = 2 × L × H × D × (ceil(S / page_size) * page_size) × B × bytes

    Internal fragmentation is captured by the padded space in the last page,
    eliminating the 40-50% external fragmentation of contiguous allocation.

    Source: Kwon et al. (2023), "Efficient Memory Management for Large Language... PagedAttention"

    Args:
        n_layers: Number of transformer layers (L)
        n_heads: Number of attention heads (H)
        head_dim: Dimension per head (D)
        seq_len: Sequence length / context window (S)
        batch_size: Batch size (B)
        page_size_tokens: Tokens per page (typically 16)
        bytes_per_elem: Bytes per element (default 2 for FP16/BF16)

    Returns:
        tuple: (Quantity[byte] size, float fragmentation_percent)
    """
    bpe = _ensure_unit(bytes_per_elem, ureg.byte, "bytes_per_elem")
    padded_seq_len = math.ceil(seq_len / page_size_tokens) * page_size_tokens
    
    internal_frag = max(0, padded_seq_len - seq_len)
    frag_pct = internal_frag / padded_seq_len if padded_seq_len > 0 else 0.0
    
    size = (2 * n_layers * n_heads * head_dim * padded_seq_len * batch_size * bpe).to(ureg.byte)
    return size, frag_pct


def calc_queue_latency_mmc(arrival_rate_hz, service_rate_hz, num_servers):
    """
    M/M/c queueing model for inference tail latency approximation.
    
    Evaluates stable queues (λ < cμ) and calculates P50/P99 wait times
    based on the Erlang C formula.
    
    Args:
        arrival_rate_hz: Request arrival rate (λ)
        service_rate_hz: Request service rate per server (μ)
        num_servers: Number of replicas (c)
        
    Returns:
        tuple: (utilization, Quantity[second] p50_wait_time, Quantity[second] p99_wait_time)
    """
    lam = _ensure_unit(arrival_rate_hz, ureg.hertz, "arrival_rate_hz").magnitude
    mu = _ensure_unit(service_rate_hz, ureg.hertz, "service_rate_hz").magnitude
    c = max(1, int(num_servers))
    
    if lam >= c * mu or mu == 0:
        return 1.0, float('inf') * ureg.second, float('inf') * ureg.second
        
    rho = lam / (c * mu)
    
    # Erlang C calculation for probability of queuing (log-space for numerical stability).
    # Standard formula overflows math.factorial(c) for c > 170.
    # Using math.lgamma avoids intermediate overflow.
    a = c * rho  # offered load
    try:
        # log of numerator term: a^c / c! * 1/(1-rho)
        log_last = c * math.log(a) - math.lgamma(c + 1) - math.log(1 - rho)
        # log of each summation term: a^i / i!
        log_terms = [i * math.log(a) - math.lgamma(i + 1) if a > 0 else (-math.inf if i > 0 else 0.0) for i in range(c)]
        # log-sum-exp for numerical stability
        max_log = max(max(log_terms) if log_terms else -math.inf, log_last)
        sum_exp = sum(math.exp(t - max_log) for t in log_terms) + math.exp(log_last - max_log)
        p_wait = math.exp(log_last - max_log) / sum_exp
    except (OverflowError, ValueError, ZeroDivisionError):
        p_wait = rho

    # Safety clamp
    if math.isnan(p_wait) or math.isinf(p_wait):
        p_wait = rho
    p_wait = max(0.0, min(1.0, p_wait))
        
    rate_param = c * mu * (1 - rho)
    
    if p_wait < 0.5:
        p50_wait = 0.0
    else:
        p50_wait = -math.log(0.5 / p_wait) / rate_param
        
    if p_wait < 0.01:
        p99_wait = 0.0
    else:
        p99_wait = -math.log(0.01 / p_wait) / rate_param
        
    return rho, p50_wait * ureg.second, p99_wait * ureg.second
