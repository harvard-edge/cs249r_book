"""Training time, scaling, roofline, and pipeline performance."""

from __future__ import annotations

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_positive, validate_at_least, validate_range, validate_nonnegative


def dTime(total_ops, num_devices, peak_flops_per_device, efficiency_eta):
    """
    Core training time calculation; returns duration in seconds.

    This function implements the Iron Law of ML Training execution time,
    accounting for hardware scale and utilization efficiency.

    Parameters
    ----------
    total_ops : Quantity
        Total floating point operations required for the workload (e.g., FLOPs).
    num_devices : int
        Number of physical accelerators (e.g., GPUs or TPUs) in the cluster.
    peak_flops_per_device : Quantity
        The theoretical maximum throughput of a single device (e.g., TFLOP/s).
    efficiency_eta : float
        The Model FLOPs Utilization (MFU), representing the fraction of 
        peak throughput actually achieved by the workload (0.0 to 1.0).

    Returns
    -------
    Quantity
        The calculated training duration in seconds.
    """
    validate_at_least(num_devices, 1, "num_devices")
    validate_range(efficiency_eta, 0.0, 1.0, "efficiency_eta")
    validate_positive(efficiency_eta, "efficiency_eta")
    effective_throughput = num_devices * peak_flops_per_device * efficiency_eta
    duration = total_ops / effective_throughput
    return duration.to(ureg.second)


calc_training_time = dTime


def calc_amdahls_speedup(p, s):
    """
    Overall system speedup (Amdahl's Law).

    Calculates the theoretical maximum speedup of a system when only a 
    fraction of the system is improved.

    Parameters
    ----------
    p : float
        The proportion of execution time that the improvement applies to (0.0 to 1.0).
    s : float
        The speedup factor for the improved portion of execution (e.g., 2.0 for 2x faster).

    Returns
    -------
    float
        The overall theoretical speedup factor for the entire system.
    """
    validate_range(p, 0.0, 1.0, "p (parallel fraction)")
    validate_positive(s, "s (speedup factor)")
    return 1 / ((1 - p) + (p / s))


def calc_partial_slowdown_stretch(stall_fraction, slowdown):
    """
    Stretch of a task when only a fraction of its time is slowed down.

    Equation:
        stretch = (1 - f) + f * S

    Amdahl's law read as a slowdown: a fraction ``f`` of the unloaded time
    (for example, time stalled on shared-cache misses) runs ``S`` times
    slower under interference while the rest is unchanged. The loaded time
    is ``t_unloaded * stretch``. Equal to ``1 / calc_amdahls_speedup(f, 1/S)``.

    Source: Amdahl, "Validity of the single processor approach to achieving
    large scale computing capabilities," AFIPS Spring Joint Computer
    Conference, 1967 (fixed-fraction model, applied to a slowdown).
    """
    validate_range(stall_fraction, 0.0, 1.0, "stall_fraction")
    validate_at_least(slowdown, 1.0, "slowdown")
    return (1.0 - stall_fraction) + stall_fraction * slowdown


def calc_stall_share_breakeven(allowed_stretch, slowdown):
    """
    Largest slowed fraction of a task that keeps it within an allowed stretch.

    Equation:
        f_max = (C / t_unloaded - 1) / (S - 1)

    The inverse of ``calc_partial_slowdown_stretch``: with a budget ``C``
    (a declared WCET or deadline) and unloaded time ``t_unloaded``, the
    allowed stretch is ``C / t_unloaded``; a slowed share above ``f_max``
    crosses the budget. Capped at 1 (the whole task may be slowed).

    Source: Amdahl 1967 fixed-fraction model, solved for the fraction
    (see ``calc_partial_slowdown_stretch``).
    """
    validate_at_least(allowed_stretch, 1.0, "allowed_stretch")
    validate_at_least(slowdown, 1.0, "slowdown")
    if slowdown == 1.0:
        return 1.0
    return min(1.0, (allowed_stretch - 1.0) / (slowdown - 1.0))


def calc_strong_scaling_speedup(num_devices, communication_fraction):
    """
    Strong scaling speedup with communication fraction overhead.

    Parameters
    ----------
    num_devices : int
        Number of participating devices (>= 1).
    communication_fraction : float
        Fraction of single-device step time spent on communication in the
        baseline system (0.0 to 1.0).

    Returns
    -------
    float
        The effective strong-scaling speedup.
    """
    validate_at_least(num_devices, 1, "num_devices")
    validate_range(communication_fraction, 0.0, 1.0, "communication_fraction")
    return num_devices / (1 + (num_devices - 1) * communication_fraction)


def calc_universal_scalability_law(n, sigma, kappa):
    """
    Universal Scalability Law (Gunther, 1993).

    Calculates relative system throughput / speedup C(N) under concurrency N,
    accounting for both serialization contention (sigma) and coherency
    crosstalk / coordination overhead (kappa).

    Model:
        C(N) = N / (1 + sigma * (N - 1) + kappa * N * (N - 1))

    When kappa = 0, USL reduces exactly to Amdahl's Law with serial fraction sigma.
    When kappa > 0, C(N) exhibits retrograde scaling beyond N* = sqrt((1 - sigma) / kappa).

    Parameters
    ----------
    n : int or float
        Concurrency level (number of workers, threads, or agents, >= 1).
    sigma : float
        Contention parameter representing serialized resource fraction (0.0 to 1.0).
    kappa : float
        Coherency / cross-talk parameter representing pairwise coordination (>= 0.0).

    Returns
    -------
    float
        Relative system capacity / speedup C(N).
    """
    validate_at_least(n, 1, "n")
    validate_range(sigma, 0.0, 1.0, "sigma")
    validate_nonnegative(kappa, "kappa")
    denom = 1.0 + sigma * (n - 1) + kappa * n * (n - 1)
    if denom <= 0.0:
        return 0.0
    return float(n / denom)


def calc_usl_optimal_concurrency(sigma, kappa):
    """
    Optimal concurrency N* under the Universal Scalability Law.

    Model:
        N* = sqrt((1 - sigma) / kappa)  for kappa > 0
        N* = inf                       for kappa == 0

    Parameters
    ----------
    sigma : float
        Contention parameter (0.0 to 1.0).
    kappa : float
        Coherency parameter (>= 0.0).

    Returns
    -------
    float
        The optimal concurrency level N* where throughput peaks before retrograde collapse.
    """
    import math

    validate_range(sigma, 0.0, 1.0, "sigma")
    validate_nonnegative(kappa, "kappa")
    if kappa == 0.0:
        return float("inf")
    if sigma >= 1.0:
        return 1.0
    return float(math.sqrt((1.0 - sigma) / kappa))


def calc_bottleneck(ops, model_bytes, device_flops, device_bw):
    """
    Roofline bottleneck analysis (Williams et al., 2009).

    Determines whether a workload is compute-bound or memory-bound on a 
    specific hardware device by evaluating the Roofline model's ridge point.

    Parameters
    ----------
    ops : Quantity
        Total operations in the workload (e.g., FLOPs).
    model_bytes : Quantity
        Total memory movement required for the workload (e.g., Bytes).
    device_flops : Quantity
        Peak compute throughput of the hardware (e.g., FLOP/s).
    device_bw : Quantity
        Peak memory bandwidth of the hardware (e.g., Byte/s).

    Returns
    -------
    dict
        A dictionary containing:
        - compute_ms (float): Time spent computing (ms).
        - memory_ms (float): Time spent moving data (ms).
        - bottleneck (str): 'Compute' or 'Memory'.
        - ratio (float): The ratio of the bottlenecked time to the non-bottlenecked time.
        - intensity (float): The arithmetic intensity (FLOP/Byte) of the workload.
    """
    # Arithmetic intensity is an intrinsic property of the workload (FLOP / Byte)
    # independent of device performance:
    ops_val = ops.m_as(ureg.flop) if hasattr(ops, "m_as") else float(ops)
    bytes_val = model_bytes.m_as(ureg.byte) if hasattr(model_bytes, "m_as") else float(model_bytes)

    if bytes_val < 1e-15:
        intensity_val = float("inf") if ops_val >= 1e-15 else 0.0
    elif ops_val < 1e-15:
        intensity_val = 0.0
    elif hasattr(ops, "to") and hasattr(model_bytes, "to"):
        # Normalize before extracting magnitude so scaled inputs such as
        # 1 TFLOP / 1 GB report 1000 flop/byte rather than Pint's raw magnitude.
        intensity_val = (ops / model_bytes).to("flop/byte").magnitude
    else:
        intensity_val = float(ops / model_bytes)

    compute_time = ops / device_flops
    memory_time = model_bytes / device_bw
    t_comp_ms = compute_time.m_as(ureg.millisecond)
    t_mem_ms = memory_time.m_as(ureg.millisecond)

    # Degenerate-workload / infinite-hardware guards:
    # If compute or memory time vanishes (< 1e-15 ms, effective threshold 1e-18 s),
    # the non-zero resource binds with infinite ratio.
    if t_comp_ms < 1e-15 and t_mem_ms < 1e-15:
        return {
            "compute_ms": 0.0,
            "memory_ms": 0.0,
            "bottleneck": "Compute",
            "ratio": 1.0,
            "intensity": intensity_val,
        }

    if t_comp_ms < 1e-15:
        return {
            "compute_ms": 0.0,
            "memory_ms": t_mem_ms,
            "bottleneck": "Memory",
            "ratio": float("inf"),
            "intensity": intensity_val,
        }

    if t_mem_ms < 1e-15:
        return {
            "compute_ms": t_comp_ms,
            "memory_ms": 0.0,
            "bottleneck": "Compute",
            "ratio": float("inf"),
            "intensity": intensity_val,
        }

    # The slower of the two independent ceilings binds; the ratio reports how
    # dominant the bottleneck is (always >= 1 by construction). Tie-break: at
    # intensity exactly equal to the ridge point (t_mem == t_comp) the strict
    # comparison reports "Compute" with ratio 1.0 — the balanced operating
    # point is classified as (just barely) compute-bound by convention.
    is_memory_bound = t_mem_ms > t_comp_ms
    ratio = t_mem_ms / t_comp_ms if is_memory_bound else t_comp_ms / t_mem_ms
    return {
        "compute_ms": t_comp_ms,
        "memory_ms": t_mem_ms,
        "bottleneck": "Memory" if is_memory_bound else "Compute",
        "ratio": ratio,
        "intensity": intensity_val,
    }


def calc_pipeline_bubble(n_stages, n_microbatches, v_stages=1):
    """
    Pipeline bubble fraction (GPipe / 1F1B / Interleaved 1F1B).

    Calculates the proportion of idle time (bubble) injected into the 
    pipeline schedule due to filling and draining the pipeline.

    Parameters
    ----------
    n_stages : int
        Number of physical pipeline stages (e.g., number of nodes/devices).
    n_microbatches : int
        Number of microbatches (M) injected into the pipeline per step.
    v_stages : int, optional
        Number of virtual stages per physical device for interleaved schedules.
        Default is 1 (standard 1F1B/GPipe).

    Returns
    -------
    float
        The fraction of time spent idle (0.0 to 1.0).
    """
    validate_at_least(n_stages, 1, "n_stages")
    validate_at_least(n_microbatches, 1, "n_microbatches")
    validate_at_least(v_stages, 1, "v_stages")
    # Fill + drain idle the pipeline for (p-1) microbatch slots out of a total
    # of (m + p - 1) slots per step. Interleaving v virtual stages per device
    # subdivides the work, which acts like having v*m microbatches: the bubble
    # shrinks as either m or v grows.
    return (n_stages - 1) / (v_stages * n_microbatches + n_stages - 1)


def calc_effective_flops(peak_flops, mfu, scaling_eff, goodput_ratio):
    """
    Effective FLOPS after MFU, scaling efficiency, and goodput overheads.

    Parameters
    ----------
    peak_flops : Quantity
        Theoretical peak throughput of the hardware (e.g., TFLOP/s).
    mfu : float
        Model FLOPs Utilization (0.0 to 1.0).
    scaling_eff : float
        Scaling efficiency accounting for distributed overheads (0.0 to 1.0).
    goodput_ratio : float
        Fraction of time doing useful work vs. recovery/checkpointing (0.0 to 1.0).

    Returns
    -------
    Quantity
        The effective usable throughput in FLOP/s.
    """
    from ._units import _ensure_unit

    validate_range(mfu, 0.0, 1.0, "mfu")
    validate_range(scaling_eff, 0.0, 1.0, "scaling_eff")
    validate_range(goodput_ratio, 0.0, 1.0, "goodput_ratio")
    pf = _ensure_unit(peak_flops, ureg.flop / ureg.second, "peak_flops")
    validate_positive(pf, "peak_flops")
    return (pf * mfu * scaling_eff * goodput_ratio).to(ureg.flop / ureg.second)


def calc_pareto_frontier(points):
    """
    Indices of the points no other point beats on both axes.

    ``points`` is a sequence of (cost, value) pairs, lower cost and higher
    value better (latency and accuracy, for example). A point is dominated
    when another is at least as good on both axes and strictly better on one.
    Added 2026-10-20.
    """
    keep = []
    for i, (c, v) in enumerate(points):
        if not any((c2 <= c and v2 >= v) and (c2 < c or v2 > v) for c2, v2 in points):
            keep.append(i)
    return keep
