"""Training time, scaling, roofline, and pipeline performance."""

from __future__ import annotations

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_positive, validate_at_least, validate_range


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


def calc_training_time_days(total_ops, num_devices, peak_flops_per_device, efficiency_eta):
    """Training duration in days."""
    duration = dTime(total_ops, num_devices, peak_flops_per_device, efficiency_eta)
    return duration.m_as(ureg.day)


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
    compute_time = ops / device_flops
    memory_time = model_bytes / device_bw
    t_comp_ms = compute_time.m_as(ureg.millisecond)
    t_mem_ms = memory_time.m_as(ureg.millisecond)

    # Degenerate-workload guards: a zero-FLOP workload (pure data movement) is
    # memory-bound by definition, and a zero-byte workload is compute-bound.
    # The 1e-15 epsilon catches float-rounded zeros; either case would make the
    # ratio/intensity divisions below blow up. The epsilon applies to MILLISECOND
    # magnitudes (effective threshold 1e-18 s) — keep that basis in mind if the
    # internal unit ever changes.
    if t_comp_ms < 1e-15:
        return {
            "compute_ms": 0.0,
            "memory_ms": t_mem_ms,
            "bottleneck": "Memory",
            "ratio": float("inf"),
            "intensity": 0.0,
        }

    if t_mem_ms < 1e-15:
        return {
            "compute_ms": t_comp_ms,
            "memory_ms": 0.0,
            "bottleneck": "Compute",
            "ratio": float("inf"),
            "intensity": float("inf"),
        }

    # The slower of the two independent ceilings binds; the ratio reports how
    # dominant the bottleneck is (always >= 1 by construction). Tie-break: at
    # intensity exactly equal to the ridge point (t_mem == t_comp) the strict
    # comparison reports "Compute" with ratio 1.0 — the balanced operating
    # point is classified as (just barely) compute-bound by convention.
    is_memory_bound = t_mem_ms > t_comp_ms
    ratio = t_mem_ms / t_comp_ms if is_memory_bound else t_comp_ms / t_mem_ms
    # Normalize before extracting magnitude so scaled inputs such as
    # 1 TFLOP / 1 GB report 1000 flop/byte rather than Pint's raw magnitude.
    intensity = (ops / model_bytes).to("flop/byte")
    return {
        "compute_ms": t_comp_ms,
        "memory_ms": t_mem_ms,
        "bottleneck": "Memory" if is_memory_bound else "Compute",
        "ratio": ratio,
        "intensity": intensity.magnitude,
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
