"""Training time, scaling, roofline, and pipeline performance."""

from __future__ import annotations

import math

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_positive, validate_at_least, validate_range, validate_nonnegative


def dTime(total_ops, num_devices, peak_flops_per_device, efficiency_eta):
    """Estimate execution time from total work and effective fleet throughput.

    Parameters
    ----------
    total_ops : Quantity
        Nonnegative total floating-point operation count.
    num_devices : int
        Number of identical accelerators, at least one.
    peak_flops_per_device : Quantity
        Positive peak computation rate per device in FLOP/s-compatible units.
    efficiency_eta : float
        Achieved fraction of aggregate peak throughput, in (0, 1].

    Returns
    -------
    Quantity
        Modeled execution duration in seconds.

    Notes
    -----
    Equation: T = C / (N F_peak eta).

    Assumptions: The workload is divisible across identical devices; eta already includes
    the utilization losses the caller intends to model.

    Excludes: Separate memory, communication, and recovery terms; add these explicitly or
    incorporate them consistently into eta.

    References
    ----------
    Derivation: Aggregate useful computation rate is N F_peak eta; divide the required work C by that rate.

    Examples
    --------
    >>> from mlsysim.core.units import Q_
    >>> dTime(Q_(8, "flop"), 2, Q_(4, "flop/s"), 0.5) == Q_(2, "s")
    True
    """
    validate_at_least(num_devices, 1, "num_devices")
    validate_range(efficiency_eta, 0.0, 1.0, "efficiency_eta")
    validate_positive(efficiency_eta, "efficiency_eta")
    effective_throughput = num_devices * peak_flops_per_device * efficiency_eta
    duration = total_ops / effective_throughput
    return duration.to(ureg.second)


calc_training_time = dTime


def calc_amdahls_speedup(p, s):
    """Calculate speedup when one fraction of execution is accelerated.

    Parameters
    ----------
    p : float
        Fraction of original elapsed time affected by the improvement, in [0, 1].
    s : float
        Positive speedup of the affected fraction; positive infinity removes its time
        entirely.

    Returns
    -------
    float
        Overall dimensionless speedup; infinity when all original time is removed.

    Notes
    -----
    Equation: S = 1 / [(1 - p) + p/s].

    Assumptions: The original work and unaffected elapsed time remain fixed.

    Excludes: New overhead, workload growth, and a changing accelerated fraction.

    References
    ----------
    Amdahl (1967), "Validity of the single processor approach to achieving
    large scale computing capabilities." https://doi.org/10.1145/1465482.1465560
    (bibliography key: amdahl1967).

    Derivation: Normalize original elapsed time to one, retain 1-p, and divide the affected p by s. This is Amdahl's fixed-work speedup relation.

    Examples
    --------
    >>> calc_amdahls_speedup(0.5, float("inf"))
    2.0
    """
    validate_range(p, 0.0, 1.0, "p (parallel fraction)")
    if s == math.inf:
        return math.inf if p == 1.0 else 1 / (1 - p)
    validate_positive(s, "s (speedup factor)")
    return 1 / ((1 - p) + (p / s))


def calc_partial_slowdown_stretch(stall_fraction, slowdown):
    """Calculate elapsed-time growth when one task fraction slows.

    Parameters
    ----------
    stall_fraction : float
        Fraction of unloaded elapsed time exposed to the slowdown, in [0, 1].
    slowdown : float
        Multiplicative duration increase for that fraction, at least one.

    Returns
    -------
    float
        Loaded duration divided by unloaded duration.

    Notes
    -----
    Equation: stretch = (1 - f) + f S.

    Assumptions: The affected fraction f is measured in the unloaded task and remains fixed.

    Excludes: Changes in work, additional interference paths, and queueing beyond the
    supplied slowdown.

    References
    ----------
    Amdahl (1967), "Validity of the single processor approach to achieving
    large scale computing capabilities." https://doi.org/10.1145/1465482.1465560
    (bibliography key: amdahl1967).

    Derivation: Apply the fixed-fraction time decomposition of Amdahl's law, with a slowdown rather than a speedup.

    Examples
    --------
    >>> calc_partial_slowdown_stretch(0.5, 3) == (1 - 0.5) + 0.5 * 3
    True
    """
    validate_range(stall_fraction, 0.0, 1.0, "stall_fraction")
    validate_at_least(slowdown, 1.0, "slowdown")
    return (1.0 - stall_fraction) + stall_fraction * slowdown


def calc_stall_share_breakeven(allowed_stretch, slowdown):
    """Find the largest slowed task fraction within a duration budget.

    Parameters
    ----------
    allowed_stretch : float
        Maximum total duration divided by unloaded duration, at least one.
    slowdown : float
        Duration multiplier for the affected portion, at least one.

    Returns
    -------
    float
        Largest admissible affected fraction, capped at one.

    Notes
    -----
    Equation: f_max = min(1, (allowed_stretch - 1) / (slowdown - 1)); f_max = 1 when
    slowdown = 1.

    Assumptions: The allowed duration is no smaller than the unloaded duration.

    Excludes: Time-varying interference and uncertainty in the baseline duration.

    References
    ----------
    Derivation: Solve 1 + f(S - 1) <= allowed_stretch for f and cap the physical fraction at one.

    Examples
    --------
    >>> calc_stall_share_breakeven(2, 3) == (2 - 1) / (3 - 1)
    True
    """
    validate_at_least(allowed_stretch, 1.0, "allowed_stretch")
    validate_at_least(slowdown, 1.0, "slowdown")
    if slowdown == 1.0:
        return 1.0
    return min(1.0, (allowed_stretch - 1.0) / (slowdown - 1.0))


def calc_strong_scaling_speedup(num_devices, communication_fraction):
    """Estimate fixed-work scaling with a nonparallel time fraction.

    Parameters
    ----------
    num_devices : int
        Participating device count, at least one.
    communication_fraction : float
        Fraction of baseline time modeled as nonparallel communication, in [0, 1].

    Returns
    -------
    float
        Dimensionless speedup relative to one device.

    Notes
    -----
    Equation: S(N) = N / [1 + (N - 1) f].

    Assumptions: Communication fraction f is constant and the remaining work scales
    perfectly.

    Excludes: Topology-dependent communication growth, overlap changes, and load imbalance.

    References
    ----------
    Derivation: Rewrite Amdahl's 1 / [f + (1-f)/N] by multiplying numerator and denominator by N.

    Examples
    --------
    >>> calc_strong_scaling_speedup(4, 0.0)
    4.0
    """
    validate_at_least(num_devices, 1, "num_devices")
    validate_range(communication_fraction, 0.0, 1.0, "communication_fraction")
    return num_devices / (1 + (num_devices - 1) * communication_fraction)


def calc_universal_scalability_law(n, sigma, kappa):
    """Evaluate capacity under serialization and pairwise coordination costs.

    Parameters
    ----------
    n : int or float
        Dimensionless concurrency, at least one.
    sigma : float
        Serialization/contention parameter in [0, 1].
    kappa : float
        Nonnegative pairwise-coordination coefficient.

    Returns
    -------
    float
        Relative capacity C(n), normalized to one at n = 1.

    Notes
    -----
    Equation: C(n) = n / [1 + sigma(n - 1) + kappa n(n - 1)].

    Assumptions: sigma and kappa are supplied workload parameters, not hardware constants.

    Excludes: A fit or independent validation of the parameters and explicit queueing
    dynamics.

    References
    ----------
    Gunther (2008), "A General Theory of Computational Scalability Based on
    Rational Functions." https://arxiv.org/abs/0808.1431
    (bibliography key: gunther2008general).

    Derivation: The Universal Scalability Law models normalized overhead as a serialization term proportional to n-1 and a coordination term proportional to n(n-1); capacity divides concurrency by this overhead. These terms define the model rather than a physical identity.

    Examples
    --------
    >>> calc_universal_scalability_law(4, 0.0, 0.0)
    4.0
    """
    validate_at_least(n, 1, "n")
    validate_range(sigma, 0.0, 1.0, "sigma")
    validate_nonnegative(kappa, "kappa")
    denom = 1.0 + sigma * (n - 1) + kappa * n * (n - 1)
    if denom <= 0.0:
        return 0.0
    return float(n / denom)


def calc_usl_optimal_concurrency(sigma, kappa):
    """Find the stationary concurrency of the scalability model.

    Parameters
    ----------
    sigma : float
        Serialization parameter in [0, 1].
    kappa : float
        Nonnegative pairwise-coordination coefficient.

    Returns
    -------
    float
        Continuous stationary concurrency, not rounded or clamped to a realizable device
        count; infinity for kappa = 0 and one for sigma = 1 with positive kappa.

    Notes
    -----
    Equation: N_star = sqrt((1 - sigma) / kappa) for kappa > 0 and sigma < 1.

    Assumptions: The continuous USL model applies with fixed coefficients. Values below one
    require boundary handling by the caller.

    Excludes: Integer search, capacity constraints, and an empirical fit of USL
    coefficients.

    References
    ----------
    Gunther (2008), "A General Theory of Computational Scalability Based on
    Rational Functions." https://arxiv.org/abs/0808.1431
    (bibliography key: gunther2008general).

    Derivation: Set the derivative of n/[``1+sigma(n-1)+kappa*n*(n-1)``] to zero. Its numerator is ``1-sigma-kappa*n**2``.

    Examples
    --------
    >>> calc_usl_optimal_concurrency(0.0, 0.25)
    2.0
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
    """Compare compute and memory-transfer times for a workload.

    Parameters
    ----------
    ops : Quantity
        Nonnegative workload operation count in FLOPs.
    model_bytes : Quantity
        Nonnegative total traffic in bytes; this argument represents traffic, not
        necessarily model storage alone.
    device_flops : Quantity
        Positive available computation rate in FLOP/s.
    device_bw : Quantity
        Positive available memory bandwidth in byte/s.

    Returns
    -------
    dict
        Compute_ms and memory_ms are time magnitudes; bottleneck is Compute or Memory; ratio
        compares larger to smaller time; intensity is FLOP/byte.

    Notes
    -----
    Equation: T_compute = C/F; T_memory = M/BW; intensity = C/M.

    Assumptions: Compute and memory are independent ceilings; ties are labeled Compute.
    Times below the implementation zero threshold use explicit degenerate-work cases.

    Excludes: Dispatch overhead, communication, nonideal overlap, and any traffic omitted
    from model_bytes.

    References
    ----------
    Williams et al. (2009), "Roofline: An Insightful Visual Performance Model for Multicore Architectures." https://doi.org/10.1145/1498765.1498785 (bibliography key: williams2009).

    Examples
    --------
    >>> from mlsysim.core.units import Q_
    >>> result = calc_bottleneck(Q_(2, "flop"), Q_(4, "byte"), Q_(2, "flop/s"), Q_(2, "byte/s"))
    >>> result["bottleneck"]
    'Memory'
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
    """Estimate the fill-and-drain fraction of a balanced pipeline.

    Parameters
    ----------
    n_stages : int
        Positive physical pipeline-stage count.
    n_microbatches : int
        Positive microbatch count per training step.
    v_stages : int, optional
        Positive virtual-stage factor; default one represents no interleaving.

    Returns
    -------
    float
        Dimensionless idle fraction in [0, 1).

    Notes
    -----
    Equation: bubble = (p - 1) / (v m + p - 1).

    Assumptions: Stages have equal duration; virtual stages reduce the fill/drain penalty
    according to the stated schedule approximation.

    Excludes: Stage imbalance, communication stalls, runtime scheduling details, and
    activation feasibility.

    References
    ----------
    Derivation: Count p-1 fill/drain slots against ``v*m`` useful slots in the balanced pipeline approximation. The virtual-stage substitution summarizes interleaving; it is not an event-level schedule simulation.

    Examples
    --------
    >>> calc_pipeline_bubble(1, 4)
    0.0
    >>> calc_pipeline_bubble(4, 8, 2) < calc_pipeline_bubble(4, 8)
    True
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
    """Apply utilization, scaling, and goodput factors to peak rate.

    Parameters
    ----------
    peak_flops : Quantity or float
        Positive computation rate; bare values are interpreted as FLOP/s.
    mfu : float
        Model FLOP utilization factor in [0, 1].
    scaling_eff : float
        Additional distributed scaling factor in [0, 1].
    goodput_ratio : float
        Fraction of elapsed time contributing useful progress, in [0, 1].

    Returns
    -------
    Quantity
        Effective useful computation rate in FLOP/s.

    Notes
    -----
    Equation: F_effective = F_peak * mfu * scaling_eff * goodput_ratio.

    Assumptions: Factors describe distinct losses; the caller must avoid charging a loss
    twice.

    Excludes: Estimating the factors, phase-dependent utilization, and changing workload
    intensity.

    References
    ----------
    Derivation: Apply successive dimensionless useful-work fractions to the same baseline rate; the product is valid only when the factors do not duplicate losses.

    Examples
    --------
    >>> from mlsysim.core.units import Q_
    >>> calc_effective_flops(Q_(8, "flop/s"), 0.5, 0.5, 1.0) == Q_(2, "flop/s")
    True
    """
    from ._units import _ensure_unit

    validate_range(mfu, 0.0, 1.0, "mfu")
    validate_range(scaling_eff, 0.0, 1.0, "scaling_eff")
    validate_range(goodput_ratio, 0.0, 1.0, "goodput_ratio")
    pf = _ensure_unit(peak_flops, ureg.flop / ureg.second, "peak_flops")
    validate_positive(pf, "peak_flops")
    return (pf * mfu * scaling_eff * goodput_ratio).to(ureg.flop / ureg.second)


def calc_pareto_frontier(points):
    """Return indices of points undominated in cost and value.

    Parameters
    ----------
    points : sequence of tuple of float
        Pairs (cost, value), with lower cost and higher value preferred. Each axis must use
        a consistent unit.

    Returns
    -------
    list of int
        Original indices of nondominated points; equal duplicates are retained.

    Notes
    -----
    Equation: Point i is excluded if some j has cost_j <= cost_i and value_j >= value_i,
    with at least one strict inequality.

    Assumptions: Both objectives are deterministic and directly comparable within their
    axes.

    Excludes: Uncertainty, scalar preference weights, and constraints beyond the two axes.

    References
    ----------
    Derivation: Apply the definition of Pareto dominance pairwise to every input point.

    Examples
    --------
    >>> calc_pareto_frontier([(1, 2), (2, 1), (2, 3)])
    [0, 2]
    """
    keep = []
    for i, (c, v) in enumerate(points):
        if not any((c2 <= c and v2 >= v) and (c2 < c or v2 > v) for c2, v2 in points):
            keep.append(i)
    return keep
