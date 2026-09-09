"""Agentic ML systems physics and economics accounting formulas (Volume III).

Domain scope:
- Trajectory execution latency budgets
- Trajectory error compounding and reliability bounds
- Radix tree KV prefix caching and memory reuse
- Test-time compute vs. capability trade-offs
- Multi-agent coordination tax
"""

from __future__ import annotations

from mlsysim.core.units import ureg
from mlsysim.core._validation import (
    validate_positive,
    validate_nonnegative,
    validate_range,
    validate_at_least,
)


def calc_trajectory_step_time(thinking_time, tool_time, verification_time=None):
    """
    Calculate the total wall-clock duration of a single agentic trajectory step.

    Parameters
    ----------
    thinking_time : Quantity
        Time spent on model inference (prefill + token decoding).
    tool_time : Quantity
        Time spent on external tool execution / environment sandboxing.
    verification_time : Quantity, optional
        Time spent on safety guards or output verification (default: 0s).

    Returns
    -------
    Quantity
        Total step duration in seconds.
    """
    total = thinking_time + tool_time
    if verification_time is not None:
        total = total + verification_time
    return total.to(ureg.second)


def calc_trajectory_reliability(step_success_rate: float, num_steps: int, verifier_recovery_rate: float = 0.0):
    """
    Calculate end-to-end trajectory success probability under compounding errors.

    Model:
        P_step_effective = step_success_rate + (1 - step_success_rate) * verifier_recovery_rate
        P_trajectory = (P_step_effective) ^ num_steps

    Parameters
    ----------
    step_success_rate : float
        Probability that an individual step succeeds without unrecoverable error (0.0 to 1.0).
    num_steps : int
        Total sequential steps in the trajectory horizon (>= 1).
    verifier_recovery_rate : float, optional
        Fraction of failed steps caught and repaired by verification checks (0.0 to 1.0).

    Returns
    -------
    float
        End-to-end trajectory success probability (0.0 to 1.0).
    """
    validate_range(step_success_rate, 0.0, 1.0, "step_success_rate")
    validate_at_least(num_steps, 1, "num_steps")
    validate_range(verifier_recovery_rate, 0.0, 1.0, "verifier_recovery_rate")

    effective_step_success = step_success_rate + (1.0 - step_success_rate) * verifier_recovery_rate
    return float(effective_step_success ** num_steps)


def calc_pareto_scale(mean_length: float, alpha: float) -> float:
    """
    Recover the Pareto scale parameter from a measured mean trajectory length.

    Agent trajectory lengths are heavy tailed: most tasks finish in a handful of
    steps while a small fraction run for hundreds. A Pareto tail with index
    ``alpha`` has mean ``alpha * x_min / (alpha - 1)``, so a measured mean pins
    the scale:

        x_min = mean * (alpha - 1) / alpha

    The mean exists only for ``alpha > 1``; below that the distribution has no
    finite mean and a scheduler cannot be sized from an average at all.

    Parameters
    ----------
    mean_length : float
        Measured mean trajectory length, in steps.
    alpha : float
        Pareto tail index, strictly greater than 1.

    Returns
    -------
    float
        Scale parameter x_min, in steps.
    """
    validate_positive(mean_length, "mean_length")
    if alpha <= 1.0:
        raise ValueError(
            "alpha must exceed 1 for the Pareto mean to exist; got %r. Below "
            "that the distribution has no finite mean." % (alpha,)
        )
    return float(mean_length * (alpha - 1.0) / alpha)


def calc_pareto_survival(length: float, scale: float, alpha: float) -> float:
    """
    Probability that a Pareto-distributed trajectory exceeds ``length`` steps.

    Model:
        P(N > x) = (x_min / x) ** alpha    for x >= x_min

    Parameters
    ----------
    length : float
        Step count to exceed.
    scale : float
        Pareto scale parameter x_min, as returned by ``calc_pareto_scale``.
    alpha : float
        Pareto tail index.

    Returns
    -------
    float
        Survival probability in [0, 1].
    """
    validate_positive(length, "length")
    validate_positive(scale, "scale")
    validate_positive(alpha, "alpha")
    if length <= scale:
        return 1.0
    return float((scale / length) ** alpha)


def calc_pareto_conditional_survival(attained: float, target: float, alpha: float) -> float:
    """
    Probability of reaching ``target`` steps given ``attained`` steps already run.

    The scale parameter cancels in the ratio of survival functions, so this
    depends only on the two step counts and the tail index:

        P(N > target | N > attained) = (attained / target) ** alpha

    This is the decreasing-hazard property that inverts operator intuition: a
    trajectory that has already run a long time is *more* likely to keep running,
    which is why attained service is the best remaining-work estimator a
    non-clairvoyant scheduler has.

    Parameters
    ----------
    attained : float
        Steps the trajectory has already completed.
    target : float
        Step count to survive to, at least ``attained``.
    alpha : float
        Pareto tail index.

    Returns
    -------
    float
        Conditional survival probability in [0, 1].
    """
    validate_positive(attained, "attained")
    validate_positive(alpha, "alpha")
    validate_at_least(target, attained, "target")
    return float((attained / target) ** alpha)


def calc_pareto_mean_residual_life(attained: float, alpha: float) -> float:
    """
    Expected remaining steps for a trajectory that has already run ``attained``.

    Model:
        E[N - x | N > x] = x / (alpha - 1)

    Remaining work grows with attained service rather than shrinking, which is
    the formal statement of why the session closest to finishing is the one that
    has barely started.

    Parameters
    ----------
    attained : float
        Steps the trajectory has already completed.
    alpha : float
        Pareto tail index, strictly greater than 1.

    Returns
    -------
    float
        Expected additional steps.
    """
    validate_positive(attained, "attained")
    if alpha <= 1.0:
        raise ValueError(
            "alpha must exceed 1 for mean residual life to be finite; got %r."
            % (alpha,)
        )
    return float(attained / (alpha - 1.0))


def calc_radix_cache_effective_latency(
    prompt_tokens: int,
    shared_prefix_tokens: int,
    prefill_rate,
    decode_rate,
    output_tokens: int = 1,
):
    """
    Calculate time to first token and total response time under Radix tree prefix caching.

    Parameters
    ----------
    prompt_tokens : int
        Total prompt tokens.
    shared_prefix_tokens : int
        Tokens already cached in KV memory from shared system prompt / previous turns.
    prefill_rate : Quantity
        Prefill processing throughput (e.g., tokens / second).
    decode_rate : Quantity
        Autoregressive token decode throughput (e.g., tokens / second).
    output_tokens : int, optional
        Number of output tokens to generate (default: 1).

    Returns
    -------
    Quantity
        Total response duration in seconds.
    """
    validate_at_least(prompt_tokens, 1, "prompt_tokens")
    validate_nonnegative(shared_prefix_tokens, "shared_prefix_tokens")
    validate_positive(prefill_rate, "prefill_rate")
    validate_positive(decode_rate, "decode_rate")

    uncached_tokens = max(0, prompt_tokens - shared_prefix_tokens)
    if hasattr(prefill_rate, "units"):
        prefill_time = (uncached_tokens / prefill_rate).to(ureg.second)
    else:
        prefill_time = (uncached_tokens / prefill_rate) * ureg.second

    if hasattr(decode_rate, "units"):
        decode_time = (output_tokens / decode_rate).to(ureg.second)
    else:
        decode_time = (output_tokens / decode_rate) * ureg.second

    return (prefill_time + decode_time).to(ureg.second)


def calc_test_time_compute_cost(
    base_sample_cost,
    num_samples: int,
    verifier_cost_per_sample=None,
    aggregation_cost=None,
):
    """
    Calculate total test-time compute expenditure for parallel or search-based generation.

    Parameters
    ----------
    base_sample_cost : Quantity
        Cost or FLOPs required to generate a single candidate trajectory.
    num_samples : int
        Number of parallel candidate samples (Best-of-N, Tree Search, etc.).
    verifier_cost_per_sample : Quantity, optional
        Verification / reward model cost evaluated per sample.
    aggregation_cost : Quantity, optional
        Cost of aggregating or consensus-ranking candidate samples.

    Returns
    -------
    Quantity
        Total compute cost in base units.
    """
    validate_at_least(num_samples, 1, "num_samples")
    total = base_sample_cost * num_samples
    if verifier_cost_per_sample is not None:
        total = total + (verifier_cost_per_sample * num_samples)
    if aggregation_cost is not None:
        total = total + aggregation_cost
    return total


def calc_multi_agent_coordination_overhead(num_agents: int, avg_message_tokens: int, cost_per_token=None):
    """
    Calculate token volume and cost overhead resulting from inter-agent coordination.

    In a fully connected interaction graph, communication volume scales as O(N^2);
    in a hierarchical or centralized coordinator topology, volume scales as O(N).

    Parameters
    ----------
    num_agents : int
        Number of active collaborating agents (>= 1).
    avg_message_tokens : int
        Average tokens exchanged per interaction edge.
    cost_per_token : Quantity, optional
        Cost per million tokens or base unit currency.

    Returns
    -------
    dict
        Dictionary containing pairwise and coordinator token totals and optional costs.
    """
    validate_at_least(num_agents, 1, "num_agents")
    validate_nonnegative(avg_message_tokens, "avg_message_tokens")

    coordinator_tokens = (num_agents - 1) * 2 * avg_message_tokens
    pairwise_tokens = num_agents * (num_agents - 1) * avg_message_tokens

    results = {
        "coordinator_tokens": coordinator_tokens,
        "pairwise_tokens": pairwise_tokens,
    }

    if cost_per_token is not None:
        results["coordinator_cost"] = coordinator_tokens * cost_per_token
        results["pairwise_cost"] = pairwise_tokens * cost_per_token

    return results
