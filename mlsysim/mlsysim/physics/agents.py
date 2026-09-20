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

    Source: modeling assumption. The step is a serial sum of inference, tool,
    and verification time, not taken from a published model; queueing,
    retries, and parallel tool calls are ignored.

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

    Source: the product over steps is the series-system reliability model
    for independent components, R_S = prod(R_i) (NIST/SEMATECH e-Handbook of
    Statistical Methods, Sec. 8.1.8.2, "Series model"), applied here with
    identical steps. The verifier_recovery_rate term is a modeling assumption,
    not taken from a published model; it treats recovery as independent of
    the step and of every other step.

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

    Source: Pareto type I distribution function (Nila, Das, and Balakrishna,
    "Goodness-of-fit testing for the Pareto type-I distribution based on a
    mean residual life characterization", arXiv:2609.04933, Eq. 1) and its
    conditional mean alpha * t / (alpha - 1) (same paper, Theorem 2, stated
    for x_min = 1; the scaled form follows by rescaling). That agent
    trajectory lengths follow a Pareto tail is a modeling assumption of this
    package, not a fitted measurement.

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

    Source: Pareto type I survival function, 1 - F(x) with
    F(x) = 1 - (x_min / x) ** alpha (Nila, Das, and Balakrishna,
    arXiv:2609.04933, Eq. 1).

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

    Source: the ratio S(target) / S(attained) of the Pareto type I survival
    function (Nila, Das, and Balakrishna, arXiv:2609.04933, Eq. 1), by the
    definition of conditional probability. The scheduling interpretation
    above is this package's framing.

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

    Source: Pareto type I mean residual life m(x) = x / (alpha - 1) for
    x >= x_min and alpha > 1 (Nila, Das, and Balakrishna, arXiv:2609.04933,
    Eq. 3).

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

    Source: the reuse mechanism is RadixAttention, which reuses the KV cache
    of requests that share a prompt prefix (Zheng et al., "SGLang: Efficient
    Execution of Structured Language Model Programs", arXiv:2312.07104). The
    latency expression, uncached_tokens / prefill_rate plus
    output_tokens / decode_rate, is a modeling assumption of this package
    (constant throughputs, no queueing, no growth of per-token cost with
    context length), not taken from the paper.

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

    Source: modeling assumption. The total is linear cost accounting,
    num_samples * (sample + verifier) + aggregation, not taken from a
    published model. Best-of-N and verifier-guided beam search, the strategies
    it prices, are described in Snell et al., "Scaling LLM Test-Time Compute
    Optimally can be More Effective than Scaling Model Parameters",
    arXiv:2408.03314.

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

    Source: the edge counts are graph combinatorics. A complete graph on N
    agents has N(N-1)/2 undirected edges (Weisstein, "Complete Graph",
    MathWorld), so one message in each direction is N(N-1) messages; a star
    around one coordinator has N-1 edges, or 2(N-1) messages. Charging
    avg_message_tokens per directed message per round is a modeling
    assumption, not taken from a published model.

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


def calc_multiagent_speedup(num_agents: int, s_serial: float, alpha: float = 0.0, beta: float = 0.0) -> float:
    """
    Calculate effective speedup for an agent ensemble under coordination tax.

    Model:
        D(M) = s_serial + (1 - s_serial) / M + alpha * M^2 + beta * M
        S(M) = 1 / D(M)

    where:
    - s_serial is the serial planning / decomposition fraction (Amdahl's serial term).
    - (1 - s_serial) is the parallelizable execution fraction.
    - alpha is the quadratic synchronization / pairwise cross-talk coefficient (O(M^2)).
    - beta is the linear coordination / supervisor provisioning coefficient (O(M)).

    Source: s_serial + (1 - s_serial) / M is Amdahl's law,
    Speedup = 1 / (s + p / N) with s + p = 1 (as stated in Gustafson,
    "Reevaluating Amdahl's Law", 1988). The alpha * M^2 and beta * M overhead
    terms are a modeling assumption of this package, not taken from a
    published model. They differ from Gunther's Universal Scalability Law,
    C(p) = p / (1 + sigma * (p - 1) + kappa * p * (p - 1)) (Gunther,
    arXiv:0808.1431, Eq. 5), whose normalized time 1 / C(p) grows linearly,
    not quadratically, in p.

    Parameters
    ----------
    num_agents : int
        Number of collaborating agents in the ensemble (>= 1).
    s_serial : float
        Serial planning fraction (0.0 to 1.0).
    alpha : float, optional
        Quadratic synchronization coefficient (default: 0.0, >= 0.0).
    beta : float, optional
        Linear coordination coefficient (default: 0.0, >= 0.0).

    Returns
    -------
    float
        Theoretical speedup factor S(M) relative to a single agent.
    """
    validate_at_least(num_agents, 1, "num_agents")
    validate_range(s_serial, 0.0, 1.0, "s_serial")
    validate_nonnegative(alpha, "alpha")
    validate_nonnegative(beta, "beta")

    m = float(num_agents)
    p_parallel = 1.0 - s_serial
    d = s_serial + (p_parallel / m) + (alpha * (m ** 2)) + (beta * m)
    if d <= 0.0:
        return 0.0
    return float(1.0 / d)


def calc_multiagent_optimal_concurrency(s_serial: float, alpha: float, beta: float = 0.0) -> float:
    """
    Calculate the optimal concurrency boundary M* for an agent ensemble.

    The stationary condition minimizing execution time D(M) satisfies:
        D'(M) = -(1 - s) / M^2 + 2 * alpha * M + beta = 0
        <=> 2 * alpha * M^3 + beta * M^2 - (1 - s) = 0

    When alpha == 0 and beta == 0, S(M) approaches 1/s monotonically as M -> inf.
    When alpha > 0, the cubic polynomial has strictly one positive real root by Descartes' Rule of Signs.

    Source: derived here by setting dD/dM = 0 for the calc_multiagent_speedup
    model, so it inherits that model's modeling assumption (the alpha and beta
    overhead terms are not taken from a published model).

    Parameters
    ----------
    s_serial : float
        Serial planning fraction (0.0 to 1.0).
    alpha : float
        Quadratic synchronization coefficient (>= 0.0).
    beta : float, optional
        Linear coordination coefficient (default: 0.0, >= 0.0).

    Returns
    -------
    float
        Optimal number of concurrent agents M* maximizing speedup.
    """
    validate_range(s_serial, 0.0, 1.0, "s_serial")
    validate_nonnegative(alpha, "alpha")
    validate_nonnegative(beta, "beta")

    p = 1.0 - s_serial
    if alpha == 0.0 and beta == 0.0:
        return float("inf")
    if p <= 0.0:
        return 1.0

    low = 0.1
    high = 1000.0

    def f(m):
        return 2.0 * alpha * (m ** 3) + beta * (m ** 2) - p

    while f(high) < 0:
        high *= 2.0
        if high > 1e6:
            return high

    for _ in range(100):
        mid = 0.5 * (low + high)
        val = f(mid)
        if abs(val) < 1e-12 or (high - low) < 1e-9:
            return float(mid)
        if val < 0:
            low = mid
        else:
            high = mid
    return float(0.5 * (low + high))
