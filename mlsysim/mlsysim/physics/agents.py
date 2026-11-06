"""Agentic ML systems physics and memory accounting formulas (Volume III)."""

from __future__ import annotations

import math
from mlsysim.core._validation import validate_at_least, validate_nonnegative, validate_range


def calc_kv_cache_bytes_per_token(
    model=None,
    *,
    n_layers: int | None = None,
    n_kv_heads: int | None = None,
    head_dim: int | None = None,
    bytes_per_elem=2,
):
    """Calculate the key/value cache footprint per token.

    Parameters
    ----------
    model : TransformerWorkload or AgentArchitecture, optional
        Architecture supplying layers, KV heads, and head width; explicit counts
        override it. Agent architectures are unwrapped through reference_model.
    n_layers : int, optional
        Positive layer count; required when model is absent.
    n_kv_heads : int, optional
        Positive key/value head count; required when model is absent.
    head_dim : int, optional
        Positive elements per head; by default hidden_dim // heads, or head_dim when
        available, otherwise 128.
    bytes_per_elem : Quantity or int, optional
        Bytes per stored cache element; default 2 represents FP16/BF16 storage.

    Returns
    -------
    Quantity
        Bytes stored for one sequence token across all layers.

    Notes
    -----
    Equation: m_token = 2 L H_kv d_head b, where b is bytes per element.

    Assumptions: One key tensor and one value tensor share the stated shape and
    precision.

    Excludes: Allocator metadata, padding, fragmentation, and compressed
    latent-attention caches.

    References
    ----------
    Derivation: Count L H_kv d_head elements separately in keys and values, then
    multiply by element width.

    Examples
    --------
    >>> from mlsysim.core.units import ureg
    >>> result = calc_kv_cache_bytes_per_token(n_layers=2, n_kv_heads=3, head_dim=4)
    >>> result == 2 * 2 * 3 * 4 * 2 * ureg.byte
    True
    """
    if model is not None:
        if hasattr(model, "reference_model"):
            model = model.reference_model
        layers = model.layers if n_layers is None else n_layers
        kv_heads = getattr(model, "kv_heads", model.heads) if n_kv_heads is None else n_kv_heads
        if head_dim is not None:
            d_head = head_dim
        elif hasattr(model, "hidden_dim") and hasattr(model, "heads") and model.heads > 0:
            d_head = model.hidden_dim // model.heads
        else:
            d_head = getattr(model, "head_dim", 128)
    else:
        layers = n_layers
        kv_heads = n_kv_heads
        d_head = head_dim

    if layers is None or kv_heads is None or d_head is None:
        raise ValueError("Must provide model or all of (n_layers, n_kv_heads, head_dim).")

    validate_at_least(layers, 1, "n_layers")
    validate_at_least(kv_heads, 1, "n_kv_heads")
    validate_at_least(d_head, 1, "head_dim")

    from .memory import calc_kv_cache_size

    return calc_kv_cache_size(layers, kv_heads, d_head, 1, 1, bytes_per_elem)


# --- Trajectory reliability and cost per accepted task (added 2026-10-20) ----
# Horizon compounding, cost per accepted task, independent retries, and the
# input an append-only trajectory resends. Each is the closed form of a
# recurring agent calculation; the solvers in engine/solvers/agents.py use them.


def calc_unchecked_success(step_success: float, horizon: float) -> float:
    """Calculate unchecked task success from per-step success.

    Parameters
    ----------
    step_success : float
        Independent per-step success probability in [0, 1].
    horizon : float
        Nonnegative number of steps; whole counts describe discrete tasks.

    Returns
    -------
    float
        Dimensionless probability that every step succeeds.

    Notes
    -----
    Equation: P_task = p_step ** H.

    Assumptions: The same success probability applies independently to each unchecked
    step.

    Excludes: Correlated errors, checks, recovery, and changing difficulty.

    References
    ----------
    Derivation: The intersection probability of independent step successes is their
    product.

    Examples
    --------
    >>> calc_unchecked_success(0.5, 3) == 0.5 ** 3
    True
    """
    return step_success**horizon


def calc_step_error_for_target(target_success: float, horizon: float) -> float:
    """Find the per-step error rate allowed by a task success target.

    Parameters
    ----------
    target_success : float
        Target task success probability in [0, 1].
    horizon : float
        Positive number of unchecked steps.

    Returns
    -------
    float
        Dimensionless maximum per-step error under the independent-step model.

    Notes
    -----
    Equation: epsilon = 1 - target_success ** (1 / horizon).

    Assumptions: Each step has an identical, independent error probability.

    Excludes: Error dependence, checking, recovery, and finite-precision root error.

    References
    ----------
    Derivation: Invert target = (1 - epsilon) ** H; for small epsilon, H epsilon
    approximates -ln(target).

    Examples
    --------
    >>> error = calc_step_error_for_target(0.5 ** 3, 3)
    >>> error == 0.5
    True
    """
    return 1 - target_success ** (1 / horizon)


def calc_first_horizon_below(step_success: float, threshold: float) -> int:
    """Find the first whole horizon below a success threshold.

    Parameters
    ----------
    step_success : float
        Per-step success probability strictly between 0 and 1.
    threshold : float
        Task success threshold strictly between 0 and 1.

    Returns
    -------
    int
        First positive whole-step horizon with task success below threshold.

    Notes
    -----
    Equation: n = floor(log(threshold) / log(step_success)) + 1.

    Assumptions: Identical independent step success and an exact strict inequality.

    Excludes: Correlations and numerical ambiguity when logarithms lie near an integer.

    References
    ----------
    Derivation: Solve p ** n < threshold for n; log(p) is negative, so n must exceed the
    log ratio.

    Examples
    --------
    >>> n = calc_first_horizon_below(0.5, 0.3)
    >>> 0.5 ** n < 0.3 <= 0.5 ** (n - 1)
    True
    """
    return math.floor(math.log(threshold) / math.log(step_success)) + 1


def calc_cost_per_accepted(attempt_cost, acceptance_probability: float):
    """Calculate expected cost per accepted task.

    Parameters
    ----------
    attempt_cost : Quantity or float
        Expected cost of one attempt; retain the chosen unit, such as seconds or
        dollars.
    acceptance_probability : float
        Independent per-attempt acceptance probability in (0, 1].

    Returns
    -------
    Quantity or float
        Expected cost per accepted task, in the attempt-cost unit.

    Notes
    -----
    Equation: C_accepted = C_attempt / p_accept.

    Assumptions: Attempts repeat until acceptance with fixed expected cost and success
    probability.

    Excludes: Retry limits, dependencies between attempts, and changes in cost after
    failure.

    References
    ----------
    Derivation: A geometric number of attempts has mean 1 / p_accept.

    Examples
    --------
    >>> from mlsysim.core.units import ureg
    >>> calc_cost_per_accepted(2 * ureg.second, 0.5) == 4 * ureg.second
    True
    """
    return attempt_cost / acceptance_probability


def calc_break_even_acceptance(acceptance_probability: float, cut: float, share: float) -> float:
    """Find the acceptance rate equivalent to a component price cut.

    Parameters
    ----------
    acceptance_probability : float
        Current acceptance probability in [0, 1].
    cut : float
        Fractional reduction of the selected cost component in [0, 1].
    share : float
        Selected component share of total attempt cost in [0, 1]; cut * share must be
        below 1.

    Returns
    -------
    float
        Required acceptance probability; a value above 1 means the target is
        unattainable.

    Notes
    -----
    Equation: p_new = p_accept / (1 - cut * share).

    Assumptions: The price cut changes cost but not acceptance; raising acceptance
    leaves attempt cost unchanged.

    Excludes: Interactions between quality and price, and clipping an unattainable
    target to 1.

    References
    ----------
    Derivation: Equate C / p_new with C (1 - cut share) / p_accept and cancel C.

    Examples
    --------
    >>> calc_break_even_acceptance(0.5, cut=0.5, share=0.5) == 0.5 / 0.75
    True
    """
    return acceptance_probability / (1 - cut * share)


def calc_independent_success(acceptance_probability: float, attempts: int) -> float:
    """Calculate the chance that at least one independent attempt passes.

    Parameters
    ----------
    acceptance_probability : float
        Acceptance probability of each attempt in [0, 1].
    attempts : int
        Nonnegative count of independent attempts.

    Returns
    -------
    float
        Probability of at least one accepted attempt.

    Notes
    -----
    Equation: P_any = 1 - (1 - p_accept) ** k.

    Assumptions: Identical independent attempts.

    Excludes: Shared failure causes, search adaptation, and dependence among attempts.

    References
    ----------
    Derivation: Subtract the probability that every attempt fails from one.

    Examples
    --------
    >>> calc_independent_success(0.5, 2) == 1 - 0.5 ** 2
    True
    """
    return 1 - (1 - acceptance_probability) ** attempts


def calc_append_only_context(initial_tokens, tokens_per_turn, turns: int):
    """Count final and cumulative input tokens in an append-only trajectory.

    Parameters
    ----------
    initial_tokens : Quantity or int
        Nonnegative tokens in the first request.
    tokens_per_turn : Quantity or int
        Nonnegative tokens appended between successive requests, using the same units as
        initial_tokens.
    turns : int
        Positive number of requests in the trajectory.

    Returns
    -------
    tuple
        Last-request token count and cumulative input-token count, retaining input
        units.

    Notes
    -----
    Equation: last = s0 + delta (H - 1); total = H s0 + delta H (H - 1) / 2.

    Assumptions: Every request resends the full earlier context; append size is
    constant.

    Excludes: Prefix caching, compaction, request-window limits, and output-token
    billing.

    References
    ----------
    Derivation: Sum the arithmetic progression s0, s0 + delta, ..., s0 + (H - 1) delta.

    Examples
    --------
    >>> last, total = calc_append_only_context(10, 2, 3)
    >>> (last, total) == (10 + 2 * 2, sum([10, 12, 14]))
    True
    """
    last_call = initial_tokens + tokens_per_turn * (turns - 1)
    cumulative = turns * initial_tokens + tokens_per_turn * (turns * (turns - 1) // 2)
    return last_call, cumulative


# --- Model calls and agent trajectories (added 2026-10-20) -------------------
# Equations behind one model call (admission, sampling, a fallible check) and
# behind a trajectory of calls (horizon compounding, cost per accepted task, and
# resent history live above). Pure arithmetic on stated inputs; the solvers in
# engine/solvers/model_call.py compose them.

import math as _math


def calc_tempered_softmax(logits, temperature: float = 1.0) -> list:
    """Normalize temperature-scaled logits into sampling probabilities.

    Parameters
    ----------
    logits : iterable of float
        Nonempty sequence of dimensionless scores whose scaled exponentials are
        representable.
    temperature : float, optional
        Positive dimensionless softmax temperature; default 1 leaves the scores
        unchanged.

    Returns
    -------
    list of float
        Probabilities in input order, normalized to sum to one.

    Notes
    -----
    Equation: p_i = exp(z_i / T) / sum_j exp(z_j / T).

    Assumptions: Finite exponentials with a positive normalizer; this implementation
    exponentiates directly.

    Excludes: Log-sum-exp stabilization, top-p truncation, and predictions of output
    quality.

    References
    ----------
    Derivation: Normalize the positive exponential weights by their sum.

    Examples
    --------
    >>> calc_tempered_softmax([0.0, 0.0]) == [0.5, 0.5]
    True
    """
    exps = [_math.exp(z / temperature) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def calc_nucleus(probabilities, top_p: float):
    """Select and renormalize the top-p probability nucleus.

    Parameters
    ----------
    probabilities : iterable of float
        Nonnegative normalized token probabilities with positive total mass.
    top_p : float
        Target retained mass in (0, 1].

    Returns
    -------
    tuple
        Retained count, original retained mass, and normalized probabilities sorted from
        largest to smallest.

    Notes
    -----
    Equation: Keep the smallest prefix k with sum_{i=1}^k p_sorted_i >= top_p; return
    p_i / retained_mass.

    Assumptions: The input is a probability distribution and highest-probability tokens
    are retained first.

    Excludes: Token identities, tie-breaking identities, temperature scaling, and
    quality guarantees.

    References
    ----------
    Derivation: A descending cumulative sum finds the shortest prefix reaching the
    threshold; division renormalizes it.

    Examples
    --------
    >>> kept, mass, probs = calc_nucleus([0.5, 0.25, 0.25], 0.5)
    >>> (kept, mass, probs) == (1, 0.5, [1.0])
    True
    """
    ordered = sorted(probabilities, reverse=True)
    mass, kept = 0.0, 0
    for p in ordered:
        mass += p
        kept += 1
        if mass >= top_p:
            break
    return kept, mass, [p / mass for p in ordered[:kept]]


def calc_pass_posterior(defect_rate: float, false_accept_rate: float, valid_accept_rate: float) -> float:
    """Calculate the defective share of passing candidates.

    Parameters
    ----------
    defect_rate : float
        Prior defective fraction in [0, 1].
    false_accept_rate : float
        Probability of passing conditional on defect, in [0, 1].
    valid_accept_rate : float
        Probability of passing conditional on validity, in [0, 1]; total pass
        probability must be positive.

    Returns
    -------
    float
        Defective share among passing candidates.

    Notes
    -----
    Equation: P(defect | pass) = q f / (q f + (1 - q) v).

    Assumptions: Binary validity and fixed conditional checker rates.

    Excludes: Distribution shift, uncertainty in checker estimates, and zero-pass
    populations.

    References
    ----------
    Derivation: Bayes rule divides the joint defect-and-pass probability by total pass
    probability.

    Examples
    --------
    >>> calc_pass_posterior(0.5, 0.25, 0.75) == 0.25
    True
    """
    defective_pass = false_accept_rate * defect_rate
    return defective_pass / (defective_pass + valid_accept_rate * (1 - defect_rate))


def calc_reused_prefix_prompt_tokens(turns: int, first_prompt: int, appended: int) -> int:
    """Count newly prefetched tokens with full prefix reuse.

    Parameters
    ----------
    turns : int
        Positive number of requests.
    first_prompt : int
        Nonnegative input-token count of the cold first request.
    appended : int
        Nonnegative fresh tokens appended on each later request.

    Returns
    -------
    int
        Total newly prefetched prompt tokens across requests.

    Notes
    -----
    Equation: S_fresh = S0 + (H - 1) D.

    Assumptions: All earlier prefix KV state remains compatible and available at every
    subsequent call.

    Excludes: Cache eviction, prefix edits, lookup overhead, and generated output
    tokens.

    References
    ----------
    Derivation: Charge the initial prompt once and only the appended suffix on each
    remaining request.

    Examples
    --------
    >>> calc_reused_prefix_prompt_tokens(3, 10, 2) == 10 + 2 * 2
    True
    """
    return first_prompt + (turns - 1) * appended


# --- Test-time deliberation: pools, checks, and search (added 2026-10-20) ----
# The closed forms behind spending more work on one decision with the weights
# fixed: the unbiased pass@k estimate, the expected winner's error under
# argmax selection, and the size of a search tree that a step check prunes.
# The solvers in engine/solvers/deliberation.py compose them; pass@k for a
# known rate is calc_independent_success above, and passing-set precision is
# one minus calc_pass_posterior.


def calc_pass_at_k_unbiased(samples: int, passing: int, k: int) -> float:
    """Estimate the probability that a size-k sample contains a pass.

    Parameters
    ----------
    samples : int
        Positive count n of graded samples.
    passing : int
        Passing sample count c in [0, n].
    k : int
        Subset size in [1, n].

    Returns
    -------
    float
        Unbiased estimate of the probability at least one of k draws passes.

    Notes
    -----
    Equation: pass_at_k = 1 - C(n - c, k) / C(n, k), or 1 if n - c < k.

    Assumptions: Exchangeable independent draws from a fixed task distribution.

    Excludes: Dependence among samples, uncertainty intervals, and changing graders.

    References
    ----------
    Derivation: Count all-failing subsets among all size-k subsets and subtract their
    fraction from one; this is the combinatorial estimator described by Chen et al.
    (2021).

    Chen et al. (2021), "Evaluating Large Language Models Trained on Code."
    https://arxiv.org/abs/2107.03374 (bibliography key: chen2021evaluating).

    Examples
    --------
    >>> calc_pass_at_k_unbiased(4, 2, 1) == 0.5
    True
    """
    if samples - passing < k:
        return 1.0
    return 1 - _math.comb(samples - passing, k) / _math.comb(samples, k)


def calc_expected_max_std_normal(n: int, lo: float = -10.0, hi: float = 10.0, steps: int = 40_000) -> float:
    """Integrate the expected maximum of independent standard normals.

    Parameters
    ----------
    n : int
        Positive number of independent standard-normal draws.
    lo : float, optional
        Lower integration limit in standard deviations; default -10.
    hi : float, optional
        Upper integration limit greater than lo; default 10.
    steps : int, optional
        Positive trapezoid count; default 40000.

    Returns
    -------
    float
        Approximate expected maximum, in units of the normal standard deviation.

    Notes
    -----
    Equation: E[max] = integral x n phi(x) Phi(x) ** (n - 1) dx over [lo, hi].

    Assumptions: Independent standard-normal score errors and a finite uniform
    integration grid.

    Excludes: Tail mass outside the bounds, quadrature error, correlated errors, and
    unequal candidate means.

    References
    ----------
    Derivation: The maximum CDF is Phi(x) ** n; differentiating gives n phi(x) Phi(x) **
    (n - 1).

    Examples
    --------
    >>> abs(calc_expected_max_std_normal(1, steps=1000)) < 1e-10
    True
    """
    dx = (hi - lo) / steps
    total = 0.0
    for i in range(steps + 1):
        x = lo + i * dx
        pdf = _math.exp(-0.5 * x * x) / _math.sqrt(2 * _math.pi)
        cdf = 0.5 * (1 + _math.erf(x / _math.sqrt(2)))
        weight = 0.5 if i in (0, steps) else 1.0
        total += weight * x * n * pdf * cdf ** (n - 1)
    return total * dx


def calc_pruned_tree_nodes(branching: int, rejection_rate: float, depth: int):
    """Count proposed search nodes with and without independent pruning.

    Parameters
    ----------
    branching : int
        Nonnegative child count proposed per retained node.
    rejection_rate : float
        Independent rejection probability in [0, 1].
    depth : int
        Nonnegative number of child levels, excluding the root.

    Returns
    -------
    tuple
        Unpruned node count and expected proposed node count with pruning; neither
        includes the root.

    Notes
    -----
    Equation: unpruned = sum_d b ** d; pruned = sum_d b [b (1 - r)] ** (d - 1).

    Assumptions: Constant branching, independent rejection, and no expansion of rejected
    nodes.

    Excludes: Checker execution cost, adaptive branching, and variance in realized tree
    size.

    References
    ----------
    Derivation: At each level the expected retained-parent count grows by b (1 - r),
    then each parent proposes b children.

    Examples
    --------
    >>> full, pruned = calc_pruned_tree_nodes(2, 0.0, 3)
    >>> full == pruned
    True
    """
    surviving = branching * (1 - rejection_rate)
    unpruned, pruned = 0, 0.0
    for d in range(1, depth + 1):
        unpruned += branching**d
        pruned += branching * surviving ** (d - 1)
    return unpruned, pruned


def calc_setup_share(setup, work):
    """Calculate the setup share of serial call time.

    Parameters
    ----------
    setup : Quantity
        Nonnegative setup time, in any time unit.
    work : Quantity
        Nonnegative useful-work time; setup + work must be positive.

    Returns
    -------
    float
        Dimensionless share of elapsed time consumed by setup.

    Notes
    -----
    Equation: share = T_setup / (T_setup + T_work).

    Assumptions: Setup and useful work execute serially with no overlap.

    Excludes: Pipelining, shared setup across calls, and concurrent execution.

    References
    ----------
    Derivation: Partition total serial elapsed time into setup and useful-work
    components.

    Examples
    --------
    >>> from mlsysim.core.units import ureg
    >>> calc_setup_share(1 * ureg.second, 3 * ureg.second) == 0.25
    True
    """
    return float((setup / (setup + work)).to("").magnitude)


def calc_work_for_setup_cap(setup, cap: float):
    """Find useful work needed to cap the setup share.

    Parameters
    ----------
    setup : Quantity or float
        Nonnegative setup duration or cost.
    cap : float
        Maximum allowed setup share in (0, 1].

    Returns
    -------
    Quantity or float
        Minimum useful work, in the same units as setup.

    Notes
    -----
    Equation: W_min = T_setup (1 - phi) / phi.

    Assumptions: Setup and work add in the same accounting dimension.

    Excludes: Overlap, discrete work-item rounding, and changing setup cost.

    References
    ----------
    Derivation: Solve T_setup / (T_setup + W) <= phi for W.

    Examples
    --------
    >>> from mlsysim.core.units import ureg
    >>> calc_work_for_setup_cap(1 * ureg.second, 0.25) == 3 * ureg.second
    True
    """
    return setup * (1 - cap) / cap


# --- Injection exposure (added 2026-10-20) -----------------------------------
# How often a task meets attacker text: the complement of every untrusted read
# being clean. The same form counts turns when the per-turn chance is known.


def calc_injection_exposure(payload_fraction: float, reads: float) -> float:
    """Calculate exposure probability across independent untrusted reads.

    Parameters
    ----------
    payload_fraction : float
        Fraction of independently read items carrying a payload, in [0, 1].
    reads : float
        Nonnegative number of reads; whole counts describe discrete items.

    Returns
    -------
    float
        Probability of encountering at least one payload, not probability of compromise.

    Notes
    -----
    Equation: P_exposure = 1 - (1 - q) ** n.

    Assumptions: Identically distributed independent reads; q may include a specified
    screening miss fraction.

    Excludes: Attack success conditional on exposure, correlated items, and adaptive
    attackers.

    References
    ----------
    Derivation: Exposure is the complement of all n reads being payload-free.

    Examples
    --------
    >>> calc_injection_exposure(0.5, 2) == 1 - 0.5 ** 2
    True
    """
    return 1 - (1 - payload_fraction) ** reads


def calc_injection_read_budget(payload_fraction: float, target_exposure: float) -> int:
    """Find the largest read budget within an exposure limit.

    Parameters
    ----------
    payload_fraction : float
        Per-read payload probability strictly between 0 and 1.
    target_exposure : float
        Maximum exposure probability in [0, 1).

    Returns
    -------
    int
        Largest whole read count satisfying the modeled exposure limit.

    Notes
    -----
    Equation: n_max = floor(log(1 - target) / log(1 - q)).

    Assumptions: Independent identically distributed payload presence on each read.

    Excludes: Correlated exposure, compromise probability, and floating-point errors
    near integer boundaries.

    References
    ----------
    Derivation: Invert 1 - (1 - q) ** n <= target and take the integer floor.

    Examples
    --------
    >>> n = calc_injection_read_budget(0.5, 0.6)
    >>> calc_injection_exposure(0.5, n) <= 0.6 < calc_injection_exposure(0.5, n + 1)
    True
    """
    return _math.floor(_math.log(1 - target_exposure) / _math.log(1 - payload_fraction))


# --- The agent harness: admission, fair share, and the budget meter (added 2026-10-20) ---
# Closed forms the harness evaluates before it admits work: the most one model
# call can bill, the long-run call rate a weighted deficit round-robin gives a
# backlogged class, and an allowance's used fraction. The solvers in
# engine/solvers/harness.py compose them.


def calc_bounded_liability(input_tokens, output_allowance, input_price, output_price):
    """Calculate the largest token bill allowed for one model call.

    Parameters
    ----------
    input_tokens : Quantity or int
        Nonnegative number of input tokens to charge.
    output_allowance : Quantity or int
        Nonnegative maximum permitted output-token count.
    input_price : Quantity or float
        Undiscounted input price per token in the chosen currency.
    output_price : Quantity or float
        Output price per token in the same currency.

    Returns
    -------
    Quantity or float
        Maximum token bill under the stated rates and allowance.

    Notes
    -----
    Equation: L = n_input p_input + K_output p_output.

    Assumptions: Every input token is billed at its listed rate and output remains
    within its allowance.

    Excludes: Tool charges, taxes, dynamic pricing, and guaranteed cache discounts.

    References
    ----------
    Derivation: Add the input invoice and the largest permitted output invoice.

    Examples
    --------
    >>> calc_bounded_liability(10, 5, 2, 3) == 10 * 2 + 5 * 3
    True
    """
    return input_tokens * input_price + output_allowance * output_price


def calc_drr_calls_per_round(weight: float, quantum: float, charge: float) -> float:
    """Calculate the long-run call rate under deficit round-robin credit.

    Parameters
    ----------
    weight : float
        Nonnegative dimensionless class weight.
    quantum : float
        Nonnegative credit allocated per round, in charge units.
    charge : float
        Positive credit consumed by one call, in the same units as quantum.

    Returns
    -------
    float
        Long-run average admitted calls per scheduling round.

    Notes
    -----
    Equation: calls_per_round = weight * quantum / charge.

    Assumptions: A continuously backlogged class, fixed per-call charge, and unspent
    credit carried across rounds.

    Excludes: Transient integer admissions, variable call sizes, and latency guarantees.

    References
    ----------
    Derivation: Conservation of scheduler credit equates long-run credit earned and
    credit spent; this is weighted deficit-round-robin accounting.

    Shreedhar and Varghese (1995), "Efficient Fair Queuing Using Deficit Round Robin."
    https://doi.org/10.1145/217382.217453 (bibliography key: shreedhar1995efficient).

    Examples
    --------
    >>> calc_drr_calls_per_round(2, 3, 4) == 2 * 3 / 4
    True
    """
    return weight * quantum / charge


def calc_used_fraction(used, allowance) -> float:
    """Calculate the consumed fraction of one budget dimension.

    Parameters
    ----------
    used : Quantity or float
        Nonnegative consumed amount.
    allowance : Quantity or float
        Positive budget in units compatible with used.

    Returns
    -------
    float
        Dimensionless consumed fraction; may exceed one when over budget.

    Notes
    -----
    Equation: rho = used / allowance.

    Assumptions: The two inputs refer to the same additive budget dimension.

    Excludes: Budget renewal, refunds, and enforcing an admission policy.

    References
    ----------
    Derivation: Normalize the consumed amount by its allowance.

    Examples
    --------
    >>> from mlsysim.core.units import ureg
    >>> calc_used_fraction(2 * ureg.second, 4 * ureg.second) == 0.5
    True
    """
    ratio = used / allowance
    return float(ratio.to("").magnitude) if hasattr(ratio, "to") else float(ratio)


# --- Failure recovery: repair, irreversible loss, cleanup order (added 2026-10-20) ---
# Closed forms the runtime evaluates when an effect has failed or must be
# undone: the break-even success rate of one bounded repair attempt, the
# expected cost of trying repair first, the expected irreversible loss of a
# plan whose checks precede or follow its pivot, and the elapsed time of a
# cleanup graph. The solvers in engine/solvers/recovery.py compose them.


def calc_repair_break_even(repair_cost, fallback_cost):
    """Find the repair success rate that breaks even with fallback.

    Parameters
    ----------
    repair_cost : Quantity or float
        Nonnegative cost of one repair attempt.
    fallback_cost : Quantity or float
        Positive fallback cost in the same units as repair_cost.

    Returns
    -------
    float
        Repair success probability at equal expected cost; above one means repair cannot
        win on this axis.

    Notes
    -----
    Equation: p_star = C_repair / C_fallback.

    Assumptions: A failed repair leaves the fallback available at unchanged cost.

    Excludes: Irreversible damage, repair repetitions, and joint optimization across
    unlike units.

    References
    ----------
    Derivation: Equate C_repair + (1 - p) C_fallback with C_fallback, then solve for p.

    Examples
    --------
    >>> from mlsysim.core.units import ureg
    >>> calc_repair_break_even(1 * ureg.second, 4 * ureg.second) == 0.25
    True
    """
    ratio = repair_cost / fallback_cost
    return float(ratio.to("").magnitude) if hasattr(ratio, "to") else float(ratio)


def calc_repair_first_expected_cost(repair_cost, fallback_cost, success_probability: float):
    """Calculate expected cost when one repair precedes fallback.

    Parameters
    ----------
    repair_cost : Quantity or float
        Nonnegative cost paid for the single repair attempt.
    fallback_cost : Quantity or float
        Nonnegative fallback cost in units compatible with repair_cost.
    success_probability : float
        Repair success probability in [0, 1].

    Returns
    -------
    Quantity or float
        Expected total cost, in the shared input-cost unit.

    Notes
    -----
    Equation: E[C] = C_repair + (1 - p) C_fallback.

    Assumptions: Every attempt pays repair cost; only repair failure incurs fallback.

    Excludes: Repeated repairs, changed fallback cost, and consequences outside the
    chosen cost dimension.

    References
    ----------
    Derivation: Weight the successful and unsuccessful branches by p and 1 - p.

    Examples
    --------
    >>> calc_repair_first_expected_cost(1, 4, 0.5) == 1 + 0.5 * 4
    True
    """
    return repair_cost + (1 - success_probability) * fallback_cost


def calc_expected_irreversible_loss(defect_probability: float, consequence, recoverable_fraction: float):
    """Calculate expected loss remaining after recovery.

    Parameters
    ----------
    defect_probability : float
        Probability of a consequential defect in [0, 1].
    consequence : Quantity or float
        Nonnegative loss conditional on defect, in the chosen cost unit.
    recoverable_fraction : float
        Fraction of consequence avoided by recovery, in [0, 1].

    Returns
    -------
    Quantity or float
        Expected residual loss, in the consequence unit.

    Notes
    -----
    Equation: L = P_defect * consequence * (1 - r).

    Assumptions: One defect event and a specified mean recoverable share; an upstream
    check can supply r = detection * rollback fidelity.

    Excludes: Tail-risk bounds, repeated defects, and proof of recovery effectiveness.

    References
    ----------
    Derivation: Multiply the event probability by the unrecoverable portion of its
    conditional loss.

    Examples
    --------
    >>> calc_expected_irreversible_loss(0.5, 8, 0.25) == 0.5 * 8 * 0.75
    True
    """
    return defect_probability * consequence * (1 - recoverable_fraction)


def calc_cleanup_critical_path(durations, depends_on):
    """Calculate elapsed cleanup time under unlimited parallelism.

    Parameters
    ----------
    durations : mapping
        Nonempty operation-to-duration mapping with nonnegative values in compatible
        units.
    depends_on : mapping
        Operation-to-prerequisite iterable mapping; every prerequisite must occur in
        durations and the graph must be acyclic.

    Returns
    -------
    Quantity or float
        Longest-path elapsed time, retaining the duration unit.

    Notes
    -----
    Equation: finish(v) = duration(v) + max_{u in parents(v)} finish(u); elapsed = max_v
    finish(v).

    Assumptions: Unlimited parallel execution of ready operations; prerequisite
    settlement is mandatory.

    Excludes: Resource contention, scheduling overhead, retries, and cyclic dependency
    resolution.

    References
    ----------
    Derivation: A topological earliest-finish recurrence yields the critical path;
    serial rollback instead sums all durations.

    Examples
    --------
    >>> calc_cleanup_critical_path({'a': 2, 'b': 3, 'c': 1}, {'c': ['a', 'b']}) == max(2, 3) + 1
    True
    """
    finish = {}

    def _finish(op, trail=()):
        if op in trail:
            raise ValueError(f"cleanup graph has a cycle through {op!r}")
        if op not in finish:
            start = max((_finish(d, trail + (op,)) for d in depends_on.get(op, ())), default=0 * durations[op])
            finish[op] = start + durations[op]
        return finish[op]

    return max(_finish(op) for op in durations)


def calc_pass_pow_k_unbiased(samples: int, passing: int, k: int) -> float:
    """Estimate the chance that every run in a size-k sample passes.

    Parameters
    ----------
    samples : int
        Positive count n of graded runs.
    passing : int
        Passing run count c in [0, n].
    k : int
        Subset size in [1, n].

    Returns
    -------
    float
        Unbiased estimate of the probability that all k draws pass.

    Notes
    -----
    Equation: pass_pow_k = C(c, k) / C(n, k), or 0 if c < k.

    Assumptions: Exchangeable independent draws from a fixed task distribution.

    Excludes: Sample dependence, grader uncertainty, and confidence intervals.

    References
    ----------
    Derivation: Count all-passing subsets among all size-k subsets; its expectation
    under independent draws is p ** k.

    Yao et al. (2024), "tau-bench: A Benchmark for Tool-Agent-User Interaction in
    Real-World Domains."
    https://arxiv.org/abs/2406.12045 (bibliography key: yao2024taubench).

    Examples
    --------
    >>> calc_pass_pow_k_unbiased(4, 2, 1) == 0.5
    True
    """
    if passing < k:
        return 0.0
    return _math.comb(passing, k) / _math.comb(samples, k)


# --- Trajectory curation: split leakage, attempt-group contrast, ordered checks (added 2026-10-20) ---


def calc_split_straddle(train_fraction: float, attempts: int) -> float:
    """Calculate the chance that repeated task runs straddle a split.

    Parameters
    ----------
    train_fraction : float
        Independent probability that each run enters training, in [0, 1].
    attempts : int
        Positive number of runs of the same task.

    Returns
    -------
    float
        Probability that the task appears on both sides of the run-level split.

    Notes
    -----
    Equation: P_straddle = 1 - f ** k - (1 - f) ** k.

    Assumptions: Runs are split independently; task identity is shared across the runs.

    Excludes: Task-grouped splitting, stratification, and estimates of downstream
    evaluation bias.

    References
    ----------
    Derivation: The disjoint alternatives to straddling are all-training and
    all-held-out assignments.

    Examples
    --------
    >>> calc_split_straddle(0.5, 2) == 1 - 0.5 ** 2 - 0.5 ** 2
    True
    """
    f = train_fraction
    return 1 - f**attempts - (1 - f) ** attempts


def calc_mixed_outcome_fraction(pass_probability: float, attempts: int) -> float:
    """Calculate the probability that a group contains both outcomes.

    Parameters
    ----------
    pass_probability : float
        Per-attempt success probability in [0, 1].
    attempts : int
        Positive number of attempts.

    Returns
    -------
    float
        Probability of observing both a success and a failure.

    Notes
    -----
    Equation: P_mixed = 1 - (1 - p) ** k - p ** k.

    Assumptions: Independent attempts with identical pass probability.

    Excludes: Correlated attempts, reward magnitudes, and usefulness of the resulting
    training signal.

    References
    ----------
    Derivation: Exclude the disjoint all-failure and all-success groups from the full
    outcome space.

    Examples
    --------
    >>> calc_mixed_outcome_fraction(0.5, 2) == 0.5
    True
    """
    p = pass_probability
    return 1 - (1 - p) ** attempts - p**attempts


def calc_cost_per_rejection(check_cost, pass_probability: float):
    """Calculate checking cost per rejected record.

    Parameters
    ----------
    check_cost : Quantity or float
        Nonnegative per-record checking cost in any additive resource unit.
    pass_probability : float
        Check pass probability in [0, 1); rejection must have positive probability.

    Returns
    -------
    Quantity or float
        Expected checking cost per rejected record, in check_cost units.

    Notes
    -----
    Equation: C_rejection = C_check / (1 - p_pass).

    Assumptions: Identically distributed records and fixed per-record check cost.

    Excludes: Dependence between checks and selectivity changes after earlier checks.

    References
    ----------
    Derivation: A geometric number of checked records is required for each rejection;
    adjacent independent checks favor lower cost per rejection.

    Examples
    --------
    >>> calc_cost_per_rejection(2, 0.5) == 4
    True
    """
    return check_cost / (1 - pass_probability)


def calc_short_circuit_cost(pass_probabilities, check_costs):
    """Calculate the cost and survival share of ordered checks.

    Parameters
    ----------
    pass_probabilities : sequence of float
        Pass probabilities in [0, 1], ordered by execution.
    check_costs : sequence of Quantity or float
        Nonempty nonnegative finite costs in compatible units, with the same
        length as pass_probabilities.

    Returns
    -------
    tuple
        Expected checking cost in the shared cost unit and dimensionless all-pass
        probability.

    Notes
    -----
    Equation: E[C] = sum_i c_i product_{j<i} p_j; P_pass = product_i p_i.

    Assumptions: Each check has fixed conditional pass probability given earlier passes;
    independent checks may use their marginal probabilities.

    Excludes: Reordering optimization and input-dependent costs. Inputs must
    contain the same nonzero number of checks.

    References
    ----------
    Derivation: A check incurs cost only on the fraction of records reaching it, found
    by multiplying preceding conditional pass rates.

    Examples
    --------
    >>> cost, passing = calc_short_circuit_cost([0.5, 0.5], [2, 4])
    >>> (cost, passing) == (2 + 0.5 * 4, 0.5 * 0.5)
    True
    """
    if len(pass_probabilities) != len(check_costs):
        raise ValueError("pass_probabilities and check_costs must have the same length")
    if len(check_costs) == 0:
        raise ValueError("at least one check is required")
    for index, (probability, check_cost) in enumerate(zip(pass_probabilities, check_costs)):
        validate_range(probability, 0, 1, f"pass_probabilities[{index}]")
        validate_nonnegative(check_cost, f"check_costs[{index}]")
    reach = 1.0
    cost = 0 * check_costs[0]
    for p, t in zip(pass_probabilities, check_costs):
        cost = cost + reach * t
        reach *= p
    return cost, reach
