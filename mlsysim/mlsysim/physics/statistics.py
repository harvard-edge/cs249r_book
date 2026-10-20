"""Statistical and workflow propagation helpers."""

from __future__ import annotations

import math


def calc_population_stability_index(expected, actual, epsilon=1e-12):
    """Population Stability Index (PSI) between two aligned distributions.

    Measures distribution drift between a reference ("expected") and an
    observed ("actual") binned distribution:
    ``PSI = sum_i (actual_i - expected_i) * ln(actual_i / expected_i)``
    (the symmetrized KL divergence). Interpretation thresholds live in
    ``Ops.Monitoring`` (the single source of truth the book follows:
    0.10 warn / 0.20 review / 0.25 critical) — do not restate bands here.

    Parameters
    ----------
    expected, actual : sequence of float
        Bin probabilities (dimensionless, each summing to ~1), aligned
        bin-for-bin and of equal length.
    epsilon : float
        Floor applied to every bin to avoid log(0) / division by zero when a
        bin is empty.

    Returns
    -------
    float
        PSI value (dimensionless, >= 0; 0 means identical distributions).
    """
    if len(expected) != len(actual):
        raise ValueError("expected and actual distributions must have the same length")

    total = 0.0
    for exp, act in zip(expected, actual):
        exp = max(float(exp), epsilon)
        act = max(float(act), epsilon)
        total += (act - exp) * math.log(act / exp)
    return total


def calc_two_proportion_sample_size(baseline_rate, detectable_lift, z_alpha=1.96, z_beta=0.84, pooled=False):
    """Per-arm sample size for a two-proportion A/B test.

    Standard normal-approximation power formula:
    ``n = 2 * (z_alpha + z_beta)^2 * p(1-p) / delta^2``, using the baseline
    rate's variance for both arms (a common simplification valid for small
    lifts). Defaults correspond to a two-sided 5% significance level
    (z = 1.96) and 80% power (z = 0.84).

    Parameters
    ----------
    baseline_rate : float
        Control-arm conversion rate p in (0, 1).
    detectable_lift : float
        Minimum absolute difference in rates to detect (same 0-1 scale as
        ``baseline_rate``, e.g. 0.01 for one percentage point). Must be > 0.
    z_alpha : float
        Normal quantile for the significance level (1.96 = two-sided 0.05).
    z_beta : float
        Normal quantile for power (0.84 = 80% power).
    pooled : bool
        When True, use the pooled two-proportion form, which carries the
        variance of both arms rather than reusing the baseline's for each:

        ``n = (z_alpha * sqrt(2 * p_bar * q_bar)
               + z_beta * sqrt(p1*q1 + p2*q2))^2 / delta^2``

        where ``p_bar`` is the mean of the two rates. This is the textbook
        form and the one the book's own appendix displays. It is not the
        default because the simplified form above is already in use and its
        results are quoted in print; pass ``pooled=True`` where the pooled
        form is the one being taught.

    Returns
    -------
    float
        Required sample size per arm (unrounded; callers should ceil).
    """
    p = float(baseline_rate)
    delta = float(detectable_lift)
    if not 0.0 < p < 1.0:
        raise ValueError(f"baseline_rate must be in (0, 1), got {p}")
    if delta <= 0.0:
        raise ValueError(f"detectable_lift must be > 0, got {delta}")
    if pooled:
        p2 = p + delta
        if not 0.0 < p2 < 1.0:
            raise ValueError(
                f"baseline_rate + detectable_lift must stay in (0, 1), got {p2}"
            )
        p_bar = (p + p2) / 2.0
        numerator = (
            float(z_alpha) * math.sqrt(2 * p_bar * (1 - p_bar))
            + float(z_beta) * math.sqrt(p * (1 - p) + p2 * (1 - p2))
        ) ** 2
        return numerator / delta ** 2
    variance = p * (1 - p)
    return 2 * (float(z_alpha) + float(z_beta)) ** 2 * variance / delta ** 2


# ── Rare-event estimation: plain Monte Carlo and importance sampling ────────
def calc_monte_carlo_sample_size(failure_probability, relative_error):
    """Plain Monte Carlo draws needed to estimate a probability to a relative error.

    Equation:
        N = (1 - p) / (p * r^2)

    The binomial estimator p_hat = k/N has relative standard error
    sqrt((1 - p) / (N p)); solving for N at a target ``r`` gives the count.

    Source: Owen, *Monte Carlo Theory, Methods and Examples* (2013): the
    variance of a sample mean of Bernoulli draws, as used for rare events in
    ch. 9.
    """
    if not (0.0 < failure_probability < 1.0):
        raise ValueError("failure_probability must be strictly between 0 and 1.")
    if relative_error <= 0.0:
        raise ValueError("relative_error must be positive.")
    return (1.0 - failure_probability) / (failure_probability * relative_error**2)


def calc_importance_sampling_exponential(threshold, operational_mean, proposal_mean):
    """Importance-sampling statistics for an exponential tail with an exponential proposal.

    Operational model p = Exp(mean m0), failure when D > gamma, proposal
    q = Exp(mean m_q) with likelihood ratio w = p/q. Writing lam = 1/m0 and
    eta = 1/m_q (eta < 2 lam so the second moment is finite):

        P_F                 = exp(-lam gamma)
        E_q[w^2 1{D>gamma}] = lam^2 / (eta (2 lam - eta)) * exp(-(2 lam - eta) gamma)
        relative variance   = E_q[w^2 1{D>gamma}] / P_F^2 - 1      (per draw)
        ESS per draw        = E_q[w]^2 / E_q[w^2] = eta (2 lam - eta) / lam^2
        P_q(D > gamma)      = exp(-eta gamma)

    Draws for a relative standard error r are ``relative_variance / r**2``.
    ``threshold``, ``operational_mean``, and ``proposal_mean`` are Pint
    quantities of one dimension (or plain floats in one unit).

    Source: likelihood-ratio estimator, its second moment, and the effective
    sample size (sum w)^2 / sum w^2 from Owen 2013, ch. 9; the exponential
    closed forms follow by integrating those expressions.
    """
    gamma, m0, mq = (float(getattr(x, "magnitude", x)) for x in _same_units(threshold, operational_mean, proposal_mean))
    if gamma <= 0 or m0 <= 0 or mq <= 0:
        raise ValueError("threshold and both means must be positive.")
    lam, eta = 1.0 / m0, 1.0 / mq
    if not eta < 2 * lam:
        raise ValueError("proposal_mean must exceed half the operational mean (finite second moment).")
    p_fail = math.exp(-lam * gamma)
    second_moment = lam**2 / (eta * (2 * lam - eta)) * math.exp(-(2 * lam - eta) * gamma)
    return {
        "failure_probability": p_fail,
        "second_moment": second_moment,
        "relative_variance": second_moment / p_fail**2 - 1.0,
        "ess_per_draw": eta * (2 * lam - eta) / lam**2,
        "proposal_failure_share": math.exp(-eta * gamma),
    }


def calc_optimal_exponential_proposal_mean(threshold, operational_mean):
    """Mean of the exponential proposal that minimizes importance-sampling variance.

    Minimizing the second moment of ``calc_importance_sampling_exponential``
    over eta gives the root below 2 lam of

        gamma eta^2 - (2 + 2 lam gamma) eta + 2 lam = 0,

    and the proposal mean is 1/eta. Returned in the units of ``threshold``.

    Source: first-order condition on the exponential second moment (Owen
    2013, ch. 9, exponential tilting of a light-tailed tail).
    """
    t_q, m_q = _same_units(threshold, operational_mean)
    gamma, m0 = float(getattr(t_q, "magnitude", t_q)), float(getattr(m_q, "magnitude", m_q))
    if gamma <= 0 or m0 <= 0:
        raise ValueError("threshold and operational_mean must be positive.")
    lam = 1.0 / m0
    b = 2.0 + 2.0 * lam * gamma
    eta = (b - math.sqrt(b**2 - 8.0 * lam * gamma)) / (2.0 * gamma)
    mean = 1.0 / eta
    units = getattr(t_q, "units", None)
    return mean * units if units is not None else mean


def calc_cross_entropy_exponential_proposal(threshold, operational_mean, elite_share):
    """Expected (large-sample) cross-entropy iteration for an exponential tail.

    From q_0 = p = Exp(m0), each round sets the level to the (1 - beta)
    quantile of the current proposal, level = -m_k ln(beta) (capped at the
    threshold), and refits the proposal mean to the likelihood-weighted
    elite draws; for this family that is E_p[D | D > level] = level + m0.
    Stops when the level reaches the threshold, so the final mean is
    threshold + m0.

    Returns a dict with ``levels`` (list, in the units of ``threshold``),
    ``n_levels``, and ``proposal_mean``.

    Source: Rubinstein and Kroese, *The Cross-Entropy Method* (Springer, 2004),
    multilevel rare-event algorithm; the exponential refit is in closed form.
    """
    if not (0.0 < elite_share < 1.0):
        raise ValueError("elite_share must be strictly between 0 and 1.")
    t_q, m_q = _same_units(threshold, operational_mean)
    gamma, m0 = float(getattr(t_q, "magnitude", t_q)), float(getattr(m_q, "magnitude", m_q))
    if gamma <= 0 or m0 <= 0:
        raise ValueError("threshold and operational_mean must be positive.")
    units = getattr(t_q, "units", None)
    m_k, levels = m0, []
    while True:
        level = min(gamma, -m_k * math.log(elite_share))
        levels.append(level)
        m_k = level + m0
        if level >= gamma:
            break
    wrap = (lambda x: x * units) if units is not None else (lambda x: x)
    return {"levels": [wrap(v) for v in levels], "n_levels": len(levels), "proposal_mean": wrap(m_k)}


def _same_units(*values):
    """Express Pint quantities in the first value's units; pass floats through."""
    first = values[0]
    units = getattr(first, "units", None)
    if units is None:
        return values
    return tuple(v.to(units) for v in values)


# ── Confidence in a structured safety argument ──────────────────────────────
def calc_conjunctive_confidence(premise_confidences):
    """Confidence in a claim that needs every one of several independent premises.

    Equation:
        P(claim) = prod_i p_i

    The conjunction is no stronger than its weakest premise and, with more
    than one premise below 1, strictly weaker.

    Source: probability of the intersection of independent events, as applied
    to conjunctive assurance arguments in Bloomfield and Littlewood 2003,
    "Multi-legged arguments: the impact of diversity upon confidence in
    dependability arguments" (DSN 2003).
    """
    ps = list(premise_confidences)
    if not ps or any(not (0.0 <= p <= 1.0) for p in ps):
        raise ValueError("premise_confidences must be a non-empty sequence of values in [0, 1].")
    return math.prod(ps)


def calc_multilegged_confidence(leg_confidences):
    """Confidence in a claim supported by independent legs, any one of which suffices.

    Equation:
        P(claim) = 1 - prod_i (1 - p_i)

    Holds only if the legs fail independently; common-cause failure collapses
    the legs back into a conjunction.

    Source: Bloomfield and Littlewood 2003 (DSN), multi-legged arguments under
    the independence assumption.
    """
    ps = list(leg_confidences)
    if not ps or any(not (0.0 <= p <= 1.0) for p in ps):
        raise ValueError("leg_confidences must be a non-empty sequence of values in [0, 1].")
    return 1.0 - math.prod(1.0 - p for p in ps)


# --- Exponential growth fit (added 2026-10-20) -------------------------------
def fit_exponential_growth(xs, ys, period=1.0):
    """Least-squares fit of ``log10(y)`` on ``x``; growth factor per ``period`` of x.

    Returns ``(factor, slope, intercept)`` where ``log10(y) = intercept + slope * x``
    and ``factor = 10 ** (slope * period)``. Trend lines on log-scale technology
    plots (compute, memory bandwidth, interconnect against year) are this fit.
    """
    n = len(xs)
    if n < 2 or n != len(ys):
        raise ValueError("fit_exponential_growth needs two or more (x, y) pairs.")
    ls = [math.log10(y) for y in ys]
    mx, my = sum(xs) / n, sum(ls) / n
    slope = (sum((x - mx) * (l - my) for x, l in zip(xs, ls))
             / sum((x - mx) ** 2 for x in xs))
    return 10 ** (slope * period), slope, my - slope * mx


# --- Inter-annotator agreement (added 2026-10-20) ----------------------------
def calc_chance_agreement(class_shares):
    """Agreement two independent annotators reach by chance: ``sum_k s_k ** 2``.

    ``class_shares`` is each annotator's label mix (fractions summing to 1);
    both annotators are assumed to share it.
    """
    shares = list(class_shares)
    if not shares or abs(sum(shares) - 1.0) > 1e-9:
        raise ValueError("class_shares must be fractions that sum to 1.")
    return sum(s * s for s in shares)


def calc_cohens_kappa(observed_agreement, chance_agreement):
    """Cohen's kappa: agreement beyond chance, ``(p_o - p_e) / (1 - p_e)`` (Cohen 1960)."""
    if not 0.0 <= chance_agreement < 1.0:
        raise ValueError("chance_agreement must be in [0, 1).")
    return (observed_agreement - chance_agreement) / (1.0 - chance_agreement)


# --- Two-sample drift statistics (added 2026-10-20) ---------------------------
def calc_ks_critical_value(coefficient, n_reference, n_current):
    """Large-sample two-sample Kolmogorov-Smirnov critical value.

    ``D_crit = c * sqrt((n_0 + n_t) / (n_0 * n_t))``, where ``c`` is the
    coefficient for the chosen significance level (1.36 at 0.05,
    ``Ops.Monitoring.KsTestCoefficient``).
    """
    if n_reference <= 0 or n_current <= 0:
        raise ValueError("sample sizes must be positive.")
    return coefficient * math.sqrt((n_reference + n_current) / (n_reference * n_current))


def calc_ks_normal_location_shift(shift_sd):
    """K-S distance between two equal-variance normals whose means differ by ``shift_sd`` sds.

    The CDF gap is largest midway between the means: ``2 * Phi(shift_sd / 2) - 1``,
    which equals ``erf(shift_sd / (2 * sqrt(2)))``.
    """
    return math.erf(abs(shift_sd) / (2 * math.sqrt(2)))
# --- Correlated samples and imitation-learning cost bounds (added 2026-10-20) --
def calc_effective_sample_size(n_records, lag1_correlation):
    """Effective number of independent samples in ``n_records`` of an AR(1) series.

    ``N_eff = N (1 - rho) / (1 + rho)`` for estimating the mean of a series whose
    adjacent records have lag-one correlation ``rho`` (Thiebaux and Zwiers 1984).
    """
    if not -1.0 < lag1_correlation < 1.0:
        raise ValueError("lag1_correlation must be in (-1, 1).")
    return n_records * (1.0 - lag1_correlation) / (1.0 + lag1_correlation)


def calc_behavior_cloning_cost_bound(horizon_steps, per_step_error):
    """Worst-case excess cost of behavioral cloning over ``horizon_steps``: ``S^2 eps``.

    Ross and Bagnell (2010), Theorem 2.1: a policy trained only on the
    demonstrator's states compounds a per-step error ``eps`` quadratically.
    """
    return horizon_steps * horizon_steps * per_step_error


def calc_dagger_cost_bound(horizon_steps, per_step_error, recoverability=1.0):
    """Worst-case excess cost with expert labels on learner states: ``u S eps``.

    Ross, Gordon, and Bagnell (2011), Theorem 3.2: linear in the horizon while
    the recoverability constant ``u`` stays bounded.
    """
    return recoverability * horizon_steps * per_step_error
