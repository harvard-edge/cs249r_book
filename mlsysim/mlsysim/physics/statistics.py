"""Statistical and workflow propagation helpers."""

from __future__ import annotations

import math


def calc_population_stability_index(expected, actual, epsilon=1e-12):
    """Measure drift between two aligned binned distributions.

    Parameters
    ----------
    expected, actual : sequence of float
        Aligned, equal-length nonnegative bin probabilities, each summing to one.
    epsilon : float, optional
        Positive dimensionless probability floor; default 1e-12 avoids log of zero.

    Returns
    -------
    float
        Dimensionless symmetrized divergence after flooring; zero for identical bins.

    Notes
    -----
    Equation: PSI = sum((a_i - e_i) * log(a_i / e_i)), after flooring each probability.

    Assumptions: Both distributions use identical bins; flooring does not renormalize them.

    Excludes: Bin selection, uncertainty intervals, and automatic drift-threshold decisions.

    Measures distribution drift between a reference ("expected") and an
    observed ("actual") binned distribution:
    ``PSI = sum_i (actual_i - expected_i) * ln(actual_i / expected_i)``
    (the symmetrized KL divergence). Interpretation thresholds live in
    ``Ops.Monitoring`` (the single source of truth the book follows:
    0.10 warn / 0.20 review / 0.25 critical) — do not restate bands here.

    References
    ----------
    Derivation: adding KL(actual || expected) and KL(expected || actual) yields the stated sum; flooring is a numerical convention.

    Examples
    --------
    >>> calc_population_stability_index([0.5, 0.5], [0.5, 0.5])
    0.0
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
    """Estimate per-arm sample size for a two-proportion comparison.

    Parameters
    ----------
    baseline_rate : float
        Control probability p in (0, 1).
    detectable_lift : float
        Positive absolute probability difference delta; pooled mode also requires p + delta < 1.
    z_alpha : float, optional
        Positive normal significance quantile; default 1.96 is the usual rounded two-sided 5-percent value.
    z_beta : float, optional
        Normal power quantile; default 0.84 is the usual rounded 80-percent value.
    pooled : bool, optional
        Use both arm variances when True; default False reuses baseline variance.

    Returns
    -------
    float
        Unrounded sample count per arm; round upward before planning collection.

    Notes
    -----
    Equation: n = ``2*(z_alpha+z_beta)^2*p*(1-p)/delta^2`` by default. Pooled mode uses ``(z_alpha*sqrt(2*p_bar*(1-p_bar))`` + ``z_beta*sqrt(p*(1-p)+(p+delta)*(1-p-delta)))^2/delta^2``.

    Assumptions: Independent Bernoulli outcomes, equal allocation, and a normal approximation.

    Excludes: Sequential monitoring, multiple-testing adjustment, clustering, and exact small-sample power.

    Standard normal-approximation power formula:
    ``n = 2 * (z_alpha + z_beta)^2 * p(1-p) / delta^2``, using the baseline
    rate's variance for both arms (a common simplification valid for small
    lifts). Defaults correspond to a two-sided 5% significance level
    (z = 1.96) and 80% power (z = 0.84).

    References
    ----------
    Derivation: a normal critical separation plus the desired power separation is equated to the mean difference; solving the variance-of-two-means expression gives n.

    Examples
    --------
    >>> small = calc_two_proportion_sample_size(0.5, 0.02)
    >>> large = calc_two_proportion_sample_size(0.5, 0.01)
    >>> large == small * 4
    True
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
            raise ValueError(f"baseline_rate + detectable_lift must stay in (0, 1), got {p2}")
        p_bar = (p + p2) / 2.0
        numerator = (
            float(z_alpha) * math.sqrt(2 * p_bar * (1 - p_bar)) + float(z_beta) * math.sqrt(p * (1 - p) + p2 * (1 - p2))
        ) ** 2
        return numerator / delta**2
    variance = p * (1 - p)
    return 2 * (float(z_alpha) + float(z_beta)) ** 2 * variance / delta**2


# ── Rare-event estimation: plain Monte Carlo and importance sampling ────────
def calc_monte_carlo_sample_size(failure_probability, relative_error):
    """Estimate independent draws for a relative standard-error target.

    Parameters
    ----------
    failure_probability : float
        Bernoulli failure probability p in (0, 1), dimensionless.
    relative_error : float
        Positive target relative standard error r, dimensionless.

    Returns
    -------
    float
        Unrounded independent draw count, not a confidence-interval coverage guarantee.

    Notes
    -----
    Equation: N = ``(1-p)/(p*r^2)``.

    Assumptions: Independent Bernoulli trials with known planning probability p.

    Excludes: Confidence quantiles, correlated draws, and model misspecification.

    Equation:
        N = (1 - p) / (p * r^2)

    The binomial estimator p_hat = k/N has relative standard error
    sqrt((1 - p) / (N p)); solving for N at a target ``r`` gives the count.

    Source: Owen, *Monte Carlo Theory, Methods and Examples* (2013): the
    variance of a sample mean of Bernoulli draws, as used for rare events in
    ch. 9.

    References
    ----------
    Derivation: Var(p_hat) = ``p*(1-p)/N``; divide its square root by p and solve for N. Owen (2013), Monte Carlo Theory, Methods and Examples, gives the sample-mean framework.

    Examples
    --------
    >>> calc_monte_carlo_sample_size(0.5, 0.5)
    4.0
    """
    if not (0.0 < failure_probability < 1.0):
        raise ValueError("failure_probability must be strictly between 0 and 1.")
    if relative_error <= 0.0:
        raise ValueError("relative_error must be positive.")
    return (1.0 - failure_probability) / (failure_probability * relative_error**2)


def calc_importance_sampling_exponential(threshold, operational_mean, proposal_mean):
    """Compute exponential-tail importance-sampling moments.

    Parameters
    ----------
    threshold : float or Quantity
        Positive failure threshold gamma; same dimension as both means.
    operational_mean : float or Quantity
        Positive exponential mean m0, in threshold units.
    proposal_mean : float or Quantity
        Proposal mean mq > m0/2; plain floats must use a common unit.

    Returns
    -------
    dict of str to float
        Dimensionless failure_probability, second_moment, relative_variance, ess_per_draw, and proposal_failure_share.

    Notes
    -----
    Equation: For lambda=1/m0 and eta=1/mq, P=``exp(-lambda*gamma)`` and M2=``lambda^2*exp(-(2*lambda-eta)*gamma)/(eta*(2*lambda-eta))``; relative_variance=M2/P^2-1.

    Assumptions: Exponential operational and proposal densities; likelihood-ratio weighting and finite second moment.

    Excludes: Random draws, parameter fitting, proposal adaptation, and non-exponential tail behavior.

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

    References
    ----------
    Derivation: integrate ``(p(x)/q(x))^2*q(x)`` from gamma to infinity. Integrating from zero gives the weight second moment used for ess_per_draw.

    Examples
    --------
    >>> result = calc_importance_sampling_exponential(1.0, 1.0, 1.0)
    >>> result["ess_per_draw"]
    1.0
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
    """Minimize the exponential importance-sampling second moment.

    Parameters
    ----------
    threshold : float or Quantity
        Positive tail threshold gamma.
    operational_mean : float or Quantity
        Positive operational exponential mean, in threshold units.

    Returns
    -------
    float or Quantity
        Optimal exponential proposal mean in threshold units.

    Notes
    -----
    Equation: With lambda=1/m0, solve ``gamma*eta^2-(2+2*lambda*gamma)*eta+2*lambda``=0 for the smaller root, then return 1/eta.

    Assumptions: The proposal family is exponential and minimizes the same tail-estimator second moment as calc_importance_sampling_exponential.

    Excludes: Optimization over other proposal families and finite-sample adaptation costs.

    Minimizing the second moment of ``calc_importance_sampling_exponential``
    over eta gives the root below 2 lam of

        gamma eta^2 - (2 + 2 lam gamma) eta + 2 lam = 0,

    and the proposal mean is 1/eta. Returned in the units of ``threshold``.

    Source: first-order condition on the exponential second moment (Owen
    2013, ch. 9, exponential tilting of a light-tailed tail).

    References
    ----------
    Derivation: differentiate the logarithm of the exponential second moment with respect to proposal rate eta and solve the resulting quadratic.

    Examples
    --------
    >>> proposal = calc_optimal_exponential_proposal_mean(1.0, 1.0)
    >>> proposal > 1.0
    True
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
    """Trace population cross-entropy updates for an exponential tail.

    Parameters
    ----------
    threshold : float or Quantity
        Positive target failure threshold gamma.
    operational_mean : float or Quantity
        Positive operational mean m0, in threshold units.
    elite_share : float
        Elite proposal fraction beta in (0, 1).

    Returns
    -------
    dict
        Levels contains thresholds in input units, n_levels is their count, and proposal_mean is the final mean in input units.

    Notes
    -----
    Equation: level_k = min(gamma, ``-m_k*log(beta))``; m_(k+1) = level_k + m0, until level_k reaches gamma.

    Assumptions: Population expectations replace empirical elites; the initial proposal mean equals the operational mean.

    Excludes: Sampling variability, weight degeneracy, finite-sample stopping, and non-exponential families.

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

    References
    ----------
    Derivation: exponential memorylessness gives E[X | X > level] = level + m0; the proposal quantile supplies the next level. Rubinstein and Kroese (2004), The Cross-Entropy Method, motivates multilevel adaptation.

    Examples
    --------
    >>> result = calc_cross_entropy_exponential_proposal(2.0, 1.0, 0.5)
    >>> result["levels"][-1] == 2.0 and result["proposal_mean"] == 3.0
    True
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
    """Combine independent premise probabilities by conjunction.

    Parameters
    ----------
    premise_confidences : sequence of float
        Nonempty probabilities p_i in [0, 1].

    Returns
    -------
    float
        Dimensionless probability that every premise holds.

    Notes
    -----
    Equation: P(all premises) = product(p_i).

    Assumptions: Independent premise events, each necessary for the conclusion.

    Excludes: Common-cause dependence and validation of supplied confidence judgments.

    Equation:
        P(claim) = prod_i p_i

    The conjunction is no stronger than its weakest premise and, with more
    than one premise below 1, strictly weaker.

    Source: probability of the intersection of independent events, as applied
    to conjunctive assurance arguments in Bloomfield and Littlewood 2003,
    "Multi-legged arguments: the impact of diversity upon confidence in
    dependability arguments" (DSN 2003).

    References
    ----------
    Derivation: repeated multiplication of independent-event intersection probabilities.

    Examples
    --------
    >>> calc_conjunctive_confidence([0.5, 0.5])
    0.25
    """
    ps = list(premise_confidences)
    if not ps or any(not (0.0 <= p <= 1.0) for p in ps):
        raise ValueError("premise_confidences must be a non-empty sequence of values in [0, 1].")
    return math.prod(ps)


def calc_multilegged_confidence(leg_confidences):
    """Combine independent supporting legs by disjunction.

    Parameters
    ----------
    leg_confidences : sequence of float
        Nonempty probabilities p_i in [0, 1].

    Returns
    -------
    float
        Dimensionless probability that at least one sufficient leg holds.

    Notes
    -----
    Equation: P(any leg) = 1-product(1-p_i).

    Assumptions: Each leg suffices alone and their failures are independent.

    Excludes: Common-cause dependence and evidential calibration of the input probabilities.

    Equation:
        P(claim) = 1 - prod_i (1 - p_i)

    Holds only if the legs fail independently; common-cause dependence
    requires a joint probability model instead of this product.

    Source: Bloomfield and Littlewood 2003 (DSN), multi-legged arguments under
    the independence assumption.

    References
    ----------
    Derivation: all legs fail with the product of their failure probabilities; take its complement.

    Examples
    --------
    >>> calc_multilegged_confidence([0.5, 0.5])
    0.75
    """
    ps = list(leg_confidences)
    if not ps or any(not (0.0 <= p <= 1.0) for p in ps):
        raise ValueError("leg_confidences must be a non-empty sequence of values in [0, 1].")
    return 1.0 - math.prod(1.0 - p for p in ps)


# --- Exponential growth fit (added 2026-10-20) -------------------------------
def fit_exponential_growth(xs, ys, period=1.0):
    """Fit a constant multiplicative growth trend in log space.

    Parameters
    ----------
    xs : sequence of float
        At least two predictor coordinates in a common unit, with nonzero spread.
    ys : sequence of float
        Positive observations, equal in length to xs and expressed in a common unit.
    period : float, optional
        Interval in x units for reporting growth; default 1.0.

    Returns
    -------
    tuple of float
        Multiplicative factor per period, log10 slope per x unit, and log10 intercept.

    Notes
    -----
    Equation: Fit log10(y)=``intercept+slope*x`` by ordinary least squares; factor=``10**(slope*period)``.

    Assumptions: Positive responses and a linear trend in logarithmic response; all points receive equal weight.

    Excludes: Uncertainty intervals, heteroskedastic weights, structural breaks, and causal interpretation.

    Returns ``(factor, slope, intercept)`` where ``log10(y) = intercept + slope * x``
    and ``factor = 10 ** (slope * period)``. Trend lines on log-scale technology
    plots (compute, memory bandwidth, interconnect against year) are this fit.

    References
    ----------
    Derivation: least-squares normal equations yield slope=Cov(x,log10(y))/Var(x), then exponentiation converts additive log growth to a multiplier.

    Examples
    --------
    >>> fit_exponential_growth([0.0, 1.0], [1.0, 10.0])
    (10.0, 1.0, 0.0)
    """
    n = len(xs)
    if n < 2 or n != len(ys):
        raise ValueError("fit_exponential_growth needs two or more (x, y) pairs.")
    ls = [math.log10(y) for y in ys]
    mx, my = sum(xs) / n, sum(ls) / n
    slope = sum((x - mx) * (l - my) for x, l in zip(xs, ls)) / sum((x - mx) ** 2 for x in xs)
    return 10 ** (slope * period), slope, my - slope * mx


# --- Inter-annotator agreement (added 2026-10-20) ----------------------------
def calc_chance_agreement(class_shares):
    """Compute chance agreement from a shared label distribution.

    Parameters
    ----------
    class_shares : sequence of float
        Nonempty nonnegative class probabilities summing to one.

    Returns
    -------
    float
        Dimensionless independent chance-agreement probability.

    Notes
    -----
    Equation: p_e = ``sum(s_k**2)``.

    Assumptions: Annotators independently sample labels from the same marginal distribution.

    Excludes: Distinct annotator marginals, label dependence, and estimation uncertainty.

    ``class_shares`` is each annotator's label mix (fractions summing to 1);
    both annotators are assumed to share it.

    References
    ----------
    Derivation: matching on class k has probability ``s_k*s_k``; sum mutually exclusive matches.

    Examples
    --------
    >>> calc_chance_agreement([0.5, 0.5])
    0.5
    """
    shares = list(class_shares)
    if not shares or abs(sum(shares) - 1.0) > 1e-9:
        raise ValueError("class_shares must be fractions that sum to 1.")
    return sum(s * s for s in shares)


def calc_cohens_kappa(observed_agreement, chance_agreement):
    """Normalize observed agreement relative to chance agreement.

    Parameters
    ----------
    observed_agreement : float
        Observed matching-label fraction in [0, 1].
    chance_agreement : float
        Chance agreement probability in [0, 1).

    Returns
    -------
    float
        Dimensionless kappa; negative values indicate agreement below chance.

    Notes
    -----
    Equation: kappa = (p_o-p_e)/(1-p_e).

    Assumptions: Chance agreement is estimated from suitable annotator marginals.

    Excludes: Confidence intervals, weighted ordinal disagreement, and prevalence adjustment.

    References
    ----------
    Derivation: scale observed excess agreement p_o-p_e by the maximum possible excess 1-p_e. Cohen (1960) introduced this agreement coefficient.

    Examples
    --------
    >>> calc_cohens_kappa(0.75, 0.5)
    0.5
    """
    if not 0.0 <= chance_agreement < 1.0:
        raise ValueError("chance_agreement must be in [0, 1).")
    return (observed_agreement - chance_agreement) / (1.0 - chance_agreement)


# --- Two-sample drift statistics (added 2026-10-20) ---------------------------
def calc_ks_critical_value(coefficient, n_reference, n_current):
    """Scale a two-sample Kolmogorov-Smirnov critical coefficient.

    Parameters
    ----------
    coefficient : float
        Positive dimensionless asymptotic coefficient for the significance level.
    n_reference : int
        Positive independent reference-sample count.
    n_current : int
        Positive independent current-sample count.

    Returns
    -------
    float
        Dimensionless critical maximum empirical-CDF difference.

    Notes
    -----
    Equation: D_crit = ``c*sqrt((n_reference+n_current)/(n_reference*n_current))``.

    Assumptions: Large independent samples from continuous distributions under the null.

    Excludes: Exact finite-sample calibration, tied observations, and fitted-null corrections.

    ``D_crit = c * sqrt((n_0 + n_t) / (n_0 * n_t))``, where ``c`` is the
    coefficient for the chosen significance level (1.36 at 0.05,
    ``Ops.Monitoring.KsTestCoefficient``).

    References
    ----------
    Derivation: the asymptotic two-sample KS statistic multiplies D by ``sqrt(n*m/(n+m))``; solve for D at the supplied critical coefficient.

    Examples
    --------
    >>> a = calc_ks_critical_value(1.0, 10, 10)
    >>> b = calc_ks_critical_value(1.0, 40, 40)
    >>> a == 2 * b
    True
    """
    if n_reference <= 0 or n_current <= 0:
        raise ValueError("sample sizes must be positive.")
    return coefficient * math.sqrt((n_reference + n_current) / (n_reference * n_current))


def calc_ks_normal_location_shift(shift_sd):
    """Compute the CDF distance for an equal-variance normal shift.

    Parameters
    ----------
    shift_sd : float
        Mean separation measured in shared standard-deviation units; either sign is allowed.

    Returns
    -------
    float
        Dimensionless population KS distance in [0, 1].

    Notes
    -----
    Equation: D = ``erf(abs(shift_sd)/(2*sqrt(2)))``.

    Assumptions: Two Gaussian populations share a positive variance.

    Excludes: Empirical sampling noise, variance changes, and significance testing.

    The CDF gap is largest midway between the means: ``2 * Phi(shift_sd / 2) - 1``,
    which equals ``erf(shift_sd / (2 * sqrt(2)))``.

    References
    ----------
    Derivation: equal-density crossings occur at the midpoint of the means; evaluate the two normal CDFs there and use ``2*Phi(z)-1``=erf(z/sqrt(2)).

    Examples
    --------
    >>> calc_ks_normal_location_shift(0.0)
    0.0
    """
    return math.erf(abs(shift_sd) / (2 * math.sqrt(2)))


# --- Correlated samples and imitation-learning cost bounds (added 2026-10-20) --
def calc_effective_sample_size(n_records, lag1_correlation):
    """Approximate the independent sample count of an AR(1) mean.

    Parameters
    ----------
    n_records : int or float
        Nonnegative number of observations.
    lag1_correlation : float
        Stationary lag-one correlation rho in (-1, 1).

    Returns
    -------
    float
        Effective count under the asymptotic AR(1) variance approximation.

    Notes
    -----
    Equation: N_eff = ``N*(1-rho)/(1+rho)``.

    Assumptions: A stationary AR(1) autocorrelation ``rho**k`` and enough records for the infinite-lag approximation.

    Excludes: Finite-sample edge corrections, nonstationary drift, and arbitrary dependence.

    ``N_eff = N (1 - rho) / (1 + rho)`` for estimating the mean of a series whose
    adjacent records have lag-one correlation ``rho`` (Thiebaux and Zwiers 1984).

    References
    ----------
    Derivation: sum ``1+2*sum(rho**k``, k>=1)=(1+rho)/(1-rho) as the variance inflation of the sample mean; divide N by that inflation.

    Examples
    --------
    >>> calc_effective_sample_size(100, 0.0)
    100.0
    """
    if not -1.0 < lag1_correlation < 1.0:
        raise ValueError("lag1_correlation must be in (-1, 1).")
    return n_records * (1.0 - lag1_correlation) / (1.0 + lag1_correlation)


def calc_behavior_cloning_cost_bound(horizon_steps, per_step_error):
    """Evaluate quadratic horizon growth of imitation excess cost.

    Parameters
    ----------
    horizon_steps : int or float
        Nonnegative finite-horizon decision count S.
    per_step_error : float
        Demonstrator-distribution classification error epsilon in [0, 1].

    Returns
    -------
    float
        Worst-case excess cumulative cost under normalized per-step costs.

    Notes
    -----
    Equation: Excess cost bound = ``S**2*epsilon``.

    Assumptions: Supervised imitation on expert states, bounded per-step cost, and possible persistent state-distribution shift after an error.

    Excludes: Task-specific recovery, learned-policy guarantees, and finite-data estimation of epsilon.

    Ross and Bagnell (2010), Theorem 2.1: a policy trained only on the
    demonstrator's states compounds a per-step error ``eps`` quadratically.

    References
    ----------
    Ross and Bagnell (2010), Efficient Reductions for Imitation Learning. https://proceedings.mlr.press/v9/ross10a.html

    Examples
    --------
    >>> calc_behavior_cloning_cost_bound(20, 0.01) == 4 * calc_behavior_cloning_cost_bound(10, 0.01)
    True
    """
    return horizon_steps * horizon_steps * per_step_error


def calc_dagger_cost_bound(horizon_steps, per_step_error, recoverability=1.0):
    """Evaluate linear horizon growth under recoverable imitation.

    Parameters
    ----------
    horizon_steps : int or float
        Nonnegative decision horizon S.
    per_step_error : float
        Learner-state error epsilon in [0, 1].
    recoverability : float, optional
        Nonnegative excess-to-go bound u per error; default 1.0.

    Returns
    -------
    float
        Excess cumulative cost bound in the cost units of recoverability.

    Notes
    -----
    Equation: Excess cost bound = ``u*S*epsilon``.

    Assumptions: Expert labels on learner-visited states and a bounded recoverability constant u.

    Excludes: Finite-iteration regret terms, policy optimization, and proof that a given system satisfies the assumptions.

    Ross, Gordon, and Bagnell (2011), Theorem 3.2: linear in the horizon while
    the recoverability constant ``u`` stays bounded.

    References
    ----------
    Ross, Gordon, and Bagnell (2011), A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. https://proceedings.mlr.press/v15/ross11a/ross11a.pdf

    Examples
    --------
    >>> calc_dagger_cost_bound(20, 0.01) == 2 * calc_dagger_cost_bound(10, 0.01)
    True
    """
    return recoverability * horizon_steps * per_step_error


# --- Poisson reserve sizing (added 2026-10-20) ---
def calc_poisson_tail(mean, at_least):
    """Compute the probability of at least a specified event count.

    Parameters
    ----------
    mean : float
        Nonnegative expected Poisson count in one observation interval.
    at_least : int
        Integer event-count threshold; nonpositive thresholds return one.

    Returns
    -------
    float
        Dimensionless upper-tail probability, evaluated by direct CDF summation.

    Notes
    -----
    Equation: P(N>=s) = ``1-sum(exp(-mean)*mean**k/k``!, k=0,...,s-1).

    Assumptions: Poisson counts in a fixed exposure interval.

    Excludes: Overdispersion, nonstationary intensity, and cancellation-resistant extreme-tail numerics.

    The chance that at least ``s`` events fall in an interval whose expected
    count is ``mean``; 1 for ``s <= 0``. A pool with ``s`` standby instances
    and a constant replenishment lead time misses a request exactly when at
    least ``s`` earlier replacements are still outstanding.

    References
    ----------
    Derivation: complement the sum of the Poisson mass function below s.

    Examples
    --------
    >>> calc_poisson_tail(0.0, 1)
    0.0
    """
    if at_least <= 0:
        return 1.0
    return 1 - sum(math.exp(-mean) * mean**k / math.factorial(k) for k in range(at_least))


def calc_poisson_reserve(mean, target):
    """Find the smallest reserve satisfying a Poisson tail target.

    Parameters
    ----------
    mean : float
        Nonnegative expected count during replenishment lead time.
    target : float
        Required upper-tail probability ceiling in (0, 1).

    Returns
    -------
    int
        Smallest reserve s with P(N >= s) < target; equality does not pass.

    Notes
    -----
    Equation: s = min{integer s>=0 : P(Poisson(mean)>=s) < target}.

    Assumptions: Poisson arrivals and fixed replenishment lead time, with a valid positive target.

    Excludes: Variable lead times, correlated failures, and a stable extreme-tail evaluation.

    References
    ----------
    Derivation: enumerate integer reserves in increasing order and stop at the first upper tail below the target.

    Examples
    --------
    >>> calc_poisson_reserve(0.0, 0.1)
    1
    """
    reserve = 0
    while calc_poisson_tail(mean, reserve) >= target:
        reserve += 1
    return reserve


def calc_poisson_mean_for_reserve(reserve, target, iterations=60):
    """Find a mean-count boundary for a fixed Poisson reserve.

    Parameters
    ----------
    reserve : int
        Positive integer reserve count.
    target : float
        Upper-tail probability target in (0, 1).
    iterations : int, optional
        Positive bisection iteration count; default 60.

    Returns
    -------
    float
        Lower-side approximation to the mean with tail probability equal to target.

    Notes
    -----
    Equation: Solve P(Poisson(mean)>=reserve)=target by expanding the upper bracket then bisecting.

    Assumptions: The Poisson upper tail increases monotonically with its mean; valid reserve and target guarantee a crossing.

    Excludes: Exact root arithmetic, extreme-tail numerical stabilization, and non-Poisson demand.

    Bisection on the tail, which rises monotonically with the mean.

    References
    ----------
    Derivation: monotonic inversion of the Poisson survival probability with bisection.

    Examples
    --------
    >>> mean = calc_poisson_mean_for_reserve(1, 0.5)
    >>> abs(calc_poisson_tail(mean, 1) - 0.5) < 1e-12
    True
    """
    lo, hi = 0.0, 1.0
    while calc_poisson_tail(hi, reserve) < target:
        hi *= 2
    for _ in range(iterations):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if calc_poisson_tail(mid, reserve) < target else (lo, mid)
    return lo


def calc_cdf_quantile(values, cumulative, fraction):
    """Select the first value attaining a supplied cumulative fraction.

    Parameters
    ----------
    values : sequence
        Nonempty support values in ascending order and common units.
    cumulative : sequence of float
        Aligned nondecreasing cumulative probabilities in [0, 1].
    fraction : float
        Requested quantile fraction in [0, 1].

    Returns
    -------
    object
        First support value at or above the fraction; last value when the supplied CDF never reaches it.

    Notes
    -----
    Equation: Return values[min{i : cumulative[i]>=fraction}], or the final value if no crossing exists.

    Assumptions: Caller supplies aligned ordered CDF data; no interpolation is performed.

    Excludes: CDF estimation, input normalization, interpolation, and uncertainty intervals.

    ``values`` ascend with ``cumulative`` (a released CDF); past the end the
    last value is returned.

    References
    ----------
    Derivation: the generalized inverse of a discrete CDF selects its first threshold crossing; endpoint fallback is the implementation convention.

    Examples
    --------
    >>> calc_cdf_quantile([1, 3], [0.5, 1.0], 0.75)
    3
    """
    for v, c in zip(values, cumulative):
        if c >= fraction:
            return v
    return values[-1]


def calc_sample_collection_time(samples, request_rate, traffic_share):
    """Estimate collection time for a fixed traffic allocation.

    Parameters
    ----------
    samples : int or Quantity
        Nonnegative target observation count.
    request_rate : float or Quantity
        Positive requests per time unit; plain floats set the returned time unit.
    traffic_share : float
        Allocated fraction in (0, 1].

    Returns
    -------
    float or Quantity
        Time to collect samples at a constant effective observation rate.

    Notes
    -----
    Equation: t = ``samples/(request_rate*traffic_share)``.

    Assumptions: Every allocated request supplies one observation and the traffic rate is constant.

    Excludes: Burstiness, dropped observations, dependence, and ramp-up scheduling.

    A canary stage that receives ``traffic_share`` of ``request_rate`` gathers
    its planned guardrail sample in ``t = n / (r p)``.

    References
    ----------
    Derivation: observation count equals effective rate times elapsed time; solve for elapsed time.

    Examples
    --------
    >>> calc_sample_collection_time(100, 20.0, 0.5)
    10.0
    """
    return samples / (request_rate * traffic_share)


# --- Drift tests, conformal ranks, and certified radii (added 2026-10-20) ------
def calc_ks_coefficient(significance):
    """Compute an asymptotic two-sample KS critical coefficient.

    Parameters
    ----------
    significance : float
        Two-sided significance probability alpha in (0, 1).

    Returns
    -------
    float
        Dimensionless large-sample coefficient for calc_ks_critical_value.

    Notes
    -----
    Equation: c(alpha) = sqrt(-log(alpha/2)/2).

    Assumptions: The leading asymptotic KS tail approximation is adequate for the selected alpha.

    Excludes: Exact finite-sample probabilities, ties, and parameter-estimation corrections.

    About 1.36 at 0.05 (``Ops.Monitoring.KsTestCoefficient``) and 1.63 at 0.01;
    ``calc_ks_critical_value`` scales it by the two sample sizes.

    References
    ----------
    Derivation: invert the leading tail approximation alpha=``2*exp(-2*c**2)``.

    Examples
    --------
    >>> calc_ks_coefficient(0.01) > calc_ks_coefficient(0.05)
    True
    """
    if not 0.0 < significance < 1.0:
        raise ValueError("significance must be in (0, 1).")
    return math.sqrt(-math.log(significance / 2) / 2)


def calc_standard_error(sd, n_samples):
    """Compute the standard error of an independent sample mean.

    Parameters
    ----------
    sd : float or Quantity
        Nonnegative population or estimated standard deviation, in observation units.
    n_samples : int
        Positive independent observation count.

    Returns
    -------
    float or Quantity
        Mean standard error in the same units as sd.

    Notes
    -----
    Equation: SE = sd/sqrt(n_samples).

    Assumptions: Independent equal-variance observations; sd describes their spread.

    Excludes: Autocorrelation, unequal weights, confidence quantiles, and uncertainty in sd.

    References
    ----------
    Derivation: the variance of an average of n independent equal-variance observations is ``sd**2/n``.

    Examples
    --------
    >>> calc_standard_error(2.0, 4)
    1.0
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    return sd / math.sqrt(n_samples)


def calc_gaussian_kl_terms(mean_p, sd_p, mean_q, sd_q):
    """Decompose the KL divergence between scalar normal distributions.

    Parameters
    ----------
    mean_p : float
        Mean of P, in the shared scalar unit.
    sd_p : float
        Positive standard deviation of P, in the shared scalar unit.
    mean_q : float
        Mean of Q, in the same unit as mean_p.
    sd_q : float
        Positive standard deviation of Q, in the shared scalar unit.

    Returns
    -------
    tuple of float
        Log ratio (dimensionless), ratio numerator (squared input unit), and ratio denominator (squared input unit); the final -1/2 term is separate.

    Notes
    -----
    Equation: ln(sd_q/sd_p) + ``(sd_p**2`` + ``(mean_p-mean_q)**2)/(2*sd_q**2)`` - 1/2.

    Assumptions: P and Q are univariate normal distributions with positive standard deviations.

    Excludes: Covariance matrices, estimation uncertainty, and nonnormal distribution shape.

    References
    ----------
    Derivation: Substitute the two normal density functions into E_P[log(p/q)] and use E_P[``(X-mean_q)**2``] = ``sd_p**2`` + ``(mean_p-mean_q)**2``.

    Examples
    --------
    >>> log_term, numerator, denominator = calc_gaussian_kl_terms(0, 1, 0, 1)
    >>> log_term + numerator / denominator - 0.5 == 0
    True
    """
    if sd_p <= 0 or sd_q <= 0:
        raise ValueError("standard deviations must be positive.")
    return math.log(sd_q / sd_p), sd_p**2 + (mean_p - mean_q) ** 2, 2 * sd_q**2


def calc_gaussian_kl(mean_p, sd_p, mean_q, sd_q):
    """Compute the KL divergence between scalar normal distributions.

    Parameters
    ----------
    mean_p : float
        Mean of P, in the shared scalar unit.
    sd_p : float
        Positive standard deviation of P, in the shared scalar unit.
    mean_q : float
        Mean of Q, in the same unit as mean_p.
    sd_q : float
        Positive standard deviation of Q, in the shared scalar unit.

    Returns
    -------
    float
        KL(P || Q), dimensionless and expressed in nats.

    Notes
    -----
    Equation: ln(sd_q/sd_p) + ``(sd_p**2`` + ``(mean_p-mean_q)**2)/(2*sd_q**2)`` - 1/2.

    Assumptions: The scalar inputs describe normal distributions, and logarithms are natural.

    Excludes: Multivariate covariance and uncertainty in estimated distribution parameters.

    References
    ----------
    Derivation: Substitute the two normal density functions into E_P[log(p/q)] and use E_P[``(X-mean_q)**2``] = ``sd_p**2`` + ``(mean_p-mean_q)**2``.

    Examples
    --------
    >>> calc_gaussian_kl(0, 1, 0, 1) == 0
    True
    """
    log_term, num, den = calc_gaussian_kl_terms(mean_p, sd_p, mean_q, sd_q)
    return log_term + num / den - 0.5


def calc_split_conformal_rank(calibration_count: int, alpha: float) -> int:
    """Find the finite-sample split-conformal calibration rank.

    Parameters
    ----------
    calibration_count : int
        Number n of calibration scores; at least one.
    alpha : float
        Desired marginal miscoverage probability, between zero and one.

    Returns
    -------
    int
        One-based rank; a value greater than n means no finite calibration quantile is available.

    Notes
    -----
    Equation: rank = ceil((n + 1) * (1 - alpha)).

    Assumptions: Calibration scores and the new score are exchangeable; the scoring rule is fixed independently of calibration.

    Excludes: Conditional coverage, distribution shift, and constructing the prediction set itself.

    References
    ----------
    Angelopoulos and Bates, A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.
    https://arxiv.org/abs/2107.07511

    Examples
    --------
    >>> calc_split_conformal_rank(100, 0.1) <= 100
    True
    """
    if calibration_count < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("calibration_count must be >= 1 and alpha in (0, 1)")
    return math.ceil((calibration_count + 1) * (1 - alpha))


def calc_smoothing_radius(noise_sd, top_class_probability):
    """Compute the binary Gaussian randomized-smoothing radius.

    Parameters
    ----------
    noise_sd : float or Quantity
        Nonnegative Gaussian noise standard deviation, in the input norm unit; the function does not validate its sign.
    top_class_probability : float
        Certified lower probability bound for the top class, between 0.5 and one.

    Returns
    -------
    float or Quantity
        Certified l2 radius in the same unit and scalar representation as noise_sd.

    Notes
    -----
    Equation: R = noise_sd * ``Phi**(-1)(top_class_probability)``.

    Assumptions: Isotropic Gaussian smoothing and a valid probability lower bound; the binary complement bounds the competing class.

    Excludes: Estimation of the probability bound, nonspherical noise, and certification of an unsmoothed classifier.

    References
    ----------
    Cohen, Rosenfeld, and Kolter (2019), Certified Adversarial Robustness via Randomized Smoothing.
    https://arxiv.org/abs/1902.02918

    Examples
    --------
    >>> calc_smoothing_radius(2.0, 0.75) == 2 * calc_smoothing_radius(1.0, 0.75)
    True
    """
    from statistics import NormalDist

    if not 0.5 < top_class_probability < 1.0:
        raise ValueError("top_class_probability must be in (0.5, 1).")
    return noise_sd * NormalDist().inv_cdf(top_class_probability)


def calc_linf_corner_l2(linf_budget, coordinates):
    """Convert an infinity-norm corner radius to its Euclidean norm.

    Parameters
    ----------
    linf_budget : float or Quantity
        Nonnegative coordinate-wise perturbation bound, in the coordinate unit.
    coordinates : int
        Nonnegative number of equal-scale coordinates.

    Returns
    -------
    float or Quantity
        Euclidean norm in the same unit as linf_budget.

    Notes
    -----
    Equation: l2 = linf_budget * sqrt(coordinates).

    Assumptions: Each coordinate has absolute value equal to the bound; coordinate units agree.

    Excludes: Typical perturbation size, weighted norms, and validation of the supplied domain.

    References
    ----------
    Derivation: The sum of squared coordinates is coordinates * ``linf_budget**2``; take its square root.

    Examples
    --------
    >>> calc_linf_corner_l2(0.0, 10) == 0
    True
    """
    return linf_budget * math.sqrt(coordinates)


# --- Sequential tests, exact two-arm tests, and Poisson intervals (added 2026-10-20) ---
def calc_sprt_increments(p_unacceptable, p_acceptable):
    """Compute Bernoulli SPRT log-likelihood increments.

    Parameters
    ----------
    p_unacceptable : float
        Failure probability under H0, greater than p_acceptable and below one.
    p_acceptable : float
        Failure probability under H1, greater than zero and below p_unacceptable.

    Returns
    -------
    tuple of float
        Success increment then failure increment, in log-likelihood nats.

    Notes
    -----
    Equation: success = ln((1-p1)/(1-p0)); failure = ln(p1/p0).

    Assumptions: Independent Bernoulli trials and fixed simple hypotheses p0 and p1.

    Excludes: Adaptive hypotheses, dependent outcomes, and stopping-rule overshoot corrections.

    References
    ----------
    Wald (1945), Sequential Tests of Statistical Hypotheses.
    Derivation: Divide each Bernoulli event probability under H1 by its probability under H0, then take the natural logarithm.

    Examples
    --------
    >>> success, failure = calc_sprt_increments(0.2, 0.1)
    >>> success > 0 > failure
    True
    """
    p0, p1 = float(p_unacceptable), float(p_acceptable)
    if not (0.0 < p1 < p0 < 1.0):
        raise ValueError("need 0 < p_acceptable < p_unacceptable < 1.")
    return math.log((1 - p1) / (1 - p0)), math.log(p1 / p0)


def calc_sprt_boundaries(alpha, beta):
    """Compute Wald stopping boundaries for a log-likelihood ratio.

    Parameters
    ----------
    alpha : float
        Allowed false acceptance of H1 under H0, between zero and one.
    beta : float
        Allowed false rejection of H1 under H1, between zero and one.

    Returns
    -------
    tuple of float
        Upper acceptance boundary for H1, then lower rejection boundary, in log-likelihood nats.

    Notes
    -----
    Equation: upper = ln((1-beta)/alpha); lower = ln(beta/(1-alpha)).

    Assumptions: Wald simple-hypothesis sequential test; alpha + beta < 1 gives the usual separated boundaries.

    Excludes: Discrete overshoot correction and validation that alpha + beta is below one.

    References
    ----------
    Wald (1945), Sequential Tests of Statistical Hypotheses.
    Derivation: Take logarithms of the likelihood-ratio approximations (1-beta)/alpha and beta/(1-alpha).

    Examples
    --------
    >>> upper, lower = calc_sprt_boundaries(0.05, 0.05)
    >>> upper > 0 > lower
    True
    """
    if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
        raise ValueError("alpha and beta must lie in (0, 1).")
    return math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))


def calc_fisher_exact_two_sided(a, b, c, d):
    """Compute a two-sided Fisher exact contingency-table p-value.

    Parameters
    ----------
    a, b, c, d : int
        Nonnegative cell counts of [[a, b], [c, d]], with nonzero total; callers must supply valid integer counts.

    Returns
    -------
    float
        Dimensionless probability of fixed-margin tables no more probable than the observed table.

    Notes
    -----
    Equation: sum_x P(X=x) for P(X=x) <= P(X=``a)*(1+1e-9)``, with X hypergeometric.

    Assumptions: Fixed row and column margins and independent observations under the null odds ratio of one.

    Excludes: Alternative two-sided definitions, confidence intervals, and automatic input-domain checks.

    References
    ----------
    Fisher, Statistical Methods for Research Workers (1934 edition), section 21.02. SciPy documents the same probability-ordering convention:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html

    Examples
    --------
    >>> math.isclose(calc_fisher_exact_two_sided(1, 1, 1, 1), 1.0, rel_tol=0.0, abs_tol=1e-12)
    True
    """
    r1, c1, n = a + b, a + c, a + b + c + d

    def _prob(x):
        return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)

    p_obs = _prob(a)
    lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    return sum(_prob(x) for x in range(lo, hi + 1) if _prob(x) <= p_obs * (1 + 1e-9))


def calc_cohens_h(p_high, p_low):
    """Compute the signed arcsine effect size for two proportions.

    Parameters
    ----------
    p_high : float
        First probability on [0, 1]; the name does not enforce ordering.
    p_low : float
        Second probability on [0, 1].

    Returns
    -------
    float
        Dimensionless signed Cohen h; positive when p_high exceeds p_low.

    Notes
    -----
    Equation: h = ``2*asin(sqrt(p_high))`` - ``2*asin(sqrt(p_low))``.

    Assumptions: Probabilities describe comparable binary outcomes.

    Excludes: Sampling uncertainty, a test decision, and validation of probability bounds.

    References
    ----------
    Cohen (1988), Statistical Power Analysis for the Behavioral Sciences, chapter 6.
    Derivation: Subtract the two variance-stabilizing arcsine-transformed probabilities.

    Examples
    --------
    >>> calc_cohens_h(0.5, 0.5) == 0
    True
    """
    return 2 * math.asin(math.sqrt(p_high)) - 2 * math.asin(math.sqrt(p_low))


def calc_arcsine_two_arm_sample_size(effect_h, alpha=0.05, power=0.80):
    """Plan equal arm sizes using an arcsine effect size.

    Parameters
    ----------
    effect_h : float
        Nonzero dimensionless Cohen h effect size; its sign does not affect sample size.
    alpha : float, optional
        Two-sided significance probability in (0, 1); default 0.05.
    power : float, optional
        Target power probability in (0, 1); default 0.80.

    Returns
    -------
    int
        Required observations per arm, rounded upward.

    Notes
    -----
    Equation: n = ``ceil(2*((Phi**(-1)(1-alpha/2)+Phi**(-1)(power))/h)**2)``.

    Assumptions: Independent equal-size arms and the large-sample variance approximation for transformed proportions.

    Excludes: Exact binomial power, unequal allocation, multiple testing, and input-domain validation.

    References
    ----------
    Cohen (1988), Statistical Power Analysis for the Behavioral Sciences, chapter 6.
    Derivation: Each transformed rate has variance 1/n; solve the two-arm normal power relation with variance 2/n.

    Examples
    --------
    >>> calc_arcsine_two_arm_sample_size(0.4) < calc_arcsine_two_arm_sample_size(0.2)
    True
    """
    from statistics import NormalDist

    z = NormalDist().inv_cdf
    return math.ceil(2 * ((z(1 - alpha / 2) + z(power)) / effect_h) ** 2)


def _poisson_cdf(k, mu):
    """P(N <= k) for N ~ Poisson(mu)."""
    term, total = math.exp(-mu), math.exp(-mu)
    for j in range(1, k + 1):
        term *= mu / j
        total += term
    return total


def calc_poisson_exact_interval(events, alpha=0.05):
    """Invert Poisson tails to bound an event-count mean.

    Parameters
    ----------
    events : int
        Nonnegative observed event count; implementation coerces with int, so callers should supply integers.
    alpha : float, optional
        Two-sided tail error probability in (0, 1); default 0.05.

    Returns
    -------
    tuple of float
        Lower and upper Poisson mean counts; divide by exposure to obtain rate units.

    Notes
    -----
    Equation: P_mu(N >= k) = alpha/2 at lower; P_mu(N <= k) = alpha/2 at upper.

    Assumptions: A homogeneous Poisson count and sufficiently moderate counts for direct probability recurrence.

    Excludes: Overdispersion, exposure uncertainty, extreme-count numerical stabilization, and validation of count or alpha.

    References
    ----------
    Garwood (1936), Fiducial Limits for the Poisson Distribution, Biometrika 28(3/4).
    Derivation: Invert the equal-tail tests of the Poisson mean; the zero-count lower endpoint is zero.

    Examples
    --------
    >>> lower, upper = calc_poisson_exact_interval(0)
    >>> lower == 0 and upper > 0
    True
    """
    k = int(events)

    def _solve(f):
        lo, hi = 0.0, 10.0 * (k + 10)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        return 0.5 * (lo + hi)

    lower = 0.0 if k == 0 else _solve(lambda mu: alpha / 2 - (1.0 - _poisson_cdf(k - 1, mu)))
    upper = _solve(lambda mu: _poisson_cdf(k, mu) - alpha / 2)
    return lower, upper


def calc_split_conformal_min_count(alpha: float) -> int:
    """Find the minimum calibration count for a finite conformal rank.

    Parameters
    ----------
    alpha : float
        Desired miscoverage probability in (0, 1).

    Returns
    -------
    int
        Minimum calibration-score count for which the one-based rank does not exceed the count.

    Notes
    -----
    Equation: n_min = ceil((1-alpha)/alpha).

    Assumptions: The split-conformal rank is ``ceil((n+1)*(1-alpha))``; arithmetic is evaluated in floating point.

    Excludes: Conditional coverage, score construction, and guarantees under nonexchangeable data.

    References
    ----------
    Derivation: Solve ``(n+1)*(1-alpha)`` <= n for n, then round upward.
    Angelopoulos and Bates: https://arxiv.org/abs/2107.07511

    Examples
    --------
    >>> n = calc_split_conformal_min_count(0.1)
    >>> calc_split_conformal_rank(n, 0.1) <= n
    True
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return math.ceil((1 - alpha) / alpha)


# --- Responsible AI: group gaps as evidence (added 2026-10-20) ---------------


def calc_two_proportion_z(rate_a, rate_b, n_a, n_b):
    """Compute a pooled two-proportion z statistic.

    Parameters
    ----------
    rate_a, rate_b : float
        Observed independent-group proportions on [0, 1].
    n_a, n_b : int
        Positive observation counts for the two groups.

    Returns
    -------
    float
        Signed dimensionless normal statistic for rate_a minus rate_b.

    Notes
    -----
    Equation: p = ``(n_a*rate_a+n_b*rate_b)/(n_a+n_b)``; z = ``(rate_a-rate_b)/sqrt(p*(1-p)*(1/n_a+1/n_b))``.

    Assumptions: Independent Bernoulli groups, a common rate under the null, and pooled p in (0, 1).

    Excludes: Small-sample exact inference, continuity correction, dependence, and input validation.

    References
    ----------
    Derivation: Estimate the common null probability by the pooled count and standardize the difference using the sum of independent Bernoulli-mean variances.

    Examples
    --------
    >>> calc_two_proportion_z(0.5, 0.5, 100, 100) == 0
    True
    """
    pooled = (n_a * rate_a + n_b * rate_b) / (n_a + n_b)
    return (rate_a - rate_b) / math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))


def calc_proportion_gap_interval(rate_a, rate_b, n_a, n_b, z_critical):
    """Bound a proportion difference using its unpooled standard error.

    Parameters
    ----------
    rate_a, rate_b : float
        Observed independent-group proportions on [0, 1].
    n_a, n_b : int
        Positive observation counts for the two groups.
    z_critical : float
        Positive normal critical value for the chosen coverage.

    Returns
    -------
    tuple of float
        Lower and upper endpoints in probability-difference units, without clipping.

    Notes
    -----
    Equation: interval = (rate_a-rate_b) +/- ``z_critical*sqrt(rate_a*(1-rate_a)/n_a`` + ``rate_b*(1-rate_b)/n_b)``.

    Assumptions: Independent groups and adequate sample counts for a normal approximation.

    Excludes: Paired observations, exact small-count coverage, and restricting endpoints to the feasible interval.

    References
    ----------
    Derivation: Add independent proportion variances, take the square root, and form a symmetric normal interval around the observed gap.

    Examples
    --------
    >>> lo, hi = calc_proportion_gap_interval(0.5, 0.5, 100, 100, 1.96)
    >>> lo < 0 < hi
    True
    """
    gap = rate_a - rate_b
    se = calc_proportion_gap_se(rate_a, rate_b, n_a, n_b)
    return gap - z_critical * se, gap + z_critical * se


def calc_proportion_gap_se(rate_a, rate_b, n_a, n_b):
    """Estimate the standard error of two independent proportions.

    Parameters
    ----------
    rate_a, rate_b : float
        Observed independent-group proportions on [0, 1].
    n_a, n_b : int
        Positive observation counts for the two groups.

    Returns
    -------
    float
        Nonnegative standard error in probability-difference units.

    Notes
    -----
    Equation: se = ``sqrt(rate_a*(1-rate_a)/n_a`` + ``rate_b*(1-rate_b)/n_b)``.

    Assumptions: Each group is an independent Bernoulli sample; plug-in observed rates estimate its variance.

    Excludes: Paired covariance, clustering, finite-population corrections, and input validation.

    References
    ----------
    Derivation: Variances of independent sample means add when subtracting the means.

    Examples
    --------
    >>> calc_proportion_gap_se(0.5, 0.5, 400, 400) < calc_proportion_gap_se(0.5, 0.5, 100, 100)
    True
    """
    return math.sqrt(rate_a * (1 - rate_a) / n_a + rate_b * (1 - rate_b) / n_b)


def calc_equal_n_for_z(rate_a, rate_b, z_critical):
    """Solve the equal-group size for a specified pooled z statistic.

    Parameters
    ----------
    rate_a, rate_b : float
        Distinct probabilities on [0, 1] with their mean in (0, 1).
    z_critical : float
        Positive target magnitude of the normal statistic.

    Returns
    -------
    float
        Unrounded count per group; callers round up when planning whole observations.

    Notes
    -----
    Equation: n = ``2*p*(1-p)*(z_critical/(rate_a-rate_b))**2``, p = (rate_a+rate_b)/2.

    Assumptions: Equal-size independent Bernoulli groups and fixed hypothesized rates.

    Excludes: Power beyond reaching a z threshold, rate-estimation uncertainty, and input validation.

    References
    ----------
    Derivation: Rearrange z = ``gap/sqrt(2*p*(1-p)/n)`` for n.

    Examples
    --------
    >>> n = calc_equal_n_for_z(0.6, 0.4, 2.0)
    >>> math.isclose(abs(calc_two_proportion_z(0.6, 0.4, n, n)), 2.0)
    True
    """
    pooled = (rate_a + rate_b) / 2
    return 2 * pooled * (1 - pooled) * (z_critical / (rate_a - rate_b)) ** 2


def calc_chi2_gate(confidence, dof):
    """Compute a chi-square consistency-test threshold.

    Parameters
    ----------
    confidence : float
        Desired lower-tail probability in (0, 1).
    dof : int
        Positive number of degrees of freedom.

    Returns
    -------
    float
        Dimensionless chi-square threshold; values above it fail the gate.

    Notes
    -----
    Equation: gate = ``F_chi2(dof)**(-1)(confidence)``.

    Assumptions: The normalized squared statistic has a central chi-square distribution with the specified degrees of freedom.

    Excludes: Verification of Gaussian residuals or covariance calibration, repeated-testing corrections, and bias modeling.

    References
    ----------
    Bar-Shalom, Li, and Kirubarajan (2001), Estimation with Applications to Tracking and Navigation, section 5.4.
    SciPy chi-square distribution and inverse CDF:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html

    Examples
    --------
    >>> calc_chi2_gate(0.99, 2) > calc_chi2_gate(0.95, 2)
    True
    """
    from scipy.stats import chi2

    if not 0.0 < confidence < 1.0 or dof < 1:
        raise ValueError("confidence must be in (0, 1) and dof >= 1")
    return float(chi2.ppf(confidence, dof))


def calc_chi2_mean_band(n_samples, dof, confidence):
    """Bound the mean of independent chi-square consistency statistics.

    Parameters
    ----------
    n_samples : int
        Positive number of independent consistency statistics.
    dof : int
        Positive degrees of freedom of each statistic.
    confidence : float
        Central interval probability in (0, 1).

    Returns
    -------
    tuple of float
        Dimensionless lower and upper thresholds for the sample mean.

    Notes
    -----
    Equation: band = (chi2_ppf((1-c)/2, ``N*d)``, chi2_ppf((1+c)/2, ``N*d))/N``.

    Assumptions: Independent central chi-square statistics with a common dimension d; their sum has ``N*d`` degrees of freedom.

    Excludes: Serially correlated residuals, estimated effective sample size, and noncentral alternatives.

    References
    ----------
    Bar-Shalom, Li, and Kirubarajan (2001), section 5.4.2.
    SciPy chi-square distribution and inverse CDF:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html

    Examples
    --------
    >>> low, high = calc_chi2_mean_band(100, 2, 0.95)
    >>> low < 2 < high
    True
    """
    from scipy.stats import chi2

    if n_samples < 1 or dof < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("n_samples and dof must be >= 1 and confidence in (0, 1)")
    tail = (1 - confidence) / 2
    k = n_samples * dof
    return float(chi2.ppf(tail, k)) / n_samples, float(chi2.ppf(1 - tail, k)) / n_samples


def calc_noncentral_gate_detection(gate, dof, noncentrality):
    """Compute the probability that a biased statistic crosses a gate.

    Parameters
    ----------
    gate : float
        Positive dimensionless squared-statistic threshold.
    dof : int
        Positive number of degrees of freedom.
    noncentrality : float
        Nonnegative dimensionless squared normalized bias; b**2/S in the scalar case.

    Returns
    -------
    float
        Probability in [0, 1] of exceeding the gate.

    Notes
    -----
    Equation: detection = 1 - F_ncx2(gate; dof, noncentrality), evaluated by the survival function.

    Assumptions: Gaussian residuals with fixed covariance and fixed normalized bias.

    Excludes: Time-varying bias, multiple-update detection time, and uncertainty in covariance or bias.

    References
    ----------
    Bar-Shalom, Li, and Kirubarajan (2001), sections 1.4.17 and 5.4.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncx2.html

    Examples
    --------
    >>> calc_noncentral_gate_detection(5.0, 2, 4.0) > calc_noncentral_gate_detection(5.0, 2, 0.0)
    True
    """
    from scipy.stats import ncx2

    if gate <= 0 or dof < 1 or noncentrality < 0:
        raise ValueError("gate must be positive, dof >= 1, and noncentrality >= 0")
    return float(ncx2.sf(gate, dof, noncentrality))


def calc_chi_square_quantile(degrees_of_freedom: int, coverage: float) -> float:
    """Compute a Gaussian confidence region's squared Mahalanobis radius.

    Parameters
    ----------
    degrees_of_freedom : int
        Positive Gaussian dimension.
    coverage : float
        Probability enclosed by the confidence region, in (0, 1).

    Returns
    -------
    float
        Dimensionless squared Mahalanobis radius; the physical semiaxes are sqrt(radius_squared) times covariance-axis standard deviations.

    Notes
    -----
    Equation: r_squared = ``F_chi2(k)**(-1)(coverage)``.

    Assumptions: A Gaussian vector with known mean and nonsingular covariance; the whitened squared norm is chi-square.

    Excludes: Uncertainty from covariance estimation, non-Gaussian coverage, and physical axis conversion.

    References
    ----------
    SciPy chi-square distribution and inverse CDF:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html

    Examples
    --------
    >>> calc_chi_square_quantile(2, 0.95) == calc_chi2_gate(0.95, 2)
    True
    """
    if degrees_of_freedom < 1 or not 0.0 < coverage < 1.0:
        raise ValueError("degrees_of_freedom must be >= 1 and coverage in (0, 1)")
    from scipy.stats import chi2

    return float(chi2.ppf(coverage, degrees_of_freedom))


# --- Acceptance rates: binomial intervals and paired comparison (added 2026-10-20) ---
# A benchmark score is a binomial proportion of accepted tasks. These are the
# closed forms an evaluation reports with it: the Wald and Wilson intervals,
# the normal half-width and the task count it implies, the two-level variance
# of a task-averaged estimate, and the paired comparison of two
# configurations on the same tasks (Newcombe's method 10 and McNemar's exact
# test). The solvers in engine/solvers/acceptance.py compose them.


# --- Agent operations: score intervals, paired differences, drift, sequential samples (added 2026-10-20) ---
def _check_count(successes, trials):
    if not isinstance(successes, int) or not isinstance(trials, int) or not 0 <= successes <= trials or trials <= 0:
        raise ValueError("Require integer counts 0 <= successes <= trials and trials > 0")


def calc_wald_interval(successes, trials, z_critical):
    """Compute an unclipped Wald interval for a binomial proportion.

    Parameters
    ----------
    successes : int
        Accepted outcomes x, from zero through trials.
    trials : int
        Positive number n of independent Bernoulli trials.
    z_critical : float
        Nonnegative normal critical value; callers choose it for the desired coverage.

    Returns
    -------
    tuple of float
        Lower and upper probability endpoints; they may fall outside [0, 1].

    Notes
    -----
    Equation: p_hat +/- ``z_critical*sqrt(p_hat*(1-p_hat)/trials)``, p_hat = successes/trials.

    Assumptions: Independent identically distributed Bernoulli trials and a normal plug-in approximation.

    Excludes: Boundary corrections and reliable small-count coverage; zero or all successes produce zero width.

    References
    ----------
    Brown, Cai, and DasGupta (2001), Interval Estimation for a Binomial Proportion.
    Derivation: Use the observed Bernoulli variance in the symmetric normal interval for its sample mean.

    Examples
    --------
    >>> calc_wald_interval(0, 10, 1.96) == (0.0, 0.0)
    True
    """
    _check_count(successes, trials)
    p = successes / trials
    half = z_critical * math.sqrt(p * (1 - p) / trials)
    return p - half, p + half


def calc_wilson_interval(successes, trials, z_critical):
    """Compute a Wilson score interval for a binomial proportion.

    Parameters
    ----------
    successes : int
        Accepted outcomes x, from zero through trials.
    trials : int
        Positive number n of independent Bernoulli trials.
    z_critical : float
        Nonnegative normal critical value; callers choose it for the desired coverage.

    Returns
    -------
    tuple of float
        Lower and upper probability endpoints, clipped to [0, 1].

    Notes
    -----
    Equation: center=``(x+z**2/2)/(n+z**2)``; half=``z*sqrt(x*(n-x)/n+z**2/4)/(n+z**2)``.

    Assumptions: Independent identically distributed Bernoulli trials and score-test inversion with a nonnegative critical value.

    Excludes: Exact binomial coverage, clustering, and uncertainty from adaptive sample selection.

    References
    ----------
    Wilson (1927), Probable Inference, the Law of Succession, and Statistical Inference.
    Derivation: Solve ``(p_hat-p)**2`` <= ``z**2*p*(1-p)/n`` as a quadratic inequality in p.

    Examples
    --------
    >>> low, high = calc_wilson_interval(0, 10, 1.96)
    >>> 0 <= low < high <= 1
    True
    """
    _check_count(successes, trials)
    x, n, z = successes, trials, z_critical
    center = (x + z * z / 2) / (n + z * z)
    spread = z * math.sqrt(x * (n - x) / n + z * z / 4) / (n + z * z)
    return max(0.0, center - spread), min(1.0, center + spread)


def calc_proportion_half_width(rate, n_samples, z_critical):
    """Estimate the normal interval half-width for a proportion.

    Parameters
    ----------
    rate : float
        Bernoulli probability estimate p on [0, 1].
    n_samples : int
        Positive number of independent samples.
    z_critical : float
        Nonnegative normal critical value for the desired coverage.

    Returns
    -------
    float
        Nonnegative half-width in probability units.

    Notes
    -----
    Equation: half_width = ``z_critical*sqrt(rate*(1-rate)/n_samples)``.

    Assumptions: Independent Bernoulli observations and a normal plug-in approximation.

    Excludes: Boundary corrections, exact coverage, dependence, and input validation.

    References
    ----------
    Derivation: Multiply the standard deviation of a Bernoulli sample mean by the chosen normal critical value.

    Examples
    --------
    >>> calc_proportion_half_width(0.5, 400, 1.96) < calc_proportion_half_width(0.5, 100, 1.96)
    True
    """
    return z_critical * math.sqrt(rate * (1 - rate) / n_samples)


def calc_samples_for_half_width(rate, half_width, z_critical):
    """Plan independent Bernoulli draws for a target half-width.

    Independent draws for a normal-approximation half-width at ``rate``: ``ceil(z^2 p (1 - p) / h^2)``.

    Parameters
    ----------
    rate : float
        Bernoulli probability in [0, 1], dimensionless.
    half_width : float
        Positive absolute rate half-width, dimensionless.
    z_critical : float
        Positive dimensionless normal quantile.

    Returns
    -------
    int
        Draw count, rounded upward.

    Notes
    -----
    Equation: n = ceil(z^2 p(1-p)/h^2).

    Assumptions: Independent Bernoulli observations; normal approximation at the supplied rate.

    Excludes: Exact binomial coverage and clustered observations.

    References
    ----------
    Derivation: Invert the normal half-width z sqrt(p(1-p)/n).

    Examples
    --------
    >>> calc_samples_for_half_width(0.5, 0.1, 2) > 0
    True
    """
    return math.ceil(z_critical**2 * rate * (1 - rate) / half_width**2)


def calc_within_task_variance(mean_rate, between_task_variance):
    """Recover within-task Bernoulli variance from task heterogeneity.

    Mean within-task run variance ``E[p_i (1 - p_i)] = mean (1 - mean) - Var(p_i)``.

    Parameters
    ----------
    mean_rate : float
        Mean task probability in [0, 1], dimensionless.
    between_task_variance : float
        Nonnegative variance of task probabilities, dimensionless.

    Returns
    -------
    float
        Dimensionless within-task outcome variance.

    Notes
    -----
    Equation: E[p_i(1-p_i)] = ``mean_rate*(1-mean_rate)-between_task_variance``.

    Assumptions: A valid mixture of Bernoulli task probabilities.

    Excludes: Estimation uncertainty or correction of inconsistent variance inputs.

    References
    ----------
    Derivation: Conditional on task i, a Bernoulli outcome has variance
    ``p_i*(1-p_i)``. Averaging over tasks gives E[p_i]-E[p_i^2]. Substitute
    E[p_i^2] = Var(p_i)+E[p_i]^2 to obtain the implemented decomposition.

    Examples
    --------
    >>> calc_within_task_variance(0.5, 0.0) == 0.25
    True
    """
    return mean_rate * (1 - mean_rate) - between_task_variance


def calc_task_averaged_variance(between_task_variance, within_task_variance, tasks, runs_per_task):
    """Calculate variance of an equally weighted task average.

    Variance of a task-averaged acceptance estimate: ``sigma_task^2 / N + sigma_run^2 / (N r)``.

    Tasks are sampled independently and runs are independent given the task;
    more runs shrink only the second term, more tasks shrink both.

    Parameters
    ----------
    between_task_variance : float
        Nonnegative variance of task probabilities, dimensionless.
    within_task_variance : float
        Nonnegative conditional outcome variance, dimensionless.
    tasks : float
        Positive number of independently sampled tasks.
    runs_per_task : float
        Positive equal run count per task.

    Returns
    -------
    float
        Dimensionless variance of the estimated average rate.

    Notes
    -----
    Equation: V = between_task_variance/tasks + ``within_task_variance/(tasks*runs_per_task)``.

    Assumptions: Independent tasks and conditionally independent runs with equal replication.

    Excludes: Correlated runs and unequal task weights.

    References
    ----------
    Derivation: Apply total variance to the mean of replicated task outcomes.

    Examples
    --------
    >>> calc_task_averaged_variance(1, 1, 10, 2) < calc_task_averaged_variance(1, 1, 10, 1)
    True
    """
    return between_task_variance / tasks + within_task_variance / (tasks * runs_per_task)


def calc_wilson_clean_runs(target_upper: float, z: float) -> int:
    """Find the clean-run count for a Wilson upper bound.

    Fewest consecutive clean runs whose Wilson upper limit on the failure rate falls below ``target_upper``.

    With zero failures in ``n`` runs the upper limit is ``z^2 / (n + z^2)``,
    which is below ``t`` once ``n > z^2 (1 - t) / t``.

    Parameters
    ----------
    target_upper : float
        Failure probability threshold in (0, 1), dimensionless.
    z : float
        Positive dimensionless normal quantile.

    Returns
    -------
    int
        Run count making the upper bound less than the target.

    Notes
    -----
    Equation: n = ``floor(z^2*(1-target_upper)/target_upper)+1``.

    Assumptions: Independent Bernoulli failures; no failures observed; positive normal quantile.

    Excludes: Optional-stopping guarantees and correlated failures.

    References
    ----------
    Derivation: Solve the zero-success Wilson upper limit z^2/(n+z^2) < target_upper.

    Examples
    --------
    >>> calc_wilson_clean_runs(0.1, 2) > calc_wilson_clean_runs(0.2, 2)
    True
    """
    if not 0.0 < target_upper < 1.0:
        raise ValueError("target_upper must be in (0, 1)")
    return math.floor(z * z * (1 - target_upper) / target_upper) + 1


def calc_newcombe_paired_difference(both, first_only, second_only, neither, z_critical):
    """Estimate a paired rate difference and its interval.

    Paired difference of two acceptance rates with Newcombe's method-10 interval (Newcombe 1998).

    The four cells count the tasks both configurations accept, only the first,
    only the second, and neither. Returns ``(delta, low, high)`` for the first
    configuration's rate minus the second's. Each marginal Wilson interval is
    combined through the continuity-corrected phi of the 2x2 table, so
    positive within-pair dependence narrows the interval.

    Parameters
    ----------
    both : float
        Nonnegative integer count of jointly accepted pairs.
    first_only : float
        Nonnegative integer count accepted only by the first configuration.
    second_only : float
        Nonnegative integer count accepted only by the second configuration.
    neither : float
        Nonnegative integer count accepted by neither configuration.
    z_critical : float
        Positive dimensionless normal quantile.

    Returns
    -------
    tuple of float
        Difference, lower limit, upper limit on the dimensionless rate scale.

    Notes
    -----
    Equation: delta = p1-p2; marginal Wilson widths combine with corrected phi covariance.

    Assumptions: Independent pairs and nonnegative integer contingency counts.

    Excludes: Unpaired sampling and repeated-measures cluster corrections.

    References
    ----------
    Derivation: Newcombe (1998), method 10: combine marginal Wilson score limits using the paired-table correlation.

    Examples
    --------
    >>> calc_newcombe_paired_difference(5, 2, 2, 5, 1.96)[0] == 0
    True
    """
    cells = (both, first_only, second_only, neither)
    if any(not isinstance(v, int) or v < 0 for v in cells) or sum(cells) <= 0:
        raise ValueError("Require nonnegative integer cells and positive total")
    n = sum(cells)
    p1, p2 = (both + first_only) / n, (both + second_only) / n
    l1, u1 = calc_wilson_interval(both + first_only, n, z_critical)
    l2, u2 = calc_wilson_interval(both + second_only, n, z_critical)
    denominator = math.sqrt(
        (both + first_only) * (second_only + neither) * (both + second_only) * (first_only + neither)
    )
    numerator = both * neither - second_only * first_only
    if numerator > 0:
        numerator = max(numerator - n / 2, 0)
    rho = numerator / denominator if denominator else 0.0
    lower_width = math.sqrt(max(0.0, (p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * rho * (p1 - l1) * (u2 - p2)))
    upper_width = math.sqrt(max(0.0, (u1 - p1) ** 2 + (p2 - l2) ** 2 - 2 * rho * (u1 - p1) * (p2 - l2)))
    return p1 - p2, p1 - p2 - lower_width, p1 - p2 + upper_width


def calc_mcnemar_exact_two_sided(first_only, second_only):
    """Compute the exact conditional McNemar p-value.

    Exact two-sided McNemar p from the discordant counts (McNemar 1947).

    Conditional on ``d = first_only + second_only`` discordant pairs, wins are
    Binomial(d, 1/2) under equal marginals:
    ``p = min(1, 2 sum_{j <= min(b, c)} C(d, j) 2^-d)``; 1 when ``d = 0``.

    Parameters
    ----------
    first_only : float
        Nonnegative integer count accepted only by the first configuration.
    second_only : float
        Nonnegative integer count accepted only by the second configuration.

    Returns
    -------
    float
        Dimensionless two-sided p-value.

    Notes
    -----
    Equation: p = min(1, ``2*P``[Binomial(b+c, 0.5) <= min(b,c)]).

    Assumptions: Independent matched pairs and exchangeable discordant labels under the null.

    Excludes: Effect-size intervals and dependence between pairs.

    References
    ----------
    Derivation: Condition on discordance; equal marginals give a fair binomial count.

    Examples
    --------
    >>> calc_mcnemar_exact_two_sided(0, 0) == 1
    True
    """
    discordant = first_only + second_only
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, j) for j in range(min(first_only, second_only) + 1))
    return min(1.0, 2 * tail / 2**discordant)


def calc_paired_difference_variance(discordant_fraction, difference):
    """Calculate variance of a signed paired outcome.

    Per-pair variance of a paired candidate-minus-baseline outcome in {-1, 0, +1}: ``d - delta^2``.

    Parameters
    ----------
    discordant_fraction : float
        Fraction of pairs with different outcomes, in [0, 1].
    difference : float
        Mean difference on the outcome scale; nonzero for sample planning.

    Returns
    -------
    float
        Dimensionless per-pair variance.

    Notes
    -----
    Equation: V = discordant_fraction - difference^2.

    Assumptions: Outcomes are -1, 0, or +1 with compatible supplied moments.

    Excludes: Variance estimation uncertainty and clustering.

    References
    ----------
    Derivation: For signed discordance X, E[X^2] is the discordant fraction.

    Examples
    --------
    >>> calc_paired_difference_variance(0.5, 0) == 0.5
    True
    """
    return discordant_fraction - difference**2


def calc_jensen_shannon_divergence(p, q) -> float:
    """Measure divergence between aligned probability distributions.

    Jensen-Shannon divergence between two aligned distributions, in nats.

    ``JS = KL(p || m) / 2 + KL(q || m) / 2`` with ``m = (p + q) / 2``. It is
    symmetric and bounded by ``ln 2``, which two distributions with disjoint
    support reach. Source: Lin, "Divergence Measures Based on the Shannon
    Entropy," IEEE Trans. Inf. Theory 37(1), 1991.

    Parameters
    ----------
    p : sequence
        Aligned normalized nonnegative probabilities, dimensionless.
    q : sequence
        Aligned normalized nonnegative probabilities, dimensionless.

    Returns
    -------
    float
        Divergence in nats.

    Notes
    -----
    Equation: JS = (KL(p||m)+KL(q||m))/2, m=(p+q)/2.

    Assumptions: Aligned, nonnegative, normalized probabilities; natural logarithms.

    Excludes: Automatic normalization, smoothing, and sampling uncertainty.

    References
    ----------
    Derivation: Lin (1991), Divergence Measures Based on the Shannon Entropy; mixture-relative entropy definition.

    Examples
    --------
    >>> calc_jensen_shannon_divergence([0.5, 0.5], [0.5, 0.5]) == 0
    True
    """
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    m = [(a + b) / 2 for a, b in zip(p, q)]

    def _kl(x, y):
        return sum(a * math.log(a / b) for a, b in zip(x, y) if a > 0)

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def calc_sprt_expected_samples(upper: float, lower: float, drift: float, upper_probability: float) -> float:
    """Approximate a likelihood walk stopping count.

    Wald's expected stopping sample count, neglecting overshoot.

    The walk ends at ``upper`` with probability ``upper_probability`` and at
    ``lower`` otherwise; dividing the expected terminal log-likelihood by the
    expected step ``drift`` gives ``E[N]`` (Wald 1945, average sample number).

    Parameters
    ----------
    upper : float
        Upper terminal log-likelihood boundary, dimensionless.
    lower : float
        Lower terminal log-likelihood boundary, dimensionless.
    drift : float
        Nonzero mean log-likelihood increment per observation.
    upper_probability : float
        Probability of terminating at the upper boundary, in [0, 1].

    Returns
    -------
    float
        Approximate sample count; inputs must describe a consistent walk.

    Notes
    -----
    Equation: E[N] = ``(upper_probability*upper+(1-upper_probability)*lower)/drift``.

    Assumptions: A stopped walk satisfying Wald expectation conditions with nonzero mean increment.

    Excludes: Boundary overshoot and finite-horizon truncation.

    References
    ----------
    Derivation: Wald expectation identity equates expected terminal sum to drift times expected count.

    Examples
    --------
    >>> calc_sprt_expected_samples(2, -2, 1, 1) == 2
    True
    """
    if drift == 0:
        raise ValueError("drift must be nonzero")
    return (upper_probability * upper + (1 - upper_probability) * lower) / drift


# ── Per-group detection: frames to a reliable detection, exact intervals (added 2026-10-20) ──
def calc_consecutive_detection_frames(per_frame_detection: float, all_miss_probability: float) -> int:
    """Find the frame count for an all-miss probability target.

    Consecutive frames a person must be in view before detection may be counted on.

    Equation:
        N = ceil( ln p_miss / ln(1 - p_frame) )

    the fewest frames for which the chance of missing the person in every one,
    (1 - p_frame)^N, is at most ``all_miss_probability``, with misses
    independent from frame to frame (the best case; correlated misses need
    more frames). A tolerance of 1e-12 keeps an exact ratio from rounding up.

    Parameters
    ----------
    per_frame_detection : float
        Detection probability per independent frame, in (0, 1).
    all_miss_probability : float
        Maximum all-miss probability in (0, 1).

    Returns
    -------
    int
        Minimum frame count under the independent-miss model.

    Notes
    -----
    Equation: N = ceil(log(all_miss_probability)/log(1-per_frame_detection)-1e-12).

    Assumptions: Independent frame misses with a constant detection probability.

    Excludes: Temporal correlation and uncertainty in detector performance.

    References
    ----------
    Derivation: Multiply independent miss probabilities and invert the geometric decay.

    Examples
    --------
    >>> calc_consecutive_detection_frames(0.5, 0.25) == 2
    True
    """
    p = float(per_frame_detection)
    q = float(all_miss_probability)
    if not 0.0 < p < 1.0:
        raise ValueError(f"per_frame_detection must be in (0, 1), got {p}")
    if not 0.0 < q < 1.0:
        raise ValueError(f"all_miss_probability must be in (0, 1), got {q}")
    return math.ceil(math.log(q) / math.log(1.0 - p) - 1e-12)


def calc_clopper_pearson_interval(successes: int, trials: int, confidence: float = 0.95):
    """Compute a two-sided exact binomial confidence interval.

    Exact two-sided Clopper-Pearson interval for ``successes`` out of ``trials``.

    Equation (beta quantiles, alpha = 1 - confidence):
        lower = B^{-1}(alpha/2; k, n - k + 1)       (0 when k = 0)
        upper = B^{-1}(1 - alpha/2; k + 1, n - k)   (1 when k = n)

    Source: Clopper and Pearson, "The Use of Confidence or Fiducial Limits
    Illustrated in the Case of the Binomial," Biometrika 26(4), 1934.

    Parameters
    ----------
    successes : float
        Integer successes from zero through trials.
    trials : float
        Positive integer trial count.
    confidence : float
        Central confidence probability in (0, 1).

    Returns
    -------
    tuple of float
        Lower and upper binomial probability limits.

    Notes
    -----
    Equation: Limits are beta quantiles at alpha/2 and 1-alpha/2 with endpoint conventions.

    Assumptions: Independent identically distributed Bernoulli trials and fixed sample size.

    Excludes: Sequential testing and dependent observations.

    References
    ----------
    Derivation: Clopper and Pearson (1934), The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial.

    Examples
    --------
    >>> calc_clopper_pearson_interval(0, 10)[0] == 0
    True
    """
    from scipy.stats import beta

    k, n = int(successes), int(trials)
    if n < 1 or not 0 <= k <= n:
        raise ValueError("need 0 <= successes <= trials and trials >= 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    a = 1.0 - confidence
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


# --- Detectable lift and exponential decay rate (added 2026-10-20) ------------
def calc_two_proportion_detectable_lift(baseline_rate, n_per_variant, z_alpha=1.96, z_beta=0.84):
    """Invert the common-variance two-arm planning formula.

    Smallest absolute lift a two-arm test with ``n_per_variant`` per arm can detect.

    The common-variance sample-size formula of ``calc_two_proportion_sample_size``
    solved for the effect: ``delta = sqrt(2 (z_alpha + z_beta)^2 p (1 - p) / n)``.

    Parameters
    ----------
    baseline_rate : float
        Baseline probability in (0, 1).
    n_per_variant : float
        Positive observation count per arm.
    z_alpha : float
        Dimensionless normal significance quantile; default 1.96 where specified.
    z_beta : float
        Dimensionless normal power quantile; default 0.84 where specified.

    Returns
    -------
    float
        Absolute rate difference, dimensionless.

    Notes
    -----
    Equation: delta = ``sqrt(2*(z_alpha+z_beta)^2*p*(1-p)/n_per_variant)``.

    Assumptions: Independent equally sized arms and baseline variance reused for both.

    Excludes: Exact power, unequal allocation, and pooled alternative variance.

    References
    ----------
    Derivation: Solve the normal sample-size equation for its absolute effect.

    Examples
    --------
    >>> calc_two_proportion_detectable_lift(0.5, 200) < calc_two_proportion_detectable_lift(0.5, 100)
    True
    """
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be in (0, 1).")
    if n_per_variant <= 0:
        raise ValueError("n_per_variant must be positive.")
    variance = baseline_rate * (1 - baseline_rate)
    return math.sqrt(2 * (z_alpha + z_beta) ** 2 * variance / n_per_variant)


def calc_exponential_decay_rate(initial, later, elapsed):
    """Fit an exponential rate through two readings.

    Rate ``gamma`` of ``A(t) = A0 exp(-gamma t)`` through two readings:
    ``gamma = ln(A0 / A1) / t``. ``elapsed`` may be a Quantity (the rate is then
    per its unit) or a plain number.

    Parameters
    ----------
    initial : float
        Positive initial reading, in the same units as later.
    later : float
        Positive later reading, in the same units as initial.
    elapsed : float
        Nonzero elapsed time, as a float or Pint quantity.

    Returns
    -------
    float or Quantity
        Inverse-time rate, retaining the inverse units of elapsed when supplied.

    Notes
    -----
    Equation: gamma = log(initial/later)/elapsed.

    Assumptions: Positive readings in matching units and nonzero elapsed time.

    Excludes: Measurement noise and multiple-timescale decay.

    References
    ----------
    Derivation: Take logarithms of later = ``initial*exp(-gamma*elapsed)``.

    Examples
    --------
    >>> calc_exponential_decay_rate(2, 2, 1) == 0
    True
    """
    if initial <= 0 or later <= 0:
        raise ValueError("initial and later must be positive.")
    return math.log(initial / later) / elapsed


# --- Benchmark run statistics: Welch intervals, run counts, paired gaps, calibration (added 2026-10-20) ---
# Repeated benchmark runs are a small sample of a noisy measurement. These are
# the closed forms a benchmark comparison reports: the Student t critical value
# for few runs, Welch's interval on a difference of means with unequal spreads,
# the half-width of a mean of n runs, the runs a two-sample comparison needs at
# a stated power, the largest disagreement count at which McNemar's test flags
# a fixed net gap, and the expected calibration error of binned predictions.


def calc_t_critical(degrees_of_freedom, confidence):
    """Return a two-sided Student t critical value.

    Two-sided Student t critical value: the ``(1 + confidence) / 2`` quantile (2.78 at 4 dof, 0.95).

    Non-integer degrees of freedom (Welch-Satterthwaite) are allowed.

    Parameters
    ----------
    degrees_of_freedom : float
        Positive degrees of freedom, possibly fractional.
    confidence : float
        Central confidence probability in (0, 1).

    Returns
    -------
    float
        Dimensionless positive t quantile.

    Notes
    -----
    Equation: t = inverse_t_cdf((1+confidence)/2, degrees_of_freedom).

    Assumptions: Positive degrees of freedom; confidence lies between zero and one.

    Excludes: Automatic choice of degrees of freedom from data.

    References
    ----------
    Derivation: Invert the symmetric Student t distribution at the requested central coverage.

    Examples
    --------
    >>> calc_t_critical(10, 0.95) > 0
    True
    """
    if degrees_of_freedom <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("degrees_of_freedom must be positive and confidence in (0, 1)")
    from scipy.stats import t

    return float(t.ppf(0.5 + confidence / 2, degrees_of_freedom))


def calc_welch_interval(mean_a, sd_a, n_a, mean_b, sd_b, n_b, confidence):
    """Construct a Welch interval for two independent means.

    Welch's interval on ``mean_a - mean_b`` for two samples with unequal spreads (Welch 1947).

    ``se = sqrt(s_a^2 / n_a + s_b^2 / n_b)``; the Welch-Satterthwaite degrees
    of freedom are ``se^4 / ((s_a^2/n_a)^2 / (n_a - 1) + (s_b^2/n_b)^2 / (n_b - 1))``.
    Returns ``dict(difference, se, dof, t, low, high)``.

    Parameters
    ----------
    mean_a : float
        First sample mean, in a common measurement unit.
    sd_a : float
        Nonnegative sample standard deviation, in the mean measurement unit.
    n_a : float
        First sample count, at least two.
    mean_b : float
        Second sample mean, in the same unit as mean_a.
    sd_b : float
        Nonnegative sample standard deviation, in the mean measurement unit.
    n_b : float
        Second sample count, at least two.
    confidence : float
        Central confidence probability in (0, 1).

    Returns
    -------
    dict
        Difference, se, low, high in input units; dof and t dimensionless.

    Notes
    -----
    Equation: SE = sqrt(sd_a^2/n_a+sd_b^2/n_b); limits = mean_a-mean_b +/- ``t*SE``.

    Assumptions: Independent samples, at least two observations each, nonzero combined spread.

    Excludes: Paired designs and exact guarantees for arbitrary heavy-tailed data.

    References
    ----------
    Derivation: Welch (1947): use Satterthwaite effective degrees of freedom for unequal sample variances.

    Examples
    --------
    >>> calc_welch_interval(1, 1, 10, 1, 1, 10, 0.95)["difference"] == 0
    True
    """
    if n_a < 2 or n_b < 2:
        raise ValueError("each sample needs at least two values")
    va, vb = sd_a**2 / n_a, sd_b**2 / n_b
    se = math.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va**2 / (n_a - 1) + vb**2 / (n_b - 1))
    t = calc_t_critical(dof, confidence)
    d = mean_a - mean_b
    return dict(difference=d, se=se, dof=dof, t=t, low=d - t * se, high=d + t * se)


def calc_mean_half_width(sd, n_samples, confidence):
    """Calculate a Student t interval half-width for a mean.

    Half-width of a ``confidence`` interval on a mean of ``n_samples`` values: ``t_{n-1} sd / sqrt(n)``.

    Parameters
    ----------
    sd : float
        Nonnegative standard deviation in the measurement unit.
    n_samples : float
        Observation count, at least two.
    confidence : float
        Central confidence probability in (0, 1).

    Returns
    -------
    float
        Half-width in the same units as sd.

    Notes
    -----
    Equation: h = ``t_(n-1)*sd/sqrt(n_samples)``.

    Assumptions: Independent observations; t coverage is exact for a normal population.

    Excludes: Serial dependence and systematic measurement bias.

    References
    ----------
    Derivation: Multiply the standard error by the two-sided Student t quantile.

    Examples
    --------
    >>> calc_mean_half_width(0, 10, 0.95) == 0
    True
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    return calc_t_critical(n_samples - 1, confidence) * calc_standard_error(sd, n_samples)


def calc_samples_per_arm_for_difference(sd, difference, z_alpha, z_beta):
    """Plan equal-arm observations for a mean difference.

    Samples per arm for a two-sided two-sample test of a mean difference: ``2 (z_a + z_b)^2 sd^2 / d^2``.

    The normal-approximation power formula; ``z_alpha`` is the two-sided
    critical value (1.96 at 95 percent) and ``z_beta`` the power quantile
    (0.84 at 80 percent power). Not rounded up.

    Parameters
    ----------
    sd : float
        Nonnegative standard deviation in the measurement unit.
    difference : float
        Mean difference on the outcome scale; nonzero for sample planning.
    z_alpha : float
        Dimensionless normal significance quantile; default 1.96 where specified.
    z_beta : float
        Dimensionless normal power quantile; default 0.84 where specified.

    Returns
    -------
    float
        Unrounded per-arm sample count; round upward for planning.

    Notes
    -----
    Equation: n = ``2*(z_alpha+z_beta)^2*sd^2/difference^2``.

    Assumptions: Independent arms with equal variance and a normal power approximation.

    Excludes: Small-sample t iteration and unequal allocation.

    References
    ----------
    Derivation: Solve the normal difference-of-means power equation for equal arm size.

    Examples
    --------
    >>> calc_samples_per_arm_for_difference(1, 1, 2, 1) > 0
    True
    """
    if difference == 0:
        raise ValueError("difference must be nonzero")
    return 2 * (z_alpha + z_beta) ** 2 * sd**2 / difference**2


def calc_mcnemar_max_discordant(net_gap, critical):
    """Bound discordance for an uncorrected McNemar threshold.

    Largest disagreement count ``b + c`` at which McNemar's statistic still exceeds ``critical``.

    The statistic (no continuity correction) is ``(b - c)^2 / (b + c)``; with
    the net gap ``b - c`` fixed, it exceeds the 1-dof chi-square critical value
    only when ``b + c <= floor(gap^2 / critical)`` (McNemar 1947).

    Parameters
    ----------
    net_gap : float
        Signed integer difference between discordant cell counts.
    critical : float
        Positive dimensionless chi-square statistic threshold.

    Returns
    -------
    int
        Maximum algebraic discordant count, before contingency-table feasibility.

    Notes
    -----
    Equation: d_max = floor(net_gap^2/critical).

    Assumptions: Uncorrected chi-square approximation and a fixed net gap.

    Excludes: Continuity correction, exact p-values, and parity feasibility of counts.

    References
    ----------
    Derivation: Rearrange gap^2/d >= critical; equality is included by the returned floor.

    Examples
    --------
    >>> calc_mcnemar_max_discordant(4, 4) == 4
    True
    """
    if critical <= 0:
        raise ValueError("critical must be positive")
    return math.floor(net_gap**2 / critical)


def calc_expected_calibration_error(bins):
    """Compute count-weighted absolute calibration error.

    Expected calibration error of binned predictions: ``sum |B_i| / n * |acc_i - conf_i|``.

    ``bins`` is a sequence of ``(count, confidence, accuracy)``; the weight of
    each bin is its share of the predictions (Pakdaman Naeini et al. 2015;
    Guo et al. 2017, eq. 3).

    Parameters
    ----------
    bins : sequence
        Sequence of (count, confidence, accuracy) triples; rates in [0, 1].

    Returns
    -------
    float
        Dimensionless weighted absolute probability error.

    Notes
    -----
    Equation: ECE = sum(count_i/total * abs(accuracy_i-confidence_i)).

    Assumptions: Nonnegative bin counts with positive total; rates on the same probability scale.

    Excludes: Bin selection, debiasing, and confidence intervals.

    References
    ----------
    Derivation: Guo et al. (2017), On Calibration of Modern Neural Networks: count-weighted bin accuracy-confidence differences.

    Examples
    --------
    >>> calc_expected_calibration_error([(10, 0.5, 0.5)]) == 0
    True
    """
    total = sum(n for n, _, _ in bins)
    if total <= 0:
        raise ValueError("bins must hold at least one prediction")
    return sum(n / total * abs(acc - conf) for n, conf, acc in bins)


def calc_spearman_rank_correlation(xs, ys):
    """Correlate paired observations through their average ranks.

    Spearman's rank correlation: Pearson correlation of the average ranks (ties share a rank).

    Parameters
    ----------
    xs : sequence
        First nonconstant sequence of numeric observations.
    ys : sequence
        Second nonconstant sequence paired with xs.

    Returns
    -------
    float
        Dimensionless rank correlation between minus one and one.

    Notes
    -----
    Equation: rho = centered rank cross-product / product of centered rank norms.

    Assumptions: Equal-length paired sequences with at least two values and nonconstant ranks.

    Excludes: P-values, missing-value handling, and causal interpretation.

    References
    ----------
    Derivation: Replace values by average ranks, then apply the Pearson correlation formula.

    Examples
    --------
    >>> calc_spearman_rank_correlation([1, 2, 3], [2, 4, 6]) == 1
    True
    """
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("xs and ys must have the same length, at least 2")

    def _ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        ranks = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                ranks[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


# --- Confirming an accuracy drop (added 2026-10-20) ---------------------------
def calc_one_proportion_sample_size(null_rate, alternative_rate, z_alpha, z_beta):
    """Plan a one-sample binomial comparison of two rates.

    Outcomes a one-sample binomial test needs to tell ``alternative_rate`` from ``null_rate``.

    Normal approximation with the variance of each hypothesis on its own side:
    ``n = ((z_alpha sqrt(p0 (1 - p0)) + z_beta sqrt(p1 (1 - p1))) / (p0 - p1))^2``.
    The count grows as one over the squared gap, so halving the gap needs about
    four times the outcomes.

    Parameters
    ----------
    null_rate : float
        Null Bernoulli probability in (0, 1).
    alternative_rate : float
        Distinct alternative Bernoulli probability in (0, 1).
    z_alpha : float
        Dimensionless normal significance quantile; default 1.96 where specified.
    z_beta : float
        Dimensionless normal power quantile; default 0.84 where specified.

    Returns
    -------
    float
        Unrounded count; round upward for planning.

    Notes
    -----
    Equation: n = ``((z_alpha*sqrt(p0*(1-p0))+z_beta*sqrt(p1*(1-p1)))/(p0-p1))^2``.

    Assumptions: Independent Bernoulli observations and distinct interior hypothesis rates.

    Excludes: Exact binomial power and sequential sampling.

    References
    ----------
    Derivation: Balance normal null and alternative standard errors against the rate gap.

    Examples
    --------
    >>> calc_one_proportion_sample_size(0.5, 0.6, 2, 1) > 0
    True
    """
    if not (0 < null_rate < 1 and 0 < alternative_rate < 1):
        raise ValueError("rates must be in (0, 1).")
    if null_rate == alternative_rate:
        raise ValueError("the rates must differ.")
    spread = z_alpha * math.sqrt(null_rate * (1 - null_rate)) + z_beta * math.sqrt(
        alternative_rate * (1 - alternative_rate)
    )
    return (spread / (null_rate - alternative_rate)) ** 2
