"""Policy training physics: discounting, trial campaigns, iterative decoders, and contact models.

Added 2026-10-20. Each function is one equation a robot-learning argument rests on: how far a
discount reaches in decisions, what a hardware trial campaign costs in wall-clock time, how long an
iterative action decoder takes, and how a soft simulated contact and a compliant arm in series with
a stiff plate set the peak force of an unbraked impact.
"""

from __future__ import annotations

import math

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_nonnegative, validate_positive


def calc_discount_effective_horizon(discount):
    """Sum the infinite sequence of discount weights.

    Parameters
    ----------
    discount : float
        Dimensionless discount gamma in [0, 1).

    Returns
    -------
    float
        Effective horizon in decision intervals.

    Notes
    -----
    Equation:
        H_eff = 1 / (1 - gamma)

    The sum of the discount weights gamma^j over all future decisions; a reward
    further away than a few H_eff decisions carries little weight.

    Assumptions: One constant geometric discount over an infinite sequence.

    Excludes: Episode termination and any guarantee about policy quality.

    References
    ----------
    Derivation: The convergent geometric series sum of ``gamma**j`` for j>=0 is 1/(1-gamma).

    Sutton and Barto (2018), Reinforcement Learning: An Introduction,
    2nd ed., Sec. 3.3 (the geometric series of discount weights).

    Examples
    --------
    >>> calc_discount_effective_horizon(0.5) == sum([1.0, 1.0])
    True
    """
    if not 0.0 <= discount < 1.0:
        raise ValueError(f"discount must be in [0, 1), got {discount}")
    return 1.0 / (1.0 - discount)


def calc_discounted_weight(discount, decisions):
    """Calculate the discount weight of a future reward.

    Parameters
    ----------
    discount : float
        Dimensionless gamma in [0, 1].
    decisions : float
        Nonnegative number of decision intervals; fractional exponents are permitted.

    Returns
    -------
    float
        Dimensionless retained reward weight.

    Notes
    -----
    Equation: w = ``gamma**decisions``.

    Assumptions: A stationary multiplicative discount; Python treats zero to the zeroth power as one.

    Excludes: Reward magnitudes, finite-episode truncation, and policy quality.

    References
    ----------
    Derivation: Repeated multiplication by gamma yields ``gamma**n`` after n decision intervals.

    Examples
    --------
    >>> calc_discounted_weight(0.5, 2) == 0.5 * 0.5
    True
    """
    if not 0.0 <= discount <= 1.0:
        raise ValueError(f"discount must be in [0, 1], got {discount}")
    validate_nonnegative(decisions, "decisions")
    return discount**decisions


def calc_discount_for_retained_weight(decisions, retained_weight):
    """Invert a retained reward weight to find the discount.

    Parameters
    ----------
    decisions : float
        Positive number of decision intervals.
    retained_weight : float
        Desired dimensionless retained weight in (0, 1).

    Returns
    -------
    float
        Minimum dimensionless discount for the requested weight.

    Notes
    -----
    Equation:
        gamma_min = w^(1/n)

    Assumptions: Constant discount over the specified positive horizon.

    Excludes: Whether this discount yields a useful learning objective.

    References
    ----------
    Derivation: Solve retained_weight = ``gamma**decisions`` for the positive root gamma.

    Examples
    --------
    >>> gamma = calc_discount_for_retained_weight(2, 0.25)
    >>> math.isclose(calc_discounted_weight(gamma, 2), 0.25)
    True
    """
    validate_positive(decisions, "decisions")
    if not 0.0 < retained_weight < 1.0:
        raise ValueError(f"retained_weight must be in (0, 1), got {retained_weight}")
    return retained_weight ** (1.0 / decisions)


def calc_trial_campaign_wall_time(trials, rollout, reset, rigs=1):
    """Calculate ideally parallelized rollout and reset time.

    Parameters
    ----------
    trials : float
        Positive number of trials; continuous work division is allowed.
    rollout : Quantity
        Positive per-trial rollout time.
    reset : Quantity
        Nonnegative per-trial reset time.
    rigs : float, optional
        Positive parallel rig count, default one; ideal continuous division is assumed.

    Returns
    -------
    Quantity
        Ideal campaign wall-clock time in seconds.

    Notes
    -----
    Equation:
        T_wall = N (t_rollout + t_reset) / N_rig

    Rigs divide the calendar; they do not divide the trials, so the wear the
    campaign puts on drivetrains is the same for any rig count.

    Assumptions: Identical independent trials and perfectly balanced work among rigs.

    Excludes: Integer assignment rounding, queueing, failures, maintenance, and hardware wear reduction.

    References
    ----------
    Derivation: Add serial rollout/reset time for each trial and divide total work by parallel capacity.

    Examples
    --------
    >>> one = calc_trial_campaign_wall_time(8, 2 * ureg.second, 1 * ureg.second)
    >>> two = calc_trial_campaign_wall_time(8, 2 * ureg.second, 1 * ureg.second, rigs=2)
    >>> bool(two * 2 == one)
    True
    """
    validate_positive(trials, "trials")
    validate_positive(rollout, "rollout")
    validate_nonnegative(reset, "reset")
    validate_positive(rigs, "rigs")
    return (trials * (rollout + reset) / rigs).to(ureg.second)


def calc_iterative_decoder_latency(encode, per_evaluation, evaluations):
    """Sum one encoding pass and serial decoder evaluations.

    Parameters
    ----------
    encode : Quantity
        Nonnegative one-time encoding duration.
    per_evaluation : Quantity
        Positive duration of one dependent decoder evaluation.
    evaluations : float
        Nonnegative evaluation count; fractional work estimates are accepted.

    Returns
    -------
    Quantity
        Total decoder latency in milliseconds.

    Notes
    -----
    Equation:
        t_infer = t_encode + N * t_eval

    Assumptions: The encoder runs once and evaluations execute serially with equal durations.

    Excludes: Overlap, variable evaluation costs, transport, and runtime contention.

    References
    ----------
    Derivation: Add the fixed encoder cost to repeated serial evaluation cost.

    Examples
    --------
    >>> base = calc_iterative_decoder_latency(2 * ureg.millisecond, 3 * ureg.millisecond, 0)
    >>> more = calc_iterative_decoder_latency(2 * ureg.millisecond, 3 * ureg.millisecond, 2)
    >>> bool(more - base == 2 * 3 * ureg.millisecond)
    True
    """
    validate_nonnegative(encode, "encode")
    validate_positive(per_evaluation, "per_evaluation")
    validate_nonnegative(evaluations, "evaluations")
    return (encode + evaluations * per_evaluation).to(ureg.millisecond)


def calc_max_decoder_evaluations(budget, encode, per_evaluation):
    """Count complete serial decoder evaluations that fit a budget.

    Parameters
    ----------
    budget : Quantity
        Available total inference time; a budget below encoding yields zero.
    encode : Quantity
        Fixed encoding duration in time units.
    per_evaluation : Quantity
        Positive duration of each dependent evaluation.

    Returns
    -------
    int
        Nonnegative count of complete evaluations that fit.

    Notes
    -----
    Equation:
        N_max = floor((t_budget - t_encode) / t_eval)

    Assumptions: Equal serial evaluation duration with the encoder charged once. Budget and encoding durations should be nonnegative; this function validates only positive per_evaluation.

    Excludes: Transport, contention, stochastic latency, and feasibility of encoding when zero is returned.

    References
    ----------
    Derivation: Subtract fixed work, divide remaining time by per-evaluation time, then retain only complete evaluations.

    Examples
    --------
    >>> count = calc_max_decoder_evaluations(10 * ureg.millisecond, 2 * ureg.millisecond, 3 * ureg.millisecond)
    >>> bool(calc_iterative_decoder_latency(2 * ureg.millisecond, 3 * ureg.millisecond, count) <= 10 * ureg.millisecond)
    True
    """
    validate_positive(per_evaluation, "per_evaluation")
    room = ((budget - encode) / per_evaluation).to("").magnitude
    return max(0, math.floor(room))


def calc_series_stiffness(stiffness_a, stiffness_b):
    """Combine two linear springs carrying the same force.

    Parameters
    ----------
    stiffness_a : Quantity
        Positive first spring stiffness in force per length.
    stiffness_b : Quantity
        Positive second spring stiffness in force per length.

    Returns
    -------
    Quantity
        Equivalent stiffness in newtons per meter.

    Notes
    -----
    Equation:
        k_eff = (1/k_a + 1/k_b)^-1

    Assumptions: Linear massless springs in series carrying a common force.

    Excludes: Damping, nonlinear compliance, preload, and dynamic resonances.

    References
    ----------
    Derivation: The extensions F/k_a and F/k_b add, so total extension is F/k_eff.

    Examples
    --------
    >>> k = 100 * ureg.newton / ureg.meter
    >>> bool(calc_series_stiffness(k, k) * 2 == k)
    True
    """
    validate_positive(stiffness_a, "stiffness_a")
    validate_positive(stiffness_b, "stiffness_b")
    k_a = stiffness_a.to(ureg.newton / ureg.meter).magnitude
    k_b = stiffness_b.to(ureg.newton / ureg.meter).magnitude
    return (1.0 / (1.0 / k_a + 1.0 / k_b)) * ureg.newton / ureg.meter


def calc_complementary_series_stiffness(series_stiffness, stiffness_a):
    """Find the second spring for a specified series stiffness.

    Parameters
    ----------
    series_stiffness : Quantity
        Positive desired equivalent stiffness, less than stiffness_a.
    stiffness_a : Quantity
        Positive known spring stiffness in force per length.

    Returns
    -------
    Quantity
        Required second spring stiffness in newtons per meter.

    Notes
    -----
    Equation:
        k_b = (1/k_eff - 1/k_a)^-1, defined only for k_eff < k_a.

    Assumptions: Linear springs in series and a target softer than the known spring.

    Excludes: Damping, nonlinear contact, and manufacturability.

    References
    ----------
    Derivation: Rearrange the sum of series compliances to solve for the missing compliance.

    Examples
    --------
    >>> k = 100 * ureg.newton / ureg.meter
    >>> bool(calc_complementary_series_stiffness(k / 2, k) == k)
    True
    """
    validate_positive(series_stiffness, "series_stiffness")
    validate_positive(stiffness_a, "stiffness_a")
    k_eff = series_stiffness.to(ureg.newton / ureg.meter).magnitude
    k_a = stiffness_a.to(ureg.newton / ureg.meter).magnitude
    if k_eff >= k_a:
        raise ValueError("series stiffness must be below either spring's stiffness")
    return (1.0 / (1.0 / k_eff - 1.0 / k_a)) * ureg.newton / ureg.meter


def calc_impact_stiffness_for_force(force_limit, contact_velocity, effective_mass):
    """Invert the ideal undamped impact peak-force relation.

    Parameters
    ----------
    force_limit : Quantity
        Positive target peak force.
    contact_velocity : Quantity
        Positive relative impact speed.
    effective_mass : Quantity
        Positive effective mass participating in contact.

    Returns
    -------
    Quantity
        Contact stiffness in newtons per meter that reaches the target peak.

    Notes
    -----
    Equation: k = ``(force_limit/contact_velocity)**2`` / effective_mass.

    Equation (inverse of F_peak = v sqrt(k m)):
        k* = (F_lim / v)^2 / m

    Assumptions: An undamped linear spring absorbs all initial kinetic energy without active braking.

    Excludes: Damping, plastic deformation, ongoing actuation, injury thresholds, and safety certification.

    References
    ----------
    Derivation: Equate mass kinetic energy and spring energy, then use force = stiffness times compression.

    Examples
    --------
    >>> k = calc_impact_stiffness_for_force(10 * ureg.newton, 1 * ureg.meter / ureg.second, 1 * ureg.kilogram)
    >>> import math
    >>> force = (k * 1 * ureg.kilogram)**0.5 * (1 * ureg.meter / ureg.second)
    >>> math.isclose(force.m_as('newton'), 10.0, rel_tol=1e-12)
    True
    """
    validate_positive(force_limit, "force_limit")
    validate_positive(contact_velocity, "contact_velocity")
    validate_positive(effective_mass, "effective_mass")
    return ((force_limit / contact_velocity) ** 2 / effective_mass).to(ureg.newton / ureg.meter)


def calc_penalty_stiffness_from_penetration(effective_mass, contact_velocity, penetration):
    """Calculate a linear contact stiffness from impact penetration.

    Parameters
    ----------
    effective_mass : Quantity
        Positive effective impact mass.
    contact_velocity : Quantity
        Positive relative speed at initial contact.
    penetration : Quantity
        Positive maximum compression distance.

    Returns
    -------
    Quantity
        Penalty-spring stiffness in newtons per meter.

    Notes
    -----
    Equation: k = effective_mass ``contact_velocity**2`` / ``penetration**2``.

    Equation (kinetic energy equals spring strain energy at maximum penetration):
        k_pen = m v^2 / delta^2

    Assumptions: All initial kinetic energy becomes elastic energy at maximum compression.

    Excludes: Damping, nonlinear tissue/material response, continued drive work, and safety certification.

    References
    ----------
    Derivation: Set one-half mass times speed squared equal to one-half stiffness times compression squared.

    Examples
    --------
    >>> m, v, x = 2 * ureg.kilogram, 1 * ureg.meter / ureg.second, 0.1 * ureg.meter
    >>> k = calc_penalty_stiffness_from_penetration(m, v, x)
    >>> math.isclose((k * x**2 / (m * v**2)).m_as(""), 1.0)
    True
    """
    validate_positive(effective_mass, "effective_mass")
    validate_positive(contact_velocity, "contact_velocity")
    validate_positive(penetration, "penetration")
    return (effective_mass * contact_velocity**2 / penetration**2).to(ureg.newton / ureg.meter)


def calc_contact_timescale(effective_mass, contact_stiffness):
    """Calculate the characteristic time of a mass-spring contact.

    Parameters
    ----------
    effective_mass : Quantity
        Positive effective oscillating mass.
    contact_stiffness : Quantity
        Positive linear contact stiffness in force per length.

    Returns
    -------
    Quantity
        Characteristic time in milliseconds; oscillation period divided by 2 pi.

    Notes
    -----
    Equation: tau = sqrt(effective_mass/contact_stiffness).

    Assumptions: An undamped linear mass-spring oscillator.

    Excludes: Damping, forcing, nonlinear contact, and a guaranteed contact duration.

    References
    ----------
    Derivation: The natural angular frequency is sqrt(k/m); its reciprocal is the characteristic time.

    Examples
    --------
    >>> m = 1 * ureg.kilogram
    >>> k = 100 * ureg.newton / ureg.meter
    >>> bool(calc_contact_timescale(m, k * 4) * 2 == calc_contact_timescale(m, k))
    True
    """
    validate_positive(effective_mass, "effective_mass")
    validate_positive(contact_stiffness, "contact_stiffness")
    return ((effective_mass / contact_stiffness) ** 0.5).to(ureg.millisecond)
