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
    """
    Effective horizon of a discounted return, in decisions.

    Equation:
        H_eff = 1 / (1 - gamma)

    The sum of the discount weights gamma^j over all future decisions; a reward
    further away than a few H_eff decisions carries little weight.

    Source: Sutton and Barto (2018), Reinforcement Learning: An Introduction,
    2nd ed., Sec. 3.3 (the geometric series of discount weights).
    """
    if not 0.0 <= discount < 1.0:
        raise ValueError(f"discount must be in [0, 1), got {discount}")
    return 1.0 / (1.0 - discount)


def calc_discounted_weight(discount, decisions):
    """
    Weight of a reward ``decisions`` steps ahead in a discounted return: gamma^n.
    """
    if not 0.0 <= discount <= 1.0:
        raise ValueError(f"discount must be in [0, 1], got {discount}")
    validate_nonnegative(decisions, "decisions")
    return discount ** decisions


def calc_discount_for_retained_weight(decisions, retained_weight):
    """
    Smallest discount that keeps ``retained_weight`` of a reward ``decisions`` steps ahead.

    Equation:
        gamma_min = w^(1/n)
    """
    validate_positive(decisions, "decisions")
    if not 0.0 < retained_weight < 1.0:
        raise ValueError(f"retained_weight must be in (0, 1), got {retained_weight}")
    return retained_weight ** (1.0 / decisions)


def calc_trial_campaign_wall_time(trials, rollout, reset, rigs=1):
    """
    Wall-clock time of a hardware trial campaign.

    Equation:
        T_wall = N (t_rollout + t_reset) / N_rig

    Rigs divide the calendar; they do not divide the trials, so the wear the
    campaign puts on drivetrains is the same for any rig count.
    """
    validate_positive(trials, "trials")
    validate_positive(rollout, "rollout")
    validate_nonnegative(reset, "reset")
    validate_positive(rigs, "rigs")
    return (trials * (rollout + reset) / rigs).to(ureg.second)


def calc_iterative_decoder_latency(encode, per_evaluation, evaluations):
    """
    Latency of an action decoder that encodes once and then runs dependent network evaluations.

    Equation:
        t_infer = t_encode + N * t_eval
    """
    validate_nonnegative(encode, "encode")
    validate_positive(per_evaluation, "per_evaluation")
    validate_nonnegative(evaluations, "evaluations")
    return (encode + evaluations * per_evaluation).to(ureg.millisecond)


def calc_max_decoder_evaluations(budget, encode, per_evaluation):
    """
    Most dependent evaluations an iterative decoder can run inside ``budget``.

    Equation:
        N_max = floor((t_budget - t_encode) / t_eval)
    """
    validate_positive(per_evaluation, "per_evaluation")
    room = ((budget - encode) / per_evaluation).to("").magnitude
    return max(0, math.floor(room))


def calc_series_stiffness(stiffness_a, stiffness_b):
    """
    Effective stiffness of two springs in series (the same force passes through both).

    Equation:
        k_eff = (1/k_a + 1/k_b)^-1
    """
    validate_positive(stiffness_a, "stiffness_a")
    validate_positive(stiffness_b, "stiffness_b")
    k_a = stiffness_a.to(ureg.newton / ureg.meter).magnitude
    k_b = stiffness_b.to(ureg.newton / ureg.meter).magnitude
    return (1.0 / (1.0 / k_a + 1.0 / k_b)) * ureg.newton / ureg.meter


def calc_complementary_series_stiffness(series_stiffness, stiffness_a):
    """
    Stiffness that, in series with ``stiffness_a``, gives ``series_stiffness``.

    Equation:
        k_b = (1/k_eff - 1/k_a)^-1, defined only for k_eff < k_a.
    """
    validate_positive(series_stiffness, "series_stiffness")
    validate_positive(stiffness_a, "stiffness_a")
    k_eff = series_stiffness.to(ureg.newton / ureg.meter).magnitude
    k_a = stiffness_a.to(ureg.newton / ureg.meter).magnitude
    if k_eff >= k_a:
        raise ValueError("series stiffness must be below either spring's stiffness")
    return (1.0 / (1.0 / k_eff - 1.0 / k_a)) * ureg.newton / ureg.meter


def calc_impact_stiffness_for_force(force_limit, contact_velocity, effective_mass):
    """
    Contact stiffness at which an unbraked impact peaks exactly at ``force_limit``.

    Equation (inverse of F_peak = v sqrt(k m)):
        k* = (F_lim / v)^2 / m
    """
    validate_positive(force_limit, "force_limit")
    validate_positive(contact_velocity, "contact_velocity")
    validate_positive(effective_mass, "effective_mass")
    return ((force_limit / contact_velocity) ** 2 / effective_mass).to(ureg.newton / ureg.meter)


def calc_penalty_stiffness_from_penetration(effective_mass, contact_velocity, penetration):
    """
    Penalty-spring stiffness that lets a mass arriving at ``contact_velocity`` sink ``penetration``.

    Equation (kinetic energy equals spring strain energy at maximum penetration):
        k_pen = m v^2 / delta^2
    """
    validate_positive(effective_mass, "effective_mass")
    validate_positive(contact_velocity, "contact_velocity")
    validate_positive(penetration, "penetration")
    return (effective_mass * contact_velocity ** 2 / penetration ** 2).to(ureg.newton / ureg.meter)


def calc_contact_timescale(effective_mass, contact_stiffness):
    """
    Contact timescale sqrt(m / k) of a mass on a contact spring (the oscillation period over 2 pi).
    """
    validate_positive(effective_mass, "effective_mass")
    validate_positive(contact_stiffness, "contact_stiffness")
    return ((effective_mass / contact_stiffness) ** 0.5).to(ureg.millisecond)
