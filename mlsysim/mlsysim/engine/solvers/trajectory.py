"""Trajectory-planning solvers: joint rates, handoff deadlines, seams, swings, and jerk-limited stops.

Domain implementations behind ``mlsysim.solvers`` (the public import path,
derived from ``engine.solvers.__init__``). The equations live in
``physics.robotics`` (the two-link Jacobian and damped least squares, the
S-curve and polynomial stops, the seam torque step, the spill bound); these
models compose them for one stated arm, joint, base, or mission, so a chapter,
a slide deck, and a lab read the same arithmetic (added 2026-10-20).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..results import (
    BrakingBlendResult,
    ChunkReplacementResult,
    DynamicWindowResult,
    HandoffExposureResult,
    JerkLimitedStopResult,
    SeamBridgeResult,
    ShoulderSwingResult,
    SpillLimitedStopResult,
    StaleAnchorResult,
    TrajectoryDeadlinesResult,
    TwoLinkJointRateResult,
)
from ...core.types import Quantity
from ...core.units import ureg
from ...physics.agents import calc_unchecked_success
from ...physics.robotics.actuation import calc_quasistatic_spill_acceleration, calc_seam_acceleration_torque_jump
from ...physics.robotics.actuation import calc_smoothstep_blend_window
from ...physics.robotics.kinematics import calc_dls_joint_rate, calc_two_link_singular_values
from ...physics.robotics.stopping import (
    calc_c2_stop_suffix,
    calc_scurve_max_permitted_velocity,
    calc_scurve_rest_to_rest_time,
    calc_scurve_stop_distance,
    calc_scurve_stop_time,
    calc_blend_reserve_use,
    calc_smoothstep_braking_blend,
)
from .base import ForwardModel

_MS = ureg.millisecond
_MM = ureg.millimeter
_RAD_S = ureg.radian / ureg.second
_RAD_S2 = ureg.radian / ureg.second ** 2
_RAD_S3 = ureg.radian / ureg.second ** 3
_NM = ureg.newton * ureg.meter
_KGM2 = ureg.kilogram * ureg.meter ** 2


def _ratio(a, b) -> float:
    return float((a / b).to("").magnitude)


class TwoLinkJointRateModel(ForwardModel):
    """
    Joint rates a tool speed asks of a planar two-link arm, exact and damped.

    The weakest direction of the Jacobian at elbow angle q2 has singular value
    sigma_min; a tool speed v along it needs a joint-rate vector of norm
    v / sigma_min, and with two joints at least one component is that norm
    divided by sqrt(2). Damped least squares with damping lambda commands
    v sigma / (sigma^2 + lambda^2), never more than v / (2 lambda), and
    delivers the fraction sigma^2 / (sigma^2 + lambda^2) of the tool motion.
    """

    requires = ("arm", "elbow_angle", "tool_speed")
    produces = TwoLinkJointRateResult

    def solve(self, upper_arm: Quantity, forearm: Quantity, elbow_angle: Quantity, tool_speed: Quantity,
              joint_speed_limit: Quantity, damping: Quantity) -> TwoLinkJointRateResult:
        """Compare exact and damped joint rates in the weakest tool direction.

        Parameters
        ----------
        upper_arm : Quantity
            Positive upper-link length, convertible to meters.
        forearm : Quantity
            Positive lower-link length, convertible to meters.
        elbow_angle : Quantity
            Elbow configuration, convertible to radians; avoid a singular pose when
            calculating the exact rate.
        tool_speed : Quantity
            Nonnegative tool-speed magnitude in meters per second along the weakest singular
            direction.
        joint_speed_limit : Quantity
            Positive allowed joint rate in radians per second.
        damping : Quantity
            Positive damped-least-squares coefficient, in meters under the Jacobian convention.

        Returns
        -------
        TwoLinkJointRateResult
            Jacobian singular values in meters, joint rates in rad/s, realized motion fraction,
            and rate-limit comparison flags. These flags do not certify reachability of an
            arbitrary trajectory.

        See Also
        --------
        mlsysim.physics.robotics.kinematics.calc_two_link_singular_values : Compute the two-link
            Jacobian singular values.
        mlsysim.physics.robotics.kinematics.calc_dls_joint_rate : Apply the scalar damped
            inverse along a singular direction.

        Notes
        -----
        Equation: For the smaller Jacobian singular value sigma, the exact joint-rate norm is
        v/sigma; at least one of two components is at least that norm/sqrt(2). The damped rate
        is ``v*sigma/(sigma**2`` + ``damping**2)``.

        Assumptions: A rigid planar two-link arm; motion lies in its weakest singular direction.
        Damping is fixed and positive.

        Excludes: Joint acceleration, torque, obstacles, tracking dynamics, and other
        tool-motion directions.

        References
        ----------
        Derivation: The singular-value decomposition reduces inverse kinematics along one
        direction to scalar division. Adding a squared-rate penalty replaces 1/sigma with
        ``sigma/(sigma**2`` + ``damping**2)``.

        Examples
        --------
        >>> from mlsysim.solvers import TwoLinkJointRateModel
        >>> from mlsysim import Q_
        >>> result = TwoLinkJointRateModel().solve(
        ...     upper_arm=Q_('0.5 m'),
        ...     forearm=Q_('0.5 m'),
        ...     elbow_angle=Q_('90 deg'),
        ...     tool_speed=Q_('0.1 m/s'),
        ...     joint_speed_limit=Q_('2 rad/s'),
        ...     damping=Q_('0.05 m'),
        ... )
        >>> 0 < result.realized_fraction < 1
        True
        """
        s_max, s_min = calc_two_link_singular_values(upper_arm, forearm, elbow_angle)
        norm = (tool_speed / s_min).to(1 / ureg.second).magnitude * _RAD_S
        one = norm / math.sqrt(2)
        damped, realized = calc_dls_joint_rate(tool_speed, s_min, damping)
        ceiling = (tool_speed / (2 * damping)).to(1 / ureg.second).magnitude * _RAD_S
        return TwoLinkJointRateResult(
            reach=(upper_arm + forearm).to(ureg.meter), sigma_max=s_max, sigma_min=s_min,
            joint_rate_norm=norm, joint_rate_one_joint=one, exceeds_joint_limit=bool(one > joint_speed_limit),
            damped_joint_rate=damped.to(_RAD_S), realized_fraction=realized, damped_ceiling=ceiling,
            damped_exceeds_joint_limit=bool(damped > joint_speed_limit),
        )


class HandoffExposureModel(ForwardModel):
    """
    A per-handoff deadline budget read across a mission of K handoffs.

    With independent misses at probability p, all K handoffs are on time with
    probability (1 - p)^K and p K are late on average. Holding a mission budget
    eps_M by the union bound leaves each handoff eps_M / K, the quantile
    1 - eps_M / K, deeper than p by the ratio p K / eps_M.
    """

    requires = ("mission_time", "replan_rate", "late_probability", "mission_budget")
    produces = HandoffExposureResult

    def solve(self, mission_time: Quantity, replan_rate: Quantity, late_probability: float,
              mission_budget: float) -> HandoffExposureResult:
        """Translate repeated deadline misses into a mission exposure budget.

        Parameters
        ----------
        mission_time : Quantity
            Positive mission duration, convertible to seconds.
        replan_rate : Quantity
            Positive handoff frequency in hertz; duration times rate must round to at least one
            handoff.
        late_probability : float
            Per-handoff miss probability in (0, 1]; positive because reciprocal and logarithm
            are reported.
        mission_budget : float
            Positive mission-level miss-probability budget, normally at most one.

        Returns
        -------
        HandoffExposureResult
            Handoff count, expected misses, dimensionless independent all-on-time probability,
            and the union-bound per-handoff budget and quantile.

        See Also
        --------
        mlsysim.physics.agents.calc_unchecked_success : Multiply independent per-event success
            probabilities.

        Notes
        -----
        Equation: K = round(mission_time * replan_rate), expected_late = ``K*p``, all_on_time =
        ``(1-p)**K``, and per_handoff_budget = mission_budget/K.

        Assumptions: The all-on-time expression assumes independent identical miss
        probabilities. The budget allocation uses a union bound and does not require
        independence.

        Excludes: Correlated-tail estimation, empirical deadline distributions, recovery, and a
        mission safety guarantee.

        References
        ----------
        Derivation: Linearity of expectation gives ``K*p``. Multiplying independent on-time
        probabilities gives ``(1-p)**K``; summing event probabilities bounds any miss by ``K*p``.

        Examples
        --------
        >>> from mlsysim.solvers import HandoffExposureModel
        >>> from mlsysim import Q_
        >>> result = HandoffExposureModel().solve(Q_('10 s'), Q_('2 Hz'), 0.01, 0.1)
        >>> result.handoffs == 20 and result.design_quantile > 0.9
        True
        """
        k = round((mission_time * replan_rate).to("").magnitude)
        budget = mission_budget / k
        depth = late_probability / budget
        return HandoffExposureResult(
            handoffs=k, expected_late=k * late_probability,
            all_on_time=calc_unchecked_success(1 - late_probability, k),
            per_handoff_budget=budget, design_quantile=1 - budget, depth_ratio=depth,
            orders_deeper=math.log10(depth), handoffs_per_expected_late=1 / late_probability,
        )


class ChunkReplacementModel(ForwardModel):
    """
    Replacement latency, capture to admission, against the renewal period and the lease.

    L = observation age at dispatch + the dispatch-to-admission allocation; the
    inference pass and whole-plan validation sit inside the allocation. Admitting
    one plan every renewal period with L longer than that period keeps
    ceil(L / T_renew) requests in flight.
    """

    requires = ("observation_age", "dispatch_to_admission", "renewal_period")
    produces = ChunkReplacementResult

    def solve(self, observation_age: Quantity, dispatch_to_admission: Quantity, inference: Quantity,
              validation: Quantity, renewal_period: Quantity, lease: Quantity,
              interference_tail: Quantity) -> ChunkReplacementResult:
        """Account for replacement latency and concurrent renewal requests.

        Parameters
        ----------
        observation_age : Quantity
            Nonnegative age of evidence at dispatch, in time units.
        dispatch_to_admission : Quantity
            Nonnegative total allocation from dispatch to admission, in time units.
        inference : Quantity
            Positive inference duration within the dispatch allocation; used as a comparison
            denominator.
        validation : Quantity
            Nonnegative validation duration within the dispatch allocation.
        renewal_period : Quantity
            Positive interval between replacement requests.
        lease : Quantity
            Nonnegative duration for which an admitted chunk remains authorized.
        interference_tail : Quantity
            Nonnegative interference-tail duration to compare with inference.

        Returns
        -------
        ChunkReplacementResult
            Latency, unassigned-stage time and lease overrun in milliseconds; latency ratios and
            the ceiling of concurrent requests. Negative residual time exposes an inconsistent
            allocation.

        Notes
        -----
        Equation: L = observation_age + dispatch_to_admission; requests_in_flight =
        ceil(L/renewal_period). Other stages equal the allocation minus inference and
        validation.

        Assumptions: Requests arrive periodically and experience the stated fixed
        capture-to-admission latency.

        Excludes: Queue dynamics, probabilistic interference, admission rejection, and
        validation of the supplied timing allocation.

        References
        ----------
        Derivation: Add serial stage durations; during a latency interval L, a periodic
        requester can have ceil(L/T) requests outstanding.

        Examples
        --------
        >>> from mlsysim.solvers import ChunkReplacementModel
        >>> from mlsysim import Q_
        >>> result = ChunkReplacementModel().solve(
        ...     observation_age=Q_('5 ms'),
        ...     dispatch_to_admission=Q_('15 ms'),
        ...     inference=Q_('10 ms'),
        ...     validation=Q_('2 ms'),
        ...     renewal_period=Q_('10 ms'),
        ...     lease=Q_('30 ms'),
        ...     interference_tail=Q_('3 ms'),
        ... )
        >>> result.requests_in_flight
        2
        """
        latency = (observation_age + dispatch_to_admission).to(_MS)
        over = _ratio(latency, renewal_period)
        return ChunkReplacementResult(
            latency=latency, other_stages=(dispatch_to_admission - inference - validation).to(_MS),
            latency_over_renewal=over, requests_in_flight=math.ceil(over),
            latency_beyond_lease=(latency - lease).to(_MS),
            tail_over_inference=_ratio(interference_tail, inference),
        )


class JerkLimitedStopModel(ForwardModel):
    """
    A delayed jerk-limited (S-curve) stop, beside the constant and polynomial stops it replaces.

    The machine travels ``v tau`` during the pre-brake delay, then stops along
    the double-S profile with deceleration bound a and jerk bound j
    (``calc_scurve_stop_distance``): above the plateau speed a^2 / j the stop is
    v^2 / (2 a) of braking plus v a / (2 j) of cruise over half a jerk ramp.
    Without a stated jerk bound, j is the peak jerk of the reference C2
    polynomial stop (``calc_c2_stop_suffix``) at ``jerk_reference_speed`` (the
    entry speed by default), which stops in 0.75 v^2 / a. A localization bound
    and a protective margin, when stated, add to every required distance; the
    speed ceiling is the fastest entry speed whose delayed S-curve stop fits,
    with the jerk bound held fixed (``calc_scurve_max_permitted_velocity``).
    """

    requires = ("speed", "deceleration")
    produces = JerkLimitedStopResult

    def solve(self, speed: Quantity, deceleration: Quantity, jerk: Optional[Quantity] = None,
              jerk_reference_speed: Optional[Quantity] = None, delay: Optional[Quantity] = None,
              clear_distance: Optional[Quantity] = None, loc_margin: Optional[Quantity] = None,
              physical_margin: Optional[Quantity] = None) -> JerkLimitedStopResult:
        """Compare a delayed S-curve stop with constant and polynomial braking.

        Parameters
        ----------
        speed : Quantity
            Positive initial speed, in meters per second.
        deceleration : Quantity
            Positive magnitude of maximum braking acceleration, in meters per second squared.
        jerk : Quantity, optional
            Positive jerk bound in meters per second cubed; None derives it from the C2
            polynomial reference stop.
        jerk_reference_speed : Quantity, optional
            Positive speed used only to derive omitted jerk; None uses speed.
        delay : Quantity, optional
            Nonnegative pre-brake duration; None means zero.
        clear_distance : Quantity, optional
            Available travel distance; None omits fit and speed-ceiling calculations.
        loc_margin : Quantity, optional
            Nonnegative localization-distance allowance; None means zero.
        physical_margin : Quantity, optional
            Nonnegative protective distance; None means zero.

        Returns
        -------
        JerkLimitedStopResult
            Braking distances in millimeters, times in milliseconds, jerk and acceleration
            bounds, and optional spare distance, fit flag and speed ceiling. Fit refers only to
            the supplied deterministic budget.

        See Also
        --------
        mlsysim.physics.robotics.stopping.calc_scurve_stop_distance : Integrate the bounded-jerk
            stopping profile.
        mlsysim.physics.robotics.stopping.calc_c2_stop_suffix : Construct the reference
            polynomial stopping profile.

        Notes
        -----
        Equation: Required distance is ``speed*delay`` + S_curve_distance(speed, deceleration, jerk)
        + margins. The plateau regime begins at speed = ``deceleration**2/jerk``.

        Assumptions: One-dimensional braking with a fixed positive deceleration and jerk limit,
        zero initial acceleration, and constant speed during delay.

        Excludes: Slope, slip, changing traction, sensing distributions, actuator faults, and
        safety certification.

        References
        ----------
        Derivation: Integrating the piecewise linear acceleration of the S-curve gives its
        travel and duration. Adding delay travel and explicit distance allowances forms the
        stopping budget.

        Examples
        --------
        >>> from mlsysim.solvers import JerkLimitedStopModel
        >>> from mlsysim import Q_
        >>> result = JerkLimitedStopModel().solve(
        ...     speed=Q_('1 m/s'),
        ...     deceleration=Q_('2 m/s^2'),
        ...     jerk=Q_('4 m/s^3'),
        ...     clear_distance=Q_('2 m'),
        ... )
        >>> result.fits and result.scurve_travel > result.braking_travel
        True
        """
        m = ureg.meter
        v_ref = speed if jerk_reference_speed is None else jerk_reference_speed
        j = calc_c2_stop_suffix(v_ref, deceleration)["peak_jerk"] if jerk is None else jerk
        j = j.to(m / ureg.second ** 3)
        a = deceleration.to(m / ureg.second ** 2)
        tau = 0 * _MS if delay is None else delay.to(_MS)
        plateau = (a ** 2 / j).to(m / ureg.second)
        braking = (speed ** 2 / (2 * a)).to(_MM)
        stop = calc_scurve_stop_distance(speed, a, j).to(_MM)
        t_stop = calc_scurve_stop_time(speed, a, j).to(_MS)
        poly = calc_c2_stop_suffix(speed, a)
        delay_travel = (speed * tau).to(_MM)
        loc = 0 * m if loc_margin is None else loc_margin
        margin = 0 * m if physical_margin is None else physical_margin
        overhead = (loc + margin).to(_MM)
        required = (delay_travel + stop + overhead).to(_MM)
        increment = (stop - braking).to(_MM)
        out = dict(
            speed=speed, deceleration=a, jerk=j, jerk_plateau_speed=plateau,
            reaches_deceleration_bound=bool(speed >= plateau),
            braking_travel=braking, braking_time=(speed / a).to(_MS),
            jerk_ramp_travel=increment, ramp_time=(a / j).to(_MS), ramp_delay=(a / (2 * j)).to(_MS),
            scurve_travel=stop, stop_time=t_stop,
            polynomial_distance=poly["distance"].to(_MM), polynomial_time=poly["duration"].to(_MS),
            delay=tau, delay_travel=delay_travel, stop_distance=(delay_travel + stop).to(_MM), overhead=overhead,
            constant_required=(delay_travel + braking + overhead).to(_MM), required_distance=required,
            polynomial_required=(delay_travel + poly["distance"] + overhead).to(_MM),
            rest_time=(tau + t_stop).to(_MS), delay_share_of_stop=_ratio(delay_travel, stop),
        )
        if clear_distance is not None:
            spare = (clear_distance - required).to(_MM)
            out.update(clear_distance=clear_distance, spare=spare, spare_time=(spare / speed).to(_MS),
                       fits=bool(spare.magnitude >= 0),
                       speed_ceiling=calc_scurve_max_permitted_velocity(
                           clear_distance, tau, a, j, loc, margin).to(m / ureg.second))
        return JerkLimitedStopResult(**out)


class SpillLimitedStopModel(ForwardModel):
    """
    The stop of a machine carrying an open container, whose liquid caps the deceleration.

    Under steady deceleration a the free surface tilts until the liquid rises
    R a / g at the wall; keeping it below the rim's freeboard h_f caps the
    deceleration at g h_f / R (``calc_quasistatic_spill_acceleration``). The
    stop at that bound keeps the stop's shape, so its jerk bound is rescaled
    with the deceleration (the reference polynomial's peak at the entry speed);
    the carry speed is the fastest entry speed whose delayed stop fits.
    """

    requires = ("free_stop", "inner_radius", "freeboard")
    produces = SpillLimitedStopResult

    def solve(self, speed: Quantity, free_deceleration: Quantity, free_jerk: Quantity, delay: Quantity,
              clear_distance: Quantity, inner_radius: Quantity, freeboard: Quantity) -> SpillLimitedStopResult:
        """Compute a stopping budget under a quasistatic spill constraint.

        Parameters
        ----------
        speed : Quantity
            Positive carrying speed in meters per second.
        free_deceleration : Quantity
            Positive unconstrained braking deceleration in meters per second squared.
        free_jerk : Quantity
            Positive reference jerk bound in meters per second cubed, used for the reused-jerk
            comparison.
        delay : Quantity
            Nonnegative pre-brake delay.
        clear_distance : Quantity
            Positive available travel distance.
        inner_radius : Quantity
            Positive horizontal radius of the open container.
        freeboard : Quantity
            Positive vertical liquid-to-rim clearance.

        Returns
        -------
        SpillLimitedStopResult
            Spill-limited deceleration, predicted liquid rise in millimeters, nested
            jerk-limited stop, distance overrun and admissible carrying speed. The spill flag is a
            quasistatic comparison.

        See Also
        --------
        mlsysim.physics.robotics.actuation.calc_quasistatic_spill_acceleration : Convert
            container freeboard into an acceleration bound.
        JerkLimitedStopModel.solve : Compose delay, braking travel, and explicit distance
            allowances.

        Notes
        -----
        Equation: Allowed deceleration is ``g*freeboard/inner_radius``; rise under unconstrained
        braking is ``inner_radius*free_deceleration/g``. The constrained stop recomputes the
        polynomial-derived jerk at that deceleration.

        Assumptions: A level open container, quasistatic free surface, fixed geometry and
        one-dimensional stopping.

        Excludes: Transient sloshing, surface tension, container tilt, active stabilization, and
        a guarantee against spilling.

        References
        ----------
        Derivation: In the accelerating frame the free-surface slope is acceleration/g.
        Multiplying the slope by radius gives rim rise and solving the freeboard inequality
        gives the acceleration ceiling.

        Examples
        --------
        >>> from mlsysim.solvers import SpillLimitedStopModel
        >>> from mlsysim import Q_
        >>> result = SpillLimitedStopModel().solve(
        ...     speed=Q_('1 m/s'),
        ...     free_deceleration=Q_('2 m/s^2'),
        ...     free_jerk=Q_('4 m/s^3'),
        ...     delay=Q_('10 ms'),
        ...     clear_distance=Q_('2 m'),
        ...     inner_radius=Q_('0.1 m'),
        ...     freeboard=Q_('0.01 m'),
        ... )
        >>> result.spill_deceleration < Q_('2 m/s^2')
        True
        """
        a_spill = calc_quasistatic_spill_acceleration(inner_radius, freeboard)
        rise = (inner_radius * free_deceleration / (1 * ureg.standard_gravity)).to(_MM)
        model = JerkLimitedStopModel()
        stop = model.solve(speed, a_spill, delay=delay, clear_distance=clear_distance)
        reused = calc_scurve_stop_distance(speed, a_spill, free_jerk).to(_MM)
        return SpillLimitedStopResult(
            spill_deceleration=a_spill, rise_at_free_deceleration=rise,
            spills_at_free_deceleration=bool(rise > freeboard), stop=stop,
            reused_jerk_stop_distance=reused, overrun=(-stop.spare).to(_MM),
            carry_speed=stop.speed_ceiling,
        )


class TrajectoryDeadlinesModel(ForwardModel):
    """
    The deadlines a trajectory record carries, and the cruise charged before its stop begins.

    A replacement requested ``request_delay`` after the schedule origin, with a
    design-tail latency target and scheduling slack, is admitted at the preferred
    admission; the full reserve adds the decision-to-onset response. The matched
    stop must physically begin at the commitment instant, so admission closes one
    response earlier; the arm rests one stop time after commitment. Cruise from
    the blend deadline to commitment is charged before the stop.
    """

    requires = ("setpoint_period", "horizon", "commit", "stop")
    produces = TrajectoryDeadlinesResult

    def solve(self, setpoint_period: Quantity, horizon: int, renewal_period: Quantity, request_delay: Quantity,
              latency_target: Quantity, schedule_slack: Quantity, response: Quantity, blend: Quantity,
              commit: Quantity, stop_time: Quantity, stop_distance: Quantity, speed: Quantity,
              clear_distance: Quantity, late_wait: Quantity) -> TrajectoryDeadlinesResult:
        """Place admission, commitment, and rest on a common trajectory timeline.

        Parameters
        ----------
        setpoint_period : Quantity
            Positive duration between successive setpoints.
        horizon : int
            Positive count of setpoints in the nominal chunk.
        renewal_period : Quantity
            Positive replacement-request cadence.
        request_delay : Quantity
            Nonnegative request instant relative to the schedule origin.
        latency_target : Quantity
            Nonnegative request-to-admission design-tail latency allocation.
        schedule_slack : Quantity
            Nonnegative scheduling reserve.
        response : Quantity
            Nonnegative decision-to-brake-onset response duration.
        blend : Quantity
            Blend-completion instant relative to the same schedule origin.
        commit : Quantity
            Stop-commitment instant, normally no earlier than blend.
        stop_time : Quantity
            Nonnegative duration from brake onset to rest.
        stop_distance : Quantity
            Nonnegative travel during the matched stop.
        speed : Quantity
            Nonnegative cruise speed before stop commitment.
        clear_distance : Quantity
            Available travel distance measured at the blend deadline.
        late_wait : Quantity
            Nonnegative extra wait after a missed admission deadline.

        Returns
        -------
        TrajectoryDeadlinesResult
            Deadlines, timing reserves and rest instant in milliseconds; cruise travel, consumed
            distance, remaining margin and late breach in millimeters. Negative margins identify
            budget overruns.

        Notes
        -----
        Equation: Preferred admission = request_delay + latency_target + schedule_slack; cutoff
        = commit-response. Used distance = ``speed*(commit-blend)`` + stop_distance.

        Assumptions: All instants share one origin, cruise speed is constant until commitment,
        and stop_time and stop_distance describe the same stop.

        Excludes: Trajectory generation, stochastic latency estimation, obstacle motion, and
        enforcement of deadline ordering.

        References
        ----------
        Derivation: Subtract the onset response from the commitment instant to obtain the last
        admission instant. Integrate constant cruise speed over commit-blend, then add the
        supplied stop distance.

        Examples
        --------
        >>> from mlsysim.solvers import TrajectoryDeadlinesModel
        >>> from mlsysim import Q_
        >>> result = TrajectoryDeadlinesModel().solve(
        ...     setpoint_period=Q_('1 ms'),
        ...     horizon=100,
        ...     renewal_period=Q_('10 ms'),
        ...     request_delay=Q_('10 ms'),
        ...     latency_target=Q_('20 ms'),
        ...     schedule_slack=Q_('5 ms'),
        ...     response=Q_('5 ms'),
        ...     blend=Q_('10 ms'),
        ...     commit=Q_('50 ms'),
        ...     stop_time=Q_('100 ms'),
        ...     stop_distance=Q_('100 mm'),
        ...     speed=Q_('1 m/s'),
        ...     clear_distance=Q_('200 mm'),
        ...     late_wait=Q_('10 ms'),
        ... )
        >>> result.margin > Q_('0 mm')
        True
        """
        preferred = (request_delay + latency_target + schedule_slack).to(_MS)
        reserve = (schedule_slack + response).to(_MS)
        rest = (commit + stop_time).to(_MS)
        cruise = (speed * (commit - blend)).to(_MM)
        used = (cruise + stop_distance).to(_MM)
        margin = (clear_distance - used).to(_MM)
        late = (speed * late_wait).to(_MM)
        return TrajectoryDeadlinesResult(
            nominal_end=(horizon * setpoint_period).to(_MS), setpoint_rate=(1 / setpoint_period).to(ureg.hertz),
            executed_setpoints=_ratio(renewal_period, setpoint_period),
            request_slack=(request_delay - renewal_period).to(_MS), preferred_admission=preferred,
            reserve=reserve, minimum_onset=(request_delay + latency_target + reserve).to(_MS),
            admission_cutoff=(commit - response).to(_MS), commit_slack=(commit - preferred).to(_MS),
            rest=rest, window_periods=_ratio(rest, setpoint_period), blend_cruise=cruise, used=used,
            margin=margin, late_cruise=late, late_breach=(late - margin).to(_MM),
        )


class SeamBridgeModel(ForwardModel):
    """
    A velocity gap at a chunk seam, closed in one servo tick or by a quintic bridge.

    Closing a gap of delta in one tick asks the reflected rotor inertia
    N^2 J_rotor for J delta / dt of torque (``calc_seam_acceleration_torque_jump``),
    reached in one tick, a torque rate of J delta / dt^2. The C2 bridge over T,
    with zero boundary accelerations and an endpoint on the incoming reference,
    is the quintic q(t) = a3 t^3 + a4 t^4 + a5 t^5 (solved from its 3x3 boundary
    system); its acceleration peaks at 1.5 delta / T mid-bridge and its jerk at
    6 delta / T^2 at both ends, so torque falls as 1 / T and torque rate as 1 / T^2.
    """

    requires = ("joint", "velocity_gap", "bridge_duration")
    produces = SeamBridgeResult

    def solve(self, gear_ratio: float, rotor_inertia: Quantity, rated_torque: Quantity, peak_torque: Quantity,
              torque_rate_limit: Quantity, velocity_gap: Quantity, tick: Quantity, bridge_duration: Quantity,
              loaded_inertia: Optional[Quantity] = None) -> SeamBridgeResult:
        """Compare a one-tick velocity correction with a quintic seam bridge.

        Parameters
        ----------
        gear_ratio : float
            Positive output-to-rotor speed ratio used to reflect rotor inertia.
        rotor_inertia : Quantity
            Positive rotor inertia in kilogram meters squared.
        rated_torque : Quantity
            Positive continuous torque rating in newton meters.
        peak_torque : Quantity
            Positive peak torque rating used for the one-tick velocity ceiling.
        torque_rate_limit : Quantity
            Positive permitted torque slew in newton meters per second.
        velocity_gap : Quantity
            Positive joint-rate gap in radians per second.
        tick : Quantity
            Positive servo period.
        bridge_duration : Quantity
            Positive duration of the quintic correction.
        loaded_inertia : Quantity, optional
            Optional positive total load-side inertia for an additional one-tick comparison;
            None omits it.

        Returns
        -------
        SeamBridgeResult
            Reflected inertia, torques, torque rates, polynomial coefficients and
            bridge-duration bound; rate/rating comparison flags refer only to the modeled inertial
            correction.

        See Also
        --------
        mlsysim.physics.robotics.actuation.calc_seam_acceleration_torque_jump : Convert a joint
            acceleration jump into reflected-inertia torque.

        Notes
        -----
        Equation: J = ``gear_ratio**2*rotor_inertia``; step torque = ``J*velocity_gap/tick``. The bridge
        peaks at ``1.5*J*velocity_gap/T`` in torque and ``6*J*velocity_gap/T**2`` in torque rate.

        Assumptions: Zero correction acceleration at both endpoints, fixed reflected inertia and
        a quintic endpoint matching velocity gap and half-gap displacement.

        Excludes: Gravity, friction, elasticity, actuator bandwidth, saturation response, and
        full loaded dynamics unless loaded_inertia is supplied.

        References
        ----------
        Derivation: Solve the three endpoint equations for the cubic, quartic and quintic
        coefficients, differentiate to acceleration and jerk, then multiply by reflected
        inertia.

        Examples
        --------
        >>> from mlsysim.solvers import SeamBridgeModel
        >>> from mlsysim import Q_
        >>> result = SeamBridgeModel().solve(
        ...     gear_ratio=10,
        ...     rotor_inertia=Q_('0.001 kg*m^2'),
        ...     rated_torque=Q_('10 N*m'),
        ...     peak_torque=Q_('20 N*m'),
        ...     torque_rate_limit=Q_('1000 N*m/s'),
        ...     velocity_gap=Q_('0.1 rad/s'),
        ...     tick=Q_('1 ms'),
        ...     bridge_duration=Q_('20 ms'),
        ... )
        >>> result.peak_torque < result.step_torque
        True
        """
        j_refl = (gear_ratio ** 2 * rotor_inertia).to(_KGM2)
        step = calc_seam_acceleration_torque_jump(gear_ratio, rotor_inertia, (velocity_gap / tick).to(_RAD_S2))
        step = step.to(_NM)
        step_rate = (step / tick).to(_NM / ureg.second)
        mean = calc_seam_acceleration_torque_jump(gear_ratio, rotor_inertia,
                                                 (velocity_gap / bridge_duration).to(_RAD_S2)).to(_NM)
        t = bridge_duration.to(ureg.second).magnitude
        dw = velocity_gap.to(_RAD_S).magnitude
        a_mat = np.array([[t ** 3, t ** 4, t ** 5], [3 * t ** 2, 4 * t ** 3, 5 * t ** 4], [6 * t, 12 * t ** 2, 20 * t ** 3]])
        a3, a4, a5 = (float(c) for c in np.linalg.solve(a_mat, np.array([dw * t / 2, dw, 0.0])))
        jm2 = j_refl.magnitude
        peak_jerk = abs(6 * a3) * _RAD_S3
        rate = (jm2 * abs(6 * a3)) * _NM / ureg.second
        peak = 1.5 * mean
        loaded = None
        if loaded_inertia is not None:
            loaded = (loaded_inertia.to(_KGM2).magnitude * dw / tick.to(ureg.second).magnitude) * _NM
        return SeamBridgeResult(
            reflected_inertia=j_refl, step_torque=step, step_torque_rate=step_rate,
            step_over_rate_limit=_ratio(step_rate, torque_rate_limit), step_within_rating=bool(step < rated_torque),
            loaded_step_torque=loaded, mean_torque=mean, peak_acceleration=(1.5 * dw / t) * _RAD_S2,
            peak_torque=peak, peak_reduction=_ratio(step, peak), peak_jerk=peak_jerk, torque_rate=rate,
            torque_rate_share=_ratio(rate, torque_rate_limit), within_rate_limit=bool(rate < torque_rate_limit),
            coefficients=(a3, a4, a5), position_residual=(dw * t / 2) * ureg.radian,
            shortest_bridge=(math.sqrt(6 * dw * jm2 / torque_rate_limit.to(_NM / ureg.second).magnitude)
                             * ureg.second).to(_MS),
            velocity_ceiling=((peak_torque * tick).to(_NM * ureg.second).magnitude / jm2) * _RAD_S,
        )


class ShoulderSwingModel(ForwardModel):
    """
    The loaded shoulder: holding torque, effective inertia, and a rest-to-rest swing.

    Holding payload m at reach r costs m g r of torque; the payload and the
    reflected rotor inertia give J = m r^2 + J_refl (link masses omitted, a lower
    bound). Gravity, held at its horizontal value, makes the torque-limited
    acceleration (tau_max - tau_g) / J lifting and (tau_max + tau_g) / J braking;
    the binding acceleration is the smallest of those and the joint limit. The
    time-optimal (bang-bang) swing of D takes 2 sqrt(D / a) and steps torque by
    2 J a at the switch point. A torque-rate limit bounds jerk at taud / J, and
    the jerk-limited swing is the double-S without cruise
    (``calc_scurve_rest_to_rest_time``).
    """

    requires = ("payload", "reach", "joint", "swing")
    produces = ShoulderSwingResult

    def solve(self, payload: Quantity, reach: Quantity, torque_limit: Quantity, reflected_inertia: Quantity,
              swing: Quantity, acceleration_limit: Quantity, velocity_limit: Quantity,
              torque_rate_limit: Quantity) -> ShoulderSwingResult:
        """Bound a loaded shoulder swing using torque and torque-rate limits.

        Parameters
        ----------
        payload : Quantity
            Nonnegative point-payload mass, convertible to kilograms.
        reach : Quantity
            Positive payload lever arm, convertible to meters.
        torque_limit : Quantity
            Positive torque ceiling that must exceed horizontal gravity torque for lifting.
        reflected_inertia : Quantity
            Positive reflected rotor inertia, in kilogram meters squared.
        swing : Quantity
            Positive rest-to-rest angular displacement in radians.
        acceleration_limit : Quantity
            Positive kinematic joint acceleration ceiling in rad/s squared.
        velocity_limit : Quantity
            Positive joint speed rating; checked after computing the unconstrained profile.
        torque_rate_limit : Quantity
            Positive torque slew bound, in newton meters per second.

        Returns
        -------
        ShoulderSwingResult
            Holding torque, effective inertia, lift/brake acceleration, bang-bang and
            jerk-limited durations in milliseconds, peak rates and a velocity-rating flag. The
            velocity limit is checked, not used to replan the profile.

        See Also
        --------
        mlsysim.physics.robotics.stopping.calc_scurve_rest_to_rest_time : Integrate a symmetric
            bounded-jerk move without cruise.

        Notes
        -----
        Equation: J = ``payload*reach**2`` + reflected_inertia and gravity torque = ``payload*g*reach``.
        Acceleration is the minimum of lifting, braking and kinematic limits; jerk =
        torque_rate_limit/J.

        Assumptions: A point payload, horizontal gravity torque held constant through the swing,
        and rest-to-rest symmetric acceleration profiles.

        Excludes: Link inertia, changing gravity geometry, friction, velocity-capped replanning,
        flexibility, and motor electrical limits.

        References
        ----------
        Derivation: Torque balance gives angular acceleration after subtracting gravity.
        Integrating symmetric acceleration gives the bang-bang time; integrating a bounded-jerk
        acceleration profile gives the S-curve comparison.

        Examples
        --------
        >>> from mlsysim.solvers import ShoulderSwingModel
        >>> from mlsysim import Q_
        >>> result = ShoulderSwingModel().solve(
        ...     payload=Q_('1 kg'),
        ...     reach=Q_('0.5 m'),
        ...     torque_limit=Q_('20 N*m'),
        ...     reflected_inertia=Q_('0.1 kg*m^2'),
        ...     swing=Q_('0.5 rad'),
        ...     acceleration_limit=Q_('2 rad/s^2'),
        ...     velocity_limit=Q_('10 rad/s'),
        ...     torque_rate_limit=Q_('100 N*m/s'),
        ... )
        >>> result.jerk_limited_time >= result.bang_bang_time
        True
        """
        g0 = 1 * ureg.standard_gravity
        tau_g = (payload * g0 * reach).to(_NM)
        j_eff = (payload * reach ** 2 + reflected_inertia).to(_KGM2)
        a_up = ((torque_limit - tau_g) / j_eff).to(ureg.second ** -2).magnitude * _RAD_S2
        a_dn = ((torque_limit + tau_g) / j_eff).to(ureg.second ** -2).magnitude * _RAD_S2
        a = min(a_up, a_dn, acceleration_limit.to(_RAD_S2))
        d, am = swing.to(ureg.radian).magnitude, a.magnitude
        t_opt = (2 * math.sqrt(d / am)) * ureg.second
        v_peak = math.sqrt(am * d) * _RAD_S
        step = (2 * j_eff * a / ureg.radian).to(_NM)
        j = (torque_rate_limit / j_eff).to(ureg.second ** -3).magnitude * _RAD_S3
        move = calc_scurve_rest_to_rest_time(swing.to(ureg.radian), a, j)
        t_jerk = move["duration"].to(ureg.second)
        v_jerk = move["peak_velocity"].to(_RAD_S)
        ramp = (am / j.magnitude) * ureg.second
        return ShoulderSwingResult(
            holding_torque=tau_g, holding_share=_ratio(tau_g, torque_limit), effective_inertia=j_eff,
            accel_up=a_up, accel_down=a_dn, binding_acceleration=a, bang_bang_time=t_opt.to(_MS),
            peak_speed=v_peak, switch_torque_step=step, switch_step_time=(step / torque_rate_limit).to(_MS),
            jerk_bound=j, jerk_limited_time=t_jerk.to(_MS), jerk_limited_peak_speed=v_jerk,
            reaches_plateau=bool(move["reaches_max_acceleration"]), penalty=(t_jerk - t_opt).to(_MS),
            penalty_share=_ratio(t_jerk - t_opt, t_opt), ramp_time=ramp.to(_MS),
            hold_time=(t_jerk / 2 - ramp).to(_MS),
            within_velocity_limit=bool(v_peak < velocity_limit and v_jerk < velocity_limit),
        )


class DynamicWindowModel(ForwardModel):
    """
    The dynamic window of a base over one renewal, and the classical admissibility test.

    Speeds reachable in the next window differ from the current one by at most
    a_max x window. The dynamic window approach admits an arc whose speed
    satisfies v <= sqrt(2 d a_b), a constant-deceleration stop within the
    distance d to the nearest obstacle (Fox, Burgard, and Thrun 1997, eq. 14),
    with no pre-brake delay, jerk ramps, or margins.
    """

    requires = ("max_acceleration", "window", "clear_distance", "deceleration")
    produces = DynamicWindowResult

    def solve(self, max_acceleration: Quantity, window: Quantity, clear_distance: Quantity,
              deceleration: Quantity, drive_limit: Quantity) -> DynamicWindowResult:
        """Compare reachable speed change with a constant-deceleration stop ceiling.

        Parameters
        ----------
        max_acceleration : Quantity
            Nonnegative acceleration magnitude, in meters per second squared.
        window : Quantity
            Nonnegative duration over which reachable speed change is evaluated.
        clear_distance : Quantity
            Nonnegative distance to the obstacle along the candidate arc.
        deceleration : Quantity
            Positive braking-acceleration magnitude, in meters per second squared.
        drive_limit : Quantity
            Nonnegative drive speed rating, in meters per second.

        Returns
        -------
        DynamicWindowResult
            Reachable speed-window half-width and admissible speed in meters per second, plus
            whether the admissible speed is strictly greater than the drive rating.

        Notes
        -----
        Equation: Half-width = ``max_acceleration*window``; admissible_speed =
        ``sqrt(2*clear_distance*deceleration)``.

        Assumptions: Constant acceleration over the renewal window and immediate
        constant-deceleration braking along an already selected arc.

        Excludes: Pre-brake latency, jerk limits, margins, moving obstacles, and trajectory
        search.

        References
        ----------
        Derivation: Integrate constant acceleration to obtain the reachable speed increment.
        Rearranging stopping distance ``v**2/(2*a)`` <= distance gives the admissibility ceiling.

        Examples
        --------
        >>> from mlsysim.solvers import DynamicWindowModel
        >>> from mlsysim import Q_
        >>> result = DynamicWindowModel().solve(
        ...     max_acceleration=Q_('1 m/s^2'),
        ...     window=Q_('0.1 s'),
        ...     clear_distance=Q_('1 m'),
        ...     deceleration=Q_('2 m/s^2'),
        ...     drive_limit=Q_('1 m/s'),
        ... )
        >>> result.admits_drive_limit
        True
        """
        v_ok = ((2 * clear_distance * deceleration) ** 0.5).to(ureg.meter / ureg.second)
        return DynamicWindowResult(
            half_width=(max_acceleration * window).to(ureg.meter / ureg.second),
            admissible_speed=v_ok, admits_drive_limit=bool(v_ok > drive_limit),
        )


class StaleAnchorModel(ForwardModel):
    """
    How far a moving reference runs past a pose sampled when a replacement was requested.

    A joint turning at omega for the replacement delay moves omega x delay; at a
    lever arm r that is up to omega x delay x r of tool motion, the step backward
    a plan anchored at the stale pose would command.
    """

    requires = ("joint_speed", "delay", "lever_arm")
    produces = StaleAnchorResult

    def solve(self, joint_speed: Quantity, delay: Quantity, lever_arm: Quantity) -> StaleAnchorResult:
        """Calculate angular drift and arc-length displacement of a stale pose.

        Parameters
        ----------
        joint_speed : Quantity
            Constant joint angular rate, convertible to radians per second; signed rates
            preserve displacement direction.
        delay : Quantity
            Nonnegative elapsed duration since the pose was sampled.
        lever_arm : Quantity
            Nonnegative distance from the joint axis to the tool.

        Returns
        -------
        StaleAnchorResult
            Signed joint displacement in radians and degrees and signed tool arc-length
            displacement in millimeters. Arc length bounds chord length for the same rotation
            magnitude.

        Notes
        -----
        Equation: Angular gap = ``joint_speed*delay``; tool arc displacement =
        ``gap_in_radians*lever_arm``.

        Assumptions: Constant angular rate about one fixed axis and a rigid lever arm.

        Excludes: Other joints, translation, changing speed, full pose composition, and a
        measured tracking-error estimate.

        References
        ----------
        Derivation: Integrating constant angular rate gives angle; multiplying that angle in
        radians by radius gives arc length.

        Examples
        --------
        >>> from mlsysim.solvers import StaleAnchorModel
        >>> from mlsysim import Q_
        >>> result = StaleAnchorModel().solve(Q_('1 rad/s'), Q_('10 ms'), Q_('0.5 m'))
        >>> result.tool_displacement == Q_('5 mm')
        True
        """
        gap = (joint_speed * delay).to(ureg.radian)
        return StaleAnchorResult(gap=gap, gap_degrees=gap.to(ureg.degree),
                                 tool_displacement=(gap.magnitude * lever_arm).to(_MM))


class BrakingBlendModel(ForwardModel):
    """
    A bumpless transfer from cruise into braking, beside the step and the resident stop.

    At a change of authority the enforcer may blend the base's command from
    cruise to the braking deceleration a with the fixed-endpoint quintic
    smoothstep. Its slope peaks at 1.875, so under the resident stop's jerk
    bound j the window is T = 1.875 a / j (``calc_smoothstep_blend_window``),
    rounded up to whole permission ticks. The same change taken in one tick
    asks a jerk of a / tick. During the blend the speed falls by a T / 2 and
    the base travels v0 T - a T^2 / 7 (``calc_smoothstep_braking_blend``);
    full braking follows. The blended stop is set against an ideal step
    (v0^2 / 2a) and the resident S-curve stop; the reserve a blend uses is the
    largest growth of travel plus resident stop over its ticks
    (``calc_blend_reserve_use``). An arm joint's torque change under a
    torque-rate ceiling gives its own window the same way (added 2026-10-20).
    """

    requires = ("speed", "deceleration", "jerk", "tick")
    produces = BrakingBlendResult

    def solve(self, speed: Quantity, deceleration: Quantity, jerk: Quantity, tick: Quantity,
              torque_change: Optional[Quantity] = None,
              torque_rate_limit: Optional[Quantity] = None) -> BrakingBlendResult:
        """Account for travel and reserve consumed while blending into braking.

        Parameters
        ----------
        speed : Quantity
            Positive initial cruise speed, in meters per second; sufficient to remain
            nonnegative over the blend.
        deceleration : Quantity
            Positive final braking-acceleration magnitude, in meters per second squared.
        jerk : Quantity
            Positive permitted jerk magnitude, in meters per second cubed.
        tick : Quantity
            Positive permission/control tick used to round budgets and sample reserve use.
        torque_change : Quantity, optional
            Optional positive arm torque change; used only together with torque_rate_limit.
        torque_rate_limit : Quantity, optional
            Optional positive arm torque slew limit; used only together with torque_change.

        Returns
        -------
        BrakingBlendResult
            Blend window and rounded budget, travel distances and reserve use in length units,
            and optional arm blend timing. The actual blend uses the unrounded window; tick
            rounding supplies its budget.

        See Also
        --------
        mlsysim.physics.robotics.actuation.calc_smoothstep_blend_window : Scale the smoothstep
            derivative to a rate ceiling.
        mlsysim.physics.robotics.stopping.calc_smoothstep_braking_blend : Integrate speed and
            travel while acceleration is blended.

        Notes
        -----
        Equation: T = ``1.875*deceleration/jerk``, exit speed = ``speed-deceleration*T/2``, and blend
        travel = ``speed*T-deceleration*T**2/7``. Constant braking follows the blend.

        Assumptions: A fixed-endpoint quintic smoothstep reaches the requested deceleration
        without reversal. Reserve use is sampled at permission ticks.

        Excludes: Actuator lag, changing traction, obstacles, continuous-time reserve
        maximization, and a safety guarantee.

        References
        ----------
        Derivation: The quintic smoothstep has maximum normalized slope 1.875. Scaling its slope
        by acceleration change and dividing by the jerk ceiling gives the shortest modeled
        window; integrating gives speed and travel.

        Examples
        --------
        >>> from mlsysim.solvers import BrakingBlendModel
        >>> from mlsysim import Q_
        >>> result = BrakingBlendModel().solve(
        ...     speed=Q_('2 m/s'),
        ...     deceleration=Q_('1 m/s^2'),
        ...     jerk=Q_('10 m/s^3'),
        ...     tick=Q_('1 ms'),
        ... )
        >>> result.blended_stop > result.step_stop
        True
        """
        m, s = ureg.meter, ureg.second
        window = calc_smoothstep_blend_window(deceleration, jerk).to(s)
        n = math.ceil((window / tick).to("").magnitude - 1e-9)
        blend = calc_smoothstep_braking_blend(speed, deceleration, window)
        v1 = blend["exit_speed"]
        after = (v1 ** 2 / (2 * deceleration)).to(m)
        total = (blend["travel"] + after).to(m)
        step = (speed ** 2 / (2 * deceleration)).to(m)
        resident = calc_scurve_stop_distance(speed, deceleration, jerk).to(m)
        out = dict(
            speed=speed, deceleration=deceleration, jerk=jerk, tick=tick, window=window,
            window_ticks=n, window_budget=(n * tick).to(_MS),
            step_jerk=(deceleration / tick).to(m / s ** 3), ramp_time=(deceleration / jerk).to(_MS),
            exit_speed=v1, blend_travel=blend["travel"], after_travel=after, blended_stop=total,
            step_stop=step, scurve_travel=resident, penalty_over_step=(total - step).to(m),
            excess_over_scurve=(total - resident).to(m),
            reserve_use=calc_blend_reserve_use(speed, deceleration, jerk, window, tick),
        )
        if torque_change is not None and torque_rate_limit is not None:
            arm = calc_smoothstep_blend_window(torque_change, torque_rate_limit).to(_MS)
            k = math.ceil((arm / tick).to("").magnitude - 1e-9)
            out.update(torque_change=torque_change, torque_rate_limit=torque_rate_limit,
                       arm_window=arm, arm_window_ticks=k, arm_window_budget=(k * tick).to(_MS))
        return BrakingBlendResult(**out)
