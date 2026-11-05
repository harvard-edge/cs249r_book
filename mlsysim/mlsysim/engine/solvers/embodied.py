"""Embodied solvers: how far a machine travels before it stops, and what the floor allows.

Domain implementations behind ``mlsysim.solvers`` (the public import path,
derived from ``engine.solvers.__init__``). The equations live in
``physics.robotics.stopping`` and ``physics.drivetrain``; these models compose
them for one stated situation (added 2026-10-20).
"""

from __future__ import annotations

import math
from typing import Optional

from ..results import StoppingBudgetResult, TakeoverBudgetResult, TractionLossResult
from ..results import ShieldCoverageBillResult  # added 2026-10-20
from ..results import OnsetTrialResult, RailHoldUpResult
from ...core.types import Quantity
from ...core.units import ureg
from ...physics.drivetrain import calc_required_friction, calc_traction_deceleration_limit
from ...physics.robotics.control import calc_watchdog_lease_bound
from ...physics.robotics.stopping import (
    calc_braking_time,
    calc_max_permitted_velocity,
    calc_scurve_max_permitted_velocity,
    calc_scurve_stop_distance,
    calc_scurve_stop_time,
    calc_stopping_crossover_speed,
    calc_travel_between_speeds,
)
from .base import ForwardModel


def _ratio(a, b) -> float:
    return float((a / b).to("").magnitude)


class StoppingBudgetModel(ForwardModel):
    """
    The elementary stopping budget: travel during the pre-brake delay plus braking travel.

    ``d_stop = travel(delay) + v^2 / (2 a) <= D_clear``. The delay travel is
    ``v * tau`` at constant speed, or the trapezoid between ``speed`` and
    ``speed_at_delay_end`` when two speeds were recorded. Braking travel and
    braking time are from ``speed``, the speed when the hazard appears, as the
    elementary budget assumes. The crossover speed ``2 a tau`` is where the
    two terms are equal; the speed ceiling is the fastest speed whose stop
    fits ``clear_distance``. A localization bound and a protective margin, when
    stated, add to the required distance and lower the ceiling
    (``calc_safe_stopping_distance``). ``added_delay`` prices one more
    delay step as travel at ``speed``. The delay ceiling is the longest
    constant-speed pre-brake delay whose stop, with the overhead, still fits
    (``calc_watchdog_lease_bound``): ``(D_clear - overhead - v^2/2a) / v``.

    The finished budget (added 2026-10-20) takes three optional terms. A
    ``jerk`` bound replaces constant-deceleration braking with the resident
    S-curve stop (``calc_scurve_stop_distance``); its extra travel over
    ``v^2/2a`` is the jerk-ramp travel, and above ``a^2/j`` it equals
    ``v a / (2 j)``, a delay of ``a / (2 j)``. A ``tracking_bound`` (the
    tracking inset) joins the required distance after the two allowances. An
    ``approach_speed`` charges a person who keeps closing the gap for the
    whole delay and stop (``v_h (tau + T_stop)``). With a jerk bound the
    speed ceiling is the S-curve root (``calc_scurve_max_permitted_velocity``)
    and the delay ceiling is ``(D_clear - overhead - inset - d_S(v)) / v``.
    The same case reports the weakest deceleration whose S-curve stop still
    fits the braking room, jerk fixed, and its shortfall from the stated one
    (added 2026-10-20).
    """
    requires = ("speed", "delay", "deceleration")
    produces = StoppingBudgetResult

    def solve(self, speed: Quantity, delay: Quantity, deceleration: Quantity,
              clear_distance: Optional[Quantity] = None,
              speed_at_delay_end: Optional[Quantity] = None,
              loc_margin: Optional[Quantity] = None,
              physical_margin: Optional[Quantity] = None,
              added_delay: Optional[Quantity] = None,
              jerk: Optional[Quantity] = None,
              tracking_bound: Optional[Quantity] = None,
              approach_speed: Optional[Quantity] = None) -> StoppingBudgetResult:
        """Compose delay, braking, and clearance budgets.

        Parameters
        ----------
        speed : Quantity
            Entry speed in m/s; positive for inverse budgets.
        delay : Quantity
            Pre-braking latency in seconds; nonnegative.
        deceleration : Quantity
            Available braking magnitude in m/s**2; positive.
        clear_distance : Optional[Quantity]
            Available clearance in meters; None omits clearance checks.
        speed_at_delay_end : Optional[Quantity]
            Recorded terminal delay speed in m/s; None uses entry speed.
        loc_margin : Optional[Quantity]
            Localization allowance in meters; None contributes zero.
        physical_margin : Optional[Quantity]
            Protective allowance in meters; None contributes zero.
        added_delay : Optional[Quantity]
            Extra delay to price as travel, in seconds; None omits that output.
        jerk : Optional[Quantity]
            Positive acceleration slew limit in m/s**3; None uses constant deceleration.
        tracking_bound : Optional[Quantity]
            Tracking inset in meters; None contributes zero.
        approach_speed : Optional[Quantity]
            Closing speed of the other body in m/s; None contributes zero.

        Returns
        -------
        StoppingBudgetResult
            Distances in meters, times in milliseconds, speed ceilings in m/s,
            and signed clearance margins; optional fits compares required clearance.

        Notes
        -----
        Equation: d_required = (v + v_end) tau / 2 + d_brake + margins + v_approach (tau + t_stop).

        Assumptions: The constant-deceleration branch brakes from entry speed, even when the delay endpoint differs. Optional jerk selects the S-curve branch.

        Excludes: Terrain changes, controller transients outside the stated delay, and safety certification.

        References
        ----------
        Derivation: Distances are integrated over the stated delay and braking profile; inverse budgets solve the same distance inequality.

        Examples
        --------
        >>> from mlsysim.engine.solvers.embodied import StoppingBudgetModel
        >>> from mlsysim.core.units import ureg
        >>> r = StoppingBudgetModel().solve(ureg('1 m/s'), ureg('0.1 s'), ureg('1 m/s**2'))
>>> round(r.stop_distance.m_as('m'), 2)
0.6
        """
        m = ureg.meter
        end = speed if speed_at_delay_end is None else speed_at_delay_end
        delay_travel = calc_travel_between_speeds(speed, end, delay).to(m)
        braking = (speed ** 2 / (2 * deceleration)).to(m)
        if jerk is None:
            ramp = 0 * m
            stop_time = calc_braking_time(speed, deceleration)
        else:
            ramp = (calc_scurve_stop_distance(speed, deceleration, jerk) - braking).to(m)
            stop_time = calc_scurve_stop_time(speed, deceleration, jerk).to(ureg.millisecond)
        stop = (delay_travel + braking + ramp).to(m)
        loc = 0 * m if loc_margin is None else loc_margin.to(m)
        margin = 0 * m if physical_margin is None else physical_margin.to(m)
        inset = 0 * m if tracking_bound is None else tracking_bound.to(m)
        overhead = (loc + margin).to(m)
        approach = (0 * m if approach_speed is None
                    else (approach_speed * (delay + stop_time)).to(m))
        required = (stop + overhead + inset + approach).to(m)
        delay_positive = delay.to(ureg.second).magnitude > 0
        out = dict(
            speed=speed, delay=delay, deceleration=deceleration,
            delay_travel=delay_travel, braking_travel=braking, stop_distance=stop,
            overhead=overhead, required_distance=required,
            braking_time=calc_braking_time(speed, deceleration),
            crossover_speed=calc_stopping_crossover_speed(delay, deceleration),
        )
        if jerk is not None:
            out.update(jerk=jerk, jerk_ramp_travel=ramp, stop_time=stop_time, scurve_travel=(braking + ramp).to(m),
                       jerk_plateau_speed=(deceleration ** 2 / jerk).to(m / ureg.second),
                       ramp_delay=(deceleration / (2 * jerk)).to(ureg.millisecond))
        if tracking_bound is not None:
            out["tracking_inset"] = inset
        if approach_speed is not None:
            out["approach_travel"] = approach
        if delay_positive:
            out["braking_over_delay"] = _ratio(braking, delay_travel)
            out["delay_share_of_braking"] = _ratio(delay_travel, braking)
        if clear_distance is not None:
            spare = (clear_distance - required).to(m)
            out.update(clear_distance=clear_distance, spare=spare,
                       fits=spare.magnitude >= 0, clear_over_stop=_ratio(clear_distance, required))
            if jerk is not None:
                out["speed_ceiling"] = calc_scurve_max_permitted_velocity(
                    clear_distance, delay, deceleration, jerk, loc_margin=loc,
                    physical_margin=margin + inset, approach_velocity=approach_speed,
                ).to(m / ureg.second)
            elif delay_positive and approach_speed is None:
                out["speed_ceiling"] = calc_max_permitted_velocity(clear_distance, delay, deceleration,
                                                                   loc_margin=loc,
                                                                   physical_margin=margin + inset)
            room = (clear_distance - overhead - inset).to(m)
            if delay_positive and approach_speed is None:
                out["rigid_speed_ceiling"] = (room / delay).to(m / ureg.second)
            if jerk is not None and approach_speed is None:
                # Above a^2/j the S-curve root solves v^2/(2a) + b v - room = 0 with
                # b = delay + a/(2j): v = a (sqrt(b^2 + 2 room / a) - b).
                if out["speed_ceiling"] >= out["jerk_plateau_speed"]:
                    lead = (delay + out["ramp_delay"]).to(ureg.second)
                    out.update(ceiling_lead_time=lead,
                               ceiling_root=((lead ** 2 + 2 * room / deceleration) ** 0.5).to(ureg.second))
            if jerk is not None and approach_speed is None and speed_at_delay_end is None:
                ceiling = ((room - braking - ramp) / speed).to(ureg.millisecond)
                out.update(delay_ceiling=ceiling, delay_spare=(ceiling - delay).to(ureg.millisecond))
            elif jerk is None and approach_speed is None and speed_at_delay_end is None and room > braking:
                ceiling = calc_watchdog_lease_bound(room, speed, deceleration)["t_total_delay_max"]
                out.update(delay_ceiling=ceiling.to(ureg.millisecond),
                           delay_spare=(ceiling - delay).to(ureg.millisecond))
            if jerk is not None and approach_speed is None and speed_at_delay_end is None:
                # Weakest deceleration whose S-curve stop still fits (added 2026-10-20), jerk
                # fixed: on the trapezoidal branch v^2/(2a) + v a/(2j) = room - delay travel,
                # i.e. (v/(2j)) a^2 - R a + v^2/2 = 0; the smaller root is the branch where
                # distance falls as a rises. Set only when that root stays trapezoidal (v >= a^2/j).
                brake_room = (room - delay_travel).to(m)
                qa = (speed / (2 * jerk)).to(ureg.second ** 2)
                disc = (brake_room ** 2 - 2 * qa * speed ** 2).to(m ** 2)
                if brake_room.magnitude > 0 and disc.magnitude >= 0:
                    a_min = ((brake_room - disc ** 0.5) / (2 * qa)).to(m / ureg.second ** 2)
                    if speed >= (a_min ** 2 / jerk).to(m / ureg.second):
                        out.update(braking_room=brake_room, min_deceleration=a_min,
                                   deceleration_shortfall=_ratio(deceleration - a_min, deceleration))
        if added_delay is not None:
            out["added_delay_travel"] = (speed * added_delay).to(m)
        return StoppingBudgetResult(**out)


class TractionLossModel(ForwardModel):
    """
    A drop in floor friction against the deceleration a machine's budget assumes.

    At the traction limit the deceleration is ``mu g phi``, so braking distance
    ``v^2 / (2 mu g phi)`` scales as ``1 / mu`` and the ratio of the two
    braking distances is ``mu_before / mu_after``. The stated deceleration
    holds only while the lower friction still supplies it
    (``calc_required_friction``).
    """
    requires = ("friction",)
    produces = TractionLossResult

    def solve(self, mu_before: float, mu_after: float, deceleration: Quantity,
              drive_load_share: float = 1.0) -> TractionLossResult:
        """Compare a friction change with required deceleration.

        Parameters
        ----------
        mu_before : float
            Positive dimensionless friction coefficient before the change.
        mu_after : float
            Positive dimensionless friction coefficient after the change.
        deceleration : Quantity
            Available braking magnitude in m/s**2; positive.
        drive_load_share : float
            Fraction of normal load on driven wheels, in (0, 1].

        Returns
        -------
        TractionLossResult
            Dimensionless braking-distance ratio and required friction, acceleration
            ceiling in m/s**2, and a friction-premise comparison.

        Notes
        -----
        Equation: d_after / d_before = mu_before / mu_after; a_max = mu_after g drive_load_share.

        Assumptions: Coulomb friction and unchanged speed and load distribution.

        Excludes: Slip dynamics, slope, tire deformation, and safety certification.

        References
        ----------
        Derivation: The constant-deceleration distance ``v**2/(2`` a) is inversely proportional to friction-limited acceleration.

        Examples
        --------
        >>> from mlsysim.engine.solvers.embodied import TractionLossModel
        >>> from mlsysim.core.units import ureg
        >>> r = TractionLossModel().solve(0.8, 0.4, ureg('1 m/s**2'))
>>> r.braking_distance_ratio
2.0
        """
        required = calc_required_friction(deceleration, drive_load_share)
        return TractionLossResult(
            braking_distance_ratio=mu_before / mu_after,
            traction_ceiling=calc_traction_deceleration_limit(mu_after, drive_load_share),
            required_friction=required,
            premise_holds=mu_after >= required,
        )


class TakeoverBudgetModel(ForwardModel):
    """
    A request for a person to take over, priced against the finished stopping budget.

    The takeover T_Sigma (the human response plus the console transport) adds
    to the machine's pre-brake delay tau: until the person commits, the
    machine keeps moving at the request speed. Four finished budgets
    (``StoppingBudgetModel`` with the S-curve jerk bound and the tracking
    inset) are solved at that speed: the machine alone, the machine with the
    dispatch-to-admission allocation Delta_da charged, and each with the
    takeover waited out before braking. The fallback deadline T_latest is the
    machine-alone delay spare: the longest wait after the request before the
    machine's own stop must begin,

        T_latest = (D_clear - delta - d_S(v)) / v - tau.

    Configured timeouts are clipped to it, T = max(0, min(T_cfg, T_latest)),
    and dispatched on the last whole permission tick. Once the operator holds
    the channel, a manual lease longer than the chunk lease adds its excess to
    the delay as travel at the request speed (added 2026-10-20).
    """
    requires = ("speed", "delay", "deceleration", "clear_distance", "takeover")
    produces = TakeoverBudgetResult

    def solve(self, speed: Quantity, delay: Quantity, deceleration: Quantity, clear_distance: Quantity,
              takeover: Quantity, loc_margin: Optional[Quantity] = None,
              physical_margin: Optional[Quantity] = None, jerk: Optional[Quantity] = None,
              tracking_bound: Optional[Quantity] = None,
              dispatch_to_admission: Optional[Quantity] = None,
              configured_timeout: Optional[Quantity] = None, tick: Optional[Quantity] = None,
              manual_lease: Optional[Quantity] = None,
              chunk_lease: Optional[Quantity] = None) -> TakeoverBudgetResult:
        """Charge takeover latency against the machine stopping reserve.

        Parameters
        ----------
        speed : Quantity
            Entry speed in m/s; positive for inverse budgets.
        delay : Quantity
            Pre-braking latency in seconds; nonnegative.
        deceleration : Quantity
            Available braking magnitude in m/s**2; positive.
        clear_distance : Quantity
            Available clearance in meters; None omits clearance checks.
        takeover : Quantity
            Human response and transport duration in seconds; nonnegative.
        loc_margin : Optional[Quantity]
            Localization allowance in meters; None contributes zero.
        physical_margin : Optional[Quantity]
            Protective allowance in meters; None contributes zero.
        jerk : Optional[Quantity]
            Positive acceleration slew limit in m/s**3; None uses constant deceleration.
        tracking_bound : Optional[Quantity]
            Tracking inset in meters; None contributes zero.
        dispatch_to_admission : Optional[Quantity]
            Additional admission delay in seconds; None contributes zero.
        configured_timeout : Optional[Quantity]
            Requested timeout in seconds; requires positive tick when supplied.
        tick : Optional[Quantity]
            Permission tick period in seconds; positive when timeout is supplied.
        manual_lease : Optional[Quantity]
            Manual command lease in seconds; requires chunk_lease when supplied.
        chunk_lease : Optional[Quantity]
            Reference command lease in seconds.

        Returns
        -------
        TakeoverBudgetResult
            Stopping budgets, travel in millimeters, and deadlines, timeouts, and
            time reserves in milliseconds; permission counts are dimensionless.

        Notes
        -----
        Equation: T_latest = (D_clear - overhead - d_brake) / v - tau; timeout = max(0, min(T_configured, T_latest)).

        Assumptions: The same speed, braking profile, and margins apply to all four compared budgets. Inputs must leave an available delay_spare on the underlying stopping result.

        Excludes: Human-response distributions, changing approach speed, and safety certification.

        References
        ----------
        Derivation: Subtract the common stopping budget from clearance and convert remaining travel into waiting time at entry speed.

        Examples
        --------
        >>> from mlsysim.engine.solvers.embodied import TakeoverBudgetModel
        >>> from mlsysim.core.units import ureg
        >>> r = TakeoverBudgetModel().solve(ureg('1 m/s'), ureg('0.1 s'), ureg('1 m/s**2'), ureg('2 m'), ureg('0.2 s'))
>>> round(r.takeover_travel.m_as('m'), 2)
0.2
        """
        ms, mm = ureg.millisecond, ureg.millimeter
        da = 0 * ms if dispatch_to_admission is None else dispatch_to_admission.to(ms)
        stop = StoppingBudgetModel()

        def budget(d):
            return stop.solve(speed, d.to(ms), deceleration, clear_distance, loc_margin=loc_margin,
                              physical_margin=physical_margin, jerk=jerk, tracking_bound=tracking_bound)

        machine = budget(delay)
        machine_da = budget(delay + da)
        with_takeover = budget(delay + takeover)
        with_both = budget(delay + da + takeover)
        deadline = machine.delay_spare.to(ms)
        deadline_da = machine_da.delay_spare.to(ms)
        travel = (speed * takeover).to(mm)
        admission = (speed * da).to(mm)
        after_takeover = (machine.spare - travel).to(mm)
        after_admission = (after_takeover - admission).to(mm)
        out = dict(
            speed=speed, takeover=takeover.to(ms), dispatch_to_admission=da,
            machine=machine, machine_with_dispatch=machine_da,
            with_takeover=with_takeover, with_takeover_and_dispatch=with_both,
            deadline=deadline, deadline_after_admission=deadline_da,
            takeover_travel=travel, admission_travel=admission,
            time_left=(deadline - takeover).to(ms), time_left_after_admission=(deadline_da - takeover).to(ms),
            spare_after_takeover=after_takeover, spare_after_admission=after_admission,
            takeover_over_spare=_ratio(travel, machine.spare),
            takeover_over_spare_after_admission=_ratio(travel, machine_da.spare),
        )
        if manual_lease is not None and chunk_lease is not None:
            extension = (manual_lease - chunk_lease).to(ms)
            extension_travel = (speed * extension).to(mm)
            out.update(lease_extension=extension, lease_extension_travel=extension_travel,
                       spare_after_lease=(after_admission - extension_travel).to(mm))
        if configured_timeout is not None and tick is not None:
            zero = 0 * ms

            def clip(limit):
                t = min(configured_timeout.to(ms), limit)
                return max(zero, t).to(ms)

            def ticks(t):
                return int(math.floor(round((t / tick).to("").magnitude, 9)))

            t_cfg, t_cfg_da = clip(deadline), clip(deadline_da)
            out.update(configured_timeout=configured_timeout.to(ms), timeout=t_cfg, timeout_ticks=ticks(t_cfg),
                       timeout_after_admission=t_cfg_da, timeout_ticks_after_admission=ticks(t_cfg_da))
        return TakeoverBudgetResult(**out)
class OnsetTrialModel(ForwardModel):
    """
    One stopping trace scored against a precommitted oracle (added 2026-10-20).

    The oracle bounds powered-deceleration onset at ``onset_bound`` after the
    reference event and travel to rest at ``travel_bound`` (onset-bound travel,
    the resident stop, and the tracking inset). A trace that begins braking
    at ``onset`` travels ``speed * onset`` plus the same resident stop. Each
    predicate keeps its own signed margin; a combined score needs scales fixed
    in advance, here the bounds themselves. The trace passes only when both
    margins are non-negative.
    """
    requires = ("onset",)
    produces = OnsetTrialResult

    def solve(self, onset_bound: Quantity, onset: Quantity, speed: Quantity, resident_stop: Quantity,
              travel_bound: Quantity, stop_time: Quantity) -> OnsetTrialResult:
        """Score one observed onset against precommitted bounds.

        Parameters
        ----------
        onset_bound : Quantity
            Positive permitted onset delay in seconds, fixed before scoring.
        onset : Quantity
            Observed powered-braking onset in seconds after the reference event.
        speed : Quantity
            Entry speed in m/s; positive for inverse budgets.
        resident_stop : Quantity
            Braking travel after onset in meters; nonnegative.
        travel_bound : Quantity
            Positive allowed total travel in meters, fixed before scoring.
        stop_time : Quantity
            Braking duration after onset in seconds; nonnegative.

        Returns
        -------
        OnsetTrialResult
            Onset/time margins in milliseconds, travel margins in millimeters,
            normalized margins, and the conjunction of the two bound checks.

        Notes
        -----
        Equation: travel = speed onset + resident_stop; passes = (onset <= onset_bound) and (travel <= travel_bound).

        Assumptions: The observed and reference cases share the resident braking profile; the two bounds are fixed externally.

        Excludes: Statistical confidence across trials and any inference of certified safety.

        References
        ----------
        Derivation: Each signed margin subtracts the observed quantity from its precommitted bound; normalized margins divide by that same bound.

        Examples
        --------
        >>> from mlsysim.engine.solvers.embodied import OnsetTrialModel
        >>> from mlsysim.core.units import ureg
        >>> r = OnsetTrialModel().solve(ureg('0.2 s'), ureg('0.1 s'), ureg('1 m/s'), ureg('0.5 m'), ureg('1 m'), ureg('1 s'))
>>> r.passes
True
        """
        ms, mm = ureg.millisecond, ureg.millimeter
        onset_margin = (onset_bound - onset).to(ms)
        travel = (speed * onset + resident_stop).to(mm)
        distance_margin = (travel_bound - travel).to(mm)
        return OnsetTrialResult(
            onset_bound=onset_bound.to(ms), onset=onset.to(ms), onset_margin=onset_margin,
            extra_travel=(speed * (onset - onset_bound)).to(mm), travel=travel,
            travel_bound=travel_bound.to(mm), distance_margin=distance_margin,
            onset_margin_normalized=_ratio(onset_margin, onset_bound),
            distance_margin_normalized=_ratio(distance_margin, travel_bound),
            rest_time=(onset + stop_time).to(ms), rest_time_bound=(onset_bound + stop_time).to(ms),
            passes=onset_margin.magnitude >= 0 and distance_margin.magnitude >= 0,
        )


class RailHoldUpModel(ForwardModel):
    """
    A capacitor-backed rail that must carry the protective response from the
    loss-of-feed flag until the holding brakes are set (added 2026-10-20).

    ``t_set = flag_path + (v/a + a/j) + t_engage``: the path from the flag to
    powered-deceleration onset, the S-curve stop from the entry speed ``v``
    (``calc_scurve_stop_time``), and the holding brake's engagement. The
    margin is the hold-up time less ``t_set``.
    """
    requires = ("holdup",)
    produces = RailHoldUpResult

    def solve(self, flag_path: Quantity, speed: Quantity, deceleration: Quantity, jerk: Quantity,
              engage: Quantity, holdup: Quantity) -> RailHoldUpResult:
        """Compare rail hold-up time with the protective stop duration.

        Parameters
        ----------
        flag_path : Quantity
            Loss-of-feed flag to brake-onset latency in seconds.
        speed : Quantity
            Entry speed in m/s; positive for inverse budgets.
        deceleration : Quantity
            Available braking magnitude in m/s**2; positive.
        jerk : Quantity
            Positive acceleration slew limit in m/s**3; None uses constant deceleration.
        engage : Quantity
            Holding-brake engagement duration in seconds.
        holdup : Quantity
            Available rail hold-up duration in seconds.

        Returns
        -------
        RailHoldUpResult
            Phase durations and remaining hold-up margin in milliseconds;
            a negative margin means the stated supply duration is insufficient.

        Notes
        -----
        Equation: t_set = flag_path + t_scurve + engage; margin = holdup - t_set.

        Assumptions: Serial flag, stopping, and engagement phases; the supplied jerk and deceleration remain available during hold-up.

        Excludes: Rail voltage trajectories, capacitor sizing, and safety certification.

        References
        ----------
        Derivation: Serial response durations add; the S-curve stopping helper supplies the powered-stop duration.

        Examples
        --------
        >>> from mlsysim.engine.solvers.embodied import RailHoldUpModel
        >>> from mlsysim.core.units import ureg
        >>> r = RailHoldUpModel().solve(ureg('0.1 s'), ureg('1 m/s'), ureg('1 m/s**2'), ureg('2 m/s**3'), ureg('0.1 s'), ureg('3 s'))
>>> r.margin.m_as('s') > 0
True
        """
        ms = ureg.millisecond
        stop = calc_scurve_stop_time(speed, deceleration, jerk).to(ms)
        t_set = (flag_path + stop + engage).to(ms)
        return RailHoldUpResult(
            flag_path=flag_path.to(ms), speed=speed, velocity_term=(speed / deceleration).to(ms),
            jerk_term=(deceleration / jerk).to(ms), stop_time=stop, engage=engage.to(ms),
            flag_to_set=t_set, holdup=holdup, margin=(holdup - t_set).to(ms),
        )


class ShieldCoverageBillModel(ForwardModel):
    """
    The evidence a shielded hazard-rate claim needs (added 2026-10-20).

    A fleet of ``fleet_units`` logs ``hours_per_day`` for one release cycle of
    ``cycle_days`` with zero hazardous proposals; the one-sided Poisson bound
    is ``-ln(1 - C) / T``. The shield passes ``1 - coverage`` of hazardous
    proposals (``calc_shielded_system_hazard_rate``), plus the common-cause
    rate. Against ``target_rate`` the model reports the coverage the bound
    would need, the proposal bound the stated coverage would need, the
    failure-free exposure and fleet days that bound it
    (``calc_empirical_testing_exposure``), and the seeded-hazard campaign
    that bounds the coverage itself (``calc_zero_failure_sample_size``), alone
    and with both bounds held jointly at ``1 - alpha/2`` each.
    """
    requires = ("fleet_units", "cycle_days", "coverage", "target_rate")
    produces = ShieldCoverageBillResult

    def solve(self, fleet_units: int, cycle_days: Quantity, hours_per_day: Quantity, coverage: float,
              common_cause_rate: Quantity, target_rate: Quantity, confidence: float) -> ShieldCoverageBillResult:
        """Compute exposure and coverage requirements for a hazard-rate bound.

        Parameters
        ----------
        fleet_units : int
            Positive number of machines contributing exposure.
        cycle_days : Quantity
            Release-cycle duration in days; positive.
        hours_per_day : Quantity
            Logging rate in hours/day; positive.
        coverage : float
            Hazard rejection fraction, strictly between zero and one.
        common_cause_rate : Quantity
            Nonnegative common-cause hazard rate in 1/hour.
        target_rate : Quantity
            Target hazard rate in 1/hour, above the common-cause rate.
        confidence : float
            One-sided confidence in (0, 1).

        Returns
        -------
        ShieldCoverageBillResult
            Exposure in hours, rate bounds in inverse hours, campaign duration
            in days, trial counts, and the comparison with the stated rate target.

        Notes
        -----
        Equation: T = fleet_units cycle_days hours_per_day; lambda_upper = -log(1-confidence)/T; lambda_residual = (1-coverage) lambda_upper + common_cause_rate.

        Assumptions: Zero observed hazardous proposals under a stationary Poisson process and representative seeded coverage trials. Joint bounds split the error budget equally.

        Excludes: Nonstationary hazards, unmodeled dependence, experimental evidence collection, and safety certification.

        References
        ----------
        Derivation: For zero Poisson events, exp(-lambda T) equals the tail probability. Residual proposals multiply by the uncovered fraction; the joint bounds use a union-bound allocation.

        Examples
        --------
        >>> from mlsysim.engine.solvers.embodied import ShieldCoverageBillModel
        >>> from mlsysim.core.units import ureg
        >>> r = ShieldCoverageBillModel().solve(10, ureg('30 day'), ureg('8 hour/day'), 0.9, ureg('0 / hour'), ureg('0.001 / hour'), 0.95)
>>> r.exposure.m_as('hour')
2400.0
        """
        from ...physics.robotics.assurance import (
            calc_empirical_testing_exposure,
            calc_shielded_system_hazard_rate,
            calc_zero_failure_sample_size,
        )
        per_hour = 1 / ureg.hour
        alpha = 1.0 - confidence
        exposure = (fleet_units * cycle_days * hours_per_day).to(ureg.hour)
        ln_inv = -math.log(alpha)
        upper = (ln_inv / exposure).to(per_hour)
        cc = common_cause_rate.to(per_hour)
        target = target_rate.to(per_hour)
        residual = calc_shielded_system_hazard_rate(upper, coverage, cc)
        limit = ((target - cc) / (1.0 - coverage)).to(per_hour)
        needed = calc_empirical_testing_exposure(limit, confidence=confidence).to(ureg.hour)
        fleet_rate = (fleet_units * hours_per_day).to(ureg.hour / ureg.day)
        days = (needed / fleet_rate).to(ureg.day)
        joint = 1.0 - alpha / 2.0
        ln_inv_half = -math.log(alpha / 2.0)
        upper_joint = (ln_inv_half / exposure).to(per_hour)
        return ShieldCoverageBillResult(
            exposure=exposure, ln_inv_alpha=ln_inv, proposal_rate_upper=upper, residual_rate=residual,
            meets_target=residual <= target, required_coverage=float(1.0 - ((target - cc) / upper).to("").magnitude),
            proposal_rate_limit=limit, exposure_needed=needed, days_needed=days,
            days_beyond_cycle=(days - cycle_days).to(ureg.day),
            coverage_trials=calc_zero_failure_sample_size(coverage, confidence=confidence),
            joint_confidence=joint,
            coverage_trials_joint=calc_zero_failure_sample_size(coverage, confidence=joint),
            ln_inv_half_alpha=ln_inv_half, proposal_rate_upper_joint=upper_joint,
            residual_rate_joint=calc_shielded_system_hazard_rate(upper_joint, coverage, cc),
        )
