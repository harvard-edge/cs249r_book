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
from ..results import OnsetTrialResult, RailHoldUpResult, StoppingBudgetResult, TractionLossResult
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
        ms = ureg.millisecond
        stop = calc_scurve_stop_time(speed, deceleration, jerk).to(ms)
        t_set = (flag_path + stop + engage).to(ms)
        return RailHoldUpResult(
            flag_path=flag_path.to(ms), speed=speed, velocity_term=(speed / deceleration).to(ms),
            jerk_term=(deceleration / jerk).to(ms), stop_time=stop, engage=engage.to(ms),
            flag_to_set=t_set, holdup=holdup, margin=(holdup - t_set).to(ms),
        )
