"""Embodied solvers: how far a machine travels before it stops, and what the floor allows.

Domain implementations behind ``mlsysim.solvers`` (the public import path,
derived from ``engine.solvers.__init__``). The equations live in
``physics.robotics.stopping`` and ``physics.drivetrain``; these models compose
them for one stated situation (added 2026-10-20).
"""

from __future__ import annotations

from typing import Optional

from ..results import StoppingBudgetResult, TractionLossResult
from ...core.types import Quantity
from ...core.units import ureg
from ...physics.drivetrain import calc_required_friction, calc_traction_deceleration_limit
from ...physics.robotics.control import calc_watchdog_lease_bound
from ...physics.robotics.stopping import (
    calc_braking_time,
    calc_max_permitted_velocity,
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
    (``calc_safe_stopping_distance``); jerk-limited braking is left out
    (``calc_scurve_stop_distance`` adds it). ``added_delay`` prices one more
    delay step as travel at ``speed``. The delay ceiling is the longest
    constant-speed pre-brake delay whose stop, with the overhead, still fits
    (``calc_watchdog_lease_bound``): ``(D_clear - overhead - v^2/2a) / v``.
    """
    requires = ("speed", "delay", "deceleration")
    produces = StoppingBudgetResult

    def solve(self, speed: Quantity, delay: Quantity, deceleration: Quantity,
              clear_distance: Optional[Quantity] = None,
              speed_at_delay_end: Optional[Quantity] = None,
              loc_margin: Optional[Quantity] = None,
              physical_margin: Optional[Quantity] = None,
              added_delay: Optional[Quantity] = None) -> StoppingBudgetResult:
        m = ureg.meter
        end = speed if speed_at_delay_end is None else speed_at_delay_end
        delay_travel = calc_travel_between_speeds(speed, end, delay).to(m)
        braking = (speed ** 2 / (2 * deceleration)).to(m)
        stop = (delay_travel + braking).to(m)
        loc = 0 * m if loc_margin is None else loc_margin.to(m)
        margin = 0 * m if physical_margin is None else physical_margin.to(m)
        overhead = (loc + margin).to(m)
        required = (stop + overhead).to(m)
        delay_positive = delay.to(ureg.second).magnitude > 0
        out = dict(
            speed=speed, delay=delay, deceleration=deceleration,
            delay_travel=delay_travel, braking_travel=braking, stop_distance=stop,
            overhead=overhead, required_distance=required,
            braking_time=calc_braking_time(speed, deceleration),
            crossover_speed=calc_stopping_crossover_speed(delay, deceleration),
        )
        if delay_positive:
            out["braking_over_delay"] = _ratio(braking, delay_travel)
            out["delay_share_of_braking"] = _ratio(delay_travel, braking)
        if clear_distance is not None:
            spare = (clear_distance - required).to(m)
            out.update(clear_distance=clear_distance, spare=spare,
                       fits=spare.magnitude >= 0, clear_over_stop=_ratio(clear_distance, required))
            if delay_positive:
                out["speed_ceiling"] = calc_max_permitted_velocity(clear_distance, delay, deceleration,
                                                                   loc_margin=loc, physical_margin=margin)
            room = (clear_distance - overhead).to(m)
            if speed_at_delay_end is None and room > braking:
                ceiling = calc_watchdog_lease_bound(room, speed, deceleration)["t_total_delay_max"]
                out.update(delay_ceiling=ceiling.to(ureg.millisecond),
                           delay_spare=(ceiling - delay).to(ureg.millisecond))
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
