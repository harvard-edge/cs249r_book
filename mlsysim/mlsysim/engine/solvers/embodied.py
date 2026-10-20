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
    fits ``clear_distance``. Jerk-limited braking, localization error, and a
    protective margin are left out (``calc_scurve_stop_distance`` and
    ``calc_safe_stopping_distance`` add them).
    """
    requires = ("speed", "delay", "deceleration")
    produces = StoppingBudgetResult

    def solve(self, speed: Quantity, delay: Quantity, deceleration: Quantity,
              clear_distance: Optional[Quantity] = None,
              speed_at_delay_end: Optional[Quantity] = None) -> StoppingBudgetResult:
        m = ureg.meter
        end = speed if speed_at_delay_end is None else speed_at_delay_end
        delay_travel = calc_travel_between_speeds(speed, end, delay).to(m)
        braking = (speed ** 2 / (2 * deceleration)).to(m)
        stop = (delay_travel + braking).to(m)
        delay_positive = delay.to(ureg.second).magnitude > 0
        out = dict(
            speed=speed, delay=delay, deceleration=deceleration,
            delay_travel=delay_travel, braking_travel=braking, stop_distance=stop,
            braking_time=calc_braking_time(speed, deceleration),
            crossover_speed=calc_stopping_crossover_speed(delay, deceleration),
        )
        if delay_positive:
            out["braking_over_delay"] = _ratio(braking, delay_travel)
            out["delay_share_of_braking"] = _ratio(delay_travel, braking)
        if clear_distance is not None:
            spare = (clear_distance - stop).to(m)
            out.update(clear_distance=clear_distance, spare=spare,
                       fits=spare.magnitude >= 0, clear_over_stop=_ratio(clear_distance, stop))
            if delay_positive:
                out["speed_ceiling"] = calc_max_permitted_velocity(clear_distance, delay, deceleration)
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
