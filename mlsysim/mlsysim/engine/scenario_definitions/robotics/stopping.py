"""Physical-machine scenarios for stopping."""

from __future__ import annotations
from typing import Any, Optional
from ....core.types import Quantity
from ..base import BaseScenario


class StoppingBudgetScenario(BaseScenario):
    """A machine at ``speed`` that must stop: travel during ``delay``, then braking at
    ``deceleration``.

    ``clear_distance`` closes the budget when one is stated. ``speed_at_delay_end``
    replaces constant-speed delay travel with the trapezoid between two recorded
    speeds; braking is still from ``speed``. With a ``site`` (an embodied site
    scenario), the deceleration and clear distance default to the site's, and
    its localization bound and protective margin join the required distance.
    ``added_delay`` prices one more delay step as travel at ``speed``.
    The finished budget adds three optional terms: ``jerk`` (the resident
    S-curve stop's jerk bound), ``tracking_bound`` (the tracking inset), and
    ``approach_speed`` (a person closing the gap through delay and stop).

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.StoppingBudgetModel``.

    Assumptions: The selected braking limit, clearance, and site margins are credible
    scenario inputs; the stopping solver determines the kinematic budget.

    Excludes: The calculation is not a safety certification, and unmodeled traction or
    control failures remain outside it.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.StoppingBudgetModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseAisleStopAtDriveLimit
    >>> str(case.a_brake)
    '2.0 m/s²'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    speed: Quantity
    delay: Quantity
    deceleration: Optional[Quantity] = None  # default: the site's credible deceleration
    clear_distance: Optional[Quantity] = None  # default: the site's clear distance
    speed_at_delay_end: Optional[Quantity] = None
    site: Optional[Any] = None  # EmbodiedSiteScenario
    added_delay: Optional[Quantity] = None
    jerk: Optional[Quantity] = None  # S-curve stop jerk bound (added 2026-10-20)
    tracking_bound: Optional[Quantity] = None  # tracking inset (added 2026-10-20)
    approach_speed: Optional[Quantity] = None  # approaching person (added 2026-10-20)

    @property
    def a_brake(self) -> Quantity:
        """Explicit braking deceleration, falling back to the site's value. Equation:
        ``deceleration`` when present, otherwise ``site.a_brake``. Units: acceleration
        magnitude; this is a specified credible bound, not a friction measurement.
        """
        return self.site.a_brake if self.deceleration is None else self.deceleration

    @property
    def d_clear(self) -> Optional[Quantity]:
        """Explicit available clearance, otherwise the site's clearance when a site exists.
        Equation: select ``clear_distance`` or ``site.d_clear``. Units: length, or None when no
        clearance is specified.
        """
        if self.clear_distance is not None or self.site is None:
            return self.clear_distance
        return self.site.d_clear

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.StoppingBudgetModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ...solvers import StoppingBudgetModel

        loc = None if self.site is None else self.site.delta_loc
        margin = None if self.site is None else self.site.delta_margin
        return StoppingBudgetModel().solve(
            self.speed,
            self.delay,
            self.a_brake,
            self.d_clear,
            self.speed_at_delay_end,
            loc,
            margin,
            self.added_delay,
            self.jerk,
            self.tracking_bound,
            self.approach_speed,
        )


class TractionLossScenario(BaseScenario):
    """Floor friction falls from ``mu_before`` to ``mu_after`` under a machine that assumes
    ``deceleration``.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.TractionLossModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.TractionLossModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.TractionLossModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.TractionLossModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.HydraulicFluidSlickTraction
    >>> type(case.result).__name__
    'TractionLossResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    mu_before: float
    mu_after: float
    deceleration: Quantity
    drive_load_share: float = 1.0

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.TractionLossModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ...solvers import TractionLossModel

        return TractionLossModel().solve(self.mu_before, self.mu_after, self.deceleration, self.drive_load_share)


class TravelDuringLatencyScenario(BaseScenario):
    """Distance a vehicle at ``speed`` covers during ``latency``; optionally a remote-path
    delay range.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: Speed is constant over the supplied delay interval.

    Excludes: Acceleration, stopping dynamics, and trajectory uncertainty are absent.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.EmergencyBraking100kmh
    >>> str(case.distance)
    '2.7777777777777777 m'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    speed: Quantity
    latency: Quantity
    remote_delay_low: Optional[Quantity] = None
    remote_delay_high: Optional[Quantity] = None

    @property
    def distance(self) -> Quantity:
        """Distance traveled at constant speed during the latency interval. Equation: ``speed *
        latency``. Units: meters; acceleration is excluded.
        """
        return (self.speed * self.latency).to("meter")


class SpeedStepScenario(BaseScenario):
    """Raising the speed setpoint from ``base`` to ``raised`` (two stopping budgets at one
    deceleration).

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.SpeedStepModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.SpeedStepModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.SpeedStepModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.SpeedStepModel``. The methods below identify any additional
    arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseBaseSpeedStep
    >>> type(case.result).__name__
    'SpeedStepResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    base: StoppingBudgetScenario
    raised: StoppingBudgetScenario

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.SpeedStepModel.solve``. Equation: pass
        the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ...solvers import SpeedStepModel

        return SpeedStepModel().solve(self.base.speed, self.raised.speed, self.base.a_brake)
