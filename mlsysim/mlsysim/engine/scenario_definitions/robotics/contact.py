"""Physical-machine scenarios for contact."""

from __future__ import annotations
import math as _math
from typing import Any
from ....core.types import Quantity
from ....core.units import Q_
from ..base import BaseScenario


class HeadingDriftScenario(BaseScenario):
    """A persistent ``heading_offset`` at ``machine``'s drive limit in ``site``'s aisle: the
    angular error an open-loop replay scores, and where the closed loop spends the side
    clearance
    (``y = v sin(eps) t``).

    Notes
    -----
    Equation: Lateral velocity equals speed times sin(heading); clearance divided by lateral
    velocity gives the deadline.

    Assumptions: A constant nonzero heading error and constant drive speed on a straight
    aisle.

    Excludes: Feedback correction, acceleration, and varying terrain.

    References
    ----------
    Derivation: Lateral velocity equals speed times sin(heading); clearance divided by
    lateral velocity gives the deadline. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.AisleHeadingDrift
    >>> round(float(scenario.speed.magnitude), 4)
    1.5"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    machine: Any
    site: Any
    heading_offset: Quantity

    @property
    def speed(self) -> Quantity:
        """Machine's maximum drive speed from its registered base specification.

        Equation: ``machine.base.max_velocity``."""
        return self.machine.base.max_velocity

    @property
    def heading(self) -> float:
        """Persistent heading error as a dimensionless angle in radians.

        The offset in radians.

        Equation: ``_math.radians(heading_offset.to('degree').magnitude)``."""
        return _math.radians(self.heading_offset.to("degree").magnitude)

    @property
    def angular_mse(self) -> Quantity:
        """Squared persistent heading error in radians squared.

        Equation: ``(Q_(heading, 'radian') ** 2).to('radian**2')``."""
        return (Q_(self.heading, "radian") ** 2).to("radian**2")

    @property
    def time_to_clearance(self) -> Quantity:
        """Seconds until constant lateral velocity ``speed*sin(heading)`` spends the available side
        clearance.

        Equation: ``(site.side_clearance / (speed * _math.sin(heading))).to('s')``."""
        return (self.site.side_clearance / (self.speed * _math.sin(self.heading))).to("s")

    @property
    def distance_to_clearance(self) -> Quantity:
        """Forward path length in meters traveled before the lateral clearance is spent.

        Equation: ``(speed * time_to_clearance).to('m')``."""
        return (self.speed * self.time_to_clearance).to("m")

    @property
    def aisle_fraction(self) -> float:
        """Dimensionless path-to-clearance length divided by aisle length.

        Equation: ``(distance_to_clearance / site.aisle_length).to('').magnitude``."""
        return (self.distance_to_clearance / self.site.aisle_length).to("").magnitude


class StiffContactScenario(BaseScenario):
    """A stiff contact of ``stiffness`` (a linear spring, ``F = k x``): the force a
    ``residual``
    penetration commands, and the penetrations at ``site``'s latch tripwire and limit.

    Notes
    -----
    Equation: Hooke's law gives force=``k*penetration`` and its inverse gives penetration at a
    force limit.

    Assumptions: A linear spring of positive stiffness with the supplied penetration.

    Excludes: Impact dynamics, damping, controller saturation, and nonlinear compliance.

    References
    ----------
    Derivation: Hooke's law gives force=``k*penetration`` and its inverse gives penetration at a
    force limit. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.StiffSurfaceResidual
    >>> round(float(scenario.residual_force.magnitude), 4)
    1000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any
    stiffness: Quantity
    residual: Quantity

    @property
    def residual_force(self) -> Quantity:
        """Newtons commanded by the residual penetration under the linear spring law F=k*x.

        Equation: ``(stiffness * residual).to('N')``."""
        return (self.stiffness * self.residual).to("N")

    def penetration_at(self, force: Quantity) -> Quantity:
        """Millimeters of linear-spring penetration corresponding to supplied force ``force``.

        Equation: ``(force / stiffness).to('mm')``."""
        return (force / self.stiffness).to("mm")

    @property
    def tripwire_penetration(self) -> Quantity:
        """Millimeters at which the spring force reaches the site's latch tripwire.

        Equation: ``penetration_at(site.f_latch_tripwire)``."""
        return self.penetration_at(self.site.f_latch_tripwire)

    @property
    def limit_penetration(self) -> Quantity:
        """Millimeters at which the spring force reaches the site's latch-force limit.

        Equation: ``penetration_at(site.f_latch_limit)``."""
        return self.penetration_at(self.site.f_latch_limit)


class ForceTailScenario(BaseScenario):
    """A contact force over ``trials`` trials with a benign ``mean_force`` and a ``p99_force``
    tail,
    against ``site``'s tripwire and structural limit.

    Notes
    -----
    Equation: Subtract the structural limit from the tail force and divide the tail by the
    mean for scale comparisons.

    Assumptions: The caller supplies both mean and p99 force for the stated trials.

    Excludes: Estimating quantiles, uncertainty intervals, and a complete force
    distribution.

    References
    ----------
    Derivation: Subtract the structural limit from the tail force and divide the tail by the
    mean for scale comparisons. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.LatchForceTail
    >>> round(float(scenario.tail_over_limit.magnitude), 4)
    15.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any
    trials: int
    mean_force: Quantity
    p99_force: Quantity

    @property
    def tail_over_limit(self) -> Quantity:
        """Newtons by which the supplied p99 force exceeds the site's latch limit; negative values
        remain below it.

        Equation: ``(p99_force - site.f_latch_limit).to('N')``."""
        return (self.p99_force - self.site.f_latch_limit).to("N")

    @property
    def tail_over_mean(self) -> float:
        """Dimensionless ratio of the supplied p99 force to the supplied mean force.

        Equation: ``(p99_force / mean_force).to('').magnitude``."""
        return (self.p99_force / self.mean_force).to("").magnitude
