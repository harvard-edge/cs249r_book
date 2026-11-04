"""Physical-machine scenarios for site limits."""

from __future__ import annotations
from typing import Any, Optional
from ....core.types import Quantity
from ....core.units import Q_
from ..base import BaseScenario


class PackagingCellScenario(BaseScenario):
    """A packaging cell's carriage that must stop before a fixed stop line.

    The pre-brake delay is a chain of four allocations (event to usable frame,
    transfer through admission, command delivery, brake onset); the stop is the
    elementary constant-deceleration budget (``StoppingBudgetModel``) with the
    allocated reserve held back from the clearance, so the speed ceiling is the
    positive root of ``v tau + v^2/(2a) = clearance - reserve``. The supervisory
    settings (force limiter, thermistor trip, stall timeout, bus overvoltage trip,
    kinematic-check rate) are selected thresholds, not measured limits.

    Notes
    -----
    Equation: Stopping uses ``v*delay+v**2/(2*deceleration)`` with reserved clearance; energy
    uses ``mass*speed**2/2``.

    Assumptions: Delay components add serially and braking deceleration is constant and
    positive.

    Excludes: Measured actuator dynamics, traction variation, threshold validation, and a
    safety certification.

    References
    ----------
    Derivation: the scenario composes
    ``mlsysim.physics.robotics.actuation.calc_kinetic_energy``,
    ``mlsysim.solvers.StoppingBudgetModel``. Their contracts supply the underlying
    equations; property equations show the remaining unit conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.PackagingCellTransit
    >>> value = scenario.delay
    >>> str(value.units)
    'ms'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    mass: Quantity  # total moving mass, including the load
    speed: Quantity
    clearance: Quantity  # to the fixed stop line
    reserve: Quantity  # allocated uncertainty and clearance reserve
    deceleration: Quantity  # braking lower bound after the delay
    exposure: Quantity  # event to usable frame
    processing: Quantity  # transfer, inference, scheduling, and admission
    bus: Quantity  # admitted command to brake-driver delivery
    onset: Quantity  # delivery to full assumed deceleration
    check_rate: Optional[Quantity] = None
    force_limit: Optional[Quantity] = None
    temperature_trip: Optional[Quantity] = None
    stall_timeout: Optional[Quantity] = None
    bus_trip: Optional[Quantity] = None

    @property
    def delay(self) -> Quantity:
        """Total allocated pre-brake latency in milliseconds, summing exposure, processing, bus,
        and onset.

        Equation: ``(exposure + processing + bus + onset).to('ms')``.
        """
        return (self.exposure + self.processing + self.bus + self.onset).to("ms")

    @property
    def stop_budget(self) -> Quantity:
        """Usable stopping clearance after physical reserve, in millimeters.

        Equation: ``(clearance - reserve).to('mm')``.
        """
        return (self.clearance - self.reserve).to("mm")

    @property
    def result(self):
        """Return the typed accounting result from ``StoppingBudgetModel`` using this scenario's
        declared inputs; units and exclusions follow that solver contract.

        Equation: ``StoppingBudgetModel().solve(speed, delay, deceleration, clearance,
        physical_margin=reserve)``.
        """
        from ...solvers import StoppingBudgetModel

        return StoppingBudgetModel().solve(
            self.speed, self.delay, self.deceleration, self.clearance, physical_margin=self.reserve
        )

    @property
    def remaining(self) -> Quantity:
        """Millimeters between the modeled stopping position and stop line, including the reserved
        margin.

        Equation: ``(clearance - result.stop_distance).to('mm')``.
        """
        return (self.clearance - self.result.stop_distance).to("mm")

    @property
    def deficit(self) -> Quantity:
        """Modeled stopping distance minus clearance in millimeters; negative values mean spare
        clearance.

        Equation: ``(result.stop_distance - clearance).to('mm')``.
        """
        return (self.result.stop_distance - self.clearance).to("mm")

    @property
    def kinetic_energy(self) -> Quantity:
        """Carriage kinetic energy in joules via the shared mass-and-speed equation.

        Equation: ``calc_kinetic_energy(mass, speed).to('J')``.
        """
        from ....physics.robotics.actuation import calc_kinetic_energy

        return calc_kinetic_energy(self.mass, self.speed).to("J")

    @property
    def root_terms(self):
        """Speed-root operands ``a*tau``, ``2*a*d``, and their combined square root in consistent
        SI units.

        Equation: ``(a_tau, two_a_d, ((a_tau ** 2 + two_a_d) ** 0.5).to('m/s'))``.
        """
        a_tau = (self.deceleration * self.delay).to("m/s")
        two_a_d = (2 * self.deceleration * self.stop_budget).to("m**2/s**2")
        return a_tau, two_a_d, ((a_tau**2 + two_a_d) ** 0.5).to("m/s")


class ShieldCoverageBillScenario(BaseScenario):
    """A fleet's zero-event proposal log and a shield's tested coverage against a hazard target.

    Notes
    -----
    Equation: The result property delegates to the named solver documented below; it
    converts declared scenario units before passing the inputs.

    Assumptions: Zero-event exposure and tested shield coverage use the declared statistical
    model and common-cause rate.

    Excludes: Certification of a hazard target or evidence that the optional
    hazardous_proposals field was incorporated into solve.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.solvers.ShieldCoverageBillModel``. Their
    contracts supply the underlying equations; property equations show the remaining unit
    conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ShieldCoverageBill
    >>> value = scenario.result
    >>> type(value).__name__
    'ShieldCoverageBillResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    fleet_units: int
    cycle_days: Quantity
    hours_per_day: Quantity  # operating hours per unit per calendar day
    coverage: float  # class-specific interception coverage
    common_cause_rate: Quantity  # hardware and common-cause residual, per hour
    target_rate: Quantity  # system hazard target, per hour
    confidence: float
    hazardous_proposals: int = 0

    @property
    def result(self):
        """Return the typed accounting result from ``ShieldCoverageBillModel`` using this
        scenario's declared inputs; units and exclusions follow that solver contract.

        Equation: ``ShieldCoverageBillModel().solve(fleet_units, cycle_days, hours_per_day,
        coverage, common_cause_rate, target_rate, confidence)``.
        """
        from ...solvers import ShieldCoverageBillModel

        return ShieldCoverageBillModel().solve(
            self.fleet_units,
            self.cycle_days,
            self.hours_per_day,
            self.coverage,
            self.common_cause_rate,
            self.target_rate,
            self.confidence,
        )


class LatchSeriesSpringScenario(BaseScenario):
    """A series spring sized so ``machine``'s arm stays under ``site``'s declared latch floor.

    The arm's stop is held to what the guarded approach on the rigid plate needs
    (``calc_latch_contact_stopping_profile``); at the fast approach the series
    stiffness must keep tripwire force plus driven stroke under the floor
    (``calc_passive_compliance_stiffness_bound``), and the spring element is the
    stiffness that gives it in series with the plate. Coasting peaks price the
    case where the response cuts torque at the tripwire instead
    (``calc_coasting_contact_peak_force``).

    Notes
    -----
    Equation: The stopping and compliance helpers bound driven stroke and effective
    stiffness; complementary series stiffness determines the added spring.

    Assumptions: The declared latch force budget, tripwire force, response delay, and
    effective arm mass describe one contact model.

    Excludes: Nonlinear material response, damping uncertainty, contact geometry variation,
    and certified human-contact safety.

    References
    ----------
    Derivation: the scenario composes
    ``mlsysim.physics.robotics.actuation.calc_coasting_contact_peak_force``,
    ``mlsysim.physics.robotics.actuation.calc_passive_compliance_stiffness_bound``,
    ``mlsysim.physics.robotics.learning.calc_complementary_series_stiffness``,
    ``mlsysim.physics.robotics.stopping.calc_latch_contact_stopping_profile``. Their
    contracts supply the underlying equations; property equations show the remaining unit
    conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseLatchSeriesSpring
    >>> value = scenario.arm_deceleration
    >>> str(value.units)
    'm/s²'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario (latch stiffness, floor, tripwire, speeds)
    machine: Any  # EmbodiedMobileManipulator (arm effective contact mass)
    preload: Quantity = Q_(0.0, "N")

    def _profile(self, speed: Quantity, stiffness: Optional[Quantity] = None):
        from ....physics.robotics.stopping import calc_latch_contact_stopping_profile

        return calc_latch_contact_stopping_profile(
            approach_velocity=speed, latch_stiffness=stiffness, scenario=self.site
        )

    @property
    def arm_deceleration(self) -> Quantity:
        """Required arm deceleration in m/s² for the guarded rigid-plate latch approach.

        Equation: ``_profile(site.v_latch_approach)['a_arm_needed'].to('m/s**2')``.
        """
        return self._profile(self.site.v_latch_approach)["a_arm_needed"].to("m/s**2")

    @property
    def arm_deceleration_rigid_fast(self) -> Quantity:
        """Required arm deceleration in m/s² for the faster rigid-plate approach.

        Equation: ``_profile(site.v_latch_high)['a_arm_needed'].to('m/s**2')``.
        """
        return self._profile(self.site.v_latch_high)["a_arm_needed"].to("m/s**2")

    def _bound(self):
        from ....physics.robotics.actuation import calc_passive_compliance_stiffness_bound

        s = self.site
        return calc_passive_compliance_stiffness_bound(
            s.f_latch_limit, s.f_latch_tripwire, s.v_latch_high, s.t_contact_response, self.arm_deceleration
        )

    @property
    def driven_stroke(self) -> Quantity:
        """Post-tripwire driven stroke in millimeters from the passive-compliance bound.

        Equation: ``_bound()[1].to('mm')``.
        """
        return self._bound()[1].to("mm")

    @property
    def series_stiffness_max(self) -> Quantity:
        """Maximum effective series stiffness in N/m under the declared force budget.

        Equation: ``_bound()[0].to('N/m')``.
        """
        return self._bound()[0].to("N/m")

    @property
    def spring_stiffness_max(self) -> Quantity:
        """Maximum added spring stiffness in N/m that realizes the effective series bound with the
        latch plate.

        Equation: ``calc_complementary_series_stiffness(series_stiffness_max,
        site.k_latch).to('N/m')``.
        """
        from ....physics.robotics.learning import calc_complementary_series_stiffness

        return calc_complementary_series_stiffness(self.series_stiffness_max, self.site.k_latch).to("N/m")

    @property
    def spring_stroke(self) -> Quantity:
        """Spring deflection in millimeters at the declared force ceiling after subtracting
        preload.

        Equation: ``((site.f_latch_limit - preload) / spring_stiffness_max).to('mm')``.
        """
        return ((self.site.f_latch_limit - self.preload) / self.spring_stiffness_max).to("mm")

    def coasting_peak(self, stiffness: Quantity, speed: Quantity) -> Quantity:
        """Peak coasting contact force in newtons from speed, effective arm mass, stiffness, and
        tripwire force.

        Equation: ``calc_coasting_contact_peak_force(speed, machine.arm_effective_contact_mass,
        stiffness, site.f_latch_tripwire).to('N')``.
        """
        from ....physics.robotics.actuation import calc_coasting_contact_peak_force

        return calc_coasting_contact_peak_force(
            speed, self.machine.arm_effective_contact_mass, stiffness, self.site.f_latch_tripwire
        ).to("N")

    @property
    def coasting_rigid_guarded(self) -> Quantity:
        """Peak coasting force in newtons for the guarded approach against the rigid plate.

        Equation: ``coasting_peak(site.k_latch, site.v_latch_approach)``.
        """
        return self.coasting_peak(self.site.k_latch, self.site.v_latch_approach)

    @property
    def coasting_rigid_fast(self) -> Quantity:
        """Peak coasting force in newtons for the fast approach against the rigid plate.

        Equation: ``coasting_peak(site.k_latch, site.v_latch_high)``.
        """
        return self.coasting_peak(self.site.k_latch, self.site.v_latch_high)

    @property
    def coasting_spring_fast(self) -> Quantity:
        """Peak coasting force in newtons for the fast approach with the bounded effective
        stiffness.

        Equation: ``coasting_peak(series_stiffness_max, site.v_latch_high)``.
        """
        return self.coasting_peak(self.series_stiffness_max, self.site.v_latch_high)

    @property
    def spring_fast_profile(self):
        """Stopping profile at fast approach using the bounded effective series stiffness.

        Equation: ``_profile(site.v_latch_high, series_stiffness_max)``.
        """
        return self._profile(self.site.v_latch_high, self.series_stiffness_max)


class RestrictedTransitScenario(BaseScenario):
    """``site``'s aisle leg at ``restricted_speed`` against its normal aisle speed.

    Notes
    -----
    Equation: Traversal time is aisle_length/speed; differences compare restricted and
    normal constant-speed traversal.

    Assumptions: Both speeds are positive and the same aisle length is used for both cases.

    Excludes: Acceleration, braking, turns, obstacles, and queuing along the aisle.

    References
    ----------
    Derivation: Traversal time is aisle_length/speed; differences compare restricted and
    normal constant-speed traversal. This is declared scenario accounting rather than a
    fitted empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseRestrictedTransit
    >>> value = scenario.speed_cut
    >>> round(float(value), 3)
    0.231
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario (aisle speed, aisle length)
    restricted_speed: Quantity

    @property
    def speed_cut(self) -> float:
        """Fractional speed reduction relative to normal aisle speed.

        Equation: ``float((1 - restricted_speed / site.v_aisle).to('').magnitude)``.
        """
        return float((1 - self.restricted_speed / self.site.v_aisle).to("").magnitude)

    @property
    def leg_time_nominal(self) -> Quantity:
        """Constant-speed aisle traversal duration in seconds at the normal speed.

        Equation: ``(site.aisle_length / site.v_aisle).to('s')``.
        """
        return (self.site.aisle_length / self.site.v_aisle).to("s")

    @property
    def leg_time_restricted(self) -> Quantity:
        """Constant-speed aisle traversal duration in seconds at the restricted speed.

        Equation: ``(site.aisle_length / restricted_speed).to('s')``.
        """
        return (self.site.aisle_length / self.restricted_speed).to("s")

    @property
    def leg_time_added(self) -> Quantity:
        """Restricted minus normal traversal duration, in seconds.

        Equation: ``(leg_time_restricted - leg_time_nominal).to('s')``.
        """
        return (self.leg_time_restricted - self.leg_time_nominal).to("s")

    @property
    def time_increase(self) -> float:
        """Additional traversal duration divided by nominal duration, dimensionless.

        Equation: ``float((leg_time_added / leg_time_nominal).to('').magnitude)``.
        """
        return float((self.leg_time_added / self.leg_time_nominal).to("").magnitude)
