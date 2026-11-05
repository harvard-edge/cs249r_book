"""Physical-machine scenarios for grounding."""

from __future__ import annotations
from typing import Any, List, Optional
from ....core.types import Quantity
from ..base import BaseScenario


class IntentCadenceScenario(BaseScenario):
    """``site``'s conveyor target proposed by ``machine``'s intent model: evidence horizons against
    the model's cadence, a host stall of ``stall``, and the stop onset after expiry.

    ``target_speed`` defaults to the conveyor's; set it for a slower drift.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: The target follows the supplied speed/slip bounds and the intent producer has fixed cadence and latency.

    Excludes: Data-dependent model latency, geometric planning, and verified command safety.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``IntentCadenceModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ConveyorMugIntentCadence
    >>> value = case.result
    >>> print(f"{value.tracked_horizon:.3f~P}")
    219.089 ms"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario (tolerance, error, belt, slip, tick, bus, onset)
    machine: Any  # EmbodiedMobileManipulator (intent rate and latency)
    stall: Quantity
    target_speed: Optional[Quantity] = None

    @property
    def speed(self) -> Quantity:
        """Target speed in length/time; use conveyor speed unless explicitly overridden.

        Equation:
            return self.site.v_conveyor if self.target_speed is None else self.target_speed"""
        return self.site.v_conveyor if self.target_speed is None else self.target_speed

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.IntentCadenceModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = IntentCadenceModel().solve(...)`` using the fields shown below."""
        from ...solvers import IntentCadenceModel

        s, rm = self.site, self.machine
        return IntentCadenceModel().solve(
            s.grasp_tolerance,
            s.grasp_initial_error,
            self.speed,
            s.a_conveyor_slip,
            rm.intent_rate,
            rm.intent_inference_latency,
            self.stall,
            s.t_tick,
            s.t_bus,
            s.t_brake_onset,
        )


class DynamicReachScenario(BaseScenario):
    """A target ``target_distance`` from the gripper at ``cadence.machine``'s TCP speed ceiling,
    against the cadence scenario's evidence windows.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: Tool travel is compared at its speed ceiling against the cadence scenario's evidence horizons.

    Excludes: Acceleration ramps, path obstacles, and reachability certification.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``DynamicReachModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ConveyorMugDynamicReach
    >>> value = case.result
    >>> print(f"{value.transit_floor:.3f~P}")
    300.000 ms"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    cadence: IntentCadenceScenario
    target_distance: Quantity

    @property
    def speed_ceiling(self) -> Quantity:
        """Configured tool-center-point speed ceiling in length/time.

        Equation:
            return self.cadence.machine.tcp_speed_limit"""
        return self.cadence.machine.tcp_speed_limit

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.DynamicReachModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = DynamicReachModel().solve(...)`` using the fields shown below."""
        from ...solvers import DynamicReachModel

        c = self.cadence.result
        return DynamicReachModel().solve(
            self.speed_ceiling, self.target_distance, c.tracked_horizon, c.untracked_horizon
        )


class ForwardOccupancyScenario(BaseScenario):
    """A coworker's forward occupancy set at ``cadence.site``'s design approach speed, screened over
    the cadence scenario's evidence windows and over a ``forecast`` horizon, against the aisle.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: A person can expand the forward occupancy set at the supplied approach speed.

    Excludes: Human acceleration, directional intent probabilities, and a spatial motion planner.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``ForwardOccupancyModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.AisleCoworkerForwardOccupancy
    >>> value = case.result
    >>> print(f"{value.growth_forecast:.3f~P}")
    3.200 m"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    cadence: IntentCadenceScenario
    forecast: Quantity

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.ForwardOccupancyModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = ForwardOccupancyModel().solve(...)`` using the fields shown below."""
        from ...solvers import ForwardOccupancyModel

        s, c = self.cadence.site, self.cadence.result
        return ForwardOccupancyModel().solve(
            s.v_human_approach, s.aisle_width, c.untracked_horizon, c.tracked_horizon, self.forecast
        )


class IntentLeaseExpiryScenario(BaseScenario):
    """The intent lease for the cadence scenario's target: its evidence horizons, an
    ``administrative_lease``, and the stopping deadline from ``clearance`` at the TCP speed ceiling,
    with the arm's credible deceleration from its handbook worst-case stop.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: The existing cadence horizons and configured stop timing constrain one administrative lease.

    Excludes: Actuator identification, changing braking capability, and safety certification.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``IntentLeaseExpiryModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ConveyorMugIntentLeaseExpiry
    >>> value = case.result
    >>> print(f"{value.expiry_tracked:.3f~P}")
    100.000 ms"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    cadence: IntentCadenceScenario
    administrative_lease: Quantity
    clearance: Quantity

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.IntentLeaseExpiryModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = IntentLeaseExpiryModel().solve(...)`` using the fields shown below."""
        from ...solvers import IntentLeaseExpiryModel

        c, rm = self.cadence.result, self.cadence.machine
        return IntentLeaseExpiryModel().solve(
            c.untracked_horizon,
            c.tracked_horizon,
            self.administrative_lease,
            c.stop_onset,
            rm.tcp_speed_limit,
            rm.arm.cartesian_velocity_limit,
            rm.arm.emergency_stop_time,
            self.clearance,
        )


class ReferentSetScenario(BaseScenario):
    """Raw per-box ``scores`` of retained candidates screened by a conformal threshold calibrated on
    ``calibration_count`` scenes at miss rate ``alpha``, whose calibration score is ``threshold_score``.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: Calibration count and score threshold are supplied, with scores interpreted by ReferentSetModel.

    Excludes: Calibration fitting, exchangeability validation, and semantic correctness of retained labels.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``ReferentSetModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.MugJarReferentSet
    >>> value = case.result
    >>> value.set_size
    2"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    labels: List[str]
    scores: List[float]
    calibration_count: int
    alpha: float
    threshold_score: float  # qhat

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.ReferentSetModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = ReferentSetModel().solve(...)`` using the fields shown below."""
        from ...solvers import ReferentSetModel

        return ReferentSetModel().solve(self.scores, self.calibration_count, self.alpha, self.threshold_score)


class ConfidenceRegionScenario(BaseScenario):
    """A Gaussian position estimate with principal standard deviations ``sigmas``: its ``coverage``
    region against ``site``'s grasp tolerance.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: Principal-axis errors follow the Gaussian geometry assumed by ConfidenceRegionModel.

    Excludes: Non-Gaussian tails, covariance estimation, and grasp-success prediction.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``ConfidenceRegionModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.DistantHandleConfidenceRegion
    >>> value = case.result
    >>> round(value.worst_over_tolerance, 3)
    2.236"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario (grasp tolerance)
    sigmas: List[Quantity]
    coverage: float

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.ConfidenceRegionModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = ConfidenceRegionModel().solve(...)`` using the fields shown below."""
        from ...solvers import ConfidenceRegionModel

        return ConfidenceRegionModel().solve(self.sigmas, self.coverage, self.site.grasp_tolerance)


class GoalSetMembershipScenario(BaseScenario):
    """Terminal TCP ``offsets`` along one axis tested against a goal set whose semi-axes all equal
    ``site``'s grasp tolerance: the normalized ellipsoidal distance, inside when at most one.

    Notes
    -----
    Equation: ``norms`` is the worked quantity; each property below states its equation and units.

    Assumptions: Offsets are nonnegative distances along one axis and tolerance is positive.

    Excludes: Multiaxis ellipsoid membership; negative signed offsets are not absolutized by this implementation.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ConveyorMugGoalSetOffsets
    >>> value = case.norms
    >>> [round(item, 3) for item in value]
    [0.667, 1.333]"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario (grasp tolerance)
    offsets: List[Quantity]

    @property
    def norms(self) -> List[float]:
        """Signed axial offsets divided by grasp tolerance, as dimensionless ratios.

        Equation:
            return [(d / self.site.grasp_tolerance).to('').magnitude for d in self.offsets]"""
        return [(d / self.site.grasp_tolerance).to("").magnitude for d in self.offsets]

    @property
    def inside(self) -> List[bool]:
        """Membership flags from ratios at most one; negative offsets are not absolutized.

        Equation:
            return [n <= 1.0 for n in self.norms]"""
        return [n <= 1.0 for n in self.norms]
