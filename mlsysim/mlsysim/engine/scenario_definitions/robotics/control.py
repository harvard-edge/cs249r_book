"""Physical-machine scenarios for control."""

from __future__ import annotations
from typing import Any, List, Optional, Tuple
from ....core.types import Quantity
from ..base import BaseScenario


class GravityBiasedJointStopScenario(BaseScenario):
    """One loaded arm joint at ``joint_speed`` braking with ``machine``'s peak torque while gravity
    holds ``bias_fraction`` of that peak along (descending) or against (ascending) the motion.

    Braking deceleration is net torque over ``inertia``; the joint stops in
    ``w^2 / (2 alpha)`` and the tool at ``lever`` travels that angle times the
    lever. The kinetic barrier ``h = theta - w^2 / (2 alpha_net)`` (descending)
    bounds the next torque by ``tau <= -tau_bias + I alpha_net (gamma h / w - 1)``;
    its negative is the braking torque the barrier demands at a tool clearance.
    With ``site``, the protective separation distance prices a person at the
    site's approach speed over the site's pre-brake delay plus the descending
    joint stop, the tool's reaction travel, and the tool's stop.

    Notes
    -----
    Equation: ``tool_stop_descending`` is the worked quantity; each property below states its equation and units.

    Assumptions: Single-joint inertia, lever arm, gravity bias, and available braking torque stay constant.

    Excludes: Multibody dynamics, torque slew, brake onset variation, and safety certification.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.LoadedArmJointGravityStop
    >>> value = case.tool_stop_descending
    >>> print(f"{value:.3f~P}")
    79.448 mm"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    machine: Any  # EmbodiedMachine: arm peak torque and TCP limit
    inertia: Quantity  # joint inertia with payload (kg m^2)
    lever: Quantity  # joint axis to tool point (m)
    joint_speed: Quantity  # approach speed of the joint (rad/s, as 1/s)
    bias_fraction: float  # share of peak torque gravity holds
    barrier_gain: Optional[Quantity] = None  # gamma (1/s)
    barrier_clearances: List[Quantity] = []  # tool clearances at which to price the barrier
    site: Optional[Any] = None  # EmbodiedSiteScenario: approach speed and pre-brake delay

    @property
    def torque_limit(self) -> Quantity:
        """Configured joint peak torque in N m.

        Equation:
            return self.machine.arm.max_torque.to('``N*m``')"""
        return self.machine.arm.max_torque.to("N*m")

    @property
    def bias(self) -> Quantity:
        """Constant gravitational bias torque in N m, specified fraction of peak torque.

        Equation:
            return (self.torque_limit * self.bias_fraction).to('``N*m``')"""
        return (self.torque_limit * self.bias_fraction).to("N*m")

    @property
    def torque_descending(self) -> Quantity:
        """Net descending braking torque in N m after subtracting gravity bias.

        Torque left for braking when gravity acts along the motion.

        Equation:
            return (self.torque_limit - self.bias).to('``N*m``')"""
        return (self.torque_limit - self.bias).to("N*m")

    @property
    def torque_ascending(self) -> Quantity:
        """Net ascending braking torque in N m after adding gravity assistance.

        Torque available for braking when gravity helps.

        Equation:
            return (self.torque_limit + self.bias).to('``N*m``')"""
        return (self.torque_limit + self.bias).to("N*m")

    def _alpha(self, torque) -> Quantity:
        return (torque / self.inertia).to("1/s**2")

    @property
    def alpha_free(self) -> Quantity:
        """Unbiased constant angular deceleration in inverse seconds squared.

        Equation:
            return self._alpha(self.torque_limit)"""
        return self._alpha(self.torque_limit)

    @property
    def alpha_descending(self) -> Quantity:
        """Descending constant angular deceleration in inverse seconds squared.

        Equation:
            return self._alpha(self.torque_descending)"""
        return self._alpha(self.torque_descending)

    @property
    def alpha_ascending(self) -> Quantity:
        """Ascending constant angular deceleration in inverse seconds squared.

        Equation:
            return self._alpha(self.torque_ascending)"""
        return self._alpha(self.torque_ascending)

    def stop_angle(self, alpha) -> float:
        """Joint stopping angle in radians at supplied positive angular deceleration.

        Joint stop angle in radians at deceleration ``alpha``.

        Equation:
            return (self.joint_speed ** 2 / (2 * alpha)).to('').magnitude"""
        return (self.joint_speed**2 / (2 * alpha)).to("").magnitude

    def tool_stop(self, alpha) -> Quantity:
        """Tool stopping arc length in mm at supplied angular deceleration.

        Equation:
            return (self.lever * self.stop_angle(alpha)).to('mm')"""
        return (self.lever * self.stop_angle(alpha)).to("mm")

    @property
    def tool_stop_free(self) -> Quantity:
        """Tool stopping arc length in mm without gravity bias.

        Equation:
            return self.tool_stop(self.alpha_free)"""
        return self.tool_stop(self.alpha_free)

    @property
    def tool_stop_descending(self) -> Quantity:
        """Tool stopping arc length in mm when gravity opposes braking.

        Equation:
            return self.tool_stop(self.alpha_descending)"""
        return self.tool_stop(self.alpha_descending)

    @property
    def tool_stop_ascending(self) -> Quantity:
        """Tool stopping arc length in mm when gravity assists braking.

        Equation:
            return self.tool_stop(self.alpha_ascending)"""
        return self.tool_stop(self.alpha_ascending)

    @property
    def overrun(self) -> Quantity:
        """Extra descending stop travel in mm compared with the unbiased stop.

        Tool travel the descending stop adds over the unbiased one.

        Equation:
            return (self.tool_stop_descending - self.tool_stop_free).to('mm')"""
        return (self.tool_stop_descending - self.tool_stop_free).to("mm")

    @property
    def stop_time_descending(self) -> Quantity:
        """Descending joint stopping duration in ms.

        Equation:
            return (self.joint_speed / self.alpha_descending).to('ms')"""
        return (self.joint_speed / self.alpha_descending).to("ms")

    @property
    def tcp_speed(self) -> Quantity:
        """Tool tangential speed in m/s at the specified lever arm.

        Equation:
            return (self.lever * self.joint_speed).to('m/s')"""
        return (self.lever * self.joint_speed).to("m/s")

    def barrier_margin(self, clearance) -> float:
        """Kinetic barrier margin in radians at the supplied tool clearance.

        Kinetic barrier h = theta - w^2 / (2 alpha_net) at a tool clearance (radians).

        Equation:
            return (clearance / self.lever).to('').magnitude - self.stop_angle(self.alpha_descending)"""
        return (clearance / self.lever).to("").magnitude - self.stop_angle(self.alpha_descending)

    def barrier_torque_demand(self, clearance) -> Quantity:
        """Required braking torque in N m from the stated kinetic-barrier inequality.

        Braking torque the barrier demands at a tool clearance (the bound's negative).

        Equation:
            h = self.barrier_margin(clearance)
            bound = (-self.bias + self.inertia * self.alpha_descending * (self.barrier_gain * h / self.joint_speed - 1)).to('``N*m``')
            return (-bound).to('``N*m``')"""
        h = self.barrier_margin(clearance)
        bound = (-self.bias + self.inertia * self.alpha_descending * (self.barrier_gain * h / self.joint_speed - 1)).to(
            "N*m"
        )
        return (-bound).to("N*m")

    @property
    def barrier_demands(self) -> List[Quantity]:
        """Required braking torques in N m for every listed clearance.

        Equation:
            return [self.barrier_torque_demand(c) for c in self.barrier_clearances]"""
        return [self.barrier_torque_demand(c) for c in self.barrier_clearances]

    @property
    def separation_person(self) -> Quantity:
        """Human approach travel in mm during reaction and descending joint stopping.

        Person's travel over the reaction (pre-brake delay) and the descending joint stop.

        Equation:
            return (self.site.v_human_approach * (self.site.tau_delay + self.stop_time_descending)).to('mm')"""
        return (self.site.v_human_approach * (self.site.tau_delay + self.stop_time_descending)).to("mm")

    @property
    def separation_reaction(self) -> Quantity:
        """Tool reaction travel in mm before braking begins.

        Tool travel during the reaction time.

        Equation:
            return (self.tcp_speed * self.site.tau_delay).to('mm')"""
        return (self.tcp_speed * self.site.tau_delay).to("mm")

    @property
    def separation_total(self) -> Quantity:
        """Total human, reaction, and stopping travel in mm before additional margins.

        Protective separation before intrusion and position margins.

        Equation:
            return (self.separation_person + self.separation_reaction + self.tool_stop_descending).to('mm')"""
        return (self.separation_person + self.separation_reaction + self.tool_stop_descending).to("mm")

    @property
    def separation_person_share(self) -> float:
        """Dimensionless human-travel share of the separation total.

        Equation:
            return (self.separation_person / self.separation_total).to('').magnitude"""
        return (self.separation_person / self.separation_total).to("").magnitude


class TorqueHalfSpaceProjectionScenario(BaseScenario):
    """A torque proposal against one barrier half-space ``normal . u <= bound``: the least-change
    projection (``calc_cbf_qp_orthogonal_projection``) against a per-axis ``clamp``.

    Torques are in N m, stored as plain floats like the projection helper's inputs.

    Notes
    -----
    Equation: ``projection`` is the worked quantity; each property below states its equation and units.

    Assumptions: One linear half-space constrains Euclidean torque correction; scalars represent N m.

    Excludes: Jointly enforcing actuator limits; actuator_limit is descriptive and the projection is unconstrained otherwise.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``calc_cbf_qp_orthogonal_projection``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.TwoJointTorqueProjection
    >>> value = case.projection
    >>> [round(item, 3) for item in value["u_projected"]]
    [1.5, 2.0]"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    proposal: List[float]
    normal: List[float]
    bound: float
    clamp: List[float]
    actuator_limit: float

    @property
    def projection(self) -> dict:
        """Least-change torque projection from the shared single-half-space helper.

        Equation:
            return calc_cbf_qp_orthogonal_projection(self.proposal, self.normal, self.bound)"""
        from ....physics.robotics import calc_cbf_qp_orthogonal_projection

        return calc_cbf_qp_orthogonal_projection(self.proposal, self.normal, self.bound)

    @staticmethod
    def _dot(a, b) -> float:
        return sum(x * y for x, y in zip(a, b))

    @property
    def proposal_value(self) -> float:
        """Proposed torque dot barrier normal, in the bound's scalar units.

        Equation:
            return self._dot(self.normal, self.proposal)"""
        return self._dot(self.normal, self.proposal)

    @property
    def breach(self) -> float:
        """Signed proposal constraint violation; positive means outside the half-space.

        Equation:
            return self.proposal_value - self.bound"""
        return self.proposal_value - self.bound

    @property
    def clamp_value(self) -> float:
        """Clamped torque dot barrier normal, in the bound's scalar units.

        Equation:
            return self._dot(self.normal, self.clamp)"""
        return self._dot(self.normal, self.clamp)

    @property
    def clamp_penetration(self) -> float:
        """Signed clamped proposal violation; positive means clamping still violates the barrier.

        Equation:
            return self.clamp_value - self.bound"""
        return self.clamp_value - self.bound

    @property
    def projected_value(self) -> float:
        """Projected torque dot barrier normal for checking the half-space inequality.

        Equation:
            return self._dot(self.normal, self.projection['u_projected'])"""
        return self._dot(self.normal, self.projection["u_projected"])

    @staticmethod
    def angle_deg(u) -> float:
        """Planar angle in degrees from the first two torque components.

        Equation:
            import math
            return math.degrees(math.atan2(u[1], u[0]))"""
        import math

        return math.degrees(math.atan2(u[1], u[0]))


class CycleBudgetScenario(BaseScenario):
    """Stage sub-budgets of one permission check against ``site``'s declared WCET and deadline.

    Notes
    -----
    Equation: ``total`` is the worked quantity; each property below states its equation and units.

    Assumptions: The named stage budgets add sequentially and the stage list is nonempty.

    Excludes: Execution-time measurement, parallel overlap, and additional blocking/interference.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseEnforcerCycleBudget
    >>> value = case.total
    >>> print(f"{value.m_as('microsecond'):.3f} microsecond")
    250.000 microsecond"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario: WCET, deadline, unloaded cycle
    stages: List[Tuple[str, Quantity]]

    @property
    def total(self) -> Quantity:
        """Sum of named stage time budgets in microseconds.

        Equation:
            return sum((q for _, q in self.stages), 0 * self.stages[0][1]).to('us')"""
        return sum((q for _, q in self.stages), 0 * self.stages[0][1]).to("us")

    def stage(self, name: str) -> Quantity:
        """Named stage duration in microseconds.

        Equation:
            return dict(self.stages)[name].to('us')"""
        return dict(self.stages)[name].to("us")

    @property
    def headroom(self) -> Quantity:
        """Microseconds left between the deadline and summed stage budgets.

        Deadline left after the declared worst case.

        Equation:
            return (self.site.enforcer_deadline - self.total).to('us')"""
        return (self.site.enforcer_deadline - self.total).to("us")

    def share(self, name: str) -> float:
        """Dimensionless named-stage share of total stage time.

        Equation:
            return (self.stage(name) / self.total).to('').magnitude"""
        return (self.stage(name) / self.total).to("").magnitude


class HoldIntervalDriftScenario(BaseScenario):
    """Drift while a drive holds the last command for one ``period``: travel at ``speed`` plus the
    half ``acceleration`` ``period^2`` an independent acceleration bound allows.

    Notes
    -----
    Equation: ``drift`` is the worked quantity; each property below states its equation and units.

    Assumptions: Speed and acceleration provide constant upper-bound terms throughout one held-command interval.

    Excludes: Jerk, direction reversal, and a proof that the supplied bounds hold.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseBaseHoldDrift
    >>> value = case.drift
    >>> print(f"{value:.3f~P}")
    1.501 mm"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    speed: Quantity
    acceleration: Quantity
    period: Quantity

    @property
    def drift(self) -> Quantity:
        """Millimeters traveled during a hold, constant-speed travel plus bounded-acceleration travel.

        Equation:
            return (self.speed * self.period + 0.5 * self.acceleration * self.period ** 2).to('mm')"""
        return (self.speed * self.period + 0.5 * self.acceleration * self.period**2).to("mm")
