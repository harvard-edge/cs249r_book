"""Physical-machine scenarios for planning."""

from __future__ import annotations
from typing import Any, Optional
from ....core.types import Quantity
from ..base import BaseScenario


class TwoLinkJointRateScenario(BaseScenario):
    """``machine``'s arm as a planar two-link chain at ``elbow_angle``: the joint rates its TCP
    speed limit asks in the weakest direction, exact and with damped least squares at ``damping``.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.TwoLinkJointRateModel.solve``.

    Assumptions: Inputs are routed to ``TwoLinkJointRateModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to TwoLinkJointRateModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseArmRightAngleElbow
    >>> value = case.result.joint_rate_norm
    >>> print(f"{value:.3g~}")
    4.9 rad / s"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    machine: Any  # MobileManipulator
    elbow_angle: Quantity
    damping: Quantity

    @property
    def result(self):
        """Evaluate the stated two link joint rate case through ``TwoLinkJointRateModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.TwoLinkJointRateModel.solve``."""
        from ...solvers import TwoLinkJointRateModel

        arm = self.machine.arm
        return TwoLinkJointRateModel().solve(
            arm.upper_arm_length,
            arm.forearm_length,
            self.elbow_angle,
            self.machine.tcp_speed_limit,
            arm.joint_velocity_limits[1],
            self.damping,
        )


class HandoffExposureScenario(BaseScenario):
    """A replanner at ``replan_rate`` over ``mission_time`` whose handoffs are each late with
    ``late_probability``, held to ``mission_budget`` for the whole mission.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.HandoffExposureModel.solve``.

    Assumptions: Inputs are routed to ``HandoffExposureModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to HandoffExposureModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.GenericReplannerMissionExposure
    >>> value = case.result.all_on_time
    >>> print(f"{value:.3g}")
    8.05e-14"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    mission_time: Quantity
    replan_rate: Quantity
    late_probability: float
    mission_budget: float

    @property
    def result(self):
        """Evaluate the stated handoff exposure case through ``HandoffExposureModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.HandoffExposureModel.solve``."""
        from ...solvers import HandoffExposureModel

        return HandoffExposureModel().solve(
            self.mission_time, self.replan_rate, self.late_probability, self.mission_budget
        )


class ChunkReplacementScenario(BaseScenario):
    """Replacing ``machine``'s admitted chunk at ``site``: capture-to-admission latency against the
    renewal period and the chunk lease, with an ``interference_tail`` on the inference pass.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.ChunkReplacementModel.solve``.

    Assumptions: Inputs are routed to ``ChunkReplacementModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to ChunkReplacementModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseArmChunkReplacement
    >>> value = case.renewal_period
    >>> print(f"{value:.3g~}")
    50 ms"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    machine: Any  # MobileManipulator
    site: Any  # EmbodiedSiteScenario
    interference_tail: Quantity

    @property
    def renewal_period(self) -> Quantity:
        """Chunk renewal interval in ms, the reciprocal of the machine's chunk rate.

        Equation: ``(1 / machine.chunk_rate).to('ms')``."""
        return (1 / self.machine.chunk_rate).to("ms")

    @property
    def result(self):
        """Evaluate the stated chunk replacement case through ``ChunkReplacementModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.ChunkReplacementModel.solve``."""
        from ...solvers import ChunkReplacementModel

        s = self.site
        return ChunkReplacementModel().solve(
            s.perception_age,
            s.t_dispatch_to_admission,
            self.machine.chunk_inference_latency,
            s.t_plan_validate,
            self.renewal_period,
            s.t_lease,
            self.interference_tail,
        )


class JerkLimitedStopScenario(BaseScenario):
    """A machine at ``speed`` that stops along an S-curve at ``deceleration`` after ``delay``.

    The jerk bound is ``jerk`` when stated, else the peak jerk of the reference C2 polynomial stop at
    ``jerk_reference_speed`` (the entry speed by default). With a ``site``, its localization bound and
    protective margin join the required distance and its clear distance closes the budget unless
    ``clear_distance`` is stated.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.JerkLimitedStopModel.solve``.

    Assumptions: Inputs are routed to ``JerkLimitedStopModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to JerkLimitedStopModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseArmStallStop
    >>> value = case.result.required_distance
    >>> print(f"{value:.3g~}")
    220 mm"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    speed: Quantity
    deceleration: Quantity
    delay: Optional[Quantity] = None
    jerk: Optional[Quantity] = None
    jerk_reference_speed: Optional[Quantity] = None
    clear_distance: Optional[Quantity] = None
    site: Optional[Any] = None  # EmbodiedSiteScenario

    @property
    def d_clear(self) -> Optional[Quantity]:
        """Available clearance as a distance quantity: explicit override, else site clearance, else None.

        Equation: ``site.d_clear; otherwise clear_distance``."""
        if self.clear_distance is not None or self.site is None:
            return self.clear_distance
        return self.site.d_clear

    @property
    def result(self):
        """Evaluate the stated jerk limited stop case through ``JerkLimitedStopModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.JerkLimitedStopModel.solve``."""
        from ...solvers import JerkLimitedStopModel

        loc = None if self.site is None else self.site.delta_loc
        margin = None if self.site is None else self.site.delta_margin
        return JerkLimitedStopModel().solve(
            self.speed, self.deceleration, self.jerk, self.jerk_reference_speed, self.delay, self.d_clear, loc, margin
        )


class SpillLimitedStopScenario(BaseScenario):
    """The ``free_stop`` scenario's machine carrying an open container of ``inner_radius`` filled
    ``freeboard`` below the rim: the deceleration the liquid allows, that stop, and the carry speed.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.SpillLimitedStopModel.solve``.

    Assumptions: Inputs are routed to ``SpillLimitedStopModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to SpillLimitedStopModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseArmOpenMugStop
    >>> value = case.result.spill_deceleration
    >>> print(f"{value:.3g~}")
    2.45 m / s ** 2"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    free_stop: JerkLimitedStopScenario
    inner_radius: Quantity
    freeboard: Quantity

    @property
    def result(self):
        """Evaluate the stated spill limited stop case through ``SpillLimitedStopModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.SpillLimitedStopModel.solve``."""
        from ...solvers import SpillLimitedStopModel

        f = self.free_stop
        return SpillLimitedStopModel().solve(
            f.speed, f.deceleration, f.result.jerk, f.delay, f.d_clear, self.inner_radius, self.freeboard
        )


class TrajectoryDeadlinesScenario(BaseScenario):
    """A trajectory record for ``machine``'s arm at ``site``: a replacement requested at
    ``request_delay`` with a ``latency_target`` and ``schedule_slack``, a ``blend`` deadline, the
    ``commit`` instant when ``stop`` (a JerkLimitedStopScenario) must begin, and a ``late_wait``.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.TrajectoryDeadlinesModel.solve``.

    Assumptions: Inputs are routed to ``TrajectoryDeadlinesModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to TrajectoryDeadlinesModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseArmCommitmentSchedule
    >>> value = case.response
    >>> print(f"{value:.3g~}")
    22 ms"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    machine: Any  # MobileManipulator
    site: Any  # EmbodiedSiteScenario
    stop: JerkLimitedStopScenario
    request_delay: Quantity
    latency_target: Quantity
    schedule_slack: Quantity
    blend: Quantity
    commit: Quantity
    late_wait: Quantity

    @property
    def response(self) -> Quantity:
        """Decision-to-brake-onset delay in ms, summing tick, bus, and brake-onset delays.

        Decision to physical onset: one tick, one bus cycle, and the brake onset.

        Equation: ``(s.t_tick + s.t_bus + s.t_brake_onset).to('ms')``."""
        s = self.site
        return (s.t_tick + s.t_bus + s.t_brake_onset).to("ms")

    @property
    def result(self):
        """Evaluate the stated trajectory deadlines case through ``TrajectoryDeadlinesModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.TrajectoryDeadlinesModel.solve``."""
        from ...solvers import TrajectoryDeadlinesModel

        rm, st = self.machine, self.stop.result
        return TrajectoryDeadlinesModel().solve(
            rm.chunk_step,
            rm.chunk_horizon,
            (1 / rm.chunk_rate).to("ms"),
            self.request_delay,
            self.latency_target,
            self.schedule_slack,
            self.response,
            self.blend,
            self.commit,
            st.stop_time,
            st.scurve_travel,
            self.stop.speed,
            self.stop.d_clear,
            self.late_wait,
        )


class SeamBridgeScenario(BaseScenario):
    """A ``velocity_gap`` at a chunk seam on ``joint``, closed in one ``tick`` or by a quintic bridge
    over ``bridge_duration``, against ``machine``'s arm torque-rate limit. ``loaded_inertia`` (optional)
    prices the one-tick step with the payload's inertia added.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.SeamBridgeModel.solve``.

    Assumptions: Inputs are routed to ``SeamBridgeModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to SeamBridgeModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.PandaClassJointSeamBridge
    >>> value = case.result.torque_rate
    >>> print(f"{value:.3g~}")
    88.1 m * N / s"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    joint: Any  # an Actuators.PandaClassJoint entry
    machine: Any  # MobileManipulator
    velocity_gap: Quantity
    tick: Quantity
    bridge_duration: Quantity
    loaded_inertia: Optional[Quantity] = None

    @property
    def result(self):
        """Evaluate the stated seam bridge case through ``SeamBridgeModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.SeamBridgeModel.solve``."""
        from ...solvers import SeamBridgeModel

        j = self.joint
        return SeamBridgeModel().solve(
            j.gear_ratio,
            j.rotor_inertia,
            j.rated_torque,
            j.peak_torque,
            self.machine.arm.joint_torque_rate_limit,
            self.velocity_gap,
            self.tick,
            self.bridge_duration,
            self.loaded_inertia,
        )


class ShoulderSwingScenario(BaseScenario):
    """``machine``'s shoulder holding its rated payload at full planar reach, with ``joint``'s
    reflected rotor inertia, swinging ``swing`` rest to rest inside its datasheet limits.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.ShoulderSwingModel.solve``.

    Assumptions: Inputs are routed to ``ShoulderSwingModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to ShoulderSwingModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseArmShoulderSwing
    >>> value = case.result.jerk_limited_time
    >>> print(f"{value:.3g~}")
    529 ms"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    machine: Any  # MobileManipulator
    joint: Any  # an Actuators.PandaClassJoint entry
    swing: Quantity

    @property
    def reach(self) -> Quantity:
        """Full planar arm reach in m, adding upper-arm and forearm lengths.

        Equation: ``(arm.upper_arm_length + arm.forearm_length).to('m')``."""
        arm = self.machine.arm
        return (arm.upper_arm_length + arm.forearm_length).to("m")

    @property
    def result(self):
        """Evaluate the stated shoulder swing case through ``ShoulderSwingModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.ShoulderSwingModel.solve``."""
        from ...solvers import ShoulderSwingModel

        arm = self.machine.arm
        return ShoulderSwingModel().solve(
            arm.payload_capacity,
            self.reach,
            arm.max_torque,
            self.joint.reflected_inertia,
            self.swing,
            arm.joint_acceleration_limits[1],
            arm.joint_velocity_limits[1],
            arm.joint_torque_rate_limit,
        )


class DynamicWindowScenario(BaseScenario):
    """``machine``'s base over one chunk renewal at ``site``: its dynamic window and the classical
    admissible speed against the site's clear distance and credible deceleration.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.DynamicWindowModel.solve``.

    Assumptions: Inputs are routed to ``DynamicWindowModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to DynamicWindowModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseBaseDynamicWindow
    >>> value = case.result.half_width
    >>> print(f"{value:.3g~}")
    0.1 m / s"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    machine: Any  # MobileManipulator
    site: Any  # EmbodiedSiteScenario

    @property
    def result(self):
        """Evaluate the stated dynamic window case through ``DynamicWindowModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.DynamicWindowModel.solve``."""
        from ...solvers import DynamicWindowModel

        b = self.machine.base
        return DynamicWindowModel().solve(
            b.max_acceleration,
            (1 / self.machine.chunk_rate).to("ms"),
            self.site.d_clear,
            self.site.a_brake,
            b.max_velocity,
        )


class StaleAnchorScenario(BaseScenario):
    """A joint of ``machine``'s arm turning at ``joint_speed`` while a replacement takes ``delay``,
    anchored at the pose sampled at the request; the lever arm is the planar reach.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.StaleAnchorModel.solve``.

    Assumptions: Inputs are routed to ``StaleAnchorModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to StaleAnchorModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseArmStaleAnchor
    >>> value = case.result.gap
    >>> print(f"{value:.3g~}")
    0.04 rad"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    machine: Any  # MobileManipulator
    joint_speed: Quantity
    delay: Quantity

    @property
    def result(self):
        """Evaluate the stated stale anchor case through ``StaleAnchorModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.StaleAnchorModel.solve``."""
        from ...solvers import StaleAnchorModel

        arm = self.machine.arm
        return StaleAnchorModel().solve(self.joint_speed, self.delay, arm.upper_arm_length + arm.forearm_length)
