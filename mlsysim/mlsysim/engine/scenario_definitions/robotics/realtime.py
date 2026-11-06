"""Physical-machine scenarios for realtime."""

from __future__ import annotations
from typing import Any, Optional
from ....core.types import Quantity
from ..base import BaseScenario


class PhaseLagBudgetScenario(BaseScenario):
    """A loop with crossover ``crossover`` sampled every ``sample_period``, with compute and transport delays.

    Notes
    -----
    Equation: Result = mlsysim.solvers.PhaseLagBudgetModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by PhaseLagBudgetModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by PhaseLagBudgetModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.PhaseLagBudgetModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseJointLoopThroughTick
    >>> round((scenario.result.hold_delay).to('ms').magnitude, 6)
    0.5
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    crossover: Quantity
    sample_period: Quantity
    compute_delay: Optional[Quantity] = None
    transport_delay: Optional[Quantity] = None
    phase_allowance: Optional[Quantity] = None  # phase erosion the design allows at crossover

    @property
    def result(self):
        """Return ``mlsysim.solvers.PhaseLagBudgetModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.PhaseLagBudgetModel.solve (scenario fields).
        """
        from ...solvers import PhaseLagBudgetModel

        return PhaseLagBudgetModel().solve(
            self.crossover, self.sample_period, self.compute_delay, self.transport_delay, self.phase_allowance
        )


class ResponseTimeScenario(BaseScenario):
    """``site``'s permission check as a periodic task: R = C + B + I against its deadline.

    Each term defaults to the site record's; a stated value replaces it.

    Notes
    -----
    Equation: Result = mlsysim.solvers.ResponseTimeModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by ResponseTimeModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by ResponseTimeModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.ResponseTimeModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehousePermissionTickResponse
    >>> round((scenario.result.wcet).to('μs').magnitude, 6)
    250.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    wcet: Optional[Quantity] = None
    blocking: Optional[Quantity] = None
    interference: Optional[Quantity] = None
    deadline: Optional[Quantity] = None

    @property
    def result(self):
        """Return ``mlsysim.solvers.ResponseTimeModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.ResponseTimeModel.solve (scenario fields).
        """
        from ...solvers import ResponseTimeModel

        s = self.site
        return ResponseTimeModel().solve(
            s.enforcer_wcet if self.wcet is None else self.wcet,
            s.enforcer_blocking if self.blocking is None else self.blocking,
            s.enforcer_interference if self.interference is None else self.interference,
            s.enforcer_deadline if self.deadline is None else self.deadline,
            unloaded=s.enforcer_unloaded,
        )


class CanFdBusScenario(BaseScenario):
    """``num_axes`` servo axes on one CAN FD bus, ``frames_per_axis`` frames each per ``cycle``.

    Notes
    -----
    Equation: Result = mlsysim.solvers.CanFdBusModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by CanFdBusModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by CanFdBusModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.CanFdBusModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseServoCanFdOneBus
    >>> round((scenario.result.frame_time).to('μs').magnitude, 6)
    56.6
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    num_axes: int
    payload_bytes: int
    frames_per_axis: int
    baud_arbitration: Quantity
    baud_data: Quantity
    cycle: Quantity

    @property
    def result(self):
        """Return ``mlsysim.solvers.CanFdBusModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.CanFdBusModel.solve (scenario fields).
        """
        from ...solvers import CanFdBusModel

        return CanFdBusModel().solve(
            self.num_axes, self.payload_bytes, self.frames_per_axis, self.baud_arbitration, self.baud_data, self.cycle
        )


class EtherCatCycleScenario(BaseScenario):
    """``num_axes`` drives on one EtherCAT chain, ``bytes_per_axis`` of process data each per ``cycle``.

    Notes
    -----
    Equation: Result = mlsysim.solvers.EtherCatCycleModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by EtherCatCycleModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by EtherCatCycleModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.EtherCatCycleModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseServoEtherCat
    >>> round((scenario.result.frame_time).to('μs').magnitude, 6)
    16.64
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    num_axes: int
    bytes_per_axis: int
    baud_rate: Quantity
    header_bytes: int
    hop_delay: Quantity
    cycle: Quantity

    @property
    def result(self):
        """Return ``mlsysim.solvers.EtherCatCycleModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.EtherCatCycleModel.solve (scenario fields).
        """
        from ...solvers import EtherCatCycleModel

        return EtherCatCycleModel().solve(
            self.num_axes, self.bytes_per_axis, self.baud_rate, self.header_bytes, self.hop_delay, self.cycle
        )


class LeaseBudgetScenario(BaseScenario):
    """``site.machine``'s chunk lease at ``speed``: its stopping-budget cost and the lease rule's bounds.

    The lease defaults to the site's; the renewal period and the chunk
    inference latency are the machine's; ``renewal_jitter`` is the allowance
    for variation in the interval between successful renewals.

    Notes
    -----
    Equation: Result = mlsysim.solvers.LeaseBudgetModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by LeaseBudgetModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by LeaseBudgetModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.LeaseBudgetModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseChunkLease
    >>> round((scenario.result.lease).to('ms').magnitude, 6)
    60.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    speed: Quantity
    renewal_jitter: Quantity
    lease: Optional[Quantity] = None

    @property
    def result(self):
        """Return ``mlsysim.solvers.LeaseBudgetModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.LeaseBudgetModel.solve (scenario fields).
        """
        from ...solvers import LeaseBudgetModel

        s, rm = self.site, self.site.machine
        return LeaseBudgetModel().solve(
            self.speed,
            s.a_brake,
            s.d_clear,
            s.delta_loc,
            s.delta_margin,
            s.t_brake_onset,
            s.t_lease if self.lease is None else self.lease,
            s.t_tick,
            s.t_bus,
            s.perception_age,
            s.t_capture_to_admission,
            rm.chunk_period,
            self.renewal_jitter,
            rm.chunk_inference_latency,
        )
