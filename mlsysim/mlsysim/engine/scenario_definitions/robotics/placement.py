"""Physical-machine scenarios for placement."""

from __future__ import annotations
from typing import Any, Dict, Optional
from ....core.types import Quantity
from ..base import BaseScenario


class SharedDramReadScenario(BaseScenario):
    """``site``'s permission tick placed on the machine's application SoC, its snapshot read
    queued behind backlogs on the shared DRAM controller.

    Bandwidth is the SoC memory's peak; the row-conflict time is ``memory_tech``'s
    t_RP + t_RCD. Addresses interleave across every channel unless
    ``one_channel`` confines the backlog and the read to one, which divides the
    bandwidth by the channel count (bus width over channel width).

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: The selected channel bandwidth and row-switch penalty price fixed competing backlogs.

    Excludes: A detailed memory-controller schedule, cache hits, and a measured WCET guarantee.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``SharedDramReadModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseSharedDieLoadedRead
    >>> value = case.result
    >>> print(f"{value.loaded_read.m_as('microsecond'):.3f} microsecond")
    1.899 microsecond"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario: WCET, deadline, blocking, interference
    memory_tech: Any  # MemoryTier: t_rp, t_rcd, channel_width
    sense: Quantity  # the tick's sense allowance (the snapshot read happens here)
    compute_after_sense: Quantity  # constraint check and drive-frame staging
    tile_bytes: int
    camera_bytes: int
    telemetry_bytes: int
    snapshot_bytes: int
    transaction_bytes: int
    bank_switches: int
    one_channel: bool = False

    @property
    def memory(self):
        """Application SoC memory specification used for the shared-read estimate.

        Equation:
            return self.site.machine.brain_soc.memory"""
        return self.site.machine.brain_soc.memory

    @property
    def channels(self) -> int:
        """Rounded channel count from memory bus width divided by channel width.

        Equation:
            return int(round((self.memory.bus_width / self.memory_tech.channel_width).to('').magnitude))"""
        return int(round((self.memory.bus_width / self.memory_tech.channel_width).to("").magnitude))

    @property
    def bank_switch_time(self) -> Quantity:
        """Row-switch penalty in ns, precharge plus activation time.

        Equation:
            return (self.memory_tech.t_rp + self.memory_tech.t_rcd).to('ns')"""
        return (self.memory_tech.t_rp + self.memory_tech.t_rcd).to("ns")

    @property
    def bandwidth(self) -> Quantity:
        """Memory bandwidth available to the modeled read, optionally restricted to one channel.

        Equation:
            bw = self.memory.bandwidth
            return bw / self.channels if self.one_channel else bw"""
        bw = self.memory.bandwidth
        return bw / self.channels if self.one_channel else bw

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.SharedDramReadModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = SharedDramReadModel().solve(...)`` using the fields shown below."""
        from ...solvers import SharedDramReadModel

        s = self.site
        return SharedDramReadModel().solve(
            tile_bytes=self.tile_bytes,
            camera_bytes=self.camera_bytes,
            telemetry_bytes=self.telemetry_bytes,
            snapshot_bytes=self.snapshot_bytes,
            transaction_bytes=self.transaction_bytes,
            bandwidth=self.bandwidth,
            bank_switches=self.bank_switches,
            bank_switch_time=self.bank_switch_time,
            sense=self.sense,
            compute_after_sense=self.compute_after_sense,
            wcet=s.enforcer_wcet,
            execution_ceiling=s.enforcer_deadline - s.enforcer_blocking - s.enforcer_interference,
        )


class StallShareScenario(BaseScenario):
    """``site``'s unloaded permission tick with a share of it slowed ``slowdown`` times.

    The tick may stretch to ``bound`` (default the declared WCET); the
    break-even share is ``(bound / unloaded - 1) / (slowdown - 1)``
    (``calc_stall_share_breakeven``).

    Notes
    -----
    Equation: ``breakeven_share`` is the worked quantity; each property below states its equation and units.

    Assumptions: Only one fraction of the unloaded tick incurs the supplied slowdown greater than one.

    Excludes: Interactions between stalls, deadline interference, and measuring the slow-path fraction.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``calc_stall_share_breakeven``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseTickMshrStallShare
    >>> value = case.breakeven_share
    >>> round(value, 3)
    0.043"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    slowdown: float
    bound: Optional[Quantity] = None

    @property
    def budget(self) -> Quantity:
        """Allowed tick duration in microseconds; defaults to declared WCET.

        Equation:
            return (self.site.enforcer_wcet if self.bound is None else self.bound).to('us')"""
        return (self.site.enforcer_wcet if self.bound is None else self.bound).to("us")

    @property
    def allowed_stretch(self) -> float:
        """Dimensionless allowed duration divided by unloaded execution time.

        Equation:
            return float((self.budget / self.site.enforcer_unloaded).to('').magnitude)"""
        return float((self.budget / self.site.enforcer_unloaded).to("").magnitude)

    @property
    def breakeven_share(self) -> float:
        """Break-even slow-path fraction from the shared partial-slowdown identity.

        Equation:
            return calc_stall_share_breakeven(self.allowed_stretch, self.slowdown)"""
        from ....physics import calc_stall_share_breakeven

        return calc_stall_share_breakeven(self.allowed_stretch, self.slowdown)

    @property
    def breakeven_stall(self) -> Quantity:
        """Microseconds of unloaded execution that can incur the specified slowdown.

        The break-even share as time of the unloaded tick.

        Equation:
            return (self.breakeven_share * self.site.enforcer_unloaded).to('us')"""
        return (self.breakeven_share * self.site.enforcer_unloaded).to("us")


class RailDroopScenario(BaseScenario):
    """A current step ``delta_i`` in ``rise_time`` through a package's inductance and resistance,
    and the alpha-power gate-delay stretch at the drooped rail.

    Notes
    -----
    Equation: ``droop`` is the worked quantity; each property below states its equation and units.

    Assumptions: Current slew is linear and lumped package resistance/inductance determine the voltage loss.

    Excludes: Regulator dynamics, decoupling-network resonance, and transistor-level timing simulation.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``calc_alpha_power_gate_delay_stretch``, ``calc_inductive_voltage_droop``, ``calc_voltage_droop``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.PackageRailDroop
    >>> value = case.droop
    >>> print(f"{value:.3f~P}")
    138.000 mV"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    rail_voltage: Quantity
    inductance: Quantity
    resistance: Quantity
    delta_i: Quantity
    rise_time: Quantity
    threshold_voltage: Quantity
    alpha: float

    @property
    def inductive_droop(self) -> Quantity:
        """Inductive voltage loss in mV from inductance times current slew.

        Equation:
            return calc_inductive_voltage_droop(self.inductance, self.delta_i / self.rise_time).to('mV')"""
        from ....physics.robotics import calc_inductive_voltage_droop

        return calc_inductive_voltage_droop(self.inductance, self.delta_i / self.rise_time).to("mV")

    @property
    def resistive_droop(self) -> Quantity:
        """Resistive voltage loss in mV from current step times resistance.

        Equation:
            return calc_voltage_droop(self.delta_i, self.resistance).to('mV')"""
        from ....physics.robotics import calc_voltage_droop

        return calc_voltage_droop(self.delta_i, self.resistance).to("mV")

    @property
    def droop(self) -> Quantity:
        """Combined inductive and resistive voltage loss in mV.

        Equation:
            return (self.inductive_droop + self.resistive_droop).to('mV')"""
        return (self.inductive_droop + self.resistive_droop).to("mV")

    @property
    def drooped_voltage(self) -> Quantity:
        """Supply voltage after the modeled drop, in V.

        Equation:
            return (self.rail_voltage - self.droop).to('V')"""
        return (self.rail_voltage - self.droop).to("V")

    @property
    def delay_expansion_percent(self) -> float:
        """Percent delay expansion under the shared alpha-power gate model.

        Equation:
            return calc_alpha_power_gate_delay_stretch(nominal_voltage=self.rail_voltage, drooped_voltage=self.drooped_voltage, threshold_voltage=self.threshold_voltage, alpha=self.alpha)['delay_expansion_percent']"""
        from ....physics.robotics import calc_alpha_power_gate_delay_stretch

        return calc_alpha_power_gate_delay_stretch(
            nominal_voltage=self.rail_voltage,
            drooped_voltage=self.drooped_voltage,
            threshold_voltage=self.threshold_voltage,
            alpha=self.alpha,
        )["delay_expansion_percent"]


class ThermalThrottleScenario(BaseScenario):
    """``site``'s permission tick on a cluster whose clock a thermal governor halves.

    The tick is sized to the declared WCET at ``nominal_clock``; the junction
    follows a first-order model from ``quiet_power`` to ``loaded_power`` through
    ``theta``; the clock throttles at ``trip`` and returns at ``clear``.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: The thermal solver uses a first-order temperature response and supplied clock/governor thresholds.

    Excludes: Measured governor scheduling, spatial temperature gradients, and universal real-time guarantees.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``ThermalThrottleModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseSharedDieThermalThrottle
    >>> value = case.result
    >>> print(f"{value.response_throttled.m_as('microsecond'):.3f} microsecond")
    590.000 microsecond"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario: WCET, deadline, B + I, tick
    nominal_clock: Quantity
    throttled_clock: Quantity
    time_constant: Quantity
    ambient: Quantity  # degC
    theta: Quantity  # junction-to-ambient, K/W
    quiet_power: Quantity
    loaded_power: Quantity
    trip: Quantity  # degC
    clear: Quantity  # degC

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.ThermalThrottleModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = ThermalThrottleModel().solve(...)`` using the fields shown below."""
        from ...solvers import ThermalThrottleModel

        s = self.site
        return ThermalThrottleModel().solve(
            wcet=s.enforcer_wcet,
            blocking_and_interference=s.enforcer_blocking + s.enforcer_interference,
            deadline=s.enforcer_deadline,
            tick=s.t_tick,
            nominal_clock=self.nominal_clock,
            throttled_clock=self.throttled_clock,
            time_constant=self.time_constant,
            ambient=self.ambient,
            theta=self.theta,
            quiet_power=self.quiet_power,
            loaded_power=self.loaded_power,
            trip=self.trip,
            clear=self.clear,
        )


class TickTraceScenario(BaseScenario):
    """Percentiles of a permission tick's execution time over a trace of ``ticks`` periods,
    cold and warm, against ``site``'s declared WCET and deadline.

    Notes
    -----
    Equation: ``duration`` is the worked quantity; each property below states its equation and units.

    Assumptions: Every counted trace tick has the configured period; percentile maps are supplied observations.

    Excludes: Estimating unseen tail probabilities or proving that observed maxima are WCET bounds.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseRejectedPlacementTrace
    >>> value = case.duration
    >>> print(f"{value:.3f~P}")
    10000.000 s"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    ticks: int
    cold: Dict[str, Quantity]  # percentile label -> execution time
    warm: Dict[str, Quantity]

    @property
    def duration(self) -> Quantity:
        """Trace duration in seconds, tick count times configured tick period.

        Equation:
            return (self.ticks * self.site.t_tick).to('s')"""
        return (self.ticks * self.site.t_tick).to("s")

    @property
    def hours_minutes_seconds(self):
        """Rounded total duration split into integer hours, minutes, and seconds.

        Equation:
            total = int(round(self.duration.to('s').magnitude))
            h, rem = divmod(total, 3600)
            m, sec = divmod(rem, 60)
            return (h, m, sec)"""
        total = int(round(self.duration.to("s").magnitude))
        h, rem = divmod(total, 3600)
        m, sec = divmod(rem, 60)
        return h, m, sec
