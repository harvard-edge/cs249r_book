"""Physical-machine scenarios for diagnostics."""

from __future__ import annotations
from typing import Any, Dict, Optional
from ....core.types import Quantity
from ..base import BaseScenario
from .stopping import StoppingBudgetScenario


class OnsetTrialScenario(BaseScenario):
    """A stopping trace that begins braking at ``onset`` after the reference event of ``oracle``.

    ``oracle`` is a StoppingBudgetScenario whose delay is the onset bound (no
    site allowances), with a jerk bound and a tracking inset: its required
    distance is the travel bound, and its delay travel plus S-curve stop is
    the travel of an on-time trace.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.OnsetTrialModel.solve``.

    Assumptions: Inputs are routed to ``OnsetTrialModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to OnsetTrialModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseRenewalStopLateOnsetTrial
    >>> value = case.result.distance_margin
    >>> print(f"{value:.3g~}")
    1 mm"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    oracle: StoppingBudgetScenario
    onset: Quantity
    lease: Optional[Quantity] = None  # the lease the trial lets expire
    probe_after_expiry: Optional[Quantity] = None  # a refused renewal replayed this long after expiry

    @property
    def probe_at(self) -> Optional[Quantity]:
        """Replay arrival time in ms after lease expiry; None if lease or replay offset is unspecified.

        When the replayed renewal arrives, after the last valid renewal.

        Equation: ``(lease + probe_after_expiry).to('ms'); otherwise None``."""
        if self.lease is None or self.probe_after_expiry is None:
            return None
        return (self.lease + self.probe_after_expiry).to("ms")

    @property
    def result(self):
        """Evaluate the stated onset trial case through ``OnsetTrialModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.OnsetTrialModel.solve``."""
        from ...solvers import OnsetTrialModel

        o = self.oracle.result
        return OnsetTrialModel().solve(
            self.oracle.delay,
            self.onset,
            self.oracle.speed,
            (o.braking_travel + o.jerk_ramp_travel).to("mm"),
            o.required_distance,
            o.stop_time,
        )


class RailHoldUpScenario(BaseScenario):
    """``site.machine``'s backed permission rail, its feed lost during the longest admitted stop.

    The longest stop starts at the speed ceiling of ``stop`` (a finished
    StoppingBudgetScenario); the flag path is one tick, one bus cycle, and the
    brake onset of ``site``; engagement and hold-up are the machine's.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.RailHoldUpModel.solve``.

    Assumptions: Inputs are routed to ``RailHoldUpModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to RailHoldUpModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehousePermissionRailHoldUp
    >>> value = case.result.flag_to_set
    >>> print(f"{value:.3g~}")
    1.31e+03 ms"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    stop: StoppingBudgetScenario

    @property
    def flag_path(self) -> Quantity:
        """Fault-to-brake-onset path in ms, tick plus bus plus brake onset.

        Equation: ``(s.t_tick + s.t_bus + s.t_brake_onset).to('ms')``."""
        s = self.site
        return (s.t_tick + s.t_bus + s.t_brake_onset).to("ms")

    @property
    def result(self):
        """Evaluate the stated rail hold up case through ``RailHoldUpModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.RailHoldUpModel.solve``."""
        from ...solvers import RailHoldUpModel

        rm = self.site.machine
        return RailHoldUpModel().solve(
            self.flag_path,
            self.stop.result.speed_ceiling,
            self.stop.a_brake,
            self.stop.jerk,
            rm.spring_brake_engage_max,
            rm.permission_rail_holdup,
        )


class ExponentialRareEventScenario(BaseScenario):
    """An exponential delay of ``mean`` that fails past the delay spare of ``budget`` (or ``threshold``).

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.ExponentialRareEventModel.solve``.

    Assumptions: Inputs are routed to ``ExponentialRareEventModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to ExponentialRareEventModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseQueueDelaySpareExhaustion
    >>> value = case.result.failure_probability
    >>> print(f"{value:.3g}")
    1.77e-09"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    mean: Quantity
    elite_share: float
    relative_error: float
    confidence: float
    budget: Optional[StoppingBudgetScenario] = None  # threshold = its delay_spare
    threshold: Optional[Quantity] = None

    @property
    def failure_threshold(self) -> Quantity:
        """Explicit delay threshold, else modeled delay spare from the budget in ms.

        Equation: ``budget.result.delay_spare.to('ms'); otherwise threshold``."""
        if self.threshold is not None:
            return self.threshold
        return self.budget.result.delay_spare.to("ms")

    @property
    def result(self):
        """Evaluate the stated exponential rare event case through ``ExponentialRareEventModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.ExponentialRareEventModel.solve``."""
        from ...solvers import ExponentialRareEventModel

        return ExponentialRareEventModel().solve(
            self.failure_threshold, self.mean, self.elite_share, self.relative_error, self.confidence
        )


class DiagnosticCoverageScenario(BaseScenario):
    """Fault-rate ``strata`` (FIT) to be left at most ``target`` FIT unmitigated, shown at ``confidence``.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.DiagnosticCoverageModel.solve``.

    Assumptions: Inputs are routed to ``DiagnosticCoverageModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to DiagnosticCoverageModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseDiagnosticCoverageHundredFit
    >>> value = case.result.coverage
    >>> print(f"{value:.3g}")
    0.99"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    strata: Dict[str, float]
    target: float
    confidence: float

    @property
    def result(self):
        """Evaluate the stated diagnostic coverage case through ``DiagnosticCoverageModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.DiagnosticCoverageModel.solve``."""
        from ...solvers import DiagnosticCoverageModel

        return DiagnosticCoverageModel().solve(self.strata, self.target, self.confidence)
