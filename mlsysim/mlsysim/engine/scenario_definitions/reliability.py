"""Failure exposure, recovery, and operational safeguards.

Public compatibility imports remain in ``mlsysim.engine.scenario_types``.
"""

from __future__ import annotations
import math as _math
from typing import Any, Dict, List, Optional
from ...core.types import Quantity
from ...core.units import Q_
from ...hardware.types import HardwareNode
from ...models.types import Workload
from ...systems.types import Fleet
from .base import BaseScenario


class FailureIntervalScenario(BaseScenario):
    """``n_components`` identical components, each with ``component_mtbf``, over a run.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.FailureIntervalModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.FailureIntervalModel``;
    its documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.FailureIntervalModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.FailureIntervalModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.GpuFailuresTraining10K
    >>> type(case.result).__name__
    'FailureIntervalResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    component_mtbf: Quantity
    n_components: int
    run_duration: Quantity

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.FailureIntervalModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import FailureIntervalModel

        return FailureIntervalModel().solve(self.component_mtbf, self.n_components, self.run_duration)


class FailureIntervalSweepScenario(BaseScenario):
    """Interval between failures of ``component_mtbf`` components, at each job size in
    ``sizes``.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.reliability.calc_mtbf_cluster``.

    Assumptions: Components have identical independent stationary failure hazards; their
    hazards add with fleet size.

    Excludes: Correlated failures, repairs, and time-varying failure rates are absent.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.reliability.calc_mtbf_cluster``. The methods below identify
    any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.GpuFailureLadder
    >>> str(next(iter(case.intervals.values())))
    '6250.0 h'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    component_mtbf: Quantity
    sizes: List[int]

    @property
    def intervals(self) -> Dict[int, Quantity]:
        """Mean failure intervals indexed by component count. Equation:
        ``mlsysim.physics.reliability.calc_mtbf_cluster(component_mtbf, n)`` for each size.
        Units: time; independent identical failure hazards are assumed.
        """
        from ...physics.reliability import calc_mtbf_cluster

        return {n: calc_mtbf_cluster(self.component_mtbf, n) for n in self.sizes}

    @property
    def failures_per_day(self) -> Dict[int, float]:
        """Expected failure counts per day at each job size. Equation: one day divided by each mean
        interval. Units: dimensionless counts per stated day, not event probabilities.
        """
        return {n: (Q_(1, "day") / t).to("").magnitude for n, t in self.intervals.items()}


class SeriesFailureScenario(BaseScenario):
    """A synchronous job on ``n_devices`` that stops when any one component fails.

    ``component_mttf`` gives each component's MTTF and ``per_node`` its count in
    one node of ``devices_per_node`` devices; a redundant part that cannot fail
    the node singly is counted 0. Added 2026-10-20.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.SeriesFailureModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.SeriesFailureModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.SeriesFailureModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.SeriesFailureModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.FullSystemFailures10K
    >>> type(case.result).__name__
    'SeriesFailureResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    n_devices: int
    devices_per_node: int
    component_mttf: Dict[str, Quantity]
    per_node: Dict[str, int]

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.SeriesFailureModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import SeriesFailureModel

        return SeriesFailureModel().solve(self.n_devices, self.devices_per_node, self.component_mttf, self.per_node)


class CheckpointIntervalScenario(BaseScenario):
    """Checkpoints of ``workload`` written at ``write_bandwidth``, against failures every
    ``mtbf``.

    ``write_time`` states the write directly when no workload is given.
    Added 2026-10-20.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.quantities.memory_from_params``,
    ``mlsysim.solvers.CheckpointIntervalModel``,
    ``mlsysim.physics.reliability.calc_checkpoint_waste``.

    Assumptions: Checkpoint writes and the supplied mean failure interval follow the
    periodic-checkpoint approximation in CheckpointIntervalModel; either a write duration or
    storage geometry must be provided.

    Excludes: Checkpoint contention, changing write rates, and recovery costs outside that
    solver are absent.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.quantities.memory_from_params``,
    ``mlsysim.solvers.CheckpointIntervalModel``,
    ``mlsysim.physics.reliability.calc_checkpoint_waste``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt3CheckpointInterval10K
    >>> str(case.checkpoint_bytes)
    '2450000000000.0 B'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    mtbf: Quantity
    workload: Optional[Workload] = None
    bytes_per_param: Optional[Quantity] = None
    write_bandwidth: Optional[Quantity] = None
    write_time: Optional[Quantity] = None
    # Optional (added 2026-10-20): factors by which a chosen interval misses the
    # optimum, priced by ``miss_penalty``.
    miss_factors: Optional[List[float]] = None

    @property
    def checkpoint_bytes(self) -> Quantity:
        """Persistent checkpoint storage derived from parameter count. Equation:
        ``mlsysim.physics.quantities.memory_from_params(parameters, bytes_per_param)``. Units:
        bytes; requires the workload and per-parameter storage convention.
        """
        from ...physics.quantities import memory_from_params

        return memory_from_params(self.workload.parameters, self.bytes_per_param).to("byte")

    @property
    def t_write(self) -> Quantity:
        """Checkpoint duration, using an explicit duration when supplied. Equation: otherwise
        ``checkpoint_bytes / write_bandwidth``. Units: seconds; the bandwidth branch omits
        startup and contention.
        """
        if self.write_time is not None:
            return self.write_time.to("second")
        return (self.checkpoint_bytes / self.write_bandwidth).to("second")

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.CheckpointIntervalModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ..solvers import CheckpointIntervalModel

        return CheckpointIntervalModel().solve(self.t_write, self.mtbf)

    def waste_at(self, interval: Quantity) -> float:
        """Expected time fraction lost at a proposed checkpoint interval. Equation:
        ``mlsysim.physics.reliability.calc_checkpoint_waste(t_write, interval, mtbf)``. Units:
        dimensionless; inherits its periodic-checkpoint approximation.
        """
        from ...physics.reliability import calc_checkpoint_waste

        return calc_checkpoint_waste(self.t_write, interval, self.mtbf)

    def miss_penalty(self, factor: float) -> float:
        """Relative excess waste when the interval differs from the optimum. Equation:
        ``waste_at(factor * result.interval) / result.waste - 1``. Units: dimensionless; the
        positive factor rescales the modeled optimum.

        Extra waste, relative to the optimum, of an interval ``factor`` times the optimum
        (added 2026-10-20); equals ``(x + 1/x) / 2 - 1``, so the minimum is flat.
        """
        r = self.result
        return self.waste_at(factor * r.interval) / r.waste - 1


class RecoveryBudgetScenario(BaseScenario):
    """Cold restarts of ``workload`` after failures of ``n_devices`` components of
    ``component_mtbf``.

    ``recovery`` is the four-term budget profile (Systems.Reliability.Recovery).
    Added 2026-10-20.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.quantities.memory_from_params``,
    ``mlsysim.physics.reliability.calc_mtbf_cluster``,
    ``mlsysim.solvers.RecoveryBudgetModel``.

    Assumptions: State size uses the specified bytes-per-parameter convention and
    independent component hazards determine failure cadence.

    Excludes: Unlisted restart stages, correlated faults, and runtime measurement are
    absent.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.quantities.memory_from_params``,
    ``mlsysim.physics.reliability.calc_mtbf_cluster``,
    ``mlsysim.solvers.RecoveryBudgetModel``. The methods below identify any additional
    arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt3RecoveryBudget1K
    >>> str(case.checkpoint_bytes)
    '2450000000000.0 B'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    bytes_per_param: Quantity
    recovery: Any
    component_mtbf: Quantity
    n_devices: int

    @property
    def checkpoint_bytes(self) -> Quantity:
        """State bytes that must be restored after failure. Equation:
        ``mlsysim.physics.quantities.memory_from_params(parameters, bytes_per_param)``. Units:
        bytes; the specified state convention determines what is included.
        """
        from ...physics.quantities import memory_from_params

        return memory_from_params(self.workload.parameters, self.bytes_per_param).to("byte")

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.RecoveryBudgetModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ...physics.reliability import calc_mtbf_cluster
        from ..solvers import RecoveryBudgetModel

        return RecoveryBudgetModel().solve(
            self.checkpoint_bytes, self.recovery, calc_mtbf_cluster(self.component_mtbf, self.n_devices), self.n_devices
        )


class HotSpareScenario(BaseScenario):
    """``spare_devices`` held idle beside the job in ``failures``, priced at
    ``device_hour_rate``.

    The spare pays off when it shortens every recovery by more than
    ``breakeven``. Added 2026-10-20.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: Daily spare rental is compared with saved billed device-time on every
    modeled failure; all affected devices benefit from the same recovery reduction.

    Excludes: Placement constraints, failure overlap, and service-value pricing beyond
    rental cost are absent.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.HotSpareNode10K
    >>> print(f"{case.spare_cost_per_day.m_as('USD'):.1f} USD")
    384.0 USD
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    failures: Any  # SeriesFailureScenario
    spare_devices: int
    device_hour_rate: Quantity

    @property
    def spare_cost_per_day(self) -> Quantity:
        """Rental cost of keeping every spare idle for a day. Equation: ``spare_devices *
        device_hour_rate * one_day``. Units: USD per stated day; no opportunity-value adjustment
        is included.
        """
        return (self.spare_devices * self.device_hour_rate * Q_(1, "day")).to("USD")

    @property
    def value_per_minute(self) -> Quantity:
        """Daily rental value of removing one minute from each recovery. Equation: expected daily
        failures times affected devices times hourly price times one minute. Units: USD per
        stated day.
        """
        r = self.failures.result
        return (r.failures_per_day * self.failures.n_devices * self.device_hour_rate * Q_(1, "minute")).to("USD")

    @property
    def breakeven(self) -> Quantity:
        """Recovery time that each spare intervention must save to cover daily spare cost.
        Equation: ``spare_cost_per_day / value_per_minute * one_minute``. Units: seconds; the
        modeled avoided loss must be positive.
        """
        return (self.spare_cost_per_day / self.value_per_minute * Q_(1, "minute")).to("second")


class StragglerCostScenario(BaseScenario):
    """One of ``n_devices`` in a synchronous job runs at ``slow_speed`` of its normal rate.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: One slower worker stalls every synchronous step; worker counts, task work,
    and hourly prices remain fixed.

    Excludes: Asynchronous execution and selective worker replacement are absent.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.HalfSpeedStraggler1K
    >>> str(case.slow_step)
    '2.0 s'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    n_devices: int
    step_time: Quantity
    slow_speed: float
    device_hour_rate: Quantity

    @property
    def slow_step(self) -> Quantity:
        """Synchronous step duration imposed by the slower worker. Equation: ``step_time /
        slow_speed``. Units: seconds; all workers wait for this one straggler.
        """
        return (self.step_time / self.slow_speed).to("second")

    @property
    def throughput(self) -> float:
        """Throughput retained relative to the healthy step. Equation: ``step_time / slow_step``.
        Units: dimensionless; work per step is unchanged.
        """
        return (self.step_time / self.slow_step).to("").magnitude

    @property
    def idle_cost(self) -> Quantity:
        """One-hour fleet rental attributed to the lost throughput fraction. Equation: ``n_devices
        * device_hour_rate * (1 - throughput) * one_hour``. Units: USD; this is an allocated
        cost, not an extra bill.
        """
        return (self.n_devices * self.device_hour_rate * (1 - self.throughput) * Q_(1, "hour")).to("USD")


class HealthProbeScenario(BaseScenario):
    """A readiness probe every ``interval``, failing after ``timeout``, removing after
    ``failure_threshold`` misses.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.reliability.calc_probe_detection_window``.

    Assumptions: A fixed-interval probe requires the stated number of misses and uses a
    fixed timeout.

    Excludes: Probe jitter, recovery oscillation, and correlated probe failures are absent.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.reliability.calc_probe_detection_window``. The methods
    below identify any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.InferenceReadinessProbe
    >>> type(case.detection_window).__name__
    'tuple'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    interval: Quantity
    timeout: Quantity
    failure_threshold: int
    success_threshold: int

    @property
    def detection_window(self):
        """Time bound for a readiness probe to remove an unresponsive instance. Equation:
        ``mlsysim.physics.reliability.calc_probe_detection_window(interval, timeout,
        failure_threshold)``. Units: time; scheduling follows that helper's probe assumptions.
        """
        from ...physics.reliability import calc_probe_detection_window

        return calc_probe_detection_window(self.interval, self.timeout, self.failure_threshold)


class SdcExposureScenario(BaseScenario):
    """Silent-corruption exposure of ``n_devices``, each at ``event_rate`` per device-hour, per
    ``step_time`` step.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.reliability.calc_step_event_probability``.

    Assumptions: Independent event-bearing steps repeat under a stationary per-device event-
    rate model.

    Excludes: Multiple corruptions inside one step are not separately counted by the step-
    probability conversion.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.reliability.calc_step_event_probability``. The methods
    below identify any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.SdcExposure10K
    >>> round(case.exposure, 4)
    5.5556
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    n_devices: int
    event_rate: float
    step_time: Quantity
    run_duration: Optional[Quantity] = None

    @property
    def exposure(self) -> float:
        """Aggregate device exposure in one step. Equation: device count times step duration in
        hours. Units: device-hours represented as a scalar.
        """
        return self.n_devices * self.step_time.to("hour").magnitude

    @property
    def p_step(self) -> float:
        """Probability of at least one corruption event during a step. Equation:
        ``mlsysim.physics.reliability.calc_step_event_probability(event_rate, n_devices,
        step_time)``. Units: probability under the stated event-rate model.
        """
        from ...physics.reliability import calc_step_event_probability

        return calc_step_event_probability(self.event_rate, self.n_devices, self.step_time)

    @property
    def steps_between(self) -> float:
        """Geometric mean number of steps until an event-bearing step. Equation: ``1 / p_step``.
        Units: steps; requires positive probability and independent repeated step trials.
        """
        return 1 / self.p_step

    @property
    def interval(self) -> Quantity:
        """Mean duration until an event-bearing step. Equation: ``steps_between * step_time``.
        Units: hours; this uses the step-level geometric model.
        """
        return (self.steps_between * self.step_time).to("hour")

    @property
    def run_events(self) -> float:
        """Expected event-bearing steps over the run. Equation: ``run_duration / interval``. Units:
        dimensionless count; multiple events in one step are counted together.
        """
        return (self.run_duration / self.interval).to("").magnitude


class AvailabilityScenario(BaseScenario):
    """A job that needs all ``n_components``, each up a fraction ``availability`` of the time.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.reliability.calc_all_up_availability``,
    ``mlsysim.physics.reliability.calc_downtime``,
    ``mlsysim.physics.reliability.calc_availability_stacked``.

    Assumptions: Components or replicas have identical stationary independent uptime
    probabilities; the job needs every component while the replicated service needs any
    replica.

    Excludes: Shared failure domains and failover delay are absent.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.reliability.calc_all_up_availability``,
    ``mlsysim.physics.reliability.calc_downtime``,
    ``mlsysim.physics.reliability.calc_availability_stacked``. The methods below identify
    any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.NodeAvailability1K
    >>> round(case.all_up, 4)
    0.3677
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    availability: float
    n_components: int

    @property
    def all_up(self) -> float:
        """Probability that every required component is simultaneously up. Equation:
        ``mlsysim.physics.reliability.calc_all_up_availability(availability, n_components)``.
        Units: probability; independent component states are assumed.
        """
        from ...physics.reliability import calc_all_up_availability

        return calc_all_up_availability(self.availability, self.n_components)

    def required_availability(self, all_up_target: float) -> float:
        """Per-component uptime needed for a target all-up probability. Equation: ``all_up_target
        ** (1 / n_components)``. Units: probability; assumes identical independent components.

        Per-component availability that keeps ``all_up`` at ``all_up_target`` (added
        2026-10-20).
        """
        return all_up_target ** (1 / self.n_components)

    def downtime(self, interval: Quantity) -> Quantity:
        """Expected single-component downtime over the supplied duration. Equation:
        ``mlsysim.physics.reliability.calc_downtime(availability, interval)``. Units: time;
        availability is treated as stationary.
        """
        from ...physics.reliability import calc_downtime

        return calc_downtime(self.availability, interval)

    @property
    def downtime_per_year(self) -> Quantity:
        """Expected single-component downtime over the package's year convention. Equation:
        ``downtime(DAYS_PER_YEAR days)``. Units: minutes; this does not describe all-component
        job uptime.
        """
        from ...core.units import DAYS_PER_YEAR, Q_

        return self.downtime(Q_(DAYS_PER_YEAR, "day")).to("minute")

    # Parallel composition (added 2026-10-20): a service up while any of
    # ``replicas`` independent copies is up.
    replicas: int = 1

    @property
    def any_up(self) -> float:
        """Availability of independent replicas when one surviving replica is sufficient. Equation:
        ``mlsysim.physics.reliability.calc_availability_stacked(availability, replicas)``.
        Units: probability; correlated failures are excluded.
        """
        from ...physics.reliability import calc_availability_stacked

        return calc_availability_stacked(self.availability, self.replicas)

    @property
    def replicated_downtime_per_year(self) -> Quantity:
        """Expected outage of the any-replica-up service over the modeled year. Equation:
        ``calc_downtime(any_up, DAYS_PER_YEAR days)``. Units: minutes; assumes successful
        failover and independent replica states.
        """
        from ...physics.reliability import calc_downtime
        from ...core.units import DAYS_PER_YEAR, Q_

        return calc_downtime(self.any_up, Q_(DAYS_PER_YEAR, "day")).to("minute")


class QueryExtractionBudgetScenario(BaseScenario):
    """An extractor that needs ``extraction_queries`` answers from a priced API, under three
    billing regimes: pay per query, a paid tier capped per day, and a lifetime per-principal
    allowance after which every query is billed at the per-query price.

    Notes
    -----
    Equation: Per-query cost multiplies answer count; tier cost multiplies quota-limited
    collection days by daily fee.

    Assumptions: Answers fill the daily quota continuously and prorated tier charges are
    permitted.

    Excludes: Calendar rounding, failed queries, and a prediction that the supplied answer
    budget reaches target fidelity.

    References
    ----------
    Derivation: Per-query cost multiplies answer count; tier cost multiplies quota-limited
    collection days by daily fee. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ResNet50ApiExtractionBudget
    >>> round(float(scenario.collection_days), 4)
    500.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    training_cost: Quantity  # USD to train the victim model
    extraction_queries: float  # answers for the target fidelity
    target_fidelity: float  # agreement with the victim the extractor aims for
    price_per_query: Quantity  # USD, pay-as-you-go
    tier_price: Quantity  # USD per billing period
    tier_daily_quota: float  # queries per day on the paid tier
    billing_period: Quantity  # days per tier billing period
    lifetime_allowance: float  # queries per principal before overage billing

    def _tier_fee_per_day(self) -> Quantity:
        return (self.tier_price / self.billing_period.to("day").magnitude).to("USD")

    def tier_days(self, queries) -> float:
        """Collection days for ``queries`` answers when the daily tier quota is fully used;
        fractional days are allowed.

        Equation: ``queries / tier_daily_quota``."""
        return queries / self.tier_daily_quota

    def pay_per_query_cost(self, queries) -> Quantity:
        """USD billed for ``queries`` answers at the supplied per-query price.

        Equation: ``(queries * price_per_query).to('USD')``."""
        return (queries * self.price_per_query).to("USD")

    def tier_cost(self, queries) -> Quantity:
        """USD in prorated tier fees for the days needed to collect ``queries`` answers.

        Daily quota only: the tier fee for every day of collection.

        Equation: ``(tier_days(queries) * _tier_fee_per_day()).to('USD')``."""
        return (self.tier_days(queries) * self._tier_fee_per_day()).to("USD")

    def lifetime_cost(self, queries) -> Quantity:
        """USD for tier collection up to the lifetime allowance plus per-query charges above it.

        Tier fees up to the allowance, then every query billed per query.

        Equation: ``within = min(queries, lifetime_allowance); overage = max(queries -
        lifetime_allowance, 0); result = (tier_cost(within) +
        pay_per_query_cost(overage)).to('USD')``."""
        within = min(queries, self.lifetime_allowance)
        overage = max(queries - self.lifetime_allowance, 0)
        return (self.tier_cost(within) + self.pay_per_query_cost(overage)).to("USD")

    @property
    def collection_days(self) -> float:
        """Days required to collect the full target answer set at the daily tier quota.

        Equation: ``tier_days(extraction_queries)``."""
        return self.tier_days(self.extraction_queries)

    @property
    def full_set_pay_per_query(self) -> Quantity:
        """USD for all target answers under pure per-query billing.

        Equation: ``pay_per_query_cost(extraction_queries)``."""
        return self.pay_per_query_cost(self.extraction_queries)

    @property
    def full_set_tier(self) -> Quantity:
        """USD for all target answers under the daily-quota tier model.

        Equation: ``tier_cost(extraction_queries)``."""
        return self.tier_cost(self.extraction_queries)

    @property
    def allowance_fees(self) -> Quantity:
        """USD in tier fees needed to exhaust the lifetime allowance.

        Equation: ``tier_cost(lifetime_allowance)``."""
        return self.tier_cost(self.lifetime_allowance)

    @property
    def overage_cost(self) -> Quantity:
        """USD billed per query for the nonnegative excess above the lifetime allowance.

        Equation: ``pay_per_query_cost(max(extraction_queries - lifetime_allowance, 0))``."""
        return self.pay_per_query_cost(max(self.extraction_queries - self.lifetime_allowance, 0))

    @property
    def full_set_lifetime(self) -> Quantity:
        """USD for the full target answer set under the lifetime-allowance policy.

        Equation: ``lifetime_cost(extraction_queries)``."""
        return self.lifetime_cost(self.extraction_queries)

    @property
    def tier_share_of_training(self) -> float:
        """Dimensionless tier-extraction bill divided by the victim model's stated training cost.

        Equation: ``(full_set_tier / training_cost).to('').magnitude``."""
        return (self.full_set_tier / self.training_cost).to("").magnitude


class OutputShapingScenario(BaseScenario):
    """An API that returns the top ``top_k`` of ``class_count`` scores, rounded from
    ``from_decimals`` to ``to_decimals`` decimal places.

    Notes
    -----
    Equation: Entry fractions count retained scores; log-two level counts express decimal
    representation sizes in bits.

    Assumptions: Positive top-k count and decimal rounding of scores in [0,1].

    Excludes: An attack-success model or information-theoretic privacy guarantee.

    References
    ----------
    Derivation: Entry fractions count retained scores; log-two level counts express decimal
    representation sizes in bits. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ImageNetTop5RoundedScores
    >>> round(float(scenario.scores_reduction), 4)
    200.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    class_count: int
    top_k: int
    from_decimals: int
    to_decimals: int

    @property
    def scores_reduction(self) -> float:
        """Dimensionless full-score count divided by returned top-k score count.

        Times fewer scores per answer than returning every class.

        Equation: ``class_count / top_k``."""
        return self.class_count / self.top_k

    @property
    def entries_removed(self) -> float:
        """Fraction of score entries suppressed by returning only the top-k entries.

        Equation: ``1 - top_k / class_count``."""
        return 1 - self.top_k / self.class_count

    @property
    def decimal_digits_removed(self) -> int:
        """Number of decimal places removed from each returned score.

        Equation: ``from_decimals - to_decimals``."""
        return self.from_decimals - self.to_decimals

    @property
    def bits_removed(self) -> float:
        """Log-two count of the removed decimal bins, in bits; this is a representation count, not
        a privacy guarantee.

        Bits per score removed under a decimal-bin count.

        Equation: ``calc_level_bits(10 ** decimal_digits_removed)``."""
        from ...physics.privacy import calc_level_bits

        return calc_level_bits(10**self.decimal_digits_removed)

    @property
    def retained_bits(self) -> float:
        """Log-two count of the retained score levels including both endpoints of [0,1].

        Bits left in a score in [0, 1] at ``to_decimals`` places (10^d + 1 levels).

        Equation: ``calc_level_bits(10 ** to_decimals + 1)``."""
        from ...physics.privacy import calc_level_bits

        return calc_level_bits(10**self.to_decimals + 1)


class DataParallelAttackSurfaceScenario(BaseScenario):
    """``workload`` trained on ``fleet`` with ``dp_degree`` data-parallel replicas: the
    gradient
    each step exchanges, the checkpoint it writes, and the traffic of a robust aggregate
    that
    gathers every replica's gradient instead of a ring AllReduce of their sum.

    Notes
    -----
    Equation: Parameter count times element width gives each payload; gather and ring
    factors compare replicated traffic.

    Assumptions: Equal data-parallel replicas and the supplied parameter widths; ratio
    comparisons require multiple replicas.

    Excludes: Cryptographic protocol work, optimizer state beyond the declared checkpoint
    width, and communication overlap.

    References
    ----------
    Derivation: Parameter count times element width gives each payload; gather and ring
    factors compare replicated traffic. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ReferenceJobAttackSurface
    >>> round(float(scenario.accelerators), 4)
    1024.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    fleet: Fleet
    dp_degree: int
    gradient_bytes_per_param: Quantity
    checkpoint_bytes_per_param: Quantity
    weight_bytes_per_param: Optional[Quantity] = None  # serving weights, when quoted

    @property
    def accelerators(self) -> int:
        """Accelerator count copied from the composed fleet definition.

        Equation: ``fleet.total_accelerators``."""
        return self.fleet.total_accelerators

    @property
    def nodes(self) -> int:
        """Node count copied from the composed fleet definition.

        Equation: ``fleet.count``."""
        return self.fleet.count

    @property
    def accelerators_per_node(self) -> int:
        """Accelerator count per node copied from the fleet's node definition.

        Equation: ``fleet.node.accelerators_per_node``."""
        return self.fleet.node.accelerators_per_node

    @property
    def accelerators_per_replica(self) -> float:
        """Mean accelerator count allocated to each data-parallel replica.

        Equation: ``accelerators / dp_degree``."""
        return self.accelerators / self.dp_degree

    @property
    def gradient_bytes(self) -> Quantity:
        """Bytes in one replica's parameter gradients, using the declared gradient width.

        Equation: ``memory_from_params(workload.parameters,
        gradient_bytes_per_param).to('byte')``."""
        from ...physics.quantities import memory_from_params

        return memory_from_params(self.workload.parameters, self.gradient_bytes_per_param).to("byte")

    @property
    def checkpoint_bytes(self) -> Quantity:
        """Bytes in one parameter checkpoint, using the declared checkpoint width.

        Equation: ``memory_from_params(workload.parameters,
        checkpoint_bytes_per_param).to('byte')``."""
        from ...physics.quantities import memory_from_params

        return memory_from_params(self.workload.parameters, self.checkpoint_bytes_per_param).to("byte")

    @property
    def weight_bytes(self) -> Quantity:
        """Bytes in the optional serving checkpoint; requires ``weight_bytes_per_param``.

        Equation: ``memory_from_params(workload.parameters,
        weight_bytes_per_param).to('byte')``."""
        from ...physics.quantities import memory_from_params

        return memory_from_params(self.workload.parameters, self.weight_bytes_per_param).to("byte")

    @property
    def allreduce_factor(self) -> float:
        """Dimensionless per-worker ring AllReduce bytes per gradient byte, from
        ``calc_ring_allreduce_data_factor``.

        Bytes each worker moves in a ring AllReduce, per byte of gradient.

        Equation: ``calc_ring_allreduce_data_factor(dp_degree)``."""
        from ...physics.communication import calc_ring_allreduce_data_factor

        return calc_ring_allreduce_data_factor(self.dp_degree)

    @property
    def gather_factor(self) -> float:
        """Per-worker gathered gradient copies from the other data-parallel replicas.

        Bytes each worker receives when it gathers every other replica's gradient.

        Equation: ``dp_degree - 1``."""
        return self.dp_degree - 1

    @property
    def gather_traffic_ratio(self) -> float:
        """Dimensionless all-gather-to-ring-AllReduce traffic ratio; requires more than one
        replica.

        Gather over ring AllReduce traffic; equals ``dp_degree / 2``.

        Equation: ``gather_factor / allreduce_factor``."""
        return self.gather_factor / self.allreduce_factor


class ThroughputCostScenario(BaseScenario):
    """The same workload and resource budget served at ``baseline_rate`` without a set of
    controls and at ``controlled_rate`` with them.

    Notes
    -----
    Equation: Subtract rates for lost throughput and normalize by the baseline for
    fractional overhead.

    Assumptions: The two rates describe the same workload and resource allocation.

    Excludes: Causal attribution of observed rate changes or a latency model.

    References
    ----------
    Derivation: Subtract rates for lost throughput and normalize by the baseline for
    fractional overhead. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.SecureMultiTenantServing
    >>> round(float(scenario.rate_lost.magnitude), 4)
    150.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    baseline_rate: Quantity
    controlled_rate: Quantity

    @property
    def rate_lost(self) -> Quantity:
        """Throughput removed by controls, expressed in the baseline rate's units.

        Equation: ``(baseline_rate - controlled_rate).to(baseline_rate.units)``."""
        return (self.baseline_rate - self.controlled_rate).to(self.baseline_rate.units)

    @property
    def overhead(self) -> float:
        """Fraction of baseline throughput lost under the controls, for the same workload and
        resource budget.

        Equation: ``(1 - controlled_rate / baseline_rate).to('').magnitude``."""
        return (1 - self.controlled_rate / self.baseline_rate).to("").magnitude


class ProtectedMemoryFitScenario(BaseScenario):
    """``workload``'s weights at ``bytes_per_weight`` against ``hardware``'s protected memory
    (an enclave page cache), before activations.

    Notes
    -----
    Equation: Parameter count times weight width divided by protected capacity gives the
    storage share.

    Assumptions: Weight-only residency in the declared protected-memory capacity.

    Excludes: Activations, allocator reserve, enclave metadata, and execution speed.

    References
    ----------
    Derivation: Parameter count times weight width divided by protected capacity gives the
    storage share. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ResNet18Fp32InSgxEnclave
    >>> round(float(scenario.weight_bytes.magnitude), 4)
    46800000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    hardware: HardwareNode
    bytes_per_weight: Quantity

    @property
    def weight_bytes(self) -> Quantity:
        """Parameter-weight storage in bytes at the declared precision; activations and workspace
        are excluded.

        Equation: ``memory_from_params(workload.parameters, bytes_per_weight / Q_(1,
        'param')).to('byte')``."""
        from ...physics.quantities import memory_from_params

        return memory_from_params(self.workload.parameters, self.bytes_per_weight / Q_(1, "param")).to("byte")

    @property
    def capacity(self) -> Quantity:
        """Protected-memory capacity from the hardware registry, retaining its physical units.

        Equation: ``hardware.memory.capacity``."""
        return self.hardware.memory.capacity

    @property
    def capacity_share(self) -> float:
        """Dimensionless weight-storage demand divided by protected-memory capacity; values above
        one do not fit.

        Equation: ``(weight_bytes / capacity).to('').magnitude``."""
        return (self.weight_bytes / self.capacity).to("").magnitude


class EncryptedInferenceLatencyScenario(BaseScenario):
    """One inference at ``plaintext_latency``, with symmetric encryption around it
    (``encryption_overhead``) and computed under homomorphic encryption
    (``fhe_multiplier``).

    Notes
    -----
    Equation: Add symmetric transport overhead to baseline latency; multiply baseline
    latency for the FHE scenario.

    Assumptions: The encryption cost and homomorphic multiplier are supplied scenario
    inputs.

    Excludes: Cipher-specific implementation, accuracy effects, and independently predicted
    cryptographic cost.

    References
    ----------
    Derivation: Add symmetric transport overhead to baseline latency; multiply baseline
    latency for the FHE scenario. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.HealthMonitoringEncryptedInference
    >>> round(float(scenario.encrypted_transport_latency.magnitude), 4)
    20.5"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    plaintext_latency: Quantity
    encryption_overhead: Quantity
    fhe_multiplier: float

    @property
    def encrypted_transport_latency(self) -> Quantity:
        """Milliseconds for plaintext inference plus the supplied symmetric-encryption overhead.

        Equation: ``(plaintext_latency + encryption_overhead).to('ms')``."""
        return (self.plaintext_latency + self.encryption_overhead).to("ms")

    @property
    def fhe_latency(self) -> Quantity:
        """Seconds for plaintext inference scaled by the assumed homomorphic-encryption multiplier.

        Equation: ``(plaintext_latency * fhe_multiplier).to('second')``."""
        return (self.plaintext_latency * self.fhe_multiplier).to("second")


class GroupCommitScenario(BaseScenario):
    """``batch_size`` intents gathered for up to ``batch_timer`` and acknowledged by one
    ``flush_time``.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.GroupCommitModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.GroupCommitModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes ``mlsysim.solvers.GroupCommitModel.solve``
    with the stored fields and any derived properties described below. Individual property
    contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.IntentQuorumGroupCommit
    >>> round(float(scenario.result.work_per_record.magnitude), 4)
    0.25"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    batch_timer: Quantity
    flush_time: Quantity
    batch_size: int

    @property
    def result(self):
        """Return the modeled GroupCommit result from ``mlsysim.solvers.GroupCommitModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import GroupCommitModel

        return GroupCommitModel().solve(self.batch_timer, self.flush_time, self.batch_size)


class SnapshotVolumeScenario(BaseScenario):
    """A trajectory of ``turns`` turns growing ``state_growth`` per turn, checkpointed every
    ``stride``.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.SnapshotVolumeModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.SnapshotVolumeModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.SnapshotVolumeModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.GrowingTrajectorySnapshots
    >>> round(float(scenario.result.checkpoints), 4)
    100.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    turns: int
    stride: int
    state_growth: Quantity

    @property
    def result(self):
        """Return the modeled SnapshotVolume result from
        ``mlsysim.solvers.SnapshotVolumeModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import SnapshotVolumeModel

        return SnapshotVolumeModel().solve(self.turns, self.stride, self.state_growth)


class CheckpointTraceScenario(BaseScenario):
    """Checkpoint ``strides`` priced on ``turns`` turns that crash after the turns in
    ``crashes``.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.CheckpointTraceModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.CheckpointTraceModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.CheckpointTraceModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.TwoCrashCheckpointTrace
    >>> round(float(scenario.result.checkpoints[10]), 4)
    10.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    turns: int
    crashes: List[int]
    strides: List[int]
    pause: Quantity  # per checkpoint, including durable publication
    restore: Quantity  # per crash
    rebuild_per_turn: Quantity  # read, validate, and fold one logged turn
    reexecute_per_turn: Quantity  # re-run one unlogged turn's model call and tools

    @property
    def result(self):
        """Return the modeled CheckpointTrace result from
        ``mlsysim.solvers.CheckpointTraceModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import CheckpointTraceModel

        return CheckpointTraceModel().solve(
            self.turns,
            self.crashes,
            self.strides,
            self.pause,
            self.restore,
            self.rebuild_per_turn,
            self.reexecute_per_turn,
        )

    def stride_overhead(self, stride: float, lost_turn_cost: Quantity) -> Quantity:
        """Dimensionless checkpoint overhead from ``calc_stride_overhead`` using the observed
        crashes-per-turn frequency.

        Expected overhead per turn at ``stride`` on this trace's crash rate (seconds).

        Equation: ``calc_stride_overhead(pause, stride, len(crashes) / turns, lost_turn_cost)``."""
        from ...physics.reliability import calc_stride_overhead

        return calc_stride_overhead(self.pause, stride, len(self.crashes) / self.turns, lost_turn_cost)


class RetentionCostScenario(BaseScenario):
    """One trajectory's log (``turns`` x envelope and payload) and workspace snapshot kept for
    ``period``.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.RetentionCostModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.RetentionCostModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.RetentionCostModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RepairTrajectoryRetention
    >>> round(float(scenario.result.log_bytes.magnitude), 4)
    3960000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    turns: int
    envelope: Quantity  # event envelopes per turn
    payload: Quantity  # delivered inputs and result bodies per turn
    workspace: Quantity  # one immutable workspace snapshot
    rate: Quantity  # object storage, USD per byte-time
    period: Quantity
    attempt_cost: Quantity  # USD of one attempt

    @property
    def result(self):
        """Return the modeled RetentionCost result from
        ``mlsysim.solvers.RetentionCostModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import RetentionCostModel

        return RetentionCostModel().solve(
            self.turns, self.envelope, self.payload, self.workspace, self.rate, self.period, self.attempt_cost
        )


class RepairBreakEvenScenario(BaseScenario):
    """One bounded repair attempt against compensating and rebuilding to the same endpoint.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.RepairBreakEvenModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.RepairBreakEvenModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.RepairBreakEvenModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.TestConfigRepairVsRebuild
    >>> round(float(scenario.result.fallback_time.magnitude), 4)
    86.6"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    unwind_time: Quantity  # compensation
    rebuild_time: Quantity  # rebuild to the same readiness endpoint
    rebuild_input: int  # rebuild model tokens
    rebuild_output: int
    repair_input: int  # the whole isolated repair request
    repair_output: int
    startup_time: Quantity  # pre-output processing, not a measured time to first token
    output_rate: Quantity  # tokens per second
    validation_time: Quantity  # local validation, charged on success and failure
    input_price: Quantity  # USD per token
    output_price: Quantity
    assumed_success: float  # a policy comparison, not a measurement

    @property
    def result(self):
        """Return the modeled RepairBreakEven result from
        ``mlsysim.solvers.RepairBreakEvenModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import RepairBreakEvenModel

        return RepairBreakEvenModel().solve(
            self.unwind_time,
            self.rebuild_time,
            self.rebuild_input,
            self.rebuild_output,
            self.repair_input,
            self.repair_output,
            self.startup_time,
            self.output_rate,
            self.validation_time,
            self.input_price,
            self.output_price,
            self.assumed_success,
        )


class IrreversibleLossScenario(BaseScenario):
    """A workload's wrong-change risk priced with its pre-pivot check, from its agent profile.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.IrreversibleLossModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.IrreversibleLossModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.IrreversibleLossModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RemediationCanaryPivotLoss
    >>> round(float(scenario.attempt_cost.magnitude), 4)
    20.45"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Any  # a RemediationWorkload (Agents.*)
    catch_step: float  # catch-rate increment for the sensitivity

    @property
    def attempt_cost(self) -> Quantity:
        """USD cost per attempt copied from the remediation workload profile.

        Equation: ``workload.attempt_cost``."""
        return self.workload.attempt_cost

    @property
    def result(self):
        """Return the modeled IrreversibleLoss result from
        ``mlsysim.solvers.IrreversibleLossModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import IrreversibleLossModel

        w = self.workload
        return IrreversibleLossModel().solve(
            w.wrong_change_probability,
            w.wrong_change_impact,
            w.canary_catch_rate,
            w.canary_recovery_fraction,
            self.attempt_cost,
            self.catch_step,
        )


class UndetectedLoopScenario(BaseScenario):
    """An append-only loop that resends its history, run to a turn ceiling or stopped early.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.UndetectedLoopModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.UndetectedLoopModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.UndetectedLoopModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RepairLoopToTurnCeiling
    >>> round(float(scenario.result.appended_per_turn), 4)
    1100.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    first_input: int  # tokens
    output_per_turn: int
    observation_per_turn: int  # appended error observation
    turn_time: Quantity  # complete serial turn
    ceiling: int  # turns, counted from the loop's start
    stop_after: int  # intervention after this completed turn
    context_window: int  # combined input and output capacity, tokens
    input_price: Quantity  # USD per token
    output_price: Quantity

    @property
    def result(self):
        """Return the modeled UndetectedLoop result from
        ``mlsysim.solvers.UndetectedLoopModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import UndetectedLoopModel

        return UndetectedLoopModel().solve(
            self.first_input,
            self.output_per_turn,
            self.observation_per_turn,
            self.turn_time,
            self.ceiling,
            self.stop_after,
            self.context_window,
            self.input_price,
            self.output_price,
        )


class RetryAmplificationScenario(BaseScenario):
    """Nested retry layers, each with its own allowance, under one failing logical operation.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.RetryAmplificationModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.RetryAmplificationModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.RetryAmplificationModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ThreeLayerRetryAmplification
    >>> round(float(scenario.result.by_layer[0]), 4)
    1.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    layer_names: List[str]  # from the caller down to the dependency
    attempts_per_layer: int  # one try plus retries

    @property
    def result(self):
        """Return the modeled RetryAmplification result from
        ``mlsysim.solvers.RetryAmplificationModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import RetryAmplificationModel

        return RetryAmplificationModel().solve(self.attempts_per_layer, len(self.layer_names))


class BulkheadAdmissionScenario(BaseScenario):
    """One burst of blocking calls to a stalled dependency, uncapped and with a bulkhead.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.BulkheadAdmissionModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.BulkheadAdmissionModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.BulkheadAdmissionModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.BlockingCallBulkhead
    >>> round(float(scenario.result.uncapped_occupied), 4)
    256.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    arrivals: int  # blocking calls in the burst
    workers: int  # shared pool
    worker_cap: int  # occupied workers the dependency may hold
    queue_cap: int  # waiting requests, which hold no worker

    @property
    def result(self):
        """Return the modeled BulkheadAdmission result from
        ``mlsysim.solvers.BulkheadAdmissionModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import BulkheadAdmissionModel

        return BulkheadAdmissionModel().solve(self.arrivals, self.workers, self.worker_cap, self.queue_cap)


class RetryJitterScenario(BaseScenario):
    """A cohort of eligible retries released together, synchronized or with full jitter.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.RetryJitterModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.RetryJitterModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes ``mlsysim.solvers.RetryJitterModel.solve``
    with the stored fields and any derived properties described below. Individual property
    contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RetryCohortJitter
    >>> round(float(scenario.result.synchronized_rate.magnitude), 4)
    64000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    retries: int
    burst_window: Quantity  # stipulated window if all arrive together
    jitter_span: Quantity  # each retry's arrival drawn uniformly over this span

    @property
    def result(self):
        """Return the modeled RetryJitter result from ``mlsysim.solvers.RetryJitterModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import RetryJitterModel

        return RetryJitterModel().solve(self.retries, self.burst_window, self.jitter_span)


class ReleaseFootprintScenario(BaseScenario):
    """Resident copies of ``model`` that release controls hold on ``device``.

    A copy is the registry ``model_size``, or the parameters at
    ``bytes_per_param`` when given; it spans ``accelerators_per_copy`` devices.
    A canary, shadow, A/B test, blue-green switch, and hot standby each hold one
    more copy whatever share of traffic they serve. Optional: the node's
    ``node_accelerators``, a retraining run of ``retrain_accelerators`` for
    ``retrain_time`` on the same node, the ``canary_share``, and the staged
    exposure ``ramp`` (traffic shares, first stage the canary).

    Notes
    -----
    Equation: A copy occupies the selected model bytes; devices per copy are
    ceil(copy_bytes/device_capacity); copies reserve that count independently.

    Assumptions: A model copy can be partitioned across the counted devices; optional
    retraining fields are supplied before using retraining properties.

    Excludes: Communication, KV cache, allocator reserve, packing efficiencies, and proof
    that a distributed serving layout executes.

    References
    ----------
    Derivation: A copy occupies the selected model bytes; devices per copy are
    ceil(copy_bytes/device_capacity); copies reserve that count independently. This is
    declared scenario accounting rather than a fitted empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.DlrmReleaseFootprintH100
    >>> value = scenario.copy_bytes
    >>> str(value.units)
    'GB'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    model: Any
    device: HardwareNode
    bytes_per_param: Optional[Quantity] = None
    node_accelerators: Optional[int] = None
    retrain_accelerators: Optional[int] = None
    retrain_time: Optional[Quantity] = None
    canary_share: Optional[float] = None
    ramp: Optional[List[float]] = None

    @property
    def copy_bytes(self) -> Quantity:
        """Model-copy storage from the registry size or the explicitly selected bytes per
        parameter.

        Equation: ``model.size_in_bytes(bytes_per_param); otherwise model.model_size``.
        """
        if self.bytes_per_param is None:
            return self.model.model_size
        return self.model.size_in_bytes(self.bytes_per_param)

    @property
    def copy_fraction(self) -> float:
        """One model copy divided by one device's capacity, a dimensionless memory share.

        Equation: ``(copy_bytes / device.memory.capacity).to('').magnitude``.
        """
        return (self.copy_bytes / self.device.memory.capacity).to("").magnitude

    @property
    def accelerators_per_copy(self) -> int:
        """Ceiling of copy size over device capacity, ignoring partitioning and runtime overhead.

        Equation: ``_math.ceil(copy_fraction)``.
        """
        return _math.ceil(self.copy_fraction)

    def accelerators_for(self, copies: int) -> int:
        """Device count for independently resident model copies, without cross-copy packing.

        Equation: ``copies * accelerators_per_copy``.
        """
        return copies * self.accelerators_per_copy

    @property
    def retrain_accelerator_hours(self) -> Quantity:
        """Device-hours consumed by the declared retraining allocation and duration.

        Equation: ``(retrain_accelerators * retrain_time).to('hour')``.
        """
        return (self.retrain_accelerators * self.retrain_time).to("hour")

    @property
    def free_during_retrain(self) -> int:
        """Device count left after reserving one serving copy and the retraining allocation.

        Equation: ``node_accelerators - accelerators_per_copy - retrain_accelerators``.
        """
        return self.node_accelerators - self.accelerators_per_copy - self.retrain_accelerators

    def free_after(self, copies: int) -> int:
        """Device count remaining after the specified serving copies, with retraining absent.

        Equation: ``node_accelerators - accelerators_for(copies)``.
        """
        return self.node_accelerators - self.accelerators_for(copies)


class DualBankUpdateScenario(BaseScenario):
    """An over-the-air update that keeps a second flash bank: two copies of ``model``
    at ``bytes_per_param`` in the flash of ``device``.

    Notes
    -----
    Equation: Required flash is twice the selected model-byte footprint; utilization divides
    this by registered flash capacity.

    Assumptions: Two full model banks coexist during an update and the device exposes flash
    capacity.

    Excludes: Firmware, filesystem, bootloader, rollback metadata, and flash alignment
    reserves.

    References
    ----------
    Derivation: Required flash is twice the selected model-byte footprint; utilization
    divides this by registered flash capacity. This is declared scenario accounting rather
    than a fitted empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.KeywordSpotterDualBankEsp32S3
    >>> value = scenario.bank_bytes
    >>> str(value.units)
    'B'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    model: Any
    device: HardwareNode
    bytes_per_param: Quantity

    @property
    def bank_bytes(self) -> Quantity:
        """Model bytes in one update bank at the selected parameter precision.

        Equation: ``model.size_in_bytes(bytes_per_param)``.
        """
        return self.model.size_in_bytes(self.bytes_per_param)

    @property
    def banks_bytes(self) -> Quantity:
        """Storage for two simultaneous model banks, in bytes.

        Equation: ``2 * bank_bytes``.
        """
        return 2 * self.bank_bytes

    @property
    def flash(self) -> Quantity:
        """Registered device flash capacity, with no inferred reserve.

        Equation: ``device.memory.flash_capacity``.
        """
        return self.device.memory.flash_capacity

    @property
    def flash_share(self) -> float:
        """Two-bank footprint divided by registered flash capacity, dimensionless.

        Equation: ``(banks_bytes / flash).to('').magnitude``.
        """
        return (self.banks_bytes / self.flash).to("").magnitude


class AutomationBreakevenScenario(BaseScenario):
    """A pipeline that costs ``build_time`` once, against ``manual_rate`` of work by hand;
    the pipeline itself needs ``upkeep_rate``. Break-even is
    ``build / (manual - upkeep)``.

    Notes
    -----
    Equation: Build effort is repaid after build_time/(manual_rate-upkeep_rate), using the
    shared rent-crossover calculation.

    Assumptions: Manual and upkeep effort rates remain constant; omitted upkeep is zero.

    Excludes: Discounting, changing maintenance effort, and non-labor benefits of
    automation.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.physics.economics.calc_rent_crossover``.
    Their contracts supply the underlying equations; property equations show the remaining
    unit conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RetrainingPipelineAutomation
    >>> value = scenario.breakeven
    >>> str(value.units)
    'week'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    build_time: Quantity
    manual_rate: Quantity  # hours per week
    upkeep_rate: Optional[Quantity] = None

    @property
    def breakeven(self) -> Quantity:
        """Weeks until avoided manual effort repays build effort, accounting for recurring upkeep
        through ``calc_rent_crossover``.

        Equation: ``calc_rent_crossover(build_time, upkeep, manual_rate).to('week')``.
        """
        from ...physics.economics import calc_rent_crossover

        upkeep = self.upkeep_rate if self.upkeep_rate is not None else 0 * self.manual_rate
        return calc_rent_crossover(self.build_time, upkeep, self.manual_rate).to("week")


class SkewCostScenario(BaseScenario):
    """Training-serving skew that makes ``skew_rate`` of ``query_rate`` wrong, each
    wrong prediction costing ``cost_per_error``.

    Notes
    -----
    Equation: Expected error cost rate is ``skew_rate*query_rate*cost_per_error``; accumulated
    cost multiplies by duration.

    Assumptions: Skew share, query rate, and per-error cost remain constant over the period.

    Excludes: Measuring skew, attributing causality, heterogeneous error severity, and
    recovery effects.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.physics.economics.calc_skew_cost``. Their
    contracts supply the underlying equations; property equations show the remaining unit
    conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RankingSkewOnePercent
    >>> value = scenario.errors_per_day
    >>> round(float(value), 3)
    10000.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    skew_rate: float
    query_rate: Quantity  # queries per day
    cost_per_error: Quantity

    @property
    def errors_per_day(self) -> float:
        """Expected daily wrong predictions from query volume times the assumed skew error share.

        Equation: ``(skew_rate * query_rate * Q_(1, 'day')).to('').magnitude``.
        """
        return (self.skew_rate * self.query_rate * Q_(1, "day")).to("").magnitude

    @property
    def cost_rate(self) -> Quantity:
        """USD per day from error share, query rate, and cost per error via ``calc_skew_cost``.

        Equation: ``calc_skew_cost(skew_rate, query_rate, cost_per_error).to('USD/day')``.
        """
        from ...physics.economics import calc_skew_cost

        return calc_skew_cost(self.skew_rate, self.query_rate, self.cost_per_error).to("USD/day")

    def cost_over(self, period: Quantity) -> Quantity:
        """Accumulated USD over ``period`` at the constant modeled error-cost rate.

        Equation: ``(cost_rate * period).to('USD')``.
        """
        return (self.cost_rate * period).to("USD")


class AccuracyDecayScenario(BaseScenario):
    """A deployed model read at ``initial_accuracy`` and, ``elapsed`` later, at
    ``later_accuracy``; ``noticed`` is when someone outside monitoring saw it.
    ``decay_rate`` fits ``A(t) = A0 exp(-gamma t)`` through the two readings.

    Notes
    -----
    Equation: Two observed accuracies determine gamma in A(t)=``A0*exp(-gamma*t)``; threshold
    times invert the same curve.

    Assumptions: Accuracies are positive, elapsed time is positive, and the selected
    exponential form remains applicable when extrapolated.

    Excludes: Causal drift diagnosis, confidence intervals, and an empirical guarantee of
    future accuracy; inverse thresholds require nonzero gamma.

    References
    ----------
    Derivation: the scenario composes
    ``mlsysim.physics.statistics.calc_exponential_decay_rate``. Their contracts supply the
    underlying equations; property equations show the remaining unit conversions and
    accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RankingServiceAccuracyDrift
    >>> value = scenario.decay_rate
    >>> str(value.units)
    '1/d'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    initial_accuracy: float
    later_accuracy: float
    elapsed: Quantity
    noticed: Optional[Quantity] = None

    @property
    def decay_rate(self) -> Quantity:
        """Inverse-time exponential decay rate fitted through the two supplied accuracy
        observations.

        Equation: ``calc_exponential_decay_rate(initial_accuracy, later_accuracy, elapsed)``.
        """
        from ...physics.statistics import calc_exponential_decay_rate

        return calc_exponential_decay_rate(self.initial_accuracy, self.later_accuracy, self.elapsed)

    @property
    def drop(self) -> float:
        """Initial minus later accuracy, in fractional accuracy units rather than relative percent.

        Equation: ``initial_accuracy - later_accuracy``.
        """
        return self.initial_accuracy - self.later_accuracy

    # Reading the fitted curve (added 2026-10-20, ML operations part 2).
    def accuracy_at(self, age: Quantity) -> float:
        """Dimensionless accuracy at ``age`` from the fitted exponential curve.

        Equation: ``initial_accuracy * _math.exp(-(decay_rate * age).to('').magnitude)``.
        """
        return self.initial_accuracy * _math.exp(-(self.decay_rate * age).to("").magnitude)

    def time_to(self, accuracy: float) -> Quantity:
        """Days until the fitted curve reaches the requested positive accuracy, requiring nonzero
        decay rate.

        Equation: ``(_math.log(initial_accuracy / accuracy) / decay_rate).to('day')``.
        """
        return (_math.log(self.initial_accuracy / accuracy) / self.decay_rate).to("day")

    def trigger_for_floor(self, floor: float, delay: Quantity) -> float:
        """Accuracy threshold that decays to ``floor`` during the declared reset delay.

        Equation: ``floor * _math.exp((decay_rate * delay).to('').magnitude)``.
        """
        return floor * _math.exp((self.decay_rate * delay).to("").magnitude)
