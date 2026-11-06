"""Physical-machine scenarios for supervision."""

from __future__ import annotations
import math as _math
from typing import Any, Dict, Optional, Tuple
from ....core.types import Quantity
from ..base import BaseScenario


class TakeoverBudgetScenario(BaseScenario):
    """A machine at ``speed`` at a site asks a person to take over within ``takeover``.

    The site (an embodied site scenario) supplies the finished stopping
    budget's terms: pre-brake delay, credible deceleration, the resident
    stop's jerk bound, clear distance, localization bound, protective margin,
    tracking inset, the dispatch-to-admission allocation, the permission tick,
    and the chunk lease. ``configured_timeout`` is the protocol ceiling on the
    handover timer; ``manual_lease`` is the operator's lease once the channel
    is held (it replaces the chunk lease in the delay).

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.TakeoverBudgetModel.solve``.

    Assumptions: Inputs are routed to ``TakeoverBudgetModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to TakeoverBudgetModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseInLoopTakeoverAtAisleSpeed
    >>> value = case.result.deadline
    >>> print(f"{value:.3g~}")
    101 ms"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    speed: Quantity
    takeover: Quantity  # T_Sigma: human response + console transport
    configured_timeout: Optional[Quantity] = None
    manual_lease: Optional[Quantity] = None

    @property
    def result(self):
        """Evaluate the stated takeover budget case through ``TakeoverBudgetModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.TakeoverBudgetModel.solve``."""
        from ...solvers import TakeoverBudgetModel

        s = self.site
        return TakeoverBudgetModel().solve(
            self.speed,
            s.tau_delay,
            s.a_brake,
            s.d_clear,
            self.takeover,
            loc_margin=s.delta_loc,
            physical_margin=s.delta_margin,
            jerk=s.j_brake,
            tracking_bound=s.eps_track,
            dispatch_to_admission=s.t_dispatch_to_admission,
            configured_timeout=self.configured_timeout,
            tick=s.t_tick,
            manual_lease=self.manual_lease,
            chunk_lease=s.t_lease,
        )


class TakeoverCeilingLadderScenario(BaseScenario):
    """One takeover budget per sourced human response, each plus the same console ``transport``.

    Notes
    -----
    Equation: The calculation members below define ``result``.

    Assumptions: The same machine budget is recomputed for each supplied human-response time plus fixed transport.

    Excludes: A human-response distribution or a safety certification.

    References
    ----------
    Derivation: One takeover budget per sourced human response, each plus the same console ``transport``. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseTakeoverCeilingLadder
    >>> value = case.result.rows['mhp'].deadline
    >>> print(f"{value:.3g~}")
    101 ms"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    base: TakeoverBudgetScenario
    transport: Quantity
    responses: Dict[str, Quantity]  # key -> human response, in ladder order

    @property
    def result(self):
        """Return a typed ladder of takeover budgets, adding transport to each supplied human-response duration.

        Equation: ``TakeoverLadderResult(transport=transport, human=dict(responses), rows=rows)``."""
        from ...results import TakeoverLadderResult

        rows = {k: self.base.variant(takeover=(t + self.transport).to("ms")).result for k, t in self.responses.items()}
        return TakeoverLadderResult(transport=self.transport, human=dict(self.responses), rows=rows)


class BrakingBlendScenario(BaseScenario):
    """A bumpless transfer into braking on a site's base at ``speed``, and an arm joint's blend.

    The site supplies the braking deceleration, the resident stop's jerk
    bound, and the permission tick; the arm blend takes a torque change under
    a torque-rate ceiling.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.BrakingBlendModel.solve``.

    Assumptions: Inputs are routed to ``BrakingBlendModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to BrakingBlendModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseTakeoverBrakingBlend
    >>> value = case.result.blended_stop
    >>> print(f"{value:.3g~}")
    0.328 m"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    speed: Quantity
    torque_change: Optional[Quantity] = None
    torque_rate_limit: Optional[Quantity] = None

    @property
    def result(self):
        """Evaluate the stated braking blend case through ``BrakingBlendModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.BrakingBlendModel.solve``."""
        from ...solvers import BrakingBlendModel

        s = self.site
        return BrakingBlendModel().solve(
            self.speed,
            s.a_brake,
            s.j_brake,
            s.t_tick,
            torque_change=self.torque_change,
            torque_rate_limit=self.torque_rate_limit,
        )


class CommandAuthenticationScenario(BaseScenario):
    """Remote commands authenticated on a machine's permission processor, priced against a takeover budget.

    ``budget`` gives the speed, the partial spare, and the fallback deadline
    before and after the dispatch-to-admission allocation; the processor clock
    turns published cycle counts into time.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.CommandAuthenticationModel.solve``.

    Assumptions: Inputs are routed to ``CommandAuthenticationModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to CommandAuthenticationModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseRemoteCommandAuthentication
    >>> value = case.result.verify_time
    >>> print(f"{value:.3g~}")
    1.8 ms"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    budget: TakeoverBudgetScenario
    clock: Quantity
    verify_cycles: float
    mac_block_cycles: float
    mac_blocks: int
    freshness_window: Quantity

    @property
    def result(self):
        """Evaluate the stated command authentication case through ``CommandAuthenticationModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.CommandAuthenticationModel.solve``."""
        from ...solvers import CommandAuthenticationModel

        b = self.budget.result
        return CommandAuthenticationModel().solve(
            self.clock,
            self.verify_cycles,
            self.mac_block_cycles,
            self.mac_blocks,
            b.speed,
            b.machine.spare,
            b.deadline,
            b.deadline_after_admission,
            self.freshness_window,
        )


class DecisionChainScenario(BaseScenario):
    """``links`` independently kept decision records behind one admitted action, each retrievable
    with probability p; the whole chain resolves with p^links for each p in ``link_probabilities``.

    Notes
    -----
    Equation: Complete-chain probability is the product of the same per-link retrievability
    probability across links.

    Assumptions: Every required record must resolve, with independent identically likely
    retrieval outcomes.

    Excludes: Correlated record loss, redundancy, and time-dependent retrieval availability.

    References
    ----------
    Derivation: the scenario composes
    ``mlsysim.physics.statistics.calc_conjunctive_confidence``. Their contracts supply the
    underlying equations; property equations show the remaining unit conversions and
    accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.AdmittedCommandDecisionChain
    >>> value = scenario.completeness_by_link
    >>> type(value).__name__
    'dict'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    links: int
    link_probabilities: Tuple[float, ...]

    def completeness(self, p: float) -> float:
        """Probability that all ``links`` are retrievable when each independently resolves with
        probability ``p``.

        Equation: ``calc_conjunctive_confidence([p] * links)``.
        """
        from ....physics.statistics import calc_conjunctive_confidence

        return calc_conjunctive_confidence([p] * self.links)

    @property
    def completeness_by_link(self) -> Dict[float, float]:
        """Map each declared per-link probability to its complete-chain probability.

        Equation: ``{p: completeness(p) for p in link_probabilities}``.
        """
        return {p: self.completeness(p) for p in self.link_probabilities}


class OversightTimeScenario(BaseScenario):
    """A person asked to act through ``budget`` (a TakeoverBudgetScenario) against the time the
    hazard at its site leaves: the clear distance at the request speed.

    Notes
    -----
    Equation: The hazard window is clear distance/speed; takeover exposure compares the
    supplied takeover duration and solver travel with that window.

    Assumptions: Speed is constant while the hazard window is counted, and the nested budget
    supplies its own reaction and braking assumptions.

    Excludes: A certification of human response performance, obstacle motion, and
    uncertainty in the declared clearance.

    References
    ----------
    Derivation: The hazard window is clear distance/speed; takeover exposure compares the
    supplied takeover duration and solver travel with that window. This is declared scenario
    accounting rather than a fitted empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseRackEndOversightTime
    >>> value = scenario.hazard_window
    >>> str(value.units)
    's'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    budget: Any  # TakeoverBudgetScenario

    @property
    def hazard_window(self) -> Quantity:
        """Seconds until the machine traverses the clear distance at its current constant speed.

        Equation: ``(budget.site.d_clear / budget.speed).to('s')``.
        """
        return (self.budget.site.d_clear / self.budget.speed).to("s")

    @property
    def takeover_travel(self) -> Quantity:
        """Travel in meters reported by the takeover budget's underlying solver.

        Equation: ``budget.result.takeover_travel.to('m')``.
        """
        return self.budget.result.takeover_travel.to("m")

    @property
    def takeover_over_window(self) -> float:
        """Takeover duration divided by the available hazard window, dimensionless.

        Equation: ``(budget.takeover / hazard_window).to('').magnitude``.
        """
        return (self.budget.takeover / self.hazard_window).to("").magnitude


class SupervisorQueueScenario(BaseScenario):
    """One remote assistant serving ``machines``, each asking for help at ``request_rate``, each
    request holding the assistant for ``service_time``; ``utilization_cap`` sizes the largest fleet.

    Notes
    -----
    Equation: SupervisorQueueModel composes machine arrival rates and service time; the
    fleet ceiling floors ``utilization_cap/(request_rate*service_time)``.

    Assumptions: Arrivals are stationary and service time is a mean; a fleet ceiling
    requires a non-null utilization cap.

    Excludes: Shift scheduling, vacations, correlated fleet incidents, and guaranteed tail
    response times.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.solvers.SupervisorQueueModel``. Their
    contracts supply the underlying equations; property equations show the remaining unit
    conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseRemoteAssistantQueue
    >>> value = scenario.result
    >>> type(value).__name__
    'SupervisorQueueResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    request_rate: Quantity  # requests per machine per operating hour
    service_time: Quantity
    machines: int
    utilization_cap: Optional[float] = None

    @property
    def result(self):
        """Queue result from ``SupervisorQueueModel.solve`` with per-machine arrivals, service
        duration, and fleet size.

        Equation: ``SupervisorQueueModel().solve(request_rate, service_time, machines)``.
        """
        from ...solvers import SupervisorQueueModel

        return SupervisorQueueModel().solve(self.request_rate, self.service_time, self.machines)

    @property
    def max_machines(self) -> int:
        """Integer fleet ceiling obtained by dividing the utilization cap by per-machine service
        demand.

        Equation: ``_math.floor(utilization_cap / (request_rate *
        service_time).to('').magnitude)``.
        """
        return _math.floor(self.utilization_cap / (self.request_rate * self.service_time).to("").magnitude)


class DetectionShortfallScenario(BaseScenario):
    """``site.machine`` at ``speed`` with a stop sized for a group detected in ``reference`` of
    frames, met by ``groups`` detected less often; misses allowed in every frame with probability
    ``all_miss_probability``, frames independent.

    Notes
    -----
    Equation: Camera frame period is reciprocal frame rate; DetectionShortfallModel converts
    independent miss streaks into stopping delay and exposure.

    Assumptions: The supplied group detection rates are per-frame probabilities and frame
    misses are independent.

    Excludes: Correlated detection failures, fitted group error rates, and a physical safety
    guarantee.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.solvers.DetectionShortfallModel``. Their
    contracts supply the underlying equations; property equations show the remaining unit
    conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseGroupDetectionShortfall
    >>> value = scenario.frame_period
    >>> str(value.units)
    'ms'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    speed: Quantity
    all_miss_probability: float
    reference: float
    groups: Tuple[float, ...]

    @property
    def frame_period(self) -> Quantity:
        """Camera frame interval in milliseconds, reciprocal of its registered frame rate.

        Equation: ``(1 / site.machine.nav_camera.frame_rate).to('ms')``.
        """
        return (1 / self.site.machine.nav_camera.frame_rate).to("ms")

    @property
    def result(self):
        """Stopping and exposure result from ``DetectionShortfallModel.solve`` using the site's
        delay and brake budget.

        Equation: ``DetectionShortfallModel().solve(speed, frame_period, s.a_brake,
        s.machine.loaded_mass, all_miss_probability, reference, groups, delay=s.tau_delay)``.
        """
        from ...solvers import DetectionShortfallModel

        s = self.site
        return DetectionShortfallModel().solve(
            self.speed,
            self.frame_period,
            s.a_brake,
            s.machine.loaded_mass,
            self.all_miss_probability,
            self.reference,
            self.groups,
            delay=s.tau_delay,
        )


class SubgroupSampleScenario(BaseScenario):
    """An evaluation set of ``total`` people with a group of ``small_share``, both detected at
    ``recall``: exact intervals for each group, and the people per group a two-sided test at
    ``significance`` with ``power`` needs to tell ``rate_a`` from ``rate_b``.

    Notes
    -----
    Equation: Group counts are rounded shares of total; recall intervals use exact binomial
    tails, while group-size planning uses a pooled normal approximation.

    Assumptions: People are independent Bernoulli observations; assumed recall is converted
    to an integer success count by rounding.

    Excludes: A study-design correction for correlated people, repeated measurements, or
    multiple subgroup comparisons.

    References
    ----------
    Derivation: the scenario composes
    ``mlsysim.physics.statistics.calc_clopper_pearson_interval``,
    ``mlsysim.physics.statistics.calc_two_proportion_sample_size``. Their contracts supply
    the underlying equations; property equations show the remaining unit conversions and
    accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.PedestrianGroupEvaluationSample
    >>> value = scenario.small
    >>> round(float(value), 3)
    200.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    total: int
    small_share: float
    recall: float
    rate_a: float
    rate_b: float
    significance: float
    power: float
    confidence: float = 0.95

    @property
    def small(self) -> int:
        """Rounded people count in the smaller subgroup, from total count times subgroup share.

        Equation: ``round(total * small_share)``.
        """
        return round(self.total * self.small_share)

    @property
    def large(self) -> int:
        """Remaining people after subtracting the rounded smaller subgroup count.

        Equation: ``total - small``.
        """
        return self.total - self.small

    def interval(self, n: int):
        """Clopper-Pearson recall interval for ``n`` observations and a rounded assumed success
        count.

        Equation: ``calc_clopper_pearson_interval(round(recall * n), n, confidence)``.
        """
        from ....physics.statistics import calc_clopper_pearson_interval

        return calc_clopper_pearson_interval(round(self.recall * n), n, self.confidence)

    @property
    def small_interval(self):
        """Dimensionless exact binomial recall interval for the smaller group.

        Equation: ``interval(small)``.
        """
        return self.interval(self.small)

    @property
    def large_interval(self):
        """Dimensionless exact binomial recall interval for the larger group.

        Equation: ``interval(large)``.
        """
        return self.interval(self.large)

    @property
    def width_ratio(self) -> float:
        """Small-group interval width divided by large-group width, dimensionless.

        Equation: ``(b - a) / (d - c)``.
        """
        (a, b), (c, d) = self.small_interval, self.large_interval
        return (b - a) / (d - c)

    @property
    def per_group_needed(self) -> int:
        """Ceiling of the pooled two-proportion planning approximation at the declared significance
        and power.

        Equation: ``_math.ceil(calc_two_proportion_sample_size(lo, hi - lo, float(norm.ppf(1 -
        significance / 2)), float(norm.ppf(power)), pooled=True))``.
        """
        from scipy.stats import norm
        from ....physics.statistics import calc_two_proportion_sample_size

        lo, hi = sorted((self.rate_a, self.rate_b))
        return _math.ceil(
            calc_two_proportion_sample_size(
                lo, hi - lo, float(norm.ppf(1 - self.significance / 2)), float(norm.ppf(self.power)), pooled=True
            )
        )

    @property
    def share_needed(self) -> float:
        """Required per-group sample count as a fraction of the current total evaluation count.

        Equation: ``per_group_needed / total``.
        """
        return self.per_group_needed / self.total


class ExposureConcentrationScenario(BaseScenario):
    """``encounter_share`` of a site's machine-person encounters fall on ``user_share`` of its people:
    their per-person exposure against the site average, and everyone else's.

    Notes
    -----
    Equation: Selected-group exposure is encounter_share/user_share; other-group exposure
    uses the complementary shares.

    Assumptions: Population and encounter shares describe the same exposure period, with
    user_share between zero and one.

    Excludes: Severity differences among encounters and causal interpretation of group
    membership.

    References
    ----------
    Derivation: Selected-group exposure is encounter_share/user_share; other-group exposure
    uses the complementary shares. This is declared scenario accounting rather than a fitted
    empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseCrossingExposure
    >>> value = scenario.concentration
    >>> round(float(value), 3)
    8.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    encounter_share: float
    user_share: float

    @property
    def concentration(self) -> float:
        """Encounter exposure per member of the selected group relative to the population mean.

        Equation: ``encounter_share / user_share``.
        """
        return self.encounter_share / self.user_share

    @property
    def rest_ratio(self) -> float:
        """Encounter exposure per person outside the group relative to the population mean.

        Equation: ``(1 - encounter_share) / (1 - user_share)``.
        """
        return (1 - self.encounter_share) / (1 - self.user_share)


class EventRetentionScenario(BaseScenario):
    """Event-triggered retention: ``pre_trigger`` before and ``post_trigger`` after each of
    ``events_per_day`` events, against keeping the whole ``operating_day``.

    Notes
    -----
    Equation: Retained duration equals ``events_per_day*(pre_trigger+post_trigger)``; its share
    divides by operating-day duration.

    Assumptions: Every event creates a distinct counted retention window of the same length.

    Excludes: Merging overlapping windows, encoding overhead, and daily retention capped to
    the operating day.

    References
    ----------
    Derivation: Retained duration equals ``events_per_day*(pre_trigger+post_trigger)``; its
    share divides by operating-day duration. This is declared scenario accounting rather
    than a fitted empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseEventTriggeredRetention
    >>> value = scenario.window
    >>> str(value.units)
    's'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    pre_trigger: Quantity
    post_trigger: Quantity
    events_per_day: int
    operating_day: Quantity

    @property
    def window(self) -> Quantity:
        """Seconds retained around each event, adding pre-trigger and post-trigger duration.

        Equation: ``(pre_trigger + post_trigger).to('s')``.
        """
        return (self.pre_trigger + self.post_trigger).to("s")

    @property
    def kept_per_day(self) -> Quantity:
        """Daily retained minutes, counting each event window independently even if windows
        overlap.

        Equation: ``(events_per_day * window).to('minute')``.
        """
        return (self.events_per_day * self.window).to("minute")

    @property
    def kept_share(self) -> float:
        """Retained event-window time divided by operating-day duration, dimensionless.

        Equation: ``(kept_per_day / operating_day).to('').magnitude``.
        """
        return (self.kept_per_day / self.operating_day).to("").magnitude

    @property
    def reduction(self) -> float:
        """Continuous-retention duration divided by event-retention duration, requiring nonzero
        retained time.

        Equation: ``1 / kept_share``.
        """
        return 1 / self.kept_share
