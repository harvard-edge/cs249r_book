"""Cost, resource pricing, staffing, and investment tradeoffs.

Public compatibility imports remain in ``mlsysim.engine.scenario_types``.
"""

from __future__ import annotations
import math as _math
from typing import Any, Dict, List, Optional
from ...core.types import Quantity
from ...core.units import Q_
from ...models.types import Workload
from ...systems.types import Fleet
from .agents import TrajectoryResendScenario
from .base import BaseScenario, _usd_per_hour
from .robotics import SupervisorQueueScenario
from .serving import BatchingKneeScenario


class IdleCapacityCostScenario(BaseScenario):
    """``gpus`` accelerators held idle for ``duration`` at ``device_hour_rate``.

    ``gang_gpus`` (optional) is the gang the held GPUs belong to, for a
    partial gang that waits for its missing members.

    Notes
    -----
    Equation: device-hours = devices * idle hours; charge = device-hours * hourly price.

    Assumptions: All held devices incur the same constant price; gang shortfall requires a
    supplied gang size.

    Excludes: Opportunity cost, energy, and changes in price during the reservation.

    References
    ----------
    Derivation: device-hours = devices * idle hours; charge
    = device-hours * hourly price. The individual properties state units and intermediate
    accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ReservedGpusIdleThreeWeeks
    >>> round(float(case.gpu_hours), 3)
    252000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    gpus: int
    duration: Quantity
    device_hour_rate: Quantity
    gang_gpus: Optional[int] = None

    @property
    def gpu_hours(self) -> float:
        """Held device-hours: multiply the accelerator count by the idle duration in hours.

        Equation: ``return gpus * duration.to('hour').magnitude``."""
        return self.gpus * self.duration.to("hour").magnitude

    @property
    def cost(self) -> float:
        """Idle-capacity charge in USD, using the same hourly rate for every held device.

        Equation: ``return gpu_hours * _usd_per_hour(device_hour_rate)``."""
        return self.gpu_hours * _usd_per_hour(self.device_hour_rate)

    @property
    def missing_gpus(self) -> int:
        """Missing devices in the gang, as gang size minus held size; requires ``gang_gpus``.

        Equation: ``return gang_gpus - gpus``."""
        return self.gang_gpus - self.gpus


class UtilizationGainScenario(BaseScenario):
    """A fleet of ``fleet_gpus`` raised from ``utilization_before`` to ``utilization_after``.

    The gain in fully active GPU-equivalents is ``N (u_after - u_before)``.
    Buying that capacity under the old policy would take ``gain / u_before``
    raw GPUs, which is what the gain is worth. ``pools`` and ``pool_metrics``
    optionally record the GPU pools and each pool's measured shares.

    Notes
    -----
    Equation: productive gain = fleet size * utilization change; raw equivalent =
    productive gain / old utilization.

    Assumptions: Utilization fractions refer to the same useful-work definition, and old
    utilization is positive.

    Excludes: A measured causal effect of a policy, workload-mix changes, and procurement
    overhead.

    References
    ----------
    Derivation: productive gain = fleet size * utilization
    change; raw equivalent = productive gain / old utilization. The individual properties
    state units and intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.MixedFleetPolicyFix1K
    >>> round(float(case.improvement), 3)
    0.24"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    fleet_gpus: int
    utilization_before: float
    utilization_after: float
    device_hour_rate: Quantity
    pools: Dict[str, int] = {}  # GPUs per pool
    pool_metrics: Dict[str, Dict[str, float]] = {}  # per pool: allocated/active, before/after
    pending_jobs: Optional[int] = None  # queue depth before the fix
    job_memory_need: Optional[Quantity] = None  # peak memory of the jobs queued on the largest pool

    @property
    def improvement(self) -> float:
        """Increase in productive utilization as a dimensionless fraction, after minus
        before.

        Equation: ``return utilization_after - utilization_before``."""
        return self.utilization_after - self.utilization_before

    @property
    def productive_gain(self) -> float:
        """Additional fully busy GPU-equivalents, counting the utilization gain across the
        fleet.

        Equation: ``return fleet_gpus * improvement``."""
        return self.fleet_gpus * self.improvement

    @property
    def raw_gpu_equivalent(self) -> float:
        """Raw devices needed to supply that gain under the old utilization; the old share
        must be positive.

        Equation: ``return productive_gain / utilization_before``."""
        return self.productive_gain / self.utilization_before

    @property
    def annual_value(self) -> float:
        """Annual USD value of the equivalent devices, assuming the same hourly rate all
        year.

        Equation: ``return raw_gpu_equivalent * _usd_per_hour(device_hour_rate) *
        HOURS_PER_YEAR``."""
        from ...core.units import HOURS_PER_YEAR

        return self.raw_gpu_equivalent * _usd_per_hour(self.device_hour_rate) * HOURS_PER_YEAR


class SpotEffectiveCostScenario(BaseScenario):
    """Spot capacity priced per hour of retained progress.

    Each interruption bills ``checkpoint_overhead + restart_overhead`` that
    retain no progress, once per ``retained_interval`` of progress, so the
    effective price is the spot price times billed time over retained time.

    Notes
    -----
    Equation: effective price = spot price * (retained time + checkpoint time + restart
    time) / retained time.

    Assumptions: The same interruption overhead accompanies every positive retained-
    progress interval.

    Excludes: Interruption distributions, unavailable replacement capacity, and nonlinear
    recovery work.

    References
    ----------
    Derivation: effective price = spot price * (retained
    time + checkpoint time + restart time) / retained time. The individual properties
    state units and intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.SpotInterruptionEffectiveCost
    >>> round(float(case.nominal_discount), 3)
    0.65"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    on_demand_rate: Quantity
    spot_rate: Quantity
    checkpoint_overhead: Quantity
    restart_overhead: Quantity
    retained_interval: Quantity

    @property
    def nominal_discount(self) -> float:
        """Fractional price reduction before interruption overhead, relative to on-demand
        price.

        Equation: ``return 1 - _usd_per_hour(spot_rate) / _usd_per_hour(on_demand_rate)``."""
        return 1 - _usd_per_hour(self.spot_rate) / _usd_per_hour(self.on_demand_rate)

    @property
    def lost_per_interruption(self) -> Quantity:
        """Minutes billed for checkpoint and restart without retaining additional progress.

        Equation: ``return (checkpoint_overhead + restart_overhead).to('minute')``."""
        return (self.checkpoint_overhead + self.restart_overhead).to("minute")

    @property
    def billed_over_retained(self) -> float:
        """Billed time per unit of useful progress, a dimensionless multiplier; retained time
        must be positive.

        Equation: ``interval = retained_interval.to('minute').magnitude; return (interval
        + lost_per_interruption.magnitude) / interval``."""
        interval = self.retained_interval.to("minute").magnitude
        return (interval + self.lost_per_interruption.magnitude) / interval

    @property
    def effective_rate(self) -> float:
        """USD per device-hour of retained progress, including the interruption time
        multiplier.

        Equation: ``return _usd_per_hour(spot_rate) * billed_over_retained``."""
        return _usd_per_hour(self.spot_rate) * self.billed_over_retained

    @property
    def effective_cost_ratio(self) -> float:
        """Effective spot price divided by on-demand price, a dimensionless cost ratio.

        Equation: ``return effective_rate / _usd_per_hour(on_demand_rate)``."""
        return self.effective_rate / _usd_per_hour(self.on_demand_rate)

    @property
    def effective_savings(self) -> float:
        """Fraction of on-demand cost saved after overhead; negative values mean spot costs
        more.

        Equation: ``return 1 - effective_cost_ratio``."""
        return 1 - self.effective_cost_ratio


class SpotRunScenario(BaseScenario):
    """A training run of ``run_duration`` on ``fleet``, on demand against spot capacity.

    Spot adds ``checkpoint_overhead_fraction`` of the run for checkpoints and
    ``restart_per_interruption`` for each of ``interruptions_per_day`` per day
    of the run.

    Notes
    -----
    Equation: spot bill = device count * (useful hours + checkpoint hours + restart hours)
    * spot price.

    Assumptions: The run keeps a fixed fleet. Interruption counts use complete baseline
    days; restart overhead does not generate more interruptions.

    Excludes: Partial-day interruptions, fluctuating spot prices, queue delay, and
    availability risk.

    References
    ----------
    Derivation: spot bill = device count * (useful hours +
    checkpoint hours + restart hours) * spot price. The individual properties state units
    and intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Llama2_70BSpotRun512
    >>> round(float(case.gpus), 3)
    512.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    fleet: Fleet
    workload: Optional[Workload] = None
    run_duration: Quantity
    on_demand_rate: Quantity
    spot_rate: Quantity
    checkpoint_overhead_fraction: float
    interruptions_per_day: int
    restart_per_interruption: Quantity

    @property
    def gpus(self) -> int:
        """Device count supplied by the scenario's fleet registry entry.

        Equation: ``return fleet.total_accelerators``."""
        return self.fleet.total_accelerators

    @property
    def run_hours(self) -> float:
        """Baseline useful run duration as a scalar number of hours.

        Equation: ``return run_duration.to('hour').magnitude``."""
        return self.run_duration.to("hour").magnitude

    @property
    def run_days(self) -> float:
        """Baseline useful run duration as a scalar number of days.

        Equation: ``return run_duration.to('day').magnitude``."""
        return self.run_duration.to("day").magnitude

    @property
    def on_demand_cost(self) -> float:
        """Baseline USD charge: device count times useful hours times on-demand hourly price.

        Equation: ``return gpus * run_hours * _usd_per_hour(on_demand_rate)``."""
        return self.gpus * self.run_hours * _usd_per_hour(self.on_demand_rate)

    @property
    def spot_base_cost(self) -> float:
        """USD charge for useful work alone at the spot hourly price, before overhead.

        Equation: ``return gpus * run_hours * _usd_per_hour(spot_rate)``."""
        return self.gpus * self.run_hours * _usd_per_hour(self.spot_rate)

    @property
    def checkpoint_extra_hours(self) -> float:
        """Additional wall-clock hours from the declared checkpoint fraction of useful run
        time.

        Equation: ``return checkpoint_overhead_fraction * run_hours``."""
        return self.checkpoint_overhead_fraction * self.run_hours

    @property
    def interruptions(self) -> float:
        """Expected interruption count for complete baseline days only; partial days are
        discarded by floor division.

        Equation: ``return interruptions_per_day * (run_hours // 24)``."""
        return self.interruptions_per_day * (self.run_hours // 24)

    @property
    def restart_total_hours(self) -> float:
        """Additional hours from interruption count times restart duration per interruption.

        Equation: ``return interruptions *
        restart_per_interruption.to('hour').magnitude``."""
        return self.interruptions * self.restart_per_interruption.to("hour").magnitude

    @property
    def spot_total_hours(self) -> float:
        """Billed spot wall-clock hours, adding useful work, checkpoint, and restart time.

        Equation: ``return run_hours + checkpoint_extra_hours + restart_total_hours``."""
        return self.run_hours + self.checkpoint_extra_hours + self.restart_total_hours

    @property
    def spot_total_cost(self) -> float:
        """USD charge for all billed spot device-hours at the spot rate.

        Equation: ``return gpus * spot_total_hours * _usd_per_hour(spot_rate)``."""
        return self.gpus * self.spot_total_hours * _usd_per_hour(self.spot_rate)

    @property
    def savings(self) -> float:
        """USD saved by spot relative to on-demand; negative means a cost increase.

        Equation: ``return on_demand_cost - spot_total_cost``."""
        return self.on_demand_cost - self.spot_total_cost

    @property
    def savings_share(self) -> float:
        """Fraction of baseline on-demand cost saved; requires nonzero baseline cost.

        Equation: ``return savings / on_demand_cost``."""
        return self.savings / self.on_demand_cost

    @property
    def longer_share(self) -> float:
        """Fractional run-time increase caused by spot checkpoint and restart overhead.

        Equation: ``return spot_total_hours / run_hours - 1``."""
        return self.spot_total_hours / self.run_hours - 1


class CloudEdgeCostScenario(BaseScenario):
    """Serving a network of sites from the cloud (pay per image) or from a device at each site (buy up front).

    Undiscounted; ``payback`` is the up-front device cost over the yearly running
    cost the devices save. ``image_size``, ``uplink``, and the two latencies are
    optional, for comparing response times.

    Notes
    -----
    Equation: annual cost = annual images * per-image price + fixed cost; payback = edge
    capital / annual savings.

    Assumptions: Volumes and prices remain constant and costs are undiscounted. Optional
    upload comparison requires image size and uplink.

    Excludes: Hardware replacement, discounting, fleet outages, and inference-quality
    differences.

    References
    ----------
    Derivation: annual cost = annual images * per-image
    price + fixed cost; payback = edge capital / annual savings. The individual properties
    state units and intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RetinopathyScreeningCloudVersusEdge
    >>> round(float(case.annual_images), 3)
    9125000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    sites: int
    cases_per_site_day: int
    images_per_case: int
    days_per_year: int
    cloud_cost_per_image: Quantity
    cloud_fixed_annual: Quantity
    edge_device_cost: Quantity
    edge_maintenance_annual: Quantity
    edge_cost_per_image: Quantity
    image_size: Optional[Quantity] = None
    uplink: Optional[Quantity] = None
    cloud_latency: Optional[Quantity] = None
    edge_latency: Optional[Quantity] = None

    @property
    def annual_images(self) -> int:
        """Images processed in one modeled year, multiplying sites, daily cases, images per
        case, and operating days.

        Equation: ``return sites * cases_per_site_day * images_per_case * days_per_year``."""
        return self.sites * self.cases_per_site_day * self.images_per_case * self.days_per_year

    @property
    def cloud_inference_annual(self) -> Quantity:
        """Cloud variable cost rate in currency per year for the annual image count.

        Equation: ``return annual_images * cloud_cost_per_image / Q_(1, 'year')``."""
        return self.annual_images * self.cloud_cost_per_image / Q_(1, "year")

    @property
    def cloud_annual(self) -> Quantity:
        """Total annual cloud cost rate, variable inference plus fixed annual charges.

        Equation: ``return cloud_inference_annual + cloud_fixed_annual``."""
        return self.cloud_inference_annual + self.cloud_fixed_annual

    @property
    def edge_capex(self) -> Quantity:
        """Up-front currency cost of one edge device per site.

        Equation: ``return sites * edge_device_cost``."""
        return self.sites * self.edge_device_cost

    @property
    def edge_inference_annual(self) -> Quantity:
        """Annual edge variable cost rate for the same image volume.

        Equation: ``return annual_images * edge_cost_per_image / Q_(1, 'year')``."""
        return self.annual_images * self.edge_cost_per_image / Q_(1, "year")

    @property
    def edge_annual(self) -> Quantity:
        """Annual edge cost rate, adding maintenance and variable inference.

        Equation: ``return edge_maintenance_annual + edge_inference_annual``."""
        return self.edge_maintenance_annual + self.edge_inference_annual

    @property
    def edge_first_year(self) -> Quantity:
        """First-year edge currency spend, up-front capital plus one year of operating costs.

        Equation: ``return edge_capex + edge_annual * Q_(1, 'year')``."""
        return self.edge_capex + self.edge_annual * Q_(1, "year")

    @property
    def annual_savings(self) -> Quantity:
        """Cloud minus edge operating cost rate in currency per year.

        Equation: ``return cloud_annual - edge_annual``."""
        return self.cloud_annual - self.edge_annual

    @property
    def payback(self) -> Quantity:
        """Undiscounted capital-recovery duration from
        ``physics.economics.calc_payback_period``.

        Equation: ``return calc_payback_period(edge_capex, annual_savings)``."""
        from ...physics.economics import calc_payback_period

        return calc_payback_period(self.edge_capex, self.annual_savings)

    @property
    def cloud_upload_time(self) -> Quantity:
        """Ideal single-image uplink time from ``physics.quantities.transfer_time`` before
        remote processing.

        Equation: ``return transfer_time(image_size, uplink)``."""
        from ...physics.quantities import transfer_time

        return transfer_time(self.image_size, self.uplink)


class ServingCostScenario(BaseScenario):
    """One model version's training spend against its first year of serving.

    Notes
    -----
    Equation: annual serving bill = users * daily requests per user * days * price per
    request.

    Assumptions: User demand and unit price remain constant; the efficiency gain is a
    stipulated cost reduction.

    Excludes: Discounted cash flow, traffic growth, retraining, and proving that
    efficiency can improve by the specified share.

    References
    ----------
    Derivation: annual serving bill = users * daily requests
    per user * days * price per request. The individual properties state units and
    intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.LlmServingFirstYear
    >>> round(float(case.daily_requests), 3)
    50000000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    training_cost: float  # USD, one-time
    daily_users: int
    requests_per_user_per_day: int
    cost_per_request: float  # USD per served request
    efficiency_gain: float  # fractional cut in cost per request
    days_per_year: int

    @property
    def daily_requests(self) -> float:
        """Requests per day from user count times requests per user.

        Equation: ``return daily_users * requests_per_user_per_day``."""
        return self.daily_users * self.requests_per_user_per_day

    @property
    def annual_requests(self) -> float:
        """Requests per modeled year using the scenario's declared number of days.

        Equation: ``return daily_requests * days_per_year``."""
        return self.daily_requests * self.days_per_year

    @property
    def annual_serving_cost(self) -> float:
        """First-year serving bill in USD at constant cost per request.

        Equation: ``return annual_requests * cost_per_request``."""
        return self.annual_requests * self.cost_per_request

    @property
    def serving_to_training(self) -> float:
        """Annual serving spend divided by one-time training spend, a dimensionless ratio.

        Equation: ``return annual_serving_cost / training_cost``."""
        return self.annual_serving_cost / self.training_cost

    @property
    def efficiency_savings(self) -> float:
        """Annual USD saved by the stipulated fractional cut in serving cost.

        Equation: ``return annual_serving_cost * efficiency_gain``."""
        return self.annual_serving_cost * self.efficiency_gain

    @property
    def monthly_serving_cost(self) -> float:
        """Average serving bill in USD per month, annual bill divided into twelve equal
        periods.

        Equation: ``return annual_serving_cost / 12``."""
        return self.annual_serving_cost / 12

    @property
    def crossover_months(self) -> float:
        """Months of average serving spend equal to the one-time training bill.

        Equation: ``return training_cost / monthly_serving_cost``."""
        return self.training_cost / self.monthly_serving_cost

    @property
    def crossover_weeks(self) -> float:
        """Equivalent serving-spend crossover in weeks, using the scenario's days per year.

        Equation: ``return crossover_months * days_per_year / 12 / 7``."""
        return self.crossover_months * self.days_per_year / 12 / 7


class OperationalLoadScenario(BaseScenario):
    """Assumed operational-load shapes for ``N`` models: alerts ``a N``, deployments ``N ln N``,
    and pairwise conflicts ``c N^2`` against one team's ``team_capacity``.

    The shapes are a sketch, not measurements; ``crossing_models`` is the first
    model count whose total reaches the capacity.

    Notes
    -----
    Equation: The constructed load is ``a*n`` + ``n*ln(n)`` + ``c*n**2``, compared with team_capacity on an integer grid.

    Assumptions: Common abstract work units and a capacity crossing within 1..max_models.

    Excludes: Empirical staffing forecasts and confidence intervals; if no crossing exists the current argmax selection returns the first grid point.

    References
    ----------
    Derivation: The linear alert term, ``n*ln(n)`` deployment term, and quadratic conflict
    term are deliberately constructed workload shapes, not fitted laws. Adding them is
    meaningful only because the caller places them in common work units; the integer
    grid then locates a crossing of the supplied team capacity.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.OperationalLoadSketch
    >>> len(s.model_counts()) == s.max_models
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    alerts_per_model: float
    conflict_coefficient: float  # N models form about N^2/2 pairs
    team_capacity: float
    max_models: int

    def model_counts(self):
        """Integer model-count grid from one through max_models.

        Equation: ``_np.arange(1, max_models + 1)``."""
        import numpy as _np

        return _np.arange(1, self.max_models + 1)

    def alerts(self, n):
        """Illustrative alert work proportional to model count; scalar or array input is accepted.

        Equation: ``alerts_per_model * n``."""
        return self.alerts_per_model * n

    def deployments(self, n):
        """Illustrative deployment work n*ln(n); this is a chosen growth shape, not measured cost.

        Equation: ``n * _np.log(n)``."""
        import numpy as _np

        return n * _np.log(n)

    def conflicts(self, n):
        """Illustrative coordination work proportional to squared model count.

        Equation: ``conflict_coefficient * n ** 2``."""
        return self.conflict_coefficient * n**2

    def total(self, n):
        """Sum of alert, deployment, and conflict work in the scenario's common capacity units.

        Equation: ``alerts(n) + deployments(n) + conflicts(n)``."""
        return self.alerts(n) + self.deployments(n) + self.conflicts(n)

    @property
    def crossing_models(self) -> int:
        """First grid count selected by the capacity-crossing mask; callers must ensure a crossing exists.

        Equation: ``n = model_counts(); int(n[int(_np.argmax(total(n) >=
        team_capacity))])``."""
        import numpy as _np

        n = self.model_counts()
        return int(n[int(_np.argmax(self.total(n) >= self.team_capacity))])


class FeatureHistoryScenario(BaseScenario):
    """Full-snapshot feature history: ``entities`` x ``features`` values every ``update_interval``,
    kept for ``retention``, each row a value, an entity key, and ``timestamps`` timestamps.

    Notes
    -----
    Equation: Stored versions are retention/update_interval; row bytes add value, key, and repeated timestamps.

    Assumptions: Full snapshots at regular intervals and equal row widths for every entity-feature pair.

    Excludes: Compression, delta storage, indexing, replication, and partial interval rounding.

    References
    ----------
    Derivation: Regular full snapshots retain retention/update_interval versions of
    every entity-feature pair. Every version stores its value plus a repeated key and
    timestamps, so multiplying version count by complete row width includes metadata
    explicitly.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.RecommendationFeatureHistory
    >>> s.total_bytes.to("byte").magnitude > 0
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    entities: float
    features: int
    retention: Quantity
    update_interval: Quantity
    value_bytes: Quantity
    key_bytes: Quantity
    timestamp_bytes: Quantity
    timestamps: int = 2  # event time and availability time

    @property
    def versions_per_feature(self) -> float:
        """Dimensionless retained snapshot count from retention duration over update interval.

        Equation: ``(retention / update_interval).to('').magnitude``."""
        return (self.retention / self.update_interval).to("").magnitude

    @property
    def values(self) -> float:
        """Total stored feature-version count across entities and features.

        Equation: ``entities * features * versions_per_feature``."""
        return self.entities * self.features * self.versions_per_feature

    @property
    def row_bytes(self) -> Quantity:
        """Bytes per retained value including its key and timestamp metadata.

        Equation: ``(value_bytes + key_bytes + timestamps * timestamp_bytes).to('byte')``."""
        return (self.value_bytes + self.key_bytes + self.timestamps * self.timestamp_bytes).to("byte")

    @property
    def total_bytes(self) -> Quantity:
        """Full snapshot-history footprint in bytes, including repeated metadata.

        Equation: ``values * row_bytes``."""
        return self.values * self.row_bytes

    @property
    def metadata_share(self) -> float:
        """Dimensionless fraction of each history row occupied by keys and timestamps.

        Equation: ``float(((row_bytes - value_bytes) / row_bytes).to('').magnitude)``."""
        return float(((self.row_bytes - self.value_bytes) / self.row_bytes).to("").magnitude)


class RevenueRegressionScenario(BaseScenario):
    """A metric regression that cuts revenue in proportion for ``undetected`` (or for ``alerted``
    with an alert and containment); revenue is ``weekly_revenue`` per week.

    Notes
    -----
    Equation: Loss is ``relative_drop*weekly_revenue*(duration/one_week)``; shorter windows reduce it proportionally.

    Assumptions: Constant revenue rate and a linear assumed revenue response to the metric regression.

    Excludes: Demand elasticity, recovery lag, and causal identification of the metric-revenue relation.

    References
    ----------
    Derivation: The scenario assumes revenue falls in direct proportion to
    relative_drop. A constant weekly revenue rate integrated over duration therefore
    loses relative_drop times revenue_rate times duration; an earlier alert changes only
    the exposure window.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.AggregationWindowRevenueIncident
    >>> s.undetected_loss.to("USD").magnitude >= s.alerted_loss.to("USD").magnitude
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    weekly_revenue: Quantity  # USD per week
    relative_drop: float  # fraction of revenue lost while the regression lasts
    metric_before: float
    metric_after: float
    undetected: Quantity
    alerted: Quantity
    feature_mean_before: Optional[float] = None
    feature_mean_after: Optional[float] = None
    freshness_slo: Optional[Quantity] = None
    shift_threshold: Optional[float] = None

    def loss(self, duration: Quantity) -> Quantity:
        """Revenue lost in USD over the supplied duration at the assumed proportional drop.

        Equation: ``(relative_drop * weekly_revenue * (duration / Q_(1,
        'week')).to('').magnitude).to('USD')``."""
        return (self.relative_drop * self.weekly_revenue * (duration / Q_(1, "week")).to("").magnitude).to("USD")

    @property
    def undetected_loss(self) -> Quantity:
        """Revenue loss in USD over the uncontained incident window.

        Equation: ``loss(undetected)``."""
        return self.loss(self.undetected)

    @property
    def alerted_loss(self) -> Quantity:
        """Revenue loss in USD over the alert-and-containment window.

        Equation: ``loss(alerted)``."""
        return self.loss(self.alerted)

    @property
    def window_ratio(self) -> float:
        """Dimensionless uncontained duration divided by the contained duration.

        Equation: ``(undetected / alerted).to('').magnitude``."""
        return (self.undetected / self.alerted).to("").magnitude

    @property
    def impact_reduction(self) -> float:
        """Dimensionless fraction of proportional loss avoided by the shorter detection window.

        Equation: ``1 - (alerted / undetected).to('').magnitude``."""
        return 1 - (self.alerted / self.undetected).to("").magnitude

    @property
    def metric_change(self) -> float:
        """Signed relative change of the monitored metric from its initial value.

        Equation: ``metric_after / metric_before - 1``."""
        return self.metric_after / self.metric_before - 1

    @property
    def feature_shift(self) -> float:
        """Signed relative shift of the feature mean from its initial value.

        Equation: ``(feature_mean_after - feature_mean_before) / feature_mean_before``."""
        return (self.feature_mean_after - self.feature_mean_before) / self.feature_mean_before


class ClickLossScenario(BaseScenario):
    """A silent click-through regression from ``ctr_before`` to ``ctr_after`` at ``request_rate``,
    caught after ``detection_time``; each click is worth ``value_per_click``.

    Notes
    -----
    Equation: Exposed requests are ``rate*time``; lost clicks are ``exposed_requests*(ctr_before-ctr_after)``.

    Assumptions: Constant request rate, click-through rates, and value per click during the delay.

    Excludes: Seasonality, click dependence, recovery effects, and causal revenue validation.

    References
    ----------
    Derivation: At constant request rate, the observation window contains ``rate*time``
    requests. Multiplying by the absolute click-through decrease yields expected lost
    clicks, and multiplying by value_per_click yields the assumed monetary loss.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.DelayedAlertClickLoss
    >>> s.loss.to("USD").magnitude >= 0
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    request_rate: Quantity
    detection_time: Quantity
    ctr_before: float
    ctr_after: float
    value_per_click: Quantity

    @property
    def requests(self) -> float:
        """Request count exposed during the configured detection delay.

        Equation: ``(request_rate * detection_time).to('').magnitude``."""
        return (self.request_rate * self.detection_time).to("").magnitude

    @property
    def ctr_drop(self) -> float:
        """Absolute click-through probability decrease, measured in probability points.

        Equation: ``ctr_before - ctr_after``."""
        return self.ctr_before - self.ctr_after

    @property
    def lost_clicks(self) -> float:
        """Expected lost-click count under constant request rate and click-through decrease.

        Equation: ``requests * ctr_drop``."""
        return self.requests * self.ctr_drop

    @property
    def loss(self) -> Quantity:
        """Expected lost revenue in USD at the supplied value per click.

        Equation: ``(lost_clicks * value_per_click).to('USD')``."""
        return (self.lost_clicks * self.value_per_click).to("USD")

    @property
    def loss_rate(self) -> Quantity:
        """Expected revenue loss rate in USD/hour during the regression.

        Equation: ``(loss / detection_time).to('USD/hour')``."""
        return (self.loss / self.detection_time).to("USD/hour")


class RolloutExposureScenario(BaseScenario):
    """Users exposed until rollback under an all-at-once cutover against a canary share,
    with the same request rate, harm, and time to rollback.

    Notes
    -----
    Equation: Exposure scales with traffic share, so equal-window exposure ratio is full_share/canary_share.

    Assumptions: Positive canary share, identical request rate, harm per request, and rollback duration.

    Excludes: Detection-time differences between rollout policies and heterogeneous user harm.

    References
    ----------
    Derivation: With identical rate, harm per exposed request, and rollback time, both
    exposure totals contain the same factors. Canceling them leaves the full-traffic
    share divided by the canary share.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.CanaryVersusCutoverExposure
    >>> s.exposure_ratio > 0
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    full_share: float
    canary_share: float
    time_to_rollback: Optional[Quantity] = None

    @property
    def exposure_ratio(self) -> float:
        """Dimensionless cutover-to-canary traffic exposure ratio for identical rollback windows.

        Equation: ``full_share / canary_share``."""
        return self.full_share / self.canary_share


class CanaryScheduleScenario(BaseScenario):
    """A staged rollout that needs ``samples_per_variant`` guardrail samples per stage at
    ``request_rate``; ``stages`` is the configured ladder of (traffic share, duration).

    Optional (added 2026-10-20): ``label_delay`` is the time from admission to a
    verdict, so ``labeled_by_completion`` counts the candidate tasks whose labels
    exist when the ladder ends (admitted at least that long before the end).

    Notes
    -----
    Equation: Admission counts integrate ``share*rate`` over stages; labeled counts truncate admissions at total_time-label_delay.

    Assumptions: Constant request rate within every stage and one deterministic
    label delay. Set label_delay before requesting labeled_by_completion.

    Excludes: Label failures, stochastic traffic, and power calculations beyond the supplied sample target.

    References
    ----------
    Derivation: Each stage contributes ``traffic_share*request_rate*duration`` admissions.
    Only admissions before completion-label_delay have available labels at completion,
    so the labeled count integrates each stage only up to that cutoff. Collection time
    delegates to mlsysim.physics.statistics.calc_sample_collection_time.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.CanaryGuardrailSchedule
    >>> s = s.variant(label_delay=s.total / 2)
    >>> s.labeled_by_completion <= s.admitted_by_completion
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    samples_per_variant: float
    request_rate: Quantity  # requests per hour
    stages: List[Any]  # [(share, Quantity duration), ...]
    label_delay: Optional[Quantity] = None

    def stage_minimum(self, share: float) -> Quantity:
        """Hours needed to collect the sample target at the supplied traffic share.

        Equation: ``calc_sample_collection_time(samples_per_variant, request_rate,
        share).to('hour')``."""
        from ...physics.statistics import calc_sample_collection_time

        return calc_sample_collection_time(self.samples_per_variant, self.request_rate, share).to("hour")

    @property
    def total(self) -> Quantity:
        """Total rollout duration in hours, summing staged durations without overlap.

        Equation: ``sum((d for _, d in stages), Q_(0, 'hour')).to('hour')``."""
        return sum((d for _, d in self.stages), Q_(0, "hour")).to("hour")

    @property
    def first_stage_buffer(self) -> float:
        """Dimensionless first-stage duration divided by its minimum collection time.

        Equation: ``share, duration = stages[0]; (duration /
        stage_minimum(share)).to('').magnitude``."""
        share, duration = self.stages[0]
        return (duration / self.stage_minimum(share)).to("").magnitude

    @property
    def admitted_by_completion(self) -> float:
        """Candidate tasks admitted over the whole ladder.

        Equation: ``sum(((share * request_rate * d).to('').magnitude for share, d in
        stages))``."""
        return sum((share * self.request_rate * d).to("").magnitude for share, d in self.stages)

    @property
    def labeled_by_completion(self) -> float:
        """Candidate tasks whose verdicts have arrived when the last stage ends.

        Equation: Sum traffic_share * request_rate * max(0, min(stage_end, total_time -
        label_delay) - stage_start) over stages."""
        cutoff = self.total - self.label_delay
        labeled, t = 0.0, Q_(0, "hour")
        for share, d in self.stages:
            end = t + d
            overlap = max(Q_(0, "hour"), min(end, cutoff) - t)
            labeled += (share * self.request_rate * overlap).to("").magnitude
            t = end
        return labeled


class OwnVsRentScenario(BaseScenario):
    """Owning ``gpus`` accelerators against renting them at ``rent_rate`` per GPU-hour for
    ``utilization`` of every hour, over ``years``.

    Owning pays the nodes, network, and facility up front, then a running cost
    per year: a fixed ``opex_per_year`` and ``staffing_per_year``, plus
    electricity for ``power`` at ``pue`` and ``electricity_price``. Electricity
    is charged every hour unless ``energy_scales_with_use``, when it is drawn
    only while busy. A year is 8,760 hours and a month one twelfth of it, the
    book's convention.

    Notes
    -----
    Equation: Ownership is upfront + ``years*annual_operation``; rental is ``gpus*rate*annual_hours*utilization*years``.

    Assumptions: Constant prices, annual duration convention, divisible node sizing, and the declared utilization/power policy.

    Excludes: Discounting, financing, residual resale value, price escalation, and utilization uncertainty.

    References
    ----------
    Derivation: Initial purchases are paid once, whereas operating and rental charges
    repeat over the ownership horizon. Separating fixed annual costs from
    utilization-dependent electricity makes the break-even utilization algebra explicit;
    mlsysim.physics.economics.calc_ownership_breakeven_utilization and
    calc_rent_crossover perform the corresponding inversions.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.OwnVsRentH100Cluster1K
    >>> s.owned_total.to("USD").magnitude > 0
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    gpus: int
    gpus_per_node: int
    node_price: Quantity
    rent_rate: Quantity  # USD per GPU-hour
    utilization: float
    years: int
    network_capex: Quantity = Q_(0, "USD")
    facility_capex: Quantity = Q_(0, "USD")
    opex_per_year: Quantity = Q_(0, "USD")
    staffing_per_year: Quantity = Q_(0, "USD")
    power: Optional[Quantity] = None
    pue: float = 1.0
    electricity_price: Optional[Quantity] = None
    energy_scales_with_use: bool = False

    @property
    def hours_per_year(self) -> Quantity:
        """Annual hours from the package's explicit year-duration convention.

        Equation: ``Q_(HOURS_PER_YEAR, 'hour')``."""
        from ...core.units import HOURS_PER_YEAR

        return Q_(HOURS_PER_YEAR, "hour")

    @property
    def nodes(self) -> int:
        """Whole-node count by integer division; the configured GPU count should be divisible by GPUs per node.

        Equation: ``gpus // gpus_per_node``."""
        return self.gpus // self.gpus_per_node

    @property
    def hardware_capex(self) -> Quantity:
        """Accelerator-node purchase cost in USD, excluding network and facility capital.

        Equation: ``(nodes * node_price).to('USD')``."""
        return (self.nodes * self.node_price).to("USD")

    @property
    def upfront(self) -> Quantity:
        """Total initial node, network, and facility capital outlay in USD.

        Equation: ``(hardware_capex + network_capex + facility_capex).to('USD')``."""
        return (self.hardware_capex + self.network_capex + self.facility_capex).to("USD")

    @property
    def electricity_per_year_full(self) -> Quantity:
        """Electricity for one year of full use (PUE included).

        Equation: Full-use electricity cost = power * annual_hours * PUE *
        electricity_price; absent power gives zero USD."""
        if self.power is None:
            return Q_(0, "USD")
        energy = (self.power * self.hours_per_year * self.pue).to("kWh")
        return (energy * self.electricity_price).to("USD")

    @property
    def fixed_cost_per_year(self) -> Quantity:
        """Annual USD cost independent of utilization, including always-on electricity when configured.

        Equation: Fixed annual cost = opex + staffing, plus full electricity cost when
        energy does not scale with use."""
        fixed = self.opex_per_year + self.staffing_per_year
        if not self.energy_scales_with_use:
            fixed = fixed + self.electricity_per_year_full
        return fixed.to("USD")

    @property
    def use_cost_per_year_full(self) -> Quantity:
        """Annual USD electricity component that scales with utilization, or zero for always-on power.

        Equation: ``electricity_per_year_full if energy_scales_with_use else Q_(0, 'USD')``."""
        return self.electricity_per_year_full if self.energy_scales_with_use else Q_(0, "USD")

    @property
    def owned_cost_per_year(self) -> Quantity:
        """Running cost of owning per year at this utilization.

        Equation: ``(fixed_cost_per_year + utilization *
        use_cost_per_year_full).to('USD')``."""
        return (self.fixed_cost_per_year + self.utilization * self.use_cost_per_year_full).to("USD")

    @property
    def owned_cost_per_year_amortized(self) -> Quantity:
        """Up-front cost spread evenly over the years, plus the running cost.

        Equation: ``(upfront / years + owned_cost_per_year).to('USD')``."""
        return (self.upfront / self.years + self.owned_cost_per_year).to("USD")

    @property
    def rent_per_year_full(self) -> Quantity:
        """Annual USD rental cost if all configured GPUs are rented for every annual hour.

        Equation: ``(gpus * rent_rate * hours_per_year).to('USD')``."""
        return (self.gpus * self.rent_rate * self.hours_per_year).to("USD")

    @property
    def rent_per_year(self) -> Quantity:
        """Annual USD rental cost at the supplied utilization.

        Equation: ``(rent_per_year_full * utilization).to('USD')``."""
        return (self.rent_per_year_full * self.utilization).to("USD")

    @property
    def rent_per_month(self) -> Quantity:
        """Monthly rental budget in USD, one twelfth of the annual total.

        Equation: ``rent_per_year / 12``."""
        return self.rent_per_year / 12

    @property
    def owned_cost_per_month(self) -> Quantity:
        """Monthly operating budget in USD, one twelfth of the annual total; capital excluded.

        Equation: ``owned_cost_per_year / 12``."""
        return self.owned_cost_per_year / 12

    @property
    def crossover_months(self) -> Optional[float]:
        """Months until cumulative rent overtakes owning; None if it never does.

        Equation: Delegates repayment time to mlsysim.physics.economics.calc_rent_crossover;
        returns None if no positive monthly saving exists."""
        from ...physics.economics import calc_rent_crossover

        t = calc_rent_crossover(self.upfront, self.owned_cost_per_month, self.rent_per_month)
        return None if t is None else t.to("").magnitude

    @property
    def owned_total(self) -> Quantity:
        """Lifetime USD ownership cost: initial capital plus annual operation for the horizon.

        Equation: ``(upfront + years * owned_cost_per_year).to('USD')``."""
        return (self.upfront + self.years * self.owned_cost_per_year).to("USD")

    @property
    def rented_total(self) -> Quantity:
        """Lifetime USD rental cost over the same horizon and utilization.

        Equation: ``(years * rent_per_year).to('USD')``."""
        return (self.years * self.rent_per_year).to("USD")

    @property
    def savings(self) -> Quantity:
        """Lifetime USD rent minus ownership cost; negative means renting costs less.

        Equation: ``(rented_total - owned_total).to('USD')``."""
        return (self.rented_total - self.owned_total).to("USD")

    @property
    def rent_lifetime_full(self) -> Quantity:
        """Rent for every hour of the whole life.

        Equation: ``(years * rent_per_year_full).to('USD')``."""
        return (self.years * self.rent_per_year_full).to("USD")

    @property
    def lifetime_hours(self) -> Quantity:
        """Total hours in the ownership horizon.

        Equation: ``(years * hours_per_year).to('hour')``."""
        return (self.years * self.hours_per_year).to("hour")

    @property
    def breakeven_utilization(self) -> float:
        """Continuous utilization where lifetime rental and ownership costs agree.

        Equation: ``float(calc_ownership_breakeven_utilization(upfront, fixed_cost_per_year,
        rent_per_year_full, years, use_cost_per_year_full).to('').magnitude)``."""
        from ...physics.economics import calc_ownership_breakeven_utilization

        return float(
            calc_ownership_breakeven_utilization(
                self.upfront, self.fixed_cost_per_year, self.rent_per_year_full, self.years, self.use_cost_per_year_full
            )
            .to("")
            .magnitude
        )


class TrainingCampaignCostScenario(BaseScenario):
    """``runs_per_year`` training runs of ``run_low`` to ``run_high`` on an owned cluster
    (``ownership``) against renting the same GPUs only while each run lasts.

    Notes
    -----
    Equation: Ownership cost per run allocates the lifetime owned total over ``runs_per_year*years``; rental prices each run duration.

    Assumptions: The same cluster serves every run and idle ownership cost remains in the allocation.

    Excludes: Concurrent sharing, rental availability, and run-duration distributions.

    References
    ----------
    Derivation: Allocating total lifetime ownership cost evenly across scheduled runs
    includes the cost of idle capacity between runs. Renting instead multiplies GPU
    count, rental rate, and each run duration, so no idle rental time is charged in that
    comparison.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.BurstyTrainingOn1KCluster
    >>> s.owned_cost_per_run.to("USD").magnitude > 0
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    ownership: OwnVsRentScenario
    runs_per_year: int
    run_low: Quantity
    run_high: Quantity

    @property
    def runs(self) -> int:
        """Total scheduled training-run count across the ownership horizon.

        Equation: ``runs_per_year * ownership.years``."""
        return self.runs_per_year * self.ownership.years

    @property
    def owned_cost_per_run(self) -> Quantity:
        """Lifetime ownership USD allocated equally across scheduled runs.

        Equation: ``(ownership.owned_total / runs).to('USD')``."""
        return (self.ownership.owned_total / self.runs).to("USD")

    def rented_cost_per_run(self, length: Quantity) -> Quantity:
        """USD rental cost of all GPUs for the supplied run duration.

        Equation: ``o = ownership; (o.gpus * length * o.rent_rate).to('USD')``."""
        o = self.ownership
        return (o.gpus * length * o.rent_rate).to("USD")


class LifecycleTcoScenario(BaseScenario):
    """Monthly lifecycle cost in four terms (training, inference, data, iteration), each counted once.

    Notes
    -----
    Equation: Total monthly cost sums the supplied category amounts; shares divide each category by the total.

    Assumptions: Categories are mutually exclusive, exhaustive for the chosen accounting boundary, and total cost is positive.

    Excludes: Capital discounting and unlisted lifecycle activities.

    References
    ----------
    Derivation: Each monthly category is an explicitly supplied, nonoverlapping cost
    component. Adding those components gives total cost, division by total gives
    category shares, and comparison in a common currency/time unit identifies the
    largest component.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.StartupOneModelTco
    >>> abs(sum(s.shares.values()) - 1) < 1e-12
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    monthly: Dict[str, Quantity]  # term -> USD per month

    @property
    def total(self) -> Quantity:
        """Monthly USD cost summed across the declared lifecycle categories.

        Equation: ``sum(monthly.values(), Q_(0, 'USD/month')).to('USD/month')``."""
        return sum(self.monthly.values(), Q_(0, "USD/month")).to("USD/month")

    @property
    def shares(self) -> Dict[str, float]:
        """Dimensionless category fractions of the total monthly cost.

        Equation: ``t = total; {k: (v / t).to('').magnitude for k, v in monthly.items()}``."""
        t = self.total
        return {k: (v / t).to("").magnitude for k, v in self.monthly.items()}

    @property
    def dominant(self) -> str:
        """Name of the category with the largest monthly USD contribution.

        Equation: ``max(monthly, key=lambda k: monthly[k].to('USD/month').magnitude)``."""
        return max(self.monthly, key=lambda k: self.monthly[k].to("USD/month").magnitude)

    def ratios_to(self, other: "LifecycleTcoScenario") -> Dict[str, float]:
        """This budget's terms (and total) over another's.

        Equation: Divide each matching monthly category by the other scenario category; add
        total / other.total under the total key."""
        out = {k: (v / other.monthly[k]).to("").magnitude for k, v in self.monthly.items()}
        out["total"] = (self.total / other.total).to("").magnitude
        return out


class OptimizationPaybackScenario(BaseScenario):
    """An optimization costing ``implementation_cost`` that cuts a ``monthly_bill`` by ``reduction``.

    Notes
    -----
    Equation: Monthly saving is ``bill*reduction``; payback is implementation_cost/monthly_saving.

    Assumptions: A sustained positive saving and constant monthly bill.

    Excludes: Discount rates, deployment delay, maintenance cost, and performance-regression risk.

    References
    ----------
    Derivation: A fractional reduction of a constant monthly bill saves ``bill*reduction``
    each month. Dividing the one-time implementation cost by that monthly saving gives
    the undiscounted number of months needed to recover the cost.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.Int8InferencePayback
    >>> s.payback.to("month").magnitude > 0
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    implementation_cost: Quantity
    monthly_bill: Quantity  # USD per month
    reduction: float

    @property
    def monthly_savings(self) -> Quantity:
        """Monthly USD savings from applying the assumed fractional bill reduction.

        Equation: ``(monthly_bill * reduction).to('USD/month')``."""
        return (self.monthly_bill * self.reduction).to("USD/month")

    @property
    def payback(self) -> Quantity:
        """Continuous months needed for accumulated savings to repay implementation cost.

        Equation: ``(implementation_cost / monthly_savings).to('month')``."""
        return (self.implementation_cost / self.monthly_savings).to("month")


class PlatformBenefitScenario(BaseScenario):
    """A shared platform costing ``platform_cost`` per month that saves each of ``models`` model
    teams ``hours_saved`` per month at ``engineer_rate``.

    Notes
    -----
    Equation: Gross monthly value is ``models*hours_saved*engineer_rate``; net benefit subtracts platform_cost.

    Assumptions: Saved engineer time is valued at the supplied hourly rate for every model team.

    Excludes: Adoption ramp, opportunity-cost uncertainty, and non-time platform benefits.

    References
    ----------
    Derivation: Saved hours per model team multiply by model count and the assumed
    hourly value of engineering labor. Subtracting the monthly platform charge gives net
    benefit; dividing gross benefit by that charge gives a benefit-cost ratio.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.SharedPlatformBenefit50
    >>> s.gross_benefit.to("USD/month").magnitude > 0
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    models: int
    hours_saved: Quantity  # hours per model per month
    engineer_rate: Quantity  # USD per hour
    platform_cost: Quantity  # USD per month

    @property
    def gross_benefit(self) -> Quantity:
        """Monthly USD value of engineering time saved across model teams.

        Equation: ``(models * hours_saved * engineer_rate).to('USD/month')``."""
        return (self.models * self.hours_saved * self.engineer_rate).to("USD/month")

    @property
    def net_benefit(self) -> Quantity:
        """Monthly USD benefit after subtracting the shared platform budget.

        Equation: ``(gross_benefit - platform_cost).to('USD/month')``."""
        return (self.gross_benefit - self.platform_cost).to("USD/month")

    @property
    def benefit_cost_ratio(self) -> float:
        """Dimensionless saved-time value divided by platform cost.

        Equation: ``(gross_benefit / platform_cost).to('').magnitude``."""
        return (self.gross_benefit / self.platform_cost).to("").magnitude


class BatchServingCostScenario(BaseScenario):
    """One accelerator rented at ``price``, serving batches back to back: cost per request by batch size.

    Notes
    -----
    Equation: The calculation members below define ``cost_per_million``, ``cost_ratio``, ``added_time``.

    Assumptions: The accelerator is rented continuously and runs batches back to back at the supplied service curve.

    Excludes: Formation delay, idle utilization, queueing, and autoscaling costs.

    References
    ----------
    Derivation: One accelerator rented at ``price``, serving batches back to back: cost per request by batch size. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ResNet50BatchOneVersusEightCostH100
    >>> value = case.cost_ratio
    >>> print(f"{value:.3g}")
    7.51"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    service: BatchingKneeScenario
    price: Quantity  # USD per accelerator-hour
    batch_sizes: List[int]

    def cost_per_million(self, batch_size) -> Quantity:
        """USD per million requests at a given batch size, rent rate divided by saturated throughput.

        Equation: ``(price / service.throughput(batch_size) * 1000000.0).to('dollar')``."""
        return (self.price / self.service.throughput(batch_size) * 1e6).to("dollar")

    @property
    def cost_ratio(self) -> float:
        """Dimensionless first-to-last batch-size cost-per-request ratio.

        Cost per request at the first batch size over the last.

        Equation: ``(cost_per_million(first) / cost_per_million(last)).to('').magnitude``."""
        first, last = self.batch_sizes[0], self.batch_sizes[-1]
        return (self.cost_per_million(first) / self.cost_per_million(last)).to("").magnitude

    @property
    def added_time(self) -> Quantity:
        """Last-batch service time minus first-batch service time in ms.

        Service time of the last batch size beyond the first.

        Equation: ``(service.service_time(last) - service.service_time(first)).to('ms')``."""
        first, last = self.batch_sizes[0], self.batch_sizes[-1]
        return (self.service.service_time(last) - self.service.service_time(first)).to("ms")


class ServingInstanceCostScenario(BaseScenario):
    """Instances with an hourly price and a saturated request rate: cost per request, and when the
    ``small`` instance sized to the load undercuts one ``large`` instance.

    Notes
    -----
    Equation: The calculation members below define ``cost_per_million``, ``requests_per_hour``, ``price_ratio``, ``breakeven_rate``, ``small_within_large_price``, ``small_fleet_rate``.

    Assumptions: Instance rates are saturated throughput; the small fleet adds capacities linearly.

    Excludes: Queueing targets, idle periods, and economies of scale beyond supplied prices.

    References
    ----------
    Derivation: Instances with an hourly price and a saturated request rate: cost per request, and when the ``small`` instance sized to the load undercuts one ``large`` instance. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ResNet50CloudInstanceCost
    >>> value = case.breakeven_rate
    >>> print(f"{value:.3g~}")
    3.02e+03 / s"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    prices: Dict[str, Quantity]  # USD per instance-hour
    rates: Dict[str, Quantity]  # requests per second, fully busy
    small: str
    large: str

    def cost_per_million(self, name) -> Quantity:
        """USD per million requests for the named instance at its saturated request rate.

        Equation: ``(prices[name] / rates[name] * 1000000.0).to('dollar')``."""
        return (self.prices[name] / self.rates[name] * 1e6).to("dollar")

    def requests_per_hour(self, name) -> float:
        """Requests served in one hour at the named instance's saturated rate.

        Equation: ``(rates[name] * Q_(1, 'hour')).to('').magnitude``."""
        return (self.rates[name] * Q_(1, "hour")).to("").magnitude

    @property
    def price_ratio(self) -> float:
        """Large-to-small hourly price ratio, dimensionless.

        Equation: ``(prices[large] / prices[small]).to('').magnitude``."""
        return (self.prices[self.large] / self.prices[self.small]).to("").magnitude

    @property
    def breakeven_rate(self) -> Quantity:
        """Requests per second obtainable from small instances at the large instance's price, allowing fractional instances.

        Load at which the large instance's price buys the same capacity of small instances.

        Equation: ``(price_ratio * rates[small]).to('1/second')``."""
        return (self.price_ratio * self.rates[self.small]).to("1/second")

    @property
    def small_within_large_price(self) -> int:
        """Largest integer small fleet costing strictly less than one large instance.

        Most small instances that cost less than one large instance.

        Equation: ``_math.ceil(price_ratio) - 1``."""
        return _math.ceil(self.price_ratio) - 1

    @property
    def small_fleet_rate(self) -> Quantity:
        """Saturated requests per second from the integer small fleet below the large-instance price.

        Equation: ``(small_within_large_price * rates[small]).to('1/second')``."""
        return (self.small_within_large_price * self.rates[self.small]).to("1/second")


class SupervisionLaborScenario(BaseScenario):
    """The person-hours a supervised fleet creates in one ``operating_day``: an assistant staffed the
    whole day (busy at ``queue``'s utilization, watching otherwise) plus ``sweeps`` corridor sweeps.

    Notes
    -----
    Equation: Busy hours are ``utilization*operating_day``; staffed labor is the whole operating
    day plus sweep duration; shifts divide by shift length.

    Assumptions: Watch time remains staffed even without requests, and sweeps add labor
    rather than overlapping watch coverage.

    Excludes: Breaks, peak staffing reserve, and a stability check; an overloaded queue can
    make residual watch time negative.

    References
    ----------
    Derivation: Busy hours are ``utilization*operating_day``; staffed labor is the whole
    operating day plus sweep duration; shifts divide by shift length. This is declared
    scenario accounting rather than a fitted empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseRemoteAssistantLabor
    >>> value = scenario.request_time
    >>> str(value.units)
    'h'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    queue: SupervisorQueueScenario
    operating_day: Quantity
    sweeps: int
    sweep_time: Quantity
    shift: Quantity  # one full-time shift

    @property
    def request_time(self) -> Quantity:
        """Expected request-handling hours within the operating day, using the queue's utilization.

        Equation: ``(queue.result.utilization * operating_day).to('hour')``.
        """
        return (self.queue.result.utilization * self.operating_day).to("hour")

    @property
    def watch_time(self) -> Quantity:
        """Non-request watch hours left in the staffed operating day; unstable demand can make this
        negative.

        Equation: ``(operating_day - request_time).to('hour')``.
        """
        return (self.operating_day - self.request_time).to("hour")

    @property
    def sweep_total(self) -> Quantity:
        """Additional labor hours allocated to corridor sweeps.

        Equation: ``(sweeps * sweep_time).to('hour')``.
        """
        return (self.sweeps * self.sweep_time).to("hour")

    @property
    def total(self) -> Quantity:
        """Staffed hours plus sweep hours, without subtracting idle watch time.

        Equation: ``(operating_day + sweep_total).to('hour')``.
        """
        return (self.operating_day + self.sweep_total).to("hour")

    @property
    def shifts(self) -> float:
        """Total labor hours divided by hours per shift, a fractional staffing equivalent.

        Equation: ``(total / shift).to('').magnitude``.
        """
        return (self.total / self.shift).to("").magnitude


class DisclosurePenaltyScenario(BaseScenario):
    """Penalties for an incomplete account: ``days_withheld`` at ``daily_maximum`` each, plus
    ``others`` (name -> amount), all in USD.

    Notes
    -----
    Equation: The illustrative charge is ``days_withheld*daily_maximum`` plus the sum of
    supplied other charges.

    Assumptions: Monetary values are plain USD scalars supplied by the caller.

    Excludes: A statement of current law, legally applicable penalties, or adjudication of a
    particular event.

    References
    ----------
    Derivation: The illustrative charge is ``days_withheld*daily_maximum`` plus the sum of
    supplied other charges. This is declared scenario accounting rather than a fitted
    empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.CruiseDisclosurePenalties
    >>> value = scenario.per_day_penalty
    >>> round(float(value), 3)
    112500.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    days_withheld: float
    daily_maximum: float
    others: Dict[str, float]

    @property
    def per_day_penalty(self) -> float:
        """USD accumulated across withheld days at the supplied daily amount; not a legal
        determination.

        Equation: ``days_withheld * daily_maximum``.
        """
        return self.days_withheld * self.daily_maximum

    @property
    def total(self) -> float:
        """USD sum of the day-based charge and separately supplied other charges.

        Equation: ``per_day_penalty + sum(others.values())``.
        """
        return self.per_day_penalty + sum(self.others.values())


class CohortAttemptCostScenario(BaseScenario):
    """One configuration's cohort of tasks, one bounded attempt each, all charges settled.

    Notes
    -----
    Equation: The result property delegates to the named solver documented below; it
    converts declared scenario units before passing the inputs.

    Assumptions: One bounded attempt per task; every billed attempt is settled before cohort
    cost is compared.

    Excludes: Repeated retries beyond turns, outstanding charges, and observed acceptance
    inference.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.solvers.CohortAttemptCostModel``. Their
    contracts supply the underlying equations; property equations show the remaining unit
    conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.FrontierModelRepairCohort
    >>> value = scenario.result
    >>> type(value).__name__
    'CohortAttemptCostResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    tasks: int
    accepted: int
    input_price: Quantity  # USD per Mtoken
    output_price: Quantity
    turns: int  # per attempt
    input_tokens_per_turn: int
    output_tokens_per_turn: int
    environment_per_turn: Quantity  # USD: tools, sandbox holding, cleanup
    assessment_per_attempt: Quantity  # USD: checks not billed as sandbox time
    triage_per_rejected: Quantity  # USD: engineer triage of each rejected attempt

    @property
    def result(self):
        """Return the typed accounting result from ``CohortAttemptCostModel`` using this scenario's
        declared inputs; units and exclusions follow that solver contract.

        Equation: ``CohortAttemptCostModel().solve(tasks, accepted, input_price, output_price,
        turns, input_tokens_per_turn, output_tokens_per_turn, environment_per_turn,
        assessment_per_attempt, triage_per_rejected)``.
        """
        from ..solvers import CohortAttemptCostModel

        return CohortAttemptCostModel().solve(
            self.tasks,
            self.accepted,
            self.input_price,
            self.output_price,
            self.turns,
            self.input_tokens_per_turn,
            self.output_tokens_per_turn,
            self.environment_per_turn,
            self.assessment_per_attempt,
            self.triage_per_rejected,
        )


class TokenPriceComparisonScenario(BaseScenario):
    """Two cohort configurations under one acceptance contract: does the cheaper token win?

    Notes
    -----
    Equation: The result property delegates to the named solver documented below; it
    converts declared scenario units before passing the inputs.

    Assumptions: Both nested cohorts use a comparable acceptance contract and include all
    billed attempts.

    Excludes: A causal claim that token price alone determines acceptance or cost.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.solvers.CohortAttemptCostModel``. Their
    contracts supply the underlying equations; property equations show the remaining unit
    conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RepairCohortTokenPriceComparison
    >>> value = scenario.result
    >>> type(value).__name__
    'TokenPriceComparisonResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    baseline: CohortAttemptCostScenario
    alternative: CohortAttemptCostScenario

    @property
    def result(self):
        """Return the typed accounting result from ``CohortAttemptCostModel`` using this scenario's
        declared inputs; units and exclusions follow that solver contract.

        Equation: ``CohortAttemptCostModel.compare(baseline.result, alternative.result,
        baseline.accepted, alternative.accepted)``.
        """
        from ..solvers import CohortAttemptCostModel

        return CohortAttemptCostModel.compare(
            self.baseline.result, self.alternative.result, self.baseline.accepted, self.alternative.accepted
        )


class ResendCacheBillScenario(BaseScenario):
    """An append-only trajectory (``TrajectoryResendScenario``) billed under one tariff, with and without reuse.

    Notes
    -----
    Equation: The result property delegates to the named solver documented below; it
    converts declared scenario units before passing the inputs.

    Assumptions: The prompt grows append-only and the supplied tariff and miss fraction
    apply to every turn.

    Excludes: Provider-specific cache eligibility rules, expiration, and tokenization
    differences.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.solvers.PrefixCacheBillModel``. Their
    contracts supply the underlying equations; property equations show the remaining unit
    conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RepairResendCacheBill
    >>> value = scenario.result
    >>> type(value).__name__
    'PrefixCacheBillResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    trajectory: TrajectoryResendScenario
    input_price: Quantity  # USD per Mtoken, fresh input
    cached_fraction: float  # cached input billed at this share of the input price
    output_price: Quantity
    miss_fraction: float  # share of the eligible prefix that misses, for sensitivity

    @property
    def result(self):
        """Return the typed accounting result from ``PrefixCacheBillModel`` using this scenario's
        declared inputs; units and exclusions follow that solver contract.

        Equation: ``PrefixCacheBillModel().solve(t.turns, t.first_prompt, t.appended,
        t.output_per_turn, input_price, cached_fraction, output_price, miss_fraction)``.
        """
        from ..solvers import PrefixCacheBillModel

        t = self.trajectory
        return PrefixCacheBillModel().solve(
            t.turns,
            t.first_prompt,
            t.appended,
            t.output_per_turn,
            self.input_price,
            self.cached_fraction,
            self.output_price,
            self.miss_fraction,
        )


class CompactionPaybackScenario(BaseScenario):
    """One compaction of a billed trajectory: a summary resent fresh, then shorter cached reads.

    Notes
    -----
    Equation: The result property delegates to the named solver documented below; it
    converts declared scenario units before passing the inputs.

    Assumptions: One summary replaces the prefix at the supplied turn and the declared
    summary output cost is charged.

    Excludes: Quality loss from compaction, repeated compactions, and provider cache
    implementation details.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.solvers.CompactionPaybackModel``. Their
    contracts supply the underlying equations; property equations show the remaining unit
    conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RepairCompactionPayback
    >>> value = scenario.result
    >>> type(value).__name__
    'CompactionPaybackResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    bill: ResendCacheBillScenario
    compact_turn: int  # the call that compacts
    compact_to: int  # tokens of the compacted context
    summary_tokens: int  # output tokens the summary costs

    @property
    def result(self):
        """Return the typed accounting result from ``CompactionPaybackModel`` using this scenario's
        declared inputs; units and exclusions follow that solver contract.

        Equation: ``CompactionPaybackModel().solve(t.turns, t.first_prompt, t.appended,
        b.input_price, b.cached_fraction, b.output_price, compact_turn, compact_to,
        summary_tokens)``.
        """
        from ..solvers import CompactionPaybackModel

        b, t = self.bill, self.bill.trajectory
        return CompactionPaybackModel().solve(
            t.turns,
            t.first_prompt,
            t.appended,
            b.input_price,
            b.cached_fraction,
            b.output_price,
            self.compact_turn,
            self.compact_to,
            self.summary_tokens,
        )


class RetryPolicyScenario(BaseScenario):
    """A cheap attempt retried up to a cap, run as a batch, or one stronger attempt.

    Notes
    -----
    Equation: The result property delegates to the named solver documented below; it
    converts declared scenario units before passing the inputs.

    Assumptions: Attempt successes are independent with fixed probability and fixed
    per-attempt charges.

    Excludes: Correlated failure causes, changing retry quality, and latency scheduling.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.solvers.RetryPolicyModel``. Their contracts
    supply the underlying equations; property equations show the remaining unit conversions
    and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.CheapRetryAgainstStrongCall
    >>> value = scenario.result
    >>> type(value).__name__
    'RetryPolicyResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    pass_probability: float
    attempt_cost: Quantity  # USD, all-in
    max_attempts: int
    strong_pass_probability: float
    strong_attempt_cost: Quantity

    @property
    def result(self):
        """Return the typed accounting result from ``RetryPolicyModel`` using this scenario's
        declared inputs; units and exclusions follow that solver contract.

        Equation: ``RetryPolicyModel().solve(pass_probability, attempt_cost, max_attempts,
        strong_pass_probability, strong_attempt_cost)``.
        """
        from ..solvers import RetryPolicyModel

        return RetryPolicyModel().solve(
            self.pass_probability,
            self.attempt_cost,
            self.max_attempts,
            self.strong_pass_probability,
            self.strong_attempt_cost,
        )


class SerialEpisodeScenario(BaseScenario):
    """One serial episode as disjoint intervals; the model's intervals sped up by ``model_speedup``.

    Notes
    -----
    Equation: The result property delegates to the named solver documented below; it
    converts declared scenario units before passing the inputs.

    Assumptions: Model, test, lifecycle, and runtime intervals are disjoint serial time
    contributions.

    Excludes: Overlap, asynchronous work, and changes to non-model intervals under the
    speedup.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.solvers.SerialEpisodeModel``. Their
    contracts supply the underlying equations; property equations show the remaining unit
    conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.SerialRepairEpisode
    >>> value = scenario.result
    >>> type(value).__name__
    'SerialEpisodeResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    model_intervals: List[Quantity]
    tests: Quantity
    lifecycle_intervals: List[Quantity]  # sandbox creation and snapshot export
    runtime_intervals: List[Quantity]  # runtime setup and status
    model_speedup: float

    @property
    def result(self):
        """Return the typed accounting result from ``SerialEpisodeModel`` using this scenario's
        declared inputs; units and exclusions follow that solver contract.

        Equation: ``SerialEpisodeModel().solve(secs(model_intervals),
        float(tests.to('second').magnitude), secs(lifecycle_intervals), secs(runtime_intervals),
        model_speedup); otherwise tuple((float(x.to('second').magnitude) for x in xs))``.
        """
        from ..solvers import SerialEpisodeModel

        def secs(xs):
            return tuple(float(x.to("second").magnitude) for x in xs)

        return SerialEpisodeModel().solve(
            secs(self.model_intervals),
            float(self.tests.to("second").magnitude),
            secs(self.lifecycle_intervals),
            secs(self.runtime_intervals),
            self.model_speedup,
        )


class SpeculativeEpisodeScenario(BaseScenario):
    """Speculative decoding on a serial episode's model time, at two draft acceptance levels.

    Notes
    -----
    Equation: The result property delegates to the named solver documented below; it
    converts declared scenario units before passing the inputs.

    Assumptions: The nested serial episode supplies model fraction; draft length, cost
    ratio, and acceptance are fixed.

    Excludes: A prediction of acceptance quality, variable sequence length, and non-model
    speedup.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.solvers.SpeculativeEpisodeModel``. Their
    contracts supply the underlying equations; property equations show the remaining unit
    conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.SerialRepairEpisodeSpeculation
    >>> value = scenario.result
    >>> type(value).__name__
    'SpeculativeEpisodeResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    episode: SerialEpisodeScenario
    draft_length: int  # gamma, tokens drafted per round
    draft_cost_ratio: float  # r, a draft token's cost as a share of a target step
    accepted_high: float  # m, drafted tokens kept per round when drafts align
    accepted_low: float

    @property
    def result(self):
        """Return the typed accounting result from ``SpeculativeEpisodeModel`` using this
        scenario's declared inputs; units and exclusions follow that solver contract.

        Equation: ``SpeculativeEpisodeModel().solve(float(e.total_time.to('second').magnitude),
        e.model_fraction, draft_length, draft_cost_ratio, accepted_high, accepted_low)``.
        """
        from ..solvers import SpeculativeEpisodeModel

        e = self.episode.result
        return SpeculativeEpisodeModel().solve(
            float(e.total_time.to("second").magnitude),
            e.model_fraction,
            self.draft_length,
            self.draft_cost_ratio,
            self.accepted_high,
            self.accepted_low,
        )


class ModelCascadeScenario(BaseScenario):
    """A tiered cascade over a completed cohort, every exit under one gate, against the top tier alone.

    Notes
    -----
    Equation: The result property delegates to the named solver documented below; it
    converts declared scenario units before passing the inputs.

    Assumptions: Tier exits use one acceptance gate and supplied cohort counts; each reached
    tier incurs its call and gate work.

    Excludes: Predicting acceptance counts or harm probability from the model architecture.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.solvers.ModelCascadeModel``. Their contracts
    supply the underlying equations; property equations show the remaining unit conversions
    and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.CodeEditThreeTierCascade
    >>> value = scenario.result
    >>> type(value).__name__
    'ModelCascadeResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    tasks: int
    call_costs: List[Quantity]  # USD per reached tier, gate excluded
    gate_cost: Quantity
    call_times: List[Quantity]
    gate_time: Quantity
    accepted_by_tier: List[int]
    direct_accepted: int  # the top tier alone over the whole cohort
    false_accept_rate: Optional[float] = None  # at the first tier, for the harm case
    harm_per_false_accept: Optional[Quantity] = None

    @property
    def result(self):
        """Return the typed accounting result from ``ModelCascadeModel`` using this scenario's
        declared inputs; units and exclusions follow that solver contract.

        Equation: ``ModelCascadeModel().solve(tasks, [float(c.to('USD').magnitude) for c in
        call_costs], float(gate_cost.to('USD').magnitude), [float(t.to('second').magnitude) for
        t in call_times], float(gate_time.to('second').magnitude), accepted_by_tier,
        direct_accepted, false_accept_rate, harm)``.
        """
        from ..solvers import ModelCascadeModel

        harm = None if self.harm_per_false_accept is None else float(self.harm_per_false_accept.to("USD").magnitude)
        return ModelCascadeModel().solve(
            self.tasks,
            [float(c.to("USD").magnitude) for c in self.call_costs],
            float(self.gate_cost.to("USD").magnitude),
            [float(t.to("second").magnitude) for t in self.call_times],
            float(self.gate_time.to("second").magnitude),
            self.accepted_by_tier,
            self.direct_accepted,
            self.false_accept_rate,
            harm,
        )


class TrajectoryAdmissionScenario(BaseScenario):
    """A stationary admitted population of trajectories against a token-per-minute quota.

    Notes
    -----
    Equation: The result property delegates to the named solver documented below; it
    converts declared scenario units before passing the inputs.

    Assumptions: Arrivals and residence are stationary; each trajectory carries fixed call
    and token counts.

    Excludes: Burst admission, heterogeneous trajectories, and quota enforcement
    implementation.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.solvers.TrajectoryAdmissionModel``. Their
    contracts supply the underlying equations; property equations show the remaining unit
    conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RepairTrajectoryQuotaAdmission
    >>> value = scenario.result
    >>> type(value).__name__
    'TrajectoryAdmissionResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    arrival: Quantity  # admissions per unit time
    residence: Quantity  # time a trajectory is present
    calls: int  # model calls per trajectory
    tokens_per_call: int  # quota-counted tokens per call
    quota: Quantity  # tokens per unit time
    acceptance_drop: Optional[float] = None  # relative acceptance drop at fixed goodput

    @property
    def result(self):
        """Return the typed accounting result from ``TrajectoryAdmissionModel`` using this
        scenario's declared inputs; units and exclusions follow that solver contract.

        Equation: ``TrajectoryAdmissionModel().solve(arrival, residence, calls, tokens_per_call,
        quota, acceptance_drop)``.
        """
        from ..solvers import TrajectoryAdmissionModel

        return TrajectoryAdmissionModel().solve(
            self.arrival, self.residence, self.calls, self.tokens_per_call, self.quota, self.acceptance_drop
        )


class KingmanQueueScenario(BaseScenario):
    """One server under Kingman's approximation at two utilizations, with a holding price.

    Notes
    -----
    Equation: The result property delegates to the named solver documented below; it
    converts declared scenario units before passing the inputs.

    Assumptions: The queue has one server with stable utilization and supplied
    arrival/service squared coefficients of variation.

    Excludes: Tail guarantees, transient overload, and exact waiting-time distributions.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.physics.serving.calc_kingman_wait_ratio``,
    ``mlsysim.solvers.KingmanQueueModel``. Their contracts supply the underlying equations;
    property equations show the remaining unit conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.BurstyModelServerQueuePrice
    >>> value = scenario.result
    >>> type(value).__name__
    'KingmanQueueResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    service: Quantity  # mean service time
    arrival_scv: float  # c_a^2
    service_scv: float  # c_s^2
    low_utilization: float
    high_utilization: float
    holding_rate: Quantity  # USD per second of held resources

    def wait_at(self, utilization: float) -> Quantity:
        """Mean queue wait at stable ``utilization`` from Kingman's variability factor times mean
        service duration.

        Equation: ``calc_kingman_wait_ratio(utilization, arrival_scv, service_scv) * service``.
        """
        from ...physics.serving import calc_kingman_wait_ratio

        return calc_kingman_wait_ratio(utilization, self.arrival_scv, self.service_scv) * self.service

    @property
    def result(self):
        """Return the typed accounting result from ``KingmanQueueModel`` using this scenario's
        declared inputs; units and exclusions follow that solver contract.

        Equation: ``KingmanQueueModel().solve(service, arrival_scv, service_scv,
        low_utilization, high_utilization, float(holding_rate.to('USD / second').magnitude))``.
        """
        from ..solvers import KingmanQueueModel

        return KingmanQueueModel().solve(
            self.service,
            self.arrival_scv,
            self.service_scv,
            self.low_utilization,
            self.high_utilization,
            float(self.holding_rate.to("USD / second").magnitude),
        )


class DedicatedBreakEvenScenario(BaseScenario):
    """Renting a matched-capability API against owning capacity at a fixed hourly cost.

    Notes
    -----
    Equation: The result property delegates to the named solver documented below; it
    converts declared scenario units before passing the inputs.

    Assumptions: Owned and API service have matched capability, fixed tariffs, and declared
    throughput/load.

    Excludes: Quality differences, capacity ramps, capital discounting, and outage reserve.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.solvers.DedicatedBreakEvenModel``. Their
    contracts supply the underlying equations; property equations show the remaining unit
    conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.DedicatedCapacityBreakEven
    >>> value = scenario.result
    >>> type(value).__name__
    'DedicatedBreakEvenResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    input_per_output: float  # input tokens per output token
    cached_fraction: float  # share of input billed at the cached price
    input_price: Quantity  # USD per Mtoken
    cached_price: Quantity
    output_price: Quantity
    fixed_per_hour: Quantity  # USD per hour, owned capacity
    variable_price: Quantity  # USD per Mtoken output, owned capacity
    reference_rate: Quantity  # output tokens per second at full load
    load: float  # operating load, a fraction of the reference rate

    def _p(self, q):
        return float(q.to("USD / Mtoken").magnitude)

    def hourly_costs(self, load: float):
        """API and owned-capacity USD per hour at the supplied load, using
        ``DedicatedBreakEvenModel.hourly_costs``.

        Equation: ``DedicatedBreakEvenModel.hourly_costs(load, reference_rate, r.api_price,
        float(fixed_per_hour.to('USD / hour').magnitude), _p(variable_price))``.
        """
        from ..solvers import DedicatedBreakEvenModel

        r = self.result
        return DedicatedBreakEvenModel.hourly_costs(
            load,
            self.reference_rate,
            r.api_price,
            float(self.fixed_per_hour.to("USD / hour").magnitude),
            self._p(self.variable_price),
        )

    @property
    def result(self):
        """Return the typed accounting result from ``DedicatedBreakEvenModel`` using this
        scenario's declared inputs; units and exclusions follow that solver contract.

        Equation: ``DedicatedBreakEvenModel().solve(input_per_output, cached_fraction,
        _p(input_price), _p(cached_price), _p(output_price), float(fixed_per_hour.to('USD /
        hour').magnitude), _p(variable_price), reference_rate, load)``.
        """
        from ..solvers import DedicatedBreakEvenModel

        return DedicatedBreakEvenModel().solve(
            self.input_per_output,
            self.cached_fraction,
            self._p(self.input_price),
            self._p(self.cached_price),
            self._p(self.output_price),
            float(self.fixed_per_hour.to("USD / hour").magnitude),
            self._p(self.variable_price),
            self.reference_rate,
            self.load,
        )


class ConservedBudgetScenario(BaseScenario):
    """A root budget granted to two children; the second stops with a remote job billed late.

    Notes
    -----
    Equation: The result property delegates to the named solver documented below; it
    converts declared scenario units before passing the inputs.

    Assumptions: Money is converted to integer milli-USD; parent and child accounts follow
    the solver settlement convention.

    Excludes: Currency exchange, sub-milli-USD precision, and guarantees about an external
    billing service.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.solvers.ConservedBudgetModel``. Their
    contracts supply the underlying equations; property equations show the remaining unit
    conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RepairBudgetLateBill
    >>> value = scenario.result
    >>> type(value).__name__
    'ConservedBudgetResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    budget: Quantity  # USD
    planning: Quantity  # settled before delegation
    allocations: List[Quantity]  # granted to the first and second child
    settled_children: List[Quantity]  # each child's settled charge at its close or stop
    pending: Quantity  # enforceable bound on the stopped child's remote job
    late_charge: Quantity  # its final bill, received after the local stop
    deliveries: int = 2  # times the final bill is delivered (settled once)

    @property
    def result(self):
        """Return the typed accounting result from ``ConservedBudgetModel`` using this scenario's
        declared inputs; units and exclusions follow that solver contract.

        Equation: ``ConservedBudgetModel().solve(m(budget), m(planning), [m(x) for x in
        allocations], [m(x) for x in settled_children], m(pending), m(late_charge), deliveries);
        otherwise int(round(float(q.to('USD').magnitude) * 1000))``.
        """
        from ..solvers import ConservedBudgetModel

        def m(q):
            return int(round(float(q.to("USD").magnitude) * 1000))

        return ConservedBudgetModel().solve(
            m(self.budget),
            m(self.planning),
            [m(x) for x in self.allocations],
            [m(x) for x in self.settled_children],
            m(self.pending),
            m(self.late_charge),
            self.deliveries,
        )


class ArchitectureChoiceScenario(BaseScenario):
    """Single-call designs and a bounded loop under one check, with a human fallback for rejects.

    Notes
    -----
    Equation: The result property delegates to the named solver documented below; it
    converts declared scenario units before passing the inputs.

    Assumptions: Acceptance and human fallback resolution probabilities are supplied and
    costs use one acceptance contract.

    Excludes: An empirical quality estimate or automatic enforcement of the stored
    resolution_floor field.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.solvers.ArchitectureChoiceModel``. Their
    contracts supply the underlying equations; property equations show the remaining unit
    conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.DependencyUpgradeArchitectures
    >>> value = scenario.result
    >>> type(value).__name__
    'ArchitectureChoiceResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    tickets: int  # per month
    single_costs: Dict[str, Quantity]  # automated charge per ticket, check excluded
    single_acceptance: Dict[str, float]
    loop_name: str
    turn_cost: Quantity  # per loop turn, check excluded
    turn_acceptance: List[float]  # conditional acceptance of each turn reached
    check_cost: Quantity  # the same acceptance check for every candidate
    human_cost: Quantity  # one fallback, its final check included
    human_resolution: float  # conditional on entering fallback
    resolution_floor: float  # declared combined-policy requirement
    point: float = 0.01  # one point of automated acceptance

    @property
    def result(self):
        """Return the typed accounting result from ``ArchitectureChoiceModel`` using this
        scenario's declared inputs; units and exclusions follow that solver contract.

        Equation: ``ArchitectureChoiceModel().solve(tickets, single_costs, single_acceptance,
        loop_name, turn_cost, turn_acceptance, check_cost, human_cost, human_resolution,
        point)``.
        """
        from ..solvers import ArchitectureChoiceModel

        return ArchitectureChoiceModel().solve(
            self.tickets,
            self.single_costs,
            self.single_acceptance,
            self.loop_name,
            self.turn_cost,
            self.turn_acceptance,
            self.check_cost,
            self.human_cost,
            self.human_resolution,
            self.point,
        )


class AdaptationPaybackScenario(BaseScenario):
    """A one-time adaptation that cuts each task's model usage, against its incremental cost.

    Notes
    -----
    Equation: The result property delegates to the named solver documented below; it
    converts declared scenario units before passing the inputs.

    Assumptions: Base/adapted usage, one-time adaptation costs, and monthly task volumes are
    supplied.

    Excludes: Forecasting quality changes, demand growth, and ongoing adaptation
    maintenance.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.solvers.AdaptationPaybackModel``. Their
    contracts supply the underlying equations; property equations show the remaining unit
    conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.CodeReviewAdaptationPayback
    >>> value = scenario.result
    >>> type(value).__name__
    'AdaptationPaybackResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    base: Dict[str, Any]  # input_tokens, output_tokens, steps, input_price, output_price
    adapted: Dict[str, Any]
    data_cost: Quantity
    training_cost: Quantity
    evaluation_cost: Quantity
    monthly_volume: int
    low_volume: int  # sensitivity case

    @property
    def result(self):
        """Return the typed accounting result from ``AdaptationPaybackModel`` using this scenario's
        declared inputs; units and exclusions follow that solver contract.

        Equation: ``AdaptationPaybackModel().solve(base, adapted, data_cost, training_cost,
        evaluation_cost, monthly_volume, low_volume)``.
        """
        from ..solvers import AdaptationPaybackModel

        return AdaptationPaybackModel().solve(
            self.base,
            self.adapted,
            self.data_cost,
            self.training_cost,
            self.evaluation_cost,
            self.monthly_volume,
            self.low_volume,
        )


class WorkloadLeverScenario(BaseScenario):
    """Attempt cost by term for several workloads, and three levers on cost per accepted task.

    Each workload gives ``acceptance``, ``environment``, ``assessment`` in
    dollars, and either ``human`` (fixed per attempt) or ``handoff`` (charged
    per rejected attempt). The model term is ``model`` dollars, or the
    ``bill``'s cached trajectory cost, or ``token_multiple`` chats of
    ``chat_input``/``chat_output`` tokens at the stated tariff.

    Notes
    -----
    Equation: The result property delegates to the named solver documented below; it
    converts declared scenario units before passing the inputs.

    Assumptions: Input workloads declare acceptance and additive USD terms; cached bills and
    token multiples expand under one tariff.

    Excludes: Interactions between workload levers or empirical prediction of acceptance
    improvement.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.solvers.WorkloadLeverModel``. Their
    contracts supply the underlying equations; property equations show the remaining unit
    conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.LighthouseWorkloadLevers
    >>> value = scenario.result
    >>> type(value).__name__
    'WorkloadLeverResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workloads: Dict[str, Dict[str, Any]]
    input_price: Quantity  # USD per Mtoken, for workloads priced in chats
    output_price: Quantity
    acceptance_gain: float  # absolute gain for the acceptance lever

    def terms(self) -> Dict[str, Dict[str, float]]:
        """Normalize workload billing into USD-per-attempt terms, expanding cached bills, token
        multiples, and per-turn environment costs.

        Equation: ``out``.
        """
        p_in = self.input_price.to("USD / Mtoken").magnitude * _usd_unit()
        p_out = self.output_price.to("USD / Mtoken").magnitude * _usd_unit()
        out = {}
        for name, w in self.workloads.items():
            t = dict(w)
            if "bill" in t:
                t["model"] = float(t.pop("bill").result.total_cached.to("USD").magnitude)
            elif "token_multiple" in t:
                chat = ((t.pop("chat_input") * p_in + t.pop("chat_output") * p_out) / 1_000_000).to("USD").magnitude
                t["model"] = t.pop("token_multiple") * chat
            if "environment_per_turn" in t:
                t["environment"] = t.pop("turns") * t.pop("environment_per_turn")
            out[name] = t
        return out

    @property
    def result(self):
        """Return the typed accounting result from ``WorkloadLeverModel`` using this scenario's
        declared inputs; units and exclusions follow that solver contract.

        Equation: ``WorkloadLeverModel().solve(terms(), acceptance_gain)``.
        """
        from ..solvers import WorkloadLeverModel

        return WorkloadLeverModel().solve(self.terms(), self.acceptance_gain)


def _usd_unit():
    from ...core.units import USD

    return USD


class ServiceLifecycleCostScenario(BaseScenario):
    """A service's multi-year cost in three terms: repeated training cycles, recurring inference,
    and operations, with a GPU-hour carbon factor.

    Inference: ``daily_users`` x ``items_per_user`` requests a day, each ``service_time`` of
    dedicated accelerator time (unbatched, not overlapped), billed at ``inference_rate``. A
    training cycle is data preparation and a final run in GPU-hours at ``training_rate`` plus a
    search of ``search_experiments`` at ``search_cost_per_experiment``; ``cycles`` of them over
    ``years``. ``serving_cut`` and ``training_cut`` are fractional savings to compare.

    Notes
    -----
    Equation: Lifecycle cost adds repeated preparation/search/training cycles, dedicated
    inference device-hours, and recurring operations; carbon multiplies device-hours by a
    supplied coefficient.

    Assumptions: Requests use serial unbatched dedicated service time; tariffs, volumes,
    annual operations, and carbon coefficients stay fixed.

    Excludes: Discounting, hardware purchase accounting, dynamic batching, shared capacity,
    and a measured power/emissions trace.

    References
    ----------
    Derivation: Lifecycle cost adds repeated preparation/search/training cycles, dedicated
    inference device-hours, and recurring operations; carbon multiplies device-hours by a
    supplied coefficient. This is declared scenario accounting rather than a fitted
    empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RankingServiceLifecycleCost
    >>> value = scenario.inferences_per_day
    >>> round(float(value), 3)
    200000000.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    daily_users: float
    items_per_user: float
    service_time: Quantity
    inference_rate: Quantity  # USD per GPU-hour
    training_rate: Quantity  # USD per GPU-hour
    data_prep_time: Quantity  # GPU-hours per cycle
    search_experiments: int
    search_cost_per_experiment: Quantity
    final_run_time: Quantity  # GPU-hours per cycle
    cycles: int
    years: int
    operations_annual: Dict[str, Quantity]  # USD per year by item
    carbon_per_gpu_hour: Quantity  # kg per GPU-hour
    serving_cut: float = 0.0
    training_cut: float = 0.0

    @property
    def inferences_per_day(self) -> float:
        """Daily request count from users times items requested per user.

        Equation: ``daily_users * items_per_user``.
        """
        return self.daily_users * self.items_per_user

    @property
    def gpu_hours_per_day(self) -> Quantity:
        """Dedicated accelerator hours per day, request count times serial service duration.

        Equation: ``(inferences_per_day * service_time).to('hour')``.
        """
        return (self.inferences_per_day * self.service_time).to("hour")

    @property
    def gpus_busy(self) -> float:
        """Average concurrently busy accelerators over a full day, not a peak-capacity requirement.

        Equation: ``gpu_hours_per_day.to('hour').magnitude / HOURS_PER_DAY``.
        """
        from ...core.units import HOURS_PER_DAY

        return self.gpu_hours_per_day.to("hour").magnitude / HOURS_PER_DAY

    @property
    def inference_cost_per_day(self) -> Quantity:
        """USD billed per day for dedicated inference device-hours.

        Equation: ``(gpu_hours_per_day * inference_rate).to('dollar')``.
        """
        return (self.gpu_hours_per_day * self.inference_rate).to("dollar")

    @property
    def inference_cost_per_year(self) -> Quantity:
        """USD inference cost over the registered number of days per year.

        Equation: ``(inference_cost_per_day * DAYS_PER_YEAR).to('dollar')``.
        """
        from ...core.units import DAYS_PER_YEAR

        return (self.inference_cost_per_day * DAYS_PER_YEAR).to("dollar")

    @property
    def data_prep_cost(self) -> Quantity:
        """USD for one cycle's preparation device-hours at the training tariff.

        Equation: ``(data_prep_time * training_rate).to('dollar')``.
        """
        return (self.data_prep_time * self.training_rate).to("dollar")

    @property
    def search_cost(self) -> Quantity:
        """USD for all candidate search experiments within one training cycle.

        Equation: ``(search_experiments * search_cost_per_experiment).to('dollar')``.
        """
        return (self.search_experiments * self.search_cost_per_experiment).to("dollar")

    @property
    def search_time(self) -> Quantity:
        """Device-hours purchased by the search budget at the training tariff.

        Equation: ``(search_cost / training_rate).to('hour')``.
        """
        return (self.search_cost / self.training_rate).to("hour")

    @property
    def final_run_cost(self) -> Quantity:
        """USD for one final training run at the training tariff.

        Equation: ``(final_run_time * training_rate).to('dollar')``.
        """
        return (self.final_run_time * self.training_rate).to("dollar")

    @property
    def cycle_cost(self) -> Quantity:
        """USD per training cycle, summing preparation, search, and final training.

        Equation: ``(data_prep_cost + search_cost + final_run_cost).to('dollar')``.
        """
        return (self.data_prep_cost + self.search_cost + self.final_run_cost).to("dollar")

    @property
    def cycle_gpu_hours(self) -> Quantity:
        """Device-hours per training cycle across preparation, search, and final training.

        Equation: ``(data_prep_time + search_time + final_run_time).to('hour')``.
        """
        return (self.data_prep_time + self.search_time + self.final_run_time).to("hour")

    @property
    def training_cost(self) -> Quantity:
        """USD for every declared training cycle over the service lifetime.

        Equation: ``(cycle_cost * cycles).to('dollar')``.
        """
        return (self.cycle_cost * self.cycles).to("dollar")

    @property
    def service_days(self) -> int:
        """Service lifetime in days, using the registered days-per-year convention.

        Equation: ``DAYS_PER_YEAR * years``.
        """
        from ...core.units import DAYS_PER_YEAR

        return DAYS_PER_YEAR * self.years

    @property
    def inference_gpu_hours(self) -> Quantity:
        """Dedicated inference device-hours over the entire service lifetime.

        Equation: ``(gpu_hours_per_day * service_days).to('hour')``.
        """
        return (self.gpu_hours_per_day * self.service_days).to("hour")

    @property
    def inference_cost(self) -> Quantity:
        """USD inference cost over the entire service lifetime.

        Equation: ``(inference_cost_per_day * service_days).to('dollar')``.
        """
        return (self.inference_cost_per_day * self.service_days).to("dollar")

    @property
    def operations_per_year(self) -> Quantity:
        """USD sum of all declared annual operations charges.

        Equation: ``sum(operations_annual.values(), Q_(0, 'dollar')).to('dollar')``.
        """
        return sum(self.operations_annual.values(), Q_(0, "dollar")).to("dollar")

    @property
    def operations_cost(self) -> Quantity:
        """USD operations cost across the declared service years, without discounting.

        Equation: ``(operations_per_year * years).to('dollar')``.
        """
        return (self.operations_per_year * self.years).to("dollar")

    @property
    def total_cost(self) -> Quantity:
        """USD lifetime sum of training, inference, and operations.

        Equation: ``(training_cost + inference_cost + operations_cost).to('dollar')``.
        """
        return (self.training_cost + self.inference_cost + self.operations_cost).to("dollar")

    @property
    def shares(self) -> Dict[str, float]:
        """Dimensionless shares of lifetime USD cost assigned to training, inference, and
        operations.

        Equation: ``{'training': (training_cost / t).to('').magnitude, 'inference':
        (inference_cost / t).to('').magnitude, 'operations': (operations_cost /
        t).to('').magnitude}``.
        """
        t = self.total_cost
        return {
            "training": (self.training_cost / t).to("").magnitude,
            "inference": (self.inference_cost / t).to("").magnitude,
            "operations": (self.operations_cost / t).to("").magnitude,
        }

    @property
    def cost_ratio(self) -> float:
        """Inference USD divided by nonzero training USD over the lifetime.

        Equation: ``(inference_cost / training_cost).to('').magnitude``.
        """
        return (self.inference_cost / self.training_cost).to("").magnitude

    @property
    def training_carbon(self) -> Quantity:
        """Kilograms attributed to training device-hours using the supplied kg-per-device-hour
        coefficient.

        Equation: ``(cycle_gpu_hours.to('hour').magnitude * cycles *
        carbon_per_gpu_hour).to('kg')``.
        """
        return (self.cycle_gpu_hours.to("hour").magnitude * self.cycles * self.carbon_per_gpu_hour).to("kg")

    @property
    def inference_carbon(self) -> Quantity:
        """Kilograms attributed to inference device-hours using the same supplied coefficient.

        Equation: ``(inference_gpu_hours.to('hour').magnitude * carbon_per_gpu_hour).to('kg')``.
        """
        return (self.inference_gpu_hours.to("hour").magnitude * self.carbon_per_gpu_hour).to("kg")

    @property
    def carbon_ratio(self) -> float:
        """Inference kilograms divided by nonzero training kilograms, dimensionless.

        Equation: ``(inference_carbon / training_carbon).to('').magnitude``.
        """
        return (self.inference_carbon / self.training_carbon).to("").magnitude

    @property
    def serving_saving(self) -> Quantity:
        """USD saved by the declared fractional inference-cost reduction.

        Equation: ``(inference_cost * serving_cut).to('dollar')``.
        """
        return (self.inference_cost * self.serving_cut).to("dollar")

    @property
    def serving_saving_carbon(self) -> Quantity:
        """Kilograms avoided under the same fractional inference-work reduction.

        Equation: ``(inference_carbon * serving_cut).to('kg')``.
        """
        return (self.inference_carbon * self.serving_cut).to("kg")

    @property
    def training_saving(self) -> Quantity:
        """USD saved by the declared fractional training-cost reduction.

        Equation: ``(training_cost * training_cut).to('dollar')``.
        """
        return (self.training_cost * self.training_cut).to("dollar")

    @property
    def training_saving_carbon(self) -> Quantity:
        """Kilograms avoided under the same fractional training-work reduction.

        Equation: ``(training_carbon * training_cut).to('kg')``.
        """
        return (self.training_carbon * self.training_cut).to("kg")


class ExplanationLoadScenario(BaseScenario):
    """Reasons owed for every denial: ``decisions_per_day`` at ``denial_share``, each explained by a
    sampling attributor of ``samples`` model evaluations or by exact Shapley over ``features``.

    Notes
    -----
    Equation: Sampling work is ``denials*samples``; exhaustive subset work is
    ``denials*2**features``.

    Assumptions: Every denial gets one explanation; exact work counts one model evaluation
    per feature subset.

    Excludes: Memoization, specialized Shapley algorithms, explanation quality, and
    non-model processing cost.

    References
    ----------
    Derivation: Sampling work is ``denials*samples``; exhaustive subset work is
    ``denials*2**features``. This is declared scenario accounting rather than a fitted empirical
    law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.CreditDenialExplanationLoad
    >>> value = scenario.notices_per_day
    >>> round(float(value), 3)
    475000.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    decisions_per_day: float
    denial_share: float
    samples: int
    features: int

    @property
    def notices_per_day(self) -> float:
        """Expected daily denials requiring an explanation.

        Equation: ``decisions_per_day * denial_share``.
        """
        return self.decisions_per_day * self.denial_share

    @property
    def sampled_evaluations(self) -> float:
        """Daily model-evaluation count for fixed samples per explained denial.

        Equation: ``notices_per_day * samples``.
        """
        return self.notices_per_day * self.samples

    @property
    def sampled_over_decisions(self) -> float:
        """Attribution model evaluations per original decision, dimensionless.

        Equation: ``sampled_evaluations / decisions_per_day``.
        """
        return self.sampled_evaluations / self.decisions_per_day

    @property
    def exact_per_explanation(self) -> int:
        """Coalition count ``2**features`` for exhaustive subset evaluation, without implementation
        reuse.

        Equation: ``2 ** features``.
        """
        return 2**self.features

    @property
    def exact_evaluations(self) -> float:
        """Daily exhaustive coalition-evaluation count across explained denials.

        Equation: ``notices_per_day * exact_per_explanation``.
        """
        return self.notices_per_day * self.exact_per_explanation


class ReviewStaffingScenario(BaseScenario):
    """Substantive human review: ``appeal_rate`` of ``decisions_per_day`` at ``time_per_case`` each,
    staffed in ``shift``-long shifts; optionally against a reported ``compare_cases`` handled at
    ``compare_time_per_case``.

    Notes
    -----
    Equation: Review labor is ``decisions_per_day*appeal_rate*time_per_case``; fractional shift
    equivalents divide by shift duration.

    Assumptions: Review duration is constant and every appealed case requires one review.

    Excludes: Integer staffing, breaks, peak demand, supervision overhead, and evidence that
    the declared time permits substantive review.

    References
    ----------
    Derivation: Review labor is ``decisions_per_day*appeal_rate*time_per_case``; fractional shift
    equivalents divide by shift duration. This is declared scenario accounting rather than a
    fitted empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.CreditAppealReview
    >>> value = scenario.appeals_per_day
    >>> round(float(value), 3)
    1000.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    decisions_per_day: float
    appeal_rate: float
    time_per_case: Quantity
    shift: Quantity
    compare_cases: Optional[float] = None
    compare_time_per_case: Optional[Quantity] = None

    @property
    def appeals_per_day(self) -> float:
        """Expected daily appealed decisions from decision count times appeal fraction.

        Equation: ``decisions_per_day * appeal_rate``.
        """
        return self.decisions_per_day * self.appeal_rate

    @property
    def reviewer_hours_per_day(self) -> Quantity:
        """Daily review labor in hours from appealed-case count times review duration.

        Equation: ``(appeals_per_day * time_per_case).to('hour')``.
        """
        return (self.appeals_per_day * self.time_per_case).to("hour")

    @property
    def reviewers(self) -> float:
        """Fractional staffed-shift equivalents, without integer rounding or peak-load reserve.

        Equation: ``(reviewer_hours_per_day / shift).to('').magnitude``.
        """
        return (self.reviewer_hours_per_day / self.shift).to("").magnitude

    @property
    def cases_per_hour(self) -> float:
        """Cases one reviewer processes in an hour at constant review duration.

        Equation: ``(Q_(1, 'hour') / time_per_case).to('').magnitude``.
        """
        return (Q_(1, "hour") / self.time_per_case).to("").magnitude

    @property
    def compare_hours(self) -> Quantity:
        """Total comparison-cohort labor hours from its count and supplied per-case duration.

        Equation: ``(compare_cases * compare_time_per_case).to('hour')``.
        """
        return (self.compare_cases * self.compare_time_per_case).to("hour")

    @property
    def compare_cases_per_hour(self) -> float:
        """Comparison-cohort cases per reviewer-hour at its supplied duration.

        Equation: ``(Q_(1, 'hour') / compare_time_per_case).to('').magnitude``.
        """
        return (Q_(1, "hour") / self.compare_time_per_case).to("").magnitude
