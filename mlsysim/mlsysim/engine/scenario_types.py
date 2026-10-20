"""Calculation scenarios: named, reusable calculations with their stated inputs.

A ``Scenario`` (``scenarios.py``) bundles a workload, a system, and SLAs for a
full multi-level evaluation. The calculation scenarios here are narrower: each one fixes
the inputs of one recurring calculation (a decode step on one accelerator, a
buffer crossing each link tier, a training run on a cluster) and hands them to
the solver that owns the arithmetic. ``.result`` runs the solver.

Every chapter, slide deck, and lab that quotes one of these numbers reads it
from the same ``Scenarios.*`` entry, so the inputs and the arithmetic each have
one home. ``.variant(...)`` returns a copy with some inputs changed (another
accelerator, another precision, another link), so a comparison reuses the
scenario rather than restating it.
"""

from __future__ import annotations

import math as _math

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from ..core.types import Quantity
from ..core.units import Q_
from ..hardware.types import HardwareNode
from ..models.types import Workload
from ..systems.types import Fleet


class BaseScenario(BaseModel):
    """Base for calculation scenarios: a name, a description, and stated inputs."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str

    def variant(self, **changes) -> "BaseScenario":
        """A copy of this scenario with some inputs changed."""
        return self.model_copy(update=changes)


class DecodeScenario(BaseScenario):
    """One decode step of ``workload`` on ``hardware`` (weight-streaming bound)."""
    workload: Workload
    hardware: HardwareNode
    precision: str = "fp16"
    batch_size: int = 1
    # Optional (added 2026-10-20): the precision whose peak rate prices the
    # compute side, when it differs from the weight precision (8-bit weights
    # on an FP8 tensor-core peak). None uses hardware.compute.peak_flops.
    peak_precision: Optional[str] = None
    # Optional (added 2026-10-20): tensor-parallel width; each device streams
    # and multiplies only its 1/tp_size shard of the weights.
    tp_size: int = 1

    @property
    def result(self):
        from .solvers import DecodeBoundModel
        peak = (self.hardware.compute.precision_flops[self.peak_precision]
                if self.peak_precision else None)
        return DecodeBoundModel().solve(self.workload, self.hardware, self.precision, self.batch_size,
                                        peak_flops=peak, tp_size=self.tp_size)


class TransferScenario(BaseScenario):
    """``copies`` of one ``payload`` moved at once over each named link tier.

    Optional (added 2026-10-20): ``egress_rate`` prices the bytes leaving a
    provider's network, ``compute_price`` converts that fee into accelerator
    hours, and ``job_time`` is the computation the move would serve, so a
    transfer can be set against the job it feeds (data gravity).
    """
    payload: Quantity
    links: Dict[str, Quantity]
    copies: int = 1
    egress_rate: Optional[Quantity] = None      # USD per byte moved out
    compute_price: Optional[Quantity] = None    # USD per accelerator-hour
    job_time: Optional[Quantity] = None

    @property
    def total_bytes(self) -> Quantity:
        return self.copies * self.payload

    def transfer_time(self, link: str) -> Quantity:
        """Bandwidth-only time to move every copy over one named link."""
        from ..physics import transfer_time
        return transfer_time(self.total_bytes, self.links[link])

    @property
    def egress_cost(self) -> Quantity:
        return (self.total_bytes * self.egress_rate).to("dollar")

    @property
    def egress_compute_hours(self) -> Quantity:
        """Accelerator-hours the egress fee would buy at ``compute_price``."""
        return (self.egress_cost / self.compute_price).to("hour")

    @property
    def result(self):
        from .solvers import TransferLadderModel
        return TransferLadderModel().solve(self.total_bytes, self.links)


class TensorParallelScenario(BaseScenario):
    """The activation AllReduces of one tensor-parallel step of ``workload``."""
    workload: Workload
    tp_size: int
    microbatch: int
    link_bandwidth: Quantity
    compute_time: Quantity          # stated compute per step, not derived
    # Optional (added 2026-10-20): tokens per sequence (default: the model's
    # context length) and AllReduces per layer per step (default: one forward,
    # one backward).
    seq_len: Optional[int] = None
    collectives_per_layer: int = 2

    @property
    def result(self):
        from .solvers import TensorParallelCommModel
        return TensorParallelCommModel().solve(
            self.workload, self.tp_size, self.microbatch, self.link_bandwidth, self.compute_time,
            seq_len=self.seq_len, collectives_per_layer=self.collectives_per_layer)


class TrainingRunScenario(BaseScenario):
    """Training ``workload`` on ``fleet``: peak time to delivered wall-clock time."""
    workload: Workload
    fleet: Fleet
    precision: str = "fp8"
    local_efficiency: float
    scaling_efficiency: float
    overheads: Dict[str, float]     # applied in order, each to the subtotal before it

    @property
    def result(self):
        from .solvers import TrainingRunModel
        return TrainingRunModel().solve(
            self.workload, self.fleet, self.local_efficiency, self.scaling_efficiency,
            self.overheads, self.precision)


class TrainingStateScenario(BaseScenario):
    """Persistent mixed-precision Adam state of ``workload`` and its shards."""
    workload: Workload
    tp_size: int
    dp_size: int
    activations: Quantity           # stated total for the whole model

    @property
    def result(self):
        from .solvers import TrainingStateModel
        return TrainingStateModel().solve(self.workload, self.tp_size, self.dp_size, self.activations)


class PowerTransientScenario(BaseScenario):
    """A barrier-aligned load step on ``hardware`` and across ``n_devices``."""
    hardware: HardwareNode
    low_power: Quantity             # communication-phase power per device
    rail_voltage: Quantity
    loop_inductance: Quantity
    rise_times: Dict[str, Quantity]
    n_devices: int
    ramp_window: Quantity

    @property
    def result(self):
        from .solvers import PowerTransientModel
        return PowerTransientModel().solve(
            self.hardware.tdp, self.low_power, self.rail_voltage, self.loop_inductance,
            self.rise_times, self.n_devices, self.ramp_window)


class CoolingScenario(BaseScenario):
    """Coolant flow that carries ``heat_load`` at ``temperature_rise``."""
    heat_load: Quantity
    temperature_rise: Quantity
    coolant: str = "air"

    @property
    def result(self):
        from .solvers import CoolantFlowModel
        return CoolantFlowModel().solve(self.heat_load, self.temperature_rise, self.coolant)


class OutageScenario(BaseScenario):
    """A failure domain of ``failed_devices`` down for ``repair_time`` during a job on ``fleet``."""
    failed_devices: int
    fleet: Fleet
    repair_time: Quantity
    device_hour_rate: Quantity
    rollback_times: Dict[str, Quantity] = {}

    @property
    def result(self):
        from .solvers import OutageCostModel
        return OutageCostModel().solve(
            self.failed_devices, self.fleet.total_accelerators, self.repair_time,
            self.device_hour_rate, self.rollback_times)


class FailureIntervalScenario(BaseScenario):
    """``n_components`` identical components, each with ``component_mtbf``, over a run."""
    component_mtbf: Quantity
    n_components: int
    run_duration: Quantity

    @property
    def result(self):
        from .solvers import FailureIntervalModel
        return FailureIntervalModel().solve(self.component_mtbf, self.n_components, self.run_duration)


class FailureIntervalSweepScenario(BaseScenario):
    """Interval between failures of ``component_mtbf`` components, at each job size in ``sizes``."""
    component_mtbf: Quantity
    sizes: List[int]

    @property
    def intervals(self) -> Dict[int, Quantity]:
        from ..physics.reliability import calc_mtbf_cluster
        return {n: calc_mtbf_cluster(self.component_mtbf, n) for n in self.sizes}

    @property
    def failures_per_day(self) -> Dict[int, float]:
        return {n: (Q_(1, "day") / t).to("").magnitude for n, t in self.intervals.items()}


class SeriesFailureScenario(BaseScenario):
    """A synchronous job on ``n_devices`` that stops when any one component fails.

    ``component_mttf`` gives each component's MTTF and ``per_node`` its count in
    one node of ``devices_per_node`` devices; a redundant part that cannot fail
    the node singly is counted 0. Added 2026-10-20.
    """
    n_devices: int
    devices_per_node: int
    component_mttf: Dict[str, Quantity]
    per_node: Dict[str, int]

    @property
    def result(self):
        from .solvers import SeriesFailureModel
        return SeriesFailureModel().solve(self.n_devices, self.devices_per_node,
                                          self.component_mttf, self.per_node)


class CheckpointIntervalScenario(BaseScenario):
    """Checkpoints of ``workload`` written at ``write_bandwidth``, against failures every ``mtbf``.

    ``write_time`` states the write directly when no workload is given.
    Added 2026-10-20.
    """
    mtbf: Quantity
    workload: Optional[Workload] = None
    bytes_per_param: Optional[Quantity] = None
    write_bandwidth: Optional[Quantity] = None
    write_time: Optional[Quantity] = None

    @property
    def checkpoint_bytes(self) -> Quantity:
        from ..physics.quantities import memory_from_params
        return memory_from_params(self.workload.parameters, self.bytes_per_param).to("byte")

    @property
    def t_write(self) -> Quantity:
        if self.write_time is not None:
            return self.write_time.to("second")
        return (self.checkpoint_bytes / self.write_bandwidth).to("second")

    @property
    def result(self):
        from .solvers import CheckpointIntervalModel
        return CheckpointIntervalModel().solve(self.t_write, self.mtbf)

    def waste_at(self, interval: Quantity) -> float:
        """Share of wall time lost when saving every ``interval`` instead."""
        from ..physics.reliability import calc_checkpoint_waste
        return calc_checkpoint_waste(self.t_write, interval, self.mtbf)


class RecoveryBudgetScenario(BaseScenario):
    """Cold restarts of ``workload`` after failures of ``n_devices`` components of ``component_mtbf``.

    ``recovery`` is the four-term budget profile (Systems.Reliability.Recovery).
    Added 2026-10-20.
    """
    workload: Workload
    bytes_per_param: Quantity
    recovery: Any
    component_mtbf: Quantity
    n_devices: int

    @property
    def checkpoint_bytes(self) -> Quantity:
        from ..physics.quantities import memory_from_params
        return memory_from_params(self.workload.parameters, self.bytes_per_param).to("byte")

    @property
    def result(self):
        from ..physics.reliability import calc_mtbf_cluster
        from .solvers import RecoveryBudgetModel
        return RecoveryBudgetModel().solve(
            self.checkpoint_bytes, self.recovery,
            calc_mtbf_cluster(self.component_mtbf, self.n_devices), self.n_devices)


class HotSpareScenario(BaseScenario):
    """``spare_devices`` held idle beside the job in ``failures``, priced at ``device_hour_rate``.

    The spare pays off when it shortens every recovery by more than
    ``breakeven``. Added 2026-10-20.
    """
    failures: Any                           # SeriesFailureScenario
    spare_devices: int
    device_hour_rate: Quantity

    @property
    def spare_cost_per_day(self) -> Quantity:
        return (self.spare_devices * self.device_hour_rate * Q_(1, "day")).to("USD")

    @property
    def value_per_minute(self) -> Quantity:
        """Daily value of one minute removed from every recovery."""
        r = self.failures.result
        return (r.failures_per_day * self.failures.n_devices * self.device_hour_rate
                * Q_(1, "minute")).to("USD")

    @property
    def breakeven(self) -> Quantity:
        return (self.spare_cost_per_day / self.value_per_minute * Q_(1, "minute")).to("second")


class StragglerCostScenario(BaseScenario):
    """One of ``n_devices`` in a synchronous job runs at ``slow_speed`` of its normal rate."""
    n_devices: int
    step_time: Quantity
    slow_speed: float
    device_hour_rate: Quantity

    @property
    def slow_step(self) -> Quantity:
        return (self.step_time / self.slow_speed).to("second")

    @property
    def throughput(self) -> float:
        """Job throughput as a share of normal: the barrier waits for the slowest rank."""
        return (self.step_time / self.slow_step).to("").magnitude

    @property
    def idle_cost(self) -> Quantity:
        """Cost per hour of the capacity the barrier leaves idle."""
        return (self.n_devices * self.device_hour_rate * (1 - self.throughput)
                * Q_(1, "hour")).to("USD")


class HealthProbeScenario(BaseScenario):
    """A readiness probe every ``interval``, failing after ``timeout``, removing after ``failure_threshold`` misses."""
    interval: Quantity
    timeout: Quantity
    failure_threshold: int
    success_threshold: int

    @property
    def detection_window(self):
        """Earliest and latest time from a crash to removal from rotation."""
        from ..physics.reliability import calc_probe_detection_window
        return calc_probe_detection_window(self.interval, self.timeout, self.failure_threshold)


class SdcExposureScenario(BaseScenario):
    """Silent-corruption exposure of ``n_devices``, each at ``event_rate`` per device-hour, per ``step_time`` step."""
    n_devices: int
    event_rate: float
    step_time: Quantity
    run_duration: Optional[Quantity] = None

    @property
    def exposure(self) -> float:
        """Device-hours exposed in one step."""
        return self.n_devices * self.step_time.to("hour").magnitude

    @property
    def p_step(self) -> float:
        from ..physics.reliability import calc_step_event_probability
        return calc_step_event_probability(self.event_rate, self.n_devices, self.step_time)

    @property
    def steps_between(self) -> float:
        return 1 / self.p_step

    @property
    def interval(self) -> Quantity:
        return (self.steps_between * self.step_time).to("hour")

    @property
    def run_events(self) -> float:
        return (self.run_duration / self.interval).to("").magnitude


class AcceleratorComparison(BaseScenario):
    """``candidate`` against ``baseline``: peak, power, and peak per watt.

    Peak rates compare at one convention (dense FP16 unless ``precision`` says
    otherwise); a ratio across precisions mixes a format change into the
    hardware change.
    """
    baseline: HardwareNode
    candidate: HardwareNode
    precision: Optional[str] = None
    baseline_device_hours: Optional[float] = None   # a job's size measured on the baseline

    def _peak(self, hw: HardwareNode) -> Quantity:
        if self.precision is None:
            return hw.compute.peak_flops
        return hw.compute.precision_flops[self.precision]

    @property
    def peak_ratio(self) -> float:
        return (self._peak(self.candidate) / self._peak(self.baseline)).to("").magnitude

    @property
    def tdp_ratio(self) -> float:
        return (self.candidate.tdp / self.baseline.tdp).to("").magnitude

    @property
    def efficiency_ratio(self) -> float:
        return self.peak_ratio / self.tdp_ratio

    def equivalent_hours(self, baseline_hours: float) -> float:
        """Candidate device-hours that match ``baseline_hours`` at equal sustained share of peak."""
        return baseline_hours / self.peak_ratio

    @property
    def candidate_device_hours(self) -> float:
        """Candidate device-hours for ``baseline_device_hours`` at equal sustained share of peak."""
        return self.equivalent_hours(self.baseline_device_hours)


class SustainedThroughputScenario(BaseScenario):
    """A run budgeted from peak throughput, then run at its sustained share of peak.

    Device-hours scale as ``1 / sustained_fraction``: a plan made from the
    datasheet rate needs that many times the hours when the workload sustains
    only part of peak.
    """
    planned_device_hours: float
    sustained_fraction: float

    @property
    def required_device_hours(self) -> float:
        return self.planned_device_hours / self.sustained_fraction

    @property
    def stretch(self) -> float:
        return 1 / self.sustained_fraction


class ServingScenario(BaseScenario):
    """Serving ``workload`` on ``hardware`` at a stated context and batch (``ServingModel``)."""
    workload: Workload
    hardware: HardwareNode
    seq_len: int
    batch_size: int = 1

    @property
    def result(self):
        from .solvers import ServingModel
        return ServingModel().solve(self.workload, self.hardware, seq_len=self.seq_len,
                                    batch_size=self.batch_size)


class GradientSyncScenario(BaseScenario):
    """A data-parallel step of ``workload``: compute, then a ring AllReduce over each link."""
    workload: Workload
    precision: str = "fp16"
    links: Dict[str, Quantity]
    compute_time: Quantity          # stated local compute per step, not derived
    energy_per_bit: Optional[Quantity] = None
    steps: int = 1
    compute_speedup: float = 1.0    # a faster accelerator divides only the compute term

    @property
    def result(self):
        from .solvers import GradientSyncModel
        return GradientSyncModel().solve(self.workload, self.precision, self.links, self.compute_time,
                                         self.energy_per_bit, self.steps, self.compute_speedup)


class DataParallelIntensityScenario(BaseScenario):
    """Communication intensity of one data-parallel worker of ``workload``."""
    workload: Workload
    sequences_per_worker: int
    link_bandwidth: Quantity
    accelerator: HardwareNode

    @property
    def result(self):
        from .solvers import CommunicationIntensityModel
        return CommunicationIntensityModel().solve(
            self.workload, self.sequences_per_worker, self.link_bandwidth, self.accelerator.compute.peak_flops)


class IntensityScenario(BaseScenario):
    """A workload stated directly as bytes sent and FLOPs done per device per step."""
    bytes_per_step: Quantity
    flops_per_step: Quantity

    @property
    def intensity(self) -> float:
        return self.bytes_per_step.to("byte").magnitude / self.flops_per_step.to("flop").magnitude


class AvailabilityScenario(BaseScenario):
    """A job that needs all ``n_components``, each up a fraction ``availability`` of the time."""
    availability: float
    n_components: int

    @property
    def all_up(self) -> float:
        from ..physics.reliability import calc_all_up_availability
        return calc_all_up_availability(self.availability, self.n_components)

    def required_availability(self, all_up_target: float) -> float:
        """Per-component availability that keeps ``all_up`` at ``all_up_target`` (added 2026-10-20)."""
        return all_up_target ** (1 / self.n_components)

    def downtime(self, interval: Quantity) -> Quantity:
        """Expected time one component is down in ``interval``."""
        from ..physics.reliability import calc_downtime
        return calc_downtime(self.availability, interval)

    @property
    def downtime_per_year(self) -> Quantity:
        """Expected time one component is down in a 365-day year."""
        from ..core.units import DAYS_PER_YEAR, Q_
        return self.downtime(Q_(DAYS_PER_YEAR, "day")).to("minute")

    # Parallel composition (added 2026-10-20): a service up while any of
    # ``replicas`` independent copies is up.
    replicas: int = 1

    @property
    def any_up(self) -> float:
        from ..physics.reliability import calc_availability_stacked
        return calc_availability_stacked(self.availability, self.replicas)

    @property
    def replicated_downtime_per_year(self) -> Quantity:
        """Expected time all ``replicas`` are down at once in a 365-day year."""
        from ..physics.reliability import calc_downtime
        from ..core.units import DAYS_PER_YEAR, Q_
        return calc_downtime(self.any_up, Q_(DAYS_PER_YEAR, "day")).to("minute")


class ComputeOptimalScenario(BaseScenario):
    """``workload``'s training budget, re-split at a compute-optimal tokens-per-parameter ratio."""
    workload: Workload
    tokens_per_param: float
    flops_per_param_token: float

    @property
    def budget_flops(self) -> float:
        p = self.workload.parameters.to("count").magnitude
        d = self.workload.training_tokens.to("count").magnitude
        return self.flops_per_param_token * p * d

    @property
    def actual_tokens_per_param(self) -> float:
        return self.workload.training_tokens.to("count").magnitude / self.workload.parameters.to("count").magnitude

    @property
    def optimal(self):
        """(parameters, tokens) that spend the same budget at ``tokens_per_param``."""
        from ..physics.scaling import calc_compute_optimal_allocation
        return calc_compute_optimal_allocation(self.budget_flops, self.tokens_per_param, self.flops_per_param_token)


class StepBreakdownScenario(BaseScenario):
    """One distributed step read through the Fleet Law: compute/N + comm + sync - overlap."""
    compute_time: Quantity          # T_compute / N
    communication_time: Quantity
    synchronization_time: Quantity
    overlap_time: Quantity
    compute_speedup: float = 1.0
    compute_phases: Dict[str, Quantity] = {}   # optional split of compute_time (forward, backward)

    @property
    def step_time(self) -> Quantity:
        return (self.compute_time / self.compute_speedup + self.communication_time
                + self.synchronization_time - self.overlap_time)

    @property
    def compute_fraction(self) -> float:
        return ((self.compute_time / self.compute_speedup) / self.step_time).to("").magnitude

    @property
    def exposed_communication_fraction(self) -> float:
        """Share of the step spent in communication and synchronization that overlap does not hide."""
        exposed = self.communication_time + self.synchronization_time - self.overlap_time
        return (exposed / self.step_time).to("").magnitude


class ScalingStepScenario(BaseScenario):
    """Step time at N and at ``factor`` x N devices (illustrative units allowed)."""
    step_at_n: float
    step_at_scaled: float
    factor: float

    @property
    def ideal_step(self) -> float:
        return self.step_at_n / self.factor

    @property
    def speedup(self) -> float:
        return self.step_at_n / self.step_at_scaled

    @property
    def efficiency(self) -> float:
        return self.speedup / self.factor


class ScalingOverheadScenario(BaseScenario):
    """A planned speedup when communication and coordination take a share of the scaled step."""
    planned_speedup: float
    overhead_fraction: float

    @property
    def realized_speedup(self) -> float:
        return self.planned_speedup * (1 - self.overhead_fraction)


class PowerLawScenario(BaseScenario):
    """A power-law term ``A R^-alpha`` and the resource multiple that halves it."""
    exponent: float

    @property
    def halving_multiple(self) -> float:
        from ..physics.scaling import calc_halving_multiple
        return calc_halving_multiple(self.exponent)


class TrainingGrowthScenario(BaseScenario):
    """Growth of training compute over a window of published runs, split into its factors."""
    series_name: str
    first_year: str
    last_year: str

    @property
    def result(self):
        from ..literature.registry import Literature
        from .solvers import TrainingGrowthModel
        rows = getattr(Literature.Series, self.series_name).entries
        return TrainingGrowthModel().solve(rows, self.first_year, self.last_year)


class PublishedRunScenario(BaseScenario):
    """A published training run: its total compute spread over its accelerators and days."""
    workload: Workload
    accelerator: HardwareNode       # the part the run used, for its peak rate

    @property
    def achieved_rate(self) -> Quantity:
        """Average delivered FLOP/s per accelerator over the whole run, idle time included."""
        w = self.workload
        n = w.training_accelerators_ref.to("count").magnitude
        return (w.training_ops / (n * w.training_days_ref)).to("TFLOP/s")

    @property
    def peak_fraction(self) -> float:
        return (self.achieved_rate / self.accelerator.compute.peak_flops).to("").magnitude



# --- Network fabrics (added 2026-10-20) --------------------------------------

class AlphaBetaScenario(BaseScenario):
    """Named messages on named paths, each path an ``alpha`` startup and a ``beta`` bandwidth."""
    alphas: Dict[str, Quantity]
    bandwidths: Dict[str, Quantity]
    messages: Dict[str, Quantity]

    @property
    def result(self):
        from .solvers import AlphaBetaModel
        return AlphaBetaModel().solve(self.alphas, self.bandwidths, self.messages)


class PathLoadScenario(BaseScenario):
    """Equal flows placed on equal-cost paths; the most loaded path drains last.

    Loads count flows per path; drain time is in units of one flow alone on a path.
    """
    path_loads: List[int]

    @property
    def flows(self) -> int:
        return sum(self.path_loads)

    @property
    def paths(self) -> int:
        return len(self.path_loads)

    @property
    def even_load(self) -> float:
        return self.flows / self.paths

    @property
    def worst_load(self) -> int:
        return max(self.path_loads)

    @property
    def idle_paths(self) -> int:
        return sum(1 for n in self.path_loads if n == 0)

    @property
    def slowdown(self) -> float:
        """Drain time of the worst path against an even split."""
        return self.worst_load / self.even_load


class LinkBuildoutScenario(BaseScenario):
    """``links_per_device`` links per accelerator of ``fleet``, each with a price and a module power."""
    fleet: Fleet
    links_per_device: int
    cost_per_link: Quantity
    power_per_link: Quantity

    @property
    def n_links(self) -> int:
        return self.links_per_device * self.fleet.total_accelerators

    @property
    def total_cost(self) -> Quantity:
        return self.n_links * self.cost_per_link

    @property
    def total_power(self) -> Quantity:
        return (self.n_links * self.power_per_link).to("kW")


class BandwidthDelayScenario(BaseScenario):
    """Data a link carries during ``delay``: line rate times delay (for example, a Go-Back-N resend window)."""
    link_bandwidth: Quantity
    delay: Quantity

    @property
    def product(self) -> Quantity:
        return (self.link_bandwidth * self.delay).to("KB")


class OversubscriptionScenario(BaseScenario):
    """A ring AllReduce whose edges all cross the bisection of ``fleet``, at 1:1 and at ``oversubscription_ratio``."""
    fleet: Fleet
    link_bandwidth: Quantity
    oversubscription_ratio: float
    payload_per_device: Quantity
    communication_fraction: float
    cluster_cost: Quantity

    @property
    def result(self):
        from .solvers import BisectionModel
        return BisectionModel().solve(
            self.fleet.total_accelerators, self.link_bandwidth, self.oversubscription_ratio,
            self.payload_per_device, self.communication_fraction, self.cluster_cost)


class FatTreeScenario(BaseScenario):
    """A non-blocking fat-tree of radix-``radix`` switches (Al-Fares et al. 2008)."""
    radix: int

    def hosts(self, tiers: int) -> int:
        from ..physics.communication import calc_fat_tree_hosts
        return calc_fat_tree_hosts(self.radix, tiers)

    def switches(self, tiers: int) -> int:
        from ..physics.communication import calc_fat_tree_switches
        return calc_fat_tree_switches(self.radix, tiers)

    @property
    def two_tier_hosts(self) -> int:
        return self.hosts(2)

    @property
    def three_tier_hosts(self) -> int:
        return self.hosts(3)

    @property
    def three_tier_switches(self) -> int:
        return self.switches(3)

    @property
    def pod_hosts(self) -> int:
        """Hosts in one pod of the three-tier design: k/2 leaves of k/2 host ports."""
        return (self.radix // 2) ** 2


class TorusScenario(BaseScenario):
    """``n_endpoints`` as a cubic torus against a non-blocking fat-tree, at equal per-link bandwidth."""
    n_endpoints: int
    link_bandwidth: Quantity
    dims: int = 3
    fat_tree_tiers: int = 3

    @property
    def side(self) -> int:
        return round(self.n_endpoints ** (1 / self.dims))

    @property
    def torus_cut_links(self) -> int:
        from ..physics.communication import calc_torus_bisection_links
        return calc_torus_bisection_links(self.side, self.dims)

    @property
    def fat_tree_cut_links(self) -> int:
        return self.n_endpoints // 2

    @property
    def torus_bisection(self) -> Quantity:
        return self.torus_cut_links * self.link_bandwidth

    @property
    def fat_tree_bisection(self) -> Quantity:
        return self.fat_tree_cut_links * self.link_bandwidth

    @property
    def torus_diameter(self) -> int:
        from ..physics.communication import calc_torus_diameter
        return calc_torus_diameter(self.side, self.dims)

    @property
    def fat_tree_path_switches(self) -> int:
        from ..physics.communication import calc_fat_tree_path_switches
        return calc_fat_tree_path_switches(self.fat_tree_tiers)


class HopPathScenario(BaseScenario):
    """Named paths through ``hops`` identical switch traversals of ``hop_latency`` each."""
    hop_latency: Quantity
    hops: Dict[str, int]

    @property
    def latencies(self) -> Dict[str, Quantity]:
        from ..physics.communication import calc_hop_latency
        return {k: calc_hop_latency(n, self.hop_latency) for k, n in self.hops.items()}


class IncastScenario(BaseScenario):
    """``n_senders`` at ``sender_bandwidth`` into one egress port with a ``buffer``."""
    n_senders: int
    sender_bandwidth: Quantity
    egress_bandwidth: Quantity
    buffer: Quantity
    feedback_rtt: Quantity

    @property
    def result(self):
        from .solvers import IncastModel
        return IncastModel().solve(self.n_senders, self.sender_bandwidth, self.egress_bandwidth,
                                   self.buffer, self.feedback_rtt)


class ComponentExposureScenario(BaseScenario):
    """Independent components, each affected with ``probability`` (per day, or at one instant).

    ``counts`` lists the factors whose product is the component count, so the
    breakdown (devices, links per device, modules per link) stays visible.
    """
    counts: Dict[str, int]
    probability: float

    @property
    def n_components(self) -> int:
        n = 1
        for c in self.counts.values():
            n *= c
        return n

    @property
    def expected_events(self) -> float:
        return self.n_components * self.probability

    @property
    def p_none(self) -> float:
        from ..physics.reliability import calc_all_up_availability
        return calc_all_up_availability(1 - self.probability, self.n_components)

    @property
    def p_any(self) -> float:
        return 1 - self.p_none


class SilentSlowdownScenario(BaseScenario):
    """A run on ``fleet`` slowed a few percent by a degraded link, with no alarm.

    ``slowdowns`` are stated per-step slowdowns; ``bandwidth_loss`` and
    ``communication_fraction`` give the no-overlap upper bound, where only the
    communicating share of the step stretches by ``1 / (1 - loss)``.
    """
    fleet: Fleet
    run_duration: Quantity
    slowdowns: Dict[str, float]
    device_hour_rate: Quantity
    bandwidth_loss: float
    communication_fraction: float

    @property
    def device_hours(self) -> float:
        return (self.fleet.total_accelerators * self.run_duration).to("hour").magnitude

    @property
    def lost_device_hours(self) -> Dict[str, float]:
        return {k: self.device_hours * f for k, f in self.slowdowns.items()}

    @property
    def lost_cost(self) -> Dict[str, Quantity]:
        rate = self.device_hour_rate.to("USD/hour").magnitude
        return {k: Q_(h * rate, "USD") for k, h in self.lost_device_hours.items()}

    @property
    def no_overlap_slowdown(self) -> float:
        f = self.communication_fraction
        return (1 - f) + f / (1 - self.bandwidth_loss) - 1


class CollectiveFitScenario(BaseScenario):
    """A floor-plus-bandwidth fit to the measured collective sweep ``series_name`` (Literature.Series)."""
    series_name: str
    floor_cutoff: Quantity
    mid_low: Quantity
    mid_high: Quantity

    @property
    def rows(self):
        from ..literature.registry import Literature
        return getattr(Literature.Series, self.series_name).entries

    @property
    def result(self):
        from .solvers import CollectiveFitModel
        return CollectiveFitModel().solve(self.rows, self.floor_cutoff, self.mid_low, self.mid_high)


class NicSharingScenario(BaseScenario):
    """One NIC split into ``n_virtual_functions`` rate-capped slices, and a burst at full line rate."""
    nic_bandwidth: Quantity
    n_virtual_functions: int
    burst_size: Quantity

    @property
    def slice_rate(self) -> Quantity:
        return self.nic_bandwidth / self.n_virtual_functions

    @property
    def burst_time(self) -> Quantity:
        return (self.burst_size / self.nic_bandwidth).to("second")


class SlowdownDiagnosisScenario(BaseScenario):
    """A training step that slowed, and a pairwise link test that finds one link at a fraction of its rate."""
    step_before: Quantity
    step_after: Quantity
    utilization_before: float
    utilization_after: float
    days_since_launch: int
    port_bandwidth: Quantity
    delivered_efficiency: float      # payload share a healthy link test reports
    measured_fraction: float         # the degraded pair's share of the expected rate

    @property
    def slowdown(self) -> float:
        return (self.step_after / self.step_before - 1).to("").magnitude

    @property
    def expected_bandwidth(self) -> Quantity:
        return self.delivered_efficiency * self.port_bandwidth

    @property
    def measured_bandwidth(self) -> Quantity:
        return self.expected_bandwidth * self.measured_fraction
# --- Storage: feeding and saving a training fleet (added 2026-10-20) --------

class StorageFootprintScenario(BaseScenario):
    """What a training run of ``workload`` reads once, saves every interval, and archives."""
    workload: Workload
    bytes_per_param: Quantity       # resumable checkpoint contract
    corpus_bytes: Quantity          # bytes read per epoch
    source_bytes: Quantity          # the compressed source kept for lineage
    run_duration: Quantity
    save_interval: Quantity
    shard_bytes: Quantity
    archive_every: int = 100        # keep every Nth checkpoint (and the final one)

    @property
    def result(self):
        from .solvers import StorageFootprintModel
        return StorageFootprintModel().solve(
            self.workload.parameters, self.bytes_per_param, self.corpus_bytes, self.source_bytes,
            self.run_duration, self.save_interval, self.shard_bytes, self.archive_every)


class TieredCheckpointScenario(BaseScenario):
    """One save of ``workload``'s state on ``fleet``: local write per node, then a durable drain."""
    workload: Workload
    bytes_per_param: Quantity
    fleet: Fleet
    local_bandwidth: Quantity       # per node
    durable_bandwidth: Quantity     # aggregate, shared by every node
    save_interval: Quantity

    @property
    def result(self):
        from ..physics.quantities import memory_from_params
        from .solvers import TieredCheckpointModel
        return TieredCheckpointModel().solve(
            memory_from_params(self.workload.parameters, self.bytes_per_param), self.fleet.count,
            self.local_bandwidth, self.durable_bandwidth, self.save_interval)


class CheckpointStormScenario(BaseScenario):
    """Every one of ``replicas`` writes ``workload``'s full resumable state through one ingest path."""
    workload: Workload
    bytes_per_param: Quantity
    replicas: int
    ingest_bandwidth: Quantity

    @property
    def result(self):
        from ..physics.quantities import memory_from_params
        from .solvers import CheckpointStormModel
        return CheckpointStormModel().solve(
            memory_from_params(self.workload.parameters, self.bytes_per_param), self.replicas,
            self.ingest_bandwidth)


class InputDemandScenario(BaseScenario):
    """``consumers`` each read ``samples_per_step`` samples per ``step_time`` of compute."""
    consumers: int
    samples_per_step: int
    bytes_per_sample: Quantity
    step_time: Quantity
    busy_fraction: float = 1.0

    @property
    def result(self):
        from .solvers import InputDemandModel
        return InputDemandModel().solve(self.consumers, self.samples_per_step, self.bytes_per_sample,
                                        self.step_time, self.busy_fraction)


class DataStallScenario(BaseScenario):
    """A batch delivered in ``io_time`` against ``compute_time`` of compute, in sequence or overlapped."""
    compute_time: Quantity
    io_time: Quantity

    @property
    def result(self):
        from .solvers import DataStallModel
        return DataStallModel().solve(self.compute_time, self.io_time)

    def relative_throughput(self, compute_speedup: float) -> float:
        """Overlapped batches per second when compute runs ``compute_speedup`` times faster, delivery fixed."""
        return (max(self.compute_time, self.io_time)
                / max(self.compute_time / compute_speedup, self.io_time)).to("").magnitude

    @property
    def plateau(self) -> float:
        """The limit of ``relative_throughput`` as compute speeds up without bound."""
        return (max(self.compute_time, self.io_time) / self.io_time).to("").magnitude


class PrefetchDepthScenario(BaseScenario):
    """Ready batches that cover each tier's tail read, and host memory at a safety depth."""
    compute_time: Quantity
    tail_latencies: Dict[str, Quantity]
    batch_bytes: Quantity
    safety_depth: int
    consumers_per_node: int

    @property
    def result(self):
        from .solvers import PrefetchDepthModel
        return PrefetchDepthModel().solve(self.compute_time, self.tail_latencies, self.batch_bytes,
                                          self.safety_depth, self.consumers_per_node)


class TieredCacheScenario(BaseScenario):
    """A fast tier in front of a slow one: a stated hit rate, or capacity over a uniformly sampled corpus."""
    hit_rate: Optional[float] = None
    hot_capacity: Optional[Quantity] = None
    corpus_parts: Optional[Dict[str, Quantity]] = None   # a corpus sampled uniformly across its parts
    miss_penalty: Optional[float] = None       # miss / hit service time, when no bandwidths are given
    hot_bandwidth: Optional[Quantity] = None
    miss_bandwidth: Optional[Quantity] = None

    @property
    def hit(self) -> float:
        if self.hit_rate is not None:
            return self.hit_rate
        return (self.hot_capacity / self.corpus_bytes).to("").magnitude

    @property
    def corpus_bytes(self) -> Quantity:
        return sum(self.corpus_parts.values())

    @property
    def result(self):
        from .solvers import TieredCacheModel
        return TieredCacheModel().solve(self.hit, self.miss_penalty, self.hot_bandwidth, self.miss_bandwidth)


class HostIoCpuScenario(BaseScenario):
    """``devices`` each consuming ``requests_per_device`` reads per second, at each path's CPU service time."""
    requests_per_device: float      # per second
    devices: int
    service_times: Dict[str, Quantity]

    @property
    def result(self):
        from .solvers import HostIoCpuModel
        return HostIoCpuModel().solve(self.devices * self.requests_per_device, self.service_times)


class ObjectStorageBillScenario(BaseScenario):
    """A dataset kept in object storage for a billing period and read ``reads`` times across a boundary."""
    dataset_bytes: Quantity
    storage_rate: Quantity
    billing_period: Quantity
    egress_rate: Quantity
    reads: int
    files: int
    request_fee: Quantity
    requests_per_fee: int
    retrieval_rate: Quantity

    @property
    def result(self):
        from .solvers import ObjectStorageBillModel
        return ObjectStorageBillModel().solve(
            self.dataset_bytes, self.storage_rate, self.billing_period, self.egress_rate, self.reads,
            self.files, self.request_fee, self.requests_per_fee, self.retrieval_rate)


class StreamVsStageScenario(BaseScenario):
    """Stream every epoch from remote object storage, or stage the dataset locally once per run."""
    dataset_bytes: Quantity
    epochs: int
    runs_per_year: int
    object_rate: Quantity
    local_rate: Quantity
    egress_rate: Quantity
    year: Quantity

    @property
    def result(self):
        from .solvers import StreamVsStageModel
        return StreamVsStageModel().solve(self.dataset_bytes, self.epochs, self.runs_per_year,
                                          self.object_rate, self.local_rate, self.egress_rate, self.year)


class UtilizationLossScenario(BaseScenario):
    """``fleet``'s accelerators billed at ``device_hour_rate`` while utilization falls."""
    fleet: Fleet
    device_hour_rate: Quantity
    healthy_utilization: float
    degraded_utilization: float
    duration: Quantity

    @property
    def result(self):
        from .solvers import UtilizationLossModel
        return UtilizationLossModel().solve(self.fleet.total_accelerators, self.device_hour_rate,
                                            self.healthy_utilization, self.degraded_utilization, self.duration)


class SyntheticFootprintScenario(BaseScenario):
    """Generated samples with provenance metadata, read once per verification pass."""
    payload_bytes: Quantity
    provenance_overhead: float
    verification_passes: int

    @property
    def result(self):
        from .solvers import SyntheticFootprintModel
        return SyntheticFootprintModel().solve(self.payload_bytes, self.provenance_overhead,
                                               self.verification_passes)


class StorageBenchmarkScenario(BaseScenario):
    """Per-accelerator storage demand measured in a benchmark series, older accelerator first."""
    series_name: str
    division: str
    workloads: tuple
    accelerators: tuple

    @property
    def rows(self) -> list:
        from ..literature.registry import Literature
        return [r for r in getattr(Literature.Series, self.series_name).entries
                if r["division"] == self.division and r["au_check"] == "pass"]

    @property
    def result(self):
        from ..literature.registry import Literature
        from .solvers import StorageBenchmarkDemandModel
        return StorageBenchmarkDemandModel().solve(getattr(Literature.Series, self.series_name).entries,
                                                   self.division, self.workloads, self.accelerators)


class ShardPackagingScenario(BaseScenario):
    """``files`` samples packaged into ``shards`` large files on a file system rated at ``rated_bandwidth``."""
    files: int
    shards: int
    rated_bandwidth: Quantity

    @property
    def files_per_shard(self) -> float:
        return self.files / self.shards


class ParallelFileSystemScenario(BaseScenario):
    """Object storage servers of ``server_bandwidth`` each, enough for ``target_bandwidth`` aggregate."""
    target_bandwidth: Quantity
    server_bandwidth: Quantity

    @property
    def servers(self) -> int:
        import math
        return math.ceil((self.target_bandwidth / self.server_bandwidth).to("").magnitude - 1e-9)


class CompressedReadScenario(BaseScenario):
    """Compressed shards read at ``read_bandwidth``, decoded on CPU cores at each codec's output rate."""
    read_bandwidth: Quantity
    compression_ratios: Dict[str, float]     # decoded bytes per stored byte
    decode_rates: Dict[str, Quantity]        # decoded output per core

    def output_bandwidth(self, codec: str) -> Quantity:
        """Decoded bytes per second when the read path runs at full rate."""
        return self.read_bandwidth * self.compression_ratios[codec]

    def decode_cores(self, codec: str) -> float:
        """Cores needed to decode at that output rate."""
        return (self.output_bandwidth(codec) / self.decode_rates[codec]).to("").magnitude


# --- Embodied stopping, floor traction, and domain compute (added 2026-10-20) --

class StoppingBudgetScenario(BaseScenario):
    """A machine at ``speed`` that must stop: travel during ``delay``, then braking at ``deceleration``.

    ``clear_distance`` closes the budget when one is stated. ``speed_at_delay_end``
    replaces constant-speed delay travel with the trapezoid between two recorded
    speeds; braking is still from ``speed``. With a ``site`` (an embodied site
    scenario), the deceleration and clear distance default to the site's, and
    its localization bound and protective margin join the required distance.
    ``added_delay`` prices one more delay step as travel at ``speed``.
    """
    speed: Quantity
    delay: Quantity
    deceleration: Optional[Quantity] = None     # default: the site's credible deceleration
    clear_distance: Optional[Quantity] = None   # default: the site's clear distance
    speed_at_delay_end: Optional[Quantity] = None
    site: Optional[Any] = None                  # EmbodiedSiteScenario
    added_delay: Optional[Quantity] = None

    @property
    def a_brake(self) -> Quantity:
        return self.site.a_brake if self.deceleration is None else self.deceleration

    @property
    def d_clear(self) -> Optional[Quantity]:
        if self.clear_distance is not None or self.site is None:
            return self.clear_distance
        return self.site.d_clear

    @property
    def result(self):
        from .solvers import StoppingBudgetModel
        loc = None if self.site is None else self.site.delta_loc
        margin = None if self.site is None else self.site.delta_margin
        return StoppingBudgetModel().solve(self.speed, self.delay, self.a_brake, self.d_clear,
                                           self.speed_at_delay_end, loc, margin, self.added_delay)
# --- Physical-plant budgets of an embodied machine (added 2026-10-20) -------
# Each scenario names one machine at one site (``Embodied.Scenario.*``) or one
# actuator (``Actuators.*``) and states the inputs the site record does not
# carry; the solvers live in ``engine/solvers/plant.py``.

from typing import Any, Tuple  # noqa: E402



class TractionLossScenario(BaseScenario):
    """Floor friction falls from ``mu_before`` to ``mu_after`` under a machine that assumes ``deceleration``."""
    mu_before: float
    mu_after: float
    deceleration: Quantity
    drive_load_share: float = 1.0

    @property
    def result(self):
        from .solvers import TractionLossModel
        return TractionLossModel().solve(self.mu_before, self.mu_after, self.deceleration,
                                         self.drive_load_share)


class DomainComputeGapScenario(BaseScenario):
    """``model`` from a Literature series against the largest ``reference_group`` model of its year."""
    series_name: str
    model: str
    reference_group: str = "Language"

    @property
    def result(self):
        from ..literature.registry import Literature
        from .solvers import DomainComputeGapModel
        rows = getattr(Literature, self.series_name).entries
        return DomainComputeGapModel().solve(rows, self.model, self.reference_group)


# --- Agent trajectories (added 2026-10-20) -----------------------------------

class HorizonCompoundingScenario(BaseScenario):
    """Independent, unchecked steps each right with ``step_success``; success at each horizon."""
    step_success: float
    short_horizon: int
    long_horizon: int                         # the horizon the target must hold at
    target_success: float
    extended_horizon: Optional[int] = None    # a further horizon to price (for example, twice the long one)

    @property
    def horizons(self):
        return tuple(h for h in (self.short_horizon, self.long_horizon, self.extended_horizon) if h is not None)

    @property
    def result(self):
        from .solvers import HorizonCompoundingModel
        return HorizonCompoundingModel().solve(self.step_success, self.horizons, self.target_success,
                                               self.long_horizon)


class AcceptedTaskCostScenario(BaseScenario):
    """An attempt of model tokens plus other cost, retried until accepted; optional levers on it."""
    model_cost: Quantity                       # USD per attempt
    other_cost: Quantity                       # USD per attempt: sandbox, tests, review
    acceptance: float
    token_price_cut: Optional[float] = None    # fraction of the token price removed
    improved_acceptance: Optional[float] = None
    other_cost_cut: Optional[float] = None     # fraction of the other cost removed
    acceptance_gain: Optional[float] = None    # added to acceptance
    independent_attempts: Optional[int] = None

    @property
    def result(self):
        from .solvers import AcceptedTaskCostModel
        return AcceptedTaskCostModel().solve(self.model_cost, self.other_cost, self.acceptance,
                                             self.token_price_cut, self.improved_acceptance,
                                             self.other_cost_cut, self.acceptance_gain,
                                             self.independent_attempts)


class AppendOnlyContextScenario(BaseScenario):
    """A trajectory that appends each turn to its prompt and resends the whole history."""
    initial_tokens: Quantity
    tokens_per_turn: Quantity
    turns: int
    context_window: Quantity

    @property
    def result(self):
        from .solvers import AppendOnlyContextModel
        return AppendOnlyContextModel().solve(self.initial_tokens, self.tokens_per_turn, self.turns,
                                              self.context_window)


class TaskHorizonTrendScenario(BaseScenario):
    """Doubling time of the frontier 50% task horizon in a published series."""
    series_name: str
    fit_cap: Quantity                          # longest horizon the fits include
    recent_from: str                           # ISO date the recent fit starts
    published_recent_doubling: Optional[Quantity] = None   # the source's own figure, to check the refit

    @property
    def rows(self):
        from ..literature.registry import Literature
        return getattr(Literature, self.series_name).entries

    @property
    def result(self):
        from .solvers import TaskHorizonTrendModel
        return TaskHorizonTrendModel().solve(self.rows, self.fit_cap, self.recent_from)

    def fitted_minutes(self, when) -> float:
        """Horizon in minutes on the recent fit line at datetime ``when`` (fractional days count)."""
        r = self.result
        day = when.toordinal() + (when.hour * 3600 + when.minute * 60 + when.second) / 86400
        return 2 ** (r.recent_intercept + r.recent_slope_per_day * day)


class AgentRunScenario(BaseScenario):
    """One recorded agent run over a benchmark (a per-task ``Literature`` series)."""
    series_name: str
    call_bin_edges: List[int]                  # model-call bin edges for the acceptance-by-length table
    # Optional (added 2026-10-20): the harness's observation staging policy. An
    # output at or above ``truncation_chars`` characters reaches the model as its
    # first ``head_chars``, a count of elided characters, and its last ``tail_chars``.
    truncation_chars: Optional[int] = None
    head_chars: Optional[int] = None
    tail_chars: Optional[int] = None

    @property
    def rows(self):
        from ..literature.registry import Literature
        return getattr(Literature, self.series_name).entries

    def task(self, instance_id: str) -> dict:
        return next(r for r in self.rows if r["instance_id"] == instance_id)

    @property
    def result(self):
        from .solvers import AgentRunSummaryModel
        return AgentRunSummaryModel().solve(self.rows, self.call_bin_edges)


class BenchmarkTaskScenario(BaseScenario):
    """One task of a recorded run, with the hidden tests that judged it."""
    run: AgentRunScenario
    instance_id: str
    fail_to_pass_tests: int                    # new tests the fix must make pass
    pass_to_pass_tests: int                    # existing tests that must keep passing

    @property
    def record(self) -> dict:
        return self.run.task(self.instance_id)

    @property
    def model_calls(self) -> int:
        return int(self.record["model_calls"])

    @property
    def cost(self) -> float:
        return self.record["cost_usd"]

    @property
    def resolved(self) -> bool:
        return bool(self.record["resolved"])

    @property
    def files_in_submission(self) -> int:
        return int(self.record["files_in_submission"])

    @property
    def hidden_tests(self) -> int:
        return self.fail_to_pass_tests + self.pass_to_pass_tests
# --- Model calls and agent trajectories (added 2026-10-20) -------------------
# One model call (admission, sampling, a fallible check, prefill and decode
# time at the weight-transfer floors) and a trajectory of calls (horizon
# compounding, cost per accepted task, resent history). The per-token floors
# come from a DecodeScenario: compute_time is 2P / R_peak, the prefill floor
# per prompt token; memory_time is P b / BW, the decode floor per output token.

class ContextAdmissionScenario(BaseScenario):
    """A serialized request (text plus an image charge) against a window with a reserved reply."""
    window: int
    output_reserve: int
    text_tokens: int
    image_tokens: int = 0

    @property
    def result(self):
        from .solvers import ContextAdmissionModel
        return ContextAdmissionModel().solve(self.window, self.output_reserve, self.text_tokens,
                                             self.image_tokens)


class SamplingScenario(BaseScenario):
    """One fixed logit vector at a base and a sharpened temperature, then top-p at the base."""
    logits: List[float]
    base_temperature: float
    sharp_temperature: float
    top_p: float

    @property
    def result(self):
        from .solvers import SamplingModel
        return SamplingModel().solve(self.logits, self.base_temperature, self.sharp_temperature,
                                     self.top_p)


class CheckerPosteriorScenario(BaseScenario):
    """A fallible check over candidates, ``defect_rate`` of which violate the property."""
    defect_rate: float
    false_accept_rate: float         # P(PASS | defective)
    valid_accept_rate: float         # P(PASS | valid)

    @property
    def result(self):
        from .solvers import CheckerPosteriorModel
        return CheckerPosteriorModel().solve(self.defect_rate, self.false_accept_rate,
                                             self.valid_accept_rate)


class ModelCallScenario(BaseScenario):
    """One call of ``prompt_tokens`` (+ ``instruction_tokens``) in and ``output_tokens`` out at ``floors``."""
    floors: DecodeScenario
    prompt_tokens: int
    output_tokens: int
    instruction_tokens: int = 0      # format instructions added to the prompt
    shared_prefix_tokens: int = 0    # prompt prefix served from cache
    deadline: Optional[Quantity] = None
    stage_success: Optional[Dict[str, float]] = None   # stages a reply must clear to be accepted

    @property
    def result(self):
        from .solvers import ModelCallModel
        f = self.floors.result
        return ModelCallModel().solve(f.compute_time, f.memory_time,
                                      self.prompt_tokens + self.instruction_tokens, self.output_tokens,
                                      self.shared_prefix_tokens, self.deadline, self.stage_success)


class StreamCancelScenario(BaseScenario):
    """A streamed reply with ``output_allowance`` tokens, rejected and canceled at ``reject_step``."""
    floors: DecodeScenario
    output_allowance: int
    reject_step: int

    @property
    def skipped_steps(self) -> int:
        return self.output_allowance - self.reject_step

    @property
    def decode_time_avoided(self) -> Quantity:
        return (self.skipped_steps * self.floors.result.memory_time).to("second")


class TrajectoryResendScenario(BaseScenario):
    """A trajectory of ``turns`` calls that each resend the whole history (``TrajectoryResendModel``)."""
    floors: DecodeScenario
    first_prompt: int
    appended: int                    # model output plus tool observation added per turn
    output_per_turn: int
    turns: int

    @property
    def result(self):
        from .solvers import TrajectoryResendModel
        f = self.floors.result
        return TrajectoryResendModel().solve(f.compute_time, f.memory_time, self.first_prompt,
                                             self.appended, self.output_per_turn, self.turns)


class TokenMixScenario(BaseScenario):
    """Prompt and output tokens of a published request trace, priced at ``floors``."""
    series_name: str
    prompt_column: str
    output_column: str
    floors: DecodeScenario

    @property
    def result(self):
        from ..literature.registry import Literature
        from .solvers import TokenMixModel
        rows = getattr(Literature, self.series_name).entries
        f = self.floors.result
        return TokenMixModel().solve([r[self.prompt_column] for r in rows],
                                     [r[self.output_column] for r in rows],
                                     f.compute_time, f.memory_time)


# --- Distributed training (added 2026-10-20) ----------------------------------

class ParallelLayoutScenario(BaseScenario):
    """``workload`` trained on a TP x PP x DP layout: step time, efficiency, and state per device."""
    workload: Workload
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int
    microbatches: int = 1
    batch_tokens: float              # tokens per optimizer update
    flops_per_param_token: float
    accelerator: HardwareNode
    precision: str = "fp8"           # the compute precision whose peak sets the window
    local_efficiency: float          # share of peak before communication
    link_bandwidth: Quantity         # per-device data-parallel link
    overlap_fraction: float          # share of the AllReduce that may hide under compute
    gradient_precision: str = "fp16"
    window_resolution: Optional[Quantity] = None   # carry the compute window at this precision
    device_memory: Optional[Quantity] = None       # for the fewest devices that hold the state

    @property
    def n_devices(self) -> int:
        return self.tp_size * self.pp_size * self.dp_size

    @property
    def result(self):
        from .solvers import ParallelLayoutModel
        return ParallelLayoutModel().solve(
            self.workload, self.tp_size, self.pp_size, self.dp_size, self.microbatches,
            self.batch_tokens, self.flops_per_param_token,
            self.accelerator.compute.precision_flops[self.precision], self.local_efficiency,
            self.link_bandwidth, self.overlap_fraction, self.gradient_precision,
            self.window_resolution, self.device_memory)


class ActivationMemoryScenario(BaseScenario):
    """Activation memory of ``workload`` at a microbatch, under each recomputation strategy."""
    workload: Workload
    micro_batch: int
    seq_len: Optional[int] = None   # defaults to the model's context length

    def activations(self, strategy: str) -> Quantity:
        from ..physics.memory import calc_activation_memory
        w = self.workload
        seq = self.seq_len if self.seq_len is not None else w.context_length
        return calc_activation_memory(w.layers, seq, self.micro_batch, w.hidden_dim,
                                      n_heads=w.heads, strategy=strategy)

    @property
    def forward_equivalents(self) -> float:
        """A training step in forward passes: 6PD over the forward pass's 2PD."""
        from ..physics.transformer import calc_transformer_training_flops, calc_transformer_decode_flops
        p = self.workload.parameters
        return (calc_transformer_training_flops(p, 1) / calc_transformer_decode_flops(p, 1)).to("").magnitude

    @property
    def recompute_forward_equivalents(self) -> float:
        """The same step with full recomputation: 8PD over 2PD."""
        return self.forward_equivalents * (1 + self.recompute_overhead)

    @property
    def recompute_overhead(self) -> float:
        """Extra training arithmetic of full recomputation: 8PD over 6PD, minus one."""
        from ..physics.transformer import calc_transformer_training_flops
        full = calc_transformer_training_flops(self.workload.parameters, 1, activation_checkpointing=True)
        base = calc_transformer_training_flops(self.workload.parameters, 1)
        return (full / base).to("").magnitude - 1


class ZeroShardingScenario(BaseScenario):
    """``workload``'s training state per device at each ZeRO stage over ``n_devices``."""
    workload: Workload
    n_devices: int
    activations: Optional[ActivationMemoryScenario] = None
    activation_strategy: str = "full"
    link_bandwidth: Optional[Quantity] = None

    @property
    def result(self):
        from .solvers import ZeroShardingModel
        act = None if self.activations is None else self.activations.activations(self.activation_strategy)
        return ZeroShardingModel().solve(self.workload, self.n_devices, act, self.link_bandwidth)


class DataParallelScaleOutScenario(BaseScenario):
    """A data-parallel job of ``workload`` on ``nodes`` nodes, from a one-device profile."""
    workload: Workload
    precision: str = "fp16"
    step_time: Quantity              # one device, one local step
    training_time: Quantity          # one device, whole run
    per_device_batch: int
    devices_per_node: int
    nodes: int = 1
    intra_bandwidth: Quantity
    inter_bandwidth: Optional[Quantity] = None
    accumulation_steps: int = 1
    node_hour_rate: Optional[Quantity] = None

    @property
    def result(self):
        from .solvers import DataParallelScaleOutModel
        return DataParallelScaleOutModel().solve(
            self.workload, self.precision, self.step_time, self.training_time, self.per_device_batch,
            self.devices_per_node, self.nodes, self.intra_bandwidth, self.inter_bandwidth,
            self.accumulation_steps, self.node_hour_rate)


class PipelineScheduleScenario(BaseScenario):
    """A synchronous pipeline of ``stages`` stages fed ``microbatches`` microbatches per step."""
    stages: int
    microbatches: int

    @property
    def bubble(self) -> float:
        from ..physics.performance import calc_pipeline_bubble
        return calc_pipeline_bubble(self.stages, self.microbatches)

    @property
    def slots(self) -> int:
        """Equal time slots per step, forward and backward each ``m + p - 1``."""
        return 2 * (self.microbatches + self.stages - 1)

    @property
    def idle_slots(self) -> int:
        """Slots each stage idles per step: ``p - 1`` to fill and ``p - 1`` to drain."""
        return 2 * (self.stages - 1)

    @property
    def fill_slots(self) -> int:
        """The bubble's numerator, ``p - 1``."""
        return self.stages - 1

    @property
    def phase_slots(self) -> int:
        """The bubble's denominator, ``m + p - 1`` slots per phase."""
        return self.microbatches + self.stages - 1

    def microbatches_for_bubble(self, target: float) -> float:
        """The microbatch count above which the bubble falls below ``target``."""
        return (self.stages - 1) * (1 - target) / target


class TensorParallelLayerScenario(BaseScenario):
    """One layer of ``workload`` split ``tp_size`` ways: compute against the exchange on each link."""
    workload: Workload
    tp_size: int
    tokens: int
    accelerator: HardwareNode
    precision: str = "fp16_dense"
    local_efficiency: float
    links: Dict[str, Quantity]
    collectives_per_layer: int = 2

    @property
    def result(self):
        from .solvers import TensorParallelLayerModel
        return TensorParallelLayerModel().solve(
            self.workload, self.tp_size, self.tokens, self.accelerator.compute.precision_flops[self.precision],
            self.local_efficiency, self.links, self.collectives_per_layer)


class RingAttentionScenario(BaseScenario):
    """Ring attention on ``accelerator``: the smallest chunk that hides each K/V tile, per link."""
    accelerator: HardwareNode
    precision: str = "fp16_dense"
    local_efficiency: float
    element_bytes: Quantity
    links: Dict[str, Quantity]

    @property
    def min_chunk_tokens(self) -> Dict[str, float]:
        from ..physics.communication import calc_ring_attention_min_chunk
        rate = self.accelerator.compute.precision_flops[self.precision] * self.local_efficiency
        elem = self.element_bytes.to("byte").magnitude
        return {k: calc_ring_attention_min_chunk(elem, rate, bw) for k, bw in self.links.items()}


class SyncStrategyScenario(BaseScenario):
    """Data-parallel training to a fixed target under one synchronization strategy.

    Iterations to the target and the communication share are stated inputs:
    each iteration costs the one-worker iteration time times ``1 + overhead``,
    so speedup over one worker is ``base / (iterations x (1 + overhead))``.
    """
    workers: int
    per_worker_batch: int
    iterations: int
    communication_overhead: float
    base_iterations: int             # one worker, same target
    base_iteration_time: Quantity
    device_hour_rate: Quantity
    staleness: int = 0               # SSP bound; 0 is BSP

    @property
    def global_batch(self) -> int:
        return self.workers * self.per_worker_batch

    @property
    def ideal_iterations(self) -> float:
        return self.base_iterations / self.workers

    @property
    def iteration_time(self) -> Quantity:
        return self.base_iteration_time * (1 + self.communication_overhead)

    @property
    def speedup(self) -> float:
        return self.base_iterations / (self.iterations * (1 + self.communication_overhead))

    @property
    def sample_efficiency(self) -> float:
        return self.ideal_iterations / self.iterations

    @property
    def cost(self) -> float:
        """Allocated device cost of the run, in dollars."""
        hours = (self.iterations * self.iteration_time * self.workers).to("hour")
        return (hours * self.device_hour_rate).to("USD").magnitude

    @property
    def stale_lr_factor(self) -> float:
        """Learning-rate damping for stale updates, ``1 / sqrt(1 + s)``."""
        return 1 / (1 + self.staleness) ** 0.5


class CriticalBatchScenario(BaseScenario):
    """A pilot gradient-noise trace and a proposed batch: where the critical batch falls."""
    gradient_noise_trace: float      # tr(Sigma), per-sample gradient covariance
    gradient_norm_sq: float          # squared norm of the true gradient
    local_batch: int
    accumulation_steps: int
    devices: int

    @property
    def critical_batch(self) -> float:
        return self.gradient_noise_trace / self.gradient_norm_sq

    @property
    def per_device_samples(self) -> int:
        return self.local_batch * self.accumulation_steps

    @property
    def global_batch(self) -> int:
        return self.devices * self.per_device_samples

    @property
    def batch_ratio(self) -> float:
        return self.global_batch / self.critical_batch

    @property
    def crossover_devices(self) -> float:
        return self.critical_batch / self.per_device_samples


class AlignmentMemoryScenario(BaseScenario):
    """PPO and DPO alignment of ``policy`` with ``critic``-sized reward and value models."""
    policy: Workload
    critic: Workload
    tp_size: int
    pp_size: int
    dp_size: int = 1
    rollout_sequences: int
    rollout_tokens: int
    device_memory: Optional[Quantity] = None

    @property
    def result(self):
        from .solvers import AlignmentMemoryModel
        return AlignmentMemoryModel().solve(self.policy, self.critic, self.tp_size, self.pp_size,
                                            self.rollout_sequences, self.rollout_tokens, self.device_memory)


class ScalingSeriesScenario(BaseScenario):
    """Named same-round series of a benchmark table and their scaling efficiencies."""
    series_name: str
    specs: List  # (label, (round, submitter prefix, benchmark, accelerator substring))

    @property
    def result(self):
        from ..literature.registry import Literature
        from .solvers import ScalingSeriesModel
        rows = getattr(Literature.Series, self.series_name).entries
        return ScalingSeriesModel().solve(rows, self.specs)


# --- Collective communication (added 2026-10-20) ----------------------------

class AllReduceScheduleScenario(BaseScenario):
    """One buffer reduced by ring, simple tree, and parameter server at each of ``group_sizes``.

    ``run_steps`` and ``time_saved_fraction`` price a what-if: removing that
    share of the ring time on every step of a run.
    """
    message: Quantity
    group_sizes: List[int]
    alpha: Quantity
    bandwidth: Quantity
    run_steps: int = 1
    time_saved_fraction: float = 0.0

    @property
    def result(self):
        from .solvers import AllReduceAlgorithmModel
        return AllReduceAlgorithmModel().solve(self.message, self.group_sizes, self.alpha, self.bandwidth)

    def saved_time(self, n: int) -> Quantity:
        """Ring time removed per step at group size ``n``."""
        return (self.result.ring_time[n] * self.time_saved_fraction).to("second")

    def saved_device_hours(self, n: int) -> float:
        """Device-hours the what-if saves over the run at group size ``n``."""
        return (self.saved_time(n) * self.run_steps * n).to("hour").magnitude


class MessageFusionScenario(BaseScenario):
    """``count`` small messages sent one by one against one fused message on the same path.

    Each small message is far below ``n* = alpha beta``, so separately each
    costs its startup alone (the byte term is dropped); fused, the batch pays
    one startup plus its bytes.
    """
    alpha: Quantity
    bandwidth: Quantity
    message: Quantity
    count: int

    @property
    def crossover(self) -> Quantity:
        return self.alpha * self.bandwidth

    @property
    def fused_message(self) -> Quantity:
        return self.count * self.message

    @property
    def separate_time(self) -> Quantity:
        return self.count * self.alpha

    @property
    def fused_time(self) -> Quantity:
        return self.alpha + self.fused_message / self.bandwidth

    @property
    def speedup(self) -> float:
        return (self.separate_time / self.fused_time).to_base_units().magnitude

    @property
    def transfer_time(self) -> Quantity:
        """Byte time of one small message, message / bandwidth (added 2026-10-20)."""
        return (self.message / self.bandwidth).to("nanosecond")

    @property
    def startup_to_transfer(self) -> float:
        """How many times one message's startup exceeds its byte time (added 2026-10-20)."""
        return (self.alpha / self.transfer_time).to("").magnitude


class HierarchicalAllReduceScenario(BaseScenario):
    """A buffer AllReduced over ``n_nodes`` nodes of ``gpus_per_node`` GPUs, in three phases or one flat ring."""
    message: Quantity
    n_nodes: int
    gpus_per_node: int
    intra_bandwidth: Quantity      # per direction, inside the node
    inter_bandwidth: Quantity      # one port per GPU, across nodes
    inter_alpha: Quantity

    @property
    def n_devices(self) -> int:
        return self.n_nodes * self.gpus_per_node

    @property
    def result(self):
        from .solvers import HierarchicalAllReduceModel
        return HierarchicalAllReduceModel().solve(self.message, self.n_nodes, self.gpus_per_node,
                                                  self.intra_bandwidth, self.inter_bandwidth, self.inter_alpha)


class AllToAllDispatchScenario(BaseScenario):
    """One expert-parallel dispatch: ``total_tokens`` of ``hidden_dim`` values over ``n_devices`` holding ``n_experts``."""
    total_tokens: int
    n_devices: int
    n_experts: int
    hidden_dim: int
    bytes_per_element: Quantity
    alpha: Quantity
    bandwidth: Quantity
    receive_skew: float = 1.0      # remote tokens at the hottest receiver over the uniform load

    @property
    def result(self):
        from .solvers import AllToAllDispatchModel
        return AllToAllDispatchModel().solve(self.total_tokens, self.n_devices, self.n_experts, self.hidden_dim,
                                             self.bytes_per_element, self.alpha, self.bandwidth, self.receive_skew)


class RingTraceScenario(BaseScenario):
    """A ring AllReduce traced step by step on one short integer vector per GPU."""
    vectors: List[List[int]]

    @property
    def n_devices(self) -> int:
        return len(self.vectors)

    @property
    def result(self):
        from .solvers import RingTraceModel
        return RingTraceModel().solve(self.vectors)


class ErrorFeedbackScenario(BaseScenario):
    """A rounding compressor applied to a gradient sequence, with and without error feedback."""
    gradients: List[float]

    @property
    def result(self):
        from .solvers import ErrorFeedbackModel
        return ErrorFeedbackModel().solve(self.gradients)


class CompressionPaybackScenario(BaseScenario):
    """One synchronization compressed by ``compression_ratio`` at a codec cost of ``codec_time``.

    Compression pays when the codec costs less than the transfer it removes,
    ``codec_time < T_comm (1 - 1/r)``; ``fast_communication_time`` is the same
    reduction on a fast link, where it does not.
    """
    communication_time: Quantity
    compression_ratio: float
    codec_time: Quantity
    fast_communication_time: Quantity

    @property
    def compressed_time(self) -> Quantity:
        return self.communication_time / self.compression_ratio

    @property
    def total_time(self) -> Quantity:
        return self.compressed_time + self.codec_time

    @property
    def speedup(self) -> float:
        return (self.communication_time / self.total_time).to_base_units().magnitude

    @property
    def saved_time(self) -> Quantity:
        return self.communication_time - self.compressed_time

    def pays(self, communication_time: Quantity) -> bool:
        return self.codec_time < communication_time * (1 - 1 / self.compression_ratio)


class CompressionRunScenario(BaseScenario):
    """A run to a quality target with and without gradient compression on several networks."""
    steps: int
    compute_time: Quantity
    communication_times: Dict[str, Quantity]
    compression_ratio: float
    step_penalty: float            # extra steps compression noise costs, as a fraction

    @property
    def result(self):
        from .solvers import CompressionRunModel
        return CompressionRunModel().solve(self.steps, self.compute_time, self.communication_times,
                                           self.compression_ratio, self.step_penalty)


class GradientEncodingScenario(BaseScenario):
    """Encoded gradient payloads against full-width baselines: bits sent per original value.

    A sparsifying encoding sends ``kept_fractions[k]`` of the values, each with
    an index of ``index_bits[k]``. The ratios are upper bounds: scales and
    other metadata are left out.
    """
    baseline_bits: Dict[str, Quantity]
    value_bits: Dict[str, Quantity]
    kept_fractions: Dict[str, float] = {}
    index_bits: Dict[str, Quantity] = {}

    def bits_per_value(self, encoding: str) -> Quantity:
        kept = self.kept_fractions.get(encoding, 1.0)
        index = self.index_bits.get(encoding, 0 * self.value_bits[encoding])
        return kept * (self.value_bits[encoding] + index)

    def ratio(self, encoding: str, baseline: str) -> float:
        return (self.baseline_bits[baseline] / self.bits_per_value(encoding)).to("").magnitude


class CollectiveCallsScenario(BaseScenario):
    """Collective calls per training step of ``workload``, ``calls_per_layer`` per layer for each scheme."""
    workload: Workload
    calls_per_layer: Dict[str, int]

    def calls(self, scheme: str) -> int:
        return self.workload.layers * self.calls_per_layer[scheme]


class OverlapBudgetScenario(BaseScenario):
    """Layer-by-layer gradient AllReduce of ``workload``: sequential against overlapped with backward."""
    workload: Workload
    bytes_per_value: Quantity
    backward_per_layer: Quantity   # stated, not derived
    bucket_size: Quantity
    time_per_bucket: Quantity
    n_devices: int

    @property
    def result(self):
        from .solvers import OverlapBudgetModel
        return OverlapBudgetModel().solve(self.workload, self.bytes_per_value, self.backward_per_layer,
                                          self.bucket_size, self.time_per_bucket)


class ResidualShareScenario(BaseScenario):
    """A residual equal to ``residual`` of the one-device step, on ``devices`` devices.

    Compute divides by N and the residual does not, so the residual's share of
    the distributed step, ``r / ((1 - r)/N + r)``, is far larger than ``r``:
    an observed share must not be read as the serial fraction.
    """
    residual: float
    devices: int

    @property
    def step_fraction(self) -> float:
        """Distributed step time in units of the one-device step."""
        return (1 - self.residual) / self.devices + self.residual

    @property
    def residual_share(self) -> float:
        return self.residual / self.step_fraction

    @property
    def speedup(self) -> float:
        from ..physics.performance import calc_strong_scaling_speedup
        return calc_strong_scaling_speedup(self.devices, self.residual)


class MemoryWallScenario(BaseScenario):
    """Training-state demand of notable models against one accelerator's memory, by year."""
    models: List                     # (label, name in Literature.Series.NotableModels)
    accelerators: List[str]          # Hardware.Cloud keys
    bytes_per_param: Quantity

    @property
    def result(self):
        from .solvers import MemoryWallModel
        return MemoryWallModel().solve(self.models, self.accelerators, self.bytes_per_param)
# --- Single-machine iron law, Amdahl, and silent degradation (added 2026-10-20) ---

from ..datasets.types import DatasetProfile  # noqa: E402


class InputSpaceScenario(BaseScenario):
    """Every possible input of ``dataset``'s image shape: ``levels`` values per channel element."""
    dataset: DatasetProfile
    levels: int                     # values one 8-bit channel element can take

    @property
    def elements(self) -> int:
        """Channel elements (bytes at 8 bits) per input image."""
        d = self.dataset
        return d.image_width * d.image_height * d.image_channels

    @property
    def log10_inputs(self) -> float:
        """log10 of the number of distinct inputs, levels ** elements."""
        import math
        return self.elements * math.log10(self.levels)

    @property
    def digits(self) -> int:
        """Decimal digits in the number of distinct inputs."""
        import math
        return math.floor(self.log10_inputs) + 1

    @property
    def test_examples(self) -> int:
        """Labeled examples the benchmark evaluates (the validation split)."""
        return int(self.dataset.validation_examples.to("count").magnitude)


class ClusterTrainingTimeScenario(BaseScenario):
    """Training ``workload`` on ``n_devices`` x ``hardware`` at one end-to-end utilization (compute-bound)."""
    workload: Workload
    hardware: HardwareNode
    n_devices: int
    efficiency: float

    @property
    def training_time(self) -> Quantity:
        from ..physics.performance import dTime
        return dTime(total_ops=self.workload.training_ops, num_devices=self.n_devices,
                     peak_flops_per_device=self.hardware.compute.peak_flops,
                     efficiency_eta=self.efficiency).to("day")


class IronLawScenario(BaseScenario):
    """The iron law's three terms for one task, and an upgrade to the memory or compute term.

    Give the memory term as ``memory_time`` or as ``data_volume`` over ``bandwidth``,
    and the compute term as ``compute_time`` or as ``operations`` over ``effective_rate``
    (peak times utilization).
    """
    overhead: Quantity
    memory_time_stated: Optional[Quantity] = None
    data_volume: Optional[Quantity] = None
    bandwidth: Optional[Quantity] = None
    compute_time_stated: Optional[Quantity] = None
    operations: Optional[Quantity] = None
    effective_rate: Optional[Quantity] = None
    memory_speedup: float = 1.0
    compute_speedup: float = 1.0

    @property
    def memory_time(self) -> Quantity:
        if self.memory_time_stated is not None:
            return self.memory_time_stated
        return (self.data_volume / self.bandwidth).to(self.overhead.units)

    @property
    def compute_time(self) -> Quantity:
        if self.compute_time_stated is not None:
            return self.compute_time_stated
        return (self.operations / self.effective_rate).to(self.overhead.units)

    @property
    def upgraded_bandwidth(self) -> Quantity:
        return self.bandwidth * self.memory_speedup

    @property
    def result(self):
        from .solvers import IronLawModel
        return IronLawModel().solve(self.memory_time, self.compute_time, self.overhead,
                                    self.memory_speedup, self.compute_speedup)


class SingleNodeScenario(BaseScenario):
    """One step of ``workload`` on one ``hardware`` (MLSysIM's roofline engine), read as iron-law terms."""
    workload: Workload
    hardware: HardwareNode
    batch_size: int = 1
    precision: str = "fp16"
    efficiency: float = 0.5
    memory_speedup: float = 1.0     # an upgrade applied to the memory term in ``iron_law``

    @property
    def result(self):
        from .solvers import SingleNodeModel
        return SingleNodeModel().solve(self.workload, self.hardware, batch_size=self.batch_size,
                                       precision=self.precision, efficiency=self.efficiency)

    @property
    def memory_traffic(self) -> Quantity:
        """Bytes the memory term moves: weights plus the engine's activation estimate."""
        p = self.result
        return (p.latency_memory * p.peak_bw_actual).to("GB")

    @property
    def iron_law(self):
        """The engine's three terms summed (no overlap), before and after ``memory_speedup``."""
        from .solvers import IronLawModel
        p = self.result
        return IronLawModel().solve(p.latency_memory, p.latency_compute, p.latency_overhead,
                                    memory_speedup=self.memory_speedup)
# --- Deployment paradigms: where a model can run (added 2026-10-20) ---------
# The light barrier, data movement off the device, the roofline on one node,
# battery and heat on a phone, and splitting a pipeline across tiers.

class PropagationScenario(BaseScenario):
    """A request sent ``distance`` over fiber and answered; optionally a budget and a measured round trip."""
    distance: Quantity
    budget: Optional[Quantity] = None
    measured_round_trip: Optional[Quantity] = None

    @property
    def round_trip(self) -> Quantity:
        """Propagation floor, ``2 x distance / v_fiber`` (no routing, queuing, or compute)."""
        from ..physics.networking import calc_network_latency_ms
        return Q_(calc_network_latency_ms(self.distance), "millisecond")

    @property
    def overrun(self) -> Quantity:
        """How far the floor alone exceeds the budget (negative when it fits)."""
        return (self.round_trip - self.budget).to("millisecond")

    @property
    def measured_over_floor(self) -> float:
        return (self.measured_round_trip / self.round_trip).to("").magnitude


class MeasuredRoutesScenario(BaseScenario):
    """Measured round trips from one source to several regions, each against its fiber floor."""
    series_name: str                 # a Literature.Series table with coordinates and p50_rtt_ms

    @property
    def routes(self) -> Dict[str, PropagationScenario]:
        """Destination label -> propagation scenario over the great-circle distance."""
        from ..literature.registry import Literature
        from ..physics.networking import calc_great_circle_distance
        out = {}
        for r in getattr(Literature.Series, self.series_name).entries:
            d = calc_great_circle_distance(r["source_lat"], r["source_lon"], r["dest_lat"], r["dest_lon"])
            out[r["dest_label"]] = PropagationScenario(
                name=f"{r['source_region']} to {r['dest_region']}",
                description=f"Measured median against the fiber floor, {r['source_metro']} to {r['dest_metro']}.",
                distance=d, measured_round_trip=Q_(r["p50_rtt_ms"], "millisecond"))
        return out


class SensorUplinkScenario(BaseScenario):
    """``streams`` raw video streams against a shared link, and the event records that replace them."""
    frame_width: int
    frame_height: int
    bytes_per_pixel: Quantity
    frame_rate: Quantity
    streams: int
    link_bandwidth: Quantity
    event_size: Quantity
    event_rate: Quantity

    @property
    def stream_rate(self) -> Quantity:
        return (self.frame_width * self.frame_height * self.bytes_per_pixel * self.frame_rate).to("byte/second")

    @property
    def total_rate(self) -> Quantity:
        return (self.streams * self.stream_rate).to("byte/second")

    @property
    def link_shortfall(self) -> float:
        """Raw aggregate over what the link carries."""
        return (self.total_rate / self.link_bandwidth).to("").magnitude

    @property
    def event_bandwidth(self) -> Quantity:
        return (self.event_size * self.event_rate).to("byte/second")

    @property
    def event_reduction(self) -> float:
        """Raw aggregate over the event-record stream."""
        return (self.total_rate / self.event_bandwidth).to("").magnitude


class OffloadDeadlineScenario(BaseScenario):
    """One input sent over an uplink to a remote model, against a per-frame deadline (lower bound)."""
    payload: Quantity
    uplink_bandwidth: Quantity
    network_latency: Quantity
    remote_compute_time: Quantity
    frame_rate: Quantity
    compression: float = 1.0          # payload bytes divided by this before sending

    @property
    def transmit_time(self) -> Quantity:
        return (self.payload / self.compression / self.uplink_bandwidth).to("millisecond")

    @property
    def remote_response(self) -> Quantity:
        return (self.network_latency + self.remote_compute_time).to("millisecond")

    @property
    def remote_lower_bound(self) -> Quantity:
        """Send + round trip + remote compute; queuing and serialization excluded."""
        return (self.transmit_time + self.remote_response).to("millisecond")

    @property
    def deadline(self) -> Quantity:
        return (1 / self.frame_rate).to("millisecond")

    @property
    def deadline_ratio(self) -> float:
        return (self.remote_lower_bound / self.deadline).to("").magnitude


class SingleNodeInferenceScenario(BaseScenario):
    """One inference of ``workload`` on ``hardware`` (roofline profile from SingleNodeModel)."""
    workload: Workload
    hardware: HardwareNode
    batch_size: int = 1
    precision: Optional[str] = None   # None: the solver's default
    efficiency: Optional[float] = None

    @property
    def result(self):
        from .solvers import SingleNodeModel
        kw = {}
        if self.precision is not None:
            kw["precision"] = self.precision
        if self.efficiency is not None:
            kw["efficiency"] = self.efficiency
        return SingleNodeModel().solve(self.workload, self.hardware, batch_size=self.batch_size, **kw)

    @property
    def traffic(self) -> Quantity:
        """Bytes the model moves per inference: memory time x bandwidth."""
        p = self.result
        return (p.latency_memory * p.peak_bw_actual).to("byte")

    @property
    def intensity(self) -> Quantity:
        return (self.workload.inference_flops * self.batch_size / self.traffic).to("flop/byte")

    @property
    def ridge(self) -> Quantity:
        p = self.result
        return (p.peak_flops_actual / p.peak_bw_actual).to("flop/byte")


class AmdahlScenario(BaseScenario):
    """A sequential request path whose ``stage`` is made ``speedup`` times faster."""
    stages: Dict[str, Quantity]
    stage: str
    speedup: float

    @property
    def result(self):
        from .solvers import AmdahlModel
        return AmdahlModel().solve(self.stages, self.stage, self.speedup)


class RecallShiftScenario(BaseScenario):
    """A detector's recall before and after a shift in its inputs."""
    recall_before: float
    recall_after: float

    @property
    def miss_rate_before(self) -> float:
        return 1 - self.recall_before

    @property
    def miss_rate_after(self) -> float:
        return 1 - self.recall_after

    @property
    def miss_rate_ratio(self) -> float:
        return self.miss_rate_after / self.miss_rate_before


class FeatureUnitSkewScenario(BaseScenario):
    """One feature value computed in one unit for training and in another for serving."""
    value: Quantity
    serving_unit: str

    @property
    def serving_value(self) -> Quantity:
        return self.value.to(self.serving_unit)

    @property
    def skew_ratio(self) -> float:
        """Serving number over training number for the same quantity."""
        return self.serving_value.magnitude / self.value.magnitude


class ShiftedTestSetScenario(BaseScenario):
    """Published models scored on an original and a re-collected test set (``Literature.Series``)."""
    series_name: str
    original_key: str
    new_key: str
    gap_key: str
    baseline_prefix: str

    @property
    def rows(self):
        from ..literature.registry import Literature
        return getattr(Literature.Series, self.series_name).entries

    @property
    def result(self):
        from .solvers import ShiftedTestSetModel
        return ShiftedTestSetModel().solve(self.rows, self.original_key, self.new_key,
                                           self.gap_key, self.baseline_prefix)


class ProxyDriftScenario(BaseScenario):
    """A proxy estimate against its reference by season (``Literature.Series`` of weekly rows)."""
    series_name: str
    proxy_key: str
    reference_key: str
    first_year: int
    last_year: int
    start_week: int
    end_week: int
    first_drift_year: int
    peak_year: int

    @property
    def rows(self):
        from ..literature.registry import Literature
        return getattr(Literature.Series, self.series_name).entries

    @property
    def result(self):
        from .solvers import ProxyDriftModel
        return ProxyDriftModel().solve(self.rows, self.proxy_key, self.reference_key, self.first_year,
                                       self.last_year, self.start_week, self.end_week,
                                       self.first_drift_year, self.peak_year)


class RooflinePointScenario(BaseScenario):
    """A workload of ``intensity`` on ``hardware``'s roofline: attainable rate and share of peak.

    A measured kernel may be placed from its counter readings instead
    (``achieved_rate`` and ``achieved_bandwidth``, added 2026-10-20): its
    intensity is their ratio and ``bandwidth_utilization`` its share of peak
    memory bandwidth.
    """
    hardware: HardwareNode
    intensity: Optional[Quantity] = None
    precision: Optional[str] = None
    achieved_rate: Optional[Quantity] = None
    achieved_bandwidth: Optional[Quantity] = None

    @property
    def point_intensity(self) -> Quantity:
        """The stated intensity, or achieved rate over achieved bandwidth for a measured kernel."""
        if self.intensity is not None:
            return self.intensity
        return (self.achieved_rate / self.achieved_bandwidth).to("flop/byte")

    @property
    def bandwidth_utilization(self) -> float:
        return (self.achieved_bandwidth / self.hardware.memory.bandwidth).to("").magnitude

    @property
    def peak(self) -> Quantity:
        if self.precision is None:
            return self.hardware.compute.peak_flops
        return self.hardware.compute.precision_flops[self.precision]

    @property
    def ridge(self) -> Quantity:
        return (self.peak / self.hardware.memory.bandwidth).to("flop/byte")

    @property
    def attainable(self) -> Quantity:
        return min(self.peak.to("flop/second"),
                   (self.hardware.memory.bandwidth * self.point_intensity).to("flop/second"))

    @property
    def peak_fraction(self) -> float:
        return (self.attainable / self.peak).to("").magnitude


class PublishedRooflineScenario(BaseScenario):
    """Measured applications placed on a published roofline: a peak and the ridge where the slope meets it."""
    peak: Quantity
    ridge: float                       # operations per byte, in the source's own counting
    series_name: str                   # Literature.Series table: app, ops_per_weight_byte, measured_tops

    @property
    def apps(self) -> List[dict]:
        """Per application: intensity, measured rate, the roof above it, its share, and the binding side."""
        from ..literature.registry import Literature
        peak = self.peak.to("TOPS").magnitude
        out = []
        for r in getattr(Literature.Series, self.series_name).entries:
            roof = min(peak, r["ops_per_weight_byte"] * peak / self.ridge)
            out.append(dict(app=r["app"], family=r["family"], intensity=r["ops_per_weight_byte"],
                            measured_tops=r["measured_tops"], roof_tops=roof,
                            roof_fraction=r["measured_tops"] / roof,
                            bound="memory" if r["ops_per_weight_byte"] < self.ridge else "compute"))
        return out


class AcceleratorPodScenario(BaseScenario):
    """A pod envelope of ``pod.chips`` copies of ``accelerator``: aggregate peak and memory."""
    pod: object                        # systems PodEnvelope
    accelerator: HardwareNode

    @property
    def peak_flops(self) -> Quantity:
        return self.pod.chips * self.accelerator.compute.peak_flops

    @property
    def chip_memory_total(self) -> Quantity:
        return self.pod.chips * self.accelerator.memory.capacity


class ClockScalingScenario(BaseScenario):
    """Dynamic power after scaling the clock by ``frequency_ratio`` (P = C V^2 f)."""
    frequency_ratio: float
    voltage_tracks_frequency: bool = True

    @property
    def power_ratio(self) -> float:
        from ..physics.power import calc_dynamic_power_ratio
        return calc_dynamic_power_ratio(self.frequency_ratio, self.voltage_tracks_frequency)


class GrowthGapScenario(BaseScenario):
    """Compute doubling every ``compute_doubling_period`` against bandwidth growing ``bandwidth_growth`` a year."""
    compute_doubling_period: Quantity
    bandwidth_growth: float           # fractional growth per year
    horizon: Quantity

    @property
    def compute_annual_factor(self) -> float:
        return 2 ** (Q_(1, "year") / self.compute_doubling_period).to("").magnitude

    @property
    def bandwidth_annual_factor(self) -> float:
        return 1 + self.bandwidth_growth

    @property
    def gap(self) -> float:
        """Compute growth over bandwidth growth across the horizon."""
        years = self.horizon.to("year").magnitude
        return (self.compute_annual_factor / self.bandwidth_annual_factor) ** years


class TrendFitScenario(BaseScenario):
    """Log-linear fit of one group of a Literature series; growth per ``period`` of the x column.

    ``group_key=None`` fits every row. ``x_is_date`` reads the x column as an ISO
    date and converts it to a fractional year (added 2026-10-20).
    """
    series_name: str
    group_key: Optional[str] = None
    group: Optional[str] = None
    x_key: str
    y_key: str
    period: float = 1.0
    x_is_date: bool = False

    def _x(self, value) -> float:
        if not self.x_is_date:
            return value
        import datetime as dt
        d = dt.date.fromisoformat(value)
        return d.year + (d.timetuple().tm_yday - 1) / 365.25

    @property
    def points(self) -> List[tuple]:
        from ..literature.registry import Literature
        rows = getattr(Literature.Series, self.series_name).entries
        return [(self._x(r[self.x_key]), r[self.y_key]) for r in rows
                if self.group_key is None or r[self.group_key] == self.group]

    def crossing(self, y: float) -> float:
        """The x at which the fitted trend reaches ``y`` (added 2026-10-20)."""
        import math
        _, slope, intercept = self.fit
        return (math.log10(y) - intercept) / slope

    @property
    def fit(self) -> tuple:
        """(growth per period, slope, intercept) of log10(y) on x."""
        from ..physics.statistics import fit_exponential_growth
        pts = self.points
        return fit_exponential_growth([p[0] for p in pts], [p[1] for p in pts], self.period)

    @property
    def growth(self) -> float:
        return self.fit[0]


class TrendWindowScenario(BaseScenario):
    """Maxima of series in a Literature table over year windows, and their ratios."""
    series_name: str
    windows: Dict[str, tuple]         # window name -> (first year, year after last)

    def peak(self, series: str, window: str) -> float:
        """Largest value of ``series`` in the named window."""
        from ..literature.registry import Literature
        lo, hi = self.windows[window]
        rows = getattr(Literature.Series, self.series_name).entries
        return max(r["value"] for r in rows if r["series"] == series and lo <= r["year"] < hi)

    def growth(self, series: str, start: str, end: str) -> float:
        return self.peak(series, end) / self.peak(series, start)


class BatteryBudgetScenario(BaseScenario):
    """A continuous ``power_draw`` against a battery and a required runtime."""
    battery_energy: Quantity
    power_draw: Quantity
    required_runtime: Quantity

    @property
    def runtime(self) -> Quantity:
        return (self.battery_energy / self.power_draw).to("hour")

    @property
    def required_energy(self) -> Quantity:
        return (self.power_draw * self.required_runtime).to("Wh")

    @property
    def budget_ratio(self) -> float:
        """Energy the required runtime needs over the battery's energy."""
        return (self.required_energy / self.battery_energy).to("").magnitude


class ThermalCeilingScenario(BaseScenario):
    """A model's sustained power against a passively cooled device's ceiling, before and after a reduction."""
    power: Quantity
    reduction_factor: float
    sustained_limit: Quantity
    heating_rate: Quantity            # junction temperature rise at ``power``
    trip_time: Quantity
    trip_temperature: Quantity
    burst_rate: float                 # frames per second before throttling
    throttled_rate: float             # frames per second after

    @property
    def reduced_power(self) -> Quantity:
        return (self.power / self.reduction_factor).to("watt")

    @property
    def headroom(self) -> Quantity:
        """Ceiling minus the reduced power (zero means no margin)."""
        return (self.sustained_limit - self.reduced_power).to("watt")


class EnergyPerInferenceScenario(BaseScenario):
    """Energy per inference across tiers: solver-derived device energies plus stated anchors."""
    anchors: Dict[str, Quantity]
    runs: Dict[str, SingleNodeInferenceScenario]

    @property
    def energies(self) -> Dict[str, Quantity]:
        out = dict(self.anchors)          # stated units kept, so ratios stay exact
        out.update({k: s.result.energy.to("joule") for k, s in self.runs.items()})
        return out

    def span(self, low: str, high: str) -> float:
        e = self.energies
        return (e[high] / e[low]).to("").magnitude


class OffloadEnergyScenario(BaseScenario):
    """Energy to send ``payload`` by radio against running the model locally."""
    payload: Quantity
    transmit_energy_per_byte: Quantity
    local_run: SingleNodeInferenceScenario

    @property
    def transmit_energy(self) -> Quantity:
        return (self.payload * self.transmit_energy_per_byte).to("mJ")

    @property
    def accelerator_energy(self) -> Quantity:
        """The workload's registry inference energy (the accelerator alone)."""
        return self.local_run.workload.inference_energy

    @property
    def device_energy(self) -> Quantity:
        """The whole device over the inference window (SingleNodeModel energy)."""
        return self.local_run.result.energy.to("joule")

    @property
    def ratio_accelerator(self) -> float:
        return (self.transmit_energy / self.accelerator_energy).to("").magnitude

    @property
    def ratio_device(self) -> float:
        return (self.transmit_energy / self.device_energy).to("").magnitude


class TravelDuringLatencyScenario(BaseScenario):
    """Distance a vehicle at ``speed`` covers during ``latency``; optionally a remote-path delay range."""
    speed: Quantity
    latency: Quantity
    remote_delay_low: Optional[Quantity] = None
    remote_delay_high: Optional[Quantity] = None

    @property
    def distance(self) -> Quantity:
        return (self.speed * self.latency).to("meter")


class WakeWordFleetScenario(BaseScenario):
    """Wake-word detection for a device fleet: stream everything to the cloud, or detect on device."""
    devices: float
    audio_sample_rate: Quantity
    audio_sample_bits: Quantity
    queries_per_day: float
    query_audio: Quantity             # audio uploaded per post-wake query
    accelerator_time_per_query: Quantity
    accelerators_per_datacenter: int
    waking_hours: Quantity            # queries cluster in these hours of the day
    peak_multiplier: float            # peak over waking-hours average
    cloud_detection_cost: Quantity    # per device per year
    local_power_low: Optional[Quantity] = None    # on-device detector power range
    local_power_high: Optional[Quantity] = None
    local_detection_cost: Optional[Quantity] = None   # per device per year

    @property
    def stream_rate(self) -> Quantity:
        return (self.audio_sample_rate * self.audio_sample_bits).to("byte/second")

    @property
    def fleet_stream_rate(self) -> Quantity:
        return (self.devices * self.stream_rate).to("byte/second")

    @property
    def upload_reduction(self) -> float:
        """Audio streamed all day over audio uploaded for queries."""
        return (Q_(1, "day") / (self.queries_per_day * self.query_audio)).to("").magnitude

    @property
    def fleet_query_rate(self) -> Quantity:
        """Fleet audio after on-device detection (the stream cut by ``upload_reduction``)."""
        return (self.fleet_stream_rate / self.upload_reduction).to("byte/second")

    @property
    def cloud_detection_cost_per_year(self) -> Quantity:
        return (self.devices * self.cloud_detection_cost).to("USD/year")

    @property
    def queries_per_fleet_day(self) -> float:
        return self.devices * self.queries_per_day

    @property
    def accelerator_hours_per_day(self) -> float:
        return (self.queries_per_fleet_day * self.accelerator_time_per_query).to("hour").magnitude

    @property
    def datacenter_hours_per_day(self) -> float:
        return (self.accelerators_per_datacenter * Q_(1, "day")).to("hour").magnitude

    @property
    def datacenters_average(self) -> float:
        return self.accelerator_hours_per_day / self.datacenter_hours_per_day

    @property
    def peak_ratio(self) -> float:
        """Peak over the all-day average: the waking-hours concentration times the peak multiplier."""
        return self.peak_multiplier * (Q_(1, "day") / self.waking_hours).to("").magnitude

    @property
    def datacenters_peak(self) -> float:
        return self.datacenters_average * self.peak_ratio


class PipelineSpeedupScenario(BaseScenario):
    """Serial pipeline stages; one stage is made ``speedup`` times faster (Amdahl's law)."""
    stages: Dict[str, Quantity]
    accelerated: str
    speedup: float

    @property
    def total(self) -> Quantity:
        return sum(self.stages.values(), Q_(0, "millisecond")).to("millisecond")

    @property
    def fraction(self) -> float:
        """Share of the total in the accelerated stage."""
        return (self.stages[self.accelerated] / self.total).to("").magnitude

    @property
    def accelerated_stage(self) -> Quantity:
        return (self.stages[self.accelerated] / self.speedup).to("millisecond")

    @property
    def accelerated_total(self) -> Quantity:
        return (self.total - self.stages[self.accelerated] + self.accelerated_stage).to("millisecond")

    @property
    def overall_speedup(self) -> float:
        from ..physics.performance import calc_amdahls_speedup
        return calc_amdahls_speedup(self.fraction, self.speedup)

    @property
    def limit(self) -> float:
        """Speedup with an infinitely fast accelerated stage."""
        from ..physics.performance import calc_amdahls_speedup
        return calc_amdahls_speedup(self.fraction, float("inf"))


class SpeedStepScenario(BaseScenario):
    """Raising the speed setpoint from ``base`` to ``raised`` (two stopping budgets at one deceleration)."""
    base: StoppingBudgetScenario
    raised: StoppingBudgetScenario

    @property
    def result(self):
        from .solvers import SpeedStepModel
        return SpeedStepModel().solve(self.base.speed, self.raised.speed, self.base.a_brake)


class FreshnessProfileScenario(BaseScenario):
    """A target moving at ``site``'s conveyor speed, grasped through a staged capture-to-onset pipeline."""
    site: Any                               # EmbodiedSiteScenario (conveyor speed, tolerance, initial error)
    stages: List[Tuple[str, Quantity, Quantity, Quantity]]   # (stage, P50, P99, max)
    focus_stage: str                        # the stage a benchmark reports
    system_p99: Quantity
    system_p999: Quantity

    @property
    def result(self):
        from .solvers import FreshnessProfileModel
        return FreshnessProfileModel().solve(self.site.v_conveyor, self.site.grasp_tolerance,
                                             self.site.grasp_initial_error, self.stages, self.focus_stage,
                                             self.system_p999)


class BrakingCeilingsScenario(BaseScenario):
    """The ceilings on ``site``'s credible deceleration for its loaded machine at the drive limit."""
    site: Any                               # EmbodiedSiteScenario

    @property
    def result(self):
        from .solvers import BrakingCeilingsModel
        rm = self.site.machine
        return BrakingCeilingsModel().solve(
            rm.loaded_mass, self.site.a_brake, rm.base.max_velocity, rm.drive_unit, rm.drive_wheel_count,
            rm.drive_current_limit, rm.drive_wheel_load_share_max, self.site.mu_dry, self.site.mu_inspected_floor,
            rm.drive_bus_battery_low, rm.cg_height, rm.tip_arm_stowed)


class HubDriveElectricalScenario(BaseScenario):
    """Current rise and phase-short braking of ``site.machine``'s hub drives on a depleted bus."""
    site: Any                               # EmbodiedSiteScenario

    @property
    def result(self):
        from .solvers import HubDriveElectricalModel
        rm = self.site.machine
        return HubDriveElectricalModel().solve(rm.drive_unit, rm.drive_current_limit, rm.drive_bus_battery_low,
                                               self.site.t_brake_onset)


class GearedRotorScenario(BaseScenario):
    """A rotor seen through a ``gear_ratio`` reduction, beside a link of ``load_inertia``."""
    gear_ratio: float
    rotor_mass: Quantity
    rotor_inertia: Quantity
    flywheel_radius: Quantity               # radius of the comparison solid flywheel
    load_inertia: Quantity

    @property
    def result(self):
        from .solvers import GearedRotorModel
        return GearedRotorModel().solve(self.gear_ratio, self.rotor_mass, self.rotor_inertia,
                                        self.flywheel_radius, self.load_inertia)


class GearRatioCostScenario(BaseScenario):
    """``motor`` in ``joint`` holding ``output_torque`` through each of ``ratios``."""
    joint: Any                              # ActuatorJoint (loss resistance, input inertia, gear ratio)
    motor: Any                              # FramelessMotor
    output_torque: Quantity
    gear_efficiency: float
    ratios: List[int]

    @property
    def result(self):
        from .solvers import GearRatioCostModel
        return GearRatioCostModel().solve(self.motor, self.joint.internal_resistance, self.joint.rotor_inertia,
                                          self.output_torque, self.gear_efficiency, self.ratios)


class ThermalBurstScenario(BaseScenario):
    """A ``burst_factor`` times rated-current burst in ``joint``, first-order lumped, fixed resistance."""
    joint: Any                              # ActuatorJoint (loss resistance)
    motor: Any                              # FramelessMotor (rated current)
    thermal_capacitance: Quantity
    thermal_resistance: Quantity
    ambient: Quantity
    limit: Quantity                         # insulation rating
    ceiling: Quantity                       # working ceiling below the rating
    burst_factor: float
    burst_on: Quantity

    def temperature(self, elapsed: Quantity, start: Quantity) -> Quantity:
        """Winding temperature ``elapsed`` into the burst from ``start``."""
        from ..physics.motors import calc_first_order_temperature
        r = self.result
        return calc_first_order_temperature(elapsed, start, r.asymptote, r.time_constant)

    @property
    def result(self):
        from .solvers import ThermalBurstModel
        return ThermalBurstModel().solve(self.motor.rated_current, self.joint.internal_resistance,
                                         self.thermal_capacitance, self.thermal_resistance, self.ambient,
                                         self.limit, self.ceiling, self.burst_factor, self.burst_on)


class StallDeratingScenario(BaseScenario):
    """``joint`` holding ``hold_torque`` at stall with copper's resistance rising with temperature."""
    joint: Any                              # ActuatorJoint (gear ratio, loss resistance)
    motor: Any                              # FramelessMotor (torque constant)
    hold_torque: Quantity
    gear_efficiency: float
    thermal_resistance: Quantity
    ambient: Quantity
    limit: Quantity

    @property
    def result(self):
        from .solvers import StallDeratingModel
        return StallDeratingModel().solve(self.hold_torque, self.joint.gear_ratio, self.gear_efficiency,
                                          self.motor.torque_constant, self.joint.internal_resistance,
                                          self.thermal_resistance, self.ambient, self.limit)


class ContinuousDeratingScenario(BaseScenario):
    """A catalog continuous rating carried to a hotter ambient and a worse heat path."""
    ceiling: Quantity
    catalog_ambient: Quantity
    installed_ambient: Quantity
    thermal_resistance_factor: float        # installed theta relative to the catalog's

    @property
    def result(self):
        from .solvers import ContinuousDeratingModel
        return ContinuousDeratingModel().solve(self.ceiling, self.catalog_ambient, self.installed_ambient,
                                               self.thermal_resistance_factor)


class DriveBusScenario(BaseScenario):
    """``site.machine``'s shared drive bus: a coincident load step, and a stop from the drive limit into it."""
    site: Any                               # EmbodiedSiteScenario (load step, deceleration)
    link_capacitance: Quantity
    regen_fraction: float                   # share of kinetic energy returned to the bus
    candidate_resistor: Quantity
    candidate_power: Quantity

    @property
    def result(self):
        from .solvers import DriveBusModel
        rm = self.site.machine
        return DriveBusModel().solve(
            rm.loaded_mass, rm.base.max_velocity, self.site.a_brake, self.site.i_rail_transient,
            rm.drive_bus_resistance, rm.drive_bus_full_charge, rm.drive_bus_nominal, rm.drive_bus_battery_low,
            rm.drive_bus_uvlo, rm.brain_soc.tdp, rm.drive_bus_converter_efficiency, self.link_capacitance,
            rm.drive_bus_clamp, self.regen_fraction)


class ZeroFailureScenario(BaseScenario):
    """``trials`` failure-free trials, bounded at one-sided ``confidence``."""
    trials: int
    confidence: float
    target_rate: Optional[float] = None     # a per-trial rate the campaign is meant to support

    @property
    def result(self):
        from .solvers import ZeroFailureModel
        return ZeroFailureModel().solve(self.trials, self.confidence)


class ThrottledStageScenario(BaseScenario):
    """``site``'s vision backbone at ``clock_ratio`` of full clock, paid as travel at ``speed``."""
    site: Any                               # EmbodiedSiteScenario (backbone time)
    clock_ratio: float
    speed: Quantity

    @property
    def result(self):
        from .solvers import ThrottledStageModel
        return ThrottledStageModel().solve(self.site.t_backbone, self.clock_ratio, self.speed)


# --- Fleet orchestration: queueing, packing, interruption, and sharing (added 2026-10-20) ---

def _usd_per_hour(rate) -> float:
    """A GPU-hour price as a plain number of dollars per hour."""
    return rate.to("USD/hour").magnitude


class QueueWaitScenario(BaseScenario):
    """A fleet modeled as one M/G/1 server at offered load ``utilization``.

    The mean queue wait, as a multiple of the mean job, is a load factor
    ``rho/(1-rho)`` times a variability factor ``(1 + C_s^2)/2``. The model
    ignores gang packing and parallel admission; it isolates the effect of
    load and of spread in job durations.
    """
    utilization: float
    service_cv: float                       # coefficient of variation of job duration

    @property
    def load_factor(self) -> float:
        return self.utilization / (1 - self.utilization)

    @property
    def variability_factor(self) -> float:
        return (1 + self.service_cv ** 2) / 2

    @property
    def wait_ratio(self) -> float:
        """Mean queue wait divided by the mean job duration."""
        from ..physics.serving import calc_mg1_wait_ratio
        return calc_mg1_wait_ratio(self.utilization, self.service_cv)


class JobSizeMixScenario(BaseScenario):
    """A cluster trace's jobs and GPU time split at ``large_threshold`` GPUs per job."""
    series: Any                             # TrendSeries with gpu_num, jobs, gpu_seconds rows
    large_threshold: int

    @property
    def result(self):
        from .solvers import JobSizeMixModel
        return JobSizeMixModel().solve(self.series.entries, self.large_threshold)


class DominantShareScenario(BaseScenario):
    """Dominant-resource fairness by progressive filling of indivisible ``bundles`` on ``capacity``.

    Resources are accelerators, CPU cores, memory (GB), and network (Gb/s).
    Each tenant's dominant share is its largest share of any one resource;
    progressive filling grants the next bundle to the tenant with the lowest
    dominant share until no bundle fits.
    """
    capacity: Dict[str, float]
    bundles: Dict[str, Dict[str, float]]

    @property
    def result(self) -> Dict[str, Any]:
        from ..labs.experiments.v2_08_experiments import dominant_resource_fairness
        return dominant_resource_fairness(self.capacity, self.bundles)


class NodePackingScenario(BaseScenario):
    """Jobs of ``job_gpus`` GPUs packed on ``nodes`` nodes of ``gpus_per_node``.

    A job must sit in one node, so each node runs ``gpus_per_node //
    job_gpus`` jobs and strands the remainder. ``free_slots`` (optional)
    records a fragmented snapshot: free GPUs scattered so that no node is
    whole.
    """
    nodes: int
    gpus_per_node: int
    job_gpus: int
    free_slots: Optional[int] = None

    @property
    def total_gpus(self) -> int:
        return self.nodes * self.gpus_per_node

    @property
    def stranded_per_node(self) -> int:
        return self.gpus_per_node - (self.gpus_per_node // self.job_gpus) * self.job_gpus

    @property
    def usable_share(self) -> float:
        return 1 - self.stranded_per_node / self.gpus_per_node

    @property
    def free_share(self) -> float:
        return self.free_slots / self.total_gpus


class IdleCapacityCostScenario(BaseScenario):
    """``gpus`` accelerators held idle for ``duration`` at ``device_hour_rate``.

    ``gang_gpus`` (optional) is the gang the held GPUs belong to, for a
    partial gang that waits for its missing members.
    """
    gpus: int
    duration: Quantity
    device_hour_rate: Quantity
    gang_gpus: Optional[int] = None

    @property
    def gpu_hours(self) -> float:
        return self.gpus * self.duration.to("hour").magnitude

    @property
    def cost(self) -> float:
        """USD."""
        return self.gpu_hours * _usd_per_hour(self.device_hour_rate)

    @property
    def missing_gpus(self) -> int:
        return self.gang_gpus - self.gpus


class UtilizationGainScenario(BaseScenario):
    """A fleet of ``fleet_gpus`` raised from ``utilization_before`` to ``utilization_after``.

    The gain in fully active GPU-equivalents is ``N (u_after - u_before)``.
    Buying that capacity under the old policy would take ``gain / u_before``
    raw GPUs, which is what the gain is worth. ``pools`` and ``pool_metrics``
    optionally record the GPU pools and each pool's measured shares.
    """
    fleet_gpus: int
    utilization_before: float
    utilization_after: float
    device_hour_rate: Quantity
    pools: Dict[str, int] = {}                          # GPUs per pool
    pool_metrics: Dict[str, Dict[str, float]] = {}      # per pool: allocated/active, before/after
    pending_jobs: Optional[int] = None                  # queue depth before the fix
    job_memory_need: Optional[Quantity] = None          # peak memory of the jobs queued on the largest pool

    @property
    def improvement(self) -> float:
        return self.utilization_after - self.utilization_before

    @property
    def productive_gain(self) -> float:
        return self.fleet_gpus * self.improvement

    @property
    def raw_gpu_equivalent(self) -> float:
        return self.productive_gain / self.utilization_before

    @property
    def annual_value(self) -> float:
        """USD a year for the raw-GPU equivalent at the scenario rate."""
        from ..core.units import HOURS_PER_YEAR
        return self.raw_gpu_equivalent * _usd_per_hour(self.device_hour_rate) * HOURS_PER_YEAR


class PreemptionTaxScenario(BaseScenario):
    """Forced preemptions of an ``n_devices`` job whose checkpoint is ``checkpoint_bytes``."""
    n_devices: int
    checkpoint_bytes: Quantity
    lost_work: Quantity
    preemption_rate: Quantity               # forced preemptions per unit time
    recovery: Any                           # RecoveryProfile (Systems.Reliability.Recovery)
    device_hour_rate: Quantity

    @property
    def result(self):
        from .solvers import PreemptionTaxModel
        return PreemptionTaxModel().solve(self.n_devices, self.checkpoint_bytes, self.lost_work,
                                          self.preemption_rate, self.recovery, self.device_hour_rate)


class SpotEffectiveCostScenario(BaseScenario):
    """Spot capacity priced per hour of retained progress.

    Each interruption bills ``checkpoint_overhead + restart_overhead`` that
    retain no progress, once per ``retained_interval`` of progress, so the
    effective price is the spot price times billed time over retained time.
    """
    on_demand_rate: Quantity
    spot_rate: Quantity
    checkpoint_overhead: Quantity
    restart_overhead: Quantity
    retained_interval: Quantity

    @property
    def nominal_discount(self) -> float:
        return 1 - _usd_per_hour(self.spot_rate) / _usd_per_hour(self.on_demand_rate)

    @property
    def lost_per_interruption(self) -> Quantity:
        return (self.checkpoint_overhead + self.restart_overhead).to("minute")

    @property
    def billed_over_retained(self) -> float:
        interval = self.retained_interval.to("minute").magnitude
        return (interval + self.lost_per_interruption.magnitude) / interval

    @property
    def effective_rate(self) -> float:
        """USD per GPU-hour of retained progress."""
        return _usd_per_hour(self.spot_rate) * self.billed_over_retained

    @property
    def effective_cost_ratio(self) -> float:
        """Effective spot price over the on-demand price."""
        return self.effective_rate / _usd_per_hour(self.on_demand_rate)

    @property
    def effective_savings(self) -> float:
        return 1 - self.effective_cost_ratio


class SpotRunScenario(BaseScenario):
    """A training run of ``run_duration`` on ``fleet``, on demand against spot capacity.

    Spot adds ``checkpoint_overhead_fraction`` of the run for checkpoints and
    ``restart_per_interruption`` for each of ``interruptions_per_day`` per day
    of the run.
    """
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
        return self.fleet.total_accelerators

    @property
    def run_hours(self) -> float:
        return self.run_duration.to("hour").magnitude

    @property
    def run_days(self) -> float:
        return self.run_duration.to("day").magnitude

    @property
    def on_demand_cost(self) -> float:
        return self.gpus * self.run_hours * _usd_per_hour(self.on_demand_rate)

    @property
    def spot_base_cost(self) -> float:
        return self.gpus * self.run_hours * _usd_per_hour(self.spot_rate)

    @property
    def checkpoint_extra_hours(self) -> float:
        return self.checkpoint_overhead_fraction * self.run_hours

    @property
    def interruptions(self) -> float:
        return self.interruptions_per_day * (self.run_hours // 24)

    @property
    def restart_total_hours(self) -> float:
        return self.interruptions * self.restart_per_interruption.to("hour").magnitude

    @property
    def spot_total_hours(self) -> float:
        return self.run_hours + self.checkpoint_extra_hours + self.restart_total_hours

    @property
    def spot_total_cost(self) -> float:
        return self.gpus * self.spot_total_hours * _usd_per_hour(self.spot_rate)

    @property
    def savings(self) -> float:
        return self.on_demand_cost - self.spot_total_cost

    @property
    def savings_share(self) -> float:
        return self.savings / self.on_demand_cost

    @property
    def longer_share(self) -> float:
        return self.spot_total_hours / self.run_hours - 1


class ElasticResizeScenario(BaseScenario):
    """Three ways to run one job over ``window``: start small, wait for the full size, or step up.

    Start small runs ``small_gpus`` for the whole window. Wait queues
    ``queue_wait`` for ``full_gpus``. Step up starts on ``small_gpus`` and,
    when the full allocation would free, pauses ``rescale_overhead`` and grows
    to ``step_up_gpus``, leaving the rest to queued jobs. Throughputs are
    epochs per hour at each size. GPU-hours count the step-up size through
    its resize pause.
    """
    window: Quantity
    queue_wait: Quantity
    rescale_overhead: Quantity
    small_gpus: int
    step_up_gpus: int
    full_gpus: int
    small_throughput: float
    step_up_throughput: float
    full_throughput: float

    def _h(self, q) -> float:
        return q.to("hour").magnitude

    @property
    def window_h(self) -> float:
        return self._h(self.window)

    @property
    def wait_h(self) -> float:
        return self._h(self.queue_wait)

    @property
    def rescale_h(self) -> float:
        return self.rescale_overhead.to("minute").magnitude / 60

    @property
    def start_small_epochs(self) -> float:
        return self.window_h * self.small_throughput

    @property
    def wait_run_h(self) -> float:
        return self.window_h - self.wait_h

    @property
    def wait_epochs(self) -> float:
        return self.wait_run_h * self.full_throughput

    @property
    def step_up_phase1_epochs(self) -> float:
        return self.wait_h * self.small_throughput

    @property
    def step_up_phase2_h(self) -> float:
        return self.window_h - self.wait_h - self.rescale_h

    @property
    def step_up_phase2_epochs(self) -> float:
        return self.step_up_phase2_h * self.step_up_throughput

    @property
    def step_up_epochs(self) -> float:
        return self.step_up_phase1_epochs + self.step_up_phase2_epochs

    @property
    def step_up_trails_wait(self) -> float:
        return (self.wait_epochs - self.step_up_epochs) / self.wait_epochs

    @property
    def step_up_over_small(self) -> float:
        return self.step_up_epochs / self.start_small_epochs

    @property
    def rescale_break_even_h(self) -> float:
        """Resize overhead at which stepping up only matches starting small."""
        return self.window_h - self.wait_h - (self.start_small_epochs - self.step_up_phase1_epochs) / self.step_up_throughput

    @property
    def step_up_efficiency(self) -> float:
        return self.step_up_throughput / (self.step_up_gpus / self.small_gpus * self.small_throughput)

    @property
    def full_efficiency(self) -> float:
        return self.full_throughput / (self.full_gpus / self.small_gpus * self.small_throughput)

    @property
    def gpu_hours(self) -> Dict[str, float]:
        return {"start_small": self.small_gpus * self.window_h,
                "wait": self.full_gpus * self.wait_run_h,
                "step_up": self.small_gpus * self.wait_h + self.step_up_gpus * self.wait_run_h}

    @property
    def epochs_per_gpu_hour(self) -> Dict[str, float]:
        epochs = {"start_small": self.start_small_epochs, "wait": self.wait_epochs, "step_up": self.step_up_epochs}
        return {k: epochs[k] / h for k, h in self.gpu_hours.items()}


class ServingConsolidationScenario(BaseScenario):
    """Small and large serving models on whole devices, exclusive against shared in pairs.

    Each small model occupies ``small_footprint`` (weights, KV cache, and
    activations) of a ``device_memory`` budget; ``tenants`` small models share
    one device when load tests allow. Each large model takes
    ``gpus_per_large`` devices for tensor parallelism either way.
    """
    device_memory: Quantity                 # branded decimal budget per device
    partition_memory: Optional[Quantity] = None   # a hardware partition (MIG profile) for comparison
    small_model: Workload
    small_footprint: Quantity
    tenants: int
    n_small: int
    large_model: Workload
    n_large: int
    gpus_per_large: int
    bytes_per_param: Quantity
    device_hour_rate: Quantity

    def _ratio(self, a, b) -> float:
        return (a / b).to("").magnitude

    @property
    def exclusive_unused(self) -> float:
        return self._ratio(self.device_memory - self.small_footprint, self.device_memory)

    @property
    def shared_footprint(self) -> Quantity:
        return self.tenants * self.small_footprint

    @property
    def shared_unused(self) -> float:
        return self._ratio(self.device_memory - self.shared_footprint, self.device_memory)

    @property
    def partition_unused(self) -> float:
        return self._ratio(self.partition_memory - self.small_footprint, self.partition_memory)

    @property
    def large_weights(self) -> Quantity:
        return self.large_model.parameters * self.bytes_per_param

    @property
    def min_devices_large(self) -> int:
        import math
        return math.ceil(self._ratio(self.large_weights, self.device_memory))

    @property
    def large_gpus(self) -> int:
        return self.n_large * self.gpus_per_large

    @property
    def exclusive_gpus(self) -> int:
        return self.n_small + self.large_gpus

    @property
    def shared_small_gpus(self) -> int:
        return self.n_small // self.tenants

    @property
    def shared_gpus(self) -> int:
        return self.shared_small_gpus + self.large_gpus

    @property
    def saved_gpus(self) -> int:
        return self.exclusive_gpus - self.shared_gpus

    @property
    def saved_share(self) -> float:
        return self.saved_gpus / self.exclusive_gpus

    @property
    def exclusive_utilization(self) -> float:
        return self._ratio(self.small_footprint, self.device_memory)

    @property
    def shared_utilization(self) -> float:
        return self._ratio(self.shared_footprint, self.device_memory)

    @property
    def annual_value(self) -> float:
        """USD a year for the released GPUs at the scenario rate."""
        from ..core.units import HOURS_PER_YEAR
        return self.saved_gpus * _usd_per_hour(self.device_hour_rate) * HOURS_PER_YEAR


class DiurnalLendingScenario(BaseScenario):
    """A fleet split between a serving reservation and a training base, static against lent.

    The reservation is the forecast serving peak plus ``surge_buffer``.
    Lending gives training whatever the reservation holds above average
    demand plus the buffer; the GPUs left after the reservation and the
    training base are maintenance headroom.
    """
    fleet_gpus: int
    serving_reservation: int
    serving_average: int
    surge_buffer: int
    training_base: int
    device_hour_rate: Quantity

    @property
    def peak_demand(self) -> int:
        return self.serving_reservation - self.surge_buffer

    @property
    def serving_reclaimable(self) -> int:
        return self.serving_reservation - self.serving_average

    @property
    def lent(self) -> int:
        """GPUs lent to training on average."""
        return self.serving_reclaimable - self.surge_buffer

    @property
    def headroom(self) -> int:
        return self.fleet_gpus - self.serving_reservation - self.training_base

    @property
    def static_active(self) -> int:
        return self.serving_average + self.training_base

    @property
    def dynamic_active(self) -> int:
        return self.static_active + self.lent

    @property
    def static_utilization(self) -> float:
        return self.static_active / self.fleet_gpus

    @property
    def dynamic_utilization(self) -> float:
        return self.dynamic_active / self.fleet_gpus

    @property
    def annual_value(self) -> float:
        """USD a year for the lent GPUs at the scenario rate."""
        from ..core.units import HOURS_PER_YEAR
        return self.lent * _usd_per_hour(self.device_hour_rate) * HOURS_PER_YEAR


class QuotaLendingScenario(BaseScenario):
    """A team guaranteed ``guarantee`` GPUs that uses ``used_share`` of them; the rest can be lent.

    Lent GPUs stay reclaimable: borrowers run them as lower-priority,
    preemptible work and return them when the owner's demand comes back.
    """
    guarantee: int
    used_share: float
    borrower_guarantee: Optional[int] = None

    @property
    def used(self) -> int:
        return round(self.guarantee * self.used_share)

    @property
    def lendable(self) -> int:
        return self.guarantee - self.used


class SelfContainedReplicaScenario(BaseScenario):
    """The smallest model replica of ``workload`` whose training state fits without sharding.

    Tensor parallelism spans one node of ``fleet``; the pipeline depth is the
    smallest power of two at which each accelerator's share of the
    ``bytes_per_param`` training state fits its memory. An elastic job that
    keeps this layout fixed grows or shrinks one whole replica at a time.
    """
    workload: Workload
    fleet: Fleet
    bytes_per_param: Quantity

    @property
    def tensor_parallel(self) -> int:
        return self.fleet.node.accelerators_per_node

    @property
    def pipeline_depth(self) -> int:
        hbm = self.fleet.node.accelerator.memory.capacity
        state = self.workload.parameters.magnitude * self.bytes_per_param
        pp = 1
        while (state / (self.tensor_parallel * pp)).to("GiB") > hbm:
            pp *= 2
        return pp

    @property
    def replica_gpus(self) -> int:
        return self.tensor_parallel * self.pipeline_depth
# --- ML lifecycle: screening requirements, late limits, iteration, deployment (added 2026-10-20) ---

class ScreeningCountsScenario(BaseScenario):
    """A screener's decisions on ``sick`` and ``healthy`` patients: sensitivity, specificity, accuracy.

    Sensitivity counts only sick patients and specificity only healthy ones, so a
    mostly healthy population cannot inflate either; accuracy pools both.
    """
    sick: int
    healthy: int
    true_positives: int
    true_negatives: int

    @property
    def population(self) -> int:
        return self.sick + self.healthy

    @property
    def sensitivity(self) -> float:
        return self.true_positives / self.sick

    @property
    def specificity(self) -> float:
        return self.true_negatives / self.healthy

    @property
    def accuracy(self) -> float:
        return (self.true_positives + self.true_negatives) / self.population

    @property
    def missed(self) -> int:
        """Sick patients the screener cleared (false negatives)."""
        return self.sick - self.true_positives


class ImageYieldScenario(BaseScenario):
    """``attempted`` captures, of which ``usable`` produced a result: the usable and rejected shares."""
    attempted: int
    usable: int

    @property
    def rejected(self) -> int:
        return self.attempted - self.usable

    @property
    def usable_share(self) -> float:
        return self.usable / self.attempted

    @property
    def rejected_share(self) -> float:
        return self.rejected / self.attempted


class ScreeningRequirementsScenario(BaseScenario):
    """A screener's release floors and device budget, with measured slices checked against the floor.

    ``device`` supplies the power envelope; ``slice_sensitivity`` maps a test slice
    (all photos, one patient subgroup) to its measured sensitivity; a release passes
    only if every slice meets the floor. ``field_operating_point`` is an optional
    (specificity, sensitivity) pair for one threshold on field data.
    """
    sensitivity_floor: float
    specificity_floor: float
    memory_budget: Quantity
    device: HardwareNode
    slice_sensitivity: Dict[str, float] = {}
    field_operating_point: Optional[Tuple[float, float]] = None

    @property
    def power_low(self) -> Quantity:
        return self.device.tdp_min

    @property
    def power_high(self) -> Quantity:
        return self.device.tdp_max

    def passes(self, slice_name: str) -> bool:
        return self.slice_sensitivity[slice_name] >= self.sensitivity_floor

    @property
    def failing_slices(self) -> List[str]:
        return [k for k in self.slice_sensitivity if not self.passes(k)]


class LateConstraintScenario(BaseScenario):
    """A model built without its deployment limit, the day the limit surfaces, and the repair it forces.

    ``validation_sensitivity`` maps a day to the sensitivity reached that day.
    ``repair_tasks`` lists the redone artifacts in order; each waits for the one
    before, so the repair takes their sum when staff and compute are free.
    """
    model_memory: Quantity
    memory_budget: Quantity
    device_memory_low: Quantity
    device_memory_high: Quantity
    request_day: int
    validation_sensitivity: Dict[int, float]
    handoff_day: int
    discovery_day: int
    check_day: int
    discard_day: int
    repair_tasks: Dict[str, Quantity]

    @property
    def overrun_ratio(self) -> float:
        """Model memory over the budget it had to fit."""
        return (self.model_memory / self.memory_budget).to("").magnitude

    @property
    def work_discarded(self) -> Quantity:
        """Elapsed time from the request to the day the work is thrown away."""
        return Q_(self.discard_day - self.request_day, "day")

    @property
    def repair_elapsed(self) -> Quantity:
        total = Q_(0, "day")
        for duration in self.repair_tasks.values():
            total = total + duration
        return total.to("day")


class IterationBudgetScenario(BaseScenario):
    """Two teams in one development window: a slow cycle against a fast one, with assumed gains per run.

    Accuracies are in percent and gains in points per run. The slow team learns
    from every run; the fast team can run far more than it can learn from, so
    ``fast_effective_runs`` caps the runs that count. Both stop at ``ceiling``.
    """
    window: Quantity
    slow_cycle: Quantity
    slow_start: float
    slow_gain: float
    fast_cycle: Quantity
    fast_start: float
    fast_gain: float
    fast_effective_runs: int
    ceiling: float

    @property
    def slow_runs(self) -> float:
        return (self.window / self.slow_cycle).to("").magnitude

    @property
    def slow_final(self) -> float:
        return min(self.slow_start + self.slow_runs * self.slow_gain, self.ceiling)

    @property
    def fast_possible_runs(self) -> float:
        return (self.window / self.fast_cycle).to("").magnitude

    @property
    def fast_uncapped(self) -> float:
        return self.fast_start + self.fast_effective_runs * self.fast_gain

    @property
    def fast_final(self) -> float:
        return min(self.fast_uncapped, self.ceiling)

    @property
    def ideas_ratio(self) -> float:
        """Experiments the fast team learns from per experiment the slow team runs."""
        return self.fast_effective_runs / self.slow_runs

    @property
    def final_gap(self) -> float:
        """Points between the two final accuracies."""
        return abs(self.fast_final - self.slow_final)


class InputResizeScenario(BaseScenario):
    """An image resized to a model's square input: every length scales by target over source height.

    ``feature_sizes`` are widths, in source pixels, of the smallest features that
    must survive (measured on the image).
    """
    source_width: int
    source_height: int
    target_size: int
    feature_sizes: List[float]

    @property
    def scale(self) -> float:
        return self.target_size / self.source_height

    def resized(self, length_px: float) -> float:
        return length_px * self.scale

    @property
    def feature_mid(self) -> float:
        return (min(self.feature_sizes) + max(self.feature_sizes)) / 2

    @property
    def resized_features(self) -> List[float]:
        return [self.resized(f) for f in self.feature_sizes]


class DailyUploadScenario(BaseScenario):
    """A site's daily captures sent over its uplink inside a working window, against compact summaries."""
    cases_per_day: int
    files_per_case: int
    file_size: Quantity
    uplink: Quantity
    window: Quantity
    summary_size: Quantity          # one summary per case

    @property
    def daily_volume(self) -> Quantity:
        return self.cases_per_day * self.files_per_case * self.file_size

    @property
    def upload_time(self) -> Quantity:
        from ..physics.quantities import transfer_time
        return transfer_time(self.daily_volume, self.uplink)

    @property
    def window_share(self) -> float:
        """Upload time as a share of the working window."""
        return (self.upload_time / self.window).to("").magnitude

    @property
    def file_upload_time(self) -> Quantity:
        from ..physics.quantities import transfer_time
        return transfer_time(self.file_size, self.uplink)

    @property
    def summary_volume(self) -> Quantity:
        return self.cases_per_day * self.summary_size

    @property
    def reduction(self) -> float:
        """Raw daily volume over the summaries that replace it."""
        return (self.daily_volume / self.summary_volume).to("").magnitude


class CloudEdgeCostScenario(BaseScenario):
    """Serving a network of sites from the cloud (pay per image) or from a device at each site (buy up front).

    Undiscounted; ``payback`` is the up-front device cost over the yearly running
    cost the devices save. ``image_size``, ``uplink``, and the two latencies are
    optional, for comparing response times.
    """
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
        return self.sites * self.cases_per_site_day * self.images_per_case * self.days_per_year

    @property
    def cloud_inference_annual(self) -> Quantity:
        return self.annual_images * self.cloud_cost_per_image / Q_(1, "year")

    @property
    def cloud_annual(self) -> Quantity:
        return self.cloud_inference_annual + self.cloud_fixed_annual

    @property
    def edge_capex(self) -> Quantity:
        return self.sites * self.edge_device_cost

    @property
    def edge_inference_annual(self) -> Quantity:
        return self.annual_images * self.edge_cost_per_image / Q_(1, "year")

    @property
    def edge_annual(self) -> Quantity:
        return self.edge_maintenance_annual + self.edge_inference_annual

    @property
    def edge_first_year(self) -> Quantity:
        return self.edge_capex + self.edge_annual * Q_(1, "year")

    @property
    def annual_savings(self) -> Quantity:
        return self.cloud_annual - self.edge_annual

    @property
    def payback(self) -> Quantity:
        from ..physics.economics import calc_payback_period
        return calc_payback_period(self.edge_capex, self.annual_savings)

    @property
    def cloud_upload_time(self) -> Quantity:
        """Time to send one image over the site uplink, before any remote processing."""
        from ..physics.quantities import transfer_time
        return transfer_time(self.image_size, self.uplink)


class MonitoringCadenceScenario(BaseScenario):
    """Production signals reviewed on different clocks, each as long as the signal takes to trust."""
    cadences: Dict[str, Quantity]   # signal clock label -> review interval

    @property
    def minutes(self) -> Dict[str, float]:
        return {k: q.to("minute").magnitude for k, q in self.cadences.items()}

    @property
    def span_orders(self) -> float:
        """Orders of magnitude from the fastest to the slowest cadence."""
        import math
        m = self.minutes.values()
        return math.log10(max(m) / min(m))
# --- Test-time deliberation (added 2026-10-20) --------------------------------
# Spending more work on one decision with the weights fixed: one output
# ceiling as depth, breadth, or feedback; pools of independent candidates; a
# fallible check over a pool; argmax selection on a noisy score; search trees,
# beams, and policies under one deadline. Times use the decode floor of a
# DecodeScenario (memory_time, P b / BW per output token). Prices are Pint
# quantities such as Q_(3.00, "USD / Mtoken").

class ProbeVsAnswerScenario(BaseScenario):
    """A generated explanation of ``answer_tokens`` against a prepared test of ``probe_time``."""
    floors: DecodeScenario
    answer_tokens: int
    probe_time: Quantity
    output_price: Quantity

    @property
    def result(self):
        from .solvers import ProbeVsAnswerModel
        return ProbeVsAnswerModel().solve(self.floors.result.memory_time, self.answer_tokens,
                                          self.probe_time, self.output_price)


class AllocationAxesScenario(BaseScenario):
    """One output ``ceiling`` spent as one chain, ``candidates`` streams, or ``rounds`` of feedback."""
    floors: DecodeScenario
    prompt: int
    ceiling: int
    candidates: int                  # breadth: concurrent streams, each billed its own prompt
    candidate_tokens: int            # generated tokens per breadth or feedback candidate
    rounds: int                      # feedback rounds
    report_tokens: int               # check report appended after each round
    check_time: Quantity             # per feedback check
    input_price: Quantity
    output_price: Quantity
    context_growth: Optional[int] = None   # a grown prompt, to price breadth's premium again

    @property
    def result(self):
        from .solvers import AllocationAxesModel
        return AllocationAxesModel().solve(self.floors.result.memory_time, self.prompt, self.ceiling,
                                           self.candidates, self.candidate_tokens, self.rounds,
                                           self.report_tokens, self.check_time, self.input_price,
                                           self.output_price, self.context_growth)


class BestOfNPoolScenario(BaseScenario):
    """Pools of independent candidates at ``pass_rate``, and a mixed workload with the same average."""
    pass_rate: float
    pool_sizes: List[int]            # ascending; the last is the pool compared with retry
    easy_pass_rate: float            # half the mixed workload
    hard_pass_rate: float            # the other half

    @property
    def result(self):
        from .solvers import BestOfNPoolModel
        return BestOfNPoolModel().solve(self.pass_rate, self.pool_sizes, self.easy_pass_rate,
                                        self.hard_pass_rate)


class PassAtKEstimateScenario(BaseScenario):
    """pass@``k`` estimated from ``samples`` draws of which ``passing`` pass."""
    samples: int
    passing: int
    k: int

    @property
    def result(self):
        from .solvers import PassAtKEstimateModel
        return PassAtKEstimateModel().solve(self.samples, self.passing, self.k)


class CoverageVsVoteScenario(BaseScenario):
    """Coverage against majority vote in a repeated-sampling study (two ``Literature`` series)."""
    majority_series: str
    coverage_series: str
    small_model: str
    large_model: str
    dataset: str
    k_max: int                       # the largest pool
    k_knee: int                      # where voting stops gaining
    repair_system: str
    repair_dataset: str
    repair_k: int

    @property
    def result(self):
        from ..literature.registry import Literature
        from .solvers import CoverageVsVoteModel
        return CoverageVsVoteModel().solve(
            getattr(Literature, self.majority_series).entries,
            getattr(Literature, self.coverage_series).entries,
            self.small_model, self.large_model, self.dataset, self.k_max, self.k_knee,
            self.repair_system, self.repair_dataset, self.repair_k)


class VerifiedPoolScenario(BaseScenario):
    """A pool of ``pool_size`` candidates, first passing one returned, under a fallible check."""
    pool_size: int
    valid_rate: float                # v
    false_reject_rate: float         # eps_FR, P(FAIL | valid)
    false_accept_rate: float         # eps_FA, P(PASS | invalid)
    tokens_per_candidate: Quantity
    output_price: Quantity
    incident_loss: Quantity          # USD per defective selection

    @property
    def result(self):
        from .solvers import VerifiedPoolModel
        return VerifiedPoolModel().solve(self.pool_size, self.valid_rate, self.false_reject_rate,
                                         self.false_accept_rate, self.tokens_per_candidate,
                                         self.output_price, self.incident_loss)


class ArgmaxNoiseScenario(BaseScenario):
    """Argmax over equally good candidates with Gaussian score noise; a rare loophole in the pool."""
    pool_sizes: List[int]
    small_pool: int
    large_pool: int
    exploit_rate: float              # fraction of single samples that exploit a check gap

    @property
    def result(self):
        from .solvers import ArgmaxNoiseModel
        return ArgmaxNoiseModel().solve(self.pool_sizes, self.large_pool, self.exploit_rate)


class ScopedRevisionScenario(BaseScenario):
    """A candidate plan of equal nodes, ``replaced_nodes`` of which a refuted premise invalidates."""
    total_nodes: int
    replaced_nodes: int
    tokens_per_node: int

    @property
    def result(self):
        from .solvers import ScopedRevisionModel
        return ScopedRevisionModel().solve(self.total_nodes, self.replaced_nodes, self.tokens_per_node)


class PrunedSearchTreeScenario(BaseScenario):
    """A search tree of ``branching`` children per node whose step check rejects ``rejection_rate``."""
    branching: int
    rejection_rate: float
    depth: int
    wide_branching: Optional[int] = None   # an unpruned tree to size by its leaves
    wide_depth: Optional[int] = None

    @property
    def result(self):
        from .solvers import PrunedSearchTreeModel
        return PrunedSearchTreeModel().solve(self.branching, self.rejection_rate, self.depth,
                                             self.wide_branching, self.wide_depth)


class SearchAccountsScenario(BaseScenario):
    """Flat best-of-N against beam search over the same task, kept as separate accounts."""
    floors: DecodeScenario
    segments: int                    # candidate length in equal segments
    segment_tokens: int
    prompt_tokens: int
    flat_candidates: int
    beam_width: int
    extensions: int                  # extensions per retained prefix
    generation_slots: int            # concurrent streams at an unchanged rate
    test_slots: int                  # concurrent final tests
    score_time: Quantity             # per full-prefix score
    test_time: Quantity              # per final test
    input_price: Quantity
    output_price: Quantity
    scorer_price: Quantity           # per scorer input token

    @property
    def result(self):
        from .solvers import SearchAccountsModel
        return SearchAccountsModel().solve(
            self.floors.result.memory_time, self.segments, self.segment_tokens, self.prompt_tokens,
            self.flat_candidates, self.beam_width, self.extensions, self.generation_slots,
            self.test_slots, self.score_time, self.test_time, self.input_price, self.output_price,
            self.scorer_price)


class DeadlinePolicyScenario(BaseScenario):
    """One candidate, ``breadth`` in parallel, or feedback, for one task under one ``deadline``."""
    floors: DecodeScenario
    prompt: int
    output: int                      # generated tokens per candidate
    diagnostic: int                  # failure-report tokens fed to a revision
    breadth: int
    max_rounds: int                  # requested feedback rounds, before the deadline
    deadline: Quantity
    check_time: Quantity
    initial_pass: float
    revision_pass: float             # conditional, after a failed first candidate
    input_price: Quantity
    output_price: Quantity
    input_ceiling: Optional[int] = None    # per-decision ceilings the policies must respect
    output_ceiling: Optional[int] = None
    check_ceiling: Optional[int] = None
    concurrency: Optional[int] = None      # generator and check capacity

    @property
    def result(self):
        from .solvers import DeadlinePolicyModel
        return DeadlinePolicyModel().solve(
            self.floors.result.memory_time, self.prompt, self.output, self.diagnostic, self.breadth,
            self.max_rounds, self.deadline, self.check_time, self.initial_pass, self.revision_pass,
            self.input_price, self.output_price)


class RecordedRunCostScenario(BaseScenario):
    """One public submission in a per-trajectory ``Literature`` series, priced as it recorded."""
    series_name: str
    submission: str

    @property
    def rows(self):
        from ..literature.registry import Literature
        return [r for r in getattr(Literature, self.series_name).entries if r["submission"] == self.submission]

    @property
    def result(self):
        from .solvers import RecordedRunCostModel
        return RecordedRunCostModel().solve(self.rows)
# --- Context engineering (added 2026-10-20) -----------------------------------
# What one agent request carries and what it costs: the solvers live in
# engine/solvers/context.py. Token counts are serialized-request counts.

class ContextBudgetScenario(BaseScenario):
    """A request budgeted by role, appending each tool result against selecting a fixed share."""
    window: int
    output_reserve: int
    system_tokens: int
    tool_tokens: int
    state_tokens: int
    observation_per_turn: int        # tool result appended per turn
    selected_observation: int        # observations a selecting policy keeps on every call
    turns: int
    digest_tokens: int               # a compacted per-turn result

    @property
    def result(self):
        from .solvers import ContextBudgetModel
        return ContextBudgetModel().solve(self.window, self.output_reserve, self.system_tokens,
                                          self.tool_tokens, self.state_tokens, self.observation_per_turn,
                                          self.selected_observation, self.turns, self.digest_tokens)


class AttentionDilutionScenario(BaseScenario):
    """One attention head with fixed logits: a target among background keys, optionally look-alikes."""
    target_logit: float
    background_logit: float
    short_keys: int                  # total keys, target included
    long_keys: int
    unrelated_keys: Optional[int] = None     # competitors at the background logit
    lookalike_keys: Optional[int] = None     # competitors that share the evidence's identifiers
    lookalike_logit: Optional[float] = None

    @property
    def result(self):
        from .solvers import AttentionDilutionModel
        return AttentionDilutionModel().solve(self.target_logit, self.background_logit, self.short_keys,
                                              self.long_keys, self.unrelated_keys, self.lookalike_keys,
                                              self.lookalike_logit)


class ComprehensionByLengthScenario(BaseScenario):
    """Models of a Literature length sweep scored at every listed length."""
    series_name: str
    lengths: List[str]               # column suffixes, shortest first
    hold_bar: float                  # score a model must hold at the longest length

    @property
    def rows(self):
        from ..literature.registry import Literature
        return getattr(Literature, self.series_name).entries

    @property
    def result(self):
        from .solvers import ComprehensionByLengthModel
        return ComprehensionByLengthModel().solve(self.rows, self.lengths, self.hold_bar)


class PrefixLayoutBillScenario(BaseScenario):
    """A trajectory of root, trunk, and leaf requests, billed volatile-first and zoned."""
    turns: int
    root_tokens: int
    trunk_tokens: int
    leaf_tokens: int
    trunk_rewrites: int              # calls after the first that rewrite the trunk
    input_price: Quantity            # USD per uncached input token
    cached_fraction: float           # reused tokens billed at this fraction of the price

    @property
    def result(self):
        from .solvers import PrefixLayoutBillModel
        return PrefixLayoutBillModel().solve(self.turns, self.root_tokens, self.trunk_tokens,
                                             self.leaf_tokens, self.trunk_rewrites, self.input_price,
                                             self.cached_fraction)


class CompactionCadenceScenario(BaseScenario):
    """Compaction between a trigger and a target watermark over a turn budget."""
    window: int
    output_reserve: int
    target: int                      # request length a pass resets to
    trigger: int                     # request length that fires a pass
    tokens_per_turn: int
    turns: int
    alternate_target: Optional[int] = None   # a narrower gap to compare

    @property
    def result(self):
        from .solvers import CompactionCadenceModel
        return CompactionCadenceModel().solve(self.window, self.output_reserve, self.target, self.trigger,
                                              self.tokens_per_turn, self.turns, self.alternate_target)


class ContextInvalidationScenario(BaseScenario):
    """A superseded excerpt right after the reused prefix: remove, replace, or append a notice."""
    request_tokens: int
    prefix_tokens: int               # reused prefix before the excerpt
    superseded_tokens: int
    marker_tokens: int               # in-place replacement marker
    notice_tokens: int               # appended notice, with its framing

    @property
    def result(self):
        from .solvers import ContextInvalidationModel
        return ContextInvalidationModel().solve(self.request_tokens, self.prefix_tokens,
                                                self.superseded_tokens, self.marker_tokens,
                                                self.notice_tokens)


class ProtectedFieldCheckScenario(BaseScenario):
    """Constraints lost independently across compaction rounds, with and without a host check."""
    constraints: int
    loss_per_round: float
    rounds: int
    acceptance_if_intact: float
    attempt_cost: Quantity           # USD
    detection: float                 # share of losses the check detects and repairs
    check_cost_per_round: Quantity   # USD
    survival_floor: float            # survival level whose first crossing is reported
    target_survival: Optional[float] = None  # survival to hold without a check

    @property
    def result(self):
        from .solvers import ProtectedFieldCheckModel
        return ProtectedFieldCheckModel().solve(self.constraints, self.loss_per_round, self.rounds,
                                                self.acceptance_if_intact, self.attempt_cost,
                                                self.detection, self.check_cost_per_round,
                                                self.survival_floor, self.target_survival)

# --- Learned proposers: VLA evaluation, decoders, bus traffic, evidence (added 2026-10-20) ---

class VlaEvaluationScenario(BaseScenario):
    """One fresh evaluation of ``machine``'s intent model on its application processor.

    Each input left as None takes the machine's deployed value: the weight
    precision, the input and action token counts, and the decoding mode. The
    sustained-bandwidth fraction and compute utilization are the machine's.
    """
    machine: Any                                          # EmbodiedMobileManipulator
    weight_bytes_per_param: Optional[Quantity] = None     # default: the deployed precision
    input_tokens: Optional[int] = None                    # default: machine.intent_input_tokens
    action_tokens: Optional[int] = None                   # default: the model's action dimension
    parallel_decode: Optional[bool] = None                # default: machine.intent_parallel_decode

    @property
    def bytes_per_param(self) -> Quantity:
        return (self.machine.intent_weight_bytes_per_param if self.weight_bytes_per_param is None
                else self.weight_bytes_per_param)

    @property
    def tokens_in(self) -> int:
        return self.machine.intent_input_tokens if self.input_tokens is None else self.input_tokens

    @property
    def tokens_out(self) -> int:
        return self.machine.intent_model.action_dim if self.action_tokens is None else self.action_tokens

    @property
    def result(self):
        from .solvers import VlaEvaluationModel
        rm = self.machine
        parallel = rm.intent_parallel_decode if self.parallel_decode is None else self.parallel_decode
        return VlaEvaluationModel().solve(rm.intent_model, rm.brain_soc, self.bytes_per_param, self.tokens_in,
                                          self.tokens_out, rm.dram_efficiency, rm.intent_compute_utilization,
                                          parallel_decode=parallel)


class RateTargetScenario(BaseScenario):
    """``evaluation`` held against a ``target_rate``: one fresh evaluation per period, not pipelined."""
    evaluation: VlaEvaluationScenario
    target_rate: Quantity

    @property
    def period(self) -> Quantity:
        return (1 / self.target_rate).to("ms")

    @property
    def fits(self) -> bool:
        return self.evaluation.result.t_total <= self.period

    @property
    def slack(self) -> Quantity:
        """Period left over (negative: the overrun)."""
        return (self.period - self.evaluation.result.t_total).to("ms")

    @property
    def slack_fraction(self) -> float:
        return (self.slack / self.period).to("").magnitude

    @property
    def prefill_budget(self) -> Quantity:
        """Prefill time that fits once the decode is paid."""
        return (self.period - self.evaluation.result.t_decode).to("ms")

    @property
    def token_fraction(self) -> float:
        """Share of the input tokens that fits, treating prefill as proportional to the token count."""
        return (self.prefill_budget / self.evaluation.result.t_prefill).to("").magnitude

    @property
    def tokens_within(self) -> float:
        return self.token_fraction * self.evaluation.result.prefill_tokens


class VlaTokenBudgetScenario(BaseScenario):
    """Tokens a VLA backbone reads from ``views`` square camera frames, and their memory per token."""
    model: Any                       # EmbodiedWorkload (layers, hidden width)
    views: int
    frame_px: int
    patch_px: int
    prompt_tokens: int
    state_tokens: int
    context_tokens: int              # the rounded context length S the prose uses
    element_bytes: Quantity          # bytes per stored activation, key, or value

    @property
    def patches_per_view(self) -> int:
        return (self.frame_px // self.patch_px) ** 2

    @property
    def patches_per_side(self) -> int:
        return self.frame_px // self.patch_px

    @property
    def visual_tokens(self) -> int:
        return self.views * self.patches_per_view

    @property
    def total_tokens(self) -> int:
        return self.visual_tokens + self.prompt_tokens + self.state_tokens

    @property
    def single_view_tokens(self) -> int:
        """One view plus the prompt: the intent model's input."""
        return self.patches_per_view + self.prompt_tokens

    @property
    def embedding_memory(self) -> Quantity:
        return (self.context_tokens * self.model.hidden_dim * self.element_bytes).to("MB")

    @property
    def kv_per_token(self) -> Quantity:
        """Full-width keys and values in every layer for one token."""
        return (2 * self.model.layers * self.model.hidden_dim * self.element_bytes).to("byte")


class KvHistoryScenario(BaseScenario):
    """Key-value cache for ``history_window`` of retained context at ``history_rate`` beside ``machine``'s weights."""
    tokens: VlaTokenBudgetScenario
    machine: Any                     # EmbodiedMobileManipulator (module memory, intent model)
    history_window: Quantity
    history_rate: Quantity
    weight_bytes_per_param: Quantity  # the stored weights the history sits beside

    @property
    def frames(self) -> int:
        return round((self.history_window * self.history_rate).to("").magnitude)

    @property
    def kv_per_frame(self) -> Quantity:
        return (self.tokens.kv_per_token * self.tokens.visual_tokens).to("MB")

    @property
    def kv_per_step(self) -> Quantity:
        return (self.tokens.kv_per_token * self.tokens.context_tokens).to("MB")

    @property
    def visual_history(self) -> Quantity:
        return (self.kv_per_frame * self.frames).to("GB")

    @property
    def history(self) -> Quantity:
        return (self.kv_per_step * self.frames).to("GB")

    @property
    def weights(self) -> Quantity:
        return (self.machine.intent_model.parameters * self.weight_bytes_per_param).to("GB")

    @property
    def combined(self) -> Quantity:
        return (self.weights + self.history).to("GB")

    @property
    def module_memory(self) -> Quantity:
        return self.machine.brain_soc.memory.capacity.to("GB")

    @property
    def combined_share(self) -> float:
        return (self.combined / self.module_memory).to("").magnitude


class DecoderFamiliesScenario(BaseScenario):
    """One fresh ``machine.chunk_horizon``-step chunk from ``backbone``, per decoder family."""
    backbone: VlaEvaluationScenario
    fast_tokens: int
    expert_params: Quantity
    expert_steps: int
    expert_bytes_per_param: Quantity
    kv_bytes_per_value: Quantity

    @property
    def result(self):
        from .solvers import DecoderFamiliesModel
        b = self.backbone
        rm = b.machine
        return DecoderFamiliesModel().solve(
            rm.intent_model, rm.brain_soc, b.bytes_per_param, b.tokens_in, rm.dram_efficiency,
            rm.intent_compute_utilization, rm.chunk_horizon, rm.intent_model.action_dim, self.fast_tokens,
            self.expert_params, self.expert_steps, self.expert_bytes_per_param, self.kv_bytes_per_value,
            rm.chunk_period)


class ProposerTrafficScenario(BaseScenario):
    """``machine``'s intent model and chunk policy sharing its memory bus (weight traffic only)."""
    machine: Any                     # EmbodiedMobileManipulator
    fp16_bytes_per_param: Quantity   # the 16-bit format the sweep and supply comparisons price

    @property
    def result(self):
        from .solvers import ProposerTrafficModel
        return ProposerTrafficModel().solve(self.machine, self.fp16_bytes_per_param)


class EvidenceHorizonScenario(BaseScenario):
    """``site``'s conveyor target tracked by ``machine``'s chunk policy: evidence horizon against renewal."""
    site: Any                                    # EmbodiedSiteScenario (conveyor, tolerance, initial error, slip)
    machine: Any                                 # EmbodiedMobileManipulator (chunk schedule)
    slip_acceleration: Optional[Quantity] = None  # default: the site's residual slip
    chunk_rate: Optional[Quantity] = None         # default: the machine's renewal rate

    @property
    def a_slip(self) -> Quantity:
        return self.site.a_conveyor_slip if self.slip_acceleration is None else self.slip_acceleration

    @property
    def renewal_rate(self) -> Quantity:
        return self.machine.chunk_rate if self.chunk_rate is None else self.chunk_rate

    @property
    def result(self):
        from .solvers import EvidenceHorizonModel
        rm = self.machine
        return EvidenceHorizonModel().solve(
            self.site.grasp_tolerance, self.site.grasp_initial_error, self.site.v_conveyor, self.a_slip,
            self.renewal_rate, rm.chunk_step, rm.chunk_horizon, rm.chunk_inference_latency)


class TemporalEnsembleScenario(BaseScenario):
    """A temporal ensemble of ``predictions`` overlapping chunks with decay ``decay`` per index."""
    decay: float
    predictions: int

    @property
    def result(self):
        from ..physics.robotics import calc_temporal_ensemble_weights
        return calc_temporal_ensemble_weights(self.decay, self.predictions)


class WeightOnlyQuantizationScenario(BaseScenario):
    """One backbone priced at several weight formats against measurements of the same formats.

    ``formats`` maps a label to its evaluation scenario; ``measured`` and
    ``scores`` map the same labels to a measured latency and a task score.
    ``group_size``, ``scale_bits``, and ``weight_bits`` price the per-group
    scale overhead of the deployed format.
    """
    formats: Dict[str, VlaEvaluationScenario]
    measured: Dict[str, Quantity]
    scores: Dict[str, float]
    deployed: str                    # label of the deployed format
    reference: str                   # label of the 16-bit reference format
    group_size: int
    scale_bits: int
    weight_bits: int

    def computed(self, label: str) -> Quantity:
        return self.formats[label].result.t_total

    def gap(self, label: str) -> float:
        """(computed - measured) / measured."""
        return ((self.computed(label) - self.measured[label]) / self.measured[label]).to("").magnitude

    @property
    def measured_speedup(self) -> float:
        return (self.measured[self.reference] / self.measured[self.deployed]).to("").magnitude

    @property
    def score_drop(self) -> float:
        return self.scores[self.reference] - self.scores[self.deployed]

    @property
    def scale_overhead(self) -> float:
        return self.scale_bits / self.group_size / self.weight_bits


class ConfidenceRegimesScenario(BaseScenario):
    """Stipulated proposer confidence against task success in several operating regimes."""
    regimes: List[Any]               # (regime, confidence, success)

    def gap(self, regime: str) -> float:
        for name, conf, success in self.regimes:
            if name == regime:
                return conf - success
        raise KeyError(regime)

    def row(self, regime: str):
        for r in self.regimes:
            if r[0] == regime:
                return r
        raise KeyError(regime)


class SingleStreamLatencyScenario(BaseScenario):
    """Published single-stream latencies of one ``system`` from a benchmark ``series``, latest round each."""
    series: Any                      # a Literature series with .entries rows
    system: str
    benchmarks: List[Any]            # (benchmark id, task, model)
    scenario: str = "SingleStream"
    metric_prefix: str = "p90"

    def rounds(self, bench: str) -> List[Any]:
        """Every matching row for ``bench``, oldest round first."""
        hits = [r for r in self.series.entries
                if r["system"] == self.system and r["benchmark"] == bench
                and r["scenario"] == self.scenario and r["metric"].startswith(self.metric_prefix)]
        return sorted(hits, key=lambda r: r["round"])

    def latest(self, bench: str) -> Quantity:
        return Q_(self.rounds(bench)[-1]["value"], "ms")

    def earliest(self, bench: str) -> Quantity:
        return Q_(self.rounds(bench)[0]["value"], "ms")

    def latest_round(self, bench: str) -> str:
        return self.rounds(bench)[-1]["round"]

    @property
    def latencies(self) -> List[Quantity]:
        return [self.latest(b[0]) for b in self.benchmarks]

    def count_within(self, period: Quantity) -> int:
        return sum(1 for t in self.latencies if t <= period)

    @property
    def span(self) -> float:
        lat = self.latencies
        return (max(lat) / min(lat)).to("").magnitude
# --- Data engineering: false-wake budgets, label budgets, agreement, ingest cost (added 2026-10-20) ---

class AlwaysOnDetectorTargetsScenario(BaseScenario):
    """An always-on detector's product contract: what it must catch, how rarely it may wake falsely,
    what it may occupy on the device, and the data budget and schedule that must deliver it."""
    detection_target: float           # fraction of spoken wake words caught
    false_wake_budget: float          # expected false wakes per false_wake_horizon
    false_wake_horizon: Quantity
    response_limit: Quantity
    model_memory: Quantity            # always-on model-size budget
    preprocessing_memory: Quantity    # separate preprocessing-state budget
    data_budget: Quantity             # USD
    timeline: Quantity
    base_detection_rate: float        # an illustrative starting detection rate


class FalseAlarmBudgetScenario(BaseScenario):
    """One detection window every ``window``, ``listening_per_day`` of each day, over ``horizon``.

    ``budget`` is the expected false alarms allowed per horizon. Each window with
    no keyword is a chance to fire, so a per-window rejection rate multiplies into
    a count; ``rejection_levels`` lists rates to compare against the budget.
    """
    window: Quantity
    listening_per_day: Quantity
    horizon: Quantity
    budget: float
    rejection_levels: List[float] = []

    @property
    def listening_time(self) -> Quantity:
        """Listening time in one horizon."""
        return (self.horizon.to("day").magnitude * self.listening_per_day).to("hour")

    @property
    def windows(self) -> float:
        """Detection windows in one horizon."""
        return (self.listening_time / self.window).to("").magnitude

    @property
    def target_false_positive_rate(self) -> float:
        """Per-window false-positive rate that meets the budget."""
        return self.budget / self.windows

    @property
    def rejection_needed(self) -> float:
        return 1 - self.target_false_positive_rate

    def false_alarms(self, rejection: float) -> float:
        """Expected false alarms per horizon at a per-window rejection rate."""
        return self.windows * (1 - rejection)

    def false_alarms_at_rate(self, rate: Quantity) -> float:
        """Expected false alarms per horizon at an hourly false-alarm rate."""
        return (rate * self.listening_time).to("").magnitude


class ZeroEventExposureScenario(BaseScenario):
    """Exposure needed, with zero events observed, to bound a rate of ``budget`` per ``horizon``
    at one-sided ``confidence``, and the size of recording that exposure at ``sample_rate``."""
    budget: float
    horizon: Quantity
    confidence: float
    sample_rate: Quantity
    sample_width: Quantity

    @property
    def target_rate(self) -> Quantity:
        return (self.budget / self.horizon).to("1/hour")

    @property
    def events_bound(self) -> float:
        """Upper confidence bound on the expected count when none is seen: -ln(1 - confidence)."""
        import math
        return -math.log(1 - self.confidence)

    @property
    def exposure_needed(self) -> Quantity:
        from ..physics import calc_empirical_testing_exposure
        return calc_empirical_testing_exposure(self.target_rate, self.confidence).to("hour")

    @property
    def horizons_needed(self) -> float:
        return (self.exposure_needed / self.horizon).to("").magnitude

    @property
    def recording_size(self) -> Quantity:
        return (self.exposure_needed * self.sample_rate * self.sample_width).to("byte")


class DataBudgetScenario(BaseScenario):
    """A data budget split into ``shares`` by category; the labeling share buys labels at
    ``label_price`` plus ``review_overhead`` for quality review."""
    total: Quantity                   # USD
    shares: Dict[str, float]
    label_price: Quantity             # USD per label
    review_overhead: float
    labeling_category: str = "labeling"

    def allocation(self, category: str) -> Quantity:
        return self.total * self.shares[category]

    @property
    def effective_label_cost(self) -> Quantity:
        return self.label_price * (1 + self.review_overhead)

    @property
    def labels(self) -> float:
        return (self.allocation(self.labeling_category) / self.effective_label_cost).to("").magnitude


class LabelingVersusTrainingScenario(BaseScenario):
    """A labeling budget against one training run: ``epochs`` over ``dataset_examples`` at
    ``throughput_per_accelerator`` on ``accelerators``, priced at a low and a high hourly rate."""
    labeling_budget: Quantity         # USD
    dataset_examples: Quantity        # count
    epochs: int
    throughput_per_accelerator: Quantity   # count/second
    accelerators: int
    price_low: Quantity               # USD per accelerator-hour
    price_high: Quantity
    labeling_time_low: Optional[Quantity] = None    # calendar time the labeling takes, low
    labeling_time_high: Optional[Quantity] = None   # and high

    @property
    def train_time(self) -> Quantity:
        return (self.dataset_examples * self.epochs
                / (self.accelerators * self.throughput_per_accelerator)).to("hour")

    @property
    def train_cost_low(self) -> Quantity:
        return (self.accelerators * self.train_time * self.price_low).to("USD")

    @property
    def train_cost_high(self) -> Quantity:
        return (self.accelerators * self.train_time * self.price_high).to("USD")

    @property
    def ratio_low(self) -> float:
        """Labeling budget over the dearer training run."""
        return (self.labeling_budget / self.train_cost_high).to("").magnitude

    @property
    def ratio_high(self) -> float:
        """Labeling budget over the cheaper training run."""
        return (self.labeling_budget / self.train_cost_low).to("").magnitude


class LabelGranularityScenario(BaseScenario):
    """Label entries for one image: a per-pixel mask against ``boxes`` boxes of ``coords_per_box``."""
    image_width: int
    image_height: int
    boxes: int
    coords_per_box: int

    @property
    def mask_entries(self) -> int:
        return self.image_width * self.image_height

    @property
    def box_entries(self) -> int:
        return self.boxes * self.coords_per_box

    @property
    def mask_to_box_ratio(self) -> float:
        return self.mask_entries / self.box_entries


class LaborEffortScenario(BaseScenario):
    """``items`` labeled by hand at ``time_per_item`` each: total hours and full-time person-years."""
    items: float
    time_per_item: Quantity
    work_year: Quantity               # working hours in one full-time person-year

    @property
    def total_time(self) -> Quantity:
        return (self.items * self.time_per_item).to("hour")

    @property
    def person_years(self) -> float:
        return (self.total_time / self.work_year).to("").magnitude


class AnnotatorAgreementScenario(BaseScenario):
    """Two independent annotators with the same label mix ``class_shares``: chance agreement,
    and Cohen's kappa at an ``observed_agreement``."""
    class_shares: List[float]
    observed_agreement: Optional[float] = None

    @property
    def chance_agreement(self) -> float:
        from ..physics import calc_chance_agreement
        return calc_chance_agreement(self.class_shares)

    @property
    def kappa(self) -> float:
        from ..physics import calc_cohens_kappa
        return calc_cohens_kappa(self.observed_agreement, self.chance_agreement)


class AgreementDriftScenario(BaseScenario):
    """A labeling campaign's kappa at a baseline and again after ``elapsed``."""
    baseline_kappa: float
    later_kappa: float
    elapsed: Quantity

    @property
    def drop(self) -> float:
        return self.baseline_kappa - self.later_kappa


class ActiveLearningBudgetScenario(BaseScenario):
    """Active learning against a fixed label budget when scoring the pool is charged to the same budget.

    Counts are examples; prices are USD. ``compute_share`` of the budget pays for
    scoring a narrowed candidate pool, and the rest buys labels.
    """
    pool: float
    label_budget: Quantity
    label_price: Quantity
    scoring_price: Quantity
    random_labels_needed: float
    active_labels_low: float
    active_labels_high: float
    compute_share: float
    baseline_accuracy: float
    target_accuracy: float

    @property
    def random_label_cost(self) -> Quantity:
        return self.random_labels_needed * self.label_price

    @property
    def active_label_cost_low(self) -> Quantity:
        return self.active_labels_low * self.label_price

    @property
    def active_label_cost_high(self) -> Quantity:
        return self.active_labels_high * self.label_price

    @property
    def full_pool_scoring_cost(self) -> Quantity:
        return self.pool * self.scoring_price

    @property
    def active_total_low(self) -> Quantity:
        return self.full_pool_scoring_cost + self.active_label_cost_low

    @property
    def active_total_high(self) -> Quantity:
        return self.full_pool_scoring_cost + self.active_label_cost_high

    @property
    def compute_budget(self) -> Quantity:
        return self.label_budget * self.compute_share

    @property
    def candidate_pool(self) -> float:
        """Candidates the compute budget can score."""
        return (self.compute_budget / self.scoring_price).to("").magnitude

    @property
    def affordable_labels(self) -> float:
        """Labels the rest of the budget buys after scoring."""
        return ((self.label_budget - self.compute_budget) / self.label_price).to("").magnitude


class HashSplitScenario(BaseScenario):
    """A released dataset split by hashing a stable key, with the loader's default percentages."""
    dataset: Any                      # DatasetProfile with training/validation/test counts
    validation_percent: float
    test_percent: float

    @property
    def train_percent(self) -> float:
        return 100 - self.validation_percent - self.test_percent

    @property
    def released_total(self) -> float:
        d = self.dataset
        return sum(q.to("count").magnitude for q in
                   (d.training_examples, d.validation_examples, d.test_examples))

    @property
    def released_test_share(self) -> float:
        return self.dataset.test_examples.to("count").magnitude / self.released_total


class BatchVersusStreamingIngestScenario(BaseScenario):
    """One event stream ingested by always-on streaming cores or by a batch job each hour.

    Streaming runs ``stream_cores`` (``redundancy_factor`` cores per primary) for
    ``stream_hours_per_day``; batch runs ``batch_cores`` for ``batch_window`` of each
    clock hour. Both pay ``core_hour_price``.
    """
    event_rate: Quantity              # events per second
    event_size: Quantity
    stream_cores: int
    stream_hours_per_day: Quantity
    batch_cores: int
    batch_window: Quantity
    core_hour_price: Quantity         # USD per core-hour
    redundancy_factor: int = 1

    @property
    def throughput(self) -> Quantity:
        return (self.event_rate * self.event_size).to("GB/s")

    @property
    def hourly_data(self) -> Quantity:
        return (self.throughput * Q_(1, "hour")).to("TB")

    @property
    def stream_core_hours_per_day(self) -> float:
        return self.stream_cores * self.stream_hours_per_day.to("hour").magnitude

    @property
    def stream_cost_per_day(self) -> Quantity:
        return (self.stream_core_hours_per_day * Q_(1, "hour") * self.core_hour_price).to("USD")

    @property
    def batch_core_hours_per_day(self) -> float:
        hours_per_day = Q_(1, "day").to("hour").magnitude
        return self.batch_cores * self.batch_window.to("hour").magnitude * hours_per_day

    @property
    def batch_cost_per_day(self) -> Quantity:
        return (self.batch_core_hours_per_day * Q_(1, "hour") * self.core_hour_price).to("USD")

    @property
    def cost_ratio(self) -> float:
        return (self.stream_cost_per_day / self.batch_cost_per_day).to("").magnitude

    @property
    def stream_primary_cores(self) -> int:
        return self.stream_cores // self.redundancy_factor

    @property
    def stream_per_core(self) -> Quantity:
        return (self.throughput / self.stream_primary_cores).to("MB/s")

    @property
    def batch_aggregate(self) -> Quantity:
        return (self.hourly_data / self.batch_window).to("GB/s")

    @property
    def batch_per_core(self) -> Quantity:
        return (self.batch_aggregate / self.batch_cores).to("MB/s")


class DesignTradeoffScenario(BaseScenario):
    """One design-space move of a knob from ``baseline`` to ``alternative``, with an illustrative
    quality gain from ``gain_low`` to ``gain_high`` (fractions); costs scale by ``ratio``."""
    baseline: float
    alternative: float
    gain_low: float
    gain_high: float

    @property
    def ratio(self) -> float:
        return self.alternative / self.baseline


# --- Data delivery and drift: feeding an accelerator, storage tiers, formats, drift tests (added 2026-10-20) ---

class AcceleratorInputFeedScenario(BaseScenario):
    """``accelerators`` copies of ``hardware`` training ``workload`` at ``sustained_rate`` samples a second each,
    fed from ``supply`` with ``encoded_sample`` bytes stored per sample and ``decoded_sample`` bytes after decode.

    The arithmetic ceiling is peak FLOP/s over the training FLOPs of one sample
    (``training_flops_factor`` times the forward pass). Demand is rate times bytes
    per sample; the feeding loss is the share of the training rate a slower
    supply removes (``DataModel``). ``worker_rate`` is what one host loader
    worker decodes and augments a second; ``host_decode_cap`` is where adding
    workers stops helping because the host's cores run out.
    """
    workload: Workload
    hardware: HardwareNode
    sustained_rate: Quantity          # samples per second per accelerator
    encoded_sample: Quantity          # bytes per sample as stored
    decoded_sample: Quantity          # bytes per sample after decode
    supply: Any                       # StorageSubsystem that feeds the accelerators
    training_flops_factor: float = 3.0
    accelerators: int = 1
    worker_rate: Optional[Quantity] = None   # samples per second per loader worker
    host_decode_cap: Optional[Quantity] = None   # samples per second the whole host can decode

    @property
    def training_flops_per_sample(self) -> Quantity:
        return (self.workload.inference_flops * self.training_flops_factor).to("flop")

    @property
    def ceiling_rate(self) -> Quantity:
        """Samples a second one accelerator could train at its peak arithmetic rate."""
        return (self.hardware.compute.peak_flops / self.training_flops_per_sample).to("count/second")

    @property
    def sustained_fraction(self) -> float:
        return (self.sustained_rate / self.ceiling_rate).to("").magnitude

    @property
    def total_rate(self) -> Quantity:
        return (self.accelerators * self.sustained_rate).to("count/second")

    def demand(self, decoded: bool = False, rate: Optional[Quantity] = None) -> Quantity:
        """Storage bandwidth for every accelerator at ``rate`` (default: the sustained rate)."""
        per = self.sustained_rate if rate is None else rate
        size = self.decoded_sample if decoded else self.encoded_sample
        return (self.accelerators * per * size / Q_(1, "count")).to("MB/s")

    @property
    def ceiling_demand(self) -> Quantity:
        return self.demand(rate=self.ceiling_rate)

    @property
    def decode_expansion(self) -> float:
        return (self.decoded_sample / self.encoded_sample).to("").magnitude

    def feeding_loss(self, decoded: bool = False) -> float:
        """Share of the sustained training rate lost when ``supply`` is the slowest stage."""
        from .solvers import DataModel
        node = self.hardware.model_copy(update={"storage": self.supply})
        r = DataModel().solve(self.demand(decoded), node)
        return 1.0 - 1.0 / max(r.utilization, 1.0)

    @property
    def supplied_fraction(self) -> float:
        """Share of the sustained demand ``supply`` delivers (capped at one)."""
        return min((self.supply.bandwidth / self.demand()).to("").magnitude, 1.0)

    def samples_supplied(self, bandwidth: Quantity) -> Quantity:
        """Encoded samples a second a storage path of ``bandwidth`` delivers."""
        return (bandwidth / self.encoded_sample * Q_(1, "count")).to("count/second")

    def streams_needed(self, bandwidth: Quantity, rate: Optional[Quantity] = None) -> float:
        """Parallel streams of ``bandwidth`` each that meet the demand at ``rate`` (default: the ceiling)."""
        target = self.ceiling_demand if rate is None else self.demand(rate=rate)
        return (target / bandwidth).to("").magnitude

    @property
    def worker_crossing(self) -> float:
        """Loader workers whose combined rate equals one accelerator's sustained rate."""
        return (self.sustained_rate / self.worker_rate).to("").magnitude

    @property
    def workers_per_accelerator(self) -> int:
        import math
        return math.ceil(self.worker_crossing - 1e-9)

    @property
    def workers_pooled(self) -> int:
        """Whole workers for every accelerator's rate drawn from one shared pool."""
        import math
        return math.ceil((self.total_rate / self.worker_rate).to("").magnitude - 1e-9)


class FeatureLookupScenario(BaseScenario):
    """An online service at ``request_rate`` that reads ``reads_per_request`` stored features per request.

    Optional (added 2026-10-20): ``request_budget`` is the end-to-end latency
    budget of one request and ``lookup_budget_low``/``lookup_budget_high`` the
    share of it the feature store may take.
    """
    request_rate: Quantity            # requests per second
    reads_per_request: int
    feature_bytes: Optional[Quantity] = None
    raw_sample_bytes: Optional[Quantity] = None   # the raw record a feature vector summarizes
    request_budget: Optional[Quantity] = None
    lookup_budget_low: Optional[Quantity] = None
    lookup_budget_high: Optional[Quantity] = None

    @property
    def reads_per_day(self) -> float:
        return (self.request_rate * self.reads_per_request).to("1/day").magnitude

    @property
    def reads_per_second(self) -> float:
        return (self.request_rate * self.reads_per_request).to("1/second").magnitude

    @property
    def storage_reduction(self) -> float:
        return (self.raw_sample_bytes / self.feature_bytes).to("").magnitude


class CorpusTierScenario(BaseScenario):
    """An audio corpus (``dataset``) kept on storage tiers at ``storage_rates`` and read at ``read_bandwidths``."""
    dataset: Any                      # DatasetProfile with count, duration, sample rate and width
    storage_rates: Dict[str, Quantity]    # USD per byte-month, by tier
    read_bandwidths: Dict[str, Quantity]  # sequential read, by tier

    @property
    def corpus_bytes(self) -> Quantity:
        d = self.dataset
        return (d.training_examples.to("count").magnitude * d.sample_duration * d.sample_rate
                * d.sample_width).to("GB")

    def monthly_cost(self, tier: str) -> Quantity:
        return (self.corpus_bytes * self.storage_rates[tier]).to("dollar/month")

    def load_time(self, tier: str) -> Quantity:
        return (self.corpus_bytes / self.read_bandwidths[tier]).to("second")


class ColumnProjectionScenario(BaseScenario):
    """A model that uses ``columns_used`` of ``columns_total`` equal-size columns, read from a row or a columnar file."""
    columns_used: int
    columns_total: int

    @property
    def read_fraction(self) -> float:
        """Share of the table a columnar reader fetches."""
        return self.columns_used / self.columns_total

    @property
    def wasted_fraction(self) -> float:
        """Share of a row-format scan the model discards."""
        return 1 - self.read_fraction

    @property
    def speedup(self) -> float:
        """Effective-bandwidth gain of reading only the used columns."""
        return 1 / self.read_fraction


class CodecEpochScenario(BaseScenario):
    """``epoch_bytes`` decompressed on the host each epoch for ``epochs`` epochs, at each codec's ``decode_rates``."""
    epoch_bytes: Quantity
    epochs: int
    decode_rates: Dict[str, Quantity]

    def decode_time(self, codec: str) -> Quantity:
        """Host CPU time to decompress one epoch."""
        return (self.epoch_bytes / self.decode_rates[codec]).to("minute")

    def time_saved(self, slow: str, fast: str) -> Quantity:
        """Per-epoch decompression time the faster codec saves."""
        return self.decode_time(slow) - self.decode_time(fast)

    def run_time_saved(self, slow: str, fast: str) -> Quantity:
        return (self.epochs * self.time_saved(slow, fast)).to("hour")

    def rate_ratio(self, fast: str, slow: str) -> float:
        return (self.decode_rates[fast] / self.decode_rates[slow]).to("").magnitude


class KsDriftScenario(BaseScenario):
    """A two-sample K-S drift test of ``n_reference`` training against ``n_current`` recent values of one feature."""
    coefficient: float                # Ops.Monitoring.KsTestCoefficient at the 0.05 level
    n_reference: int
    n_current: int
    observed_statistic: Optional[float] = None

    @property
    def critical_value(self) -> float:
        from ..physics import calc_ks_critical_value
        return calc_ks_critical_value(self.coefficient, self.n_reference, self.n_current)

    @property
    def flags(self) -> bool:
        return self.observed_statistic > self.critical_value

    def flags_statistic(self, statistic: float) -> bool:
        return statistic > self.critical_value


class LocationShiftScenario(BaseScenario):
    """A normally distributed feature (``reference_mean``, ``feature_sd``) whose mean moves by ``shift``.

    Values are plain numbers in the feature's own unit (loudness in dBFS is
    logarithmic, so it is not carried as a Pint quantity). For a pure location
    shift, the K-S distance is ``2 Phi(shift / 2 sd) - 1`` and the Wasserstein
    distance equals the shift, so the normed Wasserstein is ``shift / sd``.
    """
    reference_mean: float
    feature_sd: float
    shift: float                      # signed: negative moves the mean down

    @property
    def current_mean(self) -> float:
        return self.reference_mean + self.shift

    @property
    def shift_sd(self) -> float:
        return abs(self.shift / self.feature_sd)

    @property
    def ks_statistic(self) -> float:
        from ..physics import calc_ks_normal_location_shift
        return calc_ks_normal_location_shift(self.shift_sd)

    @property
    def normed_wasserstein(self) -> float:
        return self.shift_sd


class BinShareShiftScenario(BaseScenario):
    """A feature binned at the training quantiles: training shares against recent shares, scored by PSI."""
    baseline_shares: List[float]
    current_shares: List[float]

    @property
    def terms(self) -> List[float]:
        import math
        return [(c - b) * math.log(c / b) for b, c in zip(self.baseline_shares, self.current_shares)]

    @property
    def psi(self) -> float:
        from ..physics import calc_population_stability_index
        return calc_population_stability_index(self.baseline_shares, self.current_shares)

    def ratio(self, i: int) -> float:
        """Recent share over training share in bin ``i``."""
        return self.current_shares[i] / self.baseline_shares[i]


class MetricRegressionScenario(BaseScenario):
    """A deployed model's metric at launch and its fall, in points, after ``elapsed`` with the model unchanged."""
    launch_value: float               # fraction
    drop: float                       # fraction (points / 100)
    elapsed: Quantity


# ── Fleet goodput (added 2026-10-20) ──────────────────────────────────────────

class SchedulingSnapshotScenario(BaseScenario):
    """One snapshot of a shared pool's in-service capacity, read at the scheduling boundary.

    Capacity assigned to jobs includes gangs still waiting for members; only
    complete allocations can run a synchronous step, so scheduling goodput is
    the assigned share less the incomplete-gang share. Capacity in complete
    allocations waiting for input crosses this boundary and is a runtime loss.
    """
    assigned_share: float
    incomplete_gang_share: float
    input_wait_share: float

    @property
    def scheduling_goodput(self) -> float:
        return self.assigned_share - self.incomplete_gang_share

    @property
    def scheduling_loss(self) -> float:
        return 1 - self.scheduling_goodput

    @property
    def unassigned_share(self) -> float:
        return 1 - self.assigned_share

    @property
    def executing_share(self) -> float:
        return self.scheduling_goodput - self.input_wait_share

    @property
    def input_wait_of_held(self) -> float:
        """The input-wait share as a share of held (complete-allocation) time."""
        return self.input_wait_share / self.scheduling_goodput

    @property
    def partial_hold_ceiling(self) -> float:
        """S if every device freed from a partial gang joins a complete allocation."""
        return self.scheduling_goodput + self.incomplete_gang_share


class RuntimeGoodputScenario(BaseScenario):
    """Runtime goodput of a job on ``failures.n_devices`` devices over a held ``window``.

    ``failures`` is the job's series failure model (SeriesFailureScenario);
    the checkpoint is ``workload`` at ``bytes_per_param``; ``recovery`` is the
    four-term budget profile (Systems.Reliability.Recovery).
    """
    failures: Any                            # SeriesFailureScenario
    workload: Workload
    bytes_per_param: Quantity
    recovery: Any
    input_wait: float                        # share of held time
    window: Quantity                         # held time per launch
    lost_work: Quantity                      # work since the last save, per forced preemption
    preemption_rate: Quantity                # forced preemptions per unit time

    @property
    def checkpoint_bytes(self) -> Quantity:
        from ..physics.quantities import memory_from_params
        return memory_from_params(self.workload.parameters, self.bytes_per_param).to("byte")

    @property
    def result(self):
        from .solvers import RuntimeGoodputModel
        return RuntimeGoodputModel().solve(self.failures.result.mtbf, self.checkpoint_bytes, self.recovery,
                                           self.input_wait, self.window, self.lost_work, self.preemption_rate)

    def at(self, n_devices: int) -> "RuntimeGoodputScenario":
        """The same job and inputs on ``n_devices`` devices."""
        return self.variant(failures=self.failures.variant(n_devices=n_devices))

    def sweep(self, sizes: List[int]) -> Dict[int, Any]:
        """Results at each job size, every other input held."""
        return {n: self.at(n).result for n in sizes}

    def crossover(self, step: int) -> int:
        """Smallest size, in ``step``-device increments from the current size, at which
        checkpoint waste plus recovery exceed input waits."""
        n = self.failures.n_devices
        while self.at(n).result.failure_side <= self.input_wait:
            n += step
        return n


class GoodputScenario(BaseScenario):
    """ML productivity goodput of a job on a pool: scheduling, runtime, and program goodput.

    ``ceilings`` gives, per factor ("S", "R", "P"), the value its owner's lever
    can reach; ``device_hour_rate`` is the price of one offered device-hour.
    """
    scheduling: float
    runtime: float
    program: float
    ceilings: Dict[str, float] = {}
    device_hour_rate: Optional[Quantity] = None

    @property
    def result(self):
        from .solvers import GoodputModel
        return GoodputModel().solve(self.scheduling, self.runtime, self.program, self.ceilings,
                                    self.device_hour_rate)

    def trade_ratio(self, other: "GoodputScenario") -> float:
        """MPG of ``other`` over this one: the product of the factor ratios."""
        return other.result.mpg / self.result.mpg


class FleetMfuScenario(BaseScenario):
    """Program goodput (MFU) of a training step on one node and on a fleet of ``fleet_devices``.

    Useful work per step is the model's training arithmetic for the tokens in
    the step (6 x parameters x tokens); MFU divides it by every device's peak
    over the measured step time. Their ratio is the scaling efficiency measured
    from the node.
    """
    workload: Workload
    hardware: HardwareNode
    node_devices: int
    fleet_devices: int
    node_tokens: int                         # tokens per node step
    fleet_tokens: int                        # tokens per fleet step
    node_step_time: Quantity
    fleet_step_time: Quantity

    def _useful(self, tokens: int) -> Quantity:
        from ..physics.transformer import calc_transformer_training_flops
        return calc_transformer_training_flops(self.workload.parameters, tokens)

    @property
    def node_useful_work(self) -> Quantity:
        return self._useful(self.node_tokens)

    @property
    def fleet_useful_work(self) -> Quantity:
        return self._useful(self.fleet_tokens)

    @property
    def fleet_peak_work(self) -> Quantity:
        """Every fleet device at peak for one fleet step: the MFU denominator."""
        return self.fleet_devices * self.hardware.compute.peak_flops * self.fleet_step_time

    @property
    def node_mfu(self) -> float:
        peak = self.node_devices * self.hardware.compute.peak_flops * self.node_step_time
        return (self.node_useful_work / peak).to("").magnitude

    @property
    def fleet_mfu(self) -> float:
        return (self.fleet_useful_work / self.fleet_peak_work).to("").magnitude

    @property
    def scaling_efficiency(self) -> float:
        """Per-device throughput on the fleet relative to the node (fleet MFU / node MFU)."""
        return self.fleet_mfu / self.node_mfu

    @property
    def extra_step_time(self) -> Quantity:
        return (self.fleet_step_time - self.node_step_time).to("ms")

    @property
    def extra_share(self) -> float:
        return (self.extra_step_time / self.fleet_step_time).to("").magnitude


class CompilationPaybackScenario(BaseScenario):
    """A compiled decode step and the executions that repay its build.

    The eager step splits into ``shares`` (matrix multiplies, element-wise
    kernels, launch overhead); compilation removes ``removed[k]`` of each
    share. A build is repaid after ``build / saved_per_token`` tokens; shared
    by ``replicas`` it is repaid in a fraction of that time, and rebuilt by
    each it costs ``replicas x gpus_per_replica x build``.
    """
    token_rate: Quantity                     # eager tokens per second
    step_time: Quantity                      # eager time per token, as the example carries it
    shares: Dict[str, float]
    removed: Dict[str, float]
    builds: Dict[str, Quantity]
    replicas: int
    gpus_per_replica: int = 1
    replica_life: Optional[Quantity] = None  # held time of a short-lived replica

    @property
    def step_fraction(self) -> float:
        return sum(v * (1 - self.removed.get(k, 0)) for k, v in self.shares.items())

    @property
    def compiled_step(self) -> Quantity:
        return (self.step_time * self.step_fraction).to("ms")

    @property
    def saved_per_token(self) -> Quantity:
        return (self.step_time - self.compiled_step).to("ms")

    @property
    def compiled_rate(self) -> Quantity:
        return (1 / self.compiled_step).to("1/second")

    @property
    def speedup(self) -> float:
        return (self.compiled_rate / self.token_rate).to("").magnitude

    def tokens_to_repay(self, build: str) -> float:
        return (self.builds[build] / self.saved_per_token).to("").magnitude

    def time_to_repay(self, build: str) -> Quantity:
        """Compiled generation time on one replica that repays the build."""
        return (self.tokens_to_repay(build) / self.compiled_rate).to("minute")

    def shared_repay(self, build: str) -> Quantity:
        """Pool time per replica when every replica reuses one build."""
        return (self.time_to_repay(build) / self.replicas).to("second")

    def rollout_cost(self, build: str) -> Quantity:
        """Device time when every replica builds its own artifact."""
        return (self.builds[build] * self.gpus_per_replica * self.replicas).to("hour")

    def build_share(self, build: str) -> float:
        """The build as a share of a short-lived replica's held time."""
        return (self.builds[build] / self.replica_life).to("").magnitude


class TraceVolumeScenario(BaseScenario):
    """A profiler trace of ``window`` on every one of ``devices`` devices."""
    per_device: Quantity                     # trace bytes per device for one window
    devices: int
    window: Quantity

    @property
    def total(self) -> Quantity:
        return (self.per_device * self.devices).to("GB")


class SoftwareGainsScenario(BaseScenario):
    """Throughput of one fixed system across benchmark rounds, relative to each benchmark's first round."""
    series: Any                              # a Literature series with .entries rows
    benchmarks: List[Any]                    # (benchmark id, label)
    scenario: str = "Offline"

    def rows(self, bench: str) -> List[Any]:
        """Every matching row for ``bench``, oldest publication first."""
        hits = [r for r in self.series.entries if r["benchmark"] == bench and r["scenario"] == self.scenario]
        return sorted(hits, key=lambda r: r["pub_month"])

    def gain(self, bench: str) -> float:
        pts = self.rows(bench)
        return pts[-1]["value"] / pts[0]["value"]

    def months(self) -> int:
        """Months from the earliest to the latest publication in the scenario."""
        stamps = sorted(r["pub_month"] for r in self.series.entries if r["scenario"] == self.scenario)
        (y0, m0), (y1, m1) = (tuple(int(x) for x in stamps[i].split("-")) for i in (0, -1))
        return (y1 - y0) * 12 + (m1 - m0)


# --- Serving at scale (added 2026-10-20) ---------------------------------------
# Traces, measured frontiers, replica admission, and fleet routing for serving
# a model to users: what counts as a served request, how many fit, and what
# each one costs.

import math as _math  # noqa: E402


class ServingCostScenario(BaseScenario):
    """One model version's training spend against its first year of serving."""
    training_cost: float                 # USD, one-time
    daily_users: int
    requests_per_user_per_day: int
    cost_per_request: float              # USD per served request
    efficiency_gain: float               # fractional cut in cost per request
    days_per_year: int

    @property
    def daily_requests(self) -> float:
        return self.daily_users * self.requests_per_user_per_day

    @property
    def annual_requests(self) -> float:
        return self.daily_requests * self.days_per_year

    @property
    def annual_serving_cost(self) -> float:
        return self.annual_requests * self.cost_per_request

    @property
    def serving_to_training(self) -> float:
        return self.annual_serving_cost / self.training_cost

    @property
    def efficiency_savings(self) -> float:
        return self.annual_serving_cost * self.efficiency_gain

    @property
    def monthly_serving_cost(self) -> float:
        return self.annual_serving_cost / 12

    @property
    def crossover_months(self) -> float:
        """Months of serving that cost as much as the training run."""
        return self.training_cost / self.monthly_serving_cost

    @property
    def crossover_weeks(self) -> float:
        return self.crossover_months * self.days_per_year / 12 / 7


class SloGoodputScenario(BaseScenario):
    """Completed-request rate and the share that meet their latency targets, per configuration."""
    completed: Dict[str, Quantity]       # requests per second completed
    met_share: Dict[str, float]          # share meeting every latency target

    def goodput(self, name: str) -> Quantity:
        return (self.completed[name] * self.met_share[name]).to("1/second")

    @property
    def best(self) -> str:
        return max(self.completed, key=lambda k: self.goodput(k).magnitude)


class ServingTaxScenario(BaseScenario):
    """A request's latency target split between model execution and everything outside it."""
    latency_target: Quantity
    model_time: Quantity                 # model execution on an idle replica
    exchanges: int                       # sequential network exchanges on the request path
    exchange_low: Quantity
    exchange_high: Quantity

    @property
    def outside_model(self) -> Quantity:
        return (self.latency_target - self.model_time).to("ms")

    @property
    def exchanges_low(self) -> Quantity:
        return (self.exchanges * self.exchange_low).to("ms")

    @property
    def exchanges_high(self) -> Quantity:
        return (self.exchanges * self.exchange_high).to("ms")


class BatchingKneeScenario(BaseScenario):
    """Batch service time S(B) = t_0 + B t_req: where throughput turns over."""
    fixed_time: Quantity                 # paid once per batch
    per_request_time: Quantity           # added per request

    @property
    def knee(self) -> float:
        from ..physics.serving import calc_batch_knee
        return calc_batch_knee(self.fixed_time, self.per_request_time)

    def throughput_share(self, batch_size) -> float:
        from ..physics.serving import calc_batch_throughput_share
        return calc_batch_throughput_share(batch_size, self.fixed_time, self.per_request_time)


class LittlesLawScenario(BaseScenario):
    """Requests in flight at ``arrival_rate``, and replicas sized from a batch service time."""
    arrival_rate: Quantity
    time_in_system: Quantity
    batch_size: Optional[int] = None
    batch_service_time: Optional[Quantity] = None
    utilization_target: Optional[float] = None

    @property
    def in_flight(self) -> float:
        from ..physics.serving import calc_littles_law_occupancy
        return calc_littles_law_occupancy(self.arrival_rate, self.time_in_system).to("").magnitude

    @property
    def replica_throughput(self) -> Quantity:
        return (self.batch_size / self.batch_service_time).to("1/second")

    @property
    def throughput_floor(self) -> int:
        return _math.ceil((self.arrival_rate / self.replica_throughput).to("").magnitude)

    @property
    def replicas_needed(self) -> int:
        return _math.ceil((self.arrival_rate / (self.utilization_target * self.replica_throughput))
                          .to("").magnitude)

    @property
    def executing_positions(self) -> int:
        return self.replicas_needed * self.batch_size

    @property
    def mean_executing(self) -> float:
        return (self.arrival_rate * self.batch_service_time).to("").magnitude

    @property
    def service_utilization(self) -> float:
        return (self.arrival_rate / (self.replicas_needed * self.replica_throughput)).to("").magnitude


class RequestTraceScenario(BaseScenario):
    """A published request trace: arrival rate, token lengths, and fixed-batch waste (``RequestTraceModel``)."""
    series: Any                          # a Literature request-trace series
    timestamp_column: str = "TIMESTAMP"
    prompt_column: str = "ContextTokens"
    output_column: str = "GeneratedTokens"
    batch_size: int = 8

    @property
    def result(self):
        from .solvers import RequestTraceModel
        rows = self.series.entries
        return RequestTraceModel().solve([r[self.timestamp_column] for r in rows],
                                         [r[self.prompt_column] for r in rows],
                                         [r[self.output_column] for r in rows], self.batch_size)

    @property
    def output_tokens(self) -> List[float]:
        return [r[self.output_column] for r in self.series.entries]

    @property
    def prompt_tokens(self) -> List[float]:
        return [r[self.prompt_column] for r in self.series.entries]

    # Token summaries without parsing arrival times (cheap enough for scenario inputs).
    @property
    def median_request_tokens(self) -> int:
        """Median prompt plus median output, whole tokens."""
        import statistics
        return int(statistics.median(self.prompt_tokens) + statistics.median(self.output_tokens))

    @property
    def mean_output_tokens(self) -> float:
        out = self.output_tokens
        return sum(out) / len(out)

    @property
    def mean_resident_tokens(self) -> float:
        """Mean tokens a request holds mid-generation: its prompt plus half its answer."""
        p = self.prompt_tokens
        return sum(p) / len(p) + self.mean_output_tokens / 2


class TrafficProfileScenario(BaseScenario):
    """A binned request series: peak, trough, and weekday swing (``TrafficProfileModel``)."""
    series: Any
    bin_column: str
    count_column: str
    bin_width: Quantity

    @property
    def result(self):
        from .solvers import TrafficProfileModel
        rows = self.series.entries
        return TrafficProfileModel().solve([r[self.bin_column] for r in rows],
                                           [r[self.count_column] for r in rows], self.bin_width)


class ServingFrontierScenario(BaseScenario):
    """Measured serving configurations: per-accelerator frontier and best rate at a per-user target."""
    series: Any                          # rows with hardware, interactivity, and throughput columns
    x_column: str = "interactivity_tok_s_user"
    y_column: str = "output_tok_s_gpu"
    targets: Optional[List[float]] = None    # per-user rates at which configurations are compared

    def rows(self, hardware: Optional[str] = None, **match) -> List[dict]:
        out = [r for r in self.series.entries if hardware is None or r["hardware"] == hardware]
        return [r for r in out if all(r.get(k) == v for k, v in match.items())]

    def solve(self, hardware: str, minimum: float, **match):
        from .solvers import ServingFrontierModel
        return ServingFrontierModel().solve(self.rows(hardware, **match), self.x_column, self.y_column, minimum)

    def frontier(self, hardware: str):
        from .solvers import ServingFrontierModel
        return ServingFrontierModel.frontier(self.rows(hardware), self.x_column, self.y_column)

    def best_at(self, hardware: str, minimum: float, **match) -> float:
        from .solvers import ServingFrontierModel
        return ServingFrontierModel.best_at(self.rows(hardware, **match), self.x_column, self.y_column, minimum)

    def split_gain(self, hardware: str, minimum: float) -> float:
        """Best split-pool (disaggregated) rate over best colocated rate at ``minimum``."""
        return self.best_at(hardware, minimum, disagg=1.0) / self.best_at(hardware, minimum, disagg=0.0)

    def sweep(self, hardware: str, **match) -> List[dict]:
        """Matching rows in order of concurrency."""
        return sorted(self.rows(hardware, **match), key=lambda r: r["conc"])

    def max_concurrency_at(self, hardware: str, minimum: float, **match) -> float:
        return max(r["conc"] for r in self.rows(hardware, **match) if r[self.x_column] >= minimum)

    @property
    def config_count(self) -> int:
        return len(self.series.entries)


class TokenCostScenario(BaseScenario):
    """Cost per output token on the best measured configuration at each per-user target."""
    frontier: ServingFrontierScenario
    hardware: str
    targets: List[float]                 # per-user output tokens per second
    gpu_hour_price: Quantity

    def throughput(self, target: float) -> float:
        return self.frontier.best_at(self.hardware, target)

    def cost_per_million(self, target: float) -> Quantity:
        rate = Q_(self.throughput(target), "1/second")
        return (self.gpu_hour_price / rate * 1e6).to("USD")

    @property
    def spread(self) -> float:
        costs = [self.cost_per_million(t) for t in self.targets]
        return (costs[-1] / costs[0]).to("").magnitude

    def answer_cost(self, output_tokens: float, target: float) -> Quantity:
        return (self.cost_per_million(target) * output_tokens / 1e6).to("USD")




# --- KV cache management: footprint, paging, prefix reuse, pauses, residency (added 2026-10-20) ---
class RequestZonesScenario(BaseScenario):
    """One serialized request laid out in zones, the leading ones an invariant shared prefix."""
    zones: Dict[str, int]            # zone name -> tokens, in serialization order
    shared_zones: int                # leading zones that form the invariant prefix

    @property
    def total_tokens(self) -> int:
        return sum(self.zones.values())

    @property
    def shared_tokens(self) -> int:
        return sum(list(self.zones.values())[:self.shared_zones])

    @property
    def shared_share(self) -> float:
        return self.shared_tokens / self.total_tokens


# KvAdmissionScenario: copied verbatim from the v2-10 inference branch.
class KvAdmissionScenario(BaseScenario):
    """A replica's KV budget against one context length (``KvAdmissionModel``)."""
    workload: Workload
    accelerators: int
    hbm_per_accelerator: Quantity
    weight_bytes_per_param: Quantity
    reserve: Quantity
    context_tokens: int
    kv_bytes_per_elem: Quantity

    @property
    def result(self):
        from .solvers import KvAdmissionModel
        return KvAdmissionModel().solve(self.workload, self.accelerators, self.hbm_per_accelerator,
                                        self.weight_bytes_per_param, self.reserve, self.context_tokens,
                                        self.kv_bytes_per_elem)


class KvInFlightScenario(BaseScenario):
    """KV state held by every request in flight, against one replica's KV budget."""
    occupancy: LittlesLawScenario
    resident_tokens: float               # mean tokens each request holds while in flight
    admission: KvAdmissionScenario

    @property
    def kv_in_flight(self) -> Quantity:
        r = self.admission.result
        return (self.occupancy.in_flight * self.resident_tokens * r.kv_per_token).to("TB")

    @property
    def replicas_by_kv(self) -> float:
        return (self.kv_in_flight / self.admission.result.kv_budget).to("").magnitude


class PrefixSharingScenario(BaseScenario):
    """KV cache for users who share one prompt prefix, stored per user or once."""
    workload: Workload
    prefix_tokens: int
    users: int
    unique_tokens_per_user: int
    kv_width: int                        # key-value heads x head width cached per layer
    bytes_per_elem: Quantity

    @property
    def tokens_per_user(self) -> int:
        return self.prefix_tokens + self.unique_tokens_per_user

    @property
    def tokens_unshared(self) -> int:
        return self.tokens_per_user * self.users

    @property
    def tokens_shared(self) -> int:
        return self.prefix_tokens + self.unique_tokens_per_user * self.users

    def _kv(self, tokens: int) -> Quantity:
        from ..physics.memory import calc_kv_cache_size
        return calc_kv_cache_size(self.workload.layers, 1, self.kv_width, tokens, 1, self.bytes_per_elem)

    @property
    def kv_unshared(self) -> Quantity:
        return self._kv(self.tokens_unshared)

    @property
    def kv_shared(self) -> Quantity:
        return self._kv(self.tokens_shared)

    @property
    def reduction(self) -> float:
        return 1 - (self.kv_shared / self.kv_unshared).to("").magnitude

    @property
    def capacity_ratio(self) -> float:
        return (self.kv_unshared / self.kv_shared).to("").magnitude


class PrecisionDividendScenario(BaseScenario):
    """Maximum batch before and after lower-precision weights and KV (``PrecisionDividendModel``)."""
    workload: Workload
    tp_size: int
    hbm_per_accelerator: Quantity
    context_tokens: int
    kv_heads: int
    weight_bytes: Quantity
    weight_bytes_low: Quantity
    kv_bytes: Quantity
    kv_bytes_low: Quantity

    @property
    def result(self):
        from .solvers import PrecisionDividendModel
        return PrecisionDividendModel().solve(self.workload, self.tp_size, self.hbm_per_accelerator,
                                              self.context_tokens, self.kv_heads, self.weight_bytes,
                                              self.weight_bytes_low, self.kv_bytes, self.kv_bytes_low)
class KvAttentionDesignScenario(BaseScenario):
    """Bytes per token across registry models, with one MHA, one GQA, and one MLA reference."""
    models: List[Workload]           # display order
    mha: Workload
    gqa: Workload
    mla: Workload
    precision: str = "fp16"
    session_tokens: int              # one long session, for the GQA and MLA references

    @property
    def result(self):
        from .solvers import KvAttentionDesignModel
        return KvAttentionDesignModel().solve(self.models, self.mha, self.gqa, self.mla, self.precision,
                                              self.session_tokens)


class PagedAllocationScenario(BaseScenario):
    """Requests under a length cap in ``pool``'s KV pool: full-cap reservation against paged blocks."""
    pool: KvAdmissionScenario
    cap: int                         # admission cap, tokens
    mean_final: int                  # mean final length, tokens
    small_block: int                 # tokens per block
    large_block: int

    @property
    def result(self):
        from .solvers import PagedAllocationModel
        p = self.pool.result
        return PagedAllocationModel().solve(p.kv_per_token, p.kv_budget, self.cap, self.mean_final,
                                            self.small_block, self.large_block)


class KvForkCopyScenario(BaseScenario):
    """Candidates forked from one prefix, each copying the prefix's KV state without block sharing."""
    pool: KvAdmissionScenario
    prefix_tokens: int
    branches: int

    @property
    def copied(self) -> Quantity:
        return self.branches * self.prefix_tokens * self.pool.result.kv_per_token


class PrefixRebuildScenario(BaseScenario):
    """A cached prefix on ``node``: the prefill a miss rebuilds against one short suffix pass on a hit."""
    workload: Workload
    node: Any
    precision: str = "fp16"
    prefix_tokens: int
    suffix_tokens: int
    utilization: float               # sustained share of the node's peak rate

    @property
    def result(self):
        from .solvers import PrefixRebuildModel
        return PrefixRebuildModel().solve(self.workload, self.node, self.precision, self.prefix_tokens,
                                          self.suffix_tokens, self.utilization)


class ChunkedPrefillScenario(BaseScenario):
    """A long prompt advanced ``chunk_tokens`` per iteration beside decode."""
    # Same name and first fields as the v2-10 inference branch's type (prompt_tokens,
    # chunk_tokens, iterations); the decode-gap inputs below are optional additions.
    prompt_tokens: int
    chunk_tokens: int
    prefill: Optional[PrefixRebuildScenario] = None   # model, node, and sustained rate
    decodes: Optional[int] = None                     # ongoing decode requests

    @property
    def iterations(self) -> int:
        return _math.ceil(self.prompt_tokens / self.chunk_tokens)

    @property
    def result(self):
        from .solvers import ChunkedPrefillModel
        p = self.prefill.result
        return ChunkedPrefillModel().solve(self.prefill.workload.parameters, p.sustained_rate,
                                           p.weight_read_time, self.decodes, self.prompt_tokens,
                                           self.chunk_tokens)


class TensorParallelSpeedupScenario(BaseScenario):
    """Stated per-layer compute split ``tp_size`` ways, plus the AllReduces each layer adds."""
    serial_layer_times: Dict[str, Quantity]  # compute per layer on one device, by phase
    tp_size: int
    allreduce_time: Quantity             # per collective
    collectives_per_layer: int = 2
    layers: int

    def sharded(self, phase: str) -> Quantity:
        return (self.serial_layer_times[phase] / self.tp_size).to("ms")

    @property
    def serial_layer_time(self) -> Quantity:
        return sum(self.serial_layer_times.values(), Q_(0, "ms")).to("ms")

    @property
    def sharded_compute_time(self) -> Quantity:
        return (self.serial_layer_time / self.tp_size).to("ms")

    @property
    def parallel_layer_time(self) -> Quantity:
        return (sum((self.sharded(p) for p in self.serial_layer_times), Q_(0, "ms"))
                + self.collectives_per_layer * self.allreduce_time).to("ms")

    @property
    def speedup(self) -> float:
        return (self.serial_layer_time / self.parallel_layer_time).to("").magnitude

    @property
    def serial_total(self) -> Quantity:
        return (self.serial_layer_time * self.layers).to("ms")

    @property
    def parallel_total(self) -> Quantity:
        return (self.parallel_layer_time * self.layers).to("ms")


class DecodeBudgetScenario(BaseScenario):
    """A constructed decode-step budget, cut interval by interval, against a TPOT target."""
    workload: Workload
    tp_size: int
    accelerator: HardwareNode
    weight_bytes_per_param: Quantity
    tpot_target: Quantity
    baseline_step: Quantity
    intervals: List[str]                 # GEMM, attention, element-wise, AllReduce, launch
    baseline_shares: List[float]
    gemm_cut: float                      # after precision: share of GEMM time removed
    attention_cut: float                 # after fusion: share of attention time removed
    fused_elementwise: Quantity
    replay_overhead: Quantity
    collective_launch: Quantity
    allreduces_per_layer: int = 2

    @property
    def baseline(self) -> List[float]:
        """Baseline intervals in ms."""
        total = self.baseline_step.to("ms").magnitude
        return [total * s for s in self.baseline_shares]

    @property
    def after_precision(self) -> List[float]:
        r = self.baseline.copy()
        r[0] *= 1 - self.gemm_cut
        return r

    @property
    def after_fusion(self) -> List[float]:
        r = self.after_precision.copy()
        r[1] *= 1 - self.attention_cut
        r[2] = self.fused_elementwise.to("ms").magnitude
        r[4] = self.replay_overhead.to("ms").magnitude
        return r

    @property
    def totals(self) -> List[float]:
        return [sum(self.baseline), sum(self.after_precision), sum(self.after_fusion)]

    @property
    def weight_shard(self) -> Quantity:
        return (self.workload.size_in_bytes(self.weight_bytes_per_param) / self.tp_size).to("GB")

    @property
    def floor(self) -> Quantity:
        """Time to stream one weight shard at full HBM bandwidth."""
        return (self.weight_shard / self.accelerator.memory.bandwidth).to("ms")

    @property
    def allreduces_per_step(self) -> int:
        return self.allreduces_per_layer * self.workload.layers

    @property
    def allreduce_per_collective(self) -> Quantity:
        return Q_(self.baseline[3] / self.allreduces_per_step, "ms").to("microsecond")

    @property
    def collective_estimate(self) -> Quantity:
        """Launch plus transfer of one batch-one activation AllReduce over NVLink."""
        from ..core.units import BYTES_FP16
        payload = 2 * self.workload.hidden_dim * BYTES_FP16
        return (self.collective_launch + payload / self.accelerator.nvlink.bandwidth_per_direction
                ).to("microsecond")


class SpeculativeRoundScenario(BaseScenario):
    """A draft proposes ``draft_length`` tokens; the target verifies them in one step."""
    acceptance: float
    draft_length: int
    draft_time: Quantity
    verify_time: Quantity

    @property
    def expected_tokens(self) -> float:
        from ..physics.serving import calc_speculative_expected_tokens
        return calc_speculative_expected_tokens(self.acceptance, self.draft_length)

    @property
    def round_time(self) -> Quantity:
        return (self.draft_time + self.verify_time).to("ms")

    @property
    def per_token_time(self) -> Quantity:
        return (self.round_time / self.expected_tokens).to("ms")

    @property
    def tpot_gain(self) -> float:
        return (self.verify_time / self.per_token_time).to("").magnitude


class WeightedRoutingScenario(BaseScenario):
    """Traffic split over server pools in proportion to each pool's per-server capacity."""
    servers: Dict[str, int]
    capacity: Dict[str, float]           # requests per second one server sustains
    offered: float                       # requests per second offered to the fleet

    @property
    def total_capacity(self) -> float:
        return sum(self.servers[k] * self.capacity[k] for k in self.servers)

    def weight(self, pool: str) -> float:
        return self.capacity[pool] / self.total_capacity

    def utilization(self, pool: str) -> float:
        return self.offered * self.weight(pool) / self.capacity[pool]

    @property
    def equal_split_load(self) -> float:
        return self.offered / sum(self.servers.values())

    def equal_split_utilization(self, pool: str) -> float:
        return self.equal_split_load / self.capacity[pool]


class TailEventRateScenario(BaseScenario):
    """How often an event of probability ``event_probability`` per request occurs at ``request_rate``."""
    request_rate: Quantity
    event_probability: float

    @property
    def events_per_second(self) -> float:
        return (self.request_rate * self.event_probability).to("1/second").magnitude


class ColdStartScenario(BaseScenario):
    """A new replica: provision, fetch the weight artifact, load and warm up."""
    workload: Workload
    weight_bytes_per_param: Quantity
    fetch_bandwidth: Quantity
    provisioning_time: Quantity
    load_warmup_time: Quantity
    reactive_peak_replicas: int          # peak fleet when scaling only reacts
    predictive_peak_replicas: int        # peak fleet when replicas start ahead of the ramp
    prewarm_lead: Quantity

    @property
    def artifact(self) -> Quantity:
        return self.workload.size_in_bytes(self.weight_bytes_per_param).to("GB")

    @property
    def fetch_time(self) -> Quantity:
        return (self.artifact / self.fetch_bandwidth).to("second")

    @property
    def cold_start_time(self) -> Quantity:
        return (self.provisioning_time + self.fetch_time + self.load_warmup_time).to("second")

    @property
    def peak_saving(self) -> float:
        return 1 - self.predictive_peak_replicas / self.reactive_peak_replicas


class NoisyNeighborScenario(BaseScenario):
    """Tenants sharing a KV pool and a replica, one of them bursting, with and without quotas."""
    kv_pool: Quantity
    pool_tenants: int
    swollen_share: Quantity              # long-context tenant's allocation without limits
    base_sequences: int                  # sequences per equal share
    base_rate: float                     # requests per second per tenant
    burst_rate: float
    slo: Quantity                        # p99 target
    unprotected_p99_low: Quantity
    unprotected_p99_high: Quantity
    quota_rate: float
    protected_p99: Quantity

    @property
    def base_share(self) -> Quantity:
        return self.kv_pool / self.pool_tenants

    @property
    def other_share(self) -> Quantity:
        return (self.kv_pool - self.swollen_share) / (self.pool_tenants - 1)

    @property
    def other_sequences(self) -> int:
        per_seq = self.base_share / self.base_sequences
        return int((self.other_share / per_seq).to("").magnitude)

    @property
    def batch_reduction(self) -> float:
        return 1 - self.other_sequences / self.base_sequences

    @property
    def burst_multiple(self) -> float:
        return self.burst_rate / self.base_rate


class ReasoningLatencyScenario(BaseScenario):
    """Serial reasoning tokens before the answer, at a fixed time per token."""
    per_token_time: Quantity
    reasoning_tokens: int

    @property
    def reasoning_time(self) -> Quantity:
        return (self.per_token_time * self.reasoning_tokens).to("second")

    @property
    def expansion(self) -> int:
        return self.reasoning_tokens


class FixedBatchScheduleScenario(BaseScenario):
    """Requests in one batch of fixed membership, one token per iteration."""
    output_lengths: List[int]
    prompt_lengths: Optional[List[int]] = None
    iteration_time: Quantity

    @property
    def total_tokens(self) -> List[int]:
        return [p + o for p, o in zip(self.prompt_lengths, self.output_lengths)]

    @property
    def max_output(self) -> int:
        return max(self.output_lengths)

    @property
    def mean_output(self) -> float:
        return sum(self.output_lengths) / len(self.output_lengths)

    @property
    def slot_iterations(self) -> int:
        return len(self.output_lengths) * self.max_output

    @property
    def useful_iterations(self) -> int:
        return sum(self.output_lengths)

    @property
    def idle_share(self) -> float:
        from ..physics.serving import calc_padding_waste_fraction
        return calc_padding_waste_fraction(self.output_lengths)

    @property
    def requests_advanced(self) -> int:
        """Requests served in the same iterations when each freed slot is refilled once."""
        return len(self.output_lengths) + sum(1 for x in self.output_lengths if x < self.max_output)


class BallsIntoBinsScenario(BaseScenario):
    """n equal jobs placed into n empty replicas, random or best of two (``BallsIntoBinsModel``)."""
    bins: List[int]
    choices: List[int]
    trials: Dict[int, int]
    seed: int

    @property
    def result(self):
        from .solvers import BallsIntoBinsModel
        return BallsIntoBinsModel().solve(self.bins, self.choices, self.trials, self.seed)


# --- Retrieval from long-term memory (added 2026-10-20) ----------------------
# What a read stages, ranks, streams, and maintains: the solvers live in
# engine/solvers/retrieval.py. Token counts are serialized-request counts.

class RetrievalPayloadScenario(BaseScenario):
    """Whole serialized records staged in what a call's window leaves for retrieval."""
    window: int
    output_reserve: int
    other_input: int                 # system prompt, tool schemas, history
    envelope: int                    # response envelope, paid once
    body: int                        # body tokens per record
    metadata: int                    # provenance tokens per record
    metadata_scale: Optional[int] = None     # a metadata multiple to compare

    @property
    def result(self):
        from .solvers import RetrievalPayloadModel
        return RetrievalPayloadModel().solve(self.window, self.output_reserve, self.other_input,
                                             self.envelope, self.body, self.metadata,
                                             self.metadata_scale)


class Bm25PairScenario(BaseScenario):
    """One query term scored by BM25 in a short and a long document of one corpus."""
    documents: int
    matching_documents: int
    average_length: float            # analyzer terms
    short_length: float
    short_frequency: float
    long_length: float
    long_frequency: float
    k1: float
    b: float

    @property
    def result(self):
        from .solvers import Bm25PairModel
        return Bm25PairModel().solve(self.documents, self.matching_documents, self.average_length,
                                     self.short_length, self.short_frequency, self.long_length,
                                     self.long_frequency, self.k1, self.b)


class ExactScanScenario(BaseScenario):
    """An exact vector scan on one host: bytes, intensity, and the bandwidth floor."""
    host: HardwareNode
    vectors: int
    dimension: int
    bytes_per_value: Quantity
    searches: Optional[int] = None           # searches in one trajectory
    filtered_vectors: Optional[int] = None   # eligible set after a scope filter
    budget: Optional[Quantity] = None        # per-search latency budget

    @property
    def result(self):
        from .solvers import ExactScanFloorModel
        return ExactScanFloorModel().solve(self.host, self.vectors, self.dimension, self.bytes_per_value,
                                           self.searches, self.filtered_vectors, self.budget)


class AnnFootprintScenario(BaseScenario):
    """Resident bytes of a graph index and of product-quantization codes over one corpus."""
    vectors: int
    dimension: int
    bytes_per_value: Quantity
    links_per_node: int
    link_bytes: Quantity
    subvectors: int
    centroids: int
    code_bytes: Quantity
    table_entry_bytes: Quantity

    @property
    def result(self):
        from .solvers import AnnFootprintModel
        return AnnFootprintModel().solve(self.vectors, self.dimension, self.bytes_per_value,
                                         self.links_per_node, self.link_bytes, self.subvectors,
                                         self.centroids, self.code_bytes, self.table_entry_bytes)


class GraphNeighborhoodScenario(BaseScenario):
    """A typed traversal's exact-tree neighborhood against a call's retrieval allowance."""
    window: int
    output_reserve: int
    other_input: int
    children: int
    depth: int                       # the seed is depth zero
    node_tokens: int                 # serialized tokens per node, provenance included
    envelope_tokens: int             # edges and envelope, paid once

    @property
    def result(self):
        from .solvers import GraphNeighborhoodModel
        return GraphNeighborhoodModel().solve(self.window, self.output_reserve, self.other_input,
                                              self.children, self.depth, self.node_tokens,
                                              self.envelope_tokens)


class IndexMaintenanceScenario(BaseScenario):
    """One edit's synchronous re-indexing against exclusion of the stale entries."""
    files: int
    chunks_per_file: int
    parse_time: Quantity             # per file
    postings_time: Quantity          # per chunk
    local_batch_time: Quantity       # one local embedding batch
    batch: int                       # chunks per embedding batch
    remote_request_time: Quantity    # one remote embedding request
    exclusion_write_time: Quantity   # per exclusion entry
    candidates: int                  # candidates checked per query
    exclusion_check_time: Quantity   # per candidate

    @property
    def result(self):
        from .solvers import IndexMaintenanceModel
        return IndexMaintenanceModel().solve(self.files, self.chunks_per_file, self.parse_time,
                                             self.postings_time, self.local_batch_time, self.batch,
                                             self.remote_request_time, self.exclusion_write_time,
                                             self.candidates, self.exclusion_check_time)


class StorageAmplificationScenario(BaseScenario):
    """Auxiliary bytes and re-index time of a lexical-plus-graph and a lexical-plus-dense store."""
    files: int
    tokens_per_file: int
    bytes_per_file: Quantity
    postings_per_token: Quantity     # bytes of postings per token
    edges_per_file: int
    edge_bytes: Quantity
    passage_tokens: int              # tokens per dense chunk
    dimension: int
    bytes_per_value: Quantity
    links_per_node: int
    link_bytes: Quantity
    edited_files: int
    lexical_rate: float              # files per second
    parse_rate: float                # files per second
    embed_rate: float                # chunks per second

    @property
    def result(self):
        from .solvers import StorageAmplificationModel
        return StorageAmplificationModel().solve(
            self.files, self.tokens_per_file, self.bytes_per_file, self.postings_per_token,
            self.edges_per_file, self.edge_bytes, self.passage_tokens, self.dimension,
            self.bytes_per_value, self.links_per_node, self.link_bytes, self.edited_files,
            self.lexical_rate, self.parse_rate, self.embed_rate)


class MaintainedPolicyCostScenario(BaseScenario):
    """A memory policy that pays upkeep (a share of the baseline attempt) to raise acceptance."""
    baseline_acceptance: float
    maintained_acceptance: float
    maintenance_share: float         # upkeep per attempt, fraction of the baseline attempt cost

    @property
    def result(self):
        from .solvers import MaintainedPolicyCostModel
        return MaintainedPolicyCostModel().solve(self.baseline_acceptance, self.maintained_acceptance,
                                                 self.maintenance_share)


class WordOverlapScenario(BaseScenario):
    """Two texts compared by the distinct words they share."""
    first: str
    second: str

    @property
    def result(self):
        from .solvers import WordOverlapModel
        return WordOverlapModel().solve(self.first, self.second)


# --- Physical demonstration data: samples, labels, yield, coverage (added 2026-10-20) ---

class CorrelatedLogScenario(BaseScenario):
    """``duration`` of records logged at ``rate`` whose adjacent records have lag-one correlation
    ``lag1_correlation``: how few independent samples the record count holds."""
    rate: Quantity
    duration: Quantity
    lag1_correlation: float

    @property
    def records(self) -> float:
        return (self.rate * self.duration).to("").magnitude

    @property
    def record_interval(self) -> Quantity:
        return (1 / self.rate).to("ms")

    @property
    def effective_samples(self) -> float:
        from ..physics.statistics import calc_effective_sample_size
        return calc_effective_sample_size(self.records, self.lag1_correlation)

    @property
    def oversampling(self) -> float:
        """Records per effective independent sample."""
        return self.records / self.effective_samples


class ImitationBoundScenario(BaseScenario):
    """Worst-case imitation cost over one ``episode`` at ``control_rate`` with per-step error
    ``per_step_error``: open loop, behavioral cloning, and aggregation with recoverability ``u``."""
    episode: Quantity
    control_rate: Quantity
    per_step_error: float
    recoverability: float = 1.0

    @property
    def horizon_steps(self) -> int:
        return int(round((self.episode * self.control_rate).to("").magnitude))

    @property
    def open_loop(self) -> float:
        """S eps: the expected cost if errors did not feed back."""
        return self.horizon_steps * self.per_step_error

    @property
    def cloning(self) -> float:
        from ..physics.statistics import calc_behavior_cloning_cost_bound
        return calc_behavior_cloning_cost_bound(self.horizon_steps, self.per_step_error)

    @property
    def aggregation(self) -> float:
        from ..physics.statistics import calc_dagger_cost_bound
        return calc_dagger_cost_bound(self.horizon_steps, self.per_step_error, self.recoverability)

    @property
    def vacuous_error(self) -> float:
        """Per-step error above which the cloning bound exceeds the episode maximum S."""
        return 1.0 / self.horizon_steps

    @property
    def cloning_over_maximum(self) -> float:
        """The cloning bound as a fraction of the episode maximum S."""
        return self.cloning / self.horizon_steps


class PairingLagScenario(BaseScenario):
    """An observation paired with an action ``lag`` later, against a ``clearance``, at a transit
    speed and at ``site``'s guarded and fast latch approaches."""
    site: Any                    # EmbodiedScenario (latch approach speeds)
    lag: Quantity
    clearance: Quantity
    transit_speed: Quantity
    shutter_skew_low: Quantity   # full-frame rolling-shutter skew, fast readout
    shutter_skew_high: Quantity  # full-frame rolling-shutter skew, one frame

    @property
    def guarded_speed(self) -> Quantity:
        return self.site.v_latch_approach

    @property
    def fast_speed(self) -> Quantity:
        return self.site.v_latch_high

    def displacement(self, speed: Quantity, lag: Optional[Quantity] = None) -> Quantity:
        """Pairing error: speed times lag."""
        return (speed * (self.lag if lag is None else lag)).to("mm")

    def lag_ceiling(self, speed: Quantity) -> Quantity:
        """Largest lag that keeps the pairing error inside the clearance."""
        return (self.clearance / speed).to("ms")

    @property
    def displacement_transit(self) -> Quantity:
        return self.displacement(self.transit_speed)

    @property
    def displacement_guarded(self) -> Quantity:
        return self.displacement(self.guarded_speed)

    @property
    def displacement_fast(self) -> Quantity:
        return self.displacement(self.fast_speed)

    @property
    def transit_over_clearance(self) -> float:
        return (self.displacement_transit / self.clearance).to("").magnitude

    @property
    def fast_over_clearance(self) -> float:
        return (self.displacement_fast / self.clearance).to("").magnitude

    @property
    def lag_ceiling_guarded(self) -> Quantity:
        return self.lag_ceiling(self.guarded_speed)

    @property
    def lag_ceiling_fast(self) -> Quantity:
        return self.lag_ceiling(self.fast_speed)

    @property
    def skew_displacement_transit(self) -> Quantity:
        return self.displacement(self.transit_speed, self.shutter_skew_high)

    @property
    def skew_displacement_fast(self) -> Quantity:
        return self.displacement(self.fast_speed, self.shutter_skew_high)


class TeleopLoopScenario(BaseScenario):
    """One teleoperated correction: display, human, command transport, and mechanical response."""
    display: Quantity
    human: Quantity
    command: Quantity
    mechanical: Quantity

    @property
    def loop(self) -> Quantity:
        return (self.display + self.human + self.command + self.mechanical).to("ms")

    @property
    def command_issued(self) -> Quantity:
        """Time after the event at which the operator's correction is commanded."""
        return (self.display + self.human).to("ms")

    @property
    def human_share(self) -> float:
        return (self.human / self.loop).to("").magnitude


class TeleopIngestionScenario(BaseScenario):
    """Raw telemetry of one teleoperated session on ``machine``: its navigation and wrist cameras,
    ``external_cameras`` RGB-D cameras, and the kinematic record at the permission tick."""
    machine: Any                   # EmbodiedMobileManipulator (cameras, axes, permission rate)
    external_camera: Any           # Sensors.Camera entry (RGB and aligned depth)
    external_cameras: int
    rgb_bytes_per_pixel: int
    depth_bytes_per_pixel: int
    values_per_axis: int
    bytes_per_value: int
    timestamp_bytes: int
    shift: Quantity
    compression_ratio: float
    slow_drive_bandwidth: Quantity
    fast_drive_bandwidth: Quantity

    @property
    def record_bytes(self) -> int:
        return self.machine.servo_axes * self.values_per_axis * self.bytes_per_value + self.timestamp_bytes

    @property
    def kinematic_rate(self) -> Quantity:
        return (self.record_bytes * Q_(1, "byte") * self.machine.permission_rate).to("MB/s")

    def _budget(self, camera, count, depth, kinematics=None):
        from ..physics.robotics.perception import calc_teleop_ingestion_budget
        fps = camera.frame_rate.to("Hz").magnitude
        extra = dict(depth_width=camera.resolution_width, depth_height=camera.resolution_height,
                     depth_fps=fps, depth_bytes_per_pixel=self.depth_bytes_per_pixel) if depth else {}
        return calc_teleop_ingestion_budget(
            num_cameras=count, rgb_width=camera.resolution_width, rgb_height=camera.resolution_height,
            rgb_fps=fps, rgb_bytes_per_pixel=self.rgb_bytes_per_pixel,
            kinematics_bytes_per_sec=kinematics, **extra)

    @property
    def nav_budget(self) -> dict:
        return self._budget(self.machine.nav_camera, 1, False, self.kinematic_rate)

    @property
    def wrist_budget(self) -> dict:
        return self._budget(self.machine.wrist_camera, 1, False)

    @property
    def external_budget(self) -> dict:
        return self._budget(self.external_camera, self.external_cameras, True)

    @property
    def nav_rate(self) -> Quantity:
        return self.nav_budget["bw_rgb"]

    @property
    def wrist_rate(self) -> Quantity:
        return self.wrist_budget["bw_rgb"]

    @property
    def external_rgb_rate(self) -> Quantity:
        return self.external_budget["bw_rgb"]

    @property
    def external_depth_rate(self) -> Quantity:
        return self.external_budget["bw_depth"]

    @property
    def external_rate(self) -> Quantity:
        return self.external_budget["bw_vision"]

    @property
    def total_rate(self) -> Quantity:
        return (self.nav_budget["bw_total"] + self.wrist_budget["bw_total"]
                + self.external_budget["bw_total"]).to("MB/s")

    @property
    def storage_per_hour(self) -> Quantity:
        return (self.nav_budget["storage_per_hour"] + self.wrist_budget["storage_per_hour"]
                + self.external_budget["storage_per_hour"]).to("TB")

    @property
    def storage_per_shift(self) -> Quantity:
        return (self.storage_per_hour * self.shift.to("hour").magnitude).to("TB")

    @property
    def nav_over_slow_drive(self) -> float:
        return (self.nav_rate / self.slow_drive_bandwidth).to("").magnitude

    @property
    def kinematic_share(self) -> float:
        return (self.kinematic_rate / self.total_rate).to("").magnitude

    @property
    def compressed_rate(self) -> Quantity:
        return (self.total_rate / self.compression_ratio).to("MB/s")

    @property
    def compressed_storage_per_hour(self) -> Quantity:
        return (self.storage_per_hour / self.compression_ratio).to("TB")


class DemonstrationCampaignScenario(BaseScenario):
    """Scheduled sessions of attempt-and-reset cycles: demonstrations kept per session, the
    campaign that keeps ``target_kept`` of them, and the wear it puts on a gear and a cable."""
    session: Quantity              # scheduled session length
    startup: Quantity
    calibration: Quantity
    rest: Quantity
    attempt: Quantity
    reset: Quantity
    failure_rate: float            # attempts that fail in the trial
    qa_rate: float                 # surviving attempts discarded by quality filtering
    target_kept: int
    gear_reversals_per_attempt: int
    cable_flexes_opening: int
    cable_flexes_return: int
    gear_rated_cycles: int
    cable_rated_cycles: int
    repeat_campaigns: int = 1      # campaigns compared against the rated lives

    @property
    def active(self) -> Quantity:
        """Session time left for attempts and resets."""
        s = [q.to("second") for q in (self.session, self.startup, self.calibration, self.rest)]
        return (s[0] - s[1] - s[2] - s[3]).to("minute")

    @property
    def cycle(self) -> Quantity:
        return (self.attempt + self.reset).to("second")

    @property
    def attempts_per_session(self) -> int:
        return int(self.active.to("second").magnitude // self.cycle.to("second").magnitude)

    @property
    def idle(self) -> Quantity:
        """Active time too short for one more cycle."""
        return (self.active.to("second") - self.attempts_per_session * self.cycle.to("second")).to("second")

    @property
    def retention(self) -> float:
        return (1.0 - self.failure_rate) * (1.0 - self.qa_rate)

    @property
    def kept_per_session(self) -> float:
        return self.attempts_per_session * self.retention

    @property
    def kept_time(self) -> Quantity:
        return (self.kept_per_session * self.attempt).to("minute")

    @property
    def kept_share(self) -> float:
        return (self.kept_time / self.session).to("").magnitude

    @property
    def reset_time(self) -> Quantity:
        return (self.attempts_per_session * self.reset).to("second")

    @property
    def attempt_time(self) -> Quantity:
        return (self.attempts_per_session * self.attempt).to("second")

    @property
    def failed_time(self) -> Quantity:
        return (self.failure_rate * self.attempt_time).to("second")

    @property
    def qa_time(self) -> Quantity:
        return (self.qa_rate * (self.attempt_time - self.failed_time)).to("second")

    @property
    def failed_episodes(self) -> float:
        return self.failure_rate * self.attempts_per_session

    @property
    def qa_episodes(self) -> float:
        return self.qa_rate * (self.attempts_per_session - self.failed_episodes)

    @property
    def campaign_attempts(self) -> int:
        import math
        return int(math.ceil(self.target_kept / self.retention))

    @property
    def campaign_time(self) -> Quantity:
        """Scheduled cell time: one session per ``attempts_per_session`` attempts."""
        return (self.campaign_attempts / self.attempts_per_session * self.session).to("hour")

    @property
    def kept_motion_time(self) -> Quantity:
        return (self.target_kept * self.attempt).to("hour")

    @property
    def overhead_factor(self) -> float:
        return (self.campaign_time / self.kept_motion_time).to("").magnitude

    @property
    def cable_flexes_per_attempt(self) -> int:
        return self.cable_flexes_opening + self.cable_flexes_return

    @property
    def gear_cycles(self) -> int:
        return self.campaign_attempts * self.gear_reversals_per_attempt

    @property
    def cable_cycles(self) -> int:
        return self.campaign_attempts * self.cable_flexes_per_attempt

    @property
    def gear_life_used(self) -> float:
        return self.gear_cycles / self.gear_rated_cycles

    @property
    def cable_life_used(self) -> float:
        return self.cable_cycles / self.cable_rated_cycles

    @property
    def repeat_cable_life_used(self) -> float:
        """Cable rated life used by ``repeat_campaigns`` campaigns."""
        return self.repeat_campaigns * self.cable_life_used


class HubDatasetCountsScenario(BaseScenario):
    """Public datasets on a hub over time and episodes per dataset, from a monthly series and a
    power-of-two episode histogram (all creation years pooled)."""
    monthly: Any                   # Literature series: month, cumulative_datasets
    histogram: Any                 # Literature series: bin_lo, bin_hi, datasets
    small_episodes: int            # "small" datasets hold fewer than this many episodes
    large_episodes: int            # "large" datasets hold at least this many

    def count_at(self, month: str) -> float:
        return {m["month"]: m["cumulative_datasets"] for m in self.monthly.entries}[month]

    @property
    def latest_month(self) -> str:
        return self.monthly.entries[-1]["month"]

    @property
    def latest_count(self) -> float:
        return self.monthly.entries[-1]["cumulative_datasets"]

    def growth_since(self, month: str) -> float:
        return self.latest_count / self.count_at(month)

    @property
    def pooled(self) -> Dict[float, float]:
        """Datasets per bin lower edge, creation years pooled."""
        out: Dict[float, float] = {}
        for b in self.histogram.entries:
            out[b["bin_lo"]] = out.get(b["bin_lo"], 0) + b["datasets"]
        return out

    @property
    def counted(self) -> float:
        return sum(b["datasets"] for b in self.histogram.entries)

    @property
    def small_share(self) -> float:
        return sum(b["datasets"] for b in self.histogram.entries
                   if b["bin_hi"] <= self.small_episodes) / self.counted

    @property
    def large_share(self) -> float:
        return sum(b["datasets"] for b in self.histogram.entries
                   if b["bin_lo"] >= self.large_episodes) / self.counted

    @property
    def median_bin_low(self) -> float:
        run = 0.0
        pooled = self.pooled
        for lo in sorted(pooled):
            run += pooled[lo]
            if run >= self.counted / 2:
                return lo
        raise ValueError("empty histogram")

    @property
    def median_bin_high(self) -> float:
        """Exclusive upper edge of the median bin (power-of-two bins)."""
        return 2 * self.median_bin_low if self.median_bin_low else 1


class PooledCorpusCompositionScenario(BaseScenario):
    """Episode-weighted composition of a pooled cross-embodiment corpus against a machine's
    command rate ``rate_threshold``."""
    series: Any                    # Literature series: one row per constituent dataset
    rate_threshold: Quantity
    top_datasets: int = 2          # how many of a morphology's largest datasets top_share_within counts

    @property
    def rows(self) -> List[dict]:
        return [r for r in self.series.entries if r["episodes"]]

    def row(self, dataset: str) -> dict:
        return next(r for r in self.rows if r["dataset"] == dataset)

    @property
    def datasets(self) -> int:
        return len(self.rows)

    @property
    def total(self) -> float:
        return sum(r["episodes"] for r in self.rows)

    @property
    def by_morphology(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for r in self.rows:
            out[r["robot_morphology"]] = out.get(r["robot_morphology"], 0) + r["episodes"]
        return out

    @property
    def by_rate(self) -> Dict[Optional[float], float]:
        """Episodes per declared command rate in Hz; None collects rows with no stated rate."""
        out: Dict[Optional[float], float] = {}
        for r in self.rows:
            k = r["control_frequency_hz"] if isinstance(r["control_frequency_hz"], float) else None
            out[k] = out.get(k, 0) + r["episodes"]
        return out

    def morphology_share(self, morphology: str) -> float:
        return self.by_morphology.get(morphology, 0) / self.total

    def rate_share(self, rate_hz: Optional[float]) -> float:
        return self.by_rate.get(rate_hz, 0) / self.total

    @property
    def table_top_share(self) -> float:
        return sum(r["episodes"] for r in self.rows if "table" in str(r["scene_type"]).lower()) / self.total

    @property
    def threshold_hz(self) -> float:
        return self.rate_threshold.to("Hz").magnitude

    @property
    def at_or_above_threshold_share(self) -> float:
        return sum(v for k, v in self.by_rate.items() if k is not None and k >= self.threshold_hz) / self.total

    def top_share_within(self, morphology: str, n: Optional[int] = None) -> float:
        """Share of a morphology's episodes supplied by its ``n`` largest datasets (default ``top_datasets``)."""
        n = self.top_datasets if n is None else n
        eps = sorted(r["episodes"] for r in self.rows if r["robot_morphology"] == morphology)
        return sum(eps[-n:]) / sum(eps)


class StateGridScenario(BaseScenario):
    """A state space of ``state_components`` axes cut into ``bins`` per axis, against the records
    ``fleet`` robots log at ``log_rate`` for ``duration``."""
    state_components: Dict[str, int]
    bins: int
    fine_bins: int
    fleet: int
    log_rate: Quantity
    duration: Quantity

    @property
    def dims(self) -> int:
        return sum(self.state_components.values())

    @property
    def cells(self) -> float:
        return float(self.bins) ** self.dims

    @property
    def fine_cells(self) -> float:
        return float(self.fine_bins) ** self.dims

    @property
    def records(self) -> float:
        return self.fleet * (self.log_rate * self.duration).to("").magnitude

    @property
    def max_coverage(self) -> float:
        """Largest fraction of the coarse grid the records could visit."""
        return self.records / self.cells


class ContactGridScenario(BaseScenario):
    """A contact task binned by contact angle and normal force, and the cells each archive occupies.

    ``archives`` maps a name to (angle_low, angle_high, force_low, force_high, held-out RMS action
    error); each archive's occupied region is a rectangle of whole bins.
    """
    angle_min: Quantity
    angle_max: Quantity
    angle_bin: Quantity
    force_max: Quantity
    force_bin: Quantity
    records: int                   # transitions each archive records
    record_rate: Quantity
    archives: Dict[str, Any]

    @property
    def angle_bins(self) -> int:
        return int(round(((self.angle_max - self.angle_min) / self.angle_bin).to("").magnitude))

    @property
    def force_bins(self) -> int:
        return int(round((self.force_max / self.force_bin).to("").magnitude))

    @property
    def cells(self) -> int:
        return self.angle_bins * self.force_bins

    def bin_span(self, name: str):
        """(first angle bin, angle bins, first force bin, force bins) the archive occupies."""
        a_lo, a_hi, f_lo, f_hi, _ = self.archives[name]
        first_a = int(round(((a_lo - self.angle_min) / self.angle_bin).to("").magnitude))
        n_a = int(round(((a_hi - a_lo) / self.angle_bin).to("").magnitude))
        first_f = int(round((f_lo / self.force_bin).to("").magnitude))
        n_f = int(round(((f_hi - f_lo) / self.force_bin).to("").magnitude))
        return first_a, n_a, first_f, n_f

    def occupied(self, name: str) -> int:
        _, n_a, _, n_f = self.bin_span(name)
        return n_a * n_f

    def share(self, name: str) -> float:
        return self.occupied(name) / self.cells

    def empty(self, name: str) -> int:
        return self.cells - self.occupied(name)

    def empty_share(self, name: str) -> float:
        return self.empty(name) / self.cells

    def rms_error(self, name: str) -> Quantity:
        return self.archives[name][4]


class ScenarioTableCoverageScenario(BaseScenario):
    """A scenario table of ``offsets`` by ``speeds`` joined against an archive demonstrated within
    ``logged_offset`` at ``demonstrated_speed``, plus recovery collections at ``added_offsets``."""
    offsets: List[Quantity]
    speeds: List[Quantity]
    logged_offset: Quantity
    demonstrated_speed: Quantity
    added_offsets: List[Quantity] = []

    @property
    def cells(self) -> int:
        return len(self.offsets) * len(self.speeds)

    def supported_cell(self, offset: Quantity, speed: Quantity) -> bool:
        if speed != self.demonstrated_speed:
            return False
        return abs(offset) <= self.logged_offset or any(offset == a for a in self.added_offsets)

    @property
    def supported(self) -> int:
        return sum(self.supported_cell(d, v) for d in self.offsets for v in self.speeds)

    @property
    def absences(self) -> int:
        return self.cells - self.supported


class GripperTransferScenario(BaseScenario):
    """A normalized gripper command from a donor gripper replayed on a target with a shorter stroke
    and a lower continuous force rating, against a part of ``part_width``."""
    command: float                 # normalized opening command, 0 to 1
    donor_stroke: Quantity
    target_stroke: Quantity
    part_width: Quantity
    donor_hold_force: Quantity
    target_force_rating: Quantity

    @property
    def donor_opening(self) -> Quantity:
        return (self.command * self.donor_stroke).to("mm")

    @property
    def target_opening(self) -> Quantity:
        return (self.command * self.target_stroke).to("mm")

    @property
    def stroke_excess(self) -> Quantity:
        """How far the donor's opening exceeds the target's whole stroke."""
        return (self.donor_opening - self.target_stroke).to("mm")

    @property
    def force_ratio(self) -> float:
        return (self.donor_hold_force / self.target_force_rating).to("").magnitude

    @property
    def donor_clears(self) -> bool:
        return self.donor_opening > self.part_width

    @property
    def target_clears(self) -> bool:
        return self.target_opening > self.part_width
    @property
    def result(self):
        from .solvers import ChunkedPrefillModel
        p = self.prefill.result
        return ChunkedPrefillModel().solve(self.prefill.workload.parameters, p.sustained_rate,
                                           p.weight_read_time, self.decodes, self.prompt_tokens,
                                           self.chunk_tokens)


class PausedKvStateScenario(BaseScenario):
    """A request paused on a tool: keep, move over the host links, or drop and recompute its KV state."""
    pool: KvAdmissionScenario
    prefill: PrefixRebuildScenario   # node, links, and sustained prefill rate
    context_tokens: int
    block_tokens: int
    wait: Quantity
    link_fraction: float             # share of rated link bandwidth under contention
    returns: int                     # paused requests whose waits end together

    @property
    def link(self) -> Quantity:
        node = self.prefill.node
        return node.accelerators_per_node * node.accelerator.interconnect.bandwidth

    @property
    def result(self):
        from .solvers import PausedKvStateModel
        return PausedKvStateModel().solve(self.pool.result.kv_per_token,
                                          self.prefill.workload.parameters,
                                          self.prefill.result.sustained_rate, self.link,
                                          self.context_tokens, self.block_tokens, self.wait,
                                          self.link_fraction, self.returns)


class KvBlockSharingScenario(BaseScenario):
    """Requests with an output allowance in whole blocks, with and without a shared prefix, as a turn rate."""
    pool: KvAdmissionScenario
    headroom: float                  # fraction of the pool withheld
    context: int                     # input tokens per request
    output: int                      # output allowance per request
    common: int                      # shareable prefix tokens
    block_tokens: int
    generation: Quantity             # residence while generating
    wait: Quantity                   # tool wait with the state kept

    @property
    def result(self):
        from .solvers import KvBlockSharingModel
        p = self.pool.result
        return KvBlockSharingModel().solve(p.kv_per_token, p.kv_budget, self.headroom, self.context,
                                           self.output, self.common, self.block_tokens,
                                           self.generation, self.wait)


class KvResidencyArchetypeScenario(BaseScenario):
    """One agent workload's sharing gain, residency tax, and reload against recompute."""
    paused: PausedKvStateScenario    # pool, link, and sustained prefill rate
    context_tokens: int
    shared_prefix: int
    generation: Quantity
    wait: Quantity

    @property
    def result(self):
        from .solvers import KvResidencyArchetypeModel
        p = self.paused
        pool = p.pool.result
        return KvResidencyArchetypeModel().solve(pool.kv_per_token, pool.kv_budget,
                                                 p.prefill.workload.parameters,
                                                 p.prefill.result.sustained_rate, p.link,
                                                 self.context_tokens, self.shared_prefix,
                                                 self.generation, self.wait)
# --- The real-time permission path of an embodied machine (added 2026-10-20) ---
# Each scenario names the loop, task, bus, or lease of one machine at one site
# (``Embodied.Scenario.*``) and states the inputs the site record does not
# carry; the solvers live in ``engine/solvers/realtime.py``.

class PhaseLagBudgetScenario(BaseScenario):
    """A loop with crossover ``crossover`` sampled every ``sample_period``, with compute and transport delays."""
    crossover: Quantity
    sample_period: Quantity
    compute_delay: Optional[Quantity] = None
    transport_delay: Optional[Quantity] = None
    phase_allowance: Optional[Quantity] = None   # phase erosion the design allows at crossover

    @property
    def result(self):
        from .solvers import PhaseLagBudgetModel
        return PhaseLagBudgetModel().solve(self.crossover, self.sample_period, self.compute_delay,
                                           self.transport_delay, self.phase_allowance)


class ResponseTimeScenario(BaseScenario):
    """``site``'s permission check as a periodic task: R = C + B + I against its deadline.

    Each term defaults to the site record's; a stated value replaces it.
    """
    site: Any                                    # EmbodiedSiteScenario
    wcet: Optional[Quantity] = None
    blocking: Optional[Quantity] = None
    interference: Optional[Quantity] = None
    deadline: Optional[Quantity] = None

    @property
    def result(self):
        from .solvers import ResponseTimeModel
        s = self.site
        return ResponseTimeModel().solve(
            s.enforcer_wcet if self.wcet is None else self.wcet,
            s.enforcer_blocking if self.blocking is None else self.blocking,
            s.enforcer_interference if self.interference is None else self.interference,
            s.enforcer_deadline if self.deadline is None else self.deadline,
            unloaded=s.enforcer_unloaded)


class CanFdBusScenario(BaseScenario):
    """``num_axes`` servo axes on one CAN FD bus, ``frames_per_axis`` frames each per ``cycle``."""
    num_axes: int
    payload_bytes: int
    frames_per_axis: int
    baud_arbitration: Quantity
    baud_data: Quantity
    cycle: Quantity

    @property
    def result(self):
        from .solvers import CanFdBusModel
        return CanFdBusModel().solve(self.num_axes, self.payload_bytes, self.frames_per_axis,
                                     self.baud_arbitration, self.baud_data, self.cycle)


class EtherCatCycleScenario(BaseScenario):
    """``num_axes`` drives on one EtherCAT chain, ``bytes_per_axis`` of process data each per ``cycle``."""
    num_axes: int
    bytes_per_axis: int
    baud_rate: Quantity
    header_bytes: int
    hop_delay: Quantity
    cycle: Quantity

    @property
    def result(self):
        from .solvers import EtherCatCycleModel
        return EtherCatCycleModel().solve(self.num_axes, self.bytes_per_axis, self.baud_rate,
                                          self.header_bytes, self.hop_delay, self.cycle)


class LeaseBudgetScenario(BaseScenario):
    """``site.machine``'s chunk lease at ``speed``: its stopping-budget cost and the lease rule's bounds.

    The lease defaults to the site's; the renewal period and the chunk
    inference latency are the machine's; ``renewal_jitter`` is the allowance
    for variation in the interval between successful renewals.
    """
    site: Any                                    # EmbodiedSiteScenario
    speed: Quantity
    renewal_jitter: Quantity
    lease: Optional[Quantity] = None

    @property
    def result(self):
        from .solvers import LeaseBudgetModel
        s, rm = self.site, self.site.machine
        return LeaseBudgetModel().solve(
            self.speed, s.a_brake, s.d_clear, s.delta_loc, s.delta_margin, s.t_brake_onset,
            s.t_lease if self.lease is None else self.lease, s.t_tick, s.t_bus,
            s.perception_age, s.t_capture_to_admission,
            rm.chunk_period, self.renewal_jitter, rm.chunk_inference_latency)


# --- Tool calling (added 2026-10-20) ------------------------------------------

class BlindRetryScenario(BaseScenario):
    """Mutating calls retried once after any timeout, with independent request and reply loss."""
    request_loss: float              # probability a request is lost
    reply_loss: float                # probability a reply is lost
    workers: int
    calls_per_worker: int            # mutating calls per worker per day

    @property
    def calls(self) -> int:
        return self.workers * self.calls_per_worker

    @property
    def result(self):
        from .solvers import BlindRetryModel
        return BlindRetryModel().solve(self.request_loss, self.reply_loss, self.calls)


class ObservationCarryScenario(BaseScenario):
    """A raw tool output sized in windows, and its shaped view carried by later calls."""
    raw_bytes: Quantity
    bytes_per_token: Quantity        # assumed serialization ratio, byte/token
    window: int                      # context window, tokens
    shaped_tokens: int               # staged observation, tokens
    carry_calls: int                 # later calls that carry it
    input_price: Quantity            # USD per uncached input token
    cached_fraction: float           # cached input billed at this share of the price

    @property
    def result(self):
        from .solvers import ObservationCarryModel
        return ObservationCarryModel().solve(self.raw_bytes, self.bytes_per_token, self.window,
                                             self.shaped_tokens, self.carry_calls, self.input_price,
                                             self.cached_fraction)


class ModelPollingScenario(BaseScenario):
    """A model that checks a pending job at a fixed interval until it completes."""
    job_duration: Quantity
    poll_interval: Quantity
    context_tokens: int              # selected input per status call
    output_tokens: int               # generated tokens per status call
    cache_hit: float                 # share of input billed as cached
    input_price: Quantity            # USD per uncached input token
    cached_fraction: float           # cached input billed at this share of the price
    output_price: Quantity           # USD per output token

    @property
    def result(self):
        from .solvers import ModelPollingModel
        return ModelPollingModel().solve(self.job_duration, self.poll_interval, self.context_tokens,
                                         self.output_tokens, self.cache_hit, self.input_price,
                                         self.cached_fraction, self.output_price)


class ToolCatalogSelectionScenario(BaseScenario):
    """A registered tool catalog staged whole or as a selected subset with overhead."""
    registered: int                  # tools registered
    selected: int                    # tools staged by selection
    descriptor_tokens: int           # per descriptor, final serialization
    selection_overhead: int          # selection interface and metadata, tokens

    @property
    def result(self):
        from .solvers import ToolCatalogSelectionModel
        return ToolCatalogSelectionModel().solve(self.registered, self.selected, self.descriptor_tokens,
                                                 self.selection_overhead)


class ToolkitGranularityScenario(BaseScenario):
    """Structured tools against one shell tool, judged by cost per accepted task."""
    tools: int                       # structured tools
    tool_tokens: int                 # per structured descriptor
    shell_tokens: int                # the one shell descriptor
    calls: int                       # model calls per trajectory
    input_price: Quantity            # USD per uncached input token
    structured_attempt: Quantity     # USD per structured attempt
    structured_acceptance: float
    structured_change: float         # probability of an unconfined change
    shell_acceptance: float
    shell_change: float
    change_cost: Quantity            # USD to detect and repair one unconfined change

    @property
    def result(self):
        from .solvers import ToolkitGranularityModel
        return ToolkitGranularityModel().solve(self.tools, self.tool_tokens, self.shell_tokens, self.calls,
                                               self.input_price, self.structured_attempt,
                                               self.structured_acceptance, self.structured_change,
                                               self.shell_acceptance, self.shell_change, self.change_cost)


class IntegrationAdaptersScenario(BaseScenario):
    """Hosts connected to tool services, pairwise or through one shared protocol."""
    hosts: int
    services: int

    @property
    def result(self):
        from .solvers import IntegrationAdaptersModel
        return IntegrationAdaptersModel().solve(self.hosts, self.services)


# --- Agent sandboxes (added 2026-10-20) ---
class SyscallTaxScenario(BaseScenario):
    """``calls`` ordinary system calls priced by the terms each isolation envelope pays."""
    calls: int
    trap: Quantity                   # trap into a kernel
    filter: Quantity                 # in-kernel filter (shared-kernel container)
    emulate: Quantity                # user-space kernel emulation
    switch: Quantity                 # switch to the emulator

    @property
    def result(self):
        from .solvers import SyscallTaxModel
        return SyscallTaxModel().solve(self.calls, self.trap, self.filter, self.emulate, self.switch)


class EnvelopeAmortizationScenario(BaseScenario):
    """``calls`` sequential calls of ``work_per_call``, with envelope setup paid per call or once."""
    calls: int
    setup: Quantity                  # envelope preparation through readiness for useful work
    work_per_call: Quantity
    overhead_cap: float              # largest acceptable setup share of a fresh call

    @property
    def result(self):
        from .solvers import EnvelopeAmortizationModel
        return EnvelopeAmortizationModel().solve(self.calls, self.setup, self.work_per_call,
                                                 self.overhead_cap)


class WorkspaceProvisioningScenario(BaseScenario):
    """``workspaces`` on one pinned base, provisioned by copy or copy-on-write, then a first write."""
    workspaces: int
    base_image: Quantity             # repository, toolchain, and dependencies
    written: Quantity                # edits and build outputs per workspace
    bandwidth: Quantity              # one drive's sequential rate
    fixture: Quantity                # a large shared file each task modifies
    block: Quantity                  # block-granular copying unit
    modified: Quantity               # bytes each task changes in the fixture

    @property
    def result(self):
        from .solvers import WorkspaceProvisioningModel
        return WorkspaceProvisioningModel().solve(self.workspaces, self.base_image, self.written,
                                                  self.bandwidth, self.fixture, self.block,
                                                  self.modified)


class SandboxPoolReserveScenario(BaseScenario):
    """A warm sandbox pool under steady arrivals: active leases, a standby reserve, and memory."""
    arrival_rate: Quantity
    hold: Quantity                   # lease held for the whole trajectory
    restore: Quantity                # one clean instance from its snapshot
    guest_memory: Quantity           # full private guest image
    dirty: Quantity                  # private pages written per active lease
    standby: Quantity                # private state per standby instance
    target_miss: float               # request-level miss target
    initial_reserve: int = 1         # candidate evaluated before the minimum
    slow_factor: float = 10          # a cold boot this many times slower than restore

    @property
    def result(self):
        from .solvers import SandboxPoolReserveModel
        return SandboxPoolReserveModel().solve(self.arrival_rate, self.hold, self.restore,
                                               self.guest_memory, self.dirty, self.standby,
                                               self.target_miss, self.initial_reserve,
                                               self.slow_factor)


class BootTimeCdfScenario(BaseScenario):
    """Median and tail boot time of ``booting`` microVMs started at once, from a released CDF series."""
    series: Any                      # a Literature series with concurrency, boot_ms, booted_fraction
    booting: int
    median_fraction: float = 0.5
    tail_fraction: float = 0.99

    @property
    def result(self):
        from .solvers import BootTimeCdfModel
        rows = [r for r in self.series.entries if r["concurrency"] == self.booting]
        times = [Q_(r["boot_ms"], "ms") for r in rows]
        return BootTimeCdfModel().solve(self.booting, times, [r["booted_fraction"] for r in rows],
                                        self.median_fraction, self.tail_fraction)


# --- On-device learning and federated rounds (added 2026-10-20) --------------
# One full update on a phone (working set and DRAM traffic), parameter-efficient
# updates, personalization storage, uplink volume, the energy of a training
# second against inference checks, and the round mechanics a federated fleet
# exposes (deadlines, coverage, masking, shadow gating, spread of devices).

class UpdateWorkingSetScenario(BaseScenario):
    """A full on-device update with Adam: its working set and the DRAM bytes of one step.

    Gradients equal the weights; the optimizer keeps ``optimizer_moments``
    weight-sized arrays; saved activations are ``activation_factor`` times the
    weight bytes (``activation_factor_high`` bounds the quoted range).
    """
    parameters: float
    bytes_per_param: Quantity
    optimizer_moments: int
    activation_factor: float
    activation_factor_high: float
    bandwidth: Optional[Quantity] = None      # device DRAM bandwidth, for the traffic floor

    @property
    def weights(self) -> Quantity:
        return (self.parameters * self.bytes_per_param).to("MB")

    @property
    def gradients(self) -> Quantity:
        return self.weights

    @property
    def optimizer_state(self) -> Quantity:
        return (self.optimizer_moments * self.weights).to("MB")

    @property
    def activations(self) -> Quantity:
        return (self.activation_factor * self.weights).to("MB")

    @property
    def persistent(self) -> Quantity:
        """Weights, gradients, and optimizer state: the activation-free floor."""
        return (self.weights + self.gradients + self.optimizer_state).to("MB")

    @property
    def total(self) -> Quantity:
        return (self.persistent + self.activations).to("MB")

    @property
    def amplification(self) -> float:
        return (self.total / self.weights).to("").magnitude

    @property
    def floor(self) -> float:
        return (self.persistent / self.weights).to("").magnitude

    @property
    def ceiling(self) -> float:
        return self.floor + self.activation_factor_high

    @property
    def step_traffic(self) -> Quantity:
        from ..physics import calc_update_step_traffic
        traffic, _ = calc_update_step_traffic(self.weights, self.gradients, self.optimizer_state, self.activations)
        return traffic.to("MB")

    @property
    def weight_reads(self) -> Quantity:
        from ..physics import calc_update_step_traffic
        _, reads = calc_update_step_traffic(self.weights, self.gradients, self.optimizer_state, self.activations)
        return reads.to("MB")

    @property
    def traffic_ratio(self) -> float:
        return (self.step_traffic / self.weights).to("").magnitude

    @property
    def weight_read_share(self) -> float:
        return (self.weight_reads / self.step_traffic).to("").magnitude

    @property
    def step_floor(self) -> Quantity:
        """Time to move one step's bytes at ``bandwidth`` (traffic alone)."""
        return (self.step_traffic / self.bandwidth).to("millisecond")


class LowRankUpdateScenario(BaseScenario):
    """A rank-``rank`` update to one ``out_dim`` by ``in_dim`` weight matrix."""
    in_dim: int
    out_dim: int
    rank: int

    @property
    def full_params(self) -> int:
        return self.in_dim * self.out_dim

    @property
    def update_params(self) -> int:
        from ..physics import calc_low_rank_update_params
        return calc_low_rank_update_params(self.in_dim, self.out_dim, self.rank)

    @property
    def reduction(self) -> float:
        """Share of the full layer's trainable parameters the update removes."""
        return 1 - self.update_params / self.full_params


class ResidualAdapterScenario(BaseScenario):
    """Residual bottleneck adapters of ``width`` to ``bottleneck``, one at each of ``sites``."""
    width: int
    bottleneck: int
    sites: int
    bytes_per_param: Quantity

    @property
    def params_per_adapter(self) -> int:
        from ..physics import calc_residual_adapter_params
        return calc_residual_adapter_params(self.width, self.bottleneck)

    @property
    def params_total(self) -> int:
        return self.params_per_adapter * self.sites

    @property
    def adapter_bytes(self) -> Quantity:
        return (self.params_per_adapter * self.bytes_per_param).to("KB")

    @property
    def total_bytes(self) -> Quantity:
        return (self.params_total * self.bytes_per_param).to("MB")


class PersonalizationStorageScenario(BaseScenario):
    """``contexts`` personalized models stored as full copies or as one backbone plus adapters."""
    model_params: Quantity        # in param
    adapter_params: Quantity      # in param, per context
    bytes_per_param: Quantity     # bytes per param
    contexts: int

    @property
    def model_size(self) -> Quantity:
        from ..physics import memory_from_params
        return memory_from_params(self.model_params, self.bytes_per_param)

    @property
    def adapter_size(self) -> Quantity:
        from ..physics import memory_from_params
        return memory_from_params(self.adapter_params, self.bytes_per_param)

    @property
    def total_copies(self) -> Quantity:
        return self.contexts * self.model_size

    @property
    def total_shared(self) -> Quantity:
        return self.model_size + self.contexts * self.adapter_size

    @property
    def total_savings(self) -> float:
        return (self.total_copies / self.total_shared).to_base_units().magnitude

    @property
    def per_context_savings(self) -> float:
        return (self.model_size / self.adapter_size).to_base_units().magnitude


class FederatedUplinkScenario(BaseScenario):
    """A week of raw captures against compressed model updates sent at named round cadences."""
    items_per_week: int
    item_size: Quantity
    update_params: float
    update_bits_per_param: int
    rounds_per_week: Dict[str, int]       # cadence name -> rounds a week

    @property
    def raw_volume(self) -> Quantity:
        return self.items_per_week * self.item_size

    @property
    def update_size(self) -> Quantity:
        from ..core.units import BITS_PER_BYTE
        return (self.update_params * self.update_bits_per_param / BITS_PER_BYTE * Q_(1, "byte")).to("MB")

    @property
    def reduction(self) -> float:
        """Raw week over one update (also the break-even round count)."""
        return (self.raw_volume / self.update_size).to_base_units().magnitude

    @property
    def break_even_rounds(self) -> float:
        return self.reduction

    def weekly_upload(self, cadence: str) -> Quantity:
        return (self.rounds_per_week[cadence] * self.update_size).to("MB")


class TrainingSecondVersusInferenceScenario(BaseScenario):
    """One second of training inside a sustained power range, priced in inference checks."""
    training_power_low: Quantity
    training_power_high: Quantity
    inference_energy_low: Quantity
    inference_energy_high: Quantity
    check_rate: Quantity                  # always-on detector checks per second
    training_time: Quantity = Q_(1, "second")

    @property
    def training_energy_low(self) -> Quantity:
        return (self.training_power_low * self.training_time).to("joule")

    @property
    def training_energy_high(self) -> Quantity:
        return (self.training_power_high * self.training_time).to("joule")

    @property
    def checks_low(self) -> float:
        """Fewest checks one training second buys: low training energy over the costliest check."""
        return (self.training_energy_low / self.inference_energy_high).to("").magnitude

    @property
    def checks_high(self) -> float:
        return (self.training_energy_high / self.inference_energy_low).to("").magnitude

    @property
    def detector_power(self) -> Quantity:
        """Power of a detector at ``check_rate`` using the cheapest check."""
        return (self.check_rate * self.inference_energy_low).to("milliwatt")

    @property
    def time_to_training_second_low(self) -> Quantity:
        return (self.training_energy_low / self.detector_power).to("minute")

    @property
    def time_to_training_second_high(self) -> Quantity:
        return (self.training_energy_high / self.detector_power).to("minute")


class DeadlineVersusRoundTripScenario(BaseScenario):
    """Foreground deadlines against a network round-trip range: which must run locally."""
    deadlines: Dict[str, Quantity]
    round_trip_low: Quantity
    round_trip_high: Quantity

    def local_only(self, name: str) -> bool:
        """True when the deadline is shorter than even the fastest round trip."""
        return self.deadlines[name] < self.round_trip_low


class ParticipationCoverageScenario(BaseScenario):
    """Devices per cohort at each stage of a federated round, and each cohort's coverage ratio.

    Coverage ratio = cohort share of completed updates / cohort share of deployed devices.
    """
    stages: List[str]                        # in order, first is the deployed fleet, last completed
    counts: Dict[str, List[int]]             # cohort -> count at each stage

    def total(self, stage: str) -> int:
        i = self.stages.index(stage)
        return sum(c[i] for c in self.counts.values())

    def share(self, cohort: str, stage: str) -> float:
        i = self.stages.index(stage)
        return self.counts[cohort][i] / self.total(stage)

    def coverage_ratio(self, cohort: str) -> float:
        return self.share(cohort, self.stages[-1]) / self.share(cohort, self.stages[0])

    def stage_ratio(self, numerator: str, denominator: str) -> float:
        return self.total(numerator) / self.total(denominator)


class PairwiseMaskScenario(BaseScenario):
    """Two clients hide their updates with a shared mask that cancels in the server's sum."""
    update_a: List[float]
    update_b: List[float]
    mask: List[float]

    @property
    def upload_a(self) -> List[float]:
        return [u + m for u, m in zip(self.update_a, self.mask)]

    @property
    def upload_b(self) -> List[float]:
        return [u - m for u, m in zip(self.update_b, self.mask)]

    @property
    def total(self) -> List[float]:
        """The server's sum of the two uploads: the masks cancel."""
        return [a + b for a, b in zip(self.upload_a, self.upload_b)]


class ShadowDisagreementScenario(BaseScenario):
    """A frozen and an adapted model on the same inputs: label-free disagreement against a tolerance."""
    frozen: str                               # one output label per input
    adapted: str
    tolerance: float

    @property
    def disagreements(self) -> int:
        return sum(a != b for a, b in zip(self.frozen, self.adapted))

    @property
    def inputs(self) -> int:
        return len(self.frozen)

    @property
    def disagreement_rate(self) -> float:
        return self.disagreements / self.inputs

    @property
    def hold(self) -> bool:
        return self.disagreement_rate > self.tolerance


class SeriesSpreadScenario(BaseScenario):
    """Highest over lowest ``value_key`` among rows of a Literature series in named year windows."""
    series_name: str
    value_key: str
    windows: Dict[str, tuple]                 # window name -> (first year, last year), inclusive

    @property
    def rows(self) -> tuple:
        from ..literature.registry import Literature
        return getattr(Literature.Series, self.series_name).entries

    def window(self, name: str) -> List[dict]:
        lo, hi = self.windows[name]
        return [r for r in self.rows if r[self.value_key] and lo <= r["year"] <= hi]

    def highest(self, name: str) -> dict:
        return max(self.window(name), key=lambda r: r[self.value_key])

    def lowest(self, name: str) -> dict:
        return min(self.window(name), key=lambda r: r[self.value_key])

    def spread(self, name: str) -> float:
        return self.highest(name)[self.value_key] / self.lowest(name)[self.value_key]


# --- Policy training: discounting, decoders, trial campaigns, contact exploits (added 2026-10-20) ---

class DiscountHorizonScenario(BaseScenario):
    """One ``rollout`` of decisions ``step`` apart under discount ``discount``: how far the return
    reaches, in decisions and in seconds, and the discount that keeps ``retained_weight`` of the
    final reward at the first decision."""
    step: Quantity
    rollout: Quantity
    discount: float
    retained_weight: float = 0.5

    @property
    def decisions(self) -> int:
        return int(round((self.rollout / self.step).to("").magnitude))

    @property
    def terminal_weight(self) -> float:
        from ..physics.robotics.learning import calc_discounted_weight
        return calc_discounted_weight(self.discount, self.decisions)

    @property
    def horizon_steps(self) -> float:
        from ..physics.robotics.learning import calc_discount_effective_horizon
        return calc_discount_effective_horizon(self.discount)

    @property
    def horizon_time(self) -> Quantity:
        return (self.horizon_steps * self.step).to("second")

    @property
    def retaining_discount(self) -> float:
        from ..physics.robotics.learning import calc_discount_for_retained_weight
        return calc_discount_for_retained_weight(self.decisions, self.retained_weight)


class IterativeDecoderBudgetScenario(BaseScenario):
    """``machine``'s one-pass chunk decoder against an iterative head that encodes once and runs
    ``denoising_steps`` dependent evaluations in ``denoising_latency``, and a ``flow_steps`` head at
    the same cost per evaluation, on the setpoint step and the renewal period with
    ``renewal_jitter`` held free."""
    machine: Any                   # EmbodiedMobileManipulator (chunk step, period, p99 latency)
    denoising_steps: int
    flow_steps: int
    encode: Quantity
    denoising_latency: Quantity
    renewal_jitter: Quantity

    @property
    def setpoint_step(self) -> Quantity:
        return self.machine.chunk_step.to("ms")

    @property
    def renewal_period(self) -> Quantity:
        return self.machine.chunk_period.to("ms")

    @property
    def one_pass_latency(self) -> Quantity:
        return self.machine.chunk_inference_latency.to("ms")

    @property
    def per_evaluation(self) -> Quantity:
        return ((self.denoising_latency - self.encode) / self.denoising_steps).to("ms")

    @property
    def flow_latency(self) -> Quantity:
        from ..physics.robotics.learning import calc_iterative_decoder_latency
        return calc_iterative_decoder_latency(self.encode, self.per_evaluation, self.flow_steps)

    @property
    def one_pass_slack(self) -> Quantity:
        return (self.renewal_period - self.one_pass_latency).to("ms")

    @property
    def denoising_slack(self) -> Quantity:
        return (self.renewal_period - self.denoising_latency).to("ms")

    @property
    def one_pass_end_under_jitter(self) -> Quantity:
        return (self.one_pass_latency + self.renewal_jitter).to("ms")

    @property
    def denoising_overrun_under_jitter(self) -> Quantity:
        """How far a jitter-sized stretch carries the iterative head past its renewal."""
        return (self.denoising_latency + self.renewal_jitter - self.renewal_period).to("ms")

    @property
    def renewal_less_jitter(self) -> Quantity:
        return (self.renewal_period - self.renewal_jitter).to("ms")

    def evaluation_room(self, budget: Quantity) -> float:
        """Evaluations that fit ``budget`` after encoding, unrounded."""
        return ((budget - self.encode) / self.per_evaluation).to("").magnitude

    def max_evaluations(self, budget: Quantity) -> int:
        from ..physics.robotics.learning import calc_max_decoder_evaluations
        return calc_max_decoder_evaluations(budget, self.encode, self.per_evaluation)

    def latency_at(self, evaluations: int) -> Quantity:
        from ..physics.robotics.learning import calc_iterative_decoder_latency
        return calc_iterative_decoder_latency(self.encode, self.per_evaluation, evaluations)

    @property
    def evaluations_within_step(self) -> int:
        return self.max_evaluations(self.setpoint_step)

    @property
    def evaluations_within_renewal(self) -> int:
        """Evaluations that fit the renewal period with the jitter allowance kept free."""
        return self.max_evaluations(self.renewal_less_jitter)


class ParallelSimRunScenario(BaseScenario):
    """A GPU-parallel on-policy run: ``robots`` environments each stepping ``steps_per_update`` times
    per update for ``updates`` updates within ``training_time``, at ``policy_rate``."""
    robots: int
    steps_per_update: int
    updates: int
    training_time: Quantity        # an upper bound ("under 20 minutes")
    policy_rate: Quantity

    @property
    def batch(self) -> int:
        return self.robots * self.steps_per_update

    @property
    def samples(self) -> int:
        return self.batch * self.updates

    @property
    def throughput(self) -> Quantity:
        """Policy steps per second, a lower bound because the time is an upper bound."""
        return (self.samples / self.training_time).to("1/s")

    @property
    def simulated_experience(self) -> Quantity:
        return (self.samples / self.policy_rate).to("day")

    def time_for(self, steps: float) -> Quantity:
        """Wall-clock time for ``steps`` policy steps at this run's throughput."""
        return (steps / self.throughput).to("minute")


class TrialCampaignScenario(BaseScenario):
    """``trials`` hardware trials of ``rollout`` each, decisions ``decision_step`` apart, with automated
    and manual resets, on one machine and on a farm; what a demonstration-seeded ``seeded_budget``
    allows; and the same trials in ``simulator`` at its published throughput."""
    trials: int
    rollout: Quantity
    decision_step: Quantity
    automated_reset: Quantity
    manual_reset: Quantity
    farm_rigs: int
    reversals_per_trial: int
    seeded_budget: Quantity
    simulator: Optional[ParallelSimRunScenario] = None

    def _wall(self, reset, rigs=1) -> Quantity:
        from ..physics.robotics.learning import calc_trial_campaign_wall_time
        return calc_trial_campaign_wall_time(self.trials, self.rollout, reset, rigs)

    @property
    def execution_time(self) -> Quantity:
        return self._wall(Q_(0, "s"))

    @property
    def automated_time(self) -> Quantity:
        return self._wall(self.automated_reset)

    @property
    def manual_time(self) -> Quantity:
        return self._wall(self.manual_reset)

    @property
    def farm_manual_time(self) -> Quantity:
        return self._wall(self.manual_reset, self.farm_rigs).to("day")

    @property
    def manual_reset_share(self) -> float:
        return (self.manual_reset / (self.rollout + self.manual_reset)).to("").magnitude

    @property
    def reversals(self) -> int:
        return self.trials * self.reversals_per_trial

    @property
    def seeded_trials(self) -> int:
        """Trials with automated resets that fit the seeded budget."""
        return int(_math.floor((self.seeded_budget / (self.rollout + self.automated_reset)).to("").magnitude))

    @property
    def seeded_reversals(self) -> int:
        return self.seeded_trials * self.reversals_per_trial

    @property
    def trial_ratio(self) -> float:
        return self.trials / self.seeded_trials

    @property
    def decisions_per_trial(self) -> int:
        return int(round((self.rollout / self.decision_step).to("").magnitude))

    @property
    def simulated_steps(self) -> int:
        return self.trials * self.decisions_per_trial

    @property
    def simulated_time(self) -> Quantity:
        return self.simulator.time_for(self.simulated_steps)

    @property
    def simulation_speedup(self) -> float:
        return (self.automated_time / self.simulated_time).to("").magnitude


class ContactDeadlineScenario(BaseScenario):
    """A constant-speed drive into ``site``'s latch plate: when the force ramp k v t reaches the
    tripwire and the limit, when braking can begin, and the time left, at both approaches."""
    site: Any                      # EmbodiedScenario (latch stiffness, limit, tripwire, contact response)

    def deadline(self, speed: Quantity) -> Quantity:
        from ..physics.robotics.actuation import calc_contact_yield_deadline
        s = self.site
        return calc_contact_yield_deadline(s.f_latch_limit, s.k_latch, speed).to("ms")

    def tripwire_time(self, speed: Quantity) -> Quantity:
        return (self.site.f_latch_tripwire / (self.site.k_latch * speed)).to("ms")

    def brake_onset(self, speed: Quantity) -> Quantity:
        return (self.tripwire_time(speed) + self.site.t_contact_response).to("ms")

    def time_left(self, speed: Quantity) -> Quantity:
        return (self.deadline(speed) - self.brake_onset(speed)).to("ms")

    @property
    def guarded_speed(self) -> Quantity:
        return self.site.v_latch_approach

    @property
    def fast_speed(self) -> Quantity:
        return self.site.v_latch_high

    @property
    def deadline_guarded(self) -> Quantity:
        return self.deadline(self.guarded_speed)

    @property
    def deadline_fast(self) -> Quantity:
        return self.deadline(self.fast_speed)

    @property
    def tripwire_guarded(self) -> Quantity:
        return self.tripwire_time(self.guarded_speed)

    @property
    def tripwire_fast(self) -> Quantity:
        return self.tripwire_time(self.fast_speed)

    @property
    def brake_onset_guarded(self) -> Quantity:
        return self.brake_onset(self.guarded_speed)

    @property
    def brake_onset_fast(self) -> Quantity:
        return self.brake_onset(self.fast_speed)

    @property
    def time_left_guarded(self) -> Quantity:
        return self.time_left(self.guarded_speed)

    @property
    def time_left_fast(self) -> Quantity:
        return self.time_left(self.fast_speed)

    @property
    def fast_margin_ratio(self) -> float:
        return (self.deadline_fast / self.brake_onset_fast).to("").magnitude


class SoftContactExploitScenario(BaseScenario):
    """``machine``'s arm coasting into ``site``'s latch plate at the fast approach: the peak force a
    penalty spring that lets the gripper sink ``penalty_penetration`` reports, the peak the real
    plate imposes on a rigid arm, and the arm stiffness at which the series chain reaches the limit.
    The moving mass is the arm's effective contact mass without its payload capacity (the gripper
    holds no object at the door)."""
    machine: Any                   # EmbodiedMobileManipulator (arm contact mass, payload capacity)
    site: Any                      # EmbodiedScenario (latch stiffness, limit, approach speeds)
    penalty_penetration: Quantity

    @property
    def moving_mass(self) -> Quantity:
        return (self.machine.arm_effective_contact_mass - self.machine.arm.payload_capacity).to("kg")

    @property
    def penalty_stiffness(self) -> Quantity:
        from ..physics.robotics.learning import calc_penalty_stiffness_from_penetration
        return calc_penalty_stiffness_from_penetration(self.moving_mass, self.site.v_latch_high,
                                                       self.penalty_penetration)

    def peak_force(self, stiffness: Quantity, speed: Optional[Quantity] = None) -> Quantity:
        from ..physics.robotics.actuation import calc_transient_impact_force
        v = self.site.v_latch_high if speed is None else speed
        return calc_transient_impact_force(v, self.moving_mass, stiffness).to("N")

    def chain_peak_force(self, arm_stiffness: Quantity) -> Quantity:
        """Fast-approach peak with the arm in series with the plate."""
        from ..physics.robotics.learning import calc_series_stiffness
        return self.peak_force(calc_series_stiffness(self.site.k_latch, arm_stiffness))

    @property
    def simulated_peak(self) -> Quantity:
        return self.peak_force(self.penalty_stiffness)

    @property
    def rigid_peak(self) -> Quantity:
        return self.peak_force(self.site.k_latch)

    @property
    def guarded_rigid_peak(self) -> Quantity:
        return self.peak_force(self.site.k_latch, self.site.v_latch_approach)

    @property
    def break_even_series_stiffness(self) -> Quantity:
        from ..physics.robotics.learning import calc_impact_stiffness_for_force
        return calc_impact_stiffness_for_force(self.site.f_latch_limit, self.site.v_latch_high,
                                               self.moving_mass)

    @property
    def break_even_arm_stiffness(self) -> Quantity:
        from ..physics.robotics.learning import calc_complementary_series_stiffness
        return calc_complementary_series_stiffness(self.break_even_series_stiffness, self.site.k_latch)

    @property
    def plate_over_penalty(self) -> float:
        return (self.site.k_latch / self.penalty_stiffness).to("").magnitude

    @property
    def rigid_over_limit(self) -> float:
        return (self.rigid_peak / self.site.f_latch_limit).to("").magnitude

    @property
    def penalty_timescale(self) -> Quantity:
        from ..physics.robotics.learning import calc_contact_timescale
        return calc_contact_timescale(self.moving_mass, self.penalty_stiffness)

    @property
    def plate_timescale(self) -> Quantity:
        from ..physics.robotics.learning import calc_contact_timescale
        return calc_contact_timescale(self.moving_mass, self.site.k_latch)

    @property
    def timescale_ratio(self) -> float:
        return (self.penalty_timescale / self.plate_timescale).to("").magnitude


class ForceRiseGapScenario(BaseScenario):
    """An illustrative linear drive regulating ``target_force`` against a surface, in simulation and on
    hardware, with one stated discrepancy in each reality-gap channel: dynamics (force-rise rate and
    drive delay), timing (p99 actuation delay against the simulator's), sensing (range bias and
    dropout), and appearance (pixel difference hiding a contact-patch shift)."""
    approach_speed: Quantity
    target_force: Quantity
    simulated_rise_rate: Quantity
    physical_rise_rate: Quantity
    p99_delay: Quantity
    simulated_delay: Quantity
    simulator_step: Quantity       # integration period, not pipeline latency
    drive_delay: Quantity
    overtravel: Quantity
    range_bias: Quantity
    dropout_frames: int
    pixel_difference: float        # fraction of the image
    patch_shift: Quantity
    chamfer_tolerance: Quantity

    @property
    def simulated_time_to_target(self) -> Quantity:
        return (self.target_force / self.simulated_rise_rate).to("ms")

    @property
    def physical_time_to_target(self) -> Quantity:
        return (self.target_force / self.physical_rise_rate).to("ms")

    @property
    def unmodeled_travel(self) -> Quantity:
        """Travel before correction that the simulator's delay does not predict."""
        return (self.approach_speed * (self.p99_delay - self.simulated_delay)).to("mm")


class OnRobotFineTuneScenario(BaseScenario):
    """``interaction_steps`` setpoints ``step`` apart collected on the machine to absorb a
    ``plate_offset``, of which ``exceed_fraction`` cross the contact tripwire."""
    step: Quantity
    interaction_steps: int
    exceed_fraction: float
    plate_offset: Quantity

    @property
    def setpoint_rate(self) -> Quantity:
        return (1 / self.step).to("Hz")

    @property
    def active_time(self) -> Quantity:
        return (self.interaction_steps * self.step).to("second")

    @property
    def exceed_samples(self) -> int:
        return int(round(self.interaction_steps * self.exceed_fraction))


class PhysicalTrialAblationScenario(BaseScenario):
    """Per-trial physical results by training condition (a Literature series with ``table``,
    ``condition``, ``rotations``, and ``printed_median`` columns): the trials, their median, and
    the table's own median."""
    series: Any

    def trials(self, condition: str, table: Optional[str] = None) -> List[float]:
        rows = [e for e in self.series.entries if e["condition"] == condition
                and (table is None or str(int(e["table"])) == table)]
        return [e["rotations"] for e in rows]

    def median(self, condition: str) -> float:
        import statistics
        return float(statistics.median(self.trials(condition)))

    def printed_median(self, condition: str) -> float:
        return next(e["printed_median"] for e in self.series.entries if e["condition"] == condition)

    def conditions(self, table: str) -> List[str]:
        seen: List[str] = []
        for e in self.series.entries:
            if str(int(e["table"])) == table and e["condition"] not in seen:
                seen.append(e["condition"])
        return seen


# --- Network architectures: layer weights, work, and bytes (added 2026-10-20) ---
#
# Weight-only accounting, as the architectures chapter states it: a dense or
# convolutional weight is read once per batch and used in one multiply-add
# (two FLOPs) per example and output position; biases are counted only where
# a property says so.

from ..physics.layers import (  # noqa: E402
    calc_receptive_field as _receptive_field,
    calc_layers_for_receptive_field as _layers_for_field,
    calc_separable_cost_fraction as _separable_fraction,
    calc_gradient_product as _gradient_product,
)


class DenseLayerScenario(BaseScenario):
    """One dense layer: ``in_features`` inputs to ``out_features`` outputs, ``bytes_per_weight`` each."""
    in_features: int
    out_features: int
    bytes_per_weight: Quantity
    batch_size: int = 1

    @property
    def weights(self) -> int:
        return self.in_features * self.out_features

    @property
    def biases(self) -> int:
        return self.out_features

    @property
    def parameters(self) -> int:
        """Weights plus biases (what a framework's parameter count reports)."""
        return self.weights + self.biases

    @property
    def macs_per_example(self) -> int:
        return self.weights

    @property
    def weight_bytes(self) -> Quantity:
        return (self.weights * self.bytes_per_weight).to("byte")

    @property
    def intensity(self) -> float:
        """FLOPs per weight byte at ``batch_size``: each weight read serves the whole batch."""
        flops = 2 * self.batch_size * self.weights
        return flops / self.weight_bytes.to("byte").magnitude

    def batch_for_intensity(self, target: float) -> float:
        """Batch size at which the weight-only intensity reaches ``target`` FLOP/byte."""
        return target / self.variant(batch_size=1).intensity


class ConvLayerScenario(BaseScenario):
    """One ``kernel``-by-``kernel`` convolution, stride one, over a ``height``-by-``width``-by-``in_channels``
    input to ``out_channels`` feature maps; values of ``bytes_per_value`` each.

    ``same_padding`` keeps the output the size of the input; without it the
    window stays inside the input ("valid"), as an unpadded unfold does.
    """
    height: int
    width: int
    kernel: int
    in_channels: int
    out_channels: int
    bytes_per_value: Quantity
    same_padding: bool = True

    @property
    def weights(self) -> int:
        return self.kernel * self.kernel * self.in_channels * self.out_channels

    @property
    def output_height(self) -> int:
        return self.height if self.same_padding else self.height - self.kernel + 1

    @property
    def output_width(self) -> int:
        return self.width if self.same_padding else self.width - self.kernel + 1

    @property
    def positions(self) -> int:
        """Output positions per feature map, each a reuse of every filter weight."""
        return self.output_height * self.output_width

    @property
    def macs(self) -> int:
        return self.positions * self.weights

    @property
    def outputs(self) -> int:
        """Output activation values for one example."""
        return self.positions * self.out_channels

    @property
    def weight_bytes(self) -> Quantity:
        return (self.weights * self.bytes_per_value).to("byte")

    @property
    def activation_bytes(self) -> Quantity:
        return (self.outputs * self.bytes_per_value).to("byte")

    @property
    def activation_to_weight_ratio(self) -> float:
        return (self.activation_bytes / self.weight_bytes).to("").magnitude

    @property
    def input_values(self) -> int:
        return self.height * self.width * self.in_channels

    @property
    def dense_weights(self) -> int:
        """Weights of a dense layer from the whole input to ``out_channels`` outputs."""
        return self.input_values * self.out_channels

    @property
    def sharing_ratio(self) -> float:
        """Dense weights over convolution weights for the same input and feature count."""
        return self.dense_weights / self.weights

    def dense_weight_growth(self, height: int, width: int) -> float:
        """Factor by which the dense layer's weights grow when the input becomes ``height`` by ``width``."""
        return (height * width) / (self.height * self.width)

    @property
    def unfolded_entries(self) -> int:
        """Entries of the im2col patch matrix: one row of ``kernel**2 * in_channels`` per position."""
        return self.positions * self.kernel * self.kernel * self.in_channels

    @property
    def max_copies(self) -> int:
        """Most rows of the patch matrix one input value can appear in."""
        return self.kernel * self.kernel

    @property
    def depthwise_weights(self) -> int:
        """One ``kernel``-by-``kernel`` filter per input channel (multiply-adds per position)."""
        return self.kernel * self.kernel * self.in_channels

    @property
    def pointwise_weights(self) -> int:
        """A 1x1 convolution mixing ``in_channels`` into ``out_channels`` (multiply-adds per position)."""
        return self.in_channels * self.out_channels

    @property
    def separable_weights(self) -> int:
        return self.depthwise_weights + self.pointwise_weights

    @property
    def separable_ratio(self) -> float:
        """Standard over depthwise separable weights (and multiply-adds per position)."""
        return self.weights / self.separable_weights

    @property
    def separable_fraction(self) -> float:
        return _separable_fraction(self.kernel, self.out_channels)

    @property
    def separable_limit(self) -> int:
        """The ratio's limit with many output channels: the filter area."""
        return self.kernel * self.kernel


class LayerStackScenario(BaseScenario):
    """Weight-only count of a network built from convolution and dense layers, input first (biases omitted).

    A multilayer perceptron has no convolution layers; a small CNN ends in a
    dense classifier head. (The full forward and training costs of a dense
    network belong to the neural-computation solvers; this counts weights.)
    """
    conv_layers: List[ConvLayerScenario] = []
    dense_layers: List[DenseLayerScenario]

    @property
    def widths(self) -> List[int]:
        """Dense widths, input first."""
        return [self.dense_layers[0].in_features] + [d.out_features for d in self.dense_layers]

    @property
    def conv_weights(self) -> int:
        return sum(c.weights for c in self.conv_layers)

    @property
    def dense_weights(self) -> int:
        return sum(d.weights for d in self.dense_layers)

    @property
    def weights(self) -> int:
        return self.conv_weights + self.dense_weights

    @property
    def macs_per_example_dense(self) -> int:
        """Multiply-adds of the dense layers per example (one per weight)."""
        return self.dense_weights

    @property
    def dense_share(self) -> float:
        """Share of the weights in the dense layers (a CNN's classifier head)."""
        return self.dense_weights / self.weights


class DenseForwardExampleScenario(BaseScenario):
    """A small dense layer computed by hand: ``inputs`` times ``weights`` (one column per neuron), then ReLU."""
    inputs: List[float]
    weights: List[List[float]]       # rows: inputs; columns: neurons

    @property
    def neurons(self) -> int:
        return len(self.weights[0])

    def terms(self, neuron: int) -> List[float]:
        """The products x_i * w_ij that sum to neuron ``neuron`` (zero-based)."""
        return [x * row[neuron] for x, row in zip(self.inputs, self.weights)]

    @property
    def preactivations(self) -> List[float]:
        return [sum(self.terms(j)) for j in range(self.neurons)]

    @property
    def activations(self) -> List[float]:
        return [max(0.0, z) for z in self.preactivations]

    @property
    def multiply_adds(self) -> int:
        return len(self.inputs) * self.neurons


class ReceptiveFieldScenario(BaseScenario):
    """A stack of ``layers`` stride-one ``kernel``-wide convolutions, and the depth a ``target_field`` needs."""
    kernel: int
    layers: int
    target_field: int

    @property
    def growth_per_layer(self) -> int:
        return self.kernel - 1

    @property
    def field(self) -> int:
        return _receptive_field(self.layers, self.kernel)

    @property
    def layers_for_target(self) -> int:
        return _layers_for_field(self.target_field, self.kernel)


class GradientProductScenario(BaseScenario):
    """A plain stack of ``depth`` layers whose per-layer gradient factors average ``gain``."""
    gain: float
    depth: int

    @property
    def product(self) -> float:
        return _gradient_product(self.gain, self.depth)


class ElementwiseIntensityScenario(BaseScenario):
    """An unfused elementwise operation: ``flops_per_element`` FLOPs while ``operands_moved`` values
    of ``bytes_per_value`` each are read or written."""
    flops_per_element: int
    operands_moved: int
    bytes_per_value: Quantity

    @property
    def bytes_per_element(self) -> Quantity:
        return (self.operands_moved * self.bytes_per_value).to("byte")

    @property
    def intensity(self) -> float:
        """FLOPs per byte moved."""
        return self.flops_per_element / self.bytes_per_element.to("byte").magnitude


class FrameRateHeadroomScenario(BaseScenario):
    """``workload`` at ``frame_rate`` on an accelerator of ``peak_rate`` that sustains a
    ``utilization_low`` to ``utilization_high`` share of peak; ``alternative_flops`` is a heavier model per frame."""
    workload: Workload
    frame_rate: Quantity
    peak_rate: Quantity
    utilization_low: float
    utilization_high: float
    alternative_flops: Quantity

    @property
    def flops_per_frame(self) -> Quantity:
        return self.workload.inference_flops.to("GFLOP")

    @property
    def required_rate(self) -> Quantity:
        return (self.frame_rate * self.flops_per_frame).to("GFLOP/s")

    @property
    def sustained_low(self) -> Quantity:
        return (self.peak_rate * self.utilization_low).to("TFLOP/s")

    @property
    def sustained_high(self) -> Quantity:
        return (self.peak_rate * self.utilization_high).to("TFLOP/s")

    @property
    def headroom(self) -> float:
        """Sustained rate at the low utilization over the rate the stream needs."""
        return (self.sustained_low / self.required_rate).to("").magnitude

    @property
    def alternative_required_rate(self) -> Quantity:
        return (self.frame_rate * self.alternative_flops).to("TFLOP/s")

    @property
    def alternative_headroom(self) -> float:
        return (self.sustained_low / self.alternative_required_rate).to("").magnitude

    @property
    def peak_frame_rate(self) -> Quantity:
        """Frames a second if the accelerator ran at peak (the planning mistake)."""
        return (self.peak_rate / self.flops_per_frame).to("1/s")


class WeightStorageFitScenario(BaseScenario):
    """``workload``'s weights at ``bytes_per_weight`` against ``hardware``'s flash and SRAM."""
    workload: Workload
    hardware: HardwareNode
    bytes_per_weight: Quantity

    @property
    def weight_bytes(self) -> Quantity:
        return (self.workload.parameters.to("param").magnitude * self.bytes_per_weight).to("byte")

    @property
    def flash_share(self) -> float:
        return (self.weight_bytes / self.hardware.memory.flash_capacity).to("").magnitude

    @property
    def sram_ratio(self) -> float:
        """Weight bytes over SRAM capacity (above one: the weights cannot all sit in SRAM)."""
        return (self.weight_bytes / self.hardware.memory.sram_capacity).to("").magnitude


class RecurrentStepScenario(BaseScenario):
    """One step of a recurrent layer: ``input_dim`` inputs, a ``hidden_dim`` state, and ``gates`` affine
    maps per step (one for a plain RNN, four for an LSTM); ``steps`` steps in a sequence."""
    input_dim: int
    hidden_dim: int
    bytes_per_value: Quantity
    gates: int = 1
    steps: int = 1

    @property
    def gate_rows(self) -> int:
        """Rows of the fused input and recurrent weight matrices."""
        return self.gates * self.hidden_dim

    @property
    def recurrent_macs(self) -> int:
        return self.gate_rows * self.hidden_dim

    @property
    def input_macs(self) -> int:
        return self.gate_rows * self.input_dim

    @property
    def macs_per_step(self) -> int:
        return self.recurrent_macs + self.input_macs

    @property
    def weights(self) -> int:
        """Input and recurrent weights (biases omitted); one multiply-add each per step."""
        return self.macs_per_step

    @property
    def state_bytes(self) -> Quantity:
        """One state vector: what inference keeps at any sequence length."""
        return (self.hidden_dim * self.bytes_per_value).to("byte")

    @property
    def training_states(self) -> int:
        """States backpropagation through time keeps: one per step."""
        return self.steps


class KernelShareScenario(BaseScenario):
    """One kernel class's multiply-adds against a whole model's, per example."""
    workload: Workload
    kernel_macs: Quantity            # multiply-adds of the kernel class, per example
    total_macs: Quantity             # multiply-adds of the whole model, per example

    @property
    def share(self) -> float:
        return (self.kernel_macs / self.total_macs).to("").magnitude


# --- Neural computation: dense networks, activations, losses, and number formats (added 2026-10-20) ---


class DenseNetworkScenario(BaseScenario):
    """A dense network of ``layer_widths`` at ``batch_size``: parameters, FLOPs, traffic, memory.

    ``bytes_per_value`` is the storage width of every weight, gradient, and
    activation; ``optimizer_bytes_per_parameter`` the optimizer's state per
    parameter (two FP32 values for Adam). With a ``dataset``, the result also
    gives the steps in one epoch over its training split.
    """
    layer_widths: List[int]
    batch_size: int = 1
    bytes_per_value: Quantity
    optimizer_bytes_per_parameter: Quantity
    dataset: Optional[DatasetProfile] = None

    @property
    def result(self):
        from .solvers import DenseNetworkModel
        examples = None
        if self.dataset is not None and self.dataset.training_examples is not None:
            examples = int(self.dataset.training_examples.to("count").magnitude)
        return DenseNetworkModel().solve(self.layer_widths, self.batch_size, self.bytes_per_value,
                                         self.optimizer_bytes_per_parameter, examples)

    @property
    def architecture(self) -> str:
        """The widths joined by arrows, ``784 → 128 → 64 → 10``."""
        return " → ".join(str(w) for w in self.layer_widths)


class DenseScalingSweepScenario(BaseScenario):
    """Dense networks between a fixed input and output, swept in hidden width and in depth.

    Width sweep: ``hidden_layers`` hidden layers, each of every width in
    ``widths``. Depth sweep: 1 to ``max_depth`` hidden layers of
    ``fixed_width``. Each doubling factor compares the last setting with the
    one half its size.
    """
    input_dim: int
    output_dim: int
    hidden_layers: int
    widths: List[int]
    fixed_width: int
    max_depth: int

    def net(self, hidden_layers: int, width: int) -> List[int]:
        return [self.input_dim] + [width] * hidden_layers + [self.output_dim]

    @property
    def depths(self) -> List[int]:
        return list(range(1, self.max_depth + 1))

    @property
    def width_sweep_macs(self) -> List[int]:
        from ..physics.neural import calc_dense_macs
        return [calc_dense_macs(self.net(self.hidden_layers, w)) for w in self.widths]

    @property
    def depth_sweep_macs(self) -> List[int]:
        from ..physics.neural import calc_dense_macs
        return [calc_dense_macs(self.net(d, self.fixed_width)) for d in self.depths]

    @property
    def depth_sweep_saved_activations(self) -> List[int]:
        """Values saved per example: the input and every layer output."""
        from ..physics.neural import calc_dense_saved_activations
        return [calc_dense_saved_activations(self.net(d, self.fixed_width)) for d in self.depths]

    @property
    def width_doubled_from(self) -> int:
        return self.widths[-1] // 2

    @property
    def depth_doubled_from(self) -> int:
        return self.max_depth // 2

    @property
    def width_double_factor(self) -> float:
        m = self.width_sweep_macs
        return m[-1] / m[self.widths.index(self.width_doubled_from)]

    @property
    def depth_double_factor(self) -> float:
        m = self.depth_sweep_macs
        return m[-1] / m[self.depth_doubled_from - 1]

    @property
    def saved_activations_double_factor(self) -> float:
        a = self.depth_sweep_saved_activations
        return a[-1] / a[self.depth_doubled_from - 1]


class EpochOverheadScenario(BaseScenario):
    """One training epoch of ``network`` on a host CPU and an accelerator: work against fixed overhead.

    Every launched operation pays ``dispatch`` (the framework's Python-side
    cost); on the accelerator it also pays ``launch``. ``flop_energy`` and
    ``dram_energy_per_byte`` price arithmetic against data movement.
    """
    network: DenseNetworkScenario
    host: HardwareNode
    accelerator: HardwareNode
    precision: str = "fp32"
    dispatch: Quantity
    launch: Quantity
    dram_energy_per_byte: Quantity
    flop_energy: Quantity

    @property
    def result(self):
        from .solvers import EpochOverheadModel
        net = self.network.result
        return EpochOverheadModel().solve(
            net, self.network.variant(batch_size=1).result, self.host, self.accelerator, self.precision,
            self.dispatch, self.launch, net.steps_per_epoch, self.dram_energy_per_byte, self.flop_energy)


class ParadigmCostScenario(BaseScenario):
    """One input classified three ways: hand-written rules, engineered features, and a learned network.

    Rules: ``rule_comparisons`` comparisons over the raw pixels. Features: a
    ``feature_grid`` by ``feature_grid`` grid of cells, ``feature_bins``
    orientation bins each, ``feature_operations`` operations in
    ``feature_memory`` of working state. Network: ``network``'s MACs.
    """
    image_pixels: int
    bytes_per_pixel: Quantity
    rule_comparisons: int
    feature_grid: int
    feature_bins: int
    feature_operations: int
    feature_memory: Quantity
    network: DenseNetworkScenario

    @property
    def rule_memory(self) -> Quantity:
        return (self.image_pixels * self.bytes_per_pixel).to("byte")

    @property
    def feature_count(self) -> int:
        return self.feature_grid ** 2 * self.feature_bins

    @property
    def feature_to_rule_ratio(self) -> float:
        return self.feature_operations / self.rule_comparisons

    @property
    def network_to_rule_ratio(self) -> float:
        return self.network.result.macs_per_example / self.rule_comparisons


class ActivationLogicCostScenario(BaseScenario):
    """Logic cost of two activation datapaths, in transistor equivalents (an illustrative budget)."""
    cheap_label: str
    cheap_transistors: int
    costly_label: str
    costly_transistors: int

    @property
    def ratio(self) -> float:
        return self.costly_transistors / self.cheap_transistors


class GradientChainScenario(BaseScenario):
    """The activation-slope factor a gradient keeps after ``depth`` layers, at each slope's peak."""
    depth: int
    activation: str = "sigmoid"

    @property
    def peak_slope(self) -> float:
        from ..physics.neural import calc_sigmoid_slope
        if self.activation != "sigmoid":
            raise ValueError(f"unsupported activation {self.activation!r}")
        return calc_sigmoid_slope(0.0)

    @property
    def surviving_fraction(self) -> float:
        return self.peak_slope ** self.depth


class CrossEntropyScenario(BaseScenario):
    """Cross-entropy losses for named predicted probabilities of the correct class, among ``classes``.

    ``misread`` gives the probabilities of a confident mistake, keyed
    ``"true"`` (the correct class) and ``"wrong"`` (the class predicted);
    the gradient on each score is its probability minus its target.
    """
    classes: int
    probabilities: Dict[str, float]
    misread: Dict[str, float]

    def loss(self, key: str) -> float:
        from ..physics.neural import calc_cross_entropy
        return calc_cross_entropy(self.probabilities[key])

    @property
    def uniform_loss(self) -> float:
        from ..physics.neural import calc_cross_entropy
        return calc_cross_entropy(1 / self.classes)

    @property
    def misread_loss(self) -> float:
        from ..physics.neural import calc_cross_entropy
        return calc_cross_entropy(self.misread["true"])

    @property
    def misread_gradient_true(self) -> float:
        return self.misread["true"] - 1.0

    @property
    def misread_gradient_wrong(self) -> float:
        return self.misread["wrong"] - 0.0


class BackpropTraceScenario(BaseScenario):
    """One forward and backward pass by hand: two inputs, two ReLU hidden units, one linear output.

    Squared-error loss ``(y_hat - target)^2 / 2``, biases zero; one
    gradient-descent step at ``learning_rate``. ``w1[i][j]`` connects input
    ``i`` to hidden unit ``j``.
    """
    x: List[float]
    w1: List[List[float]]
    w2: List[float]
    target: float
    learning_rate: float

    @property
    def z(self) -> List[float]:
        return [sum(self.x[i] * self.w1[i][j] for i in range(len(self.x))) for j in range(len(self.w2))]

    @property
    def a(self) -> List[float]:
        from ..physics.neural import calc_relu
        return [calc_relu(v) for v in self.z]

    @property
    def y_hat(self) -> float:
        return sum(a * w for a, w in zip(self.a, self.w2))

    @property
    def loss(self) -> float:
        return (self.y_hat - self.target) ** 2 / 2

    @property
    def error(self) -> float:
        return self.y_hat - self.target

    @property
    def w2_gradient(self) -> List[float]:
        return [a * self.error for a in self.a]

    @property
    def sent_back(self) -> List[float]:
        return [self.error * w for w in self.w2]

    @property
    def delta(self) -> List[float]:
        return [s if z > 0 else 0.0 for s, z in zip(self.sent_back, self.z)]

    @property
    def w1_gradient(self) -> List[List[float]]:
        return [[xi * d for d in self.delta] for xi in self.x]

    def w1_updated(self, i: int, j: int) -> float:
        return self.w1[i][j] - self.learning_rate * self.w1_gradient[i][j]


class NumberFormatRangeScenario(BaseScenario):
    """What range each number format holds, and what one ``value`` becomes when stored in it.

    ``formats`` maps a name to (exponent bits, fraction bits);
    ``integer_bits`` is a signed integer width for comparison.
    """
    value: float
    formats: Dict[str, Tuple[int, int]]
    integer_bits: int

    def range(self, fmt: str):
        """``(largest, smallest_normal, smallest_subnormal)`` of ``fmt``."""
        from ..physics.numerics import calc_float_range
        return calc_float_range(*self.formats[fmt])

    def stored(self, fmt: str) -> float:
        from ..physics.numerics import calc_float_round
        return calc_float_round(self.value, *self.formats[fmt])

    @property
    def integer_max(self) -> int:
        from ..physics.numerics import calc_int_max
        return calc_int_max(self.integer_bits)


class TrainingStateFitScenario(BaseScenario):
    """Weights, gradients, and optimizer state of ``workload`` at one precision, against ``hardware``'s memory.

    ``bytes_per_value`` stores each weight and gradient; the optimizer keeps
    ``optimizer_bytes_per_parameter`` (two FP32 values for Adam, one for SGD
    with momentum). Activations are not included.
    """
    workload: Workload
    hardware: Optional[HardwareNode] = None
    bytes_per_value: Quantity
    optimizer_bytes_per_parameter: Quantity

    def _state(self):
        from ..physics.neural import calc_training_state
        p = self.workload.parameters.to("count").magnitude
        return [q.to("byte") for q in calc_training_state(p, self.bytes_per_value, self.optimizer_bytes_per_parameter)]

    @property
    def weights(self) -> Quantity:
        return self._state()[0]

    @property
    def gradients(self) -> Quantity:
        return self._state()[1]

    @property
    def optimizer_state(self) -> Quantity:
        return self._state()[2]

    @property
    def optimizer_values_per_parameter(self) -> float:
        return (self.optimizer_bytes_per_parameter / self.bytes_per_value).to("").magnitude

    @property
    def total(self) -> Quantity:
        return sum(self._state()[1:], self._state()[0])

    @property
    def copies(self) -> float:
        """Total state in units of the weights."""
        return (self.total / self.weights).to("").magnitude

    @property
    def fits(self) -> bool:
        return self.hardware is not None and self.total <= self.hardware.memory.capacity


class ParameterGrowthScenario(BaseScenario):
    """Parameter counts of a model series over time, fit separately before and after ``split_year``."""
    rows: List[Dict[str, Any]]
    split_year: int

    @property
    def result(self):
        from .solvers import ParameterGrowthModel
        return ParameterGrowthModel().solve(self.rows, self.split_year)


# ── ML operations at scale (added 2026-10-20) ────────────────────────────────
# A portfolio of models sharing features, releases, alerts, and capacity: the
# coupling sketch, feature history and lookups, incident costs, staged
# exposure, experiment sizing, alert volume, SLO budgets, and the economics of
# owning a fleet.


class OperationalLoadScenario(BaseScenario):
    """Assumed operational-load shapes for ``N`` models: alerts ``a N``, deployments ``N ln N``,
    and pairwise conflicts ``c N^2`` against one team's ``team_capacity``.

    The shapes are a sketch, not measurements; ``crossing_models`` is the first
    model count whose total reaches the capacity.
    """
    alerts_per_model: float
    conflict_coefficient: float       # N models form about N^2/2 pairs
    team_capacity: float
    max_models: int

    def model_counts(self):
        import numpy as _np
        return _np.arange(1, self.max_models + 1)

    def alerts(self, n):
        return self.alerts_per_model * n

    def deployments(self, n):
        import numpy as _np
        return n * _np.log(n)

    def conflicts(self, n):
        return self.conflict_coefficient * n ** 2

    def total(self, n):
        return self.alerts(n) + self.deployments(n) + self.conflicts(n)

    @property
    def crossing_models(self) -> int:
        import numpy as _np
        n = self.model_counts()
        return int(n[int(_np.argmax(self.total(n) >= self.team_capacity))])


class FeatureHistoryScenario(BaseScenario):
    """Full-snapshot feature history: ``entities`` x ``features`` values every ``update_interval``,
    kept for ``retention``, each row a value, an entity key, and ``timestamps`` timestamps."""
    entities: float
    features: int
    retention: Quantity
    update_interval: Quantity
    value_bytes: Quantity
    key_bytes: Quantity
    timestamp_bytes: Quantity
    timestamps: int = 2               # event time and availability time

    @property
    def versions_per_feature(self) -> float:
        return (self.retention / self.update_interval).to("").magnitude

    @property
    def values(self) -> float:
        return self.entities * self.features * self.versions_per_feature

    @property
    def row_bytes(self) -> Quantity:
        return (self.value_bytes + self.key_bytes + self.timestamps * self.timestamp_bytes).to("byte")

    @property
    def total_bytes(self) -> Quantity:
        return self.values * self.row_bytes

    @property
    def metadata_share(self) -> float:
        return float(((self.row_bytes - self.value_bytes) / self.row_bytes).to("").magnitude)


class PointInTimeJoinScenario(BaseScenario):
    """A cumulative feature joined to a training example: ``events_before_prediction`` were
    available at the prediction time, ``events_by_day_end`` by the end of the day."""
    events_before_prediction: int
    events_by_day_end: int

    @property
    def leakage_inflation(self) -> float:
        """What a naive join to the end-of-day table feeds the model, against the true value."""
        return self.events_by_day_end / self.events_before_prediction


class FeatureIncidentScenario(BaseScenario):
    """A silent shared-feature incident as reported: traffic, gates passed, share affected,
    metric change, time unalerted, and the consumers and owners it reached."""
    request_rate: Optional[Quantity] = None
    latency_gate: Optional[Quantity] = None
    affected_share: Optional[float] = None       # fraction of requests with the broken feature
    metric_drop: Optional[float] = None          # relative fall of the business metric
    undetected_time: Optional[Quantity] = None
    consumer_models: Optional[int] = None
    owning_teams: Optional[int] = None
    accuracy_drop: Optional[float] = None
    debug_time: Optional[Quantity] = None


class RevenueRegressionScenario(BaseScenario):
    """A metric regression that cuts revenue in proportion for ``undetected`` (or for ``alerted``
    with an alert and containment); revenue is ``weekly_revenue`` per week."""
    weekly_revenue: Quantity          # USD per week
    relative_drop: float              # fraction of revenue lost while the regression lasts
    metric_before: float
    metric_after: float
    undetected: Quantity
    alerted: Quantity
    feature_mean_before: Optional[float] = None
    feature_mean_after: Optional[float] = None
    freshness_slo: Optional[Quantity] = None
    shift_threshold: Optional[float] = None

    def loss(self, duration: Quantity) -> Quantity:
        return (self.relative_drop * self.weekly_revenue * (duration / Q_(1, "week")).to("").magnitude).to("USD")

    @property
    def undetected_loss(self) -> Quantity:
        return self.loss(self.undetected)

    @property
    def alerted_loss(self) -> Quantity:
        return self.loss(self.alerted)

    @property
    def window_ratio(self) -> float:
        return (self.undetected / self.alerted).to("").magnitude

    @property
    def impact_reduction(self) -> float:
        return 1 - (self.alerted / self.undetected).to("").magnitude

    @property
    def metric_change(self) -> float:
        return self.metric_after / self.metric_before - 1

    @property
    def feature_shift(self) -> float:
        return (self.feature_mean_after - self.feature_mean_before) / self.feature_mean_before


class ClickLossScenario(BaseScenario):
    """A silent click-through regression from ``ctr_before`` to ``ctr_after`` at ``request_rate``,
    caught after ``detection_time``; each click is worth ``value_per_click``."""
    request_rate: Quantity
    detection_time: Quantity
    ctr_before: float
    ctr_after: float
    value_per_click: Quantity

    @property
    def requests(self) -> float:
        return (self.request_rate * self.detection_time).to("").magnitude

    @property
    def ctr_drop(self) -> float:
        return self.ctr_before - self.ctr_after

    @property
    def lost_clicks(self) -> float:
        return self.requests * self.ctr_drop

    @property
    def loss(self) -> Quantity:
        return (self.lost_clicks * self.value_per_click).to("USD")

    @property
    def loss_rate(self) -> Quantity:
        return (self.loss / self.detection_time).to("USD/hour")


class RolloutExposureScenario(BaseScenario):
    """Users exposed until rollback under an all-at-once cutover against a canary share,
    with the same request rate, harm, and time to rollback."""
    full_share: float
    canary_share: float
    time_to_rollback: Optional[Quantity] = None

    @property
    def exposure_ratio(self) -> float:
        return self.full_share / self.canary_share


class CanaryScheduleScenario(BaseScenario):
    """A staged rollout that needs ``samples_per_variant`` guardrail samples per stage at
    ``request_rate``; ``stages`` is the configured ladder of (traffic share, duration)."""
    samples_per_variant: float
    request_rate: Quantity            # requests per hour
    stages: List[Any]                 # [(share, Quantity duration), ...]

    def stage_minimum(self, share: float) -> Quantity:
        from ..physics.statistics import calc_sample_collection_time
        return calc_sample_collection_time(self.samples_per_variant, self.request_rate, share).to("hour")

    @property
    def total(self) -> Quantity:
        return sum((d for _, d in self.stages), Q_(0, "hour")).to("hour")

    @property
    def first_stage_buffer(self) -> float:
        share, duration = self.stages[0]
        return (duration / self.stage_minimum(share)).to("").magnitude


class TwoProportionTestScenario(BaseScenario):
    """A two-arm test sized to detect an absolute ``lift`` on a ``baseline_rate`` at the
    normal quantiles ``z_alpha`` and ``z_beta``, on ``daily_requests`` split between arms.

    ``family_size`` uncorrected tests at ``significance`` give the familywise error.
    """
    baseline_rate: float
    lift: float
    z_alpha: float
    z_beta: float
    daily_requests: Optional[Quantity] = None
    significance: float = 0.05
    family_size: int = 20

    @property
    def z_sum_squared(self) -> float:
        return (self.z_alpha + self.z_beta) ** 2

    @property
    def two_variance(self) -> float:
        return 2 * self.baseline_rate * (1 - self.baseline_rate)

    @property
    def lift_squared(self) -> float:
        return self.lift ** 2

    @property
    def per_variant(self) -> float:
        from ..physics.statistics import calc_two_proportion_sample_size
        return calc_two_proportion_sample_size(self.baseline_rate, self.lift, self.z_alpha, self.z_beta)

    @property
    def total(self) -> float:
        return 2 * self.per_variant

    @property
    def collection_time(self) -> Quantity:
        return (self.total / self.daily_requests).to("hour")

    @property
    def familywise_error(self) -> float:
        from ..physics.agents import calc_independent_success
        return calc_independent_success(self.significance, self.family_size)


class MonitorFleetAlertScenario(BaseScenario):
    """``models`` x ``metrics_per_model`` monitors checked every ``check_interval`` at a per-check
    ``false_positive_rate`` on a stable process; ``dedup_share`` of crossings are merged away."""
    models: int
    metrics_per_model: int
    check_interval: Quantity
    false_positive_rate: float
    dedup_share: Optional[float] = None

    @property
    def monitors(self) -> int:
        return self.models * self.metrics_per_model

    @property
    def checks_per_day(self) -> float:
        return (Q_(1, "day") / self.check_interval).to("").magnitude

    @property
    def daily_false_alerts(self) -> float:
        return self.monitors * self.checks_per_day * self.false_positive_rate

    @property
    def remaining_after_dedup(self) -> float:
        return self.daily_false_alerts * (1 - self.dedup_share)

    def any_false_alert(self, checks: int) -> float:
        """Chance that at least one of ``checks`` independent checks crosses falsely."""
        from ..physics.agents import calc_independent_success
        return calc_independent_success(self.false_positive_rate, checks)


class SloBudgetVectorScenario(BaseScenario):
    """Budget consumed per SLO dimension (observed over allowed violating fraction); the
    worst dimension, not the mean, decides promotion."""
    consumption: Dict[str, float]
    owners: Dict[str, str] = {}

    @property
    def mean(self) -> float:
        return sum(self.consumption.values()) / len(self.consumption)

    @property
    def worst(self) -> float:
        return max(self.consumption.values())

    @property
    def worst_dimension(self) -> str:
        return max(self.consumption, key=self.consumption.get)

    @property
    def promote(self) -> bool:
        return self.worst <= 1


class ErrorBudgetBurnScenario(BaseScenario):
    """Burn-rate alert tiers on an SLO of ``slo_target`` over ``budget_period``: each tier is
    (burn rate, long window); a tier spends ``burn x window / period`` of the budget."""
    slo_target: float
    budget_period: Quantity
    tiers: Dict[str, Any]             # name -> (burn rate, Quantity window)

    @property
    def error_budget(self) -> float:
        return 1 - self.slo_target

    def budget_spent(self, tier: str) -> float:
        from ..physics.reliability import calc_error_budget_spent
        burn, window = self.tiers[tier]
        return calc_error_budget_spent(burn, window, self.budget_period)


class OwnVsRentScenario(BaseScenario):
    """Owning ``gpus`` accelerators against renting them at ``rent_rate`` per GPU-hour for
    ``utilization`` of every hour, over ``years``.

    Owning pays the nodes, network, and facility up front, then a running cost
    per year: a fixed ``opex_per_year`` and ``staffing_per_year``, plus
    electricity for ``power`` at ``pue`` and ``electricity_price``. Electricity
    is charged every hour unless ``energy_scales_with_use``, when it is drawn
    only while busy. A year is 8,760 hours and a month one twelfth of it, the
    book's convention.
    """
    gpus: int
    gpus_per_node: int
    node_price: Quantity
    rent_rate: Quantity               # USD per GPU-hour
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
        from ..core.units import HOURS_PER_YEAR
        return Q_(HOURS_PER_YEAR, "hour")

    @property
    def nodes(self) -> int:
        return self.gpus // self.gpus_per_node

    @property
    def hardware_capex(self) -> Quantity:
        return (self.nodes * self.node_price).to("USD")

    @property
    def upfront(self) -> Quantity:
        return (self.hardware_capex + self.network_capex + self.facility_capex).to("USD")

    @property
    def electricity_per_year_full(self) -> Quantity:
        """Electricity for one year of full use (PUE included)."""
        if self.power is None:
            return Q_(0, "USD")
        energy = (self.power * self.hours_per_year * self.pue).to("kWh")
        return (energy * self.electricity_price).to("USD")

    @property
    def fixed_cost_per_year(self) -> Quantity:
        fixed = self.opex_per_year + self.staffing_per_year
        if not self.energy_scales_with_use:
            fixed = fixed + self.electricity_per_year_full
        return fixed.to("USD")

    @property
    def use_cost_per_year_full(self) -> Quantity:
        return self.electricity_per_year_full if self.energy_scales_with_use else Q_(0, "USD")

    @property
    def owned_cost_per_year(self) -> Quantity:
        """Running cost of owning per year at this utilization."""
        return (self.fixed_cost_per_year + self.utilization * self.use_cost_per_year_full).to("USD")

    @property
    def owned_cost_per_year_amortized(self) -> Quantity:
        """Up-front cost spread evenly over the years, plus the running cost."""
        return (self.upfront / self.years + self.owned_cost_per_year).to("USD")

    @property
    def rent_per_year_full(self) -> Quantity:
        return (self.gpus * self.rent_rate * self.hours_per_year).to("USD")

    @property
    def rent_per_year(self) -> Quantity:
        return (self.rent_per_year_full * self.utilization).to("USD")

    @property
    def rent_per_month(self) -> Quantity:
        return self.rent_per_year / 12

    @property
    def owned_cost_per_month(self) -> Quantity:
        return self.owned_cost_per_year / 12

    @property
    def crossover_months(self) -> Optional[float]:
        """Months until cumulative rent overtakes owning; None if it never does."""
        from ..physics.economics import calc_rent_crossover
        t = calc_rent_crossover(self.upfront, self.owned_cost_per_month, self.rent_per_month)
        return None if t is None else t.to("").magnitude

    @property
    def owned_total(self) -> Quantity:
        return (self.upfront + self.years * self.owned_cost_per_year).to("USD")

    @property
    def rented_total(self) -> Quantity:
        return (self.years * self.rent_per_year).to("USD")

    @property
    def savings(self) -> Quantity:
        return (self.rented_total - self.owned_total).to("USD")

    @property
    def rent_lifetime_full(self) -> Quantity:
        """Rent for every hour of the whole life."""
        return (self.years * self.rent_per_year_full).to("USD")

    @property
    def lifetime_hours(self) -> Quantity:
        return (self.years * self.hours_per_year).to("hour")

    @property
    def breakeven_utilization(self) -> float:
        from ..physics.economics import calc_ownership_breakeven_utilization
        return float(calc_ownership_breakeven_utilization(
            self.upfront, self.fixed_cost_per_year, self.rent_per_year_full, self.years,
            self.use_cost_per_year_full).to("").magnitude)


class TrainingCampaignCostScenario(BaseScenario):
    """``runs_per_year`` training runs of ``run_low`` to ``run_high`` on an owned cluster
    (``ownership``) against renting the same GPUs only while each run lasts."""
    ownership: OwnVsRentScenario
    runs_per_year: int
    run_low: Quantity
    run_high: Quantity

    @property
    def runs(self) -> int:
        return self.runs_per_year * self.ownership.years

    @property
    def owned_cost_per_run(self) -> Quantity:
        return (self.ownership.owned_total / self.runs).to("USD")

    def rented_cost_per_run(self, length: Quantity) -> Quantity:
        o = self.ownership
        return (o.gpus * length * o.rent_rate).to("USD")


class LifecycleTcoScenario(BaseScenario):
    """Monthly lifecycle cost in four terms (training, inference, data, iteration), each counted once."""
    monthly: Dict[str, Quantity]      # term -> USD per month

    @property
    def total(self) -> Quantity:
        return sum(self.monthly.values(), Q_(0, "USD/month")).to("USD/month")

    @property
    def shares(self) -> Dict[str, float]:
        t = self.total
        return {k: (v / t).to("").magnitude for k, v in self.monthly.items()}

    @property
    def dominant(self) -> str:
        return max(self.monthly, key=lambda k: self.monthly[k].to("USD/month").magnitude)

    def ratios_to(self, other: "LifecycleTcoScenario") -> Dict[str, float]:
        """This budget's terms (and total) over another's."""
        out = {k: (v / other.monthly[k]).to("").magnitude for k, v in self.monthly.items()}
        out["total"] = (self.total / other.total).to("").magnitude
        return out


class OptimizationPaybackScenario(BaseScenario):
    """An optimization costing ``implementation_cost`` that cuts a ``monthly_bill`` by ``reduction``."""
    implementation_cost: Quantity
    monthly_bill: Quantity            # USD per month
    reduction: float

    @property
    def monthly_savings(self) -> Quantity:
        return (self.monthly_bill * self.reduction).to("USD/month")

    @property
    def payback(self) -> Quantity:
        return (self.implementation_cost / self.monthly_savings).to("month")


class PlatformBenefitScenario(BaseScenario):
    """A shared platform costing ``platform_cost`` per month that saves each of ``models`` model
    teams ``hours_saved`` per month at ``engineer_rate``."""
    models: int
    hours_saved: Quantity             # hours per model per month
    engineer_rate: Quantity           # USD per hour
    platform_cost: Quantity           # USD per month

    @property
    def gross_benefit(self) -> Quantity:
        return (self.models * self.hours_saved * self.engineer_rate).to("USD/month")

    @property
    def net_benefit(self) -> Quantity:
        return (self.gross_benefit - self.platform_cost).to("USD/month")

    @property
    def benefit_cost_ratio(self) -> float:
        return (self.gross_benefit / self.platform_cost).to("").magnitude


# --- Sensor perception: capture timing, age, and error scales (added 2026-10-20) ---
# A site scenario (``Embodied.Scenario.*``) supplies the machine's navigation
# camera and lidar, the stage allowances, and the clearances; the solvers live
# in ``engine/solvers/perception.py``.

class ObservationAgeScenario(BaseScenario):
    """``site.machine``'s navigation-camera observation age at ``speed``, with a ``batch_frames`` comparison."""
    site: Any                                   # EmbodiedSiteScenario
    speed: Quantity
    batch_frames: int = 1

    @property
    def result(self):
        from .solvers import ObservationAgeModel
        s, cam = self.site, self.site.machine.nav_camera
        return ObservationAgeModel().solve(cam.exposure_time, cam.readout_time, s.t_transport, s.t_dma,
                                           s.t_isp, s.t_backbone, s.t_ipc, cam.frame_rate, self.speed,
                                           self.batch_frames)


class RollingShutterScenario(BaseScenario):
    """``site.machine``'s rolling-shutter navigation camera at ``speed``, turning at the site's yaw rate toward the rack end."""
    site: Any                                   # EmbodiedSiteScenario
    speed: Quantity

    @property
    def result(self):
        from .solvers import RollingShutterSkewModel
        s, cam = self.site, self.site.machine.nav_camera
        return RollingShutterSkewModel().solve(cam.resolution_height, cam.readout_time, cam.exposure_time,
                                               self.speed, s.yaw_rate_turn, s.d_clear, s.delta_loc,
                                               s.side_clearance)


class TimingAttitudeErrorScenario(BaseScenario):
    """A ``timing_error`` and an ``angle_error`` on a target at ``distance``, priced against ``reference``."""
    speed: Quantity
    yaw_rate: Quantity
    distance: Quantity
    timing_error: Quantity
    angle_error: Quantity
    reference: Optional[Quantity] = None

    @property
    def result(self):
        from .solvers import TimingAttitudeErrorModel
        return TimingAttitudeErrorModel().solve(self.speed, self.yaw_rate, self.distance, self.timing_error,
                                                self.angle_error, self.reference)


class LidarSweepAgeScenario(BaseScenario):
    """``site.machine``'s spinning lidar at ``speed``: sweep motion and the age of the aisle sector at the rack end."""
    site: Any                                   # EmbodiedSiteScenario
    speed: Quantity

    @property
    def result(self):
        from .solvers import LidarSweepAgeModel
        s, lidar = self.site, self.site.machine.lidar
        return LidarSweepAgeModel().solve(lidar.scan_rate, lidar.nominal_latency, lidar.time_sync_accuracy,
                                          s.aisle_width, s.d_clear, self.speed, s.yaw_rate_turn,
                                          s.perception_age)


class StereoDepthErrorScenario(BaseScenario):
    """A stereo rig's depth error at two ranges and at ``site``'s rack end, against the site's clearances."""
    site: Any                                   # EmbodiedSiteScenario
    baseline: Quantity
    focal_length_px: float
    disparity_sigma_px: float
    near_range: Quantity
    far_range: Quantity
    ray_clearance: Quantity
    baseline_change: Quantity
    bias_range: Quantity

    @property
    def result(self):
        from .solvers import StereoDepthErrorModel
        s = self.site
        return StereoDepthErrorModel().solve(self.baseline, self.focal_length_px, self.disparity_sigma_px,
                                             self.near_range, self.far_range, s.d_clear, self.ray_clearance,
                                             s.delta_loc, s.side_clearance, self.baseline_change,
                                             self.bias_range)


class DelayPhaseMarginScenario(BaseScenario):
    """A loop with ``phase_margin`` at ``crossover``, closed through a pure ``delay``."""
    phase_margin: Quantity
    crossover: Quantity
    delay: Quantity

    @property
    def result(self):
        from .solvers import DelayPhaseMarginModel
        return DelayPhaseMarginModel().solve(self.phase_margin, self.crossover, self.delay)


class ConformalInsetScenario(BaseScenario):
    """A split-conformal bound at miss rate ``alpha`` from ``calibration_count`` residuals."""
    alpha: float
    calibration_count: int

    @property
    def result(self):
        from .solvers import ConformalInsetModel
        return ConformalInsetModel().solve(self.alpha, self.calibration_count)


class ClockConversionScenario(BaseScenario):
    """A sensor clock drifting at ``drift_rate`` between syncs every ``sync_interval``, converted under ``site``'s bound."""
    site: Any                                   # EmbodiedSiteScenario
    drift_rate: Quantity
    sync_interval: Quantity
    speed: Quantity

    @property
    def result(self):
        from .solvers import ClockConversionModel
        s = self.site
        return ClockConversionModel().solve(self.drift_rate, self.sync_interval, s.clock_conversion_bound,
                                            s.perception_age, self.speed)


class VisibilityRangeScenario(BaseScenario):
    """Fog of extinction coefficient ``extinction`` against a contrast threshold."""
    extinction: Quantity
    contrast_threshold: float

    @property
    def result(self):
        from .solvers import VisibilityRangeModel
        return VisibilityRangeModel().solve(self.extinction, self.contrast_threshold)


class ThermalExpansionScenario(BaseScenario):
    """A member of ``length`` and expansion ``coefficient`` warmed by ``temperature_rise``."""
    coefficient: Quantity
    length: Quantity
    temperature_rise: Quantity

    @property
    def result(self):
        from .solvers import ThermalExpansionModel
        return ThermalExpansionModel().solve(self.coefficient, self.length, self.temperature_rise)
