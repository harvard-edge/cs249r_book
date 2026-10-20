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

from typing import Any, Dict, List, Optional, Tuple

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
    # Optional (added 2026-10-20): factors by which a chosen interval misses the
    # optimum, priced by ``miss_penalty``.
    miss_factors: Optional[List[float]] = None

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

    def miss_penalty(self, factor: float) -> float:
        """Extra waste, relative to the optimum, of an interval ``factor`` times the optimum
        (added 2026-10-20); equals ``(x + 1/x) / 2 - 1``, so the minimum is flat."""
        r = self.result
        return self.waste_at(factor * r.interval) / r.waste - 1


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
    # Optional (added 2026-10-20): a fractional reduction of the term; the
    # resource multiple that buys it is ``reduction_multiple``.
    loss_reduction: Optional[float] = None
    # Optional (added 2026-10-20): a resource multiple; the term falls to
    # ``multiple_factor`` of its value, a fractional reduction ``multiple_reduction``.
    resource_multiple: Optional[float] = None

    @property
    def multiple_factor(self) -> float:
        from ..physics.scaling import calc_power_law_factor
        return calc_power_law_factor(self.resource_multiple, self.exponent)

    @property
    def multiple_reduction(self) -> float:
        return 1 - self.multiple_factor

    @property
    def halving_multiple(self) -> float:
        from ..physics.scaling import calc_halving_multiple
        return calc_halving_multiple(self.exponent)

    @property
    def reduction_multiple(self) -> float:
        from ..physics.scaling import calc_loss_reduction_compute_multiple
        return calc_loss_reduction_compute_multiple(self.exponent, self.loss_reduction)


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
    """A published training run: its total compute spread over its accelerators and days.

    Optional (added 2026-10-20): a run reported as ``accelerator_hours`` (a
    model card's GPU hours) at ``device_power`` per accelerator. Its compute is
    the workload's ``training_ops`` or, when unset, 6PD from its training tokens.
    """
    workload: Workload
    accelerator: HardwareNode       # the part the run used, for its peak rate
    accelerator_hours: Optional[Quantity] = None
    device_power: Optional[Quantity] = None

    @property
    def training_operations(self) -> Quantity:
        w = self.workload
        if w.training_ops is not None:
            return w.training_ops
        from ..physics.transformer import calc_transformer_training_flops
        return calc_transformer_training_flops(w.parameters, w.training_tokens).to("flop")

    @property
    def achieved_rate(self) -> Quantity:
        """Average delivered FLOP/s per accelerator over the whole run, idle time included."""
        w = self.workload
        if self.accelerator_hours is not None:
            return (self.training_operations / self.accelerator_hours).to("TFLOP/s")
        n = w.training_accelerators_ref.to("count").magnitude
        return (w.training_ops / (n * w.training_days_ref)).to("TFLOP/s")

    @property
    def energy(self) -> Quantity:
        """Every accelerator hour at ``device_power``."""
        return (self.accelerator_hours * self.device_power).to("GWh")

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
    The finished budget adds three optional terms: ``jerk`` (the resident
    S-curve stop's jerk bound), ``tracking_bound`` (the tracking inset), and
    ``approach_speed`` (a person closing the gap through delay and stop).
    """
    speed: Quantity
    delay: Quantity
    deceleration: Optional[Quantity] = None     # default: the site's credible deceleration
    clear_distance: Optional[Quantity] = None   # default: the site's clear distance
    speed_at_delay_end: Optional[Quantity] = None
    site: Optional[Any] = None                  # EmbodiedSiteScenario
    added_delay: Optional[Quantity] = None
    jerk: Optional[Quantity] = None             # S-curve stop jerk bound (added 2026-10-20)
    tracking_bound: Optional[Quantity] = None   # tracking inset (added 2026-10-20)
    approach_speed: Optional[Quantity] = None   # approaching person (added 2026-10-20)

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
                                           self.speed_at_delay_end, loc, margin, self.added_delay,
                                           self.jerk, self.tracking_bound, self.approach_speed)
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

    @property
    def allowance_decode_time(self) -> Quantity:
        """Decode time of the whole allowance (added 2026-10-20): ``K_max t_dec``, the longest
        one admitted call postpones the next turn boundary when control waits for its reply."""
        return (self.output_allowance * self.floors.result.memory_time).to("second")


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
    # Optional (added 2026-10-20): bytes per stored activation (2 for 16-bit),
    # and a stated normalization buffer per layer for the per-layer line items.
    precision_bytes: float = 2
    normalization_per_layer: Optional[Quantity] = None
    ffn_multiple: int = 4
    # Optional (added 2026-10-20): checkpoint the layers in this many equal
    # segments, keeping each segment's input and rebuilding one segment at a
    # time in the backward pass.
    checkpoint_segments: Optional[int] = None

    @property
    def sequence(self) -> int:
        return self.seq_len if self.seq_len is not None else self.workload.context_length

    # Korthikanti et al. Table 2 per layer, sbh(34 + 5as/h) (added 2026-10-20).
    @property
    def per_token_bytes_coefficient(self) -> float:
        """The 34 of sbh(34 + 5as/h): bytes per token and unit of width, at this precision."""
        from ..physics.memory import KORTHIKANTI_PER_TOKEN_BYTES
        return KORTHIKANTI_PER_TOKEN_BYTES * self.precision_bytes / 2

    @property
    def score_bytes_coefficient(self) -> float:
        """The 5as/h of sbh(34 + 5as/h): the attention scores, in the same units."""
        from ..physics.memory import KORTHIKANTI_SCORE_BYTES
        w = self.workload
        return KORTHIKANTI_SCORE_BYTES * w.heads * self.sequence / w.hidden_dim * self.precision_bytes / 2

    @property
    def score_share(self) -> float:
        """The scores' share of the saved activations, 5as/h over (34 + 5as/h)."""
        c = self.score_bytes_coefficient
        return c / (self.per_token_bytes_coefficient + c)

    @property
    def per_sequence_activations(self) -> Quantity:
        """Per-token activations of one sequence (scores not stored): the slope against batch size."""
        return (self.per_token_activations / self.micro_batch).to("GB")

    # Segment checkpointing (added 2026-10-20).
    @property
    def segment_layers(self) -> int:
        return self.workload.layers // self.checkpoint_segments

    @property
    def segment_input(self) -> Quantity:
        """One segment's input, kept for the backward pass: b x s x h values."""
        return Q_(self._elements(self.workload.hidden_dim) * self.precision_bytes, "byte").to("MB")

    @property
    def segment_inputs(self) -> Quantity:
        return (self.checkpoint_segments * self.segment_input).to("GB")

    @property
    def live_segment(self) -> Quantity:
        """One segment's per-token activations, rebuilt in the backward pass (scores not stored)."""
        return (self.per_token_activations / self.workload.layers * self.segment_layers).to("GB")

    @property
    def checkpointed_activations(self) -> Quantity:
        """Segment inputs plus one live rebuilt segment (its input counted twice: conservative)."""
        return (self.segment_inputs + self.live_segment).to("GB")

    def checkpoint_memory_units(self, checkpoints: int) -> float:
        """Activation memory with ``checkpoints`` evenly spaced checkpoints, in layers: k + L/k."""
        from ..physics.memory import calc_checkpoint_memory_units
        return calc_checkpoint_memory_units(self.workload.layers, checkpoints)

    def activations(self, strategy: str) -> Quantity:
        from ..physics.memory import calc_activation_memory
        w = self.workload
        return calc_activation_memory(w.layers, self.sequence, self.micro_batch, w.hidden_dim,
                                      n_heads=w.heads, precision_bytes=self.precision_bytes, strategy=strategy)

    @property
    def per_token_activations(self) -> Quantity:
        """Activations that grow in proportion to sequence length (everything but the attention scores)."""
        return self.activations("selective").to("GB")

    @property
    def score_activations(self) -> Quantity:
        """Saved attention scores, which grow with the square of sequence length."""
        return (self.activations("none") - self.activations("selective")).to("GB")

    # Per-layer line items (added 2026-10-20): the tensors a reader can name.
    def _elements(self, width) -> float:
        return self.micro_batch * self.sequence * width

    @property
    def projection_activations_per_layer(self) -> Quantity:
        """Query, key, value, and output projections: four tensors of the model width."""
        return Q_(4 * self._elements(self.workload.hidden_dim) * self.precision_bytes, "byte").to("MB")

    @property
    def ffn_activations_per_layer(self) -> Quantity:
        """The feed-forward expansion, ``ffn_multiple`` times the model width."""
        return Q_(self._elements(self.ffn_multiple * self.workload.hidden_dim) * self.precision_bytes,
                  "byte").to("MB")

    @property
    def score_activations_per_layer(self) -> Quantity:
        return (self.score_activations / self.workload.layers).to("MB")

    @property
    def line_items_per_layer(self) -> Quantity:
        norm = self.normalization_per_layer if self.normalization_per_layer is not None else Q_(0, "MB")
        return (self.projection_activations_per_layer + self.ffn_activations_per_layer
                + self.score_activations_per_layer + norm).to("MB")

    @property
    def line_items_total(self) -> Quantity:
        return (self.line_items_per_layer * self.workload.layers).to("GB")

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
    """Training ``workload`` on ``n_devices`` x ``hardware`` at one end-to-end utilization (compute-bound).

    Optional (added 2026-10-20): ``tokens_per_step`` turns the run into steps
    (a step costs 6P FLOPs per token); ``hourly_price`` is the rental price of
    one device for an hour. Energy is every device at its TDP for the whole run.
    """
    workload: Workload
    hardware: HardwareNode
    n_devices: int
    efficiency: float
    tokens_per_step: Optional[int] = None
    hourly_price: Optional[Quantity] = None
    # Optional (added 2026-10-20): run on the peak of another number format
    # (a key of hardware.compute.precision_flops, "fp32" or "tf32"); None uses
    # the dense 16-bit peak, hardware.compute.peak_flops.
    peak_precision: Optional[str] = None

    @property
    def peak_rate(self) -> Quantity:
        """Peak rate of one device in ``peak_precision``."""
        if self.peak_precision is None:
            return self.hardware.compute.peak_flops
        return self.hardware.compute.precision_flops[self.peak_precision]

    def _time(self, ops) -> Quantity:
        from ..physics.performance import dTime
        return dTime(total_ops=ops, num_devices=self.n_devices,
                     peak_flops_per_device=self.peak_rate,
                     efficiency_eta=self.efficiency)

    @property
    def training_time(self) -> Quantity:
        return self._time(self.workload.training_ops).to("day")

    @property
    def delivered_rate(self) -> Quantity:
        """FLOP/s the devices deliver together: peak times utilization."""
        return (self.n_devices * self.peak_rate * self.efficiency).to("TFLOP/s")

    @property
    def training_tokens(self) -> Quantity:
        """Tokens the run's FLOPs cover at 6P FLOPs per token."""
        from ..physics.transformer import calc_transformer_training_flops
        per_token = calc_transformer_training_flops(self.workload.parameters, 1)
        return Q_((self.workload.training_ops / per_token).to("").magnitude, "count")

    @property
    def step_operations(self) -> Quantity:
        from ..physics.transformer import calc_transformer_training_flops
        return calc_transformer_training_flops(self.workload.parameters, self.tokens_per_step).to("TFLOP")

    @property
    def step_time(self) -> Quantity:
        return self._time(self.step_operations).to("second")

    @property
    def steps(self) -> float:
        return (self.workload.training_ops / self.step_operations).to("").magnitude

    @property
    def energy(self) -> Quantity:
        """Every device at its TDP for the whole run."""
        return (self.n_devices * self.hardware.tdp * self.training_time).to("kWh")

    @property
    def step_energy(self) -> Quantity:
        return (self.n_devices * self.hardware.tdp * self.step_time).to("joule")

    @property
    def device_hours(self) -> Quantity:
        return (self.n_devices * self.training_time).to("hour")

    @property
    def rental_cost(self) -> Quantity:
        return (self.device_hours * self.hourly_price).to("USD")


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
    gap_key: Optional[str] = None     # None: the gap is original minus new (added 2026-10-20)
    baseline_prefix: Optional[str] = None  # None: no non-neural baselines in the series
    # Optional (added 2026-10-20): keep only models scored on the new set whose
    # original accuracy reaches this floor (a testbed that includes partially
    # trained and low-data checkpoints).
    original_floor: Optional[float] = None

    @property
    def rows(self):
        from ..literature.registry import Literature
        rows = getattr(Literature.Series, self.series_name).entries
        if self.original_floor is None:
            return rows
        return [r for r in rows if r[self.new_key] is not None and r[self.original_key] >= self.original_floor]

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
    speedup: float = 1.0                 # default (added 2026-10-20): the path as it is

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

    # Stage groups and overlap (added 2026-10-20).
    def subtotal(self, *names) -> Quantity:
        """Summed time of the named stages."""
        return sum((self.stages[n] for n in names), Q_(0, "millisecond")).to("millisecond")

    def share(self, *names) -> float:
        """Share of the serial total spent in the named stages."""
        return (self.subtotal(*names) / self.total).to("").magnitude

    def overlapped_share(self, stage: str, *feeders) -> float:
        """Busy share of ``stage`` when the ``feeders`` run concurrently on other hardware.

        With the stages pipelined, one request leaves per max(feeders, stage),
        so ``stage`` is busy stage / max(feeders, stage) of the time.
        """
        return min(1.0, (self.stages[stage] / self.subtotal(*feeders)).to("").magnitude)


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
        return ZeroFailureModel().solve(self.trials, self.confidence, self.target_rate)


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
    # Optional (added 2026-10-20): the cost limit the run's harness enforced.
    cost_limit: Optional[Quantity] = None

    @property
    def rows(self):
        from ..literature.registry import Literature
        return [r for r in getattr(Literature, self.series_name).entries if r["submission"] == self.submission]

    @property
    def result(self):
        from .solvers import RecordedRunCostModel
        limit = None if self.cost_limit is None else float(self.cost_limit.to("USD").magnitude)
        return RecordedRunCostModel().solve(self.rows, limit)
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
    # Optional (added 2026-10-20): raters who label independently of the item,
    # so their observed agreement is the chance agreement and kappa is zero.
    independent_raters: bool = False

    @property
    def chance_agreement(self) -> float:
        from ..physics import calc_chance_agreement
        return calc_chance_agreement(self.class_shares)

    @property
    def agreement(self) -> float:
        """Observed agreement: the stated value, or chance agreement for independent raters."""
        return self.chance_agreement if self.independent_raters else self.observed_agreement

    @property
    def kappa(self) -> float:
        from ..physics import calc_cohens_kappa
        return calc_cohens_kappa(self.agreement, self.chance_agreement)


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
    # Optional (added 2026-10-20): None for a compute-ceiling comparison with no input path.
    encoded_sample: Optional[Quantity] = None    # bytes per sample as stored
    decoded_sample: Optional[Quantity] = None    # bytes per sample after decode
    supply: Any = None                # StorageSubsystem that feeds the accelerators
    training_flops_factor: float = 3.0
    accelerators: int = 1
    worker_rate: Optional[Quantity] = None   # samples per second per loader worker
    host_decode_cap: Optional[Quantity] = None   # samples per second the whole host can decode
    # Optional (added 2026-10-20): loader workers the host gives each accelerator.
    host_workers: Optional[int] = None

    @property
    def host_supply(self) -> Quantity:
        """Samples a second the host's workers for one accelerator decode and augment."""
        return (self.host_workers * self.worker_rate).to("count/second")

    @property
    def busy_fraction(self) -> float:
        """Share of the time the accelerator computes when the host supply is the slower stage."""
        return min(1.0, (self.host_supply / self.sustained_rate).to("").magnitude)

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
    def achieved_flops(self) -> Quantity:
        """Model arithmetic one accelerator sustains (added 2026-10-20); over peak it is the MFU."""
        return (self.sustained_rate * self.training_flops_per_sample / Q_(1, "count")).to("TFLOP/s")

    @property
    def ceiling_gap(self) -> float:
        """Arithmetic ceiling over the sustained rate (added 2026-10-20)."""
        return 1 / self.sustained_fraction

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
    # Optional (added 2026-10-20): the significance level the coefficient is
    # for, when it is not the registry's 0.05 (calc_ks_coefficient(level)).
    significance: Optional[float] = None

    @property
    def scale(self) -> float:
        """The sample-size factor ``sqrt((n_0 + n_t) / (n_0 n_t))`` the coefficient multiplies."""
        return self.critical_value / self.coefficient

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
    bins: Optional[List[str]] = None  # bin names, aligned with the shares (added 2026-10-20)

    def term(self, name: str) -> float:
        """PSI contribution of the bin called ``name``."""
        return self.terms[self.bins.index(name)]

    def shares(self, name: str):
        """(training share, recent share) of the bin called ``name``."""
        i = self.bins.index(name)
        return self.baseline_shares[i], self.current_shares[i]

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
    batch_sizes: List[int] = []          # batch sizes at which the service is reported (added 2026-10-20)

    @property
    def knee(self) -> float:
        from ..physics.serving import calc_batch_knee
        return calc_batch_knee(self.fixed_time, self.per_request_time)

    def throughput_share(self, batch_size) -> float:
        from ..physics.serving import calc_batch_throughput_share
        return calc_batch_throughput_share(batch_size, self.fixed_time, self.per_request_time)

    # The affine service model as a function of batch size (added 2026-10-20).
    def service_time(self, batch_size) -> Quantity:
        """Time to serve one batch: t_0 + B t_req."""
        return (self.fixed_time + batch_size * self.per_request_time).to("ms")

    def throughput(self, batch_size) -> Quantity:
        """Requests per second when batches of ``batch_size`` run back to back."""
        return (batch_size / self.service_time(batch_size)).to("1/second")

    def time_per_request(self, batch_size) -> Quantity:
        """Service time of one batch divided among its requests."""
        return (self.service_time(batch_size) / batch_size).to("ms")

    def fixed_share(self, batch_size) -> float:
        """Share of a batch's service time that does not depend on the batch: t_0 / S(B)."""
        return (self.fixed_time / self.service_time(batch_size)).to("").magnitude


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
    # Optional (added 2026-10-20): a concurrent request count for ``context_fit``.
    requests: Optional[int] = None

    @property
    def result(self):
        from .solvers import KvAdmissionModel
        return KvAdmissionModel().solve(self.workload, self.accelerators, self.hbm_per_accelerator,
                                        self.weight_bytes_per_param, self.reserve, self.context_tokens,
                                        self.kv_bytes_per_elem)

    def context_fit(self, requests: Optional[int] = None) -> float:
        """Tokens of context each of ``requests`` (default: ``self.requests``) requests can hold in the
        budget (added 2026-10-20)."""
        r = self.result
        n = self.requests if requests is None else requests
        return (r.kv_budget / (n * r.kv_per_token)).to("").magnitude


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

    @property
    def prefill_tokens_skipped(self) -> int:
        """Prefix tokens every user after the first need not prefill (added 2026-10-20)."""
        return (self.users - 1) * self.prefix_tokens


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
    draft_time: Optional[Quantity] = None    # all draft steps of one round
    verify_time: Quantity
    # Optional (added 2026-10-20): one draft step; with no ``draft_time`` the
    # round's draft cost is ``draft_length`` such steps.
    draft_step: Optional[Quantity] = None

    @property
    def draft_cost(self) -> Quantity:
        if self.draft_time is not None:
            return self.draft_time.to("ms")
        return (self.draft_length * self.draft_step).to("ms")

    @property
    def expected_tokens(self) -> float:
        from ..physics.serving import calc_speculative_expected_tokens
        return calc_speculative_expected_tokens(self.acceptance, self.draft_length)

    @property
    def round_time(self) -> Quantity:
        return (self.draft_cost + self.verify_time).to("ms")

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
    iteration_time: Optional[Quantity] = None
    # Optional (added 2026-10-20): requests waiting from the start, admitted in
    # order, for the request-level against continuous schedules below.
    waiting_lengths: Optional[List[int]] = None

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

    @property
    def all_lengths(self) -> List[int]:
        return list(self.output_lengths) + list(self.waiting_lengths or [])

    @property
    def all_useful_iterations(self) -> int:
        return sum(self.all_lengths)

    def request_level_spans(self) -> List[tuple]:
        """(start, end) per request when the waiting batch starts only after the whole first batch ends."""
        first = [(0, n) for n in self.output_lengths]
        return first + [(self.max_output, self.max_output + n) for n in (self.waiting_lengths or [])]

    def continuous_spans(self) -> List[tuple]:
        """(start, end) per request when each freed slot takes the next waiting request at once."""
        import heapq
        free = [(n, i) for i, n in enumerate(self.output_lengths)]
        heapq.heapify(free)
        spans = [(0, n) for n in self.output_lengths]
        for n in self.waiting_lengths or []:
            t, slot = heapq.heappop(free)
            spans.append((t, t + n))
            heapq.heappush(free, (t + n, slot))
        return spans

    @property
    def request_level_iterations(self) -> int:
        return max(end for _, end in self.request_level_spans())

    @property
    def continuous_iterations(self) -> int:
        return max(end for _, end in self.continuous_spans())

    @property
    def request_level_utilization(self) -> float:
        return self.all_useful_iterations / (len(self.output_lengths) * self.request_level_iterations)

    @property
    def continuous_utilization(self) -> float:
        return self.all_useful_iterations / (len(self.output_lengths) * self.continuous_iterations)


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
    episode: Optional[Quantity] = None
    control_rate: Optional[Quantity] = None
    per_step_error: float
    recoverability: float = 1.0
    # Optional (added 2026-10-20): a horizon stated in decision steps (an agent's
    # turns), used when there is no episode duration and control rate.
    horizon: Optional[int] = None

    @property
    def horizon_steps(self) -> int:
        if self.horizon is not None:
            return self.horizon
        return int(round((self.episode * self.control_rate).to("").magnitude))

    # Capped bounds (added 2026-10-20): with per-step cost in [0, 1], excess cost
    # never exceeds the horizon, so each bound is min(S, bound).
    @property
    def cloning_capped(self) -> float:
        return min(self.horizon_steps, self.cloning)

    @property
    def aggregation_capped(self) -> float:
        return min(self.horizon_steps, self.aggregation)

    @property
    def capped_gap(self) -> float:
        """Cloning bound over aggregation bound, both capped: S / u below saturation."""
        return self.cloning_capped / self.aggregation_capped

    @property
    def matching_error(self) -> float:
        """Per-step error on reference states whose cloning bound equals the aggregation bound."""
        return self.aggregation_capped / self.horizon_steps ** 2

    @property
    def saturation_horizon(self) -> float:
        """Horizon at which S^2 eps reaches the cap S: 1 / eps."""
        return 1.0 / self.per_step_error

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

    # Optional (added 2026-10-20): the stored width of each value, so the same
    # rank-r product prices a factorized layer (two thin factors replacing the
    # dense matrix, r (d_in + d_out) values) as well as a LoRA update.
    bytes_per_param: Optional[Quantity] = None

    @property
    def compression(self) -> float:
        """Values of the dense matrix over values of the two factors."""
        return self.full_params / self.update_params

    @property
    def full_bytes(self) -> Quantity:
        return (self.full_params * self.bytes_per_param).to("MB")

    @property
    def update_bytes(self) -> Quantity:
        return (self.update_params * self.bytes_per_param).to("MB")


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

    # Optional (added 2026-10-20): the force when the brake command issues, and
    # a counterfactual stall of ``stall`` at the guarded approach with no local
    # trip, priced by the plate's spring model k x.
    stall: Optional[Quantity] = None

    def force_rate(self, speed: Quantity) -> Quantity:
        return (self.site.k_latch * speed).to("N/ms")

    def force_at_brake_onset(self, speed: Quantity) -> Quantity:
        return (self.force_rate(speed) * self.brake_onset(speed)).to("N")

    @property
    def force_at_brake_onset_guarded(self) -> Quantity:
        return self.force_at_brake_onset(self.guarded_speed)

    @property
    def compression_left_guarded(self) -> Quantity:
        return ((self.site.f_latch_limit - self.force_at_brake_onset_guarded) / self.site.k_latch).to("mm")

    @property
    def stall_travel(self) -> Quantity:
        return (self.guarded_speed * self.stall).to("mm")

    @property
    def stall_force(self) -> Quantity:
        from ..physics.robotics.actuation import calc_contact_force
        return calc_contact_force(self.site.k_latch, self.stall_travel).to("N")

    @property
    def stall_over_limit(self) -> float:
        return (self.stall_force / self.site.f_latch_limit).to("").magnitude


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

    # One-pass traffic model (added 2026-10-20): the layer reads its input and
    # weights once and writes its output once. ``activation_bytes_per_value``
    # sizes inputs and outputs; None uses ``bytes_per_weight``.
    activation_bytes_per_value: Optional[Quantity] = None

    @property
    def _act_bytes(self) -> float:
        b = self.activation_bytes_per_value or self.bytes_per_weight
        return b.to("byte").magnitude

    @property
    def flops(self) -> Quantity:
        """2 x batch x in x out: one multiply and one add per weight per example."""
        return Q_(2 * self.batch_size * self.weights, "flop")

    @property
    def input_bytes(self) -> Quantity:
        return Q_(self.batch_size * self.in_features * self._act_bytes, "byte")

    @property
    def output_bytes(self) -> Quantity:
        return Q_(self.batch_size * self.out_features * self._act_bytes, "byte")

    @property
    def traffic_bytes(self) -> Quantity:
        """Input read, weights read, and output written once each."""
        return self.input_bytes + self.weight_bytes + self.output_bytes

    @property
    def traffic_intensity(self) -> float:
        """FLOP per byte of the one-pass traffic."""
        return (self.flops / self.traffic_bytes).to("flop/byte").magnitude

    def batch_for_traffic_intensity(self, target: float) -> float:
        """Batch at which ``traffic_intensity`` reaches ``target`` FLOP/byte.

        Solves 2 B n m = target (B n a + n m w + B m a) for B, with n, m the
        widths and a, w the activation and weight bytes per value.
        """
        n, m = self.in_features, self.out_features
        a, w = self._act_bytes, self.bytes_per_weight.to("byte").magnitude
        return target * n * m * w / (2 * n * m - target * a * (n + m))


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

    # Batch and one-pass traffic model (added 2026-10-20): ``batch_size``
    # examples share one read of the weights; inputs are read and outputs
    # written once. ``weight_bytes_per_value`` sizes the weights when they use a
    # different format from the activations (INT8 weights, FP16 activations).
    batch_size: int = 1
    weight_bytes_per_value: Optional[Quantity] = None

    @property
    def weight_bytes(self) -> Quantity:
        return (self.weights * (self.weight_bytes_per_value or self.bytes_per_value)).to("byte")

    @property
    def activation_bytes(self) -> Quantity:
        return (self.outputs * self.bytes_per_value).to("byte")

    @property
    def flops(self) -> Quantity:
        """2 x batch x MACs: one multiply and one add per weight per position."""
        return Q_(2 * self.batch_size * self.macs, "flop")

    @property
    def batch_input_bytes(self) -> Quantity:
        return (self.batch_size * self.input_values * self.bytes_per_value).to("byte")

    @property
    def batch_output_bytes(self) -> Quantity:
        return (self.batch_size * self.outputs * self.bytes_per_value).to("byte")

    @property
    def traffic_bytes(self) -> Quantity:
        """Batch input read, weights read, and batch output written once each."""
        return self.batch_input_bytes + self.weight_bytes + self.batch_output_bytes

    @property
    def traffic_intensity(self) -> float:
        """FLOP per byte of the one-pass traffic."""
        return (self.flops / self.traffic_bytes).to("flop/byte").magnitude

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
    # Optional (added 2026-10-20): the activations training saves for the
    # backward pass at ``batch_size``, a stated estimate; saved activations
    # grow in proportion to the batch.
    batch_size: Optional[int] = None
    activations: Optional[Quantity] = None
    # Optional (added 2026-10-20): mixed precision keeps a higher-precision
    # master copy of every trained weight (4 B, FP32) beside the 16-bit
    # weights and gradients; None for a single-precision recipe.
    master_bytes_per_parameter: Optional[Quantity] = None
    # Optional (added 2026-10-20): trainable adapter parameters beside a
    # frozen base (low-rank adapters). The base keeps only its weights; each
    # adapter parameter carries weights, gradients, master copy, and
    # optimizer state.
    adapter_parameters: Optional[int] = None
    # Optional (added 2026-10-20): low-rank adapters stated by shape instead of
    # a count: rank r on ``adapted_matrices`` d_out-by-d_in matrices in each of
    # the workload's layers, r (d_in + d_out) parameters per matrix. Used when
    # ``adapter_parameters`` is None.
    adapter_rank: Optional[int] = None
    adapted_matrices: Optional[int] = None
    adapter_in_dim: Optional[int] = None
    adapter_out_dim: Optional[int] = None

    @property
    def adapter_count(self) -> Optional[int]:
        """Trainable adapter parameters, stated or from the adapter shape."""
        if self.adapter_parameters is not None:
            return self.adapter_parameters
        if self.adapter_rank is None:
            return None
        from ..physics.memory import calc_low_rank_update_params
        return self.workload.layers * self.adapted_matrices * calc_low_rank_update_params(
            self.adapter_in_dim, self.adapter_out_dim, self.adapter_rank)

    def activations_at(self, batch_size: int) -> Quantity:
        """Saved activations at another batch size, scaled linearly from ``activations``."""
        return (self.activations * batch_size / self.batch_size).to("byte")

    @property
    def training_total(self) -> Quantity:
        """Weights, gradients, optimizer state, and saved activations."""
        return (self.total + self.activations).to("byte")

    @property
    def training_copies(self) -> float:
        """Training total in units of the weights."""
        return (self.training_total / self.weights).to("").magnitude

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
    def master_weights(self) -> Quantity:
        if self.master_bytes_per_parameter is None:
            return Q_(0, "byte")
        return (self.workload.parameters.to("count").magnitude * self.master_bytes_per_parameter).to("byte")

    @property
    def bytes_per_trained_parameter(self) -> Quantity:
        """Weights, gradients, master copy, and optimizer state for one trained parameter."""
        master = Q_(0, "byte") if self.master_bytes_per_parameter is None else self.master_bytes_per_parameter
        return (2 * self.bytes_per_value + master + self.optimizer_bytes_per_parameter).to("byte")

    @property
    def adapter_state(self) -> Quantity:
        """Everything the adapters hold: weights, gradients, master copy, optimizer state."""
        return (self.adapter_count * self.bytes_per_trained_parameter).to("byte")

    # Adapter state by array (added 2026-10-20): one resident copy of each.
    @property
    def adapter_weights(self) -> Quantity:
        return (self.adapter_count * self.bytes_per_value).to("byte")

    @property
    def adapter_gradients(self) -> Quantity:
        return (self.adapter_count * self.bytes_per_value).to("byte")

    @property
    def adapter_master(self) -> Quantity:
        master = Q_(0, "byte") if self.master_bytes_per_parameter is None else self.master_bytes_per_parameter
        return (self.adapter_count * master).to("byte")

    @property
    def adapter_optimizer_state(self) -> Quantity:
        return (self.adapter_count * self.optimizer_bytes_per_parameter).to("byte")

    @property
    def adapter_fraction(self) -> float:
        return self.adapter_count / self.workload.parameters.to("count").magnitude

    # Adapter run against full fine-tuning (added 2026-10-20), state before activations.
    @property
    def full_state(self) -> Quantity:
        """State if every parameter trained: the same recipe with no adapters."""
        return self.variant(adapter_parameters=None, adapter_rank=None).total

    @property
    def adapter_saved(self) -> Quantity:
        return (self.full_state - self.total).to("byte")

    @property
    def adapter_reduction(self) -> float:
        """Full-state bytes over adapter-run bytes."""
        return (self.full_state / self.total).to("").magnitude

    @property
    def adapter_kept_share(self) -> float:
        return (self.total / self.full_state).to("").magnitude

    @property
    def frozen_share(self) -> float:
        """The frozen base's share of the adapter run's state."""
        return (self.weights / self.total).to("").magnitude

    @property
    def adapter_reduction_ceiling(self) -> float:
        """The reduction as adapters shrink to nothing: bytes per trained over bytes per frozen parameter."""
        return (self.bytes_per_trained_parameter / self.bytes_per_value).to("").magnitude

    @property
    def total(self) -> Quantity:
        if self.adapter_count is not None:
            return (self.weights + self.adapter_state).to("byte")
        return sum(self._state()[1:], self._state()[0]) + self.master_weights

    @property
    def copies(self) -> float:
        """Total state in units of the weights."""
        return (self.total / self.weights).to("").magnitude

    @property
    def fits(self) -> bool:
        return self.hardware is not None and self.total <= self.hardware.memory.capacity

    @property
    def capacity_share(self) -> float:
        """State before activations as a share of ``hardware``'s memory."""
        return (self.total / self.hardware.memory.capacity).to("").magnitude

    @property
    def training_capacity_multiple(self) -> float:
        """State and saved activations in units of ``hardware``'s memory."""
        return (self.training_total / self.hardware.memory.capacity).to("").magnitude

    @property
    def resident_batch_limit(self) -> float:
        """Sequences whose activations fit beside the state in ``hardware``'s memory (added 2026-10-20)."""
        per_sequence = self.activations / self.batch_size
        return ((self.hardware.memory.capacity - self.total) / per_sequence).to("").magnitude

    @property
    def max_resident_batch(self) -> int:
        return _math.floor(self.resident_batch_limit)

    @property
    def checkpoint(self) -> Quantity:
        """A resumable checkpoint: weights, master copy, and optimizer state; gradients are recomputed."""
        return (self.weights + self.master_weights + self.optimizer_state).to("byte")

    def checkpoint_write_time(self, bandwidth: Quantity) -> Quantity:
        """Seconds to write ``checkpoint`` at ``bandwidth``."""
        return (self.checkpoint / bandwidth).to("second")

    @property
    def update_bytes_per_parameter(self) -> Quantity:
        """Bytes one optimizer update moves per parameter: read the gradient, the updated
        weights, and the optimizer state; write the state and the weights back."""
        p = self.workload.parameters.to("count").magnitude
        updated = (self.master_weights if self.master_bytes_per_parameter is not None else self.weights) / p
        return (self.gradients / p + 2 * updated + 2 * self.optimizer_state / p).to("byte")

    @property
    def update_traffic(self) -> Quantity:
        return (self.update_bytes_per_parameter * self.workload.parameters.to("count").magnitude).to("GB")

    @property
    def update_time(self) -> Quantity:
        """The optimizer update at ``hardware``'s memory bandwidth (elementwise, bandwidth-bound)."""
        return (self.update_traffic / self.hardware.memory.bandwidth).to("ms")


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
    ``request_rate``; ``stages`` is the configured ladder of (traffic share, duration).

    Optional (added 2026-10-20): ``label_delay`` is the time from admission to a
    verdict, so ``labeled_by_completion`` counts the candidate tasks whose labels
    exist when the ladder ends (admitted at least that long before the end).
    """
    samples_per_variant: float
    request_rate: Quantity            # requests per hour
    stages: List[Any]                 # [(share, Quantity duration), ...]
    label_delay: Optional[Quantity] = None

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

    @property
    def admitted_by_completion(self) -> float:
        """Candidate tasks admitted over the whole ladder."""
        return sum((share * self.request_rate * d).to("").magnitude for share, d in self.stages)

    @property
    def labeled_by_completion(self) -> float:
        """Candidate tasks whose verdicts have arrived when the last stage ends."""
        cutoff = self.total - self.label_delay
        labeled, t = 0.0, Q_(0, "hour")
        for share, d in self.stages:
            end = t + d
            overlap = max(Q_(0, "hour"), min(end, cutoff) - t)
            labeled += (share * self.request_rate * overlap).to("").magnitude
            t = end
        return labeled


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
    # Optional (added 2026-10-20): the confidence and power the quantiles come
    # from, the lift as a share of the baseline, and the users per arm that
    # traffic actually allows (for the detectable lift).
    confidence: Optional[float] = None
    power: Optional[float] = None
    relative_lift: Optional[float] = None
    available_per_variant: Optional[float] = None

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
    def target_rate(self) -> float:
        return self.baseline_rate + self.lift

    def detectable_lift(self, n_per_variant=None) -> float:
        """Smallest absolute lift ``n_per_variant`` users per arm can detect (default:
        ``available_per_variant``). Added 2026-10-20."""
        from ..physics.statistics import calc_two_proportion_detectable_lift
        n = self.available_per_variant if n_per_variant is None else n_per_variant
        return calc_two_proportion_detectable_lift(self.baseline_rate, n, self.z_alpha, self.z_beta)

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


# --- Network architectures: attention, transformer blocks, KV caches, and recommenders (added 2026-10-20) ---


def _labeled_capacity(hardware: HardwareNode) -> Quantity:
    """The capacity a vendor label prints: binary gigabytes read as "GB" (an
    "80 GB" A100 holds 80 GiB). Shares of a labeled device divide decimal bytes
    by this figure, as the chapters do."""
    return Q_(hardware.memory.capacity.to("GiB").magnitude, "GB")


class ToyAttentionHeadScenario(BaseScenario):
    """One attention head computed by hand: one row of ``queries``, ``keys``, and ``values`` per token."""
    queries: List[List[float]]
    keys: List[List[float]]
    values: List[List[float]]

    @property
    def tokens(self) -> int:
        return len(self.queries)

    @property
    def width(self) -> int:
        return len(self.queries[0])

    @property
    def scores(self) -> List[List[float]]:
        """q_i . k_j / sqrt(d) for every pair."""
        s = _math.sqrt(self.width)
        return [[sum(a * b for a, b in zip(q, k)) / s for k in self.keys] for q in self.queries]

    @property
    def weights(self) -> List[List[float]]:
        """Softmax of each score row."""
        out = []
        for row in self.scores:
            e = [_math.exp(x) for x in row]
            out.append([x / sum(e) for x in e])
        return out

    @property
    def outputs(self) -> List[List[float]]:
        """Each token's weighted mix of the value rows."""
        return [[sum(w * v[c] for w, v in zip(row, self.values)) for c in range(len(self.values[0]))]
                for row in self.weights]

    @property
    def macs_per_product(self) -> int:
        """Multiply-adds of one of the two products (scores, or the mix): S^2 d."""
        return self.tokens ** 2 * self.width


class PackedQkvProjectionScenario(BaseScenario):
    """``tokens`` rows through ``workload``'s packed query-key-value projection, one d x 3d matrix."""
    workload: Workload
    tokens: int
    projections: int = 3             # query, key, value

    @property
    def width(self) -> int:
        return self.workload.hidden_dim

    @property
    def packed_width(self) -> int:
        return self.projections * self.width

    @property
    def macs(self) -> int:
        return self.tokens * self.width * self.packed_width


class TransformerBlockScenario(BaseScenario):
    """``workload``'s decoder blocks at a context of ``context_tokens``: the classic block's weights
    (4 d^2 attention projections, 2 x ``ffn_multiple`` d^2 MLP), its dense and attention multiply-adds,
    and, with ``hardware``, the attention arithmetic of one prompt at peak."""
    workload: Workload
    context_tokens: int
    ffn_multiple: int = 4            # feed-forward width over model width
    attention_projections: int = 4   # query, key, value, output
    hardware: Optional[HardwareNode] = None
    # Optional (added 2026-10-20): sequences of ``context_tokens`` in one training step.
    batch_size: int = 1

    def _w(self):
        from ..physics.transformer import calc_transformer_block_weights
        return calc_transformer_block_weights(self.width, self.ffn_multiple, self.attention_projections)

    @property
    def width(self) -> int:
        return self.workload.hidden_dim

    @property
    def attention_weights_per_block(self) -> int:
        return self._w()[0]

    @property
    def mlp_weights_per_block(self) -> int:
        return self._w()[1]

    @property
    def weights_per_block(self) -> int:
        return self._w()[2]

    @property
    def block_coefficient(self) -> int:
        """Weights per block in units of d^2 (12 for the classic block)."""
        return self.weights_per_block // self.width ** 2

    @property
    def mlp_share(self) -> float:
        return self.mlp_weights_per_block / self.weights_per_block

    @property
    def block_weights(self) -> int:
        """All blocks' weights."""
        return self.weights_per_block * self.workload.layers

    @property
    def attention_weights(self) -> int:
        return self.attention_weights_per_block * self.workload.layers

    @property
    def mlp_weights(self) -> int:
        return self.mlp_weights_per_block * self.workload.layers

    @property
    def token_table(self) -> int:
        """Token-embedding weights: one row of width d per vocabulary entry."""
        return self.workload.vocab_size * self.width

    @property
    def block_rule_parameters(self) -> int:
        return self.block_weights + self.token_table

    @property
    def exact_parameters(self) -> int:
        """GPT-2 layout counted in full: projection and MLP biases, two LayerNorms per block,
        the final LayerNorm, and a position table of ``context_length`` rows (nanoGPT, bias=True)."""
        d, f = self.width, self.ffn_multiple
        per_block = (d * 3 * d + 3 * d) + (d * d + d) + (d * f * d + f * d) + (f * d * d + d) + 2 * (2 * d)
        return (self.workload.layers * per_block + self.workload.vocab_size * d
                + self.workload.context_length * d + 2 * d)

    @property
    def flops_per_token(self) -> Quantity:
        """Two FLOPs (a multiply and an add) per block weight per token."""
        return Q_(2 * self.block_weights, "flop")

    @property
    def output_layer_flops(self) -> Quantity:
        """The output projection onto the vocabulary, per token."""
        return Q_(2 * self.token_table, "flop")

    @property
    def dense_macs(self) -> int:
        """One block's projection and MLP multiply-adds over the context."""
        return self.weights_per_block * self.context_tokens

    @property
    def attention_macs(self) -> int:
        """One block's two attention products over the context."""
        from ..physics.transformer import calc_attention_score_macs
        return calc_attention_score_macs(self.context_tokens, self.width)

    @property
    def block_macs(self) -> int:
        return self.dense_macs + self.attention_macs

    @property
    def attention_share(self) -> float:
        return self.attention_macs / self.block_macs

    @property
    def dense_to_attention(self) -> float:
        return self.dense_macs / self.attention_macs

    @property
    def attention_to_dense(self) -> float:
        return self.attention_macs / self.dense_macs

    @property
    def crossover(self) -> float:
        """Context length at which the attention products equal the dense products (6 d)."""
        from ..physics.transformer import calc_attention_crossover
        return calc_attention_crossover(self.width, self.ffn_multiple, self.attention_projections)

    @property
    def crossover_multiple(self) -> float:
        """The crossover in units of the width (six for the classic block)."""
        return self.crossover / self.width

    @property
    def prompt_dense_flops(self) -> Quantity:
        return (self.flops_per_token * self.context_tokens).to("flop")

    @property
    def prompt_attention_flops(self) -> Quantity:
        """Every block's attention products over the prompt, two FLOPs per multiply-add."""
        return Q_(2 * self.attention_macs * self.workload.layers, "flop")

    @property
    def prompt_flops(self) -> Quantity:
        return self.prompt_dense_flops + self.prompt_attention_flops

    @property
    def attention_time_at_peak(self) -> Quantity:
        return (self.prompt_attention_flops / self.hardware.compute.peak_flops).to("minute")

    @property
    def training_to_forward(self) -> float:
        """A training step in forward passes: 6P over 2P per token."""
        from ..physics.transformer import calc_transformer_training_flops, calc_transformer_decode_flops
        p = self.workload.parameters
        return (calc_transformer_training_flops(p, 1) / calc_transformer_decode_flops(p, 1)).to("").magnitude

    @property
    def layer_forward_flops(self) -> Quantity:
        """One block's forward FLOPs for the whole batch: dense and attention products, two FLOPs each."""
        return Q_(2 * self.block_macs * self.batch_size, "flop").to("GFLOP")

    @property
    def training_step_flops(self) -> Quantity:
        """One training step counted layer by layer, attention products included."""
        return (self.layer_forward_flops * self.workload.layers * self.training_to_forward).to("TFLOP")

    @property
    def six_p_step_flops(self) -> Quantity:
        """The same step by the 6P-per-token rule, which omits the attention products."""
        from ..physics.transformer import calc_transformer_training_flops
        return calc_transformer_training_flops(self.workload.parameters,
                                               self.batch_size * self.context_tokens).to("TFLOP")

    @property
    def attention_step_flops(self) -> Quantity:
        """The attention products' share of a training step, all layers."""
        return Q_(2 * self.attention_macs * self.batch_size * self.workload.layers
                  * self.training_to_forward, "flop").to("TFLOP")


class MaterializedScoresScenario(BaseScenario):
    """Attention scores stored in full: ``seq_len``^2 per head, ``heads`` per layer, ``layers`` layers."""
    seq_len: int
    heads: int
    layers: int
    bytes_per_score: Quantity
    hardware: Optional[HardwareNode] = None

    @property
    def scores_per_layer(self) -> int:
        return self.seq_len ** 2 * self.heads

    @property
    def layer_bytes(self) -> Quantity:
        return (self.scores_per_layer * self.bytes_per_score).to("GB")

    @property
    def total_bytes(self) -> Quantity:
        return (self.layer_bytes * self.layers).to("GB")

    @property
    def layer_device_multiple(self) -> float:
        """One layer's scores in units of ``hardware``'s labeled capacity."""
        return (self.layer_bytes / _labeled_capacity(self.hardware)).to("").magnitude


class ContextGrowthScenario(BaseScenario):
    """A context window grown from ``base_tokens`` to ``tokens``: a kept cache grows with the ratio,
    attention's score work and stored scores with its square."""
    base_tokens: int
    tokens: int

    @property
    def ratio(self) -> float:
        return self.tokens / self.base_tokens

    @property
    def score_ratio(self) -> float:
        return self.ratio ** 2


class KvFootprintScenario(BaseScenario):
    """``workload``'s KV cache for ``tokens`` kept tokens; with ``hardware`` and ``kv_share``, how many
    such requests the cache's share of the labeled device memory holds. ``kv_width_factor`` scales the
    key-value width (one half: half the key-value heads)."""
    workload: Workload
    tokens: int
    bytes_per_value: Quantity
    hardware: Optional[HardwareNode] = None
    kv_share: Optional[float] = None
    kv_width_factor: float = 1.0

    @property
    def per_token(self) -> Quantity:
        return (self.workload.get_kv_cache_size(1, 1, self.bytes_per_value) * self.kv_width_factor).to("byte")

    @property
    def total(self) -> Quantity:
        return (self.per_token * self.tokens).to("GB")

    @property
    def weights(self) -> Quantity:
        return self.workload.size_in_bytes(self.bytes_per_value).to("GB")

    @property
    def kv_budget(self) -> Quantity:
        return (_labeled_capacity(self.hardware) * self.kv_share).to("GB")

    @property
    def tokens_in_budget(self) -> float:
        return (self.kv_budget / self.per_token).to("").magnitude

    @property
    def requests(self) -> float:
        return (self.kv_budget / self.total).to("").magnitude


class ContextMemoryCrossingScenario(BaseScenario):
    """Three memories of ``workload`` as context grows on ``hardware``: fixed weights, a KV cache linear
    in tokens, and one layer's stored scores (all heads) quadratic in tokens."""
    workload: Workload
    hardware: HardwareNode
    bytes_per_value: Quantity

    @property
    def weights(self) -> Quantity:
        return self.workload.size_in_bytes(self.bytes_per_value).to("GB")

    @property
    def kv_per_token(self) -> Quantity:
        return self.workload.get_kv_cache_size(1, 1, self.bytes_per_value).to("byte")

    @property
    def score_bytes_per_pair(self) -> Quantity:
        """One layer's stored scores for one token pair, all heads."""
        return (self.workload.heads * self.bytes_per_value).to("byte")

    @property
    def kv_crossing(self) -> float:
        """Tokens at which the KV cache equals the weights."""
        return (self.weights / self.kv_per_token).to("").magnitude

    @property
    def score_crossing(self) -> float:
        """Tokens at which one layer's stored scores fill the device."""
        return _math.sqrt((self.hardware.memory.capacity / self.score_bytes_per_pair).to("").magnitude)

    @property
    def kv_score_crossing(self) -> float:
        """Tokens at which the KV cache and one layer's scores are equal."""
        return (self.kv_per_token / self.score_bytes_per_pair).to("").magnitude


class EmbeddingTableScenario(BaseScenario):
    """``tables`` embedding tables of ``rows`` rows, each ``width`` values wide, against ``hardware``."""
    rows: int
    width: int
    bytes_per_value: Quantity
    hardware: HardwareNode
    tables: int = 1

    @property
    def row_bytes(self) -> Quantity:
        return (self.width * self.bytes_per_value).to("byte")

    @property
    def table_bytes(self) -> Quantity:
        return (self.rows * self.row_bytes).to("GB")

    @property
    def total_bytes(self) -> Quantity:
        return (self.table_bytes * self.tables).to("GB")

    @property
    def labeled_share(self) -> float:
        """All tables over the device's labeled capacity."""
        return (self.total_bytes / _labeled_capacity(self.hardware)).to("").magnitude

    @property
    def fits(self) -> bool:
        return self.labeled_share <= 1


class EmbeddingGatherScenario(BaseScenario):
    """One recommender request on ``hardware``: one row of ``width`` values per categorical feature of
    ``dataset`` is gathered, then the rows and the bottom MLP's output interact by pairwise dot products."""
    dataset: DatasetProfile
    width: int
    bytes_per_value: Quantity
    hardware: HardwareNode

    @property
    def features(self) -> int:
        return self.dataset.categorical_features

    @property
    def row_bytes(self) -> Quantity:
        return (self.width * self.bytes_per_value).to("byte")

    @property
    def request_bytes(self) -> Quantity:
        return (self.features * self.row_bytes).to("byte")

    @property
    def vectors(self) -> int:
        """The gathered rows and the bottom MLP's output."""
        return self.features + 1

    @property
    def pairs(self) -> int:
        return self.vectors * (self.vectors - 1) // 2

    @property
    def interaction_macs(self) -> int:
        return self.pairs * self.width

    @property
    def intensity(self) -> Quantity:
        """Interaction FLOPs (two per multiply-add) per gathered byte."""
        return (Q_(2 * self.interaction_macs, "flop") / self.request_bytes).to("flop/byte")

    @property
    def device_rate(self) -> Quantity:
        """Requests a second if every row streamed at device-memory bandwidth (an upper bound)."""
        return (self.hardware.memory.bandwidth / self.request_bytes).to("1/s")

    @property
    def host_rate(self) -> Quantity:
        """Requests a second if every row crossed the host link (an upper bound)."""
        return (self.hardware.interconnect.bandwidth / self.request_bytes).to("1/s")

    @property
    def bandwidth_ratio(self) -> float:
        return (self.device_rate / self.host_rate).to("").magnitude
# --- Framework execution: dispatch, fusion, compilation payback, one training step (added 2026-10-20) ---

class HostDispatchScenario(BaseScenario):
    """The host's work to issue one eager step against ``device_time`` of accelerator work.

    The step has ``blocks`` blocks of ``ops_per_block`` Python-level
    operations, each paying ``dispatch``, and ``launches_per_block`` kernels
    per block (default: one per operation), each paying ``launch``. The model
    is additive with no host-device overlap: the overhead ratio is host time
    over device time, and above 1 the host takes more than half of the step.
    """
    ops_per_block: int
    blocks: int = 1
    launches_per_block: Optional[int] = None
    dispatch: Quantity
    launch: Quantity
    device_time: Quantity

    @property
    def operations(self) -> int:
        return self.blocks * self.ops_per_block

    @property
    def launches(self) -> int:
        per = self.ops_per_block if self.launches_per_block is None else self.launches_per_block
        return self.blocks * per

    @property
    def per_op_cost(self) -> Quantity:
        """Dispatch plus launch, for an operation that launches one kernel."""
        return (self.dispatch + self.launch).to("microsecond")

    @property
    def dispatch_time(self) -> Quantity:
        return (self.operations * self.dispatch).to("microsecond")

    @property
    def launch_time(self) -> Quantity:
        return (self.launches * self.launch).to("microsecond")

    @property
    def host_time(self) -> Quantity:
        return (self.dispatch_time + self.launch_time).to("microsecond")

    @property
    def overhead_ratio(self) -> float:
        """Host time over device time (the dispatch overhead ratio)."""
        return (self.host_time / self.device_time).to("").magnitude

    @property
    def launch_ratio(self) -> float:
        """Kernel launches alone over device time."""
        return (self.launch_time / self.device_time).to("").magnitude

    @property
    def host_share(self) -> float:
        """Host time as a share of the serial step, host plus device."""
        return (self.host_time / (self.host_time + self.device_time)).to("").magnitude

    @property
    def speedup_limit(self) -> float:
        """Serial step over device time: the most removing all host work could gain."""
        return ((self.host_time + self.device_time) / self.device_time).to("").magnitude


class KernelFusionScenario(BaseScenario):
    """A chain of ``chain_length`` element-wise operations, run as one kernel each or fused into one.

    Unfused, every operation reads its input from device memory and writes its
    output back (two trips each) and pays ``dispatch`` plus ``launch``; fused,
    the chain reads once, keeps intermediates on chip, writes once, and pays
    one launch. For N elements of b bytes the traffic is 2kNb against 2Nb.
    """
    chain_length: int
    dispatch: Quantity
    launch: Quantity

    @property
    def unfused_trips(self) -> int:
        return 2 * self.chain_length

    @property
    def fused_trips(self) -> int:
        return 2

    @property
    def unfused_launches(self) -> int:
        return self.chain_length

    @property
    def fused_launches(self) -> int:
        return 1

    @property
    def traffic_ratio(self) -> float:
        return self.unfused_trips / self.fused_trips

    @property
    def unfused_overhead(self) -> Quantity:
        return (self.unfused_launches * (self.dispatch + self.launch)).to("microsecond")

    @property
    def fused_overhead(self) -> Quantity:
        return (self.fused_launches * (self.dispatch + self.launch)).to("microsecond")

    @property
    def overhead_speedup(self) -> float:
        return (self.unfused_overhead / self.fused_overhead).to("").magnitude


class CompileBreakevenScenario(BaseScenario):
    """Eager against compiled execution: the work that repays ``compile_time``.

    Compiling pays when the time saved per item, summed over items, exceeds
    the compile time. An experiment of ``experiment_items`` items that changes
    the code ``recompiles`` times pays a compile for every change.
    """
    eager_rate: Quantity                 # items per second
    compiled_rate: Quantity              # items per second
    compile_time: Quantity
    experiment_items: Optional[int] = None
    recompiles: Optional[int] = None

    @property
    def eager_item_time(self) -> Quantity:
        return (1 / self.eager_rate).to("ms")

    @property
    def compiled_item_time(self) -> Quantity:
        return (1 / self.compiled_rate).to("ms")

    @property
    def saving_per_item(self) -> Quantity:
        return (self.eager_item_time - self.compiled_item_time).to("ms")

    @property
    def breakeven_items(self) -> float:
        return (self.compile_time / self.saving_per_item).to("").magnitude

    @property
    def speedup(self) -> float:
        return (self.compiled_rate / self.eager_rate).to("").magnitude

    @property
    def experiment_eager(self) -> Quantity:
        return (self.experiment_items / self.eager_rate).to("second")

    @property
    def experiment_compiled_running(self) -> Quantity:
        return (self.experiment_items / self.compiled_rate).to("second")

    @property
    def experiment_compiled_total(self) -> Quantity:
        return (self.recompiles * self.compile_time + self.experiment_compiled_running).to("second")

    @property
    def experiment_slowdown(self) -> float:
        return (self.experiment_compiled_total / self.experiment_eager).to("").magnitude


class JacobianScenario(BaseScenario):
    """The full Jacobian of one layer from ``input_width`` inputs to ``output_width`` outputs.

    Reverse-mode differentiation never builds it; it chains vector-Jacobian products.
    """
    input_width: int
    output_width: int
    bytes_per_value: Quantity

    @property
    def elements(self) -> int:
        return self.input_width * self.output_width

    @property
    def size(self) -> Quantity:
        return (self.elements * self.bytes_per_value).to("byte")


class HostInputPipelineScenario(BaseScenario):
    """A data loader feeding one accelerator over its host link.

    ``workers`` loader processes each prepare ``prefetch_factor`` batches of
    ``batch_size`` samples. A sample is ``raw_sample`` bytes as decoded pixels
    and ``model_sample`` bytes as the tensor copied to the device; the
    accelerator consumes ``consume_rate`` samples a second. Copy times are
    bandwidth-only lower bounds; ``step_times`` are compute times a serial
    copy is compared with.
    """
    batch_size: int
    workers: int
    prefetch_factor: int
    raw_sample: Quantity
    model_sample: Quantity
    consume_rate: Quantity               # samples per second
    host_link: Quantity                  # bandwidth
    device_memory: Quantity              # bandwidth
    step_times: Dict[str, Quantity] = {}

    @property
    def batches_in_flight(self) -> int:
        return self.workers * self.prefetch_factor

    @property
    def raw_feed(self) -> Quantity:
        return (self.consume_rate * self.raw_sample).to("MB/second")

    @property
    def batch_bytes(self) -> Quantity:
        return (self.batch_size * self.model_sample).to("byte")

    @property
    def copy_time(self) -> Quantity:
        from ..physics import transfer_time
        return transfer_time(self.batch_bytes, self.host_link).to("ms")

    @property
    def device_copy_time(self) -> Quantity:
        from ..physics import transfer_time
        return transfer_time(self.batch_bytes, self.device_memory).to("microsecond")

    def copy_share(self, step: str) -> float:
        """A serial copy as a share of one named step's compute time."""
        return (self.copy_time / self.step_times[step]).to("").magnitude


_MLP_FORWARD_OPS = ("matmul1", "relu", "matmul2", "loss")
_MLP_BACKWARD_OPS = ("grad_w2", "grad_h", "grad_w1", "relu_backward", "loss_backward")


class MlpTrainingStepScenario(BaseScenario):
    """One eager training step of a one-hidden-layer MLP, operator by operator.

    Forward: ``x @ W1``, ReLU, ``h @ W2``, and a combined cross-entropy loss
    (``loss_flops_per_logit`` operations per logit). Backward: the weight
    gradients of both layers and the hidden-activation gradient (raw inputs
    need no gradient), the ReLU derivative, and the loss derivative, which
    reads saved probabilities and integer labels of ``label_bytes`` each.
    Bias adds are folded into their matrix multiplies. Every value is
    ``bytes_per_value`` wide. Times are ideal lower bounds on ``hardware``;
    the host issues one Python entry per forward operator plus one for
    ``backward()``, and launches one kernel per operator.
    """
    batch_size: int
    input_width: int
    hidden_width: int
    output_width: int
    bytes_per_value: Quantity
    label_bytes: Quantity
    loss_flops_per_logit: int
    optimizer_values_moved: int          # values read and written per parameter by one optimizer step
    hardware: HardwareNode
    precision: str = "fp32"
    forward_ops: int
    backward_ops: int
    dispatch: Quantity
    launch: Quantity

    def _values(self, n: int) -> Quantity:
        return (n * self.bytes_per_value).to("byte")

    @property
    def op_flops(self) -> Dict[str, Quantity]:
        b, i, h, o = self.batch_size, self.input_width, self.hidden_width, self.output_width
        f = {
            "matmul1": 2 * b * i * h, "relu": b * h, "matmul2": 2 * b * h * o,
            "loss": self.loss_flops_per_logit * b * o,
            "grad_w2": 2 * h * b * o, "grad_h": 2 * b * o * h, "grad_w1": 2 * i * b * h,
            "relu_backward": b * h, "loss_backward": 2 * b * o,
        }
        return {k: Q_(v, "flop") for k, v in f.items()}

    @property
    def op_bytes(self) -> Dict[str, Quantity]:
        b, i, h, o = self.batch_size, self.input_width, self.hidden_width, self.output_width
        v = self._values
        return {
            "matmul1": v(b * i + i * h + b * h), "relu": v(2 * b * h),
            "matmul2": v(b * h + h * o + b * o), "loss": v(2 * b * o),
            "grad_w2": v(b * h + b * o + h * o), "grad_h": v(b * o + h * o + b * h),
            "grad_w1": v(b * i + b * h + i * h), "relu_backward": v(3 * b * h),
            "loss_backward": (v(2 * b * o) + b * self.label_bytes).to("byte"),
        }

    def _sum(self, table, keys) -> Quantity:
        return sum((table[k] for k in keys[1:]), table[keys[0]])

    @property
    def forward_flops(self) -> Quantity:
        return self._sum(self.op_flops, _MLP_FORWARD_OPS)

    @property
    def backward_flops(self) -> Quantity:
        return self._sum(self.op_flops, _MLP_BACKWARD_OPS)

    @property
    def forward_bytes(self) -> Quantity:
        return self._sum(self.op_bytes, _MLP_FORWARD_OPS)

    @property
    def backward_bytes(self) -> Quantity:
        return self._sum(self.op_bytes, _MLP_BACKWARD_OPS)

    @property
    def total_flops(self) -> Quantity:
        return self.forward_flops + self.backward_flops

    @property
    def total_bytes(self) -> Quantity:
        return self.forward_bytes + self.backward_bytes

    @property
    def dense_training_ratio(self) -> float:
        """Matrix-multiply work of the whole step over the forward matrix multiplies."""
        f = self.op_flops
        fwd = f["matmul1"] + f["matmul2"]
        return ((fwd + f["grad_w2"] + f["grad_h"] + f["grad_w1"]) / fwd).to("").magnitude

    @property
    def parameters(self) -> int:
        i, h, o = self.input_width, self.hidden_width, self.output_width
        return i * h + h + h * o + o

    @property
    def optimizer_step_bytes(self) -> Quantity:
        return (self.parameters * self.optimizer_values_moved * self.bytes_per_value).to("byte")

    @property
    def peak_rate(self) -> Quantity:
        return self.hardware.compute.precision_flops[self.precision]

    @property
    def compute_time(self) -> Quantity:
        return (self.total_flops / self.peak_rate).to("microsecond")

    @property
    def memory_time(self) -> Quantity:
        return (self.total_bytes / self.hardware.memory.bandwidth).to("microsecond")

    @property
    def host(self) -> HostDispatchScenario:
        """The host's dispatch and launch work against the step's ideal device time."""
        return HostDispatchScenario(
            name=f"Host work of: {self.name}", description="Python entries and kernel launches of the step.",
            ops_per_block=self.forward_ops + 1, launches_per_block=self.forward_ops + self.backward_ops,
            dispatch=self.dispatch, launch=self.launch, device_time=self.compute_time + self.memory_time)


class DeploymentPathScenario(BaseScenario):
    """Candidate deployment paths measured against a latency and a memory budget.

    ``paths`` maps each path to its measured ``latency`` and ``memory``; a
    path passes when both are within budget.
    """
    latency_budget: Quantity
    memory_budget: Quantity
    paths: Dict[str, Dict[str, Quantity]]

    def passes(self, path: str) -> bool:
        p = self.paths[path]
        return p["latency"] <= self.latency_budget and p["memory"] <= self.memory_budget

    @property
    def passing(self) -> List[str]:
        return [k for k in self.paths if self.passes(k)]
# --- Security and privacy of a fleet (added 2026-10-20) ---------------------

class QueryExtractionBudgetScenario(BaseScenario):
    """An extractor that needs ``extraction_queries`` answers from a priced API, under three
    billing regimes: pay per query, a paid tier capped per day, and a lifetime per-principal
    allowance after which every query is billed at the per-query price."""
    training_cost: Quantity          # USD to train the victim model
    extraction_queries: float        # answers for the target fidelity
    target_fidelity: float           # agreement with the victim the extractor aims for
    price_per_query: Quantity        # USD, pay-as-you-go
    tier_price: Quantity             # USD per billing period
    tier_daily_quota: float          # queries per day on the paid tier
    billing_period: Quantity         # days per tier billing period
    lifetime_allowance: float        # queries per principal before overage billing

    def _tier_fee_per_day(self) -> Quantity:
        return (self.tier_price / self.billing_period.to("day").magnitude).to("USD")

    def tier_days(self, queries) -> float:
        return queries / self.tier_daily_quota

    def pay_per_query_cost(self, queries) -> Quantity:
        return (queries * self.price_per_query).to("USD")

    def tier_cost(self, queries) -> Quantity:
        """Daily quota only: the tier fee for every day of collection."""
        return (self.tier_days(queries) * self._tier_fee_per_day()).to("USD")

    def lifetime_cost(self, queries) -> Quantity:
        """Tier fees up to the allowance, then every query billed per query."""
        within = min(queries, self.lifetime_allowance)
        overage = max(queries - self.lifetime_allowance, 0)
        return (self.tier_cost(within) + self.pay_per_query_cost(overage)).to("USD")

    @property
    def collection_days(self) -> float:
        return self.tier_days(self.extraction_queries)

    @property
    def full_set_pay_per_query(self) -> Quantity:
        return self.pay_per_query_cost(self.extraction_queries)

    @property
    def full_set_tier(self) -> Quantity:
        return self.tier_cost(self.extraction_queries)

    @property
    def allowance_fees(self) -> Quantity:
        return self.tier_cost(self.lifetime_allowance)

    @property
    def overage_cost(self) -> Quantity:
        return self.pay_per_query_cost(max(self.extraction_queries - self.lifetime_allowance, 0))

    @property
    def full_set_lifetime(self) -> Quantity:
        return self.lifetime_cost(self.extraction_queries)

    @property
    def tier_share_of_training(self) -> float:
        return (self.full_set_tier / self.training_cost).to("").magnitude


class OutputShapingScenario(BaseScenario):
    """An API that returns the top ``top_k`` of ``class_count`` scores, rounded from
    ``from_decimals`` to ``to_decimals`` decimal places."""
    class_count: int
    top_k: int
    from_decimals: int
    to_decimals: int

    @property
    def scores_reduction(self) -> float:
        """Times fewer scores per answer than returning every class."""
        return self.class_count / self.top_k

    @property
    def entries_removed(self) -> float:
        return 1 - self.top_k / self.class_count

    @property
    def decimal_digits_removed(self) -> int:
        return self.from_decimals - self.to_decimals

    @property
    def bits_removed(self) -> float:
        """Bits per score removed under a decimal-bin count."""
        from ..physics.privacy import calc_level_bits
        return calc_level_bits(10 ** self.decimal_digits_removed)

    @property
    def retained_bits(self) -> float:
        """Bits left in a score in [0, 1] at ``to_decimals`` places (10^d + 1 levels)."""
        from ..physics.privacy import calc_level_bits
        return calc_level_bits(10 ** self.to_decimals + 1)


class DataParallelAttackSurfaceScenario(BaseScenario):
    """``workload`` trained on ``fleet`` with ``dp_degree`` data-parallel replicas: the gradient
    each step exchanges, the checkpoint it writes, and the traffic of a robust aggregate that
    gathers every replica's gradient instead of a ring AllReduce of their sum."""
    workload: Workload
    fleet: Fleet
    dp_degree: int
    gradient_bytes_per_param: Quantity
    checkpoint_bytes_per_param: Quantity
    weight_bytes_per_param: Optional[Quantity] = None   # serving weights, when quoted

    @property
    def accelerators(self) -> int:
        return self.fleet.total_accelerators

    @property
    def nodes(self) -> int:
        return self.fleet.count

    @property
    def accelerators_per_node(self) -> int:
        return self.fleet.node.accelerators_per_node

    @property
    def accelerators_per_replica(self) -> float:
        return self.accelerators / self.dp_degree

    @property
    def gradient_bytes(self) -> Quantity:
        from ..physics.quantities import memory_from_params
        return memory_from_params(self.workload.parameters, self.gradient_bytes_per_param).to("byte")

    @property
    def checkpoint_bytes(self) -> Quantity:
        from ..physics.quantities import memory_from_params
        return memory_from_params(self.workload.parameters, self.checkpoint_bytes_per_param).to("byte")

    @property
    def weight_bytes(self) -> Quantity:
        from ..physics.quantities import memory_from_params
        return memory_from_params(self.workload.parameters, self.weight_bytes_per_param).to("byte")

    @property
    def allreduce_factor(self) -> float:
        """Bytes each worker moves in a ring AllReduce, per byte of gradient."""
        from ..physics.communication import calc_ring_allreduce_data_factor
        return calc_ring_allreduce_data_factor(self.dp_degree)

    @property
    def gather_factor(self) -> float:
        """Bytes each worker receives when it gathers every other replica's gradient."""
        return self.dp_degree - 1

    @property
    def gather_traffic_ratio(self) -> float:
        """Gather over ring AllReduce traffic; equals ``dp_degree / 2``."""
        return self.gather_factor / self.allreduce_factor


class MeanBreakdownScenario(BaseScenario):
    """``workers`` contributions, all honest ones equal to ``honest_value`` and one chosen by an
    attacker: the mean follows the attacker, the median does not."""
    workers: int
    honest_value: float
    attacker_value: float

    def mean(self, attacker_value=None) -> float:
        v = self.attacker_value if attacker_value is None else attacker_value
        return ((self.workers - 1) * self.honest_value + v) / self.workers

    def median(self, attacker_value=None) -> float:
        import statistics
        v = self.attacker_value if attacker_value is None else attacker_value
        return float(statistics.median([self.honest_value] * (self.workers - 1) + [v]))


class ThroughputCostScenario(BaseScenario):
    """The same workload and resource budget served at ``baseline_rate`` without a set of
    controls and at ``controlled_rate`` with them."""
    baseline_rate: Quantity
    controlled_rate: Quantity

    @property
    def rate_lost(self) -> Quantity:
        return (self.baseline_rate - self.controlled_rate).to(self.baseline_rate.units)

    @property
    def overhead(self) -> float:
        return (1 - self.controlled_rate / self.baseline_rate).to("").magnitude


class ProtectedMemoryFitScenario(BaseScenario):
    """``workload``'s weights at ``bytes_per_weight`` against ``hardware``'s protected memory
    (an enclave page cache), before activations."""
    workload: Workload
    hardware: HardwareNode
    bytes_per_weight: Quantity

    @property
    def weight_bytes(self) -> Quantity:
        from ..physics.quantities import memory_from_params
        return memory_from_params(self.workload.parameters, self.bytes_per_weight / Q_(1, "param")).to("byte")

    @property
    def capacity(self) -> Quantity:
        return self.hardware.memory.capacity

    @property
    def capacity_share(self) -> float:
        return (self.weight_bytes / self.capacity).to("").magnitude


class EncryptedInferenceLatencyScenario(BaseScenario):
    """One inference at ``plaintext_latency``, with symmetric encryption around it
    (``encryption_overhead``) and computed under homomorphic encryption (``fhe_multiplier``)."""
    plaintext_latency: Quantity
    encryption_overhead: Quantity
    fhe_multiplier: float

    @property
    def encrypted_transport_latency(self) -> Quantity:
        return (self.plaintext_latency + self.encryption_overhead).to("ms")

    @property
    def fhe_latency(self) -> Quantity:
        return (self.plaintext_latency * self.fhe_multiplier).to("second")


class ByteWiseSearchScenario(BaseScenario):
    """Guessing a ``secret_bytes``-byte secret whole, against one byte at a time when a side
    channel reveals how many leading bytes are correct."""
    secret_bytes: int
    bits_per_symbol: int

    @property
    def symbols(self) -> int:
        return 2 ** self.bits_per_symbol

    @property
    def brute_force_guesses(self) -> int:
        return self.symbols ** self.secret_bytes

    @property
    def bytewise_guesses(self) -> int:
        return self.secret_bytes * self.symbols


class LaplaceMeanReleaseScenario(BaseScenario):
    """The mean of a public roster of ``group_size`` values in ``[0, value_range]`` released
    under ``epsilon``-DP by one Laplace draw on the sum (replace-one adjacency)."""
    value_range: Quantity
    epsilon: float
    group_size: int
    small_group_size: Optional[int] = None

    @property
    def sensitivity(self) -> Quantity:
        """The most one person can move the sum."""
        return self.value_range

    @property
    def noise_scale(self) -> Quantity:
        from ..physics.privacy import calc_laplace_scale
        return calc_laplace_scale(self.sensitivity, self.epsilon)

    def mean_error(self, group_size=None) -> Quantity:
        """Expected absolute error of the released mean (the Laplace scale over the group)."""
        n = self.group_size if group_size is None else group_size
        return self.noise_scale / n

    @property
    def small_group_error(self) -> Quantity:
        return self.mean_error(self.small_group_size)


class BasicCompositionScenario(BaseScenario):
    """``releases`` data-dependent releases about one population, each at ``epsilon_per_release``."""
    releases: int
    epsilon_per_release: float

    @property
    def total_epsilon(self) -> float:
        from ..physics.privacy import calc_basic_composition
        return calc_basic_composition([self.epsilon_per_release] * self.releases)


class PrivacyAccountingComparisonScenario(BaseScenario):
    """Two accountants' epsilon for the same DP-SGD run at several lengths (epochs -> epsilon)."""
    sampling_rate: float
    noise_multiplier: float
    delta: float
    accountant: Dict[int, float]      # tight accountant
    strong_composition: Dict[int, float]

    def tightening(self, epochs) -> float:
        """Strong-composition epsilon over the accountant's, same run and length."""
        return self.strong_composition[epochs] / self.accountant[epochs]


class MembershipAuditScenario(BaseScenario):
    """A membership-inference audit on the ROC plane against the ceiling a pure
    ``epsilon``-DP release allows; the audit curve ``FPR + a FPR (1 - FPR)`` is illustrative."""
    epsilon: float
    audit_shape: float

    def ceiling(self, fpr) -> float:
        from ..physics.privacy import calc_membership_tpr_ceiling
        return calc_membership_tpr_ceiling(fpr, self.epsilon)

    def audit(self, fpr) -> float:
        return fpr + self.audit_shape * fpr * (1 - fpr)


class DpSgdRunScenario(BaseScenario):
    """Inputs a DP-SGD privacy accountant needs for one Poisson-sampled run on ``dataset_size``
    examples; this type does not estimate epsilon."""
    dataset_size: int
    batch_size: int
    clip_norm: float
    noise_multiplier: float
    epochs: int
    target_epsilon: float
    delta: float

    @property
    def sampling_rate(self) -> float:
        return self.batch_size / self.dataset_size

    @property
    def steps(self) -> int:
        return int(self.epochs * self.dataset_size / self.batch_size)


class PoisonableShareScenario(BaseScenario):
    """A ``poisoned_fraction`` of a ``dataset_size``-item web-scale dataset an attacker could
    control for ``cost``."""
    dataset_size: float
    poisoned_fraction: float
    cost: Quantity

    @property
    def poisoned_items(self) -> float:
        return self.dataset_size * self.poisoned_fraction
# ── Robust AI: drift tests, per-request answering, attacks, and silent faults (added 2026-10-20) ──

class MeanShiftTestScenario(BaseScenario):
    """A feature mean that moves from ``reference_mean`` to ``observed_mean`` over ``n_samples`` requests.

    The test assumes a known reference mean, independent requests, and the
    stated spread ``feature_sd``: z is the move in standard errors of the mean.
    """
    reference_mean: float
    observed_mean: float
    feature_sd: float
    n_samples: int
    significance_bound: Optional[float] = None   # the bound prose compares the p-value with

    @property
    def mean_shift(self) -> float:
        return self.observed_mean - self.reference_mean

    @property
    def standard_error(self) -> float:
        from ..physics import calc_standard_error
        return calc_standard_error(self.feature_sd, self.n_samples)

    @property
    def z_score(self) -> float:
        return self.mean_shift / self.standard_error

    @property
    def p_value(self) -> float:
        """Two-sided normal-tail probability of a move at least this large from noise alone."""
        from statistics import NormalDist
        return 2 * (1 - NormalDist().cdf(abs(self.z_score)))


class GaussianDriftScenario(BaseScenario):
    """A continuous feature fitted as a normal at reference and now: KL divergence of the reference from the current fit."""
    reference_mean: float
    reference_sd: float
    current_mean: float
    current_sd: float
    kl_trigger: Optional[float] = None    # the service's investigate threshold on KL

    @property
    def triggers(self) -> bool:
        return self.kl >= self.kl_trigger

    @property
    def kl_terms(self):
        """(log term, ratio numerator, ratio denominator) of the closed form."""
        from ..physics import calc_gaussian_kl_terms
        return calc_gaussian_kl_terms(self.reference_mean, self.reference_sd, self.current_mean, self.current_sd)

    @property
    def kl_log_term(self) -> float:
        return self.kl_terms[0]

    @property
    def kl_ratio_term(self) -> float:
        _, num, den = self.kl_terms
        return num / den

    @property
    def kl(self) -> float:
        from ..physics import calc_gaussian_kl
        return calc_gaussian_kl(self.reference_mean, self.reference_sd, self.current_mean, self.current_sd)


class SelectiveAnsweringScenario(BaseScenario):
    """A server that answers a ``coverage`` share of ``request_rate`` and routes the rest to a fallback.

    The selective risk follows an illustrative curve,
    ``risk(c) = risk_floor + risk_scale * c ** risk_exponent``, so answering
    fewer requests lowers the error on those answered.
    """
    request_rate: Quantity
    coverage: float
    risk_floor: float
    risk_scale: float
    risk_exponent: float

    def risk_at(self, coverage: float) -> float:
        return self.risk_floor + self.risk_scale * coverage ** self.risk_exponent

    @property
    def risk(self) -> float:
        """Error on the answered requests at the operating coverage."""
        return self.risk_at(self.coverage)

    @property
    def full_risk(self) -> float:
        """Error when every request is answered."""
        return self.risk_at(1.0)

    @property
    def fallback_rate(self) -> Quantity:
        return self.request_rate * (1 - self.coverage)


class ConformalThresholdScenario(BaseScenario):
    """Split conformal prediction with ``calibration_count`` held-out scores and target miscoverage ``alpha``."""
    calibration_count: int
    alpha: float

    @property
    def rank_bound(self) -> float:
        """``(n + 1)(1 - alpha)`` before the ceiling."""
        return (self.calibration_count + 1) * (1 - self.alpha)

    @property
    def rank(self) -> int:
        from ..physics import calc_split_conformal_rank
        return calc_split_conformal_rank(self.calibration_count, self.alpha)

    @property
    def target_coverage(self) -> float:
        return 1 - self.alpha


class AdversarialTrainingPassesScenario(BaseScenario):
    """Work per batch of adversarial training, in standard forward-backward passes (equal cost, no reuse)."""
    attack_passes: int                # passes spent finding the perturbation
    update_passes: int                # passes spent on the update (2 for a doubled clean-plus-adversarial batch)
    standard_passes: int = 1          # one pass per batch for standard training

    @property
    def passes(self) -> int:
        return self.attack_passes + self.update_passes

    @property
    def slowdown(self) -> float:
        """Work multiple over standard training."""
        return self.passes / self.standard_passes


class CrossConditionGapScenario(BaseScenario):
    """A standard model's clean accuracy against a robust model's attacked accuracy: two of four cells.

    The difference mixes a model change with a test change, so it is neither
    the clean-accuracy penalty nor the robustness benefit.
    """
    standard_clean_accuracy: float    # percent
    robust_attacked_accuracy: float   # percent

    @property
    def gap(self) -> float:
        return self.standard_clean_accuracy - self.robust_attacked_accuracy


class CertifiedRadiusScenario(BaseScenario):
    """Binary randomized smoothing at noise ``noise_sd``, against an l-infinity budget on one input.

    The l-infinity budget is ``linf_levels`` steps of an image whose pixels take
    ``pixel_levels`` values; its box corner has Euclidean length
    ``budget * sqrt(coordinates)``, the length the certificate must reach to
    cover the whole box.
    """
    noise_sd: float
    top_class_probability: float
    linf_levels: int
    pixel_levels: int                 # pixel values an 8-bit channel takes
    coordinates: int                  # width * height * channels

    @property
    def radius(self) -> float:
        from ..physics import calc_smoothing_radius
        return calc_smoothing_radius(self.noise_sd, self.top_class_probability)

    @property
    def quantile(self) -> float:
        """``Phi^-1(p_A)``."""
        return self.radius / self.noise_sd

    @property
    def linf_budget(self) -> float:
        return self.linf_levels / (self.pixel_levels - 1)

    @property
    def box_corner_l2(self) -> float:
        from ..physics import calc_linf_corner_l2
        return calc_linf_corner_l2(self.linf_budget, self.coordinates)

    @property
    def covers_box(self) -> bool:
        return self.radius >= self.box_corner_l2


class FloatBitFlipScenario(BaseScenario):
    """One stored binary32 value with bit ``flipped_bit`` flipped (bit 30 is the top exponent bit)."""
    value: float
    flipped_bit: int
    exponent_bits: int                # core.units.FLOAT_FORMAT_BITS["fp32"]
    fraction_bits: int

    @property
    def _pair(self):
        from ..physics import calc_float32_bit_flip
        return calc_float32_bit_flip(self.value, self.flipped_bit)

    @property
    def stored(self) -> float:
        return self._pair[0]

    @property
    def corrupted(self) -> float:
        return self._pair[1]

    @property
    def scale_log2(self) -> int:
        """The flip multiplies the value by 2 to this power."""
        import math
        return round(math.log2(self.corrupted / self.stored))

    @property
    def word_bits(self) -> int:
        return 1 + self.exponent_bits + self.fraction_bits


class StandardizedAttackScenario(BaseScenario):
    """Published defenses under one standardized attack (``Literature.Series``): best by year and broken defenses.

    Rows without a year and rows the benchmark flags unreliable are dropped. A
    defense below ``broken_floor`` percent attacked accuracy provides no robustness.
    """
    series_name: str
    first_year: int
    last_year: int
    broken_floor: float               # percent

    @property
    def rows(self):
        from ..literature.registry import Literature
        return [r for r in getattr(Literature.Series, self.series_name).entries
                if r["year"] and r["unreliable"] != "True"]

    @property
    def years(self) -> List[int]:
        return list(range(self.first_year, self.last_year + 1))

    def best(self, year: int):
        """The row with the highest attacked accuracy published in ``year``."""
        return max((r for r in self.rows if int(r["year"]) == year), key=lambda r: r["autoattack_acc"])

    @property
    def best_first(self):
        return self.best(self.first_year)

    @property
    def best_last(self):
        return self.best(self.last_year)

    @property
    def broken(self):
        return [r for r in self.rows if r["autoattack_acc"] < self.broken_floor]

    def best_so_far(self, year: int) -> float:
        return max(self.best(y)["autoattack_acc"] for y in self.years if y <= year)
# --- The agent harness (added 2026-10-20) -------------------------------------
# What the runtime around one trajectory holds, shares, and admits: approval
# escrow, fair scheduling of model calls between request classes, admission of
# the next call against a dollar allowance, the budget vector's meters, and
# call or step ceilings read from public runs. Solvers in engine/solvers/harness.py.

class ApprovalEscrowScenario(BaseScenario):
    """Approval escrow for a long-running workload (held resources) and an urgent one (time)."""
    repair_rate: Quantity            # review requests per unit time
    repair_wait: Quantity            # mean time to a review decision
    lease: Quantity                  # workspace each held trajectory keeps
    remediation_rate: Quantity
    diagnosis: Quantity              # before the proposal
    remediation_wait: Quantity       # mean time to an on-call decision
    apply_verify: Quantity           # after approval

    @property
    def result(self):
        from .solvers import ApprovalEscrowModel
        return ApprovalEscrowModel().solve(self.repair_rate, self.repair_wait, self.lease,
                                           self.remediation_rate, self.diagnosis,
                                           self.remediation_wait, self.apply_verify)


class DeficitRoundRobinScenario(BaseScenario):
    """Backlogged request classes sharing model calls under weighted deficit round-robin."""
    quantum: float                   # charge units per unit weight per round
    classes: Dict[str, Dict[str, int]]     # name -> weight, input, output (allowance)
    simulated_rounds: int = 10_000

    @property
    def result(self):
        from .solvers import DeficitRoundRobinModel
        return DeficitRoundRobinModel().solve(self.quantum, self.classes, self.simulated_rounds)


class BoundedLiabilityAdmissionScenario(BaseScenario):
    """The next model call of a trajectory, tested against its dollar allowance."""
    allowance: Quantity              # USD
    settled: Quantity                # actual spend so far
    reserved: Quantity               # outstanding reservations of unsettled work
    finalization_reserve: Quantity   # protected for checks, report, and cleanup
    input_tokens: int
    output_allowance: int
    input_price: Quantity            # USD per token (undiscounted)
    output_price: Quantity
    probe: Quantity                  # an unsettled amount for expressing sensitivity

    @property
    def result(self):
        from .solvers import BoundedLiabilityAdmissionModel
        return BoundedLiabilityAdmissionModel().solve(
            self.allowance, self.settled, self.reserved, self.finalization_reserve,
            self.input_tokens, self.output_allowance, self.input_price, self.output_price,
            self.probe)


class TrajectoryBudgetMeterScenario(BaseScenario):
    """An append-only trajectory metered against turns, tokens, time, and dollars."""
    trajectory: TrajectoryResendScenario
    input_price: Quantity
    cached_fraction: float           # cached input billed at this share of the input price
    output_price: Quantity
    turn_allowance: int
    token_allowance: int
    time_allowance: Quantity
    dollar_allowance: Quantity
    elapsed: Quantity                # wall clock at the readout

    @property
    def result(self):
        from .solvers import TrajectoryBudgetMeterModel
        t = self.trajectory.result
        allowances = {"turns": self.turn_allowance, "tokens": self.token_allowance,
                      "time": float(self.time_allowance.to("second").magnitude),
                      "dollars": float(self.dollar_allowance.to("USD").magnitude)}
        return TrajectoryBudgetMeterModel().solve(
            t.turns, t.resent_tokens, t.reused_prefix_tokens, t.output_tokens,
            self.input_price, self.input_price * self.cached_fraction, self.output_price,
            allowances, float(self.elapsed.to("second").magnitude))


class CallCeilingScenario(BaseScenario):
    """A model-call ceiling at the accepted runs' quantile, over every submission in a series."""
    series_name: str
    quantile: float

    @property
    def rows(self):
        from ..literature.registry import Literature
        return getattr(Literature, self.series_name).entries

    @property
    def result(self):
        from .solvers import CallCeilingModel
        return CallCeilingModel().solve(self.rows, self.quantile)
# --- Agent security: injection exposure and defense evaluation (added 2026-10-20) ---
class InjectionExposureScenario(BaseScenario):
    """Workloads reading ``reads`` untrusted items each, a fraction ``payload_fraction`` carrying
    a payload, with and without a screen that misses ``miss_rate`` of payloads."""
    payload_fraction: float
    reads: Dict[str, int]            # untrusted items read per task, by workload
    miss_rate: float                 # fraction of payloads the screen misses
    cohort: int                      # tasks in the per-cohort count

    @property
    def result(self):
        from .solvers import InjectionExposureModel
        return InjectionExposureModel().solve(self.payload_fraction, self.reads, self.miss_rate,
                                              self.cohort)


class TurnExposureScenario(BaseScenario):
    """``reads_per_turn`` untrusted items per turn over ``turns`` turns at ``payload_fraction``."""
    payload_fraction: float
    reads_per_turn: int
    turns: int

    @property
    def result(self):
        from .solvers import TurnExposureModel
        return TurnExposureModel().solve(self.payload_fraction, self.reads_per_turn, self.turns)


class PolicyComparisonScenario(BaseScenario):
    """Two policies, each run on ``attacked`` attacked and ``benign`` separate benign episodes.

    Each policy's counts: ``confirmed`` and ``unresolved`` harm and ``accepted_attacked`` among
    attacked episodes; ``accepted_benign`` and ``false_block`` among benign ones.
    """
    attacked: int
    benign: int
    baseline: Dict[str, int]
    candidate: Dict[str, int]
    attempt_cost: Optional[float] = None    # USD per attempt, to price cost per accepted task

    @property
    def result(self):
        from .solvers import PolicyComparisonModel
        return PolicyComparisonModel().solve(self.attacked, self.benign, self.baseline,
                                             self.candidate, self.attempt_cost)


class TaskScopedGrantScenario(BaseScenario):
    """A requester holding ``held`` resources delegates a task that needs ``needed`` of them."""
    held: int
    needed: int

    @property
    def refused(self) -> int:
        """Resources the requester's own token reaches that the task-scoped grant refuses."""
        return self.held - self.needed

    @property
    def scoped_share(self) -> float:
        return self.needed / self.held


class StepCeilingReturnScenario(BaseScenario):
    """Success of the same models at several enforced step ceilings, from a published series."""
    series_name: str
    steps: Optional[List[int]] = None    # None: every ceiling the series reports

    @property
    def result(self):
        from ..literature.registry import Literature
        from .solvers import StepCeilingReturnModel
        rows = getattr(Literature, self.series_name).entries
        steps = self.steps or sorted({int(r["steps"]) for r in rows})
        return StepCeilingReturnModel().solve(rows, steps)
# --- Closed-loop evaluation: trial campaigns, sequential tests, and exposure (added 2026-10-20) ---

class ClosedLoopCampaignScenario(BaseScenario):
    """``clean_runs`` failure-free closed-loop trials of a frozen policy in one cell of ``table`` on
    ``site``, each ``run`` long with an automated ``reset``, bounded at one-sided ``confidence``.

    What a success target ``1 - target_rate`` costs in runs, cell time, and operator time; the
    exact bound when ``recorded_failures`` runs fail; and the contrasts the bound is taught with:
    a policy that fails one run in ``weak_policy_one_in``, ``fallacy_runs`` clean runs against the
    target, ``looks`` independent looks at the significance level, and a Bonferroni split of the
    significance level over ``joint_cells`` cells. A run that jams costs ``jam_reset`` instead of
    ``reset``.
    """
    site: Any                        # EmbodiedSiteScenario (latch limits and approach speed)
    table: Any                       # ScenarioTableCoverageScenario (the cells the campaign samples)
    standoff: Quantity
    run: Quantity
    reset: Quantity
    jam_reset: Quantity
    operators: int
    shift: Quantity
    clean_runs: int
    confidence: float
    target_rate: float               # failure rate the campaign argues toward
    estimate_target_rate: float      # failure rate of the trial-budget estimate
    weak_policy_one_in: int
    recorded_failures: int
    joint_cells: int
    looks: int
    omissions: int                   # unclosed departures in the policy manifest's omission log
    fallacy_runs: int

    @property
    def alpha(self) -> float:
        return 1.0 - self.confidence

    @property
    def ln_inv_alpha(self) -> float:
        """The numerator of every zero-failure count, about 3 at 95 percent."""
        return -_math.log(self.alpha)

    def lower_bound_at(self, trials: int, confidence: Optional[float] = None) -> float:
        from ..physics.robotics.assurance import calc_clopper_pearson_zero_failure_bound
        return calc_clopper_pearson_zero_failure_bound(trials, self.confidence if confidence is None else confidence)

    def runs_for(self, target: float, confidence: Optional[float] = None) -> int:
        from ..physics.robotics.assurance import calc_zero_failure_sample_size
        return calc_zero_failure_sample_size(target, self.confidence if confidence is None else confidence)

    @property
    def lower_bound(self) -> float:
        return self.lower_bound_at(self.clean_runs)

    @property
    def failure_upper(self) -> float:
        return 1.0 - self.lower_bound

    @property
    def rule_of_three(self) -> float:
        from ..physics.robotics.assurance import calc_rule_of_three_bound
        return calc_rule_of_three_bound(self.clean_runs)

    @property
    def target(self) -> float:
        return 1.0 - self.target_rate

    @property
    def nine_targets(self) -> List[float]:
        """Success targets with one, two, ... nines, up to the campaign target (0.9, 0.99, 0.999)."""
        nines = round(-_math.log10(self.target_rate))
        return [1.0 - 10.0 ** -k for k in range(1, nines + 1)]

    @property
    def target_runs(self) -> int:
        return self.runs_for(self.target)

    @property
    def joint_runs(self) -> int:
        """Runs per cell when the significance level is split evenly over ``joint_cells``."""
        return self.runs_for(self.target, 1.0 - self.alpha / self.joint_cells)

    @property
    def estimate_target(self) -> float:
        return 1.0 - self.estimate_target_rate

    @property
    def estimate_runs(self) -> int:
        return self.runs_for(self.estimate_target)

    @property
    def estimate_rule_of_three_runs(self) -> float:
        from ..physics.robotics.assurance import calc_rule_of_three_trials
        return calc_rule_of_three_trials(self.estimate_target)

    def clean_probability(self, success: float) -> float:
        """Chance that a policy of per-run success ``success`` logs the campaign clean."""
        return success ** self.clean_runs

    @property
    def weak_policy_success(self) -> float:
        return 1.0 - 1.0 / self.weak_policy_one_in

    @property
    def weak_policy_clean_probability(self) -> float:
        return self.clean_probability(self.weak_policy_success)

    @property
    def lower_bound_with_failures(self) -> float:
        from ..physics.robotics.assurance import calc_clopper_pearson_lower_bound
        return calc_clopper_pearson_lower_bound(self.clean_runs - self.recorded_failures, self.clean_runs,
                                                self.confidence)

    @property
    def observed_rate_with_failures(self) -> float:
        return (self.clean_runs - self.recorded_failures) / self.clean_runs

    @property
    def fallacy_lower_bound(self) -> float:
        return self.lower_bound_at(self.fallacy_runs)

    @property
    def any_false_positive(self) -> float:
        """At least one false positive across ``looks`` independent looks at ``alpha``."""
        from ..physics.agents import calc_independent_success
        return calc_independent_success(self.alpha, self.looks)

    @property
    def cycle(self) -> Quantity:
        return (self.run + self.reset).to("s")

    def campaign_time(self, trials: Optional[int] = None) -> Quantity:
        return ((self.clean_runs if trials is None else trials) * self.cycle).to("s")

    def operator_time(self, trials: int) -> Quantity:
        return (self.operators * self.campaign_time(trials)).to("hour")

    def shifts(self, trials: int) -> float:
        return (self.operator_time(trials) / self.shift).to("").magnitude

    @property
    def jam_reset_share(self) -> float:
        return (self.jam_reset / (self.run + self.jam_reset)).to("").magnitude

    @property
    def jam_cycle(self) -> Quantity:
        return (self.run + self.jam_reset).to("s")


class SequentialTestScenario(BaseScenario):
    """Wald's SPRT between an unacceptable failure rate ``p_unacceptable`` and an acceptable one
    ``p_acceptable`` at error rates ``alpha`` and ``beta``; the score after ``trace_clean_runs``
    clean runs and one failure; and the fixed zero-failure test for success ``1 - p_unacceptable``
    at the same ``alpha``."""
    p_unacceptable: float
    p_acceptable: float
    alpha: float
    beta: float
    trace_clean_runs: int

    @property
    def success_increment(self) -> float:
        from ..physics.statistics import calc_sprt_increments
        return calc_sprt_increments(self.p_unacceptable, self.p_acceptable)[0]

    @property
    def failure_increment(self) -> float:
        from ..physics.statistics import calc_sprt_increments
        return calc_sprt_increments(self.p_unacceptable, self.p_acceptable)[1]

    @property
    def accept(self) -> float:
        from ..physics.statistics import calc_sprt_boundaries
        return calc_sprt_boundaries(self.alpha, self.beta)[0]

    @property
    def reject(self) -> float:
        from ..physics.statistics import calc_sprt_boundaries
        return calc_sprt_boundaries(self.alpha, self.beta)[1]

    @property
    def runs_to_accept(self) -> int:
        return _math.ceil(self.accept / self.success_increment)

    @property
    def false_accept_bound(self) -> float:
        """Ville's inequality: ``e^-A = alpha / (1 - beta)``."""
        return _math.exp(-self.accept)

    @property
    def false_reject_bound(self) -> float:
        return _math.exp(self.reject)

    @property
    def trace_score(self) -> float:
        return self.trace_clean_runs * self.success_increment + self.failure_increment

    @property
    def fixed_runs(self) -> int:
        from ..physics.robotics.assurance import calc_zero_failure_sample_size
        return calc_zero_failure_sample_size(1.0 - self.p_unacceptable, 1.0 - self.alpha)

    @property
    def fixed_rejects_acceptable(self) -> float:
        """Chance that a policy at the acceptable rate fails the fixed zero-failure test."""
        return 1.0 - (1.0 - self.p_acceptable) ** self.fixed_runs


class ArmComparisonScenario(BaseScenario):
    """Two policies run ``arm_runs`` times each, with ``successes_a`` and ``successes_b``
    successes; Fisher's exact test, and the runs per arm that separate ``p_high`` from ``p_low``
    two-sided at ``alpha`` with ``power``."""
    arm_runs: int
    successes_a: int
    successes_b: int
    alpha: float
    power: float
    p_high: float
    p_low: float

    @property
    def fisher_p(self) -> float:
        from ..physics.statistics import calc_fisher_exact_two_sided
        n = self.arm_runs
        return calc_fisher_exact_two_sided(self.successes_a, n - self.successes_a, self.successes_b, n - self.successes_b)

    @property
    def total_runs(self) -> int:
        return 2 * self.arm_runs

    @property
    def failures(self) -> int:
        return self.total_runs - self.successes_a - self.successes_b

    @property
    def all_in_one_arm(self) -> float:
        """Chance that every failure lands in one named arm under the null."""
        return _math.comb(self.arm_runs, self.failures) / _math.comb(2 * self.arm_runs, self.failures)

    @property
    def z_alpha(self) -> float:
        from statistics import NormalDist
        return NormalDist().inv_cdf(1 - self.alpha / 2)

    @property
    def z_power(self) -> float:
        from statistics import NormalDist
        return NormalDist().inv_cdf(self.power)

    @property
    def cohens_h(self) -> float:
        from ..physics.statistics import calc_cohens_h
        return calc_cohens_h(self.p_high, self.p_low)

    def runs_per_arm_for(self, effect_h: float) -> int:
        from ..physics.statistics import calc_arcsine_two_arm_sample_size
        return calc_arcsine_two_arm_sample_size(effect_h, self.alpha, self.power)

    @property
    def runs_per_arm(self) -> int:
        return self.runs_per_arm_for(self.cohens_h)


class HazardExposureScenario(BaseScenario):
    """Failure-free exposure that bounds each of ``rates`` (constant hazards per hour) at one-sided
    ``confidence``, against what ``rigs`` rigs or one rig over ``rig_time`` accumulate."""
    rates: List[Quantity]
    confidence: float
    rigs: int
    rig_time: Optional[Quantity] = None        # optional (2026-10-20): needed only for rig_rate
    # Optional (added 2026-10-20): operating hours each rig runs per calendar day,
    # so a fleet's failure-free exposure converts to the days a stage must run.
    hours_per_day: Optional[Quantity] = None

    @property
    def ln_inv_alpha(self) -> float:
        return -_math.log(1.0 - self.confidence)

    def exposure(self, rate: Quantity) -> Quantity:
        from ..physics.robotics.assurance import calc_empirical_testing_exposure
        return calc_empirical_testing_exposure(rate, confidence=self.confidence).to("hour")

    def fleet_time(self, rate: Quantity) -> Quantity:
        return (self.exposure(rate) / self.rigs).to("hour")

    @property
    def rig_rate(self) -> Quantity:
        """The smallest constant rate one rig running for ``rig_time`` can bound."""
        return (self.ln_inv_alpha / self.rig_time).to("1/hour")

    def shortfall(self, rate: Quantity) -> float:
        return (self.rig_rate / rate).to("").magnitude

    def calendar_days(self, rate: Quantity) -> Quantity:
        """Days the ``rigs`` need, at ``hours_per_day`` each, to accumulate the exposure (added 2026-10-20)."""
        return (self.fleet_time(rate) / self.hours_per_day).to("day")


class MissionExposureScenario(BaseScenario):
    """Failure-free missions of ``mission`` at ``machine``'s drive limit, with ``reset`` between
    them, that bound ``hazard_rate`` at one-sided ``confidence``; and the exposure a relaxed
    target of one failure per ``relaxed_mtbf`` needs."""
    machine: Any                     # MobileManipulator (drive limit, wheel diameter)
    hazard_rate: Quantity
    mission: Quantity
    reset: Quantity
    confidence: float
    relaxed_mtbf: Quantity

    @property
    def mission_distance(self) -> Quantity:
        return (self.machine.base.max_velocity * self.mission).to("m")

    @property
    def cycle(self) -> Quantity:
        return (self.mission + self.reset).to("s")

    @property
    def continuous_exposure(self) -> Quantity:
        from ..physics.robotics.assurance import calc_empirical_testing_exposure
        return calc_empirical_testing_exposure(self.hazard_rate, confidence=self.confidence).to("hour")

    @property
    def mission_failure_probability(self) -> float:
        return (self.hazard_rate * self.mission).to("").magnitude

    @property
    def missions_needed(self) -> float:
        from ..physics.robotics.assurance import calc_zero_failure_mission_count
        return calc_zero_failure_mission_count(self.hazard_rate, self.mission, self.confidence)

    @property
    def calendar_time(self) -> Quantity:
        return (self.missions_needed * self.cycle).to("hour")

    @property
    def wheel_revolutions(self) -> float:
        return (self.missions_needed * self.mission_distance
                / (_math.pi * self.machine.drive_unit.wheel_diameter)).to("").magnitude

    @property
    def relaxed_exposure(self) -> Quantity:
        from ..physics.robotics.assurance import calc_empirical_testing_exposure
        return calc_empirical_testing_exposure(1 / self.relaxed_mtbf, confidence=self.confidence).to("hour")


class InterventionCampaignScenario(BaseScenario):
    """``interventions`` interventions in ``missions`` supervised runs of ``mission``: burden per
    hour and per kilometer, and an exact two-sided ``1 - alpha`` interval on the rate."""
    mission: MissionExposureScenario
    missions: int
    interventions: int
    alpha: float

    @property
    def active_time(self) -> Quantity:
        return (self.missions * self.mission.mission).to("hour")

    @property
    def distance(self) -> Quantity:
        return (self.missions * self.mission.mission_distance).to("km")

    @property
    def mtbi(self) -> Quantity:
        return (self.active_time / self.interventions).to("minute")

    @property
    def density(self) -> Quantity:
        return (self.interventions / self.distance).to("1/km")

    @property
    def rate(self) -> Quantity:
        return (self.interventions / self.active_time).to("1/hour")

    @property
    def count_interval(self):
        from ..physics.statistics import calc_poisson_exact_interval
        return calc_poisson_exact_interval(self.interventions, self.alpha)

    @property
    def rate_interval(self):
        lo, hi = self.count_interval
        return (lo / self.active_time).to("1/hour"), (hi / self.active_time).to("1/hour")


class HeadingDriftScenario(BaseScenario):
    """A persistent ``heading_offset`` at ``machine``'s drive limit in ``site``'s aisle: the
    angular error an open-loop replay scores, and where the closed loop spends the side clearance
    (``y = v sin(eps) t``)."""
    machine: Any
    site: Any
    heading_offset: Quantity

    @property
    def speed(self) -> Quantity:
        return self.machine.base.max_velocity

    @property
    def heading(self) -> float:
        """The offset in radians."""
        return _math.radians(self.heading_offset.to("degree").magnitude)

    @property
    def angular_mse(self) -> Quantity:
        return (Q_(self.heading, "radian") ** 2).to("radian**2")

    @property
    def time_to_clearance(self) -> Quantity:
        return (self.site.side_clearance / (self.speed * _math.sin(self.heading))).to("s")

    @property
    def distance_to_clearance(self) -> Quantity:
        return (self.speed * self.time_to_clearance).to("m")

    @property
    def aisle_fraction(self) -> float:
        return (self.distance_to_clearance / self.site.aisle_length).to("").magnitude


class StiffContactScenario(BaseScenario):
    """A stiff contact of ``stiffness`` (a linear spring, ``F = k x``): the force a ``residual``
    penetration commands, and the penetrations at ``site``'s latch tripwire and limit."""
    site: Any
    stiffness: Quantity
    residual: Quantity

    @property
    def residual_force(self) -> Quantity:
        return (self.stiffness * self.residual).to("N")

    def penetration_at(self, force: Quantity) -> Quantity:
        return (force / self.stiffness).to("mm")

    @property
    def tripwire_penetration(self) -> Quantity:
        return self.penetration_at(self.site.f_latch_tripwire)

    @property
    def limit_penetration(self) -> Quantity:
        return self.penetration_at(self.site.f_latch_limit)


class ForceTailScenario(BaseScenario):
    """A contact force over ``trials`` trials with a benign ``mean_force`` and a ``p99_force`` tail,
    against ``site``'s tripwire and structural limit."""
    site: Any
    trials: int
    mean_force: Quantity
    p99_force: Quantity

    @property
    def tail_over_limit(self) -> Quantity:
        return (self.p99_force - self.site.f_latch_limit).to("N")

    @property
    def tail_over_mean(self) -> float:
        return (self.p99_force / self.mean_force).to("").magnitude


class RoadExposureScenario(BaseScenario):
    """Miles-to-demonstrate requirements per event class (zero failures, precision, improvement),
    in millions of miles at human benchmark ``rates`` per 100 million miles, against one dated
    ``fleet_miles`` total (millions) at one-sided ``confidence``."""
    rates: Dict[str, float]
    zero_failure: Dict[str, float]
    precision: Dict[str, float]
    improvement: Dict[str, float]
    fleet_miles: float
    confidence: float

    def recomputed_zero_failure(self, event: str) -> float:
        """``-ln(1 - C) / rate``, in millions of miles: RAND's row (1) re-derived."""
        return -_math.log(1.0 - self.confidence) / (self.rates[event] / 100.0)

    def requirements(self, event: str):
        """RAND's three rows for one event class: zero failures, precision, improvement (millions)."""
        return tuple(float(t[event]) for t in (self.zero_failure, self.precision, self.improvement))

    def shortfall(self, event: str) -> float:
        return self.zero_failure[event] / self.fleet_miles

    def precision_over_fleet(self, event: str) -> float:
        return self.precision[event] / self.fleet_miles

    @property
    def cleared(self) -> List[float]:
        """Requirements, of every event class and question, the fleet total exceeds."""
        rows = list(self.zero_failure.values()) + list(self.precision.values()) + list(self.improvement.values())
        return [m for m in rows if m < self.fleet_miles]


class CrashRateDenominatorScenario(BaseScenario):
    """A reported before-and-after crash rate and the same comparison on the cohort whose exposure
    is complete (rates per million miles)."""
    reported_before: float
    reported_after: float
    cohort_before: float
    cohort_after: float
    exact_exposure_share: float      # fraction of vehicles with exact exposure

    @property
    def reported_change(self) -> float:
        return self.reported_after / self.reported_before - 1.0

    @property
    def cohort_change(self) -> float:
        return self.cohort_after / self.cohort_before - 1.0
# --- Sensor perception: capture timing, age, and error scales (added 2026-10-20) ---
# A site scenario (``Embodied.Scenario.*``) supplies the machine's navigation
# camera and lidar, the stage allowances, and the clearances; the solvers live
# in ``engine/solvers/perception.py``.

class ObservationAgeScenario(BaseScenario):
    """``site.machine``'s navigation-camera observation age at ``speed``, with a ``batch_frames`` comparison.

    ``exposure`` replaces the camera's own exposure time (for example, a dim
    aisle that lengthens it); the other stages are unchanged (added 2026-10-20).
    """
    site: Any                                   # EmbodiedSiteScenario
    speed: Quantity
    batch_frames: int = 1
    exposure: Optional[Quantity] = None         # default: the navigation camera's exposure time

    @property
    def exposure_time(self) -> Quantity:
        return self.site.machine.nav_camera.exposure_time if self.exposure is None else self.exposure

    @property
    def result(self):
        from .solvers import ObservationAgeModel
        s, cam = self.site, self.site.machine.nav_camera
        return ObservationAgeModel().solve(self.exposure_time, cam.readout_time, s.t_transport, s.t_dma,
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


# --- Responsible AI (added 2026-10-20) -----------------------------------------
# Group fairness criteria from confusion counts, the evidence a measured gap
# carries, threshold policies, monitoring load, and the capacity each safeguard
# spends on a named path.

class GroupConfusionScenario(BaseScenario):
    """Binary decisions for two or more groups as confusion counts ``{group: {tp, fp, fn, tn}}``.

    A positive decision is an approval (or a flag); the positive label is the
    outcome the decision should select. ``gap`` compares the first two groups,
    first minus second.
    """
    groups: Dict[str, Dict[str, float]]

    def _c(self, group: str) -> Dict[str, float]:
        return self.groups[group]

    @property
    def first(self) -> str:
        return list(self.groups)[0]

    @property
    def second(self) -> str:
        return list(self.groups)[1]

    def size(self, group: str) -> float:
        c = self._c(group)
        return c["tp"] + c["fp"] + c["fn"] + c["tn"]

    def positives(self, group: str) -> float:
        """Positive decisions (approved or flagged)."""
        c = self._c(group)
        return c["tp"] + c["fp"]

    def positive_rate(self, group: str) -> float:
        """Share of the group given a positive decision (the demographic-parity quantity)."""
        return self.positives(group) / self.size(group)

    def tpr(self, group: str) -> float:
        c = self._c(group)
        return c["tp"] / (c["tp"] + c["fn"])

    def fpr(self, group: str) -> float:
        c = self._c(group)
        return c["fp"] / (c["fp"] + c["tn"])

    def ppv(self, group: str) -> float:
        c = self._c(group)
        return c["tp"] / (c["tp"] + c["fp"])

    def base_rate(self, group: str) -> float:
        c = self._c(group)
        return (c["tp"] + c["fn"]) / self.size(group)

    def fnr(self, group: str) -> float:
        """Share of the positive-label members given a negative decision (added 2026-10-20)."""
        c = self._c(group)
        return c["fn"] / (c["tp"] + c["fn"])

    def accuracy(self, group: str) -> float:
        """Share of the group decided correctly (added 2026-10-20)."""
        c = self._c(group)
        return (c["tp"] + c["tn"]) / self.size(group)

    @property
    def overall_accuracy(self) -> float:
        """Share of all members of all groups decided correctly (added 2026-10-20)."""
        right = sum(c["tp"] + c["tn"] for c in self.groups.values())
        return right / sum(self.size(g) for g in self.groups)

    def share(self, group: str) -> float:
        """The group's share of all members (added 2026-10-20)."""
        return self.size(group) / sum(self.size(g) for g in self.groups)

    def gap(self, metric: str) -> float:
        """``metric`` (positive_rate, tpr, fpr, ppv, base_rate) of the first group minus the second's."""
        f = getattr(self, metric)
        return f(self.first) - f(self.second)

    @property
    def disparate_impact(self) -> float:
        """Lower positive rate over higher positive rate across the first two groups."""
        a, b = self.positive_rate(self.first), self.positive_rate(self.second)
        return min(a, b) / max(a, b)

    @property
    def average_odds_difference(self) -> float:
        """Mean of the absolute TPR and FPR gaps."""
        return 0.5 * (abs(self.gap("tpr")) + abs(self.gap("fpr")))


class ProportionGapTestScenario(BaseScenario):
    """An observed gap between two rates, ``rate_a`` and ``rate_b``, on ``n_per_group`` decisions each.

    Tested with the pooled two-proportion z statistic against the two-sided
    normal critical value at ``significance``; the interval is the unpooled
    normal-approximation interval at the same level.
    """
    rate_a: float
    rate_b: float
    n_per_group: int
    significance: float = 0.05
    small_n_per_group: Optional[int] = None   # a second, smaller sample for comparison

    @property
    def gap(self) -> float:
        return self.rate_a - self.rate_b

    @property
    def pooled_rate(self) -> float:
        return (self.rate_a + self.rate_b) / 2

    @property
    def z_critical(self) -> float:
        from statistics import NormalDist
        return NormalDist().inv_cdf(1 - self.significance / 2)

    def z_at(self, n: float) -> float:
        from ..physics import calc_two_proportion_z
        return calc_two_proportion_z(self.rate_a, self.rate_b, n, n)

    def interval_at(self, n: float):
        from ..physics import calc_proportion_gap_interval
        return calc_proportion_gap_interval(self.rate_a, self.rate_b, n, n, self.z_critical)

    @property
    def z(self) -> float:
        return self.z_at(self.n_per_group)

    @property
    def interval(self):
        return self.interval_at(self.n_per_group)

    @property
    def n_at_critical(self) -> float:
        """Per-group sample size at which this gap first reaches the critical value."""
        from ..physics import calc_equal_n_for_z
        return calc_equal_n_for_z(self.rate_a, self.rate_b, self.z_critical)


class ThresholdPolicyScenario(BaseScenario):
    """Illustrative scores per group, ``{group: {"qualified": [...], "default": [...]}}``, under
    one ``uniform_threshold`` or group-adapted ``group_thresholds`` (approve at or above)."""
    scores: Dict[str, Dict[str, List[float]]]
    uniform_threshold: float
    group_thresholds: Dict[str, float] = {}
    alternative_threshold: Optional[float] = None   # a second shared threshold (added 2026-10-20)

    def tpr(self, group: str, threshold: float) -> float:
        q = self.scores[group]["qualified"]
        return sum(s >= threshold for s in q) / len(q)

    def fpr(self, group: str, threshold: float) -> float:
        d = self.scores[group]["default"]
        return sum(s >= threshold for s in d) / len(d)

    def errors(self, group: str, threshold: float) -> int:
        """Qualified members rejected plus defaulting members approved (added 2026-10-20)."""
        q, d = self.scores[group]["qualified"], self.scores[group]["default"]
        return sum(s < threshold for s in q) + sum(s >= threshold for s in d)

    def size(self, group: str) -> int:
        return len(self.scores[group]["qualified"]) + len(self.scores[group]["default"])

    def accuracy(self, threshold: float) -> float:
        """Share of all scored members decided correctly at one shared threshold (added 2026-10-20)."""
        total = sum(self.size(g) for g in self.scores)
        return 1 - sum(self.errors(g, threshold) for g in self.scores) / total


class ConstraintAccuracyCostScenario(BaseScenario):
    """Accuracy before and after a fairness constraint (percent), with the policy's stated rates.

    ``context`` holds the scenario's descriptive rates in percent (default
    rates, approval rates, thresholds) so prose and slides quote one source.
    """
    baseline_accuracy: float      # percent
    constrained_accuracy: float   # percent
    context: Dict[str, float] = {}

    @property
    def accuracy_cost(self) -> float:
        """Points of accuracy given up."""
        return self.baseline_accuracy - self.constrained_accuracy

    @property
    def relative_cost(self) -> float:
        """Accuracy given up as a fraction of the baseline."""
        return self.accuracy_cost / self.baseline_accuracy


class SampledDecisionMonitorScenario(BaseScenario):
    """Per-group metric monitoring of a decision service at ``request_rate``.

    ``sample_share`` of decisions are logged at ``record_size`` each; ``groups``
    subgroups x ``metrics_per_group`` metrics are each tested at
    ``false_alarm_rate`` per evaluation interval on a stable process.
    """
    request_rate: Quantity
    sample_share: float
    record_size: Quantity
    groups: int
    metrics_per_group: int
    false_alarm_rate: float

    @property
    def decisions_per_day(self) -> float:
        return (self.request_rate * Q_(1, "day")).to("").magnitude

    @property
    def records_per_day(self) -> float:
        return self.decisions_per_day * self.sample_share

    @property
    def log_per_day(self) -> Quantity:
        return (self.records_per_day * self.record_size).to("GB")

    @property
    def metrics(self) -> int:
        return self.groups * self.metrics_per_group

    @property
    def false_alerts_per_interval(self) -> float:
        """Expected false alerts per interval if nothing is wrong (no independence needed)."""
        return self.metrics * self.false_alarm_rate

    @property
    def bonferroni_level(self) -> float:
        return self.false_alarm_rate / self.metrics

    @property
    def false_alerts_bonferroni(self) -> float:
        return self.metrics * self.bonferroni_level


class ExactAttributionScenario(BaseScenario):
    """Exact Shapley attribution by subset enumeration: 2^n subsets at ``time_per_subset`` each."""
    feature_counts: List[int]
    time_per_subset: Quantity
    request_budget: Optional[Quantity] = None

    def subsets(self, features: int) -> int:
        return 2 ** features

    def time(self, features: int) -> Quantity:
        return (self.subsets(features) * self.time_per_subset).to("ms")


class InlineOverheadScenario(BaseScenario):
    """A safeguard added inline to a request path of ``base_latency``, costing
    ``overhead_low`` to ``overhead_high`` (fractions of the base) per request.

    Under the proportional assumption, holding throughput needs capacity in the
    ratio of the new latency to the old.
    """
    base_latency: Quantity
    overhead_low: float
    overhead_high: float
    request_rate: Optional[Quantity] = None

    @property
    def added_low(self) -> Quantity:
        return (self.base_latency * self.overhead_low).to("ms")

    @property
    def added_high(self) -> Quantity:
        return (self.base_latency * self.overhead_high).to("ms")

    @property
    def latency_low(self) -> Quantity:
        return (self.base_latency + self.added_low).to("ms")

    @property
    def latency_high(self) -> Quantity:
        return (self.base_latency + self.added_high).to("ms")

    @property
    def capacity_low(self) -> float:
        return (self.latency_low / self.base_latency).to("").magnitude

    @property
    def capacity_high(self) -> float:
        return (self.latency_high / self.base_latency).to("").magnitude


class ExplanationLogScenario(BaseScenario):
    """Logging an explanation with every prediction: ``predictions_per_day`` for ``days_per_month``,
    each a ``map_size_low`` to ``map_size_high`` attribution map (or tabular attributions of
    ``attribution_bytes_low`` to ``attribution_bytes_high`` per feature)."""
    predictions_per_day: float
    days_per_month: float
    map_size_low: Quantity
    map_size_high: Quantity
    attribution_bytes_low: Quantity
    attribution_bytes_high: Quantity

    @property
    def predictions_per_month(self) -> float:
        return self.predictions_per_day * self.days_per_month

    @property
    def monthly_low(self) -> Quantity:
        return (self.predictions_per_month * self.map_size_low).to("TB")

    @property
    def monthly_high(self) -> Quantity:
        return (self.predictions_per_month * self.map_size_high).to("TB")


class ShardedUnlearningScenario(BaseScenario):
    """Exact unlearning by sharded retraining (SISA) against a full retrain of ``workload``.

    The full retrain runs on ``n_accelerators`` of ``accelerator`` at ``mfu`` and
    costs ``full_cost``. Planning rounds it to whole days; one request retrains
    one of ``shards`` equal shards, at a proportional share of time and cost.
    ``requests_per_day`` arrive uniformly at random over the shards.
    """
    workload: Workload
    accelerator: HardwareNode
    n_accelerators: int
    mfu: float
    full_cost: Quantity
    shards: int
    requests_per_day: int
    accuracy_drop_low: Optional[float] = None    # fraction
    accuracy_drop_high: Optional[float] = None   # fraction

    @property
    def full_retrain_time(self) -> Quantity:
        rate = self.n_accelerators * self.accelerator.compute.peak_flops * self.mfu
        return (self.workload.training_ops / rate).to("day")

    @property
    def planning_days(self) -> int:
        """The full retrain rounded to whole days, the figure planning quotes."""
        return round(self.full_retrain_time.to("day").magnitude)

    @property
    def shard_share(self) -> float:
        return 1 / self.shards

    @property
    def shard_cost(self) -> Quantity:
        return (self.full_cost * self.shard_share).to("USD")

    @property
    def shard_time(self) -> Quantity:
        return (Q_(self.planning_days, "day") * self.shard_share).to("hour")

    @property
    def p_shard_untouched(self) -> float:
        """Chance a given shard receives none of a day's requests."""
        return (1 - self.shard_share) ** self.requests_per_day

    @property
    def shards_touched_per_day(self) -> float:
        return self.shards * (1 - self.p_shard_untouched)

    @property
    def daily_batch_cost(self) -> Quantity:
        """Cost if every touched shard retrains once a day."""
        return (self.shards_touched_per_day * self.shard_cost).to("USD")


class PreferenceLabelBudgetScenario(BaseScenario):
    """Human preference labels for a reward model: ``comparisons_low`` to ``comparisons_high``
    at ``price_low`` to ``price_high`` per comparison."""
    comparisons_low: float
    comparisons_high: float
    price_low: Quantity
    price_high: Quantity

    @property
    def budget_low(self) -> Quantity:
        return (self.comparisons_low * self.price_low).to("USD")

    @property
    def budget_high(self) -> Quantity:
        return (self.comparisons_high * self.price_high).to("USD")

    @property
    def spread(self) -> float:
        return (self.budget_high / self.budget_low).to("").magnitude


class AutomationBiasScenario(BaseScenario):
    """A human reviews an AI's output; the human accepts ``miss_acceptance`` of the AI's misses.

    The best case assumes the reviewer catches every other miss and never
    overturns a correct detection, so team sensitivity is at most one minus the
    accepted misses.
    """
    human_sensitivity: float
    ai_sensitivity: float
    miss_acceptance: float
    acceptance_low: float
    acceptance_high: float

    @property
    def ai_miss_rate(self) -> float:
        return 1 - self.ai_sensitivity

    def accepted_misses(self, acceptance: Optional[float] = None) -> float:
        a = self.miss_acceptance if acceptance is None else acceptance
        return self.ai_miss_rate * a

    @property
    def team_sensitivity_bound(self) -> float:
        return 1 - self.accepted_misses()

    @property
    def gain_over_ai(self) -> float:
        return self.team_sensitivity_bound - self.ai_sensitivity


class PerRequestFailureVolumeScenario(BaseScenario):
    """A per-request failure probability at a daily request volume: the expected failures a day."""
    failure_probability: float
    daily_requests: float

    @property
    def daily_failures(self) -> float:
        return self.failure_probability * self.daily_requests


class GradeStandardizationScenario(BaseScenario):
    """A grade-standardization outcome from a published matrix of calculated against assessed grades.

    ``matrix`` is a Literature series with one row per assessed grade: an
    ``entries`` count and row percentages per calculated grade, columns in
    ``grades`` order from highest to lowest. ``candidates`` counts candidates by
    the number of their grades lowered.
    """
    matrix: Any
    candidates: Any
    grades: List[str]
    stated_entries: Optional[float] = None      # the source's stated n for the matrix
    stated_candidates: Optional[float] = None   # the source's stated n for the candidate table

    def split(self, grade: str):
        """Row percentages for one assessed grade: (higher, same, one lower, two or more lower)."""
        row = next(r for r in self.matrix.entries if r["cag"] == grade)
        i = self.grades.index(grade)
        cells = [row[g] for g in self.grades]
        one_lower = cells[i + 1] if i + 1 < len(cells) else 0
        return (sum(cells[:i]), cells[i], one_lower, sum(cells[i + 2:]))

    @property
    def total_entries(self) -> float:
        return sum(r["entries"] for r in self.matrix.entries)

    @property
    def entries_lowered_pct(self) -> float:
        """Entry-weighted share lowered (percent), from the rounded row percentages and row counts."""
        tot = sum(r["entries"] * sum(self.split(r["cag"])[2:]) for r in self.matrix.entries)
        return tot / self.total_entries

    @property
    def row_entries_sum(self) -> float:
        """Sum of the matrix rows' entry counts (may differ from the stated n)."""
        return self.total_entries

    @property
    def candidate_rows_sum(self) -> float:
        """Sum of the candidate table's rounded rows (may differ from the stated n)."""
        return sum(r["candidates"] for r in self.candidates.entries)

    @property
    def total_candidates(self) -> float:
        """The stated candidate count when given, else the sum of the rows."""
        return self.stated_candidates if self.stated_candidates is not None else self.candidate_rows_sum

    def candidates_with(self, lowered: int) -> float:
        return next(r["candidates"] for r in self.candidates.entries if r["grades_lowered"] == lowered)

    def candidate_share(self, lowered: int) -> float:
        return self.candidates_with(lowered) / self.total_candidates

    @property
    def at_least_one_lowered_share(self) -> float:
        return 1 - self.candidate_share(0)


class DpTrainingPlanScenario(BaseScenario):
    """An illustrative differential-privacy planning scenario for one training run of ``base_cost``.

    Each privacy budget is an assumed accountant result; the stronger budget is
    assumed to need ``cost_multiplier`` times the training, with the stated
    accuracy drops and overheads (percent) as planning assumptions.
    """
    epsilon_strong: float
    epsilon_moderate: float
    epsilon_weak: float
    base_cost: Quantity
    cost_multiplier: float
    accuracy_drop_strong: float      # percent
    accuracy_drop_moderate: float    # percent
    clipped_share: float             # percent of per-example gradients clipped
    overhead_moderate: float         # percent

    @property
    def cost_strong(self) -> Quantity:
        return (self.base_cost * self.cost_multiplier).to("USD")
# --- Durable execution (added 2026-10-20) ----------------------------------------
# What a runtime pays to survive a crash: one group commit, full snapshots
# against deltas, checkpoint strides on a fixed crash trace, and the cost of
# keeping recovery and replay data. Solvers in engine/solvers/durability.py.

class GroupCommitScenario(BaseScenario):
    """``batch_size`` intents gathered for up to ``batch_timer`` and acknowledged by one ``flush_time``."""
    batch_timer: Quantity
    flush_time: Quantity
    batch_size: int

    @property
    def result(self):
        from .solvers import GroupCommitModel
        return GroupCommitModel().solve(self.batch_timer, self.flush_time, self.batch_size)


class SnapshotVolumeScenario(BaseScenario):
    """A trajectory of ``turns`` turns growing ``state_growth`` per turn, checkpointed every ``stride``."""
    turns: int
    stride: int
    state_growth: Quantity

    @property
    def result(self):
        from .solvers import SnapshotVolumeModel
        return SnapshotVolumeModel().solve(self.turns, self.stride, self.state_growth)


class CheckpointTraceScenario(BaseScenario):
    """Checkpoint ``strides`` priced on ``turns`` turns that crash after the turns in ``crashes``."""
    turns: int
    crashes: List[int]
    strides: List[int]
    pause: Quantity                 # per checkpoint, including durable publication
    restore: Quantity               # per crash
    rebuild_per_turn: Quantity      # read, validate, and fold one logged turn
    reexecute_per_turn: Quantity    # re-run one unlogged turn's model call and tools

    @property
    def result(self):
        from .solvers import CheckpointTraceModel
        return CheckpointTraceModel().solve(self.turns, self.crashes, self.strides, self.pause,
                                            self.restore, self.rebuild_per_turn,
                                            self.reexecute_per_turn)

    def stride_overhead(self, stride: float, lost_turn_cost: Quantity) -> Quantity:
        """Expected overhead per turn at ``stride`` on this trace's crash rate (seconds)."""
        from ..physics.reliability import calc_stride_overhead
        return calc_stride_overhead(self.pause, stride, len(self.crashes) / self.turns, lost_turn_cost)


class RetentionCostScenario(BaseScenario):
    """One trajectory's log (``turns`` x envelope and payload) and workspace snapshot kept for ``period``."""
    turns: int
    envelope: Quantity              # event envelopes per turn
    payload: Quantity               # delivered inputs and result bodies per turn
    workspace: Quantity             # one immutable workspace snapshot
    rate: Quantity                  # object storage, USD per byte-time
    period: Quantity
    attempt_cost: Quantity          # USD of one attempt

    @property
    def result(self):
        from .solvers import RetentionCostModel
        return RetentionCostModel().solve(self.turns, self.envelope, self.payload, self.workspace,
                                          self.rate, self.period, self.attempt_cost)
# --- Failure recovery (added 2026-10-20) --------------------------------------
# What the runtime does once an effect has failed: one repair against unwind,
# checks ahead of a pivot, a loop stopped late, retries through nested layers,
# a capped dependency, a jittered retry cohort, a cleanup graph, a breaker
# window, and published retry-sustained outages. Solvers in
# engine/solvers/recovery.py.

class RepairBreakEvenScenario(BaseScenario):
    """One bounded repair attempt against compensating and rebuilding to the same endpoint."""
    unwind_time: Quantity            # compensation
    rebuild_time: Quantity           # rebuild to the same readiness endpoint
    rebuild_input: int               # rebuild model tokens
    rebuild_output: int
    repair_input: int                # the whole isolated repair request
    repair_output: int
    startup_time: Quantity           # pre-output processing, not a measured time to first token
    output_rate: Quantity            # tokens per second
    validation_time: Quantity        # local validation, charged on success and failure
    input_price: Quantity            # USD per token
    output_price: Quantity
    assumed_success: float           # a policy comparison, not a measurement

    @property
    def result(self):
        from .solvers import RepairBreakEvenModel
        return RepairBreakEvenModel().solve(
            self.unwind_time, self.rebuild_time, self.rebuild_input, self.rebuild_output,
            self.repair_input, self.repair_output, self.startup_time, self.output_rate,
            self.validation_time, self.input_price, self.output_price, self.assumed_success)


class IrreversibleLossScenario(BaseScenario):
    """A workload's wrong-change risk priced with its pre-pivot check, from its agent profile."""
    workload: Any                    # a RemediationWorkload (Agents.*)
    catch_step: float                # catch-rate increment for the sensitivity

    @property
    def attempt_cost(self) -> Quantity:
        w = self.workload
        return (w.model_cost_per_attempt + w.environment_cost_per_attempt
                + w.assessment_cost_per_attempt + w.approval_cost_per_attempt).to("USD")

    @property
    def result(self):
        from .solvers import IrreversibleLossModel
        w = self.workload
        return IrreversibleLossModel().solve(
            w.wrong_change_probability, w.wrong_change_impact, w.canary_catch_rate,
            w.canary_recovery_fraction, self.attempt_cost, self.catch_step)


class UndetectedLoopScenario(BaseScenario):
    """An append-only loop that resends its history, run to a turn ceiling or stopped early."""
    first_input: int                 # tokens
    output_per_turn: int
    observation_per_turn: int        # appended error observation
    turn_time: Quantity              # complete serial turn
    ceiling: int                     # turns, counted from the loop's start
    stop_after: int                  # intervention after this completed turn
    context_window: int              # combined input and output capacity, tokens
    input_price: Quantity            # USD per token
    output_price: Quantity

    @property
    def result(self):
        from .solvers import UndetectedLoopModel
        return UndetectedLoopModel().solve(
            self.first_input, self.output_per_turn, self.observation_per_turn, self.turn_time,
            self.ceiling, self.stop_after, self.context_window, self.input_price, self.output_price)


class RetryAmplificationScenario(BaseScenario):
    """Nested retry layers, each with its own allowance, under one failing logical operation."""
    layer_names: List[str]           # from the caller down to the dependency
    attempts_per_layer: int          # one try plus retries

    @property
    def result(self):
        from .solvers import RetryAmplificationModel
        return RetryAmplificationModel().solve(self.attempts_per_layer, len(self.layer_names))


class BulkheadAdmissionScenario(BaseScenario):
    """One burst of blocking calls to a stalled dependency, uncapped and with a bulkhead."""
    arrivals: int                    # blocking calls in the burst
    workers: int                     # shared pool
    worker_cap: int                  # occupied workers the dependency may hold
    queue_cap: int                   # waiting requests, which hold no worker

    @property
    def result(self):
        from .solvers import BulkheadAdmissionModel
        return BulkheadAdmissionModel().solve(self.arrivals, self.workers, self.worker_cap, self.queue_cap)


class RetryJitterScenario(BaseScenario):
    """A cohort of eligible retries released together, synchronized or with full jitter."""
    retries: int
    burst_window: Quantity           # stipulated window if all arrive together
    jitter_span: Quantity            # each retry's arrival drawn uniformly over this span

    @property
    def result(self):
        from .solvers import RetryJitterModel
        return RetryJitterModel().solve(self.retries, self.burst_window, self.jitter_span)


class CleanupGraphScenario(BaseScenario):
    """Registered cleanup operations with durations and dependencies, one outcome unresolved."""
    durations: Dict[str, Quantity]   # time once admitted
    depends_on: Dict[str, List[str]]
    unresolved: str                  # the operation whose reply is lost

    @property
    def result(self):
        from .solvers import CleanupGraphModel
        return CleanupGraphModel().solve(self.durations, self.depends_on, self.unresolved)


class BreakerWindowScenario(BaseScenario):
    """One circuit-breaker window of classified outcomes and its trip rule."""
    outcomes: Dict[str, int]         # outcome class -> count in the window
    failure_classes: List[str]       # classes that show the dependency cannot do work
    success_class: str
    threshold: float                 # trip at failure fraction >= threshold
    min_outcomes: int

    @property
    def result(self):
        from .solvers import BreakerWindowModel
        return BreakerWindowModel().solve(self.outcomes, self.failure_classes, self.success_class,
                                          self.threshold, self.min_outcomes)


class RetrySustainedOutagesScenario(BaseScenario):
    """Published incidents (``Literature.Series``) counted by whether retries sustained them."""
    series_name: str
    threshold_hours: float           # count retry-sustained outages longer than this

    @property
    def rows(self):
        from ..literature.registry import Literature
        return getattr(Literature, self.series_name).entries

    @property
    def result(self):
        from .solvers import RetrySustainedOutagesModel
        return RetrySustainedOutagesModel().solve(self.rows, "incident", "retry_sustained",
                                                  "duration_h", self.threshold_hours)
# --- Agent evaluation (added 2026-10-20) ---------------------------------------
# What a set of graded trials supports about a configuration; the solvers live
# in engine/solvers/acceptance.py. Repeated-run consistency over k recurrences
# is a HorizonCompoundingScenario (recurrences compound as turns do).
class UnresolvedVerdictScenario(BaseScenario):
    """Trials that passed, failed, or ended pending or unknown at the cutoff."""
    passed: int
    failed: int
    unresolved: int

    @property
    def result(self):
        from .solvers import VerdictBoundsModel
        return VerdictBoundsModel().solve(self.passed, self.failed, self.unresolved)


class GraderCalibrationScenario(BaseScenario):
    """A grader's verdicts on results an independent reference assessed."""
    reference_failures: int
    accepted_failures: int           # reference failures the grader accepted
    reference_passes: int
    accepted_passes: int             # reference passes the grader accepted
    confidence: float = 0.95

    @property
    def result(self):
        from .solvers import GraderCalibrationModel
        return GraderCalibrationModel().solve(self.reference_failures, self.accepted_failures,
                                              self.reference_passes, self.accepted_passes, self.confidence)


class TaskRepeatAllocationScenario(BaseScenario):
    """A run budget split between new tasks and repeated runs, with and without fixture cost."""
    mean_rate: float                 # pilot mean task acceptance probability
    between_task_variance: float     # sigma_task^2
    budget_runs: int                 # graded runs (run-equivalents with fixture cost)
    free_runs: List[int]             # runs per task compared when new tasks are free
    fixture_cost_runs: int           # a new task's fixture, in graded-run equivalents
    max_runs: int                    # largest runs per task searched

    @property
    def result(self):
        from .solvers import TaskRepeatAllocationModel
        return TaskRepeatAllocationModel().solve(self.mean_rate, self.between_task_variance, self.budget_runs,
                                                 self.free_runs, self.fixture_cost_runs, self.max_runs)


class BinomialIntervalsScenario(BaseScenario):
    """Several success counts out of one number of independent tasks."""
    trials: int
    successes: List[int]
    confidence: float = 0.95

    @property
    def result(self):
        from .solvers import BinomialIntervalsModel
        return BinomialIntervalsModel().solve(self.trials, self.successes, self.confidence)


class ProportionPlanningScenario(BaseScenario):
    """Planning targets for one acceptance rate and for two independent arms."""
    rate: float                      # planning acceptance probability (0.5 is the worst case)
    planning_tasks: int              # task count whose half-width is quoted
    target_half_width: float
    differences: List[float]         # detectable differences between two arms
    significance: float              # two-sided alpha the detection quantile stands for
    power: float                     # power the second quantile stands for
    z_alpha: float                   # rounded quantile used in the planning formula
    z_beta: float
    confidence: float = 0.95

    @property
    def result(self):
        from .solvers import ProportionPlanningModel
        return ProportionPlanningModel().solve(self.rate, self.planning_tasks, self.target_half_width,
                                               self.differences, self.z_alpha, self.z_beta, self.confidence)


class PairedComparisonScenario(BaseScenario):
    """One candidate and one baseline trial on each sampled task, outcomes kept in pairs."""
    both: int                        # tasks both configurations accepted
    candidate_only: int
    baseline_only: int
    neither: int
    tolerance: float                 # noninferiority tolerance, as a fraction
    confidence: float = 0.95

    @property
    def result(self):
        from .solvers import PairedComparisonModel
        return PairedComparisonModel().solve(self.both, self.candidate_only, self.baseline_only, self.neither,
                                             self.tolerance, self.confidence)


class ScaffoldSpreadScenario(BaseScenario):
    """One leaderboard's models, each under several agent scaffolds (a ``Literature`` series)."""
    series_name: str
    min_scaffolds: int

    @property
    def rows(self):
        from ..literature.registry import Literature
        return getattr(Literature, self.series_name).entries

    @property
    def result(self):
        from .solvers import ScaffoldSpreadModel
        return ScaffoldSpreadModel().solve(self.rows, self.min_scaffolds)


class LeaderboardTiesScenario(BaseScenario):
    """The top of one evaluator's leaderboard with reported standard errors (a ``Literature`` series)."""
    series_name: str
    top: int
    confidence: float = 0.95

    @property
    def rows(self):
        from ..literature.registry import Literature
        return getattr(Literature, self.series_name).entries

    @property
    def result(self):
        from .solvers import LeaderboardTiesModel
        return LeaderboardTiesModel().solve(self.rows, self.top, self.confidence)


class TaskMixConsistencyScenario(BaseScenario):
    """Equally weighted task types and a number of runs that must all pass."""
    task_pass_rates: List[float]
    repetitions: int

    @property
    def result(self):
        from .solvers import TaskMixConsistencyModel
        return TaskMixConsistencyModel().solve(self.task_pass_rates, self.repetitions)


class CappedSelectionCostScenario(BaseScenario):
    """Candidates for one fixed task, an exact checker, and a cap on candidates."""
    pass_rate: float
    cap: int
    generation_cost: Quantity        # USD per candidate
    check_cost: Quantity             # USD per candidate

    @property
    def result(self):
        from .solvers import CappedSelectionCostModel
        return CappedSelectionCostModel().solve(self.pass_rate, self.cap, self.generation_cost, self.check_cost)


# --- Sustainability (added 2026-10-20) ---------------------------------------

def _rack_it_ratio(rack) -> float:
    """IT load over accelerator power for a rack profile (hosts, switches, conversion inside IT)."""
    return (rack.it_power / rack.accelerator_power).to("").magnitude


class FacilityPowerScenario(BaseScenario):
    """``devices`` accelerators at ``device_power`` in racks shaped like ``rack``, at ``pue``."""
    devices: float
    device_power: Quantity
    rack: Any                 # a Systems.Racks profile: supplies the IT-to-accelerator ratio
    pue: float

    @property
    def it_ratio(self) -> float:
        return _rack_it_ratio(self.rack)

    @property
    def result(self):
        from .solvers import FacilityPowerModel
        return FacilityPowerModel().solve(self.devices, self.device_power, self.it_ratio, self.pue)


class RunFootprintScenario(BaseScenario):
    """A training run of ``device_hours`` on ``devices`` of ``device``: energy, carbon, embodied carbon, water."""
    device_hours: Quantity
    devices: float
    device: HardwareNode      # rated power (TDP) and embodied carbon per device
    rack: Any                 # a Systems.Racks profile (IT-to-accelerator ratio)
    pue: float
    grids: Dict[str, Any]     # named Infrastructure.Grids profiles
    service_life: Quantity
    wue: Dict[str, Quantity] = {}

    @property
    def it_ratio(self) -> float:
        return _rack_it_ratio(self.rack)

    @property
    def result(self):
        from .solvers import RunFootprintModel
        return RunFootprintModel().solve(
            self.device_hours, self.devices, self.device.tdp, self.it_ratio, self.pue,
            {k: g.carbon_intensity for k, g in self.grids.items()},
            self.device.embodied_carbon, self.service_life, self.wue,
        )


class WaterUseScenario(BaseScenario):
    """An IT load held for ``period`` at each named WUE."""
    it_power: Quantity
    period: Quantity
    wue: Dict[str, Quantity]

    @property
    def it_energy(self) -> Quantity:
        return (self.it_power * self.period).to("kWh")

    @property
    def result(self):
        from .solvers import WaterUseModel
        return WaterUseModel().solve(self.it_energy, self.wue)


class PueRetrofitScenario(BaseScenario):
    """A fixed IT load moved from ``pue_before`` to ``pue_after`` for ``period`` at an electricity price."""
    it_power: Quantity
    pue_before: float
    pue_after: float
    period: Quantity
    electricity_price: Quantity

    @property
    def result(self):
        from .solvers import PueRetrofitModel
        return PueRetrofitModel().solve(self.it_power, self.pue_before, self.pue_after, self.period,
                                        self.electricity_price)


class FacilityCarbonRateScenario(BaseScenario):
    """Named sites with one IT load, each at its own PUE and grid intensity."""
    it_power: Quantity
    pues: Dict[str, float]
    intensities: Dict[str, Quantity]

    @property
    def result(self):
        from .solvers import FacilityCarbonRateModel
        return FacilityCarbonRateModel().solve(self.it_power, self.pues, self.intensities)


class GridPlacementScenario(BaseScenario):
    """One facility ``energy`` placed on each named grid, with an optional fractional energy cut."""
    energy: Quantity
    grids: Dict[str, Any]
    energy_cut: float = 0.0

    @property
    def intensities(self) -> Dict[str, Quantity]:
        return {k: g.carbon_intensity for k, g in self.grids.items()}

    @property
    def result(self):
        from .solvers import GridPlacementModel
        return GridPlacementModel().solve(self.energy, self.intensities, self.energy_cut)


class EmbodiedPaybackScenario(BaseScenario):
    """A replacement for ``device`` that does its work with ``saved_fraction`` less energy, on named grids."""
    device: HardwareNode
    saved_fraction: float
    grids: Dict[str, Any]

    @property
    def result(self):
        from .solvers import EmbodiedPaybackModel
        return EmbodiedPaybackModel().solve(self.device.embodied_carbon, self.device.tdp, self.saved_fraction,
                                            {k: g.carbon_intensity for k, g in self.grids.items()})


class ServiceLifeExtensionScenario(BaseScenario):
    """A hardware service life extended from ``short_life`` to ``long_life``."""
    short_life: Quantity
    long_life: Quantity

    @property
    def result(self):
        from .solvers import ServiceLifeExtensionModel
        return ServiceLifeExtensionModel().solve(self.short_life, self.long_life)


class PowerCapScenario(BaseScenario):
    """A run needing ``required_power`` for ``planned_duration`` behind a ``feed_power`` grid feed."""
    required_power: Quantity
    feed_power: Quantity
    planned_duration: Quantity

    @property
    def result(self):
        from .solvers import PowerCapModel
        return PowerCapModel().solve(self.required_power, self.feed_power, self.planned_duration)


class RepeatedWorkEnergyScenario(BaseScenario):
    """``fleet``'s accelerators at rated power, rolled back by each named interval."""
    fleet: Fleet
    rollbacks: Dict[str, Quantity]

    @property
    def result(self):
        from .solvers import RepeatedWorkEnergyModel
        return RepeatedWorkEnergyModel().solve(self.fleet.total_accelerators, self.fleet.node.accelerator.tdp,
                                               self.rollbacks)


class ServingCrossoverScenario(BaseScenario):
    """Serving ``requests_per_day`` at ``energy_per_request`` against a one-time energy."""
    one_time_energy: Quantity
    requests_per_day: float
    energy_per_request: Quantity

    @property
    def result(self):
        from .solvers import ServingCrossoverModel
        return ServingCrossoverModel().solve(self.one_time_energy, self.requests_per_day, self.energy_per_request)


class CascadeEnergyScenario(BaseScenario):
    """An edge model on every input, escalating a fraction to a cloud model over a radio link."""
    edge_energy: Quantity
    transmit_energy: Quantity
    cloud_energy: Quantity
    escalation: float

    @property
    def result(self):
        from .solvers import CascadeEnergyModel
        return CascadeEnergyModel().solve(self.edge_energy, self.transmit_energy, self.cloud_energy, self.escalation)


class DutyCycledPowerScenario(BaseScenario):
    """An always-on detector, a stage active a share of the time, and a full model run per event."""
    always_on_power: Quantity
    stage_power: Quantity
    stage_activity: float
    full_power: Quantity
    event_rate: Quantity
    full_duration: Quantity

    @property
    def full_duty(self) -> float:
        return (self.event_rate * self.full_duration).to("").magnitude

    @property
    def result(self):
        from .solvers import DutyCycledPowerModel
        return DutyCycledPowerModel().solve(
            {"always_on": self.always_on_power, "stage": self.stage_power, "full": self.full_power},
            {"always_on": 1.0, "stage": self.stage_activity, "full": self.full_duty},
            self.full_power,
        )


class DataMovementEnergyScenario(BaseScenario):
    """One weight of ``bytes_per_param`` used for ``flops_per_param`` FLOPs at stated per-op energies."""
    energy_per_flop: Quantity
    energy_per_byte: Quantity
    flops_per_param: float
    bytes_per_param: Quantity
    operand_bytes: Quantity

    @property
    def result(self):
        from .solvers import DataMovementEnergyModel
        return DataMovementEnergyModel().solve(self.energy_per_flop, self.energy_per_byte, self.flops_per_param,
                                               self.bytes_per_param, self.operand_bytes)


class DemandReboundScenario(BaseScenario):
    """Per-unit energy scaled by a factor (or by a unit-price ratio) while volume scales by another."""
    volume_factor: float
    per_unit_factor: Optional[float] = None
    unit_price_before: Optional[float] = None   # used when per_unit_factor is None: energy tracks price
    unit_price_after: Optional[float] = None

    @property
    def unit_factor(self) -> float:
        if self.per_unit_factor is not None:
            return self.per_unit_factor
        return self.unit_price_after / self.unit_price_before

    @property
    def result(self):
        from .solvers import DemandReboundModel
        return DemandReboundModel().solve(self.unit_factor, self.volume_factor)


class DevelopmentSweepScenario(BaseScenario):
    """A sweep of ``candidates`` runs, each costing ``relative_cost`` of the final run."""
    candidates: float
    relative_cost: float

    @property
    def result(self):
        from .solvers import DevelopmentSweepModel
        return DevelopmentSweepModel().solve(self.candidates, self.relative_cost)


class RollingShareScenario(BaseScenario):
    """Hourly share of ``share_columns`` in ``total_columns`` of a Literature series, over rolling windows."""
    series_name: str
    time_column: str
    share_columns: List[str]
    total_columns: List[str]
    window: int

    @property
    def rows(self):
        from ..literature.registry import Literature
        return getattr(Literature.Series, self.series_name).entries

    @property
    def shares(self) -> List[float]:
        """Hourly share, percent."""
        return [100 * sum(r[c] for c in self.share_columns) / sum(r[c] for c in self.total_columns)
                for r in self.rows]

    @property
    def times(self) -> List[str]:
        return [r[self.time_column] for r in self.rows]

    @property
    def result(self):
        from .solvers import RollingWindowModel
        return RollingWindowModel().solve(self.shares, self.window)
# --- Spatial memory: estimation, belief expiry, and volumetric maps (added 2026-10-20) ---
# A site scenario (``Embodied.Scenario.*``) supplies the machine, the
# clearances, the walking speed, and the observation age; the solvers live in
# ``engine/solvers/estimation.py`` and ``engine/solvers/mapping.py``.

class RangeFusionScenario(BaseScenario):
    """The range to ``site``'s rack-end upright, predicted over one lidar sweep and updated by the lidar, then ``stereo``; the stereo update misdated by ``age``'s arrival stamp at ``speed``."""
    site: Any                                   # EmbodiedSiteScenario
    stereo: Any                                 # StereoDepthErrorScenario
    age: Any                                    # ObservationAgeScenario
    prior_sigma: Quantity
    speed_sigma: Quantity
    comparison_scale: float = 2.0

    @property
    def result(self):
        from .solvers import RangeFusionModel
        lidar = self.site.machine.lidar
        return RangeFusionModel().solve(self.prior_sigma, self.speed_sigma, 1 / lidar.scan_rate,
                                        lidar.range_precision_max, self.stereo.result.sigma_target,
                                        self.age.result.arrival_travel, self.comparison_scale)


class EstimatorConsistencyScenario(BaseScenario):
    """A qualification campaign of ``runs`` scalar estimates, and single-update gates against ``fusion``'s misdated update."""
    fusion: Any                                 # RangeFusionScenario
    runs: int
    dimension: int = 1
    confidence: float = 0.95
    gate_confidences: List[float] = [0.95, 0.99]

    @property
    def result(self):
        from .solvers import EstimatorConsistencyModel
        return EstimatorConsistencyModel().solve(self.runs, self.dimension, self.confidence,
                                                 self.gate_confidences, self.fusion.result.nis_expected)


class ReplayBufferScenario(BaseScenario):
    """A rewind buffer at ``site.machine``'s IMU rate reaching back one observation age."""
    site: Any                                   # EmbodiedSiteScenario

    @property
    def result(self):
        from .solvers import ReplayBufferModel
        return ReplayBufferModel().solve(self.site.perception_age, self.site.machine.imu.sample_rate)


class InertialCoastScenario(BaseScenario):
    """``site.machine``'s IMU coasting ``duration`` without a fix, seen at the rack-end distance."""
    site: Any                                   # EmbodiedSiteScenario
    duration: Quantity

    @property
    def result(self):
        from .solvers import InertialCoastModel
        imu, s = self.site.machine.imu, self.site
        return InertialCoastModel().solve(imu.gyro_noise_density, imu.gyro_zero_rate_offset, imu.accel_noise_density,
                                          imu.accel_zero_g_offset, self.duration, s.d_clear, s.delta_loc)


class OdometryDriftScenario(BaseScenario):
    """A square-path odometry benchmark's return errors, priced against ``site``'s localization bound at its aisle speed and length."""
    site: Any                                   # EmbodiedSiteScenario
    square_side: Quantity
    legs: int
    return_error_uncalibrated: Quantity
    return_error_calibrated: Quantity
    run_sigma: Quantity
    test_speed: Quantity

    @property
    def path_length(self) -> Quantity:
        return self.legs * self.square_side

    @property
    def result(self):
        from .solvers import OdometryDriftModel
        s = self.site
        return OdometryDriftModel().solve(self.return_error_uncalibrated, self.return_error_calibrated,
                                          self.path_length, s.delta_loc, s.v_aisle, s.aisle_length, self.test_speed)


class OdometryLeaderboardScenario(BaseScenario):
    """Ranked odometry methods (``entries``: translation_err_pct, sensor_group) against ``site``'s bound and aisle."""
    site: Any                                   # EmbodiedSiteScenario
    entries: List[Any]

    @property
    def result(self):
        from .solvers import OdometryLeaderboardModel
        return OdometryLeaderboardModel().solve(self.entries, self.site.delta_loc, self.site.aisle_length)


class BeliefExpiryScenario(BaseScenario):
    """A belief with initial error, growth law, and task tolerance; optionally admitted at ``use_age`` under ``age_bound``."""
    initial_error: Quantity
    growth_rate: Quantity
    tolerance: Quantity
    growth_acceleration: Optional[Quantity] = None
    use_age: Optional[Quantity] = None
    age_bound: Optional[Quantity] = None

    @property
    def result(self):
        from .solvers import BeliefExpiryModel
        return BeliefExpiryModel().solve(self.initial_error, self.growth_rate, self.tolerance,
                                         self.growth_acceleration, self.use_age, self.age_bound)


class StaleTransformScenario(BaseScenario):
    """``belief``'s cached camera edge composed with the current wrist pose, one ``site.machine`` wrist-camera frame old."""
    site: Any                                   # EmbodiedSiteScenario
    belief: Any                                 # BeliefExpiryScenario
    lever: Quantity
    angular_rate: Quantity

    @property
    def result(self):
        from .solvers import StaleTransformModel
        return StaleTransformModel().solve(self.belief.result, self.lever, self.angular_rate,
                                           1 / self.site.machine.wrist_camera.frame_rate)


class FreeSpaceExpiryScenario(BaseScenario):
    """``site``'s rack end: free space ahead (clear distance) and abreast (side clearance) eroding at the walking speed."""
    site: Any                                   # EmbodiedSiteScenario
    uniform_timeout: Optional[Quantity] = None

    @property
    def result(self):
        from .solvers import FreeSpaceExpiryModel
        s, rm = self.site, self.site.machine
        return FreeSpaceExpiryModel().solve(s.d_clear, s.side_clearance, s.delta_loc, s.delta_margin,
                                            s.v_human_approach, s.perception_age, 1 / rm.nav_camera.frame_rate,
                                            rm.base.max_velocity, s.a_brake, s.tau_delay, self.uniform_timeout)


class EpochGapScenario(BaseScenario):
    """Named acquisition ``epochs`` in one cycle; the ``pair`` to fuse is projected across its gap at ``speed``."""
    epochs: Dict[str, Quantity]
    pair: List[str]
    speed: Quantity

    @property
    def result(self):
        from .solvers import EpochGapModel
        a, b = self.pair
        return EpochGapModel().solve(self.epochs[a], self.epochs[b], self.speed)


class SkewedRecordsScenario(BaseScenario):
    """Two records ``skew`` apart, compared at ``site``: the base at its drive limit and a person at the walking speed close on each other."""
    site: Any                                   # EmbodiedSiteScenario
    skew: Quantity
    separation: Quantity

    @property
    def closing_speed(self) -> Quantity:
        return self.site.machine.base.max_velocity + self.site.v_human_approach

    @property
    def result(self):
        from .solvers import EpochGapModel
        s = self.site
        return EpochGapModel().solve(0 * self.skew, self.skew, self.closing_speed, self.separation,
                                     (s.delta_loc, s.delta_margin))


class SensorBiasDriftScenario(BaseScenario):
    """A steady ``reading`` whose sensor bias drifts at ``drift_rate`` for ``span``, while the estimator reports ``reported_sigma``."""
    reading: Quantity
    reported_sigma: Quantity
    drift_rate: Quantity
    span: Quantity
    excitation: Optional[Quantity] = None       # a deliberate contact break that would expose the bias

    @property
    def result(self):
        from .solvers import SensorBiasDriftModel
        return SensorBiasDriftModel().solve(self.reading, self.drift_rate, self.span)


class FrozenTransformScenario(BaseScenario):
    """``site``'s conveyor object tracked by a camera lost at ``loss_time``; its last transform republished every ``republish_period``."""
    site: Any                                   # EmbodiedSiteScenario
    republish_period: Quantity
    freshness_threshold: Quantity
    loss_time: Quantity
    window: Quantity

    @property
    def result(self):
        from .solvers import FrozenTransformModel
        s = self.site
        return FrozenTransformModel().solve(s.v_conveyor, self.window, s.grasp_tolerance, self.republish_period,
                                            self.freshness_threshold)


class TrackConfirmationScenario(BaseScenario):
    """A tracker that outputs a track after ``min_hits`` hits, on ``site.machine``'s navigation camera at its drive limit."""
    site: Any                                   # EmbodiedSiteScenario
    min_hits: int

    @property
    def result(self):
        from .solvers import TrackConfirmationModel
        rm = self.site.machine
        return TrackConfirmationModel().solve(self.min_hits, rm.nav_camera.frame_rate, rm.base.max_velocity)


class ClearanceQueryScenario(BaseScenario):
    """Sphere-hull clearance queries on ``site.machine``'s permission controller, and a snapshot of the stopping corridor."""
    site: Any                                   # EmbodiedSiteScenario
    collision_allocation: Quantity
    hulls: int
    fewest_hulls: int
    hull_radius: Quantity
    esdf_cell_bytes: Quantity
    occupancy_cell_bytes: Quantity
    map_edge: Quantity
    slab_height: Quantity
    pitches: List[Quantity]
    trilinear_reads: int

    @property
    def slab_volume(self) -> Quantity:
        return (self.site.aisle_width * self.site.d_clear * self.slab_height).to("m**3")

    @property
    def result(self):
        from .solvers import ClearanceQueryModel
        s, rm = self.site, self.site.machine
        return ClearanceQueryModel().solve(rm.permission_rate, self.collision_allocation, self.hulls,
                                           self.hull_radius, s.delta_margin, rm.permission_mcu.memory.sram_bandwidth,
                                           self.esdf_cell_bytes, self.occupancy_cell_bytes, self.map_edge,
                                           self.slab_volume, self.pitches, self.trilinear_reads, s.enforcer_wcet)


class HostMapSizingScenario(BaseScenario):
    """A workcell of ``extent`` mapped at ``pitch``: dense grid against a hashed band of ``band_half_width`` around ``surface_area``."""
    extent: List[Quantity]
    pitch: Quantity
    voxel_bytes: Quantity
    surface_area: Quantity
    band_half_width: Quantity
    block_edge: int

    @property
    def result(self):
        from .solvers import HostMapSizingModel
        return HostMapSizingModel().solve(self.extent, self.pitch, self.voxel_bytes, self.surface_area,
                                          self.band_half_width, self.block_edge)


class RaymarchTrafficScenario(BaseScenario):
    """One depth stream ray-marched into a map on ``site.machine``'s host SoC."""
    site: Any                                   # EmbodiedSiteScenario
    width_px: int
    height_px: int
    frame_rate: Quantity
    ray_length: Quantity
    pitch: Quantity
    bytes_per_step: Quantity

    @property
    def result(self):
        from .solvers import RaymarchTrafficModel
        return RaymarchTrafficModel().solve(self.width_px, self.height_px, self.frame_rate, self.ray_length,
                                            self.pitch, self.bytes_per_step,
                                            self.site.machine.brain_soc.memory.bandwidth)


class EncoderRateScenario(BaseScenario):
    """An encoder of ``flop_per_frame`` on every ``site.machine`` navigation frame, against the host SoC's dense FP16 peak."""
    site: Any                                   # EmbodiedSiteScenario
    flop_per_frame: Quantity
    keyframe_rate_low: Optional[Quantity] = None
    keyframe_rate_high: Optional[Quantity] = None

    @property
    def result(self):
        from .solvers import EncoderRateModel
        rm = self.site.machine
        return EncoderRateModel().solve(self.flop_per_frame, rm.nav_camera.frame_rate,
                                        rm.brain_soc.compute.precision_flops["fp16"])
# --- Grounded intent: the intent lease and its gates (added 2026-10-20) --------

class IntentCadenceScenario(BaseScenario):
    """``site``'s conveyor target proposed by ``machine``'s intent model: evidence horizons against
    the model's cadence, a host stall of ``stall``, and the stop onset after expiry.

    ``target_speed`` defaults to the conveyor's; set it for a slower drift.
    """
    site: Any                                    # EmbodiedSiteScenario (tolerance, error, belt, slip, tick, bus, onset)
    machine: Any                                 # EmbodiedMobileManipulator (intent rate and latency)
    stall: Quantity
    target_speed: Optional[Quantity] = None

    @property
    def speed(self) -> Quantity:
        return self.site.v_conveyor if self.target_speed is None else self.target_speed

    @property
    def result(self):
        from .solvers import IntentCadenceModel
        s, rm = self.site, self.machine
        return IntentCadenceModel().solve(
            s.grasp_tolerance, s.grasp_initial_error, self.speed, s.a_conveyor_slip,
            rm.intent_rate, rm.intent_inference_latency, self.stall, s.t_tick, s.t_bus, s.t_brake_onset)


class DynamicReachScenario(BaseScenario):
    """A target ``target_distance`` from the gripper at ``cadence.machine``'s TCP speed ceiling,
    against the cadence scenario's evidence windows."""
    cadence: IntentCadenceScenario
    target_distance: Quantity

    @property
    def speed_ceiling(self) -> Quantity:
        return self.cadence.machine.tcp_speed_limit

    @property
    def result(self):
        from .solvers import DynamicReachModel
        c = self.cadence.result
        return DynamicReachModel().solve(self.speed_ceiling, self.target_distance,
                                         c.tracked_horizon, c.untracked_horizon)


class ForwardOccupancyScenario(BaseScenario):
    """A coworker's forward occupancy set at ``cadence.site``'s design approach speed, screened over
    the cadence scenario's evidence windows and over a ``forecast`` horizon, against the aisle."""
    cadence: IntentCadenceScenario
    forecast: Quantity

    @property
    def result(self):
        from .solvers import ForwardOccupancyModel
        s, c = self.cadence.site, self.cadence.result
        return ForwardOccupancyModel().solve(s.v_human_approach, s.aisle_width, c.untracked_horizon,
                                             c.tracked_horizon, self.forecast)


class IntentLeaseExpiryScenario(BaseScenario):
    """The intent lease for the cadence scenario's target: its evidence horizons, an
    ``administrative_lease``, and the stopping deadline from ``clearance`` at the TCP speed ceiling,
    with the arm's credible deceleration from its handbook worst-case stop."""
    cadence: IntentCadenceScenario
    administrative_lease: Quantity
    clearance: Quantity

    @property
    def result(self):
        from .solvers import IntentLeaseExpiryModel
        c, rm = self.cadence.result, self.cadence.machine
        return IntentLeaseExpiryModel().solve(
            c.untracked_horizon, c.tracked_horizon, self.administrative_lease, c.stop_onset,
            rm.tcp_speed_limit, rm.arm.cartesian_velocity_limit, rm.arm.emergency_stop_time,
            self.clearance)


class ReferentSetScenario(BaseScenario):
    """Raw per-box ``scores`` of retained candidates screened by a conformal threshold calibrated on
    ``calibration_count`` scenes at miss rate ``alpha``, whose calibration score is ``threshold_score``."""
    labels: List[str]
    scores: List[float]
    calibration_count: int
    alpha: float
    threshold_score: float                       # qhat

    @property
    def result(self):
        from .solvers import ReferentSetModel
        return ReferentSetModel().solve(self.scores, self.calibration_count, self.alpha, self.threshold_score)


class ConfidenceRegionScenario(BaseScenario):
    """A Gaussian position estimate with principal standard deviations ``sigmas``: its ``coverage``
    region against ``site``'s grasp tolerance."""
    site: Any                                    # EmbodiedSiteScenario (grasp tolerance)
    sigmas: List[Quantity]
    coverage: float

    @property
    def result(self):
        from .solvers import ConfidenceRegionModel
        return ConfidenceRegionModel().solve(self.sigmas, self.coverage, self.site.grasp_tolerance)


class GoalSetMembershipScenario(BaseScenario):
    """Terminal TCP ``offsets`` along one axis tested against a goal set whose semi-axes all equal
    ``site``'s grasp tolerance: the normalized ellipsoidal distance, inside when at most one."""
    site: Any                                    # EmbodiedSiteScenario (grasp tolerance)
    offsets: List[Quantity]

    @property
    def norms(self) -> List[float]:
        return [(d / self.site.grasp_tolerance).to("").magnitude for d in self.offsets]

    @property
    def inside(self) -> List[bool]:
        return [n <= 1.0 for n in self.norms]

# --- Data selection: corpus repetition, deduplication, coresets, echoing, labels, and cost (added 2026-10-20) ---


class CorpusRepetitionScenario(BaseScenario):
    """A training ``run`` (``ClusterTrainingTimeScenario``) set against the unique tokens of its
    workload's own corpus.

    The corpus holds ``training_dataset_size / bytes_per_token`` unique tokens. The
    run processes ``training_ops / (flops_per_param_token x parameters)`` tokens,
    so it passes over the corpus that many times. A compute-optimal fit asks for
    ``tokens_per_param`` training tokens per parameter.
    """
    run: Any                           # ClusterTrainingTimeScenario
    bytes_per_token: Quantity          # bytes per token
    tokens_per_param: float
    flops_per_param_token: float

    @property
    def workload(self) -> Workload:
        return self.run.workload

    @property
    def hardware(self) -> HardwareNode:
        return self.run.hardware

    @property
    def accelerators(self) -> int:
        return self.run.n_devices

    @property
    def utilization(self) -> float:
        return self.run.efficiency

    @property
    def parameters(self) -> float:
        return self.workload.parameters.to("param").magnitude

    @property
    def corpus_bytes(self) -> Quantity:
        return self.workload.training_dataset_size

    @property
    def training_flops(self) -> Quantity:
        return self.workload.training_ops

    @property
    def unique_tokens(self) -> Quantity:
        return (self.corpus_bytes / self.bytes_per_token).to("count")

    @property
    def optimal_tokens(self) -> Quantity:
        """Training tokens the compute-optimal ratio asks for at this parameter count."""
        return Q_(self.tokens_per_param * self.parameters, "count")

    @property
    def shortfall(self) -> float:
        """Compute-optimal tokens over the corpus's unique tokens."""
        return (self.optimal_tokens / self.unique_tokens).to("").magnitude

    @property
    def unique_tokens_per_param(self) -> float:
        return self.unique_tokens.magnitude / self.parameters

    @property
    def tokens_processed(self) -> Quantity:
        return Q_(self.training_flops.to("flop").magnitude / (self.flops_per_param_token * self.parameters), "count")

    @property
    def trained_tokens_per_param(self) -> float:
        return self.tokens_processed.magnitude / self.parameters

    @property
    def passes(self) -> float:
        """Times the run read its corpus."""
        return (self.tokens_processed / self.unique_tokens).to("").magnitude

    @property
    def run_time(self) -> Quantity:
        return self.run.training_time.to("second")

    @property
    def run_energy(self) -> Quantity:
        """Accelerator energy at board power for the whole run."""
        return (self.accelerators * self.hardware.tdp * self.run_time).to("joule")

    @property
    def compute_optimal(self):
        """(parameters, tokens) that spend the same compute at ``tokens_per_param``."""
        from ..physics.scaling import calc_compute_optimal_allocation
        return calc_compute_optimal_allocation(self.training_flops.to("flop").magnitude,
                                               self.tokens_per_param, self.flops_per_param_token)


class AllocationGrowthScenario(BaseScenario):
    """A compute-optimal allocation ``X_opt ~ C^exponent`` when compute grows by ``compute_multiple``."""
    compute_multiple: float
    exponent: float

    @property
    def growth(self) -> float:
        """Fractional growth of the allocation."""
        from ..physics.scaling import calc_allocation_growth
        return calc_allocation_growth(self.compute_multiple, self.exponent)


class DuplicateRemovalScenario(BaseScenario):
    """Near-duplicate shares removed from ``run``'s corpus: the training time and energy each
    share returns, against one read of the corpus and the MinHash signatures of its documents."""
    run: CorpusRepetitionScenario
    duplicate_shares: Dict[str, float]   # corpus name -> fraction of documents removed
    documents: float
    signature_hashes: int
    hash_bytes: Quantity
    read_bandwidth: Quantity

    def time_saved(self, corpus: str) -> Quantity:
        return (self.duplicate_shares[corpus] * self.run.run_time).to("hour")

    def energy_saved(self, corpus: str) -> Quantity:
        return (self.duplicate_shares[corpus] * self.run.run_energy).to("kWh")

    @property
    def signature_bytes(self) -> Quantity:
        return (self.documents * self.signature_hashes * self.hash_bytes).to("GB")

    @property
    def read_time(self) -> Quantity:
        return (self.run.corpus_bytes / self.read_bandwidth).to("second")

    @property
    def all_pairs(self) -> float:
        """Pairwise comparisons an exhaustive search would make."""
        return self.documents * (self.documents - 1) / 2


class MinHashBandingScenario(BaseScenario):
    """A MinHash signature cut into ``bands`` bands of ``rows`` hashes over ``shingle_words``-word shingles."""
    bands: int
    rows: int
    shingle_words: Optional[int] = None    # None when the features are not word shingles (added 2026-10-20)
    probe_similarities: List[float] = []   # Jaccard similarities at which to read the candidate chance
    # Optional (added 2026-10-20): stored width of one hash value and the records
    # indexed, for the raw signature matrix.
    hash_bytes: Optional[Quantity] = None
    records: Optional[int] = None

    @property
    def max_standard_error(self) -> float:
        """Largest standard error of the Jaccard estimate (at J = 0.5) from the full signature."""
        from ..physics.retrieval import calc_minhash_standard_error
        return calc_minhash_standard_error(0.5, self.hashes)

    @property
    def signature_bytes(self) -> Quantity:
        return (self.hashes * self.hash_bytes).to("byte")

    @property
    def index_bytes(self) -> Quantity:
        return (self.records * self.signature_bytes).to("byte")

    @property
    def hashes(self) -> int:
        return self.bands * self.rows

    @property
    def threshold(self) -> float:
        from ..physics.retrieval import calc_lsh_threshold
        return calc_lsh_threshold(self.bands, self.rows)

    def candidate_probability(self, similarity: float) -> float:
        from ..physics.retrieval import calc_lsh_candidate_probability
        return calc_lsh_candidate_probability(similarity, self.bands, self.rows)


class InformationComputeRatioScenario(BaseScenario):
    """Accuracy gained per training FLOP for one epoch over all of ``dataset`` and over a
    ``keep_fraction`` subset, with ``gain_full`` and ``gain_subset`` points of gain."""
    workload: Workload
    dataset: Any                       # DatasetProfile
    gain_full: float                   # percentage points
    gain_subset: float                 # percentage points
    keep_fraction: float
    training_flops_factor: float = 3.0

    @property
    def examples(self) -> float:
        return self.dataset.training_examples.to("count").magnitude

    @property
    def training_flops_per_sample(self) -> Quantity:
        return (self.workload.inference_flops * self.training_flops_factor).to("GFLOP")

    @property
    def epoch_flops(self) -> Quantity:
        return (self.examples * self.training_flops_per_sample).to("flop")

    @property
    def subset_examples(self) -> float:
        return self.examples * self.keep_fraction

    @property
    def subset_flops(self) -> Quantity:
        return (self.subset_examples * self.training_flops_per_sample).to("flop")

    @property
    def icr_full(self) -> float:
        return self.gain_full / self.epoch_flops.magnitude

    @property
    def icr_subset(self) -> float:
        return self.gain_subset / self.subset_flops.magnitude

    @property
    def icr_ratio(self) -> float:
        return self.icr_subset / self.icr_full


class CoresetRunScenario(BaseScenario):
    """``workload`` trained for ``epochs`` over ``dataset`` on one ``hardware`` at ``utilization``,
    against a ``keep_fraction`` coreset scored by one forward pass of ``workload`` or ``proxy``.

    The selection inequality compares scoring plus subset training with full
    training; rescoring repeats the scoring pass before every epoch. The data-path
    fields follow one epoch through storage (the archive at the node's storage
    bandwidth), the host (``host_rate`` decoded samples a second), and the
    host-to-accelerator link (``decoded_sample`` bytes each).
    """
    workload: Workload
    dataset: Any                       # DatasetProfile
    hardware: HardwareNode
    utilization: float
    keep_fraction: float
    epochs: int
    training_flops_factor: float = 3.0
    proxy: Optional[Workload] = None
    proxy_epochs: Optional[int] = None         # epochs a proxy trained from scratch runs before scoring
    host_rate: Optional[Quantity] = None       # samples per second the host prepares
    decoded_sample: Optional[Quantity] = None  # bytes per sample sent to the accelerator

    @property
    def examples(self) -> float:
        return self.dataset.training_examples.to("count").magnitude

    @property
    def sustained_rate(self) -> Quantity:
        return self.hardware.compute.peak_flops * self.utilization

    @property
    def training_flops_per_sample(self) -> Quantity:
        return self.training_flops_factor * self.workload.inference_flops

    @property
    def epoch_flops(self) -> Quantity:
        return (self.examples * self.training_flops_per_sample).to("flop")

    @property
    def epoch_time(self) -> Quantity:
        return (self.epoch_flops / self.sustained_rate).to("second")

    @property
    def epoch_energy(self) -> Quantity:
        return (self.hardware.tdp * self.epoch_time).to("kJ")

    @property
    def sample_rate(self) -> float:
        """Samples a second the accelerator consumes at the sustained rate."""
        return self.examples / self.epoch_time.magnitude

    @property
    def run_time(self) -> Quantity:
        """``epochs`` full epochs."""
        return (self.epochs * self.epoch_time).to("hour")

    # Selection inequality
    @property
    def full_time(self) -> Quantity:
        return (self.examples * self.epochs * self.training_flops_factor * self.workload.inference_flops
                / self.sustained_rate).to("second")

    @property
    def subset_time(self) -> Quantity:
        return self.keep_fraction * self.full_time

    def score_time(self, scorer: Optional[Workload] = None) -> Quantity:
        """One forward pass of ``scorer`` (default: the workload) over the whole pool."""
        model = scorer or self.workload
        return (self.examples * model.inference_flops / self.sustained_rate).to("second")

    def score_energy(self, scorer: Optional[Workload] = None) -> Quantity:
        return (self.hardware.tdp * self.score_time(scorer)).to("kJ")

    def selection_time(self, scorer: Optional[Workload] = None) -> Quantity:
        """Scoring once plus training on the subset."""
        return self.score_time(scorer) + self.subset_time

    def saving(self, scorer: Optional[Workload] = None) -> Quantity:
        return self.full_time - self.selection_time(scorer)

    def saving_fraction(self, scorer: Optional[Workload] = None) -> float:
        return (self.saving(scorer) / self.full_time).to("").magnitude

    @property
    def rescoring_time(self) -> Quantity:
        """The workload rescoring the pool before every epoch."""
        return (self.epochs * self.score_time()).to("minute")

    @property
    def rescoring_saving_fraction(self) -> float:
        return ((self.full_time - (self.rescoring_time + self.subset_time)) / self.full_time).to("").magnitude

    @property
    def rescoring_over_training(self) -> float:
        return (self.rescoring_time / self.subset_time).to("").magnitude

    # One pruned epoch through the machine
    @property
    def pruned_epoch_time(self) -> Quantity:
        return self.keep_fraction * self.epoch_time

    @property
    def pruned_epoch_energy(self) -> Quantity:
        return self.keep_fraction * self.epoch_energy

    @property
    def run_saving_time(self) -> Quantity:
        return (self.epochs * (self.epoch_time - self.pruned_epoch_time)).to("hour")

    @property
    def run_saving_energy(self) -> Quantity:
        return (self.epochs * (self.epoch_energy - self.pruned_epoch_energy)).to("MJ")

    @property
    def saving_over_score_energy(self) -> float:
        return (self.run_saving_energy / self.score_energy()).to("").magnitude

    @property
    def archive_bytes(self) -> Quantity:
        return self.dataset.training_archive_size.to("GB")

    @property
    def storage_time(self) -> Quantity:
        return (self.archive_bytes / self.hardware.storage.bandwidth).to("second")

    @property
    def materialized_bytes(self) -> Quantity:
        """The coreset written out as its own shards."""
        return (self.keep_fraction * self.archive_bytes).to("GB")

    @property
    def link_bytes(self) -> Quantity:
        return (self.examples * self.decoded_sample).to("GB")

    @property
    def link_time(self) -> Quantity:
        return (self.link_bytes / self.hardware.interconnect.bandwidth).to("second")

    @property
    def host_epoch_time(self) -> Quantity:
        return Q_(self.examples / self.host_rate.to("count/second").magnitude, "second")


class DataEchoingScenario(BaseScenario):
    """A host that prepares ``host_rate`` samples a second feeding an accelerator that consumes
    ``accelerator_rate``, over ``epochs`` of ``examples``, with each prepared sample used
    ``echo_factor`` times."""
    accelerator_rate: Quantity
    host_rate: Quantity
    examples: Quantity                 # count
    epochs: int
    echo_factor: float

    @property
    def ratio(self) -> float:
        """Accelerator rate over host rate; echoing helps up to this factor."""
        return (self.accelerator_rate / self.host_rate).to("").magnitude

    @property
    def utilization(self) -> float:
        """Accelerator busy share without echoing."""
        return (self.host_rate / self.accelerator_rate).to("").magnitude

    @property
    def idle_fraction(self) -> float:
        return 1 - (self.host_rate / self.accelerator_rate).to("").magnitude

    @property
    def run_time(self) -> Quantity:
        return self.epochs * self.examples / self.host_rate

    @property
    def echoed_rate(self) -> Quantity:
        return min(self.host_rate * self.echo_factor, self.accelerator_rate)

    @property
    def echoed_run_time(self) -> Quantity:
        return self.epochs * self.examples / self.echoed_rate


class SelectionBudgetScenario(BaseScenario):
    """Selection that takes ``selection_time`` and leaves a subset that trains in ``subset_time``,
    against full training in ``full_time`` (the selection inequality in hours)."""
    selection_time: Quantity
    subset_time: Quantity
    full_time: Quantity
    pool: Optional[float] = None
    kept: Optional[float] = None

    @property
    def total_time(self) -> Quantity:
        return self.selection_time + self.subset_time

    @property
    def net_saving(self) -> Quantity:
        return self.full_time - self.total_time

    @property
    def pays(self) -> bool:
        return self.total_time < self.full_time

    @property
    def breakeven_selection(self) -> Quantity:
        """Longest selection that still saves time."""
        return self.full_time - self.subset_time

    @property
    def overhead_ratio(self) -> float:
        """Selection time over subset training time."""
        return (self.selection_time / self.subset_time).to("").magnitude

    @property
    def selection_share(self) -> float:
        """Selection time over full training time."""
        return (self.selection_time / self.full_time).to("").magnitude


class CoverageFloorScenario(BaseScenario):
    """A ``keep_fraction`` coreset drawn without regard to class from ``pool`` examples, of which
    ``rare_share`` belong to a rare class that needs at least ``floor`` examples."""
    pool: float
    rare_share: float
    keep_fraction: float
    floor: float

    @property
    def rare(self) -> float:
        return self.pool * self.rare_share

    @property
    def expected_kept(self) -> float:
        return self.rare * self.keep_fraction

    @property
    def shortfall(self) -> float:
        return self.floor - self.expected_kept


class CurriculumScheduleScenario(BaseScenario):
    """A curriculum that reaches target in ``curriculum_epochs`` against ``baseline_epochs``
    shuffled, with an optional linear warmup of ``warmup_epochs`` (epoch j serves j/W of the data)."""
    baseline_epochs: int
    curriculum_epochs: int
    warmup_epochs: Optional[int] = None

    @property
    def epoch_reduction(self) -> float:
        return (self.baseline_epochs - self.curriculum_epochs) / self.baseline_epochs

    @property
    def warmup_samples(self) -> float:
        """Dataset passes the linear warmup serves: (W + 1) / 2."""
        return (self.warmup_epochs + 1) / 2

    @property
    def warmup_share(self) -> float:
        """Warmup samples over W full passes."""
        return self.warmup_samples / self.warmup_epochs


class ActiveLearningRoundsScenario(BaseScenario):
    """Active learning from ``initial_labels`` adding ``labels_per_round`` for ``rounds`` rounds,
    each round paying ``round_scoring_cost`` to score the pool, against random labeling that
    needs ``random_labels`` at the same ``label_price``."""
    label_price: Quantity              # USD per label
    initial_labels: int
    labels_per_round: int
    rounds: int
    round_scoring_cost: Quantity       # USD per round
    random_labels: int
    label_turnaround: Optional[Quantity] = None   # time each round waits for its labels

    @property
    def active_labels(self) -> int:
        return self.initial_labels + self.rounds * self.labels_per_round

    @property
    def random_cost(self) -> Quantity:
        return (self.random_labels * self.label_price).to("USD")

    @property
    def active_label_cost(self) -> Quantity:
        return (self.active_labels * self.label_price).to("USD")

    @property
    def scoring_cost(self) -> Quantity:
        return (self.rounds * self.round_scoring_cost).to("USD")

    @property
    def active_cost(self) -> Quantity:
        return self.active_label_cost + self.scoring_cost

    @property
    def roi(self) -> float:
        """Saving over the active-learning cost."""
        return ((self.random_cost - self.active_cost) / self.active_cost).to("").magnitude


class BudgetMatchedLabelingScenario(BaseScenario):
    """A ``pool`` to label at ``label_price`` under ``budget``: random labeling spends the budget,
    active learning reaches the same quality with ``active_labels``."""
    pool: Quantity                     # count
    label_price: Quantity              # USD per count
    budget: Quantity                   # USD
    active_labels: Quantity            # count
    deadline: Optional[Quantity] = None

    @property
    def full_cost(self) -> Quantity:
        return self.pool * self.label_price

    @property
    def random_labels(self) -> Quantity:
        return (self.budget / self.label_price).to("count")

    @property
    def active_cost(self) -> Quantity:
        return self.active_labels * self.label_price

    @property
    def unspent_share(self):
        return (self.budget - self.active_cost) / self.budget

    @property
    def label_ratio(self) -> float:
        return (self.random_labels / self.active_labels).to("").magnitude

    @property
    def saving(self) -> Quantity:
        return self.budget - self.active_cost

    @property
    def over_budget_ratio(self) -> float:
        return (self.full_cost / self.budget).to("").magnitude


class SemiSupervisedCostScenario(BaseScenario):
    """A supervised baseline with ``supervised_labels`` and ``supervised_compute`` against a
    semi-supervised run with ``ssl_labels`` and ``ssl_compute``, labels at ``label_price``."""
    label_price: Quantity              # USD per label
    supervised_labels: int
    supervised_compute: Quantity       # USD
    ssl_labels: int
    ssl_compute: Quantity              # USD

    @property
    def supervised_label_cost(self) -> Quantity:
        return (self.supervised_labels * self.label_price).to("USD")

    @property
    def supervised_total(self) -> Quantity:
        return self.supervised_label_cost + self.supervised_compute

    @property
    def ssl_label_cost(self) -> Quantity:
        return (self.ssl_labels * self.label_price).to("USD")

    @property
    def ssl_total(self) -> Quantity:
        return self.ssl_label_cost + self.ssl_compute

    @property
    def compute_multiple(self) -> float:
        return (self.ssl_compute / self.supervised_compute).to("").magnitude

    @property
    def cost_reduction(self) -> float:
        return (self.supervised_total / self.ssl_total).to("").magnitude

    @property
    def label_reduction(self) -> float:
        return self.supervised_labels / self.ssl_labels


class PretrainingAmortizationScenario(BaseScenario):
    """``tasks`` tasks trained from scratch at ``scratch_compute`` each against one pretraining run
    of ``pretrain_compute`` plus ``finetune_compute`` per task, with the labels each path needs."""
    pretrain_compute: Quantity         # accelerator-hours
    scratch_compute: Quantity          # accelerator-hours per task
    finetune_compute: Quantity         # accelerator-hours per task
    tasks: int
    scratch_labels: Quantity           # count per task
    finetune_labels: Quantity          # count per task
    label_price: Quantity              # USD per count

    @property
    def scratch_total(self) -> Quantity:
        return self.scratch_compute * self.tasks

    @property
    def foundation_total(self) -> Quantity:
        return self.pretrain_compute + (self.finetune_compute * self.tasks)

    @property
    def scratch_label_cost(self) -> Quantity:
        return self.scratch_labels * self.label_price * self.tasks

    @property
    def finetune_label_cost(self) -> Quantity:
        return self.finetune_labels * self.label_price * self.tasks

    @property
    def label_cost_reduction(self):
        return self.scratch_label_cost / self.finetune_label_cost

    @property
    def marginal_compute_reduction(self) -> float:
        return (self.scratch_compute / self.finetune_compute).to("").magnitude

    @property
    def crossover_tasks(self) -> float:
        """Tasks after which pretraining has paid for itself in compute."""
        return (self.pretrain_compute / (self.scratch_compute - self.finetune_compute)).to("").magnitude


class DataCostBreakdownScenario(BaseScenario):
    """A supervised dataset's bill: ``acquisition``, ``labels`` at ``label_price``,
    ``storage_bytes`` kept for ``storage_duration`` at ``storage_rate``, and ``runs`` training
    runs of ``run_time`` each on ``accelerators`` at ``accelerator_rate``."""
    acquisition: Quantity              # USD
    labels: Quantity                   # count
    label_price: Quantity              # USD per count
    storage_bytes: Quantity
    storage_rate: Quantity             # USD per GB per month
    storage_duration: Quantity
    run_time: Quantity
    runs: int
    accelerators: int
    accelerator_rate: Quantity         # USD per accelerator-hour

    @property
    def label_cost(self) -> Quantity:
        return self.labels * self.label_price

    @property
    def storage_cost(self) -> Quantity:
        return (self.storage_bytes * self.storage_rate * self.storage_duration).to("USD")

    @property
    def accelerator_hours_per_run(self) -> Quantity:
        return self.accelerators * self.run_time

    @property
    def accelerator_hours(self) -> Quantity:
        return self.accelerator_hours_per_run * self.runs

    @property
    def compute_cost(self) -> Quantity:
        return (self.accelerator_hours * self.accelerator_rate).to("USD")

    @property
    def total(self) -> Quantity:
        return self.acquisition + self.label_cost + self.storage_cost + self.compute_cost

    @property
    def data_share(self):
        return (self.acquisition + self.label_cost + self.storage_cost) / self.total

    @property
    def compute_share(self):
        return self.compute_cost / self.total


class RunAmortizationScenario(BaseScenario):
    """A one-time investment of ``build_cost`` plus ``operating_cost`` that saves ``saving_per_run``
    on every training run that reuses its output."""
    build_cost: Quantity               # USD
    operating_cost: Quantity           # USD, one-time
    saving_per_run: Quantity           # USD per run
    run_counts: List[int] = []         # campaign sizes at which to read the return

    @property
    def investment(self) -> Quantity:
        return self.build_cost + self.operating_cost

    def net_return(self, runs: int) -> float:
        """(runs x saving - investment) / investment."""
        return ((runs * self.saving_per_run - self.investment) / self.investment).to("").magnitude

    def savings_multiple(self, runs: int) -> float:
        """Total savings over the investment."""
        return (runs * self.saving_per_run / self.investment).to("").magnitude

    @property
    def breakeven_runs(self) -> float:
        return (self.investment / self.saving_per_run).to("").magnitude


class DataScalingReturnScenario(BaseScenario):
    """Growing a dataset from ``base_examples`` to ``scaled_examples`` for ``accuracy_gain`` points."""
    base_examples: Quantity            # count
    scaled_examples: Quantity          # count
    accuracy_gain: float               # percentage points

    @property
    def scale(self) -> float:
        return (self.scaled_examples / self.base_examples).to("").magnitude


class CarbonPricedPlacementScenario(BaseScenario):
    """A run of ``it_energy`` on ``fleet`` placed among ``regions`` with and without a carbon price."""
    fleet: Fleet
    it_energy: Quantity
    regions: List[str]
    carbon_price_per_ton: float     # dollars per tonne CO2
    mfu: float
    reference_grid: Any

    @property
    def result(self):
        from .solvers import CarbonPricedPlacementModel
        return CarbonPricedPlacementModel().solve(self.fleet, self.it_energy, self.regions,
                                                  self.carbon_price_per_ton, self.mfu, self.reference_grid)


# --- Fleet expansion, expert weight traffic, and energy shares (added 2026-10-20) ---

class ClusterExpansionScenario(BaseScenario):
    """A synchronous job grown from ``n_devices`` to ``factor`` times as many on the same checkpoint tier.

    ``checkpoint`` supplies the write path; at each size its MTBF is replaced by the
    GPU-only interval ``component_mtbf / N`` and the overhead is the checkpoint-and-rework
    waste at the Young-Daly optimum. For fixed work, run time is proportional to step
    time over the retained fraction (one minus that overhead), so the larger job shortens
    the run only when its step speedup exceeds ``breakeven_speedup``.
    """
    checkpoint: CheckpointIntervalScenario
    component_mtbf: Quantity
    n_devices: int
    factor: float

    @property
    def expanded_devices(self) -> int:
        return round(self.n_devices * self.factor)

    def at(self, n_devices: int) -> CheckpointIntervalScenario:
        """The checkpoint scenario against the GPU-only interval of ``n_devices``."""
        from ..physics.reliability import calc_mtbf_cluster
        return self.checkpoint.variant(mtbf=calc_mtbf_cluster(self.component_mtbf, n_devices).to("hour"))

    @property
    def mtbf(self) -> Quantity:
        return self.at(self.n_devices).mtbf

    @property
    def expanded_mtbf(self) -> Quantity:
        return self.at(self.expanded_devices).mtbf

    @property
    def interval(self) -> Quantity:
        return self.at(self.n_devices).result.interval

    @property
    def expanded_interval(self) -> Quantity:
        return self.at(self.expanded_devices).result.interval

    @property
    def overhead(self) -> float:
        return self.at(self.n_devices).result.waste

    @property
    def expanded_overhead(self) -> float:
        return self.at(self.expanded_devices).result.waste

    @property
    def overhead_growth(self) -> float:
        return self.expanded_overhead / self.overhead

    @property
    def retained(self) -> float:
        return 1 - self.overhead

    @property
    def expanded_retained(self) -> float:
        return 1 - self.expanded_overhead

    @property
    def breakeven_speedup(self) -> float:
        """Step speedup at which the larger job finishes the fixed work in the same time."""
        return self.retained / self.expanded_retained

    @property
    def superlinear_threshold(self) -> float:
        """Step speedup above which the larger job also uses fewer accelerator-hours."""
        return self.factor * self.breakeven_speedup

    def run_time_change(self, step_speedup: float) -> float:
        """Fractional change in wall-clock run time at a measured step speedup."""
        return self.breakeven_speedup / step_speedup - 1

    def accelerator_hours_change(self, step_speedup: float) -> float:
        """Fractional change in accelerator-hours (devices times run time) at a step speedup."""
        return self.superlinear_threshold / step_speedup - 1


class ExpertWeightTrafficScenario(BaseScenario):
    """Weight bytes of the mixture-of-experts ``workload``, resident and read per token, against a
    dense model of ``dense_parameters`` that reads every weight each step."""
    workload: Workload
    dense_parameters: Quantity
    bytes_per_param: Quantity

    @property
    def resident_bytes(self) -> Quantity:
        from ..physics.quantities import memory_from_params
        return memory_from_params(self.workload.parameters, self.bytes_per_param).to("GB")

    @property
    def active_bytes(self) -> Quantity:
        from ..physics.quantities import memory_from_params
        return memory_from_params(self.workload.active_parameters, self.bytes_per_param).to("GB")

    @property
    def dense_bytes(self) -> Quantity:
        from ..physics.quantities import memory_from_params
        return memory_from_params(self.dense_parameters, self.bytes_per_param).to("GB")

    @property
    def resident_to_active(self) -> float:
        return (self.resident_bytes / self.active_bytes).to("").magnitude

    @property
    def dense_to_active(self) -> float:
        return (self.dense_bytes / self.active_bytes).to("").magnitude


class EnergyShareSavingScenario(BaseScenario):
    """A component drawing ``share`` of fleet energy whose energy per unit of work falls to
    ``energy_ratio`` of its value at unchanged work (Amdahl's law applied to energy)."""
    share: float
    energy_ratio: float

    @property
    def saving(self) -> float:
        """Fraction of fleet energy saved."""
        from ..physics.performance import calc_amdahls_speedup
        return 1 - 1 / calc_amdahls_speedup(self.share, 1 / self.energy_ratio)

    @property
    def ceiling(self) -> float:
        """Saving if the component's energy fell to zero."""
        from ..physics.performance import calc_amdahls_speedup
        return 1 - 1 / calc_amdahls_speedup(self.share, float("inf"))
# --- Safety enforcement on the warehouse mobile manipulator (added 2026-10-20) ---
# A gravity-loaded joint's braking authority and the kinetic barrier built on
# it, the least-change projection of a torque proposal, the permission check's
# stage budget, and the drift of a held command over one control period.

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
    """
    machine: Any                         # EmbodiedMachine: arm peak torque and TCP limit
    inertia: Quantity                    # joint inertia with payload (kg m^2)
    lever: Quantity                      # joint axis to tool point (m)
    joint_speed: Quantity                # approach speed of the joint (rad/s, as 1/s)
    bias_fraction: float                 # share of peak torque gravity holds
    barrier_gain: Optional[Quantity] = None    # gamma (1/s)
    barrier_clearances: List[Quantity] = []    # tool clearances at which to price the barrier
    site: Optional[Any] = None                 # EmbodiedSiteScenario: approach speed and pre-brake delay

    @property
    def torque_limit(self) -> Quantity:
        return self.machine.arm.max_torque.to("N*m")

    @property
    def bias(self) -> Quantity:
        return (self.torque_limit * self.bias_fraction).to("N*m")

    @property
    def torque_descending(self) -> Quantity:
        """Torque left for braking when gravity acts along the motion."""
        return (self.torque_limit - self.bias).to("N*m")

    @property
    def torque_ascending(self) -> Quantity:
        """Torque available for braking when gravity helps."""
        return (self.torque_limit + self.bias).to("N*m")

    def _alpha(self, torque) -> Quantity:
        return (torque / self.inertia).to("1/s**2")

    @property
    def alpha_free(self) -> Quantity:
        return self._alpha(self.torque_limit)

    @property
    def alpha_descending(self) -> Quantity:
        return self._alpha(self.torque_descending)

    @property
    def alpha_ascending(self) -> Quantity:
        return self._alpha(self.torque_ascending)

    def stop_angle(self, alpha) -> float:
        """Joint stop angle in radians at deceleration ``alpha``."""
        return (self.joint_speed ** 2 / (2 * alpha)).to("").magnitude

    def tool_stop(self, alpha) -> Quantity:
        return (self.lever * self.stop_angle(alpha)).to("mm")

    @property
    def tool_stop_free(self) -> Quantity:
        return self.tool_stop(self.alpha_free)

    @property
    def tool_stop_descending(self) -> Quantity:
        return self.tool_stop(self.alpha_descending)

    @property
    def tool_stop_ascending(self) -> Quantity:
        return self.tool_stop(self.alpha_ascending)

    @property
    def overrun(self) -> Quantity:
        """Tool travel the descending stop adds over the unbiased one."""
        return (self.tool_stop_descending - self.tool_stop_free).to("mm")

    @property
    def stop_time_descending(self) -> Quantity:
        return (self.joint_speed / self.alpha_descending).to("ms")

    @property
    def tcp_speed(self) -> Quantity:
        return (self.lever * self.joint_speed).to("m/s")

    def barrier_margin(self, clearance) -> float:
        """Kinetic barrier h = theta - w^2 / (2 alpha_net) at a tool clearance (radians)."""
        return (clearance / self.lever).to("").magnitude - self.stop_angle(self.alpha_descending)

    def barrier_torque_demand(self, clearance) -> Quantity:
        """Braking torque the barrier demands at a tool clearance (the bound's negative)."""
        h = self.barrier_margin(clearance)
        bound = (-self.bias + self.inertia * self.alpha_descending
                 * (self.barrier_gain * h / self.joint_speed - 1)).to("N*m")
        return (-bound).to("N*m")

    @property
    def barrier_demands(self) -> List[Quantity]:
        return [self.barrier_torque_demand(c) for c in self.barrier_clearances]

    @property
    def separation_person(self) -> Quantity:
        """Person's travel over the reaction (pre-brake delay) and the descending joint stop."""
        return (self.site.v_human_approach * (self.site.tau_delay + self.stop_time_descending)).to("mm")

    @property
    def separation_reaction(self) -> Quantity:
        """Tool travel during the reaction time."""
        return (self.tcp_speed * self.site.tau_delay).to("mm")

    @property
    def separation_total(self) -> Quantity:
        """Protective separation before intrusion and position margins."""
        return (self.separation_person + self.separation_reaction + self.tool_stop_descending).to("mm")

    @property
    def separation_person_share(self) -> float:
        return (self.separation_person / self.separation_total).to("").magnitude


class TorqueHalfSpaceProjectionScenario(BaseScenario):
    """A torque proposal against one barrier half-space ``normal . u <= bound``: the least-change
    projection (``calc_cbf_qp_orthogonal_projection``) against a per-axis ``clamp``.

    Torques are in N m, stored as plain floats like the projection helper's inputs.
    """
    proposal: List[float]
    normal: List[float]
    bound: float
    clamp: List[float]
    actuator_limit: float

    @property
    def projection(self) -> dict:
        from ..physics.robotics import calc_cbf_qp_orthogonal_projection
        return calc_cbf_qp_orthogonal_projection(self.proposal, self.normal, self.bound)

    @staticmethod
    def _dot(a, b) -> float:
        return sum(x * y for x, y in zip(a, b))

    @property
    def proposal_value(self) -> float:
        return self._dot(self.normal, self.proposal)

    @property
    def breach(self) -> float:
        return self.proposal_value - self.bound

    @property
    def clamp_value(self) -> float:
        return self._dot(self.normal, self.clamp)

    @property
    def clamp_penetration(self) -> float:
        return self.clamp_value - self.bound

    @property
    def projected_value(self) -> float:
        return self._dot(self.normal, self.projection["u_projected"])

    @staticmethod
    def angle_deg(u) -> float:
        import math
        return math.degrees(math.atan2(u[1], u[0]))


class CycleBudgetScenario(BaseScenario):
    """Stage sub-budgets of one permission check against ``site``'s declared WCET and deadline."""
    site: Any                                  # EmbodiedSiteScenario: WCET, deadline, unloaded cycle
    stages: List[Tuple[str, Quantity]]

    @property
    def total(self) -> Quantity:
        return sum((q for _, q in self.stages), 0 * self.stages[0][1]).to("us")

    def stage(self, name: str) -> Quantity:
        return dict(self.stages)[name].to("us")

    @property
    def headroom(self) -> Quantity:
        """Deadline left after the declared worst case."""
        return (self.site.enforcer_deadline - self.total).to("us")

    def share(self, name: str) -> float:
        return (self.stage(name) / self.total).to("").magnitude


class HoldIntervalDriftScenario(BaseScenario):
    """Drift while a drive holds the last command for one ``period``: travel at ``speed`` plus the
    half ``acceleration`` ``period^2`` an independent acceleration bound allows."""
    speed: Quantity
    acceleration: Quantity
    period: Quantity

    @property
    def drift(self) -> Quantity:
        return (self.speed * self.period + 0.5 * self.acceleration * self.period ** 2).to("mm")


# --- Workload placement: what a permission tick shares with its proposer (added 2026-10-20) ---
# Solvers SharedDramReadModel and ThermalThrottleModel in engine/solvers/realtime.py;
# the droop, stall-share, and trace scenarios are closed forms over physics/.

class SharedDramReadScenario(BaseScenario):
    """``site``'s permission tick placed on the machine's application SoC, its snapshot read
    queued behind backlogs on the shared DRAM controller.

    Bandwidth is the SoC memory's peak; the row-conflict time is ``memory_tech``'s
    t_RP + t_RCD. Addresses interleave across every channel unless
    ``one_channel`` confines the backlog and the read to one, which divides the
    bandwidth by the channel count (bus width over channel width).
    """
    site: Any                         # EmbodiedSiteScenario: WCET, deadline, blocking, interference
    memory_tech: Any                  # MemoryTier: t_rp, t_rcd, channel_width
    sense: Quantity                   # the tick's sense allowance (the snapshot read happens here)
    compute_after_sense: Quantity     # constraint check and drive-frame staging
    tile_bytes: int
    camera_bytes: int
    telemetry_bytes: int
    snapshot_bytes: int
    transaction_bytes: int
    bank_switches: int
    one_channel: bool = False

    @property
    def memory(self):
        return self.site.machine.brain_soc.memory

    @property
    def channels(self) -> int:
        return int(round((self.memory.bus_width / self.memory_tech.channel_width).to("").magnitude))

    @property
    def bank_switch_time(self) -> Quantity:
        return (self.memory_tech.t_rp + self.memory_tech.t_rcd).to("ns")

    @property
    def bandwidth(self) -> Quantity:
        bw = self.memory.bandwidth
        return bw / self.channels if self.one_channel else bw

    @property
    def result(self):
        from .solvers import SharedDramReadModel
        s = self.site
        return SharedDramReadModel().solve(
            tile_bytes=self.tile_bytes, camera_bytes=self.camera_bytes,
            telemetry_bytes=self.telemetry_bytes, snapshot_bytes=self.snapshot_bytes,
            transaction_bytes=self.transaction_bytes, bandwidth=self.bandwidth,
            bank_switches=self.bank_switches, bank_switch_time=self.bank_switch_time,
            sense=self.sense, compute_after_sense=self.compute_after_sense, wcet=s.enforcer_wcet,
            execution_ceiling=s.enforcer_deadline - s.enforcer_blocking - s.enforcer_interference)


class StallShareScenario(BaseScenario):
    """``site``'s unloaded permission tick with a share of it slowed ``slowdown`` times.

    The tick may stretch to ``bound`` (default the declared WCET); the
    break-even share is ``(bound / unloaded - 1) / (slowdown - 1)``
    (``calc_stall_share_breakeven``).
    """
    site: Any                         # EmbodiedSiteScenario
    slowdown: float
    bound: Optional[Quantity] = None

    @property
    def budget(self) -> Quantity:
        return (self.site.enforcer_wcet if self.bound is None else self.bound).to("us")

    @property
    def allowed_stretch(self) -> float:
        return float((self.budget / self.site.enforcer_unloaded).to("").magnitude)

    @property
    def breakeven_share(self) -> float:
        from ..physics import calc_stall_share_breakeven
        return calc_stall_share_breakeven(self.allowed_stretch, self.slowdown)

    @property
    def breakeven_stall(self) -> Quantity:
        """The break-even share as time of the unloaded tick."""
        return (self.breakeven_share * self.site.enforcer_unloaded).to("us")


class RailDroopScenario(BaseScenario):
    """A current step ``delta_i`` in ``rise_time`` through a package's inductance and resistance,
    and the alpha-power gate-delay stretch at the drooped rail."""
    rail_voltage: Quantity
    inductance: Quantity
    resistance: Quantity
    delta_i: Quantity
    rise_time: Quantity
    threshold_voltage: Quantity
    alpha: float

    @property
    def inductive_droop(self) -> Quantity:
        from ..physics.robotics import calc_inductive_voltage_droop
        return calc_inductive_voltage_droop(self.inductance, self.delta_i / self.rise_time).to("mV")

    @property
    def resistive_droop(self) -> Quantity:
        from ..physics.robotics import calc_voltage_droop
        return calc_voltage_droop(self.delta_i, self.resistance).to("mV")

    @property
    def droop(self) -> Quantity:
        return (self.inductive_droop + self.resistive_droop).to("mV")

    @property
    def drooped_voltage(self) -> Quantity:
        return (self.rail_voltage - self.droop).to("V")

    @property
    def delay_expansion_percent(self) -> float:
        from ..physics.robotics import calc_alpha_power_gate_delay_stretch
        return calc_alpha_power_gate_delay_stretch(
            nominal_voltage=self.rail_voltage, drooped_voltage=self.drooped_voltage,
            threshold_voltage=self.threshold_voltage, alpha=self.alpha)["delay_expansion_percent"]


class ThermalThrottleScenario(BaseScenario):
    """``site``'s permission tick on a cluster whose clock a thermal governor halves.

    The tick is sized to the declared WCET at ``nominal_clock``; the junction
    follows a first-order model from ``quiet_power`` to ``loaded_power`` through
    ``theta``; the clock throttles at ``trip`` and returns at ``clear``.
    """
    site: Any                         # EmbodiedSiteScenario: WCET, deadline, B + I, tick
    nominal_clock: Quantity
    throttled_clock: Quantity
    time_constant: Quantity
    ambient: Quantity                 # degC
    theta: Quantity                   # junction-to-ambient, K/W
    quiet_power: Quantity
    loaded_power: Quantity
    trip: Quantity                    # degC
    clear: Quantity                   # degC

    @property
    def result(self):
        from .solvers import ThermalThrottleModel
        s = self.site
        return ThermalThrottleModel().solve(
            wcet=s.enforcer_wcet, blocking_and_interference=s.enforcer_blocking + s.enforcer_interference,
            deadline=s.enforcer_deadline, tick=s.t_tick, nominal_clock=self.nominal_clock,
            throttled_clock=self.throttled_clock, time_constant=self.time_constant,
            ambient=self.ambient, theta=self.theta, quiet_power=self.quiet_power,
            loaded_power=self.loaded_power, trip=self.trip, clear=self.clear)


class TickTraceScenario(BaseScenario):
    """Percentiles of a permission tick's execution time over a trace of ``ticks`` periods,
    cold and warm, against ``site``'s declared WCET and deadline."""
    site: Any                         # EmbodiedSiteScenario
    ticks: int
    cold: Dict[str, Quantity]         # percentile label -> execution time
    warm: Dict[str, Quantity]

    @property
    def duration(self) -> Quantity:
        return (self.ticks * self.site.t_tick).to("s")

    @property
    def hours_minutes_seconds(self):
        total = int(round(self.duration.to("s").magnitude))
        h, rem = divmod(total, 3600)
        m, sec = divmod(rem, 60)
        return h, m, sec
# --- Agent operations: traces, retention, postmortems, staged release (added 2026-10-20) ---
# What a trial's spans explain, which trace bodies a selection keeps, what a
# controlled diagnostic branch bounds, and how a release is staged and gated.
# Solvers in engine/solvers/operations.py; the closed forms in physics/statistics.py.

class SpanCriticalPathScenario(BaseScenario):
    """One trial's observed spans on a shared clock, the spans each waits for, and one intervention."""
    intervals: Dict[str, Any]          # span -> (start, end) Quantity pair after segment start
    depends_on: Dict[str, List[str]]
    intervention: Dict[str, Any] = {}  # span -> new duration (Quantity); others keep theirs

    @property
    def result(self):
        from .solvers import SpanCriticalPathModel
        return SpanCriticalPathModel().solve(self.intervals, self.depends_on, self.intervention)

    def duration(self, span: str) -> Quantity:
        start, end = self.intervals[span]
        return (end - start).to("s")


class HeadSamplingMissScenario(BaseScenario):
    """Head sampling keeps each trial with ``retention_probability``, decided before its outcome;
    a failure mode that occurs ``occurrences`` times leaves no detailed trace with ``miss``."""
    retention_probability: float
    occurrences: int

    @property
    def miss(self) -> float:
        """``(1 - p)^n``: no occurrence retained, the complement of ``calc_independent_success``."""
        from ..physics.agents import calc_independent_success
        return 1 - calc_independent_success(self.retention_probability, self.occurrences)


class TraceRetentionScenario(BaseScenario):
    """One cohort period of trials in disjoint verdict classes, each with a ``trace_size`` body,
    retained over ``window`` by keeping everything, head sampling, or verdict-aware selection."""
    failed: int                        # failed a criterion or violated a policy
    unresolved: int                    # pending or unknown verdict
    ordinary: int                      # ordinary accepted
    cohort_period: Quantity            # the period the counts cover
    trace_size: Quantity               # one permitted trace body
    window: Quantity
    head_probability: float
    control_probability: float         # control sample of ordinary accepted trials
    control_step: float = 0.01         # one percentage point of control sampling

    @property
    def trials(self) -> int:
        return self.failed + self.unresolved + self.ordinary

    @property
    def result(self):
        from .solvers import TraceRetentionModel
        return TraceRetentionModel().solve(self.failed, self.unresolved, self.ordinary, self.cohort_period,
                                           self.trace_size, self.window, self.head_probability,
                                           self.control_probability, self.control_step)


class ControlledBranchScenario(BaseScenario):
    """Fresh continuations from one saved prefix under several branches, counted by outcome;
    the Wilson bound a clean branch leaves and the clean runs a ``target_bound`` needs."""
    outcomes: Dict[str, Dict[str, int]]   # branch -> outcome class -> runs
    accepted_class: str = "accepted"
    target_bound: float                   # failure-rate upper limit to demonstrate
    confidence: float = 0.95              # two-sided

    @property
    def z(self) -> float:
        from statistics import NormalDist
        return NormalDist().inv_cdf(1 - (1 - self.confidence) / 2)

    def runs(self, branch: str) -> int:
        return sum(self.outcomes[branch].values())

    def accepted(self, branch: str) -> int:
        return self.outcomes[branch].get(self.accepted_class, 0)

    def acceptance_interval(self, branch: str):
        """Wilson interval on the branch's acceptance rate."""
        from ..physics.statistics import calc_wilson_interval
        return calc_wilson_interval(self.accepted(branch), self.runs(branch), self.z)

    def failure_upper(self, branch: str) -> float:
        """Wilson upper limit on the branch's non-acceptance rate."""
        from ..physics.statistics import calc_wilson_interval
        n = self.runs(branch)
        return calc_wilson_interval(n - self.accepted(branch), n, self.z)[1]

    def clean_failure_upper(self, runs: int) -> float:
        """Wilson upper limit on the failure rate after ``runs`` clean runs."""
        from ..physics.statistics import calc_wilson_interval
        return calc_wilson_interval(0, runs, self.z)[1]

    @property
    def clean_runs_for_target(self) -> int:
        from ..physics.statistics import calc_wilson_clean_runs
        return calc_wilson_clean_runs(self.target_bound, self.z)


class ToolDriftScenario(BaseScenario):
    """Baseline and candidate distributions of tool calls over the same tools, compared by
    the Jensen-Shannon divergence (natural logarithms)."""
    baseline: Dict[str, float]
    candidate: Dict[str, float]

    @property
    def divergence(self) -> float:
        from ..physics.statistics import calc_jensen_shannon_divergence
        keys = list(self.baseline)
        return calc_jensen_shannon_divergence([self.baseline[k] for k in keys],
                                              [self.candidate[k] for k in keys])

    @property
    def maximum(self) -> float:
        """``ln 2``, reached by two distributions with no tool in common."""
        return _math.log(2)

    @property
    def share_of_maximum(self) -> float:
        return self.divergence / self.maximum


class SequentialCanaryScenario(BaseScenario):
    """A canary monitor: Wald's SPRT between acceptance ``p_base`` and a specified degraded
    ``p_degraded`` at targets ``alpha`` (false hold) and ``beta`` (missed degradation), on
    ``fraction`` of eligible tasks arriving at ``arrival_rate``, labels ``label_delay`` late."""
    p_base: float
    p_degraded: float
    alpha: float
    beta: float
    arrival_rate: Quantity             # eligible tasks per hour
    fraction: float
    label_delay: Quantity

    @property
    def result(self):
        from .solvers import SequentialCanaryModel
        return SequentialCanaryModel().solve(self.p_base, self.p_degraded, self.alpha, self.beta,
                                             self.arrival_rate, self.fraction, self.label_delay)


class FamilywiseErrorScenario(BaseScenario):
    """``family_size`` statistical verdicts, each allowed ``significance`` of being wrong."""
    family_size: int
    significance: float

    @property
    def union_bound(self) -> float:
        """At most the sum of the individual error probabilities."""
        return min(1.0, self.family_size * self.significance)

    @property
    def independent(self) -> float:
        """Chance at least one is wrong if the errors are independent."""
        from ..physics.agents import calc_independent_success
        return calc_independent_success(self.significance, self.family_size)

    @property
    def per_test(self) -> float:
        """Bonferroni: the level each verdict gets so the family stays at ``significance``."""
        return self.significance / self.family_size

# ── Fitting a training step on one accelerator (added 2026-10-20) ────────────
# Number formats and the updates and gradients they lose, loss scaling on a
# measured gradient histogram, softmax one block at a time, and the tiles and
# stored tensors of attention.


class FloatAccumulationScenario(BaseScenario):
    """A weight stored in a binary float format receiving the same small update ``repeats`` times.

    ``formats`` maps a label to (exponent bits, fraction bits). Each add rounds
    to the format, so an update below half the spacing near the weight rounds
    away (swamping).
    """
    weight: float
    update: float
    repeats: int
    formats: Dict[str, Any]

    def spacing(self, fmt: str) -> float:
        """Gap from ``weight`` to the next larger value of ``fmt``."""
        from ..physics.numerics import calc_float_spacing
        return calc_float_spacing(self.weight, *self.formats[fmt])

    def next_value(self, fmt: str) -> float:
        return self.weight + self.spacing(fmt)

    def stored_after_one(self, fmt: str) -> float:
        return self.trace(fmt)[1]

    def trace(self, fmt: str) -> List[float]:
        """Stored weight before and after each update."""
        from ..physics.numerics import calc_repeated_add
        return calc_repeated_add(self.weight, self.update, self.repeats, *self.formats[fmt])

    def final(self, fmt: str) -> float:
        return self.trace(fmt)[-1]

    def vanishing_ratio(self, fmt: str) -> float:
        """Weight over the largest update that rounds away (half the spacing)."""
        return self.weight / (self.spacing(fmt) / 2)


class LossScalingScenario(BaseScenario):
    """A measured gradient-magnitude histogram against a 16-bit float format, before and after loss scaling.

    ``histogram`` rows hold log2 bin edges and the bar-top pixel row read from
    the published figure; ``grid_*`` are the two gridlines that calibrate the
    figure's log-scale y axis. ``scale`` multiplies the loss (and so every
    gradient). ``dynamic_scale`` and ``probe_gradient`` illustrate a dynamic
    scaler's setting.
    """
    histogram: List[Dict[str, Any]]
    grid_high_px: float
    grid_high_pct: float
    grid_low_px: float
    grid_low_pct: float
    zero_fraction: float
    scale: float
    float_bits: Any                    # (exponent bits, fraction bits)
    dynamic_scale: Optional[float] = None
    probe_gradient: Optional[float] = None

    @property
    def _range(self):
        from ..physics.numerics import calc_float_range
        return calc_float_range(*self.float_bits)

    @property
    def largest(self) -> float:
        return self._range[0]

    @property
    def smallest_normal(self) -> float:
        return self._range[1]

    @property
    def smallest_subnormal(self) -> float:
        return self._range[2]

    @property
    def normal_floor_log2(self) -> float:
        return _math.log2(self._range[1])

    @property
    def subnormal_floor_log2(self) -> float:
        return _math.log2(self._range[2])

    @property
    def bins(self) -> List[Dict[str, Any]]:
        """Bins with their share of all values, in percent (None for an empty bin)."""
        doublings = _math.log2(self.grid_high_pct / self.grid_low_pct)
        px_per_doubling = (self.grid_low_px - self.grid_high_px) / doublings
        out = []
        for row in self.histogram:
            px = row["bar_top_px"]
            pct = None if px is None else self.grid_high_pct * 2 ** (-(px - self.grid_high_px) / px_per_doubling)
            out.append(dict(low=row["log2_low"], high=row["log2_high"], percent=pct))
        return out

    def _sum(self, keep) -> float:
        return sum(b["percent"] for b in self.bins if b["percent"] and keep(b))

    @property
    def below_subnormal_pct(self) -> float:
        """Values that become zero in the format: bins wholly below the smallest subnormal."""
        return self._sum(lambda b: b["high"] <= self.subnormal_floor_log2)

    @property
    def subnormal_pct(self) -> float:
        return self._sum(lambda b: b["low"] >= self.subnormal_floor_log2 and b["high"] <= self.normal_floor_log2)

    @property
    def normal_pct(self) -> float:
        return self._sum(lambda b: b["low"] >= self.normal_floor_log2)

    @property
    def nonzero_pct(self) -> float:
        return self._sum(lambda b: True)

    @property
    def below_share_of_nonzero(self) -> float:
        return self.below_subnormal_pct / self.nonzero_pct

    @property
    def shift(self) -> float:
        """Doublings the scale adds to every gradient: log2 of the scale."""
        return _math.log2(self.scale)

    @property
    def rescued_pct(self) -> float:
        """Values the scale lifts from below the smallest subnormal to at or above it (whole bins)."""
        f = self.subnormal_floor_log2
        return self._sum(lambda b: b["high"] <= f and b["low"] >= f - self.shift)

    @property
    def top_edge_log2(self) -> float:
        return max(b["high"] for b in self.bins if b["percent"])

    @property
    def headroom_doublings(self) -> float:
        """Doublings from the largest occupied bin edge up to the format's largest value."""
        return _math.log2(self.largest) - self.top_edge_log2

    @property
    def scaled_probe(self) -> float:
        return self.probe_gradient * self.dynamic_scale

    @property
    def dynamic_overflow_threshold(self) -> float:
        """Unscaled gradient above which the dynamic scale overflows the format."""
        return self.largest / self.dynamic_scale


class OnlineSoftmaxScenario(BaseScenario):
    """One row of attention scores and values, softmax-weighted one block at a time."""
    scores: List[float]
    values: List[float]
    block_size: int

    @property
    def blocks(self) -> List[Dict[str, float]]:
        from ..physics.transformer import calc_online_softmax
        return calc_online_softmax(self.scores, self.values, self.block_size)

    @property
    def whole_row(self) -> Dict[str, float]:
        from ..physics.transformer import calc_online_softmax
        return calc_online_softmax(self.scores, self.values, len(self.scores))[-1]

    @property
    def weighted_average(self) -> float:
        last = self.blocks[-1]
        return last["output"] / last["sum"]

    def block_terms(self, block: int) -> List[float]:
        """exp(x - m) of each score of ``block`` against the running maximum after it."""
        m = self.blocks[block]["max"]
        xs = self.scores[block * self.block_size:(block + 1) * self.block_size]
        return [_math.exp(x - m) for x in xs]


class AttentionTilingScenario(BaseScenario):
    """One head's attention at ``seq_len`` tokens, whole or in square tiles, and what each kernel stores.

    Standard attention stores the scores and the probabilities (``seq_len``
    squared each) beside Q, K, V, and the output; FlashAttention stores a
    running maximum and sum per row instead.
    """
    seq_len: int
    head_dim: int
    tile: int
    heads: int
    bytes_per_value: Quantity
    io_tensors: int = 4              # Q, K, V, and the output
    stored_square_matrices: int = 2  # scores and probabilities
    row_statistics: int = 2          # running maximum and sum

    @property
    def score_matrix(self) -> Quantity:
        return (self.seq_len ** 2 * self.bytes_per_value).to("MB")

    @property
    def all_heads_scores(self) -> Quantity:
        return (self.heads * self.score_matrix).to("GB")

    @property
    def tile_bytes(self) -> Quantity:
        return (self.tile ** 2 * self.bytes_per_value).to("KB")

    @property
    def tiles_per_side(self) -> int:
        return self.seq_len // self.tile

    @property
    def tile_count(self) -> int:
        return self.tiles_per_side ** 2

    @property
    def qkv_bytes(self) -> Quantity:
        """Q, K, and V: three tensors of seq_len x head_dim."""
        return (3 * self.seq_len * self.head_dim * self.bytes_per_value).to("MB")

    @property
    def io_bytes(self) -> Quantity:
        return (self.io_tensors * self.seq_len * self.head_dim * self.bytes_per_value).to("MB")

    @property
    def stored_standard(self) -> Quantity:
        return (self.io_bytes + self.stored_square_matrices * self.score_matrix).to("MB")

    @property
    def stored_flash(self) -> Quantity:
        return (self.io_bytes + self.row_statistics * self.seq_len * self.bytes_per_value).to("MB")

    @property
    def stored_ratio(self) -> float:
        return (self.stored_standard / self.stored_flash).to("").magnitude


class TokenizedBatchFeedScenario(BaseScenario):
    """A host that tokenizes each batch and copies it to the accelerator, against the device's step.

    Without prefetching the host and the device run back to back, so the
    device is busy ``compute_time`` out of every ``prep_time + compute_time``.
    """
    compute_time: Quantity
    tokens_per_batch: int
    token_rate: Quantity             # tokens per second on one host core
    bytes_per_token: Quantity        # token ids as stored for the copy
    link_bandwidth: Quantity
    workers: int = 1

    @property
    def tokenization_time(self) -> Quantity:
        return (self.tokens_per_batch / self.token_rate).to("s")

    @property
    def parallel_tokenization_time(self) -> Quantity:
        return (self.tokenization_time / self.workers).to("s")

    @property
    def batch_bytes(self) -> Quantity:
        return (self.tokens_per_batch * self.bytes_per_token).to("byte")

    @property
    def transfer_time(self) -> Quantity:
        return (self.batch_bytes / self.link_bandwidth).to("s")

    @property
    def prep_time(self) -> Quantity:
        """Single-core preparation: tokenization plus the copy."""
        return (self.tokenization_time + self.transfer_time).to("s")

    @property
    def busy_fraction(self) -> float:
        """Share of time the device computes when preparation and compute run in sequence."""
        return (self.compute_time / (self.prep_time + self.compute_time)).to("").magnitude
# --- Model compression (added 2026-10-20) -----------------------------------

class WeightReadScenario(BaseScenario):
    """Every weight of ``workload`` read once from ``hardware``'s memory at ``bytes_per_weight``.

    The weight-streaming floor of one batch-one pass: per frame for a vision
    model, per token for a decoder. Unlike ``DecodeScenario`` it assumes no FLOP
    count, so it serves convolutional networks as well as decoders.
    ``energy_per_byte`` (optional) prices the reads.
    """
    workload: Workload
    hardware: HardwareNode
    bytes_per_weight: Quantity
    energy_per_byte: Optional[Quantity] = None

    @property
    def weight_bytes(self) -> Quantity:
        return self.workload.size_in_bytes(self.bytes_per_weight)

    @property
    def read_time(self) -> Quantity:
        return (self.weight_bytes / self.hardware.memory.bandwidth).to("millisecond")

    @property
    def read_energy(self) -> Quantity:
        return (self.weight_bytes * self.energy_per_byte).to("joule")

    @property
    def read_power(self) -> Quantity:
        """Average power of the reads alone, sustained over one pass."""
        return (self.read_energy / self.read_time).to("watt")

    # Optional (added 2026-10-20): a decoder's KV cache read with the weights
    # every token, and weights unpacked to a wider format in a separate pass.
    context_tokens: Optional[int] = None
    kv_bytes_per_value: Optional[Quantity] = None
    unpacked_bytes_per_weight: Optional[Quantity] = None

    @property
    def kv_cache_bytes(self) -> Quantity:
        """Keys and values of ``context_tokens`` tokens at batch one."""
        return self.workload.get_kv_cache_size(seq_len=self.context_tokens, batch_size=1,
                                               precision=self.kv_bytes_per_value).to("byte")

    @property
    def token_bytes(self) -> Quantity:
        """Bytes read per generated token: every weight and the whole KV cache."""
        return (self.weight_bytes + self.kv_cache_bytes).to("byte")

    @property
    def token_time(self) -> Quantity:
        return (self.token_bytes / self.hardware.memory.bandwidth).to("millisecond")

    @property
    def token_rate(self) -> Quantity:
        return (1 / self.token_time).to("1/second")

    @property
    def separate_unpack_bytes(self) -> Quantity:
        """Bytes per token when a separate kernel unpacks the weights first: read the packed
        weights, write them unpacked, read them back for the matrix multiply, read the cache."""
        unpacked = self.workload.size_in_bytes(self.unpacked_bytes_per_weight)
        return (self.weight_bytes + 2 * unpacked + self.kv_cache_bytes).to("byte")

    @property
    def separate_unpack_time(self) -> Quantity:
        return (self.separate_unpack_bytes / self.hardware.memory.bandwidth).to("millisecond")


class FrameCompressionScenario(BaseScenario):
    """A per-frame model against a camera frame rate: the full model and a channel-pruned
    variant, run in floating point on the CPU and in integers on the accelerator.

    ``cpu_rate`` and ``accelerator_rate`` are the full model's frame rates. The
    pruned model's frame time scales with its share of the arithmetic, because
    channel pruning leaves dense tensors that run the same kernels on fewer
    channels (an upper bound on the saving; measurement decides).
    """
    full: Workload
    pruned: Workload
    frame_rate: Quantity
    cpu_rate: Quantity
    accelerator_rate: Quantity
    float_bytes: Quantity
    int_bytes: Quantity

    @property
    def frame_budget(self) -> Quantity:
        return (1 / self.frame_rate).to("millisecond")

    @property
    def cpu_frame(self) -> Quantity:
        return (1 / self.cpu_rate).to("millisecond")

    @property
    def accelerator_frame(self) -> Quantity:
        return (1 / self.accelerator_rate).to("millisecond")

    @property
    def flop_fraction(self) -> float:
        return (self.pruned.inference_flops / self.full.inference_flops).to("").magnitude

    @property
    def param_fraction(self) -> float:
        return (self.pruned.parameters / self.full.parameters).to("").magnitude

    @property
    def full_bytes(self) -> Quantity:
        return self.full.size_in_bytes(self.float_bytes).to("MB")

    @property
    def pruned_bytes(self) -> Quantity:
        return self.pruned.size_in_bytes(self.float_bytes).to("MB")

    @property
    def pruned_int_bytes(self) -> Quantity:
        return self.pruned.size_in_bytes(self.int_bytes).to("MB")

    @property
    def pruned_cpu_frame(self) -> Quantity:
        return (self.cpu_frame * self.flop_fraction).to("millisecond")

    @property
    def pruned_accelerator_frame(self) -> Quantity:
        return (self.accelerator_frame * self.flop_fraction).to("millisecond")


class SmallestModelAtScoreScenario(BaseScenario):
    """The smallest model at or above ``threshold`` on ``score`` released up to each date.

    ``series`` names a ``Literature.Series`` table with ``date``, ``parameters``,
    ``model``, and the ``score`` column; the frontier is the running minimum of
    parameters among rows that clear the bar, in date order.
    """
    series: str
    score: str
    threshold: float

    @property
    def rows(self) -> list:
        from ..literature.registry import Literature
        return sorted(getattr(Literature.Series, self.series).entries, key=lambda r: r["date"])

    @property
    def frontier(self) -> list:
        front = []
        for r in self.rows:
            if r[self.score] >= self.threshold and (not front or r["parameters"] < front[-1]["parameters"]):
                front.append(r)
        return front

    @property
    def first(self) -> dict:
        return self.frontier[0]

    @property
    def last(self) -> dict:
        return self.frontier[-1]

    @property
    def shrink(self) -> float:
        return self.first["parameters"] / self.last["parameters"]

    @property
    def span_years(self) -> float:
        import datetime as _dt
        days = (_dt.date.fromisoformat(self.last["date"]) - _dt.date.fromisoformat(self.first["date"])).days
        from ..core.units import SEC_PER_DAY, SEC_PER_YEAR_LEAP
        return days / (SEC_PER_YEAR_LEAP / SEC_PER_DAY)


class ConvBnReluFusionScenario(BaseScenario):
    """A convolution, batch norm, and ReLU on one feature map, run as three kernels or one.

    Unfused, each kernel reads its input from device memory and writes its
    output back (six transfers), and the convolution and norm also read their
    parameters; fused (the norm folded into the convolution's weights and bias,
    the ReLU applied before the write), one read and one write remain. A lower
    bound on time: traffic over ``hardware``'s memory bandwidth, compute ignored.
    """
    channels: int
    height: int
    width: int
    kernel_size: int
    bytes_per_value: Quantity
    norm_params_per_channel: int
    hardware: HardwareNode

    @property
    def feature_map(self) -> Quantity:
        return (self.channels * self.height * self.width * self.bytes_per_value).to("KB")

    @property
    def conv_weights(self) -> Quantity:
        return (self.channels * self.channels * self.kernel_size ** 2 * self.bytes_per_value).to("MB")

    @property
    def norm_params(self) -> Quantity:
        return (self.channels * self.norm_params_per_channel * self.bytes_per_value).to("KB")

    @property
    def conv_traffic(self) -> Quantity:
        return (2 * self.feature_map + self.conv_weights).to("MB")

    @property
    def norm_traffic(self) -> Quantity:
        return (2 * self.feature_map + self.norm_params).to("MB")

    @property
    def relu_traffic(self) -> Quantity:
        return (2 * self.feature_map).to("MB")

    @property
    def unfused_trips(self) -> int:
        return 6

    @property
    def fused_trips(self) -> int:
        return 2

    @property
    def unfused_traffic(self) -> Quantity:
        return (self.conv_traffic + self.norm_traffic + self.relu_traffic).to("MB")

    @property
    def fused_traffic(self) -> Quantity:
        return self.conv_traffic

    @property
    def eliminated_activation_traffic(self) -> Quantity:
        """Activation transfers fusion removes (four feature-map trips)."""
        return ((self.unfused_trips - self.fused_trips) * self.feature_map).to("MB")

    @property
    def traffic_cut(self) -> float:
        return 1 - (self.fused_traffic / self.unfused_traffic).to("").magnitude

    @property
    def unfused_time(self) -> Quantity:
        return (self.unfused_traffic / self.hardware.memory.bandwidth).to("microsecond")

    @property
    def fused_time(self) -> Quantity:
        return (self.fused_traffic / self.hardware.memory.bandwidth).to("microsecond")

    @property
    def time_ratio(self) -> float:
        return (self.unfused_time / self.fused_time).to("").magnitude


class SparseMatrixScenario(BaseScenario):
    """A small weight matrix, magnitude-pruned at ``threshold`` (0 keeps every nonzero),
    counted as a dense product and stored dense or in CSR."""
    matrix: List[List[float]]
    value_bytes: Quantity
    index_bytes: Quantity
    threshold: float = 0.0

    @property
    def pruned(self) -> List[List[float]]:
        return [[w if abs(w) >= self.threshold else 0 for w in row] for row in self.matrix]

    @property
    def survivors(self) -> list:
        """Surviving nonzero weights, row by row."""
        return [w for row in self.pruned for w in row if w != 0]

    @property
    def removed(self) -> list:
        """Nonzero weights the threshold zeroed, row by row."""
        return [w for row, kept in zip(self.matrix, self.pruned) for w, k in zip(row, kept) if w != 0 and k == 0]

    @property
    def rows(self) -> int:
        return len(self.matrix)

    @property
    def entries(self) -> int:
        return sum(len(row) for row in self.matrix)

    @property
    def nonzeros(self) -> int:
        return len(self.survivors)

    @property
    def zeros(self) -> int:
        return self.entries - self.nonzeros

    @property
    def zero_share(self) -> float:
        return self.zeros / self.entries

    @property
    def dense_bytes(self) -> Quantity:
        return (self.entries * self.value_bytes).to("byte")

    @property
    def csr_value_index_bytes(self) -> Quantity:
        """Values and their column indices, one pair per nonzero."""
        return (self.nonzeros * (self.value_bytes + self.index_bytes)).to("byte")

    @property
    def row_pointers(self) -> int:
        return self.rows + 1

    @property
    def row_pointer_bytes(self) -> Quantity:
        return (self.row_pointers * self.index_bytes).to("byte")

    @property
    def csr_bytes(self) -> Quantity:
        from ..physics.memory import calc_csr_bytes
        return calc_csr_bytes(self.nonzeros, self.rows, self.value_bytes, self.index_bytes).to("byte")


class SparseFlashFitScenario(BaseScenario):
    """``workload`` pruned until its surviving values total ``surviving_bytes``, stored with one
    ``index_bytes`` position index per value, against a ``flash_budget``."""
    workload: Workload
    value_bytes: Quantity
    index_bytes: Quantity
    surviving_bytes: Quantity
    flash_budget: Quantity

    @property
    def dense_bytes(self) -> Quantity:
        return self.workload.size_in_bytes(self.value_bytes).to("MB")

    @property
    def pruned_share(self) -> float:
        return 1 - (self.surviving_bytes / self.dense_bytes).to("").magnitude

    @property
    def bytes_per_nonzero(self) -> Quantity:
        from ..physics.memory import calc_sparse_bytes_per_nonzero
        return calc_sparse_bytes_per_nonzero(self.value_bytes, self.index_bytes)

    @property
    def stored_bytes(self) -> Quantity:
        return (self.surviving_bytes * (self.bytes_per_nonzero / self.value_bytes)).to("MB")

    @property
    def fits(self) -> bool:
        return self.stored_bytes <= self.flash_budget


class ChannelTileScenario(BaseScenario):
    """A layer of ``channels`` pruned to ``pruned_channels``, run on a matrix unit that
    processes channels in tiles of ``tile`` lanes."""
    channels: int
    pruned_channels: int
    tile: int

    @property
    def tiles(self) -> int:
        return -(-self.pruned_channels // self.tile)

    @property
    def full_tiles(self) -> int:
        return self.pruned_channels // self.tile

    @property
    def last_tile_lanes(self) -> int:
        """Busy lanes in the last tile (a full tile when the count divides evenly)."""
        return self.pruned_channels % self.tile or self.tile

    @property
    def remainder(self) -> int:
        return self.pruned_channels % self.tile

    @property
    def work_removed(self) -> float:
        return 1 - self.pruned_channels / self.channels

    # Padding view (added 2026-10-20): the unit runs whole tiles, so the layer
    # occupies tiles x tile lanes and uses pruned_channels of them.
    @property
    def padded_channels(self) -> int:
        return self.tiles * self.tile

    @property
    def utilization(self) -> float:
        """Busy share of the lanes the layer occupies: pruned_channels / padded_channels."""
        return self.pruned_channels / self.padded_channels


class SoftTargetScenario(BaseScenario):
    """A teacher's class probabilities softened at ``temperature``: softmax(log p / T)."""
    labels: List[str]
    probabilities: List[float]
    temperature: float

    @property
    def softened(self) -> list:
        from ..physics.agents import calc_tempered_softmax
        return calc_tempered_softmax([_math.log(p) for p in self.probabilities], self.temperature)

    @property
    def uniform(self) -> float:
        """The limit every class approaches as the temperature grows."""
        return 1 / len(self.probabilities)


class SpeedupCompositionScenario(BaseScenario):
    """Independent paper speedups multiplied together, against the ``measured`` speedup."""
    factors: Dict[str, float]
    measured: float

    @property
    def paper(self) -> float:
        out = 1.0
        for f in self.factors.values():
            out *= f
        return out

    @property
    def realized_share(self) -> float:
        return self.measured / self.paper


class WorkLatencyGapScenario(BaseScenario):
    """A cut in work (parameters or FLOPs) against the latency cut it delivered."""
    work_reduction: float
    latency_reduction: float

    @property
    def delivered_share(self) -> float:
        """Share of the work cut that reached latency."""
        return self.latency_reduction / self.work_reduction


class StagedCompressionScenario(BaseScenario):
    """A teacher pruned and distilled into a compact student, then quantized, with the
    task accuracy after each stage and the loss when quantization comes first."""
    teacher: Workload
    student_params: Quantity
    float_bytes: Quantity
    int_bytes: Quantity
    baseline_accuracy: float
    pruned_accuracy: float
    distilled_accuracy: float
    final_accuracy: float
    mobile_speedup: float
    quantize_first_loss: float

    @property
    def full_bytes(self) -> Quantity:
        return self.teacher.size_in_bytes(self.float_bytes).to("MB")

    @property
    def student_float_bytes(self) -> Quantity:
        return (self.student_params.to("param").magnitude * self.float_bytes).to("MB")

    @property
    def compressed_bytes(self) -> Quantity:
        return (self.student_params.to("param").magnitude * self.int_bytes).to("MB")

    @property
    def parameter_reduction(self) -> float:
        return 1 - (self.student_params / self.teacher.parameters).to("").magnitude

    @property
    def compression_ratio(self) -> float:
        return (self.full_bytes / self.compressed_bytes).to("").magnitude

    @property
    def final_loss(self) -> float:
        return self.baseline_accuracy - self.final_accuracy
# --- Hardware acceleration: reuse, rooflines, and limits on one accelerator (added 2026-10-20) ---


class ElementwiseOpScenario(BaseScenario):
    """An element-wise or normalization kernel over ``elements`` values, run unfused.

    Each element is read ``reads`` times and written ``writes`` times from
    device memory; ``parameter_values`` (a LayerNorm's scale and shift) are
    read once. ``flops_per_element`` counts the arithmetic on each element.
    """
    elements: int
    flops_per_element: float
    bytes_per_value: Quantity
    parameter_values: int = 0
    reads: int = 1
    writes: int = 1

    @property
    def flops(self) -> Quantity:
        return Q_(self.elements * self.flops_per_element, "flop")

    @property
    def input_bytes(self) -> Quantity:
        return (self.reads * self.elements * self.bytes_per_value).to("byte")

    @property
    def output_bytes(self) -> Quantity:
        return (self.writes * self.elements * self.bytes_per_value).to("byte")

    @property
    def parameter_bytes(self) -> Quantity:
        return (self.parameter_values * self.bytes_per_value).to("byte")

    @property
    def traffic_bytes(self) -> Quantity:
        return self.input_bytes + self.output_bytes + self.parameter_bytes

    @property
    def intensity(self) -> float:
        """FLOP per byte of the unfused traffic."""
        return (self.flops / self.traffic_bytes).to("flop/byte").magnitude


class AcceleratedFractionScenario(BaseScenario):
    """Amdahl's law with a path gain set by two rates: ``fraction`` of the
    baseline time runs ``accelerated_rate / baseline_rate`` times faster.

    The rates may be operation rates (a compute-bound path) or bandwidths (a
    bandwidth-bound path); only their ratio enters.
    """
    fraction: float
    accelerated_rate: Quantity
    baseline_rate: Quantity

    @property
    def gain(self) -> float:
        return (self.accelerated_rate / self.baseline_rate).to("").magnitude

    @property
    def speedup(self) -> float:
        from ..physics import calc_amdahls_speedup
        return calc_amdahls_speedup(self.fraction, self.gain)

    @property
    def ceiling(self) -> float:
        """The speedup with an infinitely fast path: 1 / (1 - fraction)."""
        from ..physics import calc_amdahls_speedup
        return calc_amdahls_speedup(self.fraction, float("inf"))


class SystolicReuseEnergyScenario(BaseScenario):
    """Energy per MAC with every operand from off-chip memory, against a systolic array.

    Streaming: ``naive_accesses`` off-chip accesses per MAC (load both operands
    and the running sum, write the sum back). Systolic: ``edge_loads`` operands
    enter at the array's edge and serve ``array_dim`` MACs; partial sums stay
    in the array.
    """
    access_energy: Quantity
    mac_energy: Quantity
    naive_accesses: int
    edge_loads: int
    array_dim: int

    @property
    def naive_energy(self) -> Quantity:
        return (self.naive_accesses * self.access_energy + self.mac_energy).to("pJ")

    @property
    def systolic_accesses(self) -> float:
        """Off-chip accesses per MAC in the array."""
        return self.edge_loads / self.array_dim

    @property
    def systolic_energy(self) -> Quantity:
        return (self.systolic_accesses * self.access_energy + self.mac_energy).to("pJ")

    @property
    def ratio(self) -> float:
        return (self.naive_energy / self.systolic_energy).to("").magnitude


class MatrixInstructionScenario(BaseScenario):
    """One warp-wide matrix instruction of shape ``m`` x ``n`` x ``k``, issued by ``threads`` threads."""
    m: int
    n: int
    k: int
    threads: int

    @property
    def macs(self) -> int:
        return self.m * self.n * self.k

    @property
    def macs_per_thread(self) -> float:
        """Scalar fused multiply-adds each thread would otherwise issue."""
        return self.macs / self.threads


class ParameterBandwidthGrowthScenario(BaseScenario):
    """Dense-model parameter growth against per-device memory-bandwidth growth.

    Each series is indexed to its baseline entry; the growth is the largest
    index. ``sparse_models`` are reported but kept out of the growth figure,
    because their total parameter count does not set per-token traffic.
    """
    accelerators: List[HardwareNode]
    dense_models: List[Workload]
    sparse_models: List[Workload] = []
    baseline_accelerator: HardwareNode
    baseline_model: Workload

    @property
    def bandwidth_index(self) -> List[float]:
        b0 = self.baseline_accelerator.memory.bandwidth
        return [(h.memory.bandwidth / b0).to("").magnitude for h in self.accelerators]

    @property
    def dense_index(self) -> List[float]:
        p0 = self.baseline_model.parameters
        return [(m.parameters / p0).to("").magnitude for m in self.dense_models]

    @property
    def sparse_index(self) -> List[float]:
        p0 = self.baseline_model.parameters
        return [(m.parameters / p0).to("").magnitude for m in self.sparse_models]

    @property
    def bandwidth_growth(self) -> float:
        return max(self.bandwidth_index)

    @property
    def parameter_growth(self) -> float:
        return max(self.dense_index)


class DeviceEnvelopeScenario(BaseScenario):
    """Best-device-to-date envelopes of two columns of a ``Literature.Series`` device table.

    Rows are ordered by ``date_key``; each envelope is the running maximum,
    and growth is the final envelope over the earliest device's value.
    """
    series_name: str
    date_key: str
    compute_key: str
    bandwidth_key: str

    @property
    def rows(self) -> List[dict]:
        from ..literature.registry import Literature
        return sorted(getattr(Literature.Series, self.series_name).entries, key=lambda r: r[self.date_key])

    def _envelope(self, key) -> List[tuple]:
        best, out = 0.0, []
        for r in self.rows:
            best = max(best, r[key])
            out.append((r[self.date_key], best))
        return out

    @property
    def compute_envelope(self) -> List[tuple]:
        return self._envelope(self.compute_key)

    @property
    def bandwidth_envelope(self) -> List[tuple]:
        return self._envelope(self.bandwidth_key)

    @property
    def device_count(self) -> int:
        return len(self.rows)

    @property
    def baseline(self) -> dict:
        return self.rows[0]

    @property
    def compute_growth(self) -> float:
        return self.compute_envelope[-1][1] / self.baseline[self.compute_key]

    @property
    def bandwidth_growth(self) -> float:
        return self.bandwidth_envelope[-1][1] / self.baseline[self.bandwidth_key]

    @property
    def best_bandwidth_since(self) -> str:
        """Date of the first device that reached today's best bandwidth."""
        env = self.bandwidth_envelope
        return next(d for d, v in env if v == env[-1][1])


class AcceleratorGenerationsScenario(BaseScenario):
    """Peak compute against memory bandwidth across ``accelerators``, first to last.

    The ridge point of each is its peak over its bandwidth; the gap growth is
    how far the ridge rose from the first to the last.
    """
    accelerators: List[HardwareNode]

    @property
    def ridges(self) -> List[float]:
        return [h.ridge_point().to("flop/byte").magnitude for h in self.accelerators]

    @property
    def compute_growth(self) -> float:
        a, b = self.accelerators[0], self.accelerators[-1]
        return (b.compute.peak_flops / a.compute.peak_flops).to("").magnitude

    @property
    def bandwidth_growth(self) -> float:
        a, b = self.accelerators[0], self.accelerators[-1]
        return (b.memory.bandwidth / a.memory.bandwidth).to("").magnitude

    @property
    def gap_growth(self) -> float:
        return self.compute_growth / self.bandwidth_growth


class LatencyInCyclesScenario(BaseScenario):
    """A memory latency counted in cycles of a processor clocked at ``clock``."""
    latency: Quantity
    clock: Quantity

    @property
    def cycle_time(self) -> Quantity:
        return (1 / self.clock).to("ps")

    @property
    def cycles(self) -> float:
        return (self.latency * self.clock).to("").magnitude


class PeakRatioRangeScenario(BaseScenario):
    """One peak rate against a baseline stated as a range, ``baseline_low`` to ``baseline_high``."""
    rate: Quantity
    baseline_low: Quantity
    baseline_high: Quantity

    @property
    def ratio_low(self) -> float:
        """Against the high baseline."""
        return (self.rate / self.baseline_high).to("").magnitude

    @property
    def ratio_high(self) -> float:
        """Against the low baseline."""
        return (self.rate / self.baseline_low).to("").magnitude
# --- Trajectory planning (added 2026-10-20) ---------------------------------
# Each scenario names one machine (``Embodied.MobileManipulator.*``), its site
# (``Embodied.Scenario.*``), or one joint (``Actuators.*``), and states the
# inputs those records do not carry; the solvers live in
# ``engine/solvers/trajectory.py``.

class TwoLinkJointRateScenario(BaseScenario):
    """``machine``'s arm as a planar two-link chain at ``elbow_angle``: the joint rates its TCP
    speed limit asks in the weakest direction, exact and with damped least squares at ``damping``."""
    machine: Any                                 # MobileManipulator
    elbow_angle: Quantity
    damping: Quantity

    @property
    def result(self):
        from .solvers import TwoLinkJointRateModel
        arm = self.machine.arm
        return TwoLinkJointRateModel().solve(arm.upper_arm_length, arm.forearm_length, self.elbow_angle,
                                             self.machine.tcp_speed_limit, arm.joint_velocity_limits[1],
                                             self.damping)


class HandoffExposureScenario(BaseScenario):
    """A replanner at ``replan_rate`` over ``mission_time`` whose handoffs are each late with
    ``late_probability``, held to ``mission_budget`` for the whole mission."""
    mission_time: Quantity
    replan_rate: Quantity
    late_probability: float
    mission_budget: float

    @property
    def result(self):
        from .solvers import HandoffExposureModel
        return HandoffExposureModel().solve(self.mission_time, self.replan_rate, self.late_probability,
                                            self.mission_budget)


class ChunkReplacementScenario(BaseScenario):
    """Replacing ``machine``'s admitted chunk at ``site``: capture-to-admission latency against the
    renewal period and the chunk lease, with an ``interference_tail`` on the inference pass."""
    machine: Any                                 # MobileManipulator
    site: Any                                    # EmbodiedSiteScenario
    interference_tail: Quantity

    @property
    def renewal_period(self) -> Quantity:
        return (1 / self.machine.chunk_rate).to("ms")

    @property
    def result(self):
        from .solvers import ChunkReplacementModel
        s = self.site
        return ChunkReplacementModel().solve(s.perception_age, s.t_dispatch_to_admission,
                                             self.machine.chunk_inference_latency, s.t_plan_validate,
                                             self.renewal_period, s.t_lease, self.interference_tail)


class JerkLimitedStopScenario(BaseScenario):
    """A machine at ``speed`` that stops along an S-curve at ``deceleration`` after ``delay``.

    The jerk bound is ``jerk`` when stated, else the peak jerk of the reference C2 polynomial stop at
    ``jerk_reference_speed`` (the entry speed by default). With a ``site``, its localization bound and
    protective margin join the required distance and its clear distance closes the budget unless
    ``clear_distance`` is stated."""
    speed: Quantity
    deceleration: Quantity
    delay: Optional[Quantity] = None
    jerk: Optional[Quantity] = None
    jerk_reference_speed: Optional[Quantity] = None
    clear_distance: Optional[Quantity] = None
    site: Optional[Any] = None                   # EmbodiedSiteScenario

    @property
    def d_clear(self) -> Optional[Quantity]:
        if self.clear_distance is not None or self.site is None:
            return self.clear_distance
        return self.site.d_clear

    @property
    def result(self):
        from .solvers import JerkLimitedStopModel
        loc = None if self.site is None else self.site.delta_loc
        margin = None if self.site is None else self.site.delta_margin
        return JerkLimitedStopModel().solve(self.speed, self.deceleration, self.jerk, self.jerk_reference_speed,
                                            self.delay, self.d_clear, loc, margin)


class SpillLimitedStopScenario(BaseScenario):
    """The ``free_stop`` scenario's machine carrying an open container of ``inner_radius`` filled
    ``freeboard`` below the rim: the deceleration the liquid allows, that stop, and the carry speed."""
    free_stop: JerkLimitedStopScenario
    inner_radius: Quantity
    freeboard: Quantity

    @property
    def result(self):
        from .solvers import SpillLimitedStopModel
        f = self.free_stop
        return SpillLimitedStopModel().solve(f.speed, f.deceleration, f.result.jerk, f.delay, f.d_clear,
                                             self.inner_radius, self.freeboard)


class TrajectoryDeadlinesScenario(BaseScenario):
    """A trajectory record for ``machine``'s arm at ``site``: a replacement requested at
    ``request_delay`` with a ``latency_target`` and ``schedule_slack``, a ``blend`` deadline, the
    ``commit`` instant when ``stop`` (a JerkLimitedStopScenario) must begin, and a ``late_wait``."""
    machine: Any                                 # MobileManipulator
    site: Any                                    # EmbodiedSiteScenario
    stop: JerkLimitedStopScenario
    request_delay: Quantity
    latency_target: Quantity
    schedule_slack: Quantity
    blend: Quantity
    commit: Quantity
    late_wait: Quantity

    @property
    def response(self) -> Quantity:
        """Decision to physical onset: one tick, one bus cycle, and the brake onset."""
        s = self.site
        return (s.t_tick + s.t_bus + s.t_brake_onset).to("ms")

    @property
    def result(self):
        from .solvers import TrajectoryDeadlinesModel
        rm, st = self.machine, self.stop.result
        return TrajectoryDeadlinesModel().solve(
            rm.chunk_step, rm.chunk_horizon, (1 / rm.chunk_rate).to("ms"), self.request_delay,
            self.latency_target, self.schedule_slack, self.response, self.blend, self.commit,
            st.stop_time, st.scurve_travel, self.stop.speed, self.stop.d_clear, self.late_wait)


class SeamBridgeScenario(BaseScenario):
    """A ``velocity_gap`` at a chunk seam on ``joint``, closed in one ``tick`` or by a quintic bridge
    over ``bridge_duration``, against ``machine``'s arm torque-rate limit. ``loaded_inertia`` (optional)
    prices the one-tick step with the payload's inertia added."""
    joint: Any                                   # an Actuators.PandaClassJoint entry
    machine: Any                                 # MobileManipulator
    velocity_gap: Quantity
    tick: Quantity
    bridge_duration: Quantity
    loaded_inertia: Optional[Quantity] = None

    @property
    def result(self):
        from .solvers import SeamBridgeModel
        j = self.joint
        return SeamBridgeModel().solve(j.gear_ratio, j.rotor_inertia, j.rated_torque, j.peak_torque,
                                       self.machine.arm.joint_torque_rate_limit, self.velocity_gap, self.tick,
                                       self.bridge_duration, self.loaded_inertia)


class ShoulderSwingScenario(BaseScenario):
    """``machine``'s shoulder holding its rated payload at full planar reach, with ``joint``'s
    reflected rotor inertia, swinging ``swing`` rest to rest inside its datasheet limits."""
    machine: Any                                 # MobileManipulator
    joint: Any                                   # an Actuators.PandaClassJoint entry
    swing: Quantity

    @property
    def reach(self) -> Quantity:
        arm = self.machine.arm
        return (arm.upper_arm_length + arm.forearm_length).to("m")

    @property
    def result(self):
        from .solvers import ShoulderSwingModel
        arm = self.machine.arm
        return ShoulderSwingModel().solve(arm.payload_capacity, self.reach, arm.max_torque,
                                          self.joint.reflected_inertia, self.swing,
                                          arm.joint_acceleration_limits[1], arm.joint_velocity_limits[1],
                                          arm.joint_torque_rate_limit)


class DynamicWindowScenario(BaseScenario):
    """``machine``'s base over one chunk renewal at ``site``: its dynamic window and the classical
    admissible speed against the site's clear distance and credible deceleration."""
    machine: Any                                 # MobileManipulator
    site: Any                                    # EmbodiedSiteScenario

    @property
    def result(self):
        from .solvers import DynamicWindowModel
        b = self.machine.base
        return DynamicWindowModel().solve(b.max_acceleration, (1 / self.machine.chunk_rate).to("ms"),
                                          self.site.d_clear, self.site.a_brake, b.max_velocity)


class StaleAnchorScenario(BaseScenario):
    """A joint of ``machine``'s arm turning at ``joint_speed`` while a replacement takes ``delay``,
    anchored at the pose sampled at the request; the lever arm is the planar reach."""
    machine: Any                                 # MobileManipulator
    joint_speed: Quantity
    delay: Quantity

    @property
    def result(self):
        from .solvers import StaleAnchorModel
        arm = self.machine.arm
        return StaleAnchorModel().solve(self.joint_speed, self.delay, arm.upper_arm_length + arm.forearm_length)


# --- Trajectory curation: triage, split leakage, admission order, denominators, staging (added 2026-10-20) ---

class FailureTriageScenario(BaseScenario):
    """``runs`` trajectories a month failing at ``failure_rate``, each probe's attributed share, and a
    fine-tuning run on every failure of ``workload`` on ``devices`` x ``hardware``."""
    runs: int
    failure_rate: float
    probe_shares: Dict[str, float]      # failures each counterfactual probe attributes to its rung
    workload: Workload
    hardware: HardwareNode
    devices: int
    utilization: float                  # model FLOPs utilization
    hourly_price: Quantity              # USD per device-hour
    sequence_tokens: int
    epochs: int

    @property
    def result(self):
        from .solvers import FailureTriageModel
        return FailureTriageModel().solve(self.runs, self.failure_rate, self.probe_shares,
                                          self.workload.parameters, self.sequence_tokens, self.epochs,
                                          self.devices, self.hardware.compute.peak_flops, self.utilization,
                                          self.hourly_price)


class RunSplitLeakageScenario(BaseScenario):
    """``attempts`` runs per task, each assigned to training with probability ``train_fraction``."""
    attempts: int
    train_fraction: float
    tasks: Optional[int] = None

    @property
    def result(self):
        from .solvers import RunSplitLeakageModel
        return RunSplitLeakageModel().solve(self.attempts, self.train_fraction, self.tasks)


class AdmissionOrderScenario(BaseScenario):
    """Independent admission checks (pass probability, slot time) run in each of ``orders``."""
    checks: Dict[str, Tuple[float, Quantity]]
    orders: List[List[str]]
    candidates: Optional[int] = None

    @property
    def result(self):
        from .solvers import AdmissionOrderModel
        return AdmissionOrderModel().solve(self.checks, self.orders, self.candidates)


class CorpusDenominatorsScenario(BaseScenario):
    """A retained pool of record classes (records, stored tokens per record, selected target tokens per
    record); ``focus`` is the class whose shares are read, ``admitted`` the classes a route admits."""
    classes: Dict[str, Tuple[int, Quantity, Quantity]]
    focus: str
    admitted: List[str]

    @property
    def result(self):
        from .solvers import CorpusDenominatorsModel
        return CorpusDenominatorsModel().solve(self.classes, self.focus, self.admitted)


class AttemptGroupContrastScenario(BaseScenario):
    """``attempts`` independent attempts per fixture at each of ``pass_rates``."""
    attempts: int
    pass_rates: List[float]

    def success_yield(self, pass_rate: float) -> float:
        """Chance of at least one success."""
        from ..physics.agents import calc_independent_success
        return calc_independent_success(pass_rate, self.attempts)

    def mixed_fraction(self, pass_rate: float) -> float:
        """Chance of at least one success and one failure."""
        from ..physics.agents import calc_mixed_outcome_fraction
        return calc_mixed_outcome_fraction(pass_rate, self.attempts)

    def attempts_per_success(self, pass_rate: float) -> float:
        """Attempts spent per fixture that yields a success."""
        return self.attempts / self.success_yield(pass_rate)


class StagingBurstScenario(BaseScenario):
    """A collector of ``rollouts`` concurrent rollouts feeding ``slots`` verifiers, whose checks
    stall at ``timeout``, with records charged at ``record_ceiling`` against ``store``."""
    rollouts: int
    rollout_time: Quantity
    slots: int
    mean_service: Quantity
    timeout: Quantity
    record_ceiling: Quantity
    store: Quantity
    reaction: Quantity

    @property
    def result(self):
        from .solvers import StagingBurstModel
        return StagingBurstModel().solve(self.rollouts, self.rollout_time, self.slots, self.mean_service,
                                         self.timeout, self.record_ceiling, self.store, self.reaction)


class CohortDispositionScenario(BaseScenario):
    """A finite cohort of attempts by disposition at a cutoff, its total cost, and the disposition
    that emits a learning artifact."""
    dispositions: Dict[str, int]
    emitted: str
    cohort_cost: Quantity
    yield_multiple: Optional[float] = None   # a raised emission yield to compare, as a multiple

    @property
    def attempts(self) -> int:
        return sum(self.dispositions.values())

    @property
    def emission_yield(self) -> float:
        return self.dispositions[self.emitted] / self.attempts

    @property
    def cost_per_attempt(self) -> Quantity:
        return (self.cohort_cost / self.attempts).to("USD")

    def cost_per_artifact(self, emission_yield: Optional[float] = None) -> Quantity:
        """Attempt cost over emission yield; another yield prices the same attempts at that yield."""
        from ..physics.agents import calc_cost_per_accepted
        y = self.emission_yield if emission_yield is None else emission_yield
        return calc_cost_per_accepted(self.cost_per_attempt, y).to("USD")

# --- Fine-tuning on trajectories (added 2026-10-20) ---------------------------
# One serialized example's target mask, one example per call, recorded role
# shares, loss weighting, packing, a preference pair, context distillation,
# and a release comparison; solvers in engine/solvers/fine_tuning.py.

class TargetMaskScenario(BaseScenario):
    """One serialized training example by segment, and the responses kept as context only."""
    prompt: int                   # includes the initial boundary position
    task: int
    rationale: int                # optional reviewed rationale
    response: int                 # policy responses with their stop tokens
    observation: int
    unselected_response: int      # responses retained as context, not targets

    @property
    def result(self):
        from .solvers import TargetMaskModel
        return TargetMaskModel().solve(self.prompt, self.task, self.rationale, self.response,
                                       self.observation, self.unselected_response)


class PerInvocationSerializationScenario(BaseScenario):
    """A ``turns``-turn trajectory stored as one example per model call, each turn adding ``tokens_per_turn``."""
    turns: int
    tokens_per_turn: int

    @property
    def result(self):
        from .solvers import PerInvocationSerializationModel
        return PerInvocationSerializationModel().solve(self.turns, self.tokens_per_turn)


class TrajectoryRoleShareScenario(BaseScenario):
    """Transcript characters by role, per configuration, over a per-trajectory ``Literature`` series."""
    series_name: str

    @property
    def rows(self):
        from ..literature.registry import Literature
        return getattr(Literature, self.series_name).entries

    @property
    def result(self):
        from .solvers import TrajectoryRoleShareModel
        return TrajectoryRoleShareModel().solve(self.rows)
# --- Benchmarking (added 2026-10-20) -----------------------------------------
class EfficiencyClaimGapScenario(BaseScenario):
    """A datasheet operating point against a sustained one: operations per watt at each, and their ratio.

    The gap is the product of the throughput fall and the power rise, so a
    reader who compares only the throughputs underestimates it.
    """
    claimed_rate: Quantity
    claimed_power: Quantity
    sustained_rate: Quantity
    sustained_power: Quantity

    @property
    def claimed_efficiency(self) -> Quantity:
        return (self.claimed_rate / self.claimed_power).to("TOPS/W")

    @property
    def sustained_efficiency(self) -> Quantity:
        return (self.sustained_rate / self.sustained_power).to("TOPS/W")

    @property
    def gap(self) -> float:
        return (self.claimed_efficiency / self.sustained_efficiency).to("").magnitude

    @property
    def rate_fall(self) -> float:
        return (self.claimed_rate / self.sustained_rate).to("").magnitude

    @property
    def power_rise(self) -> float:
        return (self.sustained_power / self.claimed_power).to("").magnitude


class HostAcceleratorBoardScenario(BaseScenario):
    """One board, a host CPU and an attached accelerator: model-only and end-to-end latency, active energy.

    The end-to-end times add host preprocessing (and, on the accelerator path,
    the copy across the host link) to the model time; active energy is the
    component's active power times its model-only latency, so it leaves out
    preprocessing and idle power.
    """
    host_fp32_latency: Quantity
    host_int8_latency: Quantity
    accelerator_latency: Quantity
    host_end_to_end: Quantity
    accelerator_end_to_end: Quantity
    host_power: Quantity
    accelerator_power: Quantity
    frame_rate: Optional[Quantity] = None   # frames per second the camera delivers

    @property
    def frame_budget(self) -> Quantity:
        """Time per frame at ``frame_rate``: the latency each inference must meet."""
        return (1 / self.frame_rate).to("ms")

    @property
    def int8_speedup(self) -> float:
        return (self.host_fp32_latency / self.host_int8_latency).to("").magnitude

    @property
    def isolated_speedup(self) -> float:
        return (self.host_int8_latency / self.accelerator_latency).to("").magnitude

    @property
    def end_to_end_speedup(self) -> float:
        return (self.host_end_to_end / self.accelerator_end_to_end).to("").magnitude

    @property
    def power_ratio(self) -> float:
        return (self.accelerator_power / self.host_power).to("").magnitude

    @property
    def host_energy(self) -> Quantity:
        return (self.host_power * self.host_int8_latency).to("mJ")

    @property
    def accelerator_energy(self) -> Quantity:
        return (self.accelerator_power * self.accelerator_latency).to("mJ")

    @property
    def energy_ratio(self) -> float:
        return (self.host_energy / self.accelerator_energy).to("").magnitude


class PeriodicInferencePowerScenario(BaseScenario):
    """``workload`` run once per ``period`` on ``hardware`` at its peak ``precision`` rate, idling between runs.

    Active power is the board's ``tdp`` and idle power its ``tdp_min``; average
    power weights each by its share of the period (``DutyCycledPowerModel``).
    ``energy_cut`` (optional) divides the energy of each inference, to show how
    little a cheaper inference moves a duty-cycled average.
    """
    workload: Workload
    hardware: HardwareNode
    period: Quantity
    precision: str = "int8"
    energy_cut: Optional[float] = None

    @property
    def inference_time(self) -> Quantity:
        return (self.workload.inference_flops / self.hardware.compute.precision_flops[self.precision]).to("ms")

    @property
    def duty(self) -> float:
        return (self.inference_time / self.period).to("").magnitude

    @property
    def active_power(self) -> Quantity:
        return self.hardware.tdp

    @property
    def idle_power(self) -> Quantity:
        return self.hardware.tdp_min

    @property
    def inference_energy(self) -> Quantity:
        return (self.active_power * self.inference_time).to("mJ")

    @property
    def result(self):
        from .solvers import DutyCycledPowerModel
        return DutyCycledPowerModel().solve(
            {"active": self.active_power, "idle": self.idle_power},
            {"active": self.duty, "idle": 1 - self.duty},
            self.active_power,
        )

    @property
    def active_contribution(self) -> Quantity:
        return self.result.contributions["active"].to("mW")

    @property
    def idle_contribution(self) -> Quantity:
        return self.result.contributions["idle"].to("mW")

    @property
    def average_power(self) -> Quantity:
        return self.result.average_power.to("mW")

    @property
    def idle_share(self) -> float:
        return (self.idle_contribution / self.average_power).to("").magnitude

    @property
    def crossover_duty(self) -> float:
        """Duty cycle at which active energy equals idle energy."""
        return (self.idle_power / (self.active_power + self.idle_power)).to("").magnitude

    @property
    def cut_saving(self) -> Quantity:
        """Average power saved when each inference costs ``1 / energy_cut`` of its energy."""
        return (self.active_contribution * (1 - 1 / self.energy_cut)).to("mW")

    @property
    def cut_gain(self) -> float:
        return (self.cut_saving / self.average_power).to("").magnitude


class MeasurementBoundaryEnergyScenario(BaseScenario):
    """One request's energy under a narrow and a full measurement boundary (``Literature.Series`` rows).

    Each row is a component with its energy and a counted flag under each
    boundary; the two totals sum the counted rows.
    """
    series_name: str
    narrow_key: str = "existing"
    full_key: str = "comprehensive"

    @property
    def rows(self):
        from ..literature.registry import Literature
        return getattr(Literature.Series, self.series_name).entries

    def _total(self, key: str) -> float:
        return sum(r[f"{key}_wh"] for r in self.rows if r[f"{key}_counted"] == 1)

    @property
    def narrow(self) -> Quantity:
        return Q_(self._total(self.narrow_key), "Wh")

    @property
    def full(self) -> Quantity:
        return Q_(self._total(self.full_key), "Wh")

    @property
    def ratio(self) -> float:
        return (self.full / self.narrow).to("").magnitude

    def share(self, *prefixes: str) -> float:
        """Share of the full-boundary total in the components whose names start with ``prefixes``."""
        part = sum(r[f"{self.full_key}_wh"] for r in self.rows if r["component"].startswith(prefixes))
        return part / self._total(self.full_key)


class LossNormalizationScenario(BaseScenario):
    """A long trajectory split into ``focus_pieces`` records, batched with ``other_examples`` short ones."""
    focus_targets: int
    other_examples: int
    other_targets: int
    focus_pieces: int = 1

    @property
    def result(self):
        from .solvers import LossNormalizationModel
        return LossNormalizationModel().solve(self.focus_targets, self.other_examples,
                                              self.other_targets, self.focus_pieces)


class SequencePackingScenario(BaseScenario):
    """Stored example lengths against ``capacity``-token buffers."""
    lengths: List[int]
    capacity: int

    @property
    def result(self):
        from .solvers import SequencePackingModel
        return SequencePackingModel().solve(self.lengths, self.capacity)


class PreferencePairScenario(BaseScenario):
    """Log-probability sums over a preferred and a dispreferred record's selected tokens."""
    policy_preferred: float
    reference_preferred: float
    policy_dispreferred: float
    reference_dispreferred: float
    beta: float

    @property
    def result(self):
        from .solvers import PreferencePairModel
        return PreferencePairModel().solve(self.policy_preferred, self.reference_preferred,
                                           self.policy_dispreferred, self.reference_dispreferred,
                                           self.beta)


class ContextDistillationScenario(BaseScenario):
    """A procedural prefix read every turn, against distilling it and redistilling per revision."""
    prefix_tokens: int
    turns_per_session: int
    sessions_per_month: float
    input_price: Quantity          # USD per token
    cached_fraction: float         # cached reads billed at this share of the input price
    redistill_cost: Quantity       # USD per redistillation
    revisions_per_month: float

    @property
    def result(self):
        from .solvers import ContextDistillationModel
        return ContextDistillationModel().solve(self.prefix_tokens, self.turns_per_session,
                                                self.sessions_per_month, self.input_price,
                                                self.cached_fraction, self.redistill_cost,
                                                self.revisions_per_month)


class CandidateReleaseScenario(BaseScenario):
    """A fine-tuned candidate against the baseline: acceptance, attempt cost, and schema metrics."""
    trials: int                              # per arm
    baseline_acceptance: float
    baseline_attempt_cost: Quantity
    candidate_acceptance: float
    candidate_attempt_cost: Quantity
    baseline_sfa_familiar: float             # schema-following accuracy, familiar schemas
    candidate_sfa_familiar: float
    baseline_sfa_renamed: float              # the same on renamed schemas
    candidate_sfa_renamed: float
    baseline_pre_renamed: float              # parametric recall error on renamed schemas
    candidate_pre_renamed: float

    @property
    def result(self):
        from .solvers import CandidateReleaseModel
        return CandidateReleaseModel().solve(self.trials, self.baseline_acceptance,
                                             self.baseline_attempt_cost, self.candidate_acceptance,
                                             self.candidate_attempt_cost, self.baseline_pre_renamed,
                                             self.candidate_pre_renamed)
# --- Mapping a computation onto an accelerator: arrays, tiles, waves, and compilers (added 2026-10-20) ---


class SystolicFillDrainScenario(BaseScenario):
    """``rows`` input rows streamed through one weight load of an ``array_dim`` x ``array_dim``
    weight-stationary systolic array.

    Operands move one hop per cycle, so in cycle t cell (r, c) works on input row
    t - r - c. The last row enters in cycle rows - 1 and leaves the far corner
    2 (array_dim - 1) cycles later. Loading the weights is not counted.
    """
    array_dim: int
    rows: int

    @property
    def cells(self) -> int:
        return self.array_dim * self.array_dim

    @property
    def fill_drain_cycles(self) -> int:
        """Cycles to fill and drain the array: 2 N - 2."""
        return 2 * self.array_dim - 2

    @property
    def cycles(self) -> int:
        """Cycles from the first row entering to the last result leaving: B + 2 N - 2."""
        return self.rows + self.fill_drain_cycles

    @property
    def busy_fraction(self) -> float:
        """Average share of cells busy: each cell does ``rows`` MACs over ``cycles`` cycles."""
        return self.rows / self.cycles

    def busy_cells(self, cycle: int) -> int:
        """Cells working on some input row in ``cycle``."""
        n = self.array_dim
        return sum(1 for r in range(n) for c in range(n) if 0 <= cycle - r - c < self.rows)

    @property
    def first_full_cycle(self) -> int:
        """First cycle in which the far corner cell starts work."""
        return self.fill_drain_cycles

    @property
    def full_cycles(self) -> int:
        """Cycles in which every cell is busy (zero when rows < 2 N - 1)."""
        return max(0, self.rows - self.fill_drain_cycles)


class MatmulTileScenario(BaseScenario):
    """C = A B with A ``m`` x ``k`` and B ``k`` x ``n``, computed one ``tile_m`` x ``tile_n``
    output tile at a time, stepping through the reduction ``tile_k`` elements per step.

    Operands are ``operand_bytes`` each and the accumulator ``accumulator_bytes``. The
    on-chip store holds one A tile and one B tile (two of each when double buffered)
    and the accumulator tile. Traffic counts element moves between memory and the
    on-chip store: A is loaded once per output-tile column, B once per output-tile
    row, and C written once. The single-cell loop orders (no tile) count one held
    value per cell: the other operands stream once per MAC, and a partial sum
    that is not held is read and written once per MAC.
    """
    m: int
    n: int
    k: int
    tile_m: int
    tile_n: int
    tile_k: int
    operand_bytes: Quantity
    accumulator_bytes: Quantity
    double_buffered: bool = False

    @property
    def macs(self) -> int:
        return self.m * self.n * self.k

    @property
    def flops(self) -> Quantity:
        return Q_(2 * self.macs, "flop")

    @property
    def tiles_m(self) -> int:
        return -(-self.m // self.tile_m)

    @property
    def tiles_n(self) -> int:
        return -(-self.n // self.tile_n)

    @property
    def tiles_k(self) -> int:
        return -(-self.k // self.tile_k)

    @property
    def output_tiles(self) -> int:
        return self.tiles_m * self.tiles_n

    @property
    def tile_products(self) -> int:
        """A-tile-by-B-tile products: every output tile sums ``tiles_k`` of them."""
        return self.output_tiles * self.tiles_k

    @property
    def tile_product_flops(self) -> Quantity:
        return Q_(2 * self.tile_m * self.tile_n * self.tile_k, "flop")

    @property
    def tile_product_bytes(self) -> Quantity:
        """One A tile and one B tile loaded."""
        return ((self.tile_m * self.tile_k + self.tile_k * self.tile_n) * self.operand_bytes).to("byte")

    @property
    def tile_intensity(self) -> float:
        """FLOP per operand byte loaded: (2 / b) T_M T_N / (T_M + T_N), independent of T_K."""
        return (self.tile_product_flops / self.tile_product_bytes).to("flop/byte").magnitude

    @property
    def unblocked_intensity(self) -> float:
        """No reuse: every MAC (2 FLOP) loads its two operands."""
        return (Q_(2, "flop") / (2 * self.operand_bytes)).to("flop/byte").magnitude

    @property
    def compulsory_intensity(self) -> float:
        """Each of A, B, and C moved exactly once, all at ``operand_bytes``."""
        moved = (self.m * self.k + self.k * self.n + self.m * self.n) * self.operand_bytes
        return (self.flops / moved).to("flop/byte").magnitude

    @property
    def buffer_copies(self) -> int:
        return 2 if self.double_buffered else 1

    @property
    def a_tile_bytes(self) -> Quantity:
        return (self.buffer_copies * self.tile_m * self.tile_k * self.operand_bytes).to("byte")

    @property
    def b_tile_bytes(self) -> Quantity:
        return (self.buffer_copies * self.tile_k * self.tile_n * self.operand_bytes).to("byte")

    @property
    def accumulator_tile_bytes(self) -> Quantity:
        return (self.tile_m * self.tile_n * self.accumulator_bytes).to("byte")

    @property
    def operand_tile_bytes(self) -> Quantity:
        """The A and B buffers alone (what a GPU stages in shared memory)."""
        return self.a_tile_bytes + self.b_tile_bytes

    @property
    def live_bytes(self) -> Quantity:
        """Every buffer live at once: 2 T_M T_K + 2 T_K T_N + 4 T_M T_N bytes for FP16 operands, FP32 sums."""
        return self.operand_tile_bytes + self.accumulator_tile_bytes

    def fits(self, capacity: Quantity) -> bool:
        return bool(self.live_bytes <= capacity)

    @property
    def a_bytes(self) -> Quantity:
        return (self.m * self.k * self.operand_bytes).to("byte")

    @property
    def a_elements(self) -> int:
        return self.m * self.k

    @property
    def b_elements(self) -> int:
        return self.k * self.n

    @property
    def c_elements(self) -> int:
        return self.m * self.n

    @property
    def a_moves(self) -> int:
        return self.a_elements * self.tiles_n

    @property
    def b_moves(self) -> int:
        return self.b_elements * self.tiles_m

    @property
    def tiled_moves(self) -> int:
        """Element moves with the tiling: A per output-tile column, B per output-tile row, C once."""
        return self.a_moves + self.b_moves + self.c_elements

    @property
    def global_loads_per_mac(self) -> float:
        """Operand loads from memory per MAC inside a full tile: 1/T_N + 1/T_M."""
        return 1 / self.tile_n + 1 / self.tile_m

    @property
    def partial_sum_moves(self) -> int:
        """A partial sum read and written once per MAC."""
        return 2 * self.macs

    @property
    def a_stationary_moves(self) -> int:
        """One cell holds an A value: A loaded once; B and the partial sums stream."""
        return self.a_elements + self.macs + self.partial_sum_moves

    @property
    def output_stationary_moves(self) -> int:
        """One cell holds the running sum: A and B stream, C written once."""
        return 2 * self.macs + self.c_elements

    @property
    def b_stationary_moves(self) -> int:
        """One cell holds a B value: B loaded once; A and the partial sums stream."""
        return self.b_elements + self.macs + self.partial_sum_moves

    def row_b_reads(self, cells: int) -> int:
        """B reads when each B value passes along a row of ``cells`` cells holding A values."""
        return self.macs // cells


class WaveQuantizationScenario(BaseScenario):
    """An ``m`` x ``n`` output cut into ``tile_m`` x ``tile_n`` tiles, one tile per SM at a time.

    The SMs of ``hardware`` run the tiles in waves; a last, partial wave (the
    tail) takes about as long as a full one.
    """
    m: int
    n: int
    tile_m: int
    tile_n: int
    hardware: HardwareNode

    @property
    def tile_rows(self) -> int:
        return -(-self.m // self.tile_m)

    @property
    def tile_cols(self) -> int:
        return -(-self.n // self.tile_n)

    @property
    def tiles(self) -> int:
        return self.tile_rows * self.tile_cols

    @property
    def sms(self) -> int:
        return self.hardware.compute.sm_count

    @property
    def waves(self) -> int:
        return -(-self.tiles // self.sms)

    @property
    def tail_tiles(self) -> int:
        """Tiles in the last wave when it is partial (zero when the tiles fill whole waves)."""
        return self.tiles % self.sms

    @property
    def tail_share(self) -> float:
        """Share of the SMs the tail wave uses."""
        return self.tail_tiles / self.sms


class MappingSpaceScenario(BaseScenario):
    """Unconstrained mapping choices at one memory level, as three separate toy counts.

    ``tensors`` tensors each placed in one of ``tiers`` ways; a ``loop_depth``-deep
    loop nest in any order; ``parallel_loops`` of ``parallel_candidates`` loops
    chosen, in order, for parallel execution. The three combine by product.
    """
    tensors: int
    tiers: int
    loop_depth: int
    parallel_candidates: int
    parallel_loops: int

    @property
    def placements(self) -> int:
        return self.tiers ** self.tensors

    @property
    def loop_orders(self) -> int:
        return _math.factorial(self.loop_depth)

    @property
    def parallelizations(self) -> int:
        return _math.perm(self.parallel_candidates, self.parallel_loops)

    @property
    def per_level(self) -> int:
        return self.placements * self.loop_orders * self.parallelizations


class RegisterOccupancyScenario(BaseScenario):
    """Threads a kernel using ``registers_per_thread`` 32-bit registers leaves resident on
    one SM of ``hardware`` (register limit only; per-block limits ignored).

    Threads are allocated in whole warps and capped at the SM's resident-thread limit.
    """
    hardware: HardwareNode
    registers_per_thread: int

    @property
    def registers_per_sm(self) -> int:
        """The SM's register file in 32-bit registers."""
        return int((self.hardware.memory.register_file_per_sm / Q_(4, "byte")).to("").magnitude)

    @property
    def threads_by_registers(self) -> int:
        return self.registers_per_sm // self.registers_per_thread

    @property
    def resident_threads(self) -> int:
        warp = self.hardware.compute.warp_size
        return min(self.hardware.compute.max_threads_per_sm, self.threads_by_registers // warp * warp)

    @property
    def occupancy(self) -> float:
        """Resident threads over the SM's limit."""
        return self.resident_threads / self.hardware.compute.max_threads_per_sm


class TimedMatmulScenario(BaseScenario):
    """One ``dim``-square matrix multiply that took ``time``: 2 dim^3 FLOP over the time."""
    dim: int
    time: Quantity

    @property
    def flops(self) -> Quantity:
        return Q_(2 * self.dim ** 3, "flop")

    @property
    def rate(self) -> Quantity:
        return (self.flops / self.time).to("TFLOPs/s")


class MemoryRoundTripScenario(BaseScenario):
    """A tensor of ``tensor_bytes`` crossing to and from memory ``trips`` times at
    ``energy_per_byte`` (a layer boundary writes it and the next layer reads it back)."""
    tensor_bytes: Quantity
    energy_per_byte: Quantity
    trips: int = 2

    @property
    def energy(self) -> Quantity:
        return (self.trips * self.tensor_bytes * self.energy_per_byte).to("microjoule")


# --- Quantization (added 2026-10-20) ------------------------------------------
class QuantizationGridScenario(BaseScenario):
    """Values stored on a ``bits``-bit integer grid over a clipping range.

    Affine (``symmetric=False``): ``s = (beta - alpha) / (2^b - 1)`` and a
    zero-point ``z = qmin + round(-alpha / s)``; codes run 0 to 2^b - 1, or
    -2^(b-1) to 2^(b-1) - 1 when ``signed``. Symmetric: ``s = max(|alpha|,
    |beta|) / (2^(b-1) - 1)``, ``z = 0``, codes +-(2^(b-1) - 1). ``alpha`` and
    ``beta`` default to the values' minimum and maximum (min-max calibration).
    Rounding is to nearest, ties to even.
    """
    values: List[float] = []
    bits: int
    alpha: Optional[float] = None
    beta: Optional[float] = None
    symmetric: bool = False
    signed: bool = False

    @property
    def lo(self) -> float:
        return min(self.values) if self.alpha is None else self.alpha

    @property
    def hi(self) -> float:
        return max(self.values) if self.beta is None else self.beta

    @property
    def steps(self) -> int:
        """Steps across the range of an affine grid, ``2^b - 1``."""
        from ..physics.numerics import calc_int_levels
        return calc_int_levels(self.bits) - 1

    @property
    def qmin(self) -> int:
        from ..physics.numerics import calc_int_max
        if self.symmetric:
            return -calc_int_max(self.bits)
        return -(calc_int_max(self.bits) + 1) if self.signed else 0

    @property
    def qmax(self) -> int:
        from ..physics.numerics import calc_int_max
        return calc_int_max(self.bits) if (self.symmetric or self.signed) else self.steps

    @property
    def scale(self) -> float:
        from ..physics.numerics import calc_affine_scale, calc_symmetric_scale
        if self.symmetric:
            return calc_symmetric_scale(max(abs(self.lo), abs(self.hi)), self.bits)
        return calc_affine_scale(self.lo, self.hi, self.bits)

    @property
    def zero_point_exact(self) -> float:
        """``-alpha / s``, before rounding (0 for a symmetric grid)."""
        from ..physics.numerics import calc_zero_point
        return 0.0 if self.symmetric else calc_zero_point(self.lo, self.scale)[0]

    @property
    def zero_point(self) -> int:
        from ..physics.numerics import calc_zero_point
        return 0 if self.symmetric else calc_zero_point(self.lo, self.scale, self.qmin)[1]

    @property
    def codes(self) -> List[int]:
        from ..physics.numerics import calc_quantize
        return [calc_quantize(x, self.scale, self.zero_point, self.qmin, self.qmax) for x in self.values]

    @property
    def readback(self) -> List[float]:
        from ..physics.numerics import calc_dequantize
        return [calc_dequantize(q, self.scale, self.zero_point) for q in self.codes]

    @property
    def errors(self) -> List[float]:
        """Absolute error of each value after the round trip."""
        return [abs(r - x) for r, x in zip(self.readback, self.values)]

    @property
    def max_rounding_error(self) -> float:
        """Largest error inside the range: half a step."""
        return self.scale / 2


class IntegerDotProductScenario(BaseScenario):
    """One output of an integer layer: symmetric codes for weights and inputs (scale set by
    each tensor's largest magnitude), integer products summed in a wide accumulator, and one
    requantization to the output scale ``s_y``."""
    weights: List[float]
    inputs: List[float]
    output_scale: float
    bits: int = 8
    accumulator_bits: int = 32

    def _grid(self, values):
        return QuantizationGridScenario(name="", description="", values=values, bits=self.bits, symmetric=True)

    @property
    def weight_scale(self) -> float:
        return self._grid(self.weights).scale

    @property
    def input_scale(self) -> float:
        return self._grid(self.inputs).scale

    @property
    def weight_codes(self) -> List[int]:
        return self._grid(self.weights).codes

    @property
    def input_codes(self) -> List[int]:
        return self._grid(self.inputs).codes

    @property
    def products(self) -> List[int]:
        return [a * b for a, b in zip(self.weight_codes, self.input_codes)]

    @property
    def accumulator(self) -> int:
        return sum(self.products)

    @property
    def readback(self) -> float:
        """The accumulator read as a real number: ``acc s_w s_x``."""
        return self.accumulator * self.weight_scale * self.input_scale

    @property
    def exact(self) -> float:
        return sum(w * x for w, x in zip(self.weights, self.inputs))

    @property
    def requant_scale(self) -> float:
        """``M = s_w s_x / s_y``."""
        return self.weight_scale * self.input_scale / self.output_scale

    @property
    def output_code(self) -> int:
        return round(self.accumulator * self.requant_scale)

    @property
    def accumulator_headroom(self) -> int:
        from ..physics.numerics import calc_accumulator_headroom
        return calc_accumulator_headroom(self.bits, self.accumulator_bits)


class OperationEnergyComparisonScenario(BaseScenario):
    """A wide and a narrow format's energy per add, multiply, and multiply-accumulate, and one
    memory read against the narrow multiply-accumulate. The narrow MAC accumulates in a wider
    integer, so its add is ``accumulate_add``."""
    wide_add: Quantity
    wide_mul: Quantity
    narrow_add: Quantity
    narrow_mul: Quantity
    accumulate_add: Quantity
    memory_read: Quantity

    @property
    def wide_mac(self) -> Quantity:
        return self.wide_mul + self.wide_add

    @property
    def narrow_mac(self) -> Quantity:
        return self.narrow_mul + self.accumulate_add

    @property
    def add_ratio(self) -> float:
        return (self.wide_add / self.narrow_add).to("").magnitude

    @property
    def mul_ratio(self) -> float:
        return (self.wide_mul / self.narrow_mul).to("").magnitude

    @property
    def mac_ratio(self) -> float:
        return (self.wide_mac / self.narrow_mac).to("").magnitude

    @property
    def read_over_narrow_mac(self) -> float:
        """The memory read in narrow multiply-accumulates (per-operation energies, one op each)."""
        return self.memory_read.to("pJ").magnitude / self.narrow_mac.to("pJ/flop").magnitude


class QuantizedDeploymentProfileScenario(BaseScenario):
    """A float model and its INT8 version on one accelerator, judged against acceptance limits.

    Accuracy and confidence-calibration error are stated inputs (percent); latency and
    throughput come from ``profile`` (an engine profile with ``tensorrt(precision, batch)``
    and ``large_batch_throughput(precision)``); energy per image assumes full board power.
    """
    workload: Workload
    hardware: HardwareNode
    profile: Any
    float_bytes: Quantity
    int_bytes: Quantity
    float_top1_pct: float
    int_top1_pct: float
    float_calibration_error_pct: float
    int_calibration_error_pct: float
    top1_limit_pct: float
    calibration_error_limit_pct: float

    @property
    def top1_drop(self) -> float:
        return self.float_top1_pct - self.int_top1_pct

    @property
    def float_latency(self) -> Quantity:
        return self.profile.tensorrt("fp32", batch=1)

    @property
    def int_latency(self) -> Quantity:
        return self.profile.tensorrt("int8", batch=1)

    @property
    def latency_speedup(self) -> float:
        return (self.float_latency / self.int_latency).to("").magnitude

    @property
    def float_throughput(self) -> float:
        return self.profile.large_batch_throughput("fp32")

    @property
    def int_throughput(self) -> float:
        return self.profile.large_batch_throughput("int8")

    @property
    def throughput_gain(self) -> float:
        return self.int_throughput / self.float_throughput

    @property
    def float_size(self) -> Quantity:
        return self.workload.size_in_bytes(self.float_bytes)

    @property
    def int_size(self) -> Quantity:
        return self.workload.size_in_bytes(self.int_bytes)

    @property
    def size_reduction(self) -> float:
        return (self.float_size / self.int_size).to("").magnitude

    @property
    def float_energy(self) -> Quantity:
        return (self.hardware.tdp / (self.float_throughput / Q_(1, "second"))).to("mJ")

    @property
    def int_energy(self) -> Quantity:
        return (self.hardware.tdp / (self.int_throughput / Q_(1, "second"))).to("mJ")

    @property
    def energy_improvement(self) -> float:
        return (self.float_energy / self.int_energy).to("").magnitude

    @property
    def passes_top1(self) -> bool:
        return self.int_top1_pct >= self.top1_limit_pct

    @property
    def passes_calibration(self) -> bool:
        return self.int_calibration_error_pct <= self.calibration_error_limit_pct


# --- Supervisory intervention: takeover budgets, blends, authentication (added 2026-10-20) ---

class TakeoverBudgetScenario(BaseScenario):
    """A machine at ``speed`` at a site asks a person to take over within ``takeover``.

    The site (an embodied site scenario) supplies the finished stopping
    budget's terms: pre-brake delay, credible deceleration, the resident
    stop's jerk bound, clear distance, localization bound, protective margin,
    tracking inset, the dispatch-to-admission allocation, the permission tick,
    and the chunk lease. ``configured_timeout`` is the protocol ceiling on the
    handover timer; ``manual_lease`` is the operator's lease once the channel
    is held (it replaces the chunk lease in the delay).
    """
    site: Any                                   # EmbodiedSiteScenario
    speed: Quantity
    takeover: Quantity                          # T_Sigma: human response + console transport
    configured_timeout: Optional[Quantity] = None
    manual_lease: Optional[Quantity] = None

    @property
    def result(self):
        from .solvers import TakeoverBudgetModel
        s = self.site
        return TakeoverBudgetModel().solve(
            self.speed, s.tau_delay, s.a_brake, s.d_clear, self.takeover,
            loc_margin=s.delta_loc, physical_margin=s.delta_margin, jerk=s.j_brake,
            tracking_bound=s.eps_track, dispatch_to_admission=s.t_dispatch_to_admission,
            configured_timeout=self.configured_timeout, tick=s.t_tick,
            manual_lease=self.manual_lease, chunk_lease=s.t_lease)


class TakeoverCeilingLadderScenario(BaseScenario):
    """One takeover budget per sourced human response, each plus the same console ``transport``."""
    base: TakeoverBudgetScenario
    transport: Quantity
    responses: Dict[str, Quantity]              # key -> human response, in ladder order

    @property
    def result(self):
        from .results import TakeoverLadderResult
        rows = {k: self.base.variant(takeover=(t + self.transport).to("ms")).result
                for k, t in self.responses.items()}
        return TakeoverLadderResult(transport=self.transport, human=dict(self.responses), rows=rows)


class BrakingBlendScenario(BaseScenario):
    """A bumpless transfer into braking on a site's base at ``speed``, and an arm joint's blend.

    The site supplies the braking deceleration, the resident stop's jerk
    bound, and the permission tick; the arm blend takes a torque change under
    a torque-rate ceiling.
    """
    site: Any                                   # EmbodiedSiteScenario
    speed: Quantity
    torque_change: Optional[Quantity] = None
    torque_rate_limit: Optional[Quantity] = None

    @property
    def result(self):
        from .solvers import BrakingBlendModel
        s = self.site
        return BrakingBlendModel().solve(self.speed, s.a_brake, s.j_brake, s.t_tick,
                                         torque_change=self.torque_change,
                                         torque_rate_limit=self.torque_rate_limit)


class CommandAuthenticationScenario(BaseScenario):
    """Remote commands authenticated on a machine's permission processor, priced against a takeover budget.

    ``budget`` gives the speed, the partial spare, and the fallback deadline
    before and after the dispatch-to-admission allocation; the processor clock
    turns published cycle counts into time.
    """
    budget: TakeoverBudgetScenario
    clock: Quantity
    verify_cycles: float
    mac_block_cycles: float
    mac_blocks: int
    freshness_window: Quantity

    @property
    def result(self):
        from .solvers import CommandAuthenticationModel
        b = self.budget.result
        return CommandAuthenticationModel().solve(
            self.clock, self.verify_cycles, self.mac_block_cycles, self.mac_blocks, b.speed,
            b.machine.spare, b.deadline, b.deadline_after_admission, self.freshness_window)


class DisengagementRecordScenario(BaseScenario):
    """One company's filed disengagement record: first and latest years and a year-over-year swing."""
    entries: Tuple[Dict[str, Any], ...]         # rows of Literature.CaliforniaDisengagements
    company: str
    first_year: int
    latest_year: int
    swing_from: int
    swing_to: int
    drivered_permit: str = "drivered_AVT"
    driverless_permit: str = "driverless_AVDT"
    filer_types: Tuple[str, ...] = ("annual", "first_report")

    @property
    def rows(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(r for r in self.entries
                     if r["company"] == self.company and r["filer_type"] in self.filer_types)

    def row(self, year: int, permit: Optional[str] = None) -> Dict[str, Any]:
        p = self.drivered_permit if permit is None else permit
        return next(r for r in self.rows if r["report_year"] == year and r["permit_type"] == p)

    @property
    def first(self):
        return self.row(self.first_year)

    @property
    def latest(self):
        return self.row(self.latest_year)

    @property
    def latest_driverless(self):
        return self.row(self.latest_year, self.driverless_permit)

    @property
    def rate_drop(self) -> float:
        """Miles per disengagement in the swing's first year over its second."""
        return self.row(self.swing_from)["miles_per_disengagement"] / self.row(self.swing_to)["miles_per_disengagement"]

    @property
    def miles_growth(self) -> float:
        """Autonomous miles in the swing's second year over its first."""
        return self.row(self.swing_to)["autonomous_miles"] / self.row(self.swing_from)["autonomous_miles"]
# --- Fault-injection oracles and campaign statistics (added 2026-10-20) ------

class OnsetTrialScenario(BaseScenario):
    """A stopping trace that begins braking at ``onset`` after the reference event of ``oracle``.

    ``oracle`` is a StoppingBudgetScenario whose delay is the onset bound (no
    site allowances), with a jerk bound and a tracking inset: its required
    distance is the travel bound, and its delay travel plus S-curve stop is
    the travel of an on-time trace.
    """
    oracle: StoppingBudgetScenario
    onset: Quantity
    lease: Optional[Quantity] = None               # the lease the trial lets expire
    probe_after_expiry: Optional[Quantity] = None  # a refused renewal replayed this long after expiry

    @property
    def probe_at(self) -> Optional[Quantity]:
        """When the replayed renewal arrives, after the last valid renewal."""
        if self.lease is None or self.probe_after_expiry is None:
            return None
        return (self.lease + self.probe_after_expiry).to("ms")

    @property
    def result(self):
        from .solvers import OnsetTrialModel
        o = self.oracle.result
        return OnsetTrialModel().solve(self.oracle.delay, self.onset, self.oracle.speed,
                                       (o.braking_travel + o.jerk_ramp_travel).to("mm"),
                                       o.required_distance, o.stop_time)


class RailHoldUpScenario(BaseScenario):
    """``site.machine``'s backed permission rail, its feed lost during the longest admitted stop.

    The longest stop starts at the speed ceiling of ``stop`` (a finished
    StoppingBudgetScenario); the flag path is one tick, one bus cycle, and the
    brake onset of ``site``; engagement and hold-up are the machine's.
    """
    site: Any                                    # EmbodiedSiteScenario
    stop: StoppingBudgetScenario

    @property
    def flag_path(self) -> Quantity:
        s = self.site
        return (s.t_tick + s.t_bus + s.t_brake_onset).to("ms")

    @property
    def result(self):
        from .solvers import RailHoldUpModel
        rm = self.site.machine
        return RailHoldUpModel().solve(self.flag_path, self.stop.result.speed_ceiling, self.stop.a_brake,
                                       self.stop.jerk, rm.spring_brake_engage_max, rm.permission_rail_holdup)


class ExponentialRareEventScenario(BaseScenario):
    """An exponential delay of ``mean`` that fails past the delay spare of ``budget`` (or ``threshold``)."""
    mean: Quantity
    elite_share: float
    relative_error: float
    confidence: float
    budget: Optional[StoppingBudgetScenario] = None   # threshold = its delay_spare
    threshold: Optional[Quantity] = None

    @property
    def failure_threshold(self) -> Quantity:
        if self.threshold is not None:
            return self.threshold
        return self.budget.result.delay_spare.to("ms")

    @property
    def result(self):
        from .solvers import ExponentialRareEventModel
        return ExponentialRareEventModel().solve(self.failure_threshold, self.mean, self.elite_share,
                                                 self.relative_error, self.confidence)


class DiagnosticCoverageScenario(BaseScenario):
    """Fault-rate ``strata`` (FIT) to be left at most ``target`` FIT unmitigated, shown at ``confidence``."""
    strata: Dict[str, float]
    target: float
    confidence: float

    @property
    def result(self):
        from .solvers import DiagnosticCoverageModel
        return DiagnosticCoverageModel().solve(self.strata, self.target, self.confidence)


class StepsToQualityScenario(BaseScenario):
    """A change that makes each step ``step_speedup`` times faster but needs ``steps_multiplier`` times the steps."""
    step_speedup: float
    steps_multiplier: float

    @property
    def time_ratio(self) -> float:
        """New time to quality over old (above 1: slower)."""
        return self.steps_multiplier / self.step_speedup


class MeanVersusTailScenario(BaseScenario):
    """A benchmark's mean latency against a production tail stated as a range."""
    mean: Quantity
    tail_low: Quantity
    tail_high: Quantity

    @property
    def gap_low(self) -> float:
        return (self.tail_low / self.mean).to("").magnitude

    @property
    def gap_high(self) -> float:
        return (self.tail_high / self.mean).to("").magnitude


class MLPerfTrainingResultScenario(BaseScenario):
    """One MLPerf Training submission's runs (``Literature.Series`` rows) and its published score.

    MLPerf's rule: drop the fastest and slowest of the runs, a run that missed
    the quality target counting as the slowest, average the rest, and multiply
    by the round's scaling factor. The per-accelerator rate divides the images
    trained (epochs times the training set) by the unscaled score and the
    accelerator count.
    """
    series_name: str
    system: str
    accelerators: int
    dataset: Any                      # Datasets entry with ``training_examples``
    global_batch: int
    scaling_factor: float

    @property
    def rows(self):
        from ..literature.registry import Literature
        return tuple(r for r in getattr(Literature.Series, self.series_name).entries if r["system"] == self.system)

    @property
    def run_minutes(self) -> List[float]:
        """Run times of the runs that reached the target."""
        return [r["minutes"] for r in self.rows if r["status"] == "success"]

    @property
    def score_minutes(self) -> float:
        failed = len(self.rows) - len(self.run_minutes)
        ranked = sorted(self.run_minutes) + [_math.inf] * failed
        middle = ranked[1:-1]
        return sum(middle) / len(middle)

    @property
    def published_score(self) -> Quantity:
        return Q_(self.score_minutes * self.scaling_factor, "minute")

    # Run statistics (added 2026-10-20, Lecture 12b).
    @property
    def all_run_minutes(self) -> List[float]:
        """Every run's time, including runs that missed the quality target."""
        return [r["minutes"] for r in self.rows]

    @property
    def failed_run_minutes(self) -> List[float]:
        return [r["minutes"] for r in self.rows if r["status"] != "success"]

    @property
    def top1(self) -> List[float]:
        """Top-1 at each run's last evaluation, percent."""
        return [r["top1"] for r in self.rows]

    @property
    def mean_minutes(self) -> float:
        """Mean of the converged runs (unscaled)."""
        return sum(self.run_minutes) / len(self.run_minutes)

    @property
    def median_minutes(self) -> float:
        import statistics as _stats
        return _stats.median(self.run_minutes)

    @property
    def sd_minutes(self) -> float:
        """Sample standard deviation of the converged runs."""
        import statistics as _stats
        return _stats.stdev(self.run_minutes)

    @property
    def cv(self) -> float:
        """Coefficient of variation of the converged runs (fraction)."""
        return self.sd_minutes / self.mean_minutes

    @property
    def epochs(self) -> float:
        return self.rows[0]["epochs"]

    @property
    def per_accelerator_rate(self) -> Quantity:
        images = self.epochs * self.dataset.training_examples.to("count").magnitude
        return Q_(images / (self.score_minutes * 60) / self.accelerators, "count/second")

    @property
    def per_accelerator_batch(self) -> float:
        return self.global_batch / self.accelerators


class StrongScalingPairScenario(BaseScenario):
    """The same fixed job on ``base_devices`` and on ``scaled_devices``: speedup, ideal time, and efficiency.

    Strong-scaling efficiency is base time over (device ratio times scaled time).
    """
    base_devices: int
    base_time: Quantity
    scaled_devices: int
    scaled_time: Quantity

    @property
    def device_ratio(self) -> float:
        return self.scaled_devices / self.base_devices

    @property
    def speedup(self) -> float:
        return (self.base_time / self.scaled_time).to("").magnitude

    @property
    def ideal_time(self) -> Quantity:
        return (self.base_time / self.device_ratio).to(self.base_time.units)

    @property
    def efficiency(self) -> float:
        return self.speedup / self.device_ratio
# --- Benchmarking: deciding from measurements (added 2026-10-20) --------------
# Run-to-run noise in repeated MLPerf runs, Welch intervals on a difference,
# run and sample counts, the paired accuracy comparison, run order on a
# warming device, a proxy metric against a latency budget, binned calibration,
# and a compressed model's acceptance gates.

def _z_two_sided(confidence: float) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(0.5 + confidence / 2)


class RunTimeComparisonScenario(BaseScenario):
    """Welch's interval on ``base`` minus ``other`` mean run time (MLPerfTrainingResultScenario runs).

    Positive differences mean ``other`` finished faster. Fractions are of
    ``base``'s mean converged run. ``include_failed`` times a run that missed
    the quality target as if it were a run. ``minimum_useful`` is a speedup
    (fraction of ``base``) set before the measurement; the verdict compares the
    interval with zero and with it.
    """
    base: MLPerfTrainingResultScenario
    other: MLPerfTrainingResultScenario
    confidence: float
    include_failed: bool = False
    minimum_useful: Optional[float] = None

    def _sample(self, s: MLPerfTrainingResultScenario) -> List[float]:
        return s.all_run_minutes if self.include_failed else s.run_minutes

    def _welch(self, sd_base: Optional[float] = None, sd_other: Optional[float] = None):
        from ..physics.statistics import calc_welch_interval
        import statistics as _stats
        a, b = self._sample(self.base), self._sample(self.other)
        return calc_welch_interval(_stats.mean(a), _stats.stdev(a) if sd_base is None else sd_base, len(a),
                                   _stats.mean(b), _stats.stdev(b) if sd_other is None else sd_other, len(b),
                                   self.confidence)

    @property
    def difference(self) -> Quantity:
        return Q_(self._welch()["difference"], "minute")

    @property
    def standard_error(self) -> Quantity:
        return Q_(self._welch()["se"], "minute")

    @property
    def dof(self) -> float:
        """Welch-Satterthwaite degrees of freedom."""
        return self._welch()["dof"]

    @property
    def t_critical(self) -> float:
        return self._welch()["t"]

    @property
    def low(self) -> Quantity:
        return Q_(self._welch()["low"], "minute")

    @property
    def high(self) -> Quantity:
        return Q_(self._welch()["high"], "minute")

    def fraction_of_base(self, x: Quantity) -> float:
        """A time as a fraction of ``base``'s mean converged run."""
        return (x / Q_(self.base.mean_minutes, "minute")).to("").magnitude

    def half_width_with_spread(self, sd: Quantity) -> Quantity:
        """The interval's half-width if both sides had standard deviation ``sd`` (same run counts)."""
        s = sd.to("minute").magnitude
        w = self._welch(s, s)
        return Q_(w["t"] * w["se"], "minute")

    @property
    def verdict(self) -> str:
        """'unresolved' (spans zero), 'real, below threshold', or 'real and useful'."""
        lo, hi = self.fraction_of_base(self.low), self.fraction_of_base(self.high)
        if lo <= 0 <= hi:
            return "unresolved"
        if self.minimum_useful is not None and lo >= self.minimum_useful:
            return "real and useful"
        return "real, below threshold"


class RunCountPlanningScenario(BaseScenario):
    """How many runs a mean, or a two-system comparison, needs at ``reference``'s run-to-run spread.

    The half-width of a mean of ``n`` runs is ``t_{n-1} s / sqrt(n)``; the runs
    per system to detect a difference ``d`` (a fraction of the mean) at
    ``power`` are ``2 (z_a + z_b)^2 s^2 / d^2``.
    """
    reference: MLPerfTrainingResultScenario
    confidence: float
    power: float
    run_counts: List[int]
    differences: List[float]            # fractions of the reference mean

    @property
    def z_alpha(self) -> float:
        return _z_two_sided(self.confidence)

    @property
    def z_beta(self) -> float:
        from statistics import NormalDist
        return NormalDist().inv_cdf(self.power)

    def half_width_fraction(self, n_runs: int) -> float:
        from ..physics.statistics import calc_mean_half_width
        r = self.reference
        return calc_mean_half_width(r.sd_minutes, n_runs, self.confidence) / r.mean_minutes

    def runs_per_system(self, difference: float) -> float:
        from ..physics.statistics import calc_samples_per_arm_for_difference
        r = self.reference
        return calc_samples_per_arm_for_difference(r.sd_minutes, difference * r.mean_minutes,
                                                   self.z_alpha, self.z_beta)


class ProportionSampleSizeScenario(BaseScenario):
    """Samples that pin a pass fraction ``rate`` to within ``half_width`` at ``confidence``: ``z^2 p (1 - p) / m^2``.

    ``published_minimum`` is a suite's rounded-up count for the same target.
    """
    rate: float
    half_width: float
    confidence: float
    published_minimum: Optional[int] = None

    @property
    def z(self) -> float:
        return _z_two_sided(self.confidence)

    @property
    def samples(self) -> float:
        """Unrounded sample count."""
        return self.z ** 2 * self.rate * (1 - self.rate) / self.half_width ** 2

    @property
    def samples_needed(self) -> int:
        from ..physics.statistics import calc_samples_for_half_width
        return calc_samples_for_half_width(self.rate, self.half_width, self.z)


class ProportionPrecisionScenario(BaseScenario):
    """The normal-approximation half-width of a pass fraction ``rate`` measured on ``n_samples``."""
    rate: float
    n_samples: int
    confidence: float

    @property
    def half_width(self) -> float:
        from ..physics.statistics import calc_proportion_half_width
        return calc_proportion_half_width(self.rate, self.n_samples, _z_two_sided(self.confidence))


class SameTestSetAccuracyComparisonScenario(BaseScenario):
    """Two models scored on the same ``n_samples`` test items: independent-set interval and McNemar's limit.

    The scores fix the net gap ``b - c`` (items only the baseline gets right
    minus items only the candidate gets right) but not the disagreements
    ``b + c``; McNemar's statistic (no continuity correction) exceeds the
    chi-square critical value ``z^2`` only when ``b + c <= floor(gap^2 / z^2)``.
    """
    n_samples: int
    baseline_accuracy: float
    candidate_accuracy: float
    confidence: float

    @property
    def z(self) -> float:
        return _z_two_sided(self.confidence)

    @property
    def baseline_errors(self) -> float:
        return self.n_samples * (1 - self.baseline_accuracy)

    @property
    def candidate_errors(self) -> float:
        return self.n_samples * (1 - self.candidate_accuracy)

    @property
    def baseline_error_sd(self) -> float:
        """Binomial standard deviation of the baseline's error count."""
        p = self.baseline_accuracy
        return _math.sqrt(self.n_samples * p * (1 - p))

    @property
    def accuracy_difference(self) -> float:
        return self.baseline_accuracy - self.candidate_accuracy

    @property
    def independent_interval(self):
        """(low, high) of the accuracy difference had the scores come from independent test sets."""
        from ..physics.statistics import calc_proportion_gap_interval
        return calc_proportion_gap_interval(self.baseline_accuracy, self.candidate_accuracy,
                                            self.n_samples, self.n_samples, self.z)

    @property
    def net_gap(self) -> int:
        """b - c: items the baseline alone gets right, net."""
        return round(self.n_samples * self.accuracy_difference)

    @property
    def chi2_critical(self) -> float:
        return self.z ** 2

    @property
    def max_discordant(self) -> int:
        from ..physics.statistics import calc_mcnemar_max_discordant
        return calc_mcnemar_max_discordant(self.net_gap, self.chi2_critical)

    @property
    def max_discordant_share(self) -> float:
        return self.max_discordant / self.n_samples


class WarmingDeviceRunOrderScenario(BaseScenario):
    """Two arms of the same code timed on a device whose latency drifts from ``start`` toward ``end``.

    Latency is ``start + (end - start) (1 - exp(-t / time_constant))``. Runs
    happen at ``first_time + i spacing``; blocked order runs every A then every
    B, interleaved alternates. The bias is B's mean over A's, minus one.
    """
    start: Quantity
    end: Quantity
    time_constant: Quantity
    first_time: Quantity
    spacing: Quantity
    runs_per_arm: int

    def latency(self, t: Quantity) -> Quantity:
        x = (t / self.time_constant).to("").magnitude
        return (self.start + (self.end - self.start) * (1 - _math.exp(-x))).to(self.start.units)

    @property
    def times(self) -> List[Quantity]:
        return [self.first_time + i * self.spacing for i in range(2 * self.runs_per_arm)]

    @property
    def blocked_order(self) -> List[str]:
        return ["A"] * self.runs_per_arm + ["B"] * self.runs_per_arm

    @property
    def interleaved_order(self) -> List[str]:
        return ["A", "B"] * self.runs_per_arm

    def bias(self, order: List[str]) -> float:
        def arm(label):
            v = [self.latency(t).magnitude for t, a in zip(self.times, order) if a == label]
            return sum(v) / len(v)
        return arm("B") / arm("A") - 1

    @property
    def blocked_bias(self) -> float:
        return self.bias(self.blocked_order)

    @property
    def interleaved_bias(self) -> float:
        return self.bias(self.interleaved_order)


class ProxyMetricLatencyScenario(BaseScenario):
    """A search-width change that buys an offline metric gain with latency, against a serving budget."""
    original_metric: float
    optimized_metric: float
    original_width: int
    optimized_width: int
    original_latency: Quantity
    optimized_latency: Quantity
    latency_budget: Quantity

    @property
    def metric_gain(self) -> float:
        return self.optimized_metric - self.original_metric

    @property
    def width_ratio(self) -> float:
        return self.optimized_width / self.original_width

    @property
    def slowdown(self) -> float:
        return (self.optimized_latency / self.original_latency).to("").magnitude

    @property
    def budget_ratio(self) -> float:
        """Optimized latency over the budget (above 1: the budget is broken)."""
        return (self.optimized_latency / self.latency_budget).to("").magnitude


class ReliabilityBinsScenario(BaseScenario):
    """Predictions grouped by confidence: ``bins`` of (count, mean confidence, accuracy); the ECE."""
    bins: List[Tuple[int, float, float]]

    @property
    def total(self) -> int:
        return sum(n for n, _, _ in self.bins)

    @property
    def gaps(self) -> List[float]:
        """Confidence minus accuracy, per bin (positive: overconfident)."""
        return [c - a for _, c, a in self.bins]

    @property
    def weights(self) -> List[float]:
        return [n / self.total for n, _, _ in self.bins]

    @property
    def ece(self) -> float:
        from ..physics.statistics import calc_expected_calibration_error
        return calc_expected_calibration_error(self.bins)


class QuantizedAcceptanceGatesScenario(BaseScenario):
    """A float model and its quantized copy against acceptance gates set before compression.

    Accuracies are percent; ECE is a fraction. Edge cases are ``edge_share``
    of inputs, so a ``test_images`` test set holds ``edge_share * test_images``
    of them, and the float edge-case accuracy is known only to the binomial
    half-width at ``confidence`` on that count.
    """
    workload: Workload
    float_top1: float
    int_top1: float
    float_top5: float
    int_top5: float
    float_ece: float
    int_ece: float
    float_edge: float
    int_edge: float
    accuracy_loss_limit: float          # percentage points
    edge_share: float
    edge_occlusion: float               # fraction of the object hidden
    edge_illuminance: Quantity
    edge_rotation_deg: float
    confidence_threshold: float         # the deployment acts above this confidence
    test_images: int
    confidence: float

    @property
    def top1_loss(self) -> float:
        return self.float_top1 - self.int_top1

    @property
    def top5_loss(self) -> float:
        return self.float_top5 - self.int_top5

    @property
    def edge_drop(self) -> float:
        return self.float_edge - self.int_edge

    @property
    def ece_ratio(self) -> float:
        return self.int_ece / self.float_ece

    @property
    def edge_images(self) -> int:
        return round(self.edge_share * self.test_images)

    @property
    def edge_half_width(self) -> float:
        """Binomial half-width of the float edge-case accuracy on ``edge_images``, in points."""
        from ..physics.statistics import calc_proportion_half_width
        return 100 * calc_proportion_half_width(self.float_edge / 100, self.edge_images,
                                                _z_two_sided(self.confidence))


# --- Model serving on one machine (added 2026-10-20) --------------------------
# A request's latency budget, the one-server queue that sizes a fleet, the
# batch a request waits for, a replica's startup, and the cost per request.
# The service-time model is ``BatchingKneeScenario`` (S(B) = t_0 + B t_req).

class BatchServingCostScenario(BaseScenario):
    """One accelerator rented at ``price``, serving batches back to back: cost per request by batch size."""
    service: BatchingKneeScenario
    price: Quantity                       # USD per accelerator-hour
    batch_sizes: List[int]

    def cost_per_million(self, batch_size) -> Quantity:
        return (self.price / self.service.throughput(batch_size) * 1e6).to("dollar")

    @property
    def cost_ratio(self) -> float:
        """Cost per request at the first batch size over the last."""
        first, last = self.batch_sizes[0], self.batch_sizes[-1]
        return (self.cost_per_million(first) / self.cost_per_million(last)).to("").magnitude

    @property
    def added_time(self) -> Quantity:
        """Service time of the last batch size beyond the first."""
        first, last = self.batch_sizes[0], self.batch_sizes[-1]
        return (self.service.service_time(last) - self.service.service_time(first)).to("ms")


class PrecomputeVersusOnDemandScenario(BaseScenario):
    """``items`` inputs answered one at a time on demand, or precomputed at a large-batch ``offline_rate``."""
    items: int
    on_demand_time: Quantity              # one request at batch one
    offline_rate: Quantity                # requests per second at large batch
    latency_budget: Quantity              # the on-demand service's deadline

    @property
    def on_demand_total(self) -> Quantity:
        return (self.items * self.on_demand_time).to("second")

    @property
    def offline_total(self) -> Quantity:
        return (self.items / self.offline_rate).to("second")

    @property
    def ratio(self) -> float:
        return (self.on_demand_total / self.offline_total).to("").magnitude


class MM1CapacityPlanScenario(BaseScenario):
    """Accelerators for ``peak_rate`` when each is an M/M/1 server whose tail must meet ``tail_budget``.

    The tail is ``tail_factor`` times the mean time in system S/(1 - rho), so
    the highest safe utilization is 1 - k S / T. The plan divides the peak by
    that utilization, rounds up to whole accelerators, then adds ``headroom``.
    Optional: ``batched_rate`` re-plans with batched service at the same
    utilization; ``accelerator_price`` prices a plan per million requests.
    """
    service_time: Quantity
    tail_factor: float                    # tail as a multiple of the mean
    tail_quantile: float = 0.99           # the tail the factor stands for
    tail_budget: Quantity                 # queue-plus-service tail target
    peak_rate: Quantity
    headroom: float                       # multiplier on the rounded-up count
    utilization_points: List[float] = []  # loads at which to report time in system
    example_utilization: Optional[float] = None
    batched_rate: Optional[Quantity] = None
    accelerator_price: Optional[Quantity] = None   # USD per accelerator-hour

    @property
    def service_rate(self) -> Quantity:
        return (1 / self.service_time).to("1/second")

    def time_in_system(self, utilization) -> Quantity:
        from ..physics.serving import calc_mm1_time_in_system
        return calc_mm1_time_in_system(self.service_time, utilization).to("ms")

    def tail_time(self, utilization) -> Quantity:
        return (self.tail_factor * self.time_in_system(utilization)).to("ms")

    @property
    def exact_tail_factor(self) -> float:
        """The factor ``tail_factor`` rounds: ln(1/(1 - q)) for an exponential time in system."""
        from ..physics.serving import calc_exponential_quantile_factor
        return calc_exponential_quantile_factor(self.tail_quantile)

    @property
    def max_utilization(self) -> float:
        from ..physics.serving import calc_mm1_max_utilization
        return calc_mm1_max_utilization(self.service_time, self.tail_budget, self.tail_factor)

    @property
    def required_rate(self) -> Quantity:
        return (self.peak_rate / self.max_utilization).to("1/second")

    @property
    def accelerators_exact(self) -> float:
        return (self.required_rate / self.service_rate).to("").magnitude

    @property
    def accelerators(self) -> int:
        return _math.ceil(self.accelerators_exact)

    @property
    def accelerators_planned(self) -> int:
        return _math.ceil(self.accelerators * self.headroom)

    @property
    def utilization_after_failure(self) -> float:
        """Utilization of the planned fleet with one accelerator lost."""
        return (self.peak_rate / (self.service_rate * (self.accelerators_planned - 1))).to("").magnitude

    @property
    def saturated_accelerators_exact(self) -> float:
        """Accelerators if each ran 100 percent busy (no queueing headroom)."""
        return (self.peak_rate / self.service_rate).to("").magnitude

    @property
    def batched_accelerators(self) -> int:
        return _math.ceil((self.peak_rate / (self.max_utilization * self.batched_rate)).to("").magnitude)

    @property
    def batched_accelerators_planned(self) -> int:
        return _math.ceil(self.batched_accelerators * self.headroom)

    def cost_per_million(self, accelerators) -> Quantity:
        """Fleet cost per million requests served at ``peak_rate``."""
        return (accelerators * self.accelerator_price / self.peak_rate * 1e6).to("dollar")


class MM1ServiceSpeedupScenario(BaseScenario):
    """The same traffic on an M/M/1 server before and after its service time falls."""
    slow_service: Quantity
    fast_service: Quantity
    utilization: float                    # load with the slow service

    @property
    def arrival_rate(self) -> Quantity:
        return (self.utilization / self.slow_service).to("1/second")

    @property
    def fast_utilization(self) -> float:
        return self.utilization * (self.fast_service / self.slow_service).to("").magnitude

    @property
    def slow_time(self) -> Quantity:
        from ..physics.serving import calc_mm1_time_in_system
        return calc_mm1_time_in_system(self.slow_service, self.utilization).to("ms")

    @property
    def fast_time(self) -> Quantity:
        from ..physics.serving import calc_mm1_time_in_system
        return calc_mm1_time_in_system(self.fast_service, self.fast_utilization).to("ms")

    @property
    def slow_wait(self) -> Quantity:
        return (self.slow_time - self.slow_service).to("ms")

    @property
    def fast_wait(self) -> Quantity:
        return (self.fast_time - self.fast_service).to("ms")

    @property
    def service_gain(self) -> float:
        return (self.slow_service / self.fast_service).to("").magnitude

    @property
    def time_gain(self) -> float:
        return (self.slow_time / self.fast_time).to("").magnitude

    @property
    def wait_gain(self) -> float:
        return (self.slow_wait / self.fast_wait).to("").magnitude


class BatchWindowScenario(BaseScenario):
    """Requests arriving as a Poisson stream into a batching window of length ``window``.

    The window collects lambda W arrivals on average; a request waits half the
    window on average; the batch a request finds itself in is size-biased,
    1 + lambda W on average, and its ``tail_quantile`` is 1 plus that
    quantile of the Poisson count.
    """
    window: Quantity
    arrival_rates: List[Quantity] = []
    latency_budget: Optional[Quantity] = None
    tail_quantile: float = 0.99

    def mean_arrivals(self, arrival_rate) -> float:
        return (arrival_rate * self.window).to("").magnitude

    def arrival_spread(self, arrival_rate) -> float:
        """Standard deviation of the count, sqrt(lambda W)."""
        return _math.sqrt(self.mean_arrivals(arrival_rate))

    def empty_share(self, arrival_rate) -> float:
        return _math.exp(-self.mean_arrivals(arrival_rate))

    def count_at_least(self, arrival_rate, at_least: int) -> float:
        from ..physics.statistics import calc_poisson_tail
        return calc_poisson_tail(self.mean_arrivals(arrival_rate), at_least)

    @property
    def mean_wait(self) -> Quantity:
        return (self.window / 2).to("ms")

    @property
    def budget_share(self) -> float:
        return (self.mean_wait / self.latency_budget).to("").magnitude

    def seen_batch_mean(self, arrival_rate) -> float:
        return 1 + self.mean_arrivals(arrival_rate)

    def seen_batch_tail(self, arrival_rate) -> int:
        """1 plus the ``tail_quantile`` of the Poisson count: the smallest s with P(N >= s) below 1 - q."""
        from ..physics.statistics import calc_poisson_reserve
        return calc_poisson_reserve(self.mean_arrivals(arrival_rate), 1 - self.tail_quantile)


class BatchDispatchScenario(BaseScenario):
    """A dual-trigger batch: it may leave at ``timeout`` after its oldest arrival, then waits for a worker."""
    arrivals: List[Quantity]              # offsets from the oldest request
    timeout: Quantity
    worker_free: Quantity                 # when the worker finishes its current batch
    response_path: Quantity               # time after the batch finishes
    deadline: Quantity
    service: BatchingKneeScenario

    @property
    def formation_end(self) -> Quantity:
        return (self.arrivals[0] + self.timeout).to("ms")

    @property
    def dispatch(self) -> Quantity:
        return max(self.formation_end, self.worker_free.to("ms"))

    @property
    def backlog_wait(self) -> Quantity:
        return (self.dispatch - self.formation_end).to("ms")

    @property
    def batch_time(self) -> Quantity:
        return self.service.service_time(len(self.arrivals))

    @property
    def oldest_latency(self) -> Quantity:
        return (self.dispatch + self.batch_time + self.response_path - self.arrivals[0]).to("ms")

    @property
    def slack(self) -> Quantity:
        return (self.deadline - self.oldest_latency).to("ms")

    @property
    def timer_only_latency(self) -> Quantity:
        """The answer that ignores the busy worker and the response path."""
        return (self.timeout + self.batch_time).to("ms")


class BatchCapChoiceScenario(BaseScenario):
    """Two batch caps at one arrival rate, batches waiting until full: throughput against formation wait."""
    service: BatchingKneeScenario
    arrival_rate: Quantity
    small_cap: int
    large_cap: int

    def formation_wait(self, cap) -> Quantity:
        from ..physics.serving import calc_batch_formation_wait
        return calc_batch_formation_wait(cap, self.arrival_rate).to("ms")

    @property
    def throughput_gain(self) -> float:
        return (self.service.throughput(self.large_cap) / self.service.throughput(self.small_cap)).to("").magnitude - 1

    @property
    def capacity_over_load(self) -> float:
        return (self.service.throughput(self.small_cap) / self.arrival_rate).to("").magnitude


class ReplicaStartupScenario(BaseScenario):
    """A serving replica's startup phases; ``warm_path`` and ``cold_path`` name the phases each start pays."""
    phases: Dict[str, Quantity]
    ranges: Dict[str, List[Quantity]] = {}   # stated low and high for phases given as a range
    warm_path: List[str]
    cold_path: List[str]

    @property
    def warm_total(self) -> Quantity:
        return sum((self.phases[p] for p in self.warm_path), Q_(0, "second")).to("second")

    @property
    def cold_total(self) -> Quantity:
        return sum((self.phases[p] for p in self.cold_path), Q_(0, "second")).to("second")


class TrafficSurgeScenario(BaseScenario):
    """A service sized for ``normal_rate`` that receives ``surge_rate``; latency in each regime."""
    normal_rate: Quantity
    surge_rate: Quantity
    normal_latency: Quantity
    surge_latency: Quantity               # stated latency once the queue has grown

    @property
    def surge_factor(self) -> float:
        return (self.surge_rate / self.normal_rate).to("").magnitude


class BatchEnergyPerImageScenario(BaseScenario):
    """Bytes, arithmetic intensity, and energy per image when a batch shares one weight read.

    Each image moves the weights divided by the batch plus its own
    activations ``activation_passes`` times (written once, read once), at
    ``byte_energy``; its arithmetic costs ``flop_energy`` per FLOP.
    """
    workload: Workload
    weight_bytes_per_param: Quantity
    activation_bytes: Quantity            # one image's activation footprint
    activation_passes: int
    byte_energy: Quantity
    flop_energy: Quantity

    @property
    def weights(self) -> Quantity:
        return self.workload.size_in_bytes(self.weight_bytes_per_param)

    def weight_per_image(self, batch_size) -> Quantity:
        return self.weights / batch_size

    def bytes_per_image(self, batch_size) -> Quantity:
        return self.weight_per_image(batch_size) + self.activation_passes * self.activation_bytes

    def intensity(self, batch_size) -> Quantity:
        return (self.workload.inference_flops / self.bytes_per_image(batch_size)).to("flop/byte")

    def movement_energy(self, batch_size) -> Quantity:
        return (self.bytes_per_image(batch_size) * self.byte_energy).to("mJ")

    @property
    def compute_energy(self) -> Quantity:
        return (self.workload.inference_flops * self.flop_energy).to("mJ")

    def energy(self, batch_size) -> Quantity:
        return self.movement_energy(batch_size) + self.compute_energy

    def movement_share(self, batch_size) -> float:
        return (self.movement_energy(batch_size) / self.energy(batch_size)).to("").magnitude


class ServingInstanceCostScenario(BaseScenario):
    """Instances with an hourly price and a saturated request rate: cost per request, and when the
    ``small`` instance sized to the load undercuts one ``large`` instance."""
    prices: Dict[str, Quantity]           # USD per instance-hour
    rates: Dict[str, Quantity]            # requests per second, fully busy
    small: str
    large: str

    def cost_per_million(self, name) -> Quantity:
        return (self.prices[name] / self.rates[name] * 1e6).to("dollar")

    def requests_per_hour(self, name) -> float:
        return (self.rates[name] * Q_(1, "hour")).to("").magnitude

    @property
    def price_ratio(self) -> float:
        return (self.prices[self.large] / self.prices[self.small]).to("").magnitude

    @property
    def breakeven_rate(self) -> Quantity:
        """Load at which the large instance's price buys the same capacity of small instances."""
        return (self.price_ratio * self.rates[self.small]).to("1/second")

    @property
    def small_within_large_price(self) -> int:
        """Most small instances that cost less than one large instance."""
        return _math.ceil(self.price_ratio) - 1

    @property
    def small_fleet_rate(self) -> Quantity:
        return (self.small_within_large_price * self.rates[self.small]).to("1/second")
# --- Reinforcement learning from verifiable rewards (added 2026-10-20) ---
# One update's accepted rollouts under an imperfect verifier, a fixed baseline,
# a shaped reward, forked continuations and their bill, group-relative
# coefficients, length normalization, reward contrast per rollout step,
# repeated verification, a ratio clip, a sandbox slot's cycle, a group
# barrier, and a verifier pool. Solvers in engine/solvers/reinforcement.py.


class RewardCompositionScenario(BaseScenario):
    """One update's rollouts scored by a verifier that accepts some failures."""
    tasks: int
    group: int                       # rollouts per task
    pass_rate: float                 # true pass probability of one rollout
    false_accept_rate: float         # P(accept | failure)
    valid_accept_rate: float         # P(accept | pass)

    @property
    def result(self):
        from .solvers import RewardCompositionModel
        return RewardCompositionModel().solve(self.tasks, self.group, self.pass_rate,
                                              self.false_accept_rate, self.valid_accept_rate)


class CenteredRewardScenario(BaseScenario):
    """Binary terminal rewards centered on a fixed baseline."""
    baseline: float
    passing_reward: float
    failing_reward: float

    @property
    def result(self):
        from .solvers import CenteredRewardModel
        return CenteredRewardModel().solve(self.baseline, self.passing_reward, self.failing_reward)


class RewardShapingScenario(BaseScenario):
    """A shaped score R = c + alpha f - lambda len, with one pass and one failure to compare."""
    format_weight: float             # alpha
    length_weight: float             # lambda
    length_cap: int                  # tokens; len = tokens / cap
    pass_format: float
    pass_tokens: int
    failure_format: float
    failure_tokens: int

    @property
    def result(self):
        from .solvers import RewardShapingModel
        return RewardShapingModel().solve(self.format_weight, self.length_weight, self.length_cap,
                                          self.pass_format, self.pass_tokens, self.failure_format,
                                          self.failure_tokens)


class ForkValueScenario(BaseScenario):
    """Binary continuations forked before and after one turn, with their pass counts."""
    continuations: int               # per fork
    passes_before: int
    passes_after: int

    @property
    def result(self):
        from .solvers import ForkValueModel
        return ForkValueModel().solve(self.continuations, self.passes_before, self.passes_after)


class ContinuationForkCostScenario(BaseScenario):
    """A trajectory whose value is sampled by forking continuations at chosen turns."""
    turns: int
    tokens_per_turn: int
    branches: int                    # continuations per fork
    boundaries: Optional[List[int]] = None   # completed turns before each fork; None forks at every one

    @property
    def result(self):
        from .solvers import ContinuationForkCostModel
        return ContinuationForkCostModel().solve(self.turns, self.tokens_per_turn, self.branches,
                                                 self.boundaries)


class GroupCoefficientsScenario(BaseScenario):
    """The rewards of one group of rollouts and the coefficient convention applied to them."""
    rewards: List[float]
    stabilizer: float                # epsilon added to the dispersion
    bessel: bool = False             # divide the variance by G - 1 instead of G

    @property
    def result(self):
        from .solvers import GroupCoefficientsModel
        return GroupCoefficientsModel().solve(self.rewards, self.stabilizer, self.bessel)


class LengthNormalizationScenario(BaseScenario):
    """Two rollouts with one group coefficient and different lengths, averaged per rollout."""
    short_tokens: int
    long_tokens: int

    @property
    def result(self):
        from .solvers import LengthNormalizationModel
        return LengthNormalizationModel().solve(self.short_tokens, self.long_tokens)


class RolloutStepBudgetScenario(BaseScenario):
    """One update's tasks, group size, rollout length, and the policy's pass rate on the mix."""
    tasks: int
    group: int
    turns: int
    tokens_per_turn: int
    pass_rate: float

    @property
    def result(self):
        from .solvers import RolloutStepBudgetModel
        return RolloutStepBudgetModel().solve(self.tasks, self.group, self.turns, self.tokens_per_turn,
                                              self.pass_rate)


class RepeatedVerificationScenario(BaseScenario):
    """A timing-sensitive check repeated on one candidate, with an all-pass rule."""
    incorrect_pass: float            # P(one trial passes | incorrect candidate)
    correct_pass: float              # P(one trial passes | correct candidate)
    trials: int
    group: int                       # candidates graded once each in one group

    @property
    def result(self):
        from .solvers import RepeatedVerificationModel
        return RepeatedVerificationModel().solve(self.incorrect_pass, self.correct_pass, self.trials,
                                                 self.group)


class RatioClipScenario(BaseScenario):
    """One token's old probability under a symmetric ratio clip."""
    epsilon: float
    probability: float

    @property
    def result(self):
        from .solvers import RatioClipModel
        return RatioClipModel().solve(self.epsilon, self.probability)


class RolloutSlotCycleScenario(BaseScenario):
    """Sandbox slots running one episode at a time, with a cold and a snapshot reset."""
    slots: int
    decisions: int                   # model calls per episode
    model_time: Quantity             # per decision
    tool_time: Quantity              # per decision
    assessment_time: Quantity        # per episode
    cold_reset: Quantity
    snapshot_reset: Quantity

    @property
    def result(self):
        from .solvers import RolloutSlotCycleModel
        return RolloutSlotCycleModel().solve(self.slots, self.decisions, self.model_time, self.tool_time,
                                             self.assessment_time, self.cold_reset, self.snapshot_reset)


class GroupBarrierScenario(BaseScenario):
    """The rollout latencies of one synchronous group."""
    latencies: List[Quantity]

    @property
    def result(self):
        from .solvers import GroupBarrierModel
        return GroupBarrierModel().solve(self.latencies)


class VerifierPoolScenario(BaseScenario):
    """Completed trajectories arriving in batches, each verified by repeated trials."""
    trajectories: int                # per interval
    interval: Quantity
    trials: int                      # per trajectory; all execute in this sizing case
    trial_time: Quantity
    target_utilization: float

    @property
    def result(self):
        from .solvers import VerifierPoolModel
        return VerifierPoolModel().solve(self.trajectories, self.interval, self.trials, self.trial_time,
                                         self.target_utilization)
# --- Release arguments: how confidence combines across premises (added 2026-10-20) ---

class ArgumentConfidenceScenario(BaseScenario):
    """A stipulated safety argument: ``premises`` required premises at ``premise_confidence`` each,
    and a claim carried by ``legs`` independent legs at ``leg_confidence`` each.

    Required premises combine as a product (``calc_conjunctive_confidence``);
    independent sufficient legs fail together with the product of their
    failure probabilities (``calc_multilegged_confidence``).
    """
    premise_confidence: float
    premises: int
    leg_confidence: float
    legs: int = 2

    @property
    def conjunction(self) -> float:
        from ..physics import calc_conjunctive_confidence
        return calc_conjunctive_confidence([self.premise_confidence] * self.premises)

    @property
    def multilegged(self) -> float:
        from ..physics import calc_multilegged_confidence
        return calc_multilegged_confidence([self.leg_confidence] * self.legs)
# --- Coordinating several agents: partitions, schedules, merges, votes, and allowances (added 2026-10-20) ---


class PartitionScreenScenario(BaseScenario):
    """A fixed partition screened over worker counts: (1 - p) + p / N + c (N - 1) of the serial time.

    The best count is the integer that minimizes elapsed time under the ceiling
    ``max_workers``; no count beats 1 / (1 - p), and the continuous optimum sits
    at sqrt(p / c).
    """
    parallel_fraction: float
    overhead_per_extra_worker: float
    max_workers: int

    @property
    def elapsed(self) -> Dict[int, float]:
        """Elapsed time over the serial reference, for each count from one to the ceiling."""
        from ..physics.coordination import calc_partition_elapsed
        return {n: calc_partition_elapsed(n, self.parallel_fraction, self.overhead_per_extra_worker)
                for n in range(1, self.max_workers + 1)}

    @property
    def best_workers(self) -> int:
        t = self.elapsed
        return min(t, key=t.get)

    @property
    def best_time(self) -> float:
        return self.elapsed[self.best_workers]

    @property
    def best_speedup(self) -> float:
        return 1 / self.best_time

    @property
    def speedup_cap(self) -> float:
        """Amdahl's ceiling with free coordination: 1 / (1 - p)."""
        return 1 / (1 - self.parallel_fraction)

    @property
    def continuous_best(self) -> float:
        """sqrt(p / c), where the derivative of the elapsed time vanishes."""
        return _math.sqrt(self.parallel_fraction / self.overhead_per_extra_worker)


class ScalabilityLawScenario(BaseScenario):
    """Relative capacity N / (1 + sigma (N - 1) + kappa N (N - 1)) over integer worker counts."""
    contention: float                         # sigma
    coherency: float                          # kappa
    search_limit: int

    @property
    def capacity(self) -> Dict[int, float]:
        from ..physics.performance import calc_universal_scalability_law
        return {n: calc_universal_scalability_law(n, self.contention, self.coherency)
                for n in range(1, self.search_limit + 1)}

    @property
    def best_workers(self) -> int:
        c = self.capacity
        return max(c, key=c.get)

    @property
    def peak_capacity(self) -> float:
        return self.capacity[self.best_workers]

    @property
    def continuous_peak(self) -> float:
        """N* = sqrt((1 - sigma) / kappa)."""
        from ..physics.performance import calc_usl_optimal_concurrency
        return calc_usl_optimal_concurrency(self.contention, self.coherency)

    @property
    def retrograde_threshold(self) -> float:
        """(1 - sigma) / kappa: past it, capacity falls below one worker's."""
        return (1 - self.contention) / self.coherency

    @property
    def first_slower(self) -> Optional[int]:
        """The smallest count above one whose capacity is below one worker's."""
        return next((n for n, c in self.capacity.items() if n > 1 and c < 1), None)


class DelegationTokenBreakEvenScenario(BaseScenario):
    """Delegation at a token multiple m per attempt: cost per accepted task scales by m p_single / p_multi.

    With cost proportional to tokens, delegation breaks even only if it multiplies
    acceptance by m, which is possible only while single-agent acceptance is at
    most 1 / m.
    """
    agent_tokens_vs_chat: float               # single agent's tokens over a chat answer's
    multi_agent_tokens_vs_chat: float         # delegated system's tokens over a chat answer's
    reported_gain: float                      # a reported relative score gain, read as acceptance
    single_acceptance: Optional[float] = None  # a stated single-agent acceptance to test

    @property
    def token_multiple(self) -> float:
        return self.multi_agent_tokens_vs_chat / self.agent_tokens_vs_chat

    @property
    def max_single_acceptance(self) -> float:
        """Largest single-agent acceptance at which break-even is possible: 1 / m."""
        return 1 / self.token_multiple

    @property
    def acceptance_multiplier(self) -> float:
        return 1 + self.reported_gain

    @property
    def cost_ratio_if_gain_is_acceptance(self) -> float:
        """Delegated over single cost per accepted task if the gain multiplied acceptance."""
        return self.token_multiple / self.acceptance_multiplier

    @property
    def required_multi_acceptance(self) -> Optional[float]:
        """Delegated acceptance that breaks even against ``single_acceptance``: m p_single."""
        return None if self.single_acceptance is None else self.token_multiple * self.single_acceptance


class DependencyScheduleScenario(BaseScenario):
    """A dependency graph on interchangeable slots: aggregate work, the critical path, and both bounds.

    The schedule starts each node when its last predecessor finishes. No schedule
    on ``slots`` slots ends before ``max(W / slots, L)``.
    """
    durations: Dict[int, Quantity]
    predecessors: Dict[int, List[int]]
    slots: int

    @property
    def schedule(self):
        from ..physics.coordination import calc_earliest_schedule
        return calc_earliest_schedule(self.durations, self.predecessors)

    @property
    def start(self) -> Dict[int, Quantity]:
        return self.schedule[0]

    @property
    def finish(self) -> Dict[int, Quantity]:
        return self.schedule[1]

    @property
    def work(self) -> Quantity:
        return sum(self.durations.values(), 0 * next(iter(self.durations.values())))

    @property
    def makespan(self) -> Quantity:
        """The critical path: the latest earliest finish."""
        return max(self.finish.values())

    @property
    def work_bound(self) -> Quantity:
        return self.work / self.slots

    @property
    def speedup(self) -> float:
        """Serial time over the makespan."""
        return (self.work / self.makespan).to("").magnitude

    @property
    def peak_slots(self) -> int:
        """Most nodes running at once in the earliest schedule."""
        s, f = self.start, self.finish
        return max(sum(1 for m in self.durations if s[m] <= s[n] < f[m]) for n in self.durations)

    @property
    def free_float(self) -> Dict[int, Quantity]:
        """For each node with successors, how long it can slip before delaying the earliest one."""
        s, f = self.start, self.finish
        out = {}
        for n in self.durations:
            succ = [m for m, preds in self.predecessors.items() if n in preds]
            if succ:
                out[n] = min(s[m] for m in succ) - f[n]
        return out


class ReferencePassingScenario(BaseScenario):
    """Children handed a pasted repository slice, or a reference and the files they then read."""
    workers: int
    slice_tokens: int                         # pasted into each child
    reference_tokens: int                     # the envelope's reference instead
    read_tokens: int                          # files each child then reads
    input_price: Quantity                     # USD per million input tokens

    @property
    def pasted_tokens(self) -> int:
        return self.workers * self.slice_tokens

    @property
    def reference_total_tokens(self) -> int:
        return self.workers * (self.reference_tokens + self.read_tokens)

    @property
    def full_read_tokens(self) -> int:
        """A reference after which each child reads the whole slice."""
        return self.workers * (self.reference_tokens + self.slice_tokens)

    @property
    def pasted_cost(self) -> Quantity:
        return (Q_(self.pasted_tokens, "token") * self.input_price).to("dollar")

    @property
    def reference_cost(self) -> Quantity:
        return (Q_(self.reference_total_tokens, "token") * self.input_price).to("dollar")

    @property
    def reduction(self) -> float:
        return self.pasted_tokens / self.reference_total_tokens

    @property
    def read_fraction(self) -> float:
        return self.read_tokens / self.slice_tokens

    @property
    def break_even_read_tokens(self) -> int:
        """Reads per child at which the reference saves nothing."""
        return self.slice_tokens - self.reference_tokens


class SharedWriteScenario(BaseScenario):
    """Workers each writing ``writes_per_worker`` of ``files`` files, uniformly and independently."""
    files: int
    writes_per_worker: int
    max_fleet: int
    small_fleet: int                          # fleet sizes the comparison quotes
    large_fleet: int

    @property
    def disjoint(self) -> Dict[int, float]:
        """Exact probability that no two write sets overlap, for each fleet size."""
        from ..physics.coordination import calc_disjoint_writes
        return {n: calc_disjoint_writes(n, self.files, self.writes_per_worker) for n in range(1, self.max_fleet + 1)}

    @property
    def disjoint_approx(self) -> Dict[int, float]:
        from ..physics.coordination import calc_disjoint_writes_approx
        return {n: calc_disjoint_writes_approx(n, self.files, self.writes_per_worker)
                for n in range(1, self.max_fleet + 1)}

    @property
    def collision(self) -> Dict[int, float]:
        """Probability of at least one shared write."""
        return {n: 1 - p for n, p in self.disjoint.items()}

    @property
    def even_odds_fleet(self) -> Optional[int]:
        """The smallest fleet for which a shared write is more likely than not."""
        return next((n for n, p in self.disjoint.items() if p < 0.5), None)


class CorrelatedReviewersScenario(BaseScenario):
    """Reviewers who each falsely approve with the same marginal rate, independent or correlated.

    The correlated model draws one propensity per candidate from a beta law with
    the stated mean and pairwise correlation; reviewers err independently given
    it (a beta-binomial). The beta mass above one half floors the majority
    error of any panel size.
    """
    reviewers: int                            # an odd panel
    false_approval: float                     # marginal false-approval rate
    correlation: float                        # pairwise correlation of the beta mixture
    large_ensemble: int                       # a larger odd panel to compare

    @property
    def shape(self):
        from ..physics.coordination import calc_beta_mixture_shape
        return calc_beta_mixture_shape(self.false_approval, self.correlation)

    def independent_majority(self, panel: int) -> float:
        from ..physics.coordination import calc_binomial_tail
        return calc_binomial_tail(panel, self.false_approval, panel // 2 + 1)

    def correlated_majority(self, panel: int) -> float:
        from ..physics.coordination import calc_beta_binomial_pmf
        return sum(calc_beta_binomial_pmf(panel, self.false_approval, self.correlation)[panel // 2 + 1:])

    @property
    def independent_unanimous(self) -> float:
        return self.false_approval ** self.reviewers

    @property
    def correlated_unanimous(self) -> float:
        from ..physics.coordination import calc_beta_binomial_pmf
        return calc_beta_binomial_pmf(self.reviewers, self.false_approval, self.correlation)[-1]

    @property
    def unanimous_ratio(self) -> float:
        return self.correlated_unanimous / self.independent_unanimous

    @property
    def floor(self) -> float:
        """P(propensity > 1/2): the limit of correlated majority error as the panel grows."""
        from ..physics.coordination import calc_beta_tail
        return calc_beta_tail(self.false_approval, self.correlation, 0.5)

    @property
    def panels(self) -> List[int]:
        """Odd panel sizes from one to the large ensemble."""
        return list(range(1, self.large_ensemble + 1, 2))


class FanoutDeadlineScenario(BaseScenario):
    """A fan-out whose workers each finish by the deadline with one probability, independently.

    An all-results barrier needs every worker (horizon compounding across workers);
    a quorum contract accepts any ``required`` of ``workers``.
    """
    workers: int
    required: int
    on_time_probability: float
    deadline: Quantity
    max_fleet: int                            # fleet sizes to tabulate

    @property
    def all_on_time(self) -> float:
        from ..physics.agents import calc_unchecked_success
        return calc_unchecked_success(self.on_time_probability, self.workers)

    @property
    def quorum_on_time(self) -> float:
        from ..physics.coordination import calc_binomial_tail
        return calc_binomial_tail(self.workers, self.on_time_probability, self.required)

    @property
    def even_odds_workers(self) -> int:
        """The smallest fleet whose all-results barrier falls below even odds."""
        from ..physics.agents import calc_first_horizon_below
        return calc_first_horizon_below(self.on_time_probability, 0.5)

    @property
    def all_on_time_by_fleet(self) -> Dict[int, float]:
        from ..physics.agents import calc_unchecked_success
        return {n: calc_unchecked_success(self.on_time_probability, n) for n in range(1, self.max_fleet + 1)}


class CreditWindowScenario(BaseScenario):
    """A producer feeding a slower model consumer, unbounded or behind a credit window.

    The consumer serves one item per ``tokens_per_item`` at its decode rate. With
    no bound the queue grows at the rate difference; a saturated window of W
    credits holds each item about W / mu by Little's law.
    """
    producer_rate: Quantity                   # items per second
    decode_rate: Quantity                     # consumer tokens per second
    tokens_per_item: Quantity
    horizon: Quantity
    credits: int
    residence_target: Quantity                # a mean residence to size a window for

    @property
    def service_rate(self) -> Quantity:
        return (self.decode_rate / self.tokens_per_item).to("1/s")

    @property
    def growth_rate(self) -> Quantity:
        return (self.producer_rate - self.service_rate).to("1/s")

    @property
    def backlog(self) -> float:
        """Items waiting after the horizon."""
        return (self.growth_rate * self.horizon).to("").magnitude

    @property
    def backlog_wait(self) -> Quantity:
        """Wait of the newest item behind that backlog."""
        return (self.backlog / self.service_rate).to("s")

    @property
    def window_residence(self) -> Quantity:
        return (self.credits / self.service_rate).to("s")

    @property
    def window_for_target(self) -> int:
        """Largest window whose saturated mean residence stays within the target."""
        return _math.floor((self.residence_target * self.service_rate).to("").magnitude + 1e-9)


class SplitAllowanceScenario(BaseScenario):
    """Children asking for allowances out of a parent's remaining balance."""
    remaining: Quantity
    finalization_reserve: Quantity
    own_reserved: Quantity                    # the parent's own outstanding work
    requests: List[Quantity]                  # in dispatch order
    nested_request: Optional[Quantity] = None  # a grandchild's allowance, carved from one child's

    @property
    def available(self) -> Quantity:
        return (self.remaining - self.finalization_reserve - self.own_reserved).to("dollar")

    @property
    def copied_liability(self) -> Quantity:
        """What copying each child's allowance from the balance would promise."""
        return sum(self.requests, 0 * self.remaining).to("dollar")

    @property
    def admission(self):
        """Atomic reservation in dispatch order: (admitted flags, balance left)."""
        left, admitted = self.available, []
        for ask in self.requests:
            ok = ask <= left
            admitted.append(ok)
            if ok:
                left = left - ask
        return admitted, left.to("dollar")

    @property
    def equal_allowance(self) -> Quantity:
        """The largest equal allowance that admits every child."""
        return (self.available / len(self.requests)).to("dollar")

    @property
    def naive_allowance(self) -> Quantity:
        """An equal split of the whole remaining balance, ignoring the reserve and the parent's own work."""
        return (self.remaining / len(self.requests)).to("dollar")


class CohortArmRecord(BaseModel):
    """One policy's outcomes and settled usage over a cohort of attempted tasks."""
    accepted: int                             # accepted by the final cutoff
    on_time: int                              # accepted by the deadline
    median_accepted_time: Quantity            # among accepted tasks
    input_tokens: Quantity
    output_tokens: Quantity
    service_time: Quantity                    # metered tool, assessment, and cleanup time
    discard_input_tokens: Quantity            # failed attempts and canceled branches, included above
    discard_output_tokens: Quantity
    discard_service_time: Quantity


class DelegationCohortScenario(BaseScenario):
    """A single controller and a delegated design on one matched cohort, priced alike.

    Spend charges every initiated attempt at the same prices; cost per accepted
    task divides it by the accepted count. A policy passes the screen if it meets
    both acceptance floors within the budget.
    """
    attempted: int
    accepted_floor: int
    on_time_floor: int
    deadline: Quantity
    cutoff: Quantity
    budget: Quantity
    input_price: Quantity                     # USD per million tokens
    output_price: Quantity
    service_price: Quantity                   # USD per second
    single: CohortArmRecord
    delegated: CohortArmRecord

    def spend(self, arm: CohortArmRecord) -> Quantity:
        return (arm.input_tokens * self.input_price + arm.output_tokens * self.output_price
                + arm.service_time * self.service_price).to("dollar")

    def discarded(self, arm: CohortArmRecord) -> Quantity:
        return (arm.discard_input_tokens * self.input_price + arm.discard_output_tokens * self.output_price
                + arm.discard_service_time * self.service_price).to("dollar")

    def cost_per_accepted(self, arm: CohortArmRecord) -> Quantity:
        return self.spend(arm) / arm.accepted

    def passes(self, arm: CohortArmRecord) -> bool:
        return (arm.accepted >= self.accepted_floor and arm.on_time >= self.on_time_floor
                and self.spend(arm) <= self.budget)

    @property
    def extra_spend(self) -> Quantity:
        return self.spend(self.delegated) - self.spend(self.single)

    @property
    def extra_on_time(self) -> int:
        return self.delegated.on_time - self.single.on_time

    @property
    def marginal_on_time_cost(self) -> Quantity:
        """Added spend per added on-time acceptance."""
        return self.extra_spend / self.extra_on_time

    @property
    def cost_ratio(self) -> float:
        return (self.cost_per_accepted(self.delegated) / self.cost_per_accepted(self.single)).to("").magnitude

    @property
    def discard_share(self) -> float:
        """Delegated spend on failed and canceled work."""
        return (self.discarded(self.delegated) / self.spend(self.delegated)).to("").magnitude


class MessageRoundScenario(BaseScenario):
    """One round in which each of ``workers`` agents speaks once: full mesh against a supervisor."""
    workers: int

    @property
    def mesh_deliveries(self) -> int:
        from ..physics.coordination import calc_full_mesh_deliveries
        return calc_full_mesh_deliveries(self.workers)

    @property
    def supervisor_deliveries(self) -> int:
        from ..physics.coordination import calc_supervisor_deliveries
        return calc_supervisor_deliveries(self.workers)

    @property
    def ratio(self) -> float:
        """(N - 1) / 2."""
        return self.mesh_deliveries / self.supervisor_deliveries
# --- Responsible physical AI: decisions, oversight, stages, groups, records, labor (added 2026-10-20) ---

class DecisionChainScenario(BaseScenario):
    """``links`` independently kept decision records behind one admitted action, each retrievable
    with probability p; the whole chain resolves with p^links for each p in ``link_probabilities``."""
    links: int
    link_probabilities: Tuple[float, ...]

    def completeness(self, p: float) -> float:
        from ..physics.statistics import calc_conjunctive_confidence
        return calc_conjunctive_confidence([p] * self.links)

    @property
    def completeness_by_link(self) -> Dict[float, float]:
        return {p: self.completeness(p) for p in self.link_probabilities}


class OversightTimeScenario(BaseScenario):
    """A person asked to act through ``budget`` (a TakeoverBudgetScenario) against the time the
    hazard at its site leaves: the clear distance at the request speed."""
    budget: Any                                  # TakeoverBudgetScenario

    @property
    def hazard_window(self) -> Quantity:
        return (self.budget.site.d_clear / self.budget.speed).to("s")

    @property
    def takeover_travel(self) -> Quantity:
        return self.budget.result.takeover_travel.to("m")

    @property
    def takeover_over_window(self) -> float:
        return (self.budget.takeover / self.hazard_window).to("").magnitude


class SupervisorQueueScenario(BaseScenario):
    """One remote assistant serving ``machines``, each asking for help at ``request_rate``, each
    request holding the assistant for ``service_time``; ``utilization_cap`` sizes the largest fleet."""
    request_rate: Quantity                       # requests per machine per operating hour
    service_time: Quantity
    machines: int
    utilization_cap: Optional[float] = None

    @property
    def result(self):
        from .solvers import SupervisorQueueModel
        return SupervisorQueueModel().solve(self.request_rate, self.service_time, self.machines)

    @property
    def max_machines(self) -> int:
        """The largest fleet whose utilization stays at or below the cap."""
        return _math.floor(self.utilization_cap / (self.request_rate * self.service_time).to("").magnitude)


class SupervisionLaborScenario(BaseScenario):
    """The person-hours a supervised fleet creates in one ``operating_day``: an assistant staffed the
    whole day (busy at ``queue``'s utilization, watching otherwise) plus ``sweeps`` corridor sweeps."""
    queue: SupervisorQueueScenario
    operating_day: Quantity
    sweeps: int
    sweep_time: Quantity
    shift: Quantity                              # one full-time shift

    @property
    def request_time(self) -> Quantity:
        return (self.queue.result.utilization * self.operating_day).to("hour")

    @property
    def watch_time(self) -> Quantity:
        return (self.operating_day - self.request_time).to("hour")

    @property
    def sweep_total(self) -> Quantity:
        return (self.sweeps * self.sweep_time).to("hour")

    @property
    def total(self) -> Quantity:
        return (self.operating_day + self.sweep_total).to("hour")

    @property
    def shifts(self) -> float:
        return (self.total / self.shift).to("").magnitude


class DetectionShortfallScenario(BaseScenario):
    """``site.machine`` at ``speed`` with a stop sized for a group detected in ``reference`` of
    frames, met by ``groups`` detected less often; misses allowed in every frame with probability
    ``all_miss_probability``, frames independent."""
    site: Any                                    # EmbodiedSiteScenario
    speed: Quantity
    all_miss_probability: float
    reference: float
    groups: Tuple[float, ...]

    @property
    def frame_period(self) -> Quantity:
        return (1 / self.site.machine.nav_camera.frame_rate).to("ms")

    @property
    def result(self):
        from .solvers import DetectionShortfallModel
        s = self.site
        return DetectionShortfallModel().solve(
            self.speed, self.frame_period, s.a_brake, s.machine.loaded_mass, self.all_miss_probability,
            self.reference, self.groups, delay=s.tau_delay)


class SubgroupSampleScenario(BaseScenario):
    """An evaluation set of ``total`` people with a group of ``small_share``, both detected at
    ``recall``: exact intervals for each group, and the people per group a two-sided test at
    ``significance`` with ``power`` needs to tell ``rate_a`` from ``rate_b``."""
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
        return round(self.total * self.small_share)

    @property
    def large(self) -> int:
        return self.total - self.small

    def interval(self, n: int):
        from ..physics.statistics import calc_clopper_pearson_interval
        return calc_clopper_pearson_interval(round(self.recall * n), n, self.confidence)

    @property
    def small_interval(self):
        return self.interval(self.small)

    @property
    def large_interval(self):
        return self.interval(self.large)

    @property
    def width_ratio(self) -> float:
        (a, b), (c, d) = self.small_interval, self.large_interval
        return (b - a) / (d - c)

    @property
    def per_group_needed(self) -> int:
        from scipy.stats import norm
        from ..physics.statistics import calc_two_proportion_sample_size
        lo, hi = sorted((self.rate_a, self.rate_b))
        return _math.ceil(calc_two_proportion_sample_size(
            lo, hi - lo, float(norm.ppf(1 - self.significance / 2)), float(norm.ppf(self.power)),
            pooled=True))

    @property
    def share_needed(self) -> float:
        return self.per_group_needed / self.total


class ExposureConcentrationScenario(BaseScenario):
    """``encounter_share`` of a site's machine-person encounters fall on ``user_share`` of its people:
    their per-person exposure against the site average, and everyone else's."""
    encounter_share: float
    user_share: float

    @property
    def concentration(self) -> float:
        return self.encounter_share / self.user_share

    @property
    def rest_ratio(self) -> float:
        return (1 - self.encounter_share) / (1 - self.user_share)


class EventRetentionScenario(BaseScenario):
    """Event-triggered retention: ``pre_trigger`` before and ``post_trigger`` after each of
    ``events_per_day`` events, against keeping the whole ``operating_day``."""
    pre_trigger: Quantity
    post_trigger: Quantity
    events_per_day: int
    operating_day: Quantity

    @property
    def window(self) -> Quantity:
        return (self.pre_trigger + self.post_trigger).to("s")

    @property
    def kept_per_day(self) -> Quantity:
        return (self.events_per_day * self.window).to("minute")

    @property
    def kept_share(self) -> float:
        return (self.kept_per_day / self.operating_day).to("").magnitude

    @property
    def reduction(self) -> float:
        return 1 / self.kept_share


class DisclosurePenaltyScenario(BaseScenario):
    """Penalties for an incomplete account: ``days_withheld`` at ``daily_maximum`` each, plus
    ``others`` (name -> amount), all in USD."""
    days_withheld: float
    daily_maximum: float
    others: Dict[str, float]

    @property
    def per_day_penalty(self) -> float:
        return self.days_withheld * self.daily_maximum

    @property
    def total(self) -> float:
        return self.per_day_penalty + sum(self.others.values())



# --- ML operations: release controls, automation, skew, drift (added 2026-10-20) ---


class ReleaseFootprintScenario(BaseScenario):
    """Resident copies of ``model`` that release controls hold on ``device``.

    A copy is the registry ``model_size``, or the parameters at
    ``bytes_per_param`` when given; it spans ``accelerators_per_copy`` devices.
    A canary, shadow, A/B test, blue-green switch, and hot standby each hold one
    more copy whatever share of traffic they serve. Optional: the node's
    ``node_accelerators``, a retraining run of ``retrain_accelerators`` for
    ``retrain_time`` on the same node, the ``canary_share``, and the staged
    exposure ``ramp`` (traffic shares, first stage the canary).
    """
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
        if self.bytes_per_param is None:
            return self.model.model_size
        return self.model.size_in_bytes(self.bytes_per_param)

    @property
    def copy_fraction(self) -> float:
        """One copy as a share of one device's memory."""
        return (self.copy_bytes / self.device.memory.capacity).to("").magnitude

    @property
    def accelerators_per_copy(self) -> int:
        return _math.ceil(self.copy_fraction)

    def accelerators_for(self, copies: int) -> int:
        return copies * self.accelerators_per_copy

    @property
    def retrain_accelerator_hours(self) -> Quantity:
        return (self.retrain_accelerators * self.retrain_time).to("hour")

    @property
    def free_during_retrain(self) -> int:
        """Accelerators left beside the serving copy and the retraining run."""
        return self.node_accelerators - self.accelerators_per_copy - self.retrain_accelerators

    def free_after(self, copies: int) -> int:
        """Accelerators left once ``copies`` copies are resident and no run is active."""
        return self.node_accelerators - self.accelerators_for(copies)


class DualBankUpdateScenario(BaseScenario):
    """An over-the-air update that keeps a second flash bank: two copies of ``model``
    at ``bytes_per_param`` in the flash of ``device``."""
    model: Any
    device: HardwareNode
    bytes_per_param: Quantity

    @property
    def bank_bytes(self) -> Quantity:
        return self.model.size_in_bytes(self.bytes_per_param)

    @property
    def banks_bytes(self) -> Quantity:
        return 2 * self.bank_bytes

    @property
    def flash(self) -> Quantity:
        return self.device.memory.flash_capacity

    @property
    def flash_share(self) -> float:
        return (self.banks_bytes / self.flash).to("").magnitude


class AutomationBreakevenScenario(BaseScenario):
    """A pipeline that costs ``build_time`` once, against ``manual_rate`` of work by hand;
    the pipeline itself needs ``upkeep_rate``. Break-even is
    ``build / (manual - upkeep)``."""
    build_time: Quantity
    manual_rate: Quantity               # hours per week
    upkeep_rate: Optional[Quantity] = None

    @property
    def breakeven(self) -> Quantity:
        from ..physics.economics import calc_rent_crossover
        upkeep = self.upkeep_rate if self.upkeep_rate is not None else 0 * self.manual_rate
        return calc_rent_crossover(self.build_time, upkeep, self.manual_rate).to("week")


class SkewCostScenario(BaseScenario):
    """Training-serving skew that makes ``skew_rate`` of ``query_rate`` wrong, each
    wrong prediction costing ``cost_per_error``."""
    skew_rate: float
    query_rate: Quantity                # queries per day
    cost_per_error: Quantity

    @property
    def errors_per_day(self) -> float:
        return (self.skew_rate * self.query_rate * Q_(1, "day")).to("").magnitude

    @property
    def cost_rate(self) -> Quantity:
        from ..physics.economics import calc_skew_cost
        return calc_skew_cost(self.skew_rate, self.query_rate, self.cost_per_error).to("USD/day")

    def cost_over(self, period: Quantity) -> Quantity:
        return (self.cost_rate * period).to("USD")


class AccuracyDecayScenario(BaseScenario):
    """A deployed model read at ``initial_accuracy`` and, ``elapsed`` later, at
    ``later_accuracy``; ``noticed`` is when someone outside monitoring saw it.
    ``decay_rate`` fits ``A(t) = A0 exp(-gamma t)`` through the two readings."""
    initial_accuracy: float
    later_accuracy: float
    elapsed: Quantity
    noticed: Optional[Quantity] = None

    @property
    def decay_rate(self) -> Quantity:
        from ..physics.statistics import calc_exponential_decay_rate
        return calc_exponential_decay_rate(self.initial_accuracy, self.later_accuracy, self.elapsed)

    @property
    def drop(self) -> float:
        return self.initial_accuracy - self.later_accuracy


class AnalysisGateScenario(BaseScenario):
    """A rollout analysis that measures every ``interval``, passes a measurement at or
    above ``success_threshold``, and tolerates ``failure_limit`` failed measurements."""
    success_threshold: float
    interval: Quantity
    failure_limit: int

    @property
    def failures_to_abort(self) -> int:
        """The run fails when failed measurements exceed the limit."""
        return self.failure_limit + 1


# --- Serving a stream of tokens on one machine (added 2026-10-20) ------------
class KvDecodeBatchScenario(BaseScenario):
    """Batched decode on one accelerator when every sequence also reads its own KV cache.

    Each step reads the weights once and each sequence's cache of
    ``context_tokens``: bytes ``W + B m_seq``, time ``(W + B m_seq) / B_mem``
    (a bandwidth floor), aggregate rate ``B / step``. The rate approaches
    ``B_mem / m_seq`` as caches dominate. A TPOT target caps the batch at
    ``floor((t B_mem - W) / m_seq)``; memory caps it at
    ``floor((M - W) / m_seq)``. Prefill of ``prompt_tokens`` runs at
    ``prefill_efficiency`` of the ``peak_precision`` peak.
    """
    workload: Workload
    hardware: HardwareNode
    weight_bytes_per_param: Quantity
    kv_bytes_per_value: Quantity
    context_tokens: int
    batch_sizes: List[int]
    prompt_tokens: int
    prefill_efficiency: float
    peak_precision: str = "fp16_dense"

    @property
    def weights(self) -> Quantity:
        return self.workload.size_in_bytes(self.weight_bytes_per_param).to("GB")

    @property
    def kv_per_token(self) -> Quantity:
        return self.workload.get_kv_cache_size(1, 1, self.kv_bytes_per_value).to("byte")

    @property
    def kv_per_sequence(self) -> Quantity:
        return (self.kv_per_token * self.context_tokens).to("MB")

    @property
    def bandwidth(self) -> Quantity:
        return self.hardware.memory.bandwidth

    def step_bytes(self, batch_size: int) -> Quantity:
        return (self.weights + batch_size * self.kv_per_sequence).to("GB")

    def step_time(self, batch_size: int) -> Quantity:
        from ..physics import transfer_time
        return transfer_time(self.step_bytes(batch_size), self.bandwidth).to("ms")

    def token_rate(self, batch_size: int) -> Quantity:
        return (batch_size / self.step_time(batch_size)).to("1/second")

    def cache_share(self, batch_size: int) -> float:
        return (batch_size * self.kv_per_sequence / self.step_bytes(batch_size)).to("").magnitude

    @property
    def rate_ceiling(self) -> Quantity:
        """Tokens per second when cache reads are the whole step."""
        return (self.bandwidth / self.kv_per_sequence).to("1/second")

    def max_batch_within(self, tpot: Quantity) -> int:
        """Largest batch whose step stays within ``tpot``."""
        return _math.floor(((tpot * self.bandwidth - self.weights) / self.kv_per_sequence).to("").magnitude)

    @property
    def max_batch_in_memory(self) -> int:
        """Largest batch whose weights and full caches fit the accelerator's memory."""
        free = self.hardware.memory.capacity - self.weights
        return _math.floor((free / self.kv_per_sequence).to("").magnitude)

    @property
    def peak_rate(self) -> Quantity:
        return self.hardware.compute.precision_flops[self.peak_precision]

    @property
    def ridge(self) -> Quantity:
        return (self.peak_rate / self.bandwidth).to("flop/byte")

    @property
    def prefill_ops(self) -> Quantity:
        return (self.workload.inference_flops * self.prompt_tokens).to("flop")

    @property
    def prefill_time(self) -> Quantity:
        return (self.prefill_ops / (self.peak_rate * self.prefill_efficiency)).to("ms")


class SingleAcceleratorLlmServiceScenario(BaseScenario):
    """An LLM chat service on one accelerator: latency, KV residency, request rate, and token cost.

    ``ServingModel`` gives TTFT for the prompt and the weight footprint; TPOT
    is the weight and one request's cache read plus the solver's per-layer
    dispatch cost. The KV reserve over the cache per token sets resident
    requests. The request rate is the smaller of the prefill admission rate
    and the residency turnover (Little's law at the batch-one request time);
    tokens per hour count prompt and output tokens.
    """
    workload: Workload
    hardware: HardwareNode
    precision: str                       # weight precision for ServingModel
    kv_bytes_per_value: Quantity
    prompt_tokens: int
    decode_tokens: int
    prefill_efficiency: float
    kv_reserve: Quantity
    prefill_rate: Quantity               # prompt tokens per second
    price: Quantity                      # USD per accelerator-hour
    ttft_target: Optional[Quantity] = None
    tpot_target: Optional[Quantity] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.decode_tokens

    def _serve(self, seq_len: int):
        from .solvers import ServingModel
        return ServingModel().solve(self.workload, self.hardware, seq_len=seq_len, batch_size=1,
                                    precision=self.precision, efficiency=self.prefill_efficiency)

    @property
    def ttft(self) -> Quantity:
        """Model-side time to first token for the prompt alone."""
        return self._serve(self.prompt_tokens).ttft.to("ms")

    @property
    def weights(self) -> Quantity:
        return self._serve(self.total_tokens).model_weights_size

    @property
    def bandwidth(self) -> Quantity:
        return self.hardware.memory.bandwidth

    @property
    def kv_per_token(self) -> Quantity:
        return self.workload.get_kv_cache_size(seq_len=1, batch_size=1, precision=self.kv_bytes_per_value)

    @property
    def kv_per_request(self) -> Quantity:
        return self.workload.get_kv_cache_size(seq_len=self.total_tokens, batch_size=1,
                                               precision=self.kv_bytes_per_value)

    @property
    def weight_floor(self) -> Quantity:
        from ..physics import transfer_time
        return transfer_time(self.weights, self.bandwidth)

    @property
    def dispatch_cost(self) -> Quantity:
        """The solver's per-step overhead beyond its bandwidth floor."""
        from ..physics import transfer_time
        res = self._serve(self.total_tokens)
        return res.itl - transfer_time(res.total_memory_required, self.bandwidth)

    @property
    def tpot(self) -> Quantity:
        from ..physics import transfer_time
        return transfer_time(self.weights + self.kv_per_request, self.bandwidth) + self.dispatch_cost

    @property
    def decode_time(self) -> Quantity:
        return (self.decode_tokens * self.tpot).to("second")

    @property
    def kv_capacity_tokens(self) -> int:
        return int(round((self.kv_reserve / self.kv_per_token).to("").magnitude))

    @property
    def resident_requests(self) -> int:
        return int(self.kv_capacity_tokens / self.total_tokens)

    @property
    def request_time(self) -> Quantity:
        return (self._serve(self.total_tokens).ttft + self.decode_time).to("second")

    @property
    def prefill_request_rate(self) -> Quantity:
        return (self.prefill_rate / self.prompt_tokens).to("1/second")

    @property
    def residency_request_rate(self) -> Quantity:
        from ..physics.serving import calc_littles_law_rate
        return calc_littles_law_rate(self.resident_requests, self.request_time).to("1/second")

    @property
    def request_rate(self) -> Quantity:
        return min(self.prefill_request_rate, self.residency_request_rate)

    @property
    def tokens_per_hour(self) -> float:
        return (self.request_rate * Q_(1, "hour")).to("").magnitude * self.total_tokens

    @property
    def cost_per_million_tokens(self) -> Quantity:
        return (self.price * Q_(1, "hour") / (self.tokens_per_hour / 1e6)).to("dollar")

    @property
    def remaining_memory(self) -> Quantity:
        return (self.hardware.memory.capacity - self.weights).to("GB")


class PrefillStallScenario(BaseScenario):
    """A new prompt's prefill inside the decode steps of running streams, whole or in chunks.

    A decode step reads the weights and ``decodes`` caches of
    ``context_tokens`` at memory bandwidth; prefill time scales linearly from
    ``prefill_time`` for ``prefill_tokens``. Run whole, the prompt stretches
    one step to ``decode + prefill(prompt)``; split into ``chunks``, each step
    adds one chunk and the new request's first token comes after ``chunks``
    steps. The largest chunk that keeps a step within ``tpot_target`` is
    ``(t - decode) / prefill per token``.
    """
    weights: Quantity
    kv_per_token: Quantity
    bandwidth: Quantity
    decodes: int
    context_tokens: int
    prefill_time: Quantity
    prefill_tokens: int
    prompt_tokens: int
    chunks: int
    tpot_target: Quantity

    @property
    def decode_bytes(self) -> Quantity:
        return (self.weights + self.decodes * self.context_tokens * self.kv_per_token).to("GB")

    @property
    def decode_step(self) -> Quantity:
        from ..physics import transfer_time
        return transfer_time(self.decode_bytes, self.bandwidth).to("ms")

    @property
    def prefill_per_token(self) -> Quantity:
        return (self.prefill_time / self.prefill_tokens).to("ms")

    @property
    def prompt_prefill(self) -> Quantity:
        return (self.prefill_per_token * self.prompt_tokens).to("ms")

    @property
    def stalled_step(self) -> Quantity:
        return (self.decode_step + self.prompt_prefill).to("ms")

    @property
    def chunk_tokens(self) -> float:
        return self.prompt_tokens / self.chunks

    @property
    def chunk_prefill(self) -> Quantity:
        return (self.prompt_prefill / self.chunks).to("ms")

    @property
    def chunked_step(self) -> Quantity:
        return (self.decode_step + self.chunk_prefill).to("ms")

    @property
    def chunked_first_token(self) -> Quantity:
        return (self.chunks * self.chunked_step).to("ms")

    @property
    def largest_safe_chunk(self) -> float:
        return ((self.tpot_target - self.decode_step) / self.prefill_per_token).to("").magnitude


class DisaggregatedPoolScenario(BaseScenario):
    """Separate prefill and decode pools sized from per-accelerator rates, against both phases on one.

    ``decode_accelerators`` decode at ``decode_rate`` each; prefill
    accelerators at ``prefill_rate`` each are added until they keep up
    (``ceil``). The pool's rate per accelerator, over ``colocated_rate``, is
    the gain from splitting. Each request's cache crosses the link once:
    ``kv_per_request`` at ``kv_arrival_rate`` is the traffic.
    """
    prefill_rate: Quantity               # requests/s per prefill accelerator
    decode_rate: Quantity                # requests/s per decode accelerator
    colocated_rate: Quantity             # requests/s per accelerator running both phases
    decode_accelerators: int = 1
    kv_per_request: Optional[Quantity] = None
    kv_arrival_rate: Optional[Quantity] = None
    link_bandwidth: Optional[Quantity] = None

    @property
    def pool_rate(self) -> Quantity:
        return (self.decode_accelerators * self.decode_rate).to("1/second")

    @property
    def prefill_accelerators_exact(self) -> float:
        return (self.pool_rate / self.prefill_rate).to("").magnitude

    @property
    def prefill_accelerators(self) -> int:
        return _math.ceil(self.prefill_accelerators_exact)

    @property
    def accelerators(self) -> int:
        return self.prefill_accelerators + self.decode_accelerators

    @property
    def per_accelerator_rate(self) -> Quantity:
        return (self.pool_rate / self.accelerators).to("1/second")

    @property
    def gain(self) -> float:
        return (self.per_accelerator_rate / self.colocated_rate).to("").magnitude

    @property
    def kv_traffic(self) -> Quantity:
        return (self.kv_per_request * self.kv_arrival_rate).to("GB/second")

    @property
    def link_share(self) -> float:
        return (self.kv_traffic / self.link_bandwidth).to("").magnitude


class TokenEnergyScenario(BaseScenario):
    """Joules per generated token: node power over aggregate token rate, well used and poorly used.

    The node draws the accelerator's power plus a host share whether a step
    carries one sequence or many. ``concurrent_requests`` at
    ``per_request_rate`` set the well-used token rate; at
    ``low_use_fraction`` of that rate the accelerator still draws
    ``low_use_accelerator_power``.
    """
    accelerator_power: Quantity
    host_power: Quantity
    low_use_accelerator_power: Quantity
    concurrent_requests: int
    per_request_rate: Quantity           # tokens per second per request
    response_tokens: int
    low_use_fraction: float

    @property
    def node_power(self) -> Quantity:
        return (self.accelerator_power + self.host_power).to("watt")

    @property
    def low_use_power(self) -> Quantity:
        return (self.low_use_accelerator_power + self.host_power).to("watt")

    @property
    def token_rate(self) -> Quantity:
        return (self.concurrent_requests * self.per_request_rate).to("1/second")

    def energy_per_token(self, token_rate: Quantity, low_use: bool = False) -> Quantity:
        """Joules per token at ``token_rate``, at full or low-use power."""
        power = self.low_use_power if low_use else self.node_power
        return (power / token_rate).to("joule")

    @property
    def joules_per_token(self) -> Quantity:
        return self.energy_per_token(self.token_rate)

    @property
    def response_energy(self) -> Quantity:
        return (self.joules_per_token * self.response_tokens).to("joule")

    @property
    def low_use_token_rate(self) -> Quantity:
        return (self.token_rate * self.low_use_fraction).to("1/second")

    @property
    def low_use_joules_per_token(self) -> Quantity:
        return self.energy_per_token(self.low_use_token_rate, low_use=True)

    @property
    def low_use_ratio(self) -> float:
        return (self.low_use_joules_per_token / self.joules_per_token).to("").magnitude


class KvPoolAccountingScenario(BaseScenario):
    """A serving engine's reported KV pool checked against an estimate of free memory.

    The engine claims ``memory_utilization`` of ``capacity``; after the
    ``weights`` and the reported ``kv_pool`` the rest is its activation and
    workspace reserve. The pool's ``kv_pool_tokens`` times the cache per
    token should equal ``kv_pool``. Paged in blocks of ``block_tokens``, a
    request of ``request_tokens`` holds ``ceil(S / b)`` blocks.
    """
    capacity: Quantity
    memory_utilization: float
    weights: Quantity
    kv_pool: Quantity
    kv_pool_tokens: int
    kv_per_token: Quantity
    block_tokens: int
    context_tokens: int                  # tokens per request in the engine's concurrency line
    request_tokens: int                  # tokens per paged conversation

    @property
    def claimed(self) -> Quantity:
        return (self.memory_utilization * self.capacity).to("GiB")

    @property
    def held_back(self) -> Quantity:
        return ((1 - self.memory_utilization) * self.capacity).to("GB")

    @property
    def workspace(self) -> Quantity:
        """Claimed memory left after the weights and the KV pool."""
        return (self.claimed - self.weights - self.kv_pool).to("GiB")

    @property
    def pool_bytes(self) -> Quantity:
        return (self.kv_pool_tokens * self.kv_per_token).to("GiB")

    @property
    def concurrency(self) -> float:
        return self.kv_pool_tokens / self.context_tokens

    @property
    def blocks_per_request(self) -> int:
        from ..physics.memory import calc_kv_block_count
        return calc_kv_block_count(self.request_tokens, self.block_tokens)

    @property
    def paged_request_bytes(self) -> Quantity:
        return (self.blocks_per_request * self.block_tokens * self.kv_per_token).to("MB")

    def paged_requests(self, budget: Quantity) -> int:
        """Whole paged conversations ``budget`` holds."""
        return _math.floor((budget / self.paged_request_bytes).to("").magnitude)

    @property
    def pool_paged_requests(self) -> int:
        """Whole paged conversations in the engine's own pool of whole blocks."""
        return (self.kv_pool_tokens // self.block_tokens) // self.blocks_per_request

    @property
    def last_block_waste(self) -> Quantity:
        """Most a request wastes: all but one slot of its last block."""
        return ((self.block_tokens - 1) * self.kv_per_token).to("MB")
# --- Fairness release gates, their price, and their stability (added 2026-10-20) ---------
class SubgroupEvaluationBudgetScenario(BaseScenario):
    """The labeled data a per-group evaluation needs: one group's rate within ``half_width`` at
    ``confidence`` (worst case ``rate`` 0.5), the collection random sampling needs to reach a
    ``target_per_group`` from a group of ``group_share``, and the per-group count that pins a gap
    between two equal-size groups to ``gap_half_width``."""
    rate: float = 0.5
    half_width: float
    confidence: float
    group_share: float
    target_per_group: int
    gap_half_width: Optional[float] = None

    @property
    def z(self) -> float:
        from statistics import NormalDist
        return NormalDist().inv_cdf(0.5 + self.confidence / 2)

    @property
    def variance(self) -> float:
        """``p (1 - p)``, 0.25 at the worst case."""
        return self.rate * (1 - self.rate)

    @property
    def samples_needed(self) -> int:
        from ..physics.statistics import calc_samples_for_half_width
        return calc_samples_for_half_width(self.rate, self.half_width, self.z)

    @property
    def random_collection(self) -> float:
        """Examples drawn at random to expect ``target_per_group`` from the group."""
        return self.target_per_group / self.group_share

    @property
    def random_multiple(self) -> float:
        return self.random_collection / self.target_per_group

    def gap_samples_per_group(self, half_width: Optional[float] = None) -> int:
        from ..physics.fairness import calc_gap_samples_per_group
        return calc_gap_samples_per_group(self.rate, half_width or self.gap_half_width, self.z)

    def gap_half_width_at(self, n_per_group: int) -> float:
        from ..physics.statistics import calc_proportion_gap_se
        return self.z * calc_proportion_gap_se(self.rate, self.rate, n_per_group, n_per_group)


class EqualOpportunityPriceScenario(BaseScenario):
    """Utility per applicant of raising ``group``'s TPR to ``reference_group``'s by moving its
    threshold, when that move raises its FPR by ``fpr_increase``: a repaid loan earns
    ``repay_margin``, a default loses ``default_loss`` (both USD)."""
    confusion: GroupConfusionScenario
    group: str
    reference_group: str
    repay_margin: Quantity
    default_loss: Quantity
    fpr_increase: float

    @property
    def base_rate(self) -> float:
        return self.confusion.base_rate(self.group)

    @property
    def tpr_before(self) -> float:
        return self.confusion.tpr(self.group)

    @property
    def tpr_after(self) -> float:
        return self.confusion.tpr(self.reference_group)

    @property
    def fpr_before(self) -> float:
        return self.confusion.fpr(self.group)

    @property
    def tpr_gap(self) -> float:
        return self.tpr_after - self.tpr_before

    @property
    def baseline_utility(self) -> Quantity:
        p = self.base_rate
        return self.tpr_before * p * self.repay_margin - self.fpr_before * (1 - p) * self.default_loss

    @property
    def added_repaid_value(self) -> Quantity:
        return self.tpr_gap * self.base_rate * self.repay_margin

    @property
    def added_default_cost(self) -> Quantity:
        return self.fpr_increase * (1 - self.base_rate) * self.default_loss

    @property
    def net_change(self) -> Quantity:
        return self.added_repaid_value - self.added_default_cost

    @property
    def utility_after(self) -> Quantity:
        return self.baseline_utility + self.net_change

    @property
    def loss_fraction(self) -> float:
        """Utility given up as a fraction of the group's baseline."""
        return (-self.net_change / self.baseline_utility).to("").magnitude

    @property
    def group_share(self) -> float:
        return self.confusion.share(self.group)

    @property
    def population_average_loss(self) -> Quantity:
        """The group's net loss spread over every applicant."""
        return -self.net_change * self.group_share


class BiasFeedbackScenario(BaseScenario):
    """A gap passed at launch, refit from the next audit, iterated over retraining cycles:
    ``gap(k) = launch_gap * factor^k`` against a release ``gate`` (all in percentage points)."""
    launch_gap: float
    next_audit_gap: float
    gate: float
    sampled_factor: float              # the refit factor once censored outcomes are sampled
    unmitigated_gap: float
    horizon_cycles: int

    @property
    def factor(self) -> float:
        return self.next_audit_gap / self.launch_gap

    @property
    def crossing(self) -> float:
        from ..physics.fairness import calc_cycles_to_gate
        return calc_cycles_to_gate(self.launch_gap, self.factor, self.gate)

    @property
    def breach_cycle(self) -> int:
        return _math.ceil(self.crossing)

    def gap_at(self, cycles: int, factor: Optional[float] = None) -> float:
        from ..physics.fairness import calc_feedback_disparity
        return calc_feedback_disparity(self.launch_gap, factor or self.factor, cycles)

    @property
    def gap_at_breach(self) -> float:
        return self.gap_at(self.breach_cycle)

    @property
    def unmitigated_at_horizon(self) -> float:
        from ..physics.fairness import calc_feedback_disparity
        return calc_feedback_disparity(self.unmitigated_gap, self.factor, self.horizon_cycles)

    @property
    def sampled_gap_at_breach(self) -> float:
        return self.gap_at(self.breach_cycle, self.sampled_factor)


class ExplorationExposureScenario(BaseScenario):
    """A ranking service showing ``slots_per_user`` items a day to ``daily_users`` from a catalog of
    ``catalog_items``, with ``explore_share`` of slots served by randomized exploration."""
    daily_users: int
    slots_per_user: int
    catalog_items: int
    explore_share: float

    @property
    def impressions(self) -> int:
        return self.daily_users * self.slots_per_user

    @property
    def catalog_coverage(self) -> float:
        """Share of the catalog one user sees in a day."""
        return self.slots_per_user / self.catalog_items

    @property
    def explore_impressions(self) -> float:
        return self.impressions * self.explore_share


class PpvErrorRateIdentityScenario(BaseScenario):
    """Published per-group rates checked against Chouldechova's identity:
    ``groups = {name: {positives, total, ppv, fnr, fpr}}`` (``fpr`` as published, a fraction)."""
    groups: Dict[str, Dict[str, float]]

    def base_rate(self, group: str) -> float:
        g = self.groups[group]
        return g["positives"] / g["total"]

    def odds(self, group: str) -> float:
        p = self.base_rate(group)
        return p / (1 - p)

    def precision_term(self, group: str) -> float:
        ppv = self.groups[group]["ppv"]
        return (1 - ppv) / ppv

    def fpr(self, group: str) -> float:
        from ..physics.fairness import calc_fpr_from_ppv
        g = self.groups[group]
        return calc_fpr_from_ppv(self.base_rate(group), g["ppv"], g["fnr"])


class AccuracyFrontierScenario(BaseScenario):
    """A hypothetical accuracy-against-disparity frontier, ``floor + span (1 - exp(-rate x))``, with
    named candidate disparities on it."""
    floor: float
    span: float
    rate: float
    candidates: Dict[str, float]

    def accuracy(self, disparity):
        import numpy as np
        return self.floor + self.span * (1 - np.exp(-self.rate * np.asarray(disparity)))

    def point(self, name: str):
        x = self.candidates[name]
        return x, float(self.accuracy(x))

    def disparity_ratio(self, worse: str, better: str) -> float:
        return self.candidates[worse] / self.candidates[better]


class ProxyGoalGapScenario(BaseScenario):
    """A proxy and its goal from a common normalized start: proxy changes by ``proxy_change``,
    the goal by ``goal_change`` (fractions); the gap is proxy minus goal."""
    proxy_change: float
    goal_change: float
    start: float = 1.0

    @property
    def proxy_end(self) -> float:
        return self.start * (1 + self.proxy_change)

    @property
    def goal_end(self) -> float:
        return self.start * (1 + self.goal_change)

    @property
    def gap(self) -> float:
        return self.proxy_end - self.goal_end
# --- Agent cost accounting (added 2026-10-20) ---------------------------------
# What one accepted task costs and what each lever buys; solvers in
# engine/solvers/cost_accounting.py. Prices are Quantities in USD per Mtoken
# unless a field says otherwise.

class PriceDeclineScenario(BaseScenario):
    """Lowest price at each fixed capability in a ``Literature`` series, fitted per year."""
    series_name: str

    @property
    def rows(self):
        from ..literature.registry import Literature
        return getattr(Literature, self.series_name).entries

    @property
    def result(self):
        from .solvers import PriceDeclineModel
        return PriceDeclineModel().solve(self.rows)


class CostVsScoreScenario(BaseScenario):
    """Per-trial cost against mean score across a benchmark's configurations."""
    series_name: str
    trials: int                      # trials per configuration, by the benchmark's protocol
    budget_hours: float              # time budget per trial, by the benchmark's protocol

    @property
    def rows(self):
        from ..literature.registry import Literature
        return getattr(Literature, self.series_name).entries

    @property
    def result(self):
        from .solvers import CostVsScoreModel
        return CostVsScoreModel().solve(self.rows)


class CohortAttemptCostScenario(BaseScenario):
    """One configuration's cohort of tasks, one bounded attempt each, all charges settled."""
    tasks: int
    accepted: int
    input_price: Quantity            # USD per Mtoken
    output_price: Quantity
    turns: int                       # per attempt
    input_tokens_per_turn: int
    output_tokens_per_turn: int
    environment_per_turn: Quantity   # USD: tools, sandbox holding, cleanup
    assessment_per_attempt: Quantity # USD: checks not billed as sandbox time
    triage_per_rejected: Quantity    # USD: engineer triage of each rejected attempt

    @property
    def result(self):
        from .solvers import CohortAttemptCostModel
        return CohortAttemptCostModel().solve(self.tasks, self.accepted, self.input_price, self.output_price,
                                              self.turns, self.input_tokens_per_turn, self.output_tokens_per_turn,
                                              self.environment_per_turn, self.assessment_per_attempt,
                                              self.triage_per_rejected)


class TokenPriceComparisonScenario(BaseScenario):
    """Two cohort configurations under one acceptance contract: does the cheaper token win?"""
    baseline: CohortAttemptCostScenario
    alternative: CohortAttemptCostScenario

    @property
    def result(self):
        from .solvers import CohortAttemptCostModel
        return CohortAttemptCostModel.compare(self.baseline.result, self.alternative.result,
                                              self.baseline.accepted, self.alternative.accepted)


class ResendCacheBillScenario(BaseScenario):
    """An append-only trajectory (``TrajectoryResendScenario``) billed under one tariff, with and without reuse."""
    trajectory: TrajectoryResendScenario
    input_price: Quantity            # USD per Mtoken, fresh input
    cached_fraction: float           # cached input billed at this share of the input price
    output_price: Quantity
    miss_fraction: float             # share of the eligible prefix that misses, for sensitivity

    @property
    def result(self):
        from .solvers import PrefixCacheBillModel
        t = self.trajectory
        return PrefixCacheBillModel().solve(t.turns, t.first_prompt, t.appended, t.output_per_turn,
                                            self.input_price, self.cached_fraction, self.output_price,
                                            self.miss_fraction)


class CompactionPaybackScenario(BaseScenario):
    """One compaction of a billed trajectory: a summary resent fresh, then shorter cached reads."""
    bill: ResendCacheBillScenario
    compact_turn: int                # the call that compacts
    compact_to: int                  # tokens of the compacted context
    summary_tokens: int              # output tokens the summary costs

    @property
    def result(self):
        from .solvers import CompactionPaybackModel
        b, t = self.bill, self.bill.trajectory
        return CompactionPaybackModel().solve(t.turns, t.first_prompt, t.appended, b.input_price,
                                              b.cached_fraction, b.output_price, self.compact_turn,
                                              self.compact_to, self.summary_tokens)


class RetryPolicyScenario(BaseScenario):
    """A cheap attempt retried up to a cap, run as a batch, or one stronger attempt."""
    pass_probability: float
    attempt_cost: Quantity           # USD, all-in
    max_attempts: int
    strong_pass_probability: float
    strong_attempt_cost: Quantity

    @property
    def result(self):
        from .solvers import RetryPolicyModel
        return RetryPolicyModel().solve(self.pass_probability, self.attempt_cost, self.max_attempts,
                                        self.strong_pass_probability, self.strong_attempt_cost)


class SerialEpisodeScenario(BaseScenario):
    """One serial episode as disjoint intervals; the model's intervals sped up by ``model_speedup``."""
    model_intervals: List[Quantity]
    tests: Quantity
    lifecycle_intervals: List[Quantity]   # sandbox creation and snapshot export
    runtime_intervals: List[Quantity]     # runtime setup and status
    model_speedup: float

    @property
    def result(self):
        from .solvers import SerialEpisodeModel

        def secs(xs):
            return tuple(float(x.to("second").magnitude) for x in xs)
        return SerialEpisodeModel().solve(secs(self.model_intervals), float(self.tests.to("second").magnitude),
                                          secs(self.lifecycle_intervals), secs(self.runtime_intervals),
                                          self.model_speedup)


class SpeculativeEpisodeScenario(BaseScenario):
    """Speculative decoding on a serial episode's model time, at two draft acceptance levels."""
    episode: SerialEpisodeScenario
    draft_length: int                # gamma, tokens drafted per round
    draft_cost_ratio: float          # r, a draft token's cost as a share of a target step
    accepted_high: float             # m, drafted tokens kept per round when drafts align
    accepted_low: float

    @property
    def result(self):
        from .solvers import SpeculativeEpisodeModel
        e = self.episode.result
        return SpeculativeEpisodeModel().solve(float(e.total_time.to("second").magnitude), e.model_fraction,
                                               self.draft_length, self.draft_cost_ratio, self.accepted_high,
                                               self.accepted_low)


class ModelCascadeScenario(BaseScenario):
    """A tiered cascade over a completed cohort, every exit under one gate, against the top tier alone."""
    tasks: int
    call_costs: List[Quantity]       # USD per reached tier, gate excluded
    gate_cost: Quantity
    call_times: List[Quantity]
    gate_time: Quantity
    accepted_by_tier: List[int]
    direct_accepted: int             # the top tier alone over the whole cohort
    false_accept_rate: Optional[float] = None       # at the first tier, for the harm case
    harm_per_false_accept: Optional[Quantity] = None

    @property
    def result(self):
        from .solvers import ModelCascadeModel
        harm = None if self.harm_per_false_accept is None else float(self.harm_per_false_accept.to("USD").magnitude)
        return ModelCascadeModel().solve(
            self.tasks, [float(c.to("USD").magnitude) for c in self.call_costs],
            float(self.gate_cost.to("USD").magnitude), [float(t.to("second").magnitude) for t in self.call_times],
            float(self.gate_time.to("second").magnitude), self.accepted_by_tier, self.direct_accepted,
            self.false_accept_rate, harm)


class TrajectoryAdmissionScenario(BaseScenario):
    """A stationary admitted population of trajectories against a token-per-minute quota."""
    arrival: Quantity                # admissions per unit time
    residence: Quantity              # time a trajectory is present
    calls: int                       # model calls per trajectory
    tokens_per_call: int             # quota-counted tokens per call
    quota: Quantity                  # tokens per unit time
    acceptance_drop: Optional[float] = None   # relative acceptance drop at fixed goodput

    @property
    def result(self):
        from .solvers import TrajectoryAdmissionModel
        return TrajectoryAdmissionModel().solve(self.arrival, self.residence, self.calls, self.tokens_per_call,
                                                self.quota, self.acceptance_drop)


class KingmanQueueScenario(BaseScenario):
    """One server under Kingman's approximation at two utilizations, with a holding price."""
    service: Quantity                # mean service time
    arrival_scv: float               # c_a^2
    service_scv: float               # c_s^2
    low_utilization: float
    high_utilization: float
    holding_rate: Quantity           # USD per second of held resources

    def wait_at(self, utilization: float) -> Quantity:
        from ..physics.serving import calc_kingman_wait_ratio
        return calc_kingman_wait_ratio(utilization, self.arrival_scv, self.service_scv) * self.service

    @property
    def result(self):
        from .solvers import KingmanQueueModel
        return KingmanQueueModel().solve(self.service, self.arrival_scv, self.service_scv, self.low_utilization,
                                         self.high_utilization, float(self.holding_rate.to("USD / second").magnitude))


class SerialTailExposureScenario(BaseScenario):
    """Serial calls that each exceed their own tail latency independently."""
    calls: int
    tail_probability: float          # 0.01 for a 99th percentile

    @property
    def any_tail(self) -> float:
        """Chance that at least one call on the path exceeds its tail."""
        from ..physics.agents import calc_independent_success
        return calc_independent_success(self.tail_probability, self.calls)


class DedicatedBreakEvenScenario(BaseScenario):
    """Renting a matched-capability API against owning capacity at a fixed hourly cost."""
    input_per_output: float          # input tokens per output token
    cached_fraction: float           # share of input billed at the cached price
    input_price: Quantity            # USD per Mtoken
    cached_price: Quantity
    output_price: Quantity
    fixed_per_hour: Quantity         # USD per hour, owned capacity
    variable_price: Quantity         # USD per Mtoken output, owned capacity
    reference_rate: Quantity         # output tokens per second at full load
    load: float                      # operating load, a fraction of the reference rate

    def _p(self, q):
        return float(q.to("USD / Mtoken").magnitude)

    def hourly_costs(self, load: float):
        """(API, owned) dollars per hour at ``load``."""
        from .solvers import DedicatedBreakEvenModel
        r = self.result
        return DedicatedBreakEvenModel.hourly_costs(load, self.reference_rate, r.api_price,
                                                    float(self.fixed_per_hour.to("USD / hour").magnitude),
                                                    self._p(self.variable_price))

    @property
    def result(self):
        from .solvers import DedicatedBreakEvenModel
        return DedicatedBreakEvenModel().solve(
            self.input_per_output, self.cached_fraction, self._p(self.input_price), self._p(self.cached_price),
            self._p(self.output_price), float(self.fixed_per_hour.to("USD / hour").magnitude),
            self._p(self.variable_price), self.reference_rate, self.load)


class ConservedBudgetScenario(BaseScenario):
    """A root budget granted to two children; the second stops with a remote job billed late."""
    budget: Quantity                 # USD
    planning: Quantity               # settled before delegation
    allocations: List[Quantity]      # granted to the first and second child
    settled_children: List[Quantity] # each child's settled charge at its close or stop
    pending: Quantity                # enforceable bound on the stopped child's remote job
    late_charge: Quantity            # its final bill, received after the local stop
    deliveries: int = 2              # times the final bill is delivered (settled once)

    @property
    def result(self):
        from .solvers import ConservedBudgetModel

        def m(q):
            return int(round(float(q.to("USD").magnitude) * 1000))
        return ConservedBudgetModel().solve(m(self.budget), m(self.planning), [m(x) for x in self.allocations],
                                            [m(x) for x in self.settled_children], m(self.pending),
                                            m(self.late_charge), self.deliveries)


class ArchitectureChoiceScenario(BaseScenario):
    """Single-call designs and a bounded loop under one check, with a human fallback for rejects."""
    tickets: int                     # per month
    single_costs: Dict[str, Quantity]        # automated charge per ticket, check excluded
    single_acceptance: Dict[str, float]
    loop_name: str
    turn_cost: Quantity              # per loop turn, check excluded
    turn_acceptance: List[float]     # conditional acceptance of each turn reached
    check_cost: Quantity             # the same acceptance check for every candidate
    human_cost: Quantity             # one fallback, its final check included
    human_resolution: float          # conditional on entering fallback
    resolution_floor: float          # declared combined-policy requirement
    point: float = 0.01              # one point of automated acceptance

    @property
    def result(self):
        from .solvers import ArchitectureChoiceModel
        return ArchitectureChoiceModel().solve(self.tickets, self.single_costs, self.single_acceptance,
                                               self.loop_name, self.turn_cost, self.turn_acceptance,
                                               self.check_cost, self.human_cost, self.human_resolution, self.point)


class AdaptationPaybackScenario(BaseScenario):
    """A one-time adaptation that cuts each task's model usage, against its incremental cost."""
    base: Dict[str, Any]             # input_tokens, output_tokens, steps, input_price, output_price
    adapted: Dict[str, Any]
    data_cost: Quantity
    training_cost: Quantity
    evaluation_cost: Quantity
    monthly_volume: int
    low_volume: int                  # sensitivity case

    @property
    def result(self):
        from .solvers import AdaptationPaybackModel
        return AdaptationPaybackModel().solve(self.base, self.adapted, self.data_cost, self.training_cost,
                                              self.evaluation_cost, self.monthly_volume, self.low_volume)


class WorkloadLeverScenario(BaseScenario):
    """Attempt cost by term for several workloads, and three levers on cost per accepted task.

    Each workload gives ``acceptance``, ``environment``, ``assessment`` in
    dollars, and either ``human`` (fixed per attempt) or ``handoff`` (charged
    per rejected attempt). The model term is ``model`` dollars, or the
    ``bill``'s cached trajectory cost, or ``token_multiple`` chats of
    ``chat_input``/``chat_output`` tokens at the stated tariff.
    """
    workloads: Dict[str, Dict[str, Any]]
    input_price: Quantity            # USD per Mtoken, for workloads priced in chats
    output_price: Quantity
    acceptance_gain: float           # absolute gain for the acceptance lever

    def terms(self) -> Dict[str, Dict[str, float]]:
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
        from .solvers import WorkloadLeverModel
        return WorkloadLeverModel().solve(self.terms(), self.acceptance_gain)


def _usd_unit():
    from ..core.units import USD
    return USD


class CoverageIndependenceScenario(BaseScenario):
    """Measured repeated-sampling coverage of one system on one dataset, against independent trials."""
    series_name: str
    system: str
    dataset: str

    @property
    def rows(self):
        from ..literature.registry import Literature
        return [r for r in getattr(Literature, self.series_name).entries
                if r["model"] == self.system and r["dataset"] == self.dataset]

    @property
    def result(self):
        from .solvers import CoverageIndependenceModel
        return CoverageIndependenceModel().solve(self.rows)


class ConcurrencySweepScenario(BaseScenario):
    """One measured serving configuration from a ``ServingFrontierScenario``, swept across concurrency."""
    frontier: ServingFrontierScenario
    hardware: str
    match: Dict[str, Any]            # framework, disaggregation, run date, ...

    @property
    def rows(self):
        return self.frontier.rows(self.hardware, **self.match)

    @property
    def result(self):
        from .solvers import ConcurrencySweepModel
        f = self.frontier
        return ConcurrencySweepModel().solve(self.rows, f.x_column, f.y_column)
# --- Premises no sensor reports: a new machine, shield credit, latch compliance (added 2026-10-20) ---

class PackagingCellScenario(BaseScenario):
    """A packaging cell's carriage that must stop before a fixed stop line.

    The pre-brake delay is a chain of four allocations (event to usable frame,
    transfer through admission, command delivery, brake onset); the stop is the
    elementary constant-deceleration budget (``StoppingBudgetModel``) with the
    allocated reserve held back from the clearance, so the speed ceiling is the
    positive root of ``v tau + v^2/(2a) = clearance - reserve``. The supervisory
    settings (force limiter, thermistor trip, stall timeout, bus overvoltage trip,
    kinematic-check rate) are selected thresholds, not measured limits.
    """
    mass: Quantity               # total moving mass, including the load
    speed: Quantity
    clearance: Quantity          # to the fixed stop line
    reserve: Quantity            # allocated uncertainty and clearance reserve
    deceleration: Quantity       # braking lower bound after the delay
    exposure: Quantity           # event to usable frame
    processing: Quantity         # transfer, inference, scheduling, and admission
    bus: Quantity                # admitted command to brake-driver delivery
    onset: Quantity              # delivery to full assumed deceleration
    check_rate: Optional[Quantity] = None
    force_limit: Optional[Quantity] = None
    temperature_trip: Optional[Quantity] = None
    stall_timeout: Optional[Quantity] = None
    bus_trip: Optional[Quantity] = None

    @property
    def delay(self) -> Quantity:
        return (self.exposure + self.processing + self.bus + self.onset).to("ms")

    @property
    def stop_budget(self) -> Quantity:
        return (self.clearance - self.reserve).to("mm")

    @property
    def result(self):
        from .solvers import StoppingBudgetModel
        return StoppingBudgetModel().solve(self.speed, self.delay, self.deceleration, self.clearance,
                                           physical_margin=self.reserve)

    @property
    def remaining(self) -> Quantity:
        """Clearance left before the stop line (the reserve not subtracted)."""
        return (self.clearance - self.result.stop_distance).to("mm")

    @property
    def deficit(self) -> Quantity:
        return (self.result.stop_distance - self.clearance).to("mm")

    @property
    def kinetic_energy(self) -> Quantity:
        from ..physics.robotics.actuation import calc_kinetic_energy
        return calc_kinetic_energy(self.mass, self.speed).to("J")

    @property
    def root_terms(self):
        """Operands of the positive root: (a tau, 2 a d, sqrt((a tau)^2 + 2 a d))."""
        a_tau = (self.deceleration * self.delay).to("m/s")
        two_a_d = (2 * self.deceleration * self.stop_budget).to("m**2/s**2")
        return a_tau, two_a_d, ((a_tau ** 2 + two_a_d) ** 0.5).to("m/s")


class ShieldCoverageBillScenario(BaseScenario):
    """A fleet's zero-event proposal log and a shield's tested coverage against a hazard target."""
    fleet_units: int
    cycle_days: Quantity
    hours_per_day: Quantity      # operating hours per unit per calendar day
    coverage: float              # class-specific interception coverage
    common_cause_rate: Quantity  # hardware and common-cause residual, per hour
    target_rate: Quantity        # system hazard target, per hour
    confidence: float
    hazardous_proposals: int = 0

    @property
    def result(self):
        from .solvers import ShieldCoverageBillModel
        return ShieldCoverageBillModel().solve(self.fleet_units, self.cycle_days, self.hours_per_day,
                                               self.coverage, self.common_cause_rate, self.target_rate,
                                               self.confidence)


class LatchSeriesSpringScenario(BaseScenario):
    """A series spring sized so ``machine``'s arm stays under ``site``'s declared latch floor.

    The arm's stop is held to what the guarded approach on the rigid plate needs
    (``calc_latch_contact_stopping_profile``); at the fast approach the series
    stiffness must keep tripwire force plus driven stroke under the floor
    (``calc_passive_compliance_stiffness_bound``), and the spring element is the
    stiffness that gives it in series with the plate. Coasting peaks price the
    case where the response cuts torque at the tripwire instead
    (``calc_coasting_contact_peak_force``).
    """
    site: Any                    # EmbodiedSiteScenario (latch stiffness, floor, tripwire, speeds)
    machine: Any                 # EmbodiedMobileManipulator (arm effective contact mass)
    preload: Quantity = Q_(0.0, "N")

    def _profile(self, speed: Quantity, stiffness: Optional[Quantity] = None):
        from ..physics.robotics.stopping import calc_latch_contact_stopping_profile
        return calc_latch_contact_stopping_profile(approach_velocity=speed, latch_stiffness=stiffness,
                                                   scenario=self.site)

    @property
    def arm_deceleration(self) -> Quantity:
        return self._profile(self.site.v_latch_approach)["a_arm_needed"].to("m/s**2")

    @property
    def arm_deceleration_rigid_fast(self) -> Quantity:
        return self._profile(self.site.v_latch_high)["a_arm_needed"].to("m/s**2")

    def _bound(self):
        from ..physics.robotics.actuation import calc_passive_compliance_stiffness_bound
        s = self.site
        return calc_passive_compliance_stiffness_bound(s.f_latch_limit, s.f_latch_tripwire, s.v_latch_high,
                                                       s.t_contact_response, self.arm_deceleration)

    @property
    def driven_stroke(self) -> Quantity:
        return self._bound()[1].to("mm")

    @property
    def series_stiffness_max(self) -> Quantity:
        return self._bound()[0].to("N/m")

    @property
    def spring_stiffness_max(self) -> Quantity:
        from ..physics.robotics.learning import calc_complementary_series_stiffness
        return calc_complementary_series_stiffness(self.series_stiffness_max, self.site.k_latch).to("N/m")

    @property
    def spring_stroke(self) -> Quantity:
        """Spring deflection at the floor force."""
        return ((self.site.f_latch_limit - self.preload) / self.spring_stiffness_max).to("mm")

    def coasting_peak(self, stiffness: Quantity, speed: Quantity) -> Quantity:
        from ..physics.robotics.actuation import calc_coasting_contact_peak_force
        return calc_coasting_contact_peak_force(speed, self.machine.arm_effective_contact_mass, stiffness,
                                                self.site.f_latch_tripwire).to("N")

    @property
    def coasting_rigid_guarded(self) -> Quantity:
        return self.coasting_peak(self.site.k_latch, self.site.v_latch_approach)

    @property
    def coasting_rigid_fast(self) -> Quantity:
        return self.coasting_peak(self.site.k_latch, self.site.v_latch_high)

    @property
    def coasting_spring_fast(self) -> Quantity:
        return self.coasting_peak(self.series_stiffness_max, self.site.v_latch_high)

    @property
    def spring_fast_profile(self):
        """The latch profile at the fast approach with the series stiffness at its bound."""
        return self._profile(self.site.v_latch_high, self.series_stiffness_max)


class RestrictedTransitScenario(BaseScenario):
    """``site``'s aisle leg at ``restricted_speed`` against its normal aisle speed."""
    site: Any                    # EmbodiedSiteScenario (aisle speed, aisle length)
    restricted_speed: Quantity

    @property
    def speed_cut(self) -> float:
        return float((1 - self.restricted_speed / self.site.v_aisle).to("").magnitude)

    @property
    def leg_time_nominal(self) -> Quantity:
        return (self.site.aisle_length / self.site.v_aisle).to("s")

    @property
    def leg_time_restricted(self) -> Quantity:
        return (self.site.aisle_length / self.restricted_speed).to("s")

    @property
    def leg_time_added(self) -> Quantity:
        return (self.leg_time_restricted - self.leg_time_nominal).to("s")

    @property
    def time_increase(self) -> float:
        return float((self.leg_time_added / self.leg_time_nominal).to("").magnitude)
