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

    @property
    def result(self):
        from .solvers import DecodeBoundModel
        peak = (self.hardware.compute.precision_flops[self.peak_precision]
                if self.peak_precision else None)
        return DecodeBoundModel().solve(self.workload, self.hardware, self.precision, self.batch_size,
                                        peak_flops=peak)


class TransferScenario(BaseScenario):
    """``copies`` of one ``payload`` moved at once over each named link tier."""
    payload: Quantity
    links: Dict[str, Quantity]
    copies: int = 1

    @property
    def total_bytes(self) -> Quantity:
        return self.copies * self.payload

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

    @property
    def result(self):
        from .solvers import TensorParallelCommModel
        return TensorParallelCommModel().solve(
            self.workload, self.tp_size, self.microbatch, self.link_bandwidth, self.compute_time)


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

    def downtime(self, interval: Quantity) -> Quantity:
        """Expected time one component is down in ``interval``."""
        from ..physics.reliability import calc_downtime
        return calc_downtime(self.availability, interval)

    @property
    def downtime_per_year(self) -> Quantity:
        """Expected time one component is down in a 365-day year."""
        from ..core.units import DAYS_PER_YEAR, Q_
        return self.downtime(Q_(DAYS_PER_YEAR, "day")).to("minute")


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
    """A workload of ``intensity`` on ``hardware``'s roofline: attainable rate and share of peak."""
    hardware: HardwareNode
    intensity: Quantity
    precision: Optional[str] = None

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
        return min(self.peak.to("flop/second"), (self.hardware.memory.bandwidth * self.intensity).to("flop/second"))

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
    """Log-linear fit of one group of a Literature series; growth per ``period`` of the x column."""
    series_name: str
    group_key: str
    group: str
    x_key: str
    y_key: str
    period: float = 1.0

    @property
    def points(self) -> List[tuple]:
        from ..literature.registry import Literature
        rows = getattr(Literature.Series, self.series_name).entries
        return [(r[self.x_key], r[self.y_key]) for r in rows if r[self.group_key] == self.group]

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
