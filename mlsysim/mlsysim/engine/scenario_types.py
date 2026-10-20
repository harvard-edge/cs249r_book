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

from typing import Dict, List, Optional

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

    @property
    def result(self):
        from .solvers import DecodeBoundModel
        return DecodeBoundModel().solve(self.workload, self.hardware, self.precision, self.batch_size)


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


# --- Physical-plant budgets of an embodied machine (added 2026-10-20) -------
# Each scenario names one machine at one site (``Embodied.Scenario.*``) or one
# actuator (``Actuators.*``) and states the inputs the site record does not
# carry; the solvers live in ``engine/solvers/plant.py``.

from typing import Any, Tuple  # noqa: E402

class StoppingBudgetScenario(BaseScenario):
    """The base of ``site.machine`` stopping from ``speed`` after ``delay``, against the site's clear distance."""
    site: Any                               # EmbodiedSiteScenario
    speed: Quantity
    delay: Quantity
    deceleration: Optional[Quantity] = None     # default: the site's credible deceleration
    clear_distance: Optional[Quantity] = None   # default: the site's clear distance
    added_delay: Optional[Quantity] = None      # a delay step priced as travel at ``speed``

    @property
    def a_brake(self) -> Quantity:
        return self.site.a_brake if self.deceleration is None else self.deceleration

    @property
    def d_clear(self) -> Quantity:
        return self.site.d_clear if self.clear_distance is None else self.clear_distance

    @property
    def result(self):
        from .solvers import StoppingBudgetModel
        return StoppingBudgetModel().solve(self.speed, self.delay, self.a_brake, self.site.delta_loc,
                                           self.site.delta_margin, self.d_clear, self.added_delay)


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
