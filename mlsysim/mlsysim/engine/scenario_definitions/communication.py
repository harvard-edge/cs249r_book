"""Network topology, transfer costs, and collective communication.

Public compatibility imports remain in ``mlsysim.engine.scenario_types``.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from ...core.types import Quantity
from ...core.units import Q_
from ...hardware.types import HardwareNode
from ...models.types import Workload
from ...systems.types import Fleet
from .base import BaseScenario


class TransferScenario(BaseScenario):
    """``copies`` of one ``payload`` moved at once over each named link tier.

    Optional (added 2026-10-20): ``egress_rate`` prices the bytes leaving a
    provider's network, ``compute_price`` converts that fee into accelerator
    hours, and ``job_time`` is the computation the move would serve, so a
    transfer can be set against the job it feeds (data gravity).

    Notes
    -----
    Equation: Compose ``mlsysim.physics.transfer_time``,
    ``mlsysim.solvers.TransferLadderModel``.

    Assumptions: Payload copies are identical; each named link is priced at its supplied
    constant bandwidth. Optional prices must be populated before cost properties are used.

    Excludes: Protocol framing, connection setup, contention, and volume discounts are
    absent.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.transfer_time``, ``mlsysim.solvers.TransferLadderModel``.
    The methods below identify any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.BufferTransfer10GB
    >>> str(case.total_bytes)
    '10 GB'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    payload: Quantity
    links: Dict[str, Quantity]
    copies: int = 1
    egress_rate: Optional[Quantity] = None  # USD per byte moved out
    compute_price: Optional[Quantity] = None  # USD per accelerator-hour
    job_time: Optional[Quantity] = None

    @property
    def total_bytes(self) -> Quantity:
        """Total payload counted across every copy. Equation: ``copies * payload``. Units: the
        payload's byte unit; protocol headers are excluded.
        """
        return self.copies * self.payload

    def transfer_time(self, link: str) -> Quantity:
        """Bandwidth-only duration for the named link. Equation: ``transfer_time(total_bytes,
        links[link])`` in ``mlsysim.physics.quantities``. Units: time; startup and contention
        are excluded.
        """
        from ...physics import transfer_time

        return transfer_time(self.total_bytes, self.links[link])

    @property
    def egress_cost(self) -> Quantity:
        """Fee for moving every counted byte out of the provider. Equation: ``total_bytes *
        egress_rate``. Units: USD; the optional rate must be supplied.
        """
        return (self.total_bytes * self.egress_rate).to("dollar")

    @property
    def egress_compute_hours(self) -> Quantity:
        """Compute time purchasable for the same egress fee. Equation: ``egress_cost /
        compute_price``. Units: accelerator-hours, requiring a positive hourly price.
        """
        return (self.egress_cost / self.compute_price).to("hour")

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.TransferLadderModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import TransferLadderModel

        return TransferLadderModel().solve(self.total_bytes, self.links)


class TensorParallelScenario(BaseScenario):
    """The activation AllReduces of one tensor-parallel step of ``workload``.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.TensorParallelCommModel``.

    Assumptions: The stated fields are inputs to
    ``mlsysim.solvers.TensorParallelCommModel``; its documented validity regime, units, and
    modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.TensorParallelCommModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.TensorParallelCommModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt3TensorParallelStepDgxH100
    >>> type(case.result).__name__
    'TensorParallelCommResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    tp_size: int
    microbatch: int
    link_bandwidth: Quantity
    compute_time: Quantity  # stated compute per step, not derived
    # Optional (added 2026-10-20): tokens per sequence (default: the model's
    # context length) and AllReduces per layer per step (default: one forward,
    # one backward).
    seq_len: Optional[int] = None
    collectives_per_layer: int = 2

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.TensorParallelCommModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ..solvers import TensorParallelCommModel

        return TensorParallelCommModel().solve(
            self.workload,
            self.tp_size,
            self.microbatch,
            self.link_bandwidth,
            self.compute_time,
            seq_len=self.seq_len,
            collectives_per_layer=self.collectives_per_layer,
        )


class GradientSyncScenario(BaseScenario):
    """A data-parallel step of ``workload``: compute, then a ring AllReduce over each link.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.GradientSyncModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.GradientSyncModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.GradientSyncModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.GradientSyncModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt3GradientSync
    >>> type(case.result).__name__
    'GradientSyncResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    precision: str = "fp16"
    links: Dict[str, Quantity]
    compute_time: Quantity  # stated local compute per step, not derived
    energy_per_bit: Optional[Quantity] = None
    steps: int = 1
    compute_speedup: float = 1.0  # a faster accelerator divides only the compute term

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.GradientSyncModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import GradientSyncModel

        return GradientSyncModel().solve(
            self.workload,
            self.precision,
            self.links,
            self.compute_time,
            self.energy_per_bit,
            self.steps,
            self.compute_speedup,
        )


class DataParallelIntensityScenario(BaseScenario):
    """Communication intensity of one data-parallel worker of ``workload``.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.CommunicationIntensityModel``.

    Assumptions: The stated fields are inputs to
    ``mlsysim.solvers.CommunicationIntensityModel``; its documented validity regime, units,
    and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.CommunicationIntensityModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.CommunicationIntensityModel``. The methods below identify
    any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt3DataParallelIntensity
    >>> type(case.result).__name__
    'CommunicationIntensityResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    sequences_per_worker: int
    link_bandwidth: Quantity
    accelerator: HardwareNode

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.CommunicationIntensityModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ..solvers import CommunicationIntensityModel

        return CommunicationIntensityModel().solve(
            self.workload, self.sequences_per_worker, self.link_bandwidth, self.accelerator.compute.peak_flops
        )


class AlphaBetaScenario(BaseScenario):
    """Named messages on named paths, each path an ``alpha`` startup and a ``beta`` bandwidth.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.AlphaBetaModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.AlphaBetaModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.AlphaBetaModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.AlphaBetaModel``. The methods below identify any additional
    arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.AlphaBetaNdrRoce
    >>> type(case.result).__name__
    'AlphaBetaResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    alphas: Dict[str, Quantity]
    bandwidths: Dict[str, Quantity]
    messages: Dict[str, Quantity]

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.AlphaBetaModel.solve``. Equation: pass
        the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import AlphaBetaModel

        return AlphaBetaModel().solve(self.alphas, self.bandwidths, self.messages)


class LinkBuildoutScenario(BaseScenario):
    """``links_per_device`` links per accelerator of ``fleet``, each with a price and a module
    power.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: Every counted link receives the same unit cost and module-power value;
    links-per-device already handles any shared ownership convention.

    Excludes: Switch chassis, cooling, installation, and network utilization are absent.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.SpineOpticsTraining10K
    >>> round(case.n_links, 4)
    20000
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    fleet: Fleet
    links_per_device: int
    cost_per_link: Quantity
    power_per_link: Quantity

    @property
    def n_links(self) -> int:
        """Total link count across the fleet. Equation: ``links_per_device *
        fleet.total_accelerators``. Units: links; shared-link accounting must already be
        reflected in the per-device input.
        """
        return self.links_per_device * self.fleet.total_accelerators

    @property
    def total_cost(self) -> Quantity:
        """Purchase cost of all counted link modules. Equation: ``n_links * cost_per_link``. Units:
        the input currency; switches and installation are excluded unless priced into each link.
        """
        return self.n_links * self.cost_per_link

    @property
    def total_power(self) -> Quantity:
        """Rated power of all counted link modules. Equation: ``n_links * power_per_link``. Units:
        kW; switching equipment power is excluded.
        """
        return (self.n_links * self.power_per_link).to("kW")


class BandwidthDelayScenario(BaseScenario):
    """Data a link carries during ``delay``: line rate times delay (for example, a Go-Back-N
    resend window).

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: A link transmits continuously at the stated line rate during a fixed delay.

    Excludes: Actual buffer occupancy, packet headers, and loss behavior are absent.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.GoBackNResendNdr
    >>> str(case.product)
    '250.0 KB'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    link_bandwidth: Quantity
    delay: Quantity

    @property
    def product(self) -> Quantity:
        """Bytes transmitted at line rate during the stated delay. Equation: ``link_bandwidth *
        delay``. Units: decimal KB; this is a bandwidth-delay product, not a measured queue
        occupancy.
        """
        return (self.link_bandwidth * self.delay).to("KB")


class OversubscriptionScenario(BaseScenario):
    """A ring AllReduce whose edges all cross the bisection of ``fleet``, at 1:1 and at
    ``oversubscription_ratio``.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.BisectionModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.BisectionModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.BisectionModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.BisectionModel``. The methods below identify any additional
    arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RingAllReduceAcross4to1Spine
    >>> type(case.result).__name__
    'BisectionResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    fleet: Fleet
    link_bandwidth: Quantity
    oversubscription_ratio: float
    payload_per_device: Quantity
    communication_fraction: float
    cluster_cost: Quantity

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.BisectionModel.solve``. Equation: pass
        the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import BisectionModel

        return BisectionModel().solve(
            self.fleet.total_accelerators,
            self.link_bandwidth,
            self.oversubscription_ratio,
            self.payload_per_device,
            self.communication_fraction,
            self.cluster_cost,
        )


class FatTreeScenario(BaseScenario):
    """A non-blocking fat-tree of radix-``radix`` switches (Al-Fares et al. 2008).

    Notes
    -----
    Equation: Compose ``mlsysim.physics.communication.calc_fat_tree_hosts``,
    ``mlsysim.physics.communication.calc_fat_tree_switches``.

    Assumptions: The topology is a fully populated nonblocking fat-tree with an even radix,
    as specified by the communication helpers.

    Excludes: Partial population, oversubscription, faults, and routing imbalance are
    absent.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.communication.calc_fat_tree_hosts``,
    ``mlsysim.physics.communication.calc_fat_tree_switches``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.FatTreeNdrSwitch
    >>> round(case.two_tier_hosts, 4)
    2048
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    radix: int

    def hosts(self, tiers: int) -> int:
        """Host count supported by a full nonblocking fat-tree at a chosen tier count. Equation:
        ``mlsysim.physics.communication.calc_fat_tree_hosts(radix, tiers)``. Units: hosts;
        topology restrictions come from the helper.
        """
        from ...physics.communication import calc_fat_tree_hosts

        return calc_fat_tree_hosts(self.radix, tiers)

    def switches(self, tiers: int) -> int:
        """Switch count for the chosen full fat-tree. Equation:
        ``mlsysim.physics.communication.calc_fat_tree_switches(radix, tiers)``. Units: switches;
        partial population is excluded.
        """
        from ...physics.communication import calc_fat_tree_switches

        return calc_fat_tree_switches(self.radix, tiers)

    @property
    def two_tier_hosts(self) -> int:
        """Host count for the two-tier specialization. Equation: ``hosts(2)``. Units: hosts; uses
        the same radix and nonblocking port allocation.
        """
        return self.hosts(2)

    @property
    def three_tier_hosts(self) -> int:
        """Host count for the three-tier specialization. Equation: ``hosts(3)``. Units: hosts; uses
        the same radix and nonblocking port allocation.
        """
        return self.hosts(3)

    @property
    def three_tier_switches(self) -> int:
        """Switch count for the three-tier specialization. Equation: ``switches(3)``. Units:
        switches; all topology levels are populated.
        """
        return self.switches(3)

    @property
    def pod_hosts(self) -> int:
        """Hosts in one standard fat-tree pod. Equation: ``(radix // 2) ** 2``. Units: hosts; an
        even switch radix is assumed.
        """
        return (self.radix // 2) ** 2


class TorusScenario(BaseScenario):
    """``n_endpoints`` as a cubic torus against a non-blocking fat-tree, at equal per-link
    bandwidth.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.communication.calc_torus_bisection_links``,
    ``mlsysim.physics.communication.calc_torus_diameter``,
    ``mlsysim.physics.communication.calc_fat_tree_path_switches``.

    Assumptions: Endpoints form a regular cubic torus and links have equal bandwidth; side
    length is rounded for non-perfect powers.

    Excludes: The rounded geometry need not preserve endpoint count exactly; failures and
    adaptive-routing congestion are absent.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.communication.calc_torus_bisection_links``,
    ``mlsysim.physics.communication.calc_torus_diameter``,
    ``mlsysim.physics.communication.calc_fat_tree_path_switches``. The methods below
    identify any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.TorusVsFatTree4K
    >>> round(case.side, 4)
    16
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    n_endpoints: int
    link_bandwidth: Quantity
    dims: int = 3
    fat_tree_tiers: int = 3

    @property
    def side(self) -> int:
        """Integer side length used for the cubic-torus approximation. Equation: round the
        ``dims``-th root of ``n_endpoints``. Units: endpoints per side; non-perfect powers are
        rounded rather than rejected.
        """
        return round(self.n_endpoints ** (1 / self.dims))

    @property
    def torus_cut_links(self) -> int:
        """Link count crossing a minimum torus bisection. Equation:
        ``mlsysim.physics.communication.calc_torus_bisection_links(side, dims)``. Units: links
        under the regular-torus geometry.
        """
        from ...physics.communication import calc_torus_bisection_links

        return calc_torus_bisection_links(self.side, self.dims)

    @property
    def fat_tree_cut_links(self) -> int:
        """Links in the comparison's full-bandwidth half-fleet cut. Equation: ``n_endpoints // 2``.
        Units: links; the comparison assumes one equal-rate endpoint link per endpoint.
        """
        return self.n_endpoints // 2

    @property
    def torus_bisection(self) -> Quantity:
        """Aggregate bandwidth of the torus cut. Equation: ``torus_cut_links * link_bandwidth``.
        Units: bandwidth; contention and routing imbalance are excluded.
        """
        return self.torus_cut_links * self.link_bandwidth

    @property
    def fat_tree_bisection(self) -> Quantity:
        """Aggregate bandwidth of the fat-tree comparison cut. Equation: ``fat_tree_cut_links *
        link_bandwidth``. Units: bandwidth under the nonblocking assumption.
        """
        return self.fat_tree_cut_links * self.link_bandwidth

    @property
    def torus_diameter(self) -> int:
        """Longest minimum-hop path in the modeled regular torus. Equation:
        ``mlsysim.physics.communication.calc_torus_diameter(side, dims)``. Units: hops; failed
        links and adaptive detours are excluded.
        """
        from ...physics.communication import calc_torus_diameter

        return calc_torus_diameter(self.side, self.dims)

    @property
    def fat_tree_path_switches(self) -> int:
        """Switch traversals on the comparison's full-height fat-tree path. Equation:
        ``mlsysim.physics.communication.calc_fat_tree_path_switches(fat_tree_tiers)``. Units:
        switches; not a latency measurement.
        """
        from ...physics.communication import calc_fat_tree_path_switches

        return calc_fat_tree_path_switches(self.fat_tree_tiers)


class HopPathScenario(BaseScenario):
    """Named paths through ``hops`` identical switch traversals of ``hop_latency`` each.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.communication.calc_hop_latency``.

    Assumptions: Every switch traversal contributes the same supplied latency, which adds
    along the named path.

    Excludes: Serialization, propagation, and queueing delay are absent.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.communication.calc_hop_latency``. The methods below
    identify any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RailVsLeafSpineHops
    >>> str(next(iter(case.latencies.values())))
    '1.8×10⁻⁶ s'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    hop_latency: Quantity
    hops: Dict[str, int]

    @property
    def latencies(self) -> Dict[str, Quantity]:
        """Switch-delay contribution of each named path. Equation:
        ``mlsysim.physics.communication.calc_hop_latency(hop_count, hop_latency)``. Units: time;
        serialization and queueing are excluded.
        """
        from ...physics.communication import calc_hop_latency

        return {k: calc_hop_latency(n, self.hop_latency) for k, n in self.hops.items()}


class IncastScenario(BaseScenario):
    """``n_senders`` at ``sender_bandwidth`` into one egress port with a ``buffer``.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.IncastModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.IncastModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.IncastModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.IncastModel``. The methods below identify any additional
    arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.IncastSuperPodUnit
    >>> type(case.result).__name__
    'IncastResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    n_senders: int
    sender_bandwidth: Quantity
    egress_bandwidth: Quantity
    buffer: Quantity
    feedback_rtt: Quantity

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.IncastModel.solve``. Equation: pass the
        stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import IncastModel

        return IncastModel().solve(
            self.n_senders, self.sender_bandwidth, self.egress_bandwidth, self.buffer, self.feedback_rtt
        )


class ComponentExposureScenario(BaseScenario):
    """Independent components, each affected with ``probability`` (per day, or at one instant).

    ``counts`` lists the factors whose product is the component count, so the
    breakdown (devices, links per device, modules per link) stays visible.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.reliability.calc_all_up_availability``.

    Assumptions: The multiplicity hierarchy determines total component count; event
    probability has one shared interpretation and interval. Probability-of-any calculations
    require independence.

    Excludes: Correlated events are absent, even though expected-count arithmetic itself
    does not require independence.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.reliability.calc_all_up_availability``. The methods below
    identify any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.TransceiverDegradationRoce4K
    >>> round(case.n_components, 4)
    24576
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    counts: Dict[str, int]
    probability: float

    @property
    def n_components(self) -> int:
        """Total components in the hierarchy. Equation: product of every multiplicity in
        ``counts``. Units: components; nested dimensions are assumed independent counts.
        """
        n = 1
        for c in self.counts.values():
            n *= c
        return n

    @property
    def expected_events(self) -> float:
        """Expected affected-component count in the input probability's interval. Equation:
        ``n_components * probability``. Units: events; expectation alone does not require
        independence.
        """
        return self.n_components * self.probability

    @property
    def p_none(self) -> float:
        """Probability that none of the components is affected. Equation:
        ``calc_all_up_availability(1 - probability, n_components)``. Units: probability;
        independent equal-probability component events are assumed.
        """
        from ...physics.reliability import calc_all_up_availability

        return calc_all_up_availability(1 - self.probability, self.n_components)

    @property
    def p_any(self) -> float:
        """Probability that at least one component is affected. Equation: ``1 - p_none``. Units:
        probability under the same independence assumption.
        """
        return 1 - self.p_none


class SilentSlowdownScenario(BaseScenario):
    """A run on ``fleet`` slowed a few percent by a degraded link, with no alarm.

    ``slowdowns`` are stated per-step slowdowns; ``bandwidth_loss`` and
    ``communication_fraction`` give the no-overlap upper bound, where only the
    communicating share of the step stretches by ``1 / (1 - loss)``.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: Lost fractions allocate the fixed run's billed device-time; the no-overlap
    example keeps computation fixed and stretches only communication.

    Excludes: Root-cause inference and recovery dynamics are absent.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.DegradedTransceiverRun1K
    >>> float(round(case.device_hours, 4))
    737280.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    fleet: Fleet
    run_duration: Quantity
    slowdowns: Dict[str, float]
    device_hour_rate: Quantity
    bandwidth_loss: float
    communication_fraction: float

    @property
    def device_hours(self) -> float:
        """Aggregate billed accelerator-time over the run. Equation: fleet accelerator count times
        run duration. Units: accelerator-hours returned as a scalar.
        """
        return (self.fleet.total_accelerators * self.run_duration).to("hour").magnitude

    @property
    def lost_device_hours(self) -> Dict[str, float]:
        """Allocated accelerator-time lost to each stated slowdown fraction. Equation:
        ``device_hours * fraction`` for each named case. Units: scalar accelerator-hours; this
        is fractional allocation, not a replayed schedule.
        """
        return {k: self.device_hours * f for k, f in self.slowdowns.items()}

    @property
    def lost_cost(self) -> Dict[str, Quantity]:
        """Rental value of each allocated device-time loss. Equation: lost accelerator-hours times
        the hourly price. Units: USD per named case; the tariff is constant.
        """
        rate = self.device_hour_rate.to("USD/hour").magnitude
        return {k: Q_(h * rate, "USD") for k, h in self.lost_device_hours.items()}

    @property
    def no_overlap_slowdown(self) -> float:
        """Relative step-time increase when the communication share sees reduced bandwidth.
        Equation: ``1 - f + f / (1 - bandwidth_loss) - 1``, with ``f=communication_fraction``.
        Units: dimensionless; compute and communication are sequential.
        """
        f = self.communication_fraction
        return (1 - f) + f / (1 - self.bandwidth_loss) - 1


class NicSharingScenario(BaseScenario):
    """One NIC split into ``n_virtual_functions`` rate-capped slices, and a burst at full line
    rate.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: Virtual functions share equal line-rate caps; the separate burst example
    can temporarily use the whole NIC.

    Excludes: Fair-queueing dynamics, protocol overhead, and burst-credit rules are absent.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.SriovNdrNic
    >>> str(case.slice_rate)
    '50.0 Gbps'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    nic_bandwidth: Quantity
    n_virtual_functions: int
    burst_size: Quantity

    @property
    def slice_rate(self) -> Quantity:
        """Equal rate cap per virtual function. Equation: ``nic_bandwidth / n_virtual_functions``.
        Units: bandwidth; arbitration and protocol efficiency are excluded.
        """
        return self.nic_bandwidth / self.n_virtual_functions

    @property
    def burst_time(self) -> Quantity:
        """Serialization duration of a burst using the full NIC line rate. Equation: ``burst_size /
        nic_bandwidth``. Units: seconds; this is not the rate-capped slice duration.
        """
        return (self.burst_size / self.nic_bandwidth).to("second")


class SlowdownDiagnosisScenario(BaseScenario):
    """A training step that slowed, and a pairwise link test that finds one link at a fraction
    of its rate.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: Before/after steps contain the same work, and expected and measured
    bandwidth fractions refer to the same path.

    Excludes: The arithmetic does not identify the cause of a slowdown or collect
    measurements.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.HalfRateLinkDiagnosis
    >>> round(case.slowdown, 4)
    0.1429
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    step_before: Quantity
    step_after: Quantity
    utilization_before: float
    utilization_after: float
    days_since_launch: int
    port_bandwidth: Quantity
    delivered_efficiency: float  # payload share a healthy link test reports
    measured_fraction: float  # the degraded pair's share of the expected rate

    @property
    def slowdown(self) -> float:
        """Relative increase from the healthy step duration. Equation: ``step_after / step_before -
        1``. Units: dimensionless; the compared steps must perform equal work.
        """
        return (self.step_after / self.step_before - 1).to("").magnitude

    @property
    def expected_bandwidth(self) -> Quantity:
        """Expected delivered link rate under the stated efficiency. Equation:
        ``delivered_efficiency * port_bandwidth``. Units: bandwidth; efficiency is an input, not
        inferred by this property.
        """
        return self.delivered_efficiency * self.port_bandwidth

    @property
    def measured_bandwidth(self) -> Quantity:
        """Rate represented by the stated fraction of expected delivered rate. Equation:
        ``expected_bandwidth * measured_fraction``. Units: bandwidth; the measurement fraction
        must come from outside the model.
        """
        return self.expected_bandwidth * self.measured_fraction


class RingAttentionScenario(BaseScenario):
    """Ring attention on ``accelerator``: the smallest chunk that hides each K/V tile, per
    link.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.communication.calc_ring_attention_min_chunk``.

    Assumptions: The selected arithmetic peak is multiplied by a supplied local efficiency,
    and overlap follows the ring-attention helper's tile model.

    Excludes: Kernel scheduling, nonideal overlap, and link contention are absent.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.communication.calc_ring_attention_min_chunk``. The methods
    below identify any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.H100RingAttentionChunk
    >>> round(next(iter(case.min_chunk_tokens.values())), 4)
    1098.8889
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    accelerator: HardwareNode
    precision: str = "fp16_dense"
    local_efficiency: float
    element_bytes: Quantity
    links: Dict[str, Quantity]

    @property
    def min_chunk_tokens(self) -> Dict[str, float]:
        """Smallest chunk whose compute can hide K/V tile transfer on each link. Equation:
        ``mlsysim.physics.communication.calc_ring_attention_min_chunk`` with effective peak
        rate, element bytes, and link bandwidth. Units: tokens per link; the helper's overlap
        assumptions apply.
        """
        from ...physics.communication import calc_ring_attention_min_chunk

        rate = self.accelerator.compute.precision_flops[self.precision] * self.local_efficiency
        elem = self.element_bytes.to("byte").magnitude
        return {k: calc_ring_attention_min_chunk(elem, rate, bw) for k, bw in self.links.items()}


class SyncStrategyScenario(BaseScenario):
    """Data-parallel training to a fixed target under one synchronization strategy.

    Iterations to the target and the communication share are stated inputs:
    each iteration costs the one-worker iteration time times ``1 + overhead``,
    so speedup over one worker is ``base / (iterations x (1 + overhead))``.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: Iterations to target are supplied for each strategy, not predicted;
    communication stretches a fixed base iteration duration.

    Excludes: The inverse-square-root staleness learning-rate factor is an illustrative
    policy, not a convergence theorem.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.TransformerBsp8Workers
    >>> round(case.global_batch, 4)
    256
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workers: int
    per_worker_batch: int
    iterations: int
    communication_overhead: float
    base_iterations: int  # one worker, same target
    base_iteration_time: Quantity
    device_hour_rate: Quantity
    staleness: int = 0  # SSP bound; 0 is BSP

    @property
    def global_batch(self) -> int:
        """Samples processed in one global iteration. Equation: ``workers * per_worker_batch``.
        Units: samples; every worker contributes a full local batch.
        """
        return self.workers * self.per_worker_batch

    @property
    def ideal_iterations(self) -> float:
        """Iteration count under perfect sample-efficiency scaling. Equation: ``base_iterations /
        workers``. Units: iterations; no quality degradation is assumed for this reference.
        """
        return self.base_iterations / self.workers

    @property
    def iteration_time(self) -> Quantity:
        """Per-iteration time after the stated communication overhead. Equation:
        ``base_iteration_time * (1 + communication_overhead)``. Units: time; overhead is
        relative to base iteration time.
        """
        return self.base_iteration_time * (1 + self.communication_overhead)

    @property
    def speedup(self) -> float:
        """Time-to-target speedup against the single-worker baseline. Equation: ``base_iterations /
        (iterations * (1 + communication_overhead))``. Units: dimensionless; both runs use the
        same base iteration duration.
        """
        return self.base_iterations / (self.iterations * (1 + self.communication_overhead))

    @property
    def sample_efficiency(self) -> float:
        """Ideal iteration count divided by actual target-reaching iteration count. Equation:
        ``ideal_iterations / iterations``. Units: dimensionless; target-reaching iterations are
        supplied, not predicted.
        """
        return self.ideal_iterations / self.iterations

    @property
    def cost(self) -> float:
        """Rental bill for the modeled run. Equation: iterations times iteration duration times
        worker count times device-hour price. Units: USD returned as a scalar; workers are
        billed throughout the run.
        """
        hours = (self.iterations * self.iteration_time * self.workers).to("hour")
        return (hours * self.device_hour_rate).to("USD").magnitude

    @property
    def stale_lr_factor(self) -> float:
        """Illustrative learning-rate reduction for stale gradients. Equation: ``1 / sqrt(1 +
        staleness)``. Units: dimensionless; this prescribed heuristic is not a convergence
        guarantee.
        """
        return 1 / (1 + self.staleness) ** 0.5


class AllReduceScheduleScenario(BaseScenario):
    """One buffer reduced by ring, simple tree, and parameter server at each of
    ``group_sizes``.

    ``run_steps`` and ``time_saved_fraction`` price a what-if: removing that
    share of the ring time on every step of a run.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.AllReduceAlgorithmModel``.

    Assumptions: Named algorithm times come from AllReduceAlgorithmModel; saved-time
    fractions are hypothetical and apply on every step.

    Excludes: Application compute overlap and measurement-based validation of the assumed
    savings are absent.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.AllReduceAlgorithmModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.DdpBucketRingNdr
    >>> type(case.result).__name__
    'AllReduceAlgorithmResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    message: Quantity
    group_sizes: List[int]
    alpha: Quantity
    bandwidth: Quantity
    run_steps: int = 1
    time_saved_fraction: float = 0.0

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.AllReduceAlgorithmModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ..solvers import AllReduceAlgorithmModel

        return AllReduceAlgorithmModel().solve(self.message, self.group_sizes, self.alpha, self.bandwidth)

    def saved_time(self, n: int) -> Quantity:
        """Per-step ring time saved at a stated fractional improvement. Equation:
        ``result.ring_time[n] * time_saved_fraction``. Units: seconds; the improvement fraction
        is assumed.
        """
        return (self.result.ring_time[n] * self.time_saved_fraction).to("second")

    def saved_device_hours(self, n: int) -> float:
        """Aggregate device-time saved across the run. Equation: ``saved_time(n) * run_steps * n``.
        Units: scalar device-hours; every participating device is credited the same saved
        duration.
        """
        return (self.saved_time(n) * self.run_steps * n).to("hour").magnitude


class MessageFusionScenario(BaseScenario):
    """``count`` small messages sent one by one against one fused message on the same path.

    Each small message is far below ``n* = alpha beta``, so separately each
    costs its startup alone (the byte term is dropped); fused, the batch pays
    one startup plus its bytes.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: The separate-message baseline is startup-dominated and neglects its payload
    serialization, while the fused path includes serialization once.

    Excludes: Packing overhead and queueing are absent; the comparison becomes inaccurate
    when individual transfer time is not small relative to startup.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RoutingMessageFusionNdr
    >>> print(f"{case.crossover.m_as('kilobyte'):.1f} kB")
    75.0 kB
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    alpha: Quantity
    bandwidth: Quantity
    message: Quantity
    count: int

    @property
    def crossover(self) -> Quantity:
        """Message size at which serialization equals one startup delay. Equation: ``alpha *
        bandwidth``. Units: bytes; bandwidth and startup are constant.
        """
        return self.alpha * self.bandwidth

    @property
    def fused_message(self) -> Quantity:
        """Combined payload from all small messages. Equation: ``count * message``. Units: bytes;
        fusion metadata is excluded.
        """
        return self.count * self.message

    @property
    def separate_time(self) -> Quantity:
        """Startup-only cost of sending the small messages separately. Equation: ``count * alpha``.
        Units: time; per-message serialization is intentionally omitted in this small-message
        approximation.
        """
        return self.count * self.alpha

    @property
    def fused_time(self) -> Quantity:
        """One fused startup plus combined payload serialization. Equation: ``alpha + fused_message
        / bandwidth``. Units: time; packing overhead is excluded.
        """
        return self.alpha + self.fused_message / self.bandwidth

    @property
    def speedup(self) -> float:
        """Startup-dominated separate-send duration relative to fused duration. Equation:
        ``separate_time / fused_time``. Units: dimensionless; it inherits the asymmetric
        serialization approximation.
        """
        return (self.separate_time / self.fused_time).to_base_units().magnitude

    @property
    def transfer_time(self) -> Quantity:
        """Serialization duration of one unfused message. Equation: ``message / bandwidth``. Units:
        nanoseconds; startup is excluded.
        """
        return (self.message / self.bandwidth).to("nanosecond")

    @property
    def startup_to_transfer(self) -> float:
        """Startup delay measured in one-message serialization durations. Equation: ``alpha /
        transfer_time``. Units: dimensionless; larger values justify the startup-dominated
        approximation.
        """
        return (self.alpha / self.transfer_time).to("").magnitude


class HierarchicalAllReduceScenario(BaseScenario):
    """A buffer AllReduced over ``n_nodes`` nodes of ``gpus_per_node`` GPUs, in three phases or
    one flat ring.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.HierarchicalAllReduceModel``.

    Assumptions: Every node has the same GPU count and the hierarchy's bandwidth/startup
    values are fixed.

    Excludes: Topology placement and dynamic routing beyond the solver's three-phase
    comparison are absent.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.HierarchicalAllReduceModel``. The methods below identify
    any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.HierarchicalAllReduce8x8DgxH100
    >>> round(case.n_devices, 4)
    64
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    message: Quantity
    n_nodes: int
    gpus_per_node: int
    intra_bandwidth: Quantity  # per direction, inside the node
    inter_bandwidth: Quantity  # one port per GPU, across nodes
    inter_alpha: Quantity

    @property
    def n_devices(self) -> int:
        """Accelerator count across all nodes. Equation: ``n_nodes * gpus_per_node``. Units:
        devices; node populations are identical.
        """
        return self.n_nodes * self.gpus_per_node

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.HierarchicalAllReduceModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ..solvers import HierarchicalAllReduceModel

        return HierarchicalAllReduceModel().solve(
            self.message, self.n_nodes, self.gpus_per_node, self.intra_bandwidth, self.inter_bandwidth, self.inter_alpha
        )


class AllToAllDispatchScenario(BaseScenario):
    """One expert-parallel dispatch: ``total_tokens`` of ``hidden_dim`` values over
    ``n_devices`` holding ``n_experts``.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.AllToAllDispatchModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.AllToAllDispatchModel``;
    its documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.AllToAllDispatchModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.AllToAllDispatchModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.MoeDispatch8Gpu64Experts
    >>> type(case.result).__name__
    'AllToAllDispatchResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    total_tokens: int
    n_devices: int
    n_experts: int
    hidden_dim: int
    bytes_per_element: Quantity
    alpha: Quantity
    bandwidth: Quantity
    receive_skew: float = 1.0  # remote tokens at the hottest receiver over the uniform load

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.AllToAllDispatchModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ..solvers import AllToAllDispatchModel

        return AllToAllDispatchModel().solve(
            self.total_tokens,
            self.n_devices,
            self.n_experts,
            self.hidden_dim,
            self.bytes_per_element,
            self.alpha,
            self.bandwidth,
            self.receive_skew,
        )


class RingTraceScenario(BaseScenario):
    """A ring AllReduce traced step by step on one short integer vector per GPU.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.RingTraceModel``.

    Assumptions: One equal-shaped integer vector is supplied per participant and the trace
    follows the solver's ring schedule.

    Excludes: The trace illustrates data movement; it does not predict kernel or network
    duration.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.RingTraceModel``. The methods below identify any additional
    arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RingTraceFourGpu
    >>> round(case.n_devices, 4)
    4
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    vectors: List[List[int]]

    @property
    def n_devices(self) -> int:
        """Participants represented by the supplied vectors. Equation: ``len(vectors)``. Units:
        devices; one vector is supplied per participant.
        """
        return len(self.vectors)

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.RingTraceModel.solve``. Equation: pass
        the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import RingTraceModel

        return RingTraceModel().solve(self.vectors)


class ErrorFeedbackScenario(BaseScenario):
    """A rounding compressor applied to a gradient sequence, with and without error feedback.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.ErrorFeedbackModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.ErrorFeedbackModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.ErrorFeedbackModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.ErrorFeedbackModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ErrorFeedbackFiveSteps
    >>> type(case.result).__name__
    'ErrorFeedbackResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    gradients: List[float]

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.ErrorFeedbackModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import ErrorFeedbackModel

        return ErrorFeedbackModel().solve(self.gradients)


class CompressionPaybackScenario(BaseScenario):
    """One synchronization compressed by ``compression_ratio`` at a codec cost of
    ``codec_time``.

    Compression pays when the codec costs less than the transfer it removes,
    ``codec_time < T_comm (1 - 1/r)``; ``fast_communication_time`` is the same
    reduction on a fast link, where it does not.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: Compression divides communicated bytes by a positive ratio, leaves
    bandwidth fixed, and incurs a sequential codec cost.

    Excludes: Gradient-quality effects and codec/communication overlap are absent.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.CompressionPayback8x
    >>> str(case.compressed_time)
    '5.0 ms'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    communication_time: Quantity
    compression_ratio: float
    codec_time: Quantity
    fast_communication_time: Quantity

    @property
    def compressed_time(self) -> Quantity:
        """Communication duration after ideal payload compression. Equation: ``communication_time /
        compression_ratio``. Units: time; compression ratio divides communicated bytes and
        leaves bandwidth fixed.
        """
        return self.communication_time / self.compression_ratio

    @property
    def total_time(self) -> Quantity:
        """Sequential codec plus compressed communication duration. Equation: ``compressed_time +
        codec_time``. Units: time; codec/communication overlap is excluded.
        """
        return self.compressed_time + self.codec_time

    @property
    def speedup(self) -> float:
        """Uncompressed communication duration relative to codec-plus-transfer time. Equation:
        ``communication_time / total_time``. Units: dimensionless; compute outside the
        collective is excluded.
        """
        return (self.communication_time / self.total_time).to_base_units().magnitude

    @property
    def saved_time(self) -> Quantity:
        """Communication-only duration removed by compression. Equation: ``communication_time -
        compressed_time``. Units: time; codec cost has not yet been deducted.
        """
        return self.communication_time - self.compressed_time

    def pays(self, communication_time: Quantity) -> bool:
        """Whether codec overhead is less than transfer time saved for a proposed communication
        duration. Equation: ``codec_time < communication_time * (1 - 1 / compression_ratio)``.
        Units: boolean; equality is not counted as a gain.
        """
        return self.codec_time < communication_time * (1 - 1 / self.compression_ratio)


class CompressionRunScenario(BaseScenario):
    """A run to a quality target with and without gradient compression on several networks.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.CompressionRunModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.CompressionRunModel``;
    its documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.CompressionRunModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.CompressionRunModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Int8CompressionRun
    >>> type(case.result).__name__
    'CompressionRunResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    steps: int
    compute_time: Quantity
    communication_times: Dict[str, Quantity]
    compression_ratio: float
    step_penalty: float  # extra steps compression noise costs, as a fraction

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.CompressionRunModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import CompressionRunModel

        return CompressionRunModel().solve(
            self.steps, self.compute_time, self.communication_times, self.compression_ratio, self.step_penalty
        )


class OverlapBudgetScenario(BaseScenario):
    """Layer-by-layer gradient AllReduce of ``workload``: sequential against overlapped with
    backward.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.OverlapBudgetModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.OverlapBudgetModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.OverlapBudgetModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.OverlapBudgetModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.OverlapBudget7B64Gpu
    >>> type(case.result).__name__
    'OverlapBudgetResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    bytes_per_value: Quantity
    backward_per_layer: Quantity  # stated, not derived
    bucket_size: Quantity
    time_per_bucket: Quantity
    n_devices: int

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.OverlapBudgetModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import OverlapBudgetModel

        return OverlapBudgetModel().solve(
            self.workload, self.bytes_per_value, self.backward_per_layer, self.bucket_size, self.time_per_bucket
        )


class ResidualShareScenario(BaseScenario):
    """A residual equal to ``residual`` of the one-device step, on ``devices`` devices.

    Compute divides by N and the residual does not, so the residual's share of
    the distributed step, ``r / ((1 - r)/N + r)``, is far larger than ``r``:
    an observed share must not be read as the serial fraction.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.performance.calc_strong_scaling_speedup``.

    Assumptions: The residual is a fixed fraction of the one-device step; all remaining work
    scales inversely with device count.

    Excludes: Communication or synchronization overhead that grows with fleet size is
    absent.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.performance.calc_strong_scaling_speedup``. The methods
    below identify any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ResidualOnePercent512
    >>> round(case.step_fraction, 4)
    0.0119
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    residual: float
    devices: int

    @property
    def step_fraction(self) -> float:
        """Scaled step duration relative to one device. Equation: ``(1 - residual) / devices +
        residual``. Units: dimensionless; the residual is fixed while remaining work scales
        ideally.
        """
        return (1 - self.residual) / self.devices + self.residual

    @property
    def residual_share(self) -> float:
        """Residual's fraction of the shortened distributed step. Equation: ``residual /
        step_fraction``. Units: dimensionless; the original residual fraction is not the scaled
        fraction.
        """
        return self.residual / self.step_fraction

    @property
    def speedup(self) -> float:
        """Strong-scaling speedup with a fixed serial residual. Equation:
        ``mlsysim.physics.performance.calc_strong_scaling_speedup(devices, residual)``. Units:
        dimensionless; communication growth is excluded.
        """
        from ...physics.performance import calc_strong_scaling_speedup

        return calc_strong_scaling_speedup(self.devices, self.residual)


class PropagationScenario(BaseScenario):
    """A request sent ``distance`` over fiber and answered; optionally a budget and a measured
    round trip.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.networking.calc_network_latency_ms``.

    Assumptions: Distance is one-way fiber distance and the helper supplies its propagation
    convention; measured latency is an external optional input.

    Excludes: Routing stretch, queueing, switching, and retransmission are not part of the
    propagation floor.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.networking.calc_network_latency_ms``. The methods below
    identify any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RoboticArmRegionalCloud
    >>> str(case.round_trip)
    '15.0 ms'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    distance: Quantity
    budget: Optional[Quantity] = None
    measured_round_trip: Optional[Quantity] = None

    @property
    def round_trip(self) -> Quantity:
        """Fiber propagation lower bound for the stated one-way distance. Equation:
        ``mlsysim.physics.networking.calc_network_latency_ms(distance)``. Units: milliseconds;
        routing stretch, switching, and queueing are excluded.
        """
        from ...physics.networking import calc_network_latency_ms

        return Q_(calc_network_latency_ms(self.distance), "millisecond")

    @property
    def overrun(self) -> Quantity:
        """Amount by which propagation alone exceeds the latency budget. Equation: ``round_trip -
        budget``. Units: milliseconds; a negative value means remaining budget, not negative
        latency.
        """
        return (self.round_trip - self.budget).to("millisecond")

    @property
    def measured_over_floor(self) -> float:
        """Observed round-trip latency relative to the modeled propagation floor. Equation:
        ``measured_round_trip / round_trip``. Units: dimensionless; the observed value is an
        external input.
        """
        return (self.measured_round_trip / self.round_trip).to("").magnitude


class SensorUplinkScenario(BaseScenario):
    """``streams`` raw video streams against a shared link, and the event records that replace
    them.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: Raw video streams have equal shape and frame rate; event payload size and
    rate are supplied separately.

    Excludes: The model does not claim that event-only data preserve all raw-video
    information or detection quality.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.FactoryCameras1080p
    >>> str(case.stream_rate)
    '186624000.0 B/s'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
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
        """Raw byte rate of one uncompressed video stream. Equation: width times height times bytes
        per pixel times frame rate. Units: bytes/s; protocol framing and compression are
        excluded.
        """
        return (self.frame_width * self.frame_height * self.bytes_per_pixel * self.frame_rate).to("byte/second")

    @property
    def total_rate(self) -> Quantity:
        """Aggregate raw video uplink rate. Equation: ``streams * stream_rate``. Units: bytes/s;
        all streams operate continuously at the stated frame rate.
        """
        return (self.streams * self.stream_rate).to("byte/second")

    @property
    def link_shortfall(self) -> float:
        """Required raw-stream bandwidth relative to link capacity. Equation: ``total_rate /
        link_bandwidth``. Units: dimensionless; values above one exceed the line-rate budget.
        """
        return (self.total_rate / self.link_bandwidth).to("").magnitude

    @property
    def event_bandwidth(self) -> Quantity:
        """Uplink rate when sending only the stated event records. Equation: ``event_size *
        event_rate``. Units: bytes/s; event detection quality and additional metadata are
        excluded.
        """
        return (self.event_size * self.event_rate).to("byte/second")

    @property
    def event_reduction(self) -> float:
        """Raw video rate divided by event-record rate. Equation: ``total_rate / event_bandwidth``.
        Units: dimensionless; the two representations need not preserve equal information.
        """
        return (self.total_rate / self.event_bandwidth).to("").magnitude
