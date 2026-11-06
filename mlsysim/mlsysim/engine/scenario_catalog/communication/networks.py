"""Named communication scenarios for networks.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....systems.registry import Clusters as _Clusters
from ....reference_stats.registry import ReferenceStats as _ReferenceStats
from ....systems.registry import Fabrics as _Fabrics
from ....systems.registry import SwitchFabric as _SwitchFabric
from ....infrastructure.registry import Infrastructure as _Infrastructure
from ... import scenario_types as _st


SpineOpticsTraining10K = _st.LinkBuildoutScenario(
    name="Spine-layer optics for a 10,000-GPU cluster",
    description="Two optical spine links per GPU, each priced as one optical link and powered by one "
    "selected 400G pluggable module (the opposite end excluded).",
    fleet=_Clusters.Training_10K,
    links_per_device=2,  # scenario assumption: spine-layer optical links per accelerator
    cost_per_link=_Infrastructure.Pricing.Network.OpticsPerLinkUsd.rate,
    power_per_link=_SwitchFabric.OpticsPluggable400G,
)


GoBackNResendNdr = _st.BandwidthDelayScenario(
    name="Go-Back-N resend window on an NDR link",
    description="Data resent after one drop: line rate times a 5 microsecond loss-detection round trip.",
    link_bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
    delay=Q_(5, "microsecond"),
)  # scenario assumption: feedback round trip


AlphaBetaNdrRoce = _st.AlphaBetaScenario(
    name="A control message and a gradient shard on NDR and RoCE paths",
    description="Modeled alpha-beta paths: InfiniBand NDR, RoCE at 100 Gb/s, and RoCE's startup on a "
    "400 Gb/s link; a 4,000-byte control message and a 350 MB gradient shard.",
    alphas={"NDR": _SwitchFabric.AlphaNdr, "RoCE 100G": _SwitchFabric.AlphaRoce, "RoCE 400G": _SwitchFabric.AlphaRoce},
    bandwidths={
        "NDR": _Fabrics.InfiniBand_NDR.bandwidth,
        "RoCE 100G": _Fabrics.Ethernet_100G.bandwidth,
        "RoCE 400G": _Fabrics.RoCE_400G.bandwidth,
    },
    messages={
        "control": Q_(4000, "byte"),  # scenario assumption: pipeline control message
        "shard": Q_(350, "MB"),
    },
)  # scenario assumption: one rank's gradient shard, rounded


AlphaBetaHdrNdrSmallMessage = _st.AlphaBetaScenario(
    name="A 10 KB message on HDR and NDR InfiniBand",
    description="The NDR startup latency on both links; the link rate doubles from HDR to NDR.",
    alphas={"HDR": _SwitchFabric.AlphaNdr, "NDR": _SwitchFabric.AlphaNdr},
    bandwidths={"HDR": _Fabrics.InfiniBand_HDR.bandwidth, "NDR": _Fabrics.InfiniBand_NDR.bandwidth},
    messages={"small": Q_(10, "KB")},
)  # scenario assumption: small control packet


RingAllReduceAcross4to1Spine = _st.OversubscriptionScenario(
    name="Ring AllReduce across a 4:1 spine on 1,024 GPUs",
    description="Ranks alternate between the halves, so every ring edge crosses the cut; 5 GB per GPU, "
    "30 percent of the step communicating, a $300M cluster. The worst-case placement.",
    fleet=_Clusters.Training_1K,
    link_bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
    oversubscription_ratio=_Fabrics.InfiniBand_NDR_Oversub4to1.oversubscription_ratio,
    payload_per_device=_ReferenceStats.NetworkEconomicsProfile.PerGpuGradientPayload,
    communication_fraction=_ReferenceStats.NetworkEconomicsProfile.CommunicationFraction,
    cluster_cost=_ReferenceStats.NetworkEconomicsProfile.ClusterCapex,
)


RingAllReduceAcross2to1Spine = RingAllReduceAcross4to1Spine.variant(
    name="Ring AllReduce across a 2:1 spine on 1,024 GPUs",
    oversubscription_ratio=_Fabrics.InfiniBand_NDR_Oversub2to1.oversubscription_ratio,
)


FatTreeNdrSwitch = _st.FatTreeScenario(
    name="Fat-tree of 64-port NDR switches",
    description="Quantum-2 class radix: two tiers reach k^2/2 hosts, three reach k^3/4.",
    radix=int(_SwitchFabric.NdrSwitchPorts),
)


TorusVsFatTree4K = _st.TorusScenario(
    name="4,096 endpoints as a 3D torus and as a fat-tree",
    description="Four 1,024-host pods of the 64-port fat-tree (twice one standalone two-tier fabric), "
    "against a 16 x 16 x 16 torus at the same per-link NDR rate.",
    n_endpoints=2 * FatTreeNdrSwitch.two_tier_hosts,  # doubling past one standalone two-tier fabric
    link_bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
)  # scenario assumption: equal per-link capacity


RailVsLeafSpineHops = _st.HopPathScenario(
    name="Same-rank traffic through a rail switch and through leaf-spine-leaf",
    description="One rail switch against source leaf, spine, and destination leaf, at the per-hop latency.",
    hop_latency=_SwitchFabric.HopLatency,
    hops={"leaf-spine-leaf": 3, "rail": 1},
)


IncastSuperPodUnit = _st.IncastScenario(
    name="One SuperPOD scalable unit into one 400 Gb/s egress",
    description="Deliberate worst case: 256 NDR senders converge on one egress port with a 32 MB buffer.",
    n_senders=_Clusters.DgxH100SuperPodSU.total_accelerators,
    sender_bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
    egress_bandwidth=_Fabrics.Ethernet_400G.bandwidth,
    buffer=Q_(32, "MB"),  # scenario assumption: switch egress buffer
    feedback_rtt=Q_(5, "microsecond"),
)  # scenario assumption: congestion-feedback round trip


TransceiverDegradationRoce4K = _st.ComponentExposureScenario(
    name="Daily transceiver degradation in a 4,096-GPU RoCE cluster",
    description="Leaf, spine, and core links per GPU, an optical module at each end, each starting "
    "to degrade with probability 0.001 percent per day.",
    counts={
        "devices": 4096,  # scenario assumption: large RoCE cluster
        "links_per_device": 3,  # scenario assumption: leaf, spine, and core tiers
        "modules_per_link": 2,
    },  # one module at each end of an optical link
    probability=1e-5,
)  # scenario assumption: per module per day


DegradedLinksReferenceFabric = _st.ComponentExposureScenario(
    name="Degraded links at one instant in a 3,000-link fabric",
    description="About 3,000 high-speed links in the 1,024-GPU fabric, each degraded with probability 0.1 percent.",
    counts={"links": 3000},  # scenario assumption: links that sustain full bisection
    probability=0.001,
)  # scenario assumption: point-in-time probability per link


DegradedTransceiverRun1K = _st.SilentSlowdownScenario(
    name="A degraded transceiver during a 30-day run on 1,024 GPUs",
    description="A 10 percent bandwidth loss that slows each step 2 to 3 percent, priced at the "
    "fleet GPU-hour rate; no alarm fires.",
    fleet=_Clusters.Training_1K,
    run_duration=Q_(30, "day"),  # scenario assumption: training-run duration
    slowdowns={"low": 0.02, "high": 0.03},  # scenario assumption: per-step slowdown after overlap
    device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
    bandwidth_loss=0.10,  # scenario assumption: bandwidth lost by the degraded link
    communication_fraction=_ReferenceStats.NetworkEconomicsProfile.CommunicationFraction,
)


SriovNdrNic = _st.NicSharingScenario(
    name="An NDR NIC split into eight virtual functions",
    description="SR-IOV slices with a rate cap each, and a neighbor's 100 GB checkpoint at line rate.",
    nic_bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
    n_virtual_functions=8,  # scenario assumption
    burst_size=Q_(100, "GB"),
)  # scenario assumption: neighboring job's checkpoint


HalfRateLinkDiagnosis = _st.SlowdownDiagnosisScenario(
    name="A training step that slowed on day 3, traced to one link at half rate",
    description="Step time and utilization before and after; a pairwise RDMA test reports one pair at "
    "half the healthy rate.",
    step_before=Q_(4.2, "second"),
    step_after=Q_(4.8, "second"),  # scenario assumption
    utilization_before=0.92,
    utilization_after=0.85,  # scenario assumption: nvidia-smi
    days_since_launch=3,  # scenario assumption
    port_bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
    delivered_efficiency=0.96,  # scenario assumption: ib_write_bw payload share on a healthy link
    measured_fraction=0.5,
)  # scenario assumption: lanes or rate negotiated down by half
