"""Named robotics scenarios for demonstrations.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....systems.registry import Nodes as _Nodes
from ....systems.registry import Storage as _Storage
from ....literature.registry import Literature as _Literature
from ....embodied.registry import Embodied as _Embodied
from ....agents.registry import Agents as _Agents
from ... import scenario_types as _st
from ....sensors.registry import Sensors as _Sensors
from ....physics.constants import AGE_OF_UNIVERSE as _AGE_OF_UNIVERSE
from ..serving.kv_cache import Llama3_70BLongContextAdmissionDgxH100, RepairRequestZones


# --- Physical demonstration data (added 2026-10-20) ---
PermissionTickTelemetryHour = _st.CorrelatedLogScenario(
    name="One hour of telemetry logged at the warehouse manipulator's permission tick",
    description="A slowly varying statistic with lag-one correlation 0.999: records against "
    "effective independent samples.",
    rate=_Embodied.MobileManipulator.WarehouseMobileManipulator.permission_rate,
    duration=Q_(1, "hour"),  # scenario assumption
    lag1_correlation=0.999,  # scenario assumption: slowly varying statistic
)


ImitationBoundsFiveSecondEpisode = _st.ImitationBoundScenario(
    name="Imitation cost bounds over a 5 s episode at 100 Hz",
    description="Open loop, behavioral cloning (S^2 eps), and aggregation (u S eps, u = 1) at a "
    "per-step error of 0.001.",
    episode=Q_(5, "second"),  # scenario assumption
    control_rate=Q_(100, "Hz"),  # scenario assumption
    per_step_error=1.0e-3,  # scenario assumption
    recoverability=1.0,  # scenario assumption: a recoverable task
)


LatchPairingLag = _st.PairingLagScenario(
    name="An uncompensated 35 ms observation-action lag at the cage-door latch",
    description="Pairing error at a transit speed and the two latch approaches against a 2 mm "
    "clearance; full-frame rolling-shutter skew beside it.",
    site=_Embodied.Scenario.WarehouseAisle,
    lag=Q_(35, "ms"),  # scenario assumption: exposure-to-arrival lag
    clearance=Q_(2, "mm"),  # scenario assumption: latch clearance
    transit_speed=Q_(0.50, "m/s"),  # scenario assumption: generic arm transit speed
    shutter_skew_low=Q_(15, "ms"),  # scenario assumption: fast full-frame readout
    shutter_skew_high=Q_(33, "ms"),  # scenario assumption: one 30 Hz frame
)


LatchPairingLagTightPlate = LatchPairingLag.variant(
    name="The same lag against a tighter 1 mm strike-plate clearance",
    clearance=Q_(1, "mm"),  # scenario assumption
)


TeleopCorrectionLoop = _st.TeleopLoopScenario(
    name="One teleoperated correction in an illustrative reference setup",
    description="Display, operator reaction, command transport, and mechanical response.",
    display=Q_(50, "ms"),  # scenario assumption: exposure, decode, raster refresh
    human=Q_(200, "ms"),  # scenario assumption: perception, planning, neuromuscular
    command=Q_(10, "ms"),  # scenario assumption: quantization, packetization, bus
    mechanical=Q_(80, "ms"),  # scenario assumption: torque ramp and structural inertia
)


WarehouseTeleopIngestion = _st.TeleopIngestionScenario(
    name="Raw telemetry of one teleoperated session on the warehouse mobile manipulator",
    description="Navigation and wrist cameras, two external RGB-D cameras, and the kinematic record "
    "at the permission tick, against SATA and Gen4 NVMe write bandwidth.",
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    external_camera=_Sensors.Camera.Intel_RealSense_D435i,
    external_cameras=2,  # scenario assumption
    rgb_bytes_per_pixel=3,  # scenario assumption: 24-bit RGB after debayering
    depth_bytes_per_pixel=2,  # scenario assumption: uint16 depth aligned to color
    values_per_axis=6,  # scenario assumption: position, velocity, torque, and three action taps
    bytes_per_value=4,  # scenario assumption: float32
    timestamp_bytes=8,  # scenario assumption: one 64-bit timestamp per record
    shift=Q_(8, "hour"),  # scenario assumption
    compression_ratio=8.0,  # scenario assumption: intra-frame video codec
    slow_drive_bandwidth=_Storage.SataSsd.bandwidth,
    fast_drive_bandwidth=_Hardware.Tech.Storage.NvmeGen4.bandwidth,
)


LatchDemonstrationCampaign = _st.DemonstrationCampaignScenario(
    name="Door-opening demonstrations on the cage-door latch, an illustrative campaign",
    description="One scheduled hour of attempt-and-reset cycles, the campaign that keeps 5,000 "
    "demonstrations, and the wear on the wrist gear and flex cable.",
    session=Q_(60, "minute"),  # scenario assumption: declared session length
    startup=Q_(8, "minute"),  # scenario assumption: cell startup and bus sync
    calibration=Q_(7, "minute"),  # scenario assumption: sensor calibration and nulling
    rest=Q_(10, "minute"),  # scenario assumption: operator rest
    attempt=Q_(25, "second"),  # scenario assumption
    reset=Q_(15, "second"),  # scenario assumption: re-latch the door
    failure_rate=0.20,  # scenario assumption
    qa_rate=0.15,  # scenario assumption
    target_kept=5000,  # scenario assumption: campaign target
    gear_reversals_per_attempt=14,  # scenario assumption
    cable_flexes_opening=6,  # scenario assumption
    cable_flexes_return=2,  # scenario assumption: return stroke during the re-latch
    gear_rated_cycles=5_000_000,  # scenario assumption: rated stress cycles
    cable_rated_cycles=200_000,  # scenario assumption: rated bending cycles
    repeat_campaigns=3,  # scenario assumption: campaigns compared against the cable's life
)


WarehouseManipulatorStateGrid = _st.StateGridScenario(
    name="Brute-force coverage of a 28-dimensional manipulation state grid",
    description="Cells at 10 and 100 bins per axis against a 1,000-robot fleet logging at 1 kHz "
    "for the age of the universe.",
    state_components={  # scenario assumption: the state a contact task conditions on
        "joint angles": 7,
        "joint velocities": 7,
        "wheel states": 2,
        "wrist wrench": 6,
        "workpiece pose": 6,
    },
    bins=10,  # scenario assumption
    fine_bins=100,  # scenario assumption: fine enough to guide contact
    fleet=1000,  # scenario assumption
    log_rate=Q_(1000, "Hz"),  # scenario assumption
    duration=_AGE_OF_UNIVERSE,
)


ContactAngleCoverage = _st.ContactGridScenario(
    name="Two archives of one million records on a contact-angle and normal-force grid",
    description="A cautious teleoperator and an exploratory policy against 1 degree by 1 N bins.",
    angle_min=Q_(-15, "degree"),  # scenario assumption
    angle_max=Q_(15, "degree"),  # scenario assumption
    angle_bin=Q_(1, "degree"),  # scenario assumption
    force_max=Q_(20, "N"),  # scenario assumption
    force_bin=Q_(1, "N"),  # scenario assumption
    records=1_000_000,  # scenario assumption
    record_rate=Q_(100, "Hz"),  # scenario assumption
    archives={  # scenario assumption: occupied ranges and stipulated errors
        "cautious": (Q_(-2, "degree"), Q_(2, "degree"), Q_(4, "N"), Q_(6, "N"), Q_(0.001, "rad")),
        "exploratory": (Q_(-15, "degree"), Q_(15, "degree"), Q_(1, "N"), Q_(18, "N"), Q_(0.012, "rad")),
    },
)


LatchScenarioTable = _st.ScenarioTableCoverageScenario(
    name="Cage-door latch archive against plate offset and approach speed",
    description="Three plate offsets by the two latch approaches; the archive holds the nominal "
    "plate at the guarded speed.",
    offsets=[Q_(0, "mm"), Q_(3, "mm"), Q_(-3, "mm")],  # scenario assumption: +/-3 mm plate offset
    speeds=[_Embodied.Scenario.WarehouseAisle.v_latch_approach, _Embodied.Scenario.WarehouseAisle.v_latch_high],
    logged_offset=Q_(0.5, "mm"),  # scenario assumption: offset band the archive logged
    demonstrated_speed=_Embodied.Scenario.WarehouseAisle.v_latch_approach,
)


LatchScenarioTableAfterRecovery = LatchScenarioTable.variant(
    name="The latch table after a recovery collection at +3 mm and the guarded speed",
    added_offsets=[Q_(3, "mm")],  # scenario assumption: the stipulated latch-recovery-02
)


GripperStrokeTransfer = _st.GripperTransferScenario(
    name="A normalized gripper command moved from a 100 mm stroke to a 60 mm stroke",
    description="The same 0.80 command on both grippers against a 75 mm part; the donor's hold "
    "force against the target's continuous rating.",
    command=0.80,  # scenario assumption
    donor_stroke=Q_(100, "mm"),  # scenario assumption
    target_stroke=Q_(60, "mm"),  # scenario assumption
    part_width=Q_(75, "mm"),  # scenario assumption
    donor_hold_force=Q_(30, "N"),  # scenario assumption
    target_force_rating=Q_(15, "N"),  # scenario assumption
)


Llama3_70BAgentContextAdmissionDgxH100 = Llama3_70BLongContextAdmissionDgxH100.variant(
    name="32,768-token agent requests on the 70B GQA replica (8 H100s)",
    description="The repair request's 32,768 retained tokens per request, same replica and reserve.",
    context_tokens=RepairRequestZones.total_tokens,
)


Llama3_70BShortContextAdmissionDgxH100 = Llama3_70BLongContextAdmissionDgxH100.variant(
    name="8,192-token requests on the 70B GQA replica (8 H100s)",
    description="A short session of 8,192 retained tokens per request, same replica and reserve.",
    context_tokens=8_192,  # scenario assumption: short-session context
)


KvBytesPerTokenByAttentionDesign = _st.KvAttentionDesignScenario(
    name="KV bytes per token across multi-head, grouped-query, and latent attention",
    description="Six registry geometries at 16 bits; GPT-3 (MHA), Llama 3 70B (GQA), and "
    "DeepSeek-V3 (MLA) as references, with one 131,072-token session.",
    models=[
        _Models.Language.GPT3,
        _Models.Language.Llama3_405B,
        _Models.Language.Llama3_70B,
        _Models.Language.Llama2_70B,
        _Models.Language.Mixtral_8x7B,
        _Models.Language.DeepSeek_V3,
    ],
    mha=_Models.Language.GPT3,
    gqa=_Models.Language.Llama3_70B,
    mla=_Models.Language.DeepSeek_V3,
    session_tokens=Llama3_70BLongContextAdmissionDgxH100.context_tokens,
)


CappedRequestsPagedAllocation = _st.PagedAllocationScenario(
    name="Requests capped at 32,768 tokens with a mean final length of 2,048: reserved or paged",
    description="Full-cap contiguous reservation against 16- and 64-token blocks in the same pool; "
    "final lengths capped exponential (a modeling assumption).",
    pool=Llama3_70BAgentContextAdmissionDgxH100,
    cap=32_768,  # scenario assumption: deployment cap, not a model specification
    mean_final=2_048,  # scenario assumption: hypothetical mean final length, not a trace mean
    small_block=16,  # scenario assumption: allocator block sizes
    large_block=64,
)


ForkedCandidatesPrefixCopy = _st.KvForkCopyScenario(
    name="Four candidate repairs forked from an 8,192-token prefix without block sharing",
    description="Each candidate copies the prefix's KV state under contiguous allocation.",
    pool=Llama3_70BAgentContextAdmissionDgxH100,
    prefix_tokens=8_192,  # scenario assumption: processed prefix shared by the candidates
    branches=4,  # scenario assumption: sampled candidate repairs
)


InvariantPrefixRebuildDgxH100 = _st.PrefixRebuildScenario(
    name="A 4,096-token invariant prefix rebuilt on a miss, against a 32-token suffix pass on a hit",
    description="Llama 3 70B geometry on one DGX H100 node; sustained prefill at 90 percent of the "
    "high-MFU figure (45 percent of 16-bit peak).",
    workload=_Models.Language.Llama3_70B,
    node=_Nodes.DGX_H100,
    precision="fp16",
    prefix_tokens=4_096,  # scenario assumption: invariant instructions and tool schemas
    suffix_tokens=32,  # scenario assumption: short novel suffix after a hit
    utilization=_Literature.Training.MfuHigh * 0.9,  # scenario assumption: sustained prefill share
)


ChunkedPrefillAmongDecodes = _st.ChunkedPrefillScenario(
    name="A 4,096-token uncached prompt among 32 decodes, whole or in 512-token chunks",
    description="Weight-only step time on the DGX H100 node at the sustained prefill rate.",
    prefill=InvariantPrefixRebuildDgxH100,
    decodes=32,  # scenario assumption: ongoing decode requests
    prompt_tokens=4_096,  # scenario assumption: new uncached prefill
    chunk_tokens=512,  # scenario assumption: prefill chunk per iteration
)


RepairToolWaitPausedState = _st.PausedKvStateScenario(
    name="Repair request paused for a 45 s integration test: keep, move, or drop its KV state",
    description="The 32,768-token request in 16-token blocks, moved over the eight PCIe host links "
    "in parallel; 32 such requests returning together.",
    pool=Llama3_70BAgentContextAdmissionDgxH100,
    prefill=InvariantPrefixRebuildDgxH100,
    context_tokens=RepairRequestZones.total_tokens,
    block_tokens=16,  # scenario assumption: allocation granularity
    wait=Q_("45 s"),  # scenario assumption: stipulated integration-test wait
    link_fraction=0.25,  # scenario assumption: links sustain a quarter of rated bandwidth
    returns=32,  # scenario assumption: paused repairs whose tests finish together
)


RepairBlockSharingResidency = _st.KvBlockSharingScenario(
    name="Repair requests in 16-token blocks: unshared, sharing a workspace prefix, kept or released",
    description="32,768 input and 2,048 output tokens per request, a 24,576-token shareable "
    "prefix, 10 percent headroom, 5 s of generation and a 45 s test per turn.",
    pool=Llama3_70BAgentContextAdmissionDgxH100,
    headroom=0.10,  # scenario assumption: fraction of the pool withheld
    context=RepairRequestZones.total_tokens,
    output=2_048,  # scenario assumption: maximum private output tokens
    common=24_576,  # scenario assumption: identical, compatible, shareable prefix tokens
    block_tokens=RepairToolWaitPausedState.block_tokens,
    generation=Q_("5 s"),  # scenario assumption: residence while generating
    wait=RepairToolWaitPausedState.wait,
)


RepairAgentResidency = _st.KvResidencyArchetypeScenario(
    name="Repair agent: sharing only its system prompt and tool schemas, 5 s generation, 45 s test",
    description="The 32,768-token repair context with the support profile's 6,144-token shared "
    "prefix (taken to be the same length).",
    paused=RepairToolWaitPausedState,
    context_tokens=RepairRequestZones.total_tokens,
    shared_prefix=int(_Agents.Support.CustomerOps.shared_prefix.to("token").magnitude),
    generation=RepairBlockSharingResidency.generation,
    wait=RepairBlockSharingResidency.wait,
)


SupportAgentResidency = _st.KvResidencyArchetypeScenario(
    name="Customer-support agent: shared policy prefix, short generation, customer reply wait",
    description="The CustomerOps support profile's session context, shared prefix, generation "
    "residence, and customer reply time.",
    paused=RepairToolWaitPausedState,
    context_tokens=int(_Agents.Support.CustomerOps.session_context.to("token").magnitude),
    shared_prefix=int(_Agents.Support.CustomerOps.shared_prefix.to("token").magnitude),
    generation=_Agents.Support.CustomerOps.generation_residence,
    wait=_Agents.Support.CustomerOps.customer_reply_time,
)
