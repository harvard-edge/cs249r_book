"""Volume-level track arcs for the MLSysBook labs.

The arc registry gives each canonical track a durable narrative across Volume I
and Volume II while keeping individual labs standalone. Lab notebooks should
render concise arc context from this module instead of hand-authoring separate
journey text in every file.
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import LAB_CATALOG
from .tracks import CANONICAL_TRACKS, get_track_profile, normalize_track_id


@dataclass(frozen=True)
class TrackArc:
    track_id: str
    label: str
    mission_title: str
    learner_role: str
    system_goal: str
    device_family: str
    model_family: str
    allowed_hardware_refs: tuple[str, ...]
    allowed_model_refs: tuple[str, ...]
    recurring_metrics: tuple[str, ...]
    recurring_guardrails: tuple[str, ...]
    volume1_arc: str
    volume2_arc: str
    maintenance_note: str


@dataclass(frozen=True)
class LabArcStep:
    lab_id: str
    volume: str
    sequence: int
    chapter: str
    title: str
    concept: str
    arc_role: str
    carry_forward: str


TRACK_ARCS: tuple[TrackArc, ...] = (
    TrackArc(
        track_id="iphone",
        label="iPhone",
        mission_title="Ship a private on-device mobile ML feature",
        learner_role="Mobile product engineer",
        system_goal=(
            "Build a responsive local model that protects privacy while staying inside "
            "battery, thermal, memory, and interactive-latency limits."
        ),
        device_family="iPhone-class mobile SoC with unified memory and NPU/GPU/CPU fallback paths",
        model_family="MobileNet-class vision models and future small local language models",
        allowed_hardware_refs=("Hardware.Mobile.iPhone15Pro",),
        allowed_model_refs=("Models.Vision.MobileNetV2",),
        recurring_metrics=("battery drain", "thermal headroom", "on-device latency", "memory"),
        recurring_guardrails=("quality", "privacy", "responsiveness"),
        volume1_arc=(
            "Volume I turns the phone into a deployability test: data, workflow, "
            "architecture, compression, acceleration, serving, operations, and responsible "
            "engineering all ask whether the feature still feels instant and private."
        ),
        volume2_arc=(
            "Volume II scales the same mobile feature into a managed fleet: updates, "
            "telemetry, privacy, robustness, sustainability, and fairness must improve "
            "the product without turning it into a cloud-only service."
        ),
        maintenance_note="Use MobileNetV2 or an explicitly registered small local model unless the lab is teaching why a larger model cannot fit.",
    ),
    TrackArc(
        track_id="oura_ring",
        label="Oura Ring",
        mission_title="Fit sensing intelligence into a wearable power envelope",
        learner_role="Wearable firmware engineer",
        system_goal=(
            "Run useful sensing and inference under MCU-class SRAM, flash, OTA, radio, "
            "sampling-cadence, and battery constraints."
        ),
        device_family="Oura Ring / Cortex-M / ESP32-class TinyML envelope",
        model_family="DS-CNN, anomaly detection, and compact time-series classifiers",
        allowed_hardware_refs=("Hardware.Tiny.OuraRing",),
        allowed_model_refs=("Models.Tiny.DS_CNN", "Models.Tiny.AnomalyDetector"),
        recurring_metrics=("SRAM fit", "flash fit", "battery life", "OTA payload"),
        recurring_guardrails=("signal quality", "sampling cadence", "user comfort"),
        volume1_arc=(
            "Volume I makes every systems idea concrete through a tiny always-on device: "
            "the student repeatedly asks what must be measured, shrunk, scheduled, or "
            "validated before a wearable can run for days."
        ),
        volume2_arc=(
            "Volume II treats the wearable as a fleet of constrained sensors: data "
            "freshness, updates, reliability, privacy, robustness, carbon, and fairness "
            "are all bounded by tiny local resources."
        ),
        maintenance_note="Use Oura, ESP32, or Cortex-M style hardware and TinyML models unless the lab is explicitly contrasting infeasible larger models.",
    ),
    TrackArc(
        track_id="robotaxi",
        label="RoboTaxi",
        mission_title="Keep autonomous perception inside edge safety margins",
        learner_role="Autonomous vehicle platform engineer",
        system_goal=(
            "Maintain perception and planning evidence under p99/p999 latency, power, "
            "sensor bandwidth, reliability, and safety guardrails."
        ),
        device_family="RoboTaxi / DRIVE AGX Orin-class edge autonomy compute",
        model_family="YOLOv8 Nano-class perception plus rare-event and replay workloads",
        allowed_hardware_refs=("Hardware.Edge.RoboTaxi",),
        allowed_model_refs=("Models.Vision.YOLOv8_Nano",),
        recurring_metrics=("p99 latency", "p999 latency", "rare-event recall", "power"),
        recurring_guardrails=("safety margin", "reliability", "thermal headroom"),
        volume1_arc=(
            "Volume I frames ML systems as safety-critical edge engineering: each chapter "
            "tests whether a model, data policy, runtime, or operations choice survives "
            "rare events and tail latency."
        ),
        volume2_arc=(
            "Volume II turns the single vehicle into an autonomy fleet: orchestration, "
            "failure recovery, privacy, robustness, sustainability, and fairness are judged "
            "by operational safety margins."
        ),
        maintenance_note="Use edge perception, replay, p99/p999 latency, and safety-margin language; avoid generic datacenter-only assumptions.",
    ),
    TrackArc(
        track_id="cloud_fleet",
        label="Cloud Fleet",
        mission_title="Operate a model service under fleet-scale constraints",
        learner_role="Fleet service owner",
        system_goal=(
            "Run a service under throughput, p99 latency, SLA, cost/request, utilization, "
            "capacity, quality, and carbon constraints."
        ),
        device_family="H100-backed lab cluster and fleet service infrastructure",
        model_family="BERT/GPT/Llama-class service workloads chosen to expose scale constraints",
        allowed_hardware_refs=("Hardware.Cloud.H100",),
        allowed_model_refs=(
            "Models.Language.BERT_Base",
            "Models.Language.GPT2",
            "Models.Language.Llama2_70B",
        ),
        recurring_metrics=("throughput", "p99 latency", "cost/request", "utilization", "carbon"),
        recurring_guardrails=("SLA", "quality", "capacity headroom"),
        volume1_arc=(
            "Volume I asks how each ML systems concept changes when the deliverable is "
            "a reliable service rather than a local demo."
        ),
        volume2_arc=(
            "Volume II follows the fleet as it scales: infrastructure, communication, "
            "storage, training, inference, operations, privacy, robustness, carbon, and "
            "responsibility all become service-management tradeoffs."
        ),
        maintenance_note="Use registered cloud service models and cluster/system references; distinguish service scale from a single accelerator benchmark.",
    ),
)


_STEP_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("v1_00_architects_portal", "Track selection", "Choose a deployment lens and learn that the same ML idea has different system constraints.", "Carry the selected track into the Volume I journey."),
    ("v1_01_ai_triad", "Data/algorithm/machine diagnosis", "Separate model behavior from data and machine limits in the selected deployment.", "Use the triad to explain later failures."),
    ("v1_02_physics_of_deployment", "Physical deployment envelope", "Translate the track into concrete memory, latency, energy, bandwidth, or cost limits.", "Treat feasibility as a physical budget."),
    ("v1_03_constraint_tax", "Workflow gate", "Find where late discovery of the track constraint creates rework.", "Move validation earlier in the lifecycle."),
    ("v1_04_data_gravity", "Data movement and retention", "Decide where data should be produced, filtered, moved, stored, or summarized.", "Carry data movement costs into training and operations."),
    ("v1_05_neural_computation", "Tensor and activation budget", "Connect operator shape to memory movement and device limits.", "Use tensor cost as the bridge to architecture."),
    ("v1_06_architecture_tax", "Architecture choice", "Choose a model family that matches the track envelope.", "Carry architecture assumptions into framework/runtime decisions."),
    ("v1_07_framework_tax", "Runtime and framework support", "Check whether the chosen stack supports the needed operators and deployment path.", "Use runtime support as a guardrail before training and compression."),
    ("v1_08_training_gauntlet", "Training plan", "Shape the training plan around the selected deployment target.", "Carry validation and handoff requirements into data selection."),
    ("v1_09_selection_paradox", "Data selection policy", "Choose data coverage and rare-event policy for the track.", "Carry coverage gaps into compression, serving, and responsible engineering."),
    ("v1_10_compression_paradox", "Compression recipe", "Shrink or distill the model only if quality and track guardrails survive.", "Carry the selected recipe into hardware acceleration."),
    ("v1_11_hardware_roofline", "Hardware roofline", "Diagnose whether compute, memory bandwidth, or accelerator support binds first.", "Use bottleneck evidence before benchmarking."),
    ("v1_12_benchmarking_trap", "Benchmark design", "Design measurements that represent the track rather than a vendor-friendly average.", "Carry benchmark evidence into serving decisions."),
    ("v1_13_tail_latency_trap", "Serving policy", "Set batching, cold-start, and p99 policies for the selected deployment.", "Carry serving risk into operations."),
    ("v1_14_silent_degradation", "Operations and drift", "Choose monitoring and rollback policies that reveal track-specific degradation.", "Carry operational risk into responsibility review."),
    ("v1_15_no_free_fairness", "Responsible engineering", "Expose who benefits or is harmed by the selected deployment choices.", "Carry unresolved risks into the final audit."),
    ("v1_16_architects_audit", "Volume I synthesis", "Defend a coherent deployment decision across the full systems stack.", "Use the audit as the handoff into Volume II scale."),
    ("v2_01_scale_illusion", "Scale transition", "Show what breaks when the same track is scaled beyond a single prototype.", "Carry scale assumptions into infrastructure."),
    ("v2_02_compute_wall", "Compute infrastructure", "Balance raw compute against memory, utilization, power, or cost limits.", "Carry the infrastructure wall into communication design."),
    ("v2_03_network_fabric_design", "Communication fabric", "Choose a communication pattern that respects the track's latency and payload budget.", "Carry payload pressure into storage and data movement."),
    ("v2_04_data_pipeline_wall", "Storage and freshness", "Decide what data must be retained, refreshed, summarized, or discarded.", "Carry pipeline limits into distributed training."),
    ("v2_05_parallelism_design", "Distributed training", "Choose parallelism only when it supports the deployment objective.", "Carry training communication into collectives."),
    ("v2_06_collective_communication", "Collective communication", "Understand synchronization cost and topology for the selected scale regime.", "Carry communication evidence into reliability."),
    ("v2_07_failure_budget_engineering", "Failure budget", "Design graceful failure and recovery policies for the track.", "Carry failure modes into orchestration."),
    ("v2_08_fleet_orchestration", "Scheduling and orchestration", "Choose scheduling policies that do not violate track guardrails.", "Carry scheduling side effects into optimization."),
    ("v2_09_optimization_trap", "System optimization", "Reject local speedups that regress the track-level outcome.", "Carry optimization evidence into inference economics."),
    ("v2_10_inference_economy", "Inference economics", "Balance batching, cache/state, cost, and p99 behavior for the track.", "Carry serving economics into edge placement."),
    ("v2_11_edge_thermodynamics", "Edge placement", "Decide what should run locally, near the user, or centrally.", "Carry placement tradeoffs into fleet monitoring."),
    ("v2_12_silent_fleet", "Fleet observability", "Choose telemetry and action thresholds that reveal failures in the selected track.", "Carry observability into privacy/security decisions."),
    ("v2_13_price_of_privacy", "Privacy and security", "Pay the cost of privacy/security controls without hiding track failures.", "Carry security constraints into robustness."),
    ("v2_14_robustness_budget", "Robustness budget", "Allocate robustness effort to the failures that matter most for the track.", "Carry robustness gaps into sustainability."),
    ("v2_15_carbon_budget", "Sustainability budget", "Measure energy and carbon consequences of the selected deployment path.", "Carry sustainability tradeoffs into responsibility."),
    ("v2_16_fairness_budget", "Fleet responsibility", "Balance fairness, explanation, latency, and cost for the track.", "Carry residual risks into the final synthesis."),
    ("v2_17_fleet_synthesis", "Volume II synthesis", "Defend the full scaled system for the selected track.", "Close the narrative with a deployment review."),
)


def _build_steps() -> tuple[LabArcStep, ...]:
    metadata_by_id = {metadata.lab_id: metadata for metadata in LAB_CATALOG.values()}
    steps: list[LabArcStep] = []
    volume_counts: dict[str, int] = {}
    for lab_id, concept, arc_role, carry_forward in _STEP_SPECS:
        metadata = metadata_by_id[lab_id]
        volume_counts[metadata.volume] = volume_counts.get(metadata.volume, 0) + 1
        steps.append(
            LabArcStep(
                lab_id=lab_id,
                volume=metadata.volume,
                sequence=volume_counts[metadata.volume],
                chapter=metadata.chapter,
                title=metadata.title,
                concept=concept,
                arc_role=arc_role,
                carry_forward=carry_forward,
            )
        )
    return tuple(steps)


LAB_ARC_STEPS: tuple[LabArcStep, ...] = _build_steps()


def track_arc_map() -> dict[str, TrackArc]:
    return {arc.track_id: arc for arc in TRACK_ARCS}


def get_track_arc(track_id: str | None) -> TrackArc:
    normalized = normalize_track_id(track_id)
    arcs = track_arc_map()
    if normalized not in arcs:
        valid = ", ".join(arcs)
        raise KeyError(f"Unknown track arc {track_id!r}. Expected one of: {valid}")
    return arcs[normalized]


def list_track_arcs() -> tuple[TrackArc, ...]:
    return TRACK_ARCS


def lab_arc_step_map() -> dict[str, LabArcStep]:
    return {step.lab_id: step for step in LAB_ARC_STEPS}


def get_lab_arc_step(lab_id: str) -> LabArcStep:
    steps = lab_arc_step_map()
    if lab_id not in steps:
        valid = ", ".join(steps)
        raise KeyError(f"Unknown lab arc step {lab_id!r}. Expected one of: {valid}")
    return steps[lab_id]


def list_lab_arc_steps() -> tuple[LabArcStep, ...]:
    return LAB_ARC_STEPS


def track_arc_context_summary(track_id: str | None, lab_id: str) -> dict[str, str]:
    arc = get_track_arc(track_id)
    step = get_lab_arc_step(lab_id)
    volume_arc = arc.volume1_arc if step.volume == "Volume I" else arc.volume2_arc
    return {
        "Track mission": arc.mission_title,
        "System goal": arc.system_goal,
        "This lab's role": f"{step.concept}: {step.arc_role}",
        "Carry forward": step.carry_forward,
        "Volume arc": volume_arc,
    }


def validate_track_arcs() -> tuple[str, ...]:
    """Return human-readable arc/variant coverage issues."""
    issues: list[str] = []
    track_ids = tuple(profile.track_id for profile in CANONICAL_TRACKS)
    arc_ids = tuple(arc.track_id for arc in TRACK_ARCS)
    if arc_ids != track_ids:
        issues.append(f"Track arc IDs {arc_ids!r} do not match canonical tracks {track_ids!r}.")

    lab_ids = tuple(metadata.lab_id for metadata in LAB_CATALOG.values())
    step_ids = tuple(step.lab_id for step in LAB_ARC_STEPS)
    if step_ids != lab_ids:
        issues.append("Lab arc steps do not match LAB_CATALOG order.")

    for arc in TRACK_ARCS:
        profile = get_track_profile(arc.track_id)
        if profile.hardware_ref not in arc.allowed_hardware_refs:
            issues.append(f"{arc.track_id} profile hardware is not allowed by its arc.")
        if not arc.allowed_model_refs:
            issues.append(f"{arc.track_id} has no allowed model refs.")
    return tuple(issues)


__all__ = [
    "LAB_ARC_STEPS",
    "TRACK_ARCS",
    "LabArcStep",
    "TrackArc",
    "get_lab_arc_step",
    "get_track_arc",
    "lab_arc_step_map",
    "list_lab_arc_steps",
    "list_track_arcs",
    "track_arc_context_summary",
    "track_arc_map",
    "validate_track_arcs",
]
