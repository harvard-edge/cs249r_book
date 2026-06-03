"""Typed track-specific lab scenario variants."""

from __future__ import annotations

from .schemas import LabTrackVariant
from .tracks import CANONICAL_TRACKS, get_track_profile, normalize_track_id


PILOT_LAB_IDS = (
    "v1_00_architects_portal",
    "v1_10_compression_paradox",
    "v2_11_edge_thermodynamics",
)


def _variant(
    *,
    lab_id: str,
    track_id: str,
    scenario_id: str,
    stakeholder: str,
    workload_summary: str,
    objective: str,
    primary_metric: str,
    guardrail_metric: str,
    model_ref: str,
    defaults: dict[str, object],
    assumptions: dict[str, object],
) -> LabTrackVariant:
    profile = get_track_profile(track_id)
    return LabTrackVariant(
        lab_id=lab_id,
        track_id=profile.track_id,
        scenario_id=scenario_id,
        stakeholder=stakeholder,
        workload_summary=workload_summary,
        objective=objective,
        primary_metric=primary_metric,
        guardrail_metric=guardrail_metric,
        hardware_ref=profile.hardware_ref,
        system_ref=profile.system_ref,
        model_ref=model_ref,
        defaults=defaults,
        assumptions=assumptions,
    )


LAB_TRACK_VARIANTS: tuple[LabTrackVariant, ...] = (
    # Lab 00: orientation and track identity.
    _variant(
        lab_id="v1_00_architects_portal",
        track_id="iphone",
        scenario_id="v1_00_iphone_track_identity",
        stakeholder="Mobile product engineer",
        workload_summary="Repeated on-device app inference used to introduce battery, thermal, privacy, and memory constraints.",
        objective="Choose iPhone as the persistent course track and predict the first recurring bottleneck.",
        primary_metric="thermal and battery headroom",
        guardrail_metric="interactive responsiveness",
        model_ref="Models.Vision.MobileNetV2",
        defaults={"initial_bottleneck": "thermal envelope", "assignment_mode": "student choice"},
        assumptions={"report_artifact": "orientation memo", "solver_required": False},
    ),
    _variant(
        lab_id="v1_00_architects_portal",
        track_id="oura_ring",
        scenario_id="v1_00_oura_track_identity",
        stakeholder="Wearable firmware engineer",
        workload_summary="Always-on wearable sensing used to introduce SRAM, flash, OTA, and battery constraints.",
        objective="Choose Oura Ring as the persistent course track and predict the first recurring bottleneck.",
        primary_metric="SRAM and flash fit",
        guardrail_metric="battery life",
        model_ref="Models.Tiny.DS_CNN",
        defaults={"initial_bottleneck": "SRAM", "assignment_mode": "student choice"},
        assumptions={"report_artifact": "orientation memo", "solver_required": False},
    ),
    _variant(
        lab_id="v1_00_architects_portal",
        track_id="robotaxi",
        scenario_id="v1_00_robotaxi_track_identity",
        stakeholder="Autonomous vehicle platform engineer",
        workload_summary="Vehicle-local perception loop used to introduce p99 latency, reliability, power, and safety guardrails.",
        objective="Choose RoboTaxi as the persistent course track and predict the first recurring bottleneck.",
        primary_metric="p99 latency",
        guardrail_metric="rare-event recall",
        model_ref="Models.Vision.YOLOv8_Nano",
        defaults={"initial_bottleneck": "tail latency", "assignment_mode": "student choice"},
        assumptions={"report_artifact": "orientation memo", "solver_required": False},
    ),
    _variant(
        lab_id="v1_00_architects_portal",
        track_id="cloud_fleet",
        scenario_id="v1_00_cloud_fleet_track_identity",
        stakeholder="Fleet service owner",
        workload_summary="Production inference fleet used to introduce throughput, cost, utilization, SLA, and carbon constraints.",
        objective="Choose Cloud Fleet as the persistent course track and predict the first recurring bottleneck.",
        primary_metric="throughput",
        guardrail_metric="SLA",
        model_ref="Models.Language.BERT_Base",
        defaults={"initial_bottleneck": "utilization", "assignment_mode": "student choice"},
        assumptions={"report_artifact": "orientation memo", "solver_required": False},
    ),

    # V1-10: Compression Paradox pilot.
    _variant(
        lab_id="v1_10_compression_paradox",
        track_id="iphone",
        scenario_id="v1_10_iphone_compression",
        stakeholder="Mobile product lead",
        workload_summary="Repeated mobile vision inference with sustained UX and supported-accelerator constraints.",
        objective="Choose a compression recipe that reduces battery/thermal pressure without CPU or GPU fallback.",
        primary_metric="battery or thermal headroom",
        guardrail_metric="quality and on-device p99 latency",
        model_ref="Models.Vision.MobileNetV2",
        defaults={"candidate_methods": ("int8_quantization", "structured_pruning", "distillation"), "bit_widths": (16, 8, 6, 4)},
        assumptions={"unsupported_kernel_policy": "marks infeasible", "report_artifact": "compression deployment recipe"},
    ),
    _variant(
        lab_id="v1_10_compression_paradox",
        track_id="oura_ring",
        scenario_id="v1_10_oura_compression",
        stakeholder="Wearable firmware lead",
        workload_summary="Low-rate biosignal classifier with strict runtime memory and OTA payload constraints.",
        objective="Choose a compression recipe that fits model, runtime, and OTA package while preserving battery and signal guardrails.",
        primary_metric="flash/SRAM fit and OTA payload size",
        guardrail_metric="battery life and signal quality",
        model_ref="Models.Tiny.DS_CNN",
        defaults={"candidate_methods": ("int8_quantization", "structured_pruning", "distillation"), "bit_widths": (8, 6, 4, 3, 2)},
        assumptions={"runtime_memory_included": True, "report_artifact": "compression deployment recipe"},
    ),
    _variant(
        lab_id="v1_10_compression_paradox",
        track_id="robotaxi",
        scenario_id="v1_10_robotaxi_compression",
        stakeholder="Safety/perception lead",
        workload_summary="Vehicle-local perception model under bursty sensor workload and tail-latency safety constraints.",
        objective="Choose a compression recipe that lowers p99 latency without reducing rare-hazard recall.",
        primary_metric="p99 or p999 latency",
        guardrail_metric="rare-event recall",
        model_ref="Models.Vision.YOLOv8_Nano",
        defaults={"candidate_methods": ("int8_quantization", "structured_pruning", "distillation"), "bit_widths": (16, 8, 6, 4)},
        assumptions={"rare_event_validation_required": True, "report_artifact": "compression deployment recipe"},
    ),
    _variant(
        lab_id="v1_10_compression_paradox",
        track_id="cloud_fleet",
        scenario_id="v1_10_cloud_fleet_compression",
        stakeholder="Infrastructure lead",
        workload_summary="High-volume inference service where compression changes cost/request, throughput, quality, and SLA.",
        objective="Choose a compression recipe that improves cost/request or throughput without quality or SLA regression.",
        primary_metric="cost/request or throughput",
        guardrail_metric="quality and SLA",
        model_ref="Models.Language.BERT_Base",
        defaults={"candidate_methods": ("int8_quantization", "structured_pruning", "distillation"), "bit_widths": (16, 8, 6, 4)},
        assumptions={"fleet_profile": "Systems.Clusters.Lab_64_H100", "report_artifact": "compression deployment recipe"},
    ),

    # V2-11: Edge Thermodynamics pilot.
    _variant(
        lab_id="v2_11_edge_thermodynamics",
        track_id="iphone",
        scenario_id="v2_11_iphone_edge_placement",
        stakeholder="Mobile privacy engineer",
        workload_summary="Personalized on-device inference with optional phone-edge or cloud fallback.",
        objective="Choose what remains on-device and what can move off-device under latency, privacy, and battery constraints.",
        primary_metric="on-device latency and battery cost",
        guardrail_metric="privacy requirement",
        model_ref="Models.Vision.MobileNetV2",
        defaults={"placements": ("local", "phone_edge", "cloud", "hybrid"), "adaptation_options": ("local", "federated", "centralized")},
        assumptions={"privacy_sensitive": True, "report_artifact": "edge architecture memo"},
    ),
    _variant(
        lab_id="v2_11_edge_thermodynamics",
        track_id="oura_ring",
        scenario_id="v2_11_oura_edge_placement",
        stakeholder="Wearable systems engineer",
        workload_summary="Ring-local sensing with phone-assisted or cloud-assisted heavier computation.",
        objective="Choose a tiny local model plus handoff policy that preserves energy, memory, and privacy.",
        primary_metric="energy per inference",
        guardrail_metric="SRAM/flash fit",
        model_ref="Models.Tiny.AnomalyDetector",
        defaults={"placements": ("ring_only", "ring_phone", "ring_cloud", "hybrid"), "adaptation_options": ("calibration", "phone_mediated", "centralized")},
        assumptions={"phone_handoff_available": True, "report_artifact": "edge architecture memo"},
    ),
    _variant(
        lab_id="v2_11_edge_thermodynamics",
        track_id="robotaxi",
        scenario_id="v2_11_robotaxi_edge_placement",
        stakeholder="Autonomous fleet safety lead",
        workload_summary="Safety-critical vehicle perception with roadside/depot/cloud paths for non-critical learning loops.",
        objective="Keep the safety path vehicle-local while choosing how fleet learning and updates use edge or cloud.",
        primary_metric="vehicle-local p99 latency",
        guardrail_metric="safety-critical privacy and reliability",
        model_ref="Models.Vision.YOLOv8_Nano",
        defaults={"placements": ("vehicle_local", "roadside_edge", "depot_edge", "cloud"), "adaptation_options": ("fleet_learning", "centralized_retrain", "federated")},
        assumptions={"safety_path_must_be_local": True, "report_artifact": "edge architecture memo"},
    ),
    _variant(
        lab_id="v2_11_edge_thermodynamics",
        track_id="cloud_fleet",
        scenario_id="v2_11_cloud_fleet_edge_placement",
        stakeholder="Cloud service owner",
        workload_summary="Centralized service that may use edge caching, offload, or feedback loops to improve latency and cost.",
        objective="Choose when central serving is enough and when edge placement improves latency, cost, or feedback quality.",
        primary_metric="p99 latency and cost/request",
        guardrail_metric="SLA and quality",
        model_ref="Models.Language.BERT_Base",
        defaults={"placements": ("centralized", "edge_cache", "edge_offload", "hybrid"), "adaptation_options": ("centralized_retrain", "edge_feedback", "federated")},
        assumptions={"central_service_baseline": True, "report_artifact": "edge architecture memo"},
    ),
)


_VARIANT_MAP: dict[tuple[str, str], LabTrackVariant] = {
    (variant.lab_id, variant.track_id): variant for variant in LAB_TRACK_VARIANTS
}


def list_lab_variants(lab_id: str) -> tuple[LabTrackVariant, ...]:
    """Return all track variants for a lab ID."""
    return tuple(variant for variant in LAB_TRACK_VARIANTS if variant.lab_id == lab_id)


def get_lab_track_variant(lab_id: str, track_id: str | None) -> LabTrackVariant:
    """Return the variant for a lab and canonical or legacy track ID."""
    normalized = normalize_track_id(track_id)
    try:
        return _VARIANT_MAP[(lab_id, normalized)]
    except KeyError as exc:
        raise KeyError(f"No variant for lab_id={lab_id!r}, track_id={track_id!r}") from exc


def variant_coverage() -> dict[str, tuple[str, ...]]:
    """Return available track IDs by lab ID for tests and dashboards."""
    return {
        lab_id: tuple(variant.track_id for variant in list_lab_variants(lab_id))
        for lab_id in PILOT_LAB_IDS
    }


def canonical_track_ids() -> tuple[str, ...]:
    """Return canonical track IDs in display order."""
    return tuple(profile.track_id for profile in CANONICAL_TRACKS)


__all__ = [
    "LAB_TRACK_VARIANTS",
    "PILOT_LAB_IDS",
    "canonical_track_ids",
    "get_lab_track_variant",
    "list_lab_variants",
    "variant_coverage",
]
