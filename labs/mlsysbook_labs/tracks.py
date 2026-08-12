"""Canonical student track profiles for track-aware MLSysBook labs."""

from __future__ import annotations

from .schemas import TrackProfile


DEFAULT_TRACK_ID = "iphone"


TRACK_EMOJIS: dict[str, str] = {
    "iphone": "📱",
    "oura_ring": "💍",
    "robotaxi": "🚕",
    "cloud_fleet": "☁️",
}


CANONICAL_TRACKS: tuple[TrackProfile, ...] = (
    TrackProfile(
        track_id="iphone",
        label="iPhone",
        category="Mobile ML",
        hardware_ref="Hardware.Mobile.iPhone15Pro",
        system_ref=None,
        stakeholder="Mobile product engineer",
        primary_metrics=("battery drain", "thermal headroom", "on-device latency", "memory"),
        guardrail_metrics=("quality", "privacy", "responsiveness"),
        dominant_constraints=("battery", "thermal envelope", "unified memory", "interactive latency"),
        narrative="Ship a responsive local model without overheating or draining the phone.",
        source_policy="Hardware facts come from MLSysIM Hardware.Mobile.iPhone15Pro.",
    ),
    TrackProfile(
        track_id="oura_ring",
        label="Oura Ring",
        category="TinyML / wearable",
        hardware_ref="Hardware.Tiny.OuraRing",
        system_ref=None,
        stakeholder="Wearable firmware engineer",
        primary_metrics=("SRAM fit", "flash fit", "battery life", "OTA payload"),
        guardrail_metrics=("signal quality", "sampling cadence", "user comfort"),
        dominant_constraints=("SRAM", "flash", "battery energy", "radio/OTA budget"),
        narrative="Fit sensing and inference into a tiny wearable budget.",
        source_policy=(
            "Public Oura specs seed the profile; non-public MCU and battery internals are "
            "explicit MLSysIM estimates."
        ),
    ),
    TrackProfile(
        track_id="robotaxi",
        label="RoboTaxi",
        category="Edge AI",
        hardware_ref="Hardware.Edge.RoboTaxi",
        system_ref=None,
        stakeholder="Autonomous vehicle platform engineer",
        primary_metrics=("p99 latency", "p999 latency", "rare-event recall", "power"),
        guardrail_metrics=("safety margin", "reliability", "thermal headroom"),
        dominant_constraints=("tail latency", "power envelope", "sensor bandwidth", "safety guardrails"),
        narrative="Keep perception and planning inside tight latency and safety guardrails.",
        source_policy=(
            "The track uses a DRIVE AGX Orin-class MLSysIM reference profile; vehicle-fleet "
            "operator internals are not assumed."
        ),
    ),
    TrackProfile(
        track_id="cloud_fleet",
        label="Cloud Fleet",
        category="Cloud/Fleet",
        hardware_ref="Hardware.Cloud.H100",
        system_ref="Systems.Clusters.Lab_64_H100",
        stakeholder="Fleet service owner",
        primary_metrics=("throughput", "p99 latency", "cost/request", "utilization", "carbon"),
        guardrail_metrics=("SLA", "quality", "capacity headroom"),
        dominant_constraints=("throughput", "SLA", "cost", "utilization", "carbon intensity"),
        narrative="Run a service under SLA, cost, capacity, and sustainability constraints.",
        source_policy=(
            "Hardware facts come from Hardware.Cloud.H100; fleet topology comes from "
            "Systems.Clusters.Lab_64_H100."
        ),
    ),
)


TRACK_ALIASES: dict[str, str] = {
    "mobile": "iphone",
    "mobile_ml": "iphone",
    "phone": "iphone",
    "tiny": "oura_ring",
    "tinyml": "oura_ring",
    "wearable": "oura_ring",
    "oura": "oura_ring",
    "edge": "robotaxi",
    "edge_ai": "robotaxi",
    "taxi": "robotaxi",
    "robo_taxi": "robotaxi",
    "cloud": "cloud_fleet",
    "cloud/fleet": "cloud_fleet",
    "fleet": "cloud_fleet",
}


def normalize_track_id(track_id: str | None) -> str:
    """Normalize canonical and legacy track IDs to the canonical profile ID."""
    if not track_id:
        return DEFAULT_TRACK_ID
    key = track_id.strip().lower().replace("-", "_").replace(" ", "_")
    return TRACK_ALIASES.get(key, key)


def get_track_profile(track_id: str | None) -> TrackProfile:
    """Return the canonical track profile, accepting legacy category aliases."""
    normalized = normalize_track_id(track_id)
    for profile in CANONICAL_TRACKS:
        if profile.track_id == normalized:
            return profile
    valid = ", ".join(profile.track_id for profile in CANONICAL_TRACKS)
    raise KeyError(f"Unknown track_id {track_id!r}. Expected one of: {valid}")


def track_emoji(track_id: str | TrackProfile | None) -> str:
    """Return the canonical emoji for a track."""
    profile = track_id if isinstance(track_id, TrackProfile) else get_track_profile(track_id)
    return TRACK_EMOJIS.get(profile.track_id, "")


def track_display_label(track_id: str | TrackProfile | None, *, include_category: bool = False) -> str:
    """Return a consistent student-facing track label with its canonical emoji."""
    profile = track_id if isinstance(track_id, TrackProfile) else get_track_profile(track_id)
    suffix = f" ({profile.category})" if include_category else ""
    return f"{track_emoji(profile)} {profile.label}{suffix}".strip()


def track_options() -> dict[str, str]:
    """Return Marimo-friendly radio options keyed by student-facing label."""
    return {track_display_label(profile): profile.track_id for profile in CANONICAL_TRACKS}


def track_profile_map() -> dict[str, TrackProfile]:
    """Return canonical profiles keyed by track ID."""
    return {profile.track_id: profile for profile in CANONICAL_TRACKS}


__all__ = [
    "CANONICAL_TRACKS",
    "DEFAULT_TRACK_ID",
    "TRACK_EMOJIS",
    "TRACK_ALIASES",
    "get_track_profile",
    "normalize_track_id",
    "track_display_label",
    "track_emoji",
    "track_options",
    "track_profile_map",
]
