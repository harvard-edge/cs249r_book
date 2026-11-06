"""Volume I's deployment lenses and preferred-context handoff.

This module owns interface language only. Canonical project narratives, specs, budgets, defaults, and
physical behavior remain in MLSysIM. The existing branded profiles are accepted
as aliases without changing the context vocabulary used in other volumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from collections.abc import Mapping
from typing import Any

from mlsysim.engine.scenarios import Scenarios
from mlsysim.engine.deployment_journeys import DeploymentJourney


@dataclass(frozen=True)
class DeploymentContext:
    track_id: str
    label: str
    role: str
    workload: str
    pressure: str
    description: str

    @property
    def scenario(self) -> DeploymentJourney:
        """The canonical project brief; no narrative is copied into this bridge."""
        return Scenarios.DeploymentJourneys.get(self.track_id)


_CONTEXTS = (
    DeploymentContext(
        "tinyml", "TinyML", "Sensing-device engineer", "Local sensing and inference",
        "SRAM, flash, observation cadence, and energy",
        "A small device runs the deployed model. A separate development host prepares and trains it.",
    ),
    DeploymentContext(
        "mobile", "MobileML", "Mobile application engineer", "An interactive on-device feature",
        "Shared memory, responsiveness, battery, and sustained power",
        "The model shares a handheld device with the application and other work. Local execution still has limits.",
    ),
    DeploymentContext(
        "edge", "EdgeML", "Local inference engineer", "Sensor processing near the source",
        "Input delivery, local deadlines, memory, and power",
        "A local computer processes inputs near the sensor. Its application still has to respond in time.",
    ),
    DeploymentContext(
        "cloud", "CloudML", "Serving-endpoint engineer", "One remote inference endpoint",
        "Data movement, compute, memory, and response time",
        "A remote host or accelerator supplies capacity and adds a communication path. This volume studies one endpoint.",
    ),
)
CONTEXT_OPTIONS = {context.label: context.track_id for context in _CONTEXTS}
_CONTEXT_BY_ID = {context.track_id: context for context in _CONTEXTS}



def get_context(track_id: str) -> DeploymentContext:
    """Resolve a category or legacy device profile to the Volume I lens."""
    return _CONTEXT_BY_ID[Scenarios.DeploymentJourneys.get(track_id).track_id]


def preferred_context(ledger: Any) -> str:
    """Restore a saved preference without interpreting prior results as evidence.

    The explicit orientation preference wins over later experiments' current
    track. A missing or old ledger remains a supported standalone start.
    """
    saved = ledger.get_design(0)
    orientation = saved if isinstance(saved, Mapping) else {}
    choices = (
        orientation.get("preferred_track"), orientation.get("track_id"),
        orientation.get("deployment_context"), ledger.get_track(), "tinyml",
    )
    for choice in choices:
        if isinstance(choice, str):
            try:
                return get_context(choice).track_id
            except KeyError:
                pass
    return "tinyml"


def restore_orientation(ledger: Any) -> dict[str, Any]:
    """Return a detached, recognizable setup record or a safe fresh start."""
    saved = ledger.get_design(0)
    if not isinstance(saved, dict) or saved.get("orientation_version") != 1:
        return {}
    capture = saved.get("comparison")
    if not isinstance(capture, dict):
        return {}
    try:
        context = get_context(saved["preferred_track"]).track_id
        if (
            capture["track"] != context
            or capture["part"] != "orientation"
            or capture["baseline"]["inputs"]["case"] != "baseline"
            or capture["result"]["inputs"]["case"] != "alternative"
            or capture["baseline"]["inputs"]["context"] != context
            or capture["result"]["inputs"]["context"] != context
            or capture.get("prediction") not in ("prediction", "control", "save")
        ):
            return {}
    except (KeyError, TypeError, AttributeError):
        return {}
    return deepcopy(saved)
