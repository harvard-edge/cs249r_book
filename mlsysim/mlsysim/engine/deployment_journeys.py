"""Versioned fictional deployment briefs, separate from numerical study fixtures.

Narrative continuity does not turn a controlled experiment into a measurement of
its target application. Consumers should show ``study_boundary`` beside studies.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class DeploymentEpisode:
    chapter: int
    title: str
    question: str
    task: str
    handoff: str
    scene: str
    choice: str
    consequence: str
    core_concepts: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class DeploymentAct:
    """An authored narrative arc, not a numerical validation milestone."""

    id: str
    title: str
    chapters: tuple[int, ...]
    arc: str


@dataclass(frozen=True)
class DeploymentJourney:
    scenario_id: str
    version: str
    track_id: str
    title: str
    mission: str
    storyline: str
    stakeholder: str
    workload: str
    boundary: str
    success: tuple[str, ...]
    unknowns: tuple[str, ...]
    study_boundary: str
    episodes: tuple[DeploymentEpisode, ...]
    acts: tuple[DeploymentAct, ...]

    def episode(self, chapter: int) -> DeploymentEpisode:
        """Return an authored episode; never invent a brief for an unknown chapter."""
        for item in self.episodes:
            if item.chapter == chapter:
                return item
        raise KeyError(f"No reviewed episode for chapter {chapter!r}")


@dataclass(frozen=True)
class DeploymentInvestigation:
    """An intended learning move, not an implemented numerical experiment."""

    title: str
    question: str
    interaction: str
    notice: str


@dataclass(frozen=True)
class DeploymentLabStory:
    chapter: int
    headline: str
    opening: str
    previous: str
    goal: str
    investigations: tuple[DeploymentInvestigation, ...]
    turning_point: str
    dossier_entry: str
    transition: str
    qmd_bridge: str


@dataclass(frozen=True)
class DeploymentVolumeStory:
    title: str
    premise: str
    opening: str
    ending: str
    chapters: tuple[DeploymentLabStory, ...]

    def chapter(self, number: int) -> DeploymentLabStory:
        for chapter in self.chapters:
            if chapter.chapter == number:
                return chapter
        raise KeyError(f"No authored storyboard for chapter {number!r}")


@dataclass(frozen=True)
class DeploymentJourneyCatalog:
    version: str
    provenance: Mapping[str, str]
    aliases: Mapping[str, str]
    journeys: tuple[DeploymentJourney, ...]
    volume_storyboard: DeploymentVolumeStory

    def get(self, track_id: str) -> DeploymentJourney:
        """Resolve canonical categories and previously saved profile aliases."""
        key = track_id.strip().lower().replace("-", "_").replace(" ", "_")
        canonical = self.aliases.get(key, key)
        for journey in self.journeys:
            if journey.track_id == canonical:
                return journey
        raise KeyError(track_id)

    def list(self) -> tuple[DeploymentJourney, ...]:
        return self.journeys


def load_deployment_journeys(path: Path | None = None) -> DeploymentJourneyCatalog:
    """Load authored briefs, rejecting incomplete or ambiguous narrative coverage."""
    source = path or Path(__file__).parent / "data" / "deployment_journeys.json"
    data = json.loads(source.read_text())
    journeys = []
    for entry in data["scenarios"]:
        record = dict(entry)
        record["success"] = tuple(record["success"])
        record["unknowns"] = tuple(record["unknowns"])
        record["episodes"] = tuple(
            DeploymentEpisode(**{**item, "core_concepts": tuple(item["core_concepts"])})
            for item in record["episodes"]
        )
        record["acts"] = tuple(
            DeploymentAct(**{**item, "chapters": tuple(item["chapters"])})
            for item in record.get("acts", ())
        )
        journeys.append(DeploymentJourney(**record))
    volume = data["volume_storyboard"]
    storyboard = DeploymentVolumeStory(**{
        **volume,
        "chapters": tuple(DeploymentLabStory(**{
            **chapter,
            "investigations": tuple(DeploymentInvestigation(**item)
                                    for item in chapter["investigations"]),
        }) for chapter in volume["chapters"]),
    })
    catalog = DeploymentJourneyCatalog(
        version=data["version"], provenance=MappingProxyType(data["provenance"]),
        aliases=MappingProxyType(data["aliases"]), journeys=tuple(journeys),
        volume_storyboard=storyboard,
    )
    if tuple(chapter.chapter for chapter in storyboard.chapters) != tuple(range(17)):
        raise ValueError("Volume storyboard requires ordered chapters 0 through 16")
    for field in ("title", "premise", "opening", "ending"):
        if not getattr(storyboard, field).strip():
            raise ValueError(f"Volume storyboard requires nonempty {field}")
    for chapter in storyboard.chapters:
        for field in ("headline", "opening", "previous", "goal", "turning_point",
                      "dossier_entry", "transition", "qmd_bridge"):
            if not getattr(chapter, field).strip():
                raise ValueError(f"Lab storyboard requires nonempty {field}")
        if not chapter.investigations:
            raise ValueError("Lab storyboard requires investigations")
        for investigation in chapter.investigations:
            for field in ("title", "question", "interaction", "notice"):
                if not getattr(investigation, field).strip():
                    raise ValueError(f"Investigation requires nonempty {field}")
    if catalog.provenance.get("role") != "authored narrative":
        raise ValueError("Deployment journeys must be labeled authored narrative")
    if not catalog.provenance.get("coverage"):
        raise ValueError("Deployment journeys require an explicit implementation boundary")
    track_ids = [journey.track_id for journey in journeys]
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("Duplicate deployment journey track")
    if len({journey.scenario_id for journey in journeys}) != len(journeys):
        raise ValueError("Duplicate deployment journey identity")
    for journey in journeys:
        if not journey.storyline.strip():
            raise ValueError("Deployment journey requires a continuous storyline")
        if journey.version != catalog.version:
            raise ValueError("Deployment journey version differs from catalog")
        if tuple(episode.chapter for episode in journey.episodes) != tuple(range(17)):
            raise ValueError("Deployment journey requires ordered episodes 0 through 16")
        for episode in journey.episodes:
            for field in ("title", "question", "task", "handoff", "scene", "choice", "consequence", "status"):
                if not isinstance(getattr(episode, field), str) or not getattr(episode, field).strip():
                    raise ValueError(f"Deployment episode requires nonempty {field}")
        for episode in journey.episodes:
            if episode.chapter in (0, 16):
                if episode.core_concepts:
                    raise ValueError("Orientation and conclusion do not own primary concepts")
            elif not episode.core_concepts or any(not item.strip() for item in episode.core_concepts):
                raise ValueError("Chapter episodes require an explicit concept focus")
        if not journey.acts:
            raise ValueError("Deployment journey requires authored acts")
        if journey.acts:
            if len({act.id for act in journey.acts}) != len(journey.acts):
                raise ValueError("Duplicate deployment act identity")
            covered = tuple(chapter for act in journey.acts for chapter in act.chapters)
            if covered != tuple(range(1, 17)):
                raise ValueError("Deployment acts must partition ordered chapters 1 through 16")
            if any(not act.id or not act.title or not act.arc or not act.chapters for act in journey.acts):
                raise ValueError("Deployment acts require identity, title, arc, and chapters")
    for alias in catalog.aliases:
        catalog.get(alias)  # Reject dangling aliases at load time.
    return catalog
