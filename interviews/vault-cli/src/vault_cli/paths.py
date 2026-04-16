"""Path utilities for vault layout.

Classification lives in the filesystem path; this module is the single source
of parse/format logic. Fast-tier invariants in validator.py enforce
lowercase + enum-valid path components.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from vault_cli.models import Level, Track, Zone

# interviews/vault/questions/<track>/<level>/<zone>/<filename>
PATH_SEGMENTS = ("track", "level", "zone")

FILENAME_RE = re.compile(
    r"^(?P<topic>[a-z0-9][a-z0-9-]*)-(?P<hash>[a-f0-9]{6})-(?P<seq>\d{4})\.yaml$"
)

# Legacy filenames used during Phase-1 split (preserves historical IDs).
LEGACY_FILENAME_RE = re.compile(r"^(?P<legacy>[a-z0-9][a-z0-9-]+)\.yaml$")


@dataclass(frozen=True)
class Classification:
    track: Track
    level: Level
    zone: Zone


def vault_questions_root(vault_dir: Path) -> Path:
    return vault_dir / "questions"


def classification_from_path(path: Path, vault_dir: Path) -> Classification:
    """Extract (track, level, zone) from a path under vault/questions/."""
    root = vault_questions_root(vault_dir)
    try:
        rel = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path {path} is not under {root}") from exc
    parts = rel.parts
    if len(parts) < 4:
        raise ValueError(
            f"path {path} has {len(parts)} segments under questions/; expected ≥4 "
            "(track/level/zone/filename)"
        )
    track, level, zone = parts[0], parts[1], parts[2]
    return Classification(
        track=Track(track),
        level=Level(level),
        zone=Zone(zone),
    )


def path_for_question(
    vault_dir: Path,
    classification: Classification,
    filename: str,
) -> Path:
    return (
        vault_questions_root(vault_dir)
        / classification.track.value
        / classification.level.value
        / classification.zone.value
        / filename
    )


def is_lowercase(component: str) -> bool:
    return component == component.lower()


__all__ = [
    "Classification",
    "FILENAME_RE",
    "LEGACY_FILENAME_RE",
    "PATH_SEGMENTS",
    "classification_from_path",
    "is_lowercase",
    "path_for_question",
    "vault_questions_root",
]
