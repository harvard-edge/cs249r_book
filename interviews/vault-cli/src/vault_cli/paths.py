"""Path utilities for the v1.0 flat-by-track vault layout.

v1.0: classification lives in the YAML. The filesystem encodes only `track`
for navigability: ``vault/questions/<track>/<id>.yaml``. A fast-tier
invariant in the loader verifies that the YAML's ``track`` field matches
the directory the file lives under.

Any further classification (level, zone, topic, competency_area) is read
from the YAML directly — never from the path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from vault_cli.models import Level, Track, Zone

# Flat-by-track layout: interviews/vault/questions/<track>/<filename>
PATH_SEGMENTS = ("track",)

# Legacy filenames from the imported corpus: cloud-0185.yaml, tinyml-cell-13212.yaml
FILENAME_RE = re.compile(
    r"^(?P<track>[a-z]+)-(?P<suffix>[a-z0-9-]+)\.yaml$"
)


@dataclass(frozen=True)
class Classification:
    """Full 4-axis classification as read from the YAML body."""
    track: Track
    level: Level
    zone: Zone


def vault_questions_root(vault_dir: Path) -> Path:
    return vault_dir / "questions"


def track_from_path(path: Path, vault_dir: Path) -> str:
    """Extract just the track from a path (v1.0 — one path segment)."""
    root = vault_questions_root(vault_dir)
    try:
        rel = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path {path} is not under {root}") from exc
    parts = rel.parts
    if len(parts) != 2:
        raise ValueError(
            f"path {path} has {len(parts)} segments under questions/; expected 2 "
            "(track/filename) in v1.0 flat-by-track layout"
        )
    return parts[0]


def path_for_question(
    vault_dir: Path,
    track: str,
    filename: str,
) -> Path:
    """Build the path for a question given its track and filename."""
    return vault_questions_root(vault_dir) / track / filename


def is_lowercase(component: str) -> bool:
    return component == component.lower()


def filename_matches_track(filename: str, track: str) -> bool:
    """Fast-tier invariant: filename prefix must match track.

    This is the cheap structural check that catches mis-filed questions
    without parsing the YAML body.
    """
    m = FILENAME_RE.match(filename)
    if not m:
        return False
    return m.group("track") == track


__all__ = [
    "Classification",
    "FILENAME_RE",
    "PATH_SEGMENTS",
    "filename_matches_track",
    "is_lowercase",
    "path_for_question",
    "track_from_path",
    "vault_questions_root",
]
