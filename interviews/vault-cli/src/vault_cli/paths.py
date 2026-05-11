"""Path utilities for the vault layout.

Classification lives in the YAML body, never in the path. The path mirrors
two body fields for navigability:

  ``vault/questions/<track>/<competency_area>/<id>.yaml``

A fast-tier invariant in the loader verifies that the YAML's ``track`` and
``competency_area`` fields match the directory segments. Any further
classification (level, zone, topic, bloom_level) is read from the YAML
directly — never from the path.

Transitional note: this module accepts both the legacy 2-segment flat layout
(``<track>/<id>.yaml``) and the 3-segment hierarchical layout
(``<track>/<area>/<id>.yaml``) so a corpus migration can proceed without a
flag-day. The path-vs-body validator in ``vault check --strict`` enforces
the hierarchical form once migration is complete.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from vault_cli.models import Level, Track, Zone

# Hierarchical layout: interviews/vault/questions/<track>/<competency_area>/<filename>
# (Legacy flat layout <track>/<filename> still tolerated by track_from_path during migration.)
PATH_SEGMENTS = ("track", "competency_area")

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
    """Extract the track from a path. Tolerates both flat and hierarchical layout.

    Returns the first directory segment under ``vault/questions/``. For a
    hierarchical path ``<track>/<area>/<file>`` this is the track; for the
    legacy flat layout ``<track>/<file>`` it's also the track.
    """
    track, _ = metadata_from_path(path, vault_dir)
    return track


def metadata_from_path(path: Path, vault_dir: Path) -> tuple[str, str | None]:
    """Extract (track, competency_area) from a path. ``area`` is None on flat layout.

    Hierarchical: ``<track>/<area>/<file.yaml>`` -> (track, area)
    Flat (legacy): ``<track>/<file.yaml>``        -> (track, None)
    """
    root = vault_questions_root(vault_dir)
    try:
        rel = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path {path} is not under {root}") from exc
    parts = rel.parts
    if len(parts) == 2:
        return parts[0], None
    if len(parts) == 3:
        return parts[0], parts[1]
    raise ValueError(
        f"path {path} has {len(parts)} segments under questions/; expected 2 "
        "(track/filename — legacy flat) or 3 (track/area/filename — hierarchical)"
    )


def path_for_question(
    vault_dir: Path,
    track: str,
    filename: str,
    *,
    competency_area: str | None = None,
) -> Path:
    """Build the path for a question.

    Hierarchical (preferred): pass ``competency_area=`` to land at
    ``<track>/<area>/<filename>``. Legacy flat layout (no area) lands at
    ``<track>/<filename>`` and is retained only for tooling that hasn't
    been migrated yet.
    """
    base = vault_questions_root(vault_dir) / track
    if competency_area is not None:
        base = base / competency_area
    return base / filename


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
    "metadata_from_path",
    "path_for_question",
    "track_from_path",
    "vault_questions_root",
]
