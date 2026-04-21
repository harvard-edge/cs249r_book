"""Walk ``vault/questions/`` -> in-memory Question records (schema v1.0).

Classification comes from the YAML body, not the path. The loader enforces
one cheap structural invariant: filename prefix must match yaml.track.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from vault_cli.models import Level, Question, Track, Zone
from vault_cli.paths import (
    FILENAME_RE,
    Classification,
    track_from_path,
    vault_questions_root,
)
from vault_cli.yaml_io import load_file


@dataclass(frozen=True)
class LoadedQuestion:
    """Question plus its path. Classification is accessible via .classification."""

    question: Question
    path: Path

    @property
    def id(self) -> str:
        return self.question.id

    @property
    def classification(self) -> Classification:
        """Read the 4-axis classification from the validated YAML body."""
        return Classification(
            track=Track(self.question.track),
            level=Level(self.question.level),
            zone=Zone(self.question.zone),
        )


@dataclass(frozen=True)
class LoadError:
    path: Path
    message: str


def iter_question_files(vault_dir: Path) -> Iterator[Path]:
    root = vault_questions_root(vault_dir)
    if not root.exists():
        return
    yield from sorted(root.rglob("*.yaml"))


def load_all(vault_dir: Path) -> tuple[list[LoadedQuestion], list[LoadError]]:
    """Load every YAML under vault/questions/. Returns loaded + errors (never raises)."""
    loaded: list[LoadedQuestion] = []
    errors: list[LoadError] = []

    for path in iter_question_files(vault_dir):
        filename = path.name
        if not FILENAME_RE.match(filename):
            errors.append(LoadError(path, "filename does not match <track>-<suffix>.yaml"))
            continue

        try:
            data = load_file(path)
        except Exception as exc:  # noqa: BLE001 — report all load failures uniformly
            errors.append(LoadError(path, f"YAML load failed: {exc}"))
            continue

        try:
            q = Question.model_validate(data)
        except ValidationError as exc:
            errors.append(LoadError(path, f"schema validation failed: {exc}"))
            continue

        # Structural invariant: directory under questions/ must match yaml.track.
        # The filename prefix is a naming convention but not authoritative —
        # some corpus IDs have prefix mismatches from historical renames
        # (e.g. cloud-fill2-32021.yaml but track=edge).
        try:
            path_track = track_from_path(path, vault_dir)
        except ValueError as exc:
            errors.append(LoadError(path, str(exc)))
            continue
        if path_track != q.track:
            errors.append(
                LoadError(
                    path,
                    f"directory/track mismatch: path says {path_track!r}, "
                    f"YAML says {q.track!r}",
                )
            )
            continue

        loaded.append(LoadedQuestion(q, path))

    return loaded, errors
