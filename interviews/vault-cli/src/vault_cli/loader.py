"""Walk ``vault/questions/`` → in-memory Question records with classification."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from vault_cli.models import Question
from vault_cli.paths import (
    FILENAME_RE,
    LEGACY_FILENAME_RE,
    Classification,
    classification_from_path,
    vault_questions_root,
)
from vault_cli.yaml_io import load_file


@dataclass(frozen=True)
class LoadedQuestion:
    """Question plus the path and classification derived from it."""

    question: Question
    classification: Classification
    path: Path

    @property
    def id(self) -> str:
        return self.question.id


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
        if not (FILENAME_RE.match(filename) or LEGACY_FILENAME_RE.match(filename)):
            errors.append(LoadError(path, "filename does not match expected patterns"))
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

        try:
            classification = classification_from_path(path, vault_dir)
        except ValueError as exc:
            errors.append(LoadError(path, str(exc)))
            continue

        loaded.append(LoadedQuestion(q, classification, path))

    return loaded, errors


__all__ = ["LoadedQuestion", "LoadError", "iter_question_files", "load_all"]
