"""Walk ``vault/questions/`` -> in-memory Question records (schema v1.1).

Classification comes from the YAML body, not the path. The loader enforces
one cheap structural invariant: filename prefix must match yaml.track.

v1.1: chains.json is the authoritative chain registry. The loader joins
sidecar chain data onto each LoadedQuestion at load time so all existing
readers (q.chains) continue to work without per-call sidecar lookups.
"""

from __future__ import annotations

import json as _json
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


def _load_chain_index(vault_dir: Path) -> dict[str, list[tuple[str, int]]]:
    """qid -> [(chain_id, position), ...] from chains.json."""
    chains_path = vault_dir / "chains.json"
    if not chains_path.exists():
        return {}
    out: dict[str, list[tuple[str, int]]] = {}
    for ch in _json.loads(chains_path.read_text(encoding="utf-8")):
        cid = ch.get("chain_id") or ch.get("id")
        if not cid:
            continue
        for pos, member in enumerate(ch.get("questions", [])):
            qid = member.get("id")
            if not qid:
                continue
            out.setdefault(qid, []).append((cid, pos))
    return out


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
    """Load every YAML under vault/questions/. Returns loaded + errors (never raises).

    v1.1: chains are joined from chains.json after parsing. Question YAMLs
    no longer carry a chains: field; if one is present (transitional) it is
    overwritten by sidecar data so chains.json wins consistently.
    """
    loaded: list[LoadedQuestion] = []
    errors: list[LoadError] = []
    chain_index = _load_chain_index(vault_dir)

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

        # Inject chains from sidecar BEFORE schema validation. Anything in the
        # YAML's chains: field (transitional) is overwritten — sidecar wins.
        sidecar_chains = chain_index.get(data.get("id", ""), [])
        data["chains"] = [{"id": cid, "position": pos} for cid, pos in sidecar_chains]

        try:
            q = Question.model_validate(data)
        except ValidationError as exc:
            errors.append(LoadError(path, f"schema validation failed: {exc}"))
            continue

        # Structural invariant: directory under questions/ must match yaml.track.
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
