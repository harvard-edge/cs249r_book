"""Load the packaged, chapter-owned learning contracts without notebook copies."""
from __future__ import annotations

from copy import deepcopy
from importlib import resources
import json
from typing import Any, Mapping


# Lightweight semantic map verified against shared/styles/_brand.scss.
# These identify volumes; pass/fail/resource colors have independent meanings.
VOLUME_ACCENTS = {1: "#A51C30", 2: "#1F407A", 3: "#581C87", 4: "#065F46"}


def normalize_volume(volume: int | str) -> int:
    if isinstance(volume, bool):
        raise ValueError("Volume must be I–IV or 1–4")
    value = str(volume).strip().lower().replace("volume", "").replace("vol", "").strip()
    number = {"i": 1, "ii": 2, "iii": 3, "iv": 4}.get(value)
    if number is None and value in {"1", "2", "3", "4"}:
        number = int(value)
    if number not in VOLUME_ACCENTS:
        raise ValueError("Volume must be I–IV or 1–4")
    return number


def _text(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Learning contract needs nonempty {field}")


def validate_learning_path(contract: Mapping[str, Any]) -> None:
    """Check the visible student contract; preserve additional author provenance."""
    if not isinstance(contract, Mapping):
        raise ValueError("Learning contract must be an object")
    normalize_volume(contract.get("volume", ""))
    chapter = contract.get("chapter")
    if isinstance(chapter, bool) or not isinstance(chapter, int) or chapter < 0:
        raise ValueError("Learning contract chapter must be a nonnegative integer")
    for key in ("title", "starting_point", "deliverable", "next_part"):
        _text(contract.get(key), key)
    for key in ("learning_objectives", "prerequisites"):
        values = contract.get(key)
        if not isinstance(values, list) or (key == "learning_objectives" and not values):
            raise ValueError(f"Learning contract needs a {key} list")
        for value in values:
            _text(value, key)
    parts = contract.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Learning contract needs an ordered parts list")
    identifiers = []
    for part in parts:
        if not isinstance(part, Mapping):
            raise ValueError("Every learning part must be an object")
        for key in ("id", "label", "concept_id", "question", "goal", "artifact", "scope"):
            _text(part.get(key), f"part.{key}")
        for key in ("coverage", "limitation"):
            if key in part:
                _text(part[key], f"part.{key}")
        identifiers.append(part["id"])
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Learning part IDs must be unique")
    next_lab = contract.get("next_lab")
    if not isinstance(next_lab, Mapping):
        raise ValueError("Learning contract needs a next_lab object")
    for key in ("title", "question"):
        _text(next_lab.get(key), f"next_lab.{key}")
    narrative = contract.get("narrative", {})
    if not isinstance(narrative, Mapping):
        raise ValueError("Learning narrative must be an object")
    for key in ("project", "scene", "carry_forward", "scope"):
        if key in narrative:
            _text(narrative[key], f"narrative.{key}")


def get_learning_path(volume: int | str, chapter: int) -> dict[str, Any]:
    """Return an independent chapter contract from data/learning_paths/volN.json.

    Raises ValueError for malformed catalogs and KeyError for absent chapters.
    This is curriculum metadata, never a claim that a learner completed a lab.
    """
    number = normalize_volume(volume)
    if isinstance(chapter, bool) or not isinstance(chapter, int) or chapter < 0:
        raise ValueError("Chapter must be a nonnegative integer")
    source = resources.files("mlsysbook_labs").joinpath("data", "learning_paths", f"vol{number}.json")
    catalog = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(catalog, Mapping):
        raise ValueError("Learning catalog must be an object")
    if catalog.get("schema_version") != 1 or normalize_volume(catalog.get("volume", "")) != number:
        raise ValueError(f"Invalid Volume {number} learning catalog identity")
    entries = catalog.get("chapters")
    if not isinstance(entries, list):
        raise ValueError("Learning catalog needs a chapters list")
    seen = set()
    selected = None
    for entry in entries:
        validate_learning_path(entry)
        identity = entry["chapter"]
        if normalize_volume(entry["volume"]) != number or identity in seen:
            raise ValueError("Learning catalog has conflicting chapter identities")
        seen.add(identity)
        if identity == chapter:
            selected = entry
    if selected is None:
        raise KeyError(f"No learning contract for Volume {number}, chapter {chapter}")
    return deepcopy(selected)
