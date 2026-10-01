"""
Milestone completion ledger: the one place that decides what "complete" means.

A milestone is a sequence of parts (scripts). Some parts are required, some
are optional extensions. The ledger in ``.tito/milestones.json`` records the
outcome of every part a student runs, and a milestone counts as complete only
when every one of its required parts has a recorded pass, from this run or an
earlier one.

Why per-part records: earlier versions recorded the whole milestone as soon
as *any* run exited 0, so ``tito milestone run 03`` marked Milestone 03 done
after the XOR part alone, and ``run 06 --part 2`` marked 06 done without
Part 1 ever running (2026-09-29 release audit).

File format (schema 2) adds two keys to the historical layout::

    "schema_version": 2,
    "part_results": {
        "03": {
            "1": {"status": "passed", "timestamp": "...", "passed_at": "..."},
            "2": {"status": "failed", "timestamp": "..."}
        }
    }

``status``/``timestamp`` describe the latest run of that part; ``passed_at``
is present only once the part has passed at least once. ``completed_milestones``
and ``completion_dates`` keep their old meaning, but are written only by
:func:`record_part_result` when the completion rule is satisfied.

Legacy files (no ``schema_version``) are read without crashing and without
inventing part passes. A legacy completion is kept only for a milestone that
has a single script, because the old code could record those only after that
one script exited 0. Legacy completions of multi-part milestones are moved to
``unverified_legacy_completions`` and must be re-earned by running the
required parts.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 2
MILESTONES_FILENAME = "milestones.json"


def _registry() -> Dict[str, Dict[str, Any]]:
    from ..commands.milestone import MILESTONE_SCRIPTS  # lazy: avoid import cycle
    return MILESTONE_SCRIPTS


def _empty() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "completed_milestones": [],
        "completion_dates": {},
        "unlocked_milestones": [],
        "unlock_dates": {},
        "total_unlocked": 0,
        "achievements": [],
        "part_results": {},
    }


def part_count(milestone: Dict[str, Any]) -> int:
    """Number of runnable parts (a single-script milestone has one)."""
    return len(milestone["scripts"]) if "scripts" in milestone else 1


def required_parts(milestone: Dict[str, Any]) -> List[int]:
    """Sorted 1-based part numbers that must pass for the milestone to count."""
    return sorted(milestone.get("required_parts", [1]))


def part_name(milestone: Dict[str, Any], part: int) -> str:
    if "scripts" in milestone and 1 <= part <= len(milestone["scripts"]):
        return milestone["scripts"][part - 1]["name"]
    return milestone.get("name", f"Part {part}")


def normalize(data: Any) -> Dict[str, Any]:
    """Return a schema-2 ledger from whatever was on disk (never raises)."""
    base = _empty()
    if not isinstance(data, dict):
        return base
    merged = dict(base)
    merged.update(data)
    for key in ("completed_milestones", "unlocked_milestones", "achievements"):
        if not isinstance(merged.get(key), list):
            merged[key] = []
    for key in ("completion_dates", "unlock_dates", "part_results"):
        if not isinstance(merged.get(key), dict):
            merged[key] = {}

    if data.get("schema_version") is None:
        registry = _registry()
        kept, unverified = [], list(merged.get("unverified_legacy_completions", []))
        for mid in merged["completed_milestones"]:
            milestone = registry.get(mid)
            if milestone is not None and part_count(milestone) == 1:
                kept.append(mid)
            elif mid not in unverified:
                unverified.append(mid)
        merged["completed_milestones"] = kept
        if unverified:
            merged["unverified_legacy_completions"] = unverified
        merged["schema_version"] = SCHEMA_VERSION
    return merged


def load(tito_dir: Path) -> Dict[str, Any]:
    """Read and normalize ``<tito_dir>/milestones.json`` (missing/corrupt -> empty)."""
    path = Path(tito_dir) / MILESTONES_FILENAME
    raw: Any = None
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, IOError):
            raw = None
    return normalize(raw)


def save(tito_dir: Path, data: Dict[str, Any]) -> None:
    tito_dir = Path(tito_dir)
    tito_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(tito_dir / MILESTONES_FILENAME, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except IOError:
        pass


def passed_parts(data: Dict[str, Any], milestone_id: str) -> List[int]:
    results = data.get("part_results", {}).get(milestone_id, {})
    passed = []
    for key, record in results.items():
        if isinstance(record, dict) and record.get("passed_at"):
            try:
                passed.append(int(key))
            except ValueError:
                continue
    return sorted(passed)


def missing_required_parts(data: Dict[str, Any], milestone_id: str) -> List[int]:
    milestone = _registry()[milestone_id]
    have = set(passed_parts(data, milestone_id))
    return [p for p in required_parts(milestone) if p not in have]


def is_complete(data: Dict[str, Any], milestone_id: str) -> bool:
    """Complete under the per-part rule (``completed_milestones`` is kept in step)."""
    return milestone_id in data.get("completed_milestones", [])


def completed_ids(data: Dict[str, Any]) -> List[str]:
    return list(data.get("completed_milestones", []))


def record_part_result(tito_dir: Path, milestone_id: str, part: int, passed: bool,
                       now: Optional[str] = None) -> Dict[str, Any]:
    """Persist one part's outcome and re-evaluate completion.

    Returns the updated ledger plus ``newly_completed`` (bool) so the caller
    can decide whether to celebrate.
    """
    now = now or datetime.now().isoformat()
    data = load(tito_dir)
    parts = data["part_results"].setdefault(milestone_id, {})
    record = dict(parts.get(str(part), {}))
    record["status"] = "passed" if passed else "failed"
    record["timestamp"] = now
    if passed:
        record["passed_at"] = now
    parts[str(part)] = record

    newly_completed = False
    if not is_complete(data, milestone_id) and not missing_required_parts(data, milestone_id):
        data["completed_milestones"].append(milestone_id)
        data["completion_dates"][milestone_id] = now
        legacy = data.get("unverified_legacy_completions")
        if legacy and milestone_id in legacy:
            legacy.remove(milestone_id)
        if milestone_id not in data["unlocked_milestones"]:
            data["unlocked_milestones"].append(milestone_id)
            data["unlock_dates"][milestone_id] = now
            data["total_unlocked"] = len(data["unlocked_milestones"])
        newly_completed = True

    save(tito_dir, data)
    return {"data": data, "newly_completed": newly_completed}
