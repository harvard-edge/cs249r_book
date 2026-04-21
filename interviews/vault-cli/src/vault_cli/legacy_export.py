"""Legacy-JSON exporter (v1.0-aware).

Regenerates the ``corpus.json`` artifact in the shape the Next.js frontend
expects (field set + array-of-items). Driven from the v1.0 YAML source,
producing a deterministic, byte-stable JSON.

v1.0 source of truth: classification is on the YAML body. The `scope`
field (dead in GUI) is no longer emitted. `chain_ids` and `chain_positions`
are rebuilt from the plural `chains: [{id, position}]` YAML list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vault_cli.loader import LoadedQuestion
from vault_cli.policy import filter_questions, load_policy


def _adapt(lq: LoadedQuestion) -> dict[str, Any]:
    """YAML question → legacy-JSON item in the shape corpus.ts expects."""
    q = lq.question

    legacy: dict[str, Any] = {
        "id": q.id,
        "title": q.title,
        "topic": q.topic,
        "competency_area": q.competency_area,
        "track": q.track,
        "level": q.level,
        "zone": q.zone,
        "bloom_level": q.bloom_level or "",
        "scenario": q.scenario,
        "status": q.status,
    }
    if q.phase:
        legacy["phase"] = q.phase

    # Chain — legacy shape: chain_ids (list) + chain_positions (dict).
    # v1.0 schema already carries multi-chain membership natively.
    chain_ids: list[str] = []
    chain_positions: dict[str, int] = {}
    for c in q.chains or []:
        chain_ids.append(c.id)
        chain_positions[c.id] = c.position
    if chain_ids:
        legacy["chain_ids"] = chain_ids
        legacy["chain_positions"] = chain_positions

    # Details.
    details: dict[str, Any] = {
        "common_mistake": q.details.common_mistake or "",
        "realistic_solution": q.details.realistic_solution,
    }
    if q.details.napkin_math:
        details["napkin_math"] = q.details.napkin_math
    if q.details.options is not None:
        details["options"] = q.details.options
        details["correct_index"] = q.details.correct_index
    if q.details.resources:
        details["resources"] = [
            {"name": r.name, "url": r.url} for r in q.details.resources
        ]
    legacy["details"] = details

    # Surface validation lineage so the site can display trust badges.
    if q.validated is not None:
        legacy["validated"] = q.validated
    if q.math_verified is not None:
        legacy["math_verified"] = q.math_verified
    if q.human_reviewed and q.human_reviewed.status != "not-reviewed":
        legacy["human_reviewed"] = {
            "status": q.human_reviewed.status,
            "by": q.human_reviewed.by,
            "date": str(q.human_reviewed.date) if q.human_reviewed.date else None,
        }

    return legacy


def emit_legacy_corpus(
    vault_dir: Path,
    loaded: list[LoadedQuestion],
    output: Path,
    *,
    publish_only: bool = True,
) -> dict[str, Any]:
    """Emit legacy-shape corpus.json from YAML source."""
    if publish_only:
        policy = load_policy(vault_dir / "release-policy.yaml")
        published_dicts = filter_questions(
            (lq.question.model_dump(mode="json") for lq in loaded),
            policy,
        )
        published_ids = {q["id"] for q in published_dicts}
        items = [lq for lq in loaded if lq.id in published_ids]
    else:
        items = loaded

    items_sorted = sorted(items, key=lambda lq: lq.id)
    legacy_items = [_adapt(lq) for lq in items_sorted]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(legacy_items, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "output": str(output),
        "count": len(legacy_items),
        "mode": "published-only" if publish_only else "all-states",
    }


__all__ = ["emit_legacy_corpus"]
