"""Legacy-JSON exporter.

Regenerates the `corpus.json` artifact in the pre-migration shape that the
existing Next.js ``corpus.ts`` expects (field set + array-of-items layout).
Driven from the YAML source so ``vault build --legacy-json`` produces a
deterministic, byte-stable JSON that can be CI-diffed against the committed
``interviews/staffml/src/data/corpus.json``.

This closes ARCHITECTURE.md §11.1's "corpus.json becomes a generated
artifact" contract without requiring Phase-4 cutover: the live site keeps
reading the bundled JSON; we just prove it's reproducible from YAML.

Maps to legacy fields that the new schema dropped:
    competency_area  ← topic       (they were ~the same thing)
    bloom_level      ← zone → bloom (rollup mapping, same as export-paper)
    scope            ← track        (1:1)
    chain_positions  ← chain        (reshape {id, position} → {id: position - 1})
    chain_ids        ← [chain.id] if chain else None
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vault_cli.loader import LoadedQuestion
from vault_cli.policy import filter_questions, load_policy

# Zone → Bloom level mapping (mirrors export-paper).
_ZONE_TO_BLOOM = {
    "recall":        "remember",
    "fluency":       "understand",
    "implement":     "apply",
    "specification": "apply",
    "analyze":       "analyze",
    "diagnosis":     "analyze",
    "design":        "create",
    "evaluation":    "evaluate",
}


def _adapt(lq: LoadedQuestion) -> dict[str, Any]:
    """YAML question → legacy-JSON item in the shape corpus.ts expects."""
    q = lq.question
    c = lq.classification

    legacy: dict[str, Any] = {
        "id": q.id,
        "title": q.title,
        "topic": q.topic,
        "competency_area": q.topic,                    # legacy alias
        "scope": c.track.value,                         # legacy alias
        "track": c.track.value,
        "level": c.level.value.upper(),                 # legacy used "L1" uppercase
        "zone": c.zone.value,
        "bloom_level": _ZONE_TO_BLOOM.get(c.zone.value, "remember"),
        "scenario": q.scenario,
        "status": q.status.value,
    }

    # Chain — legacy used dict-valued chain_positions keyed by chain_id.
    if q.chain is not None:
        legacy["chain_ids"] = [q.chain.id]
        # Legacy was 0-indexed; new schema is 1-indexed. Undo the +1 so
        # legacy consumers see the same numbers they did pre-migration.
        legacy["chain_positions"] = {q.chain.id: q.chain.position - 1}
    else:
        legacy["chain_ids"] = []
        legacy["chain_positions"] = {}

    # Details — preserve common_mistake + realistic_solution + napkin_math +
    # deep_dive_{title,url} as flat keys on `details`.
    details: dict[str, Any] = {
        "common_mistake": q.details.common_mistake or "",
        "realistic_solution": q.details.realistic_solution,
    }
    if q.details.napkin_math:
        details["napkin_math"] = q.details.napkin_math
    if q.details.deep_dive:
        details["deep_dive_title"] = q.details.deep_dive.title
        details["deep_dive_url"] = q.details.deep_dive.url
    legacy["details"] = details

    return legacy


def emit_legacy_corpus(
    vault_dir: Path,
    loaded: list[LoadedQuestion],
    output: Path,
    *,
    publish_only: bool = True,
) -> dict[str, Any]:
    """Emit legacy-shape corpus.json from YAML source.

    ``publish_only=True`` (default) filters through release-policy.yaml so
    the emitted file matches what would ship to production (9,199 published).
    Set False to dump everything including drafts + deprecated.
    """
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

    # Sort by id for byte-stable output across runs.
    items_sorted = sorted(items, key=lambda lq: lq.id)
    legacy_items = [_adapt(lq) for lq in items_sorted]

    # Canonical JSON: sort_keys recursively, LF, UTF-8.
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
