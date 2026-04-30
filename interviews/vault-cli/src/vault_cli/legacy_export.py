"""Legacy-JSON exporter (v1.1: chains as sidecar metadata).

Regenerates the ``corpus.json`` artifact in the shape the Next.js frontend
expects (field set + array-of-items). Driven from the v1.0 YAML source
plus the ``chains.json`` sidecar (authoritative chain registry), producing
a deterministic, byte-stable JSON.

v1.1 architecture: chain membership is a SIDECAR — chains.json is the
authoritative source. Question YAMLs no longer carry a ``chains:`` field.
The exporter joins YAML + chains.json to produce per-question chain_ids
and chain_positions in the runtime corpus.json. This means rebuilding
chains (e.g., via the Gemini chain-builder) only touches one file, not
2k+ YAMLs.
"""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path
from typing import Any

from vault_cli.loader import LoadedQuestion
from vault_cli.policy import filter_questions, load_policy


def _build_chain_index(vault_dir: Path) -> dict[str, dict[str, int]]:
    """Map qid -> {chain_id: position} from chains.json sidecar.

    Position is derived from the array index inside chains.json — chains
    are listed in pedagogical order, so position 0 is the first member.
    """
    chains_path = vault_dir / "chains.json"
    if not chains_path.exists():
        return {}
    out: dict[str, dict[str, int]] = {}
    for ch in json.loads(chains_path.read_text(encoding="utf-8")):
        cid = ch.get("chain_id") or ch.get("id")
        if not cid: continue
        for pos, member in enumerate(ch.get("questions", [])):
            qid = member.get("id")
            if not qid: continue
            out.setdefault(qid, {})[cid] = pos
    return out


def _adapt(lq: LoadedQuestion, chain_index: dict[str, dict[str, int]]) -> dict[str, Any]:
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
    if q.question:
        legacy["question"] = q.question
    if q.visual is not None:
        visual_out: dict[str, Any] = {
            "kind": q.visual.kind,
            "path": q.visual.path,
            "alt": q.visual.alt,
        }
        if q.visual.caption:
            visual_out["caption"] = q.visual.caption
        legacy["visual"] = visual_out

    # Chain — sidecar-driven. chain_ids/chain_positions are computed by
    # joining the YAML's id with chains.json. The YAML's chains: field
    # (if still present during transition) is ignored — chains.json wins.
    member_of = chain_index.get(q.id, {})
    if member_of:
        legacy["chain_ids"] = sorted(member_of.keys())
        legacy["chain_positions"] = dict(member_of)

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
    """Emit legacy-shape corpus.json from YAML source + chains.json sidecar."""
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

    chain_index = _build_chain_index(vault_dir)
    items_sorted = sorted(items, key=lambda lq: lq.id)
    legacy_items = [_adapt(lq, chain_index) for lq in items_sorted]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(legacy_items, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Also emit a summary-only companion next to the full corpus for
    # callers that only need classification axes (~80% smaller). The
    # site bundles this file instead of the full corpus.json; scenario
    # and details are fetched on demand from the worker via
    # useFullQuestion() / getQuestionFullDetail().
    summary_output = output.with_name(output.stem + "-summary" + output.suffix)
    summary_items = [_to_summary(item) for item in legacy_items]
    summary_output.write_text(
        json.dumps(summary_items, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "output": str(output),
        "summary_output": str(summary_output),
        "count": len(legacy_items),
        "mode": "published-only" if publish_only else "all-states",
    }


def _to_summary(item: dict[str, Any]) -> dict[str, Any]:
    """Return the summary form of a legacy corpus item.

    Keeps every classification axis and chain membership. Replaces the
    heavy scenario + details.{common_mistake,realistic_solution,napkin_math}
    with empty-string stubs so the TypeScript Question interface stays
    backward-compatible. MCQ options + correct_index are PRESERVED
    (scoring logic reads them synchronously).
    """
    drop_fields = {"scenario", "details"}
    summary: dict[str, Any] = {k: v for k, v in item.items() if k not in drop_fields}
    summary["scenario"] = ""
    original_details = item.get("details", {}) or {}
    details_stub: dict[str, Any] = {
        "common_mistake": "",
        "realistic_solution": "",
        "napkin_math": "",
    }
    if original_details.get("options") is not None:
        details_stub["options"] = original_details["options"]
    if original_details.get("correct_index") is not None:
        details_stub["correct_index"] = original_details["correct_index"]
    summary["details"] = details_stub
    return summary


def copy_visual_assets(vault_dir: Path, staffml_public_dir: Path) -> dict[str, Any]:
    """Copy `interviews/vault/visuals/<track>/*.svg` → `<staffml>/public/question-visuals/<track>/`.

    The Next.js frontend serves static assets from ``public/`` at
    ``/question-visuals/<track>/<file>.svg``. This function mirrors
    the track-sharded directory layout so the same relative filename
    used in the YAML's ``visual.path`` field resolves without
    transformation.

    Overwrites destination files on every run — the vault is the
    source of truth. Removes destination files whose source no longer
    exists so renames/deletions propagate. Returns counts for the
    build summary.
    """
    import shutil

    source = vault_dir / "visuals"
    dest = staffml_public_dir / "question-visuals"
    copied = 0
    deleted = 0

    if not source.exists():
        return {"copied": 0, "deleted": 0, "note": "no visuals directory"}

    # Mirror from source into dest.
    dest.mkdir(parents=True, exist_ok=True)
    source_files: set[Path] = set()
    for svg in source.rglob("*.svg"):
        rel = svg.relative_to(source)
        source_files.add(rel)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != svg.read_bytes():
            shutil.copy2(svg, target)
            copied += 1

    # Prune destination files that no longer have a source.
    for existing in dest.rglob("*.svg"):
        rel = existing.relative_to(dest)
        if rel not in source_files:
            existing.unlink()
            deleted += 1

    return {"copied": copied, "deleted": deleted, "total_assets": len(source_files)}


def emit_manifest(
    loaded: list[LoadedQuestion],
    output: Path,
    *,
    release_id: str,
    release_hash: str,
    schema_version: str,
    policy_version: str,
    published_count: int,
    chain_count: int,
    concept_count: int = 87,
    area_count: int = 13,
    taxonomy_version: str = "87-topics",
) -> dict[str, Any]:
    """Emit `vault-manifest.json` — the staffml site's view of the release.

    Single source of truth contract: the upstream authority is
    ``releases/<release_id>/release.json`` (and the corresponding
    release_metadata row); this manifest is a build-time projection of
    that artifact PLUS the loaded set's distributions. Anything that
    needs to know "what release is the site running" reads this file.

    The site never reads ``release_id`` from environment variables for
    display: ``NEXT_PUBLIC_VAULT_RELEASE`` is reserved as an override
    used to point local builds at a different worker, not to label the
    bundle. That kept the field drift-prone for months — now closed.
    """
    from collections import Counter
    from datetime import datetime

    if not release_id:
        raise ValueError("emit_manifest: release_id is required")
    if not release_hash or len(release_hash) < 16:
        raise ValueError("emit_manifest: release_hash must be the full hex digest")

    track_dist = Counter(lq.question.track for lq in loaded
                         if lq.question.status == "published")
    level_dist = Counter(lq.question.level for lq in loaded
                         if lq.question.status == "published")

    manifest = {
        "releaseId": release_id,
        "releaseHash": release_hash,
        "schemaVersion": schema_version,
        "policyVersion": policy_version,
        "buildDate": datetime.now(UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "questionCount": published_count,
        "chainCount": chain_count,
        "conceptCount": concept_count,
        "trackDistribution": dict(track_dist),
        "levelDistribution": dict(level_dist),
        "areaCount": area_count,
        "taxonomyVersion": taxonomy_version,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "output": str(output),
        "releaseId": release_id,
        "questionCount": published_count,
        "chainCount": chain_count,
    }


__all__ = ["emit_legacy_corpus", "emit_manifest", "copy_visual_assets"]
