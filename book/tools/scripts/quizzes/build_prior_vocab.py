#!/usr/bin/env python3
"""Build the cumulative prior-vocabulary context for a target chapter.

Walks the Vol1 → Vol2 reading order and, for chapter N, produces the
union of every glossary term first defined in chapters 1..N-1. The
resulting JSON is passed to that chapter's quiz-generation sub-agent so
it knows which terms it may assume vs. which it must treat as novel to
this chapter.

Usage
-----
    python3 build_prior_vocab.py vol1 training
    python3 build_prior_vocab.py vol2 distributed_training > _context/vol2/distributed_training/prior_vocab.json

The reading order is extracted from
``book/quarto/config/_quarto-html-vol{1,2}.yml`` and must match the
actual sidebar sequence; to update, rerun ``extract_reading_order.sh``
(or edit :data:`READING_ORDER` below by hand).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Source: extracted from book/quarto/config/_quarto-html-vol{1,2}.yml on
# 2026-04-23 (git sha 5eaaa7642). If the sidebar is reordered, update
# this list and regenerate all prior_vocab.json files.
READING_ORDER: list[tuple[str, str]] = [
    # Vol1 (16 content chapters)
    ("vol1", "introduction"),
    ("vol1", "ml_systems"),
    ("vol1", "ml_workflow"),
    ("vol1", "data_engineering"),
    ("vol1", "nn_computation"),
    ("vol1", "nn_architectures"),
    ("vol1", "frameworks"),
    ("vol1", "training"),
    ("vol1", "data_selection"),
    ("vol1", "optimizations"),       # .qmd is model_compression.qmd; glossary uses dir name
    ("vol1", "hw_acceleration"),
    ("vol1", "benchmarking"),
    ("vol1", "model_serving"),
    ("vol1", "ml_ops"),
    ("vol1", "responsible_engr"),
    ("vol1", "conclusion"),
    # Vol2 (17 content chapters) — inherits all of Vol1's prior vocab
    ("vol2", "introduction"),
    ("vol2", "compute_infrastructure"),
    ("vol2", "network_fabrics"),
    ("vol2", "data_storage"),
    ("vol2", "distributed_training"),
    ("vol2", "collective_communication"),
    ("vol2", "fault_tolerance"),
    ("vol2", "fleet_orchestration"),
    ("vol2", "performance_engineering"),
    ("vol2", "inference"),
    ("vol2", "edge_intelligence"),
    ("vol2", "ops_scale"),
    ("vol2", "security_privacy"),
    ("vol2", "robust_ai"),
    ("vol2", "sustainable_ai"),
    ("vol2", "responsible_ai"),
    ("vol2", "conclusion"),
]

BASE = Path(__file__).resolve().parents[4] / "quarto" / "contents"


def _glossary_path(vol: str, chap: str) -> Path | None:
    """Return the glossary JSON path for a chapter, or ``None`` if absent.

    The directory name and the glossary file stem usually match, but some
    chapters (``optimizations``) name the glossary after the ``.qmd``
    stem instead. Fall back to any ``*_glossary.json`` in the chapter
    directory.
    """
    chap_dir = BASE / vol / chap
    direct = chap_dir / f"{chap}_glossary.json"
    if direct.is_file():
        return direct
    candidates = sorted(chap_dir.glob("*_glossary.json"))
    return candidates[0] if candidates else None


def prior_vocab_for(target_idx: int) -> list[dict]:
    """Union of every glossary term introduced in chapters before ``target_idx``.

    Deduplicates by lowercased term; the first chapter to introduce a
    term wins, and later re-definitions are ignored.
    """
    terms: list[dict] = []
    seen: set[str] = set()
    for vol, chap in READING_ORDER[:target_idx]:
        gloss = _glossary_path(vol, chap)
        if gloss is None:
            continue
        try:
            data = json.loads(gloss.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"warn: could not parse {gloss}: {e}", file=sys.stderr)
            continue
        for entry in data.get("terms", []):
            term = entry.get("term", "").strip()
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            terms.append(
                {
                    "term": term,
                    "definition": entry.get("definition", "").strip(),
                    "first_seen": f"{vol}/{chap}",
                }
            )
    return terms


def build(vol: str, chap: str) -> dict:
    try:
        target_idx = next(
            i for i, (v, c) in enumerate(READING_ORDER) if v == vol and c == chap
        )
    except StopIteration:
        raise SystemExit(f"error: {vol}/{chap} not in reading order")
    terms = prior_vocab_for(target_idx)
    return {
        "target_chapter": f"{vol}/{chap}",
        "position_in_reading_order": target_idx + 1,
        "total_chapters": len(READING_ORDER),
        "prior_term_count": len(terms),
        "terms": terms,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: build_prior_vocab.py <vol1|vol2> <chapter>", file=sys.stderr)
        return 2
    vol, chap = argv[1], argv[2]
    print(json.dumps(build(vol, chap), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
