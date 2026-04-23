#!/usr/bin/env python3
"""Phase-0 exemplar-coverage audit.

Reads the current ``corpus.json`` and reports the per-(track, level, zone) cell
distribution of questions, flagging cells with fewer than 3 eligible exemplars.

As of Phase 0, the corpus does not carry a ``provenance`` field (that lands with
the YAML split in Phase 1). We therefore report raw per-cell counts AND
explicitly mark exemplar eligibility as ``unknown`` pending Phase-1 provenance
backfill. The audit shape is stable so Phase-1 re-runs slot in without
refactoring.

Referenced from ARCHITECTURE.md §14 Phase 0 milestone and REVIEWS.md R2-3 N-H3.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

EXEMPLAR_MIN = 3  # minimum eligible exemplars per (track, level, zone) cell


def load_corpus(path: Path) -> list[dict[str, Any]]:
    """Load the monolithic corpus.json. Returns the list of question records."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected top-level list; got {type(data).__name__}")
    return data


def is_exemplar_eligible(q: dict[str, Any]) -> bool:
    """Whether this question could serve as an exemplar today.

    At Phase 0 the corpus lacks a ``provenance`` field, so nothing is eligible
    until the Phase-1 YAML split and provenance backfill. This function is
    tri-state-honest: it returns False when it cannot tell.
    """
    if q.get("status") != "published":
        return False
    if q.get("validated") is not True:
        return False
    # Post-Phase-1: require provenance ∈ {human, llm-then-human-edited} AND
    # generation_meta.human_reviewed_at set. For now, we have neither.
    provenance = q.get("provenance")  # doesn't exist yet
    if provenance is None:
        return False  # honest: cannot certify as exemplar
    return provenance in {"human", "llm-then-human-edited"}


def audit(corpus: list[dict[str, Any]]) -> dict[str, Any]:
    """Group by (track, level, zone) and count total vs eligible per cell."""
    total: Counter[tuple[str, str, str]] = Counter()
    eligible: Counter[tuple[str, str, str]] = Counter()
    for q in corpus:
        track = (q.get("track") or "").lower() or "__missing__"
        level = (q.get("level") or "").lower() or "__missing__"
        zone = (q.get("zone") or "").lower() or "__missing__"
        cell = (track, level, zone)
        total[cell] += 1
        if is_exemplar_eligible(q):
            eligible[cell] += 1

    cells = []
    for cell, count in sorted(total.items()):
        track, level, zone = cell
        elig = eligible[cell]
        cells.append({
            "track": track, "level": level, "zone": zone,
            "total_questions": count,
            "eligible_exemplars": elig,
            "gap": max(0, EXEMPLAR_MIN - elig),
        })

    return {
        "phase": 0,
        "note": (
            "Phase-0 audit: no provenance field exists yet. eligible_exemplars=0 "
            "for every cell until Phase-1 backfill. 'total_questions' is the "
            "authoring pool from which exemplars will be selected."
        ),
        "exemplar_minimum_per_cell": EXEMPLAR_MIN,
        "total_cells": len(cells),
        "cells_with_gap": sum(1 for c in cells if c["gap"] > 0),
        "cells": cells,
    }


def emit_yaml(report: dict[str, Any], out: Path) -> None:
    """Write YAML without importing PyYAML (keep Phase-0 deps minimal)."""
    lines = [
        f"phase: {report['phase']}",
        f"note: {json.dumps(report['note'])}",
        f"exemplar_minimum_per_cell: {report['exemplar_minimum_per_cell']}",
        f"total_cells: {report['total_cells']}",
        f"cells_with_gap: {report['cells_with_gap']}",
        "cells:",
    ]
    for c in report["cells"]:
        lines.append(
            f"  - {{track: {c['track']}, level: {c['level']}, zone: {c['zone']}, "
            f"total_questions: {c['total_questions']}, "
            f"eligible_exemplars: {c['eligible_exemplars']}, gap: {c['gap']}}}"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    here = Path(__file__).resolve().parents[2]  # interviews/
    corpus_path = here / "vault" / "corpus.json"
    out_path = here / "vault" / "exemplar-gaps.yaml"

    if not corpus_path.exists():
        sys.stderr.write(f"error: corpus not found at {corpus_path}\n")
        return 3  # ExitCode.IO_ERROR

    corpus = load_corpus(corpus_path)
    report = audit(corpus)
    emit_yaml(report, out_path)

    print(f"exemplar audit: {report['total_cells']} cells, "
          f"{report['cells_with_gap']} with gap < {EXEMPLAR_MIN} eligible")
    print(f"report written to {out_path.relative_to(here.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
