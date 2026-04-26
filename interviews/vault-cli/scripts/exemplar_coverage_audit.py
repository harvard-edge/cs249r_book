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
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

EXEMPLAR_MIN = 3  # minimum eligible exemplars per (track, level, zone) cell


def load_corpus_from_db(db_path: Path) -> list[dict[str, Any]]:
    """Load questions from the vault.db SQLite file."""
    if not db_path.exists():
        raise SystemExit(f"error: vault.db not found at {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "SELECT track, level, zone, status, provenance FROM questions"
    )
    # Mock 'validated' as True for published questions until we have a real field
    return [dict(row) for row in cursor.fetchall()]


def is_exemplar_eligible(q: dict[str, Any]) -> bool:
    """Whether this question could serve as an exemplar today."""
    if q.get("status") != "published":
        return False
    # For legacy compatibility with Phase 0 logic, we don't have a 'validated'
    # column in SQL yet, so we assume published questions are validated.
    provenance = q.get("provenance")
    if provenance is None:
        return False
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
        "phase": 1,
        "note": (
            "Phase-1 audit: using vault.db as source of truth. eligible_exemplars "
            "count reflects the provenance field in the SQLite database."
        ),
        "exemplar_minimum_per_cell": EXEMPLAR_MIN,
        "total_cells": len(cells),
        "cells_with_gap": sum(1 for c in cells if c["gap"] > 0),
        "cells": cells,
    }


def emit_yaml(report: dict[str, Any], out: Path) -> None:
    """Write YAML without importing PyYAML (keep Phase-0/1 deps minimal)."""
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
    db_path = here / "vault" / "vault.db"
    out_path = here / "vault" / "exemplar-gaps.yaml"

    corpus = load_corpus_from_db(db_path)
    report = audit(corpus)
    emit_yaml(report, out_path)

    print(f"exemplar audit: {report['total_cells']} cells, "
          f"{report['cells_with_gap']} with gap < {EXEMPLAR_MIN} eligible")
    print(f"report written to {out_path.relative_to(here.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
