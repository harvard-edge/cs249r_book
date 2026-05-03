#!/usr/bin/env python3
"""Merge multiple audit_corpus_batched output dirs into one canonical run.

Phase 4's parallel runs split work by --tracks, each writing to its
own output dir. This script merges them into a single 01_audit.json
suitable for downstream consumers (apply_corrections.py,
summarize_audit.py).

Merge rule: for each qid, prefer the row that has more useful data:
  1. Reject rows with format_compliance == "error" if a non-error row
     for the same qid exists in any input.
  2. Among non-error rows, prefer rows with non-empty
     suggested_corrections (more recent --propose-fixes pass).
  3. Otherwise take the latest seen.

Usage:

    python3 interviews/vault-cli/scripts/merge_audit_runs.py \\
        --inputs interviews/vault/_pipeline/runs/full-corpus-20260503 \\
                 interviews/vault/_pipeline/runs/full-corpus-20260503-mobile \\
                 interviews/vault/_pipeline/runs/full-corpus-20260503-tinyml \\
        --output interviews/vault/_pipeline/runs/full-corpus-20260503-merged
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def _row_priority(row: dict) -> tuple[int, int, int]:
    """Higher tuple = better. Used to pick between two rows for same qid.

    Priority axes:
      1. Non-error format_compliance (1) > error (0)
      2. Has suggested_corrections (1) > none (0)
      3. Audit gate count (number of non-null gate fields) — favors
         rows with denser data.
    """
    is_non_error = 1 if row.get("format_compliance") != "error" else 0
    has_corrections = 1 if row.get("suggested_corrections") else 0
    gate_density = sum(
        1 for k in ("format_compliance", "level_fit", "coherence",
                     "math_correct", "title_quality")
        if row.get(k) is not None
    )
    return (is_non_error, has_corrections, gate_density)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", type=Path, required=True,
                    help="run dirs to merge")
    ap.add_argument("--output", type=Path, required=True,
                    help="output dir (will create + write 01_audit.json)")
    args = ap.parse_args()

    by_qid: dict[str, dict] = {}
    by_qid_source: dict[str, str] = {}
    n_total_rows = 0
    n_replaced = 0

    for indir in args.inputs:
        path = indir / "01_audit.json"
        if not path.exists():
            print(f"  skip: {path} (no 01_audit.json)")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"  skip: {path} (JSON decode error: {e})")
            continue
        rows = data.get("rows", [])
        print(f"  {indir.name}: {len(rows)} rows")
        n_total_rows += len(rows)
        for r in rows:
            qid = r.get("qid")
            if not qid:
                continue
            if qid not in by_qid:
                by_qid[qid] = r
                by_qid_source[qid] = indir.name
            else:
                # Prefer the higher-priority row.
                old = by_qid[qid]
                if _row_priority(r) > _row_priority(old):
                    by_qid[qid] = r
                    by_qid_source[qid] = indir.name
                    n_replaced += 1

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / "01_audit.json"
    merged_rows = sorted(by_qid.values(), key=lambda r: r.get("qid") or "")
    out = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": "gemini-3.1-pro-preview",
        "merged_from": [str(p) for p in args.inputs],
        "rows_in": n_total_rows,
        "rows_unique": len(merged_rows),
        "rows_replaced_during_merge": n_replaced,
        "rows": merged_rows,
    }
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"\nmerged: {n_total_rows} input rows → {len(merged_rows)} unique qids")
    print(f"  replaced during merge: {n_replaced}")

    # Quick sanity: per-track count
    from collections import Counter
    by_track = Counter(r["qid"].split("-")[0] for r in merged_rows)
    print("\nper-track:")
    for t in sorted(by_track):
        print(f"  {t}: {by_track[t]}")

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
