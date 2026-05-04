#!/usr/bin/env python3
"""Apply marker-compliant common_mistake / napkin_math corrections for
published qids whose proposed format fix got skipped during Phase 5.

Phase 6 (schema tightening) wants a LinkML pattern requiring the
authoring markers (Pitfall/Rationale/Consequence and
Assumptions/Calculations/Conclusion). A pre-flight survey on
2026-05-04 found ~77 published YAMLs that still have malformed
markers. About 49 of those have a marker-compliant proposed cm/nm fix
in the audit's suggested_corrections, but mass_apply skipped them
because they were entangled with a level relabel that hit relabel-up
or chain-monotonicity-block, OR they had a realistic_solution that
classified the row as high-risk.

This script applies ONLY the cm/nm fields when:
  1. The current YAML's field is malformed (fails the marker regex), AND
  2. The audit's proposed value IS marker-compliant.

It deliberately ignores level (still a chain-team / authoring decision)
and realistic_solution (handled by the math-verify pipeline).

Usage:

    python3 interviews/vault-cli/scripts/apply_format_skip_level.py \\
        --audit interviews/vault/_pipeline/runs/full-corpus-20260503-merged/01_audit.json

CORPUS_HARDENING_PLAN.md Phase 6 — format-skip-level cleanup.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "interviews" / "vault-cli" / "src"))

from vault_cli.models import Question  # noqa: E402
from vault_cli.yaml_io import dump_str, load_file  # noqa: E402

QUESTIONS_DIR = REPO_ROOT / "interviews" / "vault" / "questions"

CM_PATTERN = re.compile(
    r"(?s).*\*\*The Pitfall:\*\*.*\*\*The Rationale:\*\*.*\*\*The Consequence:\*\*.*"
)
NM_PATTERN = re.compile(
    r"(?s).*\*\*Assumptions.*\*\*Calculations:\*\*.*\*\*Conclusion.*"
)


def find_question_file(qid: str) -> Path | None:
    for p in QUESTIONS_DIR.rglob(f"{qid}.yaml"):
        return p
    return None


def write_yaml(path: Path, body: dict) -> None:
    text = dump_str(body)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", type=Path, required=True,
                    help="01_audit.json with suggested_corrections")
    ap.add_argument("--dry-run", action="store_true",
                    help="show plan without writing")
    args = ap.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    rows_by_qid = {r["qid"]: r for r in audit["rows"]}

    counters: Counter[str] = Counter()
    dispositions: list[dict] = []

    for yp in QUESTIONS_DIR.rglob("*.yaml"):
        body = load_file(yp)
        if not isinstance(body, dict):
            continue
        if body.get("status") != "published":
            continue
        qid = body.get("id")
        row = rows_by_qid.get(qid)
        if not row:
            continue
        sc = row.get("suggested_corrections") or {}

        details = body.get("details") or {}
        cm_now = (details.get("common_mistake") or "").strip()
        nm_now = (details.get("napkin_math") or "").strip()
        cm_bad = bool(cm_now) and not CM_PATTERN.match(cm_now)
        nm_bad = bool(nm_now) and not NM_PATTERN.match(nm_now)
        if not (cm_bad or nm_bad):
            continue

        proposed = json.loads(json.dumps(body))
        pdetails = proposed.setdefault("details", {})
        applied_fields: list[str] = []

        if cm_bad and sc.get("common_mistake") and CM_PATTERN.match(sc["common_mistake"]):
            pdetails["common_mistake"] = sc["common_mistake"]
            applied_fields.append("common_mistake")
        if nm_bad and sc.get("napkin_math") and NM_PATTERN.match(sc["napkin_math"]):
            pdetails["napkin_math"] = sc["napkin_math"]
            applied_fields.append("napkin_math")

        if not applied_fields:
            counters["no-marker-compliant-fix"] += 1
            dispositions.append({"qid": qid, "result": "no-marker-compliant-fix",
                                  "cm_bad": cm_bad, "nm_bad": nm_bad})
            continue

        try:
            Question.model_validate(proposed)
        except Exception as e:
            counters["pydantic-fail"] += 1
            dispositions.append({"qid": qid, "result": "pydantic-fail",
                                  "error": str(e)[:300]})
            continue

        if args.dry_run:
            print(f"  [dry] {qid}: would apply {applied_fields}")
        else:
            write_yaml(yp, proposed)
        counters["applied"] += 1
        dispositions.append({"qid": qid, "result": "applied",
                              "fields": applied_fields})

    print(f"\ncounters: {dict(counters)}")

    out_path = args.audit.parent / "06_format_skip_level.json"
    if not args.dry_run:
        out_path.write_text(json.dumps({
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "summary": dict(counters),
            "dispositions": dispositions,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out_path}")

    return 0 if counters["applied"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
