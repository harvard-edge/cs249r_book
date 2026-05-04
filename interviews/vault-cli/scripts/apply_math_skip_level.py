#!/usr/bin/env python3
"""Apply math-only corrections for the 13 qids whose level relabel was blocked.

Phase 5 verify_math_corrections.py applied 204 of 217 Gemini-verified math
fixes. The remaining 13 were skipped because their accompanying level
relabel violated chain monotonicity or was a relabel-up (against §10 Q3
policy). The math fix itself was independently verified by Gemini-2.

This script applies napkin_math + realistic_solution + common_mistake
for those 13 qids while LEAVING the level field untouched. The level
relabel question stays in PHASE_5_UNRESOLVED.md for human review (it's
chain-team / authoring territory).

Usage:

    python3 interviews/vault-cli/scripts/apply_math_skip_level.py \\
        --merged-dir interviews/vault/_pipeline/runs/full-corpus-20260503-merged

CORPUS_HARDENING_PLAN.md Phase 5 — math-skip-level cleanup leg.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "interviews" / "vault-cli" / "src"))

from vault_cli.models import Question  # noqa: E402
from vault_cli.yaml_io import dump_str, load_file  # noqa: E402

QUESTIONS_DIR = REPO_ROOT / "interviews" / "vault" / "questions"


def find_question_file(qid: str) -> Path | None:
    for p in QUESTIONS_DIR.rglob(f"{qid}.yaml"):
        return p
    return None


def apply_math_only(body: dict, correction: dict) -> dict:
    out = json.loads(json.dumps(body))
    details = out.setdefault("details", {})
    if correction.get("napkin_math"):
        details["napkin_math"] = correction["napkin_math"]
    if correction.get("realistic_solution"):
        details["realistic_solution"] = correction["realistic_solution"]
    if correction.get("common_mistake"):
        details["common_mistake"] = correction["common_mistake"]
    return out


def write_yaml(path: Path, body: dict) -> None:
    text = dump_str(body)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged-dir", type=Path, required=True,
                    help="dir holding 01_audit.json + 04_math_applied.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="show plan without writing")
    args = ap.parse_args()

    audit = json.loads((args.merged_dir / "01_audit.json").read_text(encoding="utf-8"))
    math_applied = json.loads((args.merged_dir / "04_math_applied.json").read_text(encoding="utf-8"))

    rows_by_qid = {r["qid"]: r for r in audit["rows"]}
    target_qids = [d["qid"] for d in math_applied["dispositions"]
                   if d["result"] == "level-block"]
    print(f"target qids: {len(target_qids)}")

    dispositions: list[dict] = []
    counters = {"applied": 0, "yaml-missing": 0, "pydantic-fail": 0,
                "no-correction": 0, "no-change": 0}

    for qid in target_qids:
        row = rows_by_qid.get(qid)
        if not row or not row.get("suggested_corrections"):
            counters["no-correction"] += 1
            dispositions.append({"qid": qid, "result": "no-correction"})
            continue
        correction = row["suggested_corrections"]
        if not (correction.get("napkin_math")
                or correction.get("realistic_solution")
                or correction.get("common_mistake")):
            counters["no-change"] += 1
            dispositions.append({"qid": qid, "result": "no-change"})
            continue

        yp = find_question_file(qid)
        if not yp:
            counters["yaml-missing"] += 1
            dispositions.append({"qid": qid, "result": "yaml-missing"})
            continue
        body = load_file(yp)
        if not isinstance(body, dict):
            continue

        proposed = apply_math_only(body, correction)
        if proposed == body:
            counters["no-change"] += 1
            dispositions.append({"qid": qid, "result": "no-change"})
            continue

        try:
            Question.model_validate(proposed)
        except Exception as e:
            counters["pydantic-fail"] += 1
            dispositions.append({"qid": qid, "result": "pydantic-fail",
                                  "error": str(e)[:300]})
            continue

        if args.dry_run:
            print(f"  [dry] {qid}: would update math fields, level "
                  f"{body.get('level')!r} unchanged")
        else:
            write_yaml(yp, proposed)

        counters["applied"] += 1
        dispositions.append({"qid": qid, "result": "applied",
                              "kept_level": body.get("level")})

    print(f"\ncounters: {counters}")

    out_path = args.merged_dir / "05_math_skip_level.json"
    if not args.dry_run:
        out_path.write_text(json.dumps({
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "summary": counters,
            "dispositions": dispositions,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out_path}")

    return 0 if counters["applied"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
