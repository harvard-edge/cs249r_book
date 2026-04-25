#!/usr/bin/env python3
"""Promote LLM-judge-PASS draft YAMLs to status:published.

Reads the latest llm_judge summary.json files, collects every question
ID that received a PASS verdict and is still a status:draft, and flips
its lifecycle fields to the canonical published shape:

    status:           draft   -> published
    validated:        false   -> true
    math_verified:    false   -> true
    math_status:      <new>   -> "CORRECT"
    math_model:       <new>   -> "gemini-3.1-pro-preview"
    math_date:        <new>   -> today
    validation_status:<new>   -> "passed"
    validation_model: <new>   -> "pydantic-1.0"
    validation_date:  <new>   -> today

Idempotent: a YAML already at status:published is left untouched.

Usage:
    python3 promote_validated.py             # use latest judge run
    python3 promote_validated.py --dry-run   # preview only
    python3 promote_validated.py --since 20260425  # only judge runs since date
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

VAULT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_DIR = VAULT_DIR / "questions"
JUDGE_DIR = VAULT_DIR / "_validation_results" / "llm_judge"

PROMOTION_FIELDS = {
    "validated": True,
    "math_verified": True,
    "math_status": "CORRECT",
    "math_model": "gemini-3.1-pro-preview",
    "validation_status": "passed",
    "validation_model": "pydantic-1.0",
}


def collect_pass_ids(since: str | None) -> set[str]:
    """Aggregate PASS verdicts across all judge runs (optionally filtered)."""
    pass_ids = set()
    for sumf in sorted(glob.glob(str(JUDGE_DIR / "*/summary.json"))):
        ts_dir = Path(sumf).parent.name
        if since and ts_dir < since:
            continue
        try:
            s = json.loads(Path(sumf).read_text())
        except Exception:
            continue
        for item in s.get("details", []):
            if item.get("verdict") == "PASS" and item.get("id"):
                pass_ids.add(item["id"])
    return pass_ids


def find_drafts_by_id(ids: set[str]) -> dict[str, Path]:
    """Return id -> path for every YAML that's currently status:draft."""
    out: dict[str, Path] = {}
    for p in QUESTIONS_DIR.glob("**/*.yaml"):
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not d or d.get("status") != "draft":
            continue
        if d.get("id") in ids:
            out[d["id"]] = p
    return out


def promote_one(path: Path, dry_run: bool) -> bool:
    """Apply the lifecycle flip in-place. Returns True on actual change."""
    text = path.read_text(encoding="utf-8")
    d = yaml.safe_load(text)
    if not d or d.get("status") != "draft":
        return False
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    d["status"] = "published"
    for k, v in PROMOTION_FIELDS.items():
        d[k] = v
    d["math_date"] = today
    d["validation_date"] = today
    if dry_run:
        return True
    path.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--since", default=None,
                        help="Only consider judge runs whose timestamp dir >= this prefix.")
    args = parser.parse_args()

    pass_ids = collect_pass_ids(args.since)
    print(f"PASS verdicts collected: {len(pass_ids)}")
    drafts = find_drafts_by_id(pass_ids)
    print(f"Of those, still draft: {len(drafts)}")

    promoted = 0
    for qid, path in sorted(drafts.items()):
        if promote_one(path, args.dry_run):
            promoted += 1
            print(f"  {'[would promote]' if args.dry_run else '[promoted]'} "
                  f"{qid:25s} {path.relative_to(VAULT_DIR.parent.parent)}")

    print(f"\n{'Would promote' if args.dry_run else 'Promoted'}: {promoted}")
    if not args.dry_run and promoted:
        print("Next: run `vault build --legacy-json` to rebuild the corpus bundle "
              "and mirror visual assets to the practice site.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
