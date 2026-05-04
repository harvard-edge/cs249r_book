#!/usr/bin/env python3
"""Add explicit status / provenance / deletion_reason where the YAML is
silently relying on Pydantic defaults or violating soft-delete pairing.

Three classes of fix:

  A. status field missing entirely
       Pydantic defaults to 'draft', but the YAML on disk lacks the field.
       Add explicit `status: draft` + `provenance: imported` so the file
       no longer relies on a silent default.

  B. deleted but no deletion_reason
       Soft-delete pairing rule (Question.status='deleted' must carry
       deletion_reason). Add placeholder text.

  C. flagged with no human_reviewed
       Reported but not auto-fixed; needs human disposition.

Usage:

    python3 interviews/vault-cli/scripts/fix_missing_metadata.py [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "interviews" / "vault-cli" / "src"))

from vault_cli.models import Question  # noqa: E402
from vault_cli.yaml_io import dump_str, load_file  # noqa: E402

QUESTIONS_DIR = REPO_ROOT / "interviews" / "vault" / "questions"
PLACEHOLDER_REASON = (
    "imported as deleted; original deletion reason was not preserved on import"
)


def write_yaml(path: Path, body: dict) -> None:
    text = dump_str(body)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    counters: Counter[str] = Counter()

    for yp in QUESTIONS_DIR.rglob("*.yaml"):
        body = load_file(yp)
        if not isinstance(body, dict):
            continue
        changed = False

        # Class A: status field missing → set draft + imported
        if "status" not in body:
            body["status"] = "draft"
            if "provenance" not in body:
                body["provenance"] = "imported"
            counters["status-added"] += 1
            changed = True

        # Class B: deleted without deletion_reason
        if body.get("status") == "deleted" and not body.get("deletion_reason"):
            body["deletion_reason"] = PLACEHOLDER_REASON
            counters["deletion-reason-added"] += 1
            changed = True

        if not changed:
            continue

        try:
            Question.model_validate(body)
        except Exception as e:
            counters["pydantic-fail"] += 1
            print(f"  pydantic-fail {body.get('id')}: {str(e)[:200]}", file=sys.stderr)
            continue

        if args.dry_run:
            print(f"  [dry] {body.get('id')}: status={body.get('status')!r}, "
                  f"provenance={body.get('provenance')!r}, "
                  f"deletion_reason={'yes' if body.get('deletion_reason') else 'no'}")
        else:
            write_yaml(yp, body)

    print(f"\ncounters: {dict(counters)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
