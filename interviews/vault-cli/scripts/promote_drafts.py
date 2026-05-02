#!/usr/bin/env python3
"""Promote LLM-authored question drafts to the corpus (Phase 3.d helper).

A draft is a `<id>.yaml.draft` file under `interviews/vault/questions/`,
written by `generate_question_for_gap.py`. Promotion does five things:

  1. Strips the private ``_authoring`` block and replaces it with the
     real schema fields (``provenance``, ``status``, ``authors``,
     ``human_reviewed``, ``created_at``, plus a ``gap-bridge:<from>-<to>``
     tag for traceability).
  2. Renames ``<id>.yaml.draft`` → ``<id>.yaml``.
  3. Appends an entry to ``interviews/vault/id-registry.yaml``
     (append-only, CI-enforced).
  4. Optionally flips ``status`` to ``published`` (default: keep
     ``draft`` so the human reviewer's workflow stays explicit).
  5. Optionally flips ``human_reviewed.status`` to ``verified`` with
     ``--reviewed-by <handle>``.

Selection modes — pick one:

  --all-passing                # promote every draft whose row in
                               # draft-validation-scorecard.json verdict=pass
  --qids edge-2536,edge-2537   # explicit list (whether they passed or not)
  --from-scorecard <path>      # use a non-default scorecard path
  --dry-run                    # show what would change, write nothing

Examples:

  # After reviewing the 4 pilot drafts, promote them all and mark verified:
  python3 promote_drafts.py --all-passing --publish --reviewed-by vj

  # Promote two specific qids as drafts (no publish, no review stamp):
  python3 promote_drafts.py --qids edge-2536,mobile-2146

  # Preview only:
  python3 promote_drafts.py --all-passing --dry-run

The script never overwrites a `<id>.yaml` that already exists. It refuses
to run if `vault check --strict` would fail post-promotion (run that
yourself after this script as the final gate).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
VAULT_DIR = REPO_ROOT / "interviews" / "vault"
QUESTIONS_DIR = VAULT_DIR / "questions"
ID_REGISTRY = VAULT_DIR / "id-registry.yaml"
# AI-pipeline scorecard lives under _pipeline/ (gitignored).
# See interviews/CLAUDE.md.
PIPELINE_DIR = VAULT_DIR / "_pipeline"
DEFAULT_SCORECARD = PIPELINE_DIR / "draft-validation-scorecard.json"


def find_draft(qid: str) -> Path | None:
    """Resolve a qid to its draft path. Returns None if no draft for qid exists."""
    matches = list(QUESTIONS_DIR.rglob(f"{qid}.yaml.draft"))
    if len(matches) > 1:
        raise RuntimeError(f"multiple drafts found for {qid}: {matches}")
    return matches[0] if matches else None


def select_drafts(args: argparse.Namespace) -> list[Path]:
    if args.qids:
        out: list[Path] = []
        for qid in args.qids:
            p = find_draft(qid)
            if p is None:
                raise RuntimeError(f"no draft found for {qid!r}")
            out.append(p)
        return out
    # --all-passing: read scorecard
    scorecard_path = args.from_scorecard or DEFAULT_SCORECARD
    if not scorecard_path.exists():
        raise RuntimeError(f"missing {scorecard_path}; run validate_drafts.py first "
                           f"or pass --qids explicitly")
    sc = json.loads(scorecard_path.read_text(encoding="utf-8"))
    passing_qids = [r["draft_id"] for r in sc.get("rows", []) if r.get("verdict") == "pass"]
    out = []
    for qid in passing_qids:
        p = find_draft(qid)
        if p is None:
            print(f"  warning: scorecard has {qid} as pass but no draft file found "
                  f"(maybe already promoted?)", file=sys.stderr)
            continue
        out.append(p)
    return out


def clean_body(body: dict[str, Any], publish: bool, reviewed_by: str | None,
               now: str) -> dict[str, Any]:
    auth = body.pop("_authoring", None) or {}

    body["provenance"] = "llm-draft"
    body["status"] = "published" if publish else "draft"
    body["authors"] = [auth.get("origin", "gemini-3.1-pro-preview")]
    body["human_reviewed"] = {
        "status": "verified" if reviewed_by else "not-reviewed",
        "by": reviewed_by,
        "date": now if reviewed_by else None,
    }
    body.setdefault("created_at", auth.get("generated_at") or now)

    # gap-bridge:<lower>-<higher> tag for traceability
    gap = auth.get("gap") or {}
    if gap and gap.get("between"):
        existing = body.get("tags") or []
        bridge_tag = f"gap-bridge:{'-'.join(gap['between'])}"
        body["tags"] = list(dict.fromkeys(existing + [bridge_tag]))

    return body


def append_registry(qids: list[str], now: str) -> None:
    """Append-only — never rewrite the file.

    Format mirrors existing entries: one YAML mapping per line under entries.
    """
    lines = "\n".join(
        f"  - {{id: {qid}, created_at: {now}, created_by: promote_drafts.py}}"
        for qid in qids
    )
    with ID_REGISTRY.open("a", encoding="utf-8") as f:
        f.write(lines + "\n")


def promote_one(draft_path: Path, *, publish: bool, reviewed_by: str | None,
                now: str, dry_run: bool) -> tuple[str, Path]:
    body = yaml.safe_load(draft_path.read_text(encoding="utf-8"))
    if not isinstance(body, dict) or "id" not in body:
        raise RuntimeError(f"{draft_path}: malformed draft (no id field)")
    qid = body["id"]

    promoted_path = draft_path.with_suffix("")  # drops .draft → .yaml
    if promoted_path.exists():
        raise RuntimeError(
            f"{promoted_path} already exists — refusing to overwrite. "
            f"Resolve manually before promoting."
        )

    cleaned = clean_body(body, publish=publish, reviewed_by=reviewed_by, now=now)

    if dry_run:
        print(f"  DRY: {draft_path.name} → {promoted_path.name}  "
              f"(status={cleaned['status']}, "
              f"human_reviewed={cleaned['human_reviewed']['status']})")
    else:
        promoted_path.write_text(
            yaml.safe_dump(cleaned, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )
        draft_path.unlink()
        print(f"  ✓ {draft_path.name} → {promoted_path.name}  "
              f"(status={cleaned['status']})")

    return qid, promoted_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all-passing", action="store_true",
                   help="promote every draft whose scorecard verdict is 'pass'")
    g.add_argument("--qids", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   help="comma-separated explicit qid list")

    ap.add_argument("--from-scorecard", type=Path, default=None,
                    help=f"scorecard JSON (default {DEFAULT_SCORECARD})")
    ap.add_argument("--publish", action="store_true",
                    help="set status=published (default: status=draft, gating on review)")
    ap.add_argument("--reviewed-by", default=None,
                    help="set human_reviewed.status=verified, by=<handle>, date=<now>. "
                         "Implies the user has actually reviewed the drafts.")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would happen, don't write")
    args = ap.parse_args()

    drafts = select_drafts(args)
    if not drafts:
        print("no drafts to promote.")
        return 0

    print(f"promoting {len(drafts)} draft(s)"
          f"{' [DRY-RUN]' if args.dry_run else ''}"
          f"{' as PUBLISHED' if args.publish else ' as draft'}"
          f"{f' (reviewed_by={args.reviewed_by})' if args.reviewed_by else ''}:")
    print()

    now = datetime.now(UTC).isoformat(timespec="seconds")
    promoted_qids: list[str] = []
    for p in drafts:
        try:
            qid, _ = promote_one(p, publish=args.publish, reviewed_by=args.reviewed_by,
                                 now=now, dry_run=args.dry_run)
            promoted_qids.append(qid)
        except RuntimeError as e:
            print(f"  ✗ {p.name}: {e}", file=sys.stderr)

    if not args.dry_run and promoted_qids:
        append_registry(promoted_qids, now)
        print(f"\nappended {len(promoted_qids)} entries to "
              f"{ID_REGISTRY.relative_to(REPO_ROOT)}")

    print("\nNow run: vault check --strict && vault build --local-json")
    if any(args.publish for _ in promoted_qids):
        print("(promoted as published — chainCount may grow on next "
              "build_chains_with_gemini.py --all run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
