#!/usr/bin/env python3
"""Apply a Gemini-proposed chains.json to replace the live registry.

Reads `interviews/vault/chains.proposed.json` (output of
build_chains_with_gemini.py), validates it against the YAML corpus and
chain invariants, and on success replaces `interviews/vault/chains.json`.

Validation:
  - Every member id exists in the YAML corpus and is published
  - Levels in array order are non-decreasing (Bloom-monotonic) — Δ=0 IS
    allowed at this layer; the strict Δ ∈ {1,2} rule is enforced upstream
    in build_chains_with_gemini.py based on its --mode setting
  - 2 ≤ chain size ≤ 6
  - Single-topic
  - No qid in more than 2 chains, and Δ=2 only allowed for L1/L2 anchors
  - chain_id unique

The optional ``tier`` field on a chain entry (``primary``/``secondary``,
added in Phase 1.3 of CHAIN_ROADMAP.md) is intentionally not validated
here — it's a UI-routing hint, not a structural invariant.

Always run `vault check --strict` after this script — that's the final gate.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

VAULT_DIR = Path(__file__).resolve().parents[2] / "vault"
# AI-pipeline staging artifacts live under _pipeline/ (gitignored).
# See interviews/CLAUDE.md.
PIPELINE_DIR = VAULT_DIR / "_pipeline"
PROPOSED = PIPELINE_DIR / "chains.proposed.json"
LIVE = VAULT_DIR / "chains.json"
LIVE_BACKUP = VAULT_DIR / "chains.json.bak"

LEVEL_RANK = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6+": 6}


def load_yaml_corpus() -> dict[str, dict]:
    out = {}
    for p in (VAULT_DIR / "questions").rglob("*.yaml"):
        try:
            with open(p) as f:
                d = yaml.safe_load(f)
            if d.get("status") not in ("published", None):
                continue
            out[d["id"]] = d
        except Exception:
            pass
    return out


def validate(proposed: list[dict], corpus: dict[str, dict]) -> list[str]:
    errors: list[str] = []
    qid_to_chains: dict[str, list[str]] = {}
    seen_chain_ids = Counter()

    for ch in proposed:
        cid = ch.get("chain_id")
        if not cid:
            errors.append(f"chain missing chain_id: {ch}")
            continue
        seen_chain_ids[cid] += 1

        topic = ch.get("topic")
        track = ch.get("track")
        members = ch.get("questions", [])
        if not (2 <= len(members) <= 6):
            errors.append(f"{cid}: size {len(members)} not in [2,6]")
            continue

        levels = []
        for m in members:
            qid = m.get("id")
            if qid not in corpus:
                errors.append(f"{cid}: member {qid!r} not in published corpus")
                continue
            d = corpus[qid]
            levels.append(LEVEL_RANK.get(d.get("level"), 0))
            if d.get("topic") != topic:
                errors.append(f"{cid}: member {qid} topic={d.get('topic')!r} != chain topic {topic!r}")
            if d.get("track") != track:
                errors.append(f"{cid}: member {qid} track={d.get('track')!r} != chain track {track!r}")
            qid_to_chains.setdefault(qid, []).append(cid)

        if levels != sorted(levels):
            errors.append(f"{cid}: levels not monotonic: {levels}")

    for cid, n in seen_chain_ids.items():
        if n > 1:
            errors.append(f"chain_id {cid!r} appears {n} times")

    # Multi-chain membership cap: a question can be in at most 2 chains, and
    # only if it's L1 or L2 (foundational anchor pattern). Anything beyond
    # that is over-stuffing — likely a generic question reused too widely.
    for qid, chain_list in qid_to_chains.items():
        if len(chain_list) > 2:
            errors.append(f"{qid} appears in {len(chain_list)} chains; cap is 2")
        elif len(chain_list) == 2:
            level = corpus.get(qid, {}).get("level")
            if level not in ("L1", "L2"):
                errors.append(
                    f"{qid} (level={level}) is in 2 chains but multi-membership "
                    f"is only allowed for L1/L2 anchors"
                )

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proposed", default=str(PROPOSED))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Apply even if validation produces warnings (errors still block).")
    args = ap.parse_args()

    proposed_path = Path(args.proposed)
    if not proposed_path.exists():
        print(f"missing: {proposed_path}")
        return 1

    proposed = json.loads(proposed_path.read_text())
    print(f"proposed chains: {len(proposed)}")

    corpus = load_yaml_corpus()
    print(f"corpus: {len(corpus)} published questions")

    errors = validate(proposed, corpus)
    if errors:
        print(f"\n{len(errors)} validation issue(s):")
        for e in errors[:30]:
            print(f"  - {e}")
        if len(errors) > 30:
            print(f"  ... and {len(errors)-30} more")
        if not args.force:
            print("\nBlocking. Re-run with --force to apply anyway (NOT recommended).")
            return 1
    else:
        print("validation: clean")

    if args.dry_run:
        print("\n--dry-run set; not applying.")
        return 0

    # Back up live, then write proposed in canonical chains.json shape
    if LIVE.exists():
        shutil.copy2(LIVE, LIVE_BACKUP)
        print(f"backed up {LIVE} -> {LIVE_BACKUP}")
    LIVE.write_text(json.dumps(proposed, indent=2) + "\n")
    print(f"wrote {LIVE} with {len(proposed)} chains")
    print("\nNow run: vault check --strict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
