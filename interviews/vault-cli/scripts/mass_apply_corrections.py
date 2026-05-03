#!/usr/bin/env python3
"""Mass-apply low-risk Gemini-proposed corrections without prompting.

Reads a 01_audit.json from a --propose-fixes run, walks rows with
suggested_corrections, classifies each by risk, and auto-applies the
LOW-risk ones. HIGH-risk corrections (anything that rewrites
realistic_solution — i.e., math-driven fixes) are SKIPPED here and
must be reviewed via apply_corrections.py interactively or via a
separate Gemini-verify-then-apply pipeline.

Risk classification:

  LOW (auto-applyable here):
    Correction touches ONLY ⊆ {title, level, common_mistake, napkin_math}
    No realistic_solution rewrite (math fixes are not auto-applied
    because they require independent math verification).

  HIGH (skipped):
    Any correction that includes realistic_solution.

For each LOW-risk correction:
  1. Load the YAML
  2. Apply the correction
  3. Run Pydantic Question validation
  4. If valid: write back via atomic temp+rename
  5. If invalid: log the reason and SKIP (don't write)

After each apply, a disposition is logged to a sidecar JSON. After
the whole pass, prints a summary and an exit code (0 if anything was
applied, 1 if nothing or all failed).

Usage:

    python3 interviews/vault-cli/scripts/mass_apply_corrections.py \\
        --input interviews/vault/_pipeline/runs/full-corpus-20260503-merged/01_audit.json

    # Dry-run — count + show first 10 candidates without writing:
    python3 interviews/vault-cli/scripts/mass_apply_corrections.py \\
        --input <path> --dry-run

    # Limit per category to ramp up gradually:
    python3 interviews/vault-cli/scripts/mass_apply_corrections.py \\
        --input <path> --max-per-category 50

CORPUS_HARDENING_PLAN.md Phase 5 (low-risk auto-apply leg).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "interviews" / "vault-cli" / "src"))

from vault_cli.models import Question  # noqa: E402
from vault_cli.yaml_io import dump_str, load_file  # noqa: E402

VAULT_DIR = REPO_ROOT / "interviews" / "vault"
QUESTIONS_DIR = VAULT_DIR / "questions"
CHAINS_PATH = VAULT_DIR / "chains.json"

LOW_RISK_FIELDS = {"title", "level", "common_mistake", "napkin_math"}
HIGH_RISK_FIELDS = {"realistic_solution"}

LEVEL_RANK = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6+": 6}


def load_chains_index() -> dict[str, list[tuple[str, int]]]:
    """qid → list of (chain_id, position) memberships.

    Used to verify that a level relabel doesn't break a chain's
    monotonic-non-decreasing-level invariant.
    """
    if not CHAINS_PATH.exists():
        return {}
    chains = json.loads(CHAINS_PATH.read_text(encoding="utf-8"))
    out: dict[str, list[tuple[str, int]]] = {}
    for chain in chains:
        cid = chain.get("chain_id")
        for i, q in enumerate(chain.get("questions") or []):
            qid = q.get("id") if isinstance(q, dict) else q
            if qid:
                out.setdefault(qid, []).append((cid, i))
    return out


def load_chain_levels() -> dict[str, list[tuple[str, str]]]:
    """chain_id → list of (qid, level) ordered by position."""
    if not CHAINS_PATH.exists():
        return {}
    chains = json.loads(CHAINS_PATH.read_text(encoding="utf-8"))
    out: dict[str, list[tuple[str, str]]] = {}
    for chain in chains:
        cid = chain.get("chain_id")
        seq = []
        for q in chain.get("questions") or []:
            if isinstance(q, dict):
                seq.append((q.get("id"), q.get("level")))
        out[cid] = seq
    return out


def level_relabel_safe(qid: str, new_level: str, current_level: str,
                        chain_index: dict[str, list[tuple[str, int]]],
                        chain_levels: dict[str, list[tuple[str, str]]],
                        applied_levels: dict[str, str]) -> tuple[bool, str]:
    """Check whether changing qid's level to new_level is safe:

    1. Must be a relabel-DOWN (per CORPUS_HARDENING_PLAN.md §10 Q3 —
       relabel-up requires question rewrite, separate authoring task).
    2. Must keep every chain qid is a member of monotonic non-decreasing.

    chain_levels is the ORIGINAL chain composition; applied_levels
    overlays this with prior applies in the same script run so cascading
    same-chain relabels are caught.

    Returns (ok, reason).
    """
    new_rank = LEVEL_RANK.get(new_level)
    cur_rank = LEVEL_RANK.get(current_level)
    if new_rank is None:
        return False, f"unknown new level {new_level!r}"
    if cur_rank is None:
        return False, f"unknown current level {current_level!r}"
    # Rule 1: only relabel-DOWN allowed.
    if new_rank > cur_rank:
        return False, (
            f"relabel-up blocked ({current_level} → {new_level}); "
            "policy is relabel-down only — see CORPUS_HARDENING_PLAN.md §10 Q3"
        )
    if new_rank == cur_rank:
        return False, "no-op level change"

    memberships = chain_index.get(qid, [])
    if not memberships:
        return True, "not in any chain"

    for cid, _pos in memberships:
        seq = chain_levels.get(cid, [])
        # Build the proposed sequence of ranks with this qid relabeled,
        # AND with any prior applies in this run reflected.
        ranks = []
        seq_levels: list[tuple[str | None, str | None]] = []
        for q_id, q_level in seq:
            if q_id == qid:
                lvl = new_level
            elif q_id in applied_levels:
                lvl = applied_levels[q_id]
            else:
                lvl = q_level
            seq_levels.append((q_id, lvl))
            ranks.append(LEVEL_RANK.get(lvl, 0))
        for i in range(1, len(ranks)):
            if ranks[i] < ranks[i - 1]:
                return False, (
                    f"would break chain {cid} at position {i} "
                    f"(post-apply levels: {[s[1] for s in seq_levels]})"
                )
    return True, "all chain memberships still monotonic"


def find_question_file(qid: str) -> Path | None:
    for p in QUESTIONS_DIR.rglob(f"{qid}.yaml"):
        return p
    return None


def classify(correction: dict) -> str:
    """Return 'low' | 'high' | 'empty'."""
    keys = {k for k, v in correction.items() if v}
    if not keys:
        return "empty"
    if keys & HIGH_RISK_FIELDS:
        return "high"
    if keys.issubset(LOW_RISK_FIELDS):
        return "low"
    return "high"  # unknown keys default to high-risk


def apply_correction_to_dict(body: dict, correction: dict) -> dict:
    out = json.loads(json.dumps(body))
    details = out.setdefault("details", {})
    if "title" in correction and correction["title"]:
        out["title"] = correction["title"]
    if "level" in correction and correction["level"]:
        out["level"] = correction["level"]
    if "common_mistake" in correction and correction["common_mistake"]:
        details["common_mistake"] = correction["common_mistake"]
    if "napkin_math" in correction and correction["napkin_math"]:
        details["napkin_math"] = correction["napkin_math"]
    return out


def validate_proposed(body: dict) -> tuple[bool, str]:
    try:
        Question.model_validate(body)
        return True, ""
    except Exception as e:
        return False, str(e)[:300]


def write_yaml(path: Path, body: dict) -> None:
    text = dump_str(body)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def detect_category(correction: dict) -> str:
    """Bucket a low-risk correction for reporting."""
    keys = sorted(k for k, v in correction.items() if v)
    if keys == ["title"]:
        return "title-only"
    if keys == ["level"]:
        return "level-only"
    if set(keys).issubset({"common_mistake", "napkin_math"}):
        return "format-only"
    if "level" in keys and set(keys) & {"common_mistake", "napkin_math"}:
        return "level+format"
    return "other-low"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="01_audit.json from a --propose-fixes run")
    ap.add_argument("--dispositions-out", type=Path, default=None,
                    help="sidecar (default <input-dir>/02_mass_apply.json)")
    ap.add_argument("--max-per-category", type=int, default=None,
                    help="cap auto-applies per category (smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show plan without writing")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 1
    args.input = args.input.resolve()

    audit = json.loads(args.input.read_text(encoding="utf-8"))
    rows = audit.get("rows", [])
    print(f"loaded {len(rows)} rows from {args.input.relative_to(REPO_ROOT)}")

    candidates = [r for r in rows if r.get("suggested_corrections")]
    print(f"  with corrections: {len(candidates)}")

    by_risk = Counter(classify(r["suggested_corrections"]) for r in candidates)
    print(f"  by risk: {dict(by_risk)}")

    low = [r for r in candidates if classify(r["suggested_corrections"]) == "low"]
    print(f"\n{len(low)} low-risk candidates eligible for auto-apply")
    by_cat = Counter(detect_category(r["suggested_corrections"]) for r in low)
    for cat, n in by_cat.most_common():
        print(f"  {cat:20s} {n}")

    if args.dry_run:
        print("\n[dry-run] sample first 10:")
        for r in low[:10]:
            sc = r["suggested_corrections"]
            keys = ", ".join(k for k, v in sc.items() if v)
            print(f"  {r['qid']}  fields=[{keys}]")
        return 0

    disp_path = args.dispositions_out or (args.input.parent / "02_mass_apply.json")
    dispositions = []

    # Pre-load chain memberships + levels for the chain-monotonicity check.
    chain_index = load_chains_index()
    chain_levels = load_chain_levels()
    print(f"  chain index: {len(chain_index)} qids in chains")

    # Track levels we've changed during this run so cascade-relabels
    # in the same chain don't slip through (caught 2026-05-03: relabeling
    # both edge-0752 (L3→L4 UP — also a separate bug) and edge-0771
    # (L4→L3) in the same run broke the chain monotonicity because
    # each individual check used the stale starting state).
    applied_levels: dict[str, str] = {}

    counters = Counter()
    per_cat_counters: dict[str, Counter] = defaultdict(Counter)
    per_cat_applied: dict[str, int] = defaultdict(int)

    for row in low:
        qid = row["qid"]
        correction = row["suggested_corrections"]
        category = detect_category(correction)

        # Pre-flight: reject level relabels that would break a chain
        # OR are relabel-up (against policy).
        if "level" in correction and correction["level"]:
            # Need the current level — either from a prior apply this
            # run, or from the original chain entry, or load the YAML.
            current_level = applied_levels.get(qid)
            if current_level is None:
                # Find the qid's original level via chain_levels (cheap).
                for memberships in [chain_index.get(qid, [])]:
                    for cid, _pos in memberships:
                        for q_id, q_level in chain_levels.get(cid, []):
                            if q_id == qid:
                                current_level = q_level
                                break
                        if current_level:
                            break
            if current_level is None:
                # Fallback: load the YAML to read current level.
                yp = find_question_file(qid)
                if yp:
                    try:
                        body = load_file(yp)
                        if isinstance(body, dict):
                            current_level = body.get("level")
                    except Exception:
                        pass
            if current_level is None:
                counters["level-current-unknown"] += 1
                per_cat_counters[category]["level-current-unknown"] += 1
                dispositions.append({"qid": qid, "category": category,
                                      "result": "level-current-unknown"})
                continue

            ok, why = level_relabel_safe(qid, correction["level"],
                                          current_level,
                                          chain_index, chain_levels,
                                          applied_levels)
            if not ok:
                key = "relabel-up-block" if "relabel-up" in why else "chain-monotonicity-block"
                counters[key] += 1
                per_cat_counters[category][key] += 1
                dispositions.append({"qid": qid, "category": category,
                                      "result": key, "error": why})
                continue

        if (args.max_per_category is not None
                and per_cat_applied[category] >= args.max_per_category):
            counters["category-cap"] += 1
            continue

        yaml_path = find_question_file(qid)
        if not yaml_path:
            counters["yaml-missing"] += 1
            per_cat_counters[category]["yaml-missing"] += 1
            dispositions.append({"qid": qid, "category": category,
                                  "result": "yaml-missing"})
            continue
        try:
            body = load_file(yaml_path)
        except Exception as e:
            counters["yaml-load-failed"] += 1
            per_cat_counters[category]["yaml-load-failed"] += 1
            dispositions.append({"qid": qid, "category": category,
                                  "result": "yaml-load-failed",
                                  "error": str(e)[:200]})
            continue

        if not isinstance(body, dict):
            counters["yaml-not-dict"] += 1
            continue

        proposed = apply_correction_to_dict(body, correction)
        ok, why = validate_proposed(proposed)
        if not ok:
            counters["pydantic-fail"] += 1
            per_cat_counters[category]["pydantic-fail"] += 1
            dispositions.append({"qid": qid, "category": category,
                                  "result": "pydantic-fail",
                                  "error": why})
            continue

        write_yaml(yaml_path, proposed)
        # Record the applied level so subsequent same-chain checks see
        # the post-apply state.
        if "level" in correction and correction["level"]:
            applied_levels[qid] = correction["level"]
        counters["applied"] += 1
        per_cat_counters[category]["applied"] += 1
        per_cat_applied[category] += 1
        dispositions.append({"qid": qid, "category": category,
                              "result": "applied"})

    # Persist dispositions.
    disp_path.parent.mkdir(parents=True, exist_ok=True)
    disp_path.write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input_audit": str(args.input),
        "summary": dict(counters),
        "per_category": {k: dict(v) for k, v in per_cat_counters.items()},
        "dispositions": dispositions,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\nresult counters: {dict(counters)}")
    print("\nper-category breakdown:")
    for cat, c in per_cat_counters.items():
        print(f"  {cat:20s} {dict(c)}")
    print(f"\nwrote {disp_path}")
    return 0 if counters["applied"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
