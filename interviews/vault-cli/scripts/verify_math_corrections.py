#!/usr/bin/env python3
"""Verify Gemini-proposed math corrections via an independent Gemini pass.

The original audit run flagged ~376 questions with math errors and
proposed napkin_math + realistic_solution rewrites. Auto-applying
those without verification is risky — we'd be trusting Gemini's
fix without a second eye. This script asks Gemini to independently
re-derive the answer from the scenario and compare against the
PROPOSED napkin_math/realistic_solution.

Verification verdict per correction:
  yes              proposed math computes correctly; safe to apply
  no               proposed math is still wrong; skip
  unclear          can't tell; skip

Auto-applies the 'yes' verdicts subject to the same defensive checks
as mass_apply_corrections.py (chain monotonicity, relabel-up policy,
Pydantic).

Usage:

    python3 interviews/vault-cli/scripts/verify_math_corrections.py \\
        --input interviews/vault/_pipeline/runs/full-corpus-20260503-merged/01_audit.json

CORPUS_HARDENING_PLAN.md Phase 5 — math-fix verification leg.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "interviews" / "vault-cli" / "src"))
sys.path.insert(0, str(REPO_ROOT / "interviews" / "vault-cli" / "scripts"))

from _judges import call_gemini_judge  # noqa: E402

from vault_cli.models import Question  # noqa: E402
from vault_cli.yaml_io import dump_str, load_file  # noqa: E402

VAULT_DIR = REPO_ROOT / "interviews" / "vault"
QUESTIONS_DIR = VAULT_DIR / "questions"
CHAINS_PATH = VAULT_DIR / "chains.json"

LEVEL_RANK = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6+": 6}

DEFAULT_WORKERS = 4
DEFAULT_BATCH_SIZE = 10  # smaller batches — verification prompts are bigger
SUBMIT_STAGGER_S = 1.0


# ─── corpus + chain helpers ──────────────────────────────────────────────


def find_question_file(qid: str) -> Path | None:
    for p in QUESTIONS_DIR.rglob(f"{qid}.yaml"):
        return p
    return None


def load_chains_index() -> tuple[dict, dict]:
    if not CHAINS_PATH.exists():
        return {}, {}
    chains = json.loads(CHAINS_PATH.read_text(encoding="utf-8"))
    chain_index: dict[str, list[tuple[str, int]]] = {}
    chain_levels: dict[str, list[tuple[str, str]]] = {}
    for chain in chains:
        cid = chain.get("chain_id")
        seq = []
        for i, q in enumerate(chain.get("questions") or []):
            qid = q.get("id") if isinstance(q, dict) else q
            level = q.get("level") if isinstance(q, dict) else None
            if qid:
                chain_index.setdefault(qid, []).append((cid, i))
                seq.append((qid, level))
        chain_levels[cid] = seq
    return chain_index, chain_levels


def level_relabel_safe(qid: str, new_level: str, current_level: str,
                        chain_index: dict, chain_levels: dict,
                        applied_levels: dict) -> tuple[bool, str]:
    new_rank = LEVEL_RANK.get(new_level)
    cur_rank = LEVEL_RANK.get(current_level)
    if new_rank is None or cur_rank is None:
        return False, "unknown level"
    if new_rank > cur_rank:
        return False, f"relabel-up blocked ({current_level} → {new_level})"
    if new_rank == cur_rank:
        return False, "no-op"
    memberships = chain_index.get(qid, [])
    if not memberships:
        return True, "ok"
    for cid, _pos in memberships:
        seq = chain_levels.get(cid, [])
        ranks = []
        for q_id, q_level in seq:
            if q_id == qid:
                lvl = new_level
            elif q_id in applied_levels:
                lvl = applied_levels[q_id]
            else:
                lvl = q_level
            ranks.append(LEVEL_RANK.get(lvl, 0))
        for i in range(1, len(ranks)):
            if ranks[i] < ranks[i - 1]:
                return False, f"breaks chain {cid}"
    return True, "ok"


# ─── verification prompt + batching ──────────────────────────────────────


VERIFY_PROMPT = """You are independently checking the math in proposed
fixes for ML-systems interview questions. Each item has the question's
SCENARIO, the QUESTION, the ORIGINAL realistic_solution (which was
flagged as wrong), and the PROPOSED napkin_math + realistic_solution.

For EACH item, re-derive the calculation from the scenario as if you
hadn't seen the proposed answer. Then compare your derivation against
the PROPOSED napkin_math:

  - Are the assumptions in the proposed napkin_math correct given the
    scenario?
  - Do the calculations follow correctly from the assumptions?
  - Does the proposed conclusion follow from the calculations?
  - Does the proposed realistic_solution agree with the proposed
    napkin_math (no contradictions)?

Return STRICT JSON, no prose, no fences:

{{
  "results": [
    {{
      "qid": "<id>",
      "verdict":              "yes" | "no" | "unclear",
      "math_independent_check": "yes" | "no" | "unclear",
      "rationale":            "<one sentence pointing to the SPECIFIC issue if no/unclear>"
    }}
  ]
}}

GROUND RULES:
  - "yes" means: I independently re-derived the answer and it agrees with
    the proposed napkin_math AND the proposed realistic_solution.
  - "no" means: I found a SPECIFIC error in the proposed math (cite it).
  - "unclear" means: I can't determine without additional context.
  - Be strict — only return "yes" if you're confident. Defaulting to
    "unclear" is correct when uncertain.

INPUT (n={n}):
{items_json}
"""


def build_verify_payload(row: dict, body: dict) -> dict:
    sc = row.get("suggested_corrections") or {}
    details = body.get("details") or {}
    return {
        "qid": row.get("qid"),
        "scenario": body.get("scenario"),
        "question": body.get("question"),
        "original_realistic_solution": details.get("realistic_solution"),
        "proposed_napkin_math": sc.get("napkin_math"),
        "proposed_realistic_solution": sc.get("realistic_solution"),
    }


def pack_verify_batches(payloads: list[dict], batch_size: int) -> list[list[dict]]:
    return [payloads[i:i + batch_size]
            for i in range(0, len(payloads), batch_size)]


def verify_one_batch(batch: list[dict], idx: int, total: int) -> tuple[int, list[dict], dict | None]:
    prompt = VERIFY_PROMPT.format(
        n=len(batch),
        items_json=json.dumps(batch, indent=2),
    )
    print(f"  [{idx:3d}/{total}] verify {len(batch)} items, {len(prompt)//1000}K char prompt")
    resp = call_gemini_judge(prompt)
    return idx, batch, resp


# ─── apply ───────────────────────────────────────────────────────────────


def apply_correction_to_dict(body: dict, correction: dict) -> dict:
    out = json.loads(json.dumps(body))
    details = out.setdefault("details", {})
    if correction.get("title"):
        out["title"] = correction["title"]
    if correction.get("level"):
        out["level"] = correction["level"]
    if correction.get("common_mistake"):
        details["common_mistake"] = correction["common_mistake"]
    if correction.get("napkin_math"):
        details["napkin_math"] = correction["napkin_math"]
    if correction.get("realistic_solution"):
        details["realistic_solution"] = correction["realistic_solution"]
    return out


def write_yaml(path: Path, body: dict) -> None:
    text = dump_str(body)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="01_audit.json with suggested_corrections")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"concurrent Gemini calls (default {DEFAULT_WORKERS}, max 8)")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help=f"items per verification call (default {DEFAULT_BATCH_SIZE})")
    ap.add_argument("--max-calls", type=int, default=50,
                    help="cap Gemini calls this run (default 50)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show plan + cost without making Gemini calls")
    args = ap.parse_args()

    if args.workers > 8:
        args.workers = 8

    args.input = args.input.resolve()
    audit = json.loads(args.input.read_text(encoding="utf-8"))
    rows = audit.get("rows", [])
    candidates = [
        r for r in rows
        if r.get("suggested_corrections")
        and r["suggested_corrections"].get("realistic_solution")
    ]
    print(f"loaded {len(rows)} rows; {len(candidates)} have realistic_solution rewrites (high-risk)")

    # Skip rows where the YAML's already-applied state matches the proposed
    # correction. (Possible if mass_apply was rerun on a partial subset.)
    payloads: list[dict] = []
    skipped_already_applied = 0
    skipped_yaml_missing = 0
    rows_to_verify: list[dict] = []
    for row in candidates:
        qid = row["qid"]
        yp = find_question_file(qid)
        if not yp:
            skipped_yaml_missing += 1
            continue
        try:
            body = load_file(yp)
        except Exception:
            continue
        if not isinstance(body, dict):
            continue
        # Has every proposed math field already landed? Skip if so.
        # (Original guard only compared realistic_solution; that missed
        # cases where rs matched by coincidence but napkin_math/common_mistake
        # still diverged. 2026-05-04: broadened to all three fields.)
        details = body.get("details") or {}
        sc = row["suggested_corrections"]
        rs_match = details.get("realistic_solution") == sc.get("realistic_solution")
        nm_match = (not sc.get("napkin_math")
                    or details.get("napkin_math") == sc.get("napkin_math"))
        cm_match = (not sc.get("common_mistake")
                    or details.get("common_mistake") == sc.get("common_mistake"))
        if rs_match and nm_match and cm_match:
            skipped_already_applied += 1
            continue
        payloads.append(build_verify_payload(row, body))
        rows_to_verify.append(row)

    print(f"  to verify: {len(payloads)}")
    print(f"  skipped (already applied): {skipped_already_applied}")
    print(f"  skipped (yaml missing): {skipped_yaml_missing}")

    if not payloads:
        print("nothing to verify.")
        return 0

    batches = pack_verify_batches(payloads, args.batch_size)
    capped = min(len(batches), args.max_calls)
    print(f"  {len(batches)} batches (target {args.batch_size}/batch); will run {capped}")

    if args.dry_run:
        for b in batches[:3]:
            chars = sum(len(json.dumps(p)) for p in b)
            print(f"  sample batch: {len(b)} items, ~{chars} chars")
        return 0

    results: list[dict] = []
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for j, batch in enumerate(batches[:capped]):
            if j > 0 and SUBMIT_STAGGER_S > 0:
                time.sleep(SUBMIT_STAGGER_S)
            futures.append(pool.submit(verify_one_batch, batch, j + 1, capped))
        for fut in as_completed(futures):
            idx, batch, resp = fut.result()
            if resp and isinstance(resp.get("results"), list):
                by_qid = {r.get("qid"): r for r in resp["results"] if isinstance(r, dict)}
                for item in batch:
                    qid = item["qid"]
                    v = by_qid.get(qid) or {}
                    results.append({
                        "qid": qid,
                        "verdict": v.get("verdict", "unclear"),
                        "rationale": v.get("rationale", "no response"),
                    })
            else:
                # whole batch failed
                for item in batch:
                    results.append({
                        "qid": item["qid"],
                        "verdict": "unclear",
                        "rationale": "batch verification failed (no response)",
                    })

    elapsed = time.time() - started
    print(f"\nverification complete: {len(results)} rows in {elapsed:.1f}s")
    verdict_counts = Counter(r["verdict"] for r in results)
    print(f"  verdicts: {dict(verdict_counts)}")

    # Persist verification results.
    verify_path = args.input.parent / "03_math_verification.json"
    verify_path.write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input_audit": str(args.input),
        "verdicts": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {verify_path}")

    # Apply the 'yes' verdicts.
    yes_qids = {r["qid"] for r in results if r["verdict"] == "yes"}
    print(f"\nauto-applying {len(yes_qids)} verified-correct corrections")

    chain_index, chain_levels = load_chains_index()
    applied_levels: dict[str, str] = {}
    apply_counters = Counter()
    apply_dispositions: list[dict] = []

    for row in rows_to_verify:
        qid = row["qid"]
        if qid not in yes_qids:
            continue
        correction = row["suggested_corrections"]
        yp = find_question_file(qid)
        if not yp:
            apply_counters["yaml-missing"] += 1
            continue
        try:
            body = load_file(yp)
        except Exception as e:
            apply_counters["yaml-load-failed"] += 1
            apply_dispositions.append({"qid": qid, "result": "yaml-load-failed", "error": str(e)[:200]})
            continue
        if not isinstance(body, dict):
            continue

        # Check level relabel safety if proposed.
        if correction.get("level"):
            current_level = applied_levels.get(qid) or body.get("level")
            ok, why = level_relabel_safe(qid, correction["level"], current_level,
                                          chain_index, chain_levels, applied_levels)
            if not ok:
                apply_counters["level-block"] += 1
                apply_dispositions.append({"qid": qid, "result": "level-block", "error": why})
                continue

        proposed = apply_correction_to_dict(body, correction)
        try:
            Question.model_validate(proposed)
        except Exception as e:
            apply_counters["pydantic-fail"] += 1
            apply_dispositions.append({"qid": qid, "result": "pydantic-fail", "error": str(e)[:300]})
            continue

        write_yaml(yp, proposed)
        if correction.get("level"):
            applied_levels[qid] = correction["level"]
        apply_counters["applied"] += 1
        apply_dispositions.append({"qid": qid, "result": "applied"})

    print(f"\napply counters: {dict(apply_counters)}")

    apply_path = args.input.parent / "04_math_applied.json"
    apply_path.write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary": dict(apply_counters),
        "dispositions": apply_dispositions,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {apply_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
