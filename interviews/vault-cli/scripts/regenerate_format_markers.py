#!/usr/bin/env python3
"""Regenerate marker-compliant common_mistake / napkin_math via Gemini.

Phase 6 pre-flight (2026-05-04): after apply_format_skip_level.py drained
the 41 entangled holdouts, 36 published YAMLs still have malformed
markers. The audit either had no proposal for these or the proposal
itself didn't follow the markers. This script asks Gemini to rewrite
the existing content into the canonical Pitfall/Rationale/Consequence
or Assumptions/Calculations/Conclusion structure WITHOUT changing the
underlying claim (numbers, conclusions, technical content stay).

Usage:

    python3 interviews/vault-cli/scripts/regenerate_format_markers.py \\
        --batch-size 10 --dry-run

    # then for real:
    python3 interviews/vault-cli/scripts/regenerate_format_markers.py \\
        --batch-size 10

Each Gemini call processes <batch-size> rewrites at once. After receiving
responses, the script verifies the rewritten field is marker-compliant
before applying.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

QUESTIONS_DIR = REPO_ROOT / "interviews" / "vault" / "questions"

CM_PATTERN = re.compile(
    r"(?s).*\*\*The Pitfall:\*\*.*\*\*The Rationale:\*\*.*\*\*The Consequence:\*\*.*"
)
NM_PATTERN = re.compile(
    r"(?s).*\*\*Assumptions.*\*\*Calculations:\*\*.*\*\*Conclusion.*"
)

REWRITE_PROMPT = """You are restructuring a single field of a Staff-level ML
Systems interview question. The CONTENT is correct. Only the MARKERS need to
follow the canonical authoring convention.

You will receive a JSON array of items. Each item has:
  - qid:      the question id (echo back unchanged)
  - field:    "common_mistake" or "napkin_math"
  - scenario: question scenario for context (DO NOT rewrite)
  - current:  the existing field value (malformed markers)

For each item, return the field rewritten with the canonical markers below.

CANONICAL TEMPLATES (exact):

common_mistake (3 markers, order matters, all required):
  **The Pitfall:** <the wrong intuition or shortcut>
  **The Rationale:** <why that intuition is wrong>
  **The Consequence:** <the operational symptom — latency, cost, failure mode>

napkin_math (3 marker sections, order matters, all required):
  **Assumptions & Constraints:**
  - <assumption 1>
  - <assumption 2>

  **Calculations:**
  - <step 1 with units>
  - <step 2>

  **Conclusion:** <one-sentence interpretation>

Allowed marker variants:
  - **Assumptions:** OR **Assumptions & Constraints:**
  - **Conclusion:** OR **Conclusion & Interpretation:**
  - **Calculations:** is exact

RULES:
  1. Preserve every claim, number, formula, and conclusion from `current`.
     Do not invent new numbers or change a stated value.
  2. Reorganize content under the three required markers. Demote any
     custom sub-headers (e.g., "**Memory:**", "**Compute:**") into
     bulleted lines under **Calculations:**.
  3. If the source has only a sentence-level claim with no calculations,
     synthesize plausible Assumptions/Calculations/Conclusion that
     support that same claim WITHOUT changing the conclusion.
  4. Output is a JSON object: {"results": [{"qid": "...", "rewrite": "..."}, ...]}.
  5. Each rewrite must contain all three required markers in order.
  6. Use plain hyphen bullets ("- "), not asterisk-bullet ("* ").
  7. Keep total length within 1.5x of the source.
"""


def find_question_file(qid: str) -> Path | None:
    for p in QUESTIONS_DIR.rglob(f"{qid}.yaml"):
        return p
    return None


def write_yaml(path: Path, body: dict) -> None:
    text = dump_str(body)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def collect_targets() -> list[dict]:
    targets: list[dict] = []
    for yp in QUESTIONS_DIR.rglob("*.yaml"):
        body = load_file(yp)
        if not isinstance(body, dict):
            continue
        if body.get("status") != "published":
            continue
        d = body.get("details") or {}
        cm = (d.get("common_mistake") or "").strip()
        nm = (d.get("napkin_math") or "").strip()
        if cm and not CM_PATTERN.match(cm):
            targets.append({"qid": body["id"], "field": "common_mistake",
                            "scenario": body.get("scenario", ""),
                            "current": cm, "_path": yp})
        if nm and not NM_PATTERN.match(nm):
            targets.append({"qid": body["id"], "field": "napkin_math",
                            "scenario": body.get("scenario", ""),
                            "current": nm, "_path": yp})
    return targets


def rewrite_one_batch(batch: list[dict], idx: int, total: int):
    payload = [{k: v for k, v in t.items() if not k.startswith("_")} for t in batch]
    prompt = REWRITE_PROMPT + "\n\nINPUT:\n" + json.dumps(payload, indent=2)
    print(f"  [{idx:>2}/{total}] rewrite {len(batch)} items, ~{sum(len(json.dumps(p)) for p in payload)} chars")
    resp = call_gemini_judge(prompt)
    return idx, batch, resp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--max-calls", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "interviews" / "vault" / "_pipeline" /
                            "runs" / "full-corpus-20260503-merged" /
                            "07_format_regenerated.json")
    args = ap.parse_args()

    targets = collect_targets()
    print(f"targets: {len(targets)}")
    if not targets:
        print("nothing to do.")
        return 0
    batches = [targets[i:i + args.batch_size]
               for i in range(0, len(targets), args.batch_size)]
    capped = min(len(batches), args.max_calls)
    print(f"  {len(batches)} batches, will run {capped}")

    if args.dry_run:
        for t in targets[:5]:
            print(f"  {t['qid']}/{t['field']}: cur[:80]={t['current'][:80]!r}")
        return 0

    rewrites: dict[tuple[str, str], str] = {}
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for j, batch in enumerate(batches[:capped]):
            if j > 0:
                time.sleep(1.0)
            futures.append(pool.submit(rewrite_one_batch, batch, j + 1, capped))
        for fut in as_completed(futures):
            idx, batch, resp = fut.result()
            if not resp or not isinstance(resp.get("results"), list):
                print(f"  batch {idx}: no usable response")
                continue
            by_qid = {r.get("qid"): r for r in resp["results"] if isinstance(r, dict)}
            for item in batch:
                v = by_qid.get(item["qid"]) or {}
                rw = (v.get("rewrite") or "").strip()
                if rw:
                    rewrites[(item["qid"], item["field"])] = rw
    elapsed = time.time() - started
    print(f"\nrewrites returned: {len(rewrites)} / {len(targets)} in {elapsed:.1f}s")

    counters: Counter[str] = Counter()
    dispositions: list[dict] = []

    for t in targets:
        key = (t["qid"], t["field"])
        rw = rewrites.get(key)
        if not rw:
            counters["no-response"] += 1
            dispositions.append({"qid": t["qid"], "field": t["field"],
                                  "result": "no-response"})
            continue
        pat = CM_PATTERN if t["field"] == "common_mistake" else NM_PATTERN
        if not pat.match(rw):
            counters["regex-fail"] += 1
            dispositions.append({"qid": t["qid"], "field": t["field"],
                                  "result": "regex-fail",
                                  "rewrite_preview": rw[:200]})
            continue

        body = load_file(t["_path"])
        if not isinstance(body, dict):
            counters["yaml-bad"] += 1
            continue
        proposed = json.loads(json.dumps(body))
        proposed.setdefault("details", {})[t["field"]] = rw
        try:
            Question.model_validate(proposed)
        except Exception as e:
            counters["pydantic-fail"] += 1
            dispositions.append({"qid": t["qid"], "field": t["field"],
                                  "result": "pydantic-fail",
                                  "error": str(e)[:300]})
            continue
        write_yaml(t["_path"], proposed)
        counters["applied"] += 1
        dispositions.append({"qid": t["qid"], "field": t["field"],
                              "result": "applied"})

    print(f"\ncounters: {dict(counters)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary": dict(counters),
        "dispositions": dispositions,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0 if counters["applied"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
