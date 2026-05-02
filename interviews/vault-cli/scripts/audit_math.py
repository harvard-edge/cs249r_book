#!/usr/bin/env python3
"""Independent math verifier for question napkin_math blocks.

Standalone tool — runs ONE focused Gemini call per question to re-derive
the napkin_math arithmetic from scratch, then compares against what's
written. Catches calculation errors, unit-conversion mistakes, and
conclusions that don't follow from the calculations.

Use cases:
  1. Final gate on Phase 3-authored drafts before promotion
     (validate_drafts.py's coherence gate covers this generally; this
     gate is focused and stricter on the math specifically).
  2. Retroactive audit of any subset of the published corpus.

Usage:
  # Verify all .yaml.draft files (post-generation, pre-promotion):
  python3 verify_math.py --drafts-only

  # Verify specific files:
  python3 verify_math.py --files interviews/vault/questions/edge/latency/edge-2537.yaml ...

  # Verify a sample of published questions in a track:
  python3 verify_math.py --sample-track edge --sample-size 50

Parallelism is real: --workers N runs N concurrent Gemini calls. Default
is 4 (gentle on RPM). Cap at 8 to stay under typical rate limits.

Output:
  interviews/vault/_pipeline/math-verification.json — per-question rows
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
VAULT_DIR = REPO_ROOT / "interviews" / "vault"
QUESTIONS_DIR = VAULT_DIR / "questions"
PIPELINE_DIR = VAULT_DIR / "_pipeline"
DEFAULT_OUTPUT = PIPELINE_DIR / "math-verification.json"

GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_WORKERS = 4


PROMPT_TEMPLATE = """You are independently verifying the napkin_math block of an
ML systems interview question. Re-derive every calculation from the stated
assumptions; compare against what the question actually wrote.

Return STRICT JSON, no prose, no fences:

{{
  "arithmetic_correct": "yes" | "no" | "no_math",
  "unit_conversions_correct": "yes" | "no" | "no_conversions",
  "conclusion_follows": "yes" | "no",
  "errors": ["<specific issue>", ...],
  "rationale": "<one or two sentences>"
}}

GROUND RULES:
  - "arithmetic_correct=no_math" only if napkin_math is empty.
  - Be concrete in errors[]: "claims X = Y but X = Z" — quote the
    specific line and the correct value.
  - Tolerate small rounding (≤ 5%); flag anything bigger.
  - Don't penalize the question for being hard; only flag actual
    arithmetic / unit / logic errors.

QUESTION:
  id:        {qid}
  level:     {level}
  track:     {track}
  topic:     {topic}

  scenario:
{scenario}

  question:
{question}

  realistic_solution:
{solution}

  napkin_math:
{napkin}
"""


# ─── i/o helpers ──────────────────────────────────────────────────────────


def load_yaml(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as f:
            d = yaml.safe_load(f)
    except Exception:
        return None
    return d if isinstance(d, dict) else None


def discover_targets(args: argparse.Namespace) -> list[Path]:
    if args.files:
        return [p for p in args.files if p.exists()]
    if args.drafts_only:
        return sorted(QUESTIONS_DIR.rglob("*.yaml.draft"))
    if args.sample_track:
        pool = [p for p in QUESTIONS_DIR.rglob("*.yaml")
                if f"/{args.sample_track}/" in str(p)]
        rng = random.Random(args.seed)
        rng.shuffle(pool)
        return pool[: args.sample_size]
    return []


def has_napkin_math(body: dict) -> bool:
    details = body.get("details") or {}
    nm = (details.get("napkin_math") or "").strip()
    return bool(nm)


def indent(text: str | None, level: int = 4) -> str:
    if not text:
        return " " * level + "(empty)"
    pad = " " * level
    return "\n".join(pad + line for line in text.splitlines())


# ─── Gemini call ──────────────────────────────────────────────────────────


# Lock to avoid interleaved stderr from concurrent failure messages.
_print_lock = threading.Lock()


def call_gemini(prompt: str, timeout: int = 240) -> dict | None:
    try:
        result = subprocess.run(
            ["gemini", "-m", GEMINI_MODEL, "-p", prompt, "--yolo"],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    out = (result.stdout or "").strip()
    if out.startswith("```"):
        out = out.strip("`")
        if out.startswith("json"):
            out = out[4:].lstrip()
    i = out.find("{")
    j = out.rfind("}")
    if i == -1 or j == -1:
        if result.returncode != 0:
            with _print_lock:
                print(f"  gemini exit {result.returncode}: "
                      f"{(result.stderr or '')[:200]}", file=sys.stderr)
        return None
    try:
        return json.loads(out[i:j+1])
    except json.JSONDecodeError as e:
        with _print_lock:
            print(f"  JSON parse failed: {e}", file=sys.stderr)
        return None


# ─── core verification ────────────────────────────────────────────────────


def verify_one(path: Path) -> dict[str, Any]:
    body = load_yaml(path)
    if not body:
        return {"path": str(path), "verdict": "skip", "reason": "could not load"}
    qid = body.get("id", "?")

    if not has_napkin_math(body):
        return {"path": str(path), "qid": qid, "verdict": "skip",
                "reason": "no napkin_math present"}

    details = body.get("details") or {}
    prompt = PROMPT_TEMPLATE.format(
        qid=qid,
        level=body.get("level"),
        track=body.get("track"),
        topic=body.get("topic"),
        scenario=indent(body.get("scenario")),
        question=indent(body.get("question")),
        solution=indent(details.get("realistic_solution")),
        napkin=indent(details.get("napkin_math")),
    )
    resp = call_gemini(prompt)
    if resp is None:
        return {"path": str(path), "qid": qid, "verdict": "error",
                "reason": "no judge response"}

    arith = (resp.get("arithmetic_correct") or "").lower()
    units = (resp.get("unit_conversions_correct") or "").lower()
    concl = (resp.get("conclusion_follows") or "").lower()
    errors = resp.get("errors") or []

    # Pass iff arithmetic is correct AND (no unit conversions OR conversions correct)
    # AND conclusion follows. Empty-math drafts are scored "skip".
    has_arith_issue = arith not in ("yes", "no_math")
    has_unit_issue  = units not in ("yes", "no_conversions")
    has_concl_issue = concl != "yes"
    verdict = "fail" if (has_arith_issue or has_unit_issue or has_concl_issue) else "pass"

    return {
        "path": str(path),
        "qid": qid,
        "level": body.get("level"),
        "track": body.get("track"),
        "topic": body.get("topic"),
        "verdict": verdict,
        "arithmetic_correct": arith,
        "unit_conversions_correct": units,
        "conclusion_follows": concl,
        "errors": errors,
        "rationale": resp.get("rationale", ""),
    }


# ─── runner ───────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--drafts-only", action="store_true",
                     help="verify all *.yaml.draft files in the questions tree")
    src.add_argument("--files", nargs="+", type=Path, default=None,
                     help="explicit YAML paths")
    src.add_argument("--sample-track", choices=["cloud", "edge", "mobile", "tinyml", "global"],
                     help="random sample from a track (use --sample-size)")
    ap.add_argument("--sample-size", type=int, default=30,
                    help="sample size for --sample-track (default 30)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for sampling (default 42)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"concurrent Gemini calls (default {DEFAULT_WORKERS}, "
                         f"cap 8 to stay under typical RPM limits)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"scorecard JSON (default {DEFAULT_OUTPUT})")
    args = ap.parse_args()

    if args.workers < 1:
        args.workers = 1
    if args.workers > 8:
        print("warning: workers > 8 may hit Gemini RPM limits; capping at 8",
              file=sys.stderr)
        args.workers = 8

    targets = discover_targets(args)
    if not targets:
        print("no targets found")
        return 0

    print(f"verifying {len(targets)} question(s) with {args.workers} concurrent workers")

    results: list[dict[str, Any]] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(verify_one, p): p for p in targets}
        for i, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            results.append(row)
            v = row.get("verdict", "?")
            qid = row.get("qid", "?")
            extra = ""
            if v == "fail":
                errs = row.get("errors") or []
                extra = f"  [{len(errs)} error(s)] {(errs[0] if errs else '')[:80]}"
            elif v == "skip":
                extra = f"  ({row.get('reason')})"
            with _print_lock:
                print(f"  [{i:3d}/{len(targets)}] {qid:14s} {v:6s}{extra}")

    elapsed = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": GEMINI_MODEL,
        "workers": args.workers,
        "elapsed_seconds": round(elapsed, 1),
        "total": len(results),
        "passes": sum(1 for r in results if r.get("verdict") == "pass"),
        "fails":  sum(1 for r in results if r.get("verdict") == "fail"),
        "errors": sum(1 for r in results if r.get("verdict") == "error"),
        "skips":  sum(1 for r in results if r.get("verdict") == "skip"),
        "rows": sorted(results, key=lambda r: r.get("qid", "")),
    }, indent=2) + "\n", encoding="utf-8")

    n_pass = sum(1 for r in results if r.get("verdict") == "pass")
    n_fail = sum(1 for r in results if r.get("verdict") == "fail")
    n_err  = sum(1 for r in results if r.get("verdict") == "error")
    n_skip = sum(1 for r in results if r.get("verdict") == "skip")
    print(f"\nelapsed: {elapsed:.1f}s  pass={n_pass}  fail={n_fail}  "
          f"error={n_err}  skip={n_skip}")
    try:
        out_display = args.output.relative_to(REPO_ROOT)
    except ValueError:
        out_display = args.output
    print(f"wrote {out_display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
