#!/usr/bin/env python3
"""LLM-as-judge multi-criteria validator for generated questions.

Goes beyond `gemini_cli_math_review.py` (which checks only math) to
evaluate each draft against five criteria in one batched call:

  1. math_correct      — arithmetic, units, hardware specs all check out
  2. cell_fit          — actually targets the declared track/zone/level
  3. scenario_realism  — sounds like a real Staff-level interview, not
                          synthetic filler
  4. uniqueness        — does not duplicate canonical questions or
                          obvious LLM templates
  5. visual_alignment  — for visual items: does the diagram match the
                          quantities cited in the scenario? (skipped if
                          no visual)

The judge returns one verdict per question (PASS | NEEDS_FIX | DROP)
plus per-criterion findings. The runner then writes a structured
report so downstream steps can:
  - promote PASS items to status:published
  - hold NEEDS_FIX items for human edit
  - delete DROP items entirely (or mark them archived)

This is the "halt the loop on saturation" detector. If the DROP rate
crosses a threshold, the iterative-coverage-loop driver knows to stop:
the model is now hallucinating because the corpus is saturated.

Usage:

    # Judge all drafts marked status:draft + math_status absent:
    python3 gemini_cli_llm_judge.py --draft-only

    # Judge a specific batch of files:
    python3 gemini_cli_llm_judge.py --files-from /path/to/files.txt

    # Larger chunks for cost efficiency:
    python3 gemini_cli_llm_judge.py --draft-only --chunk-size 20 --max-calls 10
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

VAULT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_DIR = VAULT_DIR / "questions"
DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_OUTPUT_DIR = VAULT_DIR / "_validation_results" / "llm_judge"

HARDWARE_REFERENCE = """
Reference constants:
- H100 SXM: 80 GB HBM3, 3.35 TB/s, 989 TFLOP/s FP16, 700 W.
- A100 80GB SXM: 2.0 TB/s, 312 TFLOP/s FP16, 400 W.
- MI300X: 192 GB HBM3, 5.3 TB/s, 1307 TFLOP/s FP16 sparse peak.
- Jetson AGX Orin: 275 TOPS INT8, 204.8 GB/s LPDDR5, 15-60 W modes.
- Hailo-8: 26 TOPS INT8, 2.5 W.
- Apple A17 Pro NE: ~35 TOPS. Snapdragon 8 Gen 3 Hexagon: ~45 TOPS.
- Cortex-M4: 80-240 MHz, KB-scale SRAM.
- 1 byte = 8 bits. 1 GB/s = 1000 MB/s for napkin math.
- KV cache: 2*layers*KV_heads*head_dim*seqlen*batch*bytes.
- Ring AllReduce lower-bound: 2(N-1)/N * payload / bandwidth.
""".strip()

JUDGE_PROMPT = """You are an expert Staff-level ML Systems interview reviewer.

Below is a JSON array of {n} candidate questions. Judge each one against five criteria:

  1. math_correct      — Does the napkin math check out against the hardware reference?
                          ERROR if a number is wrong or units are wrong; WARN if
                          imprecise; PASS if sound enough for interview practice.
  2. cell_fit          — Does the question actually target its declared
                          (track, zone, level) coordinates? ERROR if mismatched.
  3. scenario_realism  — Does it sound like a real Staff-level interview
                          scenario, not generic filler? PASS if it grounds in
                          real hardware, real workloads, real production
                          situations. WARN if too synthetic.
  4. uniqueness        — Does it duplicate canonical questions (KV-cache for
                          Llama-70B, Ring AllReduce on 4 ranks, etc.) or look
                          like an obvious LLM template? WARN/ERROR if so.
  5. visual_alignment  — For items with a visual block: does the diagram alt
                          text describe what the scenario actually requires
                          the candidate to read? PASS/WARN/ERROR. Skip with
                          status N/A if no visual.

Then assign an overall verdict per item:
  - PASS       — all criteria PASS or N/A; safe to promote to published.
  - NEEDS_FIX  — at least one WARN, no ERRORs; can be saved with edits.
  - DROP       — at least one ERROR, OR repeated WARNs in different criteria.

Return STRICT JSON only — an array of {n} objects in input order:

[
  {{
    "id": "<question id>",
    "verdict": "PASS" | "NEEDS_FIX" | "DROP",
    "criteria": {{
      "math_correct": "PASS" | "WARN" | "ERROR",
      "cell_fit": "PASS" | "WARN" | "ERROR",
      "scenario_realism": "PASS" | "WARN" | "ERROR",
      "uniqueness": "PASS" | "WARN" | "ERROR",
      "visual_alignment": "PASS" | "WARN" | "ERROR" | "N/A"
    }},
    "issues": ["<short issue text per warn/error>"],
    "fix_suggestion": "<short recommended fix if NEEDS_FIX, else empty>"
  }}
]

{hardware_reference}

Questions to judge:

{questions_json}

Return only the JSON array.
"""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_drafts(draft_only: bool, files_from: Path | None) -> list[dict[str, Any]]:
    paths: list[Path] = []
    if files_from:
        for line in files_from.read_text().splitlines():
            line = line.strip()
            if line:
                p = Path(line)
                if not p.is_absolute():
                    p = VAULT_DIR.parent.parent / p
                paths.append(p)
    else:
        for p in QUESTIONS_DIR.glob("**/*.yaml"):
            paths.append(p)

    out = []
    for p in paths:
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not d:
            continue
        if draft_only and d.get("status") != "draft":
            continue
        d["_path"] = str(p)
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, model: str, timeout: int = 300) -> str:
    result = subprocess.run(
        ["gemini", "-m", model, "--prompt", prompt],
        input="", capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gemini CLI failed: {result.stderr.strip()}")
    return result.stdout


def extract_json_array(raw: str) -> list[Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n", "", raw)
        raw = re.sub(r"\n```\s*$", "", raw)
    start = raw.find("[")
    if start < 0:
        raise ValueError("No JSON array found")
    depth = 0
    end = -1
    for i, ch in enumerate(raw[start:], start=start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise ValueError("Unbalanced array")
    return json.loads(raw[start:end])


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def judge_chunk(items: list[dict[str, Any]], model: str, output_dir: Path
                ) -> list[dict[str, Any]]:
    """Judge a single chunk; return parsed verdicts (one per item)."""
    payload = []
    for d in items:
        entry = {
            "id": d["id"],
            "track": d["track"],
            "zone": d["zone"],
            "level": d["level"],
            "topic": d["topic"],
            "title": d.get("title", ""),
            "scenario": d.get("scenario", ""),
            "question": d.get("question", ""),
            "realistic_solution": d.get("details", {}).get("realistic_solution", ""),
            "common_mistake": d.get("details", {}).get("common_mistake", ""),
            "napkin_math": d.get("details", {}).get("napkin_math", ""),
        }
        if "visual" in d:
            entry["visual_alt"] = d["visual"].get("alt", "")
        payload.append(entry)

    prompt = JUDGE_PROMPT.format(
        n=len(items),
        questions_json=json.dumps(payload, indent=2),
        hardware_reference=HARDWARE_REFERENCE,
    )
    timestamp = int(time.time() * 1000)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"prompt_{timestamp}.txt").write_text(prompt, encoding="utf-8")

    raw = call_gemini(prompt, model)
    (output_dir / f"raw_{timestamp}.txt").write_text(raw, encoding="utf-8")

    return extract_json_array(raw)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--allow-model-override", action="store_true")
    parser.add_argument("--draft-only", action="store_true",
                        help="Restrict to status:draft questions.")
    parser.add_argument("--files-from", type=Path, default=None,
                        help="Newline-separated YAML paths (overrides discovery).")
    parser.add_argument("--chunk-size", type=int, default=15,
                        help="Items per Gemini call (max ~25 to stay safely "
                             "under context limits per call).")
    parser.add_argument("--max-calls", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0,
                        help="Hard cap on number of items judged (0 = no cap).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sleep", type=float, default=2.0)
    args = parser.parse_args()

    if args.model != DEFAULT_MODEL and not args.allow_model_override:
        sys.exit(f"Refusing --model={args.model!r}; default is {DEFAULT_MODEL!r}.")

    items = load_drafts(args.draft_only, args.files_from)
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("No items to judge.")
        return 0

    chunks = [items[i:i + args.chunk_size]
              for i in range(0, len(items), args.chunk_size)]
    print(f"Judging {len(items)} items in {len(chunks)} call(s) "
          f"(chunk size {args.chunk_size}, model={args.model})")

    output_dir = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    all_verdicts: list[dict[str, Any]] = []
    verdict_counts: Counter = Counter()

    for ci, chunk in enumerate(chunks):
        if ci >= args.max_calls:
            print(f"Reached --max-calls={args.max_calls}; stopping.")
            break
        if ci:
            time.sleep(args.sleep)
        print(f"\n[{ci+1}/{min(len(chunks), args.max_calls)}] judging {len(chunk)} items ...")
        try:
            verdicts = judge_chunk(chunk, args.model, output_dir)
        except Exception as exc:
            print(f"  ! call failed: {exc}")
            continue
        all_verdicts.extend(verdicts)
        for v in verdicts:
            verdict_counts[v.get("verdict", "?")] += 1
        print(f"  ← verdicts: " + " ".join(
            f"{k}={v}" for k, v in verdict_counts.items()))

    # Write summary
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "n_items": len(items),
        "n_judged": len(all_verdicts),
        "verdicts": dict(verdict_counts),
        "drop_rate": (verdict_counts.get("DROP", 0) / max(len(all_verdicts), 1)),
        "pass_rate": (verdict_counts.get("PASS", 0) / max(len(all_verdicts), 1)),
        "details": all_verdicts,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary: {output_dir}/summary.json")
    print(f"Verdicts: {dict(verdict_counts)}")
    print(f"Pass rate: {summary['pass_rate']:.1%}, Drop rate: {summary['drop_rate']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
