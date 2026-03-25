#!/usr/bin/env python3
"""
Gemini 3.1 Pro error fixer for StaffML corpus.

Takes batches of questions flagged with errors and asks Gemini to fix them.
Outputs corrected question JSON for each batch.

Usage:
    source ~/.zshrc_secrets
    PYTHONUNBUFFERED=1 python3 staffml/vault/scripts/gemini_fix_errors.py --workers 8
"""

import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

HARDWARE_REFERENCE = """
## Hardware Reference (mlsysim/core/constants.py — single source of truth)

| GPU | Memory | Type | Bandwidth | FP16 Tensor | TDP |
|-----|--------|------|-----------|-------------|-----|
| V100 | 32 GiB | HBM2 | 900 GB/s | 125 TFLOPS | 300W |
| A100 SXM | 80 GiB | HBM2e | 2039 GB/s (~2.0 TB/s) | 312 TFLOPS | 400W |
| H100 SXM | 80 GiB | HBM3 | 3.35 TB/s | 989 TFLOPS | 700W |
| H200 | 141 GB | HBM3e | 4.8 TB/s | 989 TFLOPS | 700W |
| B200 | 192 GiB | HBM3e | 8.0 TB/s | 2250 TFLOPS | 1000W |
| T4 | 16 GiB | GDDR6 | 320 GB/s | 65 TFLOPS | 70W |

| Interconnect | Bandwidth |
|---|---|
| NVLink A100 | 600 GB/s | NVLink H100 | 900 GB/s |
| PCIe Gen4 x16 | 32 GB/s (bidirectional) |
| PCIe Gen5 x16 | 64 GB/s (bidirectional) |
| IB HDR | 200 Gbps = 25 GB/s | IB NDR | 400 Gbps = 50 GB/s |

Key formulas:
- 1B params = 2 GB FP16, 4 GB FP32, 1 GB INT8, 0.5 GB INT4
- Training memory (Adam): 16 bytes/param
- KV cache: 2 × layers × kv_heads × head_dim × seq_len × bytes
- Ridge point: peak_FLOPS / peak_bandwidth
- AllReduce ring: 2(N-1)/N × data_size / bandwidth
- Conv2D FLOPs: 2 × K² × Cin × Cout × Hout × Wout

Edge/Mobile: Jetson Orin 275 TOPS INT8, 204.8 GB/s | Apple A17 Pro ~35 TOPS | Snapdragon 8 Gen 3 ~45 TOPS
TinyML: Cortex-M4 ~240 MHz | ESP32-S3 240 MHz, 512 KB SRAM | STM32H7 480 MHz, 1 MB SRAM
""".strip()

FIX_PROMPT = """You are an expert ML Systems engineer fixing errors in interview questions.

{hardware_reference}

## Instructions

Below are {num_questions} interview questions that were flagged with specific errors during review.
For each question:

1. Read the error description carefully
2. Determine if the error is REAL or a FALSE POSITIVE
3. If REAL: fix the question by correcting the math, specs, or logic in ALL affected fields (scenario, napkin_math, realistic_solution, common_mistake, options if MCQ)
4. If FALSE POSITIVE: leave the question unchanged

## CRITICAL RULES
- When you fix math, update ALL downstream values that depend on the corrected number
- If an MCQ correct_index needs to change, update it
- Preserve the question's pedagogical intent — fix the numbers, not the teaching goal
- Use hardware specs from the reference sheet above as ground truth

## Output Format

Return a JSON array. For each question, output:
```json
{{
  "corpus_index": <original index>,
  "id": "<question id>",
  "action": "FIXED" or "FALSE_POSITIVE",
  "fix_summary": "<what was fixed>" or "<why it's a false positive>",
  "corrected_fields": {{
    "scenario": "<new scenario if changed>",
    "details": {{
      "napkin_math": "<new napkin_math if changed>",
      "realistic_solution": "<new solution if changed>",
      "common_mistake": "<new common_mistake if changed>",
      "options": ["<new options if MCQ changed>"],
      "correct_index": <new index if changed>
    }}
  }}
}}
```

Only include fields in `corrected_fields` that actually changed. If FALSE_POSITIVE, omit `corrected_fields`.

## Questions to Fix

{questions_json}
"""


def fix_batch(batch_idx: int, batch_path: str, output_dir: Path, model: str) -> dict:
    """Send a batch to Gemini for fixing."""
    from google import genai

    with open(batch_path) as f:
        batch = json.load(f)

    # Build slim version for the prompt
    slim = []
    for item in batch:
        q = item["question"]
        entry = {
            "corpus_index": item["corpus_index"],
            "id": q.get("id", ""),
            "track": q.get("track", ""),
            "level": q.get("level", ""),
            "scenario": q.get("scenario", ""),
            "details": {
                "napkin_math": q.get("details", {}).get("napkin_math", ""),
                "realistic_solution": q.get("details", {}).get("realistic_solution", ""),
                "common_mistake": q.get("details", {}).get("common_mistake", ""),
            },
            "errors_found": item["gemini_errors"],
        }
        opts = q.get("details", {}).get("options")
        if opts:
            entry["details"]["options"] = opts
            entry["details"]["correct_index"] = q.get("details", {}).get("correct_index")
        slim.append(entry)

    prompt = FIX_PROMPT.format(
        hardware_reference=HARDWARE_REFERENCE,
        num_questions=len(slim),
        questions_json=json.dumps(slim, indent=2, ensure_ascii=False),
    )

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        raw = response.text

        # Save raw response
        (output_dir / f"raw_{batch_idx:03d}.txt").write_text(raw, encoding="utf-8")

        # Parse JSON
        fixes = json.loads(raw)
        (output_dir / f"fixes_{batch_idx:03d}.json").write_text(
            json.dumps(fixes, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        fixed = sum(1 for f in fixes if f.get("action") == "FIXED")
        fp = sum(1 for f in fixes if f.get("action") == "FALSE_POSITIVE")

        return {
            "batch": batch_idx,
            "total": len(batch),
            "fixed": fixed,
            "false_positive": fp,
            "parse_ok": True,
        }
    except json.JSONDecodeError as e:
        # Save raw even on parse failure
        (output_dir / f"raw_{batch_idx:03d}.txt").write_text(raw, encoding="utf-8")
        return {"batch": batch_idx, "total": len(batch), "error": f"JSON parse: {e}", "parse_ok": False}
    except Exception as e:
        return {"batch": batch_idx, "total": len(batch), "error": str(e), "parse_ok": False}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--batch-dir", default="/tmp/vault_gemini_fix")
    args = parser.parse_args()

    batch_files = sorted(Path(args.batch_dir).glob("batch_*.json"))
    print(f"Found {len(batch_files)} batches to fix")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"_validation_results/gemini_fixes_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    print()

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fix_batch, i, str(bf), output_dir, args.model): i
            for i, bf in enumerate(batch_files)
        }
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            if r.get("parse_ok"):
                print(f"  Batch {r['batch']:3d}: {r['fixed']} fixed, {r['false_positive']} false positives")
            else:
                print(f"  Batch {r['batch']:3d}: ERROR — {r.get('error', 'unknown')}")

    # Summary
    total_fixed = sum(r.get("fixed", 0) for r in results)
    total_fp = sum(r.get("false_positive", 0) for r in results)
    total_err = sum(1 for r in results if not r.get("parse_ok"))

    print(f"\n{'='*60}")
    print(f"FIX COMPLETE")
    print(f"{'='*60}")
    print(f"  Batches: {len(results)}")
    print(f"  Fixed: {total_fixed}")
    print(f"  False positives: {total_fp}")
    print(f"  Parse errors: {total_err}")
    print(f"  Output: {output_dir}")

    # Save summary
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
