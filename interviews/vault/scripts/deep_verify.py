#!/usr/bin/env python3
"""Deep math verification using Claude Opus with extended thinking.

Unlike the lightweight verify_math.py (which checks ~25 Qs per call
with brief responses), this script sends smaller batches and asks the
model to SHOW ITS WORK — step-by-step recalculation of every napkin
math claim, with explicit arithmetic checks.

Usage:
    python3 scripts/deep_verify.py                    # Verify all unverified
    python3 scripts/deep_verify.py --sample 200       # Random sample
    python3 scripts/deep_verify.py --topic flash-attention  # Single topic
    python3 scripts/deep_verify.py --errors-only      # Re-check previous errors
    python3 scripts/deep_verify.py --dry-run           # Plan only
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent.parent
CORPUS_PATH = BASE / "corpus.json"
RESULTS_DIR = BASE / "scripts" / "_verification_results"

# Hardware reference specs for verification prompts
HARDWARE_SPECS = """
AUTHORITATIVE HARDWARE SPECS (use these to check questions):
- H100 SXM: 80 GB HBM3, 3.35 TB/s bandwidth, 989 TFLOPS FP16 dense, 1,979 with sparsity, 700W TDP
- A100 SXM: 80 GB HBM2e, 2.0 TB/s bandwidth, 312 TFLOPS FP16 dense, 624 with sparsity, 400W TDP
- MI300X: 192 GB HBM3, 5.3 TB/s bandwidth, 1,307 TFLOPS FP16, 750W TDP, 8 XCDs
- TPU v5e: 16 GB HBM, 1.6 TB/s bandwidth, 197 TFLOPS BF16
- Jetson Orin: 32 GB LPDDR5, 275 TOPS INT8, 60W TDP
- Hailo-8: 26 TOPS INT8, 2.5W
- Coral Edge TPU: 4 TOPS INT8, 2W
- A17 Pro: 35 TOPS Neural Engine, 8 GB unified
- Snapdragon 8 Gen 3: 45 TOPS INT8 Hexagon NPU, 16 GB LPDDR5X
- Cortex-M4: 168 MHz (NOT 240 MHz), 256 KB SRAM, no FPU
- Cortex-M7: 480 MHz, 512 KB SRAM
- ESP32-S3: 240 MHz dual-core, 512 KB SRAM, 8 MB PSRAM

KEY FORMULAS:
- Model memory (FP16): params × 2 bytes
- Adam optimizer state: params × (2 + 4 + 4 + 4) = params × 14 bytes (FP16 weights + FP32 master + momentum + variance)
- KV cache per token: 2 × n_layers × n_kv_heads × head_dim × bytes_per_element
- Ridge point: peak_FLOPS / memory_bandwidth (FLOPs/Byte)
- Arithmetic intensity: FLOPs / bytes_accessed
- Ring AllReduce time: 2 × (N-1)/N × message_size / bandwidth
"""


def build_deep_verify_prompt(questions):
    """Build a prompt that asks for step-by-step verification."""
    q_texts = []
    for i, q in enumerate(questions):
        details = q.get("details", {})
        q_texts.append(f"""
--- QUESTION {i+1} [id={q['id']}] ---
Title: {q.get('title', 'N/A')}
Topic: {q.get('topic', 'N/A')} | Track: {q.get('track', 'N/A')} | Zone: {q.get('zone', 'N/A')}
Scenario: {q.get('scenario', 'N/A')}
Napkin Math: {details.get('napkin_math', 'N/A')}
Solution: {details.get('realistic_solution', 'N/A')}
Common Mistake: {details.get('common_mistake', 'N/A')}
""")

    batch_text = "\n".join(q_texts)

    return f"""You are a meticulous ML systems verification engine. For each question below,
you MUST independently recalculate every numerical claim from first principles.

{HARDWARE_SPECS}

VERIFICATION PROTOCOL:
For each question:
1. IDENTIFY all numerical claims (hardware specs, computed values, conclusions)
2. CHECK each hardware spec against the authoritative list above
3. RECALCULATE every derived number step by step (show your arithmetic)
4. VERIFY the conclusion follows from the math
5. CHECK units (TFLOPS vs GFLOPS, GB vs MB, seconds vs milliseconds, TB/s vs GB/s)

Output a JSON array. For each question:
{{
  "id": "question-id",
  "status": "CORRECT|ERROR|WARN",
  "hardware_specs_correct": true|false,
  "arithmetic_verified": true|false,
  "units_consistent": true|false,
  "conclusion_follows": true|false,
  "recalculation": "Brief step-by-step of the key calculation",
  "issues": ["specific issue descriptions"],
  "corrections": ["exact fix needed"],
  "severity": "critical|moderate|minor|none"
}}

Rules:
- CORRECT: everything checks out
- ERROR: wrong answer, wrong spec, wrong formula, or wrong conclusion
- WARN: approximation is loose but defensible, or minor inconsistency
- Be STRICT on hardware specs (use the authoritative list)
- Be STRICT on arithmetic (recalculate, don't trust)
- Be LENIENT on reasonable approximations (±10% is fine for napkin math)

Output ONLY the JSON array.

--- QUESTIONS TO VERIFY ---
{batch_text}"""


def run_claude_verify(prompt, timeout=300):
    """Run verification through Claude CLI."""
    try:
        r = subprocess.run(
            ["claude", "-p", prompt, "--model", "opus", "--output-format", "json"],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0:
            return None, f"CLI error: {r.stderr[:200]}"

        # Parse response — claude CLI with --output-format json wraps in a structure
        try:
            response = json.loads(r.stdout)
            # Extract the text content
            if isinstance(response, dict) and "result" in response:
                text = response["result"]
            elif isinstance(response, dict) and "content" in response:
                text = response["content"]
            elif isinstance(response, str):
                text = response
            else:
                text = r.stdout

            # Strip markdown fences
            import re
            if isinstance(text, str) and text.strip().startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text.strip())
                text = re.sub(r"\n?```$", "", text)

            return json.loads(text.strip()), None
        except json.JSONDecodeError:
            # Try to find JSON array in the output
            import re
            match = re.search(r'\[[\s\S]*\]', r.stdout)
            if match:
                return json.loads(match.group()), None
            return None, f"JSON parse error in response"

    except subprocess.TimeoutExpired:
        return None, "Timeout"
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Deep math verification with Opus")
    parser.add_argument("--chunk-size", type=int, default=10,
                       help="Questions per verification call (smaller = more thorough)")
    parser.add_argument("--sample", type=int, default=0,
                       help="Random sample size (0=all published)")
    parser.add_argument("--topic", type=str, default=None,
                       help="Verify only this topic")
    parser.add_argument("--errors-only", action="store_true",
                       help="Re-verify only previously flagged errors")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show plan without calling API")
    parser.add_argument("--workers", type=int, default=1,
                       help="Parallel verification workers")
    parser.add_argument("--stratified", action="store_true",
                       help="Stratified sample (N per topic)")
    parser.add_argument("--per-topic", type=int, default=3,
                       help="Questions per topic for stratified sampling")
    args = parser.parse_args()

    corpus = json.load(open(CORPUS_PATH))
    published = [q for q in corpus if q.get("status") == "published"]

    # Select questions to verify
    if args.errors_only:
        to_verify = [q for q in published if q.get("math_status") == "ERROR"]
        print(f"Re-verifying {len(to_verify)} previously flagged errors")
    elif args.topic:
        to_verify = [q for q in published if q.get("topic") == args.topic]
        print(f"Verifying topic '{args.topic}': {len(to_verify)} questions")
    elif args.stratified:
        by_topic = defaultdict(list)
        for q in published:
            by_topic[q.get("topic", "")].append(q)
        to_verify = []
        for topic, qs in sorted(by_topic.items()):
            n = min(args.per_topic, len(qs))
            to_verify.extend(random.sample(qs, n))
        print(f"Stratified sample: {args.per_topic}/topic = {len(to_verify)} questions")
    elif args.sample:
        to_verify = random.sample(published, min(args.sample, len(published)))
        print(f"Random sample: {len(to_verify)} questions")
    else:
        to_verify = published

    n_chunks = (len(to_verify) + args.chunk_size - 1) // args.chunk_size

    print(f"\nDeep Verification Plan:")
    print(f"  Questions: {len(to_verify)}")
    print(f"  Chunk size: {args.chunk_size} (smaller = more thorough)")
    print(f"  API calls: {n_chunks}")
    print(f"  Model: Claude Opus (extended thinking)")

    if args.dry_run:
        print("\nDRY RUN — no API calls")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = RESULTS_DIR / f"deep-verify-{timestamp}.json"

    all_results = []
    stats = Counter()
    start = time.time()

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * args.chunk_size
        chunk_end = min(chunk_start + args.chunk_size, len(to_verify))
        chunk = to_verify[chunk_start:chunk_end]

        prompt = build_deep_verify_prompt(chunk)
        results, error = run_claude_verify(prompt)

        if error:
            print(f"  [{chunk_idx+1}/{n_chunks}] ERROR: {error}")
            stats["api_error"] += 1
            continue

        if results:
            for r in results:
                all_results.append(r)
                status = r.get("status", "UNKNOWN")
                stats[status] += 1

                if status == "ERROR":
                    qid = r.get("id", "?")
                    severity = r.get("severity", "?")
                    issues = r.get("issues", [])
                    recalc = r.get("recalculation", "")
                    print(f"    ❌ {qid} [{severity}]: {'; '.join(issues[:2])}")
                    if recalc:
                        print(f"       Recalc: {recalc[:100]}")

            # Stamp results into corpus
            results_by_id = {r["id"]: r for r in results if "id" in r}
            for q in corpus:
                if q.get("id") in results_by_id:
                    r = results_by_id[q["id"]]
                    q["math_verified"] = True
                    q["math_status"] = r.get("status", "UNKNOWN")
                    q["math_issues"] = r.get("issues", [])
                    q["math_corrections"] = r.get("corrections", [])
                    q["math_severity"] = r.get("severity", "none")
                    q["math_model"] = "claude-opus-4-6"
                    q["math_date"] = datetime.now().strftime("%Y-%m-%d")

        done = chunk_end
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  [{chunk_idx+1}/{n_chunks}] "
              f"done={done} correct={stats.get('CORRECT',0)} "
              f"error={stats.get('ERROR',0)} warn={stats.get('WARN',0)} "
              f"({rate:.1f} q/s)")

        time.sleep(1)

    # Save
    json.dump(all_results, open(results_file, "w"), indent=2)
    json.dump(corpus, open(CORPUS_PATH, "w"), indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    total_verified = sum(stats.values()) - stats.get("api_error", 0)
    print(f"\n{'='*60}")
    print(f"DEEP VERIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"Verified: {total_verified}")
    print(f"  CORRECT: {stats.get('CORRECT', 0)} ({100*stats.get('CORRECT',0)/max(total_verified,1):.1f}%)")
    print(f"  ERROR:   {stats.get('ERROR', 0)} ({100*stats.get('ERROR',0)/max(total_verified,1):.1f}%)")
    print(f"  WARN:    {stats.get('WARN', 0)} ({100*stats.get('WARN',0)/max(total_verified,1):.1f}%)")
    print(f"Time: {elapsed:.0f}s")
    print(f"Results: {results_file}")

    if stats.get("ERROR", 0) > 0:
        error_rate = stats["ERROR"] / total_verified
        projected = int(error_rate * len(published))
        print(f"\nProjected errors corpus-wide: ~{projected}")
        print(f"Run with --errors-only to re-verify after fixing")


if __name__ == "__main__":
    main()
