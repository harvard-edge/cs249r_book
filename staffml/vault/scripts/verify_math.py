#!/usr/bin/env python3
"""Gemini-powered math verification pass for StaffML corpus.

Sends chunks of questions to gemini-3.1-pro-preview for independent
math verification. Each call checks ~25 questions. With 250 calls/day
quota, this covers ~6,250 questions per day.

Usage:
    python3 scripts/verify_math.py                    # Verify all unverified
    python3 scripts/verify_math.py --chunk-size 20    # Smaller chunks
    python3 scripts/verify_math.py --limit 500        # Only first 500
    python3 scripts/verify_math.py --dry-run           # Show plan without calling
"""

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent.parent
CORPUS_PATH = BASE / "corpus.json"
RESULTS_DIR = BASE / "scripts" / "_verification_results"

MODEL = "gemini-3.1-pro-preview"


def build_verification_prompt(questions):
    """Build a prompt asking Gemini to verify math in a batch of questions."""
    q_texts = []
    for i, q in enumerate(questions):
        details = q.get("details", {})
        q_texts.append(
            f"Q{i + 1} [id={q['id']}]: {q.get('title', 'untitled')}\n"
            f"  Scenario: {q.get('scenario', 'N/A')}\n"
            f"  Napkin Math: {details.get('napkin_math', 'N/A')}\n"
            f"  Solution: {details.get('realistic_solution', 'N/A')}\n"
            f"  Hardware: topic={q.get('topic')}, track={q.get('track')}"
        )

    batch_text = "\n\n".join(q_texts)

    return f"""You are a meticulous ML systems math verifier. For each question below,
check the napkin math and hardware specs for correctness. Output ONLY a JSON array
where each element is:
{{"id": "question-id", "status": "CORRECT|ERROR|WARN", "issues": ["issue1", ...], "corrections": ["fix1", ...]}}

Rules:
- CORRECT: math is right, specs are accurate
- ERROR: math produces wrong answer, or hardware spec is factually wrong
- WARN: minor issue (rounding, approximation is reasonable but imprecise)

Check specifically:
1. Are hardware specs accurate? (H100: 80GB HBM3, 3.35TB/s, 989 TFLOPS FP16; A100: 80GB HBM2e, 2TB/s, 312 TFLOPS; MI300X: 192GB HBM3, 5.3TB/s; Jetson Orin: 32GB LPDDR5, 275 TOPS)
2. Is the arithmetic correct? (multiplication, division, unit conversions)
3. Are the formulas correct? (roofline, KV-cache sizing, model memory, AllReduce time)
4. Are the conclusions consistent with the math?

Output ONLY the JSON array, no markdown, no explanation.

--- QUESTIONS TO VERIFY ---

{batch_text}"""


def call_gemini(prompt, timeout=300):
    """Call gemini CLI and parse JSON response."""
    try:
        r = subprocess.run(
            ["gemini", "-m", MODEL, "-o", "text"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode != 0:
            return None, r.stderr[:200]

        text = r.stdout.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        return json.loads(text.strip()), None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
    except subprocess.TimeoutExpired:
        return None, "Timeout"
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Verify math in StaffML corpus")
    parser.add_argument(
        "--chunk-size", type=int, default=25, help="Questions per API call"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max questions to verify (0=all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show plan without calling API"
    )
    parser.add_argument(
        "--unverified-only",
        action="store_true",
        help="Only verify questions not yet math-verified",
    )
    args = parser.parse_args()

    corpus = json.load(open(CORPUS_PATH))
    published = [q for q in corpus if q.get("status") == "published"]

    if args.unverified_only:
        to_verify = [q for q in published if not q.get("math_verified")]
    else:
        to_verify = published

    if args.limit:
        to_verify = to_verify[: args.limit]

    n_chunks = (len(to_verify) + args.chunk_size - 1) // args.chunk_size

    print(f"Math Verification Pass")
    print(f"  Model: {MODEL}")
    print(f"  Questions to verify: {len(to_verify)}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  API calls needed: {n_chunks}")
    print(f"  Estimated time: {n_chunks * 30 // 60} min")
    print()

    if n_chunks > 250:
        print(f"  WARNING: {n_chunks} calls exceeds daily quota of ~250")
        print(f"  Will need {(n_chunks + 249) // 250} days to complete")
        print()

    if args.dry_run:
        print("DRY RUN — no API calls made")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = RESULTS_DIR / f"math-verify-{timestamp}.json"

    all_results = []
    errors = 0
    warnings = 0
    correct = 0
    api_errors = 0

    start = time.time()

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * args.chunk_size
        chunk_end = min(chunk_start + args.chunk_size, len(to_verify))
        chunk = to_verify[chunk_start:chunk_end]

        prompt = build_verification_prompt(chunk)
        results, error = call_gemini(prompt)

        if error:
            api_errors += 1
            print(f"  [{chunk_idx + 1}/{n_chunks}] API ERROR: {error}")
            if "QuotaError" in str(error) or "exhausted" in str(error):
                print("  QUOTA EXHAUSTED — stopping. Resume tomorrow.")
                break
            continue

        if results:
            for r in results:
                all_results.append(r)
                status = r.get("status", "UNKNOWN")
                if status == "CORRECT":
                    correct += 1
                elif status == "ERROR":
                    errors += 1
                    qid = r.get("id", "?")
                    issues = r.get("issues", [])
                    print(f"    ERROR {qid}: {'; '.join(issues[:2])}")
                elif status == "WARN":
                    warnings += 1

            # Stamp results back into corpus
            results_by_id = {r["id"]: r for r in results if "id" in r}
            for q in corpus:
                if q.get("id") in results_by_id:
                    r = results_by_id[q["id"]]
                    q["math_verified"] = True
                    q["math_status"] = r.get("status", "UNKNOWN")
                    q["math_issues"] = r.get("issues", [])
                    q["math_corrections"] = r.get("corrections", [])
                    q["math_model"] = MODEL
                    q["math_date"] = datetime.now().strftime("%Y-%m-%d")

        done = chunk_end
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        print(
            f"  [{chunk_idx + 1}/{n_chunks}] "
            f"verified={done} correct={correct} "
            f"errors={errors} warnings={warnings} "
            f"({rate:.1f} q/s)"
        )

        # Rate limiting — don't hammer the API
        time.sleep(2)

    # Save results
    json.dump(all_results, open(results_file, "w"), indent=2)
    json.dump(corpus, open(CORPUS_PATH, "w"), indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print(f"\nDone: {correct} correct, {errors} errors, {warnings} warnings")
    print(f"API errors: {api_errors}")
    print(f"Time: {elapsed:.0f}s")
    print(f"Results: {results_file}")
    print(f"Corpus updated with math_verified stamps")

    if errors > 0:
        print(f"\n{errors} questions need math corrections.")
        print(f"Run: python3 scripts/verify_math.py --unverified-only")


if __name__ == "__main__":
    main()
