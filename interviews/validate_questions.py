#!/usr/bin/env python3
"""Parallel Gemini validation of corpus questions.

Validates math correctness, factual accuracy, and question quality
using gemini-3.1-pro-preview across parallel batches.

Usage:
    python3 validate_questions.py                    # Validate all 4,779 questions
    python3 validate_questions.py --new-only         # Only validate the 285 newly generated
    python3 validate_questions.py --ka F1            # Only validate one knowledge area
    python3 validate_questions.py --sample 200       # Random sample of 200
    python3 validate_questions.py --batch-size 25    # Customize batch size
    python3 validate_questions.py --workers 12       # Customize parallelism
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import subprocess

BASE = Path(__file__).parent
CORPUS_PATH = BASE / "corpus.json"
RESULTS_DIR = BASE / "_validation_results"

MODEL = "gemini-2.5-flash"

# ─── Gemini Client ────────────────────────────────────────────
# Try API first (fast), fall back to CLI (uses cached OAuth)
# Use --cli flag to force CLI mode

_use_api = False
_client = None
_force_cli = "--cli" in sys.argv


def init_gemini():
    """Initialize Gemini client. Call after argparse."""
    global _use_api, _client
    if _force_cli:
        print(f"  Using Gemini CLI (forced via --cli)")
        return
    try:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key and "expired" not in api_key.lower():
            _client = genai.Client(api_key=api_key)
            _test = _client.models.generate_content(model=MODEL, contents="Say OK")
            if _test.text:
                _use_api = True
                print(f"  Using Gemini API (fast mode)")
                return
    except Exception:
        pass
    print(f"  Using Gemini CLI (cached credentials)")


def call_gemini(prompt: str, retries: int = 2) -> str | None:
    """Call Gemini — API if available, CLI fallback."""
    for attempt in range(retries + 1):
        try:
            if _use_api:
                response = _client.models.generate_content(
                    model=MODEL, contents=prompt,
                    config={"temperature": 0.1, "max_output_tokens": 65000},
                )
                text = response.text.strip()
            else:
                # Pipe prompt via stdin to avoid ARG_MAX limits on large batches
                result = subprocess.run(
                    ["gemini", "-m", MODEL, "-o", "text"],
                    input=prompt, capture_output=True, text=True, timeout=300,
                )
                if result.returncode != 0:
                    if attempt < retries:
                        time.sleep(2 ** attempt)
                        continue
                    return None
                text = result.stdout.strip()

            # Strip markdown fences
            if text.startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            return text.strip()
        except subprocess.TimeoutExpired:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                return None
        except Exception as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                print(f"  Gemini error: {e}")
                return None


# ─── Validation Prompt ────────────────────────────────────────

VALIDATION_PROMPT = """You are a rigorous technical reviewer for Staff-level ML Systems interview questions. Review each question for:

1. **Math correctness**: Are all calculations, napkin math, and numerical claims correct? Check arithmetic, unit conversions, hardware specs (e.g., A100 = 2 TB/s HBM BW, 312 TFLOPS FP16; H100 = 3.35 TB/s, 989 TFLOPS FP16).
2. **Factual accuracy**: Are hardware specs, algorithm descriptions, and systems claims correct? Flag outdated or wrong numbers.
3. **Question quality**: Is the scenario clear? Is there exactly one correct answer? Is the common_mistake plausible? Is the realistic_solution actually correct?
4. **Classification sanity**: Does the reasoning_competency match what the question tests? Does the reasoning_mode match the question format?

For each question, output ONE JSON object:
```json
{"id": "<question-id>", "status": "OK|WARN|ERROR", "issues": ["issue1", "issue2"], "fixes": ["fix1", "fix2"]}
```

Rules:
- "OK" = no issues found
- "WARN" = minor issues (slightly imprecise numbers, could be clearer)
- "ERROR" = math wrong, factually incorrect, or fundamentally broken question
- Keep issues and fixes concise (one sentence each)
- For OK questions, issues and fixes should be empty arrays

Return a JSON array of review objects, one per question. Return ONLY the JSON array, no markdown fences.

QUESTIONS TO REVIEW:
"""


def build_batch_prompt(questions: list[dict]) -> str:
    """Build a validation prompt for a batch of questions."""
    q_text = ""
    for q in questions:
        details = q.get("details", {})
        q_text += f"""
---
ID: {q['id']}
Title: {q['title']}
Level: {q['level']} | Track: {q['track']} | RC: {q.get('reasoning_competency')} | KA: {q.get('knowledge_area')} | Mode: {q.get('reasoning_mode')}
Scenario: {q['scenario'][:500]}
Common Mistake: {details.get('common_mistake', '')[:300]}
Realistic Solution: {details.get('realistic_solution', '')[:500]}
Napkin Math: {details.get('napkin_math', '')[:500]}
"""
    return VALIDATION_PROMPT + q_text


def parse_review_response(text: str) -> list[dict] | None:
    """Parse JSON array from Gemini response."""
    if not text:
        return None
    # Strip markdown fences
    text = re.sub(r"^```\w*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return None
    except json.JSONDecodeError:
        # Try to find JSON array in the response
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None


# ─── Main Pipeline ────────────────────────────────────────────

def validate_batch(batch_idx: int, questions: list[dict]) -> list[dict]:
    """Validate a batch of questions via Gemini."""
    prompt = build_batch_prompt(questions)
    text = call_gemini(prompt)
    reviews = parse_review_response(text)

    if reviews is None:
        print(f"  Batch {batch_idx}: PARSE FAILED (will retry)")
        # Retry once
        text = call_gemini(prompt)
        reviews = parse_review_response(text)

    if reviews is None:
        # Return error for each question
        return [{"id": q["id"], "status": "PARSE_ERROR", "issues": ["Gemini response unparsable"], "fixes": []} for q in questions]

    return reviews


def main():
    parser = argparse.ArgumentParser(description="Validate corpus questions via Gemini")
    parser.add_argument("--new-only", action="store_true", help="Only validate newly generated questions")
    parser.add_argument("--ka", type=str, help="Only validate one knowledge area (e.g., F1)")
    parser.add_argument("--sample", type=int, help="Random sample of N questions")
    parser.add_argument("--batch-size", type=int, default=200, help="Questions per Gemini call (default: 200)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    parser.add_argument("--cli", action="store_true", help="Force Gemini CLI mode (OAuth, no API key)")
    args = parser.parse_args()

    # Initialize Gemini client
    init_gemini()

    # Load corpus
    corpus = json.load(open(CORPUS_PATH))
    print(f"Corpus: {len(corpus)} questions")

    # Filter
    if args.new_only:
        # New questions don't have certain legacy fields
        questions = [q for q in corpus if q.get("status") is None and q.get("version") is None]
        if not questions:
            # Fallback: questions without 'tags' field (old questions have it)
            questions = [q for q in corpus if "tags" not in q]
        if not questions:
            # Last resort: questions with IDs matching gen pattern
            gen_prefixes = tuple(f"{t}-{ka.lower()}-" for t in ["cloud", "global", "edge", "mobile"]
                                for ka in ["f1", "a1", "a2", "a3", "a4", "a6", "b4", "b6", "b7", "b8", "c4", "c7", "c8", "c9", "d1", "e3"])
            questions = [q for q in corpus if q["id"].startswith(gen_prefixes)]
        print(f"  Filtered to {len(questions)} new questions")
    elif args.ka:
        questions = [q for q in corpus if q.get("knowledge_area") == args.ka]
        print(f"  Filtered to {len(questions)} questions in {args.ka}")
    else:
        questions = corpus

    if args.sample and args.sample < len(questions):
        random.seed(42)
        questions = random.sample(questions, args.sample)
        print(f"  Sampled {len(questions)} questions")

    # Batch
    batch_size = args.batch_size
    batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]
    print(f"  {len(batches)} batches × {batch_size} questions = {len(questions)} total")
    print(f"  {args.workers} parallel workers")
    print(f"  Model: {MODEL}")
    print()

    # Run parallel validation
    all_reviews = []
    errors_count = 0
    warns_count = 0
    ok_count = 0
    parse_errors = 0

    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(validate_batch, i, batch): i for i, batch in enumerate(batches)}

        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                reviews = future.result()
                all_reviews.extend(reviews)

                for r in reviews:
                    status = r.get("status", "?")
                    if status == "ERROR":
                        errors_count += 1
                    elif status == "WARN":
                        warns_count += 1
                    elif status == "OK":
                        ok_count += 1
                    elif status == "PARSE_ERROR":
                        parse_errors += 1

                done = ok_count + warns_count + errors_count + parse_errors
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  Batch {batch_idx:>3}/{len(batches)}: "
                      f"OK={ok_count} WARN={warns_count} ERR={errors_count} "
                      f"PARSE_ERR={parse_errors} [{done}/{len(questions)} @ {rate:.1f} Q/s]")

            except Exception as e:
                print(f"  Batch {batch_idx}: EXCEPTION: {e}")

    elapsed = time.time() - start

    # ─── Report ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total: {len(all_reviews)} reviewed in {elapsed:.0f}s")
    print(f"  OK:         {ok_count} ({ok_count/max(len(all_reviews),1)*100:.1f}%)")
    print(f"  WARN:       {warns_count} ({warns_count/max(len(all_reviews),1)*100:.1f}%)")
    print(f"  ERROR:      {errors_count} ({errors_count/max(len(all_reviews),1)*100:.1f}%)")
    print(f"  PARSE_ERR:  {parse_errors}")

    # Collect errors and warnings
    issues_by_status = defaultdict(list)
    for r in all_reviews:
        if r.get("status") in ("ERROR", "WARN"):
            issues_by_status[r["status"]].append(r)

    if issues_by_status.get("ERROR"):
        print(f"\n  ── ERRORS ({len(issues_by_status['ERROR'])}) ──")
        for r in issues_by_status["ERROR"][:30]:
            print(f"    [{r['id']}]")
            for issue in r.get("issues", []):
                print(f"      ✗ {issue}")
            for fix in r.get("fixes", []):
                print(f"      → {fix}")

    if issues_by_status.get("WARN"):
        print(f"\n  ── WARNINGS ({len(issues_by_status['WARN'])}) ──")
        for r in issues_by_status["WARN"][:20]:
            print(f"    [{r['id']}]")
            for issue in r.get("issues", []):
                print(f"      ⚠ {issue}")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    today = datetime.now().strftime("%Y-%m-%d")
    results_path = RESULTS_DIR / f"validation-{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "model": MODEL,
            "total_reviewed": len(all_reviews),
            "ok": ok_count,
            "warn": warns_count,
            "error": errors_count,
            "parse_errors": parse_errors,
            "elapsed_seconds": round(elapsed, 1),
            "reviews": all_reviews,
        }, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # ─── Stamp validation into corpus.json ────────────────────
    review_map = {r["id"]: r for r in all_reviews if r.get("id")}
    stamped = 0
    for q in corpus:
        review = review_map.get(q["id"])
        if review:
            status = review.get("status", "PARSE_ERROR")
            q["validated"] = status == "OK"
            q["validation_status"] = status
            q["validation_issues"] = review.get("issues", [])
            q["validation_model"] = MODEL
            q["validation_date"] = today
            stamped += 1

    if stamped > 0:
        with open(CORPUS_PATH, "w") as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"  Stamped {stamped} questions in corpus.json")
        print(f"    validated=true:  {sum(1 for q in corpus if q.get('validated') is True)}")
        print(f"    validated=false: {sum(1 for q in corpus if q.get('validated') is False)}")
        print(f"    not yet checked: {sum(1 for q in corpus if q.get('validated') is None)}")


if __name__ == "__main__":
    main()
