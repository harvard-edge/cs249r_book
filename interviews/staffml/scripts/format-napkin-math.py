#!/usr/bin/env python3
"""
Convert all napkin_math fields in corpus.json to a clean line-per-step format.

Input formats handled:
  - Pipe-separated: "step1 | step2 | => result"
  - Already newline-separated (kept as-is)
  - Numbered lists: "1. step 2. step"
  - Dense blobs: split on sentence boundaries after calculations

Output format:
  Each step on its own line.
  Final answer lines prefixed with "=> "
"""

import json
import re
import sys
from pathlib import Path

CORPUS_PATH = Path(__file__).parent.parent / "src" / "data" / "corpus.json"


def format_napkin(text: str) -> str:
    text = text.strip()
    if not text:
        return text

    # Already has newlines with content — likely already structured
    lines = text.split("\n")
    if len(lines) > 1 and all(l.strip() for l in lines[:3]):
        return text  # keep as-is

    # Pipe-separated → newline-separated
    if " | " in text:
        parts = [p.strip() for p in text.split(" | ")]
        return "\n".join(parts)

    # Numbered list without newlines: "1. foo 2. bar 3. baz"
    numbered = re.split(r"(?:^|\s)(\d+)\.\s+", text)
    if len(numbered) > 3:  # split produces [prefix, num, text, num, text, ...]
        steps = []
        for i in range(1, len(numbered) - 1, 2):
            step_text = numbered[i + 1].strip().rstrip(".")
            if step_text:
                steps.append(f"{numbered[i]}. {step_text}")
        if steps:
            return "\n".join(steps)

    # Short text (< 150 chars) — keep as one line
    if len(text) < 150:
        return text

    # Dense blob: split on sentence boundaries
    # Pattern: end of sentence (period after number/word/paren) followed by capital letter
    sentences = re.split(r"(?<=[\d)%a-zA-Z])\.\s+(?=[A-Z])", text)
    if len(sentences) > 1:
        result = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            # Add period back if it was consumed by the split
            if not s.endswith((".",")","%")):
                s += "."
            # Mark final-answer lines
            if re.search(r"=\s*[\d,]+(?:\.\d+)?\s*(?:seconds?|minutes?|ms|GB|MB|TB|hours?|days?|TFLOPS)", s):
                if "=>" not in s:
                    # Find the last "= VALUE UNIT" and prefix with =>
                    s = re.sub(
                        r"=\s*([\d,]+(?:\.\d+)?\s*(?:seconds?|minutes?|ms|GB|MB|TB|hours?|days?|TFLOPS|samples?/sec)(?:\s*\([^)]+\))?)\s*\.?$",
                        r"=> \1",
                        s,
                    )
            result.append(s)
        if len(result) > 1:
            return "\n".join(result)

    # Fallback: try splitting on "; " (semicolons often separate steps)
    if "; " in text and text.count("; ") >= 2:
        parts = [p.strip() for p in text.split("; ")]
        return "\n".join(parts)

    # Last resort: keep as-is
    return text


def main():
    with open(CORPUS_PATH, "r") as f:
        corpus = json.load(f)

    changed = 0
    for q in corpus:
        if not q.get("details", {}).get("napkin_math"):
            continue
        original = q["details"]["napkin_math"]
        formatted = format_napkin(original)
        if formatted != original:
            q["details"]["napkin_math"] = formatted
            changed += 1

    with open(CORPUS_PATH, "w") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Formatted {changed} / {len(corpus)} napkin_math fields")

    # Verify
    with open(CORPUS_PATH, "r") as f:
        verify = json.load(f)
    multi = sum(1 for q in verify if q.get("details", {}).get("napkin_math", "").count("\n") > 0)
    print(f"Now have newlines: {multi}")


if __name__ == "__main__":
    main()
