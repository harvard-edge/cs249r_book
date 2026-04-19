#!/usr/bin/env python3
"""gemini_math_check.py

Wrapper around the ``gemini`` CLI for the MLSysBook camera-ready overnight
sweep. Two modes:

  classify     classify a single equation given context + (optional) notation
  revalidate   re-classify a set of equations in a chapter post-edit

Hard model pin: ``gemini-3.1-pro-preview``. No fallback model is permitted.

Per-volume wall-time budget tracked in
``${RUN_DIR}/qa/gemini-budget-vol<N>.txt`` (max 90 minutes per volume). Once
the budget is exhausted, all subsequent calls return immediately with
``verdict: gemini-budget-exceeded`` (still exit 0 for orchestrator clarity).

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

GEMINI_MODEL = "gemini-3.1-pro-preview"
GEMINI_TIMEOUT_SEC = 60
RETRY_BACKOFFS_SEC = (1, 4, 16)
PER_VOL_BUDGET_SEC = 90 * 60

VERDICTS = {
    "clean",
    "undefined-symbol",
    "unit-mismatch",
    "prose-equation-mismatch",
    "latex-syntax",
    "notation-inconsistency",
    "needs-human",
    "gemini-unavailable",
    "gemini-budget-exceeded",
}

DIRECTIVE_PROMPT = (
    "You are a meticulous mathematical proofreader for an MIT Press machine "
    "learning systems textbook. You are given an equation in LaTeX and the "
    "surrounding prose context. Your job is to classify the equation. Return "
    "ONLY a JSON object matching this exact schema and nothing else, no prose, "
    "no markdown fence:\n\n"
    '{"verdict": "...", "confidence": "...", "reasoning": "...", '
    '"suggested_fix": "...", "suggested_fix_safe_to_auto_apply": true|false}\n\n'
    "Verdict must be one of: clean, undefined-symbol, unit-mismatch, "
    "prose-equation-mismatch, latex-syntax, notation-inconsistency, needs-human.\n\n"
    "Confidence must be one of: high, medium, low.\n\n"
    "Reasoning is a single paragraph (no line breaks) explaining your "
    "classification. Be specific about which symbol is undefined or which "
    "units don't match if applicable.\n\n"
    "suggested_fix is a LaTeX string that would correct the equation, or null "
    "if no fix is needed or you can't determine one safely.\n\n"
    "suggested_fix_safe_to_auto_apply must be true ONLY if (a) verdict is "
    "latex-syntax, (b) confidence is high, AND (c) the fix is purely "
    "syntactic (e.g., adding a missing brace, escaping an underscore, fixing "
    "a typo in a command name). For ANY semantic change (variable name, "
    "structure, sign, exponent, subscript meaning), this MUST be false.\n\n"
    "The equation, context, and notation table follow."
)


# ---------------------------------------------------------------------------
# Budget bookkeeping
# ---------------------------------------------------------------------------
def _budget_file(vol: str) -> Optional[Path]:
    run_dir = os.environ.get("RUN_DIR")
    if not run_dir:
        return None
    return Path(run_dir) / "qa" / f"gemini-budget-{vol}.txt"


def _read_budget(vol: str) -> float:
    bf = _budget_file(vol)
    if bf is None or not bf.exists():
        return 0.0
    try:
        return float(bf.read_text().strip() or "0")
    except Exception:
        return 0.0


def _add_budget(vol: str, secs: float) -> float:
    bf = _budget_file(vol)
    if bf is None:
        return 0.0
    bf.parent.mkdir(parents=True, exist_ok=True)
    cur = _read_budget(vol)
    new_total = cur + max(0.0, secs)
    bf.write_text(f"{new_total:.2f}\n")
    return new_total


def _budget_exhausted(vol: str) -> bool:
    return _read_budget(vol) >= PER_VOL_BUDGET_SEC


# ---------------------------------------------------------------------------
# Gemini invocation
# ---------------------------------------------------------------------------
def _build_full_prompt(
    equation_latex: str,
    context_before: str,
    context_after: str,
    notation_table: str,
) -> str:
    parts = [
        DIRECTIVE_PROMPT,
        "\n\n--- CONTEXT BEFORE ---\n",
        context_before.strip() or "(none)",
        "\n\n--- EQUATION (LaTeX) ---\n",
        equation_latex.strip(),
        "\n\n--- CONTEXT AFTER ---\n",
        context_after.strip() or "(none)",
        "\n\n--- NOTATION TABLE ---\n",
        notation_table.strip() or "(none)",
    ]
    return "".join(parts)


def _parse_gemini_output(stdout: str) -> Dict:
    """Extract the inner JSON object from gemini -o json output."""
    stdout = stdout.strip()
    try:
        envelope = json.loads(stdout)
    except Exception:
        envelope = None

    candidate_text = ""
    if isinstance(envelope, dict):
        candidate_text = (
            envelope.get("response")
            or envelope.get("text")
            or envelope.get("output")
            or ""
        )
    if not candidate_text:
        candidate_text = stdout

    # Strip optional markdown fence if the model added one despite instructions.
    candidate_text = candidate_text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate_text, re.S)
    if fence:
        candidate_text = fence.group(1)

    try:
        obj = json.loads(candidate_text)
        if isinstance(obj, dict) and "verdict" in obj:
            return obj
    except Exception:
        pass
    # Last-ditch: search for the first {...} JSON object in the text.
    m = re.search(r"\{.*\}", candidate_text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {"verdict": "needs-human", "confidence": "low",
            "reasoning": "could not parse gemini response",
            "suggested_fix": None,
            "suggested_fix_safe_to_auto_apply": False,
            "_raw": stdout[:2000]}


def _invoke_gemini(prompt: str) -> Dict:
    """Invoke gemini with retries; returns a dict with parsed verdict + meta."""
    started = time.monotonic()
    last_err = None
    for attempt, backoff in enumerate(RETRY_BACKOFFS_SEC, start=1):
        try:
            cp = subprocess.run(
                [
                    "gemini",
                    "-m", GEMINI_MODEL,
                    "-o", "json",
                    "--approval-mode", "plan",
                    "-p", prompt,
                ],
                check=False,
                text=True,
                capture_output=True,
                timeout=GEMINI_TIMEOUT_SEC,
            )
            wall = time.monotonic() - started
            if cp.returncode == 0 and cp.stdout.strip():
                parsed = _parse_gemini_output(cp.stdout)
                parsed["gemini_wall_time_sec"] = round(wall, 3)
                parsed["retries"] = attempt - 1
                return parsed
            last_err = f"rc={cp.returncode} stderr={cp.stderr[:300]}"
        except subprocess.TimeoutExpired:
            last_err = "timeout"
        except Exception as e:
            last_err = f"exception:{e}"
        if attempt < len(RETRY_BACKOFFS_SEC):
            time.sleep(backoff)
    wall = time.monotonic() - started
    return {
        "verdict": "gemini-unavailable",
        "confidence": "low",
        "reasoning": f"gemini failed after {len(RETRY_BACKOFFS_SEC)} attempts: {last_err}",
        "suggested_fix": None,
        "suggested_fix_safe_to_auto_apply": False,
        "gemini_wall_time_sec": round(wall, 3),
        "retries": len(RETRY_BACKOFFS_SEC),
    }


def classify(
    equation_latex: str,
    context_before: str,
    context_after: str,
    notation_table_path: str,
    vol: str,
) -> Dict:
    if _budget_exhausted(vol):
        return {
            "verdict": "gemini-budget-exceeded",
            "confidence": "low",
            "reasoning": "per-volume gemini wall-time budget exhausted",
            "suggested_fix": None,
            "suggested_fix_safe_to_auto_apply": False,
            "gemini_wall_time_sec": 0,
            "retries": 0,
        }

    notation_table = ""
    if notation_table_path and Path(notation_table_path).is_file():
        try:
            notation_table = Path(notation_table_path).read_text(
                encoding="utf-8", errors="replace"
            )
        except Exception:
            notation_table = ""

    prompt = _build_full_prompt(equation_latex, context_before, context_after, notation_table)
    result = _invoke_gemini(prompt)
    _add_budget(vol, result.get("gemini_wall_time_sec", 0))

    # Normalize verdict
    if result.get("verdict") not in VERDICTS:
        result["verdict"] = "needs-human"
    return result


# ---------------------------------------------------------------------------
# Mode 2: revalidate
# ---------------------------------------------------------------------------
DOLLAR_BLOCK_RE = re.compile(r"\$\$([^$]+)\$\$", re.S)
DOLLAR_INLINE_RE = re.compile(r"(?<!\$)\$([^\$\n]+)\$(?!\$)")


def _extract_equation_at(line_idx: int, qmd_lines: list) -> Optional[str]:
    """Best-effort: find the equation containing the given 1-based line."""
    text = "".join(qmd_lines)
    line_offsets = [0]
    for ln in qmd_lines:
        line_offsets.append(line_offsets[-1] + len(ln))
    if line_idx < 1 or line_idx > len(qmd_lines):
        return None
    char_pos = line_offsets[line_idx - 1]
    for m in DOLLAR_BLOCK_RE.finditer(text):
        if m.start() <= char_pos <= m.end():
            return m.group(1).strip()
    line = qmd_lines[line_idx - 1]
    m = DOLLAR_INLINE_RE.search(line)
    if m:
        return m.group(1).strip()
    return None


def revalidate(chapter_qmd: str, line_numbers: str) -> int:
    qmd_path = Path(chapter_qmd)
    if not qmd_path.is_file():
        print(f"revalidate: missing chapter qmd {qmd_path}", file=sys.stderr)
        return 1
    vol = "vol1" if "/vol1/" in str(qmd_path) else "vol2"
    qmd_lines = qmd_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    nums = [int(n) for n in line_numbers.split(",") if n.strip().isdigit()]

    any_unclean = False
    for ln in nums:
        eq = _extract_equation_at(ln, qmd_lines)
        if not eq:
            continue
        ctx_before = "".join(qmd_lines[max(0, ln - 6) : ln - 1])
        ctx_after = "".join(qmd_lines[ln : ln + 5])
        result = classify(eq, ctx_before, ctx_after, "", vol)
        verdict = result.get("verdict", "needs-human")
        if verdict == "clean":
            print(f"line {ln}: clean")
        else:
            any_unclean = True
            print(f"line {ln}: {verdict} ({result.get('reasoning','')[:140]})")
            if verdict in {"gemini-unavailable", "gemini-budget-exceeded"}:
                # These are not blocking by definition for revalidation;
                # but per spec, anything other than clean is treated as failure.
                pass
    return 1 if any_unclean else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("classify")
    p1.add_argument("--equation-latex", required=True)
    p1.add_argument("--context-before", default="")
    p1.add_argument("--context-after", default="")
    p1.add_argument("--notation-table-path", default="")
    p1.add_argument("--vol", required=True, choices=["vol1", "vol2"])
    p1.add_argument("--out-json", required=True)

    p2 = sub.add_parser("revalidate")
    p2.add_argument("--chapter-qmd", required=True)
    p2.add_argument("--equation-line-numbers", required=True)

    args = parser.parse_args()

    if args.mode == "classify":
        result = classify(
            args.equation_latex,
            args.context_before,
            args.context_after,
            args.notation_table_path,
            args.vol,
        )
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 0

    if args.mode == "revalidate":
        return revalidate(args.chapter_qmd, args.equation_line_numbers)

    return 2


if __name__ == "__main__":
    sys.exit(main())
