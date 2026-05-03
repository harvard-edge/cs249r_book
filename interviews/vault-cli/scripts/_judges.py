"""Shared infrastructure for Gemini-judge gates across vault-cli scripts.

Extracted to keep the gate constants and the Gemini-call wrapper in one
place rather than duplicated across:

  - validate_drafts.py    (single-draft gate flow)
  - audit_chains_with_gemini.py (chain audit)
  - audit_math.py         (math spot-check)
  - audit_corpus_batched.py (full-corpus batched audit; CORPUS_HARDENING_PLAN.md Phase 3)

What's exported:

  - GEMINI_MODEL            — pinned model id ("gemini-3.1-pro-preview")
  - COMMON_MISTAKE_MARKERS  — bold-marker tuple for the Pitfall/Rationale/Consequence convention
  - NAPKIN_MATH_MARKERS     — bold-marker tuple for the Assumptions/Calculations/Conclusion convention
  - FAILURE_MODE_TAXONOMY   — prose block enumerating the 4 coherence-failure modes; embed in any prompt that asks Gemini to judge coherence
  - call_gemini_judge()     — subprocess wrapper around the gemini CLI, with strict-JSON parsing and lock-guarded stderr
  - strip_fences()          — small helper for response cleanup
  - gate_format()           — the regex-only format-compliance gate (no LLM call)

Single-question judge functions (gate_level_fit, gate_coherence,
gate_bridge, gate_math) live in their owning scripts because their
prompts are coupled to the script's flow (single vs. batched). The
COMMON shape — marker constants, Gemini call, format regex, failure
taxonomy text — is what's centralized here.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading

GEMINI_MODEL = "gemini-3.1-pro-preview"

# Markup-convention markers required by the format-compliance gate.
# Mirrored in vault-cli/src/vault_cli/commands/authoring.py's COMMON_MISTAKE_TEMPLATE
# and NAPKIN_MATH_TEMPLATE — and tested by tests/test_authoring_scaffold.py
# so a marker rename in one place breaks the test loudly. See
# interviews/vault/AUTHORING.md "Markup conventions" for the rationale.
COMMON_MISTAKE_MARKERS: tuple[str, ...] = (
    "**The Pitfall:**",
    "**The Rationale:**",
    "**The Consequence:**",
)
NAPKIN_MATH_MARKERS: tuple[str, ...] = (
    # Prefix-match: "Assumptions" accepts both "Assumptions:" and
    # "Assumptions & Constraints:". Same for "Conclusion".
    "**Assumptions",
    "**Calculations:**",
    "**Conclusion",
)

# Failure-mode taxonomy used by every coherence-judging prompt. Pasted
# verbatim into the prompt so the judge applies the same rubric whether
# called from validate_drafts (per-draft) or audit_corpus_batched
# (per-batch). Updates here propagate to every judge.
FAILURE_MODE_TAXONOMY = """FAILURE MODES (REJECT verdict=no on any of these — patterns from
the 2026-05-02 audit that previous coherence judges let through):

  1. PHYSICAL ABSURDITY: numbers in the scenario violate real-world
     hardware/software bounds. Examples that should be REJECTED:
       - Mobile/edge NPU wake-up time > ~50ms (real NPUs wake in
         single-digit ms; 0.5s wake-up is fiction)
       - Power figures inconsistent with the device class (e.g., 50W
         for a "smartphone NPU"; 0.05W for a "datacenter accelerator")
       - Latency or throughput figures off by >5× from realistic for
         the named hardware
       - Memory or model-size claims that don't fit the device's
         capacity envelope
       - Duty-cycling patterns that defeat the use-case

  2. VENDOR-NAME FABRICATION: hardware, accelerators, frameworks, or
     benchmarks named in the scenario that don't actually exist or are
     misattributed (e.g., "Coral Edge TPU XL" — there's no XL variant).
     If unsure, treat ambiguous-but-plausible as ok; only flag clearly
     invented names.

  3. SCENARIO/QUESTION/SOLUTION MISMATCH:
       - Question doesn't logically follow from the scenario
       - realistic_solution doesn't actually answer the question (e.g.,
         restates the question, gives generic advice, or answers a
         related-but-different question)
       - Numbers contradict across the three fields

  4. ARITHMETIC ERRORS in napkin_math: the calculations don't add up,
     unit conversions are wrong, or the conclusion doesn't follow from
     the calculations.
"""

# Lock to keep concurrent stderr from interleaving across worker threads.
_print_lock = threading.Lock()


def strip_fences(text: str) -> str:
    """Trim leading/trailing whitespace and strip ```...``` or ```json``` fences.

    The gemini CLI sometimes wraps JSON in fences despite "no fences"
    instruction; this helper makes downstream JSON parsing robust.
    """
    out = text.strip()
    if out.startswith("```"):
        out = out.strip("`")
        if out.startswith("json"):
            out = out[4:].lstrip()
    return out


def call_gemini_judge(prompt: str, *, timeout: int = 240) -> dict | None:
    """Invoke the gemini CLI and parse the strict-JSON response.

    Returns a dict on success, or None on:
      - subprocess timeout
      - non-zero exit with no parseable JSON
      - JSONDecodeError on the extracted brace-delimited substring

    The parser is lenient: it strips fences, then extracts the substring
    between the first '{' and the last '}'. This handles common
    prose-leakage patterns where the model emits "Here is the JSON:"
    before the actual object.
    """
    try:
        result = subprocess.run(
            ["gemini", "-m", GEMINI_MODEL, "-p", prompt, "--yolo"],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    out = strip_fences(result.stdout or "")
    i = out.find("{")
    j = out.rfind("}")
    if i == -1 or j == -1:
        if result.returncode != 0:
            with _print_lock:
                print(f"  gemini exit {result.returncode}: "
                      f"{(result.stderr or '')[:200]}", file=sys.stderr)
        return None
    try:
        return json.loads(out[i:j + 1])
    except json.JSONDecodeError as e:
        with _print_lock:
            print(f"  JSON parse failed: {e}", file=sys.stderr)
        return None


def gate_format(question: dict) -> dict:
    """Free regex-only format-compliance gate.

    Returns a result dict with shape:
      {
        "verdict": "pass" | "fail",
        "issues": [...],
        "common_mistake_present": bool,
        "napkin_math_present": bool,
      }

    Both common_mistake and napkin_math are technically optional in the
    schema; this gate only flags PRESENT-AND-MALFORMED, never absent.
    CORPUS_HARDENING_PLAN.md Phase 6 lifts this into vault check
    --strict's structural tier.
    """
    details = question.get("details") or {}
    issues: list[str] = []

    cm = (details.get("common_mistake") or "").strip()
    if cm:
        missing = [m for m in COMMON_MISTAKE_MARKERS if m not in cm]
        if missing:
            issues.append(f"common_mistake missing {missing!r}")

    nm = (details.get("napkin_math") or "").strip()
    if nm:
        missing = [m for m in NAPKIN_MATH_MARKERS if m not in nm]
        if missing:
            issues.append(f"napkin_math missing {missing!r}")

    return {
        "verdict": "pass" if not issues else "fail",
        "issues": issues,
        "common_mistake_present": bool(cm),
        "napkin_math_present": bool(nm),
    }


__all__ = [
    "GEMINI_MODEL",
    "COMMON_MISTAKE_MARKERS",
    "NAPKIN_MATH_MARKERS",
    "FAILURE_MODE_TAXONOMY",
    "call_gemini_judge",
    "strip_fences",
    "gate_format",
]
