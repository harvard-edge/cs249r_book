#!/usr/bin/env python3
"""Bibliography verification pipeline using Gemini CLI.

Walks every references.bib file in the repo, batches the entries, and
asks Gemini (with Google Search) to verify that each entry's author /
title / year / venue / DOI is internally consistent and externally
real. Catches the silent-corruption class that no schema check can:
when the entry's content describes a different paper than the key
suggests, or when the DOI / venue / year is plausible-looking but
wrong.

Sister of `book/tools/scripts/gemini_review.py` (chapter HTML review).
Same pattern: subprocess-invoke `gemini -m MODEL -p PROMPT`, run
batches in a ThreadPoolExecutor, retry+backoff, write results to a
timestamped directory under `book/tools/audit/out/`.

Usage:

    # Smoke test on first 5 entries of vol1
    python3 book/tools/scripts/bib_verify_gemini.py --bib vol1 --limit 5

    # Full sweep across all bib files in the repo
    python3 book/tools/scripts/bib_verify_gemini.py --all

    # Resume from a specific entry (after interruption)
    python3 book/tools/scripts/bib_verify_gemini.py --all --start-from gentry2009

    # Specific bib files only
    python3 book/tools/scripts/bib_verify_gemini.py \\
        --bib book/quarto/contents/vol2/backmatter/references.bib \\
        --batch-size 30 --max-parallel 6

    # Re-verify entries even if previously stamped (after model upgrade, etc.)
    python3 book/tools/scripts/bib_verify_gemini.py --all --reverify

Stamp lifecycle: every entry that comes back as 'verified' or
'uncertain' is stamped in-place into the .bib file with three fields:
  x-verified         = {YYYY-MM-DD}
  x-verified-by      = {gemini-MODEL-NAME}
  x-verified-status  = {verified}   (or {uncertain})
  x-verified-source  = {url1; url2}  (only when sources are returned)
On subsequent runs the script SKIPS any entry already carrying a
stamp from this script (matching x-verified-by ~= 'gemini-*') unless
--reverify is passed. Entries flagged 'broken' are NOT stamped, so
they get re-checked every run until fixed.

Output: book/tools/audit/out/bib_verify_<timestamp>/
  - summary.json     - aggregated per-entry verdicts
  - summary.md       - human-readable report grouped by status
  - batch_NNNN.json  - raw Gemini response per batch (for debugging)
  - errors.log       - any batches that failed all retries

Cost / runtime: each batch is one Gemini call. Default batch_size=25
puts 1,650 entries through ~66 calls. With --max-parallel 4 and a
~30 s round-trip per call, full sweep is ~8 minutes wall clock.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Defaults ────────────────────────────────────────────────────────────────
MODEL = "gemini-3.1-pro-preview"
BATCH_SIZE = 50
MAX_PARALLEL = 4
STAGGER_SECONDS = 25  # gap between batch launches (pipeline stagger to dodge burst rate-limits)
GEMINI_TIMEOUT = 900          # per-batch timeout in seconds (Gemini + web search is slow)
MAX_RETRIES = 1               # if a call times out at 15 min, retrying rarely helps

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_BASE = REPO_ROOT / "book" / "tools" / "audit" / "out"

# Files that look like references.bib but should be skipped (build artifacts,
# dependency caches, etc.). These globs apply to the path string, not the
# basename, so they catch nested copies too.
EXCLUDE_PATTERNS = [
    "_build", "node_modules", ".git", "__pycache__",
    ".venv", "venv", "dist", "build/intermediate",
]


# ── Prompt ──────────────────────────────────────────────────────────────────
# bib-check.md is the single source of truth for what an entry must satisfy.
# We inline the whole file into the prompt at runtime so the prompt and the
# rule can never drift. If the rule file is updated, every subsequent run
# uses the new rules without code changes.
RULES_PATH = REPO_ROOT / ".claude" / "rules" / "bib-check.md"

PROMPT_PREAMBLE = """\
You are verifying bibliographic entries for an academic textbook
(MIT Press, ML Systems by Vijay Janapa Reddi). For each entry below,
check whether the citation information is correct, internally
consistent, matches a real published work, AND complies with the
project bibliography style rules quoted below.

PRIMARY GOAL: catch silent corruption — entries whose body content
describes a different paper than the key suggests, plausible-but-wrong
DOIs, drifted years, missing required fields, etc. The bib-check.md
rules below are the canonical schema; verify against them.

When uncertain, USE GOOGLE SEARCH to look up the paper — search by
title + first author surname, then check whether the venue / year /
DOI from the search results match what the entry claims.

CRITICAL: do NOT silently rename a bib key in your `suggested_fix`.
Every `[@key]` reference in the .qmd source files would break. If you
detect that the canonical key would be different from what the entry
uses (e.g., body says "Smith 2020" but key is "jones2018"), put the
new key in `suggested_fix.rename_key_to` AND set `issue` to call out
that the rename requires propagating every `[@oldkey]` and `@oldkey`
reference in `book/quarto/contents/**/*.qmd` to the new name. The
human reviewer must do this together; partial application breaks
links.

──────────── BIBLIOGRAPHY RULES (.claude/rules/bib-check.md) ────────
{rules}
──────────────────────── END BIBLIOGRAPHY RULES ─────────────────────"""


def build_system_prompt() -> str:
    """Inline bib-check.md into the prompt + append output-format spec.

    bib-check.md is the canonical rule file (single source of truth).
    The script does not duplicate its content — only the I/O contract
    Gemini must follow lives here.
    """
    rules_text = RULES_PATH.read_text(encoding="utf-8") if RULES_PATH.exists() else "(rules file not found)"
    return PROMPT_PREAMBLE.format(rules=rules_text) + """

══════════════════════ OUTPUT FORMAT ══════════════════════
Return ONLY a single JSON object (no markdown fences, no prose around it):

{
  "verdicts": [
    {
      "key": "entry_key_here",
      "status": "verified" | "broken" | "uncertain",
      "confidence": "high" | "medium" | "low",
      "issue": "Brief description of the problem (only if broken/uncertain)",
      "suggested_fix": {
        "title": "Correct title",
        "author": "Correct author list (BibTeX format)",
        "year": "YYYY",
        "venue": "Correct journal/booktitle",
        "publisher": "Correct publisher (no city)",
        "pages": "N--M",
        "doi": "10.xxxx/yyyy",
        "rename_key_to": "new_canonical_key (only if the key itself is wrong; see CRITICAL note above)"
      },
      "sources": ["url1", "url2"]
    }
  ]
}

Verdict rules:
- "verified" = author + title + year + venue match a real paper, key
  prefix matches author/year, and the entry satisfies the bib-check.md
  schema (publisher present where required, no city in publisher,
  full author names, etc.). Include a source URL even when verified.
- "broken" = at least one field is demonstrably wrong, OR a required
  bib-check.md field is missing. Include the specific fix.
- "uncertain" = the entry might be fine but you cannot find
  authoritative confirmation in 2-3 search queries. Include what you
  tried in "issue".

Be conservative: only mark "broken" when you have a specific source
URL contradicting the entry, OR a bib-check.md schema rule that's
clearly violated. Otherwise mark "uncertain" and let a human decide.

Entries to verify follow.
"""


# ── BibTeX parsing (reuse bib_lint.py) ─────────────────────────────────────
def _load_bib_lint():
    """Dynamically import book/tools/bib_lint.py without polluting CWD/sys.path.

    Register in sys.modules BEFORE exec_module so @dataclass-decorated
    Entry can resolve its own module dict (Python 3.13+ requirement).
    """
    name = "_bib_lint_helper"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "book" / "tools" / "bib_lint.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── File discovery ─────────────────────────────────────────────────────────
def discover_bib_files(filters: list[str] | None) -> list[Path]:
    """Return all references.bib files in the repo, optionally filtered.

    `filters` is a list of either:
      - aliases like "vol1", "vol2" → mapped to canonical paths
      - bare basenames matched against discovered paths
      - explicit relative or absolute paths
    """
    aliases = {
        "vol1": REPO_ROOT / "book/quarto/contents/vol1/backmatter/references.bib",
        "vol2": REPO_ROOT / "book/quarto/contents/vol2/backmatter/references.bib",
    }
    if filters:
        out: list[Path] = []
        for f in filters:
            if f in aliases:
                out.append(aliases[f])
            else:
                p = Path(f)
                if not p.is_absolute():
                    p = REPO_ROOT / p
                if not p.exists():
                    print(f"warn: {f}: not found", file=sys.stderr)
                    continue
                out.append(p)
        return out

    # Discover everything matching references.bib under repo, minus excludes
    found = []
    for p in REPO_ROOT.rglob("references.bib"):
        if any(part in EXCLUDE_PATTERNS for part in p.parts):
            continue
        found.append(p)
    return sorted(found)


def short_path(p: Path) -> str:
    """Display path relative to REPO_ROOT when possible."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


# ── Batching ───────────────────────────────────────────────────────────────
def chunk(seq: list, size: int) -> list[list]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def serialize_entry_for_prompt(entry) -> str:
    """Render a parsed Entry as the BibTeX-ish summary the prompt expects.

    We include only the fields a verifier needs (author, title, year,
    journal/booktitle, doi, url). Skips x-verified-* and other internal
    metadata that would just bloat the prompt.
    """
    interesting = ["author", "title", "year", "journal", "booktitle",
                   "publisher", "volume", "number", "pages", "doi", "url"]
    lines = [f"@{entry.entry_type}{{{entry.key},"]
    for field_name in interesting:
        field = entry.get(field_name)
        if field is not None and field.value:
            v = re.sub(r"\s+", " ", field.value).strip()
            if len(v) > 300:
                v = v[:300] + "…"
            lines.append(f"  {field_name} = {{{v}}},")
    lines.append("}")
    return "\n".join(lines)


# ── Gemini invocation ──────────────────────────────────────────────────────
def call_gemini(prompt_input: str, system_prompt: str, model: str) -> tuple[bool, str, str]:
    """Invoke `gemini -m MODEL -p system_prompt` with stdin = prompt_input.

    Returns (ok, stdout, stderr). Filters out the gemini-CLI internal
    error noise that mixes into stdout (matches gemini_review.py).
    """
    for attempt in range(MAX_RETRIES):
        try:
            result = subprocess.run(
                ["gemini", "-m", model, f"--prompt={system_prompt}", "-o", "text"],
                input=prompt_input,
                capture_output=True,
                text=True,
                timeout=GEMINI_TIMEOUT,
            )
            stdout = result.stdout
            # Strip gemini-CLI internals from stdout (per gemini_review.py)
            stdout_lines = [
                line for line in stdout.split("\n")
                if not line.startswith("ERROR: Failed to fetch")
                and not line.strip().startswith("at ")
                and not line.strip().startswith("at async")
                and "Gaxios" not in line
                and "googleapis" not in line
            ]
            cleaned = "\n".join(stdout_lines).strip()
            if cleaned and len(cleaned) > 20:
                return True, cleaned, result.stderr
            # empty/short → retry
            if attempt < MAX_RETRIES - 1:
                time.sleep(10 * (attempt + 1))
                continue
            return False, cleaned, "empty response after retries"
        except subprocess.TimeoutExpired:
            if attempt < MAX_RETRIES - 1:
                time.sleep(10)
                continue
            return False, "", f"timeout after {GEMINI_TIMEOUT}s"
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(10)
                continue
            return False, "", str(e)
    return False, "", "fell through retry loop"


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object from Gemini's response.

    Gemini sometimes wraps the JSON in ```json fences or adds a sentence
    before/after despite the "ONLY JSON" instruction. Be lenient.
    """
    # Try fenced first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try the first { ... last } span
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        try:
            return json.loads(text[first : last + 1])
        except json.JSONDecodeError:
            return None
    return None


# ── Stamps (skip-already-verified + write-back) ────────────────────────────
GEMINI_STAMP_PREFIX = "gemini-"  # matches x-verified-by values this script writes


def is_gemini_stamped(entry) -> bool:
    """True if the entry already carries a stamp from this script."""
    field = entry.get("x-verified-by")
    if field is None:
        return False
    return field.value.strip().startswith(GEMINI_STAMP_PREFIX)


def stamp_entry_in_text(text: str, key: str, model: str, status: str,
                        sources: list[str]) -> tuple[str, bool]:
    """Insert / replace x-verified-* fields for `key` in the bib text.

    Returns (new_text, changed). Locates the entry by key, walks balanced
    braces to find the closing `}`, then either replaces the existing
    x-verified-* block or appends new fields just before the closing brace.
    """
    # Find the entry header
    pat = re.compile(rf"@\w+\s*\{{\s*{re.escape(key)}\s*,", re.M)
    hm = pat.search(text)
    if not hm:
        return text, False
    open_brace = text.find("{", hm.start())
    if open_brace < 0:
        return text, False
    depth = 0
    i = open_brace
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            i += 2; continue
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: break
        i += 1
    if depth != 0:
        return text, False
    body_start, body_end = open_brace + 1, i  # text[body_end] == '}'
    body = text[body_start:body_end]

    today = datetime.now().strftime("%Y-%m-%d")
    src_value = "; ".join(sources) if sources else ""
    # Use the model name as-is; it already starts with "gemini-" so
    # is_gemini_stamped's prefix check matches without double-prefixing.
    verified_by = model if model.startswith(GEMINI_STAMP_PREFIX) else GEMINI_STAMP_PREFIX + model
    new_lines = [
        f"  x-verified = {{{today}}},",
        f"  x-verified-by = {{{verified_by}}},",
        f"  x-verified-status = {{{status}}},",
    ]
    if src_value:
        new_lines.append(f"  x-verified-source = {{{src_value}}},")

    # Strip any existing x-verified-* lines from the body
    cleaned = re.sub(
        r"^\s*x-verified(?:-[a-z]+)?\s*=\s*\{[^}]*\},?\s*\n",
        "", body, flags=re.M,
    )
    # Ensure trailing newline before append
    if not cleaned.endswith("\n"):
        cleaned += "\n"
    new_body = cleaned + "\n".join(new_lines) + "\n"
    new_text = text[:body_start] + new_body + text[body_end:]
    return new_text, new_text != text


def apply_stamps_to_file(bib_path: Path, model: str,
                         per_key_verdict: dict[str, dict]) -> int:
    """For one bib file, write stamps for every (verified|uncertain) verdict."""
    text = bib_path.read_text(encoding="utf-8")
    changed = 0
    for key, v in per_key_verdict.items():
        status = v.get("status", "")
        if status not in ("verified", "uncertain"):
            continue
        sources = [s for s in (v.get("sources") or []) if isinstance(s, str)]
        text, ok = stamp_entry_in_text(text, key, model, status, sources)
        if ok:
            changed += 1
    if changed:
        bib_path.write_text(text, encoding="utf-8")
    return changed


# ── Per-batch worker ───────────────────────────────────────────────────────
def verify_batch(
    bib_path: Path,
    entries: list,
    batch_index: int,
    total_batches: int,
    out_dir: Path,
    model: str,
    system_prompt: str,
) -> dict:
    """Verify one batch of entries via Gemini, write raw + parsed JSON."""
    label = f"{short_path(bib_path)} batch {batch_index + 1}/{total_batches} ({len(entries)} entries)"
    print(f"  [start]  {label}", flush=True)
    body = "\n\n".join(serialize_entry_for_prompt(e) for e in entries)
    t0 = time.time()
    ok, stdout, stderr = call_gemini(body, system_prompt, model)
    elapsed = time.time() - t0

    raw_path = out_dir / f"batch_{batch_index:04d}_raw.txt"
    raw_path.write_text(stdout if ok else f"FAILED\n{stderr}\n", encoding="utf-8")

    if not ok:
        print(f"  [FAIL]   {label}  ({elapsed:.0f}s)  {stderr[:80]}", flush=True)
        return {
            "bib": short_path(bib_path),
            "batch_index": batch_index,
            "status": "failed",
            "error": stderr,
            "entry_keys": [e.key for e in entries],
        }

    parsed = extract_json(stdout)
    if not parsed or "verdicts" not in parsed:
        print(f"  [PARSE]  {label}  ({elapsed:.0f}s)  unparsable JSON", flush=True)
        return {
            "bib": short_path(bib_path),
            "batch_index": batch_index,
            "status": "unparsable",
            "raw_excerpt": stdout[:500],
            "entry_keys": [e.key for e in entries],
        }

    parsed_path = out_dir / f"batch_{batch_index:04d}.json"
    parsed_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")

    counts = {"verified": 0, "broken": 0, "uncertain": 0, "other": 0}
    for v in parsed.get("verdicts", []):
        counts[v.get("status", "other")] = counts.get(v.get("status", "other"), 0) + 1
    summary_str = f"v={counts['verified']} b={counts['broken']} u={counts['uncertain']}"
    print(f"  [done]   {label}  ({elapsed:.0f}s)  {summary_str}", flush=True)

    return {
        "bib": short_path(bib_path),
        "batch_index": batch_index,
        "status": "ok",
        "elapsed_sec": round(elapsed, 1),
        "verdicts": parsed["verdicts"],
        "entry_keys": [e.key for e in entries],
    }


# ── Aggregation + report ───────────────────────────────────────────────────
def write_summary(out_dir: Path, all_results: list[dict]) -> dict:
    """Aggregate per-batch results into summary.json + summary.md."""
    total_entries = sum(len(r.get("entry_keys", [])) for r in all_results)
    by_status = {"verified": [], "broken": [], "uncertain": [], "other": []}
    failed_batches = []
    for r in all_results:
        if r["status"] != "ok":
            failed_batches.append(r)
            continue
        for v in r.get("verdicts", []):
            v_with_bib = {**v, "bib": r["bib"]}
            status = v.get("status", "other")
            by_status.setdefault(status, []).append(v_with_bib)

    summary = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "total_entries": total_entries,
        "totals": {k: len(v) for k, v in by_status.items()},
        "failed_batches": len(failed_batches),
        "verdicts": by_status,
        "failed_batch_details": failed_batches,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Human-readable markdown
    lines = [
        f"# Bib verification report",
        f"",
        f"- **Checked**: {summary['checked_at']}",
        f"- **Total entries**: {total_entries}",
        f"- **Verified**: {summary['totals'].get('verified', 0)}",
        f"- **Broken**: {summary['totals'].get('broken', 0)}",
        f"- **Uncertain**: {summary['totals'].get('uncertain', 0)}",
        f"- **Failed batches**: {len(failed_batches)}",
        f"",
    ]
    if by_status["broken"]:
        lines.append("## Broken (need fixing)")
        lines.append("")
        for v in by_status["broken"]:
            lines.append(f"### `{v['key']}`  ({v.get('bib','')})")
            lines.append(f"")
            lines.append(f"- **Issue**: {v.get('issue', '(no detail)')}")
            if v.get("suggested_fix"):
                lines.append(f"- **Suggested fix**:")
                for k, val in v["suggested_fix"].items():
                    lines.append(f"  - `{k}`: {val}")
            if v.get("sources"):
                lines.append(f"- **Sources**:")
                for s in v["sources"]:
                    lines.append(f"  - {s}")
            lines.append("")
    if by_status["uncertain"]:
        lines.append("## Uncertain (human review needed)")
        lines.append("")
        for v in by_status["uncertain"]:
            lines.append(f"- `{v['key']}` ({v.get('bib','')}): {v.get('issue','')}")
        lines.append("")
    if failed_batches:
        lines.append("## Failed batches")
        lines.append("")
        for fb in failed_batches:
            lines.append(f"- {fb['bib']} batch {fb['batch_index']}: {fb.get('status')} — {len(fb.get('entry_keys', []))} entries skipped")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", action="store_true",
                        help="Verify every references.bib file in the repo")
    parser.add_argument("--bib", action="append", default=[],
                        help="Specific bib file or alias (vol1, vol2). Repeatable.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Verify only the first N entries per file (smoke test)")
    parser.add_argument("--start-from", type=str, default="",
                        help="Skip every entry up to and including this key (resume)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-parallel", type=int, default=MAX_PARALLEL)
    parser.add_argument("--stagger", type=int, default=STAGGER_SECONDS, help="Seconds between successive batch launches (pipeline stagger, dodges burst rate-limits)")
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--reverify", action="store_true",
                        help="Re-verify entries even if already stamped by a previous run")
    parser.add_argument("--no-stamp", action="store_true",
                        help="Do not write x-verified-* fields back into bib files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the prompts that would be sent, do not call Gemini")
    args = parser.parse_args()

    if not args.all and not args.bib:
        parser.error("Specify --all or one or more --bib FILE/ALIAS")

    bib_lint = _load_bib_lint()
    bibs = discover_bib_files(args.bib if not args.all else None)
    if not bibs:
        print("No references.bib files found.", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = OUT_BASE / f"bib_verify_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {short_path(out_dir)}")
    print(f"Model:  {args.model}")
    print(f"Batch:  size={args.batch_size}, parallel={args.max_parallel}")
    print(f"Files:  {len(bibs)}")
    print()

    # Build the full task list (one task = one batch from one file)
    tasks: list[tuple[Path, list, int, int]] = []
    skipping = bool(args.start_from)
    for bib_path in bibs:
        text = bib_path.read_text(encoding="utf-8")
        try:
            entries, _ = bib_lint.parse_bib(text)
        except Exception as e:
            print(f"  parse error: {short_path(bib_path)}: {e}", file=sys.stderr)
            continue
        if skipping:
            kept = []
            for e in entries:
                if not skipping:
                    kept.append(e)
                elif e.key == args.start_from:
                    skipping = False  # include subsequent entries
            entries = kept
        # Skip entries already stamped by a previous run (unless --reverify)
        before = len(entries)
        if not args.reverify:
            entries = [e for e in entries if not is_gemini_stamped(e)]
        skipped_already = before - len(entries)
        if args.limit:
            entries = entries[: args.limit]
        if not entries:
            note = f" (all {before} already verified — pass --reverify to re-check)" if skipped_already and skipped_already == before else ""
            print(f"  {short_path(bib_path):<55}  0 entries to verify{note}")
            continue
        batches = chunk(entries, args.batch_size)
        n = len(batches)
        for i, b in enumerate(batches):
            tasks.append((bib_path, b, i, n))
        suffix = f" ({skipped_already} already-verified skipped)" if skipped_already else ""
        print(f"  {short_path(bib_path):<55}  {len(entries):>4} entries → {n} batches{suffix}")

    if not tasks:
        print("Nothing to verify.")
        return 0

    print(f"\nTotal: {len(tasks)} batches, {sum(len(t[1]) for t in tasks)} entries")
    print()

    if args.dry_run:
        for path, batch, i, n in tasks[:3]:
            print(f"\n--- {short_path(path)} batch {i+1}/{n} ---")
            print("\n\n".join(serialize_entry_for_prompt(e) for e in batch[:2]))
            print("...(more entries)...")
        print(f"\n[dry-run] would send {len(tasks)} batches to Gemini")
        return 0

    # Build the system prompt once (inlines bib-check.md at runtime)
    system_prompt = build_system_prompt()
    rules_size = len(system_prompt)
    print(f"Prompt: {rules_size} chars (bib-check.md inlined from {short_path(RULES_PATH)})")
    print()

    # Pipeline stagger: launch one batch every STAGGER_SECONDS regardless of
    # max_parallel. This avoids the all-fire-at-once burst that triggers
    # Gemini's per-minute rate limit. With max_parallel workers and a stagger
    # ≈ (per-batch latency / max_parallel), the pool stays full but launches
    # never overlap in a thundering herd. Default 25 s × 4 workers ≈ steady
    # state of one in-flight batch per 6 s on the launch side.
    all_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futures = []
        for idx, (path, batch, i, n) in enumerate(tasks):
            if idx > 0:
                time.sleep(args.stagger)
            futures.append(ex.submit(
                verify_batch, path, batch, i, n, out_dir, args.model, system_prompt
            ))
        for fut in as_completed(futures):
            try:
                all_results.append(fut.result())
            except Exception as e:
                print(f"  worker raised: {e}", file=sys.stderr)

    # Write x-verified-* stamps back into the bib files for verified+uncertain
    if not args.no_stamp:
        per_file_verdicts: dict[Path, dict[str, dict]] = {}
        for r in all_results:
            if r.get("status") != "ok":
                continue
            bib_rel = r["bib"]
            bib_abs = REPO_ROOT / bib_rel
            d = per_file_verdicts.setdefault(bib_abs, {})
            for v in r.get("verdicts", []):
                if v.get("key"):
                    d[v["key"]] = v
        total_stamped = 0
        for bib_abs, verdict_map in per_file_verdicts.items():
            n = apply_stamps_to_file(bib_abs, args.model, verdict_map)
            total_stamped += n
            if n:
                print(f"  stamped: {n:>4} entries in {short_path(bib_abs)}")
        if total_stamped:
            print(f"Total entries stamped: {total_stamped}")

    summary = write_summary(out_dir, all_results)
    print()
    print(f"Done. Summary written to {short_path(out_dir / 'summary.md')}")
    print(f"  Verified:   {summary['totals'].get('verified', 0)}")
    print(f"  Broken:     {summary['totals'].get('broken', 0)}")
    print(f"  Uncertain:  {summary['totals'].get('uncertain', 0)}")
    print(f"  Failed:     {summary['failed_batches']} batches")
    return 0 if summary["totals"].get("broken", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
