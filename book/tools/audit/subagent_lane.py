#!/usr/bin/env python3
"""Subagent lane helper for Pass 15.

The audit loop's subagent dispatch is unusual: the Python orchestrator
cannot itself spawn parallel subagents — only the Claude conversation
that drives the orchestrator can do that. So this module is split into
two halves:

1. **Plan side** (Python, this file): Reads the ledger, groups
   needs_subagent issues by file, renders the prompt for each file,
   and emits a "dispatch plan" JSON listing every (file, prompt) pair.

2. **Driver side** (Claude conversation, not this file): Reads the
   dispatch plan, spawns N parallel Agent tool calls in a single
   message, collects the JSON results, and feeds them back to the
   apply_subagent_results function below.

3. **Apply side** (Python, this file): Takes the subagent responses,
   builds whole-line edits, applies them with the same five safety
   checks as the script lane, and updates the ledger.

This split keeps the safety logic in Python (where it can be tested
and re-run) while letting the Claude conversation handle the actual
parallel dispatch (where the speedup comes from).

Usage from the driver side:

    # Step 1: build the dispatch plan
    python3 book/tools/audit/subagent_lane.py plan \\
        --ledger audit-ledger.json \\
        --category h3-titlecase \\
        --output dispatch-plan.json

    # Step 2: Claude reads dispatch-plan.json, spawns N subagents
    # in one message, collects their JSON responses, writes them to
    # subagent-results.json with shape:
    #   [{"file": "...", "edits": [...subagent JSON...]}]

    # Step 3: apply with safety checks
    python3 book/tools/audit/subagent_lane.py apply \\
        --ledger audit-ledger.json \\
        --results subagent-results.json \\
        --apply --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit.fix_script_lane import (
    safety_byte_delta,
    safety_no_null_bytes,
    safety_no_sentinels,
    safety_no_new_issues,
    safety_quarto_minimal,
)
from audit.ledger import (
    Issue,
    Ledger,
    STATUS_FAILED_SCRIPT,
    STATUS_FAILED_SUBAGENT,
    STATUS_FIXED_SUBAGENT,
    STATUS_OPEN,
)


PROMPTS_DIR = Path(__file__).resolve().parent / "subagent_prompts"

# Map category -> prompt template filename
PROMPT_TEMPLATES = {
    "h3-titlecase": "sentence_case_h3.md",
}


# ── Plan side ───────────────────────────────────────────────────────────────


@dataclass
class DispatchEntry:
    """One subagent dispatch: a chapter file + the rendered prompt."""
    file: str
    category: str
    issue_count: int
    prompt: str

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "category": self.category,
            "issue_count": self.issue_count,
            "prompt": self.prompt,
        }


def build_dispatch_plan(
    ledger: Ledger, category: str
) -> list[DispatchEntry]:
    """Build the per-file dispatch plan for one category.

    Returns one entry per file that has open needs_subagent issues
    in the given category. The prompt is rendered from the template
    in subagent_prompts/.
    """
    if category not in PROMPT_TEMPLATES:
        raise ValueError(
            f"No subagent prompt template registered for category {category!r}. "
            f"Known: {list(PROMPT_TEMPLATES)}"
        )
    template_path = PROMPTS_DIR / PROMPT_TEMPLATES[category]
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    template = template_path.read_text(encoding="utf-8")

    # Group open subagent-required issues by file
    by_file: dict[str, list[Issue]] = defaultdict(list)
    for issue in ledger.issues:
        if issue.status != STATUS_OPEN:
            continue
        if issue.category != category:
            continue
        if not issue.needs_subagent:
            continue
        by_file[issue.file].append(issue)

    entries: list[DispatchEntry] = []
    for file_str, issues in sorted(by_file.items()):
        prompt = template.replace("{CHAPTER_FILE_PATH}", file_str)
        entries.append(
            DispatchEntry(
                file=file_str,
                category=category,
                issue_count=len(issues),
                prompt=prompt,
            )
        )
    return entries


# ── Apply side ──────────────────────────────────────────────────────────────


@dataclass
class SubagentEdit:
    """One edit returned by a subagent."""
    line: int
    before: str
    after: str
    confidence: str
    reason: str


def parse_subagent_response(
    raw: str,
) -> tuple[list[SubagentEdit] | None, str]:
    """Parse a subagent's JSON response.

    Returns (edits, error). On success, edits is a list and error is "".
    On failure, edits is None and error is a description.
    """
    raw = raw.strip()
    if not raw:
        return [], ""  # empty response = no edits

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"

    if not isinstance(data, list):
        return None, f"expected list, got {type(data).__name__}"

    edits: list[SubagentEdit] = []
    for entry in data:
        if not isinstance(entry, dict):
            return None, f"edit entry not a dict: {entry!r}"
        try:
            edits.append(SubagentEdit(
                line=int(entry["line"]),
                before=str(entry["before"]),
                after=str(entry["after"]),
                confidence=str(entry.get("confidence", "high")),
                reason=str(entry.get("reason", "")),
            ))
        except (KeyError, TypeError, ValueError) as e:
            return None, f"malformed edit entry {entry!r}: {e}"

    return edits, ""


def apply_subagent_edits_to_file(
    file_path: Path,
    edits: list[SubagentEdit],
    dry_run: bool,
    scope: str,
    primary_category: str,
) -> tuple[bool, int, str]:
    """Apply a list of subagent edits to one file with full safety checks.

    Returns (success, applied_count, error_message).
    """
    try:
        old_text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return False, 0, f"read error: {e}"

    lines = old_text.split("\n")
    expected_delta = 0
    applied = 0

    # Heading-text extractor: strip trailing {#anchor-id} and {.unnumbered}
    # markers so we can match against the heading text alone. Subagents
    # frequently drop these attribute markers from their `before` strings,
    # but the markers are protected and never modified by edits, so it's
    # safe to match without them as long as we preserve them on output.
    import re
    _HEADING_ATTR_RE = re.compile(r"\s*\{[^}]*\}\s*$")

    def _heading_text(s: str) -> str:
        return _HEADING_ATTR_RE.sub("", s).rstrip()

    def _heading_attrs(s: str) -> str:
        """Return the trailing {...} attribute string, or empty."""
        m = _HEADING_ATTR_RE.search(s)
        return m.group(0) if m else ""

    for edit in edits:
        idx = edit.line - 1
        if idx < 0 or idx >= len(lines):
            continue
        current = lines[idx]
        # Strict match (handles whitespace drift)
        if current.rstrip() == edit.before.rstrip():
            new_line = edit.after
        else:
            # Looser match: ignore trailing {#...} / {.unnumbered}.
            # The agent may have dropped attribute markers; we restore
            # them on the corrected output.
            current_text = _heading_text(current)
            before_text = _heading_text(edit.before)
            if current_text != before_text:
                # Real drift; skip
                continue
            # Reattach the original attribute markers to the agent's
            # `after` text, since they are protected and not part of
            # the case-correction.
            after_text = _heading_text(edit.after)
            attrs = _heading_attrs(current)
            new_line = after_text + attrs

        # Apply
        lines[idx] = new_line
        before_bytes = len(current.encode("utf-8"))
        after_bytes = len(new_line.encode("utf-8"))
        expected_delta += after_bytes - before_bytes
        applied += 1

    if applied == 0:
        return True, 0, "no edits applied"

    new_text = "\n".join(lines)

    # Five safety checks (same as script lane)
    checks = [
        ("no-null-bytes", lambda: safety_no_null_bytes(new_text)),
        ("no-sentinels", lambda: safety_no_sentinels(new_text)),
        ("byte-delta", lambda: safety_byte_delta(old_text, new_text, expected_delta)),
        ("quarto-minimal", lambda: safety_quarto_minimal(old_text, new_text)),
        ("no-new-issues", lambda: safety_no_new_issues(
            old_text, new_text, file_path, scope, primary_category
        )),
    ]
    for name, check_fn in checks:
        reason = check_fn()
        if reason:
            return False, 0, f"{name}: {reason}"

    if not dry_run:
        file_path.write_text(new_text, encoding="utf-8")

    return True, applied, ""


def apply_subagent_results(
    ledger: Ledger,
    results: list[dict],
    dry_run: bool,
    verbose: bool,
) -> dict:
    """Apply all subagent results with safety checks.

    `results` is a list of dicts with shape:
        [{"file": "...", "edits": [...edit dicts...]}, ...]

    Returns a summary dict with counts and per-file details.
    """
    summary = {
        "files_total": len(results),
        "files_applied": 0,
        "files_failed": 0,
        "edits_applied": 0,
        "edits_failed": 0,
        "failures": [],
    }

    # Build an index of subagent-pending issues so we can mark them
    # fixed/failed in the ledger after applying.
    subagent_issues_by_file: dict[str, list[Issue]] = defaultdict(list)
    for issue in ledger.issues:
        if issue.status != STATUS_OPEN:
            continue
        if not issue.needs_subagent:
            continue
        subagent_issues_by_file[issue.file].append(issue)

    for entry in results:
        file_str = entry["file"]
        raw_edits = entry.get("edits", [])

        # Convert raw dicts to SubagentEdit
        edits: list[SubagentEdit] = []
        for raw in raw_edits:
            try:
                edits.append(SubagentEdit(
                    line=int(raw["line"]),
                    before=str(raw["before"]),
                    after=str(raw["after"]),
                    confidence=str(raw.get("confidence", "high")),
                    reason=str(raw.get("reason", "")),
                ))
            except (KeyError, TypeError, ValueError) as e:
                summary["failures"].append(
                    f"{file_str}: malformed edit entry: {e}"
                )

        if not edits:
            if verbose:
                print(f"  [SKIP] {Path(file_str).name}: no edits", file=sys.stderr)
            continue

        # Find the primary category for this file's pending issues
        file_issues = subagent_issues_by_file.get(file_str, [])
        if not file_issues:
            primary_category = "h3-titlecase"  # default
        else:
            primary_category = file_issues[0].category

        ok, applied, error = apply_subagent_edits_to_file(
            Path(file_str),
            edits,
            dry_run=dry_run,
            scope=ledger.scope,
            primary_category=primary_category,
        )

        if ok:
            summary["files_applied"] += 1
            summary["edits_applied"] += applied
            # Mark applied issues fixed-subagent-lane
            applied_lines = {e.line for e in edits[:applied]}
            for issue in file_issues:
                if issue.line in applied_lines:
                    issue.status = STATUS_FIXED_SUBAGENT
            if verbose:
                print(
                    f"  [OK] {Path(file_str).name}: applied={applied}",
                    file=sys.stderr,
                )
        else:
            summary["files_failed"] += 1
            summary["edits_failed"] += len(edits)
            summary["failures"].append(f"{file_str}: {error}")
            for issue in file_issues:
                issue.status = STATUS_FAILED_SCRIPT
                issue.error = error
            if verbose:
                print(
                    f"  [FAIL] {Path(file_str).name}: {error}",
                    file=sys.stderr,
                )

    return summary


# ── CLI ─────────────────────────────────────────────────────────────────────


def cmd_plan(args: argparse.Namespace) -> int:
    """Build the dispatch plan and write it to a JSON file."""
    ledger = Ledger.load(args.ledger)
    entries = build_dispatch_plan(ledger, args.category)
    plan = {
        "category": args.category,
        "scope": ledger.scope,
        "entries": [e.to_dict() for e in entries],
    }
    args.output.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote dispatch plan: {len(entries)} files, "
        f"{sum(e.issue_count for e in entries)} pending issues",
        file=sys.stderr,
    )
    print(f"  Output: {args.output}", file=sys.stderr)
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    """Apply subagent results to files with safety checks."""
    ledger = Ledger.load(args.ledger)
    results_data = json.loads(args.results.read_text(encoding="utf-8"))

    if isinstance(results_data, dict) and "results" in results_data:
        results = results_data["results"]
    elif isinstance(results_data, list):
        results = results_data
    else:
        print(f"Unexpected results JSON shape", file=sys.stderr)
        return 1

    summary = apply_subagent_results(
        ledger,
        results,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(
        f"SUBAGENT LANE {'DRY RUN' if args.dry_run else 'APPLIED'}",
        file=sys.stderr,
    )
    print(f"{'=' * 60}", file=sys.stderr)
    for k, v in summary.items():
        if k == "failures":
            continue
        print(f"  {k}: {v}", file=sys.stderr)
    if summary["failures"]:
        print(f"\n{len(summary['failures'])} failure(s):", file=sys.stderr)
        for f in summary["failures"][:10]:
            print(f"  - {f}", file=sys.stderr)

    ledger.save(args.ledger)
    print(f"\nLedger updated: {args.ledger}", file=sys.stderr)
    return 0 if not summary["failures"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Pass 15 subagent lane")
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan_p = sub.add_parser("plan", help="Build dispatch plan")
    plan_p.add_argument("--ledger", type=Path, required=True)
    plan_p.add_argument("--category", required=True)
    plan_p.add_argument("--output", type=Path, required=True)
    plan_p.set_defaults(func=cmd_plan)

    apply_p = sub.add_parser("apply", help="Apply subagent results")
    apply_p.add_argument("--ledger", type=Path, required=True)
    apply_p.add_argument("--results", type=Path, required=True)
    mutex = apply_p.add_mutually_exclusive_group(required=True)
    mutex.add_argument("--dry-run", action="store_true")
    mutex.add_argument("--apply", action="store_true")
    apply_p.add_argument("--verbose", "-v", action="store_true")
    apply_p.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
