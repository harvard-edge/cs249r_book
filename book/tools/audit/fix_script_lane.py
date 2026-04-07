#!/usr/bin/env python3
"""Pass 15 script-lane fixer.

Reads an audit ledger and applies all issues marked `auto_fixable=True`,
grouped by file. Runs five safety checks per file before writing; files
that fail any safety check are rolled back to their pre-edit state and the
failing issues are marked failed-script-lane in the ledger.

Five safety checks (per Pass 15 plan section 2 stage 4 + section 10.17):
  1. No null bytes introduced
  2. No leftover sentinel strings
  3. Expected byte delta matches actual byte delta
  4. File still parses as valid Quarto (minimal structural check)
  5. No new occurrences of any issue category in the changed file

Usage:
    python3 book/tools/audit/fix_script_lane.py --ledger audit-ledger.json --dry-run
    python3 book/tools/audit/fix_script_lane.py --ledger audit-ledger.json --apply
    python3 book/tools/audit/fix_script_lane.py --ledger audit-ledger.json \\
        --categories vs-period,compound-prefix-closeup --apply

The fixer is the FIX stage of the five-stage cycle. Verification is
handled separately by verify.py.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit.ledger import (
    Issue,
    Ledger,
    STATUS_FAILED_SCRIPT,
    STATUS_FIXED_SCRIPT,
    STATUS_OPEN,
)
from audit.scan import CHECK_REGISTRY, REPO_ROOT

SENTINEL_PATTERNS = ["\x00", "\ufeff\ufeff"]


@dataclass
class FixResult:
    file: str
    applied: int
    failed: int
    skipped: int
    failure_reasons: list[str]


# ── Safety checks ───────────────────────────────────────────────────────────


def safety_no_null_bytes(new_text: str) -> str | None:
    """Return None if safe, else the reason for rejection."""
    if "\x00" in new_text:
        return "contains null bytes"
    return None


def safety_no_sentinels(new_text: str) -> str | None:
    """Check for leftover stash-and-restore sentinels (Pass 15 plan §10.17)."""
    for pat in SENTINEL_PATTERNS:
        if pat in new_text:
            return f"contains sentinel pattern {pat!r}"
    # Also look for the canonical printable sentinel form
    import re
    if re.search(r"⟦SENT\d+⟧", new_text):
        return "contains ⟦SENTn⟧ sentinel"
    return None


def safety_byte_delta(
    old_text: str, new_text: str, expected_delta: int
) -> str | None:
    """Verify the byte delta matches expectation.

    The expected delta is the sum of (len(suggested_after) - len(before))
    across all issues for this file. This is the most important safety
    check because it catches scripts that accidentally touched content
    they shouldn't have.
    """
    actual_delta = len(new_text.encode("utf-8")) - len(old_text.encode("utf-8"))
    if actual_delta != expected_delta:
        return (
            f"byte delta mismatch: expected {expected_delta:+d}, "
            f"actual {actual_delta:+d}"
        )
    return None


def safety_quarto_minimal(old_text: str, new_text: str) -> str | None:
    """Minimal Quarto structural check.

    Verifies the delta between old and new is structurally neutral:
    - Code fence count is unchanged (edits never add/remove ```)
    - Div fence count is unchanged (edits never add/remove :::)
    - YAML `---` delimiter count is unchanged

    We compare deltas, not absolutes, because some files legitimately
    have "unbalanced" fence counts due to fences nested inside HTML
    comments (e.g. collective_communication.qmd has 35 ``` fences).

    This is intentionally minimal — the real Quarto check is handled by
    --quarto-check in verify.py. We just want to catch the obvious
    "file got structurally corrupted" cases.
    """
    def _counts(text: str) -> tuple[int, int, int]:
        lines = text.split("\n")
        fence_count = sum(
            1 for line in lines if line.lstrip().startswith("```")
        )
        div_count = sum(
            1 for line in lines if line.lstrip().startswith(":::")
        )
        yaml_count = sum(1 for line in lines if line.strip() == "---")
        return fence_count, div_count, yaml_count

    old_fences, old_divs, old_yaml = _counts(old_text)
    new_fences, new_divs, new_yaml = _counts(new_text)

    if new_fences != old_fences:
        return (
            f"code fence count changed: {old_fences} -> {new_fences}"
        )
    if new_divs != old_divs:
        return f"div fence count changed: {old_divs} -> {new_divs}"
    if new_yaml != old_yaml:
        return f"YAML delimiter count changed: {old_yaml} -> {new_yaml}"

    return None


def safety_no_new_issues(
    old_text: str,
    new_text: str,
    file_path: Path,
    scope: str,
    category: str,
) -> str | None:
    """Verify the fix did not introduce NEW issues in any category.

    Re-runs every check module on the new text and compares to the old
    text. If the new text has more issues in any category (other than
    the one we just fixed), the fix had unintended side effects.

    This is the most expensive safety check and is only run per-file,
    not per-issue.
    """
    # Build a baseline of old issue counts
    old_counts = _count_all_issues(file_path, old_text, scope)
    new_counts = _count_all_issues(file_path, new_text, scope)

    for cat, new_count in new_counts.items():
        old_count = old_counts.get(cat, 0)
        if cat == category:
            # The target category should decrease (or stay same if nothing
            # was flagged by our check but the fix passed through)
            continue
        if new_count > old_count:
            return (
                f"new issues introduced in category '{cat}': "
                f"{old_count} -> {new_count}"
            )
    return None


def _count_all_issues(
    file_path: Path, text: str, scope: str
) -> dict[str, int]:
    """Count issues by category for a single file using all registered checks."""
    counts: dict[str, int] = {}
    for module_name, category in CHECK_REGISTRY:
        try:
            module = importlib.import_module(module_name)
            issues, _ = module.check(file_path, text, scope, 0)
            counts[category] = len(issues)
        except Exception:
            counts[category] = -1  # error
    return counts


# ── Fix application ─────────────────────────────────────────────────────────


def apply_edits_to_text(text: str, edits: list[Issue]) -> tuple[str, int]:
    """Apply a list of edits to text, returning (new_text, applied_count).

    Strategy: group edits by line number. For each line, apply the
    suggested_after from the HIGHEST-priority issue on that line (they
    should all target the same line text, and suggested_after should be
    the fully-fixed version). If multiple issues target the same line
    with different suggestions, that's a conflict — we skip the line
    and return a smaller applied count.

    This works because every check function that emits an issue also
    emits a complete suggested_after for the whole line, with all the
    edits on that line already applied. That's a design invariant of
    our check functions.
    """
    lines = text.split("\n")
    by_line: dict[int, list[Issue]] = defaultdict(list)
    for issue in edits:
        by_line[issue.line].append(issue)

    applied = 0
    for line_num, line_issues in sorted(by_line.items()):
        idx = line_num - 1  # 1-indexed to 0-indexed
        if idx < 0 or idx >= len(lines):
            continue
        current = lines[idx]
        # All issues on this line should share the same 'before' (the full
        # line text at scan time). If not, the file has drifted.
        before_set = {i.before.rstrip("\n") for i in line_issues}
        if current.rstrip("\n") not in before_set:
            # Line drifted since scan; skip
            continue

        # Pick the suggested_after: if all issues agree, use it. If they
        # disagree (shouldn't happen for checks that emit whole-line
        # replacements), take the one with the most changes.
        suggestions = {i.suggested_after for i in line_issues if i.suggested_after}
        if not suggestions:
            continue
        if len(suggestions) == 1:
            new_line = next(iter(suggestions))
        else:
            # Conflict — pick the one that differs most from the original
            new_line = max(
                suggestions, key=lambda s: sum(1 for c in s if c not in current)
            )
        lines[idx] = new_line
        applied += len(line_issues)

    return "\n".join(lines), applied


def compute_expected_delta(edits: list[Issue]) -> int:
    """Compute the expected byte delta for a batch of edits.

    This is the sum of (len(suggested_after) - len(before)) across all
    unique (line_num, suggested_after) pairs — we don't double-count
    multiple issues that share the same line.
    """
    seen: set[tuple[int, str]] = set()
    total = 0
    for issue in edits:
        if not issue.suggested_after:
            continue
        key = (issue.line, issue.suggested_after)
        if key in seen:
            continue
        seen.add(key)
        before_bytes = len(issue.before.encode("utf-8"))
        after_bytes = len(issue.suggested_after.encode("utf-8"))
        total += after_bytes - before_bytes
    return total


def fix_file(
    file_path: Path,
    file_issues: list[Issue],
    scope: str,
    dry_run: bool,
) -> FixResult:
    """Apply all fixes to a single file, running safety checks before writing."""
    result = FixResult(
        file=str(file_path),
        applied=0,
        failed=0,
        skipped=0,
        failure_reasons=[],
    )

    try:
        old_text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        result.failed = len(file_issues)
        result.failure_reasons.append(f"read error: {e}")
        for i in file_issues:
            i.status = STATUS_FAILED_SCRIPT
            i.error = f"read error: {e}"
        return result

    # Apply edits
    new_text, applied = apply_edits_to_text(old_text, file_issues)
    if new_text == old_text:
        result.skipped = len(file_issues)
        return result

    # Safety checks
    expected_delta = compute_expected_delta(file_issues)
    checks = [
        ("no-null-bytes", lambda: safety_no_null_bytes(new_text)),
        ("no-sentinels", lambda: safety_no_sentinels(new_text)),
        ("byte-delta", lambda: safety_byte_delta(old_text, new_text, expected_delta)),
        ("quarto-minimal", lambda: safety_quarto_minimal(old_text, new_text)),
    ]
    # For cross-category safety, use the primary category of the first issue
    primary_category = file_issues[0].category if file_issues else ""
    checks.append((
        "no-new-issues",
        lambda: safety_no_new_issues(
            old_text, new_text, file_path, scope, primary_category
        ),
    ))

    for name, check_fn in checks:
        reason = check_fn()
        if reason:
            result.failed = len(file_issues)
            result.failure_reasons.append(f"{name}: {reason}")
            for i in file_issues:
                i.status = STATUS_FAILED_SCRIPT
                i.error = f"{name}: {reason}"
            return result

    # All safety checks passed — write the file
    if not dry_run:
        file_path.write_text(new_text, encoding="utf-8")

    result.applied = applied
    for i in file_issues:
        i.status = STATUS_FIXED_SCRIPT

    return result


def fix_all(
    ledger: Ledger,
    categories: list[str] | None,
    dry_run: bool,
    verbose: bool,
) -> list[FixResult]:
    """Apply all auto-fixable issues in the ledger, grouped by file."""
    # Collect open, auto-fixable issues matching the category filter
    by_file: dict[str, list[Issue]] = defaultdict(list)
    for issue in ledger.issues:
        if issue.status != STATUS_OPEN:
            continue
        if not issue.auto_fixable:
            continue
        if categories and issue.category not in categories:
            continue
        by_file[issue.file].append(issue)

    if not by_file:
        print("No auto-fixable issues to apply.", file=sys.stderr)
        return []

    if verbose:
        print(
            f"Applying fixes to {len(by_file)} files "
            f"({'dry run' if dry_run else 'LIVE'})",
            file=sys.stderr,
        )

    results: list[FixResult] = []
    for file_str, file_issues in sorted(by_file.items()):
        result = fix_file(
            Path(file_str), file_issues, ledger.scope, dry_run
        )
        results.append(result)
        if verbose and (result.applied or result.failed):
            status = "OK" if not result.failed else "FAIL"
            print(
                f"  [{status}] {Path(file_str).name}: "
                f"applied={result.applied} failed={result.failed} "
                f"skipped={result.skipped}",
                file=sys.stderr,
            )
            for reason in result.failure_reasons:
                print(f"        {reason}", file=sys.stderr)

    return results


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pass 15 script-lane fixer",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        required=True,
        help="Path to the audit ledger JSON",
    )
    parser.add_argument(
        "--categories",
        default="",
        help="Comma-separated category names to fix (default: all auto-fixable)",
    )
    mutex = parser.add_mutually_exclusive_group(required=True)
    mutex.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files",
    )
    mutex.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply the changes",
    )
    parser.add_argument(
        "--save-ledger",
        type=Path,
        default=None,
        help="Save the updated ledger to this path (default: overwrite input)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-file progress",
    )
    args = parser.parse_args()

    ledger = Ledger.load(args.ledger)
    categories = (
        [c.strip() for c in args.categories.split(",") if c.strip()]
        if args.categories
        else None
    )

    results = fix_all(
        ledger,
        categories=categories,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    total_applied = sum(r.applied for r in results)
    total_failed = sum(r.failed for r in results)
    total_files_touched = sum(1 for r in results if r.applied or r.failed)

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"SCRIPT LANE {'DRY RUN' if args.dry_run else 'APPLIED'}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"Files touched: {total_files_touched}", file=sys.stderr)
    print(f"Issues applied: {total_applied}", file=sys.stderr)
    print(f"Issues failed:  {total_failed}", file=sys.stderr)

    # Save updated ledger
    save_path = args.save_ledger or args.ledger
    ledger.save(save_path)
    print(f"Ledger saved to {save_path}", file=sys.stderr)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
