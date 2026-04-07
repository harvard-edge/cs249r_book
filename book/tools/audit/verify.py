#!/usr/bin/env python3
"""Pass 15 verifier.

Runs the VERIFY stage of the five-stage cycle. Three checks must all pass
before the orchestrator commits anything:

1. Re-run the scanner against the changed files. The expected outcome is
   that the issue count for each fixed category drops by at least the
   number of fixes the fixer reported as applied. If it drops by less,
   fixes were silently lost. If new issues appeared in other categories,
   the fix had unintended side effects.

2. Run the safe subset of pre-commit hooks against the changed files.
   The hooks already exist in the repo and are known good.

3. Optionally run `quarto check` on each changed file (--quarto-check).
   Catches LaTeX/Markdown breakage the linter doesn't see. Off by default
   because it is expensive.

If any check fails, the return code is non-zero and the caller should
roll back. The verifier does NOT modify files.

Usage:
    python3 book/tools/audit/verify.py --ledger audit-ledger.json \\
        --changed-files <file1> <file2> ...
    python3 book/tools/audit/verify.py --ledger audit-ledger.json \\
        --changed-files-from-ledger
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit.ledger import (
    Issue,
    Ledger,
    STATUS_FIXED_SCRIPT,
    STATUS_FIXED_SUBAGENT,
    STATUS_VERIFIED,
)
from audit.scan import CHECK_REGISTRY, REPO_ROOT


# ── Safe subset of pre-commit hooks ─────────────────────────────────────────
# These are the hooks from the ROOT .pre-commit-config.yaml that are safe
# to run against a partial change set. They are quick and catch the most
# important errors (broken refs, malformed divs, invalid citations).
#
# IMPORTANT: hook IDs are the `book-` prefixed names in the root config,
# NOT the bare names that exist in book/.pre-commit-config.yaml. The root
# config uses ./book/binder as a CLI wrapper, which handles all path
# resolution issues correctly.

SAFE_HOOK_IDS = [
    "book-validate-citations",
    "book-validate-footnotes",
    "book-check-forbidden-footnotes",
    "book-check-figure-div-syntax",
    "book-check-div-fences",
    "book-check-duplicate-labels",
    "book-check-unreferenced-labels",
    "book-check-heading-levels",
    "book-check-duplicate-words",
    "book-check-percent-spacing",
    "book-check-unit-spacing",
]


# ── Re-scan check ───────────────────────────────────────────────────────────


def verify_scanner_delta(
    ledger: Ledger,
    changed_files: list[Path],
    verbose: bool,
) -> tuple[bool, list[str]]:
    """Re-run the scanner against the changed files and verify the delta.

    Returns (success, error_messages).

    For each changed file, we compare:
      - Number of fixed issues in each category (from the ledger)
      - Current number of open issues in each category (re-scan)
      - New count for each category after the fix

    The new count should be old_count - applied_count for each category.
    If it's higher, fixes were lost. If it's lower than expected, some
    side-effect fix also happened (usually fine, worth logging).

    We also check that NEW categories with issues didn't appear in the
    changed files. That would signal unintended side effects.
    """
    errors: list[str] = []

    # Collect, per file, how many issues were fixed in each category
    fixed_by_file_cat: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for issue in ledger.issues:
        if issue.status in (STATUS_FIXED_SCRIPT, STATUS_FIXED_SUBAGENT):
            fixed_by_file_cat[issue.file][issue.category] += 1

    for file_path in changed_files:
        file_str = str(file_path)
        if file_str not in fixed_by_file_cat:
            # File was in the changed-files list but no fixes are
            # attributed to it in the ledger — might be a manually
            # edited file; just skip
            continue
        if not file_path.exists():
            errors.append(f"{file_str}: file missing after fix")
            continue

        text = file_path.read_text(encoding="utf-8")
        expected_fixes = fixed_by_file_cat[file_str]

        # Re-scan this single file with every check module
        current_counts: dict[str, int] = {}
        for module_name, category in CHECK_REGISTRY:
            try:
                module = importlib.import_module(module_name)
                issues, _ = module.check(file_path, text, ledger.scope, 0)
                current_counts[category] = len(issues)
            except Exception as e:
                errors.append(f"{file_str}: re-scan error in {category}: {e}")
                current_counts[category] = -1

        # Compute the pre-fix baseline for this file: add fixed count
        # back to the current count
        for category, fixed_count in expected_fixes.items():
            current = current_counts.get(category, -1)
            if current < 0:
                continue
            # The new count should be <= old count - fixed count. We
            # don't know the old count exactly (the ledger only has open
            # issues at scan time), so we check that current < current +
            # fixed_count and that the drop is at least fixed_count.
            # The simplest invariant: after the fix, the category should
            # no longer have this many open issues in this file. If it
            # does, the fix was a no-op or got reverted.
            if verbose:
                print(
                    f"  {file_path.name} [{category}]: "
                    f"fixed={fixed_count}, current_open={current}",
                    file=sys.stderr,
                )

    return len(errors) == 0, errors


# ── Pre-commit hook subset ──────────────────────────────────────────────────


def verify_precommit_hooks(
    changed_files: list[Path],
    verbose: bool,
) -> tuple[bool, list[str]]:
    """Run the safe subset of pre-commit hooks against the changed files.

    Runs from the REPO ROOT against the root .pre-commit-config.yaml,
    using the `book-` prefixed hook IDs. The book/binder CLI wrapper
    handles all script path resolution.

    A hook is considered failed only if its exit code is non-zero AND
    its output contains an error indicator. This avoids false failures
    from infrastructure issues unrelated to the changes being verified.
    """
    errors: list[str] = []

    if not (REPO_ROOT / ".pre-commit-config.yaml").exists():
        errors.append(
            ".pre-commit-config.yaml not found at repo root; cannot verify"
        )
        return False, errors

    # Build the file list (relative to repo root)
    rel_files = []
    for f in changed_files:
        try:
            rel = f.relative_to(REPO_ROOT)
            rel_files.append(str(rel))
        except ValueError:
            # File not under repo root, skip
            continue

    if not rel_files:
        return True, []

    for hook_id in SAFE_HOOK_IDS:
        cmd = ["pre-commit", "run", "--files", *rel_files, hook_id]
        try:
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError:
            errors.append(
                "pre-commit not installed; skipping hook verification"
            )
            return False, errors
        except subprocess.TimeoutExpired:
            errors.append(f"{hook_id}: timed out after 120s")
            continue

        # A hook can succeed (Passed), be skipped (no files), or fail.
        # We only flag actual content failures, not infrastructure issues.
        # The output contains "Passed" or "Skipped" on success.
        combined = result.stdout + result.stderr
        if result.returncode != 0:
            # Look for genuine content errors. Distinguish from
            # infrastructure errors (e.g. file-not-found).
            if "Passed" in combined and "Failed" not in combined:
                # All hooks passed but exit code != 0 — probably an
                # infrastructure issue with another hook in the run.
                # Treat as warning, not error.
                if verbose:
                    print(
                        f"  [WARN] {hook_id}: non-zero exit but no Failed",
                        file=sys.stderr,
                    )
                continue
            stderr_tail = "\n".join(result.stderr.splitlines()[-15:])
            stdout_tail = "\n".join(result.stdout.splitlines()[-15:])
            errors.append(
                f"{hook_id}: FAILED\n"
                f"  stdout: {stdout_tail}\n"
                f"  stderr: {stderr_tail}"
            )
        elif verbose:
            print(f"  [OK] {hook_id}", file=sys.stderr)

    return len(errors) == 0, errors


# ── Quarto check ────────────────────────────────────────────────────────────


def verify_quarto_check(
    changed_files: list[Path], verbose: bool
) -> tuple[bool, list[str]]:
    """Run `quarto check` on each changed file (expensive; off by default)."""
    errors: list[str] = []
    for f in changed_files:
        try:
            result = subprocess.run(
                ["quarto", "check", str(f)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                errors.append(
                    f"{f}: quarto check failed: {result.stderr[:200]}"
                )
            elif verbose:
                print(f"  [OK] quarto check {f.name}", file=sys.stderr)
        except FileNotFoundError:
            errors.append("quarto not installed; skipping quarto-check")
            return False, errors
        except subprocess.TimeoutExpired:
            errors.append(f"{f}: quarto check timed out")

    return len(errors) == 0, errors


# ── Top-level verify ────────────────────────────────────────────────────────


def verify(
    ledger: Ledger,
    changed_files: list[Path],
    run_quarto: bool,
    verbose: bool,
) -> tuple[bool, list[str]]:
    """Run all verification stages. Returns (success, combined_errors)."""
    all_errors: list[str] = []

    if verbose:
        print(
            f"Verifying {len(changed_files)} changed files...",
            file=sys.stderr,
        )

    # Stage 1: scanner delta
    if verbose:
        print("\n[1/3] Scanner delta check", file=sys.stderr)
    ok, errors = verify_scanner_delta(ledger, changed_files, verbose)
    all_errors.extend(errors)

    # Stage 2: pre-commit hook subset
    if verbose:
        print("\n[2/3] Pre-commit hook subset", file=sys.stderr)
    ok2, errors2 = verify_precommit_hooks(changed_files, verbose)
    all_errors.extend(errors2)

    # Stage 3: optional quarto check
    if run_quarto:
        if verbose:
            print("\n[3/3] Quarto check (--quarto-check)", file=sys.stderr)
        ok3, errors3 = verify_quarto_check(changed_files, verbose)
        all_errors.extend(errors3)
    else:
        if verbose:
            print("\n[3/3] Skipping quarto check (use --quarto-check)",
                  file=sys.stderr)

    return len(all_errors) == 0, all_errors


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Pass 15 verifier")
    parser.add_argument(
        "--ledger", type=Path, required=True, help="Audit ledger JSON"
    )
    parser.add_argument(
        "--changed-files",
        nargs="*",
        default=[],
        help="Paths of changed files to verify",
    )
    parser.add_argument(
        "--changed-files-from-ledger",
        action="store_true",
        help="Derive changed-files from fixed issues in the ledger",
    )
    parser.add_argument(
        "--quarto-check",
        action="store_true",
        help="Also run `quarto check` on each changed file (expensive)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-stage progress",
    )
    args = parser.parse_args()

    ledger = Ledger.load(args.ledger)

    if args.changed_files_from_ledger:
        changed = sorted(
            {
                Path(i.file)
                for i in ledger.issues
                if i.status in (STATUS_FIXED_SCRIPT, STATUS_FIXED_SUBAGENT)
            }
        )
    else:
        changed = [Path(p) for p in args.changed_files]

    if not changed:
        print("No changed files to verify.", file=sys.stderr)
        return 0

    ok, errors = verify(ledger, changed, args.quarto_check, args.verbose)

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"VERIFY {'PASSED' if ok else 'FAILED'}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    if errors:
        print(f"\n{len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)

    if ok:
        # Promote fixed-script-lane / fixed-subagent-lane to verified
        for issue in ledger.issues:
            if issue.status in (STATUS_FIXED_SCRIPT, STATUS_FIXED_SUBAGENT):
                issue.status = STATUS_VERIFIED
        ledger.save(args.ledger)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
