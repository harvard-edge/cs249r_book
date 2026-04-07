#!/usr/bin/env python3
"""Pass 15 orchestrator — the five-stage cycle.

Runs SCAN -> PLAN -> FIX -> VERIFY -> REPORT. Stops when any of the
five conditions in Pass 15 plan section 2.5 is met:

  1. Zero open issues remaining in active categories -> exit success
  2. No progress in last iteration -> exit "stuck", print surviving
  3. Verification failure -> exit "verification failed", do not retry
  4. Time budget exceeded -> exit "budget exceeded"
  5. Category-level abort (> 5 verification failures in one run) ->
     disable that category, continue others

Usage:
    # Scout mode - scan only, no fixes
    python3 book/tools/audit/loop.py --scope vol2 --dry-run

    # Fix + verify, one category at a time
    python3 book/tools/audit/loop.py --scope vol2 \\
        --categories vs-period --apply --verbose

    # Fix + verify + commit, multiple categories
    python3 book/tools/audit/loop.py --scope vol2 \\
        --categories vs-period,compound-prefix-closeup \\
        --apply --commit-each-iteration --verbose

Phase B (subagent dispatch) adds a separate lane but uses the same cycle.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit.fix_script_lane import fix_all as script_lane_fix_all
from audit.ledger import (
    Issue,
    Ledger,
    STATUS_FAILED_SCRIPT,
    STATUS_FAILED_VERIFY,
    STATUS_FIXED_SCRIPT,
    STATUS_FIXED_SUBAGENT,
    STATUS_OPEN,
    STATUS_VERIFIED,
)
from audit.scan import scan, print_summary, REPO_ROOT
from audit.verify import verify

# ── Stage runners ───────────────────────────────────────────────────────────


def stage_scan(scope: str, categories: list[str] | None, verbose: bool) -> Ledger:
    """Run the SCAN stage."""
    if verbose:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"[1/5] SCAN scope={scope}", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)
    return scan(scope, categories, verbose=verbose)


def stage_plan(
    ledger: Ledger, verbose: bool
) -> dict[str, list[Issue]]:
    """Run the PLAN stage.

    Group open issues by lane. Returns a dict:
      {
        'script':     [...open auto-fixable issues...],
        'subagent':   [...open needs-subagent issues...],
        'accepted':   [...open protected-context issues...],
      }
    """
    if verbose:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"[2/5] PLAN", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)

    lanes: dict[str, list[Issue]] = {
        "script": [],
        "subagent": [],
        "accepted": [],
    }
    for issue in ledger.issues:
        if issue.status != STATUS_OPEN:
            continue
        if issue.protected_context:
            lanes["accepted"].append(issue)
        elif issue.auto_fixable:
            lanes["script"].append(issue)
        elif issue.needs_subagent:
            lanes["subagent"].append(issue)
        else:
            # Detection-only, e.g. binary-units-in-prose
            lanes["accepted"].append(issue)

    if verbose:
        for lane, issues in lanes.items():
            print(f"  {lane:10s} {len(issues):6d} issues", file=sys.stderr)

    return lanes


def stage_fix(
    ledger: Ledger,
    lanes: dict[str, list[Issue]],
    categories: list[str] | None,
    dry_run: bool,
    verbose: bool,
) -> tuple[int, list[Path]]:
    """Run the FIX stage (script lane only in Phase A).

    Returns (applied_count, list_of_changed_files).
    """
    if verbose:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(
            f"[3/5] FIX {'(dry run)' if dry_run else '(LIVE)'}",
            file=sys.stderr,
        )
        print(f"{'=' * 60}", file=sys.stderr)

    # Script lane
    script_results = script_lane_fix_all(
        ledger,
        categories=categories,
        dry_run=dry_run,
        verbose=verbose,
    )

    # Collect changed files (those with successful fixes)
    changed: set[Path] = set()
    applied = 0
    for r in script_results:
        if r.applied > 0:
            changed.add(Path(r.file))
            applied += r.applied

    return applied, sorted(changed)


def stage_verify(
    ledger: Ledger,
    changed_files: list[Path],
    dry_run: bool,
    run_quarto: bool,
    verbose: bool,
) -> bool:
    """Run the VERIFY stage. Returns True if all checks pass."""
    if verbose:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"[4/5] VERIFY", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)

    if dry_run:
        if verbose:
            print("  Skipping verification in dry-run mode", file=sys.stderr)
        return True

    if not changed_files:
        if verbose:
            print("  No changed files to verify", file=sys.stderr)
        return True

    ok, errors = verify(ledger, changed_files, run_quarto, verbose)
    if not ok:
        print(f"\nVerification failed with {len(errors)} error(s):",
              file=sys.stderr)
        for e in errors[:10]:
            print(f"  - {e}", file=sys.stderr)

        # Roll back fixed issues in the ledger (mark failed-verification).
        # We do NOT roll back file contents — the operator must do that
        # manually with `git checkout --`. That is safer than automated
        # rollback because it gives the human a chance to inspect.
        for issue in ledger.issues:
            if issue.status == STATUS_FIXED_SCRIPT:
                issue.status = STATUS_FAILED_VERIFY
                issue.error = "verification stage rejected"
    else:
        # Promote to verified
        for issue in ledger.issues:
            if issue.status in (STATUS_FIXED_SCRIPT, STATUS_FIXED_SUBAGENT):
                issue.status = STATUS_VERIFIED

    return ok


def stage_report(
    ledger: Ledger,
    applied: int,
    iteration: int,
    verbose: bool,
) -> None:
    """Run the REPORT stage."""
    if verbose:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"[5/5] REPORT (iteration {iteration})", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)
    print_summary(ledger)
    print(f"\nIteration {iteration}: {applied} fixes applied and verified",
          file=sys.stderr)


def stage_commit(
    changed_files: list[Path],
    message: str,
    verbose: bool,
) -> bool:
    """Create a git commit for the changed files."""
    if not changed_files:
        return True

    rel_files = []
    for f in changed_files:
        try:
            rel_files.append(str(f.relative_to(REPO_ROOT)))
        except ValueError:
            rel_files.append(str(f))

    try:
        subprocess.run(
            ["git", "add", "--"] + rel_files,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(
                f"Commit failed: {result.stdout}\n{result.stderr}",
                file=sys.stderr,
            )
            return False
        if verbose:
            print(f"Committed: {message}", file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"git command failed: {e}", file=sys.stderr)
        return False


# ── Orchestrator ────────────────────────────────────────────────────────────


def run_loop(args: argparse.Namespace) -> int:
    """Run the full five-stage cycle with stopping conditions."""
    start_time = time.time()
    max_wall_seconds = args.time_budget_min * 60

    categories = (
        [c.strip() for c in args.categories.split(",") if c.strip()]
        if args.categories
        else None
    )

    last_open_count = -1
    iteration = 0
    total_applied = 0

    while iteration < args.max_iterations:
        iteration += 1

        # Check time budget
        elapsed = time.time() - start_time
        if elapsed > max_wall_seconds:
            print(
                f"\n[STOP] Time budget exceeded: {elapsed:.0f}s > "
                f"{max_wall_seconds}s",
                file=sys.stderr,
            )
            return 4

        # Stage 1: SCAN
        ledger = stage_scan(args.scope, categories, args.verbose)
        ledger_path = args.ledger or Path("audit-ledger.json")
        ledger.save(ledger_path)

        # Stage 2: PLAN
        lanes = stage_plan(ledger, args.verbose)
        open_count = len(lanes["script"]) + len(lanes["subagent"])

        # Stopping condition 1: zero open issues
        if open_count == 0:
            print(
                f"\n[STOP] Zero open issues across active categories",
                file=sys.stderr,
            )
            stage_report(ledger, total_applied, iteration, args.verbose)
            return 0

        # Stopping condition 2: no progress
        if open_count == last_open_count:
            print(
                f"\n[STOP] No progress in iteration {iteration} "
                f"(open count stuck at {open_count})",
                file=sys.stderr,
            )
            stage_report(ledger, total_applied, iteration, args.verbose)
            return 2
        last_open_count = open_count

        if args.dry_run:
            stage_report(ledger, 0, iteration, args.verbose)
            print("\n[DRY RUN] Exiting after first scan+plan", file=sys.stderr)
            return 0

        # Stage 3: FIX (script lane only in Phase A)
        applied, changed = stage_fix(
            ledger, lanes, categories, args.dry_run, args.verbose
        )
        total_applied += applied
        ledger.save(ledger_path)

        if applied == 0:
            # Nothing to fix in script lane; if no subagents configured,
            # we're done with what the orchestrator can do alone.
            if not lanes["subagent"]:
                print(
                    f"\n[STOP] No auto-fixable issues remaining "
                    f"(subagent lane has {len(lanes['subagent'])} pending)",
                    file=sys.stderr,
                )
                stage_report(ledger, total_applied, iteration, args.verbose)
                return 0

        # Stage 4: VERIFY
        ok = stage_verify(
            ledger, changed, args.dry_run, args.quarto_check, args.verbose
        )
        ledger.save(ledger_path)

        if not ok:
            print(
                f"\n[STOP] Verification failed in iteration {iteration}. "
                f"The orchestrator does NOT retry failed verification — "
                f"inspect the changed files manually with `git diff` and "
                f"either commit or roll back.",
                file=sys.stderr,
            )
            return 3

        # Stage 5: REPORT
        stage_report(ledger, applied, iteration, args.verbose)

        # Commit if requested
        if args.commit_each_iteration and changed:
            if categories:
                cat_summary = ",".join(categories)
            else:
                cat_summary = "all"
            commit_msg = (
                f"pass 15 audit iter {iteration}: "
                f"apply {applied} {cat_summary} fixes to {len(changed)} files"
            )
            if not stage_commit(changed, commit_msg, args.verbose):
                print(
                    "\n[STOP] Commit failed. The working tree has staged changes.",
                    file=sys.stderr,
                )
                return 5

    print(
        f"\n[STOP] Max iterations ({args.max_iterations}) reached",
        file=sys.stderr,
    )
    return 4


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pass 15 audit loop orchestrator",
    )
    parser.add_argument(
        "--scope", required=True,
        help="vol1 | vol2 | both | path",
    )
    parser.add_argument(
        "--categories", default="",
        help="Comma-separated category names (default: all)",
    )
    parser.add_argument(
        "--ledger", type=Path, default=None,
        help="Path to the ledger JSON (default: audit-ledger.json)",
    )
    mutex = parser.add_mutually_exclusive_group(required=True)
    mutex.add_argument(
        "--dry-run", action="store_true",
        help="Run scan + plan only; no fixes, no verify, no commit",
    )
    mutex.add_argument(
        "--apply", action="store_true",
        help="Actually apply the changes",
    )
    parser.add_argument(
        "--commit-each-iteration", action="store_true",
        help="Create a git commit after each successful iteration",
    )
    parser.add_argument(
        "--quarto-check", action="store_true",
        help="Also run `quarto check` in the verify stage (expensive)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10,
        help="Maximum number of iterations (default: 10)",
    )
    parser.add_argument(
        "--time-budget-min", type=int, default=30,
        help="Wall-clock time budget in minutes (default: 30)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-stage progress",
    )
    args = parser.parse_args()
    return run_loop(args)


if __name__ == "__main__":
    sys.exit(main())
