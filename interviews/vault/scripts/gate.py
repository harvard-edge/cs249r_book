#!/usr/bin/env python3
"""Invariant gate — import this in any pipeline script to block on regressions.

Usage in other scripts:

    from gate import require_no_regressions

    # ... do work that modifies corpus/taxonomy ...

    require_no_regressions()  # exits 1 if new FAILs appeared

Or as a before/after wrapper:

    from gate import InvariantGate

    with InvariantGate():
        # ... modify corpus.json, taxonomy.json, chains.json ...
    # automatically checks for regressions on exit
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Import the invariant checker
sys.path.insert(0, str(Path(__file__).parent))
from vault_invariants import ALL_CHECKS, load_data


def run_checks() -> dict[int, str]:
    """Run all invariant checks, return {check_num: status}."""
    corpus, taxonomy, chains = load_data()
    results = {}
    for check_fn in ALL_CHECKS:
        result = check_fn(taxonomy=taxonomy, corpus=corpus, chains=chains)
        results[result.num] = result.status
    return results


def count_fails(results: dict[int, str]) -> int:
    return sum(1 for s in results.values() if s == "FAIL")


def require_no_regressions(baseline: dict[int, str] | None = None):
    """Check that no NEW fails were introduced since baseline.

    If baseline is None, just checks for zero fails total.
    """
    current = run_checks()
    current_fails = count_fails(current)

    if baseline is None:
        if current_fails > 0:
            failing = [n for n, s in current.items() if s == "FAIL"]
            print(f"\n  GATE BLOCKED: {current_fails} failing invariant checks: {failing}")
            print(f"  Run: python3 scripts/vault_invariants.py for details")
            sys.exit(1)
        return

    baseline_fails = count_fails(baseline)
    if current_fails > baseline_fails:
        new_fails = [n for n, s in current.items()
                     if s == "FAIL" and baseline.get(n) != "FAIL"]
        print(f"\n  GATE BLOCKED: {len(new_fails)} new failing checks: {new_fails}")
        print(f"  (was {baseline_fails} fails, now {current_fails})")
        print(f"  Run: python3 scripts/vault_invariants.py for details")
        sys.exit(1)


class InvariantGate:
    """Context manager that captures baseline before and checks after."""

    def __enter__(self):
        self.baseline = run_checks()
        baseline_fails = count_fails(self.baseline)
        if baseline_fails:
            print(f"  [gate] Baseline: {baseline_fails} pre-existing fails (will allow these)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False  # don't suppress exceptions
        require_no_regressions(self.baseline)
        return False
