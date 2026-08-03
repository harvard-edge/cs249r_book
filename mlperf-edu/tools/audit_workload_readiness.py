#!/usr/bin/env python3
"""Static ship-readiness audit for every registered workload.

Reads the registry and source tree only. Executes no workload, downloads no
asset, and runs no timing. Answers one question per workload: could an outside
user download this and run it?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlperf.registry import load_registry  # noqa: E402

# `id` is hoisted to the Workload attribute by the loader and is not in `raw`.
REQUIRED_TOP = ("dataset", "runner", "modes", "implemented_modes", "quality_target")
PIN_TOKENS = ("sha256", "commit", "revision")
REQUIRED_QUALITY = ("metric", "value", "direction", "target_basis", "tolerance")
REQUIRED_PROTOCOL = ("primary_metric", "scenario", "repeatability_limit")
REQUIRED_CONTRACT = ("result_role", "mode", "quality")


def missing(mapping: dict, keys: tuple[str, ...]) -> list[str]:
    return [k for k in keys if mapping.get(k) in (None, "", [], {})]


def audit(workload) -> dict:
    raw = workload.raw
    gaps: list[str] = []

    for key in missing(raw, REQUIRED_TOP):
        gaps.append(f"registry: missing {key}")
    for key in missing(raw.get("quality_target") or {}, REQUIRED_QUALITY):
        gaps.append(f"quality_target: missing {key}")
    for key in missing(raw.get("measurement_protocol") or {}, REQUIRED_PROTOCOL):
        gaps.append(f"measurement_protocol: missing {key}")

    contract = raw.get("canonical_max_contract") or {}
    if not contract:
        gaps.append("no canonical_max_contract")
    else:
        for key in missing(contract, REQUIRED_CONTRACT):
            gaps.append(f"canonical_max_contract: missing {key}")

    # A shippable max path needs pinned bytes, otherwise the user cannot get
    # the same artifact the reference result used.
    pinned = [k for k in contract if any(t in k for t in PIN_TOKENS)]
    if contract and not pinned:
        gaps.append("canonical_max_contract: no pinned digest or upstream commit")

    modes = raw.get("modes") or []
    implemented = raw.get("implemented_modes") or []
    for mode in modes:
        if mode not in implemented:
            gaps.append(f"mode declared but not implemented: {mode}")

    # execution_status lives on the max contract, not at the top level.
    status = str(contract.get("execution_status") or "quality-conformant")
    promotion = bool(raw.get("promotion_scope", True))

    blockers = []
    if status not in ("unspecified", "quality-conformant"):
        blockers.append(status)
    if not promotion:
        blockers.append("promotion_scope: false")

    if gaps:
        verdict = "NO"
    elif blockers:
        verdict = "GAP"
    else:
        verdict = "YES"

    return {
        "workload": workload.id,
        "suite": raw.get("suite") or workload.id,
        "status": status,
        "promotion_scope": promotion,
        "verdict": verdict,
        "gaps": gaps,
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    workloads = load_registry(ROOT / "registry")
    results = [audit(w) for _, w in sorted(workloads.items())]

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    width = max(len(r["workload"]) for r in results)
    print(f"{'WORKLOAD':<{width}}  {'SHIP':<5} {'STATUS':<34} BLOCKERS / GAPS")
    print("=" * 118)
    for r in results:
        notes = "; ".join(r["blockers"] + r["gaps"]) or "-"
        print(f"{r['workload']:<{width}}  {r['verdict']:<5} {r['status']:<34} {notes[:60]}")

    print("=" * 118)
    for verdict in ("YES", "GAP", "NO"):
        n = sum(1 for r in results if r["verdict"] == verdict)
        print(f"{verdict:<5} {n}")

    detailed = [r for r in results if r["gaps"]]
    if detailed:
        print("\nCONFIG GAPS (block a clean run):")
        for r in detailed:
            print(f"  {r['workload']}:")
            for g in r["gaps"]:
                print(f"    - {g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
