#!/usr/bin/env python3
"""Report the status of every registered workload.

Reads the registry and the retained evidence only. Runs no workload, fetches no
asset, produces no timing.

Three dimensions are reported separately, because conflating them is how a
workload that is doing fine gets read as unfinished:

  CONFIG   Is the workload configured well enough for someone to run it?

  QUALITY  Did the authoritative contract run, and did it meet the target?
           The registry sets `acceptance_runs: 1`, so one complete run accepts
           or rejects a quality result. This is the dimension that says whether
           a workload works.

  TIMING   Is fresh-process timing repeatability established? The registry sets
           `outer_reference_runs: 5` in the measurement protocol. This belongs
           to the later promotion phase and never gates a quality decision.

Quality decisions are recomputed against the live registry contract, so a
retained record graded under a superseded gate cannot report its own result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlperf.registry import load_registry  # noqa: E402

REQUIRED_TOP = ("dataset", "runner", "modes", "implemented_modes", "quality_target")
REQUIRED_QUALITY = ("metric", "value", "direction", "target_basis", "tolerance")
REQUIRED_PROTOCOL = ("primary_metric", "scenario", "repeatability_limit")
REQUIRED_CONTRACT = ("result_role", "mode", "quality")
PIN_TOKENS = ("sha256", "commit", "revision")


def missing(mapping: dict, keys: tuple[str, ...]) -> list[str]:
    return [k for k in keys if mapping.get(k) in (None, "", [], {})]


def gate_satisfied(value: float, gate: dict) -> bool:
    target = float(gate["target"])
    tolerance = float(gate.get("tolerance") or 0.0)
    if gate.get("direction") == "lower":
        return value <= target + tolerance
    return value >= target - tolerance


def config_status(workload) -> tuple[bool, list[str]]:
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
        if not any(any(t in k for t in PIN_TOKENS) for k in contract):
            gaps.append("canonical_max_contract: nothing pinned")

    for mode in raw.get("modes") or []:
        if mode not in (raw.get("implemented_modes") or []):
            gaps.append(f"mode declared but not implemented: {mode}")
    return (not gaps), gaps


def load_evidence() -> dict[str, list[dict]]:
    index = json.loads((ROOT / "provisional_results" / "index.json").read_text())
    cases = index if isinstance(index, list) else index.get("cases", [])
    by_workload: dict[str, list[dict]] = {}
    for entry in cases:
        record = json.loads((ROOT / entry["path"]).read_text())
        record["_entry"] = entry
        by_workload.setdefault(entry["workload"], []).append(record)
    return by_workload


def status_for(workload, records: list[dict]) -> dict:
    contract = (workload.raw.get("canonical_max_contract") or {}).get("quality") or {}
    execution_status = str(
        (workload.raw.get("canonical_max_contract") or {}).get("execution_status")
        or "quality-conformant"
    )
    ready, gaps = config_status(workload)

    score = [r for r in records if r["_entry"].get("result_role") == "score-bearing"]
    perf = [r for r in records if r["_entry"].get("result_role") == "performance-bearing"]

    quality, observed, runs, timing, stale = "NOT RUN", None, 0, "n/a", False
    if score:
        record = score[0]
        q = record.get("quality") or {}
        observed = (q.get("aggregate") or {}).get("median")
        runs = int((record.get("measurement") or {}).get("run_count") or 0)
        rec_gate = q.get("gate") or {}
        stale = rec_gate.get("target") != contract.get("target")
        quality = "PASS" if gate_satisfied(float(observed), contract) else "MISS"

        rep = record.get("repeatability") or {}
        if rep.get("passed") and runs >= 5:
            timing = f"verified ({runs})"
        elif runs:
            timing = f"not established ({runs})"
    elif execution_status == "quality-audited-target-not-met":
        quality = "MISS*"
    elif execution_status == "environment-gated-quality-conformance":
        quality = "BLOCKED"

    need = {
        "PASS": "none",
        "MISS": "target gap, investigated",
        "MISS*": "import result into evidence index",
        "BLOCKED": "local backend required",
        "NOT RUN": "run the max contract",
    }[quality]

    return {
        "workload": workload.id,
        "config": "ok" if ready else "GAPS",
        "config_gaps": gaps,
        "quality": quality,
        "observed": observed,
        "target": contract.get("target"),
        "direction": contract.get("direction"),
        "metric": contract.get("metric"),
        "runs": runs,
        "timing": timing,
        "perf_cases": len(perf),
        "stale_gate": stale,
        "needs": need,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    workloads = load_registry(ROOT / "registry")
    evidence = load_evidence()
    rows = [status_for(w, evidence.get(wid, [])) for wid, w in sorted(workloads.items())]

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    w = max(len(r["workload"]) for r in rows)
    print(
        f"{'WORKLOAD':<{w}}  {'CONFIG':<7} {'QUALITY':<8} "
        f"{'OBSERVED vs TARGET':<26} {'TIMING':<22} NEEDS"
    )
    print("=" * 124)
    for r in rows:
        if r["observed"] is None:
            cmp_text = "-"
        else:
            op = "<=" if r["direction"] == "lower" else ">="
            cmp_text = f"{r['observed']:.4f} {op} {float(r['target']):.4f}"
            if r["stale_gate"]:
                cmp_text += " !"
        print(
            f"{r['workload']:<{w}}  {r['config']:<7} {r['quality']:<8} "
            f"{cmp_text:<26} {r['timing']:<22} {r['needs']}"
        )
    print("=" * 124)

    for label, key in (("quality", "quality"), ("config", "config")):
        tally: dict[str, int] = {}
        for r in rows:
            tally[r[key]] = tally.get(r[key], 0) + 1
        summary = "  ".join(f"{k}={v}" for k, v in sorted(tally.items()))
        print(f"{label:<8} {summary}")

    verified = sum(1 for r in rows if r["timing"].startswith("verified"))
    print(
        f"{'timing':<8} repeatability established={verified}  "
        f"(later phase; never gates a quality decision)"
    )

    if any(r["stale_gate"] for r in rows):
        stale = [r["workload"] for r in rows if r["stale_gate"]]
        print(f"\n!  retained record graded under a superseded gate: {', '.join(stale)}")
        print("   the live registry contract is used above; regenerate to clear it")

    gapped = [r for r in rows if r["config_gaps"]]
    if gapped:
        print("\nCONFIG GAPS:")
        for r in gapped:
            print(f"  {r['workload']}:")
            for g in r["config_gaps"]:
                print(f"    - {g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
