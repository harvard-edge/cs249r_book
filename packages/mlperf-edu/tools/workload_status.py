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

  TIMING   How many timing measurements a case has. The registry sets
           `outer_reference_runs: 1`, matching the single-run acceptance rule.
           Repeated timing is reported where it exists and never gates a
           quality decision.

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
            timing = f"{runs} run(s)"
        elif runs:
            timing = f"{runs} run(s)"
    elif execution_status == "quality-audited-target-not-met":
        # The run happened and is digest-bound in the registry; it just never
        # made it into the evidence index. Report its value rather than a dash.
        quality = "MISS*"
        measured = (
            workload.raw.get("canonical_max_contract") or {}
        ).get("measured_evidence") or {}
        for key in ("score", "best_score"):
            if isinstance(measured.get(key), (int, float)):
                observed = float(measured[key])
                break
        runs = int(measured.get("result_count") or 1)
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


def missing_items(rows: list[dict], workloads: dict) -> list[str]:
    """Derive the gap list instead of hardcoding it.

    This block previously named recommendation as a DLRM workload blocked on
    licensed Criteo data, which stopped being true when the contract moved to
    NCF on MovieLens-20M, and it stated a miss count that drifted as workloads
    were measured. Reading both from the registry keeps the document honest.
    """
    items: list[str] = []
    for row in rows:
        if row["quality"] != "BLOCKED":
            continue
        workload = workloads[row["workload"]]
        gate = ((workload.raw.get("spiral") or {}).get("next_gate") or "").strip()
        detail = f" {gate}" if gate else ""
        items.append(f"- `{row['workload']}` cannot run its contract locally.{detail}")
    audited = sum(1 for row in rows if row["quality"] == "MISS*")
    if audited:
        items.append(
            f"- {audited} audited miss{'es are' if audited != 1 else ' is'} recorded "
            "in the registry but not imported into the evidence index, so "
            f"{'they carry' if audited != 1 else 'it carries'} digests and runtime "
            "without appearing as cases."
        )
    return items or ["- Nothing outstanding."]


def humanize(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def runtime_rows(workloads, evidence: dict[str, list[dict]]) -> list[dict]:
    """Every measured runtime we hold, from either evidence records or the registry.

    A workload whose authoritative run is recorded in the registry but not
    imported into the evidence index still has a real, digest-bound runtime.
    Reporting only the index would hide it.
    """
    rows: list[dict] = []
    for wid, workload in sorted(workloads.items()):
        for record in evidence.get(wid, []):
            measurement = record.get("measurement") or {}
            median = (measurement.get("aggregate") or {}).get("median")
            if median is None:
                continue
            phase = record.get("phase")
            metric = str(measurement.get("primary_metric") or "")
            rows.append(
                {
                    "workload": wid,
                    "case": record.get("mode") + (f" / {phase}" if phase else ""),
                    "metric": metric,
                    "value": float(median),
                    # Only a *_seconds metric is a duration. The causal inference
                    # phases report throughput, and formatting a rate as a
                    # duration produces nonsense.
                    "is_duration": metric.endswith("_seconds"),
                    "runs": int(measurement.get("run_count") or 0),
                    "source": "evidence index",
                }
            )

        contract = workload.raw.get("canonical_max_contract") or {}
        measured = contract.get("measured_evidence") or {}
        for key, value in measured.items():
            if not key.endswith("_seconds") or not isinstance(value, (int, float)):
                continue
            rows.append(
                {
                    "workload": wid,
                    "case": str(contract.get("mode") or "max"),
                    "metric": key,
                    "value": float(value),
                    "is_duration": True,
                    "runs": int(measured.get("result_count") or 1),
                    "source": "registry audit record",
                }
            )
    return rows


def render_markdown(rows: list[dict], runtimes: list[dict], workloads: dict) -> str:
    quality_tally: dict[str, int] = {}
    for r in rows:
        quality_tally[r["quality"]] = quality_tally.get(r["quality"], 0) + 1
    repeated = sum(1 for r in rows if r["perf_cases"] and (r["runs"] or 0) > 1)

    out = [
        "<!-- GENERATED FILE - do not edit by hand.",
        "     Regenerate with: python3 tools/workload_status.py --write -->",
        "",
        "# Workload Status",
        "",
        "Every number here is read from the registry and the retained evidence.",
        "Quality decisions are recomputed against the live registry contract, so a",
        "record graded under a superseded gate cannot report its own result.",
        "",
        "## Summary",
        "",
        "| Dimension | Result |",
        "|:---|:---|",
        f"| Workloads registered | {len(rows)} |",
        f"| Quality contract passed | {quality_tally.get('PASS', 0)} |",
        f"| Target missed, recorded | {quality_tally.get('MISS', 0) + quality_tally.get('MISS*', 0)} |",
        f"| Blocked on a local backend | {quality_tally.get('BLOCKED', 0)} |",
        f"| Configuration defects | {sum(1 for r in rows if r['config'] != 'ok')} |",
        f"| Cases with repeated timing | {repeated} |",
        "",
        "Quality is decided on the registry's `acceptance_runs: 1`, so one complete",
        "run accepts or rejects a result. Timing repeatability uses",
        "`outer_reference_runs: 5` and belongs to the later promotion phase; it never",
        "gates a quality decision.",
        "",
        "## Quality",
        "",
        "| Workload | Config | Quality | Observed | Target | Timing | Needs |",
        "|:---|:---|:---|---:|---:|:---|:---|",
    ]
    for r in rows:
        if r["observed"] is None:
            observed = target = "—"
        else:
            op = "≤" if r["direction"] == "lower" else "≥"
            observed = f"{r['observed']:.4f}"
            target = f"{op} {float(r['target']):.4f}"
            if r["stale_gate"]:
                observed += " ⚠"
        out.append(
            f"| `{r['workload']}` | {r['config']} | **{r['quality']}** | "
            f"{observed} | {target} | {r['timing']} | {r['needs']} |"
        )

    out += [
        "",
        "`MISS*` means the authoritative contract ran and missed its target, and the",
        "result is recorded in the registry but not imported into the evidence index.",
        "⚠ marks a retained record still carrying a superseded gate.",
        "",
        "## Measured Runtime",
        "",
        "Training and inference are separate cases under one workload identity.",
        "These are the runtimes actually recorded, not estimates.",
        "",
        "| Workload | Case | Metric | Measured | Runs | Source |",
        "|:---|:---|:---|---:|---:|:---|",
    ]
    for rt in sorted(runtimes, key=lambda x: (x["workload"], x["case"])):
        if rt["is_duration"]:
            measured = humanize(rt["value"])
        else:
            measured = f"{rt['value']:,.1f} /s"
        out.append(
            f"| `{rt['workload']}` | {rt['case']} | `{rt['metric']}` | "
            f"{measured} | {rt['runs']} | {rt['source']} |"
        )

    total = sum(
        rt["value"]
        for rt in runtimes
        if rt["is_duration"] and rt["source"] == "evidence index"
    )
    out += [
        "",
        "Rows marked `/s` are throughput, not elapsed time. The causal inference",
        "phases report tokens per second, so they are excluded from the total below.",
        "",
        f"One pass through every timed case in the evidence index is about "
        f"{humanize(total)} of compute.",
        "",
        "## What Is Missing",
        "",
        *missing_items(rows, workloads),
        "",
    ]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--markdown", action="store_true", help="emit markdown to stdout")
    ap.add_argument(
        "--write",
        action="store_true",
        help="write docs/internal/WORKLOAD_STATUS.md",
    )
    args = ap.parse_args()

    workloads = load_registry(ROOT / "registry")
    evidence = load_evidence()
    rows = [status_for(w, evidence.get(wid, [])) for wid, w in sorted(workloads.items())]

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    if args.markdown or args.write:
        document = render_markdown(rows, runtime_rows(workloads, evidence), workloads)
        if args.write:
            target = ROOT / "docs" / "internal" / "WORKLOAD_STATUS.md"
            target.write_text(document)
            print(f"wrote {target.relative_to(ROOT)}")
        else:
            print(document)
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

    repeated = sum(1 for r in rows if (r["runs"] or 0) > 1)
    print(
        f"{'timing':<8} cases with repeated timing={repeated}  "
        f"(reported, never gates a quality decision)"
    )

    # The min/max/pro contract is only meaningful if every workload has all three.
    from mlperf.registry import RESEARCH_WORKLOADS

    pro_gap = sorted(set(w["workload"] for w in rows) - set(RESEARCH_WORKLOADS))
    print(
        f"{'profiles':<8} min={len(rows)}  max={len(rows)}  "
        f"pro={len(rows) - len(pro_gap)}"
        + (f"  MISSING pro: {', '.join(pro_gap)}" if pro_gap else "")
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
