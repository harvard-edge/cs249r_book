#!/usr/bin/env python3
"""
Iter-4 taxonomy linter for MLPerf EDU.

Enforces three invariants on workloads.yaml:

  (1) Every workload has a complete `regime` block with all three axes:
      working_set, arithmetic_intensity, dispatch.
  (2) Categorical `value` on each axis is one of the allowed strings.
  (3) When `value` is non-`unmeasured`, the numerical evidence supplied
      on that axis must be consistent with the classification thresholds
      declared below (drawn from Emer's iter-4 proposal).

`unmeasured` is allowed as a value but is tracked in the summary so we
don't lose sight of which axes are still pending instrumentation.

Run: python3 tools/check_taxonomy.py
Exit codes: 0 if all invariants hold, 1 if any error, 2 if no workloads
seen.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKLOADS_YAML = REPO_ROOT / "workloads.yaml"

# Reference platform thresholds (Apple M1; documented in proposal).
LLC_BYTES = 12 * 1024 * 1024
RIDGE_FLOPS_PER_BYTE = 30
PEAK_BW_GBPS = 68.25  # M1 unified memory peak (informational)

VALID_VALUES = {
    "working_set": {"cache_resident", "dram_bound", "unmeasured"},
    "arithmetic_intensity": {"compute_bound", "bandwidth_bound", "unmeasured"},
    "dispatch": {"device_saturated", "dispatch_bound", "unmeasured"},
}


def check_regime(name: str, regime: dict) -> tuple[list[str], dict]:
    """Return (errors, axis_values) for one workload's regime block."""
    errors: list[str] = []
    values = {}

    for axis, allowed in VALID_VALUES.items():
        if axis not in regime:
            errors.append(f"{name}: missing axis '{axis}'")
            values[axis] = None
            continue
        block = regime[axis]
        if not isinstance(block, dict):
            errors.append(f"{name}.{axis}: expected dict, got {type(block).__name__}")
            values[axis] = None
            continue
        v = block.get("value")
        if v not in allowed:
            errors.append(f"{name}.{axis}: value '{v}' not in {sorted(allowed)}")
        values[axis] = v

    # Axis A — working_set numerical consistency.
    ws = regime.get("working_set", {})
    if ws.get("value") == "cache_resident":
        b = ws.get("peak_bytes_per_step")
        if b is not None and b > 0.5 * LLC_BYTES:
            errors.append(
                f"{name}.working_set: cache_resident but "
                f"peak_bytes_per_step={b:,} > 0.5 * LLC ({int(0.5*LLC_BYTES):,})"
            )
    elif ws.get("value") == "dram_bound":
        b = ws.get("peak_bytes_per_step")
        if b is not None and b < 4 * LLC_BYTES:
            errors.append(
                f"{name}.working_set: dram_bound but "
                f"peak_bytes_per_step={b:,} < 4 * LLC ({int(4*LLC_BYTES):,})"
            )

    # Axis B — arithmetic_intensity numerical consistency.
    ai = regime.get("arithmetic_intensity", {})
    fpb = ai.get("flops_per_byte")
    if ai.get("value") == "compute_bound" and fpb is not None:
        if fpb < 2 * RIDGE_FLOPS_PER_BYTE:
            errors.append(
                f"{name}.arithmetic_intensity: compute_bound but intensity "
                f"{fpb} < 2*ridge ({2 * RIDGE_FLOPS_PER_BYTE})"
            )
    if ai.get("value") == "bandwidth_bound" and fpb is not None:
        if fpb > 0.5 * RIDGE_FLOPS_PER_BYTE:
            errors.append(
                f"{name}.arithmetic_intensity: bandwidth_bound but intensity "
                f"{fpb} > 0.5*ridge ({0.5 * RIDGE_FLOPS_PER_BYTE})"
            )

    # Axis C — dispatch numerical consistency.
    d = regime.get("dispatch", {})
    util = d.get("utilization")
    if d.get("value") == "device_saturated" and util is not None and util < 0.50:
        errors.append(
            f"{name}.dispatch: device_saturated but utilization {util} < 0.50"
        )
    if d.get("value") == "dispatch_bound" and util is not None and util > 0.25:
        errors.append(
            f"{name}.dispatch: dispatch_bound but utilization {util} > 0.25"
        )

    return errors, values


def latest_sidecar_for(workload: str, sidecar_dir: Path) -> dict | None:
    """Return the most recent roofline sidecar for `workload`, or None."""
    if not sidecar_dir.exists():
        return None
    cands = sorted(sidecar_dir.glob(f"{workload}_*.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        return None
    try:
        return json.loads(cands[0].read_text())
    except (json.JSONDecodeError, OSError):
        return None


def verify_against_sidecar(name: str, regime: dict, sidecar: dict) -> list[str]:
    """Cross-check YAML regime claims against measured sidecar."""
    errs: list[str] = []
    measured = sidecar.get("regime_inference", {})
    for axis_yaml, axis_sidecar in [
        ("arithmetic_intensity", "axis_arithmetic_intensity"),
        ("dispatch", "axis_dispatch"),
    ]:
        yaml_value = regime.get(axis_yaml, {}).get("value")
        sidecar_value = measured.get(axis_sidecar)
        if yaml_value in (None, "unmeasured"):
            continue
        if sidecar_value in (None, "unmeasured"):
            continue
        if yaml_value != sidecar_value:
            errs.append(
                f"{name}.{axis_yaml}: YAML claims '{yaml_value}' but sidecar "
                f"measured '{sidecar_value}' "
                f"(rule: {measured.get('rule', 'no rule')})"
            )
    return errs


def main() -> int:
    parser = argparse.ArgumentParser(description="Lint MLPerf EDU workload taxonomy.")
    parser.add_argument("--verify-against-sidecars", type=str, default=None,
                        metavar="DIR",
                        help="Cross-check YAML regime claims against the latest "
                             "roofline sidecar in DIR for each workload.")
    args = parser.parse_args()
    sidecar_dir = Path(args.verify_against_sidecars) if args.verify_against_sidecars else None

    if not WORKLOADS_YAML.exists():
        print(f"FAIL: {WORKLOADS_YAML} not found")
        return 2

    doc = yaml.safe_load(WORKLOADS_YAML.read_text())
    suites = doc.get("suites", {})
    if not suites:
        print("FAIL: no suites found")
        return 2

    all_errors: list[str] = []
    cell_counts: dict[tuple, list[str]] = {}
    unmeasured_axes: dict[str, list[str]] = {"working_set": [], "arithmetic_intensity": [], "dispatch": []}
    n_workloads = 0

    for div, workloads in suites.items():
        for name, body in workloads.items():
            n_workloads += 1
            full_name = f"{div}/{name}"
            if "regime" not in body:
                all_errors.append(f"{full_name}: no regime block")
                continue
            errs, values = check_regime(full_name, body["regime"])
            all_errors.extend(errs)
            for axis, v in values.items():
                if v == "unmeasured":
                    unmeasured_axes[axis].append(full_name)
            if sidecar_dir is not None:
                sidecar = latest_sidecar_for(name, sidecar_dir)
                if sidecar is not None:
                    all_errors.extend(verify_against_sidecar(full_name, body["regime"], sidecar))
            cell = (
                values.get("working_set"),
                values.get("arithmetic_intensity"),
                values.get("dispatch"),
            )
            cell_counts.setdefault(cell, []).append(full_name)

    print(f"Inspected {n_workloads} workloads.")
    print()

    # Cell occupancy report.
    print("Taxonomy cell occupancy (working_set, arithmetic_intensity, dispatch):")
    for cell, members in sorted(cell_counts.items(), key=lambda kv: -len(kv[1])):
        ws, ai, di = cell
        ws = "?" if ws is None else ws
        ai = "?" if ai is None else ai
        di = "?" if di is None else di
        print(f"  ({ws}, {ai}, {di}): {len(members)}")
        for m in members:
            print(f"      {m}")
    print()

    # Unmeasured tracker.
    for axis, names in unmeasured_axes.items():
        if names:
            print(f"Axis '{axis}' unmeasured on {len(names)} workload(s):")
            for n in names:
                print(f"  {n}")
            print()

    if all_errors:
        print(f"FAIL: {len(all_errors)} taxonomy violations:")
        for e in all_errors:
            print(f"  {e}")
        return 1
    print(f"PASS: {n_workloads} workloads consistent with taxonomy invariants.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
