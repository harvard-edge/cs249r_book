#!/usr/bin/env python3
"""
Iter-5.6: sync workloads.yaml's regime claims to match the latest
roofline sidecar measurements.

For every workload that has at least one sidecar in roofline/, replace
the YAML's `arithmetic_intensity.value` and `dispatch.value` with what
the sidecar measured. Adds an `evidence_sidecar` field pointing at the
sidecar file and an `evidence_sha256_short` field for audit-trail
integrity.

This is the YAML <-> measurement loop closing: instead of guessing iter-4
placements, the YAML now reflects what measure_roofline actually saw on
this host.

Run: python3 tools/sync_yaml_from_sidecars.py
Then: python3 tools/check_taxonomy.py --verify-against-sidecars roofline/
should exit 0.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKLOADS_YAML = REPO_ROOT / "workloads.yaml"
SIDECAR_DIR = REPO_ROOT / "roofline"


def latest_sidecar(workload: str) -> tuple[Path, dict] | None:
    if not SIDECAR_DIR.exists():
        return None
    cands = sorted(SIDECAR_DIR.glob(f"{workload}_*.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        return None
    try:
        return cands[0], json.loads(cands[0].read_text())
    except (json.JSONDecodeError, OSError):
        return None


def sha256_short(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main() -> int:
    doc = yaml.safe_load(WORKLOADS_YAML.read_text())
    suites = doc.get("suites", {})
    n_synced, n_no_sidecar = 0, 0
    notes: list[str] = []

    for div, workloads in suites.items():
        for name, body in workloads.items():
            entry = latest_sidecar(name)
            if entry is None:
                n_no_sidecar += 1
                continue
            sidecar_path, sidecar = entry
            inferred = sidecar.get("regime_inference", {})
            measured_ai = inferred.get("axis_arithmetic_intensity")
            measured_disp = inferred.get("axis_dispatch")
            measured_intensity = sidecar["measurement"]["intensity_FLOPS_per_byte"]
            measured_util = sidecar["measurement"]["dispatch_utilization"]
            measured_bw = sidecar["measurement"]["achieved_BW_GBps"]

            regime = body.setdefault("regime", {})
            evidence = {
                "evidence_sidecar": str(sidecar_path.relative_to(REPO_ROOT)),
                "evidence_sha256_short": sha256_short(sidecar_path),
                "measured_at": sidecar.get("utc"),
                "platform_machine_class": sidecar.get("platform", {}).get("machine_class", "unknown"),
            }

            ai_block = regime.setdefault("arithmetic_intensity", {})
            old_ai = ai_block.get("value")
            if measured_ai is not None:
                ai_block["value"] = measured_ai
                ai_block["flops_per_byte"] = round(float(measured_intensity), 3)
                ai_block["classification_rule"] = inferred.get("rule", ai_block.get("classification_rule", ""))
                ai_block.update(evidence)
                if old_ai and old_ai != measured_ai:
                    notes.append(f"  {div}/{name}.AI: {old_ai} -> {measured_ai}")

            disp_block = regime.setdefault("dispatch", {})
            old_disp = disp_block.get("value")
            if measured_disp is not None:
                disp_block["value"] = measured_disp
                disp_block["utilization"] = round(float(measured_util), 4)
                disp_block["achieved_bw_gbps"] = round(float(measured_bw), 3)
                disp_block["classification_rule"] = inferred.get("rule", disp_block.get("classification_rule", ""))
                disp_block.update(evidence)
                if old_disp and old_disp != measured_disp:
                    notes.append(f"  {div}/{name}.dispatch: {old_disp} -> {measured_disp}")

            n_synced += 1

    out = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=120)
    WORKLOADS_YAML.write_text(out)
    print(f"Synced {n_synced} workloads from sidecars; {n_no_sidecar} have no sidecar yet.")
    if notes:
        print(f"\nClassification changes ({len(notes)}):")
        for n in notes:
            print(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
