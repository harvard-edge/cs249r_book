#!/usr/bin/env python3
"""Refresh target_source fields in book/tools/audits/mlsysim_constants/*.yaml."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
YAML_DIR = REPO_ROOT / "book" / "tools" / "audits" / "mlsysim_constants"
AUDIT_DIR = REPO_ROOT / "book" / "tools" / "audit"

def _load_merged_mapping() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location(
        "migrate_constants_to_registry",
        AUDIT_DIR / "migrate_constants_to_registry.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.merged_mapping()

def _fmt_target(target: str) -> str:
    if target.startswith("mlsysim."):
        return target
    if target.startswith("defaults."):
        return f"mlsysim.core.{target}"
    if target.startswith(("Hardware.", "Models.", "Datasets.", "Systems.", "Infrastructure.", "Platforms.")):
        return f"mlsysim.{target}"
    return target

def _tokens_from_value(value: str) -> list[str]:
    if not value or value in ("null", "~"):
        return []
    parts = re.split(r"[,\s]+", value.strip().strip('"').strip("'"))
    return [p for p in parts if re.match(r"^[A-Z][A-Z0-9_]+$", p)]

APPENDIX_NAPKIN_OVERRIDES: dict[str, dict[str, str]] = {
    "gpt3_accelerators": {
        "target_source": "mlsysim.Models.Language.GPT3.training_accelerators_ref",
        "current_source": "mlsysim",
    },
    "bytes_per_param": {
        "target_source": "mlsysim.core.units.BYTES_FP16 + TrainingMemoryModel optimizer bytes",
        "current_source": "derived",
    },
    "hours_per_day": {
        "target_source": "mlsysim.core.units.HOURS_PER_DAY",
        "current_source": "mlsysim",
    },
}

def finalize_yaml(path: Path) -> bool:
    """Mark migration audit entries complete after QMD migration lands."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    changed = False
    if data.get("status") != "migrated":
        data["status"] = "migrated"
        changed = True
    for cell in data.get("python_cells") or []:
        for entry in cell.get("constants") or []:
            name = str(entry.get("name", ""))
            if name in APPENDIX_NAPKIN_OVERRIDES:
                for key, val in APPENDIX_NAPKIN_OVERRIDES[name].items():
                    if entry.get(key) != val:
                        entry[key] = val
                        changed = True
            if entry.get("should_change") is not False:
                entry["should_change"] = False
                changed = True
    if changed:
        path.write_text(
            yaml.dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
    return changed

def refresh_yaml(path: Path, mapping: dict[str, str]) -> bool:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    changed = False
    if data.get("status") != "migrated":
        data["status"] = "migrated"
        changed = True
    for cell in data.get("python_cells") or []:
        for entry in cell.get("constants") or []:
            tokens = _tokens_from_value(str(entry.get("value", "")))
            targets = []
            for tok in tokens:
                if tok in mapping:
                    targets.append(_fmt_target(mapping[tok]))
            if targets:
                new_target = ", ".join(dict.fromkeys(targets))
                if entry.get("target_source") != new_target:
                    entry["target_source"] = new_target
                    changed = True
                if entry.get("should_change") is not False:
                    entry["should_change"] = False
                    changed = True
            elif str(entry.get("target_source", "")).startswith("mlsysim.core.units"):
                # Generic constants.py pointer with no mapped symbol — clarify physics-only.
                note = "mlsysim.core.units (physics/units) or mlsysim.physics"
                if entry.get("target_source") != note:
                    entry["target_source"] = note
                    changed = True
    if changed:
        path.write_text(
            yaml.dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
    return changed

def main() -> int:
    finalize = "--finalize" in sys.argv
    mapping = _load_merged_mapping()
    updated = 0
    for path in sorted(YAML_DIR.glob("*.yaml")):
        changed = refresh_yaml(path, mapping)
        if finalize:
            changed = finalize_yaml(path) or changed
        if changed:
            updated += 1
            print(f"updated {path.name}")
    print(f"Done: {updated}/{len(list(YAML_DIR.glob('*.yaml')))} files changed")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
