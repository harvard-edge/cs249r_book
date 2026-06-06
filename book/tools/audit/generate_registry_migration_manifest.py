#!/usr/bin/env python3
"""Generate dead_constants_verified.json and registry_migration_manifest.json.

Scans mlsysim.core.units for defined symbols, counts repo-wide references,
and assigns migration targets using scripts/map_constants.py plus heuristics.

Usage:
    python3 book/tools/audit/generate_registry_migration_manifest.py
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONSTANTS_PATH = REPO_ROOT / "mlsysim" / "mlsysim" / "core" / "constants.py"
MAP_CONSTANTS_PATH = REPO_ROOT / "scripts" / "map_constants.py"
OUT_DIR = REPO_ROOT / "book" / "tools" / "audit" / "artifacts"
DEAD_OUT = OUT_DIR / "dead_constants_verified.json"
MANIFEST_OUT = OUT_DIR / "registry_migration_manifest.json"

# Physics / units symbols that stay in constants.py (Phase 5 allowlist seed).
PHYSICS_KEEP = {
    "ureg",
    "Q_",
    "USD",
    "TFLOPs",
    "TOPS",
    "PFLOPs",
    "GiB",
    "KiB",
    "MiB",
    "TiB",
    "PiB",
    "Gbps",
    "MB",
    "GB",
    "TB",
    "PB",
    "byte",
    "bytes",
    "count",
    "param",
    "watt",
    "hour",
    "joule",
    "ms",
    "NS",
    "US",
    "day",
    "second",
    "PRECISION_MAP",
    "BYTES_FP32",
    "BYTES_FP16",
    "BYTES_INT8",
    "BYTES_INT4",
    "LATENCY_REGISTER_REF",
    "LATENCY_L1_REGISTER",
    "LATENCY_L2_CACHE",
    "LATENCY_HBM3",
    "LATENCY_NVLINK",
    "LATENCY_PCIE_GEN5",
    "LATENCY_INFINIBAND",
    "LATENCY_NVME_SSD",
    "ENERGY_DRAM_ACCESS_PJ",
    "ENERGY_DRAM_PJ_PER_BYTE",
    "ENERGY_FLOP_FP32_PJ",
    "ENERGY_FLOP_FP16_PJ",
    "ENERGY_OP_INT8_PJ",
    "ENERGY_FLOP_PJ",
    "ENERGY_SRAM_L1_PJ",
    "ENERGY_SRAM_L2_PJ",
    "ENERGY_REG_PJ",
    "ENERGY_ADD_FP32_PJ",
    "ENERGY_ADD_FP16_PJ",
    "ENERGY_ADD_INT32_PJ",
    "ENERGY_ADD_INT8_PJ",
    "NETWORK_ENERGY_1KB_PJ",
    "SPEED_OF_LIGHT_FIBER_KM_S",
    "TRANSFORMER_DECODE_FLOPS_PER_PARAM",
    "TRANSFORMER_TRAINING_FLOPS_PER_PARAM_TOKEN",
    "TRANSFORMER_HIDDEN_DIM_EXAMPLE",
    "TRANSFORMER_SEQ_LEN_EXAMPLE",
    "TRANSFORMER_HEADS_EXAMPLE",
    "SYSTOLIC_ARRAY_DIM",
    "SIMD_REGISTER_BITS",
    "FP32_BITS",
    "INT8_BITS",
    "KG_PER_METRIC_TON",
    "IMAGE_CHANNELS_RGB",
    "COLOR_DEPTH_8BIT",
    "VIDEO_BYTES_PER_PIXEL_RGB",
    "VIDEO_FPS_STANDARD",
}

# Heuristic migration buckets for symbols not in map_constants.py.
HEURISTIC_PREFIXES: list[tuple[str, str, str]] = [
    ("GPT", "Models.Language", "tier_c"),
    ("LLAMA", "Models.Language", "tier_c"),
    ("BERT", "Models.Language", "tier_c"),
    ("RESNET", "Models.Vision", "tier_c"),
    ("MOBILENET", "Models.Vision", "tier_c"),
    ("ALEXNET", "Models.Vision", "tier_c"),
    ("YOLO", "Models.Vision", "tier_c"),
    ("MAMBA", "Models.StateSpace", "tier_c"),
    ("DLRM", "Models.Recommendation", "tier_c"),
    ("STABLE_DIFFUSION", "Models.GenerativeVision", "tier_c"),
    ("IMAGENET", "Datasets.ImageNet", "tier_c"),
    ("CIFAR", "Datasets.CIFAR10", "tier_c"),
    ("MNIST", "Datasets.MNIST", "tier_c"),
    ("V100_", "Hardware.Cloud.V100", "tier_a"),
    ("A100_", "Hardware.Cloud.A100", "tier_a"),
    ("H100_", "Hardware.Cloud.H100", "tier_a"),
    ("H200_", "Hardware.Cloud.H200", "tier_a"),
    ("B200_", "Hardware.Cloud.B200", "tier_a"),
    ("MI300X_", "Hardware.Cloud.MI300X", "tier_a"),
    ("MI250X_", "Hardware.Cloud.MI250X", "tier_a"),
    ("NVL72_", "Hardware.Cloud.GB200_NVL72", "tier_a"),
    ("TPUV", "Hardware.Cloud", "tier_a"),
    ("WSE", "Hardware.Cloud", "tier_a"),
    ("NVLINK_", "Hardware.Cloud", "tier_a"),
    ("PCIE_", "Hardware.Cloud", "tier_a"),
    ("INFINIBAND_", "Hardware.Networks", "tier_b"),
    ("NETWORK_", "Hardware.Networks", "tier_b"),
    ("ETHERNET_", "Hardware.Networks", "tier_b"),
    ("FABRIC_", "Hardware.Networks", "tier_b"),
    ("SWITCH_", "Hardware.Networks", "tier_b"),
    ("OPTICS_", "Hardware.Networks", "tier_b"),
    ("FEC_", "Hardware.Networks", "tier_b"),
    ("CLOUD_", "Infrastructure.Pricing", "tier_d"),
    ("STORAGE_", "Infrastructure.Pricing", "tier_d"),
    ("LABELING_", "Infrastructure.Pricing", "tier_d"),
    ("FLEET_", "Infrastructure.Pricing", "tier_d"),
    ("TPU_V4_PER_HOUR", "Infrastructure.Pricing", "tier_d"),
    ("CARBON_", "Infrastructure.Grids", "tier_d"),
    ("LEAD_TIME_", "Infrastructure.Grids", "tier_d"),
    ("GRID_", "Infrastructure.Grids", "tier_d"),
    ("CLOUD_LATENCY", "Platforms.Cloud", "platforms"),
    ("EDGE_LATENCY", "Platforms.Edge", "platforms"),
    ("MOBILE_LATENCY", "Platforms.Mobile", "platforms"),
    ("TINY_LATENCY", "Platforms.Tiny", "platforms"),
    ("MOBILE_RAM", "Platforms.Mobile", "platforms"),
    ("MOBILE_STORAGE", "Platforms.Mobile", "platforms"),
    ("MOBILE_TDP", "Platforms.Mobile", "platforms"),
    ("CLOUD_MEM", "Platforms.Cloud", "platforms"),
    ("MOBILE_MEM", "Platforms.Mobile", "platforms"),
    ("TINY_MEM", "Platforms.Tiny", "platforms"),
    ("SMARTPHONE_RAM", "Platforms.Mobile", "platforms"),
    ("MCU_RAM", "Platforms.Tiny", "platforms"),
    ("DGX_", "Hardware.Edge", "tier_a"),
    ("ESP32_", "Hardware.Tiny", "tier_a"),
    ("JETSON_", "Hardware.Edge", "tier_a"),
    ("MOBILE_", "Hardware.Mobile", "tier_a"),
    ("SGX_", "Hardware.Cloud", "tier_a"),
    ("DAM_", "Ops.Monitoring", "tier_d"),
    ("MEMORY_BIT_ERROR", "Ops.Monitoring", "tier_d"),
    ("PSI_", "Ops.Monitoring", "tier_d"),
    ("KS_TEST", "Ops.Monitoring", "tier_d"),
]

SCAN_SUFFIXES = {".py", ".qmd", ".md", ".yaml", ".yml", ".tex", ".ipynb"}
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
    "artifacts",
    "_build",
    ".quarto",
}

@dataclass
class SymbolRecord:
    name: str
    defined_in: str
    total_refs: int = 0
    refs_outside_definition: int = 0
    chapters: dict[str, int] = field(default_factory=dict)
    replacement: str | None = None
    tier: str = "unknown"
    action: str = "migrate"  # migrate | keep | delete_dead | inventory_only
    notes: str = ""

def load_map_constants() -> dict[str, str]:
    if not MAP_CONSTANTS_PATH.exists():
        return {}
    spec = importlib.util.spec_from_file_location("map_constants", MAP_CONSTANTS_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(getattr(mod, "mapping", {}))

def parse_constants_symbols() -> set[str]:
    text = CONSTANTS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.isupper():
                names.add(node.target.id)
    return names

def git_tracked_files() -> list[Path]:
    out = subprocess.check_output(
        ["git", "ls-files"], cwd=REPO_ROOT, text=True
    )
    files = []
    for line in out.splitlines():
        p = REPO_ROOT / line
        if p.suffix in SCAN_SUFFIXES and p.is_file():
            parts = set(p.parts)
            if parts & SKIP_DIRS:
                continue
            files.append(p)
    return files

def count_symbol_refs(files: list[Path], symbols: set[str]) -> dict[str, dict[str, int]]:
    per_file: dict[str, dict[str, int]] = {s: defaultdict(int) for s in symbols}
    patterns = {s: re.compile(rf"\b{re.escape(s)}\b") for s in symbols}
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(path.relative_to(REPO_ROOT))
        for sym, pat in patterns.items():
            n = len(pat.findall(text))
            if n:
                per_file[sym][rel] += n
    return per_file

def chapter_key(rel_path: str) -> str | None:
    m = re.search(r"book/quarto/contents/(vol[12]/[^/]+)/", rel_path)
    return m.group(1) if m else None

def infer_replacement(name: str, mapping: dict[str, str]) -> tuple[str | None, str, str]:
    if name in mapping:
        return mapping[name], "map_constants", "mapped"
    if name in PHYSICS_KEEP:
        return None, "keep", "physics_allowlist"
    for prefix, target, tier in HEURISTIC_PREFIXES:
        if name.startswith(prefix) or name == prefix.rstrip("_"):
            return f"{target}.*", tier, "heuristic_prefix"
    if name.endswith("_THRESHOLD") or name.endswith("_EXAMPLE"):
        return "Ops.Monitoring.*", "tier_d", "heuristic_suffix"
    return None, "unknown", "needs_manual_mapping"

def build_records(
    symbols: set[str],
    per_file_refs: dict[str, dict[str, int]],
    mapping: dict[str, str],
) -> list[SymbolRecord]:
    records: list[SymbolRecord] = []
    for sym in sorted(symbols):
        rec = SymbolRecord(name=sym, defined_in=str(CONSTANTS_PATH.relative_to(REPO_ROOT)))
        file_refs = per_file_refs.get(sym, {})
        rec.total_refs = sum(file_refs.values())
        outside = {
            f: c
            for f, c in file_refs.items()
            if f != rec.defined_in and not f.endswith(("defaults.py", "calibration.py"))
        }
        rec.refs_outside_definition = sum(outside.values())
        for f, c in outside.items():
            ch = chapter_key(f)
            if ch:
                rec.chapters[ch] = rec.chapters.get(ch, 0) + c
        replacement, tier, note = infer_replacement(sym, mapping)
        rec.replacement = replacement
        rec.tier = tier
        rec.notes = note
        if sym in PHYSICS_KEEP or tier == "keep":
            rec.action = "keep"
        elif rec.refs_outside_definition == 0:
            rec.action = "delete_dead"
        elif tier == "unknown" and rec.refs_outside_definition <= 1:
            rec.action = "inventory_only"
        else:
            rec.action = "migrate"
        records.append(rec)
    return records

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = parse_constants_symbols()
    mapping = load_map_constants()
    files = git_tracked_files()
    per_file_refs = count_symbol_refs(files, symbols)
    records = build_records(symbols, per_file_refs, mapping)

    dead = [r for r in records if r.action == "delete_dead"]
    manifest = {
        "generated_by": "book/tools/audit/generate_registry_migration_manifest.py",
        "constants_path": str(CONSTANTS_PATH.relative_to(REPO_ROOT)),
        "symbol_count": len(records),
        "dead_count": len(dead),
        "migrate_count": sum(1 for r in records if r.action == "migrate"),
        "keep_count": sum(1 for r in records if r.action == "keep"),
        "symbols": [asdict(r) for r in records],
    }
    dead_payload = {
        "verified": True,
        "count": len(dead),
        "symbols": [r.name for r in dead],
        "details": [asdict(r) for r in dead],
    }

    DEAD_OUT.write_text(json.dumps(dead_payload, indent=2) + "\n", encoding="utf-8")
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {DEAD_OUT.relative_to(REPO_ROOT)} ({len(dead)} dead symbols)")
    print(f"Wrote {MANIFEST_OUT.relative_to(REPO_ROOT)} ({len(records)} total symbols)")
    unknown = [r.name for r in records if r.tier == "unknown" and r.action == "migrate"]
    if unknown:
        print(f"WARNING: {len(unknown)} migrate symbols need manual mapping", file=sys.stderr)
        for name in unknown[:20]:
            print(f"  - {name}", file=sys.stderr)
        if len(unknown) > 20:
            print(f"  ... and {len(unknown) - 20} more", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
