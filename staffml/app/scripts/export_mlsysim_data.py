#!/usr/bin/env python3
"""Export the MLSysIM numbers StaffML displays into src/data/mlsysim.json.

StaffML is a static site, so it cannot import MLSysIM at runtime. This script
reads the MLSysIM registries straight from the repo (folder-based import, no
pip install of mlsysim) and writes the hardware reference, interconnect,
latency, and simulator model tables that src/lib/hardware.ts serves.

Every StaffML build in CI (dev validate, dev preview, live publish) runs this
before `npm run build`, so a deployed site always carries the current MLSysIM
values. The committed JSON keeps local `npm run dev` working without Python.

Usage (from the repo root or anywhere):
    python3 staffml/app/scripts/export_mlsysim_data.py           # write
    python3 staffml/app/scripts/export_mlsysim_data.py --check   # fail if stale
    python3 staffml/app/scripts/export_mlsysim_data.py --print-deps

History: 2026-10-27, replaces a hand-copied table in hardware.ts that cited a
deleted mlsysim/core/constants.py and had drifted on four devices.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MLSYSIM_ROOT = REPO / "mlsysim"
OUT = REPO / "staffml" / "app" / "src" / "data" / "mlsysim.json"

# Which registry entries StaffML shows, in display order. Display names, tiers,
# and memory-type labels are presentation; every number comes from MLSysIM.
# precision: the precision_flops key to show, or "peak" for peak_flops.
HARDWARE = [
    ("NVIDIA H100 SXM", "cloud", "Cloud.H100", "fp16_dense", "FP16", "HBM3"),
    ("NVIDIA H200", "cloud", "Cloud.H200", "peak", "FP16", "HBM3e"),
    ("NVIDIA B200", "cloud", "Cloud.B200", "fp16_dense", "FP16", "HBM3e"),
    ("NVIDIA A100 80GB", "cloud", "Cloud.A100", "fp16_dense", "FP16", "HBM2e"),
    ("AMD MI300X", "cloud", "Cloud.MI300X", "peak", "FP16", "HBM3"),
    ("Google TPU v5p", "cloud", "Cloud.TPUv5p", "peak", "BF16", "HBM"),
    ("Jetson Orin NX", "edge", "Edge.JetsonOrinNX", "int8", "INT8", "LPDDR5"),
    ("Google Coral Edge TPU", "edge", "Edge.Coral", "peak", "INT8", "DRAM"),
    ("Apple A17 Pro", "mobile", "Mobile.iPhone15Pro", "peak", "INT8", "LPDDR5"),
    ("Snapdragon 8 Gen 3", "mobile", "Mobile.Snapdragon8Gen3", "peak", "INT8", "LPDDR5X"),
    ("ESP32-S3", "tinyml", "Tiny.ESP32_S3", "int8", "INT8", "Flash + SRAM"),
]

# (display name, source) for the interconnect table. A "nvlink:" source reads
# a GPU's NVLink; "pcie:" its host interface; "fabric:" a Systems.Fabrics entry.
INTERCONNECTS = [
    ("NVLink H100", "nvlink:Cloud.H100"),
    ("NVLink B200", "nvlink:Cloud.B200"),
    ("PCIe Gen5 x16", "pcie:Cloud.H100"),
    ("InfiniBand NDR", "fabric:InfiniBand_NDR"),
    ("InfiniBand XDR", "fabric:InfiniBand_XDR"),
    ("Ethernet 100GbE", "fabric:Ethernet_100G"),
]

# (display name, Tech path or fabric) for the latency ladder.
LATENCY = [
    ("L1 Cache / Register", "Memory.L1"),
    ("L2 Cache", "Memory.L2"),
    ("HBM Access (Hopper)", "Memory.HopperGlobalMemory"),
    ("NVLink Transfer", "Interconnect.NVLink"),
    ("PCIe Gen5", "Interconnect.PCIeGen5"),
    ("InfiniBand NDR", "fabric:InfiniBand_NDR"),
    ("NVMe SSD Read", "Storage.NvmeGen4"),
]

MODELS = [
    ("GPT-2 (1.5B)", "Language.GPT2"),
    ("Llama-2-7B", "Language.Llama2_7B"),
    ("Llama-2-70B", "Language.Llama2_70B"),
    ("Llama-3.1-405B", "Language.Llama3_405B"),
    ("BERT-Large (340M)", "Language.BERT_Large"),
]


def mlsysim_deps() -> list[str]:
    meta = tomllib.loads((MLSYSIM_ROOT / "pyproject.toml").read_text())
    return list(meta["project"]["dependencies"])


def walk(root, dotted: str):
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def num(value: float) -> float | int:
    """Round away float noise so the JSON diff stays readable."""
    value = float(f"{value:.6g}")
    return int(value) if value.is_integer() else value


def build() -> dict:
    sys.path.insert(0, str(MLSYSIM_ROOT))
    try:
        from mlsysim.hardware.registry import Hardware
        from mlsysim.models.registry import Models
        from mlsysim.systems.registry import Systems
    except ImportError as exc:  # pragma: no cover - environment problem
        sys.exit(
            f"Cannot import MLSysIM from {MLSYSIM_ROOT}: {exc}\n"
            "Install its dependencies first:\n"
            "  python3 -m pip install $(python3 staffml/app/scripts/export_mlsysim_data.py --print-deps)"
        )

    hardware = []
    for name, tier, ref, precision, unit, memory_type in HARDWARE:
        hw = walk(Hardware, ref)
        rate = hw.compute.peak_flops if precision == "peak" else hw.compute.precision_flops[precision]
        is_ops = "FLOP" not in str(rate.units)
        if is_ops != (unit == "INT8"):
            sys.exit(f"{ref}: {precision} is {rate.units}, which does not match the {unit} label")
        tera = rate.to("TOPS" if is_ops else "TFLOPs/s").magnitude
        hardware.append({
            "name": name,
            "tier": tier,
            "compute_tflops": num(tera),
            "compute_unit": unit,
            "bandwidth_tbs": num(hw.memory.bandwidth.to("TB/s").magnitude),
            # Vendors quote HBM/LPDDR capacity in binary units ("80 GB" is
            # 80 GiB), so this field carries the GiB value under the GB label.
            "memory_gb": num(hw.memory.capacity.to("GiB").magnitude),
            "memory_type": memory_type,
            "tdp_w": num(hw.tdp.to("W").magnitude),
            "year": int(hw.release_year),
            "mlsysim": f"Hardware.{ref}",
        })

    interconnects = []
    for name, source in INTERCONNECTS:
        kind, ref = source.split(":")
        if kind == "fabric":
            link = getattr(Systems.Fabrics, ref)
            bw, lat = link.bandwidth, link.latency
        else:
            hw = walk(Hardware, ref)
            link = hw.nvlink if kind == "nvlink" else hw.interconnect
            bw = link.bandwidth
            lat = link.latency if link.latency is not None else Hardware.Tech.Interconnect.PCIeGen5.latency
        interconnects.append({
            "name": name,
            "bandwidth_gbs": num(bw.to("GB/s").magnitude),
            "latency_us": num(lat.to("us").magnitude),
            "mlsysim": f"{'Systems.Fabrics' if kind == 'fabric' else 'Hardware'}.{ref}",
        })

    latency = []
    for name, ref in LATENCY:
        if ref.startswith("fabric:"):
            q = getattr(Systems.Fabrics, ref.split(":")[1]).latency
        else:
            q = walk(Hardware.Tech, ref).latency
        latency.append({"operation": name, "latency_ns": num(q.to("ns").magnitude), "mlsysim": ref})

    models = []
    for name, ref in MODELS:
        m = walk(Models, ref)
        params = m.parameters.to("param").magnitude
        models.append({
            "name": name,
            "params_b": num(params / 1e9),
            "layers": int(m.layers),
            "hidden_dim": int(m.hidden_dim),
            "heads": int(m.heads),
            # Training cost per token: 2N forward plus 4N backward.
            "flops_per_token": num(6 * params),
            "mlsysim": f"Models.{ref}",
        })

    return {
        "_generated_by": "staffml/app/scripts/export_mlsysim_data.py (do not edit by hand)",
        "hardware": hardware,
        "interconnects": interconnects,
        "latency": latency,
        "models": models,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if the committed JSON differs from MLSysIM")
    ap.add_argument("--print-deps", action="store_true", help="print MLSysIM's pip requirements and exit")
    args = ap.parse_args()

    if args.print_deps:
        print(" ".join(mlsysim_deps()))
        return 0

    text = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        if current == text:
            print(f"✓ {OUT.relative_to(REPO)} matches MLSysIM")
            return 0
        sys.stdout.writelines(difflib.unified_diff(
            current.splitlines(keepends=True), text.splitlines(keepends=True),
            fromfile="committed", tofile="mlsysim"))
        print(f"\n✗ {OUT.relative_to(REPO)} is out of date with MLSysIM. Regenerate it with:\n"
              "  python3 staffml/app/scripts/export_mlsysim_data.py")
        return 1

    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
