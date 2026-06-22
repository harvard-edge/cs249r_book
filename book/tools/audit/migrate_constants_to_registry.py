#!/usr/bin/env python3
"""Replace legacy mlsysim.core.units hardware symbols with registry paths in QMD cells."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MAP_PATH = REPO_ROOT / "scripts" / "map_constants.py"

CELL_OPEN = re.compile(r"^```\{python\}")
CELL_CLOSE = re.compile(r"^```\s*$")

INTERCONNECT_MAP = {
    "NVLINK_V100_BW": "Hardware.Cloud.V100.nvlink.bandwidth",
    "NVLINK_A100_BW": "Hardware.Cloud.A100.nvlink.bandwidth",
    "NVLINK_H100_BW": "Hardware.Cloud.H100.nvlink.bandwidth",
    "NVLINK_B200_BW": "Hardware.Cloud.B200.nvlink.bandwidth",
    "PCIE_GEN3_BW": "Hardware.Cloud.V100.interconnect.bandwidth",
    "PCIE_GEN4_BW": "Hardware.Cloud.A100.interconnect.bandwidth",
    "PCIE_GEN5_BW": "Hardware.Cloud.H100.interconnect.bandwidth",
    "INFINIBAND_HDR_BW": "Systems.Fabrics.InfiniBand_HDR.bandwidth",
    "INFINIBAND_NDR_BW": "Systems.Fabrics.InfiniBand_NDR.bandwidth",
    "INFINIBAND_XDR_BW": "Systems.Fabrics.InfiniBand_XDR.bandwidth",
    "INFINIBAND_GXDR_BW": "Systems.Fabrics.InfiniBand_GXDR.bandwidth",
    "H100_FLOPS_FP32_CUDA": 'Hardware.Cloud.H100.compute.precision_flops["fp32"]',
    "CPU_FLOPS_FP32": "Hardware.Cloud.ReferenceCPU.compute.peak_flops",
    "H100_L2_CACHE": "Hardware.Cloud.H100.memory.l2_cache",
    "TPUV5P_L2_SRAM": "Hardware.Cloud.TPUv5p.memory.l2_cache",
    "SGX_EPC_CAPACITY": "Hardware.Cloud.IntelSGX.memory.capacity",
    "SGX_BASE_LATENCY": "Hardware.Cloud.IntelSGX.dispatch_tax",
    "SGX_OVERFLOW_LATENCY": "Hardware.Cloud.IntelSGX.dispatch_tax",
}

HARDWARE_MAP = {
    "A100_FLOPS_FP16_SPARSE": 'Hardware.Cloud.A100.compute.precision_flops["fp16_sparse"]',
    "TPUV5P_ICI_BW": "Hardware.Cloud.TPUv5p.nvlink.bandwidth",
    "TPUV1_TOPS_INT8": 'Hardware.Cloud.TPUv1.compute.precision_flops["int8"]',
    "TPUV1_TDP": "Hardware.Cloud.TPUv1.tdp",
    "JETSON_AGX_ORIN_TOPS_INT8": 'Hardware.Edge.JetsonAGXOrin.compute.precision_flops["int8"]',
    "JETSON_AGX_ORIN_TDP_MIN": "Hardware.Edge.JetsonAGXOrin.tdp_min",
    "JETSON_AGX_ORIN_TDP_MAX": "Hardware.Edge.JetsonAGXOrin.tdp_max",
    "ESP32_RAM": "Hardware.Tiny.ESP32_S3.memory.sram_capacity",
    "ESP32_FLASH": "Hardware.Tiny.ESP32_S3.memory.flash_capacity",
    "ESP32_POWER_MIN": "Hardware.Tiny.ESP32_S3.tdp_min",
    "ESP32_POWER_MAX": "Hardware.Tiny.ESP32_S3.tdp_max",
    "ESP32_PRICE": "10",
    "DGX_RAM": "Hardware.Workstation.DGX_Spark.memory.capacity",
    "DGX_STORAGE": "Hardware.Workstation.DGX_Spark.storage.capacity",
    "DGX_POWER": "Hardware.Workstation.DGX_Spark.tdp",
    "DGX_PRICE_MIN": "Hardware.Workstation.DGX_Spark.unit_cost",
    "DGX_PRICE_MAX": "Hardware.Workstation.DGX_Spark.unit_cost_max",
    "STABLE_DIFFUSION_V1_5_FLOPs_PER_STEP": "Models.GenerativeVision.StableDiffusion_v1_5.inference_flops",
    "DLRM_EMBEDDING_ENTRIES": "Models.Recommendation.DLRM.embedding_entries",
}

def load_map_constants() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("map_constants", MAP_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(getattr(mod, "mapping", {}))

_MAPPING_CACHE: dict[str, str] | None = None

DEFAULTS_MAP = {
    "CLOUD_EGRESS_PER_GB": "Infrastructure.Pricing.Cloud.EgressPerGB.rate",
    "CLOUD_ELECTRICITY_PER_KWH": "Infrastructure.Pricing.Cloud.ElectricityPerKwh.rate",
    "CLOUD_GPU_TRAINING_PER_HOUR": "Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate",
    "CLOUD_GPU_INFERENCE_PER_HOUR": "Infrastructure.Pricing.Cloud.GpuInferencePerHour.rate",
    "TPU_V4_PER_HOUR": "Infrastructure.Pricing.Cloud.TpuV4PerHour.rate",
    "FLEET_GPU_HOUR_COST_REF": "Infrastructure.Pricing.Fleet.GpuHourRef.rate",
    "FLEET_SPOT_GPU_HOUR_COST_REF": "Infrastructure.Pricing.Fleet.SpotGpuHourRef.rate",
    "FLEET_INTERNAL_CHARGEBACK_PER_HOUR": "Infrastructure.Pricing.Fleet.InternalChargebackPerHour.rate",
    "CARBON_PER_GPU_HR_KG": "Infrastructure.Pricing.Fleet.CarbonPerGpuHr.rate",
    "STORAGE_COST_S3_STD": "Infrastructure.Pricing.Storage.S3StandardPerTbMonth.rate",
    "STORAGE_COST_GLACIER": "Infrastructure.Pricing.Storage.GlacierPerTbMonth.rate",
    "STORAGE_COST_NVME_LOW": "Infrastructure.Pricing.Storage.NvmeLowPerTbMonth.rate",
    "STORAGE_COST_NVME_HIGH": "Infrastructure.Pricing.Storage.NvmeHighPerTbMonth.rate",
    "RETRIEVAL_COST_GLACIER": "Infrastructure.Pricing.Storage.GlacierRetrievalPerGB.rate",
    "LABELING_COST_CROWD_LOW": "Infrastructure.Pricing.Labeling.CrowdLow.rate",
    "LABELING_COST_CROWD_HIGH": "Infrastructure.Pricing.Labeling.CrowdHigh.rate",
    "LABELING_COST_BOX_LOW": "Infrastructure.Pricing.Labeling.BoxLow.rate",
    "LABELING_COST_BOX_HIGH": "Infrastructure.Pricing.Labeling.BoxHigh.rate",
    "LABELING_COST_MEDICAL_LOW": "Infrastructure.Pricing.Labeling.MedicalLow.rate",
    "LABELING_COST_MEDICAL_HIGH": "Infrastructure.Pricing.Labeling.MedicalHigh.rate",
    "LEAD_TIME_GPU_MONTHS": "Infrastructure.Capacity.GpuLeadTimeMonths",
    "LEAD_TIME_SUBSTATION_MONTHS": "Infrastructure.Capacity.SubstationLeadTimeMonths",
    "GRID_INTERCONNECTION_QUEUE_US_GW": "Infrastructure.Capacity.GridInterconnectionQueueGw",
    "PUE_LIQUID_COOLED": "Infrastructure.PUE_LIQUID_COOLED",
    "PUE_BEST_AIR": "Infrastructure.PUE_BEST_AIR",
    "PUE_TYPICAL": "Infrastructure.PUE_TYPICAL",
    "PUE_LEGACY": "Infrastructure.PUE_LEGACY",
    "CARBON_US_AVG_GCO2_KWH": "Infrastructure.Grids.US_Avg.carbon_intensity_g_kwh",
    "CARBON_QUEBEC_GCO2_KWH": "Infrastructure.Grids.Quebec.carbon_intensity_g_kwh",
    "CARBON_IOWA_GCO2_KWH": "Infrastructure.Grids.Iowa.carbon_intensity_g_kwh",
    "CARBON_POLAND_GCO2_KWH": "Infrastructure.Grids.Poland.carbon_intensity_g_kwh",
    "CARBON_NORWAY_GCO2_KWH": "Infrastructure.Grids.Norway.carbon_intensity_g_kwh",
    "CARBON_EU_AVG_GCO2_KWH": "Infrastructure.Grids.EU_Avg.carbon_intensity_g_kwh",
    "CARBON_FRANCE_GCO2_KWH": "Infrastructure.Grids.France.carbon_intensity_g_kwh",
    "MEMORY_BIT_ERROR_RATE_PER_BIT": "Monitoring.MemoryBitErrorRatePerBit",
    "KS_TEST_COEFFICIENT": "Monitoring.KsTestCoefficient",
    "PSI_WARN_THRESHOLD": "Monitoring.PsiWarnThreshold",
    "PSI_REVIEW_THRESHOLD": "Monitoring.PsiReviewThreshold",
    "PSI_CRITICAL_THRESHOLD": "Monitoring.PsiCriticalThreshold",
    "INFINIBAND_NDR_BW_GBS": "Systems.Fabrics.InfiniBand_NDR.bandwidth",
    "INFINIBAND_HDR_BW_GBS": "Systems.Fabrics.InfiniBand_HDR.bandwidth",
    "INFINIBAND_XDR_BW_GBS": "Systems.Fabrics.InfiniBand_XDR.bandwidth",
    "ETHERNET_400G_BW_GBS": "Systems.Fabrics.Ethernet_400G.bandwidth",
    "ETHERNET_800G_BW_GBS": "Systems.Fabrics.Ethernet_800G.bandwidth",
    "ROCE_100G_BW_GBS": "Systems.Fabrics.RoCE_100G.bandwidth",
    "IB_NDR_LATENCY_US": "mlsysim.systems.registry.IB_NDR_LATENCY_US",
    "IB_HDR_LATENCY_US": "mlsysim.systems.registry.IB_HDR_LATENCY_US",
    "ROCE_LATENCY_US": "mlsysim.systems.registry.ROCE_LATENCY_US",
    "TCP_LATENCY_US": "mlsysim.systems.registry.TCP_LATENCY_US",
    "TPU_POD_CHIPS": "Systems.Pods.TPUv4_4096.chips",
    "TPU_POD_MEM": "Systems.Pods.TPUv4_4096.memory",
    "TPU_POD_POWER": "Systems.Pods.TPUv4_4096.power",
}

DATASETS_MAP = {
    "IMAGENET_IMAGES": "Datasets.ImageNet.training_examples",
    "IMAGENET_TEST_IMAGES": "Datasets.ImageNet.test_examples",
    "IMAGENET_NUM_CLASSES": "Datasets.ImageNet.num_classes",
    "CIFAR10_IMAGES": "Datasets.CIFAR10.training_examples",
    "CIFAR10_TEST_IMAGES": "Datasets.CIFAR10.test_examples",
    "MNIST_IMAGE_WIDTH": "Datasets.MNIST.image_width",
    "MNIST_IMAGE_HEIGHT": "Datasets.MNIST.image_height",
    "MNIST_NUM_CLASSES": "Datasets.MNIST.num_classes",
    "MNIST_TRAINING_EXAMPLES": "Datasets.MNIST.training_examples",
}

TRAINING_MAP = {
    "GPT3_TRAINING_TOKENS": "Models.Language.GPT3.training_tokens",
    "GPT3_TRAINING_ACCELERATORS_REF": "Models.Language.GPT3.training_accelerators_ref",
    "GPT3_TRAINING_DAYS_REF": "Models.Language.GPT3.training_days_ref",
    "GPT3_TRAINING_ENERGY_MWH": "Models.Language.GPT3.training_energy_mwh",
    "GPT4_TRAINING_GPU_DAYS": "Models.Language.GPT4.training_gpu_days",
    "GPT4_CLASS_PUBLIC_ESTIMATE_GPU_COUNT_REF": "Models.Language.GPT4.training_accelerators_ref",
    "GPT4_CLASS_PUBLIC_ESTIMATE_TRAINING_DAYS_REF": "Models.Language.GPT4.training_days_ref",
    "GPT4_CLASS_PUBLIC_ESTIMATE_HARDWARE_LABEL": "Models.Language.GPT4.training_hardware_label",
    "LLAMA2_70B_KV_HEADS": "Models.Language.Llama2_70B.kv_heads",
}

def merged_mapping() -> dict[str, str]:
    global _MAPPING_CACHE
    if _MAPPING_CACHE is None:
        out = {**load_map_constants(), **INTERCONNECT_MAP, **HARDWARE_MAP, **DEFAULTS_MAP, **DATASETS_MAP, **TRAINING_MAP}
        _MAPPING_CACHE = dict(sorted(out.items(), key=lambda kv: len(kv[0]), reverse=True))
    return _MAPPING_CACHE

def substitute_cell(cell: str, mapping: dict[str, str]) -> tuple[str, list[str]]:
    replaced: list[str] = []
    out = cell
    for sym, target in mapping.items():
        cpat = re.compile(rf"\bconstants\.{re.escape(sym)}\b")
        if cpat.search(out):
            out = cpat.sub(target, out)
            replaced.append(sym)
            continue
        pat = re.compile(rf"(?<![.\w]){re.escape(sym)}\b")
        if pat.search(out):
            out = pat.sub(target, out)
            replaced.append(sym)
    return out, replaced

def ensure_registry_imports(cell: str) -> str:
    needs_hardware = "Hardware." in cell and "import Hardware" not in cell and "from mlsysim import *" not in cell
    needs_systems = "Systems." in cell and "import Systems" not in cell and "from mlsysim import *" not in cell
    needs_models = "Models." in cell and "import Models" not in cell and "from mlsysim import *" not in cell
    needs_datasets = "Datasets." in cell and "from mlsysim import *" not in cell
    prefix = []
    if needs_hardware or needs_systems or needs_models or needs_datasets:
        if "from mlsysim import *" not in cell:
            prefix.append("from mlsysim import *")
    if not prefix:
        return cell
    lines = cell.splitlines()
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("#|") or line.startswith("# ") or line.startswith("#┌"):
            insert_at = i + 1
        elif line.strip() and not line.startswith("#"):
            break
    return "\n".join(lines[:insert_at] + prefix + lines[insert_at:])

def migrate_file(path: Path, mapping: dict[str, str], dry_run: bool) -> dict:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out_lines: list[str] = []
    stats = {"cells": 0, "cells_changed": 0, "symbols": set()}
    i = 0
    while i < len(lines):
        if CELL_OPEN.match(lines[i]):
            start = i
            j = i + 1
            while j < len(lines) and not CELL_CLOSE.match(lines[j]):
                j += 1
            block = "\n".join(lines[i + 1 : j])
            new_block, replaced = substitute_cell(block, mapping)
            new_block = ensure_registry_imports(new_block)
            stats["cells"] += 1
            if replaced:
                stats["cells_changed"] += 1
                stats["symbols"].update(replaced)
            out_lines.extend(lines[i : i + 1])
            out_lines.extend(new_block.splitlines())
            out_lines.append(lines[j])
            i = j + 1
        else:
            out_lines.append(lines[i])
            i += 1
    new_text = "\n".join(out_lines) + ("\n" if text.endswith("\n") else "")
    if not dry_run and new_text != text:
        path.write_text(new_text, encoding="utf-8")
    stats["symbols"] = sorted(stats["symbols"])
    return stats

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qmd", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    path = args.qmd if args.qmd.is_absolute() else REPO_ROOT / args.qmd
    mapping = merged_mapping()
    stats = migrate_file(path, mapping, dry_run=args.dry_run)
    print(f"{path}: {stats['cells_changed']}/{stats['cells']} cells, {len(stats['symbols'])} symbols")
    for sym in stats["symbols"]:
        print(f"  - {sym}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
