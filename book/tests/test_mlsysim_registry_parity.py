"""Parity tests: registry replacements must match legacy constants before deletion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlsysim import Hardware, Systems, Datasets, Models
from mlsysim.core import constants as legacy

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPO_ROOT / "book" / "tools" / "audit" / "artifacts" / "registry_migration_manifest.json"
)

# (constant_name, registry_value) — add pairs as symbols are hard-deleted from constants.py.
PARITY_CASES: list[tuple[str, object]] = [
    ("NVLINK_V100_BW", Hardware.Cloud.V100.nvlink.bandwidth),
    ("NVLINK_A100_BW", Hardware.Cloud.A100.nvlink.bandwidth),
    ("NVLINK_H100_BW", Hardware.Cloud.H100.nvlink.bandwidth),
    ("NVLINK_B200_BW", Hardware.Cloud.B200.nvlink.bandwidth),
    ("PCIE_GEN3_BW", Hardware.Cloud.V100.interconnect.bandwidth),
    ("PCIE_GEN4_BW", Hardware.Cloud.A100.interconnect.bandwidth),
    ("PCIE_GEN5_BW", Hardware.Cloud.H100.interconnect.bandwidth),
    ("INFINIBAND_HDR_BW", Systems.Fabrics.InfiniBand_HDR.bandwidth),
    ("INFINIBAND_NDR_BW", Systems.Fabrics.InfiniBand_NDR.bandwidth),
    ("INFINIBAND_XDR_BW", Systems.Fabrics.InfiniBand_XDR.bandwidth),
    ("INFINIBAND_GXDR_BW", Systems.Fabrics.InfiniBand_GXDR.bandwidth),
    ("IMAGENET_IMAGES", Datasets.ImageNet.training_examples),
    ("IMAGENET_NUM_CLASSES", Datasets.ImageNet.num_classes),
    ("MNIST_TRAINING_EXAMPLES", Datasets.MNIST.training_examples),
    ("GPT3_TRAINING_TOKENS", Models.Language.GPT3.training_tokens),
    ("H100_L2_CACHE", Hardware.Cloud.H100.memory.l2_cache),
    ("BERT_BASE_FLOPs", Models.Language.BERT_Base.inference_flops),
    ("BERT_LARGE_FLOPs", Models.Language.BERT_Large.inference_flops),
    ("ALEXNET_FLOPs", Models.Vision.AlexNet.inference_flops),
    ("RESNET18_PARAMS", Models.Vision.ResNet18.parameters),
    ("YOLOV8_NANO_FLOPs", Models.Vision.YOLOv8_Nano.inference_flops),
    ("WAKEVISION_FLOPs", Models.Tiny.WakeVision.inference_flops),
    ("RESNET50_FLOPs", Models.Vision.ResNet50.inference_flops),
    ("MOBILENETV2_FLOPs", Models.Vision.MobileNetV2.inference_flops),
    ("KWS_DSCNN_FLOPs", Models.Tiny.DS_CNN.inference_flops),
    ("STABLE_DIFFUSION_V1_5_FLOPs_PER_STEP", Models.GenerativeVision.StableDiffusion_v1_5.inference_flops),
    ("DLRM_EMBEDDING_ENTRIES", Models.Recommendation.DLRM.embedding_entries),
    ("TPUV5P_ICI_BW", Hardware.Cloud.TPUv5p.nvlink.bandwidth),
    ("TPUV1_TOPS_INT8", Hardware.Cloud.TPUv1.compute.precision_flops["int8"]),
    ("TPUV1_TDP", Hardware.Cloud.TPUv1.tdp),
]


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        pytest.skip(f"missing manifest: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("constant_name,registry_val", PARITY_CASES)
def test_registry_parity(constant_name: str, registry_val: object) -> None:
    if not hasattr(legacy, constant_name):
        pytest.skip(f"{constant_name} already removed from constants.py")
    legacy_val = getattr(legacy, constant_name)
    assert legacy_val == registry_val


def test_manifest_present() -> None:
    manifest = _load_manifest()
    assert manifest["symbol_count"] > 0
    assert "dead_count" in manifest
