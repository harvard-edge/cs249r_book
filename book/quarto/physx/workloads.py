"""
workloads.py
Lighthouse models and canonical workloads.
"""

from .constants import (
    RESNET50_PARAMS,
    RESNET50_FLOPs,
    GPT2_PARAMS,
    GPT3_PARAMS,
    DLRM_MODEL_SIZE_FP32,
    MOBILENETV2_PARAMS,
    MOBILENETV2_FLOPs,
    KWS_DSCNN_PARAMS,
    KWS_DSCNN_FLOPs,
)
from .archetypes import ARCHETYPES


WORKLOADS = {
    "resnet50": {
        "label": "ResNet-50",
        "archetype": ARCHETYPES["compute_beast"]["label"],
        "params": RESNET50_PARAMS,
        "flops": RESNET50_FLOPs,
        "notes": "Dense compute-bound CNN.",
    },
    "gpt2": {
        "label": "GPT-2",
        "archetype": ARCHETYPES["bandwidth_hog"]["label"],
        "params": GPT2_PARAMS,
        "notes": "Autoregressive LLM serving.",
    },
    "gpt3": {
        "label": "GPT-3",
        "archetype": ARCHETYPES["bandwidth_hog"]["label"],
        "params": GPT3_PARAMS,
        "notes": "Scaled LLM (training and inference).",
    },
    "dlrm": {
        "label": "DLRM",
        "archetype": ARCHETYPES["sparse_scatter"]["label"],
        "model_size_fp32": DLRM_MODEL_SIZE_FP32,
        "notes": "Embedding-heavy recommender.",
    },
    "mobilenetv2": {
        "label": "MobileNetV2",
        "archetype": ARCHETYPES["compute_beast"]["label"],
        "params": MOBILENETV2_PARAMS,
        "flops": MOBILENETV2_FLOPs,
        "notes": "Efficient compute-bound model.",
    },
    "kws_dscnn": {
        "label": "Keyword Spotting (DS-CNN)",
        "archetype": ARCHETYPES["tiny_constraint"]["label"],
        "params": KWS_DSCNN_PARAMS,
        "flops": KWS_DSCNN_FLOPs,
        "notes": "Always-on TinyML workload.",
    },
}
