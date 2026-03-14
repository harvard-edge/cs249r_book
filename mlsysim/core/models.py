# models.py
# Hierarchical Model Definitions for MLSys Textbook

import pint
from dataclasses import dataclass
from typing import Optional, Tuple, List
from .registry import Registry
from .constants import (
    ureg, Q_,
    GPT2_PARAMS, GPT3_PARAMS, GPT4_EST_PARAMS, GPT4_TRAINING_GPU_DAYS,
    LLAMA3_8B_PARAMS, LLAMA3_70B_PARAMS, LLAMA3_405B_PARAMS,
    BERT_BASE_PARAMS, BERT_LARGE_PARAMS,
    ALEXNET_PARAMS, RESNET50_PARAMS, MOBILENETV2_PARAMS,
    KWS_DSCNN_PARAMS, ANOMALY_MODEL_PARAMS,
    DLRM_MODEL_SIZE_FP32,
    BYTES_FP32, BYTES_FP16, BYTES_INT8,
    GPT3_TRAINING_OPS,
    RESNET50_FLOPs, MOBILENETV2_FLOPs, KWS_DSCNN_FLOPs
)

@dataclass(frozen=True)
class ModelSpec:
    name: str
    parameters: Q_
    architecture: str # "Transformer", "CNN", "MLP"
    layers: Optional[int] = None
    inference_flops: Optional[Q_] = None
    training_ops: Optional[Q_] = None
    training_gpu_days: Optional[float] = None
    model_size: Optional[Q_] = None # For models defined by size (DLRM)
    
    def __post_init__(self):
        """Validate model specs: correct dimension type first, then positive value."""
        from .constants import ureg
        if self.parameters is not None:
            if not self.parameters.is_compatible_with(ureg.count):
                raise pint.DimensionalityError(self.parameters.units, ureg.count,
                    extra_msg=f" — {self.name}.parameters must be in param/count units")
            if self.parameters.magnitude <= 0:
                raise ValueError(f"{self.name}: parameters must be positive.")
        if self.inference_flops is not None and not self.inference_flops.is_compatible_with(ureg.flop):
            raise pint.DimensionalityError(self.inference_flops.units, ureg.flop,
                extra_msg=f" — {self.name}.inference_flops must be in flop units")
        if self.training_ops is not None and not self.training_ops.is_compatible_with(ureg.flop):
            raise pint.DimensionalityError(self.training_ops.units, ureg.flop,
                extra_msg=f" — {self.name}.training_ops must be in flop units")
        if self.model_size is not None and not self.model_size.is_compatible_with(ureg.byte):
            raise pint.DimensionalityError(self.model_size.units, ureg.byte,
                extra_msg=f" — {self.name}.model_size must be in byte units")

    def size_in_bytes(self, precision: Q_ = BYTES_FP16) -> Q_:
        """Calculates the weight storage size for a given precision."""
        from .constants import ureg
        if self.model_size:
            return self.model_size
        param_count = self.parameters.to(ureg.count).magnitude
        bpp = precision.to(ureg.byte).magnitude
        return (param_count * bpp * ureg.byte).to(ureg.byte)

    def get_kv_cache_size(self, seq_len: int, batch_size: int, precision: Q_ = BYTES_FP16) -> Q_:
        """
        Backward-compatible approximation of KV-cache memory for transformer models.

        This is intended for older textbook notebooks/qmd files that expect
        ModelSpec.get_kv_cache_size(...). For non-transformer models, KV cache
        is not applicable.
        """
        from .constants import ureg

        if self.architecture != "Transformer":
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute 'get_kv_cache_size' "
                f"for non-transformer model {self.name!r}"
            )

        if self.layers is None or self.parameters is None:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute 'get_kv_cache_size' "
                f"because {self.name!r} lacks transformer metadata"
            )

        # Heuristic hidden-size estimate from parameter count and layer count.
        # For dense decoder-style transformers, params ~ 12 * L * d^2 is a
        # reasonable back-of-the-envelope approximation.
        param_count = self.parameters.to(ureg.count).magnitude
        d_model = (param_count / (12 * self.layers)) ** 0.5

        bytes_per_elem = precision.to(ureg.byte).magnitude

        # KV cache ~ 2 (K+V) * batch * seq_len * layers * d_model * bytes
        total_bytes = 2 * batch_size * seq_len * self.layers * d_model * bytes_per_elem

        return (total_bytes * ureg.byte).to(ureg.byte)

    def __repr__(self):
        return f"Model({self.name}, {self.architecture})"

class GPT(Registry):
    """GPT Model Family."""
    GPT2 = ModelSpec("GPT-2 (1.5B)", GPT2_PARAMS, "Transformer", layers=48)
    GPT3 = ModelSpec("GPT-3 (175B)", GPT3_PARAMS, "Transformer", layers=96, training_ops=GPT3_TRAINING_OPS)
    GPT4 = ModelSpec("GPT-4", GPT4_EST_PARAMS, "Transformer", layers=120, training_gpu_days=GPT4_TRAINING_GPU_DAYS)

class Language(Registry):
    """Large Language Models."""
    # GPT is a nested registry here, but for list() we want to treat it specially or flatten it
    BERT_Base = ModelSpec("BERT-Base", BERT_BASE_PARAMS, "Transformer", layers=12, inference_flops=22e9 * ureg.flop)
    BERT_Large = ModelSpec("BERT-Large", BERT_LARGE_PARAMS, "Transformer", layers=24)
    Llama2_70B = ModelSpec("Llama-2-70B", 70e9 * ureg.param, "Transformer", layers=80)
    Llama3_8B = ModelSpec("Llama-3.1-8B", LLAMA3_8B_PARAMS, "Transformer", layers=32)
    Llama3_70B = ModelSpec("Llama-3.1-70B", LLAMA3_70B_PARAMS, "Transformer", layers=80)
    Llama3_405B = ModelSpec("Llama-3.1-405B", LLAMA3_405B_PARAMS, "Transformer", layers=126)

class Recommendation(Registry):
    """Recommendation Models."""
    DLRM = ModelSpec("DLRM", 25e9 * ureg.param, "DLRM", model_size=DLRM_MODEL_SIZE_FP32)

class Vision(Registry):
    """Image Classification and Detection."""
    ALEXNET = ModelSpec("AlexNet", ALEXNET_PARAMS, "CNN", layers=8)
    ResNet50 = ModelSpec("ResNet-50", RESNET50_PARAMS, "CNN", layers=50, inference_flops=RESNET50_FLOPs)
    MobileNetV1 = ModelSpec("MobileNetV1", 4.2e6 * ureg.param, "CNN", layers=28)
    MobileNetV2 = ModelSpec("MobileNetV2", MOBILENETV2_PARAMS, "CNN", layers=54, inference_flops=MOBILENETV2_FLOPs)
    YOLOv8_Nano = ModelSpec("YOLOv8-Nano", 3.2e6 * ureg.param, "CNN", layers=225, training_ops=8.7e9 * ureg.flop)

class Tiny(Registry):
    """Always-on and Embedded models."""
    DS_CNN = ModelSpec("DS-CNN (KWS)", KWS_DSCNN_PARAMS, "CNN", inference_flops=KWS_DSCNN_FLOPs)
    AnomalyDetector = ModelSpec("Anomaly Detector", ANOMALY_MODEL_PARAMS, "MLP")
    WakeVision = ModelSpec("Wake Vision (Doorbell)", 0.25e6 * ureg.param, "CNN", inference_flops=25e6 * ureg.flop)

class Models(Registry):
    Language = Language
    Recommendation = Recommendation
    Vision = Vision
    Tiny = Tiny
    GPT = GPT
    
    # Common aliases
    GPT2 = GPT.GPT2
    GPT3 = GPT.GPT3
    GPT4 = GPT.GPT4
    BERT = Language.BERT_Base
    DLRM = Recommendation.DLRM
    ResNet50 = Vision.ResNet50
    MobileNetV2 = Vision.MobileNetV2
    ALEXNET = Vision.ALEXNET

    @classmethod
    def list(cls, sort_by: str = 'parameters', reverse: bool = False) -> List[ModelSpec]:
        """Consolidated list of all models from all domains."""
        all_items = []
        # Flatten the categories
        all_items.extend(cls.GPT.list())
        all_items.extend(cls.Language.list())
        all_items.extend(cls.Recommendation.list())
        all_items.extend(cls.Vision.list())
        all_items.extend(cls.Tiny.list())
        
        # Deduplicate (aliases might cause duplicates)
        seen = set()
        unique_items = []
        for item in all_items:
            if item.name not in seen:
                unique_items.append(item)
                seen.add(item.name)
        
        if sort_by:
            unique_items.sort(key=lambda x: getattr(x, sort_by, 0), reverse=reverse)
        return unique_items
