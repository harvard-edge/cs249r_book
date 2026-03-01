# models.py
# Hierarchical Model Definitions for MLSys Textbook

import pint
from dataclasses import dataclass
from typing import Optional, Tuple
from .constants import (
    ureg, Q_,
    GPT2_PARAMS, GPT3_PARAMS, GPT4_EST_PARAMS, GPT4_TRAINING_GPU_DAYS,
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

    def __repr__(self):
        return f"Model({self.name}, {self.architecture})"

class GPT:
    """GPT Model Family."""
    GPT2 = ModelSpec("GPT-2 (1.5B)", GPT2_PARAMS, "Transformer", layers=48)
    GPT3 = ModelSpec("GPT-3 (175B)", GPT3_PARAMS, "Transformer", layers=96, training_ops=GPT3_TRAINING_OPS)
    GPT4 = ModelSpec("GPT-4", GPT4_EST_PARAMS, "Transformer", layers=120, training_gpu_days=GPT4_TRAINING_GPU_DAYS)

class Language:
    """Large Language Models."""
    GPT = GPT
    BERT_Base = ModelSpec("BERT-Base", BERT_BASE_PARAMS, "Transformer", layers=12, inference_flops=22e9 * ureg.flop)
    BERT_Large = ModelSpec("BERT-Large", BERT_LARGE_PARAMS, "Transformer", layers=24)
    Llama2_70B = ModelSpec("Llama-2-70B", 70e9 * ureg.param, "Transformer", layers=80)
    Llama3_70B = ModelSpec("Llama-3-70B", 70.6e9 * ureg.param, "Transformer", layers=80)
    Llama3_405B = ModelSpec("Llama-3.1-405B", 405e9 * ureg.param, "Transformer", layers=126)

class Recommendation:
    """Recommendation Models."""
    DLRM = ModelSpec("DLRM", 25e9 * ureg.param, "DLRM", model_size=DLRM_MODEL_SIZE_FP32)

class Vision:
    """Image Classification and Detection."""
    ALEXNET = ModelSpec("AlexNet", ALEXNET_PARAMS, "CNN", layers=8)
    ResNet50 = ModelSpec("ResNet-50", RESNET50_PARAMS, "CNN", layers=50, inference_flops=RESNET50_FLOPs)
    MobileNetV1 = ModelSpec("MobileNetV1", 4.2e6 * ureg.param, "CNN", layers=28)
    MobileNetV2 = ModelSpec("MobileNetV2", MOBILENETV2_PARAMS, "CNN", layers=54, inference_flops=MOBILENETV2_FLOPs)
    YOLOv8_Nano = ModelSpec("YOLOv8-Nano", 3.2e6 * ureg.param, "CNN", layers=225, training_ops=8.7e9 * ureg.flop)

class Tiny:
    """Always-on and Embedded models."""
    DS_CNN = ModelSpec("DS-CNN (KWS)", KWS_DSCNN_PARAMS, "CNN", inference_flops=KWS_DSCNN_FLOPs)
    AnomalyDetector = ModelSpec("Anomaly Detector", ANOMALY_MODEL_PARAMS, "MLP")
    WakeVision = ModelSpec("Wake Vision (Doorbell)", 0.25e6 * ureg.param, "CNN", inference_flops=25e6 * ureg.flop)

class Models:
    Language = Language
    Recommendation = Recommendation
    Vision = Vision
    Tiny = Tiny
    
    # Common aliases
    GPT2 = GPT.GPT2
    GPT3 = GPT.GPT3
    GPT4 = GPT.GPT4
    BERT = Language.BERT_Base
    DLRM = Recommendation.DLRM
    ResNet50 = Vision.ResNet50
    MobileNetV2 = Vision.MobileNetV2
    ALEXNET = Vision.ALEXNET
