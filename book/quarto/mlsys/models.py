# models.py
# Hierarchical Model Definitions for MLSys Textbook

from dataclasses import dataclass
from typing import Optional, Tuple
from .constants import (
    ureg, Q_,
    GPT2_PARAMS, GPT3_PARAMS, GPT4_TRAINING_GPU_DAYS,
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
        """Validate model specs."""
        if self.parameters is not None:
            assert self.parameters.magnitude > 0, f"{self.name}: Parameter count must be positive."

    def size_in_bytes(self, precision: Q_ = BYTES_FP16) -> Q_:
        """Calculates the weight storage size for a given precision."""
        if self.model_size:
            return self.model_size
        return (self.parameters.magnitude * precision).to('byte')

    def __repr__(self):
        return f"Model({self.name}, {self.architecture})"

class GPT:
    """GPT Model Family."""
    GPT2 = ModelSpec("GPT-2 (1.5B)", GPT2_PARAMS, "Transformer", layers=48)
    GPT3 = ModelSpec("GPT-3 (175B)", GPT3_PARAMS, "Transformer", layers=96, training_ops=GPT3_TRAINING_OPS)
    GPT4 = ModelSpec("GPT-4", 1.8e12 * ureg.count, "Transformer", layers=120, training_gpu_days=GPT4_TRAINING_GPU_DAYS)

class Language:
    """Large Language Models."""
    GPT = GPT
    BERT_Base = ModelSpec("BERT-Base", BERT_BASE_PARAMS, "Transformer", layers=12, inference_flops=22e9 * ureg.flop)
    BERT_Large = ModelSpec("BERT-Large", BERT_LARGE_PARAMS, "Transformer", layers=24)
    Llama2_70B = ModelSpec("Llama-2-70B", 70e9 * ureg.count, "Transformer", layers=80)

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
