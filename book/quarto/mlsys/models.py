# models.py
# Hierarchical Model Definitions for MLSys Textbook

from dataclasses import dataclass
from .constants import (
    ureg, Q_,
    GPT2_PARAMS, GPT3_PARAMS,
    RESNET50_PARAMS, MOBILENETV2_PARAMS,
    KWS_DSCNN_PARAMS, BYTES_FP32, BYTES_FP16, BYTES_INT8
)

@dataclass(frozen=True)
class ModelSpec:
    name: str
    parameters: Q_
    architecture: str # "Transformer", "CNN", "MLP"
    
    def size_in_bytes(self, precision: Q_ = BYTES_FP16) -> Q_:
        """Calculates the weight storage size for a given precision."""
        return (self.parameters.magnitude * precision).to('byte')

    def __repr__(self):
        return f"Model({self.name}, {self.architecture})"

class Language:
    """Large Language Models."""
    GPT2 = ModelSpec("GPT-2 (1.5B)", GPT2_PARAMS, "Transformer")
    GPT3 = ModelSpec("GPT-3 (175B)", GPT3_PARAMS, "Transformer")
    Llama2_70B = ModelSpec("Llama-2-70B", 70e9 * ureg.count, "Transformer")

class Vision:
    """Image Classification and Detection."""
    ResNet50 = ModelSpec("ResNet-50", RESNET50_PARAMS, "CNN")
    MobileNetV2 = ModelSpec("MobileNetV2", MOBILENETV2_PARAMS, "CNN")

class Tiny:
    """Always-on and Embedded models."""
    DS_CNN = ModelSpec("DS-CNN (KWS)", KWS_DSCNN_PARAMS, "CNN")

class Models:
    Language = Language
    Vision = Vision
    Tiny = Tiny
    
    # Common aliases
    GPT2 = Language.GPT2
    GPT3 = Language.GPT3
    ResNet50 = Vision.ResNet50
    MobileNetV2 = Vision.MobileNetV2
