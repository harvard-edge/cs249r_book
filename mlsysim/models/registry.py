from .types import TransformerWorkload, CNNWorkload, Workload
from ..core.registry import Registry
from .types import TransformerWorkload, CNNWorkload, Workload
from ..core.constants import (
    ureg,
    GPT2_PARAMS, GPT3_PARAMS, GPT4_EST_PARAMS, GPT3_TRAINING_OPS,
    BERT_BASE_PARAMS, BERT_LARGE_PARAMS,
    RESNET50_PARAMS, RESNET50_FLOPs, MOBILENETV2_PARAMS, MOBILENETV2_FLOPs,
    LLAMA3_8B_PARAMS, LLAMA3_70B_PARAMS,
    KWS_DSCNN_PARAMS, KWS_DSCNN_FLOPs, YOLOV8_NANO_FLOPs,
    ALEXNET_PARAMS, ANOMALY_MODEL_PARAMS, DLRM_MODEL_SIZE_FP32
)

class LanguageModels(Registry):
    GPT2 = TransformerWorkload(
        name="GPT-2 (1.5B)",
        architecture="Transformer",
        parameters=GPT2_PARAMS,
        layers=48,
        hidden_dim=1600,
        heads=25,
        inference_flops=2 * GPT2_PARAMS.magnitude * ureg.flop
    )
    GPT3 = TransformerWorkload(
        name="GPT-3 (175B)",
        architecture="Transformer",
        parameters=GPT3_PARAMS,
        layers=96,
        hidden_dim=12288,
        heads=96,
        training_ops=GPT3_TRAINING_OPS,
        inference_flops=2 * GPT3_PARAMS.magnitude * ureg.flop
    )
    GPT4 = TransformerWorkload(
        name="GPT-4",
        architecture="Transformer",
        parameters=GPT4_EST_PARAMS,
        layers=120,
        hidden_dim=16384,
        heads=128,
        inference_flops=2 * GPT4_EST_PARAMS.magnitude * ureg.flop
    )
    BERT_Base = TransformerWorkload(
        name="BERT-Base",
        architecture="Transformer",
        parameters=BERT_BASE_PARAMS,
        layers=12,
        hidden_dim=768,
        heads=12,
        inference_flops=22e9 * ureg.flop
    )
    Llama2_70B = TransformerWorkload(
        name="Llama-2-70B",
        architecture="Transformer",
        parameters=70e9 * ureg.param,
        layers=80,
        hidden_dim=8192,
        heads=64,
        inference_flops=140e9 * ureg.flop
    )
    Llama3_8B = TransformerWorkload(
        name="Llama-3.1-8B",
        architecture="Transformer",
        parameters=LLAMA3_8B_PARAMS,
        layers=32,
        hidden_dim=4096,
        heads=32,
        kv_heads=8,
        inference_flops=2 * LLAMA3_8B_PARAMS.magnitude * ureg.flop
    )
    Llama3_70B = TransformerWorkload(
        name="Llama-3.1-70B",
        architecture="Transformer",
        parameters=LLAMA3_70B_PARAMS,
        layers=80,
        hidden_dim=8192,
        heads=64,
        kv_heads=8,
        inference_flops=2 * LLAMA3_70B_PARAMS.magnitude * ureg.flop
    )

class VisionModels(Registry):
    ResNet50 = CNNWorkload(
        name="ResNet-50",
        architecture="CNN",
        parameters=RESNET50_PARAMS,
        inference_flops=RESNET50_FLOPs,
        layers=50
    )
    MobileNetV2 = CNNWorkload(
        name="MobileNetV2",
        architecture="CNN",
        parameters=MOBILENETV2_PARAMS,
        inference_flops=MOBILENETV2_FLOPs,
        layers=54
    )
    YOLOv8_Nano = CNNWorkload(
        name="YOLOv8-Nano",
        architecture="CNN",
        parameters=3.2e6 * ureg.param,
        inference_flops=YOLOV8_NANO_FLOPs,
        layers=225
    )
    AlexNet = CNNWorkload(
        name="AlexNet",
        architecture="CNN",
        parameters=ALEXNET_PARAMS,
        inference_flops=1.5e9 * ureg.flop, # Estimated
        layers=8
    )

class TinyModels(Registry):
    DS_CNN = CNNWorkload(
        name="DS-CNN (KWS)",
        architecture="CNN",
        parameters=KWS_DSCNN_PARAMS,
        inference_flops=KWS_DSCNN_FLOPs
    )
    WakeVision = CNNWorkload(
        name="Wake Vision (Doorbell)",
        architecture="CNN",
        parameters=0.25e6 * ureg.param,
        inference_flops=25e6 * ureg.flop
    )
    AnomalyDetector = Workload(
        name="Anomaly Detector",
        architecture="MLP",
        # Generic Workload doesn't have params in type, but we can override
    )

class RecommendationModels(Registry):
    # Special class for DLRM as it's defined by size
    DLRM = Workload(
        name="DLRM",
        architecture="DLRM",
        model_size=DLRM_MODEL_SIZE_FP32
    )
    # Note: We'll add specialized size methods if needed, 
    # but for now we maintain string compatibility.

class Models(Registry):
    Language = LanguageModels
    Vision = VisionModels
    Tiny = TinyModels
    Recommendation = RecommendationModels
    
    GPT2 = LanguageModels.GPT2
    GPT3 = LanguageModels.GPT3
    GPT4 = LanguageModels.GPT4
    Llama2_70B = LanguageModels.Llama2_70B
    Llama3_8B = LanguageModels.Llama3_8B
    Llama3_70B = LanguageModels.Llama3_70B
    ResNet50 = VisionModels.ResNet50
    MobileNetV2 = VisionModels.MobileNetV2
    WakeVision = TinyModels.WakeVision
    DLRM = RecommendationModels.DLRM
    AlexNet = VisionModels.AlexNet
