from .types import TransformerWorkload, CNNWorkload, Workload, SSMWorkload, DiffusionWorkload
from ..core.registry import Registry
from ..core.constants import (
    ureg,
    # Language model constants
    GPT2_PARAMS, GPT2_LAYERS, GPT2_HIDDEN_DIM, GPT2_HEADS,
    GPT3_PARAMS, GPT3_LAYERS, GPT3_HIDDEN_DIM, GPT3_HEADS, GPT3_TRAINING_OPS,
    GPT4_EST_PARAMS, GPT4_LAYERS, GPT4_HIDDEN_DIM, GPT4_HEADS,
    BERT_BASE_PARAMS, BERT_BASE_LAYERS, BERT_BASE_HIDDEN_DIM, BERT_BASE_HEADS, BERT_BASE_FLOPs,
    BERT_LARGE_PARAMS, BERT_LARGE_LAYERS, BERT_LARGE_HIDDEN_DIM, BERT_LARGE_HEADS, BERT_LARGE_FLOPs,
    LLAMA2_70B_PARAMS, LLAMA2_70B_LAYERS, LLAMA2_70B_HIDDEN_DIM, LLAMA2_70B_HEADS,
    LLAMA3_8B_PARAMS, LLAMA3_8B_LAYERS, LLAMA3_8B_HIDDEN_DIM, LLAMA3_8B_HEADS, LLAMA3_8B_KV_HEADS,
    LLAMA3_70B_PARAMS, LLAMA3_70B_LAYERS, LLAMA3_70B_HIDDEN_DIM, LLAMA3_70B_HEADS, LLAMA3_70B_KV_HEADS,
    # Vision model constants
    RESNET50_PARAMS, RESNET50_FLOPs,
    MOBILENETV2_PARAMS, MOBILENETV2_FLOPs,
    ALEXNET_PARAMS, ALEXNET_FLOPs,
    YOLOV8_NANO_PARAMS, YOLOV8_NANO_FLOPs, YOLOV8_NANO_LAYERS,
    # Tiny model constants
    KWS_DSCNN_PARAMS, KWS_DSCNN_FLOPs,
    WAKEVISION_PARAMS, WAKEVISION_FLOPs,
    ANOMALY_MODEL_PARAMS,
    # Recommendation constants
    DLRM_MODEL_SIZE_FP32,
    # SSM constants
    MAMBA_130M_PARAMS, MAMBA_130M_LAYERS, MAMBA_130M_HIDDEN_DIM, MAMBA_130M_STATE_SIZE,
    MAMBA_2_8B_PARAMS, MAMBA_2_8B_LAYERS, MAMBA_2_8B_HIDDEN_DIM, MAMBA_2_8B_STATE_SIZE,
    # Diffusion constants
    STABLE_DIFFUSION_V1_5_PARAMS, STABLE_DIFFUSION_V1_5_RESOLUTION,
    STABLE_DIFFUSION_V1_5_STEPS, STABLE_DIFFUSION_V1_5_FLOPs_PER_STEP,
)

class LanguageModels(Registry):
    GPT2 = TransformerWorkload(
        name="GPT-2 (1.5B)",
        architecture="Transformer",
        parameters=GPT2_PARAMS,
        layers=GPT2_LAYERS,
        hidden_dim=GPT2_HIDDEN_DIM,
        heads=GPT2_HEADS,
        inference_flops=2 * GPT2_PARAMS.magnitude * ureg.flop
    )
    GPT3 = TransformerWorkload(
        name="GPT-3 (175B)",
        architecture="Transformer",
        parameters=GPT3_PARAMS,
        layers=GPT3_LAYERS,
        hidden_dim=GPT3_HIDDEN_DIM,
        heads=GPT3_HEADS,
        training_ops=GPT3_TRAINING_OPS,
        inference_flops=2 * GPT3_PARAMS.magnitude * ureg.flop
    )
    GPT4 = TransformerWorkload(
        name="GPT-4",
        architecture="Transformer",
        parameters=GPT4_EST_PARAMS,
        layers=GPT4_LAYERS,
        hidden_dim=GPT4_HIDDEN_DIM,
        heads=GPT4_HEADS,
        inference_flops=2 * GPT4_EST_PARAMS.magnitude * ureg.flop
    )
    BERT_Base = TransformerWorkload(
        name="BERT-Base",
        architecture="Transformer",
        parameters=BERT_BASE_PARAMS,
        layers=BERT_BASE_LAYERS,
        hidden_dim=BERT_BASE_HIDDEN_DIM,
        heads=BERT_BASE_HEADS,
        inference_flops=BERT_BASE_FLOPs
    )
    BERT_Large = TransformerWorkload(
        name="BERT-Large",
        architecture="Transformer",
        parameters=BERT_LARGE_PARAMS,
        layers=BERT_LARGE_LAYERS,
        hidden_dim=BERT_LARGE_HIDDEN_DIM,
        heads=BERT_LARGE_HEADS,
        inference_flops=BERT_LARGE_FLOPs
    )
    Llama2_70B = TransformerWorkload(
        name="Llama-2-70B",
        architecture="Transformer",
        parameters=LLAMA2_70B_PARAMS,
        layers=LLAMA2_70B_LAYERS,
        hidden_dim=LLAMA2_70B_HIDDEN_DIM,
        heads=LLAMA2_70B_HEADS,
        inference_flops=2 * LLAMA2_70B_PARAMS.magnitude * ureg.flop
    )
    Llama3_8B = TransformerWorkload(
        name="Llama-3.1-8B",
        architecture="Transformer",
        parameters=LLAMA3_8B_PARAMS,
        layers=LLAMA3_8B_LAYERS,
        hidden_dim=LLAMA3_8B_HIDDEN_DIM,
        heads=LLAMA3_8B_HEADS,
        kv_heads=LLAMA3_8B_KV_HEADS,
        inference_flops=2 * LLAMA3_8B_PARAMS.magnitude * ureg.flop
    )
    Llama3_70B = TransformerWorkload(
        name="Llama-3.1-70B",
        architecture="Transformer",
        parameters=LLAMA3_70B_PARAMS,
        layers=LLAMA3_70B_LAYERS,
        hidden_dim=LLAMA3_70B_HIDDEN_DIM,
        heads=LLAMA3_70B_HEADS,
        kv_heads=LLAMA3_70B_KV_HEADS,
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
        parameters=YOLOV8_NANO_PARAMS,
        inference_flops=YOLOV8_NANO_FLOPs,
        layers=YOLOV8_NANO_LAYERS
    )
    AlexNet = CNNWorkload(
        name="AlexNet",
        architecture="CNN",
        parameters=ALEXNET_PARAMS,
        inference_flops=ALEXNET_FLOPs,
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
        parameters=WAKEVISION_PARAMS,
        inference_flops=WAKEVISION_FLOPs
    )
    AnomalyDetector = Workload(
        name="Anomaly Detector",
        architecture="MLP",
        parameters=ANOMALY_MODEL_PARAMS,
        inference_flops=2 * ANOMALY_MODEL_PARAMS.magnitude * ureg.flop
    )

class RecommendationModels(Registry):
    DLRM = Workload(
        name="DLRM",
        architecture="DLRM",
        model_size=DLRM_MODEL_SIZE_FP32
    )

class StateSpaceModels(Registry):
    Mamba_130M = SSMWorkload(
        name="Mamba-130M",
        architecture="SSM",
        parameters=MAMBA_130M_PARAMS,
        layers=MAMBA_130M_LAYERS,
        hidden_dim=MAMBA_130M_HIDDEN_DIM,
        state_size=MAMBA_130M_STATE_SIZE,
        inference_flops=2 * MAMBA_130M_PARAMS.magnitude * ureg.flop
    )
    Mamba_2_8B = SSMWorkload(
        name="Mamba-2.8B",
        architecture="SSM",
        parameters=MAMBA_2_8B_PARAMS,
        layers=MAMBA_2_8B_LAYERS,
        hidden_dim=MAMBA_2_8B_HIDDEN_DIM,
        state_size=MAMBA_2_8B_STATE_SIZE,
        inference_flops=2 * MAMBA_2_8B_PARAMS.magnitude * ureg.flop
    )

class GenerativeVisionModels(Registry):
    StableDiffusion_v1_5 = DiffusionWorkload(
        name="Stable Diffusion v1.5",
        architecture="Diffusion/U-Net",
        parameters=STABLE_DIFFUSION_V1_5_PARAMS,
        resolution=STABLE_DIFFUSION_V1_5_RESOLUTION,
        denoising_steps=STABLE_DIFFUSION_V1_5_STEPS,
        inference_flops=STABLE_DIFFUSION_V1_5_FLOPs_PER_STEP
    )

class Models(Registry):
    Language = LanguageModels
    Vision = VisionModels
    Tiny = TinyModels
    Recommendation = RecommendationModels
    StateSpace = StateSpaceModels
    GenerativeVision = GenerativeVisionModels

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
    Mamba_2_8B = StateSpaceModels.Mamba_2_8B
    StableDiffusion_v1_5 = GenerativeVisionModels.StableDiffusion_v1_5
