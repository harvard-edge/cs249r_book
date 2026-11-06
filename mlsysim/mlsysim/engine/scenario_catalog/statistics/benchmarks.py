"""Named statistics scenarios for benchmarks.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....literature.registry import Literature as _Literature
from ... import scenario_types as _st


# --- Benchmarking (added 2026-10-20) ------------------------------------
DatasheetVersusSustainedEdgeChip = _st.EfficiencyClaimGapScenario(
    name='A "10 TOPS at 0.5 W" datasheet line against sustained load',
    description="An edge accelerator's datasheet operating point against the same chip under sustained "
    "inference load, after thermal throttling (illustrative, no named device).",
    claimed_rate=Q_(10, "TOPS"),  # scenario assumption: datasheet peak
    claimed_power=Q_(0.5, "W"),  # scenario assumption: datasheet power
    sustained_rate=Q_(3, "TOPS"),  # scenario assumption: throughput under sustained load
    sustained_power=Q_(2, "W"),  # scenario assumption: power under sustained load
)


MobileNetV2CameraTrapBoard = _st.HostAcceleratorBoardScenario(
    name="MobileNetV2 on a camera trap's host CPU and attached INT8 accelerator",
    description="A labeled composite edge board: FP32 and INT8 on the low-power host CPU, INT8 on the "
    "attached accelerator; end to end adds host preprocessing and the link copy.",
    host_fp32_latency=Q_(50, "ms"),  # scenario assumption: FP32 on the host CPU
    host_int8_latency=Q_(15, "ms"),  # scenario assumption: INT8 on the host CPU
    accelerator_latency=Q_(2, "ms"),  # scenario assumption: INT8 on the accelerator
    host_end_to_end=Q_(18, "ms"),  # scenario assumption: host CPU, full request path
    accelerator_end_to_end=Q_(6, "ms"),  # scenario assumption: accelerator, full request path
    host_power=Q_(120, "mW"),  # scenario assumption: host CPU active power
    accelerator_power=Q_(500, "mW"),  # scenario assumption: accelerator active power
    frame_rate=Q_(30, "1/second"),  # scenario assumption: 30-frame-per-second camera stream
)


PreprocessBoundVisionPipeline = _st.PipelineSpeedupScenario(
    name="A vision pipeline whose inference stage is made 5x faster",
    description="8 ms of JPEG decode, resize, and normalize on the host, then 10 ms of inference.",
    stages={
        "preprocess": Q_(8, "ms"),  # scenario assumption
        "inference": Q_(10, "ms"),
    },  # scenario assumption
    accelerated="inference",
    speedup=5,  # scenario assumption: inference-only optimization
)


ResNet50H100TrainingCeiling = _st.AcceleratorInputFeedScenario(
    name="ResNet-50 training on one H100: arithmetic ceiling against the MLPerf rate",
    description="Peak dense FP16 over three forward passes of work per image, against the per-accelerator "
    "rate of NVIDIA's 64-H100 MLPerf Training v3.1 submission.",
    workload=_Models.Vision.ResNet50,
    hardware=_Hardware.Cloud.H100,
    sustained_rate=Q_(float(_Literature.Benchmarks.ResNet50H100TrainThroughput), "count/second"),
    accelerators=int(_Literature.Benchmarks.ResNet50H100TrainThroughputAccelerators),
    training_flops_factor=3,  # convention: the backward pass costs about twice the forward pass
)


KeywordSpottingOnEsp32S3 = _st.PeriodicInferencePowerScenario(
    name="DS-CNN keyword spotting once a second on an ESP32-S3-class board",
    description="One inference per one-second audio window at the board's peak INT8 rate; active at TDP, "
    "idle at its minimum power between inferences.",
    workload=_Models.Tiny.DS_CNN,
    hardware=_Hardware.Tiny.ESP32_S3,
    period=Q_(1, "second"),  # scenario assumption: one inference per one-second window
    precision="int8",
    energy_cut=2.0,  # scenario assumption: a model that halves inference energy
)


ModelOnlyMeanVersusProductionP99 = _st.MeanVersusTailScenario(
    name="A model-only mean latency against a production p99",
    description="Mean latency on evenly spaced synthetic inputs against the p99 under production traffic "
    "(illustrative).",
    mean=Q_(15, "ms"),  # scenario assumption: model-only mean
    tail_low=Q_(150, "ms"),  # scenario assumption: production p99, low
    tail_high=Q_(200, "ms"),  # scenario assumption: production p99, high
)


ImageClassifierEightGpuScaling = _st.StrongScalingPairScenario(
    name="An image classifier trained on one GPU and on eight",
    description="A fixed dataset: 24 hours on one GPU, 4 hours on eight (illustrative).",
    base_devices=1,
    base_time=Q_(24, "hour"),  # scenario assumption: single-GPU training time
    scaled_devices=8,
    scaled_time=Q_(4, "hour"),  # scenario assumption: eight-GPU training time
)
