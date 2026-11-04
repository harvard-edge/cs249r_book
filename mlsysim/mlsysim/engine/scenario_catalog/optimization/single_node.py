"""Named optimization scenarios for single node.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....systems.registry import Clusters as _Clusters
from ....datasets.registry import Datasets as _Datasets
from ... import scenario_types as _st
from ....core.units import COLOR_DEPTH_8BIT as _COLOR_DEPTH_8BIT


# --- Single-machine iron law, Amdahl, and silent degradation (added 2026-10-20) ---
ImageNetInputSpace = _st.InputSpaceScenario(
    name="Every possible ImageNet-sized input",
    description="224 x 224 RGB images at 8 bits per channel, against the 50,000 validation images.",
    dataset=_Datasets.ImageNet,
    levels=_COLOR_DEPTH_8BIT,
)


Gpt3TrainingTime1KA100 = _st.ClusterTrainingTimeScenario(
    name="GPT-3's training work on 1,024 A100s",
    description="GPT-3's 3.14e23 FLOPs at the A100's 16-bit peak, 45 percent end-to-end utilization.",
    workload=_Models.Language.GPT3,
    hardware=_Hardware.Cloud.A100,
    n_devices=int(_Clusters.Training_1K.total_accelerators),
    efficiency=0.45,  # scenario assumption: end-to-end utilization
)


Gpt3TrainingTime1KA100HigherUtilization = Gpt3TrainingTime1KA100.variant(
    name="The same run at higher utilization",
    description="The same work and hardware at 60 percent utilization.",
    efficiency=0.60,  # scenario assumption: utilization after scheduling improvements
)


IronLawMemoryBoundExample = _st.IronLawScenario(
    name="A memory-bound task on one accelerator",
    description="Moves 2 GB at 100 GB/s and runs 1 GFLOP at an effective 100 GFLOP/s, plus 5 ms "
    "overhead; the upgrade doubles bandwidth.",
    data_volume=Q_(2, "GB"),  # scenario assumption
    bandwidth=Q_(100, "GB/s"),  # scenario assumption: effective rate
    operations=Q_(1, "GFLOP"),  # scenario assumption
    effective_rate=Q_(100, "GFLOP/s"),  # scenario assumption: peak times utilization
    overhead=Q_(5, "ms"),  # scenario assumption
    memory_speedup=2.0,  # scenario assumption: twice the bandwidth
)


IronLawComputeHeavyExample = _st.IronLawScenario(
    name="A compute-heavy task with a faster arithmetic rate",
    description="2 s of data movement, 8 s of computation, 1 s of overhead; the upgrade doubles the arithmetic rate.",
    memory_time_stated=Q_(2, "second"),  # scenario assumption
    compute_time_stated=Q_(8, "second"),  # scenario assumption
    overhead=Q_(1, "second"),  # scenario assumption
    compute_speedup=2.0,  # scenario assumption
)


ModelSpeedupRequestPath = _st.AmdahlScenario(
    name="A request path whose model gets 3x faster",
    description="60 ms preprocessing, 45 ms model, 25 ms postprocessing; only the model speeds up.",
    stages={"preprocess": Q_(60, "ms"), "model": Q_(45, "ms"), "postprocess": Q_(25, "ms")},  # scenario assumption
    stage="model",
    speedup=3.0,  # scenario assumption: better kernels and quantization
)
