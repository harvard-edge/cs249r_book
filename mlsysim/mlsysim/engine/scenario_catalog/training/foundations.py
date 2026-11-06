"""Named training scenarios for foundations.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....systems.registry import Clusters as _Clusters
from ....reference_stats.registry import ReferenceStats as _ReferenceStats
from ....systems.registry import Nodes as _Nodes
from ....ops import Ops as _Ops
from ... import scenario_types as _st


Gpt3PretrainingFrontier8K = _st.TrainingRunScenario(
    name="175B pretraining on 8,192 H100s",
    description="GPT-3 175B on its training tokens at the H100 FP8 peak, the 8,192-GPU "
    "pod's modeled scaling efficiency, and the standard run overheads.",
    workload=_Models.Language.GPT3,
    fleet=_Clusters.Frontier_8K,
    precision="fp8",
    local_efficiency=_ReferenceStats.TrainingScaleProfiles.LocalComputeEfficiency,
    scaling_efficiency=_ReferenceStats.TrainingScaleProfiles.Eff8192Gpu,
    overheads={
        "checkpoint": _Ops.TrainingRunOverheads.Checkpoint,
        "failures": _Ops.TrainingRunOverheads.FailureRecovery,
        "maintenance": _Ops.TrainingRunOverheads.Maintenance,
    },
)


Gpt3TrainingStateTp8 = _st.TrainingStateScenario(
    name="175B training state under 8-way tensor parallelism",
    description="Mixed-precision Adam state of GPT-3 175B split over one DGX H100 node "
    "(TP-8), with 128 data-parallel replicas for ZeRO sharding of the moments.",
    workload=_Models.Language.GPT3,
    tp_size=_Nodes.DGX_H100.accelerators_per_node,
    dp_size=128,  # scenario assumption: data-parallel replicas (1,024 GPUs)
    activations=Q_(200, "GB"),  # scenario assumption: inside the 100 to 400 GB activation range
)


PeakPlannedTrainingBudget = _st.SustainedThroughputScenario(
    name="Training budget planned from peak throughput",
    description="1,000 GPU-hours estimated from datasheet TFLOP/s, for a workload that sustains 40 percent of peak.",
    planned_device_hours=1000,  # scenario assumption: budget from the datasheet rate
    sustained_fraction=0.4,  # scenario assumption: sustained share of peak
)
