"""Named optimization scenarios for expansion.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....systems.registry import Clusters as _Clusters
from ....core.units import BYTES_FP16 as _BYTES_FP16
from ....systems.reliability import Reliability as _Reliability
from ... import scenario_types as _st
from ..reliability.fault_tolerance import Gpt3CheckpointInterval10K, HalfSpeedStraggler1K
from ..sustainability.sustainability import KaplanTenPercentLowerLoss


# --- Fleet-scale recall: expansion, rare events, MoE bytes, link energy (added 2026-10-20) ---
Gpt3ClusterDoubling1K = _st.ClusterExpansionScenario(
    name="Doubling the 175B job from 1,024 to 2,048 GPUs on the same checkpoint tier",
    description="The 14-byte-per-parameter checkpoint at the per-job write bandwidth, priced by "
    "Young-Daly checkpoint-and-rework overhead against the GPU-only interval at each size.",
    checkpoint=Gpt3CheckpointInterval10K,  # write path only; the MTBF is replaced at each size
    component_mtbf=Q_(_Reliability.Gpu.mttf_hours, "hour"),
    n_devices=_Clusters.Training_1K.total_accelerators,
    factor=2.0,  # scenario assumption: double the reference cluster
)


SlowWorker8K = HalfSpeedStraggler1K.variant(
    name="One of 8,000 workers at 80 percent speed",
    description="An equal-work, barrier-dominated synchronous step; every worker waits for the slow one.",
    n_devices=8_000,  # scenario assumption
    slow_speed=0.80,  # scenario assumption: one worker at 80 percent of peer speed
)


KaplanComputeDoubling = KaplanTenPercentLowerLoss.variant(
    name="Loss change when training compute doubles under Kaplan's compute law",
    description="The fitted power law multiplies loss by 2^-alpha per doubling, valid only over its studied range.",
    loss_reduction=None,
    resource_multiple=2.0,  # scenario assumption: double the training compute
)


DeepSeekV3WeightTraffic = _st.ExpertWeightTrafficScenario(
    name="DeepSeek-V3 weight bytes against a dense 400B model, FP16",
    description="All experts resident against the active parameters one token reads, beside a dense "
    "model that reads every weight each step. Payload counts, not latencies.",
    workload=_Models.Language.DeepSeek_V3,
    dense_parameters=Q_(400, "Bparam"),  # scenario assumption: illustrative dense baseline
    bytes_per_param=_BYTES_FP16,
)
