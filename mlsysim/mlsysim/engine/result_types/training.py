"""Typed result models for training calculations."""

from __future__ import annotations

from typing import Any, Dict, Optional, List
from ...core.types import Quantity
from .base import SolverResult

__all__ = [
    "DistributedResult",
    "CheckpointResult",
    "TrainingMemoryResult",
    "MoERoutingResult",
    "TopologyResult",
    "ScalingResult",
    "TrainingRunResult",
    "TrainingStateResult",
    "ParallelLayoutResult",
    "ZeroShardingResult",
    "DataParallelScaleOutResult",
    "TensorParallelLayerResult",
    "AlignmentMemoryResult",
    "DenseNetworkResult",
    "EpochOverheadResult",
    "TargetMaskResult",
    "PerInvocationSerializationResult",
    "LossNormalizationResult",
    "SequencePackingResult",
    "PreferencePairResult",
    "RewardCompositionResult",
    "CenteredRewardResult",
    "ForkValueResult",
    "ContinuationForkCostResult",
    "GroupCoefficientsResult",
    "LengthNormalizationResult",
    "RolloutStepBudgetResult",
    "RepeatedVerificationResult",
    "RatioClipResult",
    "RolloutSlotCycleResult",
    "GroupBarrierResult",
    "VerifierPoolResult",
]


class DistributedResult(SolverResult):
    """Result from DistributedModel: fleet-wide training performance."""

    __module__ = "mlsysim.engine.results"

    node_profile: Any  # PerformanceProfile (avoid circular import)
    dp_communication_latency: Quantity
    tp_communication_latency: Quantity
    ep_communication_latency: Quantity
    communication_latency: Quantity  # total across all parallelism dimensions
    pipeline_bubble_latency: Quantity
    bubble_fraction: float
    step_latency_total: Quantity
    scaling_efficiency: float
    effective_throughput: Quantity
    parallelism: Dict[str, int]


class CheckpointResult(SolverResult):
    """Result from CheckpointModel: IOPS and MFU penalty modeling."""

    __module__ = "mlsysim.engine.results"

    checkpoint_size: Quantity
    write_time_seconds: Quantity
    max_bandwidth_required: Quantity
    storage_bottleneck: bool
    mfu_penalty_pct: float

    @property
    def write_time(self) -> Quantity:
        """Return checkpoint write duration with units.

        Returns
        -------
        Quantity
            Checkpoint write duration as a Pint quantity, conventionally in seconds.

        Notes
        -----
        Equation: write_time = write_time_seconds, with no arithmetic or unit conversion.

        Assumptions: The producing solver supplied write_time_seconds as a time, energy, or
        memory quantity appropriate to this result.

        Excludes: Re-evaluation of the model and any new feasibility or unit-dimension
        check. The original quantity and its unit are returned unchanged.

        References
        ----------
        Derivation: This compatibility property returns the existing write_time_seconds
        field directly; its value is defined by that field rather than a separate formula.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import CheckpointResult
        >>> r = CheckpointResult(checkpoint_size=Q_(1, "GB"),
        ...     write_time_seconds=Q_(2, "s"), max_bandwidth_required=Q_(1, "GB/s"),
        ...     storage_bottleneck=False, mfu_penalty_pct=0)
        >>> r.write_time is r.write_time_seconds
        True
        """
        return self.write_time_seconds


class TrainingMemoryResult(SolverResult):
    """Result from TrainingMemoryModel: per-accelerator training memory accounting."""

    __module__ = "mlsysim.engine.results"

    feasible: bool
    total_memory: Quantity
    available_memory: Quantity
    memory_utilization: float
    weights: Quantity
    gradients: Quantity
    optimizer_state: Quantity
    activations: Quantity
    communication_buffers: Quantity
    precision: str
    optimizer: str
    activation_checkpointing: str
    parallelism: Dict[str, int]


class MoERoutingResult(SolverResult):
    """Result from MoERoutingModel: MoE routing imbalance and all-to-all cost."""

    __module__ = "mlsysim.engine.results"

    effective_active_experts: float
    effective_active_parameters: Quantity
    active_parameter_multiplier: float
    routing_imbalance_factor: float
    token_dispatch_bytes: Quantity
    all_to_all_latency: Optional[Quantity] = None
    ep_size: int = 1


class TopologyResult(SolverResult):
    """Result from TopologyModel: network bisection bandwidth."""

    __module__ = "mlsysim.engine.results"

    effective_bw: Quantity
    bisection_bw: Quantity
    bisection_bw_fraction: float
    hops_avg: float
    topology: str
    num_nodes: int


class ScalingResult(SolverResult):
    """Result from ScalingModel: Chinchilla-optimal training parameters."""

    __module__ = "mlsysim.engine.results"

    optimal_parameters: Quantity
    optimal_tokens: Quantity
    compute_budget_flops: Quantity
    tokens_per_parameter: float


class TrainingRunResult(SolverResult):
    """Result from TrainingRunModel: wall-clock chain from peak to delivered time."""

    __module__ = "mlsysim.engine.results"

    total_flops: Quantity
    peak_time: Quantity  # every device at its peak
    compute_time: Quantity  # at the local compute efficiency
    scaled_time: Quantity  # after the scaling efficiency
    stage_times: Dict[str, Quantity]  # after each overhead, in order
    total_time: Quantity
    delivered_fraction: float  # peak_time / total_time


class TrainingStateResult(SolverResult):
    """Result from TrainingStateModel: persistent training state and its shards."""

    __module__ = "mlsysim.engine.results"

    weights: Quantity
    gradients: Quantity
    master_weights: Quantity
    adam_moment: Quantity  # one Adam moment (m or v)
    optimizer_moments: Quantity  # both moments
    persistent_state: Quantity
    activations: Quantity
    tp_weights_per_device: Quantity
    tp_gradients_per_device: Quantity
    tp_master_per_device: Quantity
    tp_moments_per_device: Quantity
    tp_activations_per_device: Quantity
    tp_total_per_device: Quantity
    tp_offloaded_per_device: Quantity  # after moving master and moments to host
    zero_moments_per_device: Quantity  # moments sharded over TP x DP
    zero_moments_per_tp_group: Quantity
    zero_weights_and_moments_per_device: Quantity


class ParallelLayoutResult(SolverResult):
    """Result from ParallelLayoutModel: one TP x PP x DP layout's step and state."""

    __module__ = "mlsysim.engine.results"

    n_devices: int
    step_flops: Quantity
    compute_time: Quantity  # per device per step (T_compute / N)
    gradient_bytes: Quantity  # full gradient at the communication precision
    dp_shard: Quantity  # gradient bytes each data-parallel rank reduces
    ring_bytes: Quantity  # 2(DP-1)/DP x shard, exact ring volume per device
    allreduce_time: Quantity  # 2 x shard / link bandwidth (large-N ring bound)
    hidden_time: Quantity
    exposed_time: Quantity
    bubble: float  # pipeline idle fraction
    efficiency: float  # with overlap and bubble
    efficiency_no_overlap: float
    overlap_share: float  # hidden / allreduce
    step_time: Quantity  # compute / (1 - bubble) + exposed
    state_total: Quantity  # persistent mixed-precision Adam state
    state_per_device: Quantity  # split by TP x PP
    zero1_state_per_device: Quantity  # plus optimizer state sharded across DP
    fleet_mfu: float  # local efficiency x scaling efficiency
    double_counted_mfu: float  # the pitfall: scaling efficiency applied twice
    min_devices_for_state: Optional[int] = None


class ZeroShardingResult(SolverResult):
    """Result from ZeroShardingModel: per-device state at each ZeRO stage."""

    __module__ = "mlsysim.engine.results"

    weights: Quantity
    gradients: Quantity
    optimizer_state: Quantity
    stage_state: Dict[str, Quantity]  # "DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3" per device
    traffic_multiple: Dict[str, float]  # per-device bytes relative to DDP's AllReduce
    activations: Quantity
    zero3_total_per_device: Quantity  # ZeRO-3 state plus activations
    gather_traffic: Quantity  # ZeRO-3 per-device bytes per step
    gather_time: Optional[Quantity] = None


class DataParallelScaleOutResult(SolverResult):
    """Result from DataParallelScaleOutModel: a data-parallel job inside and across nodes."""

    __module__ = "mlsysim.engine.results"

    n_devices: int
    gradient_bytes: Quantity
    intra_bytes: Quantity
    inter_bytes: Quantity
    intra_time: Quantity
    inter_time: Quantity
    communication_time: Quantity
    compute_time: Quantity  # per optimizer update (accumulation x step)
    update_time: Quantity
    communication_overhead: float  # communication / compute per update
    compute_share: float
    communication_share: float
    speedup: float
    efficiency: float
    global_batch: int
    training_time: Quantity
    cost: Optional[float] = None  # dollars


class TensorParallelLayerResult(SolverResult):
    """Result from TensorParallelLayerModel: one layer's compute against its TP exchange."""

    __module__ = "mlsysim.engine.results"

    layer_flops: Quantity
    compute_time: Quantity
    layer_bytes: Quantity
    transfer_times: Dict[str, Quantity]


class AlignmentMemoryResult(SolverResult):
    """Result from AlignmentMemoryModel: PPO and DPO model state for RLHF."""

    __module__ = "mlsysim.engine.results"

    policy_state: Quantity
    reference_weights: Quantity
    reward_weights: Quantity
    value_state: Quantity
    ppo_total: Quantity
    dpo_total: Quantity
    dpo_reduction: float
    model_parallel_devices: int
    policy_per_device: Quantity
    reference_per_device: Quantity
    kv_cache: Quantity
    kv_per_device: Quantity
    min_devices_for_ppo: Optional[int] = None


class DenseNetworkResult(SolverResult):
    """Result from DenseNetworkModel: what one dense network costs at one batch size.

    Counts are plain integers; byte totals and intensity carry units. Per-layer
    lists run from the first layer to the output layer.
    """

    __module__ = "mlsysim.engine.results"

    layer_widths: List[int]
    batch_size: int
    layers: int
    weights_per_layer: List[int]
    biases_per_layer: List[int]
    params_per_layer: List[int]
    weights: int
    biases: int
    parameters: int
    macs_per_example: int  # one MAC per weight
    flops_per_example: int  # 2 FLOPs per MAC
    first_layer_share: float  # first layer's weights / all weights (= its share of the MACs)
    matmul_flops_per_layer: List[int]  # 2 B n_in n_out, at the batch
    elementwise_flops_per_layer: List[int]  # bias add + activation, 2 B n_out
    matmul_flops: int
    elementwise_flops: int
    forward_flops: int  # matmul + elementwise, at the batch
    forward_flops_per_example: float
    matmul_share: float
    elementwise_share: float
    parameter_bytes: Quantity
    traffic_values: int  # activation values crossing memory, at the batch
    traffic_bytes: Quantity  # weights once + activation traffic
    intensity: Quantity  # forward FLOPs / traffic bytes
    weight_gradient_flops: int
    input_gradient_flops: int
    training_step_flops: int  # forward matmul + both gradient products
    training_flops_ratio: float  # training step / forward matmul
    activation_values_per_layer: List[int]  # B n_l for the input and every layer output
    activation_values_per_example: int  # sum of widths
    saved_activation_values: int  # training keeps all of them, at the batch
    saved_activation_bytes: Quantity
    inference_activation_values: int  # largest adjacent pair, at the batch
    inference_activation_bytes: Quantity
    gradient_bytes: Quantity
    optimizer_bytes: Quantity
    training_memory: Quantity  # weights + saved activations + gradients + optimizer
    inference_memory: Quantity  # weights + the rolling activation pair
    training_memory_ratio: float
    steps_per_epoch: Optional[int] = None  # training examples // batch, when a dataset is given


class EpochOverheadResult(SolverResult):
    """Result from EpochOverheadModel: one training epoch's work and fixed overhead on a host and an accelerator.

    Dictionaries are keyed ``"host"`` and ``"accelerator"``.
    """

    __module__ = "mlsysim.engine.results"

    steps: int
    step_flops: Quantity
    ops_per_step: int
    peak: Dict[str, Quantity]
    bandwidth: Dict[str, Quantity]
    ridge: Dict[str, Quantity]
    rate_batch1: Dict[str, Quantity]  # attainable rate at the batch-1 intensity
    rate: Dict[str, Quantity]  # attainable rate at the network's batch
    batching_speedup: Dict[str, float]  # rate / rate_batch1
    work_time: Dict[str, Quantity]
    overhead_time: Dict[str, Quantity]
    total_time: Dict[str, Quantity]
    work_speedup: float  # host work / accelerator work
    overhead_share_accelerator: float
    movement_to_arithmetic_energy: float  # DRAM energy per FLOP's bytes / FLOP energy, at the batch
    movement_to_arithmetic_energy_batch1: float


class TargetMaskResult(SolverResult):
    """Result from TargetMaskModel: one serialized example's positions under two target masks."""

    __module__ = "mlsysim.engine.results"

    stored: int  # every stored token
    predicted: int  # stored - 1: the first position has no prediction
    selected_response: int  # responses minus the unselected ones
    targets_with_rationale: int  # selected responses plus the rationale
    targets_without_rationale: int  # selected responses only
    observation_share: float  # observations / stored
    share_with_rationale: float  # targets / stored, rationale selected
    share_without_rationale: float
    context_with_rationale: int  # stored positions that are context only
    context_without_rationale: int
    other_context_share: float  # non-observation context / stored, rationale as context
    dilution_without_rationale: float  # predicted / targets: weight lost by an unmasked token average
    dilution_with_rationale: float
    unselected_vs_selected: float  # unselected responses / selected responses
    unselected_share_of_responses: float  # unselected responses / all responses


class PerInvocationSerializationResult(SolverResult):
    """Result from PerInvocationSerializationModel: one example per call against one transcript."""

    __module__ = "mlsysim.engine.results"

    per_invocation_tokens: int  # sum over calls of each call's delivered prefix
    transcript_tokens: int  # the whole trajectory once
    expansion: float  # per-invocation / transcript = (H + 1) / 2


class LossNormalizationResult(SolverResult):
    """Result from LossNormalizationModel: one long trajectory's weight in a batch, two averages."""

    __module__ = "mlsysim.engine.results"

    examples: int  # records in the batch, the trajectory's pieces included
    total_targets: int
    focus_token_share: float  # token averaging
    other_token_share: float  # one other example, token averaging
    focus_example_share: float  # example averaging, all of the trajectory's pieces
    other_example_share: float  # one other example, example averaging
    focus_over_other: float  # target-count ratio: the token-averaging weight ratio


class SequencePackingResult(SolverResult):
    """Result from SequencePackingModel: padding every example to the longest against packing."""

    __module__ = "mlsysim.engine.results"

    examples: int
    stored: int
    longest: int
    mean_length: float
    padded_positions: int  # examples x longest
    buffers: int  # fixed-capacity buffers after first-fit decreasing
    buffer_loads: List[int]  # stored tokens per buffer
    packed_positions: int  # buffers x capacity
    padded_occupancy: float  # stored / padded positions
    packed_occupancy: float  # stored / packed positions
    position_ratio: float  # padded / packed positions


class PreferencePairResult(SolverResult):
    """Result from PreferencePairModel: one preference pair under the DPO loss."""

    __module__ = "mlsysim.engine.results"

    score_preferred: float  # z_w: policy minus reference log-probability sum
    score_dispreferred: float  # z_l
    margin: float  # beta (z_w - z_l)
    preference_score: float  # sigma(margin)
    loss: float  # -log sigma(margin)
    initial_loss: float  # ln 2, at a zero margin
    saturation_factor: float  # sigma(-margin)
    initial_saturation_factor: float  # 1/2
    gradient_coefficient: float  # beta sigma(-margin)
    initial_gradient_coefficient: float  # beta / 2


class RewardCompositionResult(SolverResult):
    """Result from RewardCompositionModel: true and false accepts among one update's rollouts."""

    __module__ = "mlsysim.engine.results"

    rollouts: int
    true_accepts: float  # expected passing rollouts the verifier accepts
    false_accepts: float  # expected failing rollouts the verifier accepts
    accepts: float
    false_share: float  # false accepts / all accepts


class CenteredRewardResult(SolverResult):
    """Result from CenteredRewardModel: binary rewards centered on a fixed baseline."""

    __module__ = "mlsysim.engine.results"

    passing_score: float  # passing reward - baseline
    failing_score: float  # failing reward - baseline


class ForkValueResult(SolverResult):
    """Result from ForkValueModel: a turn valued by forking continuations before and after it."""

    __module__ = "mlsysim.engine.results"

    value_before: float
    value_after: float
    difference: float
    se_bound: float  # 1 / (2 sqrt k), the largest standard error of one fork's mean
    difference_se_bound: float  # sqrt 2 times it, for two independent means
    resolution_threshold: float  # two difference bounds
    resolved: bool


class ContinuationForkCostResult(SolverResult):
    """Result from ContinuationForkCostModel: tokens generated when continuations are forked."""

    __module__ = "mlsysim.engine.results"

    base_tokens: int
    branches: int
    total_tokens: int
    multiple: float  # total / base


class GroupCoefficientsResult(SolverResult):
    """Result from GroupCoefficientsModel: group-relative (GRPO) coefficients of one group."""

    __module__ = "mlsysim.engine.results"

    group: int
    mean: float
    std: float
    coefficients: List[float]
    coefficient_sum: float


class LengthNormalizationResult(SolverResult):
    """Result from LengthNormalizationModel: per-token push of two rollouts sharing one coefficient."""

    __module__ = "mlsysim.engine.results"

    per_token_ratio: float  # short rollout's per-token push over the long one's


class RolloutStepBudgetResult(SolverResult):
    """Result from RolloutStepBudgetModel: one update's rollout bill and the cost of reward contrast."""

    __module__ = "mlsysim.engine.results"

    rollouts: int
    rollout_tokens: int
    step_tokens: int
    contrast: float  # P(at least one pass and one failure in a group)
    silent_share: float  # groups whose rewards all tie
    rollouts_per_informative_group: float


class RepeatedVerificationResult(SolverResult):
    """Result from RepeatedVerificationModel: an all-pass rule over repeated trials of one candidate."""

    __module__ = "mlsysim.engine.results"

    group_false_pass: float  # one trial each: P(at least one incorrect candidate passes)
    incorrect_accepted: float  # p_bad^n
    correct_rejected: float  # 1 - p_good^n
    expected_trials_incorrect: float  # stop at the first failed trial
    expected_trials_correct: float


class RatioClipResult(SolverResult):
    """Result from RatioClipModel: how far one update may raise a token's probability."""

    __module__ = "mlsysim.engine.results"

    ceiling: float  # (1 + epsilon) p
    rise: float  # epsilon p
    binds: bool  # the ceiling is below 1


class RolloutSlotCycleResult(SolverResult):
    """Result from RolloutSlotCycleModel: one sandbox slot's episode cycle under two reset times."""

    __module__ = "mlsysim.engine.results"

    work: Quantity  # decisions, tools, and assessment per episode
    cold_cycle: Quantity
    snapshot_cycle: Quantity
    cold_work_share: float
    snapshot_work_share: float
    cold_episodes_per_hour: float  # across all slots
    snapshot_episodes_per_hour: float
    capacity_ratio: float  # snapshot over cold capacity
    reset_ratio: float  # cold reset over snapshot reset


class GroupBarrierResult(SolverResult):
    """Result from GroupBarrierModel: a synchronous group that waits for its slowest member."""

    __module__ = "mlsysim.engine.results"

    group: int
    fastest: Quantity
    barrier: Quantity  # the slowest member's latency
    mean: Quantity
    idle_share: float  # 1 - mean / barrier, of the slots reserved to the barrier


class VerifierPoolResult(SolverResult):
    """Result from VerifierPoolModel: verification workers by Little's law."""

    __module__ = "mlsysim.engine.results"

    arrival_rate: Quantity  # trajectories per second
    one_trial_busy: float  # workers busy with a single trial each
    busy: float  # workers busy with every trial
    slots: int  # provisioned at the target utilization
    utilization: float
