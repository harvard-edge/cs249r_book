"""Typed result models for agents calculations."""

from __future__ import annotations

from typing import Dict, Optional, List
from ...core.types import Quantity
from .base import SolverResult

__all__ = [
    "HorizonCompoundingResult",
    "AcceptedTaskCostResult",
    "AppendOnlyContextResult",
    "ContextAdmissionResult",
    "SamplingResult",
    "CheckerPosteriorResult",
    "ModelCallResult",
    "TrajectoryResendResult",
    "ProbeVsAnswerResult",
    "AllocationAxesResult",
    "BestOfNPoolResult",
    "VerifiedPoolResult",
    "ArgmaxNoiseResult",
    "ScopedRevisionResult",
    "PrunedSearchTreeResult",
    "SearchAccountsResult",
    "DeadlinePolicyResult",
    "ContextBudgetResult",
    "AttentionDilutionResult",
    "PrefixLayoutBillResult",
    "CompactionCadenceResult",
    "ContextInvalidationResult",
    "ProtectedFieldCheckResult",
    "RetrievalPayloadResult",
    "Bm25PairResult",
    "ExactScanFloorResult",
    "AnnFootprintResult",
    "GraphNeighborhoodResult",
    "IndexMaintenanceResult",
    "StorageAmplificationResult",
    "BlindRetryResult",
    "ObservationCarryResult",
    "ModelPollingResult",
    "ToolCatalogSelectionResult",
    "ToolkitGranularityResult",
    "SyscallTaxResult",
    "EnvelopeAmortizationResult",
    "WorkspaceProvisioningResult",
    "SandboxPoolReserveResult",
    "InjectionExposureResult",
    "PolicyComparisonResult",
    "ApprovalEscrowResult",
    "DeficitRoundRobinResult",
    "BoundedLiabilityAdmissionResult",
    "TrajectoryBudgetMeterResult",
    "SpanCriticalPathResult",
    "TraceRetentionResult",
    "SequentialCanaryResult",
]


class HorizonCompoundingResult(SolverResult):
    """Result from HorizonCompoundingModel: unchecked task success across horizons."""

    __module__ = "mlsysim.engine.results"

    step_error: float  # 1 - per-step success
    success: Dict[int, float]  # horizon (turns) -> unchecked task success
    required_step_error: float  # per-step error that holds the target at the long horizon
    error_ratio: float  # step_error / required_step_error
    required_error_horizon: float  # required_step_error x long horizon
    small_error_limit: float  # -ln(target): the same product as eps -> 0
    first_horizon_below_target: int  # first horizon whose success falls below the target


class AcceptedTaskCostResult(SolverResult):
    """Result from AcceptedTaskCostModel: cost per accepted task and two levers on it (dollars)."""

    __module__ = "mlsysim.engine.results"

    attempt_cost: float
    token_share: float  # model-token share of the attempt cost
    attempts_per_acceptance: float  # 1 / p_acc
    cost_per_accepted: float
    accepted_only_understatement: float  # cost_per_accepted / attempt_cost
    cheaper_attempt_cost: Optional[float] = None
    cost_per_accepted_cheaper_tokens: Optional[float] = None
    cost_per_accepted_improved: Optional[float] = None
    break_even_acceptance: Optional[float] = None
    cost_per_accepted_cheaper_other: Optional[float] = None
    cost_per_accepted_acceptance_gain: Optional[float] = None
    independent_success: Optional[float] = None
    grown_attempt_cost: Optional[float] = None  # model cost grown by model_cost_growth (added 2026-10-20)
    grown_attempt_rise: Optional[float] = None  # grown attempt / attempt - 1
    grown_break_even_acceptance: Optional[float] = None  # acceptance that holds cost per accepted task


class AppendOnlyContextResult(SolverResult):
    """Result from AppendOnlyContextModel: what an append-only trajectory reads and resends."""

    __module__ = "mlsysim.engine.results"

    last_call_tokens: Quantity
    window_fraction: float  # last call / context window
    cumulative_input_tokens: Quantity
    reread_ratio: float  # cumulative input / last call


class ContextAdmissionResult(SolverResult):
    """Result from ContextAdmissionModel: token-measured admission, S <= S_max - K_max."""

    __module__ = "mlsysim.engine.results"

    input_ceiling: int  # window - reserved output
    headroom: int  # input ceiling - text prompt (negative: no room)
    total_demand: int  # text + image + reserved output
    excess: int  # total demand - window (positive: rejected)
    admitted: bool


class SamplingResult(SolverResult):
    """Result from SamplingModel: one logit vector at two temperatures, then top-p."""

    __module__ = "mlsysim.engine.results"

    partition_sum: float  # sum of exp(z / T) at the base temperature
    base_probabilities: List[float]
    sharp_probabilities: List[float]
    base_tail: float  # 1 - p(top token) at the base temperature
    sharp_tail: float
    tail_ratio: float  # base tail / sharp tail
    nucleus_size: int
    nucleus_mass: float
    nucleus_probabilities: List[float]


class CheckerPosteriorResult(SolverResult):
    """Result from CheckerPosteriorModel: defects among candidates a fallible check passes."""

    __module__ = "mlsysim.engine.results"

    defective_pass: float  # joint P(Def and PASS)
    pass_share: float  # P(PASS)
    posterior: float  # P(Def | PASS)
    catch_rate: float  # P(FAIL | Def)
    valid_reject_rate: float  # P(FAIL | Valid)


class ModelCallResult(SolverResult):
    """Result from ModelCallModel: prefill and decode work of one call at stated floors."""

    __module__ = "mlsysim.engine.results"

    prompt_tokens: int
    output_tokens: int
    prefill_time: Quantity
    decode_time: Quantity
    total_time: Quantity
    output_token_share: float
    output_time_share: float
    fresh_prefill_time: Quantity  # prefill of the tokens past a reused shared prefix
    decode_rate: Quantity  # output tokens per second at the decode floor
    deadline_margin: Optional[float] = None  # deadline / total time
    acceptance: Optional[float] = None  # product of the stated stage success rates


class TrajectoryResendResult(SolverResult):
    """Result from TrajectoryResendModel: input a trajectory resends, with and without prefix reuse."""

    __module__ = "mlsysim.engine.results"

    turns: int
    first_prompt_term: int  # H * S0
    appended_term: int  # D * H (H - 1) / 2
    resent_tokens: int  # their sum: input sent without reuse
    reused_prefix_tokens: int  # fresh prefill with prefix reuse
    reuse_reduction: float  # resent / reused
    output_tokens: int
    input_share: float  # resent / (resent + output)
    prefill_time: Quantity  # resent tokens at the prefill floor
    reused_prefill_time: Quantity
    decode_time: Quantity


class ProbeVsAnswerResult(SolverResult):
    """Result from ProbeVsAnswerModel: a generated explanation against a prepared test."""

    __module__ = "mlsysim.engine.results"

    token_time: Quantity  # decode floor per output token
    answer_time: Quantity  # answer_tokens * token_time
    answer_cost: Quantity  # USD of output tokens
    time_ratio: float  # answer_time / probe_time


class AllocationAxesResult(SolverResult):
    """Result from AllocationAxesModel: one output ceiling spent as depth, breadth, or feedback."""

    __module__ = "mlsysim.engine.results"

    token_time: Quantity
    depth_input: int
    depth_output: int
    depth_time: Quantity
    depth_cost: Quantity
    breadth_input: int
    breadth_output: int
    breadth_time: Quantity  # concurrent streams: one candidate's generation time
    breadth_cost: Quantity
    feedback_input: int  # append-only: every round resends earlier candidates and reports
    feedback_output: int
    feedback_time: Quantity  # generation plus every check
    feedback_cost: Quantity
    path_ratio: float  # depth_time / breadth_time
    bill_ratio: float  # breadth_cost / depth_cost
    breadth_premium: int  # (N - 1) S re-read prompt tokens
    serial_breadth_time: Quantity  # breadth when one stream runs at a time
    grown_premium: Optional[int] = None  # premium at a grown prompt
    grown_premium_cost: Optional[Quantity] = None


class BestOfNPoolResult(SolverResult):
    """Result from BestOfNPoolModel: pass@N, parallel pool against sequential retry, mixed workload."""

    __module__ = "mlsysim.engine.results"

    pass_at: Dict[int, float]  # pool size -> 1 - (1 - p)^N
    last_gain: float  # what the largest pool's last candidate adds
    parallel_per_success: float  # candidates per success, parallel pool of the largest size
    sequential_candidates: float  # expected candidates, stop at the first pass, same cap
    sequential_per_success: float  # reduces to 1 / p
    cost_ratio: float  # parallel / sequential candidates per success
    mixed_average_rate: float  # the workload's average single-sample rate
    mixed_plug_in: float  # pass@N at the averaged rate (wrong)
    mixed_true: float  # pass@N per task, then averaged
    easy_pass_at_max: float  # the easy half's pass@N


class VerifiedPoolResult(SolverResult):
    """Result from VerifiedPoolModel: first-passing selection from a fixed pool under a fallible check."""

    __module__ = "mlsysim.engine.results"

    valid_pass: float  # v (1 - eps_FR)
    invalid_pass: float  # (1 - v) eps_FA
    pass_probability: float  # q, a candidate passes
    precision: float  # passing-set precision
    defect_given_selection: float
    any_selection: float  # 1 - (1 - q)^n
    no_selection: float  # abstention
    wrong_selection: float  # defective selection per attempt
    any_false_pass: float  # the batch holds at least one false pass
    expected_valid_passes: float
    expected_invalid_passes: float
    expected_passes: float
    conditional_loss: Quantity  # loss per selection
    task_loss: Quantity  # loss per attempt
    output_bill: Quantity
    loss_to_bill: float


class ArgmaxNoiseResult(SolverResult):
    """Result from ArgmaxNoiseModel: the winner's scoring error and a loophole's presence in a pool."""

    __module__ = "mlsysim.engine.results"

    expected_max: Dict[int, float]  # pool size -> E[max of N standard normals]
    asymptote: Dict[int, float]  # pool size (N > 1) -> sqrt(2 ln N)
    exploit_presence: float  # 1 - (1 - rate)^N at the large pool


class ScopedRevisionResult(SolverResult):
    """Result from ScopedRevisionModel: output of a full against a scoped regeneration."""

    __module__ = "mlsysim.engine.results"

    full_output: int
    scoped_output: int
    output_share: float
    avoided_share: float


class PrunedSearchTreeResult(SolverResult):
    """Result from PrunedSearchTreeModel: tree size with and without a step check."""

    __module__ = "mlsysim.engine.results"

    effective_branching: float  # b (1 - r)
    critical_rejection: float  # 1 - 1/b: the tree dies out above it
    unpruned_nodes: int
    pruned_nodes: float
    unpruned_leaves: int
    pruned_leaves: float
    node_saving: float
    flat_cut_nodes: float  # the unpruned count cut by r once (a reference)
    wide_leaves: Optional[int] = None


class SearchAccountsResult(SolverResult):
    """Result from SearchAccountsModel: flat best-of-N against beam search, four accounts."""

    __module__ = "mlsysim.engine.results"

    token_time: Quantity
    frontier_counts: List[int]  # extensions generated per beam stage
    beam_segments: int
    retained: int
    flat_output_tokens: int
    beam_output_tokens: int
    flat_prompt_tokens: int
    beam_prompt_tokens: int
    beam_common_prompt_tokens: int  # the task prompt alone, once per extension
    scorer_tokens: int
    flat_waves: int
    beam_waves: int
    flat_test_waves: int
    beam_test_waves: int
    beam_scoring_time: Quantity
    flat_elapsed: Quantity
    beam_elapsed: Quantity
    flat_token_cost: Quantity
    beam_token_cost: Quantity
    output_saving: float
    output_saving_cost: Quantity
    scorer_cost: Quantity
    input_premium_cost: Quantity
    breakeven_input_share: float  # input price fraction at which the bills match


class DeadlinePolicyResult(SolverResult):
    """Result from DeadlinePolicyModel: one candidate, a parallel pool, and feedback under one deadline."""

    __module__ = "mlsysim.engine.results"

    token_time: Quantity
    round_time: Quantity
    rounds: int  # feedback rounds that fit the deadline (capped)
    requested_time: Quantity  # the requested rounds, deadline ignored
    revision_input: int
    a_pass: float
    b_pass: float
    c_pass: float
    c_rounds: float  # expected rounds of the feedback policy
    a_input: int
    b_input: int
    c_input: float
    a_output: int
    b_output: int
    c_output: float
    a_cost: Quantity
    b_cost: Quantity
    c_cost: Quantity
    c_time: Quantity
    max_c_time: Quantity
    a_cost_per_pass: Quantity
    b_cost_per_pass: Quantity
    c_cost_per_pass: Quantity
    a_output_per_pass: float
    b_output_per_pass: float
    c_output_per_pass: float
    b_over_a: float


class ContextBudgetResult(SolverResult):
    """Result from ContextBudgetModel: a request budgeted by role, appended against selected."""

    __module__ = "mlsysim.engine.results"

    input_limit: int  # window - output reserve
    fixed_tokens: int  # system + tools + task state
    evidence_tokens: int  # input limit - system - tools: state plus observations
    evidence_share: float  # evidence tokens / window
    observation_budget: int  # input limit - fixed tokens
    turns_fit: int  # appended turns that fit the observation budget
    overflow_turn: int  # first turn that does not fit
    overflow_observations: int  # observations the overflow turn needs
    digest_turns: int  # appended turns that fit at the digest size
    max_observation_per_turn: int  # largest per-turn observation that lasts every turn
    selected_request: int  # fixed tokens + selected observations
    append_requests: List[int]  # request of each turn when every result is appended
    append_cumulative: List[int]  # running input sent by appending
    select_cumulative: List[int]  # running input sent by selecting
    append_bill: int  # input tokens over the turns, appending
    select_bill: int  # input tokens over the turns, selecting
    bill_ratio: float  # append bill / select bill
    final_append_request: int  # the appending policy's last request
    final_append_with_reserve: int  # the same plus the output reserve


class AttentionDilutionResult(SolverResult):
    """Result from AttentionDilutionModel: one head's weight on a target among fixed-logit keys."""

    __module__ = "mlsysim.engine.results"

    short_weight: float  # target weight among the short context's keys
    long_weight: float  # target weight among the long context's keys
    weight_without_lookalikes: Optional[float] = None  # unrelated keys only
    weight_with_lookalikes: Optional[float] = None  # unrelated keys plus look-alike keys
    lookalike_equivalent_keys: Optional[float] = None  # unrelated keys that weigh as much as the look-alikes
    weight_half_unrelated: Optional[float] = None  # half the unrelated keys dropped, look-alikes kept
    weight_no_unrelated: Optional[float] = None  # every unrelated key dropped, look-alikes kept


class PrefixLayoutBillResult(SolverResult):
    """Result from PrefixLayoutBillModel: input bill of a trajectory, volatile first against zoned."""

    __module__ = "mlsysim.engine.results"

    request_tokens: int  # root + trunk + leaf, every call
    total_input: int  # request tokens x calls
    warm_calls: int  # calls that change only the leaf
    uncached_tokens: int  # zoned layout's prefill
    cached_tokens: int  # zoned layout's reused prefix tokens
    hit_rate: float  # cached / total input
    unstable_cost: Quantity  # USD, no prefix reused
    stable_cost: Quantity  # USD, zoned layout
    cost_ratio: float  # unstable / stable
    prefill_ratio: float  # total input / uncached tokens
    unstable_cost_per_call: Quantity
    stable_cost_per_call: Quantity
    rewrite_tokens: int  # trunk + leaf: the prefill of one trunk rewrite
    rewrite_in_leaf_calls: float  # rewrite tokens / leaf tokens


class CompactionCadenceResult(SolverResult):
    """Result from CompactionCadenceModel: passes between two watermarks over a turn budget."""

    __module__ = "mlsysim.engine.results"

    input_limit: int  # window - output reserve
    cycle_turns: int  # turns between passes
    trigger_length: int  # request length when a pass fires
    overshoot: int  # trigger length - trigger watermark
    length_with_reserve: int  # trigger length + output reserve
    fits_window: bool
    passes: int  # passes within the turn budget
    lengths: List[int]  # request length after each turn's append, before any pass
    alternate_cycle_turns: Optional[int] = None  # the same at the alternate target
    alternate_passes: Optional[int] = None
    alternate_lengths: Optional[List[int]] = None
    frequency_ratio: Optional[float] = None  # cycle / alternate cycle


class ContextInvalidationResult(SolverResult):
    """Result from ContextInvalidationModel: remove, replace, or append a notice for stale text."""

    __module__ = "mlsysim.engine.results"

    removed_total: int  # resident tokens after removal
    removed_suffix: int  # tokens after the reused prefix
    replaced_total: int  # resident tokens after an in-place marker
    replaced_suffix: int
    appended_total: int  # resident tokens after an appended notice
    appended_suffix: int
    suffix_ratio: float  # replaced suffix / appended suffix


class ProtectedFieldCheckResult(SolverResult):
    """Result from ProtectedFieldCheckModel: cost per accepted task with and without a check."""

    __module__ = "mlsysim.engine.results"

    exposures: int  # constraints x rounds
    survival_unchecked: float  # every constraint survives every round
    acceptance_unchecked: float
    cost_per_accepted_unchecked: float  # USD
    undetected_loss: float  # per-round loss the check misses
    survival_checked: float
    acceptance_checked: float
    attempt_cost_checked: float  # USD: attempt plus one check per round
    cost_per_accepted_checked: float  # USD
    checked_rounds_below_floor: int  # first round count at which checked survival falls below the floor
    max_loss_for_target: Optional[float] = None  # per-round loss that holds the target survival


class RetrievalPayloadResult(SolverResult):
    """Result from RetrievalPayloadModel: whole serialized records in a retrieval allowance."""

    __module__ = "mlsysim.engine.results"

    allowance: int  # window - output reserve - other input
    record: int  # body + metadata, tokens per serialized record
    records: int  # whole records that fit after the envelope
    used: int  # envelope + records x record
    remaining: int  # allowance - used
    body_only: int  # records a body-only count admits
    body_only_used: int  # those records serialized with metadata and envelope
    overshoot: int  # body_only_used - allowance
    overcount: int  # body_only - records
    scaled_metadata: Optional[int] = None  # metadata after the stated scale
    scaled_metadata_records: Optional[int] = None  # whole records at that metadata size


class Bm25PairResult(SolverResult):
    """Result from Bm25PairModel: one query term scored in a short and a long document."""

    __module__ = "mlsysim.engine.results"

    idf: float
    short_normalizer: float  # k1 (1 - b + b |D| / avgdl), short document
    long_normalizer: float
    short_score: float
    long_score: float


class ExactScanFloorResult(SolverResult):
    """Result from ExactScanFloorModel: the bandwidth floor of an exact vector scan."""

    __module__ = "mlsysim.engine.results"

    footprint: Quantity  # bytes streamed per scan
    bytes_per_vector: Quantity
    flops: Quantity  # 2 N d
    intensity: Quantity  # FLOP per byte
    ridge: Quantity  # host peak / bandwidth, FLOP per byte
    stream_time: Quantity  # footprint / bandwidth
    math_time: Quantity  # flops / peak
    trajectory_time: Optional[Quantity] = None  # searches x stream time
    filtered_time: Optional[Quantity] = None  # stream time of the filtered set
    budget_vectors: Optional[float] = None  # vectors the budget can stream


class AnnFootprintResult(SolverResult):
    """Result from AnnFootprintModel: resident bytes of a graph index and of PQ codes."""

    __module__ = "mlsysim.engine.results"

    raw: Quantity  # full-precision vectors
    graph: Quantity  # graph links
    graph_total: Quantity  # vectors + links
    codes: Quantity  # PQ codes
    compression: float  # raw / codes
    lookup_table: Quantity  # subvectors x centroids x entry
    sub_dim: int  # components per subvector
    bytes_per_vector: Quantity  # full precision
    code_bytes_per_vector: Quantity


class GraphNeighborhoodResult(SolverResult):
    """Result from GraphNeighborhoodModel: an exact tree's neighborhood against an allowance."""

    __module__ = "mlsysim.engine.results"

    allowance: int
    nodes_by_depth: List[int]  # cumulative nodes through depth 0, 1, ..., depth
    tokens_by_depth: List[int]  # serialized tokens through each depth
    nodes: int  # through the stated depth
    required: int  # tokens through the stated depth
    fit: int  # most whole nodes the allowance holds
    packed: int  # tokens of that packing
    shallow_nodes: int  # one level less
    shallow_required: int


class IndexMaintenanceResult(SolverResult):
    """Result from IndexMaintenanceModel: re-indexing an edit against excluding stale entries."""

    __module__ = "mlsysim.engine.results"

    chunks: int
    batches: int
    local_time: Quantity  # parse + postings + local embedding batches
    remote_batch_time: Quantity  # parse + postings + one remote request per batch
    remote_per_file_time: Quantity  # parse + postings + one remote request per file
    exclusion_write_time: Quantity  # one exclusion entry per changed file
    exclusion_check_time: Quantity  # one lookup per candidate
    write_path_ratio: float  # local re-index / exclusion write


class StorageAmplificationResult(SolverResult):
    """Result from StorageAmplificationModel: auxiliary bytes and re-index time of two stores."""

    __module__ = "mlsysim.engine.results"

    raw: Quantity  # source bytes
    postings: Quantity
    graph: Quantity  # structural edges
    chunks: int
    vectors: Quantity
    vector_links: Quantity  # graph-index links over the vectors
    aux_lexical_graph: Quantity  # postings + edges
    aux_lexical_dense: Quantity  # postings + vectors + links
    ratio_lexical_graph: float  # auxiliary / source
    ratio_lexical_dense: float
    edited_chunks: int
    reindex_lexical_graph: Quantity
    reindex_lexical_dense: Quantity
    bytes_ratio: float  # dense store / graph store, auxiliary bytes
    time_ratio: float  # dense store / graph store, re-index time


class BlindRetryResult(SolverResult):
    """Result from BlindRetryModel: timeouts and duplicate effects under one blind retry."""

    __module__ = "mlsysim.engine.results"

    timeout_probability: float  # 1 - (1 - q)(1 - r)
    duplicate_probability: float  # (1 - q) r (1 - q)
    calls: int  # calls in the period
    timeouts: float  # expected timeouts in the period
    duplicates: float  # expected duplicate effects in the period


class ObservationCarryResult(SolverResult):
    """Result from ObservationCarryModel: a raw log against the shaped view later calls carry."""

    __module__ = "mlsysim.engine.results"

    raw_tokens: Quantity  # raw bytes / bytes per token
    raw_windows: float  # raw tokens / window
    shaped_share: float  # shaped tokens / window
    carried_tokens: Quantity  # shaped tokens x later calls
    carried_cost_uncached: Quantity  # USD
    carried_cost_cached: Quantity  # USD, at the cached share of the price


class ModelPollingResult(SolverResult):
    """Result from ModelPollingModel: status calls a model makes while a job is pending."""

    __module__ = "mlsysim.engine.results"

    polls: int  # ceil(T / dt) - 1
    input_tokens: int
    output_tokens: int
    cached_price: Quantity  # USD per cached input token
    blended_input_price: Quantity  # USD per input token at the cache hit share
    input_cost: Quantity  # USD
    output_cost: Quantity  # USD
    cost: Quantity  # USD, input + output
    worst_delay: Quantity  # one poll interval


class ToolCatalogSelectionResult(SolverResult):
    """Result from ToolCatalogSelectionModel: descriptor tokens staged per call, full against selected."""

    __module__ = "mlsysim.engine.results"

    full_tokens: int  # registered x per descriptor
    selected_descriptor_tokens: int  # selected x per descriptor
    selected_tokens: int  # selected descriptors + overhead
    released_tokens: int  # full - selected
    unused_descriptor_tokens: int  # descriptors selection leaves out
    break_even_unused: float  # overhead / per descriptor


class ToolkitGranularityResult(SolverResult):
    """Result from ToolkitGranularityModel: structured tools against one shell tool, per accepted task."""

    __module__ = "mlsysim.engine.results"

    structured_descriptor_tokens: int  # tools x tokens, per call
    saved_tokens: int  # per trajectory
    saved_cost: Quantity  # USD per trajectory
    shell_attempt_cost: Quantity  # structured attempt - savings
    structured_cost_per_accepted: Quantity
    shell_cost_per_accepted: Quantity
    structured_cost_per_accepted_with_loss: Quantity
    shell_cost_per_accepted_with_loss: Quantity
    break_even_change_probability: float  # shell change probability matching structured


class SyscallTaxResult(SolverResult):
    """Result from SyscallTaxModel: per-call mediation cost and added time by envelope."""

    __module__ = "mlsysim.engine.results"

    container_call: Quantity  # trap + in-kernel filter
    microvm_call: Quantity  # native trap in the guest
    userspace_call: Quantity  # trap + emulation + switch
    container_total: Quantity  # over all ordinary calls
    microvm_total: Quantity
    userspace_total: Quantity
    userspace_vs_microvm: float  # per-call ratio


class EnvelopeAmortizationResult(SolverResult):
    """Result from EnvelopeAmortizationModel: setup paid per call or once per trajectory."""

    __module__ = "mlsysim.engine.results"

    useful_work: Quantity  # calls x work per call
    per_call_setup: Quantity  # calls x setup
    per_call_total: Quantity  # fresh envelope per call
    per_trajectory_total: Quantity  # one envelope retained
    setup_saved: Quantity
    reuse_speedup: float  # per-call total / per-trajectory total
    setup_share: float  # setup / (setup + work) of a fresh call
    work_for_cap: Quantity  # work per call that holds setup to the cap


class WorkspaceProvisioningResult(SolverResult):
    """Result from WorkspaceProvisioningModel: full copy against copy-on-write, and first-write granularity."""

    __module__ = "mlsysim.engine.results"

    full_copy: Quantity  # workspaces x base image
    cow_written: Quantity  # workspaces x bytes written
    full_copy_time: Quantity  # lower bound at the drive's sequential rate
    cow_time: Quantity
    copy_vs_cow: float  # base image / bytes written
    file_copy: Quantity  # workspaces x fixture
    block_copy: Quantity  # workspaces x one block
    logical_changes: Quantity  # workspaces x bytes modified
    file_vs_block: float  # fixture / block


class SandboxPoolReserveResult(SolverResult):
    """Result from SandboxPoolReserveModel: active leases, standby reserve, and resident memory."""

    __module__ = "mlsysim.engine.results"

    active: float  # Little's law, arrival rate x hold time
    restore_arrivals: float  # arrivals during one restore (Poisson mean)
    initial_miss: float  # miss probability with the initial reserve
    reserve: int  # smallest reserve meeting the target
    reserve_miss: float
    private_memory: Quantity  # every active lease with a private image
    pool_memory: Quantity  # copy-on-write: one image + dirty pages + standby
    memory_ratio: float
    max_restore: Quantity  # longest restore the reserve tolerates
    slow_arrivals: float  # arrivals during a slower restore
    slow_reserve: int


class InjectionExposureResult(SolverResult):
    """Result from InjectionExposureModel: per-task payload exposure by read volume."""

    __module__ = "mlsysim.engine.results"

    exposure: Dict[str, float]  # 1 - (1 - q)^n per workload
    linear: Dict[str, float]  # q n, the small-exposure estimate
    screened_rate: float  # q m, payload rate behind the screen
    screened: Dict[str, float]  # 1 - (1 - q m)^n per workload
    screened_per_cohort: Dict[str, float]  # screened exposure x cohort tasks


class PolicyComparisonResult(SolverResult):
    """Result from PolicyComparisonModel: attack success and benign acceptance for two policies."""

    __module__ = "mlsysim.engine.results"

    baseline_lower: float  # confirmed harm / attacked
    candidate_lower: float
    baseline_upper: float  # (confirmed + unresolved) / attacked
    candidate_upper: float
    reduction_observed: float  # 1 - candidate_lower / baseline_lower
    reduction_upper: float  # both arms at their upper rates
    reduction_least_favorable: float  # candidate upper against baseline lower
    baseline_acceptance: float  # accepted benign / benign
    candidate_acceptance: float
    lost_benign: int
    extra_false_blocks: int
    cost_rise: float  # p_acc(baseline) / p_acc(candidate) - 1
    baseline_cost_per_accepted: Optional[float] = None  # attempt cost / p_acc, when priced
    candidate_cost_per_accepted: Optional[float] = None


class ApprovalEscrowResult(SolverResult):
    """Result from ApprovalEscrowModel: what approval escrow holds for two workloads."""

    __module__ = "mlsysim.engine.results"

    held: float  # repair trajectories in escrow, Little's law
    retained_memory: Quantity  # held x workspace lease, GiB
    remediation_held: float  # remediation trajectories in escrow
    time_to_mitigate: Quantity  # diagnosis + decision + apply and verify
    approval_share: float  # decision / time to mitigate


class DeficitRoundRobinResult(SolverResult):
    """Result from DeficitRoundRobinModel: charge shares against call shares, per class."""

    __module__ = "mlsysim.engine.results"

    credit_per_round: Dict[str, float]  # weight x quantum
    charge_per_call: Dict[str, float]  # input + output allowance
    calls_per_round: Dict[str, float]  # long-run, backlogged
    rounds_per_call: Dict[str, float]
    charge_share: Dict[str, float]  # weight / total weight
    call_share: Dict[str, float]
    call_ratio: float  # most calls per round / fewest
    simulated_rounds: int
    simulated_call_share: Dict[str, float]
    simulated_charge_share: Dict[str, float]


class BoundedLiabilityAdmissionResult(SolverResult):
    """Result from BoundedLiabilityAdmissionModel: a spend check against the reservation test (dollars)."""

    __module__ = "mlsysim.engine.results"

    input_liability: float
    output_liability: float
    liability: float  # most the call can bill
    spend_headroom: float  # allowance - settled
    headroom: float  # allowance - settled - reserved - finalization reserve
    spend_check_admits: bool
    admitted: bool
    finalization_shortfall: float  # reserve left unfunded if the call were admitted
    input_share: float  # input liability / liability
    max_input_tokens: float  # largest input that fits beside the output allowance
    tokens_per_probe: float  # input tokens one probe amount of unsettled spend costs
    reservation_share: float  # outstanding reservation / allowance


class TrajectoryBudgetMeterResult(SolverResult):
    """Result from TrajectoryBudgetMeterModel: one trajectory's used fractions, with and without prefix reuse."""

    __module__ = "mlsysim.engine.results"

    input_tokens: int
    cached_tokens: int
    uncached_tokens: int
    output_tokens: int
    total_tokens: int
    cost_with_reuse: float  # dollars
    cost_without_reuse: float
    cost_ratio: float  # without / with
    used_with_reuse: Dict[str, float]  # turns, tokens, time, dollars
    used_without_reuse: Dict[str, float]
    leading_with_reuse: str  # dimension with the largest used fraction
    leading_without_reuse: str


class SpanCriticalPathResult(SolverResult):
    """Result from SpanCriticalPathModel: what one trial's spans explain of its elapsed time."""

    __module__ = "mlsysim.engine.results"

    leaf_sum: Quantity  # every span's duration added, overlap counted twice
    critical_path: Quantity  # longest chain of dependent span durations
    elapsed: Quantity  # the enclosing segment
    unexplained: Quantity  # elapsed time no span on the path covers
    changed_elapsed: Quantity  # elapsed time after the intervention, waits kept
    saving: Quantity
    changed_critical: List[str]  # spans on the path after the intervention


class TraceRetentionResult(SolverResult):
    """Result from TraceRetentionModel: diagnostic trace bodies kept under three selections."""

    __module__ = "mlsysim.engine.results"

    head_count: float  # traces kept per cohort period, head sampling
    head_failed: float  # expected failed traces among them
    head_unresolved: float
    tail_control: float  # ordinary traces kept as the control sample
    tail_count: float  # verdict-aware selection
    full_volume: Quantity  # every body, over the window
    head_volume: Quantity
    tail_volume: Quantity
    full_over_tail: float
    floor_volume: Quantity  # failed and unresolved bodies alone
    per_point_volume: Quantity  # one percentage point of control sampling


class SequentialCanaryResult(SolverResult):
    """Result from SequentialCanaryModel: a Wald SPRT canary monitor in labels and calendar time."""

    __module__ = "mlsysim.engine.results"

    upper: float  # log boundary: hold (select the degraded hypothesis)
    lower: float  # log boundary: select the baseline hypothesis
    failure_step: float  # log-likelihood added by one failure
    success_step: float  # log-likelihood added by one success
    drift: float  # expected step under the degraded hypothesis
    terminal: float  # expected terminal log-likelihood, no overshoot
    expected_samples: float  # labeled tasks, under the degraded hypothesis
    failures_to_hold: int  # consecutive failures from the start that cross the upper boundary
    candidate_rate: Quantity  # candidate tasks admitted per hour
    admission_time: Quantity  # to admit the expected samples
    decision_time: Quantity  # plus the label delay
    pending: float  # candidate tasks admitted while labels are pending
    excess_failures: float  # measured regression's cost at this fraction
    full_pending: float
    full_excess_failures: float  # the same at full allocation
    exposure_share: float  # unmeasured-harm exposure against full allocation
