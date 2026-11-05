"""Typed result models for robotics calculations."""

from __future__ import annotations

from typing import Dict, Optional, List, Tuple
from ...core.types import Quantity
from .base import SolverResult

__all__ = [
    "StoppingBudgetResult",
    "ShieldCoverageBillResult",
    "TractionLossResult",
    "SpeedStepResult",
    "FreshnessProfileResult",
    "BrakingCeilingsResult",
    "HubDriveElectricalResult",
    "GearedRotorResult",
    "GearRatioCostResult",
    "ThermalBurstResult",
    "StallDeratingResult",
    "ContinuousDeratingResult",
    "DriveBusResult",
    "ZeroFailureResult",
    "ThrottledStageResult",
    "VlaEvaluationResult",
    "DecoderFamiliesResult",
    "ProposerTrafficResult",
    "EvidenceHorizonResult",
    "PhaseLagBudgetResult",
    "ResponseTimeResult",
    "CanFdBusResult",
    "EtherCatCycleResult",
    "LeaseBudgetResult",
    "SharedDramReadResult",
    "ThermalThrottleResult",
    "ObservationAgeResult",
    "RollingShutterSkewResult",
    "TimingAttitudeErrorResult",
    "LidarSweepAgeResult",
    "StereoDepthErrorResult",
    "DelayPhaseMarginResult",
    "ConformalInsetResult",
    "ClockConversionResult",
    "VisibilityRangeResult",
    "ThermalExpansionResult",
    "RangeFusionResult",
    "EstimatorConsistencyResult",
    "ReplayBufferResult",
    "InertialCoastResult",
    "OdometryDriftResult",
    "BeliefExpiryResult",
    "StaleTransformResult",
    "FreeSpaceExpiryResult",
    "EpochGapResult",
    "SensorBiasDriftResult",
    "FrozenTransformResult",
    "ClearanceQueryRow",
    "ClearanceQueryResult",
    "HostMapSizingResult",
    "RaymarchTrafficResult",
    "EncoderRateResult",
    "IntentCadenceResult",
    "DynamicReachResult",
    "ForwardOccupancyResult",
    "IntentLeaseExpiryResult",
    "ReferentSetResult",
    "ConfidenceRegionResult",
    "TwoLinkJointRateResult",
    "HandoffExposureResult",
    "ChunkReplacementResult",
    "JerkLimitedStopResult",
    "SpillLimitedStopResult",
    "TrajectoryDeadlinesResult",
    "SeamBridgeResult",
    "ShoulderSwingResult",
    "DynamicWindowResult",
    "StaleAnchorResult",
    "TakeoverBudgetResult",
    "TakeoverLadderResult",
    "BrakingBlendResult",
    "CommandAuthenticationResult",
    "OnsetTrialResult",
    "RailHoldUpResult",
    "ExponentialRareEventResult",
    "DiagnosticCoverageResult",
    "DetectionShortfallRow",
    "DetectionShortfallResult",
    "SupervisorQueueResult",
    "GripperTransferResult",
]


class StoppingBudgetResult(SolverResult):
    """Result from StoppingBudgetModel: the elementary stop against a clear distance."""

    __module__ = "mlsysim.engine.results"

    speed: Quantity  # speed when the hazard appears; braking starts from it
    delay: Quantity
    deceleration: Quantity
    delay_travel: Quantity
    braking_travel: Quantity
    stop_distance: Quantity  # delay travel + braking travel: travel to rest
    overhead: Quantity  # localization bound + protective margin (zero if unstated)
    required_distance: Quantity  # stop_distance + overhead: what the clear distance must hold
    braking_time: Quantity
    crossover_speed: Quantity  # 2 a tau: delay travel equals braking travel there
    braking_over_delay: Optional[float] = None
    delay_share_of_braking: Optional[float] = None
    clear_distance: Optional[Quantity] = None
    spare: Optional[Quantity] = None
    fits: Optional[bool] = None
    clear_over_stop: Optional[float] = None
    speed_ceiling: Optional[Quantity] = None  # fastest speed whose stop, with the overhead, fits
    added_delay_travel: Optional[Quantity] = None  # travel at ``speed`` during a stated delay step
    delay_ceiling: Optional[Quantity] = None  # longest constant-speed delay whose stop, with the overhead, fits
    delay_spare: Optional[Quantity] = None  # delay_ceiling - delay
    # The finished budget (added 2026-10-20): S-curve stop, tracking inset, approaching person
    jerk: Optional[Quantity] = None  # jerk bound of the resident S-curve stop
    jerk_ramp_travel: Optional[Quantity] = None  # S-curve stop travel beyond v^2 / 2a
    stop_time: Optional[Quantity] = None  # duration of the S-curve stop
    jerk_plateau_speed: Optional[Quantity] = None  # a^2 / j: slower stops never reach a
    ramp_delay: Optional[Quantity] = None  # a / (2 j): the ramps as added delay above a^2 / j
    tracking_inset: Optional[Quantity] = None  # tracking bound subtracted from the clearance
    approach_travel: Optional[Quantity] = None  # v_h (tau + T_stop): a person closing the gap
    # The ceiling solved by hand (added 2026-10-20): the S-curve stop alone and
    # the terms of the quadratic root, set with a jerk bound and a clear distance.
    scurve_travel: Optional[Quantity] = None  # braking + jerk-ramp travel: the resident stop alone
    rigid_speed_ceiling: Optional[Quantity] = None  # (D_clear - overhead - inset) / delay: instantaneous braking
    ceiling_lead_time: Optional[Quantity] = None  # b = delay + a / (2 j), on the quadratic branch only
    ceiling_root: Optional[Quantity] = None  # sqrt(b^2 + 2 (D_clear - overhead - inset) / a), same branch
    # The deceleration premise (added 2026-10-20): with a jerk bound and a clear distance,
    # the weakest deceleration whose S-curve stop still fits, on the trapezoidal branch.
    braking_room: Optional[Quantity] = None  # D_clear - overhead - inset - delay travel
    min_deceleration: Optional[Quantity] = None  # deceleration at which the stop fills braking_room exactly
    deceleration_shortfall: Optional[float] = None  # 1 - min_deceleration / deceleration

    def travel_over(self, duration: Quantity) -> Quantity:
        """Calculate travel at the stored speed over an added delay.

        Parameters
        ----------
        duration : Quantity
            Nonnegative added delay in seconds or compatible time units.

        Returns
        -------
        Quantity
            Distance traveled during duration, in meters.

        Notes
        -----
        Equation: distance = speed * duration.

        Assumptions: The stored speed remains constant throughout the added delay.
        The caller supplies a nonnegative delay; this accessor does not validate it.

        Excludes: Braking, acceleration, tracking error, and the other terms of the
        stopping budget. This is one delay contribution, not total stopping distance.

        References
        ----------
        Derivation: Integrating a constant speed over a duration gives speed times
        duration; the product is converted to meters.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import StoppingBudgetResult
        >>> r = StoppingBudgetResult(speed=Q_(2, "m/s"), delay=Q_(1, "s"),
        ...     deceleration=Q_(1, "m/s^2"), delay_travel=Q_(2, "m"),
        ...     braking_travel=Q_(2, "m"), stop_distance=Q_(4, "m"), overhead=Q_(0, "m"),
        ...     required_distance=Q_(4, "m"), braking_time=Q_(2, "s"),
        ...     crossover_speed=Q_(2, "m/s"))
        >>> r.travel_over(Q_(3, "s")) == Q_(6, "m")
        True
        """
        return (self.speed * duration).to("meter")


class ShieldCoverageBillResult(SolverResult):
    """Result from ShieldCoverageBillModel: what a shield's tested coverage buys against a hazard target
    (added 2026-10-20). Rates are per operating hour."""

    __module__ = "mlsysim.engine.results"

    exposure: Quantity  # adjudicated proposal-log hours
    ln_inv_alpha: float  # -ln(1 - C)
    proposal_rate_upper: Quantity  # zero-event bound on hazardous proposals
    residual_rate: Quantity  # proposal bound x (1 - coverage) + common cause
    meets_target: bool
    required_coverage: float  # coverage that would meet the target at this proposal bound
    proposal_rate_limit: Quantity  # proposal bound the target needs at the stated coverage
    exposure_needed: Quantity  # failure-free proposal hours that bound it
    days_needed: Quantity  # the same exposure on the fleet, calendar days
    days_beyond_cycle: Quantity
    coverage_trials: int  # seeded hazards, no miss, that bound coverage at C
    joint_confidence: float  # each of two one-sided bounds at 1 - alpha/2 (Bonferroni)
    coverage_trials_joint: int
    ln_inv_half_alpha: float  # -ln(alpha / 2)
    proposal_rate_upper_joint: Quantity
    residual_rate_joint: Quantity


class TractionLossResult(SolverResult):
    """Result from TractionLossModel: what a drop in floor friction does to braking."""

    __module__ = "mlsysim.engine.results"

    braking_distance_ratio: float  # braking distance at the traction limit, after / before
    traction_ceiling: Quantity  # deceleration the lower friction can support
    required_friction: float  # friction the stated deceleration needs
    premise_holds: bool  # the lower friction still supports it


class SpeedStepResult(SolverResult):
    """Result from SpeedStepModel: what raising a speed setpoint costs in braking travel."""

    __module__ = "mlsysim.engine.results"

    braking_growth: float  # raised braking / base braking - 1
    braking_added: Quantity
    equivalent_delay: Quantity  # added braking expressed as delay at the base speed


class FreshnessProfileResult(SolverResult):
    """Result from FreshnessProfileModel: a target's freshness deadline against an age profile."""

    __module__ = "mlsysim.engine.results"

    allowed_error: Quantity
    deadline: Quantity
    stage_sum_p50: Quantity
    stage_sum_p99: Quantity
    stage_sum_max: Quantity
    focus_stage: Quantity  # the stage a benchmark reports, at the median
    rest_of_chain: Quantity  # median chain minus that stage
    focus_share: float  # of the median chain
    focus_rate: Quantity  # loop rate the benchmark alone suggests
    travel_p50: Quantity
    travel_tail: Quantity
    slack_p50: Quantity
    overrun_tail: Quantity


class BrakingCeilingsResult(SolverResult):
    """Result from BrakingCeilingsModel: traction, tip-over, drive-torque, and back-EMF checks."""

    __module__ = "mlsysim.engine.results"

    wheel_torque: Quantity
    force: Quantity
    rotating_mass: Quantity
    rotating_share: float
    m_eff: Quantity
    a_torque: Quantity
    torque_margin: float
    a_torque_rated: Quantity
    mu_required: float  # all weight on the driven wheels
    drive_load_share: float  # upper bound from the load rating
    mu_required_rated: float  # at that load share
    min_load_share_dry: float
    a_traction_dry: Quantity
    a_traction_inspected: Quantity
    cg_ratio_max: float
    a_tip: Optional[Quantity] = None
    passes: bool
    wheel_speed: Quantity
    back_emf: Quantity
    v_brake: Quantity
    v_drive: Quantity
    headroom: Quantity


class HubDriveElectricalResult(SolverResult):
    """Result from HubDriveElectricalModel: current rise and phase-short braking of a hub drive."""

    __module__ = "mlsysim.engine.results"

    time_constant: Quantity
    ramp_time: Quantity
    ramp_share: float  # of the brake onset
    full_step_current: Quantity
    step_over_peak: float
    short_peak_electrical: Quantity  # rad/s
    short_peak_wheel_speed: Quantity  # rpm
    pole_frequency: Optional[Quantity] = None  # 1 / (2 pi tau_e), Hz: the winding's current pole


class GearedRotorResult(SolverResult):
    """Result from GearedRotorModel: reflected rotor inertia and the inertia match."""

    __module__ = "mlsysim.engine.results"

    reflected_inertia: Quantity
    flywheel_mass: Quantity
    radius_of_gyration: Quantity
    matched_ratio: float
    acceleration_share: float  # at the stated ratio, of the matched-ratio acceleration


class GearRatioCostResult(SolverResult):
    """Result from GearRatioCostModel: holding heat and reflected inertia per gear ratio."""

    __module__ = "mlsysim.engine.results"

    motor_constant: Quantity
    motor_torque: Dict[int, Quantity]
    copper_loss: Dict[int, Quantity]
    reflected_inertia: Dict[int, Quantity]
    loss_inertia_product: Dict[int, Quantity]
    over_peak: Dict[int, float]  # motor torque / motor peak torque
    peak_over_rated: float


class ThermalBurstResult(SolverResult):
    """Result from ThermalBurstModel: burst horizons, duty screen, and periodic peak."""

    __module__ = "mlsysim.engine.results"

    time_constant: Quantity
    rated_loss: Quantity
    rated_equilibrium: Quantity
    burst_current: Quantity
    burst_loss: Quantity
    asymptote: Quantity
    horizon_cold: Quantity
    horizon_hot: Quantity
    hot_cut: float
    average_loss_max: Quantity
    duty: float
    off_time: Quantity
    off_time_scheduled: Quantity
    periodic_peak: Quantity


class StallDeratingResult(SolverResult):
    """Result from StallDeratingModel: stall equilibrium with copper feedback, and the sustainable hold."""

    __module__ = "mlsysim.engine.results"

    motor_torque: Quantity
    stall_current: Quantity
    cold_loss: Quantity
    feedback_constant: Quantity
    feedback_gain: float
    steady_rise: Quantity
    winding_temperature: Quantity
    sustainable_loss: Quantity
    hot_resistance: Quantity
    sustainable_current: Quantity
    sustainable_hold: Quantity
    derating: float


class ContinuousDeratingResult(SolverResult):
    """Result from ContinuousDeratingModel: installed continuous torque against the catalog rating."""

    __module__ = "mlsysim.engine.results"

    ambient_share: float
    combined_share: float


class DriveBusResult(SolverResult):
    """Result from DriveBusModel: sustained sag, regenerative surge, and chopper sizing."""

    __module__ = "mlsysim.engine.results"

    compute_current: Quantity
    drive_current: Quantity
    sag: Quantity
    sustained_full: Quantity
    sustained_depleted: Quantity
    margin_depleted: Quantity
    headroom_depleted: Quantity
    crossing_current: Quantity
    kinetic_energy: Quantity
    regenerated: Quantity
    capacitor_only_endpoint: Quantity
    capacitor_to_clamp: Quantity
    stop_time: Quantity
    peak_power: Quantity
    resistor_max: Quantity
    crossing_resistance: Optional[Quantity] = (
        None  # source resistance at which the step reaches lockout (added 2026-10-20)
    )


class ZeroFailureResult(SolverResult):
    """Result from ZeroFailureModel: the failure rate a clean campaign supports."""

    __module__ = "mlsysim.engine.results"

    coefficient: float  # -ln(1 - C)
    rate_upper: float
    one_in: float
    # Exact binomial bound and the campaign a target needs (added 2026-10-20).
    rate_upper_exact: Optional[float] = None  # 1 - (1 - C)^(1/n)
    trials_for_target: Optional[int] = None  # ceil(ln(1 - C) / ln(1 - target))
    trials_for_target_approx: Optional[float] = None  # -ln(1 - C) / target (rule of three)


class ThrottledStageResult(SolverResult):
    """Result from ThrottledStageModel: a compute-bound stage slowed by a lower clock."""

    __module__ = "mlsysim.engine.results"

    throttled: Quantity
    added_age: Quantity
    added_travel: Quantity


class VlaEvaluationResult(SolverResult):
    """Result from VlaEvaluationModel: one fresh VLA evaluation and its roofline coordinates."""

    __module__ = "mlsysim.engine.results"

    weight_memory: Quantity  # stored weights at the evaluated precision
    sustained_bandwidth: Quantity  # peak bandwidth x bus efficiency
    achieved_rate: Quantity  # 16-bit peak x compute utilization
    ridge_intensity: Quantity  # 16-bit peak / peak bandwidth
    prefill_tokens: int
    decode_steps: int
    prefill_linear_flops: Quantity  # 2 N S
    prefill_attention_flops: Quantity  # the S^2 attention term
    prefill_flops: Quantity
    prefill_intensity: Quantity  # linear FLOPs per weight byte
    decode_intensity: Quantity  # 2 N FLOP per weight sweep, per weight byte
    t_prefill_compute: Quantity
    t_weight_sweep: Quantity
    weight_sweep_rate: Quantity  # 1 / t_weight_sweep: the weight-streaming ceiling
    t_prefill: Quantity
    prefill_bound: str  # "compute" or "memory"
    t_decode_step: Quantity
    t_decode: Quantity
    t_total: Quantity
    f_max: Quantity
    prefill_share: float  # t_prefill / t_total
    prefill_ceiling: Quantity  # 1 / t_prefill: no weight-only format passes it


class DecoderFamiliesResult(SolverResult):
    """Result from DecoderFamiliesModel: one fresh chunk per decoder family on one backbone."""

    __module__ = "mlsysim.engine.results"

    binned_tokens: int
    t_prefill: Quantity  # prefill only, shared by every family
    t_binned: Quantity
    t_fast: Quantity
    t_parallel: Quantity
    expert_weights: Quantity
    kv_per_token: Quantity  # full-width keys and values, all layers
    prefix_kv: Quantity  # cached KV of the input tokens
    t_expert_step_bandwidth: Quantity
    t_expert_step_compute: Quantity
    t_expert_step: Quantity
    t_expert: Quantity
    binned_over_expert: float
    prefill_over_period: float  # shared prefill / renewal period


class ProposerTrafficResult(SolverResult):
    """Result from ProposerTrafficModel: two proposers' weight traffic on one memory bus."""

    __module__ = "mlsysim.engine.results"

    sustained_bandwidth: Quantity
    intent_weights_fp16: Quantity
    intent_weights_deployed: Quantity
    chunk_weights: Quantity  # chunk policy, 16-bit
    intent_sweep_fp16: Quantity
    intent_sweep_rate_fp16: Quantity  # the 16-bit weight-streaming ceiling
    intent_sweep_deployed: Quantity
    intent_sweep_rate_deployed: Quantity
    amortized_sweep: Quantity  # 16-bit sweep / H
    chunk_sweep: Quantity
    chunk_sweep_rate: Quantity
    chunk_sweep_share: float  # chunk sweep x renewal rate
    tick_over_sweep_rate: float  # permission tick / 16-bit sweep rate
    setpoint_rate: Quantity
    single_step_demand: Quantity  # one 16-bit sweep per setpoint
    single_step_overload: float  # that demand / sustained bandwidth
    chunked_demand: Quantity  # one sweep per chunk of H
    chunked_share: float
    renewal_gap: float  # intent latency in chunk renewal periods
    sweeps_per_evaluation: int
    intent_traffic: Quantity
    chunk_traffic: Quantity
    proposer_traffic: Quantity
    proposer_share: float
    decode_duty: float  # intent decode time x intent rate


class EvidenceHorizonResult(SolverResult):
    """Result from EvidenceHorizonModel: a tracked target's evidence horizon against the chunk schedule."""

    __module__ = "mlsysim.engine.results"

    margin: Quantity  # tolerance - initial error
    untracked_deadline: Quantity  # margin / target speed
    horizon: Quantity  # sqrt(2 margin / a_slip)
    renewal_period: Quantity
    age_at_replacement: Quantity  # renewal period + inference
    drift_at_replacement: Quantity
    horizon_spare: Quantity  # horizon - age at replacement
    fits: bool
    min_supply_setpoints: int
    chunk_span: Quantity
    drift_at_chunk_end: Quantity
    error_at_chunk_end: Quantity  # initial error + drift at chunk end
    outlast: Quantity  # chunk span - horizon
    setpoints_within_horizon: float


class PhaseLagBudgetResult(SolverResult):
    """Result from PhaseLagBudgetModel: delay terms of a sampled loop and the phase they spend at crossover."""

    __module__ = "mlsysim.engine.results"

    hold_delay: Quantity  # zero-order hold, half a sample period
    total_delay: Quantity  # hold + compute + transport
    phase_hold: Quantity  # degrees at crossover
    phase_compute: Quantity
    phase_transport: Quantity
    phase_total: Quantity
    phase_per_delay: Quantity  # degrees per millisecond of delay at crossover
    allowed_delay: Optional[Quantity] = None  # largest total delay the allowance absorbs
    delay_margin: Optional[Quantity] = None  # allowed_delay - total_delay (negative when it fails)
    crossover_ceiling: Optional[Quantity] = None  # highest crossover the total delay supports
    over_allowance: Optional[float] = None  # phase_total / phase_allowance
    fits: Optional[bool] = None


class ResponseTimeResult(SolverResult):
    """Result from ResponseTimeModel: R = C + B + I of one periodic task against its deadline."""

    __module__ = "mlsysim.engine.results"

    wcet: Quantity
    blocking: Quantity
    interference: Quantity
    deadline: Quantity
    response: Quantity
    margin: Quantity  # deadline - response (negative when it misses)
    meets_deadline: bool
    blocking_limit: Quantity  # largest blocking that still meets the deadline
    wcet_claim: Optional[Quantity] = None  # declared WCET above the observed unloaded run
    execution_ceiling: Optional[Quantity] = (
        None  # deadline - B - I: the longest execution that still meets the deadline (added 2026-10-20)
    )


class CanFdBusResult(SolverResult):
    """Result from CanFdBusModel: serialization occupancy of a cyclic CAN FD axis exchange."""

    __module__ = "mlsysim.engine.results"

    frame_time: Quantity  # one frame, both phases
    nominal_time: Quantity  # arbitration-phase share of one frame
    data_time: Quantity  # data-phase share of one frame
    bits_nominal: int
    bits_data: int
    stuff_bits: int
    frames: int  # frames per cycle
    occupancy: Quantity  # frames x frame_time
    utilization: float  # occupancy / cycle
    unallocated: Quantity  # cycle - occupancy (negative when it overruns)
    fits: bool
    axes_capacity: int  # most axes whose frames fit one cycle


class EtherCatCycleResult(SolverResult):
    """Result from EtherCatCycleModel: one summation frame through a chain of drives."""

    __module__ = "mlsysim.engine.results"

    frame_time: Quantity  # serialization of the one frame
    forward_time: Quantity  # per-node forwarding, all nodes
    occupancy: Quantity  # frame_time + forward_time
    utilization: float
    unallocated: Quantity
    fits: bool


class LeaseBudgetResult(SolverResult):
    """Result from LeaseBudgetModel: a chunk lease priced in the stopping budget and bounded by the lease rule."""

    __module__ = "mlsysim.engine.results"

    lease: Quantity
    lease_path: Quantity  # lease + tick + bus: last admission to the stop command
    lease_path_travel: Quantity  # travel at speed along the lease path
    admission_to_brake_onset: Quantity  # lease path + brake onset
    admission_to_brake_onset_travel: Quantity
    body_terms_distance: Quantity  # braking + overhead + onset travel
    running_total: Quantity  # body terms + lease path travel
    spare: Quantity  # clear distance - running total
    delay_ceiling: Quantity  # longest pre-brake delay whose stop fits
    delay_spare: Quantity  # delay ceiling - admission to brake onset
    lease_floor: Quantity  # renewal period + renewal jitter
    lease_ceiling: Quantity  # delay ceiling - tick - bus - brake onset
    lease_ceiling_after_age: Quantity  # ... - observation age at dispatch
    lease_ceiling_at_admission: Quantity  # ... - evidence age at admission
    fits: bool  # floor < lease <= ceiling at admission
    stall_last_admission: Quantity  # second renewal completes, then the proposer stalls
    stall_lease_expiry: Quantity
    stall_brake_onset: Quantity
    renewal_lateness_limit: Optional[Quantity] = (
        None  # lease - renewal period: a renewal this late lapses the lease (added 2026-10-20)
    )


class SharedDramReadResult(SolverResult):
    """Result from SharedDramReadModel: a permission tick's snapshot read behind FIFO backlogs on a shared DRAM controller (added 2026-10-20)."""

    __module__ = "mlsysim.engine.results"

    queued_transactions: int  # transactions in one backlog
    tile_transactions: int  # of them, the weight tile's
    backlog_bytes: int  # bytes in one backlog
    backlog_wait: Quantity  # one backlog: bytes / bandwidth + row conflicts
    snapshot_read: Quantity  # the tick's own snapshot read
    loaded_read: Quantity  # one backlog plus the snapshot read
    sense_left: Quantity  # sense allowance less the loaded read
    bypass_read: Quantity  # priority bypass: one in-flight transaction, then the read
    read_ceiling_wcet: Quantity  # declared WCET less the compute after sensing
    read_ceiling_deadline: Quantity  # execution ceiling less the compute after sensing
    backlogs_sense: int  # most backlogs whose wait still fits the sense allowance
    backlogs_wcet: int  # most backlogs inside the declared WCET
    backlogs_deadline: int  # most backlogs inside the execution ceiling
    unloaded_end: Quantity  # sense allowance + compute after sensing
    end_sense: Quantity  # tick end at backlogs_sense
    end_wcet: Quantity  # tick end at backlogs_wcet
    end_deadline: Quantity  # tick end at backlogs_deadline


class ThermalThrottleResult(SolverResult):
    """Result from ThermalThrottleModel: a governed clock halving a tick, and the hysteresis that holds it (added 2026-10-20)."""

    __module__ = "mlsysim.engine.results"

    cycles: float  # tick cycles at the nominal clock (sized to the WCET)
    execution: Quantity  # at the nominal clock
    execution_throttled: Quantity  # at the throttled clock
    response: Quantity  # execution + blocking + interference
    response_throttled: Quantity
    slack: Quantity  # deadline - nominal response
    overrun: Quantity  # throttled response - deadline
    overrun_ratio: float  # overrun / deadline
    junction_quiet: Quantity  # ambient + quiet power x theta (degC)
    junction_loaded: Quantity  # ambient + loaded power x theta (degC)
    loaded_rise: Quantity  # loaded power x theta (K)
    time_to_trip: Quantity  # from the quiet steady state, under load
    recovery: Quantity  # trip to clear, load stopped at the trip
    recovery_from_loaded: Quantity  # loaded steady state to clear
    ticks_exposed: int  # whole ticks inside the recovery
    ticks_exposed_from_loaded: int
    heat_capacity: Quantity  # tau / theta, the lumped capacitance
    theta_never_trips: Quantity  # largest theta whose loaded steady state stays below the trip


class ObservationAgeResult(SolverResult):
    """Result from ObservationAgeModel: a camera observation's age from exposure midpoint to dispatch."""

    __module__ = "mlsysim.engine.results"

    stages: List[Tuple[str, Quantity, Quantity]]  # (stage, duration, age after the stage)
    age: Quantity  # exposure midpoint to dispatch
    age_travel: Quantity
    arrival_offset: Quantity  # exposure midpoint to DMA completion (an arrival stamp's error)
    arrival_travel: Quantity
    trigger_offset: Quantity  # exposure start to midpoint (a trigger stamp's error)
    trigger_travel: Quantity
    exposure_travel: Quantity  # travel during the exposure itself (blur scale)
    capture_before_compute: Quantity  # half the exposure plus the readout: top-row sensor age
    sensor_age_middle_row: Quantity  # half the exposure plus half the readout
    top_row_excess: Quantity  # top-row sensor age minus one frame period
    capture_share: float  # capture_before_compute / age
    backbone_share: float  # backbone stage / age
    frame_period: Quantity
    frame_travel: Quantity  # travel during one frame period (one dropped frame)
    batch_frames: int
    batch_waits: List[Quantity]  # wait of each batched frame but the newest, oldest first
    batch_wait: Quantity  # the oldest frame's wait, (n - 1) frame periods
    batch_wait_travel: Quantity
    batched_age: Quantity  # age + batch_wait
    batched_age_ratio: float


class RollingShutterSkewResult(SolverResult):
    """Result from RollingShutterSkewModel: row timing and the skew a moving camera records."""

    __module__ = "mlsysim.engine.results"

    row_interval: Quantity
    midpoint_lag: Quantity  # half the exposure
    skew: Quantity  # along-track travel between the first and last rows
    skew_over_allowance: float
    lateral_skew: Quantity  # sideways shift of a target at range while turning
    lateral_share: float  # of the side clearance
    exposure_travel: Quantity  # blur: travel during one row's exposure


class TimingAttitudeErrorResult(SolverResult):
    """Result from TimingAttitudeErrorModel: position errors from a timing error and an angle error."""

    __module__ = "mlsysim.engine.results"

    timing_travel: Quantity  # v dt
    misorientation: Quantity  # omega dt
    timing_lateral: Quantity  # d omega dt
    angle_lateral: Quantity  # d theta (small angle)
    angle_lateral_share: Optional[float] = None  # of the stated reference distance


class LidarSweepAgeResult(SolverResult):
    """Result from LidarSweepAgeModel: one sweep's motion and a sector's age, streamed or whole."""

    __module__ = "mlsysim.engine.results"

    sweep_period: Quantity
    sweep_travel: Quantity
    sweep_rotation: Quantity
    sector_angle: Quantity
    sector_time: Quantity
    age_streamed: Quantity  # sector time + data latency + sync bound
    age_full: Quantity  # sweep period + data latency + sync bound
    travel_streamed: Quantity
    travel_full: Quantity
    slack_streamed: Quantity  # reference age - streamed age


class StereoDepthErrorResult(SolverResult):
    """Result from StereoDepthErrorModel: one-sigma depth error against range and the clearances it spends."""

    __module__ = "mlsysim.engine.results"

    sigma_near: Quantity
    sigma_far: Quantity
    sigma_target: Quantity
    range_ratio: float  # far / near
    sigma_growth: float  # sigma_far / sigma_near = range_ratio ** 2
    inset_near: Quantity  # ray clearance + one sigma (the shortcut inset)
    inset_far: Quantity
    range_at_loc_allowance: Quantity
    range_at_side_clearance: Quantity
    range_at_side_clearance_half_baseline: Quantity
    baseline_bias: Quantity


class DelayPhaseMarginResult(SolverResult):
    """Result from DelayPhaseMarginModel: the delay a loop's phase margin can absorb."""

    __module__ = "mlsysim.engine.results"

    phase_margin: Quantity  # radians
    crossover_frequency: Quantity  # Hz
    delay_ceiling: Quantity
    delay: Quantity
    share_used: float


class ConformalInsetResult(SolverResult):
    """Result from ConformalInsetModel: the calibration rank behind a split-conformal bound."""

    __module__ = "mlsysim.engine.results"

    alpha: float
    calibration_count: int
    min_count: int
    rank: int
    rank_from_top: int


class ClockConversionResult(SolverResult):
    """Result from ClockConversionModel: drift between syncs and the dispatch limit a conversion bound sets."""

    __module__ = "mlsysim.engine.results"

    drift_per_interval: Quantity
    dispatch_age_limit: Quantity
    bound_travel: Quantity


class VisibilityRangeResult(SolverResult):
    """Result from VisibilityRangeModel: the range at which fog erases contrast."""

    __module__ = "mlsysim.engine.results"

    visibility_range: Quantity


class ThermalExpansionResult(SolverResult):
    """Result from ThermalExpansionModel: a heated member's length change."""

    __module__ = "mlsysim.engine.results"

    expansion: Quantity


class RangeFusionResult(SolverResult):
    """Result from RangeFusionModel: one predict and two updates of a scalar range, and a misdated update."""

    __module__ = "mlsysim.engine.results"

    process_sigma: Quantity  # speed disturbance x interval
    predicted_sigma: Quantity
    first_gain: float
    first_sigma: Quantity  # spread after the first update
    second_sensor_sigma: Quantity
    second_gain: float
    fused_sigma: Quantity  # spread after both updates
    innovation_variance: Quantity  # second update's predicted innovation variance
    misdating_bias: Quantity  # reading-to-pose offset of the misdated update
    misdated_shift: Quantity  # gain x bias: how far the estimate moves off
    misdated_shift_sigmas: float  # that shift in reported standard deviations
    nis_expected: float  # 1 + bias^2 / S
    noncentrality: float  # bias^2 / S
    comparison_gain: float  # second update with its spread scaled
    comparison_sigma: Quantity


class EstimatorConsistencyResult(SolverResult):
    """Result from EstimatorConsistencyModel: acceptance band for mean NEES or NIS, and single-update gates."""

    __module__ = "mlsysim.engine.results"

    band_low: float
    band_high: float
    gates: List[float]  # chi-square thresholds, one per gate confidence
    detection: List[float]  # probability each gate rejects the biased update
    nis_expected: float


class ReplayBufferResult(SolverResult):
    """Result from ReplayBufferModel: slots a rewind buffer needs to cover an observation age."""

    __module__ = "mlsysim.engine.results"

    sample_period: Quantity
    slots: int


class InertialCoastResult(SolverResult):
    """Result from InertialCoastModel: an IMU integrated open loop between fixes."""

    __module__ = "mlsysim.engine.results"

    heading_noise: Quantity
    heading_bias: Quantity
    lateral_bias: Quantity  # heading bias seen at the target range
    position_noise: Quantity
    position_bias: Quantity
    bias_over_reference: float  # position bias / reference bound
    bias_over_noise: float


class OdometryDriftResult(SolverResult):
    """Result from OdometryDriftModel: a benchmark return error as a drift rate against a bound."""

    __module__ = "mlsysim.engine.results"

    path_length: Quantity
    rate_uncalibrated: float  # error per distance driven
    rate_calibrated: float
    distance_uncalibrated: Quantity  # distance that spends the bound
    distance_calibrated: Quantity
    time_uncalibrated: Quantity  # that distance at the operating speed
    time_calibrated: Quantity
    segment_error_uncalibrated: Quantity
    segment_error_calibrated: Quantity
    segment_over_bound_uncalibrated: float
    calibration_gain: float
    speed_ratio: float  # operating speed / benchmark speed


class BeliefExpiryResult(SolverResult):
    """Result from BeliefExpiryModel: a belief's error growth, its invalidation horizon, and an admission check."""

    __module__ = "mlsysim.engine.results"

    initial_error: Quantity
    growth_rate: Quantity
    growth_acceleration: Quantity
    tolerance: Quantity
    horizon: Quantity
    constant_velocity_horizon: Quantity  # the horizon if the acceleration term is dropped
    error_at_constant_velocity_horizon: Quantity
    overshoot: float  # that error / tolerance - 1
    overrun: Quantity  # constant-velocity horizon - horizon
    use_age_upper: Optional[Quantity] = None  # stated use age plus the clock-conversion bound
    error_at_use: Optional[Quantity] = None
    margin_at_use: Optional[Quantity] = None
    time_left: Optional[Quantity] = None
    latest_use_age: Optional[Quantity] = None

    @property
    def quadratic_coefficient(self) -> Quantity:
        """Return the coefficient of squared time in the belief-error bound.

        Returns
        -------
        Quantity
            Half the stored growth acceleration, retaining its length/time-squared unit.

        Notes
        -----
        Equation: quadratic_coefficient = growth_acceleration / 2.

        Assumptions: The error envelope uses a constant growth-acceleration bound,
        with error = initial_error + ``growth_rate*t`` + ``growth_acceleration*t**2/2``.

        Excludes: New evidence, resets, and time-varying growth acceleration.
        This property does not evaluate the error at any particular time.

        References
        ----------
        Derivation: Integrating a constant acceleration twice contributes ``a*t**2/2``;
        the returned coefficient is the a/2 multiplying squared elapsed time.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import BeliefExpiryResult
        >>> r = BeliefExpiryResult(initial_error=Q_(1, "m"), growth_rate=Q_(2, "m/s"),
        ...     growth_acceleration=Q_(2, "m/s^2"), tolerance=Q_(4, "m"),
        ...     horizon=Q_(1, "s"), constant_velocity_horizon=Q_(1.5, "s"),
        ...     error_at_constant_velocity_horizon=Q_(6.25, "m"), overshoot=0.5625,
        ...     overrun=Q_(0.5, "s"))
        >>> r.quadratic_coefficient == Q_(1, "m/s^2")
        True
        """
        return self.growth_acceleration / 2

    def error_after(self, elapsed: Quantity) -> Quantity:
        """Evaluate the stored belief-error envelope after an elapsed interval.

        Parameters
        ----------
        elapsed : Quantity
            Nonnegative time since the evidence underlying the stored error envelope.

        Returns
        -------
        Quantity
            Initial error plus linear and quadratic growth, in the initial-error unit.

        See Also
        --------
        mlsysim.physics.robotics.estimation.calc_belief_error_bound :
            Evaluate the constant-growth error envelope with quantity validation.

        Notes
        -----
        Equation: error = initial_error + ``growth_rate*elapsed`` +
        ``growth_acceleration*elapsed**2/2``.

        Assumptions: Initial error, growth rate, and growth acceleration retain their
        stored nonnegative bounds throughout the interval; no fresh evidence arrives.

        Excludes: Measurement updates, probabilistic confidence calibration, and any
        comparison with tolerance. The method reports the bound even beyond expiry.

        References
        ----------
        Derivation: The first integral of a constant growth-acceleration bound gives
        a linear rate increase; its second integral adds a quadratic error term to
        the initial error and constant-rate contribution.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import BeliefExpiryResult
        >>> r = BeliefExpiryResult(initial_error=Q_(1, "m"), growth_rate=Q_(2, "m/s"),
        ...     growth_acceleration=Q_(2, "m/s^2"), tolerance=Q_(4, "m"),
        ...     horizon=Q_(1, "s"), constant_velocity_horizon=Q_(1.5, "s"),
        ...     error_at_constant_velocity_horizon=Q_(6.25, "m"), overshoot=0.5625,
        ...     overrun=Q_(0.5, "s"))
        >>> r.error_after(Q_(1, "s")) == Q_(4, "m")
        True
        """
        from ...physics.robotics.estimation import calc_belief_error_bound

        return calc_belief_error_bound(self.initial_error, self.growth_rate, elapsed, self.growth_acceleration)


class StaleTransformResult(SolverResult):
    """Result from StaleTransformModel: a cached edge composed with a current pose, one frame stale."""

    __module__ = "mlsysim.engine.results"

    interval: Quantity
    drift_term: Quantity
    lever_term: Quantity
    total: Quantity  # initial error + drift + lever arm
    over_tolerance: float  # total / tolerance - 1
    lever_over_drift: float
    same_epoch_error: Quantity  # the belief's own bound over one interval


class FreeSpaceExpiryResult(SolverResult):
    """Result from FreeSpaceExpiryModel: occluded free space eroding at an approach speed."""

    __module__ = "mlsysim.engine.results"

    certified_depth_ahead: Quantity  # d_free - delta_loc - delta_margin, floored at zero
    certified_depth_abreast: Quantity
    horizon_ahead: Quantity
    horizon_abreast: Quantity
    time_left_at_dispatch: Quantity
    frames_bridged: float
    depth_eroded_by_age: Quantity
    time_per_depth: Quantity  # 1 / v_h
    walker_intrusion: Quantity  # v_h (tau_delay + v / a_brake)
    timeout_refused: Optional[Quantity] = None  # horizon ahead minus a uniform staleness timeout


class EpochGapResult(SolverResult):
    """Result from EpochGapModel: travel across the gap between two acquisition epochs."""

    __module__ = "mlsysim.engine.results"

    gap: Quantity
    speed: Quantity
    travel: Quantity
    slack: Optional[Quantity] = None  # separation less the committed terms


class SensorBiasDriftResult(SolverResult):
    """Result from SensorBiasDriftModel: an unobservable bias drifting under a steady reading."""

    __module__ = "mlsysim.engine.results"

    shift: Quantity
    true_value: Quantity
    lost_share: float


class FrozenTransformResult(SolverResult):
    """Result from FrozenTransformModel: a republished stale transform passing a publish-time check."""

    __module__ = "mlsysim.engine.results"

    republish_rate: Quantity
    error: Quantity
    passes_freshness: bool
    exceeds_tolerance: bool


class ClearanceQueryRow(SolverResult):
    """One voxel pitch in ClearanceQueryResult."""

    __module__ = "mlsysim.engine.results"

    pitch: Quantity
    occupancy_reads: float
    octree_depth: int
    octree_reads: float
    esdf_reads: int
    esdf_bytes: Quantity
    occupancy_bytes: Quantity
    quantization: Quantity


class ClearanceQueryResult(SolverResult):
    """Result from ClearanceQueryModel: read budget per sphere query, and reads and snapshot bytes per representation."""

    __module__ = "mlsysim.engine.results"

    tick: Quantity
    query_time: Quantity
    tick_spare_share: float
    reads_budget: float
    reach: Quantity
    slab_volume: Quantity
    rows: Dict[int, ClearanceQueryRow]  # keyed by pitch in whole centimeters
    wcet_share: Optional[float] = None


class HostMapSizingResult(SolverResult):
    """Result from HostMapSizingModel: a dense grid against a hashed narrow band."""

    __module__ = "mlsysim.engine.results"

    workspace_volume: Quantity
    dense_voxels: float
    dense_bytes: Quantity
    band_volume: Quantity
    band_share: float
    blocks: float
    block_bytes: Quantity
    hashed_bytes: Quantity
    ratio: float


class RaymarchTrafficResult(SolverResult):
    """Result from RaymarchTrafficModel: naive per-ray map-update traffic against DRAM bandwidth."""

    __module__ = "mlsysim.engine.results"

    rays: int
    steps: float
    traffic: Quantity
    share: float


class EncoderRateResult(SolverResult):
    """Result from EncoderRateModel: a per-frame encoder's sustained rate against a peak."""

    __module__ = "mlsysim.engine.results"

    required: Quantity
    peak: Quantity
    shortfall: float


class IntentCadenceResult(SolverResult):
    """Result from IntentCadenceModel: a target's evidence horizons against a slow proposer's cadence."""

    __module__ = "mlsysim.engine.results"

    margin: Quantity  # tolerance - initial error
    untracked_horizon: Quantity  # margin / target speed
    tracked_horizon: Quantity  # sqrt(2 margin / a_slip)
    intent_period: Quantity  # 1 / intent rate
    error_at_period: Quantity  # initial error + speed x period
    excess_at_period: Quantity  # error at period - tolerance
    inference_overrun: Quantity  # inference - untracked horizon
    inference_over_horizon: float  # inference / untracked horizon
    renewal_age: Quantity  # period + inference: oldest evidence behind a model-only target
    renewal_speed_ceiling: Quantity  # margin / renewal age
    renewal_share: float  # ceiling / target speed
    stall: Quantity
    stall_error_untracked: Quantity  # initial error + speed x stall
    stall_error_tracked: Quantity  # initial error + a_slip stall^2 / 2
    stop_onset: Quantity  # tick + bus + brake onset after expiry
    untracked_expiry_to_onset: Quantity  # untracked horizon + stop onset
    tracked_expiry_to_onset: Quantity  # tracked horizon + stop onset


class DynamicReachResult(SolverResult):
    """Result from DynamicReachModel: the transit lower bound against a target's evidence windows."""

    __module__ = "mlsysim.engine.results"

    transit_floor: Quantity  # distance / speed ceiling
    reach_tracked: Quantity  # speed ceiling x tracked horizon
    reach_untracked: Quantity  # speed ceiling x untracked horizon
    shortfall: Quantity  # transit floor - tracked horizon
    fits: bool  # transit floor <= tracked horizon
    fresh_captures: int  # ceil(transit floor / tracked horizon)


class ForwardOccupancyResult(SolverResult):
    """Result from ForwardOccupancyModel: growth of a person's speed-bound set against an aisle."""

    __module__ = "mlsysim.engine.results"

    growth_untracked: Quantity  # v_h x untracked horizon
    growth_tracked: Quantity  # v_h x tracked horizon
    growth_forecast: Quantity  # v_h x forecast horizon
    forecast_over_aisle: float  # growth over forecast / aisle width
    aisle_span_horizon: Quantity  # aisle width / v_h


class IntentLeaseExpiryResult(SolverResult):
    """Result from IntentLeaseExpiryModel: the three terms of the intent lease's expiry bound."""

    __module__ = "mlsysim.engine.results"

    credible_deceleration: Quantity  # velocity limit / worst-case stop time
    stop_distance: Quantity  # v t_onset + v^2 / (2 a)
    stopping_deadline: Quantity  # (clearance - stop distance) / v
    expiry_untracked: Quantity  # min(untracked horizon, lease, deadline)
    expiry_tracked: Quantity  # min(tracked horizon, lease, deadline)
    binding_untracked: str  # "evidence", "administrative", or "stopping"
    binding_tracked: str


class ReferentSetResult(SolverResult):
    """Result from ReferentSetModel: normalized candidate scores and the conformal prediction set."""

    __module__ = "mlsysim.engine.results"

    normalized: List[float]  # raw scores over their sum
    entropy: float  # Shannon entropy of the normalized scores, nats
    max_entropy: float  # ln(number of candidates)
    rank: int  # calibration rank ceil((n + 1)(1 - alpha))
    threshold: float  # 1 - qhat: the normalized score a candidate needs
    set_size: int  # candidates at or above the threshold
    refuse: bool  # set size other than one


class ConfidenceRegionResult(SolverResult):
    """Result from ConfidenceRegionModel: a Gaussian estimate's confidence region against a tolerance."""

    __module__ = "mlsysim.engine.results"

    quantile: float  # chi-square quantile r^2
    radius: float  # r, in standard deviations
    semi_axes: List[Quantity]  # r sigma_i
    axes_within: int  # semi-axes at or under the tolerance
    worst_over_tolerance: float  # largest semi-axis / tolerance


class TwoLinkJointRateResult(SolverResult):
    """Result from TwoLinkJointRateModel: the joint rates one tool speed asks of a planar two-link arm."""

    __module__ = "mlsysim.engine.results"

    reach: Quantity  # l1 + l2
    sigma_max: Quantity  # Jacobian singular values, m per rad
    sigma_min: Quantity
    joint_rate_norm: Quantity  # exact inverse in the weakest direction, v / sigma_min
    joint_rate_one_joint: Quantity  # some joint reaches at least the norm / sqrt(2)
    exceeds_joint_limit: bool  # joint_rate_one_joint > the joint velocity limit
    damped_joint_rate: Quantity  # damped least squares in the weakest direction
    realized_fraction: float  # share of the requested tool motion damping delivers
    damped_ceiling: Quantity  # v / (2 lambda), at any configuration
    damped_exceeds_joint_limit: bool


class HandoffExposureResult(SolverResult):
    """Result from HandoffExposureModel: a per-handoff deadline budget across a mission."""

    __module__ = "mlsysim.engine.results"

    handoffs: int  # mission time x replanning rate
    expected_late: float  # handoffs x late probability
    all_on_time: float  # (1 - p)^K, independent misses
    per_handoff_budget: float  # mission budget / K (union bound)
    design_quantile: float  # 1 - per-handoff budget
    depth_ratio: float  # late probability / per-handoff budget
    orders_deeper: float  # log10 of depth_ratio
    handoffs_per_expected_late: float  # 1 / p: the count at which one late handoff is expected


class ChunkReplacementResult(SolverResult):
    """Result from ChunkReplacementModel: capture-to-admission latency against the renewal period."""

    __module__ = "mlsysim.engine.results"

    latency: Quantity  # observation age + dispatch-to-admission
    other_stages: Quantity  # dispatch-to-admission - inference - validation
    latency_over_renewal: float
    requests_in_flight: int  # ceil(latency / renewal period)
    latency_beyond_lease: Quantity
    tail_over_inference: float  # interference tail / p99 inference pass


class JerkLimitedStopResult(SolverResult):
    """Result from JerkLimitedStopModel: a delayed S-curve stop beside its constant and polynomial forms."""

    __module__ = "mlsysim.engine.results"

    speed: Quantity
    deceleration: Quantity
    jerk: Quantity  # jerk bound
    jerk_plateau_speed: Quantity  # a^2 / j: above it the stop reaches the deceleration bound
    reaches_deceleration_bound: bool
    braking_travel: Quantity  # v^2 / (2 a), constant deceleration
    braking_time: Quantity  # v / a
    jerk_ramp_travel: Quantity  # S-curve travel - braking travel (v a / (2 j) above the plateau speed)
    ramp_time: Quantity  # a / j, one jerk ramp
    ramp_delay: Quantity  # a / (2 j): the jerk ramps as extra delay at speed
    scurve_travel: Quantity  # the S-curve stop alone, braking + jerk-ramp travel
    stop_time: Quantity  # the S-curve stop's duration
    polynomial_distance: Quantity  # reference C2 polynomial with the same peak deceleration
    polynomial_time: Quantity
    delay: Quantity
    delay_travel: Quantity  # v x delay
    stop_distance: Quantity  # delay travel + S-curve travel (as in StoppingBudgetResult)
    overhead: Quantity  # localization bound + protective margin
    constant_required: Quantity  # delay travel + braking travel + overhead
    required_distance: Quantity  # delay travel + S-curve stop + overhead
    polynomial_required: Quantity  # delay travel + polynomial stop + overhead
    rest_time: Quantity  # delay + stop time
    delay_share_of_stop: float  # delay travel / S-curve travel
    clear_distance: Optional[Quantity] = None
    spare: Optional[Quantity] = None  # clear distance - required distance (negative: overrun)
    spare_time: Optional[Quantity] = None  # spare / speed
    fits: Optional[bool] = None
    speed_ceiling: Optional[Quantity] = None  # fastest speed whose delayed S-curve stop fits


class SpillLimitedStopResult(SolverResult):
    """Result from SpillLimitedStopModel: an open container caps the stop's deceleration."""

    __module__ = "mlsysim.engine.results"

    spill_deceleration: Quantity  # g h_f / R
    rise_at_free_deceleration: Quantity  # R a / g at the free-space stop's deceleration
    spills_at_free_deceleration: bool
    stop: JerkLimitedStopResult  # at the spill bound, jerk rescaled with it
    reused_jerk_stop_distance: Quantity  # the same stop if the free-space jerk were kept
    overrun: Quantity  # required distance - clear distance
    carry_speed: Quantity  # fastest speed whose spill-limited stop fits


class TrajectoryDeadlinesResult(SolverResult):
    """Result from TrajectoryDeadlinesModel: a trajectory record's deadlines and the cruise it charges."""

    __module__ = "mlsysim.engine.results"

    nominal_end: Quantity  # setpoints x setpoint period
    setpoint_rate: Quantity
    executed_setpoints: float  # renewal period / setpoint period
    request_slack: Quantity  # request delay - renewal period
    preferred_admission: Quantity  # request + latency target + scheduling slack
    reserve: Quantity  # scheduling slack + decision-to-onset response
    minimum_onset: Quantity  # request + latency target + reserve
    admission_cutoff: Quantity  # commitment - response
    commit_slack: Quantity  # commitment - preferred admission
    rest: Quantity  # commitment + stop time
    window_periods: float  # rest / setpoint period
    blend_cruise: Quantity  # travel from the blend deadline to commitment
    used: Quantity  # blend cruise + stop distance
    margin: Quantity  # clear distance - used
    late_cruise: Quantity  # travel during a late wait past commitment
    late_breach: Quantity  # late cruise - margin


class SeamBridgeResult(SolverResult):
    """Result from SeamBridgeModel: a one-tick seam step against a quintic bridge on one joint."""

    __module__ = "mlsysim.engine.results"

    reflected_inertia: Quantity
    step_torque: Quantity  # J_refl x gap / tick
    step_torque_rate: Quantity  # step torque / tick
    step_over_rate_limit: float
    step_within_rating: bool  # step torque below the rated torque
    loaded_step_torque: Optional[Quantity] = None  # with the payload's inertia added
    mean_torque: Quantity  # J_refl x gap / T
    peak_acceleration: Quantity  # 1.5 gap / T
    peak_torque: Quantity
    peak_reduction: float  # step torque / bridge peak torque
    peak_jerk: Quantity  # 6 gap / T^2
    torque_rate: Quantity  # J_refl x peak jerk
    torque_rate_share: float  # of the torque-rate limit
    within_rate_limit: bool
    coefficients: Tuple[float, float, float]  # a3, a4, a5 of the bridge's quintic, SI
    position_residual: Quantity  # gap T / 2
    shortest_bridge: Quantity  # sqrt(6 gap J_refl / torque-rate limit)
    velocity_ceiling: Quantity  # repeated peak torque x tick / J_refl


class ShoulderSwingResult(SolverResult):
    """Result from ShoulderSwingModel: the loaded shoulder's holding torque and its fastest swing."""

    __module__ = "mlsysim.engine.results"

    holding_torque: Quantity  # m g r
    holding_share: float  # of the joint torque limit
    effective_inertia: Quantity  # m r^2 + J_refl
    accel_up: Quantity  # (tau_max - tau_g) / J
    accel_down: Quantity  # (tau_max + tau_g) / J
    binding_acceleration: Quantity
    bang_bang_time: Quantity
    peak_speed: Quantity
    switch_torque_step: Quantity  # 2 J a at the switch point
    switch_step_time: Quantity  # the step at the torque-rate limit
    jerk_bound: Quantity  # torque-rate limit / J
    jerk_limited_time: Quantity
    jerk_limited_peak_speed: Quantity
    reaches_plateau: bool
    penalty: Quantity  # jerk-limited - bang-bang
    penalty_share: float
    ramp_time: Quantity  # a / j
    hold_time: Quantity  # constant-acceleration hold in each half
    within_velocity_limit: bool


class DynamicWindowResult(SolverResult):
    """Result from DynamicWindowModel: a base's reachable speeds and the classical admissibility test."""

    __module__ = "mlsysim.engine.results"

    half_width: Quantity  # max acceleration x window
    admissible_speed: Quantity  # sqrt(2 d a_b)
    admits_drive_limit: bool


class StaleAnchorResult(SolverResult):
    """Result from StaleAnchorModel: how far a moving reference runs past a pose sampled at the request."""

    __module__ = "mlsysim.engine.results"

    gap: Quantity  # joint speed x delay
    gap_degrees: Quantity
    tool_displacement: Quantity  # gap x lever arm


class TakeoverBudgetResult(SolverResult):
    """Result from TakeoverBudgetModel: a takeover request priced against the finished stopping budget."""

    __module__ = "mlsysim.engine.results"

    speed: Quantity  # speed at the request
    takeover: Quantity  # T_Sigma: human response plus console transport
    dispatch_to_admission: Quantity  # allocation charged in the complete budget (zero if unstated)
    machine: StoppingBudgetResult  # the machine alone, partial budget (pre-brake delay only)
    machine_with_dispatch: StoppingBudgetResult  # the machine alone, dispatch-to-admission charged
    with_takeover: StoppingBudgetResult  # the takeover waited out before braking, partial budget
    with_takeover_and_dispatch: StoppingBudgetResult  # the same with dispatch to admission: the complete budget
    deadline: Quantity  # T_latest: the longest wait the partial budget allows
    deadline_after_admission: Quantity  # T_latest less the dispatch-to-admission allocation
    takeover_travel: Quantity  # v T_Sigma
    admission_travel: Quantity  # v Delta_da
    time_left: Quantity  # deadline - takeover (negative: the takeover does not fit)
    time_left_after_admission: Quantity  # the same after the allocation
    spare_after_takeover: Quantity  # partial spare - takeover travel
    spare_after_admission: Quantity  # less the dispatch-to-admission travel
    takeover_over_spare: float  # takeover travel / partial spare
    takeover_over_spare_after_admission: float  # takeover travel / complete spare
    lease_extension: Optional[Quantity] = None  # manual lease - chunk lease, once the operator holds the channel
    lease_extension_travel: Optional[Quantity] = None  # v x lease extension
    spare_after_lease: Optional[Quantity] = None  # spare after admission, less the lease extension travel
    configured_timeout: Optional[Quantity] = None  # protocol ceiling on the handover timer
    timeout: Optional[Quantity] = None  # max(0, min(configured, deadline))
    timeout_ticks: Optional[int] = None  # whole permission ticks inside the timeout
    timeout_after_admission: Optional[Quantity] = None  # max(0, min(configured, deadline after admission))
    timeout_ticks_after_admission: Optional[int] = None


class TakeoverLadderResult(SolverResult):
    """Result from TakeoverCeilingLadderScenario: one takeover budget per sourced human response."""

    __module__ = "mlsysim.engine.results"

    transport: Quantity  # console term added to every human response
    human: Dict[str, Quantity]  # key -> human response
    rows: Dict[str, TakeoverBudgetResult]  # key -> budget with T_Sigma = human + transport


class BrakingBlendResult(SolverResult):
    """Result from BrakingBlendModel: a bumpless transfer from cruise into braking, and an arm blend."""

    __module__ = "mlsysim.engine.results"

    speed: Quantity
    deceleration: Quantity  # the blended step, cruise to braking
    jerk: Quantity  # the resident stop's jerk bound
    tick: Quantity
    window: Quantity  # 1.875 a / j
    window_ticks: int  # the window rounded up to whole permission ticks
    window_budget: Quantity  # window_ticks x tick
    step_jerk: Quantity  # a / tick: the same change taken in one tick
    ramp_time: Quantity  # a / j: the resident stop's linear jerk ramp
    exit_speed: Quantity  # v0 - a T / 2
    blend_travel: Quantity  # v0 T - a T^2 / 7
    after_travel: Quantity  # v_exit^2 / (2 a)
    blended_stop: Quantity  # blend travel + after travel
    step_stop: Quantity  # v0^2 / (2 a): an ideal step to braking
    scurve_travel: Quantity  # the resident jerk-limited stop
    penalty_over_step: Quantity  # blended stop - step stop
    excess_over_scurve: Quantity  # blended stop - resident stop
    reserve_use: Quantity  # largest growth of travel + resident stop over the blend's ticks
    torque_change: Optional[Quantity] = None
    torque_rate_limit: Optional[Quantity] = None
    arm_window: Optional[Quantity] = None  # 1.875 |Delta tau| / torque-rate ceiling
    arm_window_ticks: Optional[int] = None
    arm_window_budget: Optional[Quantity] = None


class CommandAuthenticationResult(SolverResult):
    """Result from CommandAuthenticationModel: authentication as pre-brake delay, and a verifier flood."""

    __module__ = "mlsysim.engine.results"

    verify_time: Quantity  # signature verification on the permission processor
    mac_time: Quantity  # MAC over one frame
    verify_travel: Quantity  # v x verify time
    spare_after_verify: Quantity
    mac_travel: Quantity
    freshness_window: Quantity
    freshness_travel: Quantity  # v x freshness window: a replay at the edge
    spare_after_freshness: Quantity
    deadline: Quantity  # the verification deadline (T_latest)
    deadline_after_admission: Quantity
    forged_to_miss: int  # fewest forged signed packets queued ahead that push a veto past the deadline
    veto_verified_at: Quantity  # (n + 1) x verify time
    forged_rate: int  # whole packets per second the flood must sustain (rounded up)
    forged_mac_to_miss: int  # the same count for MAC-only frames
    forged_to_miss_after_admission: int  # the count against the deadline after admission


class OnsetTrialResult(SolverResult):
    """Result from OnsetTrialModel: one stopping trace scored against a precommitted onset and travel oracle."""

    __module__ = "mlsysim.engine.results"

    onset_bound: Quantity  # oracle: latest powered-deceleration onset after the reference event
    onset: Quantity  # measured onset after the same event
    onset_margin: Quantity  # onset_bound - onset (negative fails)
    extra_travel: Quantity  # speed * (onset - onset_bound): travel the late onset adds
    travel: Quantity  # speed * onset + resident stop
    travel_bound: Quantity  # oracle: onset-bound travel + resident stop + tracking inset
    distance_margin: Quantity  # travel_bound - travel (negative fails)
    onset_margin_normalized: float  # onset_margin / onset_bound
    distance_margin_normalized: float  # distance_margin / travel_bound
    rest_time: Quantity  # onset + stop time
    rest_time_bound: Quantity  # onset_bound + stop time
    passes: bool  # both margins non-negative


class RailHoldUpResult(SolverResult):
    """Result from RailHoldUpModel: a backed rail's hold-up against flag-to-brakes-set at the longest stop."""

    __module__ = "mlsysim.engine.results"

    flag_path: Quantity  # flag to powered-deceleration onset
    speed: Quantity  # entry speed of the longest admitted stop
    velocity_term: Quantity  # v / a
    jerk_term: Quantity  # a / j
    stop_time: Quantity  # S-curve stop time, v / a + a / j
    engage: Quantity  # holding-brake engagement after rest
    flag_to_set: Quantity  # flag path + stop time + engagement
    holdup: Quantity
    margin: Quantity  # holdup - flag_to_set


class ExponentialRareEventResult(SolverResult):
    """Result from ExponentialRareEventModel: estimating one rare exponential tail by plain and weighted sampling."""

    __module__ = "mlsysim.engine.results"

    threshold: Quantity
    failure_probability: float  # exp(-threshold / mean)
    monte_carlo_runs: float  # (1 - p) / (p r^2)
    zero_failure_runs: int  # clean runs that bound p at the stated confidence
    levels: list  # cross-entropy levels (Quantity each)
    n_levels: int
    proposal_mean: Quantity  # final cross-entropy proposal mean
    proposal_failure_share: float  # share of proposal draws past the threshold
    relative_variance: float  # per-draw relative variance of the weighted estimate
    importance_runs: float  # relative_variance / r^2
    ess_per_draw: float
    effective_sample_size: float  # importance_runs * ess_per_draw
    optimal_proposal_mean: Quantity
    proposal_penalty: float  # relative variance over the optimal proposal's, less one
    speedup: float  # monte_carlo_runs / importance_runs


class DiagnosticCoverageResult(SolverResult):
    """Result from DiagnosticCoverageModel: the coverage a residual-rate target needs and the zero-miss injections that show it."""

    __module__ = "mlsysim.engine.results"

    raw_rate: float  # FIT, sum of the strata
    coverage: float  # 1 - target / raw_rate
    injections_exact: float  # ln(1 - C) / ln(coverage)
    injections: int  # the ceiling of injections_exact
    rule_of_three: float  # 3 / (1 - coverage)
    shares: dict  # stratum -> share of the raw rate
    allocation: dict  # stratum -> injections in proportion to its rate


class DetectionShortfallRow(SolverResult):
    """One group's detection gap, priced in distance, impact speed, and impact energy."""

    __module__ = "mlsysim.engine.results"

    per_frame_detection: float
    frames: int  # consecutive frames before detection counts
    extra_frames: int  # frames beyond the reference group's
    extra_time: Quantity  # extra frames x frame period
    distance_lost: Quantity  # v x frame period x extra frames
    impact_speed: Quantity  # sqrt(2 a distance_lost), while distance_lost < braking travel
    impact_energy: Quantity  # m a distance_lost = 1/2 m v_impact^2


class DetectionShortfallResult(SolverResult):
    """Result from DetectionShortfallModel: a stop sized for the reference group, met by groups seen less often."""

    __module__ = "mlsysim.engine.results"

    frame_period: Quantity
    reference_frames: int
    braking_travel: Quantity  # v^2 / (2 a)
    reference_brake_onset: Optional[Quantity] = (
        None  # v (N_ref T_frame + delay): travel before the reference stop brakes
    )
    reference_stop: Optional[Quantity] = None  # where the reference stop ends: the person's position
    rows: List[DetectionShortfallRow]
    within_braking: bool  # every distance lost is shorter than the braking travel


class SupervisorQueueResult(SolverResult):
    """Result from SupervisorQueueModel: one remote assistant serving ``machines`` as an M/M/1 queue."""

    __module__ = "mlsysim.engine.results"

    machines: int
    arrival_rate: Quantity  # fleet help requests per hour
    utilization: float  # rho = lambda s
    mean_wait: Quantity  # W_q = rho s / (1 - rho)
    p99_wait: Quantity  # 99th-percentile queueing wait
    busy_per_hour: Quantity  # assistant minutes per operating hour


class GripperTransferResult(SolverResult):
    """Geometry and force comparison for replaying a normalized gripper command."""

    __module__ = "mlsysim.engine.results"

    donor_opening: Quantity
    target_opening: Quantity
    stroke_excess: Quantity
    force_ratio: float
    donor_clears: bool
    target_clears: bool
