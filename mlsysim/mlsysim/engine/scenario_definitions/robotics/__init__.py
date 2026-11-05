"""Physical-machine scenarios, grouped by sensing, control, motion, and learning."""

from .stopping import (
    StoppingBudgetScenario as StoppingBudgetScenario,
    TractionLossScenario as TractionLossScenario,
    TravelDuringLatencyScenario as TravelDuringLatencyScenario,
    SpeedStepScenario as SpeedStepScenario,
)
from .plant import (
    FreshnessProfileScenario as FreshnessProfileScenario,
    BrakingCeilingsScenario as BrakingCeilingsScenario,
    HubDriveElectricalScenario as HubDriveElectricalScenario,
    GearedRotorScenario as GearedRotorScenario,
    GearRatioCostScenario as GearRatioCostScenario,
    ThermalBurstScenario as ThermalBurstScenario,
    StallDeratingScenario as StallDeratingScenario,
    ContinuousDeratingScenario as ContinuousDeratingScenario,
    DriveBusScenario as DriveBusScenario,
    ZeroFailureScenario as ZeroFailureScenario,
    ThrottledStageScenario as ThrottledStageScenario,
)
from .proposers import (
    VlaEvaluationScenario as VlaEvaluationScenario,
    RateTargetScenario as RateTargetScenario,
    VlaTokenBudgetScenario as VlaTokenBudgetScenario,
    KvHistoryScenario as KvHistoryScenario,
    DecoderFamiliesScenario as DecoderFamiliesScenario,
    ProposerTrafficScenario as ProposerTrafficScenario,
    EvidenceHorizonScenario as EvidenceHorizonScenario,
    WeightOnlyQuantizationScenario as WeightOnlyQuantizationScenario,
)
from .demonstrations import (
    CorrelatedLogScenario as CorrelatedLogScenario,
    ImitationBoundScenario as ImitationBoundScenario,
    PairingLagScenario as PairingLagScenario,
    TeleopLoopScenario as TeleopLoopScenario,
    TeleopIngestionScenario as TeleopIngestionScenario,
    DemonstrationCampaignScenario as DemonstrationCampaignScenario,
    StateGridScenario as StateGridScenario,
    ContactGridScenario as ContactGridScenario,
    ScenarioTableCoverageScenario as ScenarioTableCoverageScenario,
    GripperTransferScenario as GripperTransferScenario,
)
from .realtime import (
    PhaseLagBudgetScenario as PhaseLagBudgetScenario,
    ResponseTimeScenario as ResponseTimeScenario,
    CanFdBusScenario as CanFdBusScenario,
    EtherCatCycleScenario as EtherCatCycleScenario,
    LeaseBudgetScenario as LeaseBudgetScenario,
)
from .learning import (
    DiscountHorizonScenario as DiscountHorizonScenario,
    IterativeDecoderBudgetScenario as IterativeDecoderBudgetScenario,
    ParallelSimRunScenario as ParallelSimRunScenario,
    TrialCampaignScenario as TrialCampaignScenario,
    ContactDeadlineScenario as ContactDeadlineScenario,
    SoftContactExploitScenario as SoftContactExploitScenario,
    ForceRiseGapScenario as ForceRiseGapScenario,
    OnRobotFineTuneScenario as OnRobotFineTuneScenario,
)
from .contact import (
    HeadingDriftScenario as HeadingDriftScenario,
    StiffContactScenario as StiffContactScenario,
    ForceTailScenario as ForceTailScenario,
)
from .perception import (
    ObservationAgeScenario as ObservationAgeScenario,
    RollingShutterScenario as RollingShutterScenario,
    TimingAttitudeErrorScenario as TimingAttitudeErrorScenario,
    LidarSweepAgeScenario as LidarSweepAgeScenario,
    StereoDepthErrorScenario as StereoDepthErrorScenario,
    DelayPhaseMarginScenario as DelayPhaseMarginScenario,
    ConformalInsetScenario as ConformalInsetScenario,
    ClockConversionScenario as ClockConversionScenario,
    VisibilityRangeScenario as VisibilityRangeScenario,
    ThermalExpansionScenario as ThermalExpansionScenario,
)
from .estimation import (
    RangeFusionScenario as RangeFusionScenario,
    EstimatorConsistencyScenario as EstimatorConsistencyScenario,
    ReplayBufferScenario as ReplayBufferScenario,
    InertialCoastScenario as InertialCoastScenario,
    OdometryDriftScenario as OdometryDriftScenario,
    BeliefExpiryScenario as BeliefExpiryScenario,
    StaleTransformScenario as StaleTransformScenario,
    FreeSpaceExpiryScenario as FreeSpaceExpiryScenario,
    EpochGapScenario as EpochGapScenario,
    SkewedRecordsScenario as SkewedRecordsScenario,
    SensorBiasDriftScenario as SensorBiasDriftScenario,
    FrozenTransformScenario as FrozenTransformScenario,
    ClearanceQueryScenario as ClearanceQueryScenario,
    HostMapSizingScenario as HostMapSizingScenario,
    RaymarchTrafficScenario as RaymarchTrafficScenario,
    EncoderRateScenario as EncoderRateScenario,
)
from .grounding import (
    IntentCadenceScenario as IntentCadenceScenario,
    DynamicReachScenario as DynamicReachScenario,
    ForwardOccupancyScenario as ForwardOccupancyScenario,
    IntentLeaseExpiryScenario as IntentLeaseExpiryScenario,
    ReferentSetScenario as ReferentSetScenario,
    ConfidenceRegionScenario as ConfidenceRegionScenario,
    GoalSetMembershipScenario as GoalSetMembershipScenario,
)
from .control import (
    GravityBiasedJointStopScenario as GravityBiasedJointStopScenario,
    TorqueHalfSpaceProjectionScenario as TorqueHalfSpaceProjectionScenario,
    CycleBudgetScenario as CycleBudgetScenario,
    HoldIntervalDriftScenario as HoldIntervalDriftScenario,
)
from .placement import (
    SharedDramReadScenario as SharedDramReadScenario,
    StallShareScenario as StallShareScenario,
    RailDroopScenario as RailDroopScenario,
    ThermalThrottleScenario as ThermalThrottleScenario,
    TickTraceScenario as TickTraceScenario,
)
from .planning import (
    TwoLinkJointRateScenario as TwoLinkJointRateScenario,
    HandoffExposureScenario as HandoffExposureScenario,
    ChunkReplacementScenario as ChunkReplacementScenario,
    JerkLimitedStopScenario as JerkLimitedStopScenario,
    SpillLimitedStopScenario as SpillLimitedStopScenario,
    TrajectoryDeadlinesScenario as TrajectoryDeadlinesScenario,
    SeamBridgeScenario as SeamBridgeScenario,
    ShoulderSwingScenario as ShoulderSwingScenario,
    DynamicWindowScenario as DynamicWindowScenario,
    StaleAnchorScenario as StaleAnchorScenario,
)
from .supervision import (
    TakeoverBudgetScenario as TakeoverBudgetScenario,
    TakeoverCeilingLadderScenario as TakeoverCeilingLadderScenario,
    BrakingBlendScenario as BrakingBlendScenario,
    CommandAuthenticationScenario as CommandAuthenticationScenario,
    DecisionChainScenario as DecisionChainScenario,
    OversightTimeScenario as OversightTimeScenario,
    SupervisorQueueScenario as SupervisorQueueScenario,
    DetectionShortfallScenario as DetectionShortfallScenario,
    SubgroupSampleScenario as SubgroupSampleScenario,
    ExposureConcentrationScenario as ExposureConcentrationScenario,
    EventRetentionScenario as EventRetentionScenario,
)
from .diagnostics import (
    OnsetTrialScenario as OnsetTrialScenario,
    RailHoldUpScenario as RailHoldUpScenario,
    ExponentialRareEventScenario as ExponentialRareEventScenario,
    DiagnosticCoverageScenario as DiagnosticCoverageScenario,
)
from .site_limits import (
    PackagingCellScenario as PackagingCellScenario,
    ShieldCoverageBillScenario as ShieldCoverageBillScenario,
    LatchSeriesSpringScenario as LatchSeriesSpringScenario,
    RestrictedTransitScenario as RestrictedTransitScenario,
)
