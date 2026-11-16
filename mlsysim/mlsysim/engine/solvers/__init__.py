"""Domain-oriented MLSysIM solver implementations."""

from .base import (
    BaseOptimizer,
    BaseResolver,
    BaseSolver,
    ForwardModel,
)
from .acceptance import (
    BinomialIntervalsModel,
    CappedSelectionCostModel,
    PairedComparisonModel,
    ProportionPlanningModel,
    TaskMixConsistencyModel,
)
from .agents import (
    AcceptedTaskCostModel,
    AppendOnlyContextModel,
    HorizonCompoundingModel,
)
from .collective import (
    AllReduceAlgorithmModel,
    AllToAllDispatchModel,
    CompressionRunModel,
    ErrorFeedbackModel,
    HierarchicalAllReduceModel,
    OverlapBudgetModel,
    RingTraceModel,
)
from .compression import (
    CompressionModel,
)
from .context import (
    AttentionDilutionModel,
    CompactionCadenceModel,
    ContextBudgetModel,
    ContextInvalidationModel,
    PrefixLayoutBillModel,
    ProtectedFieldCheckModel,
)
from .cost_accounting import (
    AdaptationPaybackModel,
    ArchitectureChoiceModel,
    CohortAttemptCostModel,
    CompactionPaybackModel,
    ConservedBudgetModel,
    DedicatedBreakEvenModel,
    KingmanQueueModel,
    ModelCascadeModel,
    PrefixCacheBillModel,
    RetryPolicyModel,
    SerialEpisodeModel,
    SpeculativeEpisodeModel,
    TrajectoryAdmissionModel,
    WorkloadLeverModel,
)
from .curation import (
    AdmissionOrderModel,
    CorpusDenominatorsModel,
    FailureTriageModel,
    RunSplitLeakageModel,
    StagingBurstModel,
)
from .data import (
    DataModel,
    TransformationModel,
)
from .deliberation import (
    AllocationAxesModel,
    ArgmaxNoiseModel,
    BestOfNPoolModel,
    DeadlinePolicyModel,
    ProbeVsAnswerModel,
    PrunedSearchTreeModel,
    ScopedRevisionModel,
    SearchAccountsModel,
    VerifiedPoolModel,
)
from .distributed import (
    DistributedModel,
    MoERoutingModel,
    ParallelismOptimizer,
    TopologyModel,
)
from .durability import (
    CheckpointTraceModel,
    GroupCommitModel,
    RetentionCostModel,
    SnapshotVolumeModel,
)
from .economics import (
    EconomicsModel,
    PlacementOptimizer,
    ResponsibleEngineeringModel,
    SustainabilityModel,
)
from .embodied import (
    OnsetTrialModel,
    RailHoldUpModel,
    ShieldCoverageBillModel,
    StoppingBudgetModel,
    TakeoverBudgetModel,
    TractionLossModel,
)
from .estimation import (
    BeliefExpiryModel,
    EpochGapModel,
    EstimatorConsistencyModel,
    FreeSpaceExpiryModel,
    FrozenTransformModel,
    InertialCoastModel,
    OdometryDriftModel,
    RangeFusionModel,
    ReplayBufferModel,
    SensorBiasDriftModel,
    StaleTransformModel,
)
from .facility import (
    CoolantFlowModel,
    OutageCostModel,
    PowerTransientModel,
)
from .fine_tuning import (
    LossNormalizationModel,
    PerInvocationSerializationModel,
    PreferencePairModel,
    SequencePackingModel,
    TargetMaskModel,
)
from .goodput import (
    GoodputModel,
    RuntimeGoodputModel,
)
from .grounding import (
    ConfidenceRegionModel,
    DynamicReachModel,
    ForwardOccupancyModel,
    IntentCadenceModel,
    IntentLeaseExpiryModel,
    ReferentSetModel,
)
from .harness import (
    ApprovalEscrowModel,
    BoundedLiabilityAdmissionModel,
    DeficitRoundRobinModel,
    TrajectoryBudgetMeterModel,
)
from .interconnect import (
    AlphaBetaModel,
    BisectionModel,
    CommunicationIntensityModel,
    GradientSyncModel,
    IncastModel,
    TensorParallelCommModel,
    TransferLadderModel,
)
from .kv_cache import (
    ChunkedPrefillModel,
    KvAttentionDesignModel,
    KvBlockSharingModel,
    KvResidencyArchetypeModel,
    PagedAllocationModel,
    PausedKvStateModel,
    PrefixRebuildModel,
)
from .mapping import (
    ClearanceQueryModel,
    EncoderRateModel,
    HostMapSizingModel,
    RaymarchTrafficModel,
)
from .model_call import (
    CheckerPosteriorModel,
    ContextAdmissionModel,
    ModelCallModel,
    SamplingModel,
    TrajectoryResendModel,
)
from .neural import (
    DenseNetworkModel,
    EpochOverheadModel,
)
from .operations import (
    SequentialCanaryModel,
    SpanCriticalPathModel,
    TraceRetentionModel,
)
from .orchestration import (
    OrchestrationModel,
    PreemptionTaxModel,
)
from .parallelism import (
    AlignmentMemoryModel,
    DataParallelScaleOutModel,
    ParallelLayoutModel,
    TensorParallelLayerModel,
    ZeroShardingModel,
)
from .perception import (
    ClockConversionModel,
    ConformalInsetModel,
    DelayPhaseMarginModel,
    LidarSweepAgeModel,
    ObservationAgeModel,
    RollingShutterSkewModel,
    StereoDepthErrorModel,
    ThermalExpansionModel,
    TimingAttitudeErrorModel,
    VisibilityRangeModel,
)
from .performance import (
    AmdahlModel,
    DecodeBoundModel,
    EfficiencyModel,
    IronLawModel,
    NetworkRooflineModel,
    SensitivitySolver,
    SingleNodeModel,
    SynthesisSolver,
)
from .plant import (
    BrakingCeilingsModel,
    ContinuousDeratingModel,
    DiagnosticCoverageModel,
    DriveBusModel,
    ExponentialRareEventModel,
    FreshnessProfileModel,
    GearRatioCostModel,
    GearedRotorModel,
    HubDriveElectricalModel,
    SpeedStepModel,
    StallDeratingModel,
    ThermalBurstModel,
    ThrottledStageModel,
    ZeroFailureModel,
)
from .proposer import (
    DecoderFamiliesModel,
    EvidenceHorizonModel,
    ProposerTrafficModel,
    VlaEvaluationModel,
)
from .realtime import (
    CanFdBusModel,
    CommandAuthenticationModel,
    EtherCatCycleModel,
    LeaseBudgetModel,
    PhaseLagBudgetModel,
    ResponseTimeModel,
    SharedDramReadModel,
    ThermalThrottleModel,
)
from .recovery import (
    BulkheadAdmissionModel,
    IrreversibleLossModel,
    RepairBreakEvenModel,
    RetryAmplificationModel,
    RetryJitterModel,
    UndetectedLoopModel,
)
from .reinforcement import (
    CenteredRewardModel,
    ContinuationForkCostModel,
    ForkValueModel,
    GroupBarrierModel,
    GroupCoefficientsModel,
    LengthNormalizationModel,
    RatioClipModel,
    RepeatedVerificationModel,
    RewardCompositionModel,
    RolloutSlotCycleModel,
    RolloutStepBudgetModel,
    VerifierPoolModel,
)
from .reliability import (
    CheckpointIntervalModel,
    FailureIntervalModel,
    RecoveryBudgetModel,
    ReliabilityModel,
    SeriesFailureModel,
)
from .retrieval import (
    AnnFootprintModel,
    Bm25PairModel,
    ExactScanFloorModel,
    GraphNeighborhoodModel,
    IndexMaintenanceModel,
    RetrievalPayloadModel,
    StorageAmplificationModel,
)
from .sandboxes import (
    EnvelopeAmortizationModel,
    SandboxPoolReserveModel,
    SyscallTaxModel,
    WorkspaceProvisioningModel,
)
from .security import (
    InjectionExposureModel,
    PolicyComparisonModel,
)
from .serving import (
    BatchingOptimizer,
    ContinuousBatchingModel,
    InferenceScalingModel,
    ServingCapacityModel,
    ServingModel,
    TailLatencyModel,
    WeightStreamingModel,
)
from .serving_scale import (
    KvAdmissionModel,
    PrecisionDividendModel,
)
from .storage import (
    CheckpointStormModel,
    DataStallModel,
    HostIoCpuModel,
    InputDemandModel,
    ObjectStorageBillModel,
    PrefetchDepthModel,
    StorageFootprintModel,
    StreamVsStageModel,
    SyntheticFootprintModel,
    TieredCacheModel,
    TieredCheckpointModel,
    UtilizationLossModel,
)
from .supervision import (
    DetectionShortfallModel,
    SupervisorQueueModel,
)
from .sustainability import (
    CarbonPricedPlacementModel,
    CascadeEnergyModel,
    DataMovementEnergyModel,
    DemandReboundModel,
    DevelopmentSweepModel,
    DutyCycledPowerModel,
    EmbodiedPaybackModel,
    FacilityPowerModel,
    GridPlacementModel,
    PueRetrofitModel,
    RepeatedWorkEnergyModel,
    RunFootprintModel,
    ServiceLifeExtensionModel,
    ServingCrossoverModel,
    TemporalShiftModel,
    WaterUseModel,
)
from .tool_calls import (
    BlindRetryModel,
    ModelPollingModel,
    ObservationCarryModel,
    ToolCatalogSelectionModel,
    ToolkitGranularityModel,
)
from .training import (
    CheckpointModel,
    ScalingModel,
    TrainingMemoryModel,
    TrainingRunModel,
    TrainingStateModel,
)
from .trajectory import (
    BrakingBlendModel,
    ChunkReplacementModel,
    DynamicWindowModel,
    HandoffExposureModel,
    JerkLimitedStopModel,
    SeamBridgeModel,
    ShoulderSwingModel,
    SpillLimitedStopModel,
    StaleAnchorModel,
    TrajectoryDeadlinesModel,
    TwoLinkJointRateModel,
)

__all__ = [
    # base
    "BaseOptimizer",
    "BaseResolver",
    "BaseSolver",
    "ForwardModel",
    # acceptance
    "BinomialIntervalsModel",
    "CappedSelectionCostModel",
    "PairedComparisonModel",
    "ProportionPlanningModel",
    "TaskMixConsistencyModel",
    # agents
    "AcceptedTaskCostModel",
    "AppendOnlyContextModel",
    "HorizonCompoundingModel",
    # collective
    "AllReduceAlgorithmModel",
    "AllToAllDispatchModel",
    "CompressionRunModel",
    "ErrorFeedbackModel",
    "HierarchicalAllReduceModel",
    "OverlapBudgetModel",
    "RingTraceModel",
    # compression
    "CompressionModel",
    # context
    "AttentionDilutionModel",
    "CompactionCadenceModel",
    "ContextBudgetModel",
    "ContextInvalidationModel",
    "PrefixLayoutBillModel",
    "ProtectedFieldCheckModel",
    # cost_accounting
    "AdaptationPaybackModel",
    "ArchitectureChoiceModel",
    "CohortAttemptCostModel",
    "CompactionPaybackModel",
    "ConservedBudgetModel",
    "DedicatedBreakEvenModel",
    "KingmanQueueModel",
    "ModelCascadeModel",
    "PrefixCacheBillModel",
    "RetryPolicyModel",
    "SerialEpisodeModel",
    "SpeculativeEpisodeModel",
    "TrajectoryAdmissionModel",
    "WorkloadLeverModel",
    # curation
    "AdmissionOrderModel",
    "CorpusDenominatorsModel",
    "FailureTriageModel",
    "RunSplitLeakageModel",
    "StagingBurstModel",
    # data
    "DataModel",
    "TransformationModel",
    # deliberation
    "AllocationAxesModel",
    "ArgmaxNoiseModel",
    "BestOfNPoolModel",
    "DeadlinePolicyModel",
    "ProbeVsAnswerModel",
    "PrunedSearchTreeModel",
    "ScopedRevisionModel",
    "SearchAccountsModel",
    "VerifiedPoolModel",
    # distributed
    "DistributedModel",
    "MoERoutingModel",
    "ParallelismOptimizer",
    "TopologyModel",
    # durability
    "CheckpointTraceModel",
    "GroupCommitModel",
    "RetentionCostModel",
    "SnapshotVolumeModel",
    # economics
    "EconomicsModel",
    "PlacementOptimizer",
    "ResponsibleEngineeringModel",
    "SustainabilityModel",
    # embodied
    "OnsetTrialModel",
    "RailHoldUpModel",
    "ShieldCoverageBillModel",
    "StoppingBudgetModel",
    "TakeoverBudgetModel",
    "TractionLossModel",
    # estimation
    "BeliefExpiryModel",
    "EpochGapModel",
    "EstimatorConsistencyModel",
    "FreeSpaceExpiryModel",
    "FrozenTransformModel",
    "InertialCoastModel",
    "OdometryDriftModel",
    "RangeFusionModel",
    "ReplayBufferModel",
    "SensorBiasDriftModel",
    "StaleTransformModel",
    # facility
    "CoolantFlowModel",
    "OutageCostModel",
    "PowerTransientModel",
    # fine_tuning
    "LossNormalizationModel",
    "PerInvocationSerializationModel",
    "PreferencePairModel",
    "SequencePackingModel",
    "TargetMaskModel",
    # goodput
    "GoodputModel",
    "RuntimeGoodputModel",
    # grounding
    "ConfidenceRegionModel",
    "DynamicReachModel",
    "ForwardOccupancyModel",
    "IntentCadenceModel",
    "IntentLeaseExpiryModel",
    "ReferentSetModel",
    # harness
    "ApprovalEscrowModel",
    "BoundedLiabilityAdmissionModel",
    "DeficitRoundRobinModel",
    "TrajectoryBudgetMeterModel",
    # interconnect
    "AlphaBetaModel",
    "BisectionModel",
    "CommunicationIntensityModel",
    "GradientSyncModel",
    "IncastModel",
    "TensorParallelCommModel",
    "TransferLadderModel",
    # kv_cache
    "ChunkedPrefillModel",
    "KvAttentionDesignModel",
    "KvBlockSharingModel",
    "KvResidencyArchetypeModel",
    "PagedAllocationModel",
    "PausedKvStateModel",
    "PrefixRebuildModel",
    # mapping
    "ClearanceQueryModel",
    "EncoderRateModel",
    "HostMapSizingModel",
    "RaymarchTrafficModel",
    # model_call
    "CheckerPosteriorModel",
    "ContextAdmissionModel",
    "ModelCallModel",
    "SamplingModel",
    "TrajectoryResendModel",
    # neural
    "DenseNetworkModel",
    "EpochOverheadModel",
    # operations
    "SequentialCanaryModel",
    "SpanCriticalPathModel",
    "TraceRetentionModel",
    # orchestration
    "OrchestrationModel",
    "PreemptionTaxModel",
    # parallelism
    "AlignmentMemoryModel",
    "DataParallelScaleOutModel",
    "ParallelLayoutModel",
    "TensorParallelLayerModel",
    "ZeroShardingModel",
    # perception
    "ClockConversionModel",
    "ConformalInsetModel",
    "DelayPhaseMarginModel",
    "LidarSweepAgeModel",
    "ObservationAgeModel",
    "RollingShutterSkewModel",
    "StereoDepthErrorModel",
    "ThermalExpansionModel",
    "TimingAttitudeErrorModel",
    "VisibilityRangeModel",
    # performance
    "AmdahlModel",
    "DecodeBoundModel",
    "EfficiencyModel",
    "IronLawModel",
    "NetworkRooflineModel",
    "SensitivitySolver",
    "SingleNodeModel",
    "SynthesisSolver",
    # plant
    "BrakingCeilingsModel",
    "ContinuousDeratingModel",
    "DiagnosticCoverageModel",
    "DriveBusModel",
    "ExponentialRareEventModel",
    "FreshnessProfileModel",
    "GearRatioCostModel",
    "GearedRotorModel",
    "HubDriveElectricalModel",
    "SpeedStepModel",
    "StallDeratingModel",
    "ThermalBurstModel",
    "ThrottledStageModel",
    "ZeroFailureModel",
    # proposer
    "DecoderFamiliesModel",
    "EvidenceHorizonModel",
    "ProposerTrafficModel",
    "VlaEvaluationModel",
    # realtime
    "CanFdBusModel",
    "CommandAuthenticationModel",
    "EtherCatCycleModel",
    "LeaseBudgetModel",
    "PhaseLagBudgetModel",
    "ResponseTimeModel",
    "SharedDramReadModel",
    "ThermalThrottleModel",
    # recovery
    "BulkheadAdmissionModel",
    "IrreversibleLossModel",
    "RepairBreakEvenModel",
    "RetryAmplificationModel",
    "RetryJitterModel",
    "UndetectedLoopModel",
    # reinforcement
    "CenteredRewardModel",
    "ContinuationForkCostModel",
    "ForkValueModel",
    "GroupBarrierModel",
    "GroupCoefficientsModel",
    "LengthNormalizationModel",
    "RatioClipModel",
    "RepeatedVerificationModel",
    "RewardCompositionModel",
    "RolloutSlotCycleModel",
    "RolloutStepBudgetModel",
    "VerifierPoolModel",
    # reliability
    "CheckpointIntervalModel",
    "FailureIntervalModel",
    "RecoveryBudgetModel",
    "ReliabilityModel",
    "SeriesFailureModel",
    # retrieval
    "AnnFootprintModel",
    "Bm25PairModel",
    "ExactScanFloorModel",
    "GraphNeighborhoodModel",
    "IndexMaintenanceModel",
    "RetrievalPayloadModel",
    "StorageAmplificationModel",
    # sandboxes
    "EnvelopeAmortizationModel",
    "SandboxPoolReserveModel",
    "SyscallTaxModel",
    "WorkspaceProvisioningModel",
    # security
    "InjectionExposureModel",
    "PolicyComparisonModel",
    # serving
    "BatchingOptimizer",
    "ContinuousBatchingModel",
    "InferenceScalingModel",
    "ServingCapacityModel",
    "ServingModel",
    "TailLatencyModel",
    "WeightStreamingModel",
    # serving_scale
    "KvAdmissionModel",
    "PrecisionDividendModel",
    # storage
    "CheckpointStormModel",
    "DataStallModel",
    "HostIoCpuModel",
    "InputDemandModel",
    "ObjectStorageBillModel",
    "PrefetchDepthModel",
    "StorageFootprintModel",
    "StreamVsStageModel",
    "SyntheticFootprintModel",
    "TieredCacheModel",
    "TieredCheckpointModel",
    "UtilizationLossModel",
    # supervision
    "DetectionShortfallModel",
    "SupervisorQueueModel",
    # sustainability
    "CarbonPricedPlacementModel",
    "CascadeEnergyModel",
    "DataMovementEnergyModel",
    "DemandReboundModel",
    "DevelopmentSweepModel",
    "DutyCycledPowerModel",
    "EmbodiedPaybackModel",
    "FacilityPowerModel",
    "GridPlacementModel",
    "PueRetrofitModel",
    "RepeatedWorkEnergyModel",
    "RunFootprintModel",
    "ServiceLifeExtensionModel",
    "ServingCrossoverModel",
    "TemporalShiftModel",
    "WaterUseModel",
    # tool_calls
    "BlindRetryModel",
    "ModelPollingModel",
    "ObservationCarryModel",
    "ToolCatalogSelectionModel",
    "ToolkitGranularityModel",
    # training
    "CheckpointModel",
    "ScalingModel",
    "TrainingMemoryModel",
    "TrainingRunModel",
    "TrainingStateModel",
    # trajectory
    "BrakingBlendModel",
    "ChunkReplacementModel",
    "DynamicWindowModel",
    "HandoffExposureModel",
    "JerkLimitedStopModel",
    "SeamBridgeModel",
    "ShoulderSwingModel",
    "SpillLimitedStopModel",
    "StaleAnchorModel",
    "TrajectoryDeadlinesModel",
    "TwoLinkJointRateModel",
]
