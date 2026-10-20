"""Domain-oriented MLSysIM solver implementations."""

from .base import BaseOptimizer, BaseResolver, BaseSolver, ForwardModel
from .collective import (AllReduceAlgorithmModel, AllToAllDispatchModel, CompressionRunModel, ErrorFeedbackModel,
                         HierarchicalAllReduceModel, OverlapBudgetModel, RingTraceModel)
from .compression import CompressionModel
from .data import DataModel, TransformationModel
from .distributed import DistributedModel, MoERoutingModel, ParallelismOptimizer, TopologyModel
from .economics import EconomicsModel, PlacementOptimizer, ResponsibleEngineeringModel, SustainabilityModel
from .orchestration import JobSizeMixModel, OrchestrationModel, PreemptionTaxModel
from .facility import CoolantFlowModel, OutageCostModel, PowerTransientModel
from .interconnect import (AlphaBetaModel, BisectionModel, CollectiveFitModel, CommunicationIntensityModel,
                           GradientSyncModel, IncastModel, TensorParallelCommModel, TransferLadderModel)
from .performance import DecodeBoundModel, EfficiencyModel, NetworkRooflineModel, SensitivitySolver, SingleNodeModel, SynthesisSolver
from .reliability import FailureIntervalModel, ReliabilityModel
from .reliability import (  # added 2026-10-20
    SeriesFailureModel,
    CheckpointIntervalModel,
    RecoveryBudgetModel,
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
from .storage import (
    StorageFootprintModel,
    TieredCheckpointModel,
    CheckpointStormModel,
    InputDemandModel,
    DataStallModel,
    PrefetchDepthModel,
    TieredCacheModel,
    HostIoCpuModel,
    ObjectStorageBillModel,
    StreamVsStageModel,
    UtilizationLossModel,
    SyntheticFootprintModel,
    StorageBenchmarkDemandModel,
)
from .embodied import StoppingBudgetModel, TractionLossModel  # added 2026-10-20
# Agent trajectories (added 2026-10-20)
from .agents import (AcceptedTaskCostModel, AgentRunSummaryModel, AppendOnlyContextModel,
                     HorizonCompoundingModel, TaskHorizonTrendModel)
# Model calls and agent trajectories (added 2026-10-20)
from .model_call import (CheckerPosteriorModel, ContextAdmissionModel, ModelCallModel, SamplingModel,
                         TokenMixModel, TrajectoryResendModel)
from .trends import DomainComputeGapModel, MemoryWallModel, ScalingSeriesModel, TrainingGrowthModel
from .parallelism import (AlignmentMemoryModel, DataParallelScaleOutModel, ParallelLayoutModel,
                          TensorParallelLayerModel, ZeroShardingModel)
from .trends import TrainingGrowthModel
# Single-machine iron law, Amdahl, and silent-degradation solvers (added 2026-10-20)
from .performance import AmdahlModel, IronLawModel
from .quality import ProxyDriftModel, ShiftedTestSetModel
from .training import CheckpointModel, ScalingModel, TrainingMemoryModel, TrainingRunModel, TrainingStateModel
from .goodput import GoodputModel, RuntimeGoodputModel  # fleet goodput (added 2026-10-20)

__all__ = [
    "GoodputModel",
    "RuntimeGoodputModel",
    "BaseResolver",
    "ForwardModel",
    "BaseSolver",
    "BaseOptimizer",
    "SingleNodeModel",
    "DecodeBoundModel",
    "TransferLadderModel",
    "TensorParallelCommModel",
    "TrainingRunModel",
    "TrainingStateModel",
    "PowerTransientModel",
    "CoolantFlowModel",
    "OutageCostModel",
    "FailureIntervalModel",
    "SeriesFailureModel",
    "CheckpointIntervalModel",
    "RecoveryBudgetModel",
    "GradientSyncModel",
    "CommunicationIntensityModel",
    "AlphaBetaModel",
    "BisectionModel",
    "IncastModel",
    "CollectiveFitModel",
    "AllReduceAlgorithmModel",
    "HierarchicalAllReduceModel",
    "AllToAllDispatchModel",
    "RingTraceModel",
    "ErrorFeedbackModel",
    "CompressionRunModel",
    "OverlapBudgetModel",
    "TrainingGrowthModel",
    "DomainComputeGapModel",
    "StoppingBudgetModel",
    "TractionLossModel",
    "ScalingSeriesModel",
    "MemoryWallModel",
    "ParallelLayoutModel",
    "ZeroShardingModel",
    "DataParallelScaleOutModel",
    "TensorParallelLayerModel",
    "AlignmentMemoryModel",
    # Single-machine iron law, Amdahl, and silent-degradation solvers (added 2026-10-20)
    "IronLawModel",
    "AmdahlModel",
    "ShiftedTestSetModel",
    "ProxyDriftModel",
    "StorageFootprintModel",
    "TieredCheckpointModel",
    "CheckpointStormModel",
    "InputDemandModel",
    "DataStallModel",
    "PrefetchDepthModel",
    "TieredCacheModel",
    "HostIoCpuModel",
    "ObjectStorageBillModel",
    "StreamVsStageModel",
    "UtilizationLossModel",
    "SyntheticFootprintModel",
    "StorageBenchmarkDemandModel",
    # Agent trajectories (added 2026-10-20)
    "HorizonCompoundingModel",
    "AcceptedTaskCostModel",
    "AppendOnlyContextModel",
    "TaskHorizonTrendModel",
    "AgentRunSummaryModel",
    "ContextAdmissionModel",
    "SamplingModel",
    "CheckerPosteriorModel",
    "ModelCallModel",
    "TrajectoryResendModel",
    "TokenMixModel",
    "NetworkRooflineModel",
    "EfficiencyModel",
    "SensitivitySolver",
    "SynthesisSolver",
    "DistributedModel",
    "MoERoutingModel",
    "TopologyModel",
    "ParallelismOptimizer",
    "ReliabilityModel",
    "CheckpointModel",
    "TrainingMemoryModel",
    "ScalingModel",
    "ServingModel",
    "ServingCapacityModel",
    "ContinuousBatchingModel",
    "WeightStreamingModel",
    "TailLatencyModel",
    "InferenceScalingModel",
    "BatchingOptimizer",
    "SustainabilityModel",
    "EconomicsModel",
    "ResponsibleEngineeringModel",
    "PlacementOptimizer",
    "DataModel",
    "TransformationModel",
    "OrchestrationModel",
    "JobSizeMixModel",
    "PreemptionTaxModel",
    "CompressionModel",
]

# --- Physical-plant budgets: stopping, actuation, heat, power (added 2026-10-20) ---
from .plant import (
    SpeedStepModel,
    FreshnessProfileModel,
    BrakingCeilingsModel,
    HubDriveElectricalModel,
    GearedRotorModel,
    GearRatioCostModel,
    ThermalBurstModel,
    StallDeratingModel,
    ContinuousDeratingModel,
    DriveBusModel,
    ZeroFailureModel,
    ThrottledStageModel,
)

__all__ += [
    "SpeedStepModel",
    "FreshnessProfileModel",
    "BrakingCeilingsModel",
    "HubDriveElectricalModel",
    "GearedRotorModel",
    "GearRatioCostModel",
    "ThermalBurstModel",
    "StallDeratingModel",
    "ContinuousDeratingModel",
    "DriveBusModel",
    "ZeroFailureModel",
    "ThrottledStageModel",
]

# --- Test-time deliberation: allocation, pools, checks, search (added 2026-10-20) ---
from .deliberation import (
    ProbeVsAnswerModel,
    AllocationAxesModel,
    BestOfNPoolModel,
    PassAtKEstimateModel,
    CoverageVsVoteModel,
    VerifiedPoolModel,
    ArgmaxNoiseModel,
    ScopedRevisionModel,
    PrunedSearchTreeModel,
    SearchAccountsModel,
    DeadlinePolicyModel,
    RecordedRunCostModel,
)

__all__ += [
    "ProbeVsAnswerModel",
    "AllocationAxesModel",
    "BestOfNPoolModel",
    "PassAtKEstimateModel",
    "CoverageVsVoteModel",
    "VerifiedPoolModel",
    "ArgmaxNoiseModel",
    "ScopedRevisionModel",
    "PrunedSearchTreeModel",
    "SearchAccountsModel",
    "DeadlinePolicyModel",
    "RecordedRunCostModel",
]

# --- Context engineering: budget, dilution, layout, compaction (added 2026-10-20) ---
from .context import (
    ContextBudgetModel,
    AttentionDilutionModel,
    ComprehensionByLengthModel,
    PrefixLayoutBillModel,
    CompactionCadenceModel,
    ContextInvalidationModel,
    ProtectedFieldCheckModel,
)

__all__ += [
    "ContextBudgetModel",
    "AttentionDilutionModel",
    "ComprehensionByLengthModel",
    "PrefixLayoutBillModel",
    "CompactionCadenceModel",
    "ContextInvalidationModel",
    "ProtectedFieldCheckModel",
]

# --- Learned proposers: VLA evaluation, decoders, bus traffic, evidence (added 2026-10-20) ---
from .proposer import (
    VlaEvaluationModel,
    DecoderFamiliesModel,
    ProposerTrafficModel,
    EvidenceHorizonModel,
)

__all__ += [
    "VlaEvaluationModel",
    "DecoderFamiliesModel",
    "ProposerTrafficModel",
    "EvidenceHorizonModel",
]

# --- Serving at scale: traces, frontiers, admission, routing (added 2026-10-20) ---
from .serving_scale import (
    BallsIntoBinsModel,
    KvAdmissionModel,
    PrecisionDividendModel,
    RequestTraceModel,
    ServingFrontierModel,
    TrafficProfileModel,
)

__all__ += [
    "BallsIntoBinsModel",
    "KvAdmissionModel",
    "PrecisionDividendModel",
    "RequestTraceModel",
    "ServingFrontierModel",
    "TrafficProfileModel",
]

# --- Retrieval from long-term memory (added 2026-10-20) ---
from .retrieval import (
    RetrievalPayloadModel,
    Bm25PairModel,
    ExactScanFloorModel,
    AnnFootprintModel,
    GraphNeighborhoodModel,
    IndexMaintenanceModel,
    StorageAmplificationModel,
    MaintainedPolicyCostModel,
    WordOverlapModel,
)

__all__ += [
    "RetrievalPayloadModel",
    "Bm25PairModel",
    "ExactScanFloorModel",
    "AnnFootprintModel",
    "GraphNeighborhoodModel",
    "IndexMaintenanceModel",
    "StorageAmplificationModel",
    "MaintainedPolicyCostModel",
    "WordOverlapModel",
]

# --- Tool calling (added 2026-10-20) ---
from .tool_calls import (
    BlindRetryModel,
    ObservationCarryModel,
    ModelPollingModel,
    ToolCatalogSelectionModel,
    ToolkitGranularityModel,
    IntegrationAdaptersModel,
)

__all__ += [
    "BlindRetryModel",
    "ObservationCarryModel",
    "ModelPollingModel",
    "ToolCatalogSelectionModel",
    "ToolkitGranularityModel",
    "IntegrationAdaptersModel",
]

# --- KV cache management (added 2026-10-20) ---
from .kv_cache import (
    KvAttentionDesignModel,
    PagedAllocationModel,
    PrefixRebuildModel,
    ChunkedPrefillModel,
    PausedKvStateModel,
    KvBlockSharingModel,
    KvResidencyArchetypeModel,
)

__all__ += [
    "KvAttentionDesignModel",
    "PagedAllocationModel",
    "PrefixRebuildModel",
    "ChunkedPrefillModel",
    "PausedKvStateModel",
    "KvBlockSharingModel",
    "KvResidencyArchetypeModel",
]

# --- The real-time permission path: phase, response time, fieldbus, lease (added 2026-10-20) ---
from .realtime import (
    PhaseLagBudgetModel,
    ResponseTimeModel,
    CanFdBusModel,
    EtherCatCycleModel,
    LeaseBudgetModel,
    LatencyTailModel,
)

__all__ += [
    "PhaseLagBudgetModel",
    "ResponseTimeModel",
    "CanFdBusModel",
    "EtherCatCycleModel",
    "LeaseBudgetModel",
    "LatencyTailModel",
]

# --- Agent sandboxes (added 2026-10-20) ---
from .sandboxes import (
    SyscallTaxModel,
    EnvelopeAmortizationModel,
    WorkspaceProvisioningModel,
    SandboxPoolReserveModel,
    BootTimeCdfModel,
)

__all__ += [
    "SyscallTaxModel",
    "EnvelopeAmortizationModel",
    "WorkspaceProvisioningModel",
    "SandboxPoolReserveModel",
    "BootTimeCdfModel",
]
