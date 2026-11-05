"""Stable public scenario registry.

Typed inputs live in ``scenario_definitions``; named cases live in
``scenario_catalog``. Explicit bindings retain historical order and identity.
"""

from pydantic import BaseModel as BaseModel, ConfigDict as ConfigDict
from typing import Optional as Optional, Union as Union
from ..core.units import Q_ as Q_
from ..core.types import Quantity as Quantity
from ..models.types import Workload as Workload
from ..hardware.types import HardwareNode as HardwareNode
from ..systems.types import Fleet as Fleet
from ..core.exceptions import OOMError as OOMError, SLAViolation as SLAViolation
from ..core.registry import Registry as Registry
from .evaluation import SystemEvaluation as SystemEvaluation, EvaluationLevel as EvaluationLevel
from .scenario_definitions.base import Scenario as Scenario
from .scenario_catalog.base import applications as _base_applications
from .scenario_catalog.serving import foundations as _serving_foundations
from .scenario_catalog.communication import foundations as _communication_foundations
from .scenario_catalog.training import foundations as _training_foundations
from .scenario_catalog.sustainability import foundations as _sustainability_foundations
from .scenario_catalog.reliability import foundations as _reliability_foundations
from .scenario_catalog.optimization import foundations as _optimization_foundations
from .scenario_catalog.training import data_parallel as _training_data_parallel
from .scenario_catalog.communication import networks as _communication_networks
from .scenario_catalog.data import storage as _data_storage
from .scenario_catalog.robotics import stopping as _robotics_stopping
from .scenario_catalog.agents import agents as _agents_agents
from .scenario_catalog.agents import model_calls as _agents_model_calls
from .scenario_catalog.communication import collectives as _communication_collectives
from .scenario_catalog.training import distributed as _training_distributed
from .scenario_catalog.optimization import single_node as _optimization_single_node
from .scenario_catalog.optimization import deployment as _optimization_deployment
from .scenario_catalog.robotics import plant as _robotics_plant
from .scenario_catalog.reliability import fault_tolerance as _reliability_fault_tolerance
from .scenario_catalog.optimization import orchestration as _optimization_orchestration
from .scenario_catalog.optimization import lifecycle as _optimization_lifecycle
from .scenario_catalog.agents import deliberation as _agents_deliberation
from .scenario_catalog.agents import context as _agents_context
from .scenario_catalog.robotics import proposers as _robotics_proposers
from .scenario_catalog.data import labeling as _data_labeling
from .scenario_catalog.data import delivery as _data_delivery
from .scenario_catalog.serving import goodput as _serving_goodput
from .scenario_catalog.serving import scale as _serving_scale
from .scenario_catalog.serving import kv_cache as _serving_kv_cache
from .scenario_catalog.agents import retrieval as _agents_retrieval
from .scenario_catalog.robotics import demonstrations as _robotics_demonstrations
from .scenario_catalog.robotics import realtime as _robotics_realtime
from .scenario_catalog.agents import tools as _agents_tools
from .scenario_catalog.agents import sandboxes as _agents_sandboxes
from .scenario_catalog.training import federated as _training_federated
from .scenario_catalog.robotics import policy_training as _robotics_policy_training
from .scenario_catalog.training import layers as _training_layers
from .scenario_catalog.training import neural as _training_neural
from .scenario_catalog.economics import operations as _economics_operations
from .scenario_catalog.serving import transformers as _serving_transformers
from .scenario_catalog.optimization import framework as _optimization_framework
from .scenario_catalog.reliability import security as _reliability_security
from .scenario_catalog.statistics import robustness as _statistics_robustness
from .scenario_catalog.agents import security as _agents_security
from .scenario_catalog.agents import harness as _agents_harness
from .scenario_catalog.statistics import closed_loop as _statistics_closed_loop
from .scenario_catalog.robotics import perception as _robotics_perception
from .scenario_catalog.data import selection as _data_selection
from .scenario_catalog.statistics import responsibility as _statistics_responsibility
from .scenario_catalog.statistics import fairness as _statistics_fairness
from .scenario_catalog.reliability import durability as _reliability_durability
from .scenario_catalog.reliability import recovery as _reliability_recovery
from .scenario_catalog.statistics import evaluation as _statistics_evaluation
from .scenario_catalog.robotics import spatial_memory as _robotics_spatial_memory
from .scenario_catalog.robotics import grounding as _robotics_grounding
from .scenario_catalog.sustainability import sustainability as _sustainability_sustainability
from .scenario_catalog.optimization import expansion as _optimization_expansion
from .scenario_catalog.robotics import stopping_enforcement as _robotics_stopping_enforcement
from .scenario_catalog.robotics import placement as _robotics_placement
from .scenario_catalog.agents import observability as _agents_observability
from .scenario_catalog.training import working_set as _training_working_set
from .scenario_catalog.optimization import compression as _optimization_compression
from .scenario_catalog.optimization import acceleration as _optimization_acceleration
from .scenario_catalog.optimization import mapping as _optimization_mapping
from .scenario_catalog.robotics import trajectories as _robotics_trajectories
from .scenario_catalog.data import curation as _data_curation
from .scenario_catalog.training import fine_tuning as _training_fine_tuning
from .scenario_catalog.optimization import quantization as _optimization_quantization
from .scenario_catalog.robotics import supervision as _robotics_supervision
from .scenario_catalog.statistics import fault_injection as _statistics_fault_injection
from .scenario_catalog.statistics import benchmarks as _statistics_benchmarks
from .scenario_catalog.statistics import measurement as _statistics_measurement
from .scenario_catalog.serving import request_scheduling as _serving_request_scheduling
from .scenario_catalog.training import reinforcement as _training_reinforcement
from .scenario_catalog.statistics import release as _statistics_release
from .scenario_catalog.agents import coordination as _agents_coordination
from .scenario_catalog.robotics import oversight as _robotics_oversight
from .scenario_catalog.reliability import release as _reliability_release
from .scenario_catalog.serving import token_streams as _serving_token_streams
from .scenario_catalog.economics import cost_accounting as _economics_cost_accounting
from .scenario_catalog.robotics import premises as _robotics_premises
from .scenario_catalog.agents import reliability as _agents_reliability
from .scenario_catalog.training import synthesis as _training_synthesis
from .scenario_catalog.statistics import synthesis as _statistics_synthesis
from .scenario_catalog.optimization import synthesis as _optimization_synthesis
from .scenario_catalog.statistics import monitoring as _statistics_monitoring
from .scenario_catalog.economics import duties as _economics_duties


from .scenario_catalog.optimization import controlled_studies as _controlled_studies


class Scenarios(Registry):
    """
    Executable scenario registry.

    Each entry composes one ``Models.*`` workload, one ``Hardware.*`` or
    ``Systems.*`` target, and scenario-local constraints such as latency or
    power budget. Non-executable sourced anchors live separately under
    ``ReferenceStats``.
    """

    __registry_list_exclude__ = frozenset({"DeploymentJourneys"})
    ControlledTaskFactors = _controlled_studies.ControlledTaskFactors
    DeploymentJourneys = _controlled_studies.DeploymentJourneys
    PlacementStudy = _controlled_studies.PlacementStudy
    ShallowPowerLawExample = _controlled_studies.ShallowPowerLawExample


    SmartDoorbell = _base_applications.SmartDoorbell
    TinySensor = _base_applications.TinySensor
    AutonomousVehicle = _base_applications.AutonomousVehicle
    AutonomousVehicle_Waymo = _base_applications.AutonomousVehicle_Waymo
    MobileHealth = _base_applications.MobileHealth
    FederatedMobileVision = _base_applications.FederatedMobileVision
    MobileAssistant = _base_applications.MobileAssistant
    LocalTraining = _base_applications.LocalTraining
    FrontierTraining = _base_applications.FrontierTraining
    DenseLLMPretrainingReference = _base_applications.DenseLLMPretrainingReference
    ChatbotServing = _base_applications.ChatbotServing
    KeywordSpotting = _base_applications.KeywordSpotting
    Llama3_70B_DecodeH100 = _serving_foundations.Llama3_70B_DecodeH100
    Llama2_70B_DecodeH100 = _serving_foundations.Llama2_70B_DecodeH100
    Llama3_70B_DecodeH100Batch64 = _serving_foundations.Llama3_70B_DecodeH100Batch64
    Llama3_70B_ServingH100 = _serving_foundations.Llama3_70B_ServingH100
    BufferTransfer10GB = _communication_foundations.BufferTransfer10GB
    Gpt3TensorParallelStepDgxH100 = _communication_foundations.Gpt3TensorParallelStepDgxH100
    Gpt3PretrainingFrontier8K = _training_foundations.Gpt3PretrainingFrontier8K
    Gpt3TrainingStateTp8 = _training_foundations.Gpt3TrainingStateTp8
    H100BarrierLoadStep = _sustainability_foundations.H100BarrierLoadStep
    AiRackAirCooling = _sustainability_foundations.AiRackAirCooling
    Nvl72WaterCooling = _sustainability_foundations.Nvl72WaterCooling
    H100AirCooling = _sustainability_foundations.H100AirCooling
    CduOutageTraining10K = _sustainability_foundations.CduOutageTraining10K
    GpuFailuresTraining10K = _reliability_foundations.GpuFailuresTraining10K
    NodeFailuresTraining1K = _reliability_foundations.NodeFailuresTraining1K
    V100ToB200 = _optimization_foundations.V100ToB200
    V100ToH100 = _optimization_foundations.V100ToH100
    PeakPlannedTrainingBudget = _training_foundations.PeakPlannedTrainingBudget
    Gpt3GradientSync = _training_data_parallel.Gpt3GradientSync
    Gpt3GradientSyncFasterAccelerator = _training_data_parallel.Gpt3GradientSyncFasterAccelerator
    Gpt3DataParallelIntensity = _training_data_parallel.Gpt3DataParallelIntensity
    NodeAvailability1K = _training_data_parallel.NodeAvailability1K
    NodeAvailability10K = _training_data_parallel.NodeAvailability10K
    FourNinesAvailability10K = _training_data_parallel.FourNinesAvailability10K
    CoordinationOverheadExample = _training_data_parallel.CoordinationOverheadExample
    Gpt3PublishedRun = _training_data_parallel.Gpt3PublishedRun
    SpineOpticsTraining10K = _communication_networks.SpineOpticsTraining10K
    GoBackNResendNdr = _communication_networks.GoBackNResendNdr
    AlphaBetaNdrRoce = _communication_networks.AlphaBetaNdrRoce
    AlphaBetaHdrNdrSmallMessage = _communication_networks.AlphaBetaHdrNdrSmallMessage
    RingAllReduceAcross4to1Spine = _communication_networks.RingAllReduceAcross4to1Spine
    RingAllReduceAcross2to1Spine = _communication_networks.RingAllReduceAcross2to1Spine
    FatTreeNdrSwitch = _communication_networks.FatTreeNdrSwitch
    TorusVsFatTree4K = _communication_networks.TorusVsFatTree4K
    RailVsLeafSpineHops = _communication_networks.RailVsLeafSpineHops
    IncastSuperPodUnit = _communication_networks.IncastSuperPodUnit
    TransceiverDegradationRoce4K = _communication_networks.TransceiverDegradationRoce4K
    DegradedLinksReferenceFabric = _communication_networks.DegradedLinksReferenceFabric
    DegradedTransceiverRun1K = _communication_networks.DegradedTransceiverRun1K
    SriovNdrNic = _communication_networks.SriovNdrNic
    HalfRateLinkDiagnosis = _communication_networks.HalfRateLinkDiagnosis
    Gpt3StorageFootprint = _data_storage.Gpt3StorageFootprint
    Gpt3TieredCheckpoint2K = _data_storage.Gpt3TieredCheckpoint2K
    Llama2_70BCheckpointStorm1K = _data_storage.Llama2_70BCheckpointStorm1K
    TextInputDemand2K = _data_storage.TextInputDemand2K
    ImageInputDemand2K = _data_storage.ImageInputDemand2K
    OverlappedInputStall = _data_storage.OverlappedInputStall
    PrefetchDepthByTier = _data_storage.PrefetchDepthByTier
    HotTierCacheExample = _data_storage.HotTierCacheExample
    MultimodalCorpusNodeCache = _data_storage.MultimodalCorpusNodeCache
    GpuDirectStorageNodeCpu = _data_storage.GpuDirectStorageNodeCpu
    StorageTailFanout100 = _data_storage.StorageTailFanout100
    ImageFileMetadataMeltdown = _data_storage.ImageFileMetadataMeltdown
    ObjectStorage100TB = _data_storage.ObjectStorage100TB
    VisionCorpusStreamVsStage = _data_storage.VisionCorpusStreamVsStage
    InputStallIdleCost1K = _data_storage.InputStallIdleCost1K
    SyntheticDataProvenance = _data_storage.SyntheticDataProvenance
    Gpt3WeightsColdStart = _data_storage.Gpt3WeightsColdStart
    Gpt3RolloutFanout100 = _data_storage.Gpt3RolloutFanout100
    NodeStripeCodecs = _data_storage.NodeStripeCodecs
    WarehouseAisleStopAtDriveLimit = _robotics_stopping.WarehouseAisleStopAtDriveLimit
    WarehouseAisleStopHalfSpeed = _robotics_stopping.WarehouseAisleStopHalfSpeed
    WarehouseInferenceHiccup = _robotics_stopping.WarehouseInferenceHiccup
    TempeActionSuppression = _robotics_stopping.TempeActionSuppression
    HydraulicFluidSlickTraction = _robotics_stopping.HydraulicFluidSlickTraction
    UncheckedRepairHorizon = _agents_agents.UncheckedRepairHorizon
    HorizonCompounding30Steps = _agents_agents.HorizonCompounding30Steps
    RepairAttemptCostLevers = _agents_agents.RepairAttemptCostLevers
    AppendOnlyRepairContext = _agents_agents.AppendOnlyRepairContext
    Llama3_70B_Int8DecodeH100 = _agents_model_calls.Llama3_70B_Int8DecodeH100
    CodingAgentResend30Turns = _agents_model_calls.CodingAgentResend30Turns
    CodingAgentResend60Turns = _agents_model_calls.CodingAgentResend60Turns
    CodingAgentResend90Turns = _agents_model_calls.CodingAgentResend90Turns
    CodingAgentFirstTurnCall = _agents_model_calls.CodingAgentFirstTurnCall
    CodingAgentWholeFileEdit = _agents_model_calls.CodingAgentWholeFileEdit
    CodingAgentLocalizedEdit = _agents_model_calls.CodingAgentLocalizedEdit
    SupportAgentTurnCall = _agents_model_calls.SupportAgentTurnCall
    CanceledStreamDecodeSavings = _agents_model_calls.CanceledStreamDecodeSavings
    MultimodalContextAdmission = _agents_model_calls.MultimodalContextAdmission
    TemperatureNucleusExample = _agents_model_calls.TemperatureNucleusExample
    PatchCheckerPosterior = _agents_model_calls.PatchCheckerPosterior
    PatchCheckerHalvedFalseAccept = _agents_model_calls.PatchCheckerHalvedFalseAccept
    CriticalSizeByInterconnect = _communication_collectives.CriticalSizeByInterconnect
    RoutingMessageFusionNdr = _communication_collectives.RoutingMessageFusionNdr
    DdpBucketRingNdr = _communication_collectives.DdpBucketRingNdr
    GradientBufferAllReduceNdr = _communication_collectives.GradientBufferAllReduceNdr
    Llama2_70BGradientRing64Ndr = _communication_collectives.Llama2_70BGradientRing64Ndr
    RingVsTreeSmallBuffer = _communication_collectives.RingVsTreeSmallBuffer
    HierarchicalAllReduce8x8DgxH100 = _communication_collectives.HierarchicalAllReduce8x8DgxH100
    HierarchicalAllReduceBucket100MB = _communication_collectives.HierarchicalAllReduceBucket100MB
    HierarchicalAllReduceReferenceCluster = _communication_collectives.HierarchicalAllReduceReferenceCluster
    MoeDispatch8Gpu64Experts = _communication_collectives.MoeDispatch8Gpu64Experts
    RingTraceFourGpu = _communication_collectives.RingTraceFourGpu
    ErrorFeedbackFiveSteps = _communication_collectives.ErrorFeedbackFiveSteps
    CompressionPayback8x = _communication_collectives.CompressionPayback8x
    Int8CompressionRun = _communication_collectives.Int8CompressionRun
    OverlapBudget7B64Gpu = _communication_collectives.OverlapBudget7B64Gpu
    OverlapBudget7B64GpuIdealBuckets = _communication_collectives.OverlapBudget7B64GpuIdealBuckets
    Gpt3DataParallel1K = _training_distributed.Gpt3DataParallel1K
    Gpt3TensorDataParallel1K = _training_distributed.Gpt3TensorDataParallel1K
    Gpt3ThreeDParallel1K = _training_distributed.Gpt3ThreeDParallel1K
    Gpt3ThreeDParallel16K = _training_distributed.Gpt3ThreeDParallel16K
    Gpt3ActivationsMicrobatch4 = _training_distributed.Gpt3ActivationsMicrobatch4
    Gpt3ActivationsMicrobatch1 = _training_distributed.Gpt3ActivationsMicrobatch1
    Llama2_7B_ZeroStages64 = _training_distributed.Llama2_7B_ZeroStages64
    Gpt3Zero3On64 = _training_distributed.Gpt3Zero3On64
    Gpt2DataParallelOneNode = _training_distributed.Gpt2DataParallelOneNode
    Gpt2DataParallelFourNodes10GbE = _training_distributed.Gpt2DataParallelFourNodes10GbE
    Gpt2GradientAccumulationOneNode = _training_distributed.Gpt2GradientAccumulationOneNode
    GPipeFourStageWalkthrough = _training_distributed.GPipeFourStageWalkthrough
    Pipeline8Stage32Micro = _training_distributed.Pipeline8Stage32Micro
    Gpt3TensorParallelLayerTp8 = _training_distributed.Gpt3TensorParallelLayerTp8
    H100RingAttentionChunk = _training_distributed.H100RingAttentionChunk
    TransformerBsp8Workers = _training_distributed.TransformerBsp8Workers
    TransformerBsp64Workers = _training_distributed.TransformerBsp64Workers
    TransformerSsp64Workers = _training_distributed.TransformerSsp64Workers
    Llama2_70B_PpoAlignment = _training_distributed.Llama2_70B_PpoAlignment
    GpuFailuresTraining1K = _training_distributed.GpuFailuresTraining1K
    StragglerFreeStep1K = _training_distributed.StragglerFreeStep1K
    ResidualOnePercent512 = _training_distributed.ResidualOnePercent512
    ImageNetInputSpace = _optimization_single_node.ImageNetInputSpace
    Gpt3TrainingTime1KA100 = _optimization_single_node.Gpt3TrainingTime1KA100
    Gpt3TrainingTime1KA100HigherUtilization = _optimization_single_node.Gpt3TrainingTime1KA100HigherUtilization
    IronLawMemoryBoundExample = _optimization_single_node.IronLawMemoryBoundExample
    IronLawComputeHeavyExample = _optimization_single_node.IronLawComputeHeavyExample
    ModelSpeedupRequestPath = _optimization_single_node.ModelSpeedupRequestPath
    RoboticArmRegionalCloud = _optimization_deployment.RoboticArmRegionalCloud
    CrossCountryFiberPath = _optimization_deployment.CrossCountryFiberPath
    FactoryCameras1080p = _optimization_deployment.FactoryCameras1080p
    Drone4KObstacleOffload = _optimization_deployment.Drone4KObstacleOffload
    ResNet50A100Batch1 = _optimization_deployment.ResNet50A100Batch1
    ResNet50PhoneNpuBatch1 = _optimization_deployment.ResNet50PhoneNpuBatch1
    TPUv4Pod4096 = _optimization_deployment.TPUv4Pod4096
    IllustrativeMemoryWall = _optimization_deployment.IllustrativeMemoryWall
    AlwaysOnObjectDetectorPhone = _optimization_deployment.AlwaysOnObjectDetectorPhone
    UnoptimizedVisionModelPhone = _optimization_deployment.UnoptimizedVisionModelPhone
    MobileNetV2PhoneBatch1 = _optimization_deployment.MobileNetV2PhoneBatch1
    EnergyPerInferenceTiers = _optimization_deployment.EnergyPerInferenceTiers
    RadioUploadVsMobileNetV2 = _optimization_deployment.RadioUploadVsMobileNetV2
    SmartDoorbellFrame = _optimization_deployment.SmartDoorbellFrame
    SmartDoorbellResponsiveChime = _optimization_deployment.SmartDoorbellResponsiveChime
    EmergencyBraking100kmh = _optimization_deployment.EmergencyBraking100kmh
    VoiceAssistantBillionDevices = _optimization_deployment.VoiceAssistantBillionDevices
    PhoneCameraPipeline = _optimization_deployment.PhoneCameraPipeline
    WarehouseBaseStopBodyTerms = _robotics_plant.WarehouseBaseStopBodyTerms
    WarehouseBaseStopFullDelay = _robotics_plant.WarehouseBaseStopFullDelay
    WarehouseBaseStopFastSetpoint = _robotics_plant.WarehouseBaseStopFastSetpoint
    WarehouseBaseSpeedStep = _robotics_plant.WarehouseBaseSpeedStep
    WarehouseBaseBrakingCeilings = _robotics_plant.WarehouseBaseBrakingCeilings
    WarehouseBaseHubDriveElectrical = _robotics_plant.WarehouseBaseHubDriveElectrical
    WristCameraGraspAgeProfile = _robotics_plant.WristCameraGraspAgeProfile
    GearedRotorFlywheel = _robotics_plant.GearedRotorFlywheel
    PandaClassJointGearRatios = _robotics_plant.PandaClassJointGearRatios
    PandaClassJointBurst = _robotics_plant.PandaClassJointBurst
    PandaClassElbowStall = _robotics_plant.PandaClassElbowStall
    SealedJointCatalogDerating = _robotics_plant.SealedJointCatalogDerating
    WarehouseDriveBusStop = _robotics_plant.WarehouseDriveBusStop
    CleanLoadedStopCampaign = _robotics_plant.CleanLoadedStopCampaign
    ThrottledNavigationBackbone = _robotics_plant.ThrottledNavigationBackbone
    GpuFailureLadder = _reliability_fault_tolerance.GpuFailureLadder
    FullSystemFailures10K = _reliability_fault_tolerance.FullSystemFailures10K
    Gpt3CheckpointInterval10K = _reliability_fault_tolerance.Gpt3CheckpointInterval10K
    FiveMinuteCheckpoint1K = _reliability_fault_tolerance.FiveMinuteCheckpoint1K
    FifteenMinuteCheckpoint1K = _reliability_fault_tolerance.FifteenMinuteCheckpoint1K
    Gpt3RecoveryBudget1K = _reliability_fault_tolerance.Gpt3RecoveryBudget1K
    Gpt3RecoveryBudget10K = _reliability_fault_tolerance.Gpt3RecoveryBudget10K
    HotSpareNode10K = _reliability_fault_tolerance.HotSpareNode10K
    HalfSpeedStraggler1K = _reliability_fault_tolerance.HalfSpeedStraggler1K
    ServingReplica99 = _reliability_fault_tolerance.ServingReplica99
    ServingReplicas99x2 = _reliability_fault_tolerance.ServingReplicas99x2
    ServingReplicas99x3 = _reliability_fault_tolerance.ServingReplicas99x3
    GpuAvailability10K = _reliability_fault_tolerance.GpuAvailability10K
    InferenceReadinessProbe = _reliability_fault_tolerance.InferenceReadinessProbe
    SdcExposure10K = _reliability_fault_tolerance.SdcExposure10K
    SdcExposure100K = _reliability_fault_tolerance.SdcExposure100K
    Mg1MixedJobQueueAt80 = _optimization_orchestration.Mg1MixedJobQueueAt80
    Mg1ExponentialJobQueueAt80 = _optimization_orchestration.Mg1ExponentialJobQueueAt80
    DrfTwoTenantExample = _optimization_orchestration.DrfTwoTenantExample
    SixGpuJobsOnDgxH100Nodes = _optimization_orchestration.SixGpuJobsOnDgxH100Nodes
    FragmentedEightNodeSnapshot = _optimization_orchestration.FragmentedEightNodeSnapshot
    ReservedGpusIdleThreeWeeks = _optimization_orchestration.ReservedGpusIdleThreeWeeks
    PartialGangHold = _optimization_orchestration.PartialGangHold
    Training10KIdleHour = _optimization_orchestration.Training10KIdleHour
    MixedFleetPolicyFix1K = _optimization_orchestration.MixedFleetPolicyFix1K
    ForcedPreemption175BFineTune64 = _optimization_orchestration.ForcedPreemption175BFineTune64
    SpotInterruptionEffectiveCost = _optimization_orchestration.SpotInterruptionEffectiveCost
    Llama2_70BSpotRun512 = _optimization_orchestration.Llama2_70BSpotRun512
    ElasticResizeDecision = _optimization_orchestration.ElasticResizeDecision
    ServingConsolidationA100 = _optimization_orchestration.ServingConsolidationA100
    DiurnalServingLending1K = _optimization_orchestration.DiurnalServingLending1K
    TensorParallelExchange1GB = _optimization_orchestration.TensorParallelExchange1GB
    TeamQuotaLending500 = _optimization_orchestration.TeamQuotaLending500
    Gpt3SelfContainedReplicaDgxH100 = _optimization_orchestration.Gpt3SelfContainedReplicaDgxH100
    DenseLLMPretrainingStep = _optimization_orchestration.DenseLLMPretrainingStep
    DenseLLMPretrainingStep8K = _optimization_orchestration.DenseLLMPretrainingStep8K
    DenseLLMPretrainingStep2K = _optimization_orchestration.DenseLLMPretrainingStep2K
    RetinopathyScreenerRequirements = _optimization_lifecycle.RetinopathyScreenerRequirements
    RetinopathyScreenerLateMemoryLimit = _optimization_lifecycle.RetinopathyScreenerLateMemoryLimit
    SlowVersusFastIteration = _optimization_lifecycle.SlowVersusFastIteration
    RuralClinicRetinalUpload = _optimization_lifecycle.RuralClinicRetinalUpload
    RetinopathyScreeningCloudVersusEdge = _optimization_lifecycle.RetinopathyScreeningCloudVersusEdge
    ProbeVsGeneratedExplanation = _agents_deliberation.ProbeVsGeneratedExplanation
    OutputCeilingThreeAllocations = _agents_deliberation.OutputCeilingThreeAllocations
    IndependentCandidatePools = _agents_deliberation.IndependentCandidatePools
    SparseValidPoolFirstPass = _agents_deliberation.SparseValidPoolFirstPass
    SparseValidPoolHalvedFalseAccept = _agents_deliberation.SparseValidPoolHalvedFalseAccept
    SparseValidPoolDoubledValidRate = _agents_deliberation.SparseValidPoolDoubledValidRate
    ArgmaxScoreNoise = _agents_deliberation.ArgmaxScoreNoise
    ScopedPlanRevision = _agents_deliberation.ScopedPlanRevision
    StepCheckedSearchTree = _agents_deliberation.StepCheckedSearchTree
    FlatVsBeamSearch = _agents_deliberation.FlatVsBeamSearch
    DeadlineBoundRepairPolicies = _agents_deliberation.DeadlineBoundRepairPolicies
    RepairContextBudget = _agents_context.RepairContextBudget
    ToyHeadAttentionDilution = _agents_context.ToyHeadAttentionDilution
    ToyHeadLookalikeExcerpt = _agents_context.ToyHeadLookalikeExcerpt
    RepairPrefixLayoutBill = _agents_context.RepairPrefixLayoutBill
    RepairCompactionWatermarks = _agents_context.RepairCompactionWatermarks
    SupersededExcerptInvalidation = _agents_context.SupersededExcerptInvalidation
    CompactionConstraintRetention = _agents_context.CompactionConstraintRetention
    ProtectedFieldCheckPayback = _agents_context.ProtectedFieldCheckPayback
    IntentModelEvaluationFp16 = _robotics_proposers.IntentModelEvaluationFp16
    IntentModelEvaluation8Bit = _robotics_proposers.IntentModelEvaluation8Bit
    IntentModelEvaluationInt4 = _robotics_proposers.IntentModelEvaluationInt4
    IntentModelEvaluationParallel = _robotics_proposers.IntentModelEvaluationParallel
    ThreeViewVlaTokens = _robotics_proposers.ThreeViewVlaTokens
    IntentModelEvaluationThreeViews = _robotics_proposers.IntentModelEvaluationThreeViews
    IntentModelFiveHertz = _robotics_proposers.IntentModelFiveHertz
    IntentModelParallelFiveHertz = _robotics_proposers.IntentModelParallelFiveHertz
    ThreeViewKvHistory = _robotics_proposers.ThreeViewKvHistory
    IntentBackboneDecoderFamilies = _robotics_proposers.IntentBackboneDecoderFamilies
    WarehouseProposerTraffic = _robotics_proposers.WarehouseProposerTraffic
    ConveyorMugEvidenceHorizon = _robotics_proposers.ConveyorMugEvidenceHorizon
    ConveyorMugEvidenceHorizonHalfRenewal = _robotics_proposers.ConveyorMugEvidenceHorizonHalfRenewal
    OpenVlaOrinQuantization = _robotics_proposers.OpenVlaOrinQuantization
    ProposerConfidenceRegimes = _robotics_proposers.ProposerConfidenceRegimes
    KeywordSpotterTargets = _data_labeling.KeywordSpotterTargets
    KeywordSpotterSampleRateChoice = _data_labeling.KeywordSpotterSampleRateChoice
    KeywordSpotterMfccChoice = _data_labeling.KeywordSpotterMfccChoice
    KeywordSpotterExampleCountChoice = _data_labeling.KeywordSpotterExampleCountChoice
    KeywordSpotterFalseWakes = _data_labeling.KeywordSpotterFalseWakes
    KeywordSpotterFalseWakeEvidence = _data_labeling.KeywordSpotterFalseWakeEvidence
    KeywordSpotterDataBudget = _data_labeling.KeywordSpotterDataBudget
    ImageNetLabelingVersusResNetRun = _data_labeling.ImageNetLabelingVersusResNetRun
    FullHdMaskVersusBoxes = _data_labeling.FullHdMaskVersusBoxes
    CityscapesMasksForMillionFrames = _data_labeling.CityscapesMasksForMillionFrames
    MswcByHand = _data_labeling.MswcByHand
    BinaryLabelChanceAgreement = _data_labeling.BinaryLabelChanceAgreement
    WakeWordLabelQualityDrift = _data_labeling.WakeWordLabelQualityDrift
    ActiveLearningTenMillionPool = _data_labeling.ActiveLearningTenMillionPool
    MillionEventBatchVersusStreaming = _data_labeling.MillionEventBatchVersusStreaming
    PetabyteCrossRegionMove = _data_delivery.PetabyteCrossRegionMove
    ResNet50A100InputFeed = _data_delivery.ResNet50A100InputFeed
    ResNet50EightA100InputFeed = _data_delivery.ResNet50EightA100InputFeed
    RecommendationFeatureLookups = _data_delivery.RecommendationFeatureLookups
    MswcCorpusStorageTiers = _data_delivery.MswcCorpusStorageTiers
    FraudModelColumnProjection = _data_delivery.FraudModelColumnProjection
    GzipVersusSnappyEpochs = _data_delivery.GzipVersusSnappyEpochs
    LoudnessDriftKs1000 = _data_delivery.LoudnessDriftKs1000
    SharedPoolSchedulingSnapshot = _serving_goodput.SharedPoolSchedulingSnapshot
    Llama2_70BJobFailures128 = _serving_goodput.Llama2_70BJobFailures128
    Llama2_70BRuntimeGoodput128 = _serving_goodput.Llama2_70BRuntimeGoodput128
    Llama2_70BFleetMfu128 = _serving_goodput.Llama2_70BFleetMfu128
    Llama2_70BSharedPoolGoodput128 = _serving_goodput.Llama2_70BSharedPoolGoodput128
    Llama3_405BGoodput16K = _serving_goodput.Llama3_405BGoodput16K
    Llama3_405BOnSharedPool = _serving_goodput.Llama3_405BOnSharedPool
    Llama2_70BDecodeFloorTp8 = _serving_goodput.Llama2_70BDecodeFloorTp8
    LayerNormMeasuredH100 = _serving_goodput.LayerNormMeasuredH100
    Llama2_70BNodeTensorParallelStep = _serving_goodput.Llama2_70BNodeTensorParallelStep
    Llama2_70BDecodeAllReduceTp8 = _serving_goodput.Llama2_70BDecodeAllReduceTp8
    Decode13BCompilationPayback = _serving_goodput.Decode13BCompilationPayback
    NsightTraceVolume1K = _serving_goodput.NsightTraceVolume1K
    LlmServingFirstYear = _serving_scale.LlmServingFirstYear
    InteractiveRequestServingTax = _serving_scale.InteractiveRequestServingTax
    VisionBatchingKnee = _serving_scale.VisionBatchingKnee
    LlmDecodeBatchingKnee = _serving_scale.LlmDecodeBatchingKnee
    VisionServingLittlesLaw = _serving_scale.VisionServingLittlesLaw
    Llama3_70B_ReplicaDecodeBudget = _serving_scale.Llama3_70B_ReplicaDecodeBudget
    Llama3_70B_ReplicaSpeculation = _serving_scale.Llama3_70B_ReplicaSpeculation
    RepairRequestZones = _serving_kv_cache.RepairRequestZones
    Llama3_70BLongContextAdmissionDgxH100 = _serving_kv_cache.Llama3_70BLongContextAdmissionDgxH100
    ChatbotSharedSystemPrompt = _serving_kv_cache.ChatbotSharedSystemPrompt
    Llama2_70BPrecisionDividendGqaTp8 = _serving_kv_cache.Llama2_70BPrecisionDividendGqaTp8
    Llama2_70BPrecisionDividendMhaTp8 = _serving_kv_cache.Llama2_70BPrecisionDividendMhaTp8
    LongPromptChunkedPrefill = _serving_kv_cache.LongPromptChunkedPrefill
    Llama2_70BPrefillTensorParallel8 = _serving_kv_cache.Llama2_70BPrefillTensorParallel8
    HeterogeneousH100A100Routing = _serving_kv_cache.HeterogeneousH100A100Routing
    ServingFanout100Leaves = _serving_kv_cache.ServingFanout100Leaves
    RecommenderTailEvents = _serving_kv_cache.RecommenderTailEvents
    Llama70BReplicaColdStart = _serving_kv_cache.Llama70BReplicaColdStart
    SharedPoolNoisyNeighbor = _serving_kv_cache.SharedPoolNoisyNeighbor
    ReasoningTokenLatency = _serving_kv_cache.ReasoningTokenLatency
    FourRequestDecodeBatch = _serving_kv_cache.FourRequestDecodeBatch
    RetrievalRecordsInAllowance = _agents_retrieval.RetrievalRecordsInAllowance
    Bm25ShortAgainstLongFile = _agents_retrieval.Bm25ShortAgainstLongFile
    ExactScanTenMillionVectors = _agents_retrieval.ExactScanTenMillionVectors
    AnnIndexesTenMillionVectors = _agents_retrieval.AnnIndexesTenMillionVectors
    TypedTraversalDepthThree = _agents_retrieval.TypedTraversalDepthThree
    RepairEditIndexMaintenance = _agents_retrieval.RepairEditIndexMaintenance
    LexicalGraphAgainstDenseStore = _agents_retrieval.LexicalGraphAgainstDenseStore
    PermissionTickTelemetryHour = _robotics_demonstrations.PermissionTickTelemetryHour
    ImitationBoundsFiveSecondEpisode = _robotics_demonstrations.ImitationBoundsFiveSecondEpisode
    LatchPairingLag = _robotics_demonstrations.LatchPairingLag
    LatchPairingLagTightPlate = _robotics_demonstrations.LatchPairingLagTightPlate
    TeleopCorrectionLoop = _robotics_demonstrations.TeleopCorrectionLoop
    WarehouseTeleopIngestion = _robotics_demonstrations.WarehouseTeleopIngestion
    LatchDemonstrationCampaign = _robotics_demonstrations.LatchDemonstrationCampaign
    WarehouseManipulatorStateGrid = _robotics_demonstrations.WarehouseManipulatorStateGrid
    ContactAngleCoverage = _robotics_demonstrations.ContactAngleCoverage
    LatchScenarioTable = _robotics_demonstrations.LatchScenarioTable
    LatchScenarioTableAfterRecovery = _robotics_demonstrations.LatchScenarioTableAfterRecovery
    GripperStrokeTransfer = _robotics_demonstrations.GripperStrokeTransfer
    Llama3_70BAgentContextAdmissionDgxH100 = _robotics_demonstrations.Llama3_70BAgentContextAdmissionDgxH100
    Llama3_70BShortContextAdmissionDgxH100 = _robotics_demonstrations.Llama3_70BShortContextAdmissionDgxH100
    KvBytesPerTokenByAttentionDesign = _robotics_demonstrations.KvBytesPerTokenByAttentionDesign
    CappedRequestsPagedAllocation = _robotics_demonstrations.CappedRequestsPagedAllocation
    ForkedCandidatesPrefixCopy = _robotics_demonstrations.ForkedCandidatesPrefixCopy
    InvariantPrefixRebuildDgxH100 = _robotics_demonstrations.InvariantPrefixRebuildDgxH100
    ChunkedPrefillAmongDecodes = _robotics_demonstrations.ChunkedPrefillAmongDecodes
    RepairToolWaitPausedState = _robotics_demonstrations.RepairToolWaitPausedState
    RepairBlockSharingResidency = _robotics_demonstrations.RepairBlockSharingResidency
    RepairAgentResidency = _robotics_demonstrations.RepairAgentResidency
    SupportAgentResidency = _robotics_demonstrations.SupportAgentResidency
    WarehouseJointLoopThroughTick = _robotics_realtime.WarehouseJointLoopThroughTick
    WarehouseJointLoopNextTickOutput = _robotics_realtime.WarehouseJointLoopNextTickOutput
    WarehouseJointLoopSlowTick = _robotics_realtime.WarehouseJointLoopSlowTick
    WarehousePermissionTickResponse = _robotics_realtime.WarehousePermissionTickResponse
    WarehousePermissionTickLongBlocking = _robotics_realtime.WarehousePermissionTickLongBlocking
    WarehouseServoCanFdOneBus = _robotics_realtime.WarehouseServoCanFdOneBus
    WarehouseServoCanFdArmBus = _robotics_realtime.WarehouseServoCanFdArmBus
    WarehouseServoCanFdBaseBus = _robotics_realtime.WarehouseServoCanFdBaseBus
    WarehouseServoEtherCat = _robotics_realtime.WarehouseServoEtherCat
    WarehouseChunkLease = _robotics_realtime.WarehouseChunkLease
    WarehouseChunkLeaseLengthened = _robotics_realtime.WarehouseChunkLeaseLengthened
    BlindRetryTwoPercentLoss = _agents_tools.BlindRetryTwoPercentLoss
    TwentyFourMegabyteLogObservation = _agents_tools.TwentyFourMegabyteLogObservation
    ModelPollingEveryFiveSeconds = _agents_tools.ModelPollingEveryFiveSeconds
    FiftyToolCatalogSixSelected = _agents_tools.FiftyToolCatalogSixSelected
    ShellAgainstTwelveStructuredTools = _agents_tools.ShellAgainstTwelveStructuredTools
    SyscallTaxPerMillionCalls = _agents_sandboxes.SyscallTaxPerMillionCalls
    ParserCallsEnvelopeSetup = _agents_sandboxes.ParserCallsEnvelopeSetup
    CopyOnWriteRepairWorkspaces = _agents_sandboxes.CopyOnWriteRepairWorkspaces
    CodingSandboxPoolTwoPerSecond = _agents_sandboxes.CodingSandboxPoolTwoPerSecond
    PhoneFullUpdateWorkingSet = _training_federated.PhoneFullUpdateWorkingSet
    PhoneBackgroundFineTuneBattery = _training_federated.PhoneBackgroundFineTuneBattery
    TransformerLayerRank4Update = _training_federated.TransformerLayerRank4Update
    CompactEncoderResidualAdapters = _training_federated.CompactEncoderResidualAdapters
    CameraModelTenContextPersonalization = _training_federated.CameraModelTenContextPersonalization
    PhoneImagesVersusQuantizedUpdates = _training_federated.PhoneImagesVersusQuantizedUpdates
    FederatedAdapterUploadMobileUplink = _training_federated.FederatedAdapterUploadMobileUplink
    LatchOpeningDiscount = _robotics_policy_training.LatchOpeningDiscount
    LatchDecoderBudget = _robotics_policy_training.LatchDecoderBudget
    LeggedGymQuadrupedRun = _robotics_policy_training.LeggedGymQuadrupedRun
    LatchTrialCampaign = _robotics_policy_training.LatchTrialCampaign
    LatchContactDeadlines = _robotics_policy_training.LatchContactDeadlines
    LatchSoftSpringExploit = _robotics_policy_training.LatchSoftSpringExploit
    LinearDriveRealityGap = _robotics_policy_training.LinearDriveRealityGap
    LatchResidualFineTune = _robotics_policy_training.LatchResidualFineTune
    MnistMlpFirstLayer = _training_layers.MnistMlpFirstLayer
    MnistMlpLayerStack = _training_layers.MnistMlpLayerStack
    Dense2048Layer = _training_layers.Dense2048Layer
    Dense1024LayerFp16 = _training_layers.Dense1024LayerFp16
    MnistFirstConvLayer = _training_layers.MnistFirstConvLayer
    MnistSmallCnn = _training_layers.MnistSmallCnn
    ImageNetRgbFilterVersusDense = _training_layers.ImageNetRgbFilterVersusDense
    ImageNet64ChannelConv = _training_layers.ImageNet64ChannelConv
    ResidualAddFp32 = _training_layers.ResidualAddFp32
    ResNet50ConvKernelShare = _training_layers.ResNet50ConvKernelShare
    ResNet50ThirtyFpsStream = _training_layers.ResNet50ThirtyFpsStream
    ElmanRnnStep = _training_layers.ElmanRnnStep
    RecurrentState512Fp32 = _training_layers.RecurrentState512Fp32
    MnistDenseNetwork = _training_neural.MnistDenseNetwork
    MnistDenseNetworkWiderHidden = _training_neural.MnistDenseNetworkWiderHidden
    MnistDenseNetworkDoubledInputSide = _training_neural.MnistDenseNetworkDoubledInputSide
    MnistDenseNetworkLargeBatch = _training_neural.MnistDenseNetworkLargeBatch
    DenseWidthDepthSweep = _training_neural.DenseWidthDepthSweep
    MnistEpochCpuVersusH100 = _training_neural.MnistEpochCpuVersusH100
    DigitRecognitionParadigms = _training_neural.DigitRecognitionParadigms
    ReluVersusSigmoidLogic = _training_neural.ReluVersusSigmoidLogic
    TwentySigmoidLayers = _training_neural.TwentySigmoidLayers
    DigitCrossEntropyExamples = _training_neural.DigitCrossEntropyExamples
    Gpt2Fp32AdamOnH100 = _training_neural.Gpt2Fp32AdamOnH100
    ResNet50Fp32SgdMomentum = _training_neural.ResNet50Fp32SgdMomentum
    OperationalLoadSketch = _economics_operations.OperationalLoadSketch
    RecommendationFeatureHistory = _economics_operations.RecommendationFeatureHistory
    DailyBillionRecommendationLookups = _economics_operations.DailyBillionRecommendationLookups
    AggregationWindowRevenueIncident = _economics_operations.AggregationWindowRevenueIncident
    DelayedAlertClickLoss = _economics_operations.DelayedAlertClickLoss
    CanaryVersusCutoverExposure = _economics_operations.CanaryVersusCutoverExposure
    CanaryGuardrailSchedule = _economics_operations.CanaryGuardrailSchedule
    CtrLiftExperiment = _economics_operations.CtrLiftExperiment
    ThreeSigmaModelFleetAlerts = _economics_operations.ThreeSigmaModelFleetAlerts
    TwoSigmaModelFleetAlerts = _economics_operations.TwoSigmaModelFleetAlerts
    OwnVsRentH100Cluster1K = _economics_operations.OwnVsRentH100Cluster1K
    OwnVsRentDgxH100Node = _economics_operations.OwnVsRentDgxH100Node
    OwnVsRentH100Cluster10K = _economics_operations.OwnVsRentH100Cluster10K
    BurstyTrainingOn1KCluster = _economics_operations.BurstyTrainingOn1KCluster
    StartupOneModelTco = _economics_operations.StartupOneModelTco
    FiftyModelCompanyTco = _economics_operations.FiftyModelCompanyTco
    Int8InferencePayback = _economics_operations.Int8InferencePayback
    TrainingOptimizationPayback = _economics_operations.TrainingOptimizationPayback
    SharedPlatformBenefit50 = _economics_operations.SharedPlatformBenefit50
    SharedPlatformBenefit20 = _economics_operations.SharedPlatformBenefit20
    Gpt2SmallSixTokenQkvProjection = _serving_transformers.Gpt2SmallSixTokenQkvProjection
    Gpt2XlBlockFullWindow = _serving_transformers.Gpt2XlBlockFullWindow
    TwelveHeadScoresAt100kTokens = _serving_transformers.TwelveHeadScoresAt100kTokens
    BertWindowToFourK = _serving_transformers.BertWindowToFourK
    BertWindowToOneK = _serving_transformers.BertWindowToOneK
    BertWindowToTwoK = _serving_transformers.BertWindowToTwoK
    Gpt2XlKvFullWindow = _serving_transformers.Gpt2XlKvFullWindow
    Llama2_7BKvAt2048Tokens = _serving_transformers.Llama2_7BKvAt2048Tokens
    Gpt2XlContextMemoryOnH100 = _serving_transformers.Gpt2XlContextMemoryOnH100
    Llama2_70B_DecodeTwoH100 = _serving_transformers.Llama2_70B_DecodeTwoH100
    Gpt2XlDecodeH100 = _serving_transformers.Gpt2XlDecodeH100
    CriteoShapedRequestOnH100 = _serving_transformers.CriteoShapedRequestOnH100
    HundredMillionItemTableOnA100 = _serving_transformers.HundredMillionItemTableOnA100
    BillionUserTable = _serving_transformers.BillionUserTable
    Gpt2XlFp16DecodeOnH100 = _optimization_framework.Gpt2XlFp16DecodeOnH100
    Gpt2XlEagerDecodeDispatch = _optimization_framework.Gpt2XlEagerDecodeDispatch
    SmallKernelDispatch = _optimization_framework.SmallKernelDispatch
    LargeKernelDispatch = _optimization_framework.LargeKernelDispatch
    SmallMlpDispatch = _optimization_framework.SmallMlpDispatch
    Gpt3BlockDispatch = _optimization_framework.Gpt3BlockDispatch
    ElementwisePairFusion = _optimization_framework.ElementwisePairFusion
    ResNet50CompileBreakeven = _optimization_framework.ResNet50CompileBreakeven
    Dense100kLayerJacobianFp32 = _optimization_framework.Dense100kLayerJacobianFp32
    ResNet50Fp32AdamBatch64 = _optimization_framework.ResNet50Fp32AdamBatch64
    ResNet50Fp32AdamBatch64HighActivations = _optimization_framework.ResNet50Fp32AdamBatch64HighActivations
    ResNet50Fp32AdamBatch512 = _optimization_framework.ResNet50Fp32AdamBatch512
    Fp32MillionElementTensorAcrossH100Links = _optimization_framework.Fp32MillionElementTensorAcrossH100Links
    ImageNetBatch64HostFeed = _optimization_framework.ImageNetBatch64HostFeed
    MnistTwoLayerTrainingStepOnH100 = _optimization_framework.MnistTwoLayerTrainingStepOnH100
    ResNet50ApiExtractionBudget = _reliability_security.ResNet50ApiExtractionBudget
    ImageNetTop5RoundedScores = _reliability_security.ImageNetTop5RoundedScores
    ReferenceJobAttackSurface = _reliability_security.ReferenceJobAttackSurface
    SecureMultiTenantServing = _reliability_security.SecureMultiTenantServing
    ResNet50Fp32InSgxEnclave = _reliability_security.ResNet50Fp32InSgxEnclave
    ResNet50Int8InSgxEnclave = _reliability_security.ResNet50Int8InSgxEnclave
    ResNet18Fp32InSgxEnclave = _reliability_security.ResNet18Fp32InSgxEnclave
    ResNet18Int8InSgxEnclave = _reliability_security.ResNet18Int8InSgxEnclave
    HealthMonitoringEncryptedInference = _reliability_security.HealthMonitoringEncryptedInference
    SalaryMeanLaplaceRelease = _reliability_security.SalaryMeanLaplaceRelease
    MembershipAuditAtEpsilonOne = _reliability_security.MembershipAuditAtEpsilonOne
    Cifar10DpSgdRun = _reliability_security.Cifar10DpSgdRun
    FeatureMeanShiftTest = _statistics_robustness.FeatureMeanShiftTest
    FraudCountryPsi = _statistics_robustness.FraudCountryPsi
    FraudAmountDrift = _statistics_robustness.FraudAmountDrift
    FraudAmountKs = _statistics_robustness.FraudAmountKs
    PgdSevenTraining = _statistics_robustness.PgdSevenTraining
    SmoothingRadiusCifar10 = _statistics_robustness.SmoothingRadiusCifar10
    Fp32WeightExponentFlip = _statistics_robustness.Fp32WeightExponentFlip
    InjectionExposureByReadVolume = _agents_security.InjectionExposureByReadVolume
    FixtureReleasePolicyComparison = _agents_security.FixtureReleasePolicyComparison
    CleanAttackSuite50 = _agents_security.CleanAttackSuite50
    CleanAttackSuite100 = _agents_security.CleanAttackSuite100
    PrefixCacheProbe2K = _agents_security.PrefixCacheProbe2K
    FreezeRuleOverStepBudget = _agents_harness.FreezeRuleOverStepBudget
    RepairRemediationApprovalEscrow = _agents_harness.RepairRemediationApprovalEscrow
    InteractiveMaintenanceDrr = _agents_harness.InteractiveMaintenanceDrr
    InteractiveMaintenanceDrrCompacted = _agents_harness.InteractiveMaintenanceDrrCompacted
    RepairNextCallAdmission = _agents_harness.RepairNextCallAdmission
    RepairTrajectoryBudgetMeter = _agents_harness.RepairTrajectoryBudgetMeter
    LatchEvaluationCampaign = _statistics_closed_loop.LatchEvaluationCampaign
    LatchSequentialTest = _statistics_closed_loop.LatchSequentialTest
    LatchCheckpointComparison = _statistics_closed_loop.LatchCheckpointComparison
    HazardExposureWall = _statistics_closed_loop.HazardExposureWall
    WarehouseMissionExposure = _statistics_closed_loop.WarehouseMissionExposure
    WarehouseSupervisedMissions = _statistics_closed_loop.WarehouseSupervisedMissions
    AisleHeadingDrift = _statistics_closed_loop.AisleHeadingDrift
    StiffSurfaceResidual = _statistics_closed_loop.StiffSurfaceResidual
    LatchForceTail = _statistics_closed_loop.LatchForceTail
    RoadMilesToDemonstrate = _statistics_closed_loop.RoadMilesToDemonstrate
    AutosteerCrashRateDenominator = _statistics_closed_loop.AutosteerCrashRateDenominator
    WarehouseNavCameraAge = _robotics_perception.WarehouseNavCameraAge
    WarehouseNavCameraRollingShutter = _robotics_perception.WarehouseNavCameraRollingShutter
    WarehouseNavCameraRollingShutterAisleSpeed = _robotics_perception.WarehouseNavCameraRollingShutterAisleSpeed
    WarehouseRackEndTimingAttitude = _robotics_perception.WarehouseRackEndTimingAttitude
    WarehouseLongAisleMountError = _robotics_perception.WarehouseLongAisleMountError
    WarehouseLidarExtrinsicShift = _robotics_perception.WarehouseLidarExtrinsicShift
    WarehouseRackEndLidarAge = _robotics_perception.WarehouseRackEndLidarAge
    WarehouseStereoRigDepthError = _robotics_perception.WarehouseStereoRigDepthError
    VisualServoDelayPhaseMargin = _robotics_perception.VisualServoDelayPhaseMargin
    ClearanceInsetConformalRank = _robotics_perception.ClearanceInsetConformalRank
    WarehouseLidarClockConversion = _robotics_perception.WarehouseLidarClockConversion
    DenseFogVisibility = _robotics_perception.DenseFogVisibility
    AluminumLinkWarming = _robotics_perception.AluminumLinkWarming
    Gpt2XlRunOnOneH100 = _data_selection.Gpt2XlRunOnOneH100
    Gpt2XlBatch32Block = _data_selection.Gpt2XlBatch32Block
    Gpt2XlBatch32Activations = _data_selection.Gpt2XlBatch32Activations
    Gpt2XlBatch32ActivationsFp32 = _data_selection.Gpt2XlBatch32ActivationsFp32
    Gpt2Fp32AdamBatch32OnH100 = _data_selection.Gpt2Fp32AdamBatch32OnH100
    Gpt2MixedPrecisionAdamOnH100 = _data_selection.Gpt2MixedPrecisionAdamOnH100
    Llama2_7BMixedPrecisionAdam = _data_selection.Llama2_7BMixedPrecisionAdam
    Llama3_8BBatch4Activations = _data_selection.Llama3_8BBatch4Activations
    Llama3_8BFullFineTuneOnH100 = _data_selection.Llama3_8BFullFineTuneOnH100
    Llama3_8BLoraOnH100 = _data_selection.Llama3_8BLoraOnH100
    ResNet50H100InputFeed = _data_selection.ResNet50H100InputFeed
    GilThreadedLoaderStall = _data_selection.GilThreadedLoaderStall
    Gpt2XlRunOnOneDgxH100 = _data_selection.Gpt2XlRunOnOneDgxH100
    LoanScreeningTwoGroups = _statistics_responsibility.LoanScreeningTwoGroups
    FraudFlagTwoGroups = _statistics_responsibility.FraudFlagTwoGroups
    LoanApprovalGapTest = _statistics_responsibility.LoanApprovalGapTest
    CreditParityAccuracyCost = _statistics_responsibility.CreditParityAccuracyCost
    FairnessMonitor10K = _statistics_responsibility.FairnessMonitor10K
    InlineShapRetrofit = _statistics_responsibility.InlineShapRetrofit
    ImageExplanationLog = _statistics_responsibility.ImageExplanationLog
    Gpt3ShardedUnlearning = _statistics_responsibility.Gpt3ShardedUnlearning
    RewardModelLabelBudget = _statistics_responsibility.RewardModelLabelBudget
    RadiologyAutomationBias = _statistics_responsibility.RadiologyAutomationBias
    Gpt3PrivacyPlan = _statistics_responsibility.Gpt3PrivacyPlan
    CreditServiceTwoGroups = _statistics_fairness.CreditServiceTwoGroups
    CreditServiceEqualOpportunityPrice = _statistics_fairness.CreditServiceEqualOpportunityPrice
    CreditServiceBiasFeedback = _statistics_fairness.CreditServiceBiasFeedback
    OnePercentGroupEvaluationBudget = _statistics_fairness.OnePercentGroupEvaluationBudget
    FairnessAccuracyFrontierSketch = _statistics_fairness.FairnessAccuracyFrontierSketch
    RecommenderExplorationExposure = _statistics_fairness.RecommenderExplorationExposure
    IntentQuorumGroupCommit = _reliability_durability.IntentQuorumGroupCommit
    GrowingTrajectorySnapshots = _reliability_durability.GrowingTrajectorySnapshots
    TwoCrashCheckpointTrace = _reliability_durability.TwoCrashCheckpointTrace
    ReexecutedWorkCheckpointInterval = _reliability_durability.ReexecutedWorkCheckpointInterval
    ResumedContextColdPrefill = _reliability_durability.ResumedContextColdPrefill
    GreedyRerunReproduction = _reliability_durability.GreedyRerunReproduction
    RepairTrajectoryRetention = _reliability_durability.RepairTrajectoryRetention
    TestConfigRepairVsRebuild = _reliability_recovery.TestConfigRepairVsRebuild
    RemediationCanaryPivotLoss = _reliability_recovery.RemediationCanaryPivotLoss
    RepairLoopToTurnCeiling = _reliability_recovery.RepairLoopToTurnCeiling
    ThreeLayerRetryAmplification = _reliability_recovery.ThreeLayerRetryAmplification
    BlockingCallBulkhead = _reliability_recovery.BlockingCallBulkhead
    RetryCohortJitter = _reliability_recovery.RetryCohortJitter
    RecurringTaskConsistency = _statistics_evaluation.RecurringTaskConsistency
    RecurringTaskConsistencyLowerRate = _statistics_evaluation.RecurringTaskConsistencyLowerRate
    RecurringTaskConsistencyHalvedFailure = _statistics_evaluation.RecurringTaskConsistencyHalvedFailure
    RefundAnswerFiveSessions = _statistics_evaluation.RefundAnswerFiveSessions
    IndependentJudgesNinetyPercent = _statistics_evaluation.IndependentJudgesNinetyPercent
    IndependentJudgesEightyPercent = _statistics_evaluation.IndependentJudgesEightyPercent
    FiftyTaskIntervals = _statistics_evaluation.FiftyTaskIntervals
    AcceptanceRatePlanning = _statistics_evaluation.AcceptanceRatePlanning
    RepairCandidatePairedComparison = _statistics_evaluation.RepairCandidatePairedComparison
    TwoTaskTypeConsistency = _statistics_evaluation.TwoTaskTypeConsistency
    CappedCandidateSelection = _statistics_evaluation.CappedCandidateSelection
    WarehouseRackEndRangeFusion = _robotics_spatial_memory.WarehouseRackEndRangeFusion
    RackEndMisdatedUpdateConsistency = _robotics_spatial_memory.RackEndMisdatedUpdateConsistency
    WarehouseObservationReplayBuffer = _robotics_spatial_memory.WarehouseObservationReplayBuffer
    WarehouseImuOneSecondCoast = _robotics_spatial_memory.WarehouseImuOneSecondCoast
    WarehouseUmbmarkOdometryDrift = _robotics_spatial_memory.WarehouseUmbmarkOdometryDrift
    LatchSlotFingerBelief = _robotics_spatial_memory.LatchSlotFingerBelief
    LatchSlotFingerBeliefTightSlot = _robotics_spatial_memory.LatchSlotFingerBeliefTightSlot
    LatchSlotStaleWristTransform = _robotics_spatial_memory.LatchSlotStaleWristTransform
    ConveyorMugHandleBelief = _robotics_spatial_memory.ConveyorMugHandleBelief
    CoworkerBesideArmBelief = _robotics_spatial_memory.CoworkerBesideArmBelief
    CoworkerBesideArmBeliefTightTolerance = _robotics_spatial_memory.CoworkerBesideArmBeliefTightTolerance
    WarehouseRackEndFreeSpace = _robotics_spatial_memory.WarehouseRackEndFreeSpace
    ArmSensorEpochGap = _robotics_spatial_memory.ArmSensorEpochGap
    WarehouseSkewedRecordPair = _robotics_spatial_memory.WarehouseSkewedRecordPair
    WristForceBiasDrift = _robotics_spatial_memory.WristForceBiasDrift
    ConveyorMugFrozenTransform = _robotics_spatial_memory.ConveyorMugFrozenTransform
    WarehousePermissionClearanceQuery = _robotics_spatial_memory.WarehousePermissionClearanceQuery
    WorkcellHostMapSizing = _robotics_spatial_memory.WorkcellHostMapSizing
    WarehouseDepthRaymarchTraffic = _robotics_spatial_memory.WarehouseDepthRaymarchTraffic
    WarehouseFoundationEncoderAtCameraRate = _robotics_spatial_memory.WarehouseFoundationEncoderAtCameraRate
    ConveyorMugIntentCadence = _robotics_grounding.ConveyorMugIntentCadence
    ConveyorMugSettlingDrift = _robotics_grounding.ConveyorMugSettlingDrift
    ConveyorMugSlowBelt = _robotics_grounding.ConveyorMugSlowBelt
    ConveyorMugDynamicReach = _robotics_grounding.ConveyorMugDynamicReach
    AisleCoworkerForwardOccupancy = _robotics_grounding.AisleCoworkerForwardOccupancy
    ConveyorMugIntentLeaseExpiry = _robotics_grounding.ConveyorMugIntentLeaseExpiry
    MugJarReferentSet = _robotics_grounding.MugJarReferentSet
    MugJarReferentSetLowJar = _robotics_grounding.MugJarReferentSetLowJar
    DistantHandleConfidenceRegion = _robotics_grounding.DistantHandleConfidenceRegion
    LatchUnbrakedStall = _robotics_grounding.LatchUnbrakedStall
    ConveyorMugGoalSetOffsets = _robotics_grounding.ConveyorMugGoalSetOffsets
    Gpt2XlWebTextCorpus = _robotics_grounding.Gpt2XlWebTextCorpus
    ChinchillaComputeDoubling = _robotics_grounding.ChinchillaComputeDoubling
    Gpt2WebTextNearDuplicates = _robotics_grounding.Gpt2WebTextNearDuplicates
    ResNet50HalfImageNetIcr = _robotics_grounding.ResNet50HalfImageNetIcr
    ResNet50TenPercentCoresetH100 = _robotics_grounding.ResNet50TenPercentCoresetH100
    ResNet50ProxyTrainedCoreset = _robotics_grounding.ResNet50ProxyTrainedCoreset
    ResNet50HalfCoresetEpochH100 = _robotics_grounding.ResNet50HalfCoresetEpochH100
    ResNet50EchoingH100 = _robotics_grounding.ResNet50EchoingH100
    ExpensiveSelectionPitfall = _robotics_grounding.ExpensiveSelectionPitfall
    ProxySelectionPitfall = _robotics_grounding.ProxySelectionPitfall
    RareClassTenPercentCoreset = _robotics_grounding.RareClassTenPercentCoreset
    Cifar10LinearCurriculum = _robotics_grounding.Cifar10LinearCurriculum
    Cifar100Curriculum = _robotics_grounding.Cifar100Curriculum
    ImageNetCurriculum = _robotics_grounding.ImageNetCurriculum
    MentorNetCurriculum = _robotics_grounding.MentorNetCurriculum
    ActiveLearningTenRounds = _robotics_grounding.ActiveLearningTenRounds
    MedicalScanActiveLearning = _robotics_grounding.MedicalScanActiveLearning
    FixMatchCifar10Cost = _robotics_grounding.FixMatchCifar10Cost
    FoundationModelTenTasks = _robotics_grounding.FoundationModelTenTasks
    ImageNetDataCostBreakdown = _robotics_grounding.ImageNetDataCostBreakdown
    MinHashPipelineAmortization = _robotics_grounding.MinHashPipelineAmortization
    TenfoldDataFourPoints = _robotics_grounding.TenfoldDataFourPoints
    Mega100KFacilityPower = _sustainability_sustainability.Mega100KFacilityPower
    Reference25KFacilityPower = _sustainability_sustainability.Reference25KFacilityPower
    Training10KCoolingRollback = _sustainability_sustainability.Training10KCoolingRollback
    PueRetrofit2MW = _sustainability_sustainability.PueRetrofit2MW
    QuebecVersusPolandPlacement = _sustainability_sustainability.QuebecVersusPolandPlacement
    RegionalGridPlacement = _sustainability_sustainability.RegionalGridPlacement
    H100ReplacementPayback = _sustainability_sustainability.H100ReplacementPayback
    AcceleratorLifeExtension = _sustainability_sustainability.AcceleratorLifeExtension
    EdgeFirstImageCascade = _sustainability_sustainability.EdgeFirstImageCascade
    WakeWordHierarchyPower = _sustainability_sustainability.WakeWordHierarchyPower
    BatchOneDecodeWeightEnergy = _sustainability_sustainability.BatchOneDecodeWeightEnergy
    TokenPriceRebound = _sustainability_sustainability.TokenPriceRebound
    CandidateSweep50 = _sustainability_sustainability.CandidateSweep50
    ReferenceItLoadAnnualWater = _sustainability_sustainability.ReferenceItLoadAnnualWater
    CarbonPricedRegionPlacement = _sustainability_sustainability.CarbonPricedRegionPlacement
    KaplanTenPercentLowerLoss = _sustainability_sustainability.KaplanTenPercentLowerLoss
    Lab64A100TrainingFootprint = _sustainability_sustainability.Lab64A100TrainingFootprint
    TenMillionQueriesAgainstLab64Training = _sustainability_sustainability.TenMillionQueriesAgainstLab64Training
    Gpt3ClusterDoubling1K = _optimization_expansion.Gpt3ClusterDoubling1K
    SlowWorker8K = _optimization_expansion.SlowWorker8K
    KaplanComputeDoubling = _optimization_expansion.KaplanComputeDoubling
    DeepSeekV3WeightTraffic = _optimization_expansion.DeepSeekV3WeightTraffic
    WarehouseFinishedStopAtDriveLimit = _robotics_stopping_enforcement.WarehouseFinishedStopAtDriveLimit
    WarehouseFinishedStopAtAisleSpeed = _robotics_stopping_enforcement.WarehouseFinishedStopAtAisleSpeed
    WarehouseFinishedStopWithDispatch = _robotics_stopping_enforcement.WarehouseFinishedStopWithDispatch
    WarehouseFinishedStopInspectedFloor = _robotics_stopping_enforcement.WarehouseFinishedStopInspectedFloor
    WarehouseFinishedStopInspectedFloorWithDispatch = (
        _robotics_stopping_enforcement.WarehouseFinishedStopInspectedFloorWithDispatch
    )
    WarehouseFinishedStopWalkingPerson = _robotics_stopping_enforcement.WarehouseFinishedStopWalkingPerson
    WarehouseFinishedStopWalkingPersonWithDispatch = (
        _robotics_stopping_enforcement.WarehouseFinishedStopWalkingPersonWithDispatch
    )
    WarehouseFinishedStopTenMsSlower = _robotics_stopping_enforcement.WarehouseFinishedStopTenMsSlower
    WarehouseHostLeaseStall = _robotics_stopping_enforcement.WarehouseHostLeaseStall
    WarehouseIntegratorUnwind = _robotics_stopping_enforcement.WarehouseIntegratorUnwind
    LoadedArmJointGravityStop = _robotics_stopping_enforcement.LoadedArmJointGravityStop
    TwoJointTorqueProjection = _robotics_stopping_enforcement.TwoJointTorqueProjection
    WarehouseEnforcerCycleBudget = _robotics_stopping_enforcement.WarehouseEnforcerCycleBudget
    WarehouseBaseHoldDrift = _robotics_stopping_enforcement.WarehouseBaseHoldDrift
    WarehouseEnforcerUnloadedTick = _robotics_placement.WarehouseEnforcerUnloadedTick
    WarehouseSharedDieLoadedRead = _robotics_placement.WarehouseSharedDieLoadedRead
    WarehouseSharedDieLoadedReadOneChannel = _robotics_placement.WarehouseSharedDieLoadedReadOneChannel
    WarehouseTickMshrStallShare = _robotics_placement.WarehouseTickMshrStallShare
    WarehouseTickMshrStallShareAtDeadline = _robotics_placement.WarehouseTickMshrStallShareAtDeadline
    WarehouseTickWritebackStallShare = _robotics_placement.WarehouseTickWritebackStallShare
    WarehouseTickTenfoldStallShare = _robotics_placement.WarehouseTickTenfoldStallShare
    PackageRailDroop = _robotics_placement.PackageRailDroop
    WarehouseSharedDieThermalThrottle = _robotics_placement.WarehouseSharedDieThermalThrottle
    WarehouseLoadedCycleCheckpoint = _robotics_placement.WarehouseLoadedCycleCheckpoint
    WarehouseRejectedPlacementTrace = _robotics_placement.WarehouseRejectedPlacementTrace
    RepairTrialSpanTimeline = _agents_observability.RepairTrialSpanTimeline
    RareFailureUnderHeadSampling = _agents_observability.RareFailureUnderHeadSampling
    RecurringFailureUnderHeadSampling = _agents_observability.RecurringFailureUnderHeadSampling
    FailureUnderFivePercentHeadSample = _agents_observability.FailureUnderFivePercentHeadSample
    VerdictAwareTraceRetention = _agents_observability.VerdictAwareTraceRetention
    InterfaceRepairBranches = _agents_observability.InterfaceRepairBranches
    TestLintToolDrift = _agents_observability.TestLintToolDrift
    AcceptanceCanaryMonitor = _agents_observability.AcceptanceCanaryMonitor
    AcceptanceCanaryMonitorDoubled = _agents_observability.AcceptanceCanaryMonitorDoubled
    Gpt2XlRunOnOneH100Fp32 = _training_working_set.Gpt2XlRunOnOneH100Fp32
    Gpt2XlRunOnOneH100Tf32 = _training_working_set.Gpt2XlRunOnOneH100Tf32
    Gpt2XlMicroBatch16Activations = _training_working_set.Gpt2XlMicroBatch16Activations
    Gpt2XlBatch32SegmentCheckpointed = _training_working_set.Gpt2XlBatch32SegmentCheckpointed
    Gpt2MixedPrecisionFlashAttentionOnH100 = _training_working_set.Gpt2MixedPrecisionFlashAttentionOnH100
    Gpt2MixedPrecisionTwoMicroBatchesOnH100 = _training_working_set.Gpt2MixedPrecisionTwoMicroBatchesOnH100
    Gpt2MixedPrecisionSegmentCheckpointedOnH100 = _training_working_set.Gpt2MixedPrecisionSegmentCheckpointedOnH100
    Gpt2XlSingleCoreTokenizerFeed = _training_working_set.Gpt2XlSingleCoreTokenizerFeed
    AttentionScores4096TokensFp32 = _training_working_set.AttentionScores4096TokensFp32
    Llama2_7BTokenOnFlagshipPhone = _optimization_compression.Llama2_7BTokenOnFlagshipPhone
    MobileNetV2Fp32WeightReadPhone = _optimization_compression.MobileNetV2Fp32WeightReadPhone
    MobileNetV2Int8WeightReadPhone = _optimization_compression.MobileNetV2Int8WeightReadPhone
    ResNet50Fp32WeightReadPhone = _optimization_compression.ResNet50Fp32WeightReadPhone
    Gpt2XlFp16WeightReadPhone = _optimization_compression.Gpt2XlFp16WeightReadPhone
    Gpt2XlInt4WeightReadPhone = _optimization_compression.Gpt2XlInt4WeightReadPhone
    MobileNetV2BlurPipelinePhone = _optimization_compression.MobileNetV2BlurPipelinePhone
    Rank128Factorization4096 = _optimization_compression.Rank128Factorization4096
    ResNet50Conv256FusionH100 = _optimization_compression.ResNet50Conv256FusionH100
    WearableMobileNetV2SparseFlash = _optimization_compression.WearableMobileNetV2SparseFlash
    PrunedInt8PaperVsMeasured = _optimization_compression.PrunedInt8PaperVsMeasured
    IrregularSparsityLatencyTrace = _optimization_compression.IrregularSparsityLatencyTrace
    BertBaseMobileCompression = _optimization_compression.BertBaseMobileCompression
    ModelFifthOfRequestLatency = _optimization_compression.ModelFifthOfRequestLatency
    H100OverReferenceCpu = _optimization_acceleration.H100OverReferenceCpu
    SystolicArrayReuseEnergy = _optimization_acceleration.SystolicArrayReuseEnergy
    ResNet50InferenceH100OverServerCpu = _optimization_acceleration.ResNet50InferenceH100OverServerCpu
    Gpt2DecodeH100OverServerCpu = _optimization_acceleration.Gpt2DecodeH100OverServerCpu
    Gpt2DecodeH100OverServerCpuBandwidth = _optimization_acceleration.Gpt2DecodeH100OverServerCpuBandwidth
    ResNet50Batch1H100Int8PeakRate = _optimization_acceleration.ResNet50Batch1H100Int8PeakRate
    ResNet50Batch1H100Fp16PeakRate = _optimization_acceleration.ResNet50Batch1H100Fp16PeakRate
    ResNet50Stage3ConvBatch32 = _optimization_acceleration.ResNet50Stage3ConvBatch32
    ResNet50Stage3ConvBatch32Int8 = _optimization_acceleration.ResNet50Stage3ConvBatch32Int8
    ResNet50Stage3ConvBatch32Int8Weights = _optimization_acceleration.ResNet50Stage3ConvBatch32Int8Weights
    ResNet50Stage2ConvBatch32 = _optimization_acceleration.ResNet50Stage2ConvBatch32
    ResNet50Stage2ConvBatch32Int8 = _optimization_acceleration.ResNet50Stage2ConvBatch32Int8
    ResNet50Stage2ExpansionBatch32 = _optimization_acceleration.ResNet50Stage2ExpansionBatch32
    ResNet50Stage2ExpansionBatch32Int8 = _optimization_acceleration.ResNet50Stage2ExpansionBatch32Int8
    ResNet50ClassifierHeadBatch32 = _optimization_acceleration.ResNet50ClassifierHeadBatch32
    Gpt2XlQkvProjectionPrefill = _optimization_acceleration.Gpt2XlQkvProjectionPrefill
    Gpt2XlLayerNormPrefill = _optimization_acceleration.Gpt2XlLayerNormPrefill
    Gpt2XlSoftmaxPrefill = _optimization_acceleration.Gpt2XlSoftmaxPrefill
    Gpt2XlDecodeH100Roofline = _optimization_acceleration.Gpt2XlDecodeH100Roofline
    Gpt2XlLayerNormH100Roofline = _optimization_acceleration.Gpt2XlLayerNormH100Roofline
    ResNet50ClassifierHeadH100Roofline = _optimization_acceleration.ResNet50ClassifierHeadH100Roofline
    KernelAt200FlopPerByteA100 = _optimization_acceleration.KernelAt200FlopPerByteA100
    KernelAt200FlopPerByteH100 = _optimization_acceleration.KernelAt200FlopPerByteH100
    Dense2048Fp16Batch1 = _optimization_acceleration.Dense2048Fp16Batch1
    Dense2048Fp16Batch32 = _optimization_acceleration.Dense2048Fp16Batch32
    Dense2048Fp16Batch512 = _optimization_acceleration.Dense2048Fp16Batch512
    DenseParametersVsBandwidthSince2014 = _optimization_acceleration.DenseParametersVsBandwidthSince2014
    NvidiaGenerationsV100ToB200 = _optimization_acceleration.NvidiaGenerationsV100ToB200
    Matmul4096Fp16Tile128 = _optimization_mapping.Matmul4096Fp16Tile128
    ResNet50Stage2MapInt8ReadOnce = _optimization_mapping.ResNet50Stage2MapInt8ReadOnce
    WarehouseArmRightAngleElbow = _robotics_trajectories.WarehouseArmRightAngleElbow
    WarehouseArmNearStraightElbow = _robotics_trajectories.WarehouseArmNearStraightElbow
    GenericReplannerMissionExposure = _robotics_trajectories.GenericReplannerMissionExposure
    WarehouseArmChunkReplacement = _robotics_trajectories.WarehouseArmChunkReplacement
    WarehouseArmStallStop = _robotics_trajectories.WarehouseArmStallStop
    WarehouseArmOpenMugStop = _robotics_trajectories.WarehouseArmOpenMugStop
    WarehouseBaseJerkLimitedStop = _robotics_trajectories.WarehouseBaseJerkLimitedStop
    WarehouseArmCommitmentSchedule = _robotics_trajectories.WarehouseArmCommitmentSchedule
    WarehouseArmShoulderSwing = _robotics_trajectories.WarehouseArmShoulderSwing
    PandaClassJointSeamBridge = _robotics_trajectories.PandaClassJointSeamBridge
    PandaClassJointSeamBridgeHalved = _robotics_trajectories.PandaClassJointSeamBridgeHalved
    PandaClassJointSeamBridgeCompressed = _robotics_trajectories.PandaClassJointSeamBridgeCompressed
    WarehouseBaseDynamicWindow = _robotics_trajectories.WarehouseBaseDynamicWindow
    WarehouseArmStaleAnchor = _robotics_trajectories.WarehouseArmStaleAnchor
    RepairAgentFailureTriage = _data_curation.RepairAgentFailureTriage
    EightAttemptRunSplit = _data_curation.EightAttemptRunSplit
    TrajectoryMinHashScreen = _data_curation.TrajectoryMinHashScreen
    DemonstrationAdmissionOrder = _data_curation.DemonstrationAdmissionOrder
    DemonstrationAdmissionOrderSlowAudit = _data_curation.DemonstrationAdmissionOrderSlowAudit
    RecoveryPoolDenominators = _data_curation.RecoveryPoolDenominators
    EightAttemptGroupContrast = _data_curation.EightAttemptGroupContrast
    VerifierTimeoutStagingBurst = _data_curation.VerifierTimeoutStagingBurst
    CurationCohortDispositions = _data_curation.CurationCohortDispositions
    RepairTrajectoryPerCallExamples = _training_fine_tuning.RepairTrajectoryPerCallExamples
    RecoveryRecordTargetMask = _training_fine_tuning.RecoveryRecordTargetMask
    RecoveryRecordTargetMaskDoubledObservations = _training_fine_tuning.RecoveryRecordTargetMaskDoubledObservations
    RecoveryRecordTargetMaskLongObservations = _training_fine_tuning.RecoveryRecordTargetMaskLongObservations
    RepairTraceWithLookupWeighting = _training_fine_tuning.RepairTraceWithLookupWeighting
    EightExamplePackingBatch = _training_fine_tuning.EightExamplePackingBatch
    RepairPolicyImitationBounds = _training_fine_tuning.RepairPolicyImitationBounds
    RepairPreferencePair = _training_fine_tuning.RepairPreferencePair
    RepairPreferencePairDoubledBeta = _training_fine_tuning.RepairPreferencePairDoubledBeta
    Rank16AdaptersOnHypothetical8B = _training_fine_tuning.Rank16AdaptersOnHypothetical8B
    Rank32AdaptersOnHypothetical8B = _training_fine_tuning.Rank32AdaptersOnHypothetical8B
    Uint8RangeMinusOneToThree = _optimization_quantization.Uint8RangeMinusOneToThree
    Gpt2XlFp16DecodeFullContextPhone = _optimization_quantization.Gpt2XlFp16DecodeFullContextPhone
    Gpt2XlInt4DecodeFullContextPhone = _optimization_quantization.Gpt2XlInt4DecodeFullContextPhone
    Int8VsFp32OperationEnergy45nm = _optimization_quantization.Int8VsFp32OperationEnergy45nm
    ResNet50Int8ProfileH100 = _optimization_quantization.ResNet50Int8ProfileH100
    WarehouseInLoopTakeoverAtAisleSpeed = _robotics_supervision.WarehouseInLoopTakeoverAtAisleSpeed
    WarehouseOutOfLoopTakeoverAtAisleSpeed = _robotics_supervision.WarehouseOutOfLoopTakeoverAtAisleSpeed
    WarehouseInLoopTakeoverAtRequestSpeed = _robotics_supervision.WarehouseInLoopTakeoverAtRequestSpeed
    WarehouseInLoopTakeoverAtRetiredSlowdown = _robotics_supervision.WarehouseInLoopTakeoverAtRetiredSlowdown
    WarehouseInLoopTakeoverAtFastSlowdown = _robotics_supervision.WarehouseInLoopTakeoverAtFastSlowdown
    WarehouseOutOfLoopTakeoverAtCrawl = _robotics_supervision.WarehouseOutOfLoopTakeoverAtCrawl
    WarehouseOutOfLoopTakeoverAtCrawlP99 = _robotics_supervision.WarehouseOutOfLoopTakeoverAtCrawlP99
    WarehouseFixedWindowAtAisleSpeed = _robotics_supervision.WarehouseFixedWindowAtAisleSpeed
    WarehouseFixedWindowAtCrawl = _robotics_supervision.WarehouseFixedWindowAtCrawl
    WarehouseTakeoverCeilingLadder = _robotics_supervision.WarehouseTakeoverCeilingLadder
    WarehouseTakeoverBrakingBlend = _robotics_supervision.WarehouseTakeoverBrakingBlend
    WarehouseRemoteCommandAuthentication = _robotics_supervision.WarehouseRemoteCommandAuthentication
    WarehouseRenewalReferencedStop = _statistics_fault_injection.WarehouseRenewalReferencedStop
    WarehouseRenewalReferencedStopRestricted = _statistics_fault_injection.WarehouseRenewalReferencedStopRestricted
    WarehouseFinishedStopRestricted = _statistics_fault_injection.WarehouseFinishedStopRestricted
    WarehouseFlagReferencedStop = _statistics_fault_injection.WarehouseFlagReferencedStop
    WarehouseRenewalStopLateOnsetTrial = _statistics_fault_injection.WarehouseRenewalStopLateOnsetTrial
    WarehouseRenewalStopNinetyMsOnsetTrial = _statistics_fault_injection.WarehouseRenewalStopNinetyMsOnsetTrial
    WarehousePermissionRailHoldUp = _statistics_fault_injection.WarehousePermissionRailHoldUp
    WarehouseQueueDelaySpareExhaustion = _statistics_fault_injection.WarehouseQueueDelaySpareExhaustion
    WarehouseQueueDelayChargedSpareExhaustion = _statistics_fault_injection.WarehouseQueueDelayChargedSpareExhaustion
    WarehouseDiagnosticCoverageHundredFit = _statistics_fault_injection.WarehouseDiagnosticCoverageHundredFit
    WarehouseDiagnosticCoverageTenFit = _statistics_fault_injection.WarehouseDiagnosticCoverageTenFit
    DatasheetVersusSustainedEdgeChip = _statistics_benchmarks.DatasheetVersusSustainedEdgeChip
    MobileNetV2CameraTrapBoard = _statistics_benchmarks.MobileNetV2CameraTrapBoard
    PreprocessBoundVisionPipeline = _statistics_benchmarks.PreprocessBoundVisionPipeline
    ResNet50H100TrainingCeiling = _statistics_benchmarks.ResNet50H100TrainingCeiling
    KeywordSpottingOnEsp32S3 = _statistics_benchmarks.KeywordSpottingOnEsp32S3
    ModelOnlyMeanVersusProductionP99 = _statistics_benchmarks.ModelOnlyMeanVersusProductionP99
    ImageClassifierEightGpuScaling = _statistics_benchmarks.ImageClassifierEightGpuScaling
    AccuracyWithinOnePointSampleSize = _statistics_measurement.AccuracyWithinOnePointSampleSize
    NinetyFiveAgainstNinetyFourOnSameImages = _statistics_measurement.NinetyFiveAgainstNinetyFourOnSameImages
    WiderBeamSearchTranslation = _statistics_measurement.WiderBeamSearchTranslation
    MobileNetV2Int8CameraTrapGates = _statistics_measurement.MobileNetV2Int8CameraTrapGates
    ResNet50EagerFp16BatchingH100 = _serving_request_scheduling.ResNet50EagerFp16BatchingH100
    ResNet50BatchOneVersusEightCostH100 = _serving_request_scheduling.ResNet50BatchOneVersusEightCostH100
    PhotoLibraryPrecomputeH100 = _serving_request_scheduling.PhotoLibraryPrecomputeH100
    ResNet50RequestPathH100 = _serving_request_scheduling.ResNet50RequestPathH100
    ResNet50RequestPathTensorRtH100 = _serving_request_scheduling.ResNet50RequestPathTensorRtH100
    DlrmRequestPath = _serving_request_scheduling.DlrmRequestPath
    ImageServiceLittlesLaw = _serving_request_scheduling.ImageServiceLittlesLaw
    ResNet50FleetPlanH100 = _serving_request_scheduling.ResNet50FleetPlanH100
    ResNet50FasterModelAt80PctH100 = _serving_request_scheduling.ResNet50FasterModelAt80PctH100
    BatchWindow10ms = _serving_request_scheduling.BatchWindow10ms
    BatchWindow20msIn50msBudget = _serving_request_scheduling.BatchWindow20msIn50msBudget
    DualTriggerBatchFourRequests = _serving_request_scheduling.DualTriggerBatchFourRequests
    BatchCap16Or32At500Qps = _serving_request_scheduling.BatchCap16Or32At500Qps
    ResNet50ReplicaStartup = _serving_request_scheduling.ResNet50ReplicaStartup
    ModelSwapPcieGen4 = _serving_request_scheduling.ModelSwapPcieGen4
    RecommenderTrafficSurge = _serving_request_scheduling.RecommenderTrafficSurge
    ResNet50BatchEnergyPerImage = _serving_request_scheduling.ResNet50BatchEnergyPerImage
    ResNet50CloudInstanceCost = _serving_request_scheduling.ResNet50CloudInstanceCost
    VerifierFalseAcceptsAtTenPercentPass = _training_reinforcement.VerifierFalseAcceptsAtTenPercentPass
    VerifierFalseAcceptsAtFivePercentPass = _training_reinforcement.VerifierFalseAcceptsAtFivePercentPass
    BinaryRewardFixedBaseline = _training_reinforcement.BinaryRewardFixedBaseline
    DebuggerTurnForkValue = _training_reinforcement.DebuggerTurnForkValue
    ThirtyTurnForksAtEveryBoundary = _training_reinforcement.ThirtyTurnForksAtEveryBoundary
    ThirtyTurnForksAfterTurnsTenAndTwenty = _training_reinforcement.ThirtyTurnForksAfterTurnsTenAndTwenty
    RepairGroupOnePassOfThree = _training_reinforcement.RepairGroupOnePassOfThree
    LonePassInGroupOfSixteen = _training_reinforcement.LonePassInGroupOfSixteen
    LoneFailureInGroupOfSixteen = _training_reinforcement.LoneFailureInGroupOfSixteen
    ShortAndLongFailureLengths = _training_reinforcement.ShortAndLongFailureLengths
    HardTaskRolloutStep = _training_reinforcement.HardTaskRolloutStep
    HardTaskRolloutStepDoubledGroup = _training_reinforcement.HardTaskRolloutStepDoubledGroup
    TargetedTaskRolloutStep = _training_reinforcement.TargetedTaskRolloutStep
    ModerateTaskRolloutStep = _training_reinforcement.ModerateTaskRolloutStep
    PairGroupsAtTenPercentPass = _training_reinforcement.PairGroupsAtTenPercentPass
    FlakyRaceTestAllPassRule = _training_reinforcement.FlakyRaceTestAllPassRule
    DominantTokenRatioClip = _training_reinforcement.DominantTokenRatioClip
    RareTokenRatioClip = _training_reinforcement.RareTokenRatioClip
    RepairRolloutSlotResets = _training_reinforcement.RepairRolloutSlotResets
    RolloutGroupOfFourBarrier = _training_reinforcement.RolloutGroupOfFourBarrier
    RepairVerifierPool = _training_reinforcement.RepairVerifierPool
    WarehouseNavCameraAgeAtAisleSpeed = _statistics_release.WarehouseNavCameraAgeAtAisleSpeed
    WarehouseNavCameraAgeDimAisle = _statistics_release.WarehouseNavCameraAgeDimAisle
    SafetyArgumentConfidence = _statistics_release.SafetyArgumentConfidence
    RepairPartitionScreen = _agents_coordination.RepairPartitionScreen
    RepairPartitionScreenHalvedOverhead = _agents_coordination.RepairPartitionScreenHalvedOverhead
    RepairPartitionScalability = _agents_coordination.RepairPartitionScalability
    ResearchFanoutScalability = _agents_coordination.ResearchFanoutScalability
    ResearchSystemTokenBreakEven = _agents_coordination.ResearchSystemTokenBreakEven
    InterfaceMigrationSchedule = _agents_coordination.InterfaceMigrationSchedule
    SixChildReferencePassing = _agents_coordination.SixChildReferencePassing
    ThousandFileSharedWrites = _agents_coordination.ThousandFileSharedWrites
    FiveCorrelatedReviewers = _agents_coordination.FiveCorrelatedReviewers
    EightWorkerDeadlineFanout = _agents_coordination.EightWorkerDeadlineFanout
    DiffReviewCreditWindow = _agents_coordination.DiffReviewCreditWindow
    MatchedDelegationCohort = _agents_coordination.MatchedDelegationCohort
    AdmittedCommandDecisionChain = _robotics_oversight.AdmittedCommandDecisionChain
    WarehouseRackEndOversightTime = _robotics_oversight.WarehouseRackEndOversightTime
    WarehouseRemoteAssistantQueue = _robotics_oversight.WarehouseRemoteAssistantQueue
    WarehouseRemoteAssistantQueueTenMachines = _robotics_oversight.WarehouseRemoteAssistantQueueTenMachines
    WarehouseRemoteAssistantQueueThirtyTwoMachines = _robotics_oversight.WarehouseRemoteAssistantQueueThirtyTwoMachines
    WarehouseRemoteAssistantLabor = _robotics_oversight.WarehouseRemoteAssistantLabor
    WarehouseSupervisedStageExit = _robotics_oversight.WarehouseSupervisedStageExit
    WarehouseGroupDetectionShortfall = _robotics_oversight.WarehouseGroupDetectionShortfall
    PedestrianGroupEvaluationSample = _robotics_oversight.PedestrianGroupEvaluationSample
    WarehouseCrossingExposure = _robotics_oversight.WarehouseCrossingExposure
    WarehouseEventTriggeredRetention = _robotics_oversight.WarehouseEventTriggeredRetention
    CruiseDisclosurePenalties = _robotics_oversight.CruiseDisclosurePenalties
    RankingServiceAccuracyDrift = _reliability_release.RankingServiceAccuracyDrift
    DlrmReleaseFootprintH100 = _reliability_release.DlrmReleaseFootprintH100
    ResNet50Fp16ReleaseFootprintH100 = _reliability_release.ResNet50Fp16ReleaseFootprintH100
    KeywordSpotterDualBankEsp32S3 = _reliability_release.KeywordSpotterDualBankEsp32S3
    RetrainingPipelineAutomation = _reliability_release.RetrainingPipelineAutomation
    RankingSkewOnePercent = _reliability_release.RankingSkewOnePercent
    ConversionRelativeLiftExperiment = _reliability_release.ConversionRelativeLiftExperiment
    Gpt2XlChatDecodeBatchesH100 = _serving_token_streams.Gpt2XlChatDecodeBatchesH100
    Gpt2XlSpeculativeGpt2SmallH100 = _serving_token_streams.Gpt2XlSpeculativeGpt2SmallH100
    Llama3_8BFp16ConversationsH100 = _serving_token_streams.Llama3_8BFp16ConversationsH100
    Llama3_70BInt4KvBudgetH100 = _serving_token_streams.Llama3_70BInt4KvBudgetH100
    Llama3_8BInt4ChatServiceH100 = _serving_token_streams.Llama3_8BInt4ChatServiceH100
    H100ChatEnergyPerToken = _serving_token_streams.H100ChatEnergyPerToken
    FrontierModelRepairCohort = _economics_cost_accounting.FrontierModelRepairCohort
    LowTokenPriceRepairCohort = _economics_cost_accounting.LowTokenPriceRepairCohort
    RepairCohortTokenPriceComparison = _economics_cost_accounting.RepairCohortTokenPriceComparison
    RepairResendCacheBill = _economics_cost_accounting.RepairResendCacheBill
    RepairCompactionPayback = _economics_cost_accounting.RepairCompactionPayback
    RepairCompactionPaybackHalfCachedRate = _economics_cost_accounting.RepairCompactionPaybackHalfCachedRate
    CheapRetryAgainstStrongCall = _economics_cost_accounting.CheapRetryAgainstStrongCall
    SerialRepairEpisode = _economics_cost_accounting.SerialRepairEpisode
    SerialRepairEpisodeSpeculation = _economics_cost_accounting.SerialRepairEpisodeSpeculation
    CodeEditThreeTierCascade = _economics_cost_accounting.CodeEditThreeTierCascade
    RepairTrajectoryQuotaAdmission = _economics_cost_accounting.RepairTrajectoryQuotaAdmission
    BurstyModelServerQueuePrice = _economics_cost_accounting.BurstyModelServerQueuePrice
    ThirtyCallTailExposure = _economics_cost_accounting.ThirtyCallTailExposure
    DedicatedCapacityBreakEven = _economics_cost_accounting.DedicatedCapacityBreakEven
    RepairBudgetLateBill = _economics_cost_accounting.RepairBudgetLateBill
    DependencyUpgradeArchitectures = _economics_cost_accounting.DependencyUpgradeArchitectures
    CodeReviewAdaptationPayback = _economics_cost_accounting.CodeReviewAdaptationPayback
    LighthouseWorkloadLevers = _economics_cost_accounting.LighthouseWorkloadLevers
    WarehouseFinishedStopOilFilm = _robotics_premises.WarehouseFinishedStopOilFilm
    WarehouseFilmDetectorRange = _robotics_premises.WarehouseFilmDetectorRange
    WarehouseFilmDetectorClosure = _robotics_premises.WarehouseFilmDetectorClosure
    WarehouseRestrictedTransit = _robotics_premises.WarehouseRestrictedTransit
    PackagingCellTransit = _robotics_premises.PackagingCellTransit
    PackagingCellRestrictedMode = _robotics_premises.PackagingCellRestrictedMode
    PackagingCellHalfProcessing = _robotics_premises.PackagingCellHalfProcessing
    PackagingCellHalfBrake = _robotics_premises.PackagingCellHalfBrake
    PackagingCellQualifiedBrake = _robotics_premises.PackagingCellQualifiedBrake
    ShieldCoverageBill = _robotics_premises.ShieldCoverageBill
    WarehouseLatchSeriesSpring = _robotics_premises.WarehouseLatchSeriesSpring
    RemediationIncidentAttempt = _agents_reliability.RemediationIncidentAttempt
    BlindRetryAgainstStatusLookup = _agents_reliability.BlindRetryAgainstStatusLookup
    ResidualViolationsFiftyTurns = _agents_reliability.ResidualViolationsFiftyTurns
    ThirtyPercentValidPassingSet = _agents_reliability.ThirtyPercentValidPassingSet
    HalvedHealthWindow = _agents_reliability.HalvedHealthWindow
    CheckedWindowBlameSearch = _agents_reliability.CheckedWindowBlameSearch
    Conv3x3At256Channels = _training_synthesis.Conv3x3At256Channels
    ChatEndpointTtftMeanVersusP99 = _statistics_synthesis.ChatEndpointTtftMeanVersusP99
    HostPreprocessingRequestPath = _optimization_synthesis.HostPreprocessingRequestPath
    LabeledOutcomeDriftConfirmation = _statistics_monitoring.LabeledOutcomeDriftConfirmation
    RecommenderDetectionDelayLoss = _statistics_monitoring.RecommenderDetectionDelayLoss
    RankingRetrainingInterval = _statistics_monitoring.RankingRetrainingInterval
    RankingRetrainingIntervalFourfoldTraffic = _statistics_monitoring.RankingRetrainingIntervalFourfoldTraffic
    RankingUserAgePsi = _statistics_monitoring.RankingUserAgePsi
    OneModelDailyMetricTests = _statistics_monitoring.OneModelDailyMetricTests
    RecommendationAccuracyByDevice = _statistics_monitoring.RecommendationAccuracyByDevice
    RankingServiceLifecycleCost = _economics_duties.RankingServiceLifecycleCost
    Gpt3TrainingCarbon = _economics_duties.Gpt3TrainingCarbon
    CreditDenialExplanationLoad = _economics_duties.CreditDenialExplanationLoad
    CreditAppealReview = _economics_duties.CreditAppealReview
    DpSgdPerExampleGradients = _economics_duties.DpSgdPerExampleGradients
