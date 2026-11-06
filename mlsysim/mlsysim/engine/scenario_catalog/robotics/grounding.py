"""Named robotics scenarios for grounding.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....systems.registry import Nodes as _Nodes
from ....datasets.registry import Datasets as _Datasets
from ....literature.registry import Literature as _Literature
from ....infrastructure.registry import Infrastructure as _Infrastructure
from ....embodied.registry import Embodied as _Embodied
from ... import scenario_types as _st
from ..data.delivery import ResNet50A100InputFeed
from ..data.selection import Gpt2XlRunOnOneDgxH100
from ..robotics.policy_training import LatchContactDeadlines


# --- Grounded intent: the intent lease and its gates (added 2026-10-20) ---
ConveyorMugIntentCadence = _st.IntentCadenceScenario(
    name="The conveyor mug proposed by the warehouse machine's intent model",
    description="Untracked and tracked evidence horizons against the intent model's inference and "
    "period, the oldest evidence behind a model-only renewal, a 200 ms host stall, and "
    "the stop onset after expiry.",
    site=_Embodied.Scenario.WarehouseAisle,
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    stall=Q_(200.0, "ms"),  # scenario assumption: a stalled proposer holds the mug target
)


ConveyorMugSettlingDrift = ConveyorMugIntentCadence.variant(
    name="The same mug sliding to rest at 20 mm/s",
    description="The untracked evidence horizon of a slow settling drift.",
    target_speed=Q_(20.0, "mm/s"),  # scenario assumption: settling drift
)


ConveyorMugSlowBelt = ConveyorMugIntentCadence.variant(
    name="The same mug on a belt slowed to 50 mm/s",
    description="Whether a model-only renewal fits the untracked horizon of a slower belt.",
    target_speed=Q_(50.0, "mm/s"),  # scenario assumption: slowed belt
)


ConveyorMugDynamicReach = _st.DynamicReachScenario(
    name="The mug handle 300 mm from the gripper at the arm's TCP speed ceiling",
    description="The transit lower bound against the tracked evidence window, the reach inside each "
    "window, and the fresh captures a staged approach needs.",
    cadence=ConveyorMugIntentCadence,
    target_distance=Q_(300.0, "mm"),  # scenario assumption: gripper-to-handle distance at proposal
)


AisleCoworkerForwardOccupancy = _st.ForwardOccupancyScenario(
    name="A coworker's forward occupancy set over the mug's windows and a 2 s forecast",
    description="Growth of the speed-bound set at the ISO 13855 design speed, and the horizon past "
    "which one person's set spans the aisle.",
    cadence=ConveyorMugIntentCadence,
    forecast=Q_(2.0, "s"),  # scenario assumption: a multi-second planning horizon
)


ConveyorMugIntentLeaseExpiry = _st.IntentLeaseExpiryScenario(
    name="The red-mug intent lease: evidence, host silence, and the stopping deadline",
    description="The arm's credible deceleration, its stop from the TCP ceiling, the stopping "
    "deadline, and which term binds with and without a tracker.",
    cadence=ConveyorMugIntentCadence,
    administrative_lease=Q_(100.0, "ms"),  # scenario assumption: administrative lease (six wrist-camera frames)
    clearance=Q_(300.0, "mm"),  # scenario assumption: verified clearance ahead of the gripper
)


MugJarReferentSet = _st.ReferentSetScenario(
    name="A red mug and a red glass jar in the pick window",
    description="Normalized detector scores, their entropy, and the conformal prediction set.",
    labels=["mug", "jar"],
    scores=[0.62, 0.48],  # scenario assumption: raw per-box scores
    calibration_count=200,  # scenario assumption: labeled calibration scenes
    alpha=0.10,  # scenario assumption: allowed miscoverage
    threshold_score=0.60,  # scenario assumption: stands in for a measured calibration quantile
)


MugJarReferentSetLowJar = MugJarReferentSet.variant(
    name="The same pick window with the jar scoring 0.10",
    description="A low jar score leaves a singleton set.",
    scores=[0.62, 0.10],  # scenario assumption: the jar's raw score in the contrast case
)


DistantHandleConfidenceRegion = _st.ConfidenceRegionScenario(
    name="The mug handle seen from across the station by the wrist camera",
    description="The 95 percent region of a single-view estimate against the 15 mm grasp tolerance.",
    site=_Embodied.Scenario.WarehouseAisle,
    sigmas=[Q_(4.0, "mm"), Q_(5.0, "mm"), Q_(12.0, "mm")],  # scenario assumption: lateral x, lateral y, depth
    coverage=0.95,  # scenario assumption: confidence level
)


LatchUnbrakedStall = LatchContactDeadlines.variant(
    name="The guarded latch approach with no local trip, stalled for 200 ms",
    description="Plate travel and spring-model force when only the proposer or the lease ends the press.",
    stall=Q_(200.0, "ms"),  # scenario assumption: unbraked contact interval
)


ConveyorMugGoalSetOffsets = _st.GoalSetMembershipScenario(
    name="Terminal TCP offsets of 10 and 20 mm against the mug's 15 mm goal set",
    description="Normalized ellipsoidal distance of a pure x offset, inside and outside the set.",
    site=_Embodied.Scenario.WarehouseAisle,
    offsets=[Q_(10.0, "mm"), Q_(20.0, "mm")],  # scenario assumption: terminal TCP offsets along x
)


Gpt2XlWebTextCorpus = _st.CorpusRepetitionScenario(
    name="GPT-2 XL's published training compute on one eight-H100 machine, against WebText",
    description="WebText at about 4 bytes per token and the run's 1.5e21 FLOPs at six FLOPs per parameter per token.",
    run=Gpt2XlRunOnOneDgxH100,
    bytes_per_token=Q_(4, "byte/count"),  # scenario assumption: English text averages about 4 bytes per BPE token
    tokens_per_param=float(_Literature.Chinchilla.TokensPerParam),
    flops_per_param_token=float(_Literature.Chinchilla.ComputeConstant),
)


ChinchillaComputeDoubling = _st.AllocationGrowthScenario(
    name="Compute-optimal tokens when the compute budget doubles",
    description="Chinchilla approach 1: D_opt grows as C^0.50.",
    compute_multiple=2.0,  # scenario assumption: double the budget
    exponent=float(_Literature.Chinchilla.Approach1TokensExponent),
)


Gpt2WebTextNearDuplicates = _st.DuplicateRemovalScenario(
    name="Near-duplicate removal on GPT-2's WebText run, at the C4 and RealNews shares",
    description="Lee et al. (2022) near-duplicate shares applied to the eight-H100 run; one read of WebText "
    "from the node's NVMe; 128 32-bit MinHash values per document.",
    run=Gpt2XlWebTextCorpus,
    duplicate_shares={
        "C4": 3.04 / 100,  # Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, and Carlini (20...
        "RealNews": 13.63 / 100,
    },  # Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, and Carlini (20...
    documents=8_000_000,  # scenario assumption: WebText holds slightly over 8 million documents (Radford et al. 2019)
    signature_hashes=128,  # scenario assumption: signature length per document
    hash_bytes=Q_(4, "byte"),  # 32-bit hash values
    read_bandwidth=_Hardware.Cloud.H100.storage.bandwidth,
)


ResNet50HalfImageNetIcr = _st.InformationComputeRatioScenario(
    name="Learning per FLOP of one ResNet-50 epoch on all of ImageNet and on a half-size coreset",
    description="An assumed 5-point gain on all the data against 4.5 points on the half an EL2N score keeps.",
    workload=_Models.Vision.ResNet50,
    dataset=_Datasets.ImageNet,
    gain_full=5.0,  # scenario assumption: percentage-point gain in epoch 1, all data
    gain_subset=4.5,  # scenario assumption: the same with the 50% coreset
    keep_fraction=0.5,  # scenario assumption: keep half
)


ResNet50TenPercentCoresetH100 = _st.CoresetRunScenario(
    name="ResNet-50 on ImageNet for 100 epochs on one H100, against a 10% coreset scored by ResNet-50 or ResNet-18",
    description="The A100 ResNet-50 sustained fraction of peak carried to the H100; one forward pass scores the pool.",
    workload=_Models.Vision.ResNet50,
    proxy=_Models.Vision.ResNet18,
    dataset=_Datasets.ImageNet,
    hardware=_Hardware.Cloud.H100,
    utilization=ResNet50A100InputFeed.sustained_fraction,
    keep_fraction=0.10,  # scenario assumption: 10% coreset
    epochs=100,  # scenario assumption: training epochs
)


ResNet50ProxyTrainedCoreset = ResNet50TenPercentCoresetH100.variant(
    name="A 10% ImageNet coreset for ResNet-50 chosen by a proxy trained from scratch for 5 epochs",
    description="No pretrained checkpoint: the proxy trains briefly, then EL2N scores rank the pool.",
    proxy_epochs=5,  # scenario assumption: proxy training epochs
)


ResNet50HalfCoresetEpochH100 = ResNet50TenPercentCoresetH100.variant(
    name="One ResNet-50 ImageNet epoch on one H100, full and with a half-size coreset, through the data path",
    description="Archive read from the node's NVMe, 8-bit RGB tensors over PCIe, and a host of 14 cores per "
    "accelerator at the loader-worker rate, over the standard 90-epoch schedule.",
    keep_fraction=ResNet50HalfImageNetIcr.keep_fraction,
    epochs=90,  # scenario assumption: standard ImageNet schedule
    host_rate=(_Nodes.DGX_H100.host_cpu_cores // _Nodes.DGX_H100.accelerators_per_node)
    * ResNet50A100InputFeed.worker_rate,
    decoded_sample=Q_(
        _Datasets.ImageNet.image_width * _Datasets.ImageNet.image_height * _Datasets.ImageNet.image_channels, "byte"
    ),  # 8-bit RGB tensor
)


ResNet50EchoingH100 = _st.DataEchoingScenario(
    name="Data echoing for ResNet-50 on one H100 fed by a 14-core host",
    description="The H100's demand, stated to the hundred images a second, against the host's preparation "
    "rate, over 90 epochs, each prepared image used twice.",
    accelerator_rate=Q_(
        round(ResNet50HalfCoresetEpochH100.sample_rate, -2), "count/second"
    ),  # stated as about 10,100 images/s
    host_rate=ResNet50HalfCoresetEpochH100.host_rate,
    examples=_Datasets.ImageNet.training_examples,
    epochs=ResNet50HalfCoresetEpochH100.epochs,
    echo_factor=2,  # scenario assumption: repeat each prepared batch twice
)


ExpensiveSelectionPitfall = _st.SelectionBudgetScenario(
    name="Ten hours of selection for a subset that trains in two, against an eight-hour baseline",
    description="Selection overhead that erases the saving.",
    selection_time=Q_(10, "hour"),  # scenario assumption: expensive selection
    subset_time=Q_(2, "hour"),  # scenario assumption: subset training run
    full_time=Q_(8, "hour"),  # scenario assumption: full training run
)


ProxySelectionPitfall = ExpensiveSelectionPitfall.variant(
    name="Half an hour of proxy scoring against the same eight-hour baseline",
    description="A cheap proxy keeps selection a small share of full training.",
    selection_time=Q_(0.5, "hour"),  # scenario assumption: proxy scoring
)


RareClassTenPercentCoreset = _st.CoverageFloorScenario(
    name="A rare class of 0.1 percent in a 1M-example pool under an unstratified 10% coreset",
    description="Expected rare examples kept against a coverage floor of 150.",
    pool=1_000_000,  # scenario assumption
    rare_share=0.001,  # scenario assumption: 0.1 percent rare
    keep_fraction=0.10,  # scenario assumption: 10% coreset
    floor=150,  # scenario assumption: illustrative coverage floor
)


Cifar10LinearCurriculum = _st.CurriculumScheduleScenario(
    name="A linear-warmup curriculum on CIFAR-10: 115 epochs to target against 150",
    description="Illustrative epoch counts, with an illustrative 10-epoch linear warmup.",
    baseline_epochs=150,  # scenario assumption
    curriculum_epochs=115,  # scenario assumption
    warmup_epochs=10,  # scenario assumption: illustrative warmup W
)


Cifar100Curriculum = Cifar10LinearCurriculum.variant(
    name="An illustrative CIFAR-100 curriculum: 180 epochs against 220",
    description="Illustrative epoch counts.",
    baseline_epochs=220,
    curriculum_epochs=180,
    warmup_epochs=None,
)  # scenario assumption


ImageNetCurriculum = Cifar10LinearCurriculum.variant(
    name="An illustrative ImageNet curriculum: 80 epochs against 90",
    description="Illustrative epoch counts.",
    baseline_epochs=90,
    curriculum_epochs=80,
    warmup_epochs=None,
)  # scenario assumption


MentorNetCurriculum = Cifar10LinearCurriculum.variant(
    name="An illustrative MentorNet-style curriculum: 70 epochs against 90",
    description="Illustrative epoch counts.",
    baseline_epochs=90,
    curriculum_epochs=70,
    warmup_epochs=None,
)  # scenario assumption


ActiveLearningTenRounds = _st.ActiveLearningRoundsScenario(
    name="Active learning from 1,000 labels in ten rounds of 100, against 5,000 random labels",
    description="Labels at $10, $50 of pool scoring per round.",
    label_price=Q_(10, "USD"),  # scenario assumption: per label
    initial_labels=1000,  # scenario assumption
    labels_per_round=100,  # scenario assumption
    rounds=10,  # scenario assumption
    round_scoring_cost=Q_(50, "USD"),  # scenario assumption: scoring the pool once
    random_labels=5000,  # scenario assumption: random sampling to the same quality
    label_turnaround=Q_(14, "day"),  # scenario assumption: annotation turnaround per round
)


MedicalScanActiveLearning = _st.BudgetMatchedLabelingScenario(
    name="A million medical scans at $5 a label under a $500K budget",
    description="Random labeling spends the budget; active learning reaches the same quality with 50,000 labels.",
    pool=Q_(1_000_000, "count"),  # scenario assumption: scans in the pool
    label_price=Q_(5.00, "USD/count"),  # scenario assumption: specialist label
    budget=Q_(500_000, "USD"),  # scenario assumption
    active_labels=Q_(50_000, "count"),  # scenario assumption: labels needed with active learning
    deadline=Q_(1, "month"),  # scenario assumption
)


FixMatchCifar10Cost = _st.SemiSupervisedCostScenario(
    name="Supervised training on 4,000 CIFAR-10 labels against FixMatch on 250 with five times the compute",
    description="Illustrative label and compute dollars.",
    label_price=Q_(1, "USD"),  # scenario assumption: per label
    supervised_labels=4000,  # Sohn, Berthelot, Li, Zhang, Carlini, Cubuk, Kurakin, Zhang, and Raf...
    supervised_compute=Q_(50, "USD"),  # scenario assumption
    ssl_labels=250,  # Sohn, Berthelot, Li, Zhang, Carlini, Cubuk, Kurakin, Zhang, and Raf...
    ssl_compute=Q_(250, "USD"),  # scenario assumption: five times more training
)


FoundationModelTenTasks = _st.PretrainingAmortizationScenario(
    name="Ten tasks trained from scratch against one pretrained foundation model fine-tuned ten times",
    description="GPU-hours and labels per task under illustrative assumptions.",
    pretrain_compute=Q_(10_000, "hour"),  # scenario assumption: one-time GPU-hours
    scratch_compute=Q_(1000, "hour"),  # scenario assumption: GPU-hours per task
    finetune_compute=Q_(50, "hour"),  # scenario assumption: GPU-hours per task
    tasks=10,  # scenario assumption
    scratch_labels=Q_(100_000, "count"),  # scenario assumption
    finetune_labels=Q_(1_000, "count"),  # scenario assumption
    label_price=Q_(1, "USD/count"),  # scenario assumption
)


ImageNetDataCostBreakdown = _st.DataCostBreakdownScenario(
    name="An ImageNet-scale supervised bill: license, crowd labels, a year of storage, and 30 ResNet-50 runs",
    description="Every training image labeled at the crowd rate, the archive kept a year in object storage, "
    "and a 30-run campaign of 100-epoch runs on one H100 at the cloud GPU rate.",
    acquisition=Q_(50_000, "USD"),  # scenario assumption: licensed dataset
    labels=_Datasets.ImageNet.training_examples,
    label_price=_Infrastructure.Pricing.Labeling.CrowdHigh.rate / Q_(1, "count"),
    storage_bytes=_Datasets.ImageNet.training_archive_size.to("GB"),
    storage_rate=_Infrastructure.Pricing.Storage.S3StandardLowPerTbMonth.rate,
    storage_duration=Q_(12, "month"),  # scenario assumption: retention
    run_time=ResNet50TenPercentCoresetH100.run_time,
    runs=30,  # scenario assumption: runs in the training campaign
    accelerators=1,
    accelerator_rate=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
)


MinHashPipelineAmortization = _st.RunAmortizationScenario(
    name="A MinHash deduplication pipeline: $50K to build, $5K to run once, $10K saved per training run",
    description="Illustrative engineering, compute, and per-run savings.",
    build_cost=Q_(50_000, "USD"),  # scenario assumption: engineering time
    operating_cost=Q_(5000, "USD"),  # scenario assumption: one-time MinHash compute
    saving_per_run=Q_(10_000, "USD"),  # scenario assumption
    run_counts=[1, 5, 10, 50],  # campaign sizes the amortization table reads
)


TenfoldDataFourPoints = _st.DataScalingReturnScenario(
    name="Ten times the data for about four points of accuracy",
    description="An illustrative power-law return on more data.",
    base_examples=Q_(1_000_000, "count"),  # scenario assumption
    scaled_examples=Q_(10_000_000, "count"),  # scenario assumption
    accuracy_gain=4.0,  # scenario assumption: percentage points
)
