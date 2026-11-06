"""Named data scenarios for labeling.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....datasets.registry import Datasets as _Datasets
from ....literature.registry import Literature as _Literature
from ....infrastructure.registry import Infrastructure as _Infrastructure
from ... import scenario_types as _st
from ....core.units import DAYS_PER_MONTH as _DAYS_PER_MONTH
from ....core.units import VIDEO_1080P_WIDTH as _VIDEO_1080P_WIDTH, VIDEO_1080P_HEIGHT as _VIDEO_1080P_HEIGHT


KeywordSpotterTargets = _st.AlwaysOnDetectorTargetsScenario(
    name="A smart-speaker keyword spotter's product targets",
    description="Catch 98 percent of spoken wake words, wake falsely less than once a month, answer within "
    "200 ms, fit a 256 KB always-on model and 16 KB of preprocessing state, on a $150K data "
    "budget in six months.",
    detection_target=0.98,  # scenario assumption: share of spoken wake words caught
    false_wake_budget=1,  # scenario assumption: expected false wakes per month
    false_wake_horizon=Q_(_DAYS_PER_MONTH, "day"),
    response_limit=Q_(200, "ms"),  # scenario assumption: end of keyword to response
    model_memory=Q_(256, "KB"),  # scenario assumption: always-on island sized for the INT8 DS-CNN
    preprocessing_memory=Q_(16, "KB"),  # scenario assumption: preprocessing-state budget
    data_budget=Q_(150_000, "USD"),  # scenario assumption: total data-engineering budget
    timeline=Q_(6, "month"),  # scenario assumption: time to production
    base_detection_rate=0.90,  # scenario assumption: illustrative base-model detection rate
)


KeywordSpotterSampleRateChoice = _st.DesignTradeoffScenario(
    name="Keyword spotter: 8 kHz to 16 kHz sampling",
    description="Illustrative detection-rate gain; raw-audio storage and processing scale with the rate (kHz).",
    baseline=8,
    alternative=16,  # scenario assumption: sampling rates compared, kHz
    gain_low=0.02,
    gain_high=0.04,  # scenario assumption: illustrative detection-rate gain
)


KeywordSpotterMfccChoice = _st.DesignTradeoffScenario(
    name="Keyword spotter: 13 to 40 MFCC coefficients per frame",
    description="Illustrative detection-rate gain; model input and feature memory scale with the count.",
    baseline=13,
    alternative=40,  # scenario assumption: coefficients per frame compared
    gain_low=0.03,
    gain_high=0.05,  # scenario assumption: illustrative detection-rate gain
)


KeywordSpotterExampleCountChoice = _st.DesignTradeoffScenario(
    name="Keyword spotter: 1M to 10M training examples",
    description="Illustrative detection-rate gain; labeling cost, storage, and training time scale with the count.",
    baseline=1_000_000,
    alternative=10_000_000,  # scenario assumption: training examples compared
    gain_low=0.05,
    gain_high=0.08,  # scenario assumption: illustrative detection-rate gain
)


KeywordSpotterFalseWakes = _st.FalseAlarmBudgetScenario(
    name="One-second windows checked every second, all month, against one false wake a month",
    description="An always-on detector scores one non-keyword window a second around the clock; the "
    "false-wake budget sets the per-window false-positive rate it must hold.",
    window=Q_(1, "second"),  # scenario assumption: one window scored each second
    listening_per_day=Q_(24, "hour"),  # always on
    horizon=KeywordSpotterTargets.false_wake_horizon,
    budget=KeywordSpotterTargets.false_wake_budget,
    rejection_levels=[0.999, 0.9999, 0.99999],  # scenario assumption: per-window rejection rates compared
)


KeywordSpotterFalseWakeEvidence = _st.ZeroEventExposureScenario(
    name="Listening needed to support one false wake a month with none observed",
    description="Representative non-keyword audio, at 95 percent one-sided confidence (the zero-event rule "
    "of three), recorded at 16 kHz and 16 bits.",
    budget=KeywordSpotterFalseWakes.budget,
    horizon=KeywordSpotterFalseWakes.listening_time,
    confidence=0.95,  # scenario assumption: one-sided confidence of the claim
    sample_rate=Q_(16_000, "Hz"),  # scenario assumption: the 16 kHz candidate configuration
    sample_width=Q_(2, "byte"),  # 16-bit mono audio
)


KeywordSpotterDataBudget = _st.DataBudgetScenario(
    name="A $150K keyword-spotting data budget split across labeling, storage, and governance",
    description="Sixty percent buys crowdsourced recordings and labels at $0.10 each plus 20 percent review.",
    total=KeywordSpotterTargets.data_budget,
    shares={
        "labeling": 0.60,  # scenario assumption
        "storage and processing": 0.25,  # scenario assumption
        "governance": 0.15,
    },  # scenario assumption
    label_price=Q_(0.10, "USD"),  # scenario assumption: per recorded and labeled example
    review_overhead=0.20,  # scenario assumption: quality-review surcharge
)


ImageNetLabelingVersusResNetRun = _st.LabelingVersusTrainingScenario(
    name="Labeling one million images against one ResNet-50 run on eight A100s",
    description="A $100K labeling budget against 90 epochs of ImageNet at the A100 ResNet-50 sustained rate, "
    "priced at the reference and cloud GPU-hour rates.",
    labeling_budget=Q_(100_000, "USD"),  # scenario assumption: crowd labels plus review and relabeling
    dataset_examples=_Datasets.ImageNet.training_examples,
    epochs=90,  # scenario assumption: the standard ResNet-50 schedule
    throughput_per_accelerator=Q_(float(_Literature.Benchmarks.ResNet50A100TrainThroughput), "count/second"),
    accelerators=8,  # scenario assumption: one 8-GPU server
    price_low=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
    price_high=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
    labeling_time_low=Q_(2, "week"),  # scenario assumption: crowdsourced labeling of 1M images, low
    labeling_time_high=Q_(4, "week"),  # scenario assumption: and high
)


FullHdMaskVersusBoxes = _st.LabelGranularityScenario(
    name="A 1920x1080 segmentation mask against ten bounding boxes",
    description="One class per pixel against four coordinates per box.",
    image_width=_VIDEO_1080P_WIDTH,
    image_height=_VIDEO_1080P_HEIGHT,
    boxes=10,  # scenario assumption: objects in one street scene
    coords_per_box=4,
)


CityscapesMasksForMillionFrames = _st.LaborEffortScenario(
    name="Pixel masks for one million street frames at the Cityscapes annotation time",
    description="More than 1.5 hours per fine annotation (Cordts et al. 2016), in full-time person-years.",
    items=1_000_000,  # scenario assumption: frames a driving team wants masked
    time_per_item=Q_(1.5, "hour"),  # Cordts et al. (2016), The Cityscapes Dataset for Semantic Urban Sce...
    work_year=Q_(2000, "hour"),  # convention: a full-time person-year of 2,000 working hours
)


MswcByHand = _st.LaborEffortScenario(
    name="The Multilingual Spoken Words Corpus labeled by hand at 10 s per clip",
    description="What forced alignment saved: every clip of the corpus labeled manually.",
    items=_Datasets.MSWC.training_examples.to("count").magnitude,
    time_per_item=Q_(10, "second"),  # scenario assumption: a fast manual label
    work_year=CityscapesMasksForMillionFrames.work_year,
)


BinaryLabelChanceAgreement = _st.AnnotatorAgreementScenario(
    name="Two annotators who each label 90 percent of items 'not spam'",
    description="Chance agreement from the label mix alone, and kappa at 85 percent raw agreement.",
    class_shares=[0.90, 0.10],  # scenario assumption: binary label mix
    observed_agreement=0.85,  # scenario assumption: raw agreement that looks high
)


WakeWordLabelQualityDrift = _st.AgreementDriftScenario(
    name="A wake-word labeling campaign whose agreement falls after new annotators join",
    description="Kappa falls while every input-feature monitor stays flat.",
    baseline_kappa=0.85,  # scenario assumption: initial collection
    later_kappa=0.72,  # scenario assumption: six months later
    elapsed=Q_(6, "month"),  # scenario assumption
)


ActiveLearningTenMillionPool = _st.ActiveLearningBudgetScenario(
    name="Active learning over a 10M-image pool on a $50K budget",
    description="Hosted-model scoring at $0.01 an image and labels at $0.50, from 85 to 95 percent accuracy; "
    "a tenth of the budget scores a narrowed candidate pool.",
    pool=10_000_000,  # scenario assumption: unlabeled images
    label_budget=Q_(50_000, "USD"),  # scenario assumption
    label_price=Q_(0.50, "USD"),  # scenario assumption: a label that needs careful review
    scoring_price=Q_(0.01, "USD"),  # scenario assumption: hosted multimodal model per image
    random_labels_needed=1_000_000,  # scenario assumption: random sampling to the target
    active_labels_low=100_000,  # scenario assumption: hard examples needed, low
    active_labels_high=200_000,  # scenario assumption: hard examples needed, high
    compute_share=0.10,  # scenario assumption: budget share spent on scoring
    baseline_accuracy=0.85,  # scenario assumption
    target_accuracy=0.95,  # scenario assumption
)


MillionEventBatchVersusStreaming = _st.BatchVersusStreamingIngestScenario(
    name="One million 1 KB events a second, streamed or batched hourly",
    description="Always-on streaming cores with a standby each, against a batch job that clears each hour's "
    "data in 10 minutes, at one core-hour price.",
    event_rate=Q_(1_000_000, "1/second"),  # scenario assumption
    event_size=Q_(1, "KB"),  # scenario assumption
    stream_cores=100,  # scenario assumption: 50 primary cores plus a standby each
    stream_hours_per_day=Q_(24, "hour"),
    batch_cores=200,  # scenario assumption
    batch_window=Q_(10, "minute"),  # scenario assumption: per clock hour
    core_hour_price=Q_(0.05, "USD/hour"),  # scenario assumption: per core-hour
    redundancy_factor=2,  # scenario assumption: one standby core per primary
)
