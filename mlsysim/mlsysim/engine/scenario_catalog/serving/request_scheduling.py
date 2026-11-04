"""Named serving scenarios for request scheduling.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....reference_stats.registry import ReferenceStats as _ReferenceStats
from ....core.units import BYTES_FP32 as _BYTES_FP32
from ....infrastructure.registry import Infrastructure as _Infrastructure
from ... import scenario_types as _st
from ...serving_profile import resnet50_h100_profile as _resnet50_h100_profile


# --- Model serving on one machine (added 2026-10-20) ----------------------
# The reference service is ResNet-50 in eager FP16 on one H100, from the
# shared profile (engine.serving_profile): a fixed cost per pass plus device
# work per image. Host phases, traffic, budgets, and prices are stated.
ResNet50EagerFp16BatchingH100 = _st.BatchingKneeScenario(
    name="ResNet-50 eager FP16 batches on one H100",
    description="Fixed cost per pass (kernel issue, per-kernel floor, one weight read) plus device "
    "work per image, from the shared ResNet-50 H100 profile.",
    fixed_time=_resnet50_h100_profile().eager_fp16(0),
    per_request_time=_resnet50_h100_profile().per_image_fp16,
    batch_sizes=[1, 8, 32, 128],  # scenario assumption: batch sizes compared
)


ResNet50BatchOneVersusEightCostH100 = _st.BatchServingCostScenario(
    name="ResNet-50 on one rented H100 at batch 1 and batch 8",
    description="Cost per million requests when the accelerator serves one image per pass or eight.",
    service=ResNet50EagerFp16BatchingH100,
    price=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
    batch_sizes=[1, 8],  # scenario assumption: the two batch sizes compared
)


PhotoLibraryPrecomputeH100 = _st.PrecomputeVersusOnDemandScenario(
    name="A 10,000-photo library classified on demand or overnight on one H100",
    description="Batch-one eager FP16 per photo against the MLPerf INT8 Offline rate per H100; the "
    "on-demand alternative is a moderation service with a 100 ms deadline.",
    items=10_000,  # scenario assumption: photos in one user's library
    on_demand_time=_resnet50_h100_profile().eager_fp16(1),
    offline_rate=Q_(_resnet50_h100_profile().int8_offline_img_s, "1/second"),
    latency_budget=Q_(100, "ms"),  # scenario assumption: moderation deadline
)


ResNet50RequestPathH100 = _st.PipelineSpeedupScenario(
    name="One ResNet-50 image request on an idle H100 server",
    description="Host decode, resize, and normalize; copy over PCIe; the eager FP16 forward pass; "
    "top-5 on the host. The comparison makes the model 10x faster.",
    stages={
        "JPEG decode": Q_(3.0, "ms"),  # scenario assumption
        "resize": Q_(1.0, "ms"),  # scenario assumption: to 224 x 224
        "normalize": Q_(0.5, "ms"),  # scenario assumption
        "copy to GPU": Q_(0.5, "ms"),  # scenario assumption
        "forward pass": _resnet50_h100_profile().eager_fp16(1),
        "top-5": Q_(0.1, "ms"),  # scenario assumption: softmax and top five
    },
    accelerated="forward pass",
    speedup=10,  # scenario assumption: a 10x faster model
)


ResNet50RequestPathTensorRtH100 = ResNet50RequestPathH100.variant(
    name="The same request with a TensorRT FP16 engine",
    description="Only the forward pass changes, to the profile's TensorRT FP16 batch-one time.",
    stages={**ResNet50RequestPathH100.stages, "forward pass": _resnet50_h100_profile().tensorrt("fp16", 1)},
)


DlrmRequestPath = _st.PipelineSpeedupScenario(
    name="One DLRM recommendation request",
    description="Parse the request, fetch 100+ embedding vectors, run the MLP, rank and filter.",
    stages={
        "parse": Q_(0.5, "ms"),  # scenario assumption
        "embedding lookup": Q_(6.0, "ms"),  # scenario assumption
        "MLP forward": Q_(1.5, "ms"),  # scenario assumption
        "rank and filter": Q_(1.0, "ms"),  # scenario assumption
    },
    accelerated="MLP forward",
)


ImageServiceLittlesLaw = _st.LittlesLawScenario(
    name="An image service at 1,000 requests per second, 50 ms mean time inside",
    description="The accelerator runs batches of up to 32; requests beyond those wait in host memory.",
    arrival_rate=Q_(1000, "1/second"),  # scenario assumption
    time_in_system=Q_(50, "ms"),  # scenario assumption: mean time in system
    batch_size=32,  # scenario assumption: largest batch the accelerator runs
)


ResNet50FleetPlanH100 = _st.MM1CapacityPlanScenario(
    name="H100s for 5,000 ResNet-50 requests per second under a 50 ms p99",
    description="Batch-one eager FP16 service; each H100 an M/M/1 server whose p99 is 4.6 times "
    "its mean time in system; 30 percent headroom; re-planned at the batch-32 rate.",
    service_time=_resnet50_h100_profile().eager_fp16(1),
    tail_factor=4.6,  # scenario assumption: ln(100) = 4.605 rounded, as the p99 rule of thumb prints it
    tail_budget=Q_(50, "ms"),  # scenario assumption: queue-plus-GPU p99 budget
    peak_rate=Q_(5000, "1/second"),  # scenario assumption: peak traffic
    headroom=1.3,  # scenario assumption: 30 percent for spikes
    utilization_points=[0.5, 0.7, 0.8, 0.9, 0.95],  # scenario assumption: loads compared
    example_utilization=0.7,  # scenario assumption: the load where mean and p99 are compared
    batched_rate=ResNet50EagerFp16BatchingH100.throughput(32),
    accelerator_price=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
)


ResNet50FasterModelAt80PctH100 = _st.MM1ServiceSpeedupScenario(
    name="The batch-one ResNet-50 service at 80 percent load, before and after graph execution",
    description="Eager FP16 against TorchScript (no per-operation Python dispatch) on the same traffic.",
    slow_service=_resnet50_h100_profile().eager_fp16(1),
    fast_service=_resnet50_h100_profile().torchscript_fp16(1),
    utilization=0.8,  # scenario assumption
)


BatchWindow10ms = _st.BatchWindowScenario(
    name="A 10 ms batching window at 50 to 1,000 requests per second",
    description="Poisson arrivals; the count per window, the share of empty windows, and the batch "
    "a request finds itself in.",
    window=Q_(10, "ms"),  # scenario assumption
    arrival_rates=[Q_(r, "1/second") for r in (50, 200, 500, 1000)],  # scenario assumption
)


BatchWindow20msIn50msBudget = BatchWindow10ms.variant(
    name="A 20 ms batching window inside a 50 ms latency budget",
    description="The mean wait for the window to close, as a share of the budget.",
    window=Q_(20, "ms"),  # scenario assumption
    arrival_rates=[],
    latency_budget=Q_(50, "ms"),  # scenario assumption
)


DualTriggerBatchFourRequests = _st.BatchDispatchScenario(
    name="Four requests under a 12 ms batch timeout behind a busy worker",
    description="Arrivals at 0, 2, 5, and 9 ms never reach the cap; the worker is busy until 18 ms; "
    "4 ms of response path follow; the deadline is 50 ms.",
    arrivals=[Q_(t, "ms") for t in (0, 2, 5, 9)],  # scenario assumption: hypothetical trace
    timeout=Q_(12, "ms"),  # scenario assumption: tuned batching window
    worker_free=Q_(18, "ms"),  # scenario assumption
    response_path=Q_(4, "ms"),  # scenario assumption
    deadline=Q_(50, "ms"),  # scenario assumption
    service=ResNet50EagerFp16BatchingH100,
)


BatchCap16Or32At500Qps = _st.BatchCapChoiceScenario(
    name="Batch cap 16 or 32 at 500 requests per second",
    description="Batches wait until full; saturated throughput from the eager FP16 profile.",
    service=ResNet50EagerFp16BatchingH100,
    arrival_rate=Q_(500, "1/second"),  # scenario assumption
    small_cap=16,  # scenario assumption
    large_cap=32,  # scenario assumption
)


ResNet50ReplicaStartup = _st.ReplicaStartupScenario(
    name="Startup of a ResNet-50 serving replica",
    description="Representative phases: weights from a local SSD or object storage, CUDA context, "
    "TensorRT engine build on a first deploy, warmup, runtime start.",
    phases={
        "weights from local SSD": Q_(0.5, "second"),  # scenario assumption
        "weights from object storage": Q_(4.0, "second"),  # scenario assumption
        "CUDA context": Q_(0.4, "second"),  # scenario assumption
        "TensorRT engine build": Q_(30.0, "second"),  # scenario assumption
        "warmup": Q_(0.2, "second"),  # scenario assumption
        "runtime": Q_(0.4, "second"),  # scenario assumption
    },
    ranges={
        "weights from object storage": [Q_(3, "second"), Q_(5, "second")],  # scenario assumption
        "CUDA context": [Q_(0.3, "second"), Q_(0.5, "second")],  # scenario assumption
        "TensorRT engine build": [Q_(15, "second"), Q_(30, "second")],  # scenario assumption
    },
    warm_path=["weights from local SSD", "CUDA context", "warmup", "runtime"],
    cold_path=["weights from object storage", "CUDA context", "TensorRT engine build", "warmup", "runtime"],
)


ModelSwapPcieGen4 = _st.TransferScenario(
    name="A 10 GB model swapped into an accelerator over PCIe Gen4 x16",
    description="Host memory to device at the link's theoretical bandwidth; bandwidth term only.",
    payload=_ReferenceStats.ModelLoading.PcieSwapReferenceModelSize,
    links={"PCIe Gen4 x16": _Hardware.Cloud.A100.interconnect.bandwidth},
)


RecommenderTrafficSurge = _st.TrafficSurgeScenario(
    name="A recommender sized for 1,000 requests per second receives ten times more",
    description="Utilization pins at 100 percent and the queue grows for as long as the surge lasts.",
    normal_rate=Q_(1000, "1/second"),  # scenario assumption
    surge_rate=Q_(10000, "1/second"),  # scenario assumption
    normal_latency=Q_(50, "ms"),  # scenario assumption
    surge_latency=Q_(10, "second"),  # scenario assumption: latency once the queue has grown
)


ResNet50BatchEnergyPerImage = _st.BatchEnergyPerImageScenario(
    name="ResNet-50 bytes and energy per image at batch 1, 8, and 32",
    description="FP32 weights read once per batch; 12.5 MB of activations written and read per image; "
    "DRAM energy per byte and the mean of an FP32 multiply and add per FLOP (45 nm).",
    workload=_Models.Vision.ResNet50,
    weight_bytes_per_param=_BYTES_FP32,
    activation_bytes=Q_(12.5, "MB"),  # scenario assumption: 224 x 224 activation footprint
    activation_passes=2,  # scenario assumption: written once, read once
    byte_energy=_Hardware.Tech.Memory.DRAM.energy_per_byte,
    flop_energy=(_Hardware.Tech.Op.FlopFp32.energy + _Hardware.Tech.Op.AddFp32.energy) / 2,
)


ResNet50CloudInstanceCost = _st.ServingInstanceCostScenario(
    name="ResNet-50 on a CPU instance, a T4 instance, and an H100",
    description="Hourly prices and saturated image rates; the H100 runs batch 32 of the eager FP16 profile.",
    prices={
        "CPU": _Infrastructure.Pricing.Cloud.CpuInstanceC5XlargePerHour.rate,
        "T4": _Infrastructure.Pricing.Cloud.T4InstanceG4dnXlargePerHour.rate,
        "H100": _Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
    },
    rates={
        "CPU": Q_(50, "1/second"),  # scenario assumption: c5.xlarge ResNet-50 rate
        "T4": Q_(400, "1/second"),  # scenario assumption: T4 ResNet-50 rate
        "H100": ResNet50EagerFp16BatchingH100.throughput(32),
    },
    small="T4",
    large="H100",
)
