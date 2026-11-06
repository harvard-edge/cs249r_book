"""Named reliability scenarios for security.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....reference_stats.registry import ReferenceStats as _ReferenceStats
from ....datasets.registry import Datasets as _Datasets
from ....core.units import BYTES_FP16 as _BYTES_FP16, param as _param
from ....core.units import BYTES_FP32 as _BYTES_FP32, BYTES_INT8 as _BYTES_INT8
from ... import scenario_types as _st
from ....core.units import DAYS_PER_MONTH as _DAYS_PER_MONTH
from ..base.applications import DenseLLMPretrainingReference


# --- Security and privacy of a fleet (added 2026-10-20) -----------------
ResNet50ApiExtractionBudget = _st.QueryExtractionBudgetScenario(
    name="Copying a ResNet-50 image API to 90 percent fidelity under three billing regimes",
    description="5 million answers against a $5K training cost: $0.001 per query, a $50-a-month tier "
    "capped at 10,000 queries a day, or a 100,000-query lifetime allowance per principal "
    "with every later query billed per query (constructed values).",
    training_cost=Q_(5000, "USD"),  # scenario assumption
    extraction_queries=5e6,  # scenario assumption: answers for 90 percent fidelity
    target_fidelity=0.90,  # scenario assumption
    price_per_query=Q_(0.001, "USD"),  # scenario assumption
    tier_price=Q_(50, "USD"),  # scenario assumption: per billing period
    tier_daily_quota=1e4,  # scenario assumption
    billing_period=Q_(_DAYS_PER_MONTH, "day"),
    lifetime_allowance=1e5,  # scenario assumption
)


ImageNetTop5RoundedScores = _st.OutputShapingScenario(
    name="ImageNet classifier API returning the top 5 scores rounded to two decimals",
    description="Output contract: top-k truncation of 1,000 class scores and rounding from six decimal places to two.",
    class_count=int(_Datasets.ImageNet.num_classes),
    top_k=5,  # scenario assumption
    from_decimals=6,  # scenario assumption
    to_decimals=2,  # scenario assumption
)


ReferenceJobAttackSurface = _st.DataParallelAttackSurfaceScenario(
    name="175B pretraining on the 1,024-GPU reference cluster with 64 data-parallel replicas",
    description="FP16 gradients exchanged each step, the 14-byte-per-parameter checkpoint, and a "
    "gather of every replica's gradient for a robust aggregate.",
    workload=DenseLLMPretrainingReference.workload,
    fleet=DenseLLMPretrainingReference.system,
    dp_degree=64,  # scenario assumption: two 8-GPU nodes of tensor parallelism per replica
    gradient_bytes_per_param=(
        _ReferenceStats.MixedPrecisionAdam.TrainingStateBytesPerParameter
        - _ReferenceStats.StorageTrainingCorpus.CheckpointBytesPerParameter
    ),
    checkpoint_bytes_per_param=_ReferenceStats.StorageTrainingCorpus.CheckpointBytesPerParameter,
    weight_bytes_per_param=_BYTES_FP16 / _param,
)


SecureMultiTenantServing = _st.ThroughputCostScenario(
    name="A matched serving configuration at 1,000 tokens/s, 850 with tenant isolation",
    description="Assumed throughputs for the same workload and resource budget; the loss splits "
    "into stranded partitions, scrubbing time, and lost cache hits.",
    baseline_rate=Q_(1000, "1/s"),  # scenario assumption: tokens per second
    controlled_rate=Q_(850, "1/s"),  # scenario assumption: tokens per second
)


ResNet50Fp32InSgxEnclave = _st.ProtectedMemoryFitScenario(
    name="ResNet-50 FP32 weights in a 128 MiB SGX enclave page cache",
    description="Weights only, before activations, against an older SGX configuration's EPC.",
    workload=_Models.Vision.ResNet50,
    hardware=_Hardware.Cloud.IntelSGX,
    bytes_per_weight=_BYTES_FP32,
)


ResNet50Int8InSgxEnclave = ResNet50Fp32InSgxEnclave.variant(
    name="ResNet-50 INT8 weights in the same enclave page cache", bytes_per_weight=_BYTES_INT8
)


ResNet18Fp32InSgxEnclave = ResNet50Fp32InSgxEnclave.variant(
    name="ResNet-18 FP32 weights in the same enclave page cache", workload=_Models.Vision.ResNet18
)


ResNet18Int8InSgxEnclave = ResNet50Fp32InSgxEnclave.variant(
    name="ResNet-18 INT8 weights in the same enclave page cache",
    workload=_Models.Vision.ResNet18,
    bytes_per_weight=_BYTES_INT8,
)


HealthMonitoringEncryptedInference = _st.EncryptedInferenceLatencyScenario(
    name="A 20 ms health-monitoring inference, with AES around it or under FHE",
    description="Assumed overheads: AES-NI adds 0.5 ms; FHE runs at the conservative 10,000x slowdown.",
    plaintext_latency=Q_(20, "ms"),  # scenario assumption
    encryption_overhead=Q_(0.5, "ms"),  # scenario assumption: AES-NI accelerated
    fhe_multiplier=10000.0,  # Fully homomorphic encryption compute overhead, 2-6 orders of magnit...
)


SalaryMeanLaplaceRelease = _st.LaplaceMeanReleaseScenario(
    name="Mean salary of 1,000 employees on a public roster at epsilon 1",
    description="Salaries from 0 to $200,000; one Laplace draw on the sum, divided by the public count.",
    value_range=Q_(200000, "USD"),  # scenario assumption
    epsilon=1.0,  # scenario assumption
    group_size=1000,  # scenario assumption
    small_group_size=100,  # scenario assumption
)


MembershipAuditAtEpsilonOne = _st.MembershipAuditScenario(
    name="A membership audit below the pure epsilon = 1 DP ceiling",
    description="Illustrative audit curve; the ceiling follows from the DP definition with delta = 0.",
    epsilon=1.0,  # scenario assumption
    audit_shape=0.6,  # scenario assumption: must stay under the ceiling (needs <= 1 - e^-eps)
)


Cifar10DpSgdRun = _st.DpSgdRunScenario(
    name="DP-SGD on CIFAR-10: batch 4,000, 60 epochs, clip 1, sigma 1.3, target epsilon 3",
    description="Inputs a sampled-Gaussian accountant needs; epsilon itself comes from the accountant.",
    dataset_size=int(_Datasets.CIFAR10.training_examples.to("count").magnitude),
    batch_size=4000,  # scenario assumption
    clip_norm=1.0,  # scenario assumption
    noise_multiplier=1.3,  # scenario assumption
    epochs=60,  # scenario assumption
    target_epsilon=3.0,  # scenario assumption
    delta=1e-5,  # scenario assumption
)
