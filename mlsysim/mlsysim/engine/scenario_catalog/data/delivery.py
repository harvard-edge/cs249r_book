"""Named data scenarios for delivery.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....systems.registry import Fabrics as _Fabrics
from ....systems.registry import Storage as _Storage
from ....datasets.registry import Datasets as _Datasets
from ....core.units import BYTES_FP32 as _BYTES_FP32
from ....infrastructure.registry import Infrastructure as _Infrastructure
from ....ops import Ops as _Ops
from ... import scenario_types as _st
from ..data.labeling import ImageNetLabelingVersusResNetRun


PetabyteCrossRegionMove = _st.TransferScenario(
    name="One petabyte moved between cloud regions",
    description="A training corpus in US East and accelerators in US West, over a dedicated 100 Gb/s or a "
    "10 Gb/s link, priced at an inter-region egress rate and set against a 20-hour training run.",
    payload=Q_(1, "PB"),  # scenario assumption: the corpus
    links={"100 Gb/s": _Fabrics.Ethernet_100G.bandwidth, "10 Gb/s": _Fabrics.Ethernet_10G.bandwidth},
    egress_rate=Q_(0.02, "USD/GB"),  # scenario assumption: AWS inter-region transfer rate (2024)
    compute_price=_Infrastructure.Pricing.Cloud.TpuV4PerHour.rate,
    job_time=Q_(20, "hour"),  # scenario assumption: the training run the data would feed
)


ResNet50A100InputFeed = _st.AcceleratorInputFeedScenario(
    name="ResNet-50 training on one A100, fed JPEGs from a cloud block volume",
    description="The A100's arithmetic ceiling for ResNet-50 training, the well-tuned sustained rate, 150 KB "
    "JPEGs that decode to 224x224x3 FP32 tensors, a 250 MB/s attached volume, and host loader "
    "workers at 375 images a second each.",
    workload=_Models.Vision.ResNet50,
    hardware=_Hardware.Cloud.A100,
    sustained_rate=ImageNetLabelingVersusResNetRun.throughput_per_accelerator,
    encoded_sample=Q_(150, "KB"),  # scenario assumption: average ImageNet-class JPEG as stored
    decoded_sample=(
        _Datasets.ImageNet.image_width
        * _Datasets.ImageNet.image_height
        * _Datasets.ImageNet.image_channels
        * _BYTES_FP32
    ),
    supply=_Storage.CloudBlockStorageBaseline,
    training_flops_factor=3,  # forward pass plus a backward pass of about twice its cost
    worker_rate=Q_(375, "count/second"),  # scenario assumption: JPEG decode and augment per CPU worker
    host_decode_cap=Q_(3500, "count/second"),  # scenario assumption: illustrative host-wide decode ceiling
)


ResNet50EightA100InputFeed = ResNet50A100InputFeed.variant(
    name="ResNet-50 training on eight A100s in one server, fed from one storage path",
    description="The one-A100 feed scaled to the eight accelerators of one server.",
    accelerators=8,  # one 8-GPU server
)


RecommendationFeatureLookups = _st.FeatureLookupScenario(
    name="A recommendation service reading ten stored features per request",
    description="10,000 requests a second, ten feature reads each, 5 KB feature vectors summarizing 150 KB images.",
    request_rate=Q_(10_000, "1/second"),  # scenario assumption
    reads_per_request=10,  # scenario assumption
    feature_bytes=Q_(5, "KB"),  # scenario assumption: one stored feature vector
    raw_sample_bytes=ResNet50A100InputFeed.encoded_sample,
)


MswcCorpusStorageTiers = _st.CorpusTierScenario(
    name="The Multilingual Spoken Words Corpus on object storage and on local NVMe",
    description="Every clip at 16 kHz and 16 bits, priced per month and read sequentially on each tier.",
    dataset=_Datasets.MSWC,
    storage_rates={
        "object store": _Infrastructure.Pricing.Storage.S3StandardPerTbMonth.rate,
        "NVMe low": _Infrastructure.Pricing.Storage.NvmeLowPerTbMonth.rate,
        "NVMe high": _Infrastructure.Pricing.Storage.NvmeHighPerTbMonth.rate,
    },
    read_bandwidths={
        "object store": _Storage.ObjectStoreSingleStream.bandwidth,
        "NVMe": _Storage.LocalNvmeTrainingTypical.bandwidth,
    },
)


FraudModelColumnProjection = _st.ColumnProjectionScenario(
    name="A fraud model that uses 20 of a table's 100 equal-size columns",
    description="A row format reads every column; a columnar format reads only the 20.",
    columns_used=20,  # scenario assumption
    columns_total=100,  # scenario assumption
)


GzipVersusSnappyEpochs = _st.CodecEpochScenario(
    name="A 100 GB epoch decompressed with gzip or Snappy for 50 epochs",
    description="Host CPU decompression time per epoch and over the run, at each codec's decompression rate.",
    epoch_bytes=Q_(100, "GB"),  # scenario assumption
    epochs=50,  # scenario assumption
    decode_rates={
        "gzip": Q_(120, "MB/s"),  # scenario assumption: typical gzip decompression
        "Snappy": Q_(500, "MB/s"),
    },  # scenario assumption: typical Snappy decompression
)


LoudnessDriftKs1000 = _st.KsDriftScenario(
    name="K-S drift test on clip loudness, 1,000 training against 1,000 recent clips",
    description="A quieter microphone on part of the fleet; the observed largest CDF gap against the "
    "critical value at the 0.05 level.",
    coefficient=_Ops.Monitoring.KsTestCoefficient,
    n_reference=1000,  # scenario assumption
    n_current=1000,  # scenario assumption
    observed_statistic=0.08,  # scenario assumption: observed largest CDF gap
)
