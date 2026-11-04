"""Named data scenarios for storage.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....systems.registry import Clusters as _Clusters
from ....reference_stats.registry import ReferenceStats as _ReferenceStats
from ....systems.registry import Fabrics as _Fabrics, Nodes as _Nodes
from ....systems.registry import Storage as _Storage
from ....datasets.registry import Datasets as _Datasets
from ....physics.quantities import memory_from_params as _memory_from_params
from ....core.units import BYTES_FP16 as _BYTES_FP16, param as _param
from ....infrastructure.registry import Infrastructure as _Infrastructure
from ... import scenario_types as _st


# --- Feeding and saving a training fleet (added 2026-10-20) -------------
Gpt3StorageFootprint = _st.StorageFootprintScenario(
    name="Storage footprint of a 30-day 175B pretraining run",
    description="1.5 trillion 4-byte token IDs read once; a resumable 14-byte-per-parameter "
    "checkpoint every 10 minutes; every 100th checkpoint and the final one archived.",
    workload=_Models.Language.GPT3,
    bytes_per_param=_ReferenceStats.StorageTrainingCorpus.CheckpointBytesPerParameter,
    corpus_bytes=_ReferenceStats.StorageTrainingCorpus.TokenizedText,
    source_bytes=_ReferenceStats.StorageTrainingCorpus.CompressedSource,
    run_duration=_ReferenceStats.StorageTrainingCorpus.TrainingWindow,
    save_interval=_ReferenceStats.StorageTrainingCorpus.CheckpointInterval,
    shard_bytes=_ReferenceStats.StorageTrainingCorpus.TokenizedShardSize,
    archive_every=100,  # scenario assumption: retention keeps every 100th checkpoint
)


Gpt3TieredCheckpoint2K = _st.TieredCheckpointScenario(
    name="175B checkpoint staged on local NVMe across 256 DGX H100 nodes",
    description="Each node writes its shard to four striped Gen4 NVMe drives, then the "
    "shards drain to the shared 1 TB/s parallel file system.",
    workload=_Models.Language.GPT3,
    bytes_per_param=_ReferenceStats.StorageTrainingCorpus.CheckpointBytesPerParameter,
    fleet=_Clusters.Production_2K,
    local_bandwidth=_Storage.Production2KCheckpointPath.local_stage.aggregate_bandwidth,
    durable_bandwidth=_Storage.Production2KCheckpointPath.durable_store.bandwidth,
    save_interval=_ReferenceStats.StorageTrainingCorpus.CheckpointInterval,
)


Llama2_70BCheckpointStorm1K = _st.CheckpointStormScenario(
    name="70B checkpoint written by every one of 1,024 replicas",
    description="Each data-parallel replica writes the full resumable state through "
    "100 GB/s of storage ingest; a bound on write volume, not a deployable layout.",
    workload=_Models.Language.Llama2_70B,
    bytes_per_param=_ReferenceStats.StorageTrainingCorpus.CheckpointBytesPerParameter,
    replicas=_Clusters.Training_1K.total_accelerators,
    ingest_bandwidth=_Fabrics.InfiniBand_XDR.bandwidth,
)


TextInputDemand2K = _st.InputDemandScenario(
    name="Text input demand of 2,048 GPUs",
    description="Every GPU reads one packed 4,096-token sequence of token IDs per 200 ms step.",
    consumers=_Clusters.Production_2K.total_accelerators,
    samples_per_step=4096,  # scenario assumption: packed tokens per GPU per step
    bytes_per_sample=_ReferenceStats.StorageTrainingCorpus.TokenIdBytes,
    step_time=_ReferenceStats.StorageTrainingCorpus.StepTime,
)


ImageInputDemand2K = _st.InputDemandScenario(
    name="Image input demand of 2,048 GPUs",
    description="Every GPU reads 256 ImageNet-sized JPEGs per 200 ms step.",
    consumers=_Clusters.Production_2K.total_accelerators,
    samples_per_step=256,  # scenario assumption: images per GPU per step
    bytes_per_sample=_Datasets.ImageNet.mean_compressed_image_size,
    step_time=_ReferenceStats.StorageTrainingCorpus.StepTime,
)


OverlappedInputStall = _st.DataStallScenario(
    name="250 ms delivery against 200 ms compute",
    description="Storage delivers each batch in 250 ms; compute takes one 200 ms step.",
    compute_time=_ReferenceStats.StorageTrainingCorpus.StepTime,
    io_time=Q_(250, "ms"),  # scenario assumption: delivery time per batch
)


PrefetchDepthByTier = _st.PrefetchDepthScenario(
    name="Prefetch depth against each tier's p99 read",
    description="200 ms compute windows; 40 MB decoded batches, five deep, eight GPUs per node.",
    compute_time=_ReferenceStats.StorageTrainingCorpus.StepTime,
    tail_latencies={
        "nvme": Q_(50, "ms"),  # scenario assumption: local NVMe p99 read
        "pfs": Q_(200, "ms"),  # scenario assumption: parallel file system p99 read
        "object": Q_(500, "ms"),  # scenario assumption: object storage p99 read
    },
    batch_bytes=Q_(40, "MB"),  # scenario assumption: collated batch per GPU
    safety_depth=5,  # scenario assumption: object-storage safety depth
    consumers_per_node=_Nodes.DGX_H100.accelerators_per_node,
)


HotTierCacheExample = _st.TieredCacheScenario(
    name="95 percent hits, misses 100 times slower",
    description="Illustrative two-tier cache in units of one hit's service time.",
    hit_rate=0.95,  # scenario assumption
    miss_penalty=100.0,  # scenario assumption
)


MultimodalCorpusNodeCache = _st.TieredCacheScenario(
    name="A 35 TB multimodal corpus over one node's local NVMe",
    description="Four 7.68 TB Gen4 drives hold part of a uniformly sampled corpus; misses "
    "read the node's share of the parallel file system.",
    hot_capacity=(_Storage.LocalNvmeGen4_7T68x4.devices_per_node * _Storage.LocalNvmeGen4_7T68x4.device.capacity),
    corpus_parts={
        "images": Q_(20, "TB"),  # scenario assumption
        "text": Q_(10, "TB"),  # scenario assumption
        "audio": Q_(5, "TB"),  # scenario assumption
    },
    hot_bandwidth=_Storage.LocalNvmeGen4_7T68x4.aggregate_bandwidth,
    miss_bandwidth=_Storage.PfsPerNodeShare.bandwidth,
)


GpuDirectStorageNodeCpu = _st.HostIoCpuScenario(
    name="Host CPU spent on small-image I/O in one DGX H100 node",
    description="8,000 images per second per GPU, through host buffers or GPUDirect Storage; "
    "illustrative CPU service times per request.",
    requests_per_device=8000,  # scenario assumption: images per second per GPU
    devices=_Nodes.DGX_H100.accelerators_per_node,
    service_times={"host": _Storage.TraditionalGpuIoLatency, "gds": _Storage.GpuDirectStorageLatency},
)


StorageTailFanout100 = _st.AvailabilityScenario(
    name="A step reading from 100 storage servers",
    description="Each server is slow 1 percent of the time, independently; all_up is the chance that none is slow.",
    availability=0.99,  # scenario assumption: share of fast responses per server
    n_components=100,  # scenario assumption: servers one step reads from
)


ImageFileMetadataMeltdown = _st.ShardPackagingScenario(
    name="200 million image files on a 500 GB/s file system",
    description="One file per image, then the same images packaged into 50,000 tar shards.",
    files=200_000_000,  # scenario assumption
    shards=50_000,  # scenario assumption
    rated_bandwidth=Q_(500, "GB/s"),  # scenario assumption
)


ObjectStorage100TB = _st.ObjectStorageBillScenario(
    name="A 100 TB dataset in object storage, read ten times across regions",
    description="Representative S3 Standard, cross-boundary egress, GET, and Glacier "
    "retrieval prices; one GET per file for 100 million files.",
    dataset_bytes=Q_(100, "TB"),  # scenario assumption
    storage_rate=_Infrastructure.Pricing.Storage.S3StandardLowPerTbMonth.rate,
    billing_period=Q_(12, "month"),
    egress_rate=_Infrastructure.Pricing.Cloud.EgressPerGB.rate,
    reads=10,  # scenario assumption: epochs read across the boundary
    files=100_000_000,  # scenario assumption
    request_fee=_Infrastructure.Pricing.Storage.S3GetPer1kRequests.rate,
    requests_per_fee=1000,  # the GET price is quoted per 1,000 requests
    retrieval_rate=_Infrastructure.Pricing.Storage.GlacierRetrievalPerGB.rate,
)


VisionCorpusStreamVsStage = _st.StreamVsStageScenario(
    name="Stream or stage a 50 TB vision corpus",
    description="20 epochs per run, four runs a year; object storage, local NVMe, and egress at representative prices.",
    dataset_bytes=Q_(50, "TB"),  # scenario assumption
    epochs=20,  # scenario assumption
    runs_per_year=4,  # scenario assumption
    object_rate=_Infrastructure.Pricing.Storage.S3StandardLowPerTbMonth.rate,
    local_rate=_Infrastructure.Pricing.Storage.NvmeLowPerTbMonth.rate,
    egress_rate=_Infrastructure.Pricing.Cloud.EgressPerGB.rate,
    year=Q_(12, "month"),
)


InputStallIdleCost1K = _st.UtilizationLossScenario(
    name="1,024 GPUs idled by input stalls",
    description="The reference cluster at the fleet GPU-hour price while its busy fraction "
    "falls from 90 to 70 percent, over a 30-day run.",
    fleet=_Clusters.Training_1K,
    device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
    healthy_utilization=0.90,  # scenario assumption
    degraded_utilization=0.70,  # scenario assumption
    duration=Q_(30, "day"),  # scenario assumption
)


SyntheticDataProvenance = _st.SyntheticFootprintScenario(
    name="1 TB of generated samples with provenance and verification",
    description="40 percent provenance metadata at rest; three verifier passes read the payload.",
    payload_bytes=Q_(1, "TB"),  # scenario assumption
    provenance_overhead=0.4,  # scenario assumption
    verification_passes=3,  # scenario assumption
)


Gpt3WeightsColdStart = _st.TransferScenario(
    name="Loading 175B FP16 weights into one serving node",
    description="One replica's weights from each source: a single object-store client, the "
    "node's file-system share, one Gen5 drive, eight Gen4 drives in parallel, "
    "eight PCIe links, or host DRAM shared by eight transfers.",
    payload=_memory_from_params(_Models.Language.GPT3.parameters, _BYTES_FP16 / _param),
    links={
        "object store": Q_(1, "GB/s"),  # scenario assumption: single-client object-store read
        "file system share": _Storage.PfsPerNodeShare.bandwidth,
        "one Gen5 NVMe": _Hardware.Tech.Storage.NvmeGen5.bandwidth,
        "eight Gen4 NVMe": _Nodes.DGX_H100.accelerators_per_node * _Hardware.Tech.Storage.NvmeGen4.bandwidth,
        "eight PCIe links": _Nodes.DGX_H100.accelerators_per_node * _Hardware.Cloud.H100.interconnect.bandwidth,
        "host DRAM": _Hardware.Tech.Storage.HostDram.bandwidth,
    },
)


Gpt3RolloutFanout100 = _st.TransferScenario(
    name="A rollout of 175B FP16 weights to 100 replicas at once",
    description="Every replica reads its weights from a shared tier in one burst.",
    payload=_memory_from_params(_Models.Language.GPT3.parameters, _BYTES_FP16 / _param),
    copies=100,  # scenario assumption: serving replicas updated in one rollout
    links={
        "file system": _Storage.PfsOneTbPerSecond.bandwidth,
        "object store": Q_(100, "GB/s"),  # scenario assumption: aggregate object-store rate
    },
)


NodeStripeCodecs = _st.CompressedReadScenario(
    name="Two codecs on one node's four-drive NVMe stripe",
    description="A light codec at 3:1 decoding 1.5 GB/s per core against a denser one at 4:1 "
    "decoding 0.5 GB/s per core; illustrative rates.",
    read_bandwidth=_Storage.LocalNvmeGen4_7T68x4.aggregate_bandwidth,
    compression_ratios={"light": 3.0, "dense": 4.0},  # scenario assumption
    decode_rates={"light": Q_(1.5, "GB/s"), "dense": Q_(0.5, "GB/s")},  # scenario assumption, per core
)
