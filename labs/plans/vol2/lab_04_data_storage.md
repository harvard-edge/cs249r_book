# Mission Plan: lab_04_data_storage (Volume 2)

## 1. Chapter Alignment

- **Chapter:** Data Storage (`@sec-data-storage`)
- **Core Invariant:** The **Storage-Compute Chasm** — GPU throughput has grown 236x (V100 to B200) while NVMe sequential bandwidth has grown only 4x in the same period. An H100 consumes data from HBM at 3.35 TB/s, roughly 479x faster than a single NVMe drive (7 GB/s). This gap means that without a properly tiered storage pipeline with aggressive prefetching, accelerators starve regardless of how fast they compute.
- **Central Tension:** Students believe that modern NVMe SSDs at 7 GB/s are "fast enough" to feed GPU training. The chapter shows that an H100 processes its local batch in ~200 ms, and if storage cannot deliver the next batch within that window, the GPU sits idle. A single NVMe can sustain only ~35 samples/sec for large image datasets, while 8 H100s demand ~800 samples/sec. The real bottleneck is not NVMe speed but the pipeline architecture that stages data from cold storage to hot cache fast enough to hide the 479x bandwidth gap.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students predict whether adding more local NVMe cache will improve training speed when the bottleneck is the network link from object storage (S3) to the node. The instrument reveals that NVMe cache helps only if data is pre-staged; during training, the S3-to-node link (1--10 Gbps) is the binding constraint. The aha moment: NVMe is not a speed solution but a latency-hiding buffer. Its value depends entirely on whether the pipeline can fill it faster than the GPUs drain it.

**Act 2 (Design Challenge, 23 min):** Students must size a tiered storage pipeline (Object Store -> NVMe -> HBM) to keep 8 H100s fed during training of a 175B model. They discover that checkpoint writes (~1,050 GB every 30 minutes) compete with data reads for NVMe bandwidth, creating write storms that stall training unless checkpoint I/O is isolated. The design challenge requires finding the minimum NVMe capacity and the maximum checkpoint frequency that keep GPU utilization above 90%.

---

## 3. Act 1: The Pipeline Bottleneck (Calibration -- 12 minutes)

### Pedagogical Goal

Students think of storage as a simple speed hierarchy: HBM is fast, NVMe is medium, object storage is slow. The chapter reframes storage as a pipeline problem. A single NVMe drive at 7 GB/s appears adequate, but when 8 H100s consume batches at 200 ms intervals, the aggregate demand is ~40 GB/s (if each GPU needs 1 GB per batch, 8 GPUs need 8 GB every 200 ms = 40 GB/s). Even 4 NVMe drives in RAID 0 (28 GB/s) cannot match this without prefetching from a lower tier. And the lower tier (object store over network) delivers only 1--10 Gbps (0.125--1.25 GB/s), creating an impedance mismatch that no amount of local caching can solve unless data is pre-staged before training begins.

### The Lock (Structured Prediction)

Present a **multiple-choice prediction** before any instruments unlock:

> "Your training cluster has 4 NVMe SSDs per node (28 GB/s aggregate). You are streaming training data from S3 object storage over a 10 Gbps network link (1.25 GB/s). Adding 4 more NVMe SSDs (doubling local cache to 56 GB/s) will improve GPU utilization by how much?"

Options:
- A) ~2x improvement -- double the cache, double the throughput
- B) ~1.5x improvement -- more cache helps but with diminishing returns
- C) ~1.1x improvement -- marginal because NVMe was already fast enough
- **D) ~0% improvement -- the bottleneck is the S3 network link (1.25 GB/s), not the local NVMe speed** ← correct

The answer is D because the data pipeline has a serial bottleneck: data must flow S3 -> Network -> NVMe -> HBM. The slowest link (1.25 GB/s network) limits end-to-end throughput regardless of how fast the NVMe tier is. Doubling NVMe capacity increases the buffer but does not increase the fill rate. Students learn that pipeline throughput equals the throughput of the slowest stage.

### The Instrument: Data Pipeline Waterfall

A **waterfall diagram** showing the data flow from Object Store to GPU:

- **Stages (left to right):** Object Store -> Network -> NVMe Cache -> Host DRAM -> HBM
- **Bar heights:** Throughput at each stage (GB/s, log scale)
- **Bottleneck indicator:** The slowest stage is highlighted in RedLine with a "BOTTLENECK" label

Controls:
- **Network bandwidth selector** (1 Gbps / 10 Gbps / 25 Gbps / 100 Gbps, default: 10 Gbps): Changes the S3-to-node throughput
- **NVMe count slider** (1 -- 8 drives, step: 1, default: 4): Changes aggregate NVMe bandwidth (7 GB/s per drive)
- **Prefetch toggle** (on/off, default: off): When ON, data is pre-staged to NVMe before training starts; the pipeline bottleneck shifts to the NVMe-to-HBM link
- **GPU demand indicator** (read-only): Shows required throughput for 8 H100s at the current batch size (~40 GB/s)

### The Reveal

After interaction:

> "You predicted [X]% improvement from doubling NVMe. The actual improvement is **0%** because the S3 network link (1.25 GB/s) is the bottleneck. The NVMe tier is a buffer, not a source. With prefetching enabled (data pre-staged before training), the pipeline shifts to NVMe-to-HBM throughput, and adding NVMe drives finally helps. The lesson: **identify the bottleneck before adding capacity.**"

Surface the storage-compute chasm:
> "GPU throughput has grown 236x since 2016. NVMe throughput has grown 4x. The 479x gap between HBM (3.35 TB/s) and NVMe (7 GB/s) cannot be closed by hardware alone — it requires pipeline architecture."

### Reflection (Structured)

Four-option multiple choice:

> "The chapter states that ML workloads 'invert traditional storage assumptions.' At 4 KB request size, sequential reads outperform random reads by 10x. Which storage optimization matters most for ML training?"

- A) Maximizing IOPS (input/output operations per second) for random access patterns
- **B) Maximizing sequential bandwidth through large shard sizes and streaming reads** ← correct
- C) Using caching algorithms that keep recently accessed data hot
- D) Optimizing write-ahead logs for transactional consistency

### Math Peek (collapsible)

Pipeline throughput = min(throughput at each stage):
$$\text{Throughput}_{\text{pipeline}} = \min(BW_{\text{S3}}, BW_{\text{net}}, BW_{\text{NVMe}}, BW_{\text{DRAM}}, BW_{\text{HBM}})$$

GPU demand: $\text{Demand} = \frac{\text{Batch size (bytes)}}{\text{Step time (s)}} \times N_{\text{GPUs}}$

Storage-compute ratio: $\frac{BW_{\text{HBM}}}{BW_{\text{NVMe}}} = \frac{3{,}350}{7} \approx 479\times$

---

## 4. Act 2: The Checkpoint Storm (Design Challenge -- 23 minutes)

### Pedagogical Goal

Students think of storage as a read-only problem during training: data goes in, gradients come out. The chapter reveals that **checkpoint writes** are equally demanding and create a contention problem. A 175B model checkpoint with optimizer state is ~1,050 GB. Saving every 30 minutes means writing 1 TB in under 30 seconds (to minimize training disruption), requiring 33 GB/s of sustained write bandwidth. This competes with the same NVMe drives that are serving training data. During a checkpoint storm, NVMe bandwidth is consumed by writes, potentially starving the data pipeline and idling GPUs.

### The Lock (Numeric Prediction)

Before instruments unlock:

> "You are training a 175B model on 256 nodes. Checkpoints (~1,050 GB with optimizer state) are saved every 30 minutes to local NVMe, then asynchronously moved to shared storage. If you want the checkpoint write to complete within 30 seconds, what minimum write bandwidth (GB/s) do you need?"

Students type a number (GB/s). Expected wrong answers: 5--10 GB/s (students underestimate checkpoint size). Actual: $1{,}050 / 30 = 35$ GB/s, which requires 5 NVMe drives dedicated to writes — half the node's total NVMe bandwidth, leaving only 14 GB/s for training data reads during the checkpoint window.

### The Instrument: Storage Budget Allocator

A **resource allocation panel** with three interactive components:

**Component 1: Checkpoint Size Calculator**
- Model size slider (7B / 70B / 175B, default: 175B)
- Shows breakdown: Weights (FP16) + Gradients (FP16) + Optimizer (2x FP32) = total
- Formula: $\text{Checkpoint} = P \times (2 + 2 + 8) = P \times 12$ bytes (for Adam)
- At 175B: $175 \times 10^9 \times 12 / 10^9 \approx 2{,}100$ GB (full state) or $\sim 1{,}050$ GB (with FP16 weights + optimizer)

**Component 2: NVMe Budget Split**
- Total NVMe BW bar (configurable: 4--8 drives x 7 GB/s = 28--56 GB/s)
- Draggable divider splitting between "Read (training data)" and "Write (checkpoints)"
- Real-time display of: read throughput available, write time for checkpoint, GPU utilization impact

**Component 3: Checkpoint-Training Interference Timeline**
- **X-axis:** Time (0 -- 60 minutes, showing 2 checkpoint cycles)
- **Y-axis:** GPU utilization (0% -- 100%)
- **Shaded zones:** Normal training (green, 90%+), Checkpoint write window (orange, utilization drops), Post-checkpoint recovery (green returns)
- **Controls:**
  - Checkpoint interval slider (5 / 10 / 15 / 30 / 60 min, default: 30 min)
  - Checkpoint write BW allocation (25% / 50% / 75% of NVMe, default: 50%)

### The Scaling Challenge

**"Find the checkpoint interval and NVMe allocation that keeps GPU utilization above 90% while ensuring checkpoint writes complete before the next checkpoint is due."**

Students must balance:
- Frequent checkpoints (5 min) = less work lost on failure but more write storms
- Infrequent checkpoints (60 min) = more work lost (cluster MTBF ~4.4 hours means ~1 failure per hour)
- High NVMe write allocation = faster checkpoints but slower data reads
- Low NVMe write allocation = faster reads but checkpoint may not finish in time

Optimal: checkpoint every 15--30 min with 50% NVMe write allocation; checkpoint completes in ~35s; GPU utilization drops to ~85% during write window then recovers.

### The Failure State

**Trigger condition:** `checkpoint_write_time > checkpoint_interval` OR `gpu_utilization < 0.70` during write window

**Visual change:** The timeline shows the checkpoint write bar (orange) exceeding the interval boundary and overlapping with the next checkpoint. GPU utilization drops into the red zone. The NVMe budget bar shows the read allocation segment turning red.

**Banner text (write overflow):**
> "**Checkpoint Storm -- Write Exceeds Interval.** Checkpoint size: [X] GB. Write bandwidth: [Y] GB/s. Write time: [Z] min. Interval: [W] min. The next checkpoint starts before the current one finishes. Training data reads are starved. Increase NVMe count, reduce checkpoint frequency, or implement asynchronous staging to shared storage."

**Banner text (utilization collapse):**
> "**GPU Starvation -- Storage Contention.** During checkpoint writes, only [X] GB/s remains for training data. GPUs require [Y] GB/s. Utilization drops to [Z]%. The storage pipeline is the bottleneck, not the accelerator."

### Structured Reflection

Four-option multiple choice:

> "Over a 30-day training run on 256 nodes, the total checkpoint data written is approximately 4.5 PB. The chapter identifies this as a key storage system design parameter. What is the primary risk of reducing checkpoint frequency from every 15 minutes to every 60 minutes?"

- A) The model quality degrades because weights are not saved frequently enough
- B) The NVMe drives wear out faster due to larger individual write bursts
- **C) When a hardware failure occurs (MTBF ~4.4 hours at scale), up to 60 minutes of training is lost and must be recomputed, wasting GPU-hours worth millions of dollars** ← correct
- D) The storage system runs out of capacity because larger checkpoints accumulate faster

### Math Peek (collapsible)

Checkpoint size (Adam optimizer):
$$\text{Checkpoint} = P \times (\underbrace{2}_{\text{FP16 weights}} + \underbrace{2}_{\text{FP16 grads}} + \underbrace{4}_{\text{FP32 moment 1}} + \underbrace{4}_{\text{FP32 moment 2}}) = 12P \text{ bytes}$$

For 175B: $\approx 2{,}100$ GB (full) or $\approx 1{,}050$ GB (FP16 weights + FP32 optimizer)

Write bandwidth requirement: $BW_{\text{write}} = \frac{\text{Checkpoint size}}{\text{Target write time}}$

At 1,050 GB / 30 s = 35 GB/s

Total checkpoint writes (30 days, every 30 min):
$$\text{Total} = 1{,}050 \text{ GB} \times \frac{30 \times 24 \times 60}{30} = 1{,}050 \times 1{,}440 \approx 1{,}512{,}000 \text{ GB} \approx 1.5 \text{ PB per node}$$

Fleet total (256 nodes): $\sim 4.5$ PB (with optimizer sharding across nodes)

---

## 5. Visual Layout Specification

### Act 1: Data Pipeline Waterfall
- **Chart type:** Horizontal waterfall / pipeline diagram
- **X-axis (implicit):** Pipeline stages (Object Store -> Network -> NVMe -> DRAM -> HBM)
- **Y-axis:** Throughput at each stage (GB/s, log scale)
- **Annotations:** Bottleneck stage highlighted in RedLine; GPU demand line shown as dashed threshold
- **Failure state:** N/A (Act 1)

### Act 2: NVMe Budget Split Bar
- **Chart type:** Horizontal stacked bar with draggable divider
- **Segments:** Read (BlueLine), Write (OrangeLine)
- **Total width:** Proportional to total NVMe bandwidth
- **Failure state:** When read allocation < GPU demand, read segment turns RedLine

### Act 2: Checkpoint-Training Interference Timeline
- **Chart type:** Area chart over time
- **X-axis:** Time (0 -- 60 min)
- **Y-axis:** GPU utilization (0% -- 100%)
- **Regions:** Normal training (GreenLine fill), Checkpoint write (OrangeLine fill), Starvation (RedLine fill)
- **Failure state:** When checkpoint overlaps next interval OR utilization < 70%, affected region turns RedLine

---

## 6. Deployment Context Definitions

| Context | Storage Config | Read BW | Write BW | Key Constraint |
|---|---|---|---|---|
| **NVMe local (low latency)** | 4x NVMe SSD per node, RAID 0 | 28 GB/s sequential | 20 GB/s sustained | Training data is pre-staged; NVMe is the primary data source; checkpoint writes compete with reads |
| **Distributed FS (high capacity)** | Lustre/GPFS PFS over 100G Ethernet | 1.25--12.5 GB/s per node | 1.25--12.5 GB/s per node | High capacity (PB scale) but low per-node throughput; must pre-stage hot data to local NVMe; checkpoint staging adds network contention |

The two contexts demonstrate the capacity-vs-bandwidth trade-off. Local NVMe provides the bandwidth GPUs need but with limited capacity (4--30 TB per node). Distributed FS provides limitless capacity but at 10--20x lower per-node bandwidth. The engineering challenge is building a pipeline that uses both tiers efficiently.

---

## 7. Design Ledger Output

```json
{
  "chapter": "v2_04",
  "nvme_drives_per_node": 4,
  "nvme_read_bw_gbs": 14,
  "nvme_write_bw_gbs": 14,
  "checkpoint_interval_min": 30,
  "checkpoint_size_gb": 1050,
  "checkpoint_write_time_s": 35,
  "gpu_utilization_during_ckpt_pct": 85,
  "storage_compute_gap_x": 479
}
```

The `checkpoint_interval_min` and `checkpoint_size_gb` fields feed forward to:
- **Lab V2-05 (Distributed Training):** The checkpoint overhead is included in the total training time calculation.
- **Lab V2-07 (Fault Tolerance):** The checkpoint interval directly determines the maximum wasted work on failure.

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| HBM 3.35 TB/s bandwidth | @sec-storage-fuel-line, line 141 | "H100 GPU consumes data from its HBM at 3.35 TB/s" |
| NVMe 7 GB/s sequential throughput | @sec-storage-fuel-line, line 141 | "NVMe drives achieve 7 GB/s of sequential throughput" |
| 479x gap HBM-to-NVMe | @sec-storage-fuel-line, lines 112--125 | "bw_ratio_val = 479" in StorageHierarchyAnalysis LEGO |
| GPU processes batch in ~200 ms | @sec-storage-fuel-line, line 139 | "each accelerator processes its local batch in roughly 200 ms" |
| Checkpoint ~1,050 GB for 175B model | @sec-storage-fuel-line, lines 114--124 | "ckpt_total_gb_val > 1000" in StorageHierarchyAnalysis LEGO |
| 4.5 PB total checkpoint writes (30-day run) | @sec-storage-fuel-line, line 145 | "~4.5 PB of checkpoint writes" |
| GPU throughput 236x growth vs NVMe 4x | @sec-storage-fuel-line, figure | "GPU TFLOPS (236x) vs NVMe GB/s (4x)" in @fig-storage-compute-chasm |
| Sequential outperforms random by 10x at 4 KB | @sec-storage-workload-inversion, line 378 | "at 4 KB, sequential reads outperform random reads by 10x" |
| ML workloads invert IOPS assumptions | @tbl-storage-assumptions | "ML Workloads Invert Traditional Storage Assumptions" |
| 300,000x bandwidth gap HBM to object storage | @tbl-storage-hierarchy-merged | "The 300,000x bandwidth gap between HBM and object storage" |
