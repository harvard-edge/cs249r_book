# Mission Plan: lab_04_data_engr

## 1. Chapter Alignment

- **Chapter:** Data Engineering (`@sec-data-engineering`)
- **Core Invariant:** The **Energy-Movement Invariant** -- moving a bit costs 100--1,000x more energy than computing on it. A DRAM access costs ~640x more energy than a floating-point multiply; network transfer costs ~350,000x more per bit. The **Feeding Tax** ($1 - \eta$, where $\eta = \min(BW_{\text{disk}}/BW_{\text{required}}, 1)$) quantifies how much of the accelerator's potential is wasted waiting for data.
- **Central Tension:** Students believe the GPU is the bottleneck and that data loading is a solved problem ("just call `DataLoader`"). The chapter reveals that a standard cloud disk (250 MB/s) feeding an A100 (requiring ~5.8 GB/s for ResNet-50 at peak throughput) causes the GPU to sit idle over 95% of the time. The data pipeline, not the model, is the binding constraint. Data has physical mass: moving 1 PB over a 100 Gbps link takes ~23 hours and costs over $90,000 in egress fees.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students predict GPU utilization when training ResNet-50 from a standard cloud disk. They expect ~80--90% utilization ("the GPU is doing all the work"). The chapter's Feeding Tax calculation shows that a 250 MB/s disk delivering to an A100 that demands ~5.8 GB/s yields a feeding tax exceeding 95% -- the GPU idles more than 95% of the time. Students see this in a pipeline utilization gauge and discover that storage bandwidth, not compute, is the real bottleneck.

**Act 2 (Design Challenge, 22 min):** Students confront data gravity at petabyte scale. They must decide whether to move 1 PB of training data to a remote GPU cluster (23 hours, $90,000+ egress) or move the compute to the data. They then design a storage pipeline using tiered storage (NVMe, S3, Glacier) to eliminate the Feeding Tax, discovering that the shift from cloud disk to NVMe can provide a 20x bandwidth improvement, dramatically changing the pipeline economics.

---

## 3. Act 1: The Feeding Tax (Calibration -- 12 minutes)

### Pedagogical Goal

Students assume that once data is "in the cloud," loading it is instantaneous. The chapter quantifies the mismatch between accelerator demand and storage supply. An A100 processing ResNet-50 images at peak throughput can consume ~40,000 images/sec, requiring ~5.8 GB/s of sustained I/O. A standard cloud volume (e.g., AWS gp3) delivers 250 MB/s baseline. The feeding tax ($1 - 250/5800 = 95.7\%$) means the GPU sits idle for 95.7% of each training step. This act makes the invisible I/O bottleneck visible.

### The Lock (Structured Prediction)

Present a **multiple-choice prediction** before any instruments unlock:

> "You are training ResNet-50 on an A100 GPU (312 TFLOPS) using images stored on a standard cloud disk (250 MB/s). What percentage of time is the GPU actually computing (not waiting for data)?"

Options:
- A) About 80--90% -- GPUs are expensive, so cloud providers optimize for utilization
- B) About 50--60% -- there is always some I/O overhead, but it is manageable
- C) About 20--30% -- disk is slower than GPU, but prefetching helps
- **D) Less than 5% -- the GPU spends over 95% of its time idle, starved for data** <-- correct

The correct answer is less than 5% GPU utilization (feeding tax > 95%). The chapter states: "the feeding tax can exceed [95+]%, meaning the GPU spends the majority of its time waiting for bits." Students overwhelmingly pick A or B because they assume the system is well-engineered by default.

### The Instrument: Pipeline Utilization Gauge

A **dashboard with three components:**

**Component 1: Feeding Tax Gauge** -- a circular gauge (0--100%) showing GPU idle time:
- Green zone: 0--20% idle (healthy)
- Yellow zone: 20--50% idle (degraded)
- Red zone: 50--100% idle (starving)
- Needle position computed from: $\text{Idle\%} = (1 - \min(BW_{\text{storage}} / BW_{\text{required}}, 1)) \times 100$

**Component 2: Pipeline Waterfall** -- a horizontal stacked bar showing one training step:
- **Disk Read** (RedLine): Time to read a batch from storage
- **Preprocessing** (OrangeLine): CPU decode/augment time
- **GPU Compute** (BlueLine): Forward + backward pass
- **GPU Idle** (gray, hatched): Time GPU waits for next batch

**Component 3: Throughput Counter** -- images/sec achieved vs. peak possible

Controls:
- **Storage tier** (radio): Standard Cloud Disk (250 MB/s) / Premium SSD (1 GB/s) / NVMe (5 GB/s) / RAM Disk (25 GB/s)
- **Number of dataloader workers** (slider): 0, 1, 2, 4, 8, 16 (default 0)
- **Deployment context toggle**: H100 (Cloud) / Jetson Orin NX (Edge)

At default (Cloud Disk, 0 workers): the gauge shows >95% idle (deep red). Switching to NVMe with 8 workers drops idle to ~15% (green zone). The chapter states "8 workers" as the canonical ResNet-50 dataloader configuration.

### The Reveal

After interaction:
> "You predicted [X]% GPU compute utilization. At 250 MB/s cloud disk with 0 workers: GPU utilization = **4.3%**. The GPU spent **95.7%** of its time idle, waiting for data. This is the Feeding Tax: $1 - (250 \text{ MB/s} / 5{,}800 \text{ MB/s}) = 95.7\%$. Upgrading to NVMe (5 GB/s) + 8 workers reduces idle time to ~15%, a **5.6x throughput improvement** without touching the model or the GPU."

### Reflection (Structured)

Four-option multiple choice:

> "The Feeding Tax exceeded 95% with a standard cloud disk. What is the most cost-effective fix?"

- A) Upgrade to an H100 GPU with 3x more TFLOPS
- B) Use a smaller model that requires less data per batch
- **C) Upgrade the storage tier to NVMe and add parallel dataloader workers -- the bottleneck is I/O, not compute** <-- correct
- D) Reduce the batch size to require fewer bytes per step

**Math Peek (collapsible):**
$$\eta_{\text{pipeline}} = \min\left(\frac{BW_{\text{storage}}}{BW_{\text{required}}}, 1.0\right)$$
$$\text{Feeding Tax} = 1 - \eta_{\text{pipeline}}$$
$$BW_{\text{required}} = \text{images/sec}_{\text{peak}} \times \text{bytes/image} = 40{,}000 \times 150{,}528 \text{ B} \approx 5.8 \text{ GB/s}$$

---

## 4. Act 2: Data Gravity (Design Challenge -- 22 minutes)

### Pedagogical Goal

Students believe data transfer is free and instantaneous at cloud scale. The chapter's Data Gravity calculation demolishes this: moving 1 PB over a 100 Gbps link (12.5 GB/s) takes ~23 hours and costs over $90,000 in egress fees. The rule of thumb emerges: "For petabyte-scale data, code moves to data. For gigabyte-scale data, data moves to code." Students must design a tiered storage architecture that eliminates the Feeding Tax while respecting cost constraints.

### The Lock (Numeric Prediction)

Before instruments unlock:

> "You have 1 PB of training data in US East. Your GPU cluster is in US West with a dedicated 100 Gbps link. How many hours will the transfer take? Enter a number (1--1000)."

Expected wrong answers: 1--5 hours (students drastically underestimate data gravity). Actual: $10^6 \text{ GB} / 12.5 \text{ GB/s} = 80{,}000 \text{ s} \approx$ **23 hours**. The system will overlay the student's estimate on the actual calculation.

### The Instrument: Data Gravity Calculator and Storage Tier Designer

**Primary chart: Transfer Time vs. Cost Scatter Plot**

- **X-axis:** Transfer time (hours), range 0--200
- **Y-axis:** Transfer cost (USD), range 0--$150,000
- **Data points:** One point per transfer method (labeled):
  - 10 Gbps link: ~9.3 days, $90,000+ egress
  - 100 Gbps link: ~23 hours, $90,000+ egress
  - AWS Snowball (physical truck): ~5--7 days shipping, ~$200 base fee
  - Move compute to data (no transfer): 0 hours, 0 egress (but cluster rental cost)

Controls:
- **Dataset size** (slider): 1 GB, 10 GB, 100 GB, 1 TB, 10 TB, 100 TB, 1 PB (log scale, default 1 PB)
- **Network speed** (radio): 1 Gbps / 10 Gbps / 100 Gbps
- **Egress cost per GB** (slider): $0.01--$0.12, default $0.09, step $0.01

As dataset size decreases, the "move data" options become viable (1 GB transfers in seconds for pennies). As it increases, the physical truck (Sneakernet) and "move compute" options dominate. A **crossover annotation** marks the dataset size where "move compute" becomes cheaper than "move data."

**Secondary instrument: Storage Tier Pipeline Builder**

A **pipeline diagram** showing the data path from storage to GPU:

```
[Storage Tier] --> [Network/Bus] --> [CPU Memory] --> [Dataloader Workers] --> [GPU HBM]
```

Each stage shows its bandwidth as a pipe width:
- Glacier: 0.05 GB/s (retrieval throttled)
- S3 Standard: 0.5 GB/s (network-limited)
- Local NVMe: 5 GB/s
- RAM Cache: 25 GB/s
- GPU HBM bandwidth: 2,039--3,350 GB/s (always wider than the pipe feeding it)

Controls:
- **Hot tier** (radio): NVMe / SSD / S3
- **Warm tier** (radio): S3 / Glacier
- **Dataloader workers** (slider): 0--16, default 4
- **Prefetch factor** (slider): 1--4, default 2
- **Deployment context toggle**: H100 (Cloud) / Jetson Orin NX (Edge)

The pipeline shows the narrowest pipe as the bottleneck (colored RedLine). The Feeding Tax gauge from Act 1 updates to show the combined effect of storage tier + workers. On Jetson Orin NX (Edge), the pipeline constraints differ: local NVMe is standard, network is the bottleneck for dataset updates.

### The Scaling Challenge

**"Design a storage pipeline that achieves < 10% Feeding Tax for ResNet-50 training on H100, spending less than $500/month on storage."**

Students must balance:
- NVMe ($70--140/TB/month from chapter) is fast but expensive at petabyte scale
- S3 ($23/TB/month) is cheap but slow (0.5 GB/s, insufficient)
- Tiered approach: keep hot epoch data on NVMe (e.g., 1 TB active set), warm data on S3, cold data on Glacier

At 8 workers + NVMe hot tier: Feeding Tax drops below 10%. Storage cost for 1 TB NVMe = $70--140/month. Meeting both constraints requires active data management.

On Jetson Orin NX: the dataset must be local (SD card or eMMC at ~100 MB/s). The Feeding Tax depends on the edge device's much lower compute demand (25 TFLOPS vs. 312), so a slower storage tier may still achieve acceptable utilization.

### The Failure State

**Trigger condition 1 (Budget exceeded):** `storage_cost_monthly > 500`

**Visual change:** Storage cost counter turns RedLine. Banner:
> "**BUDGET EXCEEDED -- Storage costs $[X]/month.** Your pipeline uses [Y] TB of NVMe at $[Z]/TB/month. Move cold data to S3 ($23/TB/month) or Glacier ($4/TB/month) to reduce costs."

**Trigger condition 2 (GPU starvation):** `feeding_tax > 50%`

**Visual change:** GPU utilization gauge enters red zone. Banner:
> "**GPU STARVATION -- Feeding Tax = [X]%.** The GPU is idle [X]% of the time. Upgrade from [current tier] to NVMe, or increase dataloader workers from [N] to [M]."

Both failure states are reversible by adjusting controls.

### Structured Reflection

Four-option multiple choice:

> "Moving 1 PB costs $90,000+ in egress fees and takes 23 hours. A colleague suggests 'just copy the data to the cluster.' Why is this the wrong instinct at petabyte scale?"

- A) The data would be stale by the time it arrives
- B) The receiving cluster does not have enough storage
- **C) Data gravity: at petabyte scale, the physical and economic cost of moving data exceeds the cost of moving compute. Code moves to data, not data to code** <-- correct
- D) Cloud providers do not allow cross-region transfers of that size

**Math Peek:**
$$T_{\text{transfer}} = \frac{D_{\text{vol}}}{BW_{\text{network}}} = \frac{10^6 \text{ GB}}{12.5 \text{ GB/s}} = 80{,}000 \text{ s} \approx 22 \text{ hours}$$
$$\text{Cost}_{\text{egress}} = D_{\text{vol}} \times \text{rate} = 10^6 \text{ GB} \times \$0.09/\text{GB} = \$90{,}000$$
$$\text{Rule: If } T_{\text{transfer}} > T_{\text{training}}, \text{ move compute to data.}$$

---

## 5. Visual Layout Specification

### Act 1: Pipeline Utilization
- **Primary:** Circular gauge (Feeding Tax)
  - Range: 0--100% idle
  - Zones: green (0--20%), yellow (20--50%), red (50--100%)
- **Secondary:** Horizontal stacked bar (Pipeline Waterfall)
  - X-axis: Time (ms per training step)
  - Segments: Disk Read (RedLine), Preprocessing (OrangeLine), GPU Compute (BlueLine), GPU Idle (gray hatched)
- **Tertiary:** Throughput counter (images/sec achieved vs. peak)

### Act 2: Data Gravity & Storage Design
- **Primary:** Scatter plot (Transfer Time vs. Cost)
  - X-axis: Transfer time (hours), range 0--200
  - Y-axis: Cost (USD), range 0--$150,000
  - Points: labeled by transfer method
  - Crossover annotation for "move compute" threshold
- **Secondary:** Pipeline diagram with bandwidth pipes
  - Each stage labeled with bandwidth
  - Narrowest pipe highlighted in RedLine (bottleneck)
  - Feeding Tax gauge updates in real-time
- **Failure states:** Budget counter turns RedLine; GPU gauge enters red zone

---

## 6. Deployment Context Definitions

| Context | Device | RAM | Storage BW | Key Constraint |
|---|---|---|---|---|
| **Cloud (H100)** | NVIDIA H100 | 80 GB HBM3 | 250 MB/s (gp3) to 5 GB/s (NVMe) | I/O bandwidth vs. accelerator demand; egress cost at scale; Feeding Tax dominates |
| **Edge (Jetson Orin NX)** | NVIDIA Jetson Orin NX | 16 GB LPDDR5 | 100 MB/s (SD) to 2 GB/s (NVMe) | Data must be local; network is for model/data updates, not streaming; storage capacity limited |

The two contexts demonstrate that data gravity operates differently at each scale. In the cloud, the challenge is feeding a hungry accelerator from remote storage. At the edge, the challenge is keeping local data fresh when the network is slow and expensive. Both contexts are governed by the same Iron Law data term ($D_{\text{vol}}/BW$), but the binding bandwidth differs: HBM bandwidth in cloud, storage/network bandwidth at edge.

---

## 7. Design Ledger Output

```json
{
  "chapter": 4,
  "context": "cloud | edge",
  "storage_tier_hot": "nvme | ssd | s3",
  "storage_tier_warm": "s3 | glacier",
  "dataloader_workers": 8,
  "feeding_tax_pct": 12,
  "storage_cost_monthly_usd": 280,
  "data_gravity_decision": "move_compute | move_data",
  "dataset_size_category": "gb | tb | pb"
}
```

The `feeding_tax_pct` and `storage_tier_hot` fields feed forward to:
- **Lab 08 (Training):** The storage tier and worker count initialize the data loading configuration for training throughput optimization
- **Lab 12 (Benchmarking):** The feeding tax measurement informs the profiling baseline for I/O-bound detection

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| Feeding Tax > 95% with standard cloud disk | `@sec-data-engineering-feeding-problem` | "the feeding tax can exceed [95+]%, meaning the GPU spends the majority of its time waiting for bits" |
| Required BW ~5.8 GB/s for ResNet-50 on A100 | `@sec-data-engineering-feeding-problem`; `FeedingProblem` LEGO cell | "To saturate a 300 TFLOPS processor, the pipeline must often sustain 5.8 GB/s transfer rates" |
| Standard cloud disk: 250 MB/s | `@sec-data-engineering-feeding-problem`; `FeedingProblem.disk_bw_mbs` | "the storage pipeline delivers only 250 MB/s" |
| A100 processes ~40,000 images/sec at peak | `@sec-data-engineering-feeding-problem`; `FeedingProblem.img_per_sec` | Computed from A100 FLOPS / ResNet50 FLOPs per image |
| 1 PB transfer: ~23 hours at 100 Gbps | `@sec-data-engineering-data-gravity-adcb`; `DataGravity` LEGO cell | "Transfer Time: 80,000 seconds ~ 23 hours" |
| 1 PB egress cost: > $90,000 | `@sec-data-engineering-data-gravity-adcb`; `DataGravity.transfer_cost` | "moving 1 PB costs USD 90,000+" |
| "Code moves to Data" at PB scale | `@sec-data-engineering-data-gravity-adcb`, Data Gravity callout | "For petabyte-scale data, Code moves to Data. For gigabyte-scale data, Data moves to Code." |
| Energy-Movement Invariant: DRAM access ~640x FP32 multiply | `@sec-data-engineering-physics-data-cdcb`, Energy-Movement Invariant callout | "moving a bit costs 100--1,000x more energy than computing on it" |
| NVMe bandwidth: ~5 GB/s | `@sec-data-engineering-ml-storage-systems-architecture-options-67fa`; `StorageLoadCalc.nvme_bw_gbs = 5` | "NVMe: 5 GB/s effective throughput" |
| S3 cost: $23/TB/month | `@sec-data-engineering`; `DataEngineeringSetup.storage_cost_s3` | From `constants.py`: `STORAGE_COST_S3_STD` |
| NVMe cost: $70--140/TB/month | `@sec-data-engineering`; `DataEngineeringSetup.storage_cost_nvme_low/high` | From `constants.py`: `STORAGE_COST_NVME_LOW/HIGH` |
| Glacier cost: $4/TB/month | `@sec-data-engineering`; `DataEngineeringSetup.storage_cost_glacier` | From `constants.py`: `STORAGE_COST_GLACIER` |
| 8 dataloader workers for ResNet-50 | `@sec-data-engineering`; `DataloaderStats.resnet_worker_count = 8` | "8 workers" as canonical ResNet-50 configuration |
| Data prep consumes 60--80% of ML project time | `@sec-data-engineering-data-engineering-dataset-compilation-0496` | "data work consumes 60 to 80 percent of ML project effort" |
