# Mission Plan: lab_04_data_engr

## 1. Chapter Alignment

- **Chapter:** Data Engineering (`@sec-data-engineering`)
- **Core Invariant:** The **Energy-Movement Invariant** — moving a bit costs 100–1,000× more energy than computing on it. The dominant cost in any data pipeline is not computation; it is data movement. Every architectural decision (format, location, serialization, caching) is a decision about how far bits must travel.
- **Central Tension:** Students believe that data engineering is about *data quality* — cleaning, labeling, and curation. The chapter's central claim is that data engineering is about *physics*: a DRAM access costs 640 pJ vs. 3.7 pJ for a FP32 operation. Moving data is 173× more expensive than using it. Students expect switching from JSON to Parquet to be a minor convenience; the chapter shows it is mathematically equivalent to buying a 5× faster hard drive.
- **Target Duration:** 35–40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students believe the choice of serialization format is a developer-convenience decision. This act shows it is an energy and throughput decision with a 10× gap between JSON and Parquet for identical data. The prediction question targets the specific wrong prior: that the GPU is the bottleneck in training, when in fact the disk I/O pipeline is often the binding constraint.

**Act 2 (Design Challenge, 22 min):** Students extend the energy analysis to a full system design. They must build a data pipeline that feeds a 300 TFLOPS accelerator without starvation — the "Feeding Tax." Then they confront Data Cascades: a schema change (zip code from integer to string) silently corrupts a model six months into training. The cascade takes a median 4 weeks to detect. Students diagnose the root cause using the Four Pillars Framework.

---

## 3. Act 1: The Format Tax (Calibration — 12 minutes)

### Pedagogical Goal
Students believe serialization format is a tooling choice, not a physics choice. The chapter's claim is quantitative: Parquet with columnar projection on 20 of 100 features delivers 5× the I/O throughput of CSV for the same query — not because Parquet is "better engineered," but because it reads 80% fewer bytes from disk. This act forces students to predict the throughput ratio, then confront the energy table showing that disk access costs 10,000× more per bit than a FP32 computation.

### The Lock (Structured Prediction)
Present a **multiple-choice prediction** before any instruments unlock:

> "A fraud detection model uses 20 of 100 available features. You switch the training dataset from CSV to Parquet with columnar projection. How much faster is the data ingestion step?"

Options:
- A) About the same — format doesn't affect read speed for modern NVMe drives
- B) About 2× faster — Parquet has better compression
- **C) About 5× faster — Parquet reads only the 20 needed columns, skipping the other 80%** ← correct
- D) About 50× faster — Parquet uses SIMD hardware acceleration for decoding

The correct answer is mechanical: 20 columns / 100 columns = 20% of bytes read = 5× fewer I/O operations.

### The Instrument: Format Efficiency Comparator

A side-by-side panel comparing **CSV, JSON, Parquet** on four metrics, for the fraud detection scenario (100 columns, 20 features, 100M records):

| Metric | CSV | JSON | Parquet |
|---|---|---|---|
| **Bytes read per query** | 100% (all columns) | 100% + overhead | 20% (columnar projection) |
| **Ingestion speed (relative)** | 1× | 0.1× (10× slower) | 5× |
| **I/O energy (pJ per record)** | ~10,000 pJ (SSD) | ~10,000 pJ + parsing | ~2,000 pJ |
| **GPU idle time at 300 TFLOPS** | computed live | computed live | computed live |

Controls:
- **Feature selection slider**: 1–100 features used (out of 100 total). Pareto throughput updates live: Parquet throughput = total throughput × (features_selected / 100). CSV stays flat.
- **Dataset size selector**: 1 GB / 100 GB / 1 TB / 100 TB. The "Transfer Time" annotation updates to show wall-clock impact.
- **Storage type toggle**: Cloud HDD (250 MB/s) / NVMe SSD (3 GB/s) / RAM cache (50 GB/s).

A **GPU Idle Gauge** shows: at 300 TFLOPS and a given ingestion rate, what fraction of time is the GPU starved?
- CSV at 250 MB/s disk → GPU idle > 80%
- Parquet at 250 MB/s disk → GPU idle drops to ~16% (5× fewer bytes → 5× more data per second)

### The Reveal
After exploration:
> "You predicted [X]× speedup. The actual ratio is **5×** for this workload (20 of 100 features). Note that this assumes identical storage hardware — the speedup is purely from reading fewer bytes, not from better hardware."

Surface the energy table from the chapter:
> "Why does this matter beyond throughput? A FP32 multiply costs **3.7 pJ**. Reading that data from SSD costs **~10,000 pJ per bit**. Moving data is **2,700× more expensive** than computing on it. The format choice is an energy decision."

### Reflection (Structured)
Sentence completion with dropdown:

> "Switching from CSV to Parquet is equivalent to buying a 5× faster hard drive because ___. This is an example of the [Data Gravity / Energy-Movement / Feeding Tax] invariant."

Dropdown for blank 1:
- **"columnar projection reads only the needed features, reducing bytes transferred by 80%"** ← correct
- "Parquet uses hardware acceleration not available to CSV readers"
- "Parquet compresses all columns before writing to disk"
- "Parquet eliminates the need for data preprocessing"

Dropdown for blank 2: **"Energy-Movement"** ← correct (the 10,000 pJ/bit SSD cost vs. 3.7 pJ/bit compute cost is the invariant).

**Math Peek (collapsible):**
$$\text{Feeding Tax} = 1 - \frac{\text{Pipeline Throughput}}{\text{GPU Consumption Rate}}$$
$$E_{move} \gg E_{comp}: \quad 640 \text{ pJ/DRAM access} \gg 3.7 \text{ pJ/FLOP}$$

---

## 4. Act 2: The Data Cascade (Design Challenge — 22 minutes)

### Pedagogical Goal
Students believe data pipelines fail loudly — corrupt data should produce immediate training errors. The chapter's Data Cascade case study shows the opposite: a schema change (zip_code: integer → string) silently corrupts downstream features because the pipeline casts "02139" to 2139, losing the leading zero. The model trains normally and hits 95% accuracy — but on the wrong distribution. Detection takes a median 4 weeks. Students must use the Four Pillars Framework (Quality, Reliability, Scalability, Governance) to identify which pillar was violated and what gate would have caught it.

### The Lock (Numeric Prediction)
Before instruments unlock:

> "A data pipeline schema change is introduced on Day 1. The model trains for 90 days and achieves 95% accuracy before a silent corruption is discovered. How many weeks does the chapter's research show is the median time-to-detection for Data Cascade issues?"

Students type a number (weeks). Expected wrong answers: 1–2 weeks (assuming fast CI/CD feedback). Actual answer: **4 weeks**.

### The Instrument: The Four Pillars Diagnostic

A **two-panel instrument**:

**Panel A: Pipeline Budget Solver**
The student must configure a data pipeline that satisfies the Feeding Tax inequality for a 300 TFLOPS accelerator:
$$\text{Pipeline Throughput} \geq \text{GPU Consumption Rate}$$

Controls — **exactly two** (to keep Act 2 within 22 minutes):
- **Serialization format** (CSV / JSON / Parquet): Sets base throughput multiplier (1× / 0.1× / 5×). This is the highest-leverage single decision — a format change delivers 5× without touching hardware.
- **Storage type** (Cloud HDD 250 MB/s / NVMe SSD 3 GB/s): Multiplies the format-adjusted throughput. Students discover that Parquet on a cloud HDD outperforms CSV on NVMe.

Output: A live bar showing GPU consumption rate (fixed, red line) vs. pipeline delivery rate (green bar). When the green bar exceeds the red line, the **"Flow Equilibrium" badge** appears.

> **Why only 2 controls?** Prefetch workers and preprocessing location are real levers, but they require understanding of CPU parallelism not yet introduced. They are extensions available after Flow Equilibrium is achieved. The core lesson is format × storage = the two decisions that require no infrastructure changes.

**Failure state:** When pipeline throughput < GPU consumption rate:
> "🟠 **Feeding Tax Active.** GPU is idle [X]% of the time. Effective MFU = [Y]% instead of target [Z]%."

**Panel B: Data Cascade Autopsy**
A timeline visualization of the zip_code cascade from the chapter:

```
Day 1:   Schema change: zip_code integer → string (no validation gate)
Day 1:   Pipeline silently casts "02139" → 2139 (leading zero lost)
Day 1:   Model sees zip_code 2139 as unknown → defaults to high-risk label
Day 90:  Training complete. Global accuracy: 95%
Day 94:  Field audit: "Why is zip code 02139 (Cambridge, MA) flagged as high-risk?"
Day 98:  Root cause traced to Day 1 schema change
```

Students select **which of the Four Pillars was violated**:
- **Quality** — data values were incorrect after the transform
- **Reliability** — the pipeline had no schema validation gate
- **Scalability** — the dataset grew too large to validate manually
- **Governance** — no audit trail linked the schema change to downstream impact

All four are technically valid — the chapter's point is that all four pillars interact. But **Reliability** is the primary failure: a schema validation gate (Pillar 2) would have caught the type mismatch on Day 1 rather than Day 98.

Students then configure a **Schema Validation Gate** and see the timeline truncate at Day 1 with: "Validation error: zip_code expected string, received integer. Pipeline halted."

### The Scaling Challenge
**"Configure a pipeline where the Data Cascade is caught on Day 1, not Day 98."**

Using the Schema Validation Gate from Panel B, students configure a complete validation policy:
- **Schema contract** (type checking on all fields) — catches zip_code integer→string on Day 1
- **Distribution check** (statistical test comparing today's feature distribution to baseline) — catches drift introduced by the type change
- **Lineage tag** (links each training batch to the pipeline version that produced it) — enables root-cause in hours instead of weeks

Students toggle each gate On/Off and see the cascade timeline change:
- No gates: Day 98 discovery
- Schema contract only: Day 1 discovery (type mismatch caught immediately)
- All three gates: Day 1 discovery + root cause traceable in < 1 hour

The structured question: "Schema validation catches *type* failures. What class of failure does distribution checking catch that schema validation misses?"
Expected answer: silent semantic corruption where the type is correct but the values have changed distribution (e.g., a feature that was normalized to [0,1] is now [0, 100] — same type, wrong range).

**Optional extension (for fast students):** The Data Gravity Crossover.
$$T_{transfer} = \frac{D_{vol}}{BW} = \frac{10^{15} \text{ bytes}}{12.5 \text{ GB/s}} \approx 9.3 \text{ days}$$
At what dataset size does physical shipment beat 100 Gbps streaming? This is the "Sneakernet Crossover" from the chapter. Answer: ~10–50 TB for intercontinental distances. Available as a toggle after the main scaling challenge is complete.

### Structured Reflection
Students select the correct statement:

> "The Data Cascade took 4 weeks to detect because:"
- A) The team was not monitoring the training logs
- B) The model accuracy appeared normal (95%) even on corrupted data
- **C) Silent corruption produced plausible outputs — the model trained successfully on wrong labels, with no error message** ← correct
- D) The schema change was made in a different team's repository with no notification

Then complete the sentence:
> "The Four Pillars Framework categorizes the primary failure as a [Quality / Reliability / Scalability / Governance] failure, because ___."

Expected answer: **Reliability** — "no schema validation gate existed to catch the type mismatch at the pipeline boundary."

**Math Peek:**
$$\text{Data Selection Gain} \propto \frac{\text{Information Entropy}}{\text{Data Gravity}} \qquad T_{gravity} = \frac{D_{vol}}{BW}$$

---

## 5. Visual Layout Specification

### Act 1: Format Tax
- **Primary:** Side-by-side stat cards (CSV / JSON / Parquet) — each showing bytes read, throughput multiplier, energy cost, GPU idle %
- **Secondary:** Feature selection curve — X: features selected (1–100), Y: relative throughput. Two lines: CSV (flat) vs. Parquet (linear improvement with fewer features). Intersection at 100 features (where both are equal).
- **Prediction overlay:** Student's choice highlighted; correct answer annotated with "5× = 20/100 features"

### Act 2: Data Cascade
- **Primary Panel A:** Pipeline Budget bar — GPU consumption rate (red, fixed) vs. pipeline delivery rate (green, controllable). Two controls: format selector + storage type toggle. Flow Equilibrium badge appears when green > red.
- **Primary Panel B:** Cascade timeline — Day 1 through Day 98, with schema change, silent corruption, and detection markers. Three gate toggles (schema contract, distribution check, lineage tag) truncate the timeline progressively.
- **Optional Extension Panel:** Data Gravity crossover calculator — D_vol slider, BW slider, output: streaming time vs. physical shipment crossover (visible only after Flow Equilibrium is achieved).
- **Failure states:** OrangeLine feeding tax banner when pipeline throughput < GPU consumption rate.

---

## 6. Deployment Context Definitions

| Context | Device | Storage | Key Constraint |
|---|---|---|---|
| **Training Node** | H100 (80 GB) + NVMe RAID | 3 GB/s read | Pipeline must sustain ~2 GB/s sustained to avoid Feeding Tax at batch=256 |
| **Edge Inference** | Mobile NPU (2 GB) + eMMC | 300 MB/s read | Quantized model weights must fit in 2 GB; data preprocessing must occur on-device |

The two contexts share the same Energy-Movement Invariant but at different scales: the training node confronts the Feeding Tax (disk → GPU bandwidth gap), while the edge device confronts the Data Gravity problem (any OTA update is limited by 5G energy budget: ~50,000 pJ per network transfer vs. ~3.7 pJ per FLOP).

---

## 7. Design Ledger Output

```json
{
  "chapter": 4,
  "serialization_format": "parquet | csv | json | protobuf",
  "storage_type": "nvme | cloud_hdd | ram_cache",
  "feeding_tax_eliminated": true,
  "cascade_pillar_identified": "reliability",
  "pipeline_throughput_gbps": 2.4,
  "validation_gates_active": ["schema_contract", "distribution_check", "lineage_tag"],
  "cascade_detection_day": 1
}
```

The `serialization_format` and `feeding_tax_eliminated` fields feed forward to:
- **Lab 08 (Training):** Pipeline throughput becomes the data loading baseline in the MFU analysis
- **Lab 14 (ML Ops):** The cascade diagnosis becomes the baseline for drift detection discussion

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| 3.7 pJ per FP32 FLOP | @sec-data-engineering-feeding-problem, energy table | "32-bit Floating Point MAC: 3.7 pJ, 1×" |
| 640 pJ per DRAM access | @sec-data-engineering-feeding-problem, energy table | "DRAM Memory Access (32-bit): 640 pJ" |
| ~10,000 pJ per SSD access | @sec-data-engineering-feeding-problem, energy table | "Local SSD Access (per bit): ~10,000 pJ" |
| 100–1,000× energy for movement | @sec-data-engineering-feeding-problem | "moving a bit costs 100–1,000× more energy than computing on it" |
| Parquet = 5× faster (20/100 features) | @sec-data-engineering-ml-storage-systems-architecture-options-67fa | "Switching from CSV to Parquet is…equivalent to buying a 5× faster hard drive" |
| JSON = 10× slower than Parquet | @sec-data-engineering-ingestion-data-sources-formats-protocols-4c59, fn-json-ml-overhead | "ingestion over 10x slower than with formats like Protobuf" |
| 5–10× I/O reduction (columnar) | @sec-data-engineering-ml-storage-systems-architecture-options-67fa | "Columnar storage formats…deliver this five to 10 times I/O reduction" |
| 20–100× with compression | @sec-data-engineering-ml-storage-systems-architecture-options-67fa | "total I/O reduction of 20 to 100 times compared to uncompressed row formats" |
| 4 weeks median cascade detection | @sec-data-engineering-data-cascades-systematic-foundations-matter-2efe | "cascade issues take a median of 4 weeks to discover after introduction" |
| zip_code cascade example | Lines 593–600 | Schema change integer→string; "02139"→2139; leading zero lost; model trained normally |
| Four Pillars: Quality, Reliability, Scalability, Governance | @sec-data-engineering-four-pillars-framework-4ef1 | "The Four Pillars Framework organizes these concerns into four interdependent dimensions" |
| Cloud HDD baseline 250 MB/s | @sec-data-engineering-feeding-problem | "disk_bw_mbs = 250.0" (standard cloud disk, AWS gp3 baseline) |
| Feeding Tax formula | @sec-data-engineering-feeding-problem | "Feeding Tax: wall-clock time lost to I/O wait, which directly reduces System Efficiency (η)" |
| Data Gravity formula T = D_vol / BW | @sec-data-engineering-data-gravity-adcb | "The time to move a petabyte dataset across a 10 Gbps link is fixed by physics" |
