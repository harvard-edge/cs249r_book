# Data Pipelines & Ingestion 📦

Bottlenecks in moving data from storage to the accelerator.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/03_Data_Pipelines.md)** (Edit in Browser)

### 📋 Copy-Paste Template
```markdown
<details>
<summary><b>DATA: [Your Question Title Here]</b></summary>

**Answer:** [Direct answer here]
**Why:** [Explain the physics/logic here]
</details>
```

---

<details>
<summary><b>STARVATION: Why is my GPU at low utilization during training?</b></summary>

**Answer:** You are likely hitting **Wall 9 (Transformation)**.
**Why:** Modern GPUs are so fast that the CPU often cannot keep up with JPEG decoding, random cropping, and data augmentation. If the CPU takes 20ms to prepare a batch and the GPU takes 5ms to train on it, the GPU sits idle 75% of the time. You must increase CPU workers or use GPU-accelerated preprocessing (DALI).
</details>

<details>
<summary><b>INGESTION: How many tokens can a single node ingest from storage?</b></summary>

**Answer:** Limited by the bisection between storage bandwidth and PCIe bandwidth.
**Why:** To maintain high Model FLOPS Utilization (MFU), your storage system must deliver data at the rate the accelerator consumes it (**Wall 8: Ingestion**). If you are training a 70B model on text, text is small. But if you are training on high-res video, you can easily saturate a 100 Gbps network link before saturating the GPU cores.
</details>
