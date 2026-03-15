# Data Pipelines & Ingestion 📦

Bottlenecks in moving data from storage to the accelerator.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/03_Data_Pipelines.md)** (Edit in Browser)

### 📋 Copy-Paste Template
```markdown
<details>
<summary><b>DATA: [Your Question Title Here]</b></summary>

**Answer:** [Direct answer here]
**Realistic Solution:** [Explain the physics/logic here]
**📖 Deep Dive:** [Optional: Link to Volume I or II Chapter]
</details>
```

---

<details>
<summary><b>STARVATION: Why is my GPU at low utilization during training?</b></summary>

**Answer:** CPU Preprocessing starvation.
**Realistic Solution:** Modern GPUs are so fast that the CPU often cannot keep up with JPEG decoding, random cropping, and data augmentation. If the CPU takes 20ms to prepare a batch and the GPU takes 5ms to train on it, the GPU sits idle 75% of the time. You must increase CPU workers or use GPU-accelerated preprocessing (DALI).
**📖 Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)
</details>

<details>
<summary><b>INGESTION: How many tokens can a single node ingest from storage?</b></summary>

**Answer:** Limited by the bisection between storage bandwidth and PCIe bandwidth.
**Realistic Solution:** To maintain high Model FLOPS Utilization (MFU), your storage system must deliver data at the rate the accelerator consumes it. For high-resolution data (video), storage bandwidth often becomes the primary bottleneck before GPU compute is saturated.
**📖 Deep Dive:** [Volume I: Data Efficiency](https://mlsysbook.ai/vol1/data_selection.html)
</details>
