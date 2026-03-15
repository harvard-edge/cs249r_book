# Data Pipelines & Ingestion 📦

Bottlenecks in moving data from storage to the accelerator.

---

<details>
<summary><b>UTILIZATION: Why is my GPU idle during training?</b></summary>

**Answer:** Storage I/O or CPU preprocessing starvation.
**Explanation:** If CPU JPEG decoding or data augmentation takes longer than the GPU training step, the GPU remains idle. Solutions include increasing CPU workers or using GPU-accelerated libraries.
</details>

<details>
<summary><b>INGESTION: What limits token ingestion speed from storage?</b></summary>

**Answer:** Storage throughput and PCIe/Network bandwidth.
**Explanation:** For high-resolution data (e.g., video), storage bandwidth often becomes the primary bottleneck before GPU compute is saturated.
</details>
