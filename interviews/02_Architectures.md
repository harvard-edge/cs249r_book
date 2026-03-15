# AI Models & Architectures 🏗️

Systems implications of Transformer, CNN, and Recurrent architectures.

---

<details>
<summary><b>KV-CACHE: Why is PagedAttention used for large-scale serving?</b></summary>

**Answer:** To solve memory fragmentation.
**Explanation:** Standard Transformers allocate contiguous VRAM for the KV-cache based on maximum sequence length, wasting memory. PagedAttention allocates non-contiguous "pages," reducing fragmentation and increasing throughput.
</details>

<details>
<summary><b>CNN vs MLP: Why are CNNs efficient for image processing?</b></summary>

**Answer:** Weight reuse and local connectivity.
**Explanation:** CNNs use small filters across the entire image (high reuse), whereas MLPs require unique weights for every connection. This reduces data volume and keeps CNNs compute-bound.
</details>

<details>
<summary><b>QUANTIZATION: Does 4-bit (INT4) always increase throughput?</b></summary>

**Answer:** Only if the workload is Memory Bound.
**Explanation:** Quantization reduces data movement. If a model is already compute-bound (e.g., large batch size), INT4 may not provide speedup without specialized hardware support.
</details>
