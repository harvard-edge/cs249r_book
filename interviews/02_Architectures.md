# AI Models & Architectures 🏗️

Systems implications of Transformer, CNN, and Recurrent architectures.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/02_Architectures.md)** (Edit in Browser)

### 📋 Copy-Paste Template
```markdown
<details>
<summary><b>ARCH: [Your Question Title Here]</b></summary>

**Answer:** [Direct answer here]
**Realistic Solution:** [Explain the physics/logic here]
**📖 Deep Dive:** [Optional: Link to Volume I or II Chapter]
</details>
```

---

<details>
<summary><b>KV-CACHE: Why is PagedAttention used for large-scale serving?</b></summary>

**Answer:** To solve internal and external memory fragmentation.
**Realistic Solution:** Standard Transformers allocate a contiguous block of VRAM for the KV-cache based on the *maximum* context length. This wastes ~60-80% of memory for shorter requests. PagedAttention (vLLM) allocates memory in small, non-contiguous "pages," allowing near-zero fragmentation and 2-3x higher throughput.
**📖 Deep Dive:** [Volume I: Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

<details>
<summary><b>CNN vs MLP: Why did CNNs enable the Deep Learning revolution?</b></summary>

**Answer:** Weight reuse and local connectivity.
**Realistic Solution:** A CNN applies the same small filter across the entire image (spatial weight reuse), whereas an MLP requires a unique weight for every connection. This drastically reduces the **Data Volume** ($D_{vol}$), allowing large models to fit in limited HBM and stay compute-bound rather than memory-bound.
**📖 Deep Dive:** [Volume I: Network Architectures](https://mlsysbook.ai/vol1/dnn_architectures.html)
</details>

<details>
<summary><b>FIDELITY: Does 4-bit quantization (INT4) always increase throughput?</b></summary>

**Answer:** Only if the workload is **Memory Bound**.
**Realistic Solution:** Quantization reduces the bytes moved per token. If your model is already compute-bound (e.g., small weights, huge batch size), moving to 4-bit may only yield accuracy loss without any speedup unless the hardware has specialized INT4 Tensor Cores.
**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>
