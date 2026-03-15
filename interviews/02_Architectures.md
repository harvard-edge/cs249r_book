# AI Models & Architectures 🏗️

Systems implications of Transformer, CNN, and Recurrent architectures.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/02_Architectures.md)** (Edit in Browser)

### 📋 Copy-Paste Template
```markdown
<details>
<summary><b>ARCH: [Your Question Title Here]</b></summary>

**Answer:** [Direct answer here]
**Why:** [Explain the physics/logic here]
**Deep Dive:** [Optional: Maintainers will add link to Chapter/Wall]
</details>
```

---

<details>
<summary><b>KV-CACHE: Why do we need PagedAttention for large deployments?</b></summary>

**Answer:** To solve internal and external memory fragmentation (**Wall 5: Batching**).
**Why:** Standard Transformers allocate a contiguous block of VRAM for the KV-cache based on the *maximum* context length. This wastes ~60-80% of memory for shorter requests. PagedAttention (vLLM) allocates memory in small, non-contiguous "pages," allowing near-zero fragmentation and 2-3x higher throughput.
**📖 Deep Dive:** [Volume I: Frameworks](https://mlsysbook.ai/vol1/frameworks.html) (**Wall 5: Batching**)
</details>

<details>
<summary><b>CNN vs MLP: Why did CNNs enable the Deep Learning revolution?</b></summary>

**Answer:** Weight reuse and the **Data Term** reduction.
**Why:** A CNN applies the same small filter across the entire image (spatial weight reuse), whereas an MLP requires a unique weight for every connection. This drastically reduces the **Data Volume** ($D_{vol}$), allowing large models to fit in limited HBM and stay compute-bound (**Wall 1**) rather than memory-bound (**Wall 2**).
**📖 Deep Dive:** [Volume I: Network Architectures](https://mlsysbook.ai/vol1/dnn_architectures.html) (**Wall 6: Architectures**)
</details>

<details>
<summary><b>FIDELITY: Does 4-bit quantization (INT4) always increase throughput?</b></summary>

**Answer:** Only if the workload is **Memory Bound**.
**Why:** Quantization reduces the **Data Term** (**Wall 13: Fidelity**). If your model is already compute-bound (e.g., small weights, huge batch size), moving to 4-bit may only yield accuracy loss without any speedup unless the hardware has specialized INT4 Tensor Cores.
**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html) (**Wall 13: Fidelity**)
</details>
