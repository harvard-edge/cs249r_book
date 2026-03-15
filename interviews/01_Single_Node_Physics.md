# Round 1: Single-Node Systems & Silicon Physics 🧱

The domain of the ML Systems Engineer. This round tests your understanding of what happens *inside* a single server chassis: memory hierarchies, compute bounds, and arithmetic intensity.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/01_Single_Node_Physics.md)** (Edit in Browser)

### 📋 Copy-Paste Template
```markdown
<details>
<summary><b>[LEVEL]: [Your Question Title Here]</b></summary>

**Answer:** [Direct answer here]
**Realistic Solution:** [Explain the physics/logic here]
**📖 Deep Dive:** [Optional: Link to Volume I or II Chapter]
</details>
```

---

<details>
<summary><b>🟢 LEVEL 1: What is 'Inductive Bias' and why do we use CNNs instead of MLPs for images?</b></summary>

**Answer:** Structural constraints that restrict the hypothesis space (e.g., spatial locality).
**Realistic Solution:** An MLP treats every pixel independently, requiring a unique weight for every connection. A CNN uses a local filter applied across the whole image. For a $224	imes224$ image, this reduces the parameter count (and thus the memory footprint, $D_{vol}$) by roughly $1,000	imes$, making the model computationally feasible.
**📖 Deep Dive:** [Volume I: Network Architectures](https://mlsysbook.ai/vol1/dnn_architectures.html)
</details>

<details>
<summary><b>🟡 LEVEL 2: Why are CNNs usually 'Compute-Bound' while MLPs are 'Memory-Bound'?</b></summary>

**Answer:** Spatial weight reuse changes the Arithmetic Intensity.
**Realistic Solution:** It comes down to Arithmetic Intensity ($I$). In a CNN, the same weight is used across the entire spatial dimension of the image, resulting in $50-200+$ FLOPs per byte loaded from memory. The GPU's ALUs are kept busy. In an MLP, a unique weight is loaded for a single multiply-accumulate (MAC) operation, yielding $\approx 1$ FLOP/byte, which starves the GPU while waiting for memory.
**📖 Deep Dive:** [Volume I: Network Architectures](https://mlsysbook.ai/vol1/dnn_architectures.html)
</details>

<details>
<summary><b>🔴 LEVEL 3: How does the "Transformation Wall" (CPU Starvation) limit GPU utilization during training?</b></summary>

**Answer:** The storage I/O or CPU preprocessing pipeline fails to feed the GPU fast enough.
**Realistic Solution:** Modern GPUs are so fast that the CPU often cannot keep up with JPEG decoding, random cropping, and data augmentation. If the CPU takes 20ms to prepare a batch and the GPU takes 5ms to train on it, the GPU sits idle 75% of the time. To fix this, a systems engineer must increase CPU worker threads or offload preprocessing to the GPU (e.g., using NVIDIA DALI).
**📖 Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)
</details>
