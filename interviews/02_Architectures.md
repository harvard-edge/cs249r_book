# AI Models & Architectures 🏗️

Systems implications of Transformer, CNN, and dense architectures.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/02_Architectures.md)** (Edit in Browser)

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
**Realistic Solution:** An MLP treats every pixel independently, requiring a unique weight for every connection. A CNN uses a $3\times3$ filter applied across the whole image. For a $224\times224$ image, this reduces the parameter count (and thus the memory footprint, $D_{vol}$) by roughly $1,000\times$, making the model both computationally feasible and easier to train.
**📖 Deep Dive:** [Volume I: Network Architectures](https://mlsysbook.ai/vol1/dnn_architectures.html)
</details>

<details>
<summary><b>🟡 LEVEL 2: Why are CNNs usually 'Compute-Bound' while MLPs are 'Memory-Bound'?</b></summary>

**Answer:** Spatial weight reuse.
**Realistic Solution:** It comes down to Arithmetic Intensity ($I$). In a CNN, the same weight is used across the entire spatial dimension of the image, resulting in $50-200+$ FLOPs per byte loaded from memory. The GPU's ALUs are kept busy. In an MLP, a unique weight is loaded for a single multiply-accumulate (MAC) operation, yielding $\approx 1$ FLOP/byte, which starves the GPU while waiting for memory.
**📖 Deep Dive:** [Volume I: Network Architectures](https://mlsysbook.ai/vol1/dnn_architectures.html)
</details>

<details>
<summary><b>🔴 LEVEL 3: If MLPs are Universal Approximators, why did we invent specialized architectures like ResNets?</b></summary>

**Answer:** The 'Learnability Gap' (Representation Capacity $\neq$ Learnability).
**Realistic Solution:** The Universal Approximation Theorem (UAT) guarantees an MLP *can* represent any function, but its sample complexity scales exponentially ($O(\exp(d))$). Without structural priors, gradient descent cannot find the optimal weights given finite data and compute. Architectures like ResNet embed inductive biases that match the data's manifold, restricting the search space so the model actually converges.
**📖 Deep Dive:** [Volume I: Network Architectures](https://mlsysbook.ai/vol1/dnn_architectures.html)
</details>
