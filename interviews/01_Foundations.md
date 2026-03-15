# AI Systems Foundations & Hardware 🧱

Fundamental calculations for VRAM, Peak Performance, and Memory Bandwidth.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/01_Foundations.md)** (Edit in Browser)

### 📋 Copy-Paste Template
```markdown
<details>
<summary><b>TOPIC: [Your Question Title Here]</b></summary>

**Answer:** [Direct answer here]
**Why:** [Explain the physics/logic here]
**LLM Reality Check:** [Optional: Where do LLMs get this wrong?]
</details>
```

---

<details>
<summary><b>VRAM: How much memory is needed for FP16 inference?</b></summary>

**Answer:** $2 \times \text{Parameters}$ (GB) + **KV-Cache**.
**Why:** Each parameter at FP16 precision takes 2 bytes. A 70B model requires 140GB just to fit the weights (**Wall 2: Memory**). 
**LLM Reality Check:** Most generic LLM answers forget the **KV-Cache**. For long context windows, the cache can exceed the weight size itself. To prove seniority, always discuss the sequence length impact.
</details>

<details>
<summary><b>BOTTLENECK: Is my model Compute-Bound or Memory-Bound?</b></summary>

**Answer:** Calculate Arithmetic Intensity ($I = \text{Ops} / \text{Bytes}$) and compare to the Ridge Point ($I^* = \text{Peak FLOPS} / \text{Bandwidth}$).  
**Why:** If $I < I^*$, the system is **Memory Bound** (Wall 2). Faster arithmetic (more TFLOPS) will yield **0% speedup**.
**LLM Reality Check:** LLMs often suggest "upgrading the GPU" for slow inference. A Silicon Realist knows that if you are bandwidth-bound (e.g., LLM decoding), an H100 won't help you much more than an A100 if the memory bandwidth is the same.
</details>

<details>
<summary><b>DUALITY: What are the two distinct phases of LLM inference?</b></summary>

**Answer:** Prefill and Decode.
**Why:** **Prefill** processes the prompt and is **Compute Bound** (**Wall 1**). **Decode** generates tokens one-by-one and is **Memory Bound** (**Wall 2**). This is **Wall 4 (Serving)**. Senior architects often decouple these into separate clusters (Disaggregated Serving) to optimize for both.
</details>
