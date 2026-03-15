# AI Systems Foundations & Hardware 🧱

Fundamental calculations for VRAM, Peak Performance, and Memory Bandwidth.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/01_Foundations.md)** (Edit in Browser)

### 📋 Copy-Paste Template
```markdown
<details>
<summary><b>TOPIC: [Your Question Title Here]</b></summary>

**Answer:** [Direct answer here]
**Realistic Solution:** [Explain the physics/logic here]
**📖 Deep Dive:** [Optional: Link to Volume I or II Chapter]
</details>
```

---

<details>
<summary><b>VRAM: How much memory is needed for FP16 inference?</b></summary>

**Answer:** $2 \times \text{Parameters}$ (GB) + **KV-Cache**.
**Realistic Solution:** Each parameter at FP16 precision takes 2 bytes. A 70B model requires 140GB just to fit the weights. Crucially, the **KV-Cache** for long context windows can exceed the weight size itself. Always discuss the sequence length impact to prove seniority.
**📖 Deep Dive:** [Volume I: ML Systems](https://mlsysbook.ai/vol1/ml_systems.html)
</details>

<details>
<summary><b>BOTTLENECK: Is a model Compute-Bound or Memory-Bound?</b></summary>

**Answer:** Calculate Arithmetic Intensity ($I = \text{Ops} / \text{Bytes}$) and compare to the Ridge Point ($I^* = \text{Peak FLOPS} / \text{Bandwidth}$).
**Realistic Solution:** If $I < I^*$, the system is **Memory Bound**. Faster arithmetic (more TFLOPS) will yield no speedup. If an interviewer suggests "upgrading the GPU" for slow LLM decoding, a senior engineer knows this only helps if the memory bandwidth ($BW$) is higher.
**📖 Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

<details>
<summary><b>DUALITY: What are the two distinct phases of LLM inference?</b></summary>

**Answer:** Prefill and Decode.
**Realistic Solution:** **Prefill** processes the prompt and is **Compute Bound**. **Decode** generates tokens one-by-one and is **Memory Bound**. Senior architects often decouple these into separate clusters (Disaggregated Serving) to optimize for both constraints.
**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/serving.html)
</details>
