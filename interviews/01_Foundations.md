# AI Systems Foundations & Hardware 🧱

Fundamental calculations for VRAM, Peak Performance, and Memory Bandwidth.

---

<details>
<summary><b>VRAM: How much memory is needed for FP16 inference?</b></summary>

**Answer:** $2 \times \text{Parameters}$ (GB) + **KV-Cache**.
**Explanation:** Each parameter at FP16 precision takes 2 bytes. A 70B model requires 140GB just to fit the weights. For long context windows, the KV-cache can exceed the weight size itself.
</details>

<details>
<summary><b>BOTTLENECK: Is a model Compute-Bound or Memory-Bound?</b></summary>

**Answer:** Calculate Arithmetic Intensity ($I = \text{Ops} / \text{Bytes}$) and compare to the Ridge Point ($I^* = \text{Peak FLOPS} / \text{Bandwidth}$).
**Explanation:** If $I < I^*$, the system is **Memory Bound**. Faster arithmetic (more TFLOPS) will yield no speedup. 
</details>

<details>
<summary><b>DUALITY: What are the two distinct phases of LLM inference?</b></summary>

**Answer:** Prefill and Decode.
**Explanation:** **Prefill** processes the prompt and is **Compute Bound**. **Decode** generates tokens one-by-one and is **Memory Bound**. 
</details>
