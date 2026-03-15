# Round 1: Single-Node Systems & Silicon Physics 🧱

The domain of the ML Systems Engineer. This round tests your understanding of what happens *inside* a single server chassis: memory hierarchies, compute bounds, and arithmetic intensity.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/01_Single_Node_Physics.md)** (Edit in Browser)

### 📋 Copy-Paste Template
```markdown
<details>
<summary><b>[LEVEL]: [Your Question Title Here]</b></summary>

**Prompt:** [The scenario or crisis]
**Realistic Solution:** [Explain the physics/logic here]
**📖 Deep Dive:** [Optional: Link to Volume I or II Chapter]
</details>
```

---

<details>
<summary><b>🟡 LEVEL 2: The Profiling Crisis (Arithmetic Intensity)</b></summary>

**Interviewer:** "You've deployed a custom recommendation model. The profiling dashboard shows you are achieving 120 TFLOPS out of a possible 300 TFLOPS on your GPU. Your tech lead suggests buying a faster GPU to fix the latency. Why is your tech lead wrong?"
**Realistic Solution:** The tech lead hasn't checked the Arithmetic Intensity ($Ops/Bytes$). If the model is memory-bound (intensity is lower than the ridge point of the roofline), a GPU with faster ALUs will do nothing. You must buy a GPU with higher *Memory Bandwidth (HBM)*, or optimize the model to move fewer bytes (e.g., quantization).
**📖 Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

<details>
<summary><b>🟢 LEVEL 1: The Data Pipeline Stall (The Transformation Wall)</b></summary>

**Interviewer:** "You are training a vision model on high-resolution medical images. `nvidia-smi` shows GPU utilization fluctuating violently between 0% and 100%. What is the most likely bottleneck in your node?"
**Realistic Solution:** CPU Starvation (The Transformation Wall). The GPU finishes its math instantly, then sits at 0% waiting for the CPU to decode, crop, and augment the next batch of JPEGs. You must offload preprocessing to the GPU (like NVIDIA DALI) or increase your CPU worker count.
**📖 Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)
</details>

<details>
<summary><b>🔴 LEVEL 3: The Precision Trade-off (Hardware Mapping)</b></summary>

**Interviewer:** "We are quantizing our model from FP16 to INT8. The memory footprint drops by 2x, but the throughput doesn't improve at all. Assuming we are compute-bound, what hardware architectural detail did we forget?"
**Realistic Solution:** You forgot to check if the silicon has dedicated INT8 Tensor Cores. Without specific hardware paths for 8-bit integer math, the GPU must upcast the INT8 values back to FP16 in the registers to perform the multiply-accumulate (MAC), yielding zero compute speedup despite the memory savings. 
**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

<details>
<summary><b>🟡 LEVEL 2: The Sequence Length Trap (VRAM Accounting)</b></summary>

**Interviewer:** "You need to increase your LLM's context window from 4k to 128k tokens. The model weights fit perfectly in your 80GB VRAM. What hidden memory cost will cause your node to OOM (Out of Memory) during generation?"
**Realistic Solution:** The KV-Cache. While weights are static, the KV-cache grows linearly with sequence length and batch size. At 128k context, the memory required to store the attention keys and values for a single request will massively exceed the size of the model weights themselves. 
**📖 Deep Dive:** [Volume I: Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

<details>
<summary><b>🟢 LEVEL 1: The Compilation Overhead (JIT vs AOT)</b></summary>

**Interviewer:** "You move a PyTorch training loop from a CPU to a GPU. The first few batches take 500ms each, but suddenly the latency drops to 10ms per batch. What happened inside the framework?"
**Realistic Solution:** Just-In-Time (JIT) compilation overhead. The framework spends the first few iterations tracing the computation graph and compiling optimized CUDA kernels for the specific tensor shapes you provided. Once cached, the dispatch overhead disappears. 
**📖 Deep Dive:** [Volume I: ML Systems](https://mlsysbook.ai/vol1/ml_systems.html)
</details>
