# Volume 1 Audit: Comprehensive Review Notes

## Part I: Understand
**Files:** `intro.qmd`, `ml_systems.qmd`, `data_engineering.qmd`

### Chapter 1: Introduction
*   **Quantitative Rigor:**
    *   *Action:* Formalize the "AI Triad" (Data, Algorithms, Systems). Define the fundamental metric: **Samples per Dollar**. Show how this metric forces a trade-off between model accuracy and infrastructure cost.
*   **Systems Thinking:**
    *   *Action:* Contrast "Software 1.0" (Explicit Logic) vs "Software 2.0" (Learned Logic). Quantify the "Verification Gap"—show that 100% test coverage in ML is impossible because the input space is continuous and non-deterministic.

### Chapter 2: ML Systems
*   **Systems Thinking:**
    *   *Action:* Add a sidebar on **The Distributed Systems Tax**. Explain that moving from 1 GPU to 8 GPUs introduces communication overhead (All-Reduce) that typically consumes 10-20% of the compute budget.

### Chapter 3: Data Engineering
*   **Quantitative Rigor:**
    *   *Action:* Add a "Data Movement Energy" breakdown. Show that reading 1GB from S3 to a GPU costs more in energy and time than 1,000,000 matrix multiplications. This provides the Patterson-style physical justification for **Data Locality**.

---

## Part II: Build
**Files:** `dl_primer.qmd`, `dnn_architectures.qmd`, `frameworks.qmd`, `training.qmd`

### Chapter 5: DL Primer
*   **Quantitative Rigor:**
    *   *Action:* Add "Memory Cost of Backprop" sidebar. Show that storing activations requires $O(N \cdot L)$ memory (Batch x Layers), which is the primary driver for "Out of Memory" errors and the reason we need gradient checkpointing.
*   **Systems Thinking:**
    *   *Action:* Add a "Computational Intensity" table. Matrix Mul ($N^3$ ops, $N^2$ IO) vs Element-wise ($N$ ops, $N$ IO). Explain why GPUs are architecturally optimized for the former.

### Chapter 6: DNN Architectures
*   **Systems Thinking:**
    *   *Action:* Add "Dataflow Diagram" for a Transformer Layer. Visualize the Q, K, V projections and the Attention Score matrix size ($N^2$) to explain why sequence length scaling is the quadratic bottleneck of modern AI.

### Chapter 7: Frameworks
*   **Quantitative Rigor:**
    *   *Action:* Add "Framework Overhead" breakdown. Quantify Kernel Launch latency (~5-10us) vs Execution. Explain why "small kernels" (e.g., many element-wise ops) kill GPU utilization and necessitate Graph Capture/TorchCompile.

### Chapter 8: Training
*   **Quantitative Rigor:**
    *   *Action:* Add "Roofline Model" sidebar. Explain Arithmetic Intensity (FLOPs/Byte) and how it determines if a training job is Compute-Bound (Dense layers) or Memory-Bound (Attention/Norms).

---

## Part III: Optimize
**Files:** `optimize_principles.qmd`, `data_efficiency.qmd`, `model_compression.qmd`, `hw_acceleration.qmd`

### Chapter 9: Principles of Optimization
*   **Systems Thinking:**
    *   *Action:* Add **Amdahl's Law for ML**. Show that if 95% of a model is accelerated 100x but 5% (the "tails" like Python preprocessing) stays constant, the maximum total speedup is capped at 20x.

### Chapter 10: Data Efficiency
*   **Quantitative Rigor:**
    *   *Action:* Add a "Data Pruning Case Study" (e.g., SemDeDup). Show that removing 50% redundant data can maintain 99% accuracy while cutting training energy/cost by 2x.
*   **Systems Thinking:*
    *   *Action:* Quantify the **Selection Overhead**. $Cost_{total} = Cost_{train} + Cost_{select}$. Show that active learning only provides a net benefit if $Cost_{select} < Cost_{train\_saved}$.

### Chapter 11: Model Compression
*   **Quantitative Rigor:**
    *   *Action:* Add an "Energy per Op" table:
        *   FP32 Add: 0.9 pJ
        *   INT8 Add: 0.1 pJ
        *   DRAM Access: 640 pJ
    *   This justifies why quantization is a "Data Movement" optimization, not just a "Compute" optimization.

### Chapter 12: AI Acceleration
*   **Systems Thinking:**
    *   *Action:* Add a "Dataflow Trade-off" table. Compare **Weight-Stationary** (best for CNNs) vs. **Output-Stationary** (best for Large MatMuls). Explain how this physical design choice dictates which models run efficiently on a specific chip.

---

## Part IV: Deploy
**Files:** `deploy_principles.qmd`, `benchmarking.qmd`, `serving.qmd`, `ops.qmd`, `responsible_engr.qmd`

### Chapter 13: Principles of Deployment
*   **Systems Thinking:**
    *   *Action:* Add a sidebar on **The Reliability Multiplier**. Quantify aggregate reliability: $R_{sys} = \prod R_i$. Show that a compound system with 5 dependent models each at 99% accuracy has an aggregate theoretical reliability of ~95%.

### Chapter 14: Benchmarking AI
*   **Quantitative Rigor:**
    *   *Action:* Add a "Thermal Throttling" sidebar. Show how a thermal limit can drop GPU performance from 300W peak to 200W sustained, causing a 30% reduction in TFLOPS.
*   **Systems Thinking:**
    *   *Action:* Add a "Benchmark vs. Production Gap" analysis. Explain why synthetic peak FLOPS often overestimate production throughput by 2-10x by ignoring "Killer Microseconds" of system overhead.

### Chapter 15: Model Serving Systems
*   **Quantitative Rigor:**
    *   *Action:* Quantify the **Memory Bandwidth Bottleneck** for LLM Decode phase. Formula: $T_{token} = \frac{\text{Model Size (GB)}}{\text{Memory Bandwidth (GB/s)}}$. Show that for a 7B model at FP16 (14GB), a 1TB/s bandwidth limits generation to 70 tokens/sec max.

### Chapter 16: MLOps
*   **Systems Thinking:**
    *   *Action:* Add a "Monitoring Trade-off" model. Compare the cost of high-frequency telemetry (1s resolution) vs. storage/compute costs.
*   **Quantitative Rigor:**
    *   *Action:* Add a "Retraining Economics" worked example. Show that if retraining costs $5k but prevents $10k in "staleness loss," the optimal retraining frequency is derived from the square root of the cost/value ratio.

### Chapter 17: Responsible Engineering
*   **Quantitative Rigor:**
    *   *Action:* Add a "Carbon Intensity" column to the TCO summary. Compare 3-year emissions for a project running in a coal-heavy region (e.g., 500g CO2e/kWh) vs. a renewable-heavy region (50g CO2e/kWh).

---

## Conclusion
*   **Synthesis:**
    *   *Action:* Ensure the "New Golden Age" callout explicitly maps the **AI Triad** (Data, Algorithms, Infrastructure) to the **6 Core Principles**. This provides the final unifying framework for the entire volume.