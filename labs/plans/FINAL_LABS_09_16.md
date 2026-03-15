# Final Lab Plans: Volume I, Labs 09--16

Generated: 2026-03-15
Status: FINAL (post-review, all fixes applied from MASTER_LAB_PARTS_REVIEW and EDTECH_REVIEW_SUMMARY)

---

## Lab 09: The Data Selection Paradox
**Story arc**: Less data trains faster -- until the cost of *choosing* that data exceeds the savings, and meanwhile your GPU starves waiting for the CPU to finish preprocessing.
**Time budget**: 52 min total -- A(12) + B(12) + C(12) + D(10) + Synthesis(6)

### Part A -- The ICR Frontier: Diminishing Returns (~12 min)

**Concept**: The Information-Compute Ratio (ICR) decays as 1/(O x D) -- most data in a large dataset contributes near-zero learning signal. A 50% coreset retains ~99% of accuracy, but the curve is logarithmic, not linear. Students carry a "more data = better model" prior that breaks against the flat tail of the ICR curve.

**Prediction**: "You have a 1M-image dataset. You want to maintain 99% of full-dataset accuracy. What fraction of the data can you discard?"
- A: 10% (keep 900K) -- conservative, assumes every image matters
- B: 30% (keep 700K) -- moderate, acknowledges some redundancy
- **C: 50% (keep 500K)** -- correct: the ICR curve flattens dramatically past the knee
- D: 80% (keep 200K) -- aggressive, underestimates the tail

**Common wrong answer**: A or B. Students overvalue individual data points because they think about datasets like code -- every line matters.

**Why wrong**: Redundancy in natural image datasets is massive. The ICR curve shows that gradient contributions from the bottom 50% of samples are near-zero after the first few epochs. The distribution of "informativeness" follows a power law, not a uniform distribution.

**Instrument**: ICR curve plot with a "dataset fraction" slider (5%--100%) and a "redundancy level" toggle (low/medium/high, simulating different dataset characteristics). A horizontal threshold line marks 99% of baseline accuracy. A "knee" marker shows where the curve transitions from steep to flat. The student's prediction is overlaid as a vertical line against the actual knee.

**mlsysim grounding**: Use Engine.solve() with MobileNetV2 on Jetson Orin NX to compute training time for the full dataset vs. the coreset. The time savings are real and grounded in hardware: T_train(500K) on edge hardware vs. T_train(1M).

**Transition to B**: "You have just discovered that half your data is dead weight. The obvious move: score every sample and keep the informative ones. But scoring has a cost. Is the selection investment worth it?"

---

### Part B -- The Selection Inequality: When Optimization Backfires (~12 min)

**Concept**: The Selection Inequality T_selection + T_train(subset) < T_train(full) must hold for coreset selection to be systems-efficient. Using a full model to score 1M images takes 2.8 hours; a proxy model takes 0.6 hours. On edge hardware with expensive scoring, the inequality can *break* -- selection costs more than just training on everything.

**Prediction**: "You use a ResNet-50 to score 1M images for coreset selection on an A100 GPU. Scoring takes 2.8 hours. Training on the full dataset takes 8.0 hours. Training on the 50% coreset takes 4.2 hours. Is the selection strategy worth it?"
- A: Yes, you save 1.0 hours (8.0 - 2.8 - 4.2)
- **B: Yes, but barely -- you save only 1.0 hours (12.5% time reduction)**
- C: No, you lose 0.2 hours (scoring + subset training > full training)
- D: It depends on the proxy model cost

**Common wrong answer**: Students pick A without computing the actual savings, or pick D because they sense a catch but cannot articulate it. The real lesson is that the margin is thin even in the best case.

**Why wrong**: Students mentally subtract the coreset training time from full training time and forget to add back the scoring cost. The Selection Inequality forces them to account for the full pipeline.

**Instrument**: Waterfall bar chart showing T_selection + T_train(subset) vs. T_train(full). Controls: coreset fraction slider (10%--90%), scoring model dropdown (Full ResNet-50 / Proxy MobileNetV2 / Cached embeddings), deployment context toggle (Cloud A100 / Edge Jetson). On Edge with full-model scoring, the inequality breaks -- the failure state turns the bar red and displays "SELECTION INEQUALITY VIOLATED."

**mlsysim grounding**: Engine.solve(ResNet50, A100) for T_score_full, Engine.solve(MobileNetV2, A100) for T_score_proxy, Engine.solve(ResNet50, JetsonOrinNX) for the edge case that breaks the inequality. All times are grounded in real hardware specs.

**Transition to C**: "You now know that data selection saves time only when the scoring cost is cheap relative to training savings. But there is a more fundamental bottleneck hiding in every training pipeline -- one that has nothing to do with which data you select."

---

### Part C -- The Preprocessing Tax (~12 min)

**Concept**: CPU-side data augmentation and preprocessing is often the true training bottleneck, not GPU compute. A training pipeline that applies RandAugment + normalization + resizing on CPU can starve an A100 that finishes its forward-backward pass in 12 ms while preprocessing takes 45 ms per batch. The GPU sits idle 73% of the time. This is the data pipeline problem from the Data Engineering chapter manifested specifically in the data selection context: the augmentation strategies that prevent overfitting on small coresets (Part A) are themselves a systems bottleneck.

**Prediction**: "Your coreset training pipeline applies RandAugment (5 transforms) + resize + normalize on 8 CPU workers, feeding an A100 GPU. GPU forward-backward takes 12 ms per batch. What fraction of total step time is GPU compute?"
- A: ~80% (GPU dominates, preprocessing is fast)
- B: ~50% (roughly balanced)
- **C: ~25% (GPU waits for CPU most of the time)**
- D: ~10% (GPU is almost entirely idle)

**Common wrong answer**: A. Students assume GPUs are the bottleneck because they are the expensive component. They do not think about CPU preprocessing as a pipeline stage with its own latency.

**Why wrong**: Data augmentation is CPU-bound sequential work. RandAugment applies 5 random transforms per image, each involving pixel-level operations. With 8 workers, the pipeline throughput is 8 x (1000ms / 45ms) = ~178 batches/sec from CPU, but the GPU can consume 1000ms / 12ms = ~83 batches/sec. Wait -- in this case the GPU is actually the bottleneck? No: the 45ms is per-batch with 8 workers, meaning effective CPU throughput is 8/45ms = 177 images/sec, but with heavy augmentation the per-worker time can be 200ms+, making effective throughput 8/200ms = 40 batches/sec vs. GPU capacity of 83 batches/sec. The GPU idle fraction depends on the specific augmentation pipeline.

**Instrument**: Pipeline Gantt chart showing CPU preprocessing (orange) and GPU compute (blue) stages per step. Controls: number of CPU workers slider (1--16), augmentation complexity dropdown (None / Basic flip+crop / RandAugment-5 / RandAugment-10 + MixUp), model dropdown (MobileNetV2 / ResNet-50). A GPU utilization gauge shows the fraction of time the GPU is actually computing. At high augmentation complexity with few workers, the gauge drops below 30%.

**mlsysim grounding**: Engine.solve() provides the GPU forward-backward time for each model. CPU preprocessing time is modeled as T_preprocess = N_transforms x T_per_transform / N_workers, using documented per-transform costs from the chapter (resize: 2ms, flip: 0.5ms, RandAugment transform: 8ms each). The ratio T_gpu / (T_gpu + max(0, T_preprocess - T_gpu)) gives effective utilization when pipelining is imperfect.

**Transition to D**: "Parts A through C revealed three costs hidden inside 'data selection': the diminishing returns of data, the cost of scoring, and the preprocessing tax. But all of these operate within a larger question: given a fixed compute budget, should you spend it on more data or a bigger model?"

---

### Part D -- The Cost-Optimal Frontier: Data-Starved vs. Compute-Starved (~10 min)

**Concept**: The compute-optimal frontier (Chinchilla scaling) determines whether a training run is data-starved (more data would help more than more compute) or compute-starved (more compute would help more than more data). Most teams misdiagnose their position because they think about data and compute as independent resources rather than as coupled terms in a scaling law.

**Prediction**: "You have a fixed compute budget of 10^21 FLOPs. Which configuration achieves lower loss?"
- A: 10B parameter model trained on 200B tokens
- **B: 3B parameter model trained on 660B tokens**
- C: 30B parameter model trained on 66B tokens
- D: All achieve roughly the same loss

**Common wrong answer**: A or C. Students anchor on model size ("bigger model = better") or on round numbers. They do not expect that the optimal point requires 3.3x more data tokens than model parameters.

**Why wrong**: The Chinchilla scaling law shows D proportional to N^0.74, meaning data must scale nearly linearly with model size. Most teams over-allocate to model size and under-allocate to data, placing themselves in the compute-starved regime where additional FLOPs go to waste.

**Instrument**: 2D scatter plot of dataset size (tokens) vs. model size (parameters), with IsoFLOP contour curves. The student's configuration appears as a draggable dot. The Chinchilla-optimal line cuts through the landscape. Regions above the line are labeled "Data-starved" (more data would help), regions below are labeled "Compute-starved" (more compute would help). A loss readout shows the current configuration's predicted loss.

**mlsysim grounding**: The scaling law L(N, D) = A/N^alpha + B/D^beta + E (with Chinchilla parameters) is computed directly. Engine.solve() is not directly applicable here, but the FLOP budget is grounded in real hardware: 10^21 FLOPs = approximately 1,000 A100-hours at 50% MFU (computed from A100 peak FLOPS from the hardware registry).

---

### Synthesis (~6 min)

**Mission**: You manage a computer vision pipeline for a wildlife monitoring system deployed on Jetson Orin NX edge devices. You have 2M images collected over 3 years, a training budget of 500 A100-hours, and a target model (MobileNetV2 with RandAugment).

Write a data selection strategy that addresses:
1. What coreset fraction to use (reference the ICR curve from Part A)
2. Which scoring model to use and whether the Selection Inequality holds on your hardware (reference Part B)
3. How many CPU preprocessing workers you need to keep the GPU fed (reference Part C)
4. Whether your compute budget is better spent on more data or a larger model (reference Part D)

Justify each choice with a specific number from the lab.

**Design Ledger entry**: Record which prediction you got most wrong and what mental model it revealed.

---
---

## Lab 10: The Compression Paradox
**Story arc**: Compression looks like free performance -- until you discover that the hardware ignores your zeros, the accuracy cliff is vertical, and the only reliable shortcut is stealing knowledge from a model too large to deploy.
**Time budget**: 56 min total -- A(12) + B(12) + C(12) + D(8) + E(12) + Synthesis(6)

### Part A -- The Quantization Free Lunch (~12 min)

**Concept**: Reducing precision from FP32 to INT8 costs under 1% accuracy (the "Free Lunch Zone"), then accuracy collapses catastrophically at 3--4 bits (the "Quantization Cliff"). The curve is flat-then-vertical, not gradual. Students expect a smooth accuracy-compression trade-off.

**Prediction**: "You quantize ResNet-50 from FP32 to INT8 (4x compression). How much accuracy do you lose on ImageNet?"
- A: ~5% (noticeable but tolerable)
- B: ~2% (moderate degradation)
- **C: < 1% (essentially free)**
- D: ~0.1% (unmeasurable)

**Common wrong answer**: A or B. Students assume compression must cost something proportional to the compression ratio. "4x smaller must mean significant quality loss."

**Why wrong**: The representational capacity of INT8 (256 discrete levels) is far more than sufficient for the weight distributions of most trained neural networks, which cluster tightly around zero. The information content of the weights is much lower than the precision used to store them.

**Instrument**: Precision selector (FP32 / FP16 / INT8 / INT4 / INT2) with model dropdown (ResNet-50 / MobileNetV2 / BERT-Base). Table and bar chart showing accuracy, model size, latency, and energy for each precision. The INT8 row glows green as the practical sweet spot. Below INT4, accuracy collapses -- the bar turns red and a "QUANTIZATION CLIFF" banner fires.

**mlsysim grounding**: Engine.solve(ResNet50, H100, precision="fp32") vs. Engine.solve(ResNet50, H100, precision="int8") for latency comparison. Memory footprint from the engine's weight_bytes calculation at each precision. The accuracy curve uses documented values from the chapter (@tbl-quantization-accuracy).

**Transition to B**: "Quantization gave you a free 4x compression. Pruning promises even more -- removing 90% of weights sounds like a 10x speedup. But there is a hardware trap waiting."

---

### Part B -- The Pruning Hardware Trap (~12 min)

**Concept**: Unstructured pruning at 90% sparsity gives zero latency speedup on standard GPU kernels because dense GEMM iterates over every element including zeros. The hardware cannot skip sparse multiplications without specialized support. Structured pruning removes entire channels, yielding real speedup, but destroys accuracy faster.

**Prediction**: "You prune ResNet-50 to 90% sparsity (remove 90% of weights, set to zero) using unstructured magnitude pruning. What inference speedup do you get on an H100 GPU?"
- A: ~10x (removed 90% of work)
- B: ~5x (some overhead from sparse format)
- C: ~2x (significant overhead)
- **D: ~1.0x (no speedup at all)**

**Common wrong answer**: A. This is the strongest wrong-prediction moment in the entire lab suite. Students reason linearly: "90% fewer multiplications should mean 90% less time." It is an entirely reasonable assumption that is entirely wrong on standard hardware.

**Why wrong**: GPU GEMM kernels use dense matrix multiplication. Every element of the weight matrix is loaded from HBM and multiplied, including the zeros. The zeros save no memory bandwidth (the matrix is the same size) and no compute (the multiply still executes). Without hardware support for sparse matrix formats (like NVIDIA's 2:4 structured sparsity), unstructured pruning is invisible to the hardware.

**Instrument**: Sparsity slider (0%--95%) with pruning type toggle (Unstructured / Structured / 2:4 Structured). Two charts: (1) Speedup vs. sparsity -- flat line at 1.0x for unstructured, rising curve for structured, step function for 2:4; (2) Accuracy vs. sparsity -- gradual decline for unstructured, steeper decline for structured. Pushing structured pruning past 85% triggers an accuracy collapse failure state.

**mlsysim grounding**: Engine.solve(ResNet50, H100, precision="fp16") gives the baseline latency. Structured pruning at X% reduces the effective parameter count by X%, which reduces inference_flops proportionally -- Engine.solve() with a modified workload shows real speedup. Unstructured pruning does not change the workload spec at all because the hardware sees the same dense matrix.

**Transition to C**: "Quantization compresses 4x for free. Structured pruning buys speedup but costs accuracy. What if you need to fit an 8B-parameter LLM on a phone with 4 GB of RAM? No single technique spans that gap."

---

### Part C -- The Compression Pareto Frontier (~12 min)

**Concept**: Deploying an 8B-parameter LLM across three mobile memory tiers (8 GB / 4 GB / 2 GB) requires composing INT8, INT4, and structured pruning along a Pareto frontier. No single technique spans all tiers. The Pareto frontier reveals that some compression combinations are dominated (worse accuracy AND worse compression than an alternative).

**Prediction**: "You need to deploy Llama-3 8B on a device with 4 GB RAM. Which compression strategy fits?"
- A: FP16 (16 GB) -- does not fit
- B: INT8 (8 GB) -- does not fit
- **C: INT4 (4 GB) -- fits, with measurable accuracy loss**
- D: INT4 + 50% structured pruning (2 GB) -- fits, but accuracy may be unacceptable

**Common wrong answer**: B. Students underestimate the memory footprint of INT8 for an 8B model (8B x 1 byte = 8 GB, which exceeds the 4 GB budget when you include KV cache and runtime overhead).

**Why wrong**: Memory accounting requires including not just weights but also the KV cache, activation memory, and runtime overhead. INT8 weights alone consume 8 GB, leaving zero room for anything else.

**Instrument**: Scatter plot showing accuracy vs. model size for all compression configurations. Per-tier memory budget lines at 8 GB, 4 GB, and 2 GB. Students select a compression strategy per tier from dropdowns. Pareto-optimal configurations are highlighted; dominated configurations are grayed out. Selecting a configuration that exceeds a tier's budget triggers an OOM failure state.

**mlsysim grounding**: Engine.solve(Llama3_8B, iPhone15Pro, precision="fp16") triggers OOM. Engine.solve(Llama3_8B, iPhone15Pro, precision="int4") shows feasibility. The memory_footprint field from the PerformanceProfile grounds every point on the Pareto frontier.

**Transition to D**: "Compression reduces model size. But there is a second dividend that matters more on battery devices: every bit you do not move from memory saves energy."

---

### Part D -- The Energy Dividend: Bits are Joules (~8 min)

**Concept**: Moving data costs 40,000x more energy than computing it (DRAM read: 640 pJ vs. integer add: 0.015 pJ). INT8 inference uses up to 20x less energy than FP32 because the dominant energy cost is data movement, not arithmetic. On a battery device, this is the difference between 1 hour and 20 hours of operation.

**Prediction**: "For a MobileNetV2 inference on a mobile device, what fraction of total energy is spent on memory access vs. computation?"
- A: 50/50 -- balanced between memory and compute
- B: 70/30 -- memory dominates somewhat
- C: 90/10 -- memory dominates strongly
- **D: 99/1 -- memory access is essentially all of the energy**

**Common wrong answer**: A or B. Students think of GPUs as "compute engines" and assume computation is the primary energy consumer.

**Why wrong**: The energy hierarchy is: DRAM read (640 pJ) >> SRAM read (5 pJ) >> FP32 multiply (3.7 pJ) >> INT8 multiply (0.2 pJ) >> integer add (0.015 pJ). For a network doing millions of multiplies, each requiring a weight loaded from DRAM, memory access energy dominates by 100x or more.

**Instrument**: Energy breakdown stacked bar showing DRAM access energy vs. compute energy for each precision format (FP32 / FP16 / INT8 / INT4). Deployment target toggle (Cloud H100 / Mobile iPhone / IoT Cortex-M). A battery life estimate shows hours of continuous inference. The visual starkness of the 40,000x DRAM-to-compute ratio makes the "Memory Wall" visceral.

**mlsysim grounding**: Engine.solve(MobileNetV2, iPhone15Pro, precision="fp32").energy vs. Engine.solve(MobileNetV2, iPhone15Pro, precision="int8").energy provides the energy comparison. The per-operation energy constants are from the chapter (@tbl-energy-per-operation).

**Transition to E**: "Quantization and pruning modify the original model. But what if you could train a *different*, smaller model that inherits the knowledge of a large one? That is knowledge distillation -- and it unlocks deployment scenarios that compression alone cannot reach."

---

### Part E -- Dark Knowledge Transfer (~12 min)

**Concept**: Knowledge distillation trains a small "student" model to mimic the soft probability outputs of a large "teacher" model. The student learns not just the correct class but the teacher's uncertainty structure ("dark knowledge") -- which wrong classes are similar, how confident to be. A distilled MobileNetV2 student can match 95% of a ResNet-50 teacher's accuracy at 1/10th the compute, enabling deployment on edge hardware where the teacher cannot physically run.

**Prediction**: "A ResNet-50 teacher achieves 76.1% ImageNet accuracy. You distill into a MobileNetV2 student. What accuracy does the student achieve?"
- A: ~65% (significant knowledge loss from compression)
- B: ~70% (moderate knowledge transfer)
- **C: ~73% (retains 95% of teacher accuracy)**
- D: ~76% (perfect knowledge transfer)

**Common wrong answer**: A or B. Students expect that a 10x smaller model must lose proportionally more accuracy. They do not realize that the soft labels from the teacher carry far more information than hard labels alone.

**Why wrong**: Hard labels (one-hot vectors) discard the teacher's uncertainty. Soft labels with temperature scaling preserve the relative probabilities across all classes -- a "dark" signal that tells the student which classes the teacher found confusable. This information-dense supervision signal closes most of the gap between student and teacher capacity.

**Instrument**: Teacher-student comparison dashboard. Teacher model dropdown (ResNet-50 / BERT-Large / Llama-3 70B), student model dropdown (MobileNetV2 / DistilBERT / Llama-3 8B). Temperature slider (1.0--20.0). Accuracy comparison bar chart showing: student trained on hard labels, student trained with distillation, teacher. A deployment feasibility check shows which hardware the teacher fits on vs. the student.

**mlsysim grounding**: Engine.solve(ResNet50, JetsonOrinNX, precision="fp16") shows the teacher is feasible on edge but slow. Engine.solve(MobileNetV2, JetsonOrinNX, precision="int8") shows the distilled student runs 5x faster and fits comfortably. Engine.solve(ResNet50, CortexM7) triggers OOM -- the teacher *cannot run at all* on TinyML, but the distilled student can. This is the deployment unlock that compression alone cannot achieve.

---

### Synthesis (~6 min)

**Mission**: You must deploy a text classification model across three tiers: Cloud (H100), Mobile (iPhone 15 Pro), and IoT (Cortex-M7). Your teacher model is BERT-Large (340M parameters, 93% accuracy).

Design a compression strategy for each tier:
1. Which technique(s) to apply (quantization, pruning, distillation, or combination)
2. Expected accuracy at each tier
3. Whether the Energy Dividend (Part D) or the Deployment Unlock (Part E) is the primary motivation
4. The specific mlsysim feasibility check that validates your choice

**Design Ledger entry**: Record the "Pruning Hardware Trap" prediction. How far off were you?

---
---

## Lab 11: The Hardware Roofline
**Story arc**: Your code is not broken -- your hardware has a ceiling, and that ceiling changes shape depending on what chip you are running on, what operation you are executing, and whether you fuse your kernels.
**Time budget**: 52 min total -- A(12) + B(12) + C(10) + D(10) + E(8) + Synthesis(6)

### Part A -- The Memory Wall (Roofline Diagnosis) (~12 min)

**Concept**: A GEMM kernel at N=512 achieves only 31.5% MFU on the H100 -- not because the code is broken, but because arithmetic intensity (170 FLOP/byte) falls below the ridge point (295 FLOP/byte). The kernel is correctly hitting the memory bandwidth ceiling. The Roofline model is a diagnostic tool, not a performance target.

**Prediction**: "A GEMM kernel (N=512, FP16) on an H100 achieves 31.5% of peak TFLOPS. What is the problem?"
- A: The code has a bug -- it should be 90%+
- B: The GPU is thermally throttling
- **C: The kernel is memory-bandwidth-bound -- it is hitting the Roofline ceiling correctly**
- D: FP16 precision reduces peak throughput

**Common wrong answer**: A. Students equate low utilization with bad code. They do not yet have the mental model that the hardware has two ceilings (compute and bandwidth) and the lower one wins.

**Instrument**: Log-log Roofline plot with a matrix dimension slider (N=128 to 8192) and precision toggle (FP32 / FP16 / INT8). The operation point slides along the bandwidth slope as N increases, crossing the ridge into the compute-bound regime. Metric cards display arithmetic intensity, MFU, and regime classification ("Memory-bound" / "Compute-bound") live.

**mlsysim grounding**: Engine.solve() computes arithmetic_intensity, mfu, and bottleneck classification directly. The ridge point is computed from hardware specs: H100_FLOPS_FP16_TENSOR / H100_MEM_BW.

**Transition to B**: "You now know that a single GEMM kernel can be diagnosed with the Roofline. But real models contain dozens of different operations. What happens when you mix GEMM with LayerNorm and Softmax?"

---

### Part B -- Kernel Fusion: The Elementwise Trap (~12 min)

**Concept**: LLM inference mixes kernels spanning 3 orders of magnitude in arithmetic intensity. GEMM can become compute-bound at large batch, but LayerNorm (AI ~ 0.83 FLOP/byte) and Softmax are permanently memory-bound. These operations dominate inference time not because they do a lot of work, but because each one requires a full round-trip to HBM. Kernel fusion eliminates intermediate HBM writes, collapsing 3 memory-bound kernels into 1.

**Prediction**: "You fuse LayerNorm + Dropout + ReLU into a single kernel, eliminating 2 intermediate HBM writes. What speedup do you get for this fused sequence?"
- A: ~1.3x (modest improvement)
- B: ~2x (eliminated half the work)
- **C: ~3--5x (eliminated most of the memory traffic)**
- D: ~10x (eliminated nearly all overhead)

**Common wrong answer**: A or B. Students think of fusion as eliminating "overhead" (small), not as eliminating *memory traffic* (which is the dominant cost for elementwise ops).

**Instrument**: Multi-operation Roofline showing three markers (GEMM, LayerNorm, Softmax) at current batch size. Toggle between "Eager" (each op is a separate kernel with separate HBM read/write) and "Fused" (single kernel, single HBM round-trip). Timeline visualization shows GPU compute bars vs. memory stalls. Batch size slider and sequence length slider. Increasing batch too far triggers an OOM failure state when KV cache exceeds device RAM.

**mlsysim grounding**: Engine.solve() gives total latency, which is dominated by the memory term at low AI. The per-kernel decomposition models each operation's AI separately and computes latency as sum of per-kernel max(compute, memory) in eager mode vs. combined in fused mode.

**Transition to C**: "The Roofline plot and fusion strategy both depend on the specific hardware. What happens when you move the same kernel from cloud to edge?"

---

### Part C -- The Hardware Balance Shift (~10 min)

**Concept**: The same GEMM kernel can be compute-bound on an edge device (Jetson Orin NX ridge point ~118 FLOP/byte) but memory-bound on the H100 (ridge point ~295 FLOP/byte). More powerful accelerators are *paradoxically harder to saturate*, because compute grows faster than bandwidth across generations. This means that code optimized for one hardware target may have a completely different bottleneck on another.

**Prediction**: "A GEMM at N=1024 is compute-bound on a Jetson Orin NX. Is it compute-bound on an H100?"
- A: Yes -- if it is compute-bound on weaker hardware, it must be compute-bound on stronger hardware
- **B: No -- the H100 has a higher ridge point, so the same operation is now memory-bound**
- C: It depends on the precision
- D: It is memory-bound on both

**Common wrong answer**: A. The intuition "more powerful = more headroom" is wrong because the ridge point also rises. Students need to see the Roofline redraw with different ceilings.

**Instrument**: Hardware toggle (Cloud H100 / Edge Jetson Orin NX / Mobile A17 Pro). The Roofline redraws with different compute ceilings and bandwidth slopes, shifting the ridge point. A fixed GEMM operation point stays at the same AI but changes regime classification. Side-by-side metric cards show MFU on each platform for the same operation.

**mlsysim grounding**: Engine.solve(workload, H100) vs. Engine.solve(workload, JetsonOrinNX) vs. Engine.solve(workload, iPhone15Pro). The bottleneck field directly shows the regime shift. Ridge points computed from each hardware's peak_flops / memory.bandwidth.

**Transition to D**: "You can now diagnose whether an operation is compute-bound or memory-bound on any hardware. But there is a cost your Roofline plot does not show: energy."

---

### Part D -- The Energy Roofline (~10 min)

**Concept**: Energy efficiency has its own Roofline. Operations in the memory-bound regime waste energy on data movement (640 pJ per DRAM access) rather than useful computation (3.7 pJ per FP32 multiply). The energy-optimal operating point is deep in the compute-bound regime, where most Joules go to arithmetic. This is why hardware architects build larger caches and higher-bandwidth memory: to push more operations into the energy-efficient compute-bound regime.

**Prediction**: "A memory-bound operation (AI = 10 FLOP/byte) and a compute-bound operation (AI = 500 FLOP/byte) both perform the same total FLOPs. Which uses more energy?"
- **A: The memory-bound operation uses ~10x more energy**
- B: They use roughly the same energy (same FLOPs = same work)
- C: The compute-bound operation uses more (higher utilization = more power)
- D: It depends on the clock frequency

**Common wrong answer**: B. "Same FLOPs = same energy" is a deeply held intuition. Students do not account for the energy cost of the *data movement* required to feed those FLOPs.

**Instrument**: Energy Roofline plot (energy per FLOP vs. arithmetic intensity) with the same hardware toggle from Part C. The energy floor is in the compute-bound regime; the energy penalty rises linearly in the memory-bound regime. Operations from the previous parts are plotted on this new Roofline. A total energy gauge shows Joules per inference.

**mlsysim grounding**: Engine.solve().energy provides the total energy. The energy decomposition uses hardware TDP and the compute/memory time split from the latency breakdown.

**Transition to E**: "The Roofline tells you where you are. Fusion moves you right along the AI axis. But there is one more trick: tiling -- computing on data that fits in cache before it escapes to HBM."

---

### Part E -- The Tiling Dividend (~8 min)

**Concept**: Matrix multiplication on large matrices that exceed L2 cache forces repeated HBM accesses for the same data. Tiling (blocking) the computation into cache-sized chunks keeps data in fast SRAM, effectively increasing arithmetic intensity by reducing redundant memory traffic. FlashAttention is the canonical example: by tiling the attention computation to fit in SRAM, it achieves 2--4x speedup over the standard implementation.

**Prediction**: "FlashAttention tiles the attention computation to fit in GPU SRAM. Compared to standard attention, how much faster is it at sequence length 4096?"
- A: ~1.2x (minor improvement)
- B: ~1.5x (moderate)
- **C: ~2--4x (significant, from eliminating HBM round-trips)**
- D: ~10x (transformative)

**Common wrong answer**: A. Students who learned about fusion in Part B may think tiling is just another incremental optimization. They underestimate how many redundant HBM reads standard attention performs.

**Instrument**: Tile size slider (32 to 2048 elements) and sequence length slider (512 to 16384). A memory traffic counter shows bytes read from HBM for standard vs. tiled attention. The ratio is displayed prominently. A simplified Roofline shows the effective arithmetic intensity increasing as tile size grows (because fewer bytes are transferred for the same FLOPs).

**mlsysim grounding**: Engine.solve() provides the baseline. The tiling model computes effective AI as total_flops / (bytes_accessed_with_tiling), where bytes_accessed scales with tile_size relative to SRAM capacity. The speedup is the ratio of memory-bound latency at the two AI values.

---

### Synthesis (~6 min)

**Mission**: You must deploy a Transformer inference service on both H100 (cloud) and Jetson Orin NX (edge). For each target, specify:
1. Which operations are memory-bound vs. compute-bound (Roofline diagnosis, Part A/C)
2. Which kernels to fuse (Part B)
3. Whether FlashAttention tiling helps (Part E)
4. The energy cost per inference on each platform (Part D)

Use specific ridge points and arithmetic intensities from the lab.

**Design Ledger entry**: Record your Part A prediction (GEMM utilization). Most students think low utilization = broken code.

---
---

## Lab 12: The Benchmarking Trap
**Story arc**: Vendor benchmarks are designed to make hardware look good. Your job is to make hardware look *honest* -- and the gap between the two is where millions of dollars disappear.
**Time budget**: 52 min total -- A(12) + B(12) + C(10) + D(12) + Synthesis(6)

### Part A -- The Amdahl Ceiling (~12 min)

**Concept**: A 10x inference speedup with 45% non-inference overhead yields only 2.0x end-to-end improvement. Amdahl's Law caps system speedup at 1/(1-f) where f is the non-optimized fraction. The remaining 8x of hardware investment is wasted on a bottleneck that has already moved.

**Prediction**: "You replace your inference GPU with one that is 10x faster. Your pipeline is 45% preprocessing + 55% inference. What is your end-to-end speedup?"
- A: ~10x (the new GPU is 10x faster)
- B: ~5x (about half the pipeline speeds up)
- **C: ~2.0x (Amdahl's Law caps the gain)**
- D: ~1.5x (even worse than expected)

**Common wrong answer**: A or B. Students think component speedup translates to system speedup. They have not internalized that the non-accelerated fraction becomes the new bottleneck.

**Why wrong**: Amdahl's Law: Speedup = 1 / (0.45 + 0.55/10) = 1 / 0.505 = 1.98x. The 45% preprocessing fraction is now 89% of total time. The $30K GPU upgrade delivered a 2x improvement, not 10x.

**Instrument**: Before/after waterfall bar chart showing preprocessing (fixed) + inference (accelerated). Sliders: inference speedup (1x--100x), non-inference fraction (0.05--0.80). An Amdahl saturation curve shows speedup approaching the asymptote 1/f. A "wasted speedup" metric shows the gap between component improvement and system improvement. A dollar-per-improvement gauge shows cost efficiency declining as speedup increases.

**mlsysim grounding**: Engine.solve(ResNet50, H100) vs. Engine.solve(ResNet50, A100) provides the actual speedup ratio for the inference component. The non-inference fraction is parameterized. Total system time = T_preprocess + T_inference / speedup_factor.

**Transition to B**: "Amdahl showed that one fast component does not make a fast system. But even the fast component may be lying about its speed. Vendor benchmarks measure *burst* performance. Production runs are *sustained*."

---

### Part B -- Peak vs. Sustained: The Thermal Cliff (~12 min)

**Concept**: A vendor advertises 30 FPS for an edge chip. After 5 minutes of continuous inference in a fanless enclosure, thermal throttling halves throughput to 15 FPS. Vendor benchmarks are burst measurements; production runs are sustained. The gap between peak and sustained performance can be 2x or more.

**Prediction**: "An edge device benchmarks at 30 FPS in a 1-minute vendor test. What sustained FPS do you get after 10 minutes of continuous inference in a fanless enclosure at 35C ambient?"
- A: ~28 FPS (minor degradation)
- B: ~24 FPS (some thermal throttling)
- **C: ~15 FPS (thermal throttle halves performance)**
- D: ~8 FPS (severe throttling)

**Common wrong answer**: A. Students trust vendor numbers because they assume benchmarks represent production conditions. They do not account for thermal accumulation over time.

**Why wrong**: Thermal energy accumulates as Q = P x t. In a fanless enclosure, convective cooling is limited. Junction temperature rises linearly until hitting the thermal throttle threshold (typically 85--100C), at which point the chip reduces clock frequency to stay within its thermal envelope.

**Instrument**: Time scrubber (0--10 minutes). A piecewise thermal model: Phase 1 (0 to T_throttle): FPS constant at peak, junction temperature rising linearly as T_junction(t) = T_ambient + (TDP / thermal_conductance) x (1 - e^(-t/tau)). Phase 2 (T_throttle onward): FPS drops to TDP_throttled / TDP x peak_FPS. Controls: ambient temperature slider (20--45C), cooling type toggle (active fan / passive heatsink / fanless), TDP slider.

**mlsysim grounding**: Engine.solve(MobileNetV2, JetsonOrinNX).latency provides the per-inference time at peak. The thermal model uses hardware.tdp from the registry. The piecewise model: T_junction = T_ambient + (TDP_watts / G_thermal) x (1 - exp(-t / RC_thermal)), where G_thermal and RC_thermal are parameters for each cooling type.

**Transition to C**: "Vendor benchmarks lie about sustained performance. They also lie about *which metric matters*. A chip that wins on accuracy may fail on latency, and a chip that wins on throughput may fail on power."

---

### Part C -- The Multi-Metric Trap (~10 min)

**Concept**: The model configuration with the best single-metric score (94% accuracy, 1200 QPS) violates latency and power SLOs. Only the balanced configuration (91% accuracy, 95 ms p99, 600 QPS, 4.5 W) passes all four deployment gates simultaneously. Optimizing a single metric while ignoring the others is the most common benchmarking mistake.

**Prediction**: "You have four model configurations. Configuration A has the highest accuracy (94%). Is it deployable?"
- A: Yes -- highest accuracy is always the best choice
- B: Probably -- the other metrics are secondary
- **C: No -- it violates the latency SLO (p99 > 100 ms)**
- D: No -- it violates the power budget (> 5 W)

**Common wrong answer**: A. In academic settings, accuracy is the only metric. Students have not yet internalized that production deployment has simultaneous constraints on multiple axes.

**Instrument**: Batch size slider, precision dropdown, model variant dropdown. A radar chart with four axes (accuracy, latency, throughput, power) overlays SLO threshold rings. An SLO compliance table shows pass/fail per metric with green/red indicators. Any violated SLO triggers a "DEPLOYMENT BLOCKED" failure state with a red banner naming the violated constraints.

**mlsysim grounding**: Engine.solve() at each configuration provides latency, throughput (from batch_size / latency), and energy (proxy for power). Accuracy values come from the chapter's documented configurations. The SLO thresholds are parameters: p99_latency < 100ms, power < 5W, accuracy > 90%, throughput > 500 QPS.

**Transition to D**: "You now know that single-metric benchmarks are misleading and that thermal throttling hides in sustained runs. The final trap: average latency hides catastrophic tail behavior."

---

### Part D -- The Tail Latency Diagnostic (~12 min)

**Concept**: A system with 50 ms average latency but 500 ms p99 violates a 200 ms SLO for 1% of requests. At 10K requests/sec, that is 100 failures per second. Average latency is not just insufficient -- it is *actively misleading* because it hides the distribution shape. The latency distribution for real inference systems follows a log-normal distribution with a heavy right tail.

**Prediction**: "Your inference service reports 50 ms average latency. Your SLO is 200 ms p99. Is the SLO satisfied?"
- A: Yes -- 200 ms is 4x the average, which provides plenty of headroom
- B: Probably -- p99 is usually ~2x the average
- **C: No -- the p99 is ~500 ms due to the heavy tail of the latency distribution**
- D: Cannot determine without more information

**Common wrong answer**: A or B. Students assume latency distributions are symmetric (like a Gaussian) where p99 is close to the mean. Real inference latency distributions are heavy-tailed.

**Why wrong**: Inference latency has multiple sources of variability: garbage collection pauses, memory allocation, cache misses, OS scheduling, thermal throttling. These create a log-normal or worse distribution where p99 can be 5--10x the median.

**Instrument**: Latency distribution histogram generated from Engine.solve() base latency + log-normal noise (sigma parameter from slider). Mean, p50, p95, p99, and p99.9 vertical lines on the histogram. SLO threshold line (adjustable). A violation counter shows "X% of requests exceed SLO." Toggle between training mode (shows throughput + time-to-accuracy + scaling efficiency) and inference mode (shows the latency distribution).

**mlsysim grounding**: Engine.solve(ResNet50, H100, batch_size=B) provides the deterministic base latency. The lab adds log-normal noise: latency_sample = base_latency x exp(N(0, sigma^2)), where sigma controls tail heaviness (slider: 0.1 to 1.0). This generates a realistic latency distribution. The p99 is computed from the distribution.

---

### Synthesis (~6 min)

**Mission**: A vendor claims their new edge chip delivers "30 FPS, 94% accuracy, 50 ms average latency" for your computer vision pipeline. You must write a 3-point rebuttal explaining why this benchmark is insufficient for your production deployment:

1. What is the *sustained* FPS after thermal steady-state? (Part B)
2. Does the 94% accuracy configuration pass all four SLO gates? (Part C)
3. What is the *p99* latency, and does it satisfy your 200 ms SLO? (Part D)
4. What is the *end-to-end* speedup given your preprocessing overhead? (Part A)

This is the "debunk a vendor claim" exercise. It is the most directly career-applicable synthesis in the lab series.

**Design Ledger entry**: Record your Amdahl prediction. The gap between component speedup and system speedup is the single most common mistake in systems engineering.

---
---

## Lab 13: The Tail Latency Trap
**Story arc**: Your serving system looks healthy at 50% utilization, manageable at 70%, and is on fire at 80% -- and the fire is invisible to every metric except the one you are not measuring.
**Time budget**: 50 min total -- A(12) + B(12) + C(12) + D(8) + Synthesis(6)

### Part A -- The Tail Latency Explosion (~12 min)

**Concept**: The M/M/1 queuing model predicts that P99 latency diverges nonlinearly from mean latency as server utilization increases. At 80% utilization, P99 is 23x the service time while the mean is only 5x. A system reporting "healthy 25 ms average latency" is simultaneously delivering 115 ms tail latency that violates a 50 ms SLO for 1 in 100 users.

**Prediction**: "Your ResNet-50 server has a 5 ms service time and runs at 80% utilization. What P99 latency do your slowest 1-in-100 users experience?"
- A: ~10 ms (2x service time)
- B: ~25 ms (5x service time, matches the mean)
- C: ~50 ms (10x service time)
- **D: ~115 ms (23x service time)**

**Common wrong answer**: B. Students confuse mean latency (25 ms at rho=0.8) with P99 latency. They do not expect the ln(100) = 4.6 multiplier from the exponential tail.

**Instrument**: Utilization slider (rho, 0.10--0.95). Live P99 latency histogram with mean (blue dashed) and P99 (red solid) vertical markers diverging as utilization increases. SLO threshold line (green dashed, adjustable). Service time slider (1--20 ms). Metric row: mean latency, P99 latency, P99/mean ratio (always 4.6x for M/M/1, but absolute values change dramatically).

**mlsysim grounding**: Engine.solve(ResNet50, H100, batch_size=1).latency provides the service time. The queuing model: W_mean = service_time / (1 - rho), W_p99 = 4.6 x service_time / (1 - rho). Source: @eq-mm1-wait and @eq-p99-latency in the chapter.

**Transition to B**: "You now know that utilization above 70% is dangerous for tail latency. The natural response: batch requests to improve throughput. But batching has its own hidden tax."

---

### Part B -- The Batching Tax (~12 min)

**Concept**: Larger batch sizes improve GPU throughput (batch-32 achieves 6.4x over batch-1) but impose a formation delay tax of (B-1)/(2 x lambda) that can exceed the SLO before inference even begins. The latency-throughput Pareto frontier has a sharp "knee" where throughput gains plateau and latency explodes.

**Prediction**: "You increase batch size from 1 to 32 to improve throughput. Your arrival rate is 500 QPS and your SLO is 50 ms. What happens?"
- A: Throughput improves 6.4x and latency stays under SLO
- B: Throughput improves 6.4x but latency slightly exceeds SLO
- **C: Throughput improves but the batch formation delay alone (31 ms) consumes 62% of the SLO budget before inference starts**
- D: The system crashes from memory overflow

**Common wrong answer**: A. Students think of batching as pure throughput gain without a latency cost. They do not account for the time requests spend waiting for the batch to fill.

**Why wrong**: Formation delay = (B-1) / (2 x lambda) = 31 / (2 x 500) = 31 ms. This is 62% of the 50 ms SLO consumed before the GPU even starts. Add inference time (which increases with batch size), and the SLO is violated.

**Instrument**: Batch size slider (1--64), arrival rate slider (100--2000 QPS), SLO budget slider (10--100 ms). Dual-axis chart: throughput curve (rising, saturating) and p99 latency curve (U-shaped, with minimum at optimal batch). Latency waterfall showing formation delay + inference time + queuing delay. SLO violation banner when total latency exceeds budget.

**mlsysim grounding**: Engine.solve(ResNet50, H100, batch_size=B).latency provides the inference time component. Engine.solve(ResNet50, H100, batch_size=B).throughput provides the throughput component. Formation delay = (B-1) / (2 x arrival_rate). Total latency = formation_delay + inference_latency + queuing_delay(rho).

**Transition to C**: "For vision models, the batching trade-off is between latency and throughput. For LLMs, there is a third constraint: the KV cache memory wall."

---

### Part C -- The LLM Memory Wall (~12 min)

**Concept**: LLM decode-phase token generation is entirely memory-bandwidth-bound (T_token = Model_Size / Memory_BW). KV cache capacity -- not compute TFLOPS -- determines maximum concurrent batch size. At 128K context length, even an 8xH100 node can serve only 1 concurrent request for a 70B model.

**Prediction**: "You serve Llama-2 70B at 128K context on 8xH100 (640 GB total HBM). What is the maximum concurrent batch size?"
- A: 8 (one per GPU)
- B: 4 (memory split between model and cache)
- **C: 1 (the KV cache for a single 128K context nearly fills all available memory)**
- D: 16 (tensor parallelism across 8 GPUs frees memory)

**Common wrong answer**: A or B. Students think about compute parallelism (8 GPUs = 8 requests) rather than memory capacity. They do not realize that KV cache at 128K tokens for a 70B model consumes hundreds of GB.

**Why wrong**: KV cache size = 2 x layers x heads x head_dim x seq_len x batch x bytes_per_element. For Llama-2 70B at 128K in FP16: approximately 320 GB for a single request. Model weights in FP16: 140 GB. Total: 460 GB, leaving only 180 GB headroom out of 640 GB -- not enough for a second request with KV cache.

**Instrument**: Model size dropdown (7B / 13B / 70B), precision dropdown (FP16 / INT8 / INT4 for weights), context length slider (2K--128K tokens), concurrent batch size slider (1--128), GPU count toggle (1 / 4 / 8). Stacked memory bar: weights (blue, fixed) + KV cache (orange, growing with context x batch) + available VRAM. Red "OOM ZONE" when total exceeds HBM capacity. T_token readout showing memory-bandwidth-bound decode speed.

**mlsysim grounding**: Engine.solve(Llama2_70B, H100, batch_size=B, seq_len=S, precision=P) provides memory_footprint and latency. The feasible field turns False when memory exceeds capacity. KV cache size computation uses the TransformerWorkload fields (layers, hidden_dim, heads).

**Transition to D**: "The KV cache wall limits concurrent requests. But there is another capacity killer for serving: cold starts."

---

### Part D -- The Cold Start Tax (~8 min)

**Concept**: Loading a model from disk to GPU memory before the first inference adds 5--30 seconds of latency for large models. For serverless or auto-scaling deployments, this cold start latency is experienced by real users during traffic spikes. The cold start time is dominated by PCIe or NVMe bandwidth, not compute -- it is another manifestation of the data movement wall.

**Prediction**: "You auto-scale a Llama-2 70B serving instance during a traffic spike. How long does the first user wait for a response?"
- A: ~200 ms (normal inference latency)
- B: ~2 seconds (some model loading overhead)
- **C: ~15 seconds (loading 140 GB of weights over PCIe Gen5)**
- D: ~60 seconds (loading from network storage)

**Common wrong answer**: A or B. Students do not think about model loading as a latency component because in their experience, models are already in memory.

**Why wrong**: Cold start = model_size_bytes / loading_bandwidth. For 70B in FP16 (140 GB) over PCIe Gen5 (64 GB/s): 140/64 = 2.2 seconds just for the transfer. With NVMe reads, model deserialization, CUDA context initialization, and warmup inference: 10--15 seconds total.

**Instrument**: Model size dropdown, storage type dropdown (NVMe SSD / Network FS / Cached in RAM), interconnect dropdown (PCIe Gen4 / Gen5). A timeline showing: disk read + PCIe transfer + CUDA init + warmup inference. Total cold start time displayed prominently against a "user patience threshold" of 3 seconds.

**mlsysim grounding**: Cold start time = model_memory_footprint / min(storage_bandwidth, interconnect_bandwidth) + CUDA_init_overhead. Model memory from Engine.solve().memory_footprint. Storage bandwidth from hardware registry storage.bandwidth. PCIe bandwidth from hardware registry interconnect.bandwidth.

---

### Synthesis (~6 min)

**Mission**: You must design a serving deployment for two workloads: (1) a ResNet-50 vision classifier serving 10K QPS with a 50 ms p99 SLO, and (2) a Llama-2 70B chat endpoint serving 100 concurrent users with 32K context.

For each workload, specify:
1. Maximum utilization target (Part A: where does P99 become unacceptable?)
2. Optimal batch size (Part B: where is the knee of the Pareto frontier?)
3. GPU memory budget breakdown: weights vs. KV cache vs. headroom (Part C)
4. Cold start mitigation strategy (Part D: pre-warm, keep-alive, or accept the tax?)

**Design Ledger entry**: Record your P99 latency prediction from Part A. The ratio between mean and P99 is the most underestimated quantity in serving.

---
---

## Lab 14: The Silent Degradation Problem
**Story arc**: Your model shipped on Monday. By Friday, it has silently lost 3 accuracy points. By month six, it has lost 7. Your monitoring dashboard has been green the entire time.
**Time budget**: 54 min total -- A(12) + B(12) + C(12) + D(12) + Synthesis(6)

### Part A -- PSI Drift Detection (The Silent Drift) (~12 min)

**Concept**: A fraud detection model loses 7 percentage points of accuracy over 6 months while every infrastructure metric (uptime, latency, error rate) stays green. PSI-based monitoring of input feature distributions detects drift weeks before accuracy degradation becomes visible. Infrastructure health and model health are decoupled.

**Prediction**: "Your fraud detection model has been deployed for 6 months. Your monitoring dashboard shows 100% uptime, <50 ms latency, 0.01% error rate. What has happened to model accuracy?"
- **A: Still ~95% -- the system is healthy (this is what students pick)**
- B: Down to ~91% -- slight degradation
- C: Down to ~88% -- significant degradation
- D: Model has crashed

**Common wrong answer**: A. Students conflate infrastructure health with model health. Green dashboards create false confidence.

**Why wrong**: Data distributions shift while models stay static. Transaction patterns change (new fraud vectors, seasonal spending shifts, economic changes). The model was trained on a distribution that no longer exists. PSI for the "transaction_amount" feature crossed the 0.2 alert threshold at week 8, but without PSI monitoring, no one noticed.

**Instrument**: Time slider (0--26 weeks). Split dashboard: left panel shows infrastructure metrics (all permanently green -- uptime, latency, error rate). Right panel shows PSI trajectories for three features (transaction_amount, merchant_category, time_of_day) crossing the 0.2 threshold at different weeks. Accuracy decay card shows the gap between "system healthy" and "model degraded." A "detection gap" counter shows weeks between PSI alert and accuracy drop below threshold.

**mlsysim grounding**: The PSI and accuracy decay model uses documented drift formulas from the chapter: PSI(t) = sum((p_t(i) - p_0(i)) x ln(p_t(i)/p_0(i))), with feature distributions shifting at configurable rates. Accuracy(t) = Accuracy_0 - lambda x cumulative_drift(t). The drift rate lambda and feature shift rates are parameterized.

**Transition to B**: "You now know the model is degrading. The question is: how often should you retrain? Too often wastes compute. Too rarely wastes accuracy. There is a mathematically optimal answer."

---

### Part B -- Optimal Retraining Cadence (The Half-Life of a Model) (~12 min)

**Concept**: The staleness cost model T* = sqrt(2C / C_drift) produces a sublinear relationship: 4x more expensive retraining only doubles the interval. This transforms retraining from an ad hoc decision into a quantitative economic optimization with a U-shaped cost curve.

**Prediction**: "Retraining costs $10K and your model loses $500/day in accuracy-related revenue losses. Your current retraining cadence is every 30 days. What is the optimal interval?"
- A: 7 days (retrain weekly to stay fresh)
- **B: ~6 days (T* = sqrt(2 x 10000 / 500) = 6.3 days)**
- C: 14 days (biweekly is a reasonable compromise)
- D: 30 days (your current cadence is fine)

**Common wrong answer**: C or D. Students guess round numbers based on intuition rather than computing the formula. They dramatically underestimate how frequently high-drift, high-value models should retrain.

**Why wrong**: The square-root law means T* is much more sensitive to drift rate and value than students expect. When C_drift = $500/day, even expensive retraining ($10K) should happen every 6 days.

**Instrument**: U-shaped total annual cost curve. Sliders: drift rate ($100--$5000/day in accuracy losses), retraining cost ($1K--$50K), accuracy threshold (80%--95%). The T* marker moves along the curve. Three cost components visible: checkpoint overhead (decreasing hyperbola), staleness cost (increasing line), total (U-curve). Failure state when accuracy at T* falls below minimum threshold.

**mlsysim grounding**: Engine.solve(model, hardware, is_training=True) computes the actual training time and GPU-hours for retraining, converting to dollar cost via hardware.unit_cost. The retraining cost C is grounded in real hardware: C = training_hours x GPU_hourly_rate x N_GPUs. For a ResNet-50 retrain on 4xA100: Engine.solve(ResNet50, A100, is_training=True, batch_size=64) gives step time, from which total training time = steps x step_time.

**Transition to C**: "The optimal cadence for one deployment target is clear. But the same model deployed on Cloud, Edge, and Mobile has different retraining costs at each tier. How does T* change across the deployment spectrum?"

---

### Part C -- Deployment Cost Asymmetry (Same Model, Different Economics) (~12 min)

**Concept**: The same model with identical accuracy requirements produces dramatically different optimal retraining intervals across Cloud, Edge, and Mobile purely because of deployment cost differences. Cloud T* of 7--14 days vs. Edge T* of 60--90 days, driven by T* scaling with sqrt(cost), not cost itself. A 100x cost difference produces only a 10x cadence difference.

**Prediction**: "Retraining costs $1K on Cloud and $100K on Edge (including OTA update logistics). If Cloud T* is 7 days, what is Edge T*?"
- A: 700 days (100x cost = 100x interval)
- B: 140 days (linearly proportional to cost ratio)
- **C: ~70 days (sqrt(100) = 10x interval)**
- D: 14 days (edge needs more frequent updates because devices are less reliable)

**Common wrong answer**: A or B. Students assume linear scaling between cost and interval. The square-root law is deeply counterintuitive.

**Instrument**: Three-column comparison: Cloud / Edge / Mobile. Per-environment sliders for retraining cost and drift rate. T* computed independently for each. The comparison panel highlights that a 100x cost difference produces only a 10x cadence difference. A "square root surprise" annotation shows the formula.

**mlsysim grounding**: Engine.solve() for training on each deployment target provides the base training cost. Cloud: Engine.solve(model, H100, is_training=True). Edge: same plus OTA deployment cost multiplier (documented in chapter). The sqrt relationship is the core formula.

**Transition to D**: "You now have a retraining cadence for each tier. But what happens when you defer retraining too long? The technical debt compounds -- not linearly, but as a cascade."

---

### Part D -- The Debt Cascade (~12 min)

**Concept**: Technical debt in ML systems compounds through two mechanisms simultaneously: (1) correction cascades, where patching Model A with a post-hoc fix creates a dependency that makes Model B's predictions less reliable, and (2) the snowball effect, where each deferred maintenance item increases the cost of all future maintenance. A system with 3 deferred retraining cycles does not have 3x the debt -- it has ~5x, because each cycle of drift makes the next retraining harder (more distribution shift to bridge, more stale features to update, more downstream models affected).

**Prediction**: "You defer retraining for 3 consecutive cycles (3 x T*). What is the total accumulated accuracy loss compared to a single missed cycle?"
- A: 3x (linear accumulation)
- B: 4x (slightly superlinear)
- **C: ~5--6x (debt compounds due to cascading effects)**
- D: 9x (quadratic in number of missed cycles)

**Common wrong answer**: A. Students assume drift accumulates linearly. They do not account for the compounding effects: each cycle of drift moves the production distribution further from the training distribution, making the next retraining less effective (larger domain gap) and the downstream models less calibrated.

**Why wrong**: Debt compounds through two mechanisms. First, accuracy loss is not linear in drift magnitude -- the model's performance degrades faster as the distribution shifts further from training. Second, correction cascades mean that patching one model after significant drift often requires retraining or recalibrating dependent models, multiplying the cost.

**Instrument**: Timeline visualization showing 1, 2, and 3 missed retraining cycles. For each scenario: accuracy trajectory, accumulated revenue loss (area under the curve), and a "cascade counter" showing how many downstream models are affected. Controls: number of dependent downstream models (0--5), drift rate, base retraining cost. A "debt multiplier" gauge shows total accumulated cost / (N_missed x single_cycle_cost).

**mlsysim grounding**: The base drift model from Part A provides accuracy decay. The cascade model: for each downstream model, accuracy_downstream(t) = f(accuracy_upstream(t)), where f captures the dependency. Retraining cost after N missed cycles: C_retrain(N) = C_base x (1 + cascade_factor x N_downstream), because each downstream model may need recalibration. The total debt = integral of accumulated accuracy loss + sum of cascade retraining costs.

---

### Synthesis (~6 min)

**Mission**: You operate a fraud detection system with 3 dependent models (transaction scoring, account risk, merchant reputation). The system is deployed on Cloud (retrain weekly) and Edge (retrain quarterly).

Write a complete monitoring and retraining policy:
1. Which PSI thresholds trigger alerts for each feature type (Part A)
2. Optimal retraining cadence for each deployment tier, with cost justification (Parts B and C)
3. Maximum number of retraining cycles you can safely defer before the debt cascade makes recovery uneconomical (Part D)
4. The rollback strategy when a retrained model performs worse than the stale one

**Design Ledger entry**: Record your Part A prediction about the "green dashboard." The gap between infrastructure health and model health is the defining failure mode of ML systems.

---
---

## Lab 15: No Free Fairness
**Story arc**: Fairness is mathematically impossible to achieve perfectly, fixing it costs accuracy, explaining it costs latency, and all of it costs carbon -- and none of these costs are optional.
**Time budget**: 48 min total -- A(12) + B(12) + C(10) + D(8) + Synthesis(6)

### Part A -- The Fairness Illusion (~12 min)

**Concept**: The Chouldechova impossibility theorem: when base rates differ between demographic groups, a calibrated classifier with equal accuracy on both groups is mathematically guaranteed to produce unequal error rates (FPR, FNR, PPV). Equal accuracy does not mean equal treatment. You cannot have equal FPR, equal FNR, and equal PPV simultaneously when base rates differ. This is not an engineering limitation -- it is a mathematical impossibility.

**Prediction**: "You train a loan approval model with 92% accuracy for both Group A (30% base rate) and Group B (10% base rate). Are the false positive rates equal?"
- A: Yes -- equal accuracy guarantees equal error rates
- B: Nearly equal -- small differences from random noise
- **C: No -- Group B's FPR is 3x higher than Group A's despite equal accuracy**
- D: No -- Group A's FPR is higher because it has more positive cases

**Common wrong answer**: A. Students believe accuracy is a sufficient fairness metric. "If the model is equally accurate for both groups, it must be equally fair."

**Why wrong**: When base rates differ, equal accuracy requires different threshold placements for each group. Group B (10% base rate) has far more true negatives, so achieving 92% accuracy requires a lower threshold, which increases the false positive rate. The math is inescapable.

**Instrument**: Base rate sliders for two groups (5%--50%), shared classification threshold slider (0.10--0.90). Grouped bar chart of per-group metrics (accuracy, TPR, FPR, FNR, PPV). Color-coded gap cards flag violations (>5% gap turns yellow, >10% turns red). The key visual: accuracy bars are equal height while FPR bars are dramatically different.

**mlsysim grounding**: This part uses pure statistical computation (not Engine.solve()), but the scenario is framed around a deployment context: the model runs on Edge hardware with a latency SLO that constrains threshold search. Engine.solve(model, hardware) provides the inference latency that determines how many threshold evaluations can run within the SLO.

**Transition to B**: "You have just proved that perfect fairness is impossible. The next question: how much accuracy must you sacrifice to *reduce* (not eliminate) the disparity?"

---

### Part B -- The Price of Fairness (~12 min)

**Concept**: The fairness-accuracy Pareto frontier quantifies the accuracy cost of different fairness criteria. The frontier has a "sweet spot" where the first large fairness gains cost only 3--5% accuracy, then a "cliff" where strict equality demands disproportionate sacrifice. Three mitigation strategies (threshold adjustment, reweighting, adversarial debiasing) trace different paths along this frontier.

**Prediction**: "You apply equalized odds constraints to reduce the FPR gap from 15% to 5%. How much accuracy do you lose?"
- A: ~1% (fairness is nearly free at this level)
- **B: ~3--5% (the sweet spot of the Pareto frontier)**
- C: ~10% (significant accuracy sacrifice)
- D: ~15% (proportional to the gap reduction)

**Common wrong answer**: A (optimistic) or D (pessimistic). Students do not have a mental model for the shape of the Pareto frontier. Some assume fairness is free; others assume it is proportionally expensive.

**Why wrong**: The Pareto frontier has a specific shape: concave near the sweet spot (large fairness gains for small accuracy cost) and steep near strict equality (small fairness gains for large accuracy cost). The 15% to 5% gap reduction is in the sweet spot; the 5% to 0% reduction is on the cliff.

**Instrument**: Fairness criterion radio buttons (Demographic Parity / Equal Opportunity / Equalized Odds). Mitigation method dropdown (Threshold Adjustment / Reweighting / Adversarial Debiasing). Pareto frontier chart with three annotated points: A (unconstrained, highest accuracy, largest disparity), B (sweet spot), C (strict equality, lowest accuracy). Students select a target fairness level and see the accuracy cost. Failure state when equalized odds gap exceeds 10 percentage points.

**mlsysim grounding**: The Pareto frontier is computed from the statistical model in Part A. The accuracy cost at each fairness level is deterministic given the base rates and the mitigation method.

**Transition to C**: "Reducing disparity costs accuracy. But there is another cost: explaining the model's decisions to affected users. And explanations cost compute."

---

### Part C -- The Explainability Tax (~10 min)

**Concept**: Regulatory requirements (GDPR right to explanation, ECOA adverse action notices) mandate that certain model decisions be accompanied by per-instance explanations. SHAP explanations for a single prediction require running the model hundreds of times (one per feature permutation). For a loan approval model with 50 features, a SHAP explanation requires ~50 additional forward passes, adding 50x latency overhead to the serving path. Simpler methods (LIME, feature importance) are faster but less faithful.

**Prediction**: "Your loan model has 50 input features. A SHAP explanation requires computing the model output for each feature permutation subset. How much does explanation add to inference latency?"
- A: ~2x (modest overhead)
- B: ~10x (significant but manageable)
- **C: ~50x (one forward pass per feature)**
- D: ~2500x (all permutation subsets)

**Common wrong answer**: A. Students think of explanations as a lightweight post-hoc step, not as additional model executions.

**Why wrong**: SHAP computes marginal contributions by running the model with each feature masked/permuted. For 50 features, this requires approximately 50 forward passes (for the Kernel SHAP approximation; exact SHAP requires 2^50 which is infeasible). Each forward pass has the same latency as the original prediction.

**Instrument**: Feature count slider (10--200), explanation method dropdown (SHAP / LIME / Feature Importance / None). Latency waterfall showing: base inference + explanation overhead. SLO compliance check against a 100 ms budget. At 50 features with SHAP, the total latency is 50x base, easily blowing a 100 ms SLO. Method comparison table shows fidelity vs. compute cost.

**mlsysim grounding**: Engine.solve(model, hardware) provides base inference latency. Explanation overhead = base_latency x explanation_multiplier, where multiplier depends on method and feature count: SHAP ~ N_features, LIME ~ 100 (fixed sample count), Feature Importance ~ 1 (no additional forward passes).

**Transition to D**: "Fairness costs accuracy. Explanations cost latency. And every model you train, retrain, or serve costs energy -- which becomes carbon."

---

### Part D -- The Carbon Cost of Responsibility (~8 min)

**Concept**: Every fairness mitigation, every retraining cycle, every SHAP explanation consumes energy that translates to carbon emissions. A model that retrains weekly to maintain fairness across shifting demographics emits 52x more carbon than one trained once. The question is not whether to pay this cost, but how to budget it against the harms of an unfair or unexplainable system.

**Prediction**: "Your fair model retrains weekly (to track demographic shifts) and computes SHAP explanations for 10% of predictions (regulatory requirement). How does total carbon compare to a train-once, no-explanation baseline?"
- A: ~2x (modest increase)
- B: ~10x (significant but necessary)
- **C: ~60x (52x from weekly retraining + 5x from explanations on 10% of traffic)**
- D: ~100x (even higher due to compounding)

**Common wrong answer**: A or B. Students dramatically underestimate the cumulative carbon cost of operational ML.

**Instrument**: Carbon budget dashboard. Controls: retraining frequency (weekly / monthly / quarterly / never), explanation coverage (0% / 10% / 50% / 100%), explanation method (SHAP / LIME / None). Stacked bar showing: training carbon (one-time) + retraining carbon (cumulative over 1 year) + serving carbon (per-query x query volume) + explanation carbon. A carbon intensity toggle (grid mix: clean hydro / mixed / coal-heavy) shows geographic impact.

**mlsysim grounding**: Engine.solve(model, hardware, is_training=True).energy provides per-training-run energy. Serving energy from Engine.solve(model, hardware).energy x queries_per_year. Carbon = energy x grid_carbon_intensity (gCO2/kWh, parameterized by region).

---

### Synthesis (~6 min)

**Mission**: You are deploying a loan approval model under ECOA regulations. You must present a system specification to your compliance officer that addresses:

1. Which fairness criterion you chose and why (Part A: impossibility constrains your options)
2. The accuracy cost of your fairness constraint and why it is at the sweet spot (Part B)
3. Your explanation strategy: which method, for which percentage of predictions, and the latency impact (Part C)
4. The annual carbon budget of the entire responsible-AI stack: training + retraining + serving + explanations (Part D)

Frame each choice as a trade-off, not a solution. There is no free fairness.

**Design Ledger entry**: Record your Part A prediction about equal accuracy implying equal fairness. The impossibility theorem is the most counterintuitive result in the lab series.

---
---

## Lab 16: The Architect's Audit (Capstone)
**Story arc**: Everything you have learned in 15 labs collapses into one question: given a real system with real constraints, where do you invest your engineering effort? The answer requires every tool in your toolkit -- and the discipline to recognize which tool does not apply.
**Time budget**: 58 min total -- A(12) + B(12) + C(10) + D(8) + E(10) + Synthesis(6)

### Part A -- The Cost of a Token (Calibration) (~12 min)

**Concept**: The Iron Law decomposition for single-token LLM inference reveals that memory-to-compute time ratio is ~40x on H100, not the ~5x most students expect. Autoregressive decoding has arithmetic intensity of ~1 FLOP/byte, placing it deep in the memory-bandwidth-bound regime. This means compute kernel optimization is futile without addressing data movement. This Part calibrates students' quantitative intuition one final time.

**Prediction**: "For Llama-2 70B at batch=1 on H100, what fraction of token generation time is spent waiting for memory?"
- A: ~50% (roughly balanced)
- B: ~75% (memory dominates moderately)
- C: ~90% (memory dominates strongly)
- **D: ~98% (memory access is essentially all of the time)**

**Common wrong answer**: A or B. Even after 15 labs, students underestimate the severity of the memory wall for autoregressive decoding.

**Instrument**: Model size selector (7B / 13B / 70B), precision toggle (FP32 / FP16 / INT8 / INT4), batch size slider (1--256). Iron Law decomposition bar chart: memory time (blue) vs. compute time (orange) vs. overhead (gray). Roofline operating point dot. At batch=1, memory dominates by 40x. At batch=64, the bars approximately equalize -- making the crossover from memory-bound to compute-bound visible.

**mlsysim grounding**: Engine.solve(Llama2_70B, H100, batch_size=1, precision="fp16") provides latency_compute, latency_memory, and arithmetic_intensity directly. The ratio latency_memory / latency_compute is the key number. Sweeping batch size from 1 to 256 shows the crossover.

**Transition to B**: "You now know the cost of a single token. But a deployed system is not just a single inference -- it is a living organism that degrades, drifts, and accumulates debt. The Conservation of Complexity governs what happens next."

---

### Part B -- The Conservation of Complexity (~12 min)

**Concept**: Complexity in an ML system cannot be destroyed, only moved between Data, Algorithm, and Machine. Quantizing MobileNetV2 to INT8 reduces model complexity (Algorithm axis) but creates monitoring debt (Machine axis) -- without monitoring investment, accuracy degrades ~1.3%/month while all infrastructure metrics remain green. Students must track 4--5 key invariants simultaneously to see that optimizing one axis shifts the burden to another.

**Prediction**: "You quantize MobileNetV2 to INT8 and deploy on mobile. After 6 months without monitoring, what has happened?"
- A: Nothing -- INT8 is stable and the model works fine
- B: Latency increased due to hardware wear
- **C: Accuracy silently degraded ~8% while all other metrics stayed green**
- D: The model crashed from numerical instability

**Common wrong answer**: A. Students think deployment is the end of the engineering process. They do not connect quantization (which reduces model robustness to distribution shift) with monitoring investment (which detects that shift).

**Instrument**: Configuration panel: architecture dropdown, quantization level, monitoring investment level (None / Basic / Comprehensive). Time slider (0--12 months deployed). 5-axis radar chart: accuracy, latency, memory, power, drift resilience. Without monitoring, the accuracy axis silently shrinks while all others stay green. An invariant activation table shows which of 4--5 invariants are triggered: (1) Amdahl's Law, (2) Memory Wall, (3) Silent Degradation, (4) Conservation of Complexity, (5) No Free Fairness.

**mlsysim grounding**: Engine.solve(MobileNetV2, iPhone15Pro, precision="int8") provides the initial latency, memory, and energy. The drift model from Lab 14 Part A provides accuracy decay over time. The monitoring investment parameter determines whether PSI alerts fire (Lab 14) or not.

**Transition to C**: "You have seen how one system degrades. Now turn the lens on yourself: across 15 labs, where did *your* predictions consistently fail?"

---

### Part C -- Design Ledger Archaeology (~10 min)

**Concept**: A student's own prediction history across Labs 01--15 reveals systematic blind spots -- which invariants their intuition consistently underweights. The gap between self-assessed and data-revealed weaknesses is itself an engineering insight: mental models degrade just like deployed models.

**Prediction**: "Before viewing your data, which invariant do you think you most consistently underestimated across all 15 labs?"
- Self-assessment via dropdown: Amdahl's Law / Memory Wall / Silent Degradation / Conservation of Complexity / No Free Fairness

(No "correct" answer -- the reveal comes from comparing self-assessment to actual data.)

**Instrument**: 5-axis radar chart computed from Design Ledger history (prediction error magnitude per invariant category). The student's self-assessment is overlaid as a second polygon. The gap between the two reveals metacognitive blind spots.

**Fallback (Design Ledger unavailable)**: If a student does not have complete Ledger data (e.g., they skipped labs or started late), the instrument loads a pre-generated "typical student" dataset showing the median prediction error profile across the class. The student can still self-assess and compare against the class median. A banner notes: "Using class median data. Complete your Design Ledger entries for a personalized analysis."

**mlsysim grounding**: The Design Ledger is a lab-framework feature (mlsysim.labs.state), not an Engine.solve() computation. The radar chart aggregates prediction errors from the ledger entries written at the end of each prior lab.

**Transition to D**: "Your personal blind spots are one piece of the puzzle. The other piece: the fundamental limits that constrain *every* architect. Remember the Amdahl Ceiling from Lab 12?"

---

### Part D -- The Amdahl Ceiling Revisited (~8 min)

**Concept**: This Part explicitly calls back to Lab 12 Part A (The Amdahl Ceiling), but now applies it to a complete system design rather than a single pipeline. The question shifts from "what is the speedup?" to "where should I invest my engineering effort?" Amdahl's Law is not just a speedup formula -- it is a resource allocation framework. The component with the largest time fraction has the highest-leverage optimization.

**Prediction**: "Your ML system has 4 components: preprocessing (35%), inference (40%), postprocessing (15%), logging (10%). You can invest engineering effort to optimize ONE component by 5x. Which yields the largest end-to-end speedup?"
- A: Preprocessing (largest non-inference component)
- **B: Inference (largest overall component: 1/(0.6 + 0.4/5) = 1.47x)**
- C: Postprocessing (seems like low-hanging fruit)
- D: It does not matter -- 5x on any component yields roughly the same system speedup

**Common wrong answer**: D. Students who remember Amdahl's Law from Lab 12 may think "it is always about the serial fraction" without computing which component IS the serial fraction.

**Instrument**: System component breakdown with editable time fractions (must sum to 100%). "Optimize" dropdown selects one component, speedup slider (1x--20x). Amdahl speedup readout and a "dollars per 1% improvement" cost efficiency metric. The visual callback to Lab 12: same Amdahl saturation curve, but now with real system components instead of abstract fractions.

**mlsysim grounding**: Engine.solve() provides the inference time component. The other components (preprocessing, postprocessing, logging) are parameterized. Total system time = sum of all components, with the optimized component divided by the speedup factor.

**Transition to E**: "You know where to invest effort. But what happens when you optimize one component and it shifts the constraint to another? This is the Constraint Cascade -- and it is the final lesson of the course."

---

### Part E -- The Constraint Cascade (~10 min)

**Concept**: Optimizing one constraint in an ML system does not solve the problem -- it moves the binding constraint to the next-tightest axis. Quantizing to INT4 solves the memory constraint but creates a precision constraint. Increasing batch size solves the throughput constraint but creates a latency constraint. Retraining more frequently solves the drift constraint but creates a carbon constraint. This is the Conservation of Complexity made dynamic: the system is always constrained, and the architect's job is to choose which constraint to live with.

**Prediction**: "You quantize Llama-2 70B from FP16 to INT4, solving the memory constraint on a single H100 (70B x 0.5 bytes = 35 GB < 80 GB HBM). What is the new binding constraint?"
- A: Compute (INT4 is slower per operation)
- **B: KV cache memory (weights fit, but KV cache at long context still fills HBM)**
- C: Accuracy (INT4 quality is insufficient)
- D: No constraint -- the system now works

**Common wrong answer**: D. Students think solving the immediate constraint means the system works. They do not trace the cascade to the next binding constraint.

**Why wrong**: INT4 weights for 70B fit in 35 GB. But KV cache for a 32K context in FP16 is still ~40 GB. Total: 75 GB, leaving only 5 GB headroom. At 64K context, the system OOMs again. The constraint has moved from "model too large" to "context too long."

**Instrument**: Constraint dashboard with 5 axes: Memory, Latency, Accuracy, Carbon, Fairness. Students apply optimizations one at a time via checkboxes: INT4 quantization, structured pruning, knowledge distillation, continuous retraining, SHAP explanations. Each optimization resolves one constraint (turns it green) but shifts the binding to another (turns it yellow, then red). The cascade is visible as a chain of constraint migrations.

**mlsysim grounding**: Engine.solve() at each configuration provides memory_footprint, latency, and energy. Feasibility checks (feasible field) track which constraints are satisfied. The cascade: Engine.solve(Llama2_70B, H100, precision="int4", batch_size=1, seq_len=32768) shows the KV cache exceeding available memory even with INT4 weights.

---

### Synthesis (~6 min)

**Mission**: You are the ML systems architect for a startup deploying Llama-2 70B as a chat service on a fleet of 8xH100 nodes. Write a deployment specification that addresses:

1. Precision and quantization strategy (Part A: what is the actual cost of a token?)
2. Monitoring investment to prevent silent degradation (Part B: Conservation of Complexity)
3. Your personal blind spot and how you will compensate for it (Part C: Design Ledger)
4. Where to invest your next engineering sprint (Part D: Amdahl's resource allocation)
5. The constraint cascade: which constraint are you choosing to live with? (Part E)

This is the "graduation specification." It integrates every lab into one document. Students who complete it have demonstrated quantitative reasoning about ML systems across the full stack.

**Design Ledger entry (final)**: Record your cumulative prediction accuracy across all 16 labs. Compute your mean absolute error per invariant category. This is your engineering calibration score.

---
---

# Cross-Lab Reference Table

| Lab | Title | Time | Parts | Key Invariants |
|-----|-------|------|-------|---------------|
| 09 | Data Selection Paradox | 52 min | A(ICR) + B(Selection Inequality) + C(Preprocessing Tax) + D(Chinchilla) + Synthesis | Memory Wall, Amdahl |
| 10 | Compression Paradox | 56 min | A(Quantization) + B(Pruning Trap) + C(Pareto) + D(Energy) + E(Distillation) + Synthesis | Memory Wall, Conservation of Complexity |
| 11 | Hardware Roofline | 52 min | A(Roofline) + B(Fusion) + C(Balance Shift) + D(Energy Roofline) + E(Tiling) + Synthesis | Memory Wall, Amdahl |
| 12 | Benchmarking Trap | 52 min | A(Amdahl) + B(Thermal) + C(Multi-Metric) + D(Tail Latency) + Synthesis | Amdahl, Silent Degradation |
| 13 | Tail Latency Trap | 50 min | A(Queuing) + B(Batching) + C(KV Cache) + D(Cold Start) + Synthesis | Memory Wall, Amdahl |
| 14 | Silent Degradation | 54 min | A(PSI Drift) + B(Retraining Cadence) + C(Cost Asymmetry) + D(Debt Cascade) + Synthesis | Silent Degradation, Conservation of Complexity |
| 15 | No Free Fairness | 48 min | A(Impossibility) + B(Pareto) + C(Explainability Tax) + D(Carbon) + Synthesis | No Free Fairness, Conservation of Complexity |
| 16 | Architect's Audit | 58 min | A(Token Cost) + B(Conservation) + C(Ledger) + D(Amdahl Revisited) + E(Cascade) + Synthesis | All invariants |

# Redundancy Resolution

| Concept | Owned By | Other Labs Reference (Do Not Re-teach) |
|---------|----------|--------------------------------------|
| Amdahl's Law | Lab 12 Part A | Lab 16 Part D explicitly references Lab 12 |
| Precision effects (quantization) | Lab 10 Part A | Lab 13 Part E DROPPED (was redundant) |
| Silent degradation (PSI drift) | Lab 14 Part A | Lab 16 Part B references Lab 14 |
| Energy/carbon | Lab 10 Part D (per-inference) | Lab 15 Part D (system-level) -- different scope |
| Roofline model | Lab 11 Part A | Lab 12 Part D uses it but does not re-teach |
| TCO | MOVED to Lab 12/13 territory | Lab 15 Part C DROPPED (was TCO) |
| Curriculum learning | DROPPED | Lab 09 Part C REPLACED with Preprocessing Tax |
