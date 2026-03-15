# Lab Parts Proposals — Master Review Document

Generated: 2026-03-15
Source: 8 lab-designer agents reading all chapter QMD files

---

# VOLUME I: Introduction to Machine Learning Systems

## Lab 01: The AI Triad
Chapter: `introduction.qmd`

### Proposed Parts:

- **Part A -- The D-A-M Triad: Three Axes, One System** (~12 min)
  Concept: The AI Triad (Data, Algorithm, Machine) are interdependent -- optimizing one axis shifts the bottleneck to another. Students diagnose a system that is performing poorly and must identify which of the three axes is the binding constraint. The key insight: the axes are not independent knobs.
  Interaction: Students are given three system scenarios (a recommendation system with stale data, a vision model with too few FLOPS, and a language model that exceeds device memory). For each, they predict which D-A-M axis is the bottleneck, then toggle a "fix" for the wrong axis and observe zero improvement. Fixing the correct axis resolves the problem. This establishes D-A-M as a diagnostic framework from the first minute of the lab.

- **Part B -- The Iron Law Surprise** (~12 min)
  Concept: The Iron Law of ML Systems (T = D_vol/BW + O/(R_peak * eta) + L_lat) decomposes performance into three physical terms. At batch=1 on an H100, ResNet-50 inference is memory-bound, not compute-bound -- a result that violates the naive assumption that GPUs are "compute machines."
  Interaction: Students predict which Iron Law term dominates for ResNet-50 at batch=1 on an H100. A batch size slider lets them watch the bottleneck shift from memory to compute as batch size increases. Switching to a Cortex-M7 triggers an OOM failure state (100 MB model vs. 512 KB SRAM), viscerally demonstrating that the same model demands fundamentally different engineering at different points on the deployment spectrum.

- **Part C -- The Silent Decay** (~12 min)
  Concept: The Degradation Equation (Accuracy(t) = Accuracy_0 - lambda * D(P_t || P_0)) captures how ML systems fail silently as data distributions drift. All infrastructure metrics stay green while model accuracy drops. This is the defining failure mode that separates ML systems from traditional software.
  Interaction: A split dashboard shows infrastructure metrics (uptime, latency, error rate -- all green) on the left and model accuracy (decaying) on the right. A time slider advances months since deployment. Students predict accuracy at 6 months, then watch the gap between "system healthy" and "model degraded" widen. A drift rate slider (lambda) lets them explore how model sensitivity affects decay speed.

- **Part D -- The Deployment Spectrum** (~9 min)
  Concept: The deployment spectrum spans 9 orders of magnitude in compute and memory (Cloud to TinyML). This gap is so vast that a universal ML software stack is physically impossible -- each deployment tier requires fundamentally different model architectures. This replaces the previous "Scale Blindspot" framing with a D-A-M-grounded exploration of why deployment context reshapes all three axes.
  Interaction: Students estimate the compute ratio between an H100 and a Cortex-M7 microcontroller. A scale toggle switches the bar chart between linear (where TinyML is invisible) and log scale (where both are visible). The 1,000,000x compute gap and 160,000x memory gap are the visual punch. Students then see how this gap forces different choices on every D-A-M axis: different data (compressed), different algorithms (quantized), different machines (specialized silicon).

---

## Lab 02: The Physics of Deployment
Chapter: `ml_systems.qmd`

### Proposed Parts:

- **Part A -- The Memory Wall Revelation** (~12 min)
  Concept: Arithmetic Intensity determines whether a workload is compute-bound or memory-bound. At AI = 5 FLOPs/Byte (far below the H100 ridge point of ~295), a 6x GPU upgrade (A100 to H100) yields only ~8% latency improvement because the Memory term dominates. The correct diagnostic: identify the binding term before spending money.
  Interaction: An Arithmetic Intensity slider (1-400 FLOPs/Byte) lets students watch a latency waterfall chart shift between memory-bound and compute-bound regimes. The ridge point is marked. Students predict the improvement from a $2M GPU upgrade, then discover it yields ~8% at low AI. A context toggle (Cloud/Edge) shows the same physics on different hardware with different ridge points.

- **Part B -- The Light Barrier** (~12 min)
  Concept: The speed of light in fiber (~200,000 km/s) sets an irreducible latency floor that no hardware upgrade can reduce. At 3,000 km, the minimum round-trip is 30 ms -- 3x the 10 ms autonomous vehicle safety SLA. This physical constraint is why Edge ML exists as a deployment paradigm: not preference, but physics.
  Interaction: A distance slider (0-5,000 km) lets students watch propagation delay consume the SLA budget. A stacked latency bar shows propagation + compute + overhead against a 10 ms SLA reference line. When the bar crosses the line, a failure banner reads "SLA VIOLATED -- the speed of light cannot be optimized." Students discover the threshold distance (~1,000 km) below which cloud deployment becomes physically possible.

- **Part C -- The Power Wall: Why You HAVE to Pick** (~12 min)
  Concept: Each deployment paradigm (Cloud, Edge, Mobile, TinyML) exists because of a distinct physical constraint, and each comes with both opportunities and challenges. A mobile device running at 3-5W will thermally throttle from 60 FPS to 15 FPS within a minute. There is no "one-size-fits-all" deployment -- physics forces a choice, and that choice carries trade-offs in both directions.
  Interaction: Students select a deployment target (Cloud/Edge/Mobile/TinyML) and a model (ResNet-50, MobileNet, KWS). A constraint dashboard shows which physical walls are binding: light barrier (latency), power wall (thermal throttle), memory wall (capacity). Students discover that each paradigm enables things the others cannot (TinyML: years on battery; Cloud: trillion-parameter models) AND forbids things the others allow (TinyML: no complex models; Cloud: no sub-10ms local response). A "deploy" button checks feasibility and shows pass/fail with the binding constraint identified.

- **Part D -- The Energy of Transmission** (~9 min)
  Concept: For battery-powered devices, transmitting 1 MB of raw data to the cloud costs ~1,000x more energy than processing it locally on an NPU. This energy asymmetry is why TinyML exists -- even with infinite cloud speed, the energy wall makes cloud offloading physically impossible for always-on sensing.
  Interaction: Students compare the energy budget of "send to cloud" vs "process locally" for a 1-second audio clip on a battery sensor. Sliders adjust data size and transmission energy. The ratio (cloud vs. local energy) is displayed as a large number. Students discover the 1,000x gap and see how it maps directly to the Iron Law's D_vol/BW term when "BW" is wireless bandwidth and cost is measured in joules rather than seconds.

---

## Lab 03: The Constraint Tax -- Orchestrating the ML Lifecycle
Chapter: `ml_workflow.qmd`

### Proposed Parts:

- **Part A -- Constraint Propagation: The DR Clinic Disaster** (~12 min)
  Concept: The cost of discovering a deployment constraint at lifecycle stage N grows exponentially as 2^(N-1). The DR screening case study makes this concrete: a team builds a 4 GB model for 95% accuracy, then discovers at deployment that rural clinic tablets have 512 MB of RAM. Five months of work is invalidated. The entire system must be rethought, not just the model.
  Interaction: Students use the DR screening scenario as the running example. A "discovery stage" slider (1-6) shows the exponential cost curve. At each stage, the display shows what artifacts must be rebuilt: at Stage 5, it is data pipelines + model architecture + training + evaluation -- not just the model. A failure banner at Stage 5+ shows "PROJECT RESET REQUIRED -- 145 of 150 days at risk." The key message: deployment constraints must be specified at Stage 1 (Problem Definition), where they cost 1x, not Stage 5 where they cost 16x.

- **Part B -- The Iteration Velocity Race** (~12 min)
  Concept: Iteration velocity dominates starting accuracy over realistic development windows. Using the chapter's DR screening scenario: Team A (large ensemble, 95% start, 1-week cycle) vs. Team B (lightweight edge model, 90% start, 1-hour cycle). Team B overtakes Team A because 100 experiments explore more of the design space than 26 experiments.
  Interaction: A dual-line accuracy chart shows both teams over a 26-week window. Students predict which team wins, then watch Team B's line cross Team A's. Sliders for cycle time and starting accuracy let students find the crossover point and the failure condition where iteration speed is no longer sufficient. The DR framing makes this about a real system decision, not an abstract race.

- **Part C -- The Whole System View: Where Does the Time Go?** (~12 min)
  Concept: Data-related activities consume 60-80% of ML project effort. Model development, despite receiving the most research attention, is only 10-20%. The DR case study illustrates: expert ophthalmologist annotation, image quality validation across clinics, equipment variation, privacy compliance -- these are the iceberg below the waterline. The ML workflow is the entire system, not just model optimization.
  Interaction: Students allocate a 10-person team across project phases (data, modeling, deployment/ops). A stakeholder message simulates project outcomes: underfunding data leads to "team ran out of clean training data in month 2, three modelers idle." The DR case study shows specifically where that 60-80% goes: data collection across heterogeneous clinics, labeling by expert ophthalmologists, quality validation for regulatory compliance, distributed data infrastructure for clinics with 2 Mbps uplinks. A side-by-side bar compares student allocation vs. industry reality.

- **Part D -- Feedback Loops: Why the Lifecycle Never Ends** (~9 min)
  Concept: Unlike traditional software where later stages rarely influence earlier ones, ML systems require continuous feedback loops. Monitoring drives data collection changes, deployment experience reshapes architecture choices, and production data reveals distributional properties invisible in development. The DR system illustrates: scaling from 5 pilot clinics to 200 clinics changes everything -- new camera equipment, new demographics, new failure modes. Problem definitions evolve.
  Interaction: A timeline simulation of the DR system scaling from pilot to production. Students toggle "months since launch" and watch new feedback events appear: "Month 3: Camera model changed at clinic #47 -- accuracy drops 8% at that site." "Month 6: New demographic subgroup has 40x higher error rate." Each event triggers a feedback arrow back to an earlier lifecycle stage. Students see that deployment is not the end but the beginning of the feedback loop. A counter tracks total iteration cycles required, converging on the chapter's "4-8 complete iterations for production readiness."

---

## Lab 04: The Data Gravity Trap
Chapter: `data_engineering.qmd`

### Proposed Parts:

- **Part A -- The Feeding Tax: When Your GPU Starves** (~12 min)
  Concept: A standard cloud disk (250 MB/s) leaves an A100 GPU idle over 95% of the time during ResNet-50 training. The bottleneck is not compute but I/O -- the data pipeline cannot feed the accelerator fast enough. The Iron Law's D_vol/BW term dominates. Buying more GPUs just means more expensive hardware sitting idle.
  Interaction: Students predict GPU utilization given standard storage. A storage type dropdown (HDD/SSD/NVMe/RAM disk) and DataLoader worker slider change the I/O bandwidth. A batch timeline bar shows GPU compute (green) vs. I/O wait (red). At default (HDD), the GPU computes less than 5% of the time. Students discover that the fix is faster storage + parallel loading, not more GPUs. A gauge turns green only when utilization exceeds 80%.

- **Part B -- Data Gravity: Move the Compute, Not the Data** (~12 min)
  Concept: Moving 50 TB across cloud regions costs $4,000 in egress alone -- 20x the compute cost of the training job itself. The physics: T = D_vol/BW governs transfer time, and cloud egress pricing creates an economic gravity that pins computation to wherever data resides. For datasets above a few TB, it is always cheaper to move GPUs to the data.
  Interaction: Students predict transfer time and cost for 50 TB over a 100 Gbps link. Sliders for dataset size and network bandwidth reveal the linear scaling. A cost comparison chart shows "move data" (egress + remote GPU) vs. "move compute" (local GPU + small premium). At default settings, the failure state triggers immediately: egress cost exceeds compute cost. Students find the crossover dataset size where the strategies break even.

- **Part C -- Data Cascades: The 2% Error That Ate 15% Accuracy** (~12 min)
  Concept: Data quality errors introduced at early pipeline stages amplify through downstream stages. A 2% schema error at ingestion (e.g., zip code loses leading zero) compounds to ~15% accuracy degradation at deployment. Detection takes a median 4 weeks, during which the model silently makes degraded predictions on every request. This is the Pipeline Jungle from the chapter: without data contracts and schema validation, upstream changes cause catastrophic silent failures downstream.
  Interaction: Students predict the accuracy impact of a 2% ingestion error. An error injection point selector and pipeline depth control let them trace how errors amplify at each stage. A cascade amplification chart shows error rate growing geometrically through the pipeline. A silent degradation timeline shows accuracy declining over weeks before detection. The key visual: the gap between "when the error was introduced" and "when it was detected" -- the 4-week median detection latency where damage accumulates silently.

- **Part D -- The False Positive Trap: When 99% Is Not Enough** (~9 min)
  Concept: For always-on KWS (Keyword Spotting) systems, standard accuracy metrics are meaningless. An always-on device evaluates ~2.6 million 1-second windows per month. A tolerance of 1 false wake-up per month requires 99.99996% rejection rate -- far beyond what "99% accuracy" suggests. This demonstrates how deployment context (always-on sensing with extreme duty cycle) transforms the data engineering requirements: the KWS case study from the chapter shows the four pillars of data engineering under extreme resource constraints.
  Interaction: Students set a false positive tolerance (false wake-ups per month) and a duty cycle (hours per day). The lab computes the required rejection rate given the number of classification windows. Students discover that "1 false wake per month" requires a rejection rate with five nines after the decimal. A slider for window size shows how finer temporal resolution makes the requirement even more stringent. This connects data quality directly to system-level behavior.

---

---

## Lab 05: Neural Computation
Chapter: `nn_computation.qmd`

The existing plan has 3 parts (Transistor Tax, Memory Hierarchy Cliff, Training Memory Multiplier). These are well-grounded but miss the chapter's most important systems concepts: the forward pass as a computational workload (matrix multiplication dominance, FLOP counting), and backpropagation as a memory management problem. The training memory multiplier also overlaps with Lab 08. I propose replacing Part C with something unique to this chapter and adding two new parts that address core content.

### Proposed Parts:

- **Part A -- The Transistor Tax** (~10 min)
  Concept: Activation function silicon cost disparity (ReLU: 50 transistors vs. Sigmoid: 2,500 -- a 50x gap that determines whether activation compute is negligible or dominant depending on deployment target).
  Interaction: Per-layer activation function dropdown selector (4 layers); context toggle (Cloud vs Mobile NPU). Students predict the cost ratio, then watch total silicon budget and inference time fraction change as they swap activations across layers. The Mobile context makes the 23% inference time fraction from activation compute visible; the Cloud context shows it is negligible.

- **Part B -- The Memory Hierarchy Cliff** (~10 min)
  Concept: Activation tensor size determines memory tier placement (L1/L2/HBM/DRAM), and crossing a tier boundary is a 10-100x latency step function, not a gradual slope.
  Interaction: Batch size slider (1-512) and layer width slider (64-4096) with context toggle. Students see a stacked bar chart colored by memory tier, with horizontal threshold lines. The prediction question establishes false security (16 KB fits in L2), then the instruments reveal that modest increases in batch size or width push past L2 into HBM on mobile, triggering a 10x latency cliff.

- **Part C -- Counting Operations: The Width-Squared Surprise** (~10 min)
  Concept: Dense layer FLOPs scale as O(width squared), so doubling hidden layer width causes a 4x FLOP increase, not 2x. This is the chapter's worked example showing that architecture decisions are the dominant variable in the Iron Law's O term.
  Interaction: Layer width sliders for a 3-layer MLP (input fixed at 784, output at 10). Students predict the FLOP increase from doubling hidden widths from 128 to 256. A live FLOP counter per layer and a stacked bar chart show that the 2x width increase yields approximately 3.8x total FLOPs. Students also see arithmetic intensity change, connecting to why these layers are bandwidth-bound at small batch sizes.

- **Part D -- Forward vs. Backward: Where the Memory Goes** (~10 min)
  Concept: The forward pass can discard activations layer by layer, but backpropagation requires storing every intermediate activation until the backward pass reaches that layer. Training memory is dominated by activations for deep networks, not weights.
  Interaction: A depth slider (3-20 layers) and batch size slider with a "phase toggle" (Inference vs. Training). In inference mode, only the current layer's activations appear in the memory ledger. In training mode, all layers' activations accumulate simultaneously, and students watch the stacked bar grow linearly with depth. The prediction asks students to estimate training-to-inference memory ratio; most guess 2x, the answer is 4x+ (connecting to the chapter's @eq-training-memory). This Part deliberately stops at the conceptual level and does NOT cover optimizer state (which is Lab 08's territory).

- **Synthesis** (~5 min)
  Prompt: Deploy a 10-layer model on a mobile NPU with 8 GB RAM and 5 W power budget for both inference (30 FPS) and on-device fine-tuning. Justify activation function choice, maximum batch size given memory tier constraints, and whether fine-tuning is feasible given the forward-vs-backward memory multiplier.

---

## Lab 06: Network Architectures
Chapter: `nn_architectures.qmd`

The existing plan has 3 parts (Inductive Bias / MLP explosion, Quadratic Attention Wall, Depth vs. Width). These are excellent and well-grounded. The chapter is enormous, though, and two major concepts are missing: (1) the arithmetic intensity spectrum that differentiates architectures as compute-bound vs. memory-bound workloads, and (2) the embedding table capacity wall for recommendation systems (DLRM). I recommend keeping the existing 3 parts and adding a 4th on arithmetic intensity, which is the chapter's key systems lens.

### Proposed Parts:

- **Part A -- The Cost of No Structure** (~12 min)
  Concept: Inductive bias as a physical memory constraint. An MLP processing a 224x224 RGB image requires 22.7 billion parameters in its first layer (O(d^2) scaling), while a CNN with 3x3 filters requires 1,728 parameters -- a 13.1 million-fold reduction.
  Interaction: Architecture toggle (MLP / CNN 3x3 / CNN 5x5) and image resolution slider (28x28 to 512x512). Parameter count bar chart (log scale) and memory bar with device threshold lines. Students predict MLP parameter count for 224x224; most guess ~150M, the answer is 22.7B.

- **Part B -- The Quadratic Wall** (~12 min)
  Concept: Transformer self-attention creates an N x N score matrix that scales quadratically with sequence length, imposing a hard OOM ceiling on context window size.
  Interaction: Sequence length slider (512 to 131,072) and attention heads slider (1-32). An attention memory curve (GB vs. tokens) with H100 and edge device threshold lines. Students predict the memory increase from doubling context length from 4K to 8K; most guess 2x, the answer is 4x. At 100K tokens, the memory is 625x larger than at 4K.

- **Part C -- Depth vs. Width: The Sequential Bottleneck** (~10 min)
  Concept: Two networks with identical parameter counts and total FLOPs can have dramatically different latencies because depth forces O(L) sequential execution while width exposes parallel FLOPs.
  Interaction: Depth slider (2-128 layers), width slider (32-2048), context toggle (Cloud/Edge). A latency waterfall chart decomposes per-layer compute, memory load, and dispatch overhead. Students predict whether a deep-narrow or shallow-wide network is faster; most guess "same speed" because FLOPs are equal.

- **Part D -- Workload Signatures: Why Architecture Determines the Bottleneck** (~8 min)
  Concept: Each architecture family has a characteristic arithmetic intensity (FLOPs per byte of data moved) that determines whether it is compute-bound or memory-bound. CNNs have high arithmetic intensity (compute-bound, I > 20), Transformers at inference have low arithmetic intensity (memory-bound, I < 1 for attention), and MLPs at batch=1 have very low intensity (~0.5). This is the chapter's "Workload Signatures" concept.
  Interaction: Architecture selector (MLP / CNN / RNN / Transformer / DLRM) with batch size slider (1-256). A horizontal bar chart shows arithmetic intensity for each architecture at the current batch size, with a vertical "ridge point" line for the target hardware (A100: ~156 ops/byte). Operations to the left of the ridge point are memory-bound; to the right, compute-bound. Students predict which architecture is most compute-efficient on a GPU; most pick Transformers (because they are "modern"), but CNNs have the highest arithmetic intensity and thus the highest utilization.

- **Synthesis** (~5 min)
  Prompt: Deploy a model to classify images from wildlife cameras on a 16 GB edge device with a 50 ms latency SLA. Justify architecture choice, depth/width strategy, and explain why the arithmetic intensity of your chosen architecture determines hardware utilization.

---

## Lab 07: ML Frameworks
Chapter: `frameworks.qmd`

The existing plan has 3 parts (Dispatch Tax, Compilation Break-Even, Deployment Spectrum). These are well-designed and cover the chapter's three core problems (execution, differentiation as manifested in compilation, and abstraction). The chapter also has rich content on the Memory Wall, kernel fusion, and the computational graph that the existing plan touches only indirectly. I propose keeping the existing 3 parts and adding a 4th on kernel fusion / the Memory Wall, which is the chapter's opening motivation for why execution strategy matters.

### Proposed Parts:

- **Part A -- The Dispatch Tax** (~10 min)
  Concept: Python dispatch overhead (~10 us per op) makes small-kernel models overhead-bound, not compute-bound, regardless of GPU speed. GPU utilization depends on per-kernel compute-to-dispatch ratio, not total kernel count.
  Interaction: Kernel count slider (10-2000), compute-per-kernel slider (1-10,000 us, log scale), dispatch overhead slider (1-50 us). GPU utilization gauge and stacked bar chart comparing KWS-like (1000 small kernels) vs. GPT-2-like (20 large kernels) workloads. Students predict which model has higher utilization; most pick KWS ("more operations = busier GPU").

- **Part B -- The Fusion Dividend: Why Execution Strategy Matters** (~10 min)
  Concept: Kernel fusion eliminates intermediate memory writes between operations. The Memory Wall (312 TFLOPS compute vs. 2 TB/s bandwidth = 156 ops/byte ridge point) means element-wise operations like ReLU achieve less than 1% utilization when executed individually. Fusing LayerNorm + Dropout + ReLU into one kernel yields 5x speedup by eliminating intermediate HBM writes.
  Interaction: An operation sequence builder where students chain 2-5 element-wise operations (ReLU, LayerNorm, Dropout, Add, Multiply). Toggle between "Eager" (each op launches a separate kernel with separate HBM read/write) and "Fused" (single kernel, single HBM read/write). A timeline visualization shows GPU compute bars vs. memory stalls. Students predict the speedup from fusing 3 operations; most guess 1.5x, the answer is 3-5x because memory traffic drops by 3x.

- **Part C -- The Compilation Break-Even** (~12 min)
  Concept: torch.compile has a fixed upfront cost (30s) that must be amortized over production executions. The break-even point depends on deployment volume, not model quality. A 48% speedup on ResNet-50 requires 134,000 inferences to break even.
  Interaction: Deployment volume slider (10 to 10M requests/hour, log scale), compile time slider (5-300s), context toggle (Cloud/Edge). A break-even timeline chart shows cumulative time for eager vs. compiled, with crossover point highlighted. ROI gauge turns red when break-even exceeds deployment lifetime.

- **Part D -- The Deployment Spectrum** (~8 min)
  Concept: Framework selection determines feasibility, not just speed. The same ResNet-50 spans 17x latency (PyTorch 52ms vs. TensorRT 3ms) and 56x memory (1800 MB vs. 32 MB) across frameworks, without changing a single weight. On a 256 KB MCU, the question shifts from "which is fastest" to "which fits at all."
  Interaction: Framework dropdown (PyTorch, TensorFlow, TF Lite, TensorRT, TF Lite Micro, etc.) and deployment target toggle (Cloud/Edge/MCU). Metric cards show latency, memory, utilization, energy. Feasibility banner turns red on OOM (e.g., any full framework on MCU).

- **Synthesis** (~5 min)
  Prompt: You manage a KWS model (1,000 small kernels) and a ResNet-50 deployed across cloud and edge. For each model-context pair, justify your framework choice and whether to compile, using specific numbers from the lab.

---

## Lab 08: Model Training
Chapter: `training.qmd`

The existing plan has 3 parts (Memory Budget Shock, Gradient Accumulation, Mixed Precision FP32 Trap). These are good but miss the chapter's most distinctive systems concepts: the training pipeline as a staged system with "accelerator bubbles," the Iron Law of Training Performance, and the communication tax when scaling to multiple GPUs. The memory budget already appeared in Lab 05 Part C (which I replaced above). I propose keeping Parts A and C from the existing plan and replacing Part B with a pipeline bottleneck identification Part, then adding a Part D on multi-GPU scaling.

### Proposed Parts:

- **Part A -- The Memory Budget Shock** (~10 min)
  Concept: Training memory = Weights + Gradients + Optimizer State + Activations. For Adam in FP32, static state requires 16 bytes/parameter. A 7B model needs 112 GB before a single activation is stored.
  Interaction: Model size slider (0.1B-70B), optimizer dropdown (SGD/Momentum/Adam/Adafactor), precision toggle (FP32/BF16). Stacked bar chart showing each memory component with device RAM threshold line. Students predict minimum training memory for 7B Adam; most guess 56 GB (weights + gradients), the answer is 112 GB.

- **Part B -- The Training Pipeline: Finding the Bubble** (~12 min)
  Concept: Training is a staged pipeline (Data Loading, Host-to-Device Transfer, Forward/Backward Pass, Parameter Sync). The slowest stage determines throughput. "Accelerator bubbles" are intervals where the GPU idles waiting for data or synchronization. The chapter's key insight: most training runs are NOT compute-bound -- they are data-loading-bound or communication-bound.
  Interaction: Four sliders representing stage latencies: data loading (1-100 ms), PCIe transfer (0.1-10 ms), forward+backward (5-200 ms), gradient sync (0-50 ms). A pipeline Gantt chart shows sequential vs. overlapped execution. Toggle between "Sequential" (total = sum of stages) and "Overlapped/Prefetched" (total = max of stages + overhead). A "bottleneck indicator" highlights which stage is binding. Students predict the bottleneck for GPT-2 on V100; most guess "forward pass" (compute), but the chapter shows data loading or gradient sync often dominate.

- **Part C -- Mixed Precision: The FP32 Master Copy Trap** (~10 min)
  Concept: Mixed-precision training uses FP16/BF16 for forward/backward but retains FP32 master weights and Adam state. Actual savings are ~1.5-1.7x, not the expected 2x. The parameter state cost (16 bytes/param) is identical in both modes; savings come entirely from activations.
  Interaction: Precision toggle (Full FP32 / Mixed FP16+FP32 master / BF16+FP32 master), model size slider (0.1B-13B). Side-by-side stacked bars showing FP32 baseline vs. mixed precision, with the FP32 "tail" components (master weights, Adam state) visually highlighted. Students predict GPT-2 mixed-precision memory; most guess ~38 GB (half of 77 GB), the answer is ~45 GB.

- **Part D -- The Communication Tax: When More GPUs Hurt** (~10 min)
  Concept: Multi-GPU scaling follows Speedup = N / (1 + (N-1) * r), where r is the fraction of step time spent on communication. Compute-bound workloads (r=0.05) scale well; bandwidth-bound workloads (r=0.40) see diminishing returns quickly. The chapter's @fig-communication-tax makes this concrete.
  Interaction: GPU count slider (1-256, log scale), communication fraction slider (0.01-0.50), workload preset buttons (ResNet/LLM+NVLink/Slow Network). A scaling efficiency curve shows effective throughput vs. GPU count with an "ideal linear" reference line. The shaded gap between ideal and actual is the "communication tax." Students predict speedup from 8 GPUs with r=0.15; most guess ~8x (linear), the answer is ~4.7x.

- **Synthesis** (~5 min)
  Prompt: You must train a 7B model on a cluster. Specify: precision mode, micro-batch and accumulation strategy, number of GPUs, and which pipeline stage you would optimize first. Justify each choice with numbers from the lab.

---

## Summary of Changes from Existing Plans

| Lab | Existing Parts | Proposed Parts | Key Changes |
|-----|---------------|----------------|-------------|
| **05** | 3 (Transistor Tax, Memory Cliff, Training Memory Multiplier) | 4 + Synthesis | Replaced Part C (training memory, overlaps with Lab 08) with Width-Squared FLOP scaling; added Part D (Forward vs. Backward memory, unique to this chapter) |
| **06** | 3 (Inductive Bias, Quadratic Wall, Depth vs. Width) | 4 + Synthesis | Added Part D (Workload Signatures / Arithmetic Intensity), which is the chapter's key systems lens connecting architecture to hardware utilization |
| **07** | 3 (Dispatch Tax, Compilation Break-Even, Deployment Spectrum) | 4 + Synthesis | Added Part B (Kernel Fusion / Memory Wall), which is the chapter's opening motivation and directly explains WHY execution strategy matters |
| **08** | 3 (Memory Budget, Gradient Accumulation, Mixed Precision) | 4 + Synthesis | Replaced Gradient Accumulation with Pipeline Bottleneck Identification (more unique to the training chapter); added Part D (Communication Tax / Multi-GPU Scaling) |

---

## Lab 09: Data Selection
Chapter: `/Users/VJ/GitHub/MLSysBook/book/quarto/contents/vol1/data_selection/data_selection.qmd`

### Proposed Parts:

- **Part A -- The ICR Frontier: Diminishing Returns** (~12 min)
  Concept: The Information-Compute Ratio (ICR) decays as 1/(O*D) -- most data in a large dataset contributes near-zero learning signal. A 50% coreset retains ~99% of accuracy, but the curve is logarithmic, not linear.
  Interaction: Students drag a "dataset fraction" slider (5%-100%) and a "redundancy level" toggle, watching the ICR curve flatten and a "knee" marker shift. They predict where accuracy drops below 95% of baseline before seeing the curve.

- **Part B -- The Selection Inequality: When Optimization Backfires** (~15 min)
  Concept: The cost of *scoring* data for coreset selection (T_selection) can negate training savings if T_selection + T_train(subset) >= T_train(full). Using a full model to score 1M images takes 2.8 hours; a proxy model takes 0.6 hours.
  Interaction: Students adjust coreset fraction, scoring model (full/proxy/cached), and deployment context (Cloud/Edge). A waterfall bar chart shows whether the Selection Inequality is satisfied or violated. On Edge with full-model scoring, the inequality breaks -- the failure state turns red.

- **Part C -- Curriculum Learning: The Ordering Surprise** (~12 min)
  Concept: Easy-to-hard training ordering (curriculum learning) delivers 11-23% convergence speedup over random; hard-first ordering is ~30% slower than random. The intuition that "hard examples teach more" is wrong for gradient-based optimization.
  Interaction: Students select a training order (random/easy-first/hard-first) and a warmup fraction, then watch three convergence curves diverge live. A threshold marker shows where each strategy crosses 95% accuracy, revealing that easy-first reaches it significantly earlier.

- **Part D -- The Cost-Optimal Frontier: Data-Starved vs Compute-Starved** (~10 min)
  Concept: The compute-optimal frontier (Chinchilla scaling) determines whether a training run is data-starved (more data would help more than more compute) or compute-starved (more compute would help more than more data). Most teams misdiagnose their position.
  Interaction: Students place their training configuration on a 2D plot of dataset size vs compute budget, toggling model scale. The simulation shows whether they are above (data-starved) or below (compute-starved) the optimal frontier, with a diagnostic reading that prescribes "get more data" or "get more compute."

---

## Lab 10: Model Compression
Chapter: `/Users/VJ/GitHub/MLSysBook/book/quarto/contents/vol1/optimizations/model_compression.qmd`

### Proposed Parts:

- **Part A -- The Quantization Free Lunch** (~12 min)
  Concept: Reducing precision from FP32 to INT8 costs under 1% accuracy (the "Free Lunch Zone"), then accuracy collapses catastrophically at 3-4 bits (the "Quantization Cliff"). The curve is flat then vertical, not gradual.
  Interaction: Students select a model architecture, quantization scheme (PTQ/QAT), and calibration dataset size. A table and bar chart show accuracy, model size, latency, and energy across FP32/FP16/INT8/INT4/INT2. The INT8 row glows green as the practical sweet spot.

- **Part B -- The Pruning Hardware Trap** (~15 min)
  Concept: Unstructured pruning at 50% sparsity gives zero latency speedup on standard GPU kernels because dense GEMM iterates over every element including zeros. Structured pruning removes entire channels, yielding real speedup.
  Interaction: Students drag a sparsity slider (0-95%) and toggle between unstructured/structured pruning. Two charts show speedup-vs-sparsity (flat at 1.0x for unstructured; rising for structured) and accuracy-vs-sparsity. Pushing structured pruning past 85% triggers an accuracy collapse failure state.

- **Part C -- The Compression Pareto Frontier** (~12 min)
  Concept: Deploying an 8B-parameter LLM across three mobile memory tiers (8 GB / 4 GB / 2 GB) requires composing INT8, INT4, and structured pruning along a Pareto frontier. No single technique spans all tiers.
  Interaction: Students select a compression strategy for each tier from a dropdown (FP16/INT8/INT4/INT4+pruning/distilled variants). A scatter plot shows Pareto-optimal vs dominated configurations, with per-tier memory budget lines. Selecting a config that exceeds a tier's budget triggers an OOM failure state.

- **Part D -- The Energy Dividend: Bits are Joules** (~8 min)
  Concept: Moving data costs 40,000x more energy than computing it (DRAM read vs integer add). INT8 inference uses up to 20x less energy than FP32. On a battery device, this is the difference between 1 hour and 20 hours of operation.
  Interaction: Students toggle precision format and deployment target. An energy breakdown shows DRAM access energy vs compute energy for each format, with a battery life estimate. The visual starkness of the 40,000x DRAM-to-compute ratio makes the "Memory Wall" visceral.

---

## Lab 11: Hardware Acceleration
Chapter: `/Users/VJ/GitHub/MLSysBook/book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd`

### Proposed Parts:

- **Part A -- The Memory Wall (Roofline Diagnosis)** (~15 min)
  Concept: A GEMM kernel at N=512 achieves only 31.5% MFU on the H100 -- not because the code is broken, but because arithmetic intensity (170 FLOP/byte) falls below the ridge point (295 FLOP/byte). The kernel is correctly hitting the memory bandwidth ceiling.
  Interaction: Students drag a matrix dimension slider (N=128 to 8192) and toggle precision (FP32/FP16/INT8). A log-log roofline plot shows the operation point sliding along the bandwidth slope, crossing the ridge into the compute-bound regime as N grows. Metric cards display AI, MFU, and regime classification live.

- **Part B -- Kernel Fusion: The Elementwise Trap** (~15 min)
  Concept: LLM inference mixes kernels spanning 3 orders of magnitude in arithmetic intensity. GEMM can become compute-bound at large batch, but LayerNorm/Softmax (AI ~ 0.83 FLOP/byte) are permanently memory-bound. Kernel fusion is the only lever for these ops.
  Interaction: Students adjust batch size, sequence length, and fusion strategy on a multi-operation roofline showing three markers (GEMM, LayerNorm, Softmax). A memory budget bar shows weights + KV-cache + activations. Increasing batch too far triggers an OOM failure state when KV-cache exceeds device RAM.

- **Part C -- The Hardware Balance Shift** (~10 min)
  Concept: The same kernel can be compute-bound on an edge device (ridge point ~118 FLOP/byte) but memory-bound on the H100 (ridge point ~295). More powerful accelerators are paradoxically harder to saturate, because compute grows faster than bandwidth across generations.
  Interaction: Students toggle between Cloud (H100) and Edge (Jetson Orin NX) contexts. The roofline redraws with different ceilings and ridge points. A GEMM that was compute-bound on edge hardware switches to memory-bound on the cloud without any change to the algorithm -- demonstrating that bottleneck diagnosis is hardware-specific.

---

## Lab 12: Benchmarking
Chapter: `/Users/VJ/GitHub/MLSysBook/book/quarto/contents/vol1/benchmarking/benchmarking.qmd`

### Proposed Parts:

- **Part A -- The Amdahl Ceiling** (~15 min)
  Concept: A 10x inference speedup with 45% non-inference overhead yields only 2.0x end-to-end improvement. Amdahl's Law caps system speedup at 1/f, where f is the non-optimized fraction. The remaining 8x of hardware investment is wasted on a bottleneck that has already moved.
  Interaction: Students set inference speedup (1x-100x) and non-inference fraction (0.05-0.80). A before/after waterfall bar chart and an Amdahl saturation curve show how quickly returns diminish. A metric row displays the "wasted speedup" -- the gap between component improvement and system improvement.

- **Part B -- Peak vs Sustained: The Thermal Cliff** (~15 min)
  Concept: A vendor advertises 30 FPS for an edge chip, but after 5 minutes of continuous inference in a fanless enclosure, thermal throttling halves it to 15 FPS. Vendor benchmarks are burst measurements; production runs are sustained.
  Interaction: Students scrub through time (0-10 minutes) and toggle passive/active cooling plus ambient temperature. A time-series chart shows FPS decaying and junction temperature rising until the thermal cliff hits. When sustained FPS drops below the minimum viable threshold (24 FPS), a failure banner fires.

- **Part C -- The Multi-Metric Trap** (~10 min)
  Concept: The model configuration with the best single-metric scores (94% accuracy, 1200 QPS) violates latency and power SLOs. Only the balanced configuration (91% accuracy, 95 ms p99, 600 QPS, 4.5 W) passes all four deployment gates simultaneously.
  Interaction: Students adjust batch size, precision, and model variant. A radar chart with four axes (accuracy, latency, throughput, power) overlays SLO threshold rings. An SLO compliance table shows pass/fail per metric. Any violated SLO triggers a "Deployment Blocked" failure state with a red banner naming the violated constraints.

- **Part D -- Training vs Inference: Different Games** (~8 min)
  Concept: Training benchmarks optimize for convergence time and scaling efficiency; inference benchmarks optimize for tail latency (p99) and energy-per-query. A system with 50 ms average latency but 500 ms p99 violates a 200 ms SLO for 1% of requests -- at 10K requests/sec, that is 100 failures per second.
  Interaction: Students toggle between training mode (showing throughput, time-to-accuracy, scaling efficiency metrics) and inference mode (showing p50/p95/p99 latency distributions, energy-per-query). A latency histogram reveals the long tail. Students set an SLO threshold and see what percentage of requests violate it -- demonstrating that average latency hides production failures.

---

## Lab 13: The Tail Latency Trap
Chapter: `model_serving.qmd`

### Proposed Parts:

- **Part A -- The Tail Latency Explosion** (~12 min)
  Concept: The M/M/1 queuing model predicts that P99 latency diverges nonlinearly from mean latency as server utilization increases -- at 80% utilization, P99 is 23x the service time while the mean is only 5x, making "healthy average latency" a dangerously misleading metric.
  Interaction: Students drag a utilization slider (rho, 0.10-0.95) and watch a live P99 latency histogram shift, with mean and P99 vertical markers diverging. A fixed SLO threshold line turns the tail region red as utilization crosses ~70%.

- **Part B -- The Batching Tax** (~15 min)
  Concept: The latency-throughput Pareto frontier -- larger batch sizes improve GPU throughput (batch-32 achieves 6.4x over batch-1) but impose a formation delay tax of (B-1)/(2*lambda) that can exceed the SLO before inference even begins, creating a sharp "knee" where throughput gains plateau and latency explodes.
  Interaction: Students control batch size (1-64), arrival rate (100-2000 QPS), and SLO budget (10-100 ms) to find the largest feasible batch size. A dual-axis Pareto frontier chart and a latency waterfall (formation delay + inference time) show exactly where the SLO line is breached.

- **Part C -- The LLM Memory Wall** (~12 min)
  Concept: LLM decode-phase token generation is entirely memory-bandwidth-bound (T_token = Model_Size / Memory_BW), and KV cache capacity -- not compute TFLOPS -- determines maximum concurrent batch size, creating an OOM boundary that limits throughput regardless of available compute power.
  Interaction: Students manipulate model size (1-40 GB), context length (256-32768 tokens), and concurrent batch size (1-128) to find the OOM boundary. A KV cache memory budget stacked bar chart shows weights vs. KV cache vs. remaining VRAM, with a red "OOM Zone" when the total exceeds 80 GB.

- **Synthesis** (~6 min)
  Concept: Integrating queuing theory (Part A), batching economics (Part B), and LLM memory constraints (Part C) into a capacity planning decision for two workloads: a vision classifier and a chat endpoint.
  Interaction: Students write a capacity specification justifying utilization target, batch size, and GPU count for each workload, referencing specific formulas from the lab.

---

## Lab 14: The Silent Degradation Problem
Chapter: `ml_ops.qmd`

### Proposed Parts:

- **Part A -- PSI Drift Detection (The Silent Drift)** (~12 min)
  Concept: The operational mismatch between infrastructure health and model health -- a fraud detection model can lose 7 percentage points of accuracy over 6 months while every infrastructure metric (uptime, latency, error rate) stays green. PSI-based monitoring of input feature distributions detects drift weeks or months before accuracy degradation becomes visible.
  Interaction: Students scrub a time slider (0-26 weeks) and watch PSI trajectories for three features cross the 0.2 threshold while an infrastructure status panel remains permanently green. Accuracy decay cards show the gap between "all healthy" infrastructure and silently degrading model quality.

- **Part B -- Optimal Retraining Cadence (The Half-Life of a Model)** (~15 min)
  Concept: The staleness cost model and the square-root law for optimal retraining intervals -- T* = sqrt(2C / C_drift) produces a sublinear relationship where 4x more expensive retraining only doubles the interval, transforming retraining from an ad hoc decision into a quantitative economic optimization.
  Interaction: Students adjust drift rate (7-90 days to 7% drop), retraining cost ($1K-$50K), and accuracy threshold (80-95%), watching a U-shaped total annual cost curve shift and the T* star marker move. A failure state triggers when accuracy at T* falls below the minimum threshold.

- **Part C -- Deployment Cost Asymmetry (Same Model, Different Economics)** (~12 min)
  Concept: The same model with identical accuracy requirements produces dramatically different optimal retraining intervals across Cloud, Edge, and Mobile environments purely because of deployment cost differences -- Cloud T* of 7-14 days vs. Edge T* of 60-90 days, driven by T* scaling with sqrt(cost), not cost itself.
  Interaction: Students see all three environments side by side, adjusting per-environment costs and drift rates. A comparison panel highlights that a 100x cost difference produces only a 10x cadence difference due to the square-root law.

- **Synthesis** (~6 min)
  Concept: Designing a complete monitoring and retraining policy for a multi-environment deployment, specifying PSI monitoring frequency, drift thresholds, T* with cost justification, and rollback strategy.
  Interaction: Students compose a written monitoring policy referencing specific numbers from the lab's three parts.

---

## Lab 15: There Is No Free Fairness
Chapter: `responsible_engr.qmd`

### Proposed Parts:

- **Part A -- The Fairness Illusion** (~12 min)
  Concept: The Chouldechova impossibility theorem -- when base rates differ between demographic groups, a calibrated classifier with equal accuracy on both groups is mathematically guaranteed to produce unequal error rates (FPR, FNR, PPV). Equal accuracy does not mean equal treatment.
  Interaction: Students adjust base rate sliders for two groups (5-50%) and a shared classification threshold (0.10-0.90), watching a grouped bar chart of per-group metrics (accuracy, TPR, FPR, FNR, PPV) reveal large disparities that emerge even when accuracy is identical. Color-coded gap cards flag violations.

- **Part B -- The Price of Fairness** (~15 min)
  Concept: The fairness-accuracy Pareto frontier -- quantifying the accuracy cost of different fairness criteria (demographic parity, equal opportunity, equalized odds) and mitigation strategies (threshold adjustment, reweighting, adversarial debiasing). The frontier has a "sweet spot" where the first large fairness gains cost only 3-5% accuracy, while strict equality demands disproportionate sacrifice.
  Interaction: Students select a fairness criterion and mitigation method via radio buttons, watching a Pareto frontier chart update with Points A (unconstrained), B (sweet spot), and C (strict equality) annotated. A failure state triggers when equalized odds gap exceeds 10 percentage points.

- **Part C -- Where the Money Goes (Total Cost of Ownership)** (~12 min)
  Concept: Inference costs dominate production ML systems by 10-40x over training costs, making per-query optimization the highest-leverage responsible engineering investment. A 20% inference latency reduction saves more money and carbon than eliminating training costs entirely.
  Interaction: Students adjust daily users (1-50M), recommendations per user (5-50), inference latency (1-50 ms), and retraining frequency to watch a stacked TCO bar chart and carbon accounting card update in real time. A failure state warns when high inference spend combines with infrequent retraining.

- **Synthesis** (~6 min)
  Concept: Integrating fairness constraints, accuracy trade-offs, and TCO economics into a stakeholder recommendation for a loan approval system deployment.
  Interaction: Students write a 3-5 sentence recommendation addressing fairness criterion choice, accuracy cost justification, and optimization investment focus, referencing specific numbers from the lab.

---

## Lab 16: The Architect's Audit (Capstone Synthesis)
Chapter: `conclusion.qmd`

### Proposed Parts:

- **Part A -- The Cost of a Token (Calibration)** (~12 min)
  Concept: The Iron Law decomposition for single-token LLM inference reveals that memory-to-compute time ratio is ~40x on H100, not the ~5x most students expect. Autoregressive decoding has arithmetic intensity of ~1 FLOP/byte, placing it deep in the memory-bandwidth-bound regime -- making compute kernel optimization futile without addressing data movement.
  Interaction: Students select model size (7B/13B/70B), precision (FP32/FP16/INT8/INT4), and batch size (1-256), watching an Iron Law decomposition bar chart and a Roofline operating point dot move. At batch=64, the memory and compute bars approximately equalize, making the crossover from memory-bound to compute-bound visible.

- **Part B -- The Conservation of Complexity** (~15 min)
  Concept: The meta-principle that complexity in an ML system cannot be destroyed, only moved between Data, Algorithm, and Machine. Quantizing MobileNetV2 to INT8 reduces model complexity but creates monitoring debt -- without monitoring investment, accuracy degrades ~1.3%/month while all infrastructure metrics remain green.
  Interaction: Students configure architecture, quantization, and monitoring investment level, then scrub a "months deployed" slider (0-12). A 5-axis radar chart (accuracy, latency, memory, power, drift resilience) shows the polygon silently shrinking on the accuracy axis while all other axes remain green. An invariant activation table shows which of the 12 invariants are triggered simultaneously.

- **Part C -- Design Ledger Archaeology** (~12 min)
  Concept: A student's own prediction history across Labs 01-15 reveals systematic blind spots -- which invariants their intuition consistently underweights. The gap between self-assessed and data-revealed weaknesses is itself an engineering insight: mental models degrade just like deployed models.
  Interaction: Students self-assess which invariant they most consistently underestimated, then view a personalized 7-axis radar chart computed from their Design Ledger history. The reveal compares self-assessment to actual data, diagnosing whether the student's mental model is calibrated.

- **Synthesis** (~6 min)
  Concept: A "graduation specification" integrating all three parts into a system deployment plan that addresses quantization strategy, monitoring investment, personal blind spot mitigation, and the Iron Law lesson about where to invest optimization effort.
  Interaction: Students write a deployment specification for MobileNetV2 on a mobile fleet, referencing specific numbers and invariants from their lab experience across all 16 labs.

---


# VOLUME II: Machine Learning Systems at Scale

## Lab V2-01: Introduction to Scale
**Chapter:** `introduction.qmd`
**1-hour story arc:** Students discover that scaling from one machine to a fleet changes the physics of ML systems: reliability collapses, communication dominates, and efficiency degrades in ways that naive linear extrapolation cannot predict.

### Proposed Parts:

- **Part A -- The Reliability Collapse** (~12 min)
  **Concept:** Fleet-wide availability decays exponentially as fleet size grows, even with highly reliable individual nodes.
  **Key equation/principle:** $P_{\text{fleet}} = (P_{\text{node}})^N$ (@eq-reliability-gap). At 99.9% per-node reliability, a 1,000-node cluster is healthy only 36.8% of the time; at 10,000 nodes, near zero.
  **Interaction:** Students PREDICT what fleet availability will be for a 10,000-GPU cluster with 99.9% per-node reliability. Then they use sliders for (1) per-node reliability (99% to 99.999%) and (2) fleet size (1 to 25,000 GPUs) to watch the exponential decay curve in real time. A dashboard shows "time between failures" alongside the availability percentage, making the transition from "failure is rare" to "failure is routine" viscerally concrete.

- **Part B -- The Coordination Tax** (~12 min)
  **Concept:** The Fleet Law decomposes distributed step time into Compute + Communication + Coordination, and the "Conservation of Overhead" means you cannot eliminate overhead, only redistribute it.
  **Key equation/principle:** $T_{\text{step}} = T_{\text{Compute}} + T_{\text{Communication}} + T_{\text{Coordination}}$ (@eq-fleet-law); Fleet efficiency $\eta_{\text{fleet}} = T_{\text{Compute}} / T_{\text{step}}$ (@eq-fleet-efficiency).
  **Interaction:** Students adjust three sliders: (1) number of GPUs (1 to 1024), (2) network type (InfiniBand HDR 200 Gbps vs. Ethernet 100 Gbps), and (3) model gradient size (1 GB to 700 GB). A stacked bar chart dynamically shows the three time components for each step, and a fleet efficiency gauge shows $\eta_{\text{fleet}}$. Students PREDICT whether switching from Ethernet to InfiniBand doubles efficiency for a 175B model, then discover the answer depends on whether Communication or Coordination dominates.

- **Part C -- The Scaling Law Budget Planner** (~12 min)
  **Concept:** Compute-optimal resource allocation (Chinchilla scaling) requires coordinated scaling of model size and dataset size; scaling one dimension alone wastes resources.
  **Key equation/principle:** $D \propto N^{0.74}$ (Hoffmann et al. compute-optimal ratio); IsoFLOP curves with valleys identifying optimal model size per compute budget.
  **Interaction:** Students receive a fixed compute budget (in FLOPs) and use two sliders to allocate it between model parameters and training tokens. A loss surface shows IsoFLOP curves. Students PREDICT whether a 10B model trained on 200B tokens or a 3B model trained on 600B tokens achieves lower loss for the same compute. The simulation reveals the Chinchilla optimal point and how far off-optimal each naive allocation is.

- **Part D -- The Iron Law of Scale** (~12 min)
  **Concept:** Distributed training speedup is limited by the serial fraction (Amdahl's Law with communication). Adding GPUs beyond a certain point yields diminishing returns that eventually waste money.
  **Key equation/principle:** $T_{\text{step}}(N) = T_{\text{compute}}/N + T_{\text{comm}}(N) - T_{\text{overlap}}$ (@eq-iron-law-scale); Scaling efficiency $\eta_{\text{scale}} = T_{\text{compute}} / (N \times T_{\text{step}})$.
  **Interaction:** Students adjust (1) communication fraction $r$ (0% to 50%), (2) number of GPUs (1 to 512), and (3) overlap percentage (0% to 80%). A log-log speedup chart (like @fig-scaling-tax) updates live, showing ideal linear scaling vs. actual speedup. Students PREDICT how many GPUs are needed before efficiency drops below 50% for a communication fraction of 20%, then discover the answer is surprisingly low. A cost calculator shows dollars wasted on idle GPUs.

- **Part E -- The C-Cubed Diagnostic** (~12 min)
  **Concept:** The C-cube taxonomy (Computation, Communication, Coordination) provides a diagnostic framework for identifying the dominant bottleneck in any distributed system.
  **Key equation/principle:** The C-cube triangle from @fig-c3-taxonomy, with the "Conservation of Overhead" at its center. Mapping real workloads (GPT-4, DLRM, Federated MobileNet) to their positions in the triangle.
  **Interaction:** Students are presented with three lighthouse archetypes (Archetype A: GPT-4 LLM training, Archetype B: DLRM embedding-heavy recommendation, Archetype C: Federated MobileNet on edge devices). For each archetype, students drag a point within the C-cube triangle to PREDICT where the dominant bottleneck lies. The simulation then reveals the actual proportions of time spent in Compute, Communication, and Coordination. Students discover that LLM training is Communication-dominated, DLRM is Coordination-dominated (All-to-All embedding lookups), and Federated Learning is Coordination-dominated (straggler handling and privacy overhead).

---

## Lab V2-02: Compute Infrastructure
**Chapter:** `compute_infrastructure.qmd`
**1-hour story arc:** Students build intuition for the memory wall, the roofline model, and the bandwidth hierarchy that dictates how parallelism strategies map to physical hardware, from a single chip to a pod of thousands.

### Proposed Parts:

- **Part A -- The Memory Wall** (~12 min)
  **Concept:** Token generation latency is dominated by memory bandwidth, not compute. Even with infinite compute throughput, token latency barely improves because data delivery from HBM is the binding constraint.
  **Key equation/principle:** $T_{\text{token}} = \max(T_{\text{compute}}, T_{\text{memory}})$; for a 70B model on H100, compute takes ~0.07 ms while memory transfer takes ~42 ms. The processor spends 99%+ of its time waiting for data.
  **Interaction:** Students choose a model size (7B, 70B, 175B) and an accelerator (A100, H100, B200). They PREDICT the fraction of time the arithmetic units are idle during single-token generation. Sliders for batch size (1 to 128) show how batching transforms the workload from memory-bound to compute-bound. Students discover that at batch size 1, even an H100 achieves less than 1% of peak FLOPS, while batch size 64 crosses the ridge point.

- **Part B -- The Roofline Diagnostic** (~12 min)
  **Concept:** The Roofline Model determines whether a workload is compute-bound or memory-bound. The ridge point ($I_{\text{ridge}} = R_{\text{peak}} / BW$) separates the two regimes.
  **Key equation/principle:** Achievable FLOPS = $\min(R_{\text{peak}}, BW \times I)$ (@eq-roofline). H100 ridge point at ~591 FLOP/byte (FP8).
  **Interaction:** Students place ML workloads (LLM decode batch=1, LLM decode batch=32, CNN training, LLM prefill, LLM training) on an interactive Roofline plot. They drag each workload to where they PREDICT it falls, then the simulation reveals the true arithmetic intensity and achieved performance. A hardware selector (V100, A100, H100, B200) shifts the roofline, showing how the ridge point has risen across generations (making more workloads memory-bound, not fewer).

- **Part C -- The Bandwidth Staircase** (~12 min)
  **Concept:** Data transfer speed drops by orders of magnitude at each physical boundary (HBM to NVLink to PCIe to InfiniBand), and this hierarchy dictates which parallelism strategy operates at which level.
  **Key equation/principle:** Transfer time = Data / Bandwidth. For a 10 GB buffer: 3 ms over HBM, 11 ms over NVLink, 200 ms over InfiniBand. NVLink-to-IB ratio is 18x.
  **Interaction:** Students specify a transfer size (1 MB to 10 GB) and select the interconnect tier. A visualization shows transfer time at each level of the hierarchy as a "staircase" bar chart. Students PREDICT how much slower a 350 GB gradient AllReduce is over InfiniBand vs. NVLink. The simulation then maps parallelism strategies (Tensor Parallelism = NVLink, Pipeline Parallelism = InfiniBand, Data Parallelism = InfiniBand + compression) to the appropriate staircase tier, showing why each strategy is confined to its tier.

- **Part D -- The Node Memory Budget** (~12 min)
  **Concept:** Training a frontier model (175B) requires careful memory budgeting across parameters, optimizer states, gradients, and activations. A single accelerator cannot hold the model; a full 8-GPU node barely suffices.
  **Key equation/principle:** Memory = Parameters (350 GB FP16) + Optimizer (700 GB FP32 Adam) + Gradients (350 GB) + Activations (variable). Total: ~1.4 TB. Per-GPU with ZeRO-3 on 8 GPUs: ~175 GB equivalent, still exceeding 80 GB HBM.
  **Interaction:** Students configure a model (parameters in billions), precision (FP32, FP16, INT8), optimizer (SGD vs. Adam), and number of GPUs per node (1, 2, 4, 8). A stacked memory bar shows how the budget is consumed. Students PREDICT whether a 175B model fits on a single 8-GPU DGX H100 node (640 GB total HBM). They discover it does not without ZeRO optimization, and even with ZeRO-3, activations push memory past the limit.

- **Part E -- TCO: The Hidden Cost of Scale** (~12 min)
  **Concept:** Total Cost of Ownership goes far beyond GPU purchase price. Power, cooling, networking, and utilization efficiency determine the real cost per useful FLOP.
  **Key equation/principle:** TCO = CapEx (GPUs + networking + storage) + OpEx (power + cooling + staff). Power Usage Effectiveness (PUE) multiplier of 1.1 to 1.6 for cooling. At $0.10/kWh, a 1,000-GPU H100 cluster costs ~$3M/year in electricity alone.
  **Interaction:** Students configure a cluster (GPU count, GPU type, networking tier, cooling type) and see TCO broken into components over a 3-year lifecycle. Sliders for utilization (30% to 90%) and PUE (1.1 to 1.6) dramatically shift cost-per-useful-FLOP. Students PREDICT which is cheaper for inference: a small cluster of H100s or a larger cluster of older A100s. The simulation reveals that utilization rate often matters more than hardware generation.

---

## Lab V2-03: Network Fabrics
**Chapter:** `network_fabrics.qmd`
**1-hour story arc:** Students discover that the network is not just a pipe connecting GPUs, but the system bus of the warehouse-scale computer, where topology, protocol choice, and oversubscription ratio determine whether expensive accelerators compute or idle.

### Proposed Parts:

- **Part A -- The Alpha-Beta Crossover** (~12 min)
  **Concept:** Network transfer time has two distinct regimes: latency-dominated (small messages) and bandwidth-dominated (large messages), separated by the crossover point $n^* = \alpha \cdot \beta$.
  **Key equation/principle:** $T(n) = \alpha + n/\beta$ (Hockney model). For IB NDR: $\alpha \approx 1.5 \mu s$, $\beta \approx 50$ GB/s, crossover $n^* \approx 75$ KB.
  **Interaction:** Students adjust $\alpha$ (0.5 to 10 $\mu$s) and $\beta$ (10 to 100 GB/s) via sliders. A log-log plot of transfer time vs. message size updates live, clearly showing the flat latency-dominated region and the linear bandwidth-dominated region. Students send two messages: a 4 KB pipeline control message and a 350 MB gradient shard. They PREDICT which network upgrade (halving latency vs. doubling bandwidth) helps each message more. The simulation reveals that topology (hop count) matters for control messages while raw bandwidth matters for gradients.

- **Part B -- RDMA and the Protocol Tax** (~12 min)
  **Concept:** Traditional TCP/IP imposes an unacceptable CPU overhead at 400 Gbps speeds. RDMA and GPUDirect RDMA bypass the kernel to achieve the low latency that barrier-synchronized training demands. The Go-Back-N fragility of RDMA makes losslessness mandatory.
  **Key equation/principle:** TCP latency: 50-100 $\mu$s (kernel overhead). RDMA latency: 1-2 $\mu$s. GPUDirect eliminates 700 GB of redundant memory copies per step for a 175B model. A single dropped packet in RDMA can stall a 1,024-GPU AllReduce for 100-500 ms.
  **Interaction:** Students simulate a gradient synchronization for 175B parameters over three networking stacks: TCP/IP, RDMA (with host staging), and GPUDirect RDMA (zero-copy). An animation shows the data path for each. Students toggle a "packet drop" switch and watch what happens: TCP retransmits one segment, RDMA retransmits the entire tail of the message. Students PREDICT how long a single dropped packet stalls a 1,024-GPU training job, discovering the Go-Back-N amplification effect.

- **Part C -- Fat-Tree Bisection Bandwidth** (~12 min)
  **Concept:** Network topology determines bisection bandwidth, which is the true throughput ceiling for AllReduce. Oversubscribing the spine layer creates a false economy that wastes compute.
  **Key equation/principle:** Fat-tree hosts: $N = k^3/4$ for radix-$k$ switches. Bisection bandwidth at 1:1 subscription: $N/2 \times \beta$. A 4:1 oversubscription creates a 4x slowdown for AllReduce.
  **Interaction:** Students design a network by selecting switch radix (32, 64, 128), number of tiers (2, 3), and oversubscription ratio (1:1, 2:1, 4:1). A topology diagram visualizes the resulting fat-tree. A "run AllReduce" button animates the gradient flow and displays wall-clock time. Students PREDICT whether saving 75% on spine switches (4:1 oversubscription) is a net cost saving for a $300M training cluster. The simulation shows that the compute time wasted on idle GPUs dwarfs the switch savings.

- **Part D -- Rail-Optimized Topology** (~12 min)
  **Concept:** ML training has a deterministic, stratified communication pattern (GPU-0 talks to GPU-0 across nodes) that rail-optimized topologies exploit by dedicating switch rails to each GPU position.
  **Key equation/principle:** In a rail-optimized network, tensor-parallel communication between corresponding GPUs uses a dedicated rail switch with 1-hop latency, versus 2-3 hops in a standard fat-tree. This reduces TP AllReduce latency by 2-3x for inter-node tensor parallelism.
  **Interaction:** Students compare two topologies for a 128-GPU cluster: a standard fat-tree (all GPUs share ToR switches) vs. a rail-optimized design (8 dedicated rail switches). They assign parallelism strategies (TP within rails, DP across rails) and watch how traffic flows through the network. A congestion heatmap shows hot spots. Students PREDICT whether rail-optimization helps more for TP-heavy (large model) or DP-heavy (small model) workloads, discovering it provides dramatic benefit only when inter-node TP is used.

---

## Lab V2-04: Data Storage
**Chapter:** `data_storage.qmd`
**1-hour story arc:** Students discover that storage -- the least glamorous infrastructure component -- can silently determine whether a training cluster is productive or an expensive space heater, because the gap between compute consumption and storage delivery has widened 60x in seven years.

### Proposed Parts:

- **Part A -- The Storage-Compute Chasm** (~12 min)
  **Concept:** Accelerator throughput has grown 236x (P100 to B200) while NVMe bandwidth grew only 4x over the same period. The resulting 60x widening gap means that data pipeline engineering is now a first-order concern.
  **Key equation/principle:** H100 HBM bandwidth: 3.35 TB/s. NVMe sequential read: 7 GB/s. Ratio: ~479x. The I/O Wall widens by ~5x every 7 years as compute scales faster than storage.
  **Interaction:** Students select a GPU generation (V100 through B200) and see the compute-to-storage bandwidth ratio update in real time. A dual-axis timeline chart (like @fig-storage-compute-chasm) shows the widening gap. Students PREDICT whether a next-generation GPU with 2x more TFLOPS will help a storage-bottlenecked training job. They discover that faster GPUs make the storage problem worse, not better, because the gap widens.

- **Part B -- The Data Pipeline Equation** (~12 min)
  **Concept:** The required storage bandwidth is a function of GPU count, target utilization, batch size, and iteration time. Under-provisioning starves accelerators; over-provisioning wastes money.
  **Key equation/principle:** $BW_{\text{required}} = N_{\text{GPUs}} \times U_{\text{target}} \times S_{\text{batch}} / T_{\text{iteration}}$ (@eq-pipeline-throughput). Data Stall % measures the fraction of each step where the accelerator waits for data.
  **Interaction:** Students configure a training job: GPU count (8 to 1024), batch size, model type (vision vs. language), and iteration time. The simulation calculates required bandwidth and shows where on the storage hierarchy (HDD, Network FS, NVMe, NVMe RAID) the cluster currently operates. A Data Stall Frontier S-curve (like @fig-data-stall-frontier) shows GPU utilization vs. storage bandwidth. Students PREDICT the utilization impact of doubling GPUs from 128 to 256 without upgrading storage, discovering a cliff-like utilization drop.

- **Part C -- The Shard Contention Birthday Problem** (~10 min)
  **Concept:** Even with many dataset shards, random access by many GPUs creates surprisingly high collision probability (the birthday problem), causing tail-latency spikes that stall the entire cluster.
  **Key equation/principle:** $P(\text{collision}) = 1 - e^{-n^2 / 2N}$ (birthday approximation). With 32 workers and 1,000 shards, collision probability is ~40%.
  **Interaction:** Students set the number of GPU workers (8 to 256) and the number of dataset shards (100 to 10,000). An animated visualization shows workers randomly selecting shards, with collisions highlighted in red. A probability meter shows the theoretical collision chance. Students PREDICT the collision probability for 64 workers on 1,000 shards, then discover it exceeds 87%. The simulation then demonstrates deterministic shard assignment as the fix, showing collisions drop to zero.

- **Part D -- The Data Stall Diagnostic** (~12 min)
  **Concept:** Pipelining and prefetching can hide storage latency, but only if the I/O time does not exceed compute time. When I/O exceeds compute, no amount of overlap eliminates the stall.
  **Key equation/principle:** Without pipelining: $T_{\text{step}} = T_{\text{IO}} + T_{\text{compute}}$. With pipelining: $T_{\text{step}} = \max(T_{\text{compute}}, T_{\text{IO}})$. Stall % = $(T_{\text{step}} - T_{\text{compute}}) / T_{\text{step}}$ (@eq-data-stall).
  **Interaction:** Students set compute time (100-500 ms), I/O time (50-1000 ms), and prefetch buffer depth (0 to 8 batches). A timeline visualization shows the pipeline execution: without prefetching, compute and I/O alternate sequentially; with prefetching, they overlap. A stall gauge shows the percentage of wasted GPU time. Students PREDICT whether doubling prefetch depth from 2 to 4 batches eliminates a 20% stall. The simulation reveals that prefetch depth only helps if there is compute time to hide behind; when I/O exceeds compute, the stall persists regardless.

- **Part E -- Checkpoint Economics** (~12 min)
  **Concept:** Checkpointing writes PB-scale data over a training run. The frequency trade-off is between recovery granularity (less wasted compute on failure) and I/O overhead (checkpoint writes compete with training data reads for storage bandwidth).
  **Key equation/principle:** Checkpoint size = Parameters x bytes/param x (1 + optimizer multiplier). For 175B model with Adam: ~1,050 GB per checkpoint. Over 30 days at 30-min intervals: ~4.5 PB total writes.
  **Interaction:** Students configure checkpoint frequency (every 5 min to every 2 hours), model size, and storage bandwidth. Two competing metrics update: (1) "wasted compute on failure" (how many GPU-hours are lost when a failure occurs between checkpoints, given the MTBF from Part A of Lab V2-01) and (2) "checkpoint I/O overhead" (what fraction of storage bandwidth is consumed by checkpointing). Students PREDICT the optimal checkpoint frequency for a 1,000-GPU cluster, discovering that too-frequent checkpointing creates its own bottleneck while too-infrequent checkpointing wastes millions of dollars on recomputation after failures.

---

## Lab V2-05: Distributed Training
**Chapter:** `distributed_training.qmd`
**1-hour story arc:** Students experience the fundamental trade-offs of distributed training by scaling from a single GPU to a 1,024-GPU cluster, discovering that each parallelism strategy solves one constraint while creating another, and that the "Conservation of Overhead" governs all choices.

### Proposed Parts:

- **Part A -- The Communication Wall** (~12 min)
  **Concept:** Data parallelism scales throughput linearly only until gradient synchronization dominates step time. The AllReduce communication overhead grows relative to shrinking per-GPU compute, creating a "Communication Wall."
  **Key equation/principle:** Ring AllReduce transfers $2(N-1)/N \times M$ bytes per worker. For 175B params in FP16: ~700 GB total gradient. Scaling efficiency: $\eta = 1 / (1 + (N-1) \times r)$ where $r$ is the communication fraction.
  **Interaction:** Students scale a data-parallel training job from 1 to 512 GPUs. Sliders for model size (1B to 175B) and interconnect (100G Ethernet, 200G IB HDR, 400G IB NDR) change the communication fraction $r$. A speedup chart (linear vs. actual) and an efficiency gauge update live. Students PREDICT the efficiency of 256-GPU training for a 175B model on 200G InfiniBand. They discover that even with InfiniBand, communication consumes 40%+ of step time at this scale, capping effective speedup far below 256x.

- **Part B -- ZeRO: Trading Communication for Memory** (~12 min)
  **Concept:** ZeRO optimization shards optimizer states, gradients, and parameters across workers to reduce per-GPU memory, but each ZeRO stage increases communication volume, embodying the Conservation of Overhead.
  **Key equation/principle:** ZeRO Stage 1: shard optimizer (saves ~2/3 of optimizer memory). Stage 2: + shard gradients. Stage 3: + shard parameters. Memory per GPU: Stage 0 = 2.45 TB / 8 GPUs is not feasible; ZeRO-3 on 64 GPUs = 38 GB static, but 50 GB activations push total to 88 GB, exceeding A100's 80 GB.
  **Interaction:** Students configure model size (1B to 175B), number of GPUs (8 to 256), and ZeRO stage (0, 1, 2, 3). A stacked bar shows per-GPU memory (parameters, optimizer, gradients, activations) at each stage. A second chart shows communication volume per step at each stage. Students PREDICT whether ZeRO-3 on 64 A100s can train a 175B model. They discover it cannot because activation memory pushes the total past 80 GB, forcing tensor parallelism.

- **Part C -- Pipeline Bubbles: The Idle GPU Problem** (~12 min)
  **Concept:** Pipeline parallelism overlaps computation across model stages using microbatches, but the pipeline "fill" and "drain" phases create idle GPU time (bubbles) that reduce throughput.
  **Key equation/principle:** Pipeline bubble fraction = $(P - 1) / (P + M - 1)$ where $P$ = pipeline stages and $M$ = microbatches. With 4 stages and 4 microbatches, bubble fraction = 3/7 = 43%.
  **Interaction:** Students set pipeline stages (2, 4, 8, 16) and microbatches per step (1 to 32). An animated Gantt chart (like @fig-pipline-parallelism) shows the forward and backward passes flowing through stages, with bubbles highlighted in red. A bubble percentage gauge updates. Students PREDICT the minimum number of microbatches needed to keep bubble fraction below 10% with 8 pipeline stages, discovering that $M \geq 72$ microbatches are needed, which implies very large global batch sizes.

- **Part D -- The Scaling Tax: Amdahl Meets Communication** (~12 min)
  **Concept:** Distributed training obeys an extended Amdahl's Law where the serial fraction includes both algorithmic sequential dependencies and communication overhead. Beyond a critical GPU count, adding hardware reduces cost-efficiency.
  **Key equation/principle:** $T_{\text{step}}(N) = T_{\text{compute}}/N + T_{\text{comm}}(N) - T_{\text{overlap}}$. Efficiency$(N) = T_{\text{compute}} / (N \times T_{\text{step}}(N))$. At 50% communication overhead ($r = 0.5$), maximum speedup asymptotes to $1/r = 2x$ regardless of GPU count.
  **Interaction:** Students configure a workload type (compute-bound ResNet at $r = 0.05$, balanced LLM at $r = 0.20$, bandwidth-bound embedding model at $r = 0.50$). They scale GPUs from 1 to 512 and toggle communication-computation overlap (0%, 50%, 80%). A dual chart shows (1) speedup vs. ideal and (2) cost-per-sample vs. GPU count. Students PREDICT where the cost-efficiency optimal point is for each workload. The simulation reveals that the bandwidth-bound workload hits diminishing returns at just 4 GPUs, while the compute-bound workload scales efficiently to 128+.

- **Part E -- 3D Parallelism: The Hierarchy-Aware Design** (~12 min)
  **Concept:** Production training combines Tensor Parallelism (within node, NVLink), Pipeline Parallelism (across nearby nodes, InfiniBand), and Data Parallelism (across all nodes), respecting the bandwidth hierarchy.
  **Key equation/principle:** TP confined to NVLink (900 GB/s), PP over InfiniBand (50 GB/s, small activation transfers of ~200 MB per boundary), DP over InfiniBand (large gradient AllReduce, tolerates lower bandwidth with compression). Total GPUs = TP x PP x DP.
  **Interaction:** Students configure a 256-GPU training job by selecting the TP degree (1, 2, 4, 8), PP degree (1, 2, 4, 8), and DP degree (auto-calculated). The constraint is TP x PP x DP = 256. A node/rack topology diagram shows the physical mapping. Metrics update for: (1) per-GPU memory usage, (2) communication volume per step, (3) pipeline bubble fraction, and (4) overall scaling efficiency. Students PREDICT the best configuration for a 175B model on 256 H100 GPUs. They explore configurations and discover that TP=8, PP=4, DP=8 (or similar) outperforms naive DP=256 by 2-3x in efficiency because it matches communication patterns to the bandwidth hierarchy.

---

## Lab V2-06: Collective Communication
**Chapter**: `collective_communication.qmd`
**1-hour story arc**: Students discover that choosing the right AllReduce algorithm and mapping it to network topology matters more than raw interconnect bandwidth, because the latency-bandwidth crossover and the NVLink-to-InfiniBand cliff create non-obvious performance cliffs.

### Proposed Parts:

- **Part A -- The Alpha-Beta Crossover** (~10 min)
  **Concept**: The alpha-beta model reveals that small messages are latency-bound and large messages are bandwidth-bound, with a critical crossover size that determines which optimization strategy matters.
  **Key equation**: $T(n) = \alpha + n/\beta$; critical size $n^* = \alpha \cdot \beta$
  **Interaction**: Students set interconnect type (NVLink, InfiniBand NDR, Ethernet 100G) and message size via sliders. They predict whether a given message is latency-bound or bandwidth-bound, then see the alpha-beta decomposition bar chart showing which term dominates. The crossover point shifts visibly when they change interconnect parameters.

- **Part B -- Ring vs Tree: The Algorithm Crossover** (~12 min)
  **Concept**: Ring AllReduce is bandwidth-optimal but latency-poor ($O(N)$ steps); Tree AllReduce has logarithmic latency but wastes bandwidth ($O(\log N)$ factor). The crossover point $M_{\text{crossover}} \approx N \cdot \alpha \cdot \beta$ determines which wins.
  **Key equation**: $T_{\text{ring}} = 2(N-1)\alpha + 2\frac{N-1}{N}\frac{M}{\beta}$ vs $T_{\text{tree}} = 2\log_2 N \cdot \alpha + 2\log_2 N \cdot \frac{M}{\beta}$
  **Interaction**: Students sweep message size (1 KB to 10 GB) on a log slider for a fixed GPU count (64, 256, 1024 selectable). Two animated time bars show Ring vs Tree completion time. Students predict which algorithm wins at 1 MB for 256 GPUs, then discover Tree barely wins because they are near the crossover. They can drag the crossover marker and see it match the formula.

- **Part C -- The Topology Cliff** (~12 min)
  **Concept**: Hierarchical communication reveals that NVLink-to-InfiniBand is an 18x bandwidth cliff. A flat AllReduce algorithm that ignores this hierarchy wastes up to 50% of training throughput.
  **Key equation**: Hierarchical AllReduce: $T = T_{\text{intra-node}}(\beta_{\text{NVLink}}) + T_{\text{inter-node}}(\beta_{\text{IB}})$; NVLink = 900 GB/s vs InfiniBand = 50 GB/s
  **Interaction**: Students place 64 GPUs across 8 nodes using a drag-and-drop topology widget (or toggle between "all intra-node," "scattered," "hierarchical"). They see AllReduce time update in real time. The "aha" is that a hierarchical 2-level AllReduce (local reduce then global AllReduce) dramatically outperforms a flat ring that mixes NVLink and InfiniBand links.

- **Part D -- Gradient Compression: When Does It Pay Off?** (~14 min)
  **Concept**: Gradient compression (quantization, sparsification) trades bandwidth savings for convergence slowdown. It is only worthwhile when the communication-to-computation ratio is high. Error feedback prevents information loss.
  **Key equation**: Total time = $N_{\text{steps}} \times (T_{\text{compute}} + T_{\text{comm}} / \text{compression\_ratio})$; $N_{\text{steps}}$ increases with compression aggressiveness
  **Interaction**: Students set compression method (None, FP16, INT8, Top-K 1%, 1-bit), model size, and network bandwidth. Two outputs: (1) a per-step waterfall showing compute vs communication, and (2) a total training time curve that accounts for extra convergence steps. Students discover that INT8 compression helps on slow networks but *hurts* on fast networks because the extra convergence steps outweigh per-step savings. A toggle for error feedback shows how disabling it causes the loss curve to plateau.

- **Part E -- The Communication Budget: Putting It All Together** (~12 min)
  **Concept**: For a 70B model, AllReduce takes 11+ seconds per step without optimization. Students assemble a communication strategy (algorithm choice, hierarchical decomposition, compression, overlap) and see how each technique chips away at the budget.
  **Key equation**: $T_{\text{AllReduce}} = 2\frac{N-1}{N}\frac{M}{\beta}$ for ring; overlap: $T_{\text{step}} = \max(T_{\text{backward}}, T_{\text{comm}}) + T_{\text{forward}}$
  **Interaction**: Students start with the raw 11-second AllReduce cost from the chapter's napkin math (70B model, 64 GPUs, 50 GB/s). They apply optimizations one at a time via checkboxes: hierarchical AllReduce, FP16 gradients, bucket fusion, backward overlap. A stacked bar chart shows the step time decomposition updating as each technique is toggled. The design challenge: reduce communication to under 20% of total step time.

---

## Lab V2-07: Fault Tolerance
**Chapter**: `fault_tolerance.qmd`
**1-hour story arc**: Students discover that at fleet scale, failure is a statistical certainty (not a rare event), and the optimal checkpoint interval is governed by the Young-Daly square-root law -- checkpointing too often or too rarely both waste massive compute.

### Proposed Parts:

- **Part A -- The Inevitability of Failure** (~10 min)
  **Concept**: System MTBF = Component_MTBF / N. A 10,000-GPU cluster with individually reliable components (50,000-hour MTBF per GPU) experiences a failure every 5 hours.
  **Key equation**: $\text{MTBF}_{\text{system}} = \frac{\text{MTBF}_{\text{component}}}{N}$; $R_{\text{system}}(t) = e^{-N\lambda t}$
  **Interaction**: Students drag a "cluster size" slider from 8 to 25,000 GPUs. A probability gauge shows the chance of surviving 1 hour without failure. Students predict the survival probability for 10,000 GPUs (most guess >90%), then discover it is approximately 32%. A second visualization shows expected failures per day overlaid on training durations (1 day, 1 week, 3 months).

- **Part B -- The Young-Daly Sweet Spot** (~12 min)
  **Concept**: The optimal checkpoint interval $\tau_{\text{opt}} = \sqrt{2 \cdot T_{\text{write}} \cdot \text{MTBF}}$ minimizes total wasted work (checkpoint overhead + rework from failures). The U-shaped cost curve has a clear minimum.
  **Key equation**: $\tau_{\text{opt}} = \sqrt{2 \cdot T_{\text{write}} \cdot \text{MTBF}}$; total overhead = $T_{\text{write}}/\tau + \tau/(2 \cdot \text{MTBF})$
  **Interaction**: Students set cluster size (which determines MTBF) and checkpoint write time (which depends on model size and storage bandwidth). They drag a "checkpoint interval" slider and watch the U-shaped waste curve in real time. Three components are visible: checkpoint overhead (decreasing hyperbola), expected rework (increasing line), and total waste (U-curve). Students predict the optimal interval for a 16,000-GPU cluster with 2-minute writes, then discover it is roughly 27 minutes.

- **Part C -- The Checkpoint Storm** (~12 min)
  **Concept**: When thousands of GPUs write checkpoints simultaneously, the storage system saturates. The "stop-the-world" cost scales with model size and cluster size, creating a tension between checkpoint frequency and I/O bandwidth.
  **Key equation**: Checkpoint size = $P \times 12$ bytes (weights + Adam states); write time = size / storage_bandwidth; cost per checkpoint = GPU_count x idle_time x hourly_rate
  **Interaction**: Students configure model size (1B to 175B), storage bandwidth (1 GB/s NFS to 100 GB/s parallel filesystem), and number of workers. Outputs show: (1) checkpoint size in TB, (2) write time, (3) dollar cost per checkpoint, and (4) daily checkpoint cost if using the Young-Daly optimal interval. The failure state triggers when storage bandwidth is so low that checkpoint write time exceeds the optimal interval (the system spends more time checkpointing than computing).

- **Part D -- Graceful Degradation in Serving** (~14 min)
  **Concept**: Serving fault tolerance differs fundamentally from training: failures must be invisible to users (millisecond recovery, not minute recovery). Strategies include model fallback (use a smaller model), feature fallback (drop expensive features), and load shedding.
  **Key equation**: Availability = $1 - \prod_{i=1}^{n}(1 - R_i)$ for parallel replicas; Serving latency budget decomposition from @eq-serving-tax
  **Interaction**: Students operate a serving fleet simulator. They set number of replicas, failure rate, and SLO budget. When a replica fails, they choose a degradation strategy: redirect (adds latency from rebalancing), fallback model (reduces quality), or shed load (drops requests). A live dashboard shows P99 latency, accuracy, and request success rate. Students discover that for LLM serving, simple redirection fails because KV cache state is lost, while model fallback maintains the latency SLO at the cost of response quality.

- **Part E -- The Reliability Budget** (~12 min)
  **Concept**: Fault tolerance investment has diminishing returns. The economic framework balances the cost of redundancy against the cost of downtime, with different optimal points for training (checkpoint frequency) and serving (replica count).
  **Key equation**: Cost_of_protection = checkpoint_overhead x training_cost + replica_count x serving_cost; Cost_of_failure = MTTF_failures x recovery_time x hourly_rate
  **Interaction**: Students allocate a fixed GPU budget between "productive compute" and "fault tolerance overhead" (checkpointing, spare capacity, replicas). A Pareto curve shows the trade-off between effective throughput and reliability. The optimal allocation point shifts as cluster size changes. Students discover that larger clusters require proportionally more fault tolerance investment, creating a "reliability tax" that limits the scaling benefits of adding more GPUs.

---

## Lab V2-08: Fleet Orchestration
**Chapter**: `fleet_orchestration.qmd`
**1-hour story arc**: Students discover that scheduling GPUs is fundamentally harder than scheduling CPUs because of gang scheduling constraints, topology sensitivity, and heavy-tailed job duration distributions that make high utilization feel broken.

### Proposed Parts:

- **Part A -- The Queuing Wall** (~10 min)
  **Concept**: ML workloads have heavy-tailed duration distributions ($C_s \approx 3$-5) that make queue wait times explode at utilizations where web servers feel responsive. At 80% utilization, ML queue wait is 5x worse than uniform workloads.
  **Key equation**: $W_q = \frac{\rho}{1-\rho} \cdot \frac{1+C_s^2}{2\mu}$ (Pollaczek-Khinchine formula)
  **Interaction**: Students set cluster utilization (0-100%) and workload type (uniform $C_s$=1 vs ML $C_s$=3). A queue depth animation shows jobs arriving and being served. Students predict wait time at 80% utilization for ML workloads (most assume it is similar to web serving), then discover it is 5x worse. The visualization makes the heavy tail viscerally visible: rare but massive training jobs block hundreds of short experiments.

- **Part B -- The Fragmentation Tax** (~12 min)
  **Concept**: Multi-dimensional bin packing with GPU, CPU, memory, and topology constraints creates fragmentation where the cluster has 30% free GPUs but cannot schedule any pending job because the free GPUs are scattered across nodes.
  **Key equation**: Effective capacity = total GPUs - stranded GPUs; fragmentation ratio = stranded / total
  **Interaction**: Students operate a cluster heatmap (8 nodes x 8 GPUs). Jobs of varying sizes (1, 2, 4, 8 GPUs) arrive in a queue. Students manually place jobs or use scheduling heuristics (first-fit, best-fit, first-fit-decreasing). They see fragmentation accumulate as mismatched jobs leave stranded GPUs. The challenge: maintain >70% utilization while serving the job queue without deadlocking.

- **Part C -- Gang Scheduling and Deadlock** (~12 min)
  **Concept**: Distributed training requires all-or-nothing allocation. Partial allocation creates deadlock where multiple jobs each hold some GPUs while waiting for more, and no job can make progress. The deadlock wastes 100% of held resources.
  **Key equation**: Deadlock condition: $\sum_{j} \text{allocated}(j) = N$ but $\forall j: \text{allocated}(j) < \text{required}(j)$
  **Interaction**: Students submit 2-4 large jobs to a cluster simulator. Without gang scheduling, the naive scheduler allocates partial resources and creates a deadlock (visualized as a dependency cycle). Students then toggle gang scheduling on and see the scheduler either run the job fully or queue it entirely. The trade-off becomes clear: gang scheduling prevents deadlock but may leave GPUs idle while waiting for a large contiguous block.

- **Part D -- Topology-Aware Placement** (~14 min)
  **Concept**: Random GPU placement across a datacenter can degrade training throughput by 30-50% compared to topology-aware placement, because the NVLink-to-InfiniBand bandwidth cliff (900 GB/s vs 50 GB/s) compounds at every communication step.
  **Key equation**: $S_{\text{locality}} = \sum_{i<j} w(d(g_i, g_j))$; topology-aware placement yields up to 4.8x AllReduce speedup
  **Interaction**: Students place a 64-GPU training job on a cluster topology visualization (nodes within racks, racks connected by spine switches). They choose between random, rack-aware, and topology-optimal placement. A live AllReduce latency meter shows the impact. Students predict the speedup from topology-aware placement (most underestimate the 4.8x factor) and discover that placement alone, with zero code changes, can match the impact of an algorithmic optimization.

- **Part E -- The Utilization Paradox** (~12 min)
  **Concept**: Maximizing GPU utilization, fairness, job latency, and cost efficiency simultaneously is impossible. Every scheduling policy represents a trade-off point. The conflict between throughput (favor large jobs) and latency (favor small jobs) is the central tension.
  **Key equation**: Throughput-fairness trade-off; preemption cost = lost_work_since_checkpoint x GPU_count x hourly_rate
  **Interaction**: Students operate a scheduling policy dashboard with sliders for priority weighting (throughput vs fairness vs latency). A mixed job queue (one 512-GPU month-long training run, hundreds of 8-GPU hour-long experiments) flows through. Four metrics update in real time: cluster utilization, average wait time, max wait time, and fairness index. Students discover they cannot make all four metrics green simultaneously and must choose which metric to sacrifice.

---

## Lab V2-09: Inference at Scale
**Chapter**: `inference.qmd`
**1-hour story arc**: Students discover that inference economics dominate training economics (serving cost exceeds training cost within weeks), and that the KV cache memory wall -- not compute -- is the binding constraint for LLM serving at scale.

### Proposed Parts:

- **Part A -- The Serving Cost Multiplier** (~10 min)
  **Concept**: Serving cost dominates training cost by 10-1000x over a model's lifetime. A 10% inference optimization saves more money in a month than the entire training cost.
  **Key equation**: $C_{\text{total}} = C_{\text{training}} + C_{\text{serving}} \times T_{\text{deployment}} \times Q_{\text{rate}}$
  **Interaction**: Students set training cost, QPS, cost per query, and deployment duration. An animated cumulative cost curve shows training as a flat line and serving as a growing area. Students predict the crossover month (when serving exceeds training). For a 70B LLM serving 1M DAU, the crossover occurs within 6 weeks. For recommendation systems at 100M DAU, it is within days. A slider for "inference optimization %" shows the leverage: a 10% serving improvement saves $1.8M/year against a $2M training cost.

- **Part B -- The Batching Efficiency Curve** (~12 min)
  **Concept**: Batching trades latency for throughput. Vision models saturate at small batches (high $T_{\text{variable}}$); LLMs require massive batches to amortize the weight-loading overhead (high $T_{\text{fixed}}$). The "knee" of the curve differs by 40x between model types.
  **Key equation**: $X(B) = B / (T_{\text{fixed}} + B \times T_{\text{variable}})$; knee at $B \approx T_{\text{fixed}} / T_{\text{variable}}$
  **Interaction**: Students toggle between model types (Vision CNN, LLM decode, RecSys) and drag a batch size slider. Two live charts: (1) throughput vs batch size showing the knee, and (2) per-request latency vs batch size with an SLO line. Students predict the knee for LLM decode (most guess 8-16), then discover it is 80 because the weight-loading overhead is so large. A "maximum batch under SLO" calculator shows that the optimal operating point is just below the SLO line.

- **Part C -- The KV Cache Wall** (~14 min)
  **Concept**: KV cache memory grows linearly with both sequence length and batch size, creating a "memory wall" that limits concurrent serving capacity. At 128K context, even an 8xH100 node can serve only 1 concurrent request for a 70B model.
  **Key equation**: KV cache size = $2 \times L \times H \times S \times B \times P$; available memory = HBM - weights - system overhead
  **Interaction**: Students set model size (7B, 70B, 175B), precision (FP16, INT8, INT4 for weights), context length (2K to 128K), and GPU count (1 to 8). A stacked memory bar shows weights (fixed) vs KV cache (growing) vs HBM capacity line. Students drag the context length slider and watch the KV cache consume all available memory. They predict the maximum batch size at 128K context for a 70B model on 8xH100 (most guess 4-8), then discover it is 1. The failure state: an "OOM" banner when KV cache + weights exceed HBM.

- **Part D -- Power of Two Choices** (~12 min)
  **Concept**: Random load balancing creates $O(\log n / \log \log n)$ maximum queue depth. Querying just 2 random servers and picking the less loaded one reduces this exponentially to $O(\log \log n)$. This single change halves P99 tail latency.
  **Key equation**: Random max queue: $\Theta(\log n / \log \log n)$; Two choices: $\Theta(\log \log n)$
  **Interaction**: Students run a load balancer simulation with 100 replicas and variable request arrival rate. They toggle between random, round-robin, and power-of-two-choices. A histogram of queue depths across replicas updates in real time. Students predict the P99 latency improvement from two-choices (most guess 10-20%), then discover it cuts the maximum queue depth in half. The animation makes the "tail compression" visually obvious: the right tail of the queue depth distribution collapses.

- **Part E -- The Inference System Design Challenge** (~12 min)
  **Concept**: Designing a serving system requires jointly optimizing batch size, model parallelism, quantization, and replica count under a latency SLO and cost budget. These interact non-linearly: quantization frees memory for larger batches, but larger batches increase latency.
  **Key equation**: $L_{\text{total}} = L_{\text{compute}} + L_{\text{network}} + L_{\text{serialization}} + L_{\text{queuing}}$; cost = replicas x GPU_hours x rate
  **Interaction**: Students configure a serving deployment: model precision (FP16, INT8, INT4), batch size, number of GPUs per replica (tensor parallelism degree), and replica count. A dashboard shows: throughput (QPS), P99 latency, cost per 1M tokens, and memory utilization. The constraint: serve 10,000 QPS at <200ms P99 for minimum cost. Students discover that INT4 quantization is not just a quality trade-off -- it frees KV cache memory, enabling 2x higher batch sizes, which more than compensates for the quality loss in throughput terms.

---

## Lab V2-10: Performance Engineering
**Chapter**: `performance_engineering.qmd`
**1-hour story arc**: Students discover that the memory wall -- not compute -- is the dominant bottleneck for modern ML workloads, and that the correct optimization strategy depends entirely on where the workload falls on the roofline model. Applying the wrong optimization yields zero improvement.

### Proposed Parts:

- **Part A -- The Roofline Diagnostic** (~10 min)
  **Concept**: The roofline model reveals whether a workload is compute-bound or memory-bound. The ridge point (FLOPS/bandwidth) has shifted 4x across GPU generations, making more workloads memory-bound with each new chip.
  **Key equation**: Achievable FLOPS = $\min(P, B \times I)$; ridge point $I_{\text{ridge}} = P/B$
  **Interaction**: Students select a GPU generation (V100, A100, H100, B200) and an ML operation (LayerNorm, Attention, large GEMM, LLM decode). The roofline plot highlights where the workload falls. Students predict whether LLM decode at batch size 1 is compute-bound or memory-bound on the H100 (most guess compute-bound because "H100 is fast"), then discover it is deeply memory-bound at ~1 FLOP/byte, far below the 295 FLOP/byte ridge point. Toggling GPU generations shows the same workload moving further below the ridge with each generation.

- **Part B -- Operator Fusion: Eliminating the HBM Round-Trip** (~12 min)
  **Concept**: A naive transformer layer executes 50+ kernel launches, each materializing intermediate tensors in HBM. Operator fusion reduces HBM traffic by 60-80% by keeping intermediates in SRAM. The savings compound across layers.
  **Key equation**: Unfused HBM traffic = $\sum_{\text{ops}} (\text{read} + \text{write})$; Fused traffic = initial read + final write; savings = intermediates eliminated
  **Interaction**: Students view a transformer layer computation graph. Each node (MatMul, GELU, LayerNorm, Dropout) has a "fused/unfused" toggle. An HBM traffic counter and a timeline show bytes moved and wall-clock time. Students progressively fuse operations and watch HBM traffic drop. The key discovery: fusing the three element-wise operations after the GEMM saves 64 MB per layer, but FlashAttention (fusing the attention block) saves 4 GB per layer. The relative impact is the lesson: not all fusions are equal.

- **Part C -- FlashAttention: Tiling Beats Brute Force** (~14 min)
  **Concept**: Standard attention materializes the $N \times N$ score matrix in HBM ($O(N^2)$ memory). FlashAttention tiles the computation to keep running statistics in SRAM, reducing memory to $O(N)$. The savings grow quadratically with sequence length.
  **Key equation**: Naive HBM traffic: $2 \times N^2 \times \text{bytes} \times \text{heads}$; Flash: $4 \times N \times d \times \text{bytes} \times \text{heads}$; savings ratio $\approx N / (2d)$
  **Interaction**: Students sweep sequence length from 512 to 128K on a log slider. Two memory bars show naive vs FlashAttention memory usage. At 8K, the savings are 32x. At 64K, the savings exceed 32,000x. Students predict the savings ratio at 32K (most significantly underestimate because they think linearly, not quadratically). A tile animation shows how the SRAM-based tiling works: blocks of Q iterate over blocks of K/V, with running softmax statistics maintained in registers.

- **Part D -- Precision Engineering: Bytes vs Quality** (~12 min)
  **Concept**: Reducing precision from FP16 to FP8 or INT4 halves or quarters the bytes moved per weight, doubling or quadrupling effective memory bandwidth. But outlier features in transformer models cause catastrophic accuracy loss with naive quantization, requiring block-wise or rotation-based approaches.
  **Key equation**: Effective bandwidth = HBM_bandwidth x (FP16_bytes / quantized_bytes); quality impact varies by method (GPTQ, AWQ, SmoothQuant)
  **Interaction**: Students set quantization precision (FP16, FP8, INT8, INT4) and method (naive, block-wise, AWQ). Two outputs: (1) a roofline plot showing the workload shifting as effective bandwidth changes, and (2) a quality metric (perplexity or accuracy). Students discover that naive INT4 quantization catastrophically degrades quality for large models (due to outlier channels), while AWQ preserves quality by protecting the 1% of salient weights. The roofline shift is the key visual: quantization moves the workload *closer to* the ridge point by increasing effective bandwidth.

- **Part E -- The Optimization Playbook** (~12 min)
  **Concept**: Applying the wrong optimization yields zero improvement. The diagnostic sequence is: profile the workload (roofline), identify the bottleneck (compute, memory, or overhead), then apply the matching technique. Students must diagnose before optimizing.
  **Key equation**: Iron Law: $\text{Time} = \max(\text{Compute}/\text{FLOPS}, \text{Memory}/\text{BW}) + \text{Overhead}$
  **Interaction**: Students receive a "mystery workload" (randomly assigned: LLM decode at batch 1, prefill at batch 64, or vision inference). A profiling view shows the time breakdown. Students must first diagnose the bottleneck using the roofline, then select optimizations from a menu (FlashAttention, INT4 quantization, CUDA Graphs, larger batch size, operator fusion). The simulation shows that applying the correct optimization yields 2-4x speedup, while applying the wrong one (e.g., FlashAttention on a compute-bound prefill workload) yields <5% improvement. The structured reflection asks students to match each bottleneck type to its optimization.

---

## Lab V2-11: Edge Intelligence
**Chapter:** `edge_intelligence.qmd`
**1-hour story arc:** Students discover that on-device training is not inference-plus-a-bit-extra but a thermodynamic battle, and that federated learning's communication cost explodes under non-IID data, forcing a radical rethinking of how much local computation to do.

### Proposed Parts:

- **Part A -- The Memory Amplification Tax** (~10 min)
  Concept: On-device training requires 4-12x more memory than inference due to activation caching, gradients, and optimizer state.
  Key equation/principle: `Memory_train = W + W_grads + 2W_optimizer(Adam) + A*B*L_activations`; 10M param model: 40 MB inference balloons to 200+ MB training peak. (@tbl-training-amplification, @fig-training-memory-amplifier)
  Interaction: Students predict memory for fine-tuning a 10M param model on a smartphone (most guess ~60-120 MB; actual ~200-360 MB). Slider for model scale (1M-100M) and adaptation strategy (Full/LoRA/Freeze) against a 300 MB smartphone RAM ceiling. OOM failure state turns bars red.

- **Part B -- The Adaptation Strategy Selector** (~10 min)
  Concept: LoRA and weight freezing reduce trainable parameters by 100-1000x, making on-device learning feasible where full fine-tuning causes OOM.
  Key equation/principle: LoRA storage savings: 200x per-context (40 MB full vs. 200 KB adapter). (@sec-edge-intelligence-model-adaptation-6a82, StorageWall class showing 200x savings)
  Interaction: Students toggle between Full Fine-Tune / LoRA rank-16 / Bias-Only for a multi-context personalization scenario (10 user contexts). See storage explode to 400 MB for full fine-tuning vs. 42 MB with adapters. Compute and convergence quality trade-off bars update dynamically.

- **Part C -- The Battery Drain Reality** (~8 min)
  Concept: Energy consumption for on-device training is 10-50x worse than inference; NPU vs. CPU makes the difference between feasible and impossible.
  Key equation/principle: NPU gives 20x latency speedup and 50x energy gain over CPU. Battery drain = (Power_W * Duration_H) / Battery_Wh * 100; ~15% drain for a fine-tuning session. (EdgeNpuSpeedup class, battery drain notebook)
  Interaction: Students configure training power (CPU vs. NPU), duration, and see battery percentage drain in real time. Failure state: battery drops below 20% threshold, showing the device would thermally throttle or be unusable.

- **Part D -- The Federation Paradox** (~15 min)
  Concept: Non-IID data causes federated learning communication rounds to explode (up to 28x), and increasing local epochs (E) beyond 2-5 degrades convergence due to client drift.
  Key equation/principle: `Total_time = comm_cost/E + compute_cost*E + overhead`; optimal E shifts right as bandwidth decreases. FedAvg: `theta^{t+1} = sum(n_k/n * theta_k^{t+1})`. (@fig-fl-communication-computation, convergence bound with heterogeneity penalty beta)
  Interaction: Students set data heterogeneity (beta: 0-2), local epochs (E: 1-20), and number of clients (C: 10-500). Convergence plot shows IID vs. non-IID curves. Communication budget cap creates failure state when budget exhausts before accuracy target.

- **Part E -- The Communication-Compression Trade-off** (~7 min)
  Concept: Gradient compression (quantization, sparsification) reduces per-round communication by 4-100x but can degrade convergence fidelity, especially under non-IID conditions.
  Key equation/principle: Bandwidth reduction: raw data (195 MB/week) vs. model update (2.5 MB) = 78x savings. Quantized updates (4-bit) vs. FP32 gradients. (FederatedSavings class, gradient compression techniques)
  Interaction: Students toggle between Standard (FP32), Quantized (INT8/INT4), and Sparsified (Top-K) gradient compression. See per-round bytes drop by 4-32x but convergence curves shift, requiring more rounds. Net communication cost display shows the optimal compression strategy depends on bandwidth tier.

---

## Lab V2-12: ML Ops at Scale
**Chapter:** `ops_scale.qmd`
**1-hour story arc:** Students discover that managing 100 models is not 100x the work of one model but a qualitatively different problem driven by quadratic dependency growth, and that silent model regressions cost millions before anyone notices.

### Proposed Parts:

- **Part A -- The Complexity Explosion** (~10 min)
  Concept: Operational complexity scales superlinearly with model count: alerts O(N), coordination O(N log N), dependencies O(N^2). Total crosses team capacity at ~50 models.
  Key equation/principle: `Complexity_total = 20N + N*log(N) + 0.5*N^2`; at N=50, total exceeds team capacity of ~4,000. (@fig-n-models-complexity, @tbl-ops-scale-complexity)
  Interaction: Students predict when team capacity is exceeded (most guess 100-200 models; actual ~50). Slider for model count (1-500). Three complexity curves plus total. Toggle for "Platform ON/OFF" that flattens the O(N^2) dependency curve to O(N log N). Failure state: total line crosses capacity threshold.

- **Part B -- The Platform ROI Calculator** (~8 min)
  Concept: A shared ML platform breaks even at far fewer models than organizations expect (~20 for a $2M platform), because per-model savings compound against a fixed platform cost.
  Key equation/principle: `ROI = (N_models * T_saved * C_engineer) / C_platform`; break-even at N=20 for $2M/year. Multi-tenant sharing: 70% idle to 30% idle = 57% savings. (@eq-platform-roi, @fig-platform-roi-threshold, MultiTenantEfficiency class)
  Interaction: Students configure number of models and platform cost tier ($2M/$5M). ROI gauge shows break-even point with green/red coloring. Secondary display shows multi-tenant utilization gain (30% dedicated vs. 70% shared).

- **Part C -- The Silent Failure Tax** (~12 min)
  Concept: Model regressions are silent (no crashes, no errors), and at scale the financial cost accumulates catastrophically: a 0.5% CTR drop at 5,000 QPS costs $1.08M in 24 hours.
  Key equation/principle: `Loss = QPS * 3600 * T_detection * (CTR_base - CTR_new) * Value_per_click`. At 5,000 QPS, 0.5% CTR drop, $0.50/click: $1,080,000/day. (SilentFailure class, @sec-ml-operations-scale-staged-rollout-strategies-2d1f)
  Interaction: Students configure QPS, CTR drop magnitude, and detection time. Real-time dollar counter accumulates loss. Reveals that detection latency (not the regression itself) is the cost multiplier. Failure state: loss exceeds $100K with banner showing the $1.08M chapter reference.

- **Part D -- The Canary Duration Designer** (~12 min)
  Concept: Staged rollouts require a statistically determined minimum observation window; the canary duration formula connects sample requirements to traffic volume and canary percentage.
  Key equation/principle: `t_stage = n_samples_needed / (r_requests * p_stage)`. At 1% canary with 1M req/hr: 1 hour minimum. At 5% canary: 12 minutes. (@eq-canary-duration, worked example lines 1389-1406)
  Interaction: Students design a rollout schedule (1% -> 5% -> 25% -> 50% -> 100%) by configuring canary percentage, request rate, and detection sensitivity. Timeline visualization shows stage durations. Dual failure states: (1) undetected regression if detection window > 24h; (2) deployment stall if total rollout > 48h.

- **Part E -- The Alert Fatigue Wall** (~8 min)
  Concept: At fleet scale, even highly specific alerts (99.7%, 3-sigma) produce hundreds of false alarms per day, making raw per-metric alerting useless. Hierarchical monitoring is mandatory.
  Key equation/principle: `P(at least one false alert) = 1 - (1-alpha)^N`; with alpha=0.003 (3-sigma), N=1,000 monitors, checked every 5 min: ~864 false alerts/day. (@eq-false-alert-rate, FalseAlarmTax class)
  Interaction: Students configure number of models (10-500), metrics per model (5-20), and alert threshold (2-sigma/3-sigma/4-sigma). Daily false alarm counter updates. Toggle for "Hierarchical Aggregation" that reduces effective N by grouping correlated metrics, dramatically cutting false alarms. Reveals why portfolio-level monitoring beats per-model alerting.

---

## Lab V2-13: Security & Privacy
**Chapter:** `security_privacy.qmd`
**1-hour story arc:** Students discover that privacy has a quantifiable cost that scales inversely with dataset size, that every security defense extracts a measurable throughput tax, and that the privacy budget depletes with use rather than being a switch to flip.

### Proposed Parts:

- **Part A -- The Privacy Scaling Wall** (~10 min)
  Concept: Differential privacy noise scale is Sensitivity/epsilon, and per-person error scales as 1/N. Privacy "kills utility" for small datasets because the noise magnitude is independent of dataset size while error per record is inversely proportional.
  Key equation/principle: `Noise_scale = S/epsilon`; `Error_per_person = noise_scale/N`. N=1,000: $200 error; N=100: $2,000 error at epsilon=1, S=$200K. (DPCostAnalysis class)
  Interaction: Students predict per-person error for N=100 at epsilon=1 (most guess ~$200-500; actual $2,000). Log-log plot of error vs. N for three epsilon values with green/yellow/red utility zones. Sliders for epsilon and N. Failure state: curve enters red "utility destroyed" zone for small N.

- **Part B -- The Privacy-Utility Frontier** (~10 min)
  Concept: The epsilon parameter controls a continuous privacy-accuracy trade-off; published results show MNIST retains 95% accuracy at epsilon~1 while CIFAR-10 struggles to reach 82% at epsilon=8. The "knee" at epsilon 1-3 marks the transition from practical to catastrophic.
  Key equation/principle: DP-SGD noise: `sigma >= (C * sqrt(2*ln(1.25/delta))) / epsilon`. At epsilon=1.0, delta=10^-5: sigma ~4.8 (nearly 5x the gradient signal). (@fig-privacy-utility-frontier, Privacy-Accuracy Tax callout)
  Interaction: Students set epsilon and task complexity (MNIST/CIFAR-10), then see how accuracy drops. Annotation of the "knee region" where marginal privacy improvement causes disproportionate accuracy collapse. Overlay shows noise-to-signal ratio growing as epsilon shrinks.

- **Part C -- The Model Extraction Economy** (~10 min)
  Concept: Approximate model theft via API queries is shockingly cheap ($1-$12 for smaller models, $200-$8,000 for GPT-3.5 class). Defense requires trading API utility for extraction resistance.
  Key equation/principle: Query budget vs. extraction fidelity from @tbl-openai-theft: OpenAI ada extracted with <2M queries at RMSE ~5*10^-4 for $1-$4. Defense: adaptive rate limiting `limit_effective = limit_base * exp(-alpha * anomaly_score)`.
  Interaction: Students configure defense parameters (rate limits, output precision/rounding, noise sigma). See extraction fidelity drop as defenses increase, but also see legitimate user experience degrade. Cost-benefit dashboard shows defense overhead vs. extraction cost to attacker.

- **Part D -- The Defense Tax Waterfall** (~12 min)
  Concept: Every security measure (MIG isolation, noise injection, output rounding, rate limiting) extracts a measurable throughput cost. MIG alone costs 15%. Full defense stack can reduce throughput by 30-40%.
  Key equation/principle: `T_secure = T_peak * (1 - 0.15) = 850 tokens/sec` (MIG). Defense overhead: monitoring 1.5 ms + perturbation 1.0 ms = 2.5 ms total = 3% of 100 ms baseline. (MultiTenantIsolation class, DefenseOverhead class)
  Interaction: Students build a defense stack layer by layer via toggles. Waterfall chart shows throughput dropping with each defense. Dual failure states: (1) throughput drops below 800 tokens/sec product requirement; (2) privacy budget exhausts at high query volume. Students must find the combination that satisfies both the privacy officer (epsilon < 1.0) and product manager (throughput > 800).

- **Part E -- The Privacy Budget Depletion** (~8 min)
  Concept: The privacy budget epsilon is finite and non-renewable across queries. Composition theorems show that after T queries, total privacy loss grows as sqrt(T)*epsilon_per_query (advanced composition) or T*epsilon_per_query (basic). Systems must enforce query budgets or risk cumulative leakage.
  Key equation/principle: Basic composition: `epsilon_total = T * epsilon_per_query`. Advanced: `epsilon_total = sqrt(2T*ln(1/delta)) * epsilon_per_query + T*epsilon_per_query*(e^epsilon_per_query - 1)`. (@sec-security-privacy-privacy-budget-composition-edbe)
  Interaction: Circular gauge showing epsilon budget depleting with each query. Students set daily query volume and per-query epsilon, watch budget drain. When exhausted, all subsequent queries rejected (availability = 0). Reveals the fundamental tension: more queries = more leakage, forcing explicit query budgeting in production.

---

## Lab V2-14: Robust AI
**Chapter:** `robust_ai.qmd`
**1-hour story arc:** Students discover that robustness is a budget you spend (not a switch you flip), that adversarial training costs 26 percentage points of clean accuracy plus 8x compute, and that external guardrails (monitoring + detection) are more economical than universal robustification for most systems.

### Proposed Parts:

- **Part A -- The Robustness Tax** (~10 min)
  Concept: Adversarial training at epsilon=8/255 drops ResNet-50 clean accuracy from 76% to 50% (26 pp loss) and costs 8x compute per epoch (PGD-7: 1 standard + 7 attack passes).
  Key equation/principle: `Robustness_Tax = Accuracy_standard - Accuracy_robust = 76% - 50% = 26%`. `Compute_penalty(PGD-K) = 1 + K = 8x`. (RobustnessTaxAnalysis class, AdversarialPayback class)
  Interaction: Students predict clean accuracy of adversarially trained ResNet-50 (most guess ~65-74%; actual 50%). Grouped bar chart shows clean vs. robust accuracy for four defense types (None, Adversarial Training, Randomized Smoothing, Feature Squeezing). Compute cost bars show 1x, 8x, 100,000x respectively.

- **Part B -- Silent Errors at Scale** (~10 min)
  Concept: At cluster scale (10K+ GPUs), silent data corruption becomes statistically certain. A single bit flip can drop ResNet-50 accuracy from 76% to 11%.
  Key equation/principle: `P(>=1 SDC) = 1 - (1-p)^N`. At Meta's reported rate (p=10^-4/hr), N=10,000 GPUs: P~0.63 per hour. (SilentErrorProbability class, @fig-silent-error-probability)
  Interaction: Students set cluster size (1-100,000) and per-device error rate (10^-3/10^-4/10^-5). Three S-curves show how quickly probability approaches 1.0. Annotation at the "effectively certain" threshold. Reveals why redundancy and ECC are mandatory at scale, not optional.

- **Part C -- The Distribution Drift Timeline** (~12 min)
  Concept: Unmonitored models silently degrade 20-40% over 6-12 months under distribution shift. PSI-based monitoring detects drift 3-6 weeks before accuracy breaches SLA, enabling retraining.
  Key equation/principle: PSI = sum((A_i - E_i) * ln(A_i/E_i)); PSI < 0.1 stable, >= 0.25 action required. Detection latency: N = (Z_a + Z_b)^2 * (p1*q1 + p2*q2) / (p1-p2)^2 ~2,200 samples for 2% drop. (@eq-psi-monitoring, @fig-distribution-shift-detector, DriftLatency class)
  Interaction: Dual-line time series: unmonitored model (accuracy degrades silently) vs. monitored model (PSI triggers retraining, accuracy recovers). Students set monitoring frequency (None/Monthly/Weekly/Daily) and drift rate. Compute monitoring lag: at 1,000 labeled samples/hour, a 2% accuracy drop takes ~2.2 hours to detect statistically.

- **Part D -- The Defense Stack Builder** (~15 min)
  Concept: Every defense layer has a quantifiable cost. The only economically viable strategy for most systems is detection + monitoring (external guardrails), not universal adversarial training. Feature squeezing eliminates 70-90% of attacks at <2x overhead vs. adversarial training's 8x.
  Key equation/principle: Feature squeezing: 70-90% attack elimination at 95%+ clean accuracy, ~1x compute. Confidence thresholds: reject 5-15% traffic. Monitoring: 5-15% overhead. Combined: ~1.2x total vs. adversarial training at 8x. (Multiple sources: fn-feature-squeeze-defense, fn-failsafe-ml, line 94)
  Interaction: Layered bar chart with toggles for each defense (Adversarial Training ON/OFF, Input Sanitization ON/OFF, Confidence Threshold slider, Monitoring Frequency). Left Y: accuracy under three conditions (clean, adversarial, OOD). Right Y: cumulative compute overhead. Dual failure states: (1) accuracy below 60% safety floor; (2) compute exceeds 10x budget. Students discover that guardrails achieve comparable protection at fraction of the cost.

- **Part E -- The Compression-Robustness Collision** (~8 min)
  Concept: The efficiency techniques that make deployment feasible (INT8 quantization, pruning) narrow the robustness margin, making models more susceptible to perturbations that full-precision models could absorb.
  Key equation/principle: INT8: 75% memory reduction, 2-4x speedup, 1-3% accuracy trade. But reduced numerical headroom means adversarial perturbations that a FP32 model absorbs can flip INT8 predictions. (@sec-robust-ai, line 303, line 625)
  Interaction: Toggle between FP32 (full precision) and INT8 (quantized) deployment. Under normal inputs, INT8 matches FP32 within 1-3%. Under adversarial perturbation (epsilon slider), INT8 accuracy collapses faster. Reveals the tension: "robustness engineering is a constant negotiation with efficiency and scalability constraints."

---

All four lab plans exist already at `/Users/VJ/GitHub/MLSysBook/labs/plans/vol2/lab_11_edge_intel.md` through `lab_14_robust_ai.md`. The existing plans follow a 2-Act structure (per the lab protocol). My proposed 5-part structure above maps naturally onto those 2 acts: Parts A-B typically align with Act 1 (Calibration), and Parts C-E align with Act 2 (Design Challenge), with the parts providing finer granularity for the 1-hour story arc.

The key design principles applied across all four labs:

1. **Every part has a prediction that students will get wrong** -- the aha moment is manufactured through calibrated misconceptions (memory amplification is ~2x not 9x; platform break-even is 20 models not 200; DP error is $2,000 not $200; robustness costs 26 pp not 3 pp).

2. **Parts build sequentially** -- each part's revelation motivates the next part's question. Memory amplification (Part A) motivates adaptation strategies (Part B); the complexity explosion (Part A) motivates platform ROI (Part B) and then silent failure detection (Part C).

3. **Quantitative anchors come directly from the chapter** -- every slider range, threshold, and formula traces to specific computed values in the chapter's LEGO cells or prose claims, as documented in the existing traceability tables.

4. **Failure states are reversible** -- students can always pull sliders back to recover from OOM, budget exhaustion, or SLA violation, reinforcing that the goal is finding the boundary, not punishment.

---

## Lab V2-15: The Carbon Budget
Chapter: `sustainable_ai.qmd`
1-hour story arc: Students discover that where and when you compute matters more than how efficiently you compute, and that efficiency gains can paradoxically increase total energy consumption.

### Proposed Parts:

- **Part A -- The Energy Wall** (~10 min)
  Concept: The exponential divergence between AI compute demand (350,000x growth, 2012-2019) and hardware efficiency gains (1.5x/year), establishing why sustainability is a physics problem, not an ethics problem.
  Key equation/principle: AI compute doubling time (~3.4 months) vs Moore's Law doubling time (~24 months); the Energy Wall gap = Compute_growth / Efficiency_growth ~ 195,000x over 12 years.
  Interaction: Students set a target model scale (e.g., 10x, 100x, 1000x over GPT-3) and a hardware generation timeline. A dual-curve log-scale plot shows demand vs efficiency, and a shaded "energy deficit" region grows. Students predict at what year the deficit exceeds a threshold (e.g., exceeds global datacenter capacity). The prediction question: "By what year will AI compute demand exceed projected global datacenter capacity at current efficiency trends?" Students discover it is much sooner than expected.

- **Part B -- The Geography of Carbon** (~12 min)
  Concept: Grid carbon intensity varies 40x across regions (Quebec hydro at 20 gCO2/kWh vs Poland coal at 800 gCO2/kWh), making site selection the single highest-leverage sustainability intervention.
  Key equation/principle: C_operational = E_total x CI_grid x PUE (@eq-operational-carbon). Carbon(Quebec) = 10,000,000 kWh x 20g = 200 tonnes; Carbon(Poland) = 10,000,000 kWh x 800g = 8,000 tonnes. Ratio = 40x.
  Interaction: Students predict the carbon ratio between hydro and coal regions for a 10,000 MWh training run (multiple choice: 2-3x / 5-10x / 40x / 100x). Most guess 5-10x, anchoring on algorithmic speedup scales. A bar chart reveals the 40x gap. Students then manipulate region selector, training energy slider, and PUE slider, watching carbon totals update. An annotation compares the geographic savings to the best-case compound savings from pruning + quantization + distillation (~160x compound), establishing that a single site selection decision rivals the entire algorithmic optimization toolkit.

- **Part C -- The Lifecycle Carbon Shift** (~12 min)
  Concept: As grids decarbonize, the dominant carbon term shifts from operational emissions to embodied carbon (hardware manufacturing), making hardware longevity and utilization the binding constraint in clean-grid regions.
  Key equation/principle: C_embodied_daily = C_manufacturing / (L_lifetime x 365) (@eq-embodied-daily). For H100: 150-200 kg CO2 embodied; at 700W on US avg grid, operational matches embodied in ~1-2 years. In clean-grid regions, embodied carbon can exceed 30-50% of total lifecycle emissions.
  Interaction: Students toggle between a coal-grid and hydro-grid deployment. In coal-grid mode, operational carbon dominates (the optimization lever is geographic scheduling). In hydro-grid mode, operational carbon nearly vanishes and the embodied carbon bar becomes dominant. Students adjust hardware refresh cycle (2-5 years) and utilization rate (30-90%) and watch the lifecycle balance shift. Prediction question: "In a 100% renewable datacenter, what fraction of total lifecycle carbon comes from hardware manufacturing?" (Most guess <10%; actual is 30-50%+). This reveals that "green energy" does not eliminate AI's carbon footprint.

- **Part D -- The Jevons Trap** (~14 min)
  Concept: Jevons Paradox applied to AI: making inference more efficient does not guarantee lower total energy consumption because reduced cost stimulates demand, and elastic demand can increase total consumption despite per-unit savings.
  Key equation/principle: E_total = (E_baseline / Efficiency) x V_baseline x (Efficiency)^Elasticity. For Efficiency = 2x, Elasticity = 2.0: E_total = 0.5 x 1 x 4 = 2.0 (100% increase). Jevons breakeven: Elasticity < 1 for efficiency to reduce total consumption.
  Interaction: Students enter a numeric prediction for the net change in total energy when efficiency doubles and demand is elastic (300% volume increase). Most predict -25% to -50% reduction; actual is +100% increase. The instrument shows three demand curves (inelastic, unit-elastic, elastic) on a Jevons dashboard. Students manipulate efficiency factor and demand elasticity, watching total energy. A carbon cap toggle demonstrates that only absolute limits guarantee net reduction. The failure state triggers when total carbon exceeds 2x baseline: the chart turns red with "JEVONS REBOUND" banner. The key discovery: governance (hard caps on total compute) is the only mechanism that guarantees net reduction when demand is elastic.

- **Part E -- Carbon-Aware Fleet Design** (~12 min)
  Concept: Synthesizing geographic optimization, temporal scheduling (align with renewable availability), and carbon caps into a coherent carbon-aware scheduling strategy that achieves a 50% emission reduction target.
  Key equation/principle: Carbon-aware scheduling reduces emissions by 50-80% by aligning workloads with renewable availability. Combined with geographic optimization (40x) and carbon caps, engineers can meet absolute reduction targets.
  Interaction: Students face a design challenge: achieve 50% carbon emission reduction for a fleet with elastic demand without exceeding a 48-hour project delay. They combine levers: geographic shift (region selection), temporal scheduling (shift to low-CI hours), efficiency optimization, and carbon caps. The instrument is a 24-hour carbon intensity time series with movable job placement. Students discover that efficiency alone fails (Jevons), geographic shift alone may add latency, and only the combination of geographic shift + carbon cap reliably hits the target. A Design Ledger records the student's carbon reduction strategy, feeding forward to Lab V2-17.

---

## Lab V2-16: The Fairness Budget
Chapter: `responsible_ai.qmd`
1-hour story arc: Students discover that fairness is not a single metric to optimize but a set of mathematically incompatible constraints requiring explicit policy choices, and that responsible AI infrastructure has real system costs that compound at fleet scale.

### Proposed Parts:

- **Part A -- The Impossibility Wall** (~12 min)
  Concept: The Fairness Impossibility Theorem (Kleinberg 2016, Chouldechova 2017) proves that Demographic Parity, Equalized Odds, and Calibration cannot be simultaneously satisfied when base rates differ between groups.
  Key equation/principle: P(Y_hat=1 | S=a) = P(Y_hat=1 | S=b) (Demographic Parity) vs P(Y_hat=1 | S=a, Y=y) = P(Y_hat=1 | S=b, Y=y) (Equalized Odds) -- mutually exclusive when P(Y=1|S=a) != P(Y=1|S=b).
  Interaction: Students predict whether a single threshold can satisfy both DP and Equalized Odds simultaneously (multiple choice: Yes/Yes with separate thresholds/No because impossible/No because bad model). Most choose "yes." A threshold sweep instrument plots all three fairness metrics vs classification threshold, showing that at every threshold, at least one metric is substantially violated. Students manipulate base rates for both groups; when base rates are equalized, the impossibility vanishes, confirming it is the base rate gap that drives the conflict. Side-by-side confusion matrices update in real time.

- **Part B -- The Fairness Tax** (~10 min)
  Concept: Enforcing any fairness constraint reduces overall accuracy by a quantifiable amount (the "fairness tax"), and this tax scales with the base rate divergence between groups.
  Key equation/principle: Fairness_Tax(DP) ~ |P(Y=1|S=a) - P(Y=1|S=b)| x k, where k ~ 10 percentage points per unit gap. Chapter example: 85% accuracy drops to 81% under demographic parity (4% tax).
  Interaction: Students enter a numeric prediction for the accuracy after enforcing demographic parity on a model with 85% baseline accuracy and groups with 60% vs 30% base rates. Most predict 82-84% (expecting small tax). The instrument reveals ~75-78% for heterogeneous populations. Students adjust the base rate gap slider and watch the fairness tax scale: small gap = small tax, large gap = large tax. This establishes that fairness is not a fixed-cost add-on but a variable cost proportional to demographic heterogeneity.

- **Part C -- The Feedback Loop** (~12 min)
  Concept: Deployed ML systems create feedback loops that amplify bias over time. Models trained on biased data produce biased decisions, which generate biased retraining data, compounding the initial skew across iterations.
  Key equation/principle: The Sociotechnical Feedback Invariant: future data P_{t+1}(X) is a function of the model's past decisions f_t(X). Each retraining cycle amplifies the initial bias unless intervention points break the loop.
  Interaction: Students run a simplified predictive policing simulation across 5-10 time steps. The model allocates patrols based on predicted crime rates; more patrols produce more recorded incidents; recorded incidents become retraining data. Students start with a 5% bias in initial data and predict what the bias will be after 10 iterations (most predict modest growth, perhaps 10-15%). The simulation shows exponential amplification to 40-60%+ disparity. Students then toggle on four intervention points (data auditing, fairness constraints, output monitoring, feedback governance) and observe which combinations break the loop. The key discovery: post-hoc audits alone are insufficient; breaking the loop requires intervention at multiple stages simultaneously.

- **Part D -- The Responsible AI Overhead Budget** (~14 min)
  Concept: Responsible AI techniques (fairness monitoring, explainability, differential privacy) impose quantifiable computational overhead that must be budgeted as first-class system costs alongside inference latency and memory.
  Key equation/principle: Latency_total = Inference_ms + Monitoring_ms (10-20 ms) + Explainability_ms (0-50 ms for SHAP). At fleet scale: 10B inferences/day x 10 ms overhead = 100M GPU-seconds/day of responsible AI compute. The overhead table from @tbl-responsible-ai-overhead quantifies: DP-SGD adds 15-30% training overhead; SHAP adds 50-200% inference cost.
  Interaction: Students design a fleet serving two regions (homogeneous and heterogeneous demographics) under a 100 ms latency SLA. They select fairness metric (DP / Equalized Odds / Equal Opportunity), monitoring level (none / basic / full), and explainability level (none / LIME / SHAP). A latency waterfall chart shows: inference (30 ms) + monitoring + explainability = total. The failure state triggers when total > 100 ms SLA ("SLA VIOLATED" banner). Students discover that Full SHAP + Full monitoring + inference = 100 ms, exactly at the boundary. A radar plot compares accuracy, fairness disparity, latency, and memory across both regions. The design challenge: keep accuracy > 80% in both regions, fairness disparity < 0.05, and total latency < 100 ms. The solution requires choosing Equal Opportunity (less restrictive than DP) + Basic monitoring + On-demand LIME. The Design Ledger records the fairness metric chosen and overhead budget.

- **Part E -- The Automation Bias Paradox** (~12 min)
  Concept: As AI system accuracy increases, human oversight effectiveness paradoxically decreases because operators calibrate their trust to the "near-perfect" machine and stop catching the rare errors.
  Key equation/principle: At 90% AI accuracy, human override rate ~ 15%. At 99% accuracy, override rate drops to ~2%. Combined system sensitivity: S_combined = S_AI + (1 - S_AI) x (1 - alpha) x S_human, where alpha = probability of blindly accepting AI errors (60-80%).
  Interaction: Students set AI accuracy and human override willingness on sliders and predict the combined human-AI system error rate. Most expect the combined system to outperform both human and AI alone. The instrument reveals that at high AI accuracy (99%), the combined system can perform WORSE than the AI alone on rare critical cases because the human correction channel has atrophied. Students adjust "interface friction" (mandatory justification requirements) and watch how deliberate slowdown of acceptance restores the human oversight channel. A time-series shows that as AI accuracy ramps up over deployment months, human vigilance drops, and the system becomes more vulnerable to rare catastrophic failures. The takeaway: the safer a system appears, the more dangerous its rare failures become.

---

## Lab V2-17: The Fleet Synthesis (Capstone)
Chapter: `conclusion.qmd`
1-hour story arc: Students integrate all six principles of distributed ML systems engineering and discover that the binding constraint shifts with scale, that the principles interact as a coupled system, and that the next 100x efficiency must come primarily from orchestration rather than hardware or algorithms.

### Proposed Parts:

- **Part A -- The Sensitivity Wall** (~12 min)
  Concept: At fleet scale, communication (not computation) is the most sensitive system dimension. A 10% degradation in network bandwidth causes a larger throughput drop than 10% fewer FLOPS, because synchronization barriers amplify communication bottlenecks nonlinearly.
  Key equation/principle: T_step = max(O / (R_peak x eta), 2(n-1)/n x G/BW) + T_checkpoint. At large n, the communication term dominates. Straggler effect: one worker at 80% speed reduces cluster throughput by 20%.
  Interaction: Students predict which 10% improvement yields the largest throughput gain for a 1,000-GPU cluster (multiple choice: FLOPS / bandwidth / fault tolerance / scheduling). Most choose FLOPS. A Tornado sensitivity chart shows that communication is 2-3x more sensitive than compute at 1,000 GPUs. Students toggle fleet size (8 / 64 / 1,000 / 10,000) and watch the sensitivity ordering flip: at 8 GPUs compute dominates, at 1,000+ communication dominates. This directly demonstrates Principle 6: Scale Creates Qualitative Change.

- **Part B -- The Failure Budget** (~10 min)
  Concept: At fleet scale, component failure is routine, not exceptional. Meta's Llama 3 training on 16,384 GPUs experienced 419 failures in 54 days (one every 3 hours). Checkpointing overhead and recovery strategy dominate system design.
  Key equation/principle: MTBF_cluster = MTBF_component / N. For 10,000 GPUs with individual MTBF of 10,000 hours: MTBF_cluster = 1 hour. Goodput = 1 - (T_checkpoint / MTBF) per Young-Daly model.
  Interaction: Students predict the cluster MTBF for 10,000 GPUs (numeric entry, hours). Most dramatically overestimate (predicting days or weeks; actual is ~1 hour). A calculator shows how individual reliability composes: even 99.99% per-GPU uptime yields only 37% probability that all 10,000 GPUs are simultaneously up. Students adjust checkpoint frequency and fleet size, watching goodput (useful work / total work) change. They discover the optimal checkpoint interval that maximizes goodput, balancing checkpoint overhead against wasted computation from failures.

- **Part C -- The Principle Interaction Map** (~12 min)
  Concept: The six principles of distributed ML interact as a coupled system. Optimizing one can degrade another: communication optimization may increase failure exposure, sustainability constraints limit infrastructure choices, responsible AI overhead eats into latency budgets.
  Key equation/principle: G_effective = G_total x eta_comm x eta_fault x (1 - delta_fairness) x (1 - delta_carbon). Each principle acts as a multiplicative efficiency factor on the total system gain.
  Interaction: Students use a hexagonal radar plot showing all six principles simultaneously. They push one principle to its maximum (e.g., maximize communication efficiency) and observe which other principles degrade as a result. For example, pushing communication efficiency to 99% requires aggressive gradient compression, which increases failure exposure (fault tolerance degrades) and adds compute overhead (sustainability degrades). Students discover that no configuration achieves maximum on all six axes simultaneously -- the art is finding the best compromise. A coupling matrix shows which principles are positively and negatively correlated.

- **Part D -- The 100x Challenge** (~14 min)
  Concept: The next 100x efficiency improvement decomposes as Hardware (4x) x Algorithm (2.5x) x Orchestration (10x). Hardware and algorithms are hitting diminishing returns; the majority of future scaling must come from system orchestration (compound AI systems, reasoning chains, tool use).
  Key equation/principle: Total_Gain = G_HW x G_Algo x G_Orch. 100x = 4x x 2.5x x G_Orch => G_Orch = 10x. The Compound Capability Law: Capability proportional to Model_IQ x (Tools + Context + Planning)^N.
  Interaction: Students predict how much orchestration must contribute to reach the 100x target (numeric entry, 1-50x). Most guess 2-5x, expecting hardware and algorithms to carry more weight. The instrument shows a multiplicative budget bar: three sliders (hardware 1-8x, algorithm 1-5x, orchestration 1-20x) whose product must reach 100x. A target line at 100x turns green when achieved. Students discover that even generous hardware (8x) and algorithm (5x) assumptions only yield 40x, still requiring 2.5x from orchestration. With realistic estimates (4x HW, 2.5x algo), orchestration must deliver 10x -- the dominant factor. This establishes that the future belongs to compound AI systems, not bigger models.

- **Part E -- Fleet Architecture Blueprint** (~12 min)
  Concept: Final synthesis integrating all six principles into a coherent fleet design, incorporating sustainability constraints from Lab V2-15 and fairness overhead from Lab V2-16. This is the terminal capstone for the entire two-volume series.
  Key equation/principle: All six principles evaluated simultaneously: communication (bandwidth utilization), fault tolerance (goodput), infrastructure (FLOPS multiplier), responsible engineering (fairness disparity), sustainability (carbon reduction), orchestration (compound system gain).
  Interaction: Students face the full design challenge: reach >= 100x effective system gain with all six radar axes within target zones. Design Ledger values from Labs V2-15 (carbon cap, carbon reduction strategy) and V2-16 (fairness metric, latency overhead) feed in as starting constraints. Students discover that Lab V2-15's carbon cap reduces available compute, requiring more orchestration to compensate. Lab V2-16's fairness monitoring overhead eats into the latency budget, constraining communication patterns. The radar plot shows the student's configuration polygon against the target polygon; axes turn green when satisfied, red when below threshold. The failure state triggers when effective gain < 50x with any axis in the red zone. The culminating insight: distributed ML engineering is the art of balancing all six principles within acceptable trade-offs, not maximizing any single dimension.

---

---



---

# ADDENDUM: Expanded Parts and Synthesis Sections

Generated from gap-analysis agents reading chapters and identifying missing parts.

## Source: Lab 11 (new D+E)

## Inventory of Key Chapter Concepts

| Concept | Covered in Parts A-C? |
|---------|----------------------|
| Roofline Model / Arithmetic Intensity / Ridge Point | **Part A** |
| Kernel Fusion / Elementwise Trap / Memory-bound ops | **Part B** |
| Hardware Balance shift (same kernel, different regime on different hardware) | **Part C** |
| Numerical Precision and its effect on both sides of the roofline | No |
| Batch Size as the first-order knob for arithmetic intensity | Touched in Part B (batch slider), but not the core aha |
| Tiling and data reuse (SRAM vs DRAM) | No |
| Energy cost of data movement (Horowitz numbers, memory hierarchy energy) | No |
| Systolic arrays and data reuse efficiency | No |
| Heterogeneous SoC / workload partitioning across CPU/GPU/NPU | No |
| Multi-chip scaling / Amdahl's Law / communication overhead | No |
| Cost-performance economics / accelerator selection | No |
| Hardware-software co-design (compiler, runtime) | No |
| Sustainability / carbon ROI of specialization | No |

## Explorable Concept Test

I applied the five-question filter to the strongest candidates not already in Parts A-C.

**Candidate 1: Precision's Double Dividend (Numerics)**
- Tunable parameter? YES (precision selector: FP32/FP16/BF16/INT8)
- Students predict wrong? YES (they expect 2x speedup from halving precision; actual effect is 2x on compute ceiling AND 2x on bandwidth simultaneously, shifting both roofline ceilings)
- Why instructive? YES (teaches that precision attacks both sides of the roofline, not just one)
- Simulate in 5 min? YES (shifts roofline ceilings and operation point)
- Connects to ecosystem? YES (links to model compression chapter, TinyTorch quantization module)
- Score: **5/5** -- but this is partially entangled with Part A's precision dropdown. REJECT to avoid duplication.

**Candidate 2: Batch Size as a Regime Switch**
- Tunable parameter? YES (batch size slider)
- Students predict wrong? YES (they expect linear throughput improvement; actual behavior is a regime transition with a plateau)
- Why instructive? YES (teaches that batch size changes arithmetic intensity, not just parallelism)
- Simulate in 5 min? YES (batch size vs throughput curve with regime annotation)
- Connects to ecosystem? YES (directly feeds into model serving chapter, TinyTorch inference module)
- Score: **5/5** -- but Part B already has a batch size slider affecting GEMM AI. The aha moment overlaps significantly with Part B's exploration. REJECT to avoid redundancy.

**Candidate 3: The Energy Cost of Data Movement (Horowitz Numbers)**
- Tunable parameter? YES (memory tier selector: register/SRAM/DRAM; operation type)
- Students predict wrong? YES (they expect compute to dominate energy; reality is DRAM access costs 200x more than a MAC)
- Why instructive? YES (explains WHY the memory wall exists physically, not just that it exists)
- Simulate in 5 min? YES (energy breakdown bar chart as operation moves through memory tiers)
- Connects to ecosystem? YES (sustainability section, hardware selection decisions)
- Score: **5/5**

**Candidate 4: Tiling and Data Reuse**
- Tunable parameter? YES (tile size slider)
- Students predict wrong? YES (they expect tile size to have marginal effect; actual effect is orders of magnitude in DRAM traffic)
- Why instructive? YES (teaches the fundamental mechanism that makes accelerators work)
- Simulate in 5 min? YES (DRAM accesses vs tile size, with naive baseline comparison)
- Connects to ecosystem? YES (tiling principle, systolic array section, TinyTorch GEMM module)
- Score: **5/5**

**Candidate 5: Heterogeneous SoC Workload Partitioning**
- Tunable parameter? YES (operation-to-processor assignment: CPU/GPU/NPU)
- Students predict wrong? YES (they expect NPU-for-everything to win; some ops are faster on CPU)
- Why instructive? YES (teaches that specialization has limits; irregular ops break NPU dataflow)
- Simulate in 5 min? Borderline -- requires modeling three processors and power/latency trade-offs. Could work as a simplified decision table.
- Connects to ecosystem? YES (deployment spectrum, SoC section, edge kits)
- Score: **4/5** -- viable but less natural for a roofline-themed lab progression.

**Candidate 6: Multi-chip Scaling / Amdahl's Law**
- Tunable parameter? YES (number of GPUs, serial fraction)
- Students predict wrong? YES (they expect linear scaling; reality is Amdahl ceiling)
- Why instructive? YES (teaches why 8 GPUs are not 8x faster)
- Simulate in 5 min? YES
- Connects to ecosystem? YES (distributed training chapter, but that is Vol 2 territory)
- Score: **4/5** -- strong concept but the chapter treats multi-chip scaling as a brief overview, explicitly deferring to Vol 2. Not enough chapter depth to ground this lab.

## Proposed Additional Parts

Based on the analysis, I propose **Part D** and **Part E**, plus a revised Synthesis. The candidates that best extend the A-B-C story arc are:

- **Part D: The Energy Cost of Movement** (Horowitz Numbers) -- grounded in the energy hierarchy figure, the Von Neumann bottleneck discussion, and the systolic array energy advantage calculation
- **Part E: The Tiling Dividend** -- grounded in the tiling principle section, the naive-vs-tiled matmul, and the SRAM reuse factor calculation

These two parts complete a coherent five-part narrative:

```
A: Diagnose the regime (roofline)
B: Different ops live in different regimes (kernel mix)
C: The regime depends on the hardware, not just the op
D: WHY memory access is the bottleneck (energy physics)
E: HOW to fight back (tiling as the fundamental mechanism)
Synthesis: Budget an accelerator purchase using all five insights
```

---

## Part D -- The Energy Cost of Movement

**NAME:** The Energy Cost of Movement

**ONE-LINE DESCRIPTION:** Students discover that fetching one value from DRAM costs 200x more energy than computing with it, explaining WHY the memory wall exists and WHY systolic arrays are designed around data reuse.

**KEY CONCEPT/EQUATION:** The Horowitz energy hierarchy: E_DRAM ~640 pJ per access vs E_MAC ~1 pJ per operation. Systolic energy advantage: E_ratio = E_vector / E_systolic (from the SystolicEnergy LEGO class at line 1661). The chapter grounds this in `@sec-hardware-acceleration-understanding-ai-memory-wall-3ea9` and `@fig-energy-hierarchy`.

**INTERACTION:**

| Control | Type | Range | Default | Effect |
|---------|------|-------|---------|--------|
| Memory tier | Dropdown | Register / SRAM (L1) / SRAM (L2) / HBM / DRAM | DRAM | Changes the energy cost per access; shows the hierarchy gap |
| Array dimension | Slider | 4--128 | 16 | Changes systolic reuse factor; shows how a larger array amortizes DRAM access across more MACs |
| Operation count | Slider | 100--100,000 | 1,000 | Scales the total energy; shows that at scale, data movement dominates |

**Charts:**
1. **Energy Breakdown Bar** (stacked horizontal): For a given operation count, shows energy spent on compute vs. energy spent on data movement. At default settings, the bar is overwhelmingly dominated by data movement.
2. **Systolic vs. Vector Energy Ratio** (line chart): X-axis is array dimension, Y-axis is energy ratio. Students see the ratio climb from ~5x at dim=4 to ~200x at dim=128, explaining why TPUs invest in large systolic arrays.

**AHA MOMENT:**
- Students expect: "Compute is the expensive part; that is why we buy faster chips."
- They discover: "Moving data is 200x more expensive than computing with it; the chip spends most of its energy budget on memory access, not arithmetic."
- Because: "DRAM access requires charging long metal wires across the chip package, consuming ~640 pJ per access, while a MAC operation uses ~1 pJ in local silicon. The speed of light and the energy cost of moving electrons across distance are the root cause."

**PREDICTION:**
- Question: "For 10,000 MAC operations, each requiring one DRAM load, what fraction of total energy is spent on data movement vs. computation?"
- Options: (A) ~50/50 -- compute and memory are balanced, (B) ~80% compute / 20% memory -- compute dominates, (C) ~99% memory / 1% compute -- data movement dominates [CORRECT], (D) ~70% memory / 30% compute -- memory is significant but not dominant
- Common wrong answer: A -- students assume compute and memory are roughly balanced
- Why wrong: At 640 pJ per DRAM access vs. 1 pJ per MAC, the ratio is 640:1. For 10,000 operations: 6,400,000 pJ memory vs. 10,000 pJ compute = 99.8% memory.

**CHAPTER GROUNDING:**
- `@sec-hardware-acceleration-understanding-ai-memory-wall-3ea9` (memory wall definition and energy hierarchy)
- `@fig-energy-hierarchy` (Horowitz numbers bar chart, line 2259)
- SystolicEnergy LEGO class (line 1661): E_vector = 2,561 pJ/OP, E_systolic = 4.0 pJ/OP, ratio = 641x
- AcceleratorEfficiencyAnchor class (line 1649): 128x128 systolic array achieves 16,384 MACs/cycle at 200x energy efficiency

---

## Part E -- The Tiling Dividend

**NAME:** The Tiling Dividend

**ONE-LINE DESCRIPTION:** Students discover that partitioning a matrix multiply into tiles that fit in SRAM reduces DRAM traffic by orders of magnitude, transforming a memory-bound operation into a compute-bound one without changing the algorithm.

**KEY CONCEPT/EQUATION:** For an NxN matrix multiply with tile size T, naive DRAM accesses = O(N^3) reads, while tiled DRAM accesses = O(N^3 / T) reads. Reuse factor = T (each loaded element is used T times before eviction). From the tiling principle section (`@sec-hardware-acceleration-tiling-principle`, line 1879): a 4096-wide layer on a 128-wide systolic array requires 1,024 tiles, with reuse factor = 128. From `@sec-hardware-acceleration-memoryefficient-tiling-strategies-9fce` (line 4303): the naive-vs-tiled comparison.

**INTERACTION:**

| Control | Type | Range | Default | Effect |
|---------|------|-------|---------|--------|
| Matrix dimension N | Slider | 256--4096 | 1024 | Sets the problem size; larger N means more total work |
| Tile size T | Slider | 1--256 (powers of 2) | 1 (naive) | Controls SRAM working set; T=1 is naive, T=128 is optimal for a 128-wide array |
| SRAM capacity | Display (read-only) | -- | 256 KB | Shows whether the tile fits in SRAM; tiles exceeding capacity turn red |

**Charts:**
1. **DRAM Traffic Reduction** (log-scale bar chart): X-axis is tile size (1, 2, 4, 8, ..., 256), Y-axis is total DRAM accesses. At T=1 (naive), traffic is O(N^3). At T=128, traffic drops by ~128x. The bar chart makes the orders-of-magnitude reduction visceral.
2. **Effective Arithmetic Intensity** (overlaid on mini-roofline): Shows how tiling shifts the operation point rightward on the roofline. At T=1, AI is low (memory-bound). At T=128, AI crosses the ridge point (compute-bound). This connects Part E directly back to Part A's roofline.

**AHA MOMENT:**
- Students expect: "Tiling is a minor software optimization -- maybe 10-20% improvement."
- They discover: "Tiling reduces DRAM traffic by 128x, transforming the same GEMM from memory-bound to compute-bound."
- Because: "Without tiling, every multiply-accumulate requires a fresh DRAM load. With tiling, data is loaded once into SRAM and reused T times. Since SRAM access costs ~5 pJ vs. DRAM's ~640 pJ, the energy and bandwidth savings are multiplicative."

**PREDICTION:**
- Question: "A 1024x1024 matrix multiply runs at 15% MFU with naive (untiled) execution. You apply tiling with T=128. What happens to MFU?"
- Options: (A) ~20% -- tiling helps a little, (B) ~45% -- moderate improvement, (C) ~70-80% -- the operation becomes compute-bound [CORRECT], (D) 100% -- tiling eliminates all overhead
- Common wrong answer: A -- students underestimate the impact of data reuse
- Why wrong: Tiling increases effective arithmetic intensity by the reuse factor (~128x for T=128). For a 1024x1024 GEMM at FP16 with naive execution, AI ~170 FLOP/byte (below H100 ridge of 295). With T=128 tiling and SRAM reuse, effective AI climbs well above the ridge, making the operation compute-bound with MFU approaching 70-80%.

**CHAPTER GROUNDING:**
- `@sec-hardware-acceleration-tiling-principle` (line 1879): TilingPrinciple class computes 1,024 tiles and reuse factor of 128
- `@sec-hardware-acceleration-memoryefficient-tiling-strategies-9fce` (line 4303): naive vs. tiled matmul comparison
- `@lst-naive_matmul` (line 4314) and `@lst-tiled_matmul` (line 4411): code examples
- `[^fn-tiling-cache-reuse]` (line 4310): "reduction in memory traffic is the primary source of the 10-50x speedup"
- `@tbl-tiling-strategies` (line 4493): spatial vs. temporal vs. hybrid tiling

---

## Revised Synthesis

### Decision Log Prompt

"You have now traced the full chain: (A) the roofline model diagnoses whether an operation is compute-bound or memory-bound; (B) different kernels in the same model live in different regimes; (C) the same kernel switches regimes on different hardware; (D) the root cause is that moving data costs 200x more energy than computing it; (E) tiling is the fundamental mechanism that fights back by reusing data in SRAM. In one paragraph, explain: given a budget to either (i) buy a next-generation GPU with 2x more TFLOPS or (ii) invest in kernel optimization (fusion + tiling) for your current GPU, which investment yields higher throughput for a transformer inference workload, and why."

### Key Takeaways

1. **Part A -- Diagnose the regime before optimizing the kernel.** A GEMM at N=512 achieves only 170 FLOP/byte, below the H100 ridge point of 295. MFU is 31.5% not because the code is broken, but because the kernel is memory-bandwidth-bound. Increasing matrix dimension to N=1024 crosses the ridge and MFU jumps past 50%.

2. **Part B -- Elementwise ops are always memory-bound; fuse them.** LayerNorm and Softmax have AI ~0.83 FLOP/byte, placing them ~300x below the H100 ridge point. Kernel fusion is the only lever: eliminating HBM round-trips raises effective AI by 2-3x.

3. **Part C -- More powerful accelerators are harder to saturate.** The H100 ridge point (295 FLOP/byte) is 2x the edge device's (~118 FLOP/byte). An operation that is compute-bound on edge hardware becomes memory-bound on the cloud accelerator without changing a single line of code.

4. **Part D -- Data movement costs 200x more energy than computation.** A DRAM access consumes ~640 pJ while a MAC costs ~1 pJ. Systolic arrays exploit this by amortizing loads across 128x128 = 16,384 operations, achieving 200x better energy efficiency than naive vector execution.

5. **Part E -- Tiling is the fundamental mechanism that fights the memory wall.** Partitioning a matrix multiply into tiles that fit in SRAM reduces DRAM traffic by the tile reuse factor (~128x). This transforms a memory-bound operation into a compute-bound one, connecting the physical insight from Part D to the diagnostic framework from Part A.

### What's Next

Lab 12 (Benchmarking) applies these insights to end-to-end system measurement: how do individual kernel bottlenecks compose into overall training and inference throughput, and how does MLPerf measure what matters?

---

## Concepts Rejected and Rationale

| Concept | Why Rejected |
|---------|-------------|
| Precision's Double Dividend | Partially covered by Part A's precision dropdown; adding a full Part would overlap |
| Batch Size as Regime Switch | Already a slider in Part B; the aha moment is subsumed by Part B's exploration |
| Heterogeneous SoC Partitioning | Viable (4/5 score) but breaks the lab's coherent roofline-centric narrative arc; better suited for a separate edge-deployment lab |
| Multi-chip Scaling / Amdahl's Law | Chapter treats this as a brief overview deferring to Vol 2; insufficient textbook depth to ground a full Part |
| Cost-Performance Economics | No tunable parameter with counterintuitive behavior; more suited to a textbook exercise than an interactive lab |
| Sustainability / Carbon ROI | Interesting but derivative of the energy insight in Part D; adding it would be redundant |

---

## Summary of File Paths

- Chapter source: `/Users/VJ/GitHub/MLSysBook/book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd`
- Existing lab plan: `/Users/VJ/GitHub/MLSysBook/labs/plans/LAB_11_HW_ACCELERATION_PLAN.md`

---

## Source: Labs 13-16 (new D+E each)

## Lab 13: The Tail Latency Trap (Model Serving)

**Existing parts cover**: M/M/1 queuing (Part A), batching latency-throughput Pareto (Part B), LLM KV cache memory wall (Part C).

**What the chapter covers that the lab does not**: training-serving skew / preprocessing divergence, cold start dynamics, precision-throughput tradeoffs, serving economics (cost per inference across hardware tiers), multi-model serving / GPU partitioning, and the full inference pipeline latency budget (where preprocessing/postprocessing dominate, not the model).

### Proposed Part D -- The Cold Start Tax (~10 min)
**Insert as new Part D, shift existing Synthesis to after Part E.**

**Concept**: Cold start latency decomposes into weight loading, graph compilation, CUDA context, and warmup. Students will discover that compilation (15-30 seconds) dominates by 50-100x over weight loading, and that pre-compilation eliminates the dominant term entirely (35s down to 1.5s). This teaches that initialization is a systems engineering problem, not a bandwidth problem.

**Key equation/principle**: T_cold = T_load + T_cuda + T_compile + T_warmup. The chapter shows ResNet-50 cold start ranges from 1.5s (optimized local) to 35s (cloud first deploy), with TensorRT compilation consuming 30 of those 35 seconds.

**Interaction**: Students see a stacked bar timeline of cold start phases. They toggle between loading strategies (full load, memory-mapped, lazy init) and deployment contexts (local SSD + pre-compiled, cloud S3 + first compile, serverless cold). A slider controls model size (from ResNet-50 at 98 MB to a 10 GB model) and students predict how cold start scales. The surprise: for small models, compilation dominates (not bandwidth); for large models, PCIe transfer dominates. Students must find which model size flips the bottleneck from compute-bound (compilation) to bandwidth-bound (transfer).

**Placement**: Insert as new Part D after KV cache (Part C). This transitions naturally from "how memory constrains steady-state serving" to "how initialization constrains scaling events."

### Proposed Part E -- The Precision Speedup Mirage (~8 min)
**Insert as new Part E, before Synthesis.**

**Concept**: Reducing numerical precision from FP32 to INT8 theoretically yields 4x speedup (32/8 bits), but achieved speedup is only 2.5-3.5x due to Tensor Core alignment requirements and pipeline inefficiencies. Students must also weigh the accuracy cost (0.33 pp loss for PTQ, nearly zero for QAT).

**Key equation/principle**: Throughput_INT8 / Throughput_FP32 = 32/8 = 4x (theoretical max) from @eq-precision-throughput. Achieved: 2.5-3.5x. The chapter's ResNet-50 precision tradeoff table shows FP32 at 2.8 ms, FP16 at 1.4 ms (2x, a "free lunch"), INT8 at 0.9 ms (3.1x, not 4x).

**Interaction**: Students set a target throughput multiplier ("I need 3x throughput from the same hardware") and a maximum acceptable accuracy loss. The simulation shows them which precision level achieves the target, whether calibration data is needed, and what the gap is between theoretical and achieved speedup. A second view shows the fleet-level economics: the chapter's example where switching 30 V100 GPUs from FP32 to INT8 provides 3x throughput, potentially saving 20 GPUs worth of cost. Students discover that FP16 is a genuine free lunch (2x, zero accuracy loss, no calibration), while INT8 requires engineering investment (calibration data, potential QAT retraining) for diminishing additional gains.

**Placement**: Insert as Part E. This caps the serving lab with an optimization that bridges back to model compression concepts.

---

## Lab 14: The Silent Degradation Problem (ML Ops)

**Existing parts cover**: PSI drift detection (Part A), optimal retraining cadence T* (Part B), deployment cost asymmetry (Part C).

**What the chapter covers that the lab does not**: technical debt compound cost, correction cascades, feedback loops, ML Test Score maturity rubric, CI/CD validation gates, training-serving skew mechanics, and MLOps maturity levels.

### Proposed Part D -- The Technical Debt Snowball (~10 min)
**Insert as new Part D, shift Synthesis to after Part E.**

**Concept**: Manual ML operations accumulate compound interest. The chapter shows that 4 hours/week of manual work breaks even against an 80-hour pipeline investment at week 20, but the real trap is that manual complexity *grows* as features are added. Students will discover that the crossover is not just about hours: it is about a capacity ceiling where the team cannot deploy new models because maintenance consumes 100% of engineering time.

**Key equation/principle**: The chapter's automation ROI: breakeven at T_break = C_pipeline / C_manual_weekly = 80/4 = 20 weeks. But the true model is C_manual(t) = C_base * (1 + growth_rate)^features, where feature count doubles manual time. After 1 year with growing feature count, the manual team spends 100% of time on maintenance vs. 0% for the automated team.

**Interaction**: Students control three sliders: (1) manual hours per week (2-8 hrs), (2) one-time pipeline build cost (40-200 hrs), (3) feature growth rate (0-20% per quarter). Two line charts show cumulative engineering hours over 2 years for manual vs. automated paths. Students must find the crossover point and observe how feature growth accelerates it. A second view shows the "capacity ceiling": at what point does manual maintenance consume >80% of the team's time, preventing any new model development? The surprise: even with generous assumptions (low manual cost, high pipeline cost), the crossover always arrives within the first year when features grow. The compounding effect is what makes the debt "silent" -- it feels manageable at week 10 but becomes crippling by week 40.

**Placement**: Insert as Part D. This follows retraining cadence (Part B) naturally: once students know *when* to retrain, they discover *why* manual retraining becomes infeasible at scale.

### Proposed Part E -- The Correction Cascade (~8 min)
**Insert as new Part E, before Synthesis.**

**Concept**: In ML systems, fixing one component introduces problems elsewhere because changes propagate through statistical dependencies, not explicit code paths. The CACHE principle (Change Anything Changes Everything) means that a "local" fix to a feature encoding can cascade through training, quantization, and serving accuracy. Students will discover that sequential model dependencies create amplifying chains where each fix triggers 1-3 additional fixes.

**Key equation/principle**: The chapter presents correction cascades as a debt taxonomy pattern. The cost model is: if fixing component i triggers k_i downstream fixes each costing C_fix, the total cascade cost is C_total = C_fix * sum(product(k_j for j in chain)). With k_avg = 2 and a chain of depth 3, a single fix triggers 2 + 4 + 8 = 14 additional fixes.

**Interaction**: Students see a dependency graph of an ML pipeline (data ingestion, feature engineering, model A, model B that depends on A's outputs, serving, monitoring). They "fix" a bug in one component by clicking on it, and watch the cascade propagate. Controls: (1) coupling strength (how many downstream components each fix triggers, 1-3), (2) pipeline depth (3-7 stages), (3) whether the system has isolation boundaries (modular interfaces that stop cascades). The surprise: without boundaries, a single fix in data ingestion at depth 1 can trigger 30+ downstream fixes. With proper modular interfaces at key boundaries, the same fix triggers only 2-3. Students discover that the architecture of the pipeline (not the quality of individual fixes) determines whether cascades are contained.

**Placement**: Insert as Part E. This rounds out the MLOps lab by showing that even with perfect drift detection (Part A) and optimal retraining (Part B), poor system architecture makes every correction worse.

---

## Lab 15: There Is No Free Fairness (Responsible Engineering)

**Existing parts cover**: Chouldechova impossibility (Part A), Pareto frontier / price of fairness (Part B), TCO breakdown (Part C).

**What the chapter covers that the lab does not**: explainability-latency tradeoff, carbon cost of compute, regulatory compliance requirements mapped to engineering choices, subgroup accuracy divergence under aggregate metrics, and threshold effects on subgroup outcomes.

### Proposed Part D -- The Explainability Tax (~10 min)
**Insert as new Part D, shift existing Synthesis to after Part E.**

**Concept**: Post-hoc explanation methods (LIME, SHAP) add 10-100x inference latency. Students must decide: for a credit decision system with a 100 ms latency SLO, which explainability method is feasible? The chapter states that SHAP adds 10-100x latency, making LIME the only viable option for real-time serving. But LIME produces inconsistent explanations across nearby inputs. Students discover that the explainability requirement is a hard engineering constraint that restricts both model selection and explanation method, not a bolt-on feature.

**Key equation/principle**: The chapter's interpretability spectrum: decision trees are directly auditable but limited in capacity; deep networks are powerful but require SHAP (10-100x latency) or LIME (faster but inconsistent). The latency budget framework from the serving chapter applies: T_total = T_preprocess + T_inference + T_explain, and T_explain may dominate.

**Interaction**: Students select an application domain (credit decisions, content moderation, recommendation, fraud detection) from the chapter's explainability requirements table. For each domain, they adjust: (1) model complexity (linear model, random forest, deep neural network), (2) explanation method (none, LIME, SHAP, inherent interpretability), (3) latency SLO. The simulation shows whether the configuration is feasible: latency budget, explanation fidelity, and regulatory compliance (does the domain require individual explanations?). Students discover that for high-stakes regulated domains (credit, healthcare), either the model must be inherently interpretable (constraining accuracy) or the latency budget must accommodate SHAP (constraining throughput). There is no configuration where a deep network provides real-time individual explanations within a tight latency SLO.

**Placement**: Insert as Part D after TCO (Part C). This transitions from "fairness has a cost" to "transparency has a cost too" -- completing the picture that responsible engineering imposes multiple simultaneous system constraints.

### Proposed Part E -- The Carbon Ledger (~8 min)
**Insert as new Part E, before Synthesis.**

**Concept**: Training vs. inference carbon footprint. The chapter demonstrates that inference costs dominate training costs by 10-1000x for production systems (the recommendation system example: $38K training vs. $1.5M inference over 3 years). Students discover the same asymmetry in carbon: a model's lifetime carbon footprint is dominated by inference, not training, which means per-query optimization is the highest-leverage sustainability intervention.

**Key equation/principle**: Carbon = Energy (kWh) * Carbon Intensity (kg/kWh), with 0.16 kg CO2eq per GPU-hour (chapter's baseline). The chapter's TCO calculation shows training at 2% vs. inference at 73% of 3-year TCO. The same ratio applies to carbon.

**Interaction**: Students configure: (1) training GPU-hours (100-10,000), (2) daily inference volume (1K-200M queries), (3) inference latency (1-50 ms), (4) deployment lifetime (1-5 years). Two pie charts update live: one for dollar cost, one for carbon cost. Students then apply a "green optimization" (quantization reducing inference time by 2x, or pruning reducing model size by 50%) and watch both pie charts shift. The surprise: a 2x inference speedup saves more carbon in one month of serving than the entire training run ever consumed. Students discover that "Green AI" is not primarily about reducing training costs (the headline-grabbing numbers); it is about per-query inference efficiency at scale.

**Placement**: Insert as Part E. This closes the responsible engineering lab with a quantitative environmental framing that connects back to the TCO analysis in Part C.

---

## Lab 16: The Architect's Audit -- Capstone

**Existing parts cover**: Iron Law for LLM inference / cost of a token (Part A), DAM radar chart / conservation of complexity (Part B), design ledger archaeology / personal blind spots (Part C).

**What the chapter covers that the lab does not**: the twelve quantitative invariants as an integrated framework, the Amdahl's Law pitfall (optimizing the wrong pipeline stage), constraint propagation across the full stack (architecture choice cascading to compression to hardware to serving), and the node-to-fleet transition.

### Proposed Part D -- The Amdahl Trap (~10 min)
**Insert as new Part D, shift Synthesis to after Part E.**

**Concept**: Optimizing the wrong pipeline stage yields negligible system-level speedup. The chapter's Fallacies section states: "Optimizing inference latency by 10x yields only 1.1x system speedup if data loading accounts for 90% of end-to-end latency." Students consistently over-optimize the component they understand best (usually the model) while ignoring the dominant bottleneck (usually data preprocessing or postprocessing).

**Key equation/principle**: Amdahl's Law: Speedup = 1 / ((1 - f) + f/S), where f is the fraction of time in the optimized component and S is the component-level speedup. From the serving chapter: the inference pipeline has preprocessing, inference, and postprocessing stages, and "any of these stages can become the latency bottleneck." The chapter explicitly warns that optimizing inference when preprocessing dominates is futile.

**Interaction**: Students see an end-to-end ML pipeline with 5 stages (data loading, preprocessing, inference, postprocessing, network). Each stage has a latency bar. Students choose which stage to optimize and by how much (2x, 5x, 10x speedup). The simulation applies Amdahl's Law and shows the system-level improvement. The surprise: optimizing the 5 ms inference stage by 10x when preprocessing takes 45 ms yields only 1.1x system improvement. The optimal strategy is to profile first, then optimize the dominant stage. A second mode lets students allocate a fixed "engineering budget" (100 person-hours) across stages, forcing them to prioritize. They discover that spreading effort across bottlenecks beats concentrating on any single stage.

**Placement**: Insert as Part D. This is the diagnostic skill capstone: before students can architect systems (Parts A-C), they must learn to identify where time actually goes.

### Proposed Part E -- The Constraint Cascade (~8 min)
**Insert as new Part E, before Synthesis.**

**Concept**: Architecture choices propagate through the entire stack. The chapter's opening example: choosing depthwise separable convolutions for mobile efficiency interacts with INT8 quantization and firmware-specific preprocessing, causing a 4 pp accuracy drop that no single team predicted. The twelve invariants are simultaneously active, and the Conservation of Complexity guarantees that simplifying one interface pushes complexity elsewhere.

**Interaction**: Students configure a deployment scenario by making sequential choices: (1) model architecture (MobileNetV2, EfficientNet-B0, ResNet-50 from the chapter's table), (2) compression technique (none, INT8 PTQ, INT8 QAT, pruning 50%), (3) deployment target (cloud GPU, mobile NPU, TinyML MCU), (4) serving requirements (latency SLO, batch size). After each choice, the simulation reveals which downstream options become infeasible. For example: choosing ResNet-50 eliminates mobile and TinyML targets; choosing INT8 PTQ introduces a 0.33 pp accuracy penalty that compounds with the architecture's baseline; choosing a 10 ms SLO eliminates all but the most aggressive optimizations. Students discover that the first choice (architecture) constrains everything downstream, and that undoing a bad early choice requires restarting from scratch -- the "correction cascade" from the MLOps chapter made tangible across the full stack. A "constraint ledger" tracks how many deployment options remain open after each choice, shrinking from many to few.

**Placement**: Insert as Part E. This is the true capstone moment: students experience the book's central thesis ("constraints drive architecture") by watching their choices cascade through the stack. It synthesizes concepts from serving, compression, hardware, and responsible engineering into a single interactive decision chain.

---

## Summary Table

| Lab | New Part | Name | Duration | Placement |
|-----|----------|------|----------|-----------|
| 13 | D | The Cold Start Tax | ~10 min | After Part C (KV cache), before Part E |
| 13 | E | The Precision Speedup Mirage | ~8 min | After Part D, before Synthesis |
| 14 | D | The Technical Debt Snowball | ~10 min | After Part C (deployment cost), before Part E |
| 14 | E | The Correction Cascade | ~8 min | After Part D, before Synthesis |
| 15 | D | The Explainability Tax | ~10 min | After Part C (TCO), before Part E |
| 15 | E | The Carbon Ledger | ~8 min | After Part D, before Synthesis |
| 16 | D | The Amdahl Trap | ~10 min | After Part C (design ledger), before Part E |
| 16 | E | The Constraint Cascade | ~8 min | After Part D, before Synthesis |

Every proposed part passes the Explorable Concept Test at 5/5: each has a tunable parameter, students will predict incorrectly, the explanation is instructive, the simulation runs in under 5 minutes, and the concept connects to the broader curriculum. Every quantitative anchor is grounded in equations or data from the chapter text itself.

---

## Source: Labs 01-04, 09-10, 12 (synthesis + Lab10 Part E)

## Lab-by-Lab Assessment

### Lab 01: The AI Triad (Introduction)

**Current 4 Parts:** A (DAM Triad), B (Iron Law Surprise), C (Silent Decay), D (Deployment Spectrum)

**Chapter coverage check:** The introduction chapter has these major explorable concepts:
- DAM Triad / Hardware Gap -- covered by Part A
- Iron Law of ML Systems -- covered by Part B
- Silent Degradation / Degradation Equation -- covered by Part C
- Deployment Spectrum (Cloud/Edge/Mobile/TinyML) -- covered by Part D
- Efficiency Framework (3 dimensions) -- taxonomy/historical, not explorable
- Return on Compute (RoC) -- interesting but derivative of the Iron Law
- Five-Pillar Framework -- taxonomy, no tunable parameter
- Bitter Lesson -- historical narrative, no parameters

**Verdict: 4 parts is sufficient -- add Synthesis only.**

The four parts cover the chapter's four major quantitative concepts. The Efficiency Framework and Five-Pillar Framework are organizational, not explorable. RoC is a restatement of the Iron Law's economic implications, which Part B already surfaces. Adding a 5th part would mean padding.

**Synthesis description:** Students receive a mission brief: deploy a keyword spotting model to a smart doorbell (TinyML target). They must apply all four parts in sequence: (1) check the DAM feasibility using the hardware gap from Part A, (2) diagnose whether the workload is compute-bound or memory-bound using the Iron Law from Part B, (3) set a monitoring threshold for drift using the Degradation Equation from Part C, and (4) justify their deployment paradigm choice using the spectrum from Part D. The synthesis question: "Given that the doorbell must last 1 year on battery with <100 ms wake-word latency, which Iron Law term is the binding constraint, and what does the Degradation Equation predict about accuracy after 12 months of seasonal drift?" This forces students to connect the physical constraints (Parts A/B) to the operational reality (Parts C/D) in a single design decision.

---

### Lab 02: The Physics of Deployment (ML Systems)

**Current 4 Parts:** A (Memory Wall Revelation), B (Light Barrier), C (Power Wall), D (Energy of Transmission)

**Chapter coverage check:**
- Iron Law decomposition / Memory Wall / Arithmetic Intensity -- covered by Part A
- Light Barrier / Propagation Delay -- covered by Part B
- Power Wall / Thermal constraints -- covered by Part C
- Energy of Transmission (local vs cloud cost) -- covered by Part D
- Workload Archetypes (Compute Beast, Bandwidth Hog, Sparse Scatter, Tiny Constraint) -- important but these are diagnostic categories applied through the Iron Law, which Part A already covers
- System Balance Across Paradigms -- the paradigm selection table is covered implicitly through Parts A-D
- Hybrid Architectures -- too complex for a 5-minute simulation; requires full system design

**Verdict: 4 parts is sufficient -- add Synthesis only.**

The four parts map directly to the chapter's four physical walls/barriers. The Workload Archetypes are classification labels, not separate explorable concepts -- students naturally encounter them when they manipulate Arithmetic Intensity in Part A. Hybrid architectures are a design pattern, not a tunable-parameter concept.

**Synthesis description:** Students face a concrete deployment decision: a voice assistant must process 1-second audio clips. They must determine (1) whether inference is memory-bound or compute-bound on a mobile NPU using Part A's arithmetic intensity analysis, (2) whether cloud offloading meets a 50 ms latency SLA given 500 km fiber distance using Part B, (3) whether sustained inference fits within a 4 W mobile thermal envelope using Part C, and (4) whether the energy cost of transmitting the audio to cloud exceeds local processing using Part D. The synthesis reveals that all four constraints point to the same answer (local inference wins), but for four different physical reasons. The key takeaway: "Deployment paradigm selection is not a preference -- it is a physics verdict rendered by whichever wall is tallest."

---

### Lab 03: The Constraint Tax (ML Workflow)

**Current 4 Parts:** A (DR Clinic Disaster -- 2^(N-1) cost), B (Iteration Velocity Race), C (Whole System View), D (Feedback Loops)

**Chapter coverage check:**
- Constraint Propagation Principle (2^(N-1)) -- covered by Part A
- ML vs Traditional Software / Iteration cycles -- covered by Part B
- Six Core Lifecycle Stages / Stage Interface Spec -- covered by Part C
- Multi-Scale Feedback / Emergent Complexity -- covered by Part D
- Lab-to-Field Data Gap -- subsumed by Part A's deployment constraint discovery
- Reproducibility and Technical Debt -- conceptual, no tunable parameter
- Data dominates effort (60-80% of time) -- could be quantified but Part C likely covers this

**Verdict: 4 parts is sufficient -- add Synthesis only.**

The four parts cover the chapter's four Systems Thinking principles. The lifecycle stages are organizational (taxonomy), not explorable. The constraint propagation formula is the most quantitative concept and gets its own Part. There is no significant gap.

**Synthesis description:** Students walk through a complete DR screening deployment scenario where they must apply constraint propagation (Part A) to identify the cost of a late-discovered edge case (elderly patients), estimate iteration velocity (Part B) to determine how many development cycles they can afford within a 6-month deadline, trace the whole-system impact (Part C) of choosing TinyML deployment on all upstream stages, and design a feedback loop schedule (Part D) that catches demographic drift within acceptable degradation bounds. The synthesis question: "Your DR system passed all offline evaluations but fails on 12% of patients over 70. At which lifecycle stage should this demographic constraint have been defined, what is the cost multiplier of discovering it at Stage 6, and which feedback loop timescale will detect similar drift in production?"

---

### Lab 04: The Data Gravity Trap (Data Engineering)

**Current 4 Parts:** A (Feeding Tax -- GPU starvation), B (Data Gravity -- move compute not data), C (Data Cascades -- 2% error leads to 15% accuracy loss), D (False Positive Trap -- KWS always-on)

**Chapter coverage check:**
- The Feeding Problem / IO Bottleneck / Dataloader Choke Point -- covered by Part A
- Data Gravity -- covered by Part B
- Data Cascades -- covered by Part C
- KWS Case Study / Four Pillars -- covered by Part D
- Data Drift Detection (PSI, KL divergence, covariate/label/concept drift) -- **significant concept with tunable parameters, not covered**
- Data Pipeline Architecture -- infrastructure, not explorable
- Storage Architecture -- taxonomy
- Data Labeling / Inter-annotator agreement -- covered partially in KWS
- Training-Serving Consistency -- conceptual
- Data Debt -- qualitative

Let me apply the Explorable Concept Test to Data Drift Detection:
1. Tunable parameter? YES -- drift rate, monitoring window size, PSI threshold
2. Students predict wrong? YES -- they expect drift to be obvious and sudden, not gradual and silent
3. Why is instructive? YES -- connects directly to the Degradation Equation from Ch 1, operationalizes it
4. Simulate in 5 min? YES -- simple time slider showing PSI accumulating
5. Connects to ecosystem? YES -- TinyTorch monitoring module, Ch 14 MLOps

Score: 5/5.

However, looking more carefully, Data Drift Detection is a major topic of Chapter 14 (ML Operations), not Chapter 4. Chapter 4 introduces the concepts and metrics, but the operational response is in Ch 14. And Lab 01 Part C (Silent Decay) already covers the Degradation Equation from the introduction. Adding drift detection to Lab 04 would create redundancy with Lab 01 Part C (which already has the "infrastructure metrics green while accuracy drops" insight) and with Lab 14 (which should own the operational drift detection story).

**Verdict: 4 parts is sufficient -- add Synthesis only.**

The four parts cover the chapter's four most distinctive data engineering concepts. Drift detection is introduced in this chapter but is more naturally the province of Lab 14 (ML Operations), and the "silent degradation" insight is already covered in Lab 01 Part C. The KWS case study in Part D provides the chapter's running example. There is no gap that warrants a 5th part without creating cross-lab redundancy.

**Synthesis description:** Students design a data pipeline for an always-on keyword spotting system that must achieve <1% false positive rate on a microcontroller with 512 KB SRAM. They integrate all four parts: (1) calculate whether the dataloader can feed training data fast enough to keep a GPU busy using the Feeding Tax from Part A, (2) determine whether to send raw audio to the cloud or train locally using the Data Gravity calculation from Part B, (3) trace how a 2% labeling error in the "hey device" keyword propagates through training to produce a specific false positive rate using the cascade model from Part C, and (4) evaluate whether their pipeline meets the always-on power constraint given the false positive rate, since each false positive triggers full-model inference that drains battery, using Part D. The synthesis reveals that data engineering decisions at the pipeline level directly determine whether the deployed system is physically viable.

---

### Lab 09: Data Selection

**Current 4 Parts:** A (ICR Frontier), B (Selection Inequality), C (Curriculum Learning), D (Cost-Optimal Frontier)

**Chapter coverage check:**
- ICR and diminishing returns -- covered by Part A
- Selection Inequality (selection cost vs training savings) -- covered by Part B
- Curriculum Learning (ordering matters) -- covered by Part C
- Compute-Optimal Frontier / Cost Modeling / ROI -- covered by Part D
- Self-Supervised Learning / Foundation Model Amortization -- **significant concept**
- Active Learning -- a dynamic selection method, but the "human-in-the-loop budget" aspect is less of a surprise
- Coreset Selection Algorithms -- mechanism, not a separate explorable
- Data Echoing / Hardware Empathy -- infrastructure optimization
- Distributed Selection -- systems concern

Self-Supervised Learning / Amortization: Let me apply the test:
1. Tunable parameter? YES -- number of downstream tasks, pre-training cost, fine-tuning cost per task
2. Students predict wrong? MAYBE -- students may already know foundation models are efficient. The crossover point (how many tasks before pre-training pays off) might surprise some but is fairly intuitive.
3. Why is instructive? SOMEWHAT -- it teaches amortization economics, but this is more arithmetic than insight
4. Simulate in 5 min? YES -- simple cost comparison calculator
5. Connects to ecosystem? YES -- foundation model fine-tuning is everywhere

Score: 3/5. Falls short of the 4/5 threshold. The "aha" is weak -- students who have heard of ChatGPT already understand that pre-training is expensive but amortized. The crossover calculation is arithmetic, not a trade-off curve with surprising shape. Part D (Cost-Optimal Frontier) likely already subsumes the economics.

**Verdict: 4 parts is sufficient -- add Synthesis only.**

The four parts cover the chapter's four most counterintuitive findings. Self-supervised learning economics are interesting but lack a strong "prediction-wrong" moment. The Cost-Optimal Frontier in Part D already covers ROI analysis, which is the economic lens that subsumes the amortization argument.

**Synthesis description:** Students face a budget allocation problem: they have 1,000 GPU-hours to train an image classifier on a 500K-image dataset. They must decide (1) how much of the dataset to use, applying the ICR frontier from Part A to find the knee, (2) whether to spend GPU-hours on coreset selection or just train on everything, using the Selection Inequality from Part B, (3) what training order to use for maximum convergence speed, using curriculum learning from Part C, and (4) whether the combined strategy (coreset + curriculum + reduced epochs) lands on or below the cost-optimal frontier from Part D. The synthesis reveals that composing all three techniques (selection, ordering, budget allocation) outperforms any single technique, but only if the selection overhead is accounted for honestly.

---

### Lab 10: Model Compression

**Current 4 Parts:** A (Quantization Free Lunch), B (Pruning Hardware Trap), C (Compression Pareto Frontier), D (Energy Dividend)

**Chapter coverage check:**
- Quantization accuracy-vs-bitwidth curve -- covered by Part A
- Pruning: unstructured vs structured / hardware trap -- covered by Part B
- Multi-technique Pareto frontier / deployment context -- covered by Part C
- Energy efficiency of INT8 vs FP32 -- covered by Part D
- Knowledge Distillation -- **significant concept, not covered**
- Neural Architecture Search (NAS) -- mechanism, not a quick simulation
- Sparsity Exploitation / N:M structured sparsity -- covered by Part B's pruning topic
- Compound Scaling (EfficientNet) -- conceptual, derivative
- Adaptive Computation (early exit) -- interesting but narrow

Knowledge Distillation: Let me apply the test:
1. Tunable parameter? YES -- temperature (T), student/teacher size ratio, alpha (loss weighting)
2. Students predict wrong? YES -- students expect a 10x smaller student to lose 10-20% accuracy; it actually retains 90-95%
3. Why is instructive? YES -- teaches that soft labels carry inter-class similarity information that hard labels discard
4. Simulate in 5 min? YES -- temperature slider showing how softmax distribution changes, accuracy vs compression ratio curve
5. Connects to ecosystem? YES -- TinyTorch distillation module, practical deployment

Score: 5/5.

However, I need to consider whether Part C (Compression Pareto Frontier) already covers distillation implicitly. Looking at the plan, Part C has students compose quantization AND pruning along a Pareto frontier for a 7B LLM across memory tiers. Distillation is a fundamentally different technique -- it produces a new, smaller architecture rather than compressing the existing one. Part C's Pareto frontier could include distillation as one of the techniques, but the "soft label temperature" insight is distinct from the "technique composition" insight.

The question is whether the distillation insight is important enough to warrant a 5th part, or whether it can be folded into Part C's technique options. Given that the chapter devotes a major section to knowledge distillation with its own formula (temperature-scaled softmax, KL divergence loss), and the insight about "dark knowledge" in soft labels is genuinely surprising and counterintuitive, I believe this warrants a 5th part.

**Verdict: Add Part E -- The Dark Knowledge Transfer.**

```yaml
part_e:
  name: "The Dark Knowledge Transfer"
  concept: "Knowledge distillation: a student model 10x smaller retains 90-95% of teacher accuracy because soft labels encode inter-class relationships that hard labels discard."

  aha_moment:
    students_expect: "A model 10x smaller than the teacher will lose proportionally -- maybe 20-30% accuracy drop."
    they_discover: "The student retains 90-95% of teacher accuracy -- far better than training the same small model from scratch on hard labels."
    because: "The teacher's softmax distribution at temperature T reveals inter-class similarity (cat is closer to dog than to airplane). These soft targets provide a richer gradient signal than one-hot labels, effectively transferring the teacher's learned structure."

  prediction:
    question: "A ResNet-50 teacher (76.1% top-1) is distilled into a MobileNetV2 student (10x fewer parameters). The student trained from scratch achieves 72.0%. What accuracy does the distilled student achieve?"
    options:
      - "A) ~72.0% -- the student architecture is the bottleneck, teacher cannot help"
      - "B) ~73.5% -- distillation recovers about half the gap"
      - "C) ~74.8% -- distillation recovers most of the gap (correct)"
      - "D) ~76.1% -- distillation perfectly transfers all knowledge"
    common_wrong_answer: "A -- students assume the small architecture is the binding constraint"
    why_wrong: "Hard labels (one-hot) waste information. The teacher's probability distribution over all classes provides gradient signal about class similarity that the student cannot learn from hard labels alone."

  parameters:
    - name: "Temperature (T)"
      range: "1-20"
      default: "4"
      effect: "Low T (=1) produces peaked distributions (nearly hard labels). High T softens the distribution, revealing inter-class structure. Sweet spot is T=3-5 for most tasks."
    - name: "Alpha (loss weight)"
      range: "0.0-1.0"
      default: "0.7"
      effect: "Balances distillation loss (soft targets from teacher) vs student loss (hard labels). Alpha=1.0 ignores hard labels entirely; alpha=0.0 ignores teacher."
    - name: "Student/Teacher size ratio"
      range: "0.05-0.5"
      default: "0.1"
      effect: "Smaller students benefit more from distillation (larger gap to close). Very small students (<5% of teacher) hit a capacity floor where even soft labels cannot help."

  simulation:
    primary_formula: "L_distill = alpha * T^2 * KL(softmax(z_t/T) || softmax(z_s/T)) + (1-alpha) * CE(y, softmax(z_s))"
    source: "@sec-model-compression-knowledge-distillation-1842, Hinton et al. 2015"
    simplifications: "Uses a simplified accuracy model where distillation_accuracy = scratch_accuracy + distill_boost * (teacher_accuracy - scratch_accuracy), where distill_boost depends on T and alpha. Real distillation requires actual training."

  interaction_with_other_parts: "Part C's Pareto frontier gains a third technique axis: students can now compose quantization (Part A) + structured pruning (Part B) + distillation (Part E) to reach memory tiers that no single technique can achieve."
```

**Synthesis description:** Students face the complete compression pipeline for deploying a 7B LLM on a 4 GB mobile device. They must compose all five parts: (1) quantize from FP32 to INT8 using the Free Lunch zone from Part A, (2) apply structured pruning at the right sparsity level avoiding the hardware trap from Part B, (3) evaluate whether their configuration is Pareto-optimal from Part C, (4) calculate the energy savings to determine if the compressed model meets a 5 Wh daily battery budget from Part D, and (5) use knowledge distillation to produce a smaller architecture that further closes the gap from Part E. The synthesis reveals that the optimal compression strategy depends on the deployment target: the cloud tier uses only quantization, the mobile tier needs quantization + pruning, and the TinyML tier requires all three including distillation to a fundamentally different architecture.

---

### Lab 12: Benchmarking

**Current 4 Parts:** A (Amdahl Ceiling), B (Peak vs Sustained -- Thermal Cliff), C (Multi-Metric Trap), D (Training vs Inference -- Different Games)

**Chapter coverage check:**
- Amdahl's Law / component vs system speedup -- covered by Part A
- Peak vs Sustained / Thermal throttling -- covered by Part B
- Multi-metric trade-offs / Pareto frontiers -- covered by Part C
- Training vs Inference metrics differ -- covered by Part D
- Power Measurement Techniques / TOPS-per-Watt -- partially covered by Part B's thermal story
- Benchmark Gaming (Goodhart's Law) -- **interesting concept**
- Lab-to-Production Gap -- covered by Part C implicitly
- Model and Data Evaluation (beyond system benchmarks) -- covered by Part D
- Micro/Macro/End-to-End granularity -- taxonomy, not explorable

Benchmark Gaming / Goodhart's Law: Let me apply the test:
1. Tunable parameter? SOMEWHAT -- you could toggle "gaming techniques" on/off and see reported vs real performance
2. Students predict wrong? MAYBE -- students already know vendors exaggerate, but quantifying the gap is instructive
3. Why is instructive? SOMEWHAT -- teaches skepticism about vendor claims, but it is more of a warning than a trade-off
4. Simulate in 5 min? MARGINAL -- it is more of a reveal than an exploration
5. Connects to ecosystem? YES -- critical for production decisions

Score: 3/5. Falls short. Benchmark gaming is more of a cautionary tale than an explorable trade-off. Students cannot meaningfully "tune" gaming parameters in a way that produces surprising curves. It belongs in the Fallacies section, not as a lab part.

**Verdict: 4 parts is sufficient -- add Synthesis only.**

The four parts cover the chapter's four most important quantitative insights. Benchmark gaming is important but lacks the interactive trade-off structure needed for a lab part. The power measurement topic is partially subsumed by Part B (thermal throttling directly affects TOPS/Watt claims). Training vs Inference (Part D) covers the remaining major dimension.

**Synthesis description:** Students evaluate a vendor's claim: "Our new edge accelerator delivers 50 TOPS, 5x faster inference than the competition, at only 15 W." They must debunk the claim layer by layer: (1) apply Amdahl's Law from Part A to show that 5x inference speedup in a pipeline with 40% preprocessing yields only 1.7x end-to-end improvement, (2) check whether 50 TOPS is peak or sustained by simulating 5 minutes of continuous operation from Part B, revealing thermal throttling drops it to 25 TOPS, (3) evaluate whether the 25 TOPS / 15 W ratio meets all deployment SLOs simultaneously (accuracy, latency, power, cost) using the multi-metric analysis from Part C, and (4) determine whether the "50 TOPS" figure was measured during training-style batch processing or inference-style single-query serving from Part D. The synthesis takeaway: "A single benchmark number is a marketing claim. A systems engineer needs four numbers: sustained throughput, end-to-end latency, power under load, and accuracy at operating precision."

---

## Summary

| Lab | Verdict | Rationale |
|-----|---------|-----------|
| **01 (Introduction)** | 4 parts sufficient -- add Synthesis | DAM, Iron Law, Silent Decay, Deployment Spectrum cover all major quantitative concepts. Remaining sections are taxonomy/history. |
| **02 (ML Systems)** | 4 parts sufficient -- add Synthesis | Memory Wall, Light Barrier, Power Wall, Energy of Transmission map to the four physical walls. Workload archetypes are diagnostic labels, not separate explorables. |
| **03 (ML Workflow)** | 4 parts sufficient -- add Synthesis | Constraint propagation, iteration velocity, whole-system view, and feedback loops cover the Systems Thinking section completely. Remaining sections are stage descriptions (taxonomy). |
| **04 (Data Engineering)** | 4 parts sufficient -- add Synthesis | Feeding Tax, Data Gravity, Data Cascades, and KWS False Positive Trap cover the distinctive data engineering concepts. Drift detection is better owned by Lab 14 (ML Operations). |
| **09 (Data Selection)** | 4 parts sufficient -- add Synthesis | ICR Frontier, Selection Inequality, Curriculum Learning, and Cost-Optimal Frontier cover the chapter's four most counterintuitive findings. SSL/amortization economics lack a strong "wrong prediction" moment. |
| **10 (Model Compression)** | **Add Part E -- The Dark Knowledge Transfer** | Knowledge distillation is a major chapter section with its own formula, a genuinely surprising accuracy retention result, tunable parameters (temperature, alpha), and a distinct mechanism (soft labels) not covered by the existing four parts. Score: 5/5 on the Explorable Concept Test. |
| **12 (Benchmarking)** | 4 parts sufficient -- add Synthesis | Amdahl Ceiling, Thermal Cliff, Multi-Metric Trap, and Training vs Inference cover the chapter's four quantitative insights. Benchmark gaming lacks the interactive trade-off structure needed for a lab part (3/5 on the test). |

---
