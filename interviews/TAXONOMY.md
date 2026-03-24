# The ML Systems Reasoning Taxonomy

**Version 5.3 — Expert-Reviewed, Principled, Top-Down, Citable**

> *"Constraints drive architecture."*
> — The Physics of AI Engineering

---

## 1. Motivation

How should we systematically organize the knowledge required of a Staff-level ML Systems engineer?

The naive approach extracts topics from a textbook and generates questions per topic. This produces a flat, unbalanced question bank where coverage reflects *page count*, not *engineering importance*. Professional assessment systems — USMLE, CPA, FE/PE exams — don't work this way. They start from a **competency model**: what must practitioners be able to *do*? Then they derive what practitioners must *know* to do it.

This document defines the **ML Systems Reasoning Taxonomy** — a principled, hierarchical classification of the knowledge and reasoning skills required to design, build, optimize, and operate machine learning systems. It serves as the canonical reference for:

- **StaffML** — the interview preparation platform (question classification, study plans, gap analysis)
- **The MLSysBook** — the two-volume textbook (chapter organization, learning objectives)
- **Research** — the methodology paper (citable taxonomy with academic grounding)

### 1.1 Design Principles

1. **Top-down, not bottom-up.** Structure flows from reasoning competencies → knowledge areas → concepts, not from textbook chapters → extracted topics.

2. **Reasoning-first.** The primary axis is *how you think* (bottleneck analysis, scaling reasoning, cost estimation), not *what you know* (HBM specs, AllReduce algorithms, LoRA).

3. **Enduring over ephemeral.** The taxonomy tests principles that survive technology transitions. "Arithmetic intensity determines compute- vs memory-boundedness" is enduring. "Use `torch.compile()` for graph optimization" is ephemeral.

4. **Citable and reproducible.** Every structural decision references established frameworks (Bloom's Taxonomy, Evidence-Centered Design, faceted classification).

5. **Navigable.** Like NeetCode's roadmap (~20 nodes in a prerequisite DAG), the taxonomy must be simple enough to visualize on one page but rich enough to guide study.

### 1.2 Academic Grounding

| Framework | Citation | How We Use It |
|-----------|----------|---------------|
| Revised Bloom's Taxonomy | Anderson & Krathwohl (2001) | 2D classification: Knowledge Type × Cognitive Process. Our L1-L6+ levels. |
| Evidence-Centered Design | Mislevy et al. (2003) | Tasks require multi-dimensional characterization. Our 6-axis faceted system. |
| Faceted Classification | Ranganathan (1933); Hjørland (2013) | Independent orthogonal axes, not a single hierarchy. |
| IEEE Learning Object Metadata | IEEE LOM 1484.12.1 (2002) | Supports arbitrary repeatable classification dimensions. |
| USMLE Test Blueprinting | NBME / Case & Swanson (2002) | 4+ independent axes per item. Our model for multi-axis classification. |
| User Warrant vs Literary Warrant | Soergel (1985); Hjørland (2013) | Top-down (user need) over bottom-up (source text). Our design principle #1. |

### 1.3 Expert Review

Version 5.1 incorporates feedback from 7 expert reviewers:

| Reviewer | Perspective |
|----------|------------|
| David Patterson | Computer architecture pedagogy, textbook design (Turing Award, UC Berkeley) |
| Jeff Dean | Google-scale ML infrastructure (Google Senior Fellow) |
| Jensen Huang | Accelerated computing, GPU/CUDA ecosystem (NVIDIA CEO) |
| Mark Zuckerberg | Open-source AI at social scale, PyTorch, custom silicon (Meta CEO) |
| Chip Huyen | Production ML systems, MLOps (author, *Designing ML Systems*) |
| Song Han | Efficient AI, model compression, TinyML (MIT EECS) |
| Ion Stoica | Distributed systems, Ray/Spark (UC Berkeley, Anyscale/Databricks) |

Key consensus changes in v5.1: +2 reasoning principles (Locality, Observability), +7 knowledge areas (data engineering expanded, compiler infra, compound AI systems, Edge/TinyML split), +1 reasoning mode (production-debugging, renamed to failure-to-root-cause in v5.3), ~30 new concept tags, reweighted blueprint (cloud 55%, cost-efficiency 10%).

v5.2 additionally incorporates Gemini 3.1 Pro review feedback: added Concurrency & Asynchrony as 13th reasoning principle, added High-Performance Storage knowledge area, expanded concept tags for storage tiering, RDMA, ring attention, and CUDA stream synchronization.

v5.3 incorporates academic reviewer feedback (Patterson 8/10, Emer 7/10, Reddi 7.5/10): added disambiguation protocol for multi-principle questions (target: Cohen's kappa ≥ 0.7), renamed 3 reasoning modes to eliminate Axis 3↔5 entanglement, merged 3 thin knowledge areas (44 → 35), cleaned layer leakage (tool names moved to Layer 3), reframed Domain F as user-warranted, expanded B8 benchmarking scope, reframed RC-13 around correctness-performance tradeoff, fixed 3 weak equations.

---

## 2. The Three-Layer Architecture

The taxonomy has three layers, ordered from most abstract (enduring) to most concrete (evolving):

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: REASONING PRINCIPLES (13)                         │
│  The "physics" — permanent laws of ML systems engineering    │
│  ► Memory Wall, Amdahl's Law, Locality, Concurrency         │
├─────────────────────────────────────────────────────────────┤
│  LAYER 2: KNOWLEDGE AREAS (35 topic groups)                 │
│  The "engineering domains" — stable skill clusters           │
│  ► GPU Memory Hierarchy, Distributed AllReduce, Serving     │
├─────────────────────────────────────────────────────────────┤
│  LAYER 3: CONCEPT VOCABULARY (~150 tags)                    │
│  The "specific knowledge" — evolves with technology          │
│  ► HBM3e specs, vLLM PagedAttention, LoRA rank selection    │
└─────────────────────────────────────────────────────────────┘
```

**Layer 1** answers: *What reasoning skill does this test?*
**Layer 2** answers: *What topic area is this about?*
**Layer 3** answers: *What specific knowledge is needed?*

A question lives at the intersection of all three layers. For example:

> *"An A100 GPU has 2 TB/s HBM bandwidth and 312 TFLOPS FP16. A transformer attention layer performs 8 FLOPs per byte loaded. Is this layer compute-bound or memory-bound?"*

- **Layer 1**: Bottleneck Analysis (reasoning about arithmetic intensity and roofline position)
- **Layer 2**: Compute Characterization (the topic group)
- **Layer 3**: `arithmetic-intensity`, `roofline-model`, `hbm-bandwidth` (specific concepts)

---

## 3. Layer 1: The 13 Reasoning Principles

These are the enduring principles of ML systems engineering — the "physics" that every chapter of the textbook teaches. They don't change when you swap PyTorch for JAX, or A100 for H200, or transformers for state-space models.

Each principle is stated as an **invariant** — a law that holds regardless of specific technology — accompanied by a **canonical equation** that makes the principle quantitative.

### RC-1: Resource Quantification
> **Invariant:** *Every workload has a quantifiable resource signature: compute (FLOPS), memory (bytes), bandwidth (bytes/sec), energy (watts).*

**Canonical equation:** The Iron Law of ML Systems:
$$T_{total} = \frac{D_{vol}}{BW} + \frac{O}{R_{peak} \times \eta} + L_{overhead}$$

**Textbook roots:** The Iron Law, D·A·M Taxonomy (Data, Algorithm, Machine)

**What it tests:** Can you calculate the memory footprint of a model? The FLOPS required for one forward pass? The bandwidth needed to feed the accelerator?

**Example questions:**
- "Calculate the memory required to store a 7B parameter model in FP16" (napkin-math)
- "What is the arithmetic intensity of a matrix multiply M×N×K on this hardware?" (napkin-math)
- "A recommendation model spends 80% of inference time on embedding lookups but only 5% of FLOPS. Is it compute-bound or memory-bound?" (symptom-to-cause)

### RC-2: Bottleneck Analysis & Decomposition
> **Invariant:** *Performance is limited by the scarcest resource. Identifying which resource is scarce — and quantifying the gap between current and achievable performance — is the fundamental diagnostic skill.*

**Canonical equation:** The Roofline Model:
$$\text{Attainable FLOPS} = \min(R_{peak},\ BW \times I)$$
where $I$ = arithmetic intensity (FLOPS/byte)

**Textbook roots:** Arithmetic Intensity, Memory Wall, Roofline Model, Amdahl's Law

**What it tests:** Given a system and workload, can you identify what's limiting performance? Can you decompose end-to-end time into components and identify the dominant term?

**Example questions:**
- "This GPU shows 40% compute utilization. What's the bottleneck?" (symptom-to-cause)
- "Plot this workload on the roofline. Is it compute-bound or memory-bound?" (napkin-math)
- "Applying Amdahl's Law, what's the maximum speedup from parallelizing this operation?" (napkin-math)

### RC-3: Hardware-Compiler-Algorithm Co-Design
> **Invariant:** *Algorithms, compilers, and hardware form a co-design triangle. An algorithm's efficiency depends on how the compiler maps it to available hardware primitives, and hardware evolves to support the operations that compilers and algorithms demand.*

**Canonical equation:** Effective utilization decomposition:
$$\eta = \frac{\text{Achieved FLOPS}}{\text{Peak FLOPS}} = \eta_{\text{algorithm}} \times \eta_{\text{compiler}} \times \eta_{\text{hardware}}$$
where $\eta_{\text{algorithm}}$ = fraction of FLOPS that are "useful" (vs padding/redundant), $\eta_{\text{compiler}}$ = fraction of hardware primitives the compiler can exploit (e.g., Tensor Core coverage), $\eta_{\text{hardware}}$ = fraction of peak sustained under thermal/power limits. Each factor is independently measurable.

**Textbook roots:** The Silicon Contract, Inductive Bias ↔ Hardware Mapping, Frameworks as Compilers

**What it tests:** Can you map a computation to hardware? Can you explain why Flash Attention is fast on GPUs but yields smaller gains on TPUs? Can you reason about when to use library defaults vs custom kernels?

**Example questions:**
- "Flash Attention achieves 2-4× speedup by exploiting GPU memory hierarchy. Why would the same optimization yield smaller gains on a TPU?" (tradeoff-analysis)
- "You're choosing between H100 and TPU v5e for a 70B MoE model. The MoE routing creates irregular memory access patterns. Which hardware is better suited?" (tradeoff-analysis)
- "The H100 introduced Transformer Engine with automatic FP8. Why does this exist in hardware rather than software?" (requirements-to-architecture)

### RC-4: Scaling Reasoning
> **Invariant:** *Scale is qualitatively different from magnitude. Doubling a system doesn't just double its performance — it introduces new bottlenecks (communication overhead, synchronization, fault probability). At every order of magnitude, the dominant bottleneck shifts.*

**Canonical equation:** Amdahl's Law:
$$S = \frac{1}{(1-p) + \frac{p}{N}}$$
where $p$ = parallelizable fraction, $N$ = number of processors

**Textbook roots:** Strong/Weak Scaling, Communication-Computation Overlap, The Scale Moment (Vol II)

**What it tests:** What happens when you 10× the model, data, or cluster? Can you identify phase transitions where the dominant bottleneck shifts?

**Example questions:**
- "If we 4× the batch size, how does gradient synchronization overhead change?" (napkin-math)
- "At what cluster size does network bandwidth replace compute as the bottleneck for this training job?" (napkin-math)
- "Compare strong vs weak scaling efficiency for this distributed training job" (tradeoff-analysis)

### RC-5: Representational Efficiency Reasoning
> **Invariant:** *Every model carries more representational capacity than its task requires. Identifying and removing that surplus — through precision reduction, parameter elimination, rank reduction, or architectural simplification — is a quantifiable engineering discipline. These techniques compose, and composition order matters.*

**Canonical equation:** Compression ratio:
$$CR = \frac{\text{Original size}}{\text{Compressed size}} = f(\text{precision} \times \text{sparsity} \times \text{rank} \times \text{architecture})$$

**Textbook roots:** Conservation of Complexity (precision axis), Model Compression, Quantization

**What it tests:** What precision do you need? What do you lose? Can you design a compression pipeline combining pruning, quantization, and distillation in the right order?

**Example questions:**
- "Calculate the memory savings from quantizing a 70B model from FP16 to INT4" (napkin-math)
- "A 50% sparse INT8 model and a dense INT4 model have similar memory footprints. Compare their accuracy-latency profiles." (tradeoff-analysis)
- "Design a compression pipeline for deploying a 13B model on a single consumer GPU" (requirements-to-architecture)

### RC-6: Latency Decomposition
> **Invariant:** *End-to-end latency decomposes into measurable components: compute time + memory access time + communication time + overhead. Each component has a different optimization strategy. Tail latency amplification at scale makes P99 reasoning critical.*

**Canonical equation:** E2E latency:
$$L_{e2e} = L_{compute} + L_{memory} + L_{network} + L_{queue} + L_{overhead}$$

**Textbook roots:** Serving Inversion, Pipeline Analysis, Queuing Theory, Prefill vs Decode

**What it tests:** Can you break down E2E latency into components? Can you reason about P50 vs P99? Can you identify when tail latency amplification from fan-out makes the problem qualitatively different?

**Example questions:**
- "Decompose the inference latency for a 70B LLM serving request into prefill and decode phases" (napkin-math)
- "Why does batching improve throughput but hurt latency?" (concept-recall)
- "A request fans out to 16 shards. If individual P99 is 50ms, what's the request-level P99?" (napkin-math)

### RC-7: Fault & Reliability Reasoning
> **Invariant:** *At scale, failures are statistical certainties, not exceptional events. A 25,000-GPU cluster has ~1 failure every 4 hours. Correlated failures (power events, firmware bugs) are more dangerous than independent ones. Systems must be designed to tolerate, detect, and recover from failures — including silent ones like data drift.*

**Canonical equation:** Young-Daly optimal checkpoint interval:
$$\tau_{opt} = \sqrt{2 \cdot \delta \cdot M} - \delta$$
where $\delta$ = checkpoint duration, $M$ = mean time between failures

**Textbook roots:** MTBF/MTTR, The Three Walls (Vol II), Degradation Equation, Silent Failures

**What it tests:** Can you design for failure? Can you calculate optimal checkpoint frequency? Can you reason about silent degradation?

**Example questions:**
- "Given MTBF = 4 hours and checkpoint time = 10 minutes, what's the optimal checkpoint interval?" (napkin-math)
- "A model's accuracy dropped 3% over 2 months with no code changes. Walk through your debugging process." (failure-to-root-cause)
- "Design a fault-tolerant training loop for a 1000-GPU job that handles both independent and correlated failures" (requirements-to-architecture)

### RC-8: System Design
> **Invariant:** *Constraint Propagation — every upstream decision (data quality, model architecture, hardware choice) creates downstream constraints (memory budget, latency SLA, power envelope). System design is constraint satisfaction across the full stack.*

**Canonical relationship:** Constraint propagation chain (qualitative — not a computable equation, but the reasoning pattern that system design questions test):
$$\text{Architecture} \xrightarrow{\text{constrains}} \text{Memory} \xrightarrow{\text{constrains}} \text{Hardware} \xrightarrow{\text{constrains}} \text{Batch Size} \xrightarrow{\text{constrains}} \text{Latency} \xrightarrow{\text{must meet}} \text{SLA}$$
Each arrow represents a quantifiable constraint (e.g., "model requires 28 GB → must use ≥ 40 GB GPU → batch size ≤ 4 → P99 ≥ 120ms").

**Textbook roots:** Constraint Propagation, The System Is The Model, D·A·M co-dependency

**What it tests:** Given requirements, can you design an end-to-end system? Can you trace how a constraint in one layer propagates to other layers?

**Example questions:**
- "Design a serving system for a 70B LLM with P99 < 200ms and 1000 QPS" (requirements-to-architecture)
- "How does choosing a larger batch size affect memory, latency, and throughput?" (tradeoff-analysis)
- "Design a multi-model recommendation pipeline that evaluates 50+ models per user request within 100ms" (requirements-to-architecture)

### RC-9: Optimization Methodology
> **Invariant:** *Measurement precedes optimization. Optimizing without profiling is guessing. Fixing one bottleneck shifts constraints to the next (Conservation of Complexity). The engineer's job is to optimize the binding constraint and accept the non-binding ones.*

**Canonical relationship:** Conservation of Complexity (qualitative heuristic, not a computable equation):
When optimizing a system with $n$ constrained resources, relieving the binding constraint $C_b$ shifts the bottleneck to $C_{b+1}$:
$$\text{Optimize}(C_b) \implies C_{b+1}\ \text{becomes binding}$$
This is the systems analogue of Liebig's Law of the Minimum: performance is limited by the scarcest resource, and fixing it reveals the next-scarcest.

**Textbook roots:** Iteration Tax, Benchmarking Principles, Conservation of Complexity, Profiling

**What it tests:** Can you profile a system? Can you identify the actual bottleneck (not the assumed one)? Do you know when NOT to optimize?

**Example questions:**
- "This training job is 3× slower than expected. Walk through your debugging methodology." (optimization-task)
- "After fusing 3 operators, GPU utilization dropped. Why?" (failure-to-root-cause)
- "Compare nsys vs PyTorch Profiler for diagnosing this performance issue" (tradeoff-analysis)

### RC-10: Cost-Efficiency Reasoning
> **Invariant:** *Every system decision has a Total Cost of Ownership: hardware + energy + operations + environmental + opportunity cost. At scale, cost reasoning is often the primary constraint that shapes every architectural decision.*

**Canonical equation:** TCO per inference:
$$\text{Cost/query} = \frac{\text{Hardware amortization} + \text{Energy} + \text{Ops labor} + \text{Network}}{QPS \times \text{Utilization} \times \text{Uptime}}$$

**Textbook roots:** Data Wall, Sustainable AI, Jevons Paradox, TCO Analysis

**What it tests:** Can you reason about performance-per-dollar? Can you calculate TCO? Can you decide when "good enough" beats "optimal"?

**Example questions:**
- "Compare the 3-year TCO of training on A100 vs H100 clusters" (napkin-math)
- "Is it more cost-effective to serve with 4× smaller GPUs or 1× large GPU?" (tradeoff-analysis)
- "Your inference fleet runs at 30% utilization. What's the cost of that idle capacity per year?" (napkin-math)

### RC-11: Locality Reasoning
> **Invariant:** *Performance depends on exploiting data and computation locality — temporal (reuse in time) and spatial (proximity in space). Systems that violate locality pay a bandwidth and latency tax proportional to the distance data must travel.*

**Canonical equation:** Effective bandwidth with locality:
$$BW_{effective} = BW_{peak} \times \text{hit\_rate} + BW_{next\_level} \times (1 - \text{hit\_rate})$$

**Textbook roots:** Cache hierarchy, data gravity, NUMA-aware scheduling, KV-cache reuse, communication-computation overlap

**What it tests:** Can you reason about where data lives and how far it must travel? Can you exploit temporal and spatial locality to improve performance?

**Example questions:**
- "A training job's GPU utilization drops when batch size increases beyond 256. The data loading pipeline is on a remote NFS. Diagnose." (failure-to-root-cause)
- "Why does tensor parallelism place operations on the same GPU that holds the relevant weights?" (concept-recall)
- "Flash Attention achieves speedup by exploiting which form of locality?" (concept-recall)
- "Design a KV-cache eviction policy for long-context inference that exploits temporal locality" (requirements-to-architecture)

**Expert source:** David Patterson — *"Locality is arguably the single most important principle in computer systems and it is entirely absent from v5.0."*

### RC-12: Observability & Debuggability
> **Invariant:** *A system you cannot observe is a system you cannot fix. At scale, the dominant engineering cost is not building systems but diagnosing failures in them. Every design decision must consider: how will I know when this is broken, and how will I find the root cause?*

**Canonical equation:** Mean Time to Detect + Mean Time to Diagnose:
$$MTTD + MTTD_{iag} \ll MTTR$$
(Detection and diagnosis must be fast relative to recovery)

**Textbook roots:** Silent Failures, Degradation Equation, ML Operations, Monitoring

**What it tests:** Can you design observable systems? Can you systematically debug production issues? Can you distinguish between "the system is slow" and "the system is producing wrong answers"?

**Example questions:**
- "Your LLM starts producing lower-quality outputs with no code change. How do you detect and diagnose this?" (failure-to-root-cause)
- "A distributed training job's loss diverges at step 50,000. Is it a learning rate issue, data corruption, or hardware fault? Walk through your diagnosis." (failure-to-root-cause)
- "Design a monitoring system that detects both performance degradation and quality degradation for a serving fleet" (requirements-to-architecture)

**Expert source:** Jeff Dean — *"The ability to design observable systems and systematically debug production issues accounts for roughly 30% of what differentiates successful Staff engineers."*

### RC-13: Concurrency & Asynchrony Reasoning
> **Invariant:** *High utilization requires overlapping computation, communication, and I/O. Asynchrony trades correctness guarantees for performance: when is it safe to proceed without waiting? The core tradeoff is between utilization (overlap everything) and correctness (synchronize everything). Debugging asynchronous pipeline stalls and silent correctness violations is a core Staff skill.*

**Canonical equation:** Overlap efficiency:
$$\eta_{overlap} = \frac{T_{sequential}}{T_{overlapped}} = \frac{T_{compute} + T_{comm} + T_{IO}}{\max(T_{compute}, T_{comm}, T_{IO}) + T_{sync}}$$

**Textbook roots:** CUDA streams, non-blocking NCCL, async data prefetching, pipeline parallelism bubble time

**What it tests:** Can you reason about when asynchrony is safe vs when it introduces correctness issues? Can you design overlapping pipelines with minimal synchronization points? Can you debug stalls, deadlocks, and race conditions in async systems?

**Example questions:**
- "A training job has 30% GPU idle time despite full data pipeline. The profiler shows gaps between kernel launches. Diagnose." (failure-to-root-cause)
- "Design a 3-stage async pipeline (data loading, forward pass, gradient sync) with proper synchronization points" (requirements-to-architecture)
- "Non-blocking AllReduce overlaps gradient sync with backward pass. What happens if the next forward pass starts before sync completes?" (tradeoff-analysis)

**Expert source:** Gemini 3.1 Pro review — *"ML systems are fundamentally asynchronous. Debugging pipeline stalls, thread deadlocks, or CUDA stream synchronization issues is a massive part of a Staff engineer's job."*

### 3.1 Disambiguation Protocol

When a question tests multiple reasoning principles, use this protocol to assign the **primary** principle. Without consistent disambiguation, inter-rater reliability fails (target: Cohen's kappa ≥ 0.7).

**Step 1 — Diagnostic test:** "If I removed principle X from the question, would it still be a valid question?" If yes, X is secondary.

**Step 2 — Counterfactual test:** "What is the minimum set of principles needed to answer this?" That set determines the primary.

**Step 3 — Verb test:** Does the question stem use:
- *calculate, estimate, derive* → RC-1 (Resource Quantification)
- *diagnose, identify the bottleneck, what's limiting* → RC-2 (Bottleneck Analysis)
- *design, architect, propose* → RC-8 (System Design)
- *debug, root-cause, walk through your process* → RC-12 (Observability)
- *compare, tradeoff, which is better* → check context for specific RC

**Step 4 — Specificity tie-break:** When two principles tie after Steps 1–3, choose the more specific one. For example:
- RC-6 (Latency Decomposition) over RC-2 (Bottleneck Analysis) if the question is serving-specific
- RC-11 (Locality) over RC-1 (Resource Quantification) if the question is about data placement
- RC-13 (Concurrency) over RC-9 (Optimization) if the question is about pipeline overlapping

**RC-2 vs RC-6 disambiguation:** RC-6 applies when the question is specifically about decomposing end-to-end latency in a serving or pipeline context. RC-2 applies when the question is about identifying any resource bottleneck (compute, memory, bandwidth, network). When a question involves both, choose RC-6 if the answer requires reasoning about P50/P99, TTFT/TPOT, or pipeline stage timing.

**Worked example:**
> *"A 70B LLM serving system shows P99 latency of 800ms against a 200ms SLA. The prefill phase takes 600ms and decode takes 200ms. Where do you focus optimization?"*

- Diagnostic test: Removing RC-2 (bottleneck) → still valid (it's about latency components). Removing RC-6 (latency decomposition) → no longer valid (the whole question is about decomposing latency).
- Verb test: "Where do you focus" → RC-2 or RC-6.
- Specificity: This is serving-specific with TTFT/TPOT reasoning → **RC-6 is primary**, RC-2 is secondary.

---

## 4. Layer 2: Knowledge Area Roadmap (35 Topic Groups)

Like NeetCode's ~20 topic groups with prerequisite edges, the ML Systems Knowledge Area Roadmap organizes **what you need to know** into drillable groups with clear dependencies.

Each knowledge area:
- Contains questions across all difficulty levels (L1-L6+)
- Maps to 1-3 reasoning principles from Layer 1
- Has explicit prerequisite knowledge areas
- Corresponds to specific textbook chapters

### The Roadmap

The roadmap is organized into 6 domains, each containing topic groups. Prerequisites flow top-to-bottom within a domain and left-to-right across domains.

```
DOMAIN A                 DOMAIN B                 DOMAIN C
ML Foundations           Hardware & Compute       Systems & Scale
─────────────           ──────────────────       ─────────────────

A1 ML Workflow           B1 Number Systems        C1 Memory Hierarchy
                              │                        │
A2 Neural Computation    B2 Compute Arithmetic    C2 Data Movement
     │                        │                        │
A3 Network Architectures B3 Accelerator Design    C3 Interconnects
     │                        │                        │
A4 Training Dynamics     B4 Parallel Programming  C4 Collective Comms
     │                        │                        │
A5 Data Pipelines        B5 Graph Optimization    C5 Distributed Training
     │                        │                        │
A6 Feature Engineering   B6 ML Compiler Infra     C6 Fault Tolerance
     │                        │
A7 Data Quality & Drift  B7 Accel. Libraries      C7 Fleet Orchestration
                              │                        │
                         B8 Benchmarking           C8 Distributed Data
                                                       │
                                                  C9 HP Storage


DOMAIN D                 DOMAIN E                 DOMAIN F
Deployment & Serving     Efficiency & Governance   Compound AI Systems
─────────────────       ───────────────────────   ────────────────────

D1 Model Compression     E1 Power, Energy &       F1 Compound AI Systems
     │                      Sustainability
D2 Quantization               │
     │                   E2 Security, Privacy
D3 Inference Opt.           & Robustness
     │                        │
D4 Serving Systems       E3 Responsible AI
     │
D5 Edge & Mobile AI
     │
D6 TinyML & Embedded
     │
D7 MLOps & Production
```

### Detailed Knowledge Areas (35 total)

#### Domain A: ML Foundations

| ID | Knowledge Area | Description | Prerequisites | Reasoning Principles | Textbook |
|----|---------------|-------------|---------------|---------------------|----------|
| A1 | ML Workflow | The iterative development cycle: experiment → train → evaluate → deploy. Feedback loops, iteration speed, experiment tracking. | — | RC-9 Optimization | Vol1: ML Workflow |
| A2 | Neural Computation | From math to silicon: matrix multiply, activation functions, backpropagation, computational graphs, automatic differentiation. | — | RC-1 Resource Quantification, RC-3 Co-Design | Vol1: Neural Computation |
| A3 | Network Architectures | CNN, RNN, Transformer, MoE, SSM. Inductive bias ↔ hardware mapping. Attention complexity. Embedding-table-dominated architectures (DLRM). | A2 | RC-3 Co-Design, RC-1 Resource Quantification | Vol1: NN Architectures |
| A4 | Training Dynamics | Loss landscapes, optimizer state, learning rate schedules, gradient flow, convergence. Memory cost: 4× model size (optimizer + gradients). Critical batch size. | A2, A3 | RC-1 Resource Quantification, RC-9 Optimization | Vol1: Training |
| A5 | Data Pipelines & Storage | Data ingestion, preprocessing, storage formats (Parquet, TFRecord), distributed file systems, data versioning. I/O as bottleneck. | A1 | RC-8 System Design, RC-11 Locality | Vol1: Data Engineering |
| A6 | Feature Engineering | Feature stores, feature computation at scale, real-time vs batch features, embedding tables, feature interaction patterns, training-serving consistency. | A5 | RC-8 System Design, RC-6 Latency Decomp. | Vol1: Data Engineering, Vol2: Ops at Scale |
| A7 | Data Quality & Drift | Distribution shift detection, data validation, schema evolution, label quality, training-serving skew quantification, data debugging. | A5 | RC-12 Observability, RC-7 Fault & Reliability | Vol1: Data Engineering, Vol1: ML Ops |

#### Domain B: Hardware & Compute

| ID | Knowledge Area | Description | Prerequisites | Reasoning Principles | Textbook |
|----|---------------|-------------|---------------|---------------------|----------|
| B1 | Number Systems & Precision | FP32, FP16, BF16, FP8 (E4M3/E5M2), INT8, INT4, MXFP4. Representation range, precision loss, overflow/underflow. Hardware-specific formats. | A2 | RC-5 Representational Efficiency | Vol1: Neural Computation, Model Compression |
| B2 | Compute Arithmetic | FLOPS, MACs, arithmetic intensity, operational intensity, roofline model. Compute-bound vs memory-bound classification for both dense and sparse workloads. | B1, A2 | RC-1 Resource Quantification, RC-2 Bottleneck Analysis | Vol1: HW Acceleration |
| B3 | Accelerator Design | GPU (SIMT, Tensor Cores, SMs), TPU (systolic arrays), NPU, FPGA, custom ASICs (MTIA). Dataflow architectures. Accelerator selection tradeoffs (SRAM/HBM ratios, FLOPS/watt). | B2 | RC-3 Co-Design | Vol1: HW Acceleration |
| B4 | Parallel Programming Models | GPU parallel execution model: thread/warp/block/grid hierarchy, memory hierarchy (registers/shared/L2/HBM), occupancy analysis, memory coalescing, tiling strategies. Generalizes across accelerator programming APIs. | B2, B3 | RC-3 Co-Design, RC-9 Optimization | Vol1: HW Acceleration, Frameworks |
| B5 | Graph-Level Optimization | Operator fusion, graph rewriting, memory planning, constant folding, dead code elimination. Framework graph capture mechanisms. Static vs dynamic graph tradeoffs. | B4, A2 | RC-9 Optimization, RC-3 Co-Design | Vol1: Frameworks |
| B6 | ML Compiler Infrastructure | Multi-level IR dialect hierarchies, lowering passes, target-specific code generation, auto-tuning search spaces. Compiler-hardware co-design for new operators. | B5 | RC-3 Co-Design, RC-9 Optimization | Vol1: Frameworks |
| B7 | Accelerated Libraries & Runtimes | Math libraries, inference runtimes, serving frameworks. When to use library defaults vs custom kernels. Framework dispatch architecture and operator selection. | B4 | RC-3 Co-Design, RC-9 Optimization | Vol1: Frameworks, HW Acceleration |
| B8 | Benchmarking & Profiling | MLPerf design methodology (execution scenarios, compliance rules), micro/macro/E2E benchmarks, statistical significance, profiling and roofline analysis, power measurement methodology, benchmark gaming detection, full-stack performance debugging. | B2 | RC-9 Optimization, RC-2 Bottleneck Analysis, RC-12 Observability | Vol1: Benchmarking |

#### Domain C: Systems & Scale

| ID | Knowledge Area | Description | Prerequisites | Reasoning Principles | Textbook |
|----|---------------|-------------|---------------|---------------------|----------|
| C1 | Memory Hierarchy | Registers → L1 → L2 → DRAM → HBM → NVMe. Bandwidth at each level. Cache behavior, memory allocation, OOM diagnosis. | B2 | RC-1 Resource Quantification, RC-11 Locality | Vol1: HW Acceleration, Vol2: Compute Infra |
| C2 | Data Movement & Bandwidth | PCIe, NVLink, NVSwitch, CXL. Host-device transfers, peer-to-peer, DMA. Intra-node vs inter-node bandwidth asymmetry (18× gap: NVLink 900 GB/s vs IB 50 GB/s). | C1 | RC-2 Bottleneck Analysis, RC-11 Locality | Vol1: HW Acceleration, Vol2: Network Fabrics |
| C3 | Network Interconnects | InfiniBand, RoCE, Ethernet. Topology (fat-tree, torus, dragonfly). Bisection bandwidth, congestion, RDMA, ECMP. | C2 | RC-4 Scaling Reasoning, RC-2 Bottleneck Analysis | Vol2: Network Fabrics |
| C4 | Collective Communication | AllReduce, AllGather, ReduceScatter, AllToAll. Ring, tree, butterfly algorithms. Gradient compression, communication-computation overlap. | C3 | RC-4 Scaling Reasoning, RC-6 Latency Decomp. | Vol2: Collective Communication |
| C5 | Distributed Training | Data parallelism (DDP, FSDP, ZeRO), tensor parallelism, pipeline parallelism, expert parallelism, 3D parallelism. Task parallelism and actor-model parallelism (Ray). Synchronization overhead. FSDP implementation internals. | C4, A4 | RC-4 Scaling Reasoning, RC-8 System Design | Vol2: Distributed Training |
| C6 | Fault Tolerance | Checkpointing (sync, async, incremental), failure detection, elastic training, consensus protocols (Raft), MTBF/MTTR, Young-Daly formula. Lineage-based recovery, request-level fault tolerance, graceful degradation. Applies to training, serving, and data pipelines. | C3 | RC-7 Fault & Reliability | Vol2: Fault Tolerance |
| C7 | Fleet Orchestration | Multi-resource scheduling (Dominant Resource Fairness), locality-aware placement, deadline-aware scheduling, preemption and priority policies, cluster utilization vs job completion tradeoffs, gang scheduling, heterogeneous workload management, multi-tenancy. | C5, C6 | RC-8 System Design, RC-10 Cost-Efficiency | Vol2: Fleet Orchestration |
| C8 | Distributed Data Processing | Distributed dataflow (BSP, streaming, actor model), data partitioning and shuffles, distributed joins and aggregations, petabyte-scale deduplication, data quality scoring at scale, curriculum construction for LLM pre-training. | A5, C3 | RC-4 Scaling Reasoning, RC-8 System Design | Vol2: Data Storage |
| C9 | High-Performance Storage | Parallel filesystems (Lustre, GPFS), object storage (S3), NVMe tiers, storage tiering strategies (HBM → DRAM → NVMe → PFS → Object), checkpoint I/O at 100K-GPU scale, write amplification, caching layers (Alluxio). Storage as the binding constraint for large-scale training. | C1, C6 | RC-2 Bottleneck Analysis, RC-11 Locality | Vol2: Data Storage, Compute Infra |

#### Domain D: Deployment & Serving

| ID | Knowledge Area | Description | Prerequisites | Reasoning Principles | Textbook |
|----|---------------|-------------|---------------|---------------------|----------|
| D1 | Model Compression & Efficiency | Pruning (structured/unstructured/N:M), knowledge distillation, low-rank factorization (LoRA, SVD), neural architecture search. Sparsity exploitation. Compression pipeline design: composition order matters. | A3, A4 | RC-5 Representational Efficiency, RC-2 Bottleneck Analysis | Vol1: Model Compression |
| D2 | Quantization | Post-training and training-aware quantization methods. KV-cache quantization. FP8 formats (E4M3/E5M2). Calibration strategies, per-channel vs per-tensor granularity, accuracy-efficiency Pareto curves. Quantization-aware fine-tuning. Error propagation through layers. | B1, D1 | RC-5 Representational Efficiency | Vol1: Model Compression |
| D3 | Inference Optimization | KV-cache management, continuous batching, paged memory for attention, speculative decoding, prefill-decode disaggregation. TTFT vs TPOT. Sparse inference for embedding-heavy models. | B2, B5, A3 | RC-6 Latency Decomp., RC-9 Optimization | Vol2: Inference |
| D4 | Serving Systems | Load balancing, autoscaling, batching strategies, SLA management, A/B testing, canary deployments, model routing. Social-scale serving: multi-model orchestration per request, billion-entity feature stores, caching economics. | D3 | RC-6 Latency Decomp., RC-8 System Design | Vol1: Model Serving, Vol2: Inference |
| D5 | Edge & Mobile AI | Edge devices (Jetson, Qualcomm), mobile phones (NPU delegation, Core ML, TFLite), on-device inference, model conversion (ONNX), heterogeneous compute (CPU+GPU+NPU), power-constrained optimization. 1-8 GB RAM, 5-30W budget. | D1, D2 | RC-3 Co-Design, RC-10 Cost-Efficiency | Vol2: Edge Intelligence |
| D6 | TinyML & Embedded | Microcontrollers (Cortex-M, RISC-V), 256KB SRAM, no FPU, no OS. CMSIS-NN, TFLite Micro, fixed-point arithmetic. Federated learning for IoT. Sensor fusion. Privacy through locality. | D5 | RC-3 Co-Design, RC-11 Locality | Vol2: Edge Intelligence |
| D7 | MLOps & Production | CI/CD for ML, model versioning, safe deployment (canary, shadow, blue-green), drift detection, A/B testing infrastructure, incident response, on-call debugging, model rollback, observability dashboards. Experiment infrastructure at scale (10K+ concurrent tests). | D4, A1, A7 | RC-12 Observability, RC-7 Fault & Reliability | Vol1: ML Operations, Vol2: Ops at Scale |

#### Domain E: Efficiency & Governance

| ID | Knowledge Area | Description | Prerequisites | Reasoning Principles | Textbook |
|----|---------------|-------------|---------------|---------------------|----------|
| E1 | Power, Energy & Sustainability | TDP, DVFS, power capping, perf/watt, energy-proportional computing, thermal throttling, sustained vs burst performance. Power envelope as binding constraint for datacenter design. Carbon footprint, PUE, carbon-aware scheduling, Jevons Paradox, water usage, lifecycle assessment. | B3, C1 | RC-10 Cost-Efficiency, RC-1 Resource Quantification | Vol2: Sustainable AI |
| E2 | Security, Privacy & Robustness | Threat models (poisoning, evasion, extraction), differential privacy, TEEs, federated learning privacy, supply chain security. Adversarial robustness, distribution shift, failure mode analysis, FIT rates, systematic assessment. | C5, D4 | RC-7 Fault & Reliability, RC-8 System Design | Vol2: Security & Privacy, Robust AI |
| E3 | Responsible AI | Fairness metrics (equalized odds, demographic parity), disaggregated evaluation, bias auditing, regulatory compliance (EU AI Act). | E2 | RC-8 System Design, RC-10 Cost-Efficiency | Vol1: Responsible Engr, Vol2: Responsible AI |

#### Domain F: Compound AI Systems

> **Note on warrant:** Domain F is **user-warranted** (driven by industry need for Staff-level interview preparation), not **literary-warranted** (the textbook does not teach RAG or agentic systems directly). Content in Domain F is scoped strictly to **systems challenges** — latency composition, cache invalidation, retrieval scaling, orchestration reliability — not ML challenges (prompt engineering, chain-of-thought design).

| ID | Knowledge Area | Description | Prerequisites | Reasoning Principles | Textbook |
|----|---------------|-------------|---------------|---------------------|----------|
| F1 | Compound AI Systems | Multi-model orchestration (RAG pipelines, agentic workflows, tool use, dynamic DAG execution), cascading latency, cross-component consistency, evaluation of compound systems. Vector databases, approximate nearest neighbor search, embedding indexing, hybrid search (dense + sparse), retrieval-augmented generation architecture. Scoped to systems challenges: latency composition, cache invalidation, retrieval scaling. | D4, A3, D3 | RC-8 System Design, RC-6 Latency Decomp., RC-11 Locality | Vol2: Inference, Ops at Scale |

### Prerequisite DAG Summary

```
Total knowledge areas: 35
Total prerequisite edges: ~50
Maximum chain depth: 7 (A2 → B2 → B3 → B4 → B5 → D3 → D4)
Entry points (no prerequisites): A1 ML Workflow, A2 Neural Computation, B1 Number Systems
Terminal points: C8/C9, D6/D7, E3, F1
```

---

## 5. Layer 3: Concept Vocabulary (~150 Tags)

The concept vocabulary is the most concrete layer — the specific technical knowledge tested. Unlike Layers 1 and 2 (which are stable), this layer **evolves** as technology changes.

### Organization

Each concept tag:
- Belongs to exactly one knowledge area (Layer 2)
- Is multi-label on questions (a question can test 1-5 concepts)
- Has a canonical name (lowercase-kebab-case) and display name
- Maps back to one or more of the 659 detailed taxonomy concepts (for search)

### Example Concept Tags by Knowledge Area

```
A3 Network Architectures:
  transformer-attention, cnn-convolution, recurrent-networks,
  mixture-of-experts, state-space-models, positional-encoding,
  embedding-tables, feature-interaction, two-tower-architecture

A6 Feature Engineering:
  feature-stores, real-time-features, embedding-sharding,
  feature-computation-at-scale, training-serving-consistency

A7 Data Quality & Drift:
  distribution-shift, data-validation, schema-evolution,
  label-quality, training-serving-skew

B2 Compute Arithmetic:
  arithmetic-intensity, roofline-model, flops-counting,
  tensor-core-utilization, compute-bound-vs-memory-bound

B3 Accelerator Design:
  gpu-architecture, tpu-systolic-array, npu-design,
  simt-execution, streaming-multiprocessor,
  heterogeneous-compute, custom-silicon-design

B4 Parallel Programming Models:
  cuda-execution-model, occupancy-analysis, memory-coalescing,
  shared-memory-tiling, register-pressure, warp-divergence,
  stream-concurrency, cuda-stream-synchronization, pipeline-stalls

B6 ML Compiler Infrastructure:
  mlir-dialects, auto-tuning, triton-language,
  compiler-code-generation, lowering-passes

B7 Accelerated Libraries & Runtimes:
  cublas-cudnn-stack, tensorrt-optimization,
  library-vs-custom-kernel, framework-dispatch,
  graph-capture, autograd-systems

C1 Memory Hierarchy:
  hbm-bandwidth, gpu-shared-memory, cache-hierarchy,
  memory-allocation, memory-fragmentation, oom-diagnosis

C2 Data Movement & Bandwidth:
  nvlink-nvswitch, intra-inter-node-asymmetry,
  pcie-bandwidth, host-device-transfer, cxl-memory,
  rdma-congestion-control

C4 Collective Communication:
  allreduce-algorithms, ring-allreduce, gradient-compression,
  communication-computation-overlap, bandwidth-optimal-algorithms

C5 Distributed Training:
  data-parallelism, tensor-parallelism, pipeline-parallelism,
  fsdp-zero, 3d-parallelism, gradient-accumulation,
  task-parallelism, actor-model, distributed-framework-design

C6 Fault Tolerance:
  checkpointing, checkpoint-tiering, elastic-training, young-daly-formula,
  lineage-based-recovery, request-level-fault-tolerance,
  graceful-degradation, circuit-breaker-patterns

C9 High-Performance Storage:
  parallel-filesystems, object-storage-semantics, storage-tiering,
  checkpoint-io, write-amplification, caching-layers,
  nvme-tiers, spot-instance-survivability

D1 Model Compression & Efficiency:
  structured-pruning, unstructured-pruning, n-m-sparsity,
  knowledge-distillation, low-rank-factorization,
  neural-architecture-search, compression-pipeline-design,
  activation-sparsity, sparse-attention

D2 Quantization:
  post-training-quantization, quantization-aware-training,
  weight-only-quantization, kv-cache-quantization,
  fp8-formats, calibration-methods, mixed-precision-training,
  quantization-error-propagation, quantization-finetuning

D3 Inference Optimization:
  kv-cache, continuous-batching, paged-attention,
  speculative-decoding, prefill-decode-disaggregation,
  ttft-vs-tpot, sparse-inference, real-time-ranking,
  disaggregated-inference, ring-attention, context-length-scaling

D7 MLOps & Production:
  drift-detection, model-versioning, safe-deployment,
  incident-response, experiment-infrastructure,
  observability-dashboards, full-stack-performance-debug

F1 Compound AI Systems:
  rag-pipeline, agentic-workflows, tool-use,
  cascading-latency, multi-model-serving,
  compound-system-evaluation, vector-databases,
  approximate-nearest-neighbor, hybrid-search,
  embedding-indexing
```

**Target: ~150 concept tags total, each containing 3-8 of the 659 detailed taxonomy concepts for search.**

---

## 6. The 6-Axis Classification System

Every question in the StaffML corpus is classified on 6 independent axes.

### The Axes

```
┌──────────────────────────────────────────────────────────────┐
│                     QUESTION CLASSIFICATION                   │
│                                                               │
│  Axis 1: TRACK ─────── Where is this deployed?               │
│          cloud | edge | mobile | tinyml | global              │
│                                                               │
│  Axis 2: LEVEL ─────── How hard is this?                     │
│          L1 | L2 | L3 | L4 | L5 | L6+                       │
│                                                               │
│  Axis 3: REASONING ─── What principle does this test?        │
│  COMPETENCY            (13 values from Layer 1)              │
│                                                               │
│  Axis 4: KNOWLEDGE ─── What topic area is this about?        │
│  AREA                  (35 values from Layer 2)              │
│                                                               │
│  Axis 5: REASONING ─── How does this test you?               │
│  MODE                  (7 values — see below)                │
│                                                               │
│  Axis 6: CONCEPT ───── What specific knowledge is needed?    │
│  TAGS                  (~150 tags, multi-label)              │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Axis 5: Reasoning Modes (How the Question Tests You)

| Mode | Description | Typical Levels | Example Stem |
|------|-------------|----------------|--------------|
| **concept-recall** | Retrieve a fact, definition, or hardware constant | L1-L2 | "What is the bandwidth of HBM3e?" |
| **napkin-math** | Multi-step arithmetic from specs to derive a number | L2-L3 | "Calculate the memory for a 70B FP16 model" |
| **symptom-to-cause** | Given symptoms, identify the limiting resource | L3-L4 | "GPU shows 30% utilization. What's wrong?" |
| **tradeoff-analysis** | Compare alternatives quantitatively | L4-L5 | "Compare FSDP vs pipeline parallelism for this job" |
| **requirements-to-architecture** | Propose an architecture from requirements | L5-L6+ | "Design a serving system for 1000 QPS at P99 < 200ms" |
| **optimization-task** | Profile, diagnose, fix, verify | L4-L6+ | "This training is 3× slower than expected. Debug it." |
| **failure-to-root-cause** | Work backward from unexpected failure through data/model/infra layers to root cause | L4-L6+ | "Model accuracy dropped 5% over 2 weeks with no code changes. Diagnose." |

> **Naming rationale (v5.3):** Modes are named for the *reasoning process* (symptom → cause, requirements → architecture, failure → root cause), not for the *principle* they test. This eliminates the Axis 3 ↔ Axis 5 entanglement identified by Emer: "bottleneck-diagnosis" no longer mirrors RC-2, and "system-design" no longer mirrors RC-8.

---

## 7. The Question Blueprint

The blueprint defines the **target distribution** of questions across axes.

### By Reasoning Competency (Axis 3)

| Competency | Target % | Target Count (of 5,000) | Rationale |
|-----------|----------|------------------------|-----------|
| RC-1 Resource Quantification | 13% | 650 | Foundation — most questions start here |
| RC-2 Bottleneck Analysis | 13% | 650 | The core interview skill |
| RC-3 HW-Compiler-Algorithm Co-Design | 10% | 500 | Critical for Staff level |
| RC-4 Scaling Reasoning | 9% | 450 | Distinguishes Staff from Senior |
| RC-5 Representational Efficiency | 6% | 300 | Important but narrower scope |
| RC-6 Latency Decomposition | 8% | 400 | Essential for serving roles |
| RC-7 Fault & Reliability | 7% | 350 | Critical for production systems |
| RC-8 System Design | 10% | 500 | The capstone skill |
| RC-9 Optimization Methodology | 7% | 350 | Practical engineering skill |
| RC-10 Cost-Efficiency | 10% | 500 | Primary constraint at scale |
| RC-11 Locality Reasoning | 4% | 200 | Cross-cutting principle |
| RC-12 Observability & Debuggability | 5% | 250 | Staff differentiator (30% of Staff signal per Dean) |
| RC-13 Concurrency & Asynchrony | 3% | 150 | Core to GPU pipeline utilization |

### By Level (Axis 2)

| Level | Target % | Bloom's Alignment |
|-------|----------|-------------------|
| L1 | 8% | Remember — facts, constants, definitions |
| L2 | 12% | Understand — explain, compare, interpret |
| L3 | 23% | Apply — calculate, estimate, use formulas |
| L4 | 25% | Analyze — diagnose, decompose, debug |
| L5 | 22% | Evaluate — compare tradeoffs, justify decisions |
| L6+ | 10% | Create — design systems, propose solutions |

### By Track (Axis 1)

| Track | Target % | Rationale |
|-------|----------|-----------|
| cloud | 55% | Majority of Staff+ roles are cloud-scale |
| edge | 14% | Growing deployment target |
| mobile | 13% | Distinct optimization constraints |
| tinyml | 13% | Unique hardware/software tradeoffs |
| global | 5% | Cross-cutting principles |

### By Reasoning Mode (Axis 5)

| Mode | Target % |
|------|----------|
| concept-recall | 5% |
| napkin-math | 25% |
| symptom-to-cause | 18% |
| tradeoff-analysis | 18% |
| requirements-to-architecture | 20% |
| optimization-task | 7% |
| failure-to-root-cause | 7% |

---

## 8. Relationship to Existing Data

### Mapping Existing Fields

| Current Field | New Axis | Action |
|---|---|---|
| `track` (5 values) | Axis 1: Track | **Keep as-is** |
| `level` (L1-L6+) | Axis 2: Level | **Keep as-is** |
| `competency_area` (13 values) | — | **Retire as primary axis.** Derivable from Knowledge Area. Keep as metadata. |
| `taxonomy_concept` (659 values) | Axis 6: Concept Tags (partial) | **Preserve as `primary_concept`.** Each concept maps to one of ~150 tags. |
| `scope` (38 values) | — | **Retire.** Absorbed by Knowledge Area (Axis 4). Keep as metadata. |
| `bloom_level` (6 values) | — | **Keep as metadata.** Redundant with Level but useful for validation. |
| — (new) | Axis 3: Reasoning Competency | **Classify via LLM** (13 values: RC-1 through RC-13, using disambiguation protocol §3.1) |
| — (new) | Axis 4: Knowledge Area | **Classify via LLM** (35 values: A1 through F1, bootstrap from competency_area + taxonomy_concept) |
| — (new) | Axis 5: Reasoning Mode | **Derive mechanically (~40%) + LLM (~60%)** (7 renamed modes) |
| — (new) | Axis 6: Concept Tags | **Cluster 659 concepts → ~150 tags, then multi-label classify** |

### Backward Compatibility

All existing fields are preserved. New fields are additive. No data is deleted.

```json
{
  "id": "cloud-gpu-memory-L3-042",

  "track": "cloud",
  "level": "L3",

  "reasoning_competency": "RC-2",
  "knowledge_area": "C1",
  "reasoning_mode": "napkin-math",
  "concept_tags": ["hbm-bandwidth", "arithmetic-intensity", "compute-bound-vs-memory-bound"],

  "competency_area": "memory",
  "taxonomy_concept": "hbm-bandwidth",
  "primary_concept": "hbm-bandwidth",
  "scope": "Hardware Platform",
  "bloom_level": "apply",

  "title": "...",
  "scenario": "...",
  "details": { ... }
}
```

---

## 9. Versioning & Evolution

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2025-12 | Initial flat topic list from textbook |
| v2.0 | 2026-01 | Added competency areas (13) |
| v3.0 | 2026-02 | Extracted 549 concepts from textbook chapters |
| v4.0 | 2026-03 | 659 concepts, 746 prerequisite edges, 0 cycles |
| v5.0 | 2026-03-24 | Principled 3-layer taxonomy: 10 reasoning principles → 35 knowledge areas → ~120 concept tags. Top-down design. |
| v5.1 | 2026-03-24 | Expert-reviewed: 12 principles (+Locality, +Observability), 42 knowledge areas, 7 reasoning modes (+production-debugging), ~150 concept tags. |
| v5.2 | 2026-03-24 | +Concurrency & Asynchrony (13th principle), +High-Performance Storage (C9), +8 concept tags. 44 knowledge areas total. |
| **v5.3** | **2026-03-24** | **Academic reviewer fixes (Patterson/Emer/Reddi). +Disambiguation protocol (§3.1). Renamed 3 modes (bottleneck-diagnosis→symptom-to-cause, system-design→requirements-to-architecture, production-debugging→failure-to-root-cause). Merged 3 thin areas (E1+E2→E1, E3+E4→E2, F1+F2→F1): 44→35 knowledge areas. Cleaned layer leakage. Domain F user-warrant acknowledgment. Expanded B8. Reframed RC-13. Fixed 3 weak equations (RC-3, RC-8, RC-9).** |

### Evolution Policy

- **Layer 1 (Reasoning Principles):** Changes require paper revision. Expected frequency: rarely (these are enduring).
- **Layer 2 (Knowledge Areas):** Changes require taxonomy committee review. Expected frequency: annually (new areas may emerge).
- **Layer 3 (Concept Tags):** Changes via `vault.py` tooling. Expected frequency: quarterly (technology evolves).

---

## 10. Validation Plan

Per Patterson's recommendation, the taxonomy requires empirical validation before paper submission:

1. **Expert classification agreement:** 3 independent experts classify 100 sample questions. Measure inter-rater reliability (Cohen's kappa ≥ 0.7 target).
2. **Coverage analysis:** Map 200 real Staff interview questions (from Glassdoor, Blind, interview.io) to the taxonomy. Target: ≥95% coverage with no new knowledge area needed.
3. **Discriminative validity:** Verify that each axis provides independent information (low cross-axis correlation, especially Reasoning Competency vs Knowledge Area).

---

## Appendix A: References

1. Anderson, L. W., & Krathwohl, D. R. (2001). *A Taxonomy for Learning, Teaching, and Assessing: A Revision of Bloom's Taxonomy of Educational Objectives.* Longman.

2. Case, S. M., & Swanson, D. B. (2002). *Constructing Written Test Questions for the Basic and Clinical Sciences* (3rd ed.). National Board of Medical Examiners.

3. Hjørland, B. (2013). Facet analysis: The logical approach to knowledge organization. *Information Processing & Management*, 49(2), 545-557.

4. IEEE (2002). *IEEE Standard for Learning Object Metadata* (IEEE Std 1484.12.1-2002).

5. Mislevy, R. J., Steinberg, L. S., & Almond, R. G. (2003). On the structure of educational assessments. *Measurement: Interdisciplinary Research and Perspectives*, 1(1), 3-62.

6. Ranganathan, S. R. (1933). *Colon Classification.* Madras Library Association.

7. Soergel, D. (1985). *Organizing Information: Principles of Data Base and Retrieval Systems.* Academic Press.

8. Vijay Janapa Reddi et al. (2024). *Machine Learning Systems.* Harvard University Press.
