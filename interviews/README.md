<!-- DEV-BANNER-START -->
<div align="center">
<table>
<tr><td>
<h3>🚧 Under Active Development</h3>
<p>This component is being built on the <code>dev</code> branch and is <b>not yet available</b> on the live site.<br>
Content may be incomplete or change without notice. The published curriculum lives at <a href="https://mlsysbook.ai"><b>mlsysbook.ai</b></a>.</p>
<p>
<a href="https://github.com/harvard-edge/cs249r_book/tree/dev"><img src="https://img.shields.io/badge/branch-dev-orange?logo=git&logoColor=white" alt="dev branch"></a>
<a href="https://mlsysbook.ai"><img src="https://img.shields.io/badge/live_site-mlsysbook.ai-blue?logo=safari&logoColor=white" alt="live site"></a>
</p>
</td></tr>
</table>
</div>
<!-- DEV-BANNER-END -->

# The ML Systems Interview Playbook

<p align="center">
  <b>130+ systems design questions across Cloud, Edge, Mobile & TinyML tracks.</b><br>
  <i>You cannot prompt your way out of a silicon bottleneck.</i>
</p>

<p align="center">
  <a href="cloud/README.md">☁️ Cloud</a> ·
  <a href="edge/README.md">🤖 Edge</a> ·
  <a href="mobile/README.md">📱 Mobile</a> ·
  <a href="tinyml/README.md">🔬 TinyML</a> ·
  <a href="00_The_Architects_Rubric.md">📋 Self-Eval Rubric</a>
</p>

---

## Why This Exists

Students often ask me: *"How do I prepare for ML systems interviews?"* They can write a training loop, but they can't explain why a 175B parameter model will hit a communication wall across InfiniBand, or why serving a 128k context window will fragment the KV-cache and explode P99 latency. These are **physical constraints**, not code syntax — and they're exactly what companies like Meta, Google, and OpenAI test for.

The industry calls it *Mechanical Sympathy* — the ability to see past framework abstractions and reason about the hard physics of keeping 10,000 GPUs fed and 1 million users served. This playbook is my attempt to organize that knowledge into something you can actually study.

---

## Numbers Every ML Systems Engineer Should Know

Adapted from the textbook's [Machine Foundations](https://mlsysbook.ai/vol1/) appendix. **Memorize the ratios — they're physics. Use the absolute numbers as sanity checks.** All hardware values sourced from [`mlsysim/core/constants.py`](../mlsysim/core/constants.py), the single source of truth for the book.

### The Invariants (Physics — Will Not Change)

| Relationship | Ratio | Why it's stable |
|---|---|---|
| DRAM access vs FP16 compute | **~580×** more energy | Wire capacitance scales with distance |
| FP32 vs INT8 energy | ~18× | Bit width determines switching energy |
| FP32 vs FP16 energy | ~3.4× | Halving bits roughly halves energy |
| HBM vs L1 cache latency | ~300× slower | On-chip vs off-chip |
| SSD vs L1 cache latency | ~100,000× slower | Electrical vs flash |
| Network vs local memory | ~17× slower | Speed of light + switching |
| Light in fiber | ~200 km/ms | Cross-country US ≈ 40ms RTT |

### Scaling Rules (Arithmetic — Hardware Independent)

These formulas let you estimate memory, compute, and cache requirements from a model's parameter count. "7B" means a model with 7 billion parameters.

| What you're estimating | Formula | Example |
|---|---|---|
| Inference memory (FP16) | 2 bytes × params | 7B params × 2 bytes = **14 GB** |
| Inference memory (INT8) | 1 byte × params | 7B params × 1 byte = **7 GB** |
| Training memory (Adam, FP16+FP32) | **16 bytes × params** | 7B params × 16 bytes = **112 GB** |
| Inference compute (transformer) | ~2 FLOPs × params per token | 7B → ~**14 GFLOPs/token** |
| Training compute | ~6 FLOPs × params × tokens | 7B on 1T tokens → **4×10²² FLOPs** |
| KV-cache per token (all layers) | 2 × layers × heads × head_dim × 2 bytes | Llama 70B, 128k tokens → **~335 GB** |

### Current Hardware Snapshot (2024–2025)

| Category | Metric | Value |
|---|---|---|
| **Compute** | A100 FP16 Tensor Core | 312 TFLOPS |
| | H100 FP16 Tensor Core | 989 TFLOPS |
| | H100 FP8 Tensor Core | 1,979 TFLOPS |
| | B200 FP16 Tensor Core | 2,250 TFLOPS |
| **Memory BW** | A100 HBM2e | 2.0 TB/s |
| | H100 HBM3 | 3.35 TB/s |
| | B200 HBM3e | 8.0 TB/s |
| **Interconnect** | NVLink 4.0 (H100) | 900 GB/s |
| | NVLink 5.0 (B200) | 1,800 GB/s |
| | InfiniBand NDR | 400 Gbps (50 GB/s) |
| | PCIe Gen5 x16 | 64 GB/s |
| **Roofline Ridge** | A100 (FP16) | ~153 Ops/Byte |
| | H100 (FP16) | ~295 Ops/Byte |
| **Power** | A100 TDP | 400 W |
| | H100 TDP | 700 W |
| | B200 TDP | 1,000 W |
| **Latency** | L1 / Register | ~1 ns |
| | L2 Cache | ~4 ns |
| | HBM3 | ~300 ns |
| | PCIe Gen5 | ~1 μs |
| | InfiniBand | ~5 μs |
| | NVMe SSD | ~100 μs |

> **Source:** All values from the textbook's `constants.py`. When hardware generations change, update the constants file — every calculation in the book (and this playbook) updates automatically.

---

## Quick Start

Pick your level and start drilling:

| You are... | Start here |
|---|---|
| **Preparing for a screen** (Junior/Mid) | 🟢 Green-tagged questions in any round |
| **Building applied skills** (Mid) | 🔵 Blue-tagged questions — diagnose real systems |
| **Targeting Senior (L5)** | 🟡 Yellow-tagged questions + [Round 1](cloud/01_Single_Node_Physics.md) & [Round 3](cloud/03_Production_Serving.md) |
| **Targeting Staff+ (L6+)** | 🔴 Red-tagged questions + [Round 5: Visual Debugging](cloud/05_Visual_Architecture_Debugging.md) |
| **Mock interview practice** | [The Architect's Rubric](00_The_Architects_Rubric.md) — grade your own designs |

---

## Choose Your Track

Each track targets a different deployment regime. Pick the one that matches the roles you're interviewing for — or study multiple tracks to build breadth.

| Track | Focus | Questions | Rounds |
|---|---|---|---|
| [**☁️ Cloud**](cloud/README.md) | Data center training & serving at scale | 57 | 6 rounds + visual debugging |
| [**🤖 Edge**](edge/README.md) | Autonomous vehicles, robotics, industrial AI | 27 | 2 rounds |
| [**📱 Mobile**](mobile/README.md) | On-device AI for smartphones | 27 | 2 rounds |
| [**🔬 TinyML**](tinyml/README.md) | Microcontroller & ultra-low-power AI | 27 | 2 rounds |
| [**📋 Rubric**](00_The_Architects_Rubric.md) | Self-evaluation across 6 engineering axes | — | Shared |
| [**🗺️ Topic Map**](TOPIC_MAP.md) | Master plan — competency areas, gaps, and targets | — | Planning doc |

---

## Mastery Levels

Every question is tagged with a mastery level. These levels are modeled on engineering ladders at major tech companies (Google, Meta, etc.) but represent **cognitive thresholds** — each level tests a fundamentally different kind of thinking, mapped to [Bloom's taxonomy](https://en.wikipedia.org/wiki/Bloom%27s_taxonomy) and the **scope of ownership** expected at that career stage.

### The Framework

| Level | Scope | Cognitive Skill | What the interviewer hears |
|---|---|---|---|
| 🟢 **L3 — The Screen** | Own a **task** | **Recall & Define** | "The Roofline model relates compute to memory bandwidth." |
| 🔵 **L4 — The Practitioner** | Own a **component** | **Apply & Identify** | "This workload is memory-bound because its arithmetic intensity is below the ridge point." |
| 🟡 **L5 — The Architect** | Own a **system** | **Analyze & Predict** | "Switching from A100 to H100 won't help because the ridge point shifts right while our intensity stays at ~1." |
| 🔴 **L6+ — The Lead** | Own the **architecture** | **Synthesize & Derive** | "Let me derive the optimal parallelism dimensions from the NVLink topology, memory capacity, and pipeline bubble cost." |

### The Transitions

- **L3→L4:** You stop reciting and start diagnosing. You can look at a system and correctly classify what's happening — identify the bottleneck, name the failure mode, apply the right formula.
- **L4→L5:** You stop diagnosing and start predicting. You can reason about what happens when a constraint changes — a hardware upgrade, a traffic spike, a precision change — and explain *why* the system behaves differently.
- **L5→L6+:** You stop predicting known patterns and start deriving novel solutions from first principles. You can stand at a whiteboard with incomplete information and work backward from physics to architecture.

### How This Maps to Industry

The levels correspond to real expectations at major tech companies. The titles differ, but the cognitive bar is consistent.

| Level | Google | Meta | Amazon | What systems interviews test |
|---|---|---|---|---|
| **L3** | L3 (SWE II) | E3 (IC3) | SDE I | Can you define the concepts? Do you know the vocabulary of ML systems? |
| **L4** | L4 (SWE III) | E4 (IC4) | SDE II | Given a broken system, can you diagnose the root cause? |
| **L5** | L5 (Senior) | E5 (IC5) | Senior SDE | Given a working system and a changing constraint, can you predict what breaks? |
| **L6+** | L6 (Staff) | E6 (Staff) | Principal | Given a blank whiteboard and a set of requirements, can you derive the architecture from physics? |

> **For question contributors:** When writing a new question, ask yourself: *"What scope of reasoning does this require?"* If the answer is "name the concept," it's L3. If it's "diagnose this system," it's L4. If it's "predict what happens when X changes," it's L5. If it's "derive the solution from constraints," it's L6+.

---

## Deployment Tracks

The mastery levels tell you *how deeply* a question tests your reasoning. The deployment track tells you *which physics regime* it lives in. The same fundamental principle — **constraints drive architecture** — applies everywhere, but the constraints themselves are radically different depending on where the silicon sits.

Every question in this playbook is tagged with a deployment track. All four tracks now have substantial question banks — see the [Topic Map](TOPIC_MAP.md) for the full coverage matrix and expansion plan.

| Track | Where it runs | Primary constraint | Compute scale | Memory | Power budget |
|---|---|---|---|---|---|
| **☁️ Cloud** | Data center (H100, TPU, InfiniBand) | Memory bandwidth / network | PFLOPS | 80 GB HBM | 700W per chip |
| **🤖 Edge** | Autonomous vehicles, robotics, CCTV, industrial | Thermal envelope / real-time deadlines | TOPS | 8–32 GB DRAM | 15–75W per module |
| **📱 Mobile** | Smartphones (Snapdragon NPU, Apple ANE) | Battery life / shared resources | TOPS (shared) | 6–12 GB shared | 3–5W total device |
| **🔬 TinyML** | Microcontrollers (Cortex-M, RISC-V) | SRAM capacity / hard real-time | MFLOPS | 256 KB–2 MB SRAM | 1–100 mW |

### How the Same Topic Changes Across Tracks

The physics is universal. The numbers are not.

| Topic | ☁️ Cloud | 🤖 Edge | 📱 Mobile | 🔬 TinyML |
|---|---|---|---|---|
| **Roofline** | H100 ridge point ~295 Ops/Byte | Jetson Orin ridge point ~15 Ops/Byte | NPU ridge point varies by SoC | No FPU — integer-only roofline |
| **Memory** | KV-cache fragmentation in 80 GB HBM | Model + sensor pipeline in 32 GB DRAM | Model coexists with OS in shared RAM | Entire model must fit in on-chip SRAM |
| **Quantization** | FP16→INT8 for throughput | INT8→INT4 for thermal headroom | INT8 for NPU compatibility | INT8→binary to fit on chip |
| **Serving** | Continuous batching, PagedAttention | Hard real-time inference deadlines | On-device inference, thermal throttling | Single-shot, microsecond latency |
| **Fault tolerance** | Checkpoint 10,000 GPUs (MTBF) | Graceful degradation, functional safety | Crash recovery, model fallback | Watchdog timers, hard real-time guarantees |

When submitting a question, tag it with both a **mastery level** (L3–L6+) and a **deployment track** (Cloud, Edge, Mobile, TinyML). See each track's README for the specific topics that need questions.

---

## Topic Index

Every question is tagged with a topic. Use this index to study a specific concept across all rounds, or to find where we need more questions.

| Topic | Questions | Rounds | Coverage |
|---|---|---|---|
| **`roofline`** — Arithmetic Intensity, compute vs memory bound | The Profiling Crisis · The Roofline Shift · The Amdahl Ceiling · The Decoding Bottleneck | 1, 2, 3 | 🟡 Moderate |
| **`memory`** — VRAM accounting, memory hierarchy, energy | The Sequence Length Trap · The Energy-Movement Invariant · The OOM Error · Ch.6: KV-Cache · Ch.7: Data Parallelism | 1, 2, 5 | ✅ Strong |
| **`kv-cache`** — KV-Cache sizing, fragmentation, PagedAttention | The Sequence Length Trap · The Fragmentation Crisis · The Disaggregated Serving · Ch.3: Serving Stack · Ch.6: KV-Cache | 1, 3, 5 | ✅ Strong |
| **`precision`** — FP16/BF16/INT8, quantization, underflow | The Underflow Crisis · The Precision Trade-off | 1 | 🟡 Moderate |
| **`hardware`** — Tensor Cores, sparsity, silicon architecture | The Precision Trade-off · The Sparsity Fallacy | 1 | 🟡 Moderate |
| **`frameworks`** — JIT compilation, graph tracing, kernels | The Compilation Overhead | 1 | 🔴 Thin — needs expansion |
| **`data-pipeline`** — CPU starvation, preprocessing, ingestion | The Data Pipeline Stall · Ch.1: Dataloader | 1, 5 | 🔴 Thin — needs expansion |
| **`parallelism`** — DP, TP, PP, ZeRO, 3D parallelism | The OOM Error · The Pipeline Bubble · The Amdahl Ceiling · Dimensioning the 3D Cube · The Cross-Rack Stall · Ch.4: Pipeline · Ch.7: Data Parallelism | 2, 5 | ✅ Strong |
| **`network`** — NVLink, InfiniBand, Fat-Tree, AllReduce | The Cross-Rack Stall · The Oversubscription Choke · The Ring vs Tree Dilemma · Dimensioning the 3D Cube · Ch.2: Training Cluster | 2, 5 | ✅ Strong |
| **`fault-tolerance`** — Checkpointing, MTBF, stragglers | The Straggler Problem · The MTBF Crisis | 2 | 🔴 Thin — needs expansion |
| **`latency`** — TTFT, TPOT, tail latency, queueing theory | The Serving Inversion · The LLM Metrics · The Black Friday Collapse | 3 | ✅ Strong |
| **`serving`** — Batching, cold starts, speculative decoding | The Serverless Freeze · The Pre-computation Trade-off · The Batching Dilemma · The Disaggregated Serving · The Decoding Bottleneck · Ch.3: Serving Stack | 3, 5 | ✅ Strong |
| **`mlops`** — Drift, skew, deployment, technical debt | The '95% Problem' · The Silent Failure · The Training-Serving Skew · The Retraining Math · The Deployment Risk · Ch.5: Feature Store | 4, 5 | ✅ Strong |
| **`economics`** — TCO, retraining cost, sustainability | The Energy Economics · The Retraining Math | 4 | 🔴 Thin — needs expansion |
| **`security`** — Prompt injection, adversarial attacks | The Trust Boundary | 4 | 🔴 Thin — needs expansion |
| **`privacy`** — DP-SGD, membership inference | The Privacy Audit | 4 | 🔴 Thin — needs expansion |
| **`fairness`** — Subgroup evaluation, bias amplification | The Bias Amplifier | 4 | 🔴 Thin — needs expansion |

> **Want to contribute?** Topics marked 🔴 **Thin** are where we most need new questions. See [Contributing](#contributing) below.

---

## Every Answer Links Back to the Textbook

Each question includes a **📖 Deep Dive** link to the relevant chapter of [Machine Learning Systems](https://mlsysbook.ai). The questions prove the knowledge matters; the textbook teaches it.

| Round | Textbook Chapters Referenced |
|---|---|
| Silicon Physics | HW Acceleration, Data Engineering, Model Compression, Frameworks, Neural Computation |
| Distributed Infra | Distributed Training, Fault Tolerance, Fleet Orchestration, Network Fabrics |
| Production Serving | Model Serving, Benchmarking, Inference at Scale |
| Ops & Economics | ML Operations, Responsible Engineering, Sustainable AI, Security & Privacy, Robust AI |

---

## Contributing

We welcome questions from recent AI systems interviews.

1. **Pull Request:** Click the **➕ Add a Flashcard** link at the top of any round file and submit a question using the format below.
2. **Issue:** [Open an issue](https://github.com/harvard-edge/cs249r_book/issues/new) with your question and we'll work with you to shape it.

### Question Format

Every question follows this structure. Not all fields are required — use **Common Mistake** and **Napkin Math** where they add value.

```markdown
<details>
<summary><b>[LEVEL BADGE]: [Question Title]</b> · <code>topic-tag</code> · <code>☁️ cloud</code></summary>

**Interviewer:** [The scenario or crisis]

**Common Mistake:** [What most people say wrong — creates the "aha" moment]

**Realistic Solution:** [The physics/logic behind the correct answer]

> **Napkin Math:** [Quick back-of-envelope calculation with real numbers]

> **Key Equation:** $[The formula to memorize]$

**📖 Deep Dive:** [Link to the relevant textbook chapter]
</details>
```

---

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=80" width="80px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🧠 🎨 ✍️</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

---

<p align="center">
  <i>Wishing you all the best in your interviews and your engineering journey.</i><br>
  — <b>Vijay Janapa Reddi</b>
</p>
