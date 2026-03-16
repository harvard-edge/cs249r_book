# The ML Systems Interview Playbook

<p align="center">
  <b>43 systems design questions for AI infrastructure roles at frontier labs.</b><br>
  <i>You cannot prompt your way out of a silicon bottleneck.</i>
</p>

<p align="center">
  <a href="01_Single_Node_Physics.md">🧱 Silicon Physics</a> ·
  <a href="02_Distributed_Infrastructure.md">🚀 Distributed Infra</a> ·
  <a href="03_Production_Serving.md">⚡ Production Serving</a> ·
  <a href="04_Operations_and_Economics.md">💼 Ops & Economics</a> ·
  <a href="05_Visual_Architecture_Debugging.md">🖼️ Visual Debugging</a> ·
  <a href="00_The_Architects_Rubric.md">📋 Self-Eval Rubric</a>
</p>

---

## Why This Exists

An LLM can write a flawless PyTorch training loop. It cannot calculate why a 175B parameter model will hit a communication wall across InfiniBand, or why serving a 128k context window will fragment your KV-cache and explode your P99 latency. These are **physical constraints**, not code syntax.

Companies like Meta, Google, and OpenAI are hiring engineers who possess *Mechanical Sympathy* — the ability to see past framework abstractions and reason about the hard physics of keeping 10,000 GPUs fed and 1 million users served.

This playbook is your study guide for that frontier.

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

| Rule | Formula | Example |
|---|---|---|
| Inference memory (FP16) | 2 bytes × params | 7B → 14 GB |
| Inference memory (INT8) | 1 byte × params | 7B → 7 GB |
| Training memory (Adam) | **16 bytes × params** | 7B → 112 GB |
| Inference FLOPs (transformer) | ~2 × params per token | 7B → ~14 GFLOPs/token |
| Training FLOPs | ~6 × params × tokens | 7B on 1T tokens → 4×10²² FLOPs |
| KV-cache per token per layer | 2 × H × d_h × 2 bytes | Llama 70B, 1 token, all 80 layers ≈ 1.3 MB |

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
| **Targeting Senior (L5)** | 🟡 Yellow-tagged questions + [Round 1](01_Single_Node_Physics.md) & [Round 3](03_Production_Serving.md) |
| **Targeting Staff+ (L6+)** | 🔴 Red-tagged questions + [Round 5: Visual Debugging](05_Visual_Architecture_Debugging.md) |
| **Mock interview practice** | [The Architect's Rubric](00_The_Architects_Rubric.md) — grade your own designs |

---

## The Rounds

Real-world interviews for AI Systems roles are structured around domains of operational responsibility. These rounds mirror the exact structure you will face at frontier labs.

| Round | Focus | Questions | Key Concepts |
|---|---|---|---|
| [**1. Silicon Physics**](01_Single_Node_Physics.md) | What happens inside a single server | 9 | Roofline, Arithmetic Intensity, VRAM accounting, KV-cache, Tensor Cores |
| [**2. Distributed Infra**](02_Distributed_Infrastructure.md) | What happens when you exceed one node | 9 | 3D Parallelism, NVLink vs InfiniBand, AllReduce, Pipeline Bubbles, MTBF |
| [**3. Production Serving**](03_Production_Serving.md) | Surviving real user traffic | 9 | Tail Latency, PagedAttention, Continuous Batching, Speculative Decoding |
| [**4. Ops & Economics**](04_Operations_and_Economics.md) | Keeping systems healthy over time | 9 | TCO, Data Drift, Training-Serving Skew, DP-SGD, Prompt Injection |
| [**5. Visual Debugging**](05_Visual_Architecture_Debugging.md) | Spotting bottlenecks in diagrams | 7 | Architecture review, Data Gravity, Network topology, Memory layout |
| [**The Rubric**](00_The_Architects_Rubric.md) | How staff engineers grade designs | — | Self-evaluation across 6 engineering axes |

**Total: 43 questions** across 5 rounds + a self-evaluation rubric.

---

## Mastery Levels

Every question is tagged with a mastery level reflecting the "Funnel of Mastery" used in real systems engineering interviews.

| Level | Role | What they test |
|---|---|---|
| 🟢 **L3 — The Screen** | Junior / Mid | Can you define core systems concepts and hardware constraints? |
| 🟡 **L5 — The Architect** | Senior | Can you reason about trade-offs and bottleneck physics? |
| 🔴 **L6+ — The Lead** | Staff / Principal | Can you perform "Whiteboard Physics" on the fly? |

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

We welcome questions and scenarios from recent AI systems interviews.

1. **Submit an Issue:** Use the [New Interview Question](https://github.com/harvard-edge/cs249r_book/issues/new?template=interview_question.yml) template.
2. **The 10-Upvote Rule:** Once a community question reaches **10 upvotes (👍)**, it is added to the official rounds and you are credited in the contributors list.
3. **Direct edits:** Click the **➕ Add a Flashcard** link at the top of any round file to propose a question via pull request.

### Question Format

Every question follows this structure. Not all fields are required — use **Common Mistake** and **Napkin Math** where they add value.

```markdown
<details>
<summary><b>[LEVEL BADGE]: [Question Title]</b> · <code>topic-tag</code></summary>

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
