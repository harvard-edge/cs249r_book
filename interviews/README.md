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

<table>
  <thead>
    <tr>
      <th width="35%">Relationship</th>
      <th width="25%">Ratio</th>
      <th width="40%">Why it's stable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>DRAM access vs FP16 compute</b></td>
      <td><b>~580×</b> more energy</td>
      <td>Wire capacitance scales with distance</td>
    </tr>
    <tr>
      <td><b>FP32 vs INT8 energy</b></td>
      <td>~18×</td>
      <td>Bit width determines switching energy</td>
    </tr>
    <tr>
      <td><b>FP32 vs FP16 energy</b></td>
      <td>~3.4×</td>
      <td>Halving bits roughly halves energy</td>
    </tr>
    <tr>
      <td><b>HBM vs L1 cache latency</b></td>
      <td>~300× slower</td>
      <td>On-chip vs off-chip</td>
    </tr>
    <tr>
      <td><b>SSD vs L1 cache latency</b></td>
      <td>~100,000× slower</td>
      <td>Electrical vs flash</td>
    </tr>
    <tr>
      <td><b>Network vs local memory</b></td>
      <td>~17× slower</td>
      <td>Speed of light + switching</td>
    </tr>
    <tr>
      <td><b>Light in fiber</b></td>
      <td>~200 km/ms</td>
      <td>Cross-country US ≈ 40ms RTT</td>
    </tr>
  </tbody>
</table>

### Scaling Rules (Arithmetic — Hardware Independent)

These formulas let you estimate memory, compute, and cache requirements from a model's parameter count. "7B" means a model with 7 billion parameters.

<table>
  <thead>
    <tr>
      <th width="35%">What you're estimating</th>
      <th width="35%">Formula</th>
      <th width="30%">Example</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Inference memory (FP16)</b></td>
      <td>2 bytes × params</td>
      <td>7B params × 2 bytes = <b>14 GB</b></td>
    </tr>
    <tr>
      <td><b>Inference memory (INT8)</b></td>
      <td>1 byte × params</td>
      <td>7B params × 1 byte = <b>7 GB</b></td>
    </tr>
    <tr>
      <td><b>Training memory (Adam, FP16+FP32)</b></td>
      <td><b>16 bytes × params</b></td>
      <td>7B params × 16 bytes = <b>112 GB</b></td>
    </tr>
    <tr>
      <td><b>Inference compute (transformer)</b></td>
      <td>~2 FLOPs × params per token</td>
      <td>7B → ~<b>14 GFLOPs/token</b></td>
    </tr>
    <tr>
      <td><b>Training compute</b></td>
      <td>~6 FLOPs × params × tokens</td>
      <td>7B on 1T tokens → <b>4×10²² FLOPs</b></td>
    </tr>
    <tr>
      <td><b>KV-cache per token (all layers)</b></td>
      <td>2 × layers × heads × head_dim × 2 bytes</td>
      <td>Llama 70B, 128k tokens → <b>~335 GB</b></td>
    </tr>
  </tbody>
</table>

### Current Hardware Snapshot (2024–2025)

<table>
  <thead>
    <tr>
      <th width="20%">Category</th>
      <th width="35%">Metric</th>
      <th width="45%">Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Compute</b></td>
      <td>A100 FP16 Tensor Core</td>
      <td>312 TFLOPS</td>
    </tr>
    <tr>
      <td></td>
      <td>H100 FP16 Tensor Core</td>
      <td>989 TFLOPS</td>
    </tr>
    <tr>
      <td></td>
      <td>H100 FP8 Tensor Core</td>
      <td>1,979 TFLOPS</td>
    </tr>
    <tr>
      <td></td>
      <td>B200 FP16 Tensor Core</td>
      <td>2,250 TFLOPS</td>
    </tr>
    <tr>
      <td><b>Memory BW</b></td>
      <td>A100 HBM2e</td>
      <td>2.0 TB/s</td>
    </tr>
    <tr>
      <td></td>
      <td>H100 HBM3</td>
      <td>3.35 TB/s</td>
    </tr>
    <tr>
      <td></td>
      <td>B200 HBM3e</td>
      <td>8.0 TB/s</td>
    </tr>
    <tr>
      <td><b>Interconnect</b></td>
      <td>NVLink 4.0 (H100)</td>
      <td>900 GB/s</td>
    </tr>
    <tr>
      <td></td>
      <td>NVLink 5.0 (B200)</td>
      <td>1,800 GB/s</td>
    </tr>
    <tr>
      <td></td>
      <td>InfiniBand NDR</td>
      <td>400 Gbps (50 GB/s)</td>
    </tr>
    <tr>
      <td></td>
      <td>PCIe Gen5 x16</td>
      <td>64 GB/s</td>
    </tr>
    <tr>
      <td><b>Roofline Ridge</b></td>
      <td>A100 (FP16)</td>
      <td>~153 Ops/Byte</td>
    </tr>
    <tr>
      <td></td>
      <td>H100 (FP16)</td>
      <td>~295 Ops/Byte</td>
    </tr>
    <tr>
      <td><b>Power</b></td>
      <td>A100 TDP</td>
      <td>400 W</td>
    </tr>
    <tr>
      <td></td>
      <td>H100 TDP</td>
      <td>700 W</td>
    </tr>
    <tr>
      <td></td>
      <td>B200 TDP</td>
      <td>1,000 W</td>
    </tr>
    <tr>
      <td><b>Latency</b></td>
      <td>L1 / Register</td>
      <td>~1 ns</td>
    </tr>
    <tr>
      <td></td>
      <td>L2 Cache</td>
      <td>~4 ns</td>
    </tr>
    <tr>
      <td></td>
      <td>HBM3</td>
      <td>~300 ns</td>
    </tr>
    <tr>
      <td></td>
      <td>PCIe Gen5</td>
      <td>~1 μs</td>
    </tr>
    <tr>
      <td></td>
      <td>InfiniBand</td>
      <td>~5 μs</td>
    </tr>
    <tr>
      <td></td>
      <td>NVMe SSD</td>
      <td>~100 μs</td>
    </tr>
  </tbody>
</table>

> **Source:** All values from the textbook's `constants.py`. When hardware generations change, update the constants file — every calculation in the book (and this playbook) updates automatically.

---

## Quick Start

Pick your level and start drilling:

<table>
  <thead>
    <tr>
      <th width="35%">You are...</th>
      <th width="65%">Start here</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Preparing for a screen</b> (Junior/Mid)</td>
      <td>🟢 Green-tagged questions in any round</td>
    </tr>
    <tr>
      <td><b>Building applied skills</b> (Mid)</td>
      <td>🔵 Blue-tagged questions — diagnose real systems</td>
    </tr>
    <tr>
      <td><b>Targeting Senior (L5)</b></td>
      <td>🟡 Yellow-tagged questions + [Round 1](cloud/01_Single_Node_Physics.md) & [Round 3](cloud/03_Production_Serving.md)</td>
    </tr>
    <tr>
      <td><b>Targeting Staff+ (L6+)</b></td>
      <td>🔴 Red-tagged questions + [Round 5: Visual Debugging](cloud/05_Visual_Architecture_Debugging.md)</td>
    </tr>
    <tr>
      <td><b>Mock interview practice</b></td>
      <td>[The Architect's Rubric](00_The_Architects_Rubric.md) — grade your own designs</td>
    </tr>
  </tbody>
</table>

---

## Choose Your Track

Each track targets a different deployment regime. Pick the one that matches the roles you're interviewing for — or study multiple tracks to build breadth.

<table>
  <thead>
    <tr>
      <th width="20%">Track</th>
      <th width="35%">Focus</th>
      <th width="15%">Questions</th>
      <th width="30%">Rounds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b><a href="cloud/README.md">☁️ Cloud</a></b></td>
      <td>Data center training & serving at scale</td>
      <td>57</td>
      <td>6 rounds + visual debugging</td>
    </tr>
    <tr>
      <td><b><a href="edge/README.md">🤖 Edge</a></b></td>
      <td>Autonomous vehicles, robotics, industrial AI</td>
      <td>27</td>
      <td>2 rounds</td>
    </tr>
    <tr>
      <td><b><a href="mobile/README.md">📱 Mobile</a></b></td>
      <td>On-device AI for smartphones</td>
      <td>27</td>
      <td>2 rounds</td>
    </tr>
    <tr>
      <td><b><a href="tinyml/README.md">🔬 TinyML</a></b></td>
      <td>Microcontroller & ultra-low-power AI</td>
      <td>27</td>
      <td>2 rounds</td>
    </tr>
    <tr>
      <td><b><a href="00_The_Architects_Rubric.md">📋 Rubric</a></b></td>
      <td>Self-evaluation across 6 engineering axes</td>
      <td>—</td>
      <td>Shared</td>
    </tr>
    <tr>
      <td><b><a href="TOPIC_MAP.md">🗺️ Topic Map</a></b></td>
      <td>Master plan — competency areas, gaps, and targets</td>
      <td>—</td>
      <td>Planning doc</td>
    </tr>
  </tbody>
</table>

---

## Mastery Levels

Every question is tagged with a mastery level. These levels are modeled on engineering ladders at major tech companies (Google, Meta, etc.) but represent **cognitive thresholds** — each level tests a fundamentally different kind of thinking, mapped to [Bloom's taxonomy](https://en.wikipedia.org/wiki/Bloom%27s_taxonomy) and the **scope of ownership** expected at that career stage.

### The Framework

<table>
  <thead>
    <tr>
      <th width="15%">Level</th>
      <th width="15%">Scope</th>
      <th width="20%">Cognitive Skill</th>
      <th width="50%">What the interviewer hears</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🟢 <b>L3 — The Screen</b></td>
      <td>Own a <b>task</b></td>
      <td><b>Recall & Define</b></td>
      <td>"The Roofline model relates compute to memory bandwidth."</td>
    </tr>
    <tr>
      <td>🔵 <b>L4 — The Practitioner</b></td>
      <td>Own a <b>component</b></td>
      <td><b>Apply & Identify</b></td>
      <td>"This workload is memory-bound because its arithmetic intensity is below the ridge point."</td>
    </tr>
    <tr>
      <td>🟡 <b>L5 — The Architect</b></td>
      <td>Own a <b>system</b></td>
      <td><b>Analyze & Predict</b></td>
      <td>"Switching from A100 to H100 won't help because the ridge point shifts right while our intensity stays at ~1."</td>
    </tr>
    <tr>
      <td>🔴 <b>L6+ — The Lead</b></td>
      <td>Own the <b>architecture</b></td>
      <td><b>Synthesize & Derive</b></td>
      <td>"Let me derive the optimal parallelism dimensions from the NVLink topology, memory capacity, and pipeline bubble cost."</td>
    </tr>
  </tbody>
</table>

### The Transitions

- **L3→L4:** You stop reciting and start diagnosing. You can look at a system and correctly classify what's happening — identify the bottleneck, name the failure mode, apply the right formula.
- **L4→L5:** You stop diagnosing and start predicting. You can reason about what happens when a constraint changes — a hardware upgrade, a traffic spike, a precision change — and explain *why* the system behaves differently.
- **L5→L6+:** You stop predicting known patterns and start deriving novel solutions from first principles. You can stand at a whiteboard with incomplete information and work backward from physics to architecture.

### How This Maps to Industry

The levels correspond to real expectations at major tech companies. The titles differ, but the cognitive bar is consistent.

<table>
  <thead>
    <tr>
      <th width="10%">Level</th>
      <th width="15%">Google</th>
      <th width="15%">Meta</th>
      <th width="15%">Amazon</th>
      <th width="45%">What systems interviews test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>L3</b></td>
      <td>L3 (SWE II)</td>
      <td>E3 (IC3)</td>
      <td>SDE I</td>
      <td>Can you define the concepts? Do you know the vocabulary of ML systems?</td>
    </tr>
    <tr>
      <td><b>L4</b></td>
      <td>L4 (SWE III)</td>
      <td>E4 (IC4)</td>
      <td>SDE II</td>
      <td>Given a broken system, can you diagnose the root cause?</td>
    </tr>
    <tr>
      <td><b>L5</b></td>
      <td>L5 (Senior)</td>
      <td>E5 (IC5)</td>
      <td>Senior SDE</td>
      <td>Given a working system and a changing constraint, can you predict what breaks?</td>
    </tr>
    <tr>
      <td><b>L6+</b></td>
      <td>L6 (Staff)</td>
      <td>E6 (Staff)</td>
      <td>Principal</td>
      <td>Given a blank whiteboard and a set of requirements, can you derive the architecture from physics?</td>
    </tr>
  </tbody>
</table>

> **For question contributors:** When writing a new question, ask yourself: *"What scope of reasoning does this require?"* If the answer is "name the concept," it's L3. If it's "diagnose this system," it's L4. If it's "predict what happens when X changes," it's L5. If it's "derive the solution from constraints," it's L6+.

---

## Deployment Tracks

The mastery levels tell you *how deeply* a question tests your reasoning. The deployment track tells you *which physics regime* it lives in. The same fundamental principle — **constraints drive architecture** — applies everywhere, but the constraints themselves are radically different depending on where the silicon sits.

Every question in this playbook is tagged with a deployment track. All four tracks now have substantial question banks — see the [Topic Map](TOPIC_MAP.md) for the full coverage matrix and expansion plan.

<table>
  <thead>
    <tr>
      <th width="12%">Track</th>
      <th width="25%">Where it runs</th>
      <th width="22%">Primary constraint</th>
      <th width="12%">Compute scale</th>
      <th width="14%">Memory</th>
      <th width="15%">Power budget</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>☁️ Cloud</b></td>
      <td>Data center (H100, TPU, InfiniBand)</td>
      <td>Memory bandwidth / network</td>
      <td>PFLOPS</td>
      <td>80 GB HBM</td>
      <td>700W per chip</td>
    </tr>
    <tr>
      <td><b>🤖 Edge</b></td>
      <td>Autonomous vehicles, robotics, CCTV, industrial</td>
      <td>Thermal envelope / real-time deadlines</td>
      <td>TOPS</td>
      <td>8–32 GB DRAM</td>
      <td>15–75W per module</td>
    </tr>
    <tr>
      <td><b>📱 Mobile</b></td>
      <td>Smartphones (Snapdragon NPU, Apple ANE)</td>
      <td>Battery life / shared resources</td>
      <td>TOPS (shared)</td>
      <td>6–12 GB shared</td>
      <td>3–5W total device</td>
    </tr>
    <tr>
      <td><b>🔬 TinyML</b></td>
      <td>Microcontrollers (Cortex-M, RISC-V)</td>
      <td>SRAM capacity / hard real-time</td>
      <td>MFLOPS</td>
      <td>256 KB–2 MB SRAM</td>
      <td>1–100 mW</td>
    </tr>
  </tbody>
</table>

### How the Same Topic Changes Across Tracks

The physics is universal. The numbers are not.

<table>
  <thead>
    <tr>
      <th width="18%">Topic</th>
      <th width="22%">☁️ Cloud</th>
      <th width="22%">🤖 Edge</th>
      <th width="20%">📱 Mobile</th>
      <th width="18%">🔬 TinyML</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Roofline</b></td>
      <td>H100 ridge point ~295 Ops/Byte</td>
      <td>Jetson Orin ridge point ~15 Ops/Byte</td>
      <td>NPU ridge point varies by SoC</td>
      <td>No FPU — integer-only roofline</td>
    </tr>
    <tr>
      <td><b>Memory</b></td>
      <td>KV-cache fragmentation in 80 GB HBM</td>
      <td>Model + sensor pipeline in 32 GB DRAM</td>
      <td>Model coexists with OS in shared RAM</td>
      <td>Entire model must fit in on-chip SRAM</td>
    </tr>
    <tr>
      <td><b>Quantization</b></td>
      <td>FP16→INT8 for throughput</td>
      <td>INT8→INT4 for thermal headroom</td>
      <td>INT8 for NPU compatibility</td>
      <td>INT8→binary to fit on chip</td>
    </tr>
    <tr>
      <td><b>Serving</b></td>
      <td>Continuous batching, PagedAttention</td>
      <td>Hard real-time inference deadlines</td>
      <td>On-device inference, thermal throttling</td>
      <td>Single-shot, microsecond latency</td>
    </tr>
    <tr>
      <td><b>Fault tolerance</b></td>
      <td>Checkpoint 10,000 GPUs (MTBF)</td>
      <td>Graceful degradation, functional safety</td>
      <td>Crash recovery, model fallback</td>
      <td>Watchdog timers, hard real-time guarantees</td>
    </tr>
  </tbody>
</table>

When submitting a question, tag it with both a **mastery level** (L3–L6+) and a **deployment track** (Cloud, Edge, Mobile, TinyML). See each track's README for the specific topics that need questions.

---

## Topic Index

Every question is tagged with a topic. Use this index to study a specific concept across all rounds, or to find where we need more questions.

<table>
  <thead>
    <tr>
      <th width="22%">Topic</th>
      <th width="48%">Questions</th>
      <th width="12%">Rounds</th>
      <th width="18%">Coverage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b><code>roofline</code></b> — Arithmetic Intensity, compute vs memory bound</td>
      <td>The Profiling Crisis · The Roofline Shift · The Amdahl Ceiling · The Decoding Bottleneck</td>
      <td>1, 2, 3</td>
      <td>🟡 Moderate</td>
    </tr>
    <tr>
      <td><b><code>memory</code></b> — VRAM accounting, memory hierarchy, energy</td>
      <td>The Sequence Length Trap · The Energy-Movement Invariant · The OOM Error · Ch.6: KV-Cache · Ch.7: Data Parallelism</td>
      <td>1, 2, 5</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>kv-cache</code></b> — KV-Cache sizing, fragmentation, PagedAttention</td>
      <td>The Sequence Length Trap · The Fragmentation Crisis · The Disaggregated Serving · Ch.3: Serving Stack · Ch.6: KV-Cache</td>
      <td>1, 3, 5</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>precision</code></b> — FP16/BF16/INT8, quantization, underflow</td>
      <td>The Underflow Crisis · The Precision Trade-off</td>
      <td>1</td>
      <td>🟡 Moderate</td>
    </tr>
    <tr>
      <td><b><code>hardware</code></b> — Tensor Cores, sparsity, silicon architecture</td>
      <td>The Precision Trade-off · The Sparsity Fallacy</td>
      <td>1</td>
      <td>🟡 Moderate</td>
    </tr>
    <tr>
      <td><b><code>frameworks</code></b> — JIT compilation, graph tracing, kernels</td>
      <td>The Compilation Overhead</td>
      <td>1</td>
      <td>🔴 Thin — needs expansion</td>
    </tr>
    <tr>
      <td><b><code>data-pipeline</code></b> — CPU starvation, preprocessing, ingestion</td>
      <td>The Data Pipeline Stall · Ch.1: Dataloader</td>
      <td>1, 5</td>
      <td>🔴 Thin — needs expansion</td>
    </tr>
    <tr>
      <td><b><code>parallelism</code></b> — DP, TP, PP, ZeRO, 3D parallelism</td>
      <td>The OOM Error · The Pipeline Bubble · The Amdahl Ceiling · Dimensioning the 3D Cube · The Cross-Rack Stall · Ch.4: Pipeline · Ch.7: Data Parallelism</td>
      <td>2, 5</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>network</code></b> — NVLink, InfiniBand, Fat-Tree, AllReduce</td>
      <td>The Cross-Rack Stall · The Oversubscription Choke · The Ring vs Tree Dilemma · Dimensioning the 3D Cube · Ch.2: Training Cluster</td>
      <td>2, 5</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>fault-tolerance</code></b> — Checkpointing, MTBF, stragglers</td>
      <td>The Straggler Problem · The MTBF Crisis</td>
      <td>2</td>
      <td>🔴 Thin — needs expansion</td>
    </tr>
    <tr>
      <td><b><code>latency</code></b> — TTFT, TPOT, tail latency, queueing theory</td>
      <td>The Serving Inversion · The LLM Metrics · The Black Friday Collapse</td>
      <td>3</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>serving</code></b> — Batching, cold starts, speculative decoding</td>
      <td>The Serverless Freeze · The Pre-computation Trade-off · The Batching Dilemma · The Disaggregated Serving · The Decoding Bottleneck · Ch.3: Serving Stack</td>
      <td>3, 5</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>mlops</code></b> — Drift, skew, deployment, technical debt</td>
      <td>The '95% Problem' · The Silent Failure · The Training-Serving Skew · The Retraining Math · The Deployment Risk · Ch.5: Feature Store</td>
      <td>4, 5</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>economics</code></b> — TCO, retraining cost, sustainability</td>
      <td>The Energy Economics · The Retraining Math</td>
      <td>4</td>
      <td>🔴 Thin — needs expansion</td>
    </tr>
    <tr>
      <td><b><code>security</code></b> — Prompt injection, adversarial attacks</td>
      <td>The Trust Boundary</td>
      <td>4</td>
      <td>🔴 Thin — needs expansion</td>
    </tr>
    <tr>
      <td><b><code>privacy</code></b> — DP-SGD, membership inference</td>
      <td>The Privacy Audit</td>
      <td>4</td>
      <td>🔴 Thin — needs expansion</td>
    </tr>
    <tr>
      <td><b><code>fairness</code></b> — Subgroup evaluation, bias amplification</td>
      <td>The Bias Amplifier</td>
      <td>4</td>
      <td>🔴 Thin — needs expansion</td>
    </tr>
  </tbody>
</table>

> **Want to contribute?** Topics marked 🔴 **Thin** are where we most need new questions. See [Contributing](#contributing) below.

---

## Every Answer Links Back to the Textbook

Each question includes a **📖 Deep Dive** link to the relevant chapter of [Machine Learning Systems](https://mlsysbook.ai). The questions prove the knowledge matters; the textbook teaches it.

<table>
  <thead>
    <tr>
      <th width="25%">Round</th>
      <th width="75%">Textbook Chapters Referenced</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Silicon Physics</b></td>
      <td>HW Acceleration, Data Engineering, Model Compression, Frameworks, Neural Computation</td>
    </tr>
    <tr>
      <td><b>Distributed Infra</b></td>
      <td>Distributed Training, Fault Tolerance, Fleet Orchestration, Network Fabrics</td>
    </tr>
    <tr>
      <td><b>Production Serving</b></td>
      <td>Model Serving, Benchmarking, Inference at Scale</td>
    </tr>
    <tr>
      <td><b>Ops & Economics</b></td>
      <td>ML Operations, Responsible Engineering, Sustainable AI, Security & Privacy, Robust AI</td>
    </tr>
  </tbody>
</table>

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
