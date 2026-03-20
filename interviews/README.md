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
  <b>1,063 systems design questions across Cloud, Edge, Mobile & TinyML tracks.</b><br>
  <i>You can generate the code, but you cannot prompt your way out of a silicon bottleneck.</i>
</p>

<p align="center">
  <a href="cloud/README.md">☁️ Cloud</a> ·
  <a href="edge/README.md">🤖 Edge</a> ·
  <a href="mobile/README.md">📱 Mobile</a> ·
  <a href="tinyml/README.md">🔬 TinyML</a> ·
  <a href="NUMBERS.md">📊 Numbers</a> ·
  <a href="00_The_Architects_Rubric.md">📋 Rubric</a>
</p>

---

## Why This Exists

In the age of GenAI, writing a training loop is trivial. Anyone can ask an LLM for PyTorch syntax. But an LLM cannot fix a fragmented KV-cache, it cannot un-choke a saturated InfiniBand switch, and it cannot cool a melting Edge NPU. **Code is generated; physics is enforced.**

Students often ask me: *"How do I prepare for ML systems interviews?"* This playbook is the answer. These questions test your **Mechanical Sympathy**: the ability to see past the framework abstractions and engineer the metal underneath. You must learn to reason about the physical constraints of keeping 10,000 GPUs fed and 1 million users served. This is exactly what companies like Meta, Google, and OpenAI test for.

This playbook organizes that knowledge into something you can actually study.

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
      <td>🟢 Green-tagged questions in any topic file</td>
    </tr>
    <tr>
      <td><b>Building applied skills</b> (Mid)</td>
      <td>🔵 Blue-tagged questions — diagnose real systems</td>
    </tr>
    <tr>
      <td><b>Targeting Senior (L5)</b></td>
      <td>🟡 Yellow-tagged questions + <a href="cloud/01_single_machine.md">1. Single Machine</a> & <a href="cloud/03_serving_stack.md">3. Serving Stack</a></td>
    </tr>
    <tr>
      <td><b>Targeting Staff+ (L6+)</b></td>
      <td>🔴 Red-tagged questions + <a href="cloud/05_visual_debugging.md">5. Visual Debugging</a></td>
    </tr>
    <tr>
      <td><b>Mock interview practice</b></td>
      <td><a href="00_The_Architects_Rubric.md">The Architect's Rubric</a> — grade your own designs</td>
    </tr>
  </tbody>
</table>

---

## Choose Your Track

Each track targets a different deployment regime — different physics, different constraints, different interview questions. Pick the one that matches the roles you're interviewing for, or study multiple tracks to build breadth.

<table>
  <thead>
    <tr>
      <th width="15%">Track</th>
      <th width="25%">Focus</th>
      <th width="20%">Primary Constraint</th>
      <th width="10%">Questions</th>
      <th width="15%">Topics</th>
      <th width="15%">Scale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b><a href="cloud/README.md">☁️ Cloud</a></b></td>
      <td>Data center training & serving</td>
      <td>Memory bandwidth / network</td>
      <td>296</td>
      <td>5</td>
      <td>PFLOPS, 80 GB HBM</td>
    </tr>
    <tr>
      <td><b><a href="edge/README.md">🤖 Edge</a></b></td>
      <td>Autonomous vehicles, robotics</td>
      <td>Thermal envelope / real-time</td>
      <td>268</td>
      <td>4</td>
      <td>TOPS, 8–32 GB</td>
    </tr>
    <tr>
      <td><b><a href="mobile/README.md">📱 Mobile</a></b></td>
      <td>On-device AI for smartphones</td>
      <td>Battery life / shared resources</td>
      <td>261</td>
      <td>4</td>
      <td>TOPS, 6–12 GB</td>
    </tr>
    <tr>
      <td><b><a href="tinyml/README.md">🔬 TinyML</a></b></td>
      <td>Microcontroller & ultra-low-power</td>
      <td>SRAM capacity / hard real-time</td>
      <td>238</td>
      <td>4</td>
      <td>MFLOPS, 256 KB–2 MB</td>
    </tr>
  </tbody>
</table>

> **📊 [Numbers Every ML Systems Engineer Should Know](NUMBERS.md)** — The physics constants, scaling rules, and hardware specs behind every question in this playbook.

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

---

## Mastery Levels

Every question is tagged with a mastery level. These levels mirror engineering ladders at major tech companies (Google, Meta, etc.) but represent **cognitive thresholds**: each level tests a different kind of reasoning, mapped to [Bloom's taxonomy](https://en.wikipedia.org/wiki/Bloom%27s_taxonomy) and the **scope of ownership** expected at that career stage.

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

- **L3→L4:** You stop reciting and start diagnosing. You can look at a system and correctly classify what's happening: identify the bottleneck, name the failure mode, apply the right formula.
- **L4→L5:** You stop diagnosing and start predicting. You can reason about what happens when a constraint changes (a hardware upgrade, a traffic spike, a precision change) and explain *why* the system behaves differently.
- **L5→L6+:** You stop predicting known patterns and start deriving solutions from first principles. You can stand at a whiteboard with incomplete information and work backward from physics to architecture.

### How This Maps to Industry

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

## Topic Index

Every question is tagged with a topic. Use this index to study a specific concept across all topic files. The examples below highlight key questions from across the tracks.

<table>
  <thead>
    <tr>
      <th width="22%">Topic</th>
      <th width="60%">Example Questions Across Tracks</th>
      <th width="18%">Coverage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b><code>roofline</code></b> — Arithmetic Intensity, compute vs memory bound</td>
      <td><b>Cloud:</b> The Profiling Crisis · <b>Edge:</b> The Bandwidth-Bound Orin · <b>Mobile:</b> The Budget Phone Mystery</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>memory</code></b> — VRAM accounting, memory hierarchy, energy</td>
      <td><b>Cloud:</b> The Sequence Length Trap · <b>Mobile:</b> The App Memory Pressure Levels · <b>TinyML:</b> The Flash-SRAM Boundary</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>kv-cache</code></b> — KV-Cache sizing, fragmentation, PagedAttention</td>
      <td><b>Cloud:</b> The Fragmentation Crisis · <b>Mobile:</b> The Mobile LLM KV-Cache Squeeze</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>precision</code></b> — FP16/BF16/INT8, quantization, underflow</td>
      <td><b>Cloud:</b> The Underflow Crisis · <b>Edge:</b> The QAT Cliff · <b>TinyML:</b> The 100-Layer Quantization Drift</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>hardware</code></b> — Tensor Cores, sparsity, silicon architecture</td>
      <td><b>Cloud:</b> The Sparsity Fallacy · <b>Mobile:</b> The NPU Efficiency Advantage</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>frameworks</code></b> — JIT compilation, graph tracing, kernels</td>
      <td><b>Cloud:</b> The Compilation Overhead · <b>Mobile:</b> The NPU Delegation Failure Modes</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>data-pipeline</code></b> — CPU starvation, preprocessing, ingestion</td>
      <td><b>Cloud:</b> The Data Pipeline Stall · <b>Edge:</b> The Timestamp Drift</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>parallelism</code></b> — DP, TP, PP, ZeRO, 3D parallelism</td>
      <td><b>Cloud:</b> The Pipeline Bubble · The Amdahl Ceiling · Dimensioning the 3D Cube</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>network</code></b> — NVLink, InfiniBand, Fat-Tree, AllReduce</td>
      <td><b>Cloud:</b> The Cross-Rack Stall · The Oversubscription Choke · The Ring vs Tree Dilemma</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>fault-tolerance</code></b> — Checkpointing, MTBF, stragglers</td>
      <td><b>Cloud:</b> The Straggler Problem · <b>Edge:</b> The Degradation Ladder</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>latency</code></b> — TTFT, TPOT, tail latency, queueing theory</td>
      <td><b>Cloud:</b> The Serving Inversion · <b>Mobile:</b> The Jank Budget · <b>TinyML:</b> Interrupt Overhead Impact on Inference</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>serving</code></b> — Batching, cold starts, speculative decoding</td>
      <td><b>Cloud:</b> The Batching Dilemma · <b>Edge:</b> The eMMC Cold Start</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>mlops</code></b> — Drift, skew, deployment, technical debt</td>
      <td><b>Cloud:</b> The Training-Serving Skew · <b>Mobile:</b> The Silent Accuracy Degradation</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>economics</code></b> — TCO, retraining cost, sustainability</td>
      <td><b>Cloud:</b> The Energy Economics · <b>Edge:</b> The Edge vs Cloud Cost Crossover</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>security</code></b> — Prompt injection, adversarial attacks</td>
      <td><b>Cloud:</b> The Guardrail Latency Tax · <b>Edge:</b> The Adversarial Patch Attack</td>
      <td>✅ Strong</td>
    </tr>
    <tr>
      <td><b><code>privacy</code></b> — DP-SGD, membership inference</td>
      <td><b>Cloud:</b> The Privacy Throughput Cliff · <b>Mobile:</b> The Federated Keyboard</td>
      <td>✅ Strong</td>
    </tr>
  </tbody>
</table>

---

## Every Answer Links Back to the Textbook

Each question includes a **📖 Deep Dive** link to the relevant chapter of [Machine Learning Systems](https://mlsysbook.ai). The questions prove the knowledge matters; the textbook teaches it.

<table>
  <thead>
    <tr>
      <th width="25%">Topic Area</th>
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

1. **Pull Request:** Click the **➕ Add a Flashcard** link at the top of any topic file and submit a question using the format below.
2. **Issue:** [Open an issue](https://github.com/harvard-edge/cs249r_book/issues/new) with your question and we'll work with you to shape it.

### Question Format

Every question follows this structure. The **Interviewer** prompt is visible as the question; the answer is hidden behind a "Reveal Answer" fold so readers can quiz themselves. Not all fields are required; use **Common Mistake** and **Napkin Math** where they add value.

```markdown
<details>
<summary><b>[LEVEL BADGE]: [Question Title]</b> · <code>topic-tag</code></summary>

- **Interviewer:** [The scenario or crisis]

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** [What most people say wrong — creates the "aha" moment]

  **Realistic Solution:** [The physics/logic behind the correct answer]

  > **Napkin Math:** [Quick back-of-envelope calculation with real numbers]

  > **Key Equation:** $[The formula to memorize]$

  📖 **Deep Dive:** [Link to the relevant textbook chapter]

  </details>

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
