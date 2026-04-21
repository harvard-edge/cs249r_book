<div align="center">

# StaffML

### ML Systems Interview Playbook

**9,000+ physics-grounded systems design questions across Cloud, Edge, Mobile & TinyML.**

*You can generate the code, but you cannot prompt your way out of a silicon bottleneck.*

<br>

<a href="https://mlsysbook.ai/staffml/"><img src="https://img.shields.io/badge/%F0%9F%8E%AF_Launch_StaffML-blue?style=for-the-badge&logoColor=white" alt="Launch StaffML" height="36"></a>

<br><br>

<a href="https://github.com/harvard-edge/cs249r_book"><img src="https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=social" alt="GitHub Stars"></a>
<a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/staffml-publish-live.yml"><img src="https://github.com/harvard-edge/cs249r_book/actions/workflows/staffml-publish-live.yml/badge.svg" alt="StaffML Deploy"></a>
<a href="https://mlsysbook.ai"><img src="https://img.shields.io/badge/part_of-MLSysBook.ai-a31f34" alt="MLSysBook.ai"></a>

</div>

> **Early release (2026)** — StaffML shipped with the **2026** MLSysBook refresh. The vault, apps, and question flows are **actively iterated** as we tune for real interviews—expect meaningful updates to content, UX, and scoring. Share feedback via [GitHub issues](https://github.com/harvard-edge/cs249r_book/issues) or PRs.

---

StaffML is a free, open-source interview prep platform for ML systems engineers. Browse a curated vault of questions organized by competency area, difficulty level (Bloom's Taxonomy L1–L6+), and deployment track. Built by [Prof. Vijay Janapa Reddi](https://github.com/profvjreddi), Harvard University.

| Feature | Description |
|---------|-------------|
| **Vault** | Browse questions by area, topic, and difficulty |
| **Practice** | Drill with spaced repetition and daily challenges |
| **Gauntlet** | Timed mock interview sessions with self-assessment |
| **Progress** | Track coverage across competency areas and tracks |
| **Chains** | Deepening sequences from L1 Recall to L6+ Architect |

> If StaffML helps your prep, **[give us a star](https://github.com/harvard-edge/cs249r_book)** — it helps others find this resource.

**Data:** [`vault/corpus.json`](vault/corpus.json) · [`vault/taxonomy.json`](vault/taxonomy.json) · **App source:** [`staffml/`](staffml/)

---

## Deployment Tracks

Each track targets a different deployment regime — different physics, different constraints, different interview questions.

| Track | Focus | Primary Constraint |
|-------|-------|-------------------|
| ☁️ **Cloud** | Data center training & serving | Memory bandwidth / network |
| 🤖 **Edge** | Autonomous vehicles, robotics | Thermal envelope / real-time |
| 📱 **Mobile** | On-device AI for smartphones | Battery life / shared resources |
| 🔬 **TinyML** | Microcontroller & ultra-low-power | SRAM capacity / hard real-time |

---

## Mastery Levels

Every question is tagged with a mastery level mapped to [Bloom's taxonomy](https://en.wikipedia.org/wiki/Bloom%27s_taxonomy):

| Level | Name | Scope | What the interviewer hears |
|-------|------|-------|---------------------------|
| 🔵 **L1** | Recall | Own a task | "HBM is 300x slower than L1 cache." |
| 🟢 **L2** | Understand | Own a task | "The Roofline model relates compute to memory bandwidth." |
| 🟡 **L3** | Apply | Own a component | "This workload is memory-bound because its arithmetic intensity is below the ridge point." |
| 🟠 **L4** | Analyze | Own a system | "Switching from A100 to H100 won't help because the ridge point shifts." |
| 🔴 **L5** | Evaluate | Own the architecture | "Let me derive the optimal parallelism from the NVLink topology." |
| 🟣 **L6+** | Architect | Own the org | "Here's a fault-tolerant training architecture for 1T params across 3 data centers." |

---

## Depth Chains

Questions are organized into **chains** — sequences that deepen understanding of a single topic from recall to architecture. Each chain walks you through the Bloom levels, building on the previous question.

**Example: GPU Memory Hierarchy Chain (6 questions)**

| Step | Level | Question |
|------|-------|----------|
| 1 | 🔵 L1 | The HBM vs L1 Latency Gap |
| 2 | 🟢 L2 | The FP16 Model Footprint |
| 3 | 🟡 L3 | KV Cache Memory for 7B Model Serving |
| 4 | 🟠 L4 | OOM at Step 500 but Not Step 1 |
| 5 | 🔴 L5 | CPU Offloading vs Activation Recomputation |
| 6 | 🟣 L6+ | Memory Budget for High-Concurrency LLM Serving |

> The vault contains **1,000+ chains** across all tracks. In the app, chains appear after you answer a question — click "Next in chain" to go deeper.

---

## Vault Stats

| Metric | Count |
|--------|-------|
| Questions | 9,000+ |
| Chains | 1,000+ |
| Taxonomy Concepts | 650+ |
| Competency Areas | 12 |
| Deployment Tracks | 4 + Global |
| Mastery Levels | L1–L6+ |

---

## Sample Questions

A taste of what's inside. Click any question to reveal the model answer with napkin math.

### ☁️ Cloud

<details>
<summary><b>🟢 L2 &nbsp; Physical Limits on Training Cluster Scale</b></summary>
<blockquote>Explain why you cannot simply double the number of GPUs indefinitely to halve training time, and identify the three physical ceilings that bound cluster scaling.</blockquote>

Three physical ceilings prevent infinite scaling: (1) **Communication bottleneck** — synchronous training requires AllReduce to average gradients across all GPUs every step. With N GPUs, AllReduce latency grows as O(log N) per step. At 10,000+ GPUs, communication time can exceed computation time. (2) **Power and cooling** — each GPU draws 300–700W. A 10K GPU cluster requires 4+ MW just for GPUs. (3) **Critical batch size** — beyond the critical batch size, gradient noise diminishes returns. For GPT-3, this is ~3.2M tokens.

```
10K GPUs × 400W = 4MW
AllReduce at 10K nodes: ~10ms overhead vs ~50ms compute = 17% communication tax
Critical batch size for GPT-3: ~3.2M tokens
```
</details>

<details>
<summary><b>🟠 L4 &nbsp; The Half-Baked Speedup</b></summary>
<blockquote>You converted most of your LLM to BF16 but only see 1.4x speedup instead of the expected 2x. What is happening?</blockquote>

Training involves a mix of compute-bound and memory-bound operations — only some benefit from BF16. Large GEMMs (attention, FFN) see ~2x speedup. But optimizer steps (Adam maintains FP32 master weights), normalization layers, and loss computation remain in FP32. The weighted average: 70% of time in BF16-accelerated ops × 2x + 30% in FP32 ops × 1x = 1.4x overall.

```
Forward GEMMs: 40% of time → BF16 → 2x speedup
Backward GEMMs: 30% of time → BF16 → 2x speedup
Optimizer (Adam): 15% of time → FP32 → 1x
Other (norm, loss): 15% of time → FP32 → 1x
Weighted: 0.7 × 2 + 0.3 × 1 = 1.7... but memory-bound ops don't see full 2x → ~1.4x
```
</details>

<details>
<summary><b>🟣 L6+ &nbsp; The Exploding Data Lake Bill</b></summary>
<blockquote>Your data lake on S3 has grown to 500 PB. Design a tiering strategy to cut the monthly storage bill by 60%+.</blockquote>

Intelligent data tiering based on access frequency. Classify data, apply lifecycle policies: hot data (30%) stays in S3 Standard, warm data in S3 Standard-IA, cold data (70%) moves to Glacier Deep Archive.

```
S3 Standard: $0.023/GB/month
S3 Glacier Deep Archive: $0.00099/GB/month
Current: $0.023 × 500 PB = $11.5M/month
Optimized: 150 PB × $0.023 + 350 PB × $0.00099 ≈ $3.8M/month
Savings: ~67%
```
</details>

### 🤖 Edge

<details>
<summary><b>🔵 L1 &nbsp; The Fleet's Cellular Bill</b></summary>
<blockquote>You have 1M autonomous vehicles. Compare the daily data cost of centralized retraining (10 MB upload/vehicle) vs. federated learning (50 MB gradient upload, 10% participation).</blockquote>

Centralized: 1M × 10 MB = 10 TB/day. Federated (10% participate): 100K × 50 MB = 5 TB/day. At $2/GB cellular cost: centralized = $20,000/day, federated = $10,000/day. Annual savings: $3.65M. But federated also avoids regulatory risk of centralizing raw sensor data.

```
Centralized: 1M × 10 MB = 10 TB/day × $2/GB = $20,000/day
Federated:  100K × 50 MB = 5 TB/day × $2/GB = $10,000/day
Annual savings: $3.65M + regulatory risk reduction
```
</details>

<details>
<summary><b>🟠 L4 &nbsp; The Phantom Sensor Attack</b></summary>
<blockquote>Your autonomous vehicle uses GPS, IMU, and wheel encoder. An attacker spoofs GPS signals. How do you detect and mitigate this?</blockquote>

Multi-layered defense using sensor fusion consistency checks. The IMU and wheel encoder provide *relative* motion — if GPS reports a 50m jump in 1 second while the IMU shows 0.5m movement, the innovation (residual) is 49.5m, far exceeding normal GPS noise (~3m). The state estimator (EKF/UKF) should reject GPS measurements with innovations exceeding a threshold, fall back to dead reckoning, and alert the operator.

```
IMU drift: ~1m/minute without GPS correction
GPS accuracy: 1-3m
Spoof detection threshold: innovation > 5× expected noise = 15m
At 50m jump vs 0.5m IMU: innovation = 49.5m → reject with 99.99% confidence
```
</details>

### 📱 Mobile

<details>
<summary><b>🟢 L2 &nbsp; Background Inference Limits</b></summary>
<blockquote>You want to run an LLM to summarize audio while your iOS app is in the background. What is the primary risk?</blockquote>

The iOS Watchdog Timer. iOS aggressively monitors background apps for memory and CPU usage. Background execution limits: 30 seconds for most tasks, 3 minutes for audio processing. A 7B LLM at INT4 = 3.5 GB weights + 0.5 GB KV-cache = 4 GB. iPhone 16 Pro has 8 GB total, ~5 GB available. In foreground: fits. In background: iOS reclaims memory aggressively, and sustained 3W inference drains 20% battery per hour.

```
On-device LLM: ~3W sustained on A17 Pro
Battery: 4,000 mAh × 3.7V = 14.8 Wh
Drain at 3W: 20% per hour
iOS background limit: ~30 seconds → 0.025 Wh per cycle
```
</details>

<details>
<summary><b>🟡 L3 &nbsp; The Trivial Model Paradox</b></summary>
<blockquote>A single 100-neuron dense layer runs <i>faster</i> on CPU than NPU. Why?</blockquote>

NPUs have significant startup and data transfer overheads that overshadow benefits for tiny models. Driver initialization (~100μs), data transfer to NPU memory (~20μs), and NPU compute (~5μs) total ~125μs. The CPU does the same computation in ~50μs with no transfer overhead. The crossover point is typically around 10K parameters — below that, CPU wins.

```
CPU: 50μs compute
NPU: 100μs startup + 20μs transfer + 5μs compute = 125μs
NPU is 2.5× slower for trivial models
Crossover: ~10K parameters
```
</details>

### 🔬 TinyML

<details>
<summary><b>🟢 L2 &nbsp; Microcontroller Arithmetic Intensity</b></summary>
<blockquote>Calculate the Ridge Point for a Cortex-M4 microcontroller. Is it compute-bound or memory-bound?</blockquote>

Ridge Point = Peak Compute / Peak Memory Bandwidth. Cortex-M4 at 168 MHz: ~168 MFLOPS (1 FP op/cycle). Memory: 32-bit bus at 168 MHz = 672 MB/s. Ridge Point = 0.168 GFLOPS / 0.672 GB/s = 0.25 FLOPS/byte. Most neural network layers have arithmetic intensity of 10-100 — far above the ridge point. MCUs are almost always **compute-bound**, the opposite of GPUs.

```
Cortex-M4: 168 MFLOPS / 672 MB/s = 0.25 FLOPS/byte
Conv2D AI: ~50 FLOPS/byte → compute-bound
GPU (H100): 989 TFLOPS / 3.35 TB/s = 295 FLOPS/byte → memory-bound
MCUs are the mirror image of GPUs on the roofline
```
</details>

<details>
<summary><b>🟣 L6+ &nbsp; The Ghost in the Dashboard</b></summary>
<blockquote>100,000 vehicles with Cortex-M4 voice assistants. After a year, humid-climate devices activate randomly. Design an OTA fix within 20% free Flash/SRAM.</blockquote>

Three components within the resource budget: (1) Lightweight drift detector — running mean/variance on audio energy, 12 bytes SRAM. (2) Circuit breaker — if drift exceeds threshold, suppress activations and log diagnostics. (3) Diagnostic reporter — 16-bin histogram per event, store 100 records in Flash. Total: ~10KB Flash (5% of budget), <1KB SRAM.

```
Free Flash: 20% of 1MB = 205KB
Free SRAM: 20% of 256KB = 51KB
Drift detector: 12B SRAM, 2KB Flash
Circuit breaker: 32B Flash
Diagnostics: 76B × 100 records = 7.6KB Flash
Total: ~10KB Flash, <1KB SRAM — well within budget
```
</details>

### 🌐 Global

<details>
<summary><b>🟢 L2 &nbsp; InfiniBand vs Ethernet for Training</b></summary>
<blockquote>Why do large-scale LLM training clusters prefer InfiniBand over Ethernet?</blockquote>

Three properties beyond raw bandwidth: (1) RDMA — GPU memory read/written directly over the network, bypassing CPU and OS kernel. Latency drops from ~50μs (TCP/IP) to ~1-2μs. (2) Lossless fabric — credit-based flow control guarantees zero packet loss, critical for AllReduce correctness. (3) Adaptive routing — hardware-level load balancing across multiple paths reduces congestion.

```
AllReduce for 1GB gradient buffer:
InfiniBand RDMA: 1GB/(50 GB/s) + 2μs × log₂(1024) = ~20ms
Ethernet TCP: 1GB/(50 GB/s) + 50μs × log₂(1024) + retransmit risk = 30-150ms
InfiniBand: 2-7× lower tail latency
```
</details>

<details>
<summary><b>🟠 L4 &nbsp; Mysterious 15% Throughput Drop at Noon</b></summary>
<blockquote>Your 64-GPU cluster shows 15% lower throughput between 11 AM and 3 PM. GPU utilization stays at 98%. No other jobs running. What is happening?</blockquote>

Thermal throttling. The data center's cooling struggles during peak afternoon heat. When GPU junction temperature exceeds 83°C (A100 throttle point), the GPU reduces clock frequency. Clock drops from 1410 MHz to 1200 MHz = exactly 15% reduction. The GPU reports 98% utilization because it's still busy — just at a lower clock.

```
GPU clock: 1410 MHz → 1200 MHz = 15% reduction
Night: junction 75°C, 8°C below throttle
Afternoon: ambient +8°C → junction hits 83°C → throttle
Fix: lower power limit from 400W to 350W → 5°C drop → no throttle
Net result: +3% vs current daytime (lose 12.5% power but gain back 15% clock)
```
</details>

<details>
<summary><b>🟣 L6+ &nbsp; The Agentic Memory Architecture</b></summary>
<blockquote>Design a memory system for a coding agent that maintains context across a multi-hour session with 500K tokens of history.</blockquote>

Three-tier memory: (1) Working memory (<8K tokens) — current file, last 2-3 tool results, current plan. Managed programmatically, not by the LLM. (2) Episodic memory (vector DB) — summarized past interactions, indexed by embedding. Retrieved via semantic search when relevant. (3) Persistent memory (key-value store) — facts, decisions, file states. Never evicted, always available.

```
Raw: 500K tokens × $0.003/1K = $1.50/turn (and wouldn't fit in context)
Tiered: 8K tokens/turn × $0.003/1K = $0.024/turn
Compression ratio: 62.5×
Cost reduction: 98.4%
```
</details>

---

<p align="center">
  <b>These are 15 of 9,000+ questions.</b><br>
  <a href="staffml/">Explore the full vault →</a>
</p>

---

## Development

```bash
# Run the StaffML app locally
cd interviews/staffml
npm install
npm run dev         # → http://localhost:3000

# Regenerate vault manifest after corpus updates
python3 scripts/generate-manifest.py
```

**CI/CD:** Pushes to `dev` auto-build and deploy via [GitHub Actions](https://github.com/harvard-edge/cs249r_book/actions/workflows/staffml-preview-dev.yml).

---

## Contributors

Thanks to these wonderful people who have helped build StaffML!

**Legend:** 🪲 Bug Hunter · ⚡ Code Warrior · 📚 Documentation Hero · 🎨 Design Artist · 🧠 Idea Generator · 🔎 Code Reviewer · 🧪 Test Engineer · 🛠️ Tool Builder

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=80" width="80px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🎨 ✍️ 🧠</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

**Recognize a contributor:** Comment on any issue or PR:
```
@all-contributors please add @username for code, doc, ideas, or design
```

---

<p align="center">
  <i>Wishing you all the best in your interviews and your engineering journey.</i><br>
  — <b>Vijay Janapa Reddi</b>
</p>
