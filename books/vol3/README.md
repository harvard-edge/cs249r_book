# Volume III: Agentic Machine Learning Systems

*The systems architecture of inference-time compute, autonomous trajectories, and closed-loop control.*

**Author:** Prof. Vijay Janapa Reddi (Harvard University)
**Status:** 🟣 **Early Development / Exploratory Work in Progress** *(Curriculum Architecture & Systems Principles)*

<div align="center" style="background: #faf5ff; border: 1px solid #c084fc; border-radius: 8px; padding: 18px 24px; margin: 20px 0; text-align: left;">
  <div style="font-size: 1.1em; font-weight: bold; color: #6b21a8; margin-bottom: 6px; text-align: center;">
    🟣 Author's Working Note: Durable Principles & Empirical Grounding
  </div>
  <p style="color: #581c87; font-size: 0.95em; line-height: 1.5; margin: 0;">
    <i>"Writing about Agentic AI during a rapid technology cycle carries a serious risk: <b>writing a transitional book that becomes obsolete within eighteen months.</b> Most contemporary writing documents ephemeral prompt tricks or soft anthropomorphic metaphors that obscure real systems trade-offs. We do not pretend to discover immutable 'laws' of a field that is actively in flight. Instead, following the tradition of Saltzer's End-to-End Argument and Lampson's Hints for Computer System Design, this volume is organized around <b>durable systems principles, inescapable physical trade-offs, and empirical architectures</b> observed in frontier production systems (Claude Code, Devin, OpenAI Operator, SGLang)."</i>
    <br><span style="display: block; text-align: right; font-weight: bold; margin-top: 6px;">— Vijay Janapa Reddi</span>
  </p>
</div>

---

## The MLSysBook Tetralogy

To maintain absolute architectural clarity across the series, *Machine Learning Systems* is organized into four distinct, non-overlapping volumes:

| Volume | Title | Core Systems Focus | Scope & Unit of Work | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Vol I** | **Introduction to Machine Learning Systems** | **D · A · M** *(Data, Algorithm, Machine)* | Single-node acceleration, tensor kernels, Roofline models, and baseline serving. Unit: **The Model** | 📗 **Released / In Print** (MIT Press) |
| **Vol II** | **Scaling Machine Learning Systems** | **$\mathbf{C^3}$** *(Compute, Communication, Coordination)* | Distributed supercomputers, 3D parallelism, all-reduce fabrics, and fleet orchestration. Unit: **The Fleet** | 📘 **Preview / Work in Progress** |
| **Vol III** | **Agentic Machine Learning Systems** | **H · S · A · C** *(Horizon, State, Authority, Closure)* | Closed-loop autonomous execution, state DAGs, KV context hierarchies, MCP tools, and Sagas. Unit: **The Trajectory** | 🟣 **In Development / Active Evolution** |
| **Vol IV** | **Physical AI: Machine Learning Systems** | **Physics & Causal Boundaries** | Grounded cyber-physical plants, 1 kHz deterministic control, sensory covariance, and safety shields. Unit: **The Plant / World** | 🌲 **In Development / Work in Progress** |

$$\text{Instruction} \longrightarrow \text{Process} \longrightarrow \text{Request/RPC} \longrightarrow \mathbf{\text{Model Call (Vol I)}} \longrightarrow \mathbf{\text{Cluster Job (Vol II)}} \longrightarrow \mathbf{\text{Managed Trajectory (Vol III)}} \longrightarrow \mathbf{\text{Robotic Loop (Vol IV)}}$$

---

## The Core Systems Thesis of Volume III

Volumes I and II optimize the generation of tokens for single, stateless requests ($Tokens_{\text{out}} = \text{Model}(Tokens_{\text{in}})$).

Volume III addresses what happens when that compute is spent in a stateful, closed-loop process over time: **The system ceases to be an oracle behind glass and becomes an autonomous actor.** The atomic unit of systems engineering stops being the isolated request and becomes the **Managed Trajectory**—the multi-step lifecycle from delegated goal to verified outcome.

### The Core Systems Tensions of Agency
Because autoregressive reasoning steps are conditional and history-corrupting ($P(F_i \mid F_{i-1}) \gg P(F_i)$), trajectory execution faces two fundamental tensions:

1. **The Horizon–Reliability Tension:**
   $$P_{\text{success}} \le \prod_{i=1}^{N} P(\text{Success}_i \mid \text{History}_{i-1}) \le p^{N}$$
   As autonomous horizons ($N$) expand, compounding error degrades reliability exponentially. Scaling base model size yields diminishing returns because the horizon exponent $N$ dominates the base accuracy $p$. Reliability must be engineered at runtime through verification checkpoints and test-time search.

2. **The Context Accumulation Tension:**
   $$T_{\text{traj}} = \sum_{i=1}^{N} \Big( T_{\text{prefill}}(C_i) + T_{\text{decode}}(o_i) + T_{\text{tool},i} + T_{\text{verify},i} \Big)$$
   Context accumulates linearly with history, while self-attention scales quadratically ($O(C_i^2)$) and decode latency is memory-bandwidth bound. Memory management shifts from prompt formatting to active cache hierarchy engineering (eviction, compaction, and prefix caching).

---

## Epistemological Foundation: How Do We Know What to Teach?

How do we know we are teaching foundational systems engineering rather than temporary artifacts of the current model generation? We subject every topic to **four falsification tests** and ground it in **verifiable production reality**.

```
                             THE FOUR FALSIFICATION TESTS

 ┌──────────────────────────────────────┐      ┌──────────────────────────────────────┐
 │ 1. The Model Doubling Test           │      │ 2. The Physical Invariant Test       │
 │    • Does the issue disappear or     │      │    • Is it bound by silicon physics, │
 │      intensify if models get 2x      │      │      memory bandwidth, or network    │
 │      smarter tomorrow?               │      │      latency disparities?            │
 └──────────────────┬───────────────────┘      └──────────────────┬───────────────────┘
                    │                                             │
                    ▼                                             ▼
 ┌──────────────────────────────────────┐      ┌──────────────────────────────────────┐
 │ 3. The Broken Analogy Test           │      │ 4. The Empirical Triangulation Test  │
 │    • Where does the 50-year-old OS   │      │    • Does it match what frontier     │
 │      principle break in neural MLSys?│      │      labs ship and top syllabi need? │
 └──────────────────┴───────────────────┘      └──────────────────┴───────────────────┘
```

### Test 1: The Frontier Model Doubling Test
* *What Disappears:* Prompt engineering templates, few-shot coaxes, and anthropomorphic personas. As base models become more capable, brittle prompt scaffolding becomes obsolete.
* *What Intensifies:*
  * **The Reversibility Boundary:** When an agent is capable enough to be trusted with broad tasks, it touches more real-world external systems (filesystems, databases, cloud APIs, payments). The danger of irreversible side-effects grows with model competence.
  * **Compounding Uncertainty ($p^N$):** As single-step accuracy $p$ increases from $0.90$ to $0.98$, users don't stop at 10 steps; they push the horizon to 200 or 500 steps ($0.98^{200} \approx 1.7\%$). Horizon reach will always outpace base accuracy.
  * **Prompt Injection ($W \oplus X$ failure):** Smarter models still parse instructions and untrusted data from the exact same token stream. Greater intelligence does not create hardware privilege rings.
  * **KV-Cache Swapping:** Larger models lock more High Bandwidth Memory (HBM). Swapping idle contexts over PCIe/CXL during multi-second tool round-trips becomes an economic imperative.

### Test 2: The Physical & Invariant Test
Every principle taught is bound by hardware physics and probability:
* **Attention Memory Bandwidth:** Autoregressive decode is memory-bandwidth bound (TB/s). Every token decoded streams every parameter byte and accumulated KV cache through accelerator compute cores.
* **The Asymmetric Latency Wall ($10^{-2}\text{ s}$ vs. $10^3\text{ s}$):** Silicon decodes tokens in milliseconds; humans review actions in minutes or hours. Human-in-the-loop is not a UI button; it is a queueing and state suspension object that forces memory eviction.

### Test 3: The "Where Systems Analogies Break" Test
Following Peterson & Davie: *an analogy that holds teaches vocabulary; an analogy that breaks at a named boundary teaches systems judgment.*

| Classical Systems Concept | Where It Holds in Agentic Systems | Where the Analogy Fractures (The Teachable Edge) |
| :--- | :--- | :--- |
| **Process Control Block (PCB)** | Descriptor tracking principal identity, resource quotas, and capability leases. | **No bit-exact `memcpy`:** An agent's belief state cannot be snapshotted as static registers; it is non-stationary and distributed across attention activations and external environments. |
| **Hardware Privilege Rings (Rings 0–3)** | Hierarchical boundaries between untrusted code and privileged resources. | **Zero privilege rings in MCP:** The Model Context Protocol is an application ABI wire format, NOT a security boundary. Conflating schema validation with isolation causes privilege escalation. |
| **Database Sagas (Garcia-Molina)** | Sequenced transactions with compensating actions without distributed locks. | **Real-world actions often have NO inverse ($A^{-1}$):** A sent email or dropped cloud table cannot be rolled back. Stochastic LLMs cannot generate reliable rollbacks; recovery requires deterministic compensators. |
| **Virtual Memory Paging** | Tiering active working sets between fast memory and slower backing stores. | **No synchronous page faults:** GPU decode warps cannot stall waiting on demand paging; agent serving requires *predictive prefetching* keyed to external tool completions. |

### Test 4: The Empirical Triangulation Test (Grounding in Concrete Production Evidence)
We know these are the right things to teach because they reflect the exact architectural choices shipped by frontier systems:

```
Frontier Production System   Concrete Architectural Mechanism Shipped in Production
──────────────────────────────────────────────────────────────────────────────────────────
Claude Code (Anthropic) ───► • Spawns isolated, read-only subagents for exploration to protect
                              the main context from pollution (Context Working Set).
                            • Pauses and prompts for human approval before destructive actions
                              like `git push` or file deletion (The Reversibility Boundary).
                            • Open-sourced Model Context Protocol (MCP) as a typed JSON-RPC ABI.
                            • Prompt caching with 5-minute TTL to exploit prefix locality.

OpenAI (Operator/Swarm) ───► • Replaced open-ended swarms with structured handoffs & routine graphs.
                            • Automatic 128-token prefix caching at 50% discount.
                            • Human confirmation gates before irreversible external actions.

Devin (Cognition)       ───► • Isolates execution in dedicated Linux MicroVMs with shell & browser.
                            • Inner verification loop: runs deterministic test oracles (`pytest`, linters)
                              rather than relying on subjective self-reflection.

SGLang / vLLM           ───► • RadixAttention: organizes KV-cache blocks in a Radix tree for reuse across
                              turns; invented specifically because agent prefill was 80% redundant.

OpenTelemetry (CNCF)    ───► • GenAI Semantic Conventions (`gen_ai.system`, `gen_ai.tool.call`)
                              standardized for causal distributed trajectory tracing.
```

### What Fails the Test (What We Intentionally Exclude)
* 🛑 **Ephemeral orchestration frameworks:** No chapters on LangChain, CrewAI, AutoGen, or LlamaIndex.
* 🛑 **Anthropomorphic metaphors:** No chapters on "Agent Psychology" or "Self-Reflection." (Reflection is analyzed strictly as *Runtime Invariant Verification & Circuit Breakers*).
* 🛑 **Prompt recipe collections:** No prompt engineering tricks or static templates.
* 🛑 **Unconstrained swarms:** No "societies of mind" unless strictly bounded by actor-model message passing, statecharts, and formal deadlock detection.

---

## Resolving the "Training vs. Systems" Fault Line

Top academic institutions are divided on whether students should train agents or build runtimes:
* **CMU 11-768 / Stanford CS329A:** Focus on post-training, SFT on trajectories, and RLVR (Reinforcement Learning from Verifiable Rewards like GRPO/PPO).
* **Harvard CS2680:** Focuses on systems optimizations, GPU serving, prefix caching, and kernel batching.

Volume III unifies this division through **Systems Co-Design: Amortizing Test-Time Compute into Compile-Time Weights**:

```
                           THE AGENT IMPROVEMENT LADDER

  Level 3: Environment RL (RLVR / GRPO)   ▲  High training compute, lowest runtime overhead.
  • Compiles environment feedback into    │  Model learns direct policy & internal verification.
    direct policy weights                 │
  ───────────────────────────────────────┼─────────────────────────────────────────────
  Level 2: Trajectory Distillation & SFT  │  Medium training compute.
  • Distills expensive MCTS search trees │  Amortizes test-time search branches into
    into single-pass student policies    │  straight-line forward passes.
  ───────────────────────────────────────┼─────────────────────────────────────────────
  Level 1: Test-Time Search & Verification│  Zero training compute, high inference compute.
  • Runtime scaffolding (MCTS, verifiers, │  Trades inference tokens and latency for
    critics, rollbacks)                  │  reliability.
  ───────────────────────────────────────┼─────────────────────────────────────────────
  Level 0: Prompt & Context Engineering   │  Zero compute investment.
  • Ephemeral context manipulation        │  Fragile, limited capacity, zero amortization.
```

**The Systems Principle:** Post-training is the *compile-time optimization of an agent runtime*. Running a 50-step MCTS search tree at runtime burns massive GPU compute and pins scarce HBM memory. By fine-tuning and RL-training on verified trajectory rollouts, we compile runtime search scaffolding into model weights, shortening horizon $N$, reducing KV-cache pressure, and cutting production serving costs.

---

## Master 16-Chapter Curriculum Architecture (The 4×4 Matrix)

```
====================================================================================================
PART I: THE TRAJECTORY WORKLOAD & THE CONTROL PLANE
Core Principle: P_success ≤ p^N  ·  The Horizon–Reliability Tension & Compounding Uncertainty
====================================================================================================
Chapter 01: The Post-Request Era: From Models to Managed Trajectories
Chapter 02: The Agent Control Block (ACB) & Execution Descriptors
Chapter 03: Control Topologies: Structured DAGs, Statecharts & The Anti-Swarm Principle
Chapter 04: Trajectory Scheduling, Admission Control & Heavy-Tailed Workloads

====================================================================================================
PART II: THE AI-NATIVE CONTEXT & MEMORY HIERARCHY
Core Principle: Trajectory Cost is Superlinear in N (\sum C_i)  ·  The Working-Set Dynamic
====================================================================================================
Chapter 05: Context as Working Set: Attention Economics & Denning's Memory Model (L1)
Chapter 06: Computation Reuse Across Executions: Radix Prefix Caching (L2)
Chapter 07: Hierarchical Storage & Disaggregation: KV-Swapping, CXL & Paging
Chapter 08: Persistent Episodic Memory: Vector Subsystems, Graph Stores & Context Poisoning (L3)

====================================================================================================
PART III: THE EXECUTION BOUNDARY: TOOLS, ISOLATION & RELIABILITY
Core Principle: The Reversibility Boundary & The Absence of the W⊕X Instruction/Data Split
====================================================================================================
Chapter 09: The Action Interface: Model Context Protocol (MCP) as a Typed ABI
Chapter 10: Sandboxing & Protection Domains: MicroVMs, Wasm & Blast-Radius Containment
Chapter 11: The Fail-Plausible Fault Model & The Verification Tax
Chapter 12: Transactional Consistency: The Reversibility Boundary, Sagas & Rollbacks

====================================================================================================
PART IV: TELEMETRY, FLEET SCALE & SYSTEM OPTIMIZATION
Core Principle: The Invariant Closure Principle & System Amortization
====================================================================================================
Chapter 13: Concurrency, Supervisor Hierarchies & The Coordination Tax
Chapter 14: Deterministic Replay, Time-Travel Debugging & OpenTelemetry Tracing
Chapter 15: Evaluation Gyms, Distribution Benchmarks (MLPerf Agents) & Canary CI/CD
Chapter 16: Fleet Economics, Hardware Co-Design & The Improvement Ladder
====================================================================================================
```

---

## Studio Course Architecture: `AgentKernel`

To ensure deep systems competence, students build a production-grade, bare-metal execution engine—**`AgentKernel`**—from scratch in typed Python/asyncio across four cumulative phases:

* **Phase 1 (Weeks 1–3): The Core Control Plane & ACB**
  * *Milestone 1:* 100-line bare-metal execution harness with step-horizon loops and token circuit breakers.
  * *Milestone 2:* Agent Control Block (ACB) process manager with transitive capability lease attenuation.
  * *Milestone 3:* Cycle-bounded state machine DAG router with $\epsilon$-progress metrics.
* **Phase 2 (Weeks 4–7): Context, Memory & Serving**
  * *Milestone 4:* Token LRU working-set compactor with semantic drift telemetry.
  * *Milestone 5:* Radix tree KV prefix cache engine with environment-mutation invalidation.
  * *Milestone 6:* Asynchronous PCIe KV-cache swapping manager for external tool pauses.
  * *Milestone 7:* Episodic memory store with HNSW vector indexing and exponential temporal decay.
* **Phase 3 (Weeks 8–10): The Isolated Action Boundary**
  * *Milestone 8:* Strict Model Context Protocol (MCP) server with grammar-constrained decoding.
  * *Milestone 9:* Firecracker MicroVM / Wasm sandboxed runner with network egress filtering.
  * *Milestone 10:* Distributed Saga engine with deterministic compensating rollback handlers.
* **Phase 4 (Weeks 11–14): Concurrency, Telemetry & Capstone**
  * *Milestone 11:* Actor-model multi-agent supervisor with semantic deadlock detection.
  * *Milestone 12:* Event-sourcing logger and interactive Time-Travel Replay Debugger.
  * *Milestone 13:* OpenTelemetry GenAI trace exporter and mini-SWE-bench evaluation harness.
  * *Milestone 14 (Capstone):* Trajectory distillation / RLVR optimization or vLLM/SGLang GPU serving integration.

---

## Theoretical Grounding & Landmark Literature

This volume is formally grounded in classic computer systems literature (OSDI, SOSP, NSDI, CAV, POPL, ISCA) and the author's vision paper:
* Reddi, V. J. (2026). *"Architecting the Agentic AI Systems Stack: What Should Infrastructure Manage When the Unit of Work Is a Trajectory?"*, **ACM SIGOPS Operating Systems Review (OSR)**, 60(1), 64–74. (`doi:10.1145/3830422`).
* Complete chapter specifications, empirical industry mappings, and citations are cataloged in [`OUTLINE.md`](OUTLINE.md).
