# Volume III: Agentic Machine Learning Systems

*The systems architecture of inference-time compute, autonomous trajectories, and closed-loop control.*

**Author:** Prof. Vijay Janapa Reddi (Harvard University)
**Status:** 🟣 **In Development / Work in Progress** *(Curriculum Blueprint Locked; Chapter Drafting Active)*

<div align="center" style="background: #faf5ff; border: 1px solid #c084fc; border-radius: 8px; padding: 18px 24px; margin: 20px 0; text-align: left;">
  <div style="font-size: 1.1em; font-weight: bold; color: #6b21a8; margin-bottom: 6px; text-align: center;">
    🟣 Author's Working Note: Early Draft &bull; Learning in Public
  </div>
  <p style="color: #581c87; font-size: 0.95em; line-height: 1.5; margin: 0;">
    <i>"I write these volumes primarily to learn. The field of Agentic AI is moving at breakneck speed, and putting this book together is my own way—as an educator and researcher—of wrestling with where the discipline is actually heading and boiling down the durable first principles from the noise. <b>Consider this an open research notebook:</b> read with curiosity and at your own risk! It is actively evolving and not yet finalized for classroom teaching or formal citation."</i>
    <br><span style="display: block; text-align: right; font-weight: bold; margin-top: 6px;">— Vijay Janapa Reddi</span>
  </p>
</div>

---

To avoid confusion, the complete *Machine Learning Systems* textbook tetralogy is organized into four distinct, non-overlapping volumes:

| Volume | Title | Core Invariant Triad | Scope & Unit of Work | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Vol I** | **Introduction to Machine Learning Systems** | **D · A · M** *(Data, Algorithm, Machine)* | Single-node acceleration, tensor kernels, Roofline models, and baseline serving. Unit: **The Model** | 📗 **Released / In Print** (MIT Press) |
| **Vol II** | **Scaling Machine Learning Systems** | **$\mathbf{C^3}$** *(Compute, Communication, Capacity)* | Distributed supercomputers, 3D parallelism, all-reduce fabrics, and fleet orchestration. Unit: **The Fleet** | 📘 **Preview / Work in Progress** |
| **Vol III** | **Agentic Machine Learning Systems** | **H · S · A** *(Horizon, State, Authority)* | Closed-loop autonomous execution, state DAGs, KV context hierarchies, MCP tools, and Sagas. Unit: **The Trajectory** | 🟣 **In Development / Work in Progress** |
| **Vol IV** | **Physical AI: Machine Learning Systems** | **Physics & Causal Boundaries** | Grounded cyber-physical plants, 1 kHz deterministic control, sensory covariance, and safety shields. Unit: **The Plant / World** | 🌲 **In Development / Work in Progress** |

---

## The Core Thesis of Volume III

Volumes I and II optimize the generation of tokens for single, stateless requests ($Tokens_{\text{out}} = \text{Model}(Tokens_{\text{in}})$).

Volume III addresses what happens when that compute is spent in a stateful, closed-loop process over time: **The system becomes an autonomous actor.** The atomic unit of systems engineering stops being the isolated request and becomes the **Managed Trajectory**—the multi-step lifecycle from delegated goal to verified outcome.

### The Governing Quantitative Law (The Sovereign Law of Agency)
Because errors in autoregressive generation are conditional and history-corrupting ($P(F_i \mid F_{i-1}) \gg P(F_i)$):

$$\mathbf{P_{\text{success}} = \prod_{i=1}^{N} P(\text{Success}_i \mid \text{History}_{i-1}) \le p^{N}} \qquad\text{while}\qquad \mathbf{C_{\text{traj}} = \sum_{i=1}^{N} \Big( C_{\text{ctx}}(i) + C_{\text{gen}}(i) + C_{\text{tool}}(i) + C_{\text{ver}}(i) \Big)}$$

> **The Sovereign Law of Agency:**
> *Capability demands long horizons ($N$). Long horizons destroy reliability exponentially ($p^N$) and inflate memory/token costs superlinearly ($\sum C_{\text{ctx}}(i)$). Verification ($C_{\text{ver}}$) is a non-negotiable systems tax that arrests compounding drift at a computable price.*

---

## Master 16-Chapter Curriculum Architecture

```
====================================================================================================
PART I: THE TRAJECTORY AS THE UNIT OF WORK (Foundations & Control)
====================================================================================================
Chapter 01: From Requests to Trajectories (The Invariant Closure Principle)
Chapter 02: The Agent Control Block (ACB) & Execution Descriptors
Chapter 03: Control Loops, State Graphs & Termination Guarantees
Chapter 04: Scheduling, Admission Control & Resource Contention
Chapter 05: Compounding Error & The Verification Tax

====================================================================================================
PART II: THE AI-NATIVE MEMORY & CONTEXT HIERARCHY
====================================================================================================
Chapter 06: Context as the Working Set (L1 Attention Cache)
Chapter 07: Computation Reuse Across Correlated Executions (L2 Prefix Cache)
Chapter 08: Ephemeral Scratchpads, Consistency & Transaction Isolation
Chapter 09: Durable Memory: Episodic, Semantic & Temporal Decay (L3)

====================================================================================================
PART III: THE AGENT–WORLD BOUNDARY (Tools, Sandboxes & Concurrency)
====================================================================================================
Chapter 10: The Action Interface & Typed Contracts (Model Context Protocol)
Chapter 11: Isolation, Blast Radius & Information Flow Control
Chapter 12: Recovery: Deterministic Sagas, Checkpoints & Actor Swarms

====================================================================================================
PART IV: MEASUREMENT, VERIFICATION & FLEET ECONOMICS
====================================================================================================
Chapter 13: Trajectory Benchmarking & Environment Evaluation (MLPerf Agents)
Chapter 14: Non-Deterministic Replay & Time-Travel Debugging
Chapter 15: Agentic Observability & Distributed Telemetry
Chapter 16: Fleet Economics, Pareto Frontiers & Hardware Co-Design
====================================================================================================
```

---

## 4-Phase Studio Course Lab Architecture: *"Building The Agent Engine"*

Students build a production-grade, bare-metal autonomous agent engine from the ground up without high-level prompt wrapper libraries:

*   **Phase 1 (Weeks 1–4): The Execution Core** — State Machine DAG Router, MCTS Search Engine, and Invariant Circuit Breakers.
*   **Phase 2 (Weeks 5–7): The Context & Memory Hierarchy** — Token LRU Eviction Manager, Radix Prefix Sharing & Invalidation, and Persistent Episodic Memory.
*   **Phase 3 (Weeks 8–10): The Isolated Action Boundary** — Type-Safe MCP Tool Server, Wasm/MicroVM Sandboxing, and Deterministic Saga Rollback Engine.
*   **Phase 4 (Weeks 11–14): Swarms & Trajectory CI/CD** — 3-Agent Actor Swarm with Capability Downgrades, Time-Travel Replay Debugger, and MLPerf Agents Submission Package.

---

## Theoretical Grounding & Landmark Literature

This volume is formally grounded in classic computer systems literature (OSDI, SOSP, NSDI, CAV, POPL, ISCA) and the author's vision paper:
*   Reddi, V. J. (2026). *"Architecting the Agentic AI Systems Stack: What Should Infrastructure Manage When the Unit of Work Is a Trajectory?"*, **ACM SIGOPS Operating Systems Review (OSR)**, 60(1), 64–74. (`doi:10.1145/3830422`).
*   Complete section breakdowns, mathematical proofs, and citations are cataloged in [`AGENTIC_AI_CURRICULUM_BLUEPRINT.md`](AGENTIC_AI_CURRICULUM_BLUEPRINT.md).
