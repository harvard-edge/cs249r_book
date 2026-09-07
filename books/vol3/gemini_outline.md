# Volume III: Agentic Machine Learning Systems
## Architectural Outline & Systems Rationale (Gemini Perspective)

**Author:** Prof. Vijay Janapa Reddi (Harvard University)
**Status:** 🟣 Working Blueprint & Curriculum Architecture
**Foundational Series:** *Machine Learning Systems* (Volumes I–IV)

---

### Executive Summary: The Pure Systems Stance

Writing a textbook on Agentic AI in the middle of an industry hype cycle carries a mortal risk: **writing a transitional book that becomes obsolete within eighteen months.** Most contemporary literature on agents fails because it falls into one of three traps:

1. **The Framework Trap:** Documenting ephemeral high-level library APIs, prompt templates, and orchestration wrappers that churn every quarter.
2. **The Anthropomorphic Trap:** Borrowing soft psychological and cognitive metaphors ("reflection," "debate," "agent personas," "empathy") that obscure concrete engineering trade-offs.
3. **The Forced Metaphor Trap:** Over-indexing on historical analogies—such as forcing an agent into an Operating System or a CPU datapath/control plane—without rigorously acknowledging where the physics of neural computing causes the analogy to fracture.

This outline approaches **Agentic Machine Learning Systems** strictly through the lens of classical systems engineering (Hennessy & Patterson, Saltzer, Lampson, Ousterhout).

The premise is straightforward:
> **In traditional MLSys (Volumes I & II), inference is stateless, feed-forward, and compute-dense ($Tokens_{\text{out}} = \text{Model}(Tokens_{\text{in}})$).**
> **In Agentic MLSys (Volume III), inference is stateful, closed-loop, and interleaved with non-compute environment I/O ($s_{t+1}, a_t = \text{Policy}(s_t, o_t)$).**

The moment a neural network enters an autonomous loop with external tools and persistent state, it becomes an **untrusted, stochastic process executing on a heterogeneous distributed runtime**. This creates novel systems challenges in resource scheduling, memory hierarchies, failure recovery, and isolation boundaries.

---

## 1. The Core Paradigm Shift: From Requests to Trajectories

Volumes I and II operated under a sixty-year-old computing invariant: **the stateless, bounded request**.

$$\text{Instruction} \longrightarrow \text{Process/Thread} \longrightarrow \text{Request/RPC} \longrightarrow \text{Model Call} \longrightarrow \mathbf{\text{Managed Trajectory}}$$

* In **Volume I (*The Model*)**, the unit of work was *the model* on single-node silicon, governed by **D·A·M** (*Data, Algorithm, Machine*), Roofline arithmetic intensity, and tensor compiler memory walls.
* In **Volume II (*The Fleet*)**, the unit of work was *the fleet*, governed by **$\mathbf{C^3}$** (*Compute, Communication, Capacity*), all-reduce topologies, 3D parallelism, and fail-stop hardware faults.
* In **Volume III (*The Trajectory*)**, the unit of work is **the Managed Trajectory**: an open-ended sequence of interdependent neural inferences, tool invocations, and environment mutations over time.

### The Invariant Closure Principle
A systems unit is too small when the invariants the infrastructure must enforce—budget ceilings, authorization leases, state consistency, and transaction rollbacks—can only be stated by reaching outside that unit. For multi-step agents, budget bounds, security taint boundaries, and side-effect rollbacks cannot be enforced at the single-token or single-request layer. The atomic unit of systems engineering shifts from the request to the **Managed Trajectory**, tracked via an **Agent Control Block (ACB)**.

---

## 2. The Four Governing Systems Bottlenecks

To ensure this textbook remains foundational for a decade or more, **every systems abstraction taught must be grounded in an inescapable physical or mathematical constraint**:

```
                       THE FOUR GOVERNING SYSTEMS BOTTLENECKS

 ┌──────────────────────────────────────┐      ┌──────────────────────────────────────┐
 │ 1. The Compounding Error Wall        │      │ 2. The Context & Memory Wall         │
 │    • P_success ≤ p^N                 │      │    • Superlinear attention costs     │
 │    • Fail-plausible semantic drift   │      │    • Idle HBM during tool wait I/O   │
 │    • Forces Verification Tax (C_ver) │      │    • Forces Radix Caches & Paging    │
 └──────────────────┬───────────────────┘      └──────────────────┬───────────────────┘
                    │                                             │
                    ▼                                             ▼
 ┌──────────────────────────────────────┐      ┌──────────────────────────────────────┐
 │ 3. The Unbounded Execution Loop      │      │ 4. The Action & Reversibility Wall   │
 │    • Heavy-tailed service times      │      │    • No instruction/data separation  │
 │    • Semantic deadlocks & loops      │      │    • Irreversible real-world effects │
 │    • Forces Admission & Watchdogs    │      │    • Forces Sandboxes & Saga Rollback│
 └──────────────────────────────────────┘      └──────────────────────────────────────┘
```

### 1. The Compounding Error Wall ($P_{\text{success}} \le p^N$)
Autoregressive token generation has an inherent per-step error probability $(1-p)$. Because reasoning steps are conditional and history-dependent ($P(\text{Fail}_i \mid \text{Fail}_{i-1}) \gg P(\text{Fail}_i)$), errors compound multiplicatively across horizon $N$:

$$P_{\text{success}} \le \prod_{i=1}^{N} P(\text{Success}_i \mid \text{History}_{i-1}) \approx p^N$$

At single-step accuracy $p = 0.95$, a 20-step trajectory succeeds only $35.8\%$ of the time; at $N = 50$, it collapses to $7.7\%$. You cannot buy out of this wall simply by scaling pre-training weights, because the horizon exponent $N$ dominates the base accuracy $p$. This forces the system to introduce **Test-Time Compute (search, MCTS)** and **Runtime Verification Gates**.

### 2. The Context & Memory Wall ($\sum C_i$)
In a CPU, executing instruction 10,000 costs the same energy and latency as instruction 1. In an agentic system, **step $N$ costs dramatically more than step 1**:

$$T_{\text{traj}} = \sum_{i=1}^{N} \Big( T_{\text{prefill}}(C_i) + T_{\text{decode}}(o_i) + T_{\text{tool},i} + T_{\text{verify},i} \Big)$$

As context $C_i$ accumulates with tool outputs and thoughts, attention FLOPs and KV-cache footprints expand. More critically, holding the KV cache inside expensive GPU High Bandwidth Memory (HBM) while an agent waits on hundreds of milliseconds of external tool I/O causes massive memory starvation in serving clusters. This forces **Prefix Caching (Radix trees)**, **Asynchronous KV-Swapping**, and **Context Compaction**.

### 3. The Unbounded Execution Loop & Unknown Service Times
Traditional web servers schedule requests with tight, known latency distributions (e.g., 200ms p99). An agentic trajectory has an unpredictable horizon ($N \in [1, 500]$) and heavy-tailed service times. A naive while-loop will consume infinite tokens in semantic livelocks. This forces **Dynamic State DAGs**, **Cycle Detection**, **Watchdog Timers**, and **Preemptive Schedulers**.

### 4. The Action Boundary & The Reversibility Wall
In traditional systems, code and data are separated by hardware privileges ($W \oplus X$). In language models, untrusted external data (scraped HTML, user input, API responses) is mixed directly into the instruction stream, creating an unfixable linguistic injection surface. Furthermore, internal state mutations are cheap and reversible, but external real-world actions (sending emails, modifying databases, deleting files) often have no inverse. This forces **MicroVM/Wasm Isolation Sandboxes**, **Taint Tracking**, and **Compensating Saga Transactions**.

---

## 3. Tying Agentic Theory to Concrete Systems Mechanisms

A systems student must understand how high-level theoretical concepts map directly to low-level systems implementations:

| Agent / Cognitive Theory Concept | What It Actually Is in Systems Engineering | Concrete Systems Mechanism |
| :--- | :--- | :--- |
| **Reasoning & Planning** *(MCTS, Tree-of-Thought, Best-of-N)* | **Branch Speculation & Search Tree Scheduling** | Tree-structured KV-cache allocation, branch preemption, asynchronous batching, speculative rollouts. |
| **Working & Episodic Memory** *(Reflection, Vector Stores)* | **Hierarchical Paging & Cache Eviction** | L1 Context Working Set, L2 Radix Prefix Cache, L3 Vector/KV Store with LRU eviction and semantic compaction. |
| **Tool Calling & Actuation** *(Function Calling, MCP)* | **Typed I/O Subsystem & System Call ABI** | Schema compilation, grammar-constrained decoding, JSON-RPC marshalling, latency-hiding prefetch. |
| **Sandboxing & Permissions** *(Safety Guardrails)* | **Protection Domains & Information Flow Control** | Firecracker MicroVMs, Wasm runtimes, capability leases, egress firewalls, taint tracking. |
| **Multi-Agent Collaboration** *(Debate, Swarms)* | **Distributed Systems with Stochastic Nodes** | Non-blocking message queues, deadlock/livelock cycle detection, serialization overhead, Coordination Tax. |
| **Self-Correction & Feedback** | **Runtime Invariant Verification & Circuit Breakers** | Deterministic linters, test oracles, critic model gating, early exit circuit breakers to stop token burn. |

---

## 4. Master 4-Part, 16-Chapter Curriculum Architecture

```
====================================================================================================
AGENTIC MACHINE LEARNING SYSTEMS: RUNTIMES, SCHEDULING & STATE
====================================================================================================

PART I: THE TRAJECTORY WORKLOAD & THE CONTROL PLANE
Chapter 01: The Post-Request Era: From Models to Managed Trajectories
Chapter 02: The Agent Control Block (ACB) & Execution Descriptors
Chapter 03: Control Loops, State DAGs & Termination Guarantees
Chapter 04: Compounding Error & The Verification Tax
Chapter 05: Trajectory Scheduling, Admission Control & Resource Quotas

PART II: MEMORY ARCHITECTURE & ACCELERATOR EFFICIENCY
Chapter 06: Context as Working Memory: The Limits of Quadratic Attention
Chapter 07: Prefix Caching & Computation Reuse Across Executions (Radix Trees)
Chapter 08: Hierarchical Storage: KV-Swapping, Paging & External Memory
Chapter 09: Context Eviction, Working-Set Compaction & Semantic Drift

PART III: THE EXECUTION BOUNDARY: TOOLS, I/O & CONCURRENCY
Chapter 10: The Action Interface: Typed Contracts & Schema Enforcement (MCP)
Chapter 11: Sandboxing, Protection Domains & Blast-Radius Containment
Chapter 12: Transactional Consistency: Checkpointing, Rollbacks & Sagas
Chapter 13: Distributed Stochastic Nodes: Coordination, Message Buses & Deadlocks

PART IV: RELIABILITY, OBSERVABILITY & FLEET ECONOMICS
Chapter 14: Non-Deterministic Replay & Time-Travel Debugging
Chapter 15: Distributed Trajectory Tracing & Telemetry
Chapter 16: Fleet Economics: Cost-per-Goal, Pareto Frontiers & Hardware Co-Design
====================================================================================================
```

---

## 5. Chapter-by-Chapter Architectural Breakdown

### PART I: The Trajectory Workload & The Control Plane

#### Chapter 01: The Post-Request Era: From Models to Managed Trajectories
* **Durable Anchor:** The architectural shift from stateless token generation to stateful, multi-turn trajectory execution. The Invariant Closure Principle: why request-scoped microservices fail when workflows span hundreds of interdependent model steps. Derivation of the two fundamental horizon walls: $P_{\text{success}} \le p^N$ and $\sum C_i$.
* **Systems Invariant:** The managed trajectory—not the token and not the model—is the atomic unit of scheduling, budgeting, and failure recovery.
* **What is Out of Scope:** High-level agent framework tutorials (LangChain, AutoGen, CrewAI); philosophical musings on machine consciousness.

#### Chapter 02: The Agent Control Block (ACB) & Execution Descriptors
* **Durable Anchor:** The Process Control Block (PCB) of AI systems. Structure of an ACB: Principal Identity, Goal Envelope, Step/Token Quotas, Capability Leases, Memory Table Handles, Causality Trace Vectors, and Parent-Child Delegation Pointers.
* **Systems Invariant:** The runtime must maintain a unified kernel descriptor over execution state, resource leases, and recovery vectors across time.
* **Teachable Edge:** Unlike OS processes where state can be bit-exact saved via `memcpy`, an agent's neural belief state is non-stationary and distributed across attention context and external environments.

#### Chapter 03: Control Loops, State DAGs & Termination Guarantees
* **Durable Anchor:** Formalizing agent execution as Directed Acyclic Graphs (DAGs), Markov Decision Processes (MDPs), and Statecharts. Invariant stopping conditions; cycle and livelock detection in semantic state spaces; $\epsilon$-progress metrics to kill non-productive reasoning loops.
* **Systems Invariant:** An unconstrained `while True:` loop is a Denial-of-Service vulnerability against your own budget and GPU cluster.
* **Teachable Edge:** Why traditional halting problem heuristics fail on semantic agents, requiring probabilistic liveness proofs and dynamic depth budgeting.

#### Chapter 04: Compounding Error & The Verification Tax
* **Durable Anchor:** Non-stationary Markov chain error models; mathematical derivation of the Reliability Wall. Test-time search architectures (Best-of-$N$, Beam Search, Monte Carlo Tree Search). Verifiers as first-class runtime primitives: deterministic unit test oracles, schema validators, and learned outcome/process reward models (ORM/PRM).
* **Systems Invariant:** The Verification Tax ($C_{\text{ver}}$) is a mandatory systems cost required to bend an exponential failure curve back into a stable trajectory.
* **Trade-Off Curve:** Formulating the Pareto frontier: Verification compute cost vs. wasted token expenditure from discarded trajectory rollbacks.

#### Chapter 05: Trajectory Scheduling, Admission Control & Resource Quotas
* **Durable Anchor:** Schedulers managing heavy-tailed, unpredictable execution times (Shortest Remaining Processing Time under uncertainty). Dynamic preemption policies; admission control algorithms based on predicted horizon length to prevent GPU out-of-memory (OOM) events.
* **Systems Invariant:** Irreversibility is a scheduling constraint—a scheduler cannot arbitrarily preempt, migrate, or terminate a job that has already committed irreversible external side-effects.

---

### PART II: Memory Architecture & Accelerator Efficiency

#### Chapter 06: Context as Working Memory: The Limits of Quadratic Attention
* **Durable Anchor:** Denning's working set theory applied to token context windows. The physical limits of High Bandwidth Memory (HBM) on modern accelerators; quadratic $O(N^2)$ vs. linear attention architectures; memory bandwidth bounds during autoregressive decode.
* **Systems Invariant:** Context window capacity is the L1 cache of neural compute—fast, zero-retrieval latency, but physically constrained and economically expensive.
* **Teachable Edge:** Attention degradation ("lost in the middle") treated as an architectural cache locality failure rather than a prompt engineering quirk.

#### Chapter 07: Prefix Caching & Computation Reuse (Radix Trees)
* **Durable Anchor:** Prefix locality in multi-step trajectories and tree searches. 80–95% of prompt tokens are identical across consecutive steps and speculative branches. Radix tree indexing of KV-cache blocks (PagedAttention, Chunked Prefill, SGLang runtime).
* **Systems Invariant:** State reuse is the primary determinant of agent serving throughput.
* **Systems Mechanism:** Cache invalidation policies when dynamic tool responses or environment observations mutate the shared prefix history.

#### Chapter 08: Hierarchical Storage: KV-Swapping, Paging & External Memory
* **Durable Anchor:** Solving the idle GPU memory problem during external tool wait states. Multi-tiered memory systems: Accelerator HBM $\longleftrightarrow$ Host DRAM $\longleftrightarrow$ Local NVMe $\longleftrightarrow$ Remote Storage.
* **Systems Invariant:** Decoupling context state retention from active accelerator execution.
* **Systems Mechanism:** Asynchronous KV-cache eviction and prefetching over PCIe during tool I/O round-trips; latency vs. memory density trade-offs.

#### Chapter 09: Context Eviction, Working-Set Compaction & Semantic Drift
* **Durable Anchor:** Algorithmic context management when a trajectory's length exceeds physical hardware limits. Sliding window attention, attention-score-guided pruning, and recursive summarization.
* **Systems Invariant:** Eviction in neural systems is inherently lossy; address translation is semantic, not bit-exact.
* **Systems Mechanism:** Quantifying and arresting *Semantic Drift*—the entropy and information loss that accumulates when trajectories repeatedly summarize their own past context.

---

### PART III: The Execution Boundary: Tools, I/O & Concurrency

#### Chapter 10: The Action Interface: Typed Contracts & Schema Enforcement (MCP)
* **Durable Anchor:** The System Call ABI of AI. Models must not emit unconstrained natural language to act upon the world; they must execute typed, schema-validated contracts over standard protocols (e.g., Model Context Protocol). Grammar-constrained decoding and JSON/Protobuf serialization costs.
* **Systems Invariant:** The interface between a stochastic model and a deterministic environment must be statically typed, schema-validated, and versioned.
* **Teachable Edge:** A typed wire protocol is an ABI contract, NOT a security boundary. Conflating schema validation with isolation leads to catastrophic privilege escalation.

#### Chapter 11: Sandboxing, Protection Domains & Blast-Radius Containment
* **Durable Anchor:** The absence of the Von Neumann instruction/data split ($W \oplus X$ failure): untrusted data from the web enters the exact same sequence as privileged system prompts. Sandboxing execution: Docker vs. WebAssembly (Wasm) vs. MicroVMs (Firecracker).
* **Systems Invariant:** The execution environment of an agent must assume the policy is compromised by untrusted inputs at all times.
* **Systems Mechanism:** Information Flow Control (IFC), ephemeral filesystems, network egress filtering, and fine-grained capability leases with automatic expiration.

#### Chapter 12: Transactional Consistency: Checkpointing, Rollbacks & Sagas
* **Durable Anchor:** The Reversibility Boundary: Internal state is private, cheap, and reversible; external state is shared, expensive, and often irreversible. Failure recovery in multi-step workflows without distributed two-phase locking.
* **Systems Invariant:** You cannot put real-world external side-effects into a CPU store buffer; recovery requires compensating actions.
* **Systems Mechanism:** Checkpointing state DAGs; Distributed Sagas for agent workflows with typed compensating actions ($A^{-1}$); fork-on-error branching.

#### Chapter 13: Distributed Stochastic Nodes: Coordination & The Coordination Tax
* **Durable Anchor:** Multi-agent collaboration analyzed as distributed computing over stochastic, non-deterministic compute nodes. The Coordination Tax: why $M$ agents rarely yield $M\times$ throughput due to communication overhead, serialization, and noise amplification.
* **Systems Invariant:** Distributed consensus among probabilistic actors differs fundamentally from classical Paxos/Raft.
* **Systems Mechanism:** Message queuing architectures; semantic deadlock and livelock detection (e.g., agents repeatedly modifying and reverting each other's work); hierarchical coordinator vs. peer-to-peer gossip topologies.

---

### PART IV: Reliability, Observability & Fleet Economics

#### Chapter 14: Non-Deterministic Replay & Time-Travel Debugging
* **Durable Anchor:** The systems challenge of debugging stochastic failures that only reproduce 5% of the time. Sources of non-determinism: temperature sampling, GPU floating-point non-associativity across parallel CUDA threads, and asynchronous tool network latency.
* **Systems Invariant:** An agent system is unmaintainable in production without deterministic trace reconstructibility.
* **Systems Mechanism:** Event capture-and-replay engines; pseudorandom seed injection; mocking external tool I/O; branching counterfactual debuggers.

#### Chapter 15: Distributed Trajectory Tracing & Telemetry
* **Durable Anchor:** Application Performance Monitoring (APM) for long-horizon autonomous workloads. Distributed tracing across hierarchical subagent calls, token generation phases, and tool executions.
* **Systems Invariant:** Telemetry must capture causal DAG lineage and resource burn velocity, not just point-in-time request latencies.
* **Systems Mechanism:** OpenTelemetry GenAI semantic conventions; real-time anomaly detection for looping, semantic drift, and runaway token burn.

#### Chapter 16: Fleet Economics: Cost-per-Goal, Pareto Frontiers & Hardware Co-Design
* **Durable Anchor:** The macro-economics of autonomous workloads. Why "tokens per second" is an obsolete metric; defining **Trajectory Goodput** ($\text{Verified Goals} / \text{Dollar} / \text{GPU-Hour}$). Test-time compute search vs. pre-training model scale trade-offs.
* **Systems Invariant:** The economics of agentic systems will drive future datacenter accelerator co-design (e.g., CXL memory pooling for KV-caches, silicon acceleration for microVM context switching).
* **Benchmarking Standard:** Standardizing reproducible systems evaluation via **MLPerf Agents**.

---

## 6. Four-Phase Studio Lab Architecture: *"Building The Agent Engine"*

To ensure students graduate with hands-on systems competence, they construct a production-grade, bare-metal **Agent Execution Kernel** in Python and Rust without high-level prompt wrapper libraries:

* **Phase 1 (Weeks 1–4): The Execution Core**
  * Lab 1: State Machine DAG Router with cycle detection and token circuit-breakers.
  * Lab 2: MCTS Test-Time Search Engine with speculative branch pruning.
  * Lab 3: Agent Control Block (ACB) process manager and priority scheduler.
* **Phase 2 (Weeks 5–7): Context & Memory Management**
  * Lab 4: Radix Tree KV Prefix Cache with invalidation on environment mutation.
  * Lab 5: Asynchronous KV-Cache Pager (swapping inactive agent contexts to host memory).
  * Lab 6: Token LRU Working-Set Compactor with semantic drift telemetry.
* **Phase 3 (Weeks 8–10): The Isolated Action Boundary**
  * Lab 7: Type-Safe MCP Tool Gateway with schema validation and grammar-constrained output.
  * Lab 8: MicroVM/Wasm Sandboxed Execution Runner with network egress security filters.
  * Lab 9: Distributed Saga Engine with compensating rollback actions.
* **Phase 4 (Weeks 11–14): Concurrency, Replay & Benchmarking**
  * Lab 10: Multi-Agent Actor Swarm with deadlock detection.
  * Lab 11: Non-Deterministic Event Logger and Time-Travel Replay Debugger.
  * Lab 12: End-to-end evaluation harness submitting to the MLPerf Agents benchmark suite.

---

### The Pedagogical Litmus Test

When drafting each chapter, ask:
1. *Does this chapter rely on a named product or API syntax that could disappear in two years?* (If yes, rewrite it to teach the underlying systems contract).
2. *Can the student implement the core mechanism in 150 lines of bare-metal code?* (If no, the abstraction is too murky).
3. *Does the chapter state an explicit physical or mathematical trade-off?* (Every chapter must present a quantifiable Pareto frontier: Latency vs. Memory, Accuracy vs. Verification Cost, Throughput vs. Isolation Overhead).
