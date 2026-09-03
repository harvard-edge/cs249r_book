# Master Curriculum Blueprint
# Volume III: Agentic Machine Learning Systems
**Author:** Prof. Vijay Janapa Reddi, Harvard University
**Foundational Series:** Machine Learning Systems (Volumes I–IV)
**Primary Theoretical Anchor:** Reddi, V. J., *"Architecting the Agentic AI Systems Stack: What Should Infrastructure Manage When the Unit of Work Is a Trajectory?"*, ACM SIGOPS Operating Systems Review (OSR), Vol. 60, No. 1, 2026, pp. 64–74. (`doi:10.1145/3830422`)

---

## Executive Summary & Foundational Invariants

### 1. The Core Paradigm Shift
Volume I (*The Model*) addresses single-node computation, memory hierarchies, and hardware acceleration (**D·A·M**). Volume II (*The Fleet*) addresses distributed scaling, all-reduce topologies, and cluster fault tolerance ($\mathbf{C^3}$).

Volume III (*Agentic Systems*) addresses the transition from **stateless, request-scoped token serving** to **stateful, open-ended, autonomous trajectory execution**:

$$\text{Instruction} \longrightarrow \text{Process/Thread} \longrightarrow \text{Request/RPC} \longrightarrow \text{Container} \longrightarrow \text{Model Call} \longrightarrow \mathbf{\text{Managed Trajectory}}$$

### 2. The Governing Volume Invariant: The H·S·A Triad
Every agentic design decision trades among three fundamental axes:
*   **Horizon ($H$):** The number of sequential, interdependent steps before task closure.
*   **State ($S$):** The working memory, context cache, and persistent belief that must outlast individual steps.
*   **Authority ($A$):** The scope of external side-effects, tool capabilities, and resource mutations the agent is permitted to execute.

### 3. The Governing Mathematical Law: Cascading Error vs. Verification Cost
Because errors in autoregressive generation are conditional and history-corrupting ($P(F_i \mid F_{i-1}) \gg P(F_i)$), reliability decays as a non-stationary Markov chain:

$$\mathbf{P_{\text{success}} = \prod_{i=1}^{N} P(\text{Success}_i \mid \text{History}_{i-1}) \le p^{N}}$$

while trajectory cost accumulates across context, inference, tool execution, and verification:

$$\mathbf{C_{\text{traj}} = \sum_{i=1}^{N} \Big( C_{\text{ctx}}(i) + C_{\text{gen}}(i) + C_{\text{tool}}(i) + C_{\text{ver}}(i) \Big)}$$

> **The Sovereign Law of Agency:**
> *Capability demands long horizons ($N$). Long horizons destroy reliability exponentially ($p^N$) and inflate memory/token costs superlinearly ($\sum C_{\text{ctx}}(i)$). Verification ($C_{\text{ver}}$) is a non-negotiable systems tax that arrests compounding drift at a computable price.*

---

## Complete 4-Part, 16-Chapter Master Syllabus & Section Breakdown

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

# Detailed Chapter Blueprint

## PART I: The Trajectory as the Unit of Work

### Chapter 01: From Requests to Trajectories
* **Core Systems Invariant:** *The Invariant Closure Principle* — A managed unit is too small when the invariants the system must preserve can only be stated by reaching outside that unit.
* **Sections:**
  * **1.1 The Post-Request Era:** Why 60 years of request-scoped, stateless infrastructure (HTTP, REST, microservices) fails when autonomous agents execute multi-step workflows.
  * **1.2 The Lineage of Managed Units:** From instructions and processes to containers, model calls, and managed trajectories.
  * **1.3 Invariant Closure:** The end-to-end argument applied to agentic boundaries; bounding budget, authority, recovery, evidence, and physical cost.
  * **1.4 The A·S·H Triad:** Authority, State, and Horizon as the canonical coordinates of agentic systems architecture.
  * **1.5 Where the OS Analogy Breaks:** Semantic vs. binary state, natural language as an attackable ISA, continuous token economics, and distributional correctness.
* **Seminal Citations:**
  * Reddi, V. J. (2026). *Architecting the Agentic AI Systems Stack*. ACM SIGOPS Operating Systems Review, 60(1), 64–74.
  * Saltzer, J. H., Reed, D. P., & Clark, D. D. (1984). *End-to-End Arguments in System Design*. ACM TOCS, 2(4), 277–288.
  * Lampson, B. W. (1983). *Hints for Computer System Design*. ACM Operating Systems Review, 17(5), 33–48.
  * Yao, S., et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023.
* **MLPerf Agents Anchor:** Formalizing the benchmark unit of work as a multi-step task execution rather than isolated token latency.

---

### Chapter 02: The Agent Control Block (ACB) & Execution Descriptors
* **Core Systems Invariant:** *Stateful Descriptor Binding* — The system must maintain a unified handle over identity, active budget, capability leases, memory pointers, and recovery policy throughout an execution's lifecycle.
* **Sections:**
  * **2.1 The Process Control Block (PCB) of AI:** Anatomy of an ACB (Principal ID, Goal Envelope, Token Allowance, Tool Lease Table, Memory Handles, Recovery Vector).
  * **2.2 State Representation:** Semantic state vs. structural context; the impossibility of bit-exact `memcpy` for neural belief states.
  * **2.3 The Agent Lifecycle State Machine:** `Initialized` $\rightarrow$ `Admitted` $\rightarrow$ `Executing` $\rightarrow$ `Suspended` $\rightarrow$ `Yielding` $\rightarrow$ `Completed` / `Faulted`.
  * **2.4 Capability-Based Leases:** Time-bounded, scope-bounded, and counter-bounded delegation contracts.
  * **2.5 Transitive Delegation & Token Downgrades:** Preventing privilege escalation when subagents are dynamically spawned.
* **Seminal Citations:**
  * Dennis, J. B., & Van Horn, E. C. (1966). *Programming Semantics for Multiprogrammed Computations*. Communications of the ACM, 9(3), 143–155.
  * Mei, K., et al. (2024). *AIOS: LLM Agent Operating System*. arXiv:2403.16971.
  * Packer, C., et al. (2023). *MemGPT: Towards LLMs as Operating Systems*. arXiv:2310.08560.

---

### Chapter 03: Control Loops, State Graphs & Termination Guarantees
* **Core Systems Invariant:** *Bounded Dynamic Execution* — An unconstrained while-loop is an unbounded liability; robust agents must execute as verified, cycle-bounded state graphs.
* **Sections:**
  * **3.1 The Failure of Naive Loops:** Infinite loops, cyclic reasoning traps, and runaway token burn.
  * **3.2 Agents as Directed Acyclic Graphs (DAGs) and Statecharts:** State transition tables, edge predicates, and conditional branching.
  * **3.3 Markov Decision Processes (MDPs) over Token Horizons:** States, actions, transition dynamics, and reward feedback in semantic environments.
  * **3.4 Livelock & Deadlock Detection:** Cycle detection algorithms in agent execution graphs; dynamic depth budgets.
  * **3.5 Probabilistic Safety & Termination Guarantees:** Defining invariant stopping conditions ($\epsilon$-progress thresholds, max token/step bounds, formal liveness proofs).
  * **3.6 Deep Modules for Agent Workflows:** Ousterhout's principle applied to agent node interfaces.
* **Seminal Citations:**
  * Ousterhout, J. (2018). *A Philosophy of Software Design*. Yaknyam Press.
  * Shinn, N., et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023.
  * Harel, D. (1987). *Statecharts: A Visual Formalism for Complex Systems*. Science of Computer Programming, 8(3), 231–274.
  * Kwiatkowska, M., et al. (2011). *PRISM 4.0: Verification of Probabilistic Real-Time Systems*. CAV 2011.

---

### Chapter 04: Scheduling, Admission Control & Resource Contention
* **Core Systems Invariant:** *Trajectory-Aware Scheduling* — Schedulers must allocate compute, KV cache memory, and tool bandwidth across long-running, variable-step trajectories rather than single requests.
* **Sections:**
  * **4.1 The Resource Footprint of a Trajectory:** High variance in step count, context accumulation, and heterogeneous tool latency.
  * **4.2 Admission Control Under Memory Constraints:** Preventing KV cache exhaustion through predictive context sizing.
  * **4.3 Priority Scheduling & Fair Sharing:** Dominant Resource Fairness (DRF) extended to tokens, KV memory blocks, and tool rate limits.
  * **4.4 Preemption and Suspension:** Suspending long-running agent trajectories during tool waits or human approvals; context offloading.
  * **4.5 Co-Scheduling Multi-Agent Workflows:** Gang scheduling for interacting agent swarms.
* **Seminal Citations:**
  * Ghodsi, A., et al. (2011). *Dominant Resource Fairness: Fair Allocation of Multiple Resource Types*. NSDI 2011.
  * Waldspurger, C. A., & Weihl, W. E. (1994). *Lottery Scheduling: Flexible Proportional-Share Resource Management*. OSDI 1994.
  * Sheng, Y., et al. (2024). *Fairness in Serving Large Language Models*. arXiv:2401.03044.

---

### Chapter 05: Compounding Error & The Verification Tax
* **Core Systems Invariant:** *The Verification Tax* — Error compounds conditionally across trajectory steps; deterministic verification is the only antidote, trading test-time compute for execution reliability.
* **Sections:**
  * **5.1 The Mathematics of Conditional Compounding Error:** Why single-step errors corrupt historical context ($P(F_i \mid F_{i-1}) \gg P(F_i)$) and shatter long-horizon tasks.
  * **5.2 Drift & Hallucination Amplification:** Observational contamination across multi-step execution traces.
  * **5.3 Scaling Test-Time Compute:** Monte Carlo Tree Search (MCTS), Tree-of-Thoughts (ToT), Best-of-$N$ sampling, and iterative self-refinement.
  * **5.4 The Verification Tax:** Computing the Pareto frontier of intermediate validation cost ($C_{\text{ver}}$) vs. end-to-end task completion rate.
  * **5.5 Optimal Checkpoint Frequencies:** Deriving the optimal verification cadence under latency and dollar budgets.
* **Seminal Citations:**
  * Snell, C., et al. (2024). *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*. arXiv:2408.03314.
  * Yao, S., et al. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*. NeurIPS 2023.
  * Cobbe, K., et al. (2021). *Training Verifiers to Solve Math Word Problems*. arXiv:2110.14168.

---

## PART II: The AI-Native Memory & Context Hierarchy

### Chapter 06: Context as the Working Set (L1 Attention Cache)
* **Core Systems Invariant:** *The Attention Budget* — Context is the L1 cache of intelligence: fast, expressive, strictly finite, and quadratically expensive in compute and memory.
* **Sections:**
  * **6.1 The Economics of Context:** $O(N^2)$ self-attention complexity, memory bandwidth pressure, and TTFT bloat.
  * **6.2 Denning's Working Set Theory for Agents:** Formulating active context as the working set of tokens needed for current reasoning.
  * **6.3 Context Packing & Fragmentation:** Squeezing instructions, history, retrieved documents, and tool outputs into bounded windows.
  * **6.4 Eviction Policies:** Token LRU, semantic importance scoring, attention score pruning, and sliding-window compaction.
  * **6.5 Positional Bias & Lost-in-the-Middle:** Why placement in the context window governs retrieval efficacy.
* **Seminal Citations:**
  * Denning, P. J. (1968). *The Working Set Model for Program Behavior*. Communications of the ACM, 11(5), 323–333.
  * Liu, N. F., et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts*. TACL 2023.
  * Dao, T., et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022.

---

### Chapter 07: Computation Reuse Across Correlated Executions (L2 Prefix Cache)
* **Core Systems Invariant:** *Prefix Locality & Coherence* — Multi-turn agent trajectories and branching search trees exhibit massive KV cache prefix sharing; memory architectures must exploit tree-structured prefix reuse with rigorous cache invalidation.
* **Sections:**
  * **7.1 The Physics of KV Caching in Agent Loops:** Re-reading static system prompts and growing historical trajectories.
  * **7.2 Paged Attention Primitives:** Non-contiguous physical memory allocation for dynamic token sequences.
  * **7.3 Radix Tree Prefix Caching:** Dynamic matching, insertion, and eviction of shared prefix KV blocks across branching agent runs.
  * **7.4 Cross-Agent Cache Sharing & Invalidation:** Invalidation triggers when underlying environment or file state mutates.
  * **7.5 Disaggregated Prefill-Decode for Stateful Agents:** Offloading prefill compute and streaming KV states over high-speed interconnects.
* **Seminal Citations:**
  * Kwon, W., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023.
  * Zheng, L., et al. (2023). *SGLang: Efficient Execution of Structured Language Model Programs*. arXiv:2312.07104.
  * Zhong, Y., et al. (2024). *DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving*. OSDI 2024.
  * Qin, H., et al. (2024). *Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving*. OSDI 2024.

---

### Chapter 08: Ephemeral Scratchpads, Consistency & Transaction Isolation
* **Core Systems Invariant:** *Working State Isolation* — Agents require private, disposable scratchpads for intermediate reasoning that must maintain explicit consistency models before committing to shared memory.
* **Sections:**
  * **8.1 In-Flight Reasoning vs. Committed State:** Draft generation, chain-of-thought scratchpads, and intermediate code execution.
  * **8.2 Transaction Isolation Levels for Swarms:** Read uncommitted, read committed, and serializable state mutations across multi-agent shared state.
  * **8.3 Semantic Cache Invalidation:** How to detect stale context when parallel agents mutate shared variables or files.
  * **8.4 Memory Compaction & Commit Gates:** Distilling 20 steps of scratchpad reasoning into a concise 1-paragraph committed state.
  * **8.5 Ephemeral State Garbage Collection:** Reclaiming memory and disk handles after trajectory termination.
* **Seminal Citations:**
  * Nye, M., et al. (2021). *Show Your Work: Scratchpads for Intermediate Computation with Language Models*. arXiv:2112.00114.
  * Gray, J. (1978). *Notes on Data Base Operating Systems*. Operating Systems, LNCS 60, 393–481.
  * Herlihy, M. P., & Wing, J. M. (1990). *Linearizability: A Correctness Condition for Concurrent Objects*. ACM TOPLAS, 12(3), 463–492.

---

### Chapter 09: Durable Memory: Episodic, Semantic & Temporal Decay (L3)
* **Core Systems Invariant:** *The Hierarchical Memory Fabric* — Long-term agent autonomy requires a multi-tier memory fabric that blends fast vector retrieval, structured knowledge graphs, and temporal decay functions with poison-defense.
* **Sections:**
  * **9.1 Beyond Context Limits:** Architecting persistent memory across sessions, weeks, and months.
  * **9.2 Vector Databases & Approximate Nearest Neighbors (ANN):** Embeddings, index topologies (HNSW, IVF-PQ), and distance metrics.
  * **9.3 Structured Knowledge Graphs:** Relational triples, entity linking, and graph query engines for deterministic retrieval.
  * **9.4 Episodic Memory Consolidation:** Transforming raw event streams into structured memories; reflection-triggered consolidation.
  * **9.5 Temporal Decay & Forgetting Functions:** Mathematical modeling of memory decay ($e^{-\lambda \Delta t}$) and importance weighting.
  * **9.6 State Poisoning Defense:** Guarding persistent memory from sleeper indirect injection payloads.
* **Seminal Citations:**
  * Park, J. S., et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior*. UIST 2023.
  * Malkov, Y. A., & Yashunin, D. A. (2018). *Efficient and Robust Approximate Nearest Neighbors Using Hierarchical Navigable Small World Graphs*. IEEE TPAMI, 42(4), 824–836.
  * Tulving, E. (1983). *Elements of Episodic Memory*. Oxford University Press.

---

## PART III: The Agent–World Boundary

### Chapter 10: The Action Interface & Typed Contracts (Model Context Protocol)
* **Core Systems Invariant:** *The Strict Action Contract* — Non-deterministic models must interface with deterministic systems through typed, schema-validated, idempotent protocol contracts.
* **Sections:**
  * **10.1 The System Call of AI:** How models invoke external capabilities; the transition from raw text strings to structured JSON/Protobuf RPC.
  * **10.2 Model Context Protocol (MCP):** Discovery, capability negotiation, resource subscriptions, and tool execution interfaces.
  * **10.3 Ousterhout's Deep Modules in Tool Design:** Designing narrow, deep tool interfaces that maximize capability while minimizing prompt token overhead.
  * **10.4 Type Validation, Schema Coercion & Parse Error Recovery:** Automatic schema validation and self-correcting error feedback.
  * **10.5 Idempotency & Nonce Tracking:** Preventing duplicate executions of state-mutating external APIs.
* **Seminal Citations:**
  * Model Context Protocol Specification (2025/2026). Anthropic & Open Source Consortium.
  * Schick, T., et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. NeurIPS 2023.
  * Fielding, R. T. (2000). *Architectural Styles and the Design of Network-based Software Architectures*. Doctoral dissertation, UC Irvine.

---

### Chapter 11: Isolation, Blast Radius & Information Flow Control
* **Core Systems Invariant:** *Least-Privilege Sandboxing* — An agent executing unverified code must be confined within zero-trust, capability-bounded virtualization sandboxes to restrict its blast radius.
* **Sections:**
  * **11.1 The Agentic Threat Surface:** Indirect prompt injection, jailbreaks, data exfiltration, and malicious tool invocations.
  * **11.2 The Confused Deputy Problem in LLMs:** When untrusted user data co-opts the agent’s privileged tools.
  * **11.3 Virtualization Technologies:** MicroVMs (AWS Firecracker), WebAssembly (Wasm), container sandboxes (gVisor), and lightweight process chroots.
  * **11.4 Information Flow Control (IFC):** Tainting untrusted inputs and preventing high-security state from flowing to external networks.
  * **11.5 Second-Order Injection via Sandbox stdout/stderr:** Sanitizing and bounding tool execution output streams before writing to L1 context.
  * **11.6 Denial-of-Wallet Defense:** Throttling adversarial retry loops and malicious resource exhaustion.
* **Seminal Citations:**
  * Hardy, N. (1988). *The Confused Deputy: (or why I know what you're doing, even if you don't)*. ACM Operating Systems Review, 22(4), 36–38.
  * Agache, A., et al. (2020). *Firecracker: Lightweight Virtualization for Serverless Applications*. NSDI 2020.
  * Greshake, K., et al. (2023). *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection*. ACM Workshop on AISec.
  * Krohn, M., et al. (2007). *Information Flow Control for Standard OS Abstractions*. SOSP 2007.

---

### Chapter 12: Recovery: Deterministic Sagas, Checkpoints & Actor Swarms
* **Core Systems Invariant:** *Deterministic Compensating Sagas* — Because multi-step neural actions cannot participate in 2-phase commits, fault tolerance requires distributed Sagas with statically typed, deterministic compensating rollbacks.
* **Sections:**
  * **12.1 The Impossibility of Atomic Rollbacks:** Why external side-effects (file edits, API mutations) cannot be undone via simple process aborts.
  * **12.2 The Saga Pattern for Agent Workflows:** Structuring trajectories as sequences of local transactions with matching deterministic compensating actions.
  * **12.3 The Actor Model for Swarms:** Modeling agents as asynchronous actors with private state and Erlang-style supervisor trees.
  * **12.4 Trajectory Snapshotting & Fork-on-Error:** Checkpointing execution graphs and branching alternative reasoning paths upon failure.
  * **12.5 Correlated Byzantine Faults in Agent Debates:** Reaching consensus when node hallucinations are correlated by shared pre-training weights.
* **Seminal Citations:**
  * Garcia-Molina, H., & Salem, K. (1987). *Sagas*. ACM SIGMOD Record, 16(3), 249–259.
  * Hewitt, C., et al. (1973). *A Universal Modular ACTOR Formalism for Artificial Intelligence*. IJCAI 1973.
  * Armstrong, J. (2007). *A History of Erlang*. HOPL III.
  * Chandy, K. M., & Lamport, L. (1985). *Distributed Snapshots: Determining Global States of Distributed Systems*. ACM TOCS, 3(1), 63–75.
  * Du, Y., et al. (2023). *Improving Factuality and Reasoning in Language Models through Multiagent Debate*. arXiv:2305.14325.

---

## PART IV: Measurement, Verification & Fleet Economics

### Chapter 13: Trajectory Benchmarking & Environment Evaluation (MLPerf Agents)
* **Core Systems Invariant:** *Environment-Grounded Evaluation* — Static question-answering benchmarks are useless for agents; evaluation requires reproducible, stateful software environments measured by task completion and cost.
* **Sections:**
  * **13.1 The Collapse of Static Benchmarks:** Why MMLU, GSM8K, and ARC fail to measure multi-step systems capability.
  * **13.2 Interactive Benchmark Gyms:** SWE-bench (software engineering), GAIA (general assistants), WebArena (web navigation), and OSWorld (desktop control).
  * **13.3 Systems Metrics:** Pass@k, Success Rate vs. Trajectory Length, Cost-per-Completed-Task, and Time-to-Solution.
  * **13.4 MLPerf Agents:** Standardizing open-source, reproducible systems evaluation across hardware, models, and runtimes; submission format and peer review protocols.
  * **13.5 Benchmark Contamination & Dynamic Evaluation:** Generative environment synthesis and cryptographic holdouts to combat test-set leakage.
* **Seminal Citations:**
  * Jimenez, C. E., et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024.
  * Mialon, G., et al. (2024). *GAIA: A Benchmark for General AI Assistants*. ICLR 2024.
  * Zhou, S., et al. (2024). *WebArena: A Realistic Web Environment for Building Autonomous Agents*. ICLR 2024.
  * MLCommons (2026). *MLPerf Agents: Benchmarking Autonomous AI Systems*. MLCommons Consortium.

---

### Chapter 14: Non-Deterministic Replay & Time-Travel Debugging
* **Core Systems Invariant:** *Deterministic Trace Reconstructibility* — Debugging non-deterministic agents requires recording complete environment interactions to enable deterministic replay and branch inspection.
* **Sections:**
  * **14.1 The Non-Deterministic Debugging Nightmare:** Why identical prompts yield divergent trajectories; the Heisenberg effect of print statements in LLM loops.
  * **14.2 Deterministic Mocking & Recording:** Intercepting model completions, tool responses, timestamps, and random seeds into an immutable trace file.
  * **14.3 Time-Travel Debugging:** Stepping backwards through a 50-step trajectory, inspecting intermediate context and state variables.
  * **14.4 Forking Trajectories:** Modifying a prompt or tool output at Step 12 and re-executing forward to test alternative branches.
  * **14.5 Root-Cause Attribution:** Automated diffing between successful and failed trajectory traces.
* **Seminal Citations:**
  * Geels, D., et al. (2007). *Friday: Global Comprehension for Distributed Replay*. OSDI 2007.
  * Engler, D., et al. (2001). *Bugs as Deviant Behavior: A General Approach to Inferring Errors in Systems Code*. SOSP 2001.
  * King, S. T., et al. (2005). *Debugging Operating Systems with Time-Traveling Virtual Machines*. USENIX ATC 2005.

---

### Chapter 15: Agentic Observability & Distributed Telemetry
* **Core Systems Invariant:** *Trajectory Span Lineage* — Observability must bind model tokens, tool latencies, memory lookups, and dollar costs into a hierarchical trace graph.
* **Sections:**
  * **15.1 The Telemetry Triad for Agents:** Metrics (token rates, error frequencies), Logs (thought traces), and Distributed Traces (spans).
  * **15.2 OpenTelemetry GenAI Semantic Conventions:** Standardizing span attributes for model name, prompt tokens, completion tokens, tool names, and exit codes.
  * **15.3 Cost Attribution per Goal:** Granular accounting of dollar spend per user goal and per subagent.
  * **15.4 Anomaly Detection in Long-Running Workflows:** Detecting loops, semantic drift, and excessive tool retries in real time.
  * **15.5 Distributed Telemetry in Agent Swarms:** Distributed context propagation across asynchronous agent message queues.
* **Seminal Citations:**
  * OpenTelemetry Consortium (2025/2026). *Semantic Conventions for Generative AI Systems*. OpenTelemetry Specification.
  * Sigelman, B. H., et al. (2010). *Dapper, a Large-Scale Distributed Systems Tracing Infrastructure*. Google Technical Report.
  * Sambasivan, R. R., et al. (2011). *So, you want to trace your distributed system? Key design insights from years of provenance research*. CMU-PDL-11-102.

---

### Chapter 16: Fleet Economics, Pareto Frontiers & Hardware Co-Design
* **Core Systems Invariant:** *The Macro-Economics of Autonomy* — As agency scales, the optimization frontier shifts from training-time compute to test-time search efficiency, driving hardware co-design for context memory and sandboxing.
* **Sections:**
  * **16.1 The Macro-Economics of Agentic Workloads:** Shifting expenditure from human labor to autonomous inference hours.
  * **16.2 Test-Time Compute vs. Pre-Training Scale:** Deriving the optimal economic allocation between larger base models and longer deliberative search.
  * **16.3 Energy-Per-Verified-Action:** The true sustainability metric for autonomous systems.
  * **16.4 Heterogeneous Deployment Topologies:** Routing simple reflex steps to on-device Small Language Models (SLMs) and complex reasoning to cloud clusters.
  * **16.5 Hardware Co-Design for Agent Runtimes:** Fast context memory (CXL, HBM), silicon-level microVM startup accelerators, and hardware-assisted Radix prefix caching.
  * **16.6 The Next Decade of AI Systems:** Synthesis across Volume I (The Model), Volume II (The Fleet), Volume III (The Agent), and Volume IV (The Physical Plant).
* **Seminal Citations:**
  * Reddi, V. J. (2026). *Architecting the Agentic AI Systems Stack*. ACM SIGOPS Operating Systems Review, 60(1), 64–74.
  * Patterson, D., et al. (2021). *Carbon Emissions and Large Neural Network Training*. arXiv:2104.10350.
  * Jouppi, N., et al. (2023). *TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings*. ISCA 2023.
  * Hennessy, J. L., & Patterson, D. A. (2019). *A New Golden Age for Computer Architecture*. Communications of the ACM, 62(2), 48–60.

---

## 4-Part Studio Course Lab Mapping (The Agent Engine)

| Part | Lab Assignment | Milestone Project Deliverable |
| :--- | :--- | :--- |
| **Part I** | **Lab 1:** Pure Python State Machine DAG Router<br>**Lab 2:** MCTS Search Engine over Agent Actions<br>**Lab 3:** Trajectory Invariant Circuit Breaker | **Phase 1 Deliverable:** The Core State Graph & Search Engine |
| **Part II** | **Lab 4:** Token LRU Context Compaction Manager<br>**Lab 5:** Radix Tree Prefix Cache Benchmark & Invalidator<br>**Lab 6:** Persistent Episodic Memory & Decay Store | **Phase 2 Deliverable:** The Hierarchical Context & Memory Subsystem |
| **Part III** | **Lab 7:** Strict Type-Safe MCP Tool Server<br>**Lab 8:** Wasm/MicroVM Sandbox Isolation & Second-Order Taint Filter<br>**Lab 9:** Deterministic Saga Rollback & Compensating Engine | **Phase 3 Deliverable:** The Secure Tool & Distributed Saga Runtime |
| **Part IV** | **Lab 10:** 3-Agent Actor Swarm with Capability Downgrades<br>**Lab 11:** Deterministic Record-and-Replay Debugger<br>**Lab 12:** OpenTelemetry Tracing & MLPerf Agents Submission Package | **Phase 4 Deliverable:** Full Agent Engine Evaluated on MLPerf Agents |

---
*End of Master Curriculum Blueprint.*
