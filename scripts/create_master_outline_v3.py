import re
import sys

def main():
    with open("books/vol3/MASTER_TEXTBOOK_OUTLINE_V2.md", "r", encoding="utf-8") as f:
        content = f.read()

    # Define the new V3 Preamble (Lines 1 to ~540)
    preamble_v3 = """# The Stochastic Computer: Master Textbook Curriculum Outline V3
**Agentic Machine Learning Systems: Architecture, Foundations, and Verifiable Engineering**
*A complete chapter-by-chapter architectural blueprint and authoring contract for Chapters 1–18*

This V3 master outline establishes the definitive curriculum, pedagogical charter, and architectural contract for Volume III. It preserves the macro-organizational structure of V2 (7 Parts, 18 Chapters, and the closed-loop trajectory architecture) while executing a fundamental authorial paradigm shift: **transitioning from forced computer architecture/OS metaphors to native, authentic Agentic Machine Learning Systems engineering**. 

The "Stochastic Computer" serves as the high-level conceptual spine across the volume. Once inside each chapter, the text speaks the direct, rigorous language of production ML systems and distributed systems engineering. V2 remains intact; V3 is the active operational standard for all chapter authoring.

---

## Pedagogical Vision & Target Audience Charter

### 1. Who This Textbook Is For: The Senior Undergraduate & Introductory Graduate Student

This textbook is designed specifically for **senior-level undergraduate Computer Science & Computer Engineering (CS/CE) majors** (in courses analogous to CS 152 / CS 162 at UC Berkeley, CS:APP / 15-410 at CMU, or 6.033 / 6.828 at MIT) and **introductory Master's students in Computer Systems and Machine Learning**.

The student is an aspiring AI Systems Engineer. They are not satisfied with superficial "prompt engineering" tutorials, nor are they looking for an abstract NLP linguistics treatise. They want to understand the foundational mechanics of how production-grade autonomous agent runtimes (such as Claude Code, Devin, and SWE-agent) and high-throughput inference serving engines (such as vLLM, SGLang, and TensorRT-LLM) are designed, measured, bounded, and deployed.

### 2. Student Prerequisite Matrix (What the Reader Knows vs. Does NOT Know)

To ensure instructors and authoring agents pitch explanations at the exact cognitive altitude:

* **What the Student ALREADY Knows (Assumed Background):**
  - **Core Computer Architecture:** Registers, ALUs, pipelining, multi-tier CPU cache hierarchies (L1/L2/L3), memory bus latency, DRAM burst access, and basic instruction cycle execution.
  - **Core Operating Systems:** Processes, hardware threads, virtual memory, page tables, memory-mapped files, POSIX system calls (`fork`, `exec`, `pipe`, `ioctl`), file descriptors, and concurrency primitives (mutexes, semaphores, race conditions).
  - **Programming & Software Engineering:** High proficiency in Python and solid experience in a compiled systems language (C, C++, or Rust). Understanding of Abstract Syntax Trees (ASTs), lexing, parsing, and context-free grammars.
  - **Introductory Machine Learning:** Basic linear algebra, dense matrix multiplication, weights, gradients, backpropagation, and the mathematical definition of scaled dot-product attention: $\text{Softmax}(QK^T / \sqrt{d})V$. They have called commercial LLM APIs via Python scripts.

* **What the Student Does NOT Know (What This Book Teaches From First Principles):**
  - They have **never built, profiled, or modified an LLM serving runtime**. Terms like continuous batching, chunked prefill, PagedAttention block tables, and RadixAttention prefix caching are new or vague.
  - They do not understand the **physical memory-bandwidth bottleneck of autoregression**: why single-batch token decode sits on the HBM bandwidth floor ($B=1$), why GPU Tensor Cores sit 98% idle during token generation, and why prompt prefill (GEMM) and token decode (GEMV) have radically different arithmetic intensities.
  - They have never designed an **autonomous agent host runtime**. They do not know how production systems manage multi-turn context staging, prevent prefix cache invalidation, sanitize unbounded subprocess terminal streams via circular ring buffers, enforce capability attenuation, or close verification loops deterministically without human intervention.

### 3. Intellectual Posture: The Hennessy & Patterson + CS:APP Standard

The textbook's tone, pedagogy, and technical depth are anchored directly in two landmark computer systems classics:

1. **Primary Anchor — Hennessy & Patterson (*Computer Architecture: A Quantitative Approach*):**
   - **Quantitative Rigor:** Never claim a subsystem is "fast" or "efficient" without numbers. Derive memory footprints, bandwidth saturation floors, and arithmetic intensities with explicit dimensional units.
   - **The Roofline Model:** Evaluate every computational phase against accelerator compute and memory bandwidth ceilings.
   - **Boxed Worked Examples:** Every major chapter includes step-by-step boxed worked examples (`::: {.callout-note title="Worked Example: [Title]"}`) with concrete hardware numbers (e.g., sizing an H100 parameter shuttle floor or a 128k context memory footprint).
   - **Fallacies and Pitfalls:** Every chapter concludes with an unvarnished analysis of widespread industry misconceptions and subtle systems traps.

2. **Secondary Anchor — Bryant & O'Hallaron (*Computer Systems: A Programmer's Perspective* - CS:APP):**
   - **The Programmer's Systems Interface:** Demystify the system from the programmer's perspective. No magic, no hand-waving, and no black boxes.
   - **Concrete ABI Contracts:** Anchor every boundary in concrete typed schemas, C structs, or Python `@dataclass` definitions before presenting abstract equations.

---

## Core Authoring Directives: The Authentic ML Systems Stance

### 1. Macro-Spine, Not Micro-Roleplay: Dispelling the CPU Halloween Costume

In V2, the textbook frequently fell into an awkward identity crisis: pretending attention heads are ALUs, prompts are instruction registers, tokens are opcodes, and model errors are x86 `#PF` page faults. 

**In V3, this forced roleplay is completely eliminated:**
- **The "Stochastic Computer" is an overarching macro-architectural spine:** It organizes the book's 7 Parts and 18 Chapters (treating the whole autonomous loop as an accountable computing system carrying a task from prompt to verified completion).
- **Inside the Chapters, Speak Native Agentic ML Systems Language:** Call things by their real, industry-standard names: **Large Language Model (LLM), Byte-Pair Encoding (BPE), embedding tables, autoregressive decode loop, logits, temperature scaling, KV cache, prefill phase, decode phase, PagedAttention, RadixAttention prefix trees, inference runtimes (vLLM, SGLang), and host agent controllers (Claude Code, Devin)**.
- **No Defensive Apologies / No Strawmen:** Do not spend pages defensively asserting "the model has no program counter" or "an LLM is not an Intel chip." Senior CS students already know this. State the operational model cleanly, define its mathematical and physical boundaries, and proceed immediately to the systems engineering.

### 2. The Grounding Mental Model: The Two-Tier Agentic System Topology

Every chapter evaluates its mechanisms in the context of the real-world two-tier architecture that powers modern agentic computing:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              THE DUAL AGENTIC SYSTEM TOPOLOGY                          │
├─────────────────────────────────────────┬──────────────────────────────────────────────┤
│      HOST AGENT RUNTIME (CPU/Host)      │        NEURAL SERVING ENGINE (GPU/HBM)       │
├─────────────────────────────────────────┼──────────────────────────────────────────────┤
│ • Task State & Trajectory Graph         │ • Model Weights in HBM (FP16 / FP8 / INT4)   │
│ • Context Staging & Prefix Alignment    │ • PagedAttention Physical Block Allocation   │
│ • Sandboxed Subprocesses (Docker, Fire) │ • RadixAttention Prefix Cache Index Tree     │
│ • Tool Execution & Observation Tailing  │ • Continuous Batching & Chunked Prefill      │
│ • Ring Buffers & ANSI Stream Stripping  │ • Tensor Parallelism (TP) Communication      │
│ • Deterministic Verifiers (AST, Linters)│ • Grammar-Constrained Logit Masking (DFAs)   │
│ • Token Accounting & Budget Escalations │ • Prefill (GEMM) vs Decode (GEMV) Roofline   │
└─────────────────────────────────────────┴──────────────────────────────────────────────┘
```

1. **The Host Agent Runtime (The Supervisor on CPU):** A long-running process that owns task state, assembles context windows, preserves prompt prefix stability, executes tools in sandboxes, tails terminal streams through circular ring buffers, and executes deterministic verification gates.
2. **The Neural Serving Engine (The Worker on Accelerator Silicon):** A high-performance inference engine (e.g., vLLM, SGLang, TensorRT-LLM) managing GPU High-Bandwidth Memory (HBM), compiling grammar DFAs into GPU bitmasks, executing compute-bound prefill (GEMM) and memory-bound decode (GEMV), and reusing KV-cache blocks via Radix trees.

---

### 3. The 4-Step Pedagogical Scaffolding Ladder (Concept First, Progressive Grounding)

To maximize cognitive economy and clarity for undergraduate seniors, every substantive section must follow this strict 4-step pedagogical progression:

1. **The Governing Systems Dilemma / Observable Failure:** Open with a concrete, real-world failure mode an engineer can immediately visualize (e.g., why subword splitting causes an AST syntax error; why an HTTP `200 OK` is not task completion; why decoding a 70B model on an H100 cannot exceed 48 tokens/second; why unbuffered terminal output crashes the context window).
2. **The Systems Intuition (Physical Mental Model First):** Build an intuitive, physical mental model *before* displaying mathematics. Use vivid, memorable analogies:
   - *The Sports Car Engine vs. Narrow Straw:* The GPU has massive compute horsepower (Tensor Cores), but a narrow memory straw (HBM bus). In prefill, you pour a bucket of tokens through the engine all at once (compute bound). In decode, you have to pump all 70 billion parameters through the straw just to emit a single token (memory bound).
   - *The Mars Multiple-Choice Exam:* Forcing an impossible schema on a model is like giving a student an exam question where the true answer is D, but only options A, B, and C exist on the bubble sheet. The student is forced to guess an untruth with 100% confidence.
3. **The Concrete Systems Artifact:** Present a typed Python `@dataclass`, C struct, AST trace, or structured comparison table so the student sees the exact data layout before abstract derivations.
4. **Grounded Systems Math & Physical Provenance:** Walk through quantitative equations step-by-step. Adhere strictly to **The Law of Constant Provenance**: never introduce a numerical constant or factor without explaining its physical hardware origin.

---

### 4. The Engineer's Napkin Estimate

Teach students to make sound architectural decisions before running complex benchmarks. At key decision points across the seven Parts, include an order-of-magnitude napkin calculation: state the workload assumptions, write the governing quantities with units, calculate a plausible bound, identify the dominant term, and explain what measurement would validate it.

Illustrative decisions include:
- Sizing the single-batch decode parameter shuttle floor (Part I).
- Calculating stranded HBM during long-running tool waits (Part II).
- Sizing process startup vs. execution latency across container sandboxes (Part III).
- Estimating concurrent trajectory capacity under memory and CPU core limits (Part IV).
- Sizing verifier throughput required to match training rollout generation (Part V).
- Calculating dollar cost per verified, accepted SWE-bench task (Part VI & VII).

---

### 5. Strict Subsystem Ownership & Negative Scope Invariants (Zero Concept Bleed)

To build a clean, modular 18-chapter textbook, every topic, formula, and mechanism has exactly **ONE primary architectural home**. Authoring agents and human contributors must strictly observe these jurisdictional boundaries:

* **Part I: Inference Serving & Deliberation (Chapters 02–03)**
  - **Chapter 02 (The Model Invocation Boundary):** The single model call ($H=1$, $A=0$). Tokenization, embedding lookup, autoregressive serving loop, Roofline prefill/decode physics, grammar-constrained logit masking, typed client-server contracts, in-memory static verification, and interface benchmarks.
    * 🛑 *Strict Negative Scope:* Do NOT derive multi-layer KV-cache memory formulas ($2LHd$) or 128k GPU memory crashes (evicted to Ch 05). Do NOT execute dynamic subprocesses or test runners (deferred to Ch 07). Do NOT discuss container sandboxing (deferred to Ch 08). Do NOT discuss Tensor Parallelism sharding (deferred to Ch 17).
  - **Chapter 03 (Inference-Time Deliberation):** Test-time compute allocation over reasoning ($H > 1$). Extended sequential thinking tokens, Best-of-$N$, Process Reward Models (PRMs), Outcome Reward Models (ORMs), verifier mechanics, Goodhart's law, and search budgets.
    * 🛑 *Strict Negative Scope:* Deliberation occurs *prior* to environmental tool actuation. Do NOT execute external tool calls or mutate host state (deferred to Ch 07).

* **Part II: Context Memory & Serving State (Chapters 04–06)**
  - **Chapter 04 (Context-Window Working Memory):** The host-side logical working set. Prompt context assembly, ordering prompt blocks for prefix cache stability, lossy vs. lossless compaction, and context invalidation under file mutation.
    * 🛑 *Strict Negative Scope:* Dismantle the "L1 cache" metaphor. Do NOT explain GPU physical PagedAttention block tables (deferred to Ch 05).
  - **Chapter 05 (The KV-Cache Hierarchy):** The serving-engine physical GPU memory. Physical KV tensor geometry ($2LHd$), MHA vs. GQA vs. MQA, the 128k context GPU crash worked example, PagedAttention block tables, RadixAttention prefix trees, chunked prefill (Sarathi-Serve), and the Tool-Wait memory tax.
  - **Chapter 06 (Persistent External Memory):** Durable storage beyond accelerator HBM. Authoritative source truth (Git, filesystems) vs. derivative search indexes (BM25, dense vector ANN/HNSW, AST code property graphs), and cache invalidation on code mutation.

* **Part III: Tool Actuation & Sandboxing (Chapters 07–08)**
  - **Chapter 07 (Peripherals & Tool Actuation):** Typed RPC tool dispatch, Model Context Protocol (MCP), observation engineering (circular ring buffers, ANSI escape stripping, headless tailing, stdout/stderr truncation), and idempotent execution under timeouts.
  - **Chapter 08 (Virtualization & Sandboxing):** Host protection boundaries, threat models (prompt injection, destructive commands), microVMs (Firecracker), gVisor, Linux namespaces, cgroups, seccomp-bpf syscall filtering, OverlayFS copy-on-write, and default-deny network egress firewalls.

* **Part IV: The Agent Operating System (Chapters 09–11)**
  - **Chapter 09 (The Agent OS Control Plane):** Long-horizon trajectory lifecycles, Agent Control Blocks (ACBs), state machine transitions (`READY`, `RUNNING`, `BLOCKED_ON_TOOL`), cooperative preemption, and priority scheduling.
  - **Chapter 10 (State, Persistence & Storage):** Write-Ahead Logging (WAL) for trajectory events, intent/effect ledgers, checkpointing, and deterministic replay debugging.
  - **Chapter 11 (Fault Tolerance, Compensation & Sagas):** Distributed transactions across non-transactional effectors, compensating actions (undoing partial git commits or API calls), Saga orchestrators, and idempotent recovery.

* **Part V: The Policy Compiler (Chapters 12–14)**
  - **Chapter 12 (Trajectory Data & Feedback):** The data flywheel—harvesting production execution traces, filtering by deterministic test passage, and curating regression fixtures.
  - **Chapter 13 (Supervised Policy Adaptation):** Distilling trajectory protocols into model weights via SFT; specializing models for tool syntax while maintaining external verification gates.
  - **Chapter 14 (Reinforcement Learning with Verifiable Rewards):** RLVR training loops using deterministic software environments (unit tests, compilers) as reward oracles; avoiding reward hacking and specification gaming under exploration.

* **Part VI: Distributed Fleets & Operations (Chapters 15–17)**
  - **Chapter 15 (Multi-Agent Fleets & Coordination):** Asynchronous distributed workers, task dependency DAGs, shared state contention on git repositories, message queues, and evaluating multi-agent systems against token-matched single-agent baselines.
  - **Chapter 16 (Distributed Observability & Evaluation):** OpenTelemetry distributed tracing across agent trajectories, SWE-bench execution mechanics, and failure mode taxonomies.
  - **Chapter 17 (Performance, Cost & Fleet Economics):** Capacity planning, Tensor Parallelism ($TP \ge 2$) multi-GPU sharding, space-time memory occupancy ($O_{\text{mem}}$), serving economics, and optimizing for dollar cost per verified accepted deliverable.

* **Part VII: System Synthesis (Chapter 18)**
  - **Chapter 18 (Designing the Stochastic Computer):** End-to-end architectural synthesis tracing a complete production task (e.g., resolving a real-world multi-file SWE-bench issue) through every contract, memory tier, sandbox boundary, saga recovery, and verification gate.

---

### 6. The Systems Engineering Translation Lexicon

Use this lexicon to replace vague language with precise mechanisms and systems questions:

| Concept | Explain the mechanism first | Then analyze the systems implication where relevant |
|:---|:---|:---|
| **Prompt / Context Window** | A bounded sequence of discrete tokens staged in host memory for one invocation. | Prompt layout directly dictates RadixAttention prefix-cache hit rates ($H_{\text{prefix}}$). |
| **Tokenization / BPE** | A statistical subword byte serializer mapping strings to integer IDs. | Fractured tokens create AST impedance mismatches and JSON serialization overhead. |
| **Autoregressive Generation** | Iteratively evaluating conditional probabilities along an irreducibly serial dependency chain. | Dictates single-trajectory latency ($O(K)$ forward passes); bounds throughput at $B=1$. |
| **Prefill Phase** | Parallel processing of $M$ staged prompt tokens via Matrix-Matrix multiplication (GEMM). | Compute-bound on Tensor Cores ($I \gg I_{\text{sat}}$); dictates Time-to-First-Token (TTFT). |
| **Decode Phase** | Serial generation of $K$ tokens via Matrix-Vector multiplication (GEMV). | Memory-bandwidth bound on HBM parameter shuttling ($I \ll I_{\text{sat}}$); dictates ITL. |
| **KV Cache** | Intermediate key/value tensor activations retained in GPU HBM to avoid recomputing attention. | Footprint grows linearly with sequence length; dictates serving concurrency and tool-wait costs. |
| **Grammar Constrained Decoding** | Compiling schemas/regex into FSMs to mask illegal token logits to $-\infty$ during decode. | Guarantees syntactic validity; executed via GPU bitmasks; leaves operational truth unverified. |
| **Zero Ambient Authority ($A=0$)** | An unprivileged execution boundary where emitted text alters zero host bits. | Candidate proposals reside in host memory escrow until external gates verify them. |
| **Tool Actuation** | Mediated RPC dispatch from the host runtime to an external process or service. | Non-model latency dominates wall-clock time; requires ring-buffer tailing and timeouts. |
| **Sandboxing** | Process isolation via microVMs, namespaces, cgroups, and seccomp-bpf filters. | Contains destructive commands and prompt injection; prevents unconstrained host mutation. |

---

### 7. The MLSysIM Heterogeneous Reference Platform Specification

To maintain quantitative consistency and eliminate conflicting hand-typed numbers, all reference platforms, hardware numbers, Rooflines, arithmetic intensities, memory footprint equations, and latency envelopes must derive programmatically from `mlsysim.Agents.Platforms` and `mlsysim.Systems.Nodes` via Pint-typed Python LEGO cells.

#### The Heterogeneous Agent Platform Architecture

Agentic machine learning systems run on **heterogeneous computing nodes** where the Accelerator and the Host CPU share responsibility for end-to-end task execution:

1. **Accelerator Tier (The Neural Core):**
   - Executes dense prefill (GEMM) and serialized autoregressive decode (GEMV).
   - Governed by High-Bandwidth Memory (HBM) interface bandwidth ($\text{BW}_{\text{mem}}$), tensor compute throughput ($P_{\text{peak}}$), and device memory capacity.
   - Canonical datacenter baseline in `mlsysim`:
     * `Hardware.Cloud.H100`: NVIDIA H100 SXM5 ($80\text{ GiB}$ HBM3, $3.35\text{ TB/s}$ memory bandwidth, $989\text{ TFLOP/s}$ dense FP16/BF16, $I_{\text{sat}} = 295.2\text{ FLOP/byte}$, NVLink 4 bidirectional bandwidth $900\text{ GB/s}$).
     * `Hardware.Cloud.B200`: NVIDIA B200 ($192\text{ GiB}$ HBM3e, $8.0\text{ TB/s}$ memory bandwidth, $2{,}250\text{ TFLOP/s}$ dense FP16, NVLink 5 bidirectional bandwidth $1{,}800\text{ GB/s}$).
   - Canonical workstation baseline in `mlsysim`:
     * `Hardware.Workstation.MacBookM3Max`: Apple M3 Max ($128\text{ GiB}$ Unified LPDDR5X, $400\text{ GB/s}$ bandwidth shared between CPU and GPU, $14.2\text{ TFLOP/s}$ FP16, internal unified SoC fabric).

2. **Host CPU & System Tier (The Agent Runtime & Sandboxed Environment):**
   - Executes the agent control plane, tokenization, serialization, sandbox isolation (microVM/cgroup lifecycles), and deterministic verification (compilers, test runners, linters, Git operations).
   - Canonical datacenter host nodes in `mlsysim`:
     * `Systems.Nodes.DGX_H100`: Official NVIDIA DGX H100 server: Dual Intel Xeon Platinum 8480C (112 physical cores, 224 threads, 2.0 GHz base / 3.8 GHz boost, $2\text{ TiB}$ DDR5-4800 RAM across 16 channels yielding $614\text{ GB/s}$ peak host memory bandwidth, $30.72\text{ TB}$ NVMe storage, and 8× ConnectX-7 400 Gbps InfiniBand ports).
     * `Systems.Nodes.HGX_H100_EPYC`: High-density agent evaluation node: Dual AMD EPYC 9654 (192 physical x86-64 Zen4 cores, 384 hardware threads, 2.4 GHz base / 3.7 GHz boost, $1.5\text{ TiB}$ DDR5-4800 RAM across 24 channels yielding $460\text{ GB/s}$ sustained host memory bandwidth, PCIe Gen5 $\times 16$ interconnect to each accelerator, and 4× Gen5 NVMe SSD arrays yielding $56\text{ GB/s}$ aggregate I/O).
   - Canonical workstation host tier in `mlsysim`:
     * `Systems.Nodes.Workstation_M3Max`: Apple M3 Max 16-Core CPU (12 Performance cores + 4 Efficiency cores), sharing the $400\text{ GB/s}$ unified memory fabric directly with the GPU, eliminating PCIe host-to-device data transfer penalties.

---

### Subsystem Dependency Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │   Introduction: The Stochastic Computer      │
                    │   (The Closed-Loop Trajectory Architecture)  │
                    └──────────────────────┬───────────────────────┘
                                           │
                        Demands a computational core
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │   Part I: Inference Serving & Deliberation   │
                    │   (Model Invocation, Then Deliberation)      │
                    └──────────────────────┬───────────────────────┘
                                           │
                        Extra steps and branches demand selected state
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │   Part II: Context Memory and Serving State  │
                    │   (Logical Context, KV State, Persistence)   │
                    └──────────────────────┬───────────────────────┘
                                           │
                        Compute + memory in a box cannot act
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │   Part III: Tools and Sandboxing             │
                    │   (Typed Actions, Observations, Isolation)   │
                    └──────────────────────┬───────────────────────┘
                                           │
                        Side-effecting tools demand supervision
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │   Part IV: The Agent Operating System        │
                    │   (Lifecycle, Records, Recovery)            │
                    └──────────────────────┬───────────────────────┘
                                           │
                        Runtime reveals generic base model flaws
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │   Part V: The Policy Compiler                │
                    │   (Evidence, Supervised Adaptation, RLVR)   │
                    └──────────────────────┬───────────────────────┘
                                           │
                        Delegation may help only under measured budgets
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │   Part VI: Distributed Fleets & Operations   │
                    │   (Coordination, Evaluation, Economics)      │
                    └──────────────────────┬───────────────────────┘
                                           │
                        End-to-end integration & production hardening
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │   Part VII: System Synthesis                 │
                    │   (Capstone Architecture)                   │
                    └──────────────────────────────────────────────┘
```

---

### The 18-Chapter Causal Chain (Provocation $\to$ Resolution $\to$ Next Requirement)

Every chapter transition in this book is governed by an explicit handoff:

| Ch | Title | Solved Problem | Exposed Limitation (The Provocation) | Next Ch Handoff |
| :--- | :--- | :--- | :--- | :--- |
| **01** | *The Stochastic Computer* | Defines the whole-system trajectory loop, 5-part task contract, and Fail-Plausible fault model. | We know a trajectory requires a computer, but what is its actual processing engine? | $\to$ **Ch 02** |
| **02** | *The Model Invocation Boundary* | Explains one model invocation: tokenization, prefill/decode Roofline physics, contracts, grammars, and status envelopes. | One candidate sequence may not provide enough evidence for a correct decision. | $\to$ **Ch 03** |
| **03** | *Inference-Time Deliberation* | Allocates additional inference compute to alternatives, verification, and reasoning search under a budget. | Branches and observations generate more state than can remain visible in context. | $\to$ **Ch 04** |
| **04** | *Context-Window Working Memory* | Selects the logical working set staged for the next invocation, optimizing prompt layout for prefix caching. | Selected tokens still have a physical attention-state footprint in the serving system. | $\to$ **Ch 05** |
| **05** | *The KV-Cache Hierarchy* | Allocates, shares, and reclaims physical attention state ($2LHd$, PagedAttention, Radix trees) for active requests. | A serving cache does not preserve task knowledge across sessions. | $\to$ **Ch 06** |
| **06** | *Persistent External Memory* | Stores, retrieves, and invalidates durable knowledge and records through explicit interfaces. | Compute plus state still cannot observe or change the environment without controlled interaction. | $\to$ **Ch 07** |
| **07** | *Peripherals & Tool Actuation* | Turns a candidate into a parsed, permitted, dispatched action with an observable result (MCP, ring buffers). | Valid tool requests can still reach an unsafe execution environment. | $\to$ **Ch 08** |
| **08** | *Virtualization & Sandboxing* | Bounds authority and effects using capabilities, microVMs, and untrusted-input handling. | Isolated calls still require supervision across time. | $\to$ **Ch 09** |
| **09** | *The Agent Operating System Control Plane* | Owns trajectory lifecycle, scheduling, budgets, pending work, cancellation, and completion (ACB). | Volatile runtime state is lost in a crash. | $\to$ **Ch 10** |
| **10** | *State, Persistence & Storage* | Records intent, decisions, observations, and confirmed effects for reconstruction and audit (WAL). | Recorded history cannot undo every external effect. | $\to$ **Ch 11** |
| **11** | *Fault Tolerance, Compensation & Sagas* | Recovers or escalates after partial, possibly irreversible effects. | Repeated failures expose policy gaps that runtime checks alone cannot remove. | $\to$ **Ch 12** |
| **12** | *Trajectory Data & Feedback* | Turns validated execution evidence into curated learning and evaluation fixtures. | Curated examples do not yet change the model. | $\to$ **Ch 13** |
| **13** | *Supervised Policy Adaptation* | Adapts proposal behavior from demonstrations while leaving enforcement external (SFT distillation). | Demonstrations do not explore unobserved solutions. | $\to$ **Ch 14** |
| **14** | *Reinforcement Learning (RLVR)* | Tests when protected, verifiable outcomes support policy improvement through exploration. | A better individual policy does not settle coordination at scale. | $\to$ **Ch 15** |
| **15** | *Multi-Agent Fleets & Coordination*| Measures when delegation beats a single agent under matched resources and shared-state constraints. | More actors create more causal paths and harder failures to explain. | $\to$ **Ch 16** |
| **16** | *Distributed Observability & Evaluation* | Connects task acceptance to OpenTelemetry traces, controlled evaluation (SWE-bench), and statistical evidence. | A reliable system must still meet cost and latency limits. | $\to$ **Ch 17** |
| **17** | *Performance & Cost Engineering* | Optimizes accepted tasks under whole-trajectory latency, capacity, and spending budgets ($TP \ge 2$, economics). | Local choices must be synthesized into one coherent design. | $\to$ **Ch 18** |
| **18** | *System Synthesis: The Stochastic Computer* | Traces a complete SWE-bench task through every contract, boundary, recovery path, and acceptance check. | **The Book Concludes:** Remaining limits become explicit research problems. | Complete |

---

# Detailed Chapter-by-Chapter Curricular Blueprints

---
"""

    # Define the new Chapter 02 blueprint for V3
    ch02_v3 = """## Part I: Inference Serving & Deliberation

### Chapter 02: The Model Invocation Boundary

- **Subtitle:** *The Foundation Model Execution Engine*
- **Core Takeaway:** *A model call is an unprivileged, non-deterministic inference evaluation that transforms staged token sequences into candidate continuations via compute-bound prefill and memory-bandwidth-bound decode; the agent runtime governs this boundary through typed contracts, decode-time syntax constraints, and external verification.*
- **Governing Systems Question:** *What happens computationally during a single model invocation, how does hardware constrain its latency and throughput, and what software contract must the runtime enforce to govern it safely?*
- **Curricular Role in Volume III:** *"Here is the Processing Element."* Introduces the Large Language Model as an unprivileged execution engine through the dual lens of Computer Architecture (accelerator memory bandwidth, Roofline prefill/decode physics, HBM parameter shuttle) and Operating Systems (zero ambient authority, typed client-server contracts, status envelopes, in-memory AST verification).
- **Pedagogical Stance:** Speak native, authentic Agentic Machine Learning Systems language. Ground the execution model in the physical realities of modern serving engines (vLLM, SGLang, TensorRT-LLM) and host orchestrators (Claude Code, Devin). Avoid defensive apologies and artificial CPU roleplay (no ALUs, no instruction registers, no x86 `#PF` page faults). Use the 4-step pedagogical arc (Dilemma → Intuition → Code/Artifact → Math) with Hennessy & Patterson-style boxed worked examples.

#### The Curricular Compass (Where We Are in the 18 Chapters)

```
[SYSTEMS STATE AT CHAPTER 02]:
- Subsystem Under Construction: Part I, Chapter 02 (The Model Invocation Boundary).
- Computational Scope: Strictly an atomic, single invocation (H=1, S=staged, A=0, C=external).
- Subsystems Active: Only Chapter 01 (The Trajectory Closed-Loop Architecture).
- Subsystems NOT YET BUILT (STRICTLY FORBIDDEN TO ASSUME OR EXPLAIN IN CHAPTER 02):
  * Chapter 03: Inference-Time Deliberation (Search trees, Best-of-N, MCTS, PRMs, revisable plans).
  * Chapter 04: Context-Window Working Memory (Logical context assembly, compaction, lost-in-the-middle).
  * Chapter 05: The KV-Cache Hierarchy (PagedAttention virtual block tables, Radix prefix trees, DRAM swapping).
  * Chapter 06: Persistent External Memory (Git index traversal, vector databases, freshness).
  * Chapter 07: Peripherals & Tool Actuation (Subprocess dispatch, stdout/stderr streaming ring buffers, MCP).
  * Chapter 08: Virtualization & Sandboxing (MicroVMs, Firecracker, OverlayFS, cgroups, seccomp).
  * Chapters 09-11: The Agent OS (ACB lifecycle, WAL event sourcing, Sagas).
  * Chapters 12-14: The Policy Compiler (Trajectory harvesting, SFT distillation, RLVR).
  * Chapters 15-18: Distributed Fleets, Observability, Fleet Economics, Capstone Synthesis.
```

#### Purpose {.unnumbered .unlisted}

**The Core Question:** *How do software engineers treat a statistical, non-deterministic neural network as an unprivileged, dependable processing element within an autonomous software system?*

**Why It Matters:** *Traditional computer systems rely on deterministic, fail-stop processors: instructions execute exactly as written, invalid memory accesses trigger hardware traps, and execution state is inspectable. A foundation model violates all of these invariants. It evaluates probability distributions over discrete subword integers, emits candidate strings without modifying host state, and cannot verify whether its own output is correct. Treating the model as a conversational partner leads to silent corruptions, resource exhaustion, and security failures. Building reliable agent infrastructure requires formalizing the model invocation boundary: bounding its token and latency budgets, understanding the memory-bandwidth physics of single-batch autoregressive decode, constraining output syntax at decode time via grammar logit masking, and wrapping candidate proposals in deterministic host verification.*

::: {.callout-learning-objectives}

- Contrast the execution model of a classical deterministic CPU with that of an unprivileged generative model operating under Zero Ambient Authority ($A=0$).
- Explain Byte-Pair Encoding (BPE) as statistical data serialization, demonstrating the impedance mismatch between statistical subwords and programming language ASTs.
- Trace the autoregressive execution loop, showing why next-token generation forms an irreducibly serial causal dependency chain on accelerator memory ($O(K)$ serialization).
- Diagnose invocation latency and throughput using the Accelerator Roofline Model, quantitatively distinguishing compute-bound prefill (GEMM) from memory-bandwidth-bound decode (GEMV) at $B=1$.
- Explain grammar-constrained decoding via decode-time logit masking at the execution surface, distinguishing syntactic validity from semantic correctness.
- Specify a typed model invocation contract with explicit token/deadline limits, streaming early cancellation (`RST_STREAM`), and a normalized status envelope.
- Decouple statistical sequence likelihood and text fluency from operational truth, defining the host runtime's external in-memory verification perimeter.
- Evaluate invocation interface designs (free-form markdown, JSON schema, and search/replace diffs) by syntax error rates, token overhead, latency, and verified task completion.

:::

#### Section 2.1: The Model Invocation Boundary [stage-setter]
- **Heading & Anchor:** `## The Model Invocation Boundary {#sec-vol3-processor-role}`
- **The Single Key Point:** The foundation model is an unprivileged inference engine that maps staged token sequences to candidate probability distributions; the host agent runtime is the authoritative supervisor that owns context staging, execution limits, and action verification.
- **Pedagogical Arc:**
  1. *The Systems Contrast:* Classical deterministic CPU (fixed ISA, registers, MMU hardware traps on segfaults) vs. Generative Model (parameterized function approximator, weights $\Theta$, no registers, no hardware traps).
  2. *Zero Ambient Authority ($A=0$):* Emitting `DROP TABLE users;` modifies zero bytes on disk. Candidate strings reside in host memory escrow until an external supervisor validates and executes them.
  3. *The Rosetta Stone Table (@tbl-vol3-ml-systems-rosetta):* Translating ML abstractions (prompts, decode loops, logits, hallucinations, tool calls, HTTP 200) into systems architecture primitives.
  4. *Fail-Stop vs. Fail-Plausible:* The epistemic gap between training likelihood and operational truth. Why models emit fluent, syntactically plausible, completely broken code with high confidence.
  5. *The Two-Tier Architecture (@fig-stochastic-processor-core):* Host Agent Runtime (supervisor on CPU) $\leftrightarrow$ Inference Serving Daemon (vLLM/SGLang/API) $\leftrightarrow$ Neural Core (GPU/accelerator silicon).
  6. *The Delivery Fallacy:* HTTP 200 OK $\neq$ task success. An HTTP 200 verifies only that the socket stayed open and the decode loop completed normally; it guarantees zero syntactic validity or operational truth.
- **What NOT to Cover (Negative Scope):**
  - 🛑 **DO NOT** discuss microVMs, Firecracker, containers, or test execution harnesses (Deferred to Chapter 08).
  - 🛑 **DO NOT** derive the Roofline model or hardware memory bandwidth tables (Deferred to Section 2.4).
  - 🛑 **DO NOT** discuss multi-turn deliberation, search trees, or prompt retries (Deferred to Chapter 03).
  - 🛑 **DO NOT** discuss PagedAttention block tables or host memory swapping (Deferred to Chapter 05).
- **Causal Bridge to 2.2:** Before data can cross into this inference engine, how must it be formatted, discretized, and staged?

#### Section 2.2: Discrete Tokenization and Sequence Representation [core]
- **Heading & Anchor:** `## Discrete Tokenization and Sequence Representation {#sec-vol3-processor-tokenization}`
- **The Single Key Point:** A tokenizer is a statistical byte serializer between strings and integer indices; its subwords clash with programming language ASTs and introduce a significant serialization tax on structured tool calls.
- **Pedagogical Arc:**
  1. *The Lexing Boundary:* Why words fail on open-vocabulary code ($OOV$), why characters cause an unacceptable $16\times$ explosion in attention compute ($O(S^2)$), and how BPE subwords achieve the systems sweet spot.
  2. *The Accelerator Mechanism:* How integer token IDs act as memory row indices gathering vectors from embedding matrix $\mathbf{W}_{\text{embed}} \in \mathbb{R}^{|\mathcal{V}| \times d_{\text{model}}}$ into GPU SRAM.
  3. *The AST Impedance Mismatch (@fig-token-ast-mismatch):* Grammar-aware compiler lexers vs. statistical subwords. Worked Example: `def get_user_id():` unindented (token 755) vs. indented (token 220 + 711). In GPU memory, Row 755 and Row 711 are completely distinct vectors with zero shared identity.
  4. *The Serialization Tax (@tbl-token-compression-ratios):* Empirical measurement of why JSON tool calls inflate tokens by 30%–50% due to whitespace, punctuation, and backslash escaping (`\"`, `\n`).
  5. *Context Ceilings:* The sequence length constraint $S = M + K \le S_{\max}$. Truncation hazards when generation reaches $K_{\max}$, and the Quarantining Invariant.
- **What NOT to Cover (Negative Scope):**
  - 🛑 **DO NOT** derive physical KV cache memory equations ($2LHd$) or multi-layer GPU memory footprint (Deferred to Chapter 05).
  - 🛑 **DO NOT** discuss PagedAttention virtual memory, block allocation tables, or page swapping (Deferred to Chapter 05).
  - 🛑 **DO NOT** discuss prompt compaction, retrieval, or vector databases (Deferred to Chapter 04 & 06).
  - 🛑 **DO NOT** derive GEMM vs. GEMV arithmetic intensity or Roofline models (Deferred to Section 2.4).
- **Causal Bridge to 2.3:** Once staged as an integer token array in GPU memory, how does the model compute its next token?

#### Section 2.3: Autoregressive Generation and the Serving Loop [core]
- **Heading & Anchor:** `## Autoregressive Generation and the Serving Loop {#sec-vol3-processor-autoregressive}`
- **The Single Key Point:** Generating a sequence requires an iterative serving loop that repeatedly evaluates the model forward pass and samples a token, creating an irreducibly serial causal dependency along one output path.
- **Pedagogical Arc:**
  1. *Autoregressive Factorization:* Decomposing sequence probability $P(y_{1:K}\mid x) = \prod_{t=1}^K P(y_t\mid x, y_{<t})$.
  2. *The Execution Cycle:* An atomic forward pass emits logits $\mathbf{z}_t \in \mathbb{R}^{|\mathcal{V}|}$; the serving loop applies temperature scaling $\tau$, samples $y_t$, appends to the sequence, checks stop delimiters, and repeats.
  3. *The Causal Serialization Boundary:* Why token $t$ depends strictly on token $t-1$. Contrast with superscalar CPU out-of-order execution: no speculative bypass can evaluate step 50 before step 49 is sampled along a single trajectory.
  4. *Sampling on the Vocabulary Simplex:* Temperature parameter $\tau$ scaling logits before softmax ($\tau \to 0$ collapses to greedy argmax; high $\tau$ disperses entropy). Top-$p$ (nucleus) and top-$k$ truncation filters. Worked Example: Tracing temperature scaling on a 3-token logit vector.
  5. *Stop Condition Monitoring:* How the inference engine detects completion: `EOS` tokens, user-defined stop delimiters (`\n````, `</tool_call>`), and hard token ceilings ($K_{\max}$).
- **What NOT to Cover (Negative Scope):**
  - 🛑 **DO NOT** derive the Roofline model or hardware memory bandwidth equations (Deferred to Section 2.4).
  - 🛑 **DO NOT** discuss continuous batching schedulers, iteration-level scheduling, or chunked prefill (Deferred to Chapter 05 & 17).
  - 🛑 **DO NOT** discuss multi-turn conversation memory or agent scratchpads (Deferred to Chapter 04 & 09).
- **Causal Bridge to 2.4:** What are the physical hardware constraints that govern the speed and cost of this autoregressive execution loop?

#### Section 2.4: Prefill, Decode, and Accelerator Hardware Physics [core]
- **Heading & Anchor:** `## Prefill, Decode, and Accelerator Hardware Physics {#sec-vol3-processor-cost}`
- **The Single Key Point:** Model invocation latency is split between compute-bound prompt prefill and memory-bandwidth-bound token decode; single-batch agent trajectories are physically bounded by the HBM parameter shuttle.
- **Pedagogical Arc:**
  1. *The Physical Mental Model:* The Sports Car Engine vs. Narrow Straw analogy. Accelerators possess massive compute capacity (hundreds of TFLOPs), but finite memory bandwidth (a few TB/s from HBM).
  2. *The Two Computational Phases:*
     - *Prefill Phase (Prompt Processing):* Parallel processing of $M$ prompt tokens. Matrix-Matrix multiplication (GEMM). Compute-bound on Tensor Cores. High arithmetic intensity ($I \gg I_{\text{sat}}$). Governs Time-to-First-Token (TTFT).
     - *Decode Phase (Token Generation):* Serial generation of $K$ tokens. Matrix-Vector multiplication (GEMV). Memory-bandwidth bound on HBM shuttling weights. Low arithmetic intensity ($I \ll I_{\text{sat}}$). Governs Inter-Token Latency (ITL).
  3. *The HBM Parameter Shuttle Floor:* Sizing single-batch decode ($B=1$). To generate one token on a 70B parameter model in FP16 ($140\text{ GB}$ of weights), the GPU must read all $140\text{ GB}$ across the memory bus. On an NVIDIA H100 SXM5 ($3.35\text{ TB/s}$), the absolute physical latency floor is:
     $$T_{\text{decode, token}} \ge \frac{140\text{ GB}}{3.35\text{ TB/s}} \approx 41.8\text{ ms} \implies \approx 23.9\text{ tokens/second}$$
     In FP8 ($70\text{ GB}$), the floor drops to $\approx 20.9\text{ ms/token} \approx 47.8\text{ tok/s}$.
  4. *The Radical Asymmetry of Agent Workloads:* Contrast standard conversational chatbots ($M \approx 500, K \approx 500$) with autonomous coding agents ($M \approx 40,000$ prompt tokens from repo context and linter logs vs. $K \approx 80$ tokens for a single tool call). Agent serving is 90%+ prefill-bound by token count!
  5. *End-to-End Latency Decomposition:*
     $$T_{\text{invocation}} = T_{\text{prep}} + T_{\text{queue}} + T_{\text{prefill}}(M) + \sum_{t=1}^K T_{\text{decode}}(t)$$
  6. *Single-Batch Latency vs. Multi-Tenant Batching:* Why a single agent cannot increase its batch size ($B=1$), while multi-tenant serving runtimes batch multiple concurrent agents to amortize weight transfers and shift decode toward compute saturation.
- **What NOT to Cover (Negative Scope):**
  - 🛑 **DO NOT** explain virtual memory page tables, PagedAttention block allocation, or DRAM swapping (Deferred to Chapter 05).
  - 🛑 **DO NOT** cover multi-node distributed pipeline parallelism or Tensor Parallelism sharding math (Deferred to Chapter 17).
- **Causal Bridge to 2.5:** Knowing that every decode token is physically expensive and memory-bound, how can the host runtime guarantee that emitted tokens adhere strictly to valid syntax without wasting budget?

#### Section 2.5: Constraining the Output: Grammar-Guided Decoding [core]
- **Heading & Anchor:** `## Constraining the Output: Grammar-Guided Decoding {#sec-vol3-processor-grammar-constrained}`
- **The Single Key Point:** Grammar-constrained decoding enforces structural syntax via decode-time logit masking on the accelerator, guaranteeing valid format while leaving semantic truth completely unverified.
- **Pedagogical Arc:**
  1. *The Need for Deterministic Structure:* Why autonomous agents fail when emitting free-form markdown for tool calls; JSON Schema and typed function calling standards.
  2. *Finite State Machine (FSM) Logit Masking:* Compiling JSON schemas or regular expressions into DFAs/PDAs. At generation step $t$, the FSM determines the subset of valid tokens $\mathcal{V}_{\text{valid}} \subset \mathcal{V}$ and masks illegal logits to $-\infty$.
  3. *Kernel-Level Masking vs. Host Serialization:* Why naive CPU-side masking stalls the GPU pipeline ($128\text{k}$ vocabulary iterations taking 15–40 ms). Modern inference runtimes (xgrammar, SGLang) precompute compressed token-transition bitmasks and execute masking directly inside the GPU sampling kernel.
  4. *The Syntactic vs. Semantic Divide:* Grammar masks guarantee valid brackets, quotes, and JSON data types; they provide zero guarantee that a referenced file exists, an SQL column is valid, or a shell command is safe.
  5. *Pathological Schema Forcing & The Mars Exam Dilemma:* If a schema omits an error or unknown field, logit masking forces the model to hallucinate false values to satisfy the grammar. Worked Example: Forcing a model to pick A, B, or C on an impossible multiple-choice question.
- **What NOT to Cover (Negative Scope):**
  - 🛑 **DO NOT** discuss tool subprocess execution, shell execution, or stdout capture (Deferred to Chapter 07).
  - 🛑 **DO NOT** discuss supervised fine-tuning for tool calling (Deferred to Chapter 13).
- **Causal Bridge to 2.6:** How does the host system formalize its calls to the model to enforce limits, handle streaming, and detect failures?

#### Section 2.6: The Invocation Contract and Status Envelopes [core]
- **Heading & Anchor:** `## The Invocation Contract and Status Envelopes {#sec-vol3-processor-contract}`
- **The Single Key Point:** Robust agent systems govern model calls through a typed RPC contract with explicit token and latency ceilings, streaming early cancellation, and a normalized status envelope.
- **Pedagogical Arc:**
  1. *The Typed Request Specification:* The RPC parameters: prompt tokens $\mathbf{x}$, model identifier, sampling parameters ($\tau, p$), generation ceiling $K_{\max}$, deadline $T_{\max}$, stop delimiters, and schema grammar $\mathcal{G}$.
  2. *Streaming Transport (SSE / gRPC) & Early Cancellation:* Consuming tokens incrementally. Emitting intermediate tokens allows the runtime to parse tool calls on the fly. If syntax validation fails or a safety trigger fires, issuing an early cancellation (`RST_STREAM`) immediately halts decode and frees GPU memory.
  3. *The Normalized 4-Outcome Status Envelope (@tbl-vol3-invocation-status):*
     - `COMPLETED`: Clean EOS or stop token reached within budget.
     - `TRUNCATED`: Reached $K_{\max}$ or deadline $T_{\max}$ before natural termination.
     - `REFUSED`: Engine safety or policy filter intercepted generation.
     - `TRANSPORT_ERROR`: Network drop, GPU OOM, or inference worker crash.
  4. *The Quarantining Invariant:* Truncated or malformed output must be quarantined in host memory escrow; partial strings must never reach compilers or actuation pipelines.
- **What NOT to Cover (Negative Scope):**
  - 🛑 **DO NOT** discuss multi-invocation retry loops, WAL persistence, or distributed sagas (Deferred to Chapter 10 & 11).
  - 🛑 **DO NOT** discuss gateway load balancing, cluster reverse proxies, or multi-tenant routing (Deferred to Chapter 17).
- **Causal Bridge to 2.7:** When an invocation returns `COMPLETED` with structurally valid code, what does that establish about the code's real-world correctness?

#### Section 2.7: Verification Boundaries: Likelihood Versus Operational Truth [core]
- **Heading & Anchor:** `## Verification Boundaries: Likelihood Versus Operational Truth {#sec-vol3-processor-continuations}`
- **The Single Key Point:** Sequence likelihood does not establish operational correctness; model outputs are unverified hypotheses requiring host validation before mutating state.
- **Pedagogical Arc:**
  1. *Likelihood vs. Correctness:* High probability reflects statistical typicality in training data, not factual truth or execution safety.
  2. *Treating Output as Untrusted Input:* Borrowing the foundational OS security principle: never trust input from an unprivileged process. Emitted code or tool calls are staged in memory escrow.
  3. *The Fallacy of Model Self-Checking:* Why asking a model "Are you sure?" fails to close invariants ($P < 1.0$). Self-checking amplifies attentional confirmation bias.
  4. *The External Deterministic Verification Perimeter (@tbl-verification-layers):* Compilers, AST parsers, linters, and type checkers provide deterministic, external invariant closure.
  5. *In-Memory Verification vs. Execution:* Distinguish in-memory static validation (AST parseability, schema validation, type checking) from dynamic execution (running code, running tests), keeping Chapter 02 strictly in memory.
- **What NOT to Cover (Negative Scope):**
  - 🛑 **DO NOT** describe microVM hypervisors, Firecracker, OverlayFS, cgroups, or seccomp (Deferred to Chapter 08).
  - 🛑 **DO NOT** execute dynamic subprocesses, test runners (`pytest`), or bash scripts (Deferred to Chapter 07).
  - 🛑 **DO NOT** formulate differential regression test math or SWE-bench evaluation frameworks (Deferred to Chapter 14 & 16).
  - 🛑 **DO NOT** discuss multi-turn repair loops or search trees (Deferred to Chapter 03).
- **Causal Bridge to 2.8:** Given these verification requirements and execution costs, how should systems engineers evaluate competing invocation interfaces?

#### Section 2.8: Systems Benchmarking of Invocation Interfaces [synthesis]
- **Heading & Anchor:** `## Systems Benchmarking of Invocation Interfaces {#sec-vol3-processor-interface-design}`
- **The Single Key Point:** Invocation interfaces must be evaluated by downstream verified task success under equal resource budgets, not by superficial fluency or parsing speed.
- **Pedagogical Arc:**
  1. *Controlled Systems Benchmarking:* Holding model weights, repository fixtures, and compute budgets constant while varying interface contracts.
  2. *Comparing Three Real Interface Paradigms (@tbl-vol3-interface-evaluation):*
     - *Free-form Natural Language + Regex Parsing* (Chatbot style).
     - *JSON Schema-Constrained Tool Calling* (OpenAI/Anthropic function calling).
     - *Grammar-Masked Search/Replace Diffs* (SWE-bench / Aider code editing).
  3. *Core Systems Evaluation Metrics:*
     - Structural Validity Rate ($R_{\text{syntax}}$): Percentage of invocations producing parseable structures on Turn 1.
     - Token Inflation Overhead ($\Delta M, \Delta K$): Overhead from schema definitions and JSON escaping.
     - Latency Profile: TTFT, ITL, and grammar masking overhead.
     - Verified Downstream Task Completion Rate: Does structured generation actually result in more passing test suites under equal token budgets?
  4. *Worked Example:* Sizing the token overhead tax across edit paradigms for a 5-line bug fix: Full File (2,500 tokens, 65s decode), JSON Patch (450 tokens, 12s decode), Search/Replace (75 tokens, 1.8s decode).
- **What NOT to Cover (Negative Scope):**
  - 🛑 **DO NOT** execute multi-turn agent debugging loops or interactive REPL sessions (Deferred to Chapter 03 & 09).
  - 🛑 **DO NOT** describe containerized sandbox implementations or microVMs (Deferred to Chapter 08).

#### Fallacies and Pitfalls [fallacies]
`## Fallacies and Pitfalls {#sec-vol3-processor-fallacies}`
- **Fallacy 1:** *One model response is one forward pass.* (Refutation: Prefill evaluates prompt tokens in parallel; decode requires $K$ sequential forward passes, each serializing across HBM memory bus transfers).
- **Pitfall 1:** *Treating a completed invocation as a completed task.* (Refutation: An HTTP `200 OK` or `Completed` status verifies only that the decode loop terminated normally, conveying zero guarantee of correctness or regression test passage).
- **Fallacy 2:** *Valid JSON means a safe and correct tool call.* (Refutation: Grammar constraints enforce character syntax at the logit surface; they do not verify file existence, logical correctness, or security permissions).
- **Pitfall 2:** *Collapsing incomplete, refusal, and transport failures into a generic retry loop.* (Refutation: Distinct outcome classes require distinct recovery paths; blindly retrying a budget truncation simply repeats the truncation).
- **Fallacy 3:** *Models have internal hardware traps.* (Refutation: A model forward pass is a continuous matrix multiplication pipeline; syntax errors and hallucinations are statistical outputs, not hardware interrupts or MMU faults).

#### Summary & Chapter Connection [summary]
`## Summary {#sec-vol3-processor-summary}`
- **Authoritative Synthesis:** One model invocation is a bounded learned computation mapping staged token inputs to candidate proposals via serialized autoregressive generation. The neural core possesses zero ambient authority. The agent runtime manages context, limits, status envelopes, and external verification.
- `::: {.callout-takeaways title="Core Systems Principles of the Model Invocation Boundary"}`
  1. *Tokens are the discrete integer currency and vocabulary gather indices of the foundation model.*
  2. *Autoregressive decode forms an irreducibly serial causal dependency chain on accelerator memory.*
  3. *Single-batch agent decode is memory-bandwidth bound ($B=1$ parameter shuttle floor).*
  4. *Likelihood and structural validity do not establish truth, safety, or authority.*
  5. *The invocation contract must enforce explicit budgets, early streaming cancellation, and normalized outcome envelopes.*
  6. *Candidate generations must be quarantined in host memory escrow until verified by external deterministic gates.*
- `::: {.callout-chapter-connection title="From One Candidate to Deliberate Computation"}`
  - Handoff forward: A single invocation produces one unprivileged, stochastic candidate sequence. When that candidate is ambiguous, incomplete, or fails runtime verification, the system cannot rely on simple prompt re-issuance. Chapter 3 examines how the runtime allocates additional inference-time compute—through search trees, verification loops, and environment interaction—to turn stochastic proposals into dependable systems outcomes.

---
"""

    # Split V2 into sections:
    # 1. Preamble (before Chapter 01): from start to "### Chapter 01:"
    # 2. Chapter 01: from "### Chapter 01:" to "## Part I: The Stochastic Processor"
    # 3. Chapter 02: from "## Part I: The Stochastic Processor" to "### Chapter 03: Inference-Time Deliberation"
    # 4. Chapter 03 onwards: from "### Chapter 03: Inference-Time Deliberation" to end.

    idx_intro = content.find("## Introduction: The Stochastic Computer\n\n### Chapter 01:")
    idx_part1 = content.find("## Part I: The Stochastic Processor\n\n### Chapter 02:")
    idx_ch03 = content.find("### Chapter 03: Inference-Time Deliberation")
    assert idx_intro != -1, "Could not find Chapter 01 intro"
    assert idx_part1 != -1, "Could not find Part I Ch02"
    assert idx_ch03 != -1, "Could not find Chapter 03"

    ch01_part = content[idx_intro : idx_part1]
    rest_of_book = content[idx_ch03:]

    # In rest_of_book:
    # 1. In Chapter 05 Section 5.1: Enhance with the Worked Example: "Why a 128k Context Crashes an 80 GB GPU"
    ch05_old_snippet = "- *Possible focus (Computational Boundary & Analytical Handoff):* The concurrency ceiling on physical accelerator memory ($B_{\\max} = \\frac{\\text{HBM}_{\\text{total}} - \\text{Mem}_{\\text{weights}}}{\\text{Mem}_{\\text{seq}}}$). Concludes with prose bridge to Section 5.2."
    ch05_new_snippet = """- *Possible focus (Computational Boundary & Analytical Handoff):* The concurrency ceiling on physical accelerator memory ($B_{\\max} = \\frac{\\text{HBM}_{\\text{total}} - \\text{Mem}_{\\text{weights}}}{\\text{Mem}_{\\text{seq}}}$).
  - *Boxed Worked Example:* Sizing the 128k Context Crash on an 80 GB GPU: Walk through a 70B model in FP16 ($140\\text{ GB}$ weights, requiring $TP=2$ across two 80 GB H100s, leaving $20\\text{ GB}$ free for KV cache). In FP16 ($b=2$), a 128k context consumes $41.94\\text{ GB}$ of KV cache per sequence, crashing the GPU ($41.94\\text{ GB} > 20\\text{ GB}$). Show how Grouped-Query Attention (GQA, $n_{\\text{kv}}=8$) and FP8 quantization ($b=1$) compress the KV footprint to $10.48\\text{ GB}$, allowing the sequence to fit. Concludes with prose bridge to Section 5.2."""

    if ch05_old_snippet in rest_of_book:
        rest_of_book = rest_of_book.replace(ch05_old_snippet, ch05_new_snippet)
        print("Chapter 05 successfully enriched with the 128k context worked example!")
    else:
        print("Warning: ch05_old_snippet not found exactly, proceeding.")

    # 2. Update references to Chapter 02 in rest_of_book to use "The Model Invocation Boundary"
    rest_of_book = rest_of_book.replace("Chapter 02 (The Stochastic Processor Core)", "Chapter 02 (The Model Invocation Boundary)")
    rest_of_book = rest_of_book.replace("Chapter 02 (Stochastic Processor Core)", "Chapter 02 (The Model Invocation Boundary)")
    rest_of_book = rest_of_book.replace("Section 2.7", "Section 2.4") # Roofline references updated to 2.4

    # Assemble V3
    v3_final = preamble_v3 + ch01_part + ch02_v3 + rest_of_book

    with open("books/vol3/MASTER_TEXTBOOK_OUTLINE_V3.md", "w", encoding="utf-8") as f:
        f.write(v3_final)

    print(f"Successfully generated books/vol3/MASTER_TEXTBOOK_OUTLINE_V3.md! Total characters: {len(v3_final)}")

if __name__ == "__main__":
    main()
