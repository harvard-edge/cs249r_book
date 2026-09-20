#!/usr/bin/env python3
"""
Create 5 Distinct Pedagogical Variations of the Master Outline.

Each variation retains the EXACT shared chapter blueprints (lines 540+ of MASTER_TEXTBOOK_OUTLINE_V2.md)
while substantially altering the frontmatter / general authoring guidance:

1. V3 (Quantitative Architect): Hennessy & Patterson style - quantitative trade-offs, Roofline math, empirical numbers.
2. V4 (Systems Principles Architect): Saltzer & Kaashoek style - modularity, Zero Ambient Authority, reference monitors.
3. V5 (Pragmatic Systems Engineer): Ousterhout clean abstraction style - cognitive economy, intuitive physical models first.
4. V6 (Dependability & Invariants): Dijkstra & Schneider style - fault models, fail-plausible faults, verification funnels.
5. V7 (Systems Builder & Kernel Hacker): MIT 6.828 / CS 162 lab style - runtime implementation, concrete ABI frames, profiling.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_OUTLINE = REPO_ROOT / "books" / "vol3" / "MASTER_TEXTBOOK_OUTLINE_V2.md"
OUTLINES_DIR = REPO_ROOT / "books" / "vol3" / "outlines"

# Read common body from line 540 onwards
with open(SOURCE_OUTLINE, "r", encoding="utf-8") as f:
    full_text = f.read()

split_marker = "# Detailed Chapter-by-Chapter Curricular Blueprints"
if split_marker not in full_text:
    print(f"Error: split marker '{split_marker}' not found in {SOURCE_OUTLINE}")
    sys.exit(1)

front_matter, common_blueprints = full_text.split(split_marker, 1)
common_body = split_marker + common_blueprints

COMMON_CAUSAL_CHAIN = r"""
## Book Overview: The Seven Parts

```
Volume III: Agentic Machine Learning Systems
├── Introduction: The Stochastic Computer (Chapter 01)
├── Part I: The Stochastic Processor (Chapters 02–03)
├── Part II: Context Memory and Storage (Chapters 04–06)
├── Part III: Tool Actuation and I/O Peripherals (Chapters 07–08)
├── Part IV: The Agent Operating System (Chapters 09–11)
├── Part V: The Policy Compiler (Chapters 12–14)
├── Part VI: Distributed Fleets and Operations (Chapters 15–17)
└── Part VII: Synthesis (Chapter 18)
```

### The 18-Chapter Causal Chain (Provocation $\to$ Resolution $\to$ Next Requirement)

| Ch | Title | Solved Problem | Exposed Limitation (The Provocation) | Next Ch Handoff |
| :--- | :--- | :--- | :--- | :--- |
| **01** | *The Stochastic Computer* | Defines the whole-system trajectory loop, 5-part task contract, and Fail-Plausible fault model. | We know a trajectory requires a computer, but what is its actual processing engine? | $\to$ **Ch 02** |
| **02** | *The Stochastic Processor Core* | Explains one model invocation: staged tokens, next-token generation, caller-visible status, output constraints, and latency. | One candidate sequence may not provide enough evidence for a correct decision. | $\to$ **Ch 03** |
| **03** | *Inference-Time Deliberation* | Allocates additional inference compute to alternatives, verification, and environment-conditioned revision under a budget. | Branches and observations generate more state than can remain visible in context. | $\to$ **Ch 04** |
| **04** | *Context-Window Working Memory* | Selects the logical working set staged for the next invocation. | Selected tokens still have a physical attention-state footprint in the serving system. | $\to$ **Ch 05** |
| **05** | *The KV-Cache Hierarchy* | Allocates, shares, and reclaims physical attention state for active and branching requests. | A serving cache does not preserve task knowledge across sessions. | $\to$ **Ch 06** |
| **06** | *Persistent External Memory* | Stores, retrieves, and invalidates durable knowledge and records through explicit interfaces. | Compute plus state still cannot observe or change the environment without controlled interaction. | $\to$ **Ch 07** |
| **07** | *Peripherals & Tool Actuation* | Turns a candidate into a parsed, permitted, dispatched action with an observable result. | Valid tool requests can still reach an unsafe execution environment. | $\to$ **Ch 08** |
| **08** | *Virtualization & Sandboxing* | Bounds authority and effects using capabilities, isolation, and untrusted-input handling. | Isolated calls still require supervision across time. | $\to$ **Ch 09** |
| **09** | *The Agent Operating System Control Plane* | Owns trajectory lifecycle, scheduling, budgets, pending work, cancellation, and completion. | Volatile runtime state is lost in a crash. | $\to$ **Ch 10** |
| **10** | *State, Persistence & Storage* | Records intent, decisions, observations, and confirmed effects for reconstruction and audit. | Recorded history cannot undo every external effect. | $\to$ **Ch 11** |
| **11** | *Fault Tolerance, Compensation & Sagas* | Recovers or escalates after partial, possibly irreversible effects. | Repeated failures expose policy gaps that runtime checks alone cannot remove. | $\to$ **Ch 12** |
| **12** | *Trajectory Data & Feedback* | Turns validated execution evidence into curated learning and evaluation fixtures. | Curated examples do not yet change the model. | $\to$ **Ch 13** |
| **13** | *Supervised Fine-Tuning* | Adapts proposal behavior from demonstrations while leaving enforcement external. | Demonstrations do not explore unobserved solutions. | $\to$ **Ch 14** |
| **14** | *Reinforcement Learning (RLVR)* | Tests when protected, verifiable outcomes support policy improvement through exploration. | A better individual policy does not settle coordination at scale. | $\to$ **Ch 15** |
| **15** | *Multi-Agent Fleets & Coordination*| Measures when delegation beats a single agent under matched resources and shared-state constraints. | More actors create more causal paths and harder failures to explain. | $\to$ **Ch 16** |
| **16** | *Observability & Evaluation* | Connects task acceptance to traces, controlled evaluation, and statistical evidence. | A reliable system must still meet cost and latency limits. | $\to$ **Ch 17** |
| **17** | *Performance & Cost Engineering* | Optimizes accepted tasks under whole-trajectory latency, capacity, and spending budgets. | Local choices must be synthesized into one coherent design. | $\to$ **Ch 18** |
| **18** | *System Synthesis: The Stochastic Computer* | Traces a task through every contract, boundary, recovery path, and acceptance check. | **The Book Concludes:** Remaining limits become explicit research problems. | Complete |

---
"""

# ==============================================================================
# VARIANT 3: The Quantitative Architect (Hennessy & Patterson Style)
# ==============================================================================
V3_FRONTMATTER = r"""# The Stochastic Computer: Master Textbook Curriculum Outline (Variant V3)
**Agentic Machine Learning Systems: The Quantitative Architecture Stance**
*Authoring Blueprint in the Tradition of Hennessy & Patterson*

## Pedagogical Vision: Quantitative Computer Architecture for Agentic AI

This textbook is written for **senior-level undergraduate and introductory graduate students in Computer Science and Computer Engineering** studying **Agentic Machine Learning Systems**.

Its intellectual posture is directly inspired by **Hennessy & Patterson's *Computer Architecture: A Quantitative Approach***:
1. **Measurement Over Speculation:** Every architectural trade-off must be quantified. We do not make vague qualitative claims about "fast" or "heavy" models; we calculate arithmetic intensity ($I = \text{FLOPs/byte}$), memory bus saturation thresholds ($I_{\text{sat}}$), space-time memory occupancy ($O_{\text{mem}}$), and dollar cost per verified task completion.
2. **The Memory Wall as the Central Reality:** The fundamental bottleneck of modern generative execution is the memory shuttle: the discrepancy between high accelerator tensor FLOPs and physical High-Bandwidth Memory (HBM) transfer speed. Prefill is compute-bound (GEMM); autoregressive decode is bandwidth-bound (GEMV).
3. **The Unprivileged Processing Element:** The Large Language Model (LLM) functions as an unprivileged, non-deterministic coprocessor. It possesses no program counter, no register file, and no execution authority. The host runtime acts as the supervisory system managing memory, dispatching work, and verifying output.

## Core Authoring Directives: The Quantitative Systems Stance

### 1. The 4-Step Quantitative Scaffolding Ladder
Every substantive technical section must progress through a 4-step scaffolding arc:
1. **The Systems Dilemma:** Open with an observable, concrete failure mode or physical bottleneck.
2. **Physical Architectural Intuition:** Ground the problem in the physical datapath (accelerator memory bus, SRAM capacity, interconnect bandwidth, or context capacity).
3. **The Concrete Systems Artifact:** Anchor the discussion in a typed data structure, C struct, or architectural block diagram before writing equations.
4. **Rigorous Quantitative Derivation:** Provide step-by-step mathematical derivations with explicit dimensional units ($[\text{GB/s}], [\text{TFLOP/s}], [\text{ms/token}]$). Follow **The Law of Constant Provenance**: never drop a magic number (such as a factor of 2, head dimension $d_{\text{head}}$, or byte precision $b$) without an immediate physical explanation.

### 2. Boxed Quantitative Worked Examples
Every core section must feature at least one Hennessy & Patterson-style boxed worked example using:
`::: {.callout-note title="Worked Example: Sizing the [Mechanism] Floor"}`
The worked example must follow a strict three-part structure:
- **Problem Statement:** Stating explicit workload parameters (e.g. 70B parameter model, FP8 weights, H100 with 3.35 TB/s peak bandwidth, 80% bus efficiency).
- **Step-by-Step Arithmetic:** Showing every intermediate calculation with explicit units.
- **Architectural Takeaway:** Explaining what the numerical result dictates for software systems design.

### 3. Structural Invariant: Section .1 as Unbroken Stage-Setter
- **Section .1 of every chapter must contain ZERO subsections (NO `###`) and NO bulleted outline.** It serves as an unbroken, continuous narrative that sets the chapter's stage, establishes the multi-tier operational boundary, presents the Systems Rosetta Stone table, and bridges smoothly into Section .2.
- Subsections (`###`) strictly begin in Section .2 onwards (typically 2–4 clean subheadings).

### 4. Language & Discipline Guardrails
- **Speak authentic systems language:** LLM, tokens, Byte-Pair Encoding (BPE), logits, softmax, KV cache, prefill, decode, inference engines (vLLM, TensorRT-LLM), host runtime.
- **Never anthropomorphize:** Forbid phrases like "the model thinks", "the model decides", or "the model understands". Instead write: "the neural core evaluates", "the runtime samples from the distribution", "the supervisor verifies invariants".
- **No decorative simulator boilerplate:** State hardware parameters cleanly in tables, equations, and worked examples; do not dump Python simulator class definitions into textbook prose.
"""

# ==============================================================================
# VARIANT 4: The Systems Principles Architect (Saltzer & Kaashoek Style)
# ==============================================================================
V4_FRONTMATTER = r"""# The Stochastic Computer: Master Textbook Curriculum Outline (Variant V4)
**Agentic Machine Learning Systems: The Operating Systems Principles Stance**
*Authoring Blueprint in the Tradition of Saltzer & Kaashoek*

## Pedagogical Vision: Principles of Computer System Design for Agentic AI

This textbook is written for **senior-level undergraduate and introductory graduate students in Computer Science and Computer Engineering** studying **Agentic Machine Learning Systems**.

Its intellectual posture is directly inspired by **Jerome Saltzer & M. Frans Kaashoek's *Principles of Computer System Design***:
1. **Modularity and Layering:** Complex systems must enforce strict separation of concerns. We structure the Stochastic Computer across three distinct operational tiers: the Host Agent Runtime (Supervisor), the Inference Service Daemon (Hardware Driver), and the Neural Core (Accelerator Hardware).
2. **Zero Ambient Authority ($A=0$):** Following Saltzer & Schroeder's principle of least privilege, the foundation model possesses zero implicit access to host filesystems, network sockets, or execution privileges. Emitting text modifies zero host state. Candidate strings reside in memory escrow until an authoritative host reference monitor validates and executes them.
3. **The End-to-End Argument:** Lower layers (like grammar logit masks or safety classifiers) can only enforce mechanical, syntactic constraints. Semantic correctness and task completion require end-to-end evidence verified by external software oracles (compilers, test runners, AST validators).

## Core Authoring Directives: The Systems Principles Stance

### 1. The 4-Step Contract Scaffolding Ladder
Every substantive technical section must progress through a 4-step architectural arc:
1. **The Trust Boundary or Invariant Violation:** Open with an observable failure where an unprivileged model violates a system invariant or produces an unverified side effect.
2. **The Operating Systems Intuition:** Translate the ML mechanism into classical OS design principles (e.g. Tokenizer $\leftrightarrow$ ABI serializer; KV cache $\leftrightarrow$ page-table-managed working set; Tool calling $\leftrightarrow$ mediated system call; Hallucination $\leftrightarrow$ fail-plausible semantic corruption across the epistemic gap).
3. **The Formal Interface Contract:** Specify explicit typed request/response schemas, capability masks, or status envelopes before presenting implementation details.
4. **Grounded Failure Analysis:** Evaluate failure domains, boundary conditions, and isolation mechanisms step-by-step.

### 2. Interface Contracts and Reference Monitors
Every section addressing interactions between the model, runtime, and tools must clearly document:
- **Authority Boundary:** What authority does the component possess, and how is that authority attenuated?
- **State Ownership & Placement:** Who owns the active state (host DRAM, GPU device memory, or durable disk), and what are its lifetime and reclamation policies?
- **Normalized Status Envelope:** How does the runtime categorize execution outcomes (`COMPLETED`, `TRUNCATED`, `REFUSED`, `TRANSPORT_FAILURE`)?

### 3. Structural Invariant: Section .1 as Unbroken Stage-Setter
- **Section .1 of every chapter must contain ZERO subsections (NO `###`) and NO bulleted outline.** It is a cohesive, standalone introduction that frames the governing systems dilemma, contrasts classical deterministic execution with unprivileged stochastic generation, establishes the three-tier boundary, and presents the Systems Rosetta Stone table.
- Subsections (`###`) strictly begin in Section .2 onwards.

### 4. Language & Discipline Guardrails
- **Authentic Systems Terminology:** Use real operating systems terms: reference monitor, capability mask, memory escrow, status envelope, mediated syscall, sandbox isolation, write-ahead log.
- **Zero Anthropomorphism:** Never write "the model realizes its mistake" or "the model decides". Treat the model strictly as an unprivileged, non-deterministic worker process proposing candidates.
- **American English:** Adhere strictly to American English spelling (`-ize`, `behavior`, `center`).
"""

# ==============================================================================
# VARIANT 5: The Pragmatic Systems Engineer (Ousterhout Clean Abstraction Style)
# ==============================================================================
V5_FRONTMATTER = r"""# The Stochastic Computer: Master Textbook Curriculum Outline (Variant V5)
**Agentic Machine Learning Systems: The Pragmatic Systems Engineer Stance**
*Authoring Blueprint in the Tradition of John Ousterhout & Brian Kernighan*

## Pedagogical Vision: Clean Abstractions & Cognitive Economy for Agentic AI

This textbook is written for **senior-level undergraduate and introductory graduate students in Computer Science and Computer Engineering** studying **Agentic Machine Learning Systems**.

Its intellectual posture is directly inspired by **John Ousterhout's *A Philosophy of Software Design*** and **Kernighan & Ritchie**:
1. **Cognitive Economy Above All:** We respect the reader's working memory. We never bury a student under unmotivated formulas, dense acronym soups, or decorative mathematical proofs. Every concept is explained with crystalline clarity using plain English before introducing formal notation.
2. **Deep Modules & Clean Interfaces:** The secret to managing complexity in agentic systems is creating deep modules: interfaces that are simple and powerful, hiding internal stochastic complexity behind crisp, typed runtime contracts.
3. **Intuitive Physical Mental Models:** Before writing an equation, we provide a physical mental model that an undergraduate student can visualize immediately:
   - *The Sports Car vs. The Narrow Straw:* The GPU has massive tensor compute (hundreds of TFLOPs), but must draw weights through a narrow memory straw (a few TB/s of HBM). In prefill, you pour a bucket of tokens through the engine at once (compute-bound). In decode, you must sweep the entire 70 GB model through the straw just to emit one word (memory-bound)!
   - *The Mars Optical Exam:* Forcing an agent to pick A, B, or C when the true answer is "none of the above" causes 100% confident guessing because the interface provided no escape hatch.

## Core Authoring Directives: The Pragmatic Clean Abstraction Stance

### 1. The 4-Step Intuitive Scaffolding Ladder
Every substantive technical section must progress through a 4-step intuitive arc:
1. **The Observable Failure / Real-World Dilemma:** Open with a relatable engineering problem (e.g. why an HTTP 200 OK is not task completion; why a 128k prompt crashes an 80 GB GPU; why unquoted JSON keys break AST parsers).
2. **The Intuitive Mental Model:** Establish a physical analogy or mental model before writing any code or equations. De-jargonize every technical term immediately.
3. **The Clean Code Artifact:** Anchor the discussion in a readable, idiomatic Python `@dataclass` or clean data structure.
4. **Transparent Arithmetic:** Walk through quantitative trade-offs with simple, transparent step-by-step arithmetic. Never skip algebraic steps.

### 2. De-Jargonizing Mandate
Whenever a complex technical term appears (e.g. "AST impedance mismatch", "causal serialization barrier", "epistemic gap", "pathological schema forcing"), the text must immediately provide a crisp, one-sentence plain-English translation so the student never feels lost.

### 3. Structural Invariant: Section .1 as Unbroken Stage-Setter
- **Section .1 of every chapter must contain ZERO subsections (NO `###`) and NO bulleted outline.** It serves as an unbroken, continuous narrative that introduces the chapter's central dilemma, contrasts classical systems with stochastic generation, and bridges directly into Section .2.
- Subsections (`###`) strictly begin in Section .2 onwards (typically 2–4 clean subheadings).

### 4. Language & Discipline Guardrails
- **Engaging, Authoritative Tone:** Write as an experienced, pragmatic systems mentor in a computer systems lab.
- **No Anthropomorphism:** Treat the LLM as a physical matrix processor and stochastic proposal engine.
- **American English:** Adhere strictly to American English spelling.
"""

# ==============================================================================
# VARIANT 6: The Dependability & Invariants Stance (Dijkstra / Schneider Style)
# ==============================================================================
V6_FRONTMATTER = r"""# The Stochastic Computer: Master Textbook Curriculum Outline (Variant V6)
**Agentic Machine Learning Systems: The Dependability & Fault-Tolerance Stance**
*Authoring Blueprint in the Tradition of Edsger Dijkstra & Fred Schneider*

## Pedagogical Vision: Verifiable Systems Dependability for Agentic AI

This textbook is written for **senior-level undergraduate and introductory graduate students in Computer Science and Computer Engineering** studying **Agentic Machine Learning Systems**.

Its intellectual posture is directly inspired by **Edsger W. Dijkstra, Fred Schneider, and Jim Gray**:
1. **The Fail-Plausible Fault Model:** Classical computing relies on the *fail-stop model* (Schlichting & Schneider 1983): processors execute instructions according to specification or halt upon fault (triggering hardware exceptions). Generative foundation models violate this assumption completely: they exhibit *fail-plausible behavior*, emitting fluent, high-confidence, syntactically pristine outputs that violate semantic invariants, reference nonexistent files, or invent invalid parameters.
2. **The Invariant Closure Principle:** A stochastic neural policy cannot close invariants ($P < 1.0$). Asking a model "Are you sure?" merely shifts probability mass along its distribution. Therefore, invariant closure must be enforced mechanically by external, deterministic software enclaves: compilers, type checkers, capability reference monitors, and sandboxed test harnesses.
3. **Dijkstra's Testing Principle for Trajectories:** *"Program testing can be used to show the presence of bugs, but never to show their absence"* (Dijkstra 1970). An agent cannot achieve certified global correctness; it achieves verifiable empirical evidence under isolated, reproducible test conditions.

## Core Authoring Directives: The Dependability & Invariants Stance

### 1. The 4-Step Invariant Scaffolding Ladder
Every substantive technical section must progress through a 4-step dependability arc:
1. **The Invariant Failure / Semantic Hazard:** Open with an observable failure mode where an unmonitored model violates a system invariant, causes a silent regression, or bypasses a check.
2. **The Fault Model Contrast:** Contrast how classical deterministic systems handle the fault (hardware traps, memory protection faults) with how foundation models fail plausibly across the epistemic gap.
3. **The Deterministic Verification Enclave:** Present the formal mechanical barrier (schema validator, DFA logit mask, capability filter, or sealed verifier) that intercepts and quarantines the candidate proposal.
4. **Quantitative Reliability Analysis:** Formulate error rates (False Acceptance Rate vs. False Rejection Rate), verification costs, and trade-offs step-by-step.

### 2. Failure Matrices and Defensive Design
Every chapter must emphasize:
- **Separation of Syntactic Validity from Semantic Truth:** Enforcing schema constraints (e.g. JSON DFA masks) guarantees only well-formed syntax; it guarantees zero semantic truth.
- **Quarantining Invariant:** Unverified model outputs must be held in memory escrow and never allowed to mutate host state until external gates certify compliance.
- **Compensating Actions & Reversibility:** Categorizing actions into reversible probes, compensable transactions (Sagas), and irreversible external mutations.

### 3. Structural Invariant: Section .1 as Unbroken Stage-Setter
- **Section .1 of every chapter must contain ZERO subsections (NO `###`) and NO bulleted outline.** It serves as an unbroken, continuous narrative that frames the governing systems dilemma, contrasts fail-stop with fail-plausible computing, and bridges smoothly into Section .2.
- Subsections (`###`) strictly begin in Section .2 onwards.

### 4. Language & Discipline Guardrails
- **Rigorous Systems Tone:** Vigilant, disciplined, engineering rock-solid reliability.
- **Zero Anthropomorphism:** Never attribute human consciousness, realization, or understanding to statistical distributions.
- **American English:** Adhere strictly to American English spelling.
"""

# ==============================================================================
# VARIANT 7: The Systems Builder & Kernel Hacker Stance (MIT 6.828 Style)
# ==============================================================================
V7_FRONTMATTER = r"""# The Stochastic Computer: Master Textbook Curriculum Outline (Variant V7)
**Agentic Machine Learning Systems: The Systems Builder & Kernel Hacker Stance**
*Authoring Blueprint in the Tradition of MIT 6.828 (Operating System Engineering) & UC Berkeley CS 162*

## Pedagogical Vision: Hands-On Systems Engineering for Agentic AI

This textbook is written for **senior-level undergraduate and introductory graduate students in Computer Science and Computer Engineering** studying **Agentic Machine Learning Systems**.

Its intellectual posture is directly inspired by **hands-on systems engineering lab courses (such as MIT 6.828 / 6.S081, Stanford CS 140, and UC Berkeley CS 162)**:
1. **Learn by Building the AI Runtime:** We teach systems principles by showing students how the runtime software is actually constructed. We never stop at hand-wavy architectural boxes or decorative pseudo-code. We show the concrete memory layouts, byte-level ABI frames, circular ring buffers, non-blocking polling loops (`epoll`/`kqueue`), and POSIX interface bindings that make autonomous execution possible.
2. **The Dual-Stack Runtime Engineer:** The modern AI systems engineer must be equally ambidextrous across two software planes:
   - *The User-Space Process Supervisor:* Managing long-lived agent lifecycles, sub-process execution (`fork`/`exec`), standard I/O streaming pipes, pseudo-terminals (`pty`), Linux namespaces, cgroups v2, and `seccomp-bpf` syscall filters.
   - *The High-Throughput Inference Daemon:* Managing GPU memory allocations, PagedAttention block tables, continuous batching iteration loops, shared-memory IPC, and CUDA stream synchronization.
3. **Concrete Failure Traces Over Pure Theory:** When explaining a systems bottleneck, we inspect real execution traces: raw terminal `stderr`, process exit codes (`WEXITSTATUS`), GPU memory allocation traps (`cudaErrorMemoryAllocation`), gRPC status trailers, or malformed AST byte streams.
4. **Guarding Against the Low-Level Trivia Trap:** While we embrace a kernel-hacker mindset, our readers are undergraduate seniors preparing to build production AI systems—not write monolithic device drivers. We do NOT drown the student in x86-64 assembly trivia, hardware interrupt vectors, or irrelevant CPU register allocation. Every low-level concept is strictly scoped to the *AI Systems Runtime stack*: how user-space supervisors, inference engines (such as vLLM and SGLang), and container sandboxes coordinate across process and device boundaries.

## Core Authoring Directives: The Systems Builder Stance

### 1. The 4-Step Builder Scaffolding Ladder
Every substantive technical section must progress through a 4-step builder arc:
1. **The Production Engineering Bottleneck:** Open with an observable runtime failure or resource wall (e.g., streaming stdout tokens blocking the supervisor event loop; zombie sandbox processes leaking file descriptors; external memory fragmentation in continuous batching page tables).
2. **The Architectural Design Choice & Data Layout:** Frame the decision as an engineering trade-off: *"If you were implementing this subsystem in C or Python, how would you structure the ABI frame, the ring buffer, or the page table?"*
3. **The Concrete Implementation Artifact:** Anchor the discussion in a typed Python `@dataclass`, C struct layout, POSIX syscall sequence, or clean algorithm listing showing the exact byte/field layout before writing equations.
4. **Empirical Profiling & Quantitative Accounting:** Walk through timing breakdowns (TTFT vs. inter-token latency, IPC latency, context switch overhead) and memory occupancy derivations with explicit dimensional units ($[\text{ms}], [\text{MB}], [\text{GB/s}]$).

### 2. Boxed Systems Builder Worked Examples
Every core section must feature at least one systems builder boxed worked example using:
`::: {.callout-note title="Worked Example: Systems Builder Implementation & Trace Profiling"}`
The worked example must follow a strict three-part structure:
- **Problem Statement & Production Bottleneck:** Stating explicit workload parameters, buffer capacities, process boundaries, and hardware constraints.
- **Data Layout & Step-by-Step Implementation Mechanics:** Tracing the concrete memory layout, buffer offset arithmetic, or syscall sequence with explicit dimensional units.
- **Systems Hacker Takeaway:** Explaining what this concrete implementation dictates for host runtime concurrency, memory protection, or batch scheduling.

### 3. Structural Invariant: Section .1 as Unbroken Stage-Setter
- **Section .1 of every chapter must contain ZERO subsections (NO `###`) and ZERO bulleted lists.** It serves as an unbroken, continuous narrative that frames the systems engineering challenge, delineates the three-tier boundary (Host Supervisor $\leftrightarrow$ Serving Daemon $\leftrightarrow$ Silicon Accelerator), presents the foundational Systems Rosetta Stone table, and transitions cleanly into Section .2.
- Subsections (`###`) strictly begin in Section .2 onwards (typically 2–4 clean subheadings).

### 4. Language & Discipline Guardrails
- **Authentic Systems Terminology:** Use real operating systems and runtime terms: file descriptors, circular ring buffers, memory escrow, capability masks, POSIX exit codes, continuous batching, block tables, `seccomp-bpf`, pseudo-terminals (`pty`), copy-on-write, shared memory.
- **Normalized 4-Outcome Status Envelope:** All processor invocations return a normalized status envelope. Always use these exact uppercase enumeration names: `COMPLETED`, `TRUNCATED`, `REFUSED`, `TRANSPORT_FAILURE`.
- **Zero Ambient Authority ($A=0$):** Emitted candidate strings have zero implicit privilege. Text strings reside in memory escrow until validated and dispatched by an authoritative reference monitor.
- **Zero Anthropomorphism:** Never write "the model thinks", "the model decides", or "the model realizes". Treat the LLM strictly as an accelerator-bound execution engine and candidate string generator; the host supervisor manages execution.
- **No Decorative Simulator Boilerplate:** State hardware parameters cleanly in tables, equations, and worked examples; do not dump synthetic Python simulator class definitions into textbook prose.
- **American English:** Adhere strictly to American English spelling (`-ize`, `behavior`, `center`).
"""

# ==============================================================================
# VARIANT 8: The Distributed Systems & Unreliable Service Stance (Lamport/Gray/Vogels)
# ==============================================================================
V8_FRONTMATTER = r"""# The Stochastic Computer: Master Textbook Curriculum Outline (Variant V8)
**Agentic Machine Learning Systems: The Distributed Systems & Unreliable Service Stance**
*Authoring Blueprint in the Tradition of Leslie Lamport, Jim Gray, and Werner Vogels*

## Pedagogical Vision: Distributed Computing Principles for Agentic AI

This textbook is written for **senior-level undergraduate and introductory graduate students in Computer Science and Computer Engineering** studying **Agentic Machine Learning Systems**.

Its intellectual posture is directly inspired by **the foundational architects of distributed computing: Leslie Lamport, Jim Gray, and Werner Vogels**:
1. **The Stochastic Model as an Unreliable, High-Latency Remote Microservice:** Never treat a foundation model as a synchronous, in-process function call. An LLM inference cluster is a distributed, multi-tenant, high-latency remote microservice subject to variable network queuing, transport packet loss, connection resets, HTTP 429 rate limits, and non-deterministic response times.
2. **The Fallacies of Distributed Computing Applied to LLMs:** Every architectural decision must confront the reality of Peter Deutsch and James Gosling's distributed computing fallacies adapted to AI systems:
   - *Latency is NOT zero:* Time-to-First-Token (TTFT) and autoregressive generation consume hundreds of milliseconds to tens of seconds—orders of magnitude slower than local DRAM or IPC.
   - *Bandwidth is NOT infinite:* Shuttling KV caches or full vocabulary logit vectors across PCIe buses or data-center fabrics creates massive network saturation.
   - *The network is NOT reliable:* Inference daemons drop connections mid-stream, GPUs suffer out-of-memory (OOM) faults, and model context limits truncate tokens.
   - *Transport is NOT secure:* Untrusted token payloads traverse RPC boundaries, introducing prompt injection and SSRF hazards across microservices.
   - *Topology changes constantly:* Dynamic autoscaling, spot-instance preemptions, and failovers re-route requests across heterogeneous model replicas.
3. **"Everything Fails, All the Time" (Vogels's Axiom):** Classical single-node error handling (`try/except`) collapses in agentic workflows. Because multi-turn agents execute across external databases, payment gateways, and third-party APIs, partial failures are inevitable. Reliability requires distributed fault-tolerance patterns: Saga orchestrators with compensating transactions, Write-Ahead Logging (WAL) for crash recovery, RPC deadline propagation (`grpc-timeout`), idempotency keys, hedged requests, and circuit breakers.

## Core Authoring Directives: The Distributed Systems Stance

### 1. The 4-Step Distributed Scaffolding Ladder
Every substantive technical section must progress through a 4-step distributed systems arc:
1. **The Distributed Failure Mode / Network Dilemma:** Open with a concrete distributed failure (e.g., a 30-second inference timeout leaving an external API call in an indeterminate state; a retry storm of stochastic queries overloading a serving cluster; an un-idempotent tool call repeating a destructive mutation like charging a credit card twice).
2. **The Distributed Systems Mental Model:** Map the problem to classical distributed systems theory: two-phase commit vs. Sagas, CAP theorem trade-offs, Lamport logical clocks and vector timestamps for causal event ordering, leaky bucket rate limiting, and end-to-end lease management.
3. **The Concrete Distributed Protocol Artifact:** Anchor the discussion in typed gRPC Protobuf definitions, HTTP/2 header contracts (`X-Correlation-ID`, `grpc-timeout`, `Idempotency-Key`), Saga compensation state machine tables, or vector clock structures before writing equations.
4. **Quantitative Reliability & Latency Analysis:** Formulate tail latency distributions ($P_{50}, P_{95}, P_{99}$), availability formulas ($A = \prod A_i$), retry amplification factors, and backoff jitter equations with explicit dimensional units ($[\text{ms}], [\text{req/s}], [\%]$).

### 2. Boxed Distributed Systems Worked Examples
Every core section must feature at least one distributed systems boxed worked example using:
`::: {.callout-note title="Worked Example: Distributed Fault Tolerance & Latency Engineering"}`
The worked example must follow a strict three-part structure:
- **Problem Statement & Distributed Topology:** Stating network round-trip time (RTT), inference engine TTFT, failure probabilities, timeout budgets, and downstream service SLAs.
- **Step-by-Step Protocol & Reliability Arithmetic:** Tracing deadline deductions across RPC hops, calculating retry budgets with exponential backoff and jitter, or sizing idempotency deduplication windows.
- **Distributed Architectural Takeaway:** Explaining what the derivation dictates for client-side circuit breaker thresholds, hedged request triggers, or saga rollback logs.

### 3. Structural Invariant: Section .1 as Unbroken Stage-Setter
- **Section .1 of every chapter must contain ZERO subsections (NO `###`) and ZERO bulleted lists.** It serves as an unbroken, continuous narrative that introduces the chapter's distributed systems dilemma, frames the stochastic model as an unbonded remote service, establishes the distributed microservice boundary, presents the Systems Rosetta Stone table, and bridges smoothly into Section .2.
- Subsections (`###`) strictly begin in Section .2 onwards (typically 2–4 clean subheadings).

### 4. Language & Discipline Guardrails
- **Authentic Distributed Systems Terminology:** Use real distributed computing terms: gRPC, RPC deadline propagation, idempotency keys, circuit breakers, hedged requests, tail tolerance, write-ahead logging (WAL), distributed sagas, compensating transactions, vector clocks, exponential backoff with full jitter, at-least-once delivery, partition tolerance.
- **Normalized 4-Outcome Status Envelope:** All model invocations return a normalized status envelope. Always use these exact uppercase enumeration names: `COMPLETED`, `TRUNCATED`, `REFUSED`, `TRANSPORT_FAILURE`.
- **Zero Ambient Authority ($A=0$):** Network messages carry explicit capability tokens; no unauthenticated or unverified RPC can execute.
- **Zero Anthropomorphism:** Treat model endpoints as network microservices generating stochastic streams, not thinking beings.
- **No Decorative Simulator Boilerplate:** State network and server parameters cleanly in tables, equations, and worked examples; do not dump synthetic simulator classes into textbook prose.
- **American English:** Adhere strictly to American English spelling (`-ize`, `behavior`, `center`).
"""

# ==============================================================================
# VARIANT 9: The Compiler & Language Runtime Stance (Dragon Book & LLVM)
# ==============================================================================
V9_FRONTMATTER = r"""# The Stochastic Computer: Master Textbook Curriculum Outline (Variant V9)
**Agentic Machine Learning Systems: The Compiler & Language Runtime Stance**
*Authoring Blueprint in the Tradition of Aho, Lam, Sethi, Ullman (The Dragon Book) & LLVM*

## Pedagogical Vision: Compilers & Formal Language Runtimes for Agentic AI

This textbook is written for **senior-level undergraduate and introductory graduate students in Computer Science and Computer Engineering** studying **Agentic Machine Learning Systems**.

Its intellectual posture is directly inspired by **classic compiler design, formal language theory, and modern language runtimes: Alfred Aho, Monica Lam, Ravi Sethi, Jeffrey Ullman (*Compilers: Principles, Techniques, and Tools* / The Dragon Book) and Chris Lattner (LLVM)**:
1. **The Stochastic Execution Pipeline as a Language Runtime:** Autoregressive generation is not freeform prose composition—it is program synthesis, lexical emission, and code execution under formal grammar constraints. The foundation model acts as a non-deterministic, speculative token-stream generator embedded within a formal language runtime.
2. **The Lexical, Syntactic, and Semantic Triad:**
   - *Lexical Phase (Scanning & Tokenization):* Byte-Pair Encoding (BPE) is lexical scanning. The runtime must handle scanner state machines, token boundary anomalies (e.g., whitespace merging, multi-byte UTF-8 splits), and byte fallback mechanisms that break naive downstream parsers.
   - *Syntactic Phase (Parsing & Grammar Masks):* Context-Free Grammars (CFGs), pushdown automata (PDAs), and LR parse tables are compiled into decode-time logit masks. By intersecting the grammar's follow-set with a token trie, the runtime forces *syntax-by-construction*: every emitted byte sequence is guaranteed to be syntactically valid JSON, SQL, or Python before leaving the GPU.
   - *Semantic Phase (Type Inference & AST Validation):* CFGs cannot enforce contextual semantics (e.g., variable defined before use, schema type compatibility). Emitted candidate strings are held in memory escrow, parsed into Abstract Syntax Trees (ASTs), and validated by external static type checkers and linters before execution.
3. **Symbol Tables, Scopes & Activation Records:** Prompt context assembly is lexical scope resolution. System instructions and tool schemas populate the global symbol table; multi-turn interactions push activation records onto the call stack; context compaction is dead-variable elimination and mark-sweep garbage collection.
4. **Ahead-of-Time (AOT) vs. Just-in-Time (JIT) Policy Specialization:** We view Supervised Fine-Tuning (SFT) and Reinforcement Learning with Verifiable Rewards (RLVR) as an offline compiler pipeline: compiling high-level behavioral specifications and verified execution traces into specialized model weights.

## Core Authoring Directives: The Compiler & Runtime Stance

### 1. The 4-Step Compiler Scaffolding Ladder
Every substantive technical section must progress through a 4-step compiler arc:
1. **The Syntax Breakdown / Lexical Anomaly:** Open with an observable language or parsing breakdown (e.g., a BPE token boundary split breaking JSON key-value parsing; an unquoted identifier causing an AST parse crash; a type mismatch in a generated tool call).
2. **The Formal Language / Compiler Mental Model:** Map the problem to classic compiler theory: lexical scanners, regular expressions vs. CFGs, Chomsky hierarchy, shift-reduce conflicts, LR parse states, symbol tables, and static semantic analysis.
3. **The Concrete Grammar / AST Artifact:** Anchor the discussion in formal EBNF grammar specifications, pushdown automata state diagrams, token-trie intersection tables, or concrete AST node hierarchies before writing equations.
4. **Quantitative Automata & Language Analysis:** Formulate grammar compilation space-time complexity, DFA/PDA state transition lookup costs, vocabulary mask bitset overheads ($V / 64$ uint64 words), and parser throughput with explicit units ($[\text{ms}], [\text{MB}], [\text{tokens/sec}]$).

### 2. Boxed Compiler & Runtime Worked Examples
Every core section must feature at least one compiler-focused boxed worked example using:
`::: {.callout-note title="Worked Example: Grammar Compilation & Automata-Directed Decoding"}`
The worked example must follow a strict three-part structure:
- **Grammar Specification & Token Vocabulary:** Stating explicit EBNF grammar rules, target schema, vocabulary size $V$, and token prefix.
- **Step-by-Step Automata Transition & Masking Derivation:** Tracing PDA/DFA state transitions, intersecting token prefixes with grammar follow-sets, and calculating logit bitmask tensors.
- **Compiler Runtime Takeaway:** Explaining what the derivation dictates for inference engine logit-masking latency, GPU memory footprint, and grammar precompilation.

### 3. Structural Invariant: Section .1 as Unbroken Stage-Setter
- **Section .1 of every chapter must contain ZERO subsections (NO `###`) and ZERO bulleted lists.** It serves as an unbroken, continuous narrative that introduces the chapter's formal language dilemma, establishes the compiler/runtime boundary (Scanner $\leftrightarrow$ Parser $\leftrightarrow$ Semantic Analyzer), presents the Systems Rosetta Stone table, and bridges smoothly into Section .2.
- Subsections (`###`) strictly begin in Section .2 onwards (typically 2–4 clean subheadings).

### 4. Language & Discipline Guardrails
- **Authentic Compiler Terminology:** Use real compiler and programming language terms: lexical scanning, BPE token boundary anomalies, Context-Free Grammar (CFG), Pushdown Automaton (PDA), LR parse tables, logit masking via token tries, Abstract Syntax Tree (AST), symbol table, lexical scope, activation records, type checking, static analysis.
- **Normalized 4-Outcome Status Envelope:** All model invocations return a normalized status envelope. Always use these exact uppercase enumeration names: `COMPLETED`, `TRUNCATED`, `REFUSED`, `TRANSPORT_FAILURE`.
- **Zero Ambient Authority ($A=0$):** Emitted ASTs and code proposals have zero implicit execution authority until verified by static analyzers and executed in sandboxes.
- **Zero Anthropomorphism:** Treat the LLM strictly as a non-deterministic token emitter operating across a grammar-constrained state space.
- **No Decorative Simulator Boilerplate:** State grammar and automaton parameters cleanly in tables, equations, and worked examples; do not dump synthetic simulator classes into textbook prose.
- **American English:** Adhere strictly to American English spelling (`-ize`, `behavior`, `center`).
"""

# ==============================================================================
# VARIANT 10: The AI Systems Engineering Apprenticeship Stance
# ==============================================================================
V10_FRONTMATTER = r"""# The Stochastic Computer: Master Textbook Curriculum Outline (Variant V10)
**Agentic Machine Learning Systems: The AI Systems Engineering Apprenticeship Stance**
*The 'From CS Undergrad to Production AI Engineer' Bridge*

## Pedagogical Vision: Practical Apprenticeship for AI Systems Engineering

This textbook is written for **senior-level undergraduate and introductory graduate students in Computer Science and Computer Engineering** studying **Agentic Machine Learning Systems**.

Its intellectual posture is designed specifically as an **empathetic, rigorous bridge from standard undergraduate computer science to production AI systems reality**:
1. **The Apprenticeship Model (Bridging Textbook CS to Production AI Systems):** The student already understands operating systems processes, virtual memory, threads, filesystems, and basic deep learning tensors. However, they have never built or debugged a production AI serving runtime (vLLM, SGLang, TensorRT-LLM) or an agent supervisor. We build this bridge explicitly, grounding every new AI systems concept in familiar CS fundamentals without hand-waving or skipping intermediate conceptual leaps.
2. **Production War Stories & Concrete Diagnostics:** We do not teach systems in a sanitized vacuum. Every chapter is anchored in authentic production engineering realities: unexpected GPU out-of-memory (CUDA OOM) crashes from unbudgeted KV-cache fragmentation, tail latency explosions caused by prefix cache misses, silent trajectory infinite loops from unhandled error envelopes, and sandbox escapes from unattenuated tool privileges. Students inspect real diagnostic traces, terminal logs, memory flame graphs, and Prometheus metrics.
3. **No Skipped Steps: First-Principles Derivations & Sizing Rules of Thumb:** We respect the learner's intellectual journey. We never skip intermediate algebraic steps or drop unexplained constants. Every formula is derived transparently from first principles, culminating in practical, memorable "Rules of Thumb" that engineers use on the job for cluster sizing, memory budgeting, and latency engineering.

## Core Authoring Directives: The Apprenticeship Stance

### 1. The 4-Step Apprenticeship Scaffolding Ladder
Every substantive technical section must progress through a 4-step apprenticeship arc:
1. **The Production Incident / Real-World Mystery:** Open with an authentic, relatable production incident (e.g., *"Our code-generation agent passed all single-request unit tests in 3 seconds. But when deployed with 64 concurrent users, GPU latency jumped 25x, throughput collapsed to 2 req/s, and worker processes were terminated by the OS. What broke under the hood?"*).
2. **The CS-to-AI Systems Bridge (The Conceptual Rosetta Stone):** Translate the failure directly into foundational CS concepts the student already mastered:
   - *Heap fragmentation in malloc* $\leftrightarrow$ External fragmentation in contiguous GPU KV-cache allocations $\leftrightarrow$ Virtual memory page tables $\leftrightarrow$ PagedAttention.
   - *OS thread scheduling & preemption* $\leftrightarrow$ Continuous batching and iteration-level scheduling in GPU inference runtimes.
   - *Browser / CPU cache hierarchy* $\leftrightarrow$ Radix tree prefix caching for multi-turn system prompts and few-shot exemplars.
3. **The Production Diagnostic Walkthrough & Code Artifact:** Walk through the diagnostic trace (terminal logs, memory counters, profiling spans) and provide the production-grade fix as a clean, idiomatic, fully-commented Python `@dataclass` or runtime controller.
4. **Transparent First-Principles Derivation & Production Rule of Thumb:** Walk through the quantitative sizing calculation with zero skipped algebra, establishing a clear, memorable "Rule of Thumb" the student can carry into engineering practice.

### 2. Boxed Apprenticeship Worked Examples
Every core section must feature at least one apprenticeship-focused boxed worked example using:
`::: {.callout-note title="Worked Example: Production Incident Diagnostic & Systems Sizing"}`
The worked example must follow a strict four-part structure:
- **The Production Incident:** Detailing real-world symptoms, system configuration, hardware parameters, and error logs.
- **Diagnostic Investigation:** Walking through the telemetry and traces to isolate the root cause using first principles.
- **Step-by-Step Sizing & Production Fix:** Providing a complete mathematical derivation showing every intermediate step and unit, followed by the concrete code or configuration remedy.
- **Production Engineering Rule of Thumb:** Delivering a clear, actionable guideline for real-world cluster deployment or runtime design.

### 3. Structural Invariant: Section .1 as Unbroken Stage-Setter
- **Section .1 of every chapter must contain ZERO subsections (NO `###`) and ZERO bulleted lists.** It serves as an unbroken, highly engaging, empathetic narrative that frames the transition from standard software engineering to production AI systems, introduces the core governing dilemma, presents the Systems Rosetta Stone table, and bridges smoothly into Section .2.
- Subsections (`###`) strictly begin in Section .2 onwards (typically 2–4 clean subheadings).

### 4. Language & Discipline Guardrails
- **Empathetic and Authoritative Engineering Tone:** Encouraging, rigorous, demystifying, practical. Write as a senior engineering mentor guiding an apprentice.
- **Authentic Production Systems Terminology:** Use real industry and runtime terms: vLLM, SGLang, continuous batching, PagedAttention, prefix caching, TTFT, inter-token latency, CUDA OOM, `seccomp-bpf`, gRPC context deadlines, Prometheus metrics.
- **Normalized 4-Outcome Status Envelope:** All model invocations return a normalized status envelope. Always use these exact uppercase enumeration names: `COMPLETED`, `TRUNCATED`, `REFUSED`, `TRANSPORT_FAILURE`.
- **Zero Ambient Authority ($A=0$):** Foundation models generate unprivileged candidate strings; production runtimes sandbox and verify every side effect.
- **Zero Anthropomorphism:** Treat models strictly as non-deterministic matrix processors proposing candidate sequences.
- **No Decorative Simulator Boilerplate:** State parameters cleanly in tables, equations, and worked examples; do not dump synthetic simulator classes into textbook prose.
- **American English:** Adhere strictly to American English spelling (`-ize`, `behavior`, `center`).
"""

# Write all outline variations
VARIANTS = {
    "V3_QUANTITATIVE": (V3_FRONTMATTER, "MASTER_OUTLINE_V3_QUANTITATIVE.md"),
    "V4_PRINCIPLES": (V4_FRONTMATTER, "MASTER_OUTLINE_V4_PRINCIPLES.md"),
    "V5_PRAGMATIC": (V5_FRONTMATTER, "MASTER_OUTLINE_V5_PRAGMATIC.md"),
    "V6_DEPENDABILITY": (V6_FRONTMATTER, "MASTER_OUTLINE_V6_DEPENDABILITY.md"),
    "V7_BUILDER": (V7_FRONTMATTER, "MASTER_OUTLINE_V7_BUILDER.md"),
    "V8_DISTRIBUTED": (V8_FRONTMATTER, "MASTER_OUTLINE_V8_DISTRIBUTED.md"),
    "V9_COMPILER": (V9_FRONTMATTER, "MASTER_OUTLINE_V9_COMPILER.md"),
    "V10_APPRENTICE": (V10_FRONTMATTER, "MASTER_OUTLINE_V10_APPRENTICE.md"),
}

OUTLINES_DIR.mkdir(parents=True, exist_ok=True)

for name, (frontmatter, filename) in VARIANTS.items():
    out_path = OUTLINES_DIR / filename
    full_outline_content = frontmatter + "\n" + COMMON_CAUSAL_CHAIN + "\n" + common_body
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_outline_content)
    print(f"✅ Generated {name}: {out_path} ({len(full_outline_content.splitlines())} lines)")

print(f"\nAll {len(VARIANTS)} Master Outline Variations successfully created!")
