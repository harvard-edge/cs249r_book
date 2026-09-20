#!/usr/bin/env python3
"""
Build MASTER_TEXTBOOK_OUTLINE_V4_MLSYS.md from MASTER_TEXTBOOK_OUTLINE_V2.md.

Transforms Volume III into a pure, authentic Agentic Machine Learning Systems textbook:
1. Establishes the CS Student Curriculum charter (Senior Undergrad / Intro Grad MLS class).
2. Eliminates all "X and Y" in Part and Chapter titles (Single-Concept Principle).
3. Strips all forced CPU / Operating Systems roleplay metaphors (no "stochastic processor",
   no "stochastic computer", no "coprocessor", no "peripherals", no "Agent OS", no "Policy Compiler").
4. Replaces them with native, authentic ML Systems language:
   - Foundation Model Engine / Inference Serving
   - Memory Systems / Paged Attention Memory
   - Execution Sandboxing / Environmental Isolation
   - Runtime Orchestration / Trajectory Persistence / Fault Recovery
   - Policy Optimization / Verifiable Reinforcement Learning
   - Distributed Systems / Serving Economics
5. Maintains 100% of the rigorous mathematical derivations, Roofline bounds, data structures,
   worked examples, and strict negative scopes from V2.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V2_PATH = REPO_ROOT / "books" / "vol3" / "MASTER_TEXTBOOK_OUTLINE_V2.md"
V4_PATH = REPO_ROOT / "books" / "vol3" / "MASTER_TEXTBOOK_OUTLINE_V4_MLSYS.md"

def build_v4():
    with open(V2_PATH, "r", encoding="utf-8") as f:
        v2_text = f.read()

    # Define the new V4 Frontmatter / Curriculum Charter
    v4_preamble = """# Agentic Machine Learning Systems: Master Curriculum Outline (V4)
**Architecture, Infrastructure, and Verifiable Engineering of Autonomous AI Systems**
*A complete chapter-by-chapter architectural blueprint and authoring specification for Chapters 1–18*

---

## Pedagogical Charter & Course Identity

### 1. Who This Textbook Is For: The Senior CS/CE Student
This textbook is written for **senior-level undergraduate Computer Science & Computer Engineering (CS/CE) majors** and **introductory graduate students** taking a course in **Agentic Machine Learning Systems** (analogous to Stanford CS 229S, UC Berkeley CS 294, or CMU MLSys).

The reader is an aspiring AI Systems Engineer. They are not looking for an abstract NLP linguistics treatise, nor a superficial "prompt engineering" user guide. They want to understand the foundational systems engineering required to design, profile, optimize, and orchestrate modern autonomous agent runtimes (such as Claude Code, Devin, and SWE-agent) and high-performance inference serving systems (such as vLLM, SGLang, and TensorRT-LLM).

### 2. Student Prerequisite Matrix

* **What the Student ALREADY Knows (Assumed Background):**
  - **Core Computer Systems:** Processes, threads, virtual memory, memory hierarchies, POSIX system calls, file descriptors, network sockets, concurrency, and Abstract Syntax Trees (ASTs).
  - **Core Programming:** High proficiency in Python and solid experience with a compiled systems language (C, C++, or Rust).
  - **Introductory Machine Learning:** Linear algebra, matrix multiplications, transformer attention math (Softmax(QK^T / sqrt(d))V), PyTorch, and calling commercial LLM APIs.

* **What the Student DOES NOT Know (What This Book Teaches From First Principles):**
  - **Inference Serving Physics:** Why autoregressive decode is memory-bandwidth bound on accelerator High-Bandwidth Memory (HBM), the Roofline model for prefill (GEMM) vs. decode (GEMV), and single-batch parameter shuttle floors.
  - **Attention & Context Memory:** The geometry of Key-Value (KV) cache tensors, PagedAttention virtual block tables, RadixAttention prefix caching trees, and chunked prefill.
  - **Agent Tool Interfaces & Sandboxing:** Typed Model Context Protocol (MCP) dispatch, streaming ring-buffer tailing, JSON Schema grammar-guided logit masking via Byte-DFAs, and isolation using Linux containers, seccomp-bpf filters, and microVMs (Firecracker).
  - **State, Durability & Policy Adaptation:** Agent execution loops (ReAct/Plan-Solve), Write-Ahead Logging (WAL) for trajectories, Saga compensating rollbacks, trajectory harvesting, SFT specialization, and Reinforcement Learning with Verifiable Rewards (RLVR).

### 3. The Single-Concept Principle: No "X and Y" Titles
Every Part and Chapter title in this curriculum is laser-focused on a **single primary systems concept**. Compound titles with "and" or "&" have been eliminated across the volume to enforce sharp jurisdictional boundaries and cognitive clarity.

### 4. The Architectural Progression & Flow (The 7-Part Journey)
The curriculum is engineered around a progressive, bottom-up systems stack that mirrors how production AI systems are constructed:

1. **Introduction: Foundations of Agentic Systems (Chapter 01):**
   *From Intent to Closed-Loop Verification:* Whole-book opening bookend and foundational reference architecture. Formalizes why an accurate foundation model fails operational tasks without an accountable computer around it, establishes the 5-part task contract, and defines the four functional subsystems.
2. **Part I: The Computational Engine (Chapters 02–03):**
   *From Context to Candidate Tokens & Deliberation:* Analyzes the physical invocation boundary and hardware Roofline limits of evaluating models on accelerator silicon (Ch 02), and expands a single model decision into test-time deliberation, search trees, and verifiers (Ch 03).
3. **Part II: The Memory Hierarchy (Chapters 04–06):**
   *From Ephemeral Context to Persistent State:* Directly addresses the memory bandwidth and capacity wall. Moves from host-side logical context sizing, budgeting, and lossy compaction (Ch 04), down to accelerator-level physical KV-cache memory management (PagedAttention block tables, Radix trees, chunked prefill) (Ch 05), out to persistent external databases and hybrid retrieval systems (Ch 06).
4. **Part III: Execution Sandboxing (Chapters 07–08) — The External Boundary:**
   *From Text Proposals to Isolated Side Effects:* Connects the model to real-world execution. Defines typed tool invocation contracts, schema-guided logit masking, and streaming observation pipelines (Ch 07), followed immediately by the indispensable systems requirement of containment: isolating untrusted model actions using Linux containers, seccomp filters, WASM, and Firecracker microVMs (Ch 08).
5. **Part IV: Runtime Orchestration (Chapters 09–11) — The State Machine:**
   *From Single Invocations to Long-Running Runtimes:* Assembles compute, memory, and sandboxed tools into an autonomous supervisory runtime. Establishes the supervisory control plane, Agent Control Block (ACB), signal handling, and scheduling (Ch 09), enforces crash resilience through Write-Ahead Logging (WAL) and state snapshots (Ch 10), and solves distributed partial failures across non-ACID external APIs using Saga compensating transactions (Ch 11).
6. **Part V: Policy Optimization (Chapters 12–14) — The Learning Flywheel:**
   *From Static Execution to Self-Improving Policies:* Explores how agent systems adapt and specialize. Harvests durable execution trajectories and feedback signals from the WAL (Ch 12), fine-tunes and distills specialized agent policies via supervised fine-tuning (Ch 13), and trains frontier reasoning capabilities via Reinforcement Learning with Verifiable Rewards (RLVR) (Ch 14).
7. **Part VI: Distributed Systems (Chapters 15–17) — Fleet Operations:**
   *From Single-Node Agents to Distributed Fleets:* Scales agent runtimes across distributed clusters. Coordinates multi-agent protocols and consensus (Ch 15), establishes distributed observability, tracing, and empirical benchmarking rigs (Ch 16), and models queue capacity, continuous batching economics, and dollar-per-task optimization (Ch 17).
8. **Part VII: System Synthesis (Chapter 18) — Capstone Reference Architecture:**
   *Closing the Systems Loop:* Unites all 17 preceding subsystems into an end-to-end production capstone (the Autonomous Software Engineering Agent), verifying all structural invariants, error budgets, and architectural trade-offs.

---

## Complete 18-Chapter Curriculum Structure

```
VOLUME III: AGENTIC MACHINE LEARNING SYSTEMS
├── Introduction: Foundations of Agentic Systems (Chapter 01)
├── Part I: The Computational Engine (Chapters 02–03)
│   ├── Chapter 02: The Foundation Model Engine
│   └── Chapter 03: Test-Time Deliberation
├── Part II: The Memory Hierarchy (Chapters 04–06)
│   ├── Chapter 04: Working Context
│   ├── Chapter 05: Paged Attention Memory
│   └── Chapter 06: Persistent Storage
├── Part III: Execution Sandboxing (Chapters 07–08)
│   ├── Chapter 07: Tool Execution
│   └── Chapter 08: Environmental Isolation
├── Part IV: Runtime Orchestration (Chapters 09–11)
│   ├── Chapter 09: Supervisory Control Planes
│   ├── Chapter 10: Trajectory Persistence
│   └── Chapter 11: Fault Recovery
├── Part V: Policy Optimization (Chapters 12–14)
│   ├── Chapter 12: Trajectory Harvesting
│   ├── Chapter 13: Supervised Adaptation
│   └── Chapter 14: Verifiable Reinforcement Learning
├── Part VI: Distributed Systems (Chapters 15–17)
│   ├── Chapter 15: Multi-Agent Coordination
│   ├── Chapter 16: System Observability
│   └── Chapter 17: Serving Economics
└── Part VII: System Synthesis (Chapter 18)
    └── Chapter 18: Architectural Synthesis
```

---
"""

    # Extract the Blueprint content from V2 (from line 540 onwards)
    # In V2, the blueprints start at "### Chapter 01: The Stochastic Computer"
    idx_ch1 = v2_text.find("### Chapter 01:")
    if idx_ch1 == -1:
        raise ValueError("Could not find '### Chapter 01:' in V2 outline.")

    blueprint_text = v2_text[idx_ch1:]

    # Transformations on the blueprint text:

    # 1. Update Part Titles
    part_replacements = [
        ("## Part I: The Stochastic Processor", "## Part I: The Computational Engine"),
        ("## Part II: Context Memory and Storage", "## Part II: The Memory Hierarchy"),
        ("## Part II: Context Memory and Serving State", "## Part II: The Memory Hierarchy"),
        ("## Part III: Tool Actuation and I/O Peripherals", "## Part III: Execution Sandboxing"),
        ("## Part III: Tools and Sandboxing", "## Part III: Execution Sandboxing"),
        ("## Part IV: The Agent Operating System", "## Part IV: Runtime Orchestration"),
        ("## Part V: The Policy Compiler", "## Part V: Policy Optimization"),
        ("## Part VI: Distributed Fleets and Operations", "## Part VI: Distributed Systems"),
        ("## Part VII: System Synthesis", "## Part VII: System Synthesis"),
    ]
    for old, new in part_replacements:
        blueprint_text = blueprint_text.replace(old, new)

    # 2. Update Chapter Headings (Single-concept, no "X and Y")
    chapter_replacements = [
        ("### Chapter 01: The Stochastic Computer (Whole-Book Opening Bookend)", "### Chapter 01: Foundations of Agentic Systems"),
        ("### Chapter 01: The Stochastic Computer", "### Chapter 01: Foundations of Agentic Systems"),
        ("### Chapter 02: The Foundation Model as a Processing Element", "### Chapter 02: The Foundation Model Engine"),
        ("### Chapter 03: Inference-Time Deliberation", "### Chapter 03: Test-Time Deliberation"),
        ("### Chapter 04: Context-Window Working Memory", "### Chapter 04: Working Context"),
        ("### Chapter 05: The KV-Cache Hierarchy", "### Chapter 05: Paged Attention Memory"),
        ("### Chapter 06: Persistent External Memory", "### Chapter 06: Persistent Storage"),
        ("### Chapter 07: Peripherals & Tool Actuation", "### Chapter 07: Tool Execution"),
        ("### Chapter 07: Peripherals &amp; Tool Actuation", "### Chapter 07: Tool Execution"),
        ("### Chapter 08: Virtualization & Sandboxing", "### Chapter 08: Environmental Isolation"),
        ("### Chapter 09: The Agent Operating System Control Plane", "### Chapter 09: Supervisory Control Planes"),
        ("### Chapter 09: The Agent OS Control Plane", "### Chapter 09: Supervisory Control Planes"),
        ("### Chapter 10: State, Persistence, and Trajectory Storage", "### Chapter 10: Trajectory Persistence"),
        ("### Chapter 11: Fault Tolerance, Compensation, and Sagas", "### Chapter 11: Fault Recovery"),
        ("### Chapter 12: Trajectory Data and Feedback", "### Chapter 12: Trajectory Harvesting"),
        ("### Chapter 13: Supervised Policy Adaptation (SFT & Distillation)", "### Chapter 13: Supervised Adaptation"),
        ("### Chapter 13: Supervised Policy Adaptation (SFT &amp; Distillation)", "### Chapter 13: Supervised Adaptation"),
        ("### Chapter 13: Supervised Policy Adaptation", "### Chapter 13: Supervised Adaptation"),
        ("### Chapter 14: Reinforcement Learning with Verifiable Rewards (RLVR)", "### Chapter 14: Verifiable Reinforcement Learning"),
        ("### Chapter 14: Reinforcement Learning with Verifiable Rewards", "### Chapter 14: Verifiable Reinforcement Learning"),
        ("### Chapter 15: Multi-Agent Fleets and Coordination", "### Chapter 15: Multi-Agent Coordination"),
        ("### Chapter 16: Distributed Observability and Empirical Evaluation", "### Chapter 16: System Observability"),
        ("### Chapter 17: Performance, Cost, and Fleet Economics", "### Chapter 17: Serving Economics"),
        ("### Chapter 18: System Synthesis: Designing the Stochastic Computer", "### Chapter 18: Architectural Synthesis"),
        ("### Chapter 18: Designing the Stochastic Computer", "### Chapter 18: Architectural Synthesis"),
    ]
    for old, new in chapter_replacements:
        blueprint_text = blueprint_text.replace(old, new)

    # 3. Update Section Headings (Single-concept, no "X and Y", no "processor", no "peripheral")
    section_replacements = [
        # Chapter 01
        ("#### Section 1.3: Software 1.0, 2.0, and 3.0 [core]", "#### Section 1.3: Software Paradigm Evolution [core]"),
        ("## Software 1.0, 2.0, and 3.0 {#sec-vol3-intro-the-tripartite-systems-comparison}", "## Software Paradigm Evolution {#sec-vol3-intro-the-tripartite-systems-comparison}"),
        ("#### Section 1.9: The Stochastic Computer [core]", "#### Section 1.9: The Systems Architecture Blueprint [core]"),
        ("## The Stochastic Computer {#sec-vol3-intro-stochastic-computer}", "## The Systems Architecture Blueprint {#sec-vol3-intro-stochastic-computer}"),
        # Chapter 02
        ("#### Section 2.1: A Model Call in the Stochastic Computer [stage-setter]", "#### Section 2.1: The Model Invocation Boundary [stage-setter]"),
        ("## A Model Call in the Stochastic Computer {#sec-vol3-processor-role}", "## The Model Invocation Boundary {#sec-vol3-processor-role}"),
        ("#### Section 2.2: Tokens as the Processor Interface [core]", "#### Section 2.2: Discrete Token Representation [core]"),
        ("## Tokens as the Processor Interface {#sec-vol3-processor-tokenization}", "## Discrete Token Representation {#sec-vol3-processor-tokenization}"),
        ("#### Section 2.3: Next-Token Computation [core]", "#### Section 2.3: Autoregressive Generation [core]"),
        ("## Next-Token Computation {#sec-vol3-processor-autoregressive}", "## Autoregressive Generation {#sec-vol3-processor-autoregressive}"),
        ("#### Section 2.4: Candidate Sequences Versus Valid Conclusions [core]", "#### Section 2.4: Candidate Sequence Verification [core]"),
        ("## Candidate Sequences Versus Valid Conclusions {#sec-vol3-processor-continuations}", "## Candidate Sequence Verification {#sec-vol3-processor-continuations}"),
        ("#### Section 2.5: The Invocation Contract [core]", "#### Section 2.5: The Invocation Contract [core]"),
        ("## The Invocation Contract {#sec-vol3-processor-contract}", "## The Invocation Contract {#sec-vol3-processor-contract}"),
        ("#### Section 2.6: Constraining the Output Surface [core]", "#### Section 2.6: Grammar-Guided Decoding [core]"),
        ("## Constraining the Output Surface {#sec-vol3-processor-grammar-constrained}", "## Grammar-Guided Decoding {#sec-vol3-processor-grammar-constrained}"),
        ("#### Section 2.7: The Cost of an Invocation [core]", "#### Section 2.7: Accelerator Serving Latency [core]"),
        ("## The Cost of an Invocation {#sec-vol3-processor-cost}", "## Accelerator Serving Latency {#sec-vol3-processor-cost}"),
        ("#### Section 2.8: Processor Interface Evaluation [synthesis]", "#### Section 2.8: Interface Benchmarking [synthesis]"),
        ("## Processor Interface Evaluation {#sec-vol3-processor-interface-design}", "## Interface Benchmarking {#sec-vol3-processor-interface-design}"),
        # Chapter 03
        ("#### Section 3.3: Candidate Diversity and Selection [core]", "#### Section 3.3: Candidate Selection [core]"),
        ("## Candidate Diversity and Selection {#sec-vol3-deliberation-candidate-diversity}", "## Candidate Selection {#sec-vol3-deliberation-candidate-diversity}"),
        ("#### Section 3.4: Verification and Its Failure Modes [core]", "#### Section 3.4: Process Verification [core]"),
        ("## Verification and Its Failure Modes {#sec-vol3-deliberation-verification}", "## Process Verification {#sec-vol3-deliberation-verification}"),
        ("#### Section 3.6: Bounded Search and Stopping Rules [core]", "#### Section 3.6: Search Stopping Criteria [core]"),
        ("## Bounded Search and Stopping {#sec-vol3-deliberation-bounded-search}", "## Search Stopping Criteria {#sec-vol3-deliberation-bounded-search}"),
        # Chapter 04
        ("#### Section 4.2: Capacity, Cost, and Relevance [core]", "#### Section 4.2: Working Set Capacity [core]"),
        ("## Capacity, Cost, and Relevance {#sec-vol3-working-sets-physics}", "## Working Set Capacity {#sec-vol3-working-sets-physics}"),
        ("#### Section 4.4: Filtering and Lossy Compaction [core]", "#### Section 4.4: Context Compaction [core]"),
        ("## Filtering and Lossy Compaction {#sec-vol3-working-sets-compaction}", "## Context Compaction {#sec-vol3-working-sets-compaction}"),
        ("#### Section 4.5: Working Buffers and Checkpoint Summaries [core]", "#### Section 4.5: Context Checkpointing [core]"),
        ("## Working Buffers and Checkpoint Summaries {#sec-vol3-working-sets-summarization}", "## Context Checkpointing {#sec-vol3-working-sets-summarization}"),
        ("#### Section 4.6: Freshness and Context Invalidation [core]", "#### Section 4.6: Context Invalidation [core]"),
        ("## Freshness and Context Invalidation {#sec-vol3-working-sets-context-rot}", "## Context Invalidation {#sec-vol3-working-sets-context-rot}"),
        # Chapter 05
        ("#### Section 5.2: Dynamic Allocation and Fragmentation [core]", "#### Section 5.2: Memory Fragmentation [core]"),
        ("## Dynamic Allocation and Fragmentation {#sec-vol3-kvcache-fragmentation}", "## Memory Fragmentation {#sec-vol3-kvcache-fragmentation}"),
        ("#### Section 5.4: Prefix Identity and Reuse [core]", "#### Section 5.4: Prefix Caching [core]"),
        ("## Prefix Identity and Reuse {#sec-vol3-kvcache-radix-tree}", "## Prefix Caching {#sec-vol3-kvcache-radix-tree}"),
        ("#### Section 5.5: Prefill and Decode Scheduling [core]", "#### Section 5.5: Chunked Prefill Scheduling [core]"),
        ("## Prefill and Decode Scheduling {#sec-vol3-kvcache-chunked-prefill}", "## Chunked Prefill Scheduling {#sec-vol3-kvcache-chunked-prefill}"),
        ("#### Section 5.7: Provisioning and Measurement [core]", "#### Section 5.7: Cache Capacity Provisioning [core]"),
        ("## Provisioning and Measurement {#sec-vol3-kvcache-capacity-planning}", "## Cache Capacity Provisioning {#sec-vol3-kvcache-capacity-planning}"),
        # Chapter 06
        ("#### Section 6.3: Exact and Structured Retrieval [core]", "#### Section 6.3: Structured Retrieval [core]"),
        ("## Exact and Structured Retrieval {#sec-vol3-persistent-bm25}", "## Structured Retrieval {#sec-vol3-persistent-bm25}"),
        ("#### Section 6.4: Semantic and Hybrid Retrieval [core]", "#### Section 6.4: Hybrid Retrieval [core]"),
        ("## Semantic and Hybrid Retrieval {#sec-vol3-persistent-dense-retrieval}", "## Hybrid Retrieval {#sec-vol3-persistent-dense-retrieval}"),
        ("#### Section 6.5: Relationships and Multi-Hop Evidence [core]", "#### Section 6.5: Multi-Hop Retrieval [core]"),
        ("## Relationships and Multi-Hop Evidence {#sec-vol3-persistent-graphrag}", "## Multi-Hop Retrieval {#sec-vol3-persistent-graphrag}"),
        ("#### Section 6.6: Writes, Freshness, and Invalidation [core]", "#### Section 6.6: Storage Invalidation [core]"),
        ("## Writes, Freshness, and Invalidation {#sec-vol3-persistent-invalidation}", "## Storage Invalidation {#sec-vol3-persistent-invalidation}"),
        ("#### Section 6.7: Governance and Retrieval Evaluation [core]", "#### Section 6.7: Retrieval Evaluation [core]"),
        ("## Governance and Retrieval Evaluation {#sec-vol3-persistent-governance-eval}", "## Retrieval Evaluation {#sec-vol3-persistent-governance-eval}"),
        # Chapter 07
        ("#### Section 7.1: Peripheral Subsystem Abstraction [stage-setter]", "#### Section 7.1: Tool Subsystem Architecture [stage-setter]"),
        ("## Peripheral Subsystem Abstraction {#sec-vol3-actuation-unix-analogy}", "## Tool Subsystem Architecture {#sec-vol3-actuation-unix-analogy}"),
        # Chapter 11
        ("#### Section 11.1: The Transactional Boundary and ACID Collapse [stage-setter]", "#### Section 11.1: The Transactional Boundary Collapse [stage-setter]"),
        ("## The Transactional Boundary and ACID Collapse {#sec-vol3-sagas-acid-boundary}", "## The Transactional Boundary Collapse {#sec-vol3-sagas-acid-boundary}"),
        ("#### Section 11.6: Peripheral Circuit Breakers [core]", "#### Section 11.6: Tool Circuit Breakers [core]"),
        ("## Peripheral Circuit Breakers {#sec-vol3-sagas-containment}", "## Tool Circuit Breakers {#sec-vol3-sagas-containment}"),
        # Chapter 17
        ("#### Section 17.8: Fleet Performance and Serving Economics Synthesis [synthesis]", "#### Section 17.8: Serving Economics Synthesis [synthesis]"),
        ("## Fleet Performance and Serving Economics Synthesis {#sec-vol3-tokenomics-synthesis}", "## Serving Economics Synthesis {#sec-vol3-tokenomics-synthesis}"),
    ]
    for old, new in section_replacements:
        blueprint_text = blueprint_text.replace(old, new)

    # 4. Clean up Subsystems Active lines across all 18 chapters
    active_replacements = [
        ("The Stochastic Computer", "Foundations of Agentic Systems"),
        ("The Foundation Model as a Processing Element", "The Foundation Model Engine"),
        ("The Stochastic Processor Core", "The Foundation Model Engine"),
        ("Stochastic Processor Core", "The Foundation Model Engine"),
        ("Processor Core and Deliberation", "The Foundation Model Engine and Test-Time Deliberation"),
        ("Inference-Time Deliberation", "Test-Time Deliberation"),
        ("Context-Window Working Memory", "Working Context"),
        ("The KV-Cache Hierarchy", "Paged Attention Memory"),
        ("Persistent External Memory", "Persistent Storage"),
        ("Peripherals & Tool Actuation", "Tool Execution"),
        ("Tool Peripherals", "Tool Execution"),
        ("Virtualization & Sandboxing", "Environmental Isolation"),
        ("The Agent OS Control Plane", "Runtime Orchestration"),
        ("Agent OS Control Plane", "Runtime Orchestration"),
        ("State, Persistence, and Trajectory Storage", "Trajectory Persistence"),
        ("Fault Tolerance, Compensation, and Sagas", "Fault Recovery"),
        ("Fault Tolerance/Sagas", "Fault Recovery"),
        ("Trajectory Data and Feedback", "Trajectory Harvesting"),
        ("Data Flywheel", "Trajectory Harvesting"),
        ("Supervised Policy Adaptation", "Supervised Adaptation"),
        ("SFT Distillation", "Supervised Adaptation"),
        ("Reinforcement Learning with Verifiable Rewards", "Verifiable Reinforcement Learning"),
        ("Multi-Agent Fleets and Coordination", "Multi-Agent Coordination"),
        ("Multi-Agent Fleets", "Multi-Agent Coordination"),
        ("Distributed Observability and Empirical Evaluation", "System Observability"),
        ("Performance, Cost, and Fleet Economics", "Serving Economics"),
        ("Fleet Economics", "Serving Economics"),
        ("Designing the Stochastic Computer", "Architectural Synthesis"),
    ]
    for old, new in active_replacements:
        blueprint_text = blueprint_text.replace(old, new)

    # 5. Clean up metaphor terminology and peripheral occurrences in the blueprint text
    text_replacements = [
        ("The Stochastic Computer", "Foundations of Agentic Systems"),
        ("the Stochastic Computer", "the agentic system"),
        ("Stochastic Computer", "Agentic System"),
        ("stochastic computer systems", "autonomous agent systems"),
        ("stochastic computers", "agentic systems"),
        ("stochastic computer", "agentic system"),
        ("Stochastic Processor", "Foundation Model Engine"),
        ("stochastic processor", "foundation model engine"),
        ("processor core", "inference engine"),
        ("Processor Core", "Inference Engine"),
        ("processing element", "inference engine"),
        ("Processing Element", "Inference Engine"),
        ("an unprivileged mathematical coprocessor", "an unprivileged neural inference engine"),
        ("unprivileged coprocessor", "unprivileged inference engine"),
        ("coprocessor", "inference engine"),
        ("Coprocessor", "Inference Engine"),
        ("Tool Peripherals", "Tool Interfaces"),
        ("tool peripherals", "tool interfaces"),
        ("sandboxed peripherals", "sandboxed tool interfaces"),
        ("Sandboxed Peripherals", "Execution Sandboxing"),
        ("sandboxed peripherals", "execution sandboxing"),
        ("Agent OS", "Agent Runtime"),
        ("agent OS", "agent runtime"),
        ("Operating System Runtime", "Agent Runtime"),
        ("operating system runtime", "agent runtime"),
        ("The Policy Compiler", "Policy Optimization"),
        ("the Policy Compiler", "policy optimization"),
        ("policy compiler", "policy optimization pipeline"),
        ("Distributed Fleets and Operations", "Distributed Systems"),
        ("Operating System Control Planes and Fault Tolerance", "Runtime Orchestration"),
        ("Working Memory and Persistent Stores", "Working Context and Persistent Storage"),
        ("Tool Actuation and Sandboxing", "Tool Execution and Isolation"),
        ("Part I: The Foundation Model Engine", "Part I: The Computational Engine"),
        ("Part II: Context Memory & Storage", "Part II: The Memory Hierarchy"),
        ("Part III: Tool Actuation & I/O Peripherals", "Part III: Execution Sandboxing"),
        ("Part IV: The Agent Operating System", "Part IV: Runtime Orchestration"),
        ("Part VI: Distributed Fleets & Operations", "Part VI: Distributed Systems"),
        ("Part VII: Synthesis", "Part VII: System Synthesis"),
        # Peripheral cleanup
        ("Peripheral Subsystem Abstraction", "Tool Subsystem Architecture"),
        ("Peripheral subsystem abstraction", "Tool subsystem architecture"),
        ("peripheral integration protocol", "tool integration protocol"),
        ("peripheral processes", "tool processes"),
        ("peripheral output streams", "tool execution output streams"),
        ("peripheral tool execution", "tool execution"),
        ("synthesizing peripheral I/O and tool actuation", "synthesizing tool interfaces and execution sandboxing"),
        ('Tools are the peripheral devices of the agentic system: typed schemas define their I/O bus.', 'Tools are external execution endpoints: typed schemas define their interface contract.'),
        ("standardizes peripheral connectivity across clients and servers", "standardizes external tool and resource connectivity across clients and servers"),
        ("mutating peripheral calls", "mutating tool calls"),
        ("mutating peripheral action", "mutating tool action"),
        ("mutating peripheral call", "mutating tool call"),
        ("From Peripheral Connectivity to Hardware Isolation", "From Tool Interfaces to Execution Isolation"),
        ("peripheral interfaces, typed tool schemas", "tool interfaces, typed schemas"),
        ("peripheral contracts", "tool contracts"),
        ("From Isolated Peripherals to the Agent Operating System", "From Isolated Tools to Runtime Orchestration"),
        ("peripherals, and virtualization sandboxing", "tool execution, and virtualization sandboxing"),
        ("WAITING_IO: Suspended awaiting asynchronous peripheral completion", "WAITING_IO: Suspended awaiting asynchronous tool completion"),
        ("The Human-as-an-Asynchronous-Peripheral Architecture", "The Human-in-the-Loop Asynchronous Gateway Architecture"),
        ("decouples compute allocation from peripheral latency", "decouples compute allocation from tool execution latency"),
        ("external peripheral dispatch", "external tool dispatch"),
        ("Peripheral Mutation:", "External Environment Mutation:"),
        ("Peripheral Lease Re-binding:", "Tool Channel Re-binding:"),
        ("peripheral circuit breakers", "tool circuit breakers"),
        ("Peripheral Circuit Breakers", "Tool Circuit Breakers"),
        ("Peripheral fault isolation", "Tool fault isolation"),
        ("Tool & Peripheral Manifest:", "Tool Interface Manifest:"),
        ("operating system, memory, peripheral, and verification layers", "runtime orchestration, memory, execution sandbox, and verification layers"),
        ("Peripherals & Sandboxing (mediated tool dispatch)", "Execution Sandboxing (mediated tool dispatch)"),
        ("Part III (Tool Actuation and I/O Peripherals)", "Part III (Execution Sandboxing)"),
        ("standardized peripheral protocols (Model Context Protocol / MCP)", "standardized tool integration protocols (Model Context Protocol / MCP)"),
        ("Establishes the peripheral boundary of the agentic system", "Establishes the external execution boundary of the agentic system"),
        ("The Peripheral Boundary. Moving from the internal", "The External Execution Boundary. Moving from the internal"),
        ('The Agent Peripheral Interface: "everything is an RPC tool dispatch"', 'The Agent Tool Interface: "everything is an RPC tool dispatch"'),
        ("The 4-Stage Peripheral Lifecycle:", "The 4-Stage Tool Execution Lifecycle:"),
        ("contract for peripheral tool definition.", "contract for tool definition."),
        ('The Classical I/O Abstraction (Ritchie & Thompson 1974: "everything is a file" via `open`, `read`, `write`, `close`, `ioctl`) versus the Agent Peripheral Interface:', 'The Classical I/O Abstraction (Ritchie & Thompson 1974: "everything is a file" via `open`, `read`, `write`, `close`, `ioctl`) versus the Agent Tool Interface:'),
        ("`WAITING_IO`: Suspended awaiting asynchronous peripheral completion (Ch 7).", "`WAITING_IO`: Suspended awaiting asynchronous tool completion (Ch 7)."),
        ("Synthesizing peripheral I/O and tool actuation.", "Synthesizing tool interfaces and execution sandboxing."),
        ("Processor, Memory Hierarchy, Sandboxed Peripherals, Operating System Governance", "The Computational Engine, The Memory Hierarchy, Execution Sandboxing, Runtime Orchestration"),
        ("peripherals, operating system, and learning compiler", "execution sandboxes, supervisory runtimes, and policy optimization"),
        ("How do all eighteen subsystems—processor, memory, peripherals, operating system, compilers, and fleet telemetry—integrate into an accountable, end-to-end production agentic system?", "How do all eighteen subsystems—computational engine, memory hierarchy, execution sandboxes, supervisory runtimes, policy optimization, and distributed fleets—integrate into an accountable, end-to-end production agentic system?"),
        ("the The Foundation Model Engine (Part I)", "The Computational Engine (Part I)"),
        ("Working Context and Persistent Storage (Part II)", "The Memory Hierarchy (Part II)"),
        ("Tool Execution and Isolation (Part III)", "Execution Sandboxing (Part III)"),
        ("Trajectory Feedback and Policy Compilation (Part V)", "Policy Optimization (Part V)"),
        ("Distributed Fleet Coordination, Observability, and Economics (Part VI)", "Distributed Systems (Part VI)"),
        ("Capstone Synthesis of the agentic system.", "Capstone Synthesis of Agentic Systems."),
        ("This chapter defines the operating system policies for working-set selection", "This chapter defines the runtime memory management policies for working-set selection"),
        ("This chapter synthesizes them into an operating system control plane.", "This chapter synthesizes them into a supervisory runtime control plane."),
        ("staged for the processor.", "staged for the inference engine."),
        ("@fig-stochastic-processor-core", "@fig-inference-engine-architecture"),
        ("Agent Operating System scheduler and ACB queues", "Supervisory runtime scheduler and ACB queues"),
        ("In Part IV (*The Agent Operating System*), Chapter 09 (*Runtime Orchestration*)", "In Part IV (*Runtime Orchestration*), Chapter 09 (*Supervisory Control Planes*)"),
        ("Subsystem Under Construction: Part IV, Chapter 09 (The Agent Operating System Control Plane).", "Subsystem Under Construction: Part IV, Chapter 09 (Supervisory Control Planes)."),
        ("Part IV (The Agent Operating System).", "Part IV (Runtime Orchestration)."),
        ("Processor, Context Memory, KV Cache, Tool Sandbox, OS Supervisor, WAL Storage, Sagas, and Canary Gates.", "Inference Engine, Context Memory, KV Cache, Tool Sandbox, Supervisory Runtime, WAL Storage, Sagas, and Canary Gates."),
        ("- *The Limits of the Silicon Metaphor:* Clarifying that the agentic system is a software functional architecture, not a physical chip; rejecting forced 1-to-1 mappings of transformers to CPUs or tokens to opcodes.", "- *The Software Systems Architecture:* Establishing the agentic system as a software systems architecture running on host operating systems and matrix accelerators; delineating the boundary between accelerator compute and host runtime governance."),
        ("Table: responsibility, owner, input, output, and verification boundary for each live component; replace one-to-one silicon mappings.", "Table: responsibility, owner, input, output, and verification boundary for each live component."),
        ("Figure: The Functional Architecture of the agentic system [insert link here: books/vol3/01_introduction/images/svg/stochastic_computer_architecture.svg], revised around the execution loop.", "Figure: The Functional Architecture of the Agentic System [insert link here: books/vol3/01_introduction/images/svg/agentic_systems_architecture.svg], illustrating the closed-loop execution flow across inference, memory, sandboxing, and runtime orchestration."),
        ("Machine instructions / ALU ops ($10^{-9}\\text{ s}$)", "Machine instructions ($10^{-9}\\text{ s}$)"),
        ("The Single Key Point: Foundations of Agentic Systems is a software-level functional architecture", "The Single Key Point: The agentic system is a software-level functional architecture"),
        ("Trajectory Trajectory Harvesting, Policy Compiler (SFT/RLVR), Distributed Fleet Operations.", "Trajectory Harvesting (Ch 12), Policy Optimization (Ch 13–14), Distributed Fleet Operations (Ch 15–17)."),
        ("Synthesize the four fundamental subsystems (Processor, Memory Hierarchy, Execution Sandboxing, Operating System Governance) and two lifecycle pillars (Policy Compiler, Fleet Operations) into a unified production reference architecture.", "Synthesize the four fundamental subsystems (The Computational Engine, The Memory Hierarchy, Execution Sandboxing, Runtime Orchestration) and two lifecycle pillars (Policy Optimization, Distributed Systems) into a unified production reference architecture."),
        ("explicitly mapping every action to its subsystem owner (Processor, Memory, Tools, OS, Sagas, Telemetry).", "explicitly mapping every action to its subsystem owner (Computational Engine, Memory Hierarchy, Execution Sandboxes, Supervisory Runtime, Sagas, Telemetry)."),
        ("- Subsystem Under Construction: Part I, Chapter 01 (Foundations of Agentic Systems - Whole-Book Opening Bookend).", "- Subsystem Under Construction: Introduction, Chapter 01 (Foundations of Agentic Systems - Whole-Book Opening Bookend)."),
    ]
    for old, new in text_replacements:
        blueprint_text = blueprint_text.replace(old, new)

    # Assemble complete V4 outline
    v4_full = v4_preamble + blueprint_text

    with open(V4_PATH, "w", encoding="utf-8") as f:
        f.write(v4_full)

    print(f"Successfully generated V4 outline at: {V4_PATH}")
    print(f"Total lines: {len(v4_full.splitlines())}")

if __name__ == "__main__":
    build_v4()
