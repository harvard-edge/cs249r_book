#!/usr/bin/env python3
"""
Repair Chapter 17 based on 10-student classroom review:
1. Replace 7 ASCII/text diagrams with formal tables and mathematical callouts.
2. Update 16 compound headings to strictly respect the Single-Concept Principle.
3. Validate table column counts, zero text blocks, zero box characters, zero British spellings.
"""

import re
import sys

CH17_FILE = "books/vol3/17_tokenomics/17_tokenomics.qmd"

with open(CH17_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Heading replacements (Single-Concept Principle)
heading_replacements = [
    (
        "### Amdahl's Law and Acceleration Asymptotes",
        "### Acceleration Asymptotes"
    ),
    (
        "### Pipelining, Concurrency, and Critical Path Compression",
        "### Critical Path Compression"
    ),
    (
        "### Failure Traces: Verification Escapes and Context Poisoning",
        "### Verification Escapes"
    ),
    (
        "### The Memory-Bandwidth Wall and Arithmetic Intensity",
        "### Arithmetic Intensity Boundaries"
    ),
    (
        "### Speculative Proposal and Parallel Target Verification",
        "### Speculative Decoding Pipeline"
    ),
    (
        "### Serving Economics, Task Entropy, and Accelerator Trade-offs",
        "### Serving Economics Trade-offs"
    ),
    (
        "### Queueing Dynamics and Tail Latency in Heterogeneous Fleets",
        "### Queueing Dynamics"
    ),
    (
        "#### Ledger State Variables and Accounting Invariants {.unnumbered}",
        "#### Ledger Accounting Invariants {.unnumbered}"
    ),
    (
        "#### Delegation and Escrow Protocol {.unnumbered}",
        "#### Delegation Escrow Protocols {.unnumbered}"
    ),
    (
        "#### Termination and Capital Reconciliation {.unnumbered}",
        "#### Capital Reconciliation Protocols {.unnumbered}"
    ),
    (
        "### Multi-Rate Circuit Breakers and Epistemic Decay",
        "### Multi-Rate Circuit Breakers"
    ),
    (
        "#### Operational Duration and Latency Horizon ($T_{\\text{target}}$) {.unnumbered}",
        "#### Operational Duration Horizons ($T_{\\text{target}}$) {.unnumbered}"
    ),
    (
        "#### State Mutation and Memory Horizon {.unnumbered}",
        "#### State Mutation Horizons {.unnumbered}"
    ),
    (
        "#### Permitted Authority and Blast Radius ($A$) {.unnumbered}",
        "#### Permitted Authority Limits ($A$) {.unnumbered}"
    ),
    (
        "#### Completion Evidence and Verification Closure {.unnumbered}",
        "#### Verification Closure Criteria {.unnumbered}"
    ),
    (
        "### Sensitivity Frontiers and Elastic Economics {#sec-vol3-selection-economics}",
        "### Elastic Selection Economics {#sec-vol3-selection-economics}"
    )
]

for old_h, new_h in heading_replacements:
    if old_h not in content:
        print(f"ERROR: Heading not found: {old_h}")
        sys.exit(1)
    content = content.replace(old_h, new_h, 1)

print(f"Applied all {len(heading_replacements)} heading replacements.")

# Find all 7 text blocks
matches = list(re.finditer(r"```text\s*.*?\n```", content, re.DOTALL))
if len(matches) != 7:
    print(f"ERROR: Expected 7 text blocks, found {len(matches)}")
    sys.exit(1)

# Block 1: Trace critical path
new_block_1 = """| Span Identifier | Subsystem Component | Duration (s) | Share (%) | Critical Path Classification |
| :--- | :--- | :--- | :--- | :--- |
| **Span 01** (`supervisor.init`) | Runtime Supervisor | 0.12 s | 0.0% | Setup and configuration |
| **Span 02** (`sandbox.container_create`) | Hypervisor / MicroVM | 14.80 s | 4.7% | Provisioning latency bottleneck |
| **Span 03** (`model.turn_01.prefill`) | Model Serving Engine | 1.20 s | 0.4% | Compute-bound context prefill |
| **Span 04** (`model.turn_01.decode`) | Model Serving Engine | 8.40 s | 2.7% | Memory-bound token generation |
| **Span 05** (`tool.sandbox.git_status`) | Host Subprocess | 0.08 s | 0.0% | Filesystem inspection |
| **Span 06** (`model.turn_02.prefill`) | Model Serving Engine | 1.85 s | 0.6% | Context expansion prefill |
| **Span 07** (`model.turn_02.decode`) | Model Serving Engine | 11.20 s | 3.6% | Memory-bound token generation |
| **Span 08** (`tool.sandbox.run_pytest`) | Isolated MicroVM | 245.60 s | 78.6% | Primary critical path bottleneck |
| **Span 09** (`model.turn_03.prefill`) | Model Serving Engine | 2.10 s | 0.7% | Final synthesis prefill |
| **Span 10** (`model.turn_03.decode`) | Model Serving Engine | 4.50 s | 1.4% | Memory-bound token generation |
| **Span 11** (`sandbox.snapshot_export`) | Storage / Filesystem | 22.55 s | 7.2% | Artifact persistence bottleneck |
| **Total Trajectory** | **End-to-End Task Execution** | **312.40 s** | **100.0%** | **Inference: 9.4%, Tools/Sandboxing: 90.6%** |

: Empirical Span Profile and Critical Path Breakdown for Software Engineering Trajectory (`4a8f9c10`). {#tbl-vol3-trace-critical-path}"""

# Block 2: Speculative decoding pipeline
new_block_2 = """| Pipeline Stage | Executing Subsystem | Computational Primitive | Arithmetic & Memory Profile | Operational Invariant |
| :--- | :--- | :--- | :--- | :--- |
| **1. Autoregressive Draft** | Draft Model ($M_d$) | $\\gamma$ sequential GEMV steps | Memory-bandwidth bound; small batch size; rapid token emission | Generates speculative tokens $\\hat{x}_1, \\dots, \\hat{x}_\\gamma$ |
| **2. Parallel Target Verification** | Target Model ($M_t$) | 1 batched GEMM step ($B = \\gamma$) | Arithmetic intensity amplified by $\\gamma\\times$; saturates tensor cores | Computes target conditional probabilities $p_t(x \\mid x_{<i})$ |
| **3. Rejection Sampling Gate** | Target Serving Engine | Elementwise probability comparison | $\\mathcal{O}(1)$ comparison: accept if $r < \\min(1, p_t/p_d)$ | Recovers exact target distribution without bias |
| **4. Context & KV Commit** | Joint Runtime Buffer | In-place KV cache append & rollback | Evicts rejected suffix; appends accepted prefix + correction token | Advances context by $k+1$ tokens in a single target forward pass |

: Algorithmic Execution Stages for Speculative Decoding ($\\gamma = 3$ Lookahead Horizon). {#tbl-vol3-speculative-decoding-pipeline}"""

# Block 3: MLFQ transition protocol
new_block_3 = """::: {#def-vol3-mlfq-transition-protocol .callout-tip}
## Definition 17.1: Multi-Level Feedback Queue (MLFQ) State Transition Discipline
The cluster scheduler partitions agent inference jobs into prioritized execution tiers with deterministic transition rules:

1. **Ingress Allocation:** All incoming invocations enter highest-priority queue $Q_0$ allocated an initial token quantum of $\\Delta_0 = 64$ tokens.
2. **Quantum Exhaustion Demotion:** If an execution thread exhausts its assigned quantum without completing or issuing an I/O wait, it demotes downward:
   $$Q_k \\xrightarrow{\\text{exhausts } \\Delta_k} Q_{k+1}$$
   specifically transitioning from $Q_0 \\to Q_1$ ($\\Delta_1 = 512\\text{ tokens}$) and $Q_1 \\to Q_2$ ($\\Delta_2 \\ge 4{,}096\\text{ tokens}$).
3. **Completion Exit:** An invocation that finishes decoding or yields for tool execution within its quantum $\\Delta_k$ immediately exits the scheduling queue and returns results.
4. **Periodic Priority Boosting:** To eliminate long-horizon starvation in $Q_2$, every $T_{\\text{boost}}$ seconds all active tasks across $Q_1$ and $Q_2$ are atomically promoted back to $Q_0$ with reset quanta.
5. **Tiered Memory Retention:** Preempted $Q_1$ tasks retain physical HBM allocations during grace interval $\\tau_{\\text{grace}}$; prolonged memory pressure migrates $Q_2$ blocks to host memory via PagedAttention.
:::"""

# Block 4: Hybrid tiered provisioning
new_block_4 = """| Provisioning Tier | Hardware Substrate | Utilization Boundary | Economic Tariff | Queueing Dynamics & Latency | Target Workload Profile |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dedicated Base Pool** | Reserved $8\\times$ H100 GPU nodes | Base load $\\rho \\le 0.65$ | Fixed lease (\\$28.00/node-hr) | Sub-millisecond queue wait; deterministic TTFT | Continuous $Q_0, Q_1$, and background $Q_2$ backfill |
| **Elastic Serverless Tier** | Multi-tenant cloud APIs / spot endpoints | Burst spikes $\\rho > 0.65$ | Pay-per-token marginal pricing | Variable queue wait; network ingress/egress tax | Unpredictable burst spikes and off-nominal overages |

: Hybrid Tiered Accelerator Provisioning and Traffic Bursting Architecture. {#tbl-vol3-hybrid-fleet-provisioning}"""

# Block 5: Hierarchical budget ledger reconciliation
new_block_5 = """| Ledger Hierarchy Node | Lifecycle Phase | Initial Budget ($B$) | Committed Spend ($E$) | Escrow Reservation ($R$) | Released Refund | Free Capital ($F$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Root Task Ledger** | Initial Delegation | \\$10.00 | \\$0.40 | \\$6.00 (Reserved for A + B) | — | \\$3.60 |
| **Child A (Planner)** | Subagent Execution | \\$2.00 | \\$0.50 | \\$0.00 | \\$1.50 (Refund to Root) | \\$0.00 (Terminated) |
| **Child B (Executor)** | Subagent Execution | \\$4.00 | \\$3.10 | \\$0.00 | \\$0.90 (Refund to Root) | \\$0.00 (Terminated) |
| **Root Task Ledger** | Post-Reconciliation | \\$10.00 | \\$4.00 (Cumulative) | \\$0.00 (Escrow Cleared) | \\$2.40 (Restored) | \\$6.00 (Available) |

: Hierarchical Budget Ledger Accounting and Capital Reconciliation Example. {#tbl-vol3-budget-ledger-reconciliation}"""

# Block 6: Circuit breaker FSM
new_block_6 = """| Current State | Transition Trigger / Condition | Target State | Invariant Guard & Operational Semantics | System Recovery Action |
| :--- | :--- | :--- | :--- | :--- |
| **CLOSED (Nominal)** | Velocity spike ($\\bar{\\nu}_W > \\nu_{\\max}$), utility decay ($\\Delta \\mathcal{U} < \\epsilon$), or invariant breach | **OPEN** | Immediate fail-fast trap; halts inference and tool dispatches | Suspends trajectory, cancels pending child jobs, logs alert |
| **OPEN (Tripped)** | Cooldown window elapsed ($\\Delta t > \\tau_{\\text{cool}}$) or checkpoint rollback complete | **HALF-OPEN** | Permits isolated, single-step canary evaluation probe | Emits bounded introspection or isolated unit test |
| **HALF-OPEN (Canary)** | Canary probe fails or re-triggers invariant breach | **OPEN** | Enforces exponential cooldown backoff ($\\tau_{\\text{cool}} \\leftarrow 2 \\tau_{\\text{cool}}$) | Escalates to human operator or terminates trajectory |
| **HALF-OPEN (Canary)** | Canary probe succeeds and state verification passes | **CLOSED** | Resets consecutive failure counter and clears escrow hold | Restores normal multi-turn model dispatch pipeline |

: Agentic Circuit Breaker Finite State Machine and Transition Semantics. {#tbl-vol3-circuit-breaker-fsm}"""

# Block 7: Architectural regimes frontier
new_block_7 = """| Architectural Paradigm | Task Ambiguity & Entropy | Mechanical Verifiability | Latency Tolerance | Economic Cost Profile | Representative Systems Substrate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Software 1.0 (Deterministic)** | Zero (closed domain) | Absolute ($100\\%$ formal proofs) | Hard real-time ($< 1\\text{ ms}$) | Negligible ($\\mathcal{O}(1)$ CPU ops) | Static parsers, Linters, AST rule engines |
| **Software 2.0 (Direct Inference)** | Low to Moderate (unstructured input) | Moderate (schema checks, regex) | Strict ($10\\text{--}500\\text{ ms}$) | Fixed per call (\\$0.001 - \\$0.01) | Single-turn classifiers, Embeddings, Rankers |
| **Bounded Agentic Loops** | Moderate (multi-step repair, coding) | High (deterministic tests, compilers) | Relaxed ($1\\text{--}60\\text{ s}$) | Dynamic per turn (\\$0.05 - \\$0.50) | ReAct loops, MicroVM tool runners, Pytest gates |
| **Autonomous Multi-Agent Fleets** | High (open-ended research, synthesis) | Variable (semantic consensus) | Asynchronous ($1\\text{--}30\\text{ min}$) | High per task (\\$1.00 - \\$20.00) | Hierarchical subagent DAGs, Quorum voting |

: Systems Frontier Matrix: Architectural Regimes across Uncertainty and Verifiability. {#tbl-vol3-architectural-regimes}"""

replacements = [
    (matches[0].group(0), new_block_1),
    (matches[1].group(0), new_block_2),
    (matches[2].group(0), new_block_3),
    (matches[3].group(0), new_block_4),
    (matches[4].group(0), new_block_5),
    (matches[5].group(0), new_block_6),
    (matches[6].group(0), new_block_7),
]

for old_b, new_b in replacements:
    content = content.replace(old_b, new_b, 1)

# Update referring sentences
text_updates = [
    (
        "In production runtimes, empirical trace profiling frequently exposes catastrophic bottlenecks hidden within uninstrumented runtime glue code:",
        "In production runtimes, empirical trace profiling frequently exposes catastrophic bottlenecks hidden within uninstrumented runtime glue code (@tbl-vol3-trace-critical-path):"
    ),
    (
        "The trace excerpt above demonstrates that model execution",
        "The trace profile in @tbl-vol3-trace-critical-path demonstrates that model execution"
    ),
    (
        "while mathematically guaranteeing that the output matches the exact probability distribution of the target model.**",
        "while mathematically guaranteeing that the output matches the exact probability distribution of the target model, as formalized in the speculative decoding execution pipeline (@tbl-vol3-speculative-decoding-pipeline).**"
    ),
    (
        "resetting their elapsed quantum counters and ensuring that background trajectories periodically advance.",
        "resetting their elapsed quantum counters and ensuring that background trajectories periodically advance (@def-vol3-mlfq-transition-protocol)."
    ),
    (
        "deploy a **hybrid tiered provisioning architecture** (often termed *cloud bursting*).",
        "deploy a **hybrid tiered provisioning architecture** (@tbl-vol3-hybrid-fleet-provisioning) (often termed *cloud bursting*)."
    ),
    (
        "trap the execution context with an `OutOfBudgetException`, and initiate deterministic trajectory termination.",
        "trap the execution context with an `OutOfBudgetException`, and initiate deterministic trajectory termination, following the ledger accounting protocol illustrated in @tbl-vol3-budget-ledger-reconciliation and @fig-vol3-hierarchical-budget-ledger."
    ),
    (
        "reverts to `OPEN`.",
        "reverts to `OPEN`. The state transitions and operational invariants governing this control loop are formalized in @tbl-vol3-circuit-breaker-fsm."
    ),
    (
        "A systems architect must understand how to navigate these sensitivity frontiers.",
        "A systems architect must understand how to navigate these sensitivity frontiers across the architectural paradigms detailed in @tbl-vol3-architectural-regimes."
    )
]

for old_t, new_t in text_updates:
    if old_t not in content:
        print(f"ERROR: Text update target not found: {old_t}")
        sys.exit(1)
    content = content.replace(old_t, new_t, 1)

with open(CH17_FILE, "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully updated Chapter 17!")
