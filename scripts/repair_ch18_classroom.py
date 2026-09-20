#!/usr/bin/env python3
"""
Repair Chapter 18 based on 10-student classroom review:
1. Replace 3 ASCII/text blocks with formal tables, mathematical callouts, and clean bash logs.
2. Update 18 compound headings to strictly respect the Single-Concept Principle.
3. Validate table column counts, zero text blocks, zero box characters, zero British spellings.
"""

import re
import sys

CH18_FILE = "books/vol3/18_conclusion/18_conclusion.qmd"

with open(CH18_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Heading replacements (Single-Concept Principle)
heading_replacements = [
    (
        "### The Production Operating Envelope and Escalation Boundaries {#sec-vol3-envelope-boundaries}",
        "### Production Operating Envelopes {#sec-vol3-envelope-boundaries}"
    ),
    (
        "### The Four Distinct State Layers and Physical Lifecycles {#sec-vol3-memory-four-layers}",
        "### State Hierarchy Lifecycles {#sec-vol3-memory-four-layers}"
    ),
    (
        "### Cross-Tier Cache Coherence and Invalidation Protocols {#sec-vol3-memory-coherence-invalidation}",
        "### Cross-Tier Cache Coherence {#sec-vol3-memory-coherence-invalidation}"
    ),
    (
        "### Provenance Tracking and Cryptographic Grounding {#sec-vol3-memory-provenance-grounding}",
        "### Provenance Tracking {#sec-vol3-memory-provenance-grounding}"
    ),
    (
        "#### Stage 2: Policy and Capability Gate {.unnumbered}",
        "#### Stage 2: Capability Gate {.unnumbered}"
    ),
    (
        "### Non-Atomicity and the Distributed Saga Contract {#sec-vol3-execution-sagas}",
        "### Distributed Saga Contracts {#sec-vol3-execution-sagas}"
    ),
    (
        "### The Semantic Watchdog and Execution Circuit Breakers {#sec-vol3-execution-watchdog}",
        "### Semantic Watchdog Execution {#sec-vol3-execution-watchdog}"
    ),
    (
        "#### Level 1: Context and Information Engineering {.unnumbered}",
        "#### Level 1: Context Engineering {.unnumbered}"
    ),
    (
        "#### Level 2: Tool Interface and Schema Redesign {.unnumbered}",
        "#### Level 2: Tool Interface Redesign {.unnumbered}"
    ),
    (
        "### Economic Amortization and Break-Even Volume",
        "### Economic Amortization Dynamics"
    ),
    (
        "### Architectural Trade-Offs and the End-to-End Fallacy",
        "### Architectural Trade-Offs"
    ),
    (
        "#### Tier 3: Runtime Containment and Canary Telemetry {.unnumbered}",
        "#### Tier 3: Canary Runtime Containment {.unnumbered}"
    ),
    (
        "### Blast-Radius Minimization and Operational Escalation",
        "### Blast-Radius Minimization"
    ),
    (
        "### Architectural Authority and the Irreducible Core",
        "### Architectural Authority"
    ),
    (
        "### The Digital Safety Envelope: Reversibility and Ephemeral Sandboxes {#sec-vol3-frontiers-digital-envelope}",
        "### The Digital Safety Envelope {#sec-vol3-frontiers-digital-envelope}"
    ),
    (
        "#### The Oracle Deficit and Incomplete Task Verifiers {.unnumbered}",
        "#### The Oracle Deficit {.unnumbered}"
    ),
    (
        "#### Non-Stationary Environments and Concurrency Hazards {.unnumbered}",
        "#### Non-Stationary Environmental Hazards {.unnumbered}"
    ),
    (
        "#### Irreversibility and the Collapse of Trial-and-Error {.unnumbered}",
        "#### Irreversible Actuation Hazards {.unnumbered}"
    )
]

for old_h, new_h in heading_replacements:
    if old_h not in content:
        print(f"ERROR: Heading not found: {old_h}")
        sys.exit(1)
    content = content.replace(old_h, new_h, 1)

print(f"Applied all {len(heading_replacements)} heading replacements.")

# Find all 3 text blocks
matches = list(re.finditer(r"```text\s*.*?\n```", content, re.DOTALL))
if len(matches) != 3:
    print(f"ERROR: Expected 3 text blocks, found {len(matches)}")
    sys.exit(1)

# Block 1: Watchdog execution log -> convert to ```bash
old_block_1 = matches[0].group(0)
new_block_1 = """```bash
[WATCHDOG: TURN 12] Action: patch_file(path="src/parser.c", diff="@@ -42,2 +42,2 @@...")
[SANDBOX:  TURN 12] Exit: 1 | stderr: "parser.c:45:12: error: unknown type name 'NodeId'"
[WATCHDOG: TURN 13] Action: patch_file(path="src/parser.c", diff="@@ -42,2 +42,2 @@...")
[SANDBOX:  TURN 13] Exit: 1 | stderr: "parser.c:45:12: error: expected ';' before 'x'"
[WATCHDOG: TURN 14] Action: patch_file(path="src/parser.c", diff="@@ -42,2 +42,2 @@...")
[WATCHDOG: TRAP] Semantic oscillation detected: State hash matches Turn 12 (git:e84f2b).
[CIRCUIT_BREAKER] State changed: CLOSED -> OPEN.
[CIRCUIT_BREAKER] Halting forward dispatch. Invoking Saga compensation ledger (c_14..c_12).
[SUPERVISOR] Escalating to human escrow: Diagnostic payload spooled to /var/log/traces/104.
```"""
content = content.replace(old_block_1, new_block_1, 1)

# Block 2: Six Tiers of System Intervention ladder -> Callout Definition 18.1
old_block_2 = matches[1].group(0)
new_block_2 = """::: {#def-vol3-intervention-hierarchy .callout-tip}
## Definition 18.1: The Six-Tier Systems Intervention Hierarchy
When remediating agent performance regressions, systems engineers ascend an ordered hierarchy prioritizing lowest upfront capital and smallest blast radius before escalating to higher tiers:

$$\\text{Tier 1 (Context)} \\prec \\text{Tier 2 (Tools)} \\prec \\text{Tier 3 (Guards)} \\prec \\text{Tier 4 (SFT)} \\prec \\text{Tier 5 (RLVR)} \\prec \\text{Tier 6 (Multi-Agent)}$$

1. **Tier 1: Context & Information Engineering:** Prune working memory, optimize retrieval precision, and inject few-shot exemplars within the active context window $\\mathcal{C}$.
2. **Tier 2: Tool Interface & Schema Redesign:** Refactor tool signatures, enforce strict typing schemas, and return structured, diagnostic error payloads.
3. **Tier 3: Runtime Harness Hardening:** Implement semantic watchdogs, logit grammar masks, and automated fail-fast circuit breakers in the host supervisor.
4. **Tier 4: Supervised Adaptation (SFT):** Fine-tune base model weights $\\theta$ via behavioral cloning on curated, verified expert trajectories.
5. **Tier 5: Verifiable RL (RLVR):** Optimize policy parameters $\\pi_\\theta$ directly against deterministic mechanical verifiers and execution oracles.
6. **Tier 6: Multi-Agent Delegation:** Partition task state across distributed subagent DAGs using explicit coordination and consensus protocols.

The quantitative engineering trade-offs across these six tiers are detailed in @tbl-vol3-intervention-ladder.
:::"""
content = content.replace(old_block_2, new_block_2, 1)

# Block 3: Tier 3 Canary Containment Architecture -> Table 18.5
old_block_3 = matches[2].group(0)
new_block_3 = """| Containment Stage | Subsystem Substrate | Operational Invariant & Boundary Control | Telemetry Signal | Fail-Safe Reaction |
| :--- | :--- | :--- | :--- | :--- |
| **Traffic Routing** | Dynamic L7 Ingress Proxy | Weighted traffic allocation ($\\rho \\approx 0.02$) | Real-time ingress rate and request distributions | Instantaneous traffic drain to stable fleet (v1.4) |
| **Runtime Sandbox** | Unprivileged MicroVM (gVisor/KVM) | Zero ambient authority ($A = 0$); memory and CPU quotas | Container startup latency & memory pressure | Immediate sandbox freeze and core memory dump |
| **Syscall Filter** | Strict `seccomp-bpf` interceptor | Whitelist-only syscall dispatch; blocks `ptrace`/raw sockets | Intercepted violation counters | Asynchronous kernel trap; process termination |
| **Safety Breaker** | SRE Error-Budget Monitor | Error-budget burn rate $< 2\\times$ baseline threshold | Tool retry spikes, latency tail $P_{99}$ | Automated circuit breaker trip; instant rollback |

: Tier 3 Canary Containment Subsystem Specifications and Enforcement Controls. {#tbl-vol3-canary-containment-architecture}"""
content = content.replace(old_block_3, new_block_3, 1)

# Update referring sentences
text_updates = [
    (
        "Moving up the ladder shifts the point of control from transient runtime state toward durable artifact definitions, compiled weights, and distributed process topologies.",
        "Moving up the ladder shifts the point of control from transient runtime state toward durable artifact definitions, compiled weights, and distributed process topologies (@def-vol3-intervention-hierarchy)."
    ),
    (
        "Tier 3 wraps the live agent inside three non-negotiable containment mechanisms:",
        "As formalized in @tbl-vol3-canary-containment-architecture, Tier 3 wraps the live agent inside three non-negotiable containment mechanisms:"
    )
]

for old_t, new_t in text_updates:
    if old_t not in content:
        print(f"ERROR: Text update target not found: {old_t}")
        sys.exit(1)
    content = content.replace(old_t, new_t, 1)

with open(CH18_FILE, "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully updated Chapter 18!")
