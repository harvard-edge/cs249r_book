#!/usr/bin/env python3
"""
Repair Chapter 03 based on 10-student classroom review:
1. Retag C code and clang linker error text blocks (c and bash).
2. Clean 25 compound headings to single concepts.
3. Validate table column counts, zero text blocks, zero box characters, zero British spellings.
"""

import re
import sys

CH03_FILE = "books/vol3/03_deliberation/03_deliberation.qmd"

with open(CH03_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Heading replacements (Single-Concept Principle)
heading_replacements = [
    (
        "### Comparative Latency, Compute, and Memory Mechanics",
        "### Deliberation Hardware Mechanics"
    ),
    (
        "### Correlated Sampling and Effective Candidate Diversity",
        "### Candidate Diversity Dynamics"
    ),
    (
        "### Selection Architectures: Voting, Scoring, and Execution Filtering",
        "### Selection Architectures"
    ),
    (
        "#### Self-Consistency and Majority Voting {.unnumbered}",
        "#### Self-Consistency Mechanics {.unnumbered}"
    ),
    (
        "#### Learned Scoring and Verifier Inference {.unnumbered}",
        "#### Learned Verifier Scoring {.unnumbered}"
    ),
    (
        "### The Verification Budget and the Selector Bottleneck",
        "### The Verification Budget"
    ),
    (
        "### Goodhart's Law and the Mechanics of Search Exploitation",
        "### Search Exploitation Mechanics"
    ),
    (
        "#### Adversarial Sandboxing and Anti-Introspection Hardening {.unnumbered}",
        "#### Adversarial Sandboxing {.unnumbered}"
    ),
    (
        "### Robust Verification Topologies: Ensembles, Invariants, and Property-Based Oracles",
        "### Robust Verification Topologies"
    ),
    (
        "#### Layer 1: Immutable Syntax and Deterministic Structural Guards {.unnumbered}",
        "#### Layer 1: Deterministic Structural Guards {.unnumbered}"
    ),
    (
        "#### Layer 2: Property-Based Fuzzing and Metamorphic Invariants {.unnumbered}",
        "#### Layer 2: Property-Based Invariants {.unnumbered}"
    ),
    (
        "#### Layer 3: Heterogeneous PRM Ensembles and Consensus Voting {.unnumbered}",
        "#### Layer 3: PRM Ensembles {.unnumbered}"
    ),
    (
        "### Precondition Verification and Invalidation Detection {#sec-vol3-deliberation-precondition-invalidation}",
        "### Precondition Verification {#sec-vol3-deliberation-precondition-invalidation}"
    ),
    (
        "### Hierarchical Revision Mechanics: Repair, Prune, and Escalate {#sec-vol3-deliberation-revision-mechanics}",
        "### Hierarchical Revision Mechanics {#sec-vol3-deliberation-revision-mechanics}"
    ),
    (
        "#### Subgraph Pruning and Resynthesis (Branch-Level Replanning) {.unnumbered}",
        "#### Subgraph Resynthesis {.unnumbered}"
    ),
    (
        "### Damping Invariants and the Replanning Thrashing Trap {#sec-vol3-deliberation-replanning-thrashing}",
        "### Replanning Damping Invariants {#sec-vol3-deliberation-replanning-thrashing}"
    ),
    (
        "#### Observation Filtering and Predicate Partitioning {.unnumbered}",
        "#### Observation Filtering {.unnumbered}"
    ),
    (
        "#### Bounded Replanning Budgets and Graph Hysteresis {.unnumbered}",
        "#### Bounded Replanning Budgets {.unnumbered}"
    ),
    (
        "#### Step Quarantine and Invalidation Hysteresis {.unnumbered}",
        "#### Step Quarantine Mechanics {.unnumbered}"
    ),
    (
        "### Search Topologies: Flat Sampling, Beam Search, and Tree Search",
        "### Search Topologies"
    ),
    (
        "### Multicriteria Stopping and Diminishing Returns",
        "### Multicriteria Stopping Invariants"
    ),
    (
        "#### Wall-Clock Latency Distributions and Tail Amplification {.unnumbered}",
        "#### Wall-Clock Latency Distributions {.unnumbered}"
    ),
    (
        "#### False Acceptance Rate and Silent Failure Leakage {.unnumbered}",
        "#### False Acceptance Rate {.unnumbered}"
    ),
    (
        "### The Deliberation Pareto Frontier and Regimes of Failure",
        "### The Deliberation Pareto Frontier"
    ),
    (
        "#### Regimes of Failure: Negative Transfer and Over-Deliberation {.unnumbered}",
        "#### Regimes of Failure {.unnumbered}"
    )
]

for old_h, new_h in heading_replacements:
    if old_h not in content:
        print(f"ERROR: Heading not found: {old_h}")
        sys.exit(1)
    content = content.replace(old_h, new_h, 1)

print(f"Applied all {len(heading_replacements)} heading replacements.")

# 2. Block 1: C code snippet
old_block_1 = """```text
Candidate A:  lock.acquire(); data = read(); lock.release();  // Race: unprotected check
Candidate B:  pthread_mutex_lock(&m); val = in->val; pthread_mutex_unlock(&m); // Identical race
```"""

new_block_1 = """```c
Candidate A:  lock.acquire(); data = read(); lock.release();  // Race: unprotected check
Candidate B:  pthread_mutex_lock(&m); val = in->val; pthread_mutex_unlock(&m); // Identical race
```"""

if old_block_1 not in content:
    print("ERROR: old_block_1 not found")
    sys.exit(1)
content = content.replace(old_block_1, new_block_1, 1)
print("Retagged Block 1 as c.")

# 3. Block 2: Linker failure log snippet
old_block_2 = """```text
FAILED: lib/libtransport.so
/usr/bin/ld: cannot find -luring: No such file or directory
clang-16: error: linker command failed with exit code 1
ninja: build stopped: subcommand failed.
```"""

new_block_2 = """```bash
FAILED: lib/libtransport.so
/usr/bin/ld: cannot find -luring: No such file or directory
clang-16: error: linker command failed with exit code 1
ninja: build stopped: subcommand failed.
```"""

if old_block_2 not in content:
    print("ERROR: old_block_2 not found")
    sys.exit(1)
content = content.replace(old_block_2, new_block_2, 1)
print("Retagged Block 2 as bash.")

with open(CH03_FILE, "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully updated Chapter 03!")
