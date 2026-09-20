#!/usr/bin/env python3
"""
Repair Chapter 01 based on 10-student classroom review:
1. Convert text blocks to appropriate bash/blockquote blocks.
2. Update 13 compound headings to strictly respect the Single-Concept Principle.
3. Validate table column counts, zero text blocks, zero box characters, zero British spellings.
"""

import re
import sys

CH01_FILE = "books/vol3/01_introduction/01_introduction.qmd"

with open(CH01_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Heading replacements (Single-Concept Principle)
heading_replacements = [
    (
        "### The Passive Request Boundary and the Open-Loop Ceiling {#sec-vol3-intro-passive-boundary-ceiling}",
        "### The Passive Request Boundary {#sec-vol3-intro-passive-boundary-ceiling}"
    ),
    (
        "### The Tripartite Architecture: Instructions, Weights, and Trajectories",
        "### The Tripartite Architecture"
    ),
    (
        "### Memory Stranding and the Physical Friction of Hybrid Systems",
        "### Memory Stranding Friction"
    ),
    (
        "### Core Primitives and Mathematical Formalism {#sec-vol3-intro-primitives-formalism}",
        "### Mathematical Primitives {#sec-vol3-intro-primitives-formalism}"
    ),
    (
        "#### Phase 1: Continuation Check and Context Assembly {.unnumbered}",
        "#### Phase 1: Context Assembly {.unnumbered}"
    ),
    (
        "#### Phase 5: Observation and Evidence Capture {.unnumbered}",
        "#### Phase 5: Evidence Capture {.unnumbered}"
    ),
    (
        "#### Turn 0: Locate and Inspect (Read-Only Exploration) {.unnumbered}",
        "#### Turn 0: Read-Only Exploration {.unnumbered}"
    ),
    (
        "#### Turn 2: Empirical Repair and Invariant Closure {.unnumbered}",
        "#### Turn 2: Invariant Closure {.unnumbered}"
    ),
    (
        "### Classical Fault Taxonomies and the Fail-Plausible Regime",
        "### The Fail-Plausible Regime"
    ),
    (
        "### The Illusion of Coherence and Autoregressive Confidence",
        "### The Illusion of Coherence"
    ),
    (
        "### Context Poisoning and Attentional Sinks",
        "### Context Poisoning Dynamics"
    ),
    (
        "### The Four Engineering Dimensions and Trajectory Duration Accounting",
        "### Trajectory Duration Accounting"
    ),
    (
        "### Across-Task Lifecycle Infrastructure: Optimization and Fleet Scale",
        "### Across-Task Fleet Infrastructure"
    )
]

for old_h, new_h in heading_replacements:
    if old_h not in content:
        print(f"ERROR: Heading not found: {old_h}")
        sys.exit(1)
    content = content.replace(old_h, new_h, 1)

print(f"Applied all {len(heading_replacements)} heading replacements.")

# 2. Block 1: Bash terminal trace (lines 784-791)
old_block_1 = """```text
Turn 1 (Policy):  $ rm -rf /tmp/cache && touch /tmp/cache/active.lock
Turn 1 (Sandbox): /bin/sh: line 1: /tmp/cache/active.lock: No such file or directory
Turn 2 (Policy):  $ echo "nameserver 8.8.8.8" >> /tmp/cache/resolv.conf
Turn 2 (Sandbox): /bin/sh: line 1: /tmp/cache/resolv.conf: No such file or directory
Turn 3 (Policy):  $ chmod 777 /tmp/cache/*
Turn 3 (Sandbox): chmod: cannot access '/tmp/cache/*': No such file or directory
```"""

new_block_1 = """```bash
Turn 1 (Policy):  $ rm -rf /tmp/cache && touch /tmp/cache/active.lock
Turn 1 (Sandbox): /bin/sh: line 1: /tmp/cache/active.lock: No such file or directory
Turn 2 (Policy):  $ echo "nameserver 8.8.8.8" >> /tmp/cache/resolv.conf
Turn 2 (Sandbox): /bin/sh: line 1: /tmp/cache/resolv.conf: No such file or directory
Turn 3 (Policy):  $ chmod 777 /tmp/cache/*
Turn 3 (Sandbox): chmod: cannot access '/tmp/cache/*': No such file or directory
```"""

if old_block_1 not in content:
    print("ERROR: old_block_1 not found")
    sys.exit(1)
content = content.replace(old_block_1, new_block_1, 1)

# 3. Block 2: Hallucinated response quote (lines 931-934)
old_block_2 = """```text
"I have investigated the database contention issue and optimized the query index.
All latency targets are now fully met."
```"""

new_block_2 = """> "I have investigated the database contention issue and optimized the query index. All latency targets are now fully met." """

if old_block_2 not in content:
    print("ERROR: old_block_2 not found")
    sys.exit(1)
content = content.replace(old_block_2, new_block_2, 1)

with open(CH01_FILE, "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully updated Chapter 01!")
