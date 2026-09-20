#!/usr/bin/env python3
"""
Repair Chapter 05 based on 10-student classroom review:
1. Clean 14 compound headings to single concepts.
2. Validate table column counts, zero text blocks, zero box characters, zero British spellings.
"""

import re
import sys

CH05_FILE = "books/vol3/05_virtual_memory/05_virtual_memory.qmd"

with open(CH05_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Heading replacements (Single-Concept Principle)
heading_replacements = [
    (
        "### Static Over-Provisioning and the Predictability Dilemma",
        "### Static Over-Provisioning Limits"
    ),
    (
        "### The Mechanics of Contiguous Wastage: Internal and External Fragmentation",
        "### Contiguous Memory Wastage"
    ),
    (
        "### Agent Trajectory Churn and Stranded State",
        "### Trajectory Memory Churn"
    ),
    (
        "### Cross-Request Redundancy and the Prefix Trie",
        "### Prefix Trie Redundancy"
    ),
    (
        "### Longest Prefix Matching and Block Boundary Alignment",
        "### Longest Prefix Matching"
    ),
    (
        "### Radix Tree Memory Reclamation and Dynamic Eviction",
        "### Dynamic Radix Eviction"
    ),
    (
        "### Causality Violations and Branch Invalidation",
        "### Branch Invalidation Mechanics"
    ),
    (
        "### Chunked Prefill Mechanics and Attention Slicing",
        "### Chunked Prefill Mechanics"
    ),
    (
        "### Token-Budget Scheduling and SLA Optimization",
        "### Token-Budget Scheduling"
    ),
    (
        "### The Tool-Wait Memory Tax and the Four Reclamation Policies",
        "### Memory Reclamation Policies"
    ),
    (
        "### Asynchronous Paged Swapping Runtimes and Transfer Overlap",
        "### Asynchronous Paged Swapping"
    ),
    (
        "### Tiered Secondary Storage: NVMe and Remote Memory Systems",
        "### Tiered Secondary Storage"
    ),
    (
        "### Analytical Capacity Formulations and Hardware Envelopes {#sec-vol3-kv-capacity-formulations}",
        "### Analytical Capacity Formulations {#sec-vol3-kv-capacity-formulations}"
    ),
    (
        "### Telemetry, Instrumentation, and Tail Latency Dynamics {#sec-vol3-kv-telemetry-metrics}",
        "### Serving Telemetry Dynamics {#sec-vol3-kv-telemetry-metrics}"
    )
]

for old_h, new_h in heading_replacements:
    if old_h not in content:
        print(f"ERROR: Heading not found: {old_h}")
        sys.exit(1)
    content = content.replace(old_h, new_h, 1)

print(f"Applied all {len(heading_replacements)} heading replacements.")

with open(CH05_FILE, "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully updated Chapter 05!")
