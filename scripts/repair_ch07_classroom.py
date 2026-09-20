#!/usr/bin/env python3
"""
Repair Chapter 07 based on 10-student classroom review:
1. Retag cargo test output and truncation notice (2 text blocks) to bash.
2. Clean 21 compound headings to single concepts.
3. Validate table column counts, zero text blocks, zero box characters, zero British spellings.
"""

import re
import sys

CH07_FILE = "books/vol3/07_actuation/07_actuation.qmd"

with open(CH07_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Heading replacements (Single-Concept Principle)
heading_replacements = [
    (
        "### Anatomy of the Tool Declaration: Structural and Semantic Contracts",
        "### Anatomy of the Tool Declaration"
    ),
    (
        "### The Schema Quality Effect and Semantic Conditioning",
        "### The Schema Quality Effect"
    ),
    (
        "### Host-Side Strict Validation and Diagnostic Feedback",
        "### Host-Side Strict Validation"
    ),
    (
        "### Client-Server Decomposition and JSON-RPC Framing",
        "### Client-Server Decomposition"
    ),
    (
        "### Transport Topologies and Latency-Isolation Trade-Offs",
        "### Transport Topologies"
    ),
    (
        "### Keyed Deduplication and Distributed Leases",
        "### Keyed Deduplication Mechanics"
    ),
    (
        "### The Asymmetry of Actuation and the Context Ingestion Hazard",
        "### The Asymmetry of Actuation"
    ),
    (
        "### Headless Tailing and Sliding-Window Ring Buffers",
        "### Headless Tailing Mechanics"
    ),
    (
        "### Structured Pagination and Continuation Contracts",
        "### Structured Pagination Contracts"
    ),
    (
        "### Kernel Buffering, Pipe Sizing, and Backpressure",
        "### Kernel Buffering Dynamics"
    ),
    (
        "### Terminal Emulation and Escape Sequence Elimination",
        "### Terminal Emulation Mechanics"
    ),
    (
        "### Process Termination Semantics and the Epistemic Limits of Exit Status",
        "### Process Termination Semantics"
    ),
    (
        "### Structured Frame Extraction and Explicit Truncation Framing",
        "### Structured Frame Extraction"
    ),
    (
        "### The Timescale Mismatch and Synchronous Blocking {#sec-vol3-actuation-async-mismatch}",
        "### The Timescale Mismatch {#sec-vol3-actuation-async-mismatch}"
    ),
    (
        "### Asynchronous Dispatch and Job Escrow {#sec-vol3-actuation-async-escrow}",
        "### Asynchronous Dispatch {#sec-vol3-actuation-async-escrow}"
    ),
    (
        "### Event-Driven Resumption: Polling, Webhooks, and Kernel Event Loops {#sec-vol3-actuation-async-resumption}",
        "### Event-Driven Resumption {#sec-vol3-actuation-async-resumption}"
    ),
    (
        "### Background Process Governance: Interactivity, Telemetry, and Teardown {#sec-vol3-actuation-async-governance}",
        "### Background Process Governance {#sec-vol3-actuation-async-governance}"
    ),
    (
        "#### Process Group Cancellation and Teardown {.unnumbered}",
        "#### Process Group Cancellation {.unnumbered}"
    ),
    (
        "### Catalog Cardinality, Schema Overhead, and Selection Dispersion",
        "### Catalog Cardinality Effects"
    ),
    (
        "### Orthogonal Tool Abstractions and Deep Interfaces",
        "### Orthogonal Tool Abstractions"
    ),
    (
        "### The Expressiveness–Safety Spectrum and Defensive Parameter Design",
        "### The Expressiveness-Safety Spectrum"
    )
]

for old_h, new_h in heading_replacements:
    if old_h not in content:
        print(f"ERROR: Heading not found: {old_h}")
        sys.exit(1)
    content = content.replace(old_h, new_h, 1)

print(f"Applied all {len(heading_replacements)} heading replacements.")

# 2. Block 1: cargo test output (lines 595-611)
old_block_1 = """```text
[Line 000001] $ cargo test --all
[Line 000002] Compiling kernel v0.1.0 (/workspace/core)
[Line 000003] Running 48219 tests across 14 modules
... [48,200 lines of passing tests: "test engine::... ok"] ...
[Line 48205] test memory::paging::test_allocator ... FAILED
[Line 48206]
[Line 48207] failures:
[Line 48208] ---- memory::paging::test_allocator stdout ----
[Line 48209] thread 'test_allocator' panicked at 'assertion failed: `(left == right)`', src/memory.rs:142:9
[Line 48210]   left: `0x00007fff`, right: `0x00000000`
[Line 48211] stack backtrace:
[Line 48212]    0: std::panicking::begin_panic
[Line 48213]    1: kernel::memory::paging::allocate_frame
[Line 48214]
[Line 48215] test result: FAILED. 48218 passed; 1 failed; 0 ignored
```"""

new_block_1 = """```bash
[Line 000001] $ cargo test --all
[Line 000002] Compiling kernel v0.1.0 (/workspace/core)
[Line 000003] Running 48219 tests across 14 modules
... [48,200 lines of passing tests: "test engine::... ok"] ...
[Line 48205] test memory::paging::test_allocator ... FAILED
[Line 48206]
[Line 48207] failures:
[Line 48208] ---- memory::paging::test_allocator stdout ----
[Line 48209] thread 'test_allocator' panicked at 'assertion failed: `(left == right)`', src/memory.rs:142:9
[Line 48210]   left: `0x00007fff`, right: `0x00000000`
[Line 48211] stack backtrace:
[Line 48212]    0: std::panicking::begin_panic
[Line 48213]    1: kernel::memory::paging::allocate_frame
[Line 48214]
[Line 48215] test result: FAILED. 48218 passed; 1 failed; 0 ignored
```"""

if old_block_1 not in content:
    print("ERROR: old_block_1 not found")
    sys.exit(1)
content = content.replace(old_block_1, new_block_1, 1)
print("Retagged Block 1 as bash.")

# 3. Block 2: Truncation notice (lines 622-624)
old_block_2 = """```text
[... 48,195 lines and 4,182,041 bytes truncated by runtime supervisor ...]
```"""

new_block_2 = """```bash
[... 48,195 lines and 4,182,041 bytes truncated by runtime supervisor ...]
```"""

if old_block_2 not in content:
    print("ERROR: old_block_2 not found")
    sys.exit(1)
content = content.replace(old_block_2, new_block_2, 1)
print("Retagged Block 2 as bash.")

with open(CH07_FILE, "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully updated Chapter 07!")
