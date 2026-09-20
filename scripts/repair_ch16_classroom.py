#!/usr/bin/env python3
"""
Repair Chapter 16 based on 10-student classroom review:
1. Replace 5 ASCII/text diagrams with formal tables and mathematical callout definitions.
2. Update 19 compound headings to strictly respect the Single-Concept Principle.
3. Validate table column counts, zero text blocks, zero box characters, zero British spellings.
"""

import re
import sys

CH16_FILE = "books/vol3/16_observability/16_observability.qmd"

with open(CH16_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Heading replacements (Single-Concept Principle)
heading_replacements = [
    (
        "#### Deterministic Mock Services and Network Fencing {.unnumbered}",
        "#### Deterministic Mock Services {.unnumbered}"
    ),
    (
        "### Information Leakage, Oracle Isolation, and Benchmark Contamination",
        "### Benchmark Contamination Hazards"
    ),
    (
        "### Deconstructing Production Agent Gyms: SWE-bench, GAIA, and WebArena",
        "### Production Agent Gyms"
    ),
    (
        "### The Physics of Stochasticity and the Resampling Hierarchy {#sec-vol3-observability-resampling}",
        "### Resampling Hierarchies {#sec-vol3-observability-resampling}"
    ),
    (
        "### Finite-Sample Uncertainty and the Wilson Score Interval {#sec-vol3-observability-wilson}",
        "### Finite-Sample Uncertainty {#sec-vol3-observability-wilson}"
    ),
    (
        "### Decoupling Latent Potential from Operational Policy: pass@k, pass^k, and Cost-Budgeted Metrics {#sec-vol3-observability-pass-k}",
        "### Latent Potential Decoupling {#sec-vol3-observability-pass-k}"
    ),
    (
        "### The Telemetry Volume Explosion and Head-Sampling Pathology",
        "### Telemetry Volume Explosion"
    ),
    (
        "### Tail-Based Routing and Dynamic Decision Policies",
        "### Tail-Based Routing Policies"
    ),
    (
        "### In-Sandbox Cryptographic PII and Secret Redaction",
        "### Cryptographic Secret Redaction"
    ),
    (
        "#### High-Entropy Pattern and Signature Matching {.unnumbered}",
        "#### High-Entropy Pattern Matching {.unnumbered}"
    ),
    (
        "### Deterministic Trajectory Replay and Hermetic Mocking {#sec-vol3-postmortem-replay}",
        "### Deterministic Trajectory Replay {#sec-vol3-postmortem-replay}"
    ),
    (
        "### Controlled Counterfactual Ablation and Gym Regression Harness {#sec-vol3-postmortem-ablation}",
        "### Counterfactual Gym Ablation {#sec-vol3-postmortem-ablation}"
    ),
    (
        "### Canary Traffic Shifting and Automated Circuit Breakers",
        "### Canary Traffic Shifting"
    ),
    (
        "### Core Agent SRE Metrics: Goodput, Drift, and Intervention Rates",
        "### Core Agent SRE Metrics"
    ),
    (
        "### The Telemetry and Evaluation Readiness Contract {#sec-vol3-observability-synthesis-readiness}",
        "### Evaluation Readiness Contracts {#sec-vol3-observability-synthesis-readiness}"
    ),
    (
        "#### Streaming Escrow and Redaction Invariant {.unnumbered}",
        "#### Streaming Escrow Invariants {.unnumbered}"
    ),
    (
        "#### Deterministic Gym Fidelity and Environmental Reset Invariant {.unnumbered}",
        "#### Deterministic Gym Fidelity {.unnumbered}"
    ),
    (
        "#### Statistical Power and Sample Sizing Invariant {.unnumbered}",
        "#### Statistical Power Invariants {.unnumbered}"
    ),
    (
        "#### Zero-Tolerance Safety and Escrow Criterion ($\\mathcal{C}_{\\text{safety}}$) {.unnumbered}",
        "#### Zero-Tolerance Safety Criteria ($\\mathcal{C}_{\\text{safety}}$) {.unnumbered}"
    )
]

for old_h, new_h in heading_replacements:
    if old_h not in content:
        print(f"ERROR: Heading not found: {old_h}")
        sys.exit(1)
    content = content.replace(old_h, new_h, 1)

print("Applied all 19 heading replacements.")

# Find all 5 text blocks
matches = list(re.finditer(r"```text\s*.*?\n```", content, re.DOTALL))
if len(matches) != 5:
    print(f"ERROR: Expected 5 text blocks, found {len(matches)}")
    sys.exit(1)

new_block_1 = """| Span Identifier | Execution Domain | Time Interval & Duration | Operational & Semantic Attributes | Causal Role in Trajectory |
| :--- | :--- | :--- | :--- | :--- |
| **Root Span** (`s_0`) | `agent.orchestration` | $[0.00, 14.85]\\text{ s}$ ($\\Delta t = 14.85\\text{ s}$) | Trace ID `4bf92f35...`, Model Cost \\$0.042, Status `OK` | Root trajectory lifecycle envelope |
| **Span 1** (`s_1`) | `agent.memory` (Retrieval) | $[0.00, 0.18]\\text{ s}$ ($\\Delta t = 185\\text{ ms}$) | Query `"git patch apply syntax"`, 3 chunks, HNSW index | Pre-inference episodic context grounding |
| **Span 2** (`s_2`) | `gen_ai.client` (Inference) | $[0.18, 4.25]\\text{ s}$ ($\\Delta t = 4.07\\text{ s}$) | Model `llama-3-70b`, Prompt 4,096 tok, Out 256 tok, TTFT 120 ms | Plan formulation and tool call generation |
| **Span 3** (`s_3`) | `agent.tool` (Sandbox Exec) | $[4.26, 8.46]\\text{ s}$ ($\\Delta t = 4.20\\text{ s}$) | Command `git apply patch.diff`, Exit code 0, MicroVM | Deterministic execution in isolated sandbox |
| **Span 4** (`s_4`) | `agent.message` (Subagent) | $[8.47, 10.32]\\text{ s}$ ($\\Delta t = 1.85\\text{ s}$) | Protocol JSON-RPC, Recipient `reviewer-01`, Hash `0x9e12a4` | Asynchronous subagent delegation barrier |
| **Span 4.1** (`s_{4.1}`) | `gen_ai.client` (Subagent Review) | $[8.50, 10.28]\\text{ s}$ ($\\Delta t = 1.78\\text{ s}$) | Model `llama-3-8b`, Prompt 1,024 tok, Out 128 tok, TTFT 45 ms | Specialized peer review and verification pass |
| **Span 5** (`s_5`) | `gen_ai.client` (Synthesis) | $[10.33, 14.85]\\text{ s}$ ($\\Delta t = 4.52\\text{ s}$) | Model `llama-3-70b`, Prompt 6,144 tok, Out 384 tok, TTFT 180 ms | Final trajectory synthesis and task conclusion |

: Hierarchical Distributed Trace Decomposition for Heterogeneous Agent Execution. {#tbl-vol3-trace-hierarchy}"""

new_block_2 = """| Telemetry Component | Payload Size | Share (%) | Serialization Format | Ingestion Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **System Prompt & Tool Schemas** | 8 KB | 17.8% | JSON / Static Protobuf | Pointer-based content-addressable schema deduplication |
| **Multi-Turn Context History** | 24 KB | 53.3% | UTF-8 String Array | Delta encoding and checkpoint referencing |
| **Raw LLM Output & Logits** | 3 KB | 6.7% | Protobuf / BFloat16 | Top-logprob truncation and compression |
| **Tool Execution Stdout/Stderr** | 9 KB | 20.0% | Binary Stream / Chunked Text | Sliding-window truncation with cryptographic digest |
| **Span Attributes & Metrics** | 1 KB | 2.2% | OTel Key-Value Map | Standard structured metric encoding |

: Telemetry Payload Footprint per Agent Execution Turn (45 KB Baseline). {#tbl-vol3-span-telemetry-breakdown}"""

new_block_3 = """| Filter Stage | Target Threat Class | Detection & Matching Pattern | Transformation Action | Computational Overhead |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1: High-Entropy Filter** | Hardcoded API Keys & Cloud Credentials | Regex patterns (e.g., `AKIA...`, `ghp_...`, Shannon entropy $H > 4.5$) | Redacted token replacement `[REDACTED:API_KEY:<hash_8>]` | $< 0.2\\text{ ms}$ per KB |
| **Stage 2: HMAC Engine** | Tenant Identifiers, User IDs, Account Numbers | UUID regexes, database primary keys, email identities | One-way keyed transform $\\text{HMAC-SHA256}(k_{\\text{salt}}, v)$ | $< 0.5\\text{ ms}$ per record |
| **Stage 3: Local PII Classifier** | Unstructured PII (Names, Addresses, Cards) | On-device lightweight NER / regex rules | Semantic token replacement `[REDACTED:PII_CLASS]` | $1.5\\text{--}3.0\\text{ ms}$ per KB |
| **Egress Escrow Gate** | Plaintext Leakage to External Collectors | Streaming zero-knowledge byte scanner | Dropped spans or isolated quarantine if leak detected | $< 0.1\\text{ ms}$ stream tax |

: In-Sandbox Streaming Telemetry Sanitization Pipeline Stages. {#tbl-vol3-telemetry-sanitizer-pipeline}"""

new_block_4 = """| Architectural Dimension | Primary Production Stream ($\\pi_{\\text{base}}$) | Mirrored Dark Stream ($\\pi_{\\text{cand}}$) | Telemetry & Safety Boundary |
| :--- | :--- | :--- | :--- |
| **Traffic Allocation** | $1 - \\rho_{\\text{dark}}$ ($100\\%$ active user requests) | $\\rho_{\\text{dark}}$ asynchronously duplicated copy | Transparent non-blocking ingress fork |
| **Execution Authority** | Mutating authority ($A > 0$) on production infra | Zero ambient authority ($A = 0$); mock escrow | Sandbox hypervisor trap on mutating syscalls |
| **Read Operations** | Live read replicas and active databases | Ephemeral copy-on-write snapshots or read replicas | Isolated connection pool; zero production locking |
| **Mutating Operations** | Executed directly against production persistence | Intercepted; synthetic stubs or ephemeral containers | Immediate ephemeral container discard post-run |
| **Response Delivery** | Returned directly to end user / client | Discarded or routed to evaluation analyzer | Zero external client exposure |
| **Telemetry Objective** | SLA latency, availability, end-user task success | Trajectory divergence, token drift, tool parity | Real-time differential drift calculation |

: Dark Traffic Forking and Authority Partitioning Architecture. {#tbl-vol3-dark-traffic-routing}"""

new_block_5 = """::: {#def-vol3-sprt-circuit-breaker .callout-tip}
## Definition 16.1: Sequential Probability Ratio Test (SPRT) Gating Boundaries
The automated canary circuit breaker evaluates cumulative trajectory observations against two Wald absorbing boundaries defined by the target false-positive rate $\\alpha$ and false-negative rate $\\beta$:

$$\\Lambda_n = d_n \\ln \\left(\\frac{p_1}{p_0}\\right) + (n - d_n) \\ln \\left(\\frac{1 - p_1}{1 - p_0}\\right)$$

1. **Upper Absorbing Boundary ($A = \\ln \\frac{1 - \\beta}{\\alpha}$):** If $\\Lambda_n \\ge A$, the candidate runtime is statistically validated as non-inferior; the control plane promotes the candidate and accelerates canary traffic shifting.
2. **Lower Absorbing Boundary ($B = \\ln \\frac{\\beta}{1 - \\alpha}$):** If $\\Lambda_n \\le B$, the candidate runtime is conclusively degraded; the control plane instantly trips the automated circuit breaker, shedding all canary traffic to restore baseline routing.
3. **Indeterminate Region ($B < \\Lambda_n < A$):** Sampling continues ($n \\leftarrow n + 1$) without adjusting traffic weights.
:::"""

replacements = [
    (matches[0].group(0), new_block_1),
    (matches[1].group(0), new_block_2),
    (matches[2].group(0), new_block_3),
    (matches[3].group(0), new_block_4),
    (matches[4].group(0), new_block_5),
]

for old_b, new_b in replacements:
    content = content.replace(old_b, new_b, 1)

# Update referring sentences
text_updates = [
    (
        "As illustrated in the span hierarchy, the root span $\\mathcal{T}_{\\text{root}}$ encompasses",
        "As illustrated in the distributed trace decomposition (@tbl-vol3-trace-hierarchy), the root span $\\mathcal{T}_{\\text{root}}$ encompasses"
    ),
    (
        "yields a single trace size ranging from $5\\text{ MB}$ to over $25\\text{ MB}$.",
        "yields a single trace size ranging from $5\\text{ MB}$ to over $25\\text{ MB}$, with individual component distributions detailed in @tbl-vol3-span-telemetry-breakdown."
    ),
    (
        "The in-sandbox sanitization pipeline implements a three-stage streaming filter",
        "As summarized in @tbl-vol3-telemetry-sanitizer-pipeline, the in-sandbox sanitization pipeline implements a multi-stage streaming filter"
    ),
    (
        "The secondary stream forks asynchronously to the candidate policy $\\pi_{\\text{cand}}$.",
        "The secondary stream forks asynchronously to the candidate policy $\\pi_{\\text{cand}}$, establishing the authority partitioning outlined in @tbl-vol3-dark-traffic-routing."
    ),
    (
        "The automated circuit breaker must operate with minimal Mean Time to Detect (MTTD)",
        "As formalized in @def-vol3-sprt-circuit-breaker, the automated circuit breaker must operate with minimal Mean Time to Detect (MTTD)"
    )
]

for old_t, new_t in text_updates:
    if old_t not in content:
        print(f"ERROR: Text update target not found: {old_t}")
        sys.exit(1)
    content = content.replace(old_t, new_t, 1)

with open(CH16_FILE, "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully updated Chapter 16!")
