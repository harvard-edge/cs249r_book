#!/usr/bin/env python3
"""
scripts/embed_all_72_micro_svgs.py

Embeds all 72 standardized micro-SVGs into Volume III chapters (01 through 18)
following the canonical Volume 1 / Volume 2 / Volume 4 margin figure standard.
"""

from pathlib import Path
import re

BASE = Path("/Users/VJ/GitHub/MLSysBook-vol3")
BOOKS = BASE / "books" / "vol3"

CHAPTER_FILES = {
    1: ("01_introduction", "01_introduction.qmd"),
    2: ("02_processor", "02_processor.qmd"),
    3: ("03_deliberation", "03_deliberation.qmd"),
    4: ("04_working_sets", "04_working_sets.qmd"),
    5: ("05_virtual_memory", "05_virtual_memory.qmd"),
    6: ("06_episodic_memory", "06_episodic_memory.qmd"),
    7: ("07_actuation", "07_actuation.qmd"),
    8: ("08_virtualization", "08_virtualization.qmd"),
    9: ("09_checkpointing", "09_checkpointing.qmd"),
    10: ("10_interrupts", "10_interrupts.qmd"),
    11: ("11_scheduling", "11_scheduling.qmd"),
    12: ("12_data_flywheel", "12_data_flywheel.qmd"),
    13: ("13_sft", "13_sft.qmd"),
    14: ("14_rlvr", "14_rlvr.qmd"),
    15: ("15_multi_agent", "15_multi_agent.qmd"),
    16: ("16_observability", "16_observability.qmd"),
    17: ("17_tokenomics", "17_tokenomics.qmd"),
    18: ("18_conclusion", "18_conclusion.qmd"),
}

# The 72 canonical micro-SVG specifications
ALL_FIGURES = [
    # -------------------------------------------------------------------------
    # Chapter 01: Foundations of Agentic Systems
    # -------------------------------------------------------------------------
    {
        "chapter": 1,
        "fig_idx": 1,
        "file": "vol3_introduction_margin_001.svg",
        "alt": "Vertical logarithmic scale ladder spanning thirteen orders of magnitude from hardware instructions at 10 to the minus 9 seconds to multi-turn trajectories at 10 to the 4 seconds.",
        "caption": "Agentic systems stretch execution timescales across thirteen orders of magnitude, from nanosecond tensor ops to kilosecond trajectories.",
        "anchor": "### Temporal Stretching: From Nanosecond Opcodes to Kilosecond Trajectories",
    },
    {
        "chapter": 1,
        "fig_idx": 2,
        "file": "vol3_introduction_margin_002.svg",
        "alt": "Stacked horizontal bar comparing trajectory goodput at 21.2 percent in green against computational badput at 78.8 percent in crimson across 4,125 attempted execution steps.",
        "caption": "High GPU kernel utilization masks catastrophic macro-inefficiency when 79% of cluster FLOPs are consumed by failing trajectory badput.",
        "anchor": "### Micro-Efficiency versus Macro-Efficiency",
    },
    {
        "chapter": 1,
        "fig_idx": 3,
        "file": "vol3_introduction_margin_003.svg",
        "alt": "Budget envelope comparison showing a single agent consuming 120 gigabytes and four concurrent agents consuming 480 gigabytes of high-bandwidth memory while waiting for external tool execution.",
        "caption": "Synchronous tool waits pin high-bandwidth accelerator memory, stranding 480 GB of KV cache capacity while GPU tensor cores idle.",
        "anchor": "### Memory Stranding and the Physical Friction of Hybrid Systems",
    },
    {
        "chapter": 1,
        "fig_idx": 4,
        "file": "vol3_introduction_margin_004.svg",
        "alt": "Budget comparison bar chart contrasting 450 dollars in cumulative API tokens under unconstrained advisory prompt loops with zero dollars under mechanical runtime gatekeeping.",
        "caption": "Advisory prompt instructions leak 450 dollars in circular repair loops, whereas deterministic runtime traps enforce invariants at zero token cost.",
        "anchor": "## The Invariant Closure Principle {#sec-vol3-intro-invariant-closure}",
    },

    # -------------------------------------------------------------------------
    # Chapter 02: The Foundation Model Engine
    # -------------------------------------------------------------------------
    {
        "chapter": 2,
        "fig_idx": 1,
        "file": "vol3_processor_margin_001.svg",
        "alt": "Ranked vocabulary probability distribution: the top two tokens capture 95.9% of cumulative mass under top-p nucleus truncation (p=0.90, in emerald), while rigid top-k (k=4) forces inclusion of two low-probability tail tokens (in crimson).",
        "caption": "Top-p dynamically contracts to two high-confidence tokens while rigid top-k forces low-probability tail inclusion.",
        "anchor": "### Sampling on the Vocabulary Simplex",
    },
    {
        "chapter": 2,
        "fig_idx": 2,
        "file": "vol3_processor_margin_002.svg",
        "alt": "Hardware roofline model on an NVIDIA H100 GPU: a blue memory-bound slope rises to a vertical dashed ridge at 295 FLOPs/byte before flattening into an orange compute ceiling at 989 TFLOP/s. The unbatched decode dot sits at 1 FLOP/byte (0.68% compute utilization), while the prompt prefill dot sits high on the compute plateau at 4,096 FLOPs/byte.",
        "caption": "Prefill saturates Tensor Core compute at 4,096 FLOPs/byte, while unbatched decode is trapped on the memory slope at 1 FLOP/byte.",
        "anchor": "### The Causal Serialization Barrier",
    },
    {
        "chapter": 2,
        "fig_idx": 3,
        "file": "vol3_processor_margin_003.svg",
        "alt": "Horizontal bar ladder showing the single-token memory shuttle latency floor for a 70-billion-parameter FP8 model across four accelerator generations: A100 at 34.3 milliseconds, H100 at 20.9 milliseconds, H200 at 14.6 milliseconds, and B200 at 8.8 milliseconds per token.",
        "caption": "Shuttling a 70B parameter model across the memory bus sets an irreducible step latency floor on single-token decode.",
        "anchor": "### The Structural Impedance Mismatch and the Serialization Tax",
    },
    {
        "chapter": 2,
        "fig_idx": 4,
        "file": "vol3_processor_margin_004.svg",
        "alt": "Budget envelope showing total invocation latency against a 15-second SLA deadline: Interface 1 (monolithic JSON) generates 1,850 tokens, taking 39.9 seconds and burning deep into the red violation zone, while Interface 2 (anchored diff) generates 80 tokens, completing in 2.85 seconds well within the allowable budget.",
        "caption": "Monolithic JSON breaches the 15-second latency SLA by 166%, while anchored diff completes in 2.85 seconds.",
        "anchor": "### Context Budgeting, Truncation, and the Quarantining Invariant",
    },

    # -------------------------------------------------------------------------
    # Chapter 03: Test-Time Deliberation
    # -------------------------------------------------------------------------
    {
        "chapter": 3,
        "fig_idx": 1,
        "file": "vol3_deliberation_margin_001.svg",
        "alt": "Exponential decay curve showing trajectory success dropping from 95% at step 1 down to 13% at step 40 for per-step error probability p=0.95.",
        "caption": "In unverified execution, compounding errors drive trajectory survival below 15% after 40 steps.",
        "anchor": "## Why One Candidate Can Fail {#sec-vol3-deliberation-insufficient-response}",
    },
    {
        "chapter": 3,
        "fig_idx": 2,
        "file": "vol3_deliberation_margin_002.svg",
        "alt": "Pareto frontier curves showing accuracy versus test-time compute scaling across depth, breadth, and feedback axes.",
        "caption": "Test-time deliberation scales log-linearly with verification budget before hitting domain saturation ceilings.",
        "anchor": "### Depth: Sequential Token Extension",
    },
    {
        "chapter": 3,
        "fig_idx": 3,
        "file": "vol3_deliberation_margin_003.svg",
        "alt": "Scatter and calibration curve comparing process reward model step scores against sparse terminal outcome labels.",
        "caption": "Step-level process reward models localize reasoning faults before terminal execution failure occurs.",
        "anchor": "## Process Verification {#sec-vol3-deliberation-verification}",
    },
    {
        "chapter": 3,
        "fig_idx": 4,
        "file": "vol3_deliberation_margin_004.svg",
        "alt": "Tree-search budget allocation diagram showing beam search branch pruning vs greedy rollout breadth.",
        "caption": "Pruning unpromising rollout branches reclaims 70% of inference compute for high-value reasoning trajectories.",
        "anchor": "### Selection Architectures: Voting, Scoring, and Execution Filtering",
    },

    # -------------------------------------------------------------------------
    # Chapter 04: Working Context
    # -------------------------------------------------------------------------
    {
        "chapter": 4,
        "fig_idx": 1,
        "file": "vol3_working_sets_margin_001.svg",
        "alt": "A curve rising quadratically to cross a linear projection baseline at 90k tokens, with the region beyond 90k shaded red to mark the quadratic prefill compute explosion.",
        "caption": "Above 90k tokens, quadratic self-attention overtakes linear projections, dominating prefill latency.",
        "anchor": "### Prefill Compute Scaling and Time-to-First-Token",
    },
    {
        "chapter": 4,
        "fig_idx": 2,
        "file": "vol3_working_sets_margin_002.svg",
        "alt": "Two-endpoint sparkline showing target attention probability plummeting from 97 percent in a 100-token prompt down to 3 percent in a 100,000-token prompt due to softmax denominator dilution.",
        "caption": "Accumulating 100k background tokens dilutes the softmax denominator, slashing target attention from 97% to 3%.",
        "anchor": "### Attentional Dynamics: Position Bias and Distractor Interference",
    },
    {
        "chapter": 4,
        "fig_idx": 3,
        "file": "vol3_working_sets_margin_003.svg",
        "alt": "A budget comparison bar diagram showing 57.6 gigabytes of unpruned reasoning scratchpad memory overflowing the 10 gigabyte GPU limit, compared to 0.96 gigabytes under the Two-Phase Commit protocol.",
        "caption": "Two-Phase Commit purges ephemeral scratchpads post-action, slashing 30-turn reasoning memory from 57.6 GB to 0.96 GB.",
        "anchor": "## Context Compaction {#sec-vol3-working-sets-compaction}",
    },
    {
        "chapter": 4,
        "fig_idx": 4,
        "file": "vol3_working_sets_margin_004.svg",
        "alt": "Four-rung log-scale memory ladder showing per-token KV cache footprint dropping from 2,560 kilobytes in MHA down to 320 kilobytes in GQA, 67.5 kilobytes in MLA, and 40 kilobytes in MQA.",
        "caption": "Multi-Head Latent Attention cuts KV cache footprint to 67.5 KB per token, a 38-fold compression over dense multi-head attention.",
        "anchor": "### Workload-Driven Sizing and the Effective Working-Set Frontier",
    },

    # -------------------------------------------------------------------------
    # Chapter 05: Paged Attention Memory
    # -------------------------------------------------------------------------
    {
        "chapter": 5,
        "fig_idx": 1,
        "file": "vol3_virtual_memory_margin_001.svg",
        "alt": "Memory waste percentage ladder comparing static contiguous allocation at 87.5 percent, dynamic buddy allocation at 34.4 percent, and paged key-value allocation at 0.12 percent, highlighting a 700-fold reduction.",
        "caption": "Paged attention collapses memory waste from 87.5% down to 0.12%, eliminating external fragmentation.",
        "anchor": "## Memory Fragmentation {#sec-vol3-kvcache-fragmentation}",
    },
    {
        "chapter": 5,
        "fig_idx": 2,
        "file": "vol3_virtual_memory_margin_002.svg",
        "alt": "Radix tree prefix matching diagram showing a 100-token shared system prefix partitioned into 6 full 16-token blocks with a 4-token unaligned tail truncated to prevent race conditions.",
        "caption": "Only fully packed blocks are shared across requests; partial block tails are truncated to prevent race conditions.",
        "anchor": "### Longest Prefix Matching",
    },
    {
        "chapter": 5,
        "fig_idx": 3,
        "file": "vol3_virtual_memory_margin_003.svg",
        "alt": "Execution timeline comparing monolithic prefill stalling decode loops for 1,195 milliseconds against chunked prefill pacing decode at 159 milliseconds.",
        "caption": "Chunking prompts into 512-token slices cuts P99 decode latency from 1,195 ms to 159 ms.",
        "anchor": "### Token-Budget Scheduling",
    },
    {
        "chapter": 5,
        "fig_idx": 4,
        "file": "vol3_virtual_memory_margin_004.svg",
        "alt": "Break-even curve showing PCIe host swapping beating full prompt recomputation whenever external tool execution pauses exceed 215 milliseconds.",
        "caption": "For tool pauses exceeding 215 ms, PCIe offload frees HBM while beating prefill recompute by 53.6×.",
        "anchor": "### Analytical Mechanics: The Break-Even Decision Frontier",
    },

    # -------------------------------------------------------------------------
    # Chapter 06: Persistent Storage
    # -------------------------------------------------------------------------
    {
        "chapter": 6,
        "fig_idx": 1,
        "file": "vol3_episodic_memory_margin_001.svg",
        "alt": "Accuracy decay curve showing retrieval precision falling from 94% with 5 retrieved passages down to 42% with 50 distractor passages.",
        "caption": "Injecting excess distractor passages degrades model reasoning fidelity through attentional distraction.",
        "anchor": "### Failure Modes: The Vocabulary Mismatch and Structural Blindness",
    },
    {
        "chapter": 6,
        "fig_idx": 2,
        "file": "vol3_episodic_memory_margin_002.svg",
        "alt": "Pareto frontier curve showing HNSW recall versus query latency across varying search depth hyperparameters.",
        "caption": "HNSW vector indexing achieves 96% recall within a 4 ms retrieval budget before latency knees steepen.",
        "anchor": "## Hybrid Retrieval {#sec-vol3-persistent-dense-retrieval}",
    },
    {
        "chapter": 6,
        "fig_idx": 3,
        "file": "vol3_episodic_memory_margin_003.svg",
        "alt": "Exponential fan-out tree showing 2-hop search retrieving 25 entities and 4-hop search expanding to over 600 nodes.",
        "caption": "Unbounded multi-hop graph expansion triggers exponential context bloat across traversal depth.",
        "anchor": "## Multi-Hop Retrieval {#sec-vol3-persistent-graphrag}",
    },
    {
        "chapter": 6,
        "fig_idx": 4,
        "file": "vol3_episodic_memory_margin_004.svg",
        "alt": "Timeline showing stale memory cache hits causing silent execution faults after repository branch checkouts.",
        "caption": "Stale vector embeddings violate causal consistency unless invalidation is coupled to workspace mutation events.",
        "anchor": "## Storage Invalidation {#sec-vol3-persistent-invalidation}",
    },

    # -------------------------------------------------------------------------
    # Chapter 07: Tool Execution
    # -------------------------------------------------------------------------
    {
        "chapter": 7,
        "fig_idx": 1,
        "file": "vol3_actuation_margin_001.svg",
        "alt": "Horizontal log-scale ladder comparing tool IPC transport latency: Local UDS at 6.3 microseconds, Intra-DC HTTP at 0.61 milliseconds, and WAN Cloud SSE at 35 milliseconds.",
        "caption": "Remote HTTP transports impose a 97× to 5,600× latency tax over local Unix domain sockets for inter-tool IPC.",
        "anchor": "### Transport Topologies and Latency-Isolation Trade-Offs",
    },
    {
        "chapter": 7,
        "fig_idx": 2,
        "file": "vol3_actuation_margin_002.svg",
        "alt": "Segmented stream buffer diagram contrasting a 4.2 MB unconstrained tool output against a 4 KB sandwich-truncated window preserving head and tail lines.",
        "caption": "Sandwich truncation discards intermediate execution noise while pinning critical head schemas and tail stack traces.",
        "anchor": "## Observation Stream Truncation {#sec-vol3-actuation-streaming}",
    },
    {
        "chapter": 7,
        "fig_idx": 3,
        "file": "vol3_actuation_margin_003.svg",
        "alt": "Logarithmic timescale ladder comparing 20 ms GPU token generation against 1,200 ms subprocess build execution.",
        "caption": "External tool execution outlasts neural token generation by up to five orders of magnitude, dominating end-to-end makespan.",
        "anchor": "## Idempotent Action Execution {#sec-vol3-actuation-idempotency}",
    },
    {
        "chapter": 7,
        "fig_idx": 4,
        "file": "vol3_actuation_margin_004.svg",
        "alt": "Knee curve plotting tool selection error percentage against total declared tools, showing a steep error inflection above 30 functions.",
        "caption": "Past a threshold catalog size, semantic interference triggers a sharp inflection in misrouted tool calls.",
        "anchor": "### The Schema Quality Effect and Semantic Conditioning",
    },

    # -------------------------------------------------------------------------
    # Chapter 08: Environmental Isolation
    # -------------------------------------------------------------------------
    {
        "chapter": 8,
        "fig_idx": 1,
        "file": "vol3_virtualization_margin_001.svg",
        "alt": "Horizontal bar ladder showing host system calls exposed to untrusted code: Container at 450, gVisor at 50, Firecracker at 35, and WebAssembly at 0.",
        "caption": "MicroVMs and WebAssembly collapse host kernel syscall exposure by over 90% compared to containers.",
        "anchor": "## Adversarial Threat Models {#sec-vol3-virtualization-threat-model}",
    },
    {
        "chapter": 8,
        "fig_idx": 2,
        "file": "vol3_virtualization_margin_002.svg",
        "alt": "A latency curve plotting file mutation stall against file size; latency remains sub-millisecond up to 1 megabyte, then steepens to a 10.97-second stall at 1.2 gigabytes under parallel worker contention, with the region past 100 megabytes shaded red.",
        "caption": "Modifying a single byte triggers full-file copy-up, turning sub-millisecond writes into multi-second NVMe stalls for gigabyte assets.",
        "anchor": "## Copy-on-Write Filesystem Overlays {#sec-vol3-virtualization-cow}",
    },
    {
        "chapter": 8,
        "fig_idx": 3,
        "file": "vol3_virtualization_margin_003.svg",
        "alt": "Two log-scale horizontal bars comparing covert DNS exfiltration bandwidth: an unrestricted channel at 234 kilobytes per second exfiltrates in 0.28 seconds, while a rate-clamped resolver drops bandwidth to 30 bytes per second, stretching exfiltration to 36.4 minutes.",
        "caption": "Rate-clamping DNS queries chokes covert exfiltration bandwidth by 7,800x, expanding the anomaly detection window from 0.28 seconds to 36 minutes.",
        "anchor": "## Network Egress Firewalls {#sec-vol3-virtualization-network}",
    },
    {
        "chapter": 8,
        "fig_idx": 4,
        "file": "vol3_virtualization_margin_004.svg",
        "alt": "Log-scale latency ladder comparing sandbox acquisition times: Cold MicroVM boot at 450 milliseconds, Eager Snapshot restore at 35 milliseconds, and Pre-Warmed CoW Pool with UFFD at 3 milliseconds, matching container speeds without sacrificing hardware isolation.",
        "caption": "Pre-warmed memory pooling with on-demand userfaultfd paging delivers 3 ms microVM acquisition, matching container speed while preserving hardware isolation.",
        "anchor": "## MicroVM Kernel Isolation {#sec-vol3-virtualization-microvms}",
    },

    # -------------------------------------------------------------------------
    # Chapter 09: Supervisory Control Planes
    # -------------------------------------------------------------------------
    {
        "chapter": 9,
        "fig_idx": 1,
        "file": "vol3_checkpointing_margin_001.svg",
        "alt": "Logarithmic scale bar chart comparing per-trajectory memory footprint: 256 bytes for the Agent Control Block descriptor, 128 kilobytes for the host context buffer, and 8.59 gigabytes for the physical GPU KV cache, illustrating a 35-million-fold decoupling.",
        "caption": "Decoupling 256-byte control descriptors from 8.59-gigabyte tensor state enables lightweight process scheduling across distributed worker nodes.",
        "anchor": "## The Agent Control Block {#sec-vol3-controlplane-acb}",
    },
    {
        "chapter": 9,
        "fig_idx": 2,
        "file": "vol3_checkpointing_margin_002.svg",
        "alt": "Knee curve plotting operator signal interception latency against decode polling frequency K, dropping steeply from 68.2 seconds without sub-checkpoints to 533 milliseconds at K=16 tokens.",
        "caption": "Polling interrupts every 16 tokens compresses supervisor signal response latency from 68 seconds down to 533 milliseconds.",
        "anchor": "## Trajectory Lifecycle States {#sec-vol3-controlplane-statemachine}",
    },
    {
        "chapter": 9,
        "fig_idx": 3,
        "file": "vol3_checkpointing_margin_003.svg",
        "alt": "Paired bar comparison showing 9.06 TiB of memory consumed by 36,000 synchronous thread stacks versus 141 MiB consumed by asynchronous Agent Control Block stubs during human approval pauses.",
        "caption": "Asynchronous escrow stubs reduce dormant memory footprint by 67,584x during multi-hour human-in-the-loop pauses.",
        "anchor": "## Human Escrow Protocols {#sec-vol3-controlplane-hitl}",
    },
    {
        "chapter": 9,
        "fig_idx": 4,
        "file": "vol3_checkpointing_margin_004.svg",
        "alt": "Four-tier threshold ladder illustrating progressive resource exhaustion: nominal operation below 70%, advisory warnings between 70% and 90%, defensive shedding between 90% and 100%, and immediate preemption at 100%.",
        "caption": "Tiered exhaustion thresholds trigger progressive mitigation before trajectory runaway exhausts global cluster token quotas.",
        "anchor": "## Single-Node Runtime Scheduling {#sec-vol3-controlplane-scheduling}",
    },

    # -------------------------------------------------------------------------
    # Chapter 10: Trajectory Persistence
    # -------------------------------------------------------------------------
    {
        "chapter": 10,
        "fig_idx": 1,
        "file": "vol3_interrupts_margin_001.svg",
        "alt": "Write-ahead logging timeline showing monotonic append operations decoupled from asynchronous snapshot checkpoints.",
        "caption": "Append-only event sourcing records non-deterministic external tool returns before state mutation occurs.",
        "anchor": "## Append-Only Event Sourcing {#sec-vol3-persistence-event-sourcing}",
    },
    {
        "chapter": 10,
        "fig_idx": 2,
        "file": "vol3_interrupts_margin_002.svg",
        "alt": "Latency comparison showing group commit amortizing fsync overhead across concurrent trajectory streams.",
        "caption": "Group commit amortizes synchronous storage latency across active trajectory streams.",
        "anchor": "## Write-Ahead Logging Discipline {#sec-vol3-persistence-wal}",
    },
    {
        "chapter": 10,
        "fig_idx": 3,
        "file": "vol3_interrupts_margin_003.svg",
        "alt": "Branching replay timeline illustrating how non-deterministic floating-point accumulation causes trajectory paths to diverge.",
        "caption": "Subtle kernel non-determinism forces deterministic replayers to inject logged tool outcomes rather than re-executing actions.",
        "anchor": "## Replay Divergence Diagnostics {#sec-vol3-persistence-non-determinism}",
    },
    {
        "chapter": 10,
        "fig_idx": 4,
        "file": "vol3_interrupts_margin_004.svg",
        "alt": "Compaction ratio curve showing trajectory log compression over thousands of operational turns.",
        "caption": "Periodic snapshot compaction reclaims 85% of log volume while preserving deterministic crash recoverability.",
        "anchor": "## Log Compaction Policies {#sec-vol3-persistence-compaction}",
    },

    # -------------------------------------------------------------------------
    # Chapter 11: Fault Recovery
    # -------------------------------------------------------------------------
    {
        "chapter": 11,
        "fig_idx": 1,
        "file": "vol3_scheduling_margin_001.svg",
        "alt": "Horizontal bar chart comparing forward execution latency of 5,200 ms against compensating saga unwind latency of 295 ms.",
        "caption": "Saga compensation unwinds external state in 295 ms (17.6× faster than forward execution) without blocking transaction locks.",
        "anchor": "## The Trajectory Saga Pattern {#sec-vol3-sagas-architecture}",
    },
    {
        "chapter": 11,
        "fig_idx": 2,
        "file": "vol3_scheduling_margin_002.svg",
        "alt": "Phase portrait diagram delineating the point of no return where non-compensable side-effects preclude state rollback.",
        "caption": "Crossing irreversible pivot actions requires durable physical escrow before irreversible side-effects commit.",
        "anchor": "## Forward Recovery Versus Rollback {#sec-vol3-sagas-forward-vs-backward}",
    },
    {
        "chapter": 11,
        "fig_idx": 3,
        "file": "vol3_scheduling_margin_003.svg",
        "alt": "State machine diagram showing circuit breaker transitions from closed to open upon consecutive upstream RPC timeouts.",
        "caption": "Circuit breakers trip after three consecutive external timeouts, shielding downstream infrastructure from retry storms.",
        "anchor": "## Tool Circuit Breakers {#sec-vol3-sagas-containment}",
    },
    {
        "chapter": 11,
        "fig_idx": 4,
        "file": "vol3_scheduling_margin_004.svg",
        "alt": "Blast radius concentric circles showing how directory-level chroot boundaries contain unauthorized file mutations.",
        "caption": "Subprocess chroot jail partitions confine filesystem blast radius to isolated task directories.",
        "anchor": "## Blast Radius Quarantine {#sec-vol3-sagas-quarantine}",
    },

    # -------------------------------------------------------------------------
    # Chapter 12: Trajectory Harvesting
    # -------------------------------------------------------------------------
    {
        "chapter": 12,
        "fig_idx": 1,
        "file": "vol3_data_flywheel_margin_001.svg",
        "alt": "Hierarchy ladder showing the four tiers of the Systems Intervention Ladder: Context Injection at 1 second latency, Schema Redesign at 15 minutes, Runtime Hardening at 1 hour, and Model Retraining at 10,000 seconds with high compute cost and regression risk.",
        "caption": "Exhausting deterministic runtime tiers resolves 90% of failures before escalating to model retraining across four orders of magnitude in latency.",
        "anchor": "## Capability Gap Diagnosis {#sec-vol3-flywheel-gap}",
    },
    {
        "chapter": 12,
        "fig_idx": 2,
        "file": "vol3_data_flywheel_margin_002.svg",
        "alt": "Funnel diagram depicting candidate survival rates across five verifier stages: 100% raw proposals drop to 65% after syntax checks, 26% after static typing, 6.5% after dynamic microVM tests, and 3.9% final yield after state delta and semantic verification.",
        "caption": "Progressive filtering sheds 93.5% of invalid proposals before expensive microVM test execution, cutting cluster compute by 68.6%.",
        "anchor": "### Economic Cost-Yield Formulation and Scheduling",
    },
    {
        "chapter": 12,
        "fig_idx": 3,
        "file": "vol3_data_flywheel_margin_003.svg",
        "alt": "Line graph showing task failure rate versus execution horizon up to 30 steps. Pristine-only policy failure explodes quadratically crossing a 50% knee at step 14, while recovery-trained policy failure remains bounded linearly below 25%.",
        "caption": "Without recovery demonstrations, unforced errors trigger quadratic compounding that crosses a 50% failure knee at step 14.",
        "anchor": "### The Peril of Pristine-Only Datasets and Covariate Shift",
    },
    {
        "chapter": 12,
        "fig_idx": 4,
        "file": "vol3_data_flywheel_margin_004.svg",
        "alt": "Paired stacked bar chart comparing trajectory count proportions (65% pristine, 25% recovery, 10% negative) against token volume proportions (45.5% pristine, 45% recovery, 9.5% negative), illustrating the 2.6x expansion factor of recovery traces.",
        "caption": "Multi-turn diagnostic loops cause recovery traces to consume 45% of the token budget despite comprising only 25% of trajectories.",
        "anchor": "### Corpus Composition and Optimal Mixing Ratios",
    },

    # -------------------------------------------------------------------------
    # Chapter 13: Supervised Adaptation
    # -------------------------------------------------------------------------
    {
        "chapter": 13,
        "fig_idx": 1,
        "file": "vol3_sft_margin_001.svg",
        "alt": "Four-rung logarithmic memory ladder showing prompt prefix reduction: naive serialization at 16,384 tokens down to compiled prefix at 128 tokens, shrinking per-instance KV cache from 10.5 GB to 0.08 GB.",
        "caption": "Context compilation compresses runtime prompt prefixes by 128×, reducing 70B model per-instance KV-cache consumption from 10.5 GB down to 0.08 GB.",
        "anchor": "## Trajectory Example Serialization {#sec-vol3-sft-serialization}",
    },
    {
        "chapter": 13,
        "fig_idx": 2,
        "file": "vol3_sft_margin_002.svg",
        "alt": "Efficiency comparison bar showing packed training sequences achieving 98% occupancy compared to 19.5% for padded batches, cutting bubble waste by 80%.",
        "caption": "Sequence packing with block-diagonal attention eliminates padding bubbles, boosting training throughput by 5×.",
        "anchor": "### Sequence Packing Mechanics and Buffer Utilization",
    },
    {
        "chapter": 13,
        "fig_idx": 3,
        "file": "vol3_sft_margin_003.svg",
        "alt": "Validation error curve across LoRA rank dimensions showing syntax error knee collapsing at rank 16.",
        "caption": "Low-rank adapters require rank 16 to stabilize structural JSON syntax before semantic task adaptation converges.",
        "anchor": "## Parameter-Efficient Memory Bounds {#sec-vol3-sft-peft}",
    },
    {
        "chapter": 13,
        "fig_idx": 4,
        "file": "vol3_sft_margin_004.svg",
        "alt": "Divergence sparkline comparing quadratic compounding error under naive behavioral cloning against bounded linear drift under interactive DAGGER training.",
        "caption": "Supervised behavioral cloning suffers quadratic error compounding without interactive DAGGER data aggregation.",
        "anchor": "## Autoregressive Exposure Bias {#sec-vol3-sft-exposure-bias}",
    },

    # -------------------------------------------------------------------------
    # Chapter 14: Verifiable Reinforcement Learning
    # -------------------------------------------------------------------------
    {
        "chapter": 14,
        "fig_idx": 1,
        "file": "vol3_rlvr_margin_001.svg",
        "alt": "Two-rung horizontal memory ladder comparing persistent cluster VRAM for 70B parameter models: PPO at 2,380 GB (92% VRAM, 4.8 GB context headroom) versus GRPO at 1,260 GB (49% VRAM, 41.2 GB context headroom).",
        "caption": "Eliminating the Critic network reclaims 1.12 TB of accelerator memory, freeing 41.2 GB of per-GPU headroom for extended context windows.",
        "anchor": "### The Memory Wall of Actor-Critic Runtimes {#sec-vol3-rlvr-grpo-memory-wall}",
    },
    {
        "chapter": 14,
        "fig_idx": 2,
        "file": "vol3_rlvr_margin_002.svg",
        "alt": "Concave exploration curve showing pass@k trajectory completion surging from 18% at k=1 to 79.6% at k=8 and 95.8% at k=16.",
        "caption": "Scaling parallel rollout exploration from k=1 to k=8 boosts verifiable task pass rates from 18% to nearly 80%.",
        "anchor": "## Group Relative Policy Optimization {#sec-vol3-rlvr-grpo}",
    },
    {
        "chapter": 14,
        "fig_idx": 3,
        "file": "vol3_rlvr_margin_003.svg",
        "alt": "Two-trajectory plot comparing policy entropy across training steps: unconstrained RL plunges to zero entropy degeneracy by step 40, while GRPO with reference-model KL stabilization maintains a healthy 2.2 nats plateau.",
        "caption": "Unconstrained policy optimization collapses reasoning entropy to zero, producing repetitive degeneracies unless KL penalties anchor exploration.",
        "anchor": "## Reasoning Entropy Collapse {#sec-vol3-rlvr-pathologies}",
    },
    {
        "chapter": 14,
        "fig_idx": 4,
        "file": "vol3_rlvr_margin_004.svg",
        "alt": "Two-tier operational comparison showing rollout inference operating at 4.2 FLOP/byte in memory-bandwidth-bound execution at 32% MFU versus training gradient compute operating at 145 FLOP/byte in compute-bound execution at 58% MFU.",
        "caption": "Disaggregating memory-bound rollout inference from compute-bound gradient updates prevents accelerator underutilization across the training fleet.",
        "anchor": "## Disaggregated Rollout Infrastructure {#sec-vol3-rlvr-infrastructure}",
    },

    # -------------------------------------------------------------------------
    # Chapter 15: Multi-Agent Coordination
    # -------------------------------------------------------------------------
    {
        "chapter": 15,
        "fig_idx": 1,
        "file": "vol3_multi_agent_margin_001.svg",
        "alt": "Paired horizontal bar chart comparing a single agent baseline against a four-worker agent pool, showing makespan dropping from 14.5 minutes to 7.0 minutes while token volume surges from 125,000 to 275,000 tokens.",
        "caption": "Multi-agent parallelization achieves a 2.07x makespan compression at the cost of a 2.20x token explosion, trading compute bandwidth for wall-clock concurrency.",
        "anchor": "## The Delegation Trade-Off {#sec-vol3-multiagent-need}",
    },
    {
        "chapter": 15,
        "fig_idx": 2,
        "file": "vol3_multi_agent_margin_002.svg",
        "alt": "Line plot showing write collision probability surging above 80% beyond five workers under Zipfian contention, compared to 12% under uniform access.",
        "caption": "Under Zipfian hotspot file access, unisolated agent fleets cross an 80% write collision rate beyond five workers, rendering naive shared workspaces unviable.",
        "anchor": "## Optimistic Concurrency Control {#sec-vol3-multiagent-concurrency}",
    },
    {
        "chapter": 15,
        "fig_idx": 3,
        "file": "vol3_multi_agent_margin_003.svg",
        "alt": "Four-rung horizontal ladder chart showing monotonically contracting capability scopes across three delegation hops: root supervisor, domain specialist, and single-tool worker.",
        "caption": "Delegated capabilities contract monotonically across execution depth, cryptographically chaining HMAC caveats to enforce zero ambient authority down to a single-tool sandbox.",
        "anchor": "## Attenuated Capability Delegation {#sec-vol3-multiagent-authority}",
    },
    {
        "chapter": 15,
        "fig_idx": 4,
        "file": "vol3_multi_agent_margin_004.svg",
        "alt": "Line plot of Gunther Universal Scalability Law showing multi-agent speedup peaking at 2.45x with four workers before coherency overhead forces retrograde collapse.",
        "caption": "Pairwise merge coherency (kappa = 0.02) drives multi-agent speedup into retrograde collapse beyond four workers, making eight agents nearly twice as slow as an optimized single agent.",
        "anchor": "## Single-Agent Baseline Benchmarking {#sec-vol3-multiagent-evaluation}",
    },

    # -------------------------------------------------------------------------
    # Chapter 16: System Observability
    # -------------------------------------------------------------------------
    {
        "chapter": 16,
        "fig_idx": 1,
        "file": "vol3_observability_margin_001.svg",
        "alt": "Log-scale column chart comparing trajectory telemetry footprints: 100 MB raw trace in-band vs. 17 MB zstd compressed blob vs. 100 KB OTLP metadata index, achieving a 99.9% wire reduction.",
        "caption": "Asynchronous payload offloading and tail-based sampling compress active trajectory telemetry indexes from 100 MB down to 25 KB, averting gRPC frame exhaustion while preserving 100% of production failure traces.",
        "anchor": "## Distributed Trajectory Tracing {#sec-vol3-observability-tracing}",
    },
    {
        "chapter": 16,
        "fig_idx": 2,
        "file": "vol3_observability_margin_002.svg",
        "alt": "Hyperbolic curve of 95 percent confidence interval half-width dropping from plus minus 16.4 percent at N=30 down to plus minus 2.4 percent at N=1,430 tasks under binomial benchmark sampling.",
        "caption": "Evaluating on fewer than 200 tasks produces wide confidence intervals (plus minus 16%) that obscure true model regressions behind statistical noise.",
        "anchor": "### Finite-Sample Uncertainty and the Wilson Score Interval {#sec-vol3-observability-wilson}",
    },
    {
        "chapter": 16,
        "fig_idx": 3,
        "file": "vol3_observability_margin_003.svg",
        "alt": "Sequential hypothesis testing corridor showing log-likelihood ratio boundaries: candidate routing breaches the upper abort threshold at sample 8, terminating rollout without waiting for fixed sample batching.",
        "caption": "Wald sequential probability ratio testing detects degraded candidate policies within 8 turns, cutting canary rollout blast radius by 80%.",
        "anchor": "## Staged Canary Deployments {#sec-vol3-observability-releases}",
    },
    {
        "chapter": 16,
        "fig_idx": 4,
        "file": "vol3_observability_margin_004.svg",
        "alt": "Amdahl speedup curves showing overall agent speedup capped at 1.54x when accelerating model inference alone, versus 1.73x when optimizing system scaffolding tools.",
        "caption": "Because foundation model inference accounts for only 35% of trajectory duration, accelerating host scaffolding tools yields higher end-to-end speedups than model optimization alone.",
        "anchor": "## The Multi-Layer Evaluation Contract {#sec-vol3-observability-contract}",
    },

    # -------------------------------------------------------------------------
    # Chapter 17: Serving Economics
    # -------------------------------------------------------------------------
    {
        "chapter": 17,
        "fig_idx": 1,
        "file": "vol3_tokenomics_margin_001.svg",
        "alt": "Hyperbolic curve plotting effective cost multiplier against task acceptance rate, showing a sharp upward inflection into a red failure-odds wash below 50 percent acceptance.",
        "caption": "Sub-50% task acceptance triggers a hyperbolic cost cliff that overwhelms nominal per-token discounts.",
        "anchor": "## Task Cost Accounting {#sec-vol3-tokenomics-accounting}",
    },
    {
        "chapter": 17,
        "fig_idx": 2,
        "file": "vol3_tokenomics_margin_002.svg",
        "alt": "Three horizontal bars decomposing trajectory wall-clock latency: pytest sandbox tool execution dominates at 78.6 percent, container lifecycle takes 11.9 percent, and model inference accounts for only 9.4 percent.",
        "caption": "Tool sandbox execution dominates trajectory wall-clock time, capping the system speedup achievable from GPU model acceleration.",
        "anchor": "## Critical Path Latency {#sec-vol3-tokenomics-criticalpath}",
    },
    {
        "chapter": 17,
        "fig_idx": 3,
        "file": "vol3_tokenomics_margin_003.svg",
        "alt": "Logarithmic horizontal ladder comparing per-task serving costs: Frontier monolithic at 60.00 dollars per thousand tasks versus expected cascade at 8.40 dollars, achieving an 86 percent cost reduction.",
        "caption": "Tiered model cascading resolves 88% of tasks on lightweight models, cutting expected cost by 86% against monolithic frontier routing.",
        "anchor": "## Tiered Model Cascades {#sec-vol3-tokenomics-cascades}",
    },
    {
        "chapter": 17,
        "fig_idx": 4,
        "file": "vol3_tokenomics_margin_004.svg",
        "alt": "Budget envelope bars showing committed spend of 4.00 dollars and active in-flight escrow of 2.40 dollars totaling 6.40 dollars against a hard 10.00 dollar spending ceiling.",
        "caption": "Hierarchical ledgers hold in-flight funds in escrow, preventing concurrent child subagents from violating the root spending ceiling.",
        "anchor": "## Monotonic Spending Governance {#sec-vol3-tokenomics-governance}",
    },

    # -------------------------------------------------------------------------
    # Chapter 18: Architectural Synthesis
    # -------------------------------------------------------------------------
    {
        "chapter": 18,
        "fig_idx": 1,
        "file": "vol3_conclusion_margin_001.svg",
        "alt": "Exponential decay curve illustrating the open-loop failure wall: even at 95% single-step accuracy, trajectory reliability collapses to 60% by step 10 and 36% by step 20.",
        "caption": "Even at 95% single-step accuracy, open-loop trajectories decay below 36% by step 20.",
        "anchor": "## The Capstone Reference Architecture {#sec-vol3-conclusion-capstone}",
    },
    {
        "chapter": 18,
        "fig_idx": 2,
        "file": "vol3_conclusion_margin_002.svg",
        "alt": "Four-tier escalation ladder showing supervisor state transitions: Tier 1 Autonomous below 60% budget, Tier 2 Human Review at 60-80%, Tier 3 Preemption at 80-100%, and Tier 4 Emergency Quarantine at 100%.",
        "caption": "Graduated escalation envelopes prevent catastrophic budget overruns while maintaining autonomous throughput for nominal operations.",
        "anchor": "## Workload Contract Specification {#sec-vol3-conclusion-envelope}",
    },
    {
        "chapter": 18,
        "fig_idx": 3,
        "file": "vol3_conclusion_margin_003.svg",
        "alt": "Three-tier verification pyramid displaying deterministic mechanical verification at the base, statistical gym evaluation in the middle, and runtime canary telemetry at the peak.",
        "caption": "Safety cases anchor sub-millisecond mechanical proofs beneath statistical gym bounds and canary telemetry.",
        "anchor": "## Empirical Safety Cases {#sec-vol3-conclusion-safety}",
    },
    {
        "chapter": 18,
        "fig_idx": 4,
        "file": "vol3_conclusion_margin_004.svg",
        "alt": "Stacked horizontal bars comparing Software 1.0 and Software 3.0 engineering effort, showing accidental complexity shrinking from 60 percent to 10 percent while essential complexity expands from 40 percent to 90 percent.",
        "caption": "Agentic synthesis compresses accidental syntax by 12x, elevating essential verification to 90% of effort.",
        "anchor": "## Essential Complexity {#sec-vol3-conclusion-brooks}",
    },
]

def format_margin_block(fig: dict) -> str:
    fn = fig["file"]
    alt = fig["alt"]
    cap = fig["caption"]
    return f"::: {{.column-margin}}\n![](images/svg/{fn}){{width=\"100%\" fig-alt=\"{alt}\"}}\n\n*{cap}*\n:::"

def embed_figures(dry_run: bool = True):
    # Group figures by chapter
    by_chapter = {}
    for f in ALL_FIGURES:
        by_chapter.setdefault(f["chapter"], []).append(f)

    for ch, figs in sorted(by_chapter.items()):
        cdir, cfile = CHAPTER_FILES[ch]
        qpath = BOOKS / cdir / cfile
        content = qpath.read_text(encoding="utf-8")
        orig_len = len(content)

        # Sort figures in the order they appear in the file to preserve indexing
        fig_positions = []
        for f in figs:
            pos = content.find(f["anchor"])
            if pos == -1:
                raise ValueError(f"Anchor '{f['anchor']}' not found in Ch {ch:02d} ({cfile})")
            fig_positions.append((pos, f))

        fig_positions.sort(key=lambda x: x[0], reverse=True) # insert from bottom up!

        for pos, f in fig_positions:
            block = format_margin_block(f)
            # Find the end of the paragraph following pos
            # If pos is a header (starts with ## or ###), find the next blank line after it, or after the introductory paragraph
            anchor = f["anchor"]
            anchor_idx = content.find(anchor)
            # Find the end of the line containing the anchor
            line_end = content.find("\n", anchor_idx)
            if line_end == -1:
                line_end = len(content)

            # Look for the end of the paragraph (double newline) following the anchor line
            next_blank = content.find("\n\n", line_end)
            if next_blank != -1:
                insert_pos = next_blank + 2
            else:
                insert_pos = line_end + 1

            # Insert block
            insertion = f"\n{block}\n"
            content = content[:insert_pos] + insertion + content[insert_pos:]
            print(f"[INSERT] Ch {ch:02d}: {f['file']} placed after '{anchor[:40]}...'")

        if not dry_run:
            qpath.write_text(content, encoding="utf-8")
            print(f"[SAVED] Ch {ch:02d}: {qpath.name} (+{len(content) - orig_len} chars)")

if __name__ == "__main__":
    import sys
    dry = "--apply" not in sys.argv
    print(f"=== Running with dry_run={dry} ===")
    embed_figures(dry_run=dry)
    if dry:
        print("\nTo apply insertions across all 18 chapters, run with --apply")
