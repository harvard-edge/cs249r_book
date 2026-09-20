# Volume III: Comprehensive Textbook Figure Quality Audit & Redesign Master Plan

**Volume:** Volume III (*Agentic Machine Learning Systems: The Systems Engineering of Inference-Time Compute and Autonomous Control Loops*)
**Standard of Comparison:** Graduate Computer Systems Textbooks (*Hennessy & Patterson*, *Saltzer & Kaashoek*, *Silberschatz et al.*, *Cormen et al.*, *W. Richard Stevens*)
**Design Archetype:** MLSysBook Volume III Figure 1.10 Standard (`von_neumann_agent_architecture.svg`)
**Date:** September 16, 2026
**Audited Assets:** 101 Figure References Across 18 Chapters and Appendices

---

## 1. Executive Summary & Quality Dashboard

This comprehensive audit investigates all **101 figure references** across Volume III to answer a fundamental pedagogical question:
> *"Do these diagrams function as true academic textbook schematics that illuminate underlying physical and systems mechanisms, or do they suffer from the presentation slide / pitch-deck infographic anti-pattern?"*

The forensic audit reveals that while the technical concepts are profound and mathematically grounded in the text, **over half of the current visual assets (54.5%) were authored as slide-presentation card decks**. Rather than illustrating structural dataflows, memory layouts, finite state automata, and hardware/software interfaces, they feature rounded rectangular cards packed with bullet points (`•`), marketing status badges (`153,600x INFLATION`, `STATUS: CRITICAL`), evaluative verdict banners (`VERDICT: UNVIABLE IN PRODUCTION`), and prose blocks that properly belong in the chapter body text or figure captions. Furthermore, five figures in Part III contain **zero connecting lines or vector paths**, functioning purely as graphical bulleted lists.

### Global Quality Distribution

| Classification | Count | Percentage | Definition & Architectural Disposition |
| :--- | :---: | :---: | :--- |
| **`[PASS - Textbook Ready]`** | **17** | **16.8%** | Fully conforms to graduate textbook standards. Clean structural schematics, memory block layouts, or quantitative Cartesian/logarithmic coordinate plots. Minimal/no text in canvas; restrained palette. |
| **`[NEEDS MINOR REFINEMENT]`** | **29** | **28.7%** | Strong architectural or mathematical foundation contaminated by secondary slide artifacts (embedded bottom banners, unnecessary bullet points inside state boxes, redundant card wrappers). Requires surgical cleanup into `<name>_v2.svg`. |
| **`[NEEDS V2 - Slide to Textbook Overhaul]`** | **55** | **54.5%** | Severe slide-deck anti-pattern. Built from grids/columns of cards containing bulleted prose; zero or negligible dataflow/control paths; table-as-SVG; or severe text-to-diagram mismatch. Requires ground-up structural redesign into `<name>_v2.svg`. |
| **Total Figures Audited** | **101** | **100.0%** | Full visual asset census across 18 chapters and appendices. |

```
Volume III Global Quality Distribution (101 figures):
========================================================================================
[PASS - Textbook Ready]                 ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (16.8% - 17)
[NEEDS MINOR REFINEMENT]                ████████████░░░░░░░░░░░░░░░░░░░░░░░░░  (28.7% - 29)
[NEEDS V2 - Slide to Textbook Overhaul] ██████████████████████░░░░░░░░░░░░░░░  (54.5% - 55)
========================================================================================
```

---

## 2. The Slide Infographic Anti-Pattern vs. Academic Textbook Standard

To maintain rigorous consistency across all redesigns, every figure is evaluated against the clear boundary separating presentation slides from academic textbook schematics:

```
+-------------------------------------------------------------------+-------------------------------------------------------------------+
| PRESENTATION SLIDE INFOGRAPHIC ANTI-PATTERN                       | ACADEMIC TEXTBOOK SCHEMATIC IDEAL (Hennessy & Patterson Standard) |
+-------------------------------------------------------------------+-------------------------------------------------------------------+
| * Cards/boxes packed with bulleted prose ("• Python 3.10...")    | * Schematic execution units, hardware blocks, and physical buses  |
| * Slogan badges ("TIER 1", "STATUS: OPTIMAL", "153,600x SPIKE")   | * Precise signal names, clock edges, and bus bit-widths ([B, d])  |
| * Evaluative verdicts ("VERDICT: UNVIABLE IN PRODUCTION")         | * Mathematical notations ($O(N^2)$, $\tau_{\text{prefill}}$, $\lambda$) |
| * Bottom takeaway banners ("KEY ARCHITECTURAL INVARIANT: ...")   | * Concise annotations with direct leader lines to specific pins   |
| * Checklists with checkmarks and crosses (✓ / ✗)                  | * Formal Harel statecharts (entry/do/exit) & sequence ladders     |
| * High-contrast, saturated primary colors and heavy card outlines | * Restrained semantic palette: slate, navy, muted crimson, amber  |
| * Zero-path floating card grids (tables masquerading as SVGs)     | * Explicit directional dataflow and control paths (orthogonal H/V)|
| * Prose paragraphs drawn inside boxes duplicating caption text   | * Caption owns the narrative; canvas owns the structural geometry |
+-------------------------------------------------------------------+-------------------------------------------------------------------+
```

### Canvas Hygiene Violations Identified Across Corpus
1. **The Full-Width Bottom Banner:** 28 SVGs embed full-width banners along the bottom (`y="400"` to `y="450"`) containing bold titles like `KEY ARCHITECTURAL INVARIANT:` followed by 2–3 lines of prose. This violates book style guidelines, wastes 15–20% of vertical canvas headroom, duplicates the Quarto `fig-cap`, and imparts a corporate training slide appearance.
2. **Zero-Path Vector Files:** 5 figures (`prefill_decode_disaggregation.svg`, `trajectory_scheduling_architecture.svg`, `drf_multi_resource_agent_allocation.svg`, `reversibility_boundary_preemption_matrix.svg`, and `isolation_spectrum.svg`) contain literally zero vector paths, lines, or connectors. They are formatted presentation tables.
3. **Severe Text-to-Diagram Mismatches:**
   - `tokenomics_memory_hierarchy_caching.svg`: Caption describes a Multi-Level Feedback Queue (MLFQ) with priority queues ($Q_0, Q_1, Q_2$), but the SVG contains zero queues.
   - `progressive_autonomy_spectrum.svg`: Caption describes a 4-stage canary deployment pipeline, but the SVG is a 4-tier capability table with a quote box.
   - `five_part_agent_runtime_architecture.svg`: Chapter 18 conclusion describes the Stochastic Computer's 6 subsystems, but the figure is a static stack of 5 cards with bullet points.

---

## 3. Volume-Wide Part-by-Part Scorecard

| Part | Chapters Included | Total Figures | `[PASS]` | `[NEEDS REFINEMENT]` | `[NEEDS V2]` | % Needing Overhaul |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Part I & II: Foundation & Memory** | Ch 01–06 | 44 | 12 | 18 | 14 | 31.8% |
| **Part III: Systems Foundations** | Ch 07–11 | 31 | 4 | 9 | 18 | 58.1% |
| **Part IV: Synthesis & Training** | Ch 12–14 | 14 | 1 | 1 | 12 | 85.7% |
| **Part V & VI: Consensus & Frontier** | Ch 15–18, App A | 12 | 0 | 1 | 11 | 91.7% |
| **Total** | **Chapters 01–18 + App A** | **101** | **17** | **29** | **55** | **54.5%** |

---

## 4. Comprehensive Chapter-by-Chapter Figure Audit & Redesign Blueprints

### Chapter 01: The Agentic Stack (`01_introduction`)
*Total Figures: 11 | Pass: 5 | Refinement: 1 | Needs V2: 5*

1. `fig-evolution-execution-units` (`evolution_execution_units.svg`) — **`[NEEDS V2]`**
   - *Current Flaw:* 4 horizontal card rows, 16 rectangles, 11 bullet points, zero lines/paths. Pure slide table.
   - *Textbook Redesign (`evolution_execution_units_v2.svg`):* Horizontal logarithmic time axis at top ($10^{-9}\text{ s} \to 10^{-6}\text{ s} \to 10^{-3}\text{ s} \to 10^0-10^3\text{ s}$). 4 concrete microarchitectural blocks: (1) CPU ALU + register file ($R_0..R_{31}$) + PC single-cycle clock; (2) OS Process MMU page table (CR3) + Ring 0/Ring 3 `task_struct` switch; (3) Distributed RPC socket buffer + protobuf wire framing + timeout timer; (4) Agent Control Block (ACB) + autoregressive forward pass + KV page table + tool sandbox boundary.
2. `fig-fault-models-fail-plausible` (`fault_models_fail_plausible.svg`) — **`[NEEDS V2]`**
   - *Current Flaw:* 3 vertical comparison cards with bullets and status badges ("STANDARD DEFENSE: EFFECTIVE").
   - *Textbook Redesign (`fault_models_fail_plausible_v2.svg`):* 2D phase-space trajectory corridor representing specification invariant envelope $\mathcal{I}_{\text{spec}}$. Three divergent trajectory paths: Path 1 (Fail-Stop) abruptly halts at $t_c$ with watchdog timeout; Path 2 (Byzantine) shoots outside corridor and is rejected at $2f+1$ quorum barrier; Path 3 (Fail-Plausible) stays smoothly inside syntactic corridor while breaching underlying semantic invariant.
3. `fig-request-scope-mismatches` (`request_scope_mismatches.svg`) — **`[NEEDS V2]`**
   - *Current Flaw:* 6 colored cards in $2 \times 3$ grid with 18 bullet points and warning icons.
   - *Textbook Redesign (`request_scope_mismatches_v2.svg`):* Hierarchical execution ladder showing Client Request $\to$ Agent Trajectory $\to$ Inference Steps $\to$ Tool RPCs. Six explicit boundary fault gates showing where timeouts, state leakage, budget overflows, and unvalidated writes occur.
4. `fig-empirical-triangulation-triangle` (`empirical_triangulation_triangle.svg`) — **`[NEEDS V2]`**
   - *Current Flaw:* Consulting-style triangle with 3 bulleted boxes at vertices and marketing claims ("Zero-Hallucination Barrier").
   - *Textbook Redesign (`empirical_triangulation_triangle_v2.svg`):* 3-way verification pipeline with concrete mathematical observables: Model Logits $P(\tau)$, Tool Return Code $\sigma_{\text{env}}$, and Deterministic Test Suite $\phi_{\text{spec}}$. Show comparator intersection logic $\mathcal{V} = P(\tau) \cap \sigma \cap \phi$.
5. `fig-tool-wait-memory-tax` (`tool_wait_memory_tax.svg`) — **`[NEEDS V2]`**
   - *Current Flaw:* Two horizontal bars with 8 floating text labels and 4 bullet cards.
   - *Textbook Redesign (`tool_wait_memory_tax_v2.svg`):* Dual-timeline execution trace: Timeline A (Active Compute, $T_{\text{model}} = 10\text{s}$) showing GPU SM utilization; Timeline B (Pinned Memory Footprint, $M_{\text{KV}} = 48\text{ GB}$) persisting across 30s tool wait ($T_{\text{tool}}$), quantifying stranded GPU DRAM bandwidth $\text{GB}\cdot\text{s}$.
6. `fig-von-neumann-agent-stack` (`von_neumann_agent_architecture.svg`) — **`[PASS - Volume Archetype]`**
   - Exemplary textbook architectural diagram establishing the Volume III design standard.

---

### Chapter 02: The Stochastic Processor (`02_processor`)
*Total Figures: 8 | Pass: 2 | Refinement: 4 | Needs V2: 2*

1. `fig-stochastic-processor-core` (`stochastic_processor_core.svg`) — **`[NEEDS V2]`**
   - *Current Flaw:* Conflates foundation model with CPU microarchitecture (instruction fetch/decode, ALU).
   - *Textbook Redesign (`stochastic_processor_core_v2.svg`):* Caller-facing stochastic processor contract. Input context buffer $\mathbf{c}$ entering transformer forward pass $\mathbf{z} = f_\theta(\mathbf{c})$, logit sampling simplex $\text{Softmax}(\mathbf{z}/T)$, stop sequence comparator, and tripartite output boundary: Proposed Action, Validated Effect, and Caller Status (`COMPLETED`, `TRUNCATED`, `REFUSAL`, `TRANSPORT_ERROR`).
2. `fig-vol3-processor-prefill-decode` (`prefill_vs_decode.svg`) — **`[NEEDS V2]`**
   - *Current Flaw:* High-contrast comparison cards with bullet points.
   - *Textbook Redesign (`prefill_vs_decode_v2.svg`):* Roofline Model coordinate plot showing Prefill (Compute-Bound GEMM, high arithmetic intensity $\text{FLOPs/byte} > 150$) vs. Decode (Memory-Bandwidth-Bound GEMV, low arithmetic intensity $< 2$) with hardware ridge point $\mathcal{I}^*$.

---

### Chapter 03: Test-Time Deliberation (`03_deliberation`)
*Total Figures: 8 | Pass: 1 | Refinement: 4 | Needs V2: 3*

1. `fig-vol3-deliberation-control-spectrum` (`control_spectrum.svg`) — **`[NEEDS V2]`**
   - *Current Flaw:* 4 horizontal colored cards with bullet lists.
   - *Textbook Redesign (`control_spectrum_v2.svg`):* 2D Tradeoff Coordinate Space: X-axis Deliberation Compute Budget ($N$ rollouts / tokens), Y-axis Task Complexity / State Cardinality. Shows Pareto frontiers for Greedy Decoding, Best-of-$N$, Beam Search, and MCTS with analytical error-decay bounds.
2. `fig-vol3-deliberation-anti-swarm` (`anti_swarm_supervisor.svg`) — **`[NEEDS V2]`**
   - *Current Flaw:* Card deck with bullet lists describing failure modes.
   - *Textbook Redesign (`anti_swarm_supervisor_v2.svg`):* Supervisor state machine monitoring agent fleet message bus. Shows message rate limiter, token velocity throttle, and circular delegation detector.
3. `fig-vol3-deliberation-circuit-breaker` (`circuit_breaker_fsm.svg`) — **`[NEEDS V2]`**
   - *Current Flaw:* Generic rounded cards with text labels.
   - *Textbook Redesign (`circuit_breaker_fsm_v2.svg`):* Formal Harel statechart with states `CLOSED`, `HALF-OPEN`, `OPEN`. Transitions labeled with guards ($N_{\text{fail}} \ge \theta$, $T_{\text{cool}} > \tau$) and entry/do/exit actions.

---

### Chapter 04: Context Working Sets (`04_context_working_sets`)
*Total Figures: 8 | Pass: 2 | Refinement: 4 | Needs V2: 2*

1. `fig-vol3-context-two-phase-cot` (`two_phase_cot_commit.svg`) — **`[NEEDS V2]`**
   - *Current Flaw:* Card carousel with bullets describing Scratchpad vs Working Memory.
   - *Textbook Redesign (`two_phase_cot_commit_v2.svg`):* Two-phase commit protocol datapath: Phase 1 (Speculative Deliberation in scratchpad buffer $\mathcal{B}_{\text{spec}}$), Phase 2 (Validation & Commit into canonical context $\mathcal{C}_{\text{canon}}$ with rollback upon invariant violation).
2. `fig-vol3-context-compaction-pipeline` (`context_compaction_pipeline.svg`) — **`[NEEDS V2]`**
   - *Current Flaw:* Text-heavy flow boxes with bullets.
   - *Textbook Redesign (`context_compaction_pipeline_v2.svg`):* Multi-stage pipeline showing raw token stream, syntactic token pruning (AST-aware), semantic summarization, and KV cache compaction with retention bounds.

---

### Chapter 05: Virtual Memory & Paged Attention (`05_virtual_memory`)
*Total Figures: 6 | Pass: 1 | Refinement: 3 | Needs V2: 2*

1. `fig-vol3-vmem-hierarchical-kv-topology` (`hierarchical_kv_paging_topology.svg`) — **`[NEEDS V2]`**
   - *Current Flaw:* 3 vertical bullet cards for L1/L2/L3 memory.
   - *Textbook Redesign (`hierarchical_kv_paging_topology_v2.svg`):* Three-tier memory hierarchy schematic: SRAM/HBM (GPU, 3.3 TB/s), Host DRAM (PCIe Gen5, 128 GB/s), NVMe SSD (8 GB/s) showing page table mapping, TLB miss traps, and asynchronous DMA migration.
2. `fig-vol3-vmem-prefill-decode-disagg` (`prefill_decode_disaggregation.svg`) — **`[NEEDS V2]`**
   - *Current Flaw:* 4 cards with ZERO connector paths.
   - *Textbook Redesign (`prefill_decode_disaggregation_v2.svg`):* Distributed cluster topology with Prefill Worker Pool (compute-optimized GPUs) transferring KV cache blocks via RDMA/RoCEv2 over InfiniBand to Decode Worker Pool (memory-bandwidth-optimized GPUs).

---

### Chapters 07–11: Systems Foundations (Part III)
*Total Figures: 31 | Pass: 4 | Refinement: 9 | Needs V2: 18*

1. `fig-vol3-checkpoint-fault-models` (`fault_models_fail_plausible.svg` in Ch 07) — **`[NEEDS V2]`**
   - *Textbook Redesign (`fault_models_fail_plausible_v2.svg`):* 3 structural topology panels: (A) Fail-Stop Crash ($S_t \to \bot$ with heartbeat timeout); (B) Byzantine Quorum ($3f+1$ with majority voting); (C) Fail-Plausible Semantic Defect ($\text{Cov}(e_i, e_j) \gg 0$ bypassing quorum).
2. `fig-vol3-virtualization-isolation-spectrum` (`isolation_spectrum.svg` in Ch 08) — **`[NEEDS V2]`**
   - *Current Flaw:* 108 text nodes, 20 bullets, zero paths. A pure text table masquerading as an SVG.
   - *Textbook Redesign (`isolation_spectrum_v2.svg`):* 4-tier hardware boundary schematic showing CPU rings, page tables, kernel namespaces, seccomp filters, and hypervisor EPT/VT-x virtualization with boundary crossing overheads ($\mu\text{s}$).
3. `fig-vol3-checkpoint-reversibility-boundary` (`reversibility_boundary.svg` in Ch 07) — **`[NEEDS V2]`**
   - *Textbook Redesign (`reversibility_boundary_v2.svg`):* Two-zone execution plane: Reversible In-Memory Zone (ephemeral filesystem, sandbox memory) vs Irreversible External Zone (production database, external API, customer email) separated by explicit 2PC Approval Barrier.
4. `fig-vol3-scheduling-drf` (`drf_multi_resource_agent_allocation.svg` in Ch 11) — **`[NEEDS V2]`**
   - *Current Flaw:* 3 bullet cards with zero connecting paths.
   - *Textbook Redesign (`drf_multi_resource_agent_allocation_v2.svg`):* 2D Dominant Resource Fairness coordinate plot (GPU Compute vs HBM Memory) showing resource capacities, agent demand vectors, and Pareto-optimal equal-share line.

---

### Chapters 12–14: Trajectory Synthesis & Reinforcement (Part IV)
*Total Figures: 14 | Pass: 1 | Refinement: 1 | Needs V2: 12*

1. `fig-vol3-flywheel-lifecycle` (`synthetic_data_flywheel.svg` in Ch 12) — **`[NEEDS V2]`**
   - *Current Flaw:* 6 rounded cards with bullet points in every box.
   - *Textbook Redesign (`synthetic_data_flywheel_v2.svg`):* Continuous closed-loop dataflow: Task Synthesizer $\to$ Rollout Fleet (Prefix Cache) $\to$ Kafka Event Log $\to$ Firecracker MicroVM Verification $\to$ Bifurcated Sieve ($R=1$ Golden vs $R=0$ DAgger) $\to$ Distributed Policy Compiler (FSDP) with weight return bus.
2. `fig-vol3-sft-context-compilation` (`context_compilation.svg` in Ch 13) — **`[NEEDS V2]`**
   - *Current Flaw:* "PANEL A/B" banners, "LOW/HIGH GOODPUT" badges, side bullet lists.
   - *Textbook Redesign (`context_compilation_v2.svg`):* Compiler pipeline schematic: Uncompiled Trajectory AST $\to$ Static Analysis Pass (prune failed exploratory tool calls) $\to$ Loss-Masking Generator $\to$ Packed Sequence Formatter with block diagonal attention mask.
3. `fig-vol3-rlvr-vs-rlhf` (`rlvr_vs_rlhf.svg` in Ch 14) — **`[NEEDS V2]`**
   - *Current Flaw:* 2-column pitch deck slide comparing RLHF vs RLVR with badges and bullets.
   - *Textbook Redesign (`rlvr_vs_rlhf_v2.svg`):* Dual control-loop diagrams: (A) RLHF with subjective human/reward-model proxy $\hat{r}_\psi$ (noisy, non-stationary, reward hacking); (B) RLVR with deterministic ground-truth execution verifier $\mathcal{V}(y) \in \{0, 1\}$ (unit tests, formal proofs, compiler syntax).
4. `fig-vol3-rlvr-prm-orm` (`prm_vs_orm.svg` in Ch 14) — **`[NEEDS V2]`**
   - *Current Flaw:* Bulleted step cards and pros/cons boxes.
   - *Textbook Redesign (`prm_vs_orm_v2.svg`):* Sutton-Barto style tree backup diagram comparing Outcome Reward (sparse reward at leaf $R(s_T)$) vs Process Reward (dense step rewards $r(s_t, a_t)$ at every reasoning node with credit assignment).

---

### Chapters 15–18 & Appendices: Consensus, Observability, Tokenomics & Capstone (Part V & VI)
*Total Figures: 12 | Pass: 0 | Refinement: 1 | Needs V2: 11*

1. `fig-vol3-multiagent-topologies` (`multi_agent_topologies.svg` in Ch 15) — **`[NEEDS V2]`**
   - *Current Flaw:* 4 cards with 60% bullet points and verdict banners.
   - *Textbook Redesign (`multi_agent_topologies_v2.svg`):* 4 network topology schematics: (A) Pipeline/Chain ($A \to B \to C$); (B) Star/Central Supervisor; (C) Hierarchical Tree; (D) Fully Connected Mesh with message complexity annotations ($O(N)$, $O(N^2)$).
2. `fig-vol3-observability-eval-gym` (`hermetic_eval_gym_architecture.svg` in Ch 16) — **`[NEEDS V2]`**
   - *Current Flaw:* 3 vertical bullet cards; only 2 connector lines.
   - *Textbook Redesign (`hermetic_eval_gym_architecture_v2.svg`):* Hardware/OS virtualization schematic: Host OS $\to$ KVM Hypervisor $\to$ Guest MicroVM Enclave containing Agent Runtime, Read-Only rootfs, Ephemeral overlayfs, TAP network filter, and Deterministic PRNG seed tap.
3. `fig-vol3-observability-autonomy-spectrum` (`progressive_autonomy_spectrum.svg` in Ch 16) — **`[NEEDS V2]`**
   - *Current Flaw:* 100% text table with quote box; does not show canary pipeline described in caption.
   - *Textbook Redesign (`progressive_autonomy_spectrum_v2.svg`):* 4-stage canary deployment pipeline: Stage 0 (Shadow / Read-Only) $\to$ Stage 1 (Human-in-the-Loop Approval Gate) $\to$ Stage 2 (Sandboxed Canary with automated rollback) $\to$ Stage 3 (Autonomous Production with invariant monitoring).
4. `fig-vol3-tokenomics-cascade-routing` (`multi_agent_token_cascade_routing.svg` in Ch 17) — **`[NEEDS V2]`**
   - *Current Flaw:* Pitch deck slide with "FAILURE MODE" vs "SOLUTION", "CRITICAL EXHAUSTION" badges.
   - *Textbook Redesign (`multi_agent_token_cascade_routing_v2.svg`):* FrugalGPT cascade routing state machine: Request $\to$ Fast Small Model ($M_{\text{small}}$) $\to$ Confidence Scorer $\tau_{\text{conf}}$ $\xrightarrow{<\theta}$ Medium Model ($M_{\text{med}}$) $\xrightarrow{<\theta'}$ Frontier Model ($M_{\text{large}}$). Quantifies cost and latency at each tier.
5. `fig-vol3-tokenomics-memory-caching` (`tokenomics_memory_hierarchy_caching.svg` in Ch 17) — **`[NEEDS V2]`**
   - *Current Flaw:* Severe text mismatch: caption explicitly specifies an MLFQ scheduler with priority queues $Q_0, Q_1, Q_2$, but diagram has zero queues.
   - *Textbook Redesign (`tokenomics_memory_hierarchy_caching_v2.svg`):* Multi-Level Feedback Queue (MLFQ) scheduler with 3 explicit FIFO queues ($Q_0$ High Priority / Short Burst, $Q_1$ Medium, $Q_2$ Batch Trajectory) mapped to tiered memory (GPU HBM, Host DRAM, NVMe).
6. `fig-vol3-conclusion-capstone` (`five_part_agent_runtime_architecture.svg` in Ch 18) — **`[NEEDS V2]`**
   - *Current Flaw:* 5 stacked cards with bullets; no architecture, no buses, no data paths.
   - *Textbook Redesign (`five_part_agent_runtime_architecture_v2.svg`):* Capstone Stochastic Computer System Architecture: 6 interconnected subsystems linked by a unified System Bus: (1) Stochastic Processor Core; (2) Context Working Set MMU; (3) State & Checkpointing Saga Ledger; (4) Actuation Sandbox Hypervisor; (5) Interrupt & Preemption Controller; (6) Multi-Agent Interconnect Bus.
7. `fig-vol3-conclusion-ladder` (`autonomous_improvement_ladder.svg` in Ch 18) — **`[NEEDS V2]`**
   - *Current Flaw:* Stacked colored cards with "RUNG 1–5", marketing badges, and quote box.
   - *Textbook Redesign (`autonomous_improvement_ladder_v2.svg`):* Hierarchical closed-loop control system: Inner Loop (In-Context Trajectory Correction, seconds) $\to$ Middle Loop (Replay & Synthetic SFT, hours) $\to$ Outer Loop (Verifiable Reinforcement RLVR, days) $\to$ Meta Loop (Architectural & Invariant Synthesis, weeks).

---

## 5. Master Implementation Roadmap

```
Phase 1: Priority V2 Redesigns (Immediate High-Impact & Severe Text Mismatches)
  - tokenomics_memory_hierarchy_caching_v2.svg (Ch 17: MLFQ queues + tiered memory)
  - progressive_autonomy_spectrum_v2.svg (Ch 16: 4-stage canary deployment pipeline)
  - five_part_agent_runtime_architecture_v2.svg (Ch 18: Stochastic Computer 6-subsystem capstone)
  - multi_agent_token_cascade_routing_v2.svg (Ch 17: FrugalGPT cascade statechart)
  - hermetic_eval_gym_architecture_v2.svg (Ch 16: OS virtualization & enclave datapath)
  - autonomous_improvement_ladder_v2.svg (Ch 18: Multi-tier closed-loop feedback hierarchy)
  - prefill_decode_disaggregation_v2.svg (Ch 11: Distributed cluster topology + rooflines)

Phase 2: Core System Architecture Overhauls (Chapters 01, 02, 07, 08, 12, 14)
  - evolution_execution_units_v2.svg (Ch 01)
  - fault_models_fail_plausible_v2.svg (Ch 01 & Ch 07)
  - stochastic_processor_core_v2.svg (Ch 02)
  - isolation_spectrum_v2.svg (Ch 08)
  - synthetic_data_flywheel_v2.svg (Ch 12)
  - rlvr_vs_rlhf_v2.svg (Ch 14)

Phase 3: Minor Refinements & Badge Stripping (29 figures)
  - Strip embedded bottom banners ("KEY ARCHITECTURAL INVARIANT")
  - Remove presentation verdict badges ("VERDICT: PRODUCTION STANDARD")
  - Convert bullet points inside state boxes to clean component labels
  - Consolidate duplicate figures across adjacent chapters
```
