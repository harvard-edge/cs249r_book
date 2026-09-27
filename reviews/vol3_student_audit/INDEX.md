# Volume III: Agentic Machine Learning Systems — Student Accessibility Review Index

Auditing Volume III from the perspective of a **senior undergraduate, entry-level master's student, or first- to second-year PhD** in machine learning.

## Chapter Progress & Review Status

| Part | Ch # | Chapter Title | Source Path | Audit Report | Status | P0 | P1 | P2 | Key Friction Point |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Intro** | 01 | Introduction: The Stochastic Computer | [`01_introduction.qmd`](../../books/vol3/01_introduction/01_introduction.qmd) | [`01_introduction.md`](01_introduction.md) | 🟢 Audited | 0 | 2 | 1 | Systems jargon avalanche in ACB; ungrounded PFLOP math |
| **I: Model** | 02 | The Foundation Model Engine | [`02_foundation_model.qmd`](../../books/vol3/02_foundation_model/02_foundation_model.qmd) | [`02_foundation_model.md`](02_foundation_model.md) | 🟢 Audited | 0 | 3 | 2 | PDA grammar tuple notation without stack intuition; MoE arithmetic intensity jump |
| | 03 | Test-Time Compute & Deliberation | [`03_test_time_compute.qmd`](../../books/vol3/03_test_time_compute/03_test_time_compute.qmd) | [`03_test_time_compute.md`](03_test_time_compute.md) | 🟢 Audited | 0 | 4 | 3 | Table pipe math syntax bug; quadratic PRM prefill penalty unstated; MRTS Cobb-Douglas bridge |
| **II: Memory** | 04 | Context Engineering & Working Sets | [`04_context_engineering.qmd`](../../books/vol3/04_context_engineering/04_context_engineering.qmd) | [`04_context_engineering.md`](04_context_engineering.md) | 🟢 Audited | 0 | 3 | 2 | Linear FLOP 24 multiplier derivation; CAS digest lifecycle timing; tombstone suppression |
| | 05 | KV-Cache Management & Virtual Paging | [`05_kv_cache.qmd`](../../books/vol3/05_kv_cache/05_kv_cache.qmd) | [`05_kv_cache.md`](05_kv_cache.md) | 🟢 Audited | 0 | 4 | 2 | MLA footprint B vs KiB unit typo; calculate_cache_utilization undefined seq_lens; 192 vs 195 concurrency; missing KV dimension in RDMA equation |
| | 06 | Long-Term Memory & Episodic Retrieval | [`06_long_term_memory.qmd`](../../books/vol3/06_long_term_memory/06_long_term_memory.qmd) | [`06_long_term_memory.md`](06_long_term_memory.md) | 🟢 Audited | 0 | 3 | 2 | BFS function count 1310 vs 1311 code-prose mismatch; unstated ADC lookup equation in IVF-PQ; cross-encoder latency budget trade-off |
| **III: Tools** | 07 | Tool Calling & Typed Interfaces | [`07_tool_calling.qmd`](../../books/vol3/07_tool_calling/07_tool_calling.qmd) | [`07_tool_calling.md`](07_tool_calling.md) | 🟢 Audited | 0 | 4 | 3 | Prefix cache continuity across dynamic Slot 3 turns; epoll reactor incomplete read in EPOLLET; formal probe contract for legacy tools |
| | 08 | Sandboxes & Isolation Boundaries | [`08_sandboxes.qmd`](../../books/vol3/08_sandboxes/08_sandboxes.qmd) | [`08_sandboxes.md`](08_sandboxes.md) | 🟢 Audited | 0 | 3 | 3 | EPT 2D page walk derivation intuition; bridging capability tokens to Linux kernel LSMs; duplicate GPU preemption footnote |
| | 09 | Security & Capability Attenuation | [`09_security.qmd`](../../books/vol3/09_security/09_security.qmd) | [`09_security.md`](09_security.md) | 🟢 Audited | 0 | 2 | 2 | Autoregressive string taint boundary; blob escrow lease lifecycle; alphabet entropy normalization |
| **IV: Runtime** | 10 | The Agent Harness & Control Plane | [`10_agent_harness.qmd`](../../books/vol3/10_agent_harness/10_agent_harness.qmd) | [`10_agent_harness.md`](10_agent_harness.md) | 🟢 Audited | 0 | 3 | 2 | 256-byte C ACB to Python runtime bridge; datapath phase enum naming (STAGING vs INGESTION); DRR quantum sizing intuition |
| | 11 | Durable Execution & State Persistence | [`11_durable_execution.qmd`](../../books/vol3/11_durable_execution/11_durable_execution.qmd) | [`11_durable_execution.md`](11_durable_execution.md) | 🟢 Audited | 0 | 2 | 2 | Decoupling dry replay from workspace filesystem snapshots; prefix cache survival across worker failovers |
| | 12 | Failure Recovery & Sagas | [`12_failure_recovery.qmd`](../../books/vol3/12_failure_recovery/12_failure_recovery.qmd) | `12_failure_recovery.md` | 🟡 Queued | - | - | - | - |
| | 13 | Agent Evaluation & Benchmarking | [`13_evaluation.qmd`](../../books/vol3/13_evaluation/13_evaluation.qmd) | `13_evaluation.md` | 🟡 Queued | - | - | - | - |
| **V: Learning** | 14 | Trajectory Curation & Filtering | [`14_trajectory_curation.qmd`](../../books/vol3/14_trajectory_curation/14_trajectory_curation.qmd) | `14_trajectory_curation.md` | 🟡 Queued | - | - | - | - |
| | 15 | Trajectory Fine-Tuning & Distillation | [`15_fine_tuning.qmd`](../../books/vol3/15_fine_tuning/15_fine_tuning.qmd) | `15_fine_tuning.md` | 🟡 Queued | - | - | - | - |
| | 16 | Verifiable RL (RLVR / GRPO) | [`16_rlvr.qmd`](../../books/vol3/16_rlvr/16_rlvr.qmd) | `16_rlvr.md` | 🟡 Queued | - | - | - | - |
| **VI: Scale** | 17 | Multi-Agent Systems & Coordination | [`17_multi_agent.qmd`](../../books/vol3/17_multi_agent/17_multi_agent.qmd) | `17_multi_agent.md` | 🟡 Queued | - | - | - | - |
| | 18 | Agent Economics & Tokenomics | [`18_agent_economics.qmd`](../../books/vol3/18_agent_economics/18_agent_economics.qmd) | `18_agent_economics.md` | 🟡 Queued | - | - | - | - |
| **VII: Synthesis** | 19 | Conclusion & The Unified Trajectory | [`19_conclusion.qmd`](../../books/vol3/19_conclusion/19_conclusion.qmd) | `19_conclusion.md` | 🟡 Queued | - | - | - | - |

## Special Tasks
- 🎨 **Stochastic Computer TikZ Diagram**: 🟢 Completed & Visually Verified (`books/vol3/tex/stochastic_computer_architecture.tex`, rendered to SVG/PNG/PDF and checked against Volume 1 aesthetic standards).
- 📊 **Figure Opportunities Registry**: Tracking visual aids in [`FIGURE_OPPORTUNITIES.md`](FIGURE_OPPORTUNITIES.md).
