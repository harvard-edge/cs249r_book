# Chapter 13 Micro Audit: Placement

- **Part**: Part III: Running
- **Core Thesis**: Logical boundaries between neural policies and deterministic safety enforcers collapse on physical silicon unless contention for shared resources like power rails, clock trees, and memory buses is rigorously measured and budgeted under worst-case concurrent loads.
- **Audited At**: `2026-09-19T15:23:56.137511`

## Pedagogical Strengths
- 🟢 Successfully grounds abstract software architecture concepts in concrete physical constraints like voltage drops and thermal throttling.
- 🟢 Provides a clear, actionable checklist (the Placement Deliverable Quintet) for verifying physical isolation.

## Student Cohort Friction Points
- **Priya**: The abrupt mention of '12 percent channel availability required for high-temperature DRAM refresh cycles' assumes prior knowledge of how thermal states degrade DRAM retention times.
- **Elena**: The transition into 'Windowed watchdog timers and execution liveness' breaks the chapter's focus on shared spatial resources by abruptly pivoting to temporal execution monitoring.

## Established Budgets & Invariants
- ⚖️ `Loaded enforcement latency bound ($P_{99.99}$) under concurrent maximum-bandwidth DMA transfers.`
- ⚖️ `Power rail transient limits ($L \, dI/dt$) to prevent voltage drops and subsequent DVFS clock throttling.`

## Conceptual Continuity
### Imported Prerequisites
- ↰ Runtime safety filters and Control Barrier Functions (CBFs).
- ↰ Lock-free seqlocks and monotonic version counters for cross-boundary communication.
### Exported Downstream Concepts
- ↳ The Placement Deliverable Quintet for evaluating physical silicon interference.
- ↳ Common-mode failure vectors in heterogeneous SoCs (shared PLLs, reset lines, heat sinks).

## Thematic Threads for Part Synthesis
- 🧵 How does the system arbitrate operational authority and safely manage human intervention when physical reality diverges from the fully integrated, measured autonomous pipeline?
