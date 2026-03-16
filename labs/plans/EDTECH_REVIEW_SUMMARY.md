---
title: "Ed-Tech Review Summary"
---

# Ed-Tech Review Summary — Lab Proposals

Generated: 2026-03-15
Source: 4 education-reviewer agents analyzing all 33 labs

---

## Critical Issues (Must Fix)

### 1. Labs with Zero mlsysim Grounding
These labs have no connection to Engine.solve() or the hardware/model registries:

| Lab | Issue | Recommendation |
|-----|-------|----------------|
| **V1-03 (ML Workflow)** | All 4 parts are qualitative/narrative | Restructure: use Engine.solve() to show model infeasibility across deployment targets |
| **V1-14 (ML Ops)** | All 5 parts are domain-specific formulas | Add Engine.solve() to Part B (compute retraining cost from real hardware) |
| **V2-08 (Fleet Orchestration)** | All 5 parts need custom simulators | Reduce to 3-4 parts; add mlsysim grounding via Fabrics bandwidth constants |

### 2. Redundant Concepts Across Labs
| Concept | Appears In | Resolution |
|---------|-----------|------------|
| Alpha-beta model | V2-03 Part A, V2-06 Part A | **Merge V2-03 and V2-06** into one "Communication at Scale" lab |
| Bandwidth hierarchy (NVLink-IB cliff) | V2-02 Part C, V2-03 Part C, V2-06 Part C | Assign to one lab only |
| Reliability collapse (P^N) | V2-01 Part A, V2-07 Part A | Shorten V2-07 Part A to 5-min recall |
| Amdahl + communication | V2-01 Part D, V2-05 Part D | V2-05 should explicitly build on V2-01 |
| Amdahl's Law | V1-12 Part A, V1-16 Part D | V1-16 should reference V1-12, not re-teach |
| Precision effects | V1-10 Part A, V1-13 Part E | **Drop V1-13 Part E** (redundant with Lab 10) |
| Energy/carbon | V1-10 Part D, V1-15 Part E | Consider merging or differentiating angles |

### 3. Parts That Should Be Dropped or Replaced
| Lab | Part | Issue | Action |
|-----|------|-------|--------|
| V1-01 | Part C (Silent Decay) | Ungrounded; belongs in Lab 14 | Replace with Engine.solve()-based triad diagnosis |
| V1-09 | Part C (Curriculum Learning) | Zero systems content | Replace with "Preprocessing Tax" using Engine.solve() |
| V1-13 | Part E (Precision Mirage) | Duplicates Lab 10 Part A | Drop entirely |
| V1-15 | Part C (TCO) | Not specific to responsible engineering | Move to Lab 12 or 13 |

### 4. Time Budget Overruns
Every 5-part lab exceeds 60 minutes at proposed durations. Fix: trim 15-min parts to 12 min.

| Lab | Proposed Time | Target | Fix |
|-----|--------------|--------|-----|
| V1-13 | 63 min | 60 min | Drop Part E, trim Part B to 12 min |
| V1-14 | 63 min | 60 min | Merge Parts D+E into one 12-min part |
| V1-15 | 63 min | 60 min | Drop Part C (TCO), keep A+B+D+E = 48 min |
| V1-16 | 63 min | 60 min | Trim Parts B(12) and C(10), keep D+E |

---

## Structural Recommendations

### V2 Merger: Labs V2-03 + V2-06 → "Communication at Scale"
Proposed structure:
- Part A: Alpha-beta model (network time budget)
- Part B: Ring vs Tree AllReduce (algorithm choice)
- Part C: Topology + hierarchy effects (fat-tree, rail-optimized)
- Part D: Communication budget optimization
- Part E: Design challenge

### V1 Lab 01: Restructure Part C
Replace "Silent Decay" with "The Triad Across Targets" — use Engine.solve() with same model on Cloud/Edge/Tiny to diagnose which DAM axis is binding per target. Builds directly on Part B and feeds into Part D.

### V1 Lab 03: Add mlsysim Grounding
Part A: Use Engine.solve() to show ResNet-50 is infeasible on tablet (OOM), demonstrating constraint discovery cost.
Part B: Use Engine.sweep() to show how many configs must be evaluated, grounding iteration velocity in compute time.

---

## Strongest Parts (Protect These)

| Lab | Part | Why It's Great |
|-----|------|---------------|
| V1-08 Part A | Memory Budget Shock | mlsysim's training_memory() was built for this |
| V1-06 Part D | Workload Signatures | Engine.solve() across all models = elegant |
| V1-10 Part B | Pruning Hardware Trap | Best "wrong prediction" moment in the suite |
| V1-11 A-E | Full Roofline Sequence | Best-structured progression overall |
| V1-16 Part E | Constraint Cascade | Embodies the book's thesis perfectly |
| V2-04 Full | Data Storage | Best story arc in Vol 2 |
| V2-05 Part B | ZeRO Memory | Engine.solve() showcase with zero_stage parameter |
| V2-07 Part B | Young-Daly Sweet Spot | U-shaped cost curve is pedagogically powerful |
| V2-09 Full | Inference at Scale | Best-structured Vol 2 lab |

---

## mlsysim Gaps to Fill

Features needed in mlsysim to support the labs:

1. **`kv_cache_size(seq_len, batch_size, precision)`** on TransformerWorkload — needed for V1-13 Part C, V2-09 Part C
2. **Thermal throttling model** — needed for V1-12 Part B (could be a simple piecewise model)
3. **Per-kernel decomposition** — Engine.solve() gives total latency but not per-operation breakdown (needed for V1-07 Part B fusion analysis)
4. **Latency distribution model** — Engine.solve() is deterministic; p99 analysis needs noise model (V1-12 Part D, V1-13 Part A)

---

## Vol 2 Labs 10-17: Additional Findings

### Strong Labs
| Lab | Assessment | Highlight |
|-----|-----------|-----------|
| V2-10 (Performance Eng) | Strong | Best diagnostic progression: roofline → fusion → FlashAttention → precision → playbook |
| V2-14 (Robust AI) | Strong | "Guardrails beat universal hardening" is counterintuitive and actionable |
| V2-15 (Carbon Budget) | Strong | Jevons Paradox (Part D) is the standout moment of Vol 2 |
| V2-16 (Fairness Budget) | Strong | Impossibility theorem foundation is mathematically rigorous |

### Labs Needing Work
| Lab | Issue | Fix |
|-----|-------|-----|
| V2-13 (Security/Privacy) | Tries to cover 4 topics; Part C (model extraction) breaks arc | Drop Part C, restructure around "cost of privacy" thread |
| V2-17 (Capstone) | Overloaded; Part E depends on V2-15/V2-16 Ledger values | Cut to 4 parts; provide default values for cross-lab dependencies |

### Additional Redundancies Found
| Concept | Appears In | Resolution |
|---------|-----------|------------|
| PSI drift detection | V2-12 Part A, V2-14 Part C | V2-14 owns PSI; V2-12 references it |
| Roofline model | V1-11 Part A, V2-02 Part B, V2-10 Part A | V2-10 should use fleet-scale workloads to differentiate |

### Implementation Complexity Flags
| Lab | Part | Issue |
|-----|------|-------|
| V2-11 Parts D-E | Federated learning simulator | ~200-400 lines custom code needed |
| V2-16 Part C | Iterative feedback loop simulation | Complex state management |
| V2-10 Part B | Per-operator fusion toggle | Significant UI work in Marimo |
| V2-17 Part E | Cross-lab Design Ledger consumption | Fragile dependency chain |

### Suggested mlsysim Module Additions
5. **`mlsysim/sim/federated.py`** — Parametric FL convergence model for V2-11
6. **`mlsysim/sim/policy.py`** — Shared cost/alert/fairness calculations for V2-12, V2-13, V2-14, V2-16
7. **`SynthesisSolver` coupling matrix** — For V2-17 principle interactions

---

## Overall Assessment

| Metric | Score |
|--------|-------|
| **Pedagogical coherence** | 8/10 — most labs tell strong stories; V1-03, V1-09, V2-13 need work |
| **mlsysim grounding** | 6/10 — varies wildly; some labs are excellent, some have zero connection |
| **Prediction calibration** | 9/10 — consistently well-designed "wrong prediction" moments |
| **Progressive difficulty** | 8/10 — within-lab progression is generally strong; some jarring transitions |
| **Engagement** | 8/10 — failure states and dramatic reveals keep interest; Jevons Paradox is the highlight |
| **Time management** | 5/10 — every 5-part lab overruns; systematic trimming needed |
| **Implementation feasibility** | 6/10 — ~40% of parts need significant custom code beyond mlsysim |

---

## Recommended Action Plan (Priority Order)

### Phase 1: Structural Fixes (before building)
1. Merge V2-03 + V2-06 into "Communication at Scale"
2. Drop/replace: V1-01 Part C, V1-09 Part C, V1-13 Part E, V1-15 Part C
3. Resolve all redundancies (assign concept ownership to one lab each)
4. Trim all 15-min parts to 12 min across the board

### Phase 2: mlsysim Extensions
1. Add `kv_cache_size()` to TransformerWorkload
2. Add `mlsysim/sim/federated.py` for V2-11
3. Add `mlsysim/sim/policy.py` for ops/fairness labs

### Phase 3: Build Labs (in priority order)
1. **Tier 1** (best mlsysim grounding, build first): V1-08, V1-11, V1-06, V2-05, V2-09, V2-04
2. **Tier 2** (partially grounded): V1-01, V1-02, V1-04, V1-10, V1-12, V2-01, V2-02, V2-07, V2-10
3. **Tier 3** (mostly custom): V1-03, V1-07, V1-09, V1-13, V1-14, V1-15, V1-16, V2-08, V2-11-V2-17
