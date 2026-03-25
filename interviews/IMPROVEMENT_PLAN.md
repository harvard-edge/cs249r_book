# StaffML Vault Iterative Improvement Plan

**Date**: 2026-03-24 (updated late night)
**Corpus**: 5,228 questions across 5 tracks
**Goal**: Balanced, validated, deeply-linked interview vault

## Iteration 1 Completion Status

| Phase | Status | Key Results |
|-------|--------|-------------|
| 1A. Topic Consolidation | DONE | 729 questions redistributed from 4 overloaded concepts |
| 1B. WARN Fixes | PARTIAL | 29 Opus agents did full math recomputation (2 rounds) |
| 1C. Solution Enrichment | IN PROGRESS | 4 agents writing 328 enriched solutions |
| 2. Bloom Rebalancing | NOT STARTED | L6 counts better than expected (181 cloud, 99 edge) |
| 3. Chain Building | NOT STARTED | Still at 34% coverage |
| 4. Taxonomy Audit | DONE | 438 empty→0, 224 missing→0, 141 orphans→94 |
| 5. Final Validation | NOT STARTED | Awaiting enrichment completion |

---

## Philosophy: The Feedback Loop

This plan operates as an **iterative feedback loop**, not a waterfall. Each cycle:

```
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │   MEASURE → DIAGNOSE → FIX → VALIDATE → MEASURE    │
  │       ↑                                     │       │
  │       └─────────────────────────────────────┘       │
  │                                                     │
  └─────────────────────────────────────────────────────┘
```

**MEASURE**: Compute the balance scorecard (Bloom, reasoning mode, competency, topic concentration, chain coverage)
**DIAGNOSE**: Identify the worst imbalances per track
**FIX**: Generate new questions, consolidate topics, enrich solutions, build chains
**VALIDATE**: Run Opus validation on all changes
**MEASURE**: Recompute scorecard, compare to previous iteration

Each cycle should improve the overall balance score. Stop when all metrics hit target thresholds.

---

## Current Balance Scorecard (Baseline)

### Per-Track Health

| Metric | Cloud | Edge | Mobile | TinyML | Global | Target |
|--------|-------|------|--------|--------|--------|--------|
| **Count** | 2,510 | 937 | 802 | 779 | 200 | — |
| **Bloom deviation** | 4.8% | 4.3% | 4.5% | 5.6% | 7.2% | <3% |
| **Reasoning mode CV** | 0.49 | 0.55 | 0.56 | 0.62 | 0.57 | <0.35 |
| **Competency CV** | 0.37 | 0.40 | 0.50 | 0.54 | 0.49 | <0.30 |
| **Calc format %** | 26% | 24% | 24% | 25% | 14% | 30-40% |
| **Topic HHI** | 0.005 | 0.034 | 0.038 | 0.048 | 0.012 | <0.02 |
| **Chain coverage** | ~40% | ~5% | ~4% | ~4% | 0% | >50% |
| **L6 (Create) count** | 0 | 60 | 0 | 0 | 0 | 50+ each |
| **OK rate** | — | — | — | — | — | >85% |
| **Short solutions** | — | — | — | — | — | <2% |

### Critical Imbalances Diagnosed

**Cloud**: Overpopulated at L5/evaluate (26% vs 15% ideal). Napkin-math dominates reasoning (654 vs 83 optimization-task). 746 canonical topics is absurdly fragmented — should be ~80-100. Precision (75) and parallelism (119) competencies are weak relative to memory (276) and data (298).

**Edge**: Concept-recall dominates reasoning (254 vs 46 optimization-task). Analyze level is low (18% vs 25% ideal). Parallelism (18) competency is nearly absent. 157 canonical topics is reasonable but still fragmented.

**Mobile**: Similar reasoning imbalance to edge. Analyze (17%) and evaluate (12%) are low. Parallelism (16) nearly absent. Solutions are shortest (p10=77 chars). Only 98 canonical topics — tighter but missing depth in some areas.

**TinyML**: Highest reasoning mode CV (0.62) — most imbalanced. Analyze at 14% (worst of all tracks). Only 6 parallelism questions. Topic concentration highest (HHI=0.048) — too clustered around federated-learning (67), quantization (66), roofline (64).

**Global**: Weakest overall. Only 200 questions. Apply has 12 questions (6%!). Zero L6. Low calculation format (14%). Should be the universal foundation layer — currently it's a stub.

---

## Iteration 1: Taxonomy Consolidation + WARN Fixes

### 1A. Topic Consolidation (per track)

The #1 structural problem. Cloud has 958 raw topics mapping to 746 canonical_topics. This makes coverage analysis, chain building, and gap detection unreliable.

**Process** (per track, in parallel):
1. Extract all unique `topic` and `canonical_topic` values with counts
2. Use Opus to cluster into 60-100 canonical topics per track (cloud may need 100-120)
3. Define canonical topic → knowledge_area → competency_area mapping
4. Remap all questions to consolidated canonical_topics
5. Validate: no topic has <3 questions, no topic has >80 questions

**Merge rules**:
- `kv-cache-cost`, `kv-cache-sizing`, `kv-cache-vram`, `kv-cache-memory` → `kv-cache`
- `federated-learning-economics`, `federated-learning-tco` → `federated-learning`
- `gpu-roofline`, `roofline`, `mobile-roofline-analysis`, `tinyml-roofline` → `roofline` (per-track differentiation via track field, not topic)
- `deployment`, `model-deployment`, `edge-deployment` → `deployment`

**Target**: Cloud ≤120, Edge ≤80, Mobile ≤70, TinyML ≤60, Global ≤50 canonical topics.

### 1B. WARN Fixes (parallel with 1A)

1,964 WARN questions — mostly incomplete napkin_math stubs.

**Process**:
1. Split WARNs by track
2. Further split into batches of 25
3. Launch parallel Opus agents to complete napkin_math with physics-grounded calculations
4. Use reference specs: A100 (80GB HBM2e, 2TB/s, 312 TFLOPS FP16), H100 (80GB HBM3, 3.35TB/s, 989 TFLOPS), etc.
5. Validate with Flash, confirm with Pro

**Target**: Reduce WARNs from 1,964 to <500.

### 1C. Solution Enrichment

329 questions have solutions under 100 chars.

**Process**:
1. Extract all short-solution questions
2. Batch by track (25 per batch)
3. Opus enriches each solution to 200-500 chars with reasoning chain
4. Validate consistency with scenario

**Target**: Zero solutions under 100 chars.

### Measure after Iteration 1:
```bash
python3 -c "
import json; c=json.load(open('corpus.json'))
# Recompute: canonical_topic counts, WARN rate, solution length distribution
"
```

---

## Iteration 2: Bloom + Reasoning Rebalancing (per track)

### Per-Track Generation Targets

Based on current imbalances, generate questions to fill gaps:

**Cloud** (2,510 → ~2,700):
- L1 remember: +30 (foundations that L3+ builds on)
- L6 create: +80 (system design questions)
- Reasoning: +60 optimization-task, +40 symptom-to-cause
- Competency: +40 precision, +30 parallelism

**Edge** (937 → ~1,100):
- L4 analyze: +50 (diagnostic scenarios)
- L6 create: +20 more (have 60, want 80)
- Reasoning: +40 optimization-task, +30 failure-to-root-cause
- Competency: +30 parallelism, +20 networking

**Mobile** (802 → ~1,000):
- L4 analyze: +50 (root cause analysis)
- L5 evaluate: +30 (trade-off evaluation)
- L6 create: +60 (system design)
- Reasoning: +40 optimization-task, +30 failure-to-root-cause
- Competency: +30 parallelism, +20 data

**TinyML** (779 → ~950):
- L4 analyze: +60 (biggest gap: 14% vs 25% ideal)
- L5 evaluate: +30
- L6 create: +50
- Reasoning: +40 optimization-task, +30 tradeoff-analysis
- Competency: +30 parallelism, +20 networking

**Global** (200 → ~350):
- L3 apply: +40 (only 12 currently!)
- L5 evaluate: +20
- L6 create: +30
- Reasoning: +20 optimization-task, +15 napkin-math
- Format: +30 calculation (only 14% currently)

### Generation Protocol

For each batch:
1. Specify exact track + level + reasoning_mode + competency_area + canonical_topic
2. Generate with Opus (high quality, expensive)
3. Validate with Flash (fast, cheap)
4. Confirm with Pro (second opinion)
5. Dedup against existing corpus (threshold 0.85)

### Measure after Iteration 2:
- Recompute Bloom deviation per track (target: <3%)
- Recompute reasoning mode CV per track (target: <0.35)
- Recompute competency CV per track (target: <0.30)

---

## Iteration 3: Chain Building + Cross-Track Linking

### Chain Expansion

Current: 533 chains covering 34% of corpus. Cloud has 424, others have ~30 each.

**Process**:
1. For each consolidated canonical_topic, identify questions spanning L1→L6
2. Build chains of 3-6 questions progressing through Bloom levels
3. Prioritize: edge (48→150), mobile (32→120), tinyml (29→100)
4. Backpopulate chain_ids into corpus.json

**Target**: 60%+ chain coverage across all tracks.

### Cross-Track Concept Bridges

66 topics appear in 3+ tracks. These are the cross-cutting fundamentals.

**Process**:
1. For each cross-track topic (memory-hierarchy, roofline, quantization, etc.)
2. Verify each track has questions at L1-L2 (foundations) through L5-L6 (advanced)
3. Build "cross-track chains" that show how the same concept manifests differently:
   - `roofline-cloud`: datacenter GPU ridge points, batch size optimization
   - `roofline-edge`: Jetson Orin compute vs memory bound analysis
   - `roofline-mobile`: mobile GPU thermal throttling impact on roofline
   - `roofline-tinyml`: MCU arithmetic intensity with INT8 ops

### Measure after Iteration 3:
- Chain coverage % per track (target: >50%)
- Cross-track bridge count (target: 30+ topics with full L1-L5 coverage across 3+ tracks)

---

## Iteration 4: Intra-Track Taxonomy Audit

Deep dive into each track's internal taxonomy coherence.

### Per-Track Audit (run in parallel)

For each track, an Opus agent:

1. **Concept Graph Validation**: Every taxonomy_concept used in questions should exist in taxonomy.json. Every concept in taxonomy.json should have ≥3 questions.
2. **Prerequisite Chain Integrity**: If concept A requires B, questions on A should not appear at lower Bloom levels than questions on B.
3. **Knowledge Area Coverage**: All KAs relevant to the track should have ≥20 questions. Identify blind spots.
4. **Topic Depth vs Breadth**: No single topic should have >8% of a track's questions. No topic should have <3 questions (merge it).
5. **Scenario Diversity**: Within each topic, check that scenarios don't repeat the same setup. Flag near-duplicates.

### Track-Specific Audits

**Cloud Audit Focus**:
- Are the 520 taxonomy_concepts actually distinct? Many are likely synonyms.
- Is the serving/inference pipeline fully covered (tokenization → prefill → decode → KV cache → batching)?
- Are distributed training concepts properly sequenced (data parallel → tensor parallel → pipeline parallel → 3D)?

**Edge Audit Focus**:
- Are hardware platforms balanced (Jetson, Coral, RPi, custom FPGA)?
- Is the sensor→inference→actuation pipeline covered end-to-end?
- Are automotive/industrial/agricultural verticals represented?

**Mobile Audit Focus**:
- Are iOS/Android platform-specific constraints covered?
- Is the on-device vs cloud trade-off spectrum covered?
- Are battery/thermal/memory constraints interleaved across topics?

**TinyML Audit Focus**:
- Are MCU families represented (ARM Cortex-M, RISC-V, ESP32, STM32)?
- Is the full pipeline covered (sensor → preprocess → inference → actuation → comms)?
- Are power modes (active, sleep, deep sleep, harvesting) covered?

### Measure after Iteration 4:
- Orphan concepts (target: 0)
- Prerequisite violations (target: 0)
- KA blind spots (target: 0 KAs with <20 questions)
- Topic concentration (target: max 8% per topic)

---

## Iteration 5: Final Polish + Balance Verification

### End-to-End Validation

1. Run full Opus validation on entire corpus (10 parallel agents, as done today)
2. Verify all questions have:
   - Complete napkin_math (>50 chars)
   - Substantive solution (>100 chars)
   - Valid taxonomy_concept (exists in taxonomy.json)
   - Valid knowledge_area
   - Appropriate Bloom level for content difficulty
3. Recompute full balance scorecard
4. Compare to baseline — every metric should improve

### Release Criteria

| Metric | Threshold |
|--------|-----------|
| OK rate | >85% |
| WARN rate | <15% |
| ERROR rate | 0% |
| Bloom deviation (all tracks) | <3% |
| Reasoning mode CV (all tracks) | <0.35 |
| Competency CV (all tracks) | <0.30 |
| Chain coverage (all tracks) | >50% |
| L6 count (per track) | >50 |
| Short solutions (<100 chars) | <2% |
| Empty napkin_math | 0 |
| Orphan taxonomy concepts | 0 |
| Topic max concentration | <8% per topic |

---

## Execution Model: Parallel Opus Agents

Each iteration uses parallel Opus 4.6 agents:

```
Iteration 1: 5 agents (1 per track) for topic consolidation
           + 8 agents for WARN fixes (split by track × batch)
           + 4 agents for solution enrichment

Iteration 2: 5 agents (1 per track) for targeted generation
           + 5 agents for validation

Iteration 3: 4 agents (edge/mobile/tinyml/global) for chain building
           + 1 agent for cross-track bridge analysis

Iteration 4: 5 agents (1 per track) for deep taxonomy audit

Iteration 5: 10 agents for final validation (same as today's run)
```

Total: ~5 iterations, each measurable, each building on the last.

---

## Quick-Start Command

To begin Iteration 1:
```
Read interviews/IMPROVEMENT_PLAN.md.
Execute Iteration 1 (1A + 1B + 1C in parallel).
After each sub-phase, recompute the balance scorecard.
Report deltas vs baseline before proceeding to Iteration 2.
```
