# Next Session: Solution Enrichment + Chain Building

**Goal:** Apply enriched solutions, build chains to 50%+, revalidate for convergence.

**Start by reading this file, then execute phases in order.**

---

## Current State (as of 2026-03-24 late night)

### What's Done (this session — Deep Validation + Taxonomy Coherence)

**Math Validation (2 rounds, 29 parallel Opus 4.6 agents):**
- **Round 1**: 10 agents validated all 5,228 Qs with 293 fixes
  - H100 40GB→80GB (15 fixes), T4 8GB→16GB (6), A100 BW 1.5→2.0 TB/s (8)
  - PCIe Gen4 bidirectional confusion (8), KV cache math errors (8)
  - Multi-factor multiplication truncation in TinyML (68)
  - L6+ → L6 normalization (60), deep_dive_url fixes (39)
- **Round 2**: 19 agents with track-specific hardware reference sheets
  - Full recomputation of every calculation in every question
  - Track-specific specs: edge (Jetson/Coral/RPi), mobile (Snapdragon/Apple), tinyml (Cortex-M/ESP32)
  - All 24 chunks processed and merged

**Taxonomy Coherence:**
- **438 empty taxonomy_concepts → 0** (assigned via Opus concept-matching agent)
- **224 missing concept refs → 0** (82 concepts remapped to existing taxonomy entries + 3 new concepts added)
- **729 overloaded questions redistributed** to more specific concepts
  - energy-roofline-model: 320 → 15 (split to ridge-point, arithmetic-intensity, hardware, workloads)
  - block-wise-quantization: 289 → 121 (split to PTQ, QAT, mixed-precision, model-quantization)
  - latency: 175 → 42 (split to tail-latency, latency-budget, latency-jitter)
  - application-embedded-serving: 152 → 29 (split to batching, memory, edge, cloud, ops sub-concepts)
- **Orphan concepts reduced: 141 → 94**
- **Cross-track concepts: 66 → 101** (concepts spanning 3+ deployment tracks)
- **Concept question counts updated** across all 662 concepts

**Paper (paper.tex) Updated:**
- `\numconcepts`: 193 → 662
- `\numedges`: 601 → 746
- `\numquestions`: 4,802 → 5,228
- Added "Iterative Deep Validation" subsection documenting error patterns
- Added taxonomy coherence audit documentation

### Current Numbers
- **5,228 active questions** across 5 tracks
- **662 taxonomy concepts** with 746 prerequisite edges
- **101 cross-track concepts** spanning 3+ tracks
- **0 empty taxonomy_concepts** (was 438)
- **0 missing taxonomy refs** (was 224)
- Cloud: 181 L6+, Edge: 99, Mobile: 62, TinyML: 72, Global: 24

### Solution Enrichment Results
- **Batches 0, 2, 3**: Applied — 245 solutions enriched
- **Batch 1**: Still running — check `/tmp/vault_iter2/enriched_1.json`
  - When ready, apply with this script:
    ```python
    import json
    with open('corpus.json') as f: data = json.load(f)
    with open('/tmp/vault_iter2/enriched_1.json') as f: enriched = json.load(f)
    for e in enriched:
        idx = e['corpus_index']
        if len(e['realistic_solution']) > 100 and len(data[idx]['details']['realistic_solution'] or '') < 100:
            data[idx]['details']['realistic_solution'] = e['realistic_solution']
    with open('corpus.json', 'w') as f: json.dump(data, f, indent=2, ensure_ascii=False)
    with open('staffml/src/data/corpus.json', 'w') as f: json.dump(data, f, indent=2, ensure_ascii=False)
    ```
- **Short solutions: 329 → 83** (75% reduction, will be ~0 after batch 1)

---

## Phase 1: Apply Solution Enrichments (5 min)

```python
# Check if enrichment files exist
import json, os
for i in range(4):
    path = f'/tmp/vault_iter2/enriched_{i}.json'
    if os.path.exists(path):
        enriched = json.load(open(path))
        # Apply to corpus by corpus_index
```

**Target**: Zero questions with solutions under 100 chars.

---

## Phase 2: Chain Coverage Expansion (30 min)

Current: 34% chain coverage (1,777 of 5,228 in chains).
Cloud has 424 chains, but edge (48), mobile (32), tinyml (29) need work.

**Strategy:**
1. For each consolidated canonical_topic, find questions spanning L1→L6
2. Build chains of 3-6 questions with increasing Bloom levels
3. Focus on edge/mobile/tinyml tracks

**Target**: 50%+ chain coverage across all tracks.

---

## Phase 3: Targeted Generation for Gaps (30 min)

Based on balance scorecard:
- **Global track**: Only 200 Qs, apply has 12, format only 14% calculation
- **TinyML reasoning**: CV=0.62 (most imbalanced), optimization-task has only 17 Qs
- **94 orphan concepts**: Need 3+ questions each

Generate with Opus, validate with Flash.

---

## Phase 4: Final Validation + Convergence (20 min)

1. Run full Opus validation on entire corpus
2. Compute final scorecard
3. Compare to baseline and this session's scorecard
4. If all metrics improved, declare convergence

---

## Quick Start

```
Read interviews/NEXT_SESSION_PLAN.md.
Check if /tmp/vault_iter2/enriched_*.json exist. If yes, apply them.
Execute Phases 1-4 in order.
Use parallel Opus agents for generation and validation.
```
