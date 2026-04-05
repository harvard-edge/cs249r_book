# StaffML North Star v2 (Post-Review Synthesis)

## Thesis (unchanged)
The corpus size is DERIVED, not chosen. It follows from principled constraints.

## What Changed After 10 Expert Reviews

### Constraint 1: Topics → ~90 (was 79)
Add ~11 topics based on multi-reviewer consensus:
- **autograd-computational-graphs** (Soumith: P0, "as fundamental as memory hierarchy")
- **operator-dispatch-runtime** (Soumith: P1, "where 'why is my model slow' gets answered 60% of the time")
- **memory-allocator-design** (Soumith: P1, Dean: endorsed)
- **software-portability** (Lisa Su: critical, "CUDA/ROCm/SYCL portability")
- **communication-computation-overlap** (Huang: "single most impactful distributed optimization")
- **disaggregated-serving** (Dean: "dominant LLM serving architecture 2025-2026")
- **prefix-caching** (Dean, Song Han: major KV-cache optimization)
- **sequence-parallelism** (Dean: "core to 100B+ training")
- **compression-pipeline-design** (Song Han: how compression techniques compose)
- **platform-api-design** (Satya: "entirely missing, defines Staff platform work")
- **heterogeneous-inference-orchestration** (Satya: runtime hardware routing)

### Constraint 2: Applicability → Fix Errors
Wrong exclusions to restore (Reddi + Soumith):
- kernel-fusion → add TinyML (MCU operator fusion is MORE critical)
- mixed-precision → add TinyML (INT8/INT16 per-layer selection)
- systolic-dataflow → add TinyML (Ethos-U, NPU dataflow)
- graph-compilation → add mobile + TinyML (IS the execution model)
- scheduling-resource-management → add edge (Jetson shares accelerators)

Wrong inclusions to remove (Reddi):
- datacenter-efficiency → remove TinyML
- 3d-parallelism → remove TinyML
- gradient-synchronization → remove TinyML
- collective-communication → remove TinyML

### Constraint 3: Capacity → Variable, Not Fixed
The 3/4/5 model is too simplistic (5 reviewers challenged it):
- Song Han enumerated 8+ distinct optimization questions per cell
- Dean suggests 8-10 for L5/L6+, 5-7 for L3/L4
- Reddi says recall needs 5, not 3
- Patterson demands empirical saturation curves

**New model**: Capacity varies by (zone × level), not just zone:
| | L1-L2 | L3-L4 | L5-L6+ |
|---|---|---|---|
| Simple zones (recall, implement) | 3 | 4 | 5 |
| Complex zones (analyze, design, ...) | 4 | 5 | 7 |
| Mastery | 5 | 7 | 10 |

This raises the principled total from 9,430 to ~12,000-14,000.
But: must validate empirically via semantic similarity analysis.

## Priority Actions (by reviewer consensus)

### MUST (5+ votes)
1. **Rebalance**: Cap overfilled cells, fill underfilled critical topics
   (flash-attn 25→60, spec-decode 19→50, pruning 282→cap at 150)
2. **Validate capacity**: Run semantic similarity on overfilled cells,
   plot marginal info gain, find empirical knee

### MUST (3-4 votes)
3. **Add compiler/SW stack topics** (autograd, dispatch, runtime)
4. **Add production/ops questions** to existing topics (incident response,
   debugging from incomplete info, migration scenarios)
5. **Validate zone model**: Inter-rater reliability study on 50 questions
6. **Fix applicability matrix errors** (10 pairs to flip)

### SHOULD (2-3 votes)
7. **Add platform thinking** (multi-tenant, API design)
8. **Add rubrics/timing** (weak/pass/strong answers, expected solve time)
9. **Increase vendor diversity** (MI300X to 25% of cloud, add ROCm)
10. **Expand RAI coverage** (privacy, inclusiveness, transparency)

## What NOT to Do
- Don't generate 3,000 more questions into existing structure
- Don't treat all cells as equal capacity
- Don't optimize for raw count over distribution quality
- Don't claim the zone model is validated without evidence

## Convergence Status
- Round 1+2: 10/12 reviews complete, 15+ structural issues identified
- Status: NOT CONVERGED (still getting new issues)
- Need: Round 3 with refined framework to check convergence
