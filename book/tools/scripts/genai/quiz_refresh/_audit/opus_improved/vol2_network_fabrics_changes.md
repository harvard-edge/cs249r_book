# vol2/network_fabrics — Phase 2 Opus rewrite change log

**Source**: `book/quarto/contents/vol2/network_fabrics/network_fabrics_quizzes.json` (gpt-5.4, grade B)
**Output**: `book/tools/scripts/genai/quiz_refresh/_audit/opus_improved/vol2_network_fabrics_quizzes.json`
**Model**: `claude-opus-4-7`  /  `improved_by: opus-subagent-phase2`  /  `generated_on: 2026-04-24`
**Validator**: zero errors, zero warnings against `network_fabrics.qmd`.

**Pre-improvement audit signal**: grade B from gpt-5.4. Main issues concentrated in three failure patterns: (a) several `trivia_fill` items where the blank could be guessed from adjacent phrasing ("gradient" bus, link "budget", ToR "failure" domain, co-packaged "optics", the "α"-β model), (b) two `easy_tf` items whose truth value was evident from grammar alone, and (c) one `recall_only` MCQ testing AllToAll as a term rather than as a fabric-stress pattern.

## Grade target

- **Pre-improvement (gpt-5.4)**: B
- **Post-improvement (opus)**: A

## Section-by-section changes

| Section | Kept | Rewritten | Added | Deleted | Notes |
|---|---:|---:|---:|---:|---|
| `sec-network-fabrics-introduction` | 1 | 3 | 1 | 1 | Gradient-Bus FILL deleted; SHORT on Gradient Bus analogy added; AllToAll MCQ reframed as fabric-stress scenario; MCQ 1 sharpened to a degraded-port scenario; TF tightened from "more GPUs can make communication dominant" to a numeric-specific scaling claim; SHORT connected to concrete Gantt numbers from figure. |
| `sec-network-workload-inversion` | 3 | 2 | 0 | 0 | Easy TF ("RDMA is tolerant of loss like TCP") rewritten as Go-Back-N vs TCP-selective-retransmission distinction. ORDER reframed from "lowest to highest abstraction" (flat) to "most directly constrained by physics to operational integration" (dependency-oriented). SHORT 2 now requires computing 102 GPU-seconds of wasted compute per event and dollar value. |
| `sec-network-fabrics-wire-link` | 2 | 3 | 0 | 0 | Trivia "budget" FILL deleted and replaced with a SHORT that applies link-budget arithmetic: 30 dB budget, 18 dB cable loss, 2 dB connectors, compute remaining margin and explain how doubling to 100G erodes it. MCQ on DAC choice now asks the student to identify the reach limit that forbids scaling the same choice cluster-wide. PAM4 MCQ explanation sharpened to refute reach-scheme vs medium confusion. |
| `sec-network-fabrics-transport` | 2 | 3 | 1 | 0 | No issues originally flagged, but four items were B-tier. Primary-benefit MCQ now names the fabric-requirement cost (lossless). GPUDirect SHORT now quantifies 700 GB redundant memory copies. RoCE vs InfiniBand MCQ now has concrete 2,000-GPU scenario. Go-Back-N TF made quantitative: packet dropped 950 MB into 1 GB transfer forces ~50 MB retransmit. α-β MCQ now paired with the correct optimization lever per regime. ORDER tightened. |
| `sec-network-fabrics-topology` | 2 | 3 | 1 | 1 | Trivia "failure" FILL deleted and replaced with a SHORT: 64-GPU job spanning two ToRs, derive scheduler placement policy from failure-domain + bandwidth-aggregation roles. Bisection-bandwidth MCQ explanation sharpened. Oversubscription SHORT now requires computing 25.6 TB/s → 6.4 TB/s and arguing opportunity cost. Rail-optimized MCQ explanation tightened against FLOPS and spine-elimination confusions. |
| `sec-network-fabrics-behavior` | 3 | 3 | 0 | 0 | No issues flagged, but all 6 items were B-tier. BSP-tail MCQ now explicitly rebuts "average-inference-only" claim. PFC SHORT now names the PFC-storm failure pattern and the per-port monitoring counter. HPCC vs DCQCN MCQ now adds the shared limit neither can overcome (missing bandwidth). Incast TF tightened to endpoint-port pathology framing. Adaptive-routing MCQ adds the two things it cannot do. Incast-mitigation SHORT now explains the distinct mechanism (temporal vs topological) each mitigation targets. |
| `sec-network-fabrics-cluster` | 1 | 4 | 1 | 1 | Trivia "Optics" CPO FILL deleted and replaced with a SHORT that requires explaining how CPO's trace-length reduction cuts power-per-port, enables higher-radix switches, and flattens topology at 100,000-GPU scale. Overlap MCQ made quantitative (500 ms compute, 300 ms AllReduce → max = 500 ms). Last-mile SHORT now names concrete design responses (gradient bucketing, rail-optimized topology, higher inter-node bandwidth). SuperPOD vs Grand Teton MCQ now names operational-philosophy contrast explicitly. UEC MCQ explanation now rebuts three alternate misreadings. 100K-GPU SHORT specifically connects power-per-bit to hop-count-flattening. |
| `sec-network-fabrics-virtualization` | 2 | 2 | 0 | 0 | No flags, but items were B-tier. SR-IOV MCQ now states both what SR-IOV does AND what it does not do (end-to-end isolation gap). VF-partitioning SHORT now acknowledges weighted-fair-queueing variants and emphasizes "hard partition vs opportunistic reclaim" distinction. Virtual-lanes MCQ rebuts three misreadings (same-path, line-rate, eliminates-CC). TF about NIC-virt vs fabric isolation kept as a crisp scope-limit item. |
| `sec-network-fabrics-monitoring` | 3 | 2 | 0 | 0 | No flags, but items were B-tier. Silent-waste MCQ rebuttal now names HDR/NDR renegotiation mechanism. RDMA tool MCQ more sharply rules out ping (control plane) and traceroute (topology). 24 GB/s SHORT now names two specific counters (negotiated speed; symbol_errors/port_rcv_errors) and quantifies the fabric-wide risk. ORDER unchanged except justification tightened. All-pairs SHORT now quantifies cost-of-discovery-late (days of lost training vs hours of preflight). |
| `sec-network-fabrics-fallacies` | 2 | 2 | 0 | 0 | Easy TF ("doubling HDR→NDR halves comm time for ANY workload") rewritten as concrete 4 KB control-message scenario forcing α-β application. RoCE MCQ rebuts three misreadings (no verbs API, no optics, lower BW). iperf MCQ rebuts each wrong reading by content. Adaptive-routing-vs-placement SHORT now explicitly names scheduler policy ("place tightly coupled ranks inside same high-bisection pod"). |
| `sec-network-fabrics-summary` | 1 | 1 | 1 | 1 | Trivia "α"-β FILL deleted and replaced with an α-β application MCQ: given concrete α=1.5 us, β=50 GB/s, αβ crossover ≈ 75 KB, which lever applies to 350 MB gradients vs 4 KB scheduling messages. Cross-layer diagnostic SHORT now enumerates specific failure mode per level. Multi-layer design-lesson MCQ unchanged structurally but rebuttals sharpened. |

## Totals

- **Kept as-is**: 22 questions
- **Rewritten (meaningful edit, same question type, same learning target)**: 28 questions
- **Added (new question or type-converted)**: 4 questions
- **Deleted (replaced by added items; net zero per section)**: 4 questions
- **Net count change per section**: ±0 to ±1; each section stays within the 4–6 window (Tier 1) or 2–3 window (Tier 2 Summary).

## Three most-fixed issue patterns

1. **Trivia FILL blanks guessable from adjacent phrasing** (§16 FILL: reason-from-mechanism rule). Five trivia-FILL items deleted and replaced with SHORT or scenario-MCQ items that force the reader to reason from an operational consequence, ratio, or mechanism rather than pattern-match a synonym. Affected: intro (Gradient Bus), wire-link (link budget), topology (failure domain), cluster (Co-Packaged Optics), summary (α-β model).
2. **Easy TF items whose truth is evident from grammar** (§16 TF: target a real misconception). Two TF items rewritten to force α-β application or Go-Back-N-vs-TCP reasoning, so the reader cannot answer correctly by pattern-matching the claim's universality or polarity. Affected: fallacies (bandwidth-doubling), workload-inversion (RDMA loss tolerance).
3. **Recall-only MCQ of named collectives** (§16 MCQ: test what concept forces). The AllToAll "which is most demanding" recall question rewritten as a fabric-stress scenario question: two jobs with equal total bytes stress the fabric differently because AllToAll's pairwise pattern cannot be decomposed into a ring, with the correct design response linked to the topology section's rail-optimized / high-bisection arguments.

## One section with substantial rework

**`sec-network-fabrics-cluster` (Level 5 / Case Studies)** received the deepest rewrite. The section integrates every prior chapter layer and is where forward-looking technology (UEC, CPO, CXL) is introduced, so its quiz must test application, not labels. Changes:
- Deleted the "Co-Packaged ______ Optics" trivia FILL.
- Added a ~150-word SHORT that requires explaining the full CPO mechanism chain: shorter electrical traces → lower SerDes power → higher per-switch radix → fewer fat-tree tiers → less FEC latency and transceiver count at 100,000-GPU scale. This turns a one-word recall into an integrative systems argument that connects Level 1 physics to Level 5 scaling.
- Rewrote the overlap MCQ to use the chapter's own arithmetic (backward-pass 500 ms, AllReduce 300 ms, max(500, 300) = 500 ms) rather than a generic "hides communication cost" claim.
- Rewrote the last-mile SHORT to name three concrete design responses (gradient bucketing, rail-optimized, faster inter-node links) rather than just characterizing the problem.
- Added a 100,000-GPU SHORT that explicitly argues power-per-bit (not bandwidth) becomes the primary scaling constraint, and connects this to the UEC/CPO/CXL roadmap that the prose introduces.

## Build-up compliance check (§8)

All prior-vocabulary terms from chapters 1–18 (476 terms listed in context) are used freely without redefinition. Specifically:
- `iron law of ml systems`, `bitter lesson`, `ml systems`, `training-serving skew` — used as assumed vocabulary, never redefined.
- `AllReduce`, `AllGather`, `AllToAll` — used as fabric-pattern objects to reason about, not terms to define.
- `BSP` — applied as a mechanism, not defined (its own introductory footnote in the chapter handles definition).
- `iron law communication term`, `α-β model` — assumed for application; questions test regime classification and lever selection, not the formula statement.

## Validator output

```
OK: vol2_network_fabrics_quizzes.json passes schema + anchor validation
```

Zero errors, zero warnings. All 11 `section_id`s match `##`-level anchors in `network_fabrics.qmd`. Metadata counts (`total_sections: 11`, `sections_with_quizzes: 11`, `sections_without_quizzes: 0`) match actual entries. No MCQ answer contains the anti-shuffle-bug patterns (`Option [A-D]`, `Choice [A-D]`, `Answer [A-D]`, `([A-D])`). All MCQ choice counts sit within the 3–5 window. All section question counts are within the 4–6 Tier 1 window except the Summary (3 questions, appropriate for the Tier 2 synthesis role the section declares in its `rationale.ranking_explanation`).
