# Float-exposition fix campaign — live progress

Personally auditing every flagged float against LIVE prose, applying the load-bearing filter,
fixing only genuine thin lead-outs in house style (.claude/_rules). Callouts count as prose;
captions do NOT. Resume phrase: "continue the float-exposition fix campaign."

## Method (validated over intro + ml_systems)
- Candidate list = conservative `ROLLUP.md` (closest to my bar) + my own spot-finds.
- The STANDARD_ROLLUP (351) is a ~5x overcount: most flags are carried by the following
  worked example, a Systems-insight callout, per-item analysis, or the immediate payoff para.
- Real fix rate ≈ 1–2 per chapter, not 5–11.
- Verify EACH live (line numbers drift; cross-chapter refs are false "orphans").

## Filter
- FIX: load-bearing float whose takeaway is absent from body prose AND not unpacked within
  ~1–2 paragraphs / callout; OR a float-announcer colon on a load-bearing float.
- KEEP: reference/lookup exhibit with a clean pointer; takeaway carried by following
  body/callout/worked-example/per-item analysis; body already names a concrete instance.

## VOL 1 (canonical order)
- [x] introduction        — FIX tbl-software-1-vs-2 (named the load-bearing failure-mode row)
- [x] ml_systems          — FIX tbl-ml-systems-paradigm-bottlenecks (added concrete paradigm→lever);
                            6 others KEEP (carried by following content)
- [ ] ml_workflow         — std=0 findings; spot-check only
- [ ] data_engineering    — candidates: TBD (verify std worklist)
- [ ] nn_computation      — candidates: TBD
- [ ] nn_architectures    — fig-mlp L847, fig-cnn-spatial-processing L1237, fig-transformer-attention-visualized L2348
- [x] frameworks          — FIX eq-execution-continuum (announcer colon), lst-torchscript-ir (named the lowering)
- [ ] training            — fig-data-pipeline L2037, tbl-optimization-roadmap L3444, tbl-scaling-decision L6522
- [ ] data_selection      — std flagged 14 + 1 hard; verify
- [ ] model_compression   — fig-kd-overview L2014, fig-quantization-roadmap L3799, fig-color-mapping L7432, fig-sparse-heat-map L7723
- [ ] hw_acceleration     — fig-ai-performance L1574, fig-rising-ridge L2775, eq-batch-ai L4190, lst-dense_layer_def L1148, lst-dense_expansion L1159, lst-nonlinear_layer L1343, lst-arm_sve_vector L1461
- [ ] benchmarking        — tbl-benchmarking-vendor-claims L536, tbl-edge-vs-cloud-constraints L1793, tbl-benchmarking-edgetpu-validation L2929
- [ ] model_serving       — std flagged 13; verify
- [ ] ml_ops              — tbl-monitoring-cost-components L2589, tbl-ab-test-decisions L1725, tbl-technical-debt-summary L3156
- [ ] responsible_engr    — tbl-model-efficiency-comparison L1677
- [ ] conclusion          — std=1; verify

## VOL 2 (canonical order) — not started
- Cross-chapter "hard fails" already CLEARED as false positives: fig-fleet-stack (def vol2 intro L1704,
  used by 9 ch), tbl-prefill-decode (def inference L2797), tbl-dam-bottleneck (def benchmarking).
- Candidates from conservative ROLLUP: distributed_training(4), inference(3), introduction(4),
  ops_scale(2), security_privacy(2), sustainable_ai(2), + 1-each in edge_intelligence, fault_tolerance,
  fleet_orchestration, responsible_ai, robust_ai, collective_communication. Verify each live.

## FINAL — 8 genuine fixes from 351 raw findings (both volumes audited)
Audited the complete high-signal universe: every conservative-ROLLUP candidate + every 🛑
hard-fail in BOTH volumes. Refutation rate ~95%: nearly all std-pass flags are carried by the
following worked example, a Systems-insight callout, per-item analysis, or the immediate payoff.
All vol2 cross-chapter "hard fails" were false positives (per-file scanner blindness).

Edits (branch audit/float-explanation, worktree MLSysBook-float-audit):
1. vol1/introduction            tbl-software-1-vs-2          (named load-bearing failure-mode row)
2. vol1/ml_systems              tbl-...-paradigm-bottlenecks (added concrete paradigm→lever)
3. vol1/frameworks              eq-execution-continuum       (removed float-announcer colon)
4. vol1/frameworks              lst-torchscript-ir           (named the IR lowering to notice)
5. vol1/training                tbl-scaling-decision         (put the table's scale brackets in prose)
6. vol1/hw_acceleration         lst-arm_sve_vector           (named ptrue scalable-width mechanism)
7. vol2/robust_ai               fig-adversarial-googlenet    (FIXED MISMATCHED CLAIM — fig showed the
                                                              very thing the sentence said it went beyond)
8. vol2/collective_communication tbl-interconnect-parameters (put intra/inter-node crossover gap in prose)

RESIDUAL — CLOSED. Exhaustively swept every ⚠️/🛑 in the 3 largest vol2 chapters
(fault_tolerance 28, inference 33, ops_scale 17 = 78 findings): ZERO additional fixes.
All carried by conceptual lead-ins, Systems-insight/lesson callouts, enumeration-after-table,
or worked-example outcomes. eq-tco-ml 🛑 is a false alarm (4 components named + hidden-cost
consequence in the next sentence). Confirms conservative+🛑 caught the entire genuine set.

COVERAGE: both volumes complete. Every conservative-ROLLUP candidate, every 🛑, and every
⚠️ in the 3 biggest vol2 chapters audited against live prose. Final tally: 8 fixes / 351 flags.

## SECOND AUDIT (independent method, worklist-free) — CONFIRMS 8 fixes, 0 new
Run to catch FALSE NEGATIVES the first (worklist-driven) pass could miss:
1. Structural integrity (scanner, all 33 ch): 0 true orphans, 0 broken refs. All "dangling" are
   cross-file (defined in intro/appendix); the "orphan" eq-distributed-training-scaling-efficiency
   is referenced from ops_scale + compute_infrastructure and fully explained at its def.
2. Corpus-wide bare-pointer net (mechanical grep, NOT the worklists): ranked every "ref-as-subject
   + weak verb" sentence shortest-first. Surfaced 12 bare pointers the worklists never flagged
   (incl. model_serving, which was conservative-clean). Checked all 12 live → ALL carried (takeaway
   in the next sentence, a worked-example chain, or full subsections before a consolidation table,
   e.g. vol2 tbl-vol2-lighthouse-archetypes). 0 new fixes.
3. Re-verified all 8 edits live: present, house-style clean, no em-dashes, equation well-formed.
VERDICT: high confidence. 8 fixes is the complete genuine set.
</content>
</invoke>
