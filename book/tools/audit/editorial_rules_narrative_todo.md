# Editorial Rules and Narrative Audit TODO

Status: active review ledger for the `codex/section-flow-pass` worktree.

Use this file for items that need author/editor attention or a later dedicated
pass. Routine fixes that can be made safely in the current pass should not be
listed here.

## Current Pass Notes

- One-agent-per-chapter narrative audits have completed for all primary
  Volume 1 chapters and for the active Volume 2 chapters. Fresh Volume 2
  returns now cover introduction, compute infrastructure, network fabrics,
  data storage, distributed training, collective communication, fault
  tolerance, fleet orchestration, performance engineering, inference, edge
  intelligence, ops scale, and robust AI; earlier audits remain available for
  security/privacy, sustainable AI, responsible AI, and conclusion.
- Safe local fixes from those audits have been committed separately: forward
  dependency localization, concrete-before-definition landings, terminology
  corrections, and index capitalization fixes. Larger LEGO, citation,
  algorithm, feature-store, and chapter-scope repairs remain queued below.
- The current diff has been checked against the high-risk rule files:
  `book-prose.md`, `prose-craft.md`, `emphasis.md`,
  `chapter-architecture.md`, `cross-references.md`, `mlsysim.md`, `fmt.md`,
  `lego-units.md`, `index.md`, `callouts.md`, `notations.md`, and
  `bib-check.md`.
- The branch-touched Fallacies/Pitfalls sections now follow the house
  `Fallacy`/`Pitfall` structure. A whole-book alternation cleanup remains
  separate because the mechanical sweep found many preexisting non-alternating
  sections outside this task.
- The short-section heuristic was run on branch-changed chapters. Clear
  prose-only one-paragraph subsections were collapsed; compact headings that
  introduce a table, figure, review callout, or step sequence were left in place
  because their structure is pedagogically useful.
- Edited LEGO cells were smoke-tested directly for output formatting; the
  checkpoint example now derives the `350 GB` and `1,400 GB` components from
  MLSysIM constants instead of prose-only literals.
- 2026-06-05 section-flow triage: committed `45ed408262` for ops-scale
  CI/CD/rollout/resource-management bridges and `5bea6b53e9` for Volume 2
  edge/fault/inference transitions. The layout-proxy heuristic was rerun on
  both volumes; remaining high-count warnings were inspected by context rather
  than fixed mechanically. Intentional short structures include worked-example
  calculation steps, MLPerf scenario headings, static/dynamic serving
  distinctions, ClinAIOps loop headings, rollback subcases, and figure-backed
  design-pattern headings. These should be reconsidered only in the later
  whole-section macro flow pass if the surrounding prose fails the narrative
  test.
- Standalone-book wording check found no authored-prose `Volume 1`/`Volume 2`
  hits in `book/quarto/contents/vol1` or `book/quarto/contents/vol2`. Raw
  `above`/`below`/`later` greps are noisy because they hit TikZ coordinates and
  comments, so use the prose checker and targeted authored-prose searches
  rather than treating the raw grep as actionable.

## Needs Author Attention

- Confirm whether the `.claude/rules` updates that live in `AIConfigs` should
  be committed separately there after the MLSysBook worktree lands. Current
  relevant dirty rule files are `cross-references.md` and `mlsysim.md`; unrelated
  AIConfigs scratch files are intentionally untouched.
- Decide whether artifact-style bibliography keys such as `euaiact2024` should
  be renamed to author/year keys in a later bibliography cleanup. The
  full-bibliography key/content check has longstanding failures unrelated to
  this pass; branch-touched entries only intersect that check as warnings.
- Clarify whether preexisting theorem or definition callouts may use a bold
  `Pitfall` label outside the house Fallacies/Pitfalls section. The quick sweep
  found one Amdahl's Law theorem callout using that pattern; the edited
  Fallacies/Pitfalls sections themselves follow the canonical structure.
- Schedule a whole-book Fallacies/Pitfalls alternation cleanup. A quick
  mechanical sweep found many preexisting sections in both volumes where
  `Fallacy` and `Pitfall` labels do not alternate strictly; the branch-edited
  sections now follow the canonical pattern, but fixing all older sections is a
  separate editorial pass.
- Resolve source-of-truth or citation support for the GPT-2 optimization final
  profile in `vol1/training/training.qmd`: the final table still contains
  hard-coded throughput and epoch-time values next to `GPT2WalkthroughCalc`
  outputs.
- Resolve citation support for the compressed production-case paragraph in
  `vol1/ml_ops/ml_ops.qmd`: Zillow is visibly sourced, but the adjacent YouTube,
  Tesla, and Facebook production claims should either receive citations or be
  generalized.
- Resolve MLSysIM/source backing for the Edge ML ranges in
  `vol1/ml_systems/ml_systems.qmd`, including memory bandwidth, deployable model
  size, and compression speedup values.
- Resolve MLSysIM/source backing for the TinyML lighthouse values in
  `vol1/model_compression/model_compression.qmd`, including DS-CNN model budget
  and smart-doorbell SRAM assumptions.
- Resolve citation or appendix-backed support for the staffing benchmark in
  `vol2/ops_scale/ops_scale.qmd` that says large training clusters commonly need
  5--15 infrastructure engineers per 10,000 GPUs.
- Resolve citation/source backing for the Volume 2 introduction compute-growth
  clause (`vol2/introduction/introduction.qmd`) that gives frontier training
  compute orders of magnitude and a date range.
- Resolve MLSysIM/source backing for the HBM definition in
  `vol2/compute_infrastructure/compute_infrastructure.qmd`, which contains H100
  bandwidth, DDR5 bandwidth, energy-per-bit values, ridge-point arithmetic, and
  attention-intensity examples.
- Resolve citations for the MoE routing paragraph in
  `vol2/distributed_training/distributed_training.qmd`, including capacity
  factor, auxiliary-loss weight range, and Mixtral top-2/8-expert details.
- Resolve MLSysIM/source backing for the
  `vol2/fault_tolerance/fault_tolerance.qmd` cost example using 25,000 GPUs,
  $2/GPU-hour, and a $1.2M/day calculation.
- Resolve LEGO/source backing for the fault-tolerance cluster MTBF table and
  graceful-degradation loss example. The audit flagged hardcoded derived MTBF,
  failures-per-day, CTR-loss, and outage-cost values that should either be
  computed from a nearby cell or recast as qualitative scenario prose.
- Resolve source or scenario backing for fleet-orchestration cost and failure
  claims: the 1--4 failures/day per 1,000 GPUs band, 10,000-GPU capital/waste
  figures, scheduler speedup ranges, and build-vs.-buy ROI numbers should be
  cited, computed, or clearly framed as illustrative assumptions.
- Resolve inference serving support for hardcoded cost ratios, adaptive
  batching measurements, hardware/framework support tables, and large public
  case-study scale claims. Where public architecture details are inferred, the
  case studies should say so explicitly rather than reading as internal facts.
- Resolve performance-engineering source-of-truth issues for the SRAM/HBM
  energy example, widening-gap hardware table, tensor-parallel transfer-time
  example, overlap-budget plot, MBU/MFU thresholds, hero-run tax, and the 70B
  optimization case study.
- Resolve edge-intelligence source backing for on-device training memory
  ranges, adapter-switching values, heterogeneity penalty ranges, minimum
  participation thresholds, and fallacy/pitfall quantitative anchors. The
  PyTorch-specific frozen-parameter listing should be considered for algorithm
  or framework-neutral pseudocode treatment in the algorithm pass.
- Resolve ops-scale feature-store structure and indexing. The body section
  should carry the primary `Feature Store!definition` anchor, split the
  dual-store mechanism from the training-serving skew failure, and lead with
  offline analytical vs. online key-value access constraints rather than
  vendor examples.
- Resolve ops-scale support for the Jeff Dean attribution, the 50-model
  platform threshold, proactive-maintenance utilization ranges, and case-study
  quantitative claims such as the 95 percent infrastructure-code figure.
- Resolve robust-AI citation or scenario support for definition-callout
  accuracy ranges, reliability overhead ranges, fraud-pipeline defense
  effectiveness, adversarial-training cost multipliers, medical-imaging
  adversarial example confidence, and poisoning/backdoor fallacy numbers.
- Review thin subsection scaffolds flagged by fresh audits: fault-tolerance
  degradation dimensions and replica placement; fleet heterogeneous gang
  scheduling and research scheduler mini-profiles; performance scaling
  regressions and playbook sections; edge peak memory and repeated
  design-constraint prose; ops-scale model registries, ensemble management,
  validation gates, rollout risk, dashboards, cost components, and
  organizational patterns; robust-AI retraining levels and data-poisoning
  defense landing.

## Active Pass Queue

- Rule compliance pass over the accumulated diff: emphasis, footnote heads,
  cross-reference casing, direct algorithm references, notation, citations,
  index placement, callout labels, fallacy/pitfall formatting, and LEGO output
  variable naming.
- Small-paragraph and one-paragraph-subsection pass: identify places where
  one- or two-sentence paragraphs, or `###`/`####` headings with one paragraph,
  should be fused into proper argument paragraphs or collapsed into the parent
  section.
- Bullet-list pedagogy pass: keep bullets only where they are the most effective
  teaching form, especially in examples, callouts, checklists, and comparison
  tables; convert list texture masquerading as prose only when narrative would
  teach better.
- Example/callout sequence pass: review examples that were folded into prose and
  restore step-by-step structure when the callout's pedagogical job is to show a
  sequence or trace.
- Macro section-flow pass: after local edits, re-read every changed `##` and
  `###` section so each paragraph flows from the previous paragraph and into the
  next one.
- References pass: inspect edited sections for unsupported claims; if new
  references are needed, use a dedicated staging `.bib`, run BetterBib, smoke
  with BibTeX, then copy only reviewed entries into the main references file.
- Code/LEGO pass: verify touched Python cells follow LEGO locality, registry
  sourcing, output string naming (`_gb_str`, `_pct_str`, `_pp_str`,
  `_mult_str`, and related conventions), and MLSysIM source-of-truth rules.
- Fallacies and pitfalls pass: preserve the house structure
  `**Fallacy**: *misconception.*` / corrective paragraph and the matching
  `**Pitfall**:` form across all edited chapters.

## Deferred Dedicated Passes

- Add or validate an MLSysIM scenario for variable wearable/device profiles
  such as Oura Ring, smartwatch, autonomous vehicle, or similar reusable
  device-context anchors, then source any prose values from that registry home.
- Add or validate a pre-commit check for footnote head capitalization when a
  bold footnote head contains one or more title-caseable terms.
- Re-run the final Tokenland litmus after all rule/style/narrative edits are
  done, then decide whether any remaining section lacks textbook perspective,
  engineering judgment, or narrative flow.
- Run the index placement pass after the algorithm pass and other text edits,
  because paragraph fusion can move `\index{}` tags away from the exact
  concept-bearing term.
- Run a progressive-disclosure pass in Volume 1 order and Volume 2 order
  separately. Volume 2 can assume Volume 1 background knowledge but must remain
  an independent book and must not cross-reference Volume 1.
- Run the engineering-book audit after prose cleanup: each chapter should still
  feel like machine learning systems engineering, with quantitative anchors,
  cut-and-napkin math, concrete systems, and MLSysIM-backed values where useful.
- Run the algorithm conversion review last in a separate worktree if edits are
  warranted, using `.claude/_reviews/algorithm_pass/ALGORITHM_PASS.md` as the
  guidance source.
