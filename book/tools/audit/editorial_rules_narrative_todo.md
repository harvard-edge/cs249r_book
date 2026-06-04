# Editorial Rules and Narrative Audit TODO

Status: active review ledger for the `codex/editorial-rules-narrative-audit` worktree.

Use this file for items that need author/editor attention or a later dedicated
pass. Routine fixes that can be made safely in the current pass should not be
listed here.

## Current Pass Notes

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
