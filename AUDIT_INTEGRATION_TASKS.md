# Audit Integration Task Ledger

Status: reconstructed from `/Users/VJ/GitHub/codex-session-recovery.md` and
raw session `019eb70c-cac4-7c61-9ff3-57a12b8dea45` on 2026-06-11.

Current checkout: `/Users/VJ/GitHub/MLSysBook`
Current branch: `dev`
Integrated worktree: `/Users/VJ/GitHub/MLSysBook-audit-integration`
Integrated branch: `fix/audit-integration`

## Current Execution Sequence

This section is the easiest place to see where the session is in the ordered
work. The detailed backlog remains in the sections below.

- [x] Create and recover the audit-integration worktree from local `dev`.
- [x] Read the recovery handoff, relevant plans, decisions, and `.claude/rules`.
- [x] Compare recovered ledgers/current `dev` and classify completed, open, and
  blocked work.
- [x] Complete the chapter-opening Purpose/layout pass:
  - [x] Ensure every main-chapter Purpose prose block is one paragraph.
  - [x] Build every main chapter in PDF context and inspect the first two pages.
  - [x] Fix opener layout cases where Purpose bleeds to page 2 or learning
    objectives do not begin cleanly at the top of the next page.
- [x] Clean stale ledger bookkeeping for already-merged work.
- [x] Define and verify the LEGO + MLSysIM QA method, including checker
  correctness checks.
- [x] Audit LEGO locality and split/move oversized or macro cells so exported
  prose values are close to first use.
- [x] Audit MLSysIM registry/model/scenario code for correctness,
  documentation, and source-of-truth fit while reviewing LEGO dependencies.
  - [x] Remove stale MLSysIM reference aliases and fix affected consumers/tests.
  - [x] Source the performance-engineering compilation-dividend operands from
    local LEGO.
  - [x] Source the ops-scale TCO sensitivity table from local LEGO.
  - [x] Source backed Vol. II Introduction GPT-3/GPT-4/TPU pod values from
    MLSysIM/LEGO and fix strict inline-Python render hazards.
  - [x] Record remaining public-release decisions: Meta RSC/TPU ICI anchors,
    GPT-4-class training-FLOP scenario policy, sustainable-AI edge embodied
    carbon scenario, and public scenarios that intentionally evaluate `FAIL`.
- [x] Run rendered LEGO QA across vol1 and vol2: variable names, registry
  sourcing, `fmt` usage, execution output, precision, and prose fit.
- [x] Run a rendered precision appropriateness pass: verify each displayed value
  uses the right precision for the prose role and student-facing claim, not only
  that it avoids spurious `.0` output.
  - [x] Committed as `a6f5acfa6a`: `Fix quantitative prose precision issues`.
    Covered the final coherence warnings in Vol. I `appendix_machine`,
    `ml_workflow`, `nn_architectures`, and Vol. II `appendix_assumptions`,
    `appendix_communication`, `data_storage`, `fleet_orchestration`,
    `ops_scale`, and `security_privacy`.
- [x] For every touched chapter, run chapter build/debug with verbose output and
  fix render errors, missing references, and missing figures.
- [x] Run Binder/pre-commit checks before commits.
- [x] Record source-of-truth/scenario-modeling recommendations for
  public-release quality decisions.
- [x] Merge `fix/audit-integration` into local `dev`.
- [x] Remove reader-facing durable audit labels from prose; keep durable as
  audit vocabulary only.
- [x] Re-check the GB/GiB policy: binary units may appear in internal
  calculations, but reader-facing prose keeps decimal GB unless a binary unit is
  explicitly part of the teaching point.
  - [x] Cleaned concrete reader-facing leaks in Vol. I `training`,
    Vol. I `frameworks`, and Vol. II `appendix_assumptions`: hardware capacity
    displays now use `fmt_memory_capacity(..., unit=GiB)` for branded GB
    labels, and derived memory totals use decimal `fmt_memory(..., unit=GB)`.
- [x] Verify `fmt_fps` and scan for similar common formatter candidates while
  preserving the rule that any new `fmt_*` type requires a corpus applicability,
  LEGO-output, prose, and precision pass.
  - [x] Confirmed `fmt_fps` exists in `mlsysim.fmt`, has tests, and the
    previously flagged `cam_fps_str` export uses it.
- [x] Record the current TOC convention decision: keeping Introduction inside
  Part I and Conclusion inside Part IV is acceptable when the parts represent
  teaching arcs; separate orphan parts need meaningful headers such as
  "Orientation" and "Synthesis" before they would be preferable.
- [x] Update LEGO/fmt/MLSysIM/rule guidance if recurring gaps show up.
  - [x] Verified `.claude/rules/bib-check.md` contains the BetterBib-first
    staging workflow for every new or changed bibliography entry.
  - [x] Verified `.claude/skills/audit-book-artifacts/SKILL.md` records the
    LEGO/fmt prose-boundary rules, formatter-gap policy, and artifact audit
    commands for future `/audit <type>` use.
- [x] Run one read-only parallel chapter concept-coverage audit per Vol. I and
  Vol. II chapter, in canonical chapter order, asking whether each chapter
  teaches the important concepts for an introductory ML systems textbook or an
  advanced ML systems-at-scale volume without trying to become encyclopedic.
  - [x] Vol. I concept-coverage auditors completed for Chapters 1--16; findings
    are queued as local edit candidates vs. authorial-decision packets.
  - [x] Vol. II concept-coverage auditors completed for Chapters 1--16 plus the
    conclusion; findings are queued below as local edit candidates vs.
    authorial-decision packets.
- [x] Run read-only appendix-flow audits for Vol. I and Vol. II appendices:
  verify the appendix sequence, internal flow, and reference-vs-teaching role
  make sense from the actual text rather than only the table of contents.
  - [x] Vol. I appendix-flow audit completed; D·A·M capitalization was clean,
    the Jeff Dean latency note needed a bibliography-backed source, and broader
    D·A·M/log-space/checkpoint findings are queued for authorial review.
  - [x] Vol. II appendix-flow audit completed; copyedit PDF appendix order was
    aligned with canonical Vol. II appendix order, and remaining appendix
    content findings are queued for later prose decisions.

### Concept-Audit Integration Queue

Vol. I concept coverage completed read-only for Chapters 1--16. Local edit
themes to integrate where they are clearly supported by existing chapter
context:

- Chapter 1--4: tighten early FLOP/s, weight, lifecycle, reliability,
  dataset-split, adversarial-ingestion, and provenance scaffolding without
  teaching later hardware/quantization too early.
- Chapter 5--8: add or refine tensor-layout, calibration, RNG/determinism,
  export validation, checkpoint-state, and workflow/MLOps callbacks where they
  strengthen the current teaching claim.
- Chapter 9--12: improve sampling/manifest/dedup language, compression
  distinctions, profiling/device-residency bridges, and benchmarking
  statistical-power/cost-normalized framing.
- Chapter 13--16: strengthen ingress contracts, ML-quality SLOs, retirement
  lifecycle, risk-tiering, human review capacity, and conclusion closure, while
  keeping advanced Volume II previews scoped as previews.

Vol. II concept coverage completed read-only for Chapters 1--16 plus
Conclusion. Local edit candidates:

- Chapters 1--3: introduce C³ earlier, rebalance scaling/serving coverage,
  clarify power-wall placement, define injection bandwidth, bridge collectives
  before AllReduce detail, and source or soften time-sensitive hardware claims.
- Chapters 4--6: add storage publication/restore/control-plane framing,
  clarify scaling/process-group/collective ordering concepts, bridge early
  α-β/ring calculations, and keep tool/API catalog material scoped.
- Chapters 7--9: add RTO/RPO and serving-headroom framing, scheduler
  control-plane/resource-lifecycle language, storage-aware placement, robust
  performance-experiment discipline, and fix local terminology/table/caption
  issues.
- Chapters 10--12: strengthen admission-control, version/cache invalidation,
  edge-cloud decision framing, privacy-control stack, ML ops control-plane,
  quality SLO/error-budget framing, and exact-claim citation/scenario wording.
- Chapter 13 Security/Privacy: add compact asset/boundary/adversary/control,
  ML artifact supply-chain, LLM tool/RAG boundary, incident-response, and
  validation framing while avoiding a DP mini-chapter rewrite.
- Chapter 14 Robust AI: shorten long learning objectives, fix table caption
  placement, resolve drift-response tension, add calibration/selective
  prediction and robust release-gate framing where compact, and avoid expanding
  into an incident catalog.
- Chapter 15 Sustainable AI: move the dominant-lifecycle-term decision
  procedure earlier, promote average-vs-marginal emissions into body prose,
  consolidate repeated lifecycle/PUE explanations, and keep policy content
  tied to systems control planes.
- Chapter 16 Responsible AI: add sensitive-attribute governance choices,
  validation evidence by risk class, sociotechnical accountability in summary,
  and a monitoring-without-remediation pitfall; source-check current legal
  claims before publication.
- Conclusion: remove/soften GPT-4 training-infrastructure overclaims, narrow
  TP/MoE traffic language, reduce uncited future-facing optical/brain material
  into synthesis, standardize Llama 3 naming, and add a compact closing method
  for following constraints across C³ and the fleet stack.

Authorial-decision packets to preserve rather than silently resolve:

- Whether each overfull late chapter should remain a survey or be tightened
  around a single systems decision procedure.
- Whether DP accountant depth belongs in Security/Privacy body text or an
  appendix/notebook-depth lane.
- Whether semantic/generative robustness and LLM/agent safety need dedicated
  sections or only bridges to adjacent chapters.
- Whether Responsible AI should introduce a formal risk-management lifecycle
  and causal/measurement-validity section.
- Whether the Conclusion should keep future-looking optical/biological
  efficiency material or close primarily on architectural synthesis.
- [ ] Run read-only progressive-disclosure audit by chapter using cumulative
  prior-chapter context.
- [ ] Run chapter-thread audit: Purpose, section arc, examples, summary, and
  golden-thread callbacks all point to the same central teaching claim.
- [ ] Run chapter paragraph-flow audit: every paragraph makes a point, connects
  logically, and avoids dangling single-sentence fragments unless intentional.
- [ ] Flag opportunities for student reflection questions/connective prompts
  without inventing new authorial content unilaterally.
- [~] Integrate accepted progressive-disclosure/thread/flow fixes.
  - [x] Integrated first Vol. II late-governance/conclusion concept-audit fix
    packet:
    - Security/Privacy: added compact threat-model fields, LLM tool/RAG
      boundary framing, ML artifact provenance/release-chain framing, and
      incident-response evidence requirements.
    - Robust AI: shortened overlong learning objectives, moved table captions
      before tables, resolved PSI alert vs. retraining wording, and converted
      the adversarial-training listing to framework-neutral literal
      pseudocode.
    - Sustainable AI: moved the dominant lifecycle-term decision procedure
      earlier, shortened learning objectives, and promoted average vs.
      marginal emissions into body prose.
    - Responsible AI: added sensitive-attribute governance choices, validation
      evidence by risk class, monitoring-without-ownership fallacy/pitfall, and
      accountability summary closure.
    - Conclusion: softened GPT-4/Llama 3 training-infrastructure overclaims,
      standardized Llama 3 naming, narrowed tensor-parallel/MoE communication
      language, softened optical/brain synthesis, and added the closing
      constraint-following diagnostic.
  - [x] Focused checks passed for the first Vol. II concept-fix packet:
    `refs --scope inline`, `markup`, `prose`, `punctuation`, `numbers`,
    `structure`, `tables`, `listings`, `git diff --check`, and a five-chapter
    Vol. II PDF render for `security_privacy`, `robust_ai`, `sustainable_ai`,
    `responsible_ai`, and `conclusion`. The partial render still reports
    expected unresolved cross-references to omitted chapters; full-volume gates
    remain pending.
- [~] Run late-stage LEGO prose-boundary cleanup:
  - [x] Checked the flagged GPT-3 "at least" duration case: the LEGO export now
    computes only the rounded duration, while the table prose supplies the
    narrative lower-bound qualifier.
  - [x] Removed the current obvious prose-bearing `MarkdownStr` export in
    Vol. I `appendix_data`: the KL-drift scenario prose now lives in Markdown,
    while LEGO exports typed percentage strings and structural math/vector
    strings only.
  - [ ] Revisit current and prior LEGO cells for similar prose leakage once the
    quantitative edits stabilize.
- [~] Run late-stage LEGO header-comment cleanup:
  - [x] Scanned Vol. I and Vol. II QMD headers for stale `Imports:`/`Exports:`
    inventory lines; none remain in the chapter source.
  - [x] Removed stale formatter comments that still described already-migrated
    typed formatter outputs as `MarkdownStr` escape hatches.
  - [ ] Replace any remaining import/export inventory comments with concise
    context, goal, and how/derivation notes.
  - [ ] Keep "Show" phrasing approximate and narrative-facing rather than
    pinning exact values that may distract future reviewers or LLM passes.
- [ ] After prose-changing work stabilizes, reread all `.claude/rules` and run
  an explicit prose-style compliance pass over touched prose, fixing style,
  voice, pedagogy, progressive-disclosure, and rule-consistency issues.
- [ ] Begin vol1+vol2 capitalization pass after quantitative and
  disclosure-sensitive prose stabilizes.
- [x] Check appendix acronym/framework capitalization such as D-A-M/D.A.M.R.
  and make sure formal framework labels are treated consistently without
  gratuitous capitalization in ordinary prose.
- [x] Replace or justify direct raw hyperlinks in appendix notes, including the
  Jeff Dean/interactive-latency note, preferring bibliography references when a
  stable citable source exists.
  - [x] Replaced the raw interactive-latency URL with a citation to
    `@scott2012latency`; staged the entry in
    `appendix_machine_scott_latency_ref.bib`, ran `betterbib sync --in-place`,
    reviewed the cleaned entry, smoke-tested it with BibTeX, copied only the
    reviewed entry into Vol. I `references.bib`, and cleaned staging artifacts.
- [x] Remove the reader-facing `Volume II` subtitle from the Vol. II title page
  if it appears under `Machine Learning Systems at Scale`.
- [ ] Run a dedicated column-margin figure placement/narration audit:
  - [ ] Verify every placed margin figure is near the narration it supports.
  - [ ] Verify each margin figure is useful in that location, not decorative.
  - [ ] Verify captions, alt text, and surrounding prose make the learner-facing
    connection without over-explaining the miniature visual.
- [ ] Run a dedicated footnote appropriateness/progressive-disclosure audit:
  - [ ] Verify each footnote is useful where placed.
  - [ ] Verify each footnote assumes only concepts introduced earlier in the
    book or earlier in the same chapter.
  - [ ] Fix local wording issues and queue authorial decisions separately.
- [ ] Run a late-stage Volume II SVG polish pass after text/layout work,
  including the cited gray-background/soft-rendered diagrams, rectangular arrow
  cleanup, and consistency with the sharper existing SVG style.
- [x] Redraw the Vol. II Fleet Stack, AI Triad, conclusion Fleet Stack,
  reward-hacking loop, and layers-of-responsibility body figures as clean,
  crisp SVGs that match the book visual language and avoid soft/gray
  background styling.
- [x] Replace/update the benchmarking chapter datacenter-power image using the
  user-supplied source image at `/Users/VJ/Downloads/figure5a_full.png`, making
  sure the surrounding caption/prose accurately explain the updated figure.
  - [x] Verified the repo asset
    `book/quarto/contents/vol1/benchmarking/images/png/mlperf_power_datacenter.png`
    is byte-identical to the supplied image (`sha256`
    `4911034de27a5f768b8d0103f124b6a252d7e98c144608589034986c764b6bbc`), so
    no image-copy churn was needed. Focused figure and image checks passed for
    the benchmarking chapter.
- [x] Review the continuous-batching worked analysis in the serving chapter and
  convert it to a cleaner example/callout style if that improves the learning
  flow and print layout.
- [ ] Run post-text, pre-build artifact explanation audits in canonical Vol. I
  then Vol. II chapter order, preserving pedagogical sequence and progressive
  disclosure:
  - [ ] Figures: verify rendered object accuracy, caption accuracy, and enough
    surrounding prose explanation for a learner to connect what is shown to the
    chapter claim without over-explaining every visual detail.
  - [ ] Tables: verify columns/rows/units/caption match the rendered content and
    surrounding prose explains the table's instructional purpose.
  - [ ] Equations: verify symbols, units, precision, derivation context, and
    surrounding prose connect the equation to the current teaching step.
  - [ ] Algorithms: verify pseudocode/rendered algorithm steps match the prose
    claim, prerequisites have been introduced, and the caption/lead-in explain
    the algorithm at textbook depth.
- [x] Codify reusable `.claude` audit guidance/commands for these artifact
  types so future work can invoke `/audit <type>`-style checks grounded in
  rendering, value/precision validation, prose relevance, SSOT rules, and
  chapter-order progressive disclosure.
  - [x] Verified `AIConfigs` commit `7cade0b` adds
    `.claude/skills/audit-book-artifacts/SKILL.md` and
    `.claude/workflows/audit.js`; the skill covers `lego`, `figures`,
    `tables`, `equations`, `algorithms`, and `listings` in canonical chapter
    order with progressive-disclosure constraints.
- [ ] Final local release gates:
  - [ ] Build Volume I HTML locally.
  - [ ] Build Volume I PDF locally.
  - [ ] Build Volume I EPUB locally.
  - [ ] Build Volume II HTML locally.
  - [ ] Build Volume II PDF locally.
  - [ ] Build Volume II EPUB locally.
- [ ] Run `./book/binder check all --quiet` on fresh final artifacts.
- [ ] Run final pre-commit gates.
- [ ] Push local `dev` to `origin/dev`.
- [ ] Monitor the online workflow every 5--10 minutes after pushing and keep
  fixing failures until the workflow is green.
- [ ] Final volume-level render/debug gates for vol1+vol2 with zero missing
  refs, missing figures, or build errors.
- [ ] Collect authorial, spelling, and SECID decision packets.

## Operating Rules

- Follow `.claude/rules` before editing. In this checkout, `.claude` is an
  ignored local symlink to
  `/Users/VJ/GitHub/AIConfigs/projects/MLSysBook/.claude`, which is the active
  rule source; `.claude/_rules` is absent.
- Keep `/Users/VJ/GitHub/MLSysBook` as the permanent main reference checkout.
- Do not push unless the user explicitly asks.
- Commit incrementally, by one logical batch or one file at a time.
- Every commit that completes or advances a task must update this ledger in the
  same commit, including the relevant checkbox/status and the commit message or
  SHA once known, so review can map work directly to commits.
- For any new or changed bibliography entry, use the BetterBib-first staging
  workflow from `.claude/rules/bib-check.md`: create a named staging `.bib`,
  run `betterbib sync --in-place`, review the cleaned entry against the
  canonical source, smoke-test the staged key with BibTeX, copy only the
  reviewed entry into the target `references.bib`, run project bib/ref checks,
  and delete staging artifacts. Do not type or paste raw metadata directly into
  a volume bibliography.
- Standing review patterns from user feedback:
  - Replace raw URLs with bibliography-backed references when a stable source
    exists.
  - Watch for date-sensitive or product-specific claims and either source-check
    them or frame them explicitly as scenarios/point-in-time examples.
  - Keep LEGO cells computational: calculations, typed units, and formatted
    quantities belong in LEGO; narrative qualifiers belong in prose.
  - Avoid import/export inventory comments in LEGO cells; prefer concise
    context, goal, and derivation comments.
  - Use reader-facing decimal units such as GB unless binary units are the
    teaching point; internal calculations may use binary units when appropriate.
  - Track repeated formatter shapes such as FPS and add new `fmt_*` helpers only
    with a corpus applicability, output, prose, and precision pass.
  - Keep SVGs crisp, white/transparent, and visually consistent with existing
    clean diagrams; avoid gray-background soft-rendered figures.
  - Ensure figures, tables, equations, algorithms, captions, and nearby prose
    explain the learner-facing point without over-explaining every detail.
- Do not blindly trust scripts, prior vol2 edits, or audit findings. Inspect
  the implementation, run the check, and verify the result.
- Apply small mechanical fixes autonomously. Queue authorial or structural
  decisions in a decisions packet for user review.
- Maintain this task list as new tasks are added, and reorder items by
  dependency rather than by arrival time.
- Parallelize only read-only audits that can be scoped cleanly by chapter,
  volume, or file family. Give any delegated audit the relevant `.claude/rules`
  and treat the result as evidence to verify centrally. Keep edits, rule
  conflict resolution, and commits centralized.

## Last Commit Batch

Committed as `c1a76e5aa1`: `Clean stale MarkdownStr formatter comments`

Tasks advanced in this batch:

- Removed stale `MarkdownStr` escape-hatch comments and unused imports from the
  touched Vol. I and Vol. II QMD files where typed formatters already own unit,
  percent, count, multiplier, time, currency, memory, and scientific-notation
  display.
- Moved Vol. I `appendix_data` KL-drift scenario prose out of LEGO and into the
  surrounding worked-example Markdown; the LEGO cell now exports typed percent
  values plus structural vectors/equations.
- Replaced an ad hoc `MarkdownStr(f"{elements:.1e}")` attention-memory output
  in Vol. I `nn_architectures` with `fmt_sci`.

Focused checks passed for this batch:

- `lego-prose-literals`, `lego-dead-code`, `math prose-contract`, `numbers`,
  `refs --scope inline`, `markup`, `prose`, `punctuation`, `git diff --check`,
  partial Vol. I PDF render for the touched chapters, and partial Vol. II PDF
  render for the touched chapters. The partial renders reported expected
  unresolved cross-references to omitted chapters; the full-volume gates remain
  pending.

## Previous Commit Batch

Committed as `5ce80f1e18`: `Fix memory capacity display units`

Tasks advanced in this batch:

- Applied the branded-memory-capacity display policy to the concrete
  reader-facing GiB leaks found in Vol. I `training`, Vol. I `frameworks`, and
  Vol. II `appendix_assumptions`.
- Verified `fmt_fps` is present, tested, and used by the flagged camera FPS
  export.
- Verified the flagged GPT-3 "at least" case keeps the qualifier in prose
  rather than embedding it in a LEGO `MarkdownStr`.
- Verified no stale LEGO `Imports:`/`Exports:` header inventory lines remain in
  Vol. I or Vol. II QMD source.
- Focused checks passed: `lego-prose-units`, `lego-prose-literals`,
  `lego-dead-code`, `math canonical`, `math prose-contract`, `numbers`,
  `refs --scope inline`, `markup`, `prose`, `punctuation`, `git diff --check`,
  a partial Vol. I PDF render for `training,frameworks`, and a partial Vol. II
  PDF render for `appendix_assumptions`.

## Earlier Commit Batch

Committed as `8b588a394f`: `Integrate vol2 governance concept fixes`

Tasks advanced in this batch:

- Integrated the first Vol. II late-governance/conclusion concept-audit fix
  packet and rendered the touched Vol. II chapters in PDF context.

## Earlier Commit Batch 2

Committed as `ac5f179cf6`: `Complete concept audit ledger`

Tasks advanced in this batch:

- Marked the Vol. I and Vol. II per-chapter concept-coverage audit complete.
- Added the compact concept-audit integration queue and the standing review
  patterns from user feedback.

## Earlier Commit Batch 3

Committed as `b403e2cd3a`: `Record vol2 concept audit progress`

Tasks advanced in this batch:

- Recorded Vol. II concept-audit progress through the first late-stage batch
  before the remaining chapter agents completed.

## Earlier Commit Batch 4

Committed as `ca7a783f9b`: `Record benchmarking image verification`

Tasks advanced in this batch:

- Verified that the user-supplied benchmarking datacenter-power image at
  `/Users/VJ/Downloads/figure5a_full.png` is byte-identical to the current
  chapter asset.
- Marked the benchmarking image replacement/update task complete without
  unnecessary binary churn.

## Earlier Commit Batch 5

Committed as `c8af0fa6ac`: `Fix audit follow-up diagrams and references`

Tasks advanced in this batch:

- Raw appendix hyperlink replaced with bibliography-backed citation through the
  BetterBib staging workflow.
- Vol. II title page `Volume II` subtitle removed.
- Vol. II copyedit PDF appendix order aligned with canonical appendix order.
- Vol. II inference continuous-batching analysis converted to a cleaner
  callout-style worked example.
- Vol. II Fleet Stack, AI Triad, conclusion Fleet Stack, reward-hacking loop,
  and layers-of-responsibility SVGs redrawn and rendered locally for visual QA.
- Vol. I concept-coverage read-only audit completed; Vol. II concept-coverage
  read-only audit started in canonical chapter order.

Focused checks run before commit:

- `./book/binder check bib --path book/quarto/contents/vol1/backmatter/references.bib`
- `./book/binder check refs --scope inline --path book/quarto/contents/vol1/backmatter/appendix_machine.qmd`
- `./book/binder check footnotes --path book/quarto/contents/vol1/backmatter/appendix_machine.qmd`
- XML parse of all five touched SVGs.
- Local PNG render/visual QA with `rsvg-convert` for all five touched SVGs.
- `./book/binder check figures` for Vol. II introduction, responsible AI, and
  conclusion chapters.
- `./book/binder check markup` and inline refs for Vol. II inference.
- `git diff --check`

## LEGO + MLSysIM QA Method

Validated 2026-06-11 in this worktree.

Rules read before defining the lane:

- `.claude/rules/mlsysim.md`
- `.claude/rules/fmt.md`
- `.claude/rules/lego-units.md`
- `.claude/rules/lego-verify.md`
- `.claude/rules/lego-prose-literals.md`
- `.claude/rules/numbers-and-math-in-prose.md`
- `.claude/rules/book-prose.md`
- `book/tools/audit/fmt/README.md`

Checker-correctness gate:

- Use `python3`, not `python`; this shell has no `python` shim.
- Use `PYTHONPATH=mlsysim` for all MLSysIM-aware tests and binder checks.
- Targeted checker suite passed:
  `PYTHONPATH=mlsysim python3 -m pytest book/tests/test_lego_prose_units.py book/tests/test_fmt_prose_contract.py book/tests/test_binder_lego_scope_paths.py book/tests/test_math_canonical.py book/tests/test_fmt_semantic_suffix.py book/tests/test_lego_dead_code.py book/tests/test_mlsysim_registry_coverage.py -q --no-cov`
  - Result: 46 passed in 0.69s.
  - Without `PYTHONPATH=mlsysim`, registry tests import the wrong namespace and
    fail; that is an environment issue, not a content finding.
  - `--no-cov` is needed for this targeted lane because repo pytest defaults
    enforce whole-`book/tools` coverage.

Static per-file gates:

- `PYTHONPATH=mlsysim ./book/binder check code --scope lego-prose-units --path <qmd> --json`
- `PYTHONPATH=mlsysim ./book/binder check code --scope lego-prose-literals --path <qmd> --json`
- `PYTHONPATH=mlsysim ./book/binder check code --scope lego-dead-code --path <qmd> --json`
- `PYTHONPATH=mlsysim ./book/binder check math --scope canonical --path <qmd> --json`
- `PYTHONPATH=mlsysim ./book/binder check math --scope prose-contract --path <qmd> --json`
- `PYTHONPATH=mlsysim ./book/binder check registry --scope sources --path <qmd> --json`

Smoke-test results:

- `vol1/frameworks.qmd`: path-scoped LEGO prose-units, fmt prose-contract, and
  registry-sources checks passed.
- `vol1/ml_systems.qmd`: path-scoped LEGO prose-literals, LEGO dead-code, math
  canonical, and registry-sources checks passed.

Rendered LEGO verification lane:

- Rendered-prose audits require archived HTML under
  `book/quarto/_build/html-audit/<vol>/<chapter>.html`; static checks alone are
  not enough.
- Preferred full chapter command:
  `PYTHONPATH=mlsysim ./book/tools/audit/verify_lego_chapter.sh <vol> <chapter>`
- Representative run completed for `vol1/ml_systems`:
  - Binder HTML build: PASS.
  - LEGO cells: 36/36 PASS.
  - Rendered inline references: 350/350 PASS.
  - LLM prose coherence gate: PASS.
  - Certificate:
    `book/tools/audit/artifacts/lego_chapter_reports/vol1_ml_systems_certificate.md`.
- Representative run completed for `vol2/introduction` after the GPT/TPU SSOT
  patch:
  - Binder HTML build: PASS.
  - LEGO cells: 12/12 PASS.
  - Rendered inline references: 61/61 PASS.
  - LLM prose coherence gate: PASS.
  - Certificate:
    `book/tools/audit/artifacts/lego_chapter_reports/vol2_introduction_certificate.md`.
  - Committed as `444c42778a`:
    `Source introduction scale anchors from MLSysIM`.
- The verification script updates audit artifact ledgers and creates scratch
  reports under `book/tools/audit/artifacts/`. Treat these as evidence; do not
  commit generated artifact churn unless explicitly needed for the audit record.

## LEGO Locality Pass

2026-06-11 current state:

- Volume I focal-locality verifier is clean:
  `PYTHONPATH=mlsysim python3 book/tools/audit/lego_focal_verify.py book/quarto/contents/vol1`
- Volume II initially had seven focal-locality/span findings:
  - `compute_infrastructure.qmd`: `ReticleLimitRecap`
  - `data_storage.qmd`: `StorageHierarchyTable`
  - `distributed_training.qmd`: `Gpt3ActivationBudget`
  - `fleet_orchestration.qmd`: `FleetTopologyInterconnect`
  - `inference.qmd`: `HardwareSetupSharding`
  - `ops_scale.qmd`: `TwoSigmaAlerts`
  - `sustainable_ai.qmd`: `TrainingEmissions`
- Resolution:
  - Added explicit `# │ Scope: chapter-anchor` rationale to deliberate running
    examples/callback anchors: reticle limit, GPT-3 activation budget,
    fleet interconnect hierarchy, two-sigma alert-volume example, and
    training-emissions example.
  - Split convenience reuses into local consuming cells/exports:
    `ImageNetNvmeLatency.nvme_latency_us_str` in `data_storage.qmd` and
    `TpSpeedupCalc.h100_mem_gb_str` in `inference.qmd`.
- Verification after edits:
  - Volume II focal-locality verifier: clean.
  - `audit_prose.py --flagged-only` on all seven touched files: clean.
  - Volume II binder gates passed: `lego-prose-units`, `lego-prose-literals`,
    `lego-dead-code`, `math canonical`, `fmt prose-contract`,
    `registry sources`, and `refs inline`.
  - `git diff --check` on touched QMDs and this ledger: clean.

## Immediate Recovery

- [x] Review the two dirty vol2 files before broader edits:
  - `book/quarto/contents/vol2/ops_scale/ops_scale.qmd`
  - `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd`
- [x] Decide whether the dirty edits should be preserved, repaired, committed,
  or discarded. Do not discard user work without explicit instruction.
- [x] Clean stale Gemini-audit ledger bookkeeping for work already merged.
- [x] Classify current audit state: completed, open, blocked, and authorial.

### Recovery Notes

- `ops_scale.qmd` recovered edits are focused percent/formatter cleanup.
  Verified clean with:
  - `PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_prose.py`
  - `./book/binder check math --scope canonical --path ...`
  - `./book/binder check numbers --path ...`
  - `python3 book/tools/audit/book_check_lego_prose_units.py ...`
  - `./book/binder check code --scope lego-prose-literals --path ...`
  - `./book/binder check code --scope lego-dead-code --path ...`
  - `./book/binder check math --scope multiplier-style --path ...`
  - `python3 book/tools/scripts/maintenance/validate_inline_refs.py --path ...`
- `sustainable_ai.qmd` recovered edits are prose/math-typography changes.
  Verified clean with the same focused `audit_prose`, canonical math, numbers,
  LEGO prose-units, LEGO prose-literals, LEGO dead-code, multiplier-style, and
  inline-reference checks.
- `book/tools/audit/fmt/PLAN.md`, `MIGRATION.md`, and `ASSESSMENT.md` are
  referenced by `.claude/rules/fmt.md` but are absent in this worktree. The
  available formatter docs are `book/tools/audit/fmt/README.md` and
  `book/tools/audit/fmt/PLAN_design_b_rates.md`.
- Committed recovered QMD cleanup as `e6969fa554`:
  `Fix recovered vol2 formatter cleanup`.
- Current worktree status after the recovery commit: only this untracked task
  ledger is dirty.
- 2026-06-11 bookkeeping reconciliation:
  - Confirmed merge `62b11a9769` is an ancestor of the current branch and already
    contains the Volume II LEGO/registry sub-phase plus numbers pass.
  - Updated external audit review files under
    `/Users/VJ/GitHub/AIConfigs/projects/MLSysBook/.claude/_reviews/audit_campaign_2026-06/`
    so the Vol. II LEGO/registry worklist is explicitly historical rather than
    active.
  - Reclassified 16 stale Wave 1 Vol. II entries from `deferred` to `fixed`
    where current source now demonstrably uses merged registry/LEGO anchors
    (R1 HBM FIT, R2 A100 MIG profiles, R3 edge benchmarks, R4 storage prices,
    R5 crypto/TEE anchors, R6 A100 embodied-carbon source, R8 RAI overhead).
  - Left four Wave 1 Vol. II registry/LEGO-ish items open because current source
    still supports the open status: introduction GPT/TPU/Meta RSC SSOT,
    performance-engineering compilation-dividend operands, sustainable-AI edge
    embodied-carbon scenario, and the modeled ops-scale TCO sensitivity table.
  - AIConfigs already had an unrelated dirty file:
    `projects/MLSysBook/.claude/rules/auto-layout.md`; this pass did not touch
    it.

## Audit Campaign Classification

The Gemini audit ledgers are useful evidence but not a direct to-do list. They
were created against frozen SHA `195f246`; later work merged many findings,
and registry state has changed since then. Treat them as a queue source, then
verify each item against current text, the local LEGO cell, and `.claude/rules`
before editing.

Machine-derived current ledger totals:

- Wave 1: 510 items — 318 fixed, 88 deferred, 81 queued, 23 rejected.
- Wave 2: 193 items — 52 fixed, 22 deferred, 102 queued, 17 rejected.
- Wave 4: 160 items — 72 fixed, 34 deferred, 47 queued, 7 rejected.

Open/deferred rough buckets after deriving volume from file paths and item IDs:

- Vol1: 117 open/deferred items.
  - Authorial/careful prose: 35.
  - Registry/SSOT: 30.
  - SECID sweep: 15.
  - Problem sets deferred: 14.
  - LEGO/formatter: 12.
  - Numbers/editorial: 4.
  - Capitalization: 1.
  - Other/manual classification: 6.
- Vol2: 257 open/deferred items.
  - Authorial/careful prose: 143.
  - SECID sweep: 29.
  - Registry/SSOT: 21.
  - Capitalization: 20.
  - Problem sets deferred: 20.
  - Numbers/editorial: 15.
  - LEGO/formatter: 9 by simple lens text, but `audit-vol2-DECISIONS.md`
    separately records the real dedicated vol2 LEGO pass as 175 items.

Current queue policy:

- Start with vol1 LEGO/formatter/SSOT because the user explicitly asked for
  vol1 QMD LEGO review first.
- Use ledger items as hints only; re-read the full enclosing section and the
  relevant cell before any edit.
- Leave authorial/careful prose, SECID sweeps, capitalization sweeps, and
  problem-set changes queued unless the quantitative pass directly requires a
  local correction.
- Do not perform serial registry additions in parallel with chapter edits.

## Core Mission

Make every displayed number in MLSysBook LEGO cells correct, precise,
single-sourced, and aligned with the prose that consumes it.

## Vol1 Fast-Pass Results

- Inventory pass found 499 executable class/cell anchors under
  `book/quarto/contents/vol1`.
- `lego_focal_verify.py book/quarto/contents/vol1` found four locality/span
  follow-ups:
  - `benchmarking.qmd`: `InferenceEnergy` spans two sections / 385 lines.
  - `data_selection.qmd`: `ScalingAsymmetry` spans two sections / 4485 lines.
  - `model_compression.qmd`: `BertCompression` spans two sections / 1174
    lines.
  - `responsible_engr.qmd`: `TCOSummary` spans two sections / 811 lines.
- Initial vol1 mechanical gates passed:
  - `./book/binder check code --scope lego-dead-code --vol1`
  - `./book/binder check code --scope lego-prose-literals --vol1`
  - `./book/binder check code --scope lego-units --vol1`
  - `./book/binder check math --scope canonical --vol1`
  - `./book/binder check numbers --vol1`
  - `./book/binder check refs --scope inline --vol1`
  - `./book/binder check registry --scope sources --vol1`
  - `python3 book/tools/audit/fmt/fmt_prose_contract.py --root book/quarto/contents/vol1`
- `audit_prose.py` had a checker gap: it did not resolve indexed inline refs
  such as `{python} Class.field_str[0]`. Patched the resolver and added a
  regression test in `book/tests/test_audit_prose_semantics.py`.
- After the resolver fix, vol1 flagged preview found two real legacy
  `MarkdownStr(f"{x:.1f}")` precision issues in
  `book/quarto/contents/vol1/model_serving/model_serving.qmd`.
  Replaced those outputs with typed `fmt_time(..., precision=None)` and
  `fmt(..., precision=None)` calls.
- Committed the resolver/test and `model_serving` precision cleanup as
  `47f0c9744f`: `Fix vol1 indexed prose audit precision`.
- Current vol1 fast gates are clean:
  - `audit_prose.py --flagged-only` over all 37 vol1 QMDs: 0 flagged files.
  - `audit_prose_semantics.py --root book/quarto/contents/vol1`: clean.
  - `fmt_prose_contract.py --root book/quarto/contents/vol1`: clean.
- Resolved the four initial vol1 `lego_focal_verify.py` locality/span
  failures:
  - `benchmarking.qmd`: split the later GPT-3 training-energy use into a local
    `Gpt3TrainingEnergyAnchor` cell.
  - `data_selection.qmd`: documented `ScalingAsymmetry` as a chapter-anchor
    callback and removed its stale broad imports; localized later hidden imports
    exposed by that cleanup.
  - `model_compression.qmd`: documented `BertCompression` as a chapter-anchor
    running example recalled in the summary.
  - `responsible_engr.qmd`: added `ResponsibleTcoRecap.inf_train_ratio_str`
    and changed the final takeaway to use the summary-local cell instead of
    reaching back to `TCOSummary`.
- Full `lego_focal_verify.py book/quarto/contents/vol1` now passes: 20/20
  chapter/backmatter QMDs with inline Python are clean.
- Committed the vol1 locality/dependency batch as `1da745e582`:
  `Fix vol1 LEGO locality anchors`.
- `all_ai_models` no longer appears in the repository outside this live task
  ledger.
- Area/flux formatter status: `mlsysim.fmt` already provides `fmt_area` and
  `fmt_heat_flux`; vol1 currently uses `fmt_area` for the H100 die-area example.
- Began targeted `MarkdownStr` cleanup where typed formatters are clearly
  available. In `model_compression.qmd`, `FallaciesAnalysis` now uses
  `fmt_memory`, `fmt_percent`, and `fmt` for MB, percent, and count outputs
  instead of plain literal `MarkdownStr` values.
- Committed the focused `model_compression.qmd` formatter cleanup as
  `bded1aacc1`: `Clean model compression fallacy formatters`.
- Replaced four `FalsePositiveTarget` time/count `MarkdownStr` literals in
  `data_engineering.qmd` with typed `fmt`/`fmt_time` calls and committed as
  `3886f4b44f`: `Clean data engineering time formatters`.
- Replaced several clear numeric `MarkdownStr` escape hatches in
  `model_serving.qmd` with existing memory, rate, time-range, and range
  formatters and committed as `799c114875`:
  `Clean model serving formatter ranges`.
- Replaced `BertRoofline` batch-size `MarkdownStr` conversions in
  `benchmarking.qmd` with `fmt` plus a dedicated `fmt_multiple` export for the
  prose multiplier, committed as `89f4a2c286`:
  `Clean benchmarking batch formatters`.
- Replaced the `TrainingMemoryBytes` optimizer overhead `MarkdownStr` range in
  `appendix_algorithm.qmd` with `fmt_range`, committed as `0d7f02aa41`:
  `Clean appendix algorithm memory range formatter`.
- Replaced `EdgeEfficiencyCalc.cam_fps_str` in `responsible_engr.qmd` with
  explicit `fmt_int`, committed as `b3605be4c5`:
  `Clean responsible engineering FPS formatter`.
- Replaced `JetsonSpecs` hand-built power range strings in `ml_workflow.qmd`
  with registry-backed `fmt_qty_range` calls, committed as `80f832b61c`:
  `Clean ML workflow Jetson range formatters`.
- Replaced `ParadigmSystemsCost.hog_grid_str` in `nn_computation.qmd` with
  `fmt_int` and made the HOG grid side length explicit, committed as
  `67d488e175`: `Clean neural computation HOG grid formatter`.
- Replaced `ThrottlingScenario.duration_min_str` in `ml_systems.qmd` with
  `fmt_time`, committed as `171f531ada`:
  `Clean ML systems throttling duration formatter`.
- Replaced `EdgeSizingFleetTCO.jetson_power_range_str` in `ml_systems.qmd`
  with a registry-backed Orin NX power range, updated the stale constants audit
  finding, and committed as `30ce4e5592`:
  `Fix ML systems Jetson power source`.
- Post-cleanup vol1 fast gates are clean:
  - `audit_prose.py --flagged-only` loop over all vol1 QMDs: clean.
  - `audit_prose_semantics.py --root book/quarto/contents/vol1`: clean.
  - `fmt_prose_contract.py --root book/quarto/contents/vol1`: clean.
  - `lego_focal_verify.py book/quarto/contents/vol1`: 20/20 inline-Python
    chapter/backmatter QMDs clean.
  - Binder vol1 checks clean for `lego-dead-code`, `lego-prose-literals`,
    `lego-units`, canonical math, numbers, inline refs, and registry sources.

## A. LEGO Numeric Correctness

- [x] Inventory all vol1 QMD LEGO cells.
- [ ] For every vol1 LEGO cell, execute/render and inspect the actual output.
- [ ] Tune formatter precision so rendered values do not collapse incorrectly
  such as `0 MB/s` when the meaningful value is `0.2 MB/s`.
- [ ] Verify that each rendered value reads correctly in its prose context.
- [ ] Sign off on every vol1 LEGO cell individually.
- [ ] Repeat the full pass for vol2, without trusting prior `fix/audit-vol2`
  edits.
- [ ] Ensure output strings and variable names follow the established style
  everywhere.
- [~] Fix pending `MarkdownStr` cleanup where values should use typed formatters.
  - [x] Current focused pass removed stale `MarkdownStr` escape-hatch comments,
    dropped unused imports, moved KL-drift prose out of LEGO, and replaced an
    ad hoc attention-element scientific-notation string with `fmt_sci`.
  - [ ] Continue distinguishing legitimate structural `MarkdownStr` uses
    (labels, formulas, table sentinels, registry names) from numeric values
    that should use typed formatters.
- [x] Verify any `all_ai_models` cleanup from the old session is complete.

## B. Formatter Layer

- [ ] Audit every `fmt_` helper for necessary sanity checks.
- [ ] Ensure formatter behavior is intelligent for integer-like values, such as
  rendering `153.0` as `153` where that is the right textbook display.
- [ ] Audit formatter audit scripts/checkers themselves for correctness.
- [ ] Prefer critical recurring checks inside `book/binder` instead of as
  standalone external scripts.
- [ ] Inventory uses of `fmt(` and `fmt_int`.
- [ ] Look for repeated unit-bearing patterns that need custom `fmt_` helpers.
  - [ ] Keep running notes on formatter-helper candidates discovered during
    audits; only add helpers when a repeated value kind benefits from typed
    validation/rendering rather than one-off local formatting.
  - [ ] `deployments/year` appeared once during the debt-priority cleanup.
    Current decision: do not add a new `fmt_rate` unit or formatter yet; use
    `fmt_count(..., label="deployment")` plus prose "per year" unless this
    becomes a recurring rate kind.
  - [ ] Per-GPU-hour energy adders appeared in the ops-scale training-cost
    example. Current decision: keep using `fmt_usd(..., per="GPU-hour")`; do
    not add a dedicated helper unless similar energy-price adders recur.
  - [x] Accelerator marketed capacity drift is already covered by
    `fmt_memory_capacity`; use it for vendor-facing HBM/accelerator capacity
    labels and reserve `fmt_memory` for physical decimal/binary conversion.
- [ ] Specifically check area and flux patterns across both volumes.
- [x] Evaluate or add a `fmt_fps` helper for frame-rate values, especially
  vision/camera prose where patterns like `fmt_int(round(cam_fps))` are really
  formatting frames per second.
- [ ] If a new formatter is added, apply it uniformly across vol1 and vol2.
- [ ] Add or update tests/checks for any formatter behavior change.

## C. Source Of Truth

- [ ] For every pinned value in vol1 LEGO cells, decide whether it belongs in
  MLSysIM registry or as a documented local scenario constant.
- [ ] Preserve local scenario constants when they are genuinely one-off
  pedagogical inputs, such as the settled `$3.50/GPU-hour` example.
- [ ] Promote reusable hardware, model, dataset, system, storage, grid,
  price, workload, or policy facts to the proper MLSysIM home.
- [ ] Repeat source-of-truth review for vol2 during the vol2 LEGO pass.
- [ ] Audit MLSysIM registry/model/scenario code for correctness,
  documentation, and category fit.
- [ ] Record public-release recommendations for drift risks or modeling
  questions instead of silently making authorial decisions.
  - [ ] Vol. II Introduction Meta RSC / TPU ICI anchors:
    the current chapter now sources TPU v4 pod chip count and aggregate compute
    from MLSysIM. Meta RSC and TPU interconnect/topology details still need a
    release decision: promote reusable values into `ReferenceStats`/systems
    when they are book-wide anchors, or keep local chapter constants only when
    they are one-off historical examples with explicit provenance.
  - [ ] GPT-4-class training-FLOP scenario policy:
    the introduction now uses local LEGO assumptions for GPT-4-class GPUs,
    days, and FLOPs because MLSysIM does not currently model an executable
    GPT-4 training scenario. Decide whether to add a non-executable
    `ReferenceStats` anchor, an executable `Scenarios` entry, or keep the
    approximation local to the chapter.
  - [ ] Sustainable AI edge embodied-carbon scenario:
    `sustainable_ai.qmd` currently claims that manufacturing 10,000 specialized
    edge devices adds 1,500--2,000 kg embodied carbon. Existing MLSysIM fields
    source A100/H100 embodied carbon and an ESP32-S3 device carbon value, but do
    not cleanly define the "specialized device" class or a 0.15--0.2 kg/device
    embodied-carbon source. Decide whether to define a sourced ReferenceStats or
    Scenarios anchor, revise the prose to an existing device class, or leave the
    claim as cited narrative with explicit provenance.
  - [ ] Public `Scenarios.*` default-pass policy:
    current exported scenarios include deliberate or accidental default
    failures: `AutonomousVehicle_Waymo`, `FrontierTraining`, `KeywordSpotting`,
    and `MobileAssistant`. Decide whether public scenarios must pass by default;
    if not, add expected-failure metadata/reasons and tests so failing examples
    are explicitly pedagogical rather than surprising API behavior.
  - [ ] `Scenarios` registry contract:
    `Scenarios` is exported and documented as public API, but it is not a
    `Registry`, has no `list()`, and is not covered by the provenance audit.
    Decide whether to make it a real audited registry or soften docs until that
    API exists.
  - [ ] Scenario constraint provenance and evaluation:
    `sla_latency`, `target_accuracy`, and `power_budget` are bare values on
    `Scenario`; only `sla_latency` is currently evaluated. Decide whether SLAs,
    accuracy targets, and power budgets are Tier A sourced facts, illustrative
    assumptions, or evaluator inputs with separate provenance rules.
  - [ ] MLSysIM docs split:
    `mlsysim/docs/zoo/scenarios.qmd` calls its page a scenarios zoo while the
    examples/table are `ReferenceStats` anchors. Split or rename docs so
    executable `Scenarios.*` bundles and non-executable `ReferenceStats.*`
    anchors are not conflated.
  - [ ] Provenance catalog comment cleanup:
    change the behavior-free heading in
    `mlsysim/mlsysim/core/provenance_catalog.py` from "Scenarios registry" to
    "ReferenceStats registry" when making the next MLSysIM doc/comment pass.
- [ ] Review the `ml_workflow.qmd` Jetson Orin Nano memory range
  `4--8 GB`: current prose includes a lower SKU bound while the registry-backed
  inline value covers the 8 GB capacity.
- [ ] Review `Platforms.*` string-backed latency/power ranges such as
  `Platforms.Mobile.tdp_range_w`; several render with hyphen style such as
  `3-5 W`, but the current registry stores them as strings rather than typed
  range endpoints.

## D. LEGO Structure And Locality

- [ ] Audit oversized LEGO cells and macro blocks.
- [ ] Split or move cells so definitions and exported prose values are close
  to first use.
- [ ] Avoid code-level cross-cell dependencies; share reusable values through
  MLSysIM, helper functions, or neutral scenario anchors.
- [ ] Reconsider the inputs/outputs boilerplate at the top of LEGO cells.
- [ ] Recommend a cleaner documentation pattern that explains cell goals and
  calculations without wasted text.
- [ ] Keep code comments short and useful, only where they clarify the
  calculation.

## E. Prose And Pedagogy After Numbers Stabilize

- [x] Audit the opening Purpose prose at the start of every main chapter and
  ensure it is exactly one paragraph. The likely Chapter 7 case was
  `vol1/frameworks/frameworks.qmd`; its split Purpose was normalized to one
  tighter paragraph and committed as `a6d99824f1`:
  `Fix Frameworks purpose opener layout`. A focused audit now shows all 33 main
  chapter opening Purpose blocks are one paragraph.
- [x] Build every main chapter PDF page opening and inspect the rendered first
  page: the single-paragraph Purpose must fit on page 1, and the learning
  objectives should begin at the top of page 2.
- [x] If rendered first-page layout has extra space, consider expanding the
  existing Purpose point in-place while keeping it one paragraph; do not expand
  when the layout is already right. No expansion was needed; only `ML
  Frameworks` required tightening after the one-paragraph merge.
- [x] Full PDF opener audit result: 33/33 main chapters clean across Vol. I and
  Vol. II. For every main chapter, page 1 contains Purpose, page 1 does not
  contain learning objectives, and page 2 contains the learning-objective box.
- [ ] Decide whether to normalize appendix Purpose sections and Vol. 2
  pre-section paragraphs that technically still live under `## Purpose` after
  learning objectives or setup chunks, especially `compute_infrastructure`,
  `ops_scale`, `robust_ai`, `security_privacy`, and `sustainable_ai`.
- [ ] Run a paragraph-flow audit: every paragraph makes a point, connects
  logically, and avoids dangling single-sentence fragments unless intentional.
- [ ] Run a progressive-disclosure audit by chapter: chapter N may assume only
  material up through chapter N.
- [ ] Allow forward references only when they say what will be covered later
  and give the current key point.
- [ ] Run central-thread audits: Purpose, section arc, examples, summary, and
  golden-thread callbacks should support the same teaching claim.
- [ ] Flag opportunities for reflection questions or connective prompts, but
  do not invent authorial content unilaterally.
- [ ] Integrate accepted prose fixes after quantitative edits settle.

## F. Editorial Decision Packets

- [ ] Queue drastic restructures, Purpose/LO rewrites, definition-callout
  changes, and authorial content for user review.
- [ ] Collect authorial/spelling/SECID decisions.
- [ ] Preserve the pretraining vs. pre-training conflict as a decision item
  until the user rules.
- [ ] Recommend dropping the SECID hex-suffix sweep unless explicitly requested.
- [ ] Keep vol2 caps work separate until quantitative and disclosure-sensitive
  prose stabilizes.
- [ ] Later, run vol1 and vol2 capitalization passes chapter by chapter.

## G. Verification And Commits

- [ ] For every touched chapter, run the relevant binder build/debug command
  with verbose output.
- [ ] Fix all missing cross-references, missing figures, build errors,
  tracebacks, and unevaluated `{python}` refs.
- [x] Fixed two Vol. II PDF-render blockers in
  `vol2/security_privacy/security_privacy.qmd` where inline math was split
  across `{python}` refs inside `\sqrt`/`\frac` expressions. Exported full
  `fmt_math` strings from the relevant LEGO cells and committed as
  `9be057a4e0`: `Fix security privacy PDF math fragments`.
- [x] Built full Vol. I and full Vol. II PDFs after the opener/math fixes.
  Both passed post-build PDF text validation with no unresolved refs, no
  `Figure/Table/Section ??`, and no Python tracebacks or warnings in PDF text.
  The remaining reported margin-overflow warnings are non-blocking and outside
  the chapter-opener task.
- [ ] Run relevant checks before commits, including math/code/registry/prose
  scopes as appropriate.
- [ ] Run pre-commit before each commit.
- [ ] Commit incrementally by file or coherent batch.
- [ ] Final gate: vol1 and vol2 volume-level render/debug checks are clean.

## Current Known Dirty Files

- Generated audit artifacts under `book/tools/audit/artifacts/` are expected to
  be dirty/untracked during this pass and should not be committed unless they
  become part of a deliberate audit artifact update.
