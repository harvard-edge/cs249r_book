# Claude Resume Handoff: Full Book Audit And Sign-Off

This file is a resume artifact for continuing the MLSysBook full-audit/sign-off
pass. It captures the exact state of the work, the user intent, the worktree
policy, the automated checks already run, subagent coverage, findings received,
and the next integration plan.

## Refresh Note — 2026-06-13

This handoff was refreshed after the percent/LEGO formatter-lane cleanup and
the inline-Python checker repair.

Current functional state at refresh time:

- Permanent checkout: `/Users/VJ/GitHub/MLSysBook` on `dev`
  (`b3a6a725c9de1365178bd87825d3adb59a27512d`), with an unrelated dirty
  `mlsysim/mlsysim/literature/registry.py` that this pass did not touch.
- Active worktree: `/Users/VJ/GitHub/MLSysBook-full-audit-signoff`.
- Active branch: `codex/full-audit-signoff`.
- Current functional head before staging this refreshed handoff file:
  `9f9c0d889b Fix inline Python math validation`.
- Previous commit in this pass:
  `83f816f38e Harden percent formatter prose contract`.
- MLSysBook tracked status was clean except for this untracked handoff file.
- The `.claude` directory in MLSysBook is a symlink to
  `/Users/VJ/GitHub/AIConfigs/projects/MLSysBook/.claude`. The selected
  AIConfigs rule update is in `projects/MLSysBook/.claude/rules/fmt.md` and
  should be committed separately from unrelated dirty AIConfigs work.

Checks confirmed after the functional fixes:

```bash
./book/binder check refs --scope inline-python --path book/quarto/contents
./book/binder check math --scope prose-contract --path book/quarto/contents
./book/binder check math --scope canonical --path book/quarto/contents
./book/binder check code --scope lego-dead-code --path book/quarto/contents
./book/binder check math --scope suffix-consistency --path book/quarto/contents
./book/binder check math --scope suffix-semantics --path book/quarto/contents
./book/binder check numbers --scope percent-in-prose --path book/quarto/contents
./book/binder check numbers --scope percent-in-tables --path book/quarto/contents
./book/binder check numbers --scope percent-in-captions --path book/quarto/contents
PYTHONPATH=. pytest -q --no-cov \
  book/tests/test_inline_python_math_spans.py \
  book/tests/test_fmt_prose_contract.py \
  book/tests/test_binder_lego_scope_paths.py
```

All commands above passed. Full pre-commit passed for both functional commits.

## Copy-Paste Resume Prompt

Resume the MLSysBook full audit/sign-off task from:

```bash
cd /Users/VJ/GitHub/MLSysBook-full-audit-signoff
pwd
git rev-parse --show-toplevel
git branch --show-current
git status --short
git worktree list
```

You are on branch `codex/full-audit-signoff`, based on local `dev` after the
notation fixes were merged. Treat `/Users/VJ/GitHub/MLSysBook` as the permanent
main reference checkout. Do not edit or remove it except for explicit tiny
direct fixes requested by the user. Continue work in this sibling worktree.

Goal: perform a full, methodical audit across Volume 1 and Volume 2 in canonical
chapter order, including appendices, and sign off on every chapter only after
the chapter satisfies the Claude rules, progressive disclosure, notation
single-source-of-truth, Purpose shape, emphasis/index ownership, callout/example
scaffolding, algorithm rules, MLSysIM/computed-value discipline, and Binder
coverage. Launch and harvest one parallel agent per chapter/appendix as needed.
Integrate only high-confidence fixes, review whole sections before editing, run
checks, stage touched files, and commit actual text/rule/Binder fixes.

Important: notation work is already merged into local `dev`; do not redo that
branch. AI config edits, if any become necessary, must remain isolated from
unrelated AIConfigs work. In the current state, only
`projects/MLSysBook/.claude/rules/fmt.md` has a selected rule update from this
pass.

## Current State

- Permanent checkout: `/Users/VJ/GitHub/MLSysBook`
- Active audit worktree: `/Users/VJ/GitHub/MLSysBook-full-audit-signoff`
- Active audit branch: `codex/full-audit-signoff`
- Current functional head in this worktree before this handoff refresh:
  `9f9c0d889b Fix inline Python math validation`.
- Local `dev` points at `b3a6a725c9de1365178bd87825d3adb59a27512d`.
- The previous notation worktree `/Users/VJ/GitHub/MLSysBook-notation-audit`
  was merged into local `dev`, removed, and its branch
  `codex/notation-audit-fixes` was deleted.
- The premature pre-merge full-audit worktree was removed and recreated from
  updated `dev`.
- At the time this refresh was written, MLSysBook tracked status was clean
  before this file was staged.
- A previous experimental Binder change for `purpose-shape` was discussed and
  even tested in-session, but it is not present in the current worktree. Verify
  with:

```bash
rg "purpose-shape|_run_purpose_shape" book/cli/commands/validate.py
```

Expected result currently: no matches.

## User Intent And Constraints

The user wants a serious book-closing audit, not a casual scan:

- "Full audit check."
- "Update any checks that need it, the binder, and so forth."
- "Launch parallel agents to make sure we are not missing anything at all."
- "Make sure that if they need anything added, you would add it."
- "Super detailed."
- "I want you to sign off on every single chapter. Your neck is on the line."
- Before that, the user clarified the notation sequence:
  1. Merge notation fixes into `dev`.
  2. Retire the notation worktree.
  3. Enter `dev`.
  4. From there, pick the full audit back up.

That sequence has been done. Resume from the audit worktree.

## Rules Already Read / Must Apply

Read lazily, but these rule areas are active:

- `AGENTS.md`: MLSysBook worktree policy. Never remove
  `/Users/VJ/GitHub/MLSysBook`.
- `.claude/rules/book-prose.md`: editorial spine, progressive disclosure,
  no API rule, quantitative textbook tone.
- `.claude/rules/purpose.md`: Purpose must be one hook plus one plain paragraph,
  no bold/italics/citations/index/math/Python/footnotes in the paragraph,
  teaching chapters close on D.A.M or C3 as appropriate, appendices have no
  margin stack.
- `.claude/rules/notations.md`: shared notation table is the single source of
  truth; appendices are high risk; one symbol, one meaning.
- `.claude/rules/emphasis.md`, `bold.md`, `italics.md`,
  `emphasis-audit.md`, `index.md`: first-definition ownership and index
  placement are volume/global concerns, not chapter-local decoration.
- `.claude/rules/callouts.md`: callout labels, notebook final
  `**Systems insight**:`, war-story label arcs, checkpoints, definition callouts.
- `.claude/rules/algorithms.md`: pseudocode chunk options and algorithm
  skeleton; no blank line between chunk options and `\begin{algorithm}`.
- `.claude/rules/numbers-and-math-in-prose.md`, `math.md`, `fmt.md`,
  `lego-units.md`, `lego-prose-literals.md`, `lego-verify.md`, `mlsysim.md`:
  computed values and prose-facing cells must be sourced and formatted.
- `.claude/rules/qmd-patterns.md`: inline Python must not be embedded inside
  math spans or glued to LaTeX operators.
- `.claude/rules/chapter-architecture.md`: chapter component order,
  fallacies/pitfalls, LOs, summary and connection.

Potential rule/check conflict discovered:

- `callouts.md` allows `.callout-war-story` to use
  `Context / Failure mode / Consequence / Systems lesson`, but
  `./book/binder check markup --all-scopes` currently flags that exact pattern
  as schema mismatch. This appears to be a Binder checker bug or stale schema,
  not a text defect. Fix the checker or reconcile the rule before changing 17
  war-story callouts.

## Automated Checks Already Run

On merged local `dev` before creating the fresh worktree:

```bash
./book/binder check notation --scope consistency
```

Passed.

In `/Users/VJ/GitHub/MLSysBook-full-audit-signoff`:

```bash
./book/binder check all
```

Content checks passed. Only failure was `pdf-verify` because this fresh
worktree did not have built PDF artifacts:

- missing `book/quarto/_build/pdf-vol1/Machine-Learning-Systems-Vol1.pdf`
- missing `book/quarto/_build/pdf-vol2/Machine-Learning-Systems-Vol2.pdf`

Important: do not treat this as a content failure unless PDFs have been built.

Opt-in checks also run:

```bash
./book/binder check refs --all-scopes
./book/binder check labels --all-scopes
./book/binder check footnotes --all-scopes
./book/binder check figures --all-scopes
./book/binder check markup --all-scopes
./book/binder check prose --all-scopes
```

Results:

- `refs --all-scopes`: fails with 259 `python_in_math` errors and 350 warnings.
  This is a real corpus-wide QMD/math issue class, but too broad to fix blindly.
  Prioritize chapter-reported high-value occurrences and add/strengthen checks
  after the corpus is clean enough.
- `labels --all-scopes`: fails `fig-labels` for 12 underscore figure labels.
- `footnotes --all-scopes`: 36 duplicate/cross-chapter footnote warnings.
- `figures --all-scopes`: 590 float-flow warnings, mostly orphan/late floats.
- `markup --all-scopes`: 17 war-story schema errors that likely reflect the
  rule/check conflict described above.
- `prose --all-scopes`: many ASCII warnings and 20 abbreviation-first-use
  warnings; these are opt-in/noisy.

Default checks are clean, so human/agent findings matter.

## Binder / Rule Gaps To Consider Adding

Candidate checks discovered by agents and local inspection:

1. Purpose shape gate:
   - one underscore-italic hook question;
   - exactly one paragraph after hook;
   - no bold, body italics, citations, cross-refs, Python, math, footnotes,
     or index tags in the Purpose paragraph;
   - appendices must not use a margin stack.
   This check was tested in concept and passed the current corpus, but no code is
   present now. Re-add if desired under `structure`.

2. War-story schema:
   - Update Binder to allow the `Consequence` label when it matches
     `callouts.md`, or update the rule if the allowed schema has changed.

3. Notebook scaffold:
   - `.callout-notebook` should use a canonical `Problem` or `Scenario`
     opening, `Math` where it is a calculation, at most one `Result`, and a final
     `Systems insight`.

4. Algorithm skeleton:
   - pseudocode blocks should not have a blank line between chunk options and
     `\begin{algorithm}`;
   - terminating algorithms should have an explicit `\State \Return ...`;
   - no branch should fall through to an invalid return.

5. Inline Python in math:
   - existing opt-in `refs --scope inline-python` catches many issues. Decide
     whether to fix corpus-wide, keep opt-in, or target only release blockers.

6. LEGO cell header/self-contained cells:
   - prose-facing named cells often lack full Context/Goal/Show/How headers or
     rely on previous imports/state. Agents repeatedly flagged this. Current
     Binder checks do not catch enough of it.

7. Generated percent style:
   - `fmt_percent(..., style="symbol")` can pass Binder while rendering `%` in
     prose/captions. Need either stricter call-site checks or separate
     `_pct_symbol_str` vs `_pct_prose_str` convention.

8. First-definition ownership:
   - duplicate bold/index `!definition` ownership across chapters is not caught
     well enough.

9. Progressive disclosure:
   - especially "Day 1" hardware-specific model TFLOP/s before the hardware
     chapter in Volume 1. This is widespread and may require either systematic
     generalization or a rule exception. Do not silently rewrite huge parts
     without deciding the policy.

10. Appendix consult-only policy:
    - consult-only appendices should not include checkpoints unless the appendix
      is a taxonomy/teaching appendix that explicitly allows them.

## Agent Coverage And Status

The following agents were launched. Some completed and were closed; some
completed but still need to be closed; some may still be running.

Use the active HTML render manifests as the canonical edit order when resuming:

Volume 1 body:

1. Introduction
2. ML Systems
3. ML Workflow
4. Data Engineering
5. Neural Network Computation
6. Neural Network Architectures
7. Frameworks
8. Training
9. Data Selection
10. Model Compression
11. Hardware Acceleration
12. Benchmarking
13. Model Serving
14. ML Ops
15. Responsible Engineering
16. Conclusion

Volume 1 appendices:

1. Appendix D.A.M.
2. Appendix Data
3. Appendix Algorithm
4. Appendix Machine
5. Appendix Assumptions
6. Glossary

Volume 2 body:

1. Introduction
2. Compute Infrastructure
3. Network Fabrics
4. Data Storage
5. Distributed Training
6. Collective Communication
7. Fault Tolerance
8. Fleet Orchestration
9. Performance Engineering
10. Inference
11. Edge Intelligence
12. Ops Scale
13. Security Privacy
14. Robust AI
15. Sustainable AI
16. Responsible AI
17. Conclusion

Volume 2 appendices:

1. Appendix D.A.M.
2. Appendix C3
3. Appendix Fleet
4. Appendix Communication
5. Appendix Reliability
6. Appendix Inference
7. Appendix Assumptions
8. Glossary

### Completed And Closed

- `019ec18c-9d30-7053-9267-6dc0bcbd8a8f` - Volume 1 Introduction.
- `019ec18c-9d52-7df3-a25f-0fafd0fcce6a` - Volume 1 ML Systems.
- `019ec18c-9e05-7532-a2cb-e0732711d931` - Volume 1 Frameworks.
- `019ec18c-9e2b-7c93-a057-8669abb082d5` - Volume 1 Training.

### Completed, Findings Received, Close After Harvesting

- `019ec18c-9d73-7a41-a4d1-557c28e1291c` - Volume 1 ML Workflow.
- `019ec18d-4534-7391-b669-8e2a7f6df0e5` - Volume 1 Model Compression.
- `019ec18d-45a5-7793-98e9-218b24d84973` - Volume 1 Model Serving.
- `019ec18d-45cb-7f41-9ff8-47a70e25b18a` - Volume 1 ML Ops.
- `019ec18d-c380-7c00-bbe4-e840a580b2bd` - Volume 1 Appendix Algorithm.
- `019ec18d-c3ba-7070-9a1f-8fd17281a30d` - Volume 1 Appendix Machine.
- `019ec18e-5464-7043-8756-1d9f8982ba4b` - Volume 2 Compute Infrastructure.
- `019ec18e-548b-7b92-bc07-b469b8f21086` - Volume 2 Network Fabrics.
- `019ec18e-54b0-7d30-93c8-6d1d902986ae` - Volume 2 Data Storage.
- `019ec18e-556c-71f2-a940-9740d1f03342` - Volume 2 Fault Tolerance.
- `019ec18e-5596-7ab0-8d56-9028f2d5e188` - Volume 2 Fleet Orchestration.
- `019ec18e-e1d1-76e0-a11a-72815192c7ad` - Volume 2 Inference.
- `019ec18e-e1fb-71e3-a725-ca6f3891fd53` - Volume 2 Edge Intelligence.
- `019ec18e-e225-7b11-8b47-2385c95b7b19` - Volume 2 Ops Scale.
- `019ec18e-e24e-7b83-9e1f-6cf409da266c` - Volume 2 Security Privacy.
- `019ec18e-e27a-7510-b484-6f224e81c6f8` - Volume 2 Robust AI.
- `019ec193-aa9f-7400-9951-2b5615d5bf25` - Volume 2 Sustainable AI.
- `019ec193-aac4-7f92-a70f-96e8218b3e8f` - Volume 2 Responsible AI.
- `019ec18e-e1a8-7a61-a9e8-3ccc1a48b82e` - Volume 2 Performance Engineering.
- `019ec18e-5513-7de1-8b39-bf11bc21e732` - Volume 2 Collective Communication.
- `019ec193-aae8-79a2-a7fa-c1efe0916450` - Volume 2 Conclusion.

### Launched, No Findings Seen Yet In This Thread

Volume 1:

- `019ec18c-9d95-7462-894c-46c738cb628e` - Data Engineering.
- `019ec18c-9dbb-7702-a187-0df7a2e75748` - NN Computation.
- `019ec18c-9ddf-7942-bf4f-3674495db457` - NN Architectures.
- `019ec18d-4512-7533-b0de-c479ed0463a5` - Data Selection.
- `019ec18d-4559-7303-8b6f-fc1da659c831` - HW Acceleration.
- `019ec18d-457e-79a3-b324-27267849cbec` - Benchmarking.
- `019ec18d-45f4-7622-97bb-2bdbbd9b2a34` - Responsible Engineering.
- `019ec18d-463a-7061-861e-f2a918b0b60a` - Volume 1 Conclusion.
- `019ec18d-c333-7521-a7dd-12498f4cf0e9` - Volume 1 Appendix D.A.M.
- `019ec18d-c359-74b2-9225-be9663d30d46` - Volume 1 Appendix Data.
- `019ec18d-c3e1-7691-9203-bfaa726621c1` - Volume 1 Appendix Assumptions.

Volume 2:

- `019ec18e-543f-7133-9244-80fc05aa264e` - Volume 2 Introduction.
- `019ec18e-54d7-7462-b64c-5bc165196065` - Distributed Training.

Volume 2 appendices were not launched before this handoff. They still need one
agent each:

- `book/quarto/contents/vol2/backmatter/appendix_dam.qmd`
- `book/quarto/contents/vol2/backmatter/appendix_c3.qmd`
- `book/quarto/contents/vol2/backmatter/appendix_fleet.qmd`
- `book/quarto/contents/vol2/backmatter/appendix_communication.qmd`
- `book/quarto/contents/vol2/backmatter/appendix_reliability.qmd`
- `book/quarto/contents/vol2/backmatter/appendix_inference.qmd`
- `book/quarto/contents/vol2/backmatter/appendix_assumptions.qmd`

## Findings Received So Far

This section preserves the major line-referenced findings already received.
Treat every finding as a candidate; inspect the full section before editing.
Report sections below reflect harvest order, not strict book order. Apply fixes
in the canonical order listed above.

### Volume 1 Introduction

Status: PASS WITH FIXES.

- Medium: `introduction.qmd:1664`, `:1869` call `L_{\text{lat}}` the overhead
  term even though notation defines it as latency. Use "latency term" and say it
  captures orchestration/networking/serialization delay.
- Medium: `introduction.qmd:1643`, `:2976`, `:2982` split Data Drift definition
  ownership across an index tag, bold body mention, and footnote. Choose one
  owner; likely define once at first teaching site with
  `\index{Data Drift!definition}` and demote later mentions.
- Medium: `introduction.qmd:1873` introduces "autoregressive decode",
  "KV cache", and "prefill" in Chapter 1. Keep high-level memory-bandwidth
  diagnosis; defer those terms to later inference/model chapters.
- Low: `introduction.qmd:641-642` mass index alias block is unanchored to
  visible prose. Decide whether alias banks are allowed or add/check an
  exception.
- Low: `introduction.qmd:33` Purpose parenthetical
  "converges (reaches a stable trained state)" behaves like definition
  machinery. Remove or move gloss to body prose.
- Low/check gap: repeated hard-coded training-compute and efficiency anchors at
  `:1956-1957`, `:2287-2288`, `:2448-2450` may need MLSysIM centralization.

### Volume 1 ML Systems

Status: PASS WITH FIXES.

- High: `ml_systems.qmd:587` reintroduces the Iron Law as a
  `.callout-definition` with `\index{Iron Law!definition}` though Chapter 1
  owns the formal definition. Convert to recap/perspective/plain prose; remove
  definition ownership.
- Medium: `ml_systems.qmd:856` uses
  `$M_{\text{model}} \le C_{\text{mem}}$` before defining those symbols, and
  they are not in frontmatter notation. Define locally before/after first use or
  use prose in the table.
- Medium: `ml_systems.qmd:1190-1191` expose named hardware and exact peak
  compute before hardware chapter. Keep tier-level envelopes; defer exact
  device performance specs.
- Medium: `ml_systems.qmd:2317-2457` Edge inference sizing notebook ends with
  Result/Analysis but no final `**Systems insight**:` endpoint. Add one.
- Low: `ml_systems.qmd:3518`, `:3520` autonomous-vehicle notebook uses
  noncanonical lead-ins. Normalize to `Scenario` and `Analysis` or reclassify.

### Volume 1 ML Workflow

Status: not signed off.

- High: `ml_workflow.qmd:381`, `:1103`, `:1935` put inline Python in or adjacent
  to LaTeX math. `binder check refs --scope inline-python --path ...` fails.
  Precompute complete math strings with `fmt_math`/`fmt_frac`/`MarkdownStr`.
- Medium: many `fmt_percent(..., style="symbol")` values are consumed in prose
  (`:356`, `:358`, `:361`, `:364`, `:365`, `:369`, `:934`, `:935`, `:1092`,
  `:2029-2032`). Use prose style or explicit `percent`.
- Medium: prose-facing cells `JetsonSpecs`, `StorageCosts`, `ImageNetCorpus`
  at `:1115`, `:1147`, `:1564` lack labels/full headers/self-contained imports.
- Medium: progressive disclosure overreaches: `ML operations` at `:178` before
  owning chapter; `federated learning` at `:1141` before owning treatment.
  Use safe substitutes such as "operational practices" and
  "privacy-preserving distributed approaches".
- Low: `fig-alt` at `:229` uses `%` symbols; spell out percent in alt text.
- Low: stage-transition example labels at `:947`, `:953`, `:958`, `:964` drift
  from canonical callout labels.

### Volume 1 Frameworks

Status: not signed off.

- High: `frameworks.qmd:3085`, `:3578`, `:4589` use A100-specific TFLOP/s
  anchors before hardware chapter. Generalize or defer.
- Medium: Purpose `frameworks.qmd:29` is structurally valid but overloaded like
  a coverage summary. Tighten to the stake: framework translation is not
  neutral; it creates performance and lock-in consequences and mediates
  algorithm-machine co-design.
- Medium: Purpose hook `frameworks.qmd:27` uses `---` dash syntax. Rewrite as a
  quieter single question without the dash.
- Medium: `frameworks.qmd:2056` has a blank line between pseudocode chunk
  options and `\begin{algorithm}`. Delete it.
- Low: `frameworks.qmd:1945` opens a `##` with formal definition while concrete
  motivation sits before the heading. Move/add concrete opener under heading.
- Low: `frameworks.qmd:2576` italicizes "inherently more expensive" in a
  checkpoint. Remove italics unless a real contrast pair is needed.

### Volume 1 Training

Status: not signed off.

- High: hardware progressive-disclosure violation across the chapter. Training
  precedes hardware acceleration but names V100/A100/H100 TFLOP, Tensor Core,
  warp, NVLink, HBM, H200, and TDP specifics at examples including `:574`,
  `:590`, `:1485`, `:2586`, `:4199`, `:6476`. Decide policy before large edits:
  either move hardware chapter earlier, generalize these to accelerator-class
  examples, or add a formal rule exception for reference accelerators.
- Medium: Iron Law definition at `training.qmd:227` invokes future model
  compression terms "pruning and distillation" before model compression. Replace
  with "algorithmic reductions to total operations" or similar.
- Medium: Adam API listing `:1083-1119` conflicts with No API rule unless
  explicitly framed as an interface example after framework-independent loop.
- Medium: Adam pseudocode `:1127-1141` lacks `\State \Return`; comments are
  mostly math labels. Add return and/or rewrite comments for systems meaning.
- Low: notebook drift: GELU notebook `:748-769` and optimizer memory notebook
  `:1045-1071` need canonical scaffold or reclassification.

### Volume 1 Model Compression

Status: not signed off.

- High: duplicate Model Compression definition/index ownership at
  `model_compression.qmd:95`, `:131`, with earlier Introduction ownership at
  `introduction.qmd:2908`. Prefer this chapter as owner; demote introduction and
  remove duplicate line 95 bold/index.
- High: A100/V100 hardware-specific throughput and bandwidth before hardware
  chapter at `:3477-3523`, `:5262-5271`. Generalize or move.
- Medium: `:264`, `:273` lean on later appendix material as support. Keep local
  energy table as support and recast appendix refs as optional later pointers.
- High/rule gap: prose-facing Python cells at `:52`, `:103`, `:208` are
  unlabeled; compact-header cells at `:3001`, `:5191`, `:5255` rely on prior
  imports/state. Add labels/headers/imports and consider Binder check.
- Medium: PyTorch snippets at `:558-575`, `:4005-4034` teach mechanisms through
  API-specific code. Convert to pseudocode/neutral tensor operations unless the
  API itself is the lesson.
- Low: callout labels `:436-438` use `**Region 1: The free lunch**:` style.
  Use colon outside bold label.

### Volume 1 Model Serving

Status: not signed off.

- High: serving notation conflicts with frontmatter. `model_serving.qmd:2155`
  says the book uses `L_{\text{lat,*}}` for latency components, but frontmatter
  reserves `L_{\text{lat}}` for Iron Law fixed latency and defines serving
  response time as `T_{\text{lat}}`. Drift also at `:2279`, `:2283`, `:2285`,
  `:2528`, `:2949-2955`, `:3185-3186`, `:3330`, `:3452`, `:3609-3610`.
- High: batch service notation not canonical. Use `T_{\text{svc}}(B)` instead
  of `T_{\text{service}}(B)` at `:3065`, `:3069`, `:3186`, `:3330`,
  `:3609-3610`; line `:2514` uses `T_\text{service}`.
- Medium: many generated percent values render `%` in prose/captions. Split
  symbol/prose outputs or use prose style.
- Medium: hard-coded mutable pricing data at `:257-267`, `:277-279`,
  `:4510`, `:4535-4540`, `:4566`. Move to MLSysIM/snapshot registry or label as
  illustrative.
- Medium: several notebooks do not follow scaffold at `:1753-1759`,
  `:2942-2946`, `:3166-3177`, `:3711-3745`, `:3823-3854`.
- Low: index key `Little's Law!L=lambda W formula` at `:2145` contains formula
  notation. Use plain concept text.
- Low: `Headroom` first-definition bold/index appears inside a perspective
  bullet at `:60`; move formal definition to body or make bullet plain.

### Volume 1 ML Ops

Status: not signed off.

- High: `ml_ops.qmd:3356` embeds inline Python in display math. Compute full ROI
  equation string in preceding cell with `fmt_display_math`.
- High: many LEGO cells rely on earlier notebook state/imports: examples include
  `fmt_time` at `:344/:380`, `fmt_percent` at `:1236/:1269`,
  `fmt_display_math` at `:1310/:1355`, formatters at `:1763/:1810`,
  units/registries at `:2026/:2047`, `:2081/:2112`, `:2277/:2300`.
- Medium: retraining interval notation at `:1396` recurs at `:3997`, `:4054`,
  `:4064` without local redefinition. Promote or add local reminders/prose.
- Medium: `fig-alt` at `:2959` uses escaped currency `\$500`, `\$100`. Use
  accessible text such as "500 dollars".
- Low: percent ranges use dash style at many lines; ordinals as digits at
  `:1572`, `:1574`, `:1582`; rhetorical questions at `:279`, `:1306`,
  `:1566`, `:2075`, `:3107`; "lighthouse fraud model" at `:1379` should be
  "fraud-detection model" unless canonical lighthouse.

### Volume 1 Appendix Algorithm

Status: not signed off.

- Major: inline Python expressions embedded inside display/math at
  `appendix_algorithm.qmd:98`, `:410`. Compute whole equation string or move
  computed values to prose.
- Medium: calculation cells at `:63`, `:119`, `:295`, `:319`, `:451` do not
  consistently follow LEGO/header contract or self-contained imports.
- Medium: notebook at `:392`, `:394`, `:396`, `:406`, `:413` lacks canonical
  scaffold and systems insight.
- Medium: checkpoint at `:425` appears in consult-only appendix and uses numbered
  questions rather than checkpoint checkbox syntax. Remove or convert to prose.
- Medium: `:204` `**virtual**\index{Virtual broadcasting}` mismatches
  emphasis/index. Use full concept or strip.
- Medium: footnote definitions at `:41`, `:61`, `:238`, `:421` lack index tags
  at bold term heads; use appendix refresher subentries if body owns definition.
- Minor: `:232` index key should use established automatic differentiation key.
- Minor: `:161` uses italic pseudo-label `*Result*:`; change to bold structural
  label or prose.
- Minor: `:28` appendix ref should target specific numerical representations
  subsection.
- Minor: Purpose hook `:9` is a "What" question; purpose rule prefers "Why does".

### Volume 1 Appendix Machine

Status: not signed off.

- High: consult-only appendix checkpoint at `appendix_machine.qmd:723`; remove or
  move to teaching chapter.
- High: Little's Law throughput expression at `:786` emits
  `N_{\text{max}}/L_{\text{lat}}`, but section defines `T_{\text{lat}}`. Use
  `T_{\text{lat}}`.
- Medium: notebooks at `:699`, `:1019` lack scaffold or should be perspectives.
- Medium: emphasis/index ownership issues at `:335`, `:360`, `:735`, `:1021`,
  `:1209`. Remove duplicate bolds; add `Little's Law!appendix refresher`.
- Medium: hard-typed derived/spec values at `:217`, `:244`, `:296`, `:1237`,
  including `~580x` vs `~600x` drift. Source through LEGO/MLSysIM or document.
- Low: Purpose hook `:9` is "What"; recast as "Why".
- Low: second-person prose outside allowed contexts at `:23`, `:257`, `:319`,
  `:333`, `:367`, `:439`, `:798`, `:908`, `:1229`, `:1231`.

### Volume 2 Compute Infrastructure

Status: not signed off.

- Medium: checkpoint at `compute_infrastructure.qmd:796` asks for "generality
  tax" of DDR memory, but generality tax was defined as wasted processor
  capability. Change anchor/prompt to signal path/TSVs.
- Medium: notebook at `:2116` lacks scaffold; `:2136`, `:2140` hard-code
  derived Joule values in equations. Compute math outputs in Python.
- Low: Fallacies/Pitfalls at `:3819`, `:3823` have two consecutive Fallacy
  entries. Restore Fallacy/Pitfall rhythm.
- Medium: labeled Python cells at `:282`, `:322`, `:978`, `:1025`, `:3482`
  lack full header contract or import before header.
- Low: `Generality Tax` at `:139` should be formal definition/index if Tier 2
  named principle: `**Generality Tax**\index{Generality Tax!definition}`.

### Volume 2 Network Fabrics

Status: not signed off.

- Major: progressive disclosure/future dependencies at `network_fabrics.qmd:85`,
  `:550`, `:910`, `:1066`, `:2090`, `:2130`, `:2132`, `:2134` rely on later
  Volume 2 chapters. Define locally or soften future refs.
- Major: inline Python adjacent to math at `:642`, `:665`, `:667`, `:935`,
  `:1144`, `:1201`, `:1225`; opt-in Binder reports `python_in_math`.
- Major: hard-coded/computed-number locality gaps at `:75`, `:143`, `:480`,
  `:488`, `:1206`, `:1667`, `:1686`, `:1959`, `:2082`.
- Medium: notebooks at `:651-659`, `:898-922` have duplicate Result or missing
  scaffold/systems insight.
- Medium: named cells at `:775-814`, `:1897-1901` import before header or omit
  full header.
- Medium: percent style in prose at `:1457-1462` and `:1490`.
- Low: `*flowlet switching*\index{Flowlet Switching}` at `:1525` should be first
  definition bold/index or plain mention.
- Low: unboxed synthesis paragraph after takeaways at `:2122`.

### Volume 2 Data Storage

Status: not signed off.

- High: storage hierarchy table at `data_storage.qmd:461` has hand-typed
  canonical-looking capacities, bandwidths, latencies, and costs. Generate from
  MLSysIM or mark illustrative.
- High: figure cell at `:251` hard-codes GPU/storage historical values and
  derives growth ratios. Source from MLSysIM or add there.
- High: price constants at `:1237`, `:2023`, `:2898` bypass
  `Infrastructure.Pricing`.
- High: computed values hand-typed despite nearby computations at `:208`,
  `:1409`, `:2383`, `:2710`.
- Medium: notebook at `:583` lacks scaffold/systems insight.
- Medium: notebook Problem entries at `:2196`, `:2565`, `:2788` are statements
  or commands, not direct questions.
- Medium: checkpoint-storm definition at `:2483` carries long derivation inside
  significance bullet; move arithmetic to notebook.
- Medium: API/operations-manual drift at `:803`, `:1601`, `:1699` (`pin_memory`,
  `prefetch_factor`, `num_workers`). Generalize.
- Low: Purpose close `:29` underplays coordination; revise final C3 sentence to
  include checkpointing/versioning coordination.
- Low: percent ranges at `:159`, `:815`, `:2849` use dash forms; use "to".
- Low: mid-sentence `@Sec-...` at `:163`, `:402`, `:962`, `:1407`, `:1898`,
  `:2922`; normalize to lowercase `@sec-...` unless sentence start.

### Volume 2 Fault Tolerance

Status: conditional, not full sign-off.

- Medium: unlabeled display equations at `fault_tolerance.qmd:232`, `:305`,
  `:307`, `:3054`.
- Medium: SDC probability outputs at `:1980`, `:1994` use
  `fmt_percent(..., style="symbol")` in prose.
- Medium: derived SDC cadence values at `:1992`, `:1996` are hand-written.
- Medium: named cells at `:254`, `:1963`, `:2917`, `:3066` lack full
  Context/Goal/Show/How header.
- Medium: derived reliability/storage literals at `:2315`, `:2639`, `:2653`.
- Low: straggler tax notebook at `:3102`, `:3104`, `:3114` lacks scaffold.
- Low: percent ranges at `:766`, `:1114`, `:1118`, `:1244`, `:3378` use dash.
- Low: takeaway index tags at `:3887`, `:3893` should move to body or be removed.

### Volume 2 Fleet Orchestration

Status: blocked for sign-off.

- High: algorithm at `fleet_orchestration.qmd:927` says infeasible job stays
  queued but falls through to feasible-slot return at `:934`. Add explicit
  return/else.
- High: hard-coded economic/capacity/utilization values at `:1300`, `:1577`,
  `:1670`, `:1854`, `:2103`; source from LEGO/MLSysIM.
- Medium: duplicate `Gang Scheduling!definition` at `:142`, `:146`; keep
  callout definition/index only.
- Medium: notebooks at `:1050`, `:1280`, `:1812`, `:1927`, `:2011` lack
  canonical opening/final systems insight.
- Medium-Low: `Trackable Resources (TRES)` at `:650` creates separate index
  entries; use combined key or allowlist acronym.
- Low: percent range `30--60 percent` at `:1473`; use "to".
- Low: Python cell headers at `:66`, `:271`, `:1015` incomplete.
- Low/rule drift: chapter YAML lacks `glossary` but neighbors do too; update
  rule if per-chapter glossaries are retired.

### Volume 2 Inference

Status: not signed off.

- High: display math at `inference.qmd:2537`, `:2555` wraps inline Python refs.
  Compute whole display equation using existing `fmt_display_math` pattern.
- Medium: KV-cache formula at `:2140` appears before variable definitions
  `:2143-2147`, and `:2141` uses prose factors `\text{Batch}` and
  `\text{Context}`. Move definitions before formula and use canonical/local
  symbols such as `B`, `S`, or scoped `S_{\text{ctx}}`.
- Medium: routing pseudocode at `:3939` ends without `\State \Return`.
- Medium: `:2197` anchors `\index{PagedAttention!block table}` on
  PagedAttention, not block-table phrase.
- Low: `:1158` bold mini-heading has period inside example callout.
- Low: `:5961` standalone paragraph appears between takeaways and chapter
  connection.

### Volume 2 Edge Intelligence

Status: conditional, not full sign-off.

- Medium: checkpoint at `edge_intelligence.qmd:2866` asks for DP noise with
  epsilon budget before DP budget machinery is taught, and epsilon is nearby
  convergence error. Make qualitative or teach local DP-noise notation.
- Minor: definition callouts at `:342`, `:2011`, `:2143` use `**Distinction**:`
  instead of `**Distinction (durable)**:`.
- Minor: range delimiter drift at `:952`, `:1217`, `:1241`, `:1990`, `:2861`,
  `:3103`, `:3105`.
- Low: `:1763` splits equation at hanging equals before Python output.

### Volume 2 Ops Scale

Status: not signed off.

- High: chapter connection at `ops_scale.qmd:5316` points backward to
  `@Sec-fleet-orchestration`; should point to next chapter `@sec-security-privacy`.
- Medium: `:2385`, `:3991` say fault tolerance "will develop/establish" though
  that chapter is prior. Use past-tense dependency language.
- Medium: checkpoints at `:1627`, `:1807` formatted as numbered exercises, not
  checkbox checkpoint bullets.
- Low: margin prose `20x` at `:2250`; use `20$\times$` in rendered prose.
- Low: duplicate class `H100TdpExerciseRecap` at `:1621`, `:1801`; use unique
  names or shared recap.

### Volume 2 Security Privacy

Status: not signed off.

- Medium: Purpose close at `security_privacy.qmd:29` says "coordination,
  communication, and execution" rather than formal C3 axes. Rewrite to name
  Compute, Communication, Coordination.
- Medium: `:365` uses competing triad "Data, Algorithm, and Infrastructure";
  map back to compute execution, communication exposure, coordination authority
  or avoid competing triad.
- Medium: `:2195` recommends PGD-based adversarial training with future
  `@sec-robust-ai`; replace with locally explained adversarial training bridge.
- Medium: Differential Privacy definition ownership conflict at `:2225` vs
  earlier `vol2/introduction.qmd:1915` and `edge_intelligence.qmd:3081`. Choose
  canonical owner, likely security/privacy, demote earlier previews.
- Low: `:1737` imports before header; `:564`, `:1698`, `:2109` prose-facing
  cells lack labels/full headers.
- Low: `:2962` DP-SGD notebook uses Scenario/Step; recast as Problem,
  Variables, Math, Result, Systems insight.

### Volume 2 Robust AI

Status: not signed off.

- High: `robust_ai.qmd:781` puts inline Python inside `$...$` math:
  ``$`{python} ...`/\sqrt{`{python} ...`} \approx$``. Compute the whole
  expression as a preformatted math string with `fmt_math` and insert it outside
  `$...$`.
- High: `robust_ai.qmd:2780` uses over-escaped norm delimiters inside inline
  math. Use single LaTeX escapes, preferably
  `$\lVert f(x)-f(x') \rVert \le K \lVert x-x' \rVert$`.
- Medium: `robust_ai.qmd:918` reuses bare `\epsilon` for PSI smoothing while
  adversarial radius also uses bare `\epsilon` at `:68`, `:1428`, `:2174`.
  Rename smoothing to `\epsilon_{\text{smooth}}`; consider standardizing
  adversarial radius as `\epsilon_{\text{adv}}` or adding a notation row if it
  is book-wide.
- Medium: Python cell at `:322` imports before the LEGO header at `:328`. Move
  the full header immediately after chunk options and before imports.
- Medium: cell at `:382` has incomplete LEGO-style header and duplicate marker
  at `:396-397`. Add standard header and remove duplicate marker.
- Medium: `:1278` hard-codes `1.3` and `1.1` in a rendered KL expression,
  duplicating class constants. Interpolate computed constants.
- Medium: `:2747` has partial summary calculation block and duplicates robust
  accuracy constants defined earlier. Reuse earlier calculation object or make a
  complete guarded LEGO cell.
- Low: `:2250` labels a `.text` listing as "Adversarial Training Pseudocode".
  Convert to formal pseudocode `algo-` block or retitle as a framework-neutral
  listing.
- Low: `:1391` gives a concrete "99.9 percent confidence" medical-diagnostic
  scenario without citation/computed grounding. Cite or soften as hypothetical.

### Volume 2 Sustainable AI

Status: not signed off.

- Medium: `sustainable_ai.qmd:21` Purpose heading is
  `## Purpose {.unnumbered}` but the reporting agent says `purpose.md` and
  `chapter-architecture.md` require `## Purpose {.unnumbered .unlisted}`.
  Verify the current rule before editing; Binder `structure` only checks
  `.unnumbered`.
- Medium: `:167-173` checkpoint uses PUE before the chapter teaches or defines
  PUE; first substantive definition appears around `:1093`. Move the checkpoint
  after facility-level power metrics or introduce PUE earlier.
- Medium: body section IDs at `:717`, `:772`, `:794`, `:2295`, `:2726`,
  `:3027`, `:3307` omit the required permanent four-hex suffix.
- Medium: section IDs at `:1033`, `:1037`, `:1181`, `:1202`, `:1206`, `:1229`,
  `:1298`, `:1421`, `:1493`, `:3244`, `:3454` contain duplicated hash/topic
  fragments outside the `{#sec-[chapter]-[topic]-[4hex]}` convention.
- Low: `:1183`, `:1185`, `:1657` use `CO~2~eq`; prefer
  `CO~2~-equivalent` in prose and compact `CO~2~e` units such as
  `g CO~2~e/kWh`.
- Low: `:414-415`, `:1631` combine formatter output like `g/kWh` with prose
  appending `CO~2~`, yielding awkward unit order. Rephrase so the formatter and
  unit phrase read naturally.
- Low: `:412-416` notebook `**Problem**:` lead-in is a statement while the
  direct question sits on a separate unlabeled line. Make `**Problem**:` end
  with the direct answerable question.

### Volume 2 Responsible AI

Status: not signed off.

- High: `responsible_ai.qmd:643` disparate-impact formula is reversed relative
  to the computed value. The table shows
  `Pr(h=1 \mid a) / Pr(h=1 \mid b)`, while Python computes Group B over Group A
  as `0.57`. Align formula, table, prose, and computation, likely by defining
  the reference group and using `Pr(h=1 \mid b) / Pr(h=1 \mid a)` if that is the
  intended ratio.
- Medium: frontmatter at `:1` lacks a `glossary:` key according to the reporting
  agent, likely `glossary: responsible_ai_glossary.json`. Verify this against
  current Volume 2 chapter architecture first because other Volume 2 chapters
  may share this omission and it may be rule drift.
- Medium: Purpose heading at `:21` is `## Purpose {.unnumbered}` but the agent
  reported a `.unlisted` requirement. Verify the actual active rule before
  editing, since earlier local rule inspection did not clearly require it.
- Medium: raw LaTeX `\ref{...}` at `:443` and `:2358` references principles.
  Use Quarto `@pri-...` references or the local helper convention.
- Medium: TinyML safety row at `:1187` says the system needs distributed safety
  mechanisms due to autonomy, which conflicts with the static-safeguards framing.
  Revise toward static fail-safe logic, conservative envelopes, watchdog/input
  checks, and compile-time safety validation.
- Medium: notebooks at `:1869`, `:1972`, `:2460` lack required
  Problem/Scenario starts and final Systems insight.
- Medium: prose-facing TB values at `:2199` use magic unit conversion divisors
  such as `/ 1e6`; use `fmt_qty` or MLSysIM unit constants.
- Low: cell `confusion-matrix-audit-calc` around `:899` has Context/Goal/How
  but no Show. Add Show.
- Low: prose ranges at `:82`, `:151`, `:982`, `:2164` use dash/hyphen forms
  such as `5-15 percent`, `15--30 percent`, or `100-500 ms`; use `to` for
  percent and source-clean unit ranges.
- Low: takeaway at `:3086` capitalizes ordinary C3 axis words such as
  "Coordination constraint" and "Compute is allowed to run." Lowercase ordinary
  axis usage.

### Volume 2 Performance Engineering

Status: not signed off.

- High: `performance_engineering.qmd:391`, `:419`, `:425`, `:484` hard-code
  H100/V100/B200 roofline values before or outside the computing cell. Move the
  relevant calculation before first use and replace prose/captions/equations
  with computed inline refs or generated `fmt_*` strings.
- High: `:2213` hard-codes derived "84 percent" from achieved bandwidth and H100
  bandwidth. Compute in nearby profiling cell and emit with prose-safe formatter.
- Medium: Purpose paragraph at `:29` reads like a chapter-summary list and says
  order-of-magnitude gain comes "entirely from the compute term" of C3, which is
  too narrow for overlap, fleet measurement, and scaling tax. Recast as a stakes
  paragraph: Compute as entry point, Communication and Coordination as follow-on
  limits.
- Medium: `:397` introduces `Arithmetic Intensity` as formal definition again
  despite prior Volume 2 definition in compute infrastructure. Convert to
  refresher/perspective or chapter-specific application.
- Medium: `:735` uses `$b_{\text{elem}}$` for element size, conflicting with
  canonical `$s_{\text{elem}}$`. Reuse `$s_{\text{elem}}$`.
- Medium: finite speculative decoding algorithm at `:1803` has `\Require` and
  `\Ensure` but no final `\State \Return ...`. Return accepted/emitted tokens.
- Medium: `:2410` presents Model FLOPs Utilization (MFU) as a formal definition
  again after earlier Volume 2 treatment. Make it a fleet-scale MFU extension or
  refresher.
- Medium: prose-facing calculation cells at `:810`, `:1630`, `:2174`, `:2314`,
  `:2692` are missing full Context/Goal/Show/How headers; several are unlabeled.
- Low: displayed equations at `:112`, `:425`, `:735`, `:2626` lack labels. Add
  stable `{#eq-...}` labels or convert short nonreferenced displays to inline.

### Volume 2 Collective Communication

Status: not signed off.

- High: `collective_communication.qmd:2101`, `:2103` hand-compute compression
  totals and percentages in prose. Add nearby self-contained calculation cell
  for both scenarios and inline outputs, or make comparison qualitative.
- High: named calculation cells at `:521`, `:596`, `:905`, `:1104`, `:1397`,
  `:2229` lack required Context/Goal/Show/How header immediately after chunk
  options; `:2229` imports before the header.
- High: cells at `:1397`, `:1422`, `:1434`, `:1526`, `:1545`, `:1546` rely on
  previous globals such as `Models`, `Hardware`, `Systems`. Import dependencies
  locally.
- Medium: LogP notation at `:587`, `:589`, `:590`, `:592`, `:624`, `:628`,
  `:647`, `:2221`, `:2345` introduces reusable `$o$`, `$2o$`,
  `$N_{\text{rank}}$`, and `$L_{\text{lat}}$` in ways that do not match
  frontmatter LogGP canon. Either add a formal LogP notation block to
  distributed notation or rewrite to canonical LogGP symbols with local
  explanation.
- Medium: Ring AllReduce algorithm at `:1077`, `:1080`, `:1081`, `:1086`,
  `:1087` uses `$G$`/`$G_i$` for gradient tensors, colliding with canonical
  LogGP `$G$` long-message gap. Rename tensor variable, e.g. `$X_i$`.
- Medium: dtype byte widths, gradient sizes, and model sizes are hard-coded at
  `:87`, `:185`, `:186`, `:299`, `:691`, `:1434`, `:2254`, `:2258`.
- Medium: figure caption at `:256`, `:283`, `:309`, `:313` claims proportions
  are "derived from H100 cluster measurements," but cell appears modeled or
  synthetic. Source real data or reword as modeled/illustrative.
- Medium: interconnect alpha ranges at `:449`, `:464-468` are chapter-local
  constants with pending alpha canon. Move assumptions into MLSysIM or formal
  scenario-assumption table.
- Medium: prose calculations at `:1390`, `:1391`, `:1719`, `:2311`, `:2313`
  lack adjacent executable provenance.
- Low: pseudo-heading/step emphasis at `:981`, `:1141`, `:1154`, `:1597`; use
  plain prose, headings, or `**Label**:` with colon outside bold.
- Low: checkpoint at `:1181` contains inline bold/index markup for "sequential
  steps" beyond checkbox lead-in pattern. Move indexed definition to body.
- Low: percent ranges at `:1860`, `:1862`, `:2005`, `:2223`, `:2311` use dash
  style; write `2 to 5 percent`, etc.

### Volume 2 Conclusion

Status: conditional sign-off only.

- Medium: `conclusion.qmd:400` Fermi notebook has an empty
  `**Systems insight**:` label followed by bullets. Collapse into one final
  systems-insight paragraph, or use `**Result**:` for scan bullets and reserve
  `**Systems insight**:` for the closing paragraph.
- Medium: `conclusion.qmd:451` `.callout-chapter-connection` summarizes the
  volume rather than naming unresolved problem and pointing forward. For a
  terminal conclusion, remove/reclassify as closing prose or add a
  terminal-conclusion callout rule if intentional.
- Medium/rule conflict: `conclusion.qmd:21` Purpose heading is
  `## Purpose {.unnumbered}` but agent reported current rules require
  `## Purpose {.unnumbered .unlisted}`. Verify the actual rule before editing;
  the current `purpose.md` in this worktree may not require `.unlisted`.
- Medium: scenario constants at `:288`, `:289`, `:350` are hard-coded locally.
  Promote to `mlsysim.ReferenceStats` with provenance or document one-off
  exception and extend/suppress checker.
- Low: prose percent range `20--50 percent` at `:161`; rewrite as
  `20 to 50 percent`.
- Low: Fallacies section ID at `:422` lacks current four-character suffix
  convention. Assign stable suffixed ID and update refs if any.

## Broad Themes Across Findings

1. Default Binder is necessary but insufficient. Many release blockers are
   semantic/rule gaps.
2. The notation merge improved frontmatter and several appendices, but serving
   notation still has drift in `model_serving.qmd`.
3. Purpose paragraphs mostly pass shape, but some need prose-craft tightening or
   C3/D.A.M close repairs.
4. Notebook scaffolding is inconsistent across many chapters and appendices.
5. Inline Python inside math is a corpus-wide problem, but agents identify
   specific release-blocking examples.
6. Several chapters use exact hardware-model throughput before Volume 1 hardware
   chapter. This is a major policy decision, not just a typo.
7. Many cells are not self-contained and rely on earlier imports/state. Decide
   how much to fix now versus add Binder scaffolding.
8. MLSysIM source-of-truth drift appears in data storage, serving prices, fleet
   economics, network fabrics, fault tolerance, and appendices.

## Recommended Next Steps

1. Harvest and close completed agents listed above. Wait for remaining active
   agents. Launch Volume 2 appendix agents once slots free.

2. Create a tracking table in your notes with columns:
   `chapter`, `status`, `must fix before sign-off`, `rule/check gap`,
   `edit scope`, `verification`.

3. Fix narrow high-confidence issues first:
   - notation mismatches (`model_serving`, `appendix_machine`, `inference`);
   - algorithm fallthrough/return issues (`fleet_orchestration`, `inference`,
     `training`);
   - Purpose obvious defects (`introduction`, `frameworks`,
     `security_privacy`, `data_storage`, appendix hooks);
   - chapter-connection wrong target (`ops_scale`);
   - inline Python in math for chapters where agents flagged release blockers.

4. For hardware progressive-disclosure in Volume 1, pause and decide policy
   before broad edits. Options:
   - generalize all pre-hardware exact model TFLOP/s references;
   - move detailed examples to hardware chapter/appendix;
   - update rules to allow "reference accelerator" with no exact TFLOP/s before
     hardware.
   Do not silently rewrite dozens of computed examples without an explicit
   editorial direction.

5. Update Binder/rules where findings reveal rule/check mismatch:
   - war-story schema;
   - Purpose shape;
   - notebook scaffold;
   - algorithm return/blank line;
   - appendix checkpoint policy;
   - Python cell header/self-contained import discipline.

6. After edits:
   - run targeted Binder checks on touched files;
   - run `./book/binder check notation --scope consistency`;
   - run `./book/binder check all` and document missing-PDF failure if PDFs were
     not built;
   - run relevant opt-in checks only if the scope was addressed.

7. Stage only touched files and commit actual text/check/rule fixes on
   `codex/full-audit-signoff`.

8. Final response should include:
   - commit hash(es);
   - chapter sign-off table: PASS, PASS WITH FIXES APPLIED, DEFERRED/POLICY,
     or BLOCKED;
   - checks run and remaining residual risks.

## Do Not Do

- Do not remove or retire `/Users/VJ/GitHub/MLSysBook`.
- Do not edit AI config files in this worktree; create a separate worktree if
  the user asks for AI config changes.
- Do not treat agent suggestions as automatic edits. Inspect full sections like
  a student reading the chapter before changing prose.
- Do not flatten authorial voice with broad rewrites.
- Do not update notation in chapter prose without updating the frontmatter table
  when a symbol is reusable.
- Do not make a giant hardware-specific rewrite until the Volume 1
  progressive-disclosure policy is resolved.
