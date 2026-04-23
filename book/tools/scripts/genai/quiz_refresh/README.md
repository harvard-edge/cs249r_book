# Quiz Refresh — Vol1 & Vol2 Regeneration Plan

**Status**: active
**Branch**: `feat/quiz-refresh` (in worktree `/Users/VJ/GitHub/MLSysBook-quiz-refresh/`)
**Parent**: `dev` @ `5eaaa7642` (disable-quiz-injection merged)
**Created**: 2026-04-23

## 1. Why this exists

Every `*_quizzes.json` file committed to `dev` was generated in a previous
pass against earlier drafts of Vol1 and Vol2. The prose has since changed
substantially. Running the quizzes against the current text would expose
students to stale questions (references to sections that no longer exist,
terminology we have since revised, numbers that no longer match the
canonical Python cells).

On top of that, **12 Vol2 chapters have no quiz JSON at all**
(introduction, compute_infrastructure, network_fabrics, data_storage,
distributed_training, collective_communication, fault_tolerance,
fleet_orchestration, performance_engineering, inference, ops_scale,
conclusion). The second volume is effectively un-quizzed.

Quiz injection is currently **disabled** in the HTML build (the Lua filter
is commented out in `book/quarto/config/shared/html/filters.yml` on `dev`
as of `5eaaa7642`). This refresh regenerates JSON files without any
user-visible effect; a follow-up one-line revert re-enables rendering once
the new content has landed.

## 2. Scope

**33 chapters** in reading order:

### Vol1 (16)

| # | Chapter | `.qmd` | Status |
|---|---------|--------|--------|
| 1 | Introduction | `vol1/introduction/introduction.qmd` | regenerate |
| 2 | ML Systems | `vol1/ml_systems/ml_systems.qmd` | regenerate |
| 3 | ML Workflow | `vol1/ml_workflow/ml_workflow.qmd` | regenerate |
| 4 | Data Engineering | `vol1/data_engineering/data_engineering.qmd` | regenerate |
| 5 | NN Computation | `vol1/nn_computation/nn_computation.qmd` | regenerate |
| 6 | NN Architectures | `vol1/nn_architectures/nn_architectures.qmd` | regenerate |
| 7 | Frameworks | `vol1/frameworks/frameworks.qmd` | regenerate |
| 8 | Training | `vol1/training/training.qmd` | **pilot** |
| 9 | Data Selection | `vol1/data_selection/data_selection.qmd` | regenerate |
| 10 | Model Compression | `vol1/optimizations/model_compression.qmd` | regenerate |
| 11 | HW Acceleration | `vol1/hw_acceleration/hw_acceleration.qmd` | regenerate |
| 12 | Benchmarking | `vol1/benchmarking/benchmarking.qmd` | regenerate |
| 13 | Model Serving | `vol1/model_serving/model_serving.qmd` | regenerate |
| 14 | ML Ops | `vol1/ml_ops/ml_ops.qmd` | regenerate |
| 15 | Responsible Engineering | `vol1/responsible_engr/responsible_engr.qmd` | regenerate |
| 16 | Conclusion | `vol1/conclusion/conclusion.qmd` | regenerate |

### Vol2 (17)

| # | Chapter | `.qmd` | Status |
|---|---------|--------|--------|
| 17 | Introduction | `vol2/introduction/introduction.qmd` | **create (new)** |
| 18 | Compute Infrastructure | `vol2/compute_infrastructure/…` | **create (new)** |
| 19 | Network Fabrics | `vol2/network_fabrics/…` | **create (new)** |
| 20 | Data Storage | `vol2/data_storage/…` | **create (new)** |
| 21 | Distributed Training | `vol2/distributed_training/…` | **create (new)** |
| 22 | Collective Communication | `vol2/collective_communication/…` | **create (new)** |
| 23 | Fault Tolerance | `vol2/fault_tolerance/…` | **create (new)** |
| 24 | Fleet Orchestration | `vol2/fleet_orchestration/…` | **create (new)** |
| 25 | Performance Engineering | `vol2/performance_engineering/…` | **create (new)** |
| 26 | Inference | `vol2/inference/…` | **create (new)** |
| 27 | Edge Intelligence | `vol2/edge_intelligence/…` | regenerate |
| 28 | Ops at Scale | `vol2/ops_scale/…` | **create (new)** |
| 29 | Security & Privacy | `vol2/security_privacy/…` | regenerate |
| 30 | Robust AI | `vol2/robust_ai/…` | regenerate |
| 31 | Sustainable AI | `vol2/sustainable_ai/…` | regenerate |
| 32 | Responsible AI | `vol2/responsible_ai/…` | regenerate |
| 33 | Conclusion | `vol2/conclusion/…` | **create (new)** |

## 3. Two-level quiz coverage

Every chapter gets quizzes at **two granularities**:

1. **One quiz per `##` section** (existing pattern) — the reader finishes
   a major section and self-tests the section's thesis, key trade-offs,
   and quantitative claims. Question count: **4–6**, mix-weighted toward
   section-spanning synthesis.
2. **One quiz per `###` subsection** (new) — the reader finishes a focused
   subsection and self-tests the one or two concrete ideas it carries.
   Question count: **2–3**, tighter, more factual/computational.

Rationale: sections carry the argumentative arc; subsections carry
individual mechanisms. Quizzing at both scales matches how the book is
actually read and prevents the "I read the whole chapter and now I don't
remember any of it" failure mode.

A chapter with, say, 7 `##` sections and 20 `###` subsections produces
**~34–62 questions total** (7·(4–6) + 20·(2–3)). Across 33 chapters we
expect on the order of **1,500–2,000 questions** in total.

## 4. JSON schema (extended — v2)

We keep the existing shape so the Lua filter (`inject_quizzes.lua`)
continues to work with zero filter changes — it matches any `Header`
block by identifier, so `###` anchors "just work" as `section_id` values.
We add two optional fields for self-description and QA.

```jsonc
{
  "metadata": {
    "source_file": "book/quarto/contents/vol1/training/training.qmd",
    "schema_version": 2,
    "generated_by": "quiz-refresh-agent",
    "generated_on": "2026-04-23",
    "total_sections": 7,        // ## count
    "total_subsections": 20,    // ### count
    "total_quizzes": 27         // total entries below
  },
  "sections": [
    {
      "section_id": "#sec-training-training-systems-fundamentals-a4b2",
      "section_title": "Training Systems Fundamentals",
      "level": "section",
      "quiz_data": {
        "quiz_needed": true,
        "rationale": { /* unchanged shape */ },
        "questions": [ /* 4–6 items */ ]
      }
    },
    {
      "section_id": "#sec-training-optimizer-state-memory-c81d",
      "section_title": "Optimizer State Memory",
      "level": "subsection",
      "parent_section_id": "#sec-training-training-systems-fundamentals-a4b2",
      "quiz_data": {
        "quiz_needed": true,
        "rationale": { /* unchanged */ },
        "questions": [ /* 2–3 items */ ]
      }
    }
  ]
}
```

**New fields** (both optional for backward compat):
- `level`: `"section"` for `##`, `"subsection"` for `###`.
- `parent_section_id`: on subsection entries, the containing `##` anchor.
- `metadata.schema_version`: `2`.
- `metadata.total_subsections`, `metadata.total_quizzes`.

**Unchanged**: `question_type` (`MCQ` / `SHORT` / `TF`), `question`,
`choices` (MCQ only), `answer`, `learning_objective`, `rationale` block.
Zero Lua filter changes required.

## 5. Question quality bar

Every question, regardless of scope, must pass this checklist:

1. **Grounded in the section text**. If removing the section would make
   the question unanswerable from general CS knowledge, it passes. If a
   reader who skipped the section could answer it from prior coursework,
   cut it.
2. **Tests reasoning, not recall of jargon**. Bad: "What does MFU stand
   for?" Good: "If a V100 reports 45% MFU on GPT-2, what is the dominant
   bottleneck likely to be?"
3. **Uses concrete numbers where the section does**. If the section says
   "Adam requires 3× the memory of SGD," a good question asks the reader
   to compute the optimizer-state bytes for a specific model size.
4. **Explanatory answer**. The `answer` field is not just the letter — it
   explains *why* the correct choice is correct AND why at least one
   distractor is plausibly wrong.
5. **Learning objective is concrete and testable**. Starts with a Bloom's
   Taxonomy verb (Apply, Calculate, Identify, Compare, Explain, Analyze).
   Not "Understand training systems"; instead, "Calculate optimizer-state
   memory overhead for Adam vs. SGD on a model of known size."
6. **MCQ distractors are plausible**. Each wrong choice should reflect a
   real mental-model failure mode students actually have, not absurd
   throwaways.

### Type mix per scope

| Scope | Count | MCQ | Short | TF |
|-------|:-----:|:---:|:-----:|:--:|
| `##` section | 4–6 | 2–3 | 1–2 | 0–1 |
| `###` subsection | 2–3 | 1–2 | 0–1 | 0–1 |

MCQ dominates because the Lua filter's rendering path handles them best
and they score cleanly. Short questions force deeper engagement. TF is
reserved for catching common misconceptions.

## 6. Progressive vocabulary

A chapter's agent receives a **prior-vocabulary context** listing every
term formally defined in a chapter earlier in the reading order. The
agent is instructed:

> Terms listed here have been defined in earlier chapters. You may use
> them in questions and answers without re-defining them. Assume the
> reader has met them. Do not test the reader on the *definition* of a
> prior-chapter term (that question already exists earlier in the book);
> you may still test them on *applying* a prior-chapter concept inside
> this chapter's context.

**Source of truth** for prior vocabulary:
1. Each chapter's tracked `{chapter}_glossary.json` (referenced in the
   chapter's YAML front matter). These are the canonical first-definition
   registry.
2. Bold first-definitions in body prose (`**term**\index{...}`) are a
   secondary source if the glossary is sparse.

**Builder**: `build_prior_vocab.py` (see Directory Layout below) walks
the reading order (hardcoded from the extracted list of `.qmd` paths in
`_quarto-html-vol{1,2}.yml`), and for each chapter N produces
`_context/{vol}/{chapter}/prior_vocab.json` = union of glossary entries
for chapters 1..N-1. Chapter 1 (vol1 introduction) gets an empty prior
vocabulary; chapter 33 (vol2 conclusion) gets everything.

**Cross-volume**: Vol2 chapters include all of Vol1's terms in their
prior vocab. The two volumes are a single reading sequence for
vocabulary purposes.

## 7. Generation workflow (per-chapter agent)

Each sub-agent is spawned via the Agent tool (`general-purpose` subagent
type) with the prompt template in `prompts/agent_brief.md` (see below).

**Inputs to each agent**:
1. Full chapter `.qmd` text (the agent reads it via the Read tool).
2. The chapter's extracted anchor map (section and subsection `{#sec-…}`
   IDs in reading order) — produced by `extract_anchors.py`.
3. The chapter's prior-vocabulary context.
4. The chapter's Learning Objectives callout (already in the `.qmd`).
5. The previous `_quizzes.json` if one exists, as a **stylistic
   reference only** — explicitly tell the agent not to copy content.

**Output from each agent**:
1. A new `{chapter}_quizzes.json` written to
   `book/quarto/contents/{vol}/{chapter}/{chapter}_quizzes.json.new`
   (staging suffix; final rename happens in a human-gated step).
2. A short review memo at
   `_reviews/{chapter}_memo.md` capturing question count per section,
   type-mix compliance, any anchor-matching issues, and open questions.

**Validation after every agent run**:
- Every `section_id` in the new JSON must match an anchor in the chapter.
- Every `parent_section_id` must resolve to a real `##` anchor.
- Schema version and required fields present.
- Counts per scope within 4–6 (section) / 2–3 (subsection).
- Checker: `validate_quiz_json.py` (see below). Agents run it before
  finishing; orchestrator re-runs it as acceptance gate.

## 8. Parallelization strategy

**Serial pilot, then batched parallel fan-out.**

- **Wave 0 — Pilot (1 agent, serial)**. Run on `vol1/training` (has a
  polished existing quiz for before/after comparison). Human review of
  the output sets the quality bar before fanout.
- **Wave 1 — Vol1 foundations (4 agents, parallel)**. Chapters 1–4:
  introduction, ml_systems, ml_workflow, data_engineering.
- **Wave 2 — Vol1 core (4 agents, parallel)**. Chapters 5–7 + 9:
  nn_computation, nn_architectures, frameworks, data_selection (training
  is already done in the pilot).
- **Wave 3 — Vol1 systems (4 agents, parallel)**. Chapters 10–13:
  optimizations, hw_acceleration, benchmarking, model_serving.
- **Wave 4 — Vol1 tail (3 agents, parallel)**. Chapters 14–16: ml_ops,
  responsible_engr, conclusion.
- **Wave 5 — Vol2 infrastructure (4 agents, parallel)**. Chapters 17–20:
  introduction, compute_infrastructure, network_fabrics, data_storage.
- **Wave 6 — Vol2 training (3 agents, parallel)**. Chapters 21–23:
  distributed_training, collective_communication, fault_tolerance.
- **Wave 7 — Vol2 operations (4 agents, parallel)**. Chapters 24–27:
  fleet_orchestration, performance_engineering, inference,
  edge_intelligence.
- **Wave 8 — Vol2 responsible (4 agents, parallel)**. Chapters 28–31:
  ops_scale, security_privacy, robust_ai, sustainable_ai.
- **Wave 9 — Vol2 tail (2 agents, parallel)**. Chapters 32–33:
  responsible_ai, conclusion.

**Batch size**: 4 agents per wave (one-context-window-safe). Each wave
completes before the next begins so the orchestrator can apply lessons
from the validation pass and extend the prior-vocabulary context if any
new terms landed. Between waves is also the safe point to commit the
wave's JSON files to `feat/quiz-refresh`.

## 9. Directory layout

Everything below lives in the `feat/quiz-refresh` worktree:

```
book/tools/scripts/genai/quiz_refresh/
├── README.md                          # this file (plan + ongoing log)
├── build_prior_vocab.py               # builds cumulative vocab per chapter
├── extract_anchors.py                 # pulls ## and ### {#sec-…} from .qmd
├── validate_quiz_json.py              # schema + anchor QA
├── run_agent.py                       # wrapper: brief + input + agent call
├── prompts/
│   └── agent_brief.md                 # template for sub-agent instructions
├── _context/                          # generated, gitignored locally
│   ├── vol1/{chapter}/prior_vocab.json
│   └── vol2/{chapter}/prior_vocab.json
├── _reviews/                          # per-chapter memos from agents
│   └── {chapter}_memo.md
└── .archive/                          # pre-refresh JSONs for rollback
    └── quizzes-pre-refresh-2026-04-23/
        ├── vol1/{chapter}/{chapter}_quizzes.json
        └── vol2/{chapter}/{chapter}_quizzes.json
```

Regenerated JSONs land at the canonical path (overwriting existing where
present) under:

```
book/quarto/contents/{vol}/{chapter}/{chapter}_quizzes.json
```

## 10. Rollout

1. **Per-wave commits on `feat/quiz-refresh`**. Each wave = one commit
   touching only the regenerated JSON files in that wave. Commit message
   pattern: `feat(quizzes): regenerate wave N — {chapter list}`.
2. **Final merge** to `dev` with `--no-ff` and a summary commit message.
   Since the filter is still disabled on `dev`, the merge is silent —
   readers see nothing new.
3. **Re-enable the filter** as a separate follow-up commit on `dev`:
   uncomment `- filters/inject_quizzes.lua` in
   `book/quarto/config/shared/html/filters.yml` and restore the banner
   comment. This is the moment the new quizzes become visible.
4. **Re-enable the SocratiQ widget** (optional) as a third follow-up:
   restore the `<script>` tag and the TOC entry in
   `_quarto-html-vol{1,2}.yml`. Only do this once the `bundle.js`
   behaviour has been re-verified against the new JSON.

## 11. Out of scope

- **Filter changes**: the existing `inject_quizzes.lua` stays exactly as
  it is on `dev` (post self-guard). If we need subsection-specific
  rendering differences (e.g. a visually lighter callout for
  subsection-level quizzes), that lands as a separate PR after the data
  is in place.
- **SocratiQ bundle.js**: untouched. The refresh regenerates data only;
  the widget's behaviour is a separate conversation.
- **Chapter `.qmd` edits**: no prose changes. The quizzes fit around the
  current text. If a section is genuinely too thin to quiz, the agent
  flags `quiz_needed: false` for that section; no prose surgery.
- **Bloom's Taxonomy calibration study**: we use Bloom's verbs but do not
  attempt to balance distributions across the Taxonomy levels. Later.
- **Answer-key UX experimentation**: the answer field renders
  inline-after-submit via the existing filter. We are not redesigning
  the reveal UX.

## 12. Acceptance criteria (for declaring "done")

- All 33 chapters have `{chapter}_quizzes.json` at the canonical path.
- `validate_quiz_json.py` passes for every file with no anchor mismatches.
- Every `section_id` in every JSON resolves to a real `##` or `###`
  anchor in the corresponding `.qmd`.
- Type-mix per scope is within the 4–6 (section) / 2–3 (subsection)
  window.
- Human spot-check of at least one chapter per wave confirms the
  quality bar.
- The `.archive/quizzes-pre-refresh-2026-04-23/` snapshot exists so the
  rollout is reversible.

## 13. Defaults chosen (flag to override)

These are the defaults we run with unless you say otherwise:

| Decision | Default | Rationale |
|----------|---------|-----------|
| Section question count | 4–6 | Matches existing norm; agent judgment within window |
| Subsection question count | 2–3 | Tighter scope, tighter quiz |
| Question mix (section) | 2–3 MCQ / 1–2 Short / 0–1 TF | MCQ scores cleanly; Short forces engagement |
| Question mix (subsection) | 1–2 MCQ / 0–1 Short / 0–1 TF | Same logic, scaled down |
| Pilot chapter | `vol1/training` | Has existing polished quiz → clean A/B |
| Parallel batch size | 4 agents per wave | Context-window safe for orchestrator |
| Anchor scope (`##` vs `###`) | Both | Explicit user ask |
| Cross-volume prior vocab | Yes, Vol2 inherits Vol1 | The two volumes are one reading sequence |
| Filter re-enable | Separate follow-up commit | Data lands first, rendering second |
