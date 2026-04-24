# Path to A — Quiz Quality Improvement Plan

**Status**: handoff plan (next session picks this up).
**Created**: 2026-04-24.
**Author of this plan**: drafted at the end of a long session that landed
the corpus at grade B; resume in a fresh session.

---

## Where we are now

- **33 / 33 chapters refreshed** with `gpt-5.4`. Questions: **1,685**.
- **Improve pass applied** (commit `e564c53c5`) — 564 of 1,685 questions
  (33.5%) had content edits, mostly LO tightening (456 / 27%).
- **Audit pass applied** (commit `9a4dfe2e5`). Findings:
  - Grade distribution: A:0  **B:30**  C:0  D:0  F:0
  - 271 issues flagged: 155 low / 110 medium / 6 high
  - 3 redefinition violations + 5 forward-reference + 96 missed-buildup
- **4 high-severity hand-fixes applied** (commit `24dacbbe4`). The other
  2 high-severity flags were audit false positives — the improve pass
  had already addressed them.
- **Filter live**, render verified, click navigation tested via
  Playwright. Quizzes appear inline at end of each `##` section + the
  "Self-Check Answers" section at chapter end with cross-nav links.
- **Branch**: `feat/quiz-refresh` (in worktree
  `/Users/VJ/GitHub/MLSysBook-quiz-refresh/`). All commits already
  merged into local `dev` at `ef45d2ecb`.
- **Push state**: local `dev` is 45 commits ahead, 35 behind `origin/dev`
  (diverged because parallel work landed on origin during this session).
  A `git pull --rebase` (or merge) is needed before pushing.

## Goal

**Move all 33 chapters from grade B to grade A** as judged by the audit
pipeline, AND have a corpus a human subject-matter expert would call
genuinely excellent rather than merely "good with minor issues."

## Why we're at B and not A

LLM auditor flagged 271 issues across 30 audited chapters (3 not
audited; see "coverage gap" below). The biggest issue patterns:

| Type | Count | Lever |
|---|---:|---|
| trivia_fill | 77 | FILL still leans to bare term recall |
| easy_tf | 50 | TF answer obvious from grammar |
| recall_only | 39 | memorization vs. reasoning |
| throwaway_distractor | 28 | MCQ distractors no informed reader picks |
| vague_lo | 21 | non-concrete learning objectives |
| build_up_violation | 18 | redefines a prior-chapter term |
| tautological_lo | 15 | LO restates the question stem |

The improve pass already shifted 564 questions; further LLM passes
hit diminishing returns because the model has no anchor for what
**A-grade** looks like. The spec defines rules but contains zero
worked examples — LLMs calibrate from examples much better than
from rules.

## The plan — three phases over ~3 sessions

### Phase 1 — Augment the spec with worked examples *(highest-leverage)*

**Goal**: add a `## Gold Standard Examples` section to
`.claude/rules/quiz-generation.md` containing **5 worked examples per
question type** (5 × 5 = 25 examples), each with annotations
explaining what makes it strong.

**Coverage**:
- 5 MCQ examples covering: scenario-based application,
  classification-under-framework, distractor design encoding real
  misconceptions, integrative-cross-section question, quantitative
  reasoning under constraints.
- 5 SHORT examples covering: trade-off explanation, justification of
  design choice, scenario walkthrough, system-consequence reasoning,
  cross-chapter connection.
- 5 TF examples covering: misconception refutation, edge-case
  precision, quantitative-claim verification, system-property
  invariant, scope-limit clarification.
- 5 FILL examples covering: technical term inferred from operational
  context (NOT just a vocabulary slot), one-word answer where the
  blank tests *which* concept (not which spelling), and three more
  showing how to make FILL test reasoning rather than recall.
- 5 ORDER examples covering: genuinely sequential processes (not
  cyclical/relational ones), where step ordering is causally
  necessary and swapping breaks the system.

For each example: include the question, choices/structure, the
gold-standard answer, the LO, and a short "why this is A-grade"
annotation. Pull examples from real chapter material so they ground in
actual book content.

**Where to write**: append to
`/Users/VJ/GitHub/AIConfigs/projects/MLSysBook/.claude/rules/quiz-generation.md`
as a new Section §16 or appendix. Mirror to MLSysBook local
`.claude/` if symlink isn't auto-syncing.

**Effort**: ~2 hours of careful writing. Best done by a human or by
a careful LLM session with explicit review.

**Outcome of Phase 1**: enhanced spec checked in to AIConfigs `main`
and visible to the runner via the symlink.

---

### Phase 2 — Cross-audit with Opus 4.7 *(calibration check)*

Before running another full improve pass, validate whether B is the
LLM's calibration ceiling or a real quality assessment.

**Action**: run `--mode audit --provider anthropic --model claude-opus-4-7`
across the same 33 chapters.

**Setup needed**:
- `ANTHROPIC_API_KEY` in env (was not set during this session — only
  `OPENAI_API_KEY` was active).
- Run from `MLSysBook-quiz-refresh/` worktree on `feat/quiz-refresh`
  branch:
  ```bash
  uv run --with anthropic python3 \
      book/tools/scripts/genai/quiz_refresh/generate_quizzes.py \
      --mode audit --provider anthropic --model claude-opus-4-7 \
      --all --workers 4
  ```

**Cost**: ~$50–80 (Opus is ~$15/M input, ~$75/M output; 33 chapters at
~120K input + ~5K output ≈ $50 total).

**Decision tree**:
- If Opus grades chapters **A** where gpt-5.4 said B → B was
  self-grading bias; the corpus is already A. Stop, ship as-is.
- If Opus also grades **B** → the issues are real; proceed to Phase 3.
- If Opus grades chapters **A−** with different specific findings →
  union the two audits' flagged questions; that union is the targeted
  improvement set.

**Also fix during this phase**: the audit naming collision that lost
3 chapters last time. In `generate_quizzes.py`, change
`memo_path()` and the audit output filename to use `{vol}_{chapter}`
prefix instead of `{chapter}` so vol1/introduction and vol2/introduction
don't collide. One-line fix near line ~92.

---

### Phase 3 — Targeted re-improve with enhanced spec

**Conditional on Phase 2's verdict.** If B is real, run improve mode
again with the enhanced spec from Phase 1. Two modes worth considering:

**Option A — full re-improve** (preferred if Phase 1's spec changes
are substantial):
```bash
uv run --with openai python3 \
    book/tools/scripts/genai/quiz_refresh/generate_quizzes.py \
    --mode improve --all --workers 4
```
Cost: ~$30–60 (similar to first improve pass). Time: ~25 min.

**Option B — targeted improve scoped to flagged questions only**
(if Phase 1's changes are incremental):
Build a script that reads the audit JSONs, collects every
medium-or-higher-severity issue, and applies a per-question rewrite
call (the same pattern the 4 hand-fixes used). 110 medium-severity
flags + a few high = ~120 single-question API calls. Cost: ~$5–15.
Time: ~5 min.

After either option, **re-audit** using either gpt-5.4 or Opus and
confirm the grade shift.

---

### Phase 4 — Targeted human pass on residual issues *(optional, if A
still elusive)*

If Phase 3's re-audit still grades some chapters B, the residual
issues are likely things only a domain expert would catch (factual
nuance, subtle pedagogical missteps, voice mismatch with the book).

**Action**: extract the audit's residual flagged questions to a single
review document grouped by chapter. A subject-matter expert (the book
author or a TA) rewrites them by hand — typically 30–80 questions
across the whole book. Cost: 5–10 hours of expert time.

This is the only path to a corpus that a human reviewer would call
genuinely *excellent* rather than *good*. LLM-only paths cap below this.

---

## Coverage gap to fix during Phase 2

The first audit pass missed 3 chapters due to a file-naming collision
and one quota failure:

- `vol1/introduction` and `vol2/introduction` both write to
  `introduction_audit.json` — one overwrites the other.
- `vol1/conclusion` and `vol2/conclusion` — same collision on
  `conclusion_audit.json`.
- `vol2/responsible_ai` — quota-failed at the very end of the run.

**Fix**: in `generate_quizzes.py`, when `mode == "audit"`, change the
output filename from `f"{chapter}_audit.json"` to
`f"{vol}_{chapter}_audit.json"`. One-line edit in `generate_for_chapter`.
Then the Phase 2 cross-audit naturally covers all 33 with no overlap.

## Resume instructions for the next session

1. **Open a fresh session** (new conversation; no need to import this
   one's history).
2. **Read this file** first:
   `book/tools/scripts/genai/quiz_refresh/PATH_TO_A_PLAN.md`.
3. **Read the canonical spec** (the thing you'll be augmenting in
   Phase 1):
   `.claude/rules/quiz-generation.md`.
4. **Read the previous audit's master report** for context on what
   issues exist:
   `book/tools/scripts/genai/quiz_refresh/_audit/MASTER_REPORT.md`.
5. **Sample 3–4 audit JSONs** to see the per-question issue format:
   `book/tools/scripts/genai/quiz_refresh/_audit/{chapter}_audit.json`.
6. **Sample 2–3 chapter quiz JSONs** to see current quality:
   `book/quarto/contents/vol1/training/training_quizzes.json`,
   `book/quarto/contents/vol2/inference/inference_quizzes.json`.
7. **Confirm worktree**: `cd /Users/VJ/GitHub/MLSysBook-quiz-refresh`
   and verify branch is `feat/quiz-refresh`.
8. **Begin Phase 1**: write the 25 worked examples by directly editing
   `quiz-generation.md` in AIConfigs.

## Acceptance criteria

| Phase | Done when |
|---|---|
| 1 | 25 worked examples added to `quiz-generation.md` and committed in AIConfigs |
| 2 | Audit JSONs from Opus 4.7 exist for all 33 chapters at `_audit/opus/` (use a subdirectory to keep them separate from the gpt-5.4 audit) |
| 3 | Re-improve commit on `feat/quiz-refresh`; validator clean across 33 |
| 4 | Residual flagged questions either fixed by human or accepted as residual |
| **Overall** | Re-audit median grade is **A** across at least 28/33 chapters |

## Cost summary

| Phase | API cost | Human time |
|---|---:|---|
| 1 (worked examples) | $0 | 2 hours writing |
| 2 (Opus audit) | ~$60 | 30 min review |
| 3 (re-improve) | $30–60 | 30 min review |
| 4 (human pass on residual) | $0 | 5–10 hours |
| **Total** | **~$90–120** | **~8–13 hours** |

## What NOT to do

- **Don't run the full generate pipeline again from scratch.** It would
  destroy the 564 improve-pass edits and the 4 hand-fixes. The current
  corpus is the floor we're improving FROM, not REPLACING.
- **Don't run improve in a tight loop without re-auditing.** The
  improve agent has its own ceiling; running it 5 times back-to-back
  produces diminishing returns and risks drift away from the intended
  improvements. One improve → audit → assess → maybe one more improve.
- **Don't try to fix every "low" severity issue.** 155 low issues
  averages 5 per chapter; fixing them all is expensive polish that
  doesn't move the auditor's grade. Focus on medium and high.
- **Don't skip Phase 1 spec enhancement and jump to Phase 3 re-improve.**
  Re-improving against the same rule-based spec produces the same
  result. Worked examples are the leverage point.

## Files of record

| Purpose | Location |
|---|---|
| Canonical spec (will be augmented in Phase 1) | `.claude/rules/quiz-generation.md` (canonical at `AIConfigs/projects/MLSysBook/.claude/rules/`) |
| Runner | `book/tools/scripts/genai/quiz_refresh/generate_quizzes.py` |
| Validator | `book/tools/scripts/genai/quiz_refresh/validate_quiz_json.py` |
| Audit master report | `book/tools/scripts/genai/quiz_refresh/_audit/MASTER_REPORT.md` |
| Per-chapter audits (gpt-5.4) | `book/tools/scripts/genai/quiz_refresh/_audit/*_audit.json` |
| Per-chapter improvement memos | `book/tools/scripts/genai/quiz_refresh/_reviews/*_memo.md` |
| Quiz JSONs (canonical) | `book/quarto/contents/{vol}/{chapter}/{stem}_quizzes.json` |
| Lua filter (HTML injection) | `book/quarto/filters/inject_quizzes.lua` |
| Playwright check | `book/tools/scripts/genai/quiz_refresh/playwright_verify.py` |
