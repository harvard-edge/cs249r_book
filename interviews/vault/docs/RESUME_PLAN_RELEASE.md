# Resume Plan — Release-Ready Cleanup + Balanced Generation (2026-04-25)

**Purpose:** hand the next Claude session everything it needs to take
`feat/massive-build-2026-04-25-run` from "ships with caveats" to
"stable dev branch ready for StaffML day."

**Companion doc:** `interviews/vault/docs/RESUME_PLAN_2026-04-25.md`
(the prior session's plan — completed through Phase 7, commit `ece6eccf2`).

---

## Current state

| | |
|---|---|
| **Worktree** | `/Users/VJ/GitHub/MLSysBook-massive-build` |
| **Branch** | `feat/massive-build-2026-04-25-run` (off `feat/massive-build-2026-04-25` in `vault-audit`) |
| **HEAD** | `ece6eccf2 feat(vault): massive build — 630 drafts generated, 320 PASS promoted, paper 0.1.1` |
| **Parent branch** | `feat/massive-build-2026-04-25` in `MLSysBook-vault-audit`, untouched |

**`dev` has advanced** since this branch was cut (was `4a7c64585`, now
`72a741aa1`). Future merge to `dev` will need rebase or merge resolution.
**Do not merge yet** — finish the cleanup + balanced generation first.

---

## What's already done (do NOT redo)

### From commit `ece6eccf2` (this session, 2026-04-25)

- 6-iter Gemini coverage loop ran; 50 of 80 API calls used.
- **630 drafts generated**, **320 PASS promoted** to published.
- Bundle: `9,224 → 9,544 published` (+320 exact).
- 234 visual assets mirrored to `staffml/public/question-visuals/`.
- Paper artifacts refreshed against new `0.1.1` release
  (`release_hash: 0350da5706e6`); `paper.pdf` compiles to 25 pages.
- Loop defaults bumped: `max-iters 30`, `max-calls 80`, `batch 30`,
  `calls/iter 4`, `judge-chunk 25`.
- `fix_competency_areas.py` REMAP table extended with 30+ new patterns
  (zones-as-area, bloom-verbs-as-area, underscore hallucinations,
  dash/slash track-prefix forms). All 462 malformed drafts now canonical.
- `vault-manifest.json` refreshed: questionCount 9,224 → 9,544,
  contentHash 539eb877f9cc → 0350da5706e6.
- All 8 Playwright tests pass.

### Saturation reason (carry-forward signal)

`same top-priority cell two iterations in a row — converged`. Top
priority decay 2.25 → 2.14 → 2.03 → 1.93 → 1.83 plateaued. Both halt
conditions (gap-threshold 0.8, max-calls 80) had headroom remaining;
**structural convergence fired first**. Generator cannot meaningfully
shrink `tinyml/specification/L6+` further within the current prompt
framing. **This is the central problem Phase B addresses.**

---

## Audit findings from this session (so the next session does not rediscover)

### 1. Distribution closure — PARTIAL FAILURE

The 320 PASS items did NOT close the priority gaps the analyzer flagged:

| Targeted gap | Before | After | Δ | % gap closed |
|---|---|---|---|---|
| tinyml/parallelism | 0 | 1 | +1 | 1% |
| tinyml/networking | 2 | 11 | +9 | 10% |
| **mobile/parallelism** | **0** | **0** | **+0** | **0%** |
| edge/parallelism | 11 | 13 | +2 | 1% |
| global L4–L6+ | 189 | 189 | +0 | 0% |

Where they actually landed: `mobile/memory` (16), `mobile/networking` (15),
`tinyml/cross-cutting` (13), `tinyml/power` (13), `mobile/data` (13). All
useful, none on the original priority list. **Phase B's job is to close
the actual targeted cells** with prompt templates engineered for the
content type, not just more API calls.

Why parallelism failed: judge DROPped most parallelism drafts as
"too-shallow framing" (e.g., `cloud-4490` verdict: *"Simple division of
payload by bandwidth is too trivial for L6+ Staff level"*). The fix is
**template-level, not budget-level**.

### 2. Schema completeness — STRONG (with one defect)

- 320/320 PASS items have full `details.{realistic_solution,
  common_mistake, napkin_math}` ✓
- 135/136 visual references resolve to real SVG ✓
- 1 defect: **`mobile-1962`'s graphviz render crashed silently** —
  only `.dot` source exists, no `.svg`. Judge passed it because YAML
  was structurally valid. `render_visuals.py` does not propagate
  failures.

### 3. Quality at scale — ~7.5/10 average across 10 stratified items

Strong: `edge-2431` (Jetson NvSciBuf zero-copy), `tinyml-1658` (256KB
SRAM cliff diagnosis), `mobile-1923` (UFS write-amplification),
`tinyml-1635` (closed-form duty-cycle), `edge-2313` (Hailo-8 PCIe
pipeline bubble). Math correct in all 10. Real hardware grounding in
all 10.

Weak: `edge-2423` (asks for "standard programming pattern" — too
generic, OS-textbook style).

### 4. All-checks audit

| Gate | Result |
|---|---|
| `vault check --strict` | ✓ 0 errors / 0 invariant failures |
| `vault doctor / release-integrity` | ✓ 0.1.1 verified |
| `vault doctor / content-hash-sample` | ✓ 20/20 sampled hashes match |
| `vault doctor / registry-integrity` | ✗ 5,269 missing from registry; 4,479 registry orphans |
| `vault lint` | 0 errors / **1,308 warnings** (all `zone-level-affinity`; 303 on new items, 1,005 pre-existing) |
| Playwright (8 tests) | ✓ all pass |
| Pre-commit hook | ✓ (after manifest refresh) |

**Registry drift forensics (resolved cause; not a worktree issue):**
Registry is identical across all 3 worktrees (MD5 `a9a259c559cc23b03ca371683ad81d6d`).
The 4,479 orphan registry entries are old cohort-tagged IDs
(`tinyml-exp2-desi-0184`, `cloud-fill-04027`, `tinyml-cell-13251`)
left over from commit `8a5c3ff3c`'s rename refactor that updated YAMLs
but never appended to the registry. The 5,269 disk orphans are: 4,754
renamed-INTO clean IDs + 320 from this session + ~195 prior-run
unappended items. **94% of the drift pre-existed this session.**

---

## Locked decisions (do NOT relitigate)

| Decision | Choice |
|---|---|
| **A.6 lint calibration** | Spawn 4 expert agents on a stratified sample of disputed (zone, level) pairs; consolidate via `consensus-builder`; widen rule for accepted pairs, reclassify items in rejected pairs, ack-list disputed pairs. Must hit **0 lint warnings** before proceeding. |
| **A.7 chain integrity** | Fix the data — full pass on the 29 single-question chains + 101 non-sequential. Not the relaxation shortcut. |
| **A.8 zoom UX** | `react-medium-image-zoom` (4KB, click-to-zoom modal, ESC closes). Lightest + most responsive. |
| **B.3 prompt authorship** | Claude drafts; user reviews before B.5 fires the loop. |
| **Release cadence** | One stable dev branch at the end. No mid-stream release tags. The user's framing: *"I just want the dev branch to come to a stable point for StaffML day."* |

---

## Review checkpoints (pause for user input)

1. **After A.6.3 expert consensus lands** — before applying calibration to the lint rule.
2. **After B.3 prompt drafts are written** — before B.5 fires the
   generation loop and burns API budget.
3. **Before D.2** — final atomic commit; user confirms branch is
   stable-state ready.

---

## Phase A — Cleanup (sequential, blocking everything else; ~7-8 hr)

| ID | Task | Acceptance criterion | Effort |
|---|---|---|---|
| A.1 | Re-run `render_visuals.py` for `mobile-1962`; if graphviz still crashes, fix `.dot` source or strip the `visual:` block | `interviews/vault/visuals/mobile/mobile-1962.svg` exists OR YAML's visual block removed | 10 min |
| A.2 | `render_visuals.py`: non-zero exit on any per-item crash; capture per-ID stderr to `_validation_results/render_failures.json` | Inject a broken `.dot` test; confirm exit code != 0 + log written | 30 min |
| A.3 | LinkML schema: type the `visual` block as a structured sub-schema. `kind` enum `[svg, png]`, `path` regex `^[a-z0-9-]+\.(svg\|png)$`, required `alt` (≥10 chars) + `caption` (≥5 chars) | LinkML codegen produces typed `Visual` class; existing 234 visual items still validate | 45 min |
| A.4 | Pydantic field-validator: `visual.path` MUST resolve to a real file in `visuals/<track>/`; reject otherwise | Unit test: YAML with `visual.path: nonexistent.svg` fails `Question.model_validate()` | 30 min |
| A.5 | Registry repair: write `tools/repair_registry.py` reading disk → appending 5,269 missing IDs as `created_by: registry-rebuild-2026-04-25`. Add comment block above the new entries documenting the rename history. Refactor `doctor.py:_check_registry_integrity` into two checks: `disk-coverage` (HARD FAIL if disk file unregistered) and `registry-history` (INFO only for retired IDs). | `vault doctor` shows `disk-coverage: pass`; `registry-history: info`. Registry is append-only (no deletions). | 1 hr |
| A.6 | **Expert-driven lint calibration** (replaces the original "empirical widen" version). See A.6.* breakdown below. | `vault lint interviews/vault/questions/` reports 0 errors / **0 warnings** | 2 hr |
| A.7 | Chain integrity: 29 single-question chains + 101 chains with non-sequential positions. Audit each → fix the chain (renumber positions / extend with siblings) or drop the chain entirely. | Pre-commit hook reports 0 chain warnings | 1.5 hr |
| A.8 | Practice page: render visual inline beside question + click-to-zoom modal using `react-medium-image-zoom`. Add Playwright test: load known-visual question, click image, verify modal opens, press ESC, verify modal closes. | Playwright count 8 → 9, all pass | 1.5 hr |
| A.9 | Cleanup verification gate: `vault check --strict` 0 errors • `vault lint` 0 warnings • `vault doctor` 0 fails • Playwright 9/9 • all 320 prior PASS items still in corpus | All five gates green | 15 min |
| A.10 | Atomic commit: `cleanup(vault): registry repair + visual schema + lint calibration + zoom UI` | Pre-commit hook passes without `--no-verify` | 5 min |

### A.6 expanded — expert-driven lint calibration

| Step | Action | Acceptance |
|---|---|---|
| A.6.1 | Pull all 1,308 zone-level-affinity warns; group by (zone, level) pair; pick 3-5 representative questions per disputed pair as evidence | Manifest file `tools/lint_calibration_evidence.yaml` with ~30-50 disputed-pair samples |
| A.6.2 | Spawn 4 expert agents in **parallel**: `expert-vijay-reddi`, `expert-chip-huyen`, `expert-jeff-dean`, `education-reviewer`. Each gets the same disputed-pair manifest + the question: *"for each (zone, level) pair, is it pedagogically valid? give your reasoning."* | 4 expert reports written to `.claude/_reviews/lint-calibration-<ts>/` |
| A.6.3 | **(USER REVIEW CHECKPOINT 1)** — surface the four expert reports for user review before consolidation | User signs off |
| A.6.4 | Consolidate via `consensus-builder` agent: every (zone, level) pair gets a verdict: `accepted` (≥3 experts say valid), `rejected` (≥3 say invalid), `disputed` (split) | Consensus report with verdict per pair |
| A.6.5 | For `accepted` pairs → widen lint rule. For `rejected` pairs → reclassify the affected questions (update zone or level field, vault check still passes). For `disputed` pairs → ack-list with rationale. | Updated `zone_level_affinity.yaml` rule + reclassified items committed |
| A.6.6 | Re-run `vault lint interviews/vault/questions/` → must report **0 warnings, 0 errors** | Strict pass |

---

## Phase B — Full balanced generation (after A.10 lands; ~9-10 hr)

The original Phase 1 analyzer flagged 100 cells. The first run hit
~30 of those and PASS-ed at unusual cells (mobile/memory etc.) rather
than the priority cells. Phase B systematically attacks the full
list with prompts engineered for the actual content type needed.

| ID | Task | Acceptance criterion | Effort |
|---|---|---|---|
| B.1 | Re-run analyzer against current corpus (post-cleanup): get fresh 100-cell recommended plan | Plan file written; top 20 inspected | 5 min |
| B.2 | Cell-class triage: read the 100 cells, group by failure mode the first run revealed: `parallelism-too-shallow`, `global-L6+-too-abstract`, `healthy-fillable`. Each class gets its own prompt template. | `tools/cell_triage.md` written: list of cells × class × prompt-template ref | 1 hr |
| B.3 | Author **3 specialized generator prompts**, one per failure class. **Parallelism** prompt: requires concrete topology (NVLink, IB, PCIe, RoCE, LoRa), forbids pure bandwidth division, requires synchronization or bubble cost in the question. **Global-L6+** prompt: requires cross-track synthesis (e.g., compare same constraint in tinyml + cloud), forbids generic abstractions. **Standard** prompt: refined version of current with validate-at-write fix. | 3 prompt files in `interviews/vault/scripts/prompts/`; test invocation against each produces 5 sample drafts that pass judge | 2 hr |
| B.3' | **(USER REVIEW CHECKPOINT 2)** — surface prompt drafts for user review before B.5 | User signs off | — |
| B.4 | Add validate-at-write to `gemini_cli_generate_questions.py`: every YAML round-trips through `Question.model_validate()` before write. Failures → retry once with "your previous output had X violations" prompt. Second failure → log structured error and skip. **This is the root-cause fix for the competency_area regression.** | Unit test: feed Gemini-style malformed dict → script rejects, retries, eventually skips with structured error | 1 hr |
| B.5 | Two-stage loop: Stage 1 — 30-call run targeting all 100 cells with appropriate prompt class, batch_size 30 → ~900 drafts. Stage 2 — judge in chunks of 25; re-judge any NEEDS_FIX after one auto-fix retry pass. | Loop summary shows: drafts ≥ 800, PASS rate ≥ 60%, items in priority cells (parallelism + global L6+) ≥ 80 | 4-5 hr wall clock, 50-70 calls |
| B.6 | Stratified spot-read: 20 items across (track × prompt-class × verdict). Reject drafts that read as bandwidth-math or "standard programming pattern." | Reviewed list saved; rejection rate ≤ 15% | 30 min |
| B.7 | Promote PASS items, rebuild bundle, regen paper macros, recompile PDF | `vault check --strict` clean; corpus published count grows by 200-500; macros stamped | 30 min |

---

## Phase C — NEEDS_FIX queue (parallel with B.5/B.6 once A.10 lands; ~2.5 hr)

This run's 120 NEEDS_FIX items each carry a specific `fix_suggestion`
from the judge (see `_validation_results/coverage_loop/20260425_150712/iter_*/judge_summary.json`).

| ID | Task | Acceptance | Effort |
|---|---|---|---|
| C.1 | Aggregate the 120 NEEDS_FIX from this run + any new from Phase B into a single fix manifest with per-item `fix_suggestion` + criteria flags | Manifest file written, ≥120 entries | 15 min |
| C.2 | Spawn `general-purpose` fix-agent with `quiz-generation.md` as quality bar; agent edits each YAML in place applying the judge's specific suggestion | Each YAML modified; `vault check --strict` still passes | 1.5 hr |
| C.3 | Re-judge fixed items in a small chunked run (~3-5 calls) | Verdict distribution recorded | 30 min |
| C.4 | Promote any items that flipped to PASS | Promoted count logged | 5 min |

**Concurrency safety:** Phase C touches *existing* NEEDS_FIX YAMLs;
Phase B writes *new* IDs. Different ID ranges → no write race. Both
phases must NOT run while Phase A is in flight (schema/lint changes).

---

## Phase D — Final stable state (after B + C; ~1 hr)

| ID | Task | Acceptance | Effort |
|---|---|---|---|
| D.1 | Re-run all gates: `vault check --strict` • `vault lint` (0 warnings) • `vault doctor` (0 fails) • Playwright (9/9) • paper compile (0 LaTeX errors) • registry append-only invariant verified. Wrap as `tools/release_gate.sh`. | Single shell script returns exit 0 | 30 min |
| D.2 | **(USER REVIEW CHECKPOINT 3)** — surface final state to user. | User signs off | — |
| D.3 | Atomic final commit: `feat(vault): release-ready cleanup + balanced generation` | Pre-commit clean; branch ready for StaffML day | 10 min |

---

## Common saturation outcomes (mirroring prior plan)

If Phase B's loop stops early:

| Reason | Meaning | What to do |
|---|---|---|
| `top priority gap < 0.8` | Corpus is balanced enough no cell desperately empty | Success. Move to B.6. |
| `DROP rate > 35%` | Gemini hallucinating or cells nonsensical | Inspect latest iter `judge_summary.json`; add to `TRACK_TOPIC_BLOCKLIST` in `analyze_coverage_gaps.py`. Likely indicates a prompt template needs another revision. |
| `same top cell two iters in a row` | Generator cannot fill the cell | Check raw Gemini output for that cell. Likely needs even more specialized prompt. **This is what fired in the prior run.** |
| `max-iters reached` | Hit iteration cap before saturation | Re-run with higher `--max-iters 50` if budget allows. |
| `max-calls reached` | Burned through API budget | Stop. Ship Phase C first. |

---

## What NOT to do

- ❌ Don't merge to `dev` until Phase D passes (pre-commit hook + all gates green).
- ❌ Don't push to remote without explicit user OK.
- ❌ Don't run Phase B or C concurrent with Phase A in-flight.
- ❌ Don't add `Co-Authored-By` lines or automated attribution footers.
- ❌ Don't change schema enum values (CompetencyArea, Track, Level, Zone, Status, Provenance) without explicit user direction.
- ❌ Don't auto-promote NEEDS_FIX items without re-judge.
- ❌ Don't suppress lint warnings or skip pre-commit hooks (`--no-verify` forbidden).
- ❌ Don't relitigate the locked decisions above without explicit user direction.
- ❌ Don't navigate to or modify files in sibling worktrees (`MLSysBook`, `MLSysBook-vault-audit`, `MLSysBook-404`, `MLSysBook-labs-release`). Stay in `MLSysBook-massive-build`.
- ❌ Don't auto-cut a release tag (`v0.1.2` etc.) — single stable commit is the goal, not a release ceremony.

---

## Files of interest

| File | Why |
|---|---|
| `interviews/vault/docs/RESUME_PLAN_2026-04-25.md` | Prior session's plan (completed through Phase 7). |
| `interviews/vault/docs/MASSIVE_BUILD_RUNBOOK.md` | Methodology document — the prior session's runbook. |
| `interviews/vault/_validation_results/coverage_loop/20260425_150712/` | Last loop's per-iter judge_summary.json (PASS/NEEDS_FIX/DROP details with fix_suggestion). |
| `interviews/vault/scripts/iterate_coverage_loop.py` | Main driver. Defaults bumped this session. |
| `interviews/vault/scripts/analyze_coverage_gaps.py` | Priority ranking. |
| `interviews/vault/scripts/gemini_cli_generate_questions.py` | Batched Gemini generation. **Phase B.4 adds validate-at-write here.** |
| `interviews/vault/scripts/gemini_cli_llm_judge.py` | Multi-criteria validator. |
| `interviews/vault/scripts/render_visuals.py` | DOT/matplotlib → SVG. **Phase A.2 fixes silent-failure mode here.** |
| `interviews/vault/scripts/fix_competency_areas.py` | One-time cleanup. REMAP table extended this session. |
| `interviews/vault/scripts/promote_validated.py` | Lifecycle flip. |
| `interviews/vault-cli/src/vault_cli/commands/doctor.py` | **Phase A.5 splits `_check_registry_integrity` into two checks.** |
| `interviews/vault-cli/src/vault_cli/commands/lint.py` | **Phase A.6 updates `zone_level_affinity` rule.** |
| `interviews/vault/id-registry.yaml` | Append-only ID log. **Phase A.5 appends 5,269 missing IDs.** |
| `interviews/staffml/src/data/vault-manifest.json` | GUI's authoritative count. Refresh after every bundle build. |
| `.claude/agents/expert-*.md` | Expert agent definitions for A.6.2. |
| `.claude/agents/consensus-builder.md` | Consensus aggregator for A.6.4. |

---

## One-liner status check (run first in next session)

```bash
cd /Users/VJ/GitHub/MLSysBook-massive-build && \
  git log --oneline -3 && echo "---" && \
  git status --short | head -10 && echo "---" && \
  PYTHONPATH=interviews/vault-cli/src \
    python3 -m vault_cli.main check --strict 2>&1 | tail -3 && \
  echo "---" && \
  PYTHONPATH=interviews/vault-cli/src \
    python3 -m vault_cli.main lint interviews/vault/questions/ 2>&1 | tail -3 && \
  echo "---" && \
  PYTHONPATH=interviews/vault-cli/src \
    python3 -m vault_cli.main doctor 2>&1 | tail -10 && \
  echo "---" && \
  python3 -c "
import json
c = json.load(open('interviews/staffml/src/data/corpus.json'))
print(f'published: {len(c)}')
"
```

If the output shows commit `ece6eccf2`, clean tree, `vault check`
passes, lint reports 1,308 warnings, doctor shows registry fail
(5,269/4,479) — the resume state is healthy and matches this plan's
starting assumptions. **Proceed to Phase A.1.**

If something differs, **stop and reconcile** before starting work.

---

## Pacing

This is a ~17-19 hour push, plausibly 2 focused days or 3-4 calendar
days with breaks. The work is heavy on prompt engineering (B.3) and
data-cleanup (A.6, A.7). Don't rush; the gates are the contract.

Three explicit user-review checkpoints (A.6.3, B.3', D.2). Wait for
sign-off at each before continuing.
