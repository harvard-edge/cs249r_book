# StaffML Vault — Corpus Hardening Plan

**Status:** draft for user review (2026-05-02)
**Branch:** `yaml-audit`
**Worktree:** `/Users/VJ/GitHub/MLSysBook-yaml-audit`
**Supersedes:** `RELEASE_AUDIT_PLAN.md` (which assumed sampling; we now know full-corpus audit is feasible at ~270 calls)

---

## 1. Vision in one paragraph

At the end of this plan, every one of the 9,446 published questions in the
StaffML vault conforms to a strict, end-to-end-enforced schema. The
markup conventions (Pitfall/Rationale/Consequence in `common_mistake`;
Assumptions/Calculations/Conclusion in `napkin_math`) are
**load-time-enforced** by Pydantic and gated by `vault check --strict`,
not just convention. Math correctness has been independently
Gemini-verified across the full corpus. Level fit, scenario coherence,
physical realism, and vendor-name fabrication have been audited at
corpus scale via a single batched script that costs ~300 Gemini calls
(one day at the 250/day cap). New contributors can author a passing
question from `interviews/vault/AUTHORING.md` alone, and `vault new`
scaffolds a YAML with the full template stubs in place. A monthly CI
workflow re-audits the corpus and posts a regression report. New format
violations are *impossible to land* because the gate is in the
validator, not in a one-shot audit.

---

## 2. End-state acceptance criteria

These are testable. The plan is "done" when **every** item below is true.

### 2.1 Schema and validation

- [ ] `interviews/vault/schema/question_schema.yaml` (LinkML) has `pattern`
      constraints on `Details.common_mistake` and `Details.napkin_math`
      that match the Pitfall/Rationale/Consequence and
      Assumptions/Calculations/Conclusion markers respectively.
- [ ] `vault codegen` propagates those patterns into Pydantic
      (`vault_cli/src/vault_cli/models.py`) such that loading a YAML
      with malformed markers raises `ValidationError`.
- [ ] `vault check --strict` includes a structural-tier invariant
      `format-compliance` that fails on any published question lacking
      the required markers.
- [ ] `Details` model: `extra="forbid"` (no silent acceptance of
      unknown keys). Every legitimate extra field is added to the
      explicit attribute list.
- [ ] `provenance` is a `required` field in the LinkML schema (no
      default fallback at YAML load time).

### 2.2 Corpus content

- [ ] All 9,446 published questions pass `vault check --strict` (0 invariant failures).
- [ ] Format-failure rate: **0** (currently 861 / 9,446 = 9.1%).
- [ ] Placeholder-title count: **0** (currently 134 in `global/`).
- [ ] Provenance-field-absent count: **0** (currently 407).
- [ ] Math correctness: full-corpus Gemini audit run in last 30 days,
      with all flagged failures triaged (fixed, deprecated, or acked).
- [ ] Level fit / coherence / vendor / physical-realism: same.
- [ ] No unreviewed `validation_*` or `math_*` cruft fields with
      non-null values older than 60 days.

### 2.3 Authoring infrastructure

- [ ] `interviews/vault/AUTHORING.md` exists. Includes: required-field
      table, the two markup conventions with worked examples, title
      conventions (≤ 120 chars, no trailing period, no LaTeX), level↔Bloom
      mapping table, gotchas (no `IO` use `I/O`, no `_` in titles, etc.),
      and 1 reference question per (level, common-track) cell.
- [ ] `vault new` scaffold includes `common_mistake:` and `napkin_math:`
      template stubs with the markers pre-written; authors fill in the
      content between markers.
- [ ] `vault new --help` cross-references `AUTHORING.md`.
- [ ] `vault edit` re-validates against the same schema rules so an
      existing question can't drift into format non-compliance.

### 2.4 Audit infrastructure

- [ ] Single new script `audit_corpus_batched.py` exists under
      `interviews/vault-cli/scripts/`. Batches 30-40 questions per
      Gemini call; one prompt per batch returns per-question verdicts
      across all judge dimensions plus optional suggested corrections.
- [ ] `_pipeline/runs/<UTC>/AUDIT_REPORT.md` is the output convention.
- [ ] CI workflow `.github/workflows/staffml-audit-corpus-monthly.yml`
      runs the audit on a cron, opens an issue (or PR with the report)
      if regressions exceed thresholds.
- [ ] Old / wrong-design tooling deleted: `audit_corpus.py` (this
      session's dead-end), all Gemini scripts in
      `interviews/vault/scripts/` listed in DEPRECATED.md.

### 2.5 Pipeline cleanliness

- [ ] `pytest interviews/vault-cli/tests/` — 74/74 (or higher; new tests welcome).
- [ ] `ruff check interviews/vault-cli` — clean.
- [ ] `mypy interviews/vault-cli/src` — clean (currently non-blocking).
- [ ] `vault build --local-json` — succeeds; `releaseHash` rolls
      forward; `published_count` ≥ 9,446.

### 2.6 Documentation

- [ ] `paper.tech` updated with post-audit corpus stats (counts, pass
      rates, methodology paragraph).
- [ ] `CHAIN_ROADMAP.md` Progress Log has an entry per phase below.
- [ ] `RELEASE_AUDIT_PLAN.md` carries a "superseded by
      CORPUS_HARDENING_PLAN.md" header.

---

## 3. Current baseline (numbers)

Snapshot taken 2026-05-02 against `vault build --local-json` on
worktree `yaml-audit @ 963fbfb16`.

| metric | value |
|---|---|
| `vault check --strict` | 10,711 loaded, 0 invariant failures |
| `vault build` published count | 9,446 |
| chains.json | 843 chains (after Δ=0 drop on 2026-05-02) |
| `releaseHash` | `5a4783e62d…` |
| pytest | 74 / 74 passing |
| ruff | clean |
| Schema (Pydantic load) pass rate | **100%** (all 9,446 valid) |
| Format-marker pass rate | **90.9%** (8,585 / 9,446) |
| Format failures total | **861** |
| `common_mistake` failures | 414 (only) + 164 (both) = **578** — **all** are pure prose, no markers at all |
| `napkin_math` failures | 283 (only) + 164 (both) = **447** — 212 pure prose, 118 use `**Result:**` style alts, 49 partial canonical, 68 other-bold-marker shapes |
| Per-track format-fail rate | cloud 9.5%, edge 10.9%, tinyml 11.0%, mobile 5.5%, global 6.1% |
| Provenance field absent | **407** |
| Placeholder titles (`Global New NNNN`) | **134** (all in `global/`) |
| `validated: true` / stale | unaudited; needs survey |
| Math correctness audited | **<0.2%** (Phase 3 audited 9 drafts) |
| Coherence / level fit audited | only Phase 3 drafts |

Detailed per-row data lives in
`interviews/vault/_pipeline/format-audit-full.json` (regenerable;
gitignored).

---

## 4. Phased plan

The plan is 9 phases. Phases are gated — each one's exit criteria are
the next one's entry assumptions.

### Phase 0 — Cleanup (no Gemini cost; ~30 min)

**Goal:** clear the decks of dead/wrong tooling so the rest of the work
lands on a clean base.

**Steps:**

1. Commit the staged deprecated-script deletion (already staged: 18 files in
   `interviews/vault/scripts/` per `DEPRECATED.md`'s replaced-by table, plus
   the deletion of the now-empty `archive/` subdir, plus the updated
   `DEPRECATED.md` referencing the removal).
2. Delete `interviews/vault-cli/scripts/audit_corpus.py` (this session's
   dead-end design — built on `audit_math.py`'s 1-call-per-Q skeleton
   instead of the right batched pattern).
3. Delete `interviews/vault/_pipeline/audit-progress.json`,
   `audit-cost-ledger.json`, `audit-sample.json` (orphans of the
   dead-end audit_corpus.py).

**Validation:**
- `vault check --strict` ✓ (already verified post-deletion)
- pytest 74/74 ✓
- ruff clean ✓

**Deliverable:** one commit `chore(vault): remove deprecated scripts and dead-end audit_corpus.py`.

---

### Phase 1 — Mechanical backfill (no Gemini cost; ~1 hour)

**Goal:** clean the regex-fixable issues and surface what's left.

**Steps:**

1. **Provenance backfill.** Script:
   `interviews/vault-cli/scripts/backfill_provenance.py`. For each
   published YAML lacking the `provenance:` field, write
   `provenance: imported` immediately after the `status:` line.
   - Touches 407 YAMLs.
   - Idempotent (re-runs after a partial run pick up where they left off).
   - Round-trip-safe: uses `ruamel.yaml` (preserves comments + ordering)
     OR plain string-insertion (same line discipline as `vault edit`'s
     error-comment injector). Pick whichever the existing CLI uses;
     match it for consistency.
   - Report stdout: `before=X, after=Y, untouched=Z` so a CI bot could
     wrap this safely.

2. **Sanity-check `provenance` defaulting.** Confirm Pydantic still
   loads correctly. Pydantic was already silently substituting
   `imported` at load time (default value), so explicit YAML and
   default-fallback should be functionally equivalent.

3. **`**Result:** → `**Conclusion:**` rename pass — DEFER.** While 56
   `napkin_math` blocks use `**Result:**` instead of `**Conclusion:**`,
   we cannot guarantee `**Result:**` is functioning as the conclusion
   (it could be a mid-calculation result label). Defer this rewrite to
   Phase 5's Gemini correction pass, which has the semantic
   understanding to judge per-instance.

**Validation:**
- After provenance backfill: rerun `format-audit-full.json` regen,
  expect provenance failures to drop from 407 to 0.
- `vault check --strict` ✓
- `vault build --local-json` ✓ (should produce same `releaseHash` since
  `provenance` is a leaf in the Merkle tree but its value didn't change
  for any question — was already `imported` at runtime via default).
- pytest 74/74 ✓

**Deliverable:** one commit `chore(vault): explicit provenance: imported on 407 questions`.

---

### Phase 2 — Authoring documentation + scaffolding (no Gemini cost; ~3 hours)

**Goal:** make the format conventions discoverable and easy to author against.

**Steps:**

1. **Write `interviews/vault/AUTHORING.md`.** Required sections:
   - **Required fields** (table, with each field's Pydantic constraint
     and an example value).
   - **The two markup conventions:**
     - `common_mistake`: Pitfall/Rationale/Consequence. Worked examples
       at 3 different levels.
     - `napkin_math`: Assumptions & Constraints / Calculations /
       Conclusion. Worked examples with both bullet-style and
       paragraph-style calculations.
   - **Title conventions:** ≤ 120 chars, no trailing period, no
     LaTeX, no `_` (underscores break LaTeX index tags), prefer
     descriptive nouns ("KV Cache Bandwidth Bottleneck on H100" not
     "KV Cache Q1").
   - **Level ↔ Bloom mapping table** (the same one in `models.py` and
     ARCHITECTURE.md, but contextualized for authoring decisions).
   - **Zone × Bloom affinity table** (since the validator enforces this
     and authors hit it).
   - **Gotchas:**
     - No `IO`; use `I/O`.
     - No curly apostrophes (`’`); use straight (`'`).
     - No `**bold**` or markdown in titles or `\index{}` keys.
     - Prefer concrete vendor names over generic ("Apple Neural Engine"
       not "the on-device accelerator") — but only when the vendor
       actually exists.
   - **Reference questions:** 1 well-formed per `(level, track)` cell
     (= 30 references for the 5×6 grid). Cite by `id`.
   - **How to test your draft:**
     ```bash
     vault check --strict
     python3 interviews/vault-cli/scripts/validate_drafts.py --no-llm-judge
     ```
   - **The end-to-end flow:**
     `vault new → vault edit → vault check --strict → git commit`.

2. **Extend `vault new` scaffold** (`vault_cli/src/vault_cli/commands/new.py`,
   plus the Jinja template at `vault_cli/src/vault_cli/templates/question.yaml.j2`
   if that's where it lives — locate during implementation):
   - Inject template stubs for `common_mistake` and `napkin_math` with
     the markers pre-written, e.g.:
     ```yaml
     common_mistake: |
       **The Pitfall:** <TODO: the wrong intuition or shortcut a candidate takes>
       **The Rationale:** <TODO: why that intuition is wrong, in one sentence>
       **The Consequence:** <TODO: the operational symptom — latency, cost, failure mode>
     napkin_math: |
       **Assumptions & Constraints:**
       - <TODO: assumption 1>

       **Calculations:**
       - <TODO: calc step 1>

       **Conclusion:** <TODO: one-sentence interpretation of the result>
     ```
   - The `<TODO:>` markers should be exactly the form `vault edit`
     already recognizes for "draft is incomplete" detection (verify
     during implementation).

3. **Add `vault new --help` line referencing AUTHORING.md.**

4. **Add unit test** `test_vault_new_scaffold.py`:
   - Run `vault new --topic foo --track cloud --level l1 --zone recall` in
     a tmpdir.
   - Assert the resulting YAML, with stubs replaced by trivial-but-valid
     content, **passes** `gate_format_compliance` regex.

5. **Add `AUTHORING.md` to the validated-doc list** in pre-commit if
   one exists (search for one); else just rely on PR review.

**Validation:**
- `vault new` produces a YAML that, after the operator does s/<TODO:>/
  with valid content, passes `vault check --strict` AND
  `gate_format_compliance`.
- Unit test green.

**Deliverable:** two commits:
- `docs(vault): AUTHORING.md — single-source authoring reference`
- `feat(vault-cli): vault new scaffolds full Pitfall/Rationale/Consequence + Assumptions/Calculations/Conclusion stubs`

---

### Phase 3 — Build the batched audit + correction tool (~3 hours; no Gemini cost during build)

**Goal:** one tool that audits AND proposes corrections at corpus scale.

**Specification.**

New file:
`interviews/vault-cli/scripts/audit_corpus_batched.py`.

**Architecture:**
- Batches 30-40 published questions per Gemini call (~80K-token prompt).
- One prompt per batch returns a JSON array, one entry per question:
  ```json
  {
    "qid": "<id>",
    "format_compliance": "pass" | "fail",
    "format_issues": ["common_mistake missing **The Pitfall:**", ...],
    "level_fit":     "pass" | "fail" | "skip",
    "level_fit_rationale": "<sentence>",
    "coherence":     "pass" | "fail",
    "coherence_failure_mode": "physical_absurdity" | "vendor_fabrication" | "mismatch" | "arithmetic" | "none",
    "coherence_rationale": "<sentence>",
    "math_correct":  "pass" | "fail" | "no_math",
    "math_errors":   ["<specific issue>", ...],
    "title_quality": "good" | "placeholder" | "malformed",
    "suggested_corrections": {                  // ONLY when --propose-fixes
      "title":            "<rewritten title>",
      "common_mistake":   "<reformatted with markers>",
      "napkin_math":      "<reformatted with markers>"
    }
  }
  ```
- Per-question cost: 1 / batch_size of a Gemini call. At 30/batch the
  full 9,446-question corpus costs ~315 calls. At 40/batch ~236 calls.
  Both fit in one day's 250 cap if we run two days OR raise to 500/day
  for a session.

**Reuses (don't reimplement):**
- `batch_chains()` packing logic from `audit_chains_with_gemini.py` —
  generalize as `pack_batches(payloads, max_chars)`.
- The four judge prompts from `validate_drafts.py` (level_fit /
  coherence / bridge / format_compliance) — extract into a shared
  `judges.py` module.
- The `SCHEMA_SUMMARY` text block from `generate_question_for_gap.py` —
  feed as ground-truth context to every prompt.
- `_pipeline/runs/<UTC>/` output convention from `audit_chains_with_gemini.py`.
- The `call_gemini()` subprocess wrapper from `audit_math.py`.

**New helpers (extract during refactor):**
- `interviews/vault-cli/scripts/_judges.py` — single source of truth
  for all judge prompts. Imported by both `validate_drafts.py` (drafts)
  and `audit_corpus_batched.py` (corpus).
- `interviews/vault-cli/scripts/_batching.py` — `pack_batches`,
  `MAX_PROMPT_CHARS`.

**CLI:**
```bash
audit_corpus_batched.py [--all | --tracks cloud,edge | --since-commit <sha>]
                        [--max-calls 250]
                        [--batch-size 35]
                        [--propose-fixes]              # adds suggested_corrections to each row
                        [--output _pipeline/runs/<UTC>/]
                        [--dry-run]
```

**Three modes:**
- **audit-only** (default): produces verdicts; no rewrites. Cost ~270 calls full-corpus.
- **--propose-fixes**: also returns suggested correction strings. Cost ~+50% (longer prompt + more tokens out). ~400 calls full-corpus.
- **--apply-fixes**: NOT a mode — Gemini's proposed fixes are NEVER auto-applied. They are written to `_pipeline/runs/<UTC>/proposed-corrections.json` for human review (Phase 4 reviews them).

**Output:**
- `_pipeline/runs/<UTC>/00_config.json` — CLI args, seed, model.
- `_pipeline/runs/<UTC>/01_audit.json` — list of per-question rows.
- `_pipeline/runs/<UTC>/02_proposed_corrections.json` — only when `--propose-fixes`.
- `_pipeline/runs/<UTC>/AUDIT_REPORT.md` — synthesis (one extra Gemini call).

**Validation (pre-merge of the script itself):**
- Unit-test the prompt construction (golden file).
- Smoke test: 1 batch of 30 questions, `--propose-fixes`. Verify
  output schema.
- ruff clean; mypy clean.

**Deliverable:** four commits:
- `refactor(vault-cli): extract judge prompts to _judges.py module`
- `refactor(vault-cli): extract batching helper to _batching.py module`
- `feat(vault-cli): audit_corpus_batched.py — single-call corpus audit`
- `test(vault-cli): smoke test for audit_corpus_batched batching`

---

### Phase 4 — Run the audit-only pass + first-pass triage (1 day Gemini; ~270 calls)

**Goal:** produce the corpus-wide audit signal.

**Steps:**

1. **Run** `audit_corpus_batched.py --all --batch-size 35` against the
   full 9,446-question corpus. ~270 calls. Wall clock: ~4 hours at
   4-way parallelism with the existing 4-second inter-call delay.
2. **Surface findings** in a triage doc
   `interviews/vault-cli/docs/AUDIT_FINDINGS_<DATE>.md`:
   - Format compliance: ID list + per-track breakdown.
   - Math errors: ID list + brief description per question.
   - Level inflation: ID list + claimed-vs-actual level.
   - Coherence failures: split by failure_mode (physical_absurdity / vendor_fabrication / mismatch / arithmetic).
   - Title problems: ID list with proposed-replacement candidates (from later phase).
3. **Compare against `format-audit-full.json`** to confirm the regex
   audit and the Gemini audit agree on format_compliance. If they
   disagree, that's a bug to investigate (one prompt or one regex is wrong).

**Stopping rule:** if any cell shows >30% failure rate on a non-format
gate, pause and surface to user before correcting — that bucket has a
systematic issue worth understanding before mass-rewriting.

**Validation:**
- `_pipeline/runs/<UTC>/01_audit.json` exists with rows for every
  qid in the published corpus.
- AUDIT_FINDINGS_<DATE>.md committed.
- format_compliance failures from Gemini ≈ 861 (the regex baseline).

**Deliverable:** two commits:
- `audit(vault): full-corpus audit run <date> — N format / M math / K coherence flags`
- `docs(vault): AUDIT_FINDINGS_<date>.md — triage doc for review`

---

### Phase 5 — Run --propose-fixes, human review, apply (1 day Gemini; ~150 calls)

**Goal:** get Gemini-proposed corrections for the 861 format failures
+ N math/coherence failures from Phase 4. Human review accepts or
rejects each.

**Steps:**

1. **Run** `audit_corpus_batched.py --propose-fixes` on the failing
   subset (not full corpus — only the rows flagged in Phase 4). ~150
   calls.
2. **Diff-review** the proposed corrections per question. Gemini writes
   to `_pipeline/runs/<UTC>/02_proposed_corrections.json`; a small
   helper script `apply_corrections.py`:
   - Reads `02_proposed_corrections.json`.
   - For each correction, prints a side-by-side diff and prompts
     `[a]ccept / [r]eject / [e]dit / [s]kip`.
   - On `accept`, writes the YAML.
   - On `edit`, opens `$EDITOR` with the diff.
   - Logs the disposition to `02_proposed_corrections.disposition.json`.
3. **For format-only fixes**, batch-accept candidates that:
   - Add the canonical markers without changing prose content (the easy 80%).
   - Are reviewed by `vault check --strict` after each accept.
4. **For math-error fixes**, REQUIRE per-instance human review. Gemini
   may have miscalculated.
5. **For level-inflation fixes**, the disposition options are:
   - Accept the level-relabel (e.g. L4 → L2).
   - Rewrite the question to actually be L4 (deferred manual work).
   - Mark `status: flagged` for later (preserves audit trail).
   - Mark `status: deprecated` (soft-delete).
6. **Apply accepted corrections.** Each accepted correction is one
   commit (use `vault edit` to ensure schema validation).

**Stopping rules:**
- If `vault check --strict` fails after any apply, stop and revert.
- If the human reviewer disagrees with >30% of Gemini's proposed
  fixes for a category, stop and re-prompt — the prompt is wrong.

**Validation per pass:**
- After each commit: `vault check --strict`, pytest, ruff, build.
- After all corrections: regen `format-audit-full.json`. Expect
  format-failure count to fall toward 0.

**Deliverable:** N small commits, organized as:
- `fix(vault): format markers — common_mistake batch 1 (50 questions)`
- `fix(vault): format markers — common_mistake batch 2 ...`
- `fix(vault): napkin_math markers — batch 1`
- ... etc.
- `fix(vault): math errors — N questions, see AUDIT_FINDINGS_<date>.md`
- Aim: ≤ 20 commits total. One per logical category and batch.

---

### Phase 6 — Tighten the schema + lift the format gate into the validator (~2 hours)

**Goal:** move format-compliance from "convention" to "enforced
invariant." Now that the corpus is clean, this is safe.

**Steps:**

1. **Update LinkML schema** `interviews/vault/schema/question_schema.yaml`:
   ```yaml
   Details:
     attributes:
       common_mistake:
         range: string
         pattern: '(?s).*\*\*The Pitfall:\*\*.*\*\*The Rationale:\*\*.*\*\*The Consequence:\*\*.*'
       napkin_math:
         range: string
         pattern: '(?s).*\*\*Assumptions.*\*\*Calculations:\*\*.*\*\*Conclusion.*'
   Question:
     attributes:
       provenance:
         required: true     # was implicitly defaulted; now explicit
   ```

2. **Set `extra="forbid"`** on `Details` Pydantic model (was `allow`).
   Surface every legitimate extra field as an explicit attribute.
   Survey what's there:
   ```bash
   python3 -c "
   import yaml
   from pathlib import Path
   from collections import Counter
   keys = Counter()
   for p in Path('interviews/vault/questions').rglob('*.yaml'):
       d = yaml.safe_load(p.read_text())
       if d and isinstance(d.get('details'), dict):
           keys.update(d['details'].keys())
   print(keys.most_common(40))
   "
   ```
   Add legitimate keys to the model; reject the rest by failing CI.

3. **Run** `vault codegen` to propagate. Ensure Pydantic /
   d1-schema.sql / TS types all roll forward.

4. **Lift `gate_format_compliance` into `validator.py`'s
   `structural_tier`:**
   ```python
   def structural_tier(loaded, vault_dir):
       failures = ...  # existing
       failures += _format_compliance(loaded)
       return failures

   def _format_compliance(loaded):
       fails = []
       for lq in loaded:
           if lq.question.status != "published":
               continue
           # Pattern check (Pydantic catches it on load anyway, but
           # the structural-tier check gives a clean error message)
           ...
       return fails
   ```

5. **Update `validator.py` tests** to cover the new invariant.

6. **Update CI workflow** `staffml-validate-vault.yml`:
   - Already runs `vault check --strict` — picks up the new gate
     automatically.
   - Add a step: if the format-failure delta vs. prior commit > 0,
     post a friendly review-comment.

**Validation:**
- `vault check --strict` returns 0 failures (corpus is now clean).
- pytest 74+/74+ (new tests for the structural-tier addition).
- `vault codegen --check` clean.
- `vault build --local-json` rolls `releaseHash` forward.

**Deliverable:** four commits:
- `feat(vault/schema): pattern constraints on common_mistake and napkin_math`
- `chore(vault): codegen propagation — Pydantic, SQL DDL, TS types`
- `feat(vault-cli): format-compliance as structural-tier invariant`
- `test(vault-cli): cover format-compliance invariant`

---

### Phase 7 — Title-quality pass (134 placeholder titles; ~30 Gemini calls)

**Goal:** retire the `Global New NNNN` placeholders.

**Steps:**

1. Use the existing `audit_corpus_batched.py` infrastructure with a
   new mode `--propose-titles`:
   ```python
   # adds to suggested_corrections:
   "title": "<3-8 word descriptive title>"
   ```
2. Run on the 134 placeholder-titled questions in `global/`. ~5
   questions per call → ~27 calls.
3. Human-review each proposed title. Accept / edit / reject via the
   same `apply_corrections.py` flow.
4. Apply accepted titles via `vault edit`.

**Validation:**
- Re-run `format-audit-full.json` regen (titles are not part of
  format-compliance, but verify nothing broke).
- 0 occurrences of `Global New` substring in any title.

**Deliverable:**
- `fix(vault/global): replace 134 placeholder titles with descriptive titles`
  (single commit, since the changes are localized).

---

### Phase 8 — Cron the audit (~30 min)

**Goal:** make audit findings a routine artifact, not a one-shot.

**Steps:**

1. **Create** `.github/workflows/staffml-audit-corpus-monthly.yml`:
   - Triggers: `schedule: cron: "0 14 1 * *"` (1st of month, 14:00 UTC),
     `workflow_dispatch`.
   - Job:
     - Checkout dev + install vault-cli.
     - Run `audit_corpus_batched.py --all --batch-size 35`.
     - Upload `_pipeline/runs/<UTC>/` as a build artifact.
     - Compare against last month's run.
     - If the failure rate increased OR new questions failed any gate,
       open a GitHub issue with the diff. The body links to the run
       artifact.
2. **Document the workflow** in `vault-cli/docs/AUDIT_PIPELINE.md`:
   how to read the report, how to triage findings, how to ack noise.
3. **Add a `vault audit` CLI subcommand** that wraps the script. Usage:
   ```bash
   vault audit                        # full corpus, audit-only
   vault audit --propose-fixes        # add suggestions
   vault audit --tracks cloud         # subset
   vault audit --since-commit <sha>   # only changed questions
   ```

**Validation:**
- Trigger the workflow manually. Confirm it produces the artifact.
- Run `vault audit --help` and verify the docstring matches
  `AUDIT_PIPELINE.md`.

**Deliverable:** three commits:
- `feat(ci): staffml-audit-corpus-monthly.yml — recurring corpus audit`
- `feat(vault-cli): vault audit subcommand wrapping audit_corpus_batched.py`
- `docs(vault-cli): AUDIT_PIPELINE.md — how to read + triage audit reports`

---

### Phase 9 — Update paper.tech, regen artifacts, tag a release (~1 hour)

**Goal:** capture the cleanup work in the corpus narrative.

**Steps:**

1. **Update `paper.tech`** with:
   - Post-audit corpus stats (counts, per-track breakdowns, audit pass rates).
   - Methodology paragraph referring to `audit_corpus_batched.py` and
     the failure-mode taxonomy (physical_absurdity, vendor_fabrication, ...).
   - Citation of the Gemini-3.1-pro-preview model used.
2. **Run** `vault export-paper` to refresh paper macros.
3. **Run** `vault build --local-json`. Confirm `releaseHash` rolls forward.
4. **`vault publish 1.0.0` (or whatever next version)** — full release pipeline.
5. **`vault verify 1.0.0 --git-ref v1.0.0`** — citation-grade round-trip.
6. **Tag the release.**

**Validation:**
- `vault verify` exit 0.
- paper builds with the new macros.

**Deliverable:** two commits + a tag:
- `docs(paper.tech): post-audit corpus stats + methodology paragraph`
- `release: vault 1.0.0 — corpus-hardening complete`
- Tag: `vault-1.0.0`.

---

## 5. Tooling inventory

| script | what to do | why |
|---|---|---|
| `interviews/vault-cli/scripts/audit_chains_with_gemini.py` | **keep** — extend with `corpus_sample` category if helpful | gold-standard batched pattern; reuse |
| `interviews/vault-cli/scripts/build_chains_with_gemini.py` | **keep** — extract `plan_batches` into shared `_batching.py` | active; reusable helper |
| `interviews/vault-cli/scripts/validate_drafts.py` | **keep** — refactor to import judges from `_judges.py` | active for drafts; prompts are gold |
| `interviews/vault-cli/scripts/audit_math.py` | **keep, deprecate slowly** — useful for spot-checking specific qids | active but redundant with `audit_corpus_batched.py` |
| `interviews/vault-cli/scripts/generate_question_for_gap.py` | **keep** — extract `SCHEMA_SUMMARY` into a shared module | active; reusable schema text block |
| `interviews/vault-cli/scripts/audit_corpus.py` | **DELETE** in Phase 0 | this session's wrong-design dead-end |
| `interviews/vault-cli/scripts/_judges.py` | **NEW (Phase 3)** — extracted shared judge prompts | DRY |
| `interviews/vault-cli/scripts/_batching.py` | **NEW (Phase 3)** — extracted batching helper | DRY |
| `interviews/vault-cli/scripts/audit_corpus_batched.py` | **NEW (Phase 3)** — the right corpus audit tool | the user's actual ask |
| `interviews/vault-cli/scripts/apply_corrections.py` | **NEW (Phase 5)** — interactive accept/reject for Gemini-proposed corrections | safe-by-default human review |
| `interviews/vault-cli/scripts/backfill_provenance.py` | **NEW (Phase 1)** — explicit provenance for 407 questions | one-shot mechanical pass |
| `interviews/vault/scripts/*` (legacy dir) | **already deleted in this session** — phase 0 commit | pre-YAML-migration cruft |

---

## 6. Schema evolution map

| version | change | when | breaks loading? |
|---|---|---|---|
| current (1.0.0) | as-is — `Details.common_mistake/napkin_math` are unconstrained `string`; `provenance` defaults to `imported` | now | no |
| 1.0.1 (Phase 1) | no schema change — just explicit `provenance` in YAML | Phase 1 commit | no |
| 1.1.0 (Phase 6) | `Details.common_mistake` and `napkin_math` get `pattern` constraints; `provenance` becomes `required: true`; `Details.extra` flips `allow → forbid`; new explicit attributes for any legitimate-extra fields | Phase 6 commit | **YES** — but only after Phase 5 has cleaned the corpus, so no real-world question fails the new constraints |
| 1.1.1 onward | small additions only (new audit-stamp fields, etc.) | as needed | no |

The `EVOLUTION.md` doc at `interviews/vault/schema/EVOLUTION.md` already
codifies SemVer rules for schema changes; Phase 6 is a minor bump
(1.0 → 1.1) because it adds enforcement without changing field semantics.

---

## 7. Risk register

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| Gemini misidentifies a math error → human accepts → introduces real bug | medium | high | Phase 5 step 4: math-error fixes REQUIRE per-instance review. Never auto-apply math fixes. |
| Gemini-proposed format reformatting changes meaning | low-medium | medium | Phase 5 diff-review per question. Reject if prose semantics change. |
| Schema tightening (Phase 6) lands before corpus is clean → CI red on dev | low | high | Phase 6 gated on Phase 5 completion. Final pre-merge check: regen `format-audit-full.json` returns 0 failures. |
| Audit script's batched prompt exceeds token budget → some questions silently dropped | medium | medium | Pack budget includes 4K-char wrapper margin (already in `batch_chains`). Validate output JSON has every expected qid. |
| Cron audit (Phase 8) runs against stale model → false positives | medium | low | Pin `gemini-3.1-pro-preview` in the workflow YAML. Bump deliberately when a new model justifies it. |
| Apply-corrections script corrupts a YAML round-trip | low | high | Use `vault edit` (or its underlying schema-validating writer) for every apply. After each apply, run `vault check --strict` on the touched file specifically before moving on. |
| User reviewer fatigue on Phase 5 → rubber-stamping | medium | medium | Batch reviews by failure category, not interleaved. Cap each review session to ~50 questions. Time-box the phase. |
| The 134 placeholder titles get good-but-not-great rewrites | medium | low | Phase 7 includes a hold-back: if reviewer rejects >30%, refine the title prompt before re-running. |
| `extra="forbid"` flip surfaces unexpected legitimate fields → CI red | medium | medium | Phase 6 step 2: survey first, model the survey results, only THEN flip. |
| Audit script regresses on next Gemini-CLI version | low | medium | Pin the gemini CLI version in CI; don't auto-upgrade. |

---

## 8. Rollback strategy

Per-phase escape hatches:

| phase | rollback |
|---|---|
| 0 | `git revert` the deletion commit; the deprecated scripts come back |
| 1 | `git revert` the provenance backfill; Pydantic was already defaulting to `imported`, no functional change |
| 2 | Pure docs + scaffold change; revert leaves the corpus untouched |
| 3 | Pure new-script addition; no corpus change. `git rm` the scripts |
| 4 | Audit-only run; no corpus change. Discard the run dir |
| 5 | Each commit is one logical batch. `git revert` the bad batches; the others stand |
| 6 | Schema tightening: `git revert` the schema commit; codegen regenerates the looser Pydantic. CI goes back to permissive |
| 7 | Title fixes: per-commit revert |
| 8 | Cron workflow: disable via `workflow_dispatch: false` or just delete the YML |
| 9 | Tag rollback only via `vault publish 0.9.x` from prior `releases/`. `vault verify` ensures citability |

---

## 9. Resource budget summary

| phase | hours | Gemini calls | wall clock | notes |
|---|---:|---:|---:|---|
| 0 | 0.5 | 0 | 0.5h | already mostly done |
| 1 | 1 | 0 | 1h | mechanical |
| 2 | 3 | 0 | 3h | doc + code |
| 3 | 3 | 0 | 3h | code + tests |
| 4 | 1 + 4 wall | ~270 | ~5h | run + triage |
| 5 | 6 review + 0.5 commit | ~150 | ~6.5h | the big human-time sink |
| 6 | 2 | 0 | 2h | schema + validator |
| 7 | 1 + 1 review | ~30 | ~2h | title pass |
| 8 | 0.5 | 0 | 0.5h | cron setup |
| 9 | 1 | 0 | 1h | paper + release |
| **Total** | **~19h human + 4h Gemini wall** | **~450** | **~25h** | spread over ~3-5 days |

Gemini cost: **450 calls** total. At 250/day cap, that's 2 days of
quota. At 500/day, that's 1 day.

The original `RELEASE_AUDIT_PLAN.md` estimated 2,900 calls / 12 days.
The new plan is **~6× cheaper and 4× faster** because:
- Batching at 30/call vs. 1/call (10-30× efficiency).
- Full corpus instead of stratified sample (so we don't pay for
  sample-design overhead and we get exact answers, not extrapolated ones).
- One script does audit + corrections (vs. one for each).

---

## 10. Open questions for the user (decide before Phase 3)

These shape the design; cheap to answer now, expensive to change later.

1. **`extra="forbid"` flip aggressiveness.** Should Phase 6 reject any
   YAML field not in the explicit attribute list (the strict view), or
   flip to `forbid` only on `Details` and leave the `Question` model
   permissive? The strict view forces every author to know every field;
   the lenient view lets future authors add audit-stamp fields without
   schema changes.
   - **Recommendation:** strict on `Details`, lenient on `Question`.

2. **Math-error fix policy.** When Gemini identifies a math error in
   napkin_math, should the proposed fix touch ONLY the napkin_math
   block, or should it also rewrite the `realistic_solution` if that
   was depending on the wrong number?
   - **Recommendation:** Gemini proposes both; human reviews both as
     a unit. Don't ship a question with napkin_math saying X and
     solution implying Y.

3. **Level-inflation handling.** When Gemini says a question is L1
   pretending to be L4, three options: relabel down, rewrite up,
   deprecate. Default?
   - **Recommendation:** **relabel down** as default — it's the safest
     and preserves the question. The L4-shaped rewrite is its own
     authoring task and shouldn't block the cleanup phase.

4. **Cron cadence.** Monthly was the proposal. Weekly would catch
   regressions faster but is 4× the Gemini spend. Daily is overkill.
   - **Recommendation:** monthly with a `workflow_dispatch` for
     ad-hoc audits after big content drops.

5. **Per-track audit floor.** Like the chains.json per-track-floor
   guard, should the audit cron fail (red CI) if any track's
   format-pass rate drops below, say, 99%?
   - **Recommendation:** yes, but introduce post-cleanup. Until the
     corpus is at 100%, the floor wouldn't be meaningful.

6. **`audit_math.py` deprecation.** Once `audit_corpus_batched.py`
   ships, do we keep `audit_math.py` as a spot-check tool, or retire
   it?
   - **Recommendation:** keep for one quarter; delete after we
     verify nobody is invoking it.

7. **AUTHORING.md maintenance.** Who keeps the worked examples in sync
   with the schema? When a field name changes, AUTHORING.md drifts.
   - **Recommendation:** add a pre-commit hook that fails if any field
     name in `AUTHORING.md` doesn't appear in the LinkML schema. Cheap
     to write; catches drift at commit time.

8. **Sample size for stratified Gemini audit during the cron.** Full
   corpus = 270 calls. Stratified = 35 calls. Cron should probably
   alternate: stratified weekly, full monthly. Or: full monthly only.
   - **Recommendation:** full monthly. 270 calls / month is well under
     the 250/day-spread-over-30-days budget.

---

## 11. Appendix: example cell-by-cell reference question table for AUTHORING.md

To populate Phase 2 step 1's "Reference questions" section. Pick one
gold-standard question per cell. Existing IDs to start from
(spot-checked from current corpus, may need to verify each is
canonical-format on the day):

| | L1 | L2 | L3 | L4 | L5 | L6+ |
|---|---|---|---|---|---|---|
| cloud | cloud-0001 | cloud-0152 | cloud-0411 | cloud-0598 | cloud-0752 | cloud-1117 |
| edge | edge-0001 | edge-0103 | edge-0220 | edge-0224 | edge-0501 | edge-1001 |
| mobile | mobile-0001 | ... | ... | ... | ... | ... |
| tinyml | tinyml-0001 | ... | ... | ... | ... | ... |
| global | global-0001 | ... | ... | ... | global-0155 | ... |

The exact IDs are placeholders; Phase 2 picks the actual gold-standard
references from the corpus by hand (or by `audit_corpus_batched.py`
flagging the highest-quality ones).

---

## 12. Out of scope (deliberately)

- **Re-authoring `edge-2543`** (content unrecoverable from disk; was
  never committed).
- **Cross-encoder reranking experiment** (Phase 4.5 of CHAIN_ROADMAP — OOM
  on 16GB; out of scope until better hardware).
- **`vault chains suggest` post-write hook** (Phase 4.3; depends on
  Phase 3 stabilizing; defer).
- **Visual rendering audit** (the SVG visuals attached to ~40 questions
  via the `Visual` schema). Pydantic already enforces path-resolves;
  semantic correctness of visuals is a separate workstream.
- **Topic taxonomy revision.** The 87 topics in
  `interviews/vault/taxonomy.yaml` are accepted as-is.
- **`question` field optionality.** Some questions have `question:`
  populated; some don't (the schema marks it optional). Out of scope to
  decide whether to make it required.

---

## 13. One-paragraph summary

Lift the format conventions from "regex-checked on drafts only" to
"schema-enforced across the whole corpus" in three moves: clean the
corpus (Phases 1, 4, 5, 7), tighten the schema (Phase 6), and
institutionalize the audit (Phase 8). Cost: ~19 hours of human work,
~450 Gemini calls (2 days of quota), spread over ~3-5 calendar days.
Final state: every published YAML is provably consistent with a strict
schema, math/coherence/level-fit have been independently verified, and
new violations are caught at `vault check --strict` time so they can
never silently land again. The current `RELEASE_AUDIT_PLAN.md` is
superseded — its "stratified sample at 2,900 calls / 12 days" estimate
was bloated by the wrong design (1 call per gate per question instead
of batched). The corrected design is 6× cheaper and 4× faster.

---

## 14. How to start

1. User reviews this plan (and the **open questions** in §10).
2. On approval, agent commits Phase 0 (cleanup of deprecated dir +
   audit_corpus.py).
3. Phase 1-2 land same session (mechanical + docs; no Gemini cost).
4. Phase 3 builds the tool.
5. Phase 4-5 are the big human-review sink — needs ~6 hours blocked off
   for the user.
6. Phases 6-9 each land in a single afternoon.

Total calendar time: **3-5 days** depending on how Phase 5 review is
paced.
