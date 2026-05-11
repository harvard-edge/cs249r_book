# Legacy scripts — `interviews/vault/scripts/`

Many scripts in this directory pre-date the YAML-as-source-of-truth
migration (ARCHITECTURE.md v2.x, Phase 1). YAML at
`../questions/**/*.yaml` is now authoritative; the legacy scripts ran
against the monolithic `../corpus.json`, which itself is now a
generated artifact (emitted by `vault build --local-json`).

**Do not run remaining scripts in this directory without understanding
what they were for.** New contributors should reach for the `vault`
CLI first; reach into this directory only when adapting one of the
preserved patterns below.

The directory has three classes of script:

1. **Removed** — replaced by `vault` CLI subcommands or by
   `vault-cli/scripts/` equivalents. Findable via
   `git log --diff-filter=D -- interviews/vault/scripts/`.
2. **Preserved for adaptation** — unique-capability scripts kept as
   reference patterns. Each has a `STATUS:` comment block at the top
   explaining the adaptation needed before running.
3. **Active migration / one-shot** — everything else in the directory.
   These are mostly post-v1.0 cleanups that haven't been retired yet.
   Triage individually; some may move to `vault-cli/` or be removed in
   a future cleanup pass.

---

## Removed in 2026-05

The following 12 scripts were deleted as unambiguously dead. The mapping
is preserved here for git-archaeology — find them via
`git log --diff-filter=D -- interviews/vault/scripts/` if you need to
read the historical implementation.

| Removed script | Purpose (pre-migration) | Replacement |
|---|---|---|
| `build_corpus.py` | Assembled `corpus.json` from track/zone data | `vault build` (walks YAML, emits `vault.db`) |
| `export_to_staffml.py` | Copied `corpus.json` → `staffml/src/data/` with field massaging | `vault build --local-json` (writes site-compatible JSON) |
| `extract_taxonomy.py` | Extracted topic graph from `corpus.json` | `vault/taxonomy.yaml` is the source now; see `vault/schema/EVOLUTION.md` |
| `gemini_cli_llm_judge.py` | Legacy LLM-as-judge over `corpus.json` | `vault-cli/scripts/audit_chains_with_gemini.py` (chain audit), `validate_drafts.py` (per-draft gates), and the upcoming `audit_corpus_batched.py` (CORPUS_HARDENING_PLAN.md Phase 3) |
| `gemini_cli_math_review.py` | CLI-flag variant of `gemini_math_review.py` | `vault-cli/scripts/audit_math.py` (active per-question math gate) |
| `gate.py`, `archive/expand_tracks.py`, `archive/fill_zone_gaps.py`, `archive/fill_gaps.sh`, `archive/final_balance.sh`, `archive/README.md` | Pre-launch / pre-v1.0 hierarchy-workaround one-shots | Obsolete after schema v1.0 (taxonomy lives in YAML, not in the path); YAML files authored directly via `vault new` |

---

## Preserved for adaptation

These 6 scripts have unique capability that is **not** yet covered by
the modern tooling. They are kept as reference patterns; each has a
`STATUS:` comment block at the top documenting what to adapt before
running.

| Preserved script | Why kept | When to mine it |
|---|---|---|
| `gemini_backfill_question.py` | Idempotent corpus-walk + Gemini batch + thread-pool + JSON YAML round-trip. The "fix one field across thousands of YAMLs" pattern. | **CORPUS_HARDENING_PLAN.md Phase 5** — reuse the batching + idempotency pattern when applying Gemini-proposed format-marker corrections at scale |
| `gpt_backfill_question.py` | OpenAI/GPT variant of `gemini_backfill_question.py`. Cross-provider template. | When Gemini quota is exhausted, or for A/B comparison of LLM provider quality on the same task |
| `gemini_cli_generate_questions.py` | **BATCHED** generation: 12 cells per call with balanced track × area × zone × level round-robin. `vault generate` does NOT batch — it calls once per question. | When generating > 100 questions in bulk (the 1-call-per-question shape of `vault generate` is fine for tens, wasteful for hundreds) |
| `generate.py` | Coverage-survey-driven generation engine: surveys the corpus, finds empty cells, generates to fill the emptiest first, stops when saturated. `vault generate` does targeted per-cell generation but lacks the auto-balance loop. | When you want "fill all the gaps until the corpus is X-questions-per-cell," not "give me 5 questions about Y" |
| `gemini_fix_errors.py` | Batch error-fixer with hardware-reference grounding (V100 / A100 / H100 / B200 / T4 specs as JSON-encoded ground truth in the prompt). | **CORPUS_HARDENING_PLAN.md Phase 5** — `audit_corpus_batched.py --propose-fixes` should embed the same hardware-reference table when proposing math/coherence corrections |
| `deep_verify.py` | Claude Opus + extended thinking; asks the model to SHOW ITS WORK on every napkin-math claim, step by step. Deeper than `audit_math.py`'s lightweight check. | Tiebreaker on borderline math findings from `audit_corpus_batched.py` — when the lightweight Gemini judge says "fail" but the prose looks reasonable, run deep_verify on the suspect IDs |

### Adaptation checklist (applies to all preserved scripts)

Before running any preserved script:

1. **Replace corpus loading.** Most preserved scripts read
   `interviews/vault/corpus.json`. That file no longer exists in git;
   it's a build artifact. Adapt to walk `interviews/vault/questions/**/*.yaml`
   directly. Use `vault_cli.loader.load_corpus()` or the same `yaml.safe_load`
   loop the active scripts use (e.g. `audit_chains_with_gemini.py:load_corpus`).

2. **Verify the LLM API surface.** The Gemini CLI version, Anthropic SDK
   version, and OpenAI client version may have moved on since the script
   was written. Check `pyproject.toml` for current pins.

3. **Update output paths.** Many preserved scripts wrote to
   `interviews/vault/scripts/_validation_results/<UTC>/`. The current
   convention is `interviews/vault/_pipeline/runs/<UTC>/` (see
   `interviews/vault/README.md` § "Pipeline artifacts").

4. **Re-validate the prompts.** The schema has evolved (zone × bloom
   affinity, closed competency_area enum, format-marker conventions in
   common_mistake / napkin_math). Regenerate the prompt-side schema
   summary against the current LinkML
   (`interviews/vault/schema/question_schema.yaml`).

5. **Run on a sample first.** All these scripts are batch-mode and can
   touch hundreds or thousands of YAMLs in one run. Always run `--limit
   5` (or equivalent) first; verify the diff on a couple of files; then
   widen.

---

## Active migration / one-shot scripts (not in scope of this doc)

The directory still contains several scripts not classified above
(e.g. `analyze_coverage_gaps.py`, `audit_applicability_matrix.py`,
`audit_question_backfill_balance.py`, `audit_visual_questions.py`,
`fix_competency_areas.py`, `iterate_coverage_loop.py`,
`migrate_to_*`, `plan_gap_improvements.py`, `portfolio_balance_loop.py`,
`promote_validated.py`, `reclassify_zone_bloom_mismatch.py`,
`rename_legacy_ids.py`, `repair_chains.py`, `repair_registry.py`,
`render_visuals.py`, `scorecard.py`, `validate_generation_gates.py`,
`validate_questions.py`, `vault_fill.py`, plus the shell wrappers
`review_math.sh`, `run_parallel.sh`, `run_reviews.sh`).

Several of these are referenced from active docs:

- `MASSIVE_BUILD_RUNBOOK.md` cites `analyze_coverage_gaps.py`,
  `iterate_coverage_loop.py`, `promote_validated.py`.
- `vault/visuals/ARCHITECTURE.md` cites `render_visuals.py` as the
  single entry point for figure rendering.

Triage of those scripts is out of scope for the 2026-05 deprecation
pass. They will be classified individually when each owner's workstream
next touches them.

---

## Commands that are live today

```bash
vault build --local-json                        # regenerate corpus.json
vault publish <version>                          # end-to-end release
vault export-paper <version>                     # paper macros + stats
vault verify <version>                           # academic-citability check
vault check --strict                             # 26 invariants
vault generate --topic X --zone Y --track Z --level Lz --count N   # generate new drafts
```

See `../../vault-cli/README.md` for the full 22-subcommand reference.

For Gemini-driven audit + correction at corpus scale, see
`../../vault-cli/docs/CORPUS_HARDENING_PLAN.md` (the active workplan).
