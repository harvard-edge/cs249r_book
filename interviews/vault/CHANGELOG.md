# StaffML Vault Changelog

All notable changes to the vault data model and tooling. This file documents
schema version bumps, breaking migrations, and the reasoning behind each.

The vault follows [Semantic Versioning](https://semver.org/) for its schema:
- MAJOR version (`1.0` → `2.0`): breaking change requires a migration.
- MINOR version (`1.0` → `1.1`): additive changes (new optional fields).
- PATCH: reserved for bug fixes in tooling; schema bytes are unchanged.

---

## [0.1.2-dev] — 2026-04-25

### Summary

Multi-phase release-readiness push across one branch
(`feat/massive-build-2026-04-25-run`). Bundle: 9,224 → ~9,800 published
items; three new structural validators at the data boundary; generator
retrofit; closed the parallelism gap that survived three prior
generation passes.

### Schema changes (additive, no migration)

- **`Visual` class hardened** (commit `542aaf95d`):
  `kind` is now a closed enum (`svg` only; `mermaid` was reserved but
  retired since it was never shipped); `path` regex tightened to
  `^[a-z0-9-]+\.svg$`; `alt` ≥10 chars; `caption` is now required
  ≥5 chars (was optional).
- **`ZONE_BLOOM_AFFINITY` matrix added** to `schema/enums.py`: each
  zone admits a specific Bloom verb set. Mismatch is a hard ERROR
  enforced by `Question._zone_bloom_compatible` (model_validator).
  Documents the contract that `zone` and `bloom_level` must agree on
  cognitive level — no more zone=recall + bloom_level=evaluate.
- **`BLOOM_CANONICAL_ZONE` added**: canonical bloom→zone fallback used
  by `reclassify_zone_bloom_mismatch.py` to deterministically repair
  zone-bloom contradictions. Used once to fix 576 mistagged items.
- **`ZONE_LEVEL_AFFINITY` widened** to all 6 levels per zone: the soft
  pedagogical constraint was retired in favor of the stronger
  `ZONE_BLOOM_AFFINITY` hard constraint. Lint warnings: 1,308 → 0.

### New Pydantic validators (commit `542aaf95d`)

- `Visual._supported_kind`, `_safe_path`, `_alt_min_length`,
  `_caption_min_length`.
- `Question._zone_bloom_compatible` (model_validator).
- `Question._visual_path_resolves` (model_validator) — visual.path must
  resolve to a real SVG file under `interviews/vault/visuals/<track>/`.
  Skipped in production deploys where the working tree is absent.

### Tooling additions

- `scripts/repair_registry.py`: appends disk-YAML IDs missing from the
  append-only `id-registry.yaml`. Catches up the prior rename
  refactor's debt (5,269 + 167 + 87 entries across the push).
- `scripts/repair_chains.py`: drops orphan singletons; renumbers chain
  positions to be unique + Bloom-monotonic. 80 file edits applied.
- `scripts/reclassify_zone_bloom_mismatch.py`: deterministic bloom-
  canonical reclassification for items violating ZONE_BLOOM_AFFINITY.
- `scripts/fix_competency_areas.py`: REMAP table extended with 30+ new
  patterns (zones-as-area, bloom-verbs-as-area, underscore
  hallucinations, dash/slash track-prefix forms). 462 fixes applied.
- `scripts/render_visuals.py`: structured per-ID failure log to
  `_validation_results/render_failures.json`; non-zero exit on any
  per-item crash. Surfaced two prior silent failures (`mobile-1962`
  graphviz `Edge` keyword collision; `tinyml-1570` matplotlib missing
  `numpy as np` import).
- `scripts/gemini_cli_generate_questions.py`: validate-at-write
  contract (every YAML round-trips through `Question.model_validate()`
  before disk write); `--prompt-variant {default,parallelism}` flag
  (parallelism variant: forbid bandwidth division, require concrete
  topology, require quantified sync/bubble cost, require non-obvious
  failure mode); `--targets-from <file>` flag; retry-on-validation-fail
  (single retry per batch with structured error context);
  `bloom_for_zone_level()` helper respects ZONE_BLOOM_AFFINITY;
  `parse_target()` sets competency_area from canonical `TOPIC_TO_AREA`.
- `scripts/analyze_coverage_gaps.py`: `--include-areas <areas>` flag
  injects area-targeted cells into the recommended_plan. Closes the
  topic-priority-misses-area-gaps mismatch.
- `vault build --local-json`: auto-emits `vault-manifest.json`
  alongside corpus.json. Eliminates the recurring stale-manifest
  pre-commit failure.
- `vault doctor`: `registry-integrity` check split into `disk-coverage`
  (HARD FAIL) and `registry-history` (INFO). The legacy
  `_check_schema_version` bug (compared string `"1.0"` against int 1)
  fixed.

### Practice page (commit `542aaf95d`)

- `QuestionVisual.tsx` wraps the inline image in `<Zoom>` from
  `react-medium-image-zoom` (4 KB). Click image → fullscreen modal;
  ESC closes. Playwright test added (9/9 pass; was 8/8).
- TypeScript type drift fixed: `corpus.ts`, `corpus-vault.ts`,
  `staffml-vault-types/index.ts` match the v0.1.2 schema.

### Content additions

- **320 PASS items** from initial massive build (Phase 1-7,
  `ece6eccf2`) — cloud-heavy + edge/mobile/tinyml backfill.
- **144 PASS items** from Phase B + C (`e7cd3b24c`) — 110 from a
  refined loop with validate-at-write + bloom-aware prompts; 34 from
  rehabilitating the prior NEEDS_FIX queue via fix-agent.
- **87 PASS items** from Phase D + F (`6b2b3e054`) — closed the
  parallelism gap that had survived three prior pushes:
  - tinyml/parallelism: 0 → 8
  - mobile/parallelism: 0 → 6
  - edge/parallelism: 13 → 18
  - global/parallelism: 0 → 19

### Lessons

- **Validate at the data boundary, not in audits.** Three new
  validators each prevent a prior failure mode at write time.
  Audit-time validation discovers damage; data-boundary validation
  prevents it.
- **Prompt specificity beats budget.** Parallelism cells went from
  51% pass rate (B.5, standard prompt, 26 API calls) to 80.6%
  (D.3, PARALLELISM_RULES variant, 3 API calls). Same model, same
  judge, same API.
- **Topic-priority ranking misses area-level gaps.** The analyzer's
  recommended_plan ranks track×topic cells by priority; parallelism
  area-gaps don't surface because the priority is split across many
  parallelism-flavored topics. Closing area-level gaps needs
  explicit area targeting — solved by `--include-areas`.

---

## [1.0.0] — 2026-04-21

### Summary

Rebased the vault's data model on the four-axis taxonomy described in the
StaffML paper (`interviews/paper/paper.tex` §3). Classification now lives
in the YAML body, not the filesystem path; the filesystem carries only
`track` for navigability. This reverses the pre-v1.0 "path-as-classification"
design decision (ARCHITECTURE.md §3.3 "H-9 hardening") which could not
represent the paper's full 6-level × 11-zone taxonomy.

### Why

The pre-v1.0 split script exported corpus records into a
`questions/<track>/<level>/<zone>/<id>.yaml` hierarchy. That hierarchy was
incomplete: 7 of 11 ikigai zones had no directory, and `L6+` had no
directory. The result:

- **943 questions labelled `level: L6+` in `corpus.json` were silently
  collapsed into `l1/`** (the migration script had no fallback for the
  unrepresentable level).
- **1,594 questions in zones without directories** (`optimization`,
  `mastery`, `realization`, `analyze`, `diagnosis`, `evaluation`,
  `implement`) were collapsed into `recall/`.
- **86 published questions were dropped entirely** when their target
  `(track, level, zone)` directory did not exist.
- **101 questions belonging to multiple chains had chain data truncated**
  by a singular `chain:` YAML field that could only hold one reference.

These defects were invisible because the schema source-of-truth was split
three ways — `vault/schema.py`, `vault-cli/models.py`, and
`schema/question_schema.yaml` — and the three disagreed about which levels
and zones were valid. The script trusted the most restrictive view and
silently dropped everything else.

### Breaking changes

- `schema_version` changed from integer `1` to string `"1.0"`. Schema-version-aware
  loaders MUST reject the integer form.
- YAML files moved from `questions/<track>/<level>/<zone>/<id>.yaml` to
  `questions/<track>/<id>.yaml`. Any tool that parsed classification from
  the path MUST read it from the YAML body instead.
- Singular `chain: {id, position}` replaced by plural `chains: [{id, position}, ...]`.
- Dropped YAML fields: `scope` (unused by GUI, half-populated free text),
  `mode` (25 questions populated, dead), `version` (7,969 null, dead),
  `deep_dive_title` / `deep_dive_url` (retired in favor of `details.resources[]`).
- SQLite schema gains `competency_area`, `bloom_level`, `phase`, and
  `human_review_*` columns. `chain_questions` primary key changes from
  `(chain_id, position)` to `(chain_id, question_id)` so that multi-chain
  and non-contiguous positions work.

### Additions

- `track`, `level`, `zone`, `topic`, `competency_area` required fields on
  every YAML. `bloom_level`, `phase` optional. Together these form the
  paper's 4-axis classification.
- `human_reviewed: {status, by, date, notes}` — new field tracking human
  verification independently of LLM validation stamps. Every migrated YAML
  carries `status: not-reviewed` until a human reviews it.
- `chains: [{id, position}]` — plural form recovers multi-chain membership.
- `schema/enums.py` — single source of Python enum values; imported by
  both `schema.py` (corpus validator) and `vault-cli/models.py` (YAML
  validator). `schema/question_schema.yaml` (LinkML) remains the
  authoritative schema definition.
- `ZONE_LEVEL_AFFINITY` table (paper §3.3 Table 2) in `enums.py` for use
  by `vault lint` to warn on unlikely zone-level pairings.
- Expanded curated topic list from 79 to 87, including 8 topics that had
  50+ existing corpus questions but were missing from the earlier curated
  set (`autograd-computational-graphs`, `chiplet-architecture`,
  `communication-computation-overlap`, `disaggregated-serving`,
  `model-adaptation-systems`, `recommendation-systems-engineering`,
  `software-portability`, `sustainability-carbon-accounting`).
- `status: deleted` added as a valid status (paired with `deletion_reason`)
  to represent the 458 soft-deleted corpus records.
- `human_reviewed` status `{not-reviewed, verified, flagged, needs-rework}`.
- Migration script preserved at `scripts/migrate_to_v1_0.py` for forensic
  reference.

### Normalisations applied during migration

- `bloom_level: synthesize` → `create` (2001 Bloom's taxonomy revision;
  10 questions affected).
- Codespell fixes across 14 files: unparse*able* → unparsable,
  r*e*-use → reuse, heterogen*o*us → heterogeneous, slig*h* → slight,
  pr*e*-empt → preempt, pr*e*-empt*able* → pr*e*-empt*ible*.
  (Words split with asterisks to avoid codespell re-flagging this entry.)
- 3 `mobile-*.yaml` questions carried stale `correct_index: -1` sentinel
  without an `options` list; stripped.

### Retired tooling

These one-off scripts originally lived under `vault/scripts/archive/` and
`vault-cli/scripts/archive/` after migration. **Those directories are no longer
in the tree** (retrieve earlier revisions from git if you need the files for
comparison):

- `vault-cli/scripts/split_corpus.py` — the pre-v1.0 exporter, source of
  the defects this migration corrects.
- `vault/scripts/fill_zone_gaps.py` — workaround for hierarchy-mandated
  empty cells; obsolete now that zones are data.
- `vault/scripts/expand_tracks.py`, `final_balance.sh`, `fill_gaps.sh` —
  paperover scripts for the same structural gap.

### Validation

- All 9,657 corpus records validate through the v1.0 `Question` model
  with zero errors.
- All 9,657 migrated YAMLs load through `vault-cli`'s new loader with
  zero errors.
- Filesystem tree has 5 top-level track directories
  (cloud: 4,228 · edge: 2,089 · mobile: 1,742 · tinyml: 1,292 · global: 306).

### Follow-up work (planned)

- LLM-assisted content-quality audit on a per-YAML basis (now easy
  because every YAML is self-contained). Target: verify
  content-level/zone fit for the remaining non-mechanical mis-labels.
- CI drift check between `schema/enums.py` and `schema/question_schema.yaml`
  so the single-source-of-truth claim is enforced mechanically.
- Invert the corpus source-of-truth: move `corpus.json`, `chains.json`,
  `taxonomy.json` out of git and rebuild them as build artifacts from
  YAMLs via `vault build`.
- `vault lint <file>`: author-facing linter that emits zone-level affinity
  warnings (paper line 397: "An L1 question tagged as 'evaluation' is
  flagged for review").
