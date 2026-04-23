# StaffML Vault Changelog

All notable changes to the vault data model and tooling. This file documents
schema version bumps, breaking migrations, and the reasoning behind each.

The vault follows [Semantic Versioning](https://semver.org/) for its schema:
- MAJOR version (`1.0` → `2.0`): breaking change requires a migration.
- MINOR version (`1.0` → `1.1`): additive changes (new optional fields).
- PATCH: reserved for bug fixes in tooling; schema bytes are unchanged.

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

Archived to `scripts/_archive/` (preserved for forensic reference):

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
