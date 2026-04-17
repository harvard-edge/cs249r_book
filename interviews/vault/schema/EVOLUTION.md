# Vault Schema Evolution

> **Purpose**: Rules for evolving the per-question schema (`schema_version`) and the
> canonicalization function used to compute `content_hash` / `release_hash`.
> **Referenced from**: ARCHITECTURE.md §3.3 (schema_version), §3.5 (content-hash
> canonicalization), §13 ("Version skew"), §14 (Phase 0 deliverables), H-1 in REVIEWS.md.
> **Status**: v1 — Phase 0 deliverable.

---

## 1. What the schema is, and isn't

The **schema** is the LinkML definition at
[`interviews/vault/schema/question_schema.yaml`](question_schema.yaml). Everything
that reads or writes a question — `vault-cli`, the paper exporter, the Worker,
the site, and third-party verifiers — derives its types from this file via
codegen (Pydantic, SQL DDL, TypeScript).

The schema is NOT the YAML file format itself. The YAML is an *encoding*. Two
different YAML files that parse to the same semantic payload are equivalent
under the schema and produce the same `content_hash`.

## 2. Versioning — SemVer semantics

Every question carries a `schema_version` integer. The integer follows a
**single-number** schema (no `MAJOR.MINOR.PATCH` tuple) because the whole
corpus lives in one repo and bumps atomically.

Transitions between schema versions are classified by what the bump does to
readers and writers:

| Bump class | What changes | Example | Migration required? |
|---|---|---|---|
| **Additive-minor** | A new optional field added. | Add `optional tags_v2: list[str]` | No; old readers ignore, new readers see default. |
| **Breaking-major** | Any of: required-field added/removed, existing field renamed or retyped, enum value removed, canonicalization-algorithm change. | Rename `generated_by` → `provenance` | Yes; see §4 below. |

**Rule**: We allow additive-minor bumps WITHOUT bumping `schema_version`. They
are tolerated by every loader by construction (Pydantic ignores unknown optional
fields with explicit config, or defaults are applied). `schema_version` increments
ONLY on breaking-major changes.

In practice this means `schema_version: 1` will stay `1` until the first
breaking change is required. This matches how schema numbers work in OpenAPI
specs and protobuf field-additions.

## 3. Loader contract across versions

For any schema version `N` that the loader knows about:

- **Reads version `N`**: fully supported. All fields populated.
- **Reads version `N-1`**: supported via a "reader adapter" that fills defaults
  for added fields and, if renames happened, maps old names to new names. The
  adapter is a pure function in `vault_cli/migrations/from_v<N-1>.py`.
- **Reads version `N-2` or older**: refused with a clear error — "this question
  is schema v1, loader supports v3 and v2. Run `vault migrate-schema v1 v3` to
  upgrade."
- **Reads version `N+1` or newer**: refused — "this question is schema v4, loader
  supports v3. Update vault-cli: `pip install -U staffml-vault`."

The refusal messages are stable strings, documented in
[`vault-cli/docs/EXIT_CODES.md`](../../vault-cli/docs/EXIT_CODES.md) error-class
list.

## 4. `vault migrate-schema v<from> v<to>` mechanics

The migration command is the ONLY supported way to upgrade schema versions
across the corpus. It is explicitly NOT a background auto-migration; humans
approve migrations.

### 4.1 Preconditions

- Working tree clean (no untracked YAML in `vault/questions/`).
- `vault check --strict` passes on the `from` version.
- The migration script exists at `vault-cli/migrations/v<from>_to_v<to>.py` and
  is committed.
- The migration defines a `forward()` function and a `rollback()` function.
  Both are pure transforms; neither touches filesystem paths.

### 4.2 Execution

`vault migrate-schema v<from> v<to>` does:

1. Resolves the migration chain: if `<from>=1` and `<to>=3`, applies `v1_to_v2`
   then `v2_to_v3` sequentially.
2. Writes to a **parallel tree** at `vault/.migrations/<timestamp>/` — never
   mutates `vault/questions/` in place until validation succeeds.
3. Applies each migration's `forward()` to every question.
4. Runs `vault check --strict --schema-version <to>` against the parallel tree.
5. If all checks pass: `git mv` the parallel tree into place, update a
   `schema_version` bump in every file (sed-safe because YAML has the key at a
   stable position).
6. If any check fails: writes failure log to `vault/.migrations/<timestamp>/FAILED.log`
   and exits 1. The parallel tree is kept for inspection; no source files
   modified.

### 4.3 `--dry-run`

Emits the would-be migration plan (count of files touched per version), without
writing. Safe to run at any time.

### 4.4 Rollback of a schema migration

`vault migrate-schema v<to> v<from> --rollback` runs the `rollback()` direction.
Same parallel-tree mechanics. Rollbacks are expected to be lossy only if the
forward bump removed a field; the migration author is responsible for
documenting any lossy rollback in the migration script's docstring.

## 5. Mixed-version PRs are forbidden

A PR may not contain questions at different `schema_version`s inside
`vault/questions/`. Migration is all-or-nothing per PR. CI rejects mixed-version
trees with:

```
[ERROR] Mixed schema versions detected in vault/questions/:
  schema_version: 1 → 7,143 files
  schema_version: 2 →    56 files
Run `vault migrate-schema v1 v2` in a dedicated PR before introducing v2 questions.
```

This rule prevents the "nobody knows which version to write" failure mode.

## 6. Canonicalization-version bumps

Separate from `schema_version`: the **canonicalization** version
(`CANON_VERSION` in `vault_cli/hashing.py`) versions the algorithm used to
compute `content_hash` / `release_hash` per ARCHITECTURE.md §3.5.

Bumps happen when the hashing rules themselves change — Unicode normalization
form, whitelist, sort-key recursion depth, encoding. These are extremely rare.

- `__canon_version__` is a Merkle leaf in `release_hash` (§3.5), so two releases
  hashed under different canon versions produce different `release_hash` by
  construction.
- Verifying an old release requires running the canon version that released it.
  `vault-cli` ships past canon implementations under
  `vault_cli/hashing/canon_v<N>.py` and `vault verify --canon-version <N>` picks
  the right one.
- Bumping `CANON_VERSION` REQUIRES simultaneously bumping `schema_version` — we
  do not mix "algorithm-changed but schema-same" with "schema-changed but
  algorithm-same" in the same release.

## 7. Historical record of schema versions

As of this document: only `schema_version: 1` exists.

Future bumps will append a subsection here:

```
### v1 → v2 (planned)
- Added: <field>
- Reason: <why>
- Migration: vault-cli/migrations/v1_to_v2.py
- Release: v<X.Y.Z>
```

## 8. What this doc does NOT cover

- **Taxonomy evolution**: adding/removing topics, chain definitions, zones. The
  taxonomy is versioned separately via `taxonomy.yaml` and tracked by the
  `__taxonomy__` Merkle leaf. Additions are backward-compatible; removals
  require a deprecation window (six months default).
- **Release-policy evolution**: changes to `release-policy.yaml` filter
  predicate bump `policy_version` and get recorded in `release_metadata` (the
  `__policy__` Merkle leaf makes policy changes content-bound).
- **D1 schema migrations**: separate from per-question schema; those are in
  `staffml-vault-worker/migrations/` and follow Wrangler's own conventions.

Each of the above has its own rules, covered in ARCHITECTURE.md §3 and §10.

## 9. Owners & review

Schema evolution proposals are RFC-style:

1. Open a PR changing `question_schema.yaml`.
2. Include a migration script under `vault-cli/migrations/`.
3. Include a proposed entry for §7 of this file.
4. At least one maintainer approves (two for breaking-major once external
   contributors onboard in Phase 7+).
5. Merge only after `vault check --strict` green on a representative test
   corpus.

---

**End of EVOLUTION.md.**
