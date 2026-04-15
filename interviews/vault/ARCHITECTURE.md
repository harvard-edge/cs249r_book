# StaffML Vault Architecture — Design Document

> **Status**: **v2.1** — Round-2 adversarial review integrated 2026-04-15. Pre-Phase-0.
> **Scope**: Everything from question authoring to production serving.
> **Audience**: Future-VJ, future maintainers, future collaborators.
> **Supersedes**: `SYSTEM.md` (stale — says 5,786 questions, 839 concepts).
> **Review ledger**: [`REVIEWS.md`](./REVIEWS.md) — what was raised, what was fixed, what was deferred, why. v2 resolves all 8 Critical and 21 High items from Round 1; two Lows deferred with rationale.
> **Changelog v1 → v2** (keyed to `REVIEWS.md` IDs):
> - **C-1/C-4/C-7/C-8**: Release is now atomic via `vault ship`; rollback is snapshot-restore, not inverse SQL. Schema-changing releases are gated. Site rollback is a feature flag, not a file delete.
> - **C-2**: §11 rewritten — YAML is the sole authoring surface from Day 1 of Phase 1; `corpus.json` is generated-only, pre-commit-hook-protected.
> - **C-3**: Content hash is SHA-256 over canonical JSON of whitelisted semantic fields (not SQLite bytes). Release hash is a Merkle root over `(id, content_hash)` pairs.
> - **C-5**: IDs are content-addressed (topic + short-hash-of-title + dedup suffix); `id-registry.yaml` is an append-only log.
> - **C-6**: `vault generate` draws exemplars only from `vault/exemplars/` (curated human-reviewed pool), never from LLM-generated questions.
> - **H-1**: `schema/EVOLUTION.md` added as a Phase-0 deliverable.
> - **H-2**: LinkML is sole schema SSoT; `vault publish` codegens Pydantic, SQL DDL, and TypeScript types as release artifacts.
> - **H-17**: Phase 0 adds `vault api` (local Worker-surface shim) + `CONTRIBUTING.md` so contributors can clone-and-render without a Cloudflare account.
> - **H-18**: Timeline revised to 18–22 focused days (phases 0–6). Chain-discoverability cut to a single Phase-5 intervention with instrumentation.
>
> **Changelog v2 → v2.1** (Round-2 convergent issues):
> - **Ship protocol (Chip N-C1 / Dean N-1 / Soumith H-NEW-1)**: §6.1.1 adds ordered commit protocol (D1 → Next.js → paper-last), journal file, per-leg rollback matrix, paging triggers, `--resume` primitive.
> - **X-Vault-Release soft signal (Soumith H-NEW-2 / Dean N-3)**: demoted from hard-reject to informational SLI; correctness boundary is release-keyed Cache API. §6.1.1 adds 10-min cross-release grace window to eliminate post-deploy brownout.
> - **Merkle policy + canon-version binding (Chip N-H5)**: release_hash now includes `__policy__` and `__canon_version__` leaves. §3.5.
> - **Schema-fingerprint vs actual DDL (Dean N-4)**: worker reads `sqlite_master`, hashes, compares at cold start. On mismatch, serves from Cache API in degraded read-only mode (not 5xx) — fixes Chip N-H1 total-outage risk. §10.1.
> - **SW release-keying (Chip N-H2 / Dean N-6 / David N-4)**: service-worker cache includes release_id; evicts on release change; TTL 7d. §7.1.
> - **CI equivalence via Merkle hash (David N-1 / Chip N-H4)**: compare release_hash to `corpus-equivalence-hash.txt`, not 28 MB byte-diff. §11.5.
> - **ID collision recovery (Dean N-2 / David N-2)**: `vault renumber` recovers from post-rebase dedup-seq collisions. §3.3.
> - **FTS5 cost gate (Dean N-5)**: Phase-4 entry gated on ≤500 D1 row-reads per FTS5 query in addition to latency. §10.6.
> - **Exemplar coverage audit (Chip N-H3)**: Phase 0 produces `vault/exemplar-gaps.yaml` inventory. §14.
> - **Codegen contract (Soumith H-NEW-3)**: PR authors run codegen locally; CI verifies via `vault codegen --check`, never pushes follow-up. §13.
> - **Extended static fallback retention (Dean N-10)**: kept until first post-cutover schema-major bump OR 2 releases, whichever is later. §7.1.
> - **Canary DAU-adjusted (David N-5)**: soak = max(15 min, ≥100 sessions observed). §4.3.
> - **`vault renumber` + `vault mark-exemplar` added to CLI surface (David N-9)**.
> - **Removed duplicate Observability subsection in §13**. Cleanup only.

---

## 1. Thesis

The StaffML question corpus is **curated structured content**, not a static JSON blob to be shipped. It deserves the architecture of a real content system:

1. A **human-first source of truth** — per-question YAML files. Debuggable with `less`, diffable with `git`, reviewable in a PR.
2. A **built artifact** — SQLite database compiled from the YAML. Queryable, indexed, typed.
3. An **edge-distributed serving layer** — Cloudflare D1 + Worker. No more 20 MB bundle.
4. A **single release gate** — one command produces site data, paper macros, and D1 migration atomically from the same snapshot.

This architecture is correct in itself, but the stronger argument is: it **unlocks things the current system cannot do** — real full-text search, chain discoverability, LLM-assisted generation grounded in the existing corpus, per-user analytics, meaningful academic citation of versioned releases.

---

## 2. Current State (as of 2026-04-15)

### Pipeline that exists
```
vault/schema/staffml_taxonomy.yaml         ← LinkML schema (SSoT for types)
vault/schema/taxonomy_data.yaml            ← 79 topics + 123 edges
vault/schema/zones.py                       ← 11 ikigai zones
vault/corpus.json                           ← 28 MB. All questions. THE monolith.
vault/chains.json
         │
         ├─→ vault/scripts/export_to_staffml.py → staffml/src/data/*.json
         └─→ paper/scripts/analyze_corpus.py    → paper/corpus_stats.json
                                                → paper/scripts/generate_macros.py
                                                → paper/macros.tex

staffml/ (Next.js)
  src/lib/corpus.ts:  import corpusData from '../data/corpus.json';  // 19 MB inlined
  → every page bundle: 17–20 MB of JS
  → every lookup: O(n) array.find() across 8,053 questions
```

### Three bugs we found in this pipeline

1. **Different filter predicates between consumers.** Site keeps `q.validated`, paper keeps `q.status == "published"`. These sets differ by 1,146 questions. Result: paper says 9,199, site says 8,053.
2. **Orphan topics inflate topic count.** `analyze_corpus.py` reports `summary.topics = 87` but `taxonomy_graph.total_topics = 79`. The 8 extras are corpus values that are not in the curated taxonomy — the schema doesn't enforce the relationship.
3. **No release gate.** Three separate scripts (`sync-vault.py`, `generate-manifest.py`, `generate_macros.py`) run independently. Nothing requires they operate on the same snapshot.

---

## 3. Target Architecture

### 3.1 Repository layout

```
interviews/
├── vault/                              ← DATA
│   ├── schema/
│   │   ├── staffml_taxonomy.yaml       ← LinkML (existing)
│   │   ├── question_schema.yaml        ← Per-question schema (NEW)
│   │   └── zones.yaml                  ← 11 ikigai zones (ported from zones.py)
│   ├── taxonomy.yaml                   ← 79 topics + 123 edges
│   ├── chains.yaml                     ← Chain definitions (cross-cutting)
│   ├── release-policy.yaml             ← "What counts as published" (NEW)
│   ├── id-registry.yaml                ← Ever-assigned ID ledger (NEW)
│   ├── questions/                      ← THE SOURCE OF TRUTH (NEW)
│   │   ├── cloud/{L1..L6}/<zone>/*.yaml
│   │   ├── edge/
│   │   ├── mobile/
│   │   ├── tinyml/
│   │   └── global/
│   ├── drafts/                         ← Unreviewed / LLM-generated (NEW)
│   │   └── <same shape as questions/>
│   └── releases/                       ← Built artifacts (NEW)
│       ├── v1.0.0/
│       │   ├── vault.db
│       │   ├── release.json
│       │   ├── d1-migration.sql
│       │   └── d1-rollback.sql
│       └── latest -> v1.0.0/           ← symlink
│
├── vault-cli/                          ← CODE
│   ├── pyproject.toml                  ← real Python package
│   ├── src/vault_cli/
│   │   ├── main.py                     ← Typer entry point (uses Rich)
│   │   ├── commands/                   ← one module per command
│   │   ├── loader.py                   ← filesystem → in-memory model
│   │   ├── validator.py                ← schema + invariants
│   │   ├── compiler.py                 ← in-memory → SQLite
│   │   ├── exporters/                  ← d1.py, paper.py
│   │   └── utils/
│   ├── templates/question.yaml.j2
│   └── tests/                          ← real test suite
│
├── staffml/ (Next.js)
│   ├── src/lib/vault-api.ts            ← NEW: typed worker client
│   ├── src/lib/corpus.ts               ← rewritten: thin wrapper
│   └── src/data/corpus.json            ← DELETED post-cutover
│
├── staffml-vault-worker/               ← NEW Cloudflare Worker
│   ├── src/index.ts
│   ├── src/queries.ts
│   ├── wrangler.toml
│   └── migrations/
│
└── paper/
    ├── scripts/generate_macros.py      ← rewritten: SQL against vault.db
    └── macros.tex                      ← built artifact
```

### 3.2 Tech choices and naming

| Decision | Choice | Rationale |
|---|---|---|
| Command name | `vault` | Direct, short, matches subject. Invoked from installed `staffml-vault` package. |
| CLI framework | **Typer** (by Sebastián Ramírez) | Modern Python, type-hint driven, auto-help, built on Click. |
| CLI output styling | **Rich** (by Will McGugan) | Color, tables, progress bars, beautiful tracebacks. Industry standard. Typer uses Rich natively. |
| Config files | YAML | Human-readable, comment-friendly, widely known. |
| Build target | **SQLite** (`vault.db`) | Single file, zero-ops, embedded, FTS5 built-in. |
| Production DB | **Cloudflare D1** | SQLite-compatible, edge-distributed, 10 GB free, integrates with existing Worker infra. |
| Validation | Pydantic models generated from LinkML schema | Runtime enforcement, error messages, type-safe. |
| Query layer (browsing) | **Datasette** | Zero-config web UI over SQLite. Free, battle-tested, great for ad hoc exploration. |
| Shared types | LinkML → codegen | LinkML is the sole schema source. `vault publish` emits Pydantic models (for `vault-cli`), SQL DDL (for D1), and a versioned `@staffml/vault-types` TypeScript package (for the site + worker). CI fails if committed generated artifacts don't match current LinkML output. This is the fix for H-2 (quadruple-drift risk). |

### 3.3 Per-question YAML format

**Filename**: `<topic-kebab>-<short-hash>-<4-digit-seq>.yaml`
**Location**: `vault/questions/<track>/<level>/<zone>/<filename>`
**Classification**: encoded in the path — NOT in the file. Moving the file reclassifies the question.

```yaml
# vault/questions/cloud/l4/diagnosis/kv-cache-bandwidth-7f3a9c-0001.yaml
schema_version: 1
id: cloud-l4-diagnosis-kv-cache-bandwidth-7f3a9c-0001  # immutable, URL-safe, content-addressed
title: "KV Cache Memory Bandwidth Bottleneck"
topic: kv-cache-management              # must exist in taxonomy.yaml
chain:                                  # structured form — see H-4
  id: kv-cache-depth
  position: 2
status: published                       # draft | published | deprecated
created_at: 2026-04-15T10:00:00Z
last_modified: 2026-04-15T10:00:00Z
provenance: human                       # closed enum — see H-5
generation_meta:                        # required iff provenance != human
  model: claude-opus-4-6                # must exist in vault/schema/models.yaml
  prompt_hash: sha256:abc123...         # references vault/generation-log/<date>/<hash>.txt
  prompt_cost_usd: 0.0234
  human_reviewed_at: 2026-04-15T11:00:00Z  # required before exemplar use
authors:                                # optional; populated from git config by `vault new`
  - vjreddi

scenario: |                             # plaintext only — validator rejects raw HTML
  <prompt text the student sees>

details:
  common_mistake: |                     # restricted-Markdown allowed (emphasis, inline code, lists)
    <common wrong approach>
  realistic_solution: |                 # restricted-Markdown + KaTeX ($...$ math)
    <canonical answer>
  napkin_math: |                        # restricted-Markdown + KaTeX
    <step-by-step arithmetic if applicable>
  deep_dive:
    title: "PagedAttention and KV Cache Management"
    url: "https://mlsysbook.ai/book/chapters/serving"  # scheme must be https:

tags:
  - hardware:a100
  - size:70b
```

**Design rules (v2)**:

*ID (immutable, content-addressed — fixes C-5):*
- Format: `<track>-<level>-<zone>-<topic-kebab>-<6-hex-of-sha256(title)>-<4-digit-dedup-seq>`
- The `6-hex` prefix makes collisions content-visible. The `4-digit-seq` disambiguates only when two titles collide on the 6-hex (extremely rare; `vault new` bumps the seq automatically).
- `id-registry.yaml` is an **append-only log**, one `{id, created_at, created_by_git_user}` entry per line. Merge conflicts become semantic (two commits claim the same ID → CI rejects merge). The registry is never rewritten.
- `vault new` runs `git pull --rebase` on the registry before allocation.
- **Collision recovery workflow** (v2.1 — Dean N-2, David N-2): if `vault new` collides on the dedup seq after rebase (e.g., another PR merged first and took `-0001`), the operator runs `vault renumber <old-id>` which:
  1. Bumps the seq suffix to the next free slot.
  2. Renames the YAML file to match (`git mv`, preserves history).
  3. Updates the `id:` field in the YAML.
  4. Appends a new registry entry (old ID is still reserved — no reuse).
  5. Updates any `chain:` references in other questions that point to the old ID.
  6. Emits a summary of the exact git operations performed.
- `vault check --strict` fails if the registry lists an ID whose file doesn't exist, OR whose file's `id:` field doesn't match the registry entry. This is the structural check that catches bad manual edits.
- **Birthday-collision math** (v2.1 — Chip bound-correction): within a single `(track, level, zone, topic)` bucket of ~100 titles, the probability of any 6-hex collision is ~100² / 2 / 16^6 ≈ 1/3,350. At 9,200 total questions distributed across ~400 cells, the per-cell rate is low, and per-bucket per-title collision is rarer still (1/16M for pairwise identical), but within-bucket collisions do happen at corpus scale — hence the dedup suffix. The 4-digit suffix gives 10,000 slots per `(topic, 6-hex)` bucket, far above anything we'll see.

*Path-as-classification (hardened — fixes H-9):*
- Filesystem path is authoritative for track/level/zone. YAML has no `track:` etc. fields.
- **All path components must be lowercase**. Fast-tier invariant rejects non-lowercase.
- Each component is enum-validated against `taxonomy.yaml`, the level enum, and `zones.yaml`.
- `vault move` is the **only** supported reclassification path. CI rejects direct `git mv` by checking that every file's ID-registry entry matches its current path prefix.
- Under `vault move`, the filename (`<topic-kebab>-<hash>-<seq>.yaml`) is preserved so git follows the rename.

*Chain reference (structured — fixes H-4):*
- `chain` is a mapping `{id: ..., position: ...}`, never a compact `<id>@<pos>` string.
- Loader accepts the legacy compact form during migration, but always writes structured form on save.
- Chain positions form a contiguous `[1..N]` (already invariant-checked).

*Provenance (closed enum — fixes H-5):*
- `provenance` is one of `{human, llm-draft, llm-then-human-edited, imported}`.
- `generation_meta.model` references `vault/schema/models.yaml` (a registry — new models added by explicit PR).
- `generation_meta.prompt_hash` references a git-tracked file in `vault/generation-log/<yyyy-mm-dd>/<hash>.txt` containing the full prompt (L-2).
- `generation_meta.human_reviewed_at` must be set before the question may be used as an exemplar by `vault generate` (fixes C-6).

*Content format per field (fixes H-6):*
- `title`: plaintext (≤120 chars).
- `scenario`: plaintext only. Validator rejects raw HTML tags, `<script>`, `javascript:`/`data:` URLs.
- `details.common_mistake` / `details.realistic_solution` / `details.napkin_math`: restricted Markdown (emphasis, inline code, fenced code, lists, links) plus KaTeX `$...$` and `$$...$$` math. Validator runs a CommonMark parser + allowlist pass.
- `details.deep_dive.url`: scheme must be `https:`. Rejected: `http:`, `javascript:`, `data:`, relative URLs.
- Fields are sanitized at render time on the site (DOMPurify) in addition to validator-time rejection.

*Authors (new — fixes M-15):*
- Optional `authors` list of git-config handles. `vault new` populates automatically from `git config user.email` → mapped via `vault/contributors.yaml`.
- Under `vault edit`, the current user is appended if not already present.
- Surfaces on the About page contributor list and in paper acknowledgements.

*Version (fixes H-1):*
- Every question carries `schema_version`. Schema evolution rules live in [`schema/EVOLUTION.md`](./schema/EVOLUTION.md) (SemVer — minor bumps are additive-only; major bumps require migration). CI rejects PRs that mix schema versions within `vault/questions/`.

### 3.4 SQLite schema (built by `vault build`)

See the compiler output in `vault-cli/src/vault_cli/compiler.py`. Key tables:

```
questions(id PK, title, topic FK, track, level, zone, status,
          scenario, common_mistake, realistic_solution, napkin_math,
          deep_dive_title, deep_dive_url, provenance,
          created_at, last_modified, file_path, content_hash, authors_json)

chains(id PK, name, topic FK)
chain_questions(chain_id, question_id FK, position, PK(chain_id, position))
tags(question_id FK, tag)
taxonomy(id PK, name, area, description)
taxonomy_edges(source FK, target FK, edge_type)
zones(id PK, name, skills, description)
release_metadata(key PK, value)   -- keys: release_id, release_hash, schema_fingerprint,
                                  --       schema_version, policy_version, created_at, git_sha
```

Indexed: `topic`, `(track, level)`, `zone`, `status`. FTS5 virtual table on title, scenario, realistic_solution.

**Key `release_metadata` rows (fixes C-3, C-8, H-14):**
- `release_id`: the semver string (e.g., `1.0.0`). Used as cache-key component in worker layer.
- `release_hash`: SHA-256 Merkle root — see §3.5.
- `schema_fingerprint`: SHA-256 of the current LinkML schema + DDL. Worker refuses to serve if fingerprint != expected (detects D1-vs-Worker version skew).
- `policy_version`: records which `release-policy.yaml` version filtered this build (fixes H-21).

### 3.5 Content hashing (fixes C-3)

**Why it matters**: academic citation, drift detection, cache keys, rollback integrity. The v1 plan said "immutable content hash of vault.db" which cannot work — SQLite is not byte-reproducible across versions. v2 hashes *inputs*, never the SQLite binary.

**Per-question `content_hash`** (column in `questions` table, also stamped on the YAML at build time for convenience):

```python
def canonical_question_bytes(q: Question) -> bytes:
    whitelist = {"id", "title", "topic", "chain", "status",
                 "scenario", "details", "tags", "provenance",
                 "generation_meta.model", "generation_meta.prompt_hash"}
    # Excluded: last_modified, created_at, file_path, authors, generation_meta.prompt_cost_usd
    payload = select_and_sort(q.dict(), whitelist)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

content_hash = hashlib.sha256(canonical_question_bytes(q)).hexdigest()
```

- Strings are normalized to Unicode NFC, LF line endings, trailing whitespace stripped.
- Excluded fields are metadata whose change doesn't affect semantics.

**Release hash** — a Merkle root making the entire release cryptographically verifiable:

```python
leaves = sorted((q.id, q.content_hash) for q in published_questions)
leaves += [("__taxonomy__",    sha256_of_canonical_taxonomy)]
leaves += [("__chains__",      sha256_of_canonical_chains)]
leaves += [("__zones__",       sha256_of_canonical_zones)]
leaves += [("__policy__",      sha256_of_canonical_release_policy)]  # Chip N-H5, Dean
leaves += [("__canon_version__", sha256(f"canon-v{CANON_VERSION}"))]  # Chip N-H5
release_hash = sha256(b"\n".join(f"{id}:{h}".encode() for id, h in leaves))
```

- Recorded in `release_metadata.release_hash` and `releases/<v>/release.json`.
- `vault verify <version>` reconstructs the leaves from the YAML source, recomputes, and asserts equality with the committed `release.json`. This is what Zenodo / external readers run.
- **Canonicalization-version pinning**: `CANON_VERSION` is an integer literal in `vault_cli/hashing.py`. Bumped only when the canonicalization algorithm changes (e.g., Unicode normalization form, key-sort recursion depth, field whitelist). Pinning this into the Merkle means two releases with identical source content but different canonicalizer versions produce *different* `release_hash` — correctly signaling that verification semantics changed.
- **Policy-binding**: `__policy__` is SHA-256 of canonical `release-policy.yaml`. Two releases with identical YAML source but different filter predicates produce distinct `release_hash` — so a reader of the hash can distinguish content changes from policy changes. Makes `policy_version` in `release_metadata` cryptographically bound, not just advisory.
- **Nested-dict canonicalization** (Soumith M-NEW-4): `json.dumps(sort_keys=True)` sorts keys recursively by default. Test fixture in `vault-cli/tests/test_hash_stability.py` asserts: same question, keys in different YAML-insertion orders → identical `content_hash`.
- Schema version bumps co-version the canonicalization function (documented in `schema/EVOLUTION.md`).

**What `--sign` does** (fixes M-11): `vault publish --sign` uses `minisign` with a public key committed to `vault/signing-key.pub` to produce `releases/<v>/release.json.minisig`. This is a true cryptographic signature; the `--hash` flag is the name if only a digest is wanted.

---

## 4. CLI Specification (v2)

Framework: **Typer** (declarative, type-hint-driven). Output: **Rich** (tables, progress, panels).

**Design principle (fixes H-3)**: expose primitives; compose products from them. `publish` is a composed product; `build`, `snapshot`, `migrations emit`, `export paper`, `tag` are primitives users can run independently. Users get the common case with one verb and the escape hatches for the 20% that deviates.

### 4.1 Authoring primitives

```
vault new [--topic X] [--track Y] [--level Lz] [--zone Z] [--count N] [--batch]
    Prompts for missing fields. Allocates content-addressed ID (see §3.3).
    Creates YAML from Jinja template at correct path.
    --count N (default 1): creates N drafts. With --batch, opens all N in $EDITOR
        at once (vim splits / VS Code tabs). Without --batch, opens them sequentially.
    On save: validates schema. On validation failure, error is injected as a comment
        block AT THE TOP of the file + printed to stderr; file is kept in draft state;
        `vault edit <id>` re-opens with the same error block for iteration. (Fixes H-16.)

vault edit <id> [--dry-run]
    Opens the file in $EDITOR. Re-validates on save. --dry-run shows the generated
    file path and exits without opening.

vault rm <id> [--hard]
    Default: marks status: deprecated (preserves chain slot, hides from UI).
    --hard: deletes file. Always requires typed confirmation of the question title
        — no --yes bypass. Refuses if question is in any active chain unless --force
        AND --i-understand-chain-breakage. (Fixes H-11.)

vault restore <id>
    Reverts a deprecated question to status: published. Soft-delete reversal.
    `vault rm --list-deprecated` enumerates candidates.

vault move <id> --to <track>/<level>/<zone> [--edit] [--dry-run]
    Reclassifies via `git mv` (preserves rename history). Filename (topic+hash+seq)
    is preserved so git follows the rename. (Fixes M-17.)
    Refuses on dirty working tree unless --allow-dirty.
    Refuses if target cell is excluded by the applicability matrix — catches at
    move-time, not build-time.
    Refuses if question is chained unless whole chain moves together or is unlinked first.
    --edit: opens in $EDITOR after moving, for combined reclassify+edit.
    --dry-run: prints the exact `git mv` command without executing.
```

### 4.2 Build primitives

```
vault build [--output <path>] [--legacy-json]
    Walks questions/**/*.yaml. Validates each. Compiles to vault.db.
    Target: < 10s for 10K files (parallelized across cores). Reports all errors,
        does not fast-fail on first error.
    --legacy-json: additionally emits corpus.json for generators that haven't been
        migrated. Used during Phase 1–3 only; removed at cutover. (Fixes C-2.)

vault check [--strict] [--tier fast|structural|slow]
    Runs invariants. Exit 0 on pass, non-zero on any failure. See §4.6 for exit-code
        taxonomy.
    --strict: runs fast + structural tiers (CI default).
    --tier fast: pre-commit hook tier only, <1s budget.
    --tier slow: nightly-CI-only tier (link checks, LLM math verification).
    --json emits structured errors in LSP-diagnostic format so editors can render
        inline squiggles. (See §4.5.)

vault snapshot <version>
    Primitive. Copies vault.db to releases/<v>/. Writes release.json (release_id,
        release_hash, counts, git SHA, timestamp, schema_version, policy_version).
    Does NOT git-commit, tag, or update any symlink.

vault migrations emit <from-version> <to-version>
    Primitive. Diffs two vault.db files at the row level and emits:
      releases/<to>/d1-migration.sql       (forward)
      releases/<to>/d1-rollback.sql         (inverse — embeds full prior-row bodies,
                                             not mechanical SQL inversion; see C-1)
    If the schema differs between versions, refuses unless --schema-change is passed;
        schema-changing releases require hand-authored up/down SQL at
        releases/<to>/schema-forward.sql + schema-rollback.sql. (Fixes C-1.)

vault export paper <version>
    Primitive. Runs SQL over releases/<v>/vault.db to emit paper/corpus_stats.json
        and paper/macros.tex. Uses policy_version from release_metadata so paper and
        site agree by construction. (Fixes H-21.)

vault tag <version>
    Primitive. git-commit + git-tag the artifacts in releases/<v>/. On release branch.

vault verify <version>
    Reconstructs the release hash from YAML source in releases/<v>/ (if shipped with
        release) or from current vault/questions/ (if verifying HEAD). Asserts equality
        with releases/<v>/release.json. Exit 0 on match, **1 on mismatch** (integrity
        failure per §4.6 taxonomy). (Fixes C-3 + Soumith M-NEW-3.)
```

### 4.3 Release product (composed)

```
vault publish <version> [--sign] [--schema-change] [--resume]
    Composed product. Equivalent to:
      vault check --strict
      vault build
      vault snapshot <version>
      vault migrations emit <previous> <version> [--schema-change]
      vault export paper <version>
      vault tag <version>
      update releases/latest symlink via POSIX rename(2) (atomic; last step)
    Stages output to releases/.pending-<v>/ first; atomic rename on success; deletes
        pending dir on failure. --resume detects orphaned pending dirs. (Fixes C-7.)
    --sign: produces releases/<v>/release.json.minisig via minisign against
        vault/signing-key.pub. (Fixes M-11.)

vault deploy <version> --env (staging|production)
    1. Pre-deploy R2 snapshot of target D1 (synchronous, not nightly). (Fixes C-8, M-5.)
    2. Apply schema-forward.sql if present (single transaction; chunked if > D1 tx limit).
    3. Apply d1-migration.sql (wrapped in BEGIN;...COMMIT;).
    4. Write release_metadata row: release_id, release_hash, schema_fingerprint.
    5. POP-propagation probe on N edge locations; fails if stale after 30s.
    6. Requires typed confirmation of the version string (no --yes bypass to prod).

vault rollback <version> --env (staging|production) [--method (snapshot|sql)]
    --method snapshot (default): restores the R2 snapshot taken before the previous
        deploy. Fast, always works. (Fixes C-1 as primary rollback.)
    --method sql: applies d1-rollback.sql. Useful for debugging, not primary path.

vault ship <version> --env production [--dry-run] [--canary-percent N] [--resume]
    Top-level atomic-release verb. Coordinates D1 deploy + Next.js deploy + paper
        git push with pinned NEXT_PUBLIC_VAULT_RELEASE=<version>. Writes a journal
        at releases/<version>/.ship-journal.json recording each leg's state; --resume
        detects interrupted ships and continues from the last successful leg.
        (Fixes C-4 and Round-2 convergence: Chip N-C1, Dean N-1, Soumith H-NEW-1.)
    --canary-percent: partial-rollout Cloudflare Worker version; soaks until ≥100
        sessions observed per step OR 15 min, whichever is later. (Tunes to actual
        traffic at small DAU — fixes David N-5.)
    Keeps primitives `deploy` + Next.js deploy exposed for power users who need
        to deviate.
    See §6.1.1 for the commit/rollback protocol (ordering, journal schema, partial-
        failure semantics, paging triggers).
```

### 4.4 Exploration + generation

```
vault stats [--topic X] [--track Y] [--level Z] [--zone W]
           [--format table|json|csv|prometheus]
    Scorecard over latest vault.db. --format prometheus emits scrape-ready metrics
        including authoring-health SLIs (LLM-vs-human ratio, corpus staleness,
        validation-failure rate on main). (Fixes M-9.)

vault diff <from-version> <to-version> [--classify]
    Added, removed, modified questions between two releases. --classify labels each
        modification as cosmetic (whitespace/typo), semantic (scenario text change),
        or structural (topic/chain reassignment). Semantic changes emit a warning;
        structural changes are flagged as potentially breaking for student bookmarks.
        (Fixes M-3.)

vault generate --topic X --zone Y --level Lz --track T --count N [--model M]
              [--no-context] [--yes] [--dry-run]
    LLM-assisted generation — see §12 for exemplar-pool discipline (C-6) and cost
        controls (H-8). Writes to vault/drafts/.
    Hard cap: N ≤ 25 per invocation. Higher requires --i-know-what-im-doing + env var.
    Context grounding defaults ON; --no-context opts out. (Fixes L-5.)
    Dry-run emits token/cost estimate and the prompt; no API calls.
    Daily spend ledger at vault/.llm-spend.json; refuses further calls if ceiling hit.
    Secrets from ~/.config/vault/secrets.toml (mode 0600), never env vars.

vault promote <id> [--reviewed-by <git-user>]  |  vault promote --all-drafts [--topic X]
    Moves drafts to vault/questions/. Sets status: published.
    If provenance is llm-draft, updates to llm-then-human-edited.
    --reviewed-by defaults to git config user.email; CI rejects if value doesn't
        match the committer of the promoting commit (closes Soumith L-NEW-1).

vault mark-exemplar <id> (v2.1 — David N-9)
    Moves a published question from vault/questions/ to vault/exemplars/ for use as
        a style exemplar in `vault generate`. Requires:
      (a) provenance = human, OR
      (b) provenance = llm-then-human-edited AND generation_meta.human_reviewed_at set.
    For external contributor PRs: CI gates on a maintainer-approval label
        ('exemplar-approved'). Without the label, the move is rejected.

vault renumber <id> (v2.1 — Dean N-2, David N-2)
    Recovers from a dedup-seq collision after rebase. Bumps the seq, renames file,
        updates id field, appends new registry entry, updates chain references in
        other questions. Prints the exact git operations performed. Refuses on
        dirty working tree.

vault serve [--port 8001] [--allow-remote]
    Launches Datasette on vault.db. Local-only browser over the corpus (published +
        drafts). Binds 127.0.0.1 by default. --allow-remote binds 0.0.0.0 and
        requires explicit --i-understand-this-exposes-drafts flag. (Fixes L-1.)

vault api [--port 8002] [--db <path>]
    NEW in v2 (fixes H-17). Serves the production Worker API surface (GET /manifest,
        /questions, /questions/:id, /chains/:id, /taxonomy, /search) from a local
        vault.db. Lets contributors point NEXT_PUBLIC_VAULT_API=http://localhost:8002
        and run the site without any Cloudflare account. Mirrors the worker's JSON
        schema exactly — shared types package is the contract.

vault doctor [--check <name>] [--json]
    Runs diagnostic subchecks. Each check has a stable name:
      git-state, schema-version, registry-integrity, release-integrity,
      d1-connectivity, content-hash-sample (20 IDs), link-rot (nightly only),
      llm-spend-ledger.
    --check X runs only one. --json emits {check, status, detail} per row.
    Exits 0 if all green; 1 if any red. (Fixes L-4.)
```

### 4.5 Global flags and machine-readable output

Global flags: `--vault-dir <path>`, `--verbose`, `--quiet`, `--json`, `--no-color`.

`--json` schemas are per-command, documented in `vault-cli/docs/JSON_OUTPUT.md` and versioned alongside the CLI. `vault <cmd> --json-schema` prints the schema for scripting. `vault check --json` emits LSP-diagnostic-shaped errors `{uri, range, severity, message, source}` for editor integration. (Fixes M-16.)

### 4.6 Exit-code taxonomy (fixes M-18)

Stable across all commands. Documented in `vault-cli/docs/EXIT_CODES.md`.

| Code | Meaning |
|------|---------|
| 0 | success |
| 1 | validation / invariant failure |
| 2 | usage error (bad flags, missing args) |
| 3 | filesystem / I/O error |
| 4 | network / D1 / worker error |
| 5 | user aborted confirmation |
| 64–78 | reserved for sysexits.h standard categories |

---

## 5. Validation & Invariants (v2)

Every `vault build` and `vault check` runs these. Grouped by speed tier. The fast and structural tiers together must fit in **CI's 60s budget** (revised from v1's 30s — still strict, but realistic at 9,200+ files). Validation is parallelized across cores; dedup uses MinHash/LSH candidate generation before Jaro-Winkler.

### Fast (pre-commit, runs in <1s)
1. Schema validation per question (Pydantic generated from LinkML).
2. Unique IDs across all questions and drafts (checked against `id-registry.yaml`).
3. Path-field consistency (no stray `track:`/`level:`/`zone:` fields in YAML).
4. **All path components are lowercase** (fixes H-9).
5. **Each path component is enum-valid** against `taxonomy.yaml`/level-enum/`zones.yaml`.
6. Filename format conforms to `<topic-kebab>-<6-hex>-<4-digit>.yaml` (fixes C-5).
7. ID format URL-safe and matches filesystem path prefix recorded in `id-registry.yaml`.
8. Required fields present.
9. YAML loader is hardened (fixes H-7):
   - `yaml.safe_load` only — never `yaml.load`.
   - File size ≤ 256 KB; reject larger.
   - Max document depth = 10.
   - Reject YAML aliases entirely (no legitimate use in question files).
   - Parsing time-bounded at 500ms per file.
10. Content-format per field (fixes H-6):
    - `scenario`: plaintext only — validator rejects `<`, `>`, `<script`, `javascript:`, `data:`.
    - Markdown fields: CommonMark parse + allowlist pass (no raw HTML).
    - `details.deep_dive.url`: scheme must match `^https://`.

### Structural (CI, runs in <30s for 10K files — parallelized)
11. Every `topic` exists in `taxonomy.yaml`. **(Fixes the 87/79 bug.)**
12. Every `chain` reference exists in `chains.yaml`; structured form required (no legacy `@` strings on `main`).
13. Chain positions form a contiguous sequence `[1..N]`, no holes.
14. Taxonomy prerequisite graph is a DAG.
15. Applicability matrix respected — no questions in excluded `(track, topic)` cells.
16. Status transitions valid (deprecated questions not in active chains).
17. `id-registry.yaml` invariants: append-only (no deletions in diff vs. parent commit), each entry's path prefix matches an existing file's path.
18. `provenance` enum valid; `generation_meta` present iff `provenance != human`.
19. For exemplar-eligible questions: `generation_meta.human_reviewed_at` set before `vault/exemplars/` symlink allowed (fixes C-6).
20. `release-policy.yaml` is a single imported function used by every exporter (enforced by import-graph check) (fixes H-21).

### Scenario-duplicate detection (CI, <10s budget — LSH-blocked)
21. MinHash signature per scenario (k=5 shingles, 128 hashes). LSH bucketed with Jaccard threshold 0.85. Only within-bucket pairs are Jaro-Winkler–compared at threshold 0.95. (Fixes M-6.)
22. Embedding-based near-duplicate check (any small sentence-encoder cached locally) on LSH-bucket candidates, threshold 0.92. Hits require human acknowledgement via `vault dup --ack <id1> <id2>`.

### Slow (nightly CI)
23. Deep-dive URLs reachable (HTTP 200). Failures logged to `vault/link-rot.yaml`; monthly auto-filed issue for maintainer (fixes L-3).
24. Napkin-math units dimensionally consistent (Pint check, where applicable).
25. LLM math verification pass (existing `gemini_math_review.py` flow).

### Secret-leak (weekly)
26. `vault check --secrets` greps for common patterns (API keys, emails, private URLs, AWS/GCP key formats).

---

## 6. Release Workflow (v2)

### 6.1 Happy path

```
git checkout -b release/v1.0.0

vault publish 1.0.0 --sign
  → Stages to releases/.pending-1.0.0/
  → Runs: check --strict, build, snapshot, migrations emit, export paper, tag
  → Atomic symlink swap (POSIX rename) as final step
  → On any failure: deletes pending dir; `vault publish --resume` safe-retries

vault diff v0.9.0 v1.0.0 --classify
  → Review semantic vs structural changes

vault deploy 1.0.0 --env staging
  → Pre-deploy R2 snapshot of staging D1 (synchronous)
  → Schema migration (if --schema-change) then data migration, each in BEGIN;...COMMIT;
  → POP propagation probe
  → Writes release_metadata row

vault verify 1.0.0 --source releases/1.0.0/
  → Reconstructs release_hash from YAML; asserts = release.json
  → Exit 0 = citation-grade verified

vault ship 1.0.0 --env production --canary-percent 10
  → Coordinates D1 deploy + Next.js deploy (pinned NEXT_PUBLIC_VAULT_RELEASE=1.0.0)
    + paper git push
  → 10% → 50% → 100% with 15-min soak at each step
  → Auto-rollback on sub-step failure

git push origin release/v1.0.0 && git push --tags
```

### 6.1.1 `vault ship` commit protocol (v2.1 — fixes convergent Round-2 Critical)

`vault ship` is a 2-phase coordinator across three heterogeneous systems with **different rollback costs**:
- D1: stateful; rollback via R2 snapshot (minutes; always works).
- Cloudflare Pages/Workers: deploy-id; rollback via `wrangler rollback` (seconds; idempotent).
- Git tags / paper push: remote-durable once pushed; **cannot be un-pushed** without `--force` to a protected branch (forbidden by §20.2).

**Ordering (load-bearing)**: D1 first, Next.js second, paper-tag push **last**. Rationale: the last leg must be the one whose rollback is hardest or impossible. If the last leg fails, rolling back the earlier legs is cheap; if the first leg failed, no later leg has landed yet.

**Journal**: `releases/<version>/.ship-journal.json` — git-ignored per-ship file:
```json
{
  "version": "1.0.0", "env": "production", "started_at": "...",
  "legs": [
    {"name": "d1",      "state": "deployed",    "started": "...", "completed": "...", "snapshot_id": "r2://..."},
    {"name": "nextjs",  "state": "deploying",   "started": "...", "deploy_id": "..."},
    {"name": "paper",   "state": "pending"}
  ],
  "point_of_no_return": false
}
```
After `paper` leg commits, the journal's `point_of_no_return` flips `true` — past this point, auto-rollback is no longer safe; the only path is forward-fix.

**Partial-failure semantics** (who gets paged, what runs automatically):

| Failure point | Auto-rollback action | Paging |
|---|---|---|
| D1 deploy fails | Restore R2 snapshot. Abort ship. | Info-level log. |
| D1 deployed, Next.js deploy fails | `wrangler rollback` (no-op, nothing deployed). Restore D1 R2 snapshot. Abort. | Info-level log. |
| D1 + Next.js deployed, paper push fails | **Page operator.** Do NOT auto-rollback — doing so leaves the paper artifact in an in-between state. Operator decides: retry paper push or forward-fix. | Pager alert. |
| Any leg's rollback itself fails | **Page operator immediately.** Journal state is authoritative. | Pager alert + Slack. |
| Operator Ctrl-C / laptop sleep mid-ship | Journal records incomplete state. `vault ship --resume <version>` continues from last completed leg; idempotent per leg. | None (expected). |

**Paper-leg rollback** is explicitly **manual** — the paper tag is never force-pushed. If a ship fails after paper-leg commit, the remediation is a follow-up release (e.g., `v1.0.1` that reverts or forward-fixes), not a rewrite of git history.

**`X-Vault-Release` header semantics** (revised v2.1 — fixes Soumith H-NEW-2, Dean N-3): the header is **informational**, not a hard-reject gate. Workers log mismatches as a `release_skew` SLI counter and serve from `release_metadata.release_id` regardless. The **correctness boundary** is the release-keyed Cache API key (§10.2) — a new `release_id` invalidates all cached entries atomically, so a client with an older `NEXT_PUBLIC_VAULT_RELEASE` gets the current release's data (not stale), just with a stale release pin in their bundle. This eliminates the local-dev friction and SWR-revalidation brownout risks flagged in Round 2 without weakening correctness.

**Cross-release grace window** (fixes Dean N-3): during the first 10 minutes after `vault deploy`, the worker's `release_metadata` table stores both `release_id = current` and `release_id = previous` rows (the previous row is pruned after the window). `/manifest` responses carry a `Deprecation` header for the previous release so clients migrate gently. Avoids the 2–5 minute global brownout that an 8-POP-sampled propagation probe leaves otherwise.

### 6.2 Rollback model (fixes C-1)

Rollback has **two methods**. Default is snapshot restore — always works, fast, no SQL surprises.

- **`vault rollback <version> --method snapshot` (default)**: restores the R2 snapshot taken immediately before the previous deploy. This is the primary rollback path. RTO target: 10 minutes decision-to-restored.
- **`vault rollback <version> --method sql`**: applies `d1-rollback.sql`, which embeds full prior-row bodies (not mechanical SQL inversion). Useful for debugging and targeted repairs, not the primary incident path.
- **Schema-changing releases** require hand-authored `schema-rollback.sql` at publish time; a schema-change release cannot be rolled back to a prior schema without that pair. `vault publish --schema-change` refuses if the pair is missing.
- Site rollback is a feature flag: set `NEXT_PUBLIC_VAULT_FALLBACK=static` to fall back to the inlined corpus (retained for 2 releases post-cutover — see §7). This is one redeploy, no file restore required. (Fixes the "one-line revert" falsehood flagged in C-1.)

### 6.3 Backups

- **Pre-deploy R2 snapshot** of D1, synchronous, part of `vault deploy`. RPO: zero data loss for any release event (fixes M-5).
- **Nightly R2 snapshot** via GitHub Action cron (belt-and-suspenders; covers drift between deploys).
- **Retention**: 90 days rolling for nightly; forever for pre-deploy snapshots tied to a release tag.
- **Quarterly rollback drill**: disaster-recovery exercise restoring staging D1 from an R2 snapshot end-to-end, timed.

---

## 7. Website Integration (Next.js changes, v2)

### 7.1 Surface changes

**New: `interviews/staffml/src/lib/vault-api.ts`**
```ts
import { VaultClient } from '@staffml/vault-types';  // codegen'd from LinkML (H-2)

const API = process.env.NEXT_PUBLIC_VAULT_API!;
const RELEASE = process.env.NEXT_PUBLIC_VAULT_RELEASE!;  // pinned at ship time
const FALLBACK = process.env.NEXT_PUBLIC_VAULT_FALLBACK; // 'static' | undefined

const client = new VaultClient(API, {
  release: RELEASE,
  retry: { attempts: 3, backoff: 'exponential', jitter: true },
  circuitBreaker: { failThreshold: 5, resetMs: 30_000 },
  headers: { 'X-Vault-Release': RELEASE },  // worker rejects mismatch (C-4)
});
```

All client methods use cursor pagination + `ETag`-based revalidation + `Cache-Control: public, stale-while-revalidate`. See §10.2 endpoint spec.

**Rewritten: `interviews/staffml/src/lib/corpus.ts`**
- Keeps public function names; delegates to `vault-api.ts`.
- Uses SWR with keys including `RELEASE` so a deploy invalidates all client caches atomically.
- **Service worker** (`public/sw.js`) caches the last 200 visited questions for offline resilience (fixes M-12).
- **SW release-keying** (v2.1 — Chip N-H2, Dean N-6, David N-4): SW cache keys include `release_id`. On every page load, SW fetches `/manifest` and compares `release_id` to cached entries; entries with a stale `release_id` are evicted. SW uses `skipWaiting()` + `clients.claim()` on release-change so a redeploy purges SW cleanly. TTL: individual entries evict after 7 days regardless, so offline users don't see month-stale content. Phase-4 rollback drill explicitly tests "user with active SW survives `NEXT_PUBLIC_VAULT_FALLBACK=static` rollback" scenario.
- **Graceful degradation**:
  - If API returns 5xx or circuit breaker open, show cached content + "serving from cache" indicator.
  - If `NEXT_PUBLIC_VAULT_FALLBACK=static` is set, fall back to the inlined `vault-manifest.json` + static corpus (retained for 2 releases post-cutover).
  - Top-200 most-practiced questions are inlined in `vault-manifest.json` so a cold user with a broken API still sees content.

**Deletion policy (revised v2.1, Dean N-10)**:
- `src/data/corpus.json` (19 MB): retained **until the first schema-major bump post-cutover OR 2 green releases, whichever is later**. Site reads from it only when `NEXT_PUBLIC_VAULT_FALLBACK=static`. A schema-major change invalidates the static fallback as a rollback target (the shape no longer matches), so retention extends through the schema transition window.
- `src/data/corpus-index.json`: same retention.

**Kept**:
- `src/data/vault-manifest.json` (~5 KB now — includes top-200 question IDs and content hashes for offline warm-up).

### 7.2 Bundle size target (measurement-gated; fixes M-21, H-13)

Before claiming the win, Phase 4 measures real First Contentful Paint (FCP), Time to Interactive (TTI), and JS transfer size for the actual user experience — not just the build-report page weight.

| | Before | After (target) | Gate |
|---|---|---|---|
| practice/page.js (transferred, gzipped) | ? | ≤300 KB | measured on merge PR |
| gauntlet/page.js | ? | ≤250 KB | measured |
| landing/page.js | ? | ≤200 KB | measured |
| FCP (95th pct, 4G) | ? | ≤1.2s | Lighthouse CI on PR |
| TTI (95th pct, 4G) | ? | ≤2.5s | Lighthouse CI on PR |
| Repeat-visit TTI | no measurement | ≤800ms | measured |
| API round-trip p99 (question detail) | n/a | ≤250ms | new latency; measured |

Phase-4 acceptance requires all four gates green on merge. If FCP/TTI don't improve (or regress on repeat-visit due to new network dependency), feature flag defaults to `static` until fixed.

### 7.3 New UX features this unlocks

- **Real search** in command palette (FTS5-backed; see §7.4).
- **Deep linking to filtered views** (SSR-friendly `?track=cloud&level=L5&zone=evaluation`).
- **Per-user progress** stored server-side (not just localStorage) — separate feature, §14 Phase 7.
- **Question popularity analytics** (attempts, reveal rate) — separate feature, Phase 7.

### 7.4 Command palette / search UX (fixes M-13)

- Shortcut `⌘K` / `Ctrl+K` opens modal from anywhere on the site.
- 200ms input debounce; query sent as `GET /search?q=<text>&limit=20`.
- Results ranked by BM25 (FTS5 default). Snippet highlighting from FTS5 `snippet()` function.
- Empty state: "Search 9,199 questions by title, scenario, or solution."
- No-results state: suggests clearing filters + links to browse-by-topic.
- Keyboard: `↑`/`↓` navigate, `Enter` opens, `Esc` closes, `⌘↵` opens in new tab.
- Mobile: full-screen modal, 16px+ font to avoid iOS zoom, touch targets ≥44px.
- Tracks `search_query_issued` and `search_result_clicked` events (anonymous).

---

## 8. Chain Discoverability (v2 — cut to one intervention, measure, iterate)

**v1 note**: previous revision proposed five chained-UX interventions (badge, sidebar filter, tooltip, `/chains` page, dashboard tracking) as a bundle. Round-1 review flagged this as author-projection rather than observed user need (H-19). v2 cuts to one intervention, instruments it, and ships the rest only if the data says chains are genuinely underused.

**Phase 5 intervention (the only one)**: pre-reveal chain indicator. A small "Part 2 of 4 — KV-cache depth chain" badge above the question title, visible BEFORE reveal. Click-through links to chain sibling list.

**Instrumentation shipped with the intervention**:
- `chain_badge_shown` event.
- `chain_badge_clicked` event.
- Derived metric: reveal rate on chained vs non-chained questions (matched on level + zone).
- Derived metric: within-chain completion rate (did student do part 3 after part 2?).

**Gate for additional interventions (sidebar filter / tooltip / `/chains` page / dashboard)**: after two weeks of data, ship next intervention only if:
- Click-through rate on the badge > 15%, AND
- Within-chain completion rate on chained questions > 1.5× completion on non-chained at same level/zone.

If both true, chains are discoverable-but-used → add sidebar filter next. If click-through rate low, the badge is sufficient; don't add more surface area.

Chains currently participate in ~34% of questions. The paper rightly emphasizes chain depth, so the website should at least surface they exist — one badge does that.

---

## 9. About Page — "Read the Paper" Prominence

**Current problem**: academic readers land on About, want to cite the paper, have to scroll past feature descriptions and FAQ to find the link. As an academic project whose primary contribution is the paper, the link should be at the top.

**Proposed fix**:
- Move "Read the paper" link **above the fold** on the About page — as a prominent call-out card near the page hero, not in the footer/about-section.
- Display the paper's DOI (once available) and citation-ready BibTeX snippet.
- Include a "Cite this corpus" sidebar that surfaces the current release version + content hash for academic reproducibility.

**Structure suggestion**:
```
┌──────────────────────────────────────────────────┐
│  Page hero: "StaffML — ML Systems Interviews"    │
└──────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────┐
│  📄 Read the paper (PDF, arXiv link)              │  ← NEW prominence
│  Cite:                                           │
│     bibtex snippet                               │
│     (or the release content hash for a specific  │
│      corpus snapshot)                            │
└──────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────┐
│  The rest of the About page (features, FAQ, etc) │
└──────────────────────────────────────────────────┘
```

---

## 10. Cloudflare D1 + Worker (v2)

### 10.1 Database
- Name: `staffml-vault`
- Environments: staging, production
- Schema: exactly mirrors `vault.db` (zero translation); DDL codegen'd from LinkML (H-2)
- Migrations versioned under `staffml-vault-worker/migrations/`
- **Schema-fingerprint verification** (revised v2.1 — fixes Dean N-4):
  - **At cold start**: worker reads D1's actual DDL via `SELECT sql FROM sqlite_master WHERE type IN ('table','index','trigger') ORDER BY name`, hashes it (normalized whitespace), and compares against `release_metadata.schema_fingerprint`. This checks DB-vs-metadata, not metadata-vs-metadata.
  - **On mismatch**: worker does NOT 5xx the whole API (fixes Chip N-H1 total-outage risk). Instead: (a) logs `schema_fingerprint_mismatch` as a red SLI, (b) serves from the Cache API only (read-only degraded mode), (c) returns `X-Vault-Degraded: schema-fingerprint-mismatch` header so the site banner can surface the state. Operator intervention required to restore writes, but users keep seeing content.
  - This is the "fail-open to cached" pattern — a hard reject creates a total brownout; cached-degraded keeps the site usable while fingerprint desync is debugged.

### 10.2 Worker API (v2 — SWR-friendly, release-keyed caching)

All GET endpoints:
- Include `X-Vault-Release` header in responses.
- Use `ETag: "<release_id>:<resource>:<content_hash>"` for conditional GETs.
- Set `Cache-Control: public, max-age=<ttl>, stale-while-revalidate=<2×ttl>` so SWR can revalidate.
- Use the Cloudflare Cache API keyed by `release_id` so a deploy atomically invalidates all POPs (fixes H-14).

```
GET  /manifest                                       TTL 1h  (release + top-200 IDs)
GET  /questions?track=X&level=Y&zone=Z&cursor=C&limit=N
                                                     TTL 10m (cursor-paginated, fixes H-20)
GET  /questions/:id                                  TTL 1h  (immutable per release)
GET  /chains/:id                                     TTL 1h
GET  /taxonomy                                       TTL 1d
GET  /search?q=X&limit=N                             TTL 5m  (FTS5 with BM25 ranking)
GET  /stats                                          TTL 1h
```

- Cursor pagination uses opaque `{offset, filter_hash}` tokens — clients never construct cursors.
- Default order: `ORDER BY id` for stable pagination.
- All responses carry `X-Vault-Release`; client rejects cross-release responses. (Fixes C-4 skew detection.)

**Admin endpoint removed (fixes H-10)**: the v1 `POST /admin/release` cache-bust endpoint is not shipped in phases 0–6. Release-triggered cache invalidation happens via release-keyed cache keys (new `release_id` ⇒ all old keys miss). If a manual bust is needed, operator runs `wrangler` from an authenticated CLI, not a public HTTP endpoint. If a public admin surface is ever required later, it gates via Cloudflare Access (zero-trust identity), not a static bearer token, and audit-logs every call with `{identity, ip, action}` to a separate D1 table.

### 10.3 Security
- **Parameterized queries only** — no string concatenation ever; lint-checked in CI.
- **CORS allowlist** — production domains + common localhost dev ports, enumerated in `wrangler.toml`.
- **Rate limits** — via `RATE_LIMIT_KV`: 60 req/min per IP on GETs, 10 req/min on `/search`. Bot detection on UA patterns.
- **No mutation endpoints public** — writes only via `wrangler d1 migrations apply` from an authenticated operator machine.
- **Schema shared** via `@staffml/vault-types` package codegen'd from LinkML (H-2); pinned to release version.
- **CSP**: site enforces `Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; object-src 'none'` to defense-in-depth the XSS risk (H-6).

### 10.4 Cost forecast (v2 — re-modeled per H-13)

Old v1 assumption "50 page views × 20 API calls = 1K req/user/day" undercounts by ~7×. v2 models realistically.

Per active session (SWR revalidation + search typing + chain nav + detail views):
- Cold session: 30 API calls (manifest + taxonomy + initial list + 3–5 detail views + 2–3 searches)
- Warm session: ~15 API calls (ETag-revalidated most hits)
- Typical DAU pattern: 2–3 sessions per active user per day → ~75 API calls/DAU

**D1 row reads** (the binding cost driver D1 bills on):
- `/manifest`, `/taxonomy`: ~100 rows each, but Cache API hit rate ≥95% after warm → ~5 D1 reads/session for these.
- `/questions?filter`: ~50 rows returned, ~200 row reads with index scan.
- `/questions/:id`: 1 row read.
- `/search`: ~100 row reads (FTS5 + join for snippet).
- Per session: ~400 D1 row reads.

| Scale | Sessions/day | D1 row reads/day | Worker requests/day | Tier |
|---|---|---|---|---|
| 100 DAU | 250 | 100K | 19K | Free |
| 1K DAU | 2,500 | 1.0M | 188K | Free (just barely) |
| 10K DAU | 25K | 10M | 1.9M | Paid: ~$5/mo worker + ~$25/mo D1 |

**Action**: budget paid tier from day one of production ($30/mo starting assumption). Enable Cache API aggressively; without it, D1 free tier (5M row reads/day) caps us at ~500 DAU, not "100 users" or "5% utilization" as v1 claimed.

### 10.5 Data-plane SLIs (fixes H-15)

Transport-layer metrics (5xx, p99) cannot detect silent corruption. Add data-plane SLIs as Worker crons:

| SLI | Check | Frequency | Alert threshold |
|---|---|---|---|
| Row-count parity | `COUNT(*) FROM questions` in D1 == release manifest | 5 min | any mismatch |
| Content-hash sampling | 20 random IDs: D1 `content_hash` matches release manifest | hourly | any mismatch |
| FTS5 index parity | `COUNT(*) FROM questions_fts` == `COUNT(*) FROM questions` | hourly | any mismatch |
| schema_fingerprint parity | D1 row matches committed fingerprint | on cold start | refuse service |
| `/manifest` release_id propagation | 8 sampled POPs agree | on deploy + hourly | stale > 30 min |
| Validation-failure rate on main | `vault check --strict` pass/fail per merged commit | continuous | any failure |
| LLM-vs-human ratio | `provenance` distribution | daily | drift > 5%/week |
| Corpus staleness | oldest `last_modified` for status=published | daily | > 18 months |

Dashboards: Cloudflare Analytics + a `vault stats --format prometheus` scrape exported to Grafana (existing stack).

### 10.6 FTS5 performance gate (fixes H-12)

Before entering Phase 4, run a real load test:
- Corpus size: full ~9,200 questions in staging D1.
- Concurrency: 50 sustained, 200 burst.
- Query mix: 60% term, 20% phrase, 20% multi-term BM25.
- Measure: p50/p95/p99 warm, p99 cold.

Gate thresholds for Phase 4 entry:
- p99 warm ≤ 100ms on `/search`.
- p99 cold ≤ 500ms (accepts D1 binding cold-start).
- **Cost gate (v2.1 — Dean N-5)**: ≤ 500 D1 row-reads per FTS5 query, measured via Cloudflare D1 analytics on a representative query set. FTS5 with snippet extraction commonly amplifies row-reads 5-20× the logical result count. Missing this gate silently blows the §10.4 cost forecast regardless of latency.

If ANY of the three gates miss (p99 warm, p99 cold, cost), fall back to a Worker-local compiled index (corpus is ~20 MB — fits a Worker bundle with zstd) and re-measure. This is the pre-committed escape hatch, not a panicked pivot.

---

## 11. Workflow Continuity During Migration (v2 — full rewrite)

**Change from v1 (fixes C-2)**: v1 claimed "both old `corpus.json` edit flow AND new `vault new` produce the same end state." This was dual-write and would have drifted silently. v2 kills the old flow on Day 1 of Phase 1. There is one authoring surface: YAML files under `vault/questions/`. Everything else is generated.

### 11.1 Day-1-of-Phase-1 contract

The moment the YAML split lands (Phase 1 milestone):
- `vault/questions/**/*.yaml` is the **sole source of truth** for every question.
- `vault/corpus.json` becomes a **generated artifact** (written by `vault build --legacy-json`). Read-only for humans.
- A **pre-commit hook** refuses any commit that touches `vault/corpus.json` directly (detects via `git diff --name-only` + rule). Override requires a one-line exception in a commit trailer (`Vault-Override: corpus-json-hand-edit` with justification) — used only if the author has to fix a build-time regeneration bug.
- A **CI check** runs `vault build --legacy-json` and fails the PR if the committed `corpus.json` doesn't match the regenerated output byte-for-byte. This catches any drift immediately.

### 11.2 Downstream-consumer compatibility during phases 1–3

Existing scripts must keep working until they're rewritten:
- `paper/scripts/generate_macros.py`: rewritten in Phase 2 to read `vault.db` via SQL (`vault export paper <v>`). Until that lands, it reads `corpus.json` — which is now a generated artifact. Works unchanged; just reads a file the author no longer edits directly.
- `paper/scripts/analyze_corpus.py`: same path. Folded into `exporters/paper.py` in Phase 2.
- `staffml/scripts/sync-vault.py`: same path. Deprecated in Phase 4 (site reads from Worker API).
- `staffml/src/data/corpus.json`: this is separate from `vault/corpus.json`. It's the site's bundled artifact, produced by `sync-vault.py`. Unchanged during Phases 1–3. Deleted (with retention policy, see §7.1) after Phase 4 cutover.

### 11.3 Release-policy single source (fixes H-21)

`release-policy.yaml` is loaded once by a single Python function `vault_cli/policy.py:filter()`. All exporters (paper, site, D1 migration) import and call this function. The LinkML CI job enforces "no re-implementation" via an import-graph check: only `vault_cli/policy.py` may implement the predicate. The `policy_version` is stamped into `release_metadata` at build time so `release.json` records which filter produced which release.

This eliminates the 9,199/8,053 bug class structurally — you can't re-introduce it without deleting the import-graph check.

### 11.4 Cutover (Phase 4) contract

Not "one-line revert in corpus.ts." The real contract:
- Deploy the Worker API to production (already live in staging since Phase 3).
- Deploy the site with `NEXT_PUBLIC_VAULT_API` set and `NEXT_PUBLIC_VAULT_FALLBACK` unset.
- Site reads from Worker; `src/data/corpus.json` is retained for 2 releases (fallback path).
- Rollback, if needed, is a redeploy with `NEXT_PUBLIC_VAULT_FALLBACK=static` — a config change, not a file restore. Exercised once in staging as part of Phase-4 acceptance.

### 11.5 Pre-cutover equivalence testing (v2.1 — cost-aware, fixes David N-1 + Chip N-H4)

Before Phase 4 flips production:
- CI on every PR: `vault build` produces `vault.db` + `release.json`. CI compares the **release_hash** (§3.5 Merkle) against a committed `corpus-equivalence-hash.txt` checksum that was computed from the current `staffml/src/data/corpus.json` at Phase 1 exit.
- Any mismatch fails the PR. No silent drift accumulates.
- **Why hash, not byte-diff**: committing the regenerated `corpus.json` and byte-comparing is expensive (28 MB binary diff, prone to spurious failures from PyYAML/Python version drift, creates merge-conflict pain on concurrent PRs — violates M-4). Committing a short `corpus-equivalence-hash.txt` and diffing the Merkle root is O(1) and cheat-proof because content-hash is already canonicalized per §3.5.
- `corpus-equivalence-hash.txt` is updated once per Phase 2 release (not per PR); PRs that intentionally change corpus content bump this file in the same commit with a rationale comment.
- This check runs from Phase 1 exit through Phase 4 entry (roughly 4–6 weeks).
- **CI cost budget**: target ≤ 2 min total for `vault check --strict + build + equivalence-hash + codegen-drift + LSH dedup` on a standard GitHub runner. If it exceeds budget, move to a larger runner class rather than weakening checks. Incremental build caching by content-hash (only rebuild rows for changed YAML) is a Phase 2 follow-up.

---

## 12. LLM-Assisted Generation (`vault generate`, v2)

**Change from v1 (fixes C-6, H-8, L-2)**: v1 drew style exemplars from the general corpus. Since the corpus is partially LLM-generated and will eventually accept external PRs, this created a **prompt-injection loop**: one malicious question could propagate instructions into every future generation. v2 isolates exemplars and hardens cost/secret controls.

### 12.1 Invocation

```
vault generate --topic kv-cache-management \
               --zone specification \
               --track cloud --level L5 \
               --count 3 \
               --model claude-opus-4-6 \
               [--no-context] [--yes] [--dry-run]
```

### 12.2 Exemplar pool discipline (fixes C-6)

- **`vault/exemplars/` is a separate, curated, human-only directory**. Structure mirrors `vault/questions/` but every question in it must have `provenance: human` OR (`provenance: llm-then-human-edited` AND `generation_meta.human_reviewed_at` set).
- `vault generate` draws exemplars **only from `vault/exemplars/`**, never from the general corpus. Invariant enforced by the loader.
- Minimum exemplar pool per `(track, level, zone)` cell: 3 questions. `vault generate` refuses to run if the pool is smaller — a "cold-start" manual-write task.
- When a question is considered exemplar-eligible, moving it into `vault/exemplars/` is an explicit `vault mark-exemplar <id>` operation requiring maintainer approval in CI. This prevents silent corpus poisoning.

### 12.3 Prompt construction (sanitized)

1. Load 10 exemplars from `vault/exemplars/` (or N=`--exemplar-count`).
2. **Never pass exemplar `scenario`/`solution` text directly**. Pass only structural metadata: topic, level, zone, word count, napkin-math present/absent, chain depth. Free text is stripped. (Prevents exemplar-content from becoming instructions to the generating model.)
3. Schema, taxonomy context, and writing-style guidelines come from `vault/generation-guidelines/*.md` (authored by maintainers, not derived from corpus).
4. Full prompt written to `vault/generation-log/<yyyy-mm-dd>/<prompt-hash>.txt` and git-tracked before the API call (fixes L-2). `generation_meta.prompt_hash` references this file.

### 12.4 Cost & secret controls (fixes H-8)

- **Secrets**: `ANTHROPIC_API_KEY` etc. read from `~/.config/vault/secrets.toml` (mode 0600 enforced by CLI). Never from shared env vars. Never committed. `vault generate` refuses if the file is world-readable.
- **Hard cap**: `--count ≤ 25` per invocation. Higher requires `--i-know-what-im-doing` AND `VAULT_LLM_OVERRIDE=1` env var both set.
- **Dry-run default**: `vault generate --dry-run` prints the full prompt, exemplar IDs, token estimate (input + max-output), and projected USD cost. No API call.
- **Interactive confirm**: without `--yes`, print the same summary and require typed confirmation of the topic name before calling.
- **Daily spend ledger**: `vault/.llm-spend.json` tracks per-day USD spend. Configurable ceiling in `vault/llm-budget.yaml` (default $50/day). Refuses further calls if ceiling hit; error names the ledger file.
- **Actual usage recorded**: `generation_meta.prompt_cost_usd` is filled from the API response.

### 12.5 Processing

1. Call LLM with rate limit, max-tokens budget, timeout.
2. Parse response into candidate YAML files.
3. Validate each against schema (fail-closed on any schema error; partial-parse emits as many drafts as valid).
4. Write to `vault/drafts/<track>/<level>/<zone>/*.yaml` with `status: draft`, `provenance: llm-draft`, full `generation_meta`.
5. Print summary: drafted IDs, locations, next-step pointer. Exit 0 even on partial success; exit 1 only if zero drafts emitted.

### 12.6 Promotion path

Drafts do **not** reach the published corpus without human review:
- `vault edit <id>` on a draft for iteration.
- `vault promote <id>` (or `--all-drafts --topic X`) moves to `vault/questions/`, sets `status: published`, records `provenance: llm-then-human-edited` (the operator attests they reviewed).
- `vault promote` refuses without a `--reviewed-by <git-config-user>` flag recorded in the promoted YAML's `authors` list.

---

## 13. Security, Safety, Robustness — Things to remember

### ID stability (load-bearing, v2)
- IDs are content-addressed (topic + short-hash-of-title + dedup suffix) and append-only in `id-registry.yaml`. Once assigned, never changed, never reused.
- Student localStorage references IDs; changing them breaks bookmarks.
- `vault move` preserves the ID; the registry records the path change but not the ID.
- Two contributors who collide on a 6-hex hash get a 4-digit suffix bump automatically at `vault new` time.
- The registry is append-only; any commit that removes a line fails CI. (Fixes C-5.)

### YAML safety (v2, fixes H-7)
- `yaml.safe_load` only, always via `vault_cli.util.yaml.load()` which enforces: file size ≤ 256 KB, depth ≤ 10, aliases rejected, parse timeout 500 ms.
- Validate every field against Pydantic (codegen'd from LinkML) before writing to `vault.db`.
- Content-format per field (plaintext / restricted-Markdown / HTTPS-URL-only) — see §3.3 and §5 invariant 10.

### Chain integrity under edit/delete
- Hard-delete of chained question blocked unless `--force` AND `--i-understand-chain-breakage`.
- Soft-delete (status: deprecated) preserves chain slot but hides from UI.
- Chain integrity check: positions form `[1..N]` with no holes.

### Release atomicity and rollback (v2, fixes C-1, C-4, C-7, C-8)
- `vault publish` stages to `releases/.pending-<v>/` and swaps via POSIX `rename(2)` atomically at the end. Failure leaves no half-committed state. `vault publish --resume` detects orphaned pending dirs.
- `vault ship` coordinates D1 deploy + Next.js deploy + paper tag push as a single atomic operation with auto-rollback.
- D1 migrations run in `BEGIN;...COMMIT;`; `schema_fingerprint` in `release_metadata` prevents serving across skew.
- **Primary rollback path is snapshot restore**, not inverse SQL. Pre-deploy R2 snapshot is synchronous. Secondary rollback (`--method sql`) exists for targeted repairs.
- Schema-changing releases require hand-authored up/down SQL pair.

### Content integrity and academic citation (v2, fixes C-3)
- Per-question `content_hash` and release-level Merkle `release_hash` are hashed over canonical JSON of **whitelisted semantic fields** — never over SQLite bytes (which are not reproducible).
- `vault verify <version>` reconstructs from YAML and asserts equality. This is what Zenodo / external readers run.
- Paper cites a specific release ID + release_hash, not "the corpus."
- `vault publish --sign` produces `minisign` signatures against `vault/signing-key.pub` for academic archiving.
- Consider registering a DOI per major release (via Zenodo integration).

### Observability (v2, fixes H-15)
- Transport metrics (5xx, p99 latency, request count, anomaly detection) — Cloudflare Analytics.
- Data-plane SLIs — see §10.5 table. Row-count parity, content-hash sampling, FTS5 index parity, schema_fingerprint parity. These detect silent corruption that transport metrics miss.
- Authoring health — LLM-vs-human ratio, corpus staleness, validation-failure rate on main — exported via `vault stats --format prometheus`.
- Weekly review of slow queries → candidate index additions.
- Error tracking: Cloudflare Workers observability + Sentry for the site.
- Alert SLOs: 5xx > 1% over 5min; p99 > 500ms sustained; any data-plane SLI divergence; validation failure on main.

### Corpus licensing (v2 recommendation, L-10)
- **Corpus**: CC-BY-4.0 (recommended default — permits citation and reuse with attribution; matches norms for open research datasets). Blocks Phase 3 until user confirms.
- **`vault-cli` code**: MIT (consistent with other tools in this ecosystem).
- License decision must be resolved **before Phase 3**; see §15 Open Questions.

### Collaborator safety
- External contributions via PR. CI runs `vault check --strict` on every PR.
- Never merge a PR that fails invariants.
- Reviewer guidelines: check scenario quality, taxonomy accuracy, napkin math.

### Concurrent CLI use (v2, fixes C-5)
- `vault publish` uses a lockfile (`vault/.publish.lock`) with PID + 30-min stale-lock timeout.
- `vault generate` uses a lockfile on `vault/.llm-spend.json` to serialize spend-ledger updates.
- ID allocation (in `vault new`) is content-addressed + append-only registry; no single-machine lock needed. Collision-by-hash is 1-in-16M per `(topic, title-prefix)` bucket; `--dedup-seq` suffix handles the tail.
- `vault new` requires the working tree to be current with origin (`git pull --rebase` before allocation) to minimize rebase pain downstream.

### Version skew (v2, fixes H-1)
- Each YAML carries `schema_version`. Loader rejects versions it doesn't know.
- Full schema-evolution rules in [`schema/EVOLUTION.md`](./schema/EVOLUTION.md) — Phase-0 deliverable:
  - SemVer: minor bumps are additive-only; major bumps require migration.
  - v2 loader reads v1 by filling defaults; v1 loader rejects v2 clearly.
  - `vault migrate-schema v1 v2` runs in a topic branch, writes to a parallel tree, validates *all* files pass the new schema before `git mv` swap. Supports `--dry-run`. Partial-failure logs written to `vault/.migrate.log`.
  - CI blocks PRs that mix schema versions within `vault/questions/` — migration is all-or-nothing per PR.

### Shared types contract (v2, fixes H-2)
- LinkML schema at `vault/schema/staffml_taxonomy.yaml` and `vault/schema/question_schema.yaml` is the **sole** schema SSoT.
- `vault publish` codegens:
  - Pydantic models → `vault-cli/src/vault_cli/models/` (internal use).
  - SQL DDL → `releases/<v>/schema.sql` (applied to D1).
  - TypeScript types → `@staffml/vault-types` package, versioned with the release, consumed by site + worker.
- **Codegen "who runs it" contract** (v2.1 — Soumith H-NEW-3): PR authors run `vault codegen` locally before committing any LinkML-schema change, and commit the regenerated artifacts in the same PR. CI runs `vault codegen --check` which re-runs codegen in a tempdir and diffs against the committed output — fails the PR on drift, never pushes follow-up commits. This is the "trusted-but-verify" pattern; CI is the gate, not the fixer.
- **TS package versioning** (v2.1 — Soumith M-NEW-1): `@staffml/vault-types` lives as a workspace package at `interviews/staffml-vault-types/`, referenced via pnpm `workspace:*` protocol in the site + worker `package.json`. No external npm publish. Version follows the vault release (`1.0.0`, `1.0.1`, ...); the workspace protocol ensures site and worker always pick up the current codegen output without a separate publish step.
- CI fails if committed generated artifacts don't match current LinkML output. "No drift" is structurally enforced, not aspirational.

### Don't-bypass safeguards
- Pre-commit hook is a convenience; **CI is the gate**.
- GitHub Actions runs `vault check --strict` on every PR. Merge-blocked on failure.
- Branch protection on `main`: required CI status + at least one review.

---

## 14. Rollout Phases (v2 — timeline revised per H-18)

Each phase is a safe stopping point. If priorities shift, pause at a phase boundary without leaving the system in a half-migrated state.

**Timeline honesty**: v1 estimated 11 focused days for phases 0–6. Round-1 review flagged this as ~80% under (H-18). v2 revises to **18–22 focused days**. Every phase has an "overrun policy" — what happens when the real world exceeds the budget.

### Phase 0 — Kickoff (1 day) — POST-BOOK-DEADLINE (after 2026-04-22)
- Scaffold `vault-cli/` Python package (pyproject.toml, Typer main entry, test skeleton).
- Write `vault-cli/README.md` (install, quickstart, test, architecture pointer). (Fixes L-7.)
- Write `interviews/CONTRIBUTING.md` documenting clone → first-question-visible path. (Fixes H-17.)
- Write `vault/schema/EVOLUTION.md` — SemVer rules, loader contract, migration mechanics. (Fixes H-1.)
- Write `vault-cli/docs/JSON_OUTPUT.md` and `EXIT_CODES.md`.
- CI scaffolding: `.github/workflows/vault-ci.yml` runs `vault check` placeholder (expanded in Phase 1).
- **Exemplar-coverage audit** (v2.1 — Chip N-H3): `vault stats --exemplar-coverage` reports which `(track, level, zone)` cells have <3 human-reviewed questions eligible for the `vault/exemplars/` pool. Output to `vault/exemplar-gaps.yaml`. This is a READ-ONLY audit at Phase 0; filling gaps is backlog work that unblocks `vault generate` (Phase 7), not a Phase 0 blocker.
- **Milestone**: `pip install -e vault-cli/ && vault --version` works. Skeleton CI green. Exemplar gap inventory produced.

### Phase 1 — Foundation (5 days) — was 3
- LinkML `question_schema.yaml` + Pydantic codegen + SQL DDL codegen + TS types codegen.
- `release-policy.yaml` + `vault_cli/policy.py` (single filter predicate; import-graph gate).
- `vault-cli` commands: `new`, `edit`, `move`, `rm`, `restore`, `build`, `check`, `serve`, `api`.
- `split-corpus.py` one-shot: current `corpus.json` → ~9,199 YAML files under `vault/questions/`.
- Pre-commit hook forbidding direct `vault/corpus.json` edits; CI check on equivalence (§11.1).
- `vault api` localhost Worker-surface shim (H-17).
- Fix every orphan/invariant violation discovered.
- **Milestone**: `vault build` produces `vault.db` with release_hash + per-question content_hash matching the split inputs. Site still reads `corpus.json` (unchanged behavior for users).
- **Overrun policy**: if >50 invariant violations surface after split, cap fix budget at 2 days; remaining violations go to `vault/known-issues.yaml` and get fixed in Phase 7.

### Phase 2 — Release pipeline (3 days) — was 2
- `vault snapshot`, `vault migrations emit` (with full prior-row embedding), `vault export paper`, `vault tag`.
- `vault publish` composed product with staged-rename atomicity.
- `vault verify` end-to-end.
- `paper/scripts/generate_macros.py` rewritten via `vault export paper`.
- Rollback-symmetry property test in CI on every publish.
- **Milestone**: paper and site agree on exact same counts. Fixes 9,199/8,053 and 87/79 bugs. Rollback symmetry proven on staged releases.

### Phase 3 — D1 infrastructure (5 days) — was 2
- License decision resolved (L-10, gates this phase).
- Create `staffml-vault` D1 database (staging + prod).
- Scaffold `staffml-vault-worker/` with cursor-paginated endpoints + ETag + Cache API keyed by `release_id`.
- `@staffml/vault-types` npm package codegen + publish flow.
- Import current release into staging D1; populate `release_metadata`.
- **FTS5 load-test gate** (see §10.6): 50 concurrent clients, 9,200 docs, realistic queries. p99 warm ≤ 100 ms, p99 cold ≤ 500 ms. If gate misses, implement Worker-local compiled index fallback.
- Data-plane SLI crons live in staging (row-count parity, content-hash sampling, schema_fingerprint check).
- `vault deploy` primitive with pre-deploy R2 snapshot + POP propagation probe.
- **Milestone**: staging worker serves correct data; FTS5 gate green; data-plane SLIs reporting zero divergence.

### Phase 4 — Website cutover (4 days) — was 2
- `lib/vault-api.ts` with retry/backoff/circuit-breaker + SWR.
- Service worker for offline cache of last 200 questions.
- Rewritten `lib/corpus.ts` thin wrapper.
- `NEXT_PUBLIC_VAULT_FALLBACK=static` feature flag wired end-to-end.
- Canary-staged production rollout via `vault ship --canary-percent`.
- Manual QA via `vault-cli/docs/CUTOVER_QA.md` checklist (see §19).
- Bundle/FCP/TTI Lighthouse CI gates (see §7.2 table).
- Rollback drill executed on staging before production ship.
- **Milestone**: production ship green; Lighthouse gates pass; rollback path exercised; `corpus.json` retained (not deleted) per §7.1.

### Phase 5 — Chain discoverability (1 day, then instrument)
- Ship only the pre-reveal chain indicator + instrumentation.
- **Gate** on two weeks of data (§8): additional interventions (sidebar filter / tooltip / `/chains` / dashboard) deferred unless click-through rate + within-chain completion thresholds hit.
- **Milestone**: badge live; dashboards reading instrumentation.

### Phase 6 — About page paper prominence (0.5 day)
- "Read the paper" call-out above the fold.
- Citation-ready BibTeX snippet + DOI (if registered) + release_id + release_hash footer.
- Contributor list (§3.3 `authors:` aggregated).
- **Milestone**: academic readers find the paper in ≤5s from landing on About.

### Phase 7 — Polish (ongoing, not in critical path)
- `vault generate` LLM command (per §12).
- `vault-cli` migrations, diff polish, doctor subchecks.
- R2 nightly snapshot GitHub Action (pre-deploy snapshots already in Phase 3).
- Per-user progress on D1 (separate feature; requires auth; privacy-policy work).
- Phase-5 gated interventions (if data justifies).

**Total focused work**: **18–22 working days** for phases 0–6 (vs v1's 11). Phase 7 is ongoing.

**Recommended timing**: Start Phase 0 after 2026-04-22 (MIT Press copyedit deadline). Do not overlap Phase 1–2 with paper copyedit — any change to `macros.tex` could conflict with typesetter edits.

---

## 15. Open Questions (v2)

Recommendations are v2's proposed defaults. User confirmation required for items marked **BLOCKS PHASE N**.

1. **Corpus license**. **Recommend CC-BY-4.0** (permits citation and reuse with attribution; norms for open research datasets). `vault-cli` code: MIT. **BLOCKS PHASE 3** — external contributor PRs cannot be accepted without resolving this. (L-10.)
2. **DOI registration**. Recommend Zenodo integration per major release; each release gets a citable DOI tied to `release_hash`. Resolve before first "v1.0.0" release.
3. **Draft approval model**. Recommend: maintainer self-approve for Phase 0–6; 2-reviewer model for external PRs starting in Phase 7. Documented in `CONTRIBUTING.md`.
4. **Student data policy**. Per-user progress is Phase 7+ (separate feature, not in critical path). When added: GDPR/CCPA review; privacy policy update; data-deletion UX.
5. **Worker API authentication**. Recommend: public read endpoints indefinitely (no paywall planned). Rate limits prevent abuse. Re-evaluate if scraping becomes a problem.
6. **Corpus snapshot retention**. Recommend: keep every release artifact (`releases/<v>/`) forever — cheap (each ~20 MB) and load-bearing for academic reproducibility. Rolling nightly R2 snapshots: 90-day retention. Pre-deploy snapshots tied to release tags: forever.
7. **Internationalization**. Not in scope for phases 0–6. If pursued later, adds `language:` field to schema (major version bump) — `schema/EVOLUTION.md` covers the mechanics.

**Items resolved in v2** (moved from "pending" in v1):
- CLI framework — **Typer + Rich** (§3.2).
- CLI command name — **`vault`** (§3.2).
- Package directory — **`interviews/vault-cli/`**.
- Content-hash definition — **SHA-256 over canonical JSON** (§3.5).
- Rollback model — **snapshot-restore primary, inverse SQL secondary** (§6.2).
- Concurrent-ID model — **content-addressed + append-only registry** (§3.3).
- Release atomicity — **staged-rename + `vault ship`** (§6.1, §4.3).
- Exemplar pool — **`vault/exemplars/` curated human-only** (§12.2).

---

## 16. Open Things the User Asked Me to Think About (2026-04-15 session)

1. **Chain visibility on website** — covered in §8.
2. **"Read the paper" prominence on About page** — covered in §9.
3. **Rich/Typer CLI** — covered in §3.2.
4. **Workflow continuity during dev** — covered in §11.

Additional items I'd add to the "think about this too" list:

- **Test corpus** — a frozen small subset of questions used as the test fixture for the CLI's own tests. Must never drift.
- **CI speed budget** — `vault check --strict` must stay under 30s or contributors will rage.
- **Monorepo discipline** — this architecture touches 4 subprojects (vault, vault-cli, staffml, staffml-vault-worker). Document which PRs are cross-cutting.
- **Secrets in wrangler.toml** — all rate limits, CORS allowlists, cache TTLs should be config, not hardcoded.
- **Canary deploys** — consider Cloudflare Workers' percentage-based rollout for new worker versions.
- **Question popularity analytics** — once on D1, instrument anonymously (reveal rate, avg time before reveal). Feeds back into corpus quality work.
- **Public corpus mirror** — offer a read-only `vault.db` download on the site, versioned by release. Academic readers may want offline access.
- **Author identity / contributor list** — the paper will have authors. Does the website surface them on About? How do we credit future contributors?

---

## 17. Immediate Next Actions (for the next session)

When resuming:

1. Confirm naming: `vault` command, `vault-cli/` package directory.
2. Confirm tech: Typer + Rich for the CLI.
3. Confirm phase timing: post-2026-04-22 for Phase 0.
4. **Two-hour pilot** (optional, highly recommended): convert 20 questions to per-file YAML, build one vault.db, launch Datasette, feel the ergonomics. Zero impact on production.
5. Decide the 7 open questions in §15.

---

## Appendix: References to current code

| Current component | New home |
|---|---|
| `interviews/vault/corpus.json` (28 MB) | `interviews/vault/questions/**/*.yaml` |
| `interviews/vault/scripts/generate.py` | `vault-cli/src/vault_cli/commands/generate.py` |
| `interviews/vault/scripts/export_to_staffml.py` | Gone. Replaced by `vault publish` + `exporters/d1.py`. |
| `interviews/staffml/scripts/sync-vault.py` | Gone. |
| `interviews/staffml/scripts/generate-manifest.py` | Built by `vault publish`. |
| `interviews/paper/scripts/analyze_corpus.py` | Folded into `exporters/paper.py`. |
| `interviews/paper/scripts/generate_macros.py` | Rewritten in `exporters/paper.py` — SQL-driven. |
| `interviews/staffml/src/lib/corpus.ts` (static import) | Thin wrapper over `vault-api.ts`. |
| `interviews/staffml/src/data/corpus.json` | Deleted post-cutover. |

---

## 18. Review & Iteration Protocol (pre-implementation)

Before ANY code is written, this plan must survive adversarial expert review. The review is iterative — not a single pass. Target: **2 rounds minimum, 3 if round 2 surfaces substantive issues.**

### Reviewer panel

Four reviewers, run in parallel each round. Chosen to cover orthogonal angles:

| Reviewer | Lens | What they'll catch |
|---|---|---|
| `expert-chip-huyen` | Production ML & DX | Security gaps, client trust boundaries, prompt-injection surfaces, operational pitfalls |
| `expert-jeff-dean` | Large-scale systems | Scale, reliability, observability, cost model, concurrency, data integrity |
| `expert-soumith-chintala` | Framework & API design | CLI ergonomics, API contracts, developer experience, backwards compatibility |
| `student-david` | Industry user perspective | Will the authoring flow actually feel good? What's confusing? What's missing? |

### Per-round protocol

**Round 1 — "read cold and tell me what's wrong"**
- Send each reviewer the full ARCHITECTURE.md with explicit brief: identify risks, gaps, bugs, and weaknesses. Rank by severity.
- Aggregate findings into a single table: {issue, reviewer, severity, proposed fix}.
- Integrate every `severity: critical` and `severity: high` item into ARCHITECTURE.md v2.
- `severity: medium` items get addressed OR explicitly deferred with rationale in the doc.
- `severity: low` items: logged for later, not blocking.

**Round 2 — "does v2 address your concerns?"**
- Send each reviewer ARCHITECTURE.md v2 PLUS the round-1 findings-table with our responses.
- Explicit brief: "do our responses resolve your concerns? what did we miss? anything NEW you see now?"
- If no critical/high issues remain: plan is approved. Stop.
- If critical/high issues remain: integrate into v3, run round 3.

**Round 3 — "final adversarial pass"** (conditional)
- Only if round 2 still surfaces critical/high issues.
- Same reviewers, same protocol.
- After round 3: any remaining critical/high must be resolved by explicit engineering decision + rationale OR the plan is not ready and needs a larger rework before proceeding.

### Commit discipline

- After each round: commit ARCHITECTURE.md with message `docs(vault): architecture v2 (round-N review integration)`.
- Review findings table committed alongside as `REVIEWS.md` — durable record of what was raised, what was fixed, what was deferred, why.

### Exit criteria for this stage

1. Zero open critical/high issues across all four reviewers.
2. All deferred medium issues have written rationale.
3. REVIEWS.md captures the full iteration history.
4. ARCHITECTURE.md version stamped in section 0 (status header).

---

## 19. Testing Plan (to be written as `TESTING.md`)

Testing is NOT an afterthought. It's the gate between stages. This section is the *skeleton*; full spec lives in `interviews/vault/TESTING.md` (written before Phase 0 implementation).

### 19.1 Test layers

| Layer | Scope | When it runs | Blocker for |
|---|---|---|---|
| **Unit** | Single function, pure logic (validator rules, ID generation, path parsing) | Every commit, via `pytest` | Any PR merging |
| **Integration** | Multi-component (YAML → vault.db, chain integrity, schema validation across corpus) | Every commit, via `pytest` | Any PR merging |
| **Contract** | CLI command behavior end-to-end (`vault new`, `vault publish`, etc.) | Every commit, via `pytest` | Phase transitions |
| **Data-migration** | Current `corpus.json` → YAML files → `vault.db` produces byte-equivalent content per ID | Once at Phase 1 exit | Phase 2 start |
| **Export parity** | Site + paper agree on counts exactly | Once at Phase 2 exit | Phase 3 start |
| **Worker contract** | D1 worker endpoints match TypeScript types, handle edge cases | Phase 3, continuous | Phase 4 start |
| **End-to-end** | Real site fetching from staging D1, full practice/gauntlet flow | Phase 4, pre-cutover | Production cutover |
| **Smoke** | 50 random question IDs: staging worker vs direct `vault.db` query, byte-identical JSON | Pre-every-production-deploy | Production deploy |
| **Load** | Worker under realistic traffic (100 req/s sustained, 500 req/s burst) | Phase 4, pre-cutover | Production cutover |
| **Rollback** | `vault rollback` produces state byte-identical to previous release | Phase 2, continuous | Every deploy |

### 19.2 Test fixtures

- **Test corpus**: 20 frozen questions covering all tracks, levels, zones, including 3 chained sequences. Lives at `vault-cli/tests/fixtures/test-corpus/`. Never changes unless fixture bump is explicit.
- **Golden vault.db**: expected SQLite output from the test corpus. Byte-comparison against built output is the primary integration test.
- **Schema drift fixtures**: deliberate broken YAMLs (missing fields, wrong types, invalid chain refs) to assert validator catches each class of error.

### 19.3 CI integration (v2)

GitHub Actions workflow `.github/workflows/vault-ci.yml`:
1. On every PR touching `interviews/vault/` or `interviews/vault-cli/`:
   - Run `pytest vault-cli/` (all unit + integration + contract tests).
   - Run `vault check --strict` on the full corpus (fast + structural tiers, <60s budget).
   - Run `vault build` and diff the resulting **release manifest** (`release.json`) against the committed manifest — NOT against a committed `vault.db` (binary merge conflicts unresolvable; fixes M-4).
   - Run equivalence check vs `staffml/src/data/corpus.json` during Phase 1–3 (§11.5).
   - For every merged release on `main`: run **rollback-symmetry property test** (fixes M-1) — apply forward migration, then inverse, compare full DB state to pre-migration. Fails the release if not round-trippable.
   - Run codegen drift check (fixes H-2): recompute LinkML → TS/Pydantic/DDL artifacts and diff against committed.
   - Report results as PR status check.
2. Merge-blocked on any red status.
3. Nightly workflow `.github/workflows/vault-nightly.yml`:
   - Slow-tier invariants (link rot, LLM math verification).
   - D1 R2 snapshot (belt-and-suspenders; primary snapshots are pre-deploy).
   - Data-plane SLI sweep (row-count parity, content-hash sampling across production).

### 19.4 Manual QA checklist (cutover day)

Written as `vault-cli/docs/CUTOVER_QA.md`. Every flow tested manually:
- [ ] Home page: question count matches manifest
- [ ] Practice page: load, filter by track, reveal answer, navigate chain, ask tutor
- [ ] Gauntlet: start session, complete N questions, view post-mortem
- [ ] Progress: attempts persist, due count correct
- [ ] About: "Read the paper" visible above fold, BibTeX renders
- [ ] Command palette: full-text search returns expected results
- [ ] Network tab: no request for `corpus.json`
- [ ] Bundle size: confirm practice/page.js < 500 KB
- [ ] Rollback drill: revert to previous release, site works
- [ ] Re-apply new release: site works

### 19.5 Observability during rollout (v2)

During Phase 4 cutover:
- Worker request logs → Cloudflare Analytics dashboard.
- **Transport alerts**: 5xx rate > 1% over 5min; p99 > 500ms sustained.
- **Data-plane alerts** (new, fixes H-15): row-count parity divergence, content-hash mismatch on sampled IDs, FTS5 index desync, schema_fingerprint mismatch, `/manifest` release_id staleness across POPs. Any of these trips an alert.
- **Traffic canary**: `vault ship --canary-percent 10` → 50 → 100 with 15-min soak at each level.
- **Rollback**: set `NEXT_PUBLIC_VAULT_FALLBACK=static` and redeploy — a config change, not a file restore. Rehearsed on staging as a Phase-4 acceptance step.

---

## 20. Autonomous Mode — When and How

After Stages 1 (review) and 2 (testing plan) are gated through, implementation switches to autonomous execution.

### 20.1 Pre-autonomous checklist

Before flipping to autonomous mode, verify:
- [ ] ARCHITECTURE.md v3+ (or v2 if round 2 clean) committed and pushed
- [ ] REVIEWS.md committed with full iteration history
- [ ] TESTING.md committed
- [ ] CUTOVER_QA.md drafted
- [ ] Zero open critical/high issues from reviewers
- [ ] User has explicitly green-lit autonomous execution

### 20.2 Autonomous execution rules

Once in autonomous mode, the operator (future Claude session) executes Phases 0 → 6 with:

1. **Fresh feature branch** off current `dev`: `feat/vault-architecture`.
2. **Working from** `/Users/VJ/GitHub/MLSysBook-staffml` (the StaffML worktree).
3. **Commit style** per `.claude/CLAUDE.md`:
   - No `Co-Authored-By` lines
   - No automated attribution footers
   - Atomic commits, one logical change each
   - Descriptive messages explaining the *why*
4. **Never force-push.** Never merge to `dev` without explicit user approval.
5. **Never delete data** without a rollback path in the same commit.
6. **After every phase**: run full test suite, commit, push, post phase-complete summary.

### 20.3 Checkpoint protocol

At the end of each phase:
- Summary post: what was built, what was tested, what changed in counts/bundle/latency, what's next.
- Wait for user ack before starting the next phase. (Brief — single message exchange.)
- If the user doesn't respond, continue to next phase after noting the unacked checkpoint.

### 20.4 Stop conditions

Stop autonomous execution and ask the user if:
- A test fails in a way not covered by TESTING.md
- An invariant check catches something genuinely ambiguous
- A design decision in ARCHITECTURE.md turns out to be wrong in practice
- A phase runs long enough that rolling back seems preferable to forward-fixing
- Anything touches data outside `interviews/vault/`, `interviews/vault-cli/`, `interviews/staffml/`, `interviews/staffml-vault-worker/`, `interviews/paper/scripts/` without being pre-authorized
- The user asks

### 20.5 Success definition

Autonomous execution is complete when:
1. Production site serves from D1 (no static `corpus.json` in bundle)
2. Paper reads from `vault.db` (SQL-driven macros)
3. Per-file YAML is the sole authoring surface
4. All tests green
5. Manual QA checklist fully passed
6. Zero data loss verified (set equality of question IDs, content hash per ID)
7. Paper and site agree on all counts by construction
8. CUTOVER_QA.md all items checked
9. Monitoring dashboards report healthy for 48 hours post-cutover

---

## 21. Kickoff Prompt (copy-paste into new session)

The exact text the user pastes to launch the next session is maintained in a separate file for ease of copying:

**`interviews/vault/KICKOFF.md`**

Keep that file in sync with this architecture doc. When the plan changes in a way that alters the kickoff instructions, update both.

---

**End of architecture document.**
