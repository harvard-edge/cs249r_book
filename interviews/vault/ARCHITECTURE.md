# StaffML Vault Architecture — Design Document

> **Status**: Approved direction, pre-implementation. Drafted 2026-04-15.
> **Scope**: Everything from question authoring to production serving.
> **Audience**: Future-VJ, future maintainers, future collaborators.
> **Supersedes**: `SYSTEM.md` (stale — says 5,786 questions, 839 concepts).

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

### 3.3 Per-question YAML format

**Filename**: `<topic-kebab>-<4-digit-seq>.yaml`
**Location**: `vault/questions/<track>/<level>/<zone>/<filename>`
**Classification**: encoded in the path — NOT in the file. Moving the file reclassifies the question.

```yaml
# vault/questions/cloud/L4/diagnosis/kv-cache-bandwidth-0042.yaml
schema_version: 1
id: cloud-L4-diagnosis-kv-cache-bandwidth-0042  # immutable, URL-safe
title: "KV Cache Memory Bandwidth Bottleneck"
topic: kv-cache-management              # must exist in taxonomy.yaml
chain: kv-cache-depth@2                 # optional. format: <chain-id>@<position>
status: published                       # draft | published | deprecated
created_at: 2026-04-15T10:00:00Z
last_modified: 2026-04-15T10:00:00Z
generated_by: human                     # human | llm:<model-id>
generation_meta:                        # optional, for LLM-generated
  model: claude-opus-4-6
  prompt_hash: sha256:abc123...
  temperature: 0.4

scenario: |
  <prompt text the student sees>

details:
  common_mistake: |
    <common wrong approach>
  realistic_solution: |
    <canonical answer>
  napkin_math: |
    <step-by-step arithmetic if applicable>
  deep_dive:
    title: "PagedAttention and KV Cache Management"
    url: "https://mlsysbook.ai/book/chapters/serving"

tags:
  - hardware:a100
  - size:70b
```

**Design rules**:
- `id` is immutable. Once assigned, never reused. Registered in `id-registry.yaml`.
- Filesystem path is authoritative for track/level/zone. YAML has no `track:` etc. fields.
- `chain: <id>@<position>` is a compact single-string form.
- `generated_by` is mandatory — tracks provenance.
- Every question carries `schema_version` to support future schema migrations.

### 3.4 SQLite schema (built by `vault build`)

See the compiler output in `vault-cli/src/vault_cli/compiler.py`. Key tables:

```
questions(id PK, title, topic FK, track, level, zone, status,
          scenario, common_mistake, realistic_solution, napkin_math,
          deep_dive_title, deep_dive_url, generated_by,
          created_at, last_modified, file_path, content_hash)

chains(id PK, name, topic FK)
chain_questions(chain_id, question_id FK, position, PK(chain_id, position))
tags(question_id FK, tag)
taxonomy(id PK, name, area, description)
taxonomy_edges(source FK, target FK, edge_type)
zones(id PK, name, skills, description)
release_metadata(key PK, value)

questions_fts: FTS5 virtual table on title, scenario, realistic_solution
```

Indexed: `topic`, `(track, level)`, `zone`, `status`. FTS5 for full-text search.

---

## 4. CLI Specification

Framework: **Typer** (declarative, type-hint-driven). Output: **Rich** (tables, progress, panels).

```
vault new [--topic X] [--track Y] [--level Lz] [--zone Z]
    Prompts for missing fields. Assigns next ID from id-registry.
    Creates YAML from Jinja template at correct path. Opens in $EDITOR.
    On save: validates schema; on failure, keeps file but flags error.

vault edit <id>
    Opens the file in $EDITOR. Re-validates on save.

vault rm <id> [--hard]
    Default: marks status: deprecated (preserves chain slot, hides from UI).
    --hard: deletes file. Refuses if question is in any active chain unless --force.

vault move <id> --to <track>/<level>/<zone>
    Reclassifies. `git mv` under the hood.

vault build [--output <path>]
    Walks questions/**/*.yaml. Validates each. Compiles to vault.db.
    Fast: < 10s for 10K files. Reports all errors, doesn't fast-fail.

vault check [--strict]
    Runs all invariants. Exit 0 on pass, non-zero on any failure.
    Used by pre-commit hook (--strict=false, fast checks only) and CI (--strict=true).

vault publish <version> [--sign]
    1. vault check --strict (fail-closed)
    2. vault build → vault.db
    3. Copy to releases/<version>/
    4. Generate d1-migration.sql (UPSERT deltas) + d1-rollback.sql (inverse)
    5. Generate paper/macros.tex and paper/corpus_stats.json via SQL
    6. Write release.json (content hash, counts, git SHA, timestamp)
    7. Update latest/ symlink
    8. Git-commit artifacts on a release branch
    9. Tag v<version>
    --sign: also produces a SHA-256 signature of vault.db for academic archiving.

vault deploy <version> --env (staging|production)
    Applies d1-migration.sql to target D1 environment.
    Prints release.json summary, requires interactive confirmation.
    Logs the deploy to D1's release_metadata table.

vault rollback <version> --env (staging|production)
    Applies d1-rollback.sql to revert to previous state.

vault stats [--topic X] [--track Y] [--level Z] [--zone W] [--format table|json|csv]
    Quick scorecard. Reads from latest vault.db.

vault diff <from-version> <to-version>
    Shows added, removed, modified questions between two releases.

vault generate --topic X --zone Y --level Lz --track T --count N [--model M]
    LLM-assisted generation. Queries existing vault for style context.
    Writes drafts to vault/drafts/. Each carries generated_by + prompt_hash.
    Does NOT publish. Requires `vault promote` after human review.

vault promote <id>  |  vault promote --all-drafts [--topic X]
    Moves drafts to vault/questions/. Sets status: published.

vault serve [--port 8001]
    Launches Datasette on vault.db. Local-only web browser over the corpus.

vault doctor
    Diagnostic: git state, schema versions, release integrity, D1 connectivity.
```

Global flags: `--vault-dir`, `--verbose`, `--json` (machine-readable output).

---

## 5. Validation & Invariants

Every `vault build` and `vault check` runs these. Grouped by speed tier.

### Fast (pre-commit, runs in <1s)
1. Schema validation per question (Pydantic).
2. Unique IDs across all questions and drafts.
3. Path-field consistency (no stray track/level/zone fields in YAML).
4. Filename format conforms to `<topic-kebab>-<4-digit>.yaml`.
5. ID format URL-safe.
6. Required fields present.
7. YAML parses via `yaml.safe_load` (never `yaml.load`).

### Structural (CI, runs in <10s for 10K files)
8. Every `topic` exists in `taxonomy.yaml`. **(Fixes the 87/79 bug.)**
9. Every `chain` reference exists in `chains.yaml`.
10. Chain positions form a contiguous sequence `[1..N]`, no holes.
11. Taxonomy prerequisite graph is a DAG.
12. Applicability matrix respected — no questions in excluded (track, topic) cells.
13. Status transitions valid (deprecated questions not in active chains).
14. No two questions with identical scenario (fuzzy match, Jaro-Winkler > 0.95).

### Slow (nightly CI)
15. Deep-dive URLs reachable (HTTP 200).
16. Napkin-math units dimensionally consistent (Pint check, where applicable).
17. LLM math verification pass (existing `gemini_math_review.py` flow).

### Secret-leak (weekly)
18. `vault check --secrets` greps for common patterns (API keys, emails, private URLs).

---

## 6. Release Workflow

```
git checkout -b release/v1.0.0

vault publish 1.0.0
  → Emits release artifacts.

vault diff v0.9.0 v1.0.0
  → Review what changed.

vault deploy 1.0.0 --env staging
  → Applies migration to staging D1.

# Smoke test
curl https://mlsysbook.ai/api/vault-staging/manifest
curl https://mlsysbook.ai/api/vault-staging/questions/cloud-L4-diagnosis-kv-cache-0042

vault smoke-test --env staging --samples 50
  → Compares staging worker output vs direct vault.db query.
  → Must match byte-for-byte.

vault deploy 1.0.0 --env production
  → Same migration, production target.
  → Requires typed confirmation.

git push origin release/v1.0.0
git push --tags
```

**Rollback**: `vault rollback v0.9.0 --env production` inverts the last migration.

**Backups**: nightly Cloudflare R2 snapshot of D1 (GitHub Action cron).

---

## 7. Website Integration (Next.js changes)

### 7.1 Minimal surface changes

**New: `interviews/staffml/src/lib/vault-api.ts`**
```ts
const API = process.env.NEXT_PUBLIC_VAULT_API!;

export async function getManifest(): Promise<Manifest> { ... }
export async function getQuestionById(id: string): Promise<Question | null> { ... }
export async function searchQuestions(params: SearchParams): Promise<SearchResult> { ... }
export async function getChain(id: string): Promise<Chain> { ... }
export async function getTaxonomy(): Promise<Taxonomy> { ... }  // cached aggressively
```

**Rewritten: `interviews/staffml/src/lib/corpus.ts`**
- Keeps public function names; delegates to `vault-api.ts`.
- Uses SWR for client-side caching.
- Graceful fallback: if API unreachable, show "service temporarily unavailable" rather than crashing.

**Deleted after cutover**:
- `src/data/corpus.json` (19 MB)
- `src/data/corpus-index.json`
- Static import chain inflated into JS bundle.

**Kept**:
- `src/data/vault-manifest.json` (1.2 KB) — inlined for instant first-paint counts.

### 7.2 Bundle size target

| | Before | After |
|---|---|---|
| `app/practice/page.js` | 20 MB | ~300 KB |
| `app/gauntlet/page.js` | 19 MB | ~250 KB |
| `app/page.js` (landing) | 19 MB | ~200 KB |
| First-paint JS parse time | ~800 ms | ~50 ms |

### 7.3 New UX features this unlocks

- **Real search** in command palette (FTS5-backed). Currently client-side O(n) scan.
- **Deep linking to filtered views** (SSR-friendly `?track=cloud&level=L5&zone=evaluation`).
- **Per-user progress** stored server-side (not just localStorage).
- **Question popularity analytics** (how many attempts, reveal rate, etc.).

---

## 8. Chain Discoverability (NEW — called out separately because it's invisible today)

Chains exist in the data but are **buried in the UX**. Fixes planned:

1. **Pre-reveal indicator**: Small "Part 2 of 4 in the KV-cache chain" badge above the question title. Visible BEFORE reveal, so students know they're in a chain.
2. **Sidebar filter**: "Chained questions only" toggle next to the Track filter. Students can seek out chains.
3. **First-time tooltip**: One-time callout explaining chains the first time a student lands on a chained question.
4. **`/chains` browse page**: Landing page listing all chains by topic. Each chain shows its depth profile (L1→L6+) and a "Start chain" button.
5. **Chain completion tracking**: Progress indicator in user dashboard showing completed chains.

Chains currently participate in ~34% of questions (per `corpus_stats.json`). This feature surfaces the existing depth-structure the paper rightly emphasizes.

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

## 10. Cloudflare D1 + Worker

### 10.1 Database
- Name: `staffml-vault`
- Environments: staging, production
- Schema: exactly mirrors `vault.db` (zero translation)
- Migrations versioned under `staffml-vault-worker/migrations/`

### 10.2 Worker API
```
GET  /manifest                                       — release metadata, cached 1h
GET  /questions?track=X&level=Y&zone=Z&limit=N       — filtered list
GET  /questions/:id                                  — single lookup
GET  /chains/:id                                     — chain with ordered questions
GET  /taxonomy                                       — full topic graph, cached 1d
GET  /search?q=flash+attention&limit=20              — FTS5 match
GET  /stats                                          — aggregates, cached 1h
(future, bearer-auth)
POST /admin/release                                  — explicit cache bust
```

### 10.3 Security
- **All queries parameterized** — no string concat ever.
- **CORS allowlist** — production domains + common localhost dev ports.
- **Rate limits** — reuse existing `RATE_LIMIT_KV` infrastructure.
- **No mutation endpoints public** — writes only via `wrangler d1 migrations apply`.
- **Schema shared** via TypeScript types package so client + worker don't drift.

### 10.4 Cost forecast
- D1 free tier: 5M reads/day, 10 GB storage. We'd use ~5% of each at production scale.
- Worker free tier: 100K requests/day. Assume 50 page views × 20 API calls = 1K req/user/day. Free tier covers 100 users/day; paid tier starts at $5/mo for 10M requests.

---

## 11. Workflow Continuity During Migration

**Critical**: the current site, paper, and author workflow **must not break during the migration**. Plan must support parallel operation of old and new pipelines until cutover day.

### Parallel-operation invariants

During migration phases 1–3:
- `vault/corpus.json` continues to exist and is **regenerated as a side-output** of `vault build` (from the YAML files). Existing `sync-vault.py` and `generate_macros.py` continue to work.
- Site still reads `src/data/corpus.json`. Unchanged behavior for users.
- Paper's macros still regenerate from the same (now YAML-sourced) corpus.
- Author can still add questions via the old "edit corpus.json directly" flow OR via `vault new` — both produce the same end state.

In phase 4 (cutover):
- One-day change: switch site from static import to worker API. Delete `corpus.json` from the bundle.
- Rollback plan: one-line revert in `lib/corpus.ts` to fall back to static import.

### Testing continuity

Before cutover:
- Full parallel run: current pipeline + new pipeline output must agree on:
  - Question count (exact integer match)
  - Question IDs (set equality)
  - All per-question content (content hash match per ID)
  - Chain membership (chain graph isomorphism)

---

## 12. LLM-Assisted Generation (`vault generate`)

The future workflow replaces ad-hoc `generate_batch.py` with a principled subcommand:

```
vault generate --topic kv-cache-management \
               --zone specification \
               --track cloud --level L5 \
               --count 3 \
               --model claude-opus-4-6 \
               --context-from-corpus
```

Steps:
1. Query current vault for 10 stylistically similar existing questions (`WHERE topic=X AND level=Y AND zone=Z LIMIT 10`).
2. Construct prompt with schema, taxonomy context, style exemplars.
3. Call LLM with rate limit + max-tokens budget.
4. Parse response into candidate YAML files.
5. Validate each against schema (fail-closed on any schema error).
6. Write to `vault/drafts/cloud/L5/specification/*.yaml` with `status: draft` and full generation_meta.
7. Print summary: "Wrote 3 drafts. Review with `vault edit <id>` and promote with `vault promote <id>`."

This gives LLM assistance WITHOUT letting unreviewed content reach the published corpus.

---

## 13. Security, Safety, Robustness — Things to remember

### ID stability (load-bearing)
- IDs are immutable. Once assigned (via `id-registry.yaml`), never change, never reuse.
- Student localStorage references IDs; changing them breaks bookmarks.
- CLI refuses to accept a user-supplied ID that already exists.

### YAML safety
- Always use `yaml.safe_load`. Never `yaml.load`.
- Validate every field against Pydantic before writing to `vault.db`.

### Chain integrity under edit/delete
- Hard-delete of chained question blocked unless `--force`.
- Soft-delete (status: deprecated) preserves chain slot but hides from UI.
- Chain integrity check: positions form `[1..N]` with no holes.

### Release rollback
- Every release produces both forward AND inverse migrations.
- D1 deploy is always reversible in one command.
- R2 nightly snapshots as belt-and-suspenders.

### Academic integrity
- Every release has an immutable content hash.
- Released `vault.db` committed to `vault/releases/<version>/` — permanently citable.
- Paper should cite a specific release ID, not just "the corpus."
- Consider registering a DOI per major release (via Zenodo integration).

### Corpus licensing
- **DECISION PENDING**: What license do the questions ship under?
  - Options: CC-BY, CC-BY-NC, CC-BY-SA, all rights reserved.
  - Matters if external researchers want to cite / use the corpus.

### Observability
- Worker emits structured logs to Cloudflare Analytics.
- Error tracking: use Sentry or built-in Workers observability.
- Alert on:
  - 5xx rate > 1%
  - p99 latency > 500ms
  - Daily request count anomalies
- Weekly review of slow queries → candidate index additions.

### Collaborator safety
- External contributions via PR. CI runs `vault check --strict` on every PR.
- Never merge a PR that fails invariants.
- Reviewer guidelines: check scenario quality, taxonomy accuracy, napkin math.

### Concurrent CLI use
- `vault publish` uses a lockfile (`vault/.publish.lock`) with PID + stale-lock timeout.
- ID allocation uses filesystem lock on `id-registry.yaml`.

### Version skew
- Each YAML carries `schema_version`. CLI checks compatibility.
- Future schema migrations via `vault migrate-schema v1 v2` — one-shot transform across all files.

### Don't-bypass safeguards
- Pre-commit hook is a convenience; **CI is the gate**.
- GitHub Actions runs `vault check --strict` on every PR. Merge-blocked on failure.
- Branch protection on `main`: required CI status + at least one review.

---

## 14. Rollout Phases

Each phase is a safe stopping point. If the book deadline changes or priorities shift, we can pause at any phase boundary without leaving the system in a half-migrated state.

### Phase 0 — Kickoff (0.5 day) — POST-BOOK-DEADLINE
- Create the branch.
- Scaffold `vault-cli/` Python package (pyproject.toml, Typer main entry, test skeleton).
- CI job that runs `vault check` on every PR.

### Phase 1 — Foundation (3 days)
- Write `question_schema.yaml` + Pydantic model generator.
- Write `release-policy.yaml` (settle the filter predicate).
- Write `vault-cli` commands: `new`, `edit`, `build`, `check`.
- Write `split-corpus.py` one-shot: current `corpus.json` → 9,199 YAML files.
- Run; fix every orphan/invariant violation discovered.
- **Milestone**: `vault build` produces `vault.db` that matches current `corpus.json` content byte-for-byte (by ID).

### Phase 2 — Release pipeline (2 days)
- Write `vault publish` end-to-end.
- Write `exporters/paper.py` (SQL → macros.tex).
- Write `exporters/d1.py` (SQL migration emitter).
- Migrate `paper/scripts/generate_macros.py` to read `vault.db`.
- **Milestone**: paper and site agree on exact same counts. Fixes the 9,199/8,053 and 87/79 bugs.

### Phase 3 — D1 infrastructure (2 days)
- Create `staffml-vault` D1 database (staging + prod).
- Scaffold `staffml-vault-worker/`.
- Import current release into staging D1.
- Build smoke-test tool: 50 random questions, worker response vs direct DB query.
- **Milestone**: staging worker serves correct data; p99 < 100ms.

### Phase 4 — Website cutover (2 days)
- Write `lib/vault-api.ts`.
- Rewrite `lib/corpus.ts` as thin wrapper.
- Deploy site on dev branch pointing at staging D1.
- Manual QA: practice, gauntlet, progress, landing, about.
- Switch production to production D1.
- Delete `src/data/corpus.json`.
- **Milestone**: production bundle < 500 KB. All flows working. Rollback path exercised.

### Phase 5 — Chain discoverability UX (1 day)
- Pre-reveal chain indicator.
- Sidebar chain filter.
- First-time tooltip.
- `/chains` browse page.
- **Milestone**: chains feel like a first-class feature, not a hidden one.

### Phase 6 — About page reorganization (0.5 day)
- "Read the paper" call-out above the fold.
- Citation-ready snippet (BibTeX + DOI when available).
- Release version + content hash in footer for reproducibility.
- **Milestone**: academic readers find the paper in 2 seconds.

### Phase 7 — Polish (ongoing)
- `vault generate` LLM command.
- `vault diff` release comparison.
- Datasette integration for analysis.
- R2 nightly snapshot GitHub Action.
- Per-user progress on D1 (separate feature; requires auth).

**Total focused work**: ~11 working days for phases 0–6. Phase 7 is ongoing.

**Recommended timing**: Start Phase 0 after 2026-04-22 (MIT Press copyedit deadline). Do not overlap Phase 1–2 with paper copyedit (any change to macros.tex could conflict with typesetter's edits).

---

## 15. Open Questions (decide before starting)

1. **Corpus license**. CC-BY? CC-BY-NC? All rights reserved? Affects external collaboration.
2. **DOI registration**. Register per release via Zenodo? Affects citation.
3. **Draft approval model**. Single-maintainer approve, or require 2 reviewers? Affects collaborator onboarding.
4. **Student data policy**. If we move per-user progress off localStorage to D1, GDPR/CCPA implications. Privacy policy update required.
5. **Worker API authentication**. Public read endpoints forever? Or eventual paid tier behind bearer auth?
6. **Corpus snapshot retention**. Keep every release forever? Or prune old releases?
7. **Internationalization**. Any plan to translate? Affects schema (need `language:` field on questions if yes).

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

### 19.3 CI integration

GitHub Actions workflow `.github/workflows/vault-ci.yml`:
1. On every PR touching `interviews/vault/` or `interviews/vault-cli/`:
   - Run `pytest vault-cli/` (all unit + integration + contract tests)
   - Run `vault check --strict` on the full corpus
   - Run `vault build` and diff against last committed `vault.db`
   - Report results as PR status check
2. Merge-blocked on any red status.

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

### 19.5 Observability during rollout

During Phase 4 cutover:
- Worker request logs → Cloudflare Analytics dashboard
- Error rate alert: > 1% 5xx over 5 minutes
- Latency alert: p99 > 500ms sustained
- Traffic canary: 10% → 50% → 100% with 15-min soak at each level
- Ability to revert to static-JSON site bundle via one-line code change if red metrics

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
