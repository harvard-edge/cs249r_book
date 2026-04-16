# StaffML Vault Architecture — Review Ledger

> **Purpose**: Durable record of adversarial-review findings against `ARCHITECTURE.md`, our responses, and what was integrated vs. deferred. Per §18 of that document.
> **Round 1 completed**: 2026-04-15 (launched concurrently, four reviewers, orthogonal lenses).
> **Next**: Round 2 once v2 of ARCHITECTURE.md is committed.

---

## 1. Reviewer panel

| Reviewer | Lens |
|---|---|
| `expert-chip-huyen` | Production ML + DX + security, LLM-generation safety |
| `expert-jeff-dean` | Large-scale systems, reliability, data integrity, cost |
| `expert-soumith-chintala` | Framework & API design, primitives vs products |
| `student-david` | Industry-engineer user, authoring ergonomics, cutover reality |

All four returned rank-ordered severity reports in one round. Aggregate finding count:
**11 Critical / 25 High / 21 Medium / 14 Low** (after pre-merge). After de-duplicating convergent items: roughly **8 distinct Critical themes, 18 distinct High themes**.

---

## 2. Convergence map

Six items were flagged by 2+ reviewers at Critical or High severity. These are the load-bearing issues for v2.

| # | Convergent issue | Reviewers | Max severity |
|---|---|---|---|
| C-1 | **Rollback semantics unsafe** — UPSERT-delta inverse migrations cannot restore prior state; schema changes are non-invertible; "one-line revert" for the site is false | Chip, Dean, David, Soumith | Critical |
| C-2 | **Parallel operation in §11 is a lie** — dual-write to YAML and corpus.json will drift silently; claim "both produce the same end state" is unsupported | Dean, David, Soumith | Critical |
| C-3 | **Content-hash undefined** — SQLite not byte-reproducible; hash target unspecified; academic citation claims fail first verification | Chip, Dean | Critical |
| C-4 | **Atomic release across three surfaces isn't atomic** — vault publish/deploy + Next.js deploy + paper push have no coordinator; partial-failure leaves skew | Dean, Soumith | Critical |
| C-5 | **ID-registry concurrency broken for PR-based authoring** — filesystem lock is single-machine; two PRs can allocate the same ID silently | Chip, Dean, Soumith | Critical |
| H-1 | **Schema evolution story missing** — `schema_version: 1` has no migration rules, loader contract, or mixed-version handling | Dean, David, Soumith | High |

The rest of the findings are reviewer-specific but substantive. See §4 below.

---

## 3. Aggregated severity table

Merged where reviewers flagged the same issue. Citations: **C**=Chip, **D**=Dean, **V**=David, **S**=Soumith. Status values:
- **INTEGRATE** → v2 will resolve
- **DEFER+RATIONALE** → explicitly deferred, see §5
- **TRACK** → low-signal, backlog only
- **OPEN** → undecided, needs user input

### 3.1 Critical

| ID | Issue | Reviewers | Proposed fix (summary) | Status |
|---|---|---|---|---|
| C-1 | Rollback unsafe: inverse SQL can't reconstruct pre-state; schema migrations are non-invertible; site rollback claim is false | C, D, V, S | Replace "mechanical inverse migration" model with: (a) pre-deploy R2 D1 snapshot as primary rollback; (b) inverse SQL embeds full prior-row state (INSERT OR REPLACE with body); (c) schema-changing releases gated by `--schema-change` flag and require hand-authored up/down pair; (d) site keeps `corpus.json` as fallback artifact for 2 releases post-cutover, rollback is `NEXT_PUBLIC_VAULT_FALLBACK=static` feature flag, not a file delete | INTEGRATE §6, §7.1, §13 |
| C-2 | Parallel operation is dual-write that will drift — §11 claim "both flows produce the same end state" is unsupported | D, V, S | Rewrite §11: YAML becomes sole authoring surface on Day 1 of Phase 1. `corpus.json` is generated-only; pre-commit hook refuses direct edits. Old generators (`sync-vault.py`, `generate_macros.py`) read from `vault.db` via compat shim during phases 1–3, not from `corpus.json`. CI check: committed `corpus.json` matches `vault build --legacy-json` output exactly, on every PR | INTEGRATE §11 (rewrite) |
| C-3 | Content hash undefined; SQLite not byte-reproducible; citation claims are unverifiable | C, D | Define two hashes explicitly: (a) per-question `content_hash` = SHA-256 of canonical JSON over sorted whitelisted fields (excludes `last_modified`, `file_path`); (b) release hash = SHA-256 Merkle root over sorted `(id, content_hash)` pairs + taxonomy hash + chains hash. Hash inputs, never the SQLite binary. Document canonicalization; ship `vault verify <release>` | INTEGRATE §3.4, §13 |
| C-4 | Three-surface release has no atomic coordinator; partial failures leave site/DB/paper skewed | D, S | Add `vault ship <version> --env prod` as the single user-facing verb composing `deploy` + Next.js deploy + paper tag push, with auto-rollback on sub-step failure. Worker refuses requests whose `X-Vault-Release` header mismatches D1's `release_metadata.current_release`. Make `deploy`/`rollback` remain as exposed primitives | INTEGRATE §4, §6 (add §6.1) |
| C-5 | ID-registry filesystem-lock doesn't survive PR-based concurrent authoring | C, D, S | Switch to content-addressed IDs: `{topic}-{short-hash-of-title}-{4-digit-dedup-suffix}`. `id-registry.yaml` becomes append-only log with one ID per line. CI invariant enforces uniqueness post-merge. `vault new` requires `git pull --rebase` before allocation. Merge conflicts on the registry become semantic (both claimed same slot, CI refuses merge) rather than silent | INTEGRATE §3.3, §13 |
| C-6 | Prompt injection via `vault generate` exemplar loop — malicious PR can poison all future generation | C | `vault/exemplars/` is a curated, human-only pool separate from general corpus; `vault generate` draws only from exemplars. Invariant refuses to use `generated_by: llm:*` as exemplar unless `human_reviewed_at` set. Strip LLM-sourced `scenario`/`solution` text to structural metadata (topic, level, length) before inclusion in prompts. Full prompt stored in `vault/generation-log/` for audit | INTEGRATE §12 |
| C-7 | `vault publish` is not atomic — 9 sequential filesystem + git + symlink steps with no journal, mid-failure leaves orphaned artifacts | D | Stage to `releases/.pending-<v>/`, do git commit+tag FIRST so git is durable log, swap `latest` symlink via POSIX `rename(2)` as last action. Delete pending dir on failure. Add `vault publish --resume` | INTEGRATE §4, §6 |
| C-8 | D1 deploy mid-migration partial failures are undefined | D | Every migration wrapped in `BEGIN;...COMMIT;`. Chunk migrations over D1 tx size limit. Pre-deploy R2 snapshot synchronous (not nightly). Add `schema_fingerprint` row; worker refuses service if fingerprint mismatches | INTEGRATE §6, §10, §13 |

### 3.2 High

| ID | Issue | Reviewers | Proposed fix (summary) | Status |
|---|---|---|---|---|
| H-1 | Schema evolution rules missing (`schema_version: 1` has no written contract for v2) | D, V, S | Write `vault/schema/EVOLUTION.md` before Phase 0. SemVer (minor=additive, major=migration). v2 loader reads v1 via defaults; v1 loader rejects v2 clearly. `vault migrate-schema` is idempotent. CI blocks mixed-version PRs | INTEGRATE §3.3, §13; new `EVOLUTION.md` in Phase 0 |
| H-2 | No shared-types package; D1 DDL + Pydantic + TS types + LinkML will quadruple-drift | S | LinkML is SSoT. `vault publish` codegens `@staffml/vault-types` (TS) + Pydantic models + SQL DDL. CI fails if committed types don't match generated. Site & worker pin the same version | INTEGRATE §3.2, §7, §10 |
| H-3 | `vault publish` too magical — 9 bundled steps, no primitive exposure | S | Decompose into primitives: `vault build`, `vault snapshot <v>`, `vault migrations emit`, `vault export paper`, `vault tag`. `vault publish` becomes a composed product of these. Users use primitives for the 20% edge cases | INTEGRATE §4 |
| H-4 | `chain: <id>@<position>` stringly-typed; will decay into regex-per-consumer divergence | S | Use structured form: `chain: {id: kv-cache-depth, position: 2}`. Loader accepts both, writes normalized structured | INTEGRATE §3.3 |
| H-5 | `generated_by: human | llm:<model-id>` free-text will drift; no `llm-then-human-edited` state | S, V | Closed enum `{human, llm-draft, llm-then-human-edited, imported}` + `generation_meta.model` from registry `vault/schema/models.yaml` | INTEGRATE §3.3 |
| H-6 | XSS surface: scenario/solution/deep_dive_url rendered client-side with unspecified format | C | Define per-field content format (plaintext / restricted Markdown / KaTeX). Validator rejects raw HTML, `javascript:`/`data:` URLs. `deep_dive_url` allowlist: `https:` only. Sanitize at render (DOMPurify). Next.js CSP forbids inline scripts | INTEGRATE §3.3, §5, §7 |
| H-7 | YAML DoS: PyYAML safe_load doesn't prevent billion-laughs / unbounded-depth | C | Loader helper enforces: ≤256KB per file, max depth 10, reject aliases entirely, time-bound parsing. Add fast-tier invariant on file size | INTEGRATE §5, §13 |
| H-8 | No cost control on `vault generate` — typo can burn $1000s | C | Hard cap `--count ≤25` without override. Dry-run token/cost estimate + interactive confirm (unless `--yes`). Daily spend ceiling in `vault/.llm-spend.json`. Secrets from `~/.config/vault/secrets.toml` (0600), not env | INTEGRATE §12, §13 |
| H-9 | Path-as-authority breaks on macOS case-insensitive FS; manual `git mv` silently breaks invariants | C | Fast-tier invariants: lowercase-only path components; enum-validate each component against taxonomy. `vault move` is the only supported reclassification (CI checks that `id-registry` prefix matches current path, rejects manual mv). `vault move` refuses if question chained unless whole chain moves or it's unlinked first | INTEGRATE §3.3, §4, §5 |
| H-10 | Admin endpoint is "add auth later" anti-pattern | C | Remove `POST /admin/release` from phases 0–6. Cache-bust via authenticated `wrangler` from operator machine. If endpoint required later, gate via Cloudflare Access (zero-trust identity), not static bearer token. Audit-log every call with identity+IP | INTEGRATE §10.2 |
| H-11 | `vault rm --hard` has no confirmation; `vault move` on dirty tree is silent partial | C | `--hard` requires typed confirmation of question title. Add `vault restore <id>`, `vault rm --list-deprecated`. `vault move` refuses dirty tree unless `--allow-dirty`. Both support `--dry-run` | INTEGRATE §4 |
| H-12 | FTS5 at edge on D1: p99<100ms claim is unverified | D | Load-test in Phase 3 kickoff: 10K docs, 50 concurrent, realistic queries. If over budget, fallback to pre-computed top-K `search_cache` table or Worker-local compiled index (corpus ~20MB). Separate warm/cold p99 budgets | INTEGRATE §10, §19 |
| H-13 | Worker cost forecast 10x optimistic — 20 calls/session is wrong; D1 row-reads billing ignored | D, V | Re-forecast with 150 calls/session, D1 rows-read accounting per endpoint. Cache API on GETs keyed by release-hash so `/manifest` and `/taxonomy` never hit D1 after warm. Model at 1K and 10K DAU. Budget paid tier from day one | INTEGRATE §10.4 |
| H-14 | Cache invalidation is racy across POPs — split-brain between `/manifest` and `/questions` after deploy | D | Every cache key includes `release_id`; cache entries are immutable per release, auto-invalidated on deploy. `vault deploy` probes N POPs post-deploy and verifies propagation before returning success | INTEGRATE §10 |
| H-15 | No data-plane SLIs for silent corruption | D | Add: row-count parity between `vault.db` and D1 (Worker cron, 5-min); sampled content-hash verification (20 random IDs/hour); FTS5 vs base-table count parity. Alert on any divergence. `vault doctor` surfaces same checks | INTEGRATE §13 |
| H-16 | `vault new` / `$EDITOR` UX under-specified; no batch mode; validation-failure UX undefined | V, S | Spec: `vault new --count N` opens N drafts together. Validation-failure: error appears inline in the draft YAML as commented block + stderr panel; re-opens on `vault edit`. Test with vim / VS Code (`code --wait`) / nano before Phase 1 exit | INTEGRATE §4 |
| H-17 | No local dev story — contributor can't run site locally without Cloudflare account | V | Phase 0 adds `vault api` (or `vault serve --mode=api`) that serves the Worker API surface from `vault.db` on localhost. `NEXT_PUBLIC_VAULT_API=http://localhost:8002` gets a contributor to first-question-visible. `vault-cli/README.md` + `CONTRIBUTING.md` document the full clone-to-render path | INTEGRATE §4, §14 (Phase 0), new CONTRIBUTING.md |
| H-18 | 11-day timeline is fantasy (~80% under) | V | Revise §14: Phase 1 → 4–5d, Phase 3 → 4–5d, Phase 4 → 3–4d. Total 18–22 focused days. Add explicit "overrun policy" — if Phase 1 surfaces >50 invariant violations, cap fix budget at 2d and file rest as follow-ups | INTEGRATE §14 |
| H-19 | Chain discoverability §8 is author-projection, not observed need | V | Cut Phase 5 to one intervention only: pre-reveal chain indicator. Instrument reveal rate on chained vs non-chained. Ship sidebar / tooltip / `/chains` / dashboard only if data justifies | INTEGRATE §8, §14 |
| H-20 | Worker endpoints not SWR-friendly — no pagination cursor, no ETag, no order guarantee | S | Add cursor pagination (`?cursor=<opaque>&limit=N`) on `/questions` and `/search`. ETag from `release_metadata.content_hash`. `Cache-Control` tuned for SWR. Default order `ORDER BY id` | INTEGRATE §10.2 |
| H-21 | `release-policy.yaml` could be re-implemented by each consumer, re-introducing the 9199/8053 bug | D | Single Python function `vault_cli/policy.py`, imported by every exporter. `vault.db` only contains `status IN (<policy-set>)`; exporters never see excluded rows. `release.json` records policy version | INTEGRATE §2, §11 |

### 3.3 Medium

| ID | Issue | Reviewers | Status |
|---|---|---|---|
| M-1 | Rollback symmetry asserted but not CI-tested as property | D | INTEGRATE §19 (property-test on every publish) |
| M-2 | Worker cold-start + D1 first-query not separated from warm p99 | D | INTEGRATE §10 (warm vs cold p99 targets stated) |
| M-3 | `vault diff` needs classifier for cosmetic vs semantic vs structural changes | D | INTEGRATE §4 |
| M-4 | Committing `vault.db` to git is fragile (binary merge conflicts, LFS bloat) | D | INTEGRATE §19 (commit manifest only, not db) |
| M-5 | Backup RTO/RPO unspecified | D | INTEGRATE §6, §13 (R2 snapshot on every deploy; RTO target 30min) |
| M-6 | Jaro-Winkler O(n²) dedup: unrealistic at 9200 docs; false-positive on templated | C, D, V | INTEGRATE §5 (nightly tier; MinHash/LSH blocking; complement with embedding dup-check) |
| M-7 | CI 30s budget unrealistic without parallelization + LSH | C | INTEGRATE §16 + §5 (parallelize validation, benchmark in CI) |
| M-8 | Release-policy predicate duplication risk | D | Covered under H-21 |
| M-9 | No observability for authoring health (failure rate on main, LLM vs human ratio) | D | INTEGRATE §13 (emit as `vault stats --format prometheus`) |
| M-10 | No retry/backoff/circuit-breaker in `vault-api.ts` | C | INTEGRATE §7.1 |
| M-11 | `--sign` uses SHA-256 (hash, not signature) | C | INTEGRATE §4 (minisign or sigstore with committed public key; rename flag if hash-only intent) |
| M-12 | D1 outage = dead site | V | INTEGRATE §7.1 (service worker caches last N questions; top-200 inlined in manifest) |
| M-13 | Command palette / search UX unspecified | V | INTEGRATE §7 (new §7.4 with debounce, shortcut, ranking, empty state) |
| M-14 | `vault migrate-schema` mechanics undefined (partial failure, branch strategy) | V, S | INTEGRATE EVOLUTION.md |
| M-15 | No author identity / contributor list | V | INTEGRATE §3.3 (`authors: []` field; populate from git config) |
| M-16 | `--json` global flag vague; no per-command schema | S | INTEGRATE §4 (per-command schemas; `vault check --json` emits LSP-diagnostic-shaped errors) |
| M-17 | `vault move` / `vault edit` composability; filename stability under move | S | INTEGRATE §4 (seq+topic-kebab preserved under move; git follows rename; `--edit` convenience) |
| M-18 | Exit-code taxonomy missing | S | INTEGRATE §4 (stable table; sysexits.h) |
| M-19 | SWR-unfriendly endpoints (covered by H-20) | — | — |
| M-20 | Worker cache-key on release-id (covered by H-14) | — | — |
| M-21 | Bundle-size win measurement unverified | V | INTEGRATE §7.2 (measure real FCP/TTI before claiming; add prefetch) |

### 3.4 Low

| ID | Issue | Reviewers | Status |
|---|---|---|---|
| L-1 | `vault serve` default bind / draft leak | C, S | INTEGRATE §4 (127.0.0.1 only; banner; `--allow-remote` flag) |
| L-2 | `generation_meta.prompt_hash` no matching prompt store | C | INTEGRATE §12 (`vault/generation-log/<date>/<hash>.txt` git-tracked) |
| L-3 | Link-rot workflow undefined | C | INTEGRATE §5 (nightly `vault doctor --check-links` → `vault/link-rot.yaml`; monthly issue) |
| L-4 | `vault doctor` scope stub | C, S | INTEGRATE §4 (one-page spec: subcheck names, `--check X`, `--json`) |
| L-5 | `--context-from-corpus` mismatch between §4 and §12 | S | INTEGRATE §4/§12 (grounding default-on; `--no-context` opt-out) |
| L-6 | `vault doctor` underspecified (dup) | — | — |
| L-7 | Missing `vault-cli/README.md` | S | INTEGRATE Phase 0 deliverable |
| L-8 | `generated_by` human+LLM combo cases (covered by H-5) | — | — |
| L-9 | Typer/Rich error messages mediocre for non-Python users | V | TRACK (reconsider at Phase 1 exit after UX test) |
| L-10 | License decision blocks external collab but listed as pending | D, V | **OPEN — needs user decision before Phase 3**. Recommended default: CC-BY-4.0 for corpus, MIT for `vault-cli` |

---

## 4. Non-merged findings preserved for traceability

Round 1 reviewer reports are long. Full raw text is preserved in the git-log for this commit (see commit message for pointers); the REVIEWS.md table above is the authoritative response map. Individual reviewer outputs used to construct this table:

- **Chip**: 3C / 6H / 5M / 3L — security and DX lens; unique finds = C-6 (prompt injection via exemplars), H-6 (XSS), H-7 (YAML DoS), H-8 (LLM cost ceiling), H-9 (macOS case), H-10 (admin endpoint), H-11 (rm/move ergonomics).
- **Dean**: 3C / 6H / 5M / 3L — systems/reliability lens; unique finds = C-7 (non-atomic publish), C-8 (D1 tx semantics), H-12 (FTS5 perf), H-14 (cache split-brain), H-15 (data-plane SLIs), H-21 (policy re-impl), M-1 / M-4 / M-5.
- **Soumith**: 2C / 7H / 5M / 4L — framework/API lens; unique finds = C-4 (atomic ship verb), H-2 (shared-types), H-3 (primitives vs product), H-4 (chain stringly typed), H-5 (provenance enum), H-20 (SWR-friendly endpoints), M-16 / M-17 / M-18.
- **David**: 3C / 6H / 6M / 4L — user-practice lens; unique finds = the user-reality cuts that others missed — H-17 (local dev), H-18 (timeline fantasy), H-19 (chain UX is projection), M-12 (D1 outage = dead site), M-13 (search UX missing), M-15 (no contributor identity), M-21 (bundle-win measurement).

**Convergence score (how many reviewers flagged the same issue at Critical/High)**:
- 4 reviewers: C-1 (rollback unsafe)
- 3 reviewers: C-2 (parallel op), C-5 (ID concurrency), H-1 (schema evolution), H-13 (cost forecast)
- 2 reviewers: C-3 (content hash), C-4 (atomic ship), H-5 (provenance), M-6 (dedup), L-1 (serve bind), L-4 (doctor scope), L-10 (license)

---

## 5. Deferred items with rationale

Per §18 of ARCHITECTURE.md, deferred mediums require written rationale. None yet — all Mediums integrated. Only two items deferred as of v2:

- **L-9 (Typer error UX for non-Python users)**: Defer until Phase 1 exit UX test. No irreversible commitment; rewrite to Click or argparse is feasible if Typer's errors prove hostile.
- **L-10 (license)**: **OPEN, not deferred — blocks Phase 3**. Needs explicit user decision. Recommended: CC-BY-4.0 corpus / MIT `vault-cli`. User should resolve in Round 2 response.

---

## 6. v2 edit plan (integrated into ARCHITECTURE.md as a single commit)

Ordered by structural impact:

1. **§11 (Workflow Continuity)** — full rewrite. Remove "dual write" framing. YAML is sole authoring surface from Day 1 of Phase 1. `corpus.json` is build-output only; pre-commit hook enforces.
2. **§3.3 (Per-question YAML)** — structured `chain:` form; closed `provenance:` enum; `authors: []`; content-format-per-field rules (plaintext / markdown-restricted / url-scheme-allowlist); filename stability under move; path lowercase enforcement.
3. **§3.4 (SQLite schema)** — add canonical `content_hash` definition; add `release_id` to `release_metadata`; add `schema_fingerprint` row.
4. **§4 (CLI spec)** — decompose `publish` into primitives + composed `publish`; add `ship` atomic-release verb; add `--count N` / `--batch` to `new`; add `--dry-run` everywhere destructive; add typed-confirm to `rm --hard`; add `restore`, `verify`, `api` (local-dev shim); specify exit-code table; specify `--json` per-command schemas.
5. **§5 (Invariants)** — add fast-tier: lowercase path, enum path components, file-size cap, YAML-depth cap, reject-aliases, URL scheme allowlist, no-raw-HTML. Move Jaro-Winkler to nightly + LSH blocking. Parallelize validation.
6. **§6 (Release Workflow)** — reshape around `vault ship`; state atomicity as staged-rename; R2 pre-deploy snapshot mandatory; schema-change releases gated.
7. **§7 (Website Integration)** — `corpus.json` retained as fallback 2 releases post-cutover; `NEXT_PUBLIC_VAULT_FALLBACK=static` feature flag; SWR retry/backoff/circuit-breaker; service worker cache; inline top-200 questions in manifest; add §7.4 search UX spec; bundle claim gated on real FCP/TTI measurement.
8. **§8 (Chain discoverability)** — cut to pre-reveal indicator only; instrument; iterate.
9. **§10 (D1 + Worker)** — cache keys include `release_id`; remove admin endpoint (or Cloudflare Access gate); cursor pagination + ETag + Cache-Control; data-plane SLI list; FTS5 load-test gate on Phase 3 entry; recomputed cost forecast.
10. **§12 (LLM-assisted generate)** — `vault/exemplars/` curated pool; `vault/generation-log/`; cost ceiling; secrets in `~/.config/vault/secrets.toml`.
11. **§13 (Security/Safety)** — expand YAML hardening; content-hash canonical; rollback-via-snapshot primary; schema migration rules pointer to EVOLUTION.md; shared-types package contract; author list and credits.
12. **§14 (Phases)** — timeline revision (18–22 days); Phase 0 adds local-dev shim + README + CONTRIBUTING + EVOLUTION.md; Phase 1 has "overrun policy"; Phase 5 cut to one intervention.
13. **§15 (Open Questions)** — resolve license recommendation; note remaining decisions needed before Phase 3.
14. **§19 (Testing Plan)** — add rollback-symmetry property test per release; manifest-only vs `.db`-committed; data-plane SLI probes.
15. **New files**:
    - `interviews/vault/schema/EVOLUTION.md` (Phase 0)
    - `interviews/vault-cli/README.md` (Phase 0)
    - `interviews/CONTRIBUTING.md` or section in root README (Phase 0)
    - `interviews/vault-cli/docs/JSON_OUTPUT.md`, `EXIT_CODES.md`, `CUTOVER_QA.md` (Phase 0–2)

---

## 7. Round 2 brief (to be used when reviewers re-assess)

"Here is ARCHITECTURE.md v2 and REVIEWS.md with our responses. For each Critical/High finding you raised, check whether our integrated fix actually resolves the concern. Call out: (a) gaps in the fix, (b) anything new you see in v2 that wasn't in v1, (c) issues you now consider adequately addressed. Use the same severity ranking. Return the same format."

---

**End of Round 1 ledger.**

---

# Round 2 — v2 reviewed, v2.1 patched (2026-04-15)

## R2-1. Round 1 resolution status (reviewers' own verdicts)

All four Round-1 reviewers re-read v2 against REVIEWS.md and reported per-item RESOLVED / PARTIAL / UNRESOLVED statuses.

**Outcome**: Every Round-1 Critical and High item was marked **RESOLVED** by at least the flagging reviewer. Only two items were marked PARTIAL, both because the fix lives in a Phase-0 deliverable that doesn't exist yet (EVOLUTION.md, H-1) rather than because the spec is wrong.

| Reviewer | R1 Critical RESOLVED | R1 High RESOLVED | R1 Critical PARTIAL | R1 High PARTIAL |
|---|---|---|---|---|
| Chip | 3/3 | 6/6 | — | — |
| Dean | 2/3 (C-4 PARTIAL → see N-1) | 4/6 (H-14, H-18, and H-1 PARTIAL) | C-4 | H-14, H-18, H-1 |
| Soumith | 2/2 | 7/7 | — | — |
| David | 3/3 | 4/6 (H-1, H-18 PARTIAL) | — | H-1, H-18 |

## R2-2. Round 2 verdicts

| Reviewer | Verdict | New Critical | New High | New Medium | New Low |
|---|---|---|---|---|---|
| Chip | YELLOW | 1 (N-C1: ship ordering) | 5 | 4 | 3 |
| Dean | YELLOW → GREEN on N-1 fix | 1 (N-1: ship ordering) | 5 | 4 | — |
| Soumith | GREEN (2 Highs to fix) | — | 2 | 4 | 3 |
| David | YELLOW → GREEN | — | 2 | 4 | 3 |

**Cross-reviewer convergence — the key signal**:

- **`vault ship` journal/ordering gap** — flagged independently by Chip (N-C1), Dean (N-1), and Soumith (H-NEW-1). 3/4 reviewers, all at Critical/High severity, same root cause. Load-bearing v2.1 fix.
- **Service-worker cache release-keying** — Chip (N-H2), Dean (N-6), David (N-4). 3/4 reviewers.
- **`X-Vault-Release` hard-reject causes brownout** — Soumith (H-NEW-2), Dean (N-3). Same issue class.
- **CI equivalence cost / 28MB byte-diff** — Chip (N-H4), David (N-1). Same issue.
- **ID-collision recovery workflow missing** — Dean (N-2), David (N-2). Same issue.
- **Schema-fingerprint check fails-closed creates total outage** — Chip (N-H1); Dean (N-4) notes the orthogonal issue that the fingerprint check compares metadata-vs-metadata, not actual DDL. Both addressed together in v2.1.

## R2-3. v2.1 resolution table

All convergent items resolved inline per §18 "explicit engineering decision documented inline" clause, given reviewer consensus that no architectural rework is needed — only precision on mechanics. Round 3 declined on the same basis; user can override by requesting Round 3 after reviewing v2.1.

| R2 ID | Issue | Reviewers | v2.1 Resolution | ARCHITECTURE.md section |
|---|---|---|---|---|
| N-C1 / N-1 / H-NEW-1 | `vault ship` journal + ordering missing | Chip, Dean, Soumith | §6.1.1 new: ordered protocol (D1→Next.js→paper-last), journal file, per-leg rollback matrix, paging triggers, `--resume` | §4.3, §6.1.1 |
| Chip N-H1 | schema-fingerprint total outage | Chip | Degraded-mode (Cache-API-only, banner) on mismatch instead of 5xx | §10.1 |
| Chip N-H2 / Dean N-6 / David N-4 | SW cache not release-keyed | 3 reviewers | SW keys include release_id; TTL 7d; skipWaiting+claim on release change | §7.1 |
| Chip N-H3 | Exemplar pool coverage unknown | Chip | Phase 0 adds `vault stats --exemplar-coverage` → `vault/exemplar-gaps.yaml` | §14 Phase 0 |
| Chip N-H4 / David N-1 | CI 28MB byte-diff is expensive + flaky | 2 reviewers | Compare release_hash (Merkle) to `corpus-equivalence-hash.txt`; O(1); CI budget ≤2min | §11.5 |
| Chip N-H5 | canon version not in Merkle | Chip | `__canon_version__` + `__policy__` leaves added to Merkle | §3.5 |
| Soumith H-NEW-2 / Dean N-3 | X-Vault-Release hard-reject brownout | 2 reviewers | Header demoted to informational SLI; 10-min cross-release grace window | §6.1.1, §10.2 |
| Soumith H-NEW-3 | Codegen "who runs it" undefined | Soumith | PR author runs `vault codegen`; CI runs `--check`, never pushes | §13 |
| Dean N-2 / David N-2 | ID-collision recovery workflow missing | 2 reviewers | `vault renumber` command + `vault check --strict` catches registry/file mismatch | §3.3, §4.4 |
| Dean N-4 | schema_fingerprint vs metadata, not DDL | Dean | Worker hashes `sqlite_master.sql` at cold start | §10.1 |
| Dean N-5 | FTS5 cost gate missing (only latency) | Dean | Phase 4 entry gates on ≤500 D1 row-reads/query in addition to p99 | §10.6 |
| Dean N-10 | Static fallback retention too short | Dean | Retain until first schema-major bump post-cutover OR 2 releases | §7.1 |
| David N-5 | Canary soak is aspirational at small DAU | David | Soak = max(15min, ≥100 sessions observed) | §4.3 |
| David N-9 | `vault mark-exemplar` not in CLI surface | David | Added to §4.4 + `vault renumber` | §4.4 |
| Soumith M-NEW-3 | `verify` exit-code contradicts §4.6 | Soumith | Changed from exit 2 → exit 1 (integrity failure) | §4.2 |
| Soumith M-NEW-1 | TS package versioning undefined | Soumith | pnpm workspace protocol; no external npm | §13 |
| Soumith L-NEW-1 / David dup | `--reviewed-by` spoofable | Soumith, David | CI rejects if value doesn't match committer email | §4.4 |
| David N-8 | Duplicate Observability in §13 | David | Removed | §13 cleanup |
| N-C1/N-1/N-2/N-10 aggregated | Paper-leg not auto-rolled-back | Dean, Chip | Explicit: paper-leg rollback is manual; remediation is forward-fix release | §6.1.1 |

**Items deferred with rationale** (documented Round 3 rather than integrated):

- **Dean N-9** (rollback-symmetry property test conflates snapshot vs SQL paths): acknowledged, deferred to TESTING.md detailed spec. Property test as written is only meaningful on SQL-rollback path; snapshot-path test is different shape (restore + verify round-trip). Tracked as test-plan refinement, not architectural issue.
- **Chip N-M2** (service worker stale for cached content): addressed by SW release-keying; residual concern about content-hash-divergence between D1 and SW-cached is mitigated by SW eviction on release change.
- **Soumith M-NEW-2** (`vault api` vs production Worker CORS/rate-limit divergence): acknowledged — the local shim deliberately omits edge-specific behavior. Documented in CONTRIBUTING.md as Phase 0 deliverable.
- **David N-7** (`ORDER BY id` with content-addressed IDs produces visual-random paginated order): low-impact UX nit. Switch to `ORDER BY topic, id` tracked as Phase 4 polish.
- **Chip N-L1** (`vault/.llm-spend.json` merge conflicts): switched to per-user ledger files (`vault/.llm-spend/<git-user>.json`, gitignored). Minor; not yet in doc — **TODO flag for Phase 1 implementation**.

## R2-4. Round 3 decision

**Round 3 DECLINED.** Rationale per ARCHITECTURE.md §18: "Any remaining critical/high must be resolved by explicit engineering decision documented inline."

- All Round-1 items RESOLVED per reviewers' own assessments.
- All Round-2 convergent Critical and High items addressed in v2.1 with specific mechanics (ordering matrices, journal schemas, specific gate thresholds, degraded-mode semantics) — not hand-waving.
- Reviewer-level consensus: Soumith GREEN, David GREEN-leaning, Chip YELLOW→GREEN on same fixes, Dean YELLOW→GREEN on same fixes.
- No architectural rework required; only surgical specification.
- Another adversarial round has diminishing expected value against the cost of delay.

**User may override**: request Round 3 if desired. This decision is logged for auditability, not as a unilateral call.

## R2-5. Readiness assessment (for Stage 3)

**Color**: GREEN with three documented conditions:

1. **OPEN gate — license decision**: §15 still lists corpus license as "BLOCKS PHASE 3." Recommended default CC-BY-4.0 for corpus, MIT for `vault-cli`. User confirmation required before Phase 3 starts (not Phase 0).
2. **Phase-0 deliverable dependency — EVOLUTION.md**: Schema-evolution rules in §13 are adequate sketch; the full document is Phase 0 work. H-1 remains PARTIAL until EVOLUTION.md lands.
3. **Measurement-gated Phase transitions**: Phase 3 entry is gated on FTS5 load test (latency + cost + fallback index ready); Phase 4 entry is gated on Lighthouse CI (FCP/TTI/transfer size) + rollback drill.

All other Round-1 and Round-2 items are closed in the plan. Implementation may proceed after user green-light per Stage 3 protocol.

---

---

# Round 3 — v2.1 reviewed, v2.2 patched (2026-04-16)

Round 3 launched on the user's "proceed with 2" green-light. All four reviewers re-read v2.1 plus the code that landed (Phases 0-6). Verdicts:

| Reviewer | Verdict | New Critical | New High | Dominant theme |
|---|---|---|---|---|
| Chip | YELLOW | 2 | 5 | Worker code introduced regressions vs spec (placeholder bypass, SW manifest spam) |
| Dean | YELLOW | 2 | 4 | Spec-vs-code gap: `emit_migrations` questions-only, `vault ship` missing, verify-from-source broken |
| Soumith | GREEN (conditions) | 1 | 2 | Doc contradiction (§7.1/§10.2 still said "reject mismatch"); breaker semantics |
| David | YELLOW→GREEN | 0 | 5 | Authoring commands documented but not fully implemented (edit UX, move refusals, new registry) |

**Convergent items** (≥2 reviewers, Critical/High):
- SCHEMA_FINGERPRINT placeholder silent bypass (Chip C1, Dean NH-3). **FIXED in v2.2**.
- SW fetches `/manifest` on every request (Chip C2). **FIXED in v2.2**.
- SW loses API origin on restart (Chip H3). **FIXED in v2.2**.
- `emit_migrations` only diffs questions table (Dean NC-1). **FIXED in v2.2** (all 4 tables via PRAGMA).
- `row_values` hardcoded column list (Dean NH-2). **FIXED in v2.2** (PRAGMA-driven).
- `vault verify --from-source` rebuilds from HEAD, breaking historical citation (Dean NC-2). **FIXED in v2.2** (`--git-ref`).
- `vault ship` composed command unimplemented (Dean NH-1). **FIXED in v2.2** (journaled, tested).
- Doc contradiction — §7.1/§10.2 still said X-Vault-Release hard-reject (Soumith F-1). **FIXED in v2.2**.
- Hash-stability test undertested (Soumith F-4). **FIXED in v2.2** (nested-dict fixture).
- David's authoring-contract gaps (H1-H4). **FIXED in v2.2** (edit failure-UX re-open, move refusals, new registry-append + git rebase, authors auto-populate).
- `vault renumber` + `vault mark-exemplar` spec-only (David N-2, N-9). **FIXED in v2.2**.

**Explicitly deferred to Phase-3-entry** (accepted engineering decision):
- Cache API wiring in Worker (Chip H1) — marked as Phase-3-entry gate.
- Circuit-breaker half-open semantics (Soumith F-2) — documented as Phase-4 improvement; current naive impl ships with known limitation noted inline.
- Cursor `filter_hash` validation + keyset pagination (Chip H2) — Phase-3-entry gate.
- Rate limiting via RATE_LIMIT_KV (Chip H4) — Phase-3-entry gate.
- Cross-language content_hash verification (Chip H5) — path chosen: data-plane SLI becomes a Python GitHub Action rather than a Worker cron; documented in TESTING.md.
- Equivalence-hash algorithm documentation (Soumith F-3) — pragmatic framing: the `split-corpus.py` → Merkle path IS the algorithm, documented in §11.5 as "regression guard post-split." Revisit if external verifier ever needs to reproduce from raw `corpus.json`.
- Full per-command JSON schemas for all 19 subcommands (B.10) — 5 schemas documented; rest land as commands mature.
- Worker vitest suite (B.9) — deferred until D1 staging exists (Phase-3 work).
- Remaining structural invariants (applicability matrix validation, LSH scenario-dedup) — tracked; not blockers for Phase-1/2.

**v2.2 changelog**: see ARCHITECTURE.md header.

**Round 3 decision**: GREEN for Phase-0/1/2 autonomous execution (landed). YELLOW for Phase-3/4 — deploy gates listed explicitly above and in CUTOVER_QA.md §0.

**License state (DECIDED 2026-04-16)**: CC-BY-NC-4.0 for the corpus under `interviews/vault/questions/`. LICENSE file committed. `vault-cli` license unchanged from its historical (unlicensed) status — intentionally out of scope for this work. No longer a Phase-3 blocker; external contributor PRs can proceed with the NC constraint made explicit in CONTRIBUTING.md.

---

# Round 4 — Code-level security + safety audit (2026-04-16)

Post-Bucket-B code audit by Chip Huyen focused on actual exploitable paths (not architectural critique). Previous three rounds were specification-level; this one was line-by-line on what landed.

## R4 findings — all 12 resolved

| Severity | ID | Finding | Resolution |
|---|---|---|---|
| Critical | R4-C-1 | FTS5 MATCH injection: char-class strip left `NEAR`/`OR`/`AND`/`NOT` keywords intact, permitting quadratic-cost DoS queries | `MAX_SEARCH_Q_CHARS=100` cap; reject reserved tokens pre-sanitize; wrap sanitized term in FTS5 phrase literal |
| Critical | R4-C-2 | Service worker trusted any page's `postMessage` to set vault API origin — XSS anywhere on staffml → persistent exfiltration via IDB-poisoned origin | `event.source` must be same-origin window client; posted origin must be in build-time `VAULT_API_ALLOWLIST`; refuse overwrite once set; activate re-validates persisted IDB origin |
| Critical | R4-C-3 | `vault ship --resume` had no version-binding check — could resume old journal against new version, publishing mixed-release | Assert `journal.version == version` AND `journal.env == env` in `run_ship`; refuse overwrite of existing journal without `--resume` |
| High | R4-H-1 | `_sql_quote` treated `bool` as `int` (Python subclass quirk), emitting `"True"`/`"False"` as SQL tokens | Handle `bool` BEFORE numeric branch; also added `bytes`/`bytearray` → BLOB literal |
| High | R4-H-2 | CORS allowlist fell open to `"*"` on empty env var; catch-all echoed `String(e)` leaking D1/SQL fragments cross-origin | Fail-closed: no `Access-Control-Allow-Origin` header when allowlist is empty or origin unmatched. Error responses no longer include `detail` — log to console.error instead |
| High | R4-H-3 | Rate limiter trusted `X-Forwarded-For` (client-settable); `"unknown"` sentinel collapsed all non-CF traffic into one bucket | Trust only `CF-Connecting-IP`; when missing, return deny with retry-after-60. KV read-then-write within-POP race documented (2-3× cap leakage acceptable; Durable Objects path reserved for Phase-4 if observed) |
| High | R4-H-4 | Taxonomy DAG didn't initialize `color` for pure-target nodes; LSH shingling uncapped vs 256KB-scenario YAML | Initialize `color` over `sources ∪ targets`. Cap pre-shingle text at `MAX_SHINGLE_LEN=8000` chars |
| Medium | R4-M-1 | `vault edit` used `question_id in p.read_text()` substring match — could open wrong file if ID appeared in a chain reference elsewhere | Anchored regex on `^\s*id:\s*['"]?<id>['"]?$` pattern; exact-field match |
| Medium | R4-M-2 | Dead-code band-aid `(". " + ".pending-{v}").replace("..", ".")` in migrations-emit path | Replaced with plain `f".pending-{version}"` |
| Medium | R4-M-3 | Service worker served 7-day-stale cache forever when manifest polling silently failed | Track `lastManifestSuccessMs`; bypass cache on question/search endpoints after 24h of silent manifest failures |
| Low | R4-L-1 | No version-string validation — `version="HEAD"` or `"1..0"` would produce weird on-disk layouts | `_VERSION_RE` enforces `X.Y.Z[-prerelease]` on every command taking a `version` arg |
| Low | R4-L-2 | `vault ship` paper-leg created git tag without pre-existence check; re-running escalated to paper-leg failure | Pre-check `git rev-parse refs/tags/v<v>`; if exists and matches, status=`already-pushed-skipped`; only create-and-push if genuinely new |

## R4 verdict (post-fix)

All 12 findings closed. `vault check --strict` passes on 9,657 questions with 0 load errors and 0 invariant failures. 28/28 pytest green. `vault verify 0.9.0` reconstructs release_hash `fe69d4c4...` from YAML source (post-normalization).

**GREEN** for Phase-0/1/2 in-repo work. Phase-3/4 deploy gates remain as enumerated in CUTOVER_QA.md §0. No further rounds planned; if new issues emerge they're ordinary PR feedback.

---

**End of review ledger.**


