# StaffML Vault

The authoritative home of the StaffML question corpus: the per-question YAML
source, the taxonomy, release artifacts, schema, and the review ledger that
got us here.

> **Corpus license**: [CC-BY-NC-4.0](questions/LICENSE) — non-commercial use
> only with attribution. Commercial licensing requires written permission.
> See [`../CONTRIBUTING.md`](../CONTRIBUTING.md) for details.

## What lives here

```
interviews/vault/
├── ARCHITECTURE.md          ← v2.2 design doc (1600+ lines, definitive)
├── REVIEWS.md               ← 3-round adversarial review ledger
├── TESTING.md               ← test plan; see vault-cli for impl
├── README.md                ← this file
├── release-policy.yaml      ← SINGLE filter predicate for "what is published"
├── corpus-equivalence-hash.txt ← CI regression guard (release_hash pin)
├── schema/
│   ├── question_schema.yaml ← LinkML — sole schema source of truth
│   └── EVOLUTION.md         ← SemVer rules for schema versioning
├── questions/               ← THE SOURCE OF TRUTH — 9,657 YAML files
│   ├── LICENSE              ← CC-BY-NC-4.0
│   ├── cloud/{l1..l6}/<zone>/*.yaml
│   ├── edge/
│   ├── mobile/
│   ├── tinyml/
│   └── global/
├── exemplars/               ← curated human-only pool for `vault generate`
├── drafts/                  ← LLM-generated, awaiting `vault promote`
├── releases/                ← citable release artifacts
│   ├── 0.9.0/
│   │   ├── vault.db         ← compiled SQLite (committed, citable)
│   │   ├── release.json     ← release_hash, counts, git_sha, timestamp
│   │   ├── d1-migration.sql ← forward migration
│   │   └── d1-rollback.sql  ← inverse with prior-row bodies embedded
│   └── latest -> 0.9.0      ← POSIX atomic-rename symlink
├── id-registry.yaml         ← APPEND-ONLY log; CI rejects line deletions
├── taxonomy.yaml            ← topic graph; DAG-enforced
├── chains.yaml              ← chain definitions (optional)
├── zones.yaml               ← 8 ikigai zones
└── exemplar-gaps.yaml       ← Phase-0 coverage audit output
```

### Pipeline artifacts: `_pipeline/` (gitignored)

LLM-driven tooling (chain proposals, gap detection, draft scorecards,
audit traces) writes intermediate outputs to `_pipeline/` — the
directory is gitignored as a unit. Only durable corpus artifacts
(chains.json, id-registry.yaml, questions/) belong in git; pipeline
runs are reproducible from the live tooling on demand and would
otherwise pollute history with byte-stable LLM noise.

```
interviews/vault/_pipeline/                  ← gitignored
├── chains.proposed.json                      ← build_chains_with_gemini.py
├── chains.proposed.lenient.json              ← build_chains_with_gemini.py --mode lenient
├── gaps.proposed.json                        ← gap-detection sidecar (strict)
├── gaps.proposed.lenient.json                ← gap-detection sidecar (lenient)
├── draft-validation-scorecard.json           ← validate_drafts.py output
└── runs/
    ├── AUDIT_REPORT.md                       ← latest audit_chains_with_gemini.py rollup
    └── <UTC-timestamp>/                      ← per-run audit traces
```

When adding a new pipeline tool, route default outputs through the
`PIPELINE_DIR` constant (`vault/_pipeline/`) and never commit anything
under it.

## Quick commands

```bash
# From repo root, with venv activated and `pip install -e interviews/vault-cli/[dev]`:
vault --help                      # 22 subcommands, phase-aware
vault build                       # compile YAML → vault.db
vault check --strict              # fast + structural invariants (<60s)
vault check --tier slow           # nightly tier incl. LSH scenario dedup
vault stats                       # scorecard over latest vault.db
vault doctor                      # 8 diagnostic subchecks
vault verify 0.9.0                # academic-citability round-trip
vault api --port 8002             # local Worker-surface shim for dev
```

Full CLI reference in [`../vault-cli/README.md`](../vault-cli/README.md).

## Flow — YAML → paper, YAML → site

```
                      vault/questions/*.yaml
                              │
                              ▼
                     `vault build` ─────────────► vault.db
                              │                      │
                   `vault check --strict`            │
                              │               ┌──────┴───────┐
                              ▼               ▼              ▼
                       (CI green or fail)   paper         D1 + Worker
                                            macros.tex    edge API
                                            via `vault    via `vault
                                            export-paper` deploy`
                                                           │
                                                           ▼
                                                   staffml site
                                                   (reads via vault-api.ts)
```

Every output is a function of the YAML source, versioned by `release_hash`
(SHA-256 Merkle over content_hashes + taxonomy + chains + zones + policy +
canon_version leaves; ARCHITECTURE.md §3.5). Paper and site agree by
construction — they read from the same `vault.db`.

## How to add a question

```bash
vault new --track cloud --level l4 --zone diagnosis \
          --topic kv-cache-management \
          --title "KV Cache Bandwidth Bottleneck on H100"
# Opens $EDITOR with a scaffolded YAML.
# On save: schema validates; failures inject comment block at top and re-open.

vault check --strict                    # make sure it passes invariants
git add interviews/vault/questions/...  # explicit add, not git add -A
git commit -m "question: KV cache bandwidth"
```

On `vault new`:
- A content-addressed ID is allocated (`<track>-<level>-<zone>-<topic>-<6hex>-<seq>`).
- `id-registry.yaml` is appended with `{id, created_at, created_by}`.
- `authors:` is auto-populated from `git config user.email`.
- A `git pull --rebase` runs first to reduce collision rate.

See [`../CONTRIBUTING.md`](../CONTRIBUTING.md) for the full contribution
workflow, including post-rebase collision recovery via `vault renumber`.

## Release workflow (composed)

```bash
vault publish 1.0.0                              # build + snapshot + migrations
vault verify 1.0.0 --git-ref v1.0.0              # citation-grade round-trip
vault diff 0.9.0 1.0.0 --classify                # review changes pre-ship
vault ship 1.0.0 --env staging                   # D1 → Next.js → paper-last
# Soak on staging, then:
vault ship 1.0.0 --env production --canary-percent 10
```

## Invariants (26 total, tiered)

- **Fast** (pre-commit, <1s): schema validation, unique IDs, lowercase paths,
  enum-valid path components, filename format, URL scheme allowlist, no raw
  HTML in scenarios, YAML DoS hardening (256KB cap, depth 10, no aliases).
- **Structural** (CI, <30s parallelized): topic-in-taxonomy, chain-ref exists,
  contiguous chain positions, provenance metadata consistency, registry
  append-only, taxonomy DAG, applicability matrix, release-policy single-source.
- **LSH scenario-dedup** (CI, <10s, LSH-blocked): MinHash + 16-band LSH +
  Jaro-Winkler within-bucket candidate pairs at 0.95 threshold.
- **Nightly**: deep-dive URL reachability, napkin-math dimensional analysis
  (Pint), LLM math verification.
- **Weekly**: secret-leak grep.

Full specification: ARCHITECTURE.md §5.

## Three-round adversarial review

This architecture survived **three full rounds of adversarial review** with
four reviewers each (Chip Huyen, Jeff Dean, Soumith Chintala, and an industry
ML engineer "David"). Every convergent finding was resolved in spec + code
with inline engineering decisions. See [`REVIEWS.md`](REVIEWS.md) for the
finding-by-finding ledger and the deferred-items list with rationale.

## License

The corpus under this directory (`vault/questions/`, taxonomy, chains, zones,
release artifacts) is licensed **CC-BY-NC-4.0**. See
[`questions/LICENSE`](questions/LICENSE). The `vault-cli` Python tooling
under `../vault-cli/` is a separate artifact and is not covered by this
LICENSE.
