# `vault-cli` — StaffML Question Vault CLI

Authoring, building, and releasing the StaffML question vault.

> **Status**: v1.1 — sidecar chain architecture + tier-aware UI in place.
> Chain corpus growth tracked in
> [`docs/CHAIN_ROADMAP.md`](docs/CHAIN_ROADMAP.md); design baseline is
> [`../vault/ARCHITECTURE.md`](../vault/ARCHITECTURE.md) (§3.6 captures
> the v1.1 deltas).

## Install (local editable)

```bash
# from the monorepo root
pip install -e interviews/vault-cli/
vault --version
```

Python ≥3.12 required. CI pins 3.12 exactly for hash stability (see
[`docs/EXIT_CODES.md`](docs/EXIT_CODES.md) and ARCHITECTURE.md §3.5).

## Quickstart — the 22 subcommands live today

Authoring:

```bash
vault new --track cloud --level l4 --zone diagnosis \
          --topic kv-cache --title "..."        # content-addressed ID + registry append + authors
vault edit <id>                                   # $EDITOR; failure injects comment block, re-opens
vault move <id> --to <track>/<level>/<zone>       # dirty-tree / chain / applicability refusals
vault rm <id> [--hard]                            # soft (deprecate) by default; --hard needs typed confirm
vault restore <id>                                # undo soft-delete
vault renumber <id>                               # recover from post-rebase dedup-seq collision
vault mark-exemplar <id>                          # promote to human-only exemplar pool
```

Build + check:

```bash
vault build                                       # YAML → vault.db
vault check --strict                              # fast + structural tiers
vault check --tier slow                           # nightly LSH scenario-dedup
vault stats [--format-prometheus|--exemplar-coverage]
vault codegen --check                             # shared-types drift guard
vault doctor [--check <name>]                     # 8 diagnostic subchecks
```

Release pipeline:

```bash
vault snapshot 1.0.0                              # stage to releases/.pending-1.0.0/
vault migrations-emit 0.9.0 1.0.0                 # forward + inverse SQL (all 4 tables)
vault export-paper 1.0.0                          # SQL → macros.tex + corpus_stats.json
vault tag 1.0.0                                   # git commit + tag
vault publish 1.0.0 [--resume|--sign]             # composed product; atomic POSIX rename
vault verify 1.0.0 [--git-ref v1.0.0]             # academic-citability round-trip
vault diff 0.9.0 1.0.0 --classify                 # cosmetic|semantic|structural
vault deploy 1.0.0 --env staging                  # D1 migration + snapshot + POP probe
vault rollback <v> --env production [--method snapshot|sql]
vault ship 1.0.0 --env production [--resume] [--skip-legs paper]
vault promote <id> | --all-drafts                 # drafts → published with provenance bump
```

Local dev:

```bash
vault api --db <path>.db --port 8002              # mirror Worker endpoint surface from local vault.db
vault serve                                       # Datasette over vault.db (127.0.0.1 only)
```

All commands support `--json` for machine-readable output per
[`docs/JSON_OUTPUT.md`](docs/JSON_OUTPUT.md). Exit codes are stable per
[`docs/EXIT_CODES.md`](docs/EXIT_CODES.md).

## Chain build pipeline (v1.1+)

Chains are pedagogical progressions through Bloom levels (L1→L6+) within
one (track, topic) bucket. `interviews/vault/chains.json` is the
authoritative registry; YAMLs no longer carry a `chains:` field. The
build tooling lives in `scripts/`:

```bash
# 1. Surface (track, topic) buckets that need chains. Writes
#    interviews/vault/chain-coverage.json (gitignored — regeneratable).
python3 scripts/diagnose_chain_coverage.py

# 2. Strict pass: Δ ∈ {1, 2}, primary chains. Default mode.
python3 scripts/build_chains_with_gemini.py --all \
  --output ../vault/chains.proposed.json

# 3. Lenient pass: Δ ∈ {0, 1, 2, 3}, secondary chains.
#    Use --buckets-from to scope the run to uncovered buckets only.
python3 scripts/build_chains_with_gemini.py --mode lenient \
  --buckets-from ../vault/chain-coverage.json \
  --output ../vault/chains.proposed.lenient.json

# 4. Apply a single proposed file (replaces chains.json after validation).
python3 scripts/apply_proposed_chains.py --proposed ../vault/chains.proposed.json

# 5. Merge primary + secondary into chains.json with cap enforcement
#    (each qid in ≤ 2 chains; non-L1/L2 qids capped at 1 membership).
python3 scripts/merge_chain_passes.py
```

Both `apply_proposed_chains.py` and the validator tolerate a missing
`tier` field on chain entries (defaulting to "primary"); chains
produced by `--mode lenient` are tagged `tier: "secondary"`. After any
change, run `vault check --strict` and `vault build --local-json`.

## Run tests

```bash
pip install -e interviews/vault-cli/[dev]
pytest interviews/vault-cli/tests/
```

## Layout

```
vault-cli/
├── pyproject.toml
├── README.md                          # this file
├── docs/
│   ├── CHAIN_ROADMAP.md               # resumable chain-coverage workstream
│   ├── EXIT_CODES.md                  # stable exit-code taxonomy
│   ├── JSON_OUTPUT.md                 # per-command --json schemas
│   └── CUTOVER_QA.md                  # manual cutover QA checklist
├── src/vault_cli/                     # Typer app + library
│   ├── __init__.py
│   ├── compiler.py / loader.py / yaml_io.py
│   ├── legacy_export.py               # corpus.json + chain_tiers emitter
│   ├── policy.py                      # release-policy filter
│   ├── validator.py                   # fast / structural / slow tiers
│   └── main.py                        # Typer app entry
├── scripts/                           # ops + Gemini-powered tools
│   ├── diagnose_chain_coverage.py     # surface uncovered buckets
│   ├── build_chains_with_gemini.py    # --mode {strict,lenient}
│   ├── apply_proposed_chains.py       # gate proposed chains.json
│   ├── merge_chain_passes.py          # primary + secondary, cap-enforced
│   ├── summarize_proposed_chains.py   # quick-read review
│   └── ...                            # auditing, calibration, D1 emit, etc.
└── tests/                             # pytest suite (74 tests today)
```

## Architecture

See the sibling [`vault/`](../vault/) directory:

- [`ARCHITECTURE.md`](../vault/ARCHITECTURE.md) — full design doc.
- [`REVIEWS.md`](../vault/REVIEWS.md) — adversarial review ledger.
- [`TESTING.md`](../vault/TESTING.md) — test plan.
- [`schema/EVOLUTION.md`](../vault/schema/EVOLUTION.md) — schema-version rules.

## Contributing

See [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

## License

MIT — see project root.
