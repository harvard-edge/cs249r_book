# Deprecated scripts — `interviews/vault/scripts/`

Many scripts in this directory pre-date the YAML-as-source-of-truth
migration (ARCHITECTURE.md v2.x, Phase 1). YAML at `../questions/**/*.yaml`
is now authoritative; the legacy scripts ran against the monolithic
`../corpus.json`, which itself is now a generated artifact (emitted by
`vault build --local-json`).

**Do not run remaining scripts in this directory without understanding
what they were for.** New contributors should reach for the `vault` CLI
instead.

## Removed in 2026-05

The following 18 scripts were deleted as unambiguously dead. The mapping
is preserved here for git-archaeology — find them via `git log --diff-filter=D`
if you need to read the historical implementation.

| Removed script | Purpose (pre-migration) | Replacement |
|---|---|---|
| `build_corpus.py` | Assembled corpus.json from track/zone data | `vault build` (walks YAML, emits vault.db) |
| `export_to_staffml.py` | Copied corpus.json → `staffml/src/data/` with field massaging | `vault build --local-json` (writes site-compatible JSON) |
| `extract_taxonomy.py` | Extracted topic graph from corpus.json | `vault/taxonomy.yaml` is the source now; see `vault/schema/EVOLUTION.md` |
| `deep_verify.py`, `gemini_*.py`, `gpt_*.py` | Legacy LLM-verification + backfill one-shots, ran against corpus.json | LLM authoring flow is now Phase-7 `vault generate`; LLM audit flow lives in `vault-cli/scripts/audit_chains_with_gemini.py` (batched) and `validate_drafts.py` (per-draft gates) |
| `gate.py`, `archive/expand_tracks.py`, `archive/fill_zone_gaps.py`, `archive/fill_gaps.sh`, `archive/final_balance.sh` | Pre-launch corpus-building one-shots | Obsolete after schema v1.0 (taxonomy lives in YAML, not in the path); YAML files authored directly via `vault new` |
| `generate.py` | Original LLM generator | Phase-7 `vault generate --topic X --zone Y --level Lz --count N` |

## Commands that are live today

```bash
vault build --local-json                        # regenerate corpus.json
vault publish <version>                          # end-to-end release
vault export-paper <version>                     # paper macros + stats
vault verify <version>                           # academic-citability check
vault check --strict                             # 26 invariants
```

See `../../vault-cli/README.md` for the full 22-subcommand reference.
