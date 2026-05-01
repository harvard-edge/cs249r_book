# Deprecated scripts — `interviews/vault/scripts/`

Everything in this directory pre-dates the YAML-as-source-of-truth migration
(ARCHITECTURE.md v2.x, Phase 1). YAML at `../questions/**/*.yaml` is now
authoritative; these scripts ran against the monolithic `../corpus.json`,
which itself is now a generated artifact (emitted by `vault build --local-json`).

**Do not run these scripts without understanding what they were for.**
They are kept for git-history legibility and one-shot archaeology; new
contributors should reach for the `vault` CLI instead.

## Replaced-by map

| Legacy script | Purpose (pre-migration) | Replacement |
|---|---|---|
| `build_corpus.py` | Assembled corpus.json from track/zone data | `vault build` (walks YAML, emits vault.db) |
| `export_to_staffml.py` | Copied corpus.json → `staffml/src/data/` with field massaging | `vault build --local-json` (writes site-compatible JSON) |
| `extract_taxonomy.py` | Extracted topic graph from corpus.json | `vault/taxonomy.yaml` is the source now; see `vault/schema/EVOLUTION.md` |
| `validate_specs.py` | Validated hardware constants against MLSysIM | still relevant; runs in nightly CI |
| `deep_verify.py`, `gemini_*` | Legacy LLM-verification one-shots | `vault check --tier slow` incorporates the nightly math check; LLM flow is Phase-7 `vault generate` |
| `gate.py`, `expand_tracks.py`, `fill_*.py`, `final_balance.sh` | Pre-launch corpus-building one-shots | Obsolete; YAML files authored directly via `vault new` |
| `generate.py` (if present) | Original LLM generator | Phase-7 `vault generate --topic X --zone Y --level Lz --count N` |

## Commands that are live today

```bash
vault build --local-json                        # regenerate corpus.json
vault publish <version>                          # end-to-end release
vault export-paper <version>                     # paper macros + stats
vault verify <version>                           # academic-citability check
vault check --strict                             # 26 invariants
```

See `../../vault-cli/README.md` for the full 22-subcommand reference.
