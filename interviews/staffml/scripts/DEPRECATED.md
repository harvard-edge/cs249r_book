# Deprecated scripts — `interviews/staffml/scripts/`

These pre-date the YAML migration (ARCHITECTURE.md v2.x, Phase 1). They ran
against the monolithic `interviews/vault/corpus.json` (now a generated
artifact) or pushed data into `src/data/corpus.json` (now emitted by
`vault build --legacy-json`).

## Replaced-by map

| Legacy script | Purpose | Replacement |
|---|---|---|
| `sync-vault.py` | Copied vault/corpus.json → src/data/ with filter | `vault build --legacy-json` emits site-compatible JSON directly |
| `generate-manifest.py` | Built src/data/vault-manifest.json | Built by `vault publish` as a release artifact |
| `validate-vault.py` | Sanity check on corpus shape | Covered by `vault check --strict` invariants |
| `format-napkin-math.py` | One-shot formatter | Obsolete |
| `sync-periodic-table.mjs` | Unrelated (periodic-table site feature) | Still active — NOT deprecated |

## Current flow

```bash
vault build --legacy-json                     # from repo root
# Regenerates:
#   interviews/staffml/src/data/corpus.json   (9199 questions, site-compatible shape)
#   interviews/vault/vault.db                 (25 MB SQLite build artifact)
# Verifies release_hash against corpus-equivalence-hash.txt
```

The site layout has NOT changed: `corpus.ts` still does
`import corpusData from '../data/corpus.json'`. The only difference is that
`corpus.json` is now derived from YAML rather than hand-edited — a
pre-commit hook refuses direct edits to it.

Phase-4 cutover replaces the bundled JSON with Worker-API reads via
`corpus-source.ts` + `vault-api.ts`. That's a separate step;
`corpus.json` stays through at least 2 post-cutover releases as the
rollback fallback.
