# StaffML — local development

The StaffML web app (Next.js 16) reads the question corpus from local
YAML files in `interviews/vault/questions/` so contributors can edit a
question and see their changes in the dev server immediately. This
README explains the pipeline so the local-rendering gotcha that bit us
in early 2026 (rendered pages silently fetching from the production
worker instead of local YAML) doesn't bite anyone again.

## TL;DR — first-time setup

```bash
# 1. Install the vault CLI (Python). Pinned to ≥3.12.
pip install -e interviews/vault-cli/

# 2. Install Node dependencies.
cd interviews/staffml
npm install

# 3. Start the dev server. The predev hook regenerates the local corpus.
npm run dev
```

Open http://localhost:3000 and edit any YAML under
`interviews/vault/questions/`. Restart `npm run dev` (or just `Ctrl+C`
and re-run) to pick up the change — the predev hook re-runs `vault
build --local` automatically and the page reload fetches the fresh
content.

## How the data flow works

Question content (scenario, realistic_solution, napkin_math, …) does
NOT live in the React bundle. The bundle ships only a small summary
(title, level, track, etc.). The heavy fields are loaded on demand:

| Mode | Where details come from | When it activates |
|---|---|---|
| **Production** | Cloudflare Worker at `staffml-vault.mlsysbook-ai-account.workers.dev` | default (env unset) |
| **Local dev** | `public/data/corpus.json` served as a static asset | `NEXT_PUBLIC_VAULT_FALLBACK=static` |

The committed `.env.development` sets `NEXT_PUBLIC_VAULT_FALLBACK=static`
so `npm run dev` defaults to the local-static path. If you want to test
against the production worker instead (e.g. to reproduce a prod bug),
override locally:

```bash
# .env.development.local (gitignored, personal override)
NEXT_PUBLIC_VAULT_FALLBACK=
```

## What the predev hook does

`npm run dev` runs (in this order):

1. **`scripts/sync-design-grammar.mjs`** — regenerates
   `src/data/designGrammar.ts` from `design-grammar/grammar.yml`.
2. **`scripts/build-local-corpus.mjs`** — runs `vault build --local`
   from the repo root, which:
   - compiles every YAML under `interviews/vault/questions/` to
     `interviews/vault/vault.db` (canonical SQLite),
   - emits `interviews/staffml/src/data/corpus.json` (legacy bundle
     used by some build tooling),
   - mirrors that into `interviews/staffml/public/data/corpus.json`
     (the path the loader actually fetches at runtime — Turbopack
     does not bundle `src/data/corpus.json`),
   - copies any `vault/visuals/*.svg` into
     `interviews/staffml/public/question-visuals/`.

Both `corpus.json` files are gitignored as build artifacts. The vault
YAMLs are the source of truth.

## Manual local-corpus rebuild

If you edit a YAML while `npm run dev` is already running, the page
won't pick it up automatically — Next caches the static JSON for the
session. To refresh without a full dev restart:

```bash
# from the repo root
vault build --local        # short alias; --local-json also works
```

Then hard-reload the page (`Cmd+Shift+R`).

## Env vars used in local dev

| Var | Default | Effect |
|---|---|---|
| `NEXT_PUBLIC_VAULT_FALLBACK` | `static` (via `.env.development`) | `static` = read corpus.json from `public/data/`; unset/anything else = fetch production worker |
| `NEXT_PUBLIC_VAULT_API` | `https://staffml-vault.mlsysbook-ai-account.workers.dev` | Override the worker URL when not in static mode |
| `STAFFML_SKIP_LOCAL_CORPUS` | unset | `1` = skip the predev `vault build --local` step (useful when iterating on UI without touching question content) |

## Troubleshooting

### "Could not load the full question details"

The page-level error means `getStaticFullDetail` couldn't read
`/data/corpus.json`. Causes in rough order of likelihood:

1. **`vault` CLI not installed.** `which vault` → empty? Run
   `pip install -e interviews/vault-cli/` from the repo root.
2. **Predev was skipped.** Check `STAFFML_SKIP_LOCAL_CORPUS` isn't set
   to `1`, then `npm run dev` again.
3. **`public/data/corpus.json` is missing or stale.** Force a rebuild:
   `vault build --local` then hard-reload the browser.
4. **Env mode is wrong.** `cat .env.development` should show
   `NEXT_PUBLIC_VAULT_FALLBACK=static`.

### Question I edited shows old content

Two layers cache:

- **Build-time cache:** `corpus.json` is regenerated only when the
  predev hook runs. Restart `npm run dev` or run `vault build --local`.
- **Runtime cache:** the loader caches each question after first fetch.
  Hard-reload the page (`Cmd+Shift+R`) to invalidate.

### Question shows "loading…" forever

The fetch is silently failing. Open browser devtools → Network tab,
look for `corpus.json`. A 404 means it isn't in `public/data/`; rerun
`vault build --local`. A CORS/CSP error means the env mode is wrong —
see above.

## Beyond local dev

- **Worker source of truth:** `interviews/staffml/worker/`. Deploy
  notes in [`WORKER_DEPLOY.md`](WORKER_DEPLOY.md).
- **Vault CLI documentation:** [`../vault-cli/README.md`](../vault-cli/README.md).
- **Architecture:** [`../vault/ARCHITECTURE.md`](../vault/ARCHITECTURE.md).
- **Contribution policy + CLA:** [`CONTRIBUTING.md`](CONTRIBUTING.md).
