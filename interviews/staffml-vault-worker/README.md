# staffml-vault-worker

Cloudflare Worker serving the StaffML vault API over D1.

Implements ARCHITECTURE.md §10 with v2.1 refinements.

## Deploy (user action — requires Cloudflare account + authenticated wrangler)

```bash
# One-time setup: provision D1 databases
wrangler d1 create staffml-vault                 # production
wrangler d1 create staffml-vault-staging          # staging
# Paste the returned database_id values into wrangler.toml.

# Apply schema (generated from vault.db DDL)
wrangler d1 execute staffml-vault --file ../vault-cli/scripts/d1-schema.sql

# Seed from a release
wrangler d1 execute staffml-vault --file ../vault/releases/0.9.0/d1-migration.sql

# Deploy
cd interviews/staffml-vault-worker/
pnpm install
pnpm deploy:staging
pnpm deploy:production
```

## Endpoints

- `GET /manifest` — release metadata + schema_fingerprint status.
- `GET /questions?track=&level=&zone=&topic=&status=&cursor=&limit=` — cursor-paginated list.
- `GET /questions/:id` — single-question lookup.
- `GET /search?q=&limit=` — text search (Phase-3 LIKE; FTS5 upgrade Phase-3.x).
- `GET /stats` — aggregate counts.

All responses carry `ETag` + `X-Vault-Release` + appropriate
`Cache-Control: public, max-age=<ttl>, stale-while-revalidate=<2×ttl>`.

## Degraded mode

If the schema-fingerprint check fails at cold start, the worker returns
responses with `X-Vault-Degraded: schema-fingerprint-mismatch` header. Site
reads this header to render an operator banner. The worker continues serving
from D1 read-only rather than failing all requests (Chip N-H1 fix).

## Local dev — use `vault api` instead

The vault-cli provides a Python shim that mirrors this Worker's endpoint
surface from a local `vault.db`. Contributors without a Cloudflare account use
that instead — see `interviews/CONTRIBUTING.md` (H-17 resolution).

## Phase-4 gates

Before production deploy (per CUTOVER_QA.md §0):

- FTS5 load test green: p99 warm ≤100ms, p99 cold ≤500ms, ≤500 D1 row-reads per query.
- Data-plane SLIs reporting zero divergence in staging.
- Rollback drill executed on staging: `NEXT_PUBLIC_VAULT_FALLBACK=static` path verified.
