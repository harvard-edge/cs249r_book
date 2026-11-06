# StaffML Cloudflare services

StaffML follows the repository's [Cloudflare service naming convention](../docs/README.md#cloudflare-service-naming): deployed Workers use `<product>-<role>`, and nonproduction Workers append an environment. Display names use **StaffML**. Package names and source directories identify local code; they are not Cloudflare service names.

This inventory describes the checked-in configuration and client defaults. It does not confirm the current state of the Cloudflare account. Preserve the configured resource IDs and public URLs when deploying an existing service.

## Service inventory

| Responsibility | Deployed Worker name | Source and Wrangler configuration | Local package | Public API base |
| --- | --- | --- | --- | --- |
| Question vault, search, and taxonomy | `staffml-vault` | [`vault-worker/`](vault-worker/), [`wrangler.toml`](vault-worker/wrangler.toml) | `staffml-vault-worker` | `https://staffml-vault.mlsysbook-ai-account.workers.dev` |
| Socratic interviewer and waitlist | `staffml-interviewer` | [`app/worker/`](app/worker/), [`wrangler.toml`](app/worker/wrangler.toml) | `staffml-interviewer-worker` | `https://mlsysbook.ai/api/staffml-interviewer` |
| Anonymous usage events | `staffml-analytics` | [`app/analytics-worker/`](app/analytics-worker/), [`wrangler.toml`](app/analytics-worker/wrangler.toml) | None; JavaScript Worker | `https://mlsysbook.ai/api/staffml-analytics` |
| Staging question vault | `staffml-vault-staging` | `env.staging` in [`vault-worker/wrangler.toml`](vault-worker/wrangler.toml) | Same vault Worker package | Use the URL returned by the staging deployment; override `NEXT_PUBLIC_VAULT_API` |

The Next.js app's private package is `staffml`; shared TypeScript contracts use `@staffml/vault-types`, and the Python CLI distribution is `staffml-vault` (command: `vault`). These names belong to different package ecosystems and do not provision Workers. The interviewer package's `-worker` suffix distinguishes its local code from the deployed service, as it does for the vault package.

`mlsysbook-stats` belongs to the wider MLSysBook site. It is a separate service from `staffml-analytics`; its Worker configuration is not maintained under `staffml/`.

## Storage and bindings

Wrangler bindings are uppercase snake case. They are local names used by the code, not the names or identities of remote resources. Reusing a binding name in separate Workers is supported; sharing its resource ID shares the underlying storage.

| Worker / environment | Binding | Configured resource |
| --- | --- | --- |
| `staffml-vault` | `DB` | D1 `staffml-vault`, ID `254f630f-dd6a-400e-8d86-786e92be7a70` |
| `staffml-vault-staging` | `DB` | D1 `staffml-vault-staging`, ID `f14b8691-9e3c-432b-a413-ece5287a262d` |
| Vault production **and staging** | `RATE_LIMIT_KV` | KV ID `19f1f3d7e03c4c3cb8704d951f7a22e7` (shared in the current configuration) |
| `staffml-interviewer` | `RATE_LIMIT_KV` | KV ID `a510cc7f791c40ffa28d19cf809c0e28` |
| `staffml-interviewer` | `WAITLIST_KV` | KV ID `e42c3db4eaa2471f81f3f73745959349` |
| `staffml-analytics` | `STAFFML_ANALYTICS` | KV ID `bf81298013404118beab61f55afe1d7d` |
| `staffml-interviewer` | `AI` | Cloudflare Workers AI binding |

KV namespace titles are not recorded in these Wrangler files; identify existing namespaces by ID rather than inferring a title from a binding. New storage resources should follow `<product>-<role>-<purpose>[-<environment>]`, for example `staffml-interviewer-rate-limit-staging`. Keep code bindings such as `RATE_LIMIT_KV` stable when assigning a descriptive namespace title.

## Client and release wiring

- `NEXT_PUBLIC_VAULT_API` overrides the vault base URL; [`vault-config.ts`](app/src/lib/vault-config.ts) retains the configured production `workers.dev` default. `NEXT_PUBLIC_VAULT_FALLBACK=static` selects local corpus data.
- `NEXT_PUBLIC_INTERVIEWER_ENDPOINT` overrides the interviewer base URL; [`interview-api.ts`](app/src/lib/interview-api.ts) defaults to the public `/api/staffml-interviewer` route. The Worker also accepts its `workers.dev` URL.
- `NEXT_PUBLIC_ANALYTICS_URL` selects the analytics base URL. An unset value keeps analytics local. Both site workflows configure `/api/staffml-analytics`.
- [`staffml-preview-dev.yml`](../.github/workflows/staffml-preview-dev.yml) publishes the development site; [`staffml-publish-live.yml`](../.github/workflows/staffml-publish-live.yml) publishes the production site, ships vault data, and deploys the vault Worker. The interviewer and analytics Workers have their own deployment guides: [interviewer](app/WORKER_DEPLOY.md), [analytics](app/analytics-worker/README.md).

The development site currently uses the production API defaults in its workflow. A development site preview does not imply a separate Worker environment. Existing workflow filenames remain supported; use the repository convention for new workflows.

## Changes that require a migration

No existing service needs a cosmetic rename. Keep the current D1 names, KV IDs, binding names, API routes, `workers.dev` URLs, and browser storage keys compatible.

The staging vault currently shares production's rate-limit KV namespace. Its [limiter](vault-worker/src/rate_limit.ts) keys counters as `rl:<endpoint-class>:<IP>:<minute-window>`, with no environment component. Requests from the same IP to staging and production can therefore consume the same quota within a window. Separate D1 databases and Worker names do not prevent this interference.

To isolate staging, provision a separate namespace such as `staffml-vault-rate-limit-staging`, update only the ID under `env.staging.kv_namespaces` while retaining `binding = "RATE_LIMIT_KV"`, and deploy with `--env staging`. Leave the production KV ID and both D1 IDs unchanged. Test requests from the same IP against both environments: exhausting a staging endpoint class should leave production's counter unaffected. Confirm staging still serves its own D1 data and CORS allowlist. A fresh staging namespace needs no counter copy; the counters are short-lived. Renaming the binding alone would not isolate counters. This inventory does not provision that namespace or change either environment.

For any future service or URL migration, provision the replacement and its environment-specific storage, configure secrets independently, test it, and update client build variables before retiring the old endpoint. Keep the existing route available until published clients have moved. Do not replace an existing resource ID with a newly created namespace merely to standardize its title.
