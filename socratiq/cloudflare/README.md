# SocratiQ chat Worker

Source-controlled chat gateway for SocratiQ. The production Worker is activated at `https://mlsysbook.ai/api/socratiq`, with a dedicated provider credential stored as an encrypted Cloudflare secret. Groq and Cloudflare account plans were verified as Free during activation. Activation establishes the deployed configuration; successful live inference and frontend behavior require separate checks. This Worker provides bounded chat transport; it does not create accounts, sync study data, or implement `/results` or `/score`.

## Service identity

Follow the repository's [Cloudflare naming conventions](../../docs/README.md#cloudflare-service-naming).
The production Worker is named `socratiq-api`; a future staging Worker is
`socratiq-api-staging`, and the isolated local integration fixture is
`socratiq-api-local`. The public base is
`https://mlsysbook.ai/api/socratiq`. The handler accepts that exact public prefix
as well as the direct Worker paths below. Runtime bindings
retain purpose-based names such as `AI_RATE_LIMITER` and `GROQ_API_KEY`.
Provider credentials and limiter namespaces belong to SocratiQ, independently
of StaffML. No staging service or storage resource is provisioned by these names.

## Contract

| Route | Request | Response |
| --- | --- | --- |
| `POST /api/socratiq/ai?provider=groq` or `POST /ai?provider=groq` | OpenAI-compatible chat JSON, `stream: false` or omitted | OpenAI-compatible JSON with `choices[0].message.content` |
| `POST /api/socratiq/ai/stream?provider=groq` or `POST /ai/stream?provider=groq` | Same JSON, `stream: true` or omitted | Provider SSE, including `[DONE]` |
| Either stream route above | Same JSON, `stream: false` | JSON, for legacy nonstream configuration |
| `OPTIONS` on any chat route above | Allowed `Origin`, requested method `POST`, headers `Content-Type`/`Accept` | CORS preflight |
| `GET` or `HEAD /api/socratiq/health` or `/health` | No origin or credentials required | `204`, empty body, `Cache-Control: no-store` |

Route matching uses exact URL pathnames. Trailing slashes, encoded route characters, extra prefixes, repeated slashes, and lookalikes such as `/api/socratiqx/ai` are not rewritten into chat routes. The public prefix changes only dispatch; payload validation, exact-origin CORS, the enable switch, credentials, quotas, and provider restrictions are identical on both forms.

The health endpoint checks Worker dispatch/liveness only. It returns no configuration or content and does not inspect the enable switch, origin configuration, provider key, rate limiter, or provider. It remains available when chat is disabled or unconfigured, so `204` does **not** certify chat readiness, provider health, quota, or successful inference. Other health methods return `405` with `Allow: GET, HEAD`.

The existing clients may include `url=https://api.groq.com/openai/v1/chat/completions`. If present it must match that exact string. The Worker selects the fixed Groq endpoint itself and refuses redirects. Duplicate or unknown query parameters, arbitrary destinations, client credentials in JSON, tools, and unrecognized payload fields are rejected. Browser Authorization headers are never forwarded.

Groq models are explicitly limited to the current reader `openai/gpt-oss-120b` and legacy `openai/gpt-oss-20b`. Ordered `system`, `user`, and `assistant` text messages pass through without trimming, rewriting, merging, or dropping system instructions. Optional `temperature` and `top_p` are validated. Additional providers configured in the legacy client (Gemini, Cerebras, SambaNova, Mistral, OpenRouter under `openai`, Hugging Face, Awan) return `501 provider_not_supported`; the Worker never substitutes Groq for them. Unknown providers return `400`.

These are configured transport bounds, not measured provider capacity: request bytes `131072`; messages `32`; characters per message `64000`; combined characters `96000`; output tokens `2048`; upstream output bytes `262144`; body read deadline `5000 ms`; total quota lookup, body, and upstream deadline `40000 ms`. See `LIMITS` in `worker.mjs`. Oversized or unsupported requests fail before provider fetch. A timeout returns `504` before response headers; after SSE headers it emits an `error` event and closes without a success terminator. Provider `429` remains `429`, other provider HTTP client errors retain their status, and upstream server/redirect/network failures return `502`. Error bodies redact provider diagnostics. Do not treat partial streamed output as complete.

## Local checks

No new package is required. Node supplies unit test APIs. Local integration additionally requires an installed `wrangler` executable. Run from the repository root:

```sh
node --test socratiq/cloudflare/test/worker.test.mjs
node --test socratiq/cloudflare/test/local-integration.test.mjs
```

The integration harness starts Wrangler on loopback, imports the production handler, supplies a constructed provider response instead of network fetch, and calls it through the actual reader transport on both public-prefix and direct paths. It also tests legacy JSON, content-free health checks, exact-path rejection, preflight, origin rejection, and the local rate-limit binding. It never calls an AI provider. `WORKER_TEST_PORT` can override the local test port. The compatibility date is pinned to the installed runtime's supported `2026-04-08`; the tested CLI was Wrangler `4.80.0`.

For manual local provider work, use `wrangler dev --local --config socratiq/cloudflare/wrangler.toml` after setting the nonsecret origin/switch variables and an ignored `.dev.vars` in the configuration directory. That command can make billable provider requests once a key is supplied. Do not put a provider key in frontend code, the TOML file, a committed env file, a URL, or a shell command argument. Tests use constructed fixture credentials only.

## Provisioning and operation

The checked-in configuration enables chat with `ENABLED="true"`. Set it to `false` and redeploy to disable inference while retaining the health route. It declares the `socratiq-api` Worker, the `mlsysbook.ai/api/socratiq/*` zone route, and the exact `https://mlsysbook.ai` origin; workers.dev and preview URLs remain disabled. Namespace `2001` is reserved for SocratiQ production after a read-only binding audit of the existing Workers. The initial bootstrap omitted the route and retained the disabled switch; the subsequent authorized activation deployed the public route and enabled chat. Before any further deployment, an operator must confirm the intended Cloudflare account and zone, recheck that namespace reservation against the live account, and approve the operating policy. Requests without a matching Origin are rejected.

Use a dedicated SocratiQ Groq credential from an organization verified to be on the Free plan, with no automatic upgrade or paid fallback. Keep the target Worker within Cloudflare's Free plan. The workflow never upgrades a plan, provisions a database or storage service, uploads provider credentials, or selects a different provider when a free quota is exhausted. Its configuration allowlist rejects additional service bindings, but it cannot verify account billing state or guarantee zero cost from code alone. Verify both account plans and their quota behavior before approving a deployment; stop if the free-only policy cannot be established. Store `GROQ_API_KEY` in the Cloudflare secret store for `socratiq-api`, never in GitHub Actions, plaintext vars, frontend code, or committed env files. An approved operator can use `wrangler secret put GROQ_API_KEY --config socratiq/cloudflare/wrangler.toml` interactively; the first deployment may publish the disabled Worker before this secret exists.

### Manual production deployment

The [SocratiQ publish workflow](../../.github/workflows/socratiq-publish-live.yml) accepts only a manual dispatch on `main` with confirmation `PUBLISH`. It checks out the dispatched commit by immutable SHA, checks production configuration, runs gateway unit tests and local integration with a constructed upstream, and bundles with Wrangler `4.80.0` without credentials. It does not install or build the frontend application. The deployment job then waits for the `socratiq-production` environment and deploys that same SHA. It never overrides `ENABLED`; activation requires an explicitly reviewed source change and a separate approved deployment.

Before the first run, a repository administrator must configure `socratiq-production` in GitHub Settings → Environments with required reviewers, prevention of self-review, administrator bypass disabled, and a deployment branch rule allowing only the protected `main` branch (no tags). Protect `main` with review requirements for Worker and workflow changes. Environment protection is an external prerequisite; naming an environment in YAML does not configure its approval rules. If the repository's GitHub plan cannot enforce these protections, do not use this workflow for production.

Add `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` as secrets scoped to that environment only, following the StaffML credential names. Restrict the token to the intended account's Workers Scripts edit permission and this zone's Workers Routes edit permission, with any read permissions required by Wrangler; do not grant billing or unrelated resource administration. Do not make these credentials repository or organization secrets. The workflow exposes them only to the deployment step after approval. A protected environment with the wrong account or an overprivileged token remains unsafe despite passing local checks.

The configuration gate rejects the old placeholder namespace `1001` and local namespace `1002`; the reserved production namespace `2001` passes the source check. The gate also requires the fixed service, route, origin, disabled preview URLs, disabled observability, the reviewed rate policy, and only approved configuration keys. It allows either explicit source value of `ENABLED`. Live-account namespace verification, free-plan status, token scope, and provider credential readiness still require operator verification. An approved initial deployment with `ENABLED=false` establishes infrastructure and content-free health only.

To request deployment after those prerequisites are satisfied, select **SocratiQ publish (live)** in GitHub Actions, select `main`, and enter `PUBLISH`. The reviewer must check the exact run SHA, configuration, free-plan verification, and activation state before approving the environment job. Its final `GET /api/socratiq/health` check requires `204` and makes no inference request. A successful run establishes dispatch/liveness, not chat readiness or model quality. The health check alone does not verify production inference or frontend migration.

`AI_RATE_LIMITER` is mandatory and fails closed if missing or unavailable. The example config selects a configured threshold of `10` requests per `60` seconds per connecting-IP key; change it to the approved operating policy. Missing platform IPs share a bucket. This is Cloudflare's per-location, eventually consistent limiter, **not a global spend cap or authentication**. CORS restricts browsers, but a nonbrowser caller can forge Origin. An exposed browser endpoint therefore needs provider-side project limits, monitored usage, budget alerts, and a documented owner who can disable it. Activation requires approval of those controls and the public-access policy. Disable by deploying that switch or removing the route/key. Do not represent a Worker request-count limit, CPU limit, or an alert as a hard provider budget cap. A separate global quota service would require an explicit design; none is invented here.

The handler does not log request content, selected passages, credentials, or provider bodies. Worker observability is disabled in the supplied config. If operational logs are added, use status/error codes and bounded aggregates only; do not enable body/header capture. The local harness prints HTTP method/path/status and constructed fixture results. Provider privacy and retention remain governed by the configured provider account; forwarding a passage to Groq is not a local-only operation.

The production activation used a dedicated approved Groq Free-plan credential, verified account plans and limiter namespace, an approved operating policy, and explicit deployment authorization. Subsequent deployments and endpoint migrations require the same checks and authorization. The GitHub workflow additionally requires the protected-environment setup above; an authorized operator may deploy the reviewed configuration directly with Wrangler. Access to the old proxy is not required to create this separate service. Local mocked tests establish transport behavior, not live inference quality or production quotas. Migrate the frontend only after an authorized deployment passes live system-sentinel controls and both reader/legacy workflows; preserve a rollback endpoint. To roll back, use a reviewed `main` commit restoring the known-good Worker/configuration and dispatch the workflow again. To disable chat, deploy a reviewed `ENABLED=false` change; removing the route or provider key through an authorized operator is the emergency alternative.

Official references: [Groq chat API](https://console.groq.com/docs/api-reference), [Groq billing and plans](https://console.groq.com/docs/billing-faqs), [Groq rate limits](https://console.groq.com/docs/rate-limits), [Cloudflare Workers pricing](https://developers.cloudflare.com/workers/platform/pricing/), [Cloudflare rate-limit bindings and accuracy](https://developers.cloudflare.com/workers/runtime-apis/bindings/rate-limit/), [Worker secrets](https://developers.cloudflare.com/workers/configuration/secrets/), [Wrangler configuration](https://developers.cloudflare.com/workers/wrangler/configuration/), and [GitHub deployment environments](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments).
