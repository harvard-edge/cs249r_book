# SocratiQ chat Worker

Source-controlled replacement for the missing proxy source. **It is not deployed.** The reader and legacy production endpoint settings remain unchanged. This Worker provides bounded chat transport; it does not create accounts, sync study data, or implement `/results` or `/score`.

## Contract

| Route | Request | Response |
| --- | --- | --- |
| `POST /ai?provider=groq` | OpenAI-compatible chat JSON, `stream: false` or omitted | OpenAI-compatible JSON with `choices[0].message.content` |
| `POST /ai/stream?provider=groq` | Same JSON, `stream: true` or omitted | Provider SSE, including `[DONE]` |
| `POST /ai/stream?provider=groq` | Same JSON, `stream: false` | JSON, for legacy nonstream configuration |
| `OPTIONS` on either chat route | Allowed `Origin`, requested method `POST`, headers `Content-Type`/`Accept` | CORS preflight |

The existing clients may include `url=https://api.groq.com/openai/v1/chat/completions`. If present it must match that exact string. The Worker selects the fixed Groq endpoint itself and refuses redirects. Duplicate or unknown query parameters, arbitrary destinations, client credentials in JSON, tools, and unrecognized payload fields are rejected. Browser Authorization headers are never forwarded.

Groq models are explicitly limited to the current reader `openai/gpt-oss-120b` and legacy `openai/gpt-oss-20b`. Ordered `system`, `user`, and `assistant` text messages pass through without trimming, rewriting, merging, or dropping system instructions. Optional `temperature` and `top_p` are validated. Additional providers configured in the legacy client (Gemini, Cerebras, SambaNova, Mistral, OpenRouter under `openai`, Hugging Face, Awan) return `501 provider_not_supported`; the Worker never substitutes Groq for them. Unknown providers return `400`.

These are configured transport bounds, not measured provider capacity: request bytes `131072`; messages `32`; characters per message `64000`; combined characters `96000`; output tokens `2048`; upstream output bytes `262144`; body read deadline `5000 ms`; total quota lookup, body, and upstream deadline `40000 ms`. See `LIMITS` in `worker.mjs`. Oversized or unsupported requests fail before provider fetch. A timeout returns `504` before response headers; after SSE headers it emits an `error` event and closes without a success terminator. Provider `429` remains `429`, other provider HTTP client errors retain their status, and upstream server/redirect/network failures return `502`. Error bodies redact provider diagnostics. Do not treat partial streamed output as complete.

## Local checks

No new package is required. Node supplies unit test APIs. Local integration additionally requires an installed `wrangler` executable. Run from the repository root:

```sh
node --test socratiq/cloudflare/test/worker.test.mjs
node --test socratiq/cloudflare/test/local-integration.test.mjs
```

The integration harness starts Wrangler on loopback, imports the production handler, supplies a constructed provider response instead of network fetch, and calls it through the actual reader transport. It also tests legacy JSON and the local rate-limit binding. It never calls an AI provider. `WORKER_TEST_PORT` can override the local test port. The compatibility date is pinned to the installed runtime's supported `2026-04-08`; the tested CLI was Wrangler `4.80.0`.

For manual local provider work, use `wrangler dev --local --config socratiq/cloudflare/wrangler.toml` after setting the nonsecret origin/switch variables and an ignored `.dev.vars` in the configuration directory. That command can make billable provider requests once a key is supplied. Do not put a provider key in frontend code, the TOML file, a committed env file, a URL, or a shell command argument. Tests use constructed fixture credentials only.

## Provisioning and operation

The checked-in configuration fails closed: `ENABLED="false"`, empty `ALLOWED_ORIGINS`, no provider secret, and no public route or workers.dev/preview URL. An operator must supply the intended Cloudflare account, Worker name and route, exact allowed website origins, Groq project credential, provider quota/budget policy, and a rate-limit namespace unique in that account before an approved deployment. Use the Cloudflare secret store (`wrangler secret put GROQ_API_KEY` in the approved target configuration) rather than a plaintext variable. Configure `ALLOWED_ORIGINS` as a comma-separated list of exact origins, with scheme and optional port, no path or wildcard. Requests without a matching Origin are rejected.

`AI_RATE_LIMITER` is mandatory and fails closed if missing or unavailable. The example config selects a configured threshold of `10` requests per `60` seconds per connecting-IP key; change it to the approved operating policy. Missing platform IPs share a bucket. This is Cloudflare's per-location, eventually consistent limiter, **not a global spend cap or authentication**. CORS restricts browsers, but a nonbrowser caller can forge Origin. An exposed browser endpoint therefore needs provider-side project limits, monitored usage, budget alerts, and a documented owner who can disable it. Keep `ENABLED=false` until those controls and the public-access policy are approved. Disable by deploying that switch or removing the route/key. Do not represent a Worker request-count limit, CPU limit, or an alert as a hard provider budget cap. A separate global quota service would require an explicit design; none is invented here.

The handler does not log request content, selected passages, credentials, or provider bodies. Worker observability is disabled in the supplied config. If operational logs are added, use status/error codes and bounded aggregates only; do not enable body/header capture. The local harness prints HTTP method/path/status and constructed fixture results. Provider privacy and retention remain governed by the configured provider account; forwarding a passage to Groq is not a local-only operation.

Production activation remains blocked by account access to the existing deployed proxies, an approved provider credential and operating policy, and explicit deployment/endpoint-migration authorization. Local mocked tests establish transport behavior, not live inference quality, production quotas, or the deployed proxy's instruction handling. Migrate the frontend only after an authorized deployment passes live system-sentinel controls and both reader/legacy workflows; preserve a rollback endpoint.

Official references: [Groq chat API](https://console.groq.com/docs/api-reference), [Cloudflare rate-limit bindings and accuracy](https://developers.cloudflare.com/workers/runtime-apis/bindings/rate-limit/), [Worker secrets](https://developers.cloudflare.com/workers/configuration/secrets/), and [Wrangler configuration](https://developers.cloudflare.com/workers/wrangler/configuration/).
