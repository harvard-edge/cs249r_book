# StaffML Interviewer Worker

Cloudflare Worker that powers the **Ask Interviewer** panel in StaffML's Mock Interview mode. Adapter pattern over six LLM providers, server-side Socratic system prompt enforcement, KV-backed rate limiter.

See [`../WORKER_DEPLOY.md`](../WORKER_DEPLOY.md) for the full deploy guide.

## At a glance

- **Single file**: [`src/index.ts`](src/index.ts) (~400 lines)
- **Default provider**: Cloudflare Workers AI (Llama 3.1 8B), no API key required, free
- **Optional providers**: Groq, OpenAI, Anthropic, Gemini, OpenRouter — drop in by setting the corresponding `wrangler secret put`
- **Rate limit**: 10/IP/hour, 60/IP/day, 8000/global/day (all configurable via env vars)
- **System prompt**: locked server-side, can't be bypassed by clients

## Adding a provider

1. Get an API key from the provider
2. `wrangler secret put PROVIDER_API_KEY`
3. Done — the Worker auto-detects on next deploy

The priority order is `groq → openai → anthropic → gemini → openrouter → cf-workers-ai`. To override, set:

```bash
wrangler secret put PROVIDER_PRIORITY anthropic,gemini,cf-workers-ai
```

The first available provider in the priority list wins for each request, with automatic fallback through the chain on error.

## Local dev

```bash
npm install
wrangler dev
# in another terminal:
curl -X POST http://localhost:8787/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "what is the latency budget?", "context": "DLRM-style recommender, 50 sparse lookups per request"}'
```

## Endpoints

- `POST /ask` — `{ question, context?, history? }` → `{ answer, provider, vendorLabel, modelLabel, privacyNote }`
- `GET /health` — `{ ok: true, providers: ["groq", "cf-workers-ai"] }`

## Why these specific defaults

The system prompt is the single most important piece of code in the file. It's what makes the panel safe to ship — the LLM can only answer clarifying questions, never solve the candidate's problem. Edit it in `SOCRATIC_SYSTEM_PROMPT`.

The rate limits are conservative to protect free-tier ceilings. Bump them via env vars once you have a paid key or vendor sponsorship.
