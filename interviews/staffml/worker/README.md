# StaffML Interviewer Worker

Cloudflare Worker that powers the **Ask Interviewer** panel in StaffML's Mock Interview mode. Adapter pattern over many LLM providers, server-side Socratic system prompt enforcement, KV-backed rate limiter, paid-tier waitlist.

See [`../WORKER_DEPLOY.md`](../WORKER_DEPLOY.md) for the full deploy guide.

## At a glance

- **Single file**: [`src/index.ts`](src/index.ts) (~750 lines)
- **Default provider**: Cloudflare Workers AI (Llama 3.1 8B), no API key required, free
- **Built-in providers**: Groq, OpenAI, Anthropic, Gemini, OpenRouter, Cloudflare Workers AI
- **Drop-in providers** (any OpenAI-compatible API): Together AI, DeepSeek, Fireworks, Cerebras, Mistral La Plateforme, xAI Grok, Perplexity, vLLM, Ollama, LiteLLM — see [Adding a provider](#adding-a-provider) below
- **Rate limit**: 10/IP/hour, 60/IP/day, 8000/global/day (all configurable via env vars)
- **System prompt**: locked server-side, can't be bypassed by clients
- **Public route**: `https://mlsysbook.ai/api/staffml-interviewer/` (same-origin with StaffML site)
- **Fallback route**: `https://staffml-interviewer.<account>.workers.dev/`

## Endpoints

- `POST /ask` — `{ question, context?, history? }` → `{ answer, provider, vendorLabel, modelLabel, privacyNote }`
- `POST /waitlist` — `{ email, wouldPay, need? }` → `{ ok: true }` (captures paid-tier interest to `WAITLIST_KV`)
- `GET /health` — `{ ok: true, providers: [...], waitlist: true|false }`

All endpoints accept calls at both the workers.dev URL and the custom route (`mlsysbook.ai/api/staffml-interviewer/*`). The worker strips the custom-route prefix on entry so the router matches the same paths either way.

## Adding a provider

Most commercial LLM APIs implement the OpenAI-compatible `/chat/completions` shape, which means adding them is **one array entry** — no router changes, no new `callXxx` function, no request/response adapter code. The four existing request shapes cover:

| Shape | Works for |
|---|---|
| `openai-compat` | OpenAI, Groq, Together, DeepSeek, Fireworks, Cerebras, Mistral La Plateforme, OpenRouter, xAI, Perplexity, and self-hosted vLLM / Ollama / LiteLLM servers |
| `anthropic` | Claude (all models) |
| `gemini` | Google Gemini (all models) |
| `cf-workers-ai` | Cloudflare Workers AI (Llama, Mistral, Qwen, etc.) |

### The three-step recipe

**Step 1.** Append an entry to the `ADAPTERS` array in [`src/index.ts`](src/index.ts). See the next section for copy-paste templates.

**Step 2.** Add the corresponding env var to the `Env` interface at the top of the same file:

```typescript
export interface Env {
  // ... existing fields ...
  TOGETHER_API_KEY?: string;
  TOGETHER_MODEL?: string;
  TOGETHER_BASE_URL?: string;
}
```

**Step 3.** Set the secret and redeploy:

```bash
wrangler secret put TOGETHER_API_KEY
wrangler deploy
```

That's it. The adapter registry auto-detects the new provider at request time; if the secret is present, it becomes part of the priority chain.

### Copy-paste templates (verified working as of April 2026)

Paste any of these into the `ADAPTERS` array in `src/index.ts`:

**Together AI — fast, generous free tier, wide model catalog**

```typescript
{
  name: "together",
  vendorLabel: "Together AI",
  modelLabel: "Llama 3.1 70B Turbo",
  privacyNote: "Together AI does not train on API inputs.",
  requestShape: "openai-compat",
  defaultBaseUrl: "https://api.together.xyz/v1",
  defaultModel: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
  apiKeyEnv: "TOGETHER_API_KEY",
  baseUrlEnv: "TOGETHER_BASE_URL",
  modelEnv: "TOGETHER_MODEL",
},
```

**DeepSeek — very cheap, strong on systems reasoning**

```typescript
{
  name: "deepseek",
  vendorLabel: "DeepSeek",
  modelLabel: "DeepSeek-V3",
  privacyNote: "DeepSeek's API may train on submitted data — check current TOS.",
  requestShape: "openai-compat",
  defaultBaseUrl: "https://api.deepseek.com/v1",
  defaultModel: "deepseek-chat",
  apiKeyEnv: "DEEPSEEK_API_KEY",
  baseUrlEnv: "DEEPSEEK_BASE_URL",
  modelEnv: "DEEPSEEK_MODEL",
},
```

**Fireworks AI — fast inference, production-focused**

```typescript
{
  name: "fireworks",
  vendorLabel: "Fireworks AI",
  modelLabel: "Llama 3.1 70B Instruct",
  privacyNote: "Fireworks does not train on inference traffic.",
  requestShape: "openai-compat",
  defaultBaseUrl: "https://api.fireworks.ai/inference/v1",
  defaultModel: "accounts/fireworks/models/llama-v3p1-70b-instruct",
  apiKeyEnv: "FIREWORKS_API_KEY",
  baseUrlEnv: "FIREWORKS_BASE_URL",
  modelEnv: "FIREWORKS_MODEL",
},
```

**Cerebras — fastest inference in the industry, smaller catalog**

```typescript
{
  name: "cerebras",
  vendorLabel: "Cerebras",
  modelLabel: "Llama 3.1 70B",
  privacyNote: "Cerebras does not train on inference traffic.",
  requestShape: "openai-compat",
  defaultBaseUrl: "https://api.cerebras.ai/v1",
  defaultModel: "llama3.1-70b",
  apiKeyEnv: "CEREBRAS_API_KEY",
  baseUrlEnv: "CEREBRAS_BASE_URL",
  modelEnv: "CEREBRAS_MODEL",
},
```

**Mistral La Plateforme — European provider, strong coding models**

```typescript
{
  name: "mistral",
  vendorLabel: "Mistral AI",
  modelLabel: "Mistral Large",
  privacyNote: "Mistral does not train on API inputs by default.",
  requestShape: "openai-compat",
  defaultBaseUrl: "https://api.mistral.ai/v1",
  defaultModel: "mistral-large-latest",
  apiKeyEnv: "MISTRAL_API_KEY",
  baseUrlEnv: "MISTRAL_BASE_URL",
  modelEnv: "MISTRAL_MODEL",
},
```

**xAI (Grok) — OpenAI-compatible, Grok models**

```typescript
{
  name: "xai",
  vendorLabel: "xAI",
  modelLabel: "Grok Beta",
  privacyNote: "xAI may retain data per its terms of service — check before sending sensitive content.",
  requestShape: "openai-compat",
  defaultBaseUrl: "https://api.x.ai/v1",
  defaultModel: "grok-beta",
  apiKeyEnv: "XAI_API_KEY",
  baseUrlEnv: "XAI_BASE_URL",
  modelEnv: "XAI_MODEL",
},
```

**Perplexity — strong at search-augmented answers**

```typescript
{
  name: "perplexity",
  vendorLabel: "Perplexity",
  modelLabel: "Sonar Large",
  privacyNote: "Perplexity may retain data — check current TOS.",
  requestShape: "openai-compat",
  defaultBaseUrl: "https://api.perplexity.ai",
  defaultModel: "llama-3.1-sonar-large-128k-online",
  apiKeyEnv: "PERPLEXITY_API_KEY",
  baseUrlEnv: "PERPLEXITY_BASE_URL",
  modelEnv: "PERPLEXITY_MODEL",
},
```

**Self-hosted (vLLM, LiteLLM, Ollama, etc.) — anything that exposes an OpenAI-compatible endpoint**

```typescript
{
  name: "self-hosted",
  vendorLabel: "Self-hosted",
  modelLabel: "(configured via env)",
  privacyNote: "Traffic stays on your own infrastructure.",
  requestShape: "openai-compat",
  defaultBaseUrl: "https://your-llm-server.example.com/v1",
  defaultModel: "your-model-name",
  apiKeyEnv: "SELFHOSTED_API_KEY", // can be any string if your server doesn't auth
  baseUrlEnv: "SELFHOSTED_BASE_URL",
  modelEnv: "SELFHOSTED_MODEL",
},
```

### Changing the priority order

The priority chain is controlled by the `PROVIDER_PRIORITY` env var (comma-separated). Default order:

```
groq → openai → anthropic → gemini → openrouter → cf-workers-ai
```

To pin a different provider first:

```bash
wrangler secret put PROVIDER_PRIORITY
# When prompted, enter (no spaces around commas):
#   together,groq,cf-workers-ai
wrangler deploy
```

The first adapter in the list whose API key is present wins. If none of the listed providers are available, `cf-workers-ai` is always appended as the free-tier floor. If a provider fails a request (timeout, 5xx, invalid response), the worker falls through to the next one in the chain automatically.

### Secret management

Set secrets:

```bash
wrangler secret put GROQ_API_KEY
wrangler secret put TOGETHER_API_KEY
# etc.
```

List secrets (names only, never values):

```bash
wrangler secret list
```

Delete a secret:

```bash
wrangler secret delete TOGETHER_API_KEY
```

Secrets never land in git, never appear in the deployed bundle, and are only accessible to the worker itself at runtime. Rotating a key is `delete` + `put` + `deploy`.

### Request shapes other than openai-compat

If you need to add a provider that doesn't speak the OpenAI chat completions shape (e.g., AWS Bedrock, Azure OpenAI with its quirky `/deployments/<name>/` routing, Google Vertex AI), you'll need to add a fifth `RequestShape` to the union type and implement a new `callXxx` function alongside `callOpenAICompat`, `callAnthropic`, `callGemini`, `callCloudflareWorkersAi`. The pattern is ~30 lines per shape — use any of the existing four as a template. Then wire it into the `switch` in `callAdapter` and add the adapter entry with your new `requestShape` value.

## Local dev

```bash
npm install
wrangler dev
# in another terminal:
curl -X POST http://localhost:8787/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "what is the latency budget?", "context": "DLRM-style recommender, 50 sparse lookups per request"}'
```

## Why these specific defaults

The system prompt (`SOCRATIC_SYSTEM_PROMPT` in `src/index.ts`) is the single most important piece of code in the file. It's what makes the panel safe to ship — the LLM can only answer clarifying questions, never solve the candidate's problem. Edit it only if you understand the Socratic bargain the panel is selling.

The rate limits are conservative to protect free-tier ceilings. Bump them via env vars once you have a paid key, vendor sponsorship, or a clearer sense of real traffic patterns.
