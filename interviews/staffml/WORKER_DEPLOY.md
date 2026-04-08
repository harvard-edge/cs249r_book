# Deploying the StaffML Interviewer Worker

This guide walks you through deploying the Ask Interviewer Worker to Cloudflare for the first time. Total time: ~10 minutes. Cost: $0 within Cloudflare's free tier.

The Worker code lives in [`worker/`](worker/). The StaffML client (this directory) consumes it via the `NEXT_PUBLIC_INTERVIEWER_ENDPOINT` env var.

---

## Prerequisites

- A free Cloudflare account ([sign up](https://dash.cloudflare.com/sign-up))
- Node.js 18+ installed locally
- ~10 minutes

---

## First-time deploy

### 1. Install Wrangler and dependencies

```bash
cd interviews/staffml/worker
npm install
npx wrangler login
```

The login command opens a browser to authorize Wrangler with your Cloudflare account.

### 2. Create the rate-limit KV namespace

```bash
npx wrangler kv:namespace create RATE_LIMIT_KV
```

This prints something like:

```
🌀 Creating namespace with title "staffml-interviewer-RATE_LIMIT_KV"
✨ Success!
Add the following to your configuration file in your kv_namespaces array:
{ binding = "RATE_LIMIT_KV", id = "abc123def456..." }
```

**Copy the `id` value** and paste it into [`worker/wrangler.toml`](worker/wrangler.toml), replacing `REPLACE_WITH_KV_NAMESPACE_ID`:

```toml
[[kv_namespaces]]
binding = "RATE_LIMIT_KV"
id = "abc123def456..."   # ← paste your real id here
```

### 2b. Create the waitlist KV namespace

The `POST /waitlist` endpoint writes paid-tier waitlist submissions to a
second KV namespace. If you skip this step, the worker still boots fine
but `/waitlist` will return 503 and the client UI will transparently
fall back to `mailto:` so no submissions are lost.

```bash
npx wrangler kv:namespace create WAITLIST_KV
```

**Copy the new `id` value** and paste it into `worker/wrangler.toml`,
replacing `REPLACE_WITH_WAITLIST_KV_NAMESPACE_ID`:

```toml
[[kv_namespaces]]
binding = "WAITLIST_KV"
id = "xyz789abc012..."   # ← paste your real id here
```

To read the waitlist later:

```bash
npx wrangler kv:key list --binding WAITLIST_KV
npx wrangler kv:key get --binding WAITLIST_KV "wl:2026-04-08T12:34:56.000Z:abc123..."
```

There's deliberately no admin endpoint — pulling records via `wrangler`
keeps the worker's attack surface minimal.

### 3. Deploy the Worker

```bash
npx wrangler deploy
```

Wrangler prints the deployed URL, something like:

```
Published staffml-interviewer
  https://staffml-interviewer.your-subdomain.workers.dev
```

**Copy that URL** — you need it for the next step.

### 4. Smoke-test the deploy

```bash
curl https://staffml-interviewer.your-subdomain.workers.dev/health
```

Expected response:

```json
{ "ok": true, "providers": ["cf-workers-ai"], "waitlist": true }
```

If you see `cf-workers-ai` in the providers list, the Cloudflare Workers AI binding is working. The Worker is live with the default Llama 3.1 8B floor model. `"waitlist": true` means the WAITLIST_KV binding is configured; `false` means /waitlist will return 503 until you provision the namespace.

Test an actual question:

```bash
curl -X POST https://staffml-interviewer.your-subdomain.workers.dev/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "what is the typical latency budget?",
    "context": "Real-time inference for an ad ranking model"
  }'
```

You should get back a JSON response with `answer`, `provider`, `vendorLabel`, `modelLabel`, and `privacyNote`.

### 5. Wire StaffML to the Worker

In your StaffML build environment (locally for `npm run dev`, or in your CI for production):

```bash
export NEXT_PUBLIC_INTERVIEWER_ENDPOINT=https://staffml-interviewer.your-subdomain.workers.dev
```

Or add to `interviews/staffml/.env.local`:

```
NEXT_PUBLIC_INTERVIEWER_ENDPOINT=https://staffml-interviewer.your-subdomain.workers.dev
```

Rebuild StaffML. The Ask Interviewer panel will now POST to your Worker.

---

## Adding a better model later (Groq, OpenAI, Anthropic, Gemini, OpenRouter)

The Worker is built around an adapter pattern. Adding a new provider is **one command** — no code changes, no redeploy gymnastics.

### Groq (recommended next upgrade — Llama 3.1 70B, free tier)

1. Get a free key at [console.groq.com/keys](https://console.groq.com/keys)
2. Run:
   ```bash
   cd interviews/staffml/worker
   npx wrangler secret put GROQ_API_KEY
   ```
3. Paste your key when prompted. The Worker now prefers Groq for every request, falling back to Cloudflare Workers AI on error.

Verify with:

```bash
curl https://staffml-interviewer.your-subdomain.workers.dev/health
```

You should see `"providers": ["groq", "cf-workers-ai"]`.

### OpenAI

```bash
npx wrangler secret put OPENAI_API_KEY
# default model: gpt-4o-mini
```

### Anthropic Claude

```bash
npx wrangler secret put ANTHROPIC_API_KEY
# default model: claude-3-5-haiku-latest
```

### Google Gemini

```bash
npx wrangler secret put GEMINI_API_KEY
# default model: gemini-1.5-flash
# Note: Google's free Gemini API may use prompts to improve products.
# The Worker exposes this in the privacyNote that the panel displays.
```

### OpenRouter (any of dozens of models)

```bash
npx wrangler secret put OPENROUTER_API_KEY
# default model: meta-llama/llama-3.1-70b-instruct
# To pick a different model:
npx wrangler secret put OPENROUTER_MODEL
```

### Local self-hosted (Ollama, vLLM, llama.cpp server)

```bash
npx wrangler secret put OPENAI_API_KEY     # any non-empty string
npx wrangler secret put OPENAI_BASE_URL    # http://your-host:11434/v1
npx wrangler secret put OPENAI_MODEL       # llama3.1:70b
```

The OpenAI adapter is OpenAI-API-compatible, which Ollama and most self-hosted servers also speak. This routes the calls to your local box without changing any Worker code.

---

## Changing priority order

The default chain is `groq → openai → anthropic → gemini → openrouter → cf-workers-ai`. To override:

```bash
npx wrangler secret put PROVIDER_PRIORITY anthropic,gemini,cf-workers-ai
```

Comma-separated list of provider names. The first available wins. `cf-workers-ai` is always appended as the floor whether or not you list it.

---

## Vendor sponsorship setup

If a vendor (Google, Cloudflare, Anthropic, etc.) donates a key with elevated quota:

1. `wrangler secret put VENDOR_API_KEY`
2. (Optional) `wrangler secret put PROVIDER_PRIORITY vendor,...` to put them first
3. (Optional) `wrangler secret put GLOBAL_DAILY_CEILING 100000` to raise the rate-limit ceiling
4. Done. The vendor's name + model + privacy note is automatically shown in the panel attribution.

The user-visible attribution comes from the adapter config in [`worker/src/index.ts`](worker/src/index.ts) (`vendorLabel`, `modelLabel`, `privacyNote`). If a vendor wants custom attribution text ("Powered by Google AI"), edit those fields and redeploy.

---

## Operations

### Watching live logs

```bash
cd interviews/staffml/worker
npx wrangler tail
```

Streams every request the Worker handles, including errors and rate-limit hits.

### Rotating a key

```bash
npx wrangler secret put GROQ_API_KEY     # paste new key, old key is replaced
```

No redeploy needed — secrets propagate within ~30 seconds.

### Removing a provider

```bash
npx wrangler secret delete GROQ_API_KEY
```

The Worker stops using Groq on the next request and falls back to the next available provider in the priority chain.

### Adjusting rate limits

```bash
npx wrangler secret put RATE_LIMIT_PER_HOUR 30   # default 10
npx wrangler secret put RATE_LIMIT_PER_DAY 200   # default 60
npx wrangler secret put GLOBAL_DAILY_CEILING 50000   # default 8000
```

These take effect on the next request — no redeploy.

### Restricting CORS to a specific origin

By default the Worker accepts requests from any origin (`*`). To lock to your StaffML deployment only:

```bash
npx wrangler secret put ALLOWED_ORIGINS https://staffml.ai,https://harvard-edge.github.io
```

---

## Costs

**At current scale: $0/month.** Cloudflare's free plan includes:

- 100,000 Worker requests/day
- 10,000 Workers AI neurons/day (≈ 100–300 LLM calls/day depending on prompt size)
- 100,000 KV reads/day, 1,000 KV writes/day

The KV rate limiter uses ~3 reads + 3 writes per request. At the default 8,000 global daily ceiling, that's 24k reads + 24k writes/day — well within the free tier.

If StaffML grows past the free tier on Workers AI specifically, the next step is adding any of the optional providers above (most have generous free tiers themselves) or upgrading to Cloudflare's Workers Paid plan ($5/month) which moves the Workers AI ceiling to 10M neurons/month.

---

## Tearing it down

If you ever want to remove the Worker entirely:

```bash
cd interviews/staffml/worker
npx wrangler delete
npx wrangler kv:namespace delete --binding RATE_LIMIT_KV
```

Then unset `NEXT_PUBLIC_INTERVIEWER_ENDPOINT` in StaffML's build environment. The Ask Interviewer panel will detect the missing endpoint and fall back to journal-only mode automatically.
