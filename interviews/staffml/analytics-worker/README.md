# StaffML Analytics Worker

Lightweight Cloudflare Worker for collecting anonymous usage analytics from StaffML.

## What It Collects

Anonymous events with **no PII**, no cookies, no persistent user IDs:
- Question scores (topic, zone, level, track, score 0-3)
- Gauntlet starts/completions
- Issue reports and improvement suggestions
- Daily challenge completions

Session IDs are ephemeral UUIDs that reset when the browser tab closes.

## Setup

### 1. Install Wrangler

```bash
npm install -g wrangler
wrangler login
```

### 2. Create KV Namespace

```bash
cd interviews/staffml/analytics-worker
wrangler kv:namespace create STAFFML_ANALYTICS
```

Copy the returned namespace ID and update `wrangler.toml`:

```toml
[[kv_namespaces]]
binding = "STAFFML_ANALYTICS"
id = "<YOUR_KV_NAMESPACE_ID>"
```

### 3. Deploy

```bash
wrangler deploy
```

Note the URL (e.g., `https://staffml-analytics.<your-subdomain>.workers.dev`).

### 4. Configure StaffML

Set the analytics endpoint in your build environment:

```bash
# In the GitHub Actions workflow or .env.local:
NEXT_PUBLIC_ANALYTICS_URL=https://staffml-analytics.<your-subdomain>.workers.dev
```

Without this variable, analytics works in local-only mode (dashboard shows local data).

## Endpoints

### POST /
Accepts a batch of events:
```json
{
  "events": [
    { "type": "question_scored", "topic": "roofline-analysis", "zone": "recall", "level": "L3", "track": "cloud", "score": 2, "_ts": 1712000000000, "_sid": "abc-123" }
  ]
}
```

Response: `{ "accepted": 1 }`

### GET /
Returns aggregate summary:
```json
{
  "totalEvents": 1234,
  "last7Days": {
    "uniqueSessions": 42,
    "questionsScored": 380,
    "gauntletsCompleted": 15,
    "eventsByDay": { "2026-04-01": 50, ... },
    "scoresByLevel": { "L3": { "total": 120, "count": 50, "avg": "2.40" } }
  }
}
```

## Security

- CORS restricted to `mlsysbook.ai`, `harvard-edge.github.io`, and `localhost`
- Max 100 events per request
- Max 1KB per event
- Email-pattern detection (rejects events containing PII)
- Field allowlist (strips unknown fields)
- 90-day TTL on stored data
- No authentication required (anonymous by design)

## Data Retention

Events are stored with a 90-day TTL in Cloudflare KV. After 90 days, they are automatically deleted. The running event counter (`meta:total_events`) persists indefinitely.
