# StaffML Analytics Worker

Lightweight Cloudflare Worker for collecting anonymous usage analytics from StaffML.

The deployed service is `staffml-analytics`, routed at `https://mlsysbook.ai/api/staffml-analytics`. This directory has no separate npm package. See the [Cloudflare service inventory](../../CLOUDFLARE.md) for storage identities and naming conventions.

## What It Collects

The client submits usage events without cookies or persistent user IDs:
- Question scores (topic, zone, level, track, score 0-3)
- Gauntlet starts/completions
- Issue reports and improvement suggestions
- Daily challenge completions

Session IDs are ephemeral UUIDs that reset when the browser tab closes.

Submitted fields can include free text, such as search queries and error reports. The Worker's email-pattern filter is a heuristic; it cannot guarantee that accepted events contain no personal information.

## Setup

The checked-in configuration already references the production KV namespace and route. Reuse those identities for an existing deployment. Namespace creation below is for a new account or a planned migration; a separate environment also needs a suffixed Worker name, separate storage and secrets, and its own route configuration before deployment.

### 1. Install Wrangler

```bash
npm install -g wrangler
wrangler login
```

### 2. Create KV Namespace

```bash
cd staffml/app/analytics-worker
wrangler kv namespace create staffml-analytics-events
```

Copy only the returned namespace ID into the intended environment's `wrangler.toml`, retaining the runtime binding `STAFFML_ANALYTICS`:

```toml
[[kv_namespaces]]
binding = "STAFFML_ANALYTICS"
id = "<YOUR_KV_NAMESPACE_ID>"
```

### 3. Deploy

For a new deployment, set an admin secret before exposing aggregate reads. Reuse the existing secret when updating an existing deployment:

```bash
wrangler secret put ADMIN_SECRET
wrangler deploy
```

Note the URL (e.g., `https://staffml-analytics.<your-subdomain>.workers.dev`).

### 4. Configure StaffML

The site workflows use the existing public route. Set the analytics endpoint in a local build environment, or override it for a separate deployment:

```bash
# In the GitHub Actions workflow or .env.local:
NEXT_PUBLIC_ANALYTICS_URL=https://mlsysbook.ai/api/staffml-analytics
```

Without this variable, analytics works in local-only mode (dashboard shows local data).

## Endpoints

### POST /
Accepts a batch of events without authentication. At the public API base, this is `POST https://mlsysbook.ai/api/staffml-analytics`:
```json
{
  "events": [
    { "type": "question_scored", "topic": "roofline-analysis", "zone": "recall", "level": "L3", "track": "cloud", "score": 2, "_ts": 1712000000000, "_sid": "abc-123" }
  ]
}
```

Response: `{ "accepted": 1 }`. Invalid JSON or a body without an `events` array returns `400`. Invalid entries within an array, including events with nested object or array fields, are skipped; the response counts accepted entries. Client event fields are scalar values.

### GET /
Returns an aggregate summary only when the request includes `Authorization: Bearer <ADMIN_SECRET>`. Missing or incorrect credentials return `401`. Keep this secret in an operator's environment; never put it in a `NEXT_PUBLIC_*` build variable or browser code. At the public API base, this is `GET https://mlsysbook.ai/api/staffml-analytics`.

Aggregate reads also return `401` when `ADMIN_SECRET` is missing or empty; event ingestion remains available.

Example response:
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
- Serialized event JSON capped at 1,024 characters (8,192 for `client_error` reports)
- Email-pattern detection (rejects matching events; does not cover all personal information)
- Field allowlist (strips unknown fields)
- 90-day TTL on stored data
- Anonymous event submissions; aggregate reads require the `ADMIN_SECRET` bearer token

The event count cap applies to each batch; it is not a request-frequency limiter or a total request-body byte limit. CORS controls browser access, and callers can omit or spoof `Origin` outside a browser. Repeated anonymous submissions can therefore consume storage and affect aggregates. Account-level abuse controls and stronger data minimization require a separate operational change.

## Local tests

From this directory, run `node --test test/worker.test.mjs` with Node.js 22 or later. The tests use in-memory KV and exercise configured origins, lookalike-origin rejection, admin authentication, and anonymous ingestion without contacting Cloudflare.

## Data Retention

Event batches are stored with a 90-day TTL in Cloudflare KV. The summary computes `totalEvents` from its recent seven-day window; the Worker does not maintain a lifetime event counter.
