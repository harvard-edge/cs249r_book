/**
 * Token-bucket rate limiter backed by Workers KV (Chip R3-H4 / ARCHITECTURE.md §10.3).
 *
 * One bucket per (IP, endpoint-class) keyed in RATE_LIMIT_KV. Endpoint classes:
 *   - 'default' : 60 req/min per IP on GET endpoints (overridable via env).
 *   - 'search'  : 10 req/min per IP on /search (tighter — FTS5 is expensive).
 *
 * KV is eventually consistent; bursts across POPs can exceed limits briefly.
 * That's acceptable for the "prevent abusive scraper blowing D1 budget" goal;
 * hard enforcement requires Durable Objects (deferred Phase-4 follow-up if
 * KV-level leak is observed).
 */

import type { Env } from "./types";

const BUCKET_CAP_DEFAULT = 60;
const BUCKET_CAP_SEARCH = 10;

/**
 * Resolve client IP — trust only the CF-Connecting-IP header (Chip R4-H-3).
 *
 * X-Forwarded-For is client-settable; trusting it lets an attacker rotate
 * their bucket with `X-Forwarded-For: $(uuidgen)` and bypass rate limits.
 * If CF-Connecting-IP is missing, we assume the request didn't come through
 * Cloudflare (misconfigured route, direct worker-to-worker call) and fail
 * closed by returning a sentinel that the caller treats as DENY.
 */
function clientIp(req: Request): string | null {
  const ip = req.headers.get("CF-Connecting-IP");
  return ip && ip.trim() ? ip.trim() : null;
}

function bucketCap(env: Env, className: "default" | "search"): number {
  if (className === "search") {
    return Number.parseInt(env.RATE_LIMIT_RPM_SEARCH ?? `${BUCKET_CAP_SEARCH}`, 10);
  }
  return Number.parseInt(env.RATE_LIMIT_RPM_DEFAULT ?? `${BUCKET_CAP_DEFAULT}`, 10);
}

export type RateLimitDecision = { ok: true } | { ok: false; retryAfterSec: number };

export async function checkRateLimit(
  env: Env,
  req: Request,
  className: "default" | "search",
): Promise<RateLimitDecision> {
  // If KV is not bound (e.g., local `vault api` shim), open-allow. Production
  // wrangler.toml must bind RATE_LIMIT_KV — CI + `vault doctor --check
  // d1-connectivity` surface this.
  if (!env.RATE_LIMIT_KV) return { ok: true };

  const ip = clientIp(req);
  // R4-H-3: fail-closed when CF-Connecting-IP is missing. This only happens
  // if the request bypassed Cloudflare's edge (direct worker-to-worker,
  // misconfigured route). Force caller through CF.
  if (ip === null) {
    return { ok: false, retryAfterSec: 60 };
  }

  const nowSec = Math.floor(Date.now() / 1000);
  const windowSec = Math.floor(nowSec / 60);
  const key = `rl:${className}:${ip}:${windowSec}`;

  const existing = await env.RATE_LIMIT_KV.get(key);
  const count = existing ? Number.parseInt(existing, 10) : 0;
  const cap = bucketCap(env, className);

  if (count >= cap) {
    // Next window starts at (windowSec + 1) * 60.
    const retryAfterSec = ((windowSec + 1) * 60) - nowSec;
    return { ok: false, retryAfterSec };
  }

  // NOTE on atomicity (Chip R4-H-3): KV read-then-write is not atomic across
  // concurrent requests on the same POP. Worst case: 2-3× cap leak per
  // minute per (IP, class) during bursts. Acceptable for the "prevent
  // abusive scraper blowing D1 budget" goal. Tight enforcement requires
  // Durable Objects; deferred to Phase-4 follow-up if KV-level leakage is
  // observed in telemetry.
  // Cloudflare KV rejects expirationTtl < 60. The intended value here is
  // "remainder-of-minute + 10s grace"; floor at 60 so we never ship an
  // illegal TTL. A slightly-longer-than-needed TTL on the tail end of a
  // minute is harmless (key naturally expires on the NEXT minute's rotation).
  const ttl = Math.max(60, 60 - (nowSec % 60) + 10);
  await env.RATE_LIMIT_KV.put(key, String(count + 1), { expirationTtl: ttl });
  return { ok: true };
}
