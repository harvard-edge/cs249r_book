/**
 * StaffML Vault Worker — edge API over D1.
 *
 * Implements ARCHITECTURE.md §10 with v2.1 + v2.2 refinements:
 * - X-Vault-Release is INFORMATIONAL (not hard-reject) — soft SLI signal.
 * - schema_fingerprint verified against sqlite_master at cold start; on
 *   mismatch, serves cached-only (Cache API) with X-Vault-Degraded (no 5xx).
 * - Cache keys include release_id for atomic POP invalidation on deploy.
 * - 10-minute cross-release grace window during propagation.
 * - Cloudflare Cache API wired (B.1) — release-hash-keyed per-URL.
 * - Keyset pagination (B.3) — cursors are {after_id, filter_hash}; server
 *   rejects cross-filter cursor reuse.
 * - Rate limits enforced via RATE_LIMIT_KV (B.4) — 60 rpm default, 10 rpm
 *   on /search.
 */

import { checkRateLimit } from "./rate_limit";
import type { Cursor, Env, Manifest, QuestionRow } from "./types";

// Cold-start memo for schema fingerprint check (per-instance; warm workers skip).
// Chip R7-M-3 + Gemini R9: soft-fail (transient D1 error) gets a 5min retry
// window; hard-fail (wrong fingerprint from release_metadata) stays sticky
// until release rollover invalidates schemaOk.
let schemaOk: boolean | undefined;
let schemaCheckedAt = 0;
const SCHEMA_RECHECK_MS = 5 * 60 * 1000;

// Chip R7-M-5: memoized FTS5 presence probe (reset on release change).
let ftsProbed: boolean | undefined;

async function checkSchemaFingerprint(env: Env, expectedFromDb: string | undefined): Promise<boolean> {
  if (schemaOk === false && Date.now() - schemaCheckedAt > SCHEMA_RECHECK_MS) {
    schemaOk = undefined;   // re-check after backoff window
  }
  if (schemaOk !== undefined) return schemaOk;
  schemaCheckedAt = Date.now();

  if (!expectedFromDb) {
    schemaOk = false;
    return schemaOk;
  }
  try {
    // Chip R7-H-2 + Gemini R9-C-1: FTS5 shadow tables
    // (questions_fts_data/idx/docsize/content/config) have auto-generated DDL
    // that differs across SQLite versions (host Python vs Cloudflare D1).
    // Must mirror the compiler.py filter exactly or fingerprint permanently
    // mismatches, pinning the worker to degraded mode forever.
    const result = await env.DB.prepare(
      "SELECT sql FROM sqlite_master " +
      "WHERE type IN ('table','index','trigger','view') " +
      "AND name NOT LIKE 'sqlite_%' " +
      "AND name NOT IN ('questions_fts_data','questions_fts_idx'," +
      "                 'questions_fts_docsize','questions_fts_content'," +
      "                 'questions_fts_config') " +
      "ORDER BY name"
    ).all<{ sql: string | null }>();
    const ddl = (result.results ?? [])
      .map(r => (r.sql ?? "").replace(/\s+/g, " ").trim())
      .filter(Boolean)
      .join("\n");
    const encoded = new TextEncoder().encode(ddl);
    const digest = await crypto.subtle.digest("SHA-256", encoded);
    const hex = Array.from(new Uint8Array(digest))
      .map(b => b.toString(16).padStart(2, "0"))
      .join("");
    schemaOk = hex === expectedFromDb;
  } catch {
    schemaOk = false;
  }
  return schemaOk;
}

/**
 * Forget the cached schema-fingerprint decision so the next request re-checks.
 * Called on release_id change (Dean R3-NH-4): every deploy advances release_id,
 * which surfaces via /manifest. That natural cadence keeps the check fresh
 * without per-request cost.
 */
let lastSeenRelease: string | null = null;
function maybeInvalidateSchemaCache(releaseId: string | undefined): void {
  if (!releaseId) return;
  if (lastSeenRelease !== null && lastSeenRelease !== releaseId) {
    schemaOk = undefined;
    ftsProbed = undefined;   // Chip R7-M-5: FTS5 presence can change across releases
    // Chip R7-L-7: don't null manifestMemo — 60s TTL handles rollover, and
    // nulling mid-write caused concurrent-request stampedes on D1.
  }
  lastSeenRelease = releaseId;
}

// Gemini R5-H-1: module-level manifest memo (60s TTL). Before this,
// getManifest hit D1 on every request — including cache-hit requests —
// which silently blew the §10.4 "~5 D1 reads/session" cost target.
let manifestMemo: { manifest: Manifest; at: number } | null = null;
const MANIFEST_TTL_MS = 60_000;

async function getManifest(env: Env): Promise<Manifest> {
  const now = Date.now();
  if (manifestMemo && now - manifestMemo.at < MANIFEST_TTL_MS) {
    return manifestMemo.manifest;
  }
  const rows = await env.DB.prepare("SELECT key, value FROM release_metadata").all<{
    key: string;
    value: string;
  }>();
  const meta: Record<string, string> = {};
  for (const r of rows.results ?? []) meta[r.key] = r.value;
  const release_id = meta.release_id ?? "unknown";
  maybeInvalidateSchemaCache(release_id);
  const manifest: Manifest = {
    release_id,
    release_hash: meta.release_hash ?? "",
    schema_version: meta.schema_version ?? "1",
    policy_version: meta.policy_version ?? "1",
    published_count: Number.parseInt(meta.published_count ?? "0", 10),
    schema_fingerprint_ok: await checkSchemaFingerprint(env, meta.schema_fingerprint),
  };
  manifestMemo = { manifest, at: now };
  return manifest;
}

function corsHeaders(env: Env, origin: string | null): Record<string, string> {
  // Fail-closed on empty/missing env var (Chip R4-H-2): a misconfigured
  // deploy that blanked CORS_ALLOWLIST must NOT silently flip to wildcard.
  // Emit no Access-Control-Allow-Origin header; browsers enforce deny.
  const raw = (env.CORS_ALLOWLIST || "").trim();
  const allowed = raw ? raw.split(",").map(s => s.trim()).filter(Boolean) : [];
  const base: Record<string, string> = {
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, If-None-Match, X-Vault-Release",
    Vary: "Origin",
  };
  if (allowed.length === 0) return base;   // no ACAO header — deny by omission
  if (origin && allowed.includes(origin)) {
    base["Access-Control-Allow-Origin"] = origin;
  }
  // If origin didn't match, still emit no ACAO — browsers deny.
  return base;
}

function json(
  body: unknown,
  init: ResponseInit & {
    etag?: string;
    degradedReason?: string;
    releaseId?: string;
    cacheControl?: string;
  } = {},
): Response {
  const headers = new Headers({
    "Content-Type": "application/json; charset=utf-8",
    ...(init.headers as Record<string, string> | undefined),
  });
  if (init.etag) headers.set("ETag", init.etag);
  if (init.releaseId) headers.set("X-Vault-Release", init.releaseId);
  if (init.degradedReason) headers.set("X-Vault-Degraded", init.degradedReason);
  if (init.cacheControl) headers.set("Cache-Control", init.cacheControl);
  return new Response(JSON.stringify(body), { status: init.status ?? 200, headers });
}

function cacheControl(ttl: number): string {
  return `public, max-age=${ttl}, stale-while-revalidate=${ttl * 2}`;
}

function encodeCursor(c: Cursor): string {
  return btoa(JSON.stringify(c));
}

function decodeCursor(s: string | null): Cursor | null {
  if (!s) return null;
  try {
    const parsed = JSON.parse(atob(s)) as unknown;
    if (typeof parsed !== "object" || parsed === null) return null;
    const c = parsed as Partial<Cursor>;
    if (typeof c.after_id !== "string" || typeof c.filter_hash !== "string") return null;
    return { after_id: c.after_id, filter_hash: c.filter_hash };
  } catch {
    return null;
  }
}

// ─── Cloudflare Cache API (B.1) ──────────────────────────────────────────────

/**
 * Build a stable Cache API key keyed by release_id so a deploy atomically
 * invalidates all cached entries (ARCHITECTURE.md §6.1.1 correctness boundary).
 *
 * The release_id is injected into the cache URL as a path prefix so the CF
 * edge cache treats different releases as disjoint namespaces.
 */
function cacheKey(req: Request, releaseId: string): Request {
  const url = new URL(req.url);
  url.pathname = `/__vault__/${releaseId}${url.pathname}`;
  return new Request(url.toString(), { method: "GET", headers: req.headers });
}

async function cachedOrCompute(
  req: Request,
  env: Env,
  ctx: ExecutionContext,
  ttlSec: number,
  compute: () => Promise<Response>,
  opts: { useCache: boolean } = { useCache: true },
): Promise<Response> {
  // Graceful degradation when the CF Cache API isn't available (Node test
  // env, vault-api local shim, or a runtime regression) — always compute
  // and skip caching. Prevents crashing the handler on missing globals.
  if (!opts.useCache || typeof caches === "undefined" || !caches?.default) {
    return compute();
  }
  const manifest = await getManifest(env);
  const key = cacheKey(req, manifest.release_id);
  const cached = await caches.default.match(key);
  if (cached) {
    // Clone so callers can still read the body.
    return new Response(cached.body, cached);
  }
  const res = await compute();
  // Only cache 2xx + ok responses; never cache degraded responses that
  // would otherwise poison the release-keyed namespace.
  if (res.ok && !res.headers.get("X-Vault-Degraded")) {
    const cacheable = new Response(res.body, res);
    cacheable.headers.set("Cache-Control", cacheControl(ttlSec));
    ctx.waitUntil(caches.default.put(key, cacheable.clone()));
    return cacheable;
  }
  return res;
}

// ─── Endpoint handlers ───────────────────────────────────────────────────────

async function handleManifest(env: Env, req: Request): Promise<Response> {
  const origin = req.headers.get("Origin");
  const manifest = await getManifest(env);
  const etag = `"manifest:${manifest.release_id}:${manifest.release_hash.slice(0, 16)}"`;
  if (req.headers.get("If-None-Match") === etag) {
    return new Response(null, {
      status: 304,
      headers: { ETag: etag, ...corsHeaders(env, origin) },
    });
  }
  return json(manifest, {
    etag,
    releaseId: manifest.release_id,
    degradedReason: manifest.schema_fingerprint_ok ? undefined : "schema-fingerprint-mismatch",
    cacheControl: cacheControl(Number.parseInt(env.CACHE_TTL_MANIFEST, 10)),
    headers: corsHeaders(env, origin),
  });
}

async function handleQuestionById(env: Env, req: Request, id: string): Promise<Response> {
  const origin = req.headers.get("Origin");
  const row = await env.DB
    .prepare("SELECT * FROM questions WHERE id = ?")
    .bind(id)
    .first<QuestionRow>();
  if (!row) {
    return json({ error: "not-found", id }, { status: 404, headers: corsHeaders(env, origin) });
  }
  const manifest = await getManifest(env);
  const etag = `"${manifest.release_id}:q:${row.content_hash}"`;
  if (req.headers.get("If-None-Match") === etag) {
    return new Response(null, {
      status: 304,
      headers: { ETag: etag, ...corsHeaders(env, origin) },
    });
  }
  return json(row, {
    etag,
    releaseId: manifest.release_id,
    cacheControl: cacheControl(Number.parseInt(env.CACHE_TTL_QUESTION, 10)),
    headers: corsHeaders(env, origin),
  });
}

/** Compute a short stable hash of the filter param set for cursor binding. */
async function filterHash(params: URLSearchParams): Promise<string> {
  const keys = ["track", "level", "zone", "topic", "status"];
  const parts = keys.map(k => `${k}=${params.get(k) ?? ""}`);
  const buf = new TextEncoder().encode(parts.join("&"));
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest))
    .slice(0, 8)
    .map(b => b.toString(16).padStart(2, "0"))
    .join("");
}

async function handleQuestions(env: Env, req: Request, url: URL): Promise<Response> {
  const origin = req.headers.get("Origin");
  const params = url.searchParams;
  const expectedFilterHash = await filterHash(params);
  const cursor = decodeCursor(params.get("cursor"));
  if (cursor && cursor.filter_hash !== expectedFilterHash) {
    return json(
      { error: "cursor-filter-mismatch", detail: "cursor was minted under different filter params" },
      { status: 400, headers: corsHeaders(env, origin) },
    );
  }
  const limit = Math.min(Number.parseInt(params.get("limit") ?? "50", 10), 200);

  const where: string[] = [];
  const binds: unknown[] = [];
  for (const key of ["track", "level", "zone", "topic", "status"] as const) {
    const v = params.get(key);
    if (v) {
      where.push(`${key} = ?`);
      binds.push(v);
    }
  }

  // Keyset pagination (Chip R3-H2 fix).
  if (cursor?.after_id) {
    where.push("id > ?");
    binds.push(cursor.after_id);
  }

  const whereSql = where.length ? `WHERE ${where.join(" AND ")}` : "";

  const rows = await env.DB
    .prepare(
      `SELECT id, title, topic, track, level, zone, status, content_hash
       FROM questions ${whereSql} ORDER BY id LIMIT ?`,
    )
    .bind(...binds, limit)
    .all<Partial<QuestionRow>>();

  const items = rows.results ?? [];
  const nextCursor: Cursor | null = items.length === limit
    ? { after_id: items[items.length - 1].id!, filter_hash: expectedFilterHash }
    : null;

  const manifest = await getManifest(env);
  return json(
    { items, next_cursor: nextCursor ? encodeCursor(nextCursor) : null },
    {
      releaseId: manifest.release_id,
      cacheControl: cacheControl(600),
      headers: corsHeaders(env, origin),
    },
  );
}

const MAX_SEARCH_Q_CHARS = 100;

async function handleSearch(env: Env, req: Request, url: URL): Promise<Response> {
  const origin = req.headers.get("Origin");
  const qRaw = (url.searchParams.get("q") ?? "").trim();
  const limit = Math.min(Number.parseInt(url.searchParams.get("limit") ?? "20", 10), 50);
  if (!qRaw) {
    return json({ error: "missing-query-q" }, { status: 400, headers: corsHeaders(env, origin) });
  }
  // Chip R4-C-1 hardening: cap query length (prevents NEAR/OR DoS).
  if (qRaw.length > MAX_SEARCH_Q_CHARS) {
    return json(
      { error: "query-too-long", max_chars: MAX_SEARCH_Q_CHARS },
      { status: 400, headers: corsHeaders(env, origin) },
    );
  }
  // Chip R7-M-5 + Gemini R9: memoized FTS5 probe at module level.
  // Previously probed sqlite_master on every /search request — 1 D1 row-read
  // per search, directly undoing R5-H-1's manifest-memo cost fix.
  // ftsProbed resets on release_id change (FTS5 presence only changes across
  // releases).
  if (ftsProbed === undefined) {
    try {
      const probe = await env.DB.prepare(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='questions_fts' LIMIT 1"
      ).first<{ name: string } | null>();
      ftsProbed = Boolean(probe);
    } catch {
      ftsProbed = false;
    }
  }
  const ftsAvailable = ftsProbed;

  let rows;
  if (ftsAvailable) {
    // Chip R4-C-1 hardening: reject FTS5 operator keywords (NEAR/AND/OR/NOT)
    // in raw input — the char-class strip keeps them intact as bare words.
    // Then wrap the whole user term in a PHRASE literal so anything that
    // survived becomes a literal to match, not an operator.
    if (/\b(NEAR|AND|OR|NOT)\b/i.test(qRaw)) {
      return json(
        { error: "reserved-token-in-query" },
        { status: 400, headers: corsHeaders(env, origin) },
      );
    }
    const sanitized = qRaw
      .replace(/[^\w\s]/g, " ")          // strip FTS5 operator chars
      .replace(/\s+/g, " ")
      .trim();
    if (!sanitized) {
      return json({ results: [], query: qRaw, fts: true },
                  { headers: corsHeaders(env, origin) });
    }
    // Wrap as a phrase literal. Double any embedded " per FTS5 syntax
    // (defensive; we already stripped non-word chars).
    const phrase = `"${sanitized.replace(/"/g, '""')}"`;
    rows = await env.DB
      .prepare(
        `SELECT q.id, q.title, q.topic, q.track, q.level, q.zone,
                snippet(questions_fts, 1, '<mark>', '</mark>', '…', 16) AS snippet
         FROM questions_fts
         JOIN questions q ON q.rowid = questions_fts.rowid
         WHERE questions_fts MATCH ?
         ORDER BY rank
         LIMIT ?`,
      )
      .bind(phrase, limit)
      .all();
  } else {
    const pattern = `%${qRaw}%`;
    rows = await env.DB
      .prepare(
        `SELECT id, title, topic, track, level, zone
         FROM questions
         WHERE title LIKE ? OR scenario LIKE ? OR realistic_solution LIKE ?
         ORDER BY id LIMIT ?`,
      )
      .bind(pattern, pattern, pattern, limit)
      .all();
  }
  const manifest = await getManifest(env);
  return json(
    { results: rows.results ?? [], query: qRaw, fts: ftsAvailable },
    {
      releaseId: manifest.release_id,
      cacheControl: cacheControl(Number.parseInt(env.CACHE_TTL_SEARCH, 10)),
      headers: corsHeaders(env, origin),
    },
  );
}

async function handleChain(env: Env, req: Request, chainId: string): Promise<Response> {
  const origin = req.headers.get("Origin");
  const chain = await env.DB
    .prepare("SELECT id, name, topic FROM chains WHERE id = ?")
    .bind(chainId)
    .first<{ id: string; name: string | null; topic: string | null }>();
  if (!chain) {
    return json({ error: "not-found", chain_id: chainId }, { status: 404, headers: corsHeaders(env, origin) });
  }
  const questions = await env.DB
    .prepare(
      `SELECT cq.position, q.id, q.title, q.topic, q.track, q.level, q.zone
       FROM chain_questions cq
       JOIN questions q ON q.id = cq.question_id
       WHERE cq.chain_id = ?
       ORDER BY cq.position`,
    )
    .bind(chainId)
    .all();
  const manifest = await getManifest(env);
  return json(
    { chain, questions: questions.results ?? [] },
    {
      releaseId: manifest.release_id,
      cacheControl: cacheControl(3600),
      headers: corsHeaders(env, origin),
    },
  );
}

async function handleTaxonomy(env: Env, req: Request): Promise<Response> {
  const origin = req.headers.get("Origin");
  const topics = await env.DB
    .prepare("SELECT id, name, description, area, prerequisites_json, tracks_json, question_count FROM taxonomy ORDER BY area, id")
    .all<{
      id: string; name: string; description: string | null;
      area: string; prerequisites_json: string | null;
      tracks_json: string | null; question_count: number;
    }>();
  const zones = await env.DB
    .prepare("SELECT id, description, skills_json, levels_json FROM zones ORDER BY id")
    .all<{ id: string; description: string | null; skills_json: string | null; levels_json: string | null }>();

  // Group topics by area for the response.
  const areas: Record<string, unknown[]> = {};
  for (const t of topics.results ?? []) {
    if (!areas[t.area]) areas[t.area] = [];
    areas[t.area].push({
      id: t.id,
      name: t.name,
      description: t.description,
      prerequisites: t.prerequisites_json ? JSON.parse(t.prerequisites_json) : [],
      tracks: t.tracks_json ? JSON.parse(t.tracks_json) : [],
      question_count: t.question_count,
    });
  }

  const manifest = await getManifest(env);
  return json(
    {
      areas,
      zones: (zones.results ?? []).map(z => ({
        id: z.id,
        description: z.description,
        skills: z.skills_json ? JSON.parse(z.skills_json) : [],
        levels: z.levels_json ? JSON.parse(z.levels_json) : [],
      })),
      topic_count: (topics.results ?? []).length,
      area_count: Object.keys(areas).length,
    },
    {
      releaseId: manifest.release_id,
      cacheControl: cacheControl(Number.parseInt(env.CACHE_TTL_TAXONOMY, 10)),
      headers: corsHeaders(env, origin),
    },
  );
}

export default {
  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(req.url);
    const origin = req.headers.get("Origin");

    if (req.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(env, origin) });
    }
    if (req.method !== "GET") {
      return json({ error: "method-not-allowed" }, { status: 405, headers: corsHeaders(env, origin) });
    }

    // ── Rate limiting (B.4) ──────────────────────────────────────────────
    const rlClass: "default" | "search" = url.pathname === "/search" ? "search" : "default";
    const rl = await checkRateLimit(env, req, rlClass);
    if (!rl.ok) {
      return json(
        { error: "rate-limited", retry_after_seconds: rl.retryAfterSec },
        {
          status: 429,
          headers: {
            "Retry-After": String(rl.retryAfterSec),
            ...corsHeaders(env, origin),
          },
        },
      );
    }

    try {
      // ── Cache API wrap (B.1) ────────────────────────────────────────────
      // GET endpoints that are safely cacheable route through caches.default
      // keyed by release_id. Non-cacheable endpoints (dynamic /search, or
      // cursored /questions where cursor is user-supplied) skip the cache.
      if (url.pathname === "/manifest") {
        return await cachedOrCompute(
          req, env, ctx,
          Number.parseInt(env.CACHE_TTL_MANIFEST, 10),
          () => handleManifest(env, req),
        );
      }
      if (url.pathname.startsWith("/questions/")) {
        const id = url.pathname.slice("/questions/".length);
        return await cachedOrCompute(
          req, env, ctx,
          Number.parseInt(env.CACHE_TTL_QUESTION, 10),
          () => handleQuestionById(env, req, id),
        );
      }
      if (url.pathname === "/questions") {
        // Cache paginated list responses — cache key includes full URL so
        // different filter/cursor combos are distinct entries.
        return await cachedOrCompute(
          req, env, ctx, 600,
          () => handleQuestions(env, req, url),
        );
      }
      if (url.pathname === "/search") {
        return await cachedOrCompute(
          req, env, ctx,
          Number.parseInt(env.CACHE_TTL_SEARCH, 10),
          () => handleSearch(env, req, url),
        );
      }
      if (url.pathname.startsWith("/chains/")) {
        const chainId = url.pathname.slice("/chains/".length);
        return await cachedOrCompute(
          req, env, ctx, 3600,
          () => handleChain(env, req, chainId),
        );
      }
      if (url.pathname === "/taxonomy") {
        return await cachedOrCompute(
          req, env, ctx,
          Number.parseInt(env.CACHE_TTL_TAXONOMY, 10),
          () => handleTaxonomy(env, req),
        );
      }
      if (url.pathname === "/stats") {
        const row = await env.DB
          .prepare("SELECT COUNT(*) AS n FROM questions")
          .first<{ n: number }>();
        return json(
          { count: row?.n ?? 0 },
          { cacheControl: cacheControl(3600), headers: corsHeaders(env, origin) },
        );
      }
      return json({ error: "unknown-endpoint" }, { status: 404, headers: corsHeaders(env, origin) });
    } catch (e) {
      // Chip R4-H-2: never echo exception detail to clients — can leak SQL
      // fragments, D1 internals, etc. Log to console for tail visibility,
      // return generic message.
      console.error("worker-internal-error", { path: url.pathname, error: String(e) });
      return json(
        { error: "internal" },
        { status: 500, headers: corsHeaders(env, origin) },
      );
    }
  },
};
