/**
 * StaffML Vault Worker — edge API over D1.
 *
 * Implements ARCHITECTURE.md §10 with v2.1 refinements:
 * - X-Vault-Release is INFORMATIONAL (not hard-reject) — soft SLI signal.
 * - schema_fingerprint verified against sqlite_master at cold start; on
 *   mismatch, serves cached-only with X-Vault-Degraded header (no 5xx).
 * - Cache keys include release_id for atomic POP invalidation on deploy.
 * - 10-minute cross-release grace window during propagation.
 * - All endpoints: ETag + Cache-Control SWR-friendly.
 */

import type { Cursor, DegradedReason, Env, Manifest, QuestionRow } from "./types";

// Cold-start memo for schema fingerprint check (per-instance; warm workers skip).
let schemaOk: boolean | undefined;

async function checkSchemaFingerprint(env: Env): Promise<boolean> {
  if (schemaOk !== undefined) return schemaOk;
  // Fail-closed on placeholder (Chip R3-C1, Dean R3-NH-3): a deploy that
  // forgot to substitute the real fingerprint MUST enter degraded mode.
  // Silent auto-pass on placeholder was the v2.1 Critical regression.
  if (!env.SCHEMA_FINGERPRINT || env.SCHEMA_FINGERPRINT === "PLACEHOLDER-deploy-time") {
    schemaOk = false;
    return schemaOk;
  }
  try {
    const result = await env.DB.prepare(
      // Include triggers per ARCHITECTURE.md §10.1 (Dean R3-NH-4 adjacent):
      // FTS5 materializes as tables + shadow tables + triggers. Omitting
      // triggers means FTS5-aware releases hash differently from non-FTS5.
      "SELECT sql FROM sqlite_master WHERE type IN ('table','index','trigger') AND name NOT LIKE 'sqlite_%' ORDER BY name"
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
    schemaOk = hex === env.SCHEMA_FINGERPRINT;
  } catch {
    schemaOk = false;
  }
  return schemaOk;
}

/**
 * Forget the cached schema-fingerprint decision so the next request re-checks.
 * Called on release_id change (R3-NH-4): every deploy advances release_id in
 * release_metadata, which surfaces via /manifest. That natural cadence keeps
 * the check fresh without per-request cost.
 */
let lastSeenRelease: string | null = null;
function maybeInvalidateSchemaCache(releaseId: string | undefined): void {
  if (!releaseId) return;
  if (lastSeenRelease !== null && lastSeenRelease !== releaseId) {
    schemaOk = undefined;
  }
  lastSeenRelease = releaseId;
}

async function getManifest(env: Env): Promise<Manifest> {
  const rows = await env.DB.prepare("SELECT key, value FROM release_metadata").all<{
    key: string;
    value: string;
  }>();
  const meta: Record<string, string> = {};
  for (const r of rows.results ?? []) meta[r.key] = r.value;
  const release_id = meta.release_id ?? "unknown";
  maybeInvalidateSchemaCache(release_id);
  return {
    release_id,
    release_hash: meta.release_hash ?? "",
    schema_version: meta.schema_version ?? "1",
    policy_version: meta.policy_version ?? "1",
    published_count: Number.parseInt(meta.published_count ?? "0", 10),
    schema_fingerprint_ok: await checkSchemaFingerprint(env),
  };
}

function corsHeaders(env: Env, origin: string | null): Record<string, string> {
  const allowed = (env.CORS_ALLOWLIST || "").split(",");
  const ok = origin && allowed.includes(origin);
  return {
    "Access-Control-Allow-Origin": ok ? origin : allowed[0] || "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, If-None-Match, X-Vault-Release",
    Vary: "Origin",
  };
}

function json(
  body: unknown,
  init: ResponseInit & { etag?: string; degradedReason?: DegradedReason; releaseId?: string } = {},
): Response {
  const headers = new Headers({
    "Content-Type": "application/json; charset=utf-8",
    ...(init.headers as Record<string, string> | undefined),
  });
  if (init.etag) headers.set("ETag", init.etag);
  if (init.releaseId) headers.set("X-Vault-Release", init.releaseId);
  if (init.degradedReason) headers.set("X-Vault-Degraded", init.degradedReason);
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
    return JSON.parse(atob(s)) as Cursor;
  } catch {
    return null;
  }
}

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
    headers: {
      "Cache-Control": cacheControl(Number.parseInt(env.CACHE_TTL_MANIFEST, 10)),
      ...corsHeaders(env, origin),
    },
  });
}

async function handleQuestionById(env: Env, req: Request, id: string): Promise<Response> {
  const origin = req.headers.get("Origin");
  const row = await env.DB.prepare("SELECT * FROM questions WHERE id = ?").bind(id).first<QuestionRow>();
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
    headers: {
      "Cache-Control": cacheControl(Number.parseInt(env.CACHE_TTL_QUESTION, 10)),
      ...corsHeaders(env, origin),
    },
  });
}

async function handleQuestions(env: Env, req: Request, url: URL): Promise<Response> {
  const origin = req.headers.get("Origin");
  const params = url.searchParams;
  const cursor = decodeCursor(params.get("cursor"));
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

  const whereSql = where.length ? `WHERE ${where.join(" AND ")}` : "";
  const offset = cursor?.offset ?? 0;

  const rows = await env.DB
    .prepare(
      `SELECT id, title, topic, track, level, zone, status, content_hash
       FROM questions ${whereSql} ORDER BY id LIMIT ? OFFSET ?`,
    )
    .bind(...binds, limit, offset)
    .all<Partial<QuestionRow>>();

  const filterHash = (where.join("&") || "all");
  const nextCursor: Cursor | null = (rows.results?.length ?? 0) === limit
    ? { offset: offset + limit, filter_hash: filterHash }
    : null;

  const manifest = await getManifest(env);
  return json(
    {
      items: rows.results ?? [],
      next_cursor: nextCursor ? encodeCursor(nextCursor) : null,
    },
    {
      releaseId: manifest.release_id,
      headers: {
        "Cache-Control": cacheControl(600),
        ...corsHeaders(env, origin),
      },
    },
  );
}

async function handleSearch(env: Env, req: Request, url: URL): Promise<Response> {
  const origin = req.headers.get("Origin");
  const q = url.searchParams.get("q") ?? "";
  const limit = Math.min(Number.parseInt(url.searchParams.get("limit") ?? "20", 10), 50);
  if (!q) {
    return json({ error: "missing-query-q" }, { status: 400, headers: corsHeaders(env, origin) });
  }
  // Phase 3: LIKE search; FTS5 virtual table adds in Phase 3.x once indexed.
  const pattern = `%${q}%`;
  const rows = await env.DB
    .prepare(
      `SELECT id, title, topic, track, level, zone
       FROM questions
       WHERE title LIKE ? OR scenario LIKE ? OR realistic_solution LIKE ?
       ORDER BY id LIMIT ?`,
    )
    .bind(pattern, pattern, pattern, limit)
    .all();
  const manifest = await getManifest(env);
  return json(
    { results: rows.results ?? [], query: q },
    {
      releaseId: manifest.release_id,
      headers: {
        "Cache-Control": cacheControl(Number.parseInt(env.CACHE_TTL_SEARCH, 10)),
        ...corsHeaders(env, origin),
      },
    },
  );
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    const origin = req.headers.get("Origin");

    if (req.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(env, origin) });
    }
    if (req.method !== "GET") {
      return json({ error: "method-not-allowed" }, { status: 405, headers: corsHeaders(env, origin) });
    }

    try {
      if (url.pathname === "/manifest") return await handleManifest(env, req);
      if (url.pathname === "/questions") return await handleQuestions(env, req, url);
      if (url.pathname.startsWith("/questions/")) {
        const id = url.pathname.slice("/questions/".length);
        return await handleQuestionById(env, req, id);
      }
      if (url.pathname === "/search") return await handleSearch(env, req, url);
      if (url.pathname === "/stats") {
        const row = await env.DB.prepare("SELECT COUNT(*) AS n FROM questions").first<{ n: number }>();
        return json({ count: row?.n ?? 0 }, { headers: corsHeaders(env, origin) });
      }
      return json({ error: "unknown-endpoint" }, { status: 404, headers: corsHeaders(env, origin) });
    } catch (e) {
      return json(
        { error: "internal", detail: String(e) },
        { status: 500, headers: corsHeaders(env, origin) },
      );
    }
  },
};
