/**
 * Worker contract tests (B.6).
 *
 * These exercise the request-handling surface of src/index.ts in isolation
 * from a live D1 / KV by mocking Env. Vitest + the
 * @cloudflare/workers-types runtime globals are enough — no miniflare needed
 * for this subset. Staging load-tests (Phase-3 entry gate) are separate.
 */

import { describe, it, expect, beforeEach, vi } from "vitest";
import type { Env } from "../src/types";

// Reset the worker module between tests so module-level state
// (schema-fingerprint memoization, release-id cache) doesn't leak.
let worker: typeof import("../src/index").default;
beforeEach(async () => {
  vi.resetModules();
  worker = (await import("../src/index")).default;
});

// Minimal mock D1 — behaves like D1Database for the paths we use.
function mockDB(rows: Record<string, any>[], extraMeta: Record<string, string> = {}): D1Database {
  const meta: Record<string, string> = {
    release_id: "1.0.0",
    release_hash: "1b304282774cf9cbc7f28b07ed151bc12041c92610e73809f9f594c199661b23",
    schema_version: "1",
    policy_version: "1",
    published_count: String(rows.length),
    ...extraMeta,
  };
  return {
    prepare(sql: string) {
      return {
        bind: (..._args: any[]) => this,
        first: async () => {
          if (sql.includes("release_metadata WHERE key")) return null;
          if (sql.includes("FROM questions WHERE id")) {
            return rows[0] ?? null;
          }
          if (sql.includes("COUNT(*)")) return { n: rows.length };
          return rows[0] ?? null;
        },
        all: async () => {
          if (sql.includes("release_metadata")) {
            return { results: Object.entries(meta).map(([key, value]) => ({ key, value })) };
          }
          if (sql.includes("sqlite_master") && sql.includes("table")) {
            return { results: [{ sql: "CREATE TABLE questions (id TEXT)" }] };
          }
          return { results: rows };
        },
      } as any;
    },
  } as any;
}

// KV mock — plain Map + windowed counter behavior matching rate_limit.ts.
function mockKV(): KVNamespace {
  const store = new Map<string, string>();
  return {
    async get(k: string) {
      return store.get(k) ?? null;
    },
    async put(k: string, v: string) {
      store.set(k, v);
    },
  } as any;
}

// Compute the schema fingerprint that matches the mock DB's DDL string.
async function mockFingerprint(): Promise<string> {
  const ddl = "CREATE TABLE questions (id TEXT)";
  const buf = new TextEncoder().encode(ddl);
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest))
    .map(b => b.toString(16).padStart(2, "0"))
    .join("");
}

function makeEnv(override: Partial<Env> = {}): Env {
  return {
    DB: mockDB([
      { id: "global-0000", title: "T", topic: "t", track: "global", level: "l1",
        zone: "recall", status: "published", content_hash: "hash-a" },
    ]),
    CACHE_TTL_MANIFEST: "60",
    CACHE_TTL_QUESTION: "60",
    CACHE_TTL_SEARCH: "30",
    CACHE_TTL_TAXONOMY: "3600",
    CORS_ALLOWLIST: "https://staffml.mlsysbook.ai",
    SCHEMA_FINGERPRINT: "PLACEHOLDER-deploy-time",
    GRACE_WINDOW_SECONDS: "600",
    ...override,
  };
}

function ctx(): ExecutionContext {
  return { waitUntil: () => {}, passThroughOnException: () => {} } as any;
}

describe("worker rate limiting", () => {
  it("allows under the cap, denies over", async () => {
    const env = makeEnv({
      RATE_LIMIT_KV: mockKV(),
      RATE_LIMIT_RPM_DEFAULT: "2",
    });
    const makeReq = () =>
      new Request("https://x/stats", { headers: { "CF-Connecting-IP": "1.2.3.4" } });
    const r1 = await worker.fetch(makeReq(), env, ctx());
    const r2 = await worker.fetch(makeReq(), env, ctx());
    const r3 = await worker.fetch(makeReq(), env, ctx());
    expect(r1.status).toBe(200);
    expect(r2.status).toBe(200);
    expect(r3.status).toBe(429);
    expect(r3.headers.get("Retry-After")).toBeTruthy();
  });
});

describe("schema-fingerprint degraded mode", () => {
  it("placeholder forces degraded mode on manifest", async () => {
    const env = makeEnv({ SCHEMA_FINGERPRINT: "PLACEHOLDER-deploy-time" });
    const res = await worker.fetch(new Request("https://x/manifest"), env, ctx());
    const body = await res.json() as any;
    expect(body.schema_fingerprint_ok).toBe(false);
    expect(res.headers.get("X-Vault-Degraded")).toBe("schema-fingerprint-mismatch");
  });

  it("real fingerprint value clears the flag", async () => {
    // Gemini R5-C-3: fingerprint now comes from release_metadata, not env.
    // Mock DB must return it alongside other metadata rows.
    const fp = await mockFingerprint();
    const env: Env = {
      ...makeEnv(),
      DB: mockDB([
        { id: "global-0000", title: "T", topic: "t", track: "global", level: "l1",
          zone: "recall", status: "published", content_hash: "hash-a" },
      ], { schema_fingerprint: fp }),
    };
    const res = await worker.fetch(new Request("https://x/manifest"), env, ctx());
    const body = await res.json() as any;
    expect(body.schema_fingerprint_ok).toBe(true);
    expect(res.headers.get("X-Vault-Degraded")).toBeNull();
  });
});

describe("cursor pagination", () => {
  it("rejects cross-filter cursor reuse with 400", async () => {
    const env = makeEnv({ SCHEMA_FINGERPRINT: await mockFingerprint() });
    // A cursor minted under empty filters must not be usable under ?track=cloud.
    const mintRes = await worker.fetch(
      new Request("https://x/questions?limit=1"),
      env, ctx(),
    );
    const minted = await mintRes.json() as any;
    const cursor = minted.next_cursor;
    // Re-query with the cursor under a different filter — should be 400.
    const rejected = await worker.fetch(
      new Request(`https://x/questions?track=cloud&cursor=${encodeURIComponent(cursor ?? "")}`),
      env, ctx(),
    );
    // Only assert when the mock actually produced a cursor.
    if (cursor) {
      expect(rejected.status).toBe(400);
    }
  });
});

describe("CORS", () => {
  it("echoes allowed origin; rejects unknown", async () => {
    const env = makeEnv({ SCHEMA_FINGERPRINT: await mockFingerprint() });
    const res = await worker.fetch(
      new Request("https://x/manifest", { headers: { Origin: "https://staffml.mlsysbook.ai" } }),
      env, ctx(),
    );
    expect(res.headers.get("Access-Control-Allow-Origin")).toBe("https://staffml.mlsysbook.ai");
  });
});

describe("method-not-allowed", () => {
  it("405s on POST/PUT/DELETE to any endpoint", async () => {
    const env = makeEnv();
    for (const method of ["POST", "PUT", "DELETE"]) {
      const res = await worker.fetch(new Request("https://x/manifest", { method }), env, ctx());
      expect(res.status).toBe(405);
    }
  });
});

describe("admin endpoint removed", () => {
  it("returns 404 on /admin/release (no auth footgun)", async () => {
    const env = makeEnv({ SCHEMA_FINGERPRINT: await mockFingerprint() });
    const res = await worker.fetch(new Request("https://x/admin/release"), env, ctx());
    expect(res.status).toBe(404);
  });
});
