"use client";

/**
 * CorpusProvider — hybrid data layer.
 *
 * The bundled `corpus-summary.json` is the primary data source for
 * synchronous operations (getQuestions, getQuestionsByFilter, taxonomy,
 * navigation). Heavy fields (scenario, details prose) come from the
 * Cloudflare Worker via the shared transport in `vault-fetch.ts`.
 *
 * The Worker enhances two specific operations:
 *
 * 1. **Search** — FTS5 full-text search via /search endpoint replaces the
 *    client-side O(n) string matching.
 * 2. **Service worker registration** — enables offline caching of API
 *    responses for the per-question detail fetches.
 *
 * Mode + base URL resolution is centralized in `vault-config.ts` so this
 * provider and the non-React fetchers in `corpus.ts` always agree on where
 * data comes from. In particular, production leaves NEXT_PUBLIC_VAULT_API
 * unset and relies on the worker default — so search and the offline SW stay
 * enabled in production rather than silently falling back to client search.
 */

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import type { Question } from "./corpus";
import { getVaultApiBase, getVaultMode } from "./vault-config";
import { vaultFetchJson } from "./vault-fetch";

interface VaultState {
  apiBase: string | null;
  ready: boolean;
}

const VaultContext = createContext<VaultState>({ apiBase: null, ready: false });

export function useVault(): VaultState {
  return useContext(VaultContext);
}

/**
 * Search via the Worker's FTS5 endpoint. Returns results ranked by relevance.
 * Falls back to null if the API is unavailable (caller then uses client-side
 * search over the bundled summary). Routes through the shared transport so it
 * shares retry + circuit-breaker state with the per-question detail fetches.
 */
export async function vaultSearch(
  apiBase: string,
  query: string,
  limit = 20,
): Promise<Question[] | null> {
  try {
    const params = new URLSearchParams({ q: query, limit: String(limit) });
    // The /search endpoint returns {results, query, fts}.
    // Each result has: id, title, topic, track, level, zone, snippet (if FTS).
    // These are summary-only — callers merge with bundled corpus for full data.
    const data = await vaultFetchJson<{ results?: Question[] }>(
      `${apiBase}/search?${params}`,
      { timeoutMs: 5_000 },
    );
    return data.results ?? null;
  } catch {
    return null;
  }
}

/**
 * Fetch a single question's full data from the Worker API.
 */
export async function vaultGetQuestion(
  apiBase: string,
  id: string,
): Promise<Record<string, unknown> | null> {
  try {
    return await vaultFetchJson<Record<string, unknown>>(
      `${apiBase}/questions/${encodeURIComponent(id)}`,
      { timeoutMs: 5_000 },
    );
  } catch {
    return null;
  }
}

export function CorpusProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<VaultState>({ apiBase: null, ready: false });

  useEffect(() => {
    // Static mode (local dev / offline): no worker to talk to.
    if (getVaultMode() === "static") {
      setState({ apiBase: null, ready: true });
      return;
    }

    const apiBase = getVaultApiBase();
    if (!apiBase) {
      setState({ apiBase: null, ready: true });
      return;
    }

    let cancelled = false;
    // Probe the API to confirm it's reachable before enabling worker-backed
    // search + the offline service worker. On any failure we degrade to
    // client-side search; per-question hydration in corpus.ts has its own
    // resilient path.
    vaultFetchJson(`${apiBase}/manifest`, { timeoutMs: 5_000, retries: 1 })
      .then(() => {
        if (cancelled) return;
        setState({ apiBase, ready: true });
        registerVaultServiceWorker(apiBase);
      })
      .catch(() => {
        if (!cancelled) setState({ apiBase: null, ready: true });
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <VaultContext.Provider value={state}>
      {children}
    </VaultContext.Provider>
  );
}

/**
 * Register the offline service worker and tell it which API origin to cache.
 *
 * The message must go to an ACTIVE worker. On a first-ever registration the
 * worker is still installing, so `registration.active` is null — posting to it
 * silently drops the origin and offline caching never learns the API host.
 * `navigator.serviceWorker.ready` resolves only once a worker is active, so we
 * post through it.
 */
function registerVaultServiceWorker(apiBase: string): void {
  if (!("serviceWorker" in navigator)) return;
  navigator.serviceWorker
    .register(`${process.env.NEXT_PUBLIC_BASE_PATH || ""}/sw.js`)
    .then(() => navigator.serviceWorker.ready)
    .then((reg) => {
      reg.active?.postMessage({ type: "SET_VAULT_API_ORIGIN", origin: apiBase });
    })
    .catch(() => {/* SW registration failure is non-fatal */});
}
