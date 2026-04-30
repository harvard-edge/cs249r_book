"use client";

/**
 * CorpusProvider — hybrid data layer.
 *
 * The bundled `corpus-summary.json` is the primary data source for
 * synchronous operations (getQuestions, getQuestionsByFilter, taxonomy,
 * navigation). Heavy fields (scenario, details prose) come from the
 * Cloudflare Worker via vault-api.ts.
 *
 * The Worker enhances two specific operations:
 *
 * 1. **Search** — FTS5 full-text search via /search endpoint replaces the
 *    client-side O(n) string matching.
 * 2. **Service worker registration** — enables offline caching of API
 *    responses for the per-question detail fetches.
 *
 * NEXT_PUBLIC_VAULT_FALLBACK=static is an OPT-IN local-dev affordance for
 * working without a reachable Worker (requires `vault build --local-json`
 * to materialize corpus.json). Production never sets it.
 */

import { createContext, useContext, useEffect, useState, useCallback, type ReactNode } from "react";
import type { Question } from "./corpus";

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
 * Falls back to null if the API is unavailable.
 */
export async function vaultSearch(
  apiBase: string,
  query: string,
  limit = 20,
): Promise<Question[] | null> {
  try {
    const params = new URLSearchParams({ q: query, limit: String(limit) });
    const res = await fetch(`${apiBase}/search?${params}`, {
      signal: AbortSignal.timeout(5_000),
    });
    if (!res.ok) return null;
    const data = await res.json();
    // The /search endpoint returns {results, query, fts}.
    // Each result has: id, title, topic, track, level, zone, snippet (if FTS).
    // These are summary-only — callers merge with bundled corpus for full data.
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
    const res = await fetch(`${apiBase}/questions/${encodeURIComponent(id)}`, {
      signal: AbortSignal.timeout(5_000),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export function CorpusProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<VaultState>({ apiBase: null, ready: false });

  useEffect(() => {
    const fallbackFlag = process.env.NEXT_PUBLIC_VAULT_FALLBACK?.toLowerCase();
    if (fallbackFlag === "static") {
      setState({ apiBase: null, ready: true });
      return;
    }

    const apiBase = process.env.NEXT_PUBLIC_VAULT_API ?? null;
    if (!apiBase) {
      setState({ apiBase: null, ready: true });
      return;
    }

    // Probe the API to confirm it's reachable
    fetch(`${apiBase}/manifest`, { signal: AbortSignal.timeout(5_000) })
      .then((res) => {
        if (res.ok) {
          setState({ apiBase, ready: true });
          // Register service worker for offline caching
          if ("serviceWorker" in navigator) {
            navigator.serviceWorker
              .register(`${process.env.NEXT_PUBLIC_BASE_PATH || ""}/sw.js`)
              .then((reg) => {
                // Tell the SW which API origin to cache
                reg.active?.postMessage({
                  type: "SET_VAULT_API_ORIGIN",
                  origin: apiBase,
                });
              })
              .catch(() => {/* SW registration failure is non-fatal */});
          }
        } else {
          setState({ apiBase: null, ready: true });
        }
      })
      .catch(() => {
        setState({ apiBase: null, ready: true });
      });
  }, []);

  return (
    <VaultContext.Provider value={state}>
      {children}
    </VaultContext.Provider>
  );
}

// ── Module-level getters for corpus.ts (non-React code paths) ──

let _apiBase: string | null = null;

export function setApiBase(base: string | null): void {
  _apiBase = base;
}

export function getApiBase(): string | null {
  return _apiBase;
}
