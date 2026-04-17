/**
 * Corpus data-source switch (Phase-4 cutover router).
 *
 * Components that want to be cutover-aware import from this module instead of
 * ``corpus.ts``. Returns the vault-API-backed path when
 * ``NEXT_PUBLIC_VAULT_FALLBACK`` is NOT 'static', falls back to the bundled
 * path otherwise.
 *
 * Components untouched by the cutover continue importing ``corpus.ts`` directly
 * (unchanged behavior) until the user is ready to flip them. This keeps the
 * Phase-4 cutover reviewable one component at a time.
 */

import { usingFallback } from "./vault-fallback";
import * as legacy from "./corpus";
import * as vault from "./corpus-vault";

export function getCorpusSource(): "static" | "vault-api" {
  return usingFallback() ? "static" : "vault-api";
}

export async function getQuestionById(id: string): Promise<unknown | null> {
  if (usingFallback()) {
    const qs = legacy.getQuestions();
    return qs.find(q => q.id === id) ?? null;
  }
  return vault.getQuestionById(id);
}

export async function listQuestions(
  params: { track?: string; level?: string; zone?: string; limit?: number } = {},
): Promise<unknown[]> {
  if (usingFallback()) {
    let qs = legacy.getQuestions() as any[];
    if (params.track) qs = qs.filter(q => q.track === params.track);
    if (params.level) qs = qs.filter(q => q.level === params.level);
    if (params.zone) qs = qs.filter(q => q.zone === params.zone);
    if (params.limit) qs = qs.slice(0, params.limit);
    return qs;
  }
  return vault.listQuestions(params);
}

export async function searchQuestions(q: string, limit = 20): Promise<unknown[]> {
  if (usingFallback()) {
    const qs = legacy.getQuestions() as any[];
    const needle = q.toLowerCase();
    return qs
      .filter(item =>
        (item.title ?? "").toLowerCase().includes(needle)
        || (item.scenario ?? "").toLowerCase().includes(needle)
      )
      .slice(0, limit);
  }
  return vault.searchQuestions(q, limit);
}
