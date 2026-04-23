/**
 * React hook — single-question fetch through the Phase-4 cutover router.
 *
 * Components that opt into the cutover import `useVaultQuestion` instead of
 * calling `corpus.getQuestions()` synchronously. On `NEXT_PUBLIC_VAULT_FALLBACK=static`
 * it returns the question from the bundled corpus (synchronous resolve);
 * otherwise it fetches via the Worker API through `corpus-source.ts`.
 *
 * Part of B.17 — the migration path for existing components is one-at-a-time
 * swap from `corpus.getQuestionById()` to `useVaultQuestion()`.
 */

import { useEffect, useState } from "react";
import { getQuestionById } from "../corpus-source";

export interface UseVaultQuestionState<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
}

export function useVaultQuestion<T = unknown>(id: string | null): UseVaultQuestionState<T> {
  const [state, setState] = useState<UseVaultQuestionState<T>>({
    data: null,
    loading: id !== null,
    error: null,
  });

  useEffect(() => {
    if (id === null) {
      setState({ data: null, loading: false, error: null });
      return;
    }
    let cancelled = false;
    setState(s => ({ ...s, loading: true, error: null }));
    getQuestionById(id)
      .then(result => {
        if (cancelled) return;
        setState({ data: result as T, loading: false, error: null });
      })
      .catch(err => {
        if (cancelled) return;
        setState({ data: null, loading: false, error: err instanceof Error ? err : new Error(String(err)) });
      });
    return () => { cancelled = true; };
  }, [id]);

  return state;
}
