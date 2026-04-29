/**
 * useFullQuestion — drop-in hook that hydrates a summary question to full.
 *
 * The bundled corpus is summary-only (id/title/level/zone/topic/… — no
 * scenario/details). When a component needs the heavy fields, wrap the
 * summary with this hook. It fetches from the Worker and re-renders.
 *
 * Returns { question, status }:
 *   - question: the best record we have (summary on first render, or after
 *     a failed fetch; full record once the Worker resolves)
 *   - status: 'loading' while the fetch is in flight, 'ready' on success,
 *     'error' if the Worker is unreachable. Callers can render an error
 *     hint ("Details unavailable — retry") when status === 'error'.
 *
 * Usage:
 *   const summary = getQuestionById(qId);
 *   const { question, status } = useFullQuestion(summary);
 *   if (status === 'error') return <DetailsUnavailable onRetry={…} />;
 */

"use client";

import { useEffect, useState } from "react";
import { getQuestionFullDetail, type Question } from "../corpus";

export type UseFullQuestionStatus = "loading" | "ready" | "error";

export interface UseFullQuestionResult {
  question: Question | undefined;
  status: UseFullQuestionStatus;
}

export function useFullQuestion(
  summary: Question | undefined | null,
): UseFullQuestionResult {
  const [result, setResult] = useState<UseFullQuestionResult>(() => ({
    question: summary ?? undefined,
    status: summary ? "loading" : "ready",
  }));

  useEffect(() => {
    if (!summary) {
      setResult({ question: undefined, status: "ready" });
      return;
    }
    // Already hydrated in the summary itself (rare, but possible if a
    // future bundle ships details inline). Skip the fetch.
    if (summary.scenario && summary.details?.realistic_solution) {
      setResult({ question: summary, status: "ready" });
      return;
    }
    setResult({ question: summary, status: "loading" });
    let cancelled = false;
    getQuestionFullDetail(summary.id)
      .then(full => {
        if (cancelled) return;
        if (!full) {
          setResult({ question: summary, status: "error" });
          return;
        }
        // Merge rather than replace: the Worker returns the heavy fields
        // (scenario, details) but does not necessarily carry every
        // summary-bundle field. Spread summary first so Worker values
        // win where they overlap, but summary-only fields survive.
        setResult({ question: { ...summary, ...full }, status: "ready" });
      })
      .catch(() => {
        if (cancelled) return;
        setResult({ question: summary, status: "error" });
      });
    return () => {
      cancelled = true;
    };
  }, [summary?.id]);

  return result;
}
