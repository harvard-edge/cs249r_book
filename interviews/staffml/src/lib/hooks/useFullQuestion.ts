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

import { useEffect, useRef, useState } from "react";
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

  // The effect re-runs only when the question id changes, but it reads the
  // summary through a ref so the merge always uses the CURRENT summary rather
  // than a value captured when the effect last ran. Without the ref, keying on
  // `summary?.id` alone closes over a stale `summary` if the same-id record is
  // swapped for a richer one; depending on the whole `summary` object instead
  // would refetch on every render when a caller passes a fresh object.
  const summaryRef = useRef(summary);
  summaryRef.current = summary;

  useEffect(() => {
    const current = summaryRef.current;
    if (!current) {
      setResult({ question: undefined, status: "ready" });
      return;
    }
    // Already hydrated in the summary itself (rare, but possible if a
    // future bundle ships details inline). Skip the fetch.
    if (current.scenario && current.details?.realistic_solution) {
      setResult({ question: current, status: "ready" });
      return;
    }
    setResult({ question: current, status: "loading" });
    let cancelled = false;
    getQuestionFullDetail(current.id)
      .then(full => {
        if (cancelled) return;
        if (!full) {
          setResult({ question: current, status: "error" });
          return;
        }
        // Merge rather than replace: the Worker returns the heavy fields
        // (scenario, details) but does not necessarily carry every
        // summary-bundle field. Spread summary first so Worker values
        // win where they overlap, but summary-only fields survive.
        setResult({ question: { ...current, ...full }, status: "ready" });
      })
      .catch(() => {
        if (cancelled) return;
        setResult({ question: current, status: "error" });
      });
    return () => {
      cancelled = true;
    };
  }, [summary?.id]);

  return result;
}
