/**
 * useFullQuestion — drop-in hook that hydrates a summary question to full.
 *
 * The bundled corpus is summary-only (id/title/level/zone/topic/… — no
 * scenario/details). When a component needs the heavy fields, wrap the
 * summary with this hook. It fetches from the worker and re-renders.
 *
 * Usage:
 *   const current = getQuestionById(qId);           // sync, summary only
 *   const full = useFullQuestion(current);          // async hydrate
 *   // First render: full === current (scenario/details undefined)
 *   // After fetch:  full === { ...current, scenario, details }
 */

"use client";

import { useEffect, useState } from "react";
import { getQuestionFullDetail, type Question } from "../corpus";

export function useFullQuestion(summary: Question | undefined | null): Question | undefined {
  const [hydrated, setHydrated] = useState<Question | undefined>(
    summary ?? undefined,
  );

  useEffect(() => {
    if (!summary) {
      setHydrated(undefined);
      return;
    }
    // If we already have scenario cached in the summary, skip fetch.
    if (summary.scenario && summary.details?.realistic_solution) {
      setHydrated(summary);
      return;
    }
    // Seed with summary so listing UI renders instantly; then hydrate.
    setHydrated(summary);
    let cancelled = false;
    getQuestionFullDetail(summary.id).then(full => {
      if (cancelled || !full) return;
      // Merge rather than replace: the worker returns the heavy fields
      // (scenario, details) but does not necessarily carry every
      // summary-bundle field. Summary fields like `question` (the
      // explicit-ask prompt) live in the bundle and would otherwise be
      // dropped by a straight replace. Spread summary first so worker
      // values win where they overlap (they carry the real content),
      // but summary-only fields survive.
      setHydrated({ ...summary, ...full });
    });
    return () => {
      cancelled = true;
    };
  }, [summary?.id]);   // re-run when the summary ID changes

  return hydrated;
}
