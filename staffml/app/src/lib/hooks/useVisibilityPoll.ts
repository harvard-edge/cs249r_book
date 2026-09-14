/**
 * useVisibilityPoll — run a callback on a fixed interval, but only when
 * the document is visible. When the tab is hidden the interval is cleared;
 * when it becomes visible again the callback fires immediately so the UI
 * catches up on time-based changes (e.g. cards becoming due while the tab
 * was backgrounded) and the regular interval resumes.
 *
 * The original use case (Nav.tsx's SR due-count badge) was polling every
 * 30s indefinitely, even on backgrounded tabs — cheap per call but wasted
 * battery on mobile.
 *
 * NOT a cross-tab sync primitive. If two tabs are visible at the same
 * time (split screen, multiple monitors, popped-out window), a write in
 * one tab is invisible to the other until its next interval tick. Callers
 * that need instant cross-tab freshness should pair this hook with a
 * `storage` event listener — see Nav.tsx for an example.
 *
 * The callback is held in a ref so a fresh closure on every render is
 * picked up without tearing down and re-arming the interval. Only
 * `intervalMs` is a real dependency. The callback fires once immediately
 * on mount (when visible), which is a deviation from a bare setInterval.
 *
 * Usage:
 *   useVisibilityPoll(() => setDueCount(getDueCount()), 30_000);
 */
"use client";

import { useEffect, useRef } from "react";

export function useVisibilityPoll(callback: () => void, intervalMs: number): void {
  const cbRef = useRef(callback);
  // Refresh the ref after every render so the interval always calls the
  // latest closure (the caller may close over fresh state).
  useEffect(() => { cbRef.current = callback; });

  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;
    // Throw-isolate the callback. If it ever throws (corrupted localStorage,
    // null deref in dev) the exception is logged and the next tick still
    // runs — without this guard, a throw inside start()'s synchronous tick
    // during MOUNT propagates out of the effect body before React receives
    // the cleanup, leaving the already-armed setInterval orphaned and
    // re-throwing every intervalMs with no way to cancel.
    const tick = () => {
      try { cbRef.current(); }
      catch (e) { console.error("[useVisibilityPoll] callback threw", e); }
    };

    const start = () => {
      if (interval !== null) return;
      tick();
      interval = setInterval(tick, intervalMs);
    };
    const stop = () => {
      if (interval !== null) {
        clearInterval(interval);
        interval = null;
      }
    };

    if (document.visibilityState === "visible") start();

    const onVisibility = () => {
      if (document.visibilityState === "visible") start();
      else stop();
    };
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      stop();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [intervalMs]);
}
