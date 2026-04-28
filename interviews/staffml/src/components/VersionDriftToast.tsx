"use client";

/**
 * Drift detection: compares the baked RELEASE_ID against the worker's
 * live /manifest. If a tab was open across a worker rollover, we nudge
 * the user to refresh — without auto-reloading (would nuke practice
 * state). Single-shot: one fetch per session, dismissable, silent on
 * any failure (offline, worker down, CORS — none of which are the
 * user's problem and none of which we want to surface as an error).
 *
 * The site itself is always self-consistent because RELEASE_ID is
 * baked from vault-manifest.json at build time (single source of
 * truth, see lib/stats.ts). The only thing that drifts is "the
 * worker rolled forward while this tab was open"; this component
 * exists to close that gap.
 */

import { useEffect, useState } from "react";
import { RELEASE_ID } from "@/lib/stats";

const SESSION_KEY = "staffml:driftToastDismissed";

export default function VersionDriftToast() {
  const [liveReleaseId, setLiveReleaseId] = useState<string | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    const api = process.env.NEXT_PUBLIC_VAULT_API;
    if (!api) return;

    if (typeof window !== "undefined") {
      try {
        if (sessionStorage.getItem(SESSION_KEY) === "1") {
          setDismissed(true);
          return;
        }
      } catch {
        // sessionStorage can throw under privacy modes; treat as not-dismissed.
      }
    }

    const ctrl = new AbortController();
    const timeoutId = setTimeout(() => ctrl.abort(), 5_000);

    fetch(`${api.replace(/\/+$/, "")}/manifest`, { signal: ctrl.signal })
      .then(res => (res.ok ? res.json() : null))
      .then((m: { release_id?: string } | null) => {
        if (m?.release_id && m.release_id !== RELEASE_ID) {
          setLiveReleaseId(m.release_id);
        }
      })
      .catch(() => {
        // Silent: drift detection is best-effort chrome, not a blocker.
      })
      .finally(() => clearTimeout(timeoutId));

    return () => {
      clearTimeout(timeoutId);
      ctrl.abort();
    };
  }, []);

  if (dismissed || !liveReleaseId) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      className="fixed bottom-4 right-4 z-50 max-w-xs rounded-lg border border-border bg-surface shadow-lg p-3 text-[12px] text-textSecondary"
    >
      <p className="leading-relaxed">
        New corpus version available
        {" "}
        <span className="font-mono text-textPrimary">v{liveReleaseId}</span>
        {" "}
        (you have <span className="font-mono">v{RELEASE_ID}</span>). Refresh to see updates.
      </p>
      <button
        type="button"
        onClick={() => {
          setDismissed(true);
          try {
            sessionStorage.setItem(SESSION_KEY, "1");
          } catch {
            // ignore
          }
        }}
        className="mt-2 text-[11px] text-textTertiary hover:text-textSecondary underline-offset-2 hover:underline"
      >
        Dismiss
      </button>
    </div>
  );
}
