/**
 * Minimal client-side error reporter.
 *
 * Captures two classes of runtime failures that otherwise only surface in
 * the user's devtools console:
 *
 *   1. Uncaught exceptions (window.onerror) — render crashes, null derefs,
 *      TypeErrors like the hydration shape mismatch fixed in PR #1440.
 *   2. Unhandled promise rejections (unhandledrejection) — async failures
 *      like a fetch that throws but has no .catch().
 *
 * Reports flow through the existing analytics-worker at
 * NEXT_PUBLIC_ANALYTICS_URL as `client_error` events, which the worker
 * stores in KV alongside the other analytics. Stack traces are truncated
 * at 4 KiB per event to respect the worker's size cap; they are not PII
 * (our code, not user data), but we scrub email-like patterns as a belt
 * for the PII filter the worker also applies.
 *
 * Installed once from app/layout.tsx.
 */

import { track } from "./analytics";

const MAX_STACK_BYTES = 4096;
const MAX_MESSAGE_BYTES = 512;
const EMAIL_PATTERN = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g;
// Rate-limit per tab to avoid a render loop pounding the worker — if the
// bug fires every frame we still only send the first 20 unique messages.
const MAX_REPORTS_PER_SESSION = 20;

const _seen = new Set<string>();
let _installed = false;
let _reports = 0;

function scrub(text: string, cap: number): string {
  const clean = text.replace(EMAIL_PATTERN, "[email]");
  if (clean.length <= cap) return clean;
  return clean.slice(0, cap) + "…[truncated]";
}

function report(message: string, stack: string | undefined): void {
  if (_reports >= MAX_REPORTS_PER_SESSION) return;
  const key = (stack ?? message).slice(0, 200);
  if (_seen.has(key)) return;
  _seen.add(key);
  _reports += 1;
  try {
    track({
      type: "client_error",
      message: scrub(message, MAX_MESSAGE_BYTES),
      stack: stack ? scrub(stack, MAX_STACK_BYTES) : undefined,
      url: window.location.href.slice(0, 512),
      userAgent: navigator.userAgent.slice(0, 256),
    });
  } catch {
    /* analytics itself failing shouldn't cascade. */
  }
}

export function installErrorReporter(): void {
  if (_installed) return;
  if (typeof window === "undefined") return;
  _installed = true;

  window.addEventListener("error", (ev: ErrorEvent) => {
    // ErrorEvent.error can be null for cross-origin scripts; fall back to .message.
    const err = ev.error as Error | null;
    report(err?.message ?? ev.message ?? "(unknown error)", err?.stack);
  });

  window.addEventListener("unhandledrejection", (ev: PromiseRejectionEvent) => {
    const reason = ev.reason;
    if (reason instanceof Error) {
      report(reason.message, reason.stack);
    } else {
      report(String(reason), undefined);
    }
  });
}
