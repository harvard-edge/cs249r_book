/**
 * vault-fetch — the ONE resilient transport to the vault worker.
 *
 * History: a 224-line `VaultApiClient` (vault-api.ts) carried retry +
 * circuit-breaker semantics "to close Soumith R3-F-2," but it was imported
 * by nothing — every real fetch went through a bare `fetch()` in corpus.ts
 * and corpus-provider.tsx with no retry, no breaker, and no release header.
 * Worse, its typed `getQuestion()` assumed a nested-`details` response shape
 * that the deployed worker does not return (the worker emits denormalized
 * flat rows). So the resilience the team thought they shipped did not exist.
 *
 * This module replaces it with a single, shape-agnostic transport that the
 * actual hot paths call. It returns parsed JSON and lets each caller nest
 * the worker's flat row into the site's `Question` shape, which is what the
 * deployed contract requires.
 *
 * Guarantees:
 *   - per-attempt timeout (AbortSignal.timeout), composed with a caller signal
 *   - bounded retry on transient failures (5xx / 408 / 425 / 429 / network)
 *   - X-Vault-Release header so the worker knows which bundle called it
 *   - a per-origin circuit breaker shared across all callers (search + detail
 *     fetches trip and recover together), with correct single-probe half-open
 *     semantics
 */

import { RELEASE_ID } from "./stats";

const DEFAULT_TIMEOUT_MS = 8_000;
const DEFAULT_RETRIES = 2; // 1 initial attempt + 2 retries = 3 tries max
const RETRYABLE_STATUS = new Set([408, 425, 429, 500, 502, 503, 504]);

// Circuit-breaker tuning. Shared per origin via the module-level registry.
const FAIL_THRESHOLD = 5;
const RESET_MS = 30_000;

export class VaultFetchError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
    public readonly retryable = false,
  ) {
    super(message);
    this.name = "VaultFetchError";
  }
}

export class CircuitOpenError extends VaultFetchError {
  constructor() {
    super("vault circuit breaker open", 503, false);
    this.name = "CircuitOpenError";
  }
}

export interface VaultFetchOpts {
  /** Caller cancellation signal, composed with the per-attempt timeout. */
  signal?: AbortSignal;
  /** Override the per-attempt timeout. */
  timeoutMs?: number;
  /** Number of RETRIES after the first attempt. 0 = single attempt. */
  retries?: number;
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function backoffDelay(attempt: number): number {
  const base = Math.min(2 ** attempt * 150, 2_000);
  // Full jitter so retrying clients don't synchronize into a thundering herd.
  return base * (0.5 + Math.random());
}

function isRetryable(err: unknown): boolean {
  if (err instanceof VaultFetchError) return err.retryable;
  // Network errors surface as TypeError (fetch) or a NetworkError DOMException.
  // A caller-initiated AbortError is NOT retryable.
  if (err && typeof err === "object" && "name" in err) {
    const name = (err as { name: string }).name;
    if (name === "AbortError") return false;
    return name === "TypeError" || name === "NetworkError";
  }
  return false;
}

// ── Per-origin circuit breaker ───────────────────────────────────────────────

type BreakerState =
  | { kind: "closed"; failures: number }
  | { kind: "open"; openedAt: number }
  | { kind: "half-open" };

const _breakers = new Map<string, BreakerState>();

function getBreaker(origin: string): BreakerState {
  let b = _breakers.get(origin);
  if (!b) {
    b = { kind: "closed", failures: 0 };
    _breakers.set(origin, b);
  }
  return b;
}

/**
 * Decide whether to admit a request. Returns true to proceed. In half-open we
 * admit exactly ONE probe: the admitting caller flips the state to a sentinel
 * so concurrent callers are rejected until the probe resolves. (The orphaned
 * client's version returned true for every concurrent caller, so its
 * "single-slot" comment was a lie — fixed here.)
 */
function admit(origin: string, now: number): boolean {
  const b = getBreaker(origin);
  if (b.kind === "closed") return true;
  if (b.kind === "half-open") {
    // A probe is already in flight; reject everyone else.
    return false;
  }
  // open: admit a single probe once the reset window has elapsed.
  if (now - b.openedAt >= RESET_MS) {
    _breakers.set(origin, { kind: "half-open" });
    return true; // this caller IS the probe
  }
  return false;
}

function noteSuccess(origin: string): void {
  _breakers.set(origin, { kind: "closed", failures: 0 });
}

function noteFailure(origin: string, now: number): void {
  const b = getBreaker(origin);
  if (b.kind === "half-open" || b.kind === "open") {
    // Probe failed (or we were already open) → (re)open the window.
    _breakers.set(origin, { kind: "open", openedAt: now });
    return;
  }
  const failures = b.failures + 1;
  _breakers.set(
    origin,
    failures >= FAIL_THRESHOLD
      ? { kind: "open", openedAt: now }
      : { kind: "closed", failures },
  );
}

function originOf(url: string): string {
  try {
    return new URL(url).origin;
  } catch {
    return url;
  }
}

/** Visible for testing. */
export function __resetBreakers(): void {
  _breakers.clear();
}

/** Visible for testing. */
export function __breakerKind(url: string): BreakerState["kind"] {
  return getBreaker(originOf(url)).kind;
}

// ── The transport ─────────────────────────────────────────────────────────────

/**
 * Fetch JSON from the vault worker with timeout, bounded retry, the release
 * header, and circuit breaking. Throws `CircuitOpenError` when the breaker is
 * open, `VaultFetchError` on HTTP errors, or the underlying error otherwise.
 */
export async function vaultFetchJson<T>(
  url: string,
  opts: VaultFetchOpts = {},
): Promise<T> {
  const { signal, timeoutMs = DEFAULT_TIMEOUT_MS, retries = DEFAULT_RETRIES } = opts;
  const origin = originOf(url);

  if (!admit(origin, Date.now())) throw new CircuitOpenError();

  let lastErr: unknown;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const perAttempt = AbortSignal.timeout(timeoutMs);
    const sig = signal ? AbortSignal.any([signal, perAttempt]) : perAttempt;
    try {
      const res = await fetch(url, {
        signal: sig,
        headers: { "X-Vault-Release": RELEASE_ID },
      });
      if (!res.ok) {
        throw new VaultFetchError(
          `HTTP ${res.status}`,
          res.status,
          RETRYABLE_STATUS.has(res.status),
        );
      }
      const data = (await res.json()) as T;
      noteSuccess(origin);
      return data;
    } catch (err) {
      lastErr = err;
      // Caller cancelled: surface immediately, don't count against the breaker.
      if (signal?.aborted) throw err;
      if (!isRetryable(err) || attempt === retries) break;
      await wait(backoffDelay(attempt));
    }
  }

  noteFailure(origin, Date.now());
  throw lastErr instanceof Error ? lastErr : new VaultFetchError(String(lastErr));
}
