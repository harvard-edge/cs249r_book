/**
 * Thin typed client over the staffml-vault Worker API.
 *
 * v2.2: proper half-open circuit-breaker semantics, AbortSignal.timeout on
 * every fetch, retry only on transient statuses (5xx / 429 / network), and
 * a module-level singleton so multiple `makeClientFromEnv()` calls share the
 * same breaker state. (Closes Soumith R3-F-2.)
 */

import type { Manifest, Question, VaultApiClientOptions } from "@staffml/vault-types";

interface FetchOpts {
  signal?: AbortSignal;
}

type BreakerState =
  | { kind: "closed"; failures: number }
  | { kind: "open"; openedAt: number }
  | { kind: "half-open" };

const DEFAULT_OPTIONS: Required<VaultApiClientOptions> = {
  release: "unknown",
  retry: { attempts: 3, backoff: "exponential", jitter: true },
  circuitBreaker: { failThreshold: 5, resetMs: 30_000 },
  headers: {},
};

const DEFAULT_TIMEOUT_MS = 10_000;
const RETRYABLE_STATUS = new Set([408, 425, 429, 500, 502, 503, 504]);

export class VaultApiError extends Error {
  constructor(message: string, public status?: number, public retryable = false) {
    super(message);
    this.name = "VaultApiError";
  }
}

export class CircuitOpenError extends VaultApiError {
  constructor() {
    super("circuit breaker open", 503, false);
    this.name = "CircuitOpenError";
  }
}

function wait(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function backoffDelay(attempt: number, jitter: boolean): number {
  const base = Math.min(2 ** attempt * 100, 5_000);
  return jitter ? base * (0.5 + Math.random()) : base;
}

function isRetryable(err: unknown): boolean {
  if (err instanceof VaultApiError) return err.retryable;
  // Network errors (DOMException, TypeError from fetch) are retryable;
  // AbortError from caller-cancellation is NOT.
  if (err && typeof err === "object" && "name" in err) {
    const name = (err as { name: string }).name;
    if (name === "AbortError") return false;
    return name === "TypeError" || name === "NetworkError";
  }
  return false;
}

export class VaultApiClient {
  private readonly base: string;
  private readonly opts: Required<VaultApiClientOptions>;
  private breaker: BreakerState = { kind: "closed", failures: 0 };

  constructor(base: string, opts?: Partial<VaultApiClientOptions>) {
    this.base = base.replace(/\/+$/, "");
    this.opts = {
      ...DEFAULT_OPTIONS,
      ...opts,
      retry: { ...DEFAULT_OPTIONS.retry, ...(opts?.retry ?? {}) },
      circuitBreaker: { ...DEFAULT_OPTIONS.circuitBreaker, ...(opts?.circuitBreaker ?? {}) },
      headers: { ...DEFAULT_OPTIONS.headers, ...(opts?.headers ?? {}) },
    };
  }

  /** Visible for testing. */
  breakerKind(): BreakerState["kind"] {
    return this.breaker.kind;
  }

  private noteSuccess(): void {
    this.breaker = { kind: "closed", failures: 0 };
  }

  private noteFailure(): void {
    const { failThreshold, resetMs } = this.opts.circuitBreaker;
    if (this.breaker.kind === "half-open") {
      // Probe failed → re-open immediately.
      this.breaker = { kind: "open", openedAt: Date.now() };
      return;
    }
    if (this.breaker.kind === "closed") {
      const failures = this.breaker.failures + 1;
      if (failures >= failThreshold) {
        this.breaker = { kind: "open", openedAt: Date.now() };
      } else {
        this.breaker = { kind: "closed", failures };
      }
      return;
    }
    // Already open; just push opened-at forward.
    this.breaker = { kind: "open", openedAt: Date.now() };
    void resetMs;
  }

  private admitRequest(): boolean {
    if (this.breaker.kind === "closed") return true;
    if (this.breaker.kind === "half-open") {
      // Only one request at a time in half-open; naive single-slot by reverting
      // to open so parallel callers don't stampede the probe.
      this.breaker = { kind: "open", openedAt: Date.now() };
      return true; // This caller IS the probe.
    }
    // Open: check if reset window elapsed.
    const { resetMs } = this.opts.circuitBreaker;
    if (Date.now() - this.breaker.openedAt >= resetMs) {
      this.breaker = { kind: "half-open" };
      return true; // Probe admitted.
    }
    return false;
  }

  private async request<T>(path: string, init: FetchOpts = {}): Promise<T> {
    if (!this.admitRequest()) throw new CircuitOpenError();

    const { attempts, jitter } = this.opts.retry;
    let lastErr: unknown;

    for (let i = 0; i < attempts; i++) {
      // Per-attempt timeout; caller's signal is respected if provided.
      const perAttempt = AbortSignal.timeout(DEFAULT_TIMEOUT_MS);
      const signal = init.signal
        ? AbortSignal.any([init.signal, perAttempt])
        : perAttempt;
      try {
        const res = await fetch(`${this.base}${path}`, {
          signal,
          headers: {
            "X-Vault-Release": this.opts.release,
            ...this.opts.headers,
          },
        });
        if (!res.ok) {
          throw new VaultApiError(`HTTP ${res.status}`, res.status, RETRYABLE_STATUS.has(res.status));
        }
        this.noteSuccess();
        return (await res.json()) as T;
      } catch (err) {
        lastErr = err;
        // Don't retry if caller aborted, or if the error isn't retryable.
        if (err instanceof DOMException && err.name === "AbortError" && init.signal?.aborted) {
          throw err;
        }
        if (!isRetryable(err)) break;
        if (i < attempts - 1) {
          await wait(backoffDelay(i, jitter ?? false));
        }
      }
    }

    this.noteFailure();
    throw lastErr instanceof Error ? lastErr : new VaultApiError(String(lastErr));
  }

  getManifest(init?: FetchOpts): Promise<Manifest> {
    return this.request<Manifest>("/manifest", init);
  }

  getQuestion(id: string, init?: FetchOpts): Promise<Question> {
    return this.request<Question>(`/questions/${encodeURIComponent(id)}`, init);
  }

  listQuestions(
    params: { track?: string; level?: string; zone?: string; topic?: string;
              status?: string; cursor?: string; limit?: number } = {},
    init?: FetchOpts,
  ): Promise<{ items: Question[]; next_cursor: string | null }> {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined) qs.set(k, String(v));
    }
    const query = qs.toString();
    return this.request(`/questions${query ? `?${query}` : ""}`, init);
  }

  search(
    q: string,
    limit = 20,
    init?: FetchOpts,
  ): Promise<{ results: Question[]; query: string; fts?: boolean }> {
    const qs = new URLSearchParams({ q, limit: String(limit) });
    return this.request(`/search?${qs}`, init);
  }
}

// Module-level singleton (Soumith R3-F-2): consumers call makeClientFromEnv()
// once at module load. Multiple module-load paths share the same breaker state
// via this singleton rather than each creating an independent breaker.
let _singleton: VaultApiClient | null | undefined = undefined;

export function makeClientFromEnv(): VaultApiClient | null {
  if (_singleton !== undefined) return _singleton;
  const api = process.env.NEXT_PUBLIC_VAULT_API;
  const release = process.env.NEXT_PUBLIC_VAULT_RELEASE ?? "unknown";
  _singleton = api ? new VaultApiClient(api, { release }) : null;
  return _singleton;
}

/** Visible for testing. */
export function __resetSingleton(): void {
  _singleton = undefined;
}
