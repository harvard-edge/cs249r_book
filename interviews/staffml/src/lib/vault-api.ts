/**
 * Thin typed client over the staffml-vault Worker API.
 *
 * Wires up retry-with-backoff + circuit breaker + SWR-friendly caching so the
 * UI never crashes on a flaky edge (M-10). Consumed by `corpus.ts` after the
 * Phase-4 cutover; today `corpus.ts` still reads from the static bundle
 * until `NEXT_PUBLIC_VAULT_FALLBACK` is unset.
 */

import type {
  Manifest,
  Question,
  VaultApiClientOptions,
} from "@staffml/vault-types";

interface FetchOpts {
  signal?: AbortSignal;
}

interface BreakerState {
  failures: number;
  openUntil: number | null;
}

const DEFAULT_OPTIONS: VaultApiClientOptions = {
  release: "unknown",
  retry: { attempts: 3, backoff: "exponential", jitter: true },
  circuitBreaker: { failThreshold: 5, resetMs: 30_000 },
};

export class VaultApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = "VaultApiError";
  }
}

export class CircuitOpenError extends VaultApiError {
  constructor() {
    super("circuit breaker open", 503);
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

export class VaultApiClient {
  private readonly base: string;
  private readonly opts: VaultApiClientOptions;
  private breaker: BreakerState = { failures: 0, openUntil: null };

  constructor(base: string, opts?: Partial<VaultApiClientOptions>) {
    this.base = base.replace(/\/+$/, "");
    this.opts = { ...DEFAULT_OPTIONS, ...opts };
  }

  private async request<T>(path: string, init: FetchOpts = {}): Promise<T> {
    const bk = this.opts.circuitBreaker!;
    const now = Date.now();
    if (this.breaker.openUntil && this.breaker.openUntil > now) {
      throw new CircuitOpenError();
    }
    if (this.breaker.openUntil && this.breaker.openUntil <= now) {
      this.breaker = { failures: 0, openUntil: null };
    }

    const retry = this.opts.retry!;
    const attempts = Math.max(1, retry.attempts);
    let lastErr: unknown;

    for (let i = 0; i < attempts; i++) {
      try {
        const res = await fetch(`${this.base}${path}`, {
          signal: init.signal,
          headers: {
            "X-Vault-Release": this.opts.release,
            ...(this.opts.headers ?? {}),
          },
        });
        if (!res.ok) {
          throw new VaultApiError(`HTTP ${res.status}`, res.status);
        }
        this.breaker = { failures: 0, openUntil: null };
        return (await res.json()) as T;
      } catch (err) {
        lastErr = err;
        if (i < attempts - 1) {
          await wait(backoffDelay(i, retry.jitter ?? false));
        }
      }
    }

    this.breaker.failures += 1;
    if (this.breaker.failures >= bk.failThreshold) {
      this.breaker.openUntil = Date.now() + bk.resetMs;
    }
    throw lastErr instanceof Error ? lastErr : new VaultApiError(String(lastErr));
  }

  getManifest(init?: FetchOpts): Promise<Manifest> {
    return this.request<Manifest>("/manifest", init);
  }

  getQuestion(id: string, init?: FetchOpts): Promise<Question> {
    return this.request<Question>(`/questions/${encodeURIComponent(id)}`, init);
  }

  listQuestions(
    params: { track?: string; level?: string; zone?: string; cursor?: string; limit?: number } = {},
    init?: FetchOpts,
  ): Promise<{ items: Question[]; next_cursor: string | null }> {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined) qs.set(k, String(v));
    }
    const query = qs.toString();
    return this.request(`/questions${query ? `?${query}` : ""}`, init);
  }

  search(q: string, limit = 20, init?: FetchOpts): Promise<{ results: Question[]; query: string }> {
    const qs = new URLSearchParams({ q, limit: String(limit) });
    return this.request(`/search?${qs}`, init);
  }
}

export function makeClientFromEnv(): VaultApiClient | null {
  const api = process.env.NEXT_PUBLIC_VAULT_API;
  const release = process.env.NEXT_PUBLIC_VAULT_RELEASE ?? "unknown";
  if (!api) return null;
  return new VaultApiClient(api, { release });
}
