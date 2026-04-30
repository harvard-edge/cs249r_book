// ─── GitHub Star Gate ─────────────────────────────────────────
// Once per user, after REVEAL_THRESHOLD lifetime reveals, we surface
// a single non-blocking modal asking for a GitHub star — the project's
// "only ask." Honor-system: any of star / "I already starred" / dismiss
// retires the gate forever. No username verification, no daily cap.

const STORAGE_KEY = "staffml_star_gate";
const REVEALS_KEY = "staffml_lifetime_reveals";
const STAR_COUNT_KEY = "staffml_star_count_cache";
const REVEAL_THRESHOLD = 5;
const REPO_OWNER = "harvard-edge";
const REPO_NAME = "cs249r_book";
const REPO_URL = `https://github.com/${REPO_OWNER}/${REPO_NAME}`;
const STAR_COUNT_TTL_MS = 24 * 60 * 60 * 1000;

type DismissMethod = "starred" | "honor" | "dismissed";

interface StarGateData {
  verified: boolean;
  verifiedAt?: number;
  method?: DismissMethod;
}

interface StarCountCache {
  count: number;
  fetchedAt: number;
}

function getGateData(): StarGateData {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : { verified: false };
  } catch {
    return { verified: false };
  }
}

function setGateData(data: StarGateData): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch {}
}

/** Lifetime reveal count across all sessions. */
export function getRevealCount(): number {
  try {
    const raw = window.localStorage.getItem(REVEALS_KEY);
    if (!raw) return 0;
    const n = parseInt(raw, 10);
    return Number.isFinite(n) ? n : 0;
  } catch {
    return 0;
  }
}

/** Increment the lifetime reveal count. */
export function incrementReveals(): void {
  try {
    window.localStorage.setItem(REVEALS_KEY, String(getRevealCount() + 1));
  } catch {}
}

/** Has the user already retired the gate (starred or dismissed)? */
export function isStarVerified(): boolean {
  return getGateData().verified;
}

/** Should the gate be surfaced now? */
export function shouldShowGate(): boolean {
  if (isStarVerified()) return false;
  return getRevealCount() >= REVEAL_THRESHOLD;
}

/** Repo URL for the star CTA. */
export function getStarUrl(): string {
  return REPO_URL;
}

/** Threshold of lifetime reveals before the gate first surfaces. */
export function getRevealThreshold(): number {
  return REVEAL_THRESHOLD;
}

/**
 * Mark the gate retired. Once called, shouldShowGate() returns false forever.
 * Method is purely diagnostic — all three dismiss paths fully retire the gate.
 */
export function markVerified(method: DismissMethod): void {
  setGateData({ verified: true, verifiedAt: Date.now(), method });
}

/**
 * Fetch the live stargazer count from the GitHub API, with a 24h
 * localStorage cache so we don't burn the unauthenticated rate limit.
 * Returns null on any failure — callers should fall back gracefully.
 */
export async function fetchStarCount(): Promise<number | null> {
  try {
    const raw = window.localStorage.getItem(STAR_COUNT_KEY);
    if (raw) {
      const cache: StarCountCache = JSON.parse(raw);
      if (Date.now() - cache.fetchedAt < STAR_COUNT_TTL_MS && typeof cache.count === "number") {
        return cache.count;
      }
    }
  } catch {}

  try {
    const res = await fetch(`https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}`, {
      headers: { Accept: "application/vnd.github.v3+json" },
    });
    if (!res.ok) return null;
    const data = await res.json();
    const count = data?.stargazers_count;
    if (typeof count !== "number") return null;
    try {
      window.localStorage.setItem(
        STAR_COUNT_KEY,
        JSON.stringify({ count, fetchedAt: Date.now() } satisfies StarCountCache),
      );
    } catch {}
    return count;
  } catch {
    return null;
  }
}
