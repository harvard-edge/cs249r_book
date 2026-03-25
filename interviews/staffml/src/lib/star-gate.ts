// ─── GitHub Star Gate ─────────────────────────────────────────
// After FREE_LIMIT question reveals, users must star the GitHub repo.
// Verification uses the GitHub API (unauthenticated, checks stargazers).

const STORAGE_KEY = "staffml_star_gate";
const REVEALS_KEY = "staffml_reveals_today";
const FREE_LIMIT = 20; // questions before gate appears
const REPO_OWNER = "harvard-edge";
const REPO_NAME = "cs249r_book";
const REPO_URL = `https://github.com/${REPO_OWNER}/${REPO_NAME}`;

interface StarGateData {
  verified: boolean;
  verifiedAt?: number;
  githubUsername?: string;
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

/** Get today's reveal count */
export function getRevealCount(): number {
  try {
    const raw = window.localStorage.getItem(REVEALS_KEY);
    if (!raw) return 0;
    const data = JSON.parse(raw);
    const today = new Date().toISOString().slice(0, 10);
    return data.date === today ? data.count : 0;
  } catch {
    return 0;
  }
}

/** Increment today's reveal count */
export function incrementReveals(): void {
  try {
    const today = new Date().toISOString().slice(0, 10);
    const raw = window.localStorage.getItem(REVEALS_KEY);
    const data = raw ? JSON.parse(raw) : { date: today, count: 0 };
    if (data.date !== today) {
      data.date = today;
      data.count = 0;
    }
    data.count++;
    window.localStorage.setItem(REVEALS_KEY, JSON.stringify(data));
  } catch {}
}

/** Check if user has been verified as a stargazer */
export function isStarVerified(): boolean {
  return getGateData().verified;
}

/** Check if the gate should be shown */
export function shouldShowGate(): boolean {
  if (isStarVerified()) return false;
  return getRevealCount() >= FREE_LIMIT;
}

/** Get remaining free reveals */
export function getRemainingReveals(): number {
  if (isStarVerified()) return Infinity;
  return Math.max(0, FREE_LIMIT - getRevealCount());
}

/** Get the repo URL for starring */
export function getStarUrl(): string {
  return REPO_URL;
}

/** Get the free limit */
export function getFreeLimit(): number {
  return FREE_LIMIT;
}

/**
 * Verify that a GitHub user has starred the repo.
 * Uses the unauthenticated GitHub API — rate limited to 60 req/hour.
 * Returns true if the user is a stargazer.
 */
export async function verifyGitHubStar(username: string): Promise<boolean> {
  try {
    // Check if user has starred the repo using the GitHub API
    // GET /users/{username}/starred/{owner}/{repo} returns 204 if starred, 404 if not
    const res = await fetch(
      `https://api.github.com/users/${encodeURIComponent(username)}/starred/${REPO_OWNER}/${REPO_NAME}`,
      {
        headers: { Accept: "application/vnd.github.v3+json" },
      }
    );

    if (res.status === 204) {
      // Verified! Save permanently
      setGateData({
        verified: true,
        verifiedAt: Date.now(),
        githubUsername: username,
      });
      return true;
    }

    return false;
  } catch {
    // Network error — give benefit of the doubt on first try
    return false;
  }
}

/**
 * Bypass verification (honor system fallback).
 * Used when GitHub API is rate-limited or user doesn't want to share username.
 */
export function bypassVerification(): void {
  setGateData({
    verified: true,
    verifiedAt: Date.now(),
    githubUsername: "_honor_system",
  });
}
