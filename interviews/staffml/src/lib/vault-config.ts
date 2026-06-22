/**
 * Single source of truth for HOW the site reaches question data.
 *
 * Before this module the resolution logic was duplicated and divergent:
 *   - `corpus-provider.tsx` treated an unset NEXT_PUBLIC_VAULT_API as "no
 *     worker," which in PRODUCTION (where the env is intentionally unset)
 *     silently disabled FTS5 search and the offline service worker.
 *   - `corpus.ts` independently fell back to a hardcoded production worker
 *     URL, so question hydration kept working while search/SW did not.
 *
 * Both behaviors now flow from here, so every code path agrees on the mode
 * and the base URL.
 *
 * Modes
 * -----
 *   "static" — read the materialized corpus.json from public/data/ (local
 *              dev / offline). Opt in with NEXT_PUBLIC_VAULT_FALLBACK=static
 *              (the committed .env.development sets this).
 *   "worker" — fetch heavy fields + search from the Cloudflare vault worker.
 *              Production default (env unset).
 */

/**
 * Production vault worker. Production deploys leave NEXT_PUBLIC_VAULT_API
 * unset and rely on this default; staging overrides it via the env var.
 */
export const DEFAULT_VAULT_API =
  "https://staffml-vault.mlsysbook-ai-account.workers.dev";

export type VaultMode = "static" | "worker";

/** Resolve the data-source mode from the build/runtime environment. */
export function getVaultMode(): VaultMode {
  return process.env.NEXT_PUBLIC_VAULT_FALLBACK?.toLowerCase() === "static"
    ? "static"
    : "worker";
}

/**
 * Absolute, trailing-slash-trimmed base URL of the vault worker — or `null`
 * in static mode (where there is no worker to talk to). In worker mode an
 * unset NEXT_PUBLIC_VAULT_API resolves to the production default rather than
 * to `null`, matching production reality.
 */
export function getVaultApiBase(): string | null {
  if (getVaultMode() === "static") return null;
  const base = process.env.NEXT_PUBLIC_VAULT_API ?? DEFAULT_VAULT_API;
  return base.replace(/\/+$/, "");
}
