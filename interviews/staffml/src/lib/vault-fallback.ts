/**
 * Fallback-mode detection for the Phase-4 cutover.
 *
 * When NEXT_PUBLIC_VAULT_FALLBACK=static, the site reads from the bundled
 * corpus.json (pre-cutover behavior preserved). When unset or 'vault', the
 * site reads from the Worker API via vault-api.ts.
 *
 * One config change inverts the dataflow — no file restore required
 * (ARCHITECTURE.md §7.1 / §6.2, fix for C-1 "one-line revert" lie).
 */

export type VaultSource = "static" | "vault-api";

export function getVaultSource(): VaultSource {
  const flag = process.env.NEXT_PUBLIC_VAULT_FALLBACK?.toLowerCase();
  if (flag === "static") return "static";
  return "vault-api";
}

export function usingFallback(): boolean {
  return getVaultSource() === "static";
}
