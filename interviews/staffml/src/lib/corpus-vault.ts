/**
 * Vault-API-backed corpus data source (Phase-4 cutover path).
 *
 * Mirror of the public surface of ``corpus.ts`` but sourced from the
 * staffml-vault Worker via ``vault-api.ts`` instead of the bundled
 * ``corpus.json``. Not wired into any component until cutover day —
 * the switch happens via ``corpus-source.ts``.
 *
 * This is the Phase-4 load-bearing file. Review it against ``corpus.ts``
 * for API parity before flipping the switch.
 */

import type { Question as VaultQuestion } from "@staffml/vault-types";
import { makeClientFromEnv, VaultApiClient } from "./vault-api";

// The vault worker enriches each Question with track/level/zone derived from
// the source filesystem path (classification is path-encoded per vault_cli
// models.py). These fields are not on the schema itself.
type EnrichedVaultQuestion = VaultQuestion & {
  track?: string;
  level?: string;
  zone?: string;
};

// The legacy corpus.ts exports a specific Question shape; this vault-backed
// module adapts the @staffml/vault-types Question to that shape so callers
// don't need to change.
export interface Question {
  id: string;
  track: string;
  scope?: string;
  level: string;
  title: string;
  topic: string;
  zone: string;
  competency_area: string;
  bloom_level?: string;
  scenario: string;
  chain_ids?: string[];
  chain_positions?: Record<string, number>;
  details: {
    common_mistake: string;
    realistic_solution: string;
    napkin_math?: string;
    deep_dive_title?: string;
    deep_dive_url?: string;
  };
}

function adapt(v: EnrichedVaultQuestion): Question {
  return {
    id: v.id,
    track: v.track ?? "global",
    level: v.level ?? "l1",
    title: v.title,
    topic: v.topic,
    zone: v.zone ?? "recall",
    competency_area: v.topic,
    scenario: v.scenario,
    chain_ids: v.chain ? [v.chain.id] : undefined,
    chain_positions: v.chain ? { [v.chain.id]: v.chain.position } : undefined,
    details: {
      common_mistake: v.details.common_mistake ?? "",
      realistic_solution: v.details.realistic_solution,
      napkin_math: v.details.napkin_math,
      deep_dive_title: v.details.deep_dive?.title,
      deep_dive_url: v.details.deep_dive?.url,
    },
  };
}

let _client: VaultApiClient | null | undefined = undefined;
function client(): VaultApiClient {
  if (_client === undefined) _client = makeClientFromEnv();
  if (_client === null) {
    throw new Error(
      "NEXT_PUBLIC_VAULT_API is not set. Point it at the worker or set "
      + "NEXT_PUBLIC_VAULT_FALLBACK=static to use the bundled corpus.",
    );
  }
  return _client;
}

// In-memory cache; SWR (in real consumption via hooks) layers on top.
const _byId = new Map<string, Question>();

export async function getQuestionById(id: string): Promise<Question | null> {
  if (_byId.has(id)) return _byId.get(id)!;
  try {
    const v = await client().getQuestion(id);
    const q = adapt(v as EnrichedVaultQuestion);
    _byId.set(id, q);
    return q;
  } catch {
    return null;
  }
}

export async function listQuestions(params: {
  track?: string; level?: string; zone?: string; limit?: number;
} = {}): Promise<Question[]> {
  const res = await client().listQuestions(params);
  return (res.items as EnrichedVaultQuestion[]).map(adapt);
}

export async function searchQuestions(q: string, limit = 20): Promise<Question[]> {
  const res = await client().search(q, limit);
  return (res.results as EnrichedVaultQuestion[]).map(adapt);
}

/**
 * Synchronous getQuestions() — compatibility shim for legacy call sites that
 * expect an array rather than a Promise. Returns the currently-cached set
 * (populated by prior async calls). Callers doing full-corpus scans must
 * migrate to listQuestions().
 */
export function getQuestions(): Question[] {
  return Array.from(_byId.values());
}
