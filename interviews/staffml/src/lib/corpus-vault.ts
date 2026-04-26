/**
 * Vault-API-backed corpus data source.
 *
 * Mirror of the public surface of ``corpus.ts`` but sourced from the
 * staffml-vault Worker via ``vault-api.ts`` instead of the bundled
 * ``corpus.json``. Not wired into any component until cutover — the
 * switch happens via ``corpus-source.ts``.
 *
 * Post-v1.0 (2026-04-21): the vault schema now carries track/level/zone
 * as YAML fields and uses plural `chains: [{id, position}]`, so this
 * adapter's job shrinks considerably. The defaulting to
 * `track='global'`/`level='l1'`/`zone='recall'` that existed here was
 * exactly the silent-mis-classification pattern that hid the v0.1
 * migration bug; those defaults are gone.
 */

import type { Question as VaultQuestion } from "@staffml/vault-types";
import { makeClientFromEnv, VaultApiClient } from "./vault-api";

// v1.0: classification lives on the Question itself.
type EnrichedVaultQuestion = VaultQuestion & {
  track: string;
  level: string;
  zone: string;
  competency_area: string;
  bloom_level?: string;
  phase?: string;
  question?: string;
  visual?: {
    kind: "svg";              // closed enum as of v0.1.2 (mermaid retired)
    path: string;
    alt: string;              // ≥10 chars (a11y)
    caption: string;          // required as of v0.1.2, ≥5 chars
  };
  chains?: Array<{ id: string; position: number }>;
  validated?: boolean;
  math_verified?: boolean;
  human_reviewed?: {
    status: string;
    by?: string | null;
    date?: string | null;
  };
};

// Shape the UI already expects (see corpus.ts).
export interface Question {
  id: string;
  track: string;
  level: string;
  title: string;
  topic: string;
  zone: string;
  competency_area: string;
  bloom_level?: string;
  phase?: string;
  scenario: string;
  question?: string;
  visual?: {
    kind: "svg";              // closed enum as of v0.1.2 (mermaid retired)
    path: string;
    alt: string;              // ≥10 chars (a11y)
    caption: string;          // required as of v0.1.2, ≥5 chars
  };
  chain_ids?: string[];
  chain_positions?: Record<string, number>;
  details: {
    common_mistake: string;
    realistic_solution: string;
    napkin_math?: string;
  };
  validated?: boolean;
  math_verified?: boolean;
  human_reviewed?: {
    status: string;
    by?: string | null;
    date?: string | null;
  };
}

function adapt(v: EnrichedVaultQuestion): Question {
  // Rebuild legacy chain_ids + chain_positions from the plural `chains` list.
  const chainIds: string[] = [];
  const chainPositions: Record<string, number> = {};
  for (const c of v.chains ?? []) {
    chainIds.push(c.id);
    chainPositions[c.id] = c.position;
  }
  return {
    id: v.id,
    track: v.track,
    level: v.level,
    title: v.title,
    topic: v.topic,
    zone: v.zone,
    competency_area: v.competency_area,
    bloom_level: v.bloom_level,
    phase: v.phase,
    scenario: v.scenario,
    question: v.question,
    visual: v.visual,
    chain_ids: chainIds.length ? chainIds : undefined,
    chain_positions: chainIds.length ? chainPositions : undefined,
    details: {
      common_mistake: v.details.common_mistake ?? "",
      realistic_solution: v.details.realistic_solution,
      napkin_math: v.details.napkin_math,
    },
    validated: v.validated,
    math_verified: v.math_verified,
    human_reviewed: v.human_reviewed,
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
