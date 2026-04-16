/**
 * Shared vault types — sole TS source of truth. Codegen'd from LinkML in
 * Phase 2+ via `vault codegen`. During Phase 3 this file is hand-authored
 * to match vault_cli.models; CI's codegen-drift check flags divergence.
 *
 * Both the worker (interviews/staffml-vault-worker/) and the site
 * (interviews/staffml/) import from this package via pnpm workspace
 * protocol (REVIEWS.md Soumith M-NEW-1).
 */

export type Track = "cloud" | "edge" | "mobile" | "tinyml" | "global";
export type Level = "l1" | "l2" | "l3" | "l4" | "l5" | "l6";
export type Zone =
  | "recall"
  | "fluency"
  | "implement"
  | "specification"
  | "analyze"
  | "diagnosis"
  | "design"
  | "evaluation";
export type Status = "draft" | "published" | "deprecated";
export type Provenance = "human" | "llm-draft" | "llm-then-human-edited" | "imported";

export interface ChainRef {
  id: string;
  position: number;
}

export interface GenerationMeta {
  model?: string;
  prompt_hash?: string;
  prompt_cost_usd?: number;
  human_reviewed_at?: string;
}

export interface DeepDive {
  title: string;
  url: string;
}

export interface QuestionDetails {
  common_mistake?: string;
  realistic_solution: string;
  napkin_math?: string;
  deep_dive?: DeepDive;
}

export interface Question {
  schema_version: number;
  id: string;
  title: string;
  topic: string;
  chain?: ChainRef;
  status: Status;
  created_at?: string;
  last_modified?: string;
  provenance: Provenance;
  generation_meta?: GenerationMeta;
  authors?: string[];
  scenario: string;
  details: QuestionDetails;
  tags?: string[];
}

export interface Manifest {
  release_id: string;
  release_hash: string;
  schema_version: string;
  policy_version: string;
  published_count: number;
  schema_fingerprint_ok: boolean;
}

export interface VaultApiClientOptions {
  release: string;
  retry?: { attempts: number; backoff: "exponential" | "linear"; jitter?: boolean };
  circuitBreaker?: { failThreshold: number; resetMs: number };
  headers?: Record<string, string>;
}
