/**
 * StaffML Vault Worker types.
 *
 * Mirror of @staffml/vault-types (codegen'd from LinkML). During Phase 3 the
 * shared package is defined at interviews/staffml-vault-types/ via pnpm
 * workspace protocol (ARCHITECTURE.md §13 fix for H-2).
 */

export interface Env {
  DB: D1Database;
  RATE_LIMIT_KV?: KVNamespace;
  CACHE_TTL_MANIFEST: string;
  CACHE_TTL_QUESTION: string;
  CACHE_TTL_SEARCH: string;
  CACHE_TTL_TAXONOMY: string;
  CORS_ALLOWLIST: string;
  SCHEMA_FINGERPRINT: string;
  GRACE_WINDOW_SECONDS: string;
  RATE_LIMIT_RPM_DEFAULT?: string;
  RATE_LIMIT_RPM_SEARCH?: string;
}

export interface QuestionRow {
  id: string;
  title: string;
  topic: string;
  track: string;
  level: string;
  zone: string;
  /** v1.0: 13 canonical competency areas (memory, compute, latency, …). */
  competency_area: string | null;
  /** v1.0: Bloom's revised taxonomy verb (remember/understand/apply/analyze/evaluate/create). */
  bloom_level: string | null;
  /** v1.0: ML lifecycle phase (training/inference/both). */
  phase: string | null;
  status: string;
  scenario: string;
  common_mistake: string | null;
  realistic_solution: string;
  napkin_math: string | null;
  provenance: string;
  created_at: string | null;
  last_modified: string | null;
  /** v1.0: human verification status, independent of LLM validation. */
  human_review_status: string | null;
  human_review_by: string | null;
  human_review_date: string | null;
  content_hash: string;
  authors_json: string | null;
}

export interface Manifest {
  release_id: string;
  release_hash: string;
  schema_version: string;
  policy_version: string;
  published_count: number;
  schema_fingerprint_ok: boolean;
}

export interface Cursor {
  /**
   * Keyset cursor — ``after_id`` is the last row's ``id`` in the previous
   * page. Server pages by ``WHERE id > after_id ORDER BY id LIMIT N`` so
   * pagination cost is O(N) per page, not O(offset + N). (Chip R3-H2 fix.)
   */
  after_id: string;
  /** Hash of the filter params at cursor-mint time; server rejects cross-filter reuse. */
  filter_hash: string;
}

export type DegradedReason = "schema-fingerprint-mismatch" | "db-unavailable";
