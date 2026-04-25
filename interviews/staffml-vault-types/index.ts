/**
 * Shared vault types — v1.0 (2026-04-21).
 *
 * Aligned with interviews/vault/schema/question_schema.yaml (LinkML, the
 * authoritative schema) and interviews/vault/schema/enums.py.
 *
 * Both the worker (interviews/staffml-vault-worker/) and the site
 * (interviews/staffml/) import from this package via pnpm workspace
 * protocol. CI drift checks against the Python enums in a follow-up PR.
 */

// ── Enums ───────────────────────────────────────────────────────────────────

export type Track = "cloud" | "edge" | "mobile" | "tinyml" | "global";

export type Level = "L1" | "L2" | "L3" | "L4" | "L5" | "L6+";

export type Zone =
  | "recall"
  | "analyze"
  | "design"
  | "implement"
  | "fluency"
  | "diagnosis"
  | "specification"
  | "optimization"
  | "evaluation"
  | "realization"
  | "mastery";

export type BloomLevel =
  | "remember" | "understand" | "apply" | "analyze" | "evaluate" | "create";

export type Phase = "training" | "inference" | "both";

export type Status = "draft" | "published" | "flagged" | "archived" | "deleted";

export type Provenance =
  | "human" | "llm-draft" | "llm-then-human-edited" | "imported";

export type HumanReviewStatus =
  | "not-reviewed" | "verified" | "flagged" | "needs-rework";

// ── Nested types ────────────────────────────────────────────────────────────

export interface ChainRef {
  id: string;
  position: number;
}

export interface Resource {
  name: string;
  url: string;
}

export interface HumanReview {
  status: HumanReviewStatus;
  by?: string | null;
  date?: string | null;
  notes?: string | null;
}

/**
 * Static visual attached to a question.
 *
 * v0.1.2 hardened: kind is a closed enum (svg only), path matches
 * `^[a-z0-9-]+\.svg$`, alt ≥10 chars, caption required ≥5 chars.
 * Server-side Pydantic enforces these constraints; this interface is
 * the shape the practice page consumes.
 */
export type VisualKind = "svg";

export interface Visual {
  kind: VisualKind;
  /** Bare filename under interviews/vault/visuals/<track>/. */
  path: string;
  /** Accessibility description. ≥10 chars; ≤400 chars. */
  alt: string;
  /** Author-facing caption rendered below the figure. ≥5 chars; ≤120 chars. */
  caption: string;
}

export interface QuestionDetails {
  realistic_solution: string;
  common_mistake?: string;
  napkin_math?: string;
  resources?: Resource[];
  options?: string[];
  correct_index?: number;
}

// ── Question ────────────────────────────────────────────────────────────────

export interface Question {
  schema_version: string;   // "1.0"
  id: string;

  // 4-axis classification
  track: Track;
  level: Level;
  zone: Zone;
  topic: string;
  competency_area: string;
  bloom_level?: BloomLevel;
  phase?: Phase;

  // Content
  title: string;
  scenario: string;
  question?: string;
  visual?: Visual;
  details: QuestionDetails;

  // Workflow
  status: Status;
  provenance: Provenance;
  requires_explanation?: boolean;
  expected_time_minutes?: number;
  deletion_reason?: string;

  // Chain membership (plural — a question may belong to multiple chains)
  chains?: ChainRef[];

  // LLM validation
  validated?: boolean;
  validation_status?: string;
  validation_date?: string;
  validation_model?: string;

  // Math validation (separate LLM pass)
  math_verified?: boolean;
  math_status?: string;
  math_date?: string;
  math_model?: string;

  // Human review (new in v1.0)
  human_reviewed?: HumanReview;

  // Free-form
  classification_review?: string;
  authors?: string[];
  tags?: string[];
  created_at?: string;
  updated_at?: string;
  last_modified?: string;
}

// ── Manifest / API ──────────────────────────────────────────────────────────

export interface Manifest {
  release_id: string;
  release_hash: string;
  schema_version: string;    // "1.0"
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
