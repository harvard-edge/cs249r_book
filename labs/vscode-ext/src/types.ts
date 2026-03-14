/** A recorded command run */
export interface CommandRunRecord {
  id: string;
  label: string;
  command: string;
  cwd: string;
  timestamp: string;
  status: 'started' | 'succeeded' | 'failed';
}

/** Lab status derived from filesystem presence */
export type LabStatus = 'implemented' | 'planned' | 'missing';

/** A discovered lab */
export interface LabInfo {
  /** Lab number, e.g. "01" */
  number: string;
  /** Display title derived from filename, e.g. "ML Intro" */
  title: string;
  /** Volume: 1 or 2 */
  volume: number;
  /** Full path to the lab .py file (if it exists) */
  labPath?: string;
  /** Full path to the plan .md file (if it exists) */
  planPath?: string;
  /** Current status */
  status: LabStatus;
  /** Filename slug, e.g. "lab_01_ml_intro" */
  slug: string;
}

/** Design Ledger state (matches labs/core/state.py LedgerState) */
export interface LedgerState {
  track: string | null;
  current_chapter: number;
  history: Array<{
    chapter: number;
    design: Record<string, unknown>;
  }>;
  last_updated: string;
}
