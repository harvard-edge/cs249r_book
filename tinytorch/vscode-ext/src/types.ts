/** Status of a module (from tito module list --json) */
export type ModuleStatus = 'completed' | 'started' | 'not_started';

/** A discovered TinyTorch module (from tito module list --json) */
export interface ModuleInfo {
  /** Two-digit number, e.g. "01" */
  number: string;
  /** Folder name, e.g. "01_tensor" */
  folder: string;
  /** Display name / title */
  displayName: string;
  /** Title from module.yaml if available */
  title?: string;
  /** Current status */
  status: ModuleStatus;
}

/** A recorded command run */
export interface CommandRunRecord {
  id: string;
  label: string;
  command: string;
  cwd: string;
  timestamp: string;
  status: 'started' | 'succeeded' | 'failed';
}
