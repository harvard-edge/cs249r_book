/** Status of a module from progress.json */
export type ModuleStatus = 'completed' | 'started' | 'not_started';

/** A discovered TinyTorch module */
export interface ModuleInfo {
  /** Two-digit number, e.g. "01" */
  number: string;
  /** Folder name, e.g. "01_tensor" */
  folder: string;
  /** Display name, e.g. "Tensor" */
  displayName: string;
  /** Title from module.yaml if available */
  title?: string;
  /** Current status */
  status: ModuleStatus;
}

/** Shape of progress.json */
export interface ProgressData {
  started_modules: string[];
  completed_modules: string[];
  last_worked: string | null;
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
