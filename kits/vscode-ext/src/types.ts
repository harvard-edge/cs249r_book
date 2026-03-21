/** A recorded command run */
export interface CommandRunRecord {
  id: string;
  label: string;
  command: string;
  cwd: string;
  timestamp: string;
  status: 'started' | 'succeeded' | 'failed';
}

/** Hardware platform */
export interface Platform {
  id: string;
  name: string;
  description: string;
  icon: string;
  hubPath: string;
  labs: Lab[];
}

/** A lab within a platform */
export interface Lab {
  id: string;
  name: string;
  path: string;
  category: string;
}
