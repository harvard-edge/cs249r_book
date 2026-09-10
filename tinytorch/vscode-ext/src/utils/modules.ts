import { callTitoJson, logError } from './tito';
import { ModuleInfo } from '../types';

/** Raw module record returned by `tito module list --json` */
interface TitoModuleRecord {
  number: string;
  folder: string;
  title: string;
  description: string;
  status: string;
}

/**
 * Discover all TinyTorch modules by calling `tito module list --json`.
 *
 * This delegates all logic (filesystem scanning, progress tracking,
 * metadata reading) to the Tito CLI — the extension never re-implements
 * discovery logic.
 *
 * Returns { modules, error } so callers can distinguish between
 * "no modules installed" and "Tito failed".
 */
export function discoverModules(projectRoot: string): { modules: ModuleInfo[]; error?: string } {
  const records = callTitoJson<TitoModuleRecord[]>(
    projectRoot,
    'module list --json',
    'module discovery',
  );

  if (records === null) {
    return {
      modules: [],
      error: 'Failed to discover modules — check the TinyTorch output channel for details.',
    };
  }

  if (!Array.isArray(records)) {
    logError('module discovery', 'Expected JSON array from tito module list --json');
    return { modules: [], error: 'Unexpected response from Tito CLI.' };
  }

  return {
    modules: records.map(m => ({
      number: m.number,
      folder: m.folder,
      displayName: m.title,
      title: m.title,
      status: m.status as ModuleInfo['status'],
    })),
  };
}
