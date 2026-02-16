import * as fs from 'fs';
import * as path from 'path';
import { ModuleInfo, ModuleStatus, ProgressData } from '../types';

/**
 * Discover all TinyTorch modules by scanning the src/ directory.
 *
 * Mirrors the logic in tito/core/modules.py — looks for directories
 * matching the pattern NN_name (e.g. 01_tensor, 15_quantization).
 */
export function discoverModules(projectRoot: string): ModuleInfo[] {
  const srcDir = path.join(projectRoot, 'src');
  if (!fs.existsSync(srcDir)) { return []; }

  const progress = readProgress(projectRoot);
  const pattern = /^(\d{2})_(\w+)$/;

  const entries = fs.readdirSync(srcDir, { withFileTypes: true })
    .filter(e => e.isDirectory() && pattern.test(e.name))
    .sort((a, b) => a.name.localeCompare(b.name));

  return entries.map(entry => {
    const match = entry.name.match(pattern)!;
    const num = match[1];
    const slug = match[2];
    const displayName = slug.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

    // Try to read title from module.yaml
    const title = readModuleTitle(path.join(srcDir, entry.name));

    // Determine status from progress.json
    let status: ModuleStatus = 'not_started';
    if (progress.completed_modules.includes(num)) {
      status = 'completed';
    } else if (progress.started_modules.includes(num)) {
      status = 'started';
    }

    return { number: num, folder: entry.name, displayName, title, status };
  });
}

/** Read progress.json from the project root */
function readProgress(projectRoot: string): ProgressData {
  const filePath = path.join(projectRoot, 'progress.json');
  const empty: ProgressData = { started_modules: [], completed_modules: [], last_worked: null };

  try {
    if (!fs.existsSync(filePath)) { return empty; }
    const raw = fs.readFileSync(filePath, 'utf-8');
    const data = JSON.parse(raw) as Partial<ProgressData>;
    return {
      started_modules: data.started_modules ?? [],
      completed_modules: data.completed_modules ?? [],
      last_worked: data.last_worked ?? null,
    };
  } catch {
    return empty;
  }
}

/** Read title from a module's module.yaml (simple key: value parsing) */
function readModuleTitle(moduleDir: string): string | undefined {
  const yamlPath = path.join(moduleDir, 'module.yaml');
  try {
    if (!fs.existsSync(yamlPath)) { return undefined; }
    const content = fs.readFileSync(yamlPath, 'utf-8');
    for (const line of content.split('\n')) {
      const trimmed = line.trim();
      if (trimmed.startsWith('title:')) {
        return trimmed.slice('title:'.length).trim();
      }
    }
  } catch {
    // Silently ignore — title is optional
  }
  return undefined;
}
