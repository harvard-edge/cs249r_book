import { VolumeId, BuildFormat } from '../types';
import { getBuildChannel } from './terminal';
import { getBuildEntriesForFormat } from './chapters';

const BAR = '═'.repeat(70);
const THIN = '─'.repeat(70);

export interface BuildManifestOptions {
  repoRoot: string;
  vol: VolumeId;
  format: BuildFormat;
  /** 'sequential' for series builds; 'parallel' for worktree-based parallel debug. */
  mode: 'sequential' | 'parallel';
  /** The exact shell command that will run in the terminal. */
  command: string;
  /** Worker count (parallel mode only). */
  workers?: number;
}

/**
 * Writes a pre-flight build manifest to the MLSysBook Build output channel.
 *
 * Reads the YML config (comment markers stripped) so the list always reflects
 * the full intended volume structure — not just what is currently uncommented.
 * Add a new chapter to the YML (even commented out) and it will appear here
 * and be included in the build after `binder reset` runs.
 */
export function showBuildManifest(opts: BuildManifestOptions): void {
  const { repoRoot, vol, format, mode, command, workers } = opts;
  const volLabel = vol === 'vol1' ? 'Volume I' : 'Volume II';
  const { configFile, relPaths } = getBuildEntriesForFormat(repoRoot, vol, format);
  const now = new Date().toLocaleString();
  const modeLabel =
    mode === 'parallel' ? `Parallel  (${workers ?? 4} workers)` : 'Sequential';

  const ch = getBuildChannel();
  ch.appendLine('');
  ch.appendLine(BAR);
  ch.appendLine('  MLSysBook  ·  Build Manifest');
  ch.appendLine(BAR);
  ch.appendLine('');
  ch.appendLine(`  Volume  :  ${volLabel}  (${vol})`);
  ch.appendLine(`  Format  :  ${format.toUpperCase()}`);
  ch.appendLine(`  Mode    :  ${modeLabel}`);
  ch.appendLine(`  Config  :  ${configFile}`);
  ch.appendLine(`  Started :  ${now}`);
  ch.appendLine('');

  if (relPaths.length === 0) {
    ch.appendLine('  (no entries found — check config path)');
  } else {
    ch.appendLine(`  Files to build  (${relPaths.length} total)`);
    ch.appendLine(`  ${THIN}`);
    relPaths.forEach((p, i) => {
      ch.appendLine(`    [${String(i + 1).padStart(2, '0')}]  ${p}`);
    });
  }

  ch.appendLine('');
  ch.appendLine(`  ${THIN}`);
  ch.appendLine(`  Command: ${command}`);
  ch.appendLine(`  ${THIN}`);
  ch.appendLine('');
  ch.appendLine('  Starting build in terminal...');
  ch.appendLine('');
  ch.appendLine(BAR);
  ch.appendLine('');

  ch.show(false);
}
