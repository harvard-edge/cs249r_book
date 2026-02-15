/**
 * Helpers for resetting Quarto config YAML (uncomment all chapter/appendix entries).
 * Delegates to the binder: ./book/binder <format> reset [--vol1|--vol2]
 */

import type { BuildFormat, VolumeId } from '../types';

/** Reset command for one format and one volume (e.g. "pdf reset --vol1"). */
export function getQuartoResetCommand(format: BuildFormat, volume: VolumeId): string {
  return `./book/binder ${format} reset --${volume}`;
}

/** Reset command for one format, both volumes (used by "Reset Quarto config"). */
export function getQuartoResetAllFormatsCommand(): string {
  return './book/binder pdf reset && ./book/binder html reset && ./book/binder epub reset';
}

/** Prefix to run before a build so config is uncommented: reset then build. */
export function withQuartoResetPrefix(
  format: BuildFormat,
  volume: VolumeId,
  buildCommand: string,
): string {
  return `${getQuartoResetCommand(format, volume)} && ${buildCommand}`;
}
