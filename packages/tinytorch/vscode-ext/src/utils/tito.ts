import { execSync } from 'child_process';
import * as vscode from 'vscode';

const OUTPUT_CHANNEL_NAME = 'TinyTorch';

let outputChannel: vscode.OutputChannel | undefined;

function getOutputChannel(): vscode.OutputChannel {
  if (!outputChannel) {
    outputChannel = vscode.window.createOutputChannel(OUTPUT_CHANNEL_NAME);
  }
  return outputChannel;
}

/** Log a message to the TinyTorch output channel (visible via View > Output) */
export function log(message: string): void {
  getOutputChannel().appendLine(`[${new Date().toLocaleTimeString()}] ${message}`);
}

/** Log an error and optionally show a user-facing warning */
export function logError(context: string, error: unknown, showWarning = false): void {
  const msg = error instanceof Error ? error.message : String(error);
  const channel = getOutputChannel();
  channel.appendLine(`[${new Date().toLocaleTimeString()}] ERROR (${context}): ${msg}`);

  if (showWarning) {
    void vscode.window.showWarningMessage(
      `TinyTorch: ${context} — ${msg}`,
      'Show Output',
    ).then(choice => {
      if (choice === 'Show Output') { channel.show(); }
    });
  }
}

/** Default timeout for Tito CLI calls (15 seconds) */
const TITO_TIMEOUT = 15_000;

/** The base command prefix for invoking Tito */
const TITO_CMD = 'python3 -m tito.main';

/** Standard exec options for calling Tito */
function titoExecOptions(projectRoot: string, timeout = TITO_TIMEOUT): {
  cwd: string; timeout: number; encoding: BufferEncoding;
  env: NodeJS.ProcessEnv;
} {
  return {
    cwd: projectRoot,
    timeout,
    encoding: 'utf-8' as BufferEncoding,
    env: { ...process.env, TITO_ALLOW_SYSTEM: '1' },
  };
}

/**
 * Call a Tito CLI command and return the raw stdout string.
 * Returns null on failure (and logs the error).
 */
export function callTito(
  projectRoot: string,
  subcommand: string,
  context: string,
  timeout = TITO_TIMEOUT,
): string | null {
  const cmd = `${TITO_CMD} ${subcommand}`;
  try {
    return execSync(cmd, titoExecOptions(projectRoot, timeout)).trim();
  } catch (err) {
    logError(context, err);
    return null;
  }
}

/**
 * Call a Tito CLI command that returns JSON.
 * Returns the parsed object on success, or null on failure (and logs the error).
 */
export function callTitoJson<T>(
  projectRoot: string,
  subcommand: string,
  context: string,
  timeout = TITO_TIMEOUT,
): T | null {
  const raw = callTito(projectRoot, subcommand, context, timeout);
  if (raw === null) { return null; }

  try {
    return JSON.parse(raw) as T;
  } catch (err) {
    logError(`${context} — invalid JSON`, err);
    return null;
  }
}

/**
 * Check whether the Tito CLI is importable from the current project root.
 * Returns true if `python3 -m tito.main --help` succeeds.
 */
export function isTitoAvailable(projectRoot: string): boolean {
  try {
    execSync(`${TITO_CMD} --help`, {
      ...titoExecOptions(projectRoot, 5000),
      stdio: 'pipe', // suppress stdout/stderr
    });
    return true;
  } catch {
    return false;
  }
}

/** Build the full terminal command string for a Tito subcommand */
export function titoTerminalCommand(subcommand: string): string {
  return `${TITO_CMD} ${subcommand}`;
}
