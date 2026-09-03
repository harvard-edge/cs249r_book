import * as vscode from 'vscode';
import { execSync } from 'child_process';
import { log } from './terminal';

/** Get the configured Python path */
function getPythonPath(): string {
  return vscode.workspace.getConfiguration('mlsysim').get<string>('pythonPath', 'python3');
}

/** Build a CLI command string */
export function mlsysimCommand(subcommand: string): string {
  return `${getPythonPath()} -m mlsysim ${subcommand}`;
}

/** Run mlsysim CLI and parse JSON output (synchronous, for tree providers) */
export function callMlsysimJson<T>(cwd: string, subcommand: string, label: string, timeout = 30_000): T | null {
  try {
    const cmd = `${getPythonPath()} -m mlsysim -o json ${subcommand}`;
    const stdout = execSync(cmd, {
      cwd,
      encoding: 'utf-8',
      timeout,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    return JSON.parse(stdout) as T;
  } catch (err) {
    log(`${label} failed: ${err}`);
    return null;
  }
}

/** Check if mlsysim CLI is available */
export function isMlsysimAvailable(cwd: string): boolean {
  try {
    execSync(`${getPythonPath()} -m mlsysim --help`, {
      cwd,
      encoding: 'utf-8',
      timeout: 10_000,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    return true;
  } catch {
    return false;
  }
}

/** Get Python version string */
export function getPythonVersion(cwd: string): string {
  try {
    const output = execSync(`${getPythonPath()} --version`, {
      cwd,
      encoding: 'utf-8',
      timeout: 5_000,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    return output.trim().replace('Python ', '');
  } catch {
    return 'unavailable';
  }
}
