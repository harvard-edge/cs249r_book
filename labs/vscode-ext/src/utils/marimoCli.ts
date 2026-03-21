import * as vscode from 'vscode';
import { execSync } from 'child_process';
import { log } from './terminal';

/** Get the configured Marimo path */
export function getMarimoPath(): string {
  return vscode.workspace.getConfiguration('labs').get<string>('marimoPath', 'marimo');
}

/** Build a marimo command string */
export function marimoCommand(subcommand: string, filePath: string): string {
  return `${getMarimoPath()} ${subcommand} "${filePath}"`;
}

/** Check if marimo CLI is available */
export function isMarimoAvailable(): boolean {
  try {
    execSync(`${getMarimoPath()} --version`, {
      encoding: 'utf-8',
      timeout: 10_000,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    return true;
  } catch {
    return false;
  }
}

/** Get marimo version string */
export function getMarimoVersion(): string {
  try {
    const output = execSync(`${getMarimoPath()} --version`, {
      encoding: 'utf-8',
      timeout: 5_000,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    return output.trim().replace('marimo ', '');
  } catch {
    return 'unavailable';
  }
}

/** Get Python version string */
export function getPythonVersion(): string {
  const pythonPath = vscode.workspace.getConfiguration('labs').get<string>('pythonPath', 'python3');
  try {
    const output = execSync(`${pythonPath} --version`, {
      encoding: 'utf-8',
      timeout: 5_000,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    return output.trim().replace('Python ', '');
  } catch {
    return 'unavailable';
  }
}
