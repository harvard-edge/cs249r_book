import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';

/**
 * Find the mlsysim project root.
 *
 * Looks for a directory containing `mlsysim/__init__.py`,
 * starting from workspace folders. Handles both monorepo
 * (workspace root contains mlsysim/) and direct open.
 */
export function getProjectRoot(): string | undefined {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders) { return undefined; }

  for (const folder of folders) {
    const root = folder.uri.fsPath;

    // Direct match: workspace IS the mlsysim project's parent
    if (isMlsysimRoot(root)) {
      return root;
    }

    // One level up: the workspace might be inside the monorepo
    const parent = path.dirname(root);
    if (isMlsysimRoot(parent)) {
      return parent;
    }
  }

  return undefined;
}

function isMlsysimRoot(dir: string): boolean {
  return (
    fs.existsSync(path.join(dir, 'mlsysim', '__init__.py')) &&
    fs.existsSync(path.join(dir, 'mlsysim', 'core', 'solver.py'))
  );
}
