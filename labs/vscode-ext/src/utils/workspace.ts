import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';

/**
 * Find the labs project root.
 *
 * Looks for a directory containing `labs/PROTOCOL.md`,
 * starting from workspace folders. Handles both monorepo
 * (workspace root contains labs/) and direct open.
 */
export function getProjectRoot(): string | undefined {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders) { return undefined; }

  for (const folder of folders) {
    const root = folder.uri.fsPath;

    if (isLabsRoot(root)) {
      return root;
    }

    const parent = path.dirname(root);
    if (isLabsRoot(parent)) {
      return parent;
    }
  }

  return undefined;
}

function isLabsRoot(dir: string): boolean {
  return (
    fs.existsSync(path.join(dir, 'labs', 'PROTOCOL.md')) &&
    fs.existsSync(path.join(dir, 'labs', 'core', 'state.py'))
  );
}
