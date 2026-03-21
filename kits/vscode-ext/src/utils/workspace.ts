import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';

/**
 * Find the kits project root.
 *
 * Looks for a directory containing `kits/Makefile`,
 * starting from workspace folders.
 */
export function getProjectRoot(): string | undefined {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders) { return undefined; }

  for (const folder of folders) {
    const root = folder.uri.fsPath;

    if (isKitsRoot(root)) {
      return root;
    }

    const parent = path.dirname(root);
    if (isKitsRoot(parent)) {
      return parent;
    }
  }

  return undefined;
}

function isKitsRoot(dir: string): boolean {
  return (
    fs.existsSync(path.join(dir, 'kits', 'Makefile')) &&
    fs.existsSync(path.join(dir, 'kits', 'contents'))
  );
}
