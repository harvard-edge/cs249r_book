import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';

/**
 * Find the TinyTorch project root.
 *
 * Looks for a directory containing both `src/01_tensor` and `pyproject.toml`,
 * starting from the workspace folders. This handles the case where the VS Code
 * workspace root is the monorepo and tinytorch/ is a subdirectory.
 */
export function getProjectRoot(): string | undefined {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders) { return undefined; }

  for (const folder of folders) {
    const root = folder.uri.fsPath;

    // Direct match: workspace IS the tinytorch project
    if (isTinyTorchRoot(root)) {
      return root;
    }

    // One level down: workspace contains tinytorch/
    const nested = path.join(root, 'tinytorch');
    if (isTinyTorchRoot(nested)) {
      return nested;
    }
  }

  return undefined;
}

function isTinyTorchRoot(dir: string): boolean {
  return (
    fs.existsSync(path.join(dir, 'src', '01_tensor')) &&
    fs.existsSync(path.join(dir, 'pyproject.toml'))
  );
}
