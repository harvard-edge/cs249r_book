import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { QmdFileContext, VolumeId } from '../types';

export function getRepoRoot(): string | undefined {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) { return undefined; }

  // First prefer a folder that directly contains book/binder.
  for (const folder of folders) {
    const rootCandidate = folder.uri.fsPath;
    if (fs.existsSync(path.join(rootCandidate, 'book', 'binder'))) {
      return rootCandidate;
    }
  }

  // Fallback: walk upward from the active file folder.
  const activeFile = vscode.window.activeTextEditor?.document.uri.fsPath;
  if (activeFile) {
    let cursor = path.dirname(activeFile);
    while (cursor !== path.dirname(cursor)) {
      if (fs.existsSync(path.join(cursor, 'book', 'binder'))) {
        return cursor;
      }
      cursor = path.dirname(cursor);
    }
  }

  // No book/binder found — do not return a root. Extension will bail early.
  return undefined;
}

export function parseQmdFile(uri: vscode.Uri): QmdFileContext | undefined {
  const match = uri.fsPath.match(/contents\/(vol[12])\/([^/]+)\//);
  if (!match) { return undefined; }
  return {
    volume: match[1] as VolumeId,
    chapter: match[2],
  };
}
