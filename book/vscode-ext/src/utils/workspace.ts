import * as vscode from 'vscode';
import { QmdFileContext, VolumeId } from '../types';

export function getRepoRoot(): string | undefined {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) { return undefined; }
  return folders[0].uri.fsPath;
}

export function parseQmdFile(uri: vscode.Uri): QmdFileContext | undefined {
  const match = uri.fsPath.match(/contents\/(vol[12])\/([^/]+)\//);
  if (!match) { return undefined; }
  return {
    volume: match[1] as VolumeId,
    chapter: match[2],
  };
}
