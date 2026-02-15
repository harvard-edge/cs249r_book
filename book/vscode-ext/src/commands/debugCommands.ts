import * as vscode from 'vscode';
import * as path from 'path';
import { VolumeId } from '../types';
import { getRepoRoot } from '../utils/workspace';
import { discoverChapters } from '../utils/chapters';
import { runInVisibleTerminal } from '../utils/terminal';
import {
  cancelActiveDebugSession,
  getDebugSessionById,
  getLastFailedDebugSession,
  revealParallelDebugOutput,
  rerunDebugSession,
  runParallelChapterDebug,
} from '../utils/parallelDebug';

const STATE_LAST_PARALLEL_VOLUME = 'mlsysbook.lastParallelVolume';

/**
 * Test All Chapters — prompts for volume only, then builds every chapter
 * in parallel using PDF format and the configured worker count.
 */
async function runTestAllChapters(
  root: string,
  context: vscode.ExtensionContext,
): Promise<void> {
  const defaultVolume = context.workspaceState.get<VolumeId>(STATE_LAST_PARALLEL_VOLUME);
  const picks = [
    { label: defaultVolume === 'vol1' ? 'Volume I (last used)' : 'Volume I', id: 'vol1' as VolumeId },
    { label: defaultVolume === 'vol2' ? 'Volume II (last used)' : 'Volume II', id: 'vol2' as VolumeId },
  ];
  const selection = await vscode.window.showQuickPick(picks, { placeHolder: 'Select volume to test' });
  if (!selection) { return; }
  const volumeId = selection.id;

  const volumes = discoverChapters(root);
  const volume = volumes.find(v => v.id === volumeId);
  if (!volume || volume.chapters.length === 0) {
    vscode.window.showWarningMessage(`No chapters found for ${volumeId}.`);
    return;
  }

  const workers = vscode.workspace
    .getConfiguration('mlsysbook')
    .get<number>('parallelDebugWorkers', 4);

  await context.workspaceState.update(STATE_LAST_PARALLEL_VOLUME, volumeId);

  const allChapters = volume.chapters.map(ch => ch.name);
  vscode.window.showInformationMessage(
    `Testing ${allChapters.length} chapters (PDF, ${workers} workers)...`
  );

  await runParallelChapterDebug({
    repoRoot: root,
    volume: volumeId,
    format: 'pdf',
    chapters: allChapters,
    workers,
  });
}

/**
 * Debug All Chapters (Sequential) — runs `./binder debug pdf --vol1` in a
 * visible terminal.  This builds each chapter one-by-one inside the current
 * repo (no worktrees), reports pass/fail, and binary-searches any failures.
 */
async function runDebugAllChapters(
  root: string,
  context: vscode.ExtensionContext,
): Promise<void> {
  const defaultVolume = context.workspaceState.get<VolumeId>(STATE_LAST_PARALLEL_VOLUME);
  const picks = [
    { label: defaultVolume === 'vol1' ? 'Volume I (last used)' : 'Volume I', id: 'vol1' as VolumeId },
    { label: defaultVolume === 'vol2' ? 'Volume II (last used)' : 'Volume II', id: 'vol2' as VolumeId },
  ];
  const selection = await vscode.window.showQuickPick(picks, {
    placeHolder: 'Select volume to debug (sequential build)',
  });
  if (!selection) { return; }

  await context.workspaceState.update(STATE_LAST_PARALLEL_VOLUME, selection.id);

  const bookDir = path.join(root, 'book');
  const cmd = `./binder debug pdf --${selection.id} -v`;
  runInVisibleTerminal(cmd, bookDir, `Debug All Chapters (${selection.id})`);
}

export function registerDebugCommands(context: vscode.ExtensionContext): void {
  const root = getRepoRoot();
  if (!root) { return; }

  // Debug All Chapters (Sequential) — runs binder debug in a terminal
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.debugAllChapters', async () => {
      await runDebugAllChapters(root, context);
    })
  );

  // Test All Chapters (Parallel) — worktree-based parallel build
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.testAllChaptersParallel', async () => {
      await runTestAllChapters(root, context);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.cancelParallelSession', () => {
      const cancelled = cancelActiveDebugSession();
      if (!cancelled) {
        vscode.window.showInformationMessage('No active parallel session to cancel.');
        return;
      }
      vscode.window.showWarningMessage('Requested cancellation for active parallel session.');
    }),
    vscode.commands.registerCommand('mlsysbook.rerunFailedParallel', async () => {
      const lastFailed = getLastFailedDebugSession();
      if (!lastFailed) {
        vscode.window.showInformationMessage('No recent failed parallel/bisect session found.');
        return;
      }
      const ok = await rerunDebugSession(lastFailed.id, true);
      if (!ok) {
        vscode.window.showWarningMessage('Unable to rerun failed chapters for the last failed session.');
      }
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.historyRerunSession', async (sessionId: string) => {
      const ok = await rerunDebugSession(sessionId, false);
      if (!ok) {
        vscode.window.showWarningMessage('Unable to rerun selected session.');
      }
    }),
    vscode.commands.registerCommand('mlsysbook.historyRerunFailed', async (sessionId: string) => {
      const ok = await rerunDebugSession(sessionId, true);
      if (!ok) {
        vscode.window.showWarningMessage('Unable to rerun failed chapters for selected session.');
      }
    }),
    vscode.commands.registerCommand('mlsysbook.historyOpenOutput', () => {
      revealParallelDebugOutput();
    }),
    vscode.commands.registerCommand('mlsysbook.historyOpenFailedWorktree', async (sessionId: string) => {
      const session = getDebugSessionById(sessionId);
      if (!session) {
        vscode.window.showWarningMessage('Run session not found.');
        return;
      }
      const firstPath = Object.values(session.failedWorktrees)[0];
      if (!firstPath) {
        vscode.window.showInformationMessage('Selected run has no retained failed worktree.');
        return;
      }
      await vscode.env.openExternal(vscode.Uri.file(firstPath));
    }),
    vscode.commands.registerCommand('mlsysbook.historyOpenFailureLocation', async (_sessionId: string, filePath: string, line: number) => {
      const document = await vscode.workspace.openTextDocument(vscode.Uri.file(filePath));
      const editor = await vscode.window.showTextDocument(document, { preview: false });
      const position = new vscode.Position(Math.max(0, line - 1), 0);
      editor.selection = new vscode.Selection(position, position);
      editor.revealRange(new vscode.Range(position, position), vscode.TextEditorRevealType.InCenter);
    }),
  );

}
