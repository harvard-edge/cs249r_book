import * as vscode from 'vscode';
import { VolumeId } from '../types';
import { getRepoRoot, parseQmdFile } from '../utils/workspace';
import { runBookCommand } from '../utils/terminal';
import { discoverChapters } from '../utils/chapters';
import {
  cancelActiveDebugSession,
  getDebugSessionById,
  getLastFailedDebugSession,
  revealParallelDebugOutput,
  rerunDebugSession,
  runBisectChapterDebug,
  runIsolatedDebugCommand,
  runParallelChapterDebug,
} from '../utils/parallelDebug';

const STATE_LAST_PARALLEL_VOLUME = 'mlsysbook.lastParallelVolume';
const STATE_LAST_PARALLEL_WORKERS = 'mlsysbook.lastParallelWorkers';

async function pickVolume(defaultVolume: VolumeId | undefined, placeHolder: string): Promise<VolumeId | undefined> {
  const picks = [
    { label: defaultVolume === 'vol1' ? 'Volume I (last used)' : 'Volume I', id: 'vol1' as VolumeId },
    { label: defaultVolume === 'vol2' ? 'Volume II (last used)' : 'Volume II', id: 'vol2' as VolumeId },
  ];
  const selection = await vscode.window.showQuickPick(picks, { placeHolder });
  return selection?.id;
}

async function pickWorkers(defaultWorkers: number): Promise<number | undefined> {
  const workerPick = await vscode.window.showQuickPick(
    ['1', '2', '3', '4', '6', '8'],
    { placeHolder: `Parallel workers (default: ${defaultWorkers})` },
  );
  if (!workerPick) {
    return defaultWorkers;
  }
  return Number(workerPick);
}

async function runParallelDebugWizard(
  root: string,
  context: vscode.ExtensionContext,
  forcePdf: boolean,
): Promise<void> {
  const defaultVolume = context.workspaceState.get<VolumeId>(STATE_LAST_PARALLEL_VOLUME);
  const volumeId = await pickVolume(defaultVolume, 'Select volume for parallel chapter debug');
  if (!volumeId) { return; }

  const volumes = discoverChapters(root);
  const volume = volumes.find(v => v.id === volumeId);
  if (!volume || volume.chapters.length === 0) {
    vscode.window.showWarningMessage(`No chapters found for ${volumeId}.`);
    return;
  }

  const chapterPicks = await vscode.window.showQuickPick(
    volume.chapters.map(ch => ({
      label: ch.displayName,
      description: ch.name,
      chapter: ch.name,
    })),
    {
      placeHolder: 'Select chapters to debug in parallel',
      canPickMany: true,
      matchOnDescription: true,
    },
  );
  if (!chapterPicks || chapterPicks.length === 0) { return; }

  let format = 'pdf';
  if (!forcePdf) {
    const fmtPick = await vscode.window.showQuickPick(
      ['pdf', 'html', 'epub'],
      { placeHolder: 'Select debug format (default: pdf)' },
    );
    format = fmtPick ?? 'pdf';
  }

  const defaultWorkers = context.workspaceState.get<number>(
    STATE_LAST_PARALLEL_WORKERS,
    vscode.workspace.getConfiguration('mlsysbook').get<number>('parallelDebugWorkers', 3),
  );
  const workers = await pickWorkers(defaultWorkers);
  if (!workers) { return; }

  await context.workspaceState.update(STATE_LAST_PARALLEL_VOLUME, volumeId);
  await context.workspaceState.update(STATE_LAST_PARALLEL_WORKERS, workers);

  await runParallelChapterDebug({
    repoRoot: root,
    volume: volumeId,
    format,
    chapters: chapterPicks.map(p => p.chapter),
    workers,
  });
}

export function registerDebugCommands(context: vscode.ExtensionContext): void {
  const root = getRepoRoot();
  if (!root) { return; }

  // Debug full volume
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.debugVolumePdf', (vol: VolumeId) => {
      void runIsolatedDebugCommand({
        repoRoot: root,
        command: `./book/binder debug pdf --${vol}`,
        label: `Debug Volume PDF (${vol})`,
      });
    })
  );

  // Debug chapter sections (interactive QuickPick)
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.debugChapterSections', async () => {
      const volPick = await vscode.window.showQuickPick(
        [
          { label: 'Volume I', id: 'vol1' as VolumeId },
          { label: 'Volume II', id: 'vol2' as VolumeId },
        ],
        { placeHolder: 'Select volume' },
      );
      if (!volPick) { return; }

      const volumes = discoverChapters(root);
      const volume = volumes.find(v => v.id === volPick.id);
      if (!volume) { return; }

      const chapterPick = await vscode.window.showQuickPick(
        volume.chapters.map(ch => ({ label: ch.displayName, description: ch.name, id: ch.name })),
        { placeHolder: 'Select chapter to debug' },
      );
      if (!chapterPick) { return; }

      const fmtPick = await vscode.window.showQuickPick(
        ['pdf', 'html', 'epub'],
        { placeHolder: 'Select format (default: pdf)' },
      );
      const fmt = fmtPick ?? 'pdf';

      void runIsolatedDebugCommand({
        repoRoot: root,
        command: `./book/binder debug ${fmt} --${volPick.id} --chapter ${chapterPick.id}`,
        label: `Debug Chapter ${fmt.toUpperCase()} (${volPick.id}/${chapterPick.id})`,
      });
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.debugParallelChapters', async () => {
      await runParallelDebugWizard(root, context, false);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.debugBisectChapters', async () => {
      const defaultVolume = context.workspaceState.get<VolumeId>(STATE_LAST_PARALLEL_VOLUME);
      const volumeId = await pickVolume(defaultVolume, 'Select volume for bisect debug');
      if (!volumeId) { return; }

      const volumes = discoverChapters(root);
      const volume = volumes.find(v => v.id === volumeId);
      if (!volume || volume.chapters.length === 0) {
        vscode.window.showWarningMessage(`No chapters found for ${volumeId}.`);
        return;
      }

      const candidateMode = await vscode.window.showQuickPick(
        [
          { label: 'Use all chapters in this volume', id: 'all' },
          { label: 'Select candidate chapters manually', id: 'manual' },
        ],
        { placeHolder: 'Choose bisect candidate set' },
      );
      if (!candidateMode) { return; }

      let chapters = volume.chapters.map(ch => ch.name);
      if (candidateMode.id === 'manual') {
        const picks = await vscode.window.showQuickPick(
          volume.chapters.map(ch => ({
            label: ch.displayName,
            description: ch.name,
            chapter: ch.name,
          })),
          {
            placeHolder: 'Select candidate chapters for bisect',
            canPickMany: true,
            matchOnDescription: true,
          },
        );
        if (!picks || picks.length === 0) { return; }
        chapters = picks.map(p => p.chapter);
      }

      const fmtPick = await vscode.window.showQuickPick(
        ['pdf', 'html', 'epub'],
        { placeHolder: 'Select debug format (default: pdf)' },
      );
      const fmt = fmtPick ?? 'pdf';

      const defaultWorkers = vscode.workspace
        .getConfiguration('mlsysbook')
        .get<number>('parallelDebugWorkers', 3);
      const workers = await pickWorkers(defaultWorkers);
      if (!workers) { return; }

      await context.workspaceState.update(STATE_LAST_PARALLEL_VOLUME, volumeId);
      await context.workspaceState.update(STATE_LAST_PARALLEL_WORKERS, workers);

      await runBisectChapterDebug({
        repoRoot: root,
        volume: volumeId,
        format: fmt,
        chapters,
        workers,
      });
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

  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.quickBuildCurrentChapterPdf', () => {
      const uri = vscode.window.activeTextEditor?.document.uri;
      if (!uri || !uri.fsPath.endsWith('.qmd')) {
        vscode.window.showWarningMessage('Open a chapter .qmd file to run quick chapter PDF build.');
        return;
      }
      const parsed = parseQmdFile(uri);
      if (!parsed) {
        vscode.window.showWarningMessage('Could not determine volume/chapter for active file.');
        return;
      }
      void runBookCommand(`./book/binder build pdf ${parsed.chapter} --${parsed.volume} -v`, root, {
        label: `Quick Chapter PDF (${parsed.volume}/${parsed.chapter})`,
      });
    }),
    vscode.commands.registerCommand('mlsysbook.quickParallelDebugSelectedPdf', async () => {
      await runParallelDebugWizard(root, context, true);
    }),
  );
}
