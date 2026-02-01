import * as vscode from 'vscode';
import { VolumeId } from '../types';
import { getRepoRoot } from '../utils/workspace';
import { runInTerminal } from '../utils/terminal';
import { discoverChapters } from '../utils/chapters';

export function registerDebugCommands(context: vscode.ExtensionContext): void {
  const root = getRepoRoot();
  if (!root) { return; }

  // Debug full volume
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.debugVolumePdf', (vol: VolumeId) => {
      runInTerminal(`./book/binder debug pdf --${vol}`, root);
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

      runInTerminal(`./book/binder debug ${fmt} --${volPick.id} --chapter ${chapterPick.id}`, root);
    })
  );
}
