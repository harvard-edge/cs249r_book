import * as vscode from 'vscode';
import { VolumeId, BuildFormat } from '../types';
import { getRepoRoot } from '../utils/workspace';
import { runInTerminal } from '../utils/terminal';

export function registerBuildCommands(context: vscode.ExtensionContext): void {
  const root = getRepoRoot();
  if (!root) { return; }

  // Chapter-level builds
  for (const fmt of ['Html', 'Pdf', 'Epub'] as const) {
    const fmtLower = fmt.toLowerCase() as BuildFormat;
    context.subscriptions.push(
      vscode.commands.registerCommand(`mlsysbook.buildChapter${fmt}`, (vol: VolumeId, chapter: string) => {
        runInTerminal(`./book/binder ${fmtLower} ${chapter} --${vol} -v`, root);
      })
    );
  }

  // Preview
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.previewChapter', (vol: VolumeId, chapter: string) => {
      runInTerminal(`./book/binder preview ${chapter}`, root);
    })
  );

  // Volume-level builds
  for (const fmt of ['Html', 'Pdf', 'Epub'] as const) {
    const fmtLower = fmt.toLowerCase() as BuildFormat;
    context.subscriptions.push(
      vscode.commands.registerCommand(`mlsysbook.buildVolume${fmt}`, (vol: VolumeId) => {
        runInTerminal(`./book/binder ${fmtLower} --${vol} -v`, root);
      })
    );
  }
}
