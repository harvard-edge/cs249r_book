import * as vscode from 'vscode';
import { VolumeId, BuildFormat } from '../types';
import { getRepoRoot } from '../utils/workspace';
import { runBookCommand } from '../utils/terminal';

export function registerBuildCommands(context: vscode.ExtensionContext): void {
  const root = getRepoRoot();
  if (!root) { return; }

  // Chapter-level builds
  for (const fmt of ['Html', 'Pdf', 'Epub'] as const) {
    const fmtLower = fmt.toLowerCase() as BuildFormat;
    context.subscriptions.push(
      vscode.commands.registerCommand(`mlsysbook.buildChapter${fmt}`, (vol: VolumeId, chapter: string) => {
        void runBookCommand(`./book/binder build ${fmtLower} ${chapter} --${vol} -v`, root, {
          label: `Build Chapter ${fmt.toUpperCase()} (${vol}/${chapter})`,
        });
      })
    );
  }

  // Preview
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.previewChapter', (vol: VolumeId, chapter: string) => {
      void runBookCommand(`./book/binder preview ${chapter}`, root, {
        label: `Preview Chapter (${vol}/${chapter})`,
      });
    })
  );

  // Volume-level builds
  for (const fmt of ['Html', 'Pdf', 'Epub'] as const) {
    const fmtLower = fmt.toLowerCase() as BuildFormat;
    context.subscriptions.push(
      vscode.commands.registerCommand(`mlsysbook.buildVolume${fmt}`, (vol: VolumeId) => {
        void runBookCommand(`./book/binder build ${fmtLower} --${vol} -v`, root, {
          label: `Build Volume ${fmt.toUpperCase()} (${vol})`,
        });
      })
    );
  }
}
