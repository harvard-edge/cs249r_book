import * as vscode from 'vscode';
import { getRepoRoot, parseQmdFile } from '../utils/workspace';
import { runBookCommand } from '../utils/terminal';
import { runIsolatedDebugCommand } from '../utils/parallelDebug';
import { withQuartoResetPrefix } from '../utils/quartoConfigReset';
import type { BuildFormat } from '../types';

export function registerContextMenuCommands(context: vscode.ExtensionContext): void {
  const root = getRepoRoot();
  if (!root) { return; }

  const makeHandler = (format: BuildFormat) => {
    return (uri: vscode.Uri) => {
      const ctx = parseQmdFile(uri);
      if (!ctx) {
        vscode.window.showWarningMessage('Could not determine volume/chapter from file path.');
        return;
      }
      const buildCmd = `./book/binder build ${format} ${ctx.chapter} --${ctx.volume} -v`;
      void runBookCommand(withQuartoResetPrefix(format, ctx.volume, buildCmd), root, {
        label: `Build ${format.toUpperCase()} (${ctx.volume}/${ctx.chapter})`,
      });
    };
  };

  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.contextBuildHtml', makeHandler('html')),
    vscode.commands.registerCommand('mlsysbook.contextBuildPdf', makeHandler('pdf')),
    vscode.commands.registerCommand('mlsysbook.contextBuildEpub', makeHandler('epub')),

    vscode.commands.registerCommand('mlsysbook.contextPreview', (uri: vscode.Uri) => {
      const ctx = parseQmdFile(uri);
      if (!ctx) {
        vscode.window.showWarningMessage('Could not determine volume/chapter from file path.');
        return;
      }
      void runBookCommand(`./book/binder preview ${ctx.volume}/${ctx.chapter}`, root, {
        label: `Preview (${ctx.volume}/${ctx.chapter})`,
      });
    }),

    vscode.commands.registerCommand('mlsysbook.contextDebugSections', (uri: vscode.Uri) => {
      const ctx = parseQmdFile(uri);
      if (!ctx) {
        vscode.window.showWarningMessage('Could not determine volume/chapter from file path.');
        return;
      }
      void runIsolatedDebugCommand({
        repoRoot: root,
        command: `./book/binder debug pdf --${ctx.volume} --chapter ${ctx.chapter}`,
        label: `Debug Sections (${ctx.volume}/${ctx.chapter})`,
      });
    }),
  );
}
