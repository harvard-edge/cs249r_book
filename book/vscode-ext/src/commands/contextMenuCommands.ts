import * as vscode from 'vscode';
import { getRepoRoot, parseQmdFile } from '../utils/workspace';
import { runInTerminal } from '../utils/terminal';

export function registerContextMenuCommands(context: vscode.ExtensionContext): void {
  const root = getRepoRoot();
  if (!root) { return; }

  const makeHandler = (format: string) => {
    return (uri: vscode.Uri) => {
      const ctx = parseQmdFile(uri);
      if (!ctx) {
        vscode.window.showWarningMessage('Could not determine volume/chapter from file path.');
        return;
      }
      runInTerminal(`./book/binder ${format} ${ctx.chapter} --${ctx.volume} -v`, root);
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
      runInTerminal(`./book/binder preview ${ctx.chapter}`, root);
    }),

    vscode.commands.registerCommand('mlsysbook.contextDebugSections', (uri: vscode.Uri) => {
      const ctx = parseQmdFile(uri);
      if (!ctx) {
        vscode.window.showWarningMessage('Could not determine volume/chapter from file path.');
        return;
      }
      runInTerminal(`./book/binder debug pdf --${ctx.volume} --chapter ${ctx.chapter}`, root);
    }),
  );
}
