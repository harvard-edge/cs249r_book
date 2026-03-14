import * as vscode from 'vscode';
import * as path from 'path';
import { runInTerminal } from '../utils/terminal';

export function registerBuildCommands(context: vscode.ExtensionContext, root: string): void {
  const kitsDir = path.join(root, 'kits');

  context.subscriptions.push(
    vscode.commands.registerCommand('kits.buildHtml', () => {
      runInTerminal(`cd "${kitsDir}" && make html`, root, 'Build HTML');
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kits.buildPdf', () => {
      runInTerminal(`cd "${kitsDir}" && make pdf`, root, 'Build PDF');
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kits.preview', () => {
      runInTerminal(`cd "${kitsDir}" && make preview`, root, 'Preview');
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kits.clean', () => {
      runInTerminal(`cd "${kitsDir}" && make clean`, root, 'Clean');
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kits.healthCheck', () => {
      runInTerminal('quarto --version && make --version', root, 'Health Check');
    }),
  );
}
