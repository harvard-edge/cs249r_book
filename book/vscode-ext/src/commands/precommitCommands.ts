import * as vscode from 'vscode';
import { getRepoRoot } from '../utils/workspace';
import { runInTerminal } from '../utils/terminal';

export function registerPrecommitCommands(context: vscode.ExtensionContext): void {
  const root = getRepoRoot();
  if (!root) { return; }

  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.precommitRunAll', () => {
      runInTerminal('pre-commit run --all-files', root);
    }),

    vscode.commands.registerCommand('mlsysbook.precommitRunHook', (command: string) => {
      runInTerminal(command, root);
    }),
  );
}
