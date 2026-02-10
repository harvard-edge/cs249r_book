import * as vscode from 'vscode';
import { getRepoRoot } from '../utils/workspace';
import { runBookCommand } from '../utils/terminal';

export function registerPrecommitCommands(context: vscode.ExtensionContext): void {
  const root = getRepoRoot();
  if (!root) { return; }

  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.precommitRunAll', () => {
      void runBookCommand('pre-commit run --all-files', root, {
        label: 'Pre-commit (all hooks)',
      });
    }),

    vscode.commands.registerCommand('mlsysbook.precommitRunHook', (command: string) => {
      void runBookCommand(command, root, {
        label: 'Pre-commit (selected hook)',
      });
    }),
  );
}
