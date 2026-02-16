import * as vscode from 'vscode';
import { runInTerminal } from '../utils/terminal';

/** Register test-related command palette entries */
export function registerTestCommands(
  context: vscode.ExtensionContext,
  projectRoot: string,
): void {
  // Quick test from command palette (unit + CLI, mirrors CI stages 2+4)
  context.subscriptions.push(
    vscode.commands.registerCommand('tinytorch.runTests', () => {
      runInTerminal('python3 -m tito.main dev test --unit --cli', projectRoot, 'Quick Tests');
    }),
  );
}
