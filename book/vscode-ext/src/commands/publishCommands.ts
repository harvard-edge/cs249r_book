import * as vscode from 'vscode';
import { getRepoRoot } from '../utils/workspace';
import { runInTerminal } from '../utils/terminal';

export function registerPublishCommands(context: vscode.ExtensionContext): void {
  const root = getRepoRoot();
  if (!root) { return; }

  // Generic action runner (used by tree items that pass a command string)
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.runAction', (command: string) => {
      runInTerminal(command, root);
    })
  );

  // Named aliases for command palette discoverability
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.cleanArtifacts', () => {
      runInTerminal('./book/binder clean', root);
    }),
    vscode.commands.registerCommand('mlsysbook.doctor', () => {
      runInTerminal('./book/binder doctor', root);
    }),
    vscode.commands.registerCommand('mlsysbook.buildGlossary', () => {
      runInTerminal('python3 book/tools/scripts/glossary/build_global_glossary.py', root);
    }),
    vscode.commands.registerCommand('mlsysbook.compressImages', () => {
      runInTerminal('python3 book/tools/scripts/images/compress_images.py', root);
    }),
    vscode.commands.registerCommand('mlsysbook.repoHealth', () => {
      runInTerminal('python3 book/tools/scripts/maintenance/repo_health_check.py --health-check', root);
    }),
  );
}
