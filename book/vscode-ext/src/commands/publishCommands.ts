import * as vscode from 'vscode';
import { getRepoRoot } from '../utils/workspace';
import { runBookCommand } from '../utils/terminal';

export function registerPublishCommands(context: vscode.ExtensionContext): void {
  const root = getRepoRoot();
  if (!root) { return; }

  // Generic action runner (used by tree items that pass a command string)
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.runAction', (command: string) => {
      void runBookCommand(command, root, {
        label: 'Maintenance action',
      });
    })
  );

  // Named aliases for command palette discoverability
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.cleanArtifacts', () => {
      void runBookCommand('./book/binder clean', root, {
        label: 'Clean build artifacts',
      });
    }),
    vscode.commands.registerCommand('mlsysbook.doctor', () => {
      void runBookCommand('./book/binder doctor', root, {
        label: 'Doctor (health check)',
      });
    }),
    vscode.commands.registerCommand('mlsysbook.buildGlossary', () => {
      void runBookCommand('python3 book/tools/scripts/glossary/build_global_glossary.py', root, {
        label: 'Build global glossary',
      });
    }),
    vscode.commands.registerCommand('mlsysbook.compressImages', () => {
      void runBookCommand('python3 book/tools/scripts/images/compress_images.py', root, {
        label: 'Compress images',
      });
    }),
    vscode.commands.registerCommand('mlsysbook.repoHealth', () => {
      void runBookCommand('python3 book/tools/scripts/maintenance/repo_health_check.py --health-check', root, {
        label: 'Repo health check',
      });
    }),
  );
}
