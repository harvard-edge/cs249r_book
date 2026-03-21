import * as vscode from 'vscode';
import { getProjectRoot } from './utils/workspace';
import { initializeRunManager, runInTerminal, rerunLastCommand, rerunCommand, revealTerminal, log } from './utils/terminal';
import { PlatformTreeProvider } from './providers/platformTreeProvider';
import { BuildTreeProvider } from './providers/buildTreeProvider';
import { RunHistoryProvider } from './providers/runHistoryProvider';
import { InfoTreeProvider } from './providers/infoTreeProvider';
import { registerBuildCommands } from './commands/buildCommands';
import { CommandRunRecord } from './types';

export function activate(context: vscode.ExtensionContext): void {
  const root = getProjectRoot();
  if (!root) {
    vscode.window.showWarningMessage(
      'Kits Workbench: could not find project root (kits/Makefile not found).',
    );
    return;
  }

  log(`Activated with project root: ${root}`);

  initializeRunManager(context);

  // --- Tree view providers ---
  const platformProvider = new PlatformTreeProvider(root);
  const buildProvider = new BuildTreeProvider(root);
  const runHistoryProvider = new RunHistoryProvider();
  const infoProvider = new InfoTreeProvider(root);

  context.subscriptions.push(
    vscode.window.createTreeView('kits.platforms', { treeDataProvider: platformProvider, showCollapseAll: true }),
    vscode.window.createTreeView('kits.build', { treeDataProvider: buildProvider }),
    vscode.window.createTreeView('kits.runs', { treeDataProvider: runHistoryProvider }),
    vscode.window.createTreeView('kits.info', { treeDataProvider: infoProvider }),
    runHistoryProvider,
  );

  // --- Generic action command ---
  context.subscriptions.push(
    vscode.commands.registerCommand('kits.runAction', (command: string, label: string) => {
      runInTerminal(command, root, label);
    }),
  );

  // --- Refresh commands ---
  context.subscriptions.push(
    vscode.commands.registerCommand('kits.refreshPlatforms', () => platformProvider.refresh()),
    vscode.commands.registerCommand('kits.refreshBuild', () => buildProvider.refresh()),
    vscode.commands.registerCommand('kits.refreshRuns', () => runHistoryProvider.refresh()),
  );

  // --- Terminal commands ---
  context.subscriptions.push(
    vscode.commands.registerCommand('kits.rerunLastCommand', () => rerunLastCommand()),
    vscode.commands.registerCommand('kits.rerunCommand', (record: CommandRunRecord) => rerunCommand(record)),
    vscode.commands.registerCommand('kits.revealTerminal', () => revealTerminal(root)),
  );

  // --- Register build commands ---
  registerBuildCommands(context, root);

  // --- Watch for content changes ---
  const contentWatcher = vscode.workspace.createFileSystemWatcher(
    new vscode.RelativePattern(root, 'kits/contents/**/*.qmd'),
  );
  contentWatcher.onDidCreate(() => platformProvider.refresh());
  contentWatcher.onDidDelete(() => platformProvider.refresh());
  context.subscriptions.push(contentWatcher);

  log('Extension fully activated.');
}

export function deactivate(): void {
  // nothing to clean up
}
