import * as vscode from 'vscode';
import { getProjectRoot } from './utils/workspace';
import { initializeRunManager, runInTerminal, rerunLastCommand, rerunCommand, revealTerminal } from './utils/terminal';
import { isTitoAvailable, log, titoTerminalCommand } from './utils/tito';
import { ModuleTreeProvider } from './providers/moduleTreeProvider';
import { TestTreeProvider } from './providers/testTreeProvider';
import { BuildTreeProvider } from './providers/buildTreeProvider';
import { RunHistoryProvider } from './providers/runHistoryProvider';
import { InfoTreeProvider } from './providers/infoTreeProvider';
import { registerModuleCommands } from './commands/moduleCommands';
import { registerTestCommands } from './commands/testCommands';
import { registerBuildCommands } from './commands/buildCommands';
import { CommandRunRecord } from './types';

export function activate(context: vscode.ExtensionContext): void {
  const root = getProjectRoot();
  if (!root) {
    vscode.window.showWarningMessage(
      'TinyTorch Workbench: could not find project root (src/01_tensor not found).',
    );
    return;
  }

  log(`Activated with project root: ${root}`);

  // Pre-flight: check that Tito CLI is reachable
  if (!isTitoAvailable(root)) {
    void vscode.window.showWarningMessage(
      'TinyTorch: Tito CLI is not available. Run "pip install -e ." in the tinytorch directory.',
      'Show Setup Instructions',
    ).then(choice => {
      if (choice === 'Show Setup Instructions') {
        runInTerminal(titoTerminalCommand('setup'), root, 'Setup');
      }
    });
    log('WARNING: Tito CLI not available — extension will have limited functionality.');
  }

  initializeRunManager(context);

  // --- Tree view providers ---
  const moduleProvider = new ModuleTreeProvider(root);
  const testProvider = new TestTreeProvider(root);
  const buildProvider = new BuildTreeProvider();
  const runHistoryProvider = new RunHistoryProvider();
  const infoProvider = new InfoTreeProvider(root);

  context.subscriptions.push(
    vscode.window.createTreeView('tinytorch.modules', { treeDataProvider: moduleProvider }),
    vscode.window.createTreeView('tinytorch.testing', { treeDataProvider: testProvider }),
    vscode.window.createTreeView('tinytorch.build', { treeDataProvider: buildProvider }),
    vscode.window.createTreeView('tinytorch.runs', { treeDataProvider: runHistoryProvider }),
    vscode.window.createTreeView('tinytorch.info', { treeDataProvider: infoProvider }),
    runHistoryProvider,
  );

  // --- Generic action command (used by tree items) ---
  context.subscriptions.push(
    vscode.commands.registerCommand('tinytorch.runAction', (command: string, label: string) => {
      runInTerminal(command, root, label);
    }),
  );

  // --- Refresh commands ---
  context.subscriptions.push(
    vscode.commands.registerCommand('tinytorch.refreshModules', () => moduleProvider.refresh()),
    vscode.commands.registerCommand('tinytorch.refreshRuns', () => runHistoryProvider.refresh()),
  );

  // --- Terminal commands ---
  context.subscriptions.push(
    vscode.commands.registerCommand('tinytorch.rerunLastCommand', () => rerunLastCommand()),
    vscode.commands.registerCommand('tinytorch.rerunCommand', (record: CommandRunRecord) => rerunCommand(record)),
    vscode.commands.registerCommand('tinytorch.revealTerminal', () => revealTerminal(root)),
  );

  // --- Register command groups ---
  registerModuleCommands(context, root);
  registerTestCommands(context, root);
  registerBuildCommands(context, root);

  // --- Watch .tito/progress.json for changes (where Tito stores module status) ---
  // This is the SOLE mechanism for refreshing the module tree after start/complete/reset.
  // No setTimeout hacks — the file watcher fires whenever Tito writes progress.
  const progressWatcher = vscode.workspace.createFileSystemWatcher(
    new vscode.RelativePattern(root, '.tito/progress.json'),
  );
  progressWatcher.onDidChange(() => moduleProvider.refresh());
  progressWatcher.onDidCreate(() => moduleProvider.refresh());
  context.subscriptions.push(progressWatcher);

  log('Extension fully activated.');
}

export function deactivate(): void {
  // nothing to clean up
}
