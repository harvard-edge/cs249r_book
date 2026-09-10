import * as vscode from 'vscode';
import { getProjectRoot } from './utils/workspace';
import { initializeRunManager, runInTerminal, rerunLastCommand, rerunCommand, revealTerminal, log } from './utils/terminal';
import { ZooTreeProvider } from './providers/zooTreeProvider';
import { ScenarioTreeProvider } from './providers/scenarioTreeProvider';
import { TestTreeProvider } from './providers/testTreeProvider';
import { RunHistoryProvider } from './providers/runHistoryProvider';
import { InfoTreeProvider } from './providers/infoTreeProvider';
import { registerEvalCommands } from './commands/evalCommands';
import { registerTestCommands } from './commands/testCommands';
import { registerZooCommands } from './commands/zooCommands';
import { CommandRunRecord } from './types';

export function activate(context: vscode.ExtensionContext): void {
  const root = getProjectRoot();
  if (!root) {
    vscode.window.showWarningMessage(
      'MLSysim Workbench: could not find project root (mlsysim/__init__.py not found).',
    );
    return;
  }

  log(`Activated with project root: ${root}`);

  initializeRunManager(context);

  // --- Tree view providers ---
  const zooProvider = new ZooTreeProvider(root);
  const scenarioProvider = new ScenarioTreeProvider(root);
  const testProvider = new TestTreeProvider(root);
  const runHistoryProvider = new RunHistoryProvider();
  const infoProvider = new InfoTreeProvider(root);

  context.subscriptions.push(
    vscode.window.createTreeView('mlsysim.zoo', { treeDataProvider: zooProvider, showCollapseAll: true }),
    vscode.window.createTreeView('mlsysim.scenarios', { treeDataProvider: scenarioProvider }),
    vscode.window.createTreeView('mlsysim.tests', { treeDataProvider: testProvider }),
    vscode.window.createTreeView('mlsysim.runs', { treeDataProvider: runHistoryProvider }),
    vscode.window.createTreeView('mlsysim.info', { treeDataProvider: infoProvider }),
    runHistoryProvider,
  );

  // --- Generic action command ---
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysim.runAction', (command: string, label: string) => {
      runInTerminal(command, root, label);
    }),
  );

  // --- Refresh commands ---
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysim.refreshZoo', () => zooProvider.refresh()),
    vscode.commands.registerCommand('mlsysim.refreshScenarios', () => scenarioProvider.refresh()),
    vscode.commands.registerCommand('mlsysim.refreshTests', () => testProvider.refresh()),
    vscode.commands.registerCommand('mlsysim.refreshRuns', () => runHistoryProvider.refresh()),
  );

  // --- Terminal commands ---
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysim.rerunLastCommand', () => rerunLastCommand()),
    vscode.commands.registerCommand('mlsysim.rerunCommand', (record: CommandRunRecord) => rerunCommand(record)),
    vscode.commands.registerCommand('mlsysim.revealTerminal', () => revealTerminal(root)),
  );

  // --- Register command groups ---
  registerEvalCommands(context, root);
  registerTestCommands(context, root);
  registerZooCommands(context, root);

  // --- Watch for scenario file changes ---
  const scenarioWatcher = vscode.workspace.createFileSystemWatcher(
    new vscode.RelativePattern(root, 'mlsysim/examples/yaml/*.{yaml,yml}'),
  );
  scenarioWatcher.onDidChange(() => scenarioProvider.refresh());
  scenarioWatcher.onDidCreate(() => scenarioProvider.refresh());
  scenarioWatcher.onDidDelete(() => scenarioProvider.refresh());
  context.subscriptions.push(scenarioWatcher);

  log('Extension fully activated.');
}

export function deactivate(): void {
  // nothing to clean up
}
