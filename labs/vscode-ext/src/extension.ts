import * as vscode from 'vscode';
import { getProjectRoot } from './utils/workspace';
import { initializeRunManager, runInTerminal, rerunLastCommand, rerunCommand, revealTerminal, log } from './utils/terminal';
import { LabNavigatorProvider } from './providers/labNavigatorProvider';
import { GovernanceProvider } from './providers/governanceProvider';
import { RunHistoryProvider } from './providers/runHistoryProvider';
import { InfoTreeProvider } from './providers/infoTreeProvider';
import { MarimoSymbolProvider } from './providers/marimoSymbolProvider';
import { registerLabCommands } from './commands/labCommands';
import { registerAuditCommands } from './commands/auditCommands';
import { registerLedgerCommands } from './commands/ledgerCommands';
import { CommandRunRecord } from './types';

export function activate(context: vscode.ExtensionContext): void {
  const root = getProjectRoot();
  if (!root) {
    vscode.window.showWarningMessage(
      'Labs Workbench: could not find project root (labs/PROTOCOL.md not found).',
    );
    return;
  }

  log(`Activated with project root: ${root}`);

  initializeRunManager(context);

  // --- Tree view providers ---
  const labNavigator = new LabNavigatorProvider(root);
  const governanceProvider = new GovernanceProvider(root);
  const runHistoryProvider = new RunHistoryProvider();
  const infoProvider = new InfoTreeProvider(root);

  context.subscriptions.push(
    vscode.window.createTreeView('labs.navigator', { treeDataProvider: labNavigator, showCollapseAll: true }),
    vscode.window.createTreeView('labs.governance', { treeDataProvider: governanceProvider }),
    vscode.window.createTreeView('labs.runs', { treeDataProvider: runHistoryProvider }),
    vscode.window.createTreeView('labs.info', { treeDataProvider: infoProvider }),
    runHistoryProvider,
  );

  // --- Marimo symbol provider for lab files ---
  context.subscriptions.push(
    vscode.languages.registerDocumentSymbolProvider(
      { language: 'python', pattern: '**/labs/vol*/*.py' },
      new MarimoSymbolProvider(),
    ),
  );

  // --- Generic action command ---
  context.subscriptions.push(
    vscode.commands.registerCommand('labs.runAction', (command: string, label: string) => {
      runInTerminal(command, root, label);
    }),
  );

  // --- Refresh commands ---
  context.subscriptions.push(
    vscode.commands.registerCommand('labs.refreshLabs', () => labNavigator.refresh()),
    vscode.commands.registerCommand('labs.refreshGovernance', () => governanceProvider.refresh()),
    vscode.commands.registerCommand('labs.refreshRuns', () => runHistoryProvider.refresh()),
  );

  // --- Terminal commands ---
  context.subscriptions.push(
    vscode.commands.registerCommand('labs.rerunLastCommand', () => rerunLastCommand()),
    vscode.commands.registerCommand('labs.rerunCommand', (record: CommandRunRecord) => rerunCommand(record)),
    vscode.commands.registerCommand('labs.revealTerminal', () => revealTerminal(root)),
  );

  // --- Register command groups ---
  registerLabCommands(context, root);
  registerAuditCommands(context, root);
  registerLedgerCommands(context);

  // --- Watch for lab file changes ---
  const labWatcher = vscode.workspace.createFileSystemWatcher(
    new vscode.RelativePattern(root, 'labs/vol{1,2}/*.py'),
  );
  labWatcher.onDidChange(() => labNavigator.refresh());
  labWatcher.onDidCreate(() => labNavigator.refresh());
  labWatcher.onDidDelete(() => labNavigator.refresh());
  context.subscriptions.push(labWatcher);

  // --- Watch for plan file changes ---
  const planWatcher = vscode.workspace.createFileSystemWatcher(
    new vscode.RelativePattern(root, 'labs/plans/vol{1,2}/*.md'),
  );
  planWatcher.onDidChange(() => labNavigator.refresh());
  planWatcher.onDidCreate(() => labNavigator.refresh());
  planWatcher.onDidDelete(() => labNavigator.refresh());
  context.subscriptions.push(planWatcher);

  log('Extension fully activated.');
}

export function deactivate(): void {
  // nothing to clean up
}
