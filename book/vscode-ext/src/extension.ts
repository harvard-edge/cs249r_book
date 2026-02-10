import * as vscode from 'vscode';
import { getRepoRoot } from './utils/workspace';
import { BuildTreeProvider } from './providers/buildTreeProvider';
import { DebugTreeProvider } from './providers/debugTreeProvider';
import { PrecommitTreeProvider } from './providers/precommitTreeProvider';
import { PublishTreeProvider } from './providers/publishTreeProvider';
import { registerBuildCommands } from './commands/buildCommands';
import { registerDebugCommands } from './commands/debugCommands';
import { registerPrecommitCommands } from './commands/precommitCommands';
import { registerPublishCommands } from './commands/publishCommands';
import { registerContextMenuCommands } from './commands/contextMenuCommands';
import {
  initializeRunManager,
  rerunLastCommand,
  revealRunTerminal,
  showLastFailureDetails,
  setExecutionModeInteractively,
} from './utils/terminal';

export function activate(context: vscode.ExtensionContext): void {
  const root = getRepoRoot();
  if (!root) {
    vscode.window.showWarningMessage('MLSysBook Workbench: could not find repo root (book/binder not found).');
    return;
  }
  initializeRunManager(context);

  // Tree view providers
  const buildProvider = new BuildTreeProvider(root);
  const debugProvider = new DebugTreeProvider();
  const precommitProvider = new PrecommitTreeProvider();
  const publishProvider = new PublishTreeProvider();

  context.subscriptions.push(
    vscode.window.createTreeView('mlsysbook.build', { treeDataProvider: buildProvider }),
    vscode.window.createTreeView('mlsysbook.debug', { treeDataProvider: debugProvider }),
    vscode.window.createTreeView('mlsysbook.precommit', { treeDataProvider: precommitProvider }),
    vscode.window.createTreeView('mlsysbook.publish', { treeDataProvider: publishProvider }),
  );

  // Refresh command for build tree
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.refreshBuildTree', () => buildProvider.refresh()),
    vscode.commands.registerCommand('mlsysbook.rerunLastCommand', () => rerunLastCommand(false)),
    vscode.commands.registerCommand('mlsysbook.rerunLastCommandRaw', () => rerunLastCommand(true)),
    vscode.commands.registerCommand('mlsysbook.revealTerminal', () => revealRunTerminal(root)),
    vscode.commands.registerCommand('mlsysbook.openLastFailureDetails', () => showLastFailureDetails()),
    vscode.commands.registerCommand('mlsysbook.setExecutionMode', () => void setExecutionModeInteractively()),
  );

  // Register all command groups
  registerBuildCommands(context);
  registerDebugCommands(context);
  registerPrecommitCommands(context);
  registerPublishCommands(context);
  registerContextMenuCommands(context);
}

export function deactivate(): void {
  // nothing to clean up
}
