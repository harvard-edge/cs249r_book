import * as vscode from 'vscode';
import { runInTerminal } from '../utils/terminal';
import { mlsysimCommand } from '../utils/mlsysimCli';

export function registerZooCommands(context: vscode.ExtensionContext, root: string): void {
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysim.zooHardware', () => {
      runInTerminal(mlsysimCommand('zoo hardware'), root, 'Zoo: Hardware');
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysim.zooModels', () => {
      runInTerminal(mlsysimCommand('zoo models'), root, 'Zoo: Models');
    }),
  );
}
