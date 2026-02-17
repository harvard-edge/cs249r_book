import * as vscode from 'vscode';
import { discoverModules } from '../utils/modules';
import { titoTerminalCommand } from '../utils/tito';
import { ModuleInfo } from '../types';
import { ModuleTreeItem, ActionTreeItem, InfoTreeItem } from '../models/treeItems';

type TreeNode = ModuleTreeItem | ActionTreeItem | InfoTreeItem;

export class ModuleTreeProvider implements vscode.TreeDataProvider<TreeNode> {
  private _onDidChange = new vscode.EventEmitter<TreeNode | undefined>();
  readonly onDidChangeTreeData = this._onDidChange.event;

  private modules: ModuleInfo[] = [];
  private lastError: string | undefined;

  constructor(private projectRoot: string) {
    this.refresh();
  }

  refresh(): void {
    const result = discoverModules(this.projectRoot);
    this.modules = result.modules;
    this.lastError = result.error;
    this._onDidChange.fire(undefined);
  }

  getTreeItem(element: TreeNode): vscode.TreeItem {
    return element;
  }

  getChildren(element?: TreeNode): TreeNode[] {
    // Top level: list of modules (or error/empty state)
    if (!element) {
      if (this.lastError) {
        return [
          new InfoTreeItem('Could not load modules', undefined, 'warning'),
          new ActionTreeItem('Run Setup', 'tinytorch.runAction',
            [titoTerminalCommand('setup'), 'Setup'], 'tools'),
          new ActionTreeItem('Run Health Check', 'tinytorch.runAction',
            [titoTerminalCommand('system health'), 'Health Check'], 'heart'),
        ];
      }
      if (this.modules.length === 0) {
        return [
          new InfoTreeItem('No modules found', 'run tito setup', 'info'),
          new ActionTreeItem('Run Setup', 'tinytorch.runAction',
            [titoTerminalCommand('setup'), 'Setup'], 'tools'),
        ];
      }
      return this.modules.map(m => new ModuleTreeItem(m));
    }

    // Under each module: action buttons
    if (element instanceof ModuleTreeItem) {
      const m = element.module;
      const actions: ActionTreeItem[] = [];

      actions.push(new ActionTreeItem('Open Notebook', 'tinytorch.openNotebook', [m.number], 'notebook'));

      if (m.status === 'not_started') {
        actions.push(new ActionTreeItem('Start Module', 'tinytorch.startModule', [m.number], 'play'));
      }

      actions.push(new ActionTreeItem('Test Module', 'tinytorch.testModule', [m.number], 'beaker'));

      if (m.status === 'started') {
        actions.push(new ActionTreeItem('Complete Module', 'tinytorch.completeModule', [m.number], 'check'));
      }

      actions.push(new ActionTreeItem('Open Source (.py)', 'tinytorch.openSource', [m.number], 'file-code'));
      actions.push(new ActionTreeItem('Open ABOUT', 'tinytorch.openAbout', [m.number], 'book'));
      actions.push(new ActionTreeItem('Reset Module', 'tinytorch.resetModule', [m.number], 'discard'));

      return actions;
    }

    return [];
  }
}
