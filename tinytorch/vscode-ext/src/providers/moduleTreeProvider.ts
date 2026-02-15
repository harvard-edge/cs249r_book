import * as vscode from 'vscode';
import { discoverModules } from '../utils/modules';
import { ModuleInfo } from '../types';
import { ModuleTreeItem, ActionTreeItem } from '../models/treeItems';

type TreeNode = ModuleTreeItem | ActionTreeItem;

export class ModuleTreeProvider implements vscode.TreeDataProvider<TreeNode> {
  private _onDidChange = new vscode.EventEmitter<TreeNode | undefined>();
  readonly onDidChangeTreeData = this._onDidChange.event;

  private modules: ModuleInfo[] = [];

  constructor(private projectRoot: string) {
    this.refresh();
  }

  refresh(): void {
    this.modules = discoverModules(this.projectRoot);
    this._onDidChange.fire(undefined);
  }

  getTreeItem(element: TreeNode): vscode.TreeItem {
    return element;
  }

  getChildren(element?: TreeNode): TreeNode[] {
    // Top level: list of modules
    if (!element) {
      return this.modules.map(m => new ModuleTreeItem(m));
    }

    // Under each module: action buttons
    if (element instanceof ModuleTreeItem) {
      const m = element.module;
      const actions: ActionTreeItem[] = [];

      // Open Notebook is always the top action — this is the primary dev workflow
      actions.push(new ActionTreeItem('Open Notebook', 'tinytorch.openNotebook', [m.number, m.folder], 'notebook'));

      if (m.status === 'not_started') {
        actions.push(new ActionTreeItem('Start Module', 'tinytorch.startModule', [m.number], 'play'));
      }

      actions.push(new ActionTreeItem('Test Module', 'tinytorch.testModule', [m.number], 'beaker'));

      if (m.status === 'started') {
        actions.push(new ActionTreeItem('Complete Module', 'tinytorch.completeModule', [m.number], 'check'));
      }

      actions.push(new ActionTreeItem('Open Source (.py)', 'tinytorch.openSource', [m.number, m.folder], 'file-code'));
      actions.push(new ActionTreeItem('Open ABOUT', 'tinytorch.openAbout', [m.number, m.folder], 'book'));
      actions.push(new ActionTreeItem('Reset Module', 'tinytorch.resetModule', [m.number], 'discard'));

      return actions;
    }

    return [];
  }
}
