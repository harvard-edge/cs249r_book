import * as vscode from 'vscode';
import { PUBLISH_ACTIONS, MAINTENANCE_ACTIONS } from '../constants';
import { ActionTreeItem } from '../models/treeItems';

type TreeNode = ActionTreeItem | SeparatorItem;

class SeparatorItem extends vscode.TreeItem {
  constructor(label: string) {
    super(label, vscode.TreeItemCollapsibleState.None);
    this.description = '';
    this.contextValue = 'separator';
  }
}

export class PublishTreeProvider implements vscode.TreeDataProvider<TreeNode> {
  private _onDidChangeTreeData = new vscode.EventEmitter<TreeNode | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  getTreeItem(element: TreeNode): vscode.TreeItem {
    return element;
  }

  getChildren(): TreeNode[] {
    const publishItems = PUBLISH_ACTIONS.map(a =>
      new ActionTreeItem(a.label, 'mlsysbook.runAction', [a.command], a.icon ?? 'rocket')
    );

    const maintenanceItems = MAINTENANCE_ACTIONS.map(a =>
      new ActionTreeItem(a.label, 'mlsysbook.runAction', [a.command], a.icon ?? 'tools')
    );

    return [
      ...publishItems,
      new SeparatorItem('--- Maintenance ---'),
      ...maintenanceItems,
    ];
  }
}
