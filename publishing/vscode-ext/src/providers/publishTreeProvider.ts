import * as vscode from 'vscode';
import { PUBLISH_ACTIONS } from '../constants';
import { ActionTreeItem } from '../models/treeItems';

type TreeNode = ActionTreeItem;

export class PublishTreeProvider implements vscode.TreeDataProvider<TreeNode> {
  private _onDidChangeTreeData = new vscode.EventEmitter<TreeNode | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  getTreeItem(element: TreeNode): vscode.TreeItem {
    return element;
  }

  getChildren(): TreeNode[] {
    return PUBLISH_ACTIONS.map(a =>
      new ActionTreeItem(a.label, 'mlsysbook.runAction', [a.command], a.icon ?? 'rocket')
    );
  }
}
