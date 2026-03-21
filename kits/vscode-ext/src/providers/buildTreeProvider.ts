import * as vscode from 'vscode';
import * as path from 'path';
import { CategoryTreeItem, ActionTreeItem } from '../models/treeItems';

type TreeNode = CategoryTreeItem | ActionTreeItem;

/** Build tree — Makefile targets for HTML, PDF, preview, clean */
export class BuildTreeProvider implements vscode.TreeDataProvider<TreeNode> {
  private _onDidChange = new vscode.EventEmitter<TreeNode | undefined>();
  readonly onDidChangeTreeData = this._onDidChange.event;

  constructor(private projectRoot: string) {}

  refresh(): void {
    this._onDidChange.fire(undefined);
  }

  getTreeItem(element: TreeNode): vscode.TreeItem {
    return element;
  }

  getChildren(element?: TreeNode): TreeNode[] {
    if (element) { return []; }

    const kitsDir = path.join(this.projectRoot, 'kits');

    return [
      new ActionTreeItem('Build HTML Site', 'kits.runAction',
        [`cd "${kitsDir}" && make html`, 'Build HTML'], 'globe'),
      new ActionTreeItem('Build PDF', 'kits.runAction',
        [`cd "${kitsDir}" && make pdf`, 'Build PDF'], 'file-pdf'),
      new ActionTreeItem('Preview (Live Reload)', 'kits.runAction',
        [`cd "${kitsDir}" && make preview`, 'Preview'], 'eye'),
      new ActionTreeItem('Clean Artifacts', 'kits.runAction',
        [`cd "${kitsDir}" && make clean`, 'Clean'], 'trash'),
    ];
  }
}
