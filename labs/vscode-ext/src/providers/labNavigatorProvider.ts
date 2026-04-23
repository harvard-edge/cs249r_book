import * as vscode from 'vscode';
import { discoverAllLabs } from '../utils/labDiscovery';
import { LabTreeItem, CategoryTreeItem } from '../models/treeItems';

type TreeNode = CategoryTreeItem | LabTreeItem;

/** Labs Navigator — browse Vol1/Vol2 labs with status */
export class LabNavigatorProvider implements vscode.TreeDataProvider<TreeNode> {
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
    if (!element) {
      const labs = discoverAllLabs(this.projectRoot);
      const v1Impl = labs.vol1.filter(l => l.status === 'implemented').length;
      const v2Impl = labs.vol2.filter(l => l.status === 'implemented').length;

      return [
        new CategoryTreeItem(
          `Volume 1 (${v1Impl}/${labs.vol1.length} implemented)`,
          'vol1',
          'book',
        ),
        new CategoryTreeItem(
          `Volume 2 (${v2Impl}/${labs.vol2.length} implemented)`,
          'vol2',
          'book',
        ),
      ];
    }

    if (element instanceof CategoryTreeItem) {
      const labs = discoverAllLabs(this.projectRoot);
      switch (element.categoryId) {
        case 'vol1':
          return labs.vol1.map(l => new LabTreeItem(l));
        case 'vol2':
          return labs.vol2.map(l => new LabTreeItem(l));
      }
    }

    return [];
  }
}
