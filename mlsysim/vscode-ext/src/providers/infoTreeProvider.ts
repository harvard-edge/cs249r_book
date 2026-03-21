import * as vscode from 'vscode';
import { isMlsysimAvailable, getPythonVersion, mlsysimCommand } from '../utils/mlsysimCli';
import { InfoTreeItem, ActionTreeItem } from '../models/treeItems';

type TreeNode = InfoTreeItem | ActionTreeItem;

export class InfoTreeProvider implements vscode.TreeDataProvider<TreeNode> {
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

    const items: TreeNode[] = [];
    const available = isMlsysimAvailable(this.projectRoot);
    const pyVersion = getPythonVersion(this.projectRoot);

    if (!available) {
      items.push(new InfoTreeItem('MLSysim CLI', 'unavailable', 'error'));
      items.push(new InfoTreeItem('Python', pyVersion, 'symbol-misc'));
    } else {
      items.push(new InfoTreeItem('MLSysim CLI', 'available', 'pass'));
      items.push(new InfoTreeItem('Python', pyVersion, 'symbol-misc'));
    }

    items.push(new InfoTreeItem(''));

    // Actions
    items.push(new ActionTreeItem('Run Health Check', 'mlsysim.runAction',
      [mlsysimCommand('zoo hardware'), 'Health Check'], 'heart'));
    items.push(new ActionTreeItem('Export Schema', 'mlsysim.runAction',
      [mlsysimCommand('schema'), 'Export Schema'], 'json'));
    items.push(new ActionTreeItem('Run All Tests', 'mlsysim.runAllTests', [], 'checklist'));

    return items;
  }
}
