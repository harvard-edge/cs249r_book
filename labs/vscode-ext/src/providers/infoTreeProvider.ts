import * as vscode from 'vscode';
import { isMarimoAvailable, getMarimoVersion, getPythonVersion } from '../utils/marimoCli';
import { ledgerExists, getLedgerSummary } from '../utils/ledger';
import { discoverAllLabs } from '../utils/labDiscovery';
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
    const marimoOk = isMarimoAvailable();
    const marimoVer = getMarimoVersion();
    const pyVer = getPythonVersion();

    // Environment
    items.push(new InfoTreeItem('Python', pyVer, 'symbol-misc'));
    items.push(new InfoTreeItem(
      'Marimo',
      marimoOk ? marimoVer : 'unavailable',
      marimoOk ? 'pass' : 'error',
    ));

    // Lab counts
    const labs = discoverAllLabs(this.projectRoot);
    const v1Impl = labs.vol1.filter(l => l.status === 'implemented').length;
    const v2Impl = labs.vol2.filter(l => l.status === 'implemented').length;
    items.push(new InfoTreeItem('Vol 1 Labs', `${v1Impl}/${labs.vol1.length} implemented`, 'book'));
    items.push(new InfoTreeItem('Vol 2 Labs', `${v2Impl}/${labs.vol2.length} implemented`, 'book'));

    // Design Ledger
    items.push(new InfoTreeItem(
      'Design Ledger',
      ledgerExists() ? getLedgerSummary() : 'not created',
      ledgerExists() ? 'database' : 'circle-outline',
    ));

    items.push(new InfoTreeItem(''));

    // Actions
    items.push(new ActionTreeItem('Show Design Ledger', 'labs.labShowLedger', [], 'database'));
    items.push(new ActionTreeItem('Health Check', 'labs.healthCheck', [], 'heart'));

    return items;
  }
}
