import * as vscode from 'vscode';
import { callTitoJson, log, titoTerminalCommand } from '../utils/tito';
import { InfoTreeItem, ActionTreeItem } from '../models/treeItems';

type TreeNode = InfoTreeItem | ActionTreeItem;

/** System info returned by `tito system info --json` */
interface SystemInfo {
  python_version: string;
  tinytorch_version: string;
  numpy_version: string;
  venv_active: boolean;
  platform: string;
}

const UNKNOWN_INFO: SystemInfo = {
  python_version: 'error',
  tinytorch_version: 'error',
  numpy_version: 'error',
  venv_active: false,
  platform: 'error',
};

export class InfoTreeProvider implements vscode.TreeDataProvider<TreeNode> {
  private _onDidChange = new vscode.EventEmitter<TreeNode | undefined>();
  readonly onDidChangeTreeData = this._onDidChange.event;

  private projectRoot: string;

  constructor(projectRoot: string) {
    this.projectRoot = projectRoot;
  }

  refresh(): void {
    this._onDidChange.fire(undefined);
  }

  getTreeItem(element: TreeNode): vscode.TreeItem {
    return element;
  }

  getChildren(element?: TreeNode): TreeNode[] {
    if (element) { return []; }

    const items: TreeNode[] = [];
    const info = this.getSystemInfo();
    const isError = info.python_version === 'error';

    if (isError) {
      items.push(new InfoTreeItem('Tito CLI unavailable', 'check output', 'error'));
    } else {
      items.push(new InfoTreeItem('Python', info.python_version, 'symbol-misc'));
      items.push(new InfoTreeItem('TinyTorch', info.tinytorch_version, 'flame'));
      items.push(new InfoTreeItem('NumPy', info.numpy_version, 'symbol-array'));
      items.push(new InfoTreeItem(
        'Virtual Env',
        info.venv_active ? 'active' : 'not active',
        info.venv_active ? 'pass' : 'warning',
      ));
    }

    // Separator (empty label)
    items.push(new InfoTreeItem(''));

    // Actions
    items.push(new ActionTreeItem('Open Notebook', 'tinytorch.openNotebook', [], 'notebook'));
    items.push(new ActionTreeItem('Run Health Check', 'tinytorch.runAction', [titoTerminalCommand('system health'), 'Health Check'], 'heart'));
    items.push(new ActionTreeItem('Run Setup', 'tinytorch.runAction', [titoTerminalCommand('setup'), 'Setup'], 'tools'));
    items.push(new ActionTreeItem('Clean Artifacts', 'tinytorch.runAction', [titoTerminalCommand('dev clean'), 'Clean'], 'trash'));

    return items;
  }

  /** Get system info from Tito CLI (single call instead of 4 separate ones) */
  private getSystemInfo(): SystemInfo {
    const info = callTitoJson<SystemInfo>(
      this.projectRoot,
      'system info --json',
      'system info',
      10_000,
    );

    if (!info) {
      log('System info unavailable — Tito CLI may not be installed.');
      return UNKNOWN_INFO;
    }

    return info;
  }
}
