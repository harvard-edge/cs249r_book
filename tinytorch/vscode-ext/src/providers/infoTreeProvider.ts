import * as vscode from 'vscode';
import { execSync } from 'child_process';
import { InfoTreeItem, ActionTreeItem } from '../models/treeItems';

type TreeNode = InfoTreeItem | ActionTreeItem;

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

    // Environment info
    const pythonVersion = this.runQuiet('python3 --version');
    items.push(new InfoTreeItem('Python', pythonVersion.replace('Python ', ''), 'symbol-misc'));

    const tinyTorchVersion = this.readVersion();
    items.push(new InfoTreeItem('TinyTorch', tinyTorchVersion, 'flame'));

    const numpyVersion = this.runQuiet('python3 -c "import numpy; print(numpy.__version__)"');
    items.push(new InfoTreeItem('NumPy', numpyVersion || 'not installed', 'symbol-array'));

    const inVenv = this.runQuiet('python3 -c "import sys; print(sys.prefix != sys.base_prefix)"');
    const venvStatus = inVenv.trim() === 'True' ? 'active' : 'not active';
    items.push(new InfoTreeItem('Virtual Env', venvStatus, venvStatus === 'active' ? 'pass' : 'warning'));

    // Separator (empty label)
    items.push(new InfoTreeItem(''));

    // Actions
    items.push(new ActionTreeItem('Run Health Check', 'tinytorch.runAction', ['python3 -m tito.main system health', 'Health Check'], 'heart'));
    items.push(new ActionTreeItem('Run Setup', 'tinytorch.runAction', ['python3 -m tito.main setup', 'Setup'], 'tools'));

    return items;
  }

  /** Run a shell command quietly, returning stdout or a fallback */
  private runQuiet(command: string): string {
    try {
      return execSync(command, {
        cwd: this.projectRoot,
        timeout: 5000,
        encoding: 'utf-8',
      }).trim();
    } catch {
      return 'unknown';
    }
  }

  /** Read version from pyproject.toml */
  private readVersion(): string {
    try {
      const fs = require('fs');
      const path = require('path');
      const content = fs.readFileSync(path.join(this.projectRoot, 'pyproject.toml'), 'utf-8');
      const match = content.match(/version\s*=\s*"([^"]+)"/);
      return match ? match[1] : 'unknown';
    } catch {
      return 'unknown';
    }
  }
}
