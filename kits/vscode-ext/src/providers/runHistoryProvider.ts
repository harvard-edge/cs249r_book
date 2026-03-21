import * as vscode from 'vscode';
import { getRecentRuns, onCommandRunsChanged } from '../utils/terminal';
import { RunHistoryTreeItem } from '../models/treeItems';

export class RunHistoryProvider implements vscode.TreeDataProvider<RunHistoryTreeItem>, vscode.Disposable {
  private _onDidChange = new vscode.EventEmitter<RunHistoryTreeItem | undefined>();
  readonly onDidChangeTreeData = this._onDidChange.event;

  private disposable: vscode.Disposable;

  constructor() {
    this.disposable = onCommandRunsChanged(() => this.refresh());
  }

  refresh(): void {
    this._onDidChange.fire(undefined);
  }

  dispose(): void {
    this.disposable.dispose();
  }

  getTreeItem(element: RunHistoryTreeItem): vscode.TreeItem {
    return element;
  }

  getChildren(): RunHistoryTreeItem[] {
    const runs = getRecentRuns();
    if (runs.length === 0) {
      return [];
    }
    return runs.map(r => new RunHistoryTreeItem(r));
  }
}
