import * as vscode from 'vscode';
import { ActionTreeItem } from '../models/treeItems';

export class DebugTreeProvider implements vscode.TreeDataProvider<ActionTreeItem> {
  private _onDidChangeTreeData = new vscode.EventEmitter<ActionTreeItem | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  getTreeItem(element: ActionTreeItem): vscode.TreeItem {
    return element;
  }

  getChildren(): ActionTreeItem[] {
    return [
      new ActionTreeItem('Build All Chapters (Sequential)', 'mlsysbook.debugAllChapters', [], 'run-all'),
      new ActionTreeItem('Build All Chapters (Parallel)', 'mlsysbook.testAllChaptersParallel', [], 'server-process'),
      new ActionTreeItem('Cancel Current Session', 'mlsysbook.cancelParallelSession', [], 'debug-stop'),
      new ActionTreeItem('Rerun Failed Chapters', 'mlsysbook.rerunFailedParallel', [], 'debug-rerun'),
    ];
  }
}
