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
      new ActionTreeItem('Test All Chapters (Parallel)...', 'mlsysbook.testAllChaptersParallel', [], 'beaker'),
      new ActionTreeItem('Cancel Current Parallel Session', 'mlsysbook.cancelParallelSession', [], 'debug-stop'),
      new ActionTreeItem('Rerun Failed Chapters', 'mlsysbook.rerunFailedParallel', [], 'debug-rerun'),
      new ActionTreeItem('Open Last Failure Details', 'mlsysbook.openLastFailureDetails', [], 'warning'),
    ];
  }
}
