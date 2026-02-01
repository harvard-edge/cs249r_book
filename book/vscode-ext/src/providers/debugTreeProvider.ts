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
      new ActionTreeItem('Debug Vol1 PDF', 'mlsysbook.debugVolumePdf', ['vol1'], 'bug'),
      new ActionTreeItem('Debug Vol2 PDF', 'mlsysbook.debugVolumePdf', ['vol2'], 'bug'),
      new ActionTreeItem('Debug Chapter Sections...', 'mlsysbook.debugChapterSections', [], 'search'),
    ];
  }
}
