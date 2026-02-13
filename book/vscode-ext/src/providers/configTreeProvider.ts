import * as vscode from 'vscode';
import { ActionTreeItem } from '../models/treeItems';

export class ConfigTreeProvider implements vscode.TreeDataProvider<ActionTreeItem> {
  private readonly _onDidChangeTreeData = new vscode.EventEmitter<ActionTreeItem | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  getTreeItem(element: ActionTreeItem): vscode.TreeItem {
    return element;
  }

  getChildren(): ActionTreeItem[] {
    return [
      new ActionTreeItem('Set Chapter Order Source', 'mlsysbook.setChapterOrderSource', [], 'list-ordered'),
      new ActionTreeItem('Set QMD Visual Preset', 'mlsysbook.setQmdVisualPreset', [], 'symbol-color'),
      new ActionTreeItem('Open MLSysBook Settings', 'mlsysbook.openSettings', [], 'settings-gear'),
    ];
  }
}
