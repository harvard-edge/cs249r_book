import * as vscode from 'vscode';
import { PRECOMMIT_HOOKS } from '../constants';
import { ActionTreeItem } from '../models/treeItems';

export class PrecommitTreeProvider implements vscode.TreeDataProvider<ActionTreeItem> {
  private _onDidChangeTreeData = new vscode.EventEmitter<ActionTreeItem | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  getTreeItem(element: ActionTreeItem): vscode.TreeItem {
    return element;
  }

  getChildren(): ActionTreeItem[] {
    const runAll = new ActionTreeItem(
      'Run ALL Hooks',
      'mlsysbook.precommitRunAll',
      [],
      'checklist',
    );

    const hookItems = PRECOMMIT_HOOKS.map(h =>
      new ActionTreeItem(h.label, 'mlsysbook.precommitRunHook', [h.command], 'play')
    );

    return [runAll, ...hookItems];
  }
}
