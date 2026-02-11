import * as vscode from 'vscode';
import { PRECOMMIT_HOOKS, VALIDATE_ACTIONS } from '../constants';
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

    const validateItems = VALIDATE_ACTIONS.map(action =>
      new ActionTreeItem(action.label, 'mlsysbook.validateRunAction', [action.command], action.icon)
    );

    return [runAll, ...hookItems, ...validateItems];
  }
}
