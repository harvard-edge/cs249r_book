import * as vscode from 'vscode';
import { PRECOMMIT_CHECK_HOOKS, PRECOMMIT_FIXER_HOOKS, VALIDATE_ACTIONS } from '../constants';
import { ActionTreeItem } from '../models/treeItems';

type TreeNode = ActionTreeItem | SeparatorItem;

class SeparatorItem extends vscode.TreeItem {
  constructor(label: string) {
    super(label, vscode.TreeItemCollapsibleState.None);
    this.description = '';
    this.contextValue = 'separator';
  }
}

export class PrecommitTreeProvider implements vscode.TreeDataProvider<TreeNode> {
  private _onDidChangeTreeData = new vscode.EventEmitter<TreeNode | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  getTreeItem(element: TreeNode): vscode.TreeItem {
    return element;
  }

  getChildren(): TreeNode[] {
    const runAll = new ActionTreeItem(
      'Run ALL Hooks',
      'mlsysbook.precommitRunAll',
      [],
      'checklist',
    );

    const checkItems = PRECOMMIT_CHECK_HOOKS.map(h =>
      new ActionTreeItem(h.label, 'mlsysbook.precommitRunHook', [h.command], 'play')
    );

    const fixerItems = PRECOMMIT_FIXER_HOOKS.map(h =>
      new ActionTreeItem(h.label, 'mlsysbook.precommitRunHook', [h.command], 'wrench')
    );

    const currentFileFixers = new ActionTreeItem(
      'Run QMD Fixers (Current File)',
      'mlsysbook.precommitRunFixersCurrentFile',
      [],
      'wand'
    );
    const currentFileTableFixer = new ActionTreeItem(
      'Prettify Pipe Tables (Current File)',
      'mlsysbook.precommitRunHookCurrentFile',
      ['book-prettify-pipe-tables', 'Current-file fixer: Prettify Pipe Tables'],
      'table',
    );

    const validateItems = VALIDATE_ACTIONS.map(action =>
      new ActionTreeItem(action.label, 'mlsysbook.validateRunAction', [action.command], action.icon)
    );

    return [
      runAll,
      new SeparatorItem('--- Pre-commit Checks ---'),
      ...checkItems,
      new SeparatorItem('--- Fixers / Cleanup (Manual) ---'),
      currentFileFixers,
      currentFileTableFixer,
      ...fixerItems,
      new SeparatorItem('--- Binder Validate (Fast, Focused) ---'),
      ...validateItems,
    ];
  }
}
