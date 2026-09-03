import * as vscode from 'vscode';
import { PRECOMMIT_CHECK_HOOKS, PRECOMMIT_FIXER_HOOKS, CHECK_ACTIONS } from '../constants';
import { ActionTreeItem } from '../models/treeItems';
import { HealthManager } from '../validation/healthManager';
import type { PrecommitStatusManager, HookStatus } from '../validation/precommitStatusManager';

type TreeNode = ActionTreeItem | SeparatorItem | HealthSummaryItem | HealthIssueItem | PrecommitActionItem;

function iconForStatus(status: HookStatus): vscode.ThemeIcon {
  switch (status) {
    case 'pass':
      return new vscode.ThemeIcon('pass', new vscode.ThemeColor('testing.iconPassed'));
    case 'fail':
      return new vscode.ThemeIcon('close', new vscode.ThemeColor('testing.iconFailed'));
    case 'pending':
    default:
      return new vscode.ThemeIcon('circle-outline');
  }
}

class PrecommitActionItem extends vscode.TreeItem {
  constructor(
    label: string,
    commandId: string,
    commandArgs: unknown[],
    status: HookStatus,
  ) {
    super(label, vscode.TreeItemCollapsibleState.None);
    this.command = { command: commandId, title: label, arguments: commandArgs };
    this.iconPath = iconForStatus(status);
    this.contextValue = 'precommit-action';
  }
}

// ---------------------------------------------------------------------------
// Tree item helpers
// ---------------------------------------------------------------------------

class SeparatorItem extends vscode.TreeItem {
  constructor(label: string) {
    super(label, vscode.TreeItemCollapsibleState.None);
    this.description = '';
    this.contextValue = 'separator';
  }
}

class HealthSummaryItem extends vscode.TreeItem {
  constructor(private readonly healthManager: HealthManager) {
    const status = healthManager.status;
    const hasIssues = status === 'error' || status === 'warn';
    super(
      healthManager.getSummaryText().replace('MLSysBook: ', ''),
      hasIssues
        ? vscode.TreeItemCollapsibleState.Expanded
        : vscode.TreeItemCollapsibleState.None,
    );

    this.contextValue = 'health-summary';
    this.tooltip = healthManager.getTooltip();

    switch (status) {
      case 'ok':
        this.iconPath = new vscode.ThemeIcon('pass', new vscode.ThemeColor('testing.iconPassed'));
        this.description = '';
        break;
      case 'warn':
        this.iconPath = new vscode.ThemeIcon('warning', new vscode.ThemeColor('list.warningForeground'));
        this.description = '';
        break;
      case 'error':
        this.iconPath = new vscode.ThemeIcon('error', new vscode.ThemeColor('list.errorForeground'));
        this.description = '';
        break;
      case 'pending':
      default:
        this.iconPath = new vscode.ThemeIcon('circle-outline');
        this.description = 'no files checked';
        break;
    }
  }

  getChildren(): HealthIssueItem[] {
    const allResults = this.healthManager.getAllResults();
    const items: HealthIssueItem[] = [];

    for (const fh of allResults) {
      const shortPath = fh.uri.replace(/^file:\/\//, '').split('/').slice(-2).join('/');
      for (const r of fh.results) {
        items.push(new HealthIssueItem(r.message, r.severity, shortPath, r.line, fh.uri));
      }
    }

    return items;
  }
}

class HealthIssueItem extends vscode.TreeItem {
  constructor(
    message: string,
    severity: 'error' | 'warning' | 'info',
    filePath: string,
    line: number,
    fileUri: string,
  ) {
    super(message, vscode.TreeItemCollapsibleState.None);
    this.contextValue = 'health-issue';
    this.description = `${filePath}:${line + 1}`;
    this.tooltip = `${message}\n${filePath} line ${line + 1}`;

    // Click to open file at the issue line
    const uri = vscode.Uri.parse(fileUri);
    this.command = {
      command: 'mlsysbook.openNavigatorLocation',
      title: 'Go to Issue',
      arguments: [uri, line],
    };

    switch (severity) {
      case 'error':
        this.iconPath = new vscode.ThemeIcon('circle-filled', new vscode.ThemeColor('list.errorForeground'));
        break;
      case 'warning':
        this.iconPath = new vscode.ThemeIcon('circle-filled', new vscode.ThemeColor('list.warningForeground'));
        break;
      case 'info':
        this.iconPath = new vscode.ThemeIcon('info', new vscode.ThemeColor('list.deemphasizedForeground'));
        break;
    }
  }
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export class PrecommitTreeProvider implements vscode.TreeDataProvider<TreeNode> {
  private _onDidChangeTreeData = new vscode.EventEmitter<TreeNode | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private _healthManager: HealthManager | undefined;
  private _healthSummary: HealthSummaryItem | undefined;
  private _precommitStatusManager: PrecommitStatusManager | undefined;

  /** Wire up the health manager (called from extension.ts). */
  setHealthManager(manager: HealthManager): void {
    this._healthManager = manager;
  }

  /** Wire up pre-commit status (called from extension.ts). */
  setPrecommitStatusManager(manager: PrecommitStatusManager): void {
    this._precommitStatusManager = manager;
    manager.onDidChange(() => this.refresh());
  }

  refresh(): void {
    this._healthSummary = undefined; // force rebuild
    this._onDidChangeTreeData.fire(undefined);
  }

  getTreeItem(element: TreeNode): vscode.TreeItem {
    return element;
  }

  getChildren(element?: TreeNode): TreeNode[] {
    // If expanding the health summary node, return its children
    if (element instanceof HealthSummaryItem) {
      return element.getChildren();
    }

    // Top-level children
    const items: TreeNode[] = [];

    // Health summary at the very top
    if (this._healthManager) {
      this._healthSummary = new HealthSummaryItem(this._healthManager);
      items.push(this._healthSummary);
      items.push(new SeparatorItem(''));
    }

    const runAllStatus = this._precommitStatusManager?.runAllStatus ?? 'pending';
    const runAll = new PrecommitActionItem(
      'Run ALL Hooks',
      'mlsysbook.precommitRunAll',
      [],
      runAllStatus,
    );

    const checkItems = PRECOMMIT_CHECK_HOOKS.map(h => {
      const status = this._precommitStatusManager?.getHookStatus(h.id) ?? 'pending';
      return new PrecommitActionItem(h.label, 'mlsysbook.precommitRunHook', [h.command], status);
    });

    const fixerItems = PRECOMMIT_FIXER_HOOKS.map(h => {
      const status = this._precommitStatusManager?.getHookStatus(h.id) ?? 'pending';
      return new PrecommitActionItem(h.label, 'mlsysbook.precommitRunHook', [h.command], status);
    });

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

    const binderCheckItems = CHECK_ACTIONS.map(action =>
      new ActionTreeItem(action.label, 'mlsysbook.validateRunAction', [action.command], action.icon)
    );

    items.push(
      runAll,
      new SeparatorItem('--- Pre-commit Checks ---'),
      ...checkItems,
      new SeparatorItem('--- Fixers / Cleanup (Manual) ---'),
      currentFileFixers,
      currentFileTableFixer,
      ...fixerItems,
      new SeparatorItem('--- Binder Check (Fast, Focused) ---'),
      ...binderCheckItems,
    );

    return items;
  }
}
