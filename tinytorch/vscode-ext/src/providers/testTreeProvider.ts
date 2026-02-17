import * as vscode from 'vscode';
import { discoverModules } from '../utils/modules';
import { titoTerminalCommand } from '../utils/tito';
import { ModuleInfo } from '../types';
import { CategoryTreeItem, ActionTreeItem } from '../models/treeItems';

type TreeNode = CategoryTreeItem | ActionTreeItem;

/**
 * Testing tree mirrors the CI pipeline (tinytorch-validate-dev.yml).
 *
 * Uses `tito dev test` with the same flags CI uses:
 *   --inline       Stage 1: Progressive module build
 *   --unit         Stage 2: Pytest unit tests
 *   --integration  Stage 3: Cross-module validation
 *   --cli          Stage 4: CLI command tests
 *   --e2e          Stage 5: End-to-end journey
 *   --milestone    Stage 6: Milestone script tests
 *   --user-journey Stage 7: Full destructive journey
 *   --module N     Test a single module
 */
export class TestTreeProvider implements vscode.TreeDataProvider<TreeNode> {
  private _onDidChange = new vscode.EventEmitter<TreeNode | undefined>();
  readonly onDidChangeTreeData = this._onDidChange.event;

  private modules: ModuleInfo[] = [];

  constructor(private projectRoot: string) {
    this.modules = discoverModules(projectRoot).modules;
  }

  refresh(): void {
    this.modules = discoverModules(this.projectRoot).modules;
    this._onDidChange.fire(undefined);
  }

  getTreeItem(element: TreeNode): vscode.TreeItem {
    return element;
  }

  getChildren(element?: TreeNode): TreeNode[] {
    if (!element) {
      return [
        new CategoryTreeItem('Run All', 'all', 'run-all'),
        new CategoryTreeItem('By Stage (CI Pipeline)', 'stages', 'layers'),
        new CategoryTreeItem('Single Module', 'module', 'symbol-method'),
      ];
    }

    if (element instanceof CategoryTreeItem) {
      switch (element.categoryId) {
        case 'all':
          return [
            new ActionTreeItem('Quick (Unit + CLI)', 'tinytorch.runAction',
              [titoTerminalCommand('dev test --unit --cli'), 'Quick Tests'], 'zap'),
            new ActionTreeItem('Standard (Stages 1-5)', 'tinytorch.runAction',
              [titoTerminalCommand('dev test --inline --unit --integration --cli --e2e'), 'Standard Tests'], 'test-view-icon'),
            new ActionTreeItem('All Tests', 'tinytorch.runAction',
              [titoTerminalCommand('dev test --all'), 'All Tests'], 'checklist'),
            new ActionTreeItem('User Journey (destructive)', 'tinytorch.runAction',
              [titoTerminalCommand('dev test --user-journey'), 'User Journey'], 'warning'),
          ];

        case 'stages':
          return [
            new ActionTreeItem('Stage 1: Inline Build', 'tinytorch.runAction',
              [titoTerminalCommand('dev test --inline'), 'Inline Build'], 'symbol-constructor'),
            new ActionTreeItem('Stage 2: Unit Tests', 'tinytorch.runAction',
              [titoTerminalCommand('dev test --unit'), 'Unit Tests'], 'beaker'),
            new ActionTreeItem('Stage 3: Integration', 'tinytorch.runAction',
              [titoTerminalCommand('dev test --integration'), 'Integration Tests'], 'link'),
            new ActionTreeItem('Stage 4: CLI Tests', 'tinytorch.runAction',
              [titoTerminalCommand('dev test --cli'), 'CLI Tests'], 'terminal'),
            new ActionTreeItem('Stage 5: E2E Tests', 'tinytorch.runAction',
              [titoTerminalCommand('dev test --e2e'), 'E2E Tests'], 'globe'),
            new ActionTreeItem('Stage 6: Milestone Tests', 'tinytorch.runAction',
              [titoTerminalCommand('dev test --milestone'), 'Milestone Tests'], 'mortar-board'),
          ];

        case 'module':
          return this.modules.map(m =>
            new ActionTreeItem(
              `${m.number} — ${m.title ?? m.displayName}`,
              'tinytorch.runAction',
              [titoTerminalCommand(`dev test --unit --module ${m.number}`), `Test Module ${m.number}`],
              'beaker',
            )
          );
      }
    }

    return [];
  }
}
