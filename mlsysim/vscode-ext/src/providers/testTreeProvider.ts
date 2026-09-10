import * as vscode from 'vscode';
import { mlsysimCommand } from '../utils/mlsysimCli';
import { CategoryTreeItem, ActionTreeItem } from '../models/treeItems';

type TreeNode = CategoryTreeItem | ActionTreeItem;

/** Test suite tree — mirrors mlsysim/tests/ structure */
export class TestTreeProvider implements vscode.TreeDataProvider<TreeNode> {
  private _onDidChange = new vscode.EventEmitter<TreeNode | undefined>();
  readonly onDidChangeTreeData = this._onDidChange.event;

  constructor(private projectRoot: string) {}

  refresh(): void {
    this._onDidChange.fire(undefined);
  }

  getTreeItem(element: TreeNode): vscode.TreeItem {
    return element;
  }

  getChildren(element?: TreeNode): TreeNode[] {
    if (!element) {
      return [
        new CategoryTreeItem('Run All', 'all', 'run-all'),
        new CategoryTreeItem('By Category', 'categories', 'layers'),
        new CategoryTreeItem('Validation', 'validation', 'verified'),
      ];
    }

    if (element instanceof CategoryTreeItem) {
      const pytest = `cd "${this.projectRoot}" && python3 -m pytest`;

      switch (element.categoryId) {
        case 'all':
          return [
            new ActionTreeItem('All Tests', 'mlsysim.runAction',
              [`${pytest} mlsysim/tests/ -v`, 'All Tests'], 'checklist'),
            new ActionTreeItem('Quick (Engine + Hardware)', 'mlsysim.runAction',
              [`${pytest} mlsysim/tests/test_engine.py mlsysim/tests/test_hardware.py -v`, 'Quick Tests'], 'zap'),
          ];

        case 'categories':
          return [
            new ActionTreeItem('Engine Tests', 'mlsysim.runAction',
              [`${pytest} mlsysim/tests/test_engine.py -v`, 'Engine Tests'], 'beaker'),
            new ActionTreeItem('Hardware Tests', 'mlsysim.runAction',
              [`${pytest} mlsysim/tests/test_hardware.py -v`, 'Hardware Tests'], 'chip'),
            new ActionTreeItem('Solver Suite', 'mlsysim.runAction',
              [`${pytest} mlsysim/tests/test_solver_suite.py -v`, 'Solver Suite'], 'symbol-ruler'),
            new ActionTreeItem('Empirical Validation', 'mlsysim.runAction',
              [`${pytest} mlsysim/tests/test_empirical.py -v`, 'Empirical Tests'], 'graph'),
            new ActionTreeItem('SOTA Paradigm Tests', 'mlsysim.runAction',
              [`${pytest} mlsysim/tests/test_sota.py -v`, 'SOTA Tests'], 'rocket'),
          ];

        case 'validation':
          return [
            new ActionTreeItem('Registry Contract', 'mlsysim.runAction',
              [`${pytest} mlsysim/tests/test_registry_loader_contract.py -v`, 'Registry Contract'], 'database'),
            new ActionTreeItem('Provenance Audit', 'mlsysim.runAction',
              [`cd "${this.projectRoot}" && PYTHONPATH=mlsysim python3 -m mlsysim.tools.audit_provenance --scope all --strict`, 'Provenance Audit'], 'verified'),
            new ActionTreeItem('Validate Paper Anchors', 'mlsysim.runAction',
              [`cd "${this.projectRoot}/mlsysim/paper" && python3 validate_anchors.py`, 'Paper Anchors'], 'file-text'),
          ];
      }
    }

    return [];
  }
}
