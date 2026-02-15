import * as vscode from 'vscode';
import { CategoryTreeItem, ActionTreeItem } from '../models/treeItems';

type TreeNode = CategoryTreeItem | ActionTreeItem;

export class BuildTreeProvider implements vscode.TreeDataProvider<TreeNode> {
  private _onDidChange = new vscode.EventEmitter<TreeNode | undefined>();
  readonly onDidChangeTreeData = this._onDidChange.event;

  getTreeItem(element: TreeNode): vscode.TreeItem {
    return element;
  }

  getChildren(element?: TreeNode): TreeNode[] {
    if (!element) {
      return [
        new CategoryTreeItem('Site (HTML)', 'site', 'globe'),
        new CategoryTreeItem('PDF Course Guide', 'pdf', 'file-pdf'),
        new CategoryTreeItem('Research Paper', 'paper', 'book'),
        new CategoryTreeItem('Utilities', 'utilities', 'tools'),
      ];
    }

    if (element instanceof CategoryTreeItem) {
      switch (element.categoryId) {
        case 'site':
          return [
            new ActionTreeItem('Build HTML', 'tinytorch.runSiteAction', ['cd site && make html', 'Build HTML'], 'globe'),
            new ActionTreeItem('Build & Serve (localhost:8000)', 'tinytorch.runSiteAction', ['cd site && make serve', 'Serve Site'], 'play'),
            new ActionTreeItem('Open in Browser', 'tinytorch.openSiteInBrowser', [], 'link-external'),
            new ActionTreeItem('Clean Build', 'tinytorch.runSiteAction', ['cd site && make clean', 'Clean Site'], 'trash'),
          ];

        case 'pdf':
          return [
            new ActionTreeItem('Build PDF', 'tinytorch.runSiteAction', ['cd site && make pdf', 'Build PDF'], 'file-pdf'),
            new ActionTreeItem('Open PDF', 'tinytorch.openPdf', ['site/_build/latex/tinytorch-course.pdf'], 'eye'),
          ];

        case 'paper':
          return [
            new ActionTreeItem('Build Paper (Full)', 'tinytorch.runSiteAction', ['cd site && make paper', 'Build Paper'], 'file-pdf'),
            new ActionTreeItem('Build Paper (Quick)', 'tinytorch.runPaperAction', ['cd paper && make quick', 'Paper Quick'], 'zap'),
            new ActionTreeItem('Open Paper PDF', 'tinytorch.openPdf', ['paper/paper.pdf'], 'eye'),
          ];

        case 'utilities':
          return [
            new ActionTreeItem('Health Check', 'tinytorch.runAction', ['python3 -m tito.main system health', 'Health Check'], 'heart'),
            new ActionTreeItem('Clean Artifacts', 'tinytorch.runAction', ['make clean', 'Clean'], 'trash'),
          ];
      }
    }

    return [];
  }
}
