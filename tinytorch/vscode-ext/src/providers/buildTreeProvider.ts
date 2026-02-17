import * as vscode from 'vscode';
import { CategoryTreeItem, ActionTreeItem } from '../models/treeItems';
import { titoTerminalCommand } from '../utils/tito';
import { BUILD_OUTPUTS } from '../commands/buildCommands';

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
      ];
    }

    if (element instanceof CategoryTreeItem) {
      switch (element.categoryId) {
        case 'site':
          return [
            new ActionTreeItem('Build HTML', 'tinytorch.runSiteAction', [titoTerminalCommand('dev build html'), 'Build HTML'], 'globe'),
            new ActionTreeItem('Build & Serve (localhost:8000)', 'tinytorch.runSiteAction', [titoTerminalCommand('dev build serve'), 'Serve Site'], 'play'),
            new ActionTreeItem('Open in Browser', 'tinytorch.openSiteInBrowser', [], 'link-external'),
            new ActionTreeItem('Clean Build', 'tinytorch.runSiteAction', [titoTerminalCommand('dev clean site'), 'Clean Site'], 'trash'),
          ];

        case 'pdf':
          return [
            new ActionTreeItem('Build PDF', 'tinytorch.runSiteAction', [titoTerminalCommand('dev build pdf'), 'Build PDF'], 'file-pdf'),
            new ActionTreeItem('Open PDF', 'tinytorch.openPdf', [BUILD_OUTPUTS.coursePdf], 'eye'),
          ];

        case 'paper':
          return [
            new ActionTreeItem('Build Paper', 'tinytorch.runSiteAction', [titoTerminalCommand('dev build paper'), 'Build Paper'], 'file-pdf'),
            new ActionTreeItem('Open Paper PDF', 'tinytorch.openPdf', [BUILD_OUTPUTS.paperPdf], 'eye'),
          ];
      }
    }

    return [];
  }
}
