import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { GovernanceDocTreeItem, CategoryTreeItem, ActionTreeItem } from '../models/treeItems';

type TreeNode = CategoryTreeItem | GovernanceDocTreeItem | ActionTreeItem;

/** Governance & Audit — quick access to governance docs and audit actions */
export class GovernanceProvider implements vscode.TreeDataProvider<TreeNode> {
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
        new CategoryTreeItem('Governance Documents', 'docs', 'law'),
        new CategoryTreeItem('Audit Actions', 'audit', 'checklist'),
      ];
    }

    if (element instanceof CategoryTreeItem) {
      switch (element.categoryId) {
        case 'docs':
          return this.getGovernanceDocs();
        case 'audit':
          return this.getAuditActions();
      }
    }

    return [];
  }

  private getGovernanceDocs(): GovernanceDocTreeItem[] {
    const labsDir = path.join(this.projectRoot, 'labs');
    const docs = [
      { file: 'PROTOCOL.md', label: 'Protocol', desc: '7 invariants', icon: 'shield' },
      { file: 'TEMPLATE.md', label: 'Template', desc: '22-cell standard', icon: 'symbol-structure' },
      { file: 'README.md', label: 'README', desc: 'Lab overview', icon: 'info' },
    ];

    return docs
      .map(d => {
        const filePath = path.join(labsDir, d.file);
        if (!fs.existsSync(filePath)) { return null; }
        return new GovernanceDocTreeItem(d.label, filePath, d.desc, d.icon);
      })
      .filter((d): d is GovernanceDocTreeItem => d !== null);
  }

  private getAuditActions(): ActionTreeItem[] {
    return [
      new ActionTreeItem('Audit Current Lab (30-Gate)', 'labs.labAudit', [], 'checklist'),
      new ActionTreeItem('Check Template Compliance', 'labs.labCheckTemplate', [], 'symbol-structure'),
      new ActionTreeItem('Audit All Labs', 'labs.labAuditAll', [], 'run-all'),
    ];
  }
}
