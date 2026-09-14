import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { ScenarioTreeItem, CategoryTreeItem } from '../models/treeItems';

type TreeNode = CategoryTreeItem | ScenarioTreeItem;

/** Lists YAML scenario files from mlsysim/examples/yaml/ */
export class ScenarioTreeProvider implements vscode.TreeDataProvider<TreeNode> {
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
    if (element) { return []; }

    const yamlDir = path.join(this.projectRoot, 'mlsysim', 'examples', 'yaml');
    const items: ScenarioTreeItem[] = [];

    if (fs.existsSync(yamlDir)) {
      const files = fs.readdirSync(yamlDir)
        .filter(f => f.endsWith('.yaml') || f.endsWith('.yml'))
        .sort();

      for (const file of files) {
        const filePath = path.join(yamlDir, file);
        const label = this.extractScenarioName(filePath) ?? file.replace(/\.(yaml|yml)$/, '');
        items.push(new ScenarioTreeItem(label, filePath));
      }
    }

    if (items.length === 0) {
      return [new ScenarioTreeItem('No scenarios found', '')];
    }

    return items;
  }

  /** Try to extract the scenario name from the YAML file's name: field */
  private extractScenarioName(filePath: string): string | null {
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      const match = content.match(/^name:\s*["']?(.+?)["']?\s*$/m);
      return match ? match[1] : null;
    } catch {
      return null;
    }
  }
}
