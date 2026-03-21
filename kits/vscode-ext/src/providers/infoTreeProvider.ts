import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { execSync } from 'child_process';
import { InfoTreeItem, ActionTreeItem } from '../models/treeItems';

type TreeNode = InfoTreeItem | ActionTreeItem;

export class InfoTreeProvider implements vscode.TreeDataProvider<TreeNode> {
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

    const items: TreeNode[] = [];
    const kitsDir = path.join(this.projectRoot, 'kits');

    // Quarto version
    const quartoVersion = this.getCommandOutput('quarto --version');
    items.push(new InfoTreeItem(
      'Quarto',
      quartoVersion ?? 'unavailable',
      quartoVersion ? 'pass' : 'error',
    ));

    // Make available
    const makeVersion = this.getCommandOutput('make --version');
    items.push(new InfoTreeItem(
      'Make',
      makeVersion ? 'available' : 'unavailable',
      makeVersion ? 'pass' : 'error',
    ));

    // Count labs
    const labCount = this.countLabs(kitsDir);
    items.push(new InfoTreeItem('Lab Files', `${labCount} .qmd files`, 'file-text'));

    // Build output exists
    const buildExists = fs.existsSync(path.join(kitsDir, '_build'));
    items.push(new InfoTreeItem(
      'Build Output',
      buildExists ? '_build/ exists' : 'not built',
      buildExists ? 'folder' : 'circle-outline',
    ));

    items.push(new InfoTreeItem(''));

    // Actions
    items.push(new ActionTreeItem('Build HTML', 'kits.buildHtml', [], 'globe'));
    items.push(new ActionTreeItem('Preview', 'kits.preview', [], 'eye'));
    items.push(new ActionTreeItem('Clean', 'kits.clean', [], 'trash'));

    return items;
  }

  private getCommandOutput(cmd: string): string | null {
    try {
      const output = execSync(cmd, {
        encoding: 'utf-8',
        timeout: 5_000,
        stdio: ['pipe', 'pipe', 'pipe'],
      });
      return output.trim().split('\n')[0];
    } catch {
      return null;
    }
  }

  private countLabs(kitsDir: string): number {
    const contentsDir = path.join(kitsDir, 'contents');
    if (!fs.existsSync(contentsDir)) { return 0; }

    let count = 0;
    const walk = (dir: string) => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        if (entry.isDirectory()) {
          walk(path.join(dir, entry.name));
        } else if (entry.name.endsWith('.qmd')) {
          count++;
        }
      }
    };
    walk(contentsDir);
    return count;
  }
}
