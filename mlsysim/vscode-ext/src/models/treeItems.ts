import * as vscode from 'vscode';
import { CommandRunRecord } from '../types';

/** A clickable action leaf node (used in multiple trees) */
export class ActionTreeItem extends vscode.TreeItem {
  constructor(
    label: string,
    commandId: string,
    args: unknown[] = [],
    icon?: string,
  ) {
    super(label, vscode.TreeItemCollapsibleState.None);
    if (icon) {
      this.iconPath = new vscode.ThemeIcon(icon);
    }
    this.command = { command: commandId, title: label, arguments: args };
    this.contextValue = 'action';
  }
}

/** A category header in a tree */
export class CategoryTreeItem extends vscode.TreeItem {
  constructor(
    label: string,
    public readonly categoryId: string,
    icon?: string,
  ) {
    super(label, vscode.TreeItemCollapsibleState.Expanded);
    this.iconPath = new vscode.ThemeIcon(icon ?? 'symbol-folder');
    this.contextValue = 'category';
  }
}

/** An entry in the Run History tree */
export class RunHistoryTreeItem extends vscode.TreeItem {
  constructor(public readonly record: CommandRunRecord) {
    const iconId = record.status === 'succeeded' ? 'pass'
                 : record.status === 'failed'    ? 'error'
                 : 'loading~spin';
    const time = new Date(record.timestamp).toLocaleTimeString();
    super(record.label, vscode.TreeItemCollapsibleState.None);
    this.iconPath = new vscode.ThemeIcon(iconId);
    this.description = time;
    this.tooltip = `${record.label}\n${record.command}\n${time}`;
    this.command = {
      command: 'mlsysim.rerunCommand',
      title: 'Rerun',
      arguments: [record],
    };
    this.contextValue = 'runRecord';
  }
}

/** A simple info/label row (non-clickable) */
export class InfoTreeItem extends vscode.TreeItem {
  constructor(label: string, detail?: string, icon?: string) {
    super(label, vscode.TreeItemCollapsibleState.None);
    this.description = detail;
    if (icon) {
      this.iconPath = new vscode.ThemeIcon(icon);
    }
    this.contextValue = 'info';
  }
}

/** A zoo registry entry (hardware device or model) */
export class ZooEntryTreeItem extends vscode.TreeItem {
  constructor(
    label: string,
    detail: string,
    public readonly entryType: 'hardware' | 'model',
    icon?: string,
  ) {
    super(label, vscode.TreeItemCollapsibleState.None);
    this.description = detail;
    this.iconPath = new vscode.ThemeIcon(icon ?? 'symbol-constant');
    this.contextValue = `zoo-${entryType}`;
  }
}

/** A scenario YAML file entry */
export class ScenarioTreeItem extends vscode.TreeItem {
  constructor(
    label: string,
    public readonly filePath: string,
  ) {
    super(label, vscode.TreeItemCollapsibleState.None);
    this.iconPath = new vscode.ThemeIcon('file-code');
    this.contextValue = 'scenario';
    this.command = {
      command: 'vscode.open',
      title: 'Open',
      arguments: [vscode.Uri.file(filePath)],
    };
    this.tooltip = filePath;
  }
}
