import * as vscode from 'vscode';
import { CommandRunRecord } from '../types';

/** A clickable action leaf node */
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
      command: 'kits.rerunCommand',
      title: 'Rerun',
      arguments: [record],
    };
    this.contextValue = 'runRecord';
  }
}

/** A simple info/label row */
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

/** A lab file entry that opens in the editor */
export class LabTreeItem extends vscode.TreeItem {
  constructor(
    label: string,
    public readonly filePath: string,
    icon?: string,
  ) {
    super(label, vscode.TreeItemCollapsibleState.None);
    this.iconPath = new vscode.ThemeIcon(icon ?? 'file-text');
    this.contextValue = 'lab';
    this.command = {
      command: 'vscode.open',
      title: 'Open',
      arguments: [vscode.Uri.file(filePath)],
    };
    this.tooltip = filePath;
  }
}
