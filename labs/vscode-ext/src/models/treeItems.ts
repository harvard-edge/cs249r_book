import * as vscode from 'vscode';
import { CommandRunRecord, LabInfo } from '../types';

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
      command: 'labs.rerunCommand',
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

/** A lab entry in the Labs Navigator tree */
export class LabTreeItem extends vscode.TreeItem {
  constructor(public readonly lab: LabInfo) {
    const iconId = lab.status === 'implemented' ? 'pass-filled'
                 : lab.status === 'planned'     ? 'circle-outline'
                 : 'dash';
    super(
      `${lab.number} — ${lab.title}`,
      vscode.TreeItemCollapsibleState.None,
    );
    this.iconPath = new vscode.ThemeIcon(iconId);
    this.contextValue = `lab-${lab.status}`;
    this.description = lab.status;
    this.tooltip = `Lab ${lab.number}: ${lab.title}\nVolume ${lab.volume} | Status: ${lab.status}`;

    // Click opens the lab file (if implemented) or the plan
    const openPath = lab.labPath ?? lab.planPath;
    if (openPath) {
      this.command = {
        command: 'vscode.open',
        title: 'Open',
        arguments: [vscode.Uri.file(openPath)],
      };
    }
  }
}

/** A governance document entry */
export class GovernanceDocTreeItem extends vscode.TreeItem {
  constructor(
    label: string,
    public readonly filePath: string,
    description: string,
    icon?: string,
  ) {
    super(label, vscode.TreeItemCollapsibleState.None);
    this.description = description;
    this.iconPath = new vscode.ThemeIcon(icon ?? 'file-text');
    this.contextValue = 'governance-doc';
    this.command = {
      command: 'vscode.open',
      title: 'Open',
      arguments: [vscode.Uri.file(filePath)],
    };
    this.tooltip = filePath;
  }
}
