import * as vscode from 'vscode';
import { ModuleInfo, CommandRunRecord } from '../types';

/** A module entry in the Modules tree */
export class ModuleTreeItem extends vscode.TreeItem {
  constructor(public readonly module: ModuleInfo) {
    const icon = module.status === 'completed' ? '$(pass-filled)'
               : module.status === 'started'   ? '$(edit)'
               : '$(circle-outline)';
    super(
      `${icon}  ${module.number} — ${module.title ?? module.displayName}`,
      vscode.TreeItemCollapsibleState.Collapsed,
    );
    this.contextValue = 'module';
    this.tooltip = `Module ${module.number}: ${module.title ?? module.displayName}\nStatus: ${module.status.replace('_', ' ')}`;
    this.description = module.status === 'not_started' ? '' : module.status.replace('_', ' ');
  }
}

/** A clickable action leaf node (used in multiple trees) */
export class ActionTreeItem extends vscode.TreeItem {
  constructor(
    label: string,
    commandId: string,
    args: unknown[] = [],
    icon?: string,
  ) {
    // Visually offset leaf actions under expanded categories in sidebars.
    super(`  ${label}`, vscode.TreeItemCollapsibleState.None);
    this.command = { command: commandId, title: label, arguments: args };
    if (icon) {
      this.iconPath = new vscode.ThemeIcon(icon);
    }
    this.contextValue = 'action';
  }
}

/** A category header in a tree (e.g. "Quick Tests", "E2E Tests") */
export class CategoryTreeItem extends vscode.TreeItem {
  constructor(
    label: string,
    public readonly categoryId: string,
  ) {
    super(label, vscode.TreeItemCollapsibleState.Expanded);
    this.contextValue = 'category';
  }
}

/** An entry in the Run History tree */
export class RunHistoryTreeItem extends vscode.TreeItem {
  constructor(public readonly record: CommandRunRecord) {
    const icon = record.status === 'succeeded' ? '$(pass)'
               : record.status === 'failed'    ? '$(error)'
               : '$(loading~spin)';
    const time = new Date(record.timestamp).toLocaleTimeString();
    super(`${icon}  ${record.label}`, vscode.TreeItemCollapsibleState.None);
    this.description = time;
    this.tooltip = `${record.label}\n${record.command}\n${time}`;
    this.command = {
      command: 'tinytorch.rerunCommand',
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
