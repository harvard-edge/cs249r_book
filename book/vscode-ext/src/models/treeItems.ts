import * as vscode from 'vscode';
import { ChapterInfo, VolumeId } from '../types';

export class VolumeTreeItem extends vscode.TreeItem {
  constructor(
    public readonly volumeId: VolumeId,
    label: string,
    chapterCount: number,
  ) {
    super(label, vscode.TreeItemCollapsibleState.Collapsed);
    this.description = `${chapterCount} chapters`;
    this.contextValue = 'volume';
    this.iconPath = new vscode.ThemeIcon('book');
  }
}

export class ChapterTreeItem extends vscode.TreeItem {
  constructor(public readonly chapter: ChapterInfo) {
    super(chapter.displayName, vscode.TreeItemCollapsibleState.Collapsed);
    this.contextValue = 'chapter';
    this.iconPath = new vscode.ThemeIcon('file-text');
    this.tooltip = `${chapter.volume}/${chapter.name}`;
  }
}

export class ActionTreeItem extends vscode.TreeItem {
  constructor(
    label: string,
    commandId: string,
    commandArgs: unknown[],
    icon?: string,
  ) {
    super(label, vscode.TreeItemCollapsibleState.None);
    this.command = {
      command: commandId,
      title: label,
      arguments: commandArgs,
    };
    this.iconPath = new vscode.ThemeIcon(icon ?? 'play');
    this.contextValue = 'action';
  }
}
