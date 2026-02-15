import * as vscode from 'vscode';
import { VolumeInfo } from '../types';
import { discoverChapters } from '../utils/chapters';
import { VolumeTreeItem, ChapterTreeItem, ActionTreeItem } from '../models/treeItems';

type TreeNode = VolumeTreeItem | ChapterTreeItem | ActionTreeItem;

export class BuildTreeProvider implements vscode.TreeDataProvider<TreeNode> {
  private _onDidChangeTreeData = new vscode.EventEmitter<TreeNode | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private volumes: VolumeInfo[] = [];

  constructor(private repoRoot: string) {
    this.refresh();
  }

  refresh(): void {
    this.volumes = discoverChapters(this.repoRoot);
    this._onDidChangeTreeData.fire(undefined);
  }

  getTreeItem(element: TreeNode): vscode.TreeItem {
    return element;
  }

  getChildren(element?: TreeNode): TreeNode[] {
    if (!element) {
      return this.volumes.map(v =>
        new VolumeTreeItem(v.id, v.label, v.chapters.length)
      );
    }

    if (element instanceof VolumeTreeItem) {
      const vol = this.volumes.find(v => v.id === element.volumeId);
      if (!vol) { return []; }

      const chapterItems = vol.chapters.map(ch => new ChapterTreeItem(ch));

      const volumeActions = [
        new ActionTreeItem('Build Full Volume...', 'mlsysbook.buildFullVolume', [vol.id], 'package'),
        new ActionTreeItem('Build Chapters...', 'mlsysbook.buildSelectedChapters', [vol.id], 'files'),
        new ActionTreeItem('Build All Chapters (Parallel)...', 'mlsysbook.buildAllChaptersParallel', [vol.id], 'beaker'),
      ];

      return [...volumeActions, ...chapterItems];
    }

    if (element instanceof ChapterTreeItem) {
      const ch = element.chapter;
      return [
        new ActionTreeItem('Build HTML', 'mlsysbook.buildChapterHtml', [ch.volume, ch.name], 'globe'),
        new ActionTreeItem('Build PDF', 'mlsysbook.buildChapterPdf', [ch.volume, ch.name], 'file-pdf'),
        new ActionTreeItem('Build EPUB', 'mlsysbook.buildChapterEpub', [ch.volume, ch.name], 'book'),
        new ActionTreeItem('Preview', 'mlsysbook.previewChapter', [ch.volume, ch.name], 'eye'),
      ];
    }

    return [];
  }
}
