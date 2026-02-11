import * as vscode from 'vscode';
import { ActionTreeItem } from '../models/treeItems';

export class DebugTreeProvider implements vscode.TreeDataProvider<ActionTreeItem> {
  private _onDidChangeTreeData = new vscode.EventEmitter<ActionTreeItem | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  getTreeItem(element: ActionTreeItem): vscode.TreeItem {
    return element;
  }

  getChildren(): ActionTreeItem[] {
    return [
      new ActionTreeItem('Quick Build Current Chapter (PDF)', 'mlsysbook.quickBuildCurrentChapterPdf', [], 'zap'),
      new ActionTreeItem('Quick Parallel Debug Selected (PDF)', 'mlsysbook.quickParallelDebugSelectedPdf', [], 'rocket'),
      new ActionTreeItem('Cancel Current Parallel Session', 'mlsysbook.cancelParallelSession', [], 'debug-stop'),
      new ActionTreeItem('Rerun Failed Chapters', 'mlsysbook.rerunFailedParallel', [], 'debug-rerun'),
      new ActionTreeItem('Debug Vol1 PDF', 'mlsysbook.debugVolumePdf', ['vol1'], 'bug'),
      new ActionTreeItem('Debug Vol2 PDF', 'mlsysbook.debugVolumePdf', ['vol2'], 'bug'),
      new ActionTreeItem('Debug Chapter Sections...', 'mlsysbook.debugChapterSections', [], 'search'),
      new ActionTreeItem('Parallel Debug Chapters...', 'mlsysbook.debugParallelChapters', [], 'organization'),
      new ActionTreeItem('Bisect Failing Chapters...', 'mlsysbook.debugBisectChapters', [], 'split-horizontal'),
      new ActionTreeItem('Open Last Failure Details', 'mlsysbook.openLastFailureDetails', [], 'warning'),
      new ActionTreeItem('Rerun Last Command', 'mlsysbook.rerunLastCommand', [], 'history'),
      new ActionTreeItem('Rerun Last Command (Raw)', 'mlsysbook.rerunLastCommandRaw', [], 'terminal'),
    ];
  }
}
