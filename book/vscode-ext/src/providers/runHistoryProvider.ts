import * as vscode from 'vscode';
import {
  DebugRunSession,
  FailureLocation,
  getRecentDebugSessions,
  onDebugSessionsChanged,
} from '../utils/parallelDebug';

class SessionTreeItem extends vscode.TreeItem {
  constructor(public readonly session: DebugRunSession) {
    super(`${session.label}`, vscode.TreeItemCollapsibleState.Collapsed);
    const elapsed = session.elapsedMs ? `${(session.elapsedMs / 1000).toFixed(1)}s` : 'running';
    this.description = `${session.status} · ${session.volume}/${session.format} · ${elapsed}`;
    this.iconPath = new vscode.ThemeIcon(
      session.status === 'failed' ? 'error' :
        session.status === 'completed' ? 'check' :
          session.status === 'cancelled' ? 'circle-slash' : 'sync'
    );
    this.contextValue = 'run-session';
  }
}

class SessionActionItem extends vscode.TreeItem {
  constructor(label: string, command: string, args: unknown[], icon: string) {
    super(label, vscode.TreeItemCollapsibleState.None);
    this.command = { command, title: label, arguments: args };
    this.iconPath = new vscode.ThemeIcon(icon);
    this.contextValue = 'run-session-action';
  }
}

class FailureLocationItem extends vscode.TreeItem {
  constructor(sessionId: string, location: FailureLocation) {
    super(`${location.chapter}: ${location.filePath.split('/').pop()}:${location.line}`, vscode.TreeItemCollapsibleState.None);
    this.description = location.message;
    this.tooltip = `${location.filePath}:${location.line}\n${location.message}`;
    this.iconPath = new vscode.ThemeIcon('go-to-file');
    this.command = {
      command: 'mlsysbook.historyOpenFailureLocation',
      title: 'Open Failure Location',
      arguments: [sessionId, location.filePath, location.line],
    };
    this.contextValue = 'run-failure-location';
  }
}

type HistoryNode = SessionTreeItem | SessionActionItem | FailureLocationItem;

export class RunHistoryProvider implements vscode.TreeDataProvider<HistoryNode>, vscode.Disposable {
  private readonly emitter = new vscode.EventEmitter<HistoryNode | undefined>();
  readonly onDidChangeTreeData = this.emitter.event;
  private readonly subscriptions: vscode.Disposable[] = [];

  constructor() {
    this.subscriptions.push(onDebugSessionsChanged(() => this.refresh()));
  }

  dispose(): void {
    this.subscriptions.forEach(s => s.dispose());
    this.emitter.dispose();
  }

  refresh(): void {
    this.emitter.fire(undefined);
  }

  getTreeItem(element: HistoryNode): vscode.TreeItem {
    return element;
  }

  getChildren(element?: HistoryNode): HistoryNode[] {
    if (!element) {
      return getRecentDebugSessions().slice(0, 15).map(session => new SessionTreeItem(session));
    }

    if (element instanceof SessionTreeItem) {
      const session = element.session;
      const actions: HistoryNode[] = [
        new SessionActionItem('Rerun Same', 'mlsysbook.historyRerunSession', [session.id], 'history'),
        new SessionActionItem('Rerun Failed Only', 'mlsysbook.historyRerunFailed', [session.id], 'debug-rerun'),
        new SessionActionItem('Open Parallel Output', 'mlsysbook.historyOpenOutput', [session.id], 'output'),
      ];
      if (Object.keys(session.failedWorktrees).length > 0) {
        actions.push(
          new SessionActionItem(
            'Open Failed Worktree',
            'mlsysbook.historyOpenFailedWorktree',
            [session.id],
            'folder-opened'
          )
        );
      }
      const locations = session.failureLocations.slice(0, 5).map(
        location => new FailureLocationItem(session.id, location)
      );
      return [...actions, ...locations];
    }

    return [];
  }
}
