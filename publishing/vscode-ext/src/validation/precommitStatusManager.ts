/**
 * Tracks last run status for pre-commit hooks so the tree can show check/cross icons.
 */

import * as vscode from 'vscode';

export type HookStatus = 'pending' | 'pass' | 'fail';

export class PrecommitStatusManager implements vscode.Disposable {
  private readonly _onDidChange = new vscode.EventEmitter<void>();
  readonly onDidChange = this._onDidChange.event;

  private _runAllStatus: HookStatus = 'pending';
  private _hookStatuses = new Map<string, HookStatus>();

  get runAllStatus(): HookStatus {
    return this._runAllStatus;
  }

  getHookStatus(hookId: string): HookStatus {
    return this._hookStatuses.get(hookId) ?? 'pending';
  }

  setRunAllResult(success: boolean): void {
    this._runAllStatus = success ? 'pass' : 'fail';
    this._onDidChange.fire();
  }

  setHookResult(hookId: string, success: boolean): void {
    this._hookStatuses.set(hookId, success ? 'pass' : 'fail');
    this._onDidChange.fire();
  }

  setRunAllPending(): void {
    this._runAllStatus = 'pending';
    this._onDidChange.fire();
  }

  setHookPending(hookId: string): void {
    this._hookStatuses.set(hookId, 'pending');
    this._onDidChange.fire();
  }

  dispose(): void {
    this._onDidChange.dispose();
  }
}
