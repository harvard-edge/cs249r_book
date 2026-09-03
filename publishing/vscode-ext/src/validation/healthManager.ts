/**
 * Central health-status tracker for the MLSysBook extension.
 *
 * Maintains per-file check results, aggregates an overall status,
 * and emits events so the status bar and tree view stay in sync.
 */

import * as vscode from 'vscode';
import { CheckResult, runAllChecks } from './qmdChecks';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type HealthStatus = 'ok' | 'warn' | 'error' | 'pending';

export interface FileHealth {
  uri: string;
  results: CheckResult[];
  checkedAt: number; // Date.now()
}

// ---------------------------------------------------------------------------
// HealthManager
// ---------------------------------------------------------------------------

export class HealthManager implements vscode.Disposable {
  private readonly _results = new Map<string, FileHealth>();

  private readonly _onDidUpdateHealth = new vscode.EventEmitter<void>();
  /** Fires whenever the aggregated health status changes. */
  readonly onDidUpdateHealth = this._onDidUpdateHealth.event;

  private _status: HealthStatus = 'pending';

  // -----------------------------------------------------------------------
  // Public API
  // -----------------------------------------------------------------------

  /** Run all fast in-process checks on a single document. */
  runFastChecks(document: vscode.TextDocument): CheckResult[] {
    const text = document.getText();
    const results = runAllChecks(text);

    this._results.set(document.uri.toString(), {
      uri: document.uri.toString(),
      results,
      checkedAt: Date.now(),
    });

    this._recomputeStatus();
    this._onDidUpdateHealth.fire();
    return results;
  }

  /** Remove results for a file (e.g. when it's closed). */
  clearFile(uri: vscode.Uri): void {
    if (this._results.delete(uri.toString())) {
      this._recomputeStatus();
      this._onDidUpdateHealth.fire();
    }
  }

  /** Current aggregated status across all tracked files. */
  get status(): HealthStatus {
    return this._status;
  }

  /** Total number of issues across all tracked files. */
  get totalIssues(): number {
    let count = 0;
    for (const fh of this._results.values()) {
      count += fh.results.length;
    }
    return count;
  }

  /** Number of errors (not warnings/info). */
  get errorCount(): number {
    let count = 0;
    for (const fh of this._results.values()) {
      for (const r of fh.results) {
        if (r.severity === 'error') { count++; }
      }
    }
    return count;
  }

  /** Number of warnings. */
  get warningCount(): number {
    let count = 0;
    for (const fh of this._results.values()) {
      for (const r of fh.results) {
        if (r.severity === 'warning') { count++; }
      }
    }
    return count;
  }

  /** Number of info-level issues. */
  get infoCount(): number {
    let count = 0;
    for (const fh of this._results.values()) {
      for (const r of fh.results) {
        if (r.severity === 'info') { count++; }
      }
    }
    return count;
  }

  /** Human-readable one-line summary for the status bar. */
  getSummaryText(): string {
    if (this._status === 'pending') { return 'MLSysBook'; }
    if (this._status === 'ok') { return 'MLSysBook: All Clear'; }

    const parts: string[] = [];
    const errors = this.errorCount;
    const warnings = this.warningCount;
    if (errors > 0) { parts.push(`${errors} error${errors !== 1 ? 's' : ''}`); }
    if (warnings > 0) { parts.push(`${warnings} warning${warnings !== 1 ? 's' : ''}`); }
    return `MLSysBook: ${parts.join(', ')}`;
  }

  /** Tooltip with more detail for the status bar hover. */
  getTooltip(): string {
    if (this._status === 'pending') { return 'MLSysBook — no files checked yet'; }
    if (this._status === 'ok') {
      const fileCount = this._results.size;
      return `MLSysBook — ${fileCount} file${fileCount !== 1 ? 's' : ''} checked, no issues`;
    }

    const lines: string[] = [`MLSysBook Health: ${this.totalIssues} issue(s)`];
    for (const fh of this._results.values()) {
      if (fh.results.length === 0) { continue; }
      const shortPath = fh.uri.replace(/^file:\/\//, '').split('/').slice(-2).join('/');
      lines.push(`  ${shortPath}: ${fh.results.length} issue(s)`);
    }
    return lines.join('\n');
  }

  /** All results grouped by file, for the tree view. */
  getAllResults(): FileHealth[] {
    return Array.from(this._results.values()).filter(fh => fh.results.length > 0);
  }

  /** Results for a specific file. */
  getFileResults(uri: string): CheckResult[] {
    return this._results.get(uri)?.results ?? [];
  }

  // -----------------------------------------------------------------------
  // Internals
  // -----------------------------------------------------------------------

  private _recomputeStatus(): void {
    if (this._results.size === 0) {
      this._status = 'pending';
      return;
    }

    let hasError = false;
    let hasWarning = false;

    for (const fh of this._results.values()) {
      for (const r of fh.results) {
        if (r.severity === 'error') { hasError = true; }
        if (r.severity === 'warning') { hasWarning = true; }
      }
    }

    if (hasError) { this._status = 'error'; }
    else if (hasWarning) { this._status = 'warn'; }
    else { this._status = 'ok'; }
  }

  dispose(): void {
    this._onDidUpdateHealth.dispose();
  }
}
