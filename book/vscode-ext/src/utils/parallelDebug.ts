import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { ChildProcess, spawn } from 'child_process';
import { VolumeId } from '../types';
import { getRepoRoot } from './workspace';
import { recordExternalCommandFinish, recordExternalCommandStart } from './terminal';

interface ParallelDebugJob {
  chapter: string;
  volume: VolumeId;
  format: string;
  worktreePath: string;
}

interface ParallelDebugResult {
  chapter: string;
  success: boolean;
  exitCode: number;
  elapsedMs: number;
  logPath?: string;  // path to preserved index.log for this chapter
}

interface ParallelDebugOptions {
  repoRoot: string;
  volume: VolumeId;
  format: string;
  chapters: string[];
  workers: number;
}

interface BatchRunOptions extends ParallelDebugOptions {
  sessionLabel: string;
  clearChannel: boolean;
  session: DebugRunSession;
  controller: SessionController;
}

const CHANNEL_NAME = 'MLSysBook Parallel Debug';
const MAX_SESSION_HISTORY = 30;
const CANCELLED_EXIT_CODE = 130;

export type DebugSessionMode = 'parallel' | 'bisect';
export type DebugSessionStatus = 'running' | 'completed' | 'failed' | 'cancelled';

export interface FailureLocation {
  chapter: string;
  filePath: string;
  line: number;
  message: string;
}

export interface DebugRunSession {
  id: string;
  label: string;
  mode: DebugSessionMode;
  volume: VolumeId;
  format: string;
  chapters: string[];
  workers: number;
  status: DebugSessionStatus;
  startedAt: string;
  endedAt?: string;
  elapsedMs?: number;
  failedChapters: string[];
  failedWorktrees: Record<string, string>;
  failureLocations: FailureLocation[];
}

interface SessionController {
  cancelRequested: boolean;
  activeChildren: Set<ChildProcess>;
}

const sessionsChangedEmitter = new vscode.EventEmitter<void>();
export const onDebugSessionsChanged = sessionsChangedEmitter.event;

let channelSingleton: vscode.OutputChannel | undefined;
const sessionStore: DebugRunSession[] = [];
let activeSessionId: string | undefined;
let activeController: SessionController | undefined;

function getParallelDebugChannel(): vscode.OutputChannel {
  if (!channelSingleton) {
    channelSingleton = vscode.window.createOutputChannel(CHANNEL_NAME);
  }
  return channelSingleton;
}

function nowStamp(): string {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

function getConfig() {
  return vscode.workspace.getConfiguration('mlsysbook');
}

function getWorktreeBaseDir(repoRoot: string): string {
  const relativeRoot = getConfig().get<string>('parallelDebugRoot', '.mlsysbook/worktrees');
  return path.join(repoRoot, relativeRoot);
}

function keepFailedWorktrees(): boolean {
  return getConfig().get<boolean>('keepFailedWorktrees', true);
}

function prefixAndWrite(
  channel: vscode.OutputChannel,
  tag: string,
  text: string,
  onRawLine?: (line: string) => void,
): void {
  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    if (line.trim().length === 0) { continue; }
    if (onRawLine) {
      onRawLine(line);
    }
    channel.appendLine(`[${tag}] ${line}`);
  }
}

function upsertSession(session: DebugRunSession): void {
  const idx = sessionStore.findIndex(item => item.id === session.id);
  if (idx >= 0) {
    sessionStore[idx] = session;
  } else {
    sessionStore.unshift(session);
    if (sessionStore.length > MAX_SESSION_HISTORY) {
      sessionStore.length = MAX_SESSION_HISTORY;
    }
  }
  sessionsChangedEmitter.fire();
}

function makeSession(mode: DebugSessionMode, options: ParallelDebugOptions, label: string): DebugRunSession {
  return {
    id: `${mode}-${options.volume}-${options.format}-${nowStamp()}`,
    label,
    mode,
    volume: options.volume,
    format: options.format,
    chapters: [...options.chapters],
    workers: options.workers,
    status: 'running',
    startedAt: new Date().toISOString(),
    failedChapters: [],
    failedWorktrees: {},
    failureLocations: [],
  };
}

function resolveFailurePath(rawPath: string, repoRoot: string, worktreePath: string): string {
  if (path.isAbsolute(rawPath)) {
    if (rawPath.startsWith(worktreePath)) {
      const suffix = rawPath.slice(worktreePath.length).replace(/^[/\\]/, '');
      return path.join(repoRoot, suffix);
    }
    return rawPath;
  }
  return path.join(repoRoot, rawPath);
}

function parseFailureLocations(
  chapter: string,
  lines: string[],
  repoRoot: string,
  worktreePath: string,
): FailureLocation[] {
  const locations: FailureLocation[] = [];
  const seen = new Set<string>();
  const directPattern = /([A-Za-z0-9_./\\-]+\.(?:qmd|py|tex|md|yml|yaml|json)):(\d+)(?::\d+)?/;
  const tracebackPattern = /File "([^"]+)", line (\d+)/;

  for (const line of lines) {
    let filePath: string | undefined;
    let lineNum: number | undefined;

    const directMatch = line.match(directPattern);
    if (directMatch) {
      filePath = directMatch[1];
      lineNum = Number(directMatch[2]);
    }

    if (!filePath || !lineNum) {
      const tracebackMatch = line.match(tracebackPattern);
      if (tracebackMatch) {
        filePath = tracebackMatch[1];
        lineNum = Number(tracebackMatch[2]);
      }
    }

    if (!filePath || !lineNum || Number.isNaN(lineNum)) {
      continue;
    }

    const resolvedPath = resolveFailurePath(filePath, repoRoot, worktreePath);
    const key = `${resolvedPath}:${lineNum}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    locations.push({
      chapter,
      filePath: resolvedPath,
      line: Math.max(1, lineNum),
      message: line.trim().slice(0, 180),
    });

    if (locations.length >= 10) {
      break;
    }
  }

  return locations;
}

async function runShellCommand(
  command: string,
  cwd: string,
  channel: vscode.OutputChannel,
  tag: string,
  controller?: SessionController,
  onRawLine?: (line: string) => void,
): Promise<number> {
  return new Promise<number>(resolve => {
    if (controller?.cancelRequested) {
      resolve(CANCELLED_EXIT_CODE);
      return;
    }

    const child = spawn(command, { cwd, shell: true, env: process.env });
    if (controller) {
      controller.activeChildren.add(child);
    }
    let stdoutBuffer = '';
    let stderrBuffer = '';

    child.stdout?.on('data', (chunk: Buffer | string) => {
      stdoutBuffer += chunk.toString();
      const lines = stdoutBuffer.split(/\r?\n/);
      stdoutBuffer = lines.pop() ?? '';
      prefixAndWrite(channel, tag, lines.join('\n'), onRawLine);
    });

    child.stderr?.on('data', (chunk: Buffer | string) => {
      stderrBuffer += chunk.toString();
      const lines = stderrBuffer.split(/\r?\n/);
      stderrBuffer = lines.pop() ?? '';
      prefixAndWrite(channel, tag, lines.join('\n'), onRawLine);
    });

    child.on('error', (err: Error) => {
      channel.appendLine(`[${tag}] runner error: ${err.message}`);
      if (controller) {
        controller.activeChildren.delete(child);
      }
      resolve(1);
    });

    child.on('close', (code: number | null) => {
      if (stdoutBuffer.trim().length > 0) {
        prefixAndWrite(channel, tag, stdoutBuffer, onRawLine);
      }
      if (stderrBuffer.trim().length > 0) {
        prefixAndWrite(channel, tag, stderrBuffer, onRawLine);
      }
      if (controller) {
        controller.activeChildren.delete(child);
      }
      resolve(code ?? 1);
    });
  });
}

/**
 * Copy the LaTeX index.log from a worktree build to the reports directory.
 * Returns the destination path, or undefined if no log was found.
 */
async function preserveBuildLog(
  worktreePath: string,
  chapter: string,
  reportsDir: string,
): Promise<string | undefined> {
  // Quarto places index.log in the book/quarto directory during PDF builds
  const logSource = path.join(worktreePath, 'book', 'quarto', 'index.log');
  if (!fs.existsSync(logSource)) {
    return undefined;
  }
  const logDest = path.join(reportsDir, `${chapter}.log`);
  try {
    await fs.promises.copyFile(logSource, logDest);
    return logDest;
  } catch {
    return undefined;
  }
}

/**
 * Write a markdown summary report for a completed test session.
 */
async function writeSummaryReport(
  reportsDir: string,
  results: ParallelDebugResult[],
  session: DebugRunSession,
): Promise<string> {
  const reportPath = path.join(reportsDir, 'REPORT.md');
  const passed = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);
  const totalMs = session.elapsedMs ?? 0;

  const lines: string[] = [];
  lines.push(`# Chapter Build Report`);
  lines.push('');
  lines.push(`- **Volume**: ${session.volume}`);
  lines.push(`- **Format**: ${session.format}`);
  lines.push(`- **Workers**: ${session.workers}`);
  lines.push(`- **Started**: ${session.startedAt}`);
  lines.push(`- **Finished**: ${session.endedAt ?? 'N/A'}`);
  lines.push(`- **Total time**: ${(totalMs / 1000).toFixed(1)}s`);
  lines.push(`- **Result**: ${failed.length === 0 ? 'ALL PASSED' : `${failed.length} FAILED`}`);
  lines.push('');
  lines.push(`## Summary: ${passed.length}/${results.length} passed`);
  lines.push('');
  lines.push('| Chapter | Status | Time | Log |');
  lines.push('|---------|--------|------|-----|');

  // Sort results by chapter order (same as input)
  for (const r of results) {
    const status = r.success ? 'PASS' : '**FAIL**';
    const time = `${(r.elapsedMs / 1000).toFixed(1)}s`;
    const logFile = r.logPath ? `[${path.basename(r.logPath)}](${path.basename(r.logPath)})` : '—';
    lines.push(`| ${r.chapter} | ${status} | ${time} | ${logFile} |`);
  }

  if (failed.length > 0) {
    lines.push('');
    lines.push('## Failed Chapters');
    lines.push('');
    for (const r of failed) {
      lines.push(`### ${r.chapter}`);
      lines.push('');
      lines.push(`- Exit code: ${r.exitCode}`);
      lines.push(`- Elapsed: ${(r.elapsedMs / 1000).toFixed(1)}s`);
      if (r.logPath) {
        lines.push(`- Log: \`${r.logPath}\``);
      }
      const locations = session.failureLocations.filter(l => l.chapter === r.chapter);
      if (locations.length > 0) {
        lines.push('- Error locations:');
        for (const loc of locations) {
          lines.push(`  - \`${loc.filePath}:${loc.line}\` — ${loc.message}`);
        }
      }
      lines.push('');
    }
  }

  lines.push('');
  lines.push(`## Log Files`);
  lines.push('');
  lines.push(`All logs stored in: \`${reportsDir}\``);
  lines.push('');
  for (const r of results) {
    if (r.logPath) {
      lines.push(`- \`${path.basename(r.logPath)}\` — ${r.chapter} (${r.success ? 'pass' : 'fail'})`);
    }
  }

  await fs.promises.writeFile(reportPath, lines.join('\n'), 'utf8');
  return reportPath;
}

async function cleanupWorktree(
  repoRoot: string,
  worktreePath: string,
  forceRemoveDir: boolean,
  channel: vscode.OutputChannel,
): Promise<void> {
  await runShellCommand(`git worktree remove --force "${worktreePath}"`, repoRoot, channel, 'cleanup');
  if (forceRemoveDir) {
    await fs.promises.rm(worktreePath, { recursive: true, force: true });
  }
}

async function runParallelDebugBatch(options: BatchRunOptions): Promise<{ results: ParallelDebugResult[]; reportsDir: string }> {
  const { repoRoot, volume, format, chapters, workers, sessionLabel, clearChannel, session, controller } = options;
  const channel = getParallelDebugChannel();
  if (clearChannel) {
    channel.clear();
  }
  channel.show(true);

  const sessionId = `${sessionLabel}-${volume}-${format}-${nowStamp()}`;
  const sessionDir = path.join(getWorktreeBaseDir(repoRoot), sessionId);
  await fs.promises.mkdir(sessionDir, { recursive: true });

  // Create a reports directory to store logs and the summary report
  const reportsDir = path.join(repoRoot, '.mlsysbook', 'reports', sessionId);
  await fs.promises.mkdir(reportsDir, { recursive: true });

  const jobs: ParallelDebugJob[] = chapters.map(chapter => ({
    chapter,
    volume,
    format,
    worktreePath: path.join(sessionDir, chapter),
  }));

  channel.appendLine(`Session: ${sessionId}`);
  channel.appendLine(`Volume: ${volume}`);
  channel.appendLine(`Format: ${format}`);
  channel.appendLine(`Workers: ${workers}`);
  channel.appendLine(`Chapters: ${chapters.join(', ')}`);
  channel.appendLine(`Reports: ${reportsDir}`);
  channel.appendLine('');

  const results: ParallelDebugResult[] = [];
  let nextIndex = 0;

  async function runWorker(workerId: number): Promise<void> {
    while (true) {
      const index = nextIndex++;
      if (index >= jobs.length) { return; }
      if (controller.cancelRequested) { return; }
      const job = jobs[index];
      const tag = `w${workerId}:${job.chapter}`;
      const start = Date.now();

      channel.appendLine(`[${tag}] creating worktree at ${job.worktreePath}`);
      const addCode = await runShellCommand(
        `git worktree add --detach "${job.worktreePath}" HEAD`,
        repoRoot,
        channel,
        tag,
        controller,
      );
      if (addCode !== 0) {
        results.push({
          chapter: job.chapter,
          success: false,
          exitCode: addCode,
          elapsedMs: Date.now() - start,
        });
        channel.appendLine(`[${tag}] failed to create worktree (exit ${addCode})`);
        if (index === 0) {
          channel.appendLine('[session] Tip: ensure the workspace is the repo root and `git` is in PATH.');
        }
        continue;
      }

      const buildCommand = `./book/binder build ${job.format} ${job.chapter} --${job.volume} -v`;
      const jobLogLines: string[] = [];
      channel.appendLine(`[${tag}] running ${buildCommand}`);
      const exitCode = await runShellCommand(
        buildCommand,
        job.worktreePath,
        channel,
        tag,
        controller,
        (line: string) => jobLogLines.push(line),
      );
      const elapsedMs = Date.now() - start;
      const success = exitCode === 0;

      // Preserve the build log before worktree cleanup
      const logPath = await preserveBuildLog(job.worktreePath, job.chapter, reportsDir);
      if (logPath) {
        channel.appendLine(`[${tag}] log saved: ${logPath}`);
      }

      results.push({ chapter: job.chapter, success, exitCode, elapsedMs, logPath });
      channel.appendLine(`[${tag}] ${success ? 'PASS' : 'FAIL'} in ${(elapsedMs / 1000).toFixed(1)}s`);

      if (success || !keepFailedWorktrees()) {
        await cleanupWorktree(repoRoot, job.worktreePath, true, channel);
      } else {
        session.failedWorktrees[job.chapter] = job.worktreePath;
        channel.appendLine(`[${tag}] keeping failed worktree at ${job.worktreePath}`);
      }

      if (!success) {
        const locations = parseFailureLocations(job.chapter, jobLogLines, repoRoot, job.worktreePath);
        if (locations.length > 0) {
          session.failureLocations.push(...locations);
        }
      }
    }
  }

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: `MLSysBook ${sessionLabel} (${chapters.length} chapters)`,
      cancellable: true,
    },
    async (progress, token) => {
      token.onCancellationRequested(() => {
        controller.cancelRequested = true;
        channel.appendLine('[session] cancellation requested by user');
        for (const child of controller.activeChildren) {
          try {
            child.kill('SIGTERM');
          } catch {
            // best effort kill
          }
        }
      });
      progress.report({ message: 'Running jobs...' });
      const workerCount = Math.min(Math.max(1, workers), jobs.length);
      await Promise.all(Array.from({ length: workerCount }, (_, idx) => runWorker(idx + 1)));
    },
  );

  await runShellCommand('git worktree prune', repoRoot, channel, 'cleanup');
  return { results, reportsDir };
}

export async function runParallelChapterDebug(options: ParallelDebugOptions): Promise<void> {
  const { repoRoot, volume, format, chapters, workers } = options;
  if (chapters.length === 0) {
    vscode.window.showWarningMessage('MLSysBook: select at least one chapter for parallel debug.');
    return;
  }

  const channel = getParallelDebugChannel();
  const session = makeSession('parallel', options, 'Parallel Chapter Debug');
  const controller: SessionController = { cancelRequested: false, activeChildren: new Set() };
  activeSessionId = session.id;
  activeController = controller;
  upsertSession(session);

  const start = Date.now();
  const { results, reportsDir } = await runParallelDebugBatch({
    repoRoot,
    volume,
    format,
    chapters,
    workers,
    sessionLabel: 'parallel-debug',
    clearChannel: true,
    session,
    controller,
  });

  const failed = results.filter(r => !r.success).map(r => r.chapter);
  const passedCount = results.length - failed.length;
  session.failedChapters = failed;
  session.endedAt = new Date().toISOString();
  session.elapsedMs = Date.now() - start;
  session.status = controller.cancelRequested
    ? 'cancelled'
    : failed.length > 0 ? 'failed' : 'completed';
  upsertSession(session);
  activeSessionId = undefined;
  activeController = undefined;

  // Write the summary report
  const reportPath = await writeSummaryReport(reportsDir, results, session);

  // Print summary to output channel
  channel.appendLine('');
  channel.appendLine('═'.repeat(60));
  channel.appendLine(`  BUILD REPORT: ${passedCount}/${results.length} chapters passed`);
  channel.appendLine('═'.repeat(60));
  channel.appendLine('');
  for (const r of results) {
    const status = r.success ? '✓ PASS' : '✗ FAIL';
    const time = `${(r.elapsedMs / 1000).toFixed(1)}s`;
    const log = r.logPath ? path.basename(r.logPath) : '—';
    channel.appendLine(`  ${status}  ${r.chapter.padEnd(25)} ${time.padStart(8)}   log: ${log}`);
  }
  channel.appendLine('');
  channel.appendLine(`Total time: ${((session.elapsedMs ?? 0) / 1000).toFixed(1)}s`);
  channel.appendLine(`Report: ${reportPath}`);
  channel.appendLine(`Logs:   ${reportsDir}`);

  if (failed.length > 0) {
    channel.appendLine('');
    channel.appendLine(`Failed chapters: ${failed.join(', ')}`);
    if (session.failureLocations.length > 0) {
      channel.appendLine('');
      channel.appendLine('Error locations:');
      for (const location of session.failureLocations.slice(0, 8)) {
        channel.appendLine(`  ${location.filePath}:${location.line} (${location.chapter})`);
      }
    }
  }
  channel.appendLine('═'.repeat(60));
  channel.show(true);

  if (controller.cancelRequested) {
    void vscode.window.showWarningMessage('Parallel debug cancelled.');
    return;
  }

  if (failed.length === 0) {
    const action = await vscode.window.showInformationMessage(
      `All ${results.length} chapters passed.`,
      'Open Report',
    );
    if (action === 'Open Report') {
      const doc = await vscode.workspace.openTextDocument(reportPath);
      await vscode.window.showTextDocument(doc, { preview: true });
    }
    return;
  }

  const action = await vscode.window.showErrorMessage(
    `${failed.length} chapter(s) failed: ${failed.join(', ')}`,
    'Open Report',
    'Show Output',
  );
  if (action === 'Open Report') {
    const doc = await vscode.workspace.openTextDocument(reportPath);
    await vscode.window.showTextDocument(doc, { preview: true });
  } else if (action === 'Show Output') {
    channel.show(true);
  }
}

function splitInHalf(chapters: string[]): [string[], string[]] {
  const mid = Math.ceil(chapters.length / 2);
  return [chapters.slice(0, mid), chapters.slice(mid)];
}

export async function runBisectChapterDebug(options: ParallelDebugOptions): Promise<void> {
  const { repoRoot, volume, format, workers } = options;
  let candidates = [...options.chapters];
  if (candidates.length === 0) {
    vscode.window.showWarningMessage('MLSysBook: select at least one chapter for bisect debug.');
    return;
  }

  const channel = getParallelDebugChannel();
  const session = makeSession('bisect', options, 'Bisect Chapter Debug');
  const controller: SessionController = { cancelRequested: false, activeChildren: new Set() };
  activeSessionId = session.id;
  activeController = controller;
  upsertSession(session);

  const start = Date.now();
  channel.clear();
  channel.show(true);
  channel.appendLine(`Bisect start (${candidates.length} chapters): ${candidates.join(', ')}`);
  channel.appendLine('');

  let iteration = 1;
  while (candidates.length > 1) {
    const [left, right] = splitInHalf(candidates);
    channel.appendLine(`Iteration ${iteration}: testing left half (${left.length})`);
    const leftBatch = await runParallelDebugBatch({
      repoRoot,
      volume,
      format,
      chapters: left,
      workers,
      sessionLabel: `bisect-${iteration}-left`,
      clearChannel: false,
      session,
      controller,
    });
    if (controller.cancelRequested) { break; }
    const leftFailed = leftBatch.results.filter(r => !r.success).map(r => r.chapter);
    if (leftFailed.length > 0) {
      channel.appendLine(`Iteration ${iteration}: left half contains failures -> narrowing`);
      channel.appendLine('');
      candidates = leftFailed;
      iteration++;
      continue;
    }

    channel.appendLine(`Iteration ${iteration}: testing right half (${right.length})`);
    const rightBatch = await runParallelDebugBatch({
      repoRoot,
      volume,
      format,
      chapters: right,
      workers,
      sessionLabel: `bisect-${iteration}-right`,
      clearChannel: false,
      session,
      controller,
    });
    if (controller.cancelRequested) { break; }
    const rightFailed = rightBatch.results.filter(r => !r.success).map(r => r.chapter);
    if (rightFailed.length > 0) {
      channel.appendLine(`Iteration ${iteration}: right half contains failures -> narrowing`);
      channel.appendLine('');
      candidates = rightFailed;
      iteration++;
      continue;
    }

    channel.appendLine(`Iteration ${iteration}: no failing subset found. Stopping.`);
    channel.appendLine('This can happen with flaky/non-deterministic failures.');
    break;
  }

  session.failedChapters = [...candidates];
  session.endedAt = new Date().toISOString();
  session.elapsedMs = Date.now() - start;
  session.status = controller.cancelRequested
    ? 'cancelled'
    : candidates.length === 0 ? 'completed'
      : candidates.length === 1 ? 'failed' : 'failed';
  upsertSession(session);
  activeSessionId = undefined;
  activeController = undefined;

  if (session.failureLocations.length > 0) {
    channel.appendLine('');
    channel.appendLine('Top failure locations:');
    for (const location of session.failureLocations.slice(0, 8)) {
      channel.appendLine(`- ${location.filePath}:${location.line} (${location.chapter})`);
    }
  }

  if (controller.cancelRequested) {
    void vscode.window.showWarningMessage('Bisect debug cancelled.');
    return;
  }

  if (candidates.length === 1) {
    const culprit = candidates[0];
    channel.appendLine('');
    channel.appendLine(`Bisect result: likely failing chapter = ${culprit}`);
    channel.show(true);
    void vscode.window.showErrorMessage(`Bisect isolated likely failing chapter: ${culprit}`);
    return;
  }

  channel.appendLine(`Bisect ended with ${candidates.length} candidates: ${candidates.join(', ')}`);
  channel.show(true);
  void vscode.window.showWarningMessage('Bisect completed without isolating a single chapter. Check parallel debug output.');
}

export function getRecentDebugSessions(): DebugRunSession[] {
  return [...sessionStore];
}

export function getDebugSessionById(sessionId: string): DebugRunSession | undefined {
  return sessionStore.find(session => session.id === sessionId);
}

export function cancelActiveDebugSession(): boolean {
  if (!activeController || !activeSessionId) {
    return false;
  }
  activeController.cancelRequested = true;
  for (const child of activeController.activeChildren) {
    try {
      child.kill('SIGTERM');
    } catch {
      // best effort kill
    }
  }
  const session = getDebugSessionById(activeSessionId);
  if (session && session.status === 'running') {
    session.status = 'cancelled';
    session.endedAt = new Date().toISOString();
    upsertSession(session);
  }
  return true;
}

export function getLastFailedDebugSession(): DebugRunSession | undefined {
  return sessionStore.find(session =>
    session.status !== 'running' && session.failedChapters.length > 0
  );
}

export async function rerunDebugSession(sessionId: string, failedOnly: boolean): Promise<boolean> {
  const session = getDebugSessionById(sessionId);
  if (!session) {
    return false;
  }
  const chapters = failedOnly ? session.failedChapters : session.chapters;
  if (chapters.length === 0) {
    return false;
  }
  const options: ParallelDebugOptions = {
    repoRoot: getRepoRoot() ?? '',
    volume: session.volume,
    format: session.format,
    chapters,
    workers: session.workers,
  };
  if (!options.repoRoot) {
    return false;
  }
  if (session.mode === 'bisect' && !failedOnly) {
    await runBisectChapterDebug(options);
    return true;
  }
  await runParallelChapterDebug(options);
  return true;
}

export function revealParallelDebugOutput(): void {
  getParallelDebugChannel().show(true);
}

interface IsolatedDebugRunOptions {
  repoRoot: string;
  command: string;
  label: string;
  keepFailedWorktree?: boolean;
}

export async function runIsolatedDebugCommand(options: IsolatedDebugRunOptions): Promise<boolean> {
  const { repoRoot, command, label } = options;
  const keepFailedWorktree = options.keepFailedWorktree ?? keepFailedWorktrees();
  const runId = recordExternalCommandStart(command, repoRoot, label, 'raw');
  const channel = getParallelDebugChannel();
  channel.show(true);

  const sessionId = `isolated-debug-${nowStamp()}`;
  const sessionDir = path.join(getWorktreeBaseDir(repoRoot), sessionId);
  const worktreePath = path.join(sessionDir, 'workspace');
  await fs.promises.mkdir(sessionDir, { recursive: true });

  channel.appendLine(`[isolated] ${label}`);
  channel.appendLine(`[isolated] create worktree: ${worktreePath}`);
  const addCode = await runShellCommand(
    `git worktree add --detach "${worktreePath}" HEAD`,
    repoRoot,
    channel,
    'isolated',
  );
  if (addCode !== 0) {
    channel.appendLine(`[isolated] failed to create worktree (exit ${addCode})`);
    recordExternalCommandFinish(runId, false);
    void vscode.window.showErrorMessage(`${label} failed: could not create isolated worktree.`);
    return false;
  }

  channel.appendLine(`[isolated] run: ${command}`);
  const exitCode = await runShellCommand(command, worktreePath, channel, 'isolated');
  const success = exitCode === 0;
  if (success || !keepFailedWorktree) {
    await cleanupWorktree(repoRoot, worktreePath, true, channel);
  } else {
    channel.appendLine(`[isolated] keeping failed worktree: ${worktreePath}`);
  }
  await runShellCommand('git worktree prune', repoRoot, channel, 'isolated');

  if (success) {
    recordExternalCommandFinish(runId, true);
    void vscode.window.showInformationMessage(`${label} succeeded (isolated worktree).`);
  } else {
    recordExternalCommandFinish(runId, false);
    void vscode.window.showErrorMessage(`${label} failed (isolated worktree).`);
  }
  return success;
}
