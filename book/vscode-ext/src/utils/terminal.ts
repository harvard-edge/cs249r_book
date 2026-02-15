import * as vscode from 'vscode';
import { spawn } from 'child_process';

const TERMINAL_NAME = 'MLSysBook';
const OUTPUT_CHANNEL_NAME = 'MLSysBook Build';
const STATE_LAST_COMMAND_KEY = 'mlsysbook.lastCommand';
const STATE_LAST_FAILURE_KEY = 'mlsysbook.lastFailure';
const STATE_COMMAND_HISTORY_KEY = 'mlsysbook.commandHistory';
const MAX_COMMAND_HISTORY = 50;

export type ExecutionMode = 'focused' | 'raw';

interface LastCommandState {
  command: string;
  cwd: string;
  label: string;
  timestamp: string;
}

interface LastFailureState extends LastCommandState {
  exitCode: number;
  stage: string;
  logTail: string;
}

export type CommandRunStatus = 'started' | 'succeeded' | 'failed';

export interface CommandRunRecord {
  id: string;
  label: string;
  command: string;
  cwd: string;
  timestamp: string;
  status: CommandRunStatus;
  mode: 'raw' | 'focused';
}

export interface RunOptions {
  mode?: ExecutionMode;
  label?: string;
  /** When provided, runs in focused mode to capture exit code, then calls with success. */
  onComplete?: (success: boolean) => void;
}

let extensionContext: vscode.ExtensionContext | undefined;
let outputChannel: vscode.OutputChannel | undefined;
let runCounter = 0;
const commandRuns: CommandRunRecord[] = [];
const commandRunsEmitter = new vscode.EventEmitter<void>();
export const onCommandRunsChanged = commandRunsEmitter.event;

function getOutputChannel(): vscode.OutputChannel {
  if (!outputChannel) {
    outputChannel = vscode.window.createOutputChannel(OUTPUT_CHANNEL_NAME);
  }
  return outputChannel;
}

function ensureContext(): vscode.ExtensionContext | undefined {
  if (!extensionContext) {
    vscode.window.showWarningMessage('MLSysBook: run manager is not initialized.');
  }
  return extensionContext;
}

function getExecutionMode(): ExecutionMode {
  // MLSysBook runs in raw mode by design to guarantee immediate terminal feedback.
  return 'raw';
}

function shouldRevealOnFailure(): boolean {
  return vscode.workspace
    .getConfiguration('mlsysbook')
    .get<boolean>('revealTerminalOnFailure', true);
}

function getOrCreateTerminal(cwd: string): vscode.Terminal {
  let terminal = vscode.window.terminals.find(t => t.name === TERMINAL_NAME);
  if (!terminal) {
    terminal = vscode.window.createTerminal({ name: TERMINAL_NAME, cwd });
  }
  return terminal;
}

function detectFailureStage(command: string, logTail: string): string {
  const lower = `${command}\n${logTail}`.toLowerCase();
  if (lower.includes('quarto')) { return 'Quarto render'; }
  if (lower.includes('pybtex') || lower.includes('citation')) { return 'Citation processing'; }
  if (lower.includes('pre-commit')) { return 'Pre-commit validation'; }
  if (lower.includes('traceback')) { return 'Python execution'; }
  if (lower.includes('latex') || lower.includes('xelatex')) { return 'PDF/LaTeX compilation'; }
  return 'Build command';
}

function makeSummary(runId: number, label: string, success: boolean, elapsedMs: number, command: string): string {
  const status = success ? 'SUCCEEDED' : 'FAILED';
  const seconds = (elapsedMs / 1000).toFixed(1);
  return `[run ${runId}] ${label} ${status} in ${seconds}s\nCommand: ${command}`;
}

function storeLastCommand(state: LastCommandState): void {
  const context = ensureContext();
  if (!context) { return; }
  void context.workspaceState.update(STATE_LAST_COMMAND_KEY, state);
}

function storeLastFailure(state: LastFailureState): void {
  const context = ensureContext();
  if (!context) { return; }
  void context.workspaceState.update(STATE_LAST_FAILURE_KEY, state);
}

function persistCommandRuns(): void {
  const context = ensureContext();
  if (!context) { return; }
  void context.workspaceState.update(STATE_COMMAND_HISTORY_KEY, commandRuns);
}

function recordCommandStart(command: string, cwd: string, label: string, mode: 'raw' | 'focused'): CommandRunRecord {
  const record: CommandRunRecord = {
    id: `cmd-${Date.now()}-${++runCounter}`,
    label,
    command,
    cwd,
    timestamp: new Date().toISOString(),
    status: 'started',
    mode,
  };
  commandRuns.unshift(record);
  if (commandRuns.length > MAX_COMMAND_HISTORY) {
    commandRuns.length = MAX_COMMAND_HISTORY;
  }
  persistCommandRuns();
  commandRunsEmitter.fire();
  return record;
}

function updateCommandStatus(id: string, status: CommandRunStatus): void {
  const target = commandRuns.find(item => item.id === id);
  if (!target) { return; }
  target.status = status;
  persistCommandRuns();
  commandRunsEmitter.fire();
}

function getLastCommand(): LastCommandState | undefined {
  const context = ensureContext();
  if (!context) { return undefined; }
  return context.workspaceState.get<LastCommandState>(STATE_LAST_COMMAND_KEY);
}

function getLastFailure(): LastFailureState | undefined {
  const context = ensureContext();
  if (!context) { return undefined; }
  return context.workspaceState.get<LastFailureState>(STATE_LAST_FAILURE_KEY);
}

export function initializeRunManager(context: vscode.ExtensionContext): void {
  extensionContext = context;
  const savedRuns = context.workspaceState.get<CommandRunRecord[]>(STATE_COMMAND_HISTORY_KEY, []);
  if (Array.isArray(savedRuns) && savedRuns.length > 0) {
    commandRuns.splice(0, commandRuns.length, ...savedRuns.slice(0, MAX_COMMAND_HISTORY));
  }
  context.subscriptions.push(getOutputChannel());
  context.subscriptions.push(commandRunsEmitter);
}

export async function runBookCommand(command: string, cwd: string, options: RunOptions = {}): Promise<void> {
  const mode = options.onComplete ? 'focused' : (options.mode ?? getExecutionMode());
  const label = options.label ?? 'MLSysBook command';
  const runId = Date.now();
  const start = Date.now();
  const runRecord = recordCommandStart(command, cwd, label, mode === 'raw' ? 'raw' : 'focused');

  storeLastCommand({
    command,
    cwd,
    label,
    timestamp: new Date().toISOString(),
  });

  if (mode === 'raw') {
    const terminal = getOrCreateTerminal(cwd);
    terminal.show(false);
    terminal.sendText(command);
    getOutputChannel().appendLine(`[run ${runId}] RAW started: ${label}`);
    getOutputChannel().appendLine(`Command: ${command}`);
    return;
  }

  const channel = getOutputChannel();
  channel.appendLine(`[run ${runId}] START ${label}`);
  channel.appendLine(`Command: ${command}`);

  const child = spawn(command, {
    cwd,
    shell: true,
    env: process.env,
  });

  let combinedLog = '';
  const maxLogChars = 30000;

  const appendChunk = (chunk: string): void => {
    channel.append(chunk);
    combinedLog = (combinedLog + chunk).slice(-maxLogChars);
  };

  child.stdout?.on('data', (data: Buffer | string) => appendChunk(data.toString()));
  child.stderr?.on('data', (data: Buffer | string) => appendChunk(data.toString()));

  const exitCode = await new Promise<number>(resolve => {
    child.on('error', (err: Error) => {
      appendChunk(`\n[runner-error] ${err.message}\n`);
      resolve(1);
    });
    child.on('close', (code: number | null) => resolve(code ?? 1));
  });

  const elapsedMs = Date.now() - start;
  const success = exitCode === 0;
  updateCommandStatus(runRecord.id, success ? 'succeeded' : 'failed');
  channel.appendLine('');
  channel.appendLine(makeSummary(runId, label, success, elapsedMs, command));
  channel.appendLine('');

  options.onComplete?.(success);

  if (success) {
    void vscode.window.showInformationMessage(`${label} succeeded.`);
    return;
  }

  const stage = detectFailureStage(command, combinedLog);
  const logTail = combinedLog.slice(-8000);
  storeLastFailure({
    command,
    cwd,
    label,
    timestamp: new Date().toISOString(),
    exitCode,
    stage,
    logTail,
  });

  const actions = ['Open Failure Details', 'Rerun in Raw Terminal'];
  if (shouldRevealOnFailure()) {
    actions.unshift('Reveal Terminal');
  }

  const selected = await vscode.window.showErrorMessage(
    `${label} failed (${stage}).`,
    ...actions
  );

  if (selected === 'Reveal Terminal') {
    const terminal = getOrCreateTerminal(cwd);
    terminal.show(false);
    terminal.sendText(`# Last failed command:\n${command}`);
    return;
  }
  if (selected === 'Rerun in Raw Terminal') {
    await runBookCommand(command, cwd, { mode: 'raw', label });
    return;
  }
  if (selected === 'Open Failure Details') {
    showLastFailureDetails();
  }
}

export function runInTerminal(command: string, cwd: string): void {
  void runBookCommand(command, cwd, { mode: 'raw' });
}

export function runInVisibleTerminal(command: string, cwd: string, label = 'MLSysBook command'): void {
  const terminal = getOrCreateTerminal(cwd);
  terminal.show(false);
  terminal.sendText(command);
  recordCommandStart(command, cwd, label, 'raw');
  storeLastCommand({
    command,
    cwd,
    label,
    timestamp: new Date().toISOString(),
  });
  getOutputChannel().appendLine(`[direct] ${label}`);
  getOutputChannel().appendLine(`Command: ${command}`);
}

export function getRecentCommandRuns(limit = 20): CommandRunRecord[] {
  return commandRuns.slice(0, limit);
}

export function recordExternalCommandStart(
  command: string,
  cwd: string,
  label: string,
  mode: 'raw' | 'focused' = 'raw',
): string {
  return recordCommandStart(command, cwd, label, mode).id;
}

export function recordExternalCommandFinish(runId: string, success: boolean): void {
  updateCommandStatus(runId, success ? 'succeeded' : 'failed');
}

export async function rerunSavedCommand(record: CommandRunRecord): Promise<void> {
  await runBookCommand(record.command, record.cwd, {
    mode: 'raw',
    label: `${record.label} (rerun)`,
  });
}

export function revealRunTerminal(cwd?: string): void {
  const fallbackCwd = cwd ?? getLastCommand()?.cwd ?? vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (!fallbackCwd) {
    vscode.window.showWarningMessage('MLSysBook: no workspace folder found for terminal.');
    return;
  }
  const terminal = getOrCreateTerminal(fallbackCwd);
  terminal.show(false);
}

export async function rerunLastCommand(forceRaw: boolean): Promise<void> {
  const last = getLastCommand();
  if (!last) {
    vscode.window.showWarningMessage('MLSysBook: no previous command to rerun.');
    return;
  }
  await runBookCommand(last.command, last.cwd, {
    mode: forceRaw ? 'raw' : undefined,
    label: `${last.label} (rerun)`,
  });
}

export function showLastFailureDetails(): void {
  const failure = getLastFailure();
  if (!failure) {
    vscode.window.showInformationMessage('MLSysBook: no recorded failure details.');
    return;
  }

  const channel = getOutputChannel();
  channel.appendLine('=== Last Failure Details ===');
  channel.appendLine(`Time: ${failure.timestamp}`);
  channel.appendLine(`Label: ${failure.label}`);
  channel.appendLine(`Stage: ${failure.stage}`);
  channel.appendLine(`Exit code: ${failure.exitCode}`);
  channel.appendLine(`Command: ${failure.command}`);
  channel.appendLine('--- Log tail ---');
  channel.appendLine(failure.logTail);
  channel.show(true);
}

