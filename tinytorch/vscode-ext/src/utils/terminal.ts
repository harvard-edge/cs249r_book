import * as vscode from 'vscode';
import { CommandRunRecord } from '../types';

const TERMINAL_NAME = 'TinyTorch';
const OUTPUT_CHANNEL_NAME = 'TinyTorch';
const STATE_HISTORY_KEY = 'tinytorch.commandHistory';
const MAX_HISTORY = 30;

let extensionContext: vscode.ExtensionContext | undefined;
let outputChannel: vscode.OutputChannel | undefined;
const commandRuns: CommandRunRecord[] = [];
const commandRunsEmitter = new vscode.EventEmitter<void>();
export const onCommandRunsChanged = commandRunsEmitter.event;

function getOutputChannel(): vscode.OutputChannel {
  if (!outputChannel) {
    outputChannel = vscode.window.createOutputChannel(OUTPUT_CHANNEL_NAME);
  }
  return outputChannel;
}

function getOrCreateTerminal(cwd: string): vscode.Terminal {
  let terminal = vscode.window.terminals.find(t => t.name === TERMINAL_NAME);
  if (!terminal) {
    terminal = vscode.window.createTerminal({ name: TERMINAL_NAME, cwd });
  }
  return terminal;
}

/** Initialize the run manager — call once from activate() */
export function initializeRunManager(context: vscode.ExtensionContext): void {
  extensionContext = context;
  const saved = context.workspaceState.get<CommandRunRecord[]>(STATE_HISTORY_KEY, []);
  if (Array.isArray(saved)) {
    commandRuns.splice(0, commandRuns.length, ...saved.slice(0, MAX_HISTORY));
  }
  context.subscriptions.push(commandRunsEmitter);
}

function persist(): void {
  void extensionContext?.workspaceState.update(STATE_HISTORY_KEY, commandRuns);
}

function recordStart(command: string, cwd: string, label: string): CommandRunRecord {
  const record: CommandRunRecord = {
    id: `cmd-${Date.now()}`,
    label,
    command,
    cwd,
    timestamp: new Date().toISOString(),
    status: 'started',
  };
  commandRuns.unshift(record);
  if (commandRuns.length > MAX_HISTORY) { commandRuns.length = MAX_HISTORY; }
  persist();
  commandRunsEmitter.fire();
  return record;
}

/** Run a command in the dedicated TinyTorch terminal */
export function runInTerminal(command: string, cwd: string, label: string): void {
  const terminal = getOrCreateTerminal(cwd);
  terminal.show(false);
  // Always reset to project root before running, since the terminal persists cwd
  terminal.sendText(`cd "${cwd}" && ${command}`);

  recordStart(command, cwd, label);
  const channel = getOutputChannel();
  channel.appendLine(`[${new Date().toLocaleTimeString()}] ${label}`);
  channel.appendLine(`  $ ${command}`);
}

/** Get recent command runs for the history view */
export function getRecentRuns(limit = 20): CommandRunRecord[] {
  return commandRuns.slice(0, limit);
}

/** Rerun a previously recorded command */
export function rerunCommand(record: CommandRunRecord): void {
  runInTerminal(record.command, record.cwd, `${record.label} (rerun)`);
}

/** Rerun the most recent command */
export function rerunLastCommand(): void {
  const last = commandRuns[0];
  if (!last) {
    vscode.window.showWarningMessage('TinyTorch: no previous command to rerun.');
    return;
  }
  rerunCommand(last);
}

/** Show the TinyTorch terminal */
export function revealTerminal(cwd: string): void {
  const terminal = getOrCreateTerminal(cwd);
  terminal.show(false);
}
