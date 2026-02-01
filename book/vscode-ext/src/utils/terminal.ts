import * as vscode from 'vscode';

const TERMINAL_NAME = 'MLSysBook';

export function runInTerminal(command: string, cwd: string): void {
  let terminal = vscode.window.terminals.find(t => t.name === TERMINAL_NAME);
  if (!terminal) {
    terminal = vscode.window.createTerminal({ name: TERMINAL_NAME, cwd });
  }
  terminal.show(false);
  terminal.sendText(command);
}
