import * as vscode from 'vscode';
import * as path from 'path';
import { PRECOMMIT_QMD_FILE_FIXERS } from '../constants';
import { getRepoRoot } from '../utils/workspace';
import { runBookCommand } from '../utils/terminal';

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

export function registerPrecommitCommands(context: vscode.ExtensionContext): void {
  const root = getRepoRoot();
  if (!root) { return; }

  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.precommitRunAll', () => {
      void runBookCommand('pre-commit run --all-files', root, {
        label: 'Pre-commit (all hooks)',
      });
    }),

    vscode.commands.registerCommand('mlsysbook.precommitRunHook', (command: string) => {
      void runBookCommand(command, root, {
        label: 'Pre-commit (selected hook)',
      });
    }),

    vscode.commands.registerCommand('mlsysbook.precommitRunFixersCurrentFile', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage('MLSysBook: open a .qmd file to run current-file fixers.');
        return;
      }

      const filePath = editor.document.uri.fsPath;
      if (!filePath.endsWith('.qmd')) {
        vscode.window.showWarningMessage('MLSysBook: current-file fixers only support .qmd files.');
        return;
      }

      const relativePath = path.relative(root, filePath);
      if (!relativePath || relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
        vscode.window.showWarningMessage('MLSysBook: active file must be inside this repository.');
        return;
      }

      const normalizedRelativePath = relativePath.split(path.sep).join('/');
      for (const fixer of PRECOMMIT_QMD_FILE_FIXERS) {
        const command = `pre-commit run ${fixer.hookId} --files ${shellQuote(normalizedRelativePath)}`;
        await runBookCommand(command, root, {
          label: `Current-file fixer: ${fixer.label}`,
        });
      }
    }),

    vscode.commands.registerCommand('mlsysbook.validateRunAction', (command: string) => {
      void runBookCommand(command, root, {
        label: 'Binder validate (selected action)',
      });
    }),
  );
}
