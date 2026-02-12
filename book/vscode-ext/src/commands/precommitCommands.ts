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

  const runHookCurrentFile = async (hookId: string, label: string): Promise<void> => {
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
    if (editor.document.isDirty) {
      const saved = await editor.document.save();
      if (!saved) {
        vscode.window.showWarningMessage('MLSysBook: save the file before running current-file fixers.');
        return;
      }
    }

    const relativePath = path.relative(root, filePath);
    if (!relativePath || relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
      vscode.window.showWarningMessage('MLSysBook: active file must be inside this repository.');
      return;
    }
    const normalizedRelativePath = relativePath.split(path.sep).join('/');
    const command = `pre-commit run ${hookId} --files ${shellQuote(normalizedRelativePath)}`;
    await runBookCommand(command, root, { label });
  };

  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.precommitRunAll', () => {
      void vscode.workspace.saveAll(false).then(() => runBookCommand('pre-commit run --all-files', root, {
        label: 'Pre-commit (all hooks)',
      }));
    }),

    vscode.commands.registerCommand('mlsysbook.precommitRunHook', (command: string) => {
      void vscode.workspace.saveAll(false).then(() => runBookCommand(command, root, {
        label: 'Pre-commit (selected hook)',
      }));
    }),

    vscode.commands.registerCommand('mlsysbook.precommitRunHookCurrentFile', (hookId: string, label: string) => {
      void runHookCurrentFile(hookId, label);
    }),

    vscode.commands.registerCommand('mlsysbook.precommitRunFixersCurrentFile', async () => {
      for (const fixer of PRECOMMIT_QMD_FILE_FIXERS) {
        await runHookCurrentFile(fixer.hookId, `Current-file fixer: ${fixer.label}`);
      }
    }),

    vscode.commands.registerCommand('mlsysbook.validateRunAction', (command: string) => {
      void runBookCommand(command, root, {
        label: 'Binder validate (selected action)',
      });
    }),
  );
}
