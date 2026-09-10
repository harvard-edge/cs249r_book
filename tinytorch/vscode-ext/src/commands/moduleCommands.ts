import * as vscode from 'vscode';
import * as fs from 'fs';
import { runInTerminal } from '../utils/terminal';
import { discoverModules } from '../utils/modules';
import { callTito, logError, titoTerminalCommand } from '../utils/tito';

/**
 * Get a file path from `tito module path <num> --<kind>`.
 * Returns the absolute path string, or null on error (logged to output channel).
 */
function getPathFromTito(projectRoot: string, num: string, kind: 'notebook' | 'source' | 'about'): string | null {
  return callTito(projectRoot, `module path ${num} --${kind}`, `module path (${kind})`);
}

/**
 * Register all module-related commands.
 *
 * Each command delegates to `tito` — the extension never reimplements
 * CLI logic; it just provides a GUI wrapper.
 *
 * Note: Module tree refreshes happen automatically via the .tito/progress.json
 * file watcher set up in extension.ts — no setTimeout refreshes needed.
 */
export function registerModuleCommands(
  context: vscode.ExtensionContext,
  projectRoot: string,
): void {

  // Helper: pick a module if not provided as an argument
  async function pickModule(argNumber?: string): Promise<string | undefined> {
    if (argNumber) { return argNumber; }
    const { modules, error } = discoverModules(projectRoot);
    if (error) {
      vscode.window.showWarningMessage(`TinyTorch: ${error}`);
      return undefined;
    }
    if (modules.length === 0) {
      vscode.window.showInformationMessage('No modules found. Run "tito setup" first.');
      return undefined;
    }
    const picks = modules.map(m => ({
      label: `${m.number} — ${m.title ?? m.displayName}`,
      description: m.status.replace('_', ' '),
      number: m.number,
    }));
    const selected = await vscode.window.showQuickPick(picks, {
      placeHolder: 'Select a module',
    });
    return selected?.number;
  }

  /** Open a file returned by Tito, with a user-facing message on failure */
  async function openFileFromTito(
    moduleNum: string,
    kind: 'notebook' | 'source' | 'about',
    fallbackAction?: () => void,
  ): Promise<void> {
    const filePath = getPathFromTito(projectRoot, moduleNum, kind);

    if (filePath && fs.existsSync(filePath)) {
      const uri = vscode.Uri.file(filePath);
      if (kind === 'notebook') {
        await vscode.commands.executeCommand('vscode.openWith', uri, 'jupyter-notebook');
      } else {
        const doc = await vscode.workspace.openTextDocument(uri);
        await vscode.window.showTextDocument(doc, { preview: false });
      }
      return;
    }

    // File not found — give context-specific feedback
    const kindLabel = kind === 'notebook' ? 'Notebook' : kind === 'source' ? 'Source file' : 'ABOUT.md';
    if (fallbackAction) {
      const action = await vscode.window.showWarningMessage(
        `${kindLabel} not found for module ${moduleNum}.`,
        'Start Module (generates files)',
      );
      if (action) { fallbackAction(); }
    } else {
      vscode.window.showWarningMessage(`${kindLabel} not found for module ${moduleNum}.`);
    }
  }

  context.subscriptions.push(
    // Open the .ipynb notebook directly in VS Code's notebook editor
    vscode.commands.registerCommand('tinytorch.openNotebook', async (num?: string) => {
      const moduleNum = await pickModule(num);
      if (!moduleNum) { return; }

      await openFileFromTito(moduleNum, 'notebook', () => {
        runInTerminal(
          titoTerminalCommand(`module start ${moduleNum}`),
          projectRoot,
          `Start Module ${moduleNum}`,
        );
      });
    }),

    // Open the source .py file
    vscode.commands.registerCommand('tinytorch.openSource', async (num?: string) => {
      const moduleNum = await pickModule(num);
      if (!moduleNum) { return; }
      await openFileFromTito(moduleNum, 'source');
    }),

    vscode.commands.registerCommand('tinytorch.startModule', async (num?: string) => {
      const moduleNum = await pickModule(num);
      if (!moduleNum) { return; }
      runInTerminal(
        titoTerminalCommand(`module start ${moduleNum}`),
        projectRoot,
        `Start Module ${moduleNum}`,
      );
    }),

    vscode.commands.registerCommand('tinytorch.viewModule', async (num?: string) => {
      const moduleNum = await pickModule(num);
      if (!moduleNum) { return; }
      runInTerminal(
        titoTerminalCommand(`module view ${moduleNum}`),
        projectRoot,
        `View Module ${moduleNum}`,
      );
    }),

    vscode.commands.registerCommand('tinytorch.testModule', async (num?: string) => {
      const moduleNum = await pickModule(num);
      if (!moduleNum) { return; }
      runInTerminal(
        titoTerminalCommand(`module test ${moduleNum}`),
        projectRoot,
        `Test Module ${moduleNum}`,
      );
    }),

    vscode.commands.registerCommand('tinytorch.completeModule', async (num?: string) => {
      const moduleNum = await pickModule(num);
      if (!moduleNum) { return; }
      runInTerminal(
        titoTerminalCommand(`module complete ${moduleNum}`),
        projectRoot,
        `Complete Module ${moduleNum}`,
      );
    }),

    vscode.commands.registerCommand('tinytorch.resetModule', async (num?: string) => {
      const moduleNum = await pickModule(num);
      if (!moduleNum) { return; }
      const confirm = await vscode.window.showWarningMessage(
        `Reset Module ${moduleNum}? This will clear your work.`,
        { modal: true },
        'Reset',
      );
      if (confirm !== 'Reset') { return; }
      runInTerminal(
        titoTerminalCommand(`module reset ${moduleNum}`),
        projectRoot,
        `Reset Module ${moduleNum}`,
      );
    }),

    vscode.commands.registerCommand('tinytorch.openAbout', async (num?: string) => {
      const moduleNum = await pickModule(num);
      if (!moduleNum) { return; }
      await openFileFromTito(moduleNum, 'about');
    }),
  );
}
