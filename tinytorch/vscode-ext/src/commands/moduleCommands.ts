import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { runInTerminal } from '../utils/terminal';
import { discoverModules } from '../utils/modules';

/**
 * Register all module-related commands.
 *
 * Each command shells out to `tito` — the extension never reimplements
 * CLI logic; it just provides a GUI wrapper.
 */
export function registerModuleCommands(
  context: vscode.ExtensionContext,
  projectRoot: string,
  refreshModules: () => void,
): void {

  // Helper: pick a module if not provided as an argument
  async function pickModule(argNumber?: string): Promise<string | undefined> {
    if (argNumber) { return argNumber; }
    const modules = discoverModules(projectRoot);
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

  // Helper: resolve a module's folder from number
  function resolveFolder(num: string, folder?: string): string | undefined {
    if (folder) { return folder; }
    const modules = discoverModules(projectRoot);
    return modules.find(m => m.number === num)?.folder;
  }

  context.subscriptions.push(
    // Open the .ipynb notebook directly in VS Code's notebook editor
    vscode.commands.registerCommand('tinytorch.openNotebook', async (num?: string, folder?: string) => {
      const moduleNum = await pickModule(num);
      if (!moduleNum) { return; }
      const moduleFolder = resolveFolder(moduleNum, folder);
      if (!moduleFolder) { return; }

      // Notebooks live in modules/<folder>/<slug>.ipynb
      const slug = moduleFolder.replace(/^\d{2}_/, '');
      const nbPath = path.join(projectRoot, 'modules', moduleFolder, `${slug}.ipynb`);

      if (fs.existsSync(nbPath)) {
        const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(nbPath));
        await vscode.window.showTextDocument(doc, { preview: false });
      } else {
        // Notebook might not exist yet — offer to generate it
        const action = await vscode.window.showWarningMessage(
          `Notebook not found: modules/${moduleFolder}/${slug}.ipynb`,
          'Start Module (generates notebook)',
        );
        if (action) {
          runInTerminal(
            `python3 -m tito.main module start ${moduleNum}`,
            projectRoot,
            `Start Module ${moduleNum}`,
          );
        }
      }
    }),

    // Open the source .py file
    vscode.commands.registerCommand('tinytorch.openSource', async (num?: string, folder?: string) => {
      const moduleNum = await pickModule(num);
      if (!moduleNum) { return; }
      const moduleFolder = resolveFolder(moduleNum, folder);
      if (!moduleFolder) { return; }

      const pyPath = path.join(projectRoot, 'src', moduleFolder, `${moduleFolder}.py`);
      if (fs.existsSync(pyPath)) {
        const doc = await vscode.workspace.openTextDocument(pyPath);
        await vscode.window.showTextDocument(doc, { preview: false });
      } else {
        vscode.window.showWarningMessage(`Source file not found: src/${moduleFolder}/${moduleFolder}.py`);
      }
    }),

    vscode.commands.registerCommand('tinytorch.startModule', async (num?: string) => {
      const moduleNum = await pickModule(num);
      if (!moduleNum) { return; }
      runInTerminal(
        `python3 -m tito.main module start ${moduleNum}`,
        projectRoot,
        `Start Module ${moduleNum}`,
      );
      setTimeout(refreshModules, 3000);
    }),

    vscode.commands.registerCommand('tinytorch.viewModule', async (num?: string) => {
      const moduleNum = await pickModule(num);
      if (!moduleNum) { return; }
      runInTerminal(
        `python3 -m tito.main module view ${moduleNum}`,
        projectRoot,
        `View Module ${moduleNum}`,
      );
    }),

    vscode.commands.registerCommand('tinytorch.testModule', async (num?: string) => {
      const moduleNum = await pickModule(num);
      if (!moduleNum) { return; }
      runInTerminal(
        `python3 -m tito.main module test ${moduleNum}`,
        projectRoot,
        `Test Module ${moduleNum}`,
      );
    }),

    vscode.commands.registerCommand('tinytorch.completeModule', async (num?: string) => {
      const moduleNum = await pickModule(num);
      if (!moduleNum) { return; }
      runInTerminal(
        `python3 -m tito.main module complete ${moduleNum}`,
        projectRoot,
        `Complete Module ${moduleNum}`,
      );
      setTimeout(refreshModules, 5000);
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
        `python3 -m tito.main module reset ${moduleNum}`,
        projectRoot,
        `Reset Module ${moduleNum}`,
      );
      setTimeout(refreshModules, 3000);
    }),

    vscode.commands.registerCommand('tinytorch.openAbout', async (num?: string, folder?: string) => {
      const moduleNum = await pickModule(num);
      if (!moduleNum) { return; }
      const moduleFolder = resolveFolder(moduleNum, folder);
      if (!moduleFolder) { return; }
      const aboutPath = path.join(projectRoot, 'src', moduleFolder, 'ABOUT.md');
      try {
        const doc = await vscode.workspace.openTextDocument(aboutPath);
        await vscode.window.showTextDocument(doc, { preview: false });
      } catch {
        vscode.window.showWarningMessage(`ABOUT.md not found for module ${moduleNum}`);
      }
    }),
  );
}
