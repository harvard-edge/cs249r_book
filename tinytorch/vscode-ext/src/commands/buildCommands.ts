import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { runInTerminal } from '../utils/terminal';

/** Register build/site/paper command palette entries and tree actions */
export function registerBuildCommands(
  context: vscode.ExtensionContext,
  projectRoot: string,
): void {
  // Run a command from the site/ directory
  context.subscriptions.push(
    vscode.commands.registerCommand('tinytorch.runSiteAction', (command: string, label: string) => {
      runInTerminal(command, projectRoot, label);
    }),
  );

  // Run a command from the paper/ directory
  context.subscriptions.push(
    vscode.commands.registerCommand('tinytorch.runPaperAction', (command: string, label: string) => {
      runInTerminal(command, projectRoot, label);
    }),
  );

  // Build HTML site (command palette)
  context.subscriptions.push(
    vscode.commands.registerCommand('tinytorch.buildSite', () => {
      runInTerminal('cd site && make html', projectRoot, 'Build HTML Site');
    }),
  );

  // Build PDF course guide (command palette)
  context.subscriptions.push(
    vscode.commands.registerCommand('tinytorch.buildPdf', () => {
      runInTerminal('cd site && make pdf', projectRoot, 'Build PDF');
    }),
  );

  // Build research paper (command palette)
  context.subscriptions.push(
    vscode.commands.registerCommand('tinytorch.buildPaper', () => {
      runInTerminal('cd site && make paper', projectRoot, 'Build Paper');
    }),
  );

  // Open site in browser
  context.subscriptions.push(
    vscode.commands.registerCommand('tinytorch.openSiteInBrowser', () => {
      const indexPath = path.join(projectRoot, 'site', '_build', 'html', 'index.html');
      if (fs.existsSync(indexPath)) {
        void vscode.env.openExternal(vscode.Uri.file(indexPath));
      } else {
        vscode.window.showWarningMessage(
          'Site not built yet. Run "Build HTML" first.',
        );
      }
    }),
  );

  // Open a PDF file (relative to project root)
  context.subscriptions.push(
    vscode.commands.registerCommand('tinytorch.openPdf', (relativePath: string) => {
      const pdfPath = path.join(projectRoot, relativePath);
      if (fs.existsSync(pdfPath)) {
        void vscode.env.openExternal(vscode.Uri.file(pdfPath));
      } else {
        vscode.window.showWarningMessage(
          `PDF not found: ${relativePath}. Build it first.`,
        );
      }
    }),
  );

  // Health check (command palette)
  context.subscriptions.push(
    vscode.commands.registerCommand('tinytorch.healthCheck', () => {
      runInTerminal('python3 -m tito.main system health', projectRoot, 'Health Check');
    }),
  );
}
