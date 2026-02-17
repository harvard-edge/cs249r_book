import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { runInTerminal } from '../utils/terminal';
import { titoTerminalCommand } from '../utils/tito';

/** Build output paths (relative to project root) */
const BUILD_OUTPUTS = {
  siteIndex: path.join('site', '_build', 'html', 'index.html'),
  coursePdf: path.join('site', '_build', 'latex', 'tinytorch-course.pdf'),
  paperPdf: path.join('paper', 'paper.pdf'),
} as const;

/** Register build/site/paper command palette entries and tree actions */
export function registerBuildCommands(
  context: vscode.ExtensionContext,
  projectRoot: string,
): void {
  // Generic action runner (used by tree items that pass a command string)
  context.subscriptions.push(
    vscode.commands.registerCommand('tinytorch.runSiteAction', (command: string, label: string) => {
      runInTerminal(command, projectRoot, label);
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('tinytorch.runPaperAction', (command: string, label: string) => {
      runInTerminal(command, projectRoot, label);
    }),
  );

  // Build HTML site (command palette) — delegates to tito dev build
  context.subscriptions.push(
    vscode.commands.registerCommand('tinytorch.buildSite', () => {
      runInTerminal(titoTerminalCommand('dev build html'), projectRoot, 'Build HTML Site');
    }),
  );

  // Build PDF course guide (command palette)
  context.subscriptions.push(
    vscode.commands.registerCommand('tinytorch.buildPdf', () => {
      runInTerminal(titoTerminalCommand('dev build pdf'), projectRoot, 'Build PDF');
    }),
  );

  // Build research paper (command palette)
  context.subscriptions.push(
    vscode.commands.registerCommand('tinytorch.buildPaper', () => {
      runInTerminal(titoTerminalCommand('dev build paper'), projectRoot, 'Build Paper');
    }),
  );

  // Open site in browser (VS Code UI action — stays in extension)
  context.subscriptions.push(
    vscode.commands.registerCommand('tinytorch.openSiteInBrowser', () => {
      const indexPath = path.join(projectRoot, BUILD_OUTPUTS.siteIndex);
      if (fs.existsSync(indexPath)) {
        void vscode.env.openExternal(vscode.Uri.file(indexPath));
      } else {
        void vscode.window.showWarningMessage(
          'Site not built yet. Run "Build HTML" first.',
          'Build Now',
        ).then(choice => {
          if (choice === 'Build Now') {
            runInTerminal(titoTerminalCommand('dev build html'), projectRoot, 'Build HTML Site');
          }
        });
      }
    }),
  );

  // Open a PDF file (VS Code UI action — stays in extension)
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
      runInTerminal(titoTerminalCommand('system health'), projectRoot, 'Health Check');
    }),
  );
}

/** Exported for use by buildTreeProvider */
export { BUILD_OUTPUTS };
