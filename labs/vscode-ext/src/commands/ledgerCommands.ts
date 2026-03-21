import * as vscode from 'vscode';
import { readLedger, resetLedger, getLedgerPath, ledgerExists } from '../utils/ledger';

export function registerLedgerCommands(context: vscode.ExtensionContext): void {
  // Show Design Ledger
  context.subscriptions.push(
    vscode.commands.registerCommand('labs.labShowLedger', () => {
      if (!ledgerExists()) {
        vscode.window.showInformationMessage(
          'Labs: No Design Ledger found. Complete a lab to create one.',
        );
        return;
      }

      const ledger = readLedger();
      if (!ledger) {
        vscode.window.showErrorMessage('Labs: could not read Design Ledger.');
        return;
      }

      const track = ledger.track ?? 'none';
      const chapter = ledger.current_chapter;
      const entries = ledger.history.length;

      const details = [
        `Track: ${track}`,
        `Current Chapter: ${chapter}`,
        `History Entries: ${entries}`,
        `Last Updated: ${ledger.last_updated || 'never'}`,
        '',
        `File: ${getLedgerPath()}`,
      ];

      if (entries > 0) {
        details.push('', '--- Recent Decisions ---');
        const recent = ledger.history.slice(-5);
        for (const entry of recent) {
          details.push(`  Chapter ${entry.chapter}: ${JSON.stringify(entry.design)}`);
        }
      }

      vscode.window.showInformationMessage(details.join('\n'), { modal: true });
    }),
  );

  // Reset Design Ledger
  context.subscriptions.push(
    vscode.commands.registerCommand('labs.labResetLedger', async () => {
      if (!ledgerExists()) {
        vscode.window.showInformationMessage('Labs: No Design Ledger to reset.');
        return;
      }

      const confirm = await vscode.window.showWarningMessage(
        'Reset the Design Ledger? This deletes all saved lab decisions.',
        { modal: true },
        'Reset',
      );

      if (confirm !== 'Reset') { return; }

      if (resetLedger()) {
        vscode.window.showInformationMessage('Labs: Design Ledger has been reset.');
      } else {
        vscode.window.showErrorMessage('Labs: failed to reset Design Ledger.');
      }
    }),
  );
}
