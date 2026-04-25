import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { runInTerminal } from '../utils/terminal';
import { discoverAllLabs } from '../utils/labDiscovery';
import { LabInfo } from '../types';

export function registerAuditCommands(context: vscode.ExtensionContext, root: string): void {
  const pythonPath = vscode.workspace.getConfiguration('labs').get<string>('pythonPath', 'python3');

  // Audit a single lab (30-gate review)
  context.subscriptions.push(
    vscode.commands.registerCommand('labs.labAudit', async (resource?: vscode.Uri) => {
      let filePath = resource?.fsPath;

      if (!filePath) {
        // Pick from implemented labs
        const lab = await pickImplementedLab(root, 'Select lab to audit');
        filePath = lab?.labPath;
      }

      if (!filePath) { return; }

      const labName = path.basename(filePath, '.py');
      vscode.window.showInformationMessage(
        `Labs: Opening ${labName} with the protocol checklist. ` +
        `Run the pytest lab checks for automated validation.`
      );

      // Open the available protocol doc alongside the lab for manual audit.
      const reviewPath = path.join(root, 'labs', 'PROTOCOL.md');
      if (fs.existsSync(reviewPath)) {
        await vscode.window.showTextDocument(
          vscode.Uri.file(reviewPath),
          { viewColumn: vscode.ViewColumn.Beside, preview: true },
        );
      }
      await vscode.window.showTextDocument(
        vscode.Uri.file(filePath),
        { viewColumn: vscode.ViewColumn.One },
      );
    }),
  );

  // Check template compliance (22-cell structure)
  context.subscriptions.push(
    vscode.commands.registerCommand('labs.labCheckTemplate', async () => {
      const lab = await pickImplementedLab(root, 'Select lab to check template compliance');
      if (!lab?.labPath) { return; }

      // Quick compliance check: count @app.cell decorators
      const content = fs.readFileSync(lab.labPath, 'utf-8');
      const cellCount = (content.match(/@app\.cell/g) || []).length;
      const hasZoneA = content.includes('ZONE A');
      const hasZoneB = content.includes('ZONE B') || content.includes('Act I') || content.includes('ACT I');
      const hasZoneC = content.includes('ZONE C') || content.includes('Act II') || content.includes('ACT II');
      const hasZoneD = content.includes('ZONE D') || content.includes('CLOSING');
      const hasHideCode = (content.match(/hide_code=True/g) || []).length;
      const hasPredictionLock = content.includes('mo.stop') || content.includes('mo.ui.radio');
      const hasLedger = content.includes('DesignLedger') || content.includes('ledger');

      const results = [
        `Cells: ${cellCount}/22 (target: 22)`,
        `Zone A (Opening): ${hasZoneA ? 'YES' : 'NO'}`,
        `Zone B (Act I): ${hasZoneB ? 'YES' : 'NO'}`,
        `Zone C (Act II): ${hasZoneC ? 'YES' : 'NO'}`,
        `Zone D (Closing): ${hasZoneD ? 'YES' : 'NO'}`,
        `hide_code=True: ${hasHideCode} cells`,
        `Prediction Lock: ${hasPredictionLock ? 'YES' : 'NO'}`,
        `Design Ledger: ${hasLedger ? 'YES' : 'NO'}`,
      ];

      const passing = [hasZoneA, hasZoneB, hasZoneC, hasPredictionLock, hasLedger].filter(Boolean).length;
      const total = 5;

      vscode.window.showInformationMessage(
        `Template Check: ${lab.slug} — ${passing}/${total} checks passed\n${results.join(' | ')}`,
        { modal: true },
      );
    }),
  );

  // Audit all implemented labs
  context.subscriptions.push(
    vscode.commands.registerCommand('labs.labAuditAll', () => {
      const { vol1, vol2 } = discoverAllLabs(root);
      const implemented = [...vol1, ...vol2].filter(l => l.status === 'implemented');

      if (implemented.length === 0) {
        vscode.window.showWarningMessage('Labs: no implemented labs found to audit.');
        return;
      }

      vscode.window.showInformationMessage(
        `Labs: ${implemented.length} implemented labs found. ` +
        `Use the lab-designer agent for batch auditing.`
      );
    }),
  );

  // Health check
  context.subscriptions.push(
    vscode.commands.registerCommand('labs.healthCheck', () => {
      runInTerminal(
        `${pythonPath} -c "from mlsysim.labs.state import DesignLedger; from mlsysim.labs.style import COLORS; print('Core modules OK')"`,
        root,
        'Labs Health Check',
      );
    }),
  );
}

/** Pick from implemented labs only */
async function pickImplementedLab(root: string, placeholder: string): Promise<LabInfo | undefined> {
  const { vol1, vol2 } = discoverAllLabs(root);
  const implemented = [
    ...vol1.filter(l => l.status === 'implemented').map(l => ({ ...l, volLabel: 'V1' })),
    ...vol2.filter(l => l.status === 'implemented').map(l => ({ ...l, volLabel: 'V2' })),
  ];

  if (implemented.length === 0) {
    vscode.window.showWarningMessage('Labs: no implemented labs found.');
    return undefined;
  }

  const items = implemented.map(l => ({
    label: `${l.volLabel} ${l.number} — ${l.title}`,
    lab: l as LabInfo,
  }));

  const pick = await vscode.window.showQuickPick(items, { placeHolder: placeholder });
  return pick?.lab;
}
