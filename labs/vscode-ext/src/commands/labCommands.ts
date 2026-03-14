import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { runInTerminal } from '../utils/terminal';
import { marimoCommand } from '../utils/marimoCli';
import { discoverAllLabs } from '../utils/labDiscovery';
import { LabInfo } from '../types';

export function registerLabCommands(context: vscode.ExtensionContext, root: string): void {
  // Open lab file in editor
  context.subscriptions.push(
    vscode.commands.registerCommand('labs.labOpen', async () => {
      const lab = await pickLab(root, 'Select lab to open');
      if (!lab) { return; }
      const filePath = lab.labPath ?? lab.planPath;
      if (filePath) {
        await vscode.window.showTextDocument(vscode.Uri.file(filePath));
      }
    }),
  );

  // Edit in Marimo (launches browser)
  context.subscriptions.push(
    vscode.commands.registerCommand('labs.labEdit', async (resource?: vscode.Uri) => {
      const filePath = resource?.fsPath ?? await pickLabPath(root, 'Select lab to edit in Marimo');
      if (!filePath) { return; }
      const cmd = marimoCommand('edit', filePath);
      runInTerminal(cmd, root, `Marimo Edit: ${path.basename(filePath)}`);
    }),
  );

  // Run in Marimo (read-only mode)
  context.subscriptions.push(
    vscode.commands.registerCommand('labs.labRun', async (resource?: vscode.Uri) => {
      const filePath = resource?.fsPath ?? await pickLabPath(root, 'Select lab to run');
      if (!filePath) { return; }
      const cmd = marimoCommand('run', filePath);
      runInTerminal(cmd, root, `Marimo Run: ${path.basename(filePath)}`);
    }),
  );

  // Open lab plan
  context.subscriptions.push(
    vscode.commands.registerCommand('labs.labOpenPlan', async () => {
      const lab = await pickLab(root, 'Select lab plan to open');
      if (!lab?.planPath) {
        vscode.window.showWarningMessage('Labs: no plan file found for this lab.');
        return;
      }
      await vscode.window.showTextDocument(vscode.Uri.file(lab.planPath));
    }),
  );

  // New lab from template
  context.subscriptions.push(
    vscode.commands.registerCommand('labs.labNewFromTemplate', async () => {
      const volume = await vscode.window.showQuickPick(['Volume 1', 'Volume 2'], {
        placeHolder: 'Select volume',
      });
      if (!volume) { return; }

      const volNum = volume === 'Volume 1' ? 1 : 2;
      const labNum = await vscode.window.showInputBox({
        prompt: 'Lab number (e.g. 05)',
        placeHolder: '05',
      });
      if (!labNum) { return; }

      const labName = await vscode.window.showInputBox({
        prompt: 'Lab slug (e.g. nn_compute)',
        placeHolder: 'lab_topic',
      });
      if (!labName) { return; }

      const filename = `lab_${labNum.padStart(2, '0')}_${labName}.py`;
      const targetDir = path.join(root, 'labs', `vol${volNum}`);
      const targetPath = path.join(targetDir, filename);

      if (fs.existsSync(targetPath)) {
        vscode.window.showWarningMessage(`Labs: ${filename} already exists.`);
        return;
      }

      // Read template
      const templatePath = path.join(root, 'labs', 'TEMPLATE.md');
      const templateContent = fs.existsSync(templatePath)
        ? `# Generated from TEMPLATE.md — see labs/TEMPLATE.md for the 22-cell structure\n`
        : '';

      const stub = `import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

${templateContent}
# ─────────────────────────────────────────────────────────────────────────────
# LAB ${labNum.padStart(2, '0')}: ${labName.toUpperCase().replace(/_/g, ' ')}
#
# Chapter: @sec-TODO
# Core Invariant: TODO
#
# 2-Act Structure (35-40 minutes):
#   Act I  — Calibration (12-15 min)
#   Act II — Design Challenge (20-25 min)
# ─────────────────────────────────────────────────────────────────────────────


# ─── CELL 0: SETUP ──────────────────────────────────────────────────────────
@app.cell
def _():
    import marimo as mo
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from labs.core.state import DesignLedger
    from labs.core.style import COLORS, LAB_CSS, apply_plotly_theme

    ledger = DesignLedger()
    return COLORS, LAB_CSS, DesignLedger, apply_plotly_theme, ledger, mo


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 1: HEADER ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("# TODO: Lab Header")
    return


if __name__ == "__main__":
    app.run()
`;

      fs.writeFileSync(targetPath, stub, 'utf-8');
      await vscode.window.showTextDocument(vscode.Uri.file(targetPath));
      vscode.window.showInformationMessage(`Labs: created ${filename}`);
    }),
  );

  // Show lab mapping
  context.subscriptions.push(
    vscode.commands.registerCommand('labs.labShowMapping', async () => {
      const archPath = path.join(root, 'labs', 'ARCHITECTURE.md');
      if (fs.existsSync(archPath)) {
        await vscode.window.showTextDocument(vscode.Uri.file(archPath));
      } else {
        vscode.window.showWarningMessage('Labs: ARCHITECTURE.md not found.');
      }
    }),
  );
}

/** Show a QuickPick to select a lab */
async function pickLab(root: string, placeholder: string): Promise<LabInfo | undefined> {
  const { vol1, vol2 } = discoverAllLabs(root);
  const allLabs = [...vol1.map(l => ({ ...l, volLabel: 'V1' })), ...vol2.map(l => ({ ...l, volLabel: 'V2' }))];

  const items = allLabs.map(l => ({
    label: `${l.volLabel} ${l.number} — ${l.title}`,
    description: l.status,
    lab: l as LabInfo,
  }));

  const pick = await vscode.window.showQuickPick(items, { placeHolder: placeholder });
  return pick?.lab;
}

/** Pick a lab and return its .py file path */
async function pickLabPath(root: string, placeholder: string): Promise<string | undefined> {
  const lab = await pickLab(root, placeholder);
  return lab?.labPath;
}
