import * as vscode from 'vscode';

/** Zone boundaries based on the 22-cell TEMPLATE.md standard */
const ZONES: Array<{ name: string; startCell: number; endCell: number; detail: string }> = [
  { name: 'Zone A: Opening', startCell: 0, endCell: 4, detail: 'Setup, Header, Stakeholder, Context, Prediction' },
  { name: 'Zone B: Act I — Calibration', startCell: 5, endCell: 11, detail: 'Instruments, Exploration, Reflection' },
  { name: 'Zone C: Act II — Design Challenge', startCell: 12, endCell: 19, detail: 'Full Instruments, Failure States, Synthesis' },
  { name: 'Zone D: Closing', startCell: 20, endCell: 21, detail: 'Summary, Ledger Save' },
];

/**
 * DocumentSymbolProvider for Marimo lab files.
 *
 * Parses @app.cell decorators and groups them into the 22-cell
 * template zones (A/B/C/D) for outline navigation.
 */
export class MarimoSymbolProvider implements vscode.DocumentSymbolProvider {
  provideDocumentSymbols(
    document: vscode.TextDocument,
    _token: vscode.CancellationToken,
  ): vscode.DocumentSymbol[] {
    // Only process lab files
    const filePath = document.uri.fsPath;
    if (!filePath.includes('/labs/vol') || !filePath.endsWith('.py')) {
      return [];
    }

    const text = document.getText();
    const lines = text.split('\n');
    const cells: Array<{ line: number; name: string; range: vscode.Range }> = [];

    // Find all @app.cell decorators
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (line.startsWith('@app.cell')) {
        // Look for the cell comment above (e.g. "# ─── CELL 1: HEADER")
        let cellName = `Cell ${cells.length}`;
        for (let j = i - 1; j >= Math.max(0, i - 3); j--) {
          const comment = lines[j].trim();
          const match = comment.match(/^#\s*[─═]+\s*CELL\s+(\d+):\s*(.+?)(?:\s*[─═]+)?$/i);
          if (match) {
            cellName = `Cell ${match[1]}: ${match[2]}`;
            break;
          }
          // Also match zone headers
          const zoneMatch = comment.match(/^#\s*ZONE\s+([A-D]):\s*(.+)/i);
          if (zoneMatch) {
            cellName = `Zone ${zoneMatch[1]}: ${zoneMatch[2]}`;
            break;
          }
        }

        // Find the end of this cell (next @app.cell or end of file)
        let endLine = lines.length - 1;
        for (let k = i + 1; k < lines.length; k++) {
          if (lines[k].trim().startsWith('@app.cell')) {
            endLine = k - 1;
            break;
          }
        }

        const range = new vscode.Range(i, 0, endLine, lines[endLine]?.length ?? 0);
        cells.push({ line: i, name: cellName, range });
      }
    }

    // Group cells into zones
    const symbols: vscode.DocumentSymbol[] = [];

    for (const zone of ZONES) {
      const zoneCells = cells.filter((_c, idx) =>
        idx >= zone.startCell && idx <= zone.endCell
      );

      if (zoneCells.length === 0) { continue; }

      const firstCell = zoneCells[0];
      const lastCell = zoneCells[zoneCells.length - 1];
      const zoneRange = new vscode.Range(
        firstCell.range.start,
        lastCell.range.end,
      );

      const zoneSymbol = new vscode.DocumentSymbol(
        zone.name,
        zone.detail,
        vscode.SymbolKind.Namespace,
        zoneRange,
        firstCell.range,
      );

      // Add individual cells as children
      for (const cell of zoneCells) {
        zoneSymbol.children.push(
          new vscode.DocumentSymbol(
            cell.name,
            '',
            vscode.SymbolKind.Function,
            cell.range,
            cell.range,
          ),
        );
      }

      symbols.push(zoneSymbol);
    }

    // If no zones matched (fewer cells than expected), just list cells flat
    if (symbols.length === 0) {
      for (const cell of cells) {
        symbols.push(
          new vscode.DocumentSymbol(
            cell.name,
            '',
            vscode.SymbolKind.Function,
            cell.range,
            cell.range,
          ),
        );
      }
    }

    return symbols;
  }
}
