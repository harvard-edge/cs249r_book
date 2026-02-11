import * as vscode from 'vscode';

const DIAGNOSTIC_SOURCE = 'mlsysbook-qmd';

function isQmdDocument(document: vscode.TextDocument): boolean {
  return document.uri.fsPath.endsWith('.qmd');
}

function collectDefinedLabels(text: string): Set<string> {
  const labels = new Set<string>();
  const labelRegex = /\{#((?:sec|fig|tbl|eq|lst)-[A-Za-z0-9:_-]+)\}/g;
  let match: RegExpExecArray | null;
  while ((match = labelRegex.exec(text)) !== null) {
    labels.add(match[1]);
  }
  return labels;
}

function collectDefinedPythonVariables(text: string): Set<string> {
  const vars = new Set<string>();
  const fenceRegex = /```(?:\{python\}|python)\s*\n([\s\S]*?)```/g;
  const assignRegex = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=/gm;
  let fenceMatch: RegExpExecArray | null;
  while ((fenceMatch = fenceRegex.exec(text)) !== null) {
    const code = fenceMatch[1];
    let assignMatch: RegExpExecArray | null;
    while ((assignMatch = assignRegex.exec(code)) !== null) {
      vars.add(assignMatch[1]);
    }
  }
  return vars;
}

function makeRangeForMatch(document: vscode.TextDocument, offset: number, length: number): vscode.Range {
  const start = document.positionAt(offset);
  const end = document.positionAt(offset + length);
  return new vscode.Range(start, end);
}

function buildDiagnostics(document: vscode.TextDocument): vscode.Diagnostic[] {
  const diagnostics: vscode.Diagnostic[] = [];
  const text = document.getText();
  const definedLabels = collectDefinedLabels(text);
  const definedPythonVars = collectDefinedPythonVariables(text);

  const crossrefRegex = /@((?:sec|fig|tbl|eq|lst)-[A-Za-z0-9:_-]+)/g;
  let crossrefMatch: RegExpExecArray | null;
  while ((crossrefMatch = crossrefRegex.exec(text)) !== null) {
    const ref = crossrefMatch[1];
    if (!definedLabels.has(ref)) {
      diagnostics.push(
        new vscode.Diagnostic(
          makeRangeForMatch(document, crossrefMatch.index + 1, ref.length),
          `Unresolved cross-reference: @${ref}`,
          vscode.DiagnosticSeverity.Warning,
        ),
      );
    }
  }

  const inlinePythonRegex = /`\{python\}\s+([^`]+)`/g;
  let inlineMatch: RegExpExecArray | null;
  while ((inlineMatch = inlinePythonRegex.exec(text)) !== null) {
    const rawExpr = inlineMatch[1].trim();
    const lineNumber = document.positionAt(inlineMatch.index).line;
    const lineText = document.lineAt(lineNumber).text;

    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(rawExpr)) {
      diagnostics.push(
        new vscode.Diagnostic(
          makeRangeForMatch(document, inlineMatch.index, inlineMatch[0].length),
          `Inline Python should reference a variable name only; found "${rawExpr}".`,
          vscode.DiagnosticSeverity.Information,
        ),
      );
      continue;
    }

    if (!definedPythonVars.has(rawExpr)) {
      diagnostics.push(
        new vscode.Diagnostic(
          makeRangeForMatch(document, inlineMatch.index, inlineMatch[0].length),
          `Inline Python variable "${rawExpr}" is not defined in this document's python code blocks.`,
          vscode.DiagnosticSeverity.Warning,
        ),
      );
    }

    const appearsInCaptionContext =
      lineText.includes('{#tbl-') || lineText.includes('{#fig-') || lineText.includes('fig-cap=') || lineText.includes('lst-cap=');
    if (appearsInCaptionContext) {
      diagnostics.push(
        new vscode.Diagnostic(
          makeRangeForMatch(document, inlineMatch.index, inlineMatch[0].length),
          'Inline Python inside figure/table/listing caption metadata is fragile; prefer static caption text.',
          vscode.DiagnosticSeverity.Information,
        ),
      );
    }

    const dollarCount = (lineText.match(/\$/g) ?? []).length;
    if (dollarCount >= 2) {
      diagnostics.push(
        new vscode.Diagnostic(
          makeRangeForMatch(document, inlineMatch.index, inlineMatch[0].length),
          'Inline Python inside LaTeX math can render unexpectedly; prefer pre-formatted strings in prose.',
          vscode.DiagnosticSeverity.Information,
        ),
      );
    }
  }

  return diagnostics;
}

export class QmdDiagnosticsManager implements vscode.Disposable {
  private readonly collection: vscode.DiagnosticCollection;
  private readonly disposables: vscode.Disposable[] = [];
  private refreshTimer: NodeJS.Timeout | undefined;

  constructor() {
    this.collection = vscode.languages.createDiagnosticCollection(DIAGNOSTIC_SOURCE);
    this.disposables.push(this.collection);
  }

  start(): void {
    this.refreshActiveEditorDiagnostics();
    this.disposables.push(
      vscode.window.onDidChangeActiveTextEditor(() => this.refreshActiveEditorDiagnostics()),
      vscode.workspace.onDidSaveTextDocument(document => this.refreshDocumentDiagnostics(document)),
      vscode.workspace.onDidChangeTextDocument(event => this.debouncedRefresh(event.document)),
    );
  }

  refreshActiveEditorDiagnostics(): void {
    const editor = vscode.window.activeTextEditor;
    if (!editor) { return; }
    this.refreshDocumentDiagnostics(editor.document);
  }

  refreshDocumentDiagnostics(document: vscode.TextDocument): void {
    if (!isQmdDocument(document)) {
      this.collection.delete(document.uri);
      return;
    }
    const diagnostics = buildDiagnostics(document);
    this.collection.set(document.uri, diagnostics);
  }

  private debouncedRefresh(document: vscode.TextDocument): void {
    if (!isQmdDocument(document)) { return; }
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
    }
    this.refreshTimer = setTimeout(() => {
      this.refreshDocumentDiagnostics(document);
    }, 300);
  }

  dispose(): void {
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
    }
    this.disposables.forEach(d => d.dispose());
  }
}
