import * as vscode from 'vscode';

const DIAGNOSTIC_SOURCE = 'mlsysbook-qmd';

function isQmdDocument(document: vscode.TextDocument): boolean {
  return document.uri.fsPath.endsWith('.qmd');
}

// ─── Label collection patterns ──────────────────────────────────────────────

/** Inline label definitions: {#sec-foo}, {#fig-bar}, etc. */
const INLINE_LABEL_REGEX = /\{#((?:sec|fig|tbl|eq|lst)-[A-Za-z0-9:_-]+)\}/g;

/** YAML-style label definitions: #| label: fig-foo, #| fig-label: fig-foo, etc. */
const YAML_LABEL_REGEX = /#\|\s*(?:label|fig-label|tbl-label|lst-label):\s*((?:sec|fig|tbl|eq|lst)-[A-Za-z0-9:_-]+)/g;

/** Cross-reference pattern: @sec-foo, @fig-bar, etc. */
const CROSSREF_REGEX = /@((?:sec|fig|tbl|eq|lst)-[A-Za-z0-9:_-]+)/g;

/**
 * Collect all label definitions from a document's text.
 * Handles both inline {#label} and YAML #| label: formats.
 */
function collectDefinedLabels(text: string): Set<string> {
  const labels = new Set<string>();

  let match: RegExpExecArray | null;

  INLINE_LABEL_REGEX.lastIndex = 0;
  while ((match = INLINE_LABEL_REGEX.exec(text)) !== null) {
    labels.add(match[1]);
  }

  YAML_LABEL_REGEX.lastIndex = 0;
  while ((match = YAML_LABEL_REGEX.exec(text)) !== null) {
    labels.add(match[1]);
  }

  return labels;
}

/**
 * Build a set of character-offset ranges that fall inside fenced code blocks.
 * Used to suppress false-positive diagnostics for cross-references that appear
 * in Python comments (e.g. P.I.C.O. documentation headers).
 */
function collectFencedCodeRanges(text: string): Array<{ start: number; end: number }> {
  const ranges: Array<{ start: number; end: number }> = [];
  const fenceRegex = /^(\s*(?:```|~~~)).*$/gm;
  let openStart: number | null = null;
  let openMarker: string | null = null;
  let match: RegExpExecArray | null;
  while ((match = fenceRegex.exec(text)) !== null) {
    const marker = match[1].trim().slice(0, 3); // ``` or ~~~
    if (openMarker === null) {
      openStart = match.index;
      openMarker = marker;
    } else if (marker === openMarker) {
      ranges.push({ start: openStart!, end: match.index + match[0].length });
      openStart = null;
      openMarker = null;
    }
  }
  return ranges;
}

function isInsideFence(offset: number, fenceRanges: Array<{ start: number; end: number }>): boolean {
  for (const range of fenceRanges) {
    if (offset >= range.start && offset <= range.end) { return true; }
    if (range.start > offset) { break; } // ranges are sorted by start
  }
  return false;
}

/**
 * Collect all cross-references from a document's text.
 * Skips references that appear inside fenced code blocks (```, ~~~) to avoid
 * false positives from cross-references mentioned in Python comments.
 */
function collectReferences(text: string): Array<{ ref: string; index: number }> {
  const refs: Array<{ ref: string; index: number }> = [];
  const fenceRanges = collectFencedCodeRanges(text);
  let match: RegExpExecArray | null;
  CROSSREF_REGEX.lastIndex = 0;
  while ((match = CROSSREF_REGEX.exec(text)) !== null) {
    if (!isInsideFence(match.index, fenceRanges)) {
      refs.push({ ref: match[1], index: match.index });
    }
  }
  return refs;
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

// ─── Workspace Label Index ──────────────────────────────────────────────────

/**
 * Maintains a workspace-wide index of all label definitions across .qmd files.
 * Updated incrementally when individual files are saved.
 */
export class WorkspaceLabelIndex implements vscode.Disposable {
  /** Map from label string to the set of URIs where it's defined. */
  private readonly labelToUris = new Map<string, Set<string>>();
  /** Map from file URI string to the set of labels defined in that file. */
  private readonly uriToLabels = new Map<string, Set<string>>();
  private readonly disposables: vscode.Disposable[] = [];
  private initialized = false;

  private readonly onDidUpdateEmitter = new vscode.EventEmitter<void>();
  /** Fires when the index has been updated (for triggering re-validation). */
  readonly onDidUpdate = this.onDidUpdateEmitter.event;

  start(): void {
    // Build initial index
    this.buildFullIndex().then(() => {
      this.initialized = true;
      this.onDidUpdateEmitter.fire();
    });

    // Update index when .qmd files are saved
    this.disposables.push(
      vscode.workspace.onDidSaveTextDocument(document => {
        if (isQmdDocument(document)) {
          this.updateFileLabels(document.uri.toString(), document.getText());
          this.onDidUpdateEmitter.fire();
        }
      }),
      // Handle file deletions
      vscode.workspace.onDidDeleteFiles(event => {
        let changed = false;
        for (const uri of event.files) {
          if (uri.fsPath.endsWith('.qmd')) {
            this.removeFile(uri.toString());
            changed = true;
          }
        }
        if (changed) { this.onDidUpdateEmitter.fire(); }
      }),
    );
  }

  /** Check if a label exists anywhere in the workspace. */
  hasLabel(label: string): boolean {
    const uris = this.labelToUris.get(label);
    return !!uris && uris.size > 0;
  }

  /** Check if the index has been built at least once. */
  isReady(): boolean {
    return this.initialized;
  }

  /** Get the total number of indexed labels. */
  get size(): number {
    return this.labelToUris.size;
  }

  /** Get all known labels (for completions, etc.). */
  allLabels(): string[] {
    return Array.from(this.labelToUris.keys());
  }

  private async buildFullIndex(): Promise<void> {
    const files = await vscode.workspace.findFiles('**/*.qmd', '**/node_modules/**');
    for (const uri of files) {
      try {
        const doc = await vscode.workspace.openTextDocument(uri);
        this.updateFileLabels(uri.toString(), doc.getText());
      } catch {
        // File may have been deleted between findFiles and openTextDocument
      }
    }
  }

  private updateFileLabels(uriStr: string, text: string): void {
    // Remove old labels for this file
    this.removeFile(uriStr);

    // Collect new labels
    const labels = collectDefinedLabels(text);
    this.uriToLabels.set(uriStr, labels);

    for (const label of labels) {
      let uris = this.labelToUris.get(label);
      if (!uris) {
        uris = new Set();
        this.labelToUris.set(label, uris);
      }
      uris.add(uriStr);
    }
  }

  private removeFile(uriStr: string): void {
    const oldLabels = this.uriToLabels.get(uriStr);
    if (oldLabels) {
      for (const label of oldLabels) {
        const uris = this.labelToUris.get(label);
        if (uris) {
          uris.delete(uriStr);
          if (uris.size === 0) {
            this.labelToUris.delete(label);
          }
        }
      }
      this.uriToLabels.delete(uriStr);
    }
  }

  dispose(): void {
    this.onDidUpdateEmitter.dispose();
    this.disposables.forEach(d => d.dispose());
    this.labelToUris.clear();
    this.uriToLabels.clear();
  }
}

// ─── Diagnostics Builder ────────────────────────────────────────────────────

function buildDiagnostics(
  document: vscode.TextDocument,
  workspaceIndex?: WorkspaceLabelIndex,
): vscode.Diagnostic[] {
  const diagnostics: vscode.Diagnostic[] = [];
  const text = document.getText();
  const localLabels = collectDefinedLabels(text);
  const definedPythonVars = collectDefinedPythonVariables(text);

  // Cross-reference validation
  const refs = collectReferences(text);
  for (const { ref, index } of refs) {
    // Check local labels first (fast path), then workspace index
    const isLocal = localLabels.has(ref);
    const isWorkspace = workspaceIndex?.hasLabel(ref) ?? false;

    if (!isLocal && !isWorkspace) {
      const diagnostic = new vscode.Diagnostic(
        makeRangeForMatch(document, index + 1, ref.length),
        workspaceIndex?.isReady()
          ? `Unresolved cross-reference: @${ref} (not found in any .qmd file)`
          : `Unresolved cross-reference: @${ref} (workspace index loading...)`,
        workspaceIndex?.isReady()
          ? vscode.DiagnosticSeverity.Warning
          : vscode.DiagnosticSeverity.Information,
      );
      diagnostic.source = DIAGNOSTIC_SOURCE;
      diagnostics.push(diagnostic);
    }
  }

  // Inline Python validation
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

// ─── Diagnostics Manager ────────────────────────────────────────────────────

export class QmdDiagnosticsManager implements vscode.Disposable {
  private readonly collection: vscode.DiagnosticCollection;
  private readonly disposables: vscode.Disposable[] = [];
  private refreshTimer: NodeJS.Timeout | undefined;
  private workspaceIndex: WorkspaceLabelIndex | undefined;

  constructor() {
    this.collection = vscode.languages.createDiagnosticCollection(DIAGNOSTIC_SOURCE);
    this.disposables.push(this.collection);
  }

  /**
   * Inject the workspace label index for cross-file reference validation.
   * Must be called before start().
   */
  setWorkspaceIndex(index: WorkspaceLabelIndex): void {
    this.workspaceIndex = index;
  }

  start(): void {
    // Only validate on save and editor switch — NOT on every keystroke
    this.disposables.push(
      vscode.window.onDidChangeActiveTextEditor(() => this.refreshActiveEditorDiagnostics()),
      vscode.workspace.onDidSaveTextDocument(document => this.refreshDocumentDiagnostics(document)),
      vscode.workspace.onDidChangeConfiguration(event => {
        if (event.affectsConfiguration('mlsysbook.enableQmdDiagnostics')) {
          if (this.isEnabled()) {
            this.refreshActiveEditorDiagnostics();
          } else {
            this.collection.clear();
          }
        }
      }),
    );

    // Re-validate active editor when workspace index updates
    if (this.workspaceIndex) {
      this.disposables.push(
        this.workspaceIndex.onDidUpdate(() => {
          if (this.isEnabled()) {
            this.refreshActiveEditorDiagnostics();
          }
        }),
      );
    }

    // Initial validation
    this.refreshActiveEditorDiagnostics();
  }

  refreshActiveEditorDiagnostics(): void {
    const editor = vscode.window.activeTextEditor;
    if (!editor) { return; }
    this.refreshDocumentDiagnostics(editor.document);
  }

  refreshDocumentDiagnostics(document: vscode.TextDocument): void {
    if (!isQmdDocument(document) || !this.isEnabled()) {
      this.collection.delete(document.uri);
      return;
    }
    const diagnostics = buildDiagnostics(document, this.workspaceIndex);
    this.collection.set(document.uri, diagnostics);
  }

  private isEnabled(): boolean {
    return vscode.workspace
      .getConfiguration('mlsysbook')
      .get<boolean>('enableQmdDiagnostics', true);
  }

  dispose(): void {
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
    }
    this.disposables.forEach(d => d.dispose());
  }
}
