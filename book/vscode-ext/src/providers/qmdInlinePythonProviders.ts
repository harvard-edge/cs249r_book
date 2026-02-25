import * as vscode from 'vscode';
import { QmdPythonValueResolver, extractInlineRefs } from './qmdPythonValueResolver';

// ─── Shared regex ────────────────────────────────────────────────────────────

const INLINE_PYTHON_REGEX = /`\{python\}\s+(\w+)`/g;
const CODE_BLOCK_START = /^```\{python\}/;
const CODE_BLOCK_END = /^```\s*$/;
const EXPORT_ASSIGNMENT = /^(\w+)\s*=\s*/;

// ─── Hover Provider ──────────────────────────────────────────────────────────

export class QmdPythonHoverProvider implements vscode.HoverProvider {
  constructor(private readonly resolver: QmdPythonValueResolver) {}

  async provideHover(
    document: vscode.TextDocument,
    position: vscode.Position,
    _token: vscode.CancellationToken,
  ): Promise<vscode.Hover | null> {
    if (!document.uri.fsPath.endsWith('.qmd')) {
      return null;
    }

    const line = document.lineAt(position.line).text;
    let match: RegExpExecArray | null;
    const regex = new RegExp(INLINE_PYTHON_REGEX.source, 'g');

    while ((match = regex.exec(line)) !== null) {
      const start = match.index;
      const end = start + match[0].length;
      if (position.character >= start && position.character <= end) {
        const varName = match[1];
        const range = new vscode.Range(
          new vscode.Position(position.line, start),
          new vscode.Position(position.line, end),
        );

        const values = await this.resolver.getValues(document);
        const value = values.get(varName);

        const md = new vscode.MarkdownString();
        md.isTrusted = true;

        if (value !== undefined) {
          md.appendMarkdown(`**Resolved value:** \`${value}\`\n\n`);
          md.appendMarkdown(`---\n\n`);
          md.appendMarkdown(`*Variable:* \`${varName}\`  \n`);
          md.appendMarkdown(`*Source:* Inline Python reference`);
        } else {
          md.appendMarkdown(`**Variable:** \`${varName}\`\n\n`);
          md.appendMarkdown(`⚠️ *Could not resolve value.* The Python code blocks may have errors, or this variable may not be defined.`);
        }

        return new vscode.Hover(md, range);
      }
    }

    return null;
  }
}

// ─── Ghost Text (Inline Value Decorations) ───────────────────────────────────

export class QmdPythonGhostText implements vscode.Disposable {
  private readonly disposables: vscode.Disposable[] = [];
  private ghostDecorationType: vscode.TextEditorDecorationType | undefined;

  constructor(private readonly resolver: QmdPythonValueResolver) {
    this.recreateDecorationType();
  }

  start(): void {
    // Trigger initial resolution for the active editor
    const active = vscode.window.activeTextEditor;
    if (active?.document.uri.fsPath.endsWith('.qmd')) {
      this.resolver.triggerResolve(active.document);
    }

    this.disposables.push(
      vscode.window.onDidChangeActiveTextEditor(editor => {
        if (editor?.document.uri.fsPath.endsWith('.qmd')) {
          this.resolver.triggerResolve(editor.document);
        } else if (editor && this.ghostDecorationType) {
          editor.setDecorations(this.ghostDecorationType, []);
        }
      }),
      vscode.workspace.onDidChangeTextDocument(event => {
        const activeEditor = vscode.window.activeTextEditor;
        if (!activeEditor || activeEditor.document.uri.toString() !== event.document.uri.toString()) {
          return;
        }
        if (!activeEditor.document.uri.fsPath.endsWith('.qmd')) {
          return;
        }
        this.resolver.invalidate(activeEditor.document.uri);
        this.resolver.triggerResolve(activeEditor.document);
      }),
      // Re-apply decorations when Python values are resolved
      this.resolver.onDidResolve(uri => {
        const activeEditor = vscode.window.activeTextEditor;
        if (activeEditor && activeEditor.document.uri.toString() === uri.toString()) {
          void this.applyToEditor(activeEditor);
        }
      }),
      vscode.workspace.onDidChangeConfiguration(event => {
        if (
          event.affectsConfiguration('mlsysbook.showInlinePythonValues')
          || event.affectsConfiguration('mlsysbook.qmdVisualPreset')
        ) {
          this.recreateDecorationType();
          void this.applyToEditor(vscode.window.activeTextEditor);
        }
      }),
    );
  }

  private isEnabled(): boolean {
    return vscode.workspace
      .getConfiguration('mlsysbook')
      .get<boolean>('showInlinePythonValues', true);
  }

  private recreateDecorationType(): void {
    this.ghostDecorationType?.dispose();
    // Ghost text is created per-range via DecorationOptions.renderOptions,
    // so we use a minimal decoration type here.
    this.ghostDecorationType = vscode.window.createTextEditorDecorationType({});
  }

  private async applyToEditor(editor: vscode.TextEditor | undefined): Promise<void> {
    if (!editor || !editor.document.uri.fsPath.endsWith('.qmd') || !this.isEnabled()) {
      if (editor && this.ghostDecorationType) {
        editor.setDecorations(this.ghostDecorationType, []);
      }
      return;
    }

    // Use cached values if available (synchronous), otherwise await
    const values = this.resolver.getCachedValues(editor.document)
      ?? await this.resolver.getValues(editor.document);
    if (values.size === 0) {
      if (this.ghostDecorationType) {
        editor.setDecorations(this.ghostDecorationType, []);
      }
      return;
    }

    const decorations: vscode.DecorationOptions[] = [];
    const text = editor.document.getText();
    const lines = text.split('\n');

    // Don't show ghost text inside code fences
    let fenceDepth = 0;

    for (let lineIdx = 0; lineIdx < lines.length; lineIdx++) {
      const lineText = lines[lineIdx];

      const fenceMatch = lineText.match(/^\s*(```|~~~)/);
      if (fenceMatch) {
        if (fenceDepth === 0) {
          fenceDepth++;
        } else {
          fenceDepth--;
        }
        continue;
      }

      if (fenceDepth > 0) { continue; }

      const regex = new RegExp(INLINE_PYTHON_REGEX.source, 'g');
      let match: RegExpExecArray | null;
      while ((match = regex.exec(lineText)) !== null) {
        const varName = match[1];
        const value = values.get(varName);
        if (value !== undefined) {
          const end = match.index + match[0].length;
          decorations.push({
            range: new vscode.Range(
              new vscode.Position(lineIdx, end),
              new vscode.Position(lineIdx, end),
            ),
            renderOptions: {
              after: {
                contentText: ` → ${value}`,
                color: 'rgba(249, 168, 212, 0.55)',
                fontStyle: 'italic',
                fontWeight: 'normal',
              },
            },
          });
        }
      }
    }

    if (this.ghostDecorationType) {
      editor.setDecorations(this.ghostDecorationType, decorations);
    }
  }

  dispose(): void {
    this.ghostDecorationType?.dispose();
    this.disposables.forEach(d => d.dispose());
  }
}

// ─── CodeLens Provider ───────────────────────────────────────────────────────

export class QmdPythonCodeLensProvider implements vscode.CodeLensProvider, vscode.Disposable {
  private readonly disposables: vscode.Disposable[] = [];
  private readonly onDidChangeEmitter = new vscode.EventEmitter<void>();
  readonly onDidChangeCodeLenses = this.onDidChangeEmitter.event;

  constructor(private readonly resolver: QmdPythonValueResolver) {}

  start(): void {
    this.disposables.push(
      vscode.workspace.onDidChangeTextDocument(event => {
        if (event.document.uri.fsPath.endsWith('.qmd')) {
          this.resolver.invalidate(event.document.uri);
          this.onDidChangeEmitter.fire();
        }
      }),
      this.resolver.onDidResolve(() => {
        this.onDidChangeEmitter.fire();
      }),
      vscode.workspace.onDidChangeConfiguration(event => {
        if (event.affectsConfiguration('mlsysbook.showInlinePythonCodeLens')) {
          this.onDidChangeEmitter.fire();
        }
      }),
    );
  }

  private isEnabled(): boolean {
    return vscode.workspace
      .getConfiguration('mlsysbook')
      .get<boolean>('showInlinePythonCodeLens', true);
  }

  async provideCodeLenses(
    document: vscode.TextDocument,
    _token: vscode.CancellationToken,
  ): Promise<vscode.CodeLens[]> {
    if (!document.uri.fsPath.endsWith('.qmd') || !this.isEnabled()) {
      return [];
    }

    const text = document.getText();
    const lines = text.split('\n');
    const inlineVars = extractInlineRefs(text);
    if (inlineVars.length === 0) {
      return [];
    }

    // Find the EXPORTS section of each code block — the last code block
    // that assigns variables used inline
    const codeBlockStarts: number[] = [];
    let inBlock = false;

    for (let i = 0; i < lines.length; i++) {
      if (CODE_BLOCK_START.test(lines[i])) {
        codeBlockStarts.push(i);
        inBlock = true;
      } else if (inBlock && CODE_BLOCK_END.test(lines[i])) {
        inBlock = false;
      }
    }

    if (codeBlockStarts.length === 0) {
      return [];
    }

    // Find which code block exports which variables
    const values = await this.resolver.getValues(document);
    if (values.size === 0) {
      return [];
    }

    // Find the last code block (usually the EXPORTS bridge)
    // and put a CodeLens above it showing all resolved values
    const lenses: vscode.CodeLens[] = [];

    // Find code blocks that contain export assignments for inline vars
    inBlock = false;
    let currentBlockStart = -1;
    const blockExports = new Map<number, string[]>();

    for (let i = 0; i < lines.length; i++) {
      if (CODE_BLOCK_START.test(lines[i])) {
        inBlock = true;
        currentBlockStart = i;
      } else if (inBlock && CODE_BLOCK_END.test(lines[i])) {
        inBlock = false;
        currentBlockStart = -1;
      } else if (inBlock && currentBlockStart >= 0) {
        const assignMatch = lines[i].match(EXPORT_ASSIGNMENT);
        if (assignMatch) {
          const varName = assignMatch[1];
          if (inlineVars.includes(varName) && values.has(varName)) {
            if (!blockExports.has(currentBlockStart)) {
              blockExports.set(currentBlockStart, []);
            }
            blockExports.get(currentBlockStart)!.push(varName);
          }
        }
      }
    }

    for (const [blockLine, vars] of blockExports) {
      const parts = vars.map(v => {
        const val = values.get(v);
        return `${v} = "${val}"`;
      });

      // Split into chunks if too long
      const maxLen = 120;
      let current = '📐 ';
      const chunks: string[] = [];
      for (const part of parts) {
        if (current.length + part.length + 3 > maxLen && current.length > 3) {
          chunks.push(current.trimEnd());
          current = '   ';
        }
        current += part + '  ·  ';
      }
      if (current.trim().length > 0) {
        // Remove trailing separator
        chunks.push(current.replace(/\s*·\s*$/, ''));
      }

      for (let ci = 0; ci < chunks.length; ci++) {
        lenses.push(new vscode.CodeLens(
          new vscode.Range(blockLine, 0, blockLine, 0),
          {
            title: chunks[ci],
            command: '',
          },
        ));
      }
    }

    return lenses;
  }

  dispose(): void {
    this.onDidChangeEmitter.dispose();
    this.disposables.forEach(d => d.dispose());
  }
}
