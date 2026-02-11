import * as vscode from 'vscode';

function isQmdEditor(editor: vscode.TextEditor | undefined): editor is vscode.TextEditor {
  return Boolean(editor && editor.document.uri.fsPath.endsWith('.qmd'));
}

interface BlockRange {
  start: number;
  end: number;
}

type VisualPreset = 'subtle' | 'balanced' | 'highContrast';

export class QmdChunkHighlighter implements vscode.Disposable {
  private readonly disposables: vscode.Disposable[] = [];
  private calloutDecoration: vscode.TextEditorDecorationType | undefined;
  private divDecoration: vscode.TextEditorDecorationType | undefined;
  private codeFenceDecoration: vscode.TextEditorDecorationType | undefined;
  private figureTableLineDecoration: vscode.TextEditorDecorationType | undefined;
  private inlineReferenceDecoration: vscode.TextEditorDecorationType | undefined;
  private structuralReferenceDecoration: vscode.TextEditorDecorationType | undefined;
  private labelDefinitionDecoration: vscode.TextEditorDecorationType | undefined;
  private divFenceMarkerDecoration: vscode.TextEditorDecorationType | undefined;
  private refreshTimer: NodeJS.Timeout | undefined;

  constructor() {
    this.recreateDecorations();
  }

  start(): void {
    this.applyToEditor(vscode.window.activeTextEditor);
    this.disposables.push(
      vscode.window.onDidChangeActiveTextEditor(editor => this.applyToEditor(editor)),
      vscode.workspace.onDidChangeTextDocument(event => {
        const active = vscode.window.activeTextEditor;
        if (!active || active.document.uri.toString() !== event.document.uri.toString()) { return; }
        this.debouncedApply(active);
      }),
      vscode.workspace.onDidChangeConfiguration(event => {
        if (
          event.affectsConfiguration('mlsysbook.enableQmdChunkHighlight')
          || event.affectsConfiguration('mlsysbook.qmdVisualPreset')
          || event.affectsConfiguration('mlsysbook.highlightInlineReferences')
          || event.affectsConfiguration('mlsysbook.highlightLabelDefinitions')
          || event.affectsConfiguration('mlsysbook.highlightDivFenceMarkers')
        ) {
          this.recreateDecorations();
          this.applyToEditor(vscode.window.activeTextEditor);
        }
      }),
    );
  }

  private isEnabled(): boolean {
    return vscode.workspace.getConfiguration('mlsysbook').get<boolean>('enableQmdChunkHighlight', true);
  }

  private getVisualPreset(): VisualPreset {
    return vscode.workspace
      .getConfiguration('mlsysbook')
      .get<VisualPreset>('qmdVisualPreset', 'balanced');
  }

  private getHighlightInlineReferences(): boolean {
    return vscode.workspace
      .getConfiguration('mlsysbook')
      .get<boolean>('highlightInlineReferences', true);
  }

  private getHighlightLabelDefinitions(): boolean {
    return vscode.workspace
      .getConfiguration('mlsysbook')
      .get<boolean>('highlightLabelDefinitions', true);
  }

  private getHighlightDivFenceMarkers(): boolean {
    return vscode.workspace
      .getConfiguration('mlsysbook')
      .get<boolean>('highlightDivFenceMarkers', true);
  }

  private recreateDecorations(): void {
    this.calloutDecoration?.dispose();
    this.divDecoration?.dispose();
    this.codeFenceDecoration?.dispose();
    this.figureTableLineDecoration?.dispose();
    this.inlineReferenceDecoration?.dispose();
    this.structuralReferenceDecoration?.dispose();
    this.labelDefinitionDecoration?.dispose();
    this.divFenceMarkerDecoration?.dispose();

    const preset = this.getVisualPreset();

    const styleByPreset: Record<VisualPreset, {
      calloutBg: string;
      divBg: string;
      codeBg: string;
      labelBg: string;
      inlineRefColor: string;
      structuralRefColor: string;
      labelDefColor: string;
      divFenceColor: string;
      fontWeight: 'normal' | '500' | '600';
    }> = {
      subtle: {
        calloutBg: 'editor.wordHighlightBackground',
        divBg: 'editor.selectionHighlightBackground',
        codeBg: 'editor.selectionHighlightBackground',
        labelBg: 'editor.findMatchHighlightBackground',
        inlineRefColor: 'descriptionForeground',
        structuralRefColor: 'textLink.foreground',
        labelDefColor: 'editorInfo.foreground',
        divFenceColor: 'descriptionForeground',
        fontWeight: 'normal',
      },
      balanced: {
        calloutBg: 'editor.wordHighlightStrongBackground',
        divBg: 'editor.wordHighlightBackground',
        codeBg: 'editor.selectionHighlightBackground',
        labelBg: 'editor.findMatchHighlightBackground',
        inlineRefColor: 'textLink.foreground',
        structuralRefColor: 'editorInfo.foreground',
        labelDefColor: 'editorWarning.foreground',
        divFenceColor: 'editorInfo.foreground',
        fontWeight: '500',
      },
      highContrast: {
        calloutBg: 'editor.wordHighlightStrongBackground',
        divBg: 'editor.wordHighlightStrongBackground',
        codeBg: 'editor.findMatchHighlightBackground',
        labelBg: 'editor.findMatchHighlightBackground',
        inlineRefColor: 'editorWarning.foreground',
        structuralRefColor: 'editorError.foreground',
        labelDefColor: 'editorError.foreground',
        divFenceColor: 'editorWarning.foreground',
        fontWeight: '600',
      },
    };

    const style = styleByPreset[preset];

    this.calloutDecoration = vscode.window.createTextEditorDecorationType({
      isWholeLine: true,
      backgroundColor: new vscode.ThemeColor(style.calloutBg),
    });
    this.divDecoration = vscode.window.createTextEditorDecorationType({
      isWholeLine: true,
      backgroundColor: new vscode.ThemeColor(style.divBg),
    });
    this.codeFenceDecoration = vscode.window.createTextEditorDecorationType({
      isWholeLine: true,
      backgroundColor: new vscode.ThemeColor(style.codeBg),
    });
    this.figureTableLineDecoration = vscode.window.createTextEditorDecorationType({
      isWholeLine: false,
      backgroundColor: new vscode.ThemeColor(style.labelBg),
    });
    this.inlineReferenceDecoration = vscode.window.createTextEditorDecorationType({
      color: new vscode.ThemeColor(style.inlineRefColor),
      fontWeight: style.fontWeight,
    });
    this.structuralReferenceDecoration = vscode.window.createTextEditorDecorationType({
      color: new vscode.ThemeColor(style.structuralRefColor),
      fontWeight: style.fontWeight,
      textDecoration: 'underline',
    });
    this.labelDefinitionDecoration = vscode.window.createTextEditorDecorationType({
      color: new vscode.ThemeColor(style.labelDefColor),
      fontWeight: style.fontWeight,
    });
    this.divFenceMarkerDecoration = vscode.window.createTextEditorDecorationType({
      color: new vscode.ThemeColor(style.divFenceColor),
      fontWeight: style.fontWeight,
    });
  }

  private debouncedApply(editor: vscode.TextEditor): void {
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
    }
    this.refreshTimer = setTimeout(() => this.applyToEditor(editor), 200);
  }

  private applyToEditor(editor: vscode.TextEditor | undefined): void {
    if (!isQmdEditor(editor) || !this.isEnabled()) {
      if (editor) {
        this.clearDecorations(editor);
      }
      return;
    }

    const document = editor.document;
    const calloutRanges: vscode.Range[] = [];
    const divRanges: vscode.Range[] = [];
    const codeRanges: vscode.Range[] = [];
    const lineHighlightRanges: vscode.Range[] = [];
    const inlineRefRanges: vscode.Range[] = [];
    const structuralRefRanges: vscode.Range[] = [];
    const labelDefinitionRanges: vscode.Range[] = [];
    const divFenceMarkerRanges: vscode.Range[] = [];

    const calloutBlocks: BlockRange[] = [];
    const divBlocks: BlockRange[] = [];
    const codeBlocks: BlockRange[] = [];
    const divStack: Array<{ start: number; isCallout: boolean }> = [];
    const fenceStack: Array<{ start: number; marker: string }> = [];
    let inFence = false;
    const inlineRefRegex = /@[A-Za-z][A-Za-z0-9_:-]*/g;
    const structuralRefRegex = /@(?:fig|tbl|sec|lst|eq)-[A-Za-z0-9_:-]+/g;
    const labelInlineRegex = /\{#(?:fig|tbl|sec|lst|eq)-[A-Za-z0-9_:-]+\}/g;
    const labelYamlRegex = /#\|\s*(?:label|fig-label|tbl-label|lst-label):\s*(?:fig|tbl|lst)-[A-Za-z0-9_:-]+/g;
    const highlightInlineRefs = this.getHighlightInlineReferences();
    const highlightLabelDefs = this.getHighlightLabelDefinitions();
    const highlightDivFenceMarkers = this.getHighlightDivFenceMarkers();

    for (let line = 0; line < document.lineCount; line++) {
      const text = document.lineAt(line).text;

      if (/^\s*:{3,}\s*/.test(text)) {
        if (highlightDivFenceMarkers) {
          divFenceMarkerRanges.push(new vscode.Range(
            new vscode.Position(line, 0),
            new vscode.Position(line, text.length),
          ));
        }
        if (divStack.length === 0 || /:{3,}\s*\{/.test(text)) {
          divStack.push({ start: line, isCallout: text.includes('.callout-') });
        } else {
          const last = divStack.pop();
          if (last) {
            if (last.isCallout) {
              calloutBlocks.push({ start: last.start, end: line });
            } else {
              divBlocks.push({ start: last.start, end: line });
            }
          }
        }
      }

      const fenceMatch = text.match(/^\s*(```|~~~)/);
      if (fenceMatch) {
        const marker = fenceMatch[1];
        if (fenceStack.length === 0) {
          fenceStack.push({ start: line, marker });
          inFence = true;
        } else if (fenceStack[fenceStack.length - 1].marker === marker) {
          const open = fenceStack.pop();
          if (open) {
            codeBlocks.push({ start: open.start, end: line });
          }
          if (fenceStack.length === 0) {
            inFence = false;
          }
        }
      }

      if (/#(fig-|tbl-|lst-)/.test(text)) {
        const range = new vscode.Range(
          new vscode.Position(line, 0),
          new vscode.Position(line, text.length),
        );
        lineHighlightRanges.push(range);
      }

      if (highlightLabelDefs) {
        let labelMatch: RegExpExecArray | null;
        while ((labelMatch = labelInlineRegex.exec(text)) !== null) {
          labelDefinitionRanges.push(new vscode.Range(
            new vscode.Position(line, labelMatch.index),
            new vscode.Position(line, labelMatch.index + labelMatch[0].length),
          ));
        }
        labelInlineRegex.lastIndex = 0;

        while ((labelMatch = labelYamlRegex.exec(text)) !== null) {
          labelDefinitionRanges.push(new vscode.Range(
            new vscode.Position(line, labelMatch.index),
            new vscode.Position(line, labelMatch.index + labelMatch[0].length),
          ));
        }
        labelYamlRegex.lastIndex = 0;
      }

      if (highlightInlineRefs && (!inFence || text.trim().startsWith('#|'))) {
        let match: RegExpExecArray | null;
        while ((match = inlineRefRegex.exec(text)) !== null) {
          inlineRefRanges.push(new vscode.Range(
            new vscode.Position(line, match.index),
            new vscode.Position(line, match.index + match[0].length),
          ));
        }
        inlineRefRegex.lastIndex = 0;

        while ((match = structuralRefRegex.exec(text)) !== null) {
          structuralRefRanges.push(new vscode.Range(
            new vscode.Position(line, match.index),
            new vscode.Position(line, match.index + match[0].length),
          ));
        }
        structuralRefRegex.lastIndex = 0;
      }
    }

    const toRanges = (blocks: BlockRange[]): vscode.Range[] => {
      return blocks.map(block => new vscode.Range(
        new vscode.Position(block.start, 0),
        new vscode.Position(block.end, document.lineAt(block.end).text.length),
      ));
    };

    calloutRanges.push(...toRanges(calloutBlocks));
    divRanges.push(...toRanges(divBlocks));
    codeRanges.push(...toRanges(codeBlocks));

    if (this.calloutDecoration) {
      editor.setDecorations(this.calloutDecoration, calloutRanges);
    }
    if (this.divDecoration) {
      editor.setDecorations(this.divDecoration, divRanges);
    }
    if (this.codeFenceDecoration) {
      editor.setDecorations(this.codeFenceDecoration, codeRanges);
    }
    if (this.figureTableLineDecoration) {
      editor.setDecorations(this.figureTableLineDecoration, lineHighlightRanges);
    }
    if (this.inlineReferenceDecoration) {
      editor.setDecorations(this.inlineReferenceDecoration, inlineRefRanges);
    }
    if (this.structuralReferenceDecoration) {
      editor.setDecorations(this.structuralReferenceDecoration, structuralRefRanges);
    }
    if (this.labelDefinitionDecoration) {
      editor.setDecorations(this.labelDefinitionDecoration, labelDefinitionRanges);
    }
    if (this.divFenceMarkerDecoration) {
      editor.setDecorations(this.divFenceMarkerDecoration, divFenceMarkerRanges);
    }
  }

  private clearDecorations(editor: vscode.TextEditor): void {
    if (this.calloutDecoration) {
      editor.setDecorations(this.calloutDecoration, []);
    }
    if (this.divDecoration) {
      editor.setDecorations(this.divDecoration, []);
    }
    if (this.codeFenceDecoration) {
      editor.setDecorations(this.codeFenceDecoration, []);
    }
    if (this.figureTableLineDecoration) {
      editor.setDecorations(this.figureTableLineDecoration, []);
    }
    if (this.inlineReferenceDecoration) {
      editor.setDecorations(this.inlineReferenceDecoration, []);
    }
    if (this.structuralReferenceDecoration) {
      editor.setDecorations(this.structuralReferenceDecoration, []);
    }
    if (this.labelDefinitionDecoration) {
      editor.setDecorations(this.labelDefinitionDecoration, []);
    }
    if (this.divFenceMarkerDecoration) {
      editor.setDecorations(this.divFenceMarkerDecoration, []);
    }
  }

  dispose(): void {
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
    }
    this.disposables.forEach(d => d.dispose());
  }
}
