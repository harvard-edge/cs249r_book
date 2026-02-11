import * as vscode from 'vscode';

function isQmdEditor(editor: vscode.TextEditor | undefined): editor is vscode.TextEditor {
  return Boolean(editor && editor.document.uri.fsPath.endsWith('.qmd'));
}

interface BlockRange {
  start: number;
  end: number;
}

type VisualPreset = 'subtle' | 'balanced' | 'highContrast';

function colorValue(value: string): vscode.ThemeColor | string {
  return value.startsWith('rgba(') || value.startsWith('#')
    ? value
    : new vscode.ThemeColor(value);
}

interface QmdColorOverrides {
  sectionH2Bg?: string;
  sectionH3Bg?: string;
  sectionH4Bg?: string;
  sectionH5Bg?: string;
  sectionH6Bg?: string;
  figureLineBg?: string;
  tableLineBg?: string;
  listingLineBg?: string;
  tableBg?: string;
  footnoteBg?: string;
  inlineRefColor?: string;
  structuralRefColor?: string;
  labelDefColor?: string;
  divFenceColor?: string;
  footnoteRefColor?: string;
  footnoteDefColor?: string;
}

export class QmdChunkHighlighter implements vscode.Disposable {
  private readonly disposables: vscode.Disposable[] = [];
  private calloutDecoration: vscode.TextEditorDecorationType | undefined;
  private divDecoration: vscode.TextEditorDecorationType | undefined;
  private codeFenceDecoration: vscode.TextEditorDecorationType | undefined;
  private tableDecoration: vscode.TextEditorDecorationType | undefined;
  private footnoteDecoration: vscode.TextEditorDecorationType | undefined;
  private figureLineDecoration: vscode.TextEditorDecorationType | undefined;
  private tableLineDecoration: vscode.TextEditorDecorationType | undefined;
  private listingLineDecoration: vscode.TextEditorDecorationType | undefined;
  private sectionHeaderDecorations = new Map<number, vscode.TextEditorDecorationType>();
  private footnoteReferenceDecoration: vscode.TextEditorDecorationType | undefined;
  private footnoteDefinitionMarkerDecoration: vscode.TextEditorDecorationType | undefined;
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
          || event.affectsConfiguration('mlsysbook.highlightSectionHeaders')
          || event.affectsConfiguration('mlsysbook.highlightTables')
          || event.affectsConfiguration('mlsysbook.highlightFootnotes')
          || event.affectsConfiguration('mlsysbook.colorSectionH2Bg')
          || event.affectsConfiguration('mlsysbook.colorSectionH3Bg')
          || event.affectsConfiguration('mlsysbook.colorSectionH4Bg')
          || event.affectsConfiguration('mlsysbook.colorSectionH5Bg')
          || event.affectsConfiguration('mlsysbook.colorSectionH6Bg')
          || event.affectsConfiguration('mlsysbook.colorFigureLabelBg')
          || event.affectsConfiguration('mlsysbook.colorTableLabelBg')
          || event.affectsConfiguration('mlsysbook.colorListingLabelBg')
          || event.affectsConfiguration('mlsysbook.colorTableRegionBg')
          || event.affectsConfiguration('mlsysbook.colorFootnoteRegionBg')
          || event.affectsConfiguration('mlsysbook.colorInlineReference')
          || event.affectsConfiguration('mlsysbook.colorStructuralReference')
          || event.affectsConfiguration('mlsysbook.colorLabelDefinition')
          || event.affectsConfiguration('mlsysbook.colorDivFenceMarker')
          || event.affectsConfiguration('mlsysbook.colorFootnoteReference')
          || event.affectsConfiguration('mlsysbook.colorFootnoteDefinitionMarker')
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

  private getHighlightSectionHeaders(): boolean {
    return vscode.workspace
      .getConfiguration('mlsysbook')
      .get<boolean>('highlightSectionHeaders', true);
  }

  private getHighlightTables(): boolean {
    return vscode.workspace
      .getConfiguration('mlsysbook')
      .get<boolean>('highlightTables', true);
  }

  private getHighlightFootnotes(): boolean {
    return vscode.workspace
      .getConfiguration('mlsysbook')
      .get<boolean>('highlightFootnotes', true);
  }

  private readOptionalColor(config: vscode.WorkspaceConfiguration, key: string): string | undefined {
    const value = config.get<string>(key);
    if (!value) {
      return undefined;
    }
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : undefined;
  }

  private getColorOverrides(): QmdColorOverrides {
    const config = vscode.workspace.getConfiguration('mlsysbook');
    return {
      sectionH2Bg: this.readOptionalColor(config, 'colorSectionH2Bg'),
      sectionH3Bg: this.readOptionalColor(config, 'colorSectionH3Bg'),
      sectionH4Bg: this.readOptionalColor(config, 'colorSectionH4Bg'),
      sectionH5Bg: this.readOptionalColor(config, 'colorSectionH5Bg'),
      sectionH6Bg: this.readOptionalColor(config, 'colorSectionH6Bg'),
      figureLineBg: this.readOptionalColor(config, 'colorFigureLabelBg'),
      tableLineBg: this.readOptionalColor(config, 'colorTableLabelBg'),
      listingLineBg: this.readOptionalColor(config, 'colorListingLabelBg'),
      tableBg: this.readOptionalColor(config, 'colorTableRegionBg'),
      footnoteBg: this.readOptionalColor(config, 'colorFootnoteRegionBg'),
      inlineRefColor: this.readOptionalColor(config, 'colorInlineReference'),
      structuralRefColor: this.readOptionalColor(config, 'colorStructuralReference'),
      labelDefColor: this.readOptionalColor(config, 'colorLabelDefinition'),
      divFenceColor: this.readOptionalColor(config, 'colorDivFenceMarker'),
      footnoteRefColor: this.readOptionalColor(config, 'colorFootnoteReference'),
      footnoteDefColor: this.readOptionalColor(config, 'colorFootnoteDefinitionMarker'),
    };
  }

  private recreateDecorations(): void {
    this.calloutDecoration?.dispose();
    this.divDecoration?.dispose();
    this.codeFenceDecoration?.dispose();
    this.tableDecoration?.dispose();
    this.footnoteDecoration?.dispose();
    this.figureLineDecoration?.dispose();
    this.tableLineDecoration?.dispose();
    this.listingLineDecoration?.dispose();
    this.sectionHeaderDecorations.forEach(decoration => decoration.dispose());
    this.sectionHeaderDecorations.clear();
    this.footnoteReferenceDecoration?.dispose();
    this.footnoteDefinitionMarkerDecoration?.dispose();
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
      figureLineBg: string;
      tableLineBg: string;
      listingLineBg: string;
      tableBg: string;
      footnoteBg: string;
      footnoteRefColor: string;
      footnoteDefColor: string;
      inlineRefColor: string;
      structuralRefColor: string;
      labelDefColor: string;
      divFenceColor: string;
      fontWeight: 'normal' | '500' | '600';
      sectionHeaderBgByLevel: [string, string, string, string, string];
    }> = {
      subtle: {
        calloutBg: 'rgba(56, 139, 253, 0.06)',
        divBg: 'rgba(148, 163, 184, 0.04)',
        codeBg: 'rgba(99, 102, 241, 0.04)',
        labelBg: 'rgba(116, 162, 255, 0.10)',
        figureLineBg: 'rgba(56, 189, 248, 0.20)',
        tableLineBg: 'rgba(34, 197, 94, 0.19)',
        listingLineBg: 'rgba(249, 115, 22, 0.17)',
        tableBg: 'rgba(16, 185, 129, 0.08)',
        footnoteBg: 'rgba(167, 139, 250, 0.09)',
        footnoteRefColor: 'editorInfo.foreground',
        footnoteDefColor: 'editorInfo.foreground',
        inlineRefColor: 'editorInfo.foreground',
        structuralRefColor: 'textLink.foreground',
        labelDefColor: 'editorInfo.foreground',
        divFenceColor: 'editorInfo.foreground',
        fontWeight: 'normal',
        sectionHeaderBgByLevel: [
          'rgba(88, 166, 255, 0.12)',
          'rgba(88, 166, 255, 0.09)',
          'rgba(88, 166, 255, 0.07)',
          'rgba(88, 166, 255, 0.05)',
          'rgba(88, 166, 255, 0.03)',
        ],
      },
      balanced: {
        calloutBg: 'rgba(56, 139, 253, 0.09)',
        divBg: 'rgba(148, 163, 184, 0.06)',
        codeBg: 'rgba(99, 102, 241, 0.06)',
        labelBg: 'rgba(116, 162, 255, 0.14)',
        figureLineBg: 'rgba(56, 189, 248, 0.26)',
        tableLineBg: 'rgba(34, 197, 94, 0.24)',
        listingLineBg: 'rgba(249, 115, 22, 0.22)',
        tableBg: 'rgba(16, 185, 129, 0.12)',
        footnoteBg: 'rgba(167, 139, 250, 0.16)',
        footnoteRefColor: 'textLink.foreground',
        footnoteDefColor: 'editorInfo.foreground',
        inlineRefColor: 'textLink.foreground',
        structuralRefColor: 'editorInfo.foreground',
        labelDefColor: 'editorInfo.foreground',
        divFenceColor: 'editorInfo.foreground',
        fontWeight: '500',
        sectionHeaderBgByLevel: [
          'rgba(88, 166, 255, 0.16)',
          'rgba(88, 166, 255, 0.12)',
          'rgba(88, 166, 255, 0.09)',
          'rgba(88, 166, 255, 0.07)',
          'rgba(88, 166, 255, 0.05)',
        ],
      },
      highContrast: {
        calloutBg: 'rgba(56, 139, 253, 0.13)',
        divBg: 'rgba(148, 163, 184, 0.10)',
        codeBg: 'rgba(99, 102, 241, 0.10)',
        labelBg: 'rgba(116, 162, 255, 0.20)',
        figureLineBg: 'rgba(56, 189, 248, 0.32)',
        tableLineBg: 'rgba(34, 197, 94, 0.30)',
        listingLineBg: 'rgba(249, 115, 22, 0.28)',
        tableBg: 'rgba(16, 185, 129, 0.18)',
        footnoteBg: 'rgba(167, 139, 250, 0.24)',
        footnoteRefColor: 'textLink.foreground',
        footnoteDefColor: 'editorInfo.foreground',
        inlineRefColor: 'textLink.foreground',
        structuralRefColor: 'editorInfo.foreground',
        labelDefColor: 'editorInfo.foreground',
        divFenceColor: 'textLink.foreground',
        fontWeight: '600',
        sectionHeaderBgByLevel: [
          'rgba(88, 166, 255, 0.24)',
          'rgba(88, 166, 255, 0.18)',
          'rgba(88, 166, 255, 0.14)',
          'rgba(88, 166, 255, 0.10)',
          'rgba(88, 166, 255, 0.07)',
        ],
      },
    };

    const baseStyle = styleByPreset[preset];
    const override = this.getColorOverrides();
    const style = {
      ...baseStyle,
      figureLineBg: override.figureLineBg ?? baseStyle.figureLineBg,
      tableLineBg: override.tableLineBg ?? baseStyle.tableLineBg,
      listingLineBg: override.listingLineBg ?? baseStyle.listingLineBg,
      tableBg: override.tableBg ?? baseStyle.tableBg,
      footnoteBg: override.footnoteBg ?? baseStyle.footnoteBg,
      inlineRefColor: override.inlineRefColor ?? baseStyle.inlineRefColor,
      structuralRefColor: override.structuralRefColor ?? baseStyle.structuralRefColor,
      labelDefColor: override.labelDefColor ?? baseStyle.labelDefColor,
      divFenceColor: override.divFenceColor ?? baseStyle.divFenceColor,
      footnoteRefColor: override.footnoteRefColor ?? baseStyle.footnoteRefColor,
      footnoteDefColor: override.footnoteDefColor ?? baseStyle.footnoteDefColor,
      sectionHeaderBgByLevel: [
        override.sectionH2Bg ?? baseStyle.sectionHeaderBgByLevel[0],
        override.sectionH3Bg ?? baseStyle.sectionHeaderBgByLevel[1],
        override.sectionH4Bg ?? baseStyle.sectionHeaderBgByLevel[2],
        override.sectionH5Bg ?? baseStyle.sectionHeaderBgByLevel[3],
        override.sectionH6Bg ?? baseStyle.sectionHeaderBgByLevel[4],
      ] as [string, string, string, string, string],
    };

    this.calloutDecoration = vscode.window.createTextEditorDecorationType({
      isWholeLine: true,
      backgroundColor: colorValue(style.calloutBg),
    });
    this.divDecoration = vscode.window.createTextEditorDecorationType({
      isWholeLine: true,
      backgroundColor: colorValue(style.divBg),
    });
    this.codeFenceDecoration = vscode.window.createTextEditorDecorationType({
      isWholeLine: true,
      backgroundColor: colorValue(style.codeBg),
    });
    this.tableDecoration = vscode.window.createTextEditorDecorationType({
      isWholeLine: true,
      backgroundColor: style.tableBg,
    });
    this.footnoteDecoration = vscode.window.createTextEditorDecorationType({
      isWholeLine: true,
      backgroundColor: style.footnoteBg,
    });
    this.footnoteReferenceDecoration = vscode.window.createTextEditorDecorationType({
      color: colorValue(style.footnoteRefColor),
      fontWeight: style.fontWeight,
      textDecoration: 'none',
    });
    this.footnoteDefinitionMarkerDecoration = vscode.window.createTextEditorDecorationType({
      color: colorValue(style.footnoteDefColor),
      fontWeight: style.fontWeight,
      textDecoration: 'none',
    });
    this.figureLineDecoration = vscode.window.createTextEditorDecorationType({
      isWholeLine: false,
      backgroundColor: colorValue(style.figureLineBg),
    });
    this.tableLineDecoration = vscode.window.createTextEditorDecorationType({
      isWholeLine: false,
      backgroundColor: colorValue(style.tableLineBg),
    });
    this.listingLineDecoration = vscode.window.createTextEditorDecorationType({
      isWholeLine: false,
      backgroundColor: colorValue(style.listingLineBg),
    });
    for (let level = 2; level <= 6; level++) {
      this.sectionHeaderDecorations.set(
        level,
        vscode.window.createTextEditorDecorationType({
          isWholeLine: true,
          backgroundColor: style.sectionHeaderBgByLevel[level - 2],
        }),
      );
    }
    this.inlineReferenceDecoration = vscode.window.createTextEditorDecorationType({
      color: colorValue(style.inlineRefColor),
      fontWeight: style.fontWeight,
      textDecoration: 'none',
    });
    this.structuralReferenceDecoration = vscode.window.createTextEditorDecorationType({
      color: colorValue(style.structuralRefColor),
      fontWeight: style.fontWeight,
      textDecoration: 'none',
    });
    this.labelDefinitionDecoration = vscode.window.createTextEditorDecorationType({
      color: colorValue(style.labelDefColor),
      fontWeight: style.fontWeight,
      textDecoration: 'none',
    });
    this.divFenceMarkerDecoration = vscode.window.createTextEditorDecorationType({
      color: colorValue(style.divFenceColor),
      fontWeight: style.fontWeight,
      textDecoration: 'none',
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
    const tableRanges: vscode.Range[] = [];
    const footnoteRanges: vscode.Range[] = [];
    const footnoteRefRanges: vscode.Range[] = [];
    const footnoteDefinitionMarkerRanges: vscode.Range[] = [];
    const figureLineRanges: vscode.Range[] = [];
    const tableLineRanges: vscode.Range[] = [];
    const listingLineRanges: vscode.Range[] = [];
    const sectionHeaderRanges = new Map<number, vscode.Range[]>([
      [2, []],
      [3, []],
      [4, []],
      [5, []],
      [6, []],
    ]);
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
    const highlightSectionHeaders = this.getHighlightSectionHeaders();
    const highlightTables = this.getHighlightTables();
    const highlightFootnotes = this.getHighlightFootnotes();
    const sectionHeaderRegex = /^(#{2,6})\s+\S+/;
    const tableRowRegex = /^\s*\|.*\|\s*$/;
    const tableAlignRegex = /^\s*\|?\s*:?-{3,}:?(?:\s*\|\s*:?-{3,}:?)*\s*\|?\s*$/;
    const footnoteStartRegex = /^\[\^[-\w:]+\]:/;
    const footnoteRefRegex = /\[\^[-\w:]+\]/g;
    const footnoteDefinitionMarkerRegex = /^\s*(\[\^[-\w:]+\]:)/;

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

      if (highlightSectionHeaders && !inFence) {
        const headerMatch = text.match(sectionHeaderRegex);
        if (headerMatch) {
          const level = headerMatch[1].length;
          const ranges = sectionHeaderRanges.get(level);
          if (ranges) {
            ranges.push(new vscode.Range(
              new vscode.Position(line, 0),
              new vscode.Position(line, text.length),
            ));
          }
        }
      }

      if (highlightTables && !inFence && tableRowRegex.test(text)) {
        let end = line;
        while (end + 1 < document.lineCount && tableRowRegex.test(document.lineAt(end + 1).text)) {
          end += 1;
        }
        const hasHeaderAndAlign =
          end > line
          && tableAlignRegex.test(document.lineAt(line + 1).text);
        if (hasHeaderAndAlign) {
          let blockEnd = end;
          if (
            blockEnd + 1 < document.lineCount
            && /^\s*:\s+.*\{#tbl-[^}]+\}\s*$/.test(document.lineAt(blockEnd + 1).text)
          ) {
            blockEnd += 1;
          }
          tableRanges.push(new vscode.Range(
            new vscode.Position(line, 0),
            new vscode.Position(blockEnd, document.lineAt(blockEnd).text.length),
          ));
          line = blockEnd;
          continue;
        }
      }

      if (highlightFootnotes && !inFence && footnoteStartRegex.test(text.trim())) {
        const markerMatch = text.match(footnoteDefinitionMarkerRegex);
        if (markerMatch) {
          const markerStart = (markerMatch.index ?? 0) + markerMatch[0].indexOf(markerMatch[1]);
          footnoteDefinitionMarkerRanges.push(new vscode.Range(
            new vscode.Position(line, markerStart),
            new vscode.Position(line, markerStart + markerMatch[1].length),
          ));
        }
        let end = line;
        while (end + 1 < document.lineCount) {
          const next = document.lineAt(end + 1).text;
          if (next.trim().length === 0) {
            end += 1;
            continue;
          }
          if (/^\s+/.test(next)) {
            end += 1;
            continue;
          }
          break;
        }
        footnoteRanges.push(new vscode.Range(
          new vscode.Position(line, 0),
          new vscode.Position(end, document.lineAt(end).text.length),
        ));
      }

      if (highlightFootnotes && !inFence) {
        let footnoteMatch: RegExpExecArray | null;
        while ((footnoteMatch = footnoteRefRegex.exec(text)) !== null) {
          footnoteRefRanges.push(new vscode.Range(
            new vscode.Position(line, footnoteMatch.index),
            new vscode.Position(line, footnoteMatch.index + footnoteMatch[0].length),
          ));
        }
        footnoteRefRegex.lastIndex = 0;
      }

      if (/#fig-/.test(text)) {
        const range = new vscode.Range(
          new vscode.Position(line, 0),
          new vscode.Position(line, text.length),
        );
        figureLineRanges.push(range);
      }

      if (/#tbl-/.test(text)) {
        const range = new vscode.Range(
          new vscode.Position(line, 0),
          new vscode.Position(line, text.length),
        );
        tableLineRanges.push(range);
      }

      if (/#lst-/.test(text)) {
        const range = new vscode.Range(
          new vscode.Position(line, 0),
          new vscode.Position(line, text.length),
        );
        listingLineRanges.push(range);
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
    if (this.tableDecoration) {
      editor.setDecorations(this.tableDecoration, tableRanges);
    }
    if (this.footnoteDecoration) {
      editor.setDecorations(this.footnoteDecoration, footnoteRanges);
    }
    if (this.footnoteReferenceDecoration) {
      editor.setDecorations(this.footnoteReferenceDecoration, footnoteRefRanges);
    }
    if (this.footnoteDefinitionMarkerDecoration) {
      editor.setDecorations(this.footnoteDefinitionMarkerDecoration, footnoteDefinitionMarkerRanges);
    }
    if (this.figureLineDecoration) {
      editor.setDecorations(this.figureLineDecoration, figureLineRanges);
    }
    if (this.tableLineDecoration) {
      editor.setDecorations(this.tableLineDecoration, tableLineRanges);
    }
    if (this.listingLineDecoration) {
      editor.setDecorations(this.listingLineDecoration, listingLineRanges);
    }
    for (let level = 2; level <= 6; level++) {
      const decoration = this.sectionHeaderDecorations.get(level);
      if (decoration) {
        editor.setDecorations(decoration, sectionHeaderRanges.get(level) ?? []);
      }
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
    if (this.tableDecoration) {
      editor.setDecorations(this.tableDecoration, []);
    }
    if (this.footnoteDecoration) {
      editor.setDecorations(this.footnoteDecoration, []);
    }
    if (this.footnoteReferenceDecoration) {
      editor.setDecorations(this.footnoteReferenceDecoration, []);
    }
    if (this.footnoteDefinitionMarkerDecoration) {
      editor.setDecorations(this.footnoteDefinitionMarkerDecoration, []);
    }
    if (this.figureLineDecoration) {
      editor.setDecorations(this.figureLineDecoration, []);
    }
    if (this.tableLineDecoration) {
      editor.setDecorations(this.tableLineDecoration, []);
    }
    if (this.listingLineDecoration) {
      editor.setDecorations(this.listingLineDecoration, []);
    }
    this.sectionHeaderDecorations.forEach(decoration => {
      editor.setDecorations(decoration, []);
    });
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
