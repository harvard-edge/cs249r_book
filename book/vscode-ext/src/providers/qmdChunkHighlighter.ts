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
  sectionRefColor?: string;
  figureRefColor?: string;
  tableRefColor?: string;
  listingRefColor?: string;
  equationRefColor?: string;
  sectionLabelDefColor?: string;
  figureLabelDefColor?: string;
  tableLabelDefColor?: string;
  listingLabelDefColor?: string;
  equationLabelDefColor?: string;
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
  private sectionReferenceDecoration: vscode.TextEditorDecorationType | undefined;
  private figureReferenceDecoration: vscode.TextEditorDecorationType | undefined;
  private tableReferenceDecoration: vscode.TextEditorDecorationType | undefined;
  private listingReferenceDecoration: vscode.TextEditorDecorationType | undefined;
  private equationReferenceDecoration: vscode.TextEditorDecorationType | undefined;
  private labelDefinitionDecoration: vscode.TextEditorDecorationType | undefined;
  private sectionLabelDefinitionDecoration: vscode.TextEditorDecorationType | undefined;
  private figureLabelDefinitionDecoration: vscode.TextEditorDecorationType | undefined;
  private tableLabelDefinitionDecoration: vscode.TextEditorDecorationType | undefined;
  private listingLabelDefinitionDecoration: vscode.TextEditorDecorationType | undefined;
  private equationLabelDefinitionDecoration: vscode.TextEditorDecorationType | undefined;
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
          || event.affectsConfiguration('mlsysbook.highlightFootnoteBlockBackground')
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
          || event.affectsConfiguration('mlsysbook.colorSectionReference')
          || event.affectsConfiguration('mlsysbook.colorFigureReference')
          || event.affectsConfiguration('mlsysbook.colorTableReference')
          || event.affectsConfiguration('mlsysbook.colorListingReference')
          || event.affectsConfiguration('mlsysbook.colorEquationReference')
          || event.affectsConfiguration('mlsysbook.colorSectionLabelDefinition')
          || event.affectsConfiguration('mlsysbook.colorFigureLabelDefinition')
          || event.affectsConfiguration('mlsysbook.colorTableLabelDefinition')
          || event.affectsConfiguration('mlsysbook.colorListingLabelDefinition')
          || event.affectsConfiguration('mlsysbook.colorEquationLabelDefinition')
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

  private getHighlightFootnoteBlockBackground(): boolean {
    return vscode.workspace
      .getConfiguration('mlsysbook')
      .get<boolean>('highlightFootnoteBlockBackground', false);
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
      sectionRefColor: this.readOptionalColor(config, 'colorSectionReference'),
      figureRefColor: this.readOptionalColor(config, 'colorFigureReference'),
      tableRefColor: this.readOptionalColor(config, 'colorTableReference'),
      listingRefColor: this.readOptionalColor(config, 'colorListingReference'),
      equationRefColor: this.readOptionalColor(config, 'colorEquationReference'),
      sectionLabelDefColor: this.readOptionalColor(config, 'colorSectionLabelDefinition'),
      figureLabelDefColor: this.readOptionalColor(config, 'colorFigureLabelDefinition'),
      tableLabelDefColor: this.readOptionalColor(config, 'colorTableLabelDefinition'),
      listingLabelDefColor: this.readOptionalColor(config, 'colorListingLabelDefinition'),
      equationLabelDefColor: this.readOptionalColor(config, 'colorEquationLabelDefinition'),
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
    this.sectionReferenceDecoration?.dispose();
    this.figureReferenceDecoration?.dispose();
    this.tableReferenceDecoration?.dispose();
    this.listingReferenceDecoration?.dispose();
    this.equationReferenceDecoration?.dispose();
    this.labelDefinitionDecoration?.dispose();
    this.sectionLabelDefinitionDecoration?.dispose();
    this.figureLabelDefinitionDecoration?.dispose();
    this.tableLabelDefinitionDecoration?.dispose();
    this.listingLabelDefinitionDecoration?.dispose();
    this.equationLabelDefinitionDecoration?.dispose();
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
      sectionRefColor: string;
      figureRefColor: string;
      tableRefColor: string;
      listingRefColor: string;
      equationRefColor: string;
      sectionLabelDefColor: string;
      figureLabelDefColor: string;
      tableLabelDefColor: string;
      listingLabelDefColor: string;
      equationLabelDefColor: string;
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
        footnoteRefColor: 'editorWarning.foreground',
        footnoteDefColor: 'editorWarning.foreground',
        inlineRefColor: 'editorInfo.foreground',
        structuralRefColor: 'textLink.foreground',
        sectionRefColor: '#7aa2f7',
        figureRefColor: '#22d3ee',
        tableRefColor: '#34d399',
        listingRefColor: '#fb923c',
        equationRefColor: '#c084fc',
        sectionLabelDefColor: '#7aa2f7',
        figureLabelDefColor: '#22d3ee',
        tableLabelDefColor: '#34d399',
        listingLabelDefColor: '#fb923c',
        equationLabelDefColor: '#c084fc',
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
        footnoteRefColor: 'editorWarning.foreground',
        footnoteDefColor: 'editorWarning.foreground',
        inlineRefColor: 'textLink.foreground',
        structuralRefColor: 'editorInfo.foreground',
        sectionRefColor: '#9ab5ff',
        figureRefColor: '#67e8f9',
        tableRefColor: '#6ee7b7',
        listingRefColor: '#fdba74',
        equationRefColor: '#d8b4fe',
        sectionLabelDefColor: '#9ab5ff',
        figureLabelDefColor: '#67e8f9',
        tableLabelDefColor: '#6ee7b7',
        listingLabelDefColor: '#fdba74',
        equationLabelDefColor: '#d8b4fe',
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
        footnoteRefColor: 'editorWarning.foreground',
        footnoteDefColor: 'editorWarning.foreground',
        inlineRefColor: 'textLink.foreground',
        structuralRefColor: 'editorInfo.foreground',
        sectionRefColor: '#b7c9ff',
        figureRefColor: '#a5f3fc',
        tableRefColor: '#86efac',
        listingRefColor: '#fdc58a',
        equationRefColor: '#e9d5ff',
        sectionLabelDefColor: '#b7c9ff',
        figureLabelDefColor: '#a5f3fc',
        tableLabelDefColor: '#86efac',
        listingLabelDefColor: '#fdc58a',
        equationLabelDefColor: '#e9d5ff',
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
      sectionRefColor: override.sectionRefColor ?? baseStyle.sectionRefColor,
      figureRefColor: override.figureRefColor ?? baseStyle.figureRefColor,
      tableRefColor: override.tableRefColor ?? baseStyle.tableRefColor,
      listingRefColor: override.listingRefColor ?? baseStyle.listingRefColor,
      equationRefColor: override.equationRefColor ?? baseStyle.equationRefColor,
      sectionLabelDefColor: override.sectionLabelDefColor ?? baseStyle.sectionLabelDefColor,
      figureLabelDefColor: override.figureLabelDefColor ?? baseStyle.figureLabelDefColor,
      tableLabelDefColor: override.tableLabelDefColor ?? baseStyle.tableLabelDefColor,
      listingLabelDefColor: override.listingLabelDefColor ?? baseStyle.listingLabelDefColor,
      equationLabelDefColor: override.equationLabelDefColor ?? baseStyle.equationLabelDefColor,
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
    this.sectionReferenceDecoration = vscode.window.createTextEditorDecorationType({
      color: colorValue(style.sectionRefColor),
      fontWeight: style.fontWeight,
      textDecoration: 'none',
    });
    this.figureReferenceDecoration = vscode.window.createTextEditorDecorationType({
      color: colorValue(style.figureRefColor),
      fontWeight: style.fontWeight,
      textDecoration: 'none',
    });
    this.tableReferenceDecoration = vscode.window.createTextEditorDecorationType({
      color: colorValue(style.tableRefColor),
      fontWeight: style.fontWeight,
      textDecoration: 'none',
    });
    this.listingReferenceDecoration = vscode.window.createTextEditorDecorationType({
      color: colorValue(style.listingRefColor),
      fontWeight: style.fontWeight,
      textDecoration: 'none',
    });
    this.equationReferenceDecoration = vscode.window.createTextEditorDecorationType({
      color: colorValue(style.equationRefColor),
      fontWeight: style.fontWeight,
      textDecoration: 'none',
    });
    this.labelDefinitionDecoration = vscode.window.createTextEditorDecorationType({
      color: colorValue(style.labelDefColor),
      fontWeight: style.fontWeight,
      textDecoration: 'none',
    });
    this.sectionLabelDefinitionDecoration = vscode.window.createTextEditorDecorationType({
      color: colorValue(style.sectionLabelDefColor),
      fontWeight: style.fontWeight,
      textDecoration: 'none',
    });
    this.figureLabelDefinitionDecoration = vscode.window.createTextEditorDecorationType({
      color: colorValue(style.figureLabelDefColor),
      fontWeight: style.fontWeight,
      textDecoration: 'none',
    });
    this.tableLabelDefinitionDecoration = vscode.window.createTextEditorDecorationType({
      color: colorValue(style.tableLabelDefColor),
      fontWeight: style.fontWeight,
      textDecoration: 'none',
    });
    this.listingLabelDefinitionDecoration = vscode.window.createTextEditorDecorationType({
      color: colorValue(style.listingLabelDefColor),
      fontWeight: style.fontWeight,
      textDecoration: 'none',
    });
    this.equationLabelDefinitionDecoration = vscode.window.createTextEditorDecorationType({
      color: colorValue(style.equationLabelDefColor),
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
    const sectionRefRanges: vscode.Range[] = [];
    const figureRefRanges: vscode.Range[] = [];
    const tableRefRanges: vscode.Range[] = [];
    const listingRefRanges: vscode.Range[] = [];
    const equationRefRanges: vscode.Range[] = [];
    const labelDefinitionRanges: vscode.Range[] = [];
    const sectionLabelDefinitionRanges: vscode.Range[] = [];
    const figureLabelDefinitionRanges: vscode.Range[] = [];
    const tableLabelDefinitionRanges: vscode.Range[] = [];
    const listingLabelDefinitionRanges: vscode.Range[] = [];
    const equationLabelDefinitionRanges: vscode.Range[] = [];
    const divFenceMarkerRanges: vscode.Range[] = [];

    const calloutBlocks: BlockRange[] = [];
    const divBlocks: BlockRange[] = [];
    const codeBlocks: BlockRange[] = [];
    const divStack: Array<{ start: number; isCallout: boolean }> = [];
    const fenceStack: Array<{ start: number; marker: string }> = [];
    let inFence = false;
    const inlineRefRegex = /@[A-Za-z][A-Za-z0-9_:-]*/g;
    const structuralRefRegex = /@(?:fig|tbl|sec|lst|eq)-[A-Za-z0-9_:-]+/g;
    const sectionRefRegex = /@sec-[A-Za-z0-9_:-]+/g;
    const figureRefRegex = /@fig-[A-Za-z0-9_:-]+/g;
    const tableRefRegex = /@tbl-[A-Za-z0-9_:-]+/g;
    const listingRefRegex = /@lst-[A-Za-z0-9_:-]+/g;
    const equationRefRegex = /@eq-[A-Za-z0-9_:-]+/g;
    const labelInlineRegex = /\{#(?:fig|tbl|sec|lst|eq)-[A-Za-z0-9_:-]+\}/g;
    const sectionLabelInlineRegex = /\{#sec-[A-Za-z0-9_:-]+\}/g;
    const figureLabelInlineRegex = /\{#fig-[A-Za-z0-9_:-]+\}/g;
    const tableLabelInlineRegex = /\{#tbl-[A-Za-z0-9_:-]+\}/g;
    const listingLabelInlineRegex = /\{#lst-[A-Za-z0-9_:-]+\}/g;
    const equationLabelInlineRegex = /\{#eq-[A-Za-z0-9_:-]+\}/g;
    const labelYamlRegex = /#\|\s*(?:label|fig-label|tbl-label|lst-label):\s*(?:fig|tbl|lst)-[A-Za-z0-9_:-]+/g;
    const sectionLabelYamlRegex = /#\|\s*label:\s*sec-[A-Za-z0-9_:-]+/g;
    const figureLabelYamlRegex = /#\|\s*(?:label|fig-label):\s*fig-[A-Za-z0-9_:-]+/g;
    const tableLabelYamlRegex = /#\|\s*(?:label|tbl-label):\s*tbl-[A-Za-z0-9_:-]+/g;
    const listingLabelYamlRegex = /#\|\s*(?:label|lst-label):\s*lst-[A-Za-z0-9_:-]+/g;
    const equationLabelYamlRegex = /#\|\s*label:\s*eq-[A-Za-z0-9_:-]+/g;
    const highlightInlineRefs = this.getHighlightInlineReferences();
    const highlightLabelDefs = this.getHighlightLabelDefinitions();
    const highlightDivFenceMarkers = this.getHighlightDivFenceMarkers();
    const highlightSectionHeaders = this.getHighlightSectionHeaders();
    const highlightTables = this.getHighlightTables();
    const highlightFootnotes = this.getHighlightFootnotes();
    const highlightFootnoteBlockBackground = this.getHighlightFootnoteBlockBackground();
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
        if (highlightFootnoteBlockBackground) {
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
          // Highlight only non-empty lines in a footnote block. This avoids
          // tinting separator blank lines and prevents perceived spillover.
          for (let footnoteLine = line; footnoteLine <= end; footnoteLine++) {
            const footnoteText = document.lineAt(footnoteLine).text;
            if (footnoteText.trim().length === 0) {
              continue;
            }
            footnoteRanges.push(new vscode.Range(
              new vscode.Position(footnoteLine, 0),
              new vscode.Position(footnoteLine, footnoteText.length),
            ));
          }
        }
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

        while ((labelMatch = sectionLabelInlineRegex.exec(text)) !== null) {
          sectionLabelDefinitionRanges.push(new vscode.Range(
            new vscode.Position(line, labelMatch.index),
            new vscode.Position(line, labelMatch.index + labelMatch[0].length),
          ));
        }
        sectionLabelInlineRegex.lastIndex = 0;
        while ((labelMatch = figureLabelInlineRegex.exec(text)) !== null) {
          figureLabelDefinitionRanges.push(new vscode.Range(
            new vscode.Position(line, labelMatch.index),
            new vscode.Position(line, labelMatch.index + labelMatch[0].length),
          ));
        }
        figureLabelInlineRegex.lastIndex = 0;
        while ((labelMatch = tableLabelInlineRegex.exec(text)) !== null) {
          tableLabelDefinitionRanges.push(new vscode.Range(
            new vscode.Position(line, labelMatch.index),
            new vscode.Position(line, labelMatch.index + labelMatch[0].length),
          ));
        }
        tableLabelInlineRegex.lastIndex = 0;
        while ((labelMatch = listingLabelInlineRegex.exec(text)) !== null) {
          listingLabelDefinitionRanges.push(new vscode.Range(
            new vscode.Position(line, labelMatch.index),
            new vscode.Position(line, labelMatch.index + labelMatch[0].length),
          ));
        }
        listingLabelInlineRegex.lastIndex = 0;
        while ((labelMatch = equationLabelInlineRegex.exec(text)) !== null) {
          equationLabelDefinitionRanges.push(new vscode.Range(
            new vscode.Position(line, labelMatch.index),
            new vscode.Position(line, labelMatch.index + labelMatch[0].length),
          ));
        }
        equationLabelInlineRegex.lastIndex = 0;

        while ((labelMatch = sectionLabelYamlRegex.exec(text)) !== null) {
          sectionLabelDefinitionRanges.push(new vscode.Range(
            new vscode.Position(line, labelMatch.index),
            new vscode.Position(line, labelMatch.index + labelMatch[0].length),
          ));
        }
        sectionLabelYamlRegex.lastIndex = 0;
        while ((labelMatch = figureLabelYamlRegex.exec(text)) !== null) {
          figureLabelDefinitionRanges.push(new vscode.Range(
            new vscode.Position(line, labelMatch.index),
            new vscode.Position(line, labelMatch.index + labelMatch[0].length),
          ));
        }
        figureLabelYamlRegex.lastIndex = 0;
        while ((labelMatch = tableLabelYamlRegex.exec(text)) !== null) {
          tableLabelDefinitionRanges.push(new vscode.Range(
            new vscode.Position(line, labelMatch.index),
            new vscode.Position(line, labelMatch.index + labelMatch[0].length),
          ));
        }
        tableLabelYamlRegex.lastIndex = 0;
        while ((labelMatch = listingLabelYamlRegex.exec(text)) !== null) {
          listingLabelDefinitionRanges.push(new vscode.Range(
            new vscode.Position(line, labelMatch.index),
            new vscode.Position(line, labelMatch.index + labelMatch[0].length),
          ));
        }
        listingLabelYamlRegex.lastIndex = 0;
        while ((labelMatch = equationLabelYamlRegex.exec(text)) !== null) {
          equationLabelDefinitionRanges.push(new vscode.Range(
            new vscode.Position(line, labelMatch.index),
            new vscode.Position(line, labelMatch.index + labelMatch[0].length),
          ));
        }
        equationLabelYamlRegex.lastIndex = 0;
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

        while ((match = sectionRefRegex.exec(text)) !== null) {
          sectionRefRanges.push(new vscode.Range(
            new vscode.Position(line, match.index),
            new vscode.Position(line, match.index + match[0].length),
          ));
        }
        sectionRefRegex.lastIndex = 0;

        while ((match = figureRefRegex.exec(text)) !== null) {
          figureRefRanges.push(new vscode.Range(
            new vscode.Position(line, match.index),
            new vscode.Position(line, match.index + match[0].length),
          ));
        }
        figureRefRegex.lastIndex = 0;

        while ((match = tableRefRegex.exec(text)) !== null) {
          tableRefRanges.push(new vscode.Range(
            new vscode.Position(line, match.index),
            new vscode.Position(line, match.index + match[0].length),
          ));
        }
        tableRefRegex.lastIndex = 0;

        while ((match = listingRefRegex.exec(text)) !== null) {
          listingRefRanges.push(new vscode.Range(
            new vscode.Position(line, match.index),
            new vscode.Position(line, match.index + match[0].length),
          ));
        }
        listingRefRegex.lastIndex = 0;

        while ((match = equationRefRegex.exec(text)) !== null) {
          equationRefRanges.push(new vscode.Range(
            new vscode.Position(line, match.index),
            new vscode.Position(line, match.index + match[0].length),
          ));
        }
        equationRefRegex.lastIndex = 0;
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
    if (this.sectionReferenceDecoration) {
      editor.setDecorations(this.sectionReferenceDecoration, sectionRefRanges);
    }
    if (this.figureReferenceDecoration) {
      editor.setDecorations(this.figureReferenceDecoration, figureRefRanges);
    }
    if (this.tableReferenceDecoration) {
      editor.setDecorations(this.tableReferenceDecoration, tableRefRanges);
    }
    if (this.listingReferenceDecoration) {
      editor.setDecorations(this.listingReferenceDecoration, listingRefRanges);
    }
    if (this.equationReferenceDecoration) {
      editor.setDecorations(this.equationReferenceDecoration, equationRefRanges);
    }
    if (this.labelDefinitionDecoration) {
      editor.setDecorations(this.labelDefinitionDecoration, labelDefinitionRanges);
    }
    if (this.sectionLabelDefinitionDecoration) {
      editor.setDecorations(this.sectionLabelDefinitionDecoration, sectionLabelDefinitionRanges);
    }
    if (this.figureLabelDefinitionDecoration) {
      editor.setDecorations(this.figureLabelDefinitionDecoration, figureLabelDefinitionRanges);
    }
    if (this.tableLabelDefinitionDecoration) {
      editor.setDecorations(this.tableLabelDefinitionDecoration, tableLabelDefinitionRanges);
    }
    if (this.listingLabelDefinitionDecoration) {
      editor.setDecorations(this.listingLabelDefinitionDecoration, listingLabelDefinitionRanges);
    }
    if (this.equationLabelDefinitionDecoration) {
      editor.setDecorations(this.equationLabelDefinitionDecoration, equationLabelDefinitionRanges);
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
    if (this.sectionReferenceDecoration) {
      editor.setDecorations(this.sectionReferenceDecoration, []);
    }
    if (this.figureReferenceDecoration) {
      editor.setDecorations(this.figureReferenceDecoration, []);
    }
    if (this.tableReferenceDecoration) {
      editor.setDecorations(this.tableReferenceDecoration, []);
    }
    if (this.listingReferenceDecoration) {
      editor.setDecorations(this.listingReferenceDecoration, []);
    }
    if (this.equationReferenceDecoration) {
      editor.setDecorations(this.equationReferenceDecoration, []);
    }
    if (this.labelDefinitionDecoration) {
      editor.setDecorations(this.labelDefinitionDecoration, []);
    }
    if (this.sectionLabelDefinitionDecoration) {
      editor.setDecorations(this.sectionLabelDefinitionDecoration, []);
    }
    if (this.figureLabelDefinitionDecoration) {
      editor.setDecorations(this.figureLabelDefinitionDecoration, []);
    }
    if (this.tableLabelDefinitionDecoration) {
      editor.setDecorations(this.tableLabelDefinitionDecoration, []);
    }
    if (this.listingLabelDefinitionDecoration) {
      editor.setDecorations(this.listingLabelDefinitionDecoration, []);
    }
    if (this.equationLabelDefinitionDecoration) {
      editor.setDecorations(this.equationLabelDefinitionDecoration, []);
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
