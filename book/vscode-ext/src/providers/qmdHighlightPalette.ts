export type VisualPreset = 'subtle' | 'balanced' | 'highContrast';

export interface QmdColorOverrides {
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
  inlinePythonColor?: string;
  inlinePythonBg?: string;
}

export interface HighlightStyle {
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
  inlinePythonColor: string;
  inlinePythonBg: string;
  inlinePythonKeywordColor: string;
  fontWeight: 'normal' | '500' | '600';
  sectionHeaderBgByLevel: [string, string, string, string, string];
}

const STYLE_BY_PRESET: Record<VisualPreset, HighlightStyle> = {
  subtle: {
    calloutBg: 'rgba(56, 139, 253, 0.06)',
    divBg: 'rgba(148, 163, 184, 0.04)',
    codeBg: 'rgba(99, 102, 241, 0.04)',
    labelBg: 'rgba(116, 162, 255, 0.10)',
    figureLineBg: 'rgba(34, 211, 238, 0.20)',
    tableLineBg: 'rgba(52, 211, 153, 0.19)',
    listingLineBg: 'rgba(251, 146, 60, 0.20)',
    tableBg: 'rgba(16, 185, 129, 0.08)',
    footnoteBg: 'rgba(245, 158, 11, 0.10)',
    footnoteRefColor: '#fbbf24',
    footnoteDefColor: '#fbbf24',
    inlineRefColor: 'editorInfo.foreground',
    structuralRefColor: 'textLink.foreground',
    sectionRefColor: '#8fb4ff',
    figureRefColor: '#67e8f9',
    tableRefColor: '#6ee7b7',
    listingRefColor: '#fdba74',
    equationRefColor: '#d8b4fe',
    sectionLabelDefColor: '#8fb4ff',
    figureLabelDefColor: '#67e8f9',
    tableLabelDefColor: '#6ee7b7',
    listingLabelDefColor: '#fdba74',
    equationLabelDefColor: '#d8b4fe',
    labelDefColor: 'editorInfo.foreground',
    divFenceColor: 'editorInfo.foreground',
    inlinePythonColor: '#f0abcf',
    inlinePythonBg: 'rgba(244, 114, 182, 0.08)',
    inlinePythonKeywordColor: 'rgba(240, 171, 207, 0.45)',
    fontWeight: 'normal',
    sectionHeaderBgByLevel: [
      'rgba(96, 165, 250, 0.13)',
      'rgba(96, 165, 250, 0.10)',
      'rgba(96, 165, 250, 0.07)',
      'rgba(96, 165, 250, 0.05)',
      'rgba(96, 165, 250, 0.03)',
    ],
  },
  balanced: {
    calloutBg: 'rgba(56, 139, 253, 0.09)',
    divBg: 'rgba(148, 163, 184, 0.06)',
    codeBg: 'rgba(99, 102, 241, 0.06)',
    labelBg: 'rgba(116, 162, 255, 0.14)',
    figureLineBg: 'rgba(34, 211, 238, 0.28)',
    tableLineBg: 'rgba(52, 211, 153, 0.25)',
    listingLineBg: 'rgba(251, 146, 60, 0.28)',
    tableBg: 'rgba(16, 185, 129, 0.12)',
    footnoteBg: 'rgba(245, 158, 11, 0.16)',
    footnoteRefColor: '#fbbf24',
    footnoteDefColor: '#fbbf24',
    inlineRefColor: 'textLink.foreground',
    structuralRefColor: 'editorInfo.foreground',
    sectionRefColor: '#a7c4ff',
    figureRefColor: '#7cecfb',
    tableRefColor: '#86efac',
    listingRefColor: '#fdc58a',
    equationRefColor: '#e4c7ff',
    sectionLabelDefColor: '#a7c4ff',
    figureLabelDefColor: '#7cecfb',
    tableLabelDefColor: '#86efac',
    listingLabelDefColor: '#fdc58a',
    equationLabelDefColor: '#e4c7ff',
    labelDefColor: 'editorInfo.foreground',
    divFenceColor: 'editorInfo.foreground',
    inlinePythonColor: '#f9a8d4',
    inlinePythonBg: 'rgba(244, 114, 182, 0.12)',
    inlinePythonKeywordColor: 'rgba(249, 168, 212, 0.45)',
    fontWeight: '500',
    sectionHeaderBgByLevel: [
      'rgba(96, 165, 250, 0.18)',
      'rgba(96, 165, 250, 0.14)',
      'rgba(96, 165, 250, 0.10)',
      'rgba(96, 165, 250, 0.07)',
      'rgba(96, 165, 250, 0.05)',
    ],
  },
  highContrast: {
    calloutBg: 'rgba(56, 139, 253, 0.13)',
    divBg: 'rgba(148, 163, 184, 0.10)',
    codeBg: 'rgba(99, 102, 241, 0.10)',
    labelBg: 'rgba(116, 162, 255, 0.20)',
    figureLineBg: 'rgba(34, 211, 238, 0.34)',
    tableLineBg: 'rgba(52, 211, 153, 0.32)',
    listingLineBg: 'rgba(251, 146, 60, 0.34)',
    tableBg: 'rgba(16, 185, 129, 0.18)',
    footnoteBg: 'rgba(245, 158, 11, 0.24)',
    footnoteRefColor: '#fcd34d',
    footnoteDefColor: '#fcd34d',
    inlineRefColor: 'textLink.foreground',
    structuralRefColor: 'editorInfo.foreground',
    sectionRefColor: '#c3d4ff',
    figureRefColor: '#b8f6fc',
    tableRefColor: '#9bf2bf',
    listingRefColor: '#ffd1a3',
    equationRefColor: '#eddcff',
    sectionLabelDefColor: '#c3d4ff',
    figureLabelDefColor: '#b8f6fc',
    tableLabelDefColor: '#9bf2bf',
    listingLabelDefColor: '#ffd1a3',
    equationLabelDefColor: '#eddcff',
    labelDefColor: 'editorInfo.foreground',
    divFenceColor: 'textLink.foreground',
    inlinePythonColor: '#fbb6d9',
    inlinePythonBg: 'rgba(244, 114, 182, 0.18)',
    inlinePythonKeywordColor: 'rgba(251, 182, 217, 0.50)',
    fontWeight: '600',
    sectionHeaderBgByLevel: [
      'rgba(96, 165, 250, 0.25)',
      'rgba(96, 165, 250, 0.19)',
      'rgba(96, 165, 250, 0.15)',
      'rgba(96, 165, 250, 0.11)',
      'rgba(96, 165, 250, 0.08)',
    ],
  },
};

export function getBaseHighlightStyle(preset: VisualPreset): HighlightStyle {
  return STYLE_BY_PRESET[preset];
}

export function resolveHighlightStyle(preset: VisualPreset, override: QmdColorOverrides): HighlightStyle {
  const baseStyle = getBaseHighlightStyle(preset);
  return {
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
    sectionLabelDefColor: override.sectionLabelDefColor ?? override.sectionRefColor ?? override.labelDefColor ?? baseStyle.sectionLabelDefColor,
    figureLabelDefColor: override.figureLabelDefColor ?? override.figureRefColor ?? override.labelDefColor ?? baseStyle.figureLabelDefColor,
    tableLabelDefColor: override.tableLabelDefColor ?? override.tableRefColor ?? override.labelDefColor ?? baseStyle.tableLabelDefColor,
    listingLabelDefColor: override.listingLabelDefColor ?? override.listingRefColor ?? override.labelDefColor ?? baseStyle.listingLabelDefColor,
    equationLabelDefColor: override.equationLabelDefColor ?? override.equationRefColor ?? override.labelDefColor ?? baseStyle.equationLabelDefColor,
    labelDefColor: override.labelDefColor ?? baseStyle.labelDefColor,
    divFenceColor: override.divFenceColor ?? baseStyle.divFenceColor,
    inlinePythonColor: override.inlinePythonColor ?? baseStyle.inlinePythonColor,
    inlinePythonBg: override.inlinePythonBg ?? baseStyle.inlinePythonBg,
    inlinePythonKeywordColor: baseStyle.inlinePythonKeywordColor,
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
}
