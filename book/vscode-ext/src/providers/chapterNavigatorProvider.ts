import * as vscode from 'vscode';

type NavigatorEntryKind = 'figure' | 'table' | 'listing' | 'equation' | 'callout';
type NavigatorVisibleEntryKind = 'figures' | 'tables' | 'listings' | 'equations' | 'callouts';

interface NavigatorEntry {
  kind: NavigatorEntryKind;
  id: string;
  line: number;
  preview: string;
}

interface NavigatorSection {
  id: string;
  level: number;
  title: string;
  line: number;
  parentId?: string;
  childSectionIds: string[];
  entries: NavigatorEntry[];
}

class NavigatorSectionItem extends vscode.TreeItem {
  constructor(
    uri: vscode.Uri,
    section: NavigatorSection,
    collapsibleState: vscode.TreeItemCollapsibleState,
  ) {
    const entryCount = section.entries.length;
    super(section.title, collapsibleState);
    this.sectionId = section.id;
    this.id = section.id;
    const depthLabel = section.level === 0 ? 'preamble' : `h${section.level}`;
    const listingCount = section.entries.filter(entry => entry.kind === 'listing').length;
    const figureCount = section.entries.filter(entry => entry.kind === 'figure').length;
    const tableCount = section.entries.filter(entry => entry.kind === 'table').length;
    const counts: string[] = [];
    if (figureCount > 0) { counts.push(`${figureCount} fig`); }
    if (tableCount > 0) { counts.push(`${tableCount} tbl`); }
    if (listingCount > 0) { counts.push(`${listingCount} code`); }
    const calloutCount = section.entries.filter(entry => entry.kind === 'callout').length;
    if (calloutCount > 0) { counts.push(`${calloutCount} callout`); }
    const countsText = counts.length > 0 ? ` · ${counts.join(' · ')}` : '';
    this.description = `${depthLabel} · L${section.line + 1} · ${entryCount} items${countsText}`;
    this.contextValue = 'navigator-section';
    this.iconPath = new vscode.ThemeIcon(section.level === 0 ? 'note' : 'list-tree');
    this.command = {
      command: 'mlsysbook.openNavigatorLocation',
      title: 'Open Section',
      arguments: [uri, section.line],
    };
  }

  readonly sectionId: string;
}

class NavigatorEntryItem extends vscode.TreeItem {
  constructor(uri: vscode.Uri, entry: NavigatorEntry, clickToOpen: boolean) {
    super(`${entry.id}  ·  L${entry.line + 1}`, vscode.TreeItemCollapsibleState.None);
    this.description = entry.preview;
    this.tooltip = `${entry.id}\n${entry.preview}`;
    const icon = entry.kind === 'figure'
      ? 'symbol-field'
      : entry.kind === 'table'
        ? 'table'
        : entry.kind === 'listing'
          ? 'code'
          : entry.kind === 'equation'
            ? 'symbol-operator'
            : 'comment-discussion';
    this.iconPath = new vscode.ThemeIcon(icon);
    if (clickToOpen) {
      this.command = {
        command: 'mlsysbook.openNavigatorLocation',
        title: 'Open Location',
        arguments: [uri, entry.line],
      };
    }
    this.contextValue = `navigator-entry-${entry.kind}`;
  }
}

type TreeNode = NavigatorSectionItem | NavigatorEntryItem;

export class ChapterNavigatorProvider implements vscode.TreeDataProvider<TreeNode> {
  private readonly _onDidChangeTreeData = new vscode.EventEmitter<TreeNode | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private sourceUri: vscode.Uri | undefined;
  private sections = new Map<string, NavigatorSection>();
  private topLevelSectionIds: string[] = [];
  private sectionItems = new Map<string, NavigatorSectionItem>();
  private forceExpanded = false;

  private isQmdDocument(document: vscode.TextDocument): boolean {
    return document.uri.fsPath.endsWith('.qmd');
  }

  getTreeItem(element: TreeNode): vscode.TreeItem {
    return element;
  }

  private getVisibleEntryKinds(): Set<NavigatorVisibleEntryKind> {
    const kinds = vscode.workspace
      .getConfiguration('mlsysbook')
      .get<NavigatorVisibleEntryKind[]>(
        'navigatorVisibleEntryKinds',
        ['figures', 'tables', 'listings', 'equations', 'callouts'],
      );
    return new Set(kinds);
  }

  private isEntryVisible(kind: NavigatorEntryKind): boolean {
    const map: Record<NavigatorEntryKind, NavigatorVisibleEntryKind> = {
      figure: 'figures',
      table: 'tables',
      listing: 'listings',
      equation: 'equations',
      callout: 'callouts',
    };
    return this.getVisibleEntryKinds().has(map[kind]);
  }

  private getSectionItem(section: NavigatorSection): NavigatorSectionItem {
    const cached = this.sectionItems.get(section.id);
    if (cached) {
      return cached;
    }
    const state = this.forceExpanded
      ? vscode.TreeItemCollapsibleState.Expanded
      : vscode.TreeItemCollapsibleState.Collapsed;
    const created = new NavigatorSectionItem(this.sourceUri!, section, state);
    this.sectionItems.set(section.id, created);
    return created;
  }

  private shouldClickToOpen(): boolean {
    return vscode.workspace
      .getConfiguration('mlsysbook')
      .get<boolean>('navigatorClickToOpen', false);
  }

  getChildren(element?: TreeNode): TreeNode[] {
    if (!this.sourceUri) {
      return [];
    }

    if (!element) {
      return this.topLevelSectionIds
        .map(id => this.sections.get(id))
        .filter((section): section is NavigatorSection => Boolean(section))
        .map(section => this.getSectionItem(section));
    }

    if (element instanceof NavigatorSectionItem) {
      const section = this.sections.get(element.sectionId);
      if (!section) { return []; }

      type ChildRef =
        | { type: 'section'; line: number; section: NavigatorSection }
        | { type: 'entry'; line: number; entry: NavigatorEntry };

      const children: ChildRef[] = [];
      for (const childId of section.childSectionIds) {
        const child = this.sections.get(childId);
        if (child) {
          children.push({ type: 'section', line: child.line, section: child });
        }
      }
      for (const entryRef of section.entries) {
        if (!this.isEntryVisible(entryRef.kind)) {
          continue;
        }
        children.push({ type: 'entry', line: entryRef.line, entry: entryRef });
      }

      children.sort((a, b) => a.line - b.line);
      return children.map(child =>
        child.type === 'section'
          ? this.getSectionItem(child.section)
          : new NavigatorEntryItem(this.sourceUri!, child.entry, this.shouldClickToOpen())
      );
    }

    return [];
  }

  refreshFromEditor(editor: vscode.TextEditor | undefined): void {
    if (!editor || !this.isQmdDocument(editor.document)) {
      this.sourceUri = undefined;
      this.sections.clear();
      this.topLevelSectionIds = [];
      this.sectionItems.clear();
      this._onDidChangeTreeData.fire(undefined);
      return;
    }

    this.sourceUri = editor.document.uri;
    this.scanDocument(editor.document);
    this._onDidChangeTreeData.fire(undefined);
  }

  refreshView(): void {
    this._onDidChangeTreeData.fire(undefined);
  }

  expandAll(): void {
    this.forceExpanded = true;
    this.sectionItems.clear();
    this._onDidChangeTreeData.fire(undefined);
  }

  collapseAll(): void {
    this.forceExpanded = false;
    this.sectionItems.clear();
    this._onDidChangeTreeData.fire(undefined);
  }

  refreshFromDocument(document: vscode.TextDocument): void {
    const active = vscode.window.activeTextEditor;
    if (!active || active.document.uri.toString() !== document.uri.toString()) {
      return;
    }
    this.refreshFromEditor(active);
  }

  private scanDocument(document: vscode.TextDocument): void {
    const sections = new Map<string, NavigatorSection>();
    const topLevelSectionIds: string[] = [];
    const rootSection: NavigatorSection = {
      id: 'root',
      level: 0,
      title: 'Document Preamble',
      line: 0,
      childSectionIds: [],
      entries: [],
    };
    sections.set(rootSection.id, rootSection);

    const sectionStack: NavigatorSection[] = [rootSection];
    const figureIds = new Set<string>();
    const tableIds = new Set<string>();
    const equationIds = new Set<string>();

    const headerRegex = /^(#{2,6})\s+(.+?)\s*$/;
    const figureRegex = /#(fig-[A-Za-z0-9_-]+)/g;
    const tableRegex = /#(tbl-[A-Za-z0-9_-]+)/g;
    const listingRegex = /#(lst-[A-Za-z0-9_-]+)/g;
    const equationRegex = /#(eq-[A-Za-z0-9_-]+)/g;
    const listingYamlRegex = /#\|\s*(?:label|lst-label):\s*(lst-[A-Za-z0-9_:-]+)/g;
    const equationYamlRegex = /#\|\s*label:\s*(eq-[A-Za-z0-9_:-]+)/g;
    const calloutStartRegex = /^\s*:{3,}\s*\{[^}]*\.callout-([A-Za-z0-9_-]+)[^}]*\}/;
    const calloutTitleRegex = /title\s*=\s*"([^"]+)"/;
    const fenceRegex = /^\s*(```|~~~)\s*(.*)$/;
    const listingIds = new Set<string>();
    let inFence = false;
    let currentFenceMarker = '';

    const addEntry = (entry: NavigatorEntry): void => {
      const currentSection = sectionStack[sectionStack.length - 1] ?? rootSection;
      currentSection.entries.push(entry);
    };

    for (let line = 0; line < document.lineCount; line++) {
      const text = document.lineAt(line).text;
      const preview = text.trim().slice(0, 90);

      const fenceMatch = text.match(fenceRegex);
      if (fenceMatch) {
        const marker = fenceMatch[1];
        if (!inFence) {
          inFence = true;
          currentFenceMarker = marker;
        } else if (marker === currentFenceMarker) {
          inFence = false;
          currentFenceMarker = '';
        }
      }
      if (inFence) {
        continue;
      }

      const headerMatch = text.match(headerRegex);
      if (headerMatch) {
        const level = headerMatch[1].length;
        const title = headerMatch[2].replace(/\s*\{[^}]+\}\s*$/, '').trim();
        while ((sectionStack[sectionStack.length - 1]?.level ?? 0) >= level) {
          sectionStack.pop();
        }
        const parent = sectionStack[sectionStack.length - 1] ?? rootSection;
        const section: NavigatorSection = {
          id: `sec-${line}`,
          level,
          title,
          line,
          parentId: parent.id,
          childSectionIds: [],
          entries: [],
        };
        parent.childSectionIds.push(section.id);
        sections.set(section.id, section);
        sectionStack.push(section);
      }

      let match: RegExpExecArray | null;
      while ((match = figureRegex.exec(text)) !== null) {
        const id = match[1];
        if (!figureIds.has(id)) {
          figureIds.add(id);
          addEntry({ kind: 'figure', id, line, preview });
        }
      }
      figureRegex.lastIndex = 0;

      while ((match = tableRegex.exec(text)) !== null) {
        const id = match[1];
        if (!tableIds.has(id)) {
          tableIds.add(id);
          addEntry({ kind: 'table', id, line, preview });
        }
      }
      tableRegex.lastIndex = 0;

      while ((match = listingRegex.exec(text)) !== null) {
        const id = match[1];
        if (!listingIds.has(id)) {
          listingIds.add(id);
          addEntry({ kind: 'listing', id, line, preview });
        }
      }
      listingRegex.lastIndex = 0;

      while ((match = listingYamlRegex.exec(text)) !== null) {
        const id = match[1];
        if (!listingIds.has(id)) {
          listingIds.add(id);
          addEntry({ kind: 'listing', id, line, preview });
        }
      }
      listingYamlRegex.lastIndex = 0;

      while ((match = equationRegex.exec(text)) !== null) {
        const id = match[1];
        if (!equationIds.has(id)) {
          equationIds.add(id);
          addEntry({ kind: 'equation', id, line, preview });
        }
      }
      equationRegex.lastIndex = 0;

      while ((match = equationYamlRegex.exec(text)) !== null) {
        const id = match[1];
        if (!equationIds.has(id)) {
          equationIds.add(id);
          addEntry({ kind: 'equation', id, line, preview });
        }
      }
      equationYamlRegex.lastIndex = 0;

      const calloutMatch = text.match(calloutStartRegex);
      if (calloutMatch) {
        const calloutType = calloutMatch[1];
        const titleMatch = text.match(calloutTitleRegex);
        const title = titleMatch?.[1]?.trim();
        const id = title && title.length > 0
          ? `callout-${calloutType}: ${title}`
          : `callout-${calloutType}`;
        addEntry({ kind: 'callout', id, line, preview });
      }
    }

    if (rootSection.childSectionIds.length === 0 && rootSection.entries.length > 0) {
      topLevelSectionIds.push(rootSection.id);
    } else {
      topLevelSectionIds.push(...rootSection.childSectionIds);
      if (rootSection.entries.length > 0) {
        topLevelSectionIds.unshift(rootSection.id);
      }
    }

    this.sections = sections;
    this.topLevelSectionIds = topLevelSectionIds;
    this.sectionItems.clear();
  }

  getSectionItemForLine(line: number): NavigatorSectionItem | undefined {
    if (!this.sourceUri) { return undefined; }
    const sections = [...this.sections.values()]
      .filter(section => section.level > 0)
      .sort((a, b) => a.line - b.line);
    if (sections.length === 0) { return undefined; }

    let candidate: NavigatorSection | undefined;
    for (const section of sections) {
      if (section.line <= line) {
        candidate = section;
      } else {
        break;
      }
    }

    if (!candidate) {
      return undefined;
    }
    return this.getSectionItem(candidate);
  }
}
