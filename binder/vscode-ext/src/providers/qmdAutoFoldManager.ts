import * as vscode from 'vscode';

type AutoFoldKind = 'tikz' | 'fencedCode' | 'divs';

function isQmdEditor(editor: vscode.TextEditor | undefined): editor is vscode.TextEditor {
  return Boolean(editor && editor.document.uri.fsPath.endsWith('.qmd'));
}

export class QmdAutoFoldManager implements vscode.Disposable {
  private readonly disposables: vscode.Disposable[] = [];
  private readonly foldedDocs = new Set<string>();
  private timer: NodeJS.Timeout | undefined;

  start(): void {
    this.apply(vscode.window.activeTextEditor);
    this.disposables.push(
      vscode.window.onDidChangeActiveTextEditor(editor => this.apply(editor)),
      vscode.workspace.onDidOpenTextDocument(() => this.apply(vscode.window.activeTextEditor)),
      vscode.workspace.onDidChangeConfiguration(event => {
        if (
          event.affectsConfiguration('mlsysbook.autoFoldOnOpen')
          || event.affectsConfiguration('mlsysbook.autoFoldKinds')
        ) {
          this.foldedDocs.clear();
          this.apply(vscode.window.activeTextEditor);
        }
      }),
    );
  }

  private isEnabled(): boolean {
    return vscode.workspace
      .getConfiguration('mlsysbook')
      .get<boolean>('autoFoldOnOpen', true);
  }

  private getKinds(): Set<AutoFoldKind> {
    const raw = vscode.workspace
      .getConfiguration('mlsysbook')
      .get<AutoFoldKind[]>('autoFoldKinds', ['tikz']);
    return new Set(raw);
  }

  private apply(editor: vscode.TextEditor | undefined): void {
    if (!isQmdEditor(editor) || !this.isEnabled()) {
      return;
    }
    const key = editor.document.uri.toString();
    if (this.foldedDocs.has(key)) {
      return;
    }
    if (this.timer) {
      clearTimeout(this.timer);
    }
    this.timer = setTimeout(() => {
      void this.fold(editor).then(() => {
        this.foldedDocs.add(key);
      });
    }, 120);
  }

  private async fold(editor: vscode.TextEditor): Promise<void> {
    const kinds = this.getKinds();
    const doc = editor.document;
    const selectionLines = new Set<number>();
    const fenceStack: Array<{ start: number; marker: string; meta: string; hasTikz: boolean }> = [];

    const fenceRegex = /^\s*(```+|~~~+)\s*(.*)$/;
    const divOpenRegex = /^\s*:{3,}\s*\{.+\}\s*$/;
    const divCloseRegex = /^\s*:{3,}\s*$/;
    const tikzBeginRegex = /^\s*\\begin\{tikzpicture\}/;
    const tikzEndRegex = /^\s*\\end\{tikzpicture\}/;
    const rawLatexFenceRegex = /\{=latex\}|\{latex\}|latex|tex|tikz/i;
    const tikzMetaRegex = /tikz/i;
    const divStack: number[] = [];
    const tikzLatexStack: number[] = [];

    for (let line = 0; line < doc.lineCount; line++) {
      const text = doc.lineAt(line).text;

      const fenceMatch = text.match(fenceRegex);
      if (fenceMatch) {
        const marker = fenceMatch[1];
        const meta = (fenceMatch[2] ?? '').trim();
        const top = fenceStack[fenceStack.length - 1];
        if (top && top.marker === marker) {
          const closed = fenceStack.pop();
          if (closed) {
            const isTikzFence =
              tikzMetaRegex.test(closed.meta)
              || (rawLatexFenceRegex.test(closed.meta) && closed.hasTikz);
            if ((kinds.has('fencedCode') || (kinds.has('tikz') && isTikzFence)) && line > closed.start) {
              selectionLines.add(closed.start);
            }
          }
        } else {
          fenceStack.push({ start: line, marker, meta, hasTikz: false });
        }
        continue;
      }

      const inFence = fenceStack.length > 0;
      if (inFence) {
        if (tikzBeginRegex.test(text)) {
          const top = fenceStack[fenceStack.length - 1];
          top.hasTikz = true;
        }
        continue;
      }

      if (kinds.has('divs')) {
        if (divOpenRegex.test(text)) {
          divStack.push(line);
        } else if (divCloseRegex.test(text) && divStack.length > 0) {
          const open = divStack.pop();
          if (open !== undefined && line > open) {
            selectionLines.add(open);
          }
        }
      }

      if (kinds.has('tikz')) {
        if (tikzBeginRegex.test(text)) {
          tikzLatexStack.push(line);
        } else if (tikzEndRegex.test(text) && tikzLatexStack.length > 0) {
          const open = tikzLatexStack.pop();
          if (open !== undefined && line > open) {
            selectionLines.add(open);
          }
        }
      }
    }

    if (selectionLines.size === 0) {
      return;
    }

    await vscode.window.showTextDocument(doc, editor.viewColumn, false);
    await vscode.commands.executeCommand('editor.fold', {
      levels: 1,
      selectionLines: [...selectionLines].sort((a, b) => a - b),
    });
  }

  dispose(): void {
    if (this.timer) {
      clearTimeout(this.timer);
    }
    this.disposables.forEach(d => d.dispose());
  }
}
