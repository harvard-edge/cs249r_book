import * as vscode from 'vscode';

const LABEL_PATTERN = '(sec|fig|tbl|eq|lst)-[A-Za-z0-9:_-]+';

function getLabelAtCursor(editor: vscode.TextEditor): string | undefined {
  const lineText = editor.document.lineAt(editor.selection.active.line).text;
  const cursorCol = editor.selection.active.character;
  const regex = new RegExp(`(@(${LABEL_PATTERN}))|(\\{#(${LABEL_PATTERN})\\})`, 'g');
  let match: RegExpExecArray | null;
  while ((match = regex.exec(lineText)) !== null) {
    const full = match[0];
    const start = match.index;
    const end = start + full.length;
    if (cursorCol < start || cursorCol > end) {
      continue;
    }
    if (full.startsWith('@')) {
      return full.slice(1);
    }
    if (full.startsWith('{#') && full.endsWith('}')) {
      return full.slice(2, -1);
    }
  }
  return undefined;
}

function escapeRegex(input: string): string {
  return input.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export async function renameLabelAcrossWorkspace(): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor || !editor.document.uri.fsPath.endsWith('.qmd')) {
    vscode.window.showWarningMessage('Open a .qmd file and place cursor on a label/reference to rename.');
    return;
  }

  const current = getLabelAtCursor(editor);
  if (!current) {
    vscode.window.showWarningMessage('No section/figure/table/equation/listing label found at cursor.');
    return;
  }

  const prefix = current.split('-')[0];
  const suggested = `${prefix}-`;
  const next = await vscode.window.showInputBox({
    prompt: `Rename label "${current}" to`,
    value: current,
    validateInput: value => {
      if (!new RegExp(`^${prefix}-[A-Za-z0-9:_-]+$`).test(value)) {
        return `Label must keep "${prefix}-" prefix and contain only letters, numbers, :, _, -`;
      }
      return undefined;
    },
    valueSelection: [suggested.length, current.length],
  });
  if (!next || next === current) {
    return;
  }

  const files = await vscode.workspace.findFiles('**/*.qmd', '**/node_modules/**');
  const edit = new vscode.WorkspaceEdit();
  let replacementCount = 0;

  const refRegex = new RegExp(`@${escapeRegex(current)}\\b`, 'g');
  const defRegex = new RegExp(`\\{#${escapeRegex(current)}\\}`, 'g');

  for (const uri of files) {
    const document = await vscode.workspace.openTextDocument(uri);
    const text = document.getText();

    const applyRegex = (regex: RegExp, replacement: string): void => {
      let match: RegExpExecArray | null;
      while ((match = regex.exec(text)) !== null) {
        const start = document.positionAt(match.index);
        const end = document.positionAt(match.index + match[0].length);
        edit.replace(uri, new vscode.Range(start, end), replacement);
        replacementCount += 1;
      }
    };

    applyRegex(refRegex, `@${next}`);
    applyRegex(defRegex, `{#${next}}`);
  }

  if (replacementCount === 0) {
    vscode.window.showInformationMessage('No references found to rename.');
    return;
  }

  const applied = await vscode.workspace.applyEdit(edit);
  if (!applied) {
    vscode.window.showWarningMessage('Failed to apply label rename edits.');
    return;
  }

  vscode.window.showInformationMessage(`Renamed label ${current} -> ${next} (${replacementCount} updates).`);
}
