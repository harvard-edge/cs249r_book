import * as vscode from 'vscode';

interface FenceState {
  start: number;
  marker: string;
}

interface DivState {
  start: number;
  markerLength: number;
}

interface LatexEnvState {
  start: number;
  name: string;
}

export class QmdFoldingProvider implements vscode.FoldingRangeProvider {
  provideFoldingRanges(
    document: vscode.TextDocument,
    _context: vscode.FoldingContext,
    _token: vscode.CancellationToken,
  ): vscode.ProviderResult<vscode.FoldingRange[]> {
    const ranges: vscode.FoldingRange[] = [];
    const fenceStack: FenceState[] = [];
    const divStack: DivState[] = [];
    const latexStack: LatexEnvState[] = [];

    for (let line = 0; line < document.lineCount; line++) {
      const text = document.lineAt(line).text;

      // Handle fenced code blocks (``` / ~~~).
      const fenceMatch = text.match(/^\s*(```+|~~~+)/);
      if (fenceMatch) {
        const marker = fenceMatch[1];
        const top = fenceStack[fenceStack.length - 1];
        if (top && top.marker === marker) {
          fenceStack.pop();
          if (line > top.start) {
            ranges.push(new vscode.FoldingRange(top.start, line, vscode.FoldingRangeKind.Region));
          }
        } else {
          fenceStack.push({ start: line, marker });
        }
        continue;
      }

      const inFence = fenceStack.length > 0;
      if (inFence) {
        continue;
      }

      // Handle Quarto/Pandoc div fences (:::/::::).
      const divMatch = text.match(/^(\s*)(:{3,})(.*)$/);
      if (divMatch) {
        const markerLength = divMatch[2].length;
        const trailing = divMatch[3].trim();
        const isClose = trailing.length === 0;

        if (isClose) {
          let matchedIndex = -1;
          for (let i = divStack.length - 1; i >= 0; i--) {
            if (divStack[i].markerLength === markerLength) {
              matchedIndex = i;
              break;
            }
          }

          if (matchedIndex >= 0) {
            const [open] = divStack.splice(matchedIndex, 1);
            if (line > open.start) {
              ranges.push(new vscode.FoldingRange(open.start, line, vscode.FoldingRangeKind.Region));
            }
          } else if (divStack.length > 0) {
            const open = divStack.pop();
            if (open && line > open.start) {
              ranges.push(new vscode.FoldingRange(open.start, line, vscode.FoldingRangeKind.Region));
            }
          }
        } else {
          divStack.push({ start: line, markerLength });
        }
        continue;
      }

      // Handle raw LaTeX environments such as tikzpicture.
      const beginMatch = text.match(/^\s*\\begin\{([A-Za-z*@]+)\}/);
      if (beginMatch) {
        latexStack.push({ start: line, name: beginMatch[1] });
        continue;
      }

      const endMatch = text.match(/^\s*\\end\{([A-Za-z*@]+)\}/);
      if (endMatch) {
        const envName = endMatch[1];
        for (let i = latexStack.length - 1; i >= 0; i--) {
          if (latexStack[i].name === envName) {
            const [open] = latexStack.splice(i, 1);
            if (line > open.start) {
              ranges.push(new vscode.FoldingRange(open.start, line, vscode.FoldingRangeKind.Region));
            }
            break;
          }
        }
      }
    }

    return ranges;
  }
}
