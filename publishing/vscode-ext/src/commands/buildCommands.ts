import * as vscode from 'vscode';
import * as path from 'path';
import { VolumeId, BuildFormat } from '../types';
import { discoverChapters } from '../utils/chapters';
import { getRepoRoot, parseQmdFile } from '../utils/workspace';
import { runInVisibleTerminal } from '../utils/terminal';
import { getQuartoResetAllFormatsCommand, withQuartoResetPrefix } from '../utils/quartoConfigReset';
import { showBuildManifest } from '../utils/buildManifest';

/** Map volume → PDF filename (derived from book title in Quarto config). */
const PDF_FILENAMES: Record<VolumeId, string> = {
  vol1: 'Introduction-to-Machine-Learning-Systems.pdf',
  vol2: 'Advanced-Machine-Learning-Systems.pdf',
};

/**
 * Get the expected PDF path for a given volume (may or may not exist yet).
 */
function getPdfPath(repoRoot: string, vol: VolumeId): string {
  return path.join(repoRoot, 'book', 'quarto', '_build', `pdf-${vol}`, PDF_FILENAMES[vol]);
}

/**
 * Run a PDF build in the visible terminal and watch for the output PDF.
 * When the PDF file is created or modified, automatically open it in VS Code.
 */
function runPdfBuildAndOpen(command: string, repoRoot: string, vol: VolumeId, label: string): void {
  const pdfPath = getPdfPath(repoRoot, vol);
  const pdfUri = vscode.Uri.file(pdfPath);
  const pdfDir = path.dirname(pdfPath);

  // Watch for the PDF file to be created or changed
  const pattern = new vscode.RelativePattern(pdfDir, path.basename(pdfPath));
  const watcher = vscode.workspace.createFileSystemWatcher(pattern, false, false, true);

  const openAndDispose = (): void => {
    watcher.dispose();
    void vscode.commands.executeCommand('vscode.open', pdfUri);
  };

  watcher.onDidCreate(openAndDispose);
  watcher.onDidChange(openAndDispose);

  // Safety: dispose the watcher after 10 minutes to avoid leaks on failed builds
  const timeout = setTimeout(() => watcher.dispose(), 10 * 60 * 1000);
  watcher.onDidCreate(() => clearTimeout(timeout));
  watcher.onDidChange(() => clearTimeout(timeout));

  // Run the build in the visible terminal as usual
  runInVisibleTerminal(command, repoRoot, label);
}

export function registerBuildCommands(context: vscode.ExtensionContext): void {
  const root = getRepoRoot();
  if (!root) { return; }

  // Reset all Quarto configs (uncomment all chapter/appendix entries in PDF, HTML, EPUB)
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.resetQuartoConfig', () => {
      runInVisibleTerminal(
        getQuartoResetAllFormatsCommand(),
        root,
        'Reset Quarto config (all formats)',
      );
    }),
  );

  // Chapter-level builds (reset config for this format/volume, then build)
  for (const fmt of ['Html', 'Pdf', 'Epub'] as const) {
    const fmtLower = fmt.toLowerCase() as BuildFormat;
    context.subscriptions.push(
      vscode.commands.registerCommand(`mlsysbook.buildChapter${fmt}`, (vol: VolumeId, chapter: string) => {
        const buildCmd = `./book/binder build ${fmtLower} ${chapter} --${vol} -v`;
        const label = `Build Chapter ${fmt.toUpperCase()} (${vol}/${chapter})`;
        const fullCmd = withQuartoResetPrefix(fmtLower, vol, buildCmd);
        if (fmtLower === 'pdf') {
          runPdfBuildAndOpen(fullCmd, root, vol, label);
        } else {
          runInVisibleTerminal(fullCmd, root, label);
        }
      })
    );
  }

  // Preview (no config reset; preview uses current config)
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.previewChapter', (vol: VolumeId, chapter: string) => {
      runInVisibleTerminal(
        `./book/binder preview ${vol}/${chapter}`,
        root,
        `Preview Chapter (${vol}/${chapter})`,
      );
    })
  );

  // Volume-level builds (reset config for this format/volume, then build)
  for (const fmt of ['Html', 'Pdf', 'Epub'] as const) {
    const fmtLower = fmt.toLowerCase() as BuildFormat;
    context.subscriptions.push(
      vscode.commands.registerCommand(`mlsysbook.buildVolume${fmt}`, (vol: VolumeId) => {
        const buildCmd = `./book/binder build ${fmtLower} --${vol} -v`;
        const label = `Build Volume ${fmt.toUpperCase()} (${vol})`;
        const fullCmd = withQuartoResetPrefix(fmtLower, vol, buildCmd);
        showBuildManifest({ repoRoot: root, vol, format: fmtLower, mode: 'sequential', command: fullCmd });
        if (fmtLower === 'pdf') {
          runPdfBuildAndOpen(fullCmd, root, vol, label);
        } else {
          runInVisibleTerminal(fullCmd, root, label);
        }
      })
    );
  }

  // Build Full Volume... (format picker)
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.buildFullVolume', async (vol: VolumeId) => {
      const fmtPick = await vscode.window.showQuickPick(
        [
          { label: 'HTML', id: 'html' as BuildFormat },
          { label: 'PDF', id: 'pdf' as BuildFormat },
          { label: 'EPUB', id: 'epub' as BuildFormat },
        ],
        { placeHolder: 'Select format' },
      );
      if (!fmtPick) { return; }
      const fmtLower = fmtPick.id;
      const buildCmd = `./book/binder build ${fmtLower} --${vol} -v`;
      const label = `Build Full Volume ${fmtLower.toUpperCase()} (${vol})`;
      const fullCmd = withQuartoResetPrefix(fmtLower, vol, buildCmd);
      showBuildManifest({ repoRoot: root, vol, format: fmtLower, mode: 'sequential', command: fullCmd });
      if (fmtLower === 'pdf') {
        runPdfBuildAndOpen(fullCmd, root, vol, label);
      } else {
        runInVisibleTerminal(fullCmd, root, label);
      }
    })
  );

  // Build Chapters... (multi-select chapters, then format)
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.buildSelectedChapters', async (vol: VolumeId) => {
      const volumes = discoverChapters(root);
      const volume = volumes.find(v => v.id === vol);
      if (!volume || volume.chapters.length === 0) {
        vscode.window.showWarningMessage(`No chapters found for ${vol}.`);
        return;
      }
      const chapterPicks = await vscode.window.showQuickPick(
        volume.chapters.map(ch => ({
          label: ch.displayName,
          description: ch.name,
          chapter: ch.name,
        })),
        {
          placeHolder: 'Select chapters to build (multi-select)',
          canPickMany: true,
          matchOnDescription: true,
        },
      );
      if (!chapterPicks || chapterPicks.length === 0) { return; }
      const chapters = chapterPicks.map(p => p.chapter);
      const fmtPick = await vscode.window.showQuickPick(
        [
          { label: 'HTML', id: 'html' as BuildFormat },
          { label: 'PDF', id: 'pdf' as BuildFormat },
          { label: 'EPUB', id: 'epub' as BuildFormat },
        ],
        { placeHolder: 'Select format' },
      );
      if (!fmtPick) { return; }
      const fmtLower = fmtPick.id;
      const chapterList = chapters.join(',');
      const buildCmd = `./book/binder build ${fmtLower} ${chapterList} --${vol} -v`;
      const label = `Build Chapters ${fmtLower.toUpperCase()} (${vol}): ${chapters.length} chapter(s)`;
      const fullCmd = withQuartoResetPrefix(fmtLower, vol, buildCmd);
      showBuildManifest({ repoRoot: root, vol, format: fmtLower, mode: 'sequential', command: fullCmd });
      if (fmtLower === 'pdf') {
        runPdfBuildAndOpen(fullCmd, root, vol, label);
      } else {
        runInVisibleTerminal(fullCmd, root, label);
      }
    })
  );

  // Quick Build Current Chapter (PDF) - from currently open .qmd file
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.quickBuildCurrentChapterPdf', () => {
      const uri = vscode.window.activeTextEditor?.document.uri;
      if (!uri || !uri.fsPath.endsWith('.qmd')) {
        vscode.window.showWarningMessage('Open a chapter .qmd file to run quick chapter PDF build.');
        return;
      }
      const parsed = parseQmdFile(uri);
      if (!parsed) {
        vscode.window.showWarningMessage('Could not determine volume/chapter for active file.');
        return;
      }
      const buildCmd = `./book/binder build pdf ${parsed.chapter} --${parsed.volume} -v`;
      const fullCmd = withQuartoResetPrefix('pdf', parsed.volume, buildCmd);
      runPdfBuildAndOpen(fullCmd, root, parsed.volume, `Quick Chapter PDF (${parsed.volume}/${parsed.chapter})`);
    })
  );
}
