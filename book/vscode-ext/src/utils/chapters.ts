import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { ChapterInfo, ChapterOrderSource, VolumeId, VolumeInfo } from '../types';

const EXCLUDED_CHAPTER_DIRS = new Set(['frontmatter', 'backmatter', 'parts', 'glossary']);
const APPENDIX_FILE_REGEX = /^appendix[_-].+\.qmd$/i;

function toDisplayName(dirName: string): string {
  return dirName
    .split('_')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

function getConfigCandidates(vol: VolumeId, source: ChapterOrderSource): string[] {
  if (source === 'pdf') {
    return [`_quarto-pdf-${vol}.yml`];
  }
  if (source === 'epub') {
    return [`_quarto-epub-${vol}.yml`];
  }
  if (source === 'html') {
    return [`_quarto-html-${vol}.yml`];
  }
  if (source === 'pdfCopyedit') {
    return [`_quarto-pdf-${vol}-copyedit.yml`];
  }
  return [
    `_quarto-pdf-${vol}.yml`,
    `_quarto-epub-${vol}.yml`,
    `_quarto-html-${vol}.yml`,
    `_quarto-pdf-${vol}-copyedit.yml`,
  ];
}

function readChapterOrderFromConfig(repoRoot: string, vol: VolumeId, source: ChapterOrderSource): string[] {
  const configDir = path.join(repoRoot, 'book', 'quarto', 'config');
  const chapterOrderKeys: string[] = [];
  const seen = new Set<string>();
  const chapterPathRegex = new RegExp(`contents/${vol}/([^\\s]+\\.qmd)`, 'g');

  for (const fileName of getConfigCandidates(vol, source)) {
    const configPath = path.join(configDir, fileName);
    if (!fs.existsSync(configPath)) {
      continue;
    }

    const text = fs.readFileSync(configPath, 'utf8');
    let match: RegExpExecArray | null;
    while ((match = chapterPathRegex.exec(text)) !== null) {
      const chapterPath = match[1];
      const segments = chapterPath.split('/');
      if (segments.length < 2) {
        continue;
      }

      // Backmatter appendix files: contents/volX/backmatter/appendix_foo.qmd
      if (segments[0] === 'backmatter' && segments.length === 2) {
        const fileNameOnly = segments[1];
        if (!APPENDIX_FILE_REGEX.test(fileNameOnly)) {
          continue;
        }
        const appendixStem = fileNameOnly.replace(/\.qmd$/i, '');
        if (seen.has(appendixStem)) {
          continue;
        }
        seen.add(appendixStem);
        chapterOrderKeys.push(appendixStem);
        continue;
      }

      // Regular chapter entries: use the chapter file stem as command key.
      const dirName = segments[0];
      if (EXCLUDED_CHAPTER_DIRS.has(dirName)) {
        continue;
      }
      const fileNameOnly = segments[segments.length - 1];
      const chapterStem = fileNameOnly.replace(/\.qmd$/i, '');
      if (seen.has(chapterStem)) {
        continue;
      }
      seen.add(chapterStem);
      chapterOrderKeys.push(chapterStem);
    }

    if (chapterOrderKeys.length > 0) {
      return chapterOrderKeys;
    }
  }

  return chapterOrderKeys;
}

export function discoverChapters(repoRoot: string): VolumeInfo[] {
  const volumes: VolumeInfo[] = [];
  const chapterOrderSource = vscode.workspace
    .getConfiguration('mlsysbook')
    .get<ChapterOrderSource>('chapterOrderSource', 'auto');

  for (const vol of ['vol1', 'vol2'] as VolumeId[]) {
    const contentsDir = path.join(repoRoot, 'book', 'quarto', 'contents', vol);
    if (!fs.existsSync(contentsDir)) { continue; }

    const entries = fs.readdirSync(contentsDir, { withFileTypes: true });
    const chapterOrder = readChapterOrderFromConfig(repoRoot, vol, chapterOrderSource);
    const orderByName = new Map<string, number>(chapterOrder.map((name, index) => [name, index]));

    const regularChapters: ChapterInfo[] = entries
      .filter(e => e.isDirectory() && !EXCLUDED_CHAPTER_DIRS.has(e.name))
      .filter(e => {
        // Must contain a .qmd file to count as a chapter
        const dirPath = path.join(contentsDir, e.name);
        return fs.readdirSync(dirPath).some(f => f.endsWith('.qmd'));
      })
      .map(e => {
        const dirPath = path.join(contentsDir, e.name);
        const qmdFiles = fs.readdirSync(dirPath)
          .filter(f => f.endsWith('.qmd'))
          .sort();

        const preferredByConfig = qmdFiles.find(file => {
          const stem = file.replace(/\.qmd$/i, '');
          return orderByName.has(stem);
        });
        const preferredByDirName = qmdFiles.find(file => file.replace(/\.qmd$/i, '') === e.name);
        const preferredNonHidden = qmdFiles.find(file => !path.basename(file).startsWith('_'));
        const selected = preferredByConfig ?? preferredByDirName ?? preferredNonHidden ?? qmdFiles[0];
        const chapterStem = selected.replace(/\.qmd$/i, '');

        return {
          name: chapterStem,
          volume: vol,
          dirPath,
          displayName: toDisplayName(chapterStem),
        };
      });

    const appendixDir = path.join(contentsDir, 'backmatter');
    const appendixChapters: ChapterInfo[] = fs.existsSync(appendixDir)
      ? fs.readdirSync(appendixDir, { withFileTypes: true })
        .filter(entry => entry.isFile() && APPENDIX_FILE_REGEX.test(entry.name))
        .map(entry => {
          const appendixStem = entry.name.replace(/\.qmd$/i, '');
          return {
            name: appendixStem,
            volume: vol,
            dirPath: appendixDir,
            displayName: toDisplayName(appendixStem),
          };
        })
      : [];

    const chapters: ChapterInfo[] = [...regularChapters, ...appendixChapters]
      .sort((a, b) => {
        const aOrder = orderByName.get(a.name);
        const bOrder = orderByName.get(b.name);
        if (aOrder !== undefined && bOrder !== undefined) {
          return aOrder - bOrder;
        }
        if (aOrder !== undefined) {
          return -1;
        }
        if (bOrder !== undefined) {
          return 1;
        }
        return a.displayName.localeCompare(b.displayName);
      });

    volumes.push({
      id: vol,
      label: vol === 'vol1' ? 'Volume I' : 'Volume II',
      chapters,
    });
  }

  return volumes;
}
