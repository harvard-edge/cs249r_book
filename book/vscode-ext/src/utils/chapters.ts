import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { ChapterInfo, ChapterOrderSource, VolumeId, VolumeInfo } from '../types';

const EXCLUDED_DIRS = new Set(['frontmatter', 'backmatter', 'parts', 'glossary']);

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
  const chapterDirNames: string[] = [];
  const seen = new Set<string>();
  const chapterRegex = new RegExp(`contents/${vol}/([^/]+)/[^\\s]+\\.qmd`, 'g');

  for (const fileName of getConfigCandidates(vol, source)) {
    const configPath = path.join(configDir, fileName);
    if (!fs.existsSync(configPath)) {
      continue;
    }

    const text = fs.readFileSync(configPath, 'utf8');
    let match: RegExpExecArray | null;
    while ((match = chapterRegex.exec(text)) !== null) {
      const dirName = match[1];
      if (EXCLUDED_DIRS.has(dirName) || seen.has(dirName)) {
        continue;
      }
      seen.add(dirName);
      chapterDirNames.push(dirName);
    }

    if (chapterDirNames.length > 0) {
      return chapterDirNames;
    }
  }

  return chapterDirNames;
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

    const chapters: ChapterInfo[] = entries
      .filter(e => e.isDirectory() && !EXCLUDED_DIRS.has(e.name))
      .filter(e => {
        // Must contain a .qmd file to count as a chapter
        const dirPath = path.join(contentsDir, e.name);
        return fs.readdirSync(dirPath).some(f => f.endsWith('.qmd'));
      })
      .map(e => ({
        name: e.name,
        volume: vol,
        dirPath: path.join(contentsDir, e.name),
        displayName: toDisplayName(e.name),
      }))
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
