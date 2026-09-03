import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { BuildFormat, ChapterInfo, ChapterOrderSource, VolumeId, VolumeInfo } from '../types';

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

/** Parsed entry from config: relative path (e.g. "frontmatter/dedication.qmd") and order index. */
interface ConfigEntry {
  relPath: string;
  order: number;
}

/**
 * Read all buildable file paths from YML configs (chapters, appendices, frontmatter, backmatter).
 * Returns entries in config order; duplicates are deduplicated by path (first occurrence wins).
 */
function readBuildablePathsFromConfig(
  repoRoot: string,
  vol: VolumeId,
  source: ChapterOrderSource,
): ConfigEntry[] {
  const configDir = path.join(repoRoot, 'book', 'quarto', 'config');
  const entries: ConfigEntry[] = [];
  const seen = new Set<string>();
  const chapterPathRegex = new RegExp(`contents/${vol}/([^\\s]+\\.qmd)`, 'g');

  for (const fileName of getConfigCandidates(vol, source)) {
    const configPath = path.join(configDir, fileName);
    if (!fs.existsSync(configPath)) {
      continue;
    }

    const text = fs.readFileSync(configPath, 'utf8');

    // PDF/book configs use "index.qmd" (no path); add volume-specific index
    const indexRelPath = `index.qmd`;
    if (!seen.has(indexRelPath) && /^\s+-\s+index\.qmd\s*$/m.test(text)) {
      const volIndexPath = `index.qmd`;
      const volIndexFull = path.join(repoRoot, 'book', 'quarto', 'contents', vol, 'index.qmd');
      if (fs.existsSync(volIndexFull)) {
        seen.add(volIndexPath);
        entries.push({ relPath: volIndexPath, order: entries.length });
      }
    }

    let match: RegExpExecArray | null;
    while ((match = chapterPathRegex.exec(text)) !== null) {
      const relPath = match[1];
      if (seen.has(relPath)) {
        continue;
      }
      seen.add(relPath);
      entries.push({ relPath, order: entries.length });
    }

    if (entries.length > 0) {
      return entries;
    }
  }

  return entries;
}

/**
 * Convert a config relative path (e.g. "frontmatter/dedication.qmd") to ChapterInfo.
 * Returns null if the file does not exist.
 */
function pathToChapterInfo(
  repoRoot: string,
  vol: VolumeId,
  relPath: string,
): ChapterInfo | null {
  const contentsDir = path.join(repoRoot, 'book', 'quarto', 'contents', vol);
  const fullPath = path.join(contentsDir, relPath);
  if (!fs.existsSync(fullPath)) {
    return null;
  }

  const segments = relPath.split('/');
  const fileName = segments[segments.length - 1];
  const name = fileName.replace(/\.qmd$/i, '');
  const dirPath =
    segments.length > 1
      ? path.join(contentsDir, ...segments.slice(0, -1))
      : contentsDir;

  return {
    name,
    volume: vol,
    dirPath,
    displayName: toDisplayName(name),
  };
}

/**
 * Returns all buildable paths for a given format by reading the YML config,
 * stripping comment markers so every entry — whether currently commented out
 * or active — is included in the correct book order.
 * The YML is the single source of truth: add a chapter there (even commented)
 * and it will appear in the manifest and be built after `binder reset`.
 */
export function getBuildEntriesForFormat(
  repoRoot: string,
  vol: VolumeId,
  format: BuildFormat,
): { configFile: string; relPaths: string[] } {
  const source = format as ChapterOrderSource;
  const entries = readBuildablePathsFromConfig(repoRoot, vol, source);
  const configFile = path.join('book', 'quarto', 'config', `_quarto-${format}-${vol}.yml`);
  return {
    configFile,
    relPaths: entries.map(e => e.relPath),
  };
}

export function discoverChapters(repoRoot: string): VolumeInfo[] {
  const volumes: VolumeInfo[] = [];
  const chapterOrderSource = vscode.workspace
    .getConfiguration('mlsysbook')
    .get<ChapterOrderSource>('chapterOrderSource', 'auto');

  for (const vol of ['vol1', 'vol2'] as VolumeId[]) {
    const contentsDir = path.join(repoRoot, 'book', 'quarto', 'contents', vol);
    if (!fs.existsSync(contentsDir)) {
      continue;
    }

    const configEntries = readBuildablePathsFromConfig(repoRoot, vol, chapterOrderSource);
    const orderByPath = new Map<string, number>(
      configEntries.map((e, i) => [e.relPath, i]),
    );

    const chapters: ChapterInfo[] = configEntries
      .map((e) => pathToChapterInfo(repoRoot, vol, e.relPath))
      .filter((ch): ch is ChapterInfo => ch !== null)
      .sort((a, b) => {
        const aPath = path.relative(contentsDir, path.join(a.dirPath, `${a.name}.qmd`));
        const bPath = path.relative(contentsDir, path.join(b.dirPath, `${b.name}.qmd`));
        const aOrder = orderByPath.get(aPath.replace(/\\/g, '/'));
        const bOrder = orderByPath.get(bPath.replace(/\\/g, '/'));
        if (aOrder !== undefined && bOrder !== undefined) {
          return aOrder - bOrder;
        }
        if (aOrder !== undefined) return -1;
        if (bOrder !== undefined) return 1;
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
