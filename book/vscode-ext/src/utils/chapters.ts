import * as fs from 'fs';
import * as path from 'path';
import { ChapterInfo, VolumeId, VolumeInfo } from '../types';

const EXCLUDED_DIRS = new Set(['frontmatter', 'backmatter', 'parts', 'glossary']);

function toDisplayName(dirName: string): string {
  return dirName
    .split('_')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

export function discoverChapters(repoRoot: string): VolumeInfo[] {
  const volumes: VolumeInfo[] = [];

  for (const vol of ['vol1', 'vol2'] as VolumeId[]) {
    const contentsDir = path.join(repoRoot, 'book', 'quarto', 'contents', vol);
    if (!fs.existsSync(contentsDir)) { continue; }

    const entries = fs.readdirSync(contentsDir, { withFileTypes: true });
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
      .sort((a, b) => a.displayName.localeCompare(b.displayName));

    volumes.push({
      id: vol,
      label: vol === 'vol1' ? 'Volume I' : 'Volume II',
      chapters,
    });
  }

  return volumes;
}
