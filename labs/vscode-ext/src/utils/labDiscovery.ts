import * as path from 'path';
import * as fs from 'fs';
import { LabInfo, LabStatus } from '../types';

/** Convert a filename slug to a display title */
function slugToTitle(slug: string): string {
  // lab_01_ml_intro → ML Intro
  return slug
    .replace(/^lab_\d+_/, '')
    .split('_')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

/** Discover all labs for a given volume */
function discoverVolumeLabs(projectRoot: string, volume: number): LabInfo[] {
  const labDir = path.join(projectRoot, 'labs', `vol${volume}`);
  const planDir = path.join(projectRoot, 'labs', 'plans', `vol${volume}`);

  const labFiles = new Map<string, string>();
  const planFiles = new Map<string, string>();

  // Scan lab .py files
  if (fs.existsSync(labDir)) {
    for (const file of fs.readdirSync(labDir)) {
      if (file.endsWith('.py') && file.startsWith('lab_')) {
        const slug = file.replace('.py', '');
        labFiles.set(slug, path.join(labDir, file));
      }
    }
  }

  // Scan plan .md files
  if (fs.existsSync(planDir)) {
    for (const file of fs.readdirSync(planDir)) {
      if (file.endsWith('.md') && file.startsWith('lab_')) {
        const slug = file.replace('.md', '');
        planFiles.set(slug, path.join(planDir, file));
      }
    }
  }

  // Merge: union of all known lab slugs
  const allSlugs = new Set([...labFiles.keys(), ...planFiles.keys()]);
  const labs: LabInfo[] = [];

  for (const slug of allSlugs) {
    const match = slug.match(/^lab_(\d+)/);
    if (!match) { continue; }

    const number = match[1];
    const hasLab = labFiles.has(slug);
    const hasPlan = planFiles.has(slug);

    let status: LabStatus = 'missing';
    if (hasLab) { status = 'implemented'; }
    else if (hasPlan) { status = 'planned'; }

    labs.push({
      number,
      title: slugToTitle(slug),
      volume,
      labPath: labFiles.get(slug),
      planPath: planFiles.get(slug),
      status,
      slug,
    });
  }

  return labs.sort((a, b) => a.number.localeCompare(b.number));
}

/** Discover all labs across both volumes */
export function discoverAllLabs(projectRoot: string): { vol1: LabInfo[]; vol2: LabInfo[] } {
  return {
    vol1: discoverVolumeLabs(projectRoot, 1),
    vol2: discoverVolumeLabs(projectRoot, 2),
  };
}
