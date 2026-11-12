#!/usr/bin/env node
/**
 * sync-taxonomy.mjs
 *
 * Reads the canonical Machine Learning Systems Taxonomy from
 * `taxonomy/framework-data.json` and generates `socratiq/src_shared/taxonomy-map.js`
 * with zero runtime dependencies.
 */

import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, '../..');
const FRAMEWORK_JSON_PATH = resolve(REPO_ROOT, 'taxonomy/framework-data.json');
const OUTPUT_PATH = resolve(__dirname, '../src_shared/taxonomy-map.js');

const GROUP_COLORS = {
  F: '#2563eb',
  S: '#7c3aed',
  P: '#059669',
  Q: '#d97706',
  T: '#dc2626',
};

const GROUP_ORDER = ['F', 'S', 'P', 'Q', 'T'];

async function main() {
  const raw = await readFile(FRAMEWORK_JSON_PATH, 'utf8');
  const data = JSON.parse(raw);
  const framework = data.framework || {};

  // 1. Build TAXONOMY_GROUPS
  const groupsMap = {};
  for (const group of framework.groups || []) {
    const code = String(group.code || '').trim();
    if (!code) continue;
    groupsMap[code] = {
      id: group.id,
      code,
      name: group.label,
      question: group.question,
      color: GROUP_COLORS[code] || '#2563eb',
    };
  }

  // 2. Build TAXONOMY_SUBJECTS & collect chapter associations
  const subjectsMap = {};
  const chapterIndexAccumulator = new Map();

  function ensureChapter(slug, volume, title) {
    if (!slug) return null;
    if (!chapterIndexAccumulator.has(slug)) {
      chapterIndexAccumulator.set(slug, {
        slug,
        volume: volume || slug.split('/')[0] || 'vol1',
        title: title || slug,
        groupWeights: { F: 0, S: 0, P: 0, Q: 0, T: 0 },
        groupsSet: new Set(),
        directSubjects: new Set(),
        linkedSubjects: new Set(),
      });
    } else {
      const existing = chapterIndexAccumulator.get(slug);
      if (title && (!existing.title || existing.title === slug)) {
        existing.title = title;
      }
      if (volume && !existing.volume) {
        existing.volume = volume;
      }
    }
    return chapterIndexAccumulator.get(slug);
  }

  for (const area of framework.areas || []) {
    const areaCode = String(area.code || '').trim();
    const areaName = area.display_label || area.label || areaCode;
    const groupCode = areaCode.charAt(0);

    for (const row of area.rows || []) {
      const code = String(row.code || '').trim();
      if (!code) continue;
      const name = row.display_label || row.label || code;
      const fullLabel = row.label && row.label !== name ? ` (${row.label})` : '';
      const includesText = Array.isArray(row.includes) && row.includes.length > 0
        ? ` Includes: ${row.includes.join('; ')}.`
        : '';
      const summary = `${row.definition || ''}${fullLabel}${includesText}`.trim();

      const chapterSlugs = new Set();
      for (const ch of row.book_chapters || []) {
        if (!ch?.chapter) continue;
        chapterSlugs.add(ch.chapter);
        const entry = ensureChapter(ch.chapter, ch.volume, ch.title);
        if (entry) {
          entry.groupWeights[groupCode] = (entry.groupWeights[groupCode] || 0) + 3;
          entry.groupsSet.add(groupCode);
          entry.directSubjects.add(code);
        }
      }

      for (const bl of row.book_links || []) {
        if (!bl?.chapter) continue;
        chapterSlugs.add(bl.chapter);
        const entry = ensureChapter(bl.chapter, bl.volume, bl.chapter_title);
        if (entry) {
          entry.groupWeights[groupCode] = (entry.groupWeights[groupCode] || 0) + 1;
          entry.groupsSet.add(groupCode);
          entry.linkedSubjects.add(code);
        }
      }

      subjectsMap[code] = {
        code,
        name,
        group: groupCode,
        areaCode,
        areaName,
        summary,
        chapters: [...chapterSlugs],
      };
    }
  }

  // Also ensure any top-level book_links chapters are present
  for (const bl of data.book_links || []) {
    if (bl?.chapter) {
      ensureChapter(bl.chapter, bl.volume, bl.chapter_title);
    }
  }

  // 3. Build CHAPTER_TAXONOMY_INDEX sorted by chapter slug
  const chapterTaxonomyIndex = {};
  const sortedSlugs = [...chapterIndexAccumulator.keys()].sort();
  for (const slug of sortedSlugs) {
    const item = chapterIndexAccumulator.get(slug);
    const groups = GROUP_ORDER.filter((g) => item.groupsSet.has(g));
    const primaryGroup = GROUP_ORDER.slice().sort(
      (a, b) => (item.groupWeights[b] || 0) - (item.groupWeights[a] || 0)
    )[0] || groups[0] || 'F';
    const subjects = [
      ...item.directSubjects,
      ...[...item.linkedSubjects].filter((code) => !item.directSubjects.has(code)),
    ];

    chapterTaxonomyIndex[slug] = {
      slug: item.slug,
      volume: item.volume,
      title: item.title,
      primaryGroup,
      groups: groups.length ? groups : [primaryGroup],
      subjects,
    };
  }

  const moduleSource = `// AUTO-GENERATED by socratiq/scripts/sync-taxonomy.mjs from taxonomy/framework-data.json
// Do not hand-edit without re-running: node scripts/sync-taxonomy.mjs

function attachLookupHelpers(record) {
  Object.defineProperties(record, {
    get: { value: (key) => record[key], enumerable: false },
    has: { value: (key) => Object.prototype.hasOwnProperty.call(record, key), enumerable: false },
    size: { get: () => Object.keys(record).length, enumerable: false },
  });
  return Object.freeze(record);
}

export const TAXONOMY_GROUPS = attachLookupHelpers(${JSON.stringify(groupsMap, null, 2)});

export const TAXONOMY_SUBJECTS = attachLookupHelpers(${JSON.stringify(subjectsMap, null, 2)});

export const CHAPTER_TAXONOMY_INDEX = attachLookupHelpers(${JSON.stringify(chapterTaxonomyIndex, null, 2)});

/**
 * Resolves any chapter URL, pathname, or slug (including \`/vol1/01_introduction/\`,
 * \`/vol1/chapters/01_introduction/01_introduction.html\`, or \`vol1/01_introduction\`)
 * to its canonical chapter taxonomy entry with subject details.
 */
export function resolveChapterTaxonomy(urlOrPath = '') {
  if (typeof urlOrPath !== 'string' || !urlOrPath.trim()) return null;
  const raw = urlOrPath.trim();

  // Direct slug lookup first
  if (CHAPTER_TAXONOMY_INDEX[raw]) {
    return enrichChapterEntry(CHAPTER_TAXONOMY_INDEX[raw]);
  }

  let pathname = raw;
  try {
    if (/^https?:\\/\\//i.test(raw)) {
      pathname = new URL(raw).pathname;
    }
  } catch {
    pathname = raw;
  }

  const normalized = pathname.replace(/\\\\/g, '/');

  // Match /volN/chapters/NN_slug or /volN/NN_slug
  const match = normalized.match(/(?:^|\\b|\\/)(vol[1-4])\\/(?:chapters\\/)?([0-9]{2}_[a-z0-9_]+)(?:\\/|\\.html|$|#|\\?)/i);
  if (match) {
    const slug = \`\${match[1].toLowerCase()}/\${match[2].toLowerCase()}\`;
    if (CHAPTER_TAXONOMY_INDEX[slug]) {
      return enrichChapterEntry(CHAPTER_TAXONOMY_INDEX[slug]);
    }
  }

  // Fallback: check if any indexed chapter slug or unique chapter directory appears in the path
  for (const [slug, entry] of Object.entries(CHAPTER_TAXONOMY_INDEX)) {
    const [, folder] = slug.split('/');
    if (normalized.includes(slug)) {
      return enrichChapterEntry(entry);
    }
    if (folder && normalized.includes(\`/\${entry.volume}/\`) && normalized.includes(folder)) {
      return enrichChapterEntry(entry);
    }
  }

  return null;
}

function enrichChapterEntry(entry) {
  if (!entry) return null;
  const primaryGroupInfo = TAXONOMY_GROUPS[entry.primaryGroup] || null;
  const subjectDetails = (entry.subjects || [])
    .map((code) => TAXONOMY_SUBJECTS[code])
    .filter(Boolean);
  return {
    ...entry,
    groupTitle: primaryGroupInfo?.name || entry.primaryGroup,
    primaryGroupInfo,
    subjectDetails,
  };
}

/**
 * Returns a concise (~180-240 char) grounding line when a chapter matches,
 * or a general 1-line 5-Group overview when on a non-chapter page.
 */
export function formatChapterTaxonomyPrompt(urlOrPath = '') {
  const chapter = resolveChapterTaxonomy(urlOrPath);
  if (chapter) {
    const shortCode = chapter.slug.replace(/^((?:vol[1-4])\\/[0-9]{2})_.*$/, '$1');
    const topSubjects = chapter.subjectDetails
      .slice(0, 3)
      .map((s) => \`\${s.code} \${s.name}\`)
      .join(', ');
    return \`Active chapter ML Systems Taxonomy (\${shortCode} \${chapter.title}): Groups \${chapter.groups.join(', ')} | Subjects \${topSubjects}. Situate explanations and trade-offs across Foundations (F), Stack (S), Lifecycle (P), Qualities (Q), and System Types (T).\`;
  }
  return 'ML Systems Taxonomy lens: situate mechanisms and trade-offs across the 5 Groups — Foundations (F), Stack (S), Lifecycle (P), Qualities (Q), and System Types (T).';
}

/**
 * Returns matching taxonomy items (starting with a '🧭 Current chapter taxonomy overview'
 * item and the active chapter's subjects, followed by matching subjects from TAXONOMY_SUBJECTS)
 * for use in + Context and @ mentions.
 */
export function searchTaxonomyItems(query = '', currentUrl = '') {
  const normalizedQuery = String(query || '').trim().toLowerCase();
  const chapter = resolveChapterTaxonomy(currentUrl);
  const results = [];
  const seenKeys = new Set();

  const overviewTitle = 'Current chapter taxonomy overview';
  const overviewContent = chapter
    ? \`\${formatChapterTaxonomyPrompt(currentUrl)}\\n\\nPrimary Group: \${chapter.primaryGroup} (\${TAXONOMY_GROUPS[chapter.primaryGroup]?.name || chapter.primaryGroup})\\nSubjects: \${chapter.subjectDetails.map((s) => \`\${s.code} \${s.name}\`).join(', ')}\`
    : 'ML Systems Taxonomy Overview: Foundations (F: workloads, requirements, models, performance models), Stack (S: hardware, compilers, parallelism, fleet), Lifecycle (P: data, training, evaluation, serving, MLOps), Qualities (Q: performance, reliability, security, safety, responsible AI, sustainability), System Types (T: agentic and embodied/physical systems).';

  const matchesOverview = !normalizedQuery
    || 'taxonomy'.includes(normalizedQuery)
    || overviewTitle.toLowerCase().includes(normalizedQuery)
    || (chapter && (
      chapter.slug.toLowerCase().includes(normalizedQuery)
      || chapter.title.toLowerCase().includes(normalizedQuery)
    ));

  if (matchesOverview) {
    results.push({
      type: 'taxonomy',
      icon: '🧭',
      code: 'overview',
      key: overviewTitle,
      title: overviewTitle,
      content: overviewContent,
      value: {
        title: overviewTitle,
        content: overviewContent,
        text: overviewContent,
      },
    });
    seenKeys.add(overviewTitle);
  }

  function makeSubjectItem(subject) {
    const title = \`\${subject.code} \${subject.name}\`;
    const groupInfo = TAXONOMY_GROUPS[subject.group];
    const content = \`\${subject.code} \${subject.name} [Group \${subject.group} \${groupInfo ? groupInfo.name : ''} · \${subject.areaCode} \${subject.areaName}]: \${subject.summary}\`;
    return {
      type: 'taxonomy',
      icon: '🧭',
      code: subject.code,
      group: subject.group,
      key: title,
      title,
      content,
      value: {
        title,
        content,
        text: content,
      },
    };
  }

  function subjectMatches(subject) {
    if (!normalizedQuery || 'taxonomy'.includes(normalizedQuery)) return true;
    const haystack = \`\${subject.code} \${subject.name} \${subject.areaCode} \${subject.areaName} \${subject.summary} \${TAXONOMY_GROUPS[subject.group]?.name || ''}\`.toLowerCase();
    return haystack.includes(normalizedQuery);
  }

  if (chapter) {
    for (const subject of chapter.subjectDetails) {
      const item = makeSubjectItem(subject);
      if (!seenKeys.has(item.key) && subjectMatches(subject)) {
        results.push(item);
        seenKeys.add(item.key);
      }
    }
  }

  for (const subject of Object.values(TAXONOMY_SUBJECTS)) {
    const item = makeSubjectItem(subject);
    if (!seenKeys.has(item.key) && subjectMatches(subject)) {
      results.push(item);
      seenKeys.add(item.key);
    }
  }

  return results;
}
`;

  await mkdir(dirname(OUTPUT_PATH), { recursive: true });
  await writeFile(OUTPUT_PATH, moduleSource, 'utf8');
  console.log(
    `Generated ${OUTPUT_PATH} (${Object.keys(groupsMap).length} groups, ${Object.keys(subjectsMap).length} subjects, ${Object.keys(chapterTaxonomyIndex).length} chapters)`
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
