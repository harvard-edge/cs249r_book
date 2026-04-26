#!/usr/bin/env node
/**
 * One-shot: (1) create area: mlsysim + area: staffml if missing,
 * (2) relabel all issues+PRs from area: interview -> area: staffml,
 * (3) recompute path-based area for every PR from changed files and update labels.
 *
 * Rules must match .github/workflows/auto-label.yml (path prefix order).
 * Usage: GITHUB_TOKEN=$(gh auth token) node .github/scripts/sync-area-labels.mjs [owner/repo]
 */
const REPO = process.argv[2] || 'harvard-edge/cs249r_book';
const [OWNER, REPO_NAME] = REPO.split('/');

const RULES = [
  { prefix: '.github/', label: 'area: tools' },
  { prefix: 'tools/', label: 'area: tools' },
  { prefix: 'book/', label: 'area: book' },
  { prefix: 'tinytorch/', label: 'area: tinytorch' },
  { prefix: 'kits/', label: 'area: kits' },
  { prefix: 'labs/', label: 'area: labs' },
  { prefix: 'socratiq/', label: 'area: socratiq' },
  { prefix: 'site/', label: 'area: website' },
  { prefix: 'website/', label: 'area: website' },
  { prefix: 'mlsysim/', label: 'area: mlsysim' },
  { prefix: 'interviews/', label: 'area: staffml' },
];

const TOKEN = process.env.GITHUB_TOKEN || process.env.GH_TOKEN;
if (!TOKEN) {
  console.error('Set GITHUB_TOKEN or GH_TOKEN (e.g. GITHUB_TOKEN=$(gh auth token))');
  process.exit(1);
}

const headers = {
  Authorization: `Bearer ${TOKEN}`,
  Accept: 'application/vnd.github+json',
  'X-GitHub-Api-Version': '2022-11-28',
};

async function ghGet(path) {
  const u = `https://api.github.com${path}`;
  const res = await fetch(u, { headers });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`GET ${path} ${res.status}: ${t.slice(0, 400)}`);
  }
  return res.json();
}

async function ghPut(path, body) {
  const u = `https://api.github.com${path}`;
  const res = await fetch(u, {
    method: 'PUT',
    headers: { ...headers, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`PUT ${path} ${res.status}: ${t.slice(0, 400)}`);
  }
}

function areaFromPaths(filePaths) {
  const areaCounts = {};
  for (const filePath of filePaths) {
    for (const rule of RULES) {
      if (filePath.startsWith(rule.prefix)) {
        areaCounts[rule.label] = (areaCounts[rule.label] || 0) + 1;
        break;
      }
    }
  }
  let areaLabel = '';
  let maxCount = 0;
  for (const [label, count] of Object.entries(areaCounts)) {
    if (count > maxCount) {
      maxCount = count;
      areaLabel = label;
    }
  }
  return areaLabel;
}

async function listAllPrFiles(pullNumber) {
  const paths = [];
  let page = 1;
  for (;;) {
    const data = await ghGet(
      `/repos/${OWNER}/${REPO_NAME}/pulls/${pullNumber}/files?per_page=100&page=${page}`,
    );
    for (const f of data) paths.push(f.filename);
    if (data.length < 100) break;
    page += 1;
  }
  return paths;
}

async function listAllPrNumbers() {
  const nums = [];
  let page = 1;
  for (;;) {
    const data = await ghGet(
      `/repos/${OWNER}/${REPO_NAME}/pulls?state=all&per_page=100&page=${page}`,
    );
    for (const p of data) nums.push(p.number);
    if (data.length < 100) break;
    page += 1;
  }
  return nums;
}

async function getIssueLabels(num) {
  const issue = await ghGet(
    `/repos/${OWNER}/${REPO_NAME}/issues/${num}`,
  );
  return issue.labels.map((l) => l.name);
}

function withAreaReplaced(existingNames, newArea) {
  const out = existingNames.filter((n) => !n.startsWith('area: '));
  if (newArea) out.push(newArea);
  return [...new Set(out)];
}

function sameLabelSet(a, b) {
  if (a.length !== b.length) return false;
  const x = [...a].sort();
  const y = [...b].sort();
  return x.every((v, i) => v === y[i]);
}

function labelsToPut(names) {
  return { labels: names };
}

async function ensureLabel(name, color, description) {
  const u = `https://api.github.com/repos/${OWNER}/${REPO_NAME}/labels`;
  const res = await fetch(u, {
    method: 'POST',
    headers: { ...headers, 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, color, description }),
  });
  if (res.status === 201) return 'created';
  if (res.status === 422) return 'exists';
  const t = await res.text();
  throw new Error(`create label ${name}: ${res.status} ${t.slice(0, 200)}`);
}

async function main() {
  console.log(`Target: ${REPO}\n`);

  // 1) Ensure new area labels
  for (const [name, color, desc] of [
    ['area: mlsysim', '1D76DB', 'Path mlsysim/ — auto-label'],
    ['area: staffml', '0E8A16', 'Path interviews/ — auto-label; StaffML'],
  ]) {
    const r = await ensureLabel(name, color, desc);
    console.log(r === 'exists' ? `  Label exists: ${name}` : `  Created label: ${name}`);
  }

  // 2) Migrate area: interview -> area: staffml
  const interviewNums = new Set();
  let ipage = 1;
  for (;;) {
    const batch = await ghGet(
      `/repos/${OWNER}/${REPO_NAME}/issues?state=all&labels=area%3Ainterview&per_page=100&page=${ipage}`,
    );
    for (const it of batch) {
      interviewNums.add(it.number);
    }
    if (batch.length < 100) break;
    ipage += 1;
  }
  console.log(`\nItems with "area: interview": ${interviewNums.size}`);

  for (const num of interviewNums) {
    const current = await getIssueLabels(num);
    const next = current.filter(
      (n) => n !== 'area: interview' && n !== 'area: staffml',
    );
    next.push('area: staffml');
    const deduped = [...new Set(next)];
    if (JSON.stringify(deduped.sort()) === JSON.stringify([...current].sort())) {
      continue;
    }
    await ghPut(
      `/repos/${OWNER}/${REPO_NAME}/issues/${num}/labels`,
      labelsToPut(deduped),
    );
    console.log(`  interview->staffml: #${num}`);
  }

  // 3) Path-based area for every PR
  const prNums = await listAllPrNumbers();
  console.log(`\nPRs to sync: ${prNums.length}`);

  let updated = 0;
  let skipped = 0;
  let failed = 0;
  for (let i = 0; i < prNums.length; i += 1) {
    const num = prNums[i];
    if (i > 0 && i % 20 === 0) await new Promise((r) => setTimeout(r, 1000));
    try {
      const paths = await listAllPrFiles(num);
      const newArea = areaFromPaths(paths);
      if (!newArea) {
        skipped += 1;
        continue;
      }
      const current = await getIssueLabels(num);
      const beforeArea = current.filter((n) => n.startsWith('area: '));
      const after = withAreaReplaced(current, newArea);
      if (sameLabelSet(after, current)) {
        skipped += 1;
        continue;
      }
      await ghPut(
        `/repos/${OWNER}/${REPO_NAME}/issues/${num}/labels`,
        labelsToPut(after),
      );
      console.log(
        `  #${num} before: ${beforeArea.join(', ') || '∅'} -> ${newArea} (${paths.length} files)`,
      );
      updated += 1;
    } catch (e) {
      failed += 1;
      console.error(`  #${num} ERROR:`, e.message);
    }
  }

  console.log(
    `\nDone. Updated: ${updated}, unchanged/skipped: ${skipped}, failed: ${failed}.`,
  );
  console.log('Note: non-PR issues (no file list) were not path-resynced; interview migration handled above.');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
