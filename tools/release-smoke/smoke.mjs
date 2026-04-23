import { chromium } from 'playwright';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPORTS_DIR = path.join(__dirname, 'reports');
const SCREENSHOTS_DIR = path.join(__dirname, 'screenshots');

const siteName = process.argv[2];
if (!siteName) {
  console.error('usage: node smoke.mjs <site>   (see sites.json for keys, or "all")');
  process.exit(1);
}

const sites = JSON.parse(await fs.readFile(path.join(__dirname, 'sites.json'), 'utf8'));
const targets = siteName === 'all' ? Object.keys(sites) : [siteName];
for (const t of targets) {
  if (!sites[t]) {
    console.error(`unknown site: ${t} (known: ${Object.keys(sites).join(', ')})`);
    process.exit(1);
  }
}

await fs.mkdir(REPORTS_DIR, { recursive: true });
await fs.mkdir(SCREENSHOTS_DIR, { recursive: true });

const browser = await chromium.launch();
const results = {};
let failures = 0;

for (const name of targets) {
  const cfg = sites[name];
  console.log(`\n=== ${name} :: ${cfg.url} ===`);
  const context = await browser.newContext({ userAgent: 'mlsysbook-release-smoke/1.0' });
  const page = await context.newPage();

  const consoleErrors = [];
  const pageErrors = [];
  const failedRequests = [];
  page.on('console', (msg) => { if (msg.type() === 'error') consoleErrors.push(msg.text()); });
  page.on('pageerror', (err) => pageErrors.push(err.message));
  page.on('requestfailed', (req) => failedRequests.push(`${req.method()} ${req.url()} :: ${req.failure()?.errorText}`));

  const siteResult = { url: cfg.url, checks: {}, warnings: [], errors: [] };

  try {
    const resp = await page.goto(cfg.url, { waitUntil: cfg.waitUntil || 'networkidle', timeout: 30000 });
    siteResult.checks.httpStatus = resp ? resp.status() : null;
    if (!resp || resp.status() !== 200) {
      siteResult.errors.push(`HTTP ${resp?.status() ?? 'n/a'} on landing`);
    }

    const title = await page.title();
    siteResult.checks.title = title;
    if (cfg.titleIncludes && !title.toLowerCase().includes(cfg.titleIncludes.toLowerCase())) {
      siteResult.errors.push(`title does not include "${cfg.titleIncludes}": got "${title}"`);
    }

    if (cfg.expectedH1) {
      const h1 = await page.locator('h1').first().textContent().catch(() => null);
      siteResult.checks.h1 = h1?.trim();
      if (!h1 || !h1.includes(cfg.expectedH1)) {
        siteResult.errors.push(`H1 mismatch: expected "${cfg.expectedH1}", got "${h1?.trim() ?? 'none'}"`);
      }
    }

    const navbarPresent = await page.locator('.navbar, .navbar-brand, nav').count();
    siteResult.checks.navbarElements = navbarPresent;
    if (navbarPresent === 0) siteResult.warnings.push('no navbar element found');

    const headingsFound = await page.locator('h1, h2, h3').allTextContents();
    siteResult.checks.headingsSample = headingsFound.slice(0, 20).map((s) => s.trim()).filter(Boolean);
    for (const h of (cfg.expectedHeadings || [])) {
      const found = headingsFound.some((t) => t.includes(h));
      if (!found) siteResult.errors.push(`expected heading "${h}" not found`);
    }

    const linkHrefs = await page.$$eval('a[href]', (as) => as.map((a) => a.getAttribute('href')).filter(Boolean));
    const sameOrigin = [...new Set(linkHrefs
      .map((h) => { try { return new URL(h, cfg.url).toString(); } catch { return null; } })
      .filter((u) => u && u.startsWith('https://mlsysbook.ai/'))
    )];
    siteResult.checks.sameOriginLinkCount = sameOrigin.length;

    const brokenLinks = [];
    for (const u of sameOrigin) {
      try {
        const r = await context.request.head(u, { timeout: 10000, maxRedirects: 3 });
        if (r.status() >= 400) brokenLinks.push(`${r.status()} ${u}`);
      } catch (e) {
        brokenLinks.push(`err ${u} :: ${e.message}`);
      }
    }
    siteResult.checks.brokenLinkCount = brokenLinks.length;
    if (brokenLinks.length > 0) siteResult.brokenLinks = brokenLinks;

    const additional = cfg.additionalPages || [];
    const additionalResults = {};
    for (const rel of additional) {
      const full = new URL(rel, cfg.url).toString();
      try {
        const r = await context.request.get(full, { timeout: 10000 });
        additionalResults[rel] = r.status();
        if (r.status() !== 200) siteResult.errors.push(`additional page ${rel} => ${r.status()}`);
      } catch (e) {
        additionalResults[rel] = `err:${e.message}`;
        siteResult.errors.push(`additional page ${rel} => ${e.message}`);
      }
    }
    if (Object.keys(additionalResults).length) siteResult.checks.additionalPages = additionalResults;

    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const shotPath = path.join(SCREENSHOTS_DIR, `${name}-${stamp}.png`);
    await page.screenshot({ path: shotPath, fullPage: true });
    siteResult.checks.screenshot = path.relative(__dirname, shotPath);

    siteResult.checks.consoleErrors = consoleErrors.length;
    siteResult.checks.pageErrors = pageErrors.length;
    siteResult.checks.failedRequests = failedRequests.length;
    if (consoleErrors.length) siteResult.consoleErrors = consoleErrors.slice(0, 10);
    if (pageErrors.length) siteResult.pageErrors = pageErrors.slice(0, 10);
    if (failedRequests.length) siteResult.failedRequests = failedRequests.slice(0, 10);

    if (pageErrors.length > 0) siteResult.errors.push(`${pageErrors.length} uncaught JS page errors`);
  } catch (e) {
    siteResult.errors.push(`exception: ${e.message}`);
  }

  siteResult.pass = siteResult.errors.length === 0;
  if (!siteResult.pass) failures++;

  const emoji = siteResult.pass ? 'PASS' : 'FAIL';
  console.log(`[${emoji}] ${name}  title="${siteResult.checks.title ?? ''}"  links=${siteResult.checks.sameOriginLinkCount ?? 0}  broken=${siteResult.checks.brokenLinkCount ?? 0}  console=${siteResult.checks.consoleErrors ?? 0}`);
  for (const err of siteResult.errors) console.log(`   ERR  ${err}`);
  for (const w of siteResult.warnings) console.log(`   warn ${w}`);

  results[name] = siteResult;
  await context.close();
}

await browser.close();

const reportPath = path.join(REPORTS_DIR, `smoke-${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
await fs.writeFile(reportPath, JSON.stringify(results, null, 2));
console.log(`\nReport: ${path.relative(process.cwd(), reportPath)}`);
process.exit(failures === 0 ? 0 : 1);
