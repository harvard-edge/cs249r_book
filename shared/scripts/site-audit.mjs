#!/usr/bin/env node
/* =============================================================================
 * site-audit.mjs — release-prep visual / structural QA across all subsites
 * =============================================================================
 *
 * One Playwright-driven audit script that the release-prep plan needs three
 * times over (sidebar present + visible, dark-mode renders cleanly, no
 * broken images / scripts). Implemented as a single CLI with subcommands so
 * the shared boilerplate (browser launch, URL list, output dirs, screenshot
 * naming) stays in one place.
 *
 * Subcommands
 * -----------
 *   sidebar    Assert every site URL exposes a populated, visible <#quarto-sidebar>.
 *              Skips URLs known to be sidebar-less (landing pages, StaffML).
 *
 *   darkmode   Toggle dark mode, scroll the page top→bottom, and screenshot
 *              into _audit/darkmode/<host>/<path>.png. Manual review surfaces
 *              "half-themed" widgets that CSS lints can't find.
 *
 *   assets     Listen for failed network requests (images, scripts, fonts)
 *              and report any 4xx/5xx or DNS failures. Catches broken
 *              <img src> embeds (TinyTorch PDF viewer, Vol II cover) and
 *              missing fonts/styles before they hit production.
 *
 * Usage
 * -----
 *   node shared/scripts/site-audit.mjs <subcommand> [--target dev|live|local] \
 *        [--only <substring>] [--out _audit]
 *
 * Examples:
 *   node shared/scripts/site-audit.mjs sidebar  --target dev
 *   node shared/scripts/site-audit.mjs darkmode --target dev --only vol2
 *   node shared/scripts/site-audit.mjs assets   --target live
 *
 * Requirements
 * ------------
 *   npm install --save-dev playwright
 *   npx playwright install chromium
 *
 * The script intentionally reports issues to stdout AND exits non-zero so
 * it can be wired into CI. Until baselines are clean we run it as
 * non-blocking — same staged philosophy as the Lychee link checks.
 * ============================================================================= */

import { chromium } from "playwright";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import process from "node:process";

const __dirname = dirname(fileURLToPath(import.meta.url));

// ──────────────────────────────────────────────────────────────────────────
// Site map. Each entry is [logical-name, dev-path, live-path, has-sidebar?].
// Keep this list in sync with .github/scripts/rewrite-dev-urls.sh: any new
// subsite needs to appear in both places (audit coverage + URL rewriting).
// ──────────────────────────────────────────────────────────────────────────
const SITES = [
  ["landing",     "",                "",                false],
  ["vol1",        "book/vol1/",      "vol1/",           true],
  ["vol2",        "book/vol2/",      "vol2/",           true],
  ["tinytorch",   "tinytorch/",      "tinytorch/",      true],
  ["kits",        "kits/",           "kits/",           true],
  ["labs",        "labs/",           "labs/",           true],
  ["mlsysim",     "mlsysim/",        "mlsysim/",        true],
  ["slides",      "slides/",         "slides/",         false],
  ["instructors", "instructors/",    "instructors/",    true],
  ["staffml",     "staffml/",        "staffml/",        false],
];

const TARGETS = {
  dev:   "https://harvard-edge.github.io/cs249r_book_dev/",
  live:  "https://mlsysbook.ai/",
  local: "http://127.0.0.1:8000/", // for `python -m http.server` on _site/
};

// ──────────────────────────────────────────────────────────────────────────
// CLI parsing — keep dependency-free; the audits already pull in Playwright.
// ──────────────────────────────────────────────────────────────────────────
function parseArgs(argv) {
  const [cmd, ...rest] = argv.slice(2);
  const opts = { target: "dev", only: null, out: "_audit" };
  for (let i = 0; i < rest.length; i++) {
    const arg = rest[i];
    if (arg === "--target") opts.target = rest[++i];
    else if (arg === "--only") opts.only = rest[++i];
    else if (arg === "--out") opts.out = rest[++i];
    else if (arg === "--help" || arg === "-h") opts.help = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return { cmd, opts };
}

function help() {
  console.log(
    "Usage: node shared/scripts/site-audit.mjs <sidebar|darkmode|assets> " +
      "[--target dev|live|local] [--only <substring>] [--out _audit]"
  );
}

function siteUrls({ target, only }) {
  const base = TARGETS[target];
  if (!base) throw new Error(`Unknown --target: ${target}`);
  const path = (entry) => (target === "live" ? entry[2] : entry[1]);
  const filtered = only ? SITES.filter((s) => s[0].includes(only) || path(s).includes(only)) : SITES;
  return filtered.map((entry) => ({
    name: entry[0],
    url: base + path(entry),
    hasSidebar: entry[3],
  }));
}

// ──────────────────────────────────────────────────────────────────────────
// SIDEBAR — assert every Quarto subsite renders a non-empty visible sidebar.
// ──────────────────────────────────────────────────────────────────────────
async function auditSidebar(page, site) {
  const failures = [];
  if (!site.hasSidebar) return failures;
  await page.goto(site.url, { waitUntil: "domcontentloaded", timeout: 30000 });
  // Quarto sidebars use #quarto-sidebar; some templates use .sidebar-navigation.
  const exists = await page.$("#quarto-sidebar, .sidebar-navigation");
  if (!exists) {
    failures.push(`${site.name}: no sidebar element found`);
    return failures;
  }
  const visible = await exists.isVisible();
  if (!visible) failures.push(`${site.name}: sidebar element present but hidden`);
  const links = await exists.$$("a");
  if (links.length < 3) {
    failures.push(`${site.name}: sidebar has <3 links (likely empty)`);
  }
  return failures;
}

// ──────────────────────────────────────────────────────────────────────────
// DARKMODE — flip to dark, scroll, screenshot. Reviewer eyeballs results.
// ──────────────────────────────────────────────────────────────────────────
async function auditDarkmode(page, site, outDir) {
  await page.goto(site.url, { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.evaluate(() => {
    try {
      localStorage.setItem("quarto-color-scheme", "dark");
      localStorage.setItem("staffml_theme", "dark");
    } catch (e) {}
    document.documentElement.setAttribute("data-bs-theme", "dark");
    document.documentElement.setAttribute("data-quarto-color-scheme", "dark");
    document.documentElement.dataset.theme = "dark";
  });
  await page.waitForTimeout(500);
  // Scroll to bottom in chunks so lazy-loaded content renders before capture.
  const height = await page.evaluate(() => document.body.scrollHeight);
  for (let y = 0; y < height; y += 800) {
    await page.evaluate((y) => window.scrollTo(0, y), y);
    await page.waitForTimeout(150);
  }
  await page.evaluate(() => window.scrollTo(0, 0));
  const file = join(outDir, `${site.name}.png`);
  await page.screenshot({ path: file, fullPage: true });
  return [`${site.name}: dark-mode screenshot → ${file}`];
}

// ──────────────────────────────────────────────────────────────────────────
// ASSETS — capture failed network requests and surface them per page.
// ──────────────────────────────────────────────────────────────────────────
async function auditAssets(context, site) {
  const failures = [];
  const page = await context.newPage();
  page.on("requestfailed", (req) => {
    failures.push(`${site.name}: ${req.method()} ${req.url()} (${req.failure()?.errorText || "failed"})`);
  });
  page.on("response", (res) => {
    if (res.status() >= 400) {
      failures.push(`${site.name}: ${res.status()} ${res.url()}`);
    }
  });
  try {
    await page.goto(site.url, { waitUntil: "networkidle", timeout: 45000 });
  } catch (e) {
    failures.push(`${site.name}: navigation error — ${e.message}`);
  } finally {
    await page.close();
  }
  return failures;
}

// ──────────────────────────────────────────────────────────────────────────
// Entry point.
// ──────────────────────────────────────────────────────────────────────────
async function main() {
  const { cmd, opts } = parseArgs(process.argv);
  if (!cmd || opts.help) {
    help();
    process.exit(cmd ? 0 : 1);
  }

  const sites = siteUrls(opts);
  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const allFailures = [];

  try {
    if (cmd === "sidebar") {
      for (const site of sites) {
        const page = await context.newPage();
        try {
          allFailures.push(...(await auditSidebar(page, site)));
        } catch (e) {
          allFailures.push(`${site.name}: error — ${e.message}`);
        } finally {
          await page.close();
        }
      }
    } else if (cmd === "darkmode") {
      const outDir = join(process.cwd(), opts.out, "darkmode");
      await mkdir(outDir, { recursive: true });
      for (const site of sites) {
        const page = await context.newPage();
        try {
          allFailures.push(...(await auditDarkmode(page, site, outDir)));
        } catch (e) {
          allFailures.push(`${site.name}: error — ${e.message}`);
        } finally {
          await page.close();
        }
      }
    } else if (cmd === "assets") {
      for (const site of sites) {
        try {
          allFailures.push(...(await auditAssets(context, site)));
        } catch (e) {
          allFailures.push(`${site.name}: error — ${e.message}`);
        }
      }
    } else {
      help();
      process.exit(2);
    }
  } finally {
    await browser.close();
  }

  // Persist a JSON report for machine consumption (CI).
  const reportDir = join(process.cwd(), opts.out);
  await mkdir(reportDir, { recursive: true });
  const reportPath = join(reportDir, `${cmd}.json`);
  await writeFile(reportPath, JSON.stringify({ cmd, target: opts.target, failures: allFailures }, null, 2));

  if (cmd === "darkmode") {
    // For darkmode, the "failures" array is just informational (paths to
    // screenshots). Treat them as reportable, not failing.
    console.log(`✅ Captured ${allFailures.length} dark-mode screenshots → ${opts.out}/darkmode/`);
    process.exit(0);
  }

  if (allFailures.length === 0) {
    console.log(`✅ ${cmd}: no issues found across ${sites.length} sites.`);
    process.exit(0);
  }
  console.log(`❌ ${cmd}: ${allFailures.length} issue(s):`);
  for (const f of allFailures) console.log(`   - ${f}`);
  console.log(`\nReport written to ${reportPath}`);
  process.exit(1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
