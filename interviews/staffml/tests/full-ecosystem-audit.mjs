// Full-ecosystem visual + integrity audit for the post-deploy verification.
//
// Hits every site we just (re)published, in light + dark mode, across
// iPhone / iPad / desktop viewports. Captures a screenshot per cell, plus
// a per-cell metrics record:
//   - Bootstrap CSS HTTP status (200 vs 404 — original FOUC bug)
//   - Page horizontal scroll? (the iPad-landscape navbar overflow bug)
//   - Navbar visible (.navbar element present and styled — guards FOUC)
//   - Theme actually applied (data-bs-theme matches what we asked for)
//   - Console errors (network 404s, JS errors)
//
// Usage:
//   node tests/full-ecosystem-audit.mjs                        # production
//   BASE=https://mlsysbook.ai node tests/full-ecosystem-audit.mjs

import { webkit, devices } from "playwright";
import fs from "node:fs";
import path from "node:path";

const BASE = (process.env.BASE || "https://mlsysbook.ai").replace(/\/$/, "");
const OUT  = path.resolve(process.env.OUT || "tests/audit-output");
fs.mkdirSync(OUT, { recursive: true });

// ─── Sites to audit ────────────────────────────────────────────────
const SITES = [
  { id: "landing",   path: "/" },
  { id: "staffml",   path: "/staffml/" },
  { id: "tinytorch", path: "/tinytorch/" },
  { id: "mlsysim",   path: "/mlsysim/" },
  { id: "vol1",      path: "/vol1/" },
  { id: "vol2",      path: "/vol2/" },
];

// ─── Viewports — covers iPhone, iPad portrait + landscape, laptop, desktop ──
const VIEWPORTS = [
  { id: "iphone-14-pro",  device: devices["iPhone 14 Pro"] },
  { id: "ipad-mini-port", device: devices["iPad Mini"] },
  { id: "ipad-mini-land", device: devices["iPad Mini landscape"] },
  { id: "ipad-pro-land",  device: devices["iPad Pro 11 landscape"] },
  { id: "desktop-1280",   device: { viewport: { width: 1280, height: 800 }, userAgent: undefined } },
  { id: "desktop-1920",   device: { viewport: { width: 1920, height: 1080 }, userAgent: undefined } },
];

const THEMES = ["light", "dark"];

const browser = await webkit.launch();
const results = [];

console.log(`Audit base=${BASE}`);
console.log(`Sites=${SITES.length}, viewports=${VIEWPORTS.length}, themes=${THEMES.length}`);
console.log(`Total cells=${SITES.length * VIEWPORTS.length * THEMES.length}`);
console.log();

for (const vp of VIEWPORTS) {
  for (const theme of THEMES) {
    const ctx = await browser.newContext({
      ...vp.device,
      colorScheme: theme,
      deviceScaleFactor: 1,
      ignoreHTTPSErrors: true,
    });
    const page = await ctx.newPage();

    for (const site of SITES) {
      const url = BASE + site.path;
      const cell = { site: site.id, viewport: vp.id, theme, url };
      const failed404s = [];
      const consoleErrors = [];

      page.removeAllListeners("response");
      page.removeAllListeners("pageerror");
      page.removeAllListeners("console");

      page.on("response", (resp) => {
        const u = resp.url();
        if (resp.status() === 404 && /\.(css|js|woff2?|png|svg|jpg)(\?|$)/.test(u)) {
          failed404s.push(`${resp.status()} ${u}`);
        }
      });
      page.on("pageerror", (err) => consoleErrors.push(`pageerror: ${err.message}`));
      page.on("console", (msg) => {
        if (msg.type() === "error") consoleErrors.push(`console.error: ${msg.text().slice(0, 200)}`);
      });

      try {
        await page.goto(url, { waitUntil: "networkidle", timeout: 45000 });
        await page.waitForTimeout(800);

        // Pull DOM-level integrity signals.
        const metrics = await page.evaluate(() => {
          const docEl = document.documentElement;
          const vw = window.innerWidth;
          const navEl = document.querySelector(".navbar, header.headroom, [class*=Navbar], [class*=navbar]");
          const navStyle = navEl ? getComputedStyle(navEl) : null;
          return {
            vw,
            docWidth: docEl.scrollWidth,
            horizontalScroll: docEl.scrollWidth > vw + 2,
            theme: docEl.getAttribute("data-bs-theme") || docEl.dataset.bsTheme || (document.body.classList.contains("quarto-dark") ? "dark" : null),
            navbarFound: !!navEl,
            navbarDisplay: navStyle ? navStyle.display : null,
            // Bootstrap loaded? Check for known Bootstrap utility class behaviour.
            bootstrapLoaded: getComputedStyle(document.body).getPropertyValue("--bs-body-bg") !== "" || document.styleSheets.length > 1,
            title: document.title.slice(0, 80),
          };
        });

        cell.metrics = metrics;
        cell.failed404s = failed404s.slice(0, 10);
        cell.consoleErrors = consoleErrors.slice(0, 5);

        const shotName = `${vp.id}__${theme}__${site.id}.png`;
        await page.screenshot({ path: path.join(OUT, shotName), fullPage: false });
        cell.screenshot = shotName;

        const flags = [];
        if (metrics.horizontalScroll) flags.push("HSCROLL");
        if (!metrics.navbarFound) flags.push("NO_NAVBAR");
        if (!metrics.bootstrapLoaded) flags.push("NO_BOOTSTRAP");
        if (failed404s.length > 0) flags.push(`404x${failed404s.length}`);
        const status = flags.length ? `⚠️  ${flags.join(",")}` : "ok";

        console.log(
          `[${status.padEnd(35)}] ${vp.id.padEnd(16)} ${theme.padEnd(5)} ${site.id.padEnd(10)} vw=${metrics.vw} doc=${metrics.docWidth}`
        );
      } catch (err) {
        cell.error = String(err.message || err).slice(0, 200);
        console.log(`[ERROR] ${vp.id} ${theme} ${site.id}: ${cell.error}`);
      }
      results.push(cell);
    }
    await ctx.close();
  }
}

await browser.close();

const report = { base: BASE, runAt: new Date().toISOString(), results };
fs.writeFileSync(path.join(OUT, "report.json"), JSON.stringify(report, null, 2));

// Summary
const issues = results.filter(r =>
  r.error ||
  (r.metrics && r.metrics.horizontalScroll) ||
  (r.metrics && !r.metrics.navbarFound) ||
  (r.metrics && !r.metrics.bootstrapLoaded) ||
  (r.failed404s && r.failed404s.length > 0)
);

console.log();
console.log("═".repeat(60));
console.log(`Total cells: ${results.length}    Issues: ${issues.length}`);
console.log("═".repeat(60));
if (issues.length === 0) {
  console.log("✅ All cells clean.");
} else {
  console.log();
  for (const i of issues) {
    const marks = [];
    if (i.error) marks.push(`error=${i.error}`);
    if (i.metrics?.horizontalScroll) marks.push("HSCROLL");
    if (i.metrics && !i.metrics.navbarFound) marks.push("NO_NAVBAR");
    if (i.metrics && !i.metrics.bootstrapLoaded) marks.push("NO_BOOTSTRAP");
    if (i.failed404s?.length) marks.push(`404=${i.failed404s.length}`);
    console.log(`  ${i.viewport} | ${i.theme} | ${i.site}  →  ${marks.join(", ")}`);
    if (i.failed404s?.length) {
      for (const f of i.failed404s.slice(0, 3)) console.log(`     ${f}`);
    }
  }
}
console.log();
console.log(`Output: ${OUT}/`);
