// Dark-mode consistency audit for the about/community/landing-subsite pages.
//
// Captures one screenshot per page in light + dark mode, plus a per-cell
// integrity record:
//   - per-element computed color contrast against background
//   - any text whose foreground is "almost the same as" the background
//     (foreground luminance within 0.1 of background luminance — the
//     canonical "Open by design." failure mode the user reported on
//     /about/license.html in dark mode)
//   - any element whose background is hard-coded white (#fff / #ffffff /
//     rgb(255,255,255)) inside a dark-mode page — those are candidates
//     for theme drift
//   - count of any unstyled/very-low-contrast text nodes per page

import { webkit } from "playwright";
import fs from "node:fs";
import path from "node:path";

const BASE = (process.env.BASE || "https://mlsysbook.ai").replace(/\/$/, "");
const OUT  = path.resolve(process.env.OUT || "/tmp/darkmode-audit");
fs.mkdirSync(OUT, { recursive: true });

const PAGES = [
  // Landing & about
  { id: "landing-index",    path: "/index.html" },
  { id: "about-home",       path: "/about/" },
  { id: "about-license",    path: "/about/license.html" },
  { id: "about-people",     path: "/about/people.html" },
  { id: "about-contributors", path: "/about/contributors.html" },
  // Community
  { id: "community-home",   path: "/community/" },
  { id: "community-events", path: "/community/events.html" },
  { id: "community-partners", path: "/community/partners.html" },
  // Newsletter
  { id: "newsletter-home",  path: "/newsletter/" },
];

const browser = await webkit.launch();
const ctx = await browser.newContext({
  viewport: { width: 1440, height: 900 },
  colorScheme: "dark",
  deviceScaleFactor: 1,
  ignoreHTTPSErrors: true,
});
const page = await ctx.newPage();

console.log(`Dark-mode audit base=${BASE}, ${PAGES.length} pages`);
const results = [];

for (const p of PAGES) {
  const url = BASE + p.path;
  const cell = { id: p.id, url };
  try {
    await page.goto(url, { waitUntil: "networkidle", timeout: 45000 });
    await page.waitForTimeout(800);

    const findings = await page.evaluate(() => {
      function rgbToLum(rgbStr) {
        const m = rgbStr && rgbStr.match(/(\d+(\.\d+)?)/g);
        if (!m || m.length < 3) return null;
        const [r, g, b] = m.slice(0, 3).map(Number).map((v) => v / 255);
        const lin = (c) => (c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
        return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
      }
      const bodyBg = getComputedStyle(document.body).backgroundColor;
      const docBg  = getComputedStyle(document.documentElement).backgroundColor;
      const bgLum  = rgbToLum(bodyBg) ?? rgbToLum(docBg) ?? 0;

      const lowContrast = [];
      const whiteBgInDark = [];

      const all = document.querySelectorAll("body *");
      let n = 0;
      for (const el of all) {
        if (n > 10000) break; // safety cap
        n++;
        // Only check elements with text nodes
        const hasOwnText = Array.from(el.childNodes).some(
          (c) => c.nodeType === Node.TEXT_NODE && c.textContent.trim().length > 2,
        );
        const cs = getComputedStyle(el);
        const r = el.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) continue;
        if (cs.visibility === "hidden" || cs.display === "none") continue;

        if (hasOwnText) {
          const fgLum = rgbToLum(cs.color);
          if (fgLum != null && Math.abs(fgLum - bgLum) < 0.05) {
            lowContrast.push({
              tag: el.tagName.toLowerCase(),
              cls: (typeof el.className === "string" ? el.className : "").slice(0, 60),
              text: (el.textContent || "").trim().slice(0, 80),
              color: cs.color,
              bg:    cs.backgroundColor,
              fgLum: Math.round(fgLum * 1000) / 1000,
            });
            if (lowContrast.length >= 8) /* cap */ {}
          }
        }
        // Hard-coded white bg?
        const bg = cs.backgroundColor;
        if (
          (bg === "rgb(255, 255, 255)" || bg === "rgba(255, 255, 255, 1)") &&
          bgLum < 0.3 // body is dark
        ) {
          whiteBgInDark.push({
            tag: el.tagName.toLowerCase(),
            cls: (typeof el.className === "string" ? el.className : "").slice(0, 60),
            text: (el.textContent || "").trim().slice(0, 60),
          });
          if (whiteBgInDark.length >= 8) /* cap */ {}
        }
      }
      return {
        bodyBg,
        bgLum: Math.round(bgLum * 1000) / 1000,
        scanned: n,
        lowContrast: lowContrast.slice(0, 8),
        whiteBgInDark: whiteBgInDark.slice(0, 8),
      };
    });

    cell.findings = findings;
    const shotName = `${p.id}__dark.png`;
    await page.screenshot({ path: path.join(OUT, shotName), fullPage: true });
    cell.screenshot = shotName;

    const flags = [];
    if (findings.lowContrast.length) flags.push(`LOWCONTRAST=${findings.lowContrast.length}`);
    if (findings.whiteBgInDark.length) flags.push(`WHITEBG=${findings.whiteBgInDark.length}`);
    const status = flags.length ? `⚠️  ${flags.join(",")}` : "ok";
    console.log(`[${status.padEnd(40)}] ${p.id.padEnd(22)} bgLum=${findings.bgLum}`);
  } catch (err) {
    cell.error = String(err.message || err).slice(0, 200);
    console.log(`[ERROR] ${p.id}: ${cell.error}`);
  }
  results.push(cell);
}

await ctx.close();
await browser.close();

fs.writeFileSync(path.join(OUT, "report.json"), JSON.stringify({ base: BASE, runAt: new Date().toISOString(), results }, null, 2));

console.log();
console.log("═".repeat(70));
console.log("Dark-mode findings:");
console.log("═".repeat(70));
for (const r of results) {
  if (r.error) {
    console.log(`\n${r.id}: ERROR — ${r.error}`);
    continue;
  }
  if (!r.findings.lowContrast.length && !r.findings.whiteBgInDark.length) continue;
  console.log(`\n${r.id} (bgLum=${r.findings.bgLum})`);
  if (r.findings.lowContrast.length) {
    console.log("  Low-contrast text (foreground ~= background):");
    for (const lc of r.findings.lowContrast.slice(0, 4)) {
      console.log(`    <${lc.tag}.${lc.cls}> "${lc.text}"`);
      console.log(`      color=${lc.color}  bg=${lc.bg}  fgLum=${lc.fgLum}`);
    }
  }
  if (r.findings.whiteBgInDark.length) {
    console.log("  White-bg elements inside dark page:");
    for (const wb of r.findings.whiteBgInDark.slice(0, 4)) {
      console.log(`    <${wb.tag}.${wb.cls}> "${wb.text}"`);
    }
  }
}
console.log();
console.log(`Output: ${OUT}/`);
