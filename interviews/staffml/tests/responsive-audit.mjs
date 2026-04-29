// Responsive audit script: screenshot staffml.ai (or any base URL) at
// multiple device viewports across key pages, and dump per-page DOM
// metrics that surface horizontal-overflow / clipped-element issues.
//
// Usage:
//   node tests/responsive-audit.mjs                              # https://staffml.ai
//   BASE_URL=http://localhost:3000 node tests/responsive-audit.mjs
//
// Outputs:
//   tests/responsive-output/<viewport>-<page>.png
//   tests/responsive-output/report.json

import { chromium, webkit, devices } from "playwright";
import fs from "node:fs";
import path from "node:path";

const BASE_URL = process.env.BASE_URL || "https://staffml.ai";
const ENGINE = (process.env.ENGINE || "chromium").toLowerCase();
const OUT_DIR = path.resolve(process.env.OUT_DIR || "tests/responsive-output");
const ENGINES = { chromium, webkit };
if (!ENGINES[ENGINE]) {
  console.error(`Unknown ENGINE='${ENGINE}'. Use chromium or webkit.`);
  process.exit(1);
}

const viewports = [
  { name: "iphone-se",      device: devices["iPhone SE"] },
  { name: "iphone-14-pro",  device: devices["iPhone 14 Pro"] },
  { name: "ipad-mini-port", device: devices["iPad Mini"] },
  { name: "ipad-mini-land", device: devices["iPad Mini landscape"] },
  { name: "ipad-pro-port",  device: devices["iPad Pro 11"] },
  { name: "ipad-pro-land",  device: devices["iPad Pro 11 landscape"] },
  { name: "desktop-1280",   device: { viewport: { width: 1280, height: 800 }, userAgent: undefined } },
  { name: "desktop-1920",   device: { viewport: { width: 1920, height: 1080 }, userAgent: undefined } },
];

const pages = [
  { name: "home",       path: "/" },
  { name: "welcome",    path: "/welcome" },
  { name: "about",      path: "/about" },
  { name: "explore",    path: "/explore" },
  { name: "framework",  path: "/framework" },
  { name: "practice",   path: "/practice" },
  { name: "roofline",   path: "/roofline" },
  { name: "plans",      path: "/plans" },
  { name: "contribute", path: "/contribute" },
  { name: "dashboard",  path: "/dashboard" },
  { name: "gauntlet",   path: "/gauntlet" },
  { name: "progress",   path: "/progress" },
  { name: "simulator",  path: "/simulator" },
];

fs.mkdirSync(OUT_DIR, { recursive: true });

const report = { baseUrl: BASE_URL, runAt: new Date().toISOString(), results: [] };

const browser = await ENGINES[ENGINE].launch();
console.log(`engine=${ENGINE} base=${BASE_URL} out=${OUT_DIR}`);

for (const vp of viewports) {
  const context = await browser.newContext({
    ...vp.device,
    deviceScaleFactor: 1,
    ignoreHTTPSErrors: true,
  });
  const page = await context.newPage();

  for (const p of pages) {
    const url = BASE_URL.replace(/\/$/, "") + p.path;
    const result = { viewport: vp.name, page: p.name, url };
    try {
      await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
      await page.waitForTimeout(800);

      // Measure overflow + body sizes
      const metrics = await page.evaluate(() => {
        const docEl = document.documentElement;
        const body = document.body;
        const vw = window.innerWidth;
        const overflowing = [];
        const all = document.querySelectorAll("body *");
        for (const el of all) {
          const r = el.getBoundingClientRect();
          // overflow > 2px to ignore subpixel rounding
          if (r.right > vw + 2 || r.left < -2) {
            const id = el.id ? `#${el.id}` : "";
            const cls = (typeof el.className === "string" && el.className)
              ? "." + el.className.trim().split(/\s+/).slice(0, 3).join(".")
              : "";
            overflowing.push({
              tag: el.tagName.toLowerCase(),
              sel: el.tagName.toLowerCase() + id + cls,
              left: Math.round(r.left),
              right: Math.round(r.right),
              width: Math.round(r.width),
              text: (el.textContent || "").trim().slice(0, 60),
            });
            if (overflowing.length >= 25) break;
          }
        }
        return {
          vw,
          docWidth: docEl.scrollWidth,
          bodyWidth: body.scrollWidth,
          horizontalScroll: docEl.scrollWidth > vw + 2,
          overflowing,
        };
      });

      result.metrics = metrics;
      const screenshotPath = path.join(OUT_DIR, `${vp.name}--${p.name}.png`);
      await page.screenshot({ path: screenshotPath, fullPage: true });
      result.screenshot = screenshotPath;
      console.log(
        `[ok] ${vp.name.padEnd(18)} ${p.name.padEnd(11)} ` +
          `vw=${metrics.vw} doc=${metrics.docWidth} ` +
          `${metrics.horizontalScroll ? "HSCROLL" : "ok"} ` +
          `overflowEls=${metrics.overflowing.length}`
      );
    } catch (err) {
      result.error = String(err.message || err);
      console.log(`[err] ${vp.name} ${p.name}: ${result.error}`);
    }
    report.results.push(result);
  }

  await context.close();
}

await browser.close();

fs.writeFileSync(
  path.join(OUT_DIR, "report.json"),
  JSON.stringify(report, null, 2)
);
console.log(`\nWrote ${report.results.length} results to ${OUT_DIR}`);
