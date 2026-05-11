// Capture before/after screenshots from the locally-served _build/.
// Saves to /tmp/local-renders/ with descriptive names so the user
// can flip through them.

import { webkit, devices } from "playwright";
import fs from "node:fs";

const OUT = "/tmp/local-renders";
fs.mkdirSync(OUT, { recursive: true });

const PAGES = [
  { id: "landing",          path: "/index.html" },
  { id: "about-license",    path: "/about/license.html" },
  { id: "about-index",      path: "/about/index.html" },
  { id: "community",        path: "/community/index.html" },
  { id: "newsletter",       path: "/newsletter/index.html" },
];

const VIEWPORTS = [
  { id: "desktop", device: { viewport: { width: 1280, height: 800 } } },
  { id: "iphone",  device: devices["iPhone 14 Pro"] },
];

const browser = await webkit.launch();
for (const vp of VIEWPORTS) {
  for (const theme of ["light", "dark"]) {
    const ctx = await browser.newContext({ ...vp.device, colorScheme: theme });
    const page = await ctx.newPage();
    for (const p of PAGES) {
      await page.goto("http://localhost:8765" + p.path, { waitUntil: "networkidle", timeout: 30000 });
      await page.waitForTimeout(500);
      const f = `${OUT}/RENDER-${vp.id}__${theme}__${p.id}.png`;
      await page.screenshot({ path: f, fullPage: false });
      console.log(`[ok] ${vp.id} ${theme} ${p.id}`);
    }
    await ctx.close();
  }
}
await browser.close();
console.log("done");
