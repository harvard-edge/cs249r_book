import { webkit, devices } from "playwright";
const browser = await webkit.launch();
for (const [name, dev] of [
  ["desktop-1280", { viewport: { width: 1280, height: 800 } }],
  ["ipad-mini-land", devices["iPad Mini landscape"]],
  ["iphone-14-pro", devices["iPhone 14 Pro"]],
]) {
  for (const theme of ["light", "dark"]) {
    const ctx = await browser.newContext({ ...dev, colorScheme: theme });
    const page = await ctx.newPage();
    await page.goto("https://mlsysbook.ai/index.html", { waitUntil: "networkidle", timeout: 30000 });
    await page.waitForTimeout(800);
    await page.screenshot({ path: `/tmp/full-ecosystem-audit/LANDING-DIRECT-${name}__${theme}.png`, fullPage: false });
    await ctx.close();
  }
}
await browser.close();
console.log("done");
