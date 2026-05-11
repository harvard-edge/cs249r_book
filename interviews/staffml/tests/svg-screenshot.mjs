import { webkit } from "playwright";
const browser = await webkit.launch();
const ctx = await browser.newContext({ viewport: { width: 900, height: 720 } });
const page = await ctx.newPage();
await page.goto("file:///tmp/local-renders/svg-preview.html", { waitUntil: "networkidle" });
await page.screenshot({ path: "/tmp/local-renders/RENDER-svg-redesign.png", fullPage: true });
await browser.close();
console.log("done");
