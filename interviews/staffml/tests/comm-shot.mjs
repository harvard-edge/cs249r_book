import { webkit } from "playwright";
const browser = await webkit.launch();
const ctx = await browser.newContext({ colorScheme: "dark", viewport: { width: 1280, height: 800 } });
const page = await ctx.newPage();
await page.goto("http://localhost:8765/community/index.html", { waitUntil: "networkidle" });
await page.waitForTimeout(800);
await page.screenshot({ path: "/tmp/local-renders/RENDER2-desktop__dark__community.png", fullPage: false });
await browser.close();
console.log("done");
