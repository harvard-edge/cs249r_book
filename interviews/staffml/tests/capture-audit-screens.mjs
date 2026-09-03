import { chromium, devices } from "playwright";
import fs from "node:fs";
import path from "node:path";

const OUT = "/tmp/staffml-audit-screens";
fs.mkdirSync(OUT, { recursive: true });

const BASE_URL = "http://localhost:3456";

const PAGES = [
  { id: "01-landing", path: "/" },
  { id: "02-practice", path: "/practice" },
  { id: "03-roofline", path: "/roofline" },
  { id: "04-simulator", path: "/simulator" },
  { id: "05-gauntlet", path: "/gauntlet" },
  { id: "06-interview", path: "/interview" },
  { id: "07-progress", path: "/progress" },
  { id: "08-explore", path: "/explore" },
  { id: "09-framework", path: "/framework" },
  { id: "10-plans", path: "/plans" },
  { id: "11-about", path: "/about" },
];

async function run() {
  const browser = await chromium.launch({ headless: true });

  // 1. Desktop Light & Dark
  for (const theme of ["light", "dark"]) {
    const ctx = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      colorScheme: theme,
    });
    const page = await ctx.newPage();

    for (const p of PAGES) {
      const url = `${BASE_URL}${p.path}`;
      try {
        await page.goto(url, { waitUntil: "networkidle", timeout: 15000 });
        await page.waitForTimeout(600);

        // Set theme properly in localStorage and DOM
        await page.evaluate((t) => {
          localStorage.setItem("staffml_theme", t);
          document.documentElement.dataset.theme = t;
          if (t === "dark") {
            document.documentElement.classList.add("dark");
          } else {
            document.documentElement.classList.remove("dark");
          }
        }, theme);
        await page.waitForTimeout(300);

        const filename = path.join(OUT, `desktop-${theme}-${p.id}.png`);
        await page.screenshot({ path: filename, fullPage: false });
        console.log(`[captured] desktop ${theme} ${p.id}`);

        // For practice page, let's also capture revealed answer state
        if (p.id === "02-practice") {
          try {
            // Click reveal answer if present
            const revealBtn = page.locator("button:has-text('Reveal Model Answer'), button:has-text('Reveal'), button:has-text('Show Answer')").first();
            if (await revealBtn.isVisible()) {
              await revealBtn.click();
              await page.waitForTimeout(400);
              const revealFilename = path.join(OUT, `desktop-${theme}-02-practice-revealed.png`);
              await page.screenshot({ path: revealFilename, fullPage: false });
              console.log(`[captured] desktop ${theme} practice-revealed`);
            }
          } catch (e) {
            console.log(`practice reveal interaction skipped: ${e.message}`);
          }
        }
      } catch (err) {
        console.error(`Error on ${p.id} (${theme}):`, err.message);
      }
    }
    await ctx.close();
  }

  // 2. Mobile Viewport (iPhone 14 Pro)
  const mobileCtx = await browser.newContext({
    ...devices["iPhone 14 Pro"],
    colorScheme: "dark",
  });
  const mobilePage = await mobileCtx.newPage();
  for (const p of PAGES.slice(0, 6)) {
    const url = `${BASE_URL}${p.path}`;
    try {
      await mobilePage.goto(url, { waitUntil: "networkidle", timeout: 15000 });
      await mobilePage.evaluate(() => document.documentElement.classList.add("dark"));
      await mobilePage.waitForTimeout(500);
      const filename = path.join(OUT, `mobile-dark-${p.id}.png`);
      await mobilePage.screenshot({ path: filename, fullPage: false });
      console.log(`[captured] mobile dark ${p.id}`);
    } catch (err) {
      console.error(`Error on mobile ${p.id}:`, err.message);
    }
  }
  await mobileCtx.close();

  await browser.close();
  console.log(`\nAll screenshots saved to ${OUT}`);
}

run().catch(console.error);
