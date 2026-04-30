import { chromium } from "playwright";

const BASE = "http://localhost:3457";
const SHOTS = "/tmp/stargate-shots";

const browser = await chromium.launch();
const errors = [];

/**
 * Creates a fresh context with the lifetime reveal counter pre-bumped.
 * The init script intentionally only sets reveals — it does NOT touch
 * `staffml_star_gate`, so dismissals persist across reloads.
 */
async function newPrimedContext({ theme = "light", suppressFirstRun = false } = {}) {
  const ctx = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    colorScheme: theme === "dark" ? "dark" : "light",
  });
  await ctx.addInitScript(([primeTheme, sFR]) => {
    try {
      localStorage.setItem("staffml_lifetime_reveals", "5");
      localStorage.setItem("staffml_theme", primeTheme);
      document.documentElement.classList.toggle("dark", primeTheme === "dark");
      document.documentElement.dataset.theme = primeTheme;
      if (sFR) {
        localStorage.setItem("staffml_firstrun_gauntlet", "1");
        localStorage.setItem("staffml_firstrun_practice", "1");
      }
    } catch {}
  }, [theme, suppressFirstRun]);
  return ctx;
}

async function attachConsole(page, label) {
  page.on("pageerror", (e) => {
    if (e.message?.includes("Failed to fetch")) return;
    errors.push(`[${label}] pageerror: ${e.message}`);
  });
  page.on("console", (msg) => {
    if (msg.type() === "error") {
      const t = msg.text();
      if (t.includes("staffml-vault.mlsysbook-ai-account.workers.dev")) return;
      if (t.includes("Failed to load resource")) return;
      if (t.includes("Failed to fetch")) return;
      errors.push(`[${label}] console.error: ${t}`);
    }
  });
}

async function gateScreenshot(label, theme) {
  const ctx = await newPrimedContext({ theme, suppressFirstRun: true });
  const page = await ctx.newPage();
  await attachConsole(page, `${label}/${theme}`);

  // Visit once to seed origin, then set theme + reload so the synchronous
  // theme-bootstrap (/public/theme-bootstrap.js) picks up our choice on
  // first paint of the *real* render.
  await page.goto(`${BASE}/practice`, { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.evaluate((t) => {
    localStorage.setItem("staffml_theme", t);
    localStorage.setItem("quarto-color-scheme", t);
  }, theme);
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.waitForTimeout(1500);

  const longAnswer =
    "My answer walks through the bandwidth calculation and argues the dominant cost is cross-node communication over the network fabric, which saturates around 200 GB/s in InfiniBand deployments — so the bottleneck moves from compute to fabric once we leave the node.";
  const ta = page.locator("textarea").first();
  await ta.waitFor({ state: "visible", timeout: 15000 });
  await ta.fill(longAnswer);
  await page.getByRole("button", { name: /reveal answer/i }).click();
  await page.waitForTimeout(1500);

  await page.screenshot({
    path: `${SHOTS}/${label}-01-gate-${theme}.png`,
    fullPage: false,
  });

  await page.getByRole("button", { name: /already starred/i }).click();
  await page.waitForTimeout(900);
  await page.screenshot({
    path: `${SHOTS}/${label}-02-thanks-${theme}.png`,
    fullPage: false,
  });

  await ctx.close();
}

async function dismissPersists(theme) {
  const ctx = await newPrimedContext({ theme, suppressFirstRun: true });
  const page = await ctx.newPage();
  await attachConsole(page, `dismiss/${theme}`);

  await page.goto(`${BASE}/practice`, { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.waitForTimeout(1500);
  await page.locator("textarea").first().fill(
    "A long enough answer to bypass the think-guard so the reveal goes through and surfaces the star gate cleanly here."
  );
  await page.getByRole("button", { name: /reveal answer/i }).click();
  await page.waitForTimeout(1200);

  const gateVisible = await page.getByRole("button", { name: /already starred/i }).isVisible();
  if (!gateVisible) errors.push(`[dismiss/${theme}] gate did not appear before dismiss`);

  await page.getByRole("button", { name: "Close", exact: true }).click();
  await page.waitForTimeout(700);

  const stillVisibleNow = await page.getByRole("button", { name: /already starred/i }).count();
  if (stillVisibleNow > 0) errors.push(`[dismiss/${theme}] gate still visible right after X click`);

  // Bump reveals high; reload; gate should still NOT reappear.
  await page.evaluate(() => {
    localStorage.setItem("staffml_lifetime_reveals", "10");
  });
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.waitForTimeout(1500);
  await page.locator("textarea").first().fill(
    "Yet another long answer over fifty characters and signaling deliberation past twenty seconds easily here."
  );
  await page.getByRole("button", { name: /reveal answer/i }).click();
  await page.waitForTimeout(1200);

  const reAppeared = await page.getByRole("button", { name: /already starred/i }).count();
  if (reAppeared > 0) {
    errors.push(`[dismiss/${theme}] gate re-appeared after dismissal — should retire forever`);
  }
  await page.screenshot({ path: `${SHOTS}/dismiss-aftermath-${theme}.png`, fullPage: false });

  await ctx.close();
}

async function gauntletGate(theme) {
  const ctx = await newPrimedContext({ theme, suppressFirstRun: true });
  const page = await ctx.newPage();
  await attachConsole(page, `gauntlet/${theme}`);

  await page.goto(`${BASE}/gauntlet`, { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${SHOTS}/gauntlet-setup-${theme}.png`, fullPage: false });

  // Try a few likely action button names.
  const begin = page
    .getByRole("button", { name: /begin|start gauntlet|start mock|start interview/i })
    .first();
  if (await begin.count()) {
    await begin.click();
    await page.waitForTimeout(2500);
  } else {
    errors.push(`[gauntlet/${theme}] no Begin/Start button found — see gauntlet-setup-${theme}.png`);
    await ctx.close();
    return;
  }

  const ta = page.locator("textarea").first();
  if (await ta.count()) {
    await ta.fill(
      "A long enough answer that bypasses any deliberation guard if one fires here in the gauntlet flow."
    );
  }
  const reveal = page.getByRole("button", { name: /reveal/i }).first();
  if (await reveal.count()) {
    await reveal.click();
    await page.waitForTimeout(1500);
  } else {
    errors.push(`[gauntlet/${theme}] no Reveal button after start`);
  }

  await page.screenshot({ path: `${SHOTS}/gauntlet-after-reveal-${theme}.png`, fullPage: false });

  const gate = await page.getByRole("button", { name: /already starred/i }).count();
  if (gate === 0) {
    errors.push(`[gauntlet/${theme}] star gate did NOT trigger on reveal`);
  }

  await ctx.close();
}

await gateScreenshot("practice", "light");
await gateScreenshot("practice", "dark");
await dismissPersists("light");
await gauntletGate("light");

await browser.close();

if (errors.length) {
  console.error("ERRORS:");
  for (const e of errors) console.error("  " + e);
  process.exit(1);
}
console.log("done — no errors");
