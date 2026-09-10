import { webkit } from "playwright";
const browser = await webkit.launch();
const ctx = await browser.newContext({ colorScheme: "dark", viewport: { width: 1280, height: 800 } });
const page = await ctx.newPage();
await page.goto("http://localhost:8765/about/license.html", { waitUntil: "networkidle" });
await page.waitForTimeout(800);
const inspect = await page.evaluate(() => {
  const body = document.body;
  const h1 = document.querySelector(".about-title");
  const cs = getComputedStyle(h1);
  return {
    htmlAttrs: {
      "data-bs-theme": document.documentElement.getAttribute("data-bs-theme"),
      "data-quarto-color-scheme": document.documentElement.getAttribute("data-quarto-color-scheme"),
    },
    bodyClasses: Array.from(body.classList),
    h1Color: cs.color,
    h1ColorVariable: cs.getPropertyValue("--ab-text"),
    rootAbText: getComputedStyle(document.documentElement).getPropertyValue("--ab-text"),
    bodyAbText: getComputedStyle(body).getPropertyValue("--ab-text"),
  };
});
console.log(JSON.stringify(inspect, null, 2));
await browser.close();
