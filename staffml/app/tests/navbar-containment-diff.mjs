// Diagnostic: WHY does Quarto's navbar not produce horizontal page scroll
// at iPad landscape, while StaffML's EcosystemBar does?
//
// This script visits both URLs at iPad landscape (1024×768) using WebKit,
// finds elements whose bounding rect extends past the viewport, and walks
// UP the parent chain dumping (overflowX, scrollWidth, clientWidth, position,
// display, flexWrap) for each ancestor. The difference between Quarto's
// containment and StaffML's leak should fall out of the diff.
//
// Usage:
//   node tests/navbar-containment-diff.mjs

import { webkit } from "playwright";

const TARGETS = [
  { name: "STAFFML local (post-fix)", url: "http://localhost:3010/" },
  { name: "QUARTO vol1 (reference)",  url: "https://mlsysbook.ai/vol1/" },
  { name: "STAFFML staffml.ai (live, pre-fix)", url: "https://staffml.ai/" },
];

const VIEWPORT = { width: 1024, height: 768 };

const browser = await webkit.launch();
const context = await browser.newContext({ viewport: VIEWPORT, deviceScaleFactor: 1 });
const page = await context.newPage();

for (const t of TARGETS) {
  console.log("\n" + "=".repeat(78));
  console.log(`${t.name}  ${t.url}`);
  console.log("=".repeat(78));

  await page.goto(t.url, { waitUntil: "networkidle", timeout: 30000 });
  await page.waitForTimeout(800);

  const dump = await page.evaluate(() => {
    const vw = window.innerWidth;
    const docEl = document.documentElement;
    const body = document.body;

    // Helper: describe an element succinctly.
    const desc = (el) => {
      if (!el || el === document) return "document";
      const tag = el.tagName ? el.tagName.toLowerCase() : "?";
      const id = el.id ? `#${el.id}` : "";
      const cls = (typeof el.className === "string" && el.className)
        ? "." + el.className.trim().split(/\s+/).slice(0, 4).join(".")
        : "";
      return tag + id + cls;
    };

    // Helper: pull the styles we care about.
    const styleSnapshot = (el) => {
      const cs = getComputedStyle(el);
      const r = el.getBoundingClientRect();
      return {
        sel: desc(el),
        overflowX: cs.overflowX,
        overflowY: cs.overflowY,
        position: cs.position,
        display: cs.display,
        flexWrap: cs.flexWrap,
        width: Math.round(r.width),
        scrollWidth: el.scrollWidth,
        clientWidth: el.clientWidth,
        left: Math.round(r.left),
        right: Math.round(r.right),
      };
    };

    // Find an element that overflows the viewport on the right.
    // Prefer something inside the navbar so the chain is interesting.
    const candidates = Array.from(document.querySelectorAll("body *"));
    const overflowing = candidates.filter((el) => {
      const r = el.getBoundingClientRect();
      return r.right > vw + 2 && r.width < 2000 && r.width > 0;
    });

    // Also flag whether the page actually has horizontal page-scroll.
    const docScroll = docEl.scrollWidth;
    const bodyScroll = body.scrollWidth;
    const pageHasHScroll = docScroll > vw + 2 || bodyScroll > vw + 2;

    // Pick a representative overflowing element for the chain trace.
    // Prefer ones inside .navbar / [class*=nav], else fall back to first.
    const inNav = overflowing.find((el) => el.closest(".navbar, header, nav, [class*=Navbar], [class*=navbar]"));
    const subject = inNav || overflowing[0];

    const chain = [];
    if (subject) {
      let cur = subject;
      while (cur && cur !== document) {
        chain.push(styleSnapshot(cur));
        cur = cur.parentElement;
      }
    }

    return {
      vw,
      docScrollWidth: docScroll,
      bodyScrollWidth: bodyScroll,
      pageHasHScroll,
      overflowingCount: overflowing.length,
      sample: overflowing.slice(0, 5).map(styleSnapshot),
      subject: subject ? desc(subject) : null,
      chain,
    };
  });

  console.log(`vw=${dump.vw}  doc.scrollWidth=${dump.docScrollWidth}  body.scrollWidth=${dump.bodyScrollWidth}  pageHasHScroll=${dump.pageHasHScroll}`);
  console.log(`overflowing elements: ${dump.overflowingCount}`);

  if (dump.sample.length) {
    console.log("\n  Sample overflowing elements:");
    for (const s of dump.sample) {
      console.log(`    ${s.sel}`);
      console.log(`      box: L=${s.left} R=${s.right} W=${s.width}  overflowX=${s.overflowX}`);
    }
  }

  if (dump.chain.length) {
    console.log(`\n  Ancestor chain from overflowing subject (${dump.subject}):`);
    console.log(`    ${"sel".padEnd(60)} overflowX  pos       display      flexWrap   width   scrollW`);
    for (const c of dump.chain) {
      console.log(
        `    ${c.sel.slice(0, 60).padEnd(60)} ${c.overflowX.padEnd(10)} ${c.position.padEnd(9)} ${c.display.padEnd(12)} ${(c.flexWrap || "-").padEnd(10)} ${String(c.width).padEnd(7)} ${c.scrollWidth}`
      );
    }
  }
}

await browser.close();
