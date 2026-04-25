/**
 * Smoke test for the restructured practice page (commit cb0b7ea30).
 *
 * Verifies the new layout for ANY question from the bundled corpus —
 * we don't depend on a specific question ID so the test stays stable
 * as the corpus grows. The visual-rendering path is covered by a
 * separate spec once cloud-visual-001 ships in the release bundle.
 *
 * Checks:
 *   1. Page renders without console/pageerror.
 *   2. Key layout landmarks are present in their new positions:
 *      sticky Your-task callout, scenario prose, answer textarea,
 *      Reveal button, "Stuck? Ask Interviewer" nudge, right-column
 *      Tools header.
 *   3. Submit-gradient safeguard intercepts low-effort reveals
 *      (<15s elapsed AND <50 chars typed).
 *   4. Full flow: substantive answer bypasses the guard; Reveal
 *      lands us in post-reveal with the Model Answer visible.
 */
import { test, expect } from "@playwright/test";

test.describe("Practice page — restructured layout", () => {
  test("renders without errors and shows all layout landmarks", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(`console.error: ${msg.text()}`);
    });

    await page.goto("/practice");

    // Wait for the main question to hydrate from the bundle.
    // The sidebar has its own <h2>Practice</h2>; target the question
    // title specifically via its larger text classes.
    const questionTitle = page.locator("h2.text-2xl, h2.lg\\:text-3xl").first();
    await expect(questionTitle).toBeVisible({ timeout: 10000 });

    // Sticky Your-task header — uses the "Your task" label in either
    // authored-question or inferred-fallback form.
    await expect(page.getByText(/Your task/).first()).toBeVisible();

    // Scenario prose is present (any non-empty <p> after the sticky header).
    await expect(page.locator("p").filter({ hasText: /.+/ }).first()).toBeVisible();

    // Answer textarea is in the LEFT column (not the old right panel).
    const textarea = page.locator("textarea");
    await expect(textarea).toBeVisible();

    // Reveal Answer button directly below the textarea.
    const revealBtn = page.getByRole("button", { name: /reveal answer/i });
    await expect(revealBtn).toBeVisible();

    // Stuck-nudge is the beginner scaffolding affordance.
    await expect(page.getByText(/stuck/i)).toBeVisible();

    // Right-column Tools panel header (inside role=main to exclude
    // the global nav's Tools button).
    await expect(page.getByRole("main").getByText("Tools", { exact: true })).toBeVisible();

    // No React crashes or console errors.
    expect(errors, `page errors: ${errors.join("\n")}`).toEqual([]);
  });

  test("submit-gradient safeguard fires on low-effort reveal", async ({ page }) => {
    await page.goto("/practice");
    // The sidebar has its own <h2>Practice</h2>; target the question
    // title specifically via its larger text classes.
    const questionTitle = page.locator("h2.text-2xl, h2.lg\\:text-3xl").first();
    await expect(questionTitle).toBeVisible({ timeout: 10000 });

    // Type <50 chars and reveal — should trip the think-guard.
    await page.locator("textarea").fill("idk");
    await page.getByRole("button", { name: /reveal answer/i }).click();

    // Modal dialog appears with the Think-longer? copy. Scope by the
    // accessible name so we don't collide with the sidebar filter
    // drawer (which also uses role=dialog).
    const thinkDialog = page.getByRole("dialog", { name: /think longer/i });
    await expect(thinkDialog).toBeVisible();
    await expect(thinkDialog.getByRole("button", { name: /keep thinking/i })).toBeVisible();
    await expect(thinkDialog.getByRole("button", { name: /reveal anyway/i })).toBeVisible();

    // "Keep thinking" dismisses and keeps us in pre-reveal.
    await thinkDialog.getByRole("button", { name: /keep thinking/i }).click();
    await expect(thinkDialog).not.toBeVisible();
    await expect(page.getByRole("button", { name: /reveal answer/i })).toBeVisible();
  });

  test("substantive answer bypasses guard and reveals model answer", async ({ page }) => {
    await page.goto("/practice");
    // The sidebar has its own <h2>Practice</h2>; target the question
    // title specifically via its larger text classes.
    const questionTitle = page.locator("h2.text-2xl, h2.lg\\:text-3xl").first();
    await expect(questionTitle).toBeVisible({ timeout: 10000 });

    // Type >80 chars so the deliberation-calibration threshold is met.
    const longAnswer =
      "My answer walks through the bandwidth calculation and argues that the dominant cost is the cross-node communication over the network fabric, which saturates at roughly 200 GB/s in typical datacenter InfiniBand deployments.";
    await page.locator("textarea").fill(longAnswer);

    await page.getByRole("button", { name: /reveal answer/i }).click();

    // No guard dialog — should transition to post-reveal immediately.
    await expect(page.getByRole("dialog", { name: /think longer/i })).not.toBeVisible();

    // Post-reveal landmark: the "Model Answer" section label (exact
    // match — the napkin-math result line also contains "Model answer").
    await expect(page.getByText("Model Answer", { exact: true })).toBeVisible({ timeout: 5000 });

    // Self-assessment buttons (1-4 scoring).
    await expect(page.getByRole("button", { name: /nailed it/i })).toBeVisible();
  });
});

/**
 * Smoke tests covering yesterday's filter additions and the
 * not-found-question fix.
 */
test.describe("Practice page — filters and deep-links", () => {
  test("visual filter at L5 returns a non-empty pool with an inline SVG", async ({ page }) => {
    await page.goto("/practice");
    await expect(page.locator("h2.text-2xl, h2.lg\\:text-3xl").first())
      .toBeVisible({ timeout: 10000 });

    // Switch to L5 (where most published visuals live).
    await page.getByRole("button", { name: /^L5\b/ }).click();
    await page.waitForTimeout(300);

    // Toggle visual filter.
    await page.getByText("Visual questions only").click();
    await page.waitForTimeout(500);

    // Pool counter shows N questions in pool, N > 0.
    const pool = page.locator("text=/\\d+ questions in pool/").first();
    const text = (await pool.textContent()) || "";
    const n = parseInt(text.match(/(\d+) questions/)?.[1] ?? "0", 10);
    expect(n).toBeGreaterThan(0);

    // The surfaced question renders an SVG <img> from /question-visuals/.
    const main = page.getByRole("main");
    await expect(main.locator("img[src*='/question-visuals/']").first())
      .toBeVisible({ timeout: 5000 });
  });

  test("chained-only filter reduces pool but stays non-empty", async ({ page }) => {
    await page.goto("/practice");
    await expect(page.locator("h2.text-2xl, h2.lg\\:text-3xl").first())
      .toBeVisible({ timeout: 10000 });

    const before = (await page.locator("text=/\\d+ questions in pool/").first()
      .textContent()) || "";
    const beforeN = parseInt(before.match(/(\d+) questions/)?.[1] ?? "0", 10);

    await page.getByText("Chained questions only").click();
    await page.waitForTimeout(500);

    const after = (await page.locator("text=/\\d+ questions in pool/").first()
      .textContent()) || "";
    const afterN = parseInt(after.match(/(\d+) questions/)?.[1] ?? "0", 10);

    expect(afterN).toBeGreaterThan(0);
    expect(afterN).toBeLessThan(beforeN);
  });

  test("?q=<known> deep-link surfaces the right question with its visual", async ({ page }) => {
    // cloud-2847 was promoted to published in this branch; confirm the deep
    // link lands on it AND its matplotlib-rendered SVG renders inline.
    await page.goto("/practice?q=cloud-2847");
    await page.waitForTimeout(1500);

    // Title matches the YAML
    const title = await page.locator("h2.text-2xl, h2.lg\\:text-3xl").first().textContent();
    expect(title?.toLowerCase()).toContain("hockey");

    // Visual rendered
    const main = page.getByRole("main");
    await expect(main.locator("img[src*='cloud-2847.svg']")).toBeVisible({ timeout: 5000 });
  });

  test("?q=<unknown> shows the not-found banner instead of silent fallthrough", async ({ page }) => {
    await page.goto("/practice?q=cloud-this-id-does-not-exist");
    await page.waitForTimeout(1000);

    // Banner is visible and names the bad id
    await expect(page.getByRole("alert").filter({ hasText: /cloud-this-id-does-not-exist/i }))
      .toBeVisible();
    await expect(page.getByText(/isn.t in the published bundle/i)).toBeVisible();

    // Default question pool is still shown below
    await expect(page.locator("h2.text-2xl, h2.lg\\:text-3xl").first()).toBeVisible();
  });
});
