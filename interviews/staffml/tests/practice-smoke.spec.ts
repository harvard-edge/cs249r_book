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
