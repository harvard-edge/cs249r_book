/**
 * Smoke test for the globally-mounted command palette + shortcuts overlay.
 *
 * Regression guard for the mount bug: CommandPalette and
 * KeyboardShortcutsOverlay were authored (with tests + docblocks claiming
 * they live in app/layout.tsx) but the mount line was never added, so the
 * navbar search icon, Cmd/Ctrl+K, and `?` were all no-ops on every route.
 *
 * Checks the three entry points that were dead:
 *   1. Ctrl+K opens the palette and focuses its input; Esc closes it.
 *   2. The navbar search icon dispatches staffml:open-palette → same palette.
 *   3. `?` opens the keyboard-shortcuts overlay.
 */
import { test, expect } from "@playwright/test";

test.describe("Command palette — global mount", () => {
  test("Ctrl+K opens the palette and Esc closes it", async ({ page }) => {
    await page.goto("/");

    // Closed by default — the component renders null when not open.
    const palette = page.getByRole("dialog", { name: /command palette/i });
    await expect(palette).toHaveCount(0);

    await page.keyboard.press("Control+k");
    await expect(palette).toBeVisible();

    // Input is focused on open.
    await expect(
      palette.getByPlaceholder(/search pages, topics, or questions/i),
    ).toBeFocused();

    // Pages section is available with an empty query.
    await expect(palette.getByRole("option", { name: /Practice/i }).first()).toBeVisible();

    await page.keyboard.press("Escape");
    await expect(palette).toHaveCount(0);
  });

  test("navbar search icon opens the same palette", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: /search \(cmd\+k\)/i }).click();
    await expect(page.getByRole("dialog", { name: /command palette/i })).toBeVisible();
  });

  test("? opens the keyboard-shortcuts overlay", async ({ page }) => {
    await page.goto("/");
    await page.keyboard.press("?");
    await expect(
      page.getByRole("dialog", { name: /keyboard shortcuts/i }),
    ).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(
      page.getByRole("dialog", { name: /keyboard shortcuts/i }),
    ).toHaveCount(0);
  });
});
