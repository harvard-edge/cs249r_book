import { defineConfig } from "@playwright/test";

/**
 * Playwright config for StaffML smoke tests.
 *
 * The Next.js dev server is started out-of-band so we control its
 * lifecycle and don't pay the dev-server boot cost on every test
 * run. Invoke:
 *
 *   npx next dev &                             # in one terminal
 *   npx playwright test tests/practice-smoke   # in another
 *
 * For CI / one-shot runs, pass --webServer on the command line to
 * launch the dev server inline.
 */
export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:3000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
});
