import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Run in Node; real Workers runtime is used at deploy time via miniflare.
    // These contract tests exercise the handler surface with mocked Env.
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
});
