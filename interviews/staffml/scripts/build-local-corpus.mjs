#!/usr/bin/env node
/**
 * Auto-run before `npm run dev` so the local Next.js dev server can
 * serve the question corpus from disk via NEXT_PUBLIC_VAULT_FALLBACK=static.
 *
 * What it does:
 *   1. Looks for the `vault` CLI on PATH.
 *   2. Runs `vault build --local` from the repo root, which writes:
 *        interviews/staffml/src/data/corpus.json   (legacy bundle)
 *        interviews/staffml/public/data/corpus.json (the path the loader fetches)
 *      and mirrors visual SVGs into public/question-visuals/.
 *
 * Skipped silently if `vault` is not installed (e.g. on a fresh checkout
 * that hasn't run `pip install -e interviews/vault-cli` yet). The dev
 * server still boots; it just falls back to the production worker for
 * scenario/details, which is the same behavior contributors got before
 * this hook was wired in.
 *
 * Override: set STAFFML_SKIP_LOCAL_CORPUS=1 to bypass entirely.
 */
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

if (process.env.STAFFML_SKIP_LOCAL_CORPUS === "1") {
  console.log("[build-local-corpus] STAFFML_SKIP_LOCAL_CORPUS=1, skipping");
  process.exit(0);
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");

const which = spawnSync("which", ["vault"], { encoding: "utf8" });
if (which.status !== 0 || !which.stdout.trim()) {
  console.log(
    "[build-local-corpus] `vault` CLI not on PATH; skipping local corpus rebuild.\n" +
    "  To enable full-content rendering against your local YAMLs, run:\n" +
    "    pip install -e interviews/vault-cli\n" +
    "  then re-run `npm run dev`."
  );

  // Fallback: mirror SVG visuals even without the vault CLI, so visual
  // questions render correctly in the dev server.
  const mirrorScript = path.resolve(__dirname, "mirror-visuals.sh");
  console.log("[build-local-corpus] running SVG mirror fallback ...");
  const mirror = spawnSync("bash", [mirrorScript], {
    cwd: REPO_ROOT,
    stdio: "inherit",
  });
  if (mirror.status !== 0) {
    console.warn("[build-local-corpus] SVG mirror fallback failed (non-fatal).");
  }

  process.exit(0);
}

console.log("[build-local-corpus] running `vault build --local` ...");
const r = spawnSync("vault", ["build", "--local"], {
  cwd: REPO_ROOT,
  stdio: "inherit",
});
if (r.status !== 0) {
  console.error("[build-local-corpus] vault build failed; dev server will fall back to the worker.");
  // Soft-fail: don't block dev server startup just because the local corpus
  // isn't available. The worker fallback still gives a usable site.
  process.exit(0);
}
console.log("[build-local-corpus] done.");
