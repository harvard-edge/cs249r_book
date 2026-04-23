/**
 * Ecosystem base URL — the root of the MLSysBook multi-site deployment.
 *
 * Production: "https://mlsysbook.ai"
 * Dev:        "https://harvard-edge.github.io/cs249r_book_dev"
 *
 * Set via NEXT_PUBLIC_ECOSYSTEM_BASE in the build environment.
 */
export const ECOSYSTEM_BASE =
  process.env.NEXT_PUBLIC_ECOSYSTEM_BASE || "https://mlsysbook.ai";
