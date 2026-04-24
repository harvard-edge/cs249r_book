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

/**
 * True iff this build targets the live (production) deploy at mlsysbook.ai.
 * Derived from ECOSYSTEM_BASE — dev deploys point at harvard-edge.github.io,
 * so anything else is treated as live.
 *
 * Used by AnnouncementBar.tsx to keep the dismiss button off the dev-preview
 * build: on dev the announcement bar is intentionally persistent so each
 * pageview sees the ecosystem pitch; on live it becomes dismissable so
 * returning visitors aren't nagged.
 */
export const IS_LIVE_DEPLOY = !/cs249r_book_dev/.test(ECOSYSTEM_BASE);
