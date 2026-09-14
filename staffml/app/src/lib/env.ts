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

/**
 * GitHub repository ("owner/name") behind the source, issue, and star links.
 *
 * next.config.mjs forwards GITHUB_REPOSITORY, which GitHub Actions sets on
 * every build, so renaming the repository needs no code change. The fallback
 * covers local `next dev`.
 */
export const GITHUB_REPOSITORY =
  process.env.NEXT_PUBLIC_GITHUB_REPOSITORY || "harvard-edge/cs249r_book";
export const GITHUB_REPO_URL = `https://github.com/${GITHUB_REPOSITORY}`;
