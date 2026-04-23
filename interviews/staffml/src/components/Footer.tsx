import { ECOSYSTEM_BASE } from "../lib/env";

/**
 * Page-end footer with a stable cross-link back to the ML Systems textbook.
 *
 * Per-question book links are deferred until mlsysbook.ai URLs stabilize. In
 * the meantime the site-level cross-link to the book homepage — which cannot
 * 404 — gives every StaffML page a closing funnel back to the textbook. The
 * top-of-page EcosystemBar covers discovery; this footer covers intent (the
 * user finished reading and wants to learn more).
 *
 * Intentionally minimal: one attribution row, low contrast, no dropdowns.
 */
export default function Footer() {
  return (
    <footer className="border-t border-border mt-12 py-6 px-4 lg:px-6 text-xs text-textTertiary">
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row gap-2 sm:gap-4 sm:items-center sm:justify-between">
        <p className="leading-relaxed">
          StaffML is part of the{" "}
          <a
            href={ECOSYSTEM_BASE}
            target="_blank"
            rel="noopener noreferrer"
            className="text-textSecondary hover:text-accentBlue font-medium underline-offset-2 hover:underline"
          >
            Machine Learning Systems
          </a>{" "}
          textbook ecosystem. Questions are grounded in the book&rsquo;s
          principles of quantitative ML systems reasoning.
        </p>
        <div className="flex flex-wrap gap-x-4 gap-y-1 shrink-0">
          <a
            href={ECOSYSTEM_BASE}
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-textSecondary transition-colors"
          >
            Read the book
          </a>
          <a
            href={`${ECOSYSTEM_BASE}/vol1/`}
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-textSecondary transition-colors"
          >
            Volume I
          </a>
          <a
            href={`${ECOSYSTEM_BASE}/vol2/`}
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-textSecondary transition-colors"
          >
            Volume II
          </a>
          <a
            href="https://github.com/harvard-edge/cs249r_book"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-textSecondary transition-colors"
          >
            GitHub
          </a>
        </div>
      </div>
    </footer>
  );
}
