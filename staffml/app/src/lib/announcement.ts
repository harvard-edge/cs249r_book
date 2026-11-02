/**
 * Announcement-bar content for StaffML.
 *
 * Mirrors the 4-line unified template used across the nine Quarto sites
 * (see books/config/shared/html/announcement-vol2.yml and siblings,
 * merged in PR #1505). StaffML is Next.js + React 19 rather than Quarto,
 * so the content lives in TypeScript instead of YAML — but the rendered
 * DOM produced by AnnouncementBar.tsx is byte-identical to what Quarto
 * emits for the same config, and the CSS (in globals.css) recreates
 * Bootstrap 5's .alert.alert-primary so the bar reads visually the same.
 *
 * Template (the same three lines as the other companion sites, e.g.
 * kits/config/announcement.yml):
 *   Line 1 — identity + primary CTA (what is THIS site)
 *   Line 2 — the books (Volumes III and IV wrapped in .mls-unreleased)
 *   Line 3 — "Companion resources:" the full sibling row, newsletter last
 */

const BASE = "https://mlsysbook.ai";
const STAFFML_BASE_PATH = (process.env.NEXT_PUBLIC_BASE_PATH || "").replace(/\/$/, "");

function staffmlPath(path: string): string {
  return `${STAFFML_BASE_PATH}${path}`;
}

export interface AnnouncementConfig {
  /** Bootstrap Icons class suffix, e.g. "megaphone" → "bi-megaphone" */
  icon: string;
  /** Whether the × close button is shown and dismissal is persisted. */
  dismissable: boolean;
  /** Bootstrap alert variant suffix — "primary", "info", "warning", ... */
  type: "primary" | "info" | "warning" | "success" | "danger";
  /**
   * Content as HTML-string lines; the renderer joins them with <br> into
   * one <p> (exactly what Quarto's markdown-to-HTML pipeline produces).
   * Use <strong> for bold and <a href="..."> for links — keep it minimal.
   */
  lines: string[];
}

export const ANNOUNCEMENT: AnnouncementConfig = {
  icon: "megaphone",
  dismissable: true,
  type: "primary",
  lines: [
    `<strong>StaffML</strong>: ML systems interview practice, with worked problems grounded in the physics of the hardware. <a href="${staffmlPath("/practice/")}">Start practicing →</a>`,
    `<strong>The books:</strong> <a href="${BASE}/vol1/">Introduction to Machine Learning Systems</a> · <a href="${BASE}/vol2/">Scaling Machine Learning Systems</a> (preview)<span class="mls-unreleased"> · <a href="${BASE}/vol3/">Agentic Machine Learning Systems</a> (in preparation) · <a href="${BASE}/vol4/">Physical AI</a> (in preparation)</span>. All are open access.`,
    `<strong>Companion resources:</strong> <a href="${BASE}/tinytorch/">TinyTorch</a> (build) · <a href="${BASE}/kits/">Hardware Kits</a> (deploy) · <a href="${BASE}/mlsysim/">MLSys·im</a> (model) · <a href="${BASE}/labs/">Labs</a> (explore) · <a href="${BASE}/slides/">Lecture Slides</a> (teach) · <a href="${BASE}/instructors/">Instructor Hub</a> (teach) · <a href="${BASE}/staffml/">StaffML</a> (practice) · <a href="${BASE}/newsletter/">Newsletter</a>`,
  ],
};
