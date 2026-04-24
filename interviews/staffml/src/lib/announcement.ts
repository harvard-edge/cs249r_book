/**
 * Announcement-bar content for StaffML.
 *
 * Mirrors the 4-line unified template used across the nine Quarto sites
 * (see book/quarto/config/shared/html/announcement-vol2.yml and siblings,
 * merged in PR #1505). StaffML is Next.js + React 19 rather than Quarto,
 * so the content lives in TypeScript instead of YAML — but the rendered
 * DOM produced by AnnouncementBar.tsx is byte-identical to what Quarto
 * emits for the same config, and the CSS (in globals.css) recreates
 * Bootstrap 5's .alert.alert-primary so the bar reads visually the same.
 *
 * Template:
 *   Line 1 — identity + primary CTA (what is THIS site)
 *   Line 2 — the book
 *   Line 3 — "Alongside the book:" sibling row (3 most-relevant verbs)
 *   Line 4 — newsletter
 */

const BASE = "https://mlsysbook.ai";

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
    `🎯 <strong>StaffML</strong> — physics-grounded ML systems interview prep; practice with worked-problem rigor. <a href="/practice">Start practicing →</a>`,
    `📘 <strong>The book:</strong> <a href="${BASE}/vol1/">Vol I: Foundations</a> · <a href="${BASE}/vol2/">Vol II: At Scale</a> — open access, free forever.`,
    `🛠️ <strong>Alongside the book:</strong> <a href="${BASE}/tinytorch/">TinyTorch</a> (build) · <a href="${BASE}/kits/">Hardware Kits</a> (deploy) · <a href="${BASE}/mlsysim/">MLSys·im</a> (simulate)`,
    `📬 <strong>Newsletter:</strong> ML Systems insights & updates — <a href="${BASE}/newsletter/">Subscribe →</a>`,
  ],
};
