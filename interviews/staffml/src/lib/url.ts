/**
 * URL safety helpers.
 *
 * The corpus is authored by humans + LLMs and ingested via JSON. Any URL
 * that originates from corpus content (e.g. `deep_dive_url`, `chapterUrl`)
 * is treated as untrusted user input. Without protocol validation, a single
 * malicious entry like `javascript:doEvil()` rendered into an `<a href>`
 * would create a 1-click XSS sink that runs on origin.
 *
 * `safeHref` returns the URL when it's a same-origin relative path or an
 * `http(s):` absolute URL, and `"#"` for everything else (null, empty,
 * `javascript:`, `data:`, `vbscript:`, `file:`, malformed, etc.).
 */

const ALLOWED_PROTOCOLS = new Set(["http:", "https:"]);

export function safeHref(url: string | null | undefined): string {
  if (!url) return "#";
  const trimmed = url.trim();
  if (!trimmed) return "#";

  // Same-origin relative paths and in-page anchors cannot escape the origin.
  if (
    trimmed.startsWith("/") ||
    trimmed.startsWith("#") ||
    trimmed.startsWith("?")
  ) {
    return trimmed;
  }

  try {
    const parsed = new URL(trimmed);
    if (!ALLOWED_PROTOCOLS.has(parsed.protocol)) return "#";
    return parsed.toString();
  } catch {
    return "#";
  }
}
