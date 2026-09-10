"use client";

import { useEffect, useMemo, useState } from "react";
import { ANNOUNCEMENT } from "@/lib/announcement";
import { IS_LIVE_DEPLOY } from "@/lib/env";

/**
 * StaffML announcement bar — DOM-identical to Quarto's #quarto-announcement
 * output so the nine Quarto sites and this Next.js site advertise the
 * ecosystem with the same visual voice.
 *
 * Quarto emits (from the YAML configs merged in PR #1505):
 *
 *   <div id="quarto-announcement" data-announcement-id="{hash}"
 *        class="alert alert-primary">
 *     <i class="bi bi-megaphone quarto-announcement-icon"></i>
 *     <div class="quarto-announcement-content">
 *       <p>... markdown, lines joined with <br> ...</p>
 *     </div>
 *     <i class="bi bi-x-lg quarto-announcement-action"></i>
 *   </div>
 *
 * Styles live in src/app/globals.css under the same id/class selectors.
 * Bootstrap Icons (bi-megaphone, bi-x-lg) are already loaded by
 * EcosystemBar.tsx via the jsdelivr CDN, so the glyphs render here too.
 *
 * Dismissable: click × → hide → persist in sessionStorage, keyed by
 * content hash. Same scheme Quarto uses, so a copy change auto-invalidates
 * prior dismissals.
 */

// djb2 hash: stable, cheap, ASCII-safe. No crypto guarantees needed —
// this key only has to change when the banner copy changes.
function hashContent(content: string): string {
  let hash = 5381;
  for (let i = 0; i < content.length; i++) {
    hash = ((hash << 5) + hash + content.charCodeAt(i)) | 0;
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

const SS_KEY_PREFIX = "quarto-announcement-dismissed-";

export default function AnnouncementBar() {
  const { icon, dismissable: configDismissable, type, lines } = ANNOUNCEMENT;

  // Dismiss is gated on the live-deploy flag: on the dev-preview deploy
  // (harvard-edge.github.io/cs249r_book_dev) the bar is intentionally
  // persistent so every pageview sees the ecosystem pitch; the live site
  // respects the author's `dismissable: true` config and offers the ×.
  const dismissable = configDismissable && IS_LIVE_DEPLOY;

  // Quarto joins multi-line content with <br> inside a single <p>.
  const innerHTML = useMemo(
    () => `<p>${lines.join("<br>\n")}</p>`,
    [lines],
  );
  const announcementId = useMemo(() => hashContent(innerHTML), [innerHTML]);

  // Start hidden until we've checked sessionStorage on the client — avoids
  // a flash of the bar re-appearing after dismissal on navigation.
  const [dismissed, setDismissed] = useState<boolean>(false);
  const [hydrated, setHydrated] = useState<boolean>(false);

  useEffect(() => {
    if (!dismissable) {
      setHydrated(true);
      return;
    }
    try {
      const saved = sessionStorage.getItem(SS_KEY_PREFIX + announcementId);
      if (saved === "1") setDismissed(true);
    } catch {
      // sessionStorage unavailable (strict cookie mode) — show the bar.
    }
    setHydrated(true);
  }, [announcementId, dismissable]);

  function handleDismiss() {
    setDismissed(true);
    try {
      sessionStorage.setItem(SS_KEY_PREFIX + announcementId, "1");
    } catch {
      // ignore — UI dismissal still works for this tab.
    }
  }

  const rootClass = [
    "alert",
    `alert-${type}`,
    !hydrated || dismissed ? "hidden" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div
      id="quarto-announcement"
      data-announcement-id={announcementId}
      className={rootClass}
      role="alert"
    >
      <i
        className={`bi bi-${icon} quarto-announcement-icon`}
        aria-hidden="true"
      />
      <div
        className="quarto-announcement-content"
        // Content is author-controlled at build time (lib/announcement.ts)
        // — not user input — so dangerouslySetInnerHTML is safe here.
        dangerouslySetInnerHTML={{ __html: innerHTML }}
      />
      {dismissable ? (
        <i
          className="bi bi-x-lg quarto-announcement-action"
          role="button"
          aria-label="Dismiss announcement"
          tabIndex={0}
          onClick={handleDismiss}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              handleDismiss();
            }
          }}
        />
      ) : null}
    </div>
  );
}
