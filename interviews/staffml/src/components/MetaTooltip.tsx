"use client";

/**
 * MetaTooltip — small accessible hover/focus tooltip for metadata badges.
 *
 * Wraps any inline element with a styled tooltip that appears on hover OR
 * keyboard focus. Pure CSS (no positioning library), so it can overflow
 * narrow containers; uses max-width to bound that.
 *
 * Accessibility:
 *   - The trigger gets `tabIndex={0}` so keyboard users can focus it and
 *     the tooltip appears (via `group-focus-within`). No `role` is set —
 *     the trigger is informational, not a button. Adding `role="button"`
 *     without a handler misled screen readers into announcing an
 *     interactive control whose Enter/Space did nothing.
 *   - The tooltip body is hidden visually until hover/focus but always in
 *     the DOM, with `role="tooltip"` and an aria-describedby link
 *   - Also sets a native `title=` attribute as a fallback for screen
 *     readers that don't support live tooltip semantics
 *   - When the badge is already nested inside an interactive parent
 *     (e.g. a `<button>`), pass `withTooltip={false}` on the wrapping
 *     LevelBadge so MetaTooltip isn't rendered at all — avoids nested
 *     interactives and a double tab stop.
 *
 * Usage:
 *   <MetaTooltip title="L4 — Analyze (Senior)" body="Can you compare trade-offs and diagnose?">
 *     <span>L4</span>
 *   </MetaTooltip>
 */

import { useId, type ReactNode } from "react";
import clsx from "clsx";

export default function MetaTooltip({
  title,
  body,
  children,
  side = "top",
  className,
}: {
  title: string;
  body?: string;
  children: ReactNode;
  side?: "top" | "bottom";
  className?: string;
}) {
  const id = useId();
  const fallbackTitle = body ? `${title}\n${body}` : title;

  return (
    <span
      className={clsx("group relative inline-flex", className)}
      tabIndex={0}
      aria-describedby={id}
      title={fallbackTitle}
    >
      {children}
      <span
        id={id}
        role="tooltip"
        className={clsx(
          // hidden by default, shown on group hover/focus-within
          "pointer-events-none absolute left-1/2 -translate-x-1/2 z-50",
          "w-64 max-w-[80vw] p-2.5 rounded-md bg-background border border-border shadow-lg",
          "opacity-0 invisible group-hover:opacity-100 group-hover:visible group-focus-within:opacity-100 group-focus-within:visible",
          "transition-opacity duration-150",
          side === "top" ? "bottom-full mb-2" : "top-full mt-2",
        )}
      >
        <span className="block text-[11px] font-semibold text-textPrimary mb-1">{title}</span>
        {body && (
          <span className="block text-[10px] text-textSecondary leading-relaxed whitespace-pre-line">
            {body}
          </span>
        )}
      </span>
    </span>
  );
}
