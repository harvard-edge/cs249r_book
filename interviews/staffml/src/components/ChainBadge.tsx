"use client";

/**
 * Pre-reveal chain indicator — the single Phase-5 intervention.
 *
 * ARCHITECTURE.md §8 (v2.1): chain discoverability ships as ONE intervention +
 * instrumentation. Further UX (sidebar filter, /chains browse, tooltip,
 * dashboard) only if measured engagement clears the gates (>15% CTR,
 * 1.5× within-chain completion).
 *
 * This component displays "Part N of M — <chain name>" above the question
 * title, visible BEFORE reveal. Click navigates to the chain strip / siblings.
 */

import { Link } from "lucide-react";
import clsx from "clsx";

import { track } from "@/lib/analytics";

export interface ChainBadgeProps {
  chainId: string;
  chainName?: string;
  position: number;    // 1-indexed
  total: number;
  onClick?: () => void;
  className?: string;
}

export default function ChainBadge({
  chainId,
  chainName,
  position,
  total,
  onClick,
  className,
}: ChainBadgeProps) {
  // Fire shown-event on mount (debounced per-session by analytics layer).
  if (typeof window !== "undefined") {
    queueMicrotask(() => {
      track({
        type: "chain_badge_shown",
        chainId,
        position,
        total,
      } as any);
    });
  }

  const label = `Part ${position} of ${total}${chainName ? ` — ${chainName}` : ""}`;

  return (
    <button
      type="button"
      data-testid="chain-badge"
      aria-label={`Chained question: ${label}. Click to view chain siblings.`}
      onClick={() => {
        track({
          type: "chain_badge_clicked",
          chainId,
          position,
          total,
        } as any);
        onClick?.();
      }}
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full border border-border",
        "bg-surfaceSubtle px-3 py-1",
        "text-[11px] font-mono uppercase tracking-wide",
        "text-textSecondary hover:text-textPrimary",
        "hover:bg-surface transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent",
        className,
      )}
    >
      <Link className="w-3 h-3" aria-hidden="true" />
      <span>{label}</span>
    </button>
  );
}
