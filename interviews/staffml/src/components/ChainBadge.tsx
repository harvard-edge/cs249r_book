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
 * title, visible BEFORE reveal. Click invokes the consumer-supplied
 * onClick (today: opens an inline ChainStrip in the question panel).
 */

import { useEffect } from "react";
import { Link as LinkIcon, ChevronRight } from "lucide-react";
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
  // Fire shown-event ONCE per chain mount, not on every render. The
  // analytics layer dedups per (chainId, sessionId) for CTR computation,
  // but firing on every render still bloats localStorage and pending
  // queue payloads. Deps include position/total so a true sibling
  // navigation (new chain context) re-records.
  useEffect(() => {
    track({ type: "chain_badge_shown", chainId, position, total });
  }, [chainId, position, total]);

  const label = `Part ${position} of ${total}${chainName ? ` — ${chainName}` : ""}`;

  return (
    <button
      type="button"
      data-testid="chain-badge"
      aria-label={`Chained question: ${label}. Click to view related questions in this chain.`}
      onClick={() => {
        track({ type: "chain_badge_clicked", chainId, position, total });
        onClick?.();
      }}
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full",
        "border border-accentBlue/30 bg-accentBlue/5",
        "px-3 py-1.5",
        "text-xs font-mono uppercase tracking-wide",
        "text-accentBlue",
        "hover:bg-accentBlue/10 hover:border-accentBlue/50",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accentBlue/40",
        "transition-colors group",
        className,
      )}
    >
      <LinkIcon className="w-3 h-3" aria-hidden="true" />
      <span>{label}</span>
      <ChevronRight
        className="w-3 h-3 transition-transform group-hover:translate-x-0.5"
        aria-hidden="true"
      />
    </button>
  );
}
