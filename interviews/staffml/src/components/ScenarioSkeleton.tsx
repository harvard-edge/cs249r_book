"use client";

/**
 * Shown while a question's scenario text hydrates from the vault worker.
 * The bundled summary has scenario: "" until useFullQuestion swaps in the
 * full record (~100-300ms warm, <5s cold). Without a skeleton, the user
 * sees a blank region followed by a jarring pop-in of paragraph text.
 *
 * Three pulsing lines approximate the average scenario length (~3 lines)
 * so the layout doesn't jump when real content arrives. Uses the same
 * `animate-pulse` Tailwind utility the Napkin panel already uses for
 * consistency.
 */
export function ScenarioSkeleton() {
  return (
    <div
      aria-label="Loading question"
      aria-live="polite"
      className="space-y-2 animate-pulse"
    >
      <div className="h-4 w-11/12 rounded bg-borderSubtle" />
      <div className="h-4 w-10/12 rounded bg-borderSubtle" />
      <div className="h-4 w-7/12 rounded bg-borderSubtle" />
    </div>
  );
}
