"use client";

/**
 * First-run explainer panel.
 *
 * One inline panel per mode (Vault / Practice / Mock Interview). Shown the
 * first time a user lands on a mode, dismissed on first interaction or by
 * the explicit "Got it" button. Persisted in localStorage by mode key.
 *
 * Why empty-state and not a tour:
 *   - Tour libraries (Shepherd, Intro.js) have terrible a11y stories.
 *   - Empty-state inline panels are fully keyboard-accessible by default.
 *   - The user can re-read by clearing localStorage; we never trap them.
 *
 * Add a new mode by adding an entry to MODE_CONTENT and rendering
 * <FirstRunExplainer mode="your-mode" /> at the top of the mode's page.
 */

import { useEffect, useState } from "react";
import { X, Target, Crosshair, Calculator, Cpu, Clock, Repeat } from "lucide-react";

export type ModeKey = "practice" | "gauntlet";

interface ModeContent {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  tagline: string;
  bullets: { icon: React.ComponentType<{ className?: string }>; text: string }[];
}

const MODE_CONTENT: Record<ModeKey, ModeContent> = {
  practice: {
    icon: Target,
    title: "Practice mode",
    tagline: "Untimed. Bring tools. Self-rate honestly. Build a learning loop.",
    bullets: [
      { icon: Calculator, text: "The Napkin Calc panel handles common back-of-envelope formulas (model memory, training time, KV cache, ridge point)." },
      { icon: Cpu, text: "Hardware Reference has GPU specs, latency hierarchy, and interconnects — open it any time." },
      { icon: Target, text: "After revealing the answer, rate yourself 1–4. Be honest — the rating is the whole signal." },
      { icon: Repeat, text: "Wrong answers come back tomorrow. Partial answers in 3 days. Nailed answers in 1–2 weeks. The 'due' counter in the nav tells you what's waiting." },
    ],
  },
  gauntlet: {
    icon: Crosshair,
    title: "Mock Interview",
    tagline: "Timed. The clock matters. Tools are available but you should know when to reach for them.",
    bullets: [
      { icon: Clock, text: "Pick a length (5 / 10 / 15 questions, or one deep design problem). The clock starts on Begin." },
      { icon: Calculator, text: "Hardware Ref and Napkin Calc are collapsed by default. Open them only when a real interview would let you peek." },
      { icon: Target, text: "After each question: type your answer, reveal, compare against the model, then self-rate Skip / Wrong / Partial / Nailed." },
    ],
  },
};

const STORAGE_PREFIX = "staffml_firstrun_";

function isDismissed(mode: ModeKey): boolean {
  if (typeof window === "undefined") return true;
  try {
    return localStorage.getItem(STORAGE_PREFIX + mode) === "1";
  } catch {
    return false;
  }
}

function dismiss(mode: ModeKey) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(STORAGE_PREFIX + mode, "1");
  } catch {
    /* localStorage may be disabled — fail silently */
  }
}

export default function FirstRunExplainer({ mode }: { mode: ModeKey }) {
  // Always start hidden on the server to avoid hydration mismatch.
  // Then check localStorage on the client and reveal if needed.
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (!isDismissed(mode)) setVisible(true);
  }, [mode]);

  if (!visible) return null;

  const content = MODE_CONTENT[mode];
  const Icon = content.icon;

  const close = () => {
    dismiss(mode);
    setVisible(false);
  };

  return (
    <div
      className="relative mx-4 mt-4 mb-2 lg:mx-6 rounded-xl border border-accentBlue/30 bg-accentBlue/5 p-5"
      role="region"
      aria-label={`Welcome to ${content.title}`}
    >
      <button
        onClick={close}
        aria-label="Dismiss welcome panel"
        className="absolute top-3 right-3 p-1.5 text-textTertiary hover:text-textPrimary hover:bg-surfaceHover rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-accentBlue/50"
      >
        <X className="w-4 h-4" />
      </button>

      <div className="flex items-start gap-4 mb-4">
        <div className="w-10 h-10 rounded-lg bg-accentBlue/15 flex items-center justify-center shrink-0">
          <Icon className="w-5 h-5 text-accentBlue" />
        </div>
        <div className="flex-1 min-w-0 pr-8">
          <h3 className="text-[15px] font-bold text-textPrimary">{content.title}</h3>
          <p className="text-[13px] text-textSecondary mt-0.5">{content.tagline}</p>
        </div>
      </div>

      <ul className="space-y-2 mb-4">
        {content.bullets.map((b, i) => {
          const BulletIcon = b.icon;
          return (
            <li key={i} className="flex items-start gap-2.5 text-[13px] text-textSecondary leading-relaxed">
              <BulletIcon className="w-3.5 h-3.5 mt-1 shrink-0 text-accentBlue/70" />
              <span>{b.text}</span>
            </li>
          );
        })}
      </ul>

      <button
        onClick={close}
        className="text-[12px] font-mono text-accentBlue hover:text-textPrimary transition-colors focus:outline-none focus:ring-2 focus:ring-accentBlue/50 rounded px-2 py-1 -ml-2"
      >
        Got it →
      </button>
    </div>
  );
}
