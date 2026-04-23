"use client";

/**
 * /welcome — First-run landing moment.
 *
 * Shown once to brand-new visitors (0 attempts logged AND no
 * staffml_firstrun_welcome flag in localStorage) before they land on
 * the Vault grid. The single job of this page is to make the scale
 * of the project legible in the first five seconds, then get out of
 * the way.
 *
 * Design brief:
 *   - One screen, no scrolling on desktop 1440×900.
 *   - Hero uses real numbers from src/lib/stats.ts so the hook ages
 *     with the corpus.
 *   - Three equal action cards: Try Random / Practice / Mock Interview.
 *     "Try Random" is the lowest-activation click and is placed first
 *     so a decision-fatigued visitor can just click once and see a
 *     real question.
 *   - Skip link at the bottom opens the Vault directly.
 *   - Flag is set on ANY exit path (card click or Skip), not on mount,
 *     so a visitor who closes the tab without interacting will see the
 *     page again next time. The only way to re-trigger it after dismiss
 *     is to clear localStorage.
 */

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect } from "react";
import { Target, Crosshair, Shuffle, ArrowRight } from "lucide-react";
import {
  QUESTION_COUNT_FORMATTED,
  TOPIC_COUNT,
  TRACK_COUNT,
  LEVEL_COUNT,
} from "@/lib/stats";

const FLAG_KEY = "staffml_firstrun_welcome";

function markSeen() {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(FLAG_KEY, "1");
  } catch {
    /* localStorage may be disabled — fail silently */
  }
}

export default function WelcomePage() {
  const router = useRouter();

  // Mark this tab session as "already bounced" so clicking "Vault" in
  // the nav from /welcome doesn't ping-pong back. The localStorage flag
  // (cross-session dismissal) still only sets on explicit action.
  useEffect(() => {
    try { sessionStorage.setItem("staffml_firstrun_bounced", "1"); } catch { /* noop */ }
  }, []);

  const handleAction = useCallback(
    (href: string) => {
      markSeen();
      router.push(href);
    },
    [router],
  );

  return (
    <div className="flex-1 flex flex-col items-center justify-center px-6 py-12">
      <div className="max-w-3xl w-full">
        {/* ─── Mark ─── */}
        <div className="flex items-center justify-center gap-2 mb-8">
          <svg viewBox="0 0 32 32" className="w-8 h-8 drop-shadow-[0_0_10px_rgba(59,130,246,0.4)]">
            <path d="M5,25 L16,9 L27,9" stroke="#3b82f6" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" fill="none" />
            <circle cx="16" cy="9" r="2.5" fill="#3b82f6" />
            <circle cx="16" cy="9" r="1" fill="currentColor" />
          </svg>
          <span className="text-lg tracking-tight">
            <span className="text-textPrimary font-extrabold">Staff</span>
            <span className="text-accentBlue font-bold ml-[2px]">ML</span>
          </span>
        </div>

        {/* ─── Hero ─── */}
        <h1 className="text-[28px] sm:text-4xl font-extrabold text-textPrimary tracking-tight text-center leading-tight mb-4">
          <span className="text-accentBlue">{QUESTION_COUNT_FORMATTED}</span> physics-grounded{" "}
          <br className="hidden sm:block" />
          ML systems interview questions.
        </h1>
        <p className="text-[15px] sm:text-base text-textSecondary text-center max-w-xl mx-auto leading-relaxed mb-2">
          {TOPIC_COUNT} topics across {TRACK_COUNT} deployment tracks, at {LEVEL_COUNT} difficulty levels.
          Backed by a 600-page open textbook. Runs entirely in your browser.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-2 mb-10">
          <span className="text-[11px] px-2.5 py-1 rounded-full border border-accentGreen/30 bg-accentGreen/5 text-accentGreen font-medium">No accounts</span>
          <span className="text-[11px] px-2.5 py-1 rounded-full border border-accentGreen/30 bg-accentGreen/5 text-accentGreen font-medium">No tracking</span>
          <span className="text-[11px] px-2.5 py-1 rounded-full border border-accentGreen/30 bg-accentGreen/5 text-accentGreen font-medium">100% free</span>
          <span className="text-[11px] px-2.5 py-1 rounded-full border border-accentGreen/30 bg-accentGreen/5 text-accentGreen font-medium">Open source</span>
        </div>

        {/* ─── Three action cards ─── */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-8">
          <ActionCard
            icon={Shuffle}
            accent="blue"
            title="Try one right now"
            body={`Random question from the full corpus. No setup, no filter, no commitment — see what a real StaffML question looks like in one click.`}
            cta="Shuffle"
            onClick={() => handleAction("/practice?random=1")}
            primary
          />
          <ActionCard
            icon={Target}
            accent="amber"
            title="Practice"
            body="Untimed drills with spaced repetition. Rate yourself after each answer; the app schedules what you should see tomorrow."
            cta="Start practicing"
            onClick={() => handleAction("/practice")}
          />
          <ActionCard
            icon={Crosshair}
            accent="red"
            title="Mock Interview"
            body="Timed gauntlet: 5–15 questions or one full system-design problem. The clock matters, the feedback is the signal."
            cta="Start interview"
            onClick={() => handleAction("/gauntlet")}
          />
        </div>

        {/* ─── Skip ─── */}
        <div className="flex items-center justify-center">
          <button
            onClick={() => handleAction("/")}
            className="inline-flex items-center gap-1.5 text-[12px] text-textTertiary hover:text-textSecondary transition-colors"
          >
            Skip — open the Vault <ArrowRight className="w-3 h-3" />
          </button>
        </div>

        {/* ─── About link (small, bottom) ─── */}
        <div className="flex items-center justify-center mt-8 pt-8 border-t border-borderSubtle">
          <Link
            href="/about"
            onClick={markSeen}
            className="text-[12px] text-textTertiary hover:text-accentBlue transition-colors"
          >
            How was this built? →
          </Link>
        </div>
      </div>
    </div>
  );
}

// ─── ActionCard ───────────────────────────────────────────────
function ActionCard({
  icon: Icon,
  title,
  body,
  cta,
  onClick,
  accent,
  primary = false,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  body: string;
  cta: string;
  onClick: () => void;
  accent: "blue" | "amber" | "red";
  primary?: boolean;
}) {
  const accentBorder = {
    blue: "border-accentBlue/30 hover:border-accentBlue/60",
    amber: "border-accentAmber/30 hover:border-accentAmber/60",
    red: "border-accentRed/30 hover:border-accentRed/60",
  }[accent];
  const accentIcon = {
    blue: "text-accentBlue",
    amber: "text-accentAmber",
    red: "text-accentRed",
  }[accent];
  const accentCta = {
    blue: "text-accentBlue",
    amber: "text-accentAmber",
    red: "text-accentRed",
  }[accent];

  return (
    <button
      onClick={onClick}
      className={`group p-5 rounded-xl border text-left bg-surface/50 hover:bg-surface transition-all focus:outline-none focus:ring-2 focus:ring-accentBlue/40 ${accentBorder} ${primary ? "ring-1 ring-accentBlue/20" : ""}`}
    >
      <Icon className={`w-6 h-6 mb-3 ${accentIcon}`} />
      <h3 className="text-[14px] font-bold text-textPrimary mb-1.5">{title}</h3>
      <p className="text-[12px] text-textSecondary leading-relaxed mb-3">{body}</p>
      <span className={`inline-flex items-center gap-1 text-[12px] font-bold ${accentCta}`}>
        {cta} <ArrowRight className="w-3 h-3 transition-transform group-hover:translate-x-0.5" />
      </span>
    </button>
  );
}
