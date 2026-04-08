import Link from "next/link";
import { Library, Target, Crosshair, ArrowLeft } from "lucide-react";
import { QUESTION_COUNT_DISPLAY } from "@/lib/corpus";

export default function NotFound() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-6 py-20 text-center">
      {/* Vault door illustration */}
      <div className="mb-8">
        <svg viewBox="0 0 200 200" className="w-40 h-40 mx-auto" fill="none" xmlns="http://www.w3.org/2000/svg">
          {/* Vault body */}
          <rect x="30" y="30" width="140" height="140" rx="12" fill="var(--surface)" stroke="var(--border)" strokeWidth="2"/>
          {/* Door circle */}
          <circle cx="100" cy="100" r="50" fill="var(--surface-hover)" stroke="var(--border-highlight)" strokeWidth="2"/>
          {/* Dial ticks */}
          {[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330].map((angle) => (
            <line
              key={angle}
              x1={100 + 42 * Math.cos((angle * Math.PI) / 180)}
              y1={100 + 42 * Math.sin((angle * Math.PI) / 180)}
              x2={100 + 47 * Math.cos((angle * Math.PI) / 180)}
              y2={100 + 47 * Math.sin((angle * Math.PI) / 180)}
              stroke="var(--text-muted)"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          ))}
          {/* Inner circle */}
          <circle cx="100" cy="100" r="20" fill="var(--background)" stroke="var(--border)" strokeWidth="1.5"/>
          {/* Dial pointer — pointing at "404" position */}
          <line x1="100" y1="100" x2="100" y2="60" stroke="var(--accent-red)" strokeWidth="2.5" strokeLinecap="round"/>
          <circle cx="100" cy="100" r="4" fill="var(--accent-red)"/>
          {/* Handle */}
          <rect x="155" y="90" width="20" height="20" rx="4" fill="var(--surface-elevated)" stroke="var(--border-highlight)" strokeWidth="1.5"/>
          {/* Bolt dots */}
          <circle cx="45" cy="45" r="3" fill="var(--border-highlight)"/>
          <circle cx="155" cy="45" r="3" fill="var(--border-highlight)"/>
          <circle cx="45" cy="155" r="3" fill="var(--border-highlight)"/>
          <circle cx="155" cy="155" r="3" fill="var(--border-highlight)"/>
          {/* 404 text on dial */}
          <text x="100" y="105" textAnchor="middle" fontSize="14" fontWeight="700" fontFamily="monospace" fill="var(--text-tertiary)">404</text>
        </svg>
      </div>

      <h1 className="text-2xl font-extrabold text-textPrimary mb-2">Wrong Combination</h1>
      <p className="text-[15px] text-textSecondary mb-8 max-w-md">
        This page doesn&apos;t exist — but the vault has {QUESTION_COUNT_DISPLAY} ML systems questions waiting for you.
      </p>

      <div className="flex flex-wrap items-center justify-center gap-3">
        <Link
          href="/"
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-accentBlue text-white font-bold rounded-lg text-sm hover:opacity-90 transition-opacity"
        >
          <Library className="w-4 h-4" /> Open the Vault
        </Link>
        <Link
          href="/practice"
          className="inline-flex items-center gap-2 px-4 py-2.5 bg-surface border border-border text-textSecondary font-medium rounded-lg text-sm hover:text-textPrimary transition-colors"
        >
          <Target className="w-4 h-4" /> Practice
        </Link>
        <Link
          href="/gauntlet"
          className="inline-flex items-center gap-2 px-4 py-2.5 bg-surface border border-border text-textSecondary font-medium rounded-lg text-sm hover:text-textPrimary transition-colors"
        >
          <Crosshair className="w-4 h-4" /> Mock Interview
        </Link>
      </div>

      <Link href="/" className="inline-flex items-center gap-1.5 text-xs text-textTertiary hover:text-textSecondary mt-8 transition-colors">
        <ArrowLeft className="w-3 h-3" /> Back to home
      </Link>
    </div>
  );
}
