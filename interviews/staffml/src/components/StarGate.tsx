"use client";

import { useEffect, useState } from "react";
import { Github, Star, CheckCircle2, ExternalLink, X } from "lucide-react";
import { getStarUrl, markVerified, fetchStarCount } from "@/lib/star-gate";

export default function StarGate({ onVerified }: { onVerified: () => void }) {
  const [starCount, setStarCount] = useState<number | null>(null);
  const [retired, setRetired] = useState<"starred" | "honor" | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchStarCount().then((n) => {
      if (!cancelled) setStarCount(n);
    });
    return () => { cancelled = true; };
  }, []);

  const handleStar = () => {
    window.open(getStarUrl(), "_blank", "noopener,noreferrer");
    markVerified("starred");
    setRetired("starred");
    setTimeout(onVerified, 1500);
  };

  const handleAlreadyStarred = () => {
    markVerified("honor");
    setRetired("honor");
    setTimeout(onVerified, 1500);
  };

  const handleDismiss = () => {
    markVerified("dismissed");
    onVerified();
  };

  if (retired) {
    return (
      <div className="fixed inset-0 z-50 bg-background/90 backdrop-blur-sm flex items-center justify-center p-6">
        <div className="max-w-md w-full bg-background border border-border rounded-2xl p-8 shadow-lg text-center">
          <div className="w-16 h-16 rounded-full bg-accentGreen/10 border border-accentGreen/30 flex items-center justify-center mx-auto mb-4">
            <CheckCircle2 className="w-8 h-8 text-accentGreen" />
          </div>
          <h2 className="text-xl font-bold text-textPrimary mb-2">Thank you.</h2>
          <p className="text-sm text-textSecondary">
            {retired === "starred"
              ? "That signal matters more than you know."
              : "Appreciated. Back to your questions."}
          </p>
        </div>
      </div>
    );
  }

  const formattedCount = starCount !== null ? starCount.toLocaleString() : null;

  return (
    <div className="fixed inset-0 z-50 bg-background/90 backdrop-blur-sm flex items-center justify-center p-6">
      <div className="relative max-w-md w-full bg-background border border-border rounded-2xl p-8 shadow-lg">
        {/* Close (X) — counts as dismiss */}
        <button
          onClick={handleDismiss}
          aria-label="Close"
          className="absolute top-3 right-3 p-1.5 rounded-md text-textTertiary hover:text-textPrimary hover:bg-surface transition-colors"
        >
          <X className="w-4 h-4" />
        </button>

        {/* Header */}
        <div className="flex items-center justify-center gap-2 mb-2">
          <Star className="w-5 h-5 text-accentAmber" />
          <h2 className="text-xl font-bold text-textPrimary">Our only ask.</h2>
        </div>
        <p className="text-sm text-textSecondary text-center mb-5 leading-relaxed">
          StaffML, the textbook, TinyTorch, the hardware kits — all free, forever.
        </p>

        {/* Live star count + the why */}
        <div className="bg-surface border border-border rounded-lg p-4 mb-6 text-center">
          {formattedCount !== null ? (
            <>
              <div className="text-2xl font-bold text-textPrimary leading-tight">
                {formattedCount} <span className="text-accentAmber">★</span>
              </div>
              <div className="text-[11px] font-mono uppercase tracking-wide text-textTertiary mt-0.5">
                stargazers so far
              </div>
            </>
          ) : (
            <div className="text-sm text-textTertiary">Joined by thousands of stargazers.</div>
          )}
          <p className="text-[12px] text-textSecondary mt-3 leading-relaxed">
            Every star tells universities, publishers, and funders that AI engineering
            education matters. Goal: 100,000 ★ — one million engineers by 2030.
          </p>
        </div>

        {/* Primary CTA */}
        <button
          onClick={handleStar}
          className="flex items-center justify-center gap-2 w-full py-3 bg-accentBlue text-white font-bold rounded-lg text-sm hover:opacity-90 transition-opacity mb-2"
        >
          <Github className="w-4 h-4" />
          Star on GitHub
          <ExternalLink className="w-3 h-3 opacity-70" />
        </button>

        {/* Secondary CTA — honor system */}
        <button
          onClick={handleAlreadyStarred}
          className="w-full py-2.5 bg-surface border border-border text-textPrimary font-medium rounded-lg text-sm hover:bg-surfaceHover transition-colors"
        >
          I already starred
        </button>

        {/* Footer note */}
        <p className="text-[11px] text-textMuted text-center mt-5">
          You won&apos;t see this again.
        </p>
      </div>
    </div>
  );
}
