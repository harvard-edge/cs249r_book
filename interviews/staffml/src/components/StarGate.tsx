"use client";

import { useState } from "react";
import { Github, Star, CheckCircle2, ExternalLink } from "lucide-react";
import { verifyGitHubStar, getStarUrl, getFreeLimit, bypassVerification } from "@/lib/star-gate";

export default function StarGate({ onVerified }: { onVerified: () => void }) {
  const [username, setUsername] = useState("");
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [starred, setStarred] = useState(false);

  const handleVerify = async () => {
    if (!username.trim()) {
      setError("Enter your GitHub username");
      return;
    }
    setChecking(true);
    setError(null);

    const ok = await verifyGitHubStar(username.trim());
    setChecking(false);

    if (ok) {
      setStarred(true);
      setTimeout(onVerified, 1500); // brief success state
    } else {
      setError("We couldn't find your star yet. Make sure you've starred the repo, then try again.");
    }
  };

  if (starred) {
    return (
      <div className="fixed inset-0 z-50 bg-background/90 backdrop-blur-sm flex items-center justify-center p-6">
        <div className="max-w-md w-full text-center">
          <div className="w-16 h-16 rounded-full bg-accentGreen/10 border border-accentGreen/30 flex items-center justify-center mx-auto mb-4">
            <CheckCircle2 className="w-8 h-8 text-accentGreen" />
          </div>
          <h2 className="text-xl font-bold text-textPrimary mb-2">Thank you!</h2>
          <p className="text-sm text-textSecondary">Unlimited access unlocked. Happy practicing.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 bg-background/90 backdrop-blur-sm flex items-center justify-center p-6">
      <div className="max-w-md w-full bg-background border border-border rounded-2xl p-8 shadow-lg">
        {/* Header */}
        <div className="flex items-center justify-center gap-2 mb-2">
          <Star className="w-5 h-5 text-accentAmber" />
          <h2 className="text-xl font-bold text-textPrimary">Support StaffML</h2>
        </div>
        <p className="text-sm text-textSecondary text-center mb-6">
          You&apos;ve used your {getFreeLimit()} free questions for today.
          Star us on GitHub to unlock unlimited access — it&apos;s free and takes 2 seconds.
        </p>

        {/* Step 1: Star the repo */}
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-3">
            <span className="w-5 h-5 rounded-full bg-accentBlue/10 text-accentBlue text-[11px] font-bold flex items-center justify-center">1</span>
            <span className="text-sm font-medium text-textPrimary">Star the repository</span>
          </div>
          <a
            href={getStarUrl()}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-center gap-2 w-full py-3 bg-surface border border-border rounded-lg text-sm font-medium text-textPrimary hover:bg-surfaceHover transition-colors"
          >
            <Github className="w-4 h-4" />
            Open on GitHub
            <ExternalLink className="w-3 h-3 text-textTertiary" />
          </a>
        </div>

        {/* Step 2: Verify */}
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-3">
            <span className="w-5 h-5 rounded-full bg-accentBlue/10 text-accentBlue text-[11px] font-bold flex items-center justify-center">2</span>
            <span className="text-sm font-medium text-textPrimary">Enter your GitHub username</span>
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleVerify()}
              placeholder="your-username"
              className="flex-1 px-3 py-2.5 bg-surface border border-border rounded-lg text-sm text-textPrimary placeholder:text-textMuted focus:outline-none focus:border-accentBlue/50 font-mono"
            />
            <button
              onClick={handleVerify}
              disabled={checking}
              className="px-4 py-2.5 bg-accentBlue text-white font-bold rounded-lg text-sm hover:opacity-90 transition-opacity disabled:opacity-50"
            >
              {checking ? "Checking..." : "Verify"}
            </button>
          </div>
          {error && (
            <p className="text-xs text-accentRed mt-2">{error}</p>
          )}
        </div>

        {/* Footer */}
        <p className="text-[11px] text-textMuted text-center mb-3">
          StaffML is free and open source. Your star helps us grow and keep building.
        </p>
        <button
          onClick={() => { bypassVerification(); onVerified(); }}
          className="text-[11px] text-textMuted hover:text-textTertiary transition-colors underline block mx-auto"
        >
          Continue without starring
        </button>
      </div>
    </div>
  );
}
