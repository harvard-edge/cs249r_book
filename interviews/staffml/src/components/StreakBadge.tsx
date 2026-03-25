"use client";

import { useState, useEffect } from "react";
import { Flame } from "lucide-react";
import clsx from "clsx";
import { getStreakData, getStreakMilestone } from "@/lib/progress";

export default function StreakBadge() {
  const [streak, setStreak] = useState(0);
  const [milestone, setMilestone] = useState<string | null>(null);

  useEffect(() => {
    const data = getStreakData();
    setStreak(data.currentStreak);
    setMilestone(getStreakMilestone(data.currentStreak));
  }, []);

  if (streak === 0) return null;

  return (
    <div
      className={clsx(
        "flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-mono transition-all",
        streak >= 7
          ? "bg-accentAmber/10 text-accentAmber border border-accentAmber/20"
          : "text-textTertiary"
      )}
      title={milestone ? `${milestone} — ${streak} day streak` : `${streak} day streak`}
    >
      <Flame className={clsx(
        "w-3.5 h-3.5",
        streak >= 30 ? "text-accentRed" : streak >= 7 ? "text-accentAmber" : "text-textTertiary"
      )} />
      <span>{streak}</span>
    </div>
  );
}
