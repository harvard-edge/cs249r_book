"use client";

import { Calendar, TrendingUp, TrendingDown } from "lucide-react";
import clsx from "clsx";
import type { PrepStats } from "@/lib/plans";

interface PrepDashboardProps {
  stats: PrepStats;
}

export default function PrepDashboard({ stats }: PrepDashboardProps) {
  const pct = stats.totalQuestions > 0
    ? Math.round((stats.completed / stats.totalQuestions) * 100)
    : 0;

  return (
    <div className="flex items-center gap-4 px-4 py-2 bg-surface/50 border-b border-border text-xs">
      <div className="flex items-center gap-1.5">
        <Calendar className="w-3 h-3 text-textTertiary" />
        <span className="font-mono text-textSecondary">
          {stats.daysRemaining > 0 ? `${stats.daysRemaining}d left` : "Due today"}
        </span>
      </div>
      <div className="h-3 w-px bg-border" />
      <span className="font-mono text-textSecondary">
        Today: <span className="text-accentBlue font-bold">{stats.todayCompleted}</span>/{stats.dailyTarget}
      </span>
      <div className="h-3 w-px bg-border" />
      <span className="font-mono text-textSecondary">
        {pct}% done
      </span>
      <div className="h-3 w-px bg-border" />
      <span className={clsx(
        "flex items-center gap-1 font-mono font-bold",
        stats.onTrack ? "text-accentGreen" : "text-accentAmber"
      )}>
        {stats.onTrack
          ? <><TrendingUp className="w-3 h-3" /> On track</>
          : <><TrendingDown className="w-3 h-3" /> Behind</>}
      </span>
    </div>
  );
}
