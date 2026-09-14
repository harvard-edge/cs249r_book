"use client";

import clsx from "clsx";
import type { TopicProgress } from "@/lib/progress";

interface TopicProgressBarProps {
  topic: TopicProgress;
  accentColor: string;
}

export default function TopicProgressBar({ topic, accentColor }: TopicProgressBarProps) {
  const pct = topic.totalQuestions > 0
    ? Math.round((topic.correct / topic.totalQuestions) * 100)
    : 0;
  const attempted = topic.attempted > 0;

  return (
    <div className="flex items-center gap-3 py-1.5 px-1">
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1">
          <span className={clsx(
            "text-xs truncate",
            attempted ? "text-textSecondary" : "text-textTertiary"
          )}>
            {topic.topicName}
          </span>
          <div className="flex items-center gap-2 shrink-0 ml-2">
            {topic.highestLevel && (
              <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded"
                style={{ color: accentColor, backgroundColor: accentColor + "15" }}>
                {topic.highestLevel}
              </span>
            )}
            <span className={clsx(
              "text-[10px] font-mono",
              attempted ? "text-textSecondary" : "text-textTertiary"
            )}>
              {topic.correct}/{topic.totalQuestions}
            </span>
          </div>
        </div>
        <div className="h-1.5 bg-border/50 rounded-full overflow-hidden">
          {pct > 0 && (
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{ width: `${pct}%`, backgroundColor: accentColor }}
            />
          )}
        </div>
      </div>
    </div>
  );
}
