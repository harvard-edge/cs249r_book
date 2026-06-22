"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import clsx from "clsx";
import type { AreaProgress } from "@/lib/progress";
import { getAreaStyle } from "@/lib/taxonomy";
import TopicProgressBar from "./TopicProgressBar";

interface AreaProgressSectionProps {
  area: AreaProgress;
  defaultOpen?: boolean;
}

export default function AreaProgressSection({ area, defaultOpen = false }: AreaProgressSectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  const style = getAreaStyle(area.area);
  const Icon = style.icon;
  const pct = area.totalQuestions > 0
    ? Math.round((area.correct / area.totalQuestions) * 100)
    : 0;
  const attempted = area.attempted > 0;
  const topicsTouched = area.topics.filter(t => t.attempted > 0).length;

  return (
    <div className="border border-border rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-surface/50 transition-colors"
      >
        <Icon className="w-4 h-4 shrink-0" style={{ color: style.primary }} />
        <span className="text-sm font-semibold text-textPrimary capitalize flex-1 text-left">
          {area.area.replace(/-/g, " ")}
        </span>
        <span className="text-[10px] text-textTertiary font-mono mr-2">
          {topicsTouched}/{area.topics.length} topics
        </span>
        <div className="w-24 h-1.5 bg-border/50 rounded-full overflow-hidden mr-2">
          {pct > 0 && (
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{ width: `${pct}%`, backgroundColor: style.primary }}
            />
          )}
        </div>
        <span className={clsx(
          "text-xs font-bold font-mono w-10 text-right",
          !attempted ? "text-textTertiary" :
          pct >= 70 ? "text-accentGreen" :
          pct >= 40 ? "text-accentAmber" : "text-accentRed"
        )}>
          {attempted ? `${pct}%` : "—"}
        </span>
        <ChevronDown className={clsx(
          "w-4 h-4 text-textTertiary transition-transform duration-200",
          open && "rotate-180"
        )} />
      </button>
      {open && (
        <div className="px-4 pb-3 pt-1 border-t border-border/50 space-y-0.5">
          {area.topics
            .sort((a, b) => b.correct - a.correct || b.attempted - a.attempted || a.topicName.localeCompare(b.topicName))
            .map(topic => (
              <TopicProgressBar
                key={topic.topicId}
                topic={topic}
                accentColor={style.primary}
              />
            ))}
        </div>
      )}
    </div>
  );
}
