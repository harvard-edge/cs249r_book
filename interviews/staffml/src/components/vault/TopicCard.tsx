import clsx from "clsx";
import { ChevronRight } from "lucide-react";
import type { Topic, AreaStyle } from "@/lib/taxonomy";
import { LEVELS as LEVEL_DEFS } from "@/lib/levels";

const LEVEL_IDS = LEVEL_DEFS.map(l => l.id);

export default function TopicCard({ topic, style, isSelected, onClick, compact }: {
  topic: Topic; style: AreaStyle; isSelected: boolean;
  onClick: () => void; compact?: boolean;
}) {
  const maxCount = Math.max(...LEVEL_IDS.map(l => topic.levels[l] || 0), 1);

  return (
    <button
      onClick={onClick}
      aria-label={`${topic.name}, ${topic.questionCount} questions`}
      className={clsx(
        "w-full text-left rounded-xl border transition-all duration-150 group relative overflow-hidden",
        compact ? "p-3 pt-4" : "p-4 pt-5",
        isSelected
          ? "bg-surfaceElevated border-borderHighlight shadow-[0_0_0_1px_rgba(255,255,255,0.05)]"
          : "bg-surface border-borderSubtle hover:bg-surfaceElevated hover:border-borderHighlight"
      )}
    >
      {/* Colored top accent bar */}
      <div className="absolute top-0 left-0 right-0 h-[2px] opacity-60"
        style={{ background: `linear-gradient(90deg, ${style.primary}, transparent 80%)` }} />

      {/* Title */}
      <div className="flex items-start justify-between gap-2 mb-2">
        <h3 className={clsx(
          "font-semibold text-textPrimary leading-snug",
          compact ? "text-[13px]" : "text-[14px]"
        )}>
          {topic.name}
        </h3>
        <ChevronRight className="w-4 h-4 text-textMuted shrink-0 mt-0.5 group-hover:text-textTertiary transition-colors" />
      </div>

      {/* Question count + mini histogram */}
      <div className="flex items-end justify-between gap-3">
        <div className="flex items-baseline gap-1.5">
          <span className={clsx("font-bold font-mono text-textPrimary", compact ? "text-[20px]" : "text-[22px]")}>
            {topic.questionCount}
          </span>
          <span className="text-[12px] font-medium text-textTertiary">questions</span>
        </div>

        {/* Mini histogram — same level colors as area overview */}
        <div className="flex items-end gap-[2px] h-5">
          {LEVEL_IDS.map((level) => {
            const count = topic.levels[level] || 0;
            const height = count > 0 ? Math.max(4, (count / maxCount) * 20) : 3;
            const levelDef = LEVEL_DEFS.find(l => l.id === level);
            return (
              <div key={level} className="flex flex-col items-center gap-0">
                <div
                  className="w-[4px] rounded-sm"
                  style={{
                    height,
                    backgroundColor: count > 0 ? (levelDef?.color || "#999") : "var(--surface-hover)",
                  }}
                  title={`${level} ${levelDef?.name || ''}: ${count}`}
                />
              </div>
            );
          })}
        </div>
      </div>
    </button>
  );
}
