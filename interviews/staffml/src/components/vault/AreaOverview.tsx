import { ChevronRight } from "lucide-react";
import type { CompetencyArea, Topic } from "@/lib/taxonomy";
import { getAreaStyle } from "@/lib/taxonomy";
import { LEVELS as LEVEL_DEFS } from "@/lib/levels";
import TopicCard from "./TopicCard";

const LEVEL_IDS = LEVEL_DEFS.map(l => l.id);

export default function AreaOverview({ areas, onExpand, onSelectTopic, selectedId }: {
  areas: CompetencyArea[]; onExpand: (id: string) => void;
  onSelectTopic: (t: Topic) => void; selectedId: string | null;
}) {
  return (
    <div className="space-y-2">
      {areas.map((area) => {
        const style = getAreaStyle(area.id);
        const Icon = style.icon;
        return (
          <div key={area.id} className="rounded-xl border border-borderSubtle bg-surface hover:bg-surfaceElevated hover:border-borderHighlight transition-all">
            <button
              onClick={() => onExpand(area.id)}
              className="w-full flex items-center gap-4 p-4 text-left group"
            >
              <div className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0"
                style={{ backgroundColor: style.bg, border: `1px solid ${style.border}` }}>
                <Icon className="w-5 h-5" style={{ color: style.primary }} />
              </div>
              <div className="flex-1 min-w-0">
                <h2 className="text-[16px] font-bold text-textPrimary">{area.name}</h2>
                <p className="text-[13px] text-textSecondary mt-0.5">
                  <span className="font-mono">{area.questionCount}</span> questions &middot;{" "}
                  <span className="font-mono">{area.topicCount}</span> topics
                </p>
              </div>
              {/* Level histogram */}
              <div className="hidden md:flex items-end gap-[3px] h-8 mr-2">
                {LEVEL_IDS.map((level) => {
                  const count = area.levels[level] || 0;
                  const maxCount = Math.max(...LEVEL_IDS.map(l => area.levels[l] || 0), 1);
                  const height = count > 0 ? Math.max(6, (count / maxCount) * 32) : 4;
                  const levelDef = LEVEL_DEFS.find(l => l.id === level);
                  return (
                    <div key={level} className="flex flex-col items-center gap-0.5">
                      <div className="w-[6px] rounded-sm"
                        style={{
                          height,
                          backgroundColor: count > 0 ? (levelDef?.color || "#999") : "var(--surface-hover)",
                        }}
                        title={`${level} ${levelDef?.name || ''}: ${count}`}
                      />
                      <span className="text-[9px] text-textMuted font-mono">{level.replace("L","").replace("+","")}</span>
                    </div>
                  );
                })}
              </div>
              <ChevronRight className="w-5 h-5 text-textMuted group-hover:text-textSecondary shrink-0 transition-colors" />
            </button>

            <div className="px-4 pb-4 pt-0 grid grid-cols-1 sm:grid-cols-3 gap-2">
              {area.topics.slice(0, 3).map((topic) => (
                <TopicCard key={topic.id} topic={topic} style={style}
                  isSelected={selectedId === topic.id} onClick={() => onSelectTopic(topic)} compact />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
