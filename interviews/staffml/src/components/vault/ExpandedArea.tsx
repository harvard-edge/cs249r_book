import type { CompetencyArea, Topic } from "@/lib/taxonomy";
import { getAreaStyle } from "@/lib/taxonomy";
import { LEVELS as LEVEL_DEFS } from "@/lib/levels";
import TopicCard from "./TopicCard";

export default function ExpandedArea({ area, selectedId, onSelect }: {
  area: CompetencyArea; selectedId: string | null; onSelect: (t: Topic) => void;
}) {
  const style = getAreaStyle(area.id);
  const Icon = style.icon;

  return (
    <div>
      <div className="flex items-center gap-4 mb-5 pb-4 border-b border-borderSubtle">
        <div className="w-12 h-12 rounded-xl flex items-center justify-center shrink-0"
          style={{ backgroundColor: style.bg, border: `1px solid ${style.border}` }}>
          <Icon className="w-6 h-6" style={{ color: style.primary }} />
        </div>
        <div className="flex-1">
          <h2 className="text-[22px] font-bold text-textPrimary">{area.name}</h2>
          <p className="text-[14px] text-textSecondary mt-0.5">
            <span className="font-mono font-semibold">{area.questionCount}</span> questions across{" "}
            <span className="font-mono font-semibold">{area.topicCount}</span> topics
          </p>
        </div>
        {/* Level color legend — top right, mirrors histogram position */}
        <div className="hidden md:flex items-center gap-2.5 shrink-0">
          {LEVEL_DEFS.map((level) => (
            <div key={level.id} className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-sm" style={{ backgroundColor: level.color }} />
              <span className="text-[10px] text-textTertiary font-mono">{level.id}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {area.topics.map((topic) => (
          <TopicCard key={topic.id} topic={topic} style={style}
            isSelected={selectedId === topic.id} onClick={() => onSelect(topic)} />
        ))}
      </div>
    </div>
  );
}
