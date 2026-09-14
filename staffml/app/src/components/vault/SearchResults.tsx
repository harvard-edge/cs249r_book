import { useMemo } from "react";
import type { Topic, CompetencyArea } from "@/lib/taxonomy";
import { getAreaForTopic, getAreaStyle } from "@/lib/taxonomy";
import TopicCard from "./TopicCard";

export default function SearchResults({ results, query, selectedId, onSelect }: {
  results: Topic[]; query: string; selectedId: string | null; onSelect: (t: Topic) => void;
}) {
  const grouped = useMemo(() => {
    const map = new Map<string, { area: CompetencyArea; topics: Topic[] }>();
    for (const t of results) {
      const area = getAreaForTopic(t.id);
      if (!area) continue;
      if (!map.has(area.id)) map.set(area.id, { area, topics: [] });
      map.get(area.id)!.topics.push(t);
    }
    return Array.from(map.values());
  }, [results]);

  return (
    <div>
      <p className="text-[14px] text-textSecondary mb-5">
        {results.length} topic{results.length !== 1 ? "s" : ""} matching &ldquo;{query}&rdquo;
      </p>
      {grouped.map(({ area, topics }) => {
        const style = getAreaStyle(area.id);
        const Icon = style.icon;
        return (
          <div key={area.id} className="mb-6">
            <div className="flex items-center gap-2 mb-3">
              <Icon className="w-4 h-4" style={{ color: style.primary }} />
              <span className="text-[13px] font-semibold text-textSecondary">{area.name}</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {topics.map((topic) => (
                <TopicCard key={topic.id} topic={topic} style={style}
                  isSelected={selectedId === topic.id} onClick={() => onSelect(topic)} />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
