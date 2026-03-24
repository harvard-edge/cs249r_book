"use client";

import { useState, useMemo } from "react";
import {
  Search,
  BookOpen,
  ChevronRight,
  ChevronLeft,
  X,
  ExternalLink,
  Target,
  Play,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import clsx from "clsx";
import Link from "next/link";
import {
  getAreas,
  getVaultStats,
  getAreaColor,
  getAreaForTopic,
  searchTopics,
  type Topic,
  type CompetencyArea,
} from "@/lib/taxonomy";

const LEVELS = ["L1", "L2", "L3", "L4", "L5", "L6+"];
const LEVEL_LABELS: Record<string, string> = {
  L1: "Recall",
  L2: "Understand",
  L3: "Apply",
  L4: "Analyze",
  L5: "Design",
  "L6+": "Architect",
};

export default function VaultPage() {
  const [query, setQuery] = useState("");
  const [selectedTopic, setSelectedTopic] = useState<Topic | null>(null);
  const [expandedArea, setExpandedArea] = useState<string | null>(null);
  const stats = getVaultStats();
  const areas = getAreas();

  const searchResults = useMemo(() => {
    if (!query.trim()) return null;
    return searchTopics(query);
  }, [query]);

  const selectedArea = selectedTopic
    ? getAreaForTopic(selectedTopic.id)
    : null;

  return (
    <div className="flex-1 flex flex-col min-h-screen">
      {/* Hero */}
      <div className="px-6 pt-10 pb-8 border-b border-border">
        <div className="max-w-5xl mx-auto">
          <div className="mb-6">
            <h1 className="text-3xl font-extrabold text-white tracking-tight mb-2">
              Question Vault
            </h1>
            <p className="text-[15px] text-textSecondary">
              {stats.totalQuestions.toLocaleString()} questions across{" "}
              {stats.totalTopics} topics — find your weak spots and drill them.
            </p>
          </div>

          {/* Area filter pills */}
          <div className="flex items-center gap-2 flex-wrap mb-6">
            <button
              onClick={() => setExpandedArea(null)}
              className={clsx(
                "px-3 py-1.5 rounded-full text-xs font-semibold transition-all border",
                !expandedArea
                  ? "border-white/30 bg-white/10 text-white"
                  : "border-transparent text-textSecondary hover:text-white hover:bg-white/5"
              )}
            >
              All
            </button>
            {areas.map((area) => (
              <button
                key={area.id}
                onClick={() =>
                  setExpandedArea(expandedArea === area.id ? null : area.id)
                }
                className={clsx(
                  "flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold transition-all border",
                  expandedArea === area.id
                    ? "border-white/30 bg-white/10 text-white"
                    : "border-transparent text-textSecondary hover:text-white hover:bg-white/5"
                )}
              >
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: getAreaColor(area.id) }}
                />
                {area.name}
                <span className="font-mono opacity-50">
                  {area.questionCount}
                </span>
              </button>
            ))}
          </div>

          {/* Search */}
          <div className="relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-textTertiary" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search topics — e.g. KV cache, roofline, quantization..."
              className="w-full pl-12 pr-12 py-3.5 bg-surface border border-border rounded-xl text-[15px] text-white placeholder:text-textTertiary focus:outline-none focus:border-white/30 focus:bg-surface/80 transition-all"
            />
            {query && (
              <button
                onClick={() => setQuery("")}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-textTertiary hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex">
        <div className="flex-1 overflow-auto px-6 py-6">
          <div className="max-w-5xl mx-auto">
            {searchResults ? (
              <div>
                <p className="text-sm text-textSecondary mb-4">
                  {searchResults.length} topic
                  {searchResults.length !== 1 ? "s" : ""} matching &ldquo;
                  {query}&rdquo;
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {searchResults.map((topic) => (
                    <TopicCard
                      key={topic.id}
                      topic={topic}
                      areaColor={getAreaColor(
                        getAreaForTopic(topic.id)?.id || ""
                      )}
                      areaName={getAreaForTopic(topic.id)?.name}
                      isSelected={selectedTopic?.id === topic.id}
                      onClick={() => setSelectedTopic(topic)}
                    />
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-10">
                {areas
                  .filter((a) => !expandedArea || a.id === expandedArea)
                  .map((area) => (
                    <AreaSection
                      key={area.id}
                      area={area}
                      selectedTopicId={selectedTopic?.id ?? null}
                      onSelectTopic={setSelectedTopic}
                      isExpanded={expandedArea === area.id}
                    />
                  ))}
              </div>
            )}
          </div>
        </div>

        {/* Detail panel */}
        <AnimatePresence>
          {selectedTopic && (
            <motion.div
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 420, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="shrink-0 overflow-hidden border-l border-border"
            >
              <TopicDetail
                topic={selectedTopic}
                areaName={selectedArea?.name || ""}
                areaColor={getAreaColor(selectedArea?.id || "")}
                onClose={() => setSelectedTopic(null)}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

// ─── Area Section ──────────────────────────────────────────────

function AreaSection({
  area,
  selectedTopicId,
  onSelectTopic,
  isExpanded,
}: {
  area: CompetencyArea;
  selectedTopicId: string | null;
  onSelectTopic: (t: Topic) => void;
  isExpanded: boolean;
}) {
  const color = getAreaColor(area.id);
  const visibleTopics = isExpanded ? area.topics : area.topics.slice(0, 6);

  return (
    <div>
      <div className="flex items-center gap-3 mb-4">
        <div
          className="w-3 h-3 rounded-sm"
          style={{ backgroundColor: color }}
        />
        <h2 className="text-lg font-bold text-white">{area.name}</h2>
        <span className="text-sm text-textSecondary">
          {area.questionCount} questions &middot; {area.topicCount} topics
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {visibleTopics.map((topic) => (
          <TopicCard
            key={topic.id}
            topic={topic}
            areaColor={color}
            isSelected={selectedTopicId === topic.id}
            onClick={() => onSelectTopic(topic)}
          />
        ))}
      </div>

      {!isExpanded && area.topics.length > 6 && (
        <p className="mt-3 text-xs text-textSecondary">
          +{area.topics.length - 6} more topics in {area.name}
        </p>
      )}
    </div>
  );
}

// ─── Topic Card ────────────────────────────────────────────────

function TopicCard({
  topic,
  areaColor,
  areaName,
  isSelected,
  onClick,
}: {
  topic: Topic;
  areaColor: string;
  areaName?: string;
  isSelected: boolean;
  onClick: () => void;
}) {
  const maxLevel = Math.max(...Object.values(topic.levels), 1);

  return (
    <button
      onClick={onClick}
      className={clsx(
        "w-full text-left p-4 rounded-xl border transition-all group",
        isSelected
          ? "border-white/30 bg-white/[0.07]"
          : "border-border hover:border-white/20 bg-surface/50 hover:bg-surface"
      )}
    >
      <div className="flex items-start justify-between gap-2 mb-3">
        <h3 className="text-[13px] font-semibold text-white leading-snug">
          {topic.name}
        </h3>
        <ChevronRight className="w-4 h-4 text-textTertiary shrink-0 mt-0.5 group-hover:text-white/60" />
      </div>

      {areaName && (
        <div className="flex items-center gap-1.5 mb-2">
          <span
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: areaColor }}
          />
          <span className="text-xs text-textSecondary">{areaName}</span>
        </div>
      )}

      <div className="flex items-end justify-between">
        <div>
          <span className="text-2xl font-bold font-mono text-white">
            {topic.questionCount}
          </span>
          <span className="text-xs text-textSecondary ml-1.5">Qs</span>
        </div>
        {/* Mini level bars */}
        <div className="flex items-end gap-[3px] h-5">
          {LEVELS.map((level) => {
            const count = topic.levels[level] || 0;
            const height =
              count > 0 ? Math.max(5, (count / maxLevel) * 20) : 3;
            return (
              <div
                key={level}
                className="w-[5px] rounded-sm"
                style={{
                  height,
                  backgroundColor:
                    count > 0 ? areaColor : "rgba(255,255,255,0.1)",
                  opacity: count > 0 ? 0.85 : 1,
                }}
                title={`${level}: ${count} questions`}
              />
            );
          })}
        </div>
      </div>
    </button>
  );
}

// ─── Topic Detail Panel ────────────────────────────────────────

function TopicDetail({
  topic,
  areaName,
  areaColor,
  onClose,
}: {
  topic: Topic;
  areaName: string;
  areaColor: string;
  onClose: () => void;
}) {
  const [drillLevel, setDrillLevel] = useState<string | null>(null);

  const topicId = topic.id;
  const [prevTopicId, setPrevTopicId] = useState(topicId);
  if (topicId !== prevTopicId) {
    setPrevTopicId(topicId);
    setDrillLevel(null);
  }

  const questionsForLevel = drillLevel
    ? topic.questionsByLevel[drillLevel] || []
    : [];

  return (
    <div className="w-[420px] h-full flex flex-col bg-background">
      {/* Header */}
      <div className="p-5 border-b border-border">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: areaColor }}
              />
              <span className="text-xs font-semibold text-textSecondary uppercase tracking-wide">
                {areaName}
              </span>
            </div>
            <h2 className="text-xl font-bold text-white leading-tight">
              {topic.name}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-textTertiary hover:text-white hover:bg-white/10 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-auto">
        <AnimatePresence mode="wait">
          {drillLevel ? (
            /* ─── Level drill-down view ─── */
            <motion.div
              key={`level-${drillLevel}`}
              initial={{ x: 50, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 50, opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="p-5"
            >
              <button
                onClick={() => setDrillLevel(null)}
                className="flex items-center gap-1.5 text-sm text-textSecondary hover:text-white mb-5 transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
                Back to overview
              </button>

              <div className="flex items-center justify-between mb-5">
                <div>
                  <h3 className="text-xl font-bold text-white">
                    {drillLevel}{" "}
                    <span className="text-base font-medium text-textSecondary">
                      {LEVEL_LABELS[drillLevel]}
                    </span>
                  </h3>
                  <p className="text-sm text-textSecondary mt-0.5">
                    {questionsForLevel.length} question
                    {questionsForLevel.length !== 1 ? "s" : ""}
                  </p>
                </div>
                <Link
                  href={`/drill?topic=${topic.id}&level=${drillLevel}`}
                  className="flex items-center gap-1.5 px-4 py-2 bg-white text-black text-sm font-bold rounded-lg hover:bg-gray-100 transition-all"
                >
                  <Target className="w-4 h-4" />
                  Drill
                </Link>
              </div>

              <div className="space-y-2">
                {questionsForLevel.map((q) => (
                  <Link
                    key={q.id}
                    href={`/drill?q=${q.id}`}
                    className="block p-3.5 rounded-lg border border-border bg-surface/50 hover:bg-surface hover:border-white/20 transition-all group"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="text-sm font-medium text-white leading-snug mb-1">
                          {q.title}
                        </p>
                        <p className="text-xs text-textSecondary line-clamp-2 leading-relaxed">
                          {q.scenario
                            .replace(/^-\s*\*\*Interviewer:\*\*\s*/i, "")
                            .replace(/^"/, "")
                            .replace(/"$/, "")
                            .slice(0, 140)}
                          ...
                        </p>
                      </div>
                      <div className="flex items-center gap-2 shrink-0 mt-0.5">
                        <span className="text-xs text-textSecondary capitalize">
                          {q.track}
                        </span>
                        <Play className="w-3.5 h-3.5 text-textTertiary group-hover:text-white" />
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            </motion.div>
          ) : (
            /* ─── Overview ─── */
            <motion.div
              key="overview"
              initial={{ x: -50, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -50, opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="p-5 space-y-6"
            >
              {topic.description && (
                <p className="text-sm text-textSecondary leading-relaxed">
                  {topic.description}
                </p>
              )}

              {/* Level breakdown — clickable rows */}
              <div>
                <h3 className="text-xs font-semibold text-textSecondary uppercase tracking-wide mb-3">
                  Difficulty Levels
                </h3>
                <div className="space-y-1.5">
                  {LEVELS.map((level) => {
                    const count = topic.levels[level] || 0;
                    if (count === 0) return null;
                    const pct = (count / topic.questionCount) * 100;
                    return (
                      <button
                        key={level}
                        onClick={() => setDrillLevel(level)}
                        className="w-full flex items-center gap-3 p-3 rounded-lg border border-border bg-surface/30 hover:bg-surface hover:border-white/20 transition-all group"
                      >
                        <span className="text-sm font-bold font-mono text-white w-8">
                          {level}
                        </span>
                        <div className="flex-1">
                          <div className="flex items-center justify-between mb-1.5">
                            <span className="text-xs font-medium text-textSecondary">
                              {LEVEL_LABELS[level]}
                            </span>
                            <span className="text-sm font-mono font-semibold text-white">
                              {count}
                            </span>
                          </div>
                          <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full"
                              style={{
                                width: `${pct}%`,
                                backgroundColor: areaColor,
                              }}
                            />
                          </div>
                        </div>
                        <ChevronRight className="w-4 h-4 text-textTertiary group-hover:text-white shrink-0" />
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Drill all */}
              <Link
                href={`/drill?topic=${topic.id}`}
                className="flex items-center justify-center gap-2 w-full py-3.5 bg-white text-black font-bold rounded-lg hover:bg-gray-100 transition-all text-sm"
              >
                <Target className="w-4 h-4" />
                Drill All {topic.questionCount} Questions
              </Link>

              {/* Tracks */}
              <div>
                <h3 className="text-xs font-semibold text-textSecondary uppercase tracking-wide mb-2">
                  Available In
                </h3>
                <div className="flex gap-2 flex-wrap">
                  {topic.tracks.map((t) => (
                    <span
                      key={t}
                      className="px-3 py-1.5 rounded-full bg-surface border border-border text-sm text-white font-medium"
                    >
                      {t === "tinyml"
                        ? "TinyML"
                        : t.charAt(0).toUpperCase() + t.slice(1)}
                    </span>
                  ))}
                </div>
              </div>

              {/* Textbook link */}
              {topic.chapterUrl && (
                <div>
                  <h3 className="text-xs font-semibold text-textSecondary uppercase tracking-wide mb-2">
                    Deep Dive
                  </h3>
                  <a
                    href={topic.chapterUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-3 p-4 rounded-lg border border-border bg-surface/50 hover:bg-surface hover:border-white/20 transition-all group"
                  >
                    <BookOpen className="w-5 h-5 text-textSecondary group-hover:text-accentBlue shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-white">
                        {topic.chapterName}
                      </p>
                      <p className="text-xs text-textSecondary">
                        Read in the textbook at mlsysbook.ai
                      </p>
                    </div>
                    <ExternalLink className="w-4 h-4 text-textTertiary group-hover:text-white shrink-0" />
                  </a>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
