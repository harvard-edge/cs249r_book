"use client";

import { useState, useMemo } from "react";
import {
  Search, BookOpen, ChevronRight, ChevronLeft, X,
  ExternalLink, Target, Play,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import clsx from "clsx";
import Link from "next/link";
import {
  getAreas, getVaultStats, getAreaStyle, getAreaForTopic, searchTopics,
  type Topic, type CompetencyArea, type AreaStyle,
} from "@/lib/taxonomy";

const LEVELS = ["L1", "L2", "L3", "L4", "L5", "L6+"];
const LEVEL_LABELS: Record<string, string> = {
  L1: "Recall", L2: "Understand", L3: "Apply",
  L4: "Analyze", L5: "Design", "L6+": "Architect",
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

  const selectedArea = selectedTopic ? getAreaForTopic(selectedTopic.id) : null;
  const selectedStyle = selectedArea ? getAreaStyle(selectedArea.id) : null;

  return (
    <div className="flex-1 flex flex-col min-h-screen">
      {/* ─── Header ─── */}
      <div className="px-6 pt-8 pb-6 border-b border-border">
        <div className="max-w-5xl mx-auto">
          <h1 className="text-[28px] font-extrabold text-white tracking-tight mb-1">
            Question Vault
          </h1>
          <p className="text-[15px] text-textSecondary mb-5">
            {stats.totalQuestions.toLocaleString()} questions across{" "}
            {stats.totalTopics} topics — find your weak spots and drill them.
          </p>

          {/* Area filter pills */}
          <div className="flex items-center gap-1.5 flex-wrap mb-5">
            <FilterPill
              label="All"
              isActive={!expandedArea}
              onClick={() => setExpandedArea(null)}
            />
            {areas.map((area) => {
              const style = getAreaStyle(area.id);
              const Icon = style.icon;
              return (
                <FilterPill
                  key={area.id}
                  label={area.name}
                  count={area.questionCount}
                  isActive={expandedArea === area.id}
                  color={style.primary}
                  icon={<Icon className="w-3 h-3" />}
                  onClick={() => setExpandedArea(expandedArea === area.id ? null : area.id)}
                />
              );
            })}
          </div>

          {/* Search */}
          <div className="relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-textMuted" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search topics — KV cache, roofline, quantization..."
              className="w-full pl-12 pr-12 py-3 bg-surface border border-border rounded-xl text-[15px] font-medium text-white placeholder:text-textTertiary focus:outline-none focus:border-borderHighlight transition-colors"
            />
            {query && (
              <button onClick={() => setQuery("")}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-textTertiary hover:text-white">
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* ─── Main ─── */}
      <div className="flex-1 flex overflow-hidden">
        <div className="flex-1 overflow-auto px-6 py-6">
          <div className="max-w-5xl mx-auto">
            {searchResults ? (
              <SearchResults results={searchResults} query={query}
                selectedId={selectedTopic?.id ?? null} onSelect={setSelectedTopic} />
            ) : expandedArea ? (
              <ExpandedArea area={areas.find(a => a.id === expandedArea)!}
                selectedId={selectedTopic?.id ?? null} onSelect={setSelectedTopic} />
            ) : (
              <AreaOverview areas={areas} onExpand={setExpandedArea} onSelectTopic={setSelectedTopic}
                selectedId={selectedTopic?.id ?? null} />
            )}
          </div>
        </div>

        {/* Detail panel */}
        <AnimatePresence>
          {selectedTopic && selectedStyle && (
            <motion.div
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 420, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="shrink-0 overflow-hidden border-l border-border hidden lg:block"
            >
              <TopicDetail topic={selectedTopic}
                areaName={selectedArea?.name || ""} style={selectedStyle}
                onClose={() => setSelectedTopic(null)} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Mobile detail sheet */}
      <AnimatePresence>
        {selectedTopic && selectedStyle && (
          <motion.div
            initial={{ y: "100%" }}
            animate={{ y: 0 }}
            exit={{ y: "100%" }}
            transition={{ type: "spring", damping: 30, stiffness: 300 }}
            className="fixed inset-x-0 bottom-0 z-50 border-t border-border rounded-t-2xl max-h-[85vh] overflow-auto bg-background lg:hidden"
          >
            <div className="flex justify-center pt-3 pb-1">
              <div className="w-10 h-1 rounded-full bg-borderHighlight" />
            </div>
            <TopicDetail topic={selectedTopic}
              areaName={selectedArea?.name || ""} style={selectedStyle}
              onClose={() => setSelectedTopic(null)} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Filter Pill ───────────────────────────────────────────────

function FilterPill({ label, count, isActive, color, icon, onClick }: {
  label: string; count?: number; isActive: boolean;
  color?: string; icon?: React.ReactNode; onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      aria-pressed={isActive}
      className={clsx(
        "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-semibold transition-all border",
        isActive
          ? "text-white border-borderHighlight"
          : "border-transparent text-textSecondary hover:text-white hover:bg-surface"
      )}
      style={isActive && color ? { backgroundColor: `${color}12`, borderColor: `${color}30`, color } : undefined}
    >
      {icon}
      {label}
      {count !== undefined && (
        <span className={clsx("font-mono text-[11px]", isActive ? "opacity-70" : "text-textMuted")}>
          {count}
        </span>
      )}
    </button>
  );
}

// ─── Tier 1: Area Overview ─────────────────────────────────────

function AreaOverview({ areas, onExpand, onSelectTopic, selectedId }: {
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
            {/* Area header — click to expand */}
            <button
              onClick={() => onExpand(area.id)}
              className="w-full flex items-center gap-4 p-4 text-left group"
            >
              <div className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0"
                style={{ backgroundColor: style.bg, border: `1px solid ${style.border}` }}>
                <Icon className="w-5 h-5" style={{ color: style.primary }} />
              </div>
              <div className="flex-1 min-w-0">
                <h2 className="text-[16px] font-bold text-white">{area.name}</h2>
                <p className="text-[13px] text-textSecondary mt-0.5">
                  <span className="font-mono">{area.questionCount}</span> questions &middot;{" "}
                  <span className="font-mono">{area.topicCount}</span> topics
                </p>
              </div>
              {/* Level histogram */}
              <div className="hidden md:flex items-end gap-[3px] h-8 mr-2">
                {LEVELS.map((level) => {
                  const count = area.levels[level] || 0;
                  const maxCount = Math.max(...LEVELS.map(l => area.levels[l] || 0), 1);
                  const height = count > 0 ? Math.max(6, (count / maxCount) * 32) : 4;
                  return (
                    <div key={level} className="flex flex-col items-center gap-0.5">
                      <div className="w-[6px] rounded-sm"
                        style={{
                          height,
                          backgroundColor: count > 0 ? style.primary : "rgba(255,255,255,0.06)",
                          opacity: count > 0 ? 0.8 : 1,
                        }}
                        title={`${level}: ${count}`}
                      />
                      <span className="text-[9px] text-textMuted font-mono">{level.replace("L","").replace("+","")}</span>
                    </div>
                  );
                })}
              </div>
              <ChevronRight className="w-5 h-5 text-textMuted group-hover:text-textSecondary shrink-0 transition-colors" />
            </button>

            {/* Quick-access: top 3 topic cards inline */}
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

// ─── Tier 2: Expanded Area ─────────────────────────────────────

function ExpandedArea({ area, selectedId, onSelect }: {
  area: CompetencyArea; selectedId: string | null; onSelect: (t: Topic) => void;
}) {
  const style = getAreaStyle(area.id);
  const Icon = style.icon;

  return (
    <div>
      {/* Area header */}
      <div className="flex items-center gap-4 mb-6 pb-4 border-b border-borderSubtle">
        <div className="w-12 h-12 rounded-xl flex items-center justify-center shrink-0"
          style={{ backgroundColor: style.bg, border: `1px solid ${style.border}` }}>
          <Icon className="w-6 h-6" style={{ color: style.primary }} />
        </div>
        <div>
          <h2 className="text-[22px] font-bold text-white">{area.name}</h2>
          <p className="text-[14px] text-textSecondary mt-0.5">
            <span className="font-mono font-semibold">{area.questionCount}</span> questions across{" "}
            <span className="font-mono font-semibold">{area.topicCount}</span> topics
          </p>
        </div>
      </div>

      {/* Topic grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {area.topics.map((topic) => (
          <TopicCard key={topic.id} topic={topic} style={style}
            isSelected={selectedId === topic.id} onClick={() => onSelect(topic)} />
        ))}
      </div>
    </div>
  );
}

// ─── Search Results ────────────────────────────────────────────

function SearchResults({ results, query, selectedId, onSelect }: {
  results: Topic[]; query: string; selectedId: string | null; onSelect: (t: Topic) => void;
}) {
  // Group by area
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

// ─── Topic Card ────────────────────────────────────────────────

function TopicCard({ topic, style, isSelected, onClick, compact }: {
  topic: Topic; style: AreaStyle; isSelected: boolean;
  onClick: () => void; compact?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      aria-label={`${topic.name}, ${topic.questionCount} questions`}
      className={clsx(
        "w-full text-left rounded-xl border transition-all duration-150 group relative overflow-hidden",
        compact ? "p-3 pt-4" : "p-4 pt-5",
        isSelected
          ? "bg-surfaceElevated border-borderHighlight shadow-[0_0_0_1px_rgba(255,255,255,0.05)]"
          : "bg-surface border-borderSubtle hover:bg-surfaceElevated hover:border-borderHighlight hover:shadow-[0_2px_12px_rgba(0,0,0,0.3)]"
      )}
    >
      {/* Colored top accent bar */}
      <div className="absolute top-0 left-0 right-0 h-[2px] opacity-60"
        style={{ background: `linear-gradient(90deg, ${style.primary}, transparent 80%)` }} />

      {/* Title */}
      <div className="flex items-start justify-between gap-2 mb-2">
        <h3 className={clsx(
          "font-semibold text-white leading-snug",
          compact ? "text-[13px]" : "text-[14px]"
        )}>
          {topic.name}
        </h3>
        <ChevronRight className="w-4 h-4 text-textMuted shrink-0 mt-0.5 group-hover:text-textTertiary transition-colors" />
      </div>

      {/* Question count */}
      <div className="flex items-baseline gap-1.5 mb-2.5">
        <span className={clsx("font-bold font-mono text-white", compact ? "text-[20px]" : "text-[22px]")}>
          {topic.questionCount}
        </span>
        <span className="text-[12px] font-medium text-textTertiary">questions</span>
      </div>

      {/* Horizontal stacked level bar */}
      <div className="flex h-1.5 rounded-full overflow-hidden bg-white/[0.04]">
        {LEVELS.map((level) => {
          const count = topic.levels[level] || 0;
          if (count === 0) return null;
          const pct = (count / topic.questionCount) * 100;
          return (
            <div key={level} className="h-full first:rounded-l-full last:rounded-r-full"
              style={{
                width: `${pct}%`,
                backgroundColor: style.primary,
                opacity: 0.3 + (LEVELS.indexOf(level) / LEVELS.length) * 0.7,
              }}
              title={`${level}: ${count} questions`}
            />
          );
        })}
      </div>
    </button>
  );
}

// ─── Topic Detail Panel ────────────────────────────────────────

function TopicDetail({ topic, areaName, style, onClose }: {
  topic: Topic; areaName: string; style: AreaStyle; onClose: () => void;
}) {
  const [drillLevel, setDrillLevel] = useState<string | null>(null);
  const Icon = style.icon;

  // Reset on topic change
  const [prevId, setPrevId] = useState(topic.id);
  if (topic.id !== prevId) { setPrevId(topic.id); setDrillLevel(null); }

  const levelQs = drillLevel ? topic.questionsByLevel[drillLevel] || [] : [];

  return (
    <div className="w-[420px] h-full flex flex-col bg-background">
      {/* Header with area gradient wash */}
      <div className="p-5 border-b border-border"
        style={{ background: `linear-gradient(180deg, ${style.primary}08 0%, transparent 100%)` }}>
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Icon className="w-4 h-4" style={{ color: style.primary }} />
              <span className="text-[12px] font-semibold uppercase tracking-wide"
                style={{ color: style.primary }}>{areaName}</span>
            </div>
            <h2 className="text-[20px] font-bold text-white leading-tight">{topic.name}</h2>
          </div>
          <button onClick={onClose} aria-label="Close"
            className="p-1.5 text-textTertiary hover:text-white hover:bg-white/10 rounded-lg transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-auto">
        <AnimatePresence mode="wait">
          {drillLevel ? (
            /* Level drill-down */
            <motion.div key={`level-${drillLevel}`}
              initial={{ x: 40, opacity: 0 }} animate={{ x: 0, opacity: 1 }}
              exit={{ x: 40, opacity: 0 }} transition={{ duration: 0.15 }}
              className="p-5">

              {/* Breadcrumb */}
              <button onClick={() => setDrillLevel(null)}
                className="flex items-center gap-1.5 text-[13px] font-medium text-textSecondary hover:text-white mb-5 transition-colors">
                <ChevronLeft className="w-4 h-4" /> Back to overview
              </button>

              <div className="flex items-center justify-between mb-5">
                <div>
                  <h3 className="text-[20px] font-bold text-white">
                    {drillLevel}{" "}
                    <span className="text-[15px] font-medium text-textSecondary">{LEVEL_LABELS[drillLevel]}</span>
                  </h3>
                  <p className="text-[13px] text-textSecondary mt-0.5">
                    {levelQs.length} question{levelQs.length !== 1 ? "s" : ""}
                  </p>
                </div>
                <Link href={`/drill?topic=${topic.id}&level=${drillLevel}`}
                  className="flex items-center gap-1.5 px-4 py-2 text-[13px] font-bold rounded-lg transition-all"
                  style={{ backgroundColor: style.primary, color: "#101014" }}>
                  <Target className="w-4 h-4" /> Drill
                </Link>
              </div>

              <div className="space-y-2">
                {levelQs.map((q) => (
                  <Link key={q.id} href={`/drill?q=${q.id}`}
                    className="block p-3.5 rounded-xl border border-borderSubtle bg-surface hover:bg-surfaceElevated hover:border-borderHighlight transition-all group">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="text-[14px] font-semibold text-white leading-snug mb-1">{q.title}</p>
                        <p className="text-[13px] text-textSecondary line-clamp-2 leading-relaxed">
                          {q.scenario.replace(/^-\s*\*\*Interviewer:\*\*\s*/i, "").replace(/^"/, "").replace(/"$/, "").slice(0, 140)}...
                        </p>
                      </div>
                      <div className="flex items-center gap-2 shrink-0 mt-0.5">
                        <span className="text-[12px] text-textTertiary capitalize font-medium">{q.track}</span>
                        <Play className="w-3.5 h-3.5 text-textMuted group-hover:text-white transition-colors" />
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            </motion.div>
          ) : (
            /* Overview */
            <motion.div key="overview"
              initial={{ x: -40, opacity: 0 }} animate={{ x: 0, opacity: 1 }}
              exit={{ x: -40, opacity: 0 }} transition={{ duration: 0.15 }}
              className="p-5 space-y-6">

              {topic.description && (
                <p className="text-[14px] text-textSecondary leading-relaxed">{topic.description}</p>
              )}

              {/* Difficulty levels */}
              <div>
                <SectionDivider label="Difficulty Levels" />
                <div className="space-y-1.5 mt-3">
                  {LEVELS.map((level) => {
                    const count = topic.levels[level] || 0;
                    if (count === 0) return null;
                    const pct = (count / topic.questionCount) * 100;
                    return (
                      <button key={level} onClick={() => setDrillLevel(level)}
                        className="w-full flex items-center gap-3 px-4 py-3 rounded-xl border border-borderSubtle bg-surface hover:bg-surfaceElevated hover:border-borderHighlight transition-all group">
                        <span className="text-[14px] font-bold font-mono text-white w-8">{level}</span>
                        <div className="flex-1">
                          <div className="flex items-center justify-between mb-1.5">
                            <span className="text-[13px] font-medium text-textSecondary">{LEVEL_LABELS[level]}</span>
                            <span className="text-[14px] font-mono font-bold text-white">{count}</span>
                          </div>
                          <div className="h-1.5 bg-white/[0.04] rounded-full overflow-hidden">
                            <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: style.primary }} />
                          </div>
                        </div>
                        <ChevronRight className="w-4 h-4 text-textMuted group-hover:text-textSecondary shrink-0 transition-colors" />
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Drill all CTA */}
              <Link href={`/drill?topic=${topic.id}`}
                className="flex items-center justify-center gap-2 w-full py-3.5 font-bold rounded-xl text-[14px] transition-all hover:opacity-90"
                style={{ backgroundColor: style.primary, color: "#101014" }}>
                <Target className="w-4 h-4" />
                Drill All {topic.questionCount} Questions
              </Link>

              {/* Tracks */}
              <div>
                <SectionDivider label="Available In" />
                <div className="flex gap-2 flex-wrap mt-3">
                  {topic.tracks.map((t) => (
                    <span key={t} className="px-3 py-1.5 rounded-lg bg-surface border border-borderSubtle text-[13px] text-white font-medium">
                      {t === "tinyml" ? "TinyML" : t.charAt(0).toUpperCase() + t.slice(1)}
                    </span>
                  ))}
                </div>
              </div>

              {/* Deep dive */}
              {topic.chapterUrl && (
                <div>
                  <SectionDivider label="Deep Dive" />
                  <a href={topic.chapterUrl} target="_blank" rel="noopener noreferrer"
                    className="flex items-center gap-3 p-4 mt-3 rounded-xl border border-borderSubtle bg-surface hover:bg-surfaceElevated hover:border-borderHighlight transition-all group">
                    <BookOpen className="w-5 h-5 text-textTertiary group-hover:text-accentBlue shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-[14px] font-semibold text-white">{topic.chapterName}</p>
                      <p className="text-[12px] text-textTertiary mt-0.5">Read in textbook &middot; mlsysbook.ai</p>
                    </div>
                    <ExternalLink className="w-4 h-4 text-textMuted group-hover:text-textSecondary shrink-0" />
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

// ─── Section Divider (Linear-style centered label) ─────────────

function SectionDivider({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3">
      <div className="h-px flex-1 bg-borderSubtle" />
      <span className="text-[11px] font-semibold uppercase tracking-widest text-textTertiary shrink-0">
        {label}
      </span>
      <div className="h-px flex-1 bg-borderSubtle" />
    </div>
  );
}
