"use client";

import { useState, useEffect, useMemo } from "react";
import { Search, X, Target, Crosshair, Flame, Calendar } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import {
  getAreas, getVaultStats, getAreaStyle, getAreaForTopic, searchTopics,
  type Topic,
} from "@/lib/taxonomy";
import { getTracks, searchQuestions, type Question } from "@/lib/corpus";
import { track as trackAnalytics } from "@/lib/analytics";
import { Cloud, Smartphone, Cpu, CircuitBoard } from "lucide-react";
import { getAttempts, getStreakData } from "@/lib/progress";
import { FilterPill, AreaOverview, ExpandedArea, SearchResults, TopicDetail } from "@/components/vault";
import { isDailyCompleted } from "@/lib/daily";
import manifest from "@/data/vault-manifest.json";
import { ECOSYSTEM_BASE } from "@/lib/env";

export default function HomePage() {
  const [mounted, setMounted] = useState(false);
  const [query, setQuery] = useState("");
  const [selectedTopic, setSelectedTopic] = useState<Topic | null>(null);
  const [selectedAreas, setSelectedAreas] = useState<Set<string>>(new Set());
  const [showAllAreas, setShowAllAreas] = useState(false);
  const [selectedTrack, setSelectedTrack] = useState<string | null>(null);
  const [isReturning, setIsReturning] = useState(false);
  const [streakCount, setStreakCount] = useState(0);
  const [attemptCount, setAttemptCount] = useState(0);
  const [dailyDone, setDailyDone] = useState(false);

  const stats = getVaultStats();
  const areas = getAreas();

  useEffect(() => {
    setMounted(true);
    const attempts = getAttempts();
    const streak = getStreakData();
    setAttemptCount(attempts.length);
    setStreakCount(streak.currentStreak);
    setIsReturning(attempts.length > 0);
    setDailyDone(isDailyCompleted());
  }, []);

  const searchResults = useMemo(() => {
    if (!query.trim()) return null;
    let topicResults = searchTopics(query);
    if (selectedTrack) topicResults = topicResults.filter(t => t.tracks.includes(selectedTrack));
    return topicResults;
  }, [query, selectedTrack]);

  // Full-text search across question content (scenarios, answers, napkin math)
  const questionSearchResults = useMemo(() => {
    if (!query.trim() || query.trim().length < 2) return [];
    let results = searchQuestions(query, 30);
    if (selectedTrack) results = results.filter(q => q.track === selectedTrack);
    return results;
  }, [query, selectedTrack]);

  // Track search queries (debounced — only fires when query settles for 1s)
  useEffect(() => {
    if (!query.trim() || query.trim().length < 2) return;
    const timer = setTimeout(() => {
      trackAnalytics({
        type: 'search_query',
        query: query.trim().toLowerCase().slice(0, 50),
        topicResults: searchResults?.length ?? 0,
        questionResults: questionSearchResults.length,
      });
    }, 1000);
    return () => clearTimeout(timer);
  }, [query, searchResults, questionSearchResults]);

  // Filter areas by selected track
  const filteredAreas = useMemo(() => {
    if (!selectedTrack) return areas;
    return areas
      .map(area => ({
        ...area,
        topics: area.topics.filter(t => t.tracks.includes(selectedTrack)),
      }))
      .filter(area => area.topics.length > 0)
      .map(area => ({
        ...area,
        questionCount: area.topics.reduce((s, t) => s + t.questionCount, 0),
        topicCount: area.topics.length,
      }));
  }, [areas, selectedTrack]);

  // Areas to show in pills: top 8 unless "show all" is toggled
  const MAX_VISIBLE_AREAS = 8;
  const visibleAreas = showAllAreas ? filteredAreas : filteredAreas.slice(0, MAX_VISIBLE_AREAS);
  const hiddenCount = filteredAreas.length - MAX_VISIBLE_AREAS;

  // Areas to display in the main content (filtered by multi-select)
  const displayAreas = useMemo(() => {
    if (selectedAreas.size === 0) return filteredAreas;
    return filteredAreas.filter(a => selectedAreas.has(a.id));
  }, [filteredAreas, selectedAreas]);

  // For the single-area drill-down when exactly one is selected
  const singleExpandedArea = selectedAreas.size === 1
    ? filteredAreas.find(a => a.id === Array.from(selectedAreas)[0]) ?? null
    : null;

  const selectedArea = selectedTopic ? getAreaForTopic(selectedTopic.id) : null;
  const selectedStyle = selectedArea ? getAreaStyle(selectedArea.id) : null;

  if (!mounted) {
    return <div className="flex-1" />;
  }

  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-3.5rem)] overflow-hidden">
      {/* ─── Header ─── */}
      <div className="px-6 pt-4 pb-4 border-b border-border">
        <div className="max-w-5xl mx-auto">
          {isReturning ? (
            /* Returning user — single compact row */
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-4">
                {streakCount > 0 && (
                  <div className="flex items-center gap-1.5 text-accentAmber">
                    <Flame className="w-3.5 h-3.5" />
                    <span className="text-sm font-bold font-mono">{streakCount}</span>
                    <span className="text-xs text-textTertiary">day streak</span>
                  </div>
                )}
                <span className="text-sm text-textSecondary">
                  {attemptCount} answered
                </span>
              </div>
              <div className="flex items-center gap-2">
                {!dailyDone && (
                  <Link href="/practice?daily=1"
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-accentBlue text-white font-bold rounded-lg text-xs hover:opacity-90 transition-opacity"
                  >
                    <Calendar className="w-3 h-3" /> Daily Challenge
                  </Link>
                )}
                <Link
                  href="/practice"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-surface border border-border text-textSecondary hover:text-textPrimary rounded-lg text-xs transition-colors"
                >
                  <Target className="w-3 h-3" /> Browse
                </Link>
              </div>
            </div>
          ) : (
            /* New user — welcome with guidance */
            <div className="mb-4">
              <h1 className="text-2xl font-extrabold text-textPrimary tracking-tight mb-1">
                StaffML
              </h1>
              <p className="text-[13px] text-textSecondary mb-3">
                Free, open-source interview prep for ML systems engineers.{" "}
                {stats.totalQuestions.toLocaleString()} questions across compute, memory, latency, and more.
              </p>

              {/* Welcome guide */}
              <div className="p-4 rounded-xl border border-accentBlue/20 bg-accentBlue/5 mb-3">
                <span className="text-[10px] font-mono text-accentBlue uppercase block mb-2.5">New here? Start with one of these:</span>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                  <Link
                    href="/practice?daily=1"
                    className="flex items-start gap-2.5 p-3 rounded-lg bg-background border border-border hover:border-accentBlue/40 transition-colors group"
                  >
                    <Calendar className="w-4 h-4 text-accentBlue shrink-0 mt-0.5" />
                    <div>
                      <span className="text-sm font-bold text-textPrimary group-hover:text-accentBlue transition-colors block">Daily Challenge</span>
                      <span className="text-[11px] text-textTertiary">3 questions, same for everyone. Takes 5 min.</span>
                    </div>
                  </Link>
                  <Link
                    href="/practice?level=L1"
                    className="flex items-start gap-2.5 p-3 rounded-lg bg-background border border-border hover:border-accentBlue/40 transition-colors group"
                  >
                    <Target className="w-4 h-4 text-accentGreen shrink-0 mt-0.5" />
                    <div>
                      <span className="text-sm font-bold text-textPrimary group-hover:text-accentGreen transition-colors block">Easy Mode</span>
                      <span className="text-[11px] text-textTertiary">L1 recall questions to warm up.</span>
                    </div>
                  </Link>
                  <Link
                    href="/gauntlet"
                    className="flex items-start gap-2.5 p-3 rounded-lg bg-background border border-border hover:border-accentBlue/40 transition-colors group"
                  >
                    <Crosshair className="w-4 h-4 text-accentRed shrink-0 mt-0.5" />
                    <div>
                      <span className="text-sm font-bold text-textPrimary group-hover:text-accentRed transition-colors block">Mock Interview</span>
                      <span className="text-[11px] text-textTertiary">Timed session with scoring.</span>
                    </div>
                  </Link>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <span className="text-[11px] text-textTertiary">
                  100% client-side &middot; no accounts &middot; your data stays in your browser
                </span>
              </div>
            </div>
          )}

          {/* MLSysBook ecosystem — staffml.ai is the app; book site carries curriculum context */}
          <p className="text-[11px] text-textSecondary leading-relaxed mt-1 mb-0 max-w-3xl">
            Part of the{" "}
            <a
              href={ECOSYSTEM_BASE}
              target="_blank"
              rel="noopener noreferrer"
              className="text-accentBlue hover:underline font-medium"
            >
              Machine Learning Systems
            </a>{" "}
            ecosystem (textbook, labs, tools). This app lives on{" "}
            <span className="font-mono text-textPrimary">staffml.ai</span>
            ; for how StaffML fits the book curriculum, see{" "}
            <a
              href={`${ECOSYSTEM_BASE}/staffml/`}
              target="_blank"
              rel="noopener noreferrer"
              className="font-mono text-accentBlue hover:underline"
            >
              mlsysbook.ai/staffml
            </a>
            .
          </p>

          {/* Filters row — track + areas + search on one compact strip */}
          <div className="flex flex-col gap-2">
            {/* Track pills */}
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="text-[10px] font-mono text-textMuted uppercase tracking-wide mr-1">Track</span>
              <FilterPill label="All" isActive={!selectedTrack} onClick={() => setSelectedTrack(null)} />
              {[
                { id: "cloud", label: "Cloud", icon: <Cloud className="w-3 h-3" /> },
                { id: "edge", label: "Edge", icon: <Cpu className="w-3 h-3" /> },
                { id: "mobile", label: "Mobile", icon: <Smartphone className="w-3 h-3" /> },
                { id: "tinyml", label: "TinyML", icon: <CircuitBoard className="w-3 h-3" /> },
              ].map(t => (
                <FilterPill
                  key={t.id}
                  label={t.label}
                  isActive={selectedTrack === t.id}
                  icon={t.icon}
                  onClick={() => setSelectedTrack(selectedTrack === t.id ? null : t.id)}
                />
              ))}
            </div>

            {/* Area pills — scroll on mobile, wrap on desktop */}
            <div className="flex items-center gap-1.5 overflow-x-auto md:flex-wrap scrollbar-hide">
              <span className="text-[10px] font-mono text-textMuted uppercase tracking-wide mr-1">Area</span>
              <FilterPill
                label="All"
                isActive={selectedAreas.size === 0}
                onClick={() => { setSelectedAreas(new Set()); setShowAllAreas(false); }}
              />
              {visibleAreas.map((area) => {
                const style = getAreaStyle(area.id);
                const Icon = style.icon;
                return (
                  <FilterPill
                    key={area.id}
                    label={area.name}
                    count={area.questionCount}
                    isActive={selectedAreas.has(area.id)}
                    color={style.primary}
                    icon={<Icon className="w-3 h-3" />}
                    onClick={() => {
                      setSelectedAreas(prev => {
                        const next = new Set(prev);
                        if (next.has(area.id)) next.delete(area.id);
                        else next.add(area.id);
                        return next;
                      });
                    }}
                  />
                );
              })}
              {hiddenCount > 0 && (
                <FilterPill
                  label={showAllAreas ? "Show less" : `+${hiddenCount} more`}
                  isActive={false}
                  onClick={() => setShowAllAreas(!showAllAreas)}
                />
              )}
            </div>

            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-textMuted" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search topics — KV cache, roofline, quantization..."
                aria-label="Search topics"
                className="w-full pl-10 pr-10 py-2 bg-surface border border-border rounded-lg text-[13px] font-medium text-textPrimary placeholder:text-textTertiary focus:outline-none focus:border-borderHighlight transition-colors"
              />
              {query && (
                <button onClick={() => setQuery("")}
                  aria-label="Clear search"
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-textTertiary hover:text-textPrimary">
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* ─── Main ─── */}
      <div className="flex-1 flex overflow-hidden">
        <div className="flex-1 overflow-auto px-6 py-6">
          <div className="max-w-5xl mx-auto">
            {searchResults ? (
              <>
                <SearchResults results={searchResults} query={query}
                  selectedId={selectedTopic?.id ?? null} onSelect={setSelectedTopic} />
                {/* Full-text question matches */}
                {questionSearchResults.length > 0 && (
                  <div className="mt-8">
                    <p className="text-[14px] text-textSecondary mb-4">
                      {questionSearchResults.length} question{questionSearchResults.length !== 1 ? 's' : ''} mentioning &ldquo;{query}&rdquo;
                    </p>
                    <div className="space-y-2">
                      {questionSearchResults.map(q => (
                        <Link
                          key={q.id}
                          href={`/practice?q=${q.id}`}
                          className="block p-3 rounded-lg border border-borderSubtle bg-surface/50 hover:border-borderHighlight hover:bg-surfaceHover transition-all group"
                        >
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-[10px] font-mono text-textTertiary uppercase px-1.5 py-0.5 rounded border border-border bg-background">{q.level}</span>
                            <span className="text-[10px] font-mono text-textTertiary capitalize">{q.competency_area}</span>
                            <span className="text-[10px] font-mono text-textMuted">{q.track}</span>
                          </div>
                          <p className="text-sm font-medium text-textPrimary group-hover:text-accentBlue transition-colors">{q.title}</p>
                          <p className="text-[12px] text-textTertiary mt-1 line-clamp-2">{q.scenario.slice(0, 150)}{q.scenario.length > 150 ? '...' : ''}</p>
                        </Link>
                      ))}
                    </div>
                  </div>
                )}
                {searchResults.length === 0 && questionSearchResults.length === 0 && (
                  <p className="text-sm text-textTertiary py-8 text-center">No topics or questions match &ldquo;{query}&rdquo;</p>
                )}
              </>
            ) : singleExpandedArea ? (
              <ExpandedArea area={singleExpandedArea}
                selectedId={selectedTopic?.id ?? null} onSelect={setSelectedTopic} />
            ) : (
              <AreaOverview areas={displayAreas} onExpand={(id) => setSelectedAreas(new Set([id]))} onSelectTopic={setSelectedTopic}
                selectedId={selectedTopic?.id ?? null} />
            )}

            {/* Footer */}
            <div className="mt-12 pt-6 border-t border-borderSubtle text-center pb-8">
              <p className="text-[12px] text-textMuted">
                Built at{" "}
                <a href="https://mlsysbook.ai" target="_blank" rel="noopener noreferrer" className="hover:text-textTertiary transition-colors">Harvard University</a>
                {" "}&middot;{" "}
                <a href="https://github.com/harvard-edge/cs249r_book" target="_blank" rel="noopener noreferrer" className="hover:text-textTertiary transition-colors">Open Source</a>
                {" "}&middot;{" "}
                <Link href="/about" className="hover:text-textTertiary transition-colors">About</Link>
                {" "}&middot;{" "}
                <span className="font-mono">v{manifest.version}</span>
              </p>
            </div>
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
              className="shrink-0 overflow-hidden border-l border-border hidden lg:block h-full"
            >
              <TopicDetail topic={selectedTopic}
                areaName={selectedArea?.name || ""} style={selectedStyle}
                onClose={() => setSelectedTopic(null)} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Mobile detail sheet + backdrop */}
      <AnimatePresence>
        {selectedTopic && selectedStyle && (
          <>
            {/* Backdrop — tap to close */}
            <motion.div
              key="backdrop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-40 bg-black/40 lg:hidden"
              onClick={() => setSelectedTopic(null)}
            />
            <motion.div
              key="sheet"
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
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
