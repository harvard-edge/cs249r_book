"use client";

import { useState, useEffect, useMemo } from "react";
import { Search, X, Target, Crosshair, Flame } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import {
  getAreas, getVaultStats, getAreaStyle, getAreaForTopic, searchTopics,
  type Topic,
} from "@/lib/taxonomy";
import { getAttempts, getStreakData } from "@/lib/progress";
import { FilterPill, AreaOverview, ExpandedArea, SearchResults, TopicDetail } from "@/components/vault";
import LevelExplainer from "@/components/LevelExplainer";

export default function HomePage() {
  const [mounted, setMounted] = useState(false);
  const [query, setQuery] = useState("");
  const [selectedTopic, setSelectedTopic] = useState<Topic | null>(null);
  const [expandedArea, setExpandedArea] = useState<string | null>(null);
  const [isReturning, setIsReturning] = useState(false);
  const [streakCount, setStreakCount] = useState(0);
  const [attemptCount, setAttemptCount] = useState(0);

  const stats = getVaultStats();
  const areas = getAreas();

  useEffect(() => {
    setMounted(true);
    const attempts = getAttempts();
    const streak = getStreakData();
    setAttemptCount(attempts.length);
    setStreakCount(streak.currentStreak);
    setIsReturning(attempts.length > 0);
  }, []);

  const searchResults = useMemo(() => {
    if (!query.trim()) return null;
    return searchTopics(query);
  }, [query]);

  const selectedArea = selectedTopic ? getAreaForTopic(selectedTopic.id) : null;
  const selectedStyle = selectedArea ? getAreaStyle(selectedArea.id) : null;

  if (!mounted) {
    return <div className="flex-1" />;
  }

  return (
    <div className="flex-1 flex flex-col min-h-screen">
      {/* ─── Hero ─── */}
      <div className="px-6 pt-8 pb-6 border-b border-border">
        <div className="max-w-5xl mx-auto">
          {isReturning ? (
            /* Returning user — compact row */
            <div className="flex items-center justify-between gap-4 mb-5">
              <div className="flex items-center gap-4">
                {streakCount > 0 && (
                  <div className="flex items-center gap-1.5 text-accentAmber">
                    <Flame className="w-4 h-4" />
                    <span className="text-sm font-bold font-mono">{streakCount}</span>
                    <span className="text-xs text-textTertiary">day streak</span>
                  </div>
                )}
                <span className="text-sm text-textSecondary">
                  {attemptCount} questions answered
                </span>
              </div>
              <Link
                href="/practice"
                className="inline-flex items-center gap-2 px-4 py-2 bg-accentBlue text-white font-bold rounded-lg text-sm hover:opacity-90 transition-opacity"
              >
                <Target className="w-4 h-4" /> Continue Practicing
              </Link>
            </div>
          ) : (
            /* New user — full hero */
            <div className="mb-6">
              <h1 className="text-3xl sm:text-4xl font-extrabold text-textPrimary tracking-tight mb-2">
                StaffML
              </h1>
              <p className="text-[15px] text-textSecondary mb-1">
                {stats.totalQuestions.toLocaleString()} physics-grounded ML systems interview questions.
                100% client-side. No accounts. No tracking.
              </p>
              <p className="text-[13px] text-textTertiary mb-3">
                Built on{" "}
                <a href="https://mlsysbook.ai" target="_blank" rel="noopener noreferrer"
                  className="text-accentBlue hover:underline">
                  Machine Learning Systems
                </a>
                {" "}by Prof. Vijay Janapa Reddi, Harvard University.
              </p>
              <div className="flex items-center gap-3 mb-4">
                <Link
                  href="/practice"
                  className="inline-flex items-center gap-2 px-5 py-2.5 bg-accentBlue text-white font-bold rounded-lg text-sm hover:opacity-90 transition-opacity"
                >
                  <Target className="w-4 h-4" /> Start Practicing
                </Link>
                <Link
                  href="/gauntlet"
                  className="inline-flex items-center gap-2 px-5 py-2.5 bg-surface border border-border text-textSecondary font-medium rounded-lg text-sm hover:text-textPrimary transition-colors"
                >
                  <Crosshair className="w-4 h-4" /> Mock Interview
                </Link>
              </div>
              <LevelExplainer />
            </div>
          )}

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
              className="w-full pl-12 pr-12 py-3 bg-surface border border-border rounded-xl text-[15px] font-medium text-textPrimary placeholder:text-textTertiary focus:outline-none focus:border-borderHighlight transition-colors"
            />
            {query && (
              <button onClick={() => setQuery("")}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-textTertiary hover:text-textPrimary">
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
