"use client";

import { Suspense, useState, useEffect, useMemo } from "react";
import { Search, X, Target, Crosshair, Flame, Calendar, ChevronUp, ChevronDown, Network } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { useSearchParams, useRouter } from "next/navigation";
import {
  getAreas, getVaultStats, getAreaStyle, getAreaForTopic, searchTopics,
  type Topic,
} from "@/lib/taxonomy";
import { getTracks, searchQuestions, getTrackCount, type Question } from "@/lib/corpus";
import { track as trackAnalytics } from "@/lib/analytics";
import { Cloud, Smartphone, Cpu, CircuitBoard } from "lucide-react";
import { getAttempts, getStreakData } from "@/lib/progress";
import { FilterPill, AreaOverview, ExpandedArea, SearchResults, TopicDetail } from "@/components/vault";
import { isDailyCompleted } from "@/lib/daily";
import manifest from "@/data/vault-manifest.json";
import { ECOSYSTEM_BASE } from "@/lib/env";

function formatTrackLabel(t: string) {
  return t === "tinyml" ? "TinyML" : t.charAt(0).toUpperCase() + t.slice(1);
}

export default function HomePageWrapper() {
  return (
    <Suspense fallback={
      <div className="flex-1 flex items-center justify-center">
        <Search className="w-6 h-6 text-textTertiary animate-pulse" />
      </div>
    }>
      <HomePage />
    </Suspense>
  );
}

function HomePage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [mounted, setMounted] = useState(false);
  const [query, setQuery] = useState("");
  const [selectedTopic, setSelectedTopic] = useState<Topic | null>(null);
  const [selectedAreas, setSelectedAreas] = useState<Set<string>>(new Set());
  const [showAllAreas, setShowAllAreas] = useState(false);

  // Initialize track from URL param, sync changes back to URL
  const validTracks = getTracks();
  const initialTrack = searchParams.get("track");
  const [selectedTrack, setSelectedTrackState] = useState<string | null>(
    initialTrack && validTracks.includes(initialTrack) ? initialTrack : null
  );
  const [headerCollapsed, setHeaderCollapsed] = useState(false);
  const setSelectedTrack = (track: string | null) => {
    setSelectedTrackState(track);
    const params = new URLSearchParams(searchParams.toString());
    if (track) { params.set("track", track); } else { params.delete("track"); }
    const qs = params.toString();
    router.replace(qs ? `/?${qs}` : "/", { scroll: false });
  };
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

    // First-run redirect to /welcome. Only fires for truly brand-new
    // visitors (zero attempts logged AND no dismissal flag), and only
    // when the root URL is clean — any ?track=, ?topic=, search query,
    // etc. is treated as an intentional deep link and never redirected.
    // Runs after localStorage reads so SSR can still render the Vault
    // shell without mismatch.
    // Session guard: once we've bounced to /welcome in this tab, don't
    // bounce again — otherwise clicking "Vault" in the nav from /welcome
    // ping-pongs back. Cross-session behavior (show again next tab
    // open) is preserved because sessionStorage dies with the tab.
    if (attempts.length === 0 && searchParams.toString() === "") {
      let seen = false;
      let bouncedThisSession = false;
      try {
        seen = localStorage.getItem("staffml_firstrun_welcome") === "1";
        bouncedThisSession = sessionStorage.getItem("staffml_firstrun_bounced") === "1";
      } catch {
        seen = true; // If localStorage is blocked, assume seen so we don't trap them
      }
      if (!seen && !bouncedThisSession) {
        try { sessionStorage.setItem("staffml_firstrun_bounced", "1"); } catch { /* noop */ }
        router.replace("/welcome");
      }
    }
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

  // Filter areas by selected track — transform Topic objects so downstream
  // components (TopicCard, TopicDetail) see track-accurate counts & questions
  const filteredAreas = useMemo(() => {
    if (!selectedTrack) return areas;
    return areas
      .map(area => ({
        ...area,
        topics: area.topics
          .filter(t => t.tracks.includes(selectedTrack))
          .map(t => {
            // Rebuild questionsByLevel with only this track's questions
            const filteredByLevel: Record<string, typeof t.questionsByLevel[string]> = {};
            const filteredLevels: Record<string, number> = {};
            let count = 0;
            for (const [level, qs] of Object.entries(t.questionsByLevel)) {
              const trackQs = qs.filter(q => q.track === selectedTrack);
              if (trackQs.length > 0) {
                filteredByLevel[level] = trackQs;
                filteredLevels[level] = trackQs.length;
                count += trackQs.length;
              }
            }
            return { ...t, questionsByLevel: filteredByLevel, levels: filteredLevels, questionCount: count };
          }),
      }))
      .filter(area => area.topics.length > 0)
      .map(area => {
        // Reaggregate area-level stats from the now-filtered topics
        const areaLevels: Record<string, number> = {};
        for (const t of area.topics) {
          for (const [lv, cnt] of Object.entries(t.levels)) {
            areaLevels[lv] = (areaLevels[lv] || 0) + cnt;
          }
        }
        return {
          ...area,
          questionCount: area.topics.reduce((s, t) => s + t.questionCount, 0),
          topicCount: area.topics.length,
          levels: areaLevels,
        };
      });
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

  // Flat list of topics in the current display order. Used by the
  // drawer's j/k (and arrow-key) sweep handler to navigate to the next
  // or previous topic without closing the drawer.
  const flatTopics = useMemo<Topic[]>(
    () => displayAreas.flatMap((a) => a.topics),
    [displayAreas],
  );

  // Drawer keyboard navigation: j/ArrowDown = next, k/ArrowUp = prev.
  // Skipped when the user is typing in an input (search box, textarea)
  // so we never interfere with keystrokes. Skipped when the drawer is
  // closed. Escape handling lives inside TopicDetail.
  useEffect(() => {
    if (!selectedTopic) return;
    const handleKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) {
        return;
      }
      if (flatTopics.length === 0) return;
      const idx = flatTopics.findIndex((t) => t.id === selectedTopic.id);
      if (idx === -1) return;
      if (e.key === "j" || e.key === "ArrowDown") {
        e.preventDefault();
        const next = flatTopics[(idx + 1) % flatTopics.length];
        setSelectedTopic(next);
      } else if (e.key === "k" || e.key === "ArrowUp") {
        e.preventDefault();
        const prev = flatTopics[(idx - 1 + flatTopics.length) % flatTopics.length];
        setSelectedTopic(prev);
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [selectedTopic, flatTopics]);

  if (!mounted) {
    return <div className="flex-1" />;
  }

  return (
    <div className="flex-1 flex flex-col h-[calc(100dvh-3.5rem)] overflow-hidden">
      {/* ─── Header ─── */}
      <div className="px-6 pt-4 pb-4 border-b border-border">
        <div className="max-w-5xl mx-auto">
          <AnimatePresence initial={false}>
          {!headerCollapsed && (
          <motion.div
            key="header-content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeInOut" }}
            className="overflow-hidden"
          >
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
                <Link
                  href="/explore"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-surface border border-border text-textSecondary hover:text-textPrimary rounded-lg text-xs transition-colors"
                >
                  <Network className="w-3 h-3" /> Explorer
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
                {selectedTrack ? (
                  <>
                    <span className="font-semibold">{formatTrackLabel(selectedTrack)}</span> track:{" "}
                    {filteredAreas.reduce((s, a) => s + a.questionCount, 0).toLocaleString()} of{" "}
                    {stats.totalQuestions.toLocaleString()} questions.
                  </>
                ) : (
                  <>{stats.totalQuestions.toLocaleString()} questions across compute, memory, latency, and more.</>
                )}
              </p>

              {/* Welcome guide */}
              <div className="p-4 rounded-xl border border-accentBlue/20 bg-accentBlue/5 mb-3">
                <span className="text-[10px] font-mono text-accentBlue uppercase block mb-2.5">New here? Start with one of these:</span>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
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
                    href="/explore"
                    className="flex items-start gap-2.5 p-3 rounded-lg bg-background border border-border hover:border-accentBlue/40 transition-colors group"
                  >
                    <Network className="w-4 h-4 text-accentPurple shrink-0 mt-0.5" />
                    <div>
                      <span className="text-sm font-bold text-textPrimary group-hover:text-accentPurple transition-colors block">Vault Explorer</span>
                      <span className="text-[11px] text-textTertiary">Radial map from tracks to topics.</span>
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

          </motion.div>
          )}
          </AnimatePresence>

          {/* Filters row — track + areas + search on one compact strip */}
          <div className="flex flex-col gap-2">
            {/* Track pills */}
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="text-[10px] font-mono text-textMuted uppercase tracking-wide mr-1">Track</span>
              <FilterPill
                label="All"
                count={getTrackCount()}
                isActive={!selectedTrack}
                onClick={() => setSelectedTrack(null)}
              />
              {[
                { id: "cloud", label: "Cloud", icon: <Cloud className="w-3 h-3" /> },
                { id: "edge", label: "Edge", icon: <Cpu className="w-3 h-3" /> },
                { id: "mobile", label: "Mobile", icon: <Smartphone className="w-3 h-3" /> },
                { id: "tinyml", label: "TinyML", icon: <CircuitBoard className="w-3 h-3" /> },
              ].map(t => (
                <FilterPill
                  key={t.id}
                  label={t.label}
                  count={getTrackCount(t.id)}
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
                      setHeaderCollapsed(true);
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

          {/* Collapse/expand toggle */}
          <button
            onClick={() => setHeaderCollapsed(!headerCollapsed)}
            className="w-full flex items-center justify-center pt-1.5 -mb-1 text-textMuted hover:text-textSecondary transition-colors"
            aria-label={headerCollapsed ? "Expand header" : "Collapse header"}
          >
            {headerCollapsed ? (
              <ChevronDown className="w-4 h-4" />
            ) : (
              <ChevronUp className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {/* ─── Main ─── */}
      <div className="flex-1 flex overflow-hidden relative">
        <div className="flex-1 overflow-auto px-6 py-6">
          <div className="max-w-5xl mx-auto">
            {searchResults ? (
              <>
                <SearchResults results={searchResults} query={query}
                  selectedId={selectedTopic?.id ?? null} onSelect={(topic) => {
                    setSelectedTopic(topic);
                    setHeaderCollapsed(true);
                  }} />
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
                          <p className="text-[12px] text-textTertiary mt-1 line-clamp-2">{(() => { const t = q.question ?? q.scenario; return t.length > 150 ? `${t.slice(0, 150)}…` : t; })()}</p>
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
                selectedId={selectedTopic?.id ?? null} onSelect={(topic) => {
                  setSelectedTopic(topic);
                  setHeaderCollapsed(true);
                }} />
            ) : (
              <AreaOverview 
                areas={displayAreas} 
                onExpand={(id) => {
                  setSelectedAreas(new Set([id]));
                  setHeaderCollapsed(true);
                }} 
                onSelectTopic={(topic) => {
                  setSelectedTopic(topic);
                  setHeaderCollapsed(true);
                }}
                selectedId={selectedTopic?.id ?? null} 
              />
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

        {/* Desktop detail drawer (right-anchored slide-over).
            - Width 440px: enough room for the drill CTA + filters
              without crowding the topic grid on 13" displays.
            - Soft scrim (bg-black/30) recedes the grid so the drawer
              reads as the primary task, while still showing the
              source card (anchored via accent border on TopicCard).
            - j/k (and ArrowDown/ArrowUp) sweep between topics without
              closing the drawer — see the keyboard effect above. */}
        <AnimatePresence>
          {selectedTopic && selectedStyle && (
            <motion.div
              key="desktop-scrim"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="hidden lg:block absolute inset-0 z-40 bg-black/30"
              onClick={() => setSelectedTopic(null)}
              aria-hidden="true"
            />
          )}
        </AnimatePresence>
        <AnimatePresence>
          {selectedTopic && selectedStyle && (
            <motion.div
              key="desktop-drawer"
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ type: "spring", damping: 32, stiffness: 320 }}
              className="hidden lg:block absolute top-4 right-4 z-50 w-[440px] max-w-[92vw] h-[min(700px,calc(100%-2rem))] border border-border bg-background shadow-2xl rounded-xl overflow-hidden"
            >
              <TopicDetail topic={selectedTopic}
                areaName={selectedArea?.name || ""} style={selectedStyle}
                selectedTrack={selectedTrack}
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
                selectedTrack={selectedTrack}
                onClose={() => setSelectedTopic(null)} />
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
