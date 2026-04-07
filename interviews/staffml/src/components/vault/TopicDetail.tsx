"use client";

import { useState, useEffect, useRef } from "react";
import { ChevronRight, ChevronLeft, X, BookOpen, ExternalLink, FileText, Target, Play } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import clsx from "clsx";
import Link from "next/link";
import type { Topic, AreaStyle } from "@/lib/taxonomy";
import { classifyRef } from "@/lib/refs";
import { LEVELS as LEVEL_DEFS } from "@/lib/levels";
import LevelBadge from "@/components/LevelBadge";
import SectionDivider from "./SectionDivider";

const LEVEL_IDS = LEVEL_DEFS.map(l => l.id);

function formatTrackName(t: string) {
  return t === "tinyml" ? "TinyML" : t.charAt(0).toUpperCase() + t.slice(1);
}

export default function TopicDetail({ topic, areaName, style, onClose, selectedTrack }: {
  topic: Topic; areaName: string; style: AreaStyle; onClose: () => void; selectedTrack?: string | null;
}) {
  const [drillLevel, setDrillLevel] = useState<string | null>(null);
  const Icon = style.icon;
  const closeBtnRef = useRef<HTMLButtonElement>(null);
  const headingId = `topic-detail-${topic.id}`;

  const [prevId, setPrevId] = useState(topic.id);
  if (topic.id !== prevId) { setPrevId(topic.id); setDrillLevel(null); }

  // Build track-aware query params for drill links
  const trackParam = selectedTrack ? `&track=${selectedTrack}` : "";

  const levelQs = drillLevel ? topic.questionsByLevel[drillLevel] || [] : [];

  // ─── A11y: Escape to close, focus management ─────────────
  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    closeBtnRef.current?.focus();

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown);

    return () => {
      window.removeEventListener("keydown", onKeyDown);
      // Restore focus to whatever invoked the drawer
      previouslyFocused?.focus?.();
    };
    // onClose is stable from parent; we intentionally only wire this once per mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby={headingId}
      className="w-full h-full flex flex-col bg-background"
    >
      {/* Header */}
      <div className="p-6 border-b border-border"
        style={{ background: `linear-gradient(180deg, ${style.primary}08 0%, transparent 100%)` }}>
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Icon className="w-4 h-4" style={{ color: style.primary }} />
              <span className="text-[12px] font-semibold uppercase tracking-wide"
                style={{ color: style.primary }}>{areaName}</span>
            </div>
            <h2 id={headingId} className="text-[20px] font-bold text-textPrimary leading-tight">{topic.name}</h2>
          </div>
          <button ref={closeBtnRef} onClick={onClose} aria-label="Close topic detail"
            className="p-2.5 -mr-1 text-textTertiary hover:text-textPrimary hover:bg-surfaceHover rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-accentBlue/50">
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto min-h-0">
        <AnimatePresence mode="wait">
          {drillLevel ? (
            <motion.div key={`level-${drillLevel}`}
              initial={{ x: 40, opacity: 0 }} animate={{ x: 0, opacity: 1 }}
              exit={{ x: 40, opacity: 0 }} transition={{ duration: 0.15 }}>

              {/* Pinned sub-header */}
              <div className="p-6 pb-3 border-b border-borderSubtle sticky top-0 bg-background z-10">
                <button onClick={() => setDrillLevel(null)}
                  className="flex items-center gap-1.5 text-[13px] font-medium text-textSecondary hover:text-textPrimary mb-4 transition-colors">
                  <ChevronLeft className="w-4 h-4" /> Back to overview
                </button>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <LevelBadge level={drillLevel} size="md" />
                    <p className="text-[13px] text-textSecondary">
                      {levelQs.length} question{levelQs.length !== 1 ? "s" : ""}
                    </p>
                  </div>
                  <Link href={`/practice?topic=${topic.id}&level=${drillLevel}${trackParam}`}
                    className="flex items-center gap-1.5 px-4 py-2 text-[13px] font-bold rounded-lg transition-all"
                    style={{ backgroundColor: style.primary, color: "#101014" }}>
                    <Target className="w-4 h-4" /> Drill
                  </Link>
                </div>
              </div>

              {/* Question list */}
              <div className="p-6 space-y-2">
                {levelQs.map((q) => (
                  <Link key={q.id} href={`/practice?q=${q.id}`}
                    className="block p-3.5 rounded-xl border border-borderSubtle bg-surface hover:bg-surfaceElevated hover:border-borderHighlight transition-all group">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="text-[14px] font-semibold text-textPrimary leading-snug mb-1">{q.title}</p>
                        <p className="text-[13px] text-textSecondary line-clamp-2 leading-relaxed">
                          {q.scenario.replace(/^-\s*\*\*Interviewer:\*\*\s*/i, "").replace(/^"/, "").replace(/"$/, "").slice(0, 140)}...
                        </p>
                      </div>
                      <div className="flex items-center gap-2 shrink-0 mt-0.5">
                        <span className="text-[12px] text-textTertiary capitalize font-medium">{q.track}</span>
                        <Play className="w-3.5 h-3.5 text-textMuted group-hover:text-textPrimary transition-colors" />
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            </motion.div>
          ) : (
            <motion.div key="overview"
              initial={{ x: -40, opacity: 0 }} animate={{ x: 0, opacity: 1 }}
              exit={{ x: -40, opacity: 0 }} transition={{ duration: 0.15 }}
              className="p-6 space-y-6">

              {topic.description && (
                <p className="text-[14px] text-textSecondary leading-relaxed">{topic.description}</p>
              )}

              {/* Difficulty levels */}
              <div>
                <SectionDivider label="Difficulty Levels" />
                <div className="space-y-1.5 mt-3">
                  {LEVEL_IDS.map((level) => {
                    const count = topic.levels[level] || 0;
                    if (count === 0) return null;
                    const pct = (count / topic.questionCount) * 100;
                    const levelDef = LEVEL_DEFS.find(l => l.id === level);
                    return (
                      <button key={level} onClick={() => setDrillLevel(level)}
                        className="w-full flex items-center gap-3 px-4 py-3 rounded-xl border border-borderSubtle bg-surface hover:bg-surfaceElevated hover:border-borderHighlight transition-all group">
                        <LevelBadge level={level} />
                        <div className="flex-1">
                          <div className="flex items-center justify-between mb-1.5">
                            <span className="text-[13px] font-medium text-textSecondary">{levelDef?.name}</span>
                            <span className="text-[14px] font-mono font-bold text-textPrimary">{count}</span>
                          </div>
                          <div className="h-1.5 bg-surfaceHover rounded-full overflow-hidden">
                            <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: levelDef?.color || style.primary }} />
                          </div>
                        </div>
                        <ChevronRight className="w-4 h-4 text-textMuted group-hover:text-textSecondary shrink-0 transition-colors" />
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Deep dive — show learning resource before drill.
                  Label is derived from the URL so we don't lie about destination.
                  Book-source links currently show a "may be unavailable" warning
                  pending the mlsysbook.ai chapter-route deploy fix. */}
              {topic.chapterUrl && (() => {
                const refInfo = classifyRef(topic.chapterUrl);
                const Icon = refInfo.isBook ? BookOpen : refInfo.source === "arxiv" || refInfo.source === "paper" ? FileText : ExternalLink;
                return (
                  <div>
                    <SectionDivider label="Learn First" />
                    <a href={topic.chapterUrl} target="_blank" rel="noopener noreferrer"
                      title={refInfo.mayBeUnavailable ? "This link may be temporarily unavailable while the book site is being redeployed." : refInfo.label}
                      className="flex items-center gap-3 p-4 mt-3 rounded-xl border border-borderSubtle bg-surface hover:bg-surfaceElevated hover:border-borderHighlight transition-all group">
                      <Icon className="w-5 h-5 text-textTertiary group-hover:text-accentBlue shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="text-[14px] font-semibold text-textPrimary">{topic.name}</p>
                        <p className="text-[12px] text-textTertiary mt-0.5 flex items-center gap-1.5">
                          {refInfo.label}
                          {refInfo.mayBeUnavailable && (
                            <span className="text-accentAmber text-[10px]" aria-label="May be unavailable">⚠ may be down</span>
                          )}
                        </p>
                      </div>
                      <ExternalLink className="w-4 h-4 text-textMuted group-hover:text-textSecondary shrink-0" />
                    </a>
                  </div>
                );
              })()}

              {/* Drill all CTA */}
              <Link href={`/practice?topic=${topic.id}${trackParam}`}
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
                    <span key={t} className={clsx(
                      "px-3 py-1.5 rounded-lg text-[13px] font-medium transition-colors",
                      selectedTrack === t
                        ? "bg-accentBlue/15 border-2 border-accentBlue text-accentBlue font-bold"
                        : "bg-surface border border-borderSubtle text-textPrimary"
                    )}>
                      {formatTrackName(t)}
                    </span>
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
