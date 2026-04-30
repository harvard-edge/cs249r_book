"use client";

import { useState } from "react";
import { ChevronRight, ChevronLeft, X, Target, Play } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import clsx from "clsx";
import Link from "next/link";
import type { Topic, AreaStyle } from "@/lib/taxonomy";
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

  const [prevId, setPrevId] = useState(topic.id);
  if (topic.id !== prevId) { setPrevId(topic.id); setDrillLevel(null); }

  // Build track-aware query params for drill links
  const trackParam = selectedTrack ? `&track=${selectedTrack}` : "";

  const levelQs = drillLevel ? topic.questionsByLevel[drillLevel] || [] : [];

  // Group questions by track, ordered: Cloud → Edge → Mobile → TinyML → everything else
  const TRACK_ORDER = ["cloud", "edge", "mobile", "tinyml"];
  const groupedByTrack = levelQs.reduce<Record<string, typeof levelQs>>((acc, q) => {
    const t = q.track || "other";
    (acc[t] ??= []).push(q);
    return acc;
  }, {});
  const orderedTracks = [
    ...TRACK_ORDER.filter(t => groupedByTrack[t]),
    ...Object.keys(groupedByTrack).filter(t => !TRACK_ORDER.includes(t)),
  ];

  return (
    <div className="w-full h-full flex flex-col bg-background">
      {/* Header */}
      <div className="p-5 border-b border-border"
        style={{ background: `linear-gradient(180deg, ${style.primary}08 0%, transparent 100%)` }}>
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Icon className="w-4 h-4" style={{ color: style.primary }} />
              <span className="text-[12px] font-semibold uppercase tracking-wide"
                style={{ color: style.primary }}>{areaName}</span>
            </div>
            <h2 className="text-[20px] font-bold text-textPrimary leading-tight">{topic.name}</h2>
          </div>
          <button onClick={onClose} aria-label="Close"
            className="p-2.5 -mr-1 text-textTertiary hover:text-textPrimary hover:bg-surfaceHover rounded-lg transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-scroll min-h-0 scrollbar-drawer">
        <AnimatePresence mode="wait">
          {drillLevel ? (
            <motion.div key={`level-${drillLevel}`}
              initial={{ x: 40, opacity: 0 }} animate={{ x: 0, opacity: 1 }}
              exit={{ x: 40, opacity: 0 }} transition={{ duration: 0.15 }}>

              {/* Pinned sub-header */}
              <div className="p-5 pb-3 border-b border-borderSubtle sticky top-0 bg-background z-10">
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

              {/* Question list grouped by track */}
              <div className="p-5 space-y-4">
                {orderedTracks.map((track) => (
                  <div key={track}>
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-[12px] font-semibold uppercase tracking-wide text-textTertiary">
                        {formatTrackName(track)}
                      </span>
                      <span className="text-[11px] text-textMuted">{groupedByTrack[track].length}</span>
                      <div className="flex-1 h-px bg-borderSubtle" />
                    </div>
                    <div className="space-y-2">
                      {groupedByTrack[track].map((q) => (
                        <Link key={q.id} href={`/practice?q=${q.id}`}
                          className="block p-3.5 rounded-xl border border-borderSubtle bg-surface hover:bg-surfaceElevated hover:border-borderHighlight transition-all group">
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0">
                              <p className="text-[14px] font-semibold text-textPrimary leading-snug mb-1">{q.title}</p>
                              <p className="text-[13px] text-textSecondary line-clamp-2 leading-relaxed">
                                {(() => {
                                  const text = (q.question ?? q.scenario)
                                    .replace(/^-\s*\*\*Interviewer:\*\*\s*/i, "")
                                    .replace(/^"/, "")
                                    .replace(/"$/, "")
                                    .trim();
                                  return text.length > 140 ? `${text.slice(0, 140)}…` : text;
                                })()}
                              </p>
                            </div>
                            <Play className="w-3.5 h-3.5 text-textMuted group-hover:text-textPrimary transition-colors shrink-0 mt-1" />
                          </div>
                        </Link>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          ) : (
            <motion.div key="overview"
              initial={{ x: -40, opacity: 0 }} animate={{ x: 0, opacity: 1 }}
              exit={{ x: -40, opacity: 0 }} transition={{ duration: 0.15 }}
              className="p-5 space-y-6">

              {topic.description && (
                <p className="text-[14px] text-textSecondary leading-relaxed">{topic.description}</p>
              )}

              {/* Primary CTA — placed above filters (Fitts's Law).
                  Most users want "drill everything" by default; power
                  users who want to tune by difficulty can do so below. */}
              <Link href={`/practice?topic=${topic.id}${trackParam}`}
                className="flex items-center justify-center gap-2 w-full py-3.5 font-bold rounded-xl text-[14px] transition-all hover:opacity-90 shadow-sm"
                style={{ backgroundColor: style.primary, color: "#101014" }}>
                <Target className="w-4 h-4" />
                Drill All {topic.questionCount} Questions
              </Link>

              {/* Compact stacked difficulty strip — at-a-glance mix */}
              <div>
                <div className="flex items-baseline justify-between mb-2">
                  <span className="text-[11px] font-mono text-textMuted uppercase tracking-wide">Difficulty mix</span>
                  <span className="text-[11px] text-textTertiary">{topic.questionCount} total</span>
                </div>
                <div className="flex h-2 w-full rounded-full overflow-hidden bg-surfaceHover">
                  {LEVEL_IDS.map((level) => {
                    const count = topic.levels[level] || 0;
                    if (count === 0) return null;
                    const pct = (count / topic.questionCount) * 100;
                    const levelDef = LEVEL_DEFS.find(l => l.id === level);
                    return (
                      <div key={level}
                        title={`${levelDef?.name || level}: ${count}`}
                        style={{ width: `${pct}%`, backgroundColor: levelDef?.color || style.primary }} />
                    );
                  })}
                </div>
              </div>

              {/* Drill by difficulty — secondary path */}
              <div>
                <SectionDivider label="Or drill by difficulty" />
                <div className="space-y-1.5 mt-3">
                  {LEVEL_IDS.map((level) => {
                    const count = topic.levels[level] || 0;
                    if (count === 0) return null;
                    const levelDef = LEVEL_DEFS.find(l => l.id === level);
                    return (
                      <button key={level} onClick={() => setDrillLevel(level)}
                        className="w-full flex items-center gap-3 px-4 py-2.5 rounded-xl border border-borderSubtle bg-surface hover:bg-surfaceElevated hover:border-borderHighlight transition-all group">
                        <LevelBadge level={level} />
                        <span className="flex-1 text-left text-[13px] font-medium text-textSecondary">{levelDef?.name}</span>
                        <span className="text-[13px] font-mono font-bold text-textPrimary">{count}</span>
                        <ChevronRight className="w-4 h-4 text-textMuted group-hover:text-textSecondary shrink-0 transition-colors" />
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Tracks — inline, compact (not a full section) */}
              <div className="flex items-center gap-2 flex-wrap pt-1">
                <span className="text-[11px] font-mono text-textMuted uppercase tracking-wide">Available in</span>
                {topic.tracks.map((t) => (
                  <span key={t} className={clsx(
                    "px-2.5 py-1 rounded-md text-[12px] font-medium transition-colors",
                    selectedTrack === t
                      ? "bg-accentBlue/15 border border-accentBlue text-accentBlue font-bold"
                      : "bg-surface border border-borderSubtle text-textSecondary"
                  )}>
                    {formatTrackName(t)}
                  </span>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
