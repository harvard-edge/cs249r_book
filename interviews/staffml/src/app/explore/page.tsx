"use client";

import { useMemo, useState, type ReactNode } from "react";
import Link from "next/link";
import clsx from "clsx";
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Layers,
  LocateFixed,
  Route,
  Search,
  Sparkles,
  X,
} from "lucide-react";
import { getQuestions, getTracks, type Question } from "@/lib/corpus";
import { getAreas, getAreaStyle } from "@/lib/taxonomy";
import { LEVELS } from "@/lib/levels";

const SIZE = 620;
const CX = SIZE / 2;
const CY = SIZE / 2;
const TAU = Math.PI * 2;

type Focus =
  | { kind: "root" }
  | { kind: "track"; track: string }
  | { kind: "area"; track: string; area: string }
  | { kind: "topic"; track: string; area: string; topic: string };

interface Segment {
  id: string;
  label: string;
  subtitle: string;
  count: number;
  color: string;
  focus: Focus;
}

interface LevelBucket {
  level: string;
  count: number;
  questions: Question[];
}

function formatTrack(track: string) {
  return track === "tinyml" ? "TinyML" : track.charAt(0).toUpperCase() + track.slice(1);
}

function titleCase(value: string) {
  return value
    .replace(/-/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function questionMatches(question: Question, query: string): boolean {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return true;
  const haystack = [
    question.title,
    question.question,
    question.topic,
    question.zone,
    question.competency_area,
    question.track,
    question.level,
  ].join(" ").toLowerCase();
  return normalized.split(/\s+/).every((term) => haystack.includes(term));
}

function polar(cx: number, cy: number, radius: number, angle: number) {
  return {
    x: cx + radius * Math.cos(angle - Math.PI / 2),
    y: cy + radius * Math.sin(angle - Math.PI / 2),
  };
}

function ringPath(inner: number, outer: number, start: number, end: number) {
  const gap = Math.min(0.012, Math.max(0, (end - start) / 8));
  const a0 = start + gap;
  const a1 = end - gap;
  const large = a1 - a0 > Math.PI ? 1 : 0;
  const outerStart = polar(CX, CY, outer, a0);
  const outerEnd = polar(CX, CY, outer, a1);
  const innerEnd = polar(CX, CY, inner, a1);
  const innerStart = polar(CX, CY, inner, a0);
  return [
    `M ${outerStart.x} ${outerStart.y}`,
    `A ${outer} ${outer} 0 ${large} 1 ${outerEnd.x} ${outerEnd.y}`,
    `L ${innerEnd.x} ${innerEnd.y}`,
    `A ${inner} ${inner} 0 ${large} 0 ${innerStart.x} ${innerStart.y}`,
    "Z",
  ].join(" ");
}

function midpoint(inner: number, outer: number, start: number, end: number) {
  return polar(CX, CY, (inner + outer) / 2, (start + end) / 2);
}

function byCountThenName(a: Segment, b: Segment) {
  return b.count - a.count || a.label.localeCompare(b.label);
}

function levelIndex(level: string) {
  const idx = LEVELS.findIndex((item) => item.id === level);
  return idx === -1 ? 99 : idx;
}

function buildLevelBuckets(questions: Question[]): LevelBucket[] {
  const buckets = new Map<string, Question[]>();
  for (const question of questions) {
    const bucket = buckets.get(question.level) ?? [];
    bucket.push(question);
    buckets.set(question.level, bucket);
  }
  return Array.from(buckets.entries())
    .map(([level, qs]) => ({
      level,
      count: qs.length,
      questions: qs.sort((a, b) => a.title.localeCompare(b.title)),
    }))
    .sort((a, b) => levelIndex(a.level) - levelIndex(b.level));
}

function focusLabel(focus: Focus) {
  if (focus.kind === "root") return "StaffML Vault";
  if (focus.kind === "track") return formatTrack(focus.track);
  if (focus.kind === "area") return titleCase(focus.area);
  return titleCase(focus.topic);
}

function focusParent(focus: Focus): Focus | null {
  if (focus.kind === "root") return null;
  if (focus.kind === "track") return { kind: "root" };
  if (focus.kind === "area") return { kind: "track", track: focus.track };
  return { kind: "area", track: focus.track, area: focus.area };
}

function focusBreadcrumb(focus: Focus) {
  const items: Array<{ label: string; focus: Focus }> = [{ label: "Vault", focus: { kind: "root" } }];
  if (focus.kind === "track") {
    items.push({ label: formatTrack(focus.track), focus });
  } else if (focus.kind === "area") {
    items.push({ label: formatTrack(focus.track), focus: { kind: "track", track: focus.track } });
    items.push({ label: titleCase(focus.area), focus });
  } else if (focus.kind === "topic") {
    items.push({ label: formatTrack(focus.track), focus: { kind: "track", track: focus.track } });
    items.push({ label: titleCase(focus.area), focus: { kind: "area", track: focus.track, area: focus.area } });
    items.push({ label: titleCase(focus.topic), focus });
  }
  return items;
}

export default function ExplorePage() {
  const [focus, setFocus] = useState<Focus>({ kind: "root" });
  const [query, setQuery] = useState("");
  const [selectedLevel, setSelectedLevel] = useState("all");
  const [selectedQuestionId, setSelectedQuestionId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  const questions = useMemo(() => getQuestions(), []);
  const tracks = useMemo(() => getTracks(), []);
  const areas = useMemo(() => getAreas(), []);
  const areaById = useMemo(() => new Map(areas.map((area) => [area.id, area])), [areas]);

  const filteredQuestions = useMemo(() => {
    return questions.filter((question) => {
      if (selectedLevel !== "all" && question.level !== selectedLevel) return false;
      if (!questionMatches(question, query)) return false;
      return true;
    });
  }, [questions, query, selectedLevel]);

  const focusQuestions = useMemo(() => {
    return filteredQuestions.filter((question) => {
      if (focus.kind === "root") return true;
      if (question.track !== focus.track) return false;
      if (focus.kind === "track") return true;
      if (question.competency_area !== focus.area) return false;
      if (focus.kind === "area") return true;
      return question.topic === focus.topic;
    });
  }, [filteredQuestions, focus]);

  const segments = useMemo<Segment[]>(() => {
    if (focus.kind === "root") {
      return tracks
        .map((track) => {
          const count = filteredQuestions.filter((question) => question.track === track).length;
          return {
            id: track,
            label: formatTrack(track),
            subtitle: "track",
            count,
            color: trackColor(track),
            focus: { kind: "track", track } as Focus,
          };
        })
        .filter((segment) => segment.count > 0)
        .sort(byCountThenName);
    }

    if (focus.kind === "track") {
      const areaIds = new Set(
        filteredQuestions
          .filter((question) => question.track === focus.track)
          .map((question) => question.competency_area),
      );
      return Array.from(areaIds)
        .map((areaId) => {
          const count = filteredQuestions.filter(
            (question) => question.track === focus.track && question.competency_area === areaId,
          ).length;
          const style = getAreaStyle(areaId);
          return {
            id: areaId,
            label: areaById.get(areaId)?.name ?? titleCase(areaId),
            subtitle: "area",
            count,
            color: style.primary,
            focus: { kind: "area", track: focus.track, area: areaId } as Focus,
          };
        })
        .filter((segment) => segment.count > 0)
        .sort(byCountThenName);
    }

    const track = focus.track;
    const area = focus.area;
    const topicIds = new Set(
      filteredQuestions
        .filter((question) => question.track === track && question.competency_area === area)
        .map((question) => question.topic),
    );
    const areaColor = getAreaStyle(area).primary;
    return Array.from(topicIds)
      .map((topic) => {
        const count = filteredQuestions.filter(
          (question) =>
            question.track === track &&
            question.competency_area === area &&
            question.topic === topic,
        ).length;
        return {
          id: topic,
          label: titleCase(topic),
          subtitle: "topic",
          count,
          color: areaColor,
          focus: { kind: "topic", track, area, topic } as Focus,
        };
      })
      .filter((segment) => segment.count > 0)
      .sort(byCountThenName)
      .slice(0, 36);
  }, [areaById, filteredQuestions, focus, tracks]);

  const levelBuckets = useMemo(() => buildLevelBuckets(focusQuestions), [focusQuestions]);
  const selectedQuestion = selectedQuestionId
    ? questions.find((question) => question.id === selectedQuestionId) ?? null
    : null;
  const visibleQuestions = useMemo(() => {
    return [...focusQuestions]
      .sort((a, b) => {
        const chainDelta = (b.chain_ids?.length ?? 0) - (a.chain_ids?.length ?? 0);
        if (chainDelta !== 0) return chainDelta;
        const levelDelta = levelIndex(b.level) - levelIndex(a.level);
        if (levelDelta !== 0) return levelDelta;
        return a.title.localeCompare(b.title);
      })
      .slice(0, 14);
  }, [focusQuestions]);

  const total = Math.max(1, segments.reduce((sum, segment) => sum + segment.count, 0));
  let cursor = 0;

  const reset = () => {
    setFocus({ kind: "root" });
    setQuery("");
    setSelectedLevel("all");
    setSelectedQuestionId(null);
  };

  const goToFocus = (next: Focus) => {
    setFocus(next);
    setSelectedQuestionId(null);
  };

  return (
    <div className="bg-background overflow-auto lg:overflow-hidden lg:h-[calc(100dvh-14.5rem)] lg:min-h-[620px]">
      <div className="h-full flex flex-col">
        <header className="border-b border-border bg-background/95 px-4 sm:px-6 py-3">
          <div className="max-w-7xl mx-auto">
            <Link
              href="/"
              className="inline-flex items-center gap-1.5 text-sm text-textTertiary hover:text-textSecondary transition-colors mb-2"
            >
              <ArrowLeft className="w-3.5 h-3.5" /> Back to Vault
            </Link>
            <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-4">
              <div>
                <div className="inline-flex items-center gap-1.5 text-[10px] uppercase tracking-[0.16em] font-semibold text-accentBlue px-2.5 py-1 rounded-full border border-accentBlue/30 bg-accentBlue/5 mb-2">
                  <Sparkles className="w-3 h-3" />
                  Progressive explorer preview
                </div>
                <h1 className="text-2xl sm:text-3xl font-extrabold text-textPrimary tracking-tight">
                  Explore the StaffML Vault
                </h1>
                <p className="text-sm text-textSecondary mt-1 max-w-2xl leading-relaxed">
                  Drill from track to area to topic, then choose a question path. The radial view
                  shows where the corpus is dense; the side panel keeps the exact questions readable.
                </p>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center shrink-0">
                <Metric label="showing" value={focusQuestions.length.toLocaleString()} />
                <Metric label="total" value={questions.length.toLocaleString()} />
                <Metric label="chains" value={questions.filter((q) => q.chain_ids?.length).length.toLocaleString()} />
              </div>
            </div>
          </div>
        </header>

        <section className="border-b border-borderSubtle bg-surface/40 px-4 sm:px-6 py-2.5">
          <div className="max-w-7xl mx-auto flex flex-col xl:flex-row gap-3 xl:items-center">
            <div className="relative flex-1 min-w-[220px]">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-textMuted" />
              <input
                value={query}
                onChange={(event) => {
                  setQuery(event.target.value);
                  setSelectedQuestionId(null);
                }}
                placeholder="Search: kv cache, quantization, drift, tensor parallel..."
                className="w-full pl-9 pr-9 py-2 rounded-lg border border-border bg-background text-sm text-textPrimary placeholder:text-textTertiary focus:outline-none focus:border-borderHighlight"
              />
              {query && (
                <button
                  type="button"
                  onClick={() => setQuery("")}
                  aria-label="Clear search"
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-textTertiary hover:text-textPrimary"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            <FilterSelect label="Level" value={selectedLevel} onChange={setSelectedLevel}>
              <option value="all">All levels</option>
              {LEVELS.map((level) => (
                <option key={level.id} value={level.id}>{level.id}</option>
              ))}
            </FilterSelect>

            <button
              type="button"
              onClick={reset}
              className="inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg border border-border bg-background text-sm text-textSecondary hover:text-textPrimary hover:bg-surface transition-colors"
            >
              <LocateFixed className="w-4 h-4" />
              Reset
            </button>
          </div>
        </section>

        <div className="flex-1 min-h-0 flex flex-col lg:flex-row overflow-hidden">
          <main className="relative flex-1 min-h-[560px] lg:min-h-0 overflow-auto lg:overflow-hidden p-4 sm:p-6">
            <div className="max-w-5xl mx-auto h-full flex flex-col">
              <Breadcrumb items={focusBreadcrumb(focus)} active={focus} onSelect={goToFocus} />

              <div className="flex-1 min-h-[420px] grid place-items-center">
                <svg
                  viewBox={`0 0 ${SIZE} ${SIZE}`}
                  className="w-full max-w-[540px] aspect-square overflow-visible"
                  role="img"
                  aria-label="Progressive radial explorer for StaffML questions"
                >
                  <defs>
                    <filter id="segmentGlow" x="-20%" y="-20%" width="140%" height="140%">
                      <feGaussianBlur stdDeviation="3" result="blur" />
                      <feMerge>
                        <feMergeNode in="blur" />
                        <feMergeNode in="SourceGraphic" />
                      </feMerge>
                    </filter>
                  </defs>

                  <circle cx={CX} cy={CY} r="108" className="fill-surface stroke-border" strokeWidth="1" />
                  <button
                    type="button"
                    onClick={() => {
                      const parent = focusParent(focus);
                      if (parent) goToFocus(parent);
                    }}
                    className="cursor-pointer"
                  >
                    <circle cx={CX} cy={CY} r="92" className="fill-background stroke-borderSubtle hover:stroke-accentBlue transition-colors" strokeWidth="1" />
                  </button>
                  <text x={CX} y={CY - 8} textAnchor="middle" className="fill-textPrimary text-[18px] font-bold">
                    {focusLabel(focus)}
                  </text>
                  <text x={CX} y={CY + 16} textAnchor="middle" className="fill-textTertiary text-[11px] font-mono uppercase tracking-wide">
                    {focus.kind === "topic" ? "questions by level" : "click a segment to zoom"}
                  </text>

                  {focus.kind === "topic" ? (
                    <LevelRing
                      buckets={levelBuckets}
                      color={getAreaStyle(focus.area).primary}
                      onPick={(question) => setSelectedQuestionId(question.id)}
                    />
                  ) : (
                    segments.map((segment) => {
                      const start = (cursor / total) * TAU;
                      cursor += segment.count;
                      const end = (cursor / total) * TAU;
                      const hovered = hoveredId === segment.id;
                      const labelPoint = midpoint(hovered ? 128 : 142, hovered ? 278 : 262, start, end);
                      const showLabel = end - start > 0.16;
                      return (
                        <g key={segment.id}>
                          <path
                            d={ringPath(hovered ? 128 : 132, hovered ? 278 : 268, start, end)}
                            fill={segment.color}
                            opacity={hovered ? 1 : 0.72}
                            stroke="var(--background)"
                            strokeWidth="3"
                            filter={hovered ? "url(#segmentGlow)" : undefined}
                            className="cursor-pointer transition-all duration-300 ease-out"
                            onMouseEnter={() => setHoveredId(segment.id)}
                            onMouseLeave={() => setHoveredId(null)}
                            onClick={() => goToFocus(segment.focus)}
                          />
                          {showLabel && (
                            <g className="pointer-events-none transition-transform duration-300 ease-out">
                              <text
                                x={labelPoint.x}
                                y={labelPoint.y - 3}
                                textAnchor="middle"
                                className={clsx(
                                  "fill-white text-[10px] font-bold transition-all",
                                  hovered ? "text-[11px]" : "text-[10px]"
                                )}
                              >
                                {segment.label.length > 18
                                  ? `${segment.label.slice(0, 17)}...`
                                  : segment.label}
                              </text>
                              <text
                                x={labelPoint.x}
                                y={labelPoint.y + 12}
                                textAnchor="middle"
                                className="fill-white/80 text-[9px] font-mono"
                              >
                                {segment.count}
                              </text>
                            </g>
                          )}
                        </g>
                      );
                    })
                  )}
                </svg>
              </div>
            </div>
          </main>

          <aside className="lg:w-[390px] min-h-0 border-t lg:border-t-0 lg:border-l border-border bg-background overflow-auto">
            {selectedQuestion ? (
              <QuestionPanel
                question={selectedQuestion}
                areaName={areaById.get(selectedQuestion.competency_area)?.name ?? titleCase(selectedQuestion.competency_area)}
                related={relatedQuestions(selectedQuestion, questions)}
                onSelect={(id) => setSelectedQuestionId(id)}
                onClose={() => setSelectedQuestionId(null)}
              />
            ) : (
              <ExplorerPanel
                focus={focus}
                questions={focusQuestions}
                visibleQuestions={visibleQuestions}
                segments={segments}
                levelBuckets={levelBuckets}
                onFocus={goToFocus}
                onSelectQuestion={(id) => setSelectedQuestionId(id)}
                onReset={reset}
              />
            )}
          </aside>
        </div>
      </div>
    </div>
  );
}

function trackColor(track: string) {
  const colors: Record<string, string> = {
    cloud: "#60a5fa",
    edge: "#34d399",
    mobile: "#c084fc",
    tinyml: "#fbbf24",
    global: "#94a3b8",
  };
  return colors[track] ?? "#818cf8";
}

function relatedQuestions(question: Question, all: Question[]) {
  return all
    .filter((candidate) => {
      if (candidate.id === question.id) return false;
      const sharesChain = question.chain_ids?.some((id) => candidate.chain_ids?.includes(id));
      return sharesChain || candidate.topic === question.topic;
    })
    .sort((a, b) => {
      const chainId = question.chain_ids?.find((id) => a.chain_ids?.includes(id) || b.chain_ids?.includes(id));
      if (chainId) {
        const aPos = a.chain_positions?.[chainId] ?? 999;
        const bPos = b.chain_positions?.[chainId] ?? 999;
        if (aPos !== bPos) return aPos - bPos;
      }
      const levelDelta = levelIndex(a.level) - levelIndex(b.level);
      if (levelDelta !== 0) return levelDelta;
      return a.title.localeCompare(b.title);
    })
    .slice(0, 18);
}

function LevelRing({
  buckets,
  color,
  onPick,
}: {
  buckets: LevelBucket[];
  color: string;
  onPick: (question: Question) => void;
}) {
  const [hoveredLevel, setHoveredLevel] = useState<string | null>(null);
  const total = Math.max(1, buckets.reduce((sum, bucket) => sum + bucket.count, 0));
  let cursor = 0;
  return (
    <>
      {buckets.map((bucket) => {
        const start = (cursor / total) * TAU;
        cursor += bucket.count;
        const end = (cursor / total) * TAU;
        const hovered = hoveredLevel === bucket.level;
        const point = midpoint(hovered ? 128 : 132, hovered ? 278 : 268, start, end);
        return (
          <g key={bucket.level}>
            <path
              d={ringPath(hovered ? 128 : 132, hovered ? 278 : 268, start, end)}
              fill={color}
              opacity={hovered ? 1 : 0.45 + Math.min(0.35, bucket.count / total)}
              stroke="var(--background)"
              strokeWidth="3"
              className="cursor-pointer transition-all duration-300 ease-out"
              onMouseEnter={() => setHoveredLevel(bucket.level)}
              onMouseLeave={() => setHoveredLevel(null)}
              onClick={() => onPick(bucket.questions[0])}
            />
            <g className="pointer-events-none transition-all duration-300">
              <text
                x={point.x}
                y={point.y - 3}
                textAnchor="middle"
                className={clsx(
                  "fill-white text-[12px] font-bold transition-all",
                  hovered ? "text-[13px]" : "text-[12px]"
                )}
              >
                {bucket.level}
              </text>
              <text
                x={point.x}
                y={point.y + 13}
                textAnchor="middle"
                className="fill-white/80 text-[9px] font-mono"
              >
                {bucket.count}
              </text>
            </g>
          </g>
        );
      })}
    </>
  );
}

function Breadcrumb({
  items,
  active,
  onSelect,
}: {
  items: Array<{ label: string; focus: Focus }>;
  active: Focus;
  onSelect: (focus: Focus) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-[12px] mb-4">
      {items.map((item, index) => {
        const isLast = index === items.length - 1;
        return (
          <span key={`${item.label}-${index}`} className="inline-flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => !isLast && onSelect(item.focus)}
              disabled={isLast}
              className={isLast ? "text-textPrimary font-semibold" : "text-accentBlue hover:underline"}
            >
              {item.label}
            </button>
            {!isLast && <span className="text-textMuted">/</span>}
          </span>
        );
      })}
      {active.kind !== "root" && (
        <button
          type="button"
          onClick={() => onSelect({ kind: "root" })}
          className="ml-2 text-[11px] text-textTertiary hover:text-textPrimary"
        >
          reset
        </button>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-borderSubtle bg-surface/60 px-3 py-2 min-w-[86px]">
      <div className="text-lg font-bold font-mono text-textPrimary leading-none">{value}</div>
      <div className="text-[10px] uppercase tracking-wide text-textTertiary mt-1">{label}</div>
    </div>
  );
}

function FilterSelect({
  label,
  value,
  onChange,
  children,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  children: ReactNode;
}) {
  return (
    <label className="flex items-center gap-2 text-[11px] font-mono uppercase tracking-wide text-textTertiary">
      {label}
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="normal-case tracking-normal font-sans min-w-[128px] rounded-lg border border-border bg-background px-3 py-2 text-sm text-textSecondary focus:outline-none focus:border-borderHighlight"
      >
        {children}
      </select>
    </label>
  );
}

function ExplorerPanel({
  focus,
  questions,
  visibleQuestions,
  segments,
  levelBuckets,
  onFocus,
  onSelectQuestion,
  onReset,
}: {
  focus: Focus;
  questions: Question[];
  visibleQuestions: Question[];
  segments: Segment[];
  levelBuckets: LevelBucket[];
  onFocus: (focus: Focus) => void;
  onSelectQuestion: (id: string) => void;
  onReset: () => void;
}) {
  return (
    <div className="p-5">
      <div className="w-10 h-10 rounded-xl border border-accentBlue/30 bg-accentBlue/10 flex items-center justify-center mb-4">
        <Layers className="w-5 h-5 text-accentBlue" />
      </div>
      <h2 className="text-lg font-bold text-textPrimary mb-2">Radial question explorer</h2>
      <p className="text-sm text-textSecondary leading-relaxed mb-5">
        Click a ring segment to zoom in. Start with a deployment track, drill into an area,
        then choose a topic and question level.
      </p>

      {segments.length > 0 && focus.kind !== "topic" && (
        <section className="mb-6">
          <h3 className="text-[11px] font-mono uppercase tracking-wide text-textTertiary mb-2">
            Next zoom level
          </h3>
          <div className="space-y-2">
            {segments.slice(0, 10).map((segment) => (
              <button
                key={segment.id}
                type="button"
                onClick={() => onFocus(segment.focus)}
                className="w-full text-left p-3 rounded-lg border border-borderSubtle bg-surface/50 hover:bg-surface hover:border-borderHighlight transition-colors"
              >
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: segment.color }} />
                      <span className="text-sm font-semibold text-textPrimary">{segment.label}</span>
                    </div>
                    <div className="text-[10px] font-mono uppercase text-textTertiary mt-1">{segment.subtitle}</div>
                  </div>
                  <span className="text-xs font-mono text-textTertiary">{segment.count}</span>
                </div>
              </button>
            ))}
          </div>
        </section>
      )}

      {focus.kind === "topic" && (
        <section className="mb-6">
          <h3 className="text-[11px] font-mono uppercase tracking-wide text-textTertiary mb-2">
            Question levels
          </h3>
          <div className="grid grid-cols-2 gap-2">
            {levelBuckets.map((bucket) => (
              <button
                key={bucket.level}
                type="button"
                onClick={() => onSelectQuestion(bucket.questions[0].id)}
                className="p-3 rounded-lg border border-borderSubtle bg-surface/50 text-left hover:bg-surface hover:border-borderHighlight"
              >
                <div className="text-sm font-bold text-textPrimary">{bucket.level}</div>
                <div className="text-[11px] text-textTertiary">{bucket.count} questions</div>
              </button>
            ))}
          </div>
        </section>
      )}

      <section className="mb-6">
        <h3 className="text-[11px] font-mono uppercase tracking-wide text-textTertiary mb-2">
          Questions in view
        </h3>
        {visibleQuestions.length > 0 ? (
          <div className="space-y-2">
            {visibleQuestions.map((question) => (
              <QuestionButton
                key={question.id}
                question={question}
                onClick={() => onSelectQuestion(question.id)}
              />
            ))}
          </div>
        ) : (
          <div className="rounded-xl border border-borderSubtle bg-surface/60 p-4">
            <h3 className="text-sm font-bold text-textPrimary mb-1">No matching questions</h3>
            <p className="text-sm text-textSecondary leading-relaxed mb-3">
              Try a broader phrase, clear the level filter, or return to the full Vault.
            </p>
            <button
              type="button"
              onClick={onReset}
              className="inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg border border-border bg-background text-sm text-textSecondary hover:text-textPrimary hover:bg-surface transition-colors"
            >
              <LocateFixed className="w-4 h-4" />
              Reset filters
            </button>
          </div>
        )}
      </section>

      {focus.kind === "root" && (
        <section className="rounded-xl border border-accentBlue/20 bg-accentBlue/5 p-4">
          <h3 className="text-sm font-bold text-textPrimary mb-3">How to use this</h3>
          <HowToStep step="1" title="Pick a track" text="Cloud, Edge, Mobile, TinyML, and Global are the highest-level slices." />
          <HowToStep step="2" title="Zoom into an area" text="The next ring shows the skills inside that track: memory, deployment, latency, and more." />
          <HowToStep step="3" title="Choose a topic path" text="Topic and level views lead to concrete questions you can practice immediately." />
        </section>
      )}
    </div>
  );
}

function QuestionButton({ question, onClick }: { question: Question; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="w-full text-left p-3 rounded-lg border border-borderSubtle bg-surface/50 hover:bg-surface hover:border-borderHighlight transition-colors"
    >
      <div className="flex items-center gap-1.5 mb-1 text-[10px] font-mono uppercase text-textTertiary">
        <span>{question.level}</span>
        <span>{question.track}</span>
        <span>{titleCase(question.competency_area)}</span>
        {(question.chain_ids?.length ?? 0) > 0 && <span className="text-accentAmber">chain</span>}
      </div>
      <p className="text-sm font-semibold text-textPrimary leading-snug">{question.title}</p>
    </button>
  );
}

function HowToStep({ step, title, text }: { step: string; title: string; text: string }) {
  return (
    <div className="flex gap-3 mb-3 last:mb-0">
      <div className="w-6 h-6 rounded-full bg-accentBlue text-white flex items-center justify-center text-[11px] font-bold shrink-0">
        {step}
      </div>
      <div>
        <div className="text-sm font-semibold text-textPrimary">{title}</div>
        <p className="text-[12px] text-textSecondary leading-relaxed">{text}</p>
      </div>
    </div>
  );
}

function QuestionPanel({
  question,
  areaName,
  related,
  onSelect,
  onClose,
}: {
  question: Question;
  areaName: string;
  related: Question[];
  onSelect: (id: string) => void;
  onClose: () => void;
}) {
  // Prefer primary chain when the question has both — secondary chains
  // are an alternative path the user can deep-link into (?chain=<id>) but
  // shouldn't be the default explorer surface.
  const activeChainId =
    question.chain_ids?.find((id) => question.chain_tiers?.[id] !== "secondary")
    ?? question.chain_ids?.[0]
    ?? null;
  const chainPath = activeChainId
    ? [question, ...related.filter((item) => item.chain_ids?.includes(activeChainId))]
        .sort((a, b) =>
          (a.chain_positions?.[activeChainId] ?? 999) -
          (b.chain_positions?.[activeChainId] ?? 999),
        )
    : [];
  const chainIds = new Set(chainPath.map((item) => item.id));
  const topicNeighbors = related
    .filter((item) => item.topic === question.topic && !chainIds.has(item.id))
    .slice(0, 8);

  return (
    <div className="p-5">
      <div className="flex items-start justify-between gap-3 mb-4">
        <div className="inline-flex items-center gap-2 text-[10px] font-mono uppercase tracking-wide text-textTertiary">
          <span className="px-2 py-1 rounded border border-border bg-surface">{question.level}</span>
          <span className="px-2 py-1 rounded border border-border bg-surface">{question.track}</span>
          {(question.chain_ids?.length ?? 0) > 0 && (
            <span className="px-2 py-1 rounded border border-accentAmber/30 bg-accentAmber/10 text-accentAmber">
              chain
            </span>
          )}
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close question panel"
          className="text-textTertiary hover:text-textPrimary"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      <h2 className="text-xl font-extrabold text-textPrimary tracking-tight leading-tight mb-3">
        {question.title}
      </h2>
      <p className="text-sm text-textSecondary leading-relaxed mb-4">
        {question.question || "Open the practice view to see the full scenario and solution."}
      </p>

      <div className="grid grid-cols-2 gap-2 mb-5">
        <InfoTile icon={<Layers className="w-3.5 h-3.5" />} label="Area" value={areaName} />
        <InfoTile icon={<Route className="w-3.5 h-3.5" />} label="Topic" value={titleCase(question.topic)} />
        <InfoTile icon={<Sparkles className="w-3.5 h-3.5" />} label="Zone" value={titleCase(question.zone)} />
        <InfoTile icon={<BookOpen className="w-3.5 h-3.5" />} label="Phase" value={question.phase ?? "mixed"} />
      </div>

      <Link
        href={`/practice?q=${question.id}`}
        className="inline-flex items-center justify-center gap-1.5 w-full px-4 py-2.5 rounded-lg bg-accentBlue text-white text-sm font-bold hover:opacity-90 transition-opacity mb-6"
      >
        Practice this question
        <ArrowRight className="w-4 h-4" />
      </Link>

      <section>
        <h3 className="text-[11px] font-mono uppercase tracking-wide text-textTertiary mb-2">
          Why these are nearby
        </h3>
        <p className="text-[12px] text-textSecondary leading-relaxed mb-4">
          StaffML links questions by chain first, then by topic. Use chains as a learning path;
          use topic neighbors to move sideways within the same concept.
        </p>

        {chainPath.length > 1 ? (
          <div className="mb-5">
            <h4 className="text-[10px] font-mono uppercase tracking-wide text-accentAmber mb-2">
              Chain path
            </h4>
            <div className="relative space-y-2 before:absolute before:left-[13px] before:top-4 before:bottom-4 before:w-px before:bg-accentAmber/25">
              {chainPath.map((item, index) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => item.id !== question.id && onSelect(item.id)}
                  className={clsx(
                    "relative w-full text-left pl-9 pr-3 py-3 rounded-lg border transition-colors",
                    item.id === question.id
                      ? "border-accentAmber/40 bg-accentAmber/10"
                      : "border-borderSubtle bg-surface/50 hover:bg-surface hover:border-borderHighlight",
                  )}
                >
                  <span
                    className={clsx(
                      "absolute left-[7px] top-4 w-3.5 h-3.5 rounded-full border-2 bg-background",
                      item.id === question.id ? "border-accentAmber" : "border-accentAmber/50",
                    )}
                  />
                  <div className="flex items-center gap-1.5 mb-1 text-[10px] font-mono uppercase text-textTertiary">
                    <span>Step {index + 1}</span>
                    <span>{item.level}</span>
                    {item.id === question.id && <span className="text-accentAmber">current</span>}
                  </div>
                  <p className="text-sm font-semibold text-textPrimary leading-snug">{item.title}</p>
                </button>
              ))}
            </div>
          </div>
        ) : null}

        {topicNeighbors.length > 0 ? (
          <div>
            <h4 className="text-[10px] font-mono uppercase tracking-wide text-textTertiary mb-2">
              Same topic: {titleCase(question.topic)}
            </h4>
            <div className="space-y-2">
              {topicNeighbors.map((item) => (
                <QuestionButton key={item.id} question={item} onClick={() => onSelect(item.id)} />
              ))}
            </div>
          </div>
        ) : chainPath.length > 1 ? null : (
          <p className="text-sm text-textTertiary">
            No chain or same-topic neighbors are visible for this question yet.
          </p>
        )}
      </section>
    </div>
  );
}

function InfoTile({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="rounded-lg border border-borderSubtle bg-surface/50 p-3">
      <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wide text-textTertiary mb-1">
        {icon}
        {label}
      </div>
      <div className="text-sm font-semibold text-textPrimary leading-snug">{value}</div>
    </div>
  );
}
