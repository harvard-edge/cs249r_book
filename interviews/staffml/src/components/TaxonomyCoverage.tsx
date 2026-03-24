"use client";

import { useMemo } from "react";
import clsx from "clsx";
import { getConcepts, getChapters, getTracks, formatChapter } from "@/lib/taxonomy";

const LEVELS_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6+"];

export default function TaxonomyCoverage() {
  const concepts = useMemo(() => getConcepts(), []);
  const chapters = useMemo(() => getChapters(), []);
  const tracks = useMemo(() => getTracks(), []);

  // ── Track × Level heatmap ──────────────────────────────────
  const trackLevel = useMemo(() => {
    const grid: Record<string, Record<string, number>> = {};
    tracks.forEach((t) => {
      grid[t] = {};
      LEVELS_ORDER.forEach((l) => {
        grid[t][l] = 0;
      });
    });
    concepts.forEach((c) => {
      c.tracks.forEach((t) => {
        if (!grid[t]) return;
        Object.entries(c.level_distribution).forEach(([level, count]) => {
          if (grid[t][level] !== undefined) grid[t][level] += count;
        });
      });
    });
    return grid;
  }, [concepts, tracks]);

  // ── Chapter coverage bars ──────────────────────────────────
  const chapterData = useMemo(() => {
    const byChapter: Record<string, { total: number; tested: number; qs: number }> = {};
    concepts.forEach((c) => {
      c.source_chapters.forEach((ch) => {
        if (!byChapter[ch]) byChapter[ch] = { total: 0, tested: 0, qs: 0 };
        byChapter[ch].total++;
        if (c.question_count > 0) {
          byChapter[ch].tested++;
          byChapter[ch].qs += c.question_count;
        }
      });
    });
    return Object.entries(byChapter)
      .map(([ch, data]) => ({ ch, ...data, pct: data.total > 0 ? data.tested / data.total : 0 }))
      .sort((a, b) => a.ch.localeCompare(b.ch));
  }, [concepts]);

  // ── Untested concepts (gap list) ───────────────────────────
  const untested = useMemo(
    () => concepts.filter((c) => c.question_count === 0).sort((a, b) => a.name.localeCompare(b.name)),
    [concepts]
  );

  // ── Role distribution ──────────────────────────────────────
  const roles = useMemo(() => {
    const counts = { foundational: 0, competency: 0, contextual: 0 };
    concepts.forEach((c) => counts[c.role]++);
    return counts;
  }, [concepts]);

  // Max value for heatmap scaling
  const maxCell = useMemo(() => {
    let max = 0;
    Object.values(trackLevel).forEach((levels) => {
      Object.values(levels).forEach((v) => {
        if (v > max) max = v;
      });
    });
    return max || 1;
  }, [trackLevel]);

  return (
    <div className="space-y-8 overflow-auto">
      {/* Summary stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard label="Total Concepts" value={concepts.length} />
        <StatCard label="Tested" value={concepts.length - untested.length} sub={`${((1 - untested.length / concepts.length) * 100).toFixed(0)}%`} />
        <StatCard label="Untested Gaps" value={untested.length} alert={untested.length > 0} />
        <StatCard label="Foundational" value={roles.foundational} sub={`${roles.competency} comp, ${roles.contextual} ctx`} />
      </div>

      {/* Track × Level Heatmap */}
      <div>
        <h3 className="text-sm font-bold text-white mb-3">Track x Level Coverage</h3>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="text-left text-[10px] font-mono text-textTertiary uppercase tracking-wider p-2">Track</th>
                {LEVELS_ORDER.map((l) => (
                  <th key={l} className="text-center text-[10px] font-mono text-textTertiary uppercase tracking-wider p-2">{l}</th>
                ))}
                <th className="text-center text-[10px] font-mono text-textTertiary uppercase tracking-wider p-2">Total</th>
              </tr>
            </thead>
            <tbody>
              {tracks.map((t) => {
                const total = Object.values(trackLevel[t] || {}).reduce((s, v) => s + v, 0);
                return (
                  <tr key={t} className="border-t border-border/30">
                    <td className="p-2 text-xs text-textSecondary font-medium capitalize">
                      {t === "tinyml" ? "TinyML" : t}
                    </td>
                    {LEVELS_ORDER.map((l) => {
                      const v = trackLevel[t]?.[l] || 0;
                      const intensity = v / maxCell;
                      return (
                        <td key={l} className="p-1.5 text-center">
                          <div
                            className="rounded-md py-1.5 text-[11px] font-mono"
                            style={{
                              backgroundColor: v > 0 ? `rgba(59, 130, 246, ${0.1 + intensity * 0.5})` : "rgba(255,255,255,0.03)",
                              color: v > 0 ? "#93c5fd" : "#484848",
                            }}
                          >
                            {v}
                          </div>
                        </td>
                      );
                    })}
                    <td className="p-1.5 text-center">
                      <span className="text-xs font-mono font-bold text-white">{total}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Chapter Coverage Bars */}
      <div>
        <h3 className="text-sm font-bold text-white mb-3">Chapter Coverage</h3>
        <div className="space-y-1.5">
          {chapterData.map((d) => (
            <div key={d.ch} className="flex items-center gap-3">
              <span className="text-[10px] text-textTertiary w-40 truncate shrink-0">
                {formatChapter(d.ch)}
              </span>
              <div className="flex-1 h-5 bg-surface rounded-md overflow-hidden relative">
                <div
                  className={clsx(
                    "h-full rounded-md transition-all",
                    d.pct >= 0.9 ? "bg-accentGreen/40" : d.pct >= 0.6 ? "bg-accentBlue/40" : d.pct >= 0.3 ? "bg-accentAmber/40" : "bg-accentRed/40"
                  )}
                  style={{ width: `${d.pct * 100}%` }}
                />
                <span className="absolute right-2 top-0.5 text-[10px] font-mono text-textSecondary">
                  {d.tested}/{d.total}
                </span>
              </div>
              <span className="text-[10px] font-mono text-textTertiary w-12 text-right">
                {d.qs} Qs
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Untested Concepts (Gap List) */}
      {untested.length > 0 && (
        <div>
          <h3 className="text-sm font-bold text-white mb-3">
            Untested Concepts ({untested.length})
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-1.5">
            {untested.map((c) => (
              <div
                key={c.id}
                className="flex items-center gap-2 px-2 py-1.5 rounded bg-surface/50 border border-border/30"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-accentRed shrink-0" />
                <span className="text-xs text-textSecondary truncate">{c.name}</span>
                <span className="text-[9px] text-textTertiary ml-auto shrink-0">
                  {c.tracks.join(", ")}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value, sub, alert }: { label: string; value: number; sub?: string; alert?: boolean }) {
  return (
    <div className="p-3 rounded-lg border border-border bg-surface/50">
      <div className="text-[10px] text-textTertiary uppercase tracking-wider mb-1">{label}</div>
      <div className={clsx("text-xl font-bold font-mono", alert ? "text-accentRed" : "text-white")}>
        {value}
      </div>
      {sub && <div className="text-[10px] text-textTertiary mt-0.5">{sub}</div>}
    </div>
  );
}
