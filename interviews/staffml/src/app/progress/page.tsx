"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { BarChart3, Trash2, Terminal, Crosshair, Download, Upload, Target, AlertTriangle } from "lucide-react";
import clsx from "clsx";
import Link from "next/link";
import { getCompetencyAreas, getTracks, getQuestionsByFilter } from "@/lib/corpus";
import { getAttempts, getGauntletResults, clearProgress, exportProgress, importProgress, getLastExportAt } from "@/lib/progress";
import { useToast } from "@/components/Toast";
import { track } from "@/lib/analytics";

/**
 * Format an ISO timestamp as a human-friendly relative string for the
 * "Last exported" readout. Chosen to be blunt, not cute — "3 days ago"
 * is more informative than "a few days ago" for a backup indicator.
 */
function formatRelativeExport(iso: string | null): string | null {
  if (!iso) return null;
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return null;
  const diffMs = Date.now() - then;
  const diffDays = Math.floor(diffMs / 86400000);
  if (diffDays < 1) return "today";
  if (diffDays === 1) return "yesterday";
  if (diffDays < 7) return `${diffDays} days ago`;
  if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
  if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
  return `${Math.floor(diffDays / 365)} years ago`;
}

export default function ProgressPage() {
  const { show: showToast } = useToast();
  const [mounted, setMounted] = useState(false);
  const [heatData, setHeatData] = useState<Record<string, Record<string, { attempted: number; correct: number }>>>({});
  const [gauntletCount, setGauntletCount] = useState(0);
  const [totalAttempted, setTotalAttempted] = useState(0);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [lastExport, setLastExport] = useState<string | null>(null);

  const tracks = getTracks().filter(t => t !== "global");
  const areas = getCompetencyAreas();

  useEffect(() => {
    setMounted(true);
    loadData();
    setLastExport(getLastExportAt());
  }, []);

  const loadData = () => {
    const attempts = getAttempts();
    const gauntlets = getGauntletResults();
    setGauntletCount(gauntlets.length);
    setTotalAttempted(attempts.length);

    const data: Record<string, Record<string, { attempted: number; correct: number }>> = {};
    areas.forEach(area => {
      data[area] = {};
      tracks.forEach(track => {
        data[area][track] = { attempted: 0, correct: 0 };
      });
    });

    attempts.forEach(a => {
      if (data[a.competencyArea]?.[a.track]) {
        data[a.competencyArea][a.track].attempted++;
        if (a.selfScore >= 2) data[a.competencyArea][a.track].correct++;
      }
    });

    setHeatData(data);
  };

  const handleClear = () => {
    clearProgress();
    setShowClearConfirm(false);
    loadData();
    // clearProgress also wipes the last-export timestamp in localStorage;
    // sync the UI state so the banner updates immediately.
    setLastExport(null);
  };

  const getCellColor = (attempted: number, correct: number) => {
    if (attempted === 0) return "bg-surface border-border";
    const ratio = correct / attempted;
    if (ratio >= 0.7) return "bg-accentGreen/20 border-accentGreen/40";
    if (ratio >= 0.4) return "bg-accentAmber/20 border-accentAmber/40";
    return "bg-accentRed/20 border-accentRed/40";
  };

  const getCellText = (attempted: number, correct: number) => {
    if (attempted === 0) return "text-textTertiary";
    const ratio = correct / attempted;
    if (ratio >= 0.7) return "text-accentGreen";
    if (ratio >= 0.4) return "text-accentAmber";
    return "text-accentRed";
  };

  if (!mounted) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Terminal className="w-6 h-6 text-textTertiary animate-pulse" />
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col px-6 py-10">
      <div className="max-w-5xl mx-auto w-full">
        {/* Trust banner + storage-wipe warning.
            The original banner said only the good news ("your data
            stays in your browser"). That was the "no-tracking" feature
            pitch but it hid the implied bad news: clearing browser
            storage, switching browsers, or using a private window all
            delete your progress. Now both sides are surfaced and the
            Export button is explicitly framed as the backup path. */}
        <div className="mb-6 px-4 py-3 bg-surface border border-borderSubtle rounded-lg">
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="w-3.5 h-3.5 text-accentAmber mt-0.5 shrink-0" />
            <div className="text-[12px] text-textSecondary leading-relaxed">
              <span className="text-textPrimary font-medium">Your data stays in your browser.</span>{' '}
              No accounts, no tracking. But clearing your browser storage, switching browsers, or
              using a private window will wipe your progress.{' '}
              <span className="text-textPrimary">Use Export below to back up regularly.</span>
              {lastExport && (
                <span className="block mt-1 text-[11px] text-textTertiary">
                  Last exported {formatRelativeExport(lastExport)}.
                </span>
              )}
              {!lastExport && totalAttempted > 0 && (
                <span className="block mt-1 text-[11px] text-accentAmber">
                  You haven&apos;t exported yet.
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-4 mb-8">
          <div className="flex items-center gap-3">
            <BarChart3 className="w-8 h-8 text-accentGreen shrink-0" />
            <div>
              <h1 className="text-2xl sm:text-3xl font-extrabold text-textPrimary tracking-tight">Progress</h1>
              <p className="text-sm text-textSecondary">Track &times; Competency &mdash; your readiness at a glance</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="text-[10px] text-textTertiary">Gauntlets</div>
              <div className="text-lg font-bold font-mono text-textPrimary">{gauntletCount}</div>
            </div>
            <div className="text-right">
              <div className="text-[10px] text-textTertiary">Questions</div>
              <div className="text-lg font-bold font-mono text-textPrimary">{totalAttempted}</div>
            </div>
          </div>
        </div>

        {/* Readiness summary per track */}
        {totalAttempted > 0 && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-8">
            {tracks.map(t => {
              let trackAttempted = 0;
              let trackCorrect = 0;
              areas.forEach(area => {
                const cell = heatData[area]?.[t];
                if (cell) {
                  trackAttempted += cell.attempted;
                  trackCorrect += cell.correct;
                }
              });
              const pct = trackAttempted >= 3
                ? Math.round((trackCorrect / trackAttempted) * 100)
                : -1;

              return (
                <div key={t} className="p-3 rounded-lg border border-border bg-surface/50">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium text-textSecondary capitalize">
                      {t === "tinyml" ? "TinyML" : t}
                    </span>
                    {pct >= 0 ? (
                      <span className={clsx(
                        "text-sm font-bold font-mono",
                        pct >= 70 ? "text-accentGreen" : pct >= 40 ? "text-accentAmber" : "text-accentRed"
                      )}>
                        {pct}%
                      </span>
                    ) : (
                      <span className="text-[10px] text-textTertiary">Not enough data</span>
                    )}
                  </div>
                  {pct >= 0 && (
                    <div className="h-1.5 bg-border rounded-full overflow-hidden">
                      <div
                        className={clsx(
                          "h-full rounded-full transition-all",
                          pct >= 70 ? "bg-accentGreen" : pct >= 40 ? "bg-accentAmber" : "bg-accentRed"
                        )}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Readiness verdict */}
        {totalAttempted >= 10 && (() => {
          // Find strongest and weakest areas
          const areaScores = areas.map(area => {
            let attempted = 0, correct = 0;
            tracks.forEach(t => {
              const cell = heatData[area]?.[t];
              if (cell) { attempted += cell.attempted; correct += cell.correct; }
            });
            return { area, attempted, pct: attempted >= 2 ? Math.round((correct / attempted) * 100) : -1 };
          }).filter(a => a.pct >= 0).sort((a, b) => a.pct - b.pct);

          const weakest = areaScores[0];
          const strongest = areaScores[areaScores.length - 1];
          const overallPct = Math.round(areaScores.reduce((s, a) => s + a.pct, 0) / areaScores.length);
          const readyAreas = areaScores.filter(a => a.pct >= 70).length;
          const totalAreas = areaScores.length;

          const verdict = overallPct >= 70
            ? `You're looking strong — ${readyAreas}/${totalAreas} competencies above 70%.`
            : overallPct >= 40
            ? `Making progress — ${readyAreas}/${totalAreas} competencies ready. Focus on ${weakest?.area}.`
            : `Early stages — keep drilling. Your weakest area is ${weakest?.area} (${weakest?.pct}%).`;

          return (
            <div className="mb-6 p-5 rounded-xl border border-border bg-surface/80">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-bold text-textPrimary">Readiness Verdict</span>
                <span className={clsx(
                  "text-lg font-bold font-mono",
                  overallPct >= 70 ? "text-accentGreen" : overallPct >= 40 ? "text-accentAmber" : "text-accentRed"
                )}>
                  {overallPct}%
                </span>
              </div>
              <p className="text-sm text-textSecondary mb-3">{verdict}</p>
              <div className="flex items-center gap-4 text-[10px] font-mono text-textTertiary mb-4">
                {strongest && <span>Strongest: <span className="text-accentGreen capitalize">{strongest.area}</span> ({strongest.pct}%)</span>}
                {weakest && <span>Weakest: <span className="text-accentRed capitalize">{weakest.area}</span> ({weakest.pct}%)</span>}
                <span>{totalAttempted} total answers</span>
              </div>
              {weakest && weakest.pct < 70 && (
                <Link
                  href={`/practice?area=${encodeURIComponent(weakest.area)}`}
                  className="inline-flex items-center gap-2 px-4 py-2.5 bg-accentBlue text-white font-bold rounded-lg text-sm hover:opacity-90 transition-opacity"
                >
                  <Target className="w-4 h-4" /> Drill {weakest.area}
                </Link>
              )}
            </div>
          );
        })()}

        {totalAttempted === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col items-center justify-center py-20 text-center"
          >
            <div className="w-16 h-16 rounded-full bg-surface border border-border flex items-center justify-center mb-6">
              <BarChart3 className="w-8 h-8 text-textTertiary" />
            </div>
            <h2 className="text-xl font-bold text-textPrimary mb-2">No data yet</h2>
            <p className="text-sm text-textSecondary mb-6 max-w-md leading-relaxed">
              Your progress page lights up after you've drilled questions or completed a mock interview. Pick a starting point — there's no wrong choice.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-2">
              <Link
                href="/practice?level=L1"
                className="inline-flex items-center gap-2 px-4 py-2.5 bg-surface border border-border text-textPrimary hover:border-accentBlue/60 rounded-lg transition-all text-sm font-medium"
                title="Start with the easiest level — recall questions to warm up."
              >
                <Target className="w-4 h-4 text-accentGreen" /> Drill 5 easy
              </Link>
              <Link
                href="/practice?daily=1"
                className="inline-flex items-center gap-2 px-4 py-2.5 bg-surface border border-border text-textPrimary hover:border-accentBlue/60 rounded-lg transition-all text-sm font-medium"
                title="3 hand-picked questions, same for everyone, takes ~5 min."
              >
                <Target className="w-4 h-4 text-accentBlue" /> Daily challenge
              </Link>
              <Link
                href="/gauntlet"
                className="inline-flex items-center gap-2 px-4 py-2.5 bg-textPrimary text-background font-bold rounded-lg hover:opacity-90 transition-all text-sm"
              >
                <Crosshair className="w-4 h-4" /> Mock interview
              </Link>
            </div>
            <p className="text-[11px] text-textTertiary italic mt-6 max-w-md">
              Tip: every question you answer feeds the heat map below, so you can see at a glance which competency areas need more work.
            </p>
          </motion.div>
        ) : (
          <>
            {/* Heat map grid */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="overflow-x-auto"
            >
              <table className="w-full border-collapse">
                <thead>
                  <tr>
                    <th className="text-left text-[10px] font-mono text-textTertiary uppercase tracking-widest p-2 w-36">
                      Competency
                    </th>
                    {tracks.map(t => (
                      <th key={t} className="text-center text-[10px] font-mono text-textTertiary uppercase tracking-widest p-2 capitalize">
                        {t === "tinyml" ? "TinyML" : t}
                      </th>
                    ))}
                    <th className="text-center text-[10px] font-mono text-textTertiary uppercase tracking-widest p-2">
                      Overall
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {areas.map(area => {
                    let totalAttemptedArea = 0;
                    let totalCorrectArea = 0;
                    tracks.forEach(t => {
                      const cell = heatData[area]?.[t];
                      if (cell) {
                        totalAttemptedArea += cell.attempted;
                        totalCorrectArea += cell.correct;
                      }
                    });
                    const overallPct = totalAttemptedArea > 0
                      ? Math.round((totalCorrectArea / totalAttemptedArea) * 100)
                      : -1;

                    return (
                      <tr key={area}>
                        <td className="text-xs text-textSecondary capitalize p-2 font-medium">{area}</td>
                        {tracks.map(t => {
                          const cell = heatData[area]?.[t] || { attempted: 0, correct: 0 };
                          const pct = cell.attempted > 0
                            ? Math.round((cell.correct / cell.attempted) * 100)
                            : -1;

                          return (
                            <td key={t} className="p-1.5">
                              <Link
                                href={`/practice?track=${t}&area=${area}`}
                                className={clsx(
                                  "w-full h-12 rounded-lg border flex flex-col items-center justify-center transition-all cursor-pointer hover:ring-1 hover:ring-accentBlue/40",
                                  getCellColor(cell.attempted, cell.correct)
                                )}
                              >
                                {cell.attempted > 0 ? (
                                  <>
                                    <span className={clsx("text-sm font-bold font-mono", getCellText(cell.attempted, cell.correct))}>
                                      {pct}%
                                    </span>
                                    <span className="text-[9px] text-textTertiary">{cell.attempted} Qs</span>
                                  </>
                                ) : (
                                  <span className="text-[9px] text-textTertiary">drill</span>
                                )}
                              </Link>
                            </td>
                          );
                        })}
                        <td className="p-1.5">
                          <div className={clsx(
                            "w-full h-12 rounded-lg border flex items-center justify-center",
                            totalAttemptedArea > 0 ? getCellColor(totalAttemptedArea, totalCorrectArea) : "bg-surface border-border"
                          )}>
                            {overallPct >= 0 ? (
                              <span className={clsx("text-sm font-bold font-mono", getCellText(totalAttemptedArea, totalCorrectArea))}>
                                {overallPct}%
                              </span>
                            ) : (
                              <span className="text-[9px] text-textTertiary">—</span>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </motion.div>

            {/* Legend + actions */}
            <div className="flex items-center justify-between mt-8 pt-6 border-t border-border">
              <div className="flex items-center gap-6">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded bg-surface border border-border" />
                  <span className="text-[10px] text-textTertiary">Not attempted</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded bg-accentRed/20 border border-accentRed/40" />
                  <span className="text-[10px] text-textTertiary">&lt; 40%</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded bg-accentAmber/20 border border-accentAmber/40" />
                  <span className="text-[10px] text-textTertiary">40-70%</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded bg-accentGreen/20 border border-accentGreen/40" />
                  <span className="text-[10px] text-textTertiary">&gt; 70%</span>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => {
                    const json = exportProgress();
                    const blob = new Blob([json], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `staffml-progress-${new Date().toISOString().slice(0, 10)}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                    track({ type: 'progress_exported' });
                    // Refresh the "Last exported" readout in the banner
                    // immediately after the file is written.
                    setLastExport(getLastExportAt());
                  }}
                  title="Download a JSON backup of your progress. Keep this file safe — it's your only way to restore."
                  className="text-xs text-textTertiary hover:text-accentBlue transition-colors flex items-center gap-1"
                >
                  <Download className="w-3 h-3" /> Export
                </button>
                <label className="text-xs text-textTertiary hover:text-accentGreen transition-colors flex items-center gap-1 cursor-pointer">
                  <Upload className="w-3 h-3" /> Import
                  <input
                    type="file"
                    accept=".json"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (!file) return;
                      const reader = new FileReader();
                      reader.onload = () => {
                        const ok = importProgress(reader.result as string);
                        if (ok) {
                          loadData();
                          track({ type: 'progress_imported' });
                          showToast({ type: 'success', title: 'Progress imported', description: 'Your data has been restored.' });
                        } else {
                          showToast({ type: 'info', title: 'Import failed', description: 'Invalid file format. Please use an exported StaffML JSON file.' });
                        }
                      };
                      reader.readAsText(file);
                    }}
                  />
                </label>
                <button
                  onClick={() => setShowClearConfirm(true)}
                  title="Permanently delete all your progress. Export a backup first if you might want it back."
                  className="text-xs text-textTertiary hover:text-accentRed transition-colors flex items-center gap-1"
                >
                  <Trash2 className="w-3 h-3" /> Clear
                </button>
              </div>
            </div>

            {/* Clear confirmation */}
            {showClearConfirm && (
              <div className="mt-4 p-4 bg-accentRed/5 border border-accentRed/30 rounded-lg flex items-center justify-between">
                <span className="text-sm text-textSecondary">This will delete all your progress. Are you sure?</span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setShowClearConfirm(false)}
                    className="px-3 py-1.5 text-xs text-textTertiary hover:text-textPrimary transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleClear}
                    className="px-3 py-1.5 text-xs bg-accentRed text-textPrimary rounded-md hover:bg-accentRed/80 transition-colors"
                  >
                    Delete Everything
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
