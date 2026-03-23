"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { BarChart3, Trash2, Terminal, Crosshair, Download, Upload } from "lucide-react";
import clsx from "clsx";
import Link from "next/link";
import { getCompetencyAreas, getTracks, getQuestionsByFilter } from "@/lib/corpus";
import { getAttempts, getGauntletResults, clearProgress, exportProgress, importProgress } from "@/lib/progress";

export default function HeatMapPage() {
  const [mounted, setMounted] = useState(false);
  const [heatData, setHeatData] = useState<Record<string, Record<string, { attempted: number; correct: number }>>>({});
  const [gauntletCount, setGauntletCount] = useState(0);
  const [totalAttempted, setTotalAttempted] = useState(0);
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  const tracks = getTracks().filter(t => t !== "global");
  const areas = getCompetencyAreas();

  useEffect(() => {
    setMounted(true);
    loadData();
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
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <BarChart3 className="w-8 h-8 text-accentGreen" />
            <div>
              <h1 className="text-3xl font-extrabold text-white tracking-tight">Readiness Heat Map</h1>
              <p className="text-sm text-textSecondary">Track × Competency — your interview readiness at a glance</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="text-xs text-textTertiary">Gauntlets completed</div>
              <div className="text-lg font-bold font-mono text-white">{gauntletCount}</div>
            </div>
            <div className="text-right">
              <div className="text-xs text-textTertiary">Total questions</div>
              <div className="text-lg font-bold font-mono text-white">{totalAttempted}</div>
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
                <span className="text-sm font-bold text-white">Readiness Verdict</span>
                <span className={clsx(
                  "text-lg font-bold font-mono",
                  overallPct >= 70 ? "text-accentGreen" : overallPct >= 40 ? "text-accentAmber" : "text-accentRed"
                )}>
                  {overallPct}%
                </span>
              </div>
              <p className="text-sm text-textSecondary mb-3">{verdict}</p>
              <div className="flex items-center gap-4 text-[10px] font-mono text-textTertiary">
                {strongest && <span>Strongest: <span className="text-accentGreen capitalize">{strongest.area}</span> ({strongest.pct}%)</span>}
                {weakest && <span>Weakest: <span className="text-accentRed capitalize">{weakest.area}</span> ({weakest.pct}%)</span>}
                <span>{totalAttempted} total answers</span>
              </div>
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
            <h2 className="text-xl font-bold text-white mb-2">No data yet</h2>
            <p className="text-sm text-textSecondary mb-6 max-w-md">
              Complete a Gauntlet or drill some questions to see your readiness heat map populate.
            </p>
            <Link
              href="/gauntlet"
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-white text-black font-bold rounded-lg hover:bg-gray-100 transition-all text-sm"
            >
              <Crosshair className="w-4 h-4" /> Start a Gauntlet
            </Link>
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
                                href={`/drill?track=${t}&area=${area}`}
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
                  }}
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
                        if (ok) { loadData(); }
                      };
                      reader.readAsText(file);
                    }}
                  />
                </label>
                <button
                  onClick={() => setShowClearConfirm(true)}
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
                    className="px-3 py-1.5 text-xs text-textTertiary hover:text-white transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleClear}
                    className="px-3 py-1.5 text-xs bg-accentRed text-white rounded-md hover:bg-accentRed/80 transition-colors"
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
