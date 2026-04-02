"use client";

import { useState, useEffect } from "react";
import {
  BarChart3, Terminal, Activity, Users, Target, Crosshair,
  Flag, Lightbulb, Calendar, TrendingUp, Trash2,
} from "lucide-react";
import clsx from "clsx";
import { computeSummary, getAnalyticsEvents, clearAnalytics, type AnalyticsSummary } from "@/lib/analytics";
import { getVaultStats } from "@/lib/taxonomy";

export default function DashboardPage() {
  const [mounted, setMounted] = useState(false);
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [vaultStats, setVaultStats] = useState({ totalQuestions: 0, totalTopics: 0, totalAreas: 0, totalZones: 0 });

  useEffect(() => {
    setMounted(true);
    setSummary(computeSummary());
    setVaultStats(getVaultStats());
  }, []);

  if (!mounted || !summary) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Terminal className="w-6 h-6 text-textTertiary animate-pulse" />
      </div>
    );
  }

  const levelOrder = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6+'];
  const recentDays = Object.entries(summary.eventsByDay)
    .sort(([a], [b]) => b.localeCompare(a))
    .slice(0, 14)
    .reverse();

  return (
    <div className="flex-1 flex flex-col px-6 py-10">
      <div className="max-w-5xl mx-auto w-full">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <Activity className="w-8 h-8 text-accentBlue" />
            <div>
              <h1 className="text-3xl font-extrabold text-textPrimary tracking-tight">Dashboard</h1>
              <p className="text-sm text-textSecondary">Anonymous usage analytics — all data stays in your browser</p>
            </div>
          </div>
          <button
            onClick={() => {
              clearAnalytics();
              setSummary(computeSummary());
            }}
            className="text-xs text-textTertiary hover:text-accentRed transition-colors flex items-center gap-1"
          >
            <Trash2 className="w-3 h-3" /> Clear analytics
          </button>
        </div>

        {/* Vault overview */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-8">
          <StatCard label="Questions" value={vaultStats.totalQuestions.toLocaleString()} icon={Target} />
          <StatCard label="Topics" value={vaultStats.totalTopics.toString()} icon={BarChart3} />
          <StatCard label="Areas" value={vaultStats.totalAreas.toString()} icon={BarChart3} />
          <StatCard label="Zones" value={vaultStats.totalZones.toString()} icon={BarChart3} />
        </div>

        {/* Key metrics */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-8">
          <StatCard label="Questions Scored" value={summary.questionsScored.toString()} icon={Target} color="blue" />
          <StatCard label="Gauntlets Done" value={summary.gauntletsCompleted.toString()} icon={Crosshair} color="green" />
          <StatCard label="Issues Reported" value={summary.questionsReported.toString()} icon={Flag} color="red" />
          <StatCard label="Improvements" value={summary.improvementsSuggested.toString()} icon={Lightbulb} color="amber" />
        </div>

        {/* Activity timeline */}
        {recentDays.length > 0 && (
          <div className="mb-8">
            <SectionLabel icon={Calendar} label="Activity (Last 14 Days)" />
            <div className="flex items-end gap-1 h-24 mt-3 px-1">
              {recentDays.map(([day, count]) => {
                const maxCount = Math.max(...recentDays.map(([, c]) => c));
                const height = maxCount > 0 ? (count / maxCount) * 100 : 0;
                return (
                  <div key={day} className="flex-1 flex flex-col items-center gap-1">
                    <div
                      className="w-full bg-accentBlue/30 rounded-t-sm transition-all"
                      style={{ height: `${Math.max(height, 4)}%` }}
                      title={`${day}: ${count} events`}
                    />
                    <span className="text-[8px] text-textMuted font-mono">
                      {day.slice(5)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Score by level — IRT-relevant */}
        {Object.keys(summary.scoresByLevel).length > 0 && (
          <div className="mb-8">
            <SectionLabel icon={TrendingUp} label="Average Score by Level (IRT Calibration)" />
            <p className="text-[11px] text-textTertiary mb-3 mt-1">
              If L3 averages higher than L2, the difficulty assignments may need recalibration.
            </p>
            <div className="space-y-2 mt-3">
              {levelOrder.map(level => {
                const data = summary.scoresByLevel[level];
                if (!data || data.count === 0) return null;
                const pct = (data.avg / 3) * 100;
                return (
                  <div key={level} className="flex items-center gap-3">
                    <span className="text-xs font-mono text-textSecondary w-8">{level}</span>
                    <div className="flex-1 h-3 bg-border rounded-full overflow-hidden">
                      <div
                        className={clsx(
                          "h-full rounded-full transition-all",
                          pct >= 70 ? "bg-accentGreen" : pct >= 40 ? "bg-accentAmber" : "bg-accentRed"
                        )}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className="text-xs font-mono text-textTertiary w-20 text-right">
                      {data.avg.toFixed(1)}/3 ({data.count} Qs)
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Score by zone */}
        {Object.keys(summary.scoresByZone).length > 0 && (
          <div className="mb-8">
            <SectionLabel icon={Target} label="Average Score by Zone" />
            <div className="space-y-2 mt-3">
              {Object.entries(summary.scoresByZone)
                .sort(([, a], [, b]) => a.avg - b.avg)
                .map(([zone, data]) => {
                  const pct = (data.avg / 3) * 100;
                  return (
                    <div key={zone} className="flex items-center gap-3">
                      <span className="text-xs text-textSecondary capitalize w-28 truncate">{zone}</span>
                      <div className="flex-1 h-3 bg-border rounded-full overflow-hidden">
                        <div
                          className={clsx(
                            "h-full rounded-full transition-all",
                            pct >= 70 ? "bg-accentGreen" : pct >= 40 ? "bg-accentAmber" : "bg-accentRed"
                          )}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="text-xs font-mono text-textTertiary w-20 text-right">
                        {data.avg.toFixed(1)}/3 ({data.count})
                      </span>
                    </div>
                  );
                })}
            </div>
          </div>
        )}

        {/* Most skipped topics */}
        {summary.topSkippedTopics.length > 0 && (
          <div className="mb-8">
            <SectionLabel icon={Flag} label="Most Skipped Topics (Potential Quality Issues)" />
            <div className="space-y-1.5 mt-3">
              {summary.topSkippedTopics.map(({ topic, count }) => (
                <div key={topic} className="flex items-center justify-between px-3 py-2 rounded-lg border border-borderSubtle bg-surface/50">
                  <span className="text-xs text-textSecondary capitalize">{topic.replace(/-/g, ' ')}</span>
                  <span className="text-[10px] font-mono text-textTertiary">{count} skips</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Empty state */}
        {summary.totalEvents === 0 && (
          <div className="text-center py-16">
            <Activity className="w-12 h-12 text-textTertiary mx-auto mb-4 opacity-30" />
            <h2 className="text-lg font-bold text-textPrimary mb-2">No analytics yet</h2>
            <p className="text-sm text-textTertiary max-w-md mx-auto">
              Start practicing questions and analytics will appear here automatically.
              All data stays in your browser.
            </p>
          </div>
        )}

        {/* Raw event count */}
        <div className="text-[10px] font-mono text-textMuted text-center mt-8">
          {summary.totalEvents} events · {summary.uniqueSessions} sessions · data stored locally
        </div>
      </div>
    </div>
  );
}

// ─── Sub-components ─────────────────────────────

function StatCard({ label, value, icon: Icon, color }: {
  label: string;
  value: string;
  icon: React.ComponentType<{ className?: string }>;
  color?: 'blue' | 'green' | 'red' | 'amber';
}) {
  const colorMap = {
    blue: 'text-accentBlue',
    green: 'text-accentGreen',
    red: 'text-accentRed',
    amber: 'text-accentAmber',
  };
  const valueColor = color ? colorMap[color] : 'text-textPrimary';

  return (
    <div className="p-4 rounded-xl border border-borderSubtle bg-surface/50">
      <div className="flex items-center gap-2 mb-2">
        <Icon className="w-3.5 h-3.5 text-textTertiary" />
        <span className="text-[10px] font-mono text-textTertiary uppercase">{label}</span>
      </div>
      <span className={clsx("text-2xl font-bold font-mono", valueColor)}>{value}</span>
    </div>
  );
}

function SectionLabel({ icon: Icon, label }: { icon: React.ComponentType<{ className?: string }>; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <Icon className="w-4 h-4 text-textTertiary" />
      <span className="text-[10px] font-mono text-textTertiary uppercase tracking-widest">{label}</span>
    </div>
  );
}
