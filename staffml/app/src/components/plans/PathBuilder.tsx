"use client";

import { useState } from "react";
import { Route, Play } from "lucide-react";
import clsx from "clsx";
import { getTracks, getCompetencyAreas } from "@/lib/corpus";
import { countCustomPathQuestions, saveCustomPath, type CustomPath } from "@/lib/plans";

interface PathBuilderProps {
  onStart: (path: CustomPath) => void;
}

const LEVELS = ["L1", "L2", "L3", "L4", "L5"];

export default function PathBuilder({ onStart }: PathBuilderProps) {
  const [track, setTrack] = useState("cloud");
  const [area, setArea] = useState<string | null>(null);
  const [startLevel, setStartLevel] = useState("L1");
  const [targetDate, setTargetDate] = useState("");

  const tracks = getTracks().filter(t => t !== "global");
  const areas = getCompetencyAreas();
  const questionCount = countCustomPathQuestions(track, area, startLevel);

  const handleStart = () => {
    const id = `custom-${area || "all"}-${track}-${startLevel}`.toLowerCase();
    const path: CustomPath = {
      id,
      title: `${area ? area.charAt(0).toUpperCase() + area.slice(1).replace(/-/g, " ") : "All areas"} (${track})`,
      track,
      area,
      startLevel,
      createdAt: Date.now(),
    };
    if (targetDate) {
      path.targetDate = new Date(targetDate + "T23:59:59").getTime();
      const daysRemaining = Math.max(1, Math.ceil((path.targetDate - Date.now()) / 86400000));
      path.dailyQuota = Math.ceil(questionCount / daysRemaining);
    }
    saveCustomPath(path);
    onStart(path);
  };

  return (
    <div className="p-6 rounded-xl border-2 border-dashed border-accentBlue/30 bg-accentBlue/5">
      <div className="flex items-center gap-2 mb-4">
        <Route className="w-5 h-5 text-accentBlue" />
        <h3 className="text-lg font-bold text-textPrimary">Build Your Path</h3>
      </div>
      <p className="text-sm text-textSecondary mb-5">
        Start at L1 and work your way up. Pick a track, optionally focus on one area, and set an interview date.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-5">
        <div>
          <label className="text-[10px] font-mono text-textTertiary uppercase block mb-1.5">Track</label>
          <div className="flex flex-wrap gap-1.5">
            {tracks.map(t => (
              <button
                key={t}
                onClick={() => setTrack(t)}
                className={clsx(
                  "px-3 py-1.5 rounded-md text-xs font-medium border transition-all capitalize",
                  t === track
                    ? "border-accentBlue bg-accentBlue/10 text-accentBlue"
                    : "border-border text-textTertiary hover:border-borderHighlight"
                )}
              >
                {t === "tinyml" ? "TinyML" : t}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="text-[10px] font-mono text-textTertiary uppercase block mb-1.5">Focus area</label>
          <select
            value={area || ""}
            onChange={e => setArea(e.target.value || null)}
            className="w-full px-3 py-1.5 rounded-md text-xs border border-border bg-background text-textPrimary"
          >
            <option value="">All areas</option>
            {areas.map(a => (
              <option key={a} value={a}>{a.charAt(0).toUpperCase() + a.slice(1).replace(/-/g, " ")}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="text-[10px] font-mono text-textTertiary uppercase block mb-1.5">Starting level</label>
          <div className="flex gap-1.5">
            {LEVELS.map(l => (
              <button
                key={l}
                onClick={() => setStartLevel(l)}
                className={clsx(
                  "px-3 py-1.5 rounded-md text-xs font-mono font-bold border transition-all",
                  l === startLevel
                    ? "border-accentBlue bg-accentBlue/10 text-accentBlue"
                    : "border-border text-textTertiary hover:border-borderHighlight"
                )}
              >
                {l}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="text-[10px] font-mono text-textTertiary uppercase block mb-1.5">Interview date (optional)</label>
          <input
            type="date"
            value={targetDate}
            onChange={e => setTargetDate(e.target.value)}
            min={new Date(Date.now() + 86400000).toISOString().split("T")[0]}
            className="w-full px-3 py-1.5 rounded-md text-xs border border-border bg-background text-textPrimary"
          />
        </div>
      </div>

      <div className="flex items-center justify-between">
        <span className="text-xs font-mono text-textTertiary">
          {questionCount.toLocaleString()} questions
          {targetDate && questionCount > 0 && (() => {
            const days = Math.max(1, Math.ceil((new Date(targetDate + "T23:59:59").getTime() - Date.now()) / 86400000));
            return ` · ${Math.ceil(questionCount / days)}/day`;
          })()}
        </span>
        <button
          onClick={handleStart}
          disabled={questionCount === 0}
          className={clsx(
            "px-5 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center gap-2",
            questionCount > 0
              ? "bg-accentBlue text-white hover:opacity-90"
              : "bg-border text-textTertiary cursor-not-allowed"
          )}
        >
          <Play className="w-4 h-4" /> Start Path
        </button>
      </div>
    </div>
  );
}
